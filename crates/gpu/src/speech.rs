//! Whisper on the GPU: the same encoder-decoder as `kvad::speech::whisper`,
//! in candle.
//!
//! Read it beside the CPU one. The blocks are line for line the same, and
//! the decoding loop is not here at all: `kvad::speech::Transcriber` drives
//! this through [`kvad::speech::Model`] as it drives the CPU's, so the
//! spectrogram, the prompt, the rules about timestamps and the walk through
//! a long recording are one piece of code that both run.
//!
//! What changes on a GPU is where the time goes. On the CPU, large's encoder
//! is most of a window: 32 blocks over 1500 frames, over two teraflops, at
//! the f32 matmul's 200-odd GFLOP/s. Here it is one wide batch, which is
//! what a GPU is for, 0.3 s a window, and the decoder's one-row steps become
//! the larger share, as decode does in a language model.
//!
//! - **The encoder** is a prefill with no mask. Its attention is the fused
//!   kernel at 1500 queries against 1500 keys, and its projections are
//!   1500 rows wide, so at q8 they go to the M5's matrix units
//!   ([`Loader::accelerated`]), as an image model's do. That is allowed here
//!   and not in a language model because nothing in the encoder is ever one
//!   row: `Proj::Blocks` has no one-row kernel.
//! - **The cross-attention's keys and values** are made from the encoder's
//!   output once a window, also 1500 rows, also on the matrix units.
//! - **The decoder** is everything else: one token a step, its own key and
//!   value cache (`common::KvCache`), and the cross-attention reading the
//!   window's.
//!   Its projections are loaded the ordinary way, with the one-row kernels a
//!   decode step wants.
//!
//! # Precision
//!
//! The residual stream, the norms and the position tables are f32 whatever
//! the matrices are. At q8 and q4 the matrices are blocks and the arithmetic
//! between them f32, as in every quantised model on this backend: candle's
//! quantised kernels take nothing narrower. Dense, they are [`DENSE`], f16,
//! which a server's `gpu-bf16` loads, for the reason [`DENSE`] gives. How
//! far each is from transformers' f32, stage by stage, is in
//! `docs/speech-plan.md`, from `tests::whisper_agrees_with_transformers`.

use crate::common::{Embed, Loader, Proj, Reader, Stored};
use crate::image::nn::{wide, Ctx, Linear};
use crate::qcache::Vault;
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Tensor};
use kvad::speech::whisper::Config;
use kvad::speech::{Heard, Said, Transcriber};
use kvad::weights::Watcher;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const EPS: f32 = 1e-5;

/// A Whisper from `repo` on this machine's GPU: its matrices at `quant`, or
/// dense in `dtype` without one.
pub fn load(repo: &str, dtype: DType, quant: Option<GgmlDType>, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Transcriber> {
    let device = crate::model::pick_device(None)?;
    if !device.is_metal() && !device.is_cuda() {
        return Err("this machine has no GPU that candle can use; load the Whisper at a cpu backend".into());
    }
    Transcriber::load_with(repo, progress, watch, &mut |config, files, progress| {
        Ok(Box::new(GpuWhisper::load(repo, config, &files.weights, device.clone(), dtype, quant, progress)?))
    })
}

/// The dtype a Whisper's matrices are dense in on the GPU when they are not
/// quantised: what a server's `gpu-bf16` loads one at.
///
/// f16, not bf16. Whisper was trained in f16 and its numbers stay well
/// inside f16's range, so the three bits of mantissa bf16 gives up for range
/// buy nothing here, and cost four to thirteen times the rounding: against
/// transformers' f32, bf16's encoder output on nb-whisper-large is 6.3 off
/// at its worst and f16's 0.50, and bf16 changes a word of the first window
/// of `king.mp3` where f16 changes nothing. Both are two bytes and run on
/// the same matrix units at the same speed.
pub const DENSE: DType = DType::F16;

/// What [`load`] will hold in weights, from the config alone: the matrices
/// at `quant` (dense in `dtype` without one), the filters dense in the
/// dtype it computes in, and the two position tables in f32. Norms and
/// biases are left out, as `Config::params` leaves them, and as
/// [`kvad::speech::Model::weight_bytes`] counts them once loaded.
pub fn weight_bytes(config: &Config, dtype: DType, quant: Option<GgmlDType>) -> u64 {
    let c = config;
    let (matrices, _) = c.params();
    let filters = (3 * c.d * (c.n_mels + c.d)) as u64;
    let positions = ((c.audio_ctx + c.text_ctx) * c.d) as u64;
    let dense = compute_dtype(dtype, quant).size_in_bytes() as u64;
    let matrices = match quant {
        // Every matrix's rows are its input width, a multiple of the block.
        Some(gd) => (matrices / gd.block_size() * gd.type_size()) as u64,
        None => matrices as u64 * dense,
    };
    matrices + filters * dense + positions * 4
}

/// The dtype the arithmetic is in: f32 around quantised matrices, which is
/// all candle's quantised kernels take.
fn compute_dtype(dtype: DType, quant: Option<GgmlDType>) -> DType {
    match quant {
        Some(_) => DType::F32,
        None => dtype,
    }
}

/// One attention's four projections. The key has no bias: a softmax does
/// not change when a constant is added to every score, and a key's bias
/// adds exactly that.
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Attn {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, d: usize) -> Res<Attn> {
        Ok(Attn {
            q: Linear::load(cx, r, "q_proj", d, d, true)?,
            k: Linear::load(cx, r, "k_proj", d, d, false)?,
            v: Linear::load(cx, r, "v_proj", d, d, true)?,
            o: Linear::load(cx, r, "out_proj", d, d, true)?,
        })
    }

    fn matrices(&self) -> [&Linear; 4] {
        [&self.q, &self.k, &self.v, &self.o]
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, d: usize, ffn: usize) -> Res<Mlp> {
        Ok(Mlp { fc1: Linear::load(cx, r, "fc1", d, ffn, true)?, fc2: Linear::load(cx, r, "fc2", ffn, d, true)? })
    }

    /// `x` in the matrices' dtype, `dt`, and the answer in f32, for the
    /// residual stream.
    fn forward(&self, x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
        // The exact GELU, through `erf`; GPT-2's tanh one differs by 5e-4.
        let h = crate::grad::gelu_erf(&self.fc1.forward_in(x, dt)?)?;
        self.fc2.forward_in(&h, DType::F32)
    }
}

/// A LayerNorm, in f32 whatever the matrices are in: it reads the residual
/// stream, which is f32 (see [`GpuWhisper`]), and answers in the matrices'
/// dtype for the projection after it.
struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    /// From `r`, a reader in f32.
    fn load(device: &Device, r: &Reader<'_>, name: &str, d: usize) -> Res<Norm> {
        let r = r.pp(name);
        Ok(Norm { w: r.get(d, "weight")?.to_device(device)?, b: r.get(d, "bias")?.to_device(device)? })
    }

    fn forward(&self, x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
        crate::grad::layer_norm(&x.contiguous()?, &self.w, &self.b, EPS)?.to_dtype(dt)
    }
}

struct EncoderBlock {
    ln1: Norm,
    attn: Attn,
    ln2: Norm,
    mlp: Mlp,
}

/// A decoder block. Its cross-attention's key and value projections read
/// the encoder's 1500 frames, once a window, and are loaded as the encoder
/// is; the rest read one token a step.
struct DecoderBlock {
    ln1: Norm,
    attn: Attn,
    ln_cross: Norm,
    cross: Attn,
    ln2: Norm,
    mlp: Mlp,
}

/// A convolution over time: `[out, in, 3]` filters and their bias.
struct Conv {
    w: Tensor,
    b: Tensor,
    stride: usize,
}

impl Conv {
    /// `x` is `[1, in, len]`; the answer `[1, out, len / stride]`, through
    /// the GELU that follows both of Whisper's.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.conv1d(&self.w, 1, self.stride, 1, 1)?;
        crate::grad::gelu_erf(&y.broadcast_add(&self.b.reshape((1, (), 1))?)?)
    }
}

/// The cross-attention's keys and values for one window, block by block,
/// `[1, heads, 1500, head_dim]` each. Made once by [`GpuWhisper::listen`]
/// and only read after.
pub struct Cross {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

/// Whisper on the GPU. See the module comment.
pub struct GpuWhisper {
    config: Config,
    device: Device,
    /// What the arithmetic is in ([`compute_dtype`]).
    dtype: DType,
    label: String,
    /// The dtype both attention caches are kept in, when it is not
    /// `dtype`: f16 around quantised matrices (`common::kv_store`). The
    /// cross-attention's keys and values are read whole at every step,
    /// 1500 frames by every block: at large in f32 that is 490 MB a token
    /// beside the 850 MB of q8 weights.
    store: Option<DType>,
    conv1: Conv,
    conv2: Conv,
    /// `[1500, d]`: sinusoids, stored as a weight.
    audio_pos: Tensor,
    encoder: Vec<EncoderBlock>,
    ln_audio: Norm,
    /// `[vocab, d]`: the token embedding, and the output head.
    tokens: Embed,
    /// The head, when it is the table's blocks read as a matrix; a dense
    /// table is read transposed instead, and needs no second handle.
    head: Option<Proj>,
    /// `[448, d]`: learned.
    text_pos: Tensor,
    decoder: Vec<DecoderBlock>,
    ln_text: Norm,
}

impl GpuWhisper {
    pub fn load(
        repo: &str,
        config: Config,
        paths: &[std::path::PathBuf],
        device: Device,
        dtype: DType,
        quant: Option<GgmlDType>,
        progress: &mut dyn FnMut(&str),
    ) -> Res<GpuWhisper> {
        let c = &config;
        crate::common::check_block(quant, &[("d_model", c.d), ("encoder_ffn_dim", c.encoder_ffn), ("decoder_ffn_dim", c.decoder_ffn)])?;
        let dt = compute_dtype(dtype, quant);
        let label = crate::common::label(&device, dt, quant);
        progress(&format!("onto the GPU: {label}"));
        let shape = kvad::serde_json::json!({ "component": "whisper", "d": c.d, "encoder_layers": c.encoder_layers, "decoder_layers": c.decoder_layers, "vocab": c.vocab });
        let mut vault = Vault::open_as(repo, paths, shape, quant, progress);
        let model = {
            // Two loaders over one cache: the wide one for everything that
            // only ever sees 1500 rows, the plain one for a decode step's.
            let wide_cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype: dt };
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault), dtype: dt };
            // The norms and the position tables are f32 here whatever the
            // matrices are, and are read as f32, not rounded to the
            // matrices' dtype on the way.
            let (r, exact) = (crate::image::open(paths, dt)?, crate::image::open(paths, DType::F32)?);
            let model = GpuWhisper::read(&wide_cx, &cx, &r, &exact, config, label, quant)?;
            let mut seen = r.seen();
            seen.extend(exact.seen());
            let left = crate::common::unread(paths, &seen, &r.skipped())?;
            if !left.is_empty() {
                return Err(crate::common::unread_error("whisper", &left).into());
            }
            model
        };
        vault.finish(progress);
        crate::common::settle(&device)?;
        Ok(model)
    }

    fn read(wide_cx: &Ctx<'_>, cx: &Ctx<'_>, r: &Reader<'_>, exact: &Reader<'_>, config: Config, label: String, quant: Option<GgmlDType>) -> Res<GpuWhisper> {
        let c = &config;
        let d = c.d;
        let enc = r.pp("model.encoder");
        let dec = r.pp("model.decoder");
        let conv = |name: &str, inp: usize, stride: usize| -> Res<Conv> {
            let p = enc.pp(name);
            Ok(Conv { w: cx.get(&p, (d, inp, 3), "weight")?, b: cx.get(&p, d, "bias")?, stride })
        };
        // The same places in the f32 reader.
        let (enc32, dec32) = (exact.pp("model.encoder"), exact.pp("model.decoder"));
        let norm = |r: &Reader<'_>, name: &str| Norm::load(cx.device(), r, name, d);
        let encoder = (0..c.encoder_layers)
            .map(|i| {
                let (l, l32) = (enc.pp(format!("layers.{i}")), enc32.pp(format!("layers.{i}")));
                Ok(EncoderBlock {
                    ln1: norm(&l32, "self_attn_layer_norm")?,
                    attn: Attn::load(wide_cx, &l.pp("self_attn"), d)?,
                    ln2: norm(&l32, "final_layer_norm")?,
                    mlp: Mlp::load(wide_cx, &l, d, c.encoder_ffn)?,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let decoder = (0..c.decoder_layers)
            .map(|i| {
                let (l, l32) = (dec.pp(format!("layers.{i}")), dec32.pp(format!("layers.{i}")));
                let x = l.pp("encoder_attn");
                Ok(DecoderBlock {
                    ln1: norm(&l32, "self_attn_layer_norm")?,
                    attn: Attn::load(cx, &l.pp("self_attn"), d)?,
                    ln_cross: norm(&l32, "encoder_attn_layer_norm")?,
                    cross: Attn {
                        q: Linear::load(cx, &x, "q_proj", d, d, true)?,
                        k: Linear::load(wide_cx, &x, "k_proj", d, d, false)?,
                        v: Linear::load(wide_cx, &x, "v_proj", d, d, true)?,
                        o: Linear::load(cx, &x, "out_proj", d, d, true)?,
                    },
                    ln2: norm(&l32, "final_layer_norm")?,
                    mlp: Mlp::load(cx, &l, d, c.decoder_ffn)?,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let name = "embed_tokens.weight";
        let (tokens, head) = match quant {
            Some(_) => {
                // One set of blocks for both: a row at a time for the
                // lookup (GGML's `get_rows`), and as a matrix for the head.
                let q = Arc::new(cx.ld.quantized(&dec, name, c.vocab, d, Stored::OutIn)?);
                (Embed::Quant(q.clone()), Some(Proj::Quant(QMatMul::from_arc(q)?)))
            }
            None => (Embed::Dense(cx.get(&dec, (c.vocab, d), name)?), None),
        };
        // Tied in every Whisper released; a checkpoint that writes the head
        // out anyway writes the same numbers.
        r.record("proj_out.weight");
        Ok(GpuWhisper {
            conv1: conv("conv1", c.n_mels, 1)?,
            conv2: conv("conv2", d, 2)?,
            audio_pos: enc32.get((c.audio_ctx, d), "embed_positions.weight")?.to_device(cx.device())?,
            encoder,
            ln_audio: norm(&enc32, "layer_norm")?,
            tokens,
            head,
            text_pos: dec32.get((c.text_ctx, d), "embed_positions.weight")?.to_device(cx.device())?,
            decoder,
            ln_text: norm(&dec32, "layer_norm")?,
            store: crate::common::kv_store(cx.device(), quant),
            device: cx.device().clone(),
            dtype: cx.dtype,
            label,
            config,
        })
    }

    fn matrices(&self) -> impl Iterator<Item = &Linear> {
        let enc = self.encoder.iter().flat_map(|b| b.attn.matrices().into_iter().chain([&b.mlp.fc1, &b.mlp.fc2]));
        let dec = self.decoder.iter().flat_map(|b| b.attn.matrices().into_iter().chain(b.cross.matrices()).chain([&b.mlp.fc1, &b.mlp.fc2]));
        enc.chain(dec)
    }

    /// The tensors kept dense whatever the quantisation: the filters and
    /// the two position tables.
    fn floats(&self) -> [&Tensor; 4] {
        [&self.conv1.w, &self.conv2.w, &self.audio_pos, &self.text_pos]
    }

    /// The encoder: one window of spectrogram, `[3000][n_mels]` time-major,
    /// to `[1, 1500, d]`.
    pub fn encode(&self, mel: &[f32]) -> Res<Tensor> {
        let c = &self.config;
        let frames = mel.len() / c.n_mels;
        // Time-major on the host, channels-first for a convolution.
        let x = Tensor::from_slice(mel, (frames, c.n_mels), &self.device)?.t()?.unsqueeze(0)?.to_dtype(self.dtype)?.contiguous()?;
        let x = self.conv2.forward(&self.conv1.forward(&x)?)?;
        if x.dim(2)? != c.audio_ctx {
            return Err(format!("a window is {} frames of spectrogram, not {frames}", 2 * c.audio_ctx).into());
        }
        let mut x = x.transpose(1, 2)?.to_dtype(DType::F32)?.broadcast_add(&self.audio_pos)?.contiguous()?;
        let (heads, dt) = (c.encoder_heads, self.dtype);
        for b in &self.encoder {
            let h = b.ln1.forward(&x, dt)?;
            let (q, k, v) = (self.lin(&b.attn.q, &h)?, self.lin(&b.attn.k, &h)?, self.lin(&b.attn.v, &h)?);
            let a = crate::image::nn::attention(&q, &k, &v, heads)?;
            x = (x + b.attn.o.forward_in(&a, DType::F32)?)?;
            x = (&x + b.mlp.forward(&b.ln2.forward(&x, dt)?, dt)?)?;
        }
        Ok(self.ln_audio.forward(&x, dt)?)
    }

    /// The cross-attention keys and values every decoder block will read for
    /// this window: [`GpuWhisper::encode`]'s output, projected once per
    /// block.
    pub fn listen(&self, audio: &Tensor) -> Res<Cross> {
        let mut cross = Cross { k: Vec::with_capacity(self.decoder.len()), v: Vec::with_capacity(self.decoder.len()) };
        let heads = self.config.decoder_heads;
        let kept = |t: Tensor| match self.store {
            Some(dt) => t.to_dtype(dt),
            None => Ok(t),
        };
        for b in &self.decoder {
            cross.k.push(kept(split(&self.lin(&b.cross.k, audio)?, heads)?)?);
            cross.v.push(kept(split(&self.lin(&b.cross.v, audio)?, heads)?)?);
        }
        Ok(cross)
    }

    /// `tokens` through the decoder after what `cache` holds, and the
    /// next-token scores after each of `rows`, on the host.
    pub(crate) fn step(&self, tokens: &[u32], cache: &mut [crate::common::KvCache], cross: &Cross, rows: &[usize]) -> Res<Vec<Vec<f32>>> {
        let c = &self.config;
        let (m, pos0) = (tokens.len(), cache[0].len());
        if pos0 + m > c.text_ctx {
            return Err(format!("{} text positions is past the decoder's {}", pos0 + m, c.text_ctx).into());
        }
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= c.vocab) {
            return Err(format!("token {t} is outside the vocabulary of {}", c.vocab).into());
        }
        let ids = Tensor::new(tokens, &self.device)?;
        let x = self.tokens.rows(&ids)?.to_dtype(DType::F32)?;
        let mut x = (x + self.text_pos.narrow(0, pos0, m)?)?.unsqueeze(0)?;
        let dt = self.dtype;

        let heads = c.decoder_heads;
        let scale = 1.0 / ((c.d / heads) as f64).sqrt();
        let mask = match m {
            1 => None,
            _ => Some(crate::common::causal_mask(m, pos0, &self.device, DType::F32)?),
        };
        for (l, (b, kv)) in self.decoder.iter().zip(cache.iter_mut()).enumerate() {
            let h = b.ln1.forward(&x, dt)?;
            let q = split(&self.lin(&b.attn.q, &h)?, heads)?;
            let (k, v) = kv.push(&split(&self.lin(&b.attn.k, &h)?, heads)?, &split(&self.lin(&b.attn.v, &h)?, heads)?)?;
            let a = attend(&q, &k, &v, mask.as_ref(), scale)?;
            x = (x + b.attn.o.forward_in(&join(&a)?, DType::F32)?)?;

            let h = b.ln_cross.forward(&x, dt)?;
            let q = split(&self.lin(&b.cross.q, &h)?, heads)?;
            let a = attend(&q, &cross.k[l], &cross.v[l], None, scale)?;
            x = (x + b.cross.o.forward_in(&join(&a)?, DType::F32)?)?;

            x = (&x + b.mlp.forward(&b.ln2.forward(&x, dt)?, dt)?)?;
        }
        let x = self.ln_text.forward(&x, dt)?.squeeze(0)?;
        let at = Tensor::new(rows.iter().map(|&r| r as u32).collect::<Vec<_>>(), &self.device)?;
        let h = x.index_select(&at, 0)?;
        let logits = match (&self.head, &self.tokens) {
            (Some(head), _) => head.forward(&h)?,
            (None, Embed::Dense(table)) => h.matmul(&table.t()?)?,
            (None, Embed::Quant(_)) => unreachable!("a quantised table has its head"),
        };
        let out = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        crate::common::report_failures(&self.device)?;
        Ok(out)
    }

    /// A projection, answered in the compute dtype: a quantised matrix
    /// answers in f32 whatever it was asked in.
    fn lin(&self, l: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
        l.forward_in(x, self.dtype)
    }
}

/// `[1, n, heads·hd]` to `[1, heads, n, hd]`.
fn split(x: &Tensor, heads: usize) -> candle_core::Result<Tensor> {
    let (b, n, w) = x.dims3()?;
    x.reshape((b, n, heads, w / heads))?.transpose(1, 2)?.contiguous()
}

/// `[1, heads, n, hd]` back to `[1, n, heads·hd]`.
fn join(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, h, n, hd) = x.dims4()?;
    x.transpose(1, 2)?.contiguous()?.reshape((b, n, h * hd))
}

/// Attention for the decoder, both kinds: a few text queries against the
/// text so far, under `mask` when there is more than one, or against all of
/// a window's frames, with none.
///
/// The fused kernel takes every case but one: it is right without a mask
/// at any width, and at one query, which has no future to hide; its own
/// causal masking is wrong unless the queries are a multiple of its tile
/// (see `model::fused`), and a prompt is a few tokens. So a masked prompt
/// is written out, its scores and softmax in f32: a row of scores in two
/// bytes is three significant figures, and the prompt is where the first
/// token is chosen.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, scale: f64) -> candle_core::Result<Tensor> {
    let (hd, dt) = (q.dim(3)?, q.dtype());
    let fused = q.device().is_metal() && matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256);
    if fused && mask.is_none() {
        // A cache kept narrower than the arithmetic: the kernel wants one
        // dtype, and sums in f32 whatever it is, so the query goes down.
        return candle_nn::ops::sdpa(&q.to_dtype(k.dtype())?, k, v, None, false, scale as f32, 1.0)?.to_dtype(dt);
    }
    let f = |t: &Tensor| t.to_dtype(wide(dt));
    let att = (f(q)?.matmul(&f(&k.transpose(2, 3)?.contiguous()?)?)? * scale)?;
    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    crate::grad::softmax_last_dim(&att)?.matmul(&f(&v.contiguous()?)?)?.to_dtype(dt)
}

impl kvad::speech::Model for GpuWhisper {
    fn config(&self) -> &Config {
        &self.config
    }

    fn hear(&self, window: &[f32]) -> Res<Heard> {
        // A pool a window, and the device to this thread while it runs:
        // see `common::pooled`.
        crate::common::pooled(|| {
            let cross = self.listen(&self.encode(window)?)?;
            crate::common::settle(&self.device)?;
            Ok(Heard(Box::new(cross)))
        })
    }

    fn said(&self) -> Said {
        Said(Box::new((0..self.decoder.len()).map(|_| crate::common::KvCache::new(2).stored_as(self.store)).collect::<Vec<_>>()))
    }

    fn decode(&self, tokens: &[u32], said: &mut Said, heard: &Heard, rows: &[usize]) -> Res<Vec<Vec<f32>>> {
        let cache: &mut Vec<crate::common::KvCache> = said.get_mut()?;
        let cross: &Cross = heard.get()?;
        // A pool a step, as a language model's decode has one a token.
        crate::common::pooled(|| self.step(tokens, cache, cross, rows))
    }

    fn within(&self, work: &mut (dyn FnMut() + Send)) {
        work()
    }

    fn params(&self) -> usize {
        self.matrices().map(Linear::params).sum::<usize>() + self.tokens.params() + self.floats().iter().map(|t| t.elem_count()).sum::<usize>()
    }

    fn weight_bytes(&self) -> usize {
        let floats: usize = self.floats().iter().map(|t| t.elem_count() * t.dtype().size_in_bytes()).sum();
        self.matrices().map(Linear::bytes).sum::<usize>() + self.tokens.bytes() + floats
    }

    fn backend(&self) -> String {
        self.label.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::speech::reference::{compare, Fixtures, Tolerance};

    /// What a server is told a Whisper will take, before it loads, is what
    /// it takes once loaded, at both of the server's GPU precisions. On
    /// nb-whisper-tiny, read from the cache only, and passed over where it
    /// is not there or there is no Metal.
    #[test]
    fn the_estimate_is_what_loads() {
        let repo = "NbAiLab/nb-whisper-tiny";
        let cached = kvad::weights::local_file(repo, "model.safetensors").is_some();
        let Some(config) = kvad::speech::local_config(repo).filter(|_| cached) else {
            eprintln!("{repo} is not in the cache");
            return;
        };
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("no Metal device");
            return;
        };
        let Ok(files) = kvad::speech::Files::fetch(repo, &Watcher::none()) else { return };
        for quant in [None, Some(GgmlDType::Q8_0)] {
            let model = GpuWhisper::load(repo, config.clone(), &files.weights, device.clone(), DENSE, quant, &mut |_| {}).unwrap();
            let loaded = kvad::speech::Model::weight_bytes(&model) as u64;
            assert_eq!(weight_bytes(&config, DENSE, quant), loaded, "at {quant:?}");
        }
    }

    /// Against `scripts/whisper-fixtures.py`, as `kvad::speech`'s test is,
    /// at each precision this backend loads at: transformers' f32 numbers
    /// stage by stage, and its greedy tokens, which must be the same ones.
    ///
    /// ```text
    /// KVAD_WHISPER_FIXTURES=/tmp/whisper-fx cargo test --release -p kvad-gpu speech::tests -- --ignored --nocapture
    /// ```
    ///
    /// `KVAD_WHISPER_GPU=f32,bf16,f16,q8,q4` picks the precisions; bf16 and
    /// q8, the server's two, by default. The stages may differ by what two
    /// bytes a number allow, and are reported; the tokens may not.
    #[test]
    #[ignore]
    fn whisper_agrees_with_transformers() {
        let dir = std::path::PathBuf::from(std::env::var("KVAD_WHISPER_FIXTURES").expect("KVAD_WHISPER_FIXTURES is not set"));
        let fx = Fixtures::open(&dir).unwrap();
        let repo = fx.repo().unwrap();
        let wanted = std::env::var("KVAD_WHISPER_GPU").unwrap_or_else(|_| "bf16,q8".into());
        let mut failed = Vec::new();
        for name in wanted.split(',') {
            let (dtype, quant) = match crate::model::parse_dtype(name) {
                Some(dt) => (dt, None),
                None => (DType::F32, crate::model::parse_quant(name).flatten()),
            };
            eprintln!("--- {repo} at {name}");
            let device = crate::model::pick_device(None).unwrap();
            let mut made = None;
            let t = Transcriber::load_with(&repo, &mut |m| eprintln!("{m}"), &Watcher::none(), &mut |config, files, progress| {
                let model = Arc::new(GpuWhisper::load(&repo, config, &files.weights, device.clone(), dtype, quant, progress)?);
                made = Some(model.clone());
                Ok(Box::new(model))
            })
            .unwrap();
            let model = made.unwrap();
            let encode = |window: &[f32]| -> Res<Vec<f32>> {
                crate::common::pooled(|| Ok(model.encode(window)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?))
            };
            let tol = match (dtype, quant) {
                (DType::F32, None) => Tolerance { encoder: 2e-3, logits: 2e-3 },
                // Not the measure of right: the tokens are. A bound that
                // catches a stage gone to noise.
                _ => Tolerance { encoder: 0.75, logits: 0.1 },
            };
            if let Err(e) = compare(&fx, &t, &encode, tol) {
                eprintln!("{name}: {e}");
                failed.push(format!("{name}: {e}"));
            }
            drop((t, model));
            crate::common::settle(&device).unwrap();
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
