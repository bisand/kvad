//! Whisper (2022): the encoder-decoder transformer.
//!
//! Every other model in this crate is one stack of blocks reading its own
//! output. Whisper is two:
//!
//! ```text
//!   mel [3000 × 80] ─ conv, conv ─▶ + positions ─▶ N blocks ─▶ audio [1500 × d]
//!                                                                  │
//!   tokens ─▶ embed + positions ─▶ N blocks ──────────────────────▶ logits
//!                                  each: self-attention (causal)    ▲
//!                                        cross-attention ───────────┘
//!                                        MLP
//! ```
//!
//! The **encoder** reads a 30-second window once. Its attention has no mask:
//! every frame sees every other, before and after, because the whole window
//! is there at once and nothing is being predicted. Its two convolutions at
//! the bottom turn 3000 spectrogram columns into 1500 frames (the second has
//! stride 2), so each frame the blocks see is 20 ms of sound.
//!
//! The **decoder** is a GPT-2 with one more sublayer. Between its causal
//! self-attention and its MLP sits a **cross-attention**: the queries come
//! from the text, and the keys and values from the encoder's output. That is
//! the whole of how the text gets to listen.
//!
//! # Two caches, with two lifetimes
//!
//! The decoder's self-attention keeps keys and values per text position and
//! grows by one row each token — [`TextCache`], the same idea as
//! [`crate::model::KvCache`]. The cross-attention's keys and values are a
//! function of the audio alone: computed once per window from the encoder's
//! output and read, unchanged, at every step of decoding that window —
//! [`AudioCache`]. They share a name in most write-ups and nothing else, so
//! here they are two types.
//!
//! Block by block it is GPT-2's: LayerNorm before each sublayer, biases on
//! everything except the key projections (a softmax does not change when a
//! constant is added to every score, and a key bias adds exactly that), GELU
//! in the MLP — the exact one, through `erf`, not GPT-2's tanh — and an output
//! head that is the token embedding read the other way.

use crate::qcache::Source;
use crate::quant::{Precision, Weight};
use crate::tensor::{dot, gelu_erf_inplace, gemm_bt, layer_norm, softmax_inplace, Tensor};
use rayon::prelude::*;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const EPS: f32 = 1e-5;

/// The shape of a Whisper, as `config.json` gives it.
///
/// Not [`crate::model::Spec`]. Two stacks that may differ in depth (the
/// distilled Whispers keep 32 encoder blocks and 2 decoder ones), two
/// context lengths that mean different things — 1500 audio frames, 448 text
/// tokens — and the number of mel bands, which has no counterpart in a text
/// model at all.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Bands in the spectrogram: 80, or 128 from large-v3 on.
    pub n_mels: usize,
    /// Width of both stacks.
    pub d: usize,
    pub encoder_layers: usize,
    pub encoder_heads: usize,
    pub encoder_ffn: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_ffn: usize,
    /// Frames the encoder makes of one window: 1500.
    pub audio_ctx: usize,
    /// Text positions the decoder has embeddings for: 448.
    pub text_ctx: usize,
    pub vocab: usize,
}

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Res<Config> {
        let num = |k: &str| -> Res<usize> {
            v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("config.json has no `{k}`").into())
        };
        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if model_type != "whisper" {
            return Err(format!("model_type is `{model_type}`, not `whisper`").into());
        }
        let act = v.get("activation_function").and_then(|x| x.as_str()).unwrap_or("gelu");
        if act != "gelu" {
            return Err(format!("activation_function `{act}`: only `gelu` is implemented").into());
        }
        if v.get("scale_embedding").and_then(|x| x.as_bool()) == Some(true) {
            return Err("scale_embedding is true, which no released Whisper uses and this does not implement".into());
        }
        Ok(Config {
            n_mels: num("num_mel_bins")?,
            d: num("d_model")?,
            encoder_layers: num("encoder_layers")?,
            encoder_heads: num("encoder_attention_heads")?,
            encoder_ffn: num("encoder_ffn_dim")?,
            decoder_layers: num("decoder_layers")?,
            decoder_heads: num("decoder_attention_heads")?,
            decoder_ffn: num("decoder_ffn_dim")?,
            audio_ctx: num("max_source_positions")?,
            text_ctx: num("max_target_positions")?,
            vocab: num("vocab_size")?,
        })
    }

    /// The weights [`Whisper::weight_bytes`] will count, before there are
    /// any: those in matrices loaded at the precision asked for, and those
    /// kept as f32 whatever it is — the convolutions and both position
    /// tables. LayerNorms and biases are left out there and here.
    pub fn params(&self) -> (usize, usize) {
        let d = self.d;
        let encoder = self.encoder_layers * (4 * d * d + 2 * d * self.encoder_ffn);
        let decoder = self.decoder_layers * (8 * d * d + 2 * d * self.decoder_ffn);
        let floats = 3 * d * self.n_mels + 3 * d * d + (self.audio_ctx + self.text_ctx) * d;
        (encoder + decoder + self.vocab * d, floats)
    }

    /// Whether this checkpoint is a Whisper, by its `config.json` alone.
    pub fn is_whisper(config: &Path) -> bool {
        crate::weights::read_json(config).ok().and_then(|v| v.get("model_type")?.as_str().map(|t| t == "whisper")).unwrap_or(false)
    }
}

/// A LayerNorm's gain and bias.
struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(src: &dyn Source, p: &str) -> Res<Norm> {
        Ok(Norm { g: src.vector(&format!("{p}.weight"))?, b: src.vector(&format!("{p}.bias"))? })
    }

    /// Every row of `x`, normalised by itself.
    fn rows(&self, x: &[f32]) -> Vec<f32> {
        let d = self.g.len();
        let mut out = vec![0.0; x.len()];
        out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| o.copy_from_slice(&layer_norm(r, &self.g, &self.b, EPS)));
        out
    }
}

/// One attention's four projections. The key has no bias: see the module
/// comment.
struct Attn {
    q: Weight,
    q_b: Vec<f32>,
    k: Weight,
    v: Weight,
    v_b: Vec<f32>,
    o: Weight,
    o_b: Vec<f32>,
}

impl Attn {
    fn load(src: &dyn Source, p: &str) -> Res<Attn> {
        let w = |s: &str| src.matrix(&format!("{p}.{s}.weight"));
        let b = |s: &str| src.vector(&format!("{p}.{s}.bias"));
        Ok(Attn { q: w("q_proj")?, q_b: b("q_proj")?, k: w("k_proj")?, v: w("v_proj")?, v_b: b("v_proj")?, o: w("out_proj")?, o_b: b("out_proj")? })
    }

    /// The queries for `m` rows of `x`, already divided by `√head_dim` so the
    /// scores need no scaling after. Scaling `q` once is `d` multiplies a
    /// row; scaling the scores is one per key.
    fn queries(&self, x: &[f32], m: usize, hd: usize) -> Vec<f32> {
        let mut q = self.q.matmul_bt(x, m, Some(&self.q_b));
        let scale = 1.0 / (hd as f32).sqrt();
        q.iter_mut().for_each(|v| *v *= scale);
        q
    }
}

struct Mlp {
    fc1: Weight,
    fc1_b: Vec<f32>,
    fc2: Weight,
    fc2_b: Vec<f32>,
}

impl Mlp {
    fn load(src: &dyn Source, p: &str) -> Res<Mlp> {
        Ok(Mlp {
            fc1: src.matrix(&format!("{p}.fc1.weight"))?,
            fc1_b: src.vector(&format!("{p}.fc1.bias"))?,
            fc2: src.matrix(&format!("{p}.fc2.weight"))?,
            fc2_b: src.vector(&format!("{p}.fc2.bias"))?,
        })
    }

    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut h = self.fc1.matmul_bt(x, m, Some(&self.fc1_b));
        gelu_erf_inplace(&mut h);
        self.fc2.matmul_bt(&h, m, Some(&self.fc2_b))
    }
}

struct EncoderBlock {
    ln1: Norm,
    attn: Attn,
    ln2: Norm,
    mlp: Mlp,
}

struct DecoderBlock {
    ln1: Norm,
    attn: Attn,
    ln_cross: Norm,
    cross: Attn,
    ln2: Norm,
    mlp: Mlp,
}

/// A convolution over time, as a matmul: the filters are `[out, in × k]`.
struct Conv {
    w: Weight,
    b: Vec<f32>,
    stride: usize,
}

impl Conv {
    /// Slide over `x`, `[len][in]`, with one frame of zeros at each end.
    ///
    /// Each output frame is one dot product per filter against the `k`
    /// input frames under it. Lay those `k` frames out as one row —
    /// channel-major, `in × k`, the order PyTorch stores the filter in —
    /// and every output frame at once is a matrix product of those rows
    /// against the filters. That is `im2col`, in one dimension: it costs a
    /// copy of the input three times over, and turns a convolution into the
    /// one kernel this crate has made fast.
    fn forward(&self, x: &[f32], len: usize, inp: usize) -> (Vec<f32>, usize) {
        const K: usize = 3;
        let out_len = (len + 2 - K) / self.stride + 1;
        let mut cols = vec![0.0f32; out_len * inp * K];
        cols.par_chunks_mut(inp * K).enumerate().for_each(|(t, row)| {
            for k in 0..K {
                // Input frame under tap `k` of output `t`, before the padding.
                let at = (t * self.stride + k) as isize - 1;
                if at < 0 || at as usize >= len {
                    continue;
                }
                let frame = &x[at as usize * inp..(at as usize + 1) * inp];
                for (i, &v) in frame.iter().enumerate() {
                    row[i * K + k] = v;
                }
            }
        });
        let mut y = self.w.matmul_bt(&cols, out_len, Some(&self.b));
        gelu_erf_inplace(&mut y);
        (y, out_len)
    }
}

/// The cross-attention's keys and values for one window, layer by layer.
///
/// Made once by [`Whisper::listen`] and only read after: see the module
/// comment for why this is not the other cache.
pub struct AudioCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    frames: usize,
}

/// The decoder's own keys and values, one row per text position so far.
pub struct TextCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl TextCache {
    pub fn new(layers: usize) -> TextCache {
        TextCache { k: vec![Vec::new(); layers], v: vec![Vec::new(); layers], len: 0 }
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Keep the first `len` positions and forget the rest.
    pub fn truncate(&mut self, len: usize, d: usize) {
        for (k, v) in self.k.iter_mut().zip(&mut self.v) {
            k.truncate(len * d);
            v.truncate(len * d);
        }
        self.len = self.len.min(len);
    }
}

pub struct Whisper {
    pub config: Config,
    conv1: Conv,
    conv2: Conv,
    /// `[audio_ctx, d]`: sinusoids, stored as a weight, added after the
    /// convolutions.
    audio_pos: Vec<f32>,
    encoder: Vec<EncoderBlock>,
    ln_audio: Norm,
    /// `[vocab, d]`: the token embedding, and transposed, the output head.
    tokens: Weight,
    /// `[text_ctx, d]`: learned, as in GPT-2.
    text_pos: Vec<f32>,
    decoder: Vec<DecoderBlock>,
    ln_text: Norm,
}

impl Whisper {
    /// The model in `files` (the safetensors of a Hub Whisper), with its
    /// matrices at `precision`.
    ///
    /// The two convolutions stay f32 whatever is asked: the first is 240
    /// columns wide at 80 bands, which does not cut into 32-weight blocks,
    /// and both together are under one percent of any Whisper.
    pub fn load(config: Config, files: &[std::path::PathBuf], precision: Precision) -> Res<Whisper> {
        let ckpt = crate::weights::Checkpoint::open(files)?;
        let src = crate::qcache::Live::new(&ckpt, precision);
        let model = Whisper::from_source(config, &src)?;
        let left = src.unread();
        if !left.is_empty() {
            return Err(format!(
                "this checkpoint holds tensors the Whisper loader never reads: {}",
                crate::weights::collapsed(&left).join(", ")
            )
            .into());
        }
        Ok(model)
    }

    fn from_source(config: Config, src: &dyn Source) -> Res<Whisper> {
        let c = &config;
        let conv = |name: &str, inp: usize, stride: usize| -> Res<Conv> {
            let w = src.vector(&format!("model.encoder.{name}.weight"))?;
            if w.len() != c.d * inp * 3 {
                return Err(format!("{name} has {} weights, not {} × {inp} × 3", w.len(), c.d).into());
            }
            Ok(Conv { w: Weight::from_f32(Tensor::new(c.d, inp * 3, w)), b: src.vector(&format!("model.encoder.{name}.bias"))?, stride })
        };
        let encoder = (0..c.encoder_layers)
            .map(|i| {
                let p = format!("model.encoder.layers.{i}");
                Ok(EncoderBlock {
                    ln1: Norm::load(src, &format!("{p}.self_attn_layer_norm"))?,
                    attn: Attn::load(src, &format!("{p}.self_attn"))?,
                    ln2: Norm::load(src, &format!("{p}.final_layer_norm"))?,
                    mlp: Mlp::load(src, &p)?,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let decoder = (0..c.decoder_layers)
            .map(|i| {
                let p = format!("model.decoder.layers.{i}");
                Ok(DecoderBlock {
                    ln1: Norm::load(src, &format!("{p}.self_attn_layer_norm"))?,
                    attn: Attn::load(src, &format!("{p}.self_attn"))?,
                    ln_cross: Norm::load(src, &format!("{p}.encoder_attn_layer_norm"))?,
                    cross: Attn::load(src, &format!("{p}.encoder_attn"))?,
                    ln2: Norm::load(src, &format!("{p}.final_layer_norm"))?,
                    mlp: Mlp::load(src, &p)?,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let tokens = src.matrix("model.decoder.embed_tokens.weight")?;
        // Tied in every Whisper released; a checkpoint that writes the head
        // out anyway writes the same numbers.
        src.skip("proj_out.weight");
        let audio_pos = src.vector("model.encoder.embed_positions.weight")?;
        let text_pos = src.vector("model.decoder.embed_positions.weight")?;
        if audio_pos.len() != c.audio_ctx * c.d || text_pos.len() != c.text_ctx * c.d || tokens.rows() != c.vocab {
            return Err("the checkpoint's embeddings do not have the shapes config.json gives".into());
        }
        Ok(Whisper {
            conv1: conv("conv1", c.n_mels, 1)?,
            conv2: conv("conv2", c.d, 2)?,
            audio_pos,
            encoder,
            ln_audio: Norm::load(src, "model.encoder.layer_norm")?,
            tokens,
            text_pos,
            decoder,
            ln_text: Norm::load(src, "model.decoder.layer_norm")?,
            config,
        })
    }

    pub fn params(&self) -> usize {
        let attn = |a: &Attn| a.q.param_count() + a.k.param_count() + a.v.param_count() + a.o.param_count();
        let mlp = |m: &Mlp| m.fc1.param_count() + m.fc2.param_count();
        self.conv1.w.param_count()
            + self.conv2.w.param_count()
            + self.encoder.iter().map(|b| attn(&b.attn) + mlp(&b.mlp)).sum::<usize>()
            + self.decoder.iter().map(|b| attn(&b.attn) + attn(&b.cross) + mlp(&b.mlp)).sum::<usize>()
            + self.tokens.param_count()
            + self.audio_pos.len()
            + self.text_pos.len()
    }

    pub fn weight_bytes(&self) -> usize {
        let attn = |a: &Attn| a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes();
        let mlp = |m: &Mlp| m.fc1.bytes() + m.fc2.bytes();
        self.conv1.w.bytes()
            + self.conv2.w.bytes()
            + self.encoder.iter().map(|b| attn(&b.attn) + mlp(&b.mlp)).sum::<usize>()
            + self.decoder.iter().map(|b| attn(&b.attn) + attn(&b.cross) + mlp(&b.mlp)).sum::<usize>()
            + self.tokens.bytes()
            + 4 * (self.audio_pos.len() + self.text_pos.len())
    }

    /// The encoder: one window of spectrogram, `[3000][n_mels]` time-major as
    /// [`super::mel::window`] lays it out, to `[1500][d]`.
    pub fn encode(&self, mel: &[f32]) -> Vec<f32> {
        let c = &self.config;
        let frames = mel.len() / c.n_mels;
        let (h, len) = self.conv1.forward(mel, frames, c.n_mels);
        let (mut x, len) = self.conv2.forward(&h, len, c.d);
        assert_eq!(len, c.audio_ctx, "a window is {} frames of spectrogram, not {frames}", 2 * c.audio_ctx);
        x.iter_mut().zip(&self.audio_pos).for_each(|(v, p)| *v += p);

        let hd = c.d / c.encoder_heads;
        for b in &self.encoder {
            let h = b.ln1.rows(&x);
            let q = b.attn.queries(&h, len, hd);
            let k = b.attn.k.matmul_bt(&h, len, None);
            let v = b.attn.v.matmul_bt(&h, len, Some(&b.attn.v_b));
            let a = attend_all(&q, &k, &v, len, len, c.encoder_heads, hd);
            add(&mut x, &b.attn.o.matmul_bt(&a, len, Some(&b.attn.o_b)));
            let h = b.ln2.rows(&x);
            add(&mut x, &b.mlp.forward(&h, len));
        }
        self.ln_audio.rows(&x)
    }

    /// The cross-attention keys and values every decoder block will read for
    /// this window: [`Whisper::encode`]'s output, projected once per block.
    pub fn listen(&self, audio: &[f32]) -> AudioCache {
        let frames = audio.len() / self.config.d;
        let (k, v) = self
            .decoder
            .iter()
            .map(|b| (b.cross.k.matmul_bt(audio, frames, None), b.cross.v.matmul_bt(audio, frames, Some(&b.cross.v_b))))
            .unzip();
        AudioCache { k, v, frames }
    }

    /// Run `tokens` through the decoder after the `cache.len()` already in
    /// it, and return the final hidden state of each, `[m][d]`.
    ///
    /// One token at a time while generating, and a whole prompt at once to
    /// begin a window: every row is a token, and the causal mask is the
    /// `pos0 + i + 1` positions row `i` is allowed to see.
    pub fn decode(&self, tokens: &[u32], cache: &mut TextCache, audio: &AudioCache) -> Res<Vec<f32>> {
        let c = &self.config;
        let (m, d, pos0) = (tokens.len(), c.d, cache.len);
        if pos0 + m > c.text_ctx {
            return Err(format!("{} text positions is past the decoder's {}", pos0 + m, c.text_ctx).into());
        }
        let mut x = vec![0.0f32; m * d];
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= c.vocab {
                return Err(format!("token {t} is outside the vocabulary of {}", c.vocab).into());
            }
            let row = &mut x[i * d..(i + 1) * d];
            row.copy_from_slice(&self.tokens.row(t as usize));
            row.iter_mut().zip(&self.text_pos[(pos0 + i) * d..]).for_each(|(v, p)| *v += p);
        }

        let hd = d / c.decoder_heads;
        for (l, b) in self.decoder.iter().enumerate() {
            let h = b.ln1.rows(&x);
            let q = b.attn.queries(&h, m, hd);
            cache.k[l].extend(b.attn.k.matmul_bt(&h, m, None));
            cache.v[l].extend(b.attn.v.matmul_bt(&h, m, Some(&b.attn.v_b)));
            let a = attend_rows(&q, &cache.k[l], &cache.v[l], c.decoder_heads, hd, |i| pos0 + i + 1);
            add(&mut x, &b.attn.o.matmul_bt(&a, m, Some(&b.attn.o_b)));

            let h = b.ln_cross.rows(&x);
            let q = b.cross.queries(&h, m, hd);
            let a = attend_rows(&q, &audio.k[l], &audio.v[l], c.decoder_heads, hd, |_| audio.frames);
            add(&mut x, &b.cross.o.matmul_bt(&a, m, Some(&b.cross.o_b)));

            let h = b.ln2.rows(&x);
            add(&mut x, &b.mlp.forward(&h, m));
        }
        cache.len += m;
        Ok(self.ln_text.rows(&x))
    }

    /// The next-token scores for one row of [`Whisper::decode`]'s output.
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.tokens.matvec_bt(hidden, None)
    }
}

/// The model as the decoding loop drives it ([`super::Model`]): on the CPU,
/// on threads of its own.
///
/// The threads are why this is not [`Whisper`] itself. Work is handed to
/// them whole, a window at a time: a parallel loop started from a thread
/// outside a pool takes rayon's slow path at every call, and a decoder step
/// is dozens of them. See [`crate::model::threads`].
pub struct Cpu {
    pub whisper: Whisper,
    pool: rayon::ThreadPool,
    precision: Precision,
}

impl Cpu {
    pub fn load(config: Config, files: &[std::path::PathBuf], precision: Precision) -> Res<Cpu> {
        Ok(Cpu { whisper: Whisper::load(config, files, precision)?, pool: crate::model::pool(), precision })
    }
}

impl super::Model for Cpu {
    fn config(&self) -> &Config {
        &self.whisper.config
    }

    fn hear(&self, window: &[f32]) -> Res<super::Heard> {
        Ok(super::Heard(Box::new(self.whisper.listen(&self.whisper.encode(window)))))
    }

    fn said(&self) -> super::Said {
        super::Said(Box::new(TextCache::new(self.whisper.config.decoder_layers)))
    }

    fn decode(&self, tokens: &[u32], said: &mut super::Said, heard: &super::Heard, rows: &[usize]) -> Res<Vec<Vec<f32>>> {
        let d = self.whisper.config.d;
        let h = self.whisper.decode(tokens, said.get_mut()?, heard.get()?)?;
        Ok(rows.iter().map(|&r| self.whisper.logits(&h[r * d..(r + 1) * d])).collect())
    }

    fn within(&self, work: &mut (dyn FnMut() + Send)) {
        self.pool.install(work)
    }

    fn params(&self) -> usize {
        self.whisper.params()
    }

    fn weight_bytes(&self) -> usize {
        self.whisper.weight_bytes()
    }

    fn backend(&self) -> String {
        format!("cpu {}", self.precision)
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// Attention with no mask: each of `n_q` queries over all `n_k` keys, head by
/// head, as two matrix products.
///
/// The encoder's case — 1500 frames against 1500 — is a lot of arithmetic in
/// a shape [`gemm_bt`] already does well, so it is written as that rather
/// than as [`attend_rows`]'s dot product per pair:
///
/// ```text
///   S = Q Kᵀ        [n_q × n_k]   one row of scores per query
///   P = softmax(S)  by rows
///   O = P V         [n_q × hd]
/// ```
///
/// `gemm_bt` computes `X Wᵀ` with `W` row-major, and writes it transposed.
/// The first product is asked for with `Q` as `W` and `K` as `X`, which
/// comes back `[query][key]` — rows of scores, as the softmax wants. The
/// second has `Vᵀ` as `W` and `P` as `X`, and comes back `[dim][query]`,
/// which is put back the right way round on the way out.
fn attend_all(q: &[f32], k: &[f32], v: &[f32], n_q: usize, n_k: usize, heads: usize, hd: usize) -> Vec<f32> {
    let d = heads * hd;
    let mut out = vec![0.0f32; n_q * d];
    let head_of = |x: &[f32], n: usize, h: usize| -> Vec<f32> { (0..n).flat_map(|r| x[r * d + h * hd..r * d + (h + 1) * hd].iter().copied()).collect() };
    for h in 0..heads {
        let qh = Tensor::new(n_q, hd, head_of(q, n_q, h));
        let kh = head_of(k, n_k, h);
        let mut scores = vec![0.0f32; n_q * n_k];
        gemm_bt(&kh, n_k, &qh, &mut scores);
        scores.par_chunks_mut(n_k).for_each(softmax_inplace);

        let vh = head_of(v, n_k, h);
        let vt = Tensor::new(n_k, hd, vh).transposed();
        let mut oh_t = vec![0.0f32; hd * n_q];
        gemm_bt(&scores, n_q, &vt, &mut oh_t);
        for (j, col) in oh_t.chunks(n_q).enumerate() {
            for (i, &val) in col.iter().enumerate() {
                out[i * d + h * hd + j] = val;
            }
        }
    }
    out
}

/// Attention for a few queries: query row `i` reads the first `visible(i)`
/// rows of `k` and `v`, and every (query, head) pair is a task.
///
/// The decoder's case, both kinds: a handful of text queries against the
/// text so far (`visible` is the causal mask), or against all 1500 frames
/// of audio (`visible` is all of them).
fn attend_rows(q: &[f32], k: &[f32], v: &[f32], heads: usize, hd: usize, visible: impl Fn(usize) -> usize + Sync) -> Vec<f32> {
    let d = heads * hd;
    let mut out = vec![0.0f32; q.len()];
    out.par_chunks_mut(hd).enumerate().for_each(|(task, slot)| {
        let (i, h) = (task / heads, task % heads);
        let qh = &q[i * d + h * hd..i * d + (h + 1) * hd];
        let n = visible(i);
        let mut scores: Vec<f32> = (0..n).map(|t| dot(qh, &k[t * d + h * hd..t * d + (h + 1) * hd])).collect();
        softmax_inplace(&mut scores);
        for (t, &w) in scores.iter().enumerate() {
            let vh = &v[t * d + h * hd..t * d + (h + 1) * hd];
            slot.iter_mut().zip(vh).for_each(|(o, x)| *o += w * x);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_attentions_agree_on_the_unmasked_case() {
        // The encoder's matmul form and the decoder's row form are two ways
        // to the same numbers.
        let (n_q, n_k, heads, hd) = (7, 13, 3, 8);
        let d = heads * hd;
        let f = |n: usize, s: usize| -> Vec<f32> { (0..n * d).map(|i| ((i * s % 17) as f32 - 8.0) / 9.0).collect() };
        let (q, k, v) = (f(n_q, 5), f(n_k, 7), f(n_k, 11));
        let a = attend_all(&q, &k, &v, n_q, n_k, heads, hd);
        let b = attend_rows(&q, &k, &v, heads, hd, |_| n_k);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    /// Whisper tiny's shape, from its `config.json`, comes to the 37.7
    /// million weights the loaded model counts (`kvad transcribe` prints
    /// them), without the 60 thousand in its norms and biases.
    #[test]
    fn the_config_says_what_the_weights_will_weigh() {
        let c = Config {
            n_mels: 80,
            d: 384,
            encoder_layers: 4,
            encoder_heads: 6,
            encoder_ffn: 1536,
            decoder_layers: 4,
            decoder_heads: 6,
            decoder_ffn: 1536,
            audio_ctx: 1500,
            text_ctx: 448,
            vocab: 51865,
        };
        let (matrices, floats) = c.params();
        assert_eq!(matrices + floats, 37_713_792);
        assert_eq!(floats, 92_160 + 442_368 + 748_032);
    }

    #[test]
    fn a_convolution_is_its_sum_of_taps() {
        // Two input channels, one output, stride 2, against the sum written out.
        let (len, inp) = (5, 2);
        let x: Vec<f32> = (0..len * inp).map(|i| i as f32 + 1.0).collect();
        let w: Vec<f32> = (0..inp * 3).map(|i| (i as f32 - 2.0) / 4.0).collect();
        let conv = Conv { w: Weight::from_f32(Tensor::new(1, inp * 3, w.clone())), b: vec![0.5], stride: 2 };
        let (got, out_len) = conv.forward(&x, len, inp);
        assert_eq!(out_len, 3);
        for t in 0..out_len {
            let mut want = 0.5;
            for i in 0..inp {
                for k in 0..3 {
                    let at = (2 * t + k) as isize - 1;
                    if (0..len as isize).contains(&at) {
                        want += w[i * 3 + k] * x[at as usize * inp + i];
                    }
                }
            }
            let mut want = [want];
            gelu_erf_inplace(&mut want);
            assert!((got[t] - want[0]).abs() < 1e-5, "frame {t}: {} vs {}", got[t], want[0]);
        }
    }
}
