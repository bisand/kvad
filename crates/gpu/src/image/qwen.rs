//! Qwen-Image: the same three stages as SDXL, each replaced by something
//! bigger and more like a language model.
//!
//! - The **text encoder** is not CLIP but Qwen2.5-VL-7B, a chat model. The
//!   prompt is wrapped in a system message asking for a description of an
//!   image, run through all 28 layers, and the hidden states of the prompt's
//!   own tokens — not a pooled vector — are the conditioning.
//! - The **denoiser** is not a UNet but an MMDiT: 60 transformer blocks over
//!   image patches and prompt tokens together. There is no convolution in it
//!   at all; the image is cut into 2×2 patches of the latent, and where each
//!   patch is comes from a three-axis RoPE rather than from a grid.
//! - The **VAE** is a video VAE, of which an image is a one-frame video.
//!
//! And the scheduler is flow matching rather than noise prediction (see
//! [`super::schedule`]). Every shape is in `docs/image-plan.md`, read off the
//! checkpoint.
//!
//! At twenty billion parameters the denoiser does not fit in bf16 beside
//! anything else on a 48 GB machine, so this pipeline runs quantised by
//! default, through the same [`Loader`] and the same quantised-weight cache as
//! the text models. Activations are f32, which is what candle's quantised
//! kernels take.

use super::mmdit::{norm_out, Double, Names, Shape};
use super::nn::{latent_preview, noise, timestep_embedding, to_rgb8, Conv2d, Ctx, Linear};
use super::schedule;
use super::{finish, local_file, open, read_json};
use crate::common::{Loader, Reader, Stored};
use crate::qcache::Vault;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::{json, Value};
use kvad::weights::{fetch_file, Watcher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The repo ships the tokenizer as `vocab.json` and `merges.txt`; this one
/// has the same vocabulary as a `tokenizer.json`.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen2.5-VL-7B-Instruct";

/// The system prompt every prompt is wrapped in, and how many tokens of it to
/// throw away afterwards. Both from diffusers' `QwenImagePipeline`.
const TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
const DROP: usize = 34;

/// Latent channels, and the VAE's per-channel statistics come from its config.
const Z: usize = 16;

// ---------------------------------------------------------------------------
// The text encoder: Qwen2.5-VL's language tower
// ---------------------------------------------------------------------------

struct TextLayer {
    ln1: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// A Qwen2 decoder run for its hidden states.
///
/// Its RoPE is "multimodal" — three position components, for time, height
/// and width, spread over sections of the head — but with no image in the
/// prompt all three are the token's index, and it is ordinary RoPE. So this
/// is the Llama-family forward pass, stopped before the output head, which is
/// never loaded.
struct TextEncoder {
    /// Parameters the checkpoint has and this never loads: the output head,
    /// recorded as known so the unread-weights guard is satisfied.
    unused: usize,
    embed: TextEmbed,
    layers: Vec<TextLayer>,
    norm: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f64,
    eps: f32,
}

enum TextEmbed {
    Dense(Tensor),
    Quant(Arc<QTensor>),
}

impl TextEncoder {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, config: &Value) -> Res<Self> {
        let c = config.get("text_config").unwrap_or(config);
        let n = |k: &str| -> Res<usize> {
            c.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("text encoder config has no `{k}`").into())
        };
        let (e, heads, kv_heads, inter, layers, vocab) = (
            n("hidden_size")?,
            n("num_attention_heads")?,
            n("num_key_value_heads")?,
            n("intermediate_size")?,
            n("num_hidden_layers")?,
            n("vocab_size")?,
        );
        let head_dim = e / heads;
        let kv = kv_heads * head_dim;
        // What text-to-image never uses: the vision tower (no image goes in)
        // and the output head (hidden states come out, not logits).
        r.skip_under("visual.");
        r.record("lm_head.weight");

        let m = r.pp("model");
        let embed = match cx.ld.quant {
            Some(_) => TextEmbed::Quant(Arc::new(cx.ld.quantized(&m, "embed_tokens.weight", vocab, e, Stored::OutIn)?)),
            None => TextEmbed::Dense(cx.get(&m, (vocab, e), "embed_tokens.weight")?),
        };
        let mut out = Vec::with_capacity(layers);
        for i in 0..layers {
            let l = m.pp(format!("layers.{i}"));
            let a = l.pp("self_attn");
            let f = l.pp("mlp");
            out.push(TextLayer {
                ln1: cx.get(&l, e, "input_layernorm.weight")?,
                q: Linear::load(cx, &a, "q_proj", e, e, true)?,
                k: Linear::load(cx, &a, "k_proj", e, kv, true)?,
                v: Linear::load(cx, &a, "v_proj", e, kv, true)?,
                o: Linear::load(cx, &a, "o_proj", e, e, false)?,
                ln2: cx.get(&l, e, "post_attention_layernorm.weight")?,
                gate: Linear::load(cx, &f, "gate_proj", e, inter, false)?,
                up: Linear::load(cx, &f, "up_proj", e, inter, false)?,
                down: Linear::load(cx, &f, "down_proj", inter, e, false)?,
            });
        }
        Ok(TextEncoder {
            unused: vocab * e,
            embed,
            layers: out,
            norm: cx.get(&m, e, "norm.weight")?,
            heads,
            kv_heads,
            head_dim,
            theta: c.get("rope_theta").and_then(Value::as_f64).unwrap_or(1e6),
            eps: c.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
        })
    }

    /// The last layer's hidden states after the final norm, `[1, L, width]`.
    fn forward(&self, ids: &[u32], device: &Device, dtype: DType) -> Res<Tensor> {
        let l = ids.len();
        let ids_t = Tensor::new(ids, device)?;
        let mut x = match &self.embed {
            TextEmbed::Dense(t) => t.index_select(&ids_t, 0)?,
            TextEmbed::Quant(q) => q.embedding(&ids_t)?,
        }
        .to_dtype(dtype)?
        .unsqueeze(0)?;

        // Rotation angles for positions 0..l, halves convention (the first
        // half of a head pairs with the second).
        let half = self.head_dim / 2;
        let mut angles = Vec::with_capacity(l * half);
        for p in 0..l {
            for i in 0..half {
                angles.push((p as f64 / self.theta.powf(2.0 * i as f64 / self.head_dim as f64)) as f32);
            }
        }
        let angles = Tensor::from_vec(angles, (l, half), device)?;
        let (cos, sin) = (angles.cos()?.to_dtype(dtype)?, angles.sin()?.to_dtype(dtype)?);
        let mask: Vec<f32> = (0..l * l).map(|i| if i % l > i / l { f32::NEG_INFINITY } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, (1, 1, l, l), device)?;

        let group = self.heads / self.kv_heads;
        for layer in &self.layers {
            let h = ops::rms_norm(&x, &layer.ln1, self.eps)?;
            let split = |t: Tensor, n: usize| -> candle_core::Result<Tensor> {
                t.reshape((1, l, n, self.head_dim))?.transpose(1, 2)?.contiguous()
            };
            let q = candle_nn::rotary_emb::rope(&split(layer.q.forward(&h)?, self.heads)?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&split(layer.k.forward(&h)?, self.kv_heads)?, &cos, &sin)?;
            let v = split(layer.v.forward(&h)?, self.kv_heads)?;
            // Grouped-query attention, the way `model.rs` does it: fold the
            // query heads that share a KV head into one batch of rows.
            let qg = q.reshape((1, self.kv_heads, group * l, self.head_dim))?;
            let att = (qg.matmul(&k.transpose(2, 3)?.contiguous()?)?.to_dtype(DType::F32)? / (self.head_dim as f64).sqrt())?;
            let att = att.reshape((1, self.heads, l, l))?.broadcast_add(&mask)?;
            let att = ops::softmax_last_dim(&att)?.to_dtype(dtype)?.reshape((1, self.kv_heads, group * l, l))?;
            let a = att.matmul(&v)?.reshape((1, self.heads, l, self.head_dim))?;
            let a = a.transpose(1, 2)?.contiguous()?.reshape((1, l, self.heads * self.head_dim))?;
            x = (x + layer.o.forward(&a)?)?;

            let h = ops::rms_norm(&x, &layer.ln2, self.eps)?;
            let g = (candle_nn::ops::silu(&layer.gate.forward(&h)?)? * layer.up.forward(&h)?)?;
            x = (x + layer.down.forward(&g)?)?;
        }
        Ok(ops::rms_norm(&x, &self.norm, self.eps)?)
    }
}

// ---------------------------------------------------------------------------
// The denoiser: a two-stream MMDiT
// ---------------------------------------------------------------------------

struct DitConfig {
    layers: usize,
    heads: usize,
    head_dim: usize,
    in_channels: usize,
    out_channels: usize,
    patch: usize,
    joint: usize,
    axes: [usize; 3],
}

impl DitConfig {
    fn from_json(v: &Value) -> Res<Self> {
        let n = |k: &str| -> Res<usize> {
            v.get(k).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("transformer config has no `{k}`").into())
        };
        if v.get("guidance_embeds").and_then(Value::as_bool) == Some(true) {
            return Err("a guidance-distilled Qwen-Image is not implemented".into());
        }
        let axes: Vec<usize> = v
            .get("axes_dims_rope")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect())
            .unwrap_or_default();
        let axes: [usize; 3] = axes.try_into().map_err(|_| "`axes_dims_rope` should have three entries")?;
        Ok(DitConfig {
            layers: n("num_layers")?,
            heads: n("num_attention_heads")?,
            head_dim: n("attention_head_dim")?,
            in_channels: n("in_channels")?,
            out_channels: n("out_channels")?,
            patch: n("patch_size")?,
            joint: n("joint_attention_dim")?,
            axes,
        })
    }

    fn width(&self) -> usize {
        self.heads * self.head_dim
    }

    fn shape(&self) -> Shape {
        Shape { heads: self.heads, head_dim: self.head_dim }
    }
}

struct Dit {
    cfg: DitConfig,
    img_in: Linear,
    txt_norm: Tensor,
    txt_in: Linear,
    time1: Linear,
    time2: Linear,
    blocks: Vec<Double>,
    norm_out: Linear,
    proj_out: Linear,
}

impl Dit {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: DitConfig) -> Res<Self> {
        let w = cfg.width();
        // Qwen-Image's names for the two things the shared block does not
        // fix: where each stream's modulation and MLP live.
        let img = Names { modulate: "img_mod.1", mlp_in: "img_mlp.net.0.proj", mlp_out: "img_mlp.net.2" };
        let txt = Names { modulate: "txt_mod.1", mlp_in: "txt_mlp.net.0.proj", mlp_out: "txt_mlp.net.2" };
        let shape = cfg.shape();
        let blocks = (0..cfg.layers)
            .map(|i| Double::load(cx, &r.pp(format!("transformer_blocks.{i}")), shape, &img, &txt))
            .collect::<Res<Vec<_>>>()?;
        let t = r.pp("time_text_embed.timestep_embedder");
        let patch_out = cfg.patch * cfg.patch * cfg.out_channels;
        Ok(Dit {
            img_in: Linear::load(cx, r, "img_in", cfg.in_channels, w, true)?,
            txt_norm: cx.get(r, cfg.joint, "txt_norm.weight")?,
            txt_in: Linear::load(cx, r, "txt_in", cfg.joint, w, true)?,
            time1: Linear::load(cx, &t, "linear_1", 256, w, true)?,
            time2: Linear::load(cx, &t, "linear_2", w, w, true)?,
            norm_out: Linear::load(cx, r, "norm_out.linear", w, 2 * w, true)?,
            proj_out: Linear::load(cx, r, "proj_out", w, patch_out, true)?,
            blocks,
            cfg,
        })
    }

    /// The velocity at noise level `sigma`, for packed latents `x`
    /// (`[1, rows·cols, 64]`) and the prompt's hidden states `txt`.
    fn forward(&self, x: &Tensor, txt: &Tensor, sigma: f64, rows: usize, cols: usize) -> Res<Tensor> {
        let dev = x.device();
        let dtype = x.dtype();
        let (n_img, n_txt) = (x.dim(1)?, txt.dim(1)?);
        let mut img = self.img_in.forward(x)?;
        let mut txt = self.txt_in.forward(&ops::rms_norm(txt, &self.txt_norm, 1e-6)?)?;

        // The model multiplies σ by 1000 before embedding it, as a timestep.
        let t = timestep_embedding(&[sigma * 1000.0], 256, true, 0.0, dev)?.to_dtype(dtype)?;
        let temb = self.time2.forward(&self.time1.forward(&t)?.silu()?)?;
        let temb_act = temb.silu()?;

        let half = self.cfg.head_dim / 2;
        let angles = Tensor::from_vec(angles(&self.cfg, rows, cols, n_txt), (n_img + n_txt, half), dev)?;
        let (cos, sin) = (angles.cos()?.to_dtype(dtype)?, angles.sin()?.to_dtype(dtype)?);
        let (cos_img, sin_img) = (cos.narrow(0, 0, n_img)?, sin.narrow(0, 0, n_img)?);
        let (cos_txt, sin_txt) = (cos.narrow(0, n_img, n_txt)?, sin.narrow(0, n_img, n_txt)?);

        let shape = self.cfg.shape();
        for block in &self.blocks {
            (img, txt) = block.forward(shape, &img, &txt, &temb_act, (&cos_img, &sin_img), (&cos_txt, &sin_txt))?;
        }
        let img = norm_out(&self.norm_out, &img, &temb_act, self.cfg.width())?;
        Ok(self.proj_out.forward(&img)?)
    }
}

/// Rotation angles for every token, `[tokens, head_dim / 2]`: the image's
/// patches first, then the text's.
///
/// Each head is split between three axes — frame, row, column — and each
/// axis rotates its share of the head by its own position. Rows and
/// columns are *centred*: a 64-patch side runs −32 … 31, so the middle
/// of the image is the origin whatever its size. The text sits after the
/// image on the diagonal, at `max(rows, cols) / 2 + i` on every axis, so
/// that no text token shares a position with a patch.
fn angles(cfg: &DitConfig, rows: usize, cols: usize, text: usize) -> Vec<f32> {
    let freqs = |dim: usize| -> Vec<f64> {
        (0..dim / 2).map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / dim as f64)).collect()
    };
    let (ft, fh, fw) = (freqs(cfg.axes[0]), freqs(cfg.axes[1]), freqs(cfg.axes[2]));
    let half = cfg.head_dim / 2;
    let mut out = Vec::with_capacity((rows * cols + text) * half);
    let centred = |i: usize, n: usize| i as f64 - (n - n / 2) as f64;
    for r in 0..rows {
        for c in 0..cols {
            out.extend(ft.iter().map(|f| (0.0 * f) as f32));
            out.extend(fh.iter().map(|f| (centred(r, rows) * f) as f32));
            out.extend(fw.iter().map(|f| (centred(c, cols) * f) as f32));
        }
    }
    let start = (rows / 2).max(cols / 2);
    for i in 0..text {
        let p = (start + i) as f64;
        out.extend(ft.iter().chain(&fh).chain(&fw).map(|f| (p * f) as f32));
    }
    out
}

// ---------------------------------------------------------------------------
// The VAE: a video decoder, run on one frame
// ---------------------------------------------------------------------------

/// A causal 3D convolution applied to a single frame, as the 2D convolution
/// it collapses to.
///
/// The kernel is `[out, in, t, k, k]`, and the input is padded with `t − 1`
/// zero frames *in front* so that no frame sees the future. With one frame
/// of real data, only the kernel's last temporal slice ever multiplies
/// anything that is not zero.
fn causal(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, (cin, cout, t, k): (usize, usize, usize, usize)) -> Res<Conv2d> {
    let r = r.pp(name);
    let w = r.get((cout, cin, t, k, k), "weight")?.narrow(2, t - 1, 1)?.squeeze(2)?.contiguous()?;
    let w = w.to_dtype(cx.dtype)?.to_device(cx.device())?;
    Conv2d::from_parts(w, cx.get(&r, cout, "bias")?, 1)
}

/// RMS norm over channels: each position's channel vector scaled to length
/// `√C`, then by a learned gain. The Wan VAE's replacement for group norm.
struct RmsChannels {
    gamma: Tensor,
    root: f64,
}

impl RmsChannels {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, c: usize, images: bool) -> Res<Self> {
        let g = match images {
            true => r.get((c, 1, 1), &format!("{name}.gamma"))?,
            false => r.get((c, 1, 1, 1), &format!("{name}.gamma"))?.squeeze(3)?,
        };
        Ok(RmsChannels { gamma: g.to_dtype(cx.dtype)?.to_device(cx.device())?.unsqueeze(0)?, root: (c as f64).sqrt() })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let norm = x32.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(1e-12)?;
        (x32.broadcast_div(&norm)? * self.root)?.to_dtype(dtype)?.broadcast_mul(&self.gamma)
    }
}

struct WanResnet {
    norm1: RmsChannels,
    conv1: Conv2d,
    norm2: RmsChannels,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl WanResnet {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cin: usize, cout: usize) -> Res<Self> {
        Ok(WanResnet {
            norm1: RmsChannels::load(cx, r, "norm1", cin, false)?,
            conv1: causal(cx, r, "conv1", (cin, cout, 3, 3))?,
            norm2: RmsChannels::load(cx, r, "norm2", cout, false)?,
            conv2: causal(cx, r, "conv2", (cout, cout, 3, 3))?,
            shortcut: match cin != cout {
                true => Some(causal(cx, r, "conv_shortcut", (cin, cout, 1, 1))?),
                false => None,
            },
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;
        match &self.shortcut {
            Some(s) => s.forward(x)? + h,
            None => x + h,
        }
    }
}

struct WanDecoder {
    post_quant: Conv2d,
    conv_in: Conv2d,
    mid: (WanResnet, (RmsChannels, Conv2d, Conv2d), WanResnet),
    up: Vec<(Vec<WanResnet>, Option<Conv2d>)>,
    norm_out: RmsChannels,
    conv_out: Conv2d,
    mean: Tensor,
    std: Tensor,
}

impl WanDecoder {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, config: &Value) -> Res<Self> {
        let n = |k: &str| config.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or(format!("VAE config has no `{k}`"));
        let (base, z, res) = (n("base_dim")?, n("z_dim")?, n("num_res_blocks")?);
        let mult: Vec<usize> = config["dim_mult"].as_array().ok_or("VAE config has no `dim_mult`")?.iter().filter_map(Value::as_u64).map(|v| v as usize).collect();
        let stats = |k: &str| -> Res<Tensor> {
            let v: Vec<f32> = config[k].as_array().ok_or(format!("VAE config has no `{k}`"))?.iter().filter_map(Value::as_f64).map(|f| f as f32).collect();
            Ok(Tensor::from_vec(v, (1, z, 1, 1), cx.device())?.to_dtype(cx.dtype)?)
        };
        if !config["attn_scales"].as_array().is_some_and(|a| a.is_empty()) {
            return Err("a Wan VAE with attention outside its middle block is not implemented".into());
        }

        // The encoder makes latents from images, which text-to-image never
        // does; the temporal upsamplers only run from the second frame on.
        r.skip_under("encoder.");
        r.skip_under("quant_conv.");

        let d = r.pp("decoder");
        let top = base * mult[mult.len() - 1];
        let m = d.pp("mid_block");
        let a = m.pp("attentions.0");
        let mid = (
            WanResnet::load(cx, &m.pp("resnets.0"), top, top)?,
            (
                RmsChannels::load(cx, &a, "norm", top, true)?,
                causal_2d(cx, &a, "to_qkv", (top, 3 * top))?,
                causal_2d(cx, &a, "proj", (top, top))?,
            ),
            WanResnet::load(cx, &m.pp("resnets.1"), top, top)?,
        );

        // Widths top down: the widest first, then back through `dim_mult`.
        let dims: Vec<usize> = std::iter::once(mult[mult.len() - 1]).chain(mult.iter().rev().copied()).map(|u| base * u).collect();
        let mut up = Vec::new();
        for i in 0..dims.len() - 1 {
            // After the first level, an upsampler has halved the width.
            let cin = if i > 0 { dims[i] / 2 } else { dims[i] };
            let cout = dims[i + 1];
            let b = d.pp(format!("up_blocks.{i}"));
            let resnets = (0..=res)
                .map(|j| WanResnet::load(cx, &b.pp(format!("resnets.{j}")), if j == 0 { cin } else { cout }, cout))
                .collect::<Res<Vec<_>>>()?;
            let upsample = match i + 1 < mult.len() {
                true => {
                    b.skip_under("upsamplers.0.time_conv");
                    Some(Conv2d::load(cx, &b, "upsamplers.0.resample.1", (cout, cout / 2, 3), 1)?)
                }
                false => None,
            };
            up.push((resnets, upsample));
        }
        let last = dims[dims.len() - 1];
        Ok(WanDecoder {
            post_quant: causal(cx, r, "post_quant_conv", (z, z, 1, 1))?,
            conv_in: causal(cx, &d, "conv_in", (z, top, 3, 3))?,
            norm_out: RmsChannels::load(cx, &d, "norm_out", last, false)?,
            conv_out: causal(cx, &d, "conv_out", (last, 3, 3, 3))?,
            mid,
            up,
            mean: stats("latents_mean")?,
            std: stats("latents_std")?,
        })
    }

    /// `[1, 16, h, w]` normalised latents to `[1, 3, H, W]` in `[−1, 1]`.
    fn decode(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let mut h = self.conv_in.forward(&self.post_quant.forward(&z)?)?;
        h = self.mid.0.forward(&h)?;
        h = self.attend(&h)?;
        h = self.mid.2.forward(&h)?;
        for (resnets, upsample) in &self.up {
            for r in resnets {
                h = r.forward(&h)?;
            }
            if let Some(conv) = upsample {
                let (_, _, hh, ww) = h.dims4()?;
                h = conv.forward(&h.upsample_nearest2d(hh * 2, ww * 2)?)?;
            }
        }
        self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)?.clamp(-1f32, 1f32)
    }

    /// One head as wide as the channels, over every position.
    fn attend(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (norm, qkv, proj) = &self.mid.1;
        let (_, c, h, w) = x.dims4()?;
        let s = qkv.forward(&norm.forward(x)?)?.reshape((1, 3 * c, h * w))?.transpose(1, 2)?.contiguous()?;
        let part = |i: usize| s.narrow(2, i * c, c)?.unsqueeze(1)?.contiguous();
        let a = super::nn::written_out(&part(0)?, &part(1)?, &part(2)?, 1.0 / (c as f64).sqrt())?;
        let a = a.squeeze(1)?.transpose(1, 2)?.contiguous()?.reshape((1, c, h, w))?;
        proj.forward(&a)? + x
    }
}

/// A 1×1 `Conv2d` stored as one, with its trailing unit axes.
fn causal_2d(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, (cin, cout): (usize, usize)) -> Res<Conv2d> {
    let r = r.pp(name);
    let w = cx.get(&r, (cout, cin, 1, 1), "weight")?;
    Conv2d::from_parts(w, cx.get(&r, cout, "bias")?, 1)
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

pub struct QwenImage {
    tok: tokenizers::Tokenizer,
    text: TextEncoder,
    dit: Dit,
    vae: WanDecoder,
    scheduler: Value,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    params: usize,
    bytes: usize,
}

/// The files of one component: its config and its weights, a shard index's
/// worth or one file.
fn component(repo: &str, dir: &str, weights: &str, watch: &Watcher) -> Res<(Value, Vec<PathBuf>)> {
    let config = read_json(&fetch_file(repo, &format!("{dir}/config.json"), watch)?)?;
    let index = format!("{dir}/{weights}.safetensors.index.json");
    let paths = match fetch_file(repo, &index, watch) {
        Ok(index) => {
            let map = read_json(&index)?;
            let mut shards: Vec<String> =
                map["weight_map"].as_object().ok_or("a shard index with no weight_map")?.values().filter_map(Value::as_str).map(str::to_string).collect();
            shards.sort();
            shards.dedup();
            shards.iter().map(|s| fetch_file(repo, &format!("{dir}/{s}"), watch)).collect::<Res<Vec<_>>>()?
        }
        Err(_) => vec![fetch_file(repo, &format!("{dir}/{weights}.safetensors"), watch)?],
    };
    Ok((config, paths))
}

impl QwenImage {
    pub fn load(
        repo: &str,
        quant: Option<GgmlDType>,
        device: Device,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Res<Self> {
        // Quantised weights take f32 activations; dense ones run in the
        // checkpoint's own bf16.
        let dtype = if quant.is_some() { DType::F32 } else { DType::BF16 };
        let label = quant.map(crate::common::ggml_name).unwrap_or("bf16");

        progress("reading the tokenizer");
        let tok = tokenizers::Tokenizer::from_file(fetch_file(TOKENIZER_REPO, "tokenizer.json", watch)?).map_err(|e| e.to_string())?;
        let prefix = TEMPLATE.split("{}").next().unwrap_or_default();
        let dropped = tok.encode(prefix, false).map_err(|e| e.to_string())?.len();
        if dropped != DROP {
            return Err(format!(
                "the prompt template's system part is {dropped} tokens with this tokenizer, and the pipeline drops {DROP}: \
                 they are not the same vocabulary"
            )
            .into());
        }
        let scheduler = read_json(&fetch_file(repo, "scheduler/scheduler_config.json", watch)?)?;
        schedule::flow(&scheduler, 50, 4096)?;

        let mut params = 0;
        let mut bytes = 0;
        let (config, paths) = component(repo, "text_encoder", "model", watch)?;
        progress(&format!("loading the text encoder at {label}"));
        let vault = Vault::open_as(&format!("{repo}/text_encoder"), &paths, json!({ "component": "text_encoder" }), quant, progress);
        let mut vault = vault;
        let text = {
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
            let r = open(&paths, dtype)?;
            let text = TextEncoder::load(&cx, &r, &config)?;
            params += finish("text encoder", &paths, &r)? - text.unused;
            text
        };
        vault.finish(progress);

        let (config, paths) = component(repo, "transformer", "diffusion_pytorch_model", watch)?;
        progress(&format!("loading the transformer at {label} — twenty billion parameters, which takes a while the first time"));
        let dcfg = DitConfig::from_json(&config)?;
        let shape = json!({ "component": "transformer", "layers": dcfg.layers, "width": dcfg.width() });
        let mut vault = Vault::open_as(&format!("{repo}/transformer"), &paths, shape, quant, progress);
        let dit = {
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
            let r = open(&paths, dtype)?;
            let dit = Dit::load(&cx, &r, dcfg)?;
            params += finish("transformer", &paths, &r)?;
            dit
        };
        vault.finish(progress);

        progress("loading the VAE");
        let config = read_json(&fetch_file(repo, "vae/config.json", watch)?)?;
        let paths = vec![fetch_file(repo, "vae/diffusion_pytorch_model.safetensors", watch)?];
        let vault = Vault::off();
        // The VAE is small and works at full resolution, where f32
        // activations would be gigabytes; it keeps the checkpoint's bf16.
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype: DType::BF16 };
        let r = open(&paths, DType::BF16)?;
        let vae = WanDecoder::load(&cx, &r, &config)?;
        let vae_params = finish("VAE", &paths, &r)?;
        params += vae_params;

        bytes += weight_bytes_at(params - vae_params, quant) as usize + vae_params * 2;
        device.synchronize()?;
        progress(&format!("loaded Qwen-Image: {:.1} B parameters at {label}", params as f64 / 1e9));
        Ok(QwenImage { tok, text, dit, vae, scheduler, device, dtype, quant, params, bytes })
    }

    /// The prompt as the transformer reads it: `[1, tokens, 3584]`.
    fn encode(&self, prompt: &str) -> Res<Tensor> {
        let ids = self.tok.encode(TEMPLATE.replace("{}", prompt), false).map_err(|e| e.to_string())?;
        let ids = ids.get_ids();
        // diffusers truncates at 1024 tokens after the template; so does this.
        let ids = &ids[..ids.len().min(DROP + 1024)];
        if ids.len() <= DROP {
            return Err("the prompt encoded to nothing".into());
        }
        let h = self.text.forward(ids, &self.device, self.dtype)?;
        Ok(h.narrow(1, DROP, ids.len() - DROP)?.contiguous()?)
    }
}

/// Bytes for `params` weights at a quantisation, the way candle stores them.
fn weight_bytes_at(params: usize, quant: Option<GgmlDType>) -> u64 {
    let per_block = |bytes: u64, block: u64| params as u64 * bytes / block;
    match quant {
        None => params as u64 * 2,
        Some(GgmlDType::Q8_0) => per_block(34, 32),
        Some(GgmlDType::Q4_0) => per_block(18, 32),
        Some(q) => per_block(q.type_size() as u64, q.block_size() as u64),
    }
}

/// What the pipeline will hold at `quant`, from the checkpoint headers on the
/// disk: every language-tower and transformer parameter at that quantisation,
/// the VAE's decoder in bf16. `None` until every shard is here.
pub(crate) fn weight_bytes(repo: &str, quant: Option<GgmlDType>, size: &dyn Fn(&str, &str) -> Option<u64>) -> Option<u64> {
    let params = |dir: &str, weights: &str, keep: &dyn Fn(&str) -> bool| -> Option<usize> {
        let index = read_json(&local_file(repo, &format!("{dir}/{weights}.safetensors.index.json"))?).ok()?;
        let mut shards: Vec<&str> = index["weight_map"].as_object()?.values().filter_map(Value::as_str).collect();
        shards.sort();
        shards.dedup();
        let paths: Vec<PathBuf> = shards.iter().map(|s| local_file(repo, &format!("{dir}/{s}"))).collect::<Option<_>>()?;
        // SAFETY: read-only cache files, headers only.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&paths).ok()? };
        Some(st.tensors().iter().filter(|(n, _)| keep(n)).map(|(_, v)| v.shape().iter().product::<usize>()).sum())
    };
    let text = params("text_encoder", "model", &|n| n.starts_with("model."))?;
    let dit = params("transformer", "diffusion_pytorch_model", &|_| true)?;
    // The VAE is a quarter of a gigabyte whole; charging all of it is simpler
    // than reading its header and wrong by a hundred megabytes.
    let vae = size(repo, "vae/diffusion_pytorch_model.safetensors")?;
    Some(weight_bytes_at(text + dit, quant) + vae)
}

impl Painter for QwenImage {
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let req = req.resolved(&self.defaults())?;
        let t0 = Instant::now();
        let cond = self.encode(&req.prompt)?;
        // "True" classifier-free guidance, and only with a negative prompt:
        // the reference pipeline does none without one, so neither does this.
        let guided = req.guidance > 1.0 && req.negative_prompt.is_some();
        let uncond = match (&req.negative_prompt, guided) {
            (Some(neg), true) => Some(self.encode(neg)?),
            _ => None,
        };
        self.device.synchronize()?;
        let encode_secs = t0.elapsed().as_secs_f64();

        // 8× from the VAE, 2× more from the patches.
        let (rows, cols) = (req.height / 16, req.width / 16);
        let sched = schedule::flow(&self.scheduler, req.steps, rows * cols)?;
        let patch = self.dit.cfg.in_channels;
        let mut x = noise(req.seed, &[1, rows * cols, patch], &self.device, DType::F32)?;

        let t1 = Instant::now();
        for i in 0..sched.steps() {
            let sigma = sched.timesteps[i];
            let xin = x.to_dtype(self.dtype)?;
            let v = self.dit.forward(&xin, &cond, sigma, rows, cols)?.to_dtype(DType::F32)?;
            let v = match &uncond {
                Some(u) => {
                    let vu = self.dit.forward(&xin, u, sigma, rows, cols)?.to_dtype(DType::F32)?;
                    let comb = (&vu + ((&v - &vu)? * req.guidance as f64)?)?;
                    // Rescaled to the conditional prediction's length, patch
                    // by patch, which keeps high guidance from overshooting.
                    let len = |t: &Tensor| t.sqr()?.sum_keepdim(D::Minus1)?.sqrt();
                    comb.broadcast_mul(&(len(&v)? / len(&comb)?.maximum(1e-12)?)?)?
                }
                None => v,
            };
            let preview = match req.preview {
                true => {
                    // Along the straight line to σ = 0: the image the model
                    // is heading for from here.
                    let clean = (&x - (&v * sigma)?)?;
                    Some(latent_preview(&unpack(&clean, rows, cols)?, &PREVIEW, PREVIEW_BIAS)?)
                }
                false => {
                    self.device.synchronize()?;
                    None
                }
            };
            x = (&x + (v * sched.dt(i))?)?;
            if !on_step(Step { done: i + 1, total: sched.steps(), preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        let worst = x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        if !worst.is_finite() || worst == 0.0 {
            return Err(format!("the denoiser's result is {worst} everywhere it is largest; not decoding it").into());
        }
        let t2 = Instant::now();
        let pixels = self.vae.decode(&unpack(&x, rows, cols)?.to_dtype(DType::BF16)?)?;
        let image = to_rgb8(&pixels)?;
        let decode_secs = t2.elapsed().as_secs_f64();
        Ok(Painted { image, request: req, encode_secs, denoise_secs, decode_secs })
    }

    fn defaults(&self) -> Defaults {
        // The reference's own: 1328² (its 1:1 size), 50 steps, and a true-CFG
        // scale of 4, which applies only when a negative prompt is given.
        Defaults { width: 1328, height: 1328, steps: 50, guidance: 4.0, multiple: 16, takes_guidance: true }
    }

    fn summary(&self) -> String {
        format!("Qwen-Image, {:.1} B parameters", self.params as f64 / 1e9)
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        self.bytes
    }

    fn backend(&self) -> String {
        crate::common::label(&self.device, self.dtype, self.quant)
    }
}



/// Packed latents `[1, rows·cols, 64]` back to `[1, 16, 2·rows, 2·cols]`.
///
/// Each packed row is one 2×2 patch, channel-major: element `c·4 + dy·2 + dx`.
fn unpack(x: &Tensor, rows: usize, cols: usize) -> candle_core::Result<Tensor> {
    x.reshape((1, rows, cols, Z, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((1, Z, rows * 2, cols * 2))
}

/// The Wan latent's sixteen channels (normalised, as the transformer sees
/// them) as colour, roughly: fitted by least squares the same way as SDXL's
/// (see `sdxl.rs`), from one 512² image — 20 steps, seed 5, "a red fox
/// sitting in fresh snow, photograph". It explains 96%, 97% and 98% of the
/// variance in R, G and B on that image; sixteen channels say more about
/// colour than SDXL's four.
const PREVIEW: [[f32; 3]; Z] = [
    [-0.3507, -0.1853, 0.0615],
    [-0.0511, -0.0464, -0.0369],
    [0.2778, 0.1791, 0.1158],
    [-0.1382, -0.0410, -0.0963],
    [-0.0271, -0.0600, -0.0847],
    [0.0714, -0.0568, -0.0989],
    [-0.1880, -0.2240, -0.1799],
    [0.0565, 0.1721, 0.2391],
    [-0.3462, -0.3725, -0.4518],
    [-0.0724, 0.1061, 0.2244],
    [0.0158, 0.1410, 0.1197],
    [0.0384, 0.0901, 0.1018],
    [-0.1201, 0.0422, 0.1407],
    [0.0609, 0.0515, 0.0671],
    [0.4820, 0.3590, 0.3738],
    [0.1945, 0.1514, 0.1610],
];
const PREVIEW_BIAS: [f32; 3] = [-0.1023, -0.1882, -0.2862];

#[cfg(test)]
mod tests {
    use super::*;

    /// `unpack` is the inverse of diffusers' `_pack_latents`, which views
    /// `[B, C, H, W]` as `[B, C, H/2, 2, W/2, 2]`, permutes to
    /// `(0, 2, 4, 1, 3, 5)` and flattens.
    #[test]
    fn unpacking_undoes_diffusers_packing() {
        let dev = Device::Cpu;
        let (h, w) = (4, 6);
        let grid = Tensor::arange(0f32, (Z * h * w) as f32, &dev).unwrap().reshape((1, Z, h, w)).unwrap();
        let packed = grid
            .reshape((1, Z, h / 2, 2, w / 2, 2))
            .unwrap()
            .permute((0, 2, 4, 1, 3, 5))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((1, (h / 2) * (w / 2), Z * 4))
            .unwrap();
        let back = unpack(&packed, h / 2, w / 2).unwrap();
        assert_eq!(back.flatten_all().unwrap().to_vec1::<f32>().unwrap(), grid.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }

    /// The positions diffusers' `QwenEmbedRope` gives, with `scale_rope`:
    /// rows and columns centred on the image, the frame at 0, and text on the
    /// diagonal starting at `max(rows, cols) / 2`.
    #[test]
    fn rope_positions_are_centred_on_the_image_and_text_follows_it() {
        let cfg = DitConfig { layers: 0, heads: 24, head_dim: 128, in_channels: 64, out_channels: 16, patch: 2, joint: 3584, axes: [16, 56, 56] };
        let (rows, cols, text) = (4, 6, 2);
        let a = angles(&cfg, rows, cols, text);
        let half = 64;
        let at = |token: usize, pair: usize| a[token * half + pair];
        // Pair 0 of each axis has frequency 1, so its angle is the position.
        let (frame, row, col) = (0, 8, 8 + 28);
        // First patch: row −2 (of −2..1), column −3 (of −3..2).
        assert_eq!((at(0, frame), at(0, row), at(0, col)), (0.0, -2.0, -3.0));
        // Last patch: row 1, column 2.
        let last = rows * cols - 1;
        assert_eq!((at(last, row), at(last, col)), (1.0, 2.0));
        // Text starts at max(4/2, 6/2) = 3 on every axis.
        let t0 = rows * cols;
        assert_eq!((at(t0, frame), at(t0, row), at(t0, col)), (3.0, 3.0, 3.0));
        assert_eq!(at(t0 + 1, row), 4.0);
    }
}
