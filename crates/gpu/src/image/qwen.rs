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

use super::lora::{self, Adapters};
use super::mmdit::{norm_out, Double, Names, Shape};
use super::nn::{check_latent, latent_preview, noise, timestep_embedding, to_rgb8, Conv2d, Ctx, Linear};
use super::schedule;
use super::vae;
use super::{finish, finish_gguf, local_file, open, read_json};
use crate::common::{settle, Loader, Reader, Stored};
use crate::gguf::Gguf;
use crate::qcache::Vault;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, D};
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::{json, Value};
use kvad::pipeline::component;
use kvad::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub use kvad::pipeline::qwen::TOKENIZER_REPO;

/// The system prompt every prompt is wrapped in, and how many tokens of it to
/// throw away afterwards. Both from diffusers' `QwenImagePipeline`.
const TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
const DROP: usize = 34;

/// The dtype a LoRA's factors are kept and run in: the pipeline's where it
/// is half precision, and bf16 where the pipeline runs in f32 beside
/// quantised weights. Lightning's side path is 47 dB below a layer's answer
/// and bf16's rounding as far again below that, where f32 factors would
/// double what it holds (1.7 GB for Lightning) and miss the M5's matrix
/// units.
fn lora_dtype(pipeline: DType) -> DType {
    match pipeline {
        DType::F32 => DType::BF16,
        d => d,
    }
}

/// What a LoRA's names for the transformer's layers may carry before
/// diffusers' own, all of them the one part: diffusers' `transformer.`,
/// ComfyUI's `diffusion_model.`, kohya's `lora_unet_`, or nothing.
const TRANSFORMER_PREFIXES: [(&str, &str); 4] = [("transformer.", "transformer"), ("diffusion_model.", "transformer"), ("lora_unet_", "transformer"), ("", "transformer")];

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
        self.read(&self.embed(ids, device, dtype)?)
    }

    /// The tokens as the first layer takes them, `[1, L, width]`: each
    /// one's row of the table, for the positions are a rotation inside
    /// every layer.
    fn embed(&self, ids: &[u32], device: &Device, dtype: DType) -> Res<Tensor> {
        let ids_t = Tensor::new(ids, device)?;
        let x = match &self.embed {
            TextEmbed::Dense(t) => t.index_select(&ids_t, 0)?,
            TextEmbed::Quant(q) => q.embedding(&ids_t)?,
        };
        Ok(x.to_dtype(dtype)?.unsqueeze(0)?)
    }

    /// [`TextEncoder::forward`], from the embeddings [`TextEncoder::embed`]
    /// gives: what the model is as a function of them, which is what a
    /// gradient is taken through.
    fn read(&self, x: &Tensor) -> Res<Tensor> {
        let mut x = x.clone();
        let (l, device, dtype) = (x.dim(1)?, x.device().clone(), x.dtype());
        let (device, sums) = (&device, super::nn::wide(dtype));

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
        let mask = Tensor::from_vec(mask, (1, 1, l, l), device)?.to_dtype(sums)?;

        let group = self.heads / self.kv_heads;
        for layer in &self.layers {
            let h = crate::grad::rms_norm(&x, &layer.ln1, self.eps)?;
            let split = |t: Tensor, n: usize| -> candle_core::Result<Tensor> {
                t.reshape((1, l, n, self.head_dim))?.transpose(1, 2)?.contiguous()
            };
            let q = crate::grad::rope(&split(layer.q.forward(&h)?, self.heads)?, &cos, &sin)?;
            let k = crate::grad::rope(&split(layer.k.forward(&h)?, self.kv_heads)?, &cos, &sin)?;
            let v = split(layer.v.forward(&h)?, self.kv_heads)?;
            // Grouped-query attention, the way `model.rs` does it: fold the
            // query heads that share a KV head into one batch of rows. The
            // product is taken in f32, not rounded to bf16 and then widened:
            // a score here is large enough that bf16's three digits move
            // the softmax, and the gradient through it by most of its
            // length (`tests::the_text_encoder_has_a_whole_gradient`).
            let qg = q.reshape((1, self.kv_heads, group * l, self.head_dim))?;
            let att = (qg.to_dtype(sums)?.matmul(&k.transpose(2, 3)?.contiguous()?.to_dtype(sums)?)? / (self.head_dim as f64).sqrt())?;
            let att = att.reshape((1, self.heads, l, l))?.broadcast_add(&mask)?;
            let att = crate::grad::softmax_last_dim(&att)?.to_dtype(dtype)?.reshape((1, self.kv_heads, group * l, l))?;
            let a = att.matmul(&v)?.reshape((1, self.heads, l, self.head_dim))?;
            let a = a.transpose(1, 2)?.contiguous()?.reshape((1, l, self.heads * self.head_dim))?;
            x = (x + layer.o.forward(&a)?)?;

            let h = crate::grad::rms_norm(&x, &layer.ln2, self.eps)?;
            let g = (candle_nn::ops::silu(&layer.gate.forward(&h)?)? * layer.up.forward(&h)?)?;
            x = (x + layer.down.forward(&g)?)?;
        }
        Ok(crate::grad::rms_norm(&x, &self.norm, self.eps)?)
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
        let mut txt = self.txt_in.forward(&crate::grad::rms_norm(txt, &self.txt_norm, 1e-6)?)?;

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
    mid: WanMid,
    up: Vec<(Vec<WanResnet>, Option<Conv2d>)>,
    norm_out: RmsChannels,
    conv_out: Conv2d,
    mean: Tensor,
    std: Tensor,
}

impl WanDecoder {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, config: &Value) -> Res<Self> {
        let WanConfig { base, z, res, mult } = WanConfig::from_json(config)?;

        // The encoder makes latents from images, which text-to-image never
        // does; the temporal upsamplers only run from the second frame on.
        r.skip_under("encoder.");
        r.skip_under("quant_conv.");

        let d = r.pp("decoder");
        let top = base * mult[mult.len() - 1];
        let mid = WanMid::load(cx, &d.pp("mid_block"), top)?;

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
            mean: latent_stats(cx, config, "latents_mean", z)?,
            std: latent_stats(cx, config, "latents_std", z)?,
        })
    }

    /// `[1, 16, h, w]` normalised latents to `[1, 3, H, W]`, to be clamped
    /// to `[−1, 1]`. Not clamped here: `to_rgb8` does it on the host, after
    /// its check, and on the device a clamp is `x > y ? x : y` both ways,
    /// which turns NaN into −1, black, and hides it from the check.
    fn decode(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        self.decode_failing(z, None)
    }

    /// [`Self::decode`], with the stage numbered `fail` counted from
    /// `conv_in` (0) to `conv_out` (the last) left as zeros, as a Metal
    /// command buffer that failed leaves them, and the rest run on them.
    /// As `vae::Decoder::decode_failing` counts them for the other VAEs.
    fn decode_failing(&self, z: &Tensor, fail: Option<usize>) -> candle_core::Result<Tensor> {
        let mut stage = 0;
        let mut done = |h: Tensor| -> candle_core::Result<Tensor> {
            stage += 1;
            match fail == Some(stage - 1) {
                true => h.zeros_like(),
                false => Ok(h),
            }
        };
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let mut h = done(self.conv_in.forward(&self.post_quant.forward(&z)?)?)?;
        h = done(self.mid.first.forward(&h)?)?;
        h = done(self.mid.attend(&h)?)?;
        h = done(self.mid.second.forward(&h)?)?;
        for (resnets, upsample) in &self.up {
            for r in resnets {
                h = done(r.forward(&h)?)?;
            }
            if let Some(conv) = upsample {
                let (_, _, hh, ww) = h.dims4()?;
                h = done(conv.forward(&h.upsample_nearest2d(hh * 2, ww * 2)?)?)?;
            }
        }
        let h = done(self.norm_out.forward(&h)?.silu()?)?;
        done(self.conv_out.forward(&h)?)
    }

    /// How many stages [`Self::decode_failing`] counts.
    #[cfg(test)]
    fn stages(&self) -> usize {
        4 + self.up.iter().map(|(r, u)| r.len() + u.is_some() as usize).sum::<usize>() + 2
    }
}

/// What a Wan VAE's config says about its shape. Its `attn_scales` must be
/// empty: attention in the middle block only, which is all Qwen-Image's has.
struct WanConfig {
    base: usize,
    /// The latent's channels.
    z: usize,
    res: usize,
    mult: Vec<usize>,
}

impl WanConfig {
    fn from_json(config: &Value) -> Res<Self> {
        let n = |k: &str| config.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or(format!("VAE config has no `{k}`"));
        let mult: Vec<usize> = config["dim_mult"].as_array().ok_or("VAE config has no `dim_mult`")?.iter().filter_map(Value::as_u64).map(|v| v as usize).collect();
        if !config["attn_scales"].as_array().is_some_and(|a| a.is_empty()) {
            return Err("a Wan VAE with attention outside its middle block is not implemented".into());
        }
        Ok(WanConfig { base: n("base_dim")?, z: n("z_dim")?, res: n("num_res_blocks")?, mult })
    }
}

/// The per-channel `latents_mean` or `latents_std` as `[1, z, 1, 1]`: what
/// puts the latent into the transformer's units and back.
fn latent_stats(cx: &Ctx<'_>, config: &Value, k: &str, z: usize) -> Res<Tensor> {
    let v: Vec<f32> = config[k].as_array().ok_or(format!("VAE config has no `{k}`"))?.iter().filter_map(Value::as_f64).map(|f| f as f32).collect();
    Ok(Tensor::from_vec(v, (1, z, 1, 1), cx.device())?.to_dtype(cx.dtype)?)
}

/// The middle block, the same in the encoder and the decoder: a resnet, one
/// attention over every position, and a resnet.
struct WanMid {
    first: WanResnet,
    norm: RmsChannels,
    qkv: Conv2d,
    proj: Conv2d,
    second: WanResnet,
}

impl WanMid {
    fn load(cx: &Ctx<'_>, m: &Reader<'_>, top: usize) -> Res<Self> {
        let a = m.pp("attentions.0");
        Ok(WanMid {
            first: WanResnet::load(cx, &m.pp("resnets.0"), top, top)?,
            norm: RmsChannels::load(cx, &a, "norm", top, true)?,
            qkv: causal_2d(cx, &a, "to_qkv", (top, 3 * top))?,
            proj: causal_2d(cx, &a, "proj", (top, top))?,
            second: WanResnet::load(cx, &m.pp("resnets.1"), top, top)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.second.forward(&self.attend(&self.first.forward(x)?)?)
    }

    /// One head as wide as the channels, over every position.
    fn attend(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        let s = self.qkv.forward(&self.norm.forward(x)?)?.reshape((1, 3 * c, h * w))?.transpose(1, 2)?.contiguous()?;
        let part = |i: usize| s.narrow(2, i * c, c)?.unsqueeze(1)?.contiguous();
        let a = super::nn::written_out(&part(0)?, &part(1)?, &part(2)?, 1.0 / (c as f64).sqrt())?;
        let a = a.squeeze(1)?.transpose(1, 2)?.contiguous()?.reshape((1, c, h, w))?;
        self.proj.forward(&a)? + x
    }
}

/// The encoding half, run on one frame: a picture to the Gaussian over its
/// latents, as [`vae::Encoder`] makes it for the other VAEs.
///
/// It is the decoder backwards, as theirs is: resnets, then a stride-2
/// convolution after every level but the last, the middle block, and twice
/// the latent's channels out, the mean's and the log-variance's. Two things
/// are Wan's own. Every 3D convolution collapses to its last time slice, as
/// in the decoder ([`causal`]). And two of the downsamplers also halve time,
/// with a convolution over frames that a video's first frame never reaches:
/// it runs on a frame and the one before it, and the first has none. So
/// those are skipped, and a picture goes through exactly what a video's
/// first frame does.
///
/// Peak footprint of one encode and one decode alone, in bf16 as the
/// pipeline runs the VAE, on an M5 Pro:
///
/// | | encode | decode |
/// |---|---|---|
/// | 512² | 0.8 s, 1.8 GB | 1.3 s, 2.7 GB |
/// | 1328² | 5.8 s, 14.5 GB | 9.6 s, 19.6 GB |
///
/// So, as for the other VAEs, a picture that can be decoded can be encoded.
/// `tests::one_encode` measures it.
///
/// Nothing outside the tests calls it yet; its callers are training (#75)
/// and editing (#43), as for [`vae::Encoder`].
#[cfg_attr(not(test), allow(dead_code))]
struct WanEncoder {
    conv_in: Conv2d,
    /// Flat, as the weights number them: each level's resnets, then its
    /// downsampler.
    down: Vec<(Vec<WanResnet>, Option<Conv2d>)>,
    mid: WanMid,
    norm_out: RmsChannels,
    conv_out: Conv2d,
    quant: Conv2d,
    z: usize,
    mean: Tensor,
    std: Tensor,
}

#[cfg_attr(not(test), allow(dead_code))]
impl WanEncoder {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, config: &Value) -> Res<Self> {
        let WanConfig { base, z, res, mult } = WanConfig::from_json(config)?;
        r.skip_under("decoder.");
        r.skip_under("post_quant_conv.");

        let e = r.pp("encoder");
        let dims: Vec<usize> = std::iter::once(1).chain(mult.iter().copied()).map(|u| base * u).collect();
        let mut down = Vec::new();
        let mut at = 0;
        for i in 0..mult.len() {
            let resnets = (0..res)
                .map(|j| {
                    let b = e.pp(format!("down_blocks.{}", at + j));
                    WanResnet::load(cx, &b, if j == 0 { dims[i] } else { dims[i + 1] }, dims[i + 1])
                })
                .collect::<Res<Vec<_>>>()?;
            at += res;
            let downsample = match i + 1 < mult.len() {
                true => {
                    let b = e.pp(format!("down_blocks.{at}"));
                    // Only from the second frame on; see above.
                    b.skip_under("time_conv");
                    at += 1;
                    Some(Conv2d::load(cx, &b, "resample.1", (dims[i + 1], dims[i + 1], 3), 2)?.unpadded())
                }
                false => None,
            };
            down.push((resnets, downsample));
        }
        let top = dims[dims.len() - 1];
        Ok(WanEncoder {
            conv_in: causal(cx, &e, "conv_in", (3, base, 3, 3))?,
            down,
            mid: WanMid::load(cx, &e.pp("mid_block"), top)?,
            norm_out: RmsChannels::load(cx, &e, "norm_out", top, false)?,
            conv_out: causal(cx, &e, "conv_out", (top, 2 * z, 3, 3))?,
            quant: causal(cx, r, "quant_conv", (2 * z, 2 * z, 1, 1))?,
            z,
            mean: latent_stats(cx, config, "latents_mean", z)?,
            std: latent_stats(cx, config, "latents_std", z)?,
        })
    }

    /// `[1, 3, H, W]` in `[−1, 1]`, `H` and `W` multiples of 8, to the
    /// Gaussian over its `[1, 16, H/8, W/8]` latents, in the VAE's units.
    fn encode(&self, image: &Tensor) -> candle_core::Result<vae::Posterior> {
        let mut h = self.conv_in.forward(image)?;
        for (resnets, downsample) in &self.down {
            for r in resnets {
                h = r.forward(&h)?;
            }
            if let Some(conv) = downsample {
                // Right and bottom, then a stride of 2 unpadded, as in
                // `vae::Encoder::encode`.
                h = conv.forward(&h.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?)?;
            }
        }
        let h = self.mid.forward(&h)?;
        let h = self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)?;
        let h = self.quant.forward(&h)?;
        Ok(vae::Posterior { mean: h.narrow(1, 0, self.z)?, logvar: h.narrow(1, self.z, self.z)?.clamp(-30.0, 20.0)? })
    }

    /// A latent in the VAE's units to the transformer's: less the
    /// per-channel mean, over the per-channel spread, as the edit pipelines
    /// do. The inverse of the first thing [`WanDecoder::decode`] does.
    fn to_denoiser(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        z.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
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
    /// The GGUF the transformer came from, and what it is made of.
    gguf: Option<(String, String)>,
    /// The transformer's linear layers, for LoRAs ([`super::lora`]).
    adapters: Adapters,
    /// A fixed shift in place of the schedule's own; see
    /// [`QwenImage::set_shift`].
    shift: Option<f64>,
    params: usize,
    bytes: usize,
}

impl QwenImage {
    pub fn load(
        repo: &str,
        quant: Option<GgmlDType>,
        device: Device,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Res<Self> {
        Self::load_with(repo, None, quant, device, progress, watch)
    }

    /// [`QwenImage::load`], with the transformer read from `gguf` in place
    /// of `repo`'s, in the blocks its maker chose; everything else is
    /// `repo`'s, the text encoder at `quant`.
    pub fn load_with(
        repo: &str,
        gguf: Option<&Path>,
        quant: Option<GgmlDType>,
        device: Device,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Res<Self> {
        // Quantised weights take f32 activations; dense ones run in the
        // checkpoint's own bf16. A GGUF's are quantised, so its pipeline is
        // f32 throughout, and a text encoder read whole beside it would be
        // f32 too: 28 GB. It is quantised instead.
        let quant = match (gguf, quant) {
            (Some(_), None) => {
                progress("the text encoder runs at q8 beside a GGUF's transformer");
                Some(GgmlDType::Q8_0)
            }
            _ => quant,
        };
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
        bytes += weight_bytes_at(params, quant) as usize;

        // Every linear layer of the transformer may take a LoRA. Their names
        // are diffusers', under `transformer.` in a diffusers LoRA,
        // `diffusion_model.` in ComfyUI's, and kohya's `lora_unet_`.
        let adapters = Adapters::new(&TRANSFORMER_PREFIXES);
        let (dit, made) = match gguf {
            Some(path) => {
                let config = read_json(&fetch_file(repo, "transformer/config.json", watch)?)?;
                let dcfg = DitConfig::from_json(&config)?;
                let file = Arc::new(Gguf::open(path)?);
                match file.text("general.architecture") {
                    Some("qwen_image") => {}
                    other => return Err(format!("{} is a GGUF of {}, not of Qwen-Image's transformer", path.display(), other.unwrap_or("an unnamed architecture")).into()),
                }
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                progress(&format!("loading the transformer from {name}: {}", file.make_up()));
                let vault = Vault::off();
                let cx = Ctx { ld: Loader::new(None, device.clone(), &vault).accelerated(), dtype };
                let r = Reader::gguf(Arc::clone(&file), dtype).with_adapters(adapters.part("transformer"));
                let dit = Dit::load(&cx, &r, dcfg)?;
                params += finish_gguf("transformer", &file, &r)?;
                bytes += file.device_bytes(dtype);
                (dit, Some((name, file.make_up())))
            }
            None => {
                let (config, paths) = component(repo, "transformer", "diffusion_pytorch_model", watch)?;
                progress(&format!("loading the transformer at {label} — twenty billion parameters, which takes a while the first time"));
                let dcfg = DitConfig::from_json(&config)?;
                let shape = json!({ "component": "transformer", "layers": dcfg.layers, "width": dcfg.width() });
                let mut vault = Vault::open_as(&format!("{repo}/transformer"), &paths, shape, quant, progress);
                let dit = {
                    let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
                    let r = open(&paths, dtype)?.with_adapters(adapters.part("transformer"));
                    let dit = Dit::load(&cx, &r, dcfg)?;
                    let n = finish("transformer", &paths, &r)?;
                    params += n;
                    bytes += weight_bytes_at(n, quant) as usize;
                    dit
                };
                vault.finish(progress);
                (dit, None)
            }
        };

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
        bytes += vae_params * 2;

        settle(&device)?;
        let from = made.as_ref().map(|(name, _)| format!(", the transformer from {name}")).unwrap_or_default();
        progress(&format!("loaded Qwen-Image: {:.1} B parameters at {label}{from}", params as f64 / 1e9));
        Ok(QwenImage { tok, text, dit, vae, scheduler, device, dtype, quant, gguf: made, adapters, shift: None, params, bytes })
    }

    /// Run a fixed shift, the same at every size and with no stretch at the
    /// end, in place of the one the repo's schedule works out from the
    /// image's size; `None` for the repo's own again. Lightning was
    /// distilled on a shift of 3 (`docs/lora-plan.md`).
    pub fn set_shift(&mut self, shift: Option<f64>) {
        self.shift = shift;
    }

    /// The scheduler's config, with [`QwenImage::set_shift`]'s shift in it:
    /// diffusers' own settings for a fixed one, the dynamic shift's two ends
    /// both at `ln shift`, and no terminal stretch, as Lightning's card sets
    /// them.
    fn schedule_config(&self) -> Value {
        let mut c = self.scheduler.clone();
        if let (Some(shift), Some(o)) = (self.shift, c.as_object_mut()) {
            o.insert("base_shift".into(), json!(shift.ln()));
            o.insert("max_shift".into(), json!(shift.ln()));
            o.insert("shift_terminal".into(), Value::Null);
        }
        c
    }

    /// Apply `loras` to the transformer, each at its strength, in place of
    /// any set before, until the next call; none, to take them off. The
    /// layers adapted.
    pub fn set_loras(&mut self, loras: &[(&lora::File, f64)]) -> Res<usize> {
        match loras.is_empty() {
            true => {
                self.adapters.clear();
                Ok(0)
            }
            false => self.adapters.set(loras, &self.device, lora_dtype(self.dtype)),
        }
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
///
/// With `gguf`, the transformer is that file's, as its blocks and its plain
/// tensors widened to f32, and the language tower is q8 if nothing else is
/// asked, as [`QwenImage::load_with`] loads them.
pub(crate) fn weight_bytes(repo: &str, quant: Option<GgmlDType>, size: &dyn Fn(&str, &str) -> Option<u64>, gguf: Option<&Path>) -> Option<u64> {
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
    // The VAE is a quarter of a gigabyte whole; charging all of it is simpler
    // than reading its header and wrong by a hundred megabytes.
    let vae = size(repo, "vae/diffusion_pytorch_model.safetensors")?;
    match gguf {
        Some(file) => {
            let quant = quant.or(Some(GgmlDType::Q8_0));
            let dit = Gguf::open(file).ok()?.device_bytes(DType::F32) as u64;
            Some(weight_bytes_at(text, quant) + dit + vae)
        }
        None => {
            let dit = params("transformer", "diffusion_pytorch_model", &|_| true)?;
            Some(weight_bytes_at(text + dit, quant) + vae)
        }
    }
}

impl QwenImage {
    /// One image, with whatever LoRAs the adapters hold.
    fn draw(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
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
        settle(&self.device)?;
        let encode_secs = t0.elapsed().as_secs_f64();

        // 8× from the VAE, 2× more from the patches.
        let (rows, cols) = (req.height / 16, req.width / 16);
        let sched = schedule::flow(&self.schedule_config(), req.steps, rows * cols)?;
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
                    settle(&self.device)?;
                    None
                }
            };
            x = (&x + (v * sched.dt(i))?)?;
            if !on_step(Step { done: i + 1, total: sched.steps(), preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        check_latent(&x)?;
        let t2 = Instant::now();
        let pixels = self.vae.decode(&unpack(&x, rows, cols)?.to_dtype(DType::BF16)?)?;
        let image = to_rgb8(&pixels)?;
        let decode_secs = t2.elapsed().as_secs_f64();
        Ok(Painted { image, request: req, encode_secs, denoise_secs, decode_secs })
    }
}

impl Painter for QwenImage {
    /// With the request's LoRAs set for it and taken off after.
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let (adapters, device, dtype) = (self.adapters.clone(), self.device.clone(), lora_dtype(self.dtype));
        lora::painting(&adapters, req, &device, dtype, || self.draw(req, on_step))
    }

    fn defaults(&self) -> Defaults {
        // The reference's own: 1328² (its 1:1 size), 50 steps, and a true-CFG
        // scale of 4, which applies only when a negative prompt is given.
        Defaults { width: 1328, height: 1328, steps: 50, guidance: 4.0, multiple: 16, takes_guidance: true, takes_negative: true, takes_loras: true, edits: false }
    }

    fn summary(&self) -> String {
        let from = self.gguf.as_ref().map(|(name, _)| format!(", the transformer from {name}")).unwrap_or_default();
        format!("Qwen-Image, {:.1} B parameters{from}", self.params as f64 / 1e9)
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        self.bytes
    }

    fn backend(&self) -> String {
        let label = crate::common::label(&self.device, self.dtype, self.quant);
        match &self.gguf {
            Some((_, made)) => format!("{label}, transformer {made}"),
            None => label,
        }
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

    /// A decode with any one of the Wan VAE's stages left as zeros, as a
    /// failed Metal command buffer leaves it, and the rest run on them, is
    /// refused rather than saved; the decode that did not fail is not; and
    /// NaN in the decode reaches the check rather than turning black. In
    /// bf16 on Metal as the pipeline runs it, at 512², from a latent of
    /// noise, as `sd15::tests::a_decode_that_fails_part_way_is_refused`
    /// does the other VAE:
    ///
    ///     cargo test --release -p kvad-gpu qwen::tests::a_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_decode_that_fails_part_way_is_refused() {
        let w = Watcher::none();
        let config = read_json(&fetch_file("Qwen/Qwen-Image", "vae/config.json", &w).unwrap()).unwrap();
        let paths = vec![fetch_file("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors", &w).unwrap()];
        let device = Device::new_metal(0).unwrap();
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype: DType::BF16 };
        let vae = WanDecoder::load(&cx, &open(&paths, DType::BF16).unwrap(), &config).unwrap();
        let z = noise(5, &[1, Z, 64, 64], &device, DType::BF16).unwrap();

        to_rgb8(&vae.decode(&z).unwrap()).unwrap();
        for stage in 0..vae.stages() {
            let e = to_rgb8(&vae.decode_failing(&z, Some(stage)).unwrap()).err().map(|e| e.to_string());
            eprintln!("stage {stage}: {}", e.as_deref().unwrap_or("passed"));
            assert!(e.is_some(), "stage {stage} failed and was not refused");
        }

        // One NaN in the latent spreads through the first 3×3 convolution
        // and the RMS norms after it; the device's clamp used to make it
        // black, and what `to_rgb8` saw was a finite image.
        let mut v = z.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        v[32 * 64 + 32] = f32::NAN;
        let z = Tensor::from_vec(v, (1, Z, 64, 64), &device).unwrap().to_dtype(DType::BF16).unwrap();
        let e = to_rgb8(&vae.decode(&z).unwrap()).err().map(|e| e.to_string()).unwrap_or_default();
        eprintln!("NaN: {e}");
        assert!(e.contains("NaN"), "NaN in the decode was not refused as NaN: {e}");
    }

    /// Qwen-Image's first blocks with Lightning, against diffusers with PEFT
    /// (`scripts/lora-fixtures.py`): without it, with it and at half its
    /// strength, and the LoRA's own part, the adapted output less the plain
    /// one, which after two blocks is a small part of the whole. On the CPU
    /// in f32, and on Metal at q8, as Kvad runs Qwen-Image:
    ///
    ///     KVAD_LORA_FIXTURES=/tmp/lora-fx cargo test --release -p kvad-gpu a_lora_agrees -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_lora_agrees_with_peft() {
        let dir = std::env::var("KVAD_LORA_FIXTURES").expect("KVAD_LORA_FIXTURES, from scripts/lora-fixtures.py");
        let fx = candle_core::safetensors::load(format!("{dir}/qwen_lora.safetensors"), &Device::Cpu).unwrap();
        let blocks = 2;
        let (config, paths) = component("Qwen/Qwen-Image", "transformer", "diffusion_pytorch_model", &Watcher::none()).unwrap();
        let path = crate::image::local_file("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors").expect("fetch Lightning first");
        let in_blocks = |m: &str| m.split('.').nth(1).and_then(|n| n.parse::<usize>().ok()).is_some_and(|n| n < blocks);
        let file = lora::File::open(&path).unwrap().only(in_blocks);
        let db = |want: &Tensor, got: &Tensor| -> f64 {
            let e = (want - got).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
            10.0 * (want.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64 / e).log10()
        };
        let sigma = fx["sigma"].to_vec1::<f32>().unwrap()[0] as f64;
        let (plain_ref, adapted_ref, half_ref) = (&fx["plain"], &fx["adapted"], &fx["half"]);
        let part_ref = (adapted_ref - plain_ref).unwrap();
        // The CPU in f32 throughout, to check the arithmetic; Metal as a
        // q8 pipeline runs it, the factors in bf16.
        let mut runs = vec![(Device::Cpu, None, DType::F32, "the CPU, f32")];
        if let Ok(metal) = Device::new_metal(0) {
            runs.push((metal, Some(GgmlDType::Q8_0), lora_dtype(DType::F32), "Metal, q8, bf16 factors"));
        }
        for (dev, quant, factors, what) in runs {
            let vault = Vault::off();
            let cx = Ctx { ld: Loader::new(quant, dev.clone(), &vault).accelerated(), dtype: DType::F32 };
            let mut cfg = DitConfig::from_json(&config).unwrap();
            cfg.layers = blocks;
            let adapters = Adapters::new(&TRANSFORMER_PREFIXES);
            let r = open(&paths, DType::F32).unwrap().with_adapters(adapters.part("transformer"));
            let dit = Dit::load(&cx, &r, cfg).unwrap();
            let run = || dit.forward(&fx["x"].to_device(&dev).unwrap(), &fx["txt"].to_device(&dev).unwrap(), sigma, 8, 8).unwrap().to_device(&Device::Cpu).unwrap();
            let plain = run();
            assert_eq!(adapters.set(&[(&file, 1.0)], &dev, factors).unwrap(), file.layers());
            let adapted = run();
            adapters.set(&[(&file, 0.5)], &dev, factors).unwrap();
            let half = run();
            let part = (&adapted - &plain).unwrap();
            let (p, a, h, d) = (db(plain_ref, &plain), db(adapted_ref, &adapted), db(half_ref, &half), db(&part_ref, &part));
            // The part is the difference of two outputs, each `p` dB from
            // the reference's, and it is `size` dB below them: so it can be
            // no nearer than about `p - size - 3` dB, the two outputs' own
            // rounding, however exact the LoRA.
            let size = db(plain_ref, adapted_ref);
            eprintln!("{what}: plain {p:.1} dB, adapted {a:.1}, half {h:.1}; the LoRA's own part, {size:.1} dB below the output, {d:.1} dB, over {} layers", file.layers());
            let floor = if quant.is_some() { 25.0 } else { 80.0 };
            assert!(a > floor && h > floor, "{what}");
            assert!(d > p - size - 6.0, "{what}: the LoRA's part is further from the reference's than the model's own rounding explains");
        }
    }

    /// The VAE's config and weights, and an encoder and decoder of them on
    /// `device` in `dtype`.
    fn wan(device: &Device, dtype: DType) -> (WanEncoder, WanDecoder) {
        let w = Watcher::none();
        let config = read_json(&fetch_file("Qwen/Qwen-Image", "vae/config.json", &w).unwrap()).unwrap();
        let paths = vec![fetch_file("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors", &w).unwrap()];
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let enc = WanEncoder::load(&cx, &open(&paths, dtype).unwrap(), &config).unwrap();
        let dec = WanDecoder::load(&cx, &open(&paths, dtype).unwrap(), &config).unwrap();
        (enc, dec)
    }

    /// kvad's Wan encoder against diffusers' `AutoencoderKLQwenImage`, on one
    /// frame: the mean, the log-variance and the round trip, in f32 and then
    /// in bf16 as the pipeline runs it, held to what
    /// `vae::tests::the_encoder_agrees_with_diffusers` holds the others to.
    /// The fixtures are `scripts/vae-fixtures.py --only qwen`'s.
    ///
    ///     KVAD_VAE_FIXTURES=/tmp/vae-fx cargo test --release -p kvad-gpu image::qwen::tests::the_encoder -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_encoder_agrees_with_diffusers() {
        use super::super::vae::tests::{gap, psnr};
        let dir = std::env::var("KVAD_VAE_FIXTURES").expect("KVAD_VAE_FIXTURES names the fixtures' directory");
        let fx = candle_core::safetensors::load(format!("{dir}/qwen.safetensors"), &Device::Cpu).expect("no qwen.safetensors among the fixtures");
        let (image, mean, logvar, decoded) = (&fx["image"], &fx["mean"], &fx["logvar"], &fx["decoded"]);
        let theirs = psnr(decoded, image);
        let device = Device::new_metal(0).unwrap();
        for dtype in [DType::F32, DType::BF16] {
            let (enc, dec) = wan(&device, dtype);
            let x = image.unsqueeze(0).unwrap().to_device(&device).unwrap().to_dtype(dtype).unwrap();
            let p = enc.encode(&x).unwrap();
            let back = dec.decode(&enc.to_denoiser(&p.mean).unwrap()).unwrap();
            let back = back.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().squeeze(0).unwrap();
            let (dm, dl, same, ours) = (gap(&p.mean, mean), gap(&p.logvar, logvar), psnr(&back, decoded), psnr(&back, image));
            let own = match dtype {
                DType::F32 => None,
                _ => Some(gap(&fx.get("mean_half").expect("no `mean_half` in the fixtures; make them on a Mac").unsqueeze(0).unwrap(), mean)),
            };
            if let Some(own) = own {
                eprintln!("qwen diffusers' own {dtype:?} mean is off by {:.2e} at most, {:.2e} rms", own.0, own.1);
            }
            eprintln!(
                "qwen {dtype:?}: mean off by {:.2e} at most, {:.2e} rms; logvar {:.2e}, {:.2e}; decoded {same:.1} dB from theirs; round trip {ours:.2} dB, diffusers' {theirs:.2}",
                dm.0, dm.1, dl.0, dl.1
            );
            let (close, least, db) = match own {
                None => (1e-3, 60.0, 0.01),
                Some(own) => (1.5 * own.1, 30.0, 0.5),
            };
            assert!(dm.1 < close, "qwen {dtype:?}: the mean is {:.2e} rms from diffusers'", dm.1);
            if own.is_none() {
                assert!(dl.1 < close, "qwen: the log-variance is {:.2e} rms from diffusers'", dl.1);
            }
            assert!(same > least, "qwen {dtype:?}: the decode of the mean is {same:.1} dB from diffusers'");
            assert!((ours - theirs).abs() < db, "qwen {dtype:?}: {ours:.2} dB against diffusers' {theirs:.2}");
        }
    }

    /// One encode of noise at `KVAD_VAE_SIDE` pixels square (1328, the
    /// pipeline's own size, by default) in bf16, timed after a warm-up, or
    /// one decode with `KVAD_VAE_DECODE` set; as `vae::tests::one_encode`.
    ///
    ///     KVAD_VAE_SIDE=1328 /usr/bin/time -l target/release/deps/kvad_gpu-… image::qwen::tests::one_encode --ignored --nocapture
    #[test]
    #[ignore]
    fn one_encode() {
        let side: usize = std::env::var("KVAD_VAE_SIDE").map(|s| s.parse().unwrap()).unwrap_or(1328);
        let decode = std::env::var("KVAD_VAE_DECODE").is_ok();
        let device = Device::new_metal(0).unwrap();
        let (enc, dec) = wan(&device, DType::BF16);
        let (enc, dec) = match decode {
            true => (None, Some(dec)),
            false => (Some(enc), None),
        };
        let x = Tensor::randn(0f32, 0.5, (1, 3, side, side), &device).unwrap().clamp(-1.0, 1.0).unwrap().to_dtype(DType::BF16).unwrap();
        let z = Tensor::randn(0f32, 1.0, (1, Z, side / 8, side / 8), &device).unwrap().to_dtype(DType::BF16).unwrap();
        let run = || {
            let out = match (&enc, &dec) {
                (Some(enc), _) => enc.encode(&x).unwrap().mean,
                (_, Some(dec)) => dec.decode(&z).unwrap(),
                _ => unreachable!(),
            };
            device.synchronize().unwrap();
            out.to_dtype(DType::F32).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };
        run();
        let started = Instant::now();
        let largest = run();
        let what = if decode { "decode" } else { "encode" };
        eprintln!("qwen BF16 {side}²: one {what} {:.2} s, largest |out| {largest:.2}", started.elapsed().as_secs_f64());
        assert!(largest.is_finite() && largest > 0.0, "the {what} came back {largest}");
    }

    /// The first two layers of Qwen2.5-VL's language tower and its final
    /// norm, with their real weights, as a function of the embeddings they
    /// are given and of a LoRA on two of their layers: the gradient
    /// `backward` finds is the function's own. Two layers of twenty-eight,
    /// as `flux::tests::t5_has_a_whole_gradient` takes two of T5's, and
    /// run the same ways for the same reasons. What is here and not there:
    /// the rotation, grouped queries and the causal mask; and the
    /// embeddings are taken along four directions, for it was one of them
    /// that showed what half precision did to the scores.
    ///
    ///     cargo test --release -p kvad-gpu qwen::tests::the_text_encoder_has -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_text_encoder_has_a_whole_gradient() {
        crate::cap::at(24.0);
        use candle_core::Var;
        // Of this machine or not at all: the files are 16 GB, and a check
        // is not what should fetch them. Two layers want the first shard,
        // and the final norm is in the last.
        let here = |f: &str| super::super::local_file("Qwen/Qwen-Image", &format!("text_encoder/{f}")).unwrap_or_else(|| panic!("Qwen/Qwen-Image's text_encoder/{f} is not on this machine"));
        let mut config = read_json(&here("config.json")).unwrap();
        let paths = vec![here("model-00001-of-00004.safetensors"), here("model-00004-of-00004.safetensors")];
        let c = config.get("text_config").cloned().unwrap_or_else(|| config.clone());
        let n = |k: &str| c[k].as_u64().unwrap() as usize;
        let (width, inter, kv) = (n("hidden_size"), n("intermediate_size"), n("hidden_size") / n("num_attention_heads") * n("num_key_value_heads"));
        match config.get_mut("text_config") {
            Some(t) => t["num_hidden_layers"] = json!(2),
            None => config["num_hidden_layers"] = json!(2),
        }
        let mut runs = vec![(Device::Cpu, None, DType::F64, 1e-5, 0.0, "the CPU, f64"), (Device::Cpu, None, DType::F32, 1e-2, 1e-3, "the CPU, f32")];
        if let Ok(metal) = Device::new_metal(0) {
            runs.push((metal.clone(), None, DType::F32, 1e-2, 1e-3, "Metal, f32"));
            runs.push((metal.clone(), Some(GgmlDType::Q8_0), DType::F32, 1e-2, 1e-1, "Metal, q8 weights"));
            // With the scores' product taken in bf16 this run's slopes were
            // 10–88% of a typical slope from f64's; in f32 they are 5–9%.
            runs.push((metal, None, DType::BF16, 1e-2, 0.2, "Metal, bf16"));
        }
        // Twelve tokens of ordinary text.
        let ids: [u32; 12] = [64, 2518, 38835, 11699, 304, 7722, 11794, 1790, 311, 264, 22360, 1841];
        let layers = [("model.layers.0.self_attn.k_proj", width, kv), ("model.layers.1.mlp.gate_proj", width, inter)];
        let mut exact = crate::grad::real::Exact::default();
        // The embeddings the f64 run read, for every run after it.
        let mut read: Option<Tensor> = None;
        for (dev, quant, dtype, step, tolerance, what) in runs {
            let mut run = exact.run(what, tolerance);
            let vault = Vault::off();
            let cx = Ctx { ld: Loader::new(quant, dev.clone(), &vault), dtype };
            let adapters = Adapters::new(&TRANSFORMER_PREFIXES);
            let r = open(&paths, dtype).unwrap().with_adapters(adapters.part("text_encoder"));
            let text = TextEncoder::load(&cx, &r, &config).unwrap();

            let seed = std::cell::Cell::new(74u64);
            let randn = |shape: &[usize], std: f32| {
                seed.set(seed.get() + 1);
                (noise(seed.get(), shape, &dev, DType::F32).unwrap() * std as f64).unwrap().to_dtype(dtype).unwrap()
            };
            let x = match &read {
                Some(x) => x.to_dtype(dtype).unwrap().to_device(&dev).unwrap(),
                None => text.embed(&ids, &dev, dtype).unwrap(),
            };
            read.get_or_insert_with(|| x.clone());
            let scale = x.to_dtype(DType::F32).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
            let weigh = randn(&[1, ids.len(), width], 1.0);
            let loss = |x: &Tensor| -> candle_core::Result<Tensor> {
                let h = text.read(x).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                (h * &weigh)?.sum_all()
            };

            const SCALE: f32 = 0.05;
            let factors: Vec<(Var, Var)> = layers
                .iter()
                .map(|&(name, inp, out)| {
                    let (a, b) = (Var::from_tensor(&randn(&[inp, 4], SCALE)).unwrap(), Var::from_tensor(&randn(&[4, out], SCALE)).unwrap());
                    adapters.place("text_encoder", name, a.as_tensor(), b.as_tensor()).unwrap();
                    (a, b)
                })
                .collect();
            for seed in [11, 21, 31, 41] {
                run.take("the embeddings", crate::grad::directional(&loss, &x, step * scale, seed).unwrap());
            }
            for (&(name, ..), (a, b)) in layers.iter().zip(&factors) {
                run.factors(&adapters, "text_encoder", name, (a, b), &|| loss(&x), step * SCALE as f64);
            }
        }
    }
}
