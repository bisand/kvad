//! The VAE decoder: from a latent to pixels.
//!
//! Diffusion does not happen in pixel space. A 1024×1024 RGB image is three
//! million numbers, and a denoiser run fifty times over three million numbers
//! is too slow to be useful. So a separate autoencoder was trained first to
//! squeeze an image into a grid an eighth of its size on each side, with a
//! handful of channels — SDXL's is `[4, 128, 128]`, 48 times smaller — and to
//! expand it back without visible loss. The denoiser only ever sees that grid.
//!
//! This is the expanding half. It is a convolution stack from end to end:
//! resnets with group norm, one attention layer at the bottom, and a
//! nearest-neighbour upsample then a convolution at each level. Its only
//! weights of interest to a reader are that it is large in activations and
//! small in parameters — 50 M of them, against 2.6 B in the UNet — and that it
//! is the single most memory-hungry step of the pipeline, because it is the
//! only one that works at full resolution.
//!
//! The encoding half is never used to *make* an image, and is skipped by name.

use super::nn::{attention, to_grid, to_seq, Conv2d, Ctx, GroupNorm, Linear};
use crate::common::Reader;
use candle_core::Tensor;
use kvad::serde_json::Value;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone)]
pub(crate) struct VaeConfig {
    pub(crate) channels: Vec<usize>,
    pub(crate) layers_per_block: usize,
    pub(crate) latent: usize,
    pub(crate) groups: usize,
    /// What the latents were multiplied by to give them unit variance for the
    /// denoiser, and so what they are divided by here.
    pub(crate) scaling: f64,
}

impl VaeConfig {
    pub(crate) fn from_json(v: &Value) -> Res<Self> {
        let n = |k: &str| -> Res<usize> {
            v.get(k).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("VAE config has no `{k}`").into())
        };
        Ok(VaeConfig {
            channels: v
                .get("block_out_channels")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect())
                .ok_or("VAE config has no `block_out_channels`")?,
            layers_per_block: n("layers_per_block")?,
            latent: n("latent_channels")?,
            groups: n("norm_num_groups")?,
            scaling: v.get("scaling_factor").and_then(Value::as_f64).ok_or("VAE config has no `scaling_factor`")?,
        })
    }

    /// How much smaller the latent is than the image, per side.
    pub(crate) fn factor(&self) -> usize {
        1 << (self.channels.len() - 1)
    }
}

/// The decoder's eps is 1e-6 throughout; diffusers hard-codes it.
const EPS: f64 = 1e-6;

struct Resnet {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl Resnet {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, groups: usize, cin: usize, cout: usize) -> Res<Self> {
        Ok(Resnet {
            norm1: GroupNorm::load(cx, r, "norm1", cin, groups, EPS)?,
            conv1: Conv2d::load(cx, r, "conv1", (cin, cout, 3), 1)?,
            norm2: GroupNorm::load(cx, r, "norm2", cout, groups, EPS)?,
            conv2: Conv2d::load(cx, r, "conv2", (cout, cout, 3), 1)?,
            shortcut: match cin != cout {
                true => Some(Conv2d::load(cx, r, "conv_shortcut", (cin, cout, 1), 1)?),
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

/// Single-head self-attention over every latent position, at the bottom of
/// the decoder where there are fewest of them. Biases on everything, unlike
/// the UNet's.
struct Attn {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Attn {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let s = to_seq(&self.norm.forward(x)?)?;
        let a = attention(&self.q.forward(&s)?, &self.k.forward(&s)?, &self.v.forward(&s)?, 1)?;
        to_grid(&self.out.forward(&a)?, h, w)? + x
    }
}

pub(crate) struct Decoder {
    cfg: VaeConfig,
    post_quant: Conv2d,
    conv_in: Conv2d,
    mid: (Resnet, Attn, Resnet),
    up: Vec<(Vec<Resnet>, Option<Conv2d>)>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: VaeConfig) -> Res<Self> {
        r.skip_under("encoder.");
        r.skip_under("quant_conv.");
        let g = cfg.groups;
        let top = *cfg.channels.last().unwrap();
        let d = r.pp("decoder");
        let m = d.pp("mid_block");
        let a = m.pp("attentions.0");
        let mid = (
            Resnet::load(cx, &m.pp("resnets.0"), g, top, top)?,
            Attn {
                norm: GroupNorm::load(cx, &a, "group_norm", top, g, EPS)?,
                q: Linear::load(cx, &a, "to_q", top, top, true)?,
                k: Linear::load(cx, &a, "to_k", top, top, true)?,
                v: Linear::load(cx, &a, "to_v", top, top, true)?,
                out: Linear::load(cx, &a, "to_out.0", top, top, true)?,
            },
            Resnet::load(cx, &m.pp("resnets.1"), g, top, top)?,
        );

        let rev: Vec<usize> = cfg.channels.iter().rev().copied().collect();
        let mut up = Vec::with_capacity(rev.len());
        let mut prev = top;
        for (i, &cout) in rev.iter().enumerate() {
            let b = d.pp(format!("up_blocks.{i}"));
            let resnets = (0..=cfg.layers_per_block)
                .map(|j| Resnet::load(cx, &b.pp(format!("resnets.{j}")), g, if j == 0 { prev } else { cout }, cout))
                .collect::<Res<Vec<_>>>()?;
            let upsample = match i + 1 < rev.len() {
                true => Some(Conv2d::load(cx, &b, "upsamplers.0.conv", (cout, cout, 3), 1)?),
                false => None,
            };
            up.push((resnets, upsample));
            prev = cout;
        }
        let bottom = cfg.channels[0];
        Ok(Decoder {
            post_quant: Conv2d::load(cx, r, "post_quant_conv", (cfg.latent, cfg.latent, 1), 1)?,
            conv_in: Conv2d::load(cx, &d, "conv_in", (cfg.latent, top, 3), 1)?,
            norm_out: GroupNorm::load(cx, &d, "conv_norm_out", bottom, g, EPS)?,
            conv_out: Conv2d::load(cx, &d, "conv_out", (bottom, 3, 3), 1)?,
            mid,
            up,
            cfg,
        })
    }

    pub(crate) fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// `[1, latent, h, w]`, as the denoiser left it, to `[1, 3, H, W]` in
    /// `[−1, 1]`.
    pub(crate) fn decode(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        let z = (latent / self.cfg.scaling)?;
        let mut h = self.conv_in.forward(&self.post_quant.forward(&z)?)?;
        h = self.mid.0.forward(&h)?;
        h = self.mid.1.forward(&h)?;
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
        self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)
    }
}
