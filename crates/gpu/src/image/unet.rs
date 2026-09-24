//! SDXL's denoiser: a UNet with attention in its lower levels.
//!
//! A UNet is an hourglass with bridges. The way down halves the grid and
//! widens the channels at each level, so that by the bottom every position
//! summarises a large patch of the image; the way back up doubles the grid
//! again. What makes it a UNet is the bridges: every intermediate result on
//! the way down is kept and handed to its mirror on the way up, concatenated
//! onto the channels. The bottom knows *what* is in the picture, the bridges
//! remember *where*, and the up path needs both.
//!
//! Where it differs from the UNet of the 2015 paper is that the levels below
//! the top also *attend*. A transformer block runs over the grid flattened to
//! a sequence: self-attention, so a patch can see across the whole image, and
//! cross-attention to the prompt, which is the only way the text gets in. SDXL
//! spends its parameters at the bottom — ten transformer blocks per stage at
//! 1280 channels — where the grid is small and attention is cheap.
//!
//! The noise level gets in a third way: a vector made from the timestep (and,
//! in SDXL, from the pooled prompt and the image size) is added to every
//! resnet's activations, channel by channel.
//!
//! The shapes are the ones in `docs/image-plan.md`, read off the config.

use super::nn::{attention, timestep_embedding, to_grid, to_seq, Conv2d, Ctx, GroupNorm, LayerNorm, Linear};
use crate::common::Reader;
use candle_core::{Tensor, D};
use kvad::serde_json::Value;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What this UNet needs from `unet/config.json`.
#[derive(Debug, Clone)]
pub(crate) struct UnetConfig {
    pub(crate) channels: Vec<usize>,
    pub(crate) layers_per_block: usize,
    /// Transformer blocks per attention stage, per level; 0 for a level
    /// without attention.
    pub(crate) depth: Vec<usize>,
    /// Attention heads per level. The config calls this
    /// `attention_head_dim`, which it is not (see the plan).
    pub(crate) heads: Vec<usize>,
    pub(crate) context: usize,
    pub(crate) in_channels: usize,
    pub(crate) groups: usize,
    pub(crate) eps: f64,
    /// SDXL's size conditioning: each of six numbers as a sinusoid this wide.
    pub(crate) time_ids_dim: usize,
    pub(crate) added_in: usize,
}

impl UnetConfig {
    pub(crate) fn from_json(v: &Value) -> Res<Self> {
        let list = |k: &str| -> Res<Vec<usize>> {
            v.get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect())
                .ok_or_else(|| format!("UNet config has no `{k}` list").into())
        };
        let n = |k: &str| -> Res<usize> {
            v.get(k).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("UNet config has no `{k}`").into())
        };
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("");

        // What this file implements, said as refusals rather than assumed: a
        // UNet from another pipeline reads the same keys and means other
        // things by some of them.
        if s("addition_embed_type") != "text_time" {
            return Err("this UNet implements SDXL's `text_time` conditioning and no other".into());
        }
        if v.get("use_linear_projection").and_then(Value::as_bool) != Some(true) {
            return Err("this UNet implements linear `proj_in`/`proj_out` only".into());
        }
        let channels = list("block_out_channels")?;
        let down: Vec<String> = v
            .get("down_block_types")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        let depth = list("transformer_layers_per_block")?;
        let depth = down
            .iter()
            .zip(&depth)
            .map(|(kind, &d)| if kind.starts_with("CrossAttn") { d } else { 0 })
            .collect::<Vec<_>>();
        let cfg = UnetConfig {
            layers_per_block: n("layers_per_block")?,
            heads: list("attention_head_dim")?,
            context: n("cross_attention_dim")?,
            in_channels: n("in_channels")?,
            groups: n("norm_num_groups")?,
            eps: v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5),
            time_ids_dim: n("addition_time_embed_dim")?,
            added_in: n("projection_class_embeddings_input_dim")?,
            depth,
            channels,
        };
        if cfg.heads.len() != cfg.channels.len() || cfg.depth.len() != cfg.channels.len() {
            return Err("UNet config lists disagree about how many levels there are".into());
        }
        Ok(cfg)
    }

    fn time_width(&self) -> usize {
        self.channels[0] * 4
    }
}

struct Resnet {
    norm1: GroupNorm,
    conv1: Conv2d,
    time: Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl Resnet {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: &UnetConfig, cin: usize, cout: usize) -> Res<Self> {
        Ok(Resnet {
            norm1: GroupNorm::load(cx, r, "norm1", cin, cfg.groups, cfg.eps)?,
            conv1: Conv2d::load(cx, r, "conv1", (cin, cout, 3), 1)?,
            time: Linear::load(cx, r, "time_emb_proj", cfg.time_width(), cout, true)?,
            norm2: GroupNorm::load(cx, r, "norm2", cout, cfg.groups, cfg.eps)?,
            conv2: Conv2d::load(cx, r, "conv2", (cout, cout, 3), 1)?,
            shortcut: match cin != cout {
                true => Some(Conv2d::load(cx, r, "conv_shortcut", (cin, cout, 1), 1)?),
                false => None,
            },
        })
    }

    /// `temb` has already been through its SiLU; see [`Unet::forward`].
    fn forward(&self, x: &Tensor, temb: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let t = self.time.forward(temb)?.unsqueeze(2)?.unsqueeze(3)?;
        let h = h.broadcast_add(&t)?;
        let h = self.conv2.forward(&self.norm2.forward(&h)?.silu()?)?;
        match &self.shortcut {
            Some(s) => s.forward(x)? + h,
            None => x + h,
        }
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

impl Attn {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, c: usize, kv: usize, heads: usize) -> Res<Self> {
        Ok(Attn {
            q: Linear::load(cx, r, "to_q", c, c, false)?,
            k: Linear::load(cx, r, "to_k", kv, c, false)?,
            v: Linear::load(cx, r, "to_v", kv, c, false)?,
            out: Linear::load(cx, r, "to_out.0", c, c, true)?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Tensor) -> candle_core::Result<Tensor> {
        let a = attention(&self.q.forward(x)?, &self.k.forward(ctx)?, &self.v.forward(ctx)?, self.heads)?;
        self.out.forward(&a)
    }
}

/// One transformer block over the flattened grid.
struct Block {
    norm1: LayerNorm,
    attn1: Attn,
    norm2: LayerNorm,
    attn2: Attn,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl Block {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, c: usize, context: usize, heads: usize) -> Res<Self> {
        Ok(Block {
            norm1: LayerNorm::load(cx, r, "norm1", c, 1e-5)?,
            attn1: Attn::load(cx, &r.pp("attn1"), c, c, heads)?,
            norm2: LayerNorm::load(cx, r, "norm2", c, 1e-5)?,
            attn2: Attn::load(cx, &r.pp("attn2"), c, context, heads)?,
            norm3: LayerNorm::load(cx, r, "norm3", c, 1e-5)?,
            ff_in: Linear::load(cx, r, "ff.net.0.proj", c, 8 * c, true)?,
            ff_out: Linear::load(cx, r, "ff.net.2", 4 * c, c, true)?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let x = (x + self.attn1.forward(&h, &h)?)?;
        let x = (&x + self.attn2.forward(&self.norm2.forward(&x)?, ctx)?)?;
        // GEGLU: one projection to twice the hidden width, half of it gating
        // the other half through a GELU.
        let h = self.ff_in.forward(&self.norm3.forward(&x)?)?;
        let half = h.dim(D::Minus1)? / 2;
        let h = (h.narrow(D::Minus1, 0, half)? * h.narrow(D::Minus1, half, half)?.gelu_erf()?)?;
        x + self.ff_out.forward(&h)?
    }
}

/// A stack of [`Block`]s with the grid-to-sequence adapters around it.
struct Transformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<Block>,
    proj_out: Linear,
}

impl Transformer {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: &UnetConfig, c: usize, depth: usize, heads: usize) -> Res<Self> {
        Ok(Transformer {
            norm: GroupNorm::load(cx, r, "norm", c, cfg.groups, 1e-6)?,
            proj_in: Linear::load(cx, r, "proj_in", c, c, true)?,
            blocks: (0..depth)
                .map(|i| Block::load(cx, &r.pp(format!("transformer_blocks.{i}")), c, cfg.context, heads))
                .collect::<Res<_>>()?,
            proj_out: Linear::load(cx, r, "proj_out", c, c, true)?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Tensor) -> candle_core::Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let mut s = self.proj_in.forward(&to_seq(&self.norm.forward(x)?)?)?;
        for b in &self.blocks {
            s = b.forward(&s, ctx)?;
        }
        to_grid(&self.proj_out.forward(&s)?, h, w)? + x
    }
}

/// A resnet, then the level's transformer if it has one.
struct Stage {
    resnet: Resnet,
    attn: Option<Transformer>,
}

impl Stage {
    fn forward(&self, x: &Tensor, temb: &Tensor, ctx: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.resnet.forward(x, temb)?;
        match &self.attn {
            Some(t) => t.forward(&x, ctx),
            None => Ok(x),
        }
    }
}

struct Level {
    stages: Vec<Stage>,
    /// A stride-2 convolution on the way down, nearest-2× then a convolution
    /// on the way up. The last level down and the last up have none.
    resample: Option<Conv2d>,
}

pub(crate) struct Unet {
    cfg: UnetConfig,
    time1: Linear,
    time2: Linear,
    add1: Linear,
    add2: Linear,
    conv_in: Conv2d,
    down: Vec<Level>,
    mid: (Resnet, Transformer, Resnet),
    up: Vec<Level>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Unet {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: UnetConfig) -> Res<Self> {
        let ch = &cfg.channels;
        let levels = ch.len();
        let tw = cfg.time_width();

        let mut down = Vec::with_capacity(levels);
        let mut cin = ch[0];
        for (i, &cout) in ch.iter().enumerate() {
            let lr = r.pp(format!("down_blocks.{i}"));
            let mut stages = Vec::new();
            for j in 0..cfg.layers_per_block {
                stages.push(Stage {
                    resnet: Resnet::load(cx, &lr.pp(format!("resnets.{j}")), &cfg, if j == 0 { cin } else { cout }, cout)?,
                    attn: match cfg.depth[i] {
                        0 => None,
                        d => Some(Transformer::load(cx, &lr.pp(format!("attentions.{j}")), &cfg, cout, d, cfg.heads[i])?),
                    },
                });
            }
            let resample = match i + 1 < levels {
                true => Some(Conv2d::load(cx, &lr, "downsamplers.0.conv", (cout, cout, 3), 2)?),
                false => None,
            };
            down.push(Level { stages, resample });
            cin = cout;
        }

        let top = ch[levels - 1];
        let mr = r.pp("mid_block");
        let mid = (
            Resnet::load(cx, &mr.pp("resnets.0"), &cfg, top, top)?,
            Transformer::load(cx, &mr.pp("attentions.0"), &cfg, top, cfg.depth[levels - 1], cfg.heads[levels - 1])?,
            Resnet::load(cx, &mr.pp("resnets.1"), &cfg, top, top)?,
        );

        // The way up mirrors the way down, one resnet more per level, and
        // each resnet's input is its predecessor's output *plus* a skip.
        // Which skip is arithmetic on the level widths: the last resnet of a
        // level takes the skip from the level below it on the way down, which
        // is narrower. diffusers spells this `res_skip_channels`.
        let mut up = Vec::with_capacity(levels);
        let mut prev = top;
        for i in 0..levels {
            let level = levels - 1 - i;
            let cout = ch[level];
            let below = ch[level.saturating_sub(1)];
            let lr = r.pp(format!("up_blocks.{i}"));
            let mut stages = Vec::new();
            for j in 0..=cfg.layers_per_block {
                let skip = if j == cfg.layers_per_block { below } else { cout };
                let from = if j == 0 { prev } else { cout };
                stages.push(Stage {
                    resnet: Resnet::load(cx, &lr.pp(format!("resnets.{j}")), &cfg, from + skip, cout)?,
                    attn: match cfg.depth[level] {
                        0 => None,
                        d => Some(Transformer::load(cx, &lr.pp(format!("attentions.{j}")), &cfg, cout, d, cfg.heads[level])?),
                    },
                });
            }
            let resample = match i + 1 < levels {
                true => Some(Conv2d::load(cx, &lr, "upsamplers.0.conv", (cout, cout, 3), 1)?),
                false => None,
            };
            up.push(Level { stages, resample });
            prev = cout;
        }

        let unet = Unet {
            time1: Linear::load(cx, r, "time_embedding.linear_1", ch[0], tw, true)?,
            time2: Linear::load(cx, r, "time_embedding.linear_2", tw, tw, true)?,
            add1: Linear::load(cx, r, "add_embedding.linear_1", cfg.added_in, tw, true)?,
            add2: Linear::load(cx, r, "add_embedding.linear_2", tw, tw, true)?,
            conv_in: Conv2d::load(cx, r, "conv_in", (cfg.in_channels, ch[0], 3), 1)?,
            norm_out: GroupNorm::load(cx, r, "conv_norm_out", ch[0], cfg.groups, cfg.eps)?,
            conv_out: Conv2d::load(cx, r, "conv_out", (ch[0], cfg.in_channels, 3), 1)?,
            down,
            mid,
            up,
            cfg,
        };
        Ok(unet)
    }

    /// One prediction of the noise in `x`.
    ///
    /// `x` is `[B, 4, h, w]`, `ctx` the prompt as `[B, 77, 2048]`, `pooled`
    /// `[B, 1280]` and `time_ids` the six size numbers, the same for every
    /// image in the batch.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        t: f64,
        ctx: &Tensor,
        pooled: &Tensor,
        time_ids: &[f64; 6],
    ) -> candle_core::Result<Tensor> {
        let b = x.dim(0)?;
        let dev = x.device();
        let dtype = x.dtype();
        let err = |e: Box<dyn std::error::Error>| candle_core::Error::Msg(e.to_string());

        // The noise level, as a vector.
        let t_emb = timestep_embedding(&[t], self.cfg.channels[0], true, 0.0, dev).map_err(err)?.to_dtype(dtype)?;
        let t_emb = self.time2.forward(&self.time1.forward(&t_emb)?.silu()?)?;
        // SDXL's addition: the image's size and crop, each a sinusoid of its
        // own, beside the pooled prompt.
        let ids = timestep_embedding(time_ids, self.cfg.time_ids_dim, true, 0.0, dev).map_err(err)?;
        let ids = ids.reshape((1, 6 * self.cfg.time_ids_dim))?.to_dtype(dtype)?.repeat((b, 1))?;
        let added = Tensor::cat(&[pooled, &ids], 1)?;
        let added = self.add2.forward(&self.add1.forward(&added)?.silu()?)?;
        // Every resnet applies a SiLU to this before its own projection, so
        // it is applied once here instead of forty times.
        let temb = t_emb.broadcast_add(&added)?.silu()?;

        let mut h = self.conv_in.forward(x)?;
        let mut skips = vec![h.clone()];
        for level in &self.down {
            for s in &level.stages {
                h = s.forward(&h, &temb, ctx)?;
                skips.push(h.clone());
            }
            if let Some(ds) = &level.resample {
                h = ds.forward(&h)?;
                skips.push(h.clone());
            }
        }

        h = self.mid.0.forward(&h, &temb)?;
        h = self.mid.1.forward(&h, ctx)?;
        h = self.mid.2.forward(&h, &temb)?;

        for level in &self.up {
            for s in &level.stages {
                let skip = skips.pop().expect("one skip per up-resnet, by construction");
                h = s.forward(&Tensor::cat(&[&h, &skip], 1)?, &temb, ctx)?;
            }
            if let Some(us) = &level.resample {
                let (_, _, hh, ww) = h.dims4()?;
                h = us.forward(&h.upsample_nearest2d(hh * 2, ww * 2)?)?;
            }
        }
        debug_assert!(skips.is_empty());

        self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)
    }
}
