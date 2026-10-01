//! LTX-2.5's convolutional video decoder: 128-channel latents to frames.
//!
//! `vae/ltx-2.5-video-vae-conv-bf16.safetensors`, a `CausalVideoAutoencoder`
//! whose decoder is, despite the name, not causal. Its config is in the
//! file's metadata; `docs/video-plan.md` has the stack written out.
//!
//! A latent is `[128, F, h, w]` and decodes to `8(F − 1) + 1` frames of
//! `32h × 32w`. Three ideas make that up:
//!
//! - **3×3×3 convolutions** at every level ([`Conv3d`]: on the M5's matrix
//!   units in bf16, and built from 2D ones elsewhere).
//! - **Depth to space.** Each upsampling block widens the channels with a
//!   convolution and then unfolds them into time and space: 4096 channels
//!   become 512 channels at twice the frames, rows and columns. When time
//!   doubles, the first new frame is dropped, which is how one latent frame
//!   becomes one picture at the start and every later one becomes eight.
//! - **Unpatchify.** The last convolution makes 48 channels per position,
//!   and each is one pixel of a 4×4 patch of RGB.
//!
//! There is no attention, no noise and no timestep. The decoder is a pure
//! function of the latent.

use super::conv3d::{norm_silu, Conv3d, Time};
use super::metadata;
use crate::common::{settle, Loader};
use crate::image::nn::{check_decoded, Ctx};
use crate::image::{finish, open};
use crate::prof::span;
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor};
use kvad::serde_json::Value;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The conv decoder's file in [`super::LTX_REPO`].
pub const FILE: &str = "vae/ltx-2.5-video-vae-conv-bf16.safetensors";

/// PixelNorm's epsilon in the video decoder.
const EPS: f64 = 1e-8;

enum Block {
    /// Residual blocks, each `x + conv(silu(norm(conv(silu(norm(x))))))`.
    Res(Vec<(Conv3d, Conv3d)>),
    /// A convolution to `stride product × out` channels, then depth to space.
    Up { conv: Conv3d, stride: [usize; 3], out: usize },
}

pub struct VideoDecoder {
    conv_in: Conv3d,
    blocks: Vec<Block>,
    conv_out: Conv3d,
    patch: usize,
    /// Per-channel statistics the latents were normalised by, `[1, 128, 1, 1]`.
    mean: Tensor,
    std: Tensor,
    dtype: DType,
    params: usize,
}

impl VideoDecoder {
    /// Load the decoder half of the file at `path`, computing in `dtype` on
    /// `device`.
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Res<Self> {
        let config = metadata(path, "config")?;
        let vae = &config["vae"];
        if vae["_class_name"] != "CausalVideoAutoencoder" {
            return Err(format!("{}: not LTX's conv video VAE ({})", path.display(), vae["_class_name"]).into());
        }
        // The flags this implementation assumes, checked rather than hoped.
        for (key, want) in [
            ("causal_decoder", Value::Bool(false)),
            ("timestep_conditioning", Value::Bool(false)),
            ("norm_layer", Value::from("pixel_norm")),
            ("spatial_padding_mode", Value::from("zeros")),
            ("dims", Value::from(3)),
        ] {
            if vae[key] != want {
                return Err(format!("{}: `{key}` is {}, and this decoder is written for {want}", path.display(), vae[key]).into());
            }
        }
        let latent = vae["latent_channels"].as_u64().ok_or("no latent_channels")? as usize;
        let base = vae["decoder_base_channels"].as_u64().ok_or("no decoder_base_channels")? as usize;
        let patch = vae["patch_size"].as_u64().ok_or("no patch_size")? as usize;
        let spec = vae["decoder_blocks"].as_array().ok_or("no decoder_blocks")?;

        // The decoder runs the config's list backwards, from the latent end.
        // Its widest point is the base times every upsampler's multiplier,
        // and each upsampler divides by its own on the way out.
        let spec: Vec<(&str, &Value)> = spec.iter().rev().map(|b| (b[0].as_str().unwrap_or(""), &b[1])).collect();
        let multiplier = |p: &Value| p["multiplier"].as_u64().unwrap_or(1) as usize;
        let mut c = base * spec.iter().filter(|(n, _)| n.starts_with("compress_")).map(|(_, p)| multiplier(p)).product::<usize>();

        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let paths = [path.to_path_buf()];
        let r = open(&paths, DType::BF16)?;
        r.skip_under("encoder");
        let d = r.pp("decoder");

        let conv_in = Conv3d::load(&cx, &d, "conv_in.conv", latent, c)?;
        let mut blocks = Vec::new();
        for (i, (name, p)) in spec.iter().enumerate() {
            let b = d.pp(format!("up_blocks.{i}"));
            let block = match *name {
                "res_x" => {
                    let n = p["num_layers"].as_u64().ok_or("res_x without num_layers")? as usize;
                    let res = (0..n)
                        .map(|j| {
                            let rb = b.pp(format!("res_blocks.{j}"));
                            Ok((Conv3d::load(&cx, &rb, "conv1.conv", c, c)?, Conv3d::load(&cx, &rb, "conv2.conv", c, c)?))
                        })
                        .collect::<Res<Vec<_>>>()?;
                    Block::Res(res)
                }
                "compress_all" | "compress_time" | "compress_space" => {
                    let stride = match *name {
                        "compress_all" => [2, 2, 2],
                        "compress_time" => [2, 1, 1],
                        _ => [1, 2, 2],
                    };
                    let out = c / multiplier(p);
                    let conv = Conv3d::load(&cx, &b, "conv.conv", c, stride.iter().product::<usize>() * out)?;
                    c = out;
                    Block::Up { conv, stride, out }
                }
                other => return Err(format!("{}: decoder block `{other}` is not implemented", path.display()).into()),
            };
            blocks.push(block);
        }
        let conv_out = Conv3d::load(&cx, &d, "conv_out.conv", c, 3 * patch * patch)?;

        let stats = r.pp("per_channel_statistics");
        let stat = |name: &str| -> Res<Tensor> { Ok(cx.get(&stats, latent, name)?.to_dtype(DType::F32)?.reshape((1, latent, 1, 1))?) };
        let (mean, std) = (stat("mean-of-means")?, stat("std-of-means")?);
        let params = finish("LTX video decoder", &paths, &r)?;
        Ok(VideoDecoder { conv_in, blocks, conv_out, patch, mean, std, dtype, params })
    }

    pub fn params(&self) -> usize {
        self.params
    }

    /// Normalised latents `[128, F, h, w]` to frames `[8(F − 1) + 1, 3,
    /// 32h, 32w]` in f32, to be clamped to `[0, 1]`. Not clamped here:
    /// [`to_video`] does it on the host, after its check, and on the device
    /// a clamp is `x > y ? x : y` both ways, which turns NaN into 0, black,
    /// and hides it from the check.
    pub fn decode(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        self.decode_failing(latent, None)
    }

    /// [`Self::decode`], with the stage numbered `fail` counted from
    /// `conv_in` (0) through each residual step and up block to `conv_out`
    /// (the last) left as zeros, as a Metal command buffer that failed
    /// leaves them. Each stage's result is looked at as it comes
    /// ([`check_stage`]), so that one is refused there.
    pub(crate) fn decode_failing(&self, latent: &Tensor, fail: Option<usize>) -> candle_core::Result<Tensor> {
        let mut stage = 0;
        let mut checked = |h: Tensor| -> candle_core::Result<Tensor> {
            let n = stage;
            stage += 1;
            let h = match fail == Some(n) {
                true => h.zeros_like()?,
                false => h,
            };
            check_stage(&h, "conv decoder", n)?;
            Ok(h)
        };
        // Frames first from here on: the frames are every convolution's batch.
        let z = latent.permute((1, 0, 2, 3))?.to_dtype(DType::F32)?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let dev = latent.device();
        // What `crate::prof` calls a step: its kind and the size it works at.
        let at = |kind: &str, x: &Tensor| -> String {
            let (t, c, h, w) = x.dims4().unwrap_or_default();
            format!("{kind} {c} × {t} × {h}×{w}")
        };
        let mut x = checked(span(|| at("conv in", &z), dev, || self.conv_in.forward(&z.to_dtype(self.dtype)?))?)?;
        for block in &self.blocks {
            match block {
                Block::Res(res) => {
                    for (c1, c2) in res {
                        x = checked(span(|| at("residual", &x), dev, || residual(&x, c1, c2))?)?;
                        // candle's pool lets go of what a step dropped only
                        // when the device is synchronised.
                        settle(x.device())?;
                    }
                }
                Block::Up { conv, stride, out } => {
                    x = checked(span(|| at("up", &x), dev, || up(&x, conv, *stride, *out))?)?;
                    settle(x.device())?;
                }
            }
        }
        let p = self.patch;
        let (t, _, h, w) = x.dims4()?;
        let frames = Tensor::zeros((t, 3, h * p, w * p), DType::F32, x.device())?;
        // The last stage, however many chunks it takes: counted once, and
        // each chunk looked at.
        let last = stage;
        span(|| at("out", &x), dev, || {
            for (f, n) in chunks(&x) {
                let y = self.conv_out.frames(&norm_silu(&halo(&x, f, n, (1, 1))?, EPS)?, (f.saturating_sub(1), t), (f, f + n))?;
                let y = if fail == Some(last) { y.zeros_like()? } else { y };
                check_stage(&y, "conv decoder", last)?;
                let y = ((unpatchify(&y, p)?.to_dtype(DType::F32)? + 1.0)? * 0.5)?;
                frames.slice_set(&y, 0, f)?;
            }
            Ok(())
        })?;
        Ok(frames)
    }

    /// How many stages [`Self::decode_failing`] counts.
    #[cfg(test)]
    pub(crate) fn stages(&self) -> usize {
        let steps = |b: &Block| match b {
            Block::Res(res) => res.len(),
            Block::Up { .. } => 1,
        };
        2 + self.blocks.iter().map(steps).sum::<usize>()
    }
}

/// The most elements of a layer's input [`chunks`] takes at once: 256 M,
/// 512 MB in bf16. At 1536×1024 the decoder's last stage is 3 GB a tensor,
/// and a residual step done whole kept five of them alive; the whole decode
/// ran out of memory at 77 GB.
const BUDGET: usize = 1 << 28;

/// `(first frame, frames)` for each chunk of `x` a step works through.
fn chunks(x: &Tensor) -> Vec<(usize, usize)> {
    chunks_in(x, BUDGET)
}

/// [`chunks`] with `budget` elements a chunk at most.
fn chunks_in(x: &Tensor, budget: usize) -> Vec<(usize, usize)> {
    let t = x.dim(0).unwrap_or(0);
    let n = (budget / (x.elem_count() / t.max(1)).max(1)).clamp(1, t.max(1));
    (0..t).step_by(n).map(|f| (f, n.min(t - f))).collect()
}

/// Frames `f − before .. f + n + after` of `x`, cut off at its ends: a
/// chunk and the frames either side of it that the convolutions after it
/// read.
fn halo(x: &Tensor, f: usize, n: usize, (before, after): (usize, usize)) -> candle_core::Result<Tensor> {
    let (lo, hi) = (f.saturating_sub(before), (f + n + after).min(x.dim(0)?));
    x.narrow(0, lo, hi - lo)
}

/// One residual step, `x + conv(silu(pn(conv(silu(pn(x))))))`, a chunk of
/// frames at a time.
///
/// A chunk's frames `f .. f + n` need the first convolution's output one
/// frame beyond them each way, and that needs the input two frames beyond;
/// in the causal encoder, two and four frames before and none after. Those
/// halo frames are computed twice, once for each chunk that reads them: the
/// price of never holding the steps in between at full size. A stage small
/// enough for one chunk pays nothing.
fn residual(x: &Tensor, c1: &Conv3d, c2: &Conv3d) -> candle_core::Result<Tensor> {
    residual_in(x, c1, c2, BUDGET)
}

/// [`residual`] in chunks of `budget` elements at most.
fn residual_in(x: &Tensor, c1: &Conv3d, c2: &Conv3d, budget: usize) -> candle_core::Result<Tensor> {
    let t = x.dim(0)?;
    let parts = chunks_in(x, budget);
    if parts.len() == 1 {
        let h = c1.forward(&norm_silu(x, EPS)?)?;
        return x + c2.forward(&norm_silu(&h, EPS)?)?;
    }
    let (before, after) = c1.time().reach();
    let out = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
    for (f, n) in parts {
        let (a, b) = (f.saturating_sub(before), (f + n + after).min(t));
        let h = c1.frames(&norm_silu(&halo(x, f, n, (2 * before, 2 * after))?, EPS)?, (f.saturating_sub(2 * before), t), (a, b))?;
        let h = c2.frames(&norm_silu(&h, EPS)?, (a, t), (f, f + n))?;
        out.slice_set(&(x.narrow(0, f, n)? + h)?, 0, f)?;
    }
    Ok(out)
}

/// An up block, the convolution and the depth to space, a chunk of frames
/// at a time.
fn up(x: &Tensor, conv: &Conv3d, stride: [usize; 3], c: usize) -> candle_core::Result<Tensor> {
    let (t, _, h, w) = x.dims4()?;
    let parts = chunks(x);
    if parts.len() == 1 {
        return depth_to_space(&conv.forward(x)?, stride, c);
    }
    let [p1, p2, p3] = stride;
    // Doubling time drops the first frame of the result.
    let drop = (p1 == 2) as usize;
    let out = Tensor::zeros((t * p1 - drop, c, h * p2, w * p3), x.dtype(), x.device())?;
    for (f, n) in parts {
        let y = unfold(&conv.frames(&halo(x, f, n, (1, 1))?, (f.saturating_sub(1), t), (f, f + n))?, stride, c)?;
        match (f, drop) {
            (0, 1) => out.slice_set(&y.narrow(0, 1, n * p1 - 1)?, 0, 0)?,
            _ => out.slice_set(&y, 0, f * p1 - drop)?,
        }
    }
    Ok(out)
}

/// `[T, (c·p₁·p₂·p₃), H, W]` to `[T·p₁ (− 1), c, H·p₂, W·p₃]`: each group of
/// channels unfolded into a block of frames, rows and columns.
///
/// The channel index is `((c·p₁ + i)·p₂ + j)·p₃ + k`, as the reference's
/// `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)` has it. When time
/// doubles, the first frame of the result is dropped.
fn depth_to_space(x: &Tensor, stride: [usize; 3], c: usize) -> candle_core::Result<Tensor> {
    let x = unfold(x, stride, c)?;
    match stride[0] {
        2 => x.narrow(0, 1, x.dim(0)? - 1),
        _ => Ok(x),
    }
}

/// [`depth_to_space`] without dropping a frame: `[T·p₁, c, H·p₂, W·p₃]`.
fn unfold(x: &Tensor, [p1, p2, p3]: [usize; 3], c: usize) -> candle_core::Result<Tensor> {
    let (t, _, h, w) = x.dims4()?;
    x.reshape(&[t, c, p1, p2, p3, h, w][..])?
        // [t, p1, c, h, p2, w, p3]
        .permute(&[0, 2, 1, 5, 3, 6, 4][..])?
        .contiguous()?
        .reshape((t * p1, c, h * p2, w * p3))
}

/// `[T, 3·p·p, H, W]` to `[T, 3, H·p, W·p]`.
///
/// The channel index is `c·p² + r·p + q`, where `q` is the offset in the row
/// and `r` the offset in the column — `b (c p r q) f h w -> b c (f p) (h q)
/// (w r)` in the reference. Width is the middle factor, not height; the
/// natural guess is the other way round.
pub(crate) fn unpatchify(x: &Tensor, p: usize) -> candle_core::Result<Tensor> {
    let (t, c, h, w) = x.dims4()?;
    let c = c / (p * p);
    x.reshape(&[t, c, p, p, h, w][..])?
        // [t, c, r, q, h, w] → [t, c, h, q, w, r]
        .permute(&[0, 1, 4, 3, 5, 2][..])?
        .contiguous()?
        .reshape((t, c, h * p, w * p))
}

/// A step of the encoder.
enum Down {
    /// Residual blocks, as the decoder's.
    Res(Vec<(Conv3d, Conv3d)>),
    /// Space to depth, with a residual: a convolution to `out / (p₁·p₂·p₃)`
    /// channels folded into `out`, plus the input folded the same way and
    /// averaged down to `out` channels in groups of `group`.
    Fold { conv: Conv3d, stride: [usize; 3], group: usize },
}

/// The encoder half of the same file: a clip to its latents, or one
/// picture to the latent frame the DiT holds it as, for image-to-video.
///
/// The mirror of [`VideoDecoder`]: patchify 4×4 pixels into 48 channels,
/// then residual blocks and space-to-depth steps down to `[128, F, h, w]`,
/// normalised by the same statistics. `8(F − 1) + 1` frames make `F` latent
/// frames, the inverse of what the decoder makes of them.
///
/// **Causal**, unlike the decoder: every convolution reads its frame and
/// the two before it, the first frame standing in for those before the clip
/// ([`Time::Causal`]), and a step that halves time puts one more copy of the
/// first frame in front. So the first latent frame is the first frame's
/// alone, whatever follows it, and each later one reads only frames up to
/// the end of its own eight. A picture is that first frame.
///
/// One encode of 121 frames at 768×512 in bf16 on an M5 Pro takes 10.2 s,
/// with a peak footprint of 11.6 GB (`examples/ltx_encode.rs --measure`).
/// The residual steps work a chunk of frames at a time, as the decoder's
/// do; the space-to-depth steps work on the whole clip.
pub struct VideoEncoder {
    conv_in: Conv3d,
    blocks: Vec<Down>,
    conv_out: Conv3d,
    patch: usize,
    latent: usize,
    mean: Tensor,
    std: Tensor,
    dtype: DType,
    params: usize,
}

impl VideoEncoder {
    /// Load the encoder half of the file at `path`, computing in `dtype` on
    /// `device`.
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Res<Self> {
        let config = metadata(path, "config")?;
        let vae = &config["vae"];
        if vae["_class_name"] != "CausalVideoAutoencoder" {
            return Err(format!("{}: not LTX's conv video VAE ({})", path.display(), vae["_class_name"]).into());
        }
        for (key, want) in [
            ("norm_layer", Value::from("pixel_norm")),
            ("spatial_padding_mode", Value::from("zeros")),
            ("dims", Value::from(3)),
            // 128 means and one log-variance shared by all of them, which
            // an encoder that keeps the means has no use for.
            ("latent_log_var", Value::from("uniform")),
        ] {
            if vae[key] != want {
                return Err(format!("{}: `{key}` is {}, and this encoder is written for {want}", path.display(), vae[key]).into());
            }
        }
        let latent = vae["latent_channels"].as_u64().ok_or("no latent_channels")? as usize;
        let mut c = vae["encoder_base_channels"].as_u64().ok_or("no encoder_base_channels")? as usize;
        let patch = vae["patch_size"].as_u64().ok_or("no patch_size")? as usize;
        let spec = vae["encoder_blocks"].as_array().ok_or("no encoder_blocks")?;

        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let paths = [path.to_path_buf()];
        let r = open(&paths, DType::BF16)?;
        r.skip_under("decoder");
        let e = r.pp("encoder");

        let causal = |name: &str, d: &crate::common::Reader<'_>, cin: usize, cout: usize| -> Res<Conv3d> { Ok(Conv3d::load(&cx, d, name, cin, cout)?.padded(Time::Causal)) };
        let conv_in = causal("conv_in.conv", &e, 3 * patch * patch, c)?;
        let mut blocks = Vec::new();
        for (i, b) in spec.iter().enumerate() {
            let (name, p) = (b[0].as_str().unwrap_or(""), &b[1]);
            let d = e.pp(format!("down_blocks.{i}"));
            let block = match name {
                "res_x" => {
                    let n = p["num_layers"].as_u64().ok_or("res_x without num_layers")? as usize;
                    let res = (0..n)
                        .map(|j| {
                            let rb = d.pp(format!("res_blocks.{j}"));
                            Ok((causal("conv1.conv", &rb, c, c)?, causal("conv2.conv", &rb, c, c)?))
                        })
                        .collect::<Res<Vec<_>>>()?;
                    Down::Res(res)
                }
                "compress_all_res" | "compress_time_res" | "compress_space_res" => {
                    let stride = match name {
                        "compress_all_res" => [2, 2, 2],
                        "compress_time_res" => [2, 1, 1],
                        _ => [1, 2, 2],
                    };
                    let out = c * p["multiplier"].as_u64().unwrap_or(2) as usize;
                    let n: usize = stride.iter().product();
                    let conv = causal("conv.conv", &d, c, out / n)?;
                    let group = c * n / out;
                    c = out;
                    Down::Fold { conv, stride, group }
                }
                other => return Err(format!("{}: encoder block `{other}` is not implemented", path.display()).into()),
            };
            blocks.push(block);
        }
        // 128 means and the shared log-variance.
        let conv_out = causal("conv_out.conv", &e, c, latent + 1)?;

        let stats = r.pp("per_channel_statistics");
        let stat = |name: &str| -> Res<Tensor> { Ok(cx.get(&stats, latent, name)?.to_dtype(DType::F32)?.reshape((1, latent, 1, 1))?) };
        let (mean, std) = (stat("mean-of-means")?, stat("std-of-means")?);
        let params = finish("LTX video encoder", &paths, &r)?;
        Ok(VideoEncoder { conv_in, blocks, conv_out, patch, latent, mean, std, dtype, params })
    }

    pub fn params(&self) -> usize {
        self.params
    }

    /// A picture `[3, H, W]` in `[−1, 1]`, both sides a multiple of 32, to
    /// its normalised latent `[128, 1, H/32, W/32]`, as f32.
    pub fn encode(&self, picture: &Tensor) -> candle_core::Result<Tensor> {
        self.encode_clip(&picture.unsqueeze(0)?)
    }

    /// A clip `[T, 3, H, W]` in `[−1, 1]`, frames first, `T = 8k + 1` and
    /// both sides a multiple of 32, to its normalised latents
    /// `[128, k + 1, H/32, W/32]`, as f32.
    pub fn encode_clip(&self, frames: &Tensor) -> candle_core::Result<Tensor> {
        let (t, _, h, w) = frames.dims4()?;
        if h % 32 != 0 || w % 32 != 0 {
            candle_core::bail!("{w}×{h}: the encoder wants both sides a multiple of 32");
        }
        // The reference cuts the last frames off a clip of any other length,
        // and says so in a warning; a caller here is told instead.
        if t % 8 != 1 {
            candle_core::bail!("{t} frames: the encoder wants 8k + 1, such as {} or {}", t / 8 * 8 + 1, t / 8 * 8 + 9);
        }
        let x = patchify(&frames.to_dtype(self.dtype)?, self.patch)?;
        let mut x = self.conv_in.forward(&x)?;
        for block in &self.blocks {
            x = match block {
                Down::Res(res) => {
                    for (c1, c2) in res {
                        x = residual(&x, c1, c2)?;
                    }
                    x
                }
                Down::Fold { conv, stride, group } => {
                    // Halving time puts a copy of the first frame in front,
                    // so that one frame folds into one.
                    let x = match stride[0] {
                        2 => Tensor::cat(&[&x.narrow(0, 0, 1)?, &x], 0)?,
                        _ => x,
                    };
                    // The groups averaged as `[t·c/g, g, h, w]`: candle's
                    // Metal reductions are wrong over a five-axis tensor
                    // (mean, sum and max alike; 1e38 out), and right at four.
                    let skip = fold(&x, *stride)?;
                    let (t, c, h, w) = skip.dims4()?;
                    let skip = skip.reshape((t * c / group, *group, h, w))?.to_dtype(DType::F32)?.mean(1)?.reshape((t, c / group, h, w))?;
                    (fold(&conv.forward(&x)?, *stride)?.to_dtype(DType::F32)? + skip)?.to_dtype(self.dtype)?
                }
            };
            settle(x.device())?;
        }
        let x = self.conv_out.forward(&norm_silu(&x, EPS)?)?;
        let means = x.narrow(1, 0, self.latent)?.to_dtype(DType::F32)?;
        let z = means.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?;
        // [F, 128, h, w] → [128, F, h, w].
        z.permute((1, 0, 2, 3))?.contiguous()
    }
}

/// `[T, 3, H, W]` to `[T, 3·p·p, H/p, W/p]`: the inverse of [`unpatchify`].
///
/// The channel index is `c·p² + r·p + q`, `q` the offset in the row and `r`
/// in the column: `b c (f p) (h q) (w r) -> b (c p r q) f h w` in the
/// reference, with one frame a patch.
pub(crate) fn patchify(x: &Tensor, p: usize) -> candle_core::Result<Tensor> {
    let (t, c, h, w) = x.dims4()?;
    x.reshape(&[t, c, h / p, p, w / p, p][..])?
        // [t, c, h, q, w, r] → [t, c, r, q, h, w]
        .permute(&[0, 1, 5, 3, 2, 4][..])?
        .contiguous()?
        .reshape((t, c * p * p, h / p, w / p))
}

/// `[T, c, H, W]` to `[T/p₁, c·p₁·p₂·p₃, H/p₂, W/p₃]`: each block of frames,
/// rows and columns folded into channels, the inverse of [`unfold`].
///
/// The channel index is `((c·p₁ + i)·p₂ + j)·p₃ + k`, as the reference's
/// `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w` has it.
fn fold(x: &Tensor, [p1, p2, p3]: [usize; 3]) -> candle_core::Result<Tensor> {
    let (t, c, h, w) = x.dims4()?;
    x.reshape(&[t / p1, p1, c, h / p2, p2, w / p3, p3][..])?
        // [d, p1, c, h, p2, w, p3] → [d, c, p1, p2, p3, h, w]
        .permute(&[0, 2, 1, 4, 6, 3, 5][..])?
        .contiguous()?
        .reshape((t / p1, c * p1 * p2 * p3, h / p2, w / p3))
}

/// A picture `rgb` of `width × height` scaled to cover `w × h` and cut to it
/// from the middle, as `[3, h, w]` in `[−1, 1]`, f32, on the host: how the
/// reference prepares a picture for the encoder.
///
/// Its `resize_and_center_crop`: scale by whichever of `h/height` and
/// `w/width` is larger, round the new size up, resample bilinearly, and cut
/// the middle out. The resampling is torch's `interpolate(mode="bilinear",
/// align_corners=False)` written out, with no antialiasing, as the
/// reference calls it: output pixel `i` reads the input at
/// `(i + ½)·(in/out) − ½`, clamped at 0, and blends its two neighbours.
pub fn picture(rgb: &[u8], width: usize, height: usize, w: usize, h: usize) -> candle_core::Result<Tensor> {
    if rgb.len() != width * height * 3 || width == 0 || height == 0 {
        candle_core::bail!("a {width}×{height} picture of {} bytes", rgb.len());
    }
    let scale = (h as f64 / height as f64).max(w as f64 / width as f64);
    let (nh, nw) = ((height as f64 * scale).ceil() as usize, (width as f64 * scale).ceil() as usize);
    let (top, left) = ((nh - h) / 2, (nw - w) / 2);
    // Each output row's (or column's) two input neighbours and the weight of
    // the second, in f32 as torch computes them.
    let taps = |out: usize, inp: usize, from: usize, n: usize| -> Vec<(usize, usize, f32)> {
        let ratio = inp as f32 / out as f32;
        (from..from + n)
            .map(|i| {
                let src = ((i as f32 + 0.5) * ratio - 0.5).max(0.0);
                let i0 = (src as usize).min(inp - 1);
                (i0, (i0 + 1).min(inp - 1), src - i0 as f32)
            })
            .collect()
    };
    let (rows, cols) = (taps(nh, height, top, h), taps(nw, width, left, w));
    let mut out = vec![0f32; 3 * h * w];
    for (y, &(y0, y1, fy)) in rows.iter().enumerate() {
        for (x, &(x0, x1, fx)) in cols.iter().enumerate() {
            for c in 0..3 {
                let at = |yy: usize, xx: usize| rgb[(yy * width + xx) * 3 + c] as f32;
                let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                out[(c * h + y) * w + x] = (top * (1.0 - fy) + bottom * fy) / 127.5 - 1.0;
            }
        }
    }
    Tensor::from_vec(out, (3, h, w), &Device::Cpu)
}

/// The per-channel statistics video latents are normalised by, from the VAE
/// file at `path`: `(mean, std)`, `[128]` each, in f32 on the CPU.
///
/// The latent upsampler works on un-normalised latents and reads these from
/// the VAE's file, as the reference does; it has none of its own.
pub fn statistics(path: &Path) -> Res<(Tensor, Tensor)> {
    // SAFETY: a read-only cache entry, as in `open`.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    let get = |n: &str| -> Res<Tensor> { Ok(st.load(&format!("per_channel_statistics.{n}"), &Device::Cpu)?.to_dtype(DType::F32)?) };
    Ok((get("mean-of-means")?, get("std-of-means")?))
}

/// Frames `[T, 3, H, W]` in `[0, 1]` as a [`kvad::video::Video`].
pub fn to_video(frames: &Tensor, fps: u32) -> Res<kvad::video::Video> {
    let (_, _, h, w) = frames.dims4()?;
    // Truncated rather than rounded, as the reference quantises.
    let x = (frames.to_dtype(DType::F32)?.permute((0, 2, 3, 1))?.contiguous()? * 255.0)?;
    let x = x.flatten_all()?.to_vec1::<f32>()?;
    // Both decoders hand their frames over on the CPU, so a failed buffer is
    // already in them as zeros, and the scaling above keeps a zero a zero.
    check_decoded(&x)?;
    check_lattice(&x, w, h)?;
    // `as u8` saturates: the clamp to [0, 1], after the checks.
    let rgb = x.into_iter().map(|v| v as u8).collect();
    Ok(kvad::video::Video { width: w, height: h, fps, rgb })
}

/// Refuses a stage's result that came back zeros: what a Metal command
/// buffer that failed leaves, most often for want of GPU memory, and with
/// no error said. `stage` counts from the decoder's first, for the message.
///
/// Both decoders synchronise the device after each step anyway, to let the
/// pool have its buffers back, so a look at [`STAGE_SAMPLE`] values from the
/// middle of the result costs a few kilobytes and no waiting. A real
/// stage's result is a residual stream, a convolution's or a linear layer's
/// with its bias, or noise through one, and is never all zeros. Looked at
/// here, a failure is caught at the stage that failed, whatever it goes on
/// to draw: the conv decoder's first stage, whose lattice [`check_lattice`]
/// cannot tell at 768×512, and the diffusion decoder's, whose frames are
/// drawn from noise and show nothing but the last two
/// (`ltx_diffvae::DiffDecoder::checked`).
pub(crate) fn check_stage(x: &Tensor, decoder: &str, stage: usize) -> candle_core::Result<()> {
    let flat = x.flatten_all()?;
    let (len, k) = (flat.dim(0)?, flat.dim(0)?.min(STAGE_SAMPLE));
    let sample = flat.narrow(0, (len - k) / 2, k)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if sample.iter().all(|&v| v == 0.0) {
        candle_core::bail!(
            "stage {stage} of the {decoder} came back zeros: its work on the GPU failed, most likely for want of \
             memory, and a failed command buffer on Metal reads as zeros rather than an error. Nothing was saved; \
             free some memory (another model loaded?) and try again"
        );
    }
    Ok(())
}

/// How many values [`check_stage`] reads: 16 KB in f32.
const STAGE_SAMPLE: usize = 4096;

/// Pixels per side of a latent cell, in both decoders.
const CELL: usize = 32;

/// Refuses a clip every frame of which repeats itself a latent cell
/// ([`CELL`]) across and down: what a decode that failed part-way and
/// carried on draws.
///
/// After a stage left as zeros, every layer after it sees the same numbers
/// everywhere but near the edges, and adds its biases. In the image VAEs
/// that makes a flat colour (`image::nn::check_flat`); here each up block
/// unfolds a position's channels into a block of pixels, so the same
/// becomes the same block at every latent cell, 32 pixels a side, and a
/// flat colour is only the case of plain blocks. The conv decoder, with
/// each of its 24 stages in turn zeroed on Metal at 768×512, drew such
/// lattices, grey, purple, black and white, one colour to thousands; and
/// from the second stage on, each frame's middle, inside a margin of a
/// side's eighth and never under 128 pixels, repeated to within half an
/// 8-bit level at all but at most 1.12% of its places. A failure of the
/// first stage let the edges' structure reach most of the middle at that
/// size (43% did not repeat), and passes here. The diffusion decoder draws
/// its frames from noise, and only the last two of its 31 stages failing
/// left a pattern: the others drew a murky texture, or, failing in stage 5,
/// where the context still carries the picture, frames much like a real
/// decode's, and none of that repeats. Both decoders look at each stage's
/// result as they go ([`check_stage`]), and that catches every one of those
/// stages; this is what is left for a failure that stage missed, the pattern
/// it still draws.
///
/// A real frame does not repeat, even a plain one: its grain differs from
/// cell to cell. Of 30 clips from both decoders at 768×512, plain walls,
/// grey, white paper, a fade and black screens among them, the frame that
/// repeated most was a black screen's from the diffusion decoder, and
/// 14.7% of its middle did not. The frames are not yet clamped here, so
/// what lies below black is still grain. Every frame must repeat, so a real
/// clip passes at its first frame, most often a few rows into it. Smaller
/// than 320 pixels a side, there is too little middle to judge.
fn check_lattice(x: &[f32], width: usize, height: usize) -> Res<()> {
    let (mx, my) = ((width / 8).max(128), (height / 8).max(128));
    if width < 2 * (mx + CELL) || height < 2 * (my + CELL) {
        return Ok(());
    }
    // Each place in the middle against the one a cell to its right and the
    // one a cell below, all three still in the middle.
    let (row, frame) = (width * 3, width * height * 3);
    let (cols, rows) = (width - 2 * mx - CELL, height - 2 * my - CELL);
    let allowed = cols * 3 * rows / MISSES;
    for f in x.chunks_exact(frame) {
        let mut misses = 0;
        for y in my..my + rows {
            let at = y * row + mx * 3;
            let here = &f[at..at + cols * 3];
            let right = &f[at + CELL * 3..at + (CELL + cols) * 3];
            let below = &f[at + CELL * row..at + CELL * row + cols * 3];
            // Half a level: the early stages' lattices repeat to within it,
            // not exactly, their edges' structure still fading.
            misses += here.iter().zip(right).zip(below).filter(|((a, r), b)| (*a - *r).abs() > 0.5 || (*a - *b).abs() > 0.5).count();
            // A picture: a frame that does not repeat is enough.
            if misses > allowed {
                return Ok(());
            }
        }
    }
    Err("the decoded video is a pattern: every frame repeats itself every 32 pixels across and down, where a picture \
         never does. The decode failed part-way, most likely for want of GPU memory, and the layers after the failure \
         drew only their biases, the same at every latent cell. Nothing was saved; free some memory (another model \
         loaded?) and try again"
        .into())
}

/// A frame repeats if fewer than one place in this many in its middle does
/// not: about 3%, 2.8 times the most a failed decode left and under a
/// quarter of the fewest a real frame had. Nearer the failures on purpose,
/// as `image::nn::FLAT` is: refusing a real clip blames memory for nothing,
/// where missing a failure saves what was saved before the check.
const MISSES: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    /// A residual step done a few frames at a time is the step done whole,
    /// with the decoder's padding and with the encoder's causal one, whose
    /// chunks read four frames back and none ahead.
    #[test]
    fn a_residual_step_in_chunks_is_the_step_whole() {
        let dev = Device::Cpu;
        let (t, c, h, w) = (9, 4, 3, 5);
        let x = Tensor::randn(0f32, 1.0, (t, c, h, w), &dev).unwrap();
        let k = || Tensor::randn(0f32, 0.3, (c, c, 3, 3, 3), &dev).unwrap();
        let b = || Tensor::randn(0f32, 0.3, c, &dev).unwrap();
        for time in [Time::Replicate, Time::Causal] {
            let c1 = Conv3d::folded(&k(), b(), time).unwrap();
            let c2 = Conv3d::folded(&k(), b(), time).unwrap();
            let whole = residual_in(&x, &c1, &c2, usize::MAX).unwrap();
            // Two frames a chunk: five chunks, the last of one frame.
            let parts = residual_in(&x, &c1, &c2, 2 * c * h * w).unwrap();
            assert_eq!(chunks_in(&x, 2 * c * h * w).len(), 5);
            let worst = (parts - &whole).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            assert!(worst < 1e-5, "{time:?}: the chunks differ from the whole by {worst}");
        }
    }

    #[test]
    fn depth_to_space_puts_each_channel_where_the_reference_rearrange_does() {
        // One frame, c = 1, stride (2, 2, 2): eight channels at one position
        // become a 2×2×2 block, and the first of its two frames is dropped.
        let x = Tensor::arange(0f32, 8.0, &Device::Cpu).unwrap().reshape((1, 8, 1, 1)).unwrap();
        let y = depth_to_space(&x, [2, 2, 2], 1).unwrap();
        assert_eq!(y.dims(), [1, 1, 2, 2]);
        // Channel ((i·2 + j)·2 + k) lands at frame i, row j, column k; frame
        // 1 is what is left.
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), [4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn fold_and_patchify_undo_unfold_and_unpatchify() {
        let x = Tensor::arange(0f32, (2 * 16 * 2 * 4) as f32, &Device::Cpu).unwrap().reshape((2, 16, 2, 4)).unwrap();
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let back = unfold(&fold(&unfold(&x, [2, 2, 2], 2).unwrap(), [2, 2, 2]).unwrap(), [2, 2, 2], 2).unwrap();
        assert_eq!(v(&back), v(&unfold(&x, [2, 2, 2], 2).unwrap()));
        assert_eq!(v(&fold(&unfold(&x, [1, 2, 2], 4).unwrap(), [1, 2, 2]).unwrap()), v(&x));
        let x = x.narrow(1, 0, 12).unwrap().contiguous().unwrap().reshape((2, 12, 2, 4)).unwrap();
        assert_eq!(v(&patchify(&unpatchify(&x, 2).unwrap(), 2).unwrap()), v(&x));
    }

    #[test]
    fn a_picture_is_scaled_to_cover_and_cut_from_the_middle() {
        // 4×2, each pixel's red its column·10 and green its row·10: scaled
        // ×2 to 8×4 and cut to 4×4, the columns kept are 2 to 5.
        let rgb: Vec<u8> = (0..2).flat_map(|y| (0..4).flat_map(move |x| [x * 10, y * 10, 0])).collect();
        let p = picture(&rgb, 4, 2, 4, 4).unwrap();
        assert_eq!(p.dims(), [3, 4, 4]);
        let red: Vec<f32> = p.get(0).unwrap().get(0).unwrap().to_vec1::<f32>().unwrap();
        // Output column 2 reads input (2.5)·½ − ½ = 0.75: 7.5; then 12.5,
        // 17.5, 22.5. Scaled to [−1, 1].
        let want = [7.5f32, 12.5, 17.5, 22.5].map(|v| v / 127.5 - 1.0);
        for (a, b) in red.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{red:?}");
        }
        // Row 0 of the output reads input row max(0.25 − ½, 0) = 0: green 0.
        assert!((p.get(1).unwrap().get(0).unwrap().get(0).unwrap().to_scalar::<f32>().unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn unpatchify_takes_the_width_offset_from_the_middle_factor() {
        // One colour, p = 2: channel r·2 + q goes to row q, column r.
        let x = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (1, 4, 1, 1), &Device::Cpu).unwrap();
        let y = unpatchify(&x, 2).unwrap();
        // Rows: (q = 0: r = 0, 1 → channels 0, 2), (q = 1 → channels 1, 3).
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), [0.0, 2.0, 1.0, 3.0]);
    }

    /// Frames that repeat a latent cell across and down, in a frame of
    /// structure, as a decode that failed part-way draws them, are refused,
    /// and so are frames one flat colour; a clip with a frame that does not
    /// repeat passes, and so does a plain one with a level's grain.
    #[test]
    fn a_lattice_is_refused_and_a_picture_is_not() {
        let (t, h, w) = (3, 384, 448);
        // Xorshift: uniform enough in [0, n) for grain.
        let mut state = 5u64;
        let mut grain = |n: f32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1000) as f32 / 1000.0 * n
        };
        let mut frames = vec![0f32; t * h * w * 3];
        for (i, v) in frames.iter_mut().enumerate() {
            let (y, x, c) = (i / (w * 3) % h, i / 3 % w, i % 3);
            let edge = x.min(y).min(w - 1 - x).min(h - 1 - y) < 100;
            *v = if edge { grain(255.0) } else { ((x % CELL) * 5 + (y % CELL) * 3 + c * 40) as f32 % 255.0 + grain(0.4) };
        }
        let e = check_lattice(&frames, w, h).unwrap_err().to_string();
        assert!(e.contains("repeats itself every 32 pixels"), "{e}");
        assert!(check_lattice(&vec![77.0; t * h * w * 3], w, h).is_err());
        assert!(check_lattice(&vec![77.0; t * 300 * w * 3], w, 300).is_ok(), "too short to judge");

        // One frame of the three a picture, or all a black with a level's
        // grain, and the clip passes.
        let mut one = frames.clone();
        let f = h * w * 3;
        for v in &mut one[f..2 * f] {
            *v = grain(255.0);
        }
        check_lattice(&one, w, h).unwrap();
        let black: Vec<f32> = (0..t * h * w * 3).map(|_| grain(2.0) - 1.0).collect();
        check_lattice(&black, w, h).unwrap();

        // Through `to_video`, frames first in [0, 1].
        let x = Tensor::from_vec(frames, (t, h, w, 3), &Device::Cpu).unwrap().permute((0, 3, 1, 2)).unwrap();
        let x = (x.contiguous().unwrap() / 255.0).unwrap();
        assert!(to_video(&x, 24).err().map(|e| e.to_string()).unwrap_or_default().contains("pattern"));
    }

    /// The conv decoder with any one of its stages left as zeros, as a
    /// failed Metal command buffer leaves it, is refused at that stage
    /// ([`check_stage`]); the decode that did not fail passes. At 768×512
    /// and 9 frames in bf16 on Metal, as the pipeline runs it, from a latent
    /// of noise: what a failure draws does not depend on the latent, only
    /// on the size.
    ///
    ///     cargo test --release -p kvad-gpu ltx_vae::tests::a_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_decode_that_fails_part_way_is_refused() {
        let path = crate::image::local_file(super::super::LTX_REPO, FILE).expect("fetch LTX-2.5 first");
        let device = Device::new_metal(0).unwrap();
        let dec = VideoDecoder::load(&path, &device, DType::BF16).unwrap();
        let z = crate::image::nn::noise(5, &[128, 2, 16, 24], &device, DType::F32).unwrap();

        let frames = dec.decode(&z).unwrap().to_device(&Device::Cpu).unwrap();
        to_video(&frames, 24).unwrap();
        for stage in 0..dec.stages() {
            let e = dec.decode_failing(&z, Some(stage)).err().map(|e| e.to_string()).unwrap_or_default();
            eprintln!("stage {stage}: {e}");
            assert!(e.starts_with(&format!("stage {stage} of the conv decoder came back zeros")), "stage {stage}: {e}");
        }
    }
}
