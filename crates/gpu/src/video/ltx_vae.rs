//! LTX-2.5's convolutional video decoder: 128-channel latents to frames.
//!
//! `vae/ltx-2.5-video-vae-conv-bf16.safetensors`, a `CausalVideoAutoencoder`
//! whose decoder is, despite the name, not causal. Its config is in the
//! file's metadata; `docs/video-plan.md` has the stack written out.
//!
//! A latent is `[128, F, h, w]` and decodes to `8(F − 1) + 1` frames of
//! `32h × 32w`. Three ideas make that up:
//!
//! - **3×3×3 convolutions** at every level ([`Conv3d`], built from 2D ones).
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

use super::conv3d::{norm_silu, Conv3d};
use super::metadata;
use crate::common::Loader;
use crate::image::nn::Ctx;
use crate::image::{finish, open};
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
    /// 32h, 32w]` in `[0, 1]`, as f32.
    pub fn decode(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        // Frames first from here on: the frames are every convolution's batch.
        let z = latent.permute((1, 0, 2, 3))?.to_dtype(DType::F32)?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let mut x = self.conv_in.forward(&z.to_dtype(self.dtype)?)?;
        for block in &self.blocks {
            match block {
                Block::Res(res) => {
                    for (c1, c2) in res {
                        x = residual(&x, c1, c2)?;
                        // candle's pool lets go of what a step dropped only
                        // when the device is synchronised.
                        x.device().synchronize()?;
                    }
                }
                Block::Up { conv, stride, out } => {
                    x = up(&x, conv, *stride, *out)?;
                    x.device().synchronize()?;
                }
            }
        }
        let p = self.patch;
        let (t, _, h, w) = x.dims4()?;
        let frames = Tensor::zeros((t, 3, h * p, w * p), DType::F32, x.device())?;
        for (f, n) in chunks(&x) {
            let y = self.conv_out.frames(&norm_silu(&halo(&x, f, n, 1)?, EPS)?, (f.saturating_sub(1), t), (f, f + n))?;
            let y = ((unpatchify(&y, p)?.to_dtype(DType::F32)? + 1.0)? * 0.5)?.clamp(0f32, 1f32)?;
            frames.slice_set(&y, 0, f)?;
        }
        Ok(frames)
    }
}

/// The most elements of a layer's input [`chunks`] takes at once: 256 M,
/// 512 MB in bf16. At 1536×1024 the decoder's last stage is 3 GB a tensor,
/// and a residual step done whole kept five of them alive; the whole decode
/// ran out of memory at 77 GB.
const BUDGET: usize = 1 << 28;

/// `(first frame, frames)` for each chunk of `x` a step works through.
fn chunks(x: &Tensor) -> Vec<(usize, usize)> {
    let t = x.dim(0).unwrap_or(0);
    let n = (BUDGET / (x.elem_count() / t.max(1)).max(1)).clamp(1, t.max(1));
    (0..t).step_by(n).map(|f| (f, n.min(t - f))).collect()
}

/// Frames `f − k .. f + n + k` of `x`, cut off at its ends: a chunk and the
/// `k` frames either side that `k` convolutions in a row read.
fn halo(x: &Tensor, f: usize, n: usize, k: usize) -> candle_core::Result<Tensor> {
    let (lo, hi) = (f.saturating_sub(k), (f + n + k).min(x.dim(0)?));
    x.narrow(0, lo, hi - lo)
}

/// One residual step, `x + conv(silu(pn(conv(silu(pn(x))))))`, a chunk of
/// frames at a time.
///
/// A chunk's frames `f .. f + n` need the first convolution's output one
/// frame beyond them each way, and that needs the input two frames beyond.
/// Those halo frames are computed twice, once for each chunk that reads
/// them: the price of never holding the steps in between at full size. A
/// stage small enough for one chunk pays nothing.
fn residual(x: &Tensor, c1: &Conv3d, c2: &Conv3d) -> candle_core::Result<Tensor> {
    let t = x.dim(0)?;
    let parts = chunks(x);
    if parts.len() == 1 {
        let h = c1.forward(&norm_silu(x, EPS)?)?;
        return x + c2.forward(&norm_silu(&h, EPS)?)?;
    }
    let out = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
    for (f, n) in parts {
        let (a, b) = (f.saturating_sub(1), (f + n + 1).min(t));
        let h = c1.frames(&norm_silu(&halo(x, f, n, 2)?, EPS)?, (f.saturating_sub(2), t), (a, b))?;
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
        let y = unfold(&conv.frames(&halo(x, f, n, 1)?, (f.saturating_sub(1), t), (f, f + n))?, stride, c)?;
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
fn unpatchify(x: &Tensor, p: usize) -> candle_core::Result<Tensor> {
    let (t, c, h, w) = x.dims4()?;
    let c = c / (p * p);
    x.reshape(&[t, c, p, p, h, w][..])?
        // [t, c, r, q, h, w] → [t, c, h, q, w, r]
        .permute(&[0, 1, 4, 3, 5, 2][..])?
        .contiguous()?
        .reshape((t, c, h * p, w * p))
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
    let rgb = x.flatten_all()?.to_vec1::<f32>()?.into_iter().map(|v| v as u8).collect();
    Ok(kvad::video::Video { width: w, height: h, fps, rgb })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn unpatchify_takes_the_width_offset_from_the_middle_factor() {
        // One colour, p = 2: channel r·2 + q goes to row q, column r.
        let x = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (1, 4, 1, 1), &Device::Cpu).unwrap();
        let y = unpatchify(&x, 2).unwrap();
        // Rows: (q = 0: r = 0, 1 → channels 0, 2), (q = 1 → channels 1, 3).
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), [0.0, 2.0, 1.0, 3.0]);
    }
}
