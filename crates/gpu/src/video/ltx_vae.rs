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

use super::conv3d::{pixel_norm, Conv3d};
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
            x = match block {
                Block::Res(res) => {
                    for (c1, c2) in res {
                        let h = c1.forward(&pixel_norm(&x, EPS)?.silu()?)?;
                        let h = c2.forward(&pixel_norm(&h, EPS)?.silu()?)?;
                        x = (x + h)?;
                    }
                    x
                }
                Block::Up { conv, stride, out } => depth_to_space(&conv.forward(&x)?, *stride, *out)?,
            };
        }
        let x = self.conv_out.forward(&pixel_norm(&x, EPS)?.silu()?)?;
        let x = unpatchify(&x, self.patch)?.to_dtype(DType::F32)?;
        ((x + 1.0)? * 0.5)?.clamp(0f32, 1f32)
    }
}

/// `[T, (c·p₁·p₂·p₃), H, W]` to `[T·p₁ (− 1), c, H·p₂, W·p₃]`: each group of
/// channels unfolded into a block of frames, rows and columns.
///
/// The channel index is `((c·p₁ + i)·p₂ + j)·p₃ + k`, as the reference's
/// `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)` has it. When time
/// doubles, the first frame of the result is dropped.
fn depth_to_space(x: &Tensor, [p1, p2, p3]: [usize; 3], c: usize) -> candle_core::Result<Tensor> {
    let (t, _, h, w) = x.dims4()?;
    let x = x
        .reshape(&[t, c, p1, p2, p3, h, w][..])?
        // [t, p1, c, h, p2, w, p3]
        .permute(&[0, 2, 1, 5, 3, 6, 4][..])?
        .contiguous()?
        .reshape((t * p1, c, h * p2, w * p3))?;
    match p1 {
        2 => x.narrow(0, 1, t * p1 - 1),
        _ => Ok(x),
    }
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
