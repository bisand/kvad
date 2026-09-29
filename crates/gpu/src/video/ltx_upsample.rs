//! LTX-2.5's latent upsamplers: the spatial one, a stage-1 latent to one
//! twice as wide and twice as high, for stage 2 to refine; and the temporal
//! one, a latent to one with twice the frames, for DFR's temporal rounds to
//! refine (`docs/video-plan.md`, "The temporal upsampler").
//!
//! ```text
//! latent [128, F, h, w] ─ un-normalise ─ conv3d 128 → mid ─ group norm ─ silu
//!   ─ 4 residual blocks ─ the upsampling ─ 4 residual blocks
//!   ─ conv3d mid → 128 ─ normalise
//!
//! spatial, mid 1024: per frame, conv2d 1024 → 4096, pixel shuffle ×2
//!   ─ [128, F, 2h, 2w]
//! temporal, mid 512: conv3d 512 → 1024, each frame's channel pairs to two
//!   frames, the first frame dropped ─ [128, 2F − 1, h, w]
//! ```
//!
//! The temporal one drops the first upsampled frame because a latent's first
//! frame stands for one pixel frame, where each after it stands for eight:
//! `8(F − 1) + 1` pixel frames become `16(F − 1) + 1`, twice the frame rate.
//!
//! A residual block is `silu(x + gn(conv(silu(gn(conv(x))))))`: the last
//! activation comes after the sum. Two things differ from the VAE decoder,
//! whose conv3d this borrows:
//!
//! - **The padding is zeros in time as well as space.** These are plain
//!   `Conv3d(padding=1)`, not the decoder's repeated edge frames.
//! - **A group norm spans every frame.** PyTorch's `GroupNorm` on a 5D
//!   tensor takes each group's statistics over its channels and the whole
//!   `F × H × W` volume, not frame by frame.
//!
//! The latents it takes and gives are normalised by the video VAE's
//! per-channel statistics; it works between them un-normalised.

use super::conv3d::{Conv3d, Time};
use super::metadata;
use crate::common::{settle, Loader, Reader};
use crate::image::nn::{Conv2d, Ctx};
use crate::image::{finish, open};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor, D};
use kvad::serde_json::Value;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The spatial ×2 upsampler's file in [`super::LTX_REPO`].
pub const FILE: &str = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";

/// The temporal ×2 upsampler's, which only DFR reads.
pub const TEMPORAL_FILE: &str = "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors";

/// How an upsampler makes its latent bigger.
enum Up {
    /// Per frame, 2D: to four times the channels, which the pixel shuffle
    /// turns into twice the rows and columns.
    Space(Conv2d),
    /// 3D, zeros in time: to twice the channels, which become twice the
    /// frames, less the first.
    Time(Conv3d),
}

/// PyTorch's `GroupNorm` default.
const EPS: f64 = 1e-5;

/// Group norm over a whole clip, on frames-first `[T, C, H, W]`: each of
/// `groups` groups is normalised over its channels and every frame and
/// pixel, then scaled and shifted per channel. In f32.
struct ClipNorm {
    groups: usize,
    w: Tensor,
    b: Tensor,
}

impl ClipNorm {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, channels: usize) -> Res<Self> {
        let r = r.pp(name);
        let get = |n: &str| -> Res<Tensor> { Ok(cx.get(&r, channels, n)?.to_dtype(DType::F32)?.reshape((1, channels, 1, 1))?) };
        Ok(ClipNorm { groups: 32, w: get("weight")?, b: get("bias")? })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (t, c, h, w) = x.dims4()?;
        let dtype = x.dtype();
        // Channels first, so that a group is one contiguous run of numbers.
        let g = x.to_dtype(DType::F32)?.permute((1, 0, 2, 3))?.contiguous()?.reshape((self.groups, (c / self.groups) * t * h * w))?;
        let g = g.broadcast_sub(&g.mean_keepdim(D::Minus1)?)?;
        let g = g.broadcast_div(&(g.sqr()?.mean_keepdim(D::Minus1)? + EPS)?.sqrt()?)?;
        let g = g.reshape((c, t, h, w))?.permute((1, 0, 2, 3))?;
        g.broadcast_mul(&self.w)?.broadcast_add(&self.b)?.to_dtype(dtype)
    }
}

struct ResBlock {
    conv1: Conv3d,
    norm1: ClipNorm,
    conv2: Conv3d,
    norm2: ClipNorm,
}

impl ResBlock {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, c: usize) -> Res<Self> {
        Ok(ResBlock {
            conv1: Conv3d::load(cx, r, "conv1", c, c)?.padded(Time::Zeros),
            norm1: ClipNorm::load(cx, r, "norm1", c)?,
            conv2: Conv3d::load(cx, r, "conv2", c, c)?.padded(Time::Zeros),
            norm2: ClipNorm::load(cx, r, "norm2", c)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.norm1.forward(&self.conv1.forward(x)?)?.silu()?;
        let h = self.norm2.forward(&self.conv2.forward(&h)?)?;
        (h + x)?.silu()
    }
}

pub struct Upsampler {
    initial: Conv3d,
    norm: ClipNorm,
    res: Vec<ResBlock>,
    up: Up,
    post: Vec<ResBlock>,
    last: Conv3d,
    /// The VAE's statistics, `[1, 128, 1, 1]` in f32.
    mean: Tensor,
    std: Tensor,
    dtype: DType,
    params: usize,
}

impl Upsampler {
    /// The upsampler in the file at `path`, spatial or temporal as its
    /// config says, with the statistics of the VAE file at `vae`, computing
    /// in `dtype` on `device`.
    pub fn load(path: &Path, vae: &Path, device: &Device, dtype: DType) -> Res<Self> {
        let config = metadata(path, "config")?;
        let temporal = config["temporal_upsample"] == Value::Bool(true);
        // The temporal file says `rational_resampler: true`, which only the
        // spatial branch reads: the reference upsamples in time alone when
        // `spatial_upsample` is false.
        let want: &[(&str, Value)] = match temporal {
            false => &[("spatial_upsample", Value::Bool(true)), ("rational_resampler", Value::Bool(false)), ("spatial_scale", Value::from(2.0))],
            true => &[("spatial_upsample", Value::Bool(false))],
        };
        for (key, want) in [("_class_name", Value::from("LatentUpsampler")), ("dims", Value::from(3))].iter().chain(want) {
            let (key, want) = (*key, want.clone());
            if config[key] != want {
                return Err(format!("{}: `{key}` is {}, and this upsampler is written for {want}", path.display(), config[key]).into());
            }
        }
        let num = |k: &str| config[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("upsampler config: no `{k}`"));
        let (cin, mid, blocks) = (num("in_channels")?, num("mid_channels")?, num("num_blocks_per_stage")?);

        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let paths = [path.to_path_buf()];
        let r = open(&paths, DType::BF16)?;
        let stack = |prefix: &str| -> Res<Vec<ResBlock>> { (0..blocks).map(|i| ResBlock::load(&cx, &r.pp(format!("{prefix}.{i}")), mid)).collect() };
        let (mean, std) = super::ltx_vae::statistics(vae)?;
        let stat = |t: Tensor| -> Res<Tensor> { Ok(t.reshape((1, cin, 1, 1))?.to_device(device)?) };
        Ok(Upsampler {
            initial: Conv3d::load(&cx, &r, "initial_conv", cin, mid)?.padded(Time::Zeros),
            norm: ClipNorm::load(&cx, &r, "initial_norm", mid)?,
            res: stack("res_blocks")?,
            up: match temporal {
                false => Up::Space(Conv2d::load(&cx, &r, "upsampler.0", (mid, 4 * mid, 3), 1)?),
                true => Up::Time(Conv3d::load(&cx, &r, "upsampler.0", mid, 2 * mid)?.padded(Time::Zeros)),
            },
            post: stack("post_upsample_res_blocks")?,
            last: Conv3d::load(&cx, &r, "final_conv", mid, cin)?.padded(Time::Zeros),
            mean: stat(mean)?,
            std: stat(std)?,
            dtype,
            params: finish("LTX latent upsampler", &paths, &r)?,
        })
    }

    pub fn params(&self) -> usize {
        self.params
    }

    /// Whether it upsamples in time rather than space.
    pub fn temporal(&self) -> bool {
        matches!(self.up, Up::Time(_))
    }

    /// A normalised latent `[C, F, h, w]` to `[C, F, 2h, 2w]`, or in time to
    /// `[C, 2F − 1, h, w]`, normalised again, in f32.
    pub fn forward(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        // Frames first from here on, as the convolutions want them.
        let z = latent.to_dtype(DType::F32)?.permute((1, 0, 2, 3))?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?.to_dtype(self.dtype)?;
        // Synchronised after every block: candle's Metal pool frees a dropped
        // buffer only then, and without it all 18 steps' temporaries stayed
        // allocated to the end.
        let step = |x: candle_core::Result<Tensor>| -> candle_core::Result<Tensor> {
            let x = x?;
            settle(x.device())?;
            Ok(x)
        };
        let mut x = step(self.norm.forward(&self.initial.forward(&z)?)?.silu())?;
        for b in &self.res {
            x = step(b.forward(&x))?;
        }
        x = step(match &self.up {
            Up::Space(c) => shuffle(&c.forward(&x)?),
            Up::Time(c) => {
                let y = shuffle_time(&c.forward(&x)?)?;
                y.narrow(0, 1, y.dim(0)? - 1)
            }
        })?;
        for b in &self.post {
            x = step(b.forward(&x))?;
        }
        let z = self.last.forward(&x)?.to_dtype(DType::F32)?;
        z.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?.permute((1, 0, 2, 3))?.contiguous()
    }
}

/// Pixel shuffle ×2: `[T, 4C, H, W]` to `[T, C, 2H, 2W]`, channel
/// `c·4 + 2·p + q` going to row `2y + p` and column `2x + q`.
fn shuffle(x: &Tensor) -> candle_core::Result<Tensor> {
    let (t, c4, h, w) = x.dims4()?;
    let c = c4 / 4;
    x.reshape((t, c, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.contiguous()?.reshape((t, c, 2 * h, 2 * w))
}

/// Pixel shuffle ×2 in time: `[T, 2C, H, W]` to `[2T, C, H, W]`, channel
/// `c·2 + p` of frame `t` going to frame `2t + p`.
fn shuffle_time(x: &Tensor) -> candle_core::Result<Tensor> {
    let (t, c2, h, w) = x.dims4()?;
    let c = c2 / 2;
    x.reshape((t, c, 2, h, w))?.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((2 * t, c, h, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_puts_each_channel_at_its_offset() {
        let dev = Device::Cpu;
        // One frame, one output channel, a 1×2 grid: channel 2p + q of cell
        // (0, x) lands at (p, 2x + q).
        let x = Tensor::arange(0f32, 8.0, &dev).unwrap().reshape((1, 4, 1, 2)).unwrap();
        let y = shuffle(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Input [ch][x]: ch0 = (0, 1), ch1 = (2, 3), ch2 = (4, 5), ch3 = (6, 7).
        // Row 0: (x0,q0)=ch0[0], (x0,q1)=ch1[0], (x1,q0)=ch0[1], (x1,q1)=ch1[1].
        assert_eq!(y, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn shuffle_time_puts_each_channel_pair_in_its_two_frames() {
        let dev = Device::Cpu;
        // Two frames of four channels, one pixel: out channel c of frame
        // 2t + p is in channel 2c + p of frame t.
        let x = Tensor::arange(0f32, 8.0, &dev).unwrap().reshape((2, 4, 1, 1)).unwrap();
        let y = shuffle_time(&x).unwrap();
        assert_eq!(y.dims(), &[4, 2, 1, 1]);
        assert_eq!(y.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn clip_norm_takes_its_statistics_over_every_frame() {
        let dev = Device::Cpu;
        // Two frames at different levels, one group: normalised together, the
        // first frame stays below the second rather than each going to zero
        // mean on its own.
        let x = Tensor::cat(&[Tensor::full(1f32, (1, 32, 1, 1), &dev).unwrap(), Tensor::full(3f32, (1, 32, 1, 1), &dev).unwrap()], 0).unwrap();
        let n = ClipNorm { groups: 32, w: Tensor::ones((1, 32, 1, 1), DType::F32, &dev).unwrap(), b: Tensor::zeros((1, 32, 1, 1), DType::F32, &dev).unwrap() };
        let y = n.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((y[0] + 1.0).abs() < 1e-3 && (y[32] - 1.0).abs() < 1e-3, "{:?}", &y[..2]);
    }
}
