//! The VAE: from a latent to pixels, and from pixels to a latent.
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
//! The encoding half is never used to *make* an image, and the pipelines skip
//! it by name. It is here for what starts from a picture: training, where
//! every picture has to become a latent before the denoiser can learn from
//! it, and editing. It is the decoder run backwards, level for level: a
//! strided convolution where the decoder upsamples, and at the end not one
//! latent but two, a mean and a log-variance ([`Posterior`]).

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
    /// What was subtracted before that scaling, and so is added back: FLUX's
    /// VAE centres its latents, SDXL's does not.
    pub(crate) shift: f64,
    /// Whether a 1×1 convolution sits between the latent and the decoder.
    /// SDXL's has one; FLUX's was trained without it.
    pub(crate) post_quant: bool,
    /// And its twin between the encoder and the latent.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) quant: bool,
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
            // Left out, as SD 1.5's config leaves it, it is diffusers'
            // default for `AutoencoderKL`, which is SD 1.5's own.
            scaling: v.get("scaling_factor").and_then(Value::as_f64).unwrap_or(0.18215),
            shift: v.get("shift_factor").and_then(Value::as_f64).unwrap_or(0.0),
            post_quant: v.get("use_post_quant_conv").and_then(Value::as_bool).unwrap_or(true),
            quant: v.get("use_quant_conv").and_then(Value::as_bool).unwrap_or(true),
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

/// The mid block, which the encoder and decoder share the shape of.
fn load_mid(cx: &Ctx<'_>, m: &Reader<'_>, g: usize, top: usize) -> Res<(Resnet, Attn, Resnet)> {
    let a = m.pp("attentions.0");
    Ok((
        Resnet::load(cx, &m.pp("resnets.0"), g, top, top)?,
        Attn {
            norm: GroupNorm::load(cx, &a, "group_norm", top, g, EPS)?,
            q: Linear::load(cx, &a, "to_q", top, top, true)?,
            k: Linear::load(cx, &a, "to_k", top, top, true)?,
            v: Linear::load(cx, &a, "to_v", top, top, true)?,
            out: Linear::load(cx, &a, "to_out.0", top, top, true)?,
        },
        Resnet::load(cx, &m.pp("resnets.1"), g, top, top)?,
    ))
}

pub(crate) struct Decoder {
    cfg: VaeConfig,
    post_quant: Option<Conv2d>,
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
        let mid = load_mid(cx, &d.pp("mid_block"), g, top)?;

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
            post_quant: match cfg.post_quant {
                true => Some(Conv2d::load(cx, r, "post_quant_conv", (cfg.latent, cfg.latent, 1), 1)?),
                false => None,
            },
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
        self.decode_failing(latent, None)
    }

    /// [`Self::decode`], with the stage numbered `fail` counted from
    /// `conv_in` (0) to `conv_out` (the last) left as zeros, as a Metal
    /// command buffer that failed leaves them, and the rest run on them.
    pub(crate) fn decode_failing(&self, latent: &Tensor, fail: Option<usize>) -> candle_core::Result<Tensor> {
        let mut stage = 0;
        let mut done = |h: Tensor| -> candle_core::Result<Tensor> {
            stage += 1;
            match fail == Some(stage - 1) {
                true => h.zeros_like(),
                false => Ok(h),
            }
        };
        let z = ((latent / self.cfg.scaling)? + self.cfg.shift)?;
        let z = match &self.post_quant {
            Some(conv) => conv.forward(&z)?,
            None => z,
        };
        let mut h = done(self.conv_in.forward(&z)?)?;
        h = done(self.mid.0.forward(&h)?)?;
        h = done(self.mid.1.forward(&h)?)?;
        h = done(self.mid.2.forward(&h)?)?;
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
    pub(crate) fn stages(&self) -> usize {
        4 + self.up.iter().map(|(r, u)| r.len() + u.is_some() as usize).sum::<usize>() + 2
    }
}

// ---------------------------------------------------------------------------
// The encoder
// ---------------------------------------------------------------------------

/// What the encoder says about a picture: not one latent but a Gaussian over
/// them, `N(mean, exp(logvar))` in every number, in the VAE's own units.
///
/// Training draws from it, which is what the VAE was trained to make the
/// decoder robust to; editing usually takes the mean, which is the likeliest
/// latent and the same every time. Both are the caller's to choose, and
/// [`Encoder::to_denoiser`] is the step both need after.
///
/// Nothing outside the tests calls the encoder yet: its callers are training
/// (#75) and editing (#43), and it is here first so that both start from an
/// encoder already checked against diffusers'.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Posterior {
    pub(crate) mean: Tensor,
    /// Clamped to [-30, 20], as diffusers does, so that `exp` stays finite.
    pub(crate) logvar: Tensor,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Posterior {
    /// `mean + exp(logvar / 2) · noise`, where `noise` is standard normal and
    /// the mean's shape.
    pub(crate) fn sample(&self, noise: &Tensor) -> candle_core::Result<Tensor> {
        self.mean.broadcast_add(&(self.logvar.affine(0.5, 0.0)?.exp()? * noise)?)
    }
}

/// The encoding half.
///
/// It works at full resolution, as the decoder does, and so has the
/// decoder's appetite for memory, at a bit over half the size. Peak
/// footprint of one encode and one decode alone, in the precision each
/// pipeline runs its VAE in, on an M5 Pro:
///
/// | | encode | decode |
/// |---|---|---|
/// | SDXL, f16, 512² | 1.1 s, 3.2 GB | 2.4 s, 5.7 GB |
/// | SDXL, f16, 1024² | 6.2 s, 12.0 GB | 9.9 s, 21.3 GB |
/// | FLUX, bf16, 1024² | 5.0 s, 11.9 GB | 9.5 s, 21.3 GB |
///
/// So wherever a picture can be decoded it can be encoded, and it has no
/// tiling. `tests::one_encode` measures it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Encoder {
    cfg: VaeConfig,
    conv_in: Conv2d,
    /// Each level's resnets, then a stride-2 convolution that halves the grid
    /// on every level but the last.
    down: Vec<(Vec<Resnet>, Option<Conv2d>)>,
    mid: (Resnet, Attn, Resnet),
    norm_out: GroupNorm,
    /// To twice the latent's channels: the mean's, then the log-variance's.
    conv_out: Conv2d,
    quant: Option<Conv2d>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Encoder {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: VaeConfig) -> Res<Self> {
        r.skip_under("decoder.");
        r.skip_under("post_quant_conv.");
        let g = cfg.groups;
        let e = r.pp("encoder");
        let mut down = Vec::with_capacity(cfg.channels.len());
        let mut prev = cfg.channels[0];
        for (i, &cout) in cfg.channels.iter().enumerate() {
            let b = e.pp(format!("down_blocks.{i}"));
            let resnets = (0..cfg.layers_per_block)
                .map(|j| Resnet::load(cx, &b.pp(format!("resnets.{j}")), g, if j == 0 { prev } else { cout }, cout))
                .collect::<Res<Vec<_>>>()?;
            let downsample = match i + 1 < cfg.channels.len() {
                true => Some(Conv2d::load(cx, &b, "downsamplers.0.conv", (cout, cout, 3), 2)?.unpadded()),
                false => None,
            };
            down.push((resnets, downsample));
            prev = cout;
        }
        let top = *cfg.channels.last().unwrap();
        let two = 2 * cfg.latent;
        Ok(Encoder {
            conv_in: Conv2d::load(cx, &e, "conv_in", (3, cfg.channels[0], 3), 1)?,
            mid: load_mid(cx, &e.pp("mid_block"), g, top)?,
            norm_out: GroupNorm::load(cx, &e, "conv_norm_out", top, g, EPS)?,
            conv_out: Conv2d::load(cx, &e, "conv_out", (top, two, 3), 1)?,
            quant: match cfg.quant {
                true => Some(Conv2d::load(cx, r, "quant_conv", (two, two, 1), 1)?),
                false => None,
            },
            down,
            cfg,
        })
    }

    pub(crate) fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// `[1, 3, H, W]` in `[−1, 1]`, `H` and `W` multiples of
    /// [`VaeConfig::factor`], to the Gaussian over its latents.
    pub(crate) fn encode(&self, image: &Tensor) -> candle_core::Result<Posterior> {
        let mut h = self.conv_in.forward(image)?;
        for (resnets, downsample) in &self.down {
            for r in resnets {
                h = r.forward(&h)?;
            }
            if let Some(conv) = downsample {
                // One column on the right and one row at the bottom, then a
                // stride of 2 with no padding of the convolution's own: the
                // grid halves, and the kernel's window starts on the first
                // pixel rather than half a pixel before it. diffusers'
                // `Downsample2D` with `padding=0`.
                h = conv.forward(&h.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?)?;
            }
        }
        h = self.mid.0.forward(&h)?;
        h = self.mid.1.forward(&h)?;
        h = self.mid.2.forward(&h)?;
        let h = self.conv_out.forward(&self.norm_out.forward(&h)?.silu()?)?;
        let h = match &self.quant {
            Some(conv) => conv.forward(&h)?,
            None => h,
        };
        let c = self.cfg.latent;
        Ok(Posterior { mean: h.narrow(1, 0, c)?, logvar: h.narrow(1, c, c)?.clamp(-30.0, 20.0)? })
    }

    /// A latent in the VAE's units to the denoiser's: shifted and scaled to
    /// unit variance, the inverse of what [`Decoder::decode`] does first.
    pub(crate) fn to_denoiser(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        (z - self.cfg.shift)? * self.cfg.scaling
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use super::super::{open, read_json};
    use crate::common::Loader;
    use crate::qcache::Vault;
    use candle_core::{DType, Device};
    use kvad::weights::{fetch_file, Watcher};

    /// A VAE's config and weights, by the name `scripts/vae-fixtures.py`
    /// writes its fixtures under, and the precision its pipeline runs it in.
    fn vae(name: &str) -> (Value, Vec<std::path::PathBuf>, DType) {
        let w = Watcher::none();
        let (repo, dir, dtype) = match name {
            "sdxl" => (super::super::sdxl::VAE_REPO, "", DType::F16),
            "flux" => ("black-forest-labs/FLUX.1-schnell", "vae/", DType::BF16),
            "sd15" => (super::super::sd15::REPO, "vae/", DType::F16),
            _ => unreachable!(),
        };
        let config = read_json(&fetch_file(repo, &format!("{dir}config.json"), &w).unwrap()).unwrap();
        let weights = match dir {
            "" => fetch_file(repo, "diffusion_pytorch_model.safetensors", &w).unwrap(),
            _ => super::super::sdxl::weights(repo, "vae", "diffusion_pytorch_model", &w).unwrap(),
        };
        (config, vec![weights], dtype)
    }

    fn load(name: &str, device: &Device, dtype: DType) -> (Encoder, Decoder) {
        let (config, paths, _) = vae(name);
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let cfg = VaeConfig::from_json(&config).unwrap();
        let enc = Encoder::load(&cx, &open(&paths, dtype).unwrap(), cfg.clone()).unwrap();
        let dec = Decoder::load(&cx, &open(&paths, dtype).unwrap(), cfg).unwrap();
        (enc, dec)
    }

    /// A draw is the mean plus the standard deviation times the noise, and
    /// the standard deviation is `exp(logvar / 2)`: a log-variance of 2 is a
    /// standard deviation of e, and one of 0 adds the noise as it is.
    #[test]
    fn a_sample_is_the_mean_plus_the_spread_times_the_noise() {
        let dev = Device::Cpu;
        let p = Posterior {
            mean: Tensor::new(&[[1.0f32, -2.0, 0.5]], &dev).unwrap(),
            logvar: Tensor::new(&[[0.0f32, 2.0, -30.0]], &dev).unwrap(),
        };
        let noise = Tensor::new(&[[0.5f32, -1.0, 3.0]], &dev).unwrap();
        let got: Vec<f32> = p.sample(&noise).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let want = [1.0 + 0.5, -2.0 - std::f32::consts::E, 0.5 + (-15.0f32).exp() * 3.0];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} against {want:?}");
        }
    }

    /// Peak signal-to-noise ratio of `a` against `b`, both in [-1, 1]: the
    /// range is 2 wide, so the peak's square is 4.
    pub(crate) fn psnr(a: &Tensor, b: &Tensor) -> f64 {
        let mse = (a - b).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
        10.0 * (4.0 / mse).log10()
    }

    /// How far `got` is from `want`: the largest difference as a share of
    /// the largest value, and the root-mean-square difference as a share of
    /// the root-mean-square value.
    pub(crate) fn gap(got: &Tensor, want: &Tensor) -> (f32, f32) {
        let got = got.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().squeeze(0).unwrap();
        let diff = (got - want).unwrap();
        let max = |t: &Tensor| t.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let rms = |t: &Tensor| t.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        (max(&diff) / max(want), rms(&diff) / rms(want))
    }

    /// kvad's encoder against diffusers', on the same picture and weights:
    /// the mean, the log-variance, and the round trip through both halves.
    /// In f32 on Metal, and then in the precision the pipeline runs the VAE
    /// in, for the drift that costs.
    ///
    ///     KVAD_VAE_FIXTURES=/tmp/vae-fx cargo test --release -p kvad-gpu vae::tests -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_encoder_agrees_with_diffusers() {
        let dir = std::env::var("KVAD_VAE_FIXTURES").expect("KVAD_VAE_FIXTURES names the fixtures' directory");
        let device = Device::new_metal(0).unwrap();
        let mut checked = 0;
        for name in ["sdxl", "flux", "sd15"] {
            let Ok(fx) = candle_core::safetensors::load(format!("{dir}/{name}.safetensors"), &Device::Cpu) else {
                eprintln!("{name}: no fixtures");
                continue;
            };
            let (image, mean, logvar, decoded) = (&fx["image"], &fx["mean"], &fx["logvar"], &fx["decoded"]);
            let theirs = psnr(decoded, image);
            let (_, _, pipeline) = vae(name);
            for dtype in [DType::F32, pipeline] {
                let (enc, dec) = load(name, &device, dtype);
                let x = image.unsqueeze(0).unwrap().to_device(&device).unwrap().to_dtype(dtype).unwrap();
                let p = enc.encode(&x).unwrap();
                let back = dec.decode(&enc.to_denoiser(&p.mean).unwrap()).unwrap();
                let back = back.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().squeeze(0).unwrap();
                let (dm, dl, same, ours) = (gap(&p.mean, mean), gap(&p.logvar, logvar), psnr(&back, decoded), psnr(&back, image));
                // In half precision the reference drifts too. Measured, its
                // drift and ours are the same size (SDXL f16 1.31e-2 and
                // 1.47e-2 rms, FLUX bf16 3.76e-2 and 4.18e-2, SD 1.5 f16
                // 1.86e-3 and 1.74e-3), so ours is held to half again its own.
                let own = match (dtype, fx.get("mean_half")) {
                    (DType::F32, _) => None,
                    (_, Some(half)) => Some(gap(&half.unsqueeze(0).unwrap(), mean)),
                    (_, None) => panic!("{name}: no `mean_half` in the fixtures; make them on a Mac, where MPS runs it"),
                };
                if let Some(own) = own {
                    eprintln!("{name} diffusers' own {dtype:?} mean is off by {:.2e} at most, {:.2e} rms", own.0, own.1);
                }
                eprintln!(
                    "{name} {dtype:?}: mean off by {:.2e} at most, {:.2e} rms; logvar {:.2e}, {:.2e}; decoded {same:.1} dB from theirs; round trip {ours:.2} dB, diffusers' {theirs:.2}",
                    dm.0, dm.1, dl.0, dl.1
                );
                // The encoder is what is new here, and held to its own
                // numbers; the decode is measured as the decoder's own test
                // measures it, as a picture.
                let (close, least, db) = match own {
                    None => (1e-3, 60.0, 0.01),
                    Some(own) => (1.5 * own.1, 30.0, 0.5),
                };
                assert!(dm.1 < close, "{name} {dtype:?}: the mean is {:.2e} rms from diffusers'", dm.1);
                if own.is_none() {
                    assert!(dl.1 < close, "{name}: the log-variance is {:.2e} rms from diffusers'", dl.1);
                }
                assert!(same > least, "{name} {dtype:?}: the decode of the mean is {same:.1} dB from diffusers'");
                assert!((ours - theirs).abs() < db, "{name} {dtype:?}: {ours:.2} dB against diffusers' {theirs:.2}");
            }
            checked += 1;
        }
        assert!(checked > 0, "no fixtures in {dir}");
    }

    /// One encode, of noise, at `KVAD_VAE_SIDE` pixels square (512 by
    /// default) with `KVAD_VAE` (sdxl, flux or sd15), in the precision its
    /// pipeline runs it in, timed after one warm-up; and nothing else, so that
    /// `/usr/bin/time -l` around it measures the encode.
    ///
    ///     KVAD_VAE=sdxl KVAD_VAE_SIDE=1024 /usr/bin/time -l cargo test --release -p kvad-gpu image::vae::tests::one_encode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn one_encode() {
        let name = std::env::var("KVAD_VAE").unwrap_or_else(|_| "sdxl".into());
        let side: usize = std::env::var("KVAD_VAE_SIDE").map(|s| s.parse().unwrap()).unwrap_or(512);
        let device = Device::new_metal(0).unwrap();
        let (_, _, dtype) = vae(&name);
        let (enc, _) = load(&name, &device, dtype);
        let x = Tensor::randn(0f32, 0.5, (1, 3, side, side), &device).unwrap().clamp(-1.0, 1.0).unwrap().to_dtype(dtype).unwrap();
        // `KVAD_VAE_DECODE=1` times the decoder instead, from a latent the
        // encoder's size: the yardstick, since every picture already pays it.
        let decode = std::env::var("KVAD_VAE_DECODE").is_ok();
        let dec = decode.then(|| load(&name, &device, dtype).1);
        let f = enc.config().factor();
        let z = Tensor::randn(0f32, 1.0, (1, enc.config().latent, side / f, side / f), &device).unwrap().to_dtype(dtype).unwrap();
        let run = || {
            let out = match &dec {
                Some(dec) => dec.decode(&z).unwrap(),
                None => enc.encode(&x).unwrap().mean,
            };
            device.synchronize().unwrap();
            out.to_dtype(DType::F32).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };
        run();
        let started = std::time::Instant::now();
        let largest = run();
        let what = if decode { "decode" } else { "encode" };
        eprintln!("{name} {dtype:?} {side}²: one {what} {:.2} s, largest |out| {largest:.2}", started.elapsed().as_secs_f64());
        assert!(largest.is_finite() && largest > 0.0, "the encode came back {largest}");
    }
}
