//! LTX-2.5's sound: an audio latent to 48 kHz stereo.
//!
//! `vae/ltx-2.5-audio-vae-bf16.safetensors` holds three models, and sound
//! goes through all of them in turn:
//!
//! 1. **The audio VAE's decoder** turns the latent, `[8, T, 16]` (channels,
//!    time, mel bins), into a log-mel spectrogram, `[2, 4T − 3, 64]`: one per
//!    ear. It is a small 2D convolutional decoder whose convolutions are
//!    causal in time, so a sample never depends on what comes after it.
//! 2. **A vocoder** turns each spectrogram into a 16 kHz waveform, 160
//!    samples per spectrogram frame. It is BigVGAN v2: transposed
//!    convolutions that upsample, each followed by residual blocks whose
//!    activation, *snake*, is periodic, because sound is.
//! 3. **Bandwidth extension** takes that to 48 kHz. It measures the 16 kHz
//!    sound's own spectrogram, runs a second, smaller vocoder on it to
//!    predict what the upper frequencies should add, and adds that to the
//!    16 kHz sound resampled threefold.
//!
//! **All of it runs in f32**, whatever the rest of the pipeline uses. The
//! reference runs its vocoders in f32 because a chain of 108 convolutions in
//! bf16 audibly degrades. It runs the audio VAE decoder in bf16, but that
//! turns out to cost as much: a bf16 spectrogram is 53 dB from the f32 one,
//! and the vocoder turns that into a waveform only 20 dB from it, because
//! small errors in a spectrogram become shifts in phase. In f32 on Metal the
//! whole chain is 97 dB from the reference's f32, and the three models are
//! 160 M parameters and about a second of work.

use super::metadata;
use crate::common::{Loader, Reader};
use crate::image::nn::Ctx;
use crate::image::{finish, open};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor};
use kvad::serde_json::Value;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The audio file in [`super::LTX_REPO`].
pub const FILE: &str = "vae/ltx-2.5-audio-vae-bf16.safetensors";

// ---------------------------------------------------------------------------
// The audio VAE decoder
// ---------------------------------------------------------------------------

/// A 2D convolution that is causal along the first spatial axis (time): the
/// input is padded with `k − 1` zero rows in front and none behind, and
/// symmetrically along the other axis (mel bins).
struct CausalConv2d {
    w: Tensor,
    b: Tensor,
    k: usize,
}

impl CausalConv2d {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, cin: usize, cout: usize, k: usize) -> Res<Self> {
        let r = r.pp(name);
        Ok(CausalConv2d { w: cx.get(&r, (cout, cin, k, k), "weight")?, b: cx.get(&r, cout, "bias")?.reshape((1, cout, 1, 1))?, k })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let p = self.k - 1;
        let x = x.pad_with_zeros(2, p, 0)?.pad_with_zeros(3, p / 2, p - p / 2)?;
        x.conv2d(&self.w, 0, 1, 1, 1)?.broadcast_add(&self.b)
    }
}

/// `x + conv(silu(norm(conv(silu(norm(x))))))`, with a 1×1 convolution on
/// the shortcut when the widths differ.
struct AudioResnet {
    conv1: CausalConv2d,
    conv2: CausalConv2d,
    shortcut: Option<CausalConv2d>,
}

impl AudioResnet {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cin: usize, cout: usize) -> Res<Self> {
        Ok(AudioResnet {
            conv1: CausalConv2d::load(cx, r, "conv1.conv", cin, cout, 3)?,
            conv2: CausalConv2d::load(cx, r, "conv2.conv", cout, cout, 3)?,
            shortcut: match cin != cout {
                true => Some(CausalConv2d::load(cx, r, "nin_shortcut.conv", cin, cout, 1)?),
                false => None,
            },
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.conv1.forward(&pixel_norm(x)?.silu()?)?;
        let h = self.conv2.forward(&pixel_norm(&h)?.silu()?)?;
        match &self.shortcut {
            Some(s) => s.forward(x)? + h,
            None => x + h,
        }
    }
}

/// PixelNorm over channels, with the audio VAE's epsilon (the video
/// decoder's is 1e-8).
fn pixel_norm(x: &Tensor) -> candle_core::Result<Tensor> {
    super::conv3d::pixel_norm(x, 1e-6)
}

pub struct AudioDecoder {
    conv_in: CausalConv2d,
    mid: [AudioResnet; 2],
    /// From the narrowest level out: each level's resnets, then (but for the
    /// last) an upsampling convolution.
    up: Vec<(Vec<AudioResnet>, Option<CausalConv2d>)>,
    conv_out: CausalConv2d,
    /// `[1, 8, 1, 16]`: the statistics are per (channel, mel bin).
    mean: Tensor,
    std: Tensor,
    dtype: DType,
}

impl AudioDecoder {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, config: &Value) -> Res<Self> {
        let dd = &config["audio_vae"]["model"]["params"]["ddconfig"];
        for (key, want) in [
            ("norm_type", Value::from("pixel")),
            ("causality_axis", Value::from("height")),
            ("mid_block_add_attention", Value::Bool(false)),
            ("attn_resolutions", Value::Array(vec![])),
        ] {
            if dd[key] != want {
                return Err(format!("audio VAE: `{key}` is {}, and this decoder is written for {want}", dd[key]).into());
            }
        }
        let num = |k: &str| dd[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("audio VAE: no `{k}`"));
        let (ch, z, out, res) = (num("ch")?, num("z_channels")?, num("out_ch")?, num("num_res_blocks")?);
        let mults: Vec<usize> = dd["ch_mult"].as_array().ok_or("audio VAE: no ch_mult")?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
        let bins = num("mel_bins")?;

        let d = r.pp("audio_vae.decoder");
        let top = ch * mults[mults.len() - 1];
        let conv_in = CausalConv2d::load(cx, &d, "conv_in.conv", z, top, 3)?;
        let mid = [AudioResnet::load(cx, &d.pp("mid.block_1"), top, top)?, AudioResnet::load(cx, &d.pp("mid.block_2"), top, top)?];
        let mut up = Vec::new();
        let mut c = top;
        for level in (0..mults.len()).rev() {
            let s = d.pp(format!("up.{level}"));
            let cout = ch * mults[level];
            let mut blocks = Vec::new();
            for j in 0..=res {
                blocks.push(AudioResnet::load(cx, &s.pp(format!("block.{j}")), c, cout)?);
                c = cout;
            }
            let upsample = match level {
                0 => None,
                _ => Some(CausalConv2d::load(cx, &s, "upsample.conv.conv", c, c, 3)?),
            };
            up.push((blocks, upsample));
        }
        let conv_out = CausalConv2d::load(cx, &d, "conv_out.conv", c, out, 3)?;

        // The latent's 8 channels × 16 bins are normalised as 128 numbers per
        // time step, channel-major.
        let stats = r.pp("audio_vae.per_channel_statistics");
        let per = bins / 4;
        let stat = |name: &str| -> Res<Tensor> { Ok(cx.get(&stats, z * per, name)?.to_dtype(DType::F32)?.reshape((1, z, 1, per))?) };
        Ok(AudioDecoder { conv_in, mid, up, conv_out, mean: stat("mean-of-means")?, std: stat("std-of-means")?, dtype: cx.dtype })
    }

    /// A normalised latent `[8, T, 16]` to log-mel spectrograms `[2, 4T − 3,
    /// 64]`.
    pub fn decode(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        let z = latent.unsqueeze(0)?.to_dtype(DType::F32)?.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let mut h = self.conv_in.forward(&z.to_dtype(self.dtype)?)?;
        for m in &self.mid {
            h = m.forward(&h)?;
        }
        for (blocks, upsample) in &self.up {
            for b in blocks {
                h = b.forward(&h)?;
            }
            if let Some(conv) = upsample {
                let (_, _, t, f) = h.dims4()?;
                // Nearest ×2, then a causal convolution; the first time step
                // is dropped, because its window saw only padding and one
                // copy of the first input.
                h = conv.forward(&h.upsample_nearest2d(2 * t, 2 * f)?)?;
                h = h.narrow(2, 1, 2 * t - 1)?;
            }
        }
        self.conv_out.forward(&pixel_norm(&h)?.silu()?)?.squeeze(0)
    }
}

// ---------------------------------------------------------------------------
// The vocoder
// ---------------------------------------------------------------------------

/// A filter shared by every channel and applied to each on its own: `[1, C,
/// L]` is treated as `C` one-channel signals, which candle does in one
/// convolution rather than a loop over `C` groups.
fn depthwise(x: &Tensor, filter: &Tensor, stride: usize) -> candle_core::Result<Tensor> {
    let (b, c, l) = x.dims3()?;
    let y = x.reshape((b * c, 1, l))?.conv1d(filter, 0, stride, 1, 1)?;
    let n = y.dim(2)?;
    y.reshape((b, c, n))
}

/// [`depthwise`] as a transposed convolution: every input sample becomes
/// `stride` output samples, spread by the filter.
fn depthwise_t(x: &Tensor, filter: &Tensor, stride: usize) -> candle_core::Result<Tensor> {
    let (b, c, l) = x.dims3()?;
    let y = x.reshape((b * c, 1, l))?.conv_transpose1d(filter, 0, 0, stride, 1, 1)?;
    let n = y.dim(2)?;
    y.reshape((b, c, n))
}

/// A transposed convolution with PyTorch's `padding`, which trims that many
/// samples from each end of the full output. candle's Metal kernel wants no
/// padding, so the trimming is done here.
fn conv_t(x: &Tensor, w: &Tensor, b: &Tensor, stride: usize, pad: usize) -> candle_core::Result<Tensor> {
    let y = x.conv_transpose1d(w, 0, 0, stride, 1, 1)?;
    let n = y.dim(2)?;
    y.narrow(2, pad, n - 2 * pad)?.broadcast_add(b)
}

/// *Snake-beta*, `x + sin²(αx) / β` per channel (α and β are stored as
/// logarithms), done at twice the sample rate so that its harmonics do not
/// fold back into the audible band.
///
/// Up by two with a 12-tap filter (edges repeated), the activation, then down
/// by two with another.
struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
    up: Tensor,
    down: Tensor,
}

impl Snake {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, channels: usize) -> Res<Self> {
        let log = |name: &str| -> Res<Tensor> { Ok(cx.get(r, channels, name)?.to_dtype(DType::F32)?.exp()?.reshape((1, channels, 1))?) };
        let alpha = log("act.alpha")?;
        let inv_beta = (log("act.beta")? + 1e-9)?.recip()?;
        let up = cx.get(r, (1, 1, 12), "upsample.filter")?.to_dtype(DType::F32)?;
        let down = cx.get(r, (1, 1, 12), "downsample.lowpass.filter")?.to_dtype(DType::F32)?;
        Ok(Snake { alpha, inv_beta, up, down })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let l = x.dim(2)?;
        // ×2: pad 5 either side, spread, then trim 15 either side.
        let y = (depthwise_t(&x.pad_with_same(2, 5, 5)?, &self.up, 2)? * 2.0)?;
        let y = y.narrow(2, 15, 2 * l)?;
        let y = (&y + y.broadcast_mul(&self.alpha)?.sin()?.sqr()?.broadcast_mul(&self.inv_beta)?)?;
        // ÷2: pad 5 in front and 6 behind, filter with stride 2.
        depthwise(&y.pad_with_same(2, 5, 6)?, &self.down, 2)
    }
}

/// One of BigVGAN's "AMP" residual blocks: three dilated convolutions, each
/// wrapped in snake activations and a plain convolution, each added on.
struct AmpBlock {
    steps: Vec<(Snake, Tensor, Tensor, usize, Snake, Tensor, Tensor)>,
    k: usize,
}

impl AmpBlock {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, c: usize, k: usize, dilations: &[usize]) -> Res<Self> {
        let mut steps = Vec::new();
        for (m, &d) in dilations.iter().enumerate() {
            let conv = |name: &str| -> Res<(Tensor, Tensor)> {
                let r = r.pp(format!("{name}.{m}"));
                Ok((cx.get(&r, (c, c, k), "weight")?, cx.get(&r, c, "bias")?.reshape((1, c, 1))?))
            };
            let (w1, b1) = conv("convs1")?;
            let (w2, b2) = conv("convs2")?;
            steps.push((
                Snake::load(cx, &r.pp(format!("acts1.{m}")), c)?,
                w1,
                b1,
                d,
                Snake::load(cx, &r.pp(format!("acts2.{m}")), c)?,
                w2,
                b2,
            ));
        }
        Ok(AmpBlock { steps, k })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for (a1, w1, b1, d, a2, w2, b2) in &self.steps {
            let h = a1.forward(&x)?.conv1d(w1, (self.k * d - d) / 2, 1, *d, 1)?.broadcast_add(b1)?;
            let h = a2.forward(&h)?.conv1d(w2, (self.k - 1) / 2, 1, 1, 1)?.broadcast_add(b2)?;
            x = (x + h)?;
        }
        Ok(x)
    }
}

/// A BigVGAN v2 generator: spectrogram frames in, waveform out.
struct Vocoder {
    conv_pre: (Tensor, Tensor),
    /// Each upsampling: weight, bias, stride, and the padding it trims.
    ups: Vec<(Tensor, Tensor, usize, usize)>,
    /// `kernels` blocks per upsampling, averaged.
    blocks: Vec<AmpBlock>,
    kernels: usize,
    act_post: Snake,
    conv_post: Tensor,
    /// Whether the output is clamped to [−1, 1] (the main vocoder) or left
    /// as it is (the bandwidth extension's, which predicts a residual).
    clamp: bool,
}

impl Vocoder {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: &Value, clamp: bool) -> Res<Self> {
        for (key, want) in [("resblock", Value::from("AMP1")), ("activation", Value::from("snakebeta")), ("stereo", Value::Bool(true)), ("use_bias_at_final", Value::Bool(false))] {
            if cfg[key] != want {
                return Err(format!("vocoder: `{key}` is {}, and this one is written for {want}", cfg[key]).into());
            }
        }
        if clamp && cfg["use_tanh_at_final"] != Value::Bool(false) {
            return Err("vocoder: a tanh at the output is not implemented".into());
        }
        let list = |k: &str| -> Res<Vec<usize>> {
            Ok(cfg[k].as_array().ok_or_else(|| format!("vocoder: no `{k}`"))?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
        };
        let (rates, up_k, res_k) = (list("upsample_rates")?, list("upsample_kernel_sizes")?, list("resblock_kernel_sizes")?);
        let dilations: Vec<Vec<usize>> = cfg["resblock_dilation_sizes"]
            .as_array()
            .ok_or("vocoder: no resblock_dilation_sizes")?
            .iter()
            .map(|d| d.as_array().map(|d| d.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect()).unwrap_or_default())
            .collect();
        let c0 = cfg["upsample_initial_channel"].as_u64().ok_or("vocoder: no upsample_initial_channel")? as usize;
        let mels = 128; // two ears of 64 bins

        let pre = r.pp("conv_pre");
        let conv_pre = (cx.get(&pre, (c0, mels, 7), "weight")?, cx.get(&pre, c0, "bias")?.reshape((1, c0, 1))?);
        let mut ups = Vec::new();
        let mut blocks = Vec::new();
        for (i, (&s, &k)) in rates.iter().zip(&up_k).enumerate() {
            let (cin, cout) = (c0 >> i, c0 >> (i + 1));
            let u = r.pp(format!("ups.{i}"));
            ups.push((cx.get(&u, (cin, cout, k), "weight")?, cx.get(&u, cout, "bias")?.reshape((1, cout, 1))?, s, (k - s) / 2));
            for (j, (&rk, d)) in res_k.iter().zip(&dilations).enumerate() {
                blocks.push(AmpBlock::load(cx, &r.pp(format!("resblocks.{}", i * res_k.len() + j)), cout, rk, d)?);
            }
        }
        let last = c0 >> rates.len();
        Ok(Vocoder {
            conv_pre,
            ups,
            blocks,
            kernels: res_k.len(),
            act_post: Snake::load(cx, &r.pp("act_post"), last)?,
            conv_post: cx.get(&r.pp("conv_post"), (2, last, 7), "weight")?,
            clamp,
        })
    }

    /// `[1, 128, frames]` (ear-major mel bins) to `[1, 2, frames × Π rates]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.conv1d(&self.conv_pre.0, 3, 1, 1, 1)?.broadcast_add(&self.conv_pre.1)?;
        for (i, (w, b, s, p)) in self.ups.iter().enumerate() {
            x = conv_t(&x, w, b, *s, *p)?;
            let outs = self.blocks[i * self.kernels..(i + 1) * self.kernels].iter().map(|blk| blk.forward(&x)).collect::<candle_core::Result<Vec<_>>>()?;
            x = (Tensor::stack(&outs, 0)?.sum(0)? / self.kernels as f64)?;
        }
        let x = self.act_post.forward(&x)?.conv1d(&self.conv_post, 3, 1, 1, 1)?;
        match self.clamp {
            true => x.clamp(-1f32, 1f32),
            false => Ok(x),
        }
    }
}

/// Two spectrograms `[2, T, 64]` as a vocoder's input, `[1, 128, T]`, ear
/// major: channel `ear·64 + bin`.
fn to_vocoder(mel: &Tensor) -> candle_core::Result<Tensor> {
    let (e, t, m) = mel.dims3()?;
    mel.transpose(1, 2)?.contiguous()?.reshape((1, e * m, t))
}

// ---------------------------------------------------------------------------
// Bandwidth extension, and the whole chain
// ---------------------------------------------------------------------------

/// Bandwidth extension: 16 kHz to 48 kHz.
struct Bwe {
    generator: Vocoder,
    /// `[514, 1, 512]`: the real and imaginary DFT rows with a Hann window
    /// already applied, exactly as training used them.
    stft: Tensor,
    /// `[64, 257]`: frequency bins to mel bins.
    mel: Tensor,
    hop: usize,
    fft: usize,
    /// The ×3 resampler's 43-tap filter, `[1, 1, 43]`, and its ratio.
    sinc: Tensor,
    ratio: usize,
}

impl Bwe {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: &Value) -> Res<Self> {
        let num = |k: &str| cfg[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("bwe: no `{k}`"));
        let (fft, hop, mels) = (num("n_fft")?, num("hop_length")?, num("num_mels")?);
        let ratio = num("output_sampling_rate")? / num("input_sampling_rate")?;
        let bins = fft / 2 + 1;
        let s = r.pp("mel_stft");
        let stft = cx.get(&s, (2 * bins, 1, fft), "stft_fn.forward_basis")?.to_dtype(DType::F32)?;
        // The inverse is for resynthesis, which nothing here does.
        s.record("stft_fn.inverse_basis");
        let mel = cx.get(&s, (mels, bins), "mel_basis")?.to_dtype(DType::F32)?;
        Ok(Bwe {
            generator: Vocoder::load(cx, &r.pp("bwe_generator"), cfg, false)?,
            stft,
            mel,
            hop,
            fft,
            sinc: Tensor::from_vec(hann_sinc(ratio), (1, 1, 12 * ratio + 7), cx.device())?,
            ratio,
        })
    }

    /// A 16 kHz waveform `[1, 2, L]` to 48 kHz, `[1, 2, 3L]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let l = x.dim(2)?;
        let (_, residual, skip) = self.stages(x)?;
        (residual + skip)?.clamp(-1f32, 1f32)?.narrow(2, 0, l * self.ratio)
    }

    /// What [`Bwe::forward`] adds up: the 16 kHz sound's log-mel spectrogram
    /// `[2, 64, frames]`, the residual the generator predicts from it and the
    /// resampled sound it is added to, both `[1, 2, samples]`.
    fn stages(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (_, ears, l) = x.dims3()?;
        let x = match l % self.hop {
            0 => x.clone(),
            r => x.pad_with_zeros(2, 0, self.hop - r)?,
        };
        // Each ear's causal log-mel spectrogram: pad the window's overhang in
        // front only, then one strided convolution computes every frame's DFT.
        let n = x.dim(2)?;
        let spec = x.reshape((ears, 1, n))?.pad_with_zeros(2, self.fft - self.hop, 0)?.conv1d(&self.stft, 0, self.hop, 1, 1)?;
        let bins = spec.dim(1)? / 2;
        let mag = (spec.narrow(1, 0, bins)?.sqr()? + spec.narrow(1, bins, bins)?.sqr()?)?.sqrt()?;
        let mel = self.mel.broadcast_left(ears)?.contiguous()?.matmul(&mag)?.maximum(1e-5)?.log()?;
        let (_, m, frames) = mel.dims3()?;
        let residual = self.generator.forward(&mel.reshape((1, ears * m, frames))?)?;
        Ok((mel, residual, self.resample(&x)?))
    }

    /// Up by [`Bwe::ratio`] through the Hann-windowed sinc: edges repeated
    /// by 7, spread, scaled by the ratio, then trimmed.
    fn resample(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let l = x.dim(2)?;
        let y = (depthwise_t(&x.pad_with_same(2, 7, 7)?, &self.sinc, self.ratio)? * self.ratio as f64)?;
        y.narrow(2, 14 * self.ratio, l * self.ratio)
    }
}

/// The resampler's filter: a sinc windowed by cos², rolled off to 0.99 of
/// the band, `2·7·ratio + 1` taps. Computed rather than stored, as the
/// reference does.
fn hann_sinc(ratio: usize) -> Vec<f32> {
    const ROLLOFF: f64 = 0.99;
    const WIDTH: f64 = 6.0;
    let half = (WIDTH / ROLLOFF).ceil() as usize; // 7
    (0..2 * half * ratio + 1)
        .map(|i| {
            let t = (i as f64 / ratio as f64 - half as f64) * ROLLOFF;
            let sinc = if t == 0.0 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
            let window = (t.clamp(-WIDTH, WIDTH) * std::f64::consts::PI / WIDTH / 2.0).cos().powi(2);
            (sinc * window * ROLLOFF / ratio as f64) as f32
        })
        .collect()
}

/// The whole of LTX-2.5's sound path.
pub struct AudioPath {
    pub decoder: AudioDecoder,
    vocoder: Vocoder,
    bwe: Bwe,
    /// Samples a second at the end.
    pub rate: u32,
    params: usize,
}

impl AudioPath {
    /// Load the decoder half of the audio VAE and both vocoders from the file
    /// at `path`, all in f32.
    pub fn load(path: &Path, device: &Device) -> Res<Self> {
        let config = metadata(path, "config")?;
        let vault = Vault::off();
        let paths = [path.to_path_buf()];
        let r = open(&paths, DType::BF16)?;
        r.skip_under("audio_vae.encoder");

        let f32 = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype: DType::F32 };
        let decoder = AudioDecoder::load(&f32, &r, &config)?;
        let v = &config["vocoder"];
        let vocoder = Vocoder::load(&f32, &r.pp("vocoder.vocoder"), &v["vocoder"], true)?;
        let bwe = Bwe::load(&f32, &r.pp("vocoder"), &v["bwe"])?;
        let rate = v["bwe"]["output_sampling_rate"].as_u64().ok_or("bwe: no output_sampling_rate")? as u32;
        let params = finish("LTX audio decoder and vocoder", &paths, &r)?;
        Ok(AudioPath { decoder, vocoder, bwe, rate, params })
    }

    pub fn params(&self) -> usize {
        self.params
    }

    /// A normalised latent `[8, T, 16]` to the log-mel spectrograms, the
    /// 16 kHz waveform and the final one, `[2, samples]` each, in f32.
    pub fn decode_stages(&self, latent: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let mel = self.decoder.decode(latent)?.to_dtype(DType::F32)?;
        let low = self.vocoder.forward(&to_vocoder(&mel)?)?;
        let high = self.bwe.forward(&low)?;
        Ok((mel, low.squeeze(0)?, high.squeeze(0)?))
    }

    /// The bandwidth extension's insides for the 16 kHz sound `low`, `[2,
    /// samples]`: see [`Bwe::stages`].
    pub fn bwe_stages(&self, low: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (mel, residual, skip) = self.bwe.stages(&low.unsqueeze(0)?)?;
        Ok((mel, residual.squeeze(0)?, skip.squeeze(0)?))
    }

    /// A normalised latent `[8, T, 16]` to stereo at [`AudioPath::rate`].
    pub fn decode(&self, latent: &Tensor) -> Res<kvad::video::Audio> {
        let (_, _, wave) = self.decode_stages(latent)?;
        // Interleaved, left then right, as a WAV or FLAC frame holds them.
        let samples = wave.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        Ok(kvad::video::Audio { rate: self.rate, channels: 2, samples })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resampler_filter_has_the_reference_length_and_passes_dc_at_unit_gain() {
        let h = hann_sinc(3);
        assert_eq!(h.len(), 43);
        // Upsampling by 3 then scaling by 3: a constant must come out the
        // same constant, so each of the three phases sums to a third.
        for phase in 0..3 {
            let s: f32 = h.iter().skip(phase).step_by(3).sum();
            assert!((s * 3.0 - 1.0).abs() < 0.02, "phase {phase} sums to {s}");
        }
    }

    #[test]
    fn a_long_strided_kernel_is_a_strided_dot_product() {
        // The STFT is one conv1d with a 512-sample kernel and a stride of 80;
        // the same shape of thing, smaller.
        let dev = Device::Cpu;
        let (n, k, stride, out) = (300usize, 40usize, 7usize, 5usize);
        let x: Vec<f32> = (0..2 * n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let w: Vec<f32> = (0..out * k).map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0).collect();
        let y = Tensor::from_vec(x.clone(), (2, 1, n), &dev).unwrap().conv1d(&Tensor::from_vec(w.clone(), (out, 1, k), &dev).unwrap(), 0, stride, 1, 1).unwrap();
        let frames = (n - k) / stride + 1;
        assert_eq!(y.dims(), [2, out, frames]);
        let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for b in 0..2 {
            for o in 0..out {
                for f in 0..frames {
                    let want: f32 = (0..k).map(|j| x[b * n + f * stride + j] * w[o * k + j]).sum();
                    let got = y[(b * out + o) * frames + f];
                    assert!((got - want).abs() < 1e-3, "b {b} o {o} f {f}: {got} against {want}");
                }
            }
        }
    }

    /// candle 0.11's matmul, given a *left* operand broadcast along the batch
    /// axis (a stride of 0 there), silently multiplies the wrong numbers on
    /// the CPU; Metal, and a broadcast right operand on either, are fine.
    /// [`Bwe::stages`] makes its mel basis contiguous first. This test is how
    /// that was found, and prints whether the bug is still there.
    #[test]
    fn a_broadcast_left_operand_must_be_made_contiguous_before_a_matmul() {
        for dev in [Some(Device::Cpu), Device::new_metal(0).ok()].into_iter().flatten() {
            let a = Tensor::from_vec((0..12).map(|v| v as f32).collect(), (3, 4), &dev).unwrap();
            let b = Tensor::from_vec((0..40).map(|v| (v % 7) as f32).collect(), (2, 4, 5), &dev).unwrap();
            let right = a.broadcast_left(2).unwrap().contiguous().unwrap().matmul(&b).unwrap();
            let want: Vec<f32> = (0..2)
                .flat_map(|n| (0..3).flat_map(move |i| (0..5).map(move |j| (0..4).map(|k| (i * 4 + k) as f32 * ((n * 20 + k * 5 + j) % 7) as f32).sum())))
                .collect();
            assert_eq!(right.flatten_all().unwrap().to_vec1::<f32>().unwrap(), want, "{dev:?}");
            let raw = a.broadcast_left(2).unwrap().matmul(&b).map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
            eprintln!("{dev:?}: an uncopied broadcast operand {}", match raw {
                Ok(v) if v == want => "gives the right answer".to_string(),
                Ok(_) => "gives a WRONG answer".to_string(),
                Err(e) => format!("is refused: {e}"),
            });
        }
    }

    #[test]
    fn the_trimmed_transposed_convolution_is_pytorchs_padding() {
        // PyTorch: ConvTranspose1d(padding = p) drops p samples from each end
        // of the full output, so length (L − 1)·s − 2p + k.
        let dev = Device::Cpu;
        let x = Tensor::from_vec((0..10).map(|v| v as f32).collect(), (1, 2, 5), &dev).unwrap();
        let w = Tensor::ones((2, 3, 4), DType::F32, &dev).unwrap();
        let b = Tensor::zeros((1, 3, 1), DType::F32, &dev).unwrap();
        let y = conv_t(&x, &w, &b, 2, 1).unwrap();
        assert_eq!(y.dims(), [1, 3, 4 * 2 - 2 + 4]);
    }
}
