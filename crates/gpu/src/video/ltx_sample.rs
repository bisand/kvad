//! Sampling LTX-2.5's distilled DiT: noise to a video latent and its sound.
//!
//! The distilled model runs a fixed schedule with no guidance, one DiT call a
//! step, in two stages:
//!
//! 1. [`one_stage`]: eight *ancestral* steps from pure noise, at half the
//!    requested width and height. Each takes a deterministic step to below
//!    the next noise level and adds fresh noise back up to it.
//! 2. The video latent is upsampled ×2 (`ltx_upsample`), and [`refine`]
//!    re-noises both latents to σ = 0.909375 and takes three plain Euler
//!    steps at the full size. The sound is refined too, not frozen.
//!
//! Stage 1 alone at the full size is the cheaper, rougher path.
//!
//! The DiT predicts velocity `v = ε − x₀`, with `x_σ = (1 − σ)·x₀ + σ·ε`, so
//! `x₀ = x − σ·v`. The latents are kept in bf16 between steps, as the
//! reference keeps them, and every update is computed in f32.
//!
//! **From a picture.** Image-to-video gives each stage the picture encoded
//! at that stage's size, `still`: the first latent frame's tokens. They start
//! as the picture instead of noise, the DiT sees them at σ = 0, and after
//! every update they are put back, as the reference's `post_process_latent`
//! does with a denoise mask of 0 there. Their prediction is the picture too:
//! the reference's `x − σ·v` with their own σ, 0, is `x`. The noise is drawn
//! for every token all the same, so a seed makes the same noise with a
//! picture as without.

use super::ltx_dit::{audio_latent, audio_tokens, video_latent, video_tokens, Dit, Grid};
use super::ltx_text::Contexts;
use crate::image::nn::noise;
use candle_core::{DType, Tensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Stage 1's noise levels: fixed, with no shift for resolution or length.
pub const STAGE_1: [f32; 9] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];

/// Stage 2's noise levels: the upsampled latent is re-noised to the first,
/// then three Euler steps. (The reference's docs say four steps; four
/// levels make three.)
pub const STAGE_2: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// How much of each step's deterministic move is replaced by noise: 1 is
/// fully ancestral, 0 plain Euler.
const ETA: f32 = 1.0;

/// One ancestral step from σ to σₙ, as `x ← a·(r·x + (1 − r)·x₀) + c·ε`.
///
/// The deterministic part goes to σ_down = σₙ·(1 + (σₙ/σ − 1)·η), below σₙ;
/// `a` rescales the signal from σ_down's `1 − σ_down` to σₙ's `1 − σₙ`, and
/// `c` adds back what variance that leaves short of σₙ. In f32, in the
/// reference's order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ancestral {
    pub a: f32,
    pub r: f32,
    pub c: f32,
}

impl Ancestral {
    pub fn new(sigma: f32, next: f32) -> Self {
        let down = next * (1.0 + (next / sigma - 1.0) * ETA);
        let r = down / sigma;
        let (alpha_next, alpha_down) = (1.0 - next, 1.0 - down);
        let c = (next * next - down * down * alpha_next * alpha_next / (alpha_down * alpha_down)).max(0.0).sqrt();
        Ancestral { a: alpha_next / alpha_down, r, c }
    }
}

/// Where each draw of noise comes from: the seed, moved along so that no
/// two draws in one generation share a stream. kvad's noise is its own
/// (`image::nn::noise`), not torch's, so a seed makes the same clip here
/// every time and not the reference's clip.
fn stream(seed: u64, draw: u64) -> u64 {
    seed.wrapping_add(draw.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// A clip's latents: video `[128, F, h, w]` and audio `[8, T, 16]`, in f32.
pub struct Latents {
    pub video: Tensor,
    pub audio: Tensor,
}

/// What a step's callback hears after the step: its number, the σ it went
/// to, and the DiT's prediction of the clean video, as tokens `[n, 128]` in
/// bf16 — what a preview shows, since the latent itself is still mostly
/// noise.
pub type OnStep<'a> = &'a mut dyn FnMut(usize, f32, &Tensor) -> Res<()>;

/// `x` with its first tokens put back to `still`'s, when there is a picture.
fn hold(x: &Tensor, still: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let Some(still) = still else { return Ok(x.clone()) };
    let n = still.dim(0)?;
    Tensor::cat(&[&still.to_device(x.device())?.to_dtype(x.dtype())?, &x.narrow(0, n, x.dim(0)? - n)?], 0)
}

/// How many tokens a picture holds, and checked against the grid.
fn held(still: Option<&Tensor>, shape: super::ltx_dit::Shape) -> Res<usize> {
    let Some(still) = still else { return Ok(0) };
    match still.dims() {
        [n, 128] if *n == shape.frame_tokens() => Ok(*n),
        d => Err(format!("a picture of {d:?} tokens, where a {}×{} frame is [{}, 128]", shape.width, shape.height, shape.frame_tokens()).into()),
    }
}

/// Stage 1 at the grid's own size, from pure noise to the latents the
/// decoders read; from a picture, when `still` holds one as tokens of the
/// first latent frame. `step` hears each step as it ends; see [`OnStep`].
pub fn one_stage(dit: &Dit, ctx: &Contexts, grid: &Grid, seed: u64, still: Option<&Tensor>, step: OnStep<'_>) -> Res<Latents> {
    let shape = grid.shape();
    let (dev, keep) = (dit.device(), DType::BF16);
    let (nv, na) = (shape.video_tokens(), shape.audio_latents());
    let n0 = held(still, shape)?;
    // Both latents start as noise (σ = 1 is all noise), drawn in token order.
    let mut xv = hold(&noise(stream(seed, 0), &[nv, 128], dev, keep)?, still)?;
    let mut xa = noise(stream(seed, 1), &[na, 128], dev, keep)?;
    let sigmas = &STAGE_1;
    for i in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[i], sigmas[i + 1]);
        let (vv, va) = dit.forward(&xv, &xa, (s, s), n0, ctx, grid)?;
        let f = |t: &Tensor| t.to_dtype(DType::F32);
        // The prediction, rounded to the latent's dtype as the reference's is.
        let x0v = hold(&(f(&xv)? - (f(&vv)? * s as f64)?)?.to_dtype(keep)?, still)?;
        let x0a = (f(&xa)? - (f(&va)? * s as f64)?)?.to_dtype(keep)?;
        step(i, next, &x0v)?;
        if next == 0.0 {
            (xv, xa) = (x0v, x0a);
        } else {
            let k = Ancestral::new(s, next);
            let update = |x: &Tensor, x0: &Tensor, draw: u64| -> Res<Tensor> {
                let det = ((f(x)? * k.r as f64)? + (f(x0)? * (1.0 - k.r) as f64)?)?;
                let eps = noise(stream(seed, draw), x.dims(), dev, keep)?;
                Ok(((det * k.a as f64)? + (f(&eps)? * k.c as f64)?)?.to_dtype(keep)?)
            };
            xv = hold(&update(&xv, &x0v, 2 + 2 * i as u64)?, still)?;
            xa = update(&xa, &x0a, 3 + 2 * i as u64)?;
        }
    }
    Ok(Latents { video: video_latent(&xv.to_dtype(DType::F32)?, shape)?, audio: audio_latent(&xa.to_dtype(DType::F32)?, 8)? })
}

/// One Euler step from σ to σₙ, given the prediction `x0`: the velocity
/// `(x − x₀)/σ`, rounded to the latent's dtype, then `x + v·(σₙ − σ)` in f32,
/// rounded again, as the reference does both.
fn euler(x: &Tensor, x0: &Tensor, sigma: f32, next: f32) -> candle_core::Result<Tensor> {
    let keep = x.dtype();
    let f = |t: &Tensor| t.to_dtype(DType::F32);
    let v = ((f(x)? - f(x0)?)? / sigma as f64)?.to_dtype(keep)?;
    (f(x)? + (f(&v)? * (next - sigma) as f64)?)?.to_dtype(keep)
}

/// Stage 2: `latents`, the upsampled video and stage 1's sound, re-noised
/// to [`STAGE_2`]'s first level and refined by three Euler steps at the
/// grid's size; `still` is the picture at this size, when there is one.
/// `step` hears each step as it ends; see [`OnStep`].
pub fn refine(dit: &Dit, ctx: &Contexts, grid: &Grid, latents: &Latents, seed: u64, still: Option<&Tensor>, step: OnStep<'_>) -> Res<Latents> {
    let shape = grid.shape();
    let n0 = held(still, shape)?;
    let (dev, keep) = (dit.device(), DType::BF16);
    let f = |t: &Tensor| t.to_dtype(DType::F32);
    let sigmas = &STAGE_2;
    // `lerp(x, ε, σ₀)`: most of the way back to noise, keeping σ₀'s share of
    // the signal. Draws 100 and 101, clear of stage 1's.
    let renoise = |x: Tensor, draw: u64| -> Res<Tensor> {
        let x = x.to_device(dev)?.to_dtype(keep)?;
        let eps = noise(stream(seed, draw), x.dims(), dev, keep)?;
        Ok((&f(&x)? + ((f(&eps)? - f(&x)?)? * sigmas[0] as f64)?)?.to_dtype(keep)?)
    };
    let mut xv = hold(&renoise(video_tokens(&latents.video)?, 100)?, still)?;
    let mut xa = renoise(audio_tokens(&latents.audio)?, 101)?;
    if xv.dim(0)? != shape.video_tokens() || xa.dim(0)? != shape.audio_latents() {
        return Err(format!("stage 2 at {}×{} wants {} video and {} audio tokens, and was given {} and {}", shape.width, shape.height, shape.video_tokens(), shape.audio_latents(), xv.dim(0)?, xa.dim(0)?).into());
    }
    for i in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[i], sigmas[i + 1]);
        let (vv, va) = dit.forward(&xv, &xa, (s, s), n0, ctx, grid)?;
        let x0v = hold(&(f(&xv)? - (f(&vv)? * s as f64)?)?.to_dtype(keep)?, still)?;
        let x0a = (f(&xa)? - (f(&va)? * s as f64)?)?.to_dtype(keep)?;
        // The picture's velocity is (x − x₀)/σ = 0, so it stays put.
        xv = euler(&xv, &x0v, s, next)?;
        xa = euler(&xa, &x0a, s, next)?;
        step(i, next, &x0v)?;
    }
    Ok(Latents { video: video_latent(&xv.to_dtype(DType::F32)?, shape)?, audio: audio_latent(&xa.to_dtype(DType::F32)?, 8)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestral_coefficients_match_the_plans_table() {
        // `docs/video-plan.md`, derived from the reference's formula in f64
        // and printed to six places. The step computes in f32, as the
        // reference does, where 1 − σ_down with σ_down near 0.98 keeps only
        // about five places: hence 1e-5.
        let table = [
            (0.501567, 0.987539, 0.861510),
            (0.668067, 0.987461, 0.738504),
            (0.751189, 0.987382, 0.652982),
            (0.801020, 0.987302, 0.590269),
            (0.596873, 0.869915, 0.755431),
            (0.651669, 0.635609, 0.619472),
            (0.766223, 0.338604, 0.377621),
        ];
        for (i, (a, r, c)) in table.into_iter().enumerate() {
            let k = Ancestral::new(STAGE_1[i], STAGE_1[i + 1]);
            assert!((k.a - a).abs() < 1e-5 && (k.r - r).abs() < 1e-5 && (k.c - c).abs() < 1e-5, "step {i}: {k:?}");
        }
    }

    #[test]
    fn the_last_euler_step_lands_on_the_prediction() {
        // σₙ = 0: x + (x − x₀)/σ·(0 − σ) is x₀, up to the velocity's rounding.
        let dev = candle_core::Device::Cpu;
        let x = Tensor::new(&[0.8f32, -1.3, 2.0], &dev).unwrap();
        let x0 = Tensor::new(&[0.1f32, 0.4, -0.7], &dev).unwrap();
        let y = euler(&x, &x0, 0.421875, 0.0).unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in y.iter().zip([0.1f32, 0.4, -0.7]) {
            assert!((a - b).abs() < 1e-6, "{y:?}");
        }
    }

    #[test]
    fn eta_zero_would_be_euler() {
        // With η = 0, σ_down is σₙ: `a` is 1, `c` is 0, and r·x + (1 − r)·x₀
        // is x + (σₙ − σ)·v. This checks the algebra the step relies on.
        let (s, n) = (0.9f32, 0.7f32);
        let down = n * (1.0 + (n / s - 1.0) * 0.0);
        assert_eq!(down, n);
        let (x, v) = (0.3f32, -1.2f32);
        let x0 = x - s * v;
        let r = down / s;
        assert!((r * x + (1.0 - r) * x0 - (x + (n - s) * v)).abs() < 1e-6);
    }
}
