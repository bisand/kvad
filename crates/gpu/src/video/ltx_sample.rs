//! Sampling LTX-2.5's distilled DiT: noise to a video latent and its sound.
//!
//! The distilled model runs a fixed schedule with no guidance, one DiT call a
//! step. Stage 1 is eight *ancestral* steps: each takes a deterministic
//! step to below the next noise level and adds fresh noise back up to it.
//! Two-stage generation then upsamples and refines with three plain Euler
//! steps; that is step 6 of `docs/video-plan.md`. Here is one stage at the
//! requested size, which is stage 1 with no upsampler.
//!
//! The DiT predicts velocity `v = ε − x₀`, with `x_σ = (1 − σ)·x₀ + σ·ε`, so
//! `x₀ = x − σ·v`. The latents are kept in bf16 between steps, as the
//! reference keeps them, and every update is computed in f32.

use super::ltx_dit::{audio_latent, video_latent, Dit, Grid};
use super::ltx_text::Contexts;
use crate::image::nn::noise;
use candle_core::{DType, Tensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Stage 1's noise levels: fixed, with no shift for resolution or length.
pub const STAGE_1: [f32; 9] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];

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

/// Stage 1 at the grid's own size, from pure noise to the latents the
/// decoders read. `step` hears each step's number and σ after it runs.
pub fn one_stage(dit: &Dit, ctx: &Contexts, grid: &Grid, seed: u64, step: &mut dyn FnMut(usize, f32) -> Res<()>) -> Res<Latents> {
    let shape = grid.shape();
    let (dev, keep) = (dit.device(), DType::BF16);
    let (nv, na) = (shape.video_tokens(), shape.audio_latents());
    // Both latents start as noise (σ = 1 is all noise), drawn in token order.
    let mut xv = noise(stream(seed, 0), &[nv, 128], dev, keep)?;
    let mut xa = noise(stream(seed, 1), &[na, 128], dev, keep)?;
    let sigmas = &STAGE_1;
    for i in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[i], sigmas[i + 1]);
        let (vv, va) = dit.forward(&xv, &xa, (s, s), ctx, grid)?;
        let f = |t: &Tensor| t.to_dtype(DType::F32);
        // The prediction, rounded to the latent's dtype as the reference's is.
        let x0v = (f(&xv)? - (f(&vv)? * s as f64)?)?.to_dtype(keep)?;
        let x0a = (f(&xa)? - (f(&va)? * s as f64)?)?.to_dtype(keep)?;
        if next == 0.0 {
            (xv, xa) = (x0v, x0a);
        } else {
            let k = Ancestral::new(s, next);
            let update = |x: &Tensor, x0: &Tensor, draw: u64| -> Res<Tensor> {
                let det = ((f(x)? * k.r as f64)? + (f(x0)? * (1.0 - k.r) as f64)?)?;
                let eps = noise(stream(seed, draw), x.dims(), dev, keep)?;
                Ok(((det * k.a as f64)? + (f(&eps)? * k.c as f64)?)?.to_dtype(keep)?)
            };
            xv = update(&xv, &x0v, 2 + 2 * i as u64)?;
            xa = update(&xa, &x0a, 3 + 2 * i as u64)?;
        }
        step(i, next)?;
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
