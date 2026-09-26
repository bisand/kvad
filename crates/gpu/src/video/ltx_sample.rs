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

use super::ltx_dit::{audio_latent, audio_tokens, video_latent, video_tokens, Dit, Grid, Perturb};
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

// ---------------------------------------------------------------------------
// The dev model: guided, in more steps
// ---------------------------------------------------------------------------

/// Stage 1's steps for the dev model, as LTX-2.5's reference pipelines run it.
pub const DEV_STEPS: usize = 30;

/// The dev model's noise levels for `steps` steps: LTX's `LTX2Scheduler`.
///
/// Evenly spaced levels from 1 to 0, shifted towards 1 by `e^s / (e^s +
/// 1/σ − 1)` with `s` 2.05, and then stretched so that the last level before
/// 0 is 0.1. The shift depends on the token count in the reference's
/// formula, but its pipelines never pass a latent, so it is always the one
/// for 4096 tokens. In f32, in the reference's order, where it first
/// comes to 0.99999994 rather than 1.
pub fn dev_sigmas(steps: usize) -> Vec<f32> {
    let (base, most, base_at, most_at) = (0.95f64, 2.05f64, 1024.0f64, 4096.0f64);
    let shift = 4096.0 * ((most - base) / (most_at - base_at)) + (base - (most - base) / (most_at - base_at) * base_at);
    let e = shift.exp() as f32;
    let shifted: Vec<f32> = (0..=steps)
        .map(|i| 1.0 - i as f32 / steps as f32)
        .map(|s| match s == 0.0 {
            true => 0.0,
            false => e / (e + (1.0 / s - 1.0)),
        })
        .collect();
    // The last level before 0 is stretched to 0.1, and the rest with it.
    let last = 1.0 - shifted[steps - 1];
    let scale = last / (1.0 - 0.1f32);
    shifted.iter().map(|&s| if s == 0.0 { 0.0 } else { 1.0 - (1.0 - s) / scale }).collect()
}

/// How hard to steer one stream, and away from what.
///
/// Each prediction of the clean latent is `cond + (cfg − 1)·(cond − uncond)
/// + stg·(cond − blind) + (modality − 1)·(cond − deaf)`: away from the
/// negative prompt, away from the model with its STG block's self-attention
/// skipped, and away from each stream made without the other. Then it is
/// scaled so that its spread is `rescale` of the way back to `cond`'s, which
/// keeps a strong guidance from washing out the colours.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guide {
    pub cfg: f32,
    pub stg: f32,
    pub modality: f32,
    pub rescale: f32,
}

/// LTX-2.5's guidance, from its reference's `PipelineParams` for 2.3 and
/// after: CFG 3 for the video and 7 for the sound, STG 1 on block 28,
/// modality 3, rescale 0.7.
pub const VIDEO_GUIDE: Guide = Guide { cfg: 3.0, stg: 1.0, modality: 3.0, rescale: 0.7 };
pub const AUDIO_GUIDE: Guide = Guide { cfg: 7.0, stg: 1.0, modality: 3.0, rescale: 0.7 };
pub const STG_BLOCK: usize = 28;

/// What the guided pipelines steer away from when a request names nothing
/// else: the reference's `DEFAULT_NEGATIVE_PROMPT`, word for word.
pub const NEGATIVE_PROMPT: &str = "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, \
    blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
    grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
    deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
    wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
    field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
    lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
    valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
    mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
    off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
    pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
    inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

impl Guide {
    /// The guided prediction, from the four in the model's dtype, computed
    /// in f32 and rounded back once, as the reference's `MultiModalGuider`
    /// does. The spread is the unbiased standard deviation over the whole
    /// stream, tokens and channels alike.
    fn combine(&self, cond: &Tensor, uncond: &Tensor, blind: &Tensor, deaf: &Tensor) -> candle_core::Result<Tensor> {
        let f = |t: &Tensor| t.to_dtype(DType::F32);
        let c = f(cond)?;
        let pred = ((&c + ((&c - f(uncond)?)? * (self.cfg - 1.0) as f64)?)? + ((&c - f(blind)?)? * self.stg as f64)?)?;
        let pred = (pred + ((&c - f(deaf)?)? * (self.modality - 1.0) as f64)?)?;
        let pred = match self.rescale {
            0.0 => pred,
            r => {
                let factor = r * (std(&c)? / std(&pred)?) + (1.0 - r);
                (pred * factor as f64)?
            }
        };
        pred.to_dtype(cond.dtype())
    }
}

/// The unbiased standard deviation of every element, as torch's `std()`.
fn std(t: &Tensor) -> candle_core::Result<f32> {
    let x = t.flatten_all()?;
    let n = x.dim(0)? as f64;
    let mean = x.mean_all()?.to_scalar::<f32>()? as f64;
    let ss = x.broadcast_sub(&Tensor::new(mean as f32, x.device())?)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok((ss / (n - 1.0)).sqrt() as f32)
}

/// One guided prediction of the clean latents at σ `s`, from video tokens
/// `xv` and audio tokens `xa`, in their dtype: the reference's
/// `_guided_denoise`, with its four passes one after another.
///
/// Each pass's prediction is `x − σ·v`, rounded to the latents' dtype as
/// the reference's `X0Model` rounds it; the held frame's is the frame
/// itself, its own σ being 0, and after the guidance it is put back, since
/// the rescale scales it with the rest.
#[allow(clippy::too_many_arguments)]
pub fn guided_x0(
    dit: &Dit,
    xv: &Tensor,
    xa: &Tensor,
    s: f32,
    pos: &Contexts,
    neg: &Contexts,
    grid: &Grid,
    (gv, ga): (Guide, Guide),
    still: Option<&Tensor>,
) -> Res<(Tensor, Tensor)> {
    let n0 = held(still, grid.shape())?;
    let keep = xv.dtype();
    let f = |t: &Tensor| t.to_dtype(DType::F32);
    let x0 = |ctx: &Contexts, p: &Perturb| -> Res<(Tensor, Tensor)> {
        let (vv, va) = dit.forward_perturbed(xv, xa, (s, s), n0, ctx, grid, p)?;
        let v0 = hold(&(f(xv)? - (f(&vv)? * s as f64)?)?.to_dtype(keep)?, still)?;
        Ok((v0, (f(xa)? - (f(&va)? * s as f64)?)?.to_dtype(xa.dtype())?))
    };
    let blind = Perturb { blind: vec![STG_BLOCK.min(dit.layers().saturating_sub(1))], deaf: false };
    let deaf = Perturb { blind: vec![], deaf: true };
    let (cv, ca) = x0(pos, &Perturb::default())?;
    let (uv, ua) = x0(neg, &Perturb::default())?;
    let (bv, ba) = x0(pos, &blind)?;
    let (dv, da) = x0(pos, &deaf)?;
    Ok((hold(&gv.combine(&cv, &uv, &bv, &dv)?, still)?, ga.combine(&ca, &ua, &ba, &da)?))
}

/// Stage 1 for the dev model: from pure noise, in [`dev_sigmas`]' steps of
/// Euler, each prediction of the clean latent guided as `guides` say
/// (video, sound). `pos` is the prompt's contexts and `neg` the negative
/// prompt's. Four DiT calls a step, one after another rather than batched,
/// so that the memory at its largest is one call's: the prompt, the
/// negative prompt, the model blinded at [`STG_BLOCK`], and each stream
/// deaf to the other. `still`, `step`: as [`one_stage`].
#[allow(clippy::too_many_arguments)]
pub fn guided(
    dit: &Dit,
    pos: &Contexts,
    neg: &Contexts,
    grid: &Grid,
    seed: u64,
    sigmas: &[f32],
    guides: (Guide, Guide),
    still: Option<&Tensor>,
    step: OnStep<'_>,
) -> Res<Latents> {
    let shape = grid.shape();
    let (dev, keep) = (dit.device(), DType::BF16);
    let (nv, na) = (shape.video_tokens(), shape.audio_latents());
    // Checked here, before thirty steps of work, rather than at the first.
    held(still, shape)?;
    let mut xv = hold(&noise(stream(seed, 0), &[nv, 128], dev, keep)?, still)?;
    let mut xa = noise(stream(seed, 1), &[na, 128], dev, keep)?;
    for i in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[i], sigmas[i + 1]);
        let (x0v, x0a) = guided_x0(dit, &xv, &xa, s, pos, neg, grid, guides, still)?;
        xv = hold(&euler(&xv, &x0v, s, next)?, still)?;
        xa = euler(&xa, &x0a, s, next)?;
        step(i, next, &x0v)?;
    }
    Ok(Latents { video: video_latent(&xv.to_dtype(DType::F32)?, shape)?, audio: audio_latent(&xa.to_dtype(DType::F32)?, 8)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_negative_prompt_is_one_line_with_single_spaces() {
        // Rust's `\` line continuation eats the next line's indent, so the
        // string is the reference's: its joined literals, one space apart.
        assert!(!NEGATIVE_PROMPT.contains("  ") && !NEGATIVE_PROMPT.contains('\n'));
        assert!(NEGATIVE_PROMPT.starts_with("has_subtitles, has_blurbox,") && NEGATIVE_PROMPT.ends_with("or AI artifacts."));
        assert_eq!(NEGATIVE_PROMPT.len(), 1171);
    }

    #[test]
    fn the_dev_schedule_is_the_reference_s() {
        // `LTX2Scheduler().execute(steps=30)`, printed to nine figures.
        let want = [
            0.99999994, 0.99495703, 0.989603043, 0.983908415, 0.977839589, 0.97135824, 0.964421213, 0.956978142, 0.948972046,
            0.940336406, 0.930993795, 0.920853794, 0.909809828, 0.897735238, 0.884478688, 0.869858027, 0.853650749, 0.835584104,
            0.815318584, 0.792426705, 0.766362846, 0.736418009, 0.701656222, 0.660813332, 0.612140656, 0.553147852, 0.480163693,
            0.387540817, 0.266120434, 0.100000024, 0.0,
        ];
        let got = dev_sigmas(DEV_STEPS);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w as f32).abs() < 3e-7, "step {i}: {g} against {w}");
        }
    }

    #[test]
    fn guidance_steers_away_and_rescales_as_the_reference_does() {
        let dev = candle_core::Device::Cpu;
        let t = |v: &[f32]| Tensor::new(v, &dev).unwrap();
        let (c, u, b, d) = (t(&[1.0, 2.0, 3.0, 4.0]), t(&[1.0, 1.0, 1.0, 1.0]), t(&[0.0, 2.0, 3.0, 4.0]), t(&[1.0, 2.0, 3.0, 5.0]));
        let g = Guide { cfg: 3.0, stg: 1.0, modality: 3.0, rescale: 0.0 };
        // cond + 2(cond − uncond) + (cond − blind) + 2(cond − deaf).
        let want = [1.0 + 0.0 + 1.0 + 0.0, 2.0 + 2.0, 3.0 + 4.0, 4.0 + 6.0 - 2.0];
        assert_eq!(g.combine(&c, &u, &b, &d).unwrap().to_vec1::<f32>().unwrap(), want);
        // Rescaled: std(cond)/std(pred) of the way, times 0.7, plus 0.3.
        let r = Guide { rescale: 0.7, ..g }.combine(&c, &u, &b, &d).unwrap().to_vec1::<f32>().unwrap();
        let sd = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32).sqrt()
        };
        let factor = 0.7 * sd(&[1.0, 2.0, 3.0, 4.0]) / sd(&want) + 0.3;
        for (x, w) in r.iter().zip(want) {
            assert!((x - w * factor).abs() < 1e-5, "{r:?}");
        }
    }

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
