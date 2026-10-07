//! DFR, LTX-2.5's "Diffusion Fidelity Rendering" (the reference's
//! `dfr_pipeline.py`): its canvas, its first two stages, and its temporal
//! rounds.
//!
//! DFR is the reference's two stages with keyframes in them. The clip is
//! padded to a whole number of *segments*, 24 or 32 pixel frames, whichever
//! pads it less ([`canvas`]), and a keyframe is generated at the end of each:
//! one latent frame of extra tokens per keyframe, placed at that one pixel
//! frame and marked with the DiT's learned keyframe vector (`ltx_cond`). They
//! are what the temporal rounds later cut the clip at and hold it to, and
//! what the diffusion decoder reads beside the video.
//!
//! 1. [`first`]: at half the width and height, from noise, the video and its
//!    keyframes together, by the distilled schedule's eight steps, ancestral
//!    at η 1 as the plain distilled pipeline's are.
//! 2. The video and the keyframes are upsampled ×2 by the spatial upsampler,
//!    each on its own: the keyframes as a clip of their own, one frame each.
//! 3. [`second`]: at the full size, the upsampled video and keyframes
//!    re-noised to 0.909375, three steps, ancestral too, with stage 1's video
//!    appended clean as a *reference latent* at its own half size, and the
//!    DiT with the detailing IC-LoRA fused in at 0.5. The sound is re-noised
//!    and denoised with the video, which reads it, and then dropped: DFR
//!    keeps stage 1's.
//! 4. [`round`], none, once or twice: the clip's frames doubled by the
//!    temporal upsampler, and denoised again in time tiles that meet at the
//!    keyframes; see below.
//!
//! The DiT is told [`conditioning_fps`]: above 30 frames a second, 60. RoPE
//! places a frame at `frame / fps`, and the model has not seen rates such as
//! 48, so a clip played at 48 is laid out at 60; its sound still lasts the
//! clip, at the frame rate it plays at.
//!
//! **A step** ([`stage`]) is the reference's `euler_ancestral_denoising_loop`
//! with its `X0Model` and `post_process_latent`. Each token's prediction is
//! `x − σ·m·v` for its mask `m` (`ltx_cond`), rounded to the latents' dtype;
//! then blended towards its clean latent, `x₀·m + clean·(1 − m)`, in f32,
//! which puts a reference latent (`m` 0) back as it was and pulls an anchor
//! keyframe most of the way to its own. Then the ancestral step at the
//! step's σ for every token alike, with noise of its own for the video and
//! for the sound, and the blend again after the noise; the last step is the
//! prediction. At η 0 it is the reference's `euler_denoising_loop` instead:
//! the prediction rounded again, and a plain Euler step.
//!
//! Stages 1 and 2 were plain Euler until the reference's 1.4.0 made them
//! ancestral on LTX-2.5's checkpoints, with the distilled pipeline's stage
//! 2. Its spatial epilogue, ancestral as well, is not made here. The
//! temporal rounds stay at η 0.5.
//!
//! **The temporal rounds.** Each doubles the frames, `F` latent frames to
//! `2F − 1` and `N` pixel frames to `2(N − 1) + 1`, and the frame rate with
//! them; the keyframes' places double too, and become *seams*. The clip is
//! cut at them into `2^round` tiles ([`tiles`]), the leftover segments to
//! the first, and every tile after the first starts a segment and one latent
//! frame early: a lead-in it denoises for context and then drops, so that
//! the tile before keeps the seam's own frame and nothing is blended. In
//! each tile, every seam it holds is an *anchor*, a keyframe held at 0.95
//! (`ltx_cond`), and a new keyframe is generated halfway between each pair
//! of marks, seeded with the video's nearest latent frame. The tile is
//! re-noised to 0.975 and takes four *ancestral* steps at η 0.5, seeded
//! apart from every other tile; after each step's noise the tokens are
//! blended towards their clean latents again. Its sound is stage 1's,
//! frozen: resampled linearly over the seconds the tile plays
//! ([`tile_sound`]) to as many latents as the DiT expects at its frame rate,
//! and at σ 0. The anchors and the new keyframes, the earlier tile's where
//! two made one, are the next round's seams.

use super::ltx_cond::{bf16, lerp, State};
pub use kvad::video::{dfr_canvas as canvas, dfr_tiles as tiles, DfrCanvas as Canvas, DfrTile as Tile, DFR_SEGMENTS as SEGMENTS};
use super::ltx_dit::{audio_latent, audio_tokens, video_latent, Dit, Shape};
use super::ltx_sample::{euler, Ancestral, OnStep, STAGE_1, STAGE_2};
use super::ltx_upsample::Upsampler;
use super::ltx_text::Contexts;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The detailing IC-LoRA, DFR's stage 2: gated on the Hub, so its terms
/// must have been accepted by the account whose token fetches it.
pub const DETAILING_REPO: &str = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler";
pub const DETAILING_FILE: &str = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors";
/// How strongly it is fused in: DFR's `_DETAILING_LORA_STRENGTH`.
pub const DETAILING_STRENGTH: f32 = 0.5;

/// Latent channels.
const C: usize = 128;

/// The frame rate the DiT is told for a clip that plays at `fps`: the
/// reference's `_conditioning_fps`, 60 above 30.
pub fn conditioning_fps(fps: f64) -> f64 {
    if fps > 30.0 { 60.0 } else { fps }
}

/// How much smaller than the target a reference latent is, by the LoRA
/// that reads it: `reference_downscale_factor` in its metadata, 1 when it
/// says nothing, as the reference's `read_lora_reference_downscale_factor`.
pub fn reference_downscale(lora: &Path) -> Res<usize> {
    match super::metadata_raw(lora)?.get("reference_downscale_factor") {
        None => Ok(1),
        Some(v) => Ok(v.trim().parse().map_err(|_| format!("{}: a reference downscale of {v:?}", lora.display()))?),
    }
}

/// What a stage makes: the video `[128, F, h, w]`, its generated keyframes
/// `[128, K, h, w]` when the state had slots, and the sound `[8, T, 16]`, in
/// f32.
pub struct Staged {
    pub video: Tensor,
    pub keyframes: Option<Tensor>,
    pub audio: Tensor,
}

/// How a stage steps from one σ to the next.
pub enum Steps<'a> {
    /// Plain Euler: DFR's stages 1 and 2 as the reference's 1.3 ran them.
    Euler,
    /// Ancestral at `eta`: 1 in stages 1 and 2, 0.5 in the temporal rounds.
    /// Every step but the last asks `noise` for the video's noise and then
    /// the sound's.
    Ancestral { eta: f32, noise: Noise<'a> },
}

impl<'a> Steps<'a> {
    /// Ancestral at `eta` on `noise`, or plain Euler at 0, which draws none:
    /// the reference's ancestral loop draws only when η is above 0.
    fn at(eta: f32, noise: Noise<'a>) -> Self {
        if eta > 0.0 { Steps::Ancestral { eta, noise } } else { Steps::Euler }
    }
}

/// Where a stage's noise comes from: `[n, 128]` numbers for the dims asked,
/// in f32, each call the next draw.
pub type Noise<'a> = &'a mut dyn FnMut(&[usize]) -> Res<Vec<f32>>;

/// A stage's sound: noised tokens `[T, 128]`, the shape that times them,
/// and whether it is frozen, held at σ 0 as the temporal rounds hold it.
pub struct Sound<'a> {
    pub tokens: &'a [f32],
    pub shape: Shape,
    pub frozen: bool,
}

/// A stage's steps through `sigmas` on a noised conditioned `state` and its
/// `sound`: the reference's `euler_denoising_loop`, or its
/// `euler_ancestral_denoising_loop`, as `steps` says. See the module notes;
/// `step` hears each step with the prediction of the video's own tokens.
pub fn stage(dit: &Dit, ctx: &Contexts, state: &State, sound: &Sound<'_>, sigmas: &[f32], mut steps: Steps<'_>, step: OnStep<'_>) -> Res<Staged> {
    let (dev, keep) = (dit.device(), dit.dtype());
    let (n, na) = (state.len(), sound.shape.audio_latents());
    if sound.tokens.len() != na * C {
        return Err(format!("{} numbers of sound for {na} latents", sound.tokens.len()).into());
    }
    let f = |t: &Tensor| t.to_dtype(DType::F32);
    let upload = |v: &[f32], rows: usize| -> Res<Tensor> { Ok(Tensor::from_slice(v, (rows, C), dev)?.to_dtype(keep)?) };
    let column = |v: Vec<f32>| -> Res<Tensor> { Ok(Tensor::from_vec(v, (n, 1), dev)?) };
    let (mut xv, mut xa) = (upload(&state.latent, n)?, upload(sound.tokens, na)?);
    // The clean latents in the latents' dtype, as the reference's state
    // keeps them, and each token's share of them, `1 − m`, in f32.
    let clean = f(&upload(&state.clean, n)?)?;
    let mask = column(state.mask.clone())?;
    let rest = column(state.mask.iter().map(|m| 1.0 - m).collect())?;
    let blend = |x: &Tensor| -> candle_core::Result<Tensor> { x.broadcast_mul(&mask)? + clean.broadcast_mul(&rest)? };
    let grid = dit.grid_at(state.shape, state.positions.clone(), &state.marks, sound.shape.audio_positions())?;
    for i in 0..sigmas.len() - 1 {
        let (s, next) = (sigmas[i], sigmas[i + 1]);
        let audio_sigma = if sound.frozen { 0.0 } else { s };
        let (vv, va) = dit.forward_masked(&xv, &xa, (s, audio_sigma), &state.mask, ctx, &grid)?;
        // Each token's σ, `m · σ` in f32 as the reference's timesteps; the
        // prediction rounded, as its `X0Model` rounds it, then blended in f32.
        let sigma = column(state.mask.iter().map(|m| m * s).collect())?;
        let x0 = blend(&f(&(f(&xv)? - f(&vv)?.broadcast_mul(&sigma)?)?.to_dtype(keep)?)?)?;
        // The sound's mask is 1, or 0 when frozen, when it stays as it is.
        let x0a = (f(&xa)? - (f(&va)? * s as f64)?)?.to_dtype(keep)?;
        let look = x0.to_dtype(keep)?;
        match &mut steps {
            Steps::Euler => {
                xv = euler(&xv, &look, s, next)?;
                if !sound.frozen {
                    xa = euler(&xa, &x0a, s, next)?;
                }
            }
            // The last step is the prediction itself.
            Steps::Ancestral { .. } if next == 0.0 => {
                xv = look.clone();
                if !sound.frozen {
                    xa = x0a;
                }
            }
            Steps::Ancestral { eta, noise } => {
                let k = Ancestral::with_eta(s, next, *eta);
                // Drawn in the latents' dtype, as the reference draws it.
                let (ev, ea) = (upload(&noise(&[n, C])?, n)?, upload(&noise(&[na, C])?, na)?);
                let ancestral = |x: &Tensor, x0: &Tensor, e: &Tensor| -> candle_core::Result<Tensor> {
                    (((f(x)? * k.r as f64)? + (x0 * (1.0 - k.r) as f64)?)? * k.a as f64)? + (f(e)? * k.c as f64)?
                };
                // Blended again after the noise, then rounded.
                xv = blend(&ancestral(&xv, &x0, &ev)?)?.to_dtype(keep)?;
                if !sound.frozen {
                    xa = ancestral(&xa, &f(&x0a)?, &ea)?.to_dtype(keep)?;
                }
            }
        }
        step(i, next, &look.narrow(0, 0, state.video_len())?)?;
    }
    let xv = f(&xv)?;
    Ok(Staged {
        video: video_latent(&xv.narrow(0, 0, state.video_len())?, state.shape)?,
        keyframes: match state.slots {
            Some(_) => Some(state.keyframes(&xv)?),
            None => None,
        },
        audio: audio_latent(&f(&xa)?, 8)?,
    })
}

/// `t` as the DiT's dtype would keep it, back in f32 on the host: what the
/// reference's state holds of a latent it is given.
fn kept(t: &Tensor, dtype: DType) -> Res<Tensor> {
    Ok(t.to_device(&Device::Cpu)?.to_dtype(dtype)?.to_dtype(DType::F32)?)
}

/// A draw of noise as the reference draws it: in the latents' dtype.
fn drawn(v: Vec<f32>, dtype: DType) -> Vec<f32> {
    match dtype {
        DType::BF16 => v.into_iter().map(bf16).collect(),
        _ => v,
    }
}

/// Tokens `x` noised as the reference's `GaussianNoiser` noises a latent
/// whose mask is 1 throughout: `lerp(x, noise, scale)`.
fn noised(x: &[f32], noise: &[f32], scale: f32) -> Res<Vec<f32>> {
    if noise.len() != x.len() {
        return Err(format!("{} numbers of noise for {}", noise.len(), x.len()).into());
    }
    Ok(x.iter().zip(noise).map(|(&x, &e)| lerp(x, e, scale)).collect())
}

fn host(t: &Tensor) -> Res<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

/// DFR's stage 1: at `shape`, half the clip's width and height and its
/// canvas's frames at [`conditioning_fps`], from noise, the video with
/// keyframes at `keyframes`; the sound timed by `sound`, the same at the
/// frame rate the clip plays at. `still` is the picture it starts from, as
/// the first latent frame's tokens at this size. `noise` gives the video's,
/// every token's including the keyframes', then the sound's, and then each
/// step's. `eta` is how ancestral the steps are: `ltx_sample::ETA` as the
/// reference runs them, 0 for plain Euler.
#[allow(clippy::too_many_arguments)]
pub fn first(dit: &Dit, ctx: &Contexts, shape: Shape, sound: Shape, keyframes: &[usize], eta: f32, still: Option<&Tensor>, noise: Noise<'_>, step: OnStep<'_>)
 -> Res<Staged> {
    let (rows, cols) = shape.grid();
    let dt = dit.dtype();
    let zeros = Tensor::zeros((C, shape.latent_frames(), rows, cols), DType::F32, &Device::Cpu)?;
    let mut state = State::video(&zeros, shape)?;
    if let Some(s) = still {
        state = state.held(&kept(s, dt)?)?;
    }
    let state = state.slots(keyframes, None)?;
    let len = state.len();
    let state = state.noised(&drawn(noise(&[len, C])?, dt), STAGE_1[0])?;
    let na = sound.audio_latents();
    let audio = noised(&vec![0.0; na * C], &drawn(noise(&[na, C])?, dt), STAGE_1[0])?;
    stage(dit, ctx, &state, &Sound { tokens: &audio, shape: sound, frozen: false }, &STAGE_1, Steps::at(eta, noise), step)
}

/// What DFR's stage 2 starts from: stage 1's video, upsampled and as it
/// was, its keyframes upsampled, and its sound.
pub struct Detailing<'a> {
    pub upsampled: &'a Tensor,
    pub keyframes: &'a Tensor,
    pub reference: &'a Tensor,
    pub audio: &'a Tensor,
}

/// DFR's stage 2: at `shape`, the clip's size, the upsampled video and
/// keyframes re-noised to [`STAGE_2`]'s first level, stage 1's video appended
/// as a reference latent `downscale` times smaller, and three steps; `dit`
/// is the one with the detailing LoRA fused in. `eta`, `still` and `noise`
/// as for [`first`], the picture at this size and the noise drawn afresh.
#[allow(clippy::too_many_arguments)]
pub fn second(dit: &Dit, ctx: &Contexts, shape: Shape, sound: Shape, keyframes: &[usize], from: &Detailing<'_>, downscale: usize, eta: f32, still: Option<&Tensor>,
              noise: Noise<'_>, step: OnStep<'_>) -> Res<Staged> {
    let dt = dit.dtype();
    let mut state = State::video(&kept(from.upsampled, dt)?, shape)?;
    if let Some(s) = still {
        state = state.held(&kept(s, dt)?)?;
    }
    let state = state.slots(keyframes, Some(&kept(from.keyframes, dt)?))?.reference(&kept(from.reference, dt)?, downscale, 1.0)?;
    let len = state.len();
    let state = state.noised(&drawn(noise(&[len, C])?, dt), STAGE_2[0])?;
    let tokens = host(&audio_tokens(&kept(from.audio, dt)?)?)?;
    let audio = noised(&tokens, &drawn(noise(&[tokens.len() / C, C])?, dt), STAGE_2[0])?;
    stage(dit, ctx, &state, &Sound { tokens: &audio, shape: sound, frozen: false }, &STAGE_2, Steps::at(eta, noise), step)
}

// ---------------------------------------------------------------------------
// The temporal rounds
// ---------------------------------------------------------------------------

/// How strongly a tile holds its seams: DFR's `_ANCHOR_KEYFRAME_STRENGTH`,
/// just short of clean, so that a tile can still settle the seam's frame.
pub const ANCHOR_STRENGTH: f32 = 0.95;
/// How ancestral a tile's steps are: `_TEMPORAL_ANCESTRAL_ETA`.
pub const TEMPORAL_ETA: f32 = 0.5;
/// A tile's noise levels: the distilled schedule from its fifth, four steps.
pub const TEMPORAL: &[f32] = STAGE_1.split_at(4).1;

/// A tile's frozen sound, the reference's `_audio_latent_for_tile`: the
/// seconds of `audio` `[8, T, 16]` that `frames` pixel frames from
/// `pixel_start` play at `fps`, of a clip of `duration` seconds, resampled
/// linearly to as many latents as `frames` make at `cond_fps`. In `dtype`,
/// each operation rounded to it as torch's are.
pub fn tile_sound(audio: &Tensor, pixel_start: usize, frames: usize, fps: f64, duration: f64, cond_fps: f64, dtype: DType) -> Res<Tensor> {
    let (c, full, w) = audio.dims3()?;
    let round = |x: f32| if dtype == DType::BF16 { bf16(x) } else { x };
    let a: Vec<f32> = host(audio)?.into_iter().map(round).collect();
    // In f64 as Python's floats, then the positions in f32 as torch's.
    let from = pixel_start as f64 / fps / duration * full as f64;
    let to = (pixel_start + frames) as f64 / fps / duration * full as f64;
    let out = Shape::new(32, 32, frames, cond_fps)?.audio_latents();
    let step = (to - from) / out as f64;
    let mut y = vec![0f32; c * out * w];
    for i in 0..out {
        let p = (from as f32 + step as f32 * i as f32).clamp(0.0, (full - 1) as f32);
        let lo = p.floor() as usize;
        let hi = (lo + 1).min(full - 1);
        let weight = round(p - lo as f32);
        let keep = round(1.0 - weight);
        for ch in 0..c {
            for k in 0..w {
                let (l, h) = (a[(ch * full + lo) * w + k], a[(ch * full + hi) * w + k]);
                y[(ch * out + i) * w + k] = round(round(l * keep) + round(h * weight));
            }
        }
    }
    Ok(Tensor::from_vec(y, (c, out, w), &Device::Cpu)?)
}

/// The latent frames of `video` `[128, F, h, w]` nearest pixel frames
/// `frames`, as `[128, K, h, w]`: the reference's `_slot_initials_from_video`,
/// the new keyframes' seeds.
fn initials(video: &Tensor, frames: &[usize]) -> Res<Tensor> {
    let last = video.dim(1)? - 1;
    let pick: Vec<Tensor> = frames
        .iter()
        .map(|&p| video.narrow(1, ((p as f64 / 8.0).round_ties_even() as usize).min(last), 1))
        .collect::<candle_core::Result<_>>()?;
    Ok(Tensor::cat(&pick, 1)?)
}

/// What a round's callback hears: the tile and how many there are, then a
/// step's number, σ and prediction as [`OnStep`] hears them.
pub type OnTile<'a> = &'a mut dyn FnMut(usize, usize, usize, f32, &Tensor) -> Res<()>;

/// A clip between rounds: its latent `[128, F, h, w]`, its keyframes
/// `[128, K, h, w]` at pixel frames `positions`, its pixel frames and the
/// frame rate it plays at.
pub struct Clip {
    pub video: Tensor,
    pub keyframes: Tensor,
    pub positions: Vec<usize>,
    pub frames: usize,
    pub fps: f64,
}

/// Temporal round `round`, from 1, on `clip`: `up` is the temporal
/// upsampler; `sound` is stage 1's `[8, T, 16]`, of a clip of `duration`
/// seconds as stage 1 played it; `still` is the picture the clip starts
/// from at its size, held in every tile that starts at its first frame;
/// `noise` gives every tile's noise in the reference's order: the video's
/// and the sound's as the tile is noised, then each step's. `step` hears
/// each tile's steps, with the tile.
#[allow(clippy::too_many_arguments)]
pub fn round(
    dit: &Dit,
    ctx: &Contexts,
    up: &Upsampler,
    clip: &Clip,
    round: u32,
    sound: &Tensor,
    duration: f64,
    still: Option<&Tensor>,
    noise: Noise<'_>,
    step: OnTile<'_>,
) -> Res<Clip> {
    if !up.temporal() {
        return Err("a temporal round wants the temporal upsampler".into());
    }
    let (dev, dt) = (dit.device(), dit.dtype());
    let (_, _, rows, cols) = clip.video.dims4()?;
    let video = up.forward(&clip.video.to_device(dev)?.to_dtype(DType::F32)?)?;
    let (frames, fps) = (2 * (clip.frames - 1) + 1, 2.0 * clip.fps);
    let seams: Vec<usize> = clip.positions.iter().map(|p| 2 * p).collect();
    let cond = conditioning_fps(fps);
    let plan = tiles(&seams, frames, 1 << round)?;
    let mut kept_video = Vec::with_capacity(plan.len());
    let mut slots: Vec<(usize, Tensor)> = Vec::new();
    for (t, tile) in plan.iter().enumerate() {
        let shape = Shape::new(32 * cols, 32 * rows, tile.frames(), cond)?;
        let part = kept(&video.narrow(1, tile.start, tile.end - tile.start)?, dt)?;
        let mut state = State::video(&part, shape)?;
        // The picture, where the tile starts where the clip does.
        if let (Some(s), 0) = (still, tile.pixel_start) {
            state = state.held(&kept(s, dt)?)?;
        }
        for &p in &tile.anchors {
            let k = seams.iter().position(|&s| s == p).ok_or("an anchor that is not a seam")?;
            state = state.anchor(&clip.keyframes.narrow(1, k, 1)?, p - tile.pixel_start, ANCHOR_STRENGTH)?;
        }
        let local: Vec<usize> = tile.slots.iter().map(|p| p - tile.pixel_start).collect();
        if !local.is_empty() {
            state = state.slots(&local, Some(&initials(&part, &local)?))?;
        }
        // Noised to the first level; the sound's noise is drawn, at 0.
        let len = state.len();
        let state = state.noised(&drawn(noise(&[len, C])?, dt), TEMPORAL[0])?;
        let tokens = host(&audio_tokens(&tile_sound(sound, tile.pixel_start, tile.frames(), fps, duration, cond, dt)?)?)?;
        let sound_shape = Shape::new(32 * cols, 32 * rows, tile.frames(), cond)?;
        noise(&[sound_shape.audio_latents(), C])?;
        let steps = Steps::Ancestral { eta: TEMPORAL_ETA, noise: &mut *noise };
        let n = plan.len();
        let got = stage(dit, ctx, &state, &Sound { tokens: &tokens, shape: sound_shape, frozen: true }, TEMPORAL, steps, &mut |i, s, x| step(t, n, i, s, x))?;
        kept_video.push(got.video.narrow(1, tile.lead, got.video.dim(1)? - tile.lead)?);
        if let Some(k) = got.keyframes {
            for (j, &p) in tile.slots.iter().enumerate() {
                // Where two tiles made one keyframe, the earlier's.
                if !slots.iter().any(|(q, _)| *q == p) {
                    slots.push((p, k.narrow(1, j, 1)?));
                }
            }
        }
    }
    // The next round's seams: this round's, and the new keyframes.
    let mut carried: std::collections::BTreeMap<usize, Tensor> = std::collections::BTreeMap::new();
    for (k, &p) in seams.iter().enumerate() {
        carried.insert(p, clip.keyframes.narrow(1, k, 1)?.to_device(&Device::Cpu)?);
    }
    for (p, k) in slots {
        carried.insert(p, k.to_device(&Device::Cpu)?);
    }
    let video = Tensor::cat(&kept_video, 1)?;
    if video.dim(1)? != (frames - 1) / 8 + 1 {
        return Err(format!("round {round}'s tiles stitched to {} latent frames, not {}", video.dim(1)?, (frames - 1) / 8 + 1).into());
    }
    Ok(Clip {
        video,
        keyframes: Tensor::cat(&carried.values().collect::<Vec<_>>(), 1)?,
        positions: carried.into_keys().collect(),
        frames,
        fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dit_is_told_sixty_above_thirty() {
        assert_eq!((conditioning_fps(24.0), conditioning_fps(30.0), conditioning_fps(48.0)), (24.0, 30.0, 60.0));
    }

    /// A tile's sound is the stretch it plays, stretched to its latents.
    #[test]
    fn a_tile_hears_its_own_seconds() {
        // 8 latents, each its own index.
        let a = Tensor::arange(0f32, 8.0, &Device::Cpu).unwrap().reshape((1, 8, 1)).unwrap();
        // The second half of a 2 s clip, 25 frames at 24 fps from frame 24.
        let y = tile_sound(&a, 24, 25, 24.0, 2.0, 24.0, DType::F32).unwrap();
        let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y.len(), 26);
        assert_eq!(y[0], 4.0);
        // Clamped at the last latent.
        assert_eq!(y[25], 7.0);
    }
}
