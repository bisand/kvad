//! Conditioned video states: the tokens DFR appends after a video's own, and
//! the positions, masks and keyframe marks the DiT reads them by.
//!
//! The reference builds a stage's video as a sequence of tokens with, for
//! each one, a noisy latent, a clean latent, a denoise mask and a place
//! (`LatentState`); conditioning items append tokens to its end, in the order
//! they are given (`ltx_core.conditioning`). DFR uses three:
//!
//! | Item | Latent, clean | Mask | Place in time | Keyframe mark |
//! |---|---|---|---|---|
//! | the video itself | the latent, the latent | 1 | its latent frames' spans | the first frame |
//! | [`State::held`], a picture it starts from, or the frames a tile is given | the latent, what is held | 0 on those frames' tokens | their own | — |
//! | [`State::anchor`], a given keyframe | 0, the keyframe | `1 − strength`, 0.05 in DFR | one pixel frame, `[f, f + 1)` | no |
//! | [`State::slots`], keyframes to generate | the initial latents or 0, 0 | 1 | one pixel frame each | yes |
//! | [`State::reference`], a smaller latent | 0, the latent | `1 − strength`, 0 in DFR | its own frames' spans, rows and columns × the downscale | no |
//!
//! Every place is a span, `[start, end)` in seconds on time and pixels on
//! rows and columns, and the DiT's RoPE reads its middle. A token's σ is the
//! step's times its mask ([`crate::video::ltx_dit::Tokens::Masked`]); the
//! keyframe mark adds the DiT's learned keyframe vector. [`State::noised`]
//! noises the lot as the reference's `GaussianNoiser` does, and after a
//! stage [`State::keyframes`] reads the generated keyframes back out.

use super::ltx_dit::{video_tokens, Shape};
use candle_core::{DType, Device, Tensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A video's tokens and what was appended to them.
#[derive(Clone)]
pub struct State {
    /// Noisy and clean latents, tokens `[n, 128]` in f32 on the host.
    pub latent: Vec<f32>,
    pub clean: Vec<f32>,
    /// Each token's denoise mask: its σ over the step's.
    pub mask: Vec<f32>,
    /// Each token's place, the middle of its span: time in seconds, rows
    /// and columns in pixels.
    pub positions: [Vec<f32>; 3],
    /// Whether each token is a keyframe, to the DiT's keyframe vector.
    pub marks: Vec<bool>,
    /// The video's own shape, at the frame rate the DiT is told.
    pub shape: Shape,
    /// Where the generated keyframes are: their first token, and their
    /// pixel frames, one latent frame of tokens each.
    pub slots: Option<(usize, Vec<usize>)>,
}

/// Latent channels.
const C: usize = 128;

/// `1 − strength` in bf16, as the reference's keyframe items make their
/// masks: in the keyframe latents' dtype, which is bf16 in DFR. 0.95 is a
/// mask of 0.050048828125, not 0.05.
fn bf16_mask(strength: f32) -> f32 {
    // Python's `1.0 - strength` in f64, to f32, then to bf16.
    bf16((1.0 - strength as f64) as f32)
}

/// `x` rounded to bf16, to the nearest and ties to even, as torch rounds.
pub(crate) fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

impl State {
    /// A video latent `[128, F, h, w]` of `shape`, whose frame rate is the
    /// one the DiT is told (DFR's `_conditioning_fps`), as a state with
    /// nothing appended: mask 1, the first latent frame marked.
    pub fn video(latent: &Tensor, shape: Shape) -> Res<State> {
        let (c, f, h, w) = latent.dims4()?;
        let (rows, cols) = shape.grid();
        if c != C || f != shape.latent_frames() || (h, w) != (rows, cols) {
            return Err(format!("a latent {:?} for {}×{} × {} frames", latent.dims(), shape.width, shape.height, shape.frames).into());
        }
        let tokens = host(&video_tokens(latent)?)?;
        let n = shape.video_tokens();
        Ok(State {
            clean: tokens.clone(),
            latent: tokens,
            mask: vec![1.0; n],
            positions: shape.video_positions(),
            marks: (0..n).map(|i| i < shape.frame_tokens()).collect(),
            shape,
            slots: None,
        })
    }

    /// The video's first latent frames held to `still`, their tokens `[k·h·w,
    /// 128]`: the reference's `VideoConditionByLatentIndex` at index 0 and
    /// strength 1. One frame is a picture a video starts from; several are
    /// what a temporal round's tile is given of the tile before
    /// (`lead_in_carryover`). Their tokens' clean latent is `still`'s and
    /// their mask 0, so that noising leaves them as they are and the DiT
    /// sees them at σ 0.
    pub fn held(mut self, still: &Tensor) -> Res<State> {
        let n = self.shape.frame_tokens();
        let rows = still.dim(0)?;
        if still.rank() != 2 || still.dim(1)? != C || rows == 0 || rows % n != 0 || rows > self.shape.video_tokens() {
            return Err(format!("{:?} tokens to hold, where a {}×{} frame is [{n}, 128] and the clip has {}", still.dims(), self.shape.width, self.shape.height, self.shape.latent_frames()).into());
        }
        self.clean[..rows * C].copy_from_slice(&host(still)?);
        self.mask[..rows].fill(0.0);
        Ok(self)
    }

    /// Tokens so far.
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// The video's own tokens, before anything appended.
    pub fn video_len(&self) -> usize {
        self.shape.video_tokens()
    }

    fn push(&mut self, latent: &[f32], clean: &[f32], mask: f32, positions: [Vec<f32>; 3], mark: bool) {
        let n = positions[0].len();
        self.latent.extend_from_slice(latent);
        self.clean.extend_from_slice(clean);
        self.mask.extend(std::iter::repeat_n(mask, n));
        for (all, p) in self.positions.iter_mut().zip(positions) {
            all.extend(p);
        }
        self.marks.extend(std::iter::repeat_n(mark, n));
    }

    /// Places for one latent frame's tokens of `rows × cols`, at pixel frame
    /// `frame` for one pixel frame, rows and columns `scale` times a
    /// token's 32 pixels: `[f, f + 1)` over the frame rate, in f32 as the
    /// reference divides, and the middle of each.
    fn one_frame(&self, frame: usize, rows: usize, cols: usize, scale: usize) -> [Vec<f32>; 3] {
        let fps = self.shape.fps as f32;
        let t = (frame as f32 / fps + (frame + 1) as f32 / fps) / 2.0;
        let mut out = [Vec::new(), Vec::new(), Vec::new()];
        for r in 0..rows {
            for c in 0..cols {
                out[0].push(t);
                out[1].push(((32 * r * scale) + (32 * r + 32) * scale) as f32 / 2.0);
                out[2].push(((32 * c * scale) + (32 * c + 32) * scale) as f32 / 2.0);
            }
        }
        out
    }

    /// A given keyframe `[128, 1, h, w]` at pixel frame `frame`, held
    /// towards itself at `strength`: the reference's
    /// `VideoConditionByKeyframeIndex`, as DFR's anchors between temporal
    /// tiles (0.95). In bf16, as DFR carries its keyframes, whatever the
    /// DiT runs in. Appended, unmarked.
    pub fn anchor(mut self, keyframe: &Tensor, frame: usize, strength: f32) -> Res<State> {
        let (rows, cols) = self.shape.grid();
        if keyframe.dims() != [C, 1, rows, cols] {
            return Err(format!("a keyframe {:?} for a video of {rows}×{cols} latents", keyframe.dims()).into());
        }
        let clean: Vec<f32> = host(&video_tokens(keyframe)?)?.into_iter().map(bf16).collect();
        let places = self.one_frame(frame, rows, cols, 1);
        self.push(&vec![0.0; clean.len()], &clean, bf16_mask(strength), places, false);
        Ok(self)
    }

    /// Keyframes to generate at pixel frames `frames`, each one latent
    /// frame of tokens at the video's size, starting from `initial`
    /// `[128, K, h, w]` or from zeros: the reference's
    /// `VideoGeneratedKeyframeSlots`, which DFR adds at every stage.
    /// Appended together, marked, mask 1.
    pub fn slots(mut self, frames: &[usize], initial: Option<&Tensor>) -> Res<State> {
        let (rows, cols) = self.shape.grid();
        if self.slots.is_some() {
            return Err("generated keyframes were already added to this state".into());
        }
        if frames.is_empty() || frames.windows(2).any(|w| w[1] <= w[0]) || frames[frames.len() - 1] >= self.shape.frames {
            return Err(format!("keyframes at {frames:?} in {} frames", self.shape.frames).into());
        }
        let per = rows * cols;
        let latent = match initial {
            None => vec![0.0; frames.len() * per * C],
            Some(t) if t.dims() == [C, frames.len(), rows, cols] => host(&video_tokens(t)?)?,
            Some(t) => return Err(format!("initial keyframes {:?} for {} at {rows}×{cols}", t.dims(), frames.len()).into()),
        };
        let first = self.len();
        for (k, &f) in frames.iter().enumerate() {
            let places = self.one_frame(f, rows, cols, 1);
            let part = &latent[k * per * C..(k + 1) * per * C];
            self.push(part, &vec![0.0; part.len()], 1.0, places, true);
        }
        self.slots = Some((first, frames.to_vec()));
        Ok(self)
    }

    /// A latent `[128, F, h′, w′]` of the same clip at `downscale` times
    /// smaller, held at `strength`: the reference's
    /// `VideoConditionByReferenceLatent`, DFR's stage-1 video in its stage 2
    /// (strength 1). Its tokens take its own frames' spans in time and the
    /// target's pixels on rows and columns. Appended, unmarked.
    pub fn reference(mut self, latent: &Tensor, downscale: usize, strength: f32) -> Res<State> {
        let (c, f, h, w) = latent.dims4()?;
        if c != C {
            return Err(format!("a reference latent {:?}", latent.dims()).into());
        }
        // The reference's own shape, at this state's frame rate: its spans
        // in time are the video's frames' (the reference's
        // `temporal_scale_factor` is 1 in DFR).
        let own = Shape::new(32 * w, 32 * h, 8 * (f - 1) + 1, self.shape.fps)?;
        let [t, rows, cols] = own.video_positions();
        let d = downscale as f32;
        let places = [t, rows.into_iter().map(|p| p * d).collect(), cols.into_iter().map(|p| p * d).collect()];
        let clean = host(&video_tokens(latent)?)?;
        // Its mask in its dtype, which is f32 in DFR; 0 at strength 1
        // whichever.
        self.push(&vec![0.0; clean.len()], &clean, 1.0 - strength, places, false);
        Ok(self)
    }

    /// Noised as the reference's `GaussianNoiser` noises, `noise` `[n, 128]`
    /// in f32 covering every token: `x = lerp(latent, noise, scale)`, then
    /// `lerp(clean, x, mask)`, in f32, each `lerp` as PyTorch computes it.
    pub fn noised(mut self, noise: &[f32], scale: f32) -> Res<State> {
        if noise.len() != self.latent.len() {
            return Err(format!("{} numbers of noise for {} tokens", noise.len(), self.len()).into());
        }
        for (i, x) in self.latent.iter_mut().enumerate() {
            let m = self.mask[i / C];
            *x = lerp(self.clean[i], lerp(*x, noise[i], scale), m);
        }
        Ok(self)
    }

    /// The generated keyframes in `latent` `[n, 128]`, a stage's answer for
    /// this state, as `[128, K, h, w]`.
    pub fn keyframes(&self, latent: &Tensor) -> Res<Tensor> {
        let (first, frames) = self.slots.as_ref().ok_or("this state has no generated keyframes")?;
        let (rows, cols) = self.shape.grid();
        let tokens = latent.narrow(0, *first, frames.len() * rows * cols)?;
        Ok(tokens.t()?.contiguous()?.reshape((C, frames.len(), rows, cols))?)
    }

    /// The noisy latent's tokens `[n, 128]` on `device`.
    pub fn tokens(&self, device: &Device) -> Res<Tensor> {
        Ok(Tensor::from_slice(&self.latent, (self.len(), C), device)?)
    }
}

/// PyTorch's `lerp`, as its CPU kernel computes it: from the nearer end,
/// so that a weight of 1 gives `end` exactly, each a fused multiply-add. The
/// same without fusing was 159 dB from the reference's noised state, last
/// bits apart; with it, identical.
pub(crate) fn lerp(start: f32, end: f32, weight: f32) -> f32 {
    match weight < 0.5 {
        true => weight.mul_add(end - start, start),
        false => (-(end - start)).mul_add(1.0 - weight, end),
    }
}

fn host(t: &Tensor) -> Res<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(f: usize, h: usize, w: usize) -> Tensor {
        Tensor::arange(0f32, (C * f * h * w) as f32, &Device::Cpu).unwrap().reshape((C, f, h, w)).unwrap()
    }

    /// DFR's anchor mask is 0.95's complement rounded to bf16, and a mask of
    /// 1 or 0 leaves the noised latent all noise or all clean.
    #[test]
    fn masks_and_noise_follow_the_reference() {
        // 0.050048828125.
        let anchor = f32::from_bits(0x3d4d_0000);
        assert_eq!(bf16_mask(0.95), anchor);
        assert_eq!(bf16_mask(1.0), 0.0);
        let shape = Shape::new(64, 64, 9, 24.0).unwrap();
        let s = State::video(&latent(2, 2, 2), shape).unwrap().reference(&latent(2, 1, 1), 2, 1.0).unwrap();
        let noise = vec![0.5f32; s.latent.len()];
        let clean = s.clean.clone();
        let n = s.noised(&noise, 1.0).unwrap();
        // The video all noise; the reference all clean.
        assert!(n.latent[..8 * C].iter().all(|&x| x == 0.5));
        assert_eq!(&n.latent[8 * C..], &clean[8 * C..]);
    }

    /// Anchors and slots sit on one pixel frame; a reference latent's rows
    /// and columns are the target's pixels, and its tokens are unmarked.
    #[test]
    fn places_and_marks() {
        let shape = Shape::new(64, 32, 17, 60.0).unwrap();
        let s = State::video(&latent(3, 1, 2), shape).unwrap();
        let s = s.anchor(&latent(1, 1, 2), 16, 0.95).unwrap().slots(&[8], None).unwrap().reference(&latent(3, 1, 1), 2, 1.0).unwrap();
        assert_eq!(s.len(), 6 + 2 + 2 + 3);
        // The first latent frame and the slot are keyframes; nothing else.
        let marked: Vec<usize> = (0..s.len()).filter(|&i| s.marks[i]).collect();
        assert_eq!(marked, vec![0, 1, 8, 9]);
        let fps = 60f32;
        assert_eq!(s.positions[0][6], (16.0 / fps + 17.0 / fps) / 2.0);
        assert_eq!(s.positions[0][8], (8.0 / fps + 9.0 / fps) / 2.0);
        // A reference token of 32 of its pixels covers 64 of the target's.
        assert_eq!((s.positions[1][10], s.positions[2][10]), (32.0, 32.0));
        let anchor = f32::from_bits(0x3d4d_0000);
        assert_eq!(s.mask[6..10], [anchor, anchor, 1.0, 1.0]);
        assert_eq!(s.slots, Some((8, vec![8])));
    }
}
