//! LTX-2.5's duration head: how long a prompt's clip wants to be.
//!
//! ```text
//! video context [1024, 4096] ─ linear to 256 ─ + video embedding ─┐
//! audio context [1024, 2048] ─ linear to 256 ─ + audio embedding ─┤ 2048 rows
//!                                                                  │
//! one learned query ─ 4-head attention over the rows ─ 256 ─ linear ─ tanh-GELU
//!                   ─ linear to 1 ─ exp ─ seconds
//! ```
//!
//! 1.9 M parameters, in `model_patches/ltx-2.5-duration-head-bf16.safetensors`,
//! reading the two contexts the text path already made. The reference's
//! pipelines use it whenever no frame count is given: they round the seconds
//! to frames at the clip's frame rate, clamp that to 1–20 s, and floor it to
//! the 8k + 1 frames the VAE makes ([`frames_for`]). Here the clamp's top is
//! also what this machine can make at the clip's size, which is lower.
//!
//! It runs in f32 on whatever device the contexts are on: it is a few
//! hundred million multiply-adds, and bf16 would move a prediction near a
//! frame boundary to the other side of it for nothing.

use candle_core::{DType, Device, Tensor};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The duration head's file in [`super::LTX_REPO`].
pub const FILE: &str = "model_patches/ltx-2.5-duration-head-bf16.safetensors";

/// The reference's clamp on a prediction, in seconds.
pub const MIN_SECONDS: f64 = 1.0;
pub const MAX_SECONDS: f64 = 20.0;

/// A linear layer's weight `[out, in]` and bias `[out]`, in f32.
struct Linear {
    w: Tensor,
    b: Tensor,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.matmul(&self.w.t()?)?.broadcast_add(&self.b)
    }
}

pub struct DurationHead {
    video_in: Linear,
    audio_in: Linear,
    video_emb: Tensor,
    audio_emb: Tensor,
    /// The pooler's learned query `[1, 256]`.
    query: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    hidden: Linear,
    last: Linear,
}

impl DurationHead {
    /// The head in the file at `path`, in f32 on `device`.
    pub fn load(path: &Path, device: &Device) -> Res<Self> {
        // SAFETY: a read-only file in the Hub's cache, as everywhere here.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        let get = |name: &str| -> Res<Tensor> {
            let t = st.load(&format!("duration_head.{name}"), &Device::Cpu).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
            Ok(t.to_dtype(DType::F32)?.to_device(device)?)
        };
        let lin = |name: &str| -> Res<Linear> { Ok(Linear { w: get(&format!("{name}.weight"))?, b: get(&format!("{name}.bias"))? }) };
        // `nn.MultiheadAttention` packs the query, key and value projections
        // into one `[3·256, 256]`, in that order.
        let (packed, bias) = (get("attention_pooler.cross_attn.in_proj_weight")?, get("attention_pooler.cross_attn.in_proj_bias")?);
        let width = packed.dim(1)?;
        let part = |i: usize| -> Res<Linear> { Ok(Linear { w: packed.narrow(0, i * width, width)?, b: bias.narrow(0, i * width, width)? }) };
        let config = super::metadata(path, "config").ok();
        let heads = config.as_ref().and_then(|c| c["duration_head"]["num_pooler_heads"].as_u64()).unwrap_or(4) as usize;
        let query = get("attention_pooler.query_tokens")?;
        if query.dim(0)? != 1 {
            return Err(format!("{}: {} pooling queries, and one is implemented", path.display(), query.dim(0)?).into());
        }
        Ok(DurationHead {
            video_in: lin("video_input_proj")?,
            audio_in: lin("audio_input_proj")?,
            video_emb: get("video_modality_emb")?,
            audio_emb: get("audio_modality_emb")?,
            query,
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: lin("attention_pooler.cross_attn.out_proj")?,
            heads,
            hidden: lin("mlp_hidden")?,
            last: lin("mlp_out")?,
        })
    }

    /// The clip's length in seconds, as the head predicts it, from the video
    /// context `[n, 4096]` and the audio context `[m, 2048]`.
    pub fn seconds(&self, video: &Tensor, audio: &Tensor) -> Res<f64> {
        let f = |t: &Tensor| t.to_device(self.query.device())?.to_dtype(DType::F32);
        let v = self.video_in.forward(&f(video)?)?.broadcast_add(&self.video_emb)?;
        let a = self.audio_in.forward(&f(audio)?)?.broadcast_add(&self.audio_emb)?;
        let rows = Tensor::cat(&[v, a], 0)?;
        // One query against every row, each head on its own slice of 256.
        let (q, k, val) = (self.q.forward(&self.query)?, self.k.forward(&rows)?, self.v.forward(&rows)?);
        let (n, width) = rows.dims2()?;
        let d = width / self.heads;
        let split = |t: &Tensor, n: usize| t.reshape((n, self.heads, d))?.transpose(0, 1)?.contiguous();
        let (q, k, val) = (split(&q, 1)?, split(&k, n)?, split(&val, n)?);
        let scores = (q.matmul(&k.transpose(1, 2)?)? / (d as f64).sqrt())?;
        let p = candle_nn::ops::softmax_last_dim(&scores)?;
        let pooled = p.matmul(&val)?.transpose(0, 1)?.reshape((1, width))?;
        let pooled = self.out.forward(&pooled)?;
        let h = self.hidden.forward(&pooled)?.gelu()?;
        let log = self.last.forward(&h)?.flatten_all()?.to_vec1::<f32>()?[0];
        Ok((log as f64).exp())
    }
}

/// A predicted length in frames: `round(seconds · fps)`, clamped to
/// `[min, max]` frames, floored to 8k + 1; and if the floor went under `min`,
/// the next 8k + 1 up instead. The reference's `seconds_to_clamped_num_frames`,
/// with its `min` and `max` from [`MIN_SECONDS`] and [`MAX_SECONDS`] at the
/// frame rate, or less where the caller can make less.
pub fn frames_for(seconds: f64, fps: f64, min: usize, max: usize) -> usize {
    // Python's `round`, half to even, as the reference rounds.
    let raw = (seconds * fps).round_ties_even().max(0.0) as usize;
    let raw = raw.clamp(min, max).max(1);
    let frames = (raw - 1) / 8 * 8 + 1;
    match frames < min {
        true => ((min - 1).div_ceil(8) * 8 + 1).min(max),
        false => frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_follow_the_reference_s_rounding_and_grid() {
        // 3.2 s at 24 fps is 76.8, 77 frames: 8·9 + 5, floored to 73.
        assert_eq!(frames_for(3.2, 24.0, 24, 480), 73);
        // Exactly on the grid stays there.
        assert_eq!(frames_for(121.0 / 24.0, 24.0, 24, 480), 121);
        // Past the top: the top, floored to the grid.
        assert_eq!(frames_for(30.0, 24.0, 24, 480), 473);
        assert_eq!(frames_for(30.0, 24.0, 24, 121), 121);
        // Under a second: the floor of 24 frames is 17, under the minimum,
        // so the next point up, 25.
        assert_eq!(frames_for(0.3, 24.0, 24, 480), 25);
        // Half to even, as Python rounds: 2.5 frames is 2, not 3.
        assert_eq!(frames_for(2.5, 1.0, 1, 100), 1);
    }
}
