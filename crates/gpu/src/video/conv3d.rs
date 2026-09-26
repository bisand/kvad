//! A 3D convolution, built exactly out of 2D ones.
//!
//! A video decoder's convolutions are 3×3×3: each output sample mixes its
//! neighbours in time as well as in space. candle has `conv1d` and `conv2d`
//! and nothing three-dimensional, but a 3D convolution over time is a sum of
//! 2D ones over neighbouring frames:
//!
//! ```text
//! y[t] = b + Σₖ conv2d(x[t + k − 1], W[:, :, k])        k = 0, 1, 2
//! ```
//!
//! and a sum of three 2D convolutions is one 2D convolution over the three
//! frames stacked on the channel axis, with the kernel's time axis folded
//! into its input channels: `[out, in, 3, 3, 3]` becomes `[out, 3·in, 3, 3]`.
//! That is what [`Conv3d`] does. It is exact — the same products, summed in a
//! different order — and it is one matrix multiply per chunk rather than
//! three.
//!
//! Tensors here are **frames first**, `[T, C, H, W]`: the frames are the 2D
//! convolution's batch, which is what makes stacking them cheap.
//!
//! **Padding.** LTX's decoder is not causal: the first frame is repeated once
//! in front and the last once behind, so the output has as many frames as the
//! input. Its latent upsampler pads time with zeros instead ([`Time`]). Space
//! is padded with zeros. Each chunk's window is cut from the input and padded
//! on its own, and the 2D convolution then runs unpadded. (Padding the whole
//! input first made two full-size copies of it: 6 GB for one convolution at
//! 1536×1024.)
//!
//! **Memory.** candle's Metal convolution copies every 3×3 neighbourhood out
//! into a row of its own (`im2col`) before multiplying, so its scratch is
//! `frames × H × W × 27·C` numbers. For all 121 frames at once that is tens
//! of gigabytes. So the output is made in chunks of frames, and a frame too
//! large for one chunk in bands of rows, each with the one-sample halo it
//! reads from its neighbours. Chunking changes nothing but the order.

use crate::common::Reader;
use crate::image::nn::Ctx;
use candle_core::{DType, Tensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The most `im2col` scratch one chunk may build, in elements: 256 M, which
/// is 512 MB in bf16.
const SCRATCH: usize = 1 << 28;

/// What a [`Conv3d`] reads before the first frame and after the last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Time {
    /// The edge frame again: the VAE decoder.
    Replicate,
    /// Nothing: the latent upsampler, a plain `Conv3d(padding=1)`.
    Zeros,
}

/// A 3×3×3 convolution with stride 1, padded in time as [`Time`] says and
/// with zeros in space, on frames-first tensors.
pub struct Conv3d {
    /// `[out, 3·in, 3, 3]`: the kernel with its time taps folded into its
    /// input channels, earliest frame first.
    w: Tensor,
    b: Tensor,
    cin: usize,
    time: Time,
}

impl Conv3d {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, cin: usize, cout: usize) -> Res<Self> {
        let r = r.pp(name);
        // Folded on the host and uploaded once. Folded on the device, the
        // upload stays allocated beside the folded copy until the next
        // synchronise, and the copy is rounded up to a power of two: the
        // upsampler held 4.5 GB for its 2 GB of f32 weights.
        let w = r.get((cout, cin, 3, 3, 3), "weight")?.to_dtype(cx.dtype)?;
        Ok(Conv3d {
            w: fold(&w)?.to_device(cx.device())?,
            b: cx.get(&r, cout, "bias")?.reshape((1, cout, 1, 1))?,
            cin,
            time: Time::Replicate,
        })
    }

    /// The same convolution padded in time as `time` says.
    pub(crate) fn padded(self, time: Time) -> Self {
        Conv3d { time, ..self }
    }

    /// `[T, C_in, H, W]` to `[T, C_out, H, W]`.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let t = x.dim(0)?;
        self.frames_in(x, (0, t), (0, t), SCRATCH)
    }

    /// Output frames `lo .. hi` of a clip `total` frames long, from `x`,
    /// which holds the clip's frames from `start` on: at least those the
    /// outputs read, `lo − 1 .. hi + 1` within the clip.
    ///
    /// For a caller that works through a clip a chunk at a time: the padding
    /// goes on at the clip's own first and last frames and nowhere else, so
    /// the chunks' outputs are the whole clip's.
    pub fn frames(&self, x: &Tensor, (start, total): (usize, usize), (lo, hi): (usize, usize)) -> candle_core::Result<Tensor> {
        self.frames_in(x, (start, total), (lo, hi), SCRATCH)
    }

    /// [`Conv3d::frames`] with `scratch` elements of `im2col` at most.
    fn frames_in(&self, x: &Tensor, clip: (usize, usize), (lo, hi): (usize, usize), scratch: usize) -> candle_core::Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        debug_assert_eq!(c, self.cin);

        // Rows per band, then frames per chunk, so that neither the stacked
        // input nor candle's copy of every neighbourhood grows past SCRATCH.
        let per_row = w * 27 * c;
        let rows = (scratch / per_row).clamp(1, h);
        let frames = (scratch / (per_row * rows)).clamp(1, hi - lo);

        let mut chunks = Vec::with_capacity((hi - lo).div_ceil(frames));
        let mut f = lo;
        while f < hi {
            let n = frames.min(hi - f);
            let mut bands = Vec::with_capacity(h.div_ceil(rows));
            let mut r = 0;
            while r < h {
                let m = rows.min(h - r);
                // Output frame f + i reads window frames i, i + 1 and i + 2.
                let xp = self.window(x, clip, (f, n), (r, m))?;
                let taps = (0..3).map(|k| xp.narrow(0, k, n)).collect::<candle_core::Result<Vec<_>>>()?;
                let stacked = Tensor::cat(&taps, 1)?;
                bands.push(stacked.conv2d(&self.w, 0, 1, 1, 1)?.broadcast_add(&self.b)?);
                r += m;
            }
            chunks.push(if bands.len() == 1 { bands.pop().unwrap() } else { Tensor::cat(&bands, 2)? });
            f += n;
        }
        if chunks.len() == 1 {
            Ok(chunks.pop().unwrap())
        } else {
            Tensor::cat(&chunks, 0)
        }
    }

    /// What output frames `f .. f + n`, rows `r .. r + m` read: frames
    /// `f − 1 .. f + n + 1` and rows `r − 1 .. r + m + 1`, padded where those
    /// run off the clip. `[n + 2, C, m + 2, W + 2]`. `x` holds the clip's
    /// frames from `start` on, of `total`.
    fn window(&self, x: &Tensor, (start, total): (usize, usize), (f, n): (usize, usize), (r, m): (usize, usize)) -> candle_core::Result<Tensor> {
        let h = x.dim(2)?;
        // Rows first, so that everything after copies only the band.
        let (top, bottom) = (r.saturating_sub(1), (r + m + 1).min(h));
        let x = x.narrow(2, top, bottom - top)?;
        let (first, last) = (f.saturating_sub(1), (f + n + 1).min(total));
        let at = |g: usize| -> candle_core::Result<usize> {
            match g.checked_sub(start).filter(|&i| i < x.dim(0).unwrap_or(0)) {
                Some(i) => Ok(i),
                None => candle_core::bail!("conv3d: frame {g} is not in the {} given from {start}", x.dim(0)?),
            }
        };
        let mut parts = Vec::with_capacity(3);
        let edge = |g: usize| -> candle_core::Result<Tensor> {
            let frame = x.narrow(0, at(g)?, 1)?;
            match self.time {
                Time::Replicate => Ok(frame),
                Time::Zeros => frame.zeros_like(),
            }
        };
        if f == 0 {
            parts.push(edge(0)?);
        }
        at(last - 1)?;
        parts.push(x.narrow(0, at(first)?, last - first)?);
        if f + n == total {
            parts.push(edge(total - 1)?);
        }
        let x = if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(&parts, 0)? };
        x.pad_with_zeros(2, (r == 0) as usize, (r + m == h) as usize)?.pad_with_zeros(3, 1, 1)
    }
}

/// `[out, in, kt, kh, kw]` to `[out, kt·in, kh, kw]`, time tap major: the
/// order [`Conv3d::forward`] stacks the frames in.
fn fold(w: &Tensor) -> candle_core::Result<Tensor> {
    let (o, i, kt, kh, kw) = w.dims5()?;
    w.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((o, kt * i, kh, kw))
}

/// Each position's channel vector divided by its root mean square: LTX's
/// replacement for group norm, with no learned scale.
///
/// Over axis 1, which is channels in a frames-first tensor. Computed in f32.
pub fn pixel_norm(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let rms = (x.sqr()?.mean_keepdim(1)? + eps)?.sqrt()?;
    x.broadcast_div(&rms)?.to_dtype(dtype)
}

/// The most elements [`by_frames`] hands its function at once: 64 M, 256 MB
/// in f32.
const PIECE: usize = 1 << 26;

/// `f` applied a few frames at a time, and the pieces put back together.
///
/// For work that treats every frame on its own, such as a norm over channels
/// or an activation, so that its temporaries are a piece's size rather than
/// the clip's: [`pixel_norm`] of a whole 1536×1024 decoder stage makes three
/// f32 copies of 6 GB each.
pub fn by_frames(x: &Tensor, f: impl Fn(&Tensor) -> candle_core::Result<Tensor>) -> candle_core::Result<Tensor> {
    let t = x.dim(0)?;
    let per = (PIECE / (x.elem_count() / t).max(1)).clamp(1, t);
    if per == t {
        return f(x);
    }
    let pieces = (0..t).step_by(per).map(|s| f(&x.narrow(0, s, per.min(t - s))?)).collect::<candle_core::Result<Vec<_>>>()?;
    Tensor::cat(&pieces, 0)
}

/// `silu(pixel_norm(x))`, a few frames at a time: what precedes every
/// convolution in the decoder.
pub fn norm_silu(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    by_frames(x, |x| pixel_norm(x, eps)?.silu())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The definition, written out: pad, then for every output sample sum
    /// over input channels and the 3×3×3 neighbourhood.
    fn naive(x: &[f32], (t, c, h, w): (usize, usize, usize, usize), k: &[f32], bias: &[f32], cout: usize, time: Time) -> Vec<f32> {
        let at = |f: isize, ch: usize, y: isize, xx: isize| -> f32 {
            if time == Time::Zeros && (f < 0 || f >= t as isize) {
                return 0.0;
            }
            let f = f.clamp(0, t as isize - 1) as usize;
            if y < 0 || xx < 0 || y >= h as isize || xx >= w as isize {
                return 0.0;
            }
            x[((f * c + ch) * h + y as usize) * w + xx as usize]
        };
        let mut out = vec![0f32; t * cout * h * w];
        for f in 0..t {
            for o in 0..cout {
                for y in 0..h {
                    for xx in 0..w {
                        let mut s = bias[o];
                        for i in 0..c {
                            for dt in 0..3 {
                                for dy in 0..3 {
                                    for dx in 0..3 {
                                        let kv = k[(((o * c + i) * 3 + dt) * 3 + dy) * 3 + dx];
                                        s += kv * at(f as isize + dt as isize - 1, i, y as isize + dy as isize - 1, xx as isize + dx as isize - 1);
                                    }
                                }
                            }
                        }
                        out[((f * cout + o) * h + y) * w + xx] = s;
                    }
                }
            }
        }
        out
    }

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8) as f32 / (1 << 24) as f32 - 0.5).collect()
    }

    #[test]
    fn the_folded_convolution_is_the_3d_one_including_the_edges() {
        let dev = Device::Cpu;
        let (t, c, h, w, cout) = (5, 3, 6, 7, 4);
        let x = values(t * c * h * w, 1);
        let k = values(cout * c * 27, 2);
        let b = values(cout, 3);
        for time in [Time::Replicate, Time::Zeros] {
            let conv = Conv3d {
                w: fold(&Tensor::from_vec(k.clone(), (cout, c, 3, 3, 3), &dev).unwrap()).unwrap(),
                b: Tensor::from_vec(b.clone(), (1, cout, 1, 1), &dev).unwrap(),
                cin: c,
                time,
            };
            let got = conv.forward(&Tensor::from_vec(x.clone(), (t, c, h, w), &dev).unwrap()).unwrap();
            let want = naive(&x, (t, c, h, w), &k, &b, cout, time);
            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-5, "{time:?} differs by {worst}");
        }
    }

    #[test]
    fn chunks_and_bands_change_nothing() {
        // A budget of five rows' scratch: bands of five rows, one frame at a
        // time, with a short last band; each window padded on its own, at
        // the clip's edges and not between chunks.
        let dev = Device::Cpu;
        let (t, c, h, w, cout) = (4, 3, 13, 9, 2);
        let x = values(t * c * h * w, 4);
        let k = values(cout * c * 27, 5);
        for time in [Time::Replicate, Time::Zeros] {
            let conv = Conv3d {
                w: fold(&Tensor::from_vec(k.clone(), (cout, c, 3, 3, 3), &dev).unwrap()).unwrap(),
                b: Tensor::zeros((1, cout, 1, 1), DType::F32, &dev).unwrap(),
                cin: c,
                time,
            };
            let got = conv.frames_in(&Tensor::from_vec(x.clone(), (t, c, h, w), &dev).unwrap(), (0, t), (0, t), 5 * w * 27 * c).unwrap();
            let want = naive(&x, (t, c, h, w), &k, &vec![0.0; cout], cout, time);
            let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-5, "{time:?} differs by {worst}");
        }
    }

    #[test]
    fn frames_from_a_slice_are_the_whole_clips() {
        // A clip of five frames, taken in pieces that each come with only
        // the one frame of halo they read: padding at the clip's ends, and
        // none between the pieces.
        let dev = Device::Cpu;
        let (t, c, h, w, cout) = (5, 3, 4, 5, 2);
        let x = Tensor::from_vec(values(t * c * h * w, 8), (t, c, h, w), &dev).unwrap();
        let k = Tensor::from_vec(values(cout * c * 27, 9), (cout, c, 3, 3, 3), &dev).unwrap();
        for time in [Time::Replicate, Time::Zeros] {
            let conv = Conv3d { w: fold(&k).unwrap(), b: Tensor::zeros((1, cout, 1, 1), DType::F32, &dev).unwrap(), cin: c, time };
            let whole = conv.forward(&x).unwrap();
            for (lo, hi) in [(0usize, 2usize), (2, 3), (3, 5), (0, 5)] {
                let (s, e) = (lo.saturating_sub(1), (hi + 1).min(t));
                let got = conv.frames(&x.narrow(0, s, e - s).unwrap(), (s, t), (lo, hi)).unwrap();
                let want = whole.narrow(0, lo, hi - lo).unwrap();
                let worst = (got - want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
                assert!(worst < 1e-6, "{time:?} frames {lo}..{hi} differ by {worst}");
            }
            // A slice missing a frame the outputs read is refused.
            assert!(conv.frames(&x.narrow(0, 2, 2).unwrap(), (2, t), (2, 4)).is_err());
        }
    }

    #[test]
    fn by_frames_is_the_whole_thing_in_pieces() {
        let dev = Device::Cpu;
        // 3 frames of 2^25 elements: two frames a piece, then one.
        let x = Tensor::from_vec(values(3 * 8, 7), (3, 8, 1, 1), &dev).unwrap().broadcast_as((3, 8, 1 << 11, 1 << 11)).unwrap().contiguous().unwrap();
        let whole = pixel_norm(&x, 1e-8).unwrap().silu().unwrap();
        let pieces = norm_silu(&x, 1e-8).unwrap();
        assert_eq!(whole.flatten_all().unwrap().to_vec1::<f32>().unwrap(), pieces.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }

    #[test]
    fn pixel_norm_leaves_every_position_with_unit_rms() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(values(2 * 8 * 3 * 3, 6), (2, 8, 3, 3), &dev).unwrap();
        let y = pixel_norm(&x, 1e-8).unwrap();
        let ms = y.sqr().unwrap().mean_keepdim(1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(ms.iter().all(|m| (m - 1.0).abs() < 1e-5), "{ms:?}");
    }
}
