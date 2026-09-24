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
//! input. Space is padded with zeros. Both are written into one padded copy
//! of the input, and the 2D convolution then runs unpadded.
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

/// A 3×3×3 convolution with stride 1, replicate padding in time and zero
/// padding in space, on frames-first tensors.
pub struct Conv3d {
    /// `[out, 3·in, 3, 3]`: the kernel with its time taps folded into its
    /// input channels, earliest frame first.
    w: Tensor,
    b: Tensor,
    cin: usize,
}

impl Conv3d {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, cin: usize, cout: usize) -> Res<Self> {
        let r = r.pp(name);
        let w = cx.get(&r, (cout, cin, 3, 3, 3), "weight")?;
        Ok(Conv3d {
            w: fold(&w)?,
            b: cx.get(&r, cout, "bias")?.reshape((1, cout, 1, 1))?,
            cin,
        })
    }

    /// `[T, C_in, H, W]` to `[T, C_out, H, W]`.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_in(x, SCRATCH)
    }

    /// [`Conv3d::forward`] with `scratch` elements of `im2col` at most.
    fn forward_in(&self, x: &Tensor, scratch: usize) -> candle_core::Result<Tensor> {
        let (t, c, h, w) = x.dims4()?;
        debug_assert_eq!(c, self.cin);
        // The first frame once in front, the last once behind; a zero border
        // in space.
        let xp = Tensor::cat(&[&x.narrow(0, 0, 1)?, x, &x.narrow(0, t - 1, 1)?], 0)?;
        let xp = xp.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;

        // Rows per band, then frames per chunk, so that neither the stacked
        // input nor candle's copy of every neighbourhood grows past SCRATCH.
        let per_row = w * 27 * c;
        let rows = (scratch / per_row).clamp(1, h);
        let frames = (scratch / (per_row * rows)).clamp(1, t);

        let mut chunks = Vec::with_capacity(t.div_ceil(frames));
        let mut f = 0;
        while f < t {
            let n = frames.min(t - f);
            let mut bands = Vec::with_capacity(h.div_ceil(rows));
            let mut r = 0;
            while r < h {
                let m = rows.min(h - r);
                // Frames f−1, f and f+1 of the original are f, f+1 and f+2
                // of the padded copy; rows r−1 … r+m likewise.
                let taps = (0..3).map(|k| xp.narrow(0, f + k, n)?.narrow(2, r, m + 2)).collect::<candle_core::Result<Vec<_>>>()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The definition, written out: pad, then for every output sample sum
    /// over input channels and the 3×3×3 neighbourhood.
    fn naive(x: &[f32], (t, c, h, w): (usize, usize, usize, usize), k: &[f32], bias: &[f32], cout: usize) -> Vec<f32> {
        let at = |f: isize, ch: usize, y: isize, xx: isize| -> f32 {
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
        let conv = Conv3d {
            w: fold(&Tensor::from_vec(k.clone(), (cout, c, 3, 3, 3), &dev).unwrap()).unwrap(),
            b: Tensor::from_vec(b.clone(), (1, cout, 1, 1), &dev).unwrap(),
            cin: c,
        };
        let got = conv.forward(&Tensor::from_vec(x.clone(), (t, c, h, w), &dev).unwrap()).unwrap();
        let want = naive(&x, (t, c, h, w), &k, &b, cout);
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "differs by {worst}");
    }

    #[test]
    fn chunks_and_bands_change_nothing() {
        // A budget of five rows' scratch: bands of five rows, one frame at a
        // time, with a short last band.
        let dev = Device::Cpu;
        let (t, c, h, w, cout) = (4, 3, 13, 9, 2);
        let x = Tensor::from_vec(values(t * c * h * w, 4), (t, c, h, w), &dev).unwrap();
        let k = Tensor::from_vec(values(cout * c * 27, 5), (cout, c, 3, 3, 3), &dev).unwrap();
        let conv = Conv3d { w: fold(&k).unwrap(), b: Tensor::zeros((1, cout, 1, 1), DType::F32, &dev).unwrap(), cin: c };
        let got = conv.forward_in(&x, 5 * w * 27 * c).unwrap();
        // The same convolution done in one piece.
        let xp = Tensor::cat(&[&x.narrow(0, 0, 1).unwrap(), &x, &x.narrow(0, t - 1, 1).unwrap()], 0).unwrap();
        let taps: Vec<_> = (0..3).map(|k| xp.narrow(0, k, t).unwrap()).collect();
        let whole = Tensor::cat(&taps, 1).unwrap().conv2d(&conv.w, 1, 1, 1, 1).unwrap();
        assert_eq!(got.dims(), whole.dims());
        let worst = (got - whole).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(worst < 1e-4, "differs by {worst}");
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
