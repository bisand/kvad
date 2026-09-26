//! The pieces LTX-2.5's transformers share: the text connectors and the DiT
//! (`docs/video-plan.md`).
//!
//! - [`rms`], RMS norm with no weight of its own, which every LTX block uses
//!   because modulation or the next projection supplies the scale;
//! - [`gelu`], in f32 whatever its input, because candle's bf16 one is not;
//! - [`RmsNorm`], the weighted one, over a whole attention width at once;
//! - [`Rope`], LTX's "split" rotary embedding: one frequency vector across
//!   the *whole* attention width, cut into a slice per head, applied to each
//!   head's two halves;
//! - [`GatedAttention`], attention whose output each head scales by
//!   `2·sigmoid(gate)`, a gate the block learns from its own input.

use crate::common::Reader;
use crate::image::nn::{Ctx, Linear};
use crate::prof::span;
use candle_core::{DType, Device, Tensor, D};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `x / √(mean(x²) + eps)` over the last axis, in f32.
pub(crate) fn rms(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let r = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&r)?.to_dtype(dtype)
}

/// Tanh-approximated GELU, computed in f32 whatever `x` is.
///
/// candle's Metal kernel evaluates the polynomial and the tanh in the
/// tensor's own type, so in bf16 every intermediate is rounded to eight bits
/// of mantissa; PyTorch computes in f32 and rounds once. In the DiT's video
/// feed-forward, 16 384 wide, that difference alone cost 8 dB against the
/// reference after one block: 38.7 dB where the reference's own bf16 is 46.8.
pub(crate) fn gelu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.to_dtype(DType::F32)?.gelu()?.to_dtype(x.dtype())
}

/// RMS norm over the last axis, times a learned weight (not `1 + weight`).
pub(crate) struct RmsNorm {
    w: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, width: usize, eps: f64) -> Res<Self> {
        Ok(RmsNorm { w: r.get(width, name)?.to_dtype(DType::F32)?.to_device(cx.device())?, eps })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        rms(&x.to_dtype(DType::F32)?, self.eps)?.broadcast_mul(&self.w)?.to_dtype(dtype)
    }

    /// `x` `[n, width]` normed over its whole width, then, with `rope`, each
    /// of its `heads` rotated: in `dt`, whatever `x` came in. One kernel
    /// where [`super::ltx_fused::norm_rope`] takes it.
    pub(crate) fn rotated(&self, x: &Tensor, rope: Option<&Rope>, heads: usize, dt: DType) -> candle_core::Result<Tensor> {
        if let Some(y) = super::ltx_fused::norm_rope(x, &self.w, rope.map(|r| (&r.cos, &r.sin)), heads, self.eps as f32, dt)? {
            return Ok(y);
        }
        let y = self.forward(&x.to_dtype(dt)?)?;
        match rope {
            Some(r) => {
                let (n, width) = y.dims2()?;
                r.rotate(&y.reshape((n, heads, width / heads))?)?.reshape((n, width))
            }
            None => Ok(y),
        }
    }
}

/// Rotary embedding tables, `cos` and `sin`, each `[T, heads, half]`.
///
/// [`Rope::rotate`] turns every head's two halves `(x₁, x₂)` by the angles:
/// `(x₁·cos − x₂·sin, x₂·cos + x₁·sin)`. What differs between models is only
/// how the angles are laid out, which the constructors say.
pub(crate) struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    /// LTX's split layout.
    ///
    /// `positions[a][t]` is token `t`'s position on axis `a`, and `max_pos[a]`
    /// that axis' scale. There are `N = width / (2·axes)` frequencies,
    /// `ω_k = (π/2)·θ^(k/(N−1))`, built in f64 and rounded to f32. Token `t`'s
    /// angles are `ω_k · (2·p/max − 1)` for every frequency and axis, the axes
    /// interleaved (`ω₀a₀, ω₀a₁, …, ω₁a₀, …`), padded at the *front* with
    /// unrotated entries to `width / 2`, and cut into `heads` equal slices.
    ///
    /// The angles are computed in f32, as the reference computes them. They
    /// reach about 15 700 radians, where f32 is good to a thousandth of one
    /// and bf16 to sixty-four, so they are never computed on the GPU.
    pub(crate) fn split(positions: &[Vec<f32>], max_pos: &[f32], width: usize, heads: usize, theta: f64, device: &Device, dtype: DType) -> Res<Self> {
        let axes = positions.len();
        let t = positions[0].len();
        let n = width / (2 * axes);
        let omega: Vec<f32> = (0..n).map(|k| (std::f64::consts::FRAC_PI_2 * theta.powf(k as f64 / (n - 1).max(1) as f64)) as f32).collect();
        let half = width / 2;
        let pad = half - n * axes;
        let (mut cos, mut sin) = (Vec::with_capacity(t * half), Vec::with_capacity(t * half));
        for i in 0..t {
            cos.extend(std::iter::repeat(1f32).take(pad));
            sin.extend(std::iter::repeat(0f32).take(pad));
            for w in &omega {
                for a in 0..axes {
                    let frac = positions[a][i] / max_pos[a];
                    let angle = w * (frac * 2.0 - 1.0);
                    cos.push((angle as f64).cos() as f32);
                    sin.push((angle as f64).sin() as f32);
                }
            }
        }
        let shape = (t, heads, half / heads);
        Ok(Rope {
            cos: Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?,
        })
    }

    /// Ordinary rotate-half RoPE with one table for every head: token `t` at
    /// `positions[t]`, turned by `positions[t] · inv_freq[k]`, computed in f32.
    pub(crate) fn standard(positions: &[f32], inv_freq: &[f32], device: &Device, dtype: DType) -> Res<Self> {
        let (t, half) = (positions.len(), inv_freq.len());
        let (mut cos, mut sin) = (Vec::with_capacity(t * half), Vec::with_capacity(t * half));
        for &p in positions {
            for &f in inv_freq {
                let angle = p * f;
                cos.push((angle as f64).cos() as f32);
                sin.push((angle as f64).sin() as f32);
            }
        }
        Ok(Rope {
            cos: Tensor::from_vec(cos, (t, 1, half), device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (t, 1, half), device)?.to_dtype(dtype)?,
        })
    }

    /// `x` is `[T, heads, head_dim]`; so is the answer.
    pub(crate) fn rotate(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (t, h, d) = x.dims3()?;
        let x = x.reshape((t, h, 2, d / 2))?;
        let (x1, x2) = (x.narrow(2, 0, 1)?.squeeze(2)?, x.narrow(2, 1, 1)?.squeeze(2)?);
        let (cos, sin) = (self.cos.to_dtype(x1.dtype())?, self.sin.to_dtype(x1.dtype())?);
        let a = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let b = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Tensor::cat(&[a, b], 2)
    }
}

/// LTX's attention block: q, k, v and out projections with biases; q and k
/// RMS-normed over the whole width with learned weights, then rotated; and,
/// when the checkpoint has `to_gate_logits`, each head's output scaled by
/// `2·sigmoid` of a gate computed from the block's input.
pub(crate) struct GatedAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: Option<Linear>,
    heads: usize,
}

impl GatedAttention {
    /// `query` is the width queries come from, `context` the width keys and
    /// values come from, `heads × head_dim` the attention's own width.
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, (query, context): (usize, usize), heads: usize, head_dim: usize, gated: bool) -> Res<Self> {
        let inner = heads * head_dim;
        Ok(GatedAttention {
            q: Linear::load(cx, r, "to_q", query, inner, true)?,
            k: Linear::load(cx, r, "to_k", context, inner, true)?,
            v: Linear::load(cx, r, "to_v", context, inner, true)?,
            out: Linear::load(cx, r, "to_out.0", inner, query, true)?,
            q_norm: RmsNorm::load(cx, r, "q_norm.weight", inner, 1e-6)?,
            k_norm: RmsNorm::load(cx, r, "k_norm.weight", inner, 1e-6)?,
            gate: match gated {
                true => Some(Linear::load(cx, r, "to_gate_logits", query, heads, true)?),
                false => None,
            },
            heads,
        })
    }

    /// `x` is `[T, query]`, `context` `[S, context]` (or `x` itself); the
    /// answer is `[T, query]`. `rope_q` and `rope_k` rotate the queries and
    /// keys when given. No mask: nothing LTX attends over is causal.
    pub(crate) fn forward(&self, x: &Tensor, context: Option<&Tensor>, rope_q: Option<&Rope>, rope_k: Option<&Rope>) -> candle_core::Result<Tensor> {
        let ctx = context.unwrap_or(x);
        let h = self.heads;
        // A Q8_0 projection on the M5's matrix units sums in f32 whatever it
        // was asked in; everything here stays in the input's dtype.
        let dtype = x.dtype();
        let lin = |l: &Linear, y: &Tensor| l.forward_in(y, dtype);
        let dev = x.device();
        // The queries and keys go to their norms as the projections answer:
        // the norm reads a q8 projection's f32 and rounds once.
        let (q, k, v) = span(|| "q, k, v", dev, || Ok((self.q.forward(x)?, self.k.forward(ctx)?, lin(&self.v, ctx)?)))?;
        let (q, k) = span(|| "norm, rope", dev, || {
            Ok((self.q_norm.rotated(&q, rope_q, h, dtype)?, self.k_norm.rotated(&k, rope_k, h, dtype)?))
        })?;
        // The gates' logits as the projection answers them: attention on
        // the matrix units applies them itself, before it rounds.
        let gate = self.gate.as_ref().map(|g| span(|| "gate", dev, || g.forward(x))).transpose()?;
        let o = span(|| "attention", dev, || attend(&q, &k, &v, h, gate.as_ref()))?;
        span(|| "out", dev, || lin(&self.out, &o))
    }

    /// Self-attention with the attention itself skipped: the value
    /// projection of `x`, gated and projected out as the attention's output
    /// would have been. The reference's `all_perturbed` path, which is what
    /// spatio-temporal guidance (STG) asks of a block: what the model makes
    /// when that block cannot look at the other tokens.
    pub(crate) fn value_only(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let v = self.v.forward_in(x, dtype)?;
        let o = match &self.gate {
            Some(g) => {
                let (t, width) = v.dims2()?;
                let gates = (candle_nn::ops::sigmoid(&g.forward(x)?.to_dtype(DType::F32)?)? * 2.0)?;
                let o = v.to_dtype(DType::F32)?.reshape((t, self.heads, width / self.heads))?.broadcast_mul(&gates.unsqueeze(2)?)?;
                o.reshape((t, width))?.to_dtype(dtype)?
            }
            None => v,
        };
        self.out.forward_in(&o, dtype)
    }
}

/// Attention over `[t, heads · d]` queries and `[s, heads · d]` keys and
/// values, answering `[t, heads · d]`, each head scaled by `2·sigmoid` of
/// its `[t, heads]` gate logits when there are some: on the M5's matrix
/// units where `mpp_attention` takes it, which reads these rows as they are
/// and gates in f32 before rounding; otherwise candle's, which splits the
/// heads into copies first, then the gates in the model's dtype.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize, gate: Option<&Tensor>) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    if let Some(o) = crate::mpp_attention::attention(q, k, v, heads, gate)? {
        return Ok(o);
    }
    let one = |x: &Tensor| x.unsqueeze(0);
    let o = crate::image::nn::attention(&one(q)?, &one(k)?, &one(v)?, heads)?.squeeze(0)?;
    match gate {
        Some(g) => {
            let (t, width) = o.dims2()?;
            let gates = (candle_nn::ops::sigmoid(&g.to_dtype(o.dtype())?)? * 2.0)?;
            o.reshape((t, heads, width / heads))?.broadcast_mul(&gates.unsqueeze(2)?)?.reshape((t, width))
        }
        None => Ok(o),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_rope_interleaves_axes_pads_in_front_and_slices_per_head() {
        // Width 16, two heads, two axes: N = 4 frequencies per axis, 8 angles,
        // no padding; one head gets the first four entries, the other the rest.
        let dev = Device::Cpu;
        let p = vec![vec![3f32], vec![5f32]];
        let r = Rope::split(&p, &[10.0, 20.0], 16, 2, 10000.0, &dev, DType::F32).unwrap();
        assert_eq!(r.cos.dims(), [1, 2, 4]);
        let sin = r.sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let w = |k: i32| (std::f64::consts::FRAC_PI_2 * 10000f64.powf(k as f64 / 3.0)) as f32;
        let (f0, f1) = (3f32 / 10.0 * 2.0 - 1.0, 5f32 / 20.0 * 2.0 - 1.0);
        let want = [w(0) * f0, w(0) * f1, w(1) * f0, w(1) * f1, w(2) * f0, w(2) * f1, w(3) * f0, w(3) * f1];
        for (got, a) in sin.iter().zip(want) {
            assert!((got - (a as f64).sin() as f32).abs() < 1e-6);
        }
        // Width 16 over three axes: N = 16 / 6 = 2 frequencies, 6 angles, and
        // two unrotated entries in front to make the 8 a head's half needs.
        let p3 = vec![vec![1f32], vec![2f32], vec![3f32]];
        let r3 = Rope::split(&p3, &[4.0, 4.0, 4.0], 16, 1, 10000.0, &dev, DType::F32).unwrap();
        let c3 = r3.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s3 = r3.sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&c3[..2], &[1.0, 1.0]);
        assert_eq!(&s3[..2], &[0.0, 0.0]);
    }

    #[test]
    fn rotate_turns_each_heads_two_halves() {
        let dev = Device::Cpu;
        // One token, one head of width 4: halves (1, 2) and (3, 4), angles
        // (a, b) from inv_freq at position 1.
        let r = Rope::standard(&[1.0], &[0.5, 0.25], &dev, DType::F32).unwrap();
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 4), &dev).unwrap();
        let y = r.rotate(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (ca, sa, cb, sb) = (0.5f32.cos(), 0.5f32.sin(), 0.25f32.cos(), 0.25f32.sin());
        let want = [1.0 * ca - 3.0 * sa, 2.0 * cb - 4.0 * sb, 3.0 * ca + 1.0 * sa, 4.0 * cb + 2.0 * sb];
        for (g, w) in y.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{y:?} against {want:?}");
        }
    }

    /// The kernel against the chain it replaces: the norm and the rotation
    /// in candle's ops, from the projection's answer rounded to `dt` first.
    #[cfg(target_os = "macos")]
    #[test]
    fn rotated_agrees_with_the_ops_it_replaces() {
        let Ok(dev) = Device::new_metal(0) else { return };
        let rand = |shape: &[usize], dt: DType| Tensor::randn(0f32, 1f32, shape, &dev).unwrap().to_dtype(dt).unwrap();
        // The DiT's video and audio heads, and something small.
        for (width, heads) in [(4096, 32), (2048, 32), (64, 2)] {
            // From a q8 projection's f32, and from the model's own dtype.
            for (from, dt) in [(DType::F32, DType::BF16), (DType::BF16, DType::BF16), (DType::F32, DType::F32), (DType::F16, DType::F16)] {
                for rotate in [true, false] {
                    let n = 5;
                    let norm = RmsNorm { w: rand(&[width], DType::F32), eps: 1e-6 };
                    let angles = (rand(&[n, heads, width / heads / 2], DType::F32) * 40.0).unwrap();
                    let rope = Rope { cos: angles.cos().unwrap().to_dtype(dt).unwrap(), sin: angles.sin().unwrap().to_dtype(dt).unwrap() };
                    let x = (rand(&[n, width], from) * 5.0).unwrap();
                    let before = crate::fused::tests_ran();
                    let got = norm.rotated(&x, rotate.then_some(&rope), heads, dt).unwrap();
                    assert_eq!(crate::fused::tests_ran(), before + 1, "the kernel did not run");
                    let y = norm.forward(&x.to_dtype(dt).unwrap()).unwrap();
                    let want = match rotate {
                        true => rope.rotate(&y.reshape((n, heads, width / heads)).unwrap()).unwrap().reshape((n, width)).unwrap(),
                        false => y,
                    };
                    assert_eq!(got.dtype(), dt);
                    let (a, b) = (got.to_dtype(DType::F32).unwrap(), want.to_dtype(DType::F32).unwrap());
                    assert!(a.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "a NaN or an infinity");
                    let scale = b.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
                    let worst = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() / scale;
                    let ulp = match dt {
                        DType::F32 => 1e-5,
                        DType::F16 => 2e-3,
                        _ => 1.6e-2,
                    };
                    assert!(worst < 4.0 * ulp, "{from:?} to {dt:?}, width {width}, rotate {rotate}: off by {worst}");
                }
            }
        }
    }
}
