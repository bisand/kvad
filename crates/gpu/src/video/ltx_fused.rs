//! The DiT's element-wise ops as one kernel each, on Metal.
//!
//! At stage 2's 24 576 video tokens a block spent about a third of its time
//! outside its matmuls and attention (`examples/ltx_cost.rs`): norms,
//! modulation, rotations, residuals and GELU over `[24576, 4096]`, each a
//! chain of candle ops with casts to f32 and back between them. The q/k
//! norms and RoPE alone took 0.33 s for about 2 GB of traffic.
//!
//! Five kernels take their places, each reading its inputs once and writing
//! once, and a sixth serves the video decoder:
//! - [`modulate`]: `rms(x)·(1 + scale) + shift`, how every norm in a block
//!   is modulated;
//! - [`gated_modulate`]: the gated residual `x + y·g`, and the modulated norm
//!   of its result that follows it;
//! - [`gated_add`]: the gated residual alone, where no norm follows;
//! - [`norm_rope`]: the RMS norm of a query or key projection over its whole
//!   width, then each head's rotation, read straight from a q8 projection's
//!   f32 answer;
//! - [`gelu`]: tanh-GELU, from the up projection's answer to the dtype the
//!   down projection takes;
//! - [`norm_silu`]: the decoder's pixel norm and SiLU, which come before
//!   every convolution in its residual blocks and its output tail.
//!
//! Three more serve the diffusion decoder (`ltx_diffvae`), whose stage 5
//! does the same kinds of thing over 3M tokens:
//! - [`norm_affine`]: `rms(x)·a + b` with f32 rows, a weighted norm and its
//!   modulation in one;
//! - [`head_norm_rope`]: q and k from a fused `qkv` projection, each head
//!   normed on its own and turned by 3D RoPE on adjacent pairs, and v with
//!   them, written into q, k and v for the whole grid;
//! - [`swiglu`]: `silu(g)·u`.
//!
//! Each does its arithmetic in f32 and rounds once, where candle's chains
//! round after each op, so the two differ in the last bit of bf16; the tests
//! hold each to the chain it replaces. Anything a kernel cannot take goes
//! down that chain, as does everything under `KVAD_GPU_FUSED=0`.

use super::ltx_nn::rms;
use candle_core::{DType, Tensor};

/// The kernels, where this device has them.
#[cfg(target_os = "macos")]
fn kernels(device: &candle_core::Device) -> Option<&'static crate::fused::metal::Kernels> {
    crate::fused::metal::library(device, "LTX", metal::SOURCE)
}

fn type_name(dt: DType) -> Option<&'static str> {
    match dt {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        _ => None,
    }
}

/// Whether a kernel can read `t`: on a device with the kernels, in a dtype
/// they were built for, and laid out in one run.
fn readable(t: &Tensor) -> bool {
    #[cfg(target_os = "macos")]
    let on = kernels(t.device()).is_some();
    #[cfg(not(target_os = "macos"))]
    let on = false;
    on && type_name(t.dtype()).is_some() && t.is_contiguous()
}

/// Rows of `e` numbers in `x`'s dtype: one, as `[e]` or `[1, e]`; or two,
/// as `[2, e]`, or with [`Held::Rows`] any number `[k, e]`; see [`Held`].
fn row_of(t: &Tensor, x: &Tensor, e: usize, held: &Held) -> bool {
    let n = t.elem_count();
    let many = match held {
        Held::First(_) => n == 2 * e,
        Held::Rows(_) => n % e == 0,
    };
    readable(t) && t.dtype() == x.dtype() && (n == e || many)
}

/// How the tokens of `x` `[n, e]` read their modulation rows.
///
/// A row argument is one row, `[1, e]`, which every token reads, or more,
/// one for each σ some tokens are at: the DiT modulates each token by its
/// own σ.
/// - [`Held::First`]: two rows `[2, e]`, where the first `n` tokens read row
///   0 and the rest row 1. That is image-to-video: the first latent frame is
///   the picture, held at σ = 0 while the rest is denoised, so the picture's
///   tokens have rows of their own, and they come first. `First(0)` is
///   every token alike.
/// - [`Held::Rows`]: `[k, e]`, token `i` reading row `index[i]`: DFR's
///   conditioning tokens, whose σ is the step's times their mask (1 for the
///   video and its generated keyframes, 0.05 for anchor keyframes, 0 for a
///   reference latent), in whatever order they were appended.
#[derive(Clone, Debug)]
pub(crate) enum Held {
    First(usize),
    /// `[n]` u32, on the tensors' device.
    Rows(Tensor),
}

impl Held {
    /// Every token alike.
    pub(crate) const ALIKE: Held = Held::First(0);

    /// Whether a kernel can read the index.
    fn readable(&self, x: &Tensor) -> bool {
        match self {
            Held::First(_) => true,
            Held::Rows(i) => i.dtype() == DType::U32 && i.is_contiguous() && i.elem_count() == x.dim(0).unwrap_or(0) && i.device().same_device(x.device()),
        }
    }

    /// With [`Held::Rows`], every token's own row of `t`, `[n, e]`, for the
    /// chain; a one-row `t` stays one row, to broadcast.
    fn spread(&self, t: &Tensor, e: usize) -> candle_core::Result<Tensor> {
        match (self, t.elem_count() == e) {
            (Held::Rows(i), false) => t.reshape((t.elem_count() / e, e))?.index_select(i, 0),
            _ => t.reshape((1, e)),
        }
    }
    /// The row the held tokens read, when `held`, or the rest: row 0 or 1
    /// of a two-row argument, and a one-row argument as it is.
    fn pick(t: &Tensor, e: usize, held: bool) -> candle_core::Result<Tensor> {
        match t.elem_count() == 2 * e {
            true => t.reshape((2, e))?.narrow(0, !held as usize, 1),
            false => Ok(t.clone()),
        }
    }

    /// Whether any of `rows` has two rows that tokens of `x` read apart.
    fn splits(&self, x: &Tensor, rows: &[&Tensor]) -> candle_core::Result<bool> {
        let e = x.dim(candle_core::D::Minus1)?;
        Ok(matches!(self, Held::First(n) if *n > 0) && rows.iter().any(|t| t.elem_count() == 2 * e))
    }

    /// `f` on the held tokens with their rows and on the rest with theirs,
    /// the two answers joined again: the chain's way, where no kernel runs.
    fn apart(&self, x: &Tensor, rows: &[&Tensor], f: impl Fn(&Tensor, &[Tensor]) -> candle_core::Result<Tensor>) -> candle_core::Result<Tensor> {
        let (n, e) = x.dims2()?;
        let held = match self {
            Held::First(h) => (*h).min(n),
            Held::Rows(_) => candle_core::bail!("Held::apart is for a split, not an index"),
        };
        let part = |lo: usize, len: usize, h: bool| -> candle_core::Result<Tensor> {
            let rows = rows.iter().map(|t| Held::pick(t, e, h)).collect::<candle_core::Result<Vec<_>>>()?;
            f(&x.narrow(0, lo, len)?, &rows)
        };
        match held == n {
            true => part(0, n, true),
            false => Tensor::cat(&[part(0, held, true)?, part(held, n - held, false)?], 0),
        }
    }

    /// `rows` for a kernel: each as many rows as the most any has, a
    /// one-row argument repeated, so that every row is read at the same
    /// offset; and, for a split, the second row's offset, `e` or 0.
    fn paired(rows: &[&Tensor], e: usize) -> candle_core::Result<(Vec<Tensor>, usize)> {
        let k = rows.iter().map(|t| t.elem_count() / e).max().unwrap_or(1);
        if k == 1 {
            return Ok((rows.iter().map(|t| (*t).clone()).collect(), 0));
        }
        let all = |t: &&Tensor| match t.elem_count() == e {
            true => t.reshape((1, e))?.broadcast_as((k, e))?.contiguous(),
            false => Ok((*t).clone()),
        };
        Ok((rows.iter().map(all).collect::<candle_core::Result<Vec<_>>>()?, e))
    }

    /// What the kernels are told: the split point and the second row's
    /// offset, and the index when there is one.
    fn for_kernel(&self, later: usize) -> ((usize, usize), Option<Tensor>) {
        match self {
            Held::First(n) => ((*n, later), None),
            Held::Rows(i) => ((0, later), Some(i.clone())),
        }
    }
}

/// `rms(x)·(1 + scale) + shift`, over `x`'s last axis; `scale` and `shift`
/// are rows as [`Held`] says.
pub(crate) fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor, held: &Held, eps: f32) -> candle_core::Result<Tensor> {
    let e = x.dim(candle_core::D::Minus1)?;
    if !(readable(x) && row_of(scale, x, e, held) && row_of(shift, x, e, held) && held.readable(x)) {
        if let Held::Rows(_) = held {
            let (scale, shift) = (held.spread(scale, e)?, held.spread(shift, e)?);
            return rms(x, eps as f64)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift);
        }
        if held.splits(x, &[scale, shift])? {
            return held.apart(x, &[scale, shift], |x, r| modulate(x, &r[0], &r[1], &Held::ALIKE, eps));
        }
        return rms(x, eps as f64)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift);
    }
    let (r, later) = Held::paired(&[scale, shift], e)?;
    let (split, index) = held.for_kernel(later);
    metal::modulate(x, None, &r[0], &r[1], split, index.as_ref(), eps)
}

/// `x + y·g`, and [`modulate`] of it: `(modulated, sum)`. The rows are as
/// [`Held`] says.
///
/// Both come back from one buffer, the modulated norm first, so that it
/// starts at offset zero: it goes straight into a projection, and candle's
/// quantised matmul does not honour an offset (see `Proj::forward`).
pub(crate) fn gated_modulate(x: &Tensor, y: &Tensor, g: &Tensor, scale: &Tensor, shift: &Tensor, held: &Held, eps: f32) -> candle_core::Result<(Tensor, Tensor)> {
    let e = x.dim(candle_core::D::Minus1)?;
    let fits = readable(x) && readable(y) && y.dtype() == x.dtype() && y.dims() == x.dims() && [g, scale, shift].iter().all(|t| row_of(t, x, e, held)) && held.readable(x);
    if !fits {
        let sum = gated_add(x, y, g, held)?;
        return Ok((modulate(&sum, scale, shift, held, eps)?, sum));
    }
    let (r, later) = Held::paired(&[g, scale, shift], e)?;
    let (split, index) = held.for_kernel(later);
    let both = metal::modulate(x, Some((y, &r[0])), &r[1], &r[2], split, index.as_ref(), eps)?;
    Ok((both.get(0)?, both.get(1)?))
}

/// `x + y·g`, with `g` rows as [`Held`] says.
pub(crate) fn gated_add(x: &Tensor, y: &Tensor, g: &Tensor, held: &Held) -> candle_core::Result<Tensor> {
    let e = x.dim(candle_core::D::Minus1)?;
    if !(readable(x) && readable(y) && y.dtype() == x.dtype() && y.dims() == x.dims() && row_of(g, x, e, held) && held.readable(x)) {
        if let Held::Rows(_) = held {
            return x + y.broadcast_mul(&held.spread(g, e)?)?;
        }
        if held.splits(x, &[g])? {
            let h = match held {
                Held::First(h) => *h,
                Held::Rows(_) => unreachable!(),
            };
            let (n, h) = (x.dim(0)?, h.min(x.dim(0)?));
            let ys = [y.narrow(0, 0, h)?, y.narrow(0, h, n - h)?];
            let xs = [x.narrow(0, 0, h)?, x.narrow(0, h, n - h)?];
            let parts = (0..2)
                .filter(|&i| xs[i].dim(0).unwrap_or(0) > 0)
                .map(|i| &xs[i] + ys[i].broadcast_mul(&Held::pick(g, e, i == 0)?)?)
                .collect::<candle_core::Result<Vec<_>>>()?;
            return Tensor::cat(&parts, 0);
        }
        return x + y.broadcast_mul(g)?;
    }
    let (r, later) = Held::paired(&[g], e)?;
    let (split, index) = held.for_kernel(later);
    metal::gated_add(x, y, &r[0], split, index.as_ref())
}

/// The RMS norm of `x` `[n, width]` over its whole width under the f32
/// weight `w`, then, with `rope`, each of its heads rotated: the halves of a
/// head turned by the tables' angles, `(x₁·cos − x₂·sin, x₂·cos + x₁·sin)`.
/// The tables are `[n, heads, head_dim / 2]` in `dt`, and the answer is
/// `[n, width]` in `dt`, whatever `x` came in.
///
/// `None` where the kernel cannot take it: the caller has the chain.
pub(crate) fn norm_rope(x: &Tensor, w: &Tensor, rope: Option<(&Tensor, &Tensor)>, heads: usize, eps: f32, dt: DType) -> candle_core::Result<Option<Tensor>> {
    let (n, width) = x.dims2()?;
    let tables = rope.is_none_or(|(c, s)| [c, s].iter().all(|t| readable(t) && t.dtype() == dt && t.dims() == [n, heads, width / heads / 2]));
    let fits = readable(x)
        && type_name(dt).is_some()
        && w.dtype() == DType::F32
        && w.is_contiguous()
        && w.elem_count() == width
        && width % (2 * heads) == 0
        && tables;
    if !fits {
        return Ok(None);
    }
    metal::norm_rope(x, w, rope, width / heads / 2, eps, dt).map(Some)
}

/// `silu(x / rms(x))`, the RMS taken over the channels of a frames-first
/// `[T, C, H, W]` at each pixel: the video decoder's pixel norm and the
/// SiLU after it, in `x`'s dtype, computed in f32 and rounded once.
///
/// `None` where the kernel cannot take it: the caller has the chain.
pub(crate) fn norm_silu(x: &Tensor, eps: f32) -> candle_core::Result<Option<Tensor>> {
    if !(readable(x) && x.rank() == 4) {
        return Ok(None);
    }
    metal::norm_silu(x, eps).map(Some)
}

/// Tanh-GELU of `x`, in `dt`, computed in f32 from whatever `x` is.
pub(crate) fn gelu(x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
    if !(readable(x) && type_name(dt).is_some()) {
        return super::ltx_nn::gelu(&x.to_dtype(dt)?);
    }
    metal::gelu(x, dt)
}

/// `rms(x)·a + b` over `x` `[n, e]`'s last axis, `a` and `b` `[e]` f32
/// rows: a norm's weight, or its weight times `1 + scale` and a shift. In
/// `x`'s dtype, computed in f32 and rounded once. `None` where the kernel
/// cannot take it.
pub(crate) fn norm_affine(x: &Tensor, a: &Tensor, b: &Tensor, eps: f32) -> candle_core::Result<Option<Tensor>> {
    let e = x.dim(candle_core::D::Minus1)?;
    let row = |t: &Tensor| readable(t) && t.dtype() == DType::F32 && t.elem_count() == e;
    if !(readable(x) && x.rank() == 2 && row(a) && row(b)) {
        return Ok(None);
    }
    metal::norm_affine(x, a, b, eps).map(Some)
}

/// Where [`head_norm_rope`]'s tokens are and how they turn.
///
/// Built on every platform, but only the Metal kernel reads the grid's shape
/// and the tables' lengths, so off macOS they are never read.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct HeadRope<'a> {
    /// The grid's rows and columns, and the chunk's first token in it.
    pub h: usize,
    pub w: usize,
    pub row0: usize,
    /// cos then sin of the angles, each for time, rows and columns in turn,
    /// `[len, pairs]` of each axis, all in one f32 tensor; and the pairs
    /// each axis takes of a head's 32.
    pub tables: &'a Tensor,
    pub pairs: [usize; 3],
    pub lens: [usize; 3],
}

/// q, k and v for a chunk of tokens, from their fused projection `qkv`
/// `[rows, 3·width]`: q and k each RMS-normed over every 64-wide head under
/// `qn` and `kn` (f32 `[64]`), q scaled by `q_scale`, both turned pair by
/// adjacent pair `(2i, 2i + 1)` by the angles of the token's place, and v as
/// it is; written into `out` `[3, tokens, width]` at the chunk's rows.
/// Returns whether the kernel ran; if not, nothing is written.
pub(crate) fn head_norm_rope(qkv: &Tensor, qn: &Tensor, kn: &Tensor, rope: &HeadRope<'_>, eps: f32, q_scale: f32, out: &Tensor) -> candle_core::Result<bool> {
    let (rows, three) = qkv.dims2()?;
    let width = three / 3;
    let fits = readable(qkv)
        && readable(out)
        && out.dtype() == qkv.dtype()
        && out.rank() == 3
        && out.dim(0)? == 3
        && out.dim(2)? == width
        && rope.row0 + rows <= out.dim(1)?
        && width % 64 == 0
        && rope.pairs.iter().sum::<usize>() == 32
        && [qn, kn].iter().all(|t| readable(t) && t.dtype() == DType::F32 && t.elem_count() == 64)
        && readable(rope.tables)
        && rope.tables.dtype() == DType::F32;
    if !fits {
        return Ok(false);
    }
    metal::head_norm_rope(qkv, qn, kn, rope, eps, q_scale, out)?;
    Ok(true)
}

/// `silu(g)·u`, in `g`'s dtype, computed in f32 and rounded once: a SwiGLU's
/// gate and its up projection. `None` where the kernel cannot take it.
pub(crate) fn swiglu(g: &Tensor, u: &Tensor) -> candle_core::Result<Option<Tensor>> {
    if !(readable(g) && readable(u) && u.dtype() == g.dtype() && u.dims() == g.dims()) {
        return Ok(None);
    }
    metal::swiglu(g, u).map(Some)
}

#[cfg(target_os = "macos")]
mod metal {
    use super::{kernels, type_name};
    use crate::fused::metal::{buffer, output};
    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, CustomOp1, CustomOp3, DType, Device, Layout, MetalStorage, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
    use objc2_metal::MTLSize;

    pub(super) const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    // The sum of `v` over a threadgroup, in every thread.
    inline float group_sum(float v, threadgroup float *part, uint tpg, uint lane, uint sg) {
        v = simd_sum(v);
        if (lane == 0) part[sg] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        for (uint s = 0; s < (tpg + 31) / 32; s++) total += part[s];
        return total;
    }

    struct ModParams {
        uint e;
        // 1: there is a residual, `x + y·g`, to add first and write to `sum`.
        uint flags;
        float eps;
        // Rows below `held` read `g`, `scale` and `shift` from their start,
        // and the rest from `later` on: `e` when each holds two rows, else 0.
        uint held;
        uint later;
        // 1: rows are read by the index instead, `index[row] · e`.
        uint indexed;
    };

    // One threadgroup a row. `sum` is `x + y·g` rounded to T, which is what the
    // residual stream holds; the norm is taken of that.
    template<typename T>
    kernel void modulate(
        device const T *x [[buffer(0)]],
        device const T *y [[buffer(1)]],
        device const T *g [[buffer(2)]],
        device const T *scale [[buffer(3)]],
        device const T *shift [[buffer(4)]],
        device T *out [[buffer(5)]],
        device T *sum [[buffer(6)]],
        constant ModParams &p [[buffer(7)]],
        device const uint *index [[buffer(8)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        threadgroup float part[32];
        const ulong base = ulong(row) * p.e;
        const bool res = p.flags & 1;
        const uint r = p.indexed ? index[row] * p.e : (row < p.held ? 0 : p.later);
        float acc = 0;
        for (uint j = tid; j < p.e; j += tpg) {
            float v = float(x[base + j]);
            if (res) {
                const T s = T(v + float(y[base + j]) * float(g[r + j]));
                sum[base + j] = s;
                v = float(s);
            }
            acc += v * v;
        }
        const float inv = 1.0f / sqrt(group_sum(acc, part, tpg, lane, sg) / float(p.e) + p.eps);
        for (uint j = tid; j < p.e; j += tpg) {
            const float v = res ? float(sum[base + j]) : float(x[base + j]);
            out[base + j] = T(v * inv * (1.0f + float(scale[r + j])) + float(shift[r + j]));
        }
    }

    template<typename T>
    kernel void gated_add(
        device const T *x [[buffer(0)]],
        device const T *y [[buffer(1)]],
        device const T *g [[buffer(2)]],
        device T *out [[buffer(3)]],
        constant uint &e [[buffer(4)]],
        constant uint &n [[buffer(5)]],
        constant uint4 &held [[buffer(6)]],
        device const uint *index [[buffer(7)]],
        uint i [[thread_position_in_grid]])
    {
        if (i >= n) return;
        // `held`: rows below `.x` read `g` from its start, the rest from `.y`;
        // or with `.z`, each row reads the row `index` gives.
        const uint r = held.z ? index[i / e] * e : (i / e < held.x ? 0 : held.y);
        out[i] = T(float(x[i]) + float(y[i]) * float(g[r + i % e]));
    }

    struct RopeParams {
        uint e;
        // Half a head: the distance between the two numbers RoPE turns together.
        uint half_;
        // 1: rotate.
        uint flags;
        float eps;
    };

    // One threadgroup a row; each thread takes pairs `(i, i + half)` of a head.
    template<typename TI, typename T>
    kernel void norm_rope(
        device const TI *x [[buffer(0)]],
        device const float *w [[buffer(1)]],
        device const T *cos_ [[buffer(2)]],
        device const T *sin_ [[buffer(3)]],
        device T *out [[buffer(4)]],
        constant RopeParams &p [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        threadgroup float part[32];
        const ulong base = ulong(row) * p.e;
        float acc = 0;
        for (uint j = tid; j < p.e; j += tpg) {
            const float v = float(x[base + j]);
            acc += v * v;
        }
        const float inv = 1.0f / sqrt(group_sum(acc, part, tpg, lane, sg) / float(p.e) + p.eps);
        const uint pairs = p.e / 2;
        for (uint q = tid; q < pairs; q += tpg) {
            const uint i = q % p.half_;
            const uint j1 = (q / p.half_) * 2 * p.half_ + i, j2 = j1 + p.half_;
            const float a = float(x[base + j1]) * inv * w[j1];
            const float b = float(x[base + j2]) * inv * w[j2];
            if (p.flags & 1) {
                const ulong at = ulong(row) * pairs + q;
                const float c = float(cos_[at]), s = float(sin_[at]);
                out[base + j1] = T(a * c - b * s);
                out[base + j2] = T(b * c + a * s);
            } else {
                out[base + j1] = T(a);
                out[base + j2] = T(b);
            }
        }
    }

    template<typename TI, typename T>
    kernel void gelu(
        device const TI *x [[buffer(0)]],
        device T *out [[buffer(1)]],
        constant uint &n [[buffer(2)]],
        uint i [[thread_position_in_grid]])
    {
        if (i >= n) return;
        const float v = float(x[i]);
        out[i] = T(0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * v * (1.0f + 0.044715f * v * v))));
    }

    struct PixParams {
        uint c;
        uint hw;
        float eps;
    };

    // One thread a pixel of one frame, walking its channels twice: once for
    // the sum of squares, once to write. Neighbouring threads take
    // neighbouring pixels, so every read is of a run.
    template<typename T>
    kernel void norm_silu(
        device const T *x [[buffer(0)]],
        device T *out [[buffer(1)]],
        constant PixParams &p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= p.hw) return;
        const ulong base = ulong(gid.y) * p.c * p.hw + gid.x;
        float acc = 0;
        for (uint j = 0; j < p.c; j++) {
            const float v = float(x[base + ulong(j) * p.hw]);
            acc += v * v;
        }
        const float inv = 1.0f / sqrt(acc / float(p.c) + p.eps);
        for (uint j = 0; j < p.c; j++) {
            const float v = float(x[base + ulong(j) * p.hw]) * inv;
            out[base + ulong(j) * p.hw] = T(v / (1.0f + exp(-v)));
        }
    }

    // One threadgroup a row: `rms(x)·a + b`, `a` and `b` f32 rows.
    template<typename T>
    kernel void norm_affine(
        device const T *x [[buffer(0)]],
        device const float *a [[buffer(1)]],
        device const float *b [[buffer(2)]],
        device T *out [[buffer(3)]],
        constant uint &e [[buffer(4)]],
        constant float &eps [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        threadgroup float part[32];
        const ulong base = ulong(row) * e;
        float acc = 0;
        for (uint j = tid; j < e; j += tpg) {
            const float v = float(x[base + j]);
            acc += v * v;
        }
        const float inv = 1.0f / sqrt(group_sum(acc, part, tpg, lane, sg) / float(e) + eps);
        for (uint j = tid; j < e; j += tpg) {
            out[base + j] = T(float(x[base + j]) * inv * a[j] + b[j]);
        }
    }

    struct HeadRopeParams {
        uint width, heads, rows, tokens, row0, h, w;
        // Pairs for time and rows; columns take the rest of 32.
        uint pt, ph;
        float eps, q_scale;
        // Where each table starts: cos for time, rows, columns, then sin.
        uint at[6];
    };

    // One SIMD group a token's head of 64; lane l turns the pair (2l, 2l + 1).
    template<typename T>
    kernel void head_norm_rope(
        device const T *qkv [[buffer(0)]],
        device const float *qn [[buffer(1)]],
        device const float *kn [[buffer(2)]],
        device const float *tables [[buffer(3)]],
        device T *out [[buffer(4)]],
        constant HeadRopeParams &p [[buffer(5)]],
        uint group [[threadgroup_position_in_grid]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]])
    {
        // A whole SIMD group leaves together, so its sums stay whole.
        const uint item = group * 4 + sg;
        if (item >= p.rows * p.heads) return;
        const uint r = item / p.heads, head = item % p.heads;
        const ulong token = ulong(p.row0) + r;
        const uint t = uint(token / (p.h * p.w)), y = uint(token / p.w) % p.h, x = uint(token % p.w);
        uint axis, pos, i, np;
        if (lane < p.pt) {
            axis = 0; pos = t; i = lane; np = p.pt;
        } else if (lane < p.pt + p.ph) {
            axis = 1; pos = y; i = lane - p.pt; np = p.ph;
        } else {
            axis = 2; pos = x; i = lane - p.pt - p.ph; np = 32 - p.pt - p.ph;
        }
        const float c = tables[p.at[axis] + pos * np + i], s = tables[p.at[3 + axis] + pos * np + i];
        const ulong src = ulong(r) * 3 * p.width + head * 64 + 2 * lane;
        const ulong dst = token * p.width + head * 64 + 2 * lane;
        const ulong part = ulong(p.tokens) * p.width;
        for (uint which = 0; which < 2; which++) {
            const float x0 = float(qkv[src + which * p.width]), x1 = float(qkv[src + which * p.width + 1]);
            const float inv = 1.0f / sqrt(simd_sum(x0 * x0 + x1 * x1) / 64.0f + p.eps);
            device const float *nw = which == 0 ? qn : kn;
            const float k = which == 0 ? p.q_scale : 1.0f;
            const float a0 = x0 * inv * nw[2 * lane] * k, a1 = x1 * inv * nw[2 * lane + 1] * k;
            out[which * part + dst] = T(a0 * c - a1 * s);
            out[which * part + dst + 1] = T(a0 * s + a1 * c);
        }
        out[2 * part + dst] = qkv[src + 2 * p.width];
        out[2 * part + dst + 1] = qkv[src + 2 * p.width + 1];
    }

    template<typename T>
    kernel void swiglu(
        device const T *g [[buffer(0)]],
        device const T *u [[buffer(1)]],
        device T *out [[buffer(2)]],
        constant uint &n [[buffer(3)]],
        uint i [[thread_position_in_grid]])
    {
        if (i >= n) return;
        const float v = float(g[i]);
        out[i] = T(v / (1.0f + exp(-v)) * float(u[i]));
    }

    #define DIFF(T, N) \
    template [[host_name("norm_affine_" #N)]] kernel void norm_affine<T>( \
        device const T *, device const float *, device const float *, device T *, constant uint &, constant float &, \
        uint, uint, uint, uint, uint); \
    template [[host_name("head_norm_rope_" #N)]] kernel void head_norm_rope<T>( \
        device const T *, device const float *, device const float *, device const float *, device T *, \
        constant HeadRopeParams &, uint, uint, uint); \
    template [[host_name("swiglu_" #N)]] kernel void swiglu<T>( \
        device const T *, device const T *, device T *, constant uint &, uint);
    DIFF(float, f32)
    DIFF(half, f16)
    DIFF(bfloat, bf16)

    #define ONE(T, N) \
    template [[host_name("modulate_" #N)]] kernel void modulate<T>( \
        device const T *, device const T *, device const T *, device const T *, device const T *, \
        device T *, device T *, constant ModParams &, device const uint *, uint, uint, uint, uint, uint); \
    template [[host_name("gated_add_" #N)]] kernel void gated_add<T>( \
        device const T *, device const T *, device const T *, device T *, constant uint &, constant uint &, constant uint4 &, \
        device const uint *, uint); \
    template [[host_name("norm_silu_" #N)]] kernel void norm_silu<T>( \
        device const T *, device T *, constant PixParams &, uint2);
    ONE(float, f32)
    ONE(half, f16)
    ONE(bfloat, bf16)

    #define TWO(TI, NI, T, N) \
    template [[host_name("norm_rope_" #NI "_" #N)]] kernel void norm_rope<TI, T>( \
        device const TI *, device const float *, device const T *, device const T *, device T *, \
        constant RopeParams &, uint, uint, uint, uint, uint); \
    template [[host_name("gelu_" #NI "_" #N)]] kernel void gelu<TI, T>( \
        device const TI *, device T *, constant uint &, uint);
    #define TWO_OUT(TI, NI) TWO(TI, NI, float, f32) TWO(TI, NI, half, f16) TWO(TI, NI, bfloat, bf16)
    TWO_OUT(float, f32)
    TWO_OUT(half, f16)
    TWO_OUT(bfloat, bf16)
    "#;

    fn pipe(dev: &candle_core::MetalDevice, name: &str) -> candle_core::Result<ComputePipeline> {
        let k = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no LTX kernels".into()))?;
        k.pipe(name)
    }

    /// Another tensor's Metal buffer and start. Each is a separate storage
    /// from the op's own, or the same one read again, so taking its read lock
    /// here cannot wait on candle's lock of that one.
    fn metal(t: &Tensor) -> candle_core::Result<(Buffer, usize)> {
        let (s, l) = t.storage_and_layout();
        match &*s {
            Storage::Metal(s) => Ok(buffer(s, l)),
            _ => candle_core::bail!("an LTX kernel's input is not on Metal"),
        }
    }

    /// Threads for one row of `e`: a multiple of a SIMD group, at most 1024.
    fn row_threads(e: usize) -> MTLSize {
        MTLSize { width: e.div_ceil(4).clamp(32, 1024).next_multiple_of(32), height: 1, depth: 1 }
    }

    fn flat(n: usize) -> (MTLSize, MTLSize) {
        (MTLSize { width: n.div_ceil(256), height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 })
    }

    /// `held`: tokens below `.0` read the rows from their start, the rest
    /// from `.1` on.
    pub(super) fn modulate(x: &Tensor, residual: Option<(&Tensor, &Tensor)>, scale: &Tensor, shift: &Tensor, held: (usize, usize), index: Option<&Tensor>, eps: f32)
     -> candle_core::Result<Tensor> {
        let op = Modulate { residual: residual.map(|(y, g)| (y.clone(), g.clone())), scale: scale.clone(), shift: shift.clone(), held, index: index.cloned(), eps };
        x.apply_op1_no_bwd(&op)
    }

    pub(super) fn gated_add(x: &Tensor, y: &Tensor, g: &Tensor, held: (usize, usize), index: Option<&Tensor>) -> candle_core::Result<Tensor> {
        x.apply_op3_no_bwd(y, g, &GatedAdd { held, index: index.cloned() })
    }

    pub(super) fn norm_rope(x: &Tensor, w: &Tensor, rope: Option<(&Tensor, &Tensor)>, half: usize, eps: f32, dt: DType) -> candle_core::Result<Tensor> {
        let op = NormRope { w: w.clone(), rope: rope.map(|(c, s)| (c.clone(), s.clone())), half, eps, dt };
        x.apply_op1_no_bwd(&op)
    }

    pub(super) fn gelu(x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
        x.apply_op1_no_bwd(&Gelu { dt })
    }

    pub(super) fn norm_silu(x: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
        x.apply_op1_no_bwd(&NormSilu { eps })
    }

    pub(super) fn norm_affine(x: &Tensor, a: &Tensor, b: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
        x.apply_op3_no_bwd(a, b, &NormAffine { eps })
    }

    struct NormAffine {
        eps: f32,
    }

    impl CustomOp3 for NormAffine {
        fn name(&self) -> &'static str {
            "ltx_norm_affine"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
         -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_norm_affine runs on Metal only")
        }

        fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, a: &MetalStorage, la: &Layout, b: &MetalStorage, lb: &Layout)
         -> candle_core::Result<(MetalStorage, Shape)> {
            let (dt, dev) = (x.dtype(), x.device());
            let (rows, e) = lx.shape().dims2()?;
            let pipe = pipe(dev, &format!("norm_affine_{}", type_name(dt).unwrap()))?;
            let out = output(dev, rows * e * dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_norm_affine");
            enc.set_compute_pipeline_state(&pipe);
            for (i, (bf, o)) in [(0, buffer(x, lx)), (1, buffer(a, la)), (2, buffer(b, lb))] {
                enc.set_input_buffer(i, Some(&bf), o);
            }
            enc.set_output_buffer(3, Some(&out), 0);
            enc.set_bytes(4, &(e as u32));
            enc.set_bytes(5, &self.eps);
            enc.dispatch_thread_groups(MTLSize { width: rows, height: 1, depth: 1 }, row_threads(e));
            Ok((MetalStorage::new(out, dev.clone(), rows * e, dt), lx.shape().clone()))
        }
    }

    pub(super) fn head_norm_rope(qkv: &Tensor, qn: &Tensor, kn: &Tensor, rope: &super::HeadRope<'_>, eps: f32, q_scale: f32, out: &Tensor)
     -> candle_core::Result<()> {
        let (rows, three) = qkv.dims2()?;
        let width = three / 3;
        let [pt, ph, pw] = rope.pairs;
        let [lt, lh, lw] = rope.lens;
        // cos for time, rows, columns, then sin, each `[len, pairs]`.
        let sizes = [lt * pt, lh * ph, lw * pw];
        let mut at = [0u32; 6];
        for i in 1..6 {
            at[i] = at[i - 1] + sizes[(i - 1) % 3] as u32;
        }
        let params = HeadRopeParams {
            width: width as u32,
            heads: (width / 64) as u32,
            rows: rows as u32,
            tokens: out.dim(1)? as u32,
            row0: rope.row0 as u32,
            h: rope.h as u32,
            w: rope.w as u32,
            pt: pt as u32,
            ph: ph as u32,
            eps,
            q_scale,
            at,
        };
        let op = HeadNormRope { qn: qn.clone(), kn: kn.clone(), tables: rope.tables.clone(), params };
        out.inplace_op2(&qkv.contiguous()?, &op)
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct HeadRopeParams {
        width: u32,
        heads: u32,
        rows: u32,
        tokens: u32,
        row0: u32,
        h: u32,
        w: u32,
        pt: u32,
        ph: u32,
        eps: f32,
        q_scale: f32,
        at: [u32; 6],
    }

    struct HeadNormRope {
        qn: Tensor,
        kn: Tensor,
        tables: Tensor,
        params: HeadRopeParams,
    }

    impl candle_core::InplaceOp2 for HeadNormRope {
        fn name(&self) -> &'static str {
            "ltx_head_norm_rope"
        }

        fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<()> {
            candle_core::bail!("ltx_head_norm_rope runs on Metal only")
        }

        fn metal_fwd(&self, out: &mut MetalStorage, lo: &Layout, qkv: &MetalStorage, lq: &Layout) -> candle_core::Result<()> {
            if !lo.is_contiguous() || !lq.is_contiguous() {
                candle_core::bail!("ltx_head_norm_rope: wants contiguous rows");
            }
            let dev = qkv.device().clone();
            let pipe = pipe(&dev, &format!("head_norm_rope_{}", type_name(qkv.dtype()).unwrap()))?;
            let (qn, kn, tables) = (metal(&self.qn)?, metal(&self.kn)?, metal(&self.tables)?);
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_head_norm_rope");
            enc.set_compute_pipeline_state(&pipe);
            let (qb, qo) = buffer(qkv, lq);
            enc.set_input_buffer(0, Some(&qb), qo);
            enc.set_input_buffer(1, Some(&qn.0), qn.1);
            enc.set_input_buffer(2, Some(&kn.0), kn.1);
            enc.set_input_buffer(3, Some(&tables.0), tables.1);
            enc.set_output_buffer(4, Some(out.buffer()), lo.start_offset() * out.dtype().size_in_bytes());
            enc.set_bytes(5, &self.params);
            let items = (self.params.rows * self.params.heads) as usize;
            // Four SIMD groups a threadgroup, a (token, head) each.
            enc.dispatch_thread_groups(MTLSize { width: items.div_ceil(4), height: 1, depth: 1 }, MTLSize { width: 32, height: 4, depth: 1 });
            Ok(())
        }
    }

    pub(super) fn swiglu(g: &Tensor, u: &Tensor) -> candle_core::Result<Tensor> {
        g.apply_op2_no_bwd(u, &SwiGlu)
    }

    struct SwiGlu;

    impl candle_core::CustomOp2 for SwiGlu {
        fn name(&self) -> &'static str {
            "ltx_swiglu"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_swiglu runs on Metal only")
        }

        fn metal_fwd(&self, g: &MetalStorage, lg: &Layout, u: &MetalStorage, lu: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
            let (dt, dev) = (g.dtype(), g.device());
            let n = lg.shape().elem_count();
            let pipe = pipe(dev, &format!("swiglu_{}", type_name(dt).unwrap()))?;
            let out = output(dev, n * dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_swiglu");
            enc.set_compute_pipeline_state(&pipe);
            let ((gb, go), (ub, uo)) = (buffer(g, lg), buffer(u, lu));
            enc.set_input_buffer(0, Some(&gb), go);
            enc.set_input_buffer(1, Some(&ub), uo);
            enc.set_output_buffer(2, Some(&out), 0);
            enc.set_bytes(3, &(n as u32));
            let (groups, threads) = flat(n);
            enc.dispatch_thread_groups(groups, threads);
            Ok((MetalStorage::new(out, dev.clone(), n, dt), lg.shape().clone()))
        }
    }

    struct NormSilu {
        eps: f32,
    }

    #[repr(C)]
    struct PixParams {
        c: u32,
        hw: u32,
        eps: f32,
    }

    impl CustomOp1 for NormSilu {
        fn name(&self) -> &'static str {
            "ltx_norm_silu"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_norm_silu runs on Metal only")
        }

        fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
            let (dt, dev) = (x.dtype(), x.device());
            let (t, c, h, w) = lx.shape().dims4()?;
            let n = t * c * h * w;
            let pipe = pipe(dev, &format!("norm_silu_{}", type_name(dt).unwrap()))?;
            let out = output(dev, n * dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_norm_silu");
            enc.set_compute_pipeline_state(&pipe);
            let (xb, xo) = buffer(x, lx);
            enc.set_input_buffer(0, Some(&xb), xo);
            enc.set_output_buffer(1, Some(&out), 0);
            enc.set_bytes(2, &PixParams { c: c as u32, hw: (h * w) as u32, eps: self.eps });
            enc.dispatch_thread_groups(
                MTLSize { width: (h * w).div_ceil(256), height: t, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            Ok((MetalStorage::new(out, dev.clone(), n, dt), lx.shape().clone()))
        }
    }

    struct Modulate {
        residual: Option<(Tensor, Tensor)>,
        scale: Tensor,
        shift: Tensor,
        held: (usize, usize),
        /// `[n]` u32: each row's modulation row, in place of the split.
        index: Option<Tensor>,
        eps: f32,
    }

    #[repr(C)]
    struct ModParams {
        e: u32,
        flags: u32,
        eps: f32,
        held: u32,
        later: u32,
        indexed: u32,
    }

    impl CustomOp1 for Modulate {
        fn name(&self) -> &'static str {
            "ltx_modulate"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_modulate runs on Metal only")
        }

        /// `[..]` alone, or `[2, ..]` with a residual: the modulated norm, then
        /// the sum.
        fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
            let (dt, dev) = (x.dtype(), x.device());
            let e = *lx.dims().last().unwrap();
            let n = lx.shape().elem_count();
            let parts = 1 + self.residual.is_some() as usize;
            let pipe = pipe(dev, &format!("modulate_{}", type_name(dt).unwrap()))?;
            let out = output(dev, parts * n * dt.size_in_bytes())?;
            let (sc, sh) = (metal(&self.scale)?, metal(&self.shift)?);
            let res = self.residual.as_ref().map(|(y, g)| Ok::<_, candle_core::Error>((metal(y)?, metal(g)?))).transpose()?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_modulate");
            enc.set_compute_pipeline_state(&pipe);
            let (xb, xo) = buffer(x, lx);
            enc.set_input_buffer(0, Some(&xb), xo);
            // Without a residual, `y`, `g` and `sum` are bound to nothing: the
            // flag is clear, and the kernel never touches them.
            match &res {
                Some(((yb, yo), (gb, go))) => {
                    enc.set_input_buffer(1, Some(yb), *yo);
                    enc.set_input_buffer(2, Some(gb), *go);
                    enc.set_output_buffer(6, Some(&out), n * dt.size_in_bytes());
                }
                None => {
                    enc.set_input_buffer(1, None, 0);
                    enc.set_input_buffer(2, None, 0);
                    enc.set_output_buffer(6, None, 0);
                }
            }
            enc.set_input_buffer(3, Some(&sc.0), sc.1);
            enc.set_input_buffer(4, Some(&sh.0), sh.1);
            enc.set_output_buffer(5, Some(&out), 0);
            let (held, later) = (self.held.0.min(u32::MAX as usize) as u32, self.held.1 as u32);
            let index = self.index.as_ref().map(metal).transpose()?;
            enc.set_bytes(7, &ModParams { e: e as u32, flags: res.is_some() as u32, eps: self.eps, held, later, indexed: index.is_some() as u32 });
            // Without an index, x's buffer stands in, bound but never read.
            match &index {
                Some((ib, io)) => enc.set_input_buffer(8, Some(ib), *io),
                None => enc.set_input_buffer(8, Some(&xb), xo),
            }
            enc.dispatch_thread_groups(MTLSize { width: n / e, height: 1, depth: 1 }, row_threads(e));
            let mut shape = lx.dims().to_vec();
            if parts == 2 {
                shape.insert(0, 2);
            }
            Ok((MetalStorage::new(out, dev.clone(), parts * n, dt), Shape::from(shape)))
        }
    }

    struct GatedAdd {
        held: (usize, usize),
        index: Option<Tensor>,
    }

    impl CustomOp3 for GatedAdd {
        fn name(&self) -> &'static str {
            "ltx_gated_add"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
         -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_gated_add runs on Metal only")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            lx: &Layout,
            y: &MetalStorage,
            ly: &Layout,
            g: &MetalStorage,
            lg: &Layout,
        ) -> candle_core::Result<(MetalStorage, Shape)> {
            let (dt, dev) = (x.dtype(), x.device());
            let (e, n) = (*lx.dims().last().unwrap(), lx.shape().elem_count());
            let pipe = pipe(dev, &format!("gated_add_{}", type_name(dt).unwrap()))?;
            let out = output(dev, n * dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_gated_add");
            enc.set_compute_pipeline_state(&pipe);
            for (i, (b, o)) in [(0, buffer(x, lx)), (1, buffer(y, ly)), (2, buffer(g, lg))] {
                enc.set_input_buffer(i, Some(&b), o);
            }
            enc.set_output_buffer(3, Some(&out), 0);
            enc.set_bytes(4, &(e as u32));
            enc.set_bytes(5, &(n as u32));
            let index = self.index.as_ref().map(metal).transpose()?;
            enc.set_bytes(6, &[self.held.0.min(u32::MAX as usize) as u32, self.held.1 as u32, index.is_some() as u32, 0]);
            match &index {
                Some((ib, io)) => enc.set_input_buffer(7, Some(ib), *io),
                None => {
                    let (xb, xo) = buffer(x, lx);
                    enc.set_input_buffer(7, Some(&xb), xo)
                }
            }
            let (groups, threads) = flat(n);
            enc.dispatch_thread_groups(groups, threads);
            Ok((MetalStorage::new(out, dev.clone(), n, dt), lx.shape().clone()))
        }
    }

    struct NormRope {
        w: Tensor,
        rope: Option<(Tensor, Tensor)>,
        half: usize,
        eps: f32,
        dt: DType,
    }

    #[repr(C)]
    struct RopeParams {
        e: u32,
        half: u32,
        flags: u32,
        eps: f32,
    }

    impl CustomOp1 for NormRope {
        fn name(&self) -> &'static str {
            "ltx_norm_rope"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_norm_rope runs on Metal only")
        }

        fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
            let dev = x.device();
            let (rows, e) = lx.shape().dims2()?;
            let n = rows * e;
            let pipe = pipe(dev, &format!("norm_rope_{}_{}", type_name(x.dtype()).unwrap(), type_name(self.dt).unwrap()))?;
            let out = output(dev, n * self.dt.size_in_bytes())?;
            let w = metal(&self.w)?;
            let tables = self.rope.as_ref().map(|(c, s)| Ok::<_, candle_core::Error>((metal(c)?, metal(s)?))).transpose()?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_norm_rope");
            enc.set_compute_pipeline_state(&pipe);
            let (xb, xo) = buffer(x, lx);
            enc.set_input_buffer(0, Some(&xb), xo);
            enc.set_input_buffer(1, Some(&w.0), w.1);
            match &tables {
                Some(((cb, co), (sb, so))) => {
                    enc.set_input_buffer(2, Some(cb), *co);
                    enc.set_input_buffer(3, Some(sb), *so);
                }
                None => {
                    enc.set_input_buffer(2, None, 0);
                    enc.set_input_buffer(3, None, 0);
                }
            }
            enc.set_output_buffer(4, Some(&out), 0);
            enc.set_bytes(5, &RopeParams { e: e as u32, half: self.half as u32, flags: tables.is_some() as u32, eps: self.eps });
            enc.dispatch_thread_groups(MTLSize { width: rows, height: 1, depth: 1 }, row_threads(e / 2));
            Ok((MetalStorage::new(out, dev.clone(), n, self.dt), lx.shape().clone()))
        }
    }

    struct Gelu {
        dt: DType,
    }

    impl CustomOp1 for Gelu {
        fn name(&self) -> &'static str {
            "ltx_gelu"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("ltx_gelu runs on Metal only")
        }

        fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
            let dev = x.device();
            let n = lx.shape().elem_count();
            let pipe = pipe(dev, &format!("gelu_{}_{}", type_name(x.dtype()).unwrap(), type_name(self.dt).unwrap()))?;
            let out = output(dev, n * self.dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("ltx_gelu");
            enc.set_compute_pipeline_state(&pipe);
            let (xb, xo) = buffer(x, lx);
            enc.set_input_buffer(0, Some(&xb), xo);
            enc.set_output_buffer(1, Some(&out), 0);
            enc.set_bytes(2, &(n as u32));
            let (groups, threads) = flat(n);
            enc.dispatch_thread_groups(groups, threads);
            Ok((MetalStorage::new(out, dev.clone(), n, self.dt), lx.shape().clone()))
        }
    }
}

/// Never reached: [`readable`] is false wherever this is compiled, and every
/// caller asks it first.
#[cfg(not(target_os = "macos"))]
mod metal {
    use super::*;
    pub(super) fn modulate(_: &Tensor, _: Option<(&Tensor, &Tensor)>, _: &Tensor, _: &Tensor, _: (usize, usize), _: Option<&Tensor>, _: f32) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn gated_add(_: &Tensor, _: &Tensor, _: &Tensor, _: (usize, usize), _: Option<&Tensor>) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn norm_rope(_: &Tensor, _: &Tensor, _: Option<(&Tensor, &Tensor)>, _: usize, _: f32, _: DType) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn gelu(_: &Tensor, _: DType) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn norm_silu(_: &Tensor, _: f32) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn norm_affine(_: &Tensor, _: &Tensor, _: &Tensor, _: f32) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn head_norm_rope(_: &Tensor, _: &Tensor, _: &Tensor, _: &super::HeadRope<'_>, _: f32, _: f32, _: &Tensor) -> candle_core::Result<()> {
        unreachable!()
    }
    pub(super) fn swiglu(_: &Tensor, _: &Tensor) -> candle_core::Result<Tensor> {
        unreachable!()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::fused::tests_ran as ran;
    use candle_core::Device;

    fn gpu() -> Option<Device> {
        let dev = Device::new_metal(0).ok()?;
        kernels(&dev).is_some().then_some(dev)
    }

    /// Largest difference, relative to the chain's largest magnitude, after
    /// checking the answer is finite: `max_all` skips NaN, and a kernel's
    /// unwritten output is NaN under test.
    fn off(ours: &Tensor, theirs: &Tensor) -> f32 {
        assert_eq!(ours.dims(), theirs.dims());
        let (a, b) = (ours.to_dtype(DType::F32).unwrap(), theirs.to_dtype(DType::F32).unwrap());
        assert!(a.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "a NaN or an infinity");
        let scale = b.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().max(1e-6);
        let worst = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        worst / scale
    }

    /// What one rounding costs, relative to the largest magnitude.
    fn ulp(dt: DType) -> f32 {
        match dt {
            DType::F32 => 1e-5,
            DType::F16 => 2e-3,
            _ => 1.6e-2,
        }
    }

    fn rand(shape: &[usize], dt: DType, dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, dev).unwrap().to_dtype(dt).unwrap()
    }

    /// Rows of a table, as the block's are: views at an offset.
    fn rows(e: usize, dt: DType, dev: &Device) -> (Tensor, Tensor, Tensor) {
        let t = (rand(&[4, e], dt, dev) * 0.3).unwrap();
        (t.narrow(0, 1, 1).unwrap(), t.narrow(0, 2, 1).unwrap(), t.narrow(0, 3, 1).unwrap())
    }

    /// The decoder's pixel norm and SiLU against candle's chain: its widths,
    /// a frame too small for a threadgroup, and a strided input declined.
    #[test]
    fn norm_silu_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for dims in [[3, 128, 5, 7], [2, 512, 16, 24], [2, 1024, 3, 4], [1, 32, 1, 1]] {
                let x = (rand(&dims, dt, &dev) * 3.0).unwrap();
                let before = ran();
                let got = norm_silu(&x, 1e-8).unwrap().expect("the kernel declined");
                assert_eq!(ran(), before + 1, "the kernel did not run");
                let want = crate::video::conv3d::pixel_norm(&x, 1e-8).unwrap().silu().unwrap();
                let got = off(&got, &want);
                assert!(got < 3.0 * ulp(dt), "{dt:?} {dims:?}: off by {got}");
            }
        }
        let x = rand(&[2, 8, 4, 4], DType::BF16, &dev).transpose(2, 3).unwrap();
        assert!(norm_silu(&x, 1e-8).unwrap().is_none());
    }

    #[test]
    fn modulate_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for (m, e) in [(3, 4096), (5, 2048), (2, 100), (1, 24)] {
                let x = rand(&[m, e], dt, &dev);
                let (scale, shift, _) = rows(e, dt, &dev);
                let before = ran();
                let got = modulate(&x, &scale, &shift, &Held::ALIKE, 1e-6).unwrap();
                assert_eq!(ran(), before + 1, "the kernel did not run");
                let want = rms(&x, 1e-6).unwrap().broadcast_mul(&(&scale + 1.0).unwrap()).unwrap().broadcast_add(&shift).unwrap();
                let got = off(&got, &want);
                assert!(got < 3.0 * ulp(dt), "{dt:?} [{m}, {e}]: off by {got}");
            }
        }
    }

    #[test]
    fn gated_modulate_and_gated_add_agree_with_the_ops_they_replace() {
        let Some(dev) = gpu() else { return };
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for (m, e) in [(3, 4096), (5, 2048), (2, 100)] {
                // `x` at an offset, as the residual stream is: the sum half of
                // the last call's output.
                let x = rand(&[2, m, e], dt, &dev).get(1).unwrap();
                let y = rand(&[m, e], dt, &dev);
                let (g, scale, shift) = rows(e, dt, &dev);
                let before = ran();
                let (h, sum) = gated_modulate(&x, &y, &g, &scale, &shift, &Held::ALIKE, 1e-6).unwrap();
                let added = gated_add(&x, &y, &g, &Held::ALIKE).unwrap();
                assert_eq!(ran(), before + 2, "a kernel did not run");
                let want_sum = (&x + y.broadcast_mul(&g).unwrap()).unwrap();
                let want = rms(&want_sum, 1e-6).unwrap().broadcast_mul(&(&scale + 1.0).unwrap()).unwrap().broadcast_add(&shift).unwrap();
                for (what, got, want, tol) in [("sum", &sum, &want_sum, 2.0), ("add", &added, &want_sum, 2.0), ("norm", &h, &want, 4.0)] {
                    let got = off(got, want);
                    assert!(got < tol * ulp(dt), "{dt:?} [{m}, {e}] {what}: off by {got}");
                }
            }
        }
    }

    /// With an index, each token reads its own row of `[k, e]` arguments,
    /// on the kernels and down the chain alike: three rows, read out of
    /// order, as DFR's conditioning tokens read theirs; each token's answer
    /// what the one-row ops make of it with its row.
    #[test]
    fn indexed_tokens_read_the_row_their_index_names() {
        let Some(dev) = gpu() else { return };
        let (m, e) = (9, 256);
        let which: Vec<u32> = vec![0, 0, 2, 1, 1, 2, 0, 1, 2];
        for dt in [DType::F32, DType::BF16] {
            let x = rand(&[m, e], dt, &dev);
            let y = rand(&[m, e], dt, &dev);
            let three = |dev: &Device| (rand(&[3, e], dt, dev) * 0.3).unwrap();
            let (g, scale, shift) = (three(&dev), three(&dev), three(&dev));
            let token = |i: usize| {
                let r = |t: &Tensor| t.narrow(0, which[i] as usize, 1).unwrap();
                let sum = (x.narrow(0, i, 1).unwrap() + y.narrow(0, i, 1).unwrap().broadcast_mul(&r(&g)).unwrap()).unwrap();
                let norm = rms(&sum, 1e-6).unwrap().broadcast_mul(&(r(&scale) + 1.0).unwrap()).unwrap().broadcast_add(&r(&shift)).unwrap();
                (norm, sum)
            };
            let parts: Vec<(Tensor, Tensor)> = (0..m).map(token).collect();
            let want_norm = Tensor::cat(&parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>(), 0).unwrap();
            let want_sum = Tensor::cat(&parts.iter().map(|p| p.1.clone()).collect::<Vec<_>>(), 0).unwrap();
            for device in [dev.clone(), Device::Cpu] {
                let on = |t: &Tensor| t.to_device(&device).unwrap();
                let held = Held::Rows(Tensor::new(which.as_slice(), &device).unwrap());
                let before = ran();
                let (norm, sum) = gated_modulate(&on(&x), &on(&y), &on(&g), &on(&scale), &on(&shift), &held, 1e-6).unwrap();
                let added = gated_add(&on(&x), &on(&y), &on(&g), &held).unwrap();
                let alone = modulate(&on(&want_sum), &on(&scale), &on(&shift), &held, 1e-6).unwrap();
                if device.is_metal() {
                    assert_eq!(ran(), before + 3, "{dt:?}: the kernels did not all run");
                }
                for (what, got, want, tol) in [("sum", &sum, &want_sum, 2.0), ("add", &added, &want_sum, 2.0), ("norm", &norm, &want_norm, 4.0), ("modulate", &alone, &want_norm, 4.0)] {
                    let got = off(&got.to_device(&dev).unwrap(), want);
                    assert!(got < tol * ulp(dt), "{dt:?} on {device:?}, {what}: off by {got}");
                }
            }
        }
    }

    /// Held tokens read row 0 of a two-row argument and the rest row 1, on
    /// the kernels and down the chain alike, whichever arguments have two
    /// rows; and each part is what the one-row ops make of it.
    #[test]
    fn held_tokens_read_their_own_rows() {
        let Some(dev) = gpu() else { return };
        let (m, e, h) = (7, 256, 3);
        for dt in [DType::F32, DType::BF16] {
            let x = rand(&[m, e], dt, &dev);
            let y = rand(&[m, e], dt, &dev);
            let two = |dev: &Device| (rand(&[2, e], dt, dev) * 0.3).unwrap();
            let (g, scale, shift) = (two(&dev), two(&dev), two(&dev));
            let one = rows(e, dt, &dev).0;
            // What each part should be: the one-row ops on it.
            let by_parts = |g: &Tensor, scale: &Tensor, shift: &Tensor| -> (Tensor, Tensor) {
                let part = |lo: usize, len: usize, r: usize| {
                    let pick = |t: &Tensor| match t.dim(0).unwrap() {
                        2 => t.narrow(0, r, 1).unwrap(),
                        _ => t.clone(),
                    };
                    let sum = (x.narrow(0, lo, len).unwrap() + y.narrow(0, lo, len).unwrap().broadcast_mul(&pick(g)).unwrap()).unwrap();
                    let norm = rms(&sum, 1e-6).unwrap().broadcast_mul(&(pick(scale) + 1.0).unwrap()).unwrap().broadcast_add(&pick(shift)).unwrap();
                    (norm, sum)
                };
                let (a, b) = (part(0, h, 0), part(h, m - h, 1));
                (Tensor::cat(&[a.0, b.0], 0).unwrap(), Tensor::cat(&[a.1, b.1], 0).unwrap())
            };
            // Every argument two rows; and the gate one row, as the audio →
            // video gate is.
            for g in [&g, &one] {
                let (want_norm, want_sum) = by_parts(g, &scale, &shift);
                for device in [dev.clone(), Device::Cpu] {
                    let on = |t: &Tensor| t.to_device(&device).unwrap();
                    let (norm, sum) = gated_modulate(&on(&x), &on(&y), &on(g), &on(&scale), &on(&shift), &Held::First(h), 1e-6).unwrap();
                    let added = gated_add(&on(&x), &on(&y), &on(g), &Held::First(h)).unwrap();
                    let alone = modulate(&on(&want_sum), &on(&scale), &on(&shift), &Held::First(h), 1e-6).unwrap();
                    for (what, got, want, tol) in [("sum", &sum, &want_sum, 2.0), ("add", &added, &want_sum, 2.0), ("norm", &norm, &want_norm, 4.0), ("modulate", &alone, &want_norm, 4.0)] {
                        let got = off(&got.to_device(&dev).unwrap(), want);
                        assert!(got < tol * ulp(dt), "{dt:?} on {device:?}, {what}: off by {got}");
                    }
                }
            }
        }
    }

    #[test]
    fn gelu_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        // From a q8 projection's f32, and from the model's own dtype.
        for (from, to) in [(DType::F32, DType::BF16), (DType::BF16, DType::BF16), (DType::F32, DType::F32), (DType::F16, DType::F16)] {
            let x = (rand(&[7, 1000], from, &dev) * 3.0).unwrap();
            let before = ran();
            let got = gelu(&x, to).unwrap();
            assert_eq!(ran(), before + 1, "the kernel did not run");
            assert_eq!(got.dtype(), to);
            let want = super::super::ltx_nn::gelu(&x.to_dtype(to).unwrap()).unwrap();
            let got = off(&got, &want);
            assert!(got < 2.0 * ulp(to), "{from:?} to {to:?}: off by {got}");
        }
    }

    /// Off Metal, or for what they cannot read, the kernels decline, and the
    /// answer is the chain's.
    #[test]
    fn what_the_kernels_cannot_read_goes_down_the_chain() {
        let Some(dev) = gpu() else { return };
        let (m, e) = (3, 64);
        let x = rand(&[m, e], DType::F32, &dev);
        let (scale, shift, _) = rows(e, DType::F32, &dev);
        let before = ran();
        // A row in another dtype, a strided input, and the CPU.
        modulate(&x, &scale.to_dtype(DType::BF16).unwrap(), &shift, &Held::ALIKE, 1e-6).unwrap_err();
        let wide = rand(&[m, 2 * e], DType::F32, &dev).narrow(1, 0, e).unwrap();
        modulate(&wide, &scale, &shift, &Held::ALIKE, 1e-6).unwrap();
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap();
        modulate(&cpu(&x), &cpu(&scale), &cpu(&shift), &Held::ALIKE, 1e-6).unwrap();
        let w = Tensor::ones(e, DType::F32, &dev).unwrap();
        assert!(norm_rope(&wide, &w, None, 2, 1e-6, DType::F32).unwrap().is_none());
        assert!(norm_rope(&x, &w.to_dtype(DType::BF16).unwrap(), None, 2, 1e-6, DType::F32).unwrap().is_none());
        assert_eq!(ran(), before, "a kernel ran on what it cannot read");
    }
}
