//! The DiT's element-wise ops as one kernel each, on Metal.
//!
//! At stage 2's 24 576 video tokens a block spent about a third of its time
//! outside its matmuls and attention (`examples/ltx_cost.rs`): norms,
//! modulation, rotations, residuals and GELU over `[24576, 4096]`, each a
//! chain of candle ops with casts to f32 and back between them. The q/k
//! norms and RoPE alone took 0.33 s for about 2 GB of traffic.
//!
//! Five kernels take their places, each reading its inputs once and writing
//! once:
//! - [`modulate`]: `rms(x)·(1 + scale) + shift`, how every norm in a block
//!   is modulated;
//! - [`gated_modulate`]: the gated residual `x + y·g`, and the modulated norm
//!   of its result that follows it;
//! - [`gated_add`]: the gated residual alone, where no norm follows;
//! - [`norm_rope`]: the RMS norm of a query or key projection over its whole
//!   width, then each head's rotation, read straight from a q8 projection's
//!   f32 answer;
//! - [`gelu`]: tanh-GELU, from the up projection's answer to the dtype the
//!   down projection takes.
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

/// A row of `e` numbers in `x`'s dtype, as `[e]` or `[1, e]`.
fn row_of(t: &Tensor, x: &Tensor, e: usize) -> bool {
    readable(t) && t.dtype() == x.dtype() && t.elem_count() == e
}

/// `rms(x)·(1 + scale) + shift`, over `x`'s last axis; `scale` and `shift`
/// are one row each.
pub(crate) fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let e = x.dim(candle_core::D::Minus1)?;
    if !(readable(x) && row_of(scale, x, e) && row_of(shift, x, e)) {
        return rms(x, eps as f64)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift);
    }
    metal::modulate(x, None, scale, shift, eps)
}

/// `x + y·g`, and [`modulate`] of it: `(modulated, sum)`. `g` is one row.
///
/// Both come back from one buffer, the modulated norm first, so that it
/// starts at offset zero: it goes straight into a projection, and candle's
/// quantised matmul does not honour an offset (see `Proj::forward`).
pub(crate) fn gated_modulate(x: &Tensor, y: &Tensor, g: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> candle_core::Result<(Tensor, Tensor)> {
    let e = x.dim(candle_core::D::Minus1)?;
    let fits = readable(x) && readable(y) && y.dtype() == x.dtype() && y.dims() == x.dims() && [g, scale, shift].iter().all(|t| row_of(t, x, e));
    if !fits {
        let sum = (x + y.broadcast_mul(g)?)?;
        return Ok((modulate(&sum, scale, shift, eps)?, sum));
    }
    let both = metal::modulate(x, Some((y, g)), scale, shift, eps)?;
    Ok((both.get(0)?, both.get(1)?))
}

/// `x + y·g`, with `g` one row.
pub(crate) fn gated_add(x: &Tensor, y: &Tensor, g: &Tensor) -> candle_core::Result<Tensor> {
    let e = x.dim(candle_core::D::Minus1)?;
    if !(readable(x) && readable(y) && y.dtype() == x.dtype() && y.dims() == x.dims() && row_of(g, x, e)) {
        return x + y.broadcast_mul(g)?;
    }
    metal::gated_add(x, y, g)
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

/// Tanh-GELU of `x`, in `dt`, computed in f32 from whatever `x` is.
pub(crate) fn gelu(x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
    if !(readable(x) && type_name(dt).is_some()) {
        return super::ltx_nn::gelu(&x.to_dtype(dt)?);
    }
    metal::gelu(x, dt)
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
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        threadgroup float part[32];
        const ulong base = ulong(row) * p.e;
        const bool res = p.flags & 1;
        float acc = 0;
        for (uint j = tid; j < p.e; j += tpg) {
            float v = float(x[base + j]);
            if (res) {
                const T s = T(v + float(y[base + j]) * float(g[j]));
                sum[base + j] = s;
                v = float(s);
            }
            acc += v * v;
        }
        const float inv = 1.0f / sqrt(group_sum(acc, part, tpg, lane, sg) / float(p.e) + p.eps);
        for (uint j = tid; j < p.e; j += tpg) {
            const float v = res ? float(sum[base + j]) : float(x[base + j]);
            out[base + j] = T(v * inv * (1.0f + float(scale[j])) + float(shift[j]));
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
        uint i [[thread_position_in_grid]])
    {
        if (i >= n) return;
        out[i] = T(float(x[i]) + float(y[i]) * float(g[i % e]));
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

    #define ONE(T, N) \
    template [[host_name("modulate_" #N)]] kernel void modulate<T>( \
        device const T *, device const T *, device const T *, device const T *, device const T *, \
        device T *, device T *, constant ModParams &, uint, uint, uint, uint, uint); \
    template [[host_name("gated_add_" #N)]] kernel void gated_add<T>( \
        device const T *, device const T *, device const T *, device T *, constant uint &, constant uint &, uint);
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

    pub(super) fn modulate(x: &Tensor, residual: Option<(&Tensor, &Tensor)>, scale: &Tensor, shift: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
        let op = Modulate { residual: residual.map(|(y, g)| (y.clone(), g.clone())), scale: scale.clone(), shift: shift.clone(), eps };
        x.apply_op1_no_bwd(&op)
    }

    pub(super) fn gated_add(x: &Tensor, y: &Tensor, g: &Tensor) -> candle_core::Result<Tensor> {
        x.apply_op3_no_bwd(y, g, &GatedAdd)
    }

    pub(super) fn norm_rope(x: &Tensor, w: &Tensor, rope: Option<(&Tensor, &Tensor)>, half: usize, eps: f32, dt: DType) -> candle_core::Result<Tensor> {
        let op = NormRope { w: w.clone(), rope: rope.map(|(c, s)| (c.clone(), s.clone())), half, eps, dt };
        x.apply_op1_no_bwd(&op)
    }

    pub(super) fn gelu(x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
        x.apply_op1_no_bwd(&Gelu { dt })
    }

    struct Modulate {
        residual: Option<(Tensor, Tensor)>,
        scale: Tensor,
        shift: Tensor,
        eps: f32,
    }

    #[repr(C)]
    struct ModParams {
        e: u32,
        flags: u32,
        eps: f32,
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
            enc.set_bytes(7, &ModParams { e: e as u32, flags: res.is_some() as u32, eps: self.eps });
            enc.dispatch_thread_groups(MTLSize { width: n / e, height: 1, depth: 1 }, row_threads(e));
            let mut shape = lx.dims().to_vec();
            if parts == 2 {
                shape.insert(0, 2);
            }
            Ok((MetalStorage::new(out, dev.clone(), parts * n, dt), Shape::from(shape)))
        }
    }

    struct GatedAdd;

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
    pub(super) fn modulate(_: &Tensor, _: Option<(&Tensor, &Tensor)>, _: &Tensor, _: &Tensor, _: f32) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn gated_add(_: &Tensor, _: &Tensor, _: &Tensor) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn norm_rope(_: &Tensor, _: &Tensor, _: Option<(&Tensor, &Tensor)>, _: usize, _: f32, _: DType) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    pub(super) fn gelu(_: &Tensor, _: DType) -> candle_core::Result<Tensor> {
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

    #[test]
    fn modulate_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for (m, e) in [(3, 4096), (5, 2048), (2, 100), (1, 24)] {
                let x = rand(&[m, e], dt, &dev);
                let (scale, shift, _) = rows(e, dt, &dev);
                let before = ran();
                let got = modulate(&x, &scale, &shift, 1e-6).unwrap();
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
                let (h, sum) = gated_modulate(&x, &y, &g, &scale, &shift, 1e-6).unwrap();
                let added = gated_add(&x, &y, &g).unwrap();
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
        modulate(&x, &scale.to_dtype(DType::BF16).unwrap(), &shift, 1e-6).unwrap_err();
        let wide = rand(&[m, 2 * e], DType::F32, &dev).narrow(1, 0, e).unwrap();
        modulate(&wide, &scale, &shift, 1e-6).unwrap();
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu).unwrap();
        modulate(&cpu(&x), &cpu(&scale), &cpu(&shift), 1e-6).unwrap();
        let w = Tensor::ones(e, DType::F32, &dev).unwrap();
        assert!(norm_rope(&wide, &w, None, 2, 1e-6, DType::F32).unwrap().is_none());
        assert!(norm_rope(&x, &w.to_dtype(DType::BF16).unwrap(), None, 2, 1e-6, DType::F32).unwrap().is_none());
        assert_eq!(ran(), before, "a kernel ran on what it cannot read");
    }
}
