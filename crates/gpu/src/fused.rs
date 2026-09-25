//! Several of a decode step's small ops as one kernel each, on Metal.
//!
//! A decode token of Qwen2.5-1.5B at q8 took 25 kernels a layer, and only
//! eight of them were matmuls. The other seventeen have next to nothing to
//! do — a norm over 1536 numbers, a rotation of two heads — and each still
//! costs a dispatch and a memory barrier. `decode_breakdown` put them at
//! about 0.7 ms of an 8.4 ms token, and skipping eleven of them a layer, with
//! the matmuls left alone, took 0.6 ms off.
//!
//! Three kernels take their places:
//! - [`add_rms_norm`]: the residual add, and the RMS norm of its result that
//!   always follows it;
//! - [`silu_mul`]: `silu(gate) * up`, from the one tensor a merged gate-and-up
//!   projection returns;
//! - [`rope_cache`]: everything between the Q/K/V projection and attention —
//!   the biases, Qwen3's per-head norms, the rotation, the cast to the
//!   cache's dtype, and the writes into the cache.
//!
//! Each does its arithmetic in f32 and rounds once where candle's chain
//! rounds after every op, so the two can differ in the last bit of bf16 or
//! f16. The tests below hold every kernel to the chain it replaces.
//!
//! Anything a kernel cannot take — another device, a dtype it was not built
//! for, a strided input — goes down that chain instead, which is the code
//! that ran before these kernels existed. `KVAD_GPU_FUSED=0` sends
//! everything that way, to compare.
//!
//! # Measured
//!
//! Together with the merged projections (`Loader::proj_cat`), a layer of
//! Qwen2.5-1.5B went from 25 kernels to 10. Raced against the build before,
//! three alternating rounds each:
//! - q8 decode: 114–117 → 126–129 tok/s at 128 positions of context, and
//!   102–108 → 110–115 at 2,048;
//! - bf16 decode: 62–64 → 68–69 tok/s, most of it the merged projections;
//! - prefill of 1,812 tokens: 0.36 → 0.31 s in bf16, 0.87 → 0.82 s at q8;
//! - Qwen3-14B at q8: the same within noise. A token there reads 15 GB of
//!   weights, and a millisecond of small ops is lost in it.
//!
//! Greedy text is word for word the same at q8, on both models. In bf16 it
//! parts after about forty tokens, as rounding in different places does;
//! against an f32 run of the same model, the fused path's KL divergence was
//! 4.18e-3 to candle's 4.32e-3, and its top-1 agreement 583 of 600 tokens to
//! 576.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{ops, rotary_emb};

/// Whether this device runs the kernels here.
pub(crate) fn available(device: &Device) -> bool {
    #[cfg(target_os = "macos")]
    return metal::kernels(device).is_some();
    #[cfg(not(target_os = "macos"))]
    return { let _ = device; false };
}

fn type_name(dt: DType) -> Option<&'static str> {
    match dt {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Residual add and RMS norm
// ---------------------------------------------------------------------------

/// `x + d`, and the RMS norm of that under `w`: `(norm, sum)`.
///
/// Both come back from one buffer, the norm first, so that the norm starts
/// at offset zero: it goes straight into a projection, and candle's quantised
/// matmul does not honour an offset (see `Proj::forward`).
pub(crate) fn add_rms_norm(x: &Tensor, d: &Tensor, w: &Tensor, eps: f32) -> candle_core::Result<(Tensor, Tensor)> {
    let fits = available(x.device())
        && type_name(x.dtype()).is_some()
        && d.dtype() == x.dtype()
        && w.dtype() == x.dtype()
        && x.dims() == d.dims()
        && w.dims() == [x.dim(D::Minus1)?]
        && x.is_contiguous()
        && d.is_contiguous()
        && w.is_contiguous();
    if !fits {
        let sum = (x + d)?;
        return Ok((ops::rms_norm(&sum, w, eps)?, sum));
    }
    metal::add_rms_norm(x, d, w, eps)
}

// ---------------------------------------------------------------------------
// SiLU and multiply
// ---------------------------------------------------------------------------

/// `silu(gate) * up`, where `gu` is `[.., 2n]`: the gate's `n` columns, then
/// the up projection's, as [`crate::common::Loader::proj_cat`] merges them.
pub(crate) fn silu_mul(gu: &Tensor, n: usize) -> candle_core::Result<Tensor> {
    let fits = available(gu.device())
        && type_name(gu.dtype()).is_some()
        && gu.dim(D::Minus1)? == 2 * n
        && gu.is_contiguous();
    if !fits {
        let gate = ops::silu(&gu.narrow(D::Minus1, 0, n)?.contiguous()?)?;
        return gate * gu.narrow(D::Minus1, n, n)?.contiguous()?;
    }
    metal::silu_mul(gu, n)
}

// ---------------------------------------------------------------------------
// From the Q/K/V projection to the cache
// ---------------------------------------------------------------------------

/// What turns a merged Q/K/V projection's output into attention's inputs.
pub(crate) struct Heads<'a> {
    pub(crate) n_head: usize,
    pub(crate) n_kv: usize,
    pub(crate) head_dim: usize,
    /// All three projections' biases, merged as their weights are.
    pub(crate) bias: Option<&'a Tensor>,
    /// Qwen3's per-head norms, one `head_dim` vector each.
    pub(crate) q_norm: Option<&'a Tensor>,
    pub(crate) k_norm: Option<&'a Tensor>,
    pub(crate) eps: f32,
}

/// Whether [`rope_cache`] takes this step.
///
/// `qkv` is the merged projection's `[m, (n_head + 2 n_kv) head_dim]`; `cos`
/// and `sin` are `[m, head_dim / 2]`. Each part must share `qkv`'s dtype, be
/// contiguous, and be the size it says it is.
pub(crate) fn rope_cache_fits(qkv: &Tensor, heads: &Heads, cos: &Tensor, sin: &Tensor, cache: DType, q: DType) -> bool {
    let dt = qkv.dtype();
    let hd = heads.head_dim;
    let same = |t: Option<&Tensor>, len: usize| {
        t.is_none_or(|t| t.dtype() == dt && t.is_contiguous() && t.dims() == [len])
    };
    let Ok((m, width)) = qkv.dims2() else { return false };
    available(qkv.device())
        && type_name(dt).is_some()
        && type_name(cache).is_some()
        && type_name(q).is_some()
        && qkv.is_contiguous()
        && width == (heads.n_head + 2 * heads.n_kv) * hd
        && hd % 2 == 0
        && hd / 2 <= 1024
        && same(heads.bias, width)
        && same(heads.q_norm, hd)
        && same(heads.k_norm, hd)
        && [cos, sin].iter().all(|t| t.dtype() == dt && t.is_contiguous() && t.dims() == [m, hd / 2])
}

/// Rotate the queries and keys of `qkv` and write keys and values into the
/// caches `k` and `v` at position `pos`: the queries come back as
/// `[1, n_head, m, head_dim]` in `q`'s dtype.
///
/// `k` and `v` are the whole cache buffers, `[1, n_kv, cap, head_dim]`,
/// which this writes in place. The caller has made room for `m` more
/// positions after `pos`, and checked [`rope_cache_fits`].
pub(crate) fn rope_cache(
    qkv: &Tensor,
    heads: &Heads,
    cos: &Tensor,
    sin: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pos: usize,
    q: DType,
) -> candle_core::Result<Tensor> {
    metal::rope_cache(qkv, heads, cos, sin, k, v, pos, q)
}

/// The queries, keys and values of `qkv`, each `[1, heads, m, hd]`, the
/// queries and keys rotated: what [`rope_cache`] does, in candle's ops, for
/// every device and every step that kernel does not take. The keys and
/// values are the caller's to put in the cache.
pub(crate) fn heads(
    qkv: &Tensor,
    heads: &Heads,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let (m, _) = qkv.dims2()?;
    let (n_head, n_kv, hd) = (heads.n_head, heads.n_kv, heads.head_dim);
    let qkv = match heads.bias {
        Some(b) => qkv.broadcast_add(b)?,
        None => qkv.clone(),
    };
    // [m, heads * hd] -> [1, m, heads, hd]: the heads are split out
    // with `hd` still last, which is the layout the per-head norm
    // below wants. Reshaped whole and then taken apart along the heads, so
    // that each part is a view and the transpose below is its one copy.
    let all = qkv.reshape((1, m, n_head + 2 * n_kv, hd))?;
    let (q, k, v) = (all.narrow(2, 0, n_head)?, all.narrow(2, n_head, n_kv)?, all.narrow(2, n_head + n_kv, n_kv)?);

    // Qwen3 normalises each head of Q and K here, between the
    // projection and the rotation. Nothing else in this family does,
    // and for everything else this is a no-op.
    let q = norm_heads(&q, heads.q_norm, heads.eps)?;
    let k = norm_heads(&k, heads.k_norm, heads.eps)?;

    // -> [1, heads, m, hd], which is what the rotary kernel and the
    // batched attention matmuls want.
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;

    let q = rotary_emb::rope(&q, cos, sin)?;
    let k = rotary_emb::rope(&k, cos, sin)?;
    Ok((q, k, v))
}

/// RMSNorm every head of a projection shaped `[.., heads, head_dim]`, if this
/// model has the weights for it.
///
/// The mirror of `norm_heads` in [`kvad::model::llama`], and much shorter for
/// one reason: `rms_norm` normalises along the last axis, so putting the heads
/// on the axis before it turns a loop over heads into a single kernel call.
///
/// `None` is every other model in this family, and costs one branch per layer.
fn norm_heads(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> candle_core::Result<Tensor> {
    match weight {
        None => Ok(x.clone()),
        Some(w) => ops::rms_norm(&x.contiguous()?, w, eps),
    }
}

#[cfg(target_os = "macos")]
pub(crate) mod metal {
    use super::{type_name, Heads};
    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, CustomOp1, CustomOp3, DType, Device, Layout, MetalStorage, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
    use objc2_metal::MTLSize;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    // One threadgroup a row. `sum` is `x + d` rounded to T, which is what the
    // residual stream holds; the norm is taken of that, not of the unrounded sum.
    template<typename T>
    kernel void add_rms_norm(
        device const T *x [[buffer(0)]],
        device const T *d [[buffer(1)]],
        device const T *w [[buffer(2)]],
        device T *norm [[buffer(3)]],
        device T *sum [[buffer(4)]],
        constant uint &e [[buffer(5)]],
        constant float &eps [[buffer(6)]],
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
            const T s = T(float(x[base + j]) + float(d[base + j]));
            sum[base + j] = s;
            acc += float(s) * float(s);
        }
        acc = simd_sum(acc);
        if (lane == 0) part[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        for (uint s = 0; s < (tpg + 31) / 32; s++) total += part[s];
        const float inv = 1.0f / sqrt(total / float(e) + eps);
        for (uint j = tid; j < e; j += tpg) {
            norm[base + j] = T(float(sum[base + j]) * inv * float(w[j]));
        }
    }

    // `gu` is rows of `2n`: the gate's `n`, then the up projection's.
    template<typename T>
    kernel void silu_mul(
        device const T *gu [[buffer(0)]],
        device T *out [[buffer(1)]],
        constant uint &n [[buffer(2)]],
        uint2 at [[thread_position_in_grid]])
    {
        if (at.x >= n) return;
        const ulong i = ulong(at.y) * 2 * n + at.x;
        const float g = float(gu[i]);
        const T s = T(g / (1.0f + exp(-g)));
        out[ulong(at.y) * n + at.x] = T(float(s) * float(gu[i + n]));
    }

    struct RopeParams {
        uint m, n_head, n_kv, hd;
        // Where the first of the `m` positions goes in the cache, and how many
        // positions the cache has room for.
        uint pos, cap;
        // 1: add `bias`; 2: norm the queries' heads; 4: norm the keys'.
        uint flags;
        float eps;
    };

    // One threadgroup per head per token, `hd / 2` threads: thread `i` holds the
    // pair RoPE turns together, `i` and `i + hd / 2`. Heads are numbered queries,
    // then keys, then values, as the merged projection lays them out.
    template<typename TI, typename TQ, typename TC>
    kernel void rope_cache(
        device const TI *qkv [[buffer(0)]],
        device const TI *bias [[buffer(1)]],
        device const TI *q_norm [[buffer(2)]],
        device const TI *k_norm [[buffer(3)]],
        device const TI *cos [[buffer(4)]],
        device const TI *sin [[buffer(5)]],
        device TQ *q [[buffer(6)]],
        device TC *k_cache [[buffer(7)]],
        device TC *v_cache [[buffer(8)]],
        constant RopeParams &p [[buffer(9)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 local [[thread_position_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        threadgroup float part[32];
        const uint h = tg.x, t = tg.y, i = local.x, half_ = p.hd / 2;
        const uint width = (p.n_head + 2 * p.n_kv) * p.hd;
        const uint col = h * p.hd + i;
        float a = float(qkv[ulong(t) * width + col]);
        float c = float(qkv[ulong(t) * width + col + half_]);
        if (p.flags & 1) {
            a = float(TI(a + float(bias[col])));
            c = float(TI(c + float(bias[col + half_])));
        }
        // 0 a query head, 1 a key head, 2 a value head.
        const uint kind = h < p.n_head ? 0 : h < p.n_head + p.n_kv ? 1 : 2;
        const uint hh = kind == 0 ? h : kind == 1 ? h - p.n_head : h - p.n_head - p.n_kv;

        // The same for every thread of a threadgroup, so the barrier inside is
        // reached by all of them or by none.
        const bool normed = (kind == 0 && (p.flags & 2)) || (kind == 1 && (p.flags & 4));
        if (normed) {
            device const TI *w = kind == 0 ? q_norm : k_norm;
            const float acc = simd_sum(a * a + c * c);
            if (lane == 0) part[sg] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0;
            for (uint s = 0; s < (half_ + 31) / 32; s++) total += part[s];
            const float inv = 1.0f / sqrt(total / float(p.hd) + p.eps);
            a = float(TI(a * inv * float(w[i])));
            c = float(TI(c * inv * float(w[i + half_])));
        }

        const ulong cached = (ulong(hh) * p.cap + p.pos + t) * p.hd + i;
        if (kind == 2) {
            v_cache[cached] = TC(a);
            v_cache[cached + half_] = TC(c);
            return;
        }
        const float co = float(cos[t * half_ + i]), si = float(sin[t * half_ + i]);
        const float r1 = a * co - c * si, r2 = a * si + c * co;
        if (kind == 0) {
            const ulong o = (ulong(hh) * p.m + t) * p.hd + i;
            q[o] = TQ(r1);
            q[o + half_] = TQ(r2);
        } else {
            k_cache[cached] = TC(r1);
            k_cache[cached + half_] = TC(r2);
        }
    }

    #define ONE(T, N) \
    template [[host_name("add_rms_norm_" #N)]] kernel void add_rms_norm<T>( \
        device const T *, device const T *, device const T *, device T *, device T *, \
        constant uint &, constant float &, uint, uint, uint, uint, uint); \
    template [[host_name("silu_mul_" #N)]] kernel void silu_mul<T>( \
        device const T *, device T *, constant uint &, uint2);
    ONE(float, f32)
    ONE(half, f16)
    ONE(bfloat, bf16)

    #define ROPE(TI, NI, TQ, NQ, TC, NC) \
    template [[host_name("rope_cache_" #NI "_" #NQ "_" #NC)]] kernel void rope_cache<TI, TQ, TC>( \
        device const TI *, device const TI *, device const TI *, device const TI *, \
        device const TI *, device const TI *, device TQ *, device TC *, device TC *, \
        constant RopeParams &, uint2, uint2, uint, uint);
    #define ROPE_C(TI, NI, TQ, NQ) \
        ROPE(TI, NI, TQ, NQ, float, f32) ROPE(TI, NI, TQ, NQ, half, f16) ROPE(TI, NI, TQ, NQ, bfloat, bf16)
    #define ROPE_Q(TI, NI) \
        ROPE_C(TI, NI, float, f32) ROPE_C(TI, NI, half, f16) ROPE_C(TI, NI, bfloat, bf16)
    ROPE_Q(float, f32)
    ROPE_Q(half, f16)
    ROPE_Q(bfloat, bf16)
    "#;

    /// A library of kernels, built once per process from its source.
    pub(crate) struct Kernels {
        lib: candle_metal_kernels::metal::Library,
        device: candle_metal_kernels::metal::Device,
        pipes: Mutex<HashMap<String, ComputePipeline>>,
    }

    impl Kernels {
        pub(crate) fn pipe(&self, name: &str) -> candle_core::Result<ComputePipeline> {
            let mut pipes = self.pipes.lock().unwrap();
            if let Some(p) = pipes.get(name) {
                return Ok(p.clone());
            }
            let f = self.lib.get_function(name, None).map_err(candle_core::Error::wrap)?;
            let p = self.device.new_compute_pipeline_state_with_function(&f).map_err(candle_core::Error::wrap)?;
            pipes.insert(name.to_string(), p.clone());
            Ok(p)
        }
    }

    /// The decode kernels: `None` where there is no Metal device, the source
    /// did not build, or `KVAD_GPU_FUSED=0` says not to.
    pub(super) fn kernels(device: &Device) -> Option<&'static Kernels> {
        library(device, "decode", SOURCE)
    }

    /// The kernels in `source`, built the first time they are asked for and
    /// kept for the life of the process; `what` names them if they do not
    /// build. `None` where there is no Metal device, the source did not
    /// build, or `KVAD_GPU_FUSED=0` says not to use fused kernels at all.
    pub(crate) fn library(device: &Device, what: &'static str, source: &'static str) -> Option<&'static Kernels> {
        if matches!(std::env::var("KVAD_GPU_FUSED").as_deref(), Ok("0") | Ok("false")) {
            return None;
        }
        build(device, what, source, None)
    }

    /// As [`library`], for kernels on the M5's matrix units: built as Metal 4,
    /// which their tensor API needs, and only where `mpp` runs, so
    /// `KVAD_GPU_MPP=0` turns them off with the matmuls.
    pub(crate) fn tensor_library(device: &Device, what: &'static str, source: &'static str) -> Option<&'static Kernels> {
        if !crate::mpp::available(device) {
            return None;
        }
        let opts = objc2_metal::MTLCompileOptions::new();
        opts.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version4_0);
        build(device, what, source, Some(&*opts))
    }

    fn build(device: &Device, what: &'static str, source: &'static str, opts: Option<&objc2_metal::MTLCompileOptions>)
     -> Option<&'static Kernels> {
        static LIBS: OnceLock<Mutex<HashMap<&'static str, Option<&'static Kernels>>>> = OnceLock::new();
        let Device::Metal(md) = device else { return None };
        let mut libs = LIBS.get_or_init(Default::default).lock().unwrap();
        *libs.entry(what).or_insert_with(|| {
            match md.metal_device().new_library_with_source(source, opts) {
                Ok(lib) => Some(&*Box::leak(Box::new(Kernels { lib, device: md.metal_device().clone(), pipes: Mutex::new(HashMap::new()) }))),
                Err(e) => {
                    eprintln!("kvad: the fused {what} kernels did not build, so candle's ops are used: {e}");
                    None
                }
            }
        })
    }

    /// A tensor's Metal buffer and its start, in bytes.
    pub(crate) fn buffer(storage: &MetalStorage, layout: &Layout) -> (Buffer, usize) {
        (storage.buffer().clone(), layout.start_offset() * storage.dtype().size_in_bytes())
    }

    #[cfg(test)]
    thread_local! {
        /// Kernels this thread has launched, so a test can tell a kernel's
        /// answer from the fallback's.
        pub(crate) static RAN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// A new output buffer. Under test it starts as NaN, so a kernel that left
    /// part of it unwritten cannot pass by agreeing with the last tenant.
    pub(crate) fn output(dev: &candle_core::MetalDevice, bytes: usize) -> candle_core::Result<std::sync::Arc<Buffer>> {
        let out = dev.allocate_buffer(bytes)?;
        #[cfg(test)]
        {
            let mut blit = dev.blit_command_encoder()?;
            blit.fill_buffer(&out, (0, bytes), 0xff);
            RAN.with(|r| r.set(r.get() + 1));
        }
        Ok(out)
    }

    pub(super) fn add_rms_norm(x: &Tensor, d: &Tensor, w: &Tensor, eps: f32) -> candle_core::Result<(Tensor, Tensor)> {
        let both = x.apply_op3_no_bwd(d, w, &AddRmsNorm { eps })?;
        Ok((both.get(0)?, both.get(1)?))
    }

    pub(super) fn silu_mul(gu: &Tensor, n: usize) -> candle_core::Result<Tensor> {
        gu.apply_op1_no_bwd(&SiluMul { n })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_cache(
        qkv: &Tensor,
        heads: &Heads,
        cos: &Tensor,
        sin: &Tensor,
        k: &Tensor,
        v: &Tensor,
        pos: usize,
        q: DType,
    ) -> candle_core::Result<Tensor> {
        let op = RopeCache {
            n_head: heads.n_head,
            n_kv: heads.n_kv,
            hd: heads.head_dim,
            eps: heads.eps,
            bias: heads.bias.cloned(),
            q_norm: heads.q_norm.cloned(),
            k_norm: heads.k_norm.cloned(),
            k: k.clone(),
            v: v.clone(),
            pos,
            q,
        };
        qkv.apply_op3_no_bwd(cos, sin, &op)
    }

    struct AddRmsNorm {
        eps: f32,
    }

    impl CustomOp3 for AddRmsNorm {
        fn name(&self) -> &'static str {
            "add_rms_norm"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
         -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("add_rms_norm runs on Metal only")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            lx: &Layout,
            d: &MetalStorage,
            ld: &Layout,
            w: &MetalStorage,
            lw: &Layout,
        ) -> candle_core::Result<(MetalStorage, Shape)> {
            let dt = x.dtype();
            let e = lx.dims()[lx.dims().len() - 1];
            let n = lx.shape().elem_count();
            let dev = x.device();
            let k = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no fused kernels".into()))?;
            let pipe = k.pipe(&format!("add_rms_norm_{}", type_name(dt).unwrap()))?;
            let out = output(dev, 2 * n * dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("add_rms_norm");
            enc.set_compute_pipeline_state(&pipe);
            let ((xb, xo), (db, dof), (wb, wo)) = (buffer(x, lx), buffer(d, ld), buffer(w, lw));
            enc.set_input_buffer(0, Some(&xb), xo);
            enc.set_input_buffer(1, Some(&db), dof);
            enc.set_input_buffer(2, Some(&wb), wo);
            enc.set_output_buffer(3, Some(&out), 0);
            enc.set_output_buffer(4, Some(&out), n * dt.size_in_bytes());
            enc.set_bytes(5, &(e as u32));
            enc.set_bytes(6, &self.eps);
            enc.dispatch_thread_groups(
                MTLSize { width: n / e, height: 1, depth: 1 },
                MTLSize { width: e.min(1024).next_multiple_of(32), height: 1, depth: 1 },
            );
            let mut shape = vec![2];
            shape.extend_from_slice(lx.dims());
            Ok((MetalStorage::new(out, dev.clone(), 2 * n, dt), Shape::from(shape)))
        }
    }

    struct SiluMul {
        n: usize,
    }

    impl CustomOp1 for SiluMul {
        fn name(&self) -> &'static str {
            "silu_mul"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("silu_mul runs on Metal only")
        }

        fn metal_fwd(&self, gu: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
            let (dt, n) = (gu.dtype(), self.n);
            let rows = l.shape().elem_count() / (2 * n);
            let dev = gu.device();
            let k = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no fused kernels".into()))?;
            let pipe = k.pipe(&format!("silu_mul_{}", type_name(dt).unwrap()))?;
            let out = output(dev, rows * n * dt.size_in_bytes())?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("silu_mul");
            enc.set_compute_pipeline_state(&pipe);
            let (b, o) = buffer(gu, l);
            enc.set_input_buffer(0, Some(&b), o);
            enc.set_output_buffer(1, Some(&out), 0);
            enc.set_bytes(2, &(n as u32));
            let per = 256;
            enc.dispatch_thread_groups(
                MTLSize { width: n.div_ceil(per), height: rows, depth: 1 },
                MTLSize { width: per, height: 1, depth: 1 },
            );
            let mut shape = l.dims().to_vec();
            *shape.last_mut().unwrap() = n;
            Ok((MetalStorage::new(out, dev.clone(), rows * n, dt), Shape::from(shape)))
        }
    }

    struct RopeCache {
        n_head: usize,
        n_kv: usize,
        hd: usize,
        eps: f32,
        bias: Option<Tensor>,
        q_norm: Option<Tensor>,
        k_norm: Option<Tensor>,
        /// The cache buffers, written in place; candle's op traits have no slot
        /// for a second output, let alone a third.
        k: Tensor,
        v: Tensor,
        pos: usize,
        q: DType,
    }

    #[repr(C)]
    struct RopeParams {
        m: u32,
        n_head: u32,
        n_kv: u32,
        hd: u32,
        pos: u32,
        cap: u32,
        flags: u32,
        eps: f32,
    }

    impl CustomOp3 for RopeCache {
        fn name(&self) -> &'static str {
            "rope_cache"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
         -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("rope_cache runs on Metal only")
        }

        fn metal_fwd(
            &self,
            qkv: &MetalStorage,
            lqkv: &Layout,
            cos: &MetalStorage,
            lcos: &Layout,
            sin: &MetalStorage,
            lsin: &Layout,
        ) -> candle_core::Result<(MetalStorage, Shape)> {
            let (m, _) = lqkv.shape().dims2()?;
            let (_, n_kv, cap, hd) = self.k.dims4()?;
            if n_kv != self.n_kv || hd != self.hd || self.v.dims() != self.k.dims() || self.pos + m > cap {
                candle_core::bail!("rope_cache: no room for {m} positions at {} in {:?}", self.pos, self.k.shape());
            }
            let cache = self.k.dtype();
            let name = format!(
                "rope_cache_{}_{}_{}",
                type_name(qkv.dtype()).unwrap(),
                type_name(self.q).unwrap(),
                type_name(cache).unwrap()
            );
            let dev = qkv.device();
            let k = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no fused kernels".into()))?;
            let pipe = k.pipe(&name)?;
            let q_len = self.n_head * m * hd;
            let out = output(dev, q_len * self.q.size_in_bytes())?;

            // The other tensors' buffers. Each is a separate storage from `qkv`,
            // so taking its lock here cannot wait on candle's lock of that one.
            let metal = |t: &Tensor| -> candle_core::Result<(Buffer, usize)> {
                let (s, l) = t.storage_and_layout();
                match &*s {
                    Storage::Metal(s) => Ok(buffer(s, l)),
                    _ => candle_core::bail!("rope_cache: a tensor is not on Metal"),
                }
            };
            let opt = |t: &Option<Tensor>| t.as_ref().map(metal).transpose();
            let (bias, qn, kn) = (opt(&self.bias)?, opt(&self.q_norm)?, opt(&self.k_norm)?);
            let (kc, vc) = (metal(&self.k)?, metal(&self.v)?);
            let flags = bias.is_some() as u32 | (qn.is_some() as u32) << 1 | (kn.is_some() as u32) << 2;
            let params = RopeParams {
                m: m as u32,
                n_head: self.n_head as u32,
                n_kv: self.n_kv as u32,
                hd: hd as u32,
                pos: self.pos as u32,
                cap: cap as u32,
                flags,
                eps: self.eps,
            };

            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_label("rope_cache");
            enc.set_compute_pipeline_state(&pipe);
            let (b, o) = buffer(qkv, lqkv);
            enc.set_input_buffer(0, Some(&b), o);
            // An absent part is bound to nothing: its flag is clear, and the
            // kernel never reads it.
            for (i, part) in [(1, &bias), (2, &qn), (3, &kn)] {
                match part {
                    Some((b, o)) => enc.set_input_buffer(i, Some(b), *o),
                    None => enc.set_input_buffer(i, None, 0),
                }
            }
            let ((cb, co), (sb, so)) = (buffer(cos, lcos), buffer(sin, lsin));
            enc.set_input_buffer(4, Some(&cb), co);
            enc.set_input_buffer(5, Some(&sb), so);
            enc.set_output_buffer(6, Some(&out), 0);
            // Declared as outputs, so that attention, which reads them next,
            // waits for these writes.
            enc.set_output_buffer(7, Some(&kc.0), kc.1);
            enc.set_output_buffer(8, Some(&vc.0), vc.1);
            enc.set_bytes(9, &params);
            enc.dispatch_thread_groups(
                MTLSize { width: self.n_head + 2 * self.n_kv, height: m, depth: 1 },
                MTLSize { width: hd / 2, height: 1, depth: 1 },
            );
            Ok((MetalStorage::new(out, dev.clone(), q_len, self.q), Shape::from((1, self.n_head, m, hd))))
        }
    }
}

/// Never reached: [`available`] is false wherever this is compiled, and every
/// caller asks it first.
#[cfg(not(target_os = "macos"))]
pub(crate) mod metal {
    use super::*;
    pub(super) fn add_rms_norm(_: &Tensor, _: &Tensor, _: &Tensor, _: f32) -> candle_core::Result<(Tensor, Tensor)> {
        unreachable!()
    }
    pub(super) fn silu_mul(_: &Tensor, _: usize) -> candle_core::Result<Tensor> {
        unreachable!()
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_cache(
        _: &Tensor, _: &Heads, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: usize, _: DType,
    ) -> candle_core::Result<Tensor> {
        unreachable!()
    }
}

/// Kernels this thread has launched: for a test that must know it is
/// looking at a kernel's answer and not the fallback's.
#[cfg(test)]
pub(crate) fn tests_ran() -> usize {
    #[cfg(target_os = "macos")]
    return metal::RAN.with(|r| r.get());
    #[cfg(not(target_os = "macos"))]
    return 0;
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn ran() -> usize {
        tests_ran()
    }

    fn gpu() -> Option<Device> {
        let dev = Device::new_metal(0).ok()?;
        available(&dev).then_some(dev)
    }

    /// Largest difference, relative to the reference's largest magnitude,
    /// after checking the answer is finite: `max_all` skips NaN, and a
    /// kernel's unwritten output is NaN under test.
    fn off(ours: &Tensor, theirs: &Tensor) -> f32 {
        assert_eq!(ours.dims(), theirs.dims());
        let (a, b) = (ours.to_dtype(DType::F32).unwrap(), theirs.to_dtype(DType::F32).unwrap());
        assert!(a.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "a NaN or an infinity");
        let scale = b.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().max(1e-6);
        let worst = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        worst / scale
    }

    /// What one rounding of a dtype costs, relative: the two sides round in
    /// different places, and each place can cost this much.
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

    #[test]
    fn add_rms_norm_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for (m, e) in [(1, 1536), (5, 1000), (3, 4096), (2, 24)] {
                // `x` at an offset, as the residual stream is after the first
                // layer: the sum half of the last call's output.
                let x = rand(&[2, m, e], dt, &dev).get(1).unwrap();
                let (d, w) = (rand(&[m, e], dt, &dev), rand(&[e], dt, &dev));
                let before = ran();
                let (norm, sum) = add_rms_norm(&x, &d, &w, 1e-6).unwrap();
                assert_eq!(ran(), before + 1, "the kernel did not run");
                let want_sum = (&x + &d).unwrap();
                let want = ops::rms_norm(&want_sum, &w, 1e-6).unwrap();
                assert_eq!(off(&sum, &want_sum), 0.0, "{dt:?} [{m}, {e}]: the sum is one rounding either way");
                let got = off(&norm, &want);
                assert!(got < 2.0 * ulp(dt), "{dt:?} [{m}, {e}]: the norm is off by {got}");
            }
        }
    }

    #[test]
    fn silu_mul_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for (m, n) in [(1, 8960), (4, 100), (3, 1)] {
                let gu = rand(&[m, 2 * n], dt, &dev);
                let before = ran();
                let got = silu_mul(&gu, n).unwrap();
                assert_eq!(ran(), before + 1, "the kernel did not run");
                let want = (ops::silu(&gu.narrow(1, 0, n).unwrap()).unwrap() * gu.narrow(1, n, n).unwrap()).unwrap();
                let got = off(&got, &want);
                assert!(got < 2.0 * ulp(dt), "{dt:?} [{m}, {n}]: off by {got}");
            }
        }
    }

    /// Every case the model can send: with and without biases and Qwen3's
    /// norms, one position and several, an empty cache and one with
    /// positions in it, and each pairing of dtypes a load can make.
    #[test]
    fn rope_cache_agrees_with_the_ops_it_replaces() {
        let Some(dev) = gpu() else { return };
        let (n_head, n_kv, cap) = (4, 2, 16);
        let dtypes = [
            // Quantised: f32 arithmetic, an f16 cache, and queries in either.
            (DType::F32, DType::F16, DType::F16),
            (DType::F32, DType::F16, DType::F32),
            (DType::F32, DType::F32, DType::F32),
            (DType::BF16, DType::BF16, DType::BF16),
            (DType::F16, DType::F16, DType::F16),
        ];
        for (dt, cache, q_dt) in dtypes {
            for hd in [64, 128] {
                for (bias, norms) in [(false, false), (true, false), (false, true), (true, true)] {
                    for (m, pos) in [(1, 0), (1, 7), (3, 0), (5, 9)] {
                        let what = format!("{dt:?}/{cache:?}/{q_dt:?} hd {hd} bias {bias} norms {norms} m {m} at {pos}");
                        let width = (n_head + 2 * n_kv) * hd;
                        let qkv = rand(&[m, width], dt, &dev);
                        let b = bias.then(|| rand(&[width], dt, &dev));
                        let (qn, kn) = match norms {
                            true => (Some(rand(&[hd], dt, &dev)), Some(rand(&[hd], dt, &dev))),
                            false => (None, None),
                        };
                        let heads = Heads {
                            n_head,
                            n_kv,
                            head_dim: hd,
                            bias: b.as_ref(),
                            q_norm: qn.as_ref(),
                            k_norm: kn.as_ref(),
                            eps: 1e-6,
                        };
                        // Rotations are cos and sin of something: keep them
                        // in range, as the model's table does.
                        let angles = rand(&[m, hd / 2], DType::F32, &dev);
                        let (cos, sin) = (angles.cos().unwrap().to_dtype(dt).unwrap(), angles.sin().unwrap().to_dtype(dt).unwrap());
                        assert!(rope_cache_fits(&qkv, &heads, &cos, &sin, cache, q_dt), "{what}: declined");

                        // Caches full of a value nothing computes, to see
                        // that the kernel writes where it should and nowhere
                        // else.
                        let fill = || (Tensor::ones((1, n_kv, cap, hd), cache, &dev).unwrap() * 7.0).unwrap();
                        let (kc, vc) = (fill(), fill());
                        let before = ran();
                        let q = rope_cache(&qkv, &heads, &cos, &sin, &kc, &vc, pos, q_dt).unwrap();
                        assert_eq!(ran(), before + 1, "{what}: the kernel did not run");
                        assert_eq!(q.dtype(), q_dt);

                        let (wq, wk, wv) = super::heads(&qkv, &heads, &cos, &sin).unwrap();
                        let got = off(&q, &wq);
                        assert!(got < 3.0 * ulp(dt).max(ulp(q_dt)), "{what}: queries off by {got}");
                        for (name, c, want) in [("keys", &kc, wk), ("values", &vc, wv)] {
                            let written = c.narrow(2, pos, m).unwrap();
                            let got = off(&written, &want);
                            assert!(got < 3.0 * ulp(dt).max(ulp(cache)), "{what}: {name} off by {got}");
                            let untouched = Tensor::cat(
                                &[c.narrow(2, 0, pos).unwrap(), c.narrow(2, pos + m, cap - pos - m).unwrap()],
                                2,
                            )
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap();
                            assert!(untouched.iter().all(|&x| x == 7.0), "{what}: {name} written outside [{pos}, {})", pos + m);
                        }
                    }
                }
            }
        }
    }

    /// The checks the model relies on to fall back, rather than feed the
    /// kernel something it would misread.
    #[test]
    fn rope_cache_declines_what_it_cannot_read() {
        let Some(dev) = gpu() else { return };
        let (n_head, n_kv, hd, m) = (4, 2, 64, 2);
        let width = (n_head + 2 * n_kv) * hd;
        let heads = |bias| Heads { n_head, n_kv, head_dim: hd, bias, q_norm: None, k_norm: None, eps: 1e-6 };
        let qkv = rand(&[m, width], DType::F32, &dev);
        let cs = rand(&[m, hd / 2], DType::F32, &dev);
        let f = DType::F32;
        assert!(rope_cache_fits(&qkv, &heads(None), &cs, &cs, f, f));
        // Strided rows.
        let wide = rand(&[m, 2 * width], DType::F32, &dev).narrow(1, 0, width).unwrap();
        assert!(!rope_cache_fits(&wide, &heads(None), &cs, &cs, f, f));
        // A width that is not the heads'.
        let short = rand(&[m, width - hd], DType::F32, &dev);
        assert!(!rope_cache_fits(&short, &heads(None), &cs, &cs, f, f));
        // A bias in another dtype.
        let b = rand(&[width], DType::BF16, &dev);
        assert!(!rope_cache_fits(&qkv, &heads(Some(&b)), &cs, &cs, f, f));
        // Rotations for another number of positions.
        let cs3 = rand(&[m + 1, hd / 2], DType::F32, &dev);
        assert!(!rope_cache_fits(&qkv, &heads(None), &cs3, &cs3, f, f));
        // A dtype there is no kernel for.
        assert!(!rope_cache_fits(&qkv, &heads(None), &cs, &cs, DType::U8, f));
        // Off Metal, nothing fits.
        let cpu = qkv.to_device(&Device::Cpu).unwrap();
        let cs_cpu = cs.to_device(&Device::Cpu).unwrap();
        assert!(!rope_cache_fits(&cpu, &heads(None), &cs_cpu, &cs_cpu, f, f));
    }
}
