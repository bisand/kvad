//! Attention on the M5 GPU's neural accelerators (#51, #52).
//!
//! candle's fused attention is MLX's older kernel. It multiplies with
//! `simdgroup_matrix` on the ordinary ALUs, and at stage 2 of LTX-2.5 it
//! runs video self-attention at about 6.5 TFLOP/s, 48% of a DiT block. This
//! one does both of attention's products through `matmul2d`, as `mpp` does
//! the projections', and never writes the scores down. At stage 2's shapes
//! it measured 3× candle on video self-attention, 7× on the attention to
//! the text, and 20× or more on the small attentions between video and
//! sound, where candle's kernel has too few queries or keys to fill the GPU.
//!
//! # The kernel
//!
//! Flash attention, laid out as MLX's attention for the M5
//! (`steel_attention_nax`) lays it out. A threadgroup is 64 queries of one
//! head: four SIMD groups of 16 rows. Each walks the keys 32 at a time:
//! 1. `S = Q · Kᵀ` for its 16 queries and the step's keys, in f32, on the
//!    matrix unit. Its queries are loaded once and kept in registers.
//! 2. The softmax, online: each row's running maximum and sum are updated,
//!    and what is already summed into the output is scaled down to match.
//!    The scores are in powers of two, `exp2(s · log₂e / √d)`, which is the
//!    same softmax for one multiply less.
//! 3. `O += P · V`, again on the matrix unit, into f32.
//!
//! At the end each row is divided by its sum and rounded once. Only the last
//! step, when it is partial, checks rows and hides the keys past the end.
//!
//! # Fragments, and what makes them fast
//!
//! The tensor API's cooperative tensors belong to a SIMD group, and a
//! product can take one as its left or right operand only at that scope.
//! Everything here is a `16 × 16` fragment in the layout `matmul2d` gives a
//! SIMD group for a `16 × 32 × 16` product: eight numbers a lane, from rows
//! `r` and `r + 8`, columns `c` to `c + 3`. MLX relies on the same layout,
//! and the tests hold this kernel to a written-out attention, so a different
//! layout would fail them rather than pass quietly. A row's four columns of
//! four sit in four lanes that differ in their bits 0 and 3, so a row's
//! maximum and sum are two `simd_shuffle_xor`s.
//!
//! Four things decide its speed, all measured at stage 2's video
//! self-attention:
//! - **Every index into an array of fragments must be a constant.** One
//!   index the compiler cannot resolve, even in a loop that only zeroes the
//!   array, puts the whole array in memory. A probe of the bare product ran
//!   at 32 TFLOP/s with its accumulators in registers and 5–9 with them in
//!   an array under `#pragma unroll`, which does not promise to unroll a loop
//!   around the tensor op. So every such loop is expanded by a template
//!   (`EACH`), and the first version, written with ordinary loops, ran at
//!   3.3 TFLOP/s.
//! - **The four SIMD groups stay in step.** They all read the same keys and
//!   values, and a threadgroup barrier once a step keeps them close enough
//!   to share those reads in the core's cache: 9.6 → 16 TFLOP/s.
//! - **The threadgroup is dispatched as 32 × 4 threads, not 128 × 1.** It is
//!   the same 128 threads in the same four SIMD groups, and nothing in the
//!   kernel reads the shape, yet 128 × 1 held this kernel at about 16 TFLOP/s
//!   and took MLX's own, compiled here from its source, from 21.4 to 15.6.
//! - **Fragments are loaded element by element**, as MLX loads them, rather
//!   than as a vector of four copied out: with the dispatch above, 16.2 →
//!   18.6 TFLOP/s.
//!
//! Together these bring it level with MLX's own M5 attention, raced in one
//! process on the same inputs. Staging each step's keys and values once in
//! threadgroup memory, instead of each SIMD group loading its own, was
//! slower: 11 TFLOP/s copying then computing, and 7 with the next step
//! prefetched into registers, which crowded out the accumulators.
//!
//! # What it reads
//!
//! `q` is `[lq, heads · d]`, and `k` and `v` are `[lk, heads · d]`: the
//! projections' own layout, so the heads are never split into copies of
//! their own, and the answer comes back in the same layout. `[heads, l, d]`
//! is read too. Rows may be strided, as a narrowed tensor's are. `d` is 64
//! or 128, which covers every attention in LTX-2.5.
//!
//! It runs wherever `mpp` does, and `KVAD_GPU_MPP_ATTENTION=0` leaves
//! attention alone to candle, for measuring what this is worth.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;

const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
    int lq, lk;
    // Row strides, in elements.
    int ldq, ldk, ldv, ldo;
    // Head strides, in elements.
    int hq, hk, hv;
    // 1/√d × log₂e: the softmax in powers of two.
    float scale;
};

// `f(0)` to `f(N - 1)`, each index a compile-time constant. Every array of
// fragments below is indexed only through this: see the module's notes.
template <int I, int N>
struct each {
    template <typename F>
    static __attribute__((always_inline)) void run(thread const F &f) {
        f(integral_constant<int, I>());
        each<I + 1, N>::run(f);
    }
};
template <int N>
struct each<N, N> {
    template <typename F>
    static __attribute__((always_inline)) void run(thread const F &) {}
};
// `EACH(N, i, body)`: `body` for `i` from 0 to N - 1, `i` a constant.
#define EACH(N, I, ...) each<0, N>::run([&](auto I##_) { constexpr int I = decltype(I##_)::value; __VA_ARGS__ })

// A 16 × 16 fragment: eight numbers a lane, rows r and r + 8, columns c to
// c + 3.
template <typename T> using frag = vec<T, 8>;

// This lane's (c, r).
inline short2 place(ushort lane) {
    const short g = lane >> 2;
    return short2(((g & 2) | (lane & 1)) * 4, (g & 4) | ((lane >> 1) & 3));
}

// A fragment from row-major memory, element by element: reading each row's
// four as one vector and copying them out was 13–15% slower. With EDGE,
// only `rows` rows exist and the rest read as zeros; without, all 16 do and
// nothing is checked.
template <bool EDGE = true, typename T>
inline frag<T> load(device const T *p, int ld, short2 at, int rows) {
    frag<T> f;
    p += at.y * ld + at.x;
    EACH(2, i,
        if (!EDGE || at.y + i * 8 < rows) {
            EACH(4, c, f[i * 4 + c] = p[i * 8 * ld + c];);
        } else {
            EACH(4, c, f[i * 4 + c] = T(0););
        }
    );
    return f;
}

// `c0 | c1 += a · (b0 | b1)`: one SIMD group's 16 × 32 × 16 product on the
// matrix unit. `b0` and `b1` are the right operand's two 16-column halves;
// with `TR`, they are the two 16-row halves of what is stored, and the
// product takes its transpose.
template <bool TR, typename TA, typename TB>
inline void mma(thread frag<float> &c0, thread frag<float> &c1, thread const frag<TA> &a,
                thread const frag<TB> &b0, thread const frag<TB> &b1) {
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, TR, true,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<TA, TB, float>();
    auto cb = op.template get_right_input_cooperative_tensor<TA, TB, float>();
    auto cc = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>,
                                                             remove_addrspace_t<decltype(cb)>, float>();
    EACH(8, i, ca[i] = a[i]; cb[i] = b0[i]; cb[8 + i] = b1[i]; cc[i] = c0[i]; cc[8 + i] = c1[i];);
    op.run(ca, cb, cc);
    EACH(8, i, c0[i] = cc[i]; c1[i] = cc[8 + i];);
}

// Keys a step.
constant constexpr int BK = 32;

// T is what is read and written, D the head's width.
template <typename T, int D>
[[kernel, max_total_threads_per_threadgroup(128)]] void attention(
        device const T *q [[buffer(0)]],
        device const T *k [[buffer(1)]],
        device const T *v [[buffer(2)]],
        device T *o [[buffer(3)]],
        constant Params &p [[buffer(4)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int TD = D / 16;
    constexpr int TK = BK / 16;
    const int q0 = int(tg.x) * 64 + sg * 16;
    // A SIMD group past the last query still walks the keys, on zeros,
    // since the barrier below needs all four; it only writes nothing.
    const int rows = p.lq - q0;
    const int h = int(tg.y);
    const short2 at = place(lane);
    q += q0 * p.ldq + h * p.hq;
    k += h * p.hk;
    v += h * p.hv;
    o += q0 * p.ldo + h * D;

    frag<T> qf[TD];
    frag<float> acc[TD];
    EACH(TD, d, qf[d] = load(q + d * 16, p.ldq, at, rows); acc[d] = 0;);
    // Each lane's two rows: the running maximum and sum. Finite, not
    // infinite, since the compiler assumes fast maths.
    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0, 0};

    // One step of BK keys. EDGE is the last step when it is partial: only
    // there are rows checked and keys hidden.
    int kb = 0;
    auto step = [&](auto edge) {
        constexpr bool EDGE = decltype(edge)::value;
        const int keys = p.lk - kb;
        device const T *kt = k + kb * p.ldk;
        device const T *vt = v + kb * p.ldv;

        // S = Q · Kᵀ, two fragments of keys at a time.
        frag<float> s[TK];
        EACH(TK, j, s[j] = 0;);
        EACH(TK / 2, jj,
            constexpr int j = 2 * jj;
            EACH(TD, d,
                const frag<T> k0 = load<EDGE>(kt + j * 16 * p.ldk + d * 16, p.ldk, at, keys - j * 16);
                const frag<T> k1 = load<EDGE>(kt + (j + 1) * 16 * p.ldk + d * 16, p.ldk, at, keys - (j + 1) * 16);
                mma<true>(s[j], s[j + 1], qf[d], k0, k1);
            );
        );

        // Scaled, with the keys past the end hidden, and each row's new
        // maximum.
        float mx[2] = {m[0], m[1]};
        EACH(TK, j, EACH(2, i, EACH(4, c,
            const float x = !EDGE || j * 16 + at.x + c < keys ? s[j][i * 4 + c] * p.scale : -FLT_MAX;
            s[j][i * 4 + c] = x;
            mx[i] = max(mx[i], x);
        );););
        // The factor that brings what is summed so far down to the new
        // maximum; the probabilities, in place of the scores; the new sums.
        float f[2], sum[2] = {0, 0};
        EACH(2, i,
            mx[i] = max(mx[i], simd_shuffle_xor(mx[i], 1));
            mx[i] = max(mx[i], simd_shuffle_xor(mx[i], 8));
            f[i] = fast::exp2(m[i] - mx[i]);
            m[i] = mx[i];
        );
        EACH(TK, j, EACH(2, i, EACH(4, c,
            const float e = fast::exp2(s[j][i * 4 + c] - m[i]);
            s[j][i * 4 + c] = e;
            sum[i] += e;
        );););
        EACH(2, i,
            sum[i] += simd_shuffle_xor(sum[i], 1);
            sum[i] += simd_shuffle_xor(sum[i], 8);
            l[i] = l[i] * f[i] + sum[i];
        );
        EACH(TD, d, EACH(2, i, EACH(4, c, acc[d][i * 4 + c] *= f[i];);););

        // O += P · V, two fragments of the head's width at a time. Halfway,
        // the SIMD groups wait for one another, so that they go on reading
        // the same keys and values from the core's cache.
        EACH(TD / 2, dd,
            constexpr int d = 2 * dd;
            if constexpr (d == TD / 2) {
                threadgroup_barrier(mem_flags::mem_none);
            }
            EACH(TK, j,
                const frag<T> v0 = load<EDGE>(vt + j * 16 * p.ldv + d * 16, p.ldv, at, keys - j * 16);
                const frag<T> v1 = load<EDGE>(vt + j * 16 * p.ldv + (d + 1) * 16, p.ldv, at, keys - j * 16);
                mma<false>(acc[d], acc[d + 1], s[j], v0, v1);
            );
        );
    };
    for (; kb + BK <= p.lk; kb += BK) {
        step(false_type());
    }
    if (kb < p.lk) {
        step(true_type());
    }

    EACH(2, i,
        const short r = at.y + i * 8;
        if (r < rows) {
            const float inv = 1.0f / l[i];
            EACH(TD, d,
                vec<T, 4> y;
                EACH(4, c, y[c] = T(acc[d][i * 4 + c] * inv););
                *(device vec<T, 4> *)(o + r * p.ldo + d * 16 + at.x) = y;
            );
        }
    );
}

#define ATTN(T, TN, D) \
    template [[host_name("attention_" #TN "_d" #D)]] [[kernel]] decltype(attention<T, D>) attention<T, D>;
ATTN(half, f16, 64)
ATTN(half, f16, 128)
ATTN(bfloat, bf16, 64)
ATTN(bfloat, bf16, 128)
"#;

/// Queries a threadgroup: four SIMD groups of 16.
const BQ: usize = 64;

#[repr(C)]
struct Params {
    lq: i32,
    lk: i32,
    ldq: i32,
    ldk: i32,
    ldv: i32,
    ldo: i32,
    hq: i32,
    hk: i32,
    hv: i32,
    scale: f32,
}

/// Where a head's rows are: `(rows, row stride, head stride)`, from
/// `[rows, heads · d]` or `[heads, rows, d]`. `None` if its rows are not
/// contiguous, or any stride or its start is not a multiple of four
/// elements, which the kernel's vector loads need.
fn rows_of(l: &Layout, heads: usize, d: usize) -> Option<(usize, usize, usize)> {
    let (dims, st) = (l.dims(), l.stride());
    let found = match *dims {
        [n, w] if w == heads * d && st[1] == 1 => (n, st[0], d),
        [h, n, w] if h == heads && w == d && st[2] == 1 => (n, st[1], st[0]),
        _ => return None,
    };
    (found.1 % 4 == 0 && found.2 % 4 == 0 && l.start_offset() % 4 == 0).then_some(found)
}

/// `softmax(q · kᵀ / √d) · v` for each of `heads` heads, or `None` where
/// the caller should use candle's attention.
///
/// `q` is `[lq, heads · d]`, `k` and `v` `[lk, heads · d]`, or any of the
/// three `[heads, l, d]`; the answer is `[lq, heads · d]` in `q`'s dtype.
/// `None` means one of these:
/// - the device has no matrix units, or `KVAD_GPU_MPP=0` or
///   `KVAD_GPU_MPP_ATTENTION=0`;
/// - the dtype is not f16 or bf16, or the three differ;
/// - `d` is not 64 or 128;
/// - a row is not contiguous, or a stride or start is not a multiple of four
///   elements;
/// - there are no keys.
pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> candle_core::Result<Option<Tensor>> {
    let d = match q.rank() {
        3 => q.dim(2)?,
        _ => q.dim(q.rank() - 1)? / heads.max(1),
    };
    let (qr, kr, vr) = (rows_of(q.layout(), heads, d), rows_of(k.layout(), heads, d), rows_of(v.layout(), heads, d));
    if kernels(q.device()).is_none()
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || k.dtype() != q.dtype()
        || v.dtype() != q.dtype()
        || !matches!(d, 64 | 128)
        || qr.is_none()
        || kr.is_none()
        || kr.map(|r| r.0) != vr.map(|r| r.0)
        || kr.is_some_and(|r| r.0 == 0)
    {
        return Ok(None);
    }
    Ok(Some(q.apply_op3_no_bwd(k, v, &Attention { heads, d })?))
}

fn kernels(device: &candle_core::Device) -> Option<&'static crate::fused::metal::Kernels> {
    if matches!(std::env::var("KVAD_GPU_MPP_ATTENTION").as_deref(), Ok("0") | Ok("false")) {
        return None;
    }
    crate::fused::metal::tensor_library(device, "attention", SOURCE)
}

struct Attention {
    heads: usize,
    d: usize,
}

impl CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "mpp_attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_attention runs on Metal only")
    }

    fn metal_fwd(&self, q: &MetalStorage, lq: &Layout, k: &MetalStorage, lk: &Layout, v: &MetalStorage, lv: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        use crate::fused::metal::{buffer, output};
        let dt = q.dtype();
        let dev = q.device();
        let Some(lib) = kernels(&candle_core::Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_attention: this device cannot run it");
        };
        let (h, d) = (self.heads, self.d);
        let geometry = |l: &Layout| rows_of(l, h, d).ok_or_else(|| candle_core::Error::Msg(format!("mpp_attention: {l:?}")));
        let ((rows, ldq, hq), (keys, ldk, hk), (_, ldv, hv)) = (geometry(lq)?, geometry(lk)?, geometry(lv)?);
        let width = h * d;
        let tn = if dt == DType::F16 { "f16" } else { "bf16" };
        let pipe = lib.pipe(&format!("attention_{tn}_d{d}"))?;
        let out = output(dev, rows * width * dt.size_in_bytes())?;
        let params = Params {
            lq: rows as i32,
            lk: keys as i32,
            ldq: ldq as i32,
            ldk: ldk as i32,
            ldv: ldv as i32,
            ldo: width as i32,
            hq: hq as i32,
            hk: hk as i32,
            hv: hv as i32,
            scale: std::f32::consts::LOG2_E / (d as f32).sqrt(),
        };
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_attention");
        enc.set_compute_pipeline_state(&pipe);
        for (i, (s, l)) in [(q, lq), (k, lk), (v, lv)].into_iter().enumerate() {
            let (b, at) = buffer(s, l);
            enc.set_input_buffer(i, Some(&b), at);
        }
        enc.set_output_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &params);
        enc.dispatch_thread_groups(
            MTLSize { width: rows.div_ceil(BQ), height: h, depth: 1 },
            // 32 × 4, a SIMD group to a row, not 128 × 1: the same threads
            // in the same SIMD groups, and yet see the module's notes.
            MTLSize { width: 32, height: BQ / 16, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), rows * width, dt), Shape::from((rows, width))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The device, where it has matrix units. There the kernels must
    /// build: a source that does not compile would otherwise skip every
    /// test here and pass.
    fn gpu() -> Option<Device> {
        let dev = Device::new_metal(0).ok()?;
        if !crate::mpp::available(&dev) {
            return None;
        }
        assert!(kernels(&dev).is_some(), "the attention kernels did not build");
        Some(dev)
    }

    /// Attention written out in f32, head by head, from the same numbers.
    fn reference(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Tensor {
        let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
        let (lq, width) = q.dims2().unwrap();
        let lk = k.dim(0).unwrap();
        let d = width / heads;
        let split = |t: &Tensor, l: usize| f(t).reshape((1, l, heads, d)).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
        let o = crate::image::nn::written_out(&split(q, lq), &split(k, lk), &split(v, lk), 1.0 / (d as f64).sqrt()).unwrap();
        o.transpose(1, 2).unwrap().contiguous().unwrap().reshape((lq, width)).unwrap()
    }

    /// Signal to error, in dB, of `got` against `want`.
    fn db(got: &Tensor, want: &Tensor) -> f32 {
        let got = got.to_dtype(DType::F32).unwrap();
        // A NaN never wins a comparison, so check that everything was
        // written before comparing anything.
        assert!(got.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "a NaN or an infinity: not all written");
        let err = (&got - want).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let sig = want.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        10.0 * (sig / err.max(1e-30)).log10()
    }

    /// The kernel against attention written out in f32, and against
    /// candle's own attention in the same dtype: both widths, whole steps
    /// and every kind of ragged edge, including one query and one key.
    ///
    /// Measured when written: 55–58 dB in bf16 and 70–74 dB in f16, where
    /// candle's attention gives 44–49 and 61–67 on the same inputs.
    #[test]
    fn agrees_with_attention_written_out() {
        let Some(dev) = gpu() else {
            eprintln!("no matrix units for matmul2d; nothing to test");
            return;
        };
        // (queries, keys, heads, d)
        let shapes = [(128, 128, 2, 128), (77, 50, 3, 64), (300, 1024, 2, 128), (126, 126, 4, 64), (1, 33, 1, 64), (65, 1, 2, 128)];
        for dt in [DType::BF16, DType::F16] {
            for (lq, lk, heads, d) in shapes {
                let w = heads * d;
                // Spread wide enough that the softmax is peaked in places, as
                // a normed query's is, and not flat.
                let rand = |l: usize| (Tensor::randn(0f32, 1.0, (l, w), &dev).unwrap() * 2.0).unwrap().to_dtype(dt).unwrap();
                let (q, k, v) = (rand(lq), rand(lk), rand(lk));
                let want = reference(&q, &k, &v, heads);
                let theirs = {
                    let r = |t: &Tensor, l: usize| t.reshape((1, l, w)).unwrap();
                    crate::image::nn::attention(&r(&q, lq), &r(&k, lk), &r(&v, lk), heads).unwrap().squeeze(0).unwrap()
                };
                let before = crate::fused::tests_ran();
                let got = attention(&q, &k, &v, heads).unwrap().expect("the kernel declined");
                assert_eq!(crate::fused::tests_ran(), before + 1, "the kernel did not run");
                assert_eq!(got.dims(), &[lq, w]);
                assert_eq!(got.dtype(), dt);
                let (ours, floor) = (db(&got, &want), db(&theirs, &want));
                // Within 3 dB of candle's is the same rounding, not a bug.
                assert!(ours > floor - 3.0, "{dt:?} [{lq}, {lk}] × {heads} × {d}: {ours:.1} dB, candle {floor:.1}");
            }
        }
    }

    /// Rows need not be packed, and heads may come first: a narrowed slice
    /// of a wider tensor, and `[heads, l, d]`, give what the packed rows do.
    #[test]
    fn reads_strided_rows_and_head_major() {
        let Some(dev) = gpu() else { return };
        let (l, heads, d) = (70, 2, 64);
        let wide = Tensor::randn(0f32, 1.0, (l, 3 * heads * d), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        let (q, k, v) = (wide.narrow(1, heads * d, heads * d).unwrap(), wide.narrow(1, 0, heads * d).unwrap(), wide.narrow(1, 2 * heads * d, heads * d).unwrap());
        let host = |t: Tensor| t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
        let got = host(attention(&q, &k, &v, heads).unwrap().unwrap());
        let packed = |t: &Tensor| t.contiguous().unwrap();
        assert_eq!(got, host(attention(&packed(&q), &packed(&k), &packed(&v), heads).unwrap().unwrap()));
        let first = |t: &Tensor| t.reshape((l, heads, d)).unwrap().transpose(0, 1).unwrap().contiguous().unwrap();
        assert_eq!(got, host(attention(&first(&q), &first(&k), &first(&v), heads).unwrap().unwrap()));
    }

    /// What it leaves to candle.
    #[test]
    fn declines_what_it_cannot_read() {
        let Some(dev) = gpu() else { return };
        let t = |l: usize, w: usize, dt: DType| Tensor::zeros((l, w), dt, &dev).unwrap();
        let bf = |l: usize| t(l, 256, DType::BF16);
        assert!(attention(&bf(8), &bf(8), &bf(8), 2).unwrap().is_some());
        // f32, mixed dtypes, a head width of 32, a transposed input, no keys.
        assert!(attention(&t(8, 256, DType::F32), &t(8, 256, DType::F32), &t(8, 256, DType::F32), 2).unwrap().is_none());
        assert!(attention(&bf(8), &t(8, 256, DType::F16), &bf(8), 2).unwrap().is_none());
        assert!(attention(&bf(8), &bf(8), &bf(8), 8).unwrap().is_none());
        let tr = t(256, 8, DType::BF16).t().unwrap();
        assert!(attention(&tr, &bf(8), &bf(8), 2).unwrap().is_none());
        assert!(attention(&bf(8), &bf(0), &bf(0), 2).unwrap().is_none());
        let cpu = Tensor::zeros((8, 256), DType::BF16, &Device::Cpu).unwrap();
        assert!(attention(&cpu, &cpu, &cpu, 2).unwrap().is_none());
    }

    /// Not a test, a measurement: the kernel against candle's attention at
    /// LTX-2.5's shapes, taking turns round by round, candle's including its
    /// copies to and from heads-first. Run it with
    ///
    ///     cargo test --release -p kvad-gpu attention_race -- --ignored --nocapture
    ///
    /// `RACE_ONLY=text` runs only the shapes whose names contain `text`.
    #[test]
    #[ignore]
    fn attention_race() {
        let dev = gpu().expect("no matrix units");
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        // (what, queries, keys, heads, d)
        let shapes = [
            ("stage 2 video self", 24576, 24576, 32, 128),
            ("stage 1 video self", 6144, 6144, 32, 128),
            ("stage 2 video text", 24576, 1024, 32, 128),
            ("stage 2 audio to video", 24576, 126, 32, 64),
            ("stage 2 video to audio", 126, 24576, 32, 64),
            ("audio self", 126, 126, 32, 64),
        ];
        let only = std::env::var("RACE_ONLY").unwrap_or_default();
        for (what, lq, lk, heads, d) in shapes.into_iter().filter(|s| s.0.contains(only.as_str())) {
            let w = heads * d;
            let rand = |l: usize| Tensor::randn(0f32, 1.0, (l, w), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let (q, k, v) = (rand(lq), rand(lk), rand(lk));
            let r = |t: &Tensor, l: usize| t.reshape((1, l, w)).unwrap();
            let (q3, k3, v3) = (r(&q, lq), r(&k, lk), r(&v, lk));
            let flops = 4.0 * (lq * lk * w) as f64;
            let reps = ((2e12 / flops) as usize).clamp(1, 50);
            let time = |f: &dyn Fn()| {
                dev.synchronize().unwrap();
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    f();
                }
                dev.synchronize().unwrap();
                t.elapsed().as_secs_f64() / reps as f64
            };
            let candle = || drop(crate::image::nn::attention(&q3, &k3, &v3, heads).unwrap());
            let ours = || drop(attention(&q, &k, &v, heads).unwrap().unwrap());
            time(&candle);
            time(&ours);
            let (mut theirs, mut mine) = (vec![], vec![]);
            for _ in 0..5 {
                theirs.push(time(&candle));
                mine.push(time(&ours));
            }
            let (b, s) = (median(theirs), median(mine));
            println!("{what:<24} candle {:9.3} ms {:5.1} TFLOP/s | ours {:9.3} ms {:5.1} TFLOP/s | {:5.2}x",
                     b * 1e3, flops / b / 1e12, s * 1e3, flops / s / 1e12, b / s);
        }
    }
}
