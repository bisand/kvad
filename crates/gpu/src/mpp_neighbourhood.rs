//! Neighbourhood attention on the M5 GPU's neural accelerators, for
//! LTX-2.5's diffusion decoder (`video::ltx_diffvae`).
//!
//! Every token of a `T × H × W` grid attends to the `kt × kh × kw` window
//! around it, shifted inward at the edges (NATTEN's `na3d`). It is flash
//! attention as [`crate::mpp_attention`] does it, fragments, online softmax
//! and all; what differs is which keys a query walks.
//!
//! # The key box
//!
//! A SIMD group takes 16 queries: a patch of 4 rows × 4 columns in one
//! frame. They share the frame, so they share the time window; their
//! windows differ only across rows and columns, and together they cover a
//! box of `kt` frames × at most `kh + 3` rows × at most `kw + 3` columns. With
//! `kw ≤ 13` the columns fit in 16, so each (frame, row) of the box is one
//! run of 16 consecutive tokens: one fragment of keys, read as `mpp_attention`
//! reads its keys, at a row stride. The group walks those runs two at a time
//! and hides, for each query, the keys outside its own window.
//!
//! At stage 5's 11 × 11 × 11 that is 11 × 14 runs of 16, 2464 keys for the
//! 1331 each query attends to: 54% of the products are used. A patch of
//! 16 × 1 would use 34% (26 columns take two fragments), and 2 × 8 45%.
//!
//! Four SIMD groups sit side by side along the columns, a threadgroup 4 rows
//! × 16 columns, so they share the rows of their boxes, walk the same number
//! of runs, and read overlapping keys; a barrier once a step keeps them close
//! enough to share them in the core's cache, as in `mpp_attention`.
//!
//! # What it reads
//!
//! `q`, `k` and `v` are `[T·H·W, heads · d]`, tokens row by row, rows
//! possibly strided (a narrowed slice of a fused projection's answer); the
//! answer is `[T·H·W, heads · d]`. `d` is 64 or 128. Offsets are 64-bit: a
//! 1536 × 1024 clip's stage 5 is 12M tokens, and its keys' offsets pass 2³¹.

use crate::mpp::fragments;
use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;

const SOURCE: &str = concat!(
    r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"#,
    fragments!(),
    r#"
struct Params {
    // The grid, and the window.
    int T, H, W;
    int kt, kh, kw;
    // Row strides, in elements.
    int ldq, ldk, ldv;
    int heads;
    // The scores' scale × log₂e: the softmax in powers of two.
    float scale;
    // The joint kernel's other stream's row strides.
    int ldpk, ldpv;
};

// A SIMD group's queries: 4 rows × 4 columns of one frame. Its fragment row
// r is the query (r / 4, r % 4), so a lane's rows r and r + 8 are one column
// and two rows apart.
constant constexpr int QH = 4;
constant constexpr int QW = 4;
// SIMD groups side by side along the columns.
constant constexpr int NSG = 4;

// Where a window of k starts along an axis of n, for index i.
inline int window(int i, int n, int k) {
    return clamp(i - k / 2, 0, n - k);
}

// The body of both kernels. With JOINT, one side of a joint attention:
// each query's window, cut at the edges rather than shifted, and then the
// same rows and columns on each of the other stream's frames `slots` names
// for its frame (two, −1 for none), `pk` and `pv`, all in one softmax.
template <typename T, int D, bool JOINT>
inline void na(device const T *q, device const T *k, device const T *v, device T *o, constant Params &p,
               device const T *pk, device const T *pv, device const int *slots,
               uint3 tg, ushort sg, ushort lane) {
    constexpr int TD = D / 16;
    const short2 at = place(lane);
    const int head = int(tg.z) % p.heads;
    const int t = int(tg.z) / p.heads;
    const int h0 = int(tg.y) * QH;
    const int w0 = (int(tg.x) * NSG + sg) * QW;
    q += head * D;
    k += head * D;
    v += head * D;
    o += head * D;
    if (JOINT) {
        pk += head * D;
        pv += head * D;
    }

    // This lane's two queries, whether they exist, and their windows'
    // starts. A query past the edge takes the last one's window, walks
    // with the others on zeros, and is never written. A joint window
    // starts half a window before, and may hang over the edge.
    auto start = [&](int i, int n, int kk) { return JOINT ? i - kk / 2 : window(i, n, kk); };
    const int qw = w0 + at.y % 4;
    const int qh[2] = {h0 + at.y / 4, h0 + at.y / 4 + 2};
    bool ok[2];
    long row[2];
    int hs[2];
    const int ws = start(min(qw, p.W - 1), p.W, p.kw);
    EACH(2, i,
        ok[i] = qh[i] < p.H && qw < p.W;
        row[i] = (long(t) * p.H + qh[i]) * p.W + qw;
        hs[i] = start(min(qh[i], p.H - 1), p.H, p.kh);
    );
    frag<T> qf[TD];
    frag<float> acc[TD];
    EACH(TD, d,
        EACH(2, i, EACH(4, c, qf[d][i * 4 + c] = ok[i] ? q[row[i] * p.ldq + d * 16 + at.x + c] : T(0);););
        acc[d] = 0;
    );
    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0, 0};

    // The box: the frames of the time window; the rows from the first
    // query row's window to the last's; 16 columns from this group's first
    // window, pulled back to end at the edge, as far as the grid goes. A
    // joint one's, cut to the grid.
    int ts, frames, hlo, rows, wlo;
    if (JOINT) {
        const int first = t - p.kt / 2;
        ts = max(first, 0);
        frames = min(first + p.kt, p.T) - ts;
        hlo = max(min(h0, p.H - 1) - p.kh / 2, 0);
        rows = min(min(h0 + QH - 1, p.H - 1) - p.kh / 2 + p.kh, p.H) - hlo;
        wlo = max(min(min(w0, p.W - 1) - p.kw / 2, p.W - 16), 0);
    } else {
        ts = window(t, p.T, p.kt);
        frames = p.kt;
        hlo = window(min(h0, p.H - 1), p.H, p.kh);
        rows = window(min(h0 + QH - 1, p.H - 1), p.H, p.kh) + p.kh - hlo;
        wlo = max(min(window(min(w0, p.W - 1), p.W, p.kw), p.W - 16), 0);
    }
    const int cols = min(16, p.W - wlo);
    // The planes' runs follow the window's: the same rows of each, the
    // present ones first.
    const int vruns = frames * rows;
    int p0 = -1, p1 = -1;
    if (JOINT) {
        p0 = slots[2 * t];
        p1 = slots[2 * t + 1];
        if (p0 < 0) {
            p0 = p1;
            p1 = -1;
        }
    }
    const int runs = vruns + (int(p0 >= 0) + int(p1 >= 0)) * rows;

    // One step: runs f and f + 1, or with EDGE only f, the last of an odd
    // number; with PLANE, of the planes' runs, `f` counting from theirs.
    int f = 0;
    auto step = [&](auto edge, auto plane) {
        constexpr bool EDGE = decltype(edge)::value;
        constexpr bool PLANE = decltype(plane)::value;
        const int ldk = PLANE ? p.ldpk : p.ldk;
        const int ldv = PLANE ? p.ldpv : p.ldv;
        int b[2];
        device const T *kr[2];
        device const T *vr[2];
        EACH(2, j,
            if (PLANE) {
                const int r = min(f + j, runs - vruns - 1);
                const int slot = r / rows;
                b[j] = r - slot * rows;
                const long first = (long(slot == 0 ? p0 : p1) * p.H + hlo + b[j]) * p.W + wlo;
                kr[j] = pk + first * ldk;
                vr[j] = pv + first * ldv;
            } else {
                const int r = min(f + j, vruns - 1);
                const int a = r / rows;
                b[j] = r - a * rows;
                const long first = (long(ts + a) * p.H + hlo + b[j]) * p.W + wlo;
                kr[j] = k + first * ldk;
                vr[j] = v + first * ldv;
            }
        );

        frag<float> s[2];
        s[0] = 0;
        s[1] = 0;
        EACH(TD, d,
            const frag<T> k0 = load<true>(kr[0] + d * 16, ldk, at, cols);
            const frag<T> k1 = EDGE ? frag<T>(0) : load<true>(kr[1] + d * 16, ldk, at, cols);
            mma<true>(s[0], s[1], qf[d], k0, k1);
        );

        // Scaled, each query's keys outside its window hidden, and each
        // row's new maximum.
        float mx[2] = {m[0], m[1]};
        EACH(2, j, EACH(2, i,
            const bool in_rows = (!EDGE || j == 0) && uint(hlo + b[j] - hs[i]) < uint(p.kh);
            EACH(4, c,
                const bool in = in_rows && uint(wlo + at.x + c - ws) < uint(p.kw) && (!JOINT || wlo + at.x + c < p.W);
                const float x = in ? s[j][i * 4 + c] * p.scale : -FLT_MAX;
                s[j][i * 4 + c] = x;
                mx[i] = max(mx[i], x);
            );
        ););
        float g[2], sum[2] = {0, 0};
        EACH(2, i,
            mx[i] = max(mx[i], simd_shuffle_xor(mx[i], 1));
            mx[i] = max(mx[i], simd_shuffle_xor(mx[i], 8));
            g[i] = fast::exp2(m[i] - mx[i]);
            m[i] = mx[i];
        );
        // A row whose keys so far were all hidden has summed ones at a
        // maximum of −FLT_MAX; its first real key's maximum scales them to
        // nothing, as every query has keys in its window.
        EACH(2, j, EACH(2, i, EACH(4, c,
            const float e = fast::exp2(s[j][i * 4 + c] - m[i]);
            s[j][i * 4 + c] = e;
            sum[i] += e;
        );););
        EACH(2, i,
            sum[i] += simd_shuffle_xor(sum[i], 1);
            sum[i] += simd_shuffle_xor(sum[i], 8);
            l[i] = l[i] * g[i] + sum[i];
        );
        EACH(TD, d, EACH(2, i, EACH(4, c, acc[d][i * 4 + c] *= g[i];);););

        // O += P · V, halfway waiting for the other SIMD groups.
        EACH(TD / 2, dd,
            constexpr int d = 2 * dd;
            if constexpr (d == TD / 2) {
                threadgroup_barrier(mem_flags::mem_none);
            }
            EACH(2, j,
                if (!EDGE || j == 0) {
                    const frag<T> v0 = load<true>(vr[j] + d * 16, ldv, at, cols);
                    const frag<T> v1 = load<true>(vr[j] + (d + 1) * 16, ldv, at, cols);
                    mma<false>(acc[d], acc[d + 1], s[j], v0, v1);
                }
            );
        );
    };
    // The window's runs, and then the planes'.
    for (; f + 2 <= vruns; f += 2) {
        step(false_type(), false_type());
    }
    if (f < vruns) {
        step(true_type(), false_type());
    }
    if (JOINT) {
        const int pruns = runs - vruns;
        for (f = 0; f + 2 <= pruns; f += 2) {
            step(false_type(), true_type());
        }
        if (f < pruns) {
            step(true_type(), true_type());
        }
    }

    EACH(2, i,
        if (ok[i]) {
            const float inv = 1.0f / l[i];
            device T *out = o + row[i] * (p.heads * D);
            EACH(TD, d,
                vec<T, 4> y;
                EACH(4, c, y[c] = T(acc[d][i * 4 + c] * inv););
                *(device vec<T, 4> *)(out + d * 16 + at.x) = y;
            );
        }
    );
}

template <typename T, int D>
[[kernel, max_total_threads_per_threadgroup(128)]] void neighbourhood(
        device const T *q [[buffer(0)]],
        device const T *k [[buffer(1)]],
        device const T *v [[buffer(2)]],
        device T *o [[buffer(3)]],
        constant Params &p [[buffer(4)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    na<T, D, false>(q, k, v, o, p, nullptr, nullptr, nullptr, tg, sg, lane);
}

template <typename T, int D>
[[kernel, max_total_threads_per_threadgroup(128)]] void neighbourhood_joint(
        device const T *q [[buffer(0)]],
        device const T *k [[buffer(1)]],
        device const T *v [[buffer(2)]],
        device T *o [[buffer(3)]],
        constant Params &p [[buffer(4)]],
        device const int *slots [[buffer(5)]],
        device const T *pk [[buffer(6)]],
        device const T *pv [[buffer(7)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    na<T, D, true>(q, k, v, o, p, pk, pv, slots, tg, sg, lane);
}

#define NA(T, TN, D) \
    template [[host_name("neighbourhood_" #TN "_d" #D)]] [[kernel]] decltype(neighbourhood<T, D>) neighbourhood<T, D>; \
    template [[host_name("neighbourhood_joint_" #TN "_d" #D)]] [[kernel]] decltype(neighbourhood_joint<T, D>) neighbourhood_joint<T, D>;
NA(half, f16, 64)
NA(half, f16, 128)
NA(bfloat, bf16, 64)
NA(bfloat, bf16, 128)
"#
);

/// A threadgroup's queries: 4 rows × 16 columns.
const TG_ROWS: usize = 4;
const TG_COLS: usize = 16;

#[repr(C)]
struct Params {
    t: i32,
    h: i32,
    w: i32,
    kt: i32,
    kh: i32,
    kw: i32,
    ldq: i32,
    ldk: i32,
    ldv: i32,
    heads: i32,
    scale: f32,
    ldpk: i32,
    ldpv: i32,
}

/// A row stride from `[rows, heads · d]` whose rows are contiguous.
fn stride_of(l: &Layout, width: usize) -> Option<usize> {
    match (l.dims(), l.stride()) {
        ([_, w], [ld, 1]) if *w == width => Some(*ld),
        _ => None,
    }
}

/// `softmax(scale · q·kᵀ)·v` within each query's window, or `None` where the
/// caller should use its own: q, k, v `[t·h·w, heads · d]` on `grid`
/// `(t, h, w)`, the window `kernel`, the answer `[t·h·w, heads · d]` in q's
/// dtype. `None` means one of these:
/// - the device has no matrix units, or `KVAD_GPU_MPP=0` or
///   `KVAD_GPU_MPP_NEIGHBOURHOOD=0`;
/// - the dtype is not f16 or bf16, or the three differ;
/// - `d` is not 64 or 128;
/// - a window is wider than 13 columns, or larger than the grid;
/// - a row is not contiguous, or the shapes disagree.
pub(crate) fn neighbourhood(q: &Tensor, k: &Tensor, v: &Tensor, grid: [usize; 3], kernel: [usize; 3], heads: usize, scale: f32)
 -> candle_core::Result<Option<Tensor>> {
    let (n, width) = q.dims2()?;
    let d = width / heads.max(1);
    let fits = (0..3).all(|a| kernel[a] >= 1 && kernel[a] <= grid[a]);
    if kernels(q.device()).is_none()
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || k.dtype() != q.dtype()
        || v.dtype() != q.dtype()
        || !matches!(d, 64 | 128)
        || d * heads != width
        || grid.iter().product::<usize>() != n
        || !fits
        || kernel[2] > 13
        || k.dims() != q.dims()
        || v.dims() != q.dims()
        || [q, k, v].iter().any(|t| stride_of(t.layout(), width).is_none())
    {
        return Ok(None);
    }
    Ok(Some(q.apply_op3_no_bwd(k, v, &Neighbourhood { grid, kernel, heads, d, scale })?))
}

/// One side of a joint attention whole (`video::ltx_diffvae::joint`), or
/// `None` where the caller should use its parts: queries, keys and values
/// `[t·h·w, heads · d]` on `grid`, each query's `kernel` window cut at the
/// edges rather than shifted, and then the same rows and columns on the
/// other stream's frames `slots` names for its frame, two each (−1 for
/// none), of `pk` and `pv` `[P·h·w, heads · d]`, in one softmax: a video's
/// frames with keyframe planes as the other stream, or planes, a window of
/// one frame, with the video's. The answer in q's dtype. `None` as for
/// [`neighbourhood`], but for a window larger than the grid, which a joint
/// one allows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn neighbourhood_joint(q: &Tensor, k: &Tensor, v: &Tensor, grid: [usize; 3], pk: &Tensor, pv: &Tensor, slots: &[[i32; 2]], kernel: [usize; 3], heads: usize, scale: f32)
 -> candle_core::Result<Option<Tensor>> {
    let (n, width) = q.dims2()?;
    let d = width / heads.max(1);
    let plane = grid[1] * grid[2];
    let planes = pk.dim(0)? / plane.max(1);
    if kernels(q.device()).is_none()
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || [k, v, pk, pv].iter().any(|t| t.dtype() != q.dtype())
        || !matches!(d, 64 | 128)
        || d * heads != width
        || grid.iter().product::<usize>() != n
        || kernel.contains(&0)
        || kernel[2] > 13
        || slots.len() != grid[0]
        || slots.iter().flatten().any(|&p| p >= planes as i32)
        || k.dims() != q.dims()
        || v.dims() != q.dims()
        || pk.dims2()? != (planes * plane, width)
        || pv.dims() != pk.dims()
        || [q, k, v, pk, pv].iter().any(|t| stride_of(t.layout(), width).is_none())
    {
        return Ok(None);
    }
    let op = Joint { grid, kernel, heads, d, scale, pk: pk.clone(), pv: pv.clone(), slots: slots.iter().flatten().copied().collect() };
    Ok(Some(q.apply_op3_no_bwd(k, v, &op)?))
}

fn kernels(device: &candle_core::Device) -> Option<&'static crate::fused::metal::Kernels> {
    if matches!(std::env::var("KVAD_GPU_MPP_NEIGHBOURHOOD").as_deref(), Ok("0") | Ok("false")) {
        return None;
    }
    crate::fused::metal::tensor_library(device, "neighbourhood", SOURCE)
}

struct Neighbourhood {
    grid: [usize; 3],
    kernel: [usize; 3],
    heads: usize,
    d: usize,
    scale: f32,
}

impl CustomOp3 for Neighbourhood {
    fn name(&self) -> &'static str {
        "mpp_neighbourhood"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_neighbourhood runs on Metal only")
    }

    fn metal_fwd(&self, q: &MetalStorage, lq: &Layout, k: &MetalStorage, lk: &Layout, v: &MetalStorage, lv: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        use crate::fused::metal::{buffer, unpooled};
        let dt = q.dtype();
        let dev = q.device();
        let Some(lib) = kernels(&candle_core::Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_neighbourhood: this device cannot run it");
        };
        let (h, d) = (self.heads, self.d);
        let width = h * d;
        let ld = |l: &Layout| stride_of(l, width).ok_or_else(|| candle_core::Error::Msg(format!("mpp_neighbourhood: {l:?}")));
        let (ldq, ldk, ldv) = (ld(lq)?, ld(lk)?, ld(lv)?);
        let [gt, gh, gw] = self.grid;
        let [kt, kh, kw] = self.kernel;
        let n = gt * gh * gw;
        let tn = if dt == DType::F16 { "f16" } else { "bf16" };
        let pipe = lib.pipe(&format!("neighbourhood_{tn}_d{d}"))?;
        // At stage 5 the answer is a whole activation: of its own size, and
        // not in candle's pool, which would round it up to a power of two.
        let out = unpooled(dev, n * width * dt.size_in_bytes())?;
        #[cfg(test)]
        crate::fused::metal::RAN.with(|r| r.set(r.get() + 1));
        let i = |x: usize| x as i32;
        let params = Params {
            t: i(gt),
            h: i(gh),
            w: i(gw),
            kt: i(kt),
            kh: i(kh),
            kw: i(kw),
            ldq: i(ldq),
            ldk: i(ldk),
            ldv: i(ldv),
            heads: i(h),
            scale: self.scale * std::f32::consts::LOG2_E,
            ldpk: 0,
            ldpv: 0,
        };
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_neighbourhood");
        enc.set_compute_pipeline_state(&pipe);
        for (i, (s, l)) in [(q, lq), (k, lk), (v, lv)].into_iter().enumerate() {
            let (b, at) = buffer(s, l);
            enc.set_input_buffer(i, Some(&b), at);
        }
        enc.set_output_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &params);
        enc.dispatch_thread_groups(
            MTLSize { width: gw.div_ceil(TG_COLS), height: gh.div_ceil(TG_ROWS), depth: gt * h },
            // 32 × 4, as `mpp_attention` dispatches.
            MTLSize { width: 32, height: 4, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), n * width, dt), Shape::from((n, width))))
    }
}

/// [`neighbourhood_joint`]'s op: the planes' keys and values ride in it,
/// since an op takes three tensors.
struct Joint {
    grid: [usize; 3],
    kernel: [usize; 3],
    heads: usize,
    d: usize,
    scale: f32,
    pk: Tensor,
    pv: Tensor,
    slots: Vec<i32>,
}

impl CustomOp3 for Joint {
    fn name(&self) -> &'static str {
        "mpp_neighbourhood_joint"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_neighbourhood_joint runs on Metal only")
    }

    fn metal_fwd(&self, q: &MetalStorage, lq: &Layout, k: &MetalStorage, lk: &Layout, v: &MetalStorage, lv: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        use crate::fused::metal::{buffer, unpooled};
        let dt = q.dtype();
        let dev = q.device();
        let Some(lib) = kernels(&candle_core::Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_neighbourhood_joint: this device cannot run it");
        };
        let (h, d) = (self.heads, self.d);
        let width = h * d;
        let ld = |l: &Layout| stride_of(l, width).ok_or_else(|| candle_core::Error::Msg(format!("mpp_neighbourhood_joint: {l:?}")));
        let (ldq, ldk, ldv) = (ld(lq)?, ld(lk)?, ld(lv)?);
        let (pks, pkl) = self.pk.storage_and_layout();
        let (pvs, pvl) = self.pv.storage_and_layout();
        let (candle_core::Storage::Metal(pk), candle_core::Storage::Metal(pv)) = (&*pks, &*pvs) else {
            candle_core::bail!("mpp_neighbourhood_joint: planes not on Metal");
        };
        let (ldpk, ldpv) = (ld(pkl)?, ld(pvl)?);
        let [gt, gh, gw] = self.grid;
        let [kt, kh, kw] = self.kernel;
        let n = gt * gh * gw;
        let tn = if dt == DType::F16 { "f16" } else { "bf16" };
        let pipe = lib.pipe(&format!("neighbourhood_joint_{tn}_d{d}"))?;
        let out = unpooled(dev, n * width * dt.size_in_bytes())?;
        let slots = dev.new_buffer_with_data(&self.slots)?;
        #[cfg(test)]
        crate::fused::metal::RAN.with(|r| r.set(r.get() + 1));
        let i = |x: usize| x as i32;
        let params = Params {
            t: i(gt),
            h: i(gh),
            w: i(gw),
            kt: i(kt),
            kh: i(kh),
            kw: i(kw),
            ldq: i(ldq),
            ldk: i(ldk),
            ldv: i(ldv),
            heads: i(h),
            scale: self.scale * std::f32::consts::LOG2_E,
            ldpk: i(ldpk),
            ldpv: i(ldpv),
        };
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_neighbourhood_joint");
        enc.set_compute_pipeline_state(&pipe);
        for (i, (s, l)) in [(q, lq), (k, lk), (v, lv)].into_iter().enumerate() {
            let (b, at) = buffer(s, l);
            enc.set_input_buffer(i, Some(&b), at);
        }
        enc.set_output_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &params);
        enc.set_input_buffer(5, Some(&slots), 0);
        for (i, (s, l)) in [(pk, pkl), (pv, pvl)].into_iter().enumerate() {
            let (b, at) = buffer(s, l);
            enc.set_input_buffer(6 + i, Some(&b), at);
        }
        enc.dispatch_thread_groups(
            MTLSize { width: gw.div_ceil(TG_COLS), height: gh.div_ceil(TG_ROWS), depth: gt * h },
            MTLSize { width: 32, height: 4, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), n * width, dt), Shape::from((n, width))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn gpu() -> Option<Device> {
        let dev = Device::new_metal(0).ok()?;
        if !crate::mpp::available(&dev) {
            return None;
        }
        assert!(kernels(&dev).is_some(), "the neighbourhood kernels did not build");
        Some(dev)
    }

    fn db(got: &Tensor, want: &Tensor) -> f32 {
        let got = got.to_dtype(DType::F32).unwrap();
        assert!(got.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "a NaN or an infinity: not all written");
        let err = (&got - want).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let sig = want.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        10.0 * (sig / err.max(1e-30)).log10()
    }

    /// The kernel against the decoder's plain neighbourhood attention in
    /// f32, from the same bf16 numbers: every window the decoder has, grids
    /// that are and are not whole threadgroups, narrower than 16 columns,
    /// and an odd number of key runs.
    ///
    /// Measured when written: 57.0–59.4 dB in bf16 and 74.2–76.4 in f16,
    /// which is the answer's own rounding to 8 and 11 bits.
    #[test]
    fn agrees_with_the_plain_neighbourhood() {
        let Some(dev) = gpu() else {
            eprintln!("no matrix units for matmul2d; nothing to test");
            return;
        };
        // (grid, window, heads, d)
        let cases = [
            ([3, 8, 8], [3, 7, 7], 2, 64),
            ([5, 9, 21], [3, 5, 5], 3, 64),
            ([4, 7, 5], [3, 5, 5], 1, 64),
            ([3, 16, 16], [3, 3, 3], 2, 128),
            ([12, 13, 19], [11, 11, 11], 2, 64),
            ([11, 11, 11], [11, 11, 11], 1, 64),
            ([2, 6, 40], [1, 3, 13], 2, 64),
        ];
        for dt in [DType::BF16, DType::F16] {
            for (grid, kernel, heads, d) in cases {
                let n: usize = grid.iter().product();
                let w = heads * d;
                // Wide, so that softmaxes are peaked in places.
                let rand = |s: u64| (crate::image::nn::noise(s, &[n, w], &dev, DType::F32).unwrap() * 1.5).unwrap().to_dtype(dt).unwrap();
                let (q, k, v) = (rand(1), rand(2), rand(3));
                let g = crate::video::ltx_diffvae::Grid { t: grid[0], h: grid[1], w: grid[2] };
                let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
                let want = crate::video::ltx_diffvae::plain(&f(&q), &f(&k), &f(&v), g, kernel, heads).unwrap();
                let before = crate::fused::tests_ran();
                let got = neighbourhood(&q, &k, &v, grid, kernel, heads, 1.0).unwrap().expect("the kernel declined");
                assert_eq!(crate::fused::tests_ran(), before + 1, "the kernel did not run");
                let floor = if dt == DType::BF16 { 54.0 } else { 70.0 };
                let got_db = db(&got, &want);
                assert!(got_db > floor, "{dt:?} {grid:?} window {kernel:?} × {heads} × {d}: {got_db:.1} dB");
            }
        }
    }

    /// The joint kernel against the decoder's joint attention done by parts
    /// in f32: frames with two planes, one and none, windows cut at every
    /// edge, and the planes' side, onto frames.
    ///
    /// Measured when written: 57.1–58.8 dB, the answer's rounding to bf16,
    /// as the plain kernel's.
    #[test]
    fn the_joint_kernel_agrees_with_the_parts() {
        let Some(dev) = gpu() else { return };
        // (grid, planes, slots, window, heads, d)
        type Case = ([usize; 3], usize, Vec<[i32; 2]>, [usize; 3], usize, usize);
        let cases: [Case; 4] = [
            ([3, 8, 8], 2, vec![[0, 1], [1, 0], [1, -1]], [3, 7, 7], 2, 64),
            ([12, 13, 19], 3, (0..12).map(|t: i32| [t / 4, if t < 11 { (t / 4 + 1) % 3 } else { -1 }]).collect(), [11, 11, 11], 2, 64),
            ([5, 9, 21], 1, vec![[0, -1]; 5], [3, 5, 5], 1, 128),
            // The planes' side: a window of one plane, onto the frames.
            ([2, 16, 16], 7, vec![[6, 5], [3, 2]], [1, 3, 3], 2, 128),
        ];
        for (grid, planes, slots, kernel, heads, d) in cases {
            let n: usize = grid.iter().product();
            let np = planes * grid[1] * grid[2];
            let w = heads * d;
            let rand = |s: u64, rows: usize| (crate::image::nn::noise(s, &[rows, w], &dev, DType::F32).unwrap() * 1.5).unwrap().to_dtype(DType::BF16).unwrap();
            let (q, k, v, pk, pv) = (rand(1, n), rand(2, n), rand(3, n), rand(4, np), rand(5, np));
            let g = crate::video::ltx_diffvae::Grid { t: grid[0], h: grid[1], w: grid[2] };
            let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
            let (fk, fv, fpk, fpv) = (f(&k), f(&v), f(&pk), f(&pv));
            use crate::video::ltx_diffvae::{plain_part, Part};
            let mut parts = vec![Part { k: &fk, v: &fv, frames: grid[0], centres: (0..grid[0] as i32).collect(), kernel }];
            for j in 0..2 {
                parts.push(Part { k: &fpk, v: &fpv, frames: planes, centres: slots.iter().map(|s| s[j]).collect(), kernel: [1, kernel[1], kernel[2]] });
            }
            let answers: Vec<_> = parts.iter().map(|p| plain_part(&f(&q), g, p, heads).unwrap()).collect();
            let want = crate::video::ltx_diffvae::merge(&answers, DType::F32).unwrap();
            let before = crate::fused::tests_ran();
            let got = neighbourhood_joint(&q, &k, &v, grid, &pk, &pv, &slots, kernel, heads, 1.0).unwrap().expect("the kernel declined");
            assert_eq!(crate::fused::tests_ran(), before + 1, "the kernel did not run");
            let got_db = db(&got, &want);
            assert!(got_db > 54.0, "{grid:?} with {planes} planes, window {kernel:?}: {got_db:.1} dB");
        }
    }

    /// Rows may be a narrowed slice of the fused projection's answer.
    #[test]
    fn reads_a_slice_of_the_fused_projection() {
        let Some(dev) = gpu() else { return };
        let (grid, kernel, heads, d) = ([3, 9, 18], [3, 5, 5], 2, 64);
        let n: usize = grid.iter().product();
        let w = heads * d;
        let qkv = crate::image::nn::noise(4, &[n, 3 * w], &dev, DType::BF16).unwrap();
        let part = |i: usize| qkv.narrow(1, i * w, w).unwrap();
        let host = |t: Tensor| t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
        let got = host(neighbourhood(&part(0), &part(1), &part(2), grid, kernel, heads, 0.125).unwrap().unwrap());
        let packed = |i: usize| part(i).contiguous().unwrap();
        assert_eq!(got, host(neighbourhood(&packed(0), &packed(1), &packed(2), grid, kernel, heads, 0.125).unwrap().unwrap()));
    }

    /// What it leaves to the caller.
    #[test]
    fn declines_what_it_cannot_do() {
        let Some(dev) = gpu() else { return };
        let t = |n: usize, w: usize, dt: DType| Tensor::zeros((n, w), dt, &dev).unwrap();
        let bf = t(3 * 8 * 8, 128, DType::BF16);
        assert!(neighbourhood(&bf, &bf, &bf, [3, 8, 8], [3, 7, 7], 2, 1.0).unwrap().is_some());
        let f32 = t(3 * 8 * 8, 128, DType::F32);
        assert!(neighbourhood(&f32, &f32, &f32, [3, 8, 8], [3, 7, 7], 2, 1.0).unwrap().is_none());
        // A window larger than the grid, or wider than 13 columns.
        assert!(neighbourhood(&bf, &bf, &bf, [3, 8, 8], [3, 9, 7], 2, 1.0).unwrap().is_none());
        let wide = t(3 * 8 * 16, 128, DType::BF16);
        assert!(neighbourhood(&wide, &wide, &wide, [3, 8, 16], [3, 7, 15], 2, 1.0).unwrap().is_none());
        // A head of 32, and a grid that is not the tokens.
        assert!(neighbourhood(&bf, &bf, &bf, [3, 8, 8], [3, 7, 7], 4, 1.0).unwrap().is_none());
        assert!(neighbourhood(&bf, &bf, &bf, [3, 8, 7], [3, 7, 7], 2, 1.0).unwrap().is_none());
    }

    /// Not a test, a measurement: the decoder's attentions at a 768 × 512
    /// clip's sizes, 16 frames of stages 4 and 5. TFLOP/s counts the
    /// products each query's window asks for, not the hidden ones.
    ///
    /// Measured when written: stage 5 47 ms, 11.4 TFLOP/s (about 21 with the
    /// hidden keys, where `mpp_attention` runs); stage 4 8.2 ms and stage 1
    /// 4.9 ms, 1.7–1.8, their narrow windows wasting most of each run of 16.
    ///
    ///     cargo test --release -p kvad-gpu neighbourhood_race -- --ignored --nocapture
    #[test]
    #[ignore]
    fn neighbourhood_race() {
        let dev = gpu().expect("no matrix units");
        // (what, grid, window, heads)
        let shapes = [
            ("stage 5, 768 × 512, 16 frames", [16, 128, 192], [11, 11, 11], 4),
            ("stage 4, 768 × 512, 16 frames", [16, 64, 96], [3, 5, 5], 8),
            ("stage 1, 768 × 512", [18, 16, 24], [3, 7, 7], 32),
        ];
        for (what, grid, kernel, heads) in shapes {
            let n: usize = grid.iter().product();
            let w = heads * 64;
            let rand = |s: u64| crate::image::nn::noise(s, &[n, w], &dev, DType::BF16).unwrap();
            let (q, k, v) = (rand(1), rand(2), rand(3));
            let flops = 4.0 * (n * kernel.iter().product::<usize>() * w) as f64;
            let run = || drop(neighbourhood(&q, &k, &v, grid, kernel, heads, 0.125).unwrap().unwrap());
            run();
            dev.synchronize().unwrap();
            let mut times = vec![];
            for _ in 0..5 {
                let t = std::time::Instant::now();
                run();
                dev.synchronize().unwrap();
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(f64::total_cmp);
            let s = times[2];
            println!("{what:<32} {:9.2} ms  {:5.1} TFLOP/s of the window's products", s * 1e3, flops / s / 1e12);
        }
        // Stage 5 with two planes a frame, as a keyframe decode has it.
        let (grid, kernel, heads) = ([16, 128, 192], [11, 11, 11], 4);
        let n: usize = grid.iter().product();
        let w = heads * 64;
        let rand = |s: u64, rows: usize| crate::image::nn::noise(s, &[rows, w], &dev, DType::BF16).unwrap();
        let (q, k, v, pk, pv) = (rand(1, n), rand(2, n), rand(3, n), rand(4, 2 * 128 * 192), rand(5, 2 * 128 * 192));
        let slots = vec![[0, 1]; 16];
        let run = || drop(neighbourhood_joint(&q, &k, &v, grid, &pk, &pv, &slots, kernel, heads, 0.125).unwrap().unwrap());
        run();
        dev.synchronize().unwrap();
        let mut times = vec![];
        for _ in 0..5 {
            let t = std::time::Instant::now();
            run();
            dev.synchronize().unwrap();
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        println!("{:<32} {:9.2} ms", "stage 5, joint, two planes", times[2] * 1e3);
    }
}
