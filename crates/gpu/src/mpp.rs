//! Matmuls on the M5 GPU's neural accelerators (#52): GGML blocks in
//! [`Blocks`] (Kvad's own Q8_0, and a GGUF's Q4_K, Q5_K and Q6_K), and f16
//! or bf16 in [`dense`].
//!
//! Every M5 GPU core carries a matrix unit, and a shader reaches it only
//! through Metal 4's tensor API, `mpp::tensor_ops::matmul2d`. candle's
//! kernels predate that API. They multiply with `simdgroup_matrix`, on the
//! ordinary ALUs, and at a diffusion model's shapes they stop at about
//! 7.5 TFLOP/s on an M5 Pro whatever the dtype. `examples/neural_accel`
//! measured this kernel at 2.3–2.7× that, counting the f32→f16 cast in front
//! of it.
//!
//! # Only where there is no decode step
//!
//! `matmul2d` cannot read GGML blocks, so the kernel needs the raw block
//! bytes. `QTensor` keeps its Metal buffer private, and candle offers no way
//! to build one around a buffer someone else holds. So a weight is either a
//! `QTensor` or [`Blocks`], never both, unless it is kept twice.
//!
//! A language model needs `QTensor`: its decode step is one row, which is
//! candle's matrix-vector kernel's job and no matrix unit's. An image model
//! has no decode step. Every product it takes has hundreds or thousands of
//! rows: patches, pixels, or a prompt's tokens. So the image pipelines hold
//! their Q8_0 projections as [`Blocks`] and nothing else, and the text models
//! keep `QMatMul`.
//!
//! # The kernel
//!
//! `C = A · Wᵀ + b`, where `A` is `[M, K]` in f16 or bf16, `W` is `[N, K]` in
//! Q8_0 blocks of 32 weights along `K` (an f16 scale, then 32 `int8`s), and
//! `C` is `[M, N]` in f32, f16 or bf16. A threadgroup owns a `64 × 64` tile
//! of `C`, four SIMD groups a `32 × 32` corner each, and walks `K` 32 at a
//! time. At each step:
//! 1. its threads unpack the `64 × 32` slab of `W` the tile needs into
//!    threadgroup memory as f16: `scale × int8`, rounded once;
//! 2. each SIMD group loads its rows of `A` straight from device memory,
//!    element by element, into `16 × 16` fragments in registers, converting
//!    bf16 to f16 as it goes, and multiplies them by the slab's fragments on
//!    the matrix unit, `16 × 32 × 16` at a time, into f32 sums that stay in
//!    registers until the end. This is how `mpp_attention` and MLX's M5
//!    kernels are built, and every index into a fragment array is a
//!    constant, for the reasons `mpp_attention` gives.
//!
//! There are two slabs, and step 1 for the next step overlaps step 2 for
//! this one. Each thread loads the next step's raw blocks into registers
//! before the multiply, and unpacks them into the other slab after it. That
//! needs one barrier a step instead of two, and the probe measured it
//! 1.02–1.09× faster than one slab on every shape it tried, with the same
//! output bit for bit.
//!
//! The sums leave through an epilogue: plus the bias, through a
//! feed-forward's GELU where asked, and rounded once, to the dtype asked
//! for. Nothing reads an f32 answer back to finish it.
//!
//! At the LTX-2.5 DiT's shapes this runs a projection at 21–25 TFLOP/s,
//! bias and all, where MLX's 8-bit `quantized_matmul` runs 22–26 on the same
//! machine. The kernel before it, which gave `matmul2d` whole slices over
//! four SIMD groups and answered in f32 for candle to add the bias and
//! round, ran the same projections at 8.5–17 TFLOP/s counting what came
//! after it. Sums are bit for bit the same either way. Stepping `K` 64 at a
//! time, as MLX does, was slower here (21 against 25 TFLOP/s), and a
//! `128 × 64` tile over eight SIMD groups was no faster.
//!
//! Edges: rows of `A` past `M` load as zeros, rows of `W` past `N` unpack
//! as zeros, and the store writes only what exists. `K` is always a
//! multiple of 32, because a Q8_0 block is 32 wide.
//!
//! # Other blocks
//!
//! Only step 1 knows what a block is. It turns a run of eight weights into
//! f16, and one decoder for each format does that ([`KINDS`]): Q8_0's is
//! `scale × int8` in f16, as it always was, and its output is the same bit
//! for bit. GGML's k-quants, which the community's GGUFs are made of, pack
//! 256 weights to a super-block, in sub-blocks of 32 (Q4_K, Q5_K) or 16
//! (Q6_K), each with its own scale. Any run of eight lies in one sub-block,
//! so it decodes on its own: the super-block's scales, then four or eight
//! bytes of `q`s, in GGML's own arithmetic, in f32, rounded once. `K` is
//! then a multiple of 256.
//!
//! At the LTX-2.5 DiT's shapes they run at 16–21 TFLOP/s, 75–90% of Q8_0's
//! rate: the decoding is more arithmetic and more reads a run. candle's
//! `QMatMul` runs the same blocks at 6–7, on the ALUs.
//!
//! # Where it runs
//!
//! Only on a GPU of Apple's tenth family, the M5's, because that is where
//! the matrix units are. Before that, `matmul2d` runs on the ALUs, and there
//! it is not measured. It must also compile: the tensor API needs Metal 4,
//! which is macOS 26. `KVAD_GPU_MPP=0` turns it off, for measuring what it is
//! worth.
//!
//! The activations are read as f16, from f16 or bf16; f32 is cast to f16
//! first. `matmul2d` also takes f32 on the left, and returns candle's
//! product bit for bit at candle's speed: the f32 path is not accelerated.
//!
//! # Dense
//!
//! A dense weight is a plain tensor, so nothing stops candle and this kernel
//! from sharing it, and the language models get it too. [`dense`] takes
//! `x · w` whenever it is f16 or bf16 and at least [`DENSE_ROWS`] rows: a
//! prefill chunk, an image model's every projection. A decode step's one row
//! is still candle's, because a matrix-vector product is not a matrix unit's
//! job, and there `matmul2d` is slower.

use candle_core::backend::BackendStorage;
use candle_core::quantized::GgmlDType;
use candle_core::{CpuStorage, CustomOp2, CustomOp3, DType, Device, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLSize};
use std::sync::OnceLock;

/// The block formats the kernel reads: GGML's type, the kernel's name for
/// it, and one block's width in weights and size in bytes.
const KINDS: [(GgmlDType, &str, usize, usize); 4] =
    [(GgmlDType::Q8_0, "q8_0", 32, 34), (GgmlDType::Q4K, "q4_K", 256, 144), (GgmlDType::Q5K, "q5_K", 256, 176), (GgmlDType::Q6K, "q6_K", 256, 210)];

/// Whether [`Blocks`] reads `kind`, a matrix `k` wide.
pub(crate) fn reads(kind: GgmlDType, k: usize) -> bool {
    KINDS.iter().any(|&(t, _, width, _)| t == kind && k % width == 0)
}

/// Rows of `C` per threadgroup, columns, and the SIMD groups sharing the
/// tile, each a `32 × 32` corner.
const BM: usize = 64;
const BN: usize = 64;
const SIMD_GROUPS: usize = 4;

/// A slab row in halves: 32 weights, then 8 of padding, so that the SIMD
/// groups' fragment loads do not collide in threadgroup memory's banks.
const SLAB_ROW: usize = 40;

/// Fragments of `matmul2d`'s `16 × 32 × 16` product, and the loops that
/// index them, in Metal: shared by this module's Q8_0 kernel and
/// `mpp_attention`. A macro, so that `concat!` can splice it into each
/// source.
macro_rules! fragments {
    () => {
        r#"
// `f(0)` to `f(N - 1)`, each index a compile-time constant. Every array of
// fragments is indexed only through this: an index the compiler cannot
// resolve puts the whole array in memory (`mpp_attention` says more).
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

// A fragment from row-major device memory, element by element: reading each
// row's four as one vector and copying them out was 13–15% slower. With
// EDGE, only `rows` rows exist and the rest read as zeros; without, all 16
// do and nothing is checked.
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

// The same from threadgroup memory, all 16 rows.
template <typename T>
inline frag<T> load(threadgroup const T *p, int ld, short2 at) {
    frag<T> f;
    p += at.y * ld + at.x;
    EACH(2, i, EACH(4, c, f[i * 4 + c] = p[i * 8 * ld + c];););
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
"#
    };
}
pub(crate) use fragments;

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
constant constexpr int BM = 64;
constant constexpr int BN = 64;
constant constexpr int BK = 32;
constant constexpr int NSG = 4;
constant constexpr int SLAB_ROW = 40;

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

// GGML's k-quants: super-blocks of 256 weights along a row.
//
// Q4_K and Q5_K: eight sub-blocks of 32, each with a 6-bit scale and a 6-bit
// minimum packed into `scales`, under the super-block's own `d` and `dmin`:
// a weight is `d·sc·q − dmin·m`. The 4-bit `q`s of sub-blocks `2c` and
// `2c + 1` share the 32 bytes `qs[32c…]`, low nibble and high. Q5_K adds a
// fifth bit, bit `2c` or `2c + 1` of `qh[l]`.
struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

// Q6_K: sixteen sub-blocks of 16, each with an 8-bit scale, and 6-bit `q`s
// centred on 32: a weight is `d·sc·(q − 32)`. Each half of the super-block
// is 128 weights in four quarters of 32. Their low four bits share `ql`'s 64
// bytes, two quarters to a nibble, and their top two bits share `qh`'s 32,
// two bits a quarter.
struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q4_K) == 144 && sizeof(block_q5_K) == 176 && sizeof(block_q6_K) == 210,
              "the blocks are GGML's, byte for byte");

// A sub-block's 6-bit scale and minimum, from Q4_K's and Q5_K's `scales`:
// the first four whole in the low six bits of bytes 0–3 and 4–7, the last
// four as nibbles of bytes 8–11 topped up with those bytes' spare two bits.
inline uchar2 scale_min(device const uchar *q, int s) {
    return s < 4 ? uchar2(q[s] & 63, q[s + 4] & 63)
                 : uchar2((q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4), (q[s + 4] >> 4) | ((q[s] >> 6) << 4));
}

inline uchar4 bytes4(device const uchar *p) {
    return uchar4(*(device const packed_uchar4 *)p);
}

// A run of eight weights of one row as f16, `lo` then `hi`: those at `k + j`
// onwards, `j` a multiple of 8 below 32. `row` is the row's first block.
// The arithmetic is GGML's own dequantisation in f32, rounded once; Q8_0's
// is the multiply in f16 this kernel has always done.
struct q8_0 {
    static constant constexpr int WIDTH = 32;
    static inline void run(device const uchar *row, int k, int j, thread half4 &lo, thread half4 &hi) {
        device const block_q8_0 &b = ((device const block_q8_0 *)row)[k / 32];
        // The `int8`s sit two bytes into a 34-byte block, so only a packed
        // (byte-aligned) load may read them.
        device const packed_char4 *q = (device const packed_char4 *)(b.qs + j);
        lo = b.d * half4(char4(q[0]));
        hi = b.d * half4(char4(q[1]));
    }
};

struct q4_K {
    static constant constexpr int WIDTH = 256;
    static inline void run(device const uchar *row, int k, int j, thread half4 &lo, thread half4 &hi) {
        device const block_q4_K &b = ((device const block_q4_K *)row)[k / 256];
        const int p = k % 256 + j, c = p / 64, h = (p / 32) & 1, l = p % 32;
        const uchar2 sm = scale_min(b.scales, 2 * c + h);
        const float dl = float(b.d) * sm.x, ml = float(b.dmin) * sm.y;
        const uchar shift = h * 4;
        lo = half4(dl * float4((bytes4(b.qs + 32 * c + l) >> shift) & 0xF) - ml);
        hi = half4(dl * float4((bytes4(b.qs + 32 * c + l + 4) >> shift) & 0xF) - ml);
    }
};

struct q5_K {
    static constant constexpr int WIDTH = 256;
    static inline void run(device const uchar *row, int k, int j, thread half4 &lo, thread half4 &hi) {
        device const block_q5_K &b = ((device const block_q5_K *)row)[k / 256];
        const int p = k % 256 + j, c = p / 64, h = (p / 32) & 1, l = p % 32;
        const uchar2 sm = scale_min(b.scales, 2 * c + h);
        const float dl = float(b.d) * sm.x, ml = float(b.dmin) * sm.y;
        const uchar shift = h * 4, bit = uchar(1 << (2 * c + h));
        auto q = [&](int o) {
            const uchar4 top = select(uchar4(0), uchar4(16), (bytes4(b.qh + l + o) & bit) != 0);
            return float4(((bytes4(b.qs + 32 * c + l + o) >> shift) & 0xF) + top);
        };
        lo = half4(dl * q(0) - ml);
        hi = half4(dl * q(4) - ml);
    }
};

struct q6_K {
    static constant constexpr int WIDTH = 256;
    static inline void run(device const uchar *row, int k, int j, thread half4 &lo, thread half4 &hi) {
        device const block_q6_K &b = ((device const block_q6_K *)row)[k / 256];
        const int p = k % 256 + j, n = p / 128, quarter = (p % 128) / 32, l = p % 32;
        const float d = float(b.d) * float(b.scales[8 * n + l / 16 + 2 * quarter]);
        device const uchar *ql = b.ql + 64 * n + (quarter & 1) * 32 + l;
        device const uchar *qh = b.qh + 32 * n + l;
        const uchar shift = (quarter >> 1) * 4, up = 2 * quarter;
        auto q = [&](int o) {
            const uchar4 v = ((bytes4(ql + o) >> shift) & 0xF) | (((bytes4(qh + o) >> up) & 3) << 4);
            return float4(v) - 32.0f;
        };
        lo = half4(d * q(0));
        hi = half4(d * q(4));
    }
};

// Each decoder's block, for a row's stride.
template <typename B> struct block_of;
template <> struct block_of<q8_0> { using type = block_q8_0; };
template <> struct block_of<q4_K> { using type = block_q4_K; };
template <> struct block_of<q5_K> { using type = block_q5_K; };
template <> struct block_of<q6_K> { using type = block_q6_K; };
template <typename B> using B_block = typename block_of<B>::type;

// Each thread's share of a slab: `PER` runs of 8 weights. Adjacent threads
// take adjacent runs of the same row, so a SIMD group reads a row
// contiguously.
constant constexpr int RUN = 8;
constant constexpr int PER = BN * BK / RUN / (32 * NSG);
static_assert(PER * RUN * 32 * NSG == BN * BK, "every thread unpacks the same number of runs");

// A step's weights, from device memory into this thread's registers as f16.
// Rows past `N` read as zeros. `stride` is a row's bytes.
template <typename B>
inline void fetch(device const uchar *w, int N, int n0, long stride, int k, ushort tid,
                  thread half4 *lo, thread half4 *hi) {
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int r = tid + p * 32 * NSG;
        const int n = r / (BK / RUN);
        const int j = (r % (BK / RUN)) * RUN;
        if (n0 + n < N) {
            B::run(w + (n0 + n) * stride, k, j, lo[p], hi[p]);
        } else {
            lo[p] = half4(0);
            hi[p] = half4(0);
        }
    }
}

// Those registers into a slab.
inline void unpack(threadgroup half *slab, ushort tid, thread half4 *lo, thread half4 *hi) {
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int r = tid + p * 32 * NSG;
        const int n = r / (BK / RUN);
        const int j = (r % (BK / RUN)) * RUN;
        threadgroup half4 *dst = (threadgroup half4 *)(slab + n * SLAB_ROW + j);
        dst[0] = lo[p];
        dst[1] = hi[p];
    }
}

// What the store does to each sum on its way out, in f32: add the bias,
// then, for a feed-forward's up projection, take the tanh-GELU, which is
// `ltx_fused`'s formula.
struct Epilogue {
    int bias;
    int gelu;
};

// B is the weights' block format, TI what `A` is read in, O what `C` is
// written in.
template <typename B, typename TI, typename O>
[[kernel, max_total_threads_per_threadgroup(128)]] void mm_q(
        device const TI *a [[buffer(0)]],
        device const uchar *w [[buffer(1)]],
        device O *c [[buffer(2)]],
        constant int &M [[buffer(3)]],
        constant int &N [[buffer(4)]],
        constant int &K [[buffer(5)]],
        device const float *bias [[buffer(6)]],
        constant Epilogue &ep [[buffer(7)]],
        threadgroup half *slab [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    // This SIMD group's corner: 32 rows from `m0`, 32 of the tile's columns
    // from `tn`.
    const int n0 = tg.x * BN;
    const int m0 = tg.y * BM + (sg / 2) * 32;
    const int tn = (sg % 2) * 32;
    const short2 at = place(lane);
    const int rows = M - m0;
    a += long(m0) * K;

    // `acc[i * 2 + j]`: rows `16·i`, columns `16·j` of the corner.
    frag<float> acc[4];
    EACH(4, i, acc[i] = 0;);

    const long stride = long(K / B::WIDTH) * sizeof(B_block<B>);
    half4 lo[PER], hi[PER];
    fetch<B>(w, N, n0, stride, 0, tid, lo, hi);
    unpack(slab, tid, lo, hi);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // EDGE is a threadgroup whose tile runs past `M`: only there are rows
    // checked.
    auto walk = [&](auto edge) {
        constexpr bool EDGE = decltype(edge)::value;
        int s = 0;
        for (int k = 0; k < K; k += BK, ++s) {
            const bool more = k + BK < K;
            // The next step's device reads go out before this step's
            // multiply, so they are in flight while it runs.
            if (more) {
                fetch<B>(w, N, n0, stride, k + BK, tid, lo, hi);
            }
            threadgroup const half *ws = slab + (s & 1) * BN * SLAB_ROW + tn * SLAB_ROW;
            EACH(BK / 16, kk,
                frag<half> af[2];
                EACH(2, i, af[i] = frag<half>(load<EDGE>(a + i * 16 * K + k + kk * 16, K, at, rows - i * 16)););
                const frag<half> b0 = load(ws + kk * 16, SLAB_ROW, at);
                const frag<half> b1 = load(ws + 16 * SLAB_ROW + kk * 16, SLAB_ROW, at);
                EACH(2, i, mma<true>(acc[i * 2], acc[i * 2 + 1], af[i], b0, b1););
            );
            if (more) {
                unpack(slab + ((s + 1) & 1) * BN * SLAB_ROW, tid, lo, hi);
            }
            // One barrier does both jobs. The slab written just now is
            // complete before anyone multiplies from it. And the slab the
            // step after next will overwrite has been read by everyone: that
            // was this step's multiply. One slab needed a barrier on each
            // side.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    };
    // The whole threadgroup takes one branch, so its barriers match.
    if (int(tg.y) * BM + BM <= M) {
        walk(false_type());
    } else {
        walk(true_type());
    }

    // Each sum leaves once, finished: nothing reads an f32 `C` back to add
    // a bias or round it.
    EACH(2, i, EACH(2, j, EACH(2, h,
        const int m = m0 + i * 16 + at.y + h * 8;
        if (m < M) {
            EACH(4, cc,
                const int n = n0 + tn + j * 16 + at.x + cc;
                if (n < N) {
                    float v = acc[i * 2 + j][h * 4 + cc];
                    if (ep.bias) {
                        v += bias[n];
                    }
                    if (ep.gelu) {
                        v = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * v * (1.0f + 0.044715f * v * v)));
                    }
                    c[long(m) * N + n] = O(v);
                }
            );
        }
    );););
}

#define MMQ(B, TI, IN, O, ON) \
    template [[host_name("mm_" #B "_" #IN "_" #ON)]] [[kernel]] decltype(mm_q<B, TI, O>) mm_q<B, TI, O>;
#define MMQ_ALL(B) \
    MMQ(B, half, f16, float, f32) \
    MMQ(B, half, f16, half, f16) \
    MMQ(B, half, f16, bfloat, bf16) \
    MMQ(B, bfloat, bf16, float, f32) \
    MMQ(B, bfloat, bf16, half, f16) \
    MMQ(B, bfloat, bf16, bfloat, bf16)
MMQ_ALL(q8_0)
MMQ_ALL(q4_K)
MMQ_ALL(q5_K)
MMQ_ALL(q6_K)

// Dense: `C = A · B`, all three row-major and of one dtype. `A` is
// `[M, K]`, `B` is `[K, N]` (a weight stored `[in, out]`, as `Proj::Dense`
// keeps it), and `C` is `[M, N]`. `matmul2d` reads both straight from device
// memory and walks all of `K` itself; the kernel only says which tile is
// whose. Measured against feeding `K` in steps, or staging `B` in
// threadgroup memory, this is the fastest there is.
//
// With ACC, `C += A · B`: what is in `C` joins the f32 sums before they are
// rounded, once. `C` filled with a bias's rows is then `A · B + b` rounded
// as PyTorch's `addmm` rounds it, where adding the bias to the rounded
// product rounds twice.
template <typename T, int TM, int TN, int TSG, bool ACC>
kernel void mm_dense(device T *a [[buffer(0)]],
                     device T *b [[buffer(1)]],
                     device T *c [[buffer(2)]],
                     constant int &M [[buffer(3)]],
                     constant int &N [[buffer(4)]],
                     constant int &K [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device T, dextents<int32_t, 2>, tensor_inline> ta(a, dextents<int32_t, 2>(K, M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tb(b, dextents<int32_t, 2>(N, K));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tc(c, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, false,
                                              ACC ? matmul2d_descriptor::mode::multiply_accumulate
                                                  : matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<TSG>> op;
    auto ma = ta.slice(0, tg.y * TM);
    auto mb = tb.slice(tg.x * TN, 0);
    auto mc = tc.slice(tg.x * TN, tg.y * TM);
    op.run(ma, mb, mc);
}

#define DENSE(T, tn, TM, TN, TSG) \
    template [[host_name("mm_dense_" #tn "_" #TM "x" #TN)]] [[kernel]] \
    decltype(mm_dense<T, TM, TN, TSG, false>) mm_dense<T, TM, TN, TSG, false>; \
    template [[host_name("mm_dense_acc_" #tn "_" #TM "x" #TN)]] [[kernel]] \
    decltype(mm_dense<T, TM, TN, TSG, true>) mm_dense<T, TM, TN, TSG, true>;
DENSE(half, f16, 64, 64, 4)
DENSE(half, f16, 128, 128, 8)
DENSE(bfloat, bf16, 64, 64, 4)
DENSE(bfloat, bf16, 128, 128, 8)
"#
);

/// Whether this device runs [`Blocks`] and [`dense`], answered once per process.
///
/// Compiling is part of the answer. A GPU of the right family on a macOS
/// without Metal 4 fails here, and says why on stderr, instead of failing at
/// the first matmul of a generation.
pub(crate) fn available(device: &Device) -> bool {
    pipes(device).is_some()
}

/// Every kernel in [`SOURCE`], built.
struct Pipes {
    /// Each of [`KINDS`], `[f16, bf16]` in, each `[f32, f16, bf16]` out.
    q: [[[ComputePipeline; 3]; 2]; 4],
    /// `[C = A·B, C += A·B]`, each `[f16, bf16]`, each in [`DENSE_TILES`]'
    /// order.
    dense: [[[ComputePipeline; 2]; 2]; 2],
}

fn pipes(device: &Device) -> Option<&'static Pipes> {
    static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();
    let Device::Metal(md) = device else { return None };
    PIPES.get_or_init(|| {
        if matches!(std::env::var("KVAD_GPU_MPP").as_deref(), Ok("0") | Ok("false")) {
            return None;
        }
        let raw = md.metal_device().as_ref();
        if !raw.supportsFamily(MTLGPUFamily::Apple10) {
            return None;
        }
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let built = md.metal_device().new_library_with_source(SOURCE, Some(&opts)).and_then(|lib| {
            let pipe = |name: &str| {
                lib.get_function(name, None)
                    .and_then(|f| md.metal_device().new_compute_pipeline_state_with_function(&f))
            };
            let dense = |tn: &str| -> Result<[ComputePipeline; 2], candle_metal_kernels::MetalKernelError> {
                let [(m0, n0, _), (m1, n1, _)] = DENSE_TILES;
                Ok([pipe(&format!("mm_dense_{tn}_{m0}x{n0}"))?, pipe(&format!("mm_dense_{tn}_{m1}x{n1}"))?])
            };
            let dense = |acc: &str| -> Result<[[ComputePipeline; 2]; 2], candle_metal_kernels::MetalKernelError> {
                Ok([dense(&format!("{acc}f16"))?, dense(&format!("{acc}bf16"))?])
            };
            let q = |kind: &str, inp: &str| -> Result<[ComputePipeline; 3], candle_metal_kernels::MetalKernelError> {
                let p = |on: &str| pipe(&format!("mm_{kind}_{inp}_{on}"));
                Ok([p("f32")?, p("f16")?, p("bf16")?])
            };
            let q = |kind: &str| -> Result<[[ComputePipeline; 3]; 2], candle_metal_kernels::MetalKernelError> { Ok([q(kind, "f16")?, q(kind, "bf16")?]) };
            let [a, b, c, d] = KINDS.map(|(_, name, _, _)| name);
            Ok(Pipes { q: [q(a)?, q(b)?, q(c)?, q(d)?], dense: [dense("")?, dense("acc_")?] })
        });
        match built {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("kvad: the M5 matmul kernels did not build, so candle's are used: {e}");
                None
            }
        }
    })
    .as_ref()
}

/// A threadgroup of `sg` SIMD groups, laid out 32 × `sg`: a SIMD group to a
/// row. `32·sg × 1` is the same threads in the same SIMD groups, and nothing
/// in the kernels reads the shape, yet at the DiT's shapes it was slower:
/// raced in one process, 32 × `sg` ran Q8_0 at 1.14–1.22× and bf16 at
/// 1.09–1.31× where `m` is 24 576, and within a few percent either way at
/// 1024 rows. `mpp_attention` found it first: MLX dispatches its M5
/// attention this way, and dispatched `128 × 1` it lost a quarter.
fn shape(sg: usize) -> MTLSize {
    MTLSize { width: 32, height: sg, depth: 1 }
}

/// One `[n, k]` weight matrix, in GGML blocks on the GPU, for [`SOURCE`]:
/// Q8_0, which is Kvad's own quantisation, or Q4_K, Q5_K or Q6_K, which is
/// a GGUF's.
pub(crate) struct Blocks {
    blocks: Tensor,
    /// Its place in [`KINDS`].
    kind: usize,
    n: usize,
    k: usize,
}

impl Blocks {
    /// `blocks` are `[n, k]` as GGML lays `kind` out: row after row, each
    /// row `k / width` blocks.
    pub(crate) fn new(kind: GgmlDType, blocks: &[u8], n: usize, k: usize, device: &Device) -> candle_core::Result<Self> {
        let Some(at) = KINDS.iter().position(|&(t, ..)| t == kind) else {
            candle_core::bail!("the M5 kernel reads no {kind:?} blocks");
        };
        let (_, name, width, bytes) = KINDS[at];
        if k % width != 0 || blocks.len() != n * k / width * bytes {
            candle_core::bail!("{} bytes are not a [{n}, {k}] matrix of {name} blocks", blocks.len());
        }
        Ok(Blocks { blocks: Tensor::from_slice(blocks, blocks.len(), device)?, kind: at, n, k })
    }

    /// `x · Wᵀ` over the last axis, in f32, whatever `x`'s dtype and leading
    /// axes.
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.linear(x, None, DType::F32, false)
    }

    /// `x · Wᵀ + b` over the last axis, then tanh-GELU if `gelu`, answered
    /// in `out` (f32, f16 or bf16), whatever `x`'s dtype and leading axes.
    /// The bias and GELU are applied to the f32 sum as the kernel stores
    /// it, and it is rounded once, to `out`.
    pub(crate) fn linear(&self, x: &Tensor, bias: Option<&Tensor>, out: DType, gelu: bool) -> candle_core::Result<Tensor> {
        let dims = x.dims().to_vec();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        // The kernel reads bf16 as it is and rounds it to f16 itself.
        let x2 = x.reshape((rows, self.k))?;
        let x2 = match x2.dtype() {
            DType::F16 | DType::BF16 => x2,
            _ => x2.to_dtype(DType::F16)?,
        }
        .contiguous()?;
        let op = Q8Op { kind: self.kind, n: self.n, k: self.k, out, gelu, bias: bias.is_some() };
        let y = match bias {
            Some(b) => {
                if b.elem_count() != self.n {
                    candle_core::bail!("mpp_q8_0: a bias of {} for {} outputs", b.elem_count(), self.n);
                }
                x2.apply_op3_no_bwd(&self.blocks, &b.to_dtype(DType::F32)?.contiguous()?, &op)?
            }
            // The bias buffer is bound whatever, and read only when there
            // is one: the blocks stand in.
            None => x2.apply_op3_no_bwd(&self.blocks, &self.blocks, &op)?,
        };
        let mut shape = dims;
        *shape.last_mut().unwrap() = self.n;
        y.reshape(shape)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.blocks.elem_count()
    }

    pub(crate) fn params(&self) -> usize {
        self.n * self.k
    }
}

/// One call of the kernel: the matrix's blocks and shape, and what its
/// store does.
struct Q8Op {
    kind: usize,
    n: usize,
    k: usize,
    out: DType,
    gelu: bool,
    bias: bool,
}

/// The kernel's `Epilogue`, as Metal lays it out.
#[repr(C)]
struct Epilogue {
    bias: i32,
    gelu: i32,
}

impl CustomOp3 for Q8Op {
    fn name(&self) -> &'static str {
        "mpp_q8_0"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_q8_0 runs on Metal only")
    }

    fn metal_fwd(&self, a: &MetalStorage, la: &Layout, w: &MetalStorage, lw: &Layout, b: &MetalStorage, lb: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        let (m, k) = la.shape().dims2()?;
        let n = self.n;
        let inp = match a.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            dt => candle_core::bail!("mpp_q8_0: cannot read {dt:?}"),
        };
        if k != self.k || !la.is_contiguous() || lw.start_offset() != 0 {
            candle_core::bail!("mpp_q8_0: wants contiguous [m, {}], got {:?}", self.k, la);
        }
        if self.bias && (b.dtype() != DType::F32 || !lb.is_contiguous()) {
            candle_core::bail!("mpp_q8_0: wants a contiguous f32 bias, got {:?} {:?}", b.dtype(), lb);
        }
        let which = match self.out {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            dt => candle_core::bail!("mpp_q8_0: cannot answer in {dt:?}"),
        };
        let dev = a.device();
        let Some(pipes) = pipes(&Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_q8_0: this device cannot run it");
        };
        let bytes = m * n * self.out.size_in_bytes();
        let out = dev.allocate_buffer(bytes)?;
        // A pooled buffer can still hold the last product of the same size,
        // so a kernel that wrote nothing would pass a test by agreeing with
        // it. Under test, every output starts as NaN.
        #[cfg(test)]
        {
            let mut blit = dev.blit_command_encoder()?;
            blit.fill_buffer(&out, (0, bytes), 0xff);
        }
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_q8_0");
        enc.set_compute_pipeline_state(&pipes.q[self.kind][inp][which]);
        enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * a.dtype().size_in_bytes());
        enc.set_input_buffer(1, Some(w.buffer()), 0);
        enc.set_output_buffer(2, Some(&out), 0);
        enc.set_bytes(3, &(m as i32));
        enc.set_bytes(4, &(n as i32));
        enc.set_bytes(5, &(k as i32));
        let off = if self.bias { lb.start_offset() * 4 } else { 0 };
        enc.set_input_buffer(6, Some(b.buffer()), off);
        enc.set_bytes(7, &Epilogue { bias: self.bias as i32, gelu: self.gelu as i32 });
        // Two slabs of `BN` padded rows of halves.
        enc.set_threadgroup_memory_length(0, 2 * BN * SLAB_ROW * 2);
        enc.dispatch_thread_groups(
            MTLSize { width: n.div_ceil(BN), height: m.div_ceil(BM), depth: 1 },
            // Apple GPUs run 32 threads to a SIMD group.
            shape(SIMD_GROUPS),
        );
        Ok((MetalStorage::new(out, dev.clone(), m * n, self.out), Shape::from((m, n))))
    }
}

// ---------------------------------------------------------------------------
// Dense
// ---------------------------------------------------------------------------

/// The dense kernel's tiles: rows of `C`, columns, SIMD groups. The probe
/// measured 128 × 128 over eight SIMD groups fastest at the video DiT's
/// shapes, and 64 × 64 over four close to the best at a prefill chunk's.
const DENSE_TILES: [(usize, usize, usize); 2] = [(64, 64, 4), (128, 128, 8)];

/// Rows below which candle's kernel is left to do it.
///
/// From `dense_crossover`, bf16, twice, at `[k, n]` of `[3584, 18944]`,
/// `[4096, 4096]` and `[3072, 3072]`. One row, a decode step, is candle's at
/// all three: this kernel runs it at 0.87–0.95×. Eight rows are already
/// this kernel's, at 1.22–1.62×, and from 16 rows it is 2.4–3.6×. Two to
/// seven were not measured, so eight is where it starts.
const DENSE_ROWS: usize = 8;

/// Rows from which the 128 × 128 tile is used instead of 64 × 64.
///
/// Up to 512 rows the small tile is as good or better at two of the three
/// shapes, since more, smaller tiles keep more of the GPU busy. From 1024 the
/// large tile ties there, is 16–19% faster at `[3072, 3072]`, and at 4096
/// rows is 13–21% faster at all three.
const BIG_TILE_ROWS: usize = 1024;

/// `x · w` on the matrix units, or `None` where candle should do it.
///
/// `x` is `[m, k]`, `w` is `[k, n]`, both f16 or both bf16. `None` means
/// any of these, and the caller runs `x.matmul(w)` as it always has:
/// - the device cannot run the kernel;
/// - the dtype is f32, which `matmul2d` takes but does not accelerate;
/// - `m` is below [`DENSE_ROWS`], a decode step above all, where a
///   matrix-vector kernel is the right tool and a matrix unit is not.
pub(crate) fn dense(x: &Tensor, w: &Tensor) -> candle_core::Result<Option<Tensor>> {
    if dense_declines(x, w, DENSE_ROWS)? {
        return Ok(None);
    }
    let tile = usize::from(x.dim(0)? >= BIG_TILE_ROWS);
    Ok(Some(dense_with(x, w, tile)?))
}

fn dense_declines(x: &Tensor, w: &Tensor, rows: usize) -> candle_core::Result<bool> {
    Ok(pipes(x.device()).is_none()
        || x.rank() != 2
        || w.rank() != 2
        || x.dtype() != w.dtype()
        || !matches!(x.dtype(), DType::F16 | DType::BF16)
        || !w.is_contiguous()
        || x.dim(0)? < rows)
}

/// `x · w + b` on the matrix units, the bias added to the f32 sums and the
/// answer rounded once, as PyTorch's `addmm` rounds it; or `None` where
/// [`dense`] would decline, except that any number of rows is taken. `b` is
/// `[n]`, in `x`'s dtype or any other.
///
/// Rounding the product and then adding the bias rounds twice, and in bf16
/// that is not nothing. In the DiffVAE decoder, whose every layer has a
/// bias, it cost 3.2 dB against the reference by the frames: 47.6 dB where
/// the reference's own bf16 is 50.8. The token-wide layers taken here won
/// back 1.2 of that, and the one-row products that make the step's
/// modulation the other 2.0: an error there is every token's. So a single
/// row is taken too, where [`dense`] leaves it to candle for speed; a layer
/// with a bias and one row is an embedding, not a decode step.
///
/// It is faster too: filling the answer with the bias's rows and adding the
/// product into it beat the product and candle's broadcast add by 1.65–7×
/// (`dense_bias_race`).
pub(crate) fn dense_bias(x: &Tensor, w: &Tensor, b: &Tensor) -> candle_core::Result<Option<Tensor>> {
    if dense_declines(x, w, 1)? || b.elem_count() != w.dim(1)? {
        return Ok(None);
    }
    let tile = usize::from(x.dim(0)? >= BIG_TILE_ROWS);
    Ok(Some(dense_bias_with(x, w, b, tile)?))
}

fn dense_bias_with(x: &Tensor, w: &Tensor, b: &Tensor, tile: usize) -> candle_core::Result<Tensor> {
    let (m, n) = (x.dim(0)?, w.dim(1)?);
    // The answer, starting as the bias's rows, and accumulated into. One
    // row of the bias is already contiguous, and `contiguous` would hand
    // back the bias's own storage to add the product into; so would `copy`,
    // whose Metal storage shares its buffer. `affine(1, 0)` writes a new
    // one, and is exact.
    let rows = b.to_dtype(x.dtype())?.reshape((1, n))?.broadcast_as((m, n))?;
    let c = if rows.is_contiguous() { rows.affine(1.0, 0.0)? } else { rows.contiguous()? };
    c.inplace_op3(&x.contiguous()?, w, &Dense { tile })?;
    Ok(c)
}

/// [`dense`] in a given tile, whatever the shape: for the tests, and for
/// measuring where the thresholds belong.
fn dense_with(x: &Tensor, w: &Tensor, tile: usize) -> candle_core::Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(w, &Dense { tile })
}

struct Dense {
    /// Which of [`DENSE_TILES`].
    tile: usize,
}

impl Dense {
    /// `a · b` into `out`, or with `acc`, `out + a · b`.
    fn run(&self, a: &MetalStorage, la: &Layout, b: &MetalStorage, lb: &Layout, out: &candle_metal_kernels::metal::Buffer, acc: bool)
     -> candle_core::Result<()> {
        let (m, k) = la.shape().dims2()?;
        let (_, n) = lb.shape().dims2()?;
        let dt = a.dtype();
        let dev = a.device();
        let Some(pipes) = pipes(&Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_dense: this device cannot run it");
        };
        let (tm, tn, sg) = DENSE_TILES[self.tile];
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_dense");
        enc.set_compute_pipeline_state(&pipes.dense[usize::from(acc)][usize::from(dt == DType::BF16)][self.tile]);
        enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * dt.size_in_bytes());
        enc.set_input_buffer(1, Some(b.buffer()), lb.start_offset() * dt.size_in_bytes());
        enc.set_output_buffer(2, Some(out), 0);
        enc.set_bytes(3, &(m as i32));
        enc.set_bytes(4, &(n as i32));
        enc.set_bytes(5, &(k as i32));
        enc.dispatch_thread_groups(
            MTLSize { width: n.div_ceil(tn), height: m.div_ceil(tm), depth: 1 },
            shape(sg),
        );
        Ok(())
    }

    /// `[m, n]` for `a · b`, if the two are what the kernel reads.
    fn shape_of(a: &MetalStorage, la: &Layout, b: &MetalStorage, lb: &Layout) -> candle_core::Result<(usize, usize)> {
        let (m, k) = la.shape().dims2()?;
        let (kb, n) = lb.shape().dims2()?;
        let dt = a.dtype();
        if !matches!(dt, DType::F16 | DType::BF16) || k != kb || b.dtype() != dt || !la.is_contiguous() || !lb.is_contiguous() {
            candle_core::bail!("mpp_dense: [{m}, {k}] x [{kb}, {n}], {dt:?} x {:?}", b.dtype());
        }
        Ok((m, n))
    }
}

impl CustomOp2 for Dense {
    fn name(&self) -> &'static str {
        "mpp_dense"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_dense runs on Metal only")
    }

    fn metal_fwd(&self, a: &MetalStorage, la: &Layout, b: &MetalStorage, lb: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        let (m, n) = Dense::shape_of(a, la, b, lb)?;
        let dt = a.dtype();
        let dev = a.device();
        let bytes = m * n * dt.size_in_bytes();
        let out = dev.allocate_buffer(bytes)?;
        // As for `Blocks`: under test every output starts as NaN.
        #[cfg(test)]
        {
            let mut blit = dev.blit_command_encoder()?;
            blit.fill_buffer(&out, (0, bytes), 0xff);
        }
        self.run(a, la, b, lb, &out, false)?;
        Ok((MetalStorage::new(out, dev.clone(), m * n, dt), Shape::from((m, n))))
    }
}

/// `c += a · b`, in place: [`dense_bias`], `c` holding the bias's rows.
impl candle_core::InplaceOp3 for Dense {
    fn name(&self) -> &'static str {
        "mpp_dense_acc"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<()> {
        candle_core::bail!("mpp_dense runs on Metal only")
    }

    fn metal_fwd(&self, c: &mut MetalStorage, lc: &Layout, a: &MetalStorage, la: &Layout, b: &MetalStorage, lb: &Layout)
     -> candle_core::Result<()> {
        let (m, n) = Dense::shape_of(a, la, b, lb)?;
        if lc.dims() != [m, n] || c.dtype() != a.dtype() || !lc.is_contiguous() || lc.start_offset() != 0 {
            candle_core::bail!("mpp_dense: accumulating [{m}, {n}] {:?} into {lc:?} {:?}", a.dtype(), c.dtype());
        }
        self.run(a, la, b, lb, c.buffer(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::Module;

    fn rand(n: usize, seed: f32) -> Tensor {
        let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 12.9898 + seed).sin() * 43758.547).fract() - 0.5).collect();
        Tensor::from_vec(v, n, &Device::Cpu).unwrap()
    }

    /// Where the GPU has matrix units, the kernels build. Every other test
    /// here skips where [`available`] is false, so a source that does not
    /// compile would otherwise pass them all.
    #[test]
    fn builds_where_there_are_matrix_units() {
        let Ok(Device::Metal(md)) = Device::new_metal(0) else { return };
        let off = matches!(std::env::var("KVAD_GPU_MPP").as_deref(), Ok("0") | Ok("false"));
        if md.metal_device().as_ref().supportsFamily(MTLGPUFamily::Apple10) && !off {
            assert!(available(&Device::Metal(md)), "the M5 matmul kernels did not build");
        }
    }

    /// candle's `QMatMul` and this kernel, on the same blocks, agree to f16
    /// rounding: whole tiles, and every kind of ragged edge, including the
    /// one row of a decode step.
    #[test]
    fn agrees_with_candle_on_the_same_blocks_at_every_edge() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device");
            return;
        };
        if !available(&dev) {
            eprintln!("this GPU has no matrix units for matmul2d; nothing to test");
            return;
        }
        // (m, k, n): whole tiles; a ragged m (a 77-token prompt); a ragged n;
        // both; a narrow k; and one row.
        for (m, k, n) in [(128, 256, 128), (77, 512, 192), (128, 256, 100), (300, 1024, 70), (65, 32, 65), (1, 256, 64)] {
            let w = rand(n * k, n as f32).reshape((n, k)).unwrap();
            let blocks = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap().data().unwrap().into_owned();
            let ours = Blocks::new(GgmlDType::Q8_0, &blocks, n, k, &dev).unwrap();
            let theirs = QMatMul::from_qtensor(QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()).unwrap();
            let x = (rand(m * k, 7.0 + m as f32).reshape((m, k)).unwrap() * 4.0).unwrap().to_device(&dev).unwrap();

            let got = ours.forward(&x).unwrap();
            let want = theirs.forward(&x).unwrap();
            assert_eq!(got.dims(), &[m, n]);
            // Every element written. A max does not show this: candle's
            // `max_all` compares with `>`, which a NaN never wins, so an
            // output left entirely poisoned has a max difference of nothing
            // at all. A sum carries a NaN through.
            let sum = got.sum_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(sum.is_finite(), "[{m}, {k}] x [{n}, {k}]: output not all written");
            let scale = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            let apart = (got - &want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(apart <= 2e-3 * scale, "[{m}, {k}] x [{n}, {k}]: {apart} apart at scale {scale}");
        }
    }

    /// Q4_K, Q5_K and Q6_K, against the same blocks dequantised by candle
    /// and multiplied in f32 on the CPU: whole tiles, ragged edges, one
    /// row, and a `k` of one super-block. What is left is the kernel's
    /// rounding of weights and activations to f16, and it must be near
    /// f16's own: 60 dB is a thousandth of the signal's amplitude.
    #[test]
    fn k_quants_agree_with_their_own_numbers() {
        let Ok(dev) = Device::new_metal(0) else { return };
        if !available(&dev) {
            return;
        }
        for kind in [GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
            for (m, k, n) in [(128, 256, 128), (77, 512, 192), (300, 1024, 70), (1, 256, 64), (65, 2048, 33)] {
                let w = rand(n * k, n as f32 + k as f32).reshape((n, k)).unwrap();
                let q = QTensor::quantize(&w, kind).unwrap();
                let ours = Blocks::new(kind, &q.data().unwrap(), n, k, &dev).unwrap();
                // f32 in, as the image pipelines give it, and bf16, as
                // LTX-2.5's DiT does: each against its own input in f32.
                for dt in [DType::F32, DType::BF16] {
                    let x = (rand(m * k, 7.0 + m as f32).reshape((m, k)).unwrap() * 4.0).unwrap().to_dtype(dt).unwrap();
                    let want = x.to_dtype(DType::F32).unwrap().matmul(&q.dequantize(&Device::Cpu).unwrap().t().unwrap()).unwrap();
                    let got = ours.forward(&x.to_device(&dev).unwrap()).unwrap().to_device(&Device::Cpu).unwrap();
                    assert_eq!(got.dims(), &[m, n]);
                    let sum = got.sum_all().unwrap().to_scalar::<f32>().unwrap();
                    assert!(sum.is_finite(), "{kind:?} {dt:?} [{m}, {k}] x [{n}, {k}]: output not all written");
                    let err = (&got - &want).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                    let sig = want.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                    let db = 10.0 * (sig / err.max(1e-30)).log10();
                    assert!(db > 60.0, "{kind:?} {dt:?} [{m}, {k}] x [{n}, {k}]: {db:.1} dB");
                }
            }
        }
    }

    /// The store's bias, GELU and rounding, against the same product in
    /// f32 with the ops written out after it.
    #[test]
    fn epilogue_agrees_with_the_ops_after_it() {
        let Ok(dev) = Device::new_metal(0) else { return };
        if !available(&dev) {
            return;
        }
        for (m, k, n) in [(128, 256, 128), (77, 512, 192), (300, 1024, 70), (1, 64, 64)] {
            let w = rand(n * k, n as f32).reshape((n, k)).unwrap();
            let blocks = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap().data().unwrap().into_owned();
            let q = Blocks::new(GgmlDType::Q8_0, &blocks, n, k, &dev).unwrap();
            let x = (rand(m * k, 5.0 + m as f32).reshape((m, k)).unwrap() * 4.0).unwrap().to_device(&dev).unwrap();
            let b = (rand(n, 9.0) * 8.0).unwrap().to_dtype(DType::BF16).unwrap().to_device(&dev).unwrap();
            let plain = q.forward(&x).unwrap();
            for out in [DType::F32, DType::F16, DType::BF16] {
                for (bias, gelu) in [(true, false), (true, true), (false, true)] {
                    let got = q.linear(&x, bias.then_some(&b), out, gelu).unwrap();
                    assert_eq!(got.dtype(), out);
                    let mut want = plain.clone();
                    if bias {
                        want = want.broadcast_add(&b.to_dtype(DType::F32).unwrap()).unwrap();
                    }
                    if gelu {
                        want = want.gelu().unwrap();
                    }
                    let got = got.to_dtype(DType::F32).unwrap();
                    let sum = got.sum_all().unwrap().to_scalar::<f32>().unwrap();
                    assert!(sum.is_finite(), "output not all written");
                    let scale = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
                    let apart = (got - &want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
                    let tol = match out {
                        DType::F32 => 1e-5,
                        DType::F16 => 1e-3,
                        _ => 8e-3,
                    };
                    assert!(apart <= tol * scale, "{out:?} bias {bias} gelu {gelu} [{m}, {k}] x [{n}, {k}]: {apart} at {scale}");
                }
            }
        }
    }

    /// Leading axes pass through, as they do for `QMatMul`.
    #[test]
    fn keeps_leading_axes() {
        let Ok(dev) = Device::new_metal(0) else { return };
        if !available(&dev) {
            return;
        }
        let (n, k) = (64, 64);
        let w = rand(n * k, 3.0).reshape((n, k)).unwrap();
        let blocks = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap().data().unwrap().into_owned();
        let q = Blocks::new(GgmlDType::Q8_0, &blocks, n, k, &dev).unwrap();
        let x = rand(2 * 5 * k, 1.0).reshape((2, 5, k)).unwrap().to_device(&dev).unwrap();
        let flat = q.forward(&x.reshape((10, k)).unwrap()).unwrap();
        assert_eq!(q.forward(&x).unwrap().reshape((10, n)).unwrap().to_vec2::<f32>().unwrap(),
                   flat.to_vec2::<f32>().unwrap());
    }

    /// The dense kernel against candle's matmul on the same inputs, both
    /// dtypes, both tiles, at every kind of ragged edge.
    #[test]
    fn dense_agrees_with_candle_at_every_edge() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device");
            return;
        };
        if !available(&dev) {
            eprintln!("this GPU has no matrix units for matmul2d; nothing to test");
            return;
        }
        for dt in [DType::F16, DType::BF16] {
            // bf16 keeps 8 bits of mantissa to f16's 11: the two kernels'
            // roundings differ by that much more.
            let tol = if dt == DType::F16 { 2e-3 } else { 1.6e-2 };
            for tile in 0..DENSE_TILES.len() {
                for (m, k, n) in [(256, 128, 256), (77, 320, 192), (128, 64, 100), (300, 96, 70), (1, 64, 64), (129, 17, 130)] {
                    let x = (rand(m * k, m as f32).reshape((m, k)).unwrap() * 4.0).unwrap().to_dtype(dt).unwrap()
                        .to_device(&dev).unwrap();
                    let w = rand(k * n, n as f32 + 0.5).reshape((k, n)).unwrap().to_dtype(dt).unwrap()
                        .to_device(&dev).unwrap();
                    let got = dense_with(&x, &w, tile).unwrap().to_dtype(DType::F32).unwrap();
                    let want = x.matmul(&w).unwrap().to_dtype(DType::F32).unwrap();
                    assert_eq!(got.dims(), &[m, n]);
                    let sum = got.sum_all().unwrap().to_scalar::<f32>().unwrap();
                    assert!(sum.is_finite(), "{dt:?} tile {tile} [{m}, {k}] x [{k}, {n}]: output not all written");
                    let scale = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
                    let apart = (got - &want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
                    assert!(apart <= tol * scale, "{dt:?} tile {tile} [{m}, {k}] x [{k}, {n}]: {apart} apart at scale {scale}");
                }
            }
        }
    }

    /// With a bias, the answer is the f32 `x · w + b` rounded once: nearer
    /// to it than the product rounded, the bias added and rounded again, in
    /// both dtypes and tiles, and within one rounding of it everywhere.
    #[test]
    fn dense_bias_rounds_once() {
        let Ok(dev) = Device::new_metal(0) else { return };
        if !available(&dev) {
            return;
        }
        for dt in [DType::F16, DType::BF16] {
            for tile in 0..DENSE_TILES.len() {
                for (m, k, n) in [(256, 128, 256), (77, 320, 192), (300, 96, 70), (129, 32, 130), (1, 256, 384), (7, 64, 20)] {
                    let on = |t: Tensor| t.to_dtype(dt).unwrap().to_device(&dev).unwrap();
                    let x = on((rand(m * k, m as f32).reshape((m, k)).unwrap() * 4.0).unwrap());
                    let w = on(rand(k * n, n as f32 + 0.5).reshape((k, n)).unwrap());
                    let b = on((rand(n, 7.0) * 8.0).unwrap());
                    let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
                    let exact = f(&x).matmul(&f(&w)).unwrap().broadcast_add(&f(&b)).unwrap();
                    let before = f(&b).to_vec1::<f32>().unwrap();
                    let got = f(&dense_bias_with(&x, &w, &b, tile).unwrap());
                    // The bias is read, never written: with one row it once
                    // took the product itself.
                    assert_eq!(f(&b).to_vec1::<f32>().unwrap(), before, "{dt:?} [{m}, {k}]: the bias changed");
                    let twice = f(&dense_with(&x, &w, tile).unwrap().broadcast_add(&b).unwrap());
                    assert!(got.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "not all written");
                    let off = |t: &Tensor| (t - &exact).unwrap().abs().unwrap();
                    let mean = |t: &Tensor| off(t).mean_all().unwrap().to_scalar::<f32>().unwrap();
                    let (ours, theirs) = (mean(&got), mean(&twice));
                    let what = format!("{dt:?} tile {tile} [{m}, {k}] x [{k}, {n}]");
                    assert!(ours < theirs, "{what}: {ours} from the exact answer, rounding twice {theirs}");
                    // One rounding is at most half a unit in the last place;
                    // allow a whole one for the order the sums are taken in.
                    let ulp = if dt == DType::F16 { 1.0 / 1024.0 } else { 1.0 / 128.0 };
                    let most = off(&got).broadcast_div(&exact.abs().unwrap().maximum(1e-3).unwrap()).unwrap()
                        .max_all().unwrap().to_scalar::<f32>().unwrap();
                    assert!(most <= ulp, "{what}: {most} of its own size from the exact answer");
                }
            }
        }
    }

    /// Not a test, a measurement: [`dense_bias`] against the product and
    /// then the bias added, taking turns.
    ///
    /// Measured when written, bf16: 1.65× at `[4096, 3072] × [3072, 3072]`,
    /// 2.0× at the DiffVAE's stage-1 feed-forward, 6.6–7.1× at its stage 5's
    /// 256-wide projections, where candle's broadcast add took six times as
    /// long as the product.
    ///
    ///     cargo test --release -p kvad-gpu dense_bias_race -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dense_bias_race() {
        let dev = Device::new_metal(0).unwrap();
        assert!(available(&dev));
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        // (what, m, k, n)
        for (what, m, k, n) in [
            ("DiffVAE stage 5 w_gate", 262_144, 256, 1024),
            ("DiffVAE stage 5 qkv", 262_144, 256, 768),
            ("DiffVAE stage 1 w_up", 6912, 2048, 8192),
            ("square, 4096 rows", 4096, 3072, 3072),
        ] {
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let w = Tensor::randn(0f32, 0.05, (k, n), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let b = Tensor::randn(0f32, 1.0, n, &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let time = |f: &dyn Fn()| {
                dev.synchronize().unwrap();
                let t = std::time::Instant::now();
                for _ in 0..10 {
                    f();
                }
                dev.synchronize().unwrap();
                t.elapsed().as_secs_f64() / 10.0
            };
            let twice = || drop(dense(&x, &w).unwrap().unwrap().broadcast_add(&b).unwrap());
            let once = || drop(dense_bias(&x, &w, &b).unwrap().unwrap());
            time(&twice);
            time(&once);
            let (mut a, mut c) = (vec![], vec![]);
            for _ in 0..5 {
                a.push(time(&twice));
                c.push(time(&once));
            }
            let (a, c) = (median(a), median(c));
            let flops = 2.0 * (m * k * n) as f64;
            println!("{what:<24} then added {:8.3} ms {:5.1} TFLOP/s | in the sums {:8.3} ms {:5.1} TFLOP/s | {:4.2}x",
                     a * 1e3, flops / a / 1e12, c * 1e3, flops / c / 1e12, a / c);
        }
    }

    /// What [`dense`] leaves to candle: f32, a decode step, and mixed
    /// dtypes.
    #[test]
    fn dense_declines_what_it_should_not_take() {
        let Ok(dev) = Device::new_metal(0) else { return };
        if !available(&dev) {
            return;
        }
        let x = |m: usize, dt: DType| Tensor::zeros((m, 64), dt, &dev).unwrap();
        let w = |dt: DType| Tensor::zeros((64, 64), dt, &dev).unwrap();
        assert!(dense(&x(512, DType::F32), &w(DType::F32)).unwrap().is_none());
        assert!(dense(&x(1, DType::BF16), &w(DType::BF16)).unwrap().is_none());
        assert!(dense(&x(512, DType::BF16), &w(DType::F16)).unwrap().is_none());
        assert!(dense(&x(512, DType::BF16), &w(DType::BF16)).unwrap().is_some());
        let cpu = Tensor::zeros((512, 64), DType::BF16, &Device::Cpu).unwrap();
        assert!(dense(&cpu, &cpu.t().unwrap().contiguous().unwrap()).unwrap().is_none());
    }

    /// Where [`DENSE_ROWS`] and [`BIG_TILE_ROWS`] come from. Not a test, a
    /// measurement; run it with
    ///
    ///     cargo test --release -p kvad-gpu dense_crossover -- --ignored --nocapture
    ///
    /// Candle and each tile take turns round by round, and each ratio is
    /// the median of per-round ratios, for the reason `examples/neural_accel`
    /// gives: this machine's speed drifts between runs.
    #[test]
    #[ignore]
    fn dense_crossover() {
        let dev = Device::new_metal(0).unwrap();
        assert!(available(&dev));
        let once = |f: &dyn Fn() -> Tensor| {
            let t = std::time::Instant::now();
            for _ in 0..10 {
                let _ = f();
            }
            dev.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 100.0
        };
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        for (k, n) in [(3584, 18944), (4096, 4096), (3072, 3072)] {
            println!("\nbf16 [m, {k}] x [{k}, {n}]: candle ms, then each tile's speedup over it");
            let w = rand(k * n, 1.0).reshape((k, n)).unwrap().to_dtype(DType::BF16).unwrap().to_device(&dev).unwrap();
            for m in [1, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096] {
                let x = rand(m * k, 2.0).reshape((m, k)).unwrap().to_dtype(DType::BF16).unwrap().to_device(&dev).unwrap();
                let fs: [&dyn Fn() -> Tensor; 3] = [
                    &|| x.matmul(&w).unwrap(),
                    &|| dense_with(&x, &w, 0).unwrap(),
                    &|| dense_with(&x, &w, 1).unwrap(),
                ];
                for f in fs {
                    let _ = f();
                }
                dev.synchronize().unwrap();
                let (mut base, mut r0, mut r1) = (vec![], vec![], vec![]);
                for _ in 0..7 {
                    let b = once(fs[0]);
                    r0.push(b / once(fs[1]));
                    r1.push(b / once(fs[2]));
                    base.push(b);
                }
                println!("  m {m:5}: candle {:8.3} ms   64x64 {:5.2}x   128x128 {:5.2}x", median(base), median(r0), median(r1));
            }
        }
    }

    /// The kernel's rate for each kind of block at the LTX-2.5 DiT's stage-2
    /// shapes, bias and bf16 in and out, as its projections run, and
    /// candle's `QMatMul` on the same blocks, f32 in and out, for the
    /// k-quants it would otherwise run. Not a test, a measurement:
    ///
    ///     cargo test --release -p kvad-gpu q_rates -- --ignored --nocapture
    #[test]
    #[ignore]
    fn q_rates() {
        let dev = Device::new_metal(0).unwrap();
        assert!(available(&dev));
        let once = |f: &dyn Fn() -> Tensor| {
            let t = std::time::Instant::now();
            for _ in 0..4 {
                let _ = f();
            }
            dev.synchronize().unwrap();
            t.elapsed().as_secs_f64() / 4.0
        };
        let median = |f: &dyn Fn() -> Tensor| {
            let _ = once(f);
            let mut v: Vec<f64> = (0..7).map(|_| once(f)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[3]
        };
        let m = 24576;
        for (k, n) in [(4096, 4096), (4096, 16384), (16384, 4096), (4096, 2048), (2048, 4096), (4096, 32)] {
            let w = rand(n * k, 1.0).reshape((n, k)).unwrap();
            let b = rand(n, 3.0).to_dtype(DType::BF16).unwrap().to_device(&dev).unwrap();
            let x = rand(m * k, 2.0).reshape((m, k)).unwrap().to_dtype(DType::BF16).unwrap().to_device(&dev).unwrap();
            let x32 = x.to_dtype(DType::F32).unwrap();
            let flop = 2.0 * (m * k * n) as f64 / 1e12;
            let mut line = format!("[{m}, {k}] x [{k}, {n}]:");
            for kind in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
                let q = QTensor::quantize(&w, kind).unwrap();
                let ours = Blocks::new(kind, &q.data().unwrap(), n, k, &dev).unwrap();
                let f = || ours.linear(&x, Some(&b), DType::BF16, false).unwrap();
                line += &format!(" {kind:?} {:5.1}", flop / median(&f));
                if kind != GgmlDType::Q8_0 {
                    let theirs = QMatMul::from_qtensor(QTensor::quantize_onto(&w, kind, &dev).unwrap()).unwrap();
                    let g = || theirs.forward(&x32).unwrap();
                    line += &format!(" (candle {:4.1})", flop / median(&g));
                }
            }
            println!("{line} TFLOP/s");
        }
    }
}
