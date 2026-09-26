//! Matmuls on the M5 GPU's neural accelerators (#52): Q8_0 in [`Q8`], and
//! f16 or bf16 in [`dense`].
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
//! `matmul2d` cannot read GGML blocks, so the kernel needs the raw Q8_0
//! bytes. `QTensor` keeps its Metal buffer private, and candle offers no way
//! to build one around a buffer someone else holds. So a weight is either a
//! `QTensor` or [`Q8`], never both, unless it is kept twice.
//!
//! A language model needs `QTensor`: its decode step is one row, which is
//! candle's matrix-vector kernel's job and no matrix unit's. An image model
//! has no decode step. Every product it takes has hundreds or thousands of
//! rows: patches, pixels, or a prompt's tokens. So the image pipelines hold
//! their Q8_0 projections as [`Q8`] and nothing else, and the text models keep
//! `QMatMul`.
//!
//! # The kernel
//!
//! `C = A · Wᵀ`, where `A` is `[M, K]` in f16, `W` is `[N, K]` in Q8_0 blocks
//! of 32 weights along `K` (an f16 scale, then 32 `int8`s), and `C` is
//! `[M, N]` in f32, as `QMatMul` returns it. A threadgroup owns a `64 × 64`
//! tile of `C` and walks `K` 32 at a time. At each step:
//! 1. its threads unpack the `64 × 32` slab of `W` the tile needs into
//!    threadgroup memory as f16: `scale × int8`, rounded once;
//! 2. `matmul2d` multiplies the `64 × 32` slab of `A` by it, reading `A`
//!    straight from device memory, and adds into a cooperative tensor. That
//!    is the tile's running sum, held in the SIMD groups' registers until
//!    the end.
//!
//! There are two slabs, and step 1 for the next step overlaps step 2 for
//! this one. Each thread loads the next step's raw blocks into registers
//! before the multiply, and unpacks them into the other slab after it. That
//! needs one barrier a step instead of two, and the probe measured it
//! 1.02–1.09× faster than one slab on every shape it tried, with the same
//! output bit for bit.
//!
//! Edges need no code of their own. `slice` clips a tensor to its extents,
//! and `matmul2d` and the final store both honour that, so a tile hanging
//! off the bottom of `A` or the right of `C` reads and writes only what
//! exists. Rows of `W` past `N` are unpacked as zeros. `K` is always a
//! multiple of 32, because a Q8_0 block is 32 wide.
//!
//! # Where it runs
//!
//! Only on a GPU of Apple's tenth family, the M5's, because that is where
//! the matrix units are. Before that, `matmul2d` runs on the ALUs, and there
//! it is not measured. It must also compile: the tensor API needs Metal 4,
//! which is macOS 26. `KVAD_GPU_MPP=0` turns it off, for measuring what it is
//! worth.
//!
//! The activations must be f16. `matmul2d` also takes f32 on the left, and
//! returns candle's product bit for bit at candle's speed: the f32 path is
//! not accelerated.
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
use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLSize};
use std::sync::OnceLock;

/// Bytes in one Q8_0 block: an f16 scale and 32 `int8`s.
const BLOCK: usize = 34;

/// Rows of `C` per threadgroup, columns, the step along `K`, and the SIMD
/// groups sharing the tile. `64 × 64 × 32` over four SIMD groups was the
/// fastest of the probe's five tiles at the video DiT's shapes, and within a
/// few percent of the best at a prefill chunk's.
const BM: usize = 64;
const BN: usize = 64;
const SIMD_GROUPS: usize = 4;

const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BM = 64;
constant constexpr int BN = 64;
constant constexpr int BK = 32;
constant constexpr int NSG = 4;

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

// Each thread's share of a slab: `PER` runs of 8 weights. Adjacent threads
// take adjacent runs of the same row, so a SIMD group reads a row
// contiguously.
constant constexpr int RUN = 8;
constant constexpr int PER = BN * BK / RUN / (32 * NSG);
static_assert(PER * RUN * 32 * NSG == BN * BK, "every thread unpacks the same number of runs");

// A step's raw blocks, from device memory into this thread's registers.
// Rows past `N` read as zeros. The `int8`s sit two bytes into a 34-byte
// block, so only a packed (byte-aligned) load may read them.
inline void fetch(device const block_q8_0 *w, int N, int n0, int blocks, int k, ushort tid,
                  thread half *d, thread char4 *lo, thread char4 *hi) {
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int r = tid + p * 32 * NSG;
        const int n = r / (BK / RUN);
        const int j = (r % (BK / RUN)) * RUN;
        if (n0 + n < N) {
            device const block_q8_0 &b = w[(n0 + n) * blocks + k / 32];
            device const packed_char4 *q = (device const packed_char4 *)(b.qs + j);
            d[p] = b.d;
            lo[p] = char4(q[0]);
            hi[p] = char4(q[1]);
        } else {
            d[p] = 0;
            lo[p] = char4(0);
            hi[p] = char4(0);
        }
    }
}

// Those registers, as f16 weights, into a slab: `scale × int8`, rounded once.
inline void unpack(threadgroup half *slab, ushort tid,
                   thread half *d, thread char4 *lo, thread char4 *hi) {
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int r = tid + p * 32 * NSG;
        const int n = r / (BK / RUN);
        const int j = (r % (BK / RUN)) * RUN;
        threadgroup half4 *dst = (threadgroup half4 *)(slab + n * BK + j);
        dst[0] = d[p] * half4(lo[p]);
        dst[1] = d[p] * half4(hi[p]);
    }
}

kernel void mm_q8_0(device half *a [[buffer(0)]],
                    device const block_q8_0 *w [[buffer(1)]],
                    device float *c [[buffer(2)]],
                    constant int &M [[buffer(3)]],
                    constant int &N [[buffer(4)]],
                    constant int &K [[buffer(5)]],
                    threadgroup half *slab [[threadgroup(0)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> ta(a, dextents<int32_t, 2>(K, M));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tc(c, dextents<int32_t, 2>(N, M));
    // Two slabs: step `s` multiplies from slab `s % 2` while the next step's
    // weights are unpacked into the other.
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw0(slab, dextents<int32_t, 2>(BK, BN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw1(slab + BN * BK, dextents<int32_t, 2>(BK, BN));

    constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto ma = ta.slice(0, tg.y * BM);
    auto acc = op.template get_destination_cooperative_tensor<decltype(ma), decltype(tw0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }

    const int blocks = K / 32;
    const int n0 = tg.x * BN;
    half d[PER];
    char4 lo[PER], hi[PER];

    fetch(w, N, n0, blocks, 0, tid, d, lo, hi);
    unpack(slab, tid, d, lo, hi);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int s = 0;
    for (int k = 0; k < K; k += BK, ++s) {
        const bool more = k + BK < K;
        // The next step's device reads go out before this step's multiply,
        // so they are in flight while it runs.
        if (more) {
            fetch(w, N, n0, blocks, k + BK, tid, d, lo, hi);
        }
        // The slice runs to the end of `K`; the descriptor's `BK` bounds what
        // this step reads. (The header's comments show a `static_slice` for
        // this, which the compiler does not have.) `s` is the same in every
        // thread, so the whole threadgroup takes the same branch, as
        // `matmul2d` requires.
        auto sa = ta.slice(k, tg.y * BM);
        if (s & 1) {
            op.run(sa, tw1, acc);
        } else {
            op.run(sa, tw0, acc);
        }
        if (more) {
            unpack(slab + ((s + 1) & 1) * BN * BK, tid, d, lo, hi);
        }
        // One barrier does both jobs. The slab written just now is complete
        // before anyone multiplies from it. And the slab the step after
        // next will overwrite has been read by everyone: that was this
        // step's multiply. One slab needed a barrier on each side.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mc = tc.slice(n0, tg.y * BM);
    acc.store(mc);
}

// Dense: `C = A · B`, all three row-major and of one dtype. `A` is
// `[M, K]`, `B` is `[K, N]` (a weight stored `[in, out]`, as `Proj::Dense`
// keeps it), and `C` is `[M, N]`. `matmul2d` reads both straight from device
// memory and walks all of `K` itself; the kernel only says which tile is
// whose. Measured against feeding `K` in steps, or staging `B` in
// threadgroup memory, this is the fastest there is.
template <typename T, int TM, int TN, int TSG>
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
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<TSG>> op;
    auto ma = ta.slice(0, tg.y * TM);
    auto mb = tb.slice(tg.x * TN, 0);
    auto mc = tc.slice(tg.x * TN, tg.y * TM);
    op.run(ma, mb, mc);
}

#define DENSE(T, tn, TM, TN, TSG) \
    template [[host_name("mm_dense_" #tn "_" #TM "x" #TN)]] [[kernel]] \
    decltype(mm_dense<T, TM, TN, TSG>) mm_dense<T, TM, TN, TSG>;
DENSE(half, f16, 64, 64, 4)
DENSE(half, f16, 128, 128, 8)
DENSE(bfloat, bf16, 64, 64, 4)
DENSE(bfloat, bf16, 128, 128, 8)
"#;

/// Whether this device runs [`Q8`] and [`dense`], answered once per process.
///
/// Compiling is part of the answer. A GPU of the right family on a macOS
/// without Metal 4 fails here, and says why on stderr, instead of failing at
/// the first matmul of a generation.
pub(crate) fn available(device: &Device) -> bool {
    pipes(device).is_some()
}

/// Every kernel in [`SOURCE`], built.
struct Pipes {
    q8: ComputePipeline,
    /// `[f16, bf16]`, each in [`DENSE_TILES`]' order.
    dense: [[ComputePipeline; 2]; 2],
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
            Ok(Pipes { q8: pipe("mm_q8_0")?, dense: [dense("f16")?, dense("bf16")?] })
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

/// One `[n, k]` weight matrix, in Q8_0 blocks on the GPU, for [`SOURCE`].
pub(crate) struct Q8 {
    blocks: Tensor,
    n: usize,
    k: usize,
}

impl Q8 {
    /// `blocks` are `[n, k]` as GGML lays Q8_0 out: row after row, each row
    /// `k / 32` blocks.
    pub(crate) fn new(blocks: &[u8], n: usize, k: usize, device: &Device) -> candle_core::Result<Self> {
        if k % 32 != 0 || blocks.len() != n * k / 32 * BLOCK {
            candle_core::bail!("{} bytes are not a [{n}, {k}] matrix of Q8_0 blocks", blocks.len());
        }
        Ok(Q8 { blocks: Tensor::from_slice(blocks, blocks.len(), device)?, n, k })
    }

    /// `x · Wᵀ` over the last axis, in f32, whatever `x`'s dtype and leading
    /// axes.
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dims = x.dims().to_vec();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((rows, self.k))?.to_dtype(DType::F16)?.contiguous()?;
        let y = x2.apply_op2_no_bwd(&self.blocks, self)?;
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

impl CustomOp2 for Q8 {
    fn name(&self) -> &'static str {
        "mpp_q8_0"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_q8_0 runs on Metal only")
    }

    fn metal_fwd(&self, a: &MetalStorage, la: &Layout, w: &MetalStorage, lw: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        let (m, k) = la.shape().dims2()?;
        let n = self.n;
        if k != self.k || a.dtype() != DType::F16 || !la.is_contiguous() || lw.start_offset() != 0 {
            candle_core::bail!("mpp_q8_0: wants contiguous f16 [m, {}], got {:?} {:?}", self.k, a.dtype(), la);
        }
        let dev = a.device();
        let Some(pipes) = pipes(&Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_q8_0: this device cannot run it");
        };
        let out = dev.allocate_buffer(m * n * 4)?;
        // A pooled buffer can still hold the last product of the same size,
        // so a kernel that wrote nothing would pass a test by agreeing with
        // it. Under test, every output starts as NaN.
        #[cfg(test)]
        {
            let mut blit = dev.blit_command_encoder()?;
            blit.fill_buffer(&out, (0, m * n * 4), 0xff);
        }
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_q8_0");
        enc.set_compute_pipeline_state(&pipes.q8);
        enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * 2);
        enc.set_input_buffer(1, Some(w.buffer()), 0);
        enc.set_output_buffer(2, Some(&out), 0);
        enc.set_bytes(3, &(m as i32));
        enc.set_bytes(4, &(n as i32));
        enc.set_bytes(5, &(k as i32));
        // Two slabs of `BN × 32` halves.
        enc.set_threadgroup_memory_length(0, 2 * BN * 32 * 2);
        enc.dispatch_thread_groups(
            MTLSize { width: n.div_ceil(BN), height: m.div_ceil(BM), depth: 1 },
            // Apple GPUs run 32 threads to a SIMD group.
            shape(SIMD_GROUPS),
        );
        Ok((MetalStorage::new(out, dev.clone(), m * n, DType::F32), Shape::from((m, n))))
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
    if pipes(x.device()).is_none()
        || x.rank() != 2
        || w.rank() != 2
        || x.dtype() != w.dtype()
        || !matches!(x.dtype(), DType::F16 | DType::BF16)
        || !w.is_contiguous()
        || x.dim(0)? < DENSE_ROWS
    {
        return Ok(None);
    }
    let tile = usize::from(x.dim(0)? >= BIG_TILE_ROWS);
    Ok(Some(dense_with(x, w, tile)?))
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
        let (m, k) = la.shape().dims2()?;
        let (kb, n) = lb.shape().dims2()?;
        let dt = a.dtype();
        let which = match dt {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => candle_core::bail!("mpp_dense: {dt:?} is not f16 or bf16"),
        };
        if k != kb || b.dtype() != dt || !la.is_contiguous() || !lb.is_contiguous() {
            candle_core::bail!("mpp_dense: [{m}, {k}] x [{kb}, {n}], {dt:?} x {:?}", b.dtype());
        }
        let dev = a.device();
        let Some(pipes) = pipes(&Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_dense: this device cannot run it");
        };
        let (tm, tn, sg) = DENSE_TILES[self.tile];
        let bytes = m * n * dt.size_in_bytes();
        let out = dev.allocate_buffer(bytes)?;
        // As for `Q8`: under test every output starts as NaN.
        #[cfg(test)]
        {
            let mut blit = dev.blit_command_encoder()?;
            blit.fill_buffer(&out, (0, bytes), 0xff);
        }
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_dense");
        enc.set_compute_pipeline_state(&pipes.dense[which][self.tile]);
        enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * dt.size_in_bytes());
        enc.set_input_buffer(1, Some(b.buffer()), lb.start_offset() * dt.size_in_bytes());
        enc.set_output_buffer(2, Some(&out), 0);
        enc.set_bytes(3, &(m as i32));
        enc.set_bytes(4, &(n as i32));
        enc.set_bytes(5, &(k as i32));
        enc.dispatch_thread_groups(
            MTLSize { width: n.div_ceil(tn), height: m.div_ceil(tm), depth: 1 },
            shape(sg),
        );
        Ok((MetalStorage::new(out, dev.clone(), m * n, dt), Shape::from((m, n))))
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
            let ours = Q8::new(&blocks, n, k, &dev).unwrap();
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
        let q = Q8::new(&blocks, n, k, &dev).unwrap();
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

}
