//! Does a matmul on the M5 GPU's neural accelerators beat candle's? (#52)
//!
//! Every M5 GPU core carries a matrix unit, and a shader reaches it only
//! through Metal 4's tensor API: `mpp::tensor_ops::matmul2d`. candle's
//! kernels predate that API. They multiply with `simdgroup_matrix`, which runs
//! on the ordinary ALUs exactly as it did on an M4, so today kvad leaves the
//! units idle. This probe writes the smallest kernel that uses them, runs it
//! on candle's own device and queue as a `CustomOp2`, and times it against
//! what `Proj::forward` runs now.
//!
//! It asks twice. `dense` runs f16 and bf16 weights, which is `Proj::Dense`.
//! `q8` runs GGML's Q8_0 blocks, which is `Proj::Quant` and what the image
//! and video models load. `matmul2d` cannot read those blocks, so that kernel
//! unpacks each slab of weights to f16 in threadgroup memory first.
//!
//! The question is narrow on purpose: is there a gain at the shapes that
//! matter, before anything is built around it? The shapes are a video DiT's
//! (LTX-2.5, #51) and one prefill chunk of a 7B model. Decode is left out: at
//! one row a matmul is bound by memory, and no matrix unit changes that.
//!
//!     cargo run --release -p kvad-gpu --example neural_accel [dense|q8]
//!
//! # Reading the numbers
//!
//! - Each timing is the median of several rounds. A round runs the op a few
//!   times behind **one** device sync, never a sync per repeat, for the reason
//!   `prefill_cost` gives: a sync hands candle's pooled buffers back and the
//!   next repeat pays to allocate its output afresh.
//! - candle and the candidate take turns, round by round, and the speedup is
//!   the median of the per-round ratios. This machine's speed drifts by a
//!   third from one run to the next, and timing them apart put that drift in
//!   the ratio.
//! - `err` is the largest absolute difference from an f32 product of the same
//!   inputs, over the largest absolute value of that product. Compare it with
//!   candle's own kernel at the same dtype, not with zero: both round.
//! - A failed Metal command buffer makes candle hand back zeros without a
//!   word, which would time beautifully. A kernel whose error is not small
//!   is reported as wrong, whatever its speed.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("neural_accel needs Metal, and so macOS.");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    probe::main()
}

#[cfg(target_os = "macos")]
mod probe {
    use candle_core::backend::BackendStorage;
    use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
    use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, MetalStorage, Module, Shape, Tensor};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
    use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};

    /// One kernel, instantiated per element type and tile. `C = A · B`, all
    /// three row-major: `A` is `[M, K]`, `B` is `[K, N]`, `C` is `[M, N]`.
    /// That is `Proj::Dense`'s layout, with the weight stored `[in, out]`.
    ///
    /// A threadgroup owns one `BM × BN` tile of `C` and hands the whole
    /// reduction over `K` to `matmul2d`. How the op walks `K`, and what it
    /// keeps in registers, is Apple's to decide: that is the point of it.
    ///
    /// Two things read backwards if you are used to `[rows, cols]`:
    /// - A tensor's extents run innermost first. `A`'s are `(K, M)`: `K`
    ///   columns, each row contiguous, `M` rows.
    /// - So `slice(x, y)` takes a column offset, then a row offset.
    const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename T, int BM, int BN, int NSG>
kernel void mm(device T *a [[buffer(0)]],
               device T *b [[buffer(1)]],
               device T *c [[buffer(2)]],
               constant int &M [[buffer(3)]],
               constant int &N [[buffer(4)]],
               constant int &K [[buffer(5)]],
               uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device T, dextents<int32_t, 2>, tensor_inline> ta(a, dextents<int32_t, 2>(K, M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tb(b, dextents<int32_t, 2>(N, K));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tc(c, dextents<int32_t, 2>(N, M));

    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto ma = ta.slice(0, tg.y * BM);
    auto mb = tb.slice(tg.x * BN, 0);
    auto mc = tc.slice(tg.x * BN, tg.y * BM);
    op.run(ma, mb, mc);
}

#define INST(T, tn, BM, BN, NSG) \
    template [[host_name("mm_" #tn "_" #BM "x" #BN "_" #NSG)]] [[kernel]] \
    decltype(mm<T, BM, BN, NSG>) mm<T, BM, BN, NSG>;

#define TILES(T, tn) \
    INST(T, tn, 64, 32, 4) \
    INST(T, tn, 64, 64, 4) \
    INST(T, tn, 128, 64, 4) \
    INST(T, tn, 128, 128, 4) \
    INST(T, tn, 128, 128, 8)

TILES(half, f16)
TILES(bfloat, bf16)
"#;

    /// The tiles `SOURCE` instantiates: rows of `C` per threadgroup, columns,
    /// and SIMD groups sharing the work.
    const TILES: [(usize, usize, usize); 5] =
        [(64, 32, 4), (64, 64, 4), (128, 64, 4), (128, 128, 4), (128, 128, 8)];

    /// The kernel as a candle op, so it runs on candle's command queue and
    /// its output is an ordinary `Tensor`.
    #[derive(Clone)]
    struct Mpp {
        pipe: ComputePipeline,
        tile: (usize, usize, usize),
        poison: bool,
    }

    impl CustomOp2 for Mpp {
        fn name(&self) -> &'static str {
            "mpp_matmul2d"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
         -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("mpp_matmul2d runs on Metal only")
        }

        fn metal_fwd(&self, a: &MetalStorage, la: &Layout, b: &MetalStorage, lb: &Layout)
         -> candle_core::Result<(MetalStorage, Shape)> {
            let (m, k) = la.shape().dims2()?;
            let (kb, n) = lb.shape().dims2()?;
            let (bm, bn, nsg) = self.tile;
            // A probe, not a kernel for every shape: whole tiles and packed
            // rows only. `slice` would clip a ragged edge, but that is not
            // what is being measured.
            if k != kb || m % bm != 0 || n % bn != 0 || !la.is_contiguous() || !lb.is_contiguous() {
                candle_core::bail!("mpp_matmul2d: [{m}, {k}] x [{kb}, {n}] does not fit tile {bm}x{bn}");
            }
            let dev = a.device();
            let dt = a.dtype();
            let out = output(dev, m * n * dt.size_in_bytes(), self.poison)?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_compute_pipeline_state(&self.pipe);
            enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * dt.size_in_bytes());
            enc.set_input_buffer(1, Some(b.buffer()), lb.start_offset() * dt.size_in_bytes());
            enc.set_output_buffer(2, Some(&out), 0);
            enc.set_bytes(3, &(m as i32));
            enc.set_bytes(4, &(n as i32));
            enc.set_bytes(5, &(k as i32));
            // Apple GPUs run 32 threads to a SIMD group.
            enc.dispatch_thread_groups(
                MTLSize { width: n / bn, height: m / bm, depth: 1 },
                MTLSize { width: 32 * nsg, height: 1, depth: 1 },
            );
            Ok((MetalStorage::new(out, dev.clone(), m * n, dt), Shape::from((m, n))))
        }
    }

    /// The same product with the weight in GGML's Q8_0, which is what a q8
    /// model's `Proj::Quant` holds: `C = A · Wᵀ`, `A` `[M, K]` in f32 or f16,
    /// `W` `[N, K]` in blocks of 32 weights along `K`, each block one f16
    /// scale and 32 `int8`s. `C` is f32, as `QMatMul` returns it.
    ///
    /// `matmul2d` does not read GGML blocks, so each threadgroup walks `K` in
    /// steps of `BK` and, at every step:
    /// 1. all its threads unpack the `BN × BK` slab of `W` it needs into
    ///    threadgroup memory as f16, `scale × int8`, which is exact enough:
    ///    the product of an f16 and a small integer, rounded once;
    /// 2. `matmul2d` multiplies the `BM × BK` slab of `A`, read straight from
    ///    device memory, by that slab, and adds into a cooperative tensor:
    ///    the running `BM × BN` sum, held in the SIMD groups' registers across
    ///    all of `K` and written out once at the end.
    ///
    /// The slab is stored as `W` is, a row per output column, so the right
    /// operand is transposed: extents `(BK, BN)`, `transpose_right`.
    const SOURCE_Q8: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

template <typename TA, int BM, int BN, int BK, int NSG>
kernel void mm_q8(device TA *a [[buffer(0)]],
                  device const block_q8_0 *w [[buffer(1)]],
                  device float *c [[buffer(2)]],
                  constant int &M [[buffer(3)]],
                  constant int &N [[buffer(4)]],
                  constant int &K [[buffer(5)]],
                  threadgroup half *slab [[threadgroup(0)]],
                  uint2 tg [[threadgroup_position_in_grid]],
                  ushort tid [[thread_index_in_threadgroup]]) {
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> ta(a, dextents<int32_t, 2>(K, M));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tc(c, dextents<int32_t, 2>(N, M));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tw(slab, dextents<int32_t, 2>(BK, BN));

    constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto ma = ta.slice(0, tg.y * BM);
    auto acc = op.template get_destination_cooperative_tensor<decltype(ma), decltype(tw), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }

    // Each thread unpacks runs of 8 weights, adjacent threads adjacent runs
    // of the same row, so a SIMD group's reads of a row are contiguous.
    constexpr int RUN = 8;
    constexpr int RUNS = BN * BK / RUN;
    const int blocks = K / 32;
    device const block_q8_0 *wrow = w + tg.x * BN * blocks;
    for (int k = 0; k < K; k += BK) {
        for (int r = tid; r < RUNS; r += 32 * NSG) {
            const int n = r / (BK / RUN);
            const int j = (r % (BK / RUN)) * RUN;
            device const block_q8_0 &b = wrow[n * blocks + (k + j) / 32];
            const half d = b.d;
            const int o = j % 32;
            // Four at a time. The `int8`s sit two bytes into a 34-byte block,
            // so only a packed (byte-aligned) load may read them; the slab
            // is the kernel's own and aligned, so a plain `half4` store is.
            device const packed_char4 *q = (device const packed_char4 *)(b.qs + o);
            threadgroup half4 *dst = (threadgroup half4 *)(slab + n * BK + j);
            #pragma unroll
            for (int i = 0; i < RUN / 4; ++i) {
                dst[i] = d * half4(char4(q[i]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The slice runs to the end of `K`; the descriptor's `BK` is what
        // says how much of it this step reads. (The header's comments show
        // a `static_slice` for this. The compiler has no such member.)
        auto sa = ta.slice(k, tg.y * BM);
        op.run(sa, tw, acc);
        // The next step overwrites the slab: nobody may still be reading it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mc = tc.slice(tg.x * BN, tg.y * BM);
    acc.store(mc);
}

#define INST(TA, an, BM, BN, BK, NSG) \
    template [[host_name("mm_q8_" #an "_" #BM "x" #BN "x" #BK "_" #NSG)]] [[kernel]] \
    decltype(mm_q8<TA, BM, BN, BK, NSG>) mm_q8<TA, BM, BN, BK, NSG>;

#define TILES(TA, an) \
    INST(TA, an, 64, 64, 32, 4) \
    INST(TA, an, 64, 64, 64, 4) \
    INST(TA, an, 128, 64, 64, 4) \
    INST(TA, an, 128, 128, 32, 8) \
    INST(TA, an, 128, 128, 64, 8)

TILES(half, f16)
TILES(float, f32)
"#;

    /// `SOURCE_Q8`'s tiles: rows of `C`, columns, the `K` step, SIMD groups.
    const Q8_TILES: [(usize, usize, usize, usize); 5] =
        [(64, 64, 32, 4), (64, 64, 64, 4), (128, 64, 64, 4), (128, 128, 32, 8), (128, 128, 64, 8)];

    /// Bytes in one Q8_0 block: an f16 scale and 32 `int8`s.
    const Q8_BLOCK: usize = 34;

    /// The Q8_0 kernel as a candle op. The weight arrives as a flat `u8`
    /// tensor of blocks, because `QTensor` keeps its Metal buffer private;
    /// `n` says how many rows the bytes hold.
    #[derive(Clone)]
    struct MppQ8 {
        pipe: ComputePipeline,
        tile: (usize, usize, usize, usize),
        n: usize,
        poison: bool,
    }

    impl CustomOp2 for MppQ8 {
        fn name(&self) -> &'static str {
            "mpp_matmul2d_q8_0"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
         -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("mpp_matmul2d_q8_0 runs on Metal only")
        }

        fn metal_fwd(&self, a: &MetalStorage, la: &Layout, w: &MetalStorage, lw: &Layout)
         -> candle_core::Result<(MetalStorage, Shape)> {
            let (m, k) = la.shape().dims2()?;
            let n = self.n;
            let (bm, bn, bk, nsg) = self.tile;
            if m % bm != 0 || n % bn != 0 || k % bk != 0 || bk % 32 != 0 || !la.is_contiguous()
               || lw.shape().elem_count() != n * k / 32 * Q8_BLOCK || lw.start_offset() != 0 {
                candle_core::bail!("mpp_matmul2d_q8_0: [{m}, {k}] x [{n}, {k}] does not fit tile \
                                    {bm}x{bn}x{bk}");
            }
            let dev = a.device();
            let out = output(dev, m * n * 4, self.poison)?;
            let guard = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = guard.as_ref();
            enc.set_compute_pipeline_state(&self.pipe);
            enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * a.dtype().size_in_bytes());
            enc.set_input_buffer(1, Some(w.buffer()), 0);
            enc.set_output_buffer(2, Some(&out), 0);
            enc.set_bytes(3, &(m as i32));
            enc.set_bytes(4, &(n as i32));
            enc.set_bytes(5, &(k as i32));
            enc.set_threadgroup_memory_length(0, bn * bk * 2);
            enc.dispatch_thread_groups(
                MTLSize { width: n / bn, height: m / bm, depth: 1 },
                MTLSize { width: 32 * nsg, height: 1, depth: 1 },
            );
            Ok((MetalStorage::new(out, dev.clone(), m * n, DType::F32), Shape::from((m, n))))
        }
    }

    /// The output buffer. candle hands out buffers from a pool, and a pooled
    /// buffer can still hold the last product of the same size: a kernel
    /// that wrote nothing would then "agree" with it exactly. So the
    /// correctness pass fills its output with `0xff` bytes, NaN in f32 and
    /// f16 alike, and anything the kernel leaves unwritten shows up as NaN.
    /// The timed runs skip the fill.
    fn output(dev: &candle_core::MetalDevice, bytes: usize, poison: bool)
     -> candle_core::Result<std::sync::Arc<candle_metal_kernels::metal::Buffer>> {
        let out = dev.allocate_buffer(bytes)?;
        if poison {
            let mut blit = dev.blit_command_encoder()?;
            blit.fill_buffer(&out, (0, bytes), 0xff);
        }
        Ok(out)
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// `max |got - want| / max |want|`, on the GPU, in f32.
    fn err(got: &Tensor, want: &Tensor) -> candle_core::Result<f32> {
        let d = (got.to_dtype(DType::F32)? - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let s = want.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok(d / s)
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::new_metal(0)?;
        let Device::Metal(md) = &dev else { unreachable!() };
        println!("{}", md.metal_device().architecture_name());

        // Compiled at run time by the OS, the way candle compiles its own:
        // no offline Metal toolchain needed. The tensor API wants 4.0.
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let lib = md.metal_device().new_library_with_source(SOURCE, Some(&opts))?;
        let lib_q8 = md.metal_device().new_library_with_source(SOURCE_Q8, Some(&opts))?;
        let op = |tn: &str, tile: (usize, usize, usize)| -> Result<Mpp, Box<dyn std::error::Error>> {
            let (bm, bn, nsg) = tile;
            let f = lib.get_function(&format!("mm_{tn}_{bm}x{bn}_{nsg}"), None)?;
            Ok(Mpp { pipe: md.metal_device().new_compute_pipeline_state_with_function(&f)?, tile, poison: false })
        };
        let op_q8 = |an: &str, tile: (usize, usize, usize, usize), n: usize|
         -> Result<MppQ8, Box<dyn std::error::Error>> {
            let (bm, bn, bk, nsg) = tile;
            let f = lib_q8.get_function(&format!("mm_q8_{an}_{bm}x{bn}x{bk}_{nsg}"), None)?;
            Ok(MppQ8 { pipe: md.metal_device().new_compute_pipeline_state_with_function(&f)?, tile, n,
                       poison: false })
        };

        // `dense`, `q8`, or both when no argument is given.
        let only = std::env::args().nth(1);
        let dense = only.as_deref() != Some("q8");
        let q8 = only.as_deref() != Some("dense");

        // One timed run of `f`: `reps` calls behind one sync, in ms per call.
        let (rounds, reps) = (7, 5);
        let once = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> candle_core::Result<f64> {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let _ = f()?;
            }
            dev.synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1000.0 / reps as f64)
        };
        // candle against a candidate, **in alternating rounds**. This
        // machine's speed drifts by a third between runs, candle's included,
        // so timing one and then the other would put the drift in the ratio.
        // Here each round times both back to back and the ratio is taken per
        // round: (median candle ms, median candidate ms, median ratio).
        let race = |base: &dyn Fn() -> candle_core::Result<Tensor>,
                    cand: &dyn Fn() -> candle_core::Result<Tensor>|
         -> candle_core::Result<(f64, f64, f64)> {
            let _ = base()?;
            let _ = cand()?;
            dev.synchronize()?;
            let (mut tb, mut tc, mut r) = (vec![], vec![], vec![]);
            for _ in 0..rounds {
                let b = once(base)?;
                let c = once(cand)?;
                tb.push(b);
                tc.push(c);
                r.push(b / c);
            }
            Ok((median(tb), median(tc), median(r)))
        };
        let row = |what: String, (tb, tc, x): (f64, f64, f64), tflops: f64, e: f32, apart: f32| {
            let verdict = if !(e <= 1e-2) { "  WRONG" } else { "" };
            println!("  {what:28} {tc:7.2} ms {tflops:5.1} TFLOP/s  candle {tb:7.2} ms  {x:4.2}x  \
                      err {e:.1e}  vs candle {apart:.1e}{verdict}");
        };

        let shapes = [
            ("LTX-2.5 video FFN up, 768x512x121", 6144, 4096, 16384),
            ("LTX-2.5 video attention proj", 6144, 4096, 4096),
            ("7B prefill chunk, gate/up", 512, 3584, 18944),
        ];
        for (what, m, k, n) in shapes {
            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            let tflops = |ms: f64| flop / (ms / 1000.0) / 1e12;
            println!("\n{what}: [{m}, {k}] x [{k}, {n}]");

            let a32 = Tensor::randn(0f32, 1f32, (m, k), &dev)?;
            let b32 = Tensor::randn(0f32, 0.02f32, (k, n), &dev)?;
            let want = a32.matmul(&b32)?;

            // What a q8 model runs now: `Proj::Quant`, f32 activations,
            // because candle's quantised Metal matmul takes nothing else.
            // The same blocks go to both kernels, so any difference between
            // them is the kernels', not the quantisation's.
            let blocks = {
                let wt = b32.t()?.contiguous()?.to_device(&Device::Cpu)?;
                QTensor::quantize(&wt, GgmlDType::Q8_0)?.data()?.into_owned()
            };
            let q = QMatMul::from_qtensor(QTensor::new(
                QStorage::from_data(std::borrow::Cow::Borrowed(&blocks), &dev, GgmlDType::Q8_0)?, (n, k))?)?;
            let theirs_q8 = q.forward(&a32)?;
            println!("  candle q8_0, f32 in: err {:.1e}", err(&theirs_q8, &want)?);

            if q8 {
                let wq = Tensor::from_slice(&blocks, blocks.len(), &dev)?;
                // kvad's activations are f32, so the f16 kernels are timed
                // with the cast in front of them: that is what `Proj::Quant`
                // would run instead of `q.forward`.
                //
                // `matmul2d` also takes f32 on the left, and gives back
                // candle's product bit for bit at candle's speed: it runs on
                // the ALUs, not the matrix units. One tile shows that; the
                // rest would only show it again.
                for an in ["f32", "f16"] {
                    let a = if an == "f16" { a32.to_dtype(DType::F16)? } else { a32.clone() };
                    let tiles = if an == "f16" { &Q8_TILES[..] } else { &Q8_TILES[..1] };
                    for &tile in tiles {
                        let (bm, bn, bk, nsg) = tile;
                        if m % bm != 0 || n % bn != 0 || k % bk != 0 {
                            continue;
                        }
                        let mpp = op_q8(an, tile, n)?;
                        let ours = a.apply_op2_no_bwd(&wq, &MppQ8 { poison: true, ..mpp.clone() })?;
                        let e = err(&ours, &want)?;
                        let apart = err(&ours, &theirs_q8)?;
                        let r = race(&|| q.forward(&a32), &|| {
                            let a = if an == "f16" { a32.to_dtype(DType::F16)? } else { a32.clone() };
                            a.apply_op2_no_bwd(&wq, &mpp)
                        })?;
                        row(format!("q8 {an} in {bm}x{bn}x{bk}/{nsg}sg"), r, tflops(r.1), e, apart);
                    }
                }
            }
            if !dense {
                continue;
            }

            for (dt, tn) in [(DType::F16, "f16"), (DType::BF16, "bf16")] {
                let a = a32.to_dtype(dt)?;
                let b = b32.to_dtype(dt)?;
                let theirs = a.matmul(&b)?.to_dtype(DType::F32)?;
                println!("  candle {tn}: err {:.1e}", err(&theirs, &want)?);
                for tile in TILES {
                    let (bm, bn, nsg) = tile;
                    if m % bm != 0 || n % bn != 0 {
                        continue;
                    }
                    let mpp = op(tn, tile)?;
                    let ours = a.apply_op2_no_bwd(&b, &Mpp { poison: true, ..mpp.clone() })?;
                    let e = err(&ours, &want)?;
                    // Against candle's own output too. Two kernels that each
                    // round correctly differ by about one rounding step of the
                    // largest output: the same order as either one's `err`,
                    // and never much more.
                    let apart = err(&ours, &theirs)?;
                    let r = race(&|| a.matmul(&b), &|| a.apply_op2_no_bwd(&b, &mpp))?;
                    row(format!("matmul2d {tn} {bm}x{bn}/{nsg}sg"), r, tflops(r.1), e, apart);
                }
            }
        }
        Ok(())
    }
}
