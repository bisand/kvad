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
//! The question is narrow on purpose: is there a gain at the shapes that
//! matter, before anything is built around it? The shapes are a video DiT's
//! (LTX-2.5, #51) and one prefill chunk of a 7B model. Decode is left out: at
//! one row a matmul is bound by memory, and no matrix unit changes that.
//!
//!     cargo run --release -p kvad-gpu --example neural_accel
//!
//! # Reading the numbers
//!
//! - Each timing is the median of several rounds. A round runs the op a few
//!   times behind **one** device sync, never a sync per repeat, for the reason
//!   `prefill_cost` gives: a sync hands candle's pooled buffers back and the
//!   next repeat pays to allocate its output afresh.
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
    struct Mpp {
        pipe: ComputePipeline,
        tile: (usize, usize, usize),
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
            let out = dev.new_buffer(m * n, dt, "mpp_matmul2d")?;
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
        let op = |tn: &str, tile: (usize, usize, usize)| -> Result<Mpp, Box<dyn std::error::Error>> {
            let (bm, bn, nsg) = tile;
            let f = lib.get_function(&format!("mm_{tn}_{bm}x{bn}_{nsg}"), None)?;
            Ok(Mpp { pipe: md.metal_device().new_compute_pipeline_state_with_function(&f)?, tile })
        };

        let (rounds, reps) = (7, 5);
        let time = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> candle_core::Result<f64> {
            let _ = f()?;
            dev.synchronize()?;
            let mut ms = Vec::with_capacity(rounds);
            for _ in 0..rounds {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = f()?;
                }
                dev.synchronize()?;
                ms.push(t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
            }
            Ok(median(ms))
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
            let q = {
                let wt = b32.t()?.contiguous()?.to_device(&Device::Cpu)?;
                let cpu = QTensor::quantize(&wt, GgmlDType::Q8_0)?;
                QMatMul::from_qtensor(QTensor::new(
                    QStorage::from_data(cpu.data()?, &dev, GgmlDType::Q8_0)?, (n, k))?)?
            };
            let t = time(&|| q.forward(&a32))?;
            println!("  {:28} {t:8.2} ms  {:5.1} TFLOP/s  err {:.1e}",
                     "candle q8_0, f32 in", tflops(t), err(&q.forward(&a32)?, &want)?);

            for (dt, tn) in [(DType::F16, "f16"), (DType::BF16, "bf16")] {
                let a = a32.to_dtype(dt)?;
                let b = b32.to_dtype(dt)?;
                let base = time(&|| a.matmul(&b))?;
                let theirs = a.matmul(&b)?.to_dtype(DType::F32)?;
                println!("  {:28} {base:8.2} ms  {:5.1} TFLOP/s  err {:.1e}",
                         format!("candle {tn}"), tflops(base), err(&theirs, &want)?);
                for tile in TILES {
                    let (bm, bn, nsg) = tile;
                    if m % bm != 0 || n % bn != 0 {
                        continue;
                    }
                    let mpp = op(tn, tile)?;
                    let ours = a.apply_op2_no_bwd(&b, &mpp)?;
                    let e = err(&ours, &want)?;
                    // Against candle's own output too. Two kernels that each
                    // round correctly differ by about one rounding step of the
                    // largest output: the same order as either one's `err`,
                    // and never much more.
                    let apart = err(&ours, &theirs)?;
                    let t = time(&|| a.apply_op2_no_bwd(&b, &mpp))?;
                    let verdict = if e > 1e-2 { "  WRONG" } else { "" };
                    println!("  {:28} {t:8.2} ms  {:5.1} TFLOP/s  err {e:.1e}  vs candle {apart:.1e}  \
                              {:4.2}x{verdict}",
                             format!("matmul2d {tn} {bm}x{bn}/{nsg}sg"), tflops(t), base / t);
                }
            }
        }
        Ok(())
    }
}
