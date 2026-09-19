//! Microbenchmark for the quantised matmul kernels.
//!
//!     cargo run --release -p kvad --example bench_matvec
//!
//! Shapes are taken from Qwen2.5-0.5B: the output head dominates, so it is the
//! one worth tuning.

use kvad::quant::{Precision, Weight};
use kvad::tensor::Tensor;
use nanograd::rng::Rng;
use std::time::Instant;

fn bench(label: &str, rows: usize, cols: usize) {
    let mut rng = Rng::new(42);
    let t = Tensor::new(rows, cols, (0..rows * cols).map(|_| rng.normal() * 0.05).collect());
    let x: Vec<f32> = (0..cols).map(|_| rng.normal()).collect();

    println!("\n{label}  [{rows} x {cols}]");
    let mut baseline = 0.0f64;
    for precision in [Precision::F32, Precision::Q8, Precision::Q4] {
        let w = Weight::quantize(t.clone(), precision);
        let bytes = w.bytes();

        // Warm up, then time enough iterations to be meaningful.
        let run = |w: &Weight| w.matvec_bt(&x, None);
        for _ in 0..3 {
            std::hint::black_box(run(&w));
        }

        // Calibrate the iteration count so every round takes roughly the same
        // wall time regardless of matrix size. A fixed count makes small
        // matrices finish in a few milliseconds, which is short enough for
        // background load to dominate the measurement.
        let probe = Instant::now();
        std::hint::black_box(run(&w));
        let one = probe.elapsed().as_secs_f64().max(1e-9);
        let iters = ((0.1 / one) as usize).clamp(20, 20_000);

        // Best of several rounds rather than a single average. A shared
        // machine produces occasional slow rounds from scheduling and clock
        // changes; those only ever add time, so the minimum is the most stable
        // estimate of what the code actually costs.
        let mut per_call = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(run(&w));
            }
            per_call = per_call.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        let gbps = bytes as f64 / per_call / 1e9;
        if precision == Precision::F32 {
            baseline = per_call;
        }

        println!(
            "  {:<4} {:>7.0} MB  {:>7.2} ms/call  {:>6.1} GB/s  {:>5.2}x",
            precision.to_string(),
            bytes as f64 / 1e6,
            per_call * 1e3,
            gbps,
            baseline / per_call
        );
    }
}

/// The prefill shape: one weight matrix against a batch of activations.
fn bench_batch(label: &str, rows: usize, cols: usize, m: usize) {
    let mut rng = Rng::new(7);
    let t = Tensor::new(rows, cols, (0..rows * cols).map(|_| rng.normal() * 0.05).collect());
    let xs: Vec<f32> = (0..m * cols).map(|_| rng.normal()).collect();

    println!("\n{label}  [{rows} x {cols}] x {m} tokens");
    let f32w = Weight::quantize(t.clone(), Precision::F32);
    let w = Weight::quantize(t, Precision::Q8);
    let mut baseline = 0.0f64;
    for (name, allow) in [("f32 gemm", false), ("q8 sdot", false), ("q8 smmla", true)] {
        let is_f32 = name.starts_with("f32");
        let run = || {
            if is_f32 {
                f32w.matmul_bt_with(&xs, m, None, false)
            } else {
                w.matmul_bt_with(&xs, m, None, allow)
            }
        };
        for _ in 0..2 {
            std::hint::black_box(run());
        }
        let probe = Instant::now();
        std::hint::black_box(run());
        let one = probe.elapsed().as_secs_f64().max(1e-9);
        let iters = ((0.1 / one) as usize).clamp(5, 5_000);

        let mut per_call = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(run());
            }
            per_call = per_call.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        let gmac = (m * rows * cols) as f64 / per_call / 1e9;
        if is_f32 {
            baseline = per_call;
        }
        println!(
            "  {name:<10} {:>7.2} ms/call  {:>7.1} GMAC/s  {:>5.2}x",
            per_call * 1e3,
            gmac,
            baseline / per_call
        );
    }
    if !kvad::simd::has_i8mm() {
        println!("  (i8mm unavailable or disabled — both rows are the same kernel)");
    }
}

fn main() {
    println!("threads: {}", rayon::current_num_threads());
    println!("i8mm:    {}", kvad::simd::has_i8mm());
    // The output head: by far the largest single matmul per token.
    bench("qwen lm_head", 151936, 896);
    // A representative MLP matrix.
    bench("qwen mlp.down", 896, 4864);
    // GPT-2's widening MLP, in the transposed layout it is now stored in.
    bench("gpt2 c_fc", 3072, 768);

    // Prefill: the same matrices, but against a batch. This is the only place
    // SMMLA applies, because it needs two independent activation rows.
    bench_batch("qwen mlp.down", 896, 4864, 64);
    bench_batch("qwen q_proj", 896, 896, 64);
}
