//! Where a decode step's time goes, on Metal.
//!
//! The third of `attn_cost` and `prefill_cost`. A decode step multiplies one
//! row by every weight in the model, so each weight is read once and used
//! once, and the step is bound by how fast memory hands the weights over,
//! not by arithmetic. That is why the M5's matrix units do nothing for it.
//! What this asks is how close to that bound it runs, and where the rest
//! goes:
//!
//! 1. **The ceiling:** what this machine's memory delivers to the simplest
//!    kernels there are. Row sums read, `x·1 + 0` reads and writes, and one very
//!    large matrix-vector product is the decode kernel at its most favourable
//!    size.
//! 2. **The matmuls of one token:** every matrix-vector product one token of
//!    Qwen2.5-1.5B-Instruct takes, at each precision, timed alone. Every
//!    layer has its own weights, as in the model, so no layer is served from
//!    a cache the previous one warmed.
//! 3. **One small op:** what a GPU op costs when it has almost nothing to do.
//!    A decode step is hundreds of them: norms, rotations, adds and casts
//!    between the matmuls.
//!
//! Set these beside `kvad-gpu run`'s decode rate at the same precision, and
//! whatever the matmuls do not account for is (3), times however many small
//! ops a token takes.
//!
//!     cargo run --release -p kvad-gpu --example decode_cost
//!
//! Timing is as in `prefill_cost`: repeats behind one sync, the median of
//! several rounds.
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Qwen2.5-1.5B-Instruct, from its `config.json`.
const LAYERS: usize = 28;
const E: usize = 1536;
const KV: usize = 2 * 128;
const FFN: usize = 8960;
const VOCAB: usize = 151936;

fn main() -> Res<()> {
    let dev = Device::new_metal(0)?;
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // Median ms per call of `f`, over 7 rounds of `reps` calls, one sync each.
    let time = |reps: usize, f: &dyn Fn() -> candle_core::Result<()>| -> candle_core::Result<f64> {
        f()?;
        dev.synchronize()?;
        let mut ms = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f()?;
            }
            dev.synchronize()?;
            ms.push(t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
        }
        Ok(median(ms))
    };
    let gbs = |bytes: f64, ms: f64| bytes / (ms / 1000.0) / 1e9;

    println!("1. the ceiling: 1 GiB through the simplest kernels there are\n");
    let n = 256 << 20;
    let bytes = (n * 4) as f64;
    let big = Tensor::rand(0f32, 1f32, n, &dev)?;
    // Row sums, not `sum_all`: reducing a gigabyte to one number keeps too
    // few threads busy and measures the reduction, 17 GB/s. And not `copy()`,
    // which on Metal never reached the GPU at all.
    let rows = big.reshape((n / 1024, 1024))?;
    let read = time(5, &|| rows.sum_keepdim(1).map(drop))?;
    let copy = time(5, &|| big.affine(1.0, 0.0).map(drop))?;
    let w = Tensor::rand(0f32, 1f32, (8192, 65536), &dev)?.to_dtype(DType::BF16)?;
    let x = Tensor::rand(0f32, 1f32, (1, 8192), &dev)?.to_dtype(DType::BF16)?;
    let gemv = time(5, &|| x.matmul(&w).map(drop))?;
    drop((big, w));
    println!("  row sums, read 1 GiB                {read:6.2} ms  {:5.0} GB/s", gbs(bytes, read));
    println!("  x·1 + 0, reads and writes 1 GiB     {copy:6.2} ms  {:5.0} GB/s", gbs(2.0 * bytes, copy));
    println!("  bf16 [1, 8192] x [8192, 65536]      {gemv:6.2} ms  {:5.0} GB/s", gbs(bytes, gemv));

    println!("\n2. one token's matmuls, Qwen2.5-1.5B-Instruct: {LAYERS} layers x 7, and the head\n");
    // (in, out) of each projection in a layer: q, k, v, o, gate, up, down.
    let shapes = [(E, E), (E, KV), (E, KV), (E, E), (E, FFN), (E, FFN), (FFN, E)];
    for (what, quant) in [("bf16", None), ("q8_0", Some(GgmlDType::Q8_0)), ("q4_0", Some(GgmlDType::Q4_0)),
                          ("q4_k", Some(GgmlDType::Q4K))] {
        // Dense: `Proj::Dense`, weight stored [in, out], bf16 activations.
        // Quantised: `Proj::Quant`, [out, in] blocks, f32 activations.
        let (mut weight_bytes, mut layers) = (0usize, Vec::new());
        let mut proj = |inp: usize, out: usize| -> Res<Box<dyn Fn(&Tensor) -> candle_core::Result<Tensor>>> {
            Ok(match quant {
                None => {
                    let w = Tensor::rand(-0.02f32, 0.02f32, (inp, out), &dev)?.to_dtype(DType::BF16)?;
                    weight_bytes += inp * out * 2;
                    Box::new(move |x: &Tensor| x.matmul(&w))
                }
                Some(gd) => {
                    let w = Tensor::rand(-0.02f32, 0.02f32, (out, inp), &Device::Cpu)?;
                    let q = QTensor::quantize_onto(&w, gd, &dev)?;
                    weight_bytes += q.storage_size_in_bytes();
                    let q = QMatMul::from_qtensor(q)?;
                    Box::new(move |x: &Tensor| q.forward(x))
                }
            })
        };
        for _ in 0..LAYERS {
            let mut layer = Vec::new();
            for &(inp, out) in &shapes {
                layer.push((inp, proj(inp, out)?));
            }
            layers.push(layer);
        }
        let head = proj(E, VOCAB)?;
        let act = if quant.is_some() { DType::F32 } else { DType::BF16 };
        let xe = Tensor::rand(0f32, 1f32, (1, E), &dev)?.to_dtype(act)?;
        let xf = Tensor::rand(0f32, 1f32, (1, FFN), &dev)?.to_dtype(act)?;
        let ops = LAYERS * shapes.len() + 1;
        let token = time(3, &|| {
            for layer in &layers {
                for (inp, p) in layer {
                    p(if *inp == E { &xe } else { &xf })?;
                }
            }
            head(&xe).map(drop)
        })?;
        println!(
            "  {what:5} {:5.2} GB of weights  {token:6.2} ms  {:5.0} GB/s  = {:5.1} tok/s if nothing else ran \
             ({ops} matmuls, {:.1} µs each)",
            weight_bytes as f64 / 1e9,
            gbs(weight_bytes as f64, token),
            1000.0 / token,
            token * 1000.0 / ops as f64
        );
    }

    println!("\n3. one small op: a [1, {E}] add, which has next to nothing to do\n");
    let a = Tensor::rand(0f32, 1f32, (1, E), &dev)?.to_dtype(DType::BF16)?;
    let add = time(1000, &|| (&a + &a).map(drop))?;
    println!("  {:.1} µs each", add * 1000.0);
    Ok(())
}
