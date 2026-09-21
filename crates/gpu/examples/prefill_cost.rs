//! Where a prefill chunk's time goes at long context, on Metal.
//!
//! The sibling of `attn_cost`, asking the other half of the question. That
//! one times a decode step — one query row against a long cache. This one
//! times a chunk of `PREFILL_CHUNK` query rows, which is the shape the
//! forward pass runs thirteen times to read a 6.5k-token prompt.
//!
//! Qwen2.5-Coder-7B's shapes, f32 activations: candle's quantised Metal
//! matmul takes nothing else, so that is what the server runs whenever the
//! weights are quantised. Run:
//!
//!     cargo run --release -p kvad-gpu --example prefill_cost
//!
//! # Measuring this without lying to yourself
//!
//! Each piece is run several times with **one** device sync at the end,
//! never a sync per repeat. A sync is not just a wait: `wait_until_completed`
//! calls `drop_unused_buffers`, which returns candle's pooled allocations to
//! the device. Sync after every op and each repeat re-allocates and
//! first-touches its output — 382 MB of it, for the score matrix below. An
//! earlier version of this file did exactly that and reported the three
//! elementwise passes at roughly twice their cost. The forward pass syncs
//! once per chunk, so the numbers here are the ones it pays.
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops;

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_metal(0)?;
    let (n_embd, inter) = (3584usize, 18944usize);
    let (n_head, n_kv, hd) = (28usize, 4usize, 128usize);
    let group = n_head / n_kv;
    let layers = 28;
    let m = 512; // PREFILL_CHUNK
    let scale = 1.0 / (hd as f64).sqrt();
    let reps = 5;

    let run = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> candle_core::Result<f64> {
        let _ = f()?;
        dev.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let _ = f()?;
        }
        dev.synchronize()?;
        Ok(ms(t.elapsed()) / reps as f64)
    };

    let quantised = |out: usize, inp: usize| -> Result<QMatMul, Box<dyn std::error::Error>> {
        let w = Tensor::randn(0f32, 0.02f32, (out, inp), &Device::Cpu)?;
        let cpu = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let blocks = cpu.data()?;
        let q = QTensor::new(QStorage::from_data(blocks, &dev, GgmlDType::Q8_0)?, (out, inp))?;
        Ok(QMatMul::from_qtensor(q)?)
    };

    // Attention as this backend wrote it before `ops::sdpa`: fold the query
    // heads onto their KV head, write the scores down, scale, mask, softmax.
    let written_out = |q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>|
     -> candle_core::Result<Tensor> {
        let seq = k.dim(2)?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        let qg = q.reshape((1, n_kv, group * m, hd))?;
        let mut att = (qg.matmul(&kt)? * scale)?;
        if let Some(msk) = mask {
            att = att.reshape((1, n_head, m, seq))?.broadcast_add(msk)?
                     .reshape((1, n_kv, group * m, seq))?;
        }
        ops::softmax_last_dim(&att)?.matmul(v)?.reshape((1, n_head, m, hd))
    };

    println!("Qwen2.5-Coder-7B, one layer, m = {m} queries, q8 weights, f32 activations\n");
    println!("the seven weight matrices, which is most of a prefill:");
    let shapes = [
        ("q", n_head * hd, n_embd), ("k", n_kv * hd, n_embd), ("v", n_kv * hd, n_embd),
        ("o", n_embd, n_head * hd), ("gate", inter, n_embd), ("up", inter, n_embd),
        ("down", n_embd, inter),
    ];
    let mut gemm = 0.0;
    let mut flops = 0.0;
    for (_, out, inp) in shapes {
        let w = quantised(out, inp)?;
        let x = Tensor::randn(0f32, 1f32, (m, inp), &dev)?;
        gemm += run(&|| w.forward(&x))?;
        flops += 2.0 * m as f64 * out as f64 * inp as f64;
    }
    println!(
        "  {gemm:6.2} ms/layer = {:.2} TFLOP/s, and the same within 10% dense in bf16, f16 or \n  \
         f32 — quantisation here buys memory, not prefill speed. Over {layers} layers x 13 \n  \
         chunks that is {:.1} s, and it is a floor.\n",
        flops / (gemm / 1000.0) / 1e12,
        gemm * layers as f64 * 13.0 / 1000.0
    );

    println!("attention, written out against the fused kernel:");
    for pos0 in [0usize, 2048, 6144] {
        let seq = pos0 + m;
        let q = Tensor::randn(0f32, 1f32, (1, n_head, m, hd), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv, seq, hd), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv, seq, hd), &dev)?;
        let mask = Tensor::zeros((m, seq), DType::F32, &dev)?;
        let scores = (q.elem_count() / hd) as f64 * seq as f64 * 4.0 / 1e6;

        let slow = run(&|| written_out(&q, &k, &v, Some(&mask)))?;
        let fast = run(&|| ops::sdpa(&q, &k, &v, None, true, scale as f32, 1.0))?;
        println!(
            "  cache {seq:5}: written out {slow:6.2} ms   fused {fast:5.2} ms   {:4.1}x   \
             (the score matrix it does not write is {scores:.0} MB)",
            slow / fast
        );
    }

    println!("\nand the three elementwise passes over that matrix, which are why:");
    let seq = 6144 + m;
    let att = Tensor::ones((1, n_kv, group * m, seq), DType::F32, &dev)?;
    let mask = Tensor::zeros((m, seq), DType::F32, &dev)?;
    let same = Tensor::zeros((1, n_kv, group * m, seq), DType::F32, &dev)?;
    let mb = att.elem_count() as f64 * 4.0 / 1e6;
    for (what, t) in [
        ("* scale", run(&|| &att * scale)?),
        ("+ mask, broadcast from [m, seq]", run(&|| {
            att.reshape((1, n_head, m, seq))?.broadcast_add(&mask)?
               .reshape((1, n_kv, group * m, seq))
        })?),
        ("+ mask, already the right shape", run(&|| &att + &same)?),
        ("softmax_last_dim", run(&|| ops::softmax_last_dim(&att))?),
    ] {
        println!("  {what:34} {t:6.2} ms = {:5.0} GB/s", 2.0 * mb / t);
    }
    println!(
        "\n  A broadcast costs 3x a plain add of the same bytes. None of it is arithmetic,\n  \
         and the fused kernel pays none of it."
    );
    Ok(())
}
