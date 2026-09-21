//! Where a decode step's time goes at long context, on Metal.
//!
//! Qwen2.5-Coder-7B's shapes, one layer, timed with a device sync after each
//! piece so the numbers are the GPU's and not the queue's. Run:
//!
//!     cargo run --release -p kvad-gpu --example attn_cost
use candle_core::{DType, Device, Tensor};

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_metal(0)?;
    let (kv_heads, group, hd) = (4usize, 7usize, 128usize);
    let heads = kv_heads * group;
    let layers = 28;
    let dt = DType::BF16;

    // Every piece is run twice and the second run is the one reported.
    // Metal compiles a pipeline the first time it dispatches one, and that
    // cost belongs to the pipeline rather than to the shape being measured —
    // it is what made the first row of this table read 580 ms for a matmul
    // that takes 0.4.
    let twice = |what: &dyn Fn() -> candle_core::Result<Tensor>| -> candle_core::Result<f64> {
        let _ = what()?;
        dev.synchronize()?;
        let t = std::time::Instant::now();
        let _ = what()?;
        dev.synchronize()?;
        Ok(ms(t.elapsed()))
    };

    for seq in [512usize, 2048, 6518] {
        let pk = Tensor::zeros((1, kv_heads, seq, hd), dt, &dev)?;
        let k1 = Tensor::zeros((1, kv_heads, 1, hd), dt, &dev)?;
        let q = Tensor::zeros((1, heads, 1, hd), dt, &dev)?;
        let k = Tensor::cat(&[&pk, &k1], 2)?;
        let kx = k
            .unsqueeze(2)?
            .expand((1, kv_heads, group, seq + 1, hd))?
            .reshape((1, kv_heads * group, seq + 1, hd))?;
        let kt = kx.transpose(2, 3)?.contiguous()?;
        dev.synchronize()?;

        let cat = twice(&|| Tensor::cat(&[&pk, &k1], 2))?;
        let repeat = twice(&|| {
            k.unsqueeze(2)?
                .expand((1, kv_heads, group, seq + 1, hd))?
                .reshape((1, kv_heads * group, seq + 1, hd))
        })?;
        let contig = twice(&|| kx.transpose(2, 3)?.contiguous())?;
        let mm = twice(&|| q.matmul(&kt))?;

        // What the same attention costs with no copies at all: the query
        // heads folded onto their KV head, so the KV cache is read where it
        // lies. This is what the model does now.
        let folded = twice(&|| {
            let qg = q.reshape((1, kv_heads, group, hd))?;
            qg.matmul(&k.transpose(2, 3)?.contiguous()?)
        })?;

        let step = (cat + repeat + contig + mm) * layers as f64;
        println!(
            "seq {seq:5}: cat {cat:5.2} ms | repeat_kv {repeat:5.2} ms | contiguous {contig:5.2} ms \
             | matmul {mm:5.2} ms  => {step:6.1} ms per token over {layers} layers"
        );
        println!("            folded onto the KV head, no copies: {folded:5.2} ms per layer, \
{:5.1} ms per token", folded * layers as f64);
    }

    Ok(())
}
