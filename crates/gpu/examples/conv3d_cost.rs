//! What one of the LTX video decoder's 3D convolutions costs on Metal, and
//! where the time goes: the stacking of neighbouring frames, candle's
//! `im2col`, or the multiply.
//!
//!     cargo run --release -p kvad-gpu --example conv3d_cost -- [C T H W]
//!
//! Defaults to the decoder's costliest level: 256 channels, 121 frames of
//! 64×96 (the six residual blocks there are a third of its arithmetic).

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|v| v.parse().ok()).collect();
    let (c, t, h, w) = match a[..] {
        [c, t, h, w] => (c, t, h, w),
        _ => (256, 121, 64, 96),
    };
    let dev = Device::new_metal(0)?;
    let x = Tensor::randn(0f32, 1.0, (t, c, h, w), &dev)?.to_dtype(DType::BF16)?;
    let k = (Tensor::randn(0f32, 1.0, (c, 3 * c, 3, 3), &dev)? * 0.01)?.to_dtype(DType::BF16)?;
    let flops = 2.0 * (t * h * w) as f64 * (27 * c * c) as f64;
    let time = |label: &str, f: &dyn Fn() -> candle_core::Result<Tensor>| -> Res<()> {
        f()?; // warm
        dev.synchronize()?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let s = Instant::now();
            let y = f()?;
            dev.synchronize()?;
            best = best.min(s.elapsed().as_secs_f64());
            drop(y);
        }
        println!("{label:<44} {best:>7.3} s  {:>6.2} TFLOP/s", flops / best / 1e12);
        Ok(())
    };
    println!("[{t}, {c}, {h}, {w}], {:.2} TFLOP a convolution", flops / 1e12);

    let xp = Tensor::cat(&[&x.narrow(0, 0, 1)?, &x, &x.narrow(0, t - 1, 1)?], 0)?.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
    time("stack the three frames only (copies)", &|| {
        let taps = (0..3).map(|k| xp.narrow(0, k, t)).collect::<candle_core::Result<Vec<_>>>()?;
        Tensor::cat(&taps, 1)
    })?;
    for frames in [1, 6, 16] {
        time(&format!("folded conv2d, {frames} frames a chunk"), &|| {
            let mut out = Vec::new();
            let mut f = 0;
            while f < t {
                let n = frames.min(t - f);
                let taps = (0..3).map(|kk| xp.narrow(0, f + kk, n)).collect::<candle_core::Result<Vec<_>>>()?;
                out.push(Tensor::cat(&taps, 1)?.conv2d(&k, 0, 1, 1, 1)?);
                f += n;
            }
            Tensor::cat(&out, 0)
        })?;
    }
    // The same arithmetic as one matrix multiply, with the im2col done by
    // hand from nine shifted views: what the convolution could cost.
    if t * h * w * 27 * c > 1 << 31 {
        println!("(skipping the hand im2col: its buffer would be past Metal's largest)");
    } else {
    time("hand im2col (27 shifted views) + one matmul", &|| {
        let mut cols = Vec::with_capacity(27);
        for kt in 0..3 {
            for ky in 0..3 {
                for kx in 0..3 {
                    cols.push(xp.narrow(0, kt, t)?.narrow(2, ky, h)?.narrow(3, kx, w)?);
                }
            }
        }
        // [t, 27, c, h, w] → [t·h·w, 27·c]
        let m = Tensor::stack(&cols, 1)?.permute((0, 3, 4, 1, 2))?.contiguous()?.reshape((t * h * w, 27 * c))?;
        let wk = k.reshape((c, 3, c, 3, 3))?.permute((1, 3, 4, 2, 0))?.contiguous()?.reshape((27 * c, c))?;
        m.matmul(&wk)
    })?;
    }
    // The multiply alone, in chunks of eight frames (a whole clip's im2col
    // is past Metal's largest buffer).
    let rows = 8 * h * w;
    let m = Tensor::randn(0f32, 1.0, (rows, 27 * c), &dev)?.to_dtype(DType::BF16)?;
    let wk = Tensor::randn(0f32, 1.0, (27 * c, c), &dev)?.to_dtype(DType::BF16)?;
    let chunks = t.div_ceil(8);
    time("the matmul alone, 8 frames at a time", &|| {
        let mut last = m.matmul(&wk)?;
        for _ in 1..chunks {
            last = m.matmul(&wk)?;
        }
        Ok(last)
    })?;
    Ok(())
}
