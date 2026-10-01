//! What one step of LoRA training costs on SDXL's UNet (#74).
//!
//!     cargo build --release -p kvad-gpu --example tune_budget
//!     /usr/bin/time -l target/release/examples/tune_budget --side 512
//!
//! Options: `--side N` (pixels, 256), `--rank N` (16), `--steps N` (4),
//! `--f32` (the UNet in f32 rather than f16), `--cpu`, and `--cap N`, the
//! gigabytes of memory at which the run ends itself (24; see
//! `kvad_gpu::cap`). `--prove-cap` shows the ceiling holding, on 1 GB.
//! `--whole` records the whole UNet at once, where gradient checkpointing
//! walks back a stretch at a time: for comparing the two where both fit.
//!
//! It prints each step's seconds and loss; `/usr/bin/time -l` around the
//! binary gives the peak memory footprint. Run it on the binary, not on
//! `cargo run`, whose own footprint it would report.
//!
//! Step the side up from 256, by a third at most, and work out what the
//! next size should take from the last two before running it. The ceiling
//! is what keeps a wrong estimate from taking the machine down: without it
//! and without checkpointing, 512 did. `--trace` prints the last step a
//! stretch at a time.

use candle_core::{DType, Device};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let has = |f: &str| argv.iter().any(|a| a == f);
    let number = |f: &str, or: usize| -> Res<usize> {
        match argv.iter().position(|a| a == f) {
            Some(i) => Ok(argv.get(i + 1).ok_or_else(|| format!("{f} needs a value"))?.parse()?),
            None => Ok(or),
        }
    };
    let (side, rank, steps) = (number("--side", 256)?, number("--rank", 16)?, number("--steps", 4)?);
    let device = if has("--cpu") { Device::Cpu } else { Device::new_metal(0)? };
    if has("--prove-cap") {
        // The ceiling, shown to hold for the GPU's buffers: 1 GB above
        // where the process stands, then 256 MB at a time to 4 GB at most.
        // It should stop near the fourth.
        kvad_gpu::cap::at(kvad_gpu::cap::footprint() as f64 / 1e9 + 1.0);
        let mut held = Vec::new();
        for i in 0..16 {
            held.push(candle_core::Tensor::ones((256 << 20,), DType::U8, &device)?);
            device.synchronize()?;
            eprintln!("{} MB held, footprint {:.2} GB", 256 * (i + 1), kvad_gpu::cap::footprint() as f64 / 1e9);
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        return Err("the ceiling did not hold: 4 GB allocated under a 1 GB allowance".into());
    }
    kvad_gpu::cap::at(number("--cap", 24)? as f64);
    let dtype = if has("--f32") { DType::F32 } else { DType::F16 };
    let whole = has("--whole");
    let b = kvad_gpu::image::tune::sdxl(kvad_gpu::image::sdxl::REPO, &device, dtype, side, rank, steps, whole)?;
    eprintln!(
        "SDXL's UNet, {:.2} B frozen parameters in {dtype:?}; a rank-{rank} LoRA on {} layers, {:.1} M trained, in f32; {side}×{side}, one picture a step",
        b.frozen as f64 / 1e9,
        b.layers,
        b.trained as f64 / 1e6
    );
    eprintln!("{}; one forward pass as drawing runs it: {:.2} s", if whole { "through the whole UNet at once" } else { "a stretch at a time" }, b.drawing);
    for (i, (s, l)) in b.secs.iter().zip(&b.loss).enumerate() {
        let [found, opt] = b.parts[i];
        eprintln!("step {}: {s:.2} s (the gradients {found:.2}, the optimiser {opt:.2}), loss {l:.4}", i + 1);
    }
    if has("--trace") && !b.trace.is_empty() {
        // n stretches forward, the loss, n stretches back, last first.
        let n = (b.trace.len() - 1) / 2;
        let gb = |bytes: u64| bytes as f64 / 1e9;
        eprintln!("the last step, a stretch at a time: forward, then back; the footprint each reached, and left");
        for i in 0..n {
            let (fwd, fp, fl) = b.trace[i];
            let (back, peak, left) = b.trace[2 * n - i];
            eprintln!("  stretch {i:>2}: forward {fwd:.3} s, {:.1} then {:.1} GB; back {back:.3} s, {:.1} then {:.1} GB", gb(fp), gb(fl), gb(peak), gb(left));
        }
        let (fwd, back): (f64, f64) = (b.trace[..n].iter().map(|t| t.0).sum(), b.trace[n + 1..].iter().map(|t| t.0).sum());
        eprintln!("  {n} stretches: {fwd:.2} s forward, {back:.2} s back");
    }
    eprintln!("footprint at the end: {:.1} GB", kvad_gpu::cap::footprint() as f64 / 1e9);
    if b.loss.iter().any(|l| !l.is_finite()) {
        return Err("a loss is not a number".into());
    }
    Ok(())
}
