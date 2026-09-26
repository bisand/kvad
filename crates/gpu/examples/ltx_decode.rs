//! Decode an LTX-2.5 latent to frames, with no model but the decoder.
//!
//!     cargo run --release -p kvad-gpu --example ltx_decode -- \
//!         --latent z.safetensors [--expect frames.safetensors] [--out clip.mp4] [--cpu]
//!
//! `--latent` is a safetensors file holding `latent`, `[128, F, h, w]` (or
//! with a leading batch axis of 1), normalised as the DiT produces it.
//! `--expect` holds `frames`, `[3, 8(F − 1) + 1, 32h, 32w]` in `[0, 1]`, what
//! the reference decoder made of the same latent; the example says how far
//! apart the two are. `--cpu` decodes on the CPU in f32 rather than on Metal
//! in bf16, which is the closest this gets to the reference's arithmetic.
//!
//! `--f32` keeps Metal but computes in f32, to tell bf16's error from
//! Metal's.
//!
//! `--path FILE` loads the decoder from a file rather than the Hub, and
//! `--where` only prints where the Hub's copy is (downloading it first).
//! `--profile` prints how long each step of the decoder took
//! ([`kvad_gpu::prof`]), with the steps at one size summed.

use candle_core::{DType, Device};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::{ltx_vae, LTX_REPO};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| argv.iter().any(|a| a == f);
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();

    let path = match value("--path") {
        Some(p) => p.into(),
        None => fetch_file(LTX_REPO, ltx_vae::FILE, &Watcher::none())?,
    };
    if flag("--where") {
        println!("{}", path.display());
        return Ok(());
    }
    let (device, dtype) = match (flag("--cpu"), flag("--f32")) {
        (true, _) => (Device::Cpu, DType::F32),
        (false, f32) => (Device::new_metal(0)?, if f32 { DType::F32 } else { DType::BF16 }),
    };

    let t = Instant::now();
    let decoder = ltx_vae::VideoDecoder::load(&path, &device, dtype)?;
    eprintln!("decoder: {:.0} M parameters, loaded in {:.1} s", decoder.params() as f64 / 1e6, t.elapsed().as_secs_f64());

    let latent_file = value("--latent").ok_or("--latent FILE is required")?;
    let mut latent = candle_core::safetensors::load(&latent_file, &Device::Cpu)?.remove("latent").ok_or("no `latent` tensor")?;
    if latent.rank() == 5 {
        latent = latent.squeeze(0)?;
    }
    let latent = latent.to_device(&device)?;

    if flag("--profile") {
        kvad_gpu::prof::start();
    }
    let t = Instant::now();
    let frames = decoder.decode(&latent)?.to_device(&Device::Cpu)?;
    let (n, _, h, w) = frames.dims4()?;
    let took = t.elapsed().as_secs_f64();
    eprintln!("decoded {:?} to {n} frames of {w}×{h} in {took:.2} s", latent.dims());
    for r in kvad_gpu::prof::stop() {
        eprintln!("   {:<34} {:>2}× {:>7.2} s {:>5.1}%", r.label, r.calls, r.seconds, 100.0 * r.seconds / took);
    }

    if let Some(expect) = value("--expect") {
        let mut want = candle_core::safetensors::load(&expect, &Device::Cpu)?.remove("frames").ok_or("no `frames` tensor")?;
        if want.rank() == 5 {
            want = want.squeeze(0)?;
        }
        // The reference is channels first, [3, T, H, W]; ours is frames first.
        let want = want.permute((1, 0, 2, 3))?.to_dtype(DType::F32)?;
        let diff = (&frames - &want)?.abs()?.flatten_all()?;
        let worst = diff.max(0)?.to_scalar::<f32>()?;
        let mean = diff.mean_all()?.to_scalar::<f32>()?;
        let mse = (&frames - &want)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        eprintln!(
            "against the reference: mean |Δ| {mean:.5}, worst {worst:.4}, PSNR {:.1} dB (in 8-bit levels: mean {:.2}, worst {:.1})",
            10.0 * (1.0 / mse.max(1e-20)).log10(),
            mean * 255.0,
            worst * 255.0
        );
    }

    if let Some(out) = value("--out") {
        let video = ltx_vae::to_video(&frames, value("--fps").map(|f| f.parse()).transpose()?.unwrap_or(24))?;
        std::fs::write(&out, video.mp4(None))?;
        eprintln!("{out}: {} frames", video.frames());
    }
    Ok(())
}
