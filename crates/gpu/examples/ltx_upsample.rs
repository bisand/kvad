//! LTX-2.5's spatial latent upsampler, checked against the reference's own.
//!
//!     cargo run --release -p kvad-gpu --example ltx_upsample -- --fixtures DIR [--cpu]
//!
//! `DIR` is what `scripts/ltx-fixtures.py --upsampler … --vae … --latent …`
//! wrote: a stage-1 latent, and what the reference's `upsample_video` made of
//! it in f32 on the CPU and in bf16 on MPS. This upsamples the same latent,
//! on Metal in bf16 (or on the CPU in f32 with `--cpu`, or on Metal in f32
//! with `--f32`), and says how far apart they are. `--where` prints the
//! upsampler's and the VAE's paths, downloading them first.
//!
//! `--latent IN --save OUT` upsamples a latent of your own instead (`video`
//! or `latent` in IN, as `examples/ltx.rs --latents` writes), in f32 on
//! Metal, into `latent` in OUT: a latent twice the size, for trying the
//! decoder at that size without running the DiT.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::{ltx_upsample, ltx_vae, LTX_REPO};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Signal-to-error ratio in dB of `got` against `want`.
fn db(got: &Tensor, want: &Tensor) -> Res<f32> {
    let cpu = |t: &Tensor| -> candle_core::Result<Tensor> { t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all() };
    let (g, w) = (cpu(got)?, cpu(want)?);
    let err = (&g - &w)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let sig = w.sqr()?.sum_all()?.to_scalar::<f32>()?;
    if !(err + sig).is_finite() {
        return Err("a comparison met a NaN or an infinity".into());
    }
    Ok(10.0 * (sig / err.max(1e-30)).log10())
}

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| argv.iter().any(|a| a == f);
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let path = fetch_file(LTX_REPO, ltx_upsample::FILE, &Watcher::none())?;
    let vae = fetch_file(LTX_REPO, ltx_vae::FILE, &Watcher::none())?;
    if flag("--where") {
        println!("{}\n{}", path.display(), vae.display());
        return Ok(());
    }
    if let (Some(input), Some(save)) = (value("--latent"), value("--save")) {
        let device = Device::new_metal(0)?;
        let mut m = candle_core::safetensors::load(&input, &Device::Cpu)?;
        let z = m.remove("video").or_else(|| m.remove("latent")).ok_or("no `video` or `latent`")?;
        let up = ltx_upsample::Upsampler::load(&path, &vae, &device, DType::F32)?;
        let y = up.forward(&z.to_device(&device)?)?.to_device(&Device::Cpu)?;
        eprintln!("{:?} to {:?}", z.dims(), y.dims());
        candle_core::safetensors::save(&[("latent", y)].into_iter().collect(), &save)?;
        return Ok(());
    }
    let dir = value("--fixtures").ok_or("--fixtures DIR is required")?;
    let (device, dtype) = match (flag("--cpu"), flag("--f32")) {
        (true, _) => (Device::Cpu, DType::F32),
        (false, true) => (Device::new_metal(0)?, DType::F32),
        _ => (Device::new_metal(0)?, DType::BF16),
    };
    let mut f32s = candle_core::safetensors::load(format!("{dir}/upsample_f32.safetensors"), &Device::Cpu)?;
    let latent = f32s.remove("latent").ok_or("no `latent`")?;
    let want = f32s.remove("upsampled").ok_or("no `upsampled`")?;

    let t = Instant::now();
    let up = ltx_upsample::Upsampler::load(&path, &vae, &device, dtype)?;
    eprintln!("upsampler: {:.0} M parameters, {dtype:?} on {}, loaded in {:.1} s", up.params() as f64 / 1e6, if device.is_cpu() { "the CPU" } else { "Metal" }, t.elapsed().as_secs_f64());
    for run in 0..3 {
        let t = Instant::now();
        let got = up.forward(&latent.to_device(&device)?)?;
        device.synchronize()?;
        let took = t.elapsed().as_secs_f64();
        if run == 0 {
            eprintln!("{:?} to {:?}: {:.1} dB against the reference's f32", latent.dims(), got.dims(), db(&got, &want)?);
        }
        eprintln!("   in {took:.2} s{}", if run == 0 { " (first run)" } else { "" });
    }
    if let Ok(mut r) = candle_core::safetensors::load(format!("{dir}/upsample_bf16.safetensors"), &Device::Cpu) {
        let theirs = r.remove("upsampled").ok_or("no `upsampled`")?;
        eprintln!("the reference's own bf16 on MPS: {:.1} dB against its f32", db(&theirs, &want)?);
    }
    Ok(())
}
