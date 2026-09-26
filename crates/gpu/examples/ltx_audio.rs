//! Decode an LTX-2.5 audio latent to 48 kHz stereo, with no model but the
//! audio decoder, the vocoder and the bandwidth extension.
//!
//!     cargo run --release -p kvad-gpu --example ltx_audio -- \
//!         --latent z.safetensors [--expect out.safetensors] [--out sound.wav] [--cpu]
//!
//! `--latent` holds `latent`, `[8, T, 16]` (or with a leading batch axis of
//! 1). `--expect` holds the reference's `mel`, `low` (16 kHz) and `high`
//! (48 kHz) for the same latent; each stage is compared. `--cpu` runs on the
//! CPU rather than on Metal. Either way it is all f32.
//!
//! `--path FILE` loads from a file rather than the Hub, and `--where` only
//! prints where the Hub's copy is (downloading it first).

use candle_core::{DType, Device};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::{ltx_audio, LTX_REPO};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| argv.iter().any(|a| a == f);
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();

    let path = match value("--path") {
        Some(p) => p.into(),
        None => fetch_file(LTX_REPO, ltx_audio::FILE, &Watcher::none())?,
    };
    if flag("--where") {
        println!("{}", path.display());
        return Ok(());
    }
    let device = match flag("--cpu") {
        true => Device::Cpu,
        false => Device::new_metal(0)?,
    };

    let t = Instant::now();
    let audio = ltx_audio::AudioPath::load(&path, &device)?;
    eprintln!("audio path: {:.0} M parameters, loaded in {:.1} s", audio.params() as f64 / 1e6, t.elapsed().as_secs_f64());

    let file = value("--latent").ok_or("--latent FILE is required")?;
    let mut latent = candle_core::safetensors::load(&file, &Device::Cpu)?.remove("latent").ok_or("no `latent` tensor")?;
    if latent.rank() == 4 {
        latent = latent.squeeze(0)?;
    }
    let latent = latent.to_device(&device)?;

    let t = Instant::now();
    let (mel, low, high) = audio.decode_stages(&latent)?;
    let (mel, low, high) = (mel.to_device(&Device::Cpu)?, low.to_device(&Device::Cpu)?, high.to_device(&Device::Cpu)?);
    eprintln!("decoded {:?}: mel {:?}, 16 kHz {:?}, {} Hz {:?} in {:.2} s", latent.dims(), mel.dims(), low.dims(), audio.rate, high.dims(), t.elapsed().as_secs_f64());

    if let Some(expect) = value("--expect") {
        let mut want = candle_core::safetensors::load(&expect, &Device::Cpu)?;
        for (name, got) in [("mel", &mel), ("low", &low), ("high", &high)] {
            let w = want.remove(name).ok_or_else(|| format!("no `{name}` in {expect}"))?.to_dtype(DType::F32)?;
            if w.dims() != got.dims() {
                eprintln!("  {name}: shapes differ, ours {:?}, reference {:?}", got.dims(), w.dims());
                continue;
            }
            let d = (got - &w)?;
            let err = d.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let sig = w.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let worst = d.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            eprintln!("  {name}: SNR {:.1} dB against the reference, worst |Δ| {worst:.5}", 10.0 * (sig / err.max(1e-30)).log10());
        }
    }

    if let Some(expect) = value("--expect-bwe") {
        // The bandwidth extension's insides, fed our own 16 kHz sound.
        let (bmel, residual, skip) = audio.bwe_stages(&low.to_device(&device)?)?;
        if let Some(dump) = value("--dump") {
            let t = |x: &candle_core::Tensor| x.to_device(&Device::Cpu).and_then(|x| x.to_dtype(DType::F32));
            candle_core::safetensors::save(
                &[("bwe_mel".to_string(), t(&bmel)?), ("residual".to_string(), t(&residual)?), ("skip".to_string(), t(&skip)?), ("low".to_string(), t(&low)?)].into_iter().collect(),
                &dump,
            )?;
        }
        let mut want = candle_core::safetensors::load(&expect, &Device::Cpu)?;
        for (name, got) in [("bwe_mel", bmel), ("residual", residual), ("skip", skip)] {
            let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let w = want.remove(name).ok_or_else(|| format!("no `{name}` in {expect}"))?.to_dtype(DType::F32)?;
            if w.dims() != got.dims() {
                eprintln!("  {name}: shapes differ, ours {:?}, reference {:?}", got.dims(), w.dims());
                continue;
            }
            let d = (&got - &w)?;
            let err = d.sqr()?.mean_all()?.to_scalar::<f32>()?;
            let sig = w.sqr()?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("  {name}: SNR {:.1} dB against the reference", 10.0 * (sig / err.max(1e-30)).log10());
        }
    }

    if let Some(out) = value("--out") {
        let samples = high.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let wav = kvad::video::Audio { rate: audio.rate, channels: 2, samples };
        std::fs::write(&out, wav.wav())?;
        eprintln!("{out}: {:.2} s", high.dim(1)? as f64 / audio.rate as f64);
    }
    Ok(())
}
