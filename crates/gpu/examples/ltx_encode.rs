//! LTX-2.5's video encoder on one picture, checked against the reference's.
//!
//!     cargo run --release -p kvad-gpu --example ltx_encode -- --fixtures DIR
//!
//! `DIR` is what `scripts/ltx-fixtures.py --video … --picture …` wrote: a
//! picture after the reference's H.264 round trip, what the reference's own
//! scaling and cutting made of it at stage 1's size and stage 2's, and what
//! its encoder made of those, in f32 on the CPU and in bf16 on MPS. In turn:
//!
//! 1. **The picture**, scaled and cut here ([`ltx_vae::picture`]), against
//!    the reference's pixels.
//! 2. **f32, CPU**: the encoder on the reference's pixels, against its
//!    latents.
//! 3. **bf16, Metal**, as the pipeline runs it; and the reference's own bf16
//!    against its f32, for the drift bf16 costs it.
//!
//! With `--picture FILE`, the picture the fixture was made from, step 0 puts
//! it through kvad's H.264 round trip (`kvad::video::picture_from_file`, by
//! `ffmpeg`) and compares that with the reference's (by PyAV).
//!
//! `--only N` runs just step N; `--f32` runs step 3 in f32.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::{ltx_vae, LTX_REPO};
use std::collections::HashMap;
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
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let only: Option<usize> = value("--only").map(|v| v.parse()).transpose()?;
    let step = |n: usize| only.is_none_or(|o| o == n);
    let dir = value("--fixtures").ok_or("--fixtures DIR is required")?;
    let file = |name: &str| -> Res<HashMap<String, Tensor>> { Ok(candle_core::safetensors::load(format!("{dir}/{name}"), &Device::Cpu)?) };
    let want = file("picture_f32.safetensors")?;
    let get = |m: &HashMap<String, Tensor>, k: &str| -> Res<Tensor> { Ok(m.get(k).ok_or_else(|| format!("no `{k}`"))?.clone()) };
    let path = fetch_file(LTX_REPO, ltx_vae::FILE, &Watcher::none())?;
    let sizes = [(384, 256), (768, 512)];

    if let (true, Some(source)) = (step(0), value("--picture")) {
        let ffmpeg = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"].into_iter().find(|p| std::path::Path::new(p).is_file()).ok_or("no ffmpeg")?;
        let mine = kvad::video::picture_from_file(ffmpeg.as_ref(), source.as_ref(), kvad::video::PICTURE_CRF)?;
        let theirs = get(&want, "rgb")?;
        let (h, w, _) = theirs.dims3()?;
        if (mine.width, mine.height) != (w, h) {
            return Err(format!("0. {}×{} from ffmpeg, {w}×{h} from the reference", mine.width, mine.height).into());
        }
        let mine = Tensor::from_vec(mine.rgb, (h, w, 3), &Device::Cpu)?.to_dtype(DType::F32)?;
        let theirs = theirs.to_dtype(DType::F32)?;
        let diff = (&mine - &theirs)?.abs()?;
        let most = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean = diff.mean_all()?.to_scalar::<f32>()?;
        let psnr = 10.0 * (255f32 * 255.0 / (&mine - &theirs)?.sqr()?.mean_all()?.to_scalar::<f32>()?.max(1e-12)).log10();
        eprintln!("0. the H.264 round trip, {w}×{h}: {psnr:.1} dB PSNR from the reference's; {mean:.2} levels apart on average, {most} at most");
    }

    if step(1) {
        let rgb = get(&want, "rgb")?;
        let (h, w, _) = rgb.dims3()?;
        let bytes = rgb.flatten_all()?.to_vec1::<u8>()?;
        for (sw, sh) in sizes {
            let mine = ltx_vae::picture(&bytes, w, h, sw, sh)?;
            let theirs = get(&want, &format!("pixels_{sw}x{sh}"))?;
            let most = (&mine - &theirs)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            eprintln!("1. {w}×{h} to {sw}×{sh}: {:.1} dB, at most {:.2} of an 8-bit level apart", db(&mine, &theirs)?, most * 127.5);
        }
    }

    let run = |device: &Device, dtype: DType, label: &str, n: usize| -> Res<()> {
        let t = Instant::now();
        let enc = ltx_vae::ImageEncoder::load(&path, device, dtype)?;
        eprintln!("{n}. {label}: {:.0} M parameters in {:.1} s", enc.params() as f64 / 1e6, t.elapsed().as_secs_f64());
        let bf16 = file("picture_bf16.safetensors").ok();
        for (sw, sh) in sizes {
            let pixels = get(&want, &format!("pixels_{sw}x{sh}"))?.to_device(device)?;
            let t = Instant::now();
            let z = enc.encode(&pixels)?;
            device.synchronize()?;
            let theirs = get(&want, &format!("latent_{sw}x{sh}"))?;
            let mut line = format!("   {sw}×{sh}: {:.1} dB in {:.2} s", db(&z, &theirs)?, t.elapsed().as_secs_f64());
            if let (DType::BF16, Some(r)) = (dtype, &bf16) {
                line += &format!("; the reference's own bf16 on MPS {:.1} dB", db(&get(r, &format!("latent_{sw}x{sh}"))?, &theirs)?);
            }
            eprintln!("{line}");
        }
        Ok(())
    };
    if step(2) {
        run(&Device::Cpu, DType::F32, "f32 on the CPU", 2)?;
    }
    if step(3) {
        // `--f32`: Metal's kernels in f32, which should be as exact as the CPU.
        match argv.iter().any(|a| a == "--f32") {
            true => run(&Device::new_metal(0)?, DType::F32, "f32 on Metal", 3)?,
            false => run(&Device::new_metal(0)?, DType::BF16, "bf16 on Metal", 3)?,
        }
    }
    Ok(())
}
