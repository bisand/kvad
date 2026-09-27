//! LTX-2.5's diffusion decoder (DiffVAE), checked against the reference's.
//!
//!     cargo run --release -p kvad-gpu --example ltx_diffvae -- --fixtures DIR [--cpu | --f32]
//!
//! `DIR` is what `scripts/ltx-fixtures.py --diffvae …` wrote: a seeded
//! 3 × 8 × 8 latent, and what the reference's own decoder made of it at each
//! stage on its eager neighbourhood-attention path, in f32 on the CPU, with
//! the stage-5 noise it started from. Each stage here runs from the
//! reference's input to it, so an error shows in the stage that makes it:
//!
//! 1. stages 1 to 3, the latent to stage 4's input;
//! 2. stage 4, to the context;
//! 3. stage 5, the context and the noise to the frames.
//!
//! With `--keyframes`, the keyframe-aware decode instead, against what
//! `--diffvae … --keyframes` wrote: the same latent and two keyframe planes
//! at pixel frames 8 and 16, each stage's video and planes compared.
//!
//! On Metal in bf16 by default; `--f32` keeps Metal in f32, `--cpu` runs on
//! the CPU in f32, the closest to the reference's arithmetic. `--where`
//! prints where the file is (downloading it first).
//!
//! A whole clip instead, from a latent the pipeline made:
//!
//!     cargo run --release -p kvad-gpu --example ltx_diffvae -- --latent z.safetensors \
//!         [--seed N] [--budget TOKENS] [--against TOKENS] [--out clip.mp4]
//!
//! `z.safetensors` holds `latent`, `[128, F, h, w]`. `--budget` is the most
//! stage-5 tokens a tile may have (`ltx_diffvae::BUDGET` by default);
//! `--against` decodes the clip a second time under another budget and says
//! how far apart the two are, which is what tiling costs. `--profile` says
//! where the time went ([`kvad_gpu::prof`]). `--keyframes-at 24,48,…`
//! decodes with keyframe planes at those pixel frames, standing in for
//! DFR's the latent's own frames there: a measure of what planes cost, not
//! of what DFR's make.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::ltx_diffvae::{DiffDecoder, Grid, Planes, BUDGET, FILE};
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
    let flag = |f: &str| argv.iter().any(|a| a == f);
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let path = fetch_file(LTX_REPO, FILE, &Watcher::none())?;
    if flag("--where") {
        println!("{}", path.display());
        return Ok(());
    }
    let (device, dtype) = match (flag("--cpu"), flag("--f32")) {
        (true, _) => (Device::Cpu, DType::F32),
        (false, f32) => (Device::new_metal(0)?, if f32 { DType::F32 } else { DType::BF16 }),
    };
    let t = Instant::now();
    let dec = DiffDecoder::load(&path, &device, dtype)?;
    eprintln!("decoder: {:.0} M parameters in {dtype:?} on {device:?}, loaded in {:.1} s", dec.params() as f64 / 1e6, t.elapsed().as_secs_f64());

    if let Some(file) = value("--latent") {
        return clip(&dec, &device, &file, &value);
    }
    let dir = value("--fixtures").ok_or("--fixtures DIR or --latent FILE is required")?;
    if flag("--keyframes") {
        return keyframes(&dec, &device, dtype, &dir);
    }
    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(format!("{dir}/diffvae_f32.safetensors"), &Device::Cpu)?;
    let get = |k: &str| -> Res<Tensor> { Ok(fx.get(k).ok_or_else(|| format!("no `{k}`"))?.to_device(&device)?) };
    // The reference's own bf16, each stage from the same f32 input, when the
    // fixture has it: what bf16 costs the reference itself.
    let bf16: Option<HashMap<String, Tensor>> = candle_core::safetensors::load(format!("{dir}/diffvae_bf16.safetensors"), &Device::Cpu).ok();
    let theirs = |k: &str, want: &Tensor| -> Res<String> {
        match (dtype, &bf16) {
            (DType::BF16, Some(b)) => {
                let t = b.get(k).ok_or_else(|| format!("no bf16 `{k}`"))?;
                let t = match k {
                    "pixels" => t.clone(),
                    _ => {
                        let (tt, h, w, c) = t.dims4()?;
                        t.reshape((tt * h * w, c))?
                    }
                };
                Ok(format!("; the reference's own bf16 on MPS {:.1} dB", db(&t, want)?))
            }
            _ => Ok(String::new()),
        }
    };
    // The reference keeps features channels-last, [T, H, W, C]: as tokens,
    // [T·H·W, C], and a grid.
    let tokens = |t: Tensor| -> Res<(Tensor, Grid)> {
        let (tt, h, w, c) = t.dims4()?;
        Ok((t.reshape((tt * h * w, c))?, Grid { t: tt, h, w }))
    };
    let sync = |t: &Tensor| t.device().synchronize();

    // 1. The latent, its last frame repeated twice as the reference's
    // border workaround does, to stage 4's input.
    let z = get("latent")?;
    let f = z.dim(1)?;
    let last = z.narrow(1, f - 1, 1)?;
    let padded = Tensor::cat(&[&z, &last, &last], 1)?;
    let t = Instant::now();
    let (feat, g4) = dec.stages_1_to_3(&padded)?;
    sync(&feat)?;
    let (want, wg) = tokens(get("stage4_input")?)?;
    if g4 != wg {
        return Err(format!("stages 1-3 made {g4:?}; the reference {wg:?}").into());
    }
    eprintln!("1. stages 1-3 to {g4:?}: {:.1} dB, {:.1} s{}", db(&feat, &want)?, t.elapsed().as_secs_f64(), theirs("stage4_input", &want)?);

    // 2. Stage 4, from the reference's input to it.
    let t = Instant::now();
    let (ctx, g5) = dec.stage_4(&want.to_dtype(dtype)?, wg, 2)?;
    sync(&ctx)?;
    let (want_ctx, wg5) = tokens(get("context")?)?;
    if g5 != wg5 {
        return Err(format!("stage 4 made {g5:?}; the reference {wg5:?}").into());
    }
    eprintln!("2. stage 4 to {g5:?}: {:.1} dB, {:.1} s{}", db(&ctx, &want_ctx)?, t.elapsed().as_secs_f64(), theirs("context", &want_ctx)?);

    // 3. Stage 5, from the reference's context and noise.
    let t = Instant::now();
    let pixels = dec.stage_5(&want_ctx.to_dtype(dtype)?, wg5, &get("noise")?)?;
    sync(&pixels)?;
    let want_px = get("pixels")?;
    eprintln!("3. stage 5 to {:?}: {:.1} dB, {:.1} s{}", pixels.dims(), db(&pixels, &want_px)?, t.elapsed().as_secs_f64(), theirs("pixels", &want_px)?);
    // In 8-bit levels, as a video would show it.
    let most = (pixels.to_dtype(DType::F32)? - want_px.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    eprintln!("   at most {:.2} of an 8-bit level apart", most * 127.5);
    Ok(())
}

/// A whole clip: decoded under `--budget`, and with `--against` again under
/// another, the two compared.
fn clip(dec: &DiffDecoder, device: &Device, file: &str, value: &dyn Fn(&str) -> Option<String>) -> Res<()> {
    let mut latent = candle_core::safetensors::load(file, &Device::Cpu)?.remove("latent").ok_or("no `latent` tensor")?;
    if latent.rank() == 5 {
        latent = latent.squeeze(0)?;
    }
    let latent = latent.to_device(device)?;
    let seed = value("--seed").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let at: Vec<usize> = match value("--keyframes-at") {
        Some(v) => v.split(',').map(|p| p.trim().parse()).collect::<Result<_, _>>()?,
        None => vec![],
    };
    let planes = match at.is_empty() {
        true => None,
        false => {
            let pick = Tensor::from_vec(at.iter().map(|&p| (p / 8) as u32).collect::<Vec<_>>(), at.len(), device)?;
            Some(latent.index_select(&pick, 1)?)
        }
    };
    let keys = planes.as_ref().map(|p| (p, at.as_slice()));
    if !at.is_empty() {
        eprintln!("with {} keyframe planes, at pixel frames {at:?}", at.len());
    }
    let run = |budget: usize| -> Res<Tensor> {
        let t = Instant::now();
        let (frames, r) = dec.decode_keyed(&latent, keys, seed, budget, &mut |i, n| Ok(eprint!("\r   tile {i} of {n}")))?;
        eprintln!();
        let (n, _, h, w) = frames.dims4()?;
        eprintln!("{:?} to {n} frames of {w}×{h} in {:.1} s: stages 1-3 {:.1} s, 4-5 {:.1} s; tiles {:?}, {:.2} M stage-5 tokens, the largest {:.2} M",
                  latent.dims(), t.elapsed().as_secs_f64(), r.stages_1_to_3, r.stages_4_to_5, r.tiles, r.tokens as f64 / 1e6, r.largest as f64 / 1e6);
        Ok(frames)
    };
    let budget = value("--budget").map(|s| s.parse()).transpose()?.unwrap_or(BUDGET);
    let profile = value("--profile").is_some() || std::env::args().any(|a| a == "--profile");
    if profile {
        kvad_gpu::prof::start();
    }
    let frames = run(budget)?;
    for r in kvad_gpu::prof::stop() {
        eprintln!("   {:<34} {:>4}× {:>7.2} s", r.label, r.calls, r.seconds);
    }
    if let Some(other) = value("--against") {
        let theirs = run(other.parse()?)?;
        let d = (&frames - &theirs)?;
        let mse = d.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let worst = d.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        eprintln!("the two: PSNR {:.1} dB, {:.1} dB against the frames' own power; at most {:.1} of an 8-bit level apart",
                  10.0 * (1.0 / mse.max(1e-20)).log10(), db(&theirs, &frames)?, worst * 255.0);
    }
    if let Some(out) = value("--out") {
        let video = ltx_vae::to_video(&frames, 24)?;
        std::fs::write(&out, video.mp4(None))?;
        eprintln!("{out}: {} frames", video.frames());
    }
    Ok(())
}

/// The keyframe-aware decode, stage by stage, each from the reference's
/// input to it, its video and its planes compared.
fn keyframes(dec: &DiffDecoder, device: &Device, dtype: DType, dir: &str) -> Res<()> {
    let fx: HashMap<String, Tensor> = candle_core::safetensors::load(format!("{dir}/diffvae_kf_f32.safetensors"), &Device::Cpu)?;
    let bf16: Option<HashMap<String, Tensor>> = candle_core::safetensors::load(format!("{dir}/diffvae_kf_bf16.safetensors"), &Device::Cpu).ok();
    let get = |k: &str| -> Res<Tensor> { Ok(fx.get(k).ok_or_else(|| format!("no `{k}`"))?.to_device(device)?) };
    let tokens = |t: Tensor| -> Res<(Tensor, Grid)> {
        let (tt, h, w, c) = t.dims4()?;
        Ok((t.reshape((tt * h * w, c))?, Grid { t: tt, h, w }))
    };
    // What the reference's own bf16 made of the same, when there is one.
    let theirs = |k: &str, want: &Tensor| -> Res<String> {
        match (dtype, &bf16) {
            (DType::BF16, Some(b)) => {
                let t = b.get(k).ok_or_else(|| format!("no bf16 `{k}`"))?;
                let t = if k == "pixels" { t.clone() } else { tokens(t.clone())?.0 };
                Ok(format!("; the reference's own bf16 {:.1} dB", db(&t, want)?))
            }
            _ => Ok(String::new()),
        }
    };
    let frames: Vec<usize> = get("indices")?.to_vec1::<f32>()?.iter().map(|&f| f as usize).collect();

    // 1. Stages 1-3, both streams from the latents.
    let z = get("latent")?;
    let f = z.dim(1)?;
    let last = z.narrow(1, f - 1, 1)?;
    let padded = Tensor::cat(&[&z, &last, &last], 1)?;
    let t = Instant::now();
    let (feat, g4, planes) = dec.stages_1_to_3_keyed(&padded, dec.planes(&get("planes")?, &frames)?)?;
    device.synchronize()?;
    let (want, wg) = tokens(get("stage4_input")?)?;
    let (pwant, pwg) = tokens(get("kf_stage4_input")?)?;
    if g4 != wg || planes.grid != pwg {
        return Err(format!("stages 1-3 made {g4:?} and {:?}; the reference {wg:?} and {pwg:?}", planes.grid).into());
    }
    eprintln!("1. stages 1-3 to {g4:?}, {:.1} s", t.elapsed().as_secs_f64());
    eprintln!("   video  {:5.1} dB{}", db(&feat, &want)?, theirs("stage4_input", &want)?);
    eprintln!("   planes {:5.1} dB{}", db(&planes.x, &pwant)?, theirs("kf_stage4_input", &pwant)?);

    // 2. Stage 4, from the reference's inputs.
    let t = Instant::now();
    let given = Planes { x: pwant.to_dtype(dtype)?, grid: pwg, frames: frames.clone() };
    let (ctx, g5, pctx) = dec.stage_4_keyed(&want.to_dtype(dtype)?, wg, 2, given)?;
    device.synchronize()?;
    let (want_ctx, wg5) = tokens(get("context")?)?;
    let (pwant_ctx, pwg5) = tokens(get("kf_context")?)?;
    if g5 != wg5 || pctx.grid != pwg5 {
        return Err(format!("stage 4 made {g5:?} and {:?}; the reference {wg5:?} and {pwg5:?}", pctx.grid).into());
    }
    let times = dec.times(&frames, 4, 0.0);
    let same = times == get("kf_times")?.to_vec1::<f32>()?;
    eprintln!("2. stage 4 to {g5:?}, {:.1} s; the planes at stage 5's times {times:?}: {}", t.elapsed().as_secs_f64(), if same { "the reference's" } else { "DIFFERENT" });
    eprintln!("   video  {:5.1} dB{}", db(&ctx, &want_ctx)?, theirs("context", &want_ctx)?);
    eprintln!("   planes {:5.1} dB{}", db(&pctx.x, &pwant_ctx)?, theirs("kf_context", &pwant_ctx)?);

    // 3. Stage 5, from the reference's contexts and noise.
    let t = Instant::now();
    let given = Planes { x: pwant_ctx.to_dtype(dtype)?, grid: pwg5, frames: frames.clone() };
    let pixels = dec.stage_5_keyed(&want_ctx.to_dtype(dtype)?, wg5, &get("noise")?, Some((&given, &get("kf_noise")?, 0.0)))?;
    device.synchronize()?;
    let want_px = get("pixels")?;
    eprintln!("3. stage 5 to {:?}: {:.1} dB, {:.1} s{}", pixels.dims(), db(&pixels, &want_px)?, t.elapsed().as_secs_f64(), theirs("pixels", &want_px)?);
    let most = (pixels.to_dtype(DType::F32)? - want_px.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    eprintln!("   at most {:.2} of an 8-bit level apart", most * 127.5);
    // And how much the planes change the frames: the same step without them.
    let alone = dec.stage_5(&want_ctx.to_dtype(dtype)?, wg5, &get("noise")?)?;
    eprintln!("   without the planes: {:.1} dB from the reference's with them", db(&alone, &want_px)?);
    Ok(())
}
