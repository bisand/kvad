//! LTX-2.5 from a prompt to an MP4 with sound, with no server: one stage of
//! the distilled model at the requested size (`docs/video-plan.md`, step 5).
//!
//!     cargo run --release -p kvad-gpu --example ltx -- --prompt "…" \
//!         [--width 512] [--height 320] [--frames 25] [--fps 24] [--seed 0] \
//!         [--quant q8|bf16] [--out ltx.mp4] [--latents FILE]
//!
//! In phases, so that no two large models are resident at once:
//!
//! 1. the text path (Gemma 4, projections, connectors) encodes the prompt,
//!    and is dropped;
//! 2. the DiT runs stage 1's eight steps, and is dropped;
//! 3. the video decoder and the audio path decode, and the MP4 is written.
//!
//! `--quant` is the text path's and the DiT's weights: q8 by default, about
//! 19 GB for the text phase and 20 GB for the DiT. `--latents` also saves the
//! two latents, to decode again without the DiT.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::ltx_dit::{Dit, Shape};
use kvad_gpu::video::ltx_sample::{one_stage, STAGE_1};
use kvad_gpu::video::ltx_text::{Contexts, TextEncoder, DIT_FILE, TEXT_FILE};
use kvad_gpu::video::{ltx_audio, ltx_vae, LTX_REPO};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let num = |f: &str, default: usize| -> Res<usize> { Ok(value(f).map(|v| v.parse()).transpose()?.unwrap_or(default)) };
    let prompt = value("--prompt").ok_or("--prompt TEXT is required")?;
    let fps: f64 = value("--fps").map(|v| v.parse()).transpose()?.unwrap_or(24.0);
    let shape = Shape::new(num("--width", 512)?, num("--height", 320)?, num("--frames", 25)?, fps)?;
    let seed = num("--seed", 0)? as u64;
    let quant = match value("--quant").as_deref() {
        Some("bf16") => None,
        None => kvad_gpu::model::parse_quant("q8").ok_or("no q8")?,
        Some(q) => kvad_gpu::model::parse_quant(q).ok_or("--quant is bf16 or q8")?,
    };
    let out = value("--out").unwrap_or_else(|| "ltx.mp4".into());
    let device = Device::new_metal(0)?;
    let dtype = DType::BF16;
    let started = Instant::now();
    let mut say = |m: &str| eprintln!("   {m}");
    let fetch = |f: &str| fetch_file(LTX_REPO, f, &Watcher::none());
    let (text, dit_path) = (fetch(TEXT_FILE)?, fetch(DIT_FILE)?);
    eprintln!(
        "{}×{}, {} frames at {fps} fps ({:.2} s): {} video tokens, {} audio latents; seed {seed}",
        shape.width,
        shape.height,
        shape.frames,
        shape.frames as f64 / fps,
        shape.video_tokens(),
        shape.audio_latents()
    );

    // 1. The prompt.
    let ctx = {
        let t = Instant::now();
        let enc = TextEncoder::load(&text, &dit_path, &device, dtype, quant, &mut say)?;
        eprintln!("1. text path: {:.1} B parameters, loaded in {:.1} s", enc.params() as f64 / 1e9, t.elapsed().as_secs_f64());
        let t = Instant::now();
        let c = enc.encode(&prompt)?;
        device.synchronize()?;
        eprintln!("   {} tokens encoded in {:.2} s", enc.tokens(&prompt)?.len(), t.elapsed().as_secs_f64());
        c
    };
    // The text path is dropped, but candle's Metal pool only lets go of a
    // dropped tensor's buffer when the device is next synchronised. Without
    // this the text path's 13 GB stayed resident beside the DiT's 20 GB,
    // and 768×512 × 121 ran out of memory in its first step at 37 GB.
    device.synchronize()?;

    // 2. The latents.
    let latents = {
        let t = Instant::now();
        let dit = Dit::load(&dit_path, &device, dtype, None, quant, &mut say)?;
        eprintln!("2. DiT: {:.1} B parameters, loaded in {:.1} s", dit.params() as f64 / 1e9, t.elapsed().as_secs_f64());
        let grid = dit.grid(shape)?;
        let ctx = Contexts { video: ctx.video.clone(), audio: ctx.audio.clone() };
        let t = Instant::now();
        let mut last = Instant::now();
        let l = one_stage(&dit, &ctx, &grid, seed, &mut |i, sigma| {
            device.synchronize()?;
            eprintln!("   step {} of {}: σ {sigma:.4} in {:.2} s", i + 1, STAGE_1.len() - 1, last.elapsed().as_secs_f64());
            last = Instant::now();
            Ok(())
        })?;
        eprintln!("   sampled in {:.1} s", t.elapsed().as_secs_f64());
        l
    };
    // And the DiT's 20 GB, before the decoders need room for full-size frames.
    device.synchronize()?;
    for (name, t) in [("video", &latents.video), ("audio", &latents.audio)] {
        let s = t.to_device(&Device::Cpu)?.flatten_all()?;
        let (mean, rms) = (s.mean_all()?.to_scalar::<f32>()?, s.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt());
        if !rms.is_finite() {
            return Err(format!("the {name} latent is not finite").into());
        }
        eprintln!("   {name} latent {:?}: mean {mean:.3}, rms {rms:.3}", t.dims());
    }
    if let Some(f) = value("--latents") {
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu);
        candle_core::safetensors::save(&[("video", cpu(&latents.video)?), ("audio", cpu(&latents.audio)?)].into_iter().collect(), &f)?;
    }
    drop(ctx);

    // 3. Pictures and sound.
    let t = Instant::now();
    let frames = ltx_vae::VideoDecoder::load(&fetch(ltx_vae::FILE)?, &device, dtype)?.decode(&latents.video)?.to_device(&Device::Cpu)?;
    eprintln!("3. video decoded to {:?} in {:.1} s", frames.dims(), t.elapsed().as_secs_f64());
    let t = Instant::now();
    let sound = ltx_audio::AudioPath::load(&fetch(ltx_audio::FILE)?, &device)?.decode(&latents.audio)?;
    eprintln!("   audio decoded to {:.2} s at {} Hz in {:.1} s", sound.samples.len() as f64 / (sound.rate as f64 * sound.channels as f64), sound.rate, t.elapsed().as_secs_f64());
    let video = ltx_vae::to_video(&frames, fps.round() as u32)?;
    std::fs::write(&out, video.mp4(Some(&sound)))?;
    eprintln!("wrote {out} in {:.1} s all told", started.elapsed().as_secs_f64());
    Ok(())
}
