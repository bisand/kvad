//! DFR's canvas and its first two stages, checked against the reference's
//! own `DiffusionStage` on the same noise.
//!
//!     cargo run --release -p kvad-gpu --example ltx_dfr -- --fixtures DIR [--only N] [--detailing LORA]
//!
//! `DIR` is what `scripts/ltx-fixtures.py --dit … --contexts random --dfr
//! --upsampler … --vae …` wrote: a 512×320 × 49 clip at 48 fps, so keyframes
//! at 24 and 48 and the DiT told 60 fps; the reference's stage 1 at 256×160
//! with the DiT cut to its first two blocks, its video and keyframes
//! upsampled, and its stage 2, with the noise each drew: both by plain
//! Euler, η 0, as the reference's 1.3 ran them and as the fixtures' script
//! still does, where Kvad's pipeline now runs them at η 1 as its 1.4.0 does.
//! The ancestral step itself is checked by the rounds. `--detailing` is
//! the detailing IC-LoRA, when the fixtures were made with it. With
//! `--temporal` there too, two temporal rounds follow: 97 frames at 96 fps
//! in two tiles, then 193 at 192 in four.
//!
//! 1. **The canvas** the reference laid out.
//! 2. **f32, CPU**: both stages and the upsampling between them, each
//!    against the reference's f32; stage 2 from Kvad's own stage 1, and
//!    each round from Kvad's own stage before it.
//! 3. **bf16, Metal**, the upsampler in f32 as Kvad runs it, and the
//!    reference's own bf16 on MPS, both against its f32.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::ltx_dfr::{self, Detailing, Staged};
use kvad_gpu::video::ltx_dit::{Dit, Shape};
use kvad_gpu::video::ltx_text::{Contexts, DIT_FILE};
use kvad_gpu::video::{ltx_upsample, ltx_vae, LTX_REPO};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Signal-to-error ratio in dB of `got` against `want`.
fn db(got: &Tensor, want: &Tensor) -> Res<f32> {
    let cpu = |t: &Tensor| -> candle_core::Result<Tensor> { t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all() };
    let (g, w) = (cpu(got)?, cpu(want)?);
    if g.dims() != w.dims() {
        return Err(format!("{:?} against {:?}", got.dims(), want.dims()).into());
    }
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
    let detailing = value("--detailing").map(PathBuf::from);
    let dir = value("--fixtures").ok_or("--fixtures DIR is required")?;
    let file = |name: &str| -> Res<HashMap<String, Tensor>> { Ok(candle_core::safetensors::load(format!("{dir}/{name}"), &Device::Cpu)?) };
    let get = |m: &HashMap<String, Tensor>, k: &str| -> Res<Tensor> { Ok(m.get(k).ok_or_else(|| format!("no `{k}`"))?.clone()) };
    let (inputs, want) = (file("dit_inputs.safetensors")?, file("dfr_f32.safetensors")?);
    let s = get(&want, "shape")?.to_vec1::<f32>()?;
    let (width, height, fps) = (s[0] as usize, s[1] as usize, s[3] as f64);
    let downscale = get(&want, "downscale")?.to_vec1::<f32>()?[0] as usize;

    // 1. The canvas.
    let canvas = ltx_dfr::canvas(s[2] as usize)?;
    let theirs: Vec<usize> = get(&want, "positions")?.to_vec1::<f32>()?.iter().map(|&p| p as usize).collect();
    if step(1) {
        let same = canvas.keyframes == theirs && canvas.frames == s[2] as usize;
        eprintln!("1. canvas: {} frames in segments of {}, keyframes at {:?}: {}", canvas.frames, canvas.segment, canvas.keyframes, if same { "the reference's" } else { "DIFFERENT" });
        if !same {
            return Err(format!("the reference's keyframes are at {theirs:?}").into());
        }
    }
    if let Some(l) = &detailing {
        let d = ltx_dfr::reference_downscale(l)?;
        if d != downscale {
            return Err(format!("the LoRA says a reference downscale of {d}, the fixtures {downscale}").into());
        }
    }

    // The video at the frame rate the DiT is told, the sound at the clip's.
    let cfps = ltx_dfr::conditioning_fps(fps);
    let (half, full) = (Shape::new(width / 2, height / 2, canvas.frames, cfps)?, Shape::new(width, height, canvas.frames, cfps)?);
    let sound = Shape::new(width, height, canvas.frames, fps)?;
    let contexts = [get(&inputs, "video_context")?, get(&inputs, "audio_context")?];
    let path = fetch_file(LTX_REPO, DIT_FILE, &Watcher::none())?;
    let up_path = fetch_file(LTX_REPO, ltx_upsample::FILE, &Watcher::none())?;
    let rounds = if want.contains_key("round_1_video") { 2 } else { 0 };
    let temporal = match rounds {
        0 => None,
        _ => Some(fetch_file(LTX_REPO, ltx_upsample::TEMPORAL_FILE, &Watcher::none())?),
    };
    let vae = fetch_file(LTX_REPO, ltx_vae::FILE, &Watcher::none())?;
    let blocks = 2;
    let run = |device: &Device, dtype: DType, label: &str, n: usize| -> Res<()> {
        // The noise as the reference drew it, in its latents' dtype.
        let noise = |i: usize| -> Res<Vec<f32>> {
            Ok(get(&want, &format!("noise_{i}"))?.to_dtype(dtype)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
        };
        // Every draw in the reference's order, stages and rounds alike.
        let mut drawn = 0;
        let mut replay = |dims: &[usize]| -> Res<Vec<f32>> {
            let n = noise(drawn)?;
            drawn += 1;
            match n.len() == dims.iter().product::<usize>() {
                true => Ok(n),
                false => Err(format!("draw {} is {} numbers, where {dims:?} were asked", drawn - 1, n.len()).into()),
            }
        };
        let ctx = Contexts { video: contexts[0].to_device(device)?, audio: contexts[1].to_device(device)? };
        let t = Instant::now();
        let dit = Dit::load_as(&path, None, "transformer-check", device, dtype, Some(blocks), None, &mut |_| {})?;
        eprintln!("{n}. {label}: {:.2} B parameters in {:.1} s", dit.params() as f64 / 1e9, t.elapsed().as_secs_f64());
        let say = |name: &str, got: &Staged, stage: usize| -> Res<()> {
            let w = |k: &str| get(&want, &format!("stage_{stage}_{k}"));
            let keys = got.keyframes.as_ref().ok_or("no keyframes")?;
            eprintln!("   {name}: video {:5.1} dB, keyframes {:5.1} dB, sound {:5.1} dB", db(&got.video, &w("video")?)?, db(keys, &w("keyframes")?)?, db(&got.audio, &w("audio")?)?);
            Ok(())
        };
        let t = Instant::now();
        let one = ltx_dfr::first(&dit, &ctx, half, sound, &canvas.keyframes, 0.0, None, &mut replay, &mut |_, _, _| Ok(()))?;
        device.synchronize()?;
        eprintln!("   stage 1 in {:.2} s", t.elapsed().as_secs_f64());
        say("stage 1", &one, 1)?;

        // Kvad's upsampler runs in f32 wherever the DiT runs.
        let up = ltx_upsample::Upsampler::load(&up_path, &vae, device, DType::F32)?;
        let upsampled = up.forward(&one.video.to_device(device)?)?;
        let keys = up.forward(&one.keyframes.as_ref().ok_or("no keyframes")?.to_device(device)?)?;
        eprintln!("   upsampled: video {:5.1} dB, keyframes {:5.1} dB", db(&upsampled, &get(&want, "upsampled_video")?)?, db(&keys, &get(&want, "upsampled_keyframes")?)?);
        drop(up);

        let fused;
        let second = match &detailing {
            Some(l) => {
                fused = Dit::load_as(&path, Some((l, ltx_dfr::DETAILING_STRENGTH as f64)), "transformer-check", device, dtype, Some(blocks), None, &mut |_| {})?;
                &fused
            }
            None => &dit,
        };
        let from = Detailing { upsampled: &upsampled, keyframes: &keys, reference: &one.video, audio: &one.audio };
        let t = Instant::now();
        let two = ltx_dfr::second(second, &ctx, full, sound, &canvas.keyframes, &from, downscale, 0.0, None, &mut replay, &mut |_, _, _| Ok(()))?;
        device.synchronize()?;
        eprintln!("   stage 2 in {:.2} s", t.elapsed().as_secs_f64());
        say("stage 2", &two, 2)?;

        // The rounds, on the reference's draws after the stages' four.
        let Some(temporal) = &temporal else { return Ok(()) };
        let up = ltx_upsample::Upsampler::load(temporal, &vae, device, DType::F32)?;
        let mut clip = ltx_dfr::Clip {
            video: two.video,
            keyframes: two.keyframes.ok_or("no keyframes")?,
            positions: canvas.keyframes.clone(),
            frames: canvas.frames,
            fps,
        };
        for r in 1..=rounds {
            let t = Instant::now();
            clip = ltx_dfr::round(&dit, &ctx, &up, &clip, r, &one.audio, canvas.frames as f64 / fps, None, &mut replay, &mut |_, _, _, _, _| Ok(()))?;
            device.synchronize()?;
            let w = |k: &str| get(&want, &format!("round_{r}_{k}"));
            let theirs: Vec<usize> = w("positions")?.to_vec1::<f32>()?.iter().map(|&p| p as usize).collect();
            eprintln!("   round {r} in {:.2} s: {} frames at {} fps, keyframes at {:?}{}", t.elapsed().as_secs_f64(), clip.frames, clip.fps, clip.positions,
                      if clip.positions == theirs { "" } else { " (DIFFERENT)" });
            eprintln!("   round {r}: video {:5.1} dB, keyframes {:5.1} dB", db(&clip.video, &w("video")?)?, db(&clip.keyframes, &w("keyframes")?)?);
        }
        if drawn != want.keys().filter(|k| k.starts_with("noise_")).count() {
            return Err(format!("{drawn} draws used of the reference's {}", want.keys().filter(|k| k.starts_with("noise_")).count()).into());
        }
        Ok(())
    };
    if step(2) {
        run(&Device::Cpu, DType::F32, "f32 on the CPU", 2)?;
    }
    if step(3) {
        run(&Device::new_metal(0)?, DType::BF16, "bf16 on Metal", 3)?;
        if let Ok(r) = file("dfr_bf16.safetensors") {
            let d = |k: &str| -> Res<f32> { db(&get(&r, k)?, &get(&want, k)?) };
            eprintln!("   the reference's own bf16 on MPS:");
            eprintln!("   stage 1: video {:5.1} dB, keyframes {:5.1} dB, sound {:5.1} dB", d("stage_1_video")?, d("stage_1_keyframes")?, d("stage_1_audio")?);
            eprintln!("   upsampled: video {:5.1} dB, keyframes {:5.1} dB", d("upsampled_video")?, d("upsampled_keyframes")?);
            eprintln!("   stage 2: video {:5.1} dB, keyframes {:5.1} dB, sound {:5.1} dB", d("stage_2_video")?, d("stage_2_keyframes")?, d("stage_2_audio")?);
            for r in 1..=rounds {
                eprintln!("   round {r}: video {:5.1} dB, keyframes {:5.1} dB", d(&format!("round_{r}_video"))?, d(&format!("round_{r}_keyframes"))?);
            }
        }
    }
    Ok(())
}
