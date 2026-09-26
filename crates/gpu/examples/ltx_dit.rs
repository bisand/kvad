//! LTX-2.5's DiT, its first blocks, checked against the reference's own.
//!
//!     cargo run --release -p kvad-gpu --example ltx_dit -- --fixtures DIR [--quant q8]
//!
//! `DIR` is what `scripts/ltx-fixtures.py --dit … --contexts …` wrote: a
//! seeded 512×320×25 latent and its sound at σ = 0.9875, the reference's
//! positions for them, and what its first two blocks and the output heads
//! make of them, in f32 on the CPU and in bf16 on MPS. In turn:
//!
//! 1. **Positions**, against the reference's, exactly.
//! 2. **f32, CPU.** Both streams after each block, and the velocities.
//! 3. **bf16, Metal** (or q8 with `--quant q8`), as the pipeline will run,
//!    against the same f32 reference; and the reference's own bf16 against
//!    its f32, for the drift bf16 costs it.
//!
//! `--held` does steps 2 and 3 as image-to-video does them: the first latent
//! frame held at σ = 0 while the rest is at the fixture's σ, against the
//! reference's `dit_held_*`.
//!
//! `--only N` runs just step N; `--blocks N` loads that many blocks (the
//! fixture's count by default); `--f32` runs step 3 in f32.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::ltx_dit::{audio_tokens, video_tokens, Dit, Shape};
use kvad_gpu::video::ltx_text::{Contexts, DIT_FILE};
use kvad_gpu::video::LTX_REPO;
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
    let path = fetch_file(LTX_REPO, DIT_FILE, &Watcher::none())?;
    let quant = match value("--quant").as_deref() {
        None | Some("bf16") => None,
        Some(q) => kvad_gpu::model::parse_quant(q).ok_or("--quant is bf16 or q8")?,
    };
    let only: Option<usize> = value("--only").map(|v| v.parse()).transpose()?;
    let step = |n: usize| only.is_none_or(|o| o == n);
    let dir = value("--fixtures").ok_or("--fixtures DIR is required")?;
    let file = |name: &str| -> Res<HashMap<String, Tensor>> { Ok(candle_core::safetensors::load(format!("{dir}/{name}"), &Device::Cpu)?) };
    let inputs = file("dit_inputs.safetensors")?;
    let get = |m: &HashMap<String, Tensor>, k: &str| -> Res<Tensor> { Ok(m.get(k).ok_or_else(|| format!("no `{k}`"))?.clone()) };
    let s = get(&inputs, "shape")?.to_vec1::<f32>()?;
    let shape = Shape::new(s[0] as usize, s[1] as usize, s[2] as usize, s[3] as f64)?;
    let sigma = get(&inputs, "sigma")?.to_vec1::<f32>()?[0];
    // From the latents, through this crate's patchifying, which step 1 checks
    // against the reference's.
    let video = video_tokens(&get(&inputs, "video_latent")?)?;
    let audio = audio_tokens(&get(&inputs, "audio_latent")?)?;
    // The contexts the reference ran with: its own text path's, or seeded
    // ones, which the fixture keeps beside its inputs.
    let contexts = match inputs.contains_key("video_context") {
        true => HashMap::from([("video".to_string(), get(&inputs, "video_context")?), ("audio".to_string(), get(&inputs, "audio_context")?)]),
        false => file("text_contexts_f32.safetensors")?,
    };
    let held = argv.iter().any(|a| a == "--held");
    let fixture = if held { "dit_held" } else { "dit" };
    let want = file(&format!("{fixture}_f32.safetensors"))?;
    let held = if held { shape.frame_tokens() } else { 0 };
    let blocks: usize = match value("--blocks") {
        Some(b) => b.parse()?,
        None => (0..).take_while(|i| want.contains_key(&format!("video_{i}"))).count(),
    };
    eprintln!("{}×{}×{} at {} fps: {} video tokens, {} audio; σ {sigma}; {blocks} blocks", shape.width, shape.height, shape.frames, shape.fps, shape.video_tokens(), shape.audio_latents());
    if held > 0 {
        eprintln!("the first {held} video tokens held at σ = 0");
    }

    if step(1) {
        // The reference's positions are [start, end) pairs; RoPE reads the
        // midpoint, averaged in f32.
        let mid = |t: Tensor| -> Res<Vec<Vec<f32>>> {
            let t = t.to_vec3::<f32>()?;
            Ok(t.iter().map(|axis| axis.iter().map(|p| (p[0] + p[1]) / 2.0).collect()).collect())
        };
        let (v, a) = (mid(get(&inputs, "video_positions")?)?, mid(get(&inputs, "audio_positions")?)?);
        let mine = shape.video_positions();
        let same = (0..3).all(|i| mine[i] == v[i]) && shape.audio_positions() == a[0];
        eprintln!("1. positions: {}", if same { "identical" } else { "DIFFERENT" });
        let tokens = |a: &Tensor, b: &Tensor| -> Res<bool> { Ok(a.flatten_all()?.to_vec1::<f32>()? == b.flatten_all()?.to_vec1::<f32>()?) };
        let same = tokens(&video, &get(&inputs, "video")?)? && tokens(&audio, &get(&inputs, "audio")?)?;
        eprintln!("   tokens from latents: {}", if same { "identical" } else { "DIFFERENT" });
        if !same {
            eprintln!("   time {:?}… against {:?}…", &mine[0][..4.min(mine[0].len())], &v[0][..4.min(v[0].len())]);
            eprintln!("   audio {:?}… against {:?}…", &shape.audio_positions()[..4], &a[0][..4]);
        }
    }

    let run = |device: &Device, dtype: DType, quant, label: &str, n: usize| -> Res<()> {
        let t = Instant::now();
        let dit = Dit::load(&path, device, dtype, Some(blocks), quant, &mut |m| eprintln!("   {m}"))?;
        eprintln!("{n}. {label}: {:.2} B parameters in {:.1} s", dit.params() as f64 / 1e9, t.elapsed().as_secs_f64());
        let ctx = Contexts { video: get(&contexts, "video")?.to_device(device)?, audio: get(&contexts, "audio")?.to_device(device)? };
        let grid = dit.grid(shape)?;
        let (v, a) = (video.to_device(device)?, audio.to_device(device)?);
        let mut out = vec![];
        let t = Instant::now();
        let (vv, va) = dit.forward_watched(&v, &a, (sigma, sigma), held, &ctx, &grid, &mut |i, vx, ax| {
            out.push((i, vx.clone(), ax.clone()));
            Ok(())
        })?;
        device.synchronize()?;
        eprintln!("   forward in {:.2} s", t.elapsed().as_secs_f64());
        for (i, vx, ax) in &out {
            eprintln!("   block {i}: video {:6.1} dB, audio {:6.1} dB", db(vx, &get(&want, &format!("video_{i}"))?)?, db(ax, &get(&want, &format!("audio_{i}"))?)?);
        }
        eprintln!("   velocity: video {:6.1} dB, audio {:6.1} dB", db(&vv, &get(&want, "video_out")?)?, db(&va, &get(&want, "audio_out")?)?);
        Ok(())
    };

    if step(2) {
        run(&Device::Cpu, DType::F32, None, "f32 on the CPU", 2)?;
    }
    if step(3) {
        // `--f32`: Metal's kernels in f32, which should be as exact as the CPU.
        let dtype = if argv.iter().any(|a| a == "--f32") { DType::F32 } else { DType::BF16 };
        let label = match (quant.is_some(), dtype) {
            (_, DType::F32) => "f32 on Metal",
            (true, _) => "q8 weights, bf16 on Metal",
            _ => "bf16 on Metal",
        };
        run(&Device::new_metal(0)?, dtype, quant, label, 3)?;
        if let Ok(r) = file(&format!("{fixture}_bf16.safetensors")) {
            let mut line = String::new();
            for i in 0..blocks {
                line += &format!(" block {i} {:.1}/{:.1},", db(&get(&r, &format!("video_{i}"))?, &get(&want, &format!("video_{i}"))?)?, db(&get(&r, &format!("audio_{i}"))?, &get(&want, &format!("audio_{i}"))?)?);
            }
            eprintln!(
                "   the reference's own bf16 on MPS, video/audio dB:{line} velocity {:.1}/{:.1}",
                db(&get(&r, "video_out")?, &get(&want, "video_out")?)?,
                db(&get(&r, "audio_out")?, &get(&want, "audio_out")?)?
            );
        }
    }
    Ok(())
}
