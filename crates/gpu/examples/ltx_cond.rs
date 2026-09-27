//! A conditioned state (DFR's anchors, generated keyframes and reference
//! latent) and the DiT on it, checked against the reference's own.
//!
//!     cargo run --release -p kvad-gpu --example ltx_cond -- --fixtures DIR [--only N]
//!
//! `DIR` is what `scripts/ltx-fixtures.py --dit … --contexts random
//! --conditioned` wrote: the reference's items on its 512×320 × 25 latent at
//! 60 fps (two anchors at pixel frames 0 and 16, two generated keyframes at 8
//! and 24, a half-size reference latent), noised at 0.975 from noise it
//! saved, and its DiT's first blocks on that with the sound frozen at σ 0.
//!
//! 1. **The state**, built here by `ltx_cond` from the same latents and
//!    noise: every token's place, mask, keyframe mark, clean and noisy
//!    latent, against the reference's.
//! 2. **f32, CPU**: the DiT on it, against the reference's f32.
//! 3. **bf16, Metal**, and the reference's own bf16 against its f32.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::ltx_cond::State;
use kvad_gpu::video::ltx_dit::{audio_tokens, Dit, Shape};
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
    let only: Option<usize> = value("--only").map(|v| v.parse()).transpose()?;
    let step = |n: usize| only.is_none_or(|o| o == n);
    let dir = value("--fixtures").ok_or("--fixtures DIR is required")?;
    let file = |name: &str| -> Res<HashMap<String, Tensor>> { Ok(candle_core::safetensors::load(format!("{dir}/{name}"), &Device::Cpu)?) };
    let get = |m: &HashMap<String, Tensor>, k: &str| -> Res<Tensor> { Ok(m.get(k).ok_or_else(|| format!("no `{k}`"))?.clone()) };
    let (inputs, fx) = (file("dit_inputs.safetensors")?, file("dit_cond_state.safetensors")?);
    let s = get(&inputs, "shape")?.to_vec1::<f32>()?;
    let sigma = get(&inputs, "sigma")?.to_vec1::<f32>()?[0];
    // The video at the 60 fps DFR conditions at; the sound at the clip's own.
    let shape = Shape::new(s[0] as usize, s[1] as usize, s[2] as usize, 60.0)?;
    let own = Shape::new(s[0] as usize, s[1] as usize, s[2] as usize, s[3] as f64)?;

    // 1. The state.
    let anchors = get(&fx, "anchors")?;
    let mut state = State::video(&get(&inputs, "video_latent")?, shape)?;
    for (i, f) in [0, 16].into_iter().enumerate() {
        state = state.anchor(&anchors.narrow(1, i, 1)?, f, 0.95)?;
    }
    let state = state.slots(&[8, 24], Some(&get(&fx, "initials")?))?.reference(&get(&fx, "reference")?, 2, 1.0)?;
    let noise = get(&fx, "noise")?.flatten_all()?.to_vec1::<f32>()?;
    let state = state.noised(&noise, 0.975)?;
    if step(1) {
        let mid = get(&fx, "positions")?.to_vec3::<f32>()?;
        let mid: Vec<Vec<f32>> = mid.iter().map(|axis| axis.iter().map(|p| (p[0] + p[1]) / 2.0).collect()).collect();
        let same = |a: &[f32], b: &[f32]| a == b;
        let places = (0..3).all(|i| same(&state.positions[i], &mid[i]));
        let masks = same(&state.mask, &get(&fx, "mask")?.to_vec1::<f32>()?);
        let marks: Vec<f32> = state.marks.iter().map(|&m| m as u8 as f32).collect();
        let marked = same(&marks, &get(&fx, "marks")?.to_vec1::<f32>()?);
        let clean = same(&state.clean, &get(&fx, "clean")?.flatten_all()?.to_vec1::<f32>()?);
        let noised = same(&state.latent, &get(&fx, "noised")?.flatten_all()?.to_vec1::<f32>()?);
        let word = |b: bool| if b { "identical" } else { "DIFFERENT" };
        eprintln!("1. {} tokens: places {}, masks {}, marks {}, clean {}, noised {}", state.len(), word(places), word(masks), word(marked), word(clean), word(noised));
        if !places {
            for (i, (ours, theirs)) in state.positions.iter().zip(&mid).enumerate() {
                let worst = ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                eprintln!("   axis {i}: {} against {}, at most {worst} apart", ours.len(), theirs.len());
            }
        }
        if !noised {
            let n = Tensor::from_slice(&state.latent, state.latent.len(), &Device::Cpu)?;
            eprintln!("   noised: {:.1} dB", db(&n, &get(&fx, "noised")?)?);
        }
    }

    let audio = audio_tokens(&get(&inputs, "audio_latent")?)?;
    let contexts = [get(&inputs, "video_context")?, get(&inputs, "audio_context")?];
    let path = fetch_file(LTX_REPO, DIT_FILE, &Watcher::none())?;
    let want = file("dit_cond_f32.safetensors")?;
    let blocks = 2;
    let run = |device: &Device, dtype: DType, label: &str, n: usize| -> Res<(Tensor, Tensor)> {
        let t = Instant::now();
        let dit = Dit::load_as(&path, None, "transformer-check", device, dtype, Some(blocks), None, &mut |_| {})?;
        eprintln!("{n}. {label}: {:.2} B parameters in {:.1} s", dit.params() as f64 / 1e9, t.elapsed().as_secs_f64());
        let ctx = Contexts { video: contexts[0].to_device(device)?, audio: contexts[1].to_device(device)? };
        let grid = dit.grid_at(shape, state.positions.clone(), &state.marks, own.audio_positions())?;
        let t = Instant::now();
        // The sound frozen: σ 0 for all of it, as DFR's temporal tiles have it.
        let (v, a) = dit.forward_masked(&state.tokens(device)?, &audio.to_device(device)?, (sigma, 0.0), &state.mask, &ctx, &grid)?;
        device.synchronize()?;
        eprintln!("   forward in {:.2} s", t.elapsed().as_secs_f64());
        let (wv, wa) = (get(&want, "video_out")?, get(&want, "audio_out")?);
        eprintln!("   velocity: video {:5.1} dB, audio {:5.1} dB", db(&v, &wv)?, db(&a, &wa)?);
        // The appended tokens on their own: anchors, keyframes, reference.
        let video = state.video_len();
        let appended = |t: &Tensor| t.narrow(0, video, state.len() - video);
        eprintln!("   the {} appended tokens: {:.1} dB", state.len() - video, db(&appended(&v)?, &appended(&wv)?)?);
        Ok((v, a))
    };
    if step(2) {
        run(&Device::Cpu, DType::F32, "f32 on the CPU", 2)?;
    }
    if step(3) {
        run(&Device::new_metal(0)?, DType::BF16, "bf16 on Metal", 3)?;
        if let Ok(r) = file("dit_cond_bf16.safetensors") {
            eprintln!("   the reference's own bf16 on MPS: video {:.1} dB, audio {:.1} dB",
                      db(&get(&r, "video_out")?, &get(&want, "video_out")?)?, db(&get(&r, "audio_out")?, &get(&want, "audio_out")?)?);
        }
    }
    Ok(())
}
