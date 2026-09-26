//! LTX-2.5's duration head: how long each prompt's clip wants to be.
//!
//!     cargo run --release -p kvad-gpu --example ltx_duration -- [--fps 24] "PROMPT" …
//!     cargo run --release -p kvad-gpu --example ltx_duration -- --contexts OUT
//!     cargo run --release -p kvad-gpu --example ltx_duration -- --fixtures DIR
//!
//! With prompts, the text path encodes each and the head says how many
//! seconds it predicts, and the frames that makes. `--contexts OUT` also
//! writes each prompt's contexts to `OUT/duration_contexts.safetensors`, for
//! `scripts/ltx-fixtures.py --duration` to run the reference's head on them;
//! `--fixtures DIR` then compares this head with the reference's on those
//! same contexts. `--where` prints where the head's file is (downloading it
//! first).

use candle_core::{Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::ltx_duration::{frames_for, DurationHead, FILE, MAX_SECONDS, MIN_SECONDS};
use kvad_gpu::video::ltx_text::{TextEncoder, DIT_FILE, TEXT_FILE};
use kvad_gpu::video::LTX_REPO;
use std::collections::HashMap;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let fetch = |f: &str| fetch_file(LTX_REPO, f, &Watcher::none());
    let path = fetch(FILE)?;
    if argv.iter().any(|a| a == "--where") {
        println!("{}", path.display());
        return Ok(());
    }
    let fps: f64 = value("--fps").map(|v| v.parse()).transpose()?.unwrap_or(24.0);
    let (lo, hi) = ((MIN_SECONDS * fps).round() as usize, (MAX_SECONDS * fps).round() as usize);

    if let Some(dir) = value("--fixtures") {
        let ctx = candle_core::safetensors::load(format!("{dir}/duration_contexts.safetensors"), &Device::Cpu)?;
        let want = candle_core::safetensors::load(format!("{dir}/duration_f32.safetensors"), &Device::Cpu)?;
        let bf16 = candle_core::safetensors::load(format!("{dir}/duration_bf16.safetensors"), &Device::Cpu).ok();
        let get = |m: &HashMap<String, Tensor>, k: &str| -> Res<Tensor> { Ok(m.get(k).ok_or_else(|| format!("no `{k}`"))?.clone()) };
        for (name, device) in [("the CPU", Device::Cpu), ("Metal", Device::new_metal(0)?)] {
            let head = DurationHead::load(&path, &device)?;
            eprintln!("f32 on {name}:");
            for i in 0.. {
                let Ok(v) = get(&ctx, &format!("video_{i}")) else { break };
                let s = head.seconds(&v, &get(&ctx, &format!("audio_{i}"))?)?;
                let theirs = get(&want, &format!("seconds_{i}"))?.to_vec1::<f32>()?[0] as f64;
                let mut line = format!("   {i}: {s:.5} s against {theirs:.5} ({:+.1e}); {} frames against {}", s / theirs - 1.0, frames_for(s, fps, lo, hi), frames_for(theirs, fps, lo, hi));
                if let Some(b) = &bf16 {
                    let r = get(b, &format!("seconds_{i}"))?.to_vec1::<f32>()?[0] as f64;
                    line += &format!("; the reference's bf16 {r:.5} ({:+.1e}), {} frames", r / theirs - 1.0, frames_for(r, fps, lo, hi));
                }
                eprintln!("{line}");
            }
        }
        return Ok(());
    }

    let prompts: Vec<&String> = argv.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || !["--fps", "--contexts"].contains(&argv[i - 1].as_str()))).map(|(_, a)| a).collect();
    if prompts.is_empty() {
        return Err("give a prompt or two, or --fixtures DIR".into());
    }
    let device = Device::new_metal(0)?;
    let t = Instant::now();
    let enc = TextEncoder::load(&fetch(TEXT_FILE)?, &fetch(DIT_FILE)?, &device, candle_core::DType::BF16, kvad_gpu::model::parse_quant("q8").ok_or("q8")?, &mut |_| {})?;
    let head = DurationHead::load(&path, &device)?;
    eprintln!("text path and head loaded in {:.1} s", t.elapsed().as_secs_f64());
    let mut saved = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let ctx = enc.encode(p)?;
        let t = Instant::now();
        let s = head.seconds(&ctx.video, &ctx.audio)?;
        device.synchronize()?;
        println!("{s:6.2} s  {:3} frames at {fps} fps  ({:.0} ms)  {p}", frames_for(s, fps, lo, hi), t.elapsed().as_secs_f64() * 1e3);
        let cpu = |t: &Tensor| t.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32);
        saved.push((format!("video_{i}"), cpu(&ctx.video)?));
        saved.push((format!("audio_{i}"), cpu(&ctx.audio)?));
    }
    if let Some(out) = value("--contexts") {
        std::fs::create_dir_all(&out)?;
        candle_core::safetensors::save(&saved.into_iter().collect(), format!("{out}/duration_contexts.safetensors"))?;
    }
    Ok(())
}
