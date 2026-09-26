//! LTX-2.5's text path — Gemma 4, the projections, the two connectors —
//! checked against the reference's own outputs.
//!
//!     cargo run --release -p kvad-gpu --example ltx_text -- --fixtures DIR [--prompt TEXT]
//!
//! `DIR` is what `scripts/ltx-fixtures.py --text … --dit …` wrote. In turn:
//!
//! 1. **Tokens.** Every prompt in `text_tokens.json`, tokenised here, against
//!    the ids the reference's own tokenizer made.
//! 2. **Six layers, f32, CPU.** Gemma's first six layers (one of them
//!    global) on the first prompt, against the reference's in f32: exact.
//! 3. **The rest, f32, CPU.** The reference's 49 hidden states through the
//!    projections and connectors here, against the reference's own: exact.
//! 4. **The tower, bf16, Metal.** All 48 layers on the first prompt, as they
//!    will run, against the reference's in bf16 on MPS.
//! 5. **All of it, bf16, Metal.** The whole path, against the reference's
//!    f32 contexts (from its bf16 hidden states).
//!
//! `--quant q8` runs steps 4 and 5 with Gemma and the projections at Q8_0.
//! `--only N` runs just step N. `--prompt` also encodes a prompt of your own
//! on Metal and reports the time. `--where` prints the two files' paths
//! (downloading them first).

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::gemma::Gemma;
use kvad_gpu::video::ltx_text::{TextEncoder, Tokenizer, DIT_FILE, LENGTH, TEXT_FILE};
use kvad_gpu::video::LTX_REPO;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Signal-to-error ratio in dB, and the cosine similarity, of `got` against
/// `want`.
fn compare(got: &Tensor, want: &Tensor) -> Res<(f32, f32)> {
    let cpu = |t: &Tensor| -> candle_core::Result<Tensor> { t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all() };
    let (g, w) = (cpu(got)?, cpu(want)?);
    let err = (&g - &w)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let sig = w.sqr()?.sum_all()?.to_scalar::<f32>()?;
    if !(err + sig).is_finite() {
        return Err("a comparison met a NaN or an infinity".into());
    }
    let dot = (&g * &w)?.sum_all()?.to_scalar::<f32>()?;
    let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() * sig.sqrt();
    Ok((10.0 * (sig / err.max(1e-30)).log10(), dot / norm))
}

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| argv.iter().any(|a| a == f);
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let text = fetch_file(LTX_REPO, TEXT_FILE, &Watcher::none())?;
    // Only the steps with connectors need the DiT's 42 GB file.
    let dit = || fetch_file(LTX_REPO, DIT_FILE, &Watcher::none());
    if flag("--where") {
        println!("{}\n{}", text.display(), dit()?.display());
        return Ok(());
    }
    // `--quant q8`: Gemma and the projections at Q8_0, as the pipeline will
    // run them.
    let quant = match value("--quant").as_deref() {
        None | Some("bf16") => None,
        Some(q) => kvad_gpu::model::parse_quant(q).ok_or("--quant is bf16 or q8")?,
    };
    let only: Option<usize> = value("--only").map(|v| v.parse()).transpose()?;
    let step = |n: usize| only.is_none_or(|o| o == n);
    let dir = value("--fixtures").ok_or("--fixtures DIR is required")?;
    let tokens: kvad::serde_json::Value = kvad::serde_json::from_str(&std::fs::read_to_string(format!("{dir}/text_tokens.json"))?)?;
    let first: Vec<u32> = tokens[0]["ids"].as_array().ok_or("no ids")?.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect();
    let load = |name: &str, key: &str| -> Res<Tensor> {
        Ok(candle_core::safetensors::load(format!("{dir}/{name}"), &Device::Cpu)?.remove(key).ok_or_else(|| format!("no `{key}` in {name}"))?)
    };

    if step(1) {
        let tok = Tokenizer::load(&text)?;
        for p in tokens.as_array().ok_or("no prompts")? {
            let want: Vec<u32> = p["ids"].as_array().ok_or("no ids")?.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect();
            let got = tok.tokens(p["prompt"].as_str().unwrap_or(""))?;
            let same = got == want;
            eprintln!("1. tokens: {} of {} {}", got.len(), want.len(), if same { "— identical" } else { "— DIFFERENT" });
            if !same {
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                eprintln!("   first difference at {at}: ours {:?}, reference {:?}", &got[at..(at + 5).min(got.len())], &want[at..(at + 5).min(want.len())]);
            }
        }
    }

    if step(3) {
        // The encoder without Gemma's layers is cheap to load on the CPU.
        let t = Instant::now();
        let enc = TextEncoder::load_with(&text, &dit()?, &Device::Cpu, DType::F32, Some(0), None, &mut |m| eprintln!("  {m}"))?;
        eprintln!("text encoder without Gemma's layers, f32 on the CPU: loaded in {:.1} s", t.elapsed().as_secs_f64());
        {
            let states = load("text_gemma_bf16.safetensors", "states")?;
            let list: Vec<Tensor> = (0..states.dim(0)?).map(|i| states.get(i)).collect::<candle_core::Result<_>>()?;
            let t = Instant::now();
            let ctx = enc.project(&list)?;
            let (v, a) = (load("text_contexts_f32.safetensors", "video")?, load("text_contexts_f32.safetensors", "audio")?);
            let ((sv, cv), (sa, ca)) = (compare(&ctx.video, &v)?, compare(&ctx.audio, &a)?);
            eprintln!("3. projections and connectors, f32: video {sv:.1} dB (cos {cv:.6}), audio {sa:.1} dB (cos {ca:.6}), {:.1} s", t.elapsed().as_secs_f64());
        }
    }

    if step(2) {
        // `--metal` runs the six layers as they will really run, in bf16 on
        // Metal, to see how far bf16 alone moves them from the f32 reference.
        let (dev, dtype) = match flag("--metal") {
            true => (Device::new_metal(0)?, DType::BF16),
            false => (Device::Cpu, DType::F32),
        };
        let g = Gemma::load_file(&text, &dev, dtype, Some(6), None, &mut |m| eprintln!("  {m}"))?;
        let want = load("text_gemma6_f32.safetensors", "states")?;
        let got = g.hidden_states(&first, LENGTH - first.len())?;
        for (i, s) in got.iter().enumerate() {
            let (snr, cos) = compare(s, &want.get(i)?)?;
            eprintln!("2. six layers, {dtype:?}: state {i}: {snr:.1} dB (cos {cos:.7})");
        }
    }

    if step(4) {
        let dev = Device::new_metal(0)?;
        let t = Instant::now();
        let g = Gemma::load_file(&text, &dev, DType::BF16, None, quant, &mut |m| eprintln!("  {m}"))?;
        let label = if quant.is_some() { "q8" } else { "bf16" };
        eprintln!("Gemma, {label} weights and bf16 activations, on Metal: loaded in {:.1} s", t.elapsed().as_secs_f64());
        let want = load("text_gemma_bf16.safetensors", "states")?;
        for run in 0..2 {
            let t = Instant::now();
            let states = g.hidden_states(&first, LENGTH - first.len())?;
            states[states.len() - 1].to_device(&Device::Cpu)?;
            eprintln!("4. {} tokens through 48 layers in {:.2} s{}", first.len(), t.elapsed().as_secs_f64(), if run == 0 { " (first run)" } else { "" });
            if run == 1 {
                let mut worst = (f32::MAX, 0);
                for (i, s) in states.iter().enumerate() {
                    let (snr, cos) = compare(s, &want.get(i)?)?;
                    if snr < worst.0 {
                        worst = (snr, i);
                    }
                    if i % 8 == 0 || i == states.len() - 1 {
                        eprintln!("4. bf16 against the reference's bf16 on MPS: state {i}: {snr:.1} dB (cos {cos:.5})");
                    }
                }
                eprintln!("   the furthest apart is state {} at {:.1} dB", worst.1, worst.0);
            }
        }
        // The worst case for time: the fixtures' third prompt, cut at 1024.
        if let Some(long) = tokens.get(2).and_then(|p| p["ids"].as_array()) {
            let long: Vec<u32> = long.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect();
            for run in 0..3 {
                let t = Instant::now();
                let states = g.hidden_states(&long, LENGTH - long.len())?;
                states[states.len() - 1].to_device(&Device::Cpu)?;
                eprintln!("4. {} tokens through 48 layers in {:.2} s{}", long.len(), t.elapsed().as_secs_f64(), if run == 0 { " (first run)" } else { "" });
            }
        }
    }

    if step(5) || value("--prompt").is_some() {
        let dev = Device::new_metal(0)?;
        let t = Instant::now();
        let enc = TextEncoder::load(&text, &dit()?, &dev, DType::BF16, quant, &mut |m| eprintln!("  {m}"))?;
        eprintln!("text encoder, {} weights and bf16 activations on Metal: {:.2} B parameters, loaded in {:.1} s", if quant.is_some() { "q8" } else { "bf16" }, enc.params() as f64 / 1e9, t.elapsed().as_secs_f64());
        if step(5) {
            let states = enc.gemma.hidden_states(&first, LENGTH - first.len())?;
            let ctx = enc.project(&states)?;
            let (v, a) = (load("text_contexts_f32.safetensors", "video")?, load("text_contexts_f32.safetensors", "audio")?);
            let ((sv, cv), (sa, ca)) = (compare(&ctx.video, &v)?, compare(&ctx.audio, &a)?);
            eprintln!("5. contexts, bf16 on Metal, against the reference's f32 from its bf16 states: video {sv:.1} dB (cos {cv:.5}), audio {sa:.1} dB (cos {ca:.5})");
        }
        if let Some(p) = value("--prompt") {
            for run in 0..2 {
                let t = Instant::now();
                let ctx = enc.encode(&p)?;
                ctx.video.to_device(&Device::Cpu)?;
                eprintln!("encode ({}): {} tokens in {:.2} s", if run == 0 { "first" } else { "again" }, enc.tokens(&p)?.len(), t.elapsed().as_secs_f64());
            }
        }
    }
    Ok(())
}
