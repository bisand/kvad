//! What a GPU stage's token costs after a wait, and what kind of wait.
//!
//! A stage of a model across machines works in bursts: a token's rows
//! arrive, its layers run, and then nothing until the token has been round
//! the other machines. On the CPU that idle made the next burst several
//! times slower, and one spinning thread cured it (`kvad::model::linger`).
//! This asks the same of a GPU stage, and takes the wait apart:
//!
//! * `sleep` — the thread sleeps: what a stage does between tokens.
//! * `spin`  — this thread spins: the CPU side stays awake, the GPU idles.
//! * `other` — another thread spins, which is what `linger` does today.
//! * `gpu`   — this thread keeps the GPU busy with a small kernel.
//!
//! The stage keeps its own device busy between tokens now
//! (`GpuStage`'s `Warm`), so `sleep` shows that unless `KVAD_LINGER_MS=0`
//! turns it off, which is how to see what it is for.
//!
//!     KVAD_QUANT_CACHE=/tmp/q cargo run --release -p kvad-gpu --example stage_gaps -- \
//!         HuggingFaceTB/SmolLM2-135M-Instruct 0..15 q8

use candle_core::{DType, Tensor};
use kvad::cluster::stage::Stage;
use kvad_gpu::model::stage::GpuStage;
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = args.first().map(String::as_str).unwrap_or("HuggingFaceTB/SmolLM2-135M-Instruct");
    let layers = args.get(1).map(String::as_str).unwrap_or("0..15");
    let (from, to) = layers.split_once("..").ok_or("layers as 0..15")?;
    let quant = kvad_gpu::model::parse_quant(args.get(2).map(String::as_str).unwrap_or("q8")).ok_or("quant: none, q8 or q4")?;
    let files = kvad::weights::fetch_with(repo, &mut |_| {})?;
    let spec = kvad::model::Spec::from_json(&files.config)?;
    let device = kvad_gpu::model::pick_device(None)?;
    let dtype = kvad_gpu::model::parse_dtype("bf16").ok_or("dtype")?;
    let mut stage = GpuStage::load(repo, &files.weights, spec.clone(), dtype, quant, device.clone(), from.parse()?..to.parse()?, &mut |m| eprintln!("  {m}"))?;

    let e = spec.n_embd;
    let row = || vec![0.01f32; e];
    // A prompt behind every token, so a token's attention has something to
    // attend to and every timed token is at the same position.
    const PROMPT: usize = 32;
    stage.run(0, 0, PROMPT, vec![0.01f32; PROMPT * e]).map_err(|e| e.to_string())?;
    for _ in 0..20 {
        stage.run(0, PROMPT, 1, row()).map_err(|e| e.to_string())?;
        stage.truncate(0, PROMPT).map_err(|e| e.to_string())?;
    }

    let small = Tensor::ones((64, 64), DType::F32, &device)?;
    let spinning = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let wait = |how: &str, gap: Duration| -> Res<()> {
        let until = Instant::now() + gap;
        match how {
            "sleep" => std::thread::sleep(gap),
            "spin" => {
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
            }
            "other" => {
                let flag = spinning.clone();
                flag.store(true, std::sync::atomic::Ordering::Release);
                let other = std::thread::spawn(move || {
                    while flag.load(std::sync::atomic::Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                });
                std::thread::sleep(gap);
                spinning.store(false, std::sync::atomic::Ordering::Release);
                let _ = other.join();
            }
            "gpu" => {
                // In a pool, which is also a turn at the device: the
                // stage's own thread may be there.
                while Instant::now() < until {
                    kvad_gpu::pooled(|| small.matmul(&small)?.sum_all()?.to_scalar::<f32>())?;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    };

    let pool = std::env::var("STAGE_GAPS_POOL").is_ok().then(kvad::model::pool);
    let gaps = [0u64, 1, 3, 10, 30, 100];
    let hows = ["sleep", "spin", "other", "gpu"];
    const ROUNDS: usize = 5;
    const TOKENS: usize = 30;
    // Untimed at the head of each cell: a device takes tens of
    // milliseconds to step back up, and the cell before was another wait.
    const SETTLE: usize = 12;
    let mut times = vec![vec![Vec::new(); gaps.len()]; hows.len()];
    // Interleaved: every round visits every kind of wait and every gap.
    for _ in 0..ROUNDS {
        for (h, how) in hows.iter().enumerate() {
            for (g, gap) in gaps.iter().enumerate() {
                for token in 0..SETTLE + TOKENS {
                    wait(how, Duration::from_millis(*gap))?;
                    let start = Instant::now();
                    match &pool {
                        // As a worker ran a stage: on a thread of the
                        // pool its CPU kernels use, this one waiting.
                        Some(pool) => pool.install(|| stage.run(0, PROMPT, 1, row())).map_err(|e| e.to_string())?,
                        None => stage.run(0, PROMPT, 1, row()).map_err(|e| e.to_string())?,
                    };
                    if token >= SETTLE {
                        times[h][g].push(start.elapsed().as_secs_f64() * 1e3);
                    }
                    stage.truncate(0, PROMPT).map_err(|e| e.to_string())?;
                }
            }
        }
    }

    println!("{repo}, layers {layers}, ms a token after a wait: the median of {} tokens\n", ROUNDS * TOKENS);
    println!("| wait | {} |", gaps.iter().map(|g| format!("{g} ms")).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---|".repeat(gaps.len()));
    for (h, how) in hows.iter().enumerate() {
        let row: Vec<String> = times[h]
            .iter_mut()
            .map(|t| {
                t.sort_by(f64::total_cmp);
                format!("{:.2}", t[t.len() / 2])
            })
            .collect();
        println!("| {how} | {} |", row.join(" | "));
    }
    Ok(())
}
