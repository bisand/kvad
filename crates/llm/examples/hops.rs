//! Where a token's time goes when a model is across machines.
//!
//! The plan says a token costs each stage's layers and a round trip for
//! each worker. This prints the three things that sum is made of, as they
//! were and not as they were expected: what the layers kept here took,
//! what each round trip took, and how much of each round trip the worker
//! says it was inside its own layers. What is left of a round trip is the
//! hop. Beside them, the same model whole on this machine, for what a
//! layer costs here when nothing interrupts it.
//!
//! ```text
//! cargo run --release -p kvad --example hops -- MODEL q8 'fe80::1c%en12' 0..15,15..30
//! ```
//!
//! Arguments: the model, its precision, the workers and the split as
//! `--workers` and `--split` take them, then optionally the tokens decoded
//! (200) and the prompt's length (64), and then a number of milliseconds
//! to wait between tokens: the time a larger model's other stages would
//! take, in which this model's machines sit idle. The model has to be on
//! disk already.

use kvad::cluster::key::Key;
use kvad::cluster::session::{self, Across};
use kvad::model::{CpuSession, Session, Spec};
use kvad::quant::Precision;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// Milliseconds a token, decoding `steps` greedily after `prompt`.
fn decode(session: &mut dyn Session, prompt: &[u32], steps: usize, before: &mut dyn FnMut(&mut dyn Session)) -> Res<f64> {
    let _ = session.truncate(0)?;
    let mut logits = session.forward(prompt)?;
    before(session);
    let started = Instant::now();
    for _ in 0..steps {
        logits = session.forward(&[argmax(&logits)])?;
    }
    Ok(started.elapsed().as_secs_f64() * 1e3 / steps as f64)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: hops MODEL PRECISION WORKERS SPLIT [TOKENS] [PROMPT]";
    let (model, workers, split) = (args.first().ok_or(usage)?, args.get(2).ok_or(usage)?, args.get(3).ok_or(usage)?);
    let precision = args.get(1).and_then(|p| Precision::parse(p)).ok_or(usage)?;
    let number = |i: usize, default: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default);
    let (steps, length, gap) = (number(4, 200), number(5, 64), std::time::Duration::from_millis(number(6, 0) as u64));

    let files = kvad::weights::fetch(model)?;
    let spec = Spec::from_json(&files.config)?;
    let prompt: Vec<u32> = (0..length).map(|i| ((i * 7919 + 13) % spec.vocab_size) as u32).collect();
    let repo = kvad::weights::model_id(model);
    let workers: Vec<String> = workers.split(',').map(str::to_string).collect();
    let split = session::parse_split(split)?;
    let key = Key::ours()?;
    let want = Across { repo: &repo, files: &files, spec: &spec, precision, workers: &workers, split: Some(&split), key: &key, params: None, context: spec.n_ctx.min(8192), mix: kvad::cluster::plan::Mix::REPLY, watch: &kvad::weights::Watcher::none(), memory: None, runs: None };
    let mut across = session::pipeline(&want, &mut |msg| eprintln!("  {msg}"))?;
    let mut whole = CpuSession::new(kvad::qcache::load(&repo, &files, &spec, precision, &mut |_| {})?, precision);

    // Once each to warm, then the ones that count: whole, across, whole.
    decode(&mut whole, &prompt, 16, &mut |_| {})?;
    decode(&mut across, &prompt, 16, &mut |_| {})?;
    let whole_before = decode(&mut whole, &prompt, steps, &mut |_| {})?;
    // The prefill's passes are not a token's: asked for here, and dropped.
    let split_ms = {
        let _ = across.truncate(0)?;
        let mut logits = across.forward(&prompt)?;
        across.timing();
        let mut spent = std::time::Duration::ZERO;
        for _ in 0..steps {
            std::thread::sleep(gap);
            let started = Instant::now();
            logits = across.forward(&[argmax(&logits)])?;
            spent += started.elapsed();
        }
        spent.as_secs_f64() * 1e3 / steps as f64
    };
    let (stages, head) = across.timing();
    let whole_after = decode(&mut whole, &prompt, steps, &mut |_| {})?;
    let whole_ms = (whole_before + whole_after) / 2.0;
    let a_layer_here = whole_ms / spec.n_layer as f64;

    println!("{model} at {precision}, {steps} tokens after a prompt of {length}");
    println!("whole, here: {whole_ms:.2} ms a token ({whole_before:.2} before and {whole_after:.2} after), which is {a_layer_here:.3} ms a layer and the ends");
    println!("across:      {split_ms:.2} ms a token{}", if gap.is_zero() { String::new() } else { format!(", with {} ms of nothing before each", gap.as_millis()) });
    println!();
    println!("| stage | layers | ms a token | of that, inside its layers | ms a layer | the hop |");
    println!("|---|---|---|---|---|---|");
    let mut accounted = 0.0;
    for stage in &stages {
        let per = stage.ms / stage.runs.max(1) as f64;
        accounted += per;
        let n = stage.shard.layers.len().max(1) as f64;
        match stage.inside_ms {
            None => println!("| here | {}..{} | {per:.2} | | {:.3} | |", stage.shard.layers.start, stage.shard.layers.end, per / n),
            Some(inside) => {
                let inside = inside / stage.runs.max(1) as f64;
                println!("| a worker | {}..{} | {per:.2} | {inside:.2} | {:.3} | {:.2} |", stage.shard.layers.start, stage.shard.layers.end, inside / n, per - inside);
            }
        }
    }
    let head_ms = head.1 / head.0.max(1) as f64;
    println!("| the head, here | | {head_ms:.2} | | | |");
    println!();
    println!("stages and head come to {:.2} ms of the {split_ms:.2}; the rest is the embedding, the sampler here and the loop.", accounted + head_ms);
    Ok(())
}
