//! What cutting a model into shards costs when nothing crosses a wire.
//!
//! The model is loaded whole and then in pieces, every piece in this
//! process, and the two are held to each other twice. First by the bit:
//! every logit of a prefill and of each decoded token has to be the same
//! number, or the run stops. Then by the clock: the pipeline is the same
//! arithmetic with a function call at each cut, so it should cost nothing,
//! and if it costs something this is where that is found and not on a
//! cluster, where a hop would hide it.
//!
//! Whole and cut are run back to back within a round and their order swaps
//! each round, because this machine's load moves. Medians and minimums: a
//! median that is not near its minimum is a measurement that was disturbed.
//!
//! ```text
//! cargo run --release -p kvad --example split -- Qwen/Qwen2.5-1.5B-Instruct q8
//! ```
//!
//! Arguments: the model, its precision (`f32`, `q8`, `q4`), then optionally
//! the rounds (5), the tokens decoded a run (48) and the prompt's length in
//! tokens (256). The model has to be on disk already.

use kvad::cluster::session::PipelineSession;
use kvad::model::{CpuSession, Session, Shard, Spec};
use kvad::qcache;
use kvad::quant::Precision;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// A prefill and `steps` greedy tokens from an empty cache: every row of
/// logits on the way, and the seconds each half took.
fn run(session: &mut dyn Session, prompt: &[u32], steps: usize) -> Res<(Vec<Vec<f32>>, f64, f64)> {
    let _ = session.truncate(0)?;
    let started = Instant::now();
    let mut rows = vec![session.forward(prompt)?];
    let prefill = started.elapsed().as_secs_f64();
    let started = Instant::now();
    for _ in 0..steps {
        let next = argmax(rows.last().expect("one row at least"));
        rows.push(session.forward(&[next])?);
    }
    Ok((rows, prefill, started.elapsed().as_secs_f64()))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn best(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args.first().ok_or("usage: split MODEL [f32|q8|q4] [ROUNDS] [TOKENS] [PROMPT]")?;
    let precision = args.get(1).and_then(|p| Precision::parse(p)).unwrap_or(Precision::Q8);
    let number = |i: usize, default: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default);
    let (rounds, steps, length) = (number(2, 5), number(3, 48), number(4, 256));

    let files = kvad::weights::fetch(model)?;
    let spec = Spec::from_json(&files.config)?;
    eprintln!("{model} at {precision}: {}", spec.summary());
    // Not text, and it does not need to be: the arithmetic is the same on
    // any tokens, and these need no tokenizer to make.
    let prompt: Vec<u32> = (0..length).map(|i| ((i * 7919 + 13) % spec.vocab_size) as u32).collect();

    // Whole and cut are read the same way, straight from the checkpoint,
    // so that the cut is the only difference between them. A whole model
    // mapped from its cache file measured about 4% slower than the same
    // weights quantised onto the heap, on one model, and that is a fact
    // about where weights live and not about where a model is cut.
    let mut whole = CpuSession::new(qcache::load_shard(&files, &spec, precision, &Shard::whole(&spec))?, precision);
    let (reference, ..) = run(&mut whole, &prompt, steps)?;

    println!("| shards | prefill, tok/s whole | cut | cut / whole | decode, tok/s whole | cut | cut / whole |");
    println!("|---|---|---|---|---|---|---|");
    for parts in [1, 2, 3, 4] {
        let cut = Shard::cut(spec.n_layer, parts);
        eprintln!("loading {parts} shard(s): {}", cut.iter().map(Shard::to_string).collect::<Vec<_>>().join("; "));
        let shards = cut.iter().map(|shard| qcache::load_shard(&files, &spec, precision, shard)).collect::<Res<Vec<_>>>()?;
        let mut split = PipelineSession::local(shards, precision)?;

        // The answer first. A speed for a model that says something else is
        // a speed for a different model.
        let (rows, ..) = run(&mut split, &prompt, steps)?;
        for (i, (a, b)) in reference.iter().zip(&rows).enumerate() {
            if bits(a) != bits(b) {
                let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                return Err(format!("{parts} shards: row {i} of {} differs from the whole model's, by up to {worst}", rows.len()).into());
            }
        }

        let (mut prefill, mut decode) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        for round in 0..rounds {
            // Whole first on even rounds, cut first on odd ones.
            for which in [round % 2, 1 - round % 2] {
                let session: &mut dyn Session = if which == 0 { &mut whole } else { &mut split };
                let (_, p, d) = run(session, &prompt, steps)?;
                prefill[which].push(length as f64 / p);
                decode[which].push(steps as f64 / d);
            }
        }
        let cell = |v: &[f64]| format!("{:.1} ({:.1})", median(v), best(v));
        println!(
            "| {parts} | {} | {} | {:.3} | {} | {} | {:.3} |",
            cell(&prefill[0]),
            cell(&prefill[1]),
            median(&prefill[1]) / median(&prefill[0]),
            cell(&decode[0]),
            cell(&decode[1]),
            median(&decode[1]) / median(&decode[0]),
        );
    }
    println!("\nMedian of {rounds} rounds, and the best in brackets. Every logit of every run agreed to the bit.");
    Ok(())
}
