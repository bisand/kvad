//! What batching decode buys on the GPU, measured: `kvad`'s
//! `examples/batch_decode`, with the model on Metal.
//!
//! `B` conversations are prefilled and set aside, then advanced by one token
//! at a time, two ways: each in turn — take it back, one token, set it aside
//! — and all in one pass, [`Session::forward_each`]. Batching raises the
//! total and raises each conversation's wait for its next token, so both are
//! printed, side by side, for each `B`.
//!
//! The two ways are run back to back within a round and their order swaps
//! each round, because this machine's load moves and a batch-16 run and a
//! batch-1 run minutes apart measure the afternoon as much as the code.
//! Medians and minimums of the rounds: a median that is not near its minimum
//! is a measurement that was disturbed.
//!
//! ```text
//! cargo run --release -p kvad-gpu --example batch_decode -- Qwen/Qwen2.5-1.5B-Instruct q8
//! ```
//!
//! Arguments: the model, how it is held (`bf16`, `q8`, `q4`), then optionally
//! the rounds (5), the tokens decoded per run (24) and the prompt length in
//! tokens (64).

use kvad::model::Parked;
use kvad::runtime::Llm;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const BATCHES: [usize; 7] = [1, 2, 3, 4, 6, 8, 16];

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// `b` conversations, each a different prompt of `len` tokens, set aside;
/// and the first token each will be given.
fn prefilled(llm: &mut Llm, b: usize, len: usize) -> Res<(Vec<Parked>, Vec<u32>)> {
    let text = "The keeper of the lighthouse wrote in the log each evening what the sea had done, \
                and what the lamp had needed, and who had come by the rock in a boat that day. "
        .repeat(8);
    let ids = llm.encode(&text)?;
    let (mut caches, mut next) = (Vec::new(), Vec::new());
    for i in 0..b {
        // A different stretch for each, so no two conversations are alike.
        let prompt: Vec<u32> = ids.iter().cycle().skip(i * 7).take(len).copied().collect();
        llm.session.truncate(0)?;
        next.push(argmax(&llm.session.forward(&prompt)?));
        caches.push(llm.session.park().ok_or("this backend keeps one conversation")?);
    }
    Ok((caches, next))
}

/// Seconds to advance every conversation `steps` tokens, one pass each.
fn together(llm: &mut Llm, caches: &mut [Parked], mut next: Vec<u32>, steps: usize) -> Res<f64> {
    let started = Instant::now();
    for _ in 0..steps {
        let rows = llm.session.forward_each(&next, caches)?;
        next = rows.iter().map(|row| argmax(row)).collect();
    }
    Ok(started.elapsed().as_secs_f64())
}

/// The same, a conversation at a time.
fn in_turn(llm: &mut Llm, caches: &mut [Parked], mut next: Vec<u32>, steps: usize) -> Res<f64> {
    let started = Instant::now();
    for _ in 0..steps {
        for (cache, token) in caches.iter_mut().zip(next.iter_mut()) {
            llm.session.resume(std::mem::replace(cache, Box::new(())))?;
            *token = argmax(&llm.session.forward(&[*token])?);
            *cache = llm.session.park().ok_or("this backend keeps one conversation")?;
        }
    }
    Ok(started.elapsed().as_secs_f64())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let repo = args.first().ok_or("usage: batch_decode MODEL [bf16|q8|q4] [ROUNDS] [TOKENS] [PROMPT]")?;
    let held = args.get(1).map_or("q8", String::as_str);
    let quant = kvad_gpu::model::parse_quant(if held == "bf16" { "none" } else { held }).ok_or("bf16, q8 or q4")?;
    let dtype = kvad_gpu::model::parse_dtype("bf16").ok_or("bf16 is a dtype")?;
    let number = |at: usize, otherwise: usize| args.get(at).and_then(|v| v.parse().ok()).unwrap_or(otherwise);
    let (rounds, steps, prompt) = (number(2, 5), number(3, 24), number(4, 64));

    let id = kvad::weights::model_id(repo);
    let mut llm = Llm::load_custom(repo, &mut |_| {}, &kvad::weights::Watcher::default(), &mut |files, spec, progress| {
        let device = kvad_gpu::model::pick_device(None)?;
        kvad_gpu::model::session(&id, &files.weights, spec, dtype, quant, device, progress)
    })?;
    println!("{repo} on {}", llm.backend());
    println!("{rounds} rounds, {steps} tokens a run, prompts of {prompt} tokens\n");

    // One run of each before anything is timed: the first pass after a load
    // is not like the rest.
    let (mut caches, next) = prefilled(&mut llm, 2, prompt)?;
    together(&mut llm, &mut caches, next.clone(), 4)?;
    let (mut caches, next) = prefilled(&mut llm, 2, prompt)?;
    in_turn(&mut llm, &mut caches, next, 4)?;

    // secs[batch][way][round]
    let mut secs = vec![[Vec::new(), Vec::new()]; BATCHES.len()];
    for round in 0..rounds {
        for (at, &b) in BATCHES.iter().enumerate() {
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            for way in order {
                let (mut caches, next) = prefilled(&mut llm, b, prompt)?;
                let took = match way {
                    0 => in_turn(&mut llm, &mut caches, next, steps)?,
                    _ => together(&mut llm, &mut caches, next, steps)?,
                };
                secs[at][way].push(took);
            }
        }
    }

    println!("       |      in turn       |      one pass      |");
    println!(" batch | tok/s  ms a token  | tok/s  ms a token  | faster  (fastest round)");
    for (at, &b) in BATCHES.iter().enumerate() {
        let tokens = (b * steps) as f64;
        let [turn, pass] = &mut secs[at];
        let (turn_best, pass_best) =
            (turn.iter().copied().fold(f64::MAX, f64::min), pass.iter().copied().fold(f64::MAX, f64::min));
        let (turn, pass) = (median(turn), median(pass));
        // What one conversation waits between its own tokens: the whole
        // step, since its next token comes when everybody's has.
        let wait = |secs: f64| secs / steps as f64 * 1000.0;
        println!(
            " {b:5} | {:5.1}  {:9.1}  | {:5.1}  {:9.1}  | {:5.2}x  ({:.2}x)",
            tokens / turn,
            wait(turn),
            tokens / pass,
            wait(pass),
            turn / pass,
            turn_best / pass_best,
        );
    }
    Ok(())
}
