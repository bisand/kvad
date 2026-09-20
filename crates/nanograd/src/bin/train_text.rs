//! Train a small GPT on a text file, one character at a time, and watch it
//! learn to write.
//!
//!     ./scripts/get-text.sh
//!     cargo run --release -p nanograd --bin train_text
//!
//! Any plain text file will do: `--data path/to/file.txt`.
//!
//! Keep what it learned, and come back to it:
//!
//!     train_text --save out/shakespeare
//!     train_text --load out/shakespeare --steps 0 --prompt "ROMEO:"   # just write
//!     train_text --load out/shakespeare --steps 500 --save out/more   # train on
//!
//! Options: --data PATH --steps N --batch N --context N --d-model N --heads N
//!          --layers N --lr F --warmup N --decay-to F --clip F --eval-every N
//!          --sample N --temperature F --prompt TEXT --seed N --save DIR
//!          --load DIR --threads N
//!
//! `--lr` is the *peak* rate: `--warmup N` climbs to it over the first N
//! steps and `--decay-to F` falls to F times it by the last, over a cosine.
//! `--warmup 0 --decay-to 1` is a flat rate, which is what this binary did
//! before the schedule existed. `--clip 0` turns gradient clipping off.
//! See `optim::Schedule` for what each is for, and the README for what each
//! measured.
//!
//! Training uses every core unless told otherwise. `--threads 1` is the plain
//! loop in `text::train_step`; more is `text::Replicas`. A run is reproducible
//! for a given seed *and* number of threads.
//!
//! With `--load`, the model's shape comes from the checkpoint, and --context,
//! --d-model, --heads and --layers are ignored.
//!
//! `--save` writes the best model this run saw, not the last one: every time
//! the validation loss improves, the directory is rewritten. See
//! `text::Training` for why, and for what "best" means after a `--load`.
//!
//! This binary is the `nanograd` way in. The same loop, with trained models
//! given names and a home of their own, is `kvad train`.

use nanograd::checkpoint;
use nanograd::model::{Gpt, GptConfig};
use nanograd::rng::Rng;
use nanograd::text::{
    evaluate, generate, human_secs, train, unigram_loss, CharTokenizer, Corpus, Report, Training,
};
use std::path::PathBuf;

struct Args {
    data: Option<PathBuf>,
    context: usize,
    d_model: usize,
    heads: usize,
    layers: usize,
    sample: usize,
    temperature: f32,
    prompt: Option<String>,
    seed: u64,
    load: Option<PathBuf>,
    training: Training,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            data: None,
            context: 64,
            d_model: 64,
            heads: 4,
            layers: 2,
            sample: 200,
            temperature: 0.8,
            prompt: None,
            seed: 1337,
            load: None,
            // Steps, batch, learning rate, evaluation and every core there is.
            training: Training::default(),
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        let value = |i: usize| -> String {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {flag}");
                std::process::exit(2);
            })
        };
        let parse = |i: usize| -> f64 {
            value(i).parse().unwrap_or_else(|_| {
                eprintln!("{flag} expects a number");
                std::process::exit(2);
            })
        };
        match argv[i].as_str() {
            "--data" => a.data = Some(PathBuf::from(value(i))),
            "--steps" => a.training.steps = parse(i) as usize,
            "--batch" => a.training.batch = parse(i) as usize,
            "--context" => a.context = parse(i) as usize,
            "--d-model" => a.d_model = parse(i) as usize,
            "--heads" => a.heads = parse(i) as usize,
            "--layers" => a.layers = parse(i) as usize,
            "--lr" => a.training.lr = parse(i) as f32,
            "--warmup" => a.training.warmup = Some(parse(i) as usize),
            "--decay-to" => a.training.decay_to = parse(i) as f32,
            // Zero is not a clip anyone would want, so it means "none".
            "--clip" => a.training.clip = Some(parse(i) as f32).filter(|&c| c > 0.0),
            "--eval-every" => a.training.eval_every = parse(i) as usize,
            "--sample" => a.sample = parse(i) as usize,
            "--temperature" => a.temperature = parse(i) as f32,
            "--prompt" => a.prompt = Some(value(i)),
            "--seed" => a.seed = parse(i) as u64,
            "--save" => a.training.save = Some(PathBuf::from(value(i))),
            "--load" => a.load = Some(PathBuf::from(value(i))),
            "--threads" => a.training.threads = (parse(i) as usize).max(1),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    a
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let mut rng = Rng::new(args.seed);

    // A saved model comes with the tokeniser it was trained with, and only
    // that one will do: id 17 means whatever was seventeenth in *its* text.
    let loaded = match &args.load {
        Some(dir) => {
            let both = checkpoint::load(dir).and_then(|model| Ok((model, CharTokenizer::load(dir)?)));
            let (model, tok) = both.unwrap_or_else(|e| {
                eprintln!("could not load a model: {e}");
                std::process::exit(1);
            });
            if model.config().vocab != tok.vocab() {
                eprintln!("{}: the model and the tokeniser disagree about the vocabulary", dir.display());
                std::process::exit(1);
            }
            println!("loaded {}", dir.display());
            Some((model, tok))
        }
        None => None,
    };

    // Nothing to learn from is needed just to write.
    if args.training.steps == 0 {
        let Some((mut model, tok)) = loaded else {
            eprintln!("--steps 0 trains nothing; it is only useful with --load");
            std::process::exit(2);
        };
        println!("model: {}", model.summary());
        let prompt = encode_prompt(&tok, args.prompt.as_deref().unwrap_or("\n"), args.prompt.is_none());
        let out = generate(&mut model, &prompt, args.sample, args.temperature, &mut rng);
        println!("---\n{}\n---", tok.decode(&out).trim());
        return Ok(());
    }

    let path = args
        .data
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/input.txt"));
    let text = std::fs::read_to_string(&path).map_err(|e| {
        eprintln!("could not read {}: {e}\nrun ./scripts/get-text.sh, or pass --data FILE", path.display());
        e
    })?;

    let (mut model, tok) = loaded.unwrap_or_else(|| {
        let tok = CharTokenizer::from_text(&text);
        let config = GptConfig {
            vocab: tok.vocab(),
            context: args.context,
            d_model: args.d_model,
            n_heads: args.heads,
            n_layers: args.layers,
        };
        (Gpt::new(config, &mut rng), tok)
    });
    // Only a loaded tokeniser can fail here. The vocabulary is as fixed as the
    // weights are: a character the model never saw has no row in its tables.
    let tokens = tok.encode(&text).unwrap_or_else(|c| {
        eprintln!("{} contains {c:?}, which the loaded model has no token for", path.display());
        std::process::exit(1);
    });
    let corpus = Corpus::new(tokens, 0.1);
    println!(
        "text: {} characters, {} distinct   train: {}   validation: {}",
        corpus.train.len() + corpus.val.len(),
        tok.vocab(),
        corpus.train.len(),
        corpus.val.len()
    );

    let cfg = &args.training;
    let threads = cfg.threads.min(cfg.batch);
    println!("model: {}", model.summary());
    let schedule = cfg.schedule();
    println!(
        "hyperparams: steps={} batch={} lr={} warmup={} decay-to={} clip={} threads={threads}",
        cfg.steps,
        cfg.batch,
        cfg.lr,
        schedule.warmup,
        cfg.decay_to,
        cfg.clip.map_or("none".to_string(), |c| c.to_string())
    );

    // Two numbers to hold the loss against. A model that knows nothing scores
    // the first; one that knows only which characters are common scores the
    // second. Anything below that was learned from the *order* of the text.
    println!(
        "loss to beat: {:.3} knowing nothing, {:.3} knowing only letter frequencies",
        (tok.vocab() as f32).ln(),
        unigram_loss(&corpus.train, tok.vocab())
    );
    if args.load.is_some() {
        let val = evaluate(&mut model, &corpus.val, 50, &mut rng);
        println!("validation loss as loaded: {val:.3}");
    }
    println!();

    // A newline, if the text has one: "start a fresh line".
    let prompt = encode_prompt(&tok, args.prompt.as_deref().unwrap_or("\n"), args.prompt.is_none());

    let mut sampler = Rng::new(args.seed ^ 0x5a5a);
    let done = train(&mut model, &tok, &corpus, cfg, &mut rng, &mut |report| match report {
        // Only when there is time to act on it. A run that is already over
        // does not need an estimate of when it will be.
        Report::Pace { chars_per_sec, remaining_secs } if remaining_secs >= 20.0 => {
            println!("at {chars_per_sec:.0} chars/s here, about {} to go\n", human_secs(remaining_secs));
        }
        Report::Pace { .. } => {}
        // Unreachable from here: this binary always starts a new model, so
        // `already_trained` is never set. It is written out rather than
        // waved away with a wildcard, so that the day it grows a `--from`
        // the number turns up instead of being swallowed.
        Report::Baseline { val_loss } => {
            println!("the model being continued scores {val_loss:.3}; a checkpoint has to beat that\n");
        }
        Report::Step { step, train_loss, val_loss, best, saved, elapsed_secs, chars_per_sec, model } => {
            let mark = if saved { "  *saved" } else if best { "  *best" } else { "" };
            println!(
                "step {step:>5}  train loss {train_loss:.3}  validation loss {val_loss:.3}  ({elapsed_secs:.0}s, {chars_per_sec:.0} chars/s){mark}"
            );
            if args.sample > 0 {
                let out = generate(model, &prompt, args.sample, args.temperature, &mut sampler);
                println!("---\n{}\n---\n", tok.decode(&out).trim());
            }
        }
    })?;

    if let Some(dir) = &cfg.save {
        println!(
            "saved to {}: step {} of {}, validation loss {:.3} (last was {:.3})",
            dir.display(),
            done.best_step,
            cfg.steps,
            done.best_val,
            done.last_val
        );
    }
    Ok(())
}

/// The prompt as token ids. The default prompt is allowed to be missing from
/// the vocabulary, and falls back to token 0; one the user typed is not.
fn encode_prompt(tok: &CharTokenizer, prompt: &str, is_default: bool) -> Vec<usize> {
    match tok.encode(prompt) {
        Ok(ids) if !ids.is_empty() => ids,
        _ if is_default => vec![0],
        Ok(_) => {
            eprintln!("the prompt is empty");
            std::process::exit(2);
        }
        Err(c) => {
            eprintln!("the prompt contains {c:?}, which is not in the model's vocabulary");
            std::process::exit(2);
        }
    }
}
