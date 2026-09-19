//! Train a small GPT on a text file, one character at a time, and watch it
//! learn to write.
//!
//!     ./scripts/get-text.sh
//!     cargo run --release -p nanograd --bin train_text
//!
//! Any plain text file will do: `--data path/to/file.txt`.
//!
//! Options: --data PATH --steps N --batch N --context N --d-model N --heads N
//!          --layers N --lr F --eval-every N --sample N --temperature F
//!          --prompt TEXT --seed N

use nanograd::model::{Gpt, GptConfig};
use nanograd::optim::AdamW;
use nanograd::rng::Rng;
use nanograd::text::{evaluate, generate, train_step, unigram_loss, CharTokenizer, Corpus};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    data: Option<PathBuf>,
    steps: usize,
    batch: usize,
    context: usize,
    d_model: usize,
    heads: usize,
    layers: usize,
    lr: f32,
    eval_every: usize,
    sample: usize,
    temperature: f32,
    prompt: Option<String>,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            data: None,
            steps: 2000,
            batch: 16,
            context: 64,
            d_model: 64,
            heads: 4,
            layers: 2,
            lr: 3e-3,
            eval_every: 250,
            sample: 200,
            temperature: 0.8,
            prompt: None,
            seed: 1337,
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
            "--steps" => a.steps = parse(i) as usize,
            "--batch" => a.batch = parse(i) as usize,
            "--context" => a.context = parse(i) as usize,
            "--d-model" => a.d_model = parse(i) as usize,
            "--heads" => a.heads = parse(i) as usize,
            "--layers" => a.layers = parse(i) as usize,
            "--lr" => a.lr = parse(i) as f32,
            "--eval-every" => a.eval_every = parse(i) as usize,
            "--sample" => a.sample = parse(i) as usize,
            "--temperature" => a.temperature = parse(i) as f32,
            "--prompt" => a.prompt = Some(value(i)),
            "--seed" => a.seed = parse(i) as u64,
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
    let path = args
        .data
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/input.txt"));
    let text = std::fs::read_to_string(&path).map_err(|e| {
        eprintln!("could not read {}: {e}\nrun ./scripts/get-text.sh, or pass --data FILE", path.display());
        e
    })?;

    let tok = CharTokenizer::from_text(&text);
    let tokens = tok.encode(&text).expect("the tokeniser was built from this text");
    let corpus = Corpus::new(tokens, 0.1);
    println!(
        "text: {} characters, {} distinct   train: {}   validation: {}",
        corpus.train.len() + corpus.val.len(),
        tok.vocab(),
        corpus.train.len(),
        corpus.val.len()
    );

    let mut rng = Rng::new(args.seed);
    let config = GptConfig {
        vocab: tok.vocab(),
        context: args.context,
        d_model: args.d_model,
        n_heads: args.heads,
        n_layers: args.layers,
    };
    let mut model = Gpt::new(config, &mut rng);
    let mut opt = AdamW::new(args.lr);
    println!("model: {}", model.summary());
    println!("hyperparams: steps={} batch={} lr={}", args.steps, args.batch, args.lr);

    // Two numbers to hold the loss against. A model that knows nothing scores
    // the first; one that knows only which characters are common scores the
    // second. Anything below that was learned from the *order* of the text.
    println!(
        "loss to beat: {:.3} knowing nothing, {:.3} knowing only letter frequencies\n",
        (tok.vocab() as f32).ln(),
        unigram_loss(&corpus.train, tok.vocab())
    );

    let prompt = match &args.prompt {
        Some(p) => tok.encode(p).unwrap_or_else(|c| {
            eprintln!("the prompt contains {c:?}, which is not in the text");
            std::process::exit(2);
        }),
        // A newline, if the text has one: "start a fresh line".
        None => tok.encode("\n").unwrap_or_else(|_| vec![corpus.train[0]]),
    };

    let started = Instant::now();
    let mut running = 0.0;
    for step in 1..=args.steps {
        running += train_step(&mut model, &mut opt, &corpus.train, args.batch, &mut rng);

        if step % args.eval_every == 0 || step == args.steps {
            let since = if step % args.eval_every == 0 { args.eval_every } else { step % args.eval_every };
            let elapsed = started.elapsed().as_secs_f32();
            let val = evaluate(&mut model, &corpus.val, 50, &mut rng);
            println!(
                "step {step:>5}  train loss {:.3}  validation loss {val:.3}  ({elapsed:.0}s, {:.0} chars/s)",
                running / since as f32,
                (step * args.batch * args.context) as f32 / elapsed
            );
            running = 0.0;

            if args.sample > 0 {
                let out = generate(&mut model, &prompt, args.sample, args.temperature, &mut rng);
                println!("---\n{}\n---\n", tok.decode(&out).trim());
            }
        }
    }
    Ok(())
}
