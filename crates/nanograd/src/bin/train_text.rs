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
//!          --layers N --lr F --eval-every N --sample N --temperature F
//!          --prompt TEXT --seed N --save DIR --load DIR --threads N
//!
//! Training uses every core unless told otherwise. `--threads 1` is the plain
//! loop in `text::train_step`; more is `text::Replicas`. A run is reproducible
//! for a given seed *and* number of threads.
//!
//! With `--load`, the model's shape comes from the checkpoint, and --context,
//! --d-model, --heads and --layers are ignored.

use nanograd::checkpoint;
use nanograd::model::{Gpt, GptConfig};
use nanograd::optim::AdamW;
use nanograd::rng::Rng;
use nanograd::text::{evaluate, generate, train_step, unigram_loss, CharTokenizer, Corpus, Replicas};
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
    save: Option<PathBuf>,
    load: Option<PathBuf>,
    threads: usize,
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
            save: None,
            load: None,
            // Every core there is. A batch cannot be split finer than one
            // window to a thread, which `main` sees to.
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
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
            "--save" => a.save = Some(PathBuf::from(value(i))),
            "--load" => a.load = Some(PathBuf::from(value(i))),
            "--threads" => a.threads = (parse(i) as usize).max(1),
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
    if args.steps == 0 {
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

    let context = model.config().context;
    let threads = args.threads.min(args.batch);
    let mut opt = AdamW::new(args.lr);
    println!("model: {}", model.summary());
    println!("hyperparams: steps={} batch={} lr={} threads={threads}", args.steps, args.batch, args.lr);

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

    let mut replicas = (threads > 1).then(|| Replicas::new(&model, threads));
    let started = Instant::now();
    let mut running = 0.0;
    for step in 1..=args.steps {
        running += match &mut replicas {
            Some(replicas) => replicas.train_step(&mut model, &mut opt, &corpus.train, args.batch, &mut rng),
            None => train_step(&mut model, &mut opt, &corpus.train, args.batch, &mut rng),
        };

        if step % args.eval_every == 0 || step == args.steps {
            let since = if step % args.eval_every == 0 { args.eval_every } else { step % args.eval_every };
            let elapsed = started.elapsed().as_secs_f32();
            let val = evaluate(&mut model, &corpus.val, 50, &mut rng);
            println!(
                "step {step:>5}  train loss {:.3}  validation loss {val:.3}  ({elapsed:.0}s, {:.0} chars/s)",
                running / since as f32,
                (step * args.batch * context) as f32 / elapsed
            );
            running = 0.0;

            if args.sample > 0 {
                let out = generate(&mut model, &prompt, args.sample, args.temperature, &mut rng);
                println!("---\n{}\n---\n", tok.decode(&out).trim());
            }
        }
    }

    if let Some(dir) = &args.save {
        checkpoint::save(dir, &mut model)?;
        tok.save(dir)?;
        println!("saved to {}", dir.display());
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
