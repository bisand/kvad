//! Command-line front end.
//!
//!     llm run   [--model REPO] [--prompt TEXT] [flags]   one-shot completion
//!     llm chat  [--model REPO] [--system TEXT]           interactive
//!     llm info  [--model REPO]                           config only, no weights
//!
//! Sampling flags: --max-tokens N --temperature F --top-k N --top-p F --seed N
//! --greedy

use llm::chat::Message;
use llm::model::{KvCache, Spec};
use llm::runtime::Llm;
use llm::sampler::Sampler;
use llm::weights;
use std::io::{BufRead, Write};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A small instruction-tuned model: 135M parameters, genuinely converses, and
/// downloads in seconds. See the README for why the default is not GPT-2.
const DEFAULT_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

struct Args {
    command: String,
    model: String,
    prompt: Option<String>,
    system: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            command: "run".into(),
            model: DEFAULT_MODEL.into(),
            prompt: None,
            system: None,
            max_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            seed: 7,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: llm <run|chat|info> [options]\n\n\
         options:\n  \
           --model REPO        HuggingFace repo id (default {DEFAULT_MODEL})\n  \
           --prompt TEXT       prompt for `run`\n  \
           --system TEXT       system prompt for `chat`\n  \
           --max-tokens N      generation budget (default 256)\n  \
           --temperature F     0 is greedy (default 0.7)\n  \
           --top-k N           keep the N best candidates (default 40)\n  \
           --top-p F           nucleus threshold (default 0.95)\n  \
           --seed N            sampling seed (default 7)\n  \
           --greedy            shorthand for --temperature 0"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;

    if let Some(first) = argv.first() {
        if !first.starts_with("--") {
            a.command = first.clone();
            i = 1;
        }
    }
    if matches!(a.command.as_str(), "-h" | "--help" | "help") {
        usage();
    }

    while i < argv.len() {
        let flag = argv[i].clone();
        if flag == "--greedy" {
            a.temperature = 0.0;
            i += 1;
            continue;
        }
        let Some(value) = argv.get(i + 1).cloned() else {
            eprintln!("missing value for {flag}");
            usage();
        };
        let num = || -> f64 {
            value.parse().unwrap_or_else(|_| {
                eprintln!("{flag} expects a number, got `{value}`");
                std::process::exit(2);
            })
        };
        match flag.as_str() {
            "--model" => a.model = value.clone(),
            "--prompt" => a.prompt = Some(value.clone()),
            "--system" => a.system = Some(value.clone()),
            "--max-tokens" => a.max_tokens = num() as usize,
            "--temperature" => a.temperature = num() as f32,
            "--top-k" => a.top_k = num() as usize,
            "--top-p" => a.top_p = num() as f32,
            "--seed" => a.seed = num() as u64,
            _ => {
                eprintln!("unknown flag {flag}");
                usage();
            }
        }
        i += 2;
    }
    a
}

fn load(args: &Args) -> Res<Llm> {
    eprintln!("model: {}", args.model);
    let t0 = std::time::Instant::now();
    let llm = Llm::load(&args.model)?;
    eprintln!("  {}", llm.spec.summary());
    eprintln!(
        "  {:.1}M parameters in {:.1}s · KV cache at full context {:.0} MB · {}",
        llm.param_count as f64 / 1e6,
        t0.elapsed().as_secs_f32(),
        KvCache::max_bytes(&llm.spec) as f64 / 1e6,
        if llm.is_instruct() { "instruction-tuned" } else { "base model (completion only)" }
    );
    Ok(llm)
}

fn main() -> Res<()> {
    let args = parse_args();

    match args.command.as_str() {
        "info" => {
            let files = weights::fetch(&args.model)?;
            let spec = Spec::from_json(&files.config)?;
            println!("{}", spec.summary());
            println!("  weight files: {}", files.weights.len());
            println!(
                "  KV cache at full context: {:.0} MB",
                KvCache::max_bytes(&spec) as f64 / 1e6
            );
            println!(
                "  chat template: {}",
                match &files.tokenizer_config {
                    Some(p) => match llm::chat::ChatTemplate::from_tokenizer_config(p)? {
                        Some(_) => "yes (instruction-tuned)",
                        None => "no (base model)",
                    },
                    None => "no (base model)",
                }
            );
            Ok(())
        }
        "run" => run(args),
        "chat" => chat(args),
        other => {
            eprintln!("unknown command `{other}`");
            usage();
        }
    }
}

fn run(args: Args) -> Res<()> {
    let llm = load(&args)?;

    // An instruction-tuned model given a bare prompt still needs its template,
    // or it falls back to base-model behaviour and rambles.
    let prompt = args.prompt.clone().unwrap_or_else(|| {
        if llm.is_instruct() {
            "Explain what a neural network is, in two sentences.".into()
        } else {
            "The first time I saw the sea,".into()
        }
    });

    let ids = if llm.is_instruct() {
        let mut messages = Vec::new();
        if let Some(s) = &args.system {
            messages.push(Message::system(s.clone()));
        }
        messages.push(Message::user(prompt.clone()));
        llm.encode_chat(&messages)?
    } else {
        print!("{prompt}");
        std::io::stdout().flush()?;
        llm.encode(&prompt)?
    };
    eprintln!("  prompt is {} tokens\n", ids.len());

    let mut cache = llm.new_cache();
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let stats = llm.generate(&ids, &mut cache, &mut sampler, args.max_tokens, |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
        true
    })?;

    eprintln!(
        "\n\n[prefill {} tokens in {:.2}s · generated {} in {:.2}s = {:.1} tok/s]",
        stats.prompt_tokens, stats.prefill_secs, stats.generated_tokens, stats.decode_secs,
        stats.tokens_per_sec()
    );
    Ok(())
}

fn chat(args: Args) -> Res<()> {
    let llm = load(&args)?;
    if !llm.is_instruct() {
        eprintln!(
            "\nwarning: {} is a base model with no chat template. It will continue\n\
             text rather than answer questions. Try --model {DEFAULT_MODEL}.",
            args.model
        );
    }

    let mut messages = Vec::new();
    if let Some(s) = &args.system {
        messages.push(Message::system(s.clone()));
    }

    eprintln!("\nType a message, or /reset to clear history, /quit to exit.\n");
    let stdin = std::io::stdin();
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);

    loop {
        print!("\x1b[1m>\x1b[0m ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                messages.retain(|m: &Message| m.role == "system");
                eprintln!("(history cleared)");
                continue;
            }
            _ => {}
        }

        messages.push(Message::user(line));
        let ids = llm.encode_chat(&messages)?;

        // Re-encoding and re-prefilling the whole conversation each turn is
        // the simple thing, and wrong at scale: turn N re-reads turns 1..N-1.
        // Keeping the cache across turns is the fix, and is what the TUI does.
        let mut cache = llm.new_cache();
        let mut reply = String::new();
        let stats = llm.generate(&ids, &mut cache, &mut sampler, args.max_tokens, |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            reply.push_str(piece);
            true
        })?;
        println!();
        eprintln!("  [{} tok, {:.1} tok/s]\n", stats.generated_tokens, stats.tokens_per_sec());
        messages.push(Message::assistant(reply));
    }
    Ok(())
}
