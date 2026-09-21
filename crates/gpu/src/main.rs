//! The same models, on the GPU.
//!
//!     kvad-gpu run  [--model REPO] [--prompt TEXT] [--device metal] [--dtype bf16]
//!     kvad-gpu chat [--model REPO] [--system TEXT]
//!
//! Everything except the forward pass is shared with the CPU engine: the same
//! Hub client, tokenizer, chat template, sampler and generation loop. Only the
//! [`Session`](kvad::model::Session) behind it changes, which is the whole
//! point of that trait.

use kvad_gpu::model;

use kvad::chat::Message;
use kvad::hub::State;
use kvad::runtime::Llm;
use kvad::sampler::Sampler;
use std::io::{BufRead, Write};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const DEFAULT_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

struct Args {
    command: String,
    model: Option<String>,
    prompt: Option<String>,
    system: Option<String>,
    device: Option<String>,
    dtype: String,
    quant: String,
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
            model: None,
            prompt: None,
            system: None,
            device: None,
            // bf16 is what these checkpoints ship as, so it is both the
            // smallest and the most faithful default.
            dtype: "bf16".into(),
            quant: "none".into(),
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
        "usage: kvad-gpu <run|chat> [options]\n\n\
         options:\n  \
           --model REPO        HuggingFace repo id (default: active, else {DEFAULT_MODEL})\n  \
           --device D          metal | cuda | cpu (default: best available)\n  \
           --dtype T           bf16 | f16 | f32 (default bf16)\n  \
           --quant Q           none | q8 | q4 | q4k | q6k (default none)\n  \
           --prompt TEXT       prompt for `run`\n  \
           --system TEXT       system prompt for `chat`\n  \
           --max-tokens N      generation budget (default 256)\n  \
           --temperature F     0 is greedy (default 0.7)\n  \
           --top-k N / --top-p F / --seed N\n  \
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
            "--model" => a.model = Some(value.clone()),
            "--device" => a.device = Some(value.clone()),
            "--dtype" => a.dtype = value.clone(),
            "--quant" => a.quant = value.clone(),
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
    let repo = args
        .model
        .clone()
        .or_else(State::active)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let dtype = model::parse_dtype(&args.dtype)
        .ok_or_else(|| format!("unknown dtype `{}` (try bf16, f16 or f32)", args.dtype))?;
    let quant = model::parse_quant(&args.quant)
        .ok_or_else(|| format!("unknown quant `{}` (try none, q8, q4, q4k or q6k)", args.quant))?;
    let device = model::pick_device(args.device.as_deref())?;

    eprintln!("model: {repo}");
    let t0 = std::time::Instant::now();
    // No watcher: a terminal has the progress lines and wants no bar.
    let watch = kvad::weights::Watcher::none();
    let llm = Llm::load_custom(&repo, &mut |m| eprintln!("  {m}"), &watch, &mut |files, spec, _| {
        model::session(&files.weights, spec, dtype, quant, device.clone())
    })?;

    eprintln!("  {}", llm.spec.summary());
    eprintln!(
        "  {:.1}M parameters · weights {:.0} MB ({}) · loaded in {:.1}s",
        llm.param_count as f64 / 1e6,
        llm.weight_bytes as f64 / 1e6,
        llm.backend(),
        t0.elapsed().as_secs_f32()
    );
    Ok(llm)
}

// Reporting the error rather than returning it from `main`, because returning
// it prints the `Debug` form — and for a `Box<dyn Error>` made from a `String`
// that is the quoted, backslash-escaped spelling. The loader's errors are
// several lines of prose meant to be read: the unread-tensor guard names the
// weights it found, and the quantiser names the block size that does not fit.
fn main() {
    let args = parse_args();
    let done = match args.command.as_str() {
        "run" => run(args),
        "chat" => chat(args),
        other => {
            eprintln!("unknown command `{other}`");
            usage();
        }
    };
    if let Err(e) = done {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Res<()> {
    let mut llm = load(&args)?;
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
        llm.encode_chat(&messages, &[])?
    } else {
        print!("{prompt}");
        std::io::stdout().flush()?;
        llm.encode(&prompt)?
    };
    eprintln!("  prompt is {} tokens\n", ids.len());

    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let (stats, _) = llm.generate(&ids, &mut sampler, args.max_tokens, |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
        true
    })?;

    eprintln!(
        "\n\n[prefill {} tokens in {:.2}s · generated {} in {:.2}s = {:.1} tok/s]",
        stats.prompt_tokens,
        stats.prefill_secs,
        stats.generated_tokens,
        stats.decode_secs,
        stats.tokens_per_sec()
    );
    Ok(())
}

fn chat(args: Args) -> Res<()> {
    let mut llm = load(&args)?;
    if !llm.is_instruct() {
        eprintln!("\nwarning: this is a base model with no chat template; it will continue text.");
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
                llm.reset()?;
                eprintln!("(history cleared)");
                continue;
            }
            _ => {}
        }

        messages.push(Message::user(line));
        let ids = llm.encode_chat(&messages, &[])?;
        let mut reply = String::new();
        let (stats, _) = llm.generate(&ids, &mut sampler, args.max_tokens, |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            reply.push_str(piece);
            true
        })?;
        println!();
        eprintln!(
            "  [{} tok, {:.1} tok/s, {} cached]\n",
            stats.generated_tokens,
            stats.tokens_per_sec(),
            stats.cached_tokens
        );
        messages.push(Message::assistant(reply));
    }
    Ok(())
}
