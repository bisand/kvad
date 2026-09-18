//! Command-line front end.
//!
//!     llm search QUERY      find models on the Hub, flagging which we can run
//!     llm pull REPO         download a model into the local cache
//!     llm ls                list downloaded models
//!     llm use REPO          set the default model
//!     llm rm REPO           delete a model from the cache
//!     llm info [--model R]  read the config without downloading weights
//!     llm run  [--model R] [--prompt TEXT]
//!     llm chat [--model R] [--system TEXT]
//!
//! Sampling flags: --max-tokens N --temperature F --top-k N --top-p F --seed N
//! --greedy

use llm::chat::Message;
use llm::hub::{self, State};
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
    /// Positional argument after the subcommand: a query for `search`, a repo
    /// id for `pull` / `use` / `rm`.
    target: Option<String>,
    model: Option<String>,
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
            target: None,
            model: None,
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
        "usage: llm <command> [options]\n\n\
         commands:\n  \
           search QUERY        find models on the Hub\n  \
           pull REPO           download a model\n  \
           ls                  list downloaded models\n  \
           use REPO            set the default model\n  \
           rm REPO             delete a model from the cache\n  \
           info                show a model's config without downloading weights\n  \
           run                 one-shot completion\n  \
           chat                interactive conversation\n\n\
         options:\n  \
           --model REPO        HuggingFace repo id (default: active, else {DEFAULT_MODEL})\n  \
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

    // Subcommands that take a bare positional argument.
    if matches!(a.command.as_str(), "search" | "pull" | "use" | "rm") {
        let mut words = Vec::new();
        while let Some(w) = argv.get(i) {
            if w.starts_with("--") {
                break;
            }
            words.push(w.clone());
            i += 1;
        }
        if !words.is_empty() {
            a.target = Some(words.join(" "));
        }
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

/// Which model to use: the flag, else whatever `llm use` selected, else the
/// built-in default.
fn resolve_model(args: &Args) -> String {
    args.model
        .clone()
        .or_else(State::active)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

fn load(args: &Args) -> Res<Llm> {
    let repo = resolve_model(args);
    eprintln!("model: {repo}");
    let t0 = std::time::Instant::now();
    let llm = Llm::load(&repo)?;
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
            let files = weights::fetch(&resolve_model(&args))?;
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
        "search" => search(args),
        "pull" => pull(args),
        "ls" => list_local(),
        "use" => use_model(args),
        "rm" => remove(args),
        "run" => run(args),
        "chat" => chat(args),
        other => {
            eprintln!("unknown command `{other}`");
            usage();
        }
    }
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

fn search(args: Args) -> Res<()> {
    let Some(query) = args.target else {
        eprintln!("usage: llm search QUERY");
        std::process::exit(2);
    };

    let results = hub::search(&query, 20)?;
    if results.is_empty() {
        println!("no models matched `{query}`");
        return Ok(());
    }

    let local: Vec<String> = hub::local_models().into_iter().map(|m| m.id).collect();
    println!(
        "{:<46} {:>10}  {:<7} {}",
        "MODEL", "DOWNLOADS", "ARCH", "STATUS"
    );
    for m in &results {
        let status = match m.blocker() {
            Some(reason) => reason,
            None if local.contains(&m.id) => "downloaded".into(),
            None if m.looks_instruct => "runnable · chat".into(),
            None => "runnable · completion".into(),
        };
        // Names come from the Hub; print them, never act on them.
        println!(
            "{:<46} {:>10}  {:<7} {}",
            truncate(&m.id, 46),
            m.downloads,
            m.arch.map(|a| a.to_string()).unwrap_or_else(|| "-".into()),
            status
        );
    }

    let runnable = results.iter().filter(|m| m.runnable()).count();
    println!("\n{runnable} of {} runnable here.", results.len());
    Ok(())
}

fn pull(args: Args) -> Res<()> {
    let Some(repo) = args.target.as_deref().map(str::to_string).or_else(|| args.model.clone())
    else {
        eprintln!("usage: llm pull REPO");
        std::process::exit(2);
    };

    // Read the config first: no point downloading gigabytes for an
    // architecture we cannot run.
    eprintln!("pulling {repo}");
    let files = weights::fetch(&repo)?;
    let spec = Spec::from_json(&files.config)?;
    println!("  {}", spec.summary());

    let instruct = match &files.tokenizer_config {
        Some(p) => llm::chat::ChatTemplate::from_tokenizer_config(p)?.is_some(),
        None => false,
    };
    println!(
        "  {}",
        if instruct {
            "instruction-tuned — usable with `llm chat`"
        } else {
            "base model — completion only, will not answer questions"
        }
    );

    if let Some(local) = hub::find_local(&repo) {
        println!("  {} on disk", hub::human_bytes(local.bytes));
    }
    println!("\nrun it with:  llm run --model {repo}");
    Ok(())
}

fn list_local() -> Res<()> {
    let models = hub::local_models();
    if models.is_empty() {
        println!("no models downloaded yet. try:  llm search smollm");
        return Ok(());
    }

    let active = State::active();
    let mut total = 0;
    println!("{:<46} {:<7} {:>9}", "MODEL", "ARCH", "SIZE");
    for m in &models {
        total += m.bytes;
        let marker = if active.as_deref() == Some(m.id.as_str()) { " *" } else { "" };
        println!(
            "{:<46} {:<7} {:>9}{}{}",
            truncate(&m.id, 46),
            m.arch.map(|a| a.to_string()).unwrap_or_else(|| "?".into()),
            hub::human_bytes(m.bytes),
            marker,
            if m.complete { "" } else { "  (config only)" }
        );
    }
    println!("\n{} models, {}", models.len(), hub::human_bytes(total));
    if let Some(a) = active {
        println!("* active: {a}");
    } else {
        println!("no active model set (using {DEFAULT_MODEL})");
    }
    println!("cache: {}", hub::cache_dir().display());
    Ok(())
}

fn use_model(args: Args) -> Res<()> {
    let Some(repo) = args.target else {
        eprintln!("usage: llm use REPO");
        std::process::exit(2);
    };
    if hub::find_local(&repo).is_none() {
        eprintln!("`{repo}` is not downloaded. Run:  llm pull {repo}");
        std::process::exit(1);
    }
    State::set_active(&repo)?;
    println!("active model is now {repo}");
    Ok(())
}

fn remove(args: Args) -> Res<()> {
    let Some(repo) = args.target else {
        eprintln!("usage: llm rm REPO");
        std::process::exit(2);
    };
    let Some(local) = hub::find_local(&repo) else {
        eprintln!("`{repo}` is not in the cache. Run `llm ls` to see what is.");
        std::process::exit(1);
    };

    // Deleting is irreversible and the download may have been slow, so say
    // exactly what will go and require a typed yes.
    println!("about to delete:");
    println!("  {}", local.path.display());
    println!("  {} ({})", local.id, hub::human_bytes(local.bytes));
    print!("\nre-download would be needed to use it again. delete? [y/N] ");
    std::io::stdout().flush()?;

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        println!("cancelled");
        return Ok(());
    }

    std::fs::remove_dir_all(&local.path)?;
    if State::active().as_deref() == Some(local.id.as_str()) {
        State::clear()?;
        println!("(was the active model; cleared)");
    }
    println!("deleted {}", local.id);
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
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
    let (stats, _) = llm.generate(&ids, &mut cache, &mut sampler, args.max_tokens, |piece| {
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
            resolve_model(&args)
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
        let (stats, _) = llm.generate(&ids, &mut cache, &mut sampler, args.max_tokens, |piece| {
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
