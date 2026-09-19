//! Command-line front end.
//!
//!     kvad search QUERY      find models on the Hub, flagging which we can run
//!     kvad pull REPO         download a model into the local cache
//!     kvad ls                list downloaded models
//!     kvad use REPO          set the default model
//!     kvad rm REPO           delete a model from the cache
//!     kvad cache [REPO]      list (or delete) pre-quantised weight files
//!     kvad info [--model R]  read the config without downloading weights
//!     kvad run  [--model R] [--prompt TEXT]
//!     kvad chat [--model R] [--system TEXT]
//!
//! Sampling flags: --max-tokens N --temperature F --top-k N --top-p F --seed N
//! --greedy

use kvad::chat::Message;
use kvad::hub::{self, State};
use kvad::model::{KvCache, Spec};
use kvad::qcache;
use kvad::quant::Precision;
use kvad::runtime::Llm;
use kvad::sampler::Sampler;
use kvad::weights;
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
    quant: Precision,
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
            quant: Precision::F32,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: kvad <command> [options]\n\n\
         commands:\n  \
           search QUERY        find models on the Hub\n  \
           pull REPO           download a model\n  \
           ls                  list downloaded models\n  \
           use REPO            set the default model\n  \
           rm REPO             delete a model from the cache\n  \
           cache [REPO|clear]  list or delete pre-quantised weight files\n  \
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
           --quant f32|q8|q4   quantise weights on load (default f32)\n  \
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
    if matches!(a.command.as_str(), "search" | "pull" | "use" | "rm" | "cache") {
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
            "--quant" => {
                a.quant = Precision::parse(&value).unwrap_or_else(|| {
                    eprintln!("--quant expects f32, q8 or q4, got `{value}`");
                    std::process::exit(2);
                })
            }
            _ => {
                eprintln!("unknown flag {flag}");
                usage();
            }
        }
        i += 2;
    }
    a
}

/// Which model to use: the flag, else whatever `kvad use` selected, else the
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
    // Timestamped progress, because the interesting question about a load is
    // not how long it took but which part of it took that long.
    let t0 = std::time::Instant::now();
    let llm = Llm::load_with(&repo, args.quant, &mut |msg| {
        eprintln!("  [{:>6.3}s] {msg}", t0.elapsed().as_secs_f32())
    })?;
    eprintln!("  {}", llm.spec.summary());
    eprintln!(
        "  {:.1}M parameters · weights {:.0} MB ({}) · loaded in {:.1}s",
        llm.param_count as f64 / 1e6,
        llm.weight_bytes as f64 / 1e6,
        llm.backend(),
        t0.elapsed().as_secs_f32(),
    );
    eprintln!(
        "  KV cache at full context {:.0} MB · {}",
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
                    Some(p) => match kvad::chat::ChatTemplate::from_tokenizer_config(p)? {
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
        "cache" => cache(args),
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
        eprintln!("usage: kvad search QUERY");
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
        eprintln!("usage: kvad pull REPO");
        std::process::exit(2);
    };

    // Read the config first: no point downloading gigabytes for an
    // architecture we cannot run.
    eprintln!("pulling {repo}");
    let files = weights::fetch(&repo)?;
    let spec = Spec::from_json(&files.config)?;
    println!("  {}", spec.summary());

    let instruct = match &files.tokenizer_config {
        Some(p) => kvad::chat::ChatTemplate::from_tokenizer_config(p)?.is_some(),
        None => false,
    };
    println!(
        "  {}",
        if instruct {
            "instruction-tuned — usable with `kvad chat`"
        } else {
            "base model — completion only, will not answer questions"
        }
    );

    if let Some(local) = hub::find_local(&repo) {
        println!("  {} on disk", hub::human_bytes(local.bytes));
    }
    println!("\nrun it with:  kvad run --model {repo}");
    Ok(())
}

fn list_local() -> Res<()> {
    let models = hub::local_models();
    if models.is_empty() {
        println!("no models downloaded yet. try:  kvad search smollm");
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
        eprintln!("usage: kvad use REPO");
        std::process::exit(2);
    };
    if hub::find_local(&repo).is_none() {
        eprintln!("`{repo}` is not downloaded. Run:  kvad pull {repo}");
        std::process::exit(1);
    }
    State::set_active(&repo)?;
    println!("active model is now {repo}");
    Ok(())
}

fn remove(args: Args) -> Res<()> {
    let Some(repo) = args.target else {
        eprintln!("usage: kvad rm REPO");
        std::process::exit(2);
    };
    let Some(local) = hub::find_local(&repo) else {
        eprintln!("`{repo}` is not in the cache. Run `kvad ls` to see what is.");
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
    // The quantised copies are derived from what just went; leaving them
    // would be a cache with nothing behind it.
    match qcache::forget(&local.id) {
        0 => {}
        n => println!("also deleted {n} pre-quantised file(s)"),
    }
    if State::active().as_deref() == Some(local.id.as_str()) {
        State::clear()?;
        println!("(was the active model; cleared)");
    }
    println!("deleted {}", local.id);
    Ok(())
}

/// `kvad cache` — what has been pre-quantised, and how to get rid of it.
///
/// These files are pure derived data: deleting one costs a few seconds on the
/// next load of that model and nothing else, which is why there is no
/// confirmation prompt here and there is one on `kvad rm`.
fn cache(args: Args) -> Res<()> {
    match args.target.as_deref() {
        Some("clear") => {
            let n = qcache::entries()
                .into_iter()
                .filter(|(path, ..)| std::fs::remove_file(path).is_ok())
                .count();
            println!("deleted {n} file(s)");
            return Ok(());
        }
        Some(repo) => {
            println!("deleted {} file(s) for {repo}", qcache::forget(repo));
            return Ok(());
        }
        None => {}
    }

    let entries = qcache::entries();
    if entries.is_empty() {
        println!("nothing pre-quantised yet.");
        println!("the first `kvad run --quant q8` writes a file here; later runs map it.");
    } else {
        println!("{:<46} {:<6} {:>10}", "MODEL", "QUANT", "SIZE");
        let mut total = 0;
        for (_, repo, precision, bytes) in &entries {
            let size = hub::human_bytes(*bytes);
            println!("{:<46} {:<6} {:>10}", truncate(repo, 45), precision, size);
            total += bytes;
        }
        println!("\n{} file(s), {}", entries.len(), hub::human_bytes(total));
    }
    println!("cache: {}", qcache::cache_root().display());
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
    let mut llm = load(&args)?;

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

    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);
    let (stats, _) = llm.generate(&ids, &mut sampler, args.max_tokens, |piece| {
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
    let mut llm = load(&args)?;
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
                llm.reset()?;
                eprintln!("(history cleared)");
                continue;
            }
            _ => {}
        }

        messages.push(Message::user(line));
        let ids = llm.encode_chat(&messages)?;

        // The cache lives in the session and survives between turns: only the
        // newest message is prefilled, not the whole transcript.
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
