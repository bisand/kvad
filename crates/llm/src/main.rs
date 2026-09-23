//! Command-line front end.
//!
//!     kvad search QUERY      find models on the Hub, flagging which we can run
//!     kvad pull REPO         download a model into the local cache
//!     kvad crawl URL         read a documentation site into a text file
//!     kvad train --data F    train a model of your own, from a text file
//!     kvad ls                list downloaded and trained models
//!     kvad use REPO          set the default model
//!     kvad rm REPO           delete a model
//!     kvad cache [REPO]      list (or delete) pre-quantised weight files
//!     kvad info [--model R]  read the config without downloading weights
//!     kvad run  [--model R] [--prompt TEXT]
//!     kvad chat [--model R] [--system TEXT]
//!     kvad serve [...]       the HTTP server and web UI
//!     kvad service ...       that server as a service that starts at login
//!
//! Sampling flags: --max-tokens N --temperature F --top-k N --top-p F --seed N
//! --greedy
//!
//! Wherever a model is named, three things are accepted and tried in this
//! order: a directory that exists, a model trained here by that name, a Hub
//! repo id. See `kvad::weights`.
//!
//! # Here, or on the server
//!
//! Most of these commands can be answered two ways: in this process, which is
//! how the CLI has always worked, or by a running `kvad-serve`. When one is
//! running they go to it, so that `kvad chat` talks to the model the service
//! already holds instead of loading a second copy beside it. Which one, and
//! why, is `kvad::client`'s question, and the answer is the first line every
//! such command prints. The commands that only a server can answer — `load`,
//! `conversations`, `keys` and the rest — live in [`cli`].

use kvad::chat::Message;
use kvad::crawl;
use kvad::hub::{self, State};
use kvad::model::{KvCache, Spec};
use kvad::qcache;
use kvad::quant::Precision;
use kvad::runtime::Llm;
use kvad::train;
use kvad::sampler::Sampler;
use kvad::weights;
use std::io::{BufRead, Write};
use std::path::PathBuf;

mod cli;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A small instruction-tuned model: 135M parameters, genuinely converses, and
/// downloads in seconds. See the README for why the default is not GPT-2.
const DEFAULT_MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

struct Args {
    command: String,
    /// Positional argument after the subcommand: a query for `search`, a repo
    /// id for `pull` / `use` / `rm`.
    target: Option<String>,
    /// The same words one at a time, for the commands that take several:
    /// `kvad keys rm 4`, `kvad service logs`.
    words: Vec<String>,
    /// Where to run: `--remote URL`, `--local`, or neither. See
    /// `kvad::client`.
    choice: kvad::client::Choice,
    /// Print what the server sent, as JSON, instead of a table.
    json: bool,
    /// Answer yes to the one question a command would ask.
    yes: bool,
    /// Keep going: `kvad service logs -f`.
    follow: bool,
    /// Asked for with `--help` after a command: that command's usage.
    help: bool,
    /// Whether `--quant` was given, which only matters on a server: there it
    /// picks a CPU backend, and not giving it leaves the server to choose.
    quant_given: bool,
    backend: Option<String>,
    host: Option<String>,
    port: Option<u16>,
    limit: Option<usize>,
    role: Option<String>,
    title: Option<String>,
    dataset: Option<String>,
    conversation: Option<i64>,
    /// `kvad chat --save`: keep the conversation on the server.
    save: bool,
    /// `kvad run --raw`: continue the prompt, with no chat template.
    raw: bool,
    /// `kvad service install --force`: install over the objections.
    force: bool,
    /// `kvad auth login --key`: paste a key rather than sign in.
    key: bool,
    /// `kvad users edit --password`: set a new one.
    password: bool,
    k: Option<usize>,
    window: Option<usize>,
    rounds: Option<usize>,
    tokens: Option<usize>,
    model: Option<String>,
    prompt: Option<String>,
    system: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    quant: Precision,
    /// `crawl` only.
    out: Option<String>,
    pages: Option<usize>,
    megabytes: Option<f64>,
    pause: Option<u64>,
    drop_rare: Option<usize>,
    same_host: bool,
    /// `train` only.
    data: Option<String>,
    from: Option<String>,
    name: Option<String>,
    size: Option<String>,
    steps: Option<usize>,
    batch: Option<usize>,
    lr: Option<f32>,
    eval_every: Option<usize>,
    threads: Option<usize>,
    sample: Option<usize>,
    warmup: Option<usize>,
    decay_to: Option<f32>,
    clip: Option<f32>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            command: "run".into(),
            target: None,
            words: Vec::new(),
            choice: kvad::client::Choice::Unset,
            json: false,
            yes: false,
            follow: false,
            help: false,
            quant_given: false,
            backend: None,
            host: None,
            port: None,
            limit: None,
            role: None,
            title: None,
            dataset: None,
            conversation: None,
            save: false,
            raw: false,
            force: false,
            key: false,
            password: false,
            k: None,
            window: None,
            rounds: None,
            tokens: None,
            model: None,
            prompt: None,
            system: None,
            max_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            seed: 7,
            quant: Precision::F32,
            out: None,
            pages: None,
            megabytes: None,
            pause: None,
            drop_rare: None,
            same_host: false,
            data: None,
            from: None,
            name: None,
            size: None,
            steps: None,
            batch: None,
            lr: None,
            eval_every: None,
            threads: None,
            sample: None,
            warmup: None,
            decay_to: None,
            clip: None,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: kvad <command> [options]\n\n\
         commands:\n  \
           search QUERY        find models on the Hub\n  \
           pull REPO           download a model\n  \
           crawl URL           read a documentation site into a text file\n  \
           train               train a model of your own from a text file\n  \
           ls                  list downloaded and trained models\n  \
           use MODEL           set the default model\n  \
           rm MODEL            delete a downloaded or trained model\n  \
           cache [REPO|clear]  list or delete pre-quantised weight files\n  \
           info [MODEL]        show a model's config without downloading weights
  arch                list the architectures this build can run\n  \
           run                 one-shot completion\n  \
           chat                interactive conversation\n  \
           serve [...]         HTTP server and web UI; options are passed through\n  \
           service ...         run that server at login: status, start, stop,\n  \
           \u{20}                   restart, logs [-f], install, uninstall\n\n\
         commands a running server answers (kvad-serve, or `kvad service start`):\n  \
           ps                  the models in memory, and what is left\n  \
           load MODEL          put a model in memory  [--backend ID]\n  \
           unload [ID]         take one out, or all of them\n  \
           cancel              stop whatever is generating\n  \
           tokenize TEXT       how the model splits a text\n  \
           conversations       ls, show ID, edit ID, rm ID\n  \
           jobs                ls, show ID, watch ID, cancel ID\n  \
           datasets            ls, add FILE, crawl URL, show ID, check ID, search ID Q, rm ID\n  \
           evals               runs, show ID, suites, add FILE, edit ID FILE, rm ID,\n  \
           \u{20}                   run SUITE MODEL..., perplexity DATASET MODEL...\n  \
           bench               runs, show ID, run MODEL...\n  \
           metrics             the machine; requests; log\n  \
           auth                status, login [--key], logout, setup TOKEN, password\n  \
           users               ls, add NAME, edit ID, rm ID\n  \
           sessions | keys     ls, rm ID; keys add NAME\n  \
           api [METHOD PATH [JSON]]\n  \
           \u{20}                   any route, raw; with no arguments, the list of them\n\n\
         -V, --version         print the version and exit\n\n\
         where it runs:\n  \
           --remote URL        send the command to the kvad-serve at URL\n  \
           --local             run it in this process, even with a server running\n  \
           \u{20}                   Otherwise: $KVAD_URL, then [client] url in kvad.toml,\n  \
           \u{20}                   then this machine's service if it answers, then here.\n  \
           \u{20}                   The first line of output says which, and why.\n  \
           --json              print what the server sent, as JSON\n  \
           -y, --yes           answer yes to the question a command would ask\n\n\
         options:\n  \
           --model MODEL       a name trained here, a directory, or a Hub repo id\n  \
           \u{20}                   (default: active, else {DEFAULT_MODEL})\n  \
           --prompt TEXT       prompt for `run`\n  \
           --system TEXT       system prompt for `chat`\n  \
           --max-tokens N      generation budget (default 256)\n  \
           --temperature F     0 is greedy (default 0.7)\n  \
           --top-k N           keep the N best candidates (default 40)\n  \
           --top-p F           nucleus threshold (default 0.95)\n  \
           --seed N            sampling seed (default 7)\n  \
           --quant f32|q8|q4   quantise weights on load (default f32); on a server,\n  \
           \u{20}                   the CPU backend at that precision\n  \
           --backend ID        on a server: which backend to load on, e.g. gpu-q8\n  \
           --greedy            shorthand for --temperature 0\n  \
           --raw               `run` on a server: continue the prompt, no chat template\n  \
           --save              `chat` on a server: keep the conversation there\n  \
           --conversation ID   `chat` on a server: carry on with a kept one\n\n\
         kvad crawl options:\n  \
           --out FILE          where to write it (default: a name from the address)\n  \
           --pages N           most pages to read (default 400)\n  \
           --mb F              most text to collect (default 16)\n  \
           --pause MS          wait between requests (default 250)\n  \
           --drop-rare N       drop characters seen fewer than N times (default 10)\n  \
           --same-host         follow links anywhere on the host, not just under\n  \
           \u{20}                   the address's own directory\n\n\
         Links are followed under the starting address's directory only, because\n\
         `/book/` links into the standard library's documentation on nearly every page\n\
         and that is a hundred times the book. robots.txt is obeyed. Ctrl-C stops it\n\
         and loses what it has read; the web UI's Stop keeps it.\n\n\
         kvad train options:\n  \
           --data FILE         plain text to learn from (required here; on a server,\n  \
           \u{20}                   uploaded as a dataset first)\n  \
           --dataset ID|NAME   on a server: train on a dataset it already has\n  \
           --name NAME         what to call it; one word, no `/`\n  \
           --from MODEL        train an existing model further, instead of a new one\n  \
           --size NAME         model shape: {sizes} (default {default_size})\n  \
           --steps N           training steps (default 2000)\n  \
           --batch N           windows per step (default 16)\n  \
           --lr F              peak learning rate (default 0.003)\n  \
           --warmup N          steps spent climbing to it (default: a tenth of the run)\n  \
           --decay-to F        fraction of --lr left at the last step (default 0.1)\n  \
           --clip F            longest the whole gradient may be; 0 for none (default 1)\n  \
           --eval-every N      steps between checkpoints (default 250)\n  \
           --threads N         replicas to split each batch across (default: every core)\n  \
           --sample N          characters to write at each checkpoint, 0 for none\n\n\
         `--from` has two limits worth knowing. The character vocabulary is fixed at\n\
         first training, so text with a character the model never saw is refused. And\n\
         the optimiser's state is not saved, so a resumed run restarts AdamW's running\n\
         averages: measured at up to 0.06 of training loss over 50 steps, gone by 100.",
        sizes = train::SIZES.iter().map(|s| s.name).collect::<Vec<_>>().join("|"),
        default_size = train::SIZES[train::DEFAULT_SIZE].name,
    );
    std::process::exit(2);
}

/// Commands that take words after them, rather than only flags.
const POSITIONAL: &[&str] = &[
    "search", "pull", "use", "rm", "cache", "crawl", "info", "train", "load", "unload",
    "tokenize", "service", "conversations", "jobs", "datasets", "evals", "bench", "metrics",
    "auth", "users", "sessions", "keys", "api",
];

/// Flags that stand alone, and the short spellings some of them have.
fn switch(a: &mut Args, flag: &str) -> bool {
    match flag {
        "--greedy" => a.temperature = 0.0,
        "--same-host" => a.same_host = true,
        "--local" => a.choice = kvad::client::Choice::Local,
        "--json" => a.json = true,
        "--yes" | "-y" => a.yes = true,
        "--follow" | "-f" => a.follow = true,
        "--help" | "-h" => a.help = true,
        "--save" => a.save = true,
        "--raw" => a.raw = true,
        "--force" => a.force = true,
        "--key" => a.key = true,
        "--password" => a.password = true,
        _ => return false,
    }
    true
}

fn parse_args() -> Args {
    parse_from(std::env::args().skip(1).collect())
}

fn parse_from(argv: Vec<String>) -> Args {
    let mut a = Args::default();
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
    let positional = POSITIONAL.contains(&a.command.as_str());

    while i < argv.len() {
        let flag = argv[i].clone();
        // A word, for a command that takes them. Anywhere on the line, so
        // that `kvad keys rm --json 4` means what it looks like.
        if positional && !flag.starts_with('-') {
            a.words.push(flag);
            i += 1;
            continue;
        }
        if switch(&mut a, &flag) {
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
            "--out" => a.out = Some(value.clone()),
            "--pages" => a.pages = Some(num() as usize),
            "--mb" => a.megabytes = Some(num()),
            "--pause" => a.pause = Some(num() as u64),
            "--drop-rare" => a.drop_rare = Some(num() as usize),
            "--data" => a.data = Some(value.clone()),
            "--from" => a.from = Some(value.clone()),
            "--name" => a.name = Some(value.clone()),
            "--size" => a.size = Some(value.clone()),
            "--steps" => a.steps = Some(num() as usize),
            "--batch" => a.batch = Some(num() as usize),
            "--lr" => a.lr = Some(num() as f32),
            "--eval-every" => a.eval_every = Some(num() as usize),
            "--warmup" => a.warmup = Some(num() as usize),
            "--decay-to" => a.decay_to = Some(num() as f32),
            "--clip" => a.clip = Some(num() as f32),
            "--threads" => a.threads = Some((num() as usize).max(1)),
            "--sample" => a.sample = Some(num() as usize),
            "--quant" => {
                a.quant = Precision::parse(&value).unwrap_or_else(|| {
                    eprintln!("--quant expects f32, q8 or q4, got `{value}`");
                    std::process::exit(2);
                });
                a.quant_given = true;
            }
            "--remote" => a.choice = kvad::client::Choice::Remote(value.clone()),
            "--backend" => a.backend = Some(value.clone()),
            "--host" => a.host = Some(value.clone()),
            "--port" => {
                a.port = Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("--port expects a port number, 1 to 65535, got `{value}`");
                    std::process::exit(2);
                }))
            }
            "--limit" | "--lines" | "-n" => a.limit = Some(num() as usize),
            "--role" => a.role = Some(value.clone()),
            "--title" => a.title = Some(value.clone()),
            "--dataset" => a.dataset = Some(value.clone()),
            "--conversation" => a.conversation = Some(num() as i64),
            "--k" => a.k = Some(num() as usize),
            "--window" => a.window = Some(num() as usize),
            "--rounds" => a.rounds = Some(num() as usize),
            "--tokens" => a.tokens = Some(num() as usize),
            _ => {
                eprintln!("unknown flag {flag}");
                usage();
            }
        }
        i += 2;
    }
    if !a.words.is_empty() {
        a.target = Some(a.words.join(" "));
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

/// Hand over to `kvad-serve`, passing everything through.
///
/// # Why this is an exec and not a function call
///
/// `kvad-serve` depends on this crate — it is built around this engine — so
/// this crate cannot depend on it back. Cargo would refuse the cycle, and it
/// would be right to: the server knows about the engine and the engine must
/// not know about the server.
///
/// So there are two binaries, and this subcommand is the bridge. It looks
/// beside itself first, which is where `cargo build` and any sane
/// installation put them both, and then on `PATH`.
///
/// On Unix it *replaces* this process rather than spawning a child. That
/// matters more than it looks: Ctrl-C reaches the server directly, its exit
/// status is this command's exit status, and there is no parent sitting in
/// the process table forwarding signals it might get wrong.
fn serve(args: Vec<std::ffi::OsString>) -> ! {
    const BIN: &str = "kvad-serve";

    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(BIN)))
        .filter(|path| path.is_file());

    let mut command = std::process::Command::new(match &beside {
        Some(path) => path.as_os_str(),
        // Not beside us: let the OS look on PATH, and if that fails the error
        // below says what to do about it.
        None => std::ffi::OsStr::new(BIN),
    });
    command.args(args);

    #[cfg(unix)]
    let failure = {
        use std::os::unix::process::CommandExt;
        // Only returns if the exec failed.
        command.exec()
    };
    #[cfg(not(unix))]
    let failure = match command.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => e,
    };

    // Printed and exited rather than returned: the whole point of it is
    // that somebody reads it, and it is written with its own layout.
    eprintln!(
        "could not start `{BIN}`: {failure}\n\n\
         It is a separate binary, because the server depends on this engine and so\n\
         cannot be linked into it. Build it once:\n\n    \
         cargo build --release -p kvad-serve\n\n\
         and it will be found beside this one."
    );
    std::process::exit(1);
}

fn main() {
    // Printed with `Display`, not the `Debug` that returning the error from
    // `main` would use: that shows a message as one quoted line with `\n`
    // in it, and the messages here are written to be read.
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Res<()> {
    // Handled before anything else is parsed, because everything after
    // `serve` belongs to the server and this program must not have opinions
    // about it.
    let mut raw = std::env::args_os().skip(1);
    if raw.next().is_some_and(|first| first == "serve") {
        serve(raw.collect());
    }

    // Before parsing too, and for the same reason the installer wants it:
    // "what is on this machine already" has to be answerable by a binary
    // that may be older than whatever is asking.
    if std::env::args().skip(1).any(|a| a == "--version" || a == "-V") {
        println!("kvad {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let args = parse_args();
    if args.help {
        cli::help(&args.command);
    }

    // Local-only commands first. `arch` is a fact about this binary, and
    // `crawl` writes a file here: the server's version of it makes a dataset
    // instead, and is `kvad datasets crawl`.
    match args.command.as_str() {
        "arch" => return arch(),
        "crawl" => return crawl_site(args),
        "service" => return cli::service::run(&args),
        _ => {}
    }

    if !cli::known(&args.command) {
        eprintln!("unknown command `{}`", args.command);
        usage();
    }

    // Everything else can be answered by a server, and some of it only by
    // one. Where it goes is decided once and said first.
    let target = cli::target(&args)?;
    let remote = match target {
        kvad::client::Target::Remote(remote) => remote,
        kvad::client::Target::Local { .. } => {
            return match args.command.as_str() {
                "info" => info(&args),
                "search" => search(args),
                "train" => train_model(args),
                "pull" => pull(args),
                "ls" => list_local(),
                "use" => use_model(args),
                "rm" => remove(args),
                "cache" => cache(args),
                "run" => run(args),
                "chat" => chat(args),
                other if cli::remote_only(other) => cli::no_server(other),
                other => {
                    eprintln!("unknown command `{other}`");
                    usage();
                }
            };
        }
    };
    cli::remote(&remote, &args)
}

/// `kvad info` — a model's config, read without downloading its weights.
fn info(args: &Args) -> Res<()> {
    let model = args.target.clone().or_else(|| args.model.clone()).unwrap_or_else(|| resolve_model(args));
    let files = weights::fetch(&model)?;
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

/// `kvad arch` — what *this* binary can run, which is a build-time question:
/// every architecture is a Cargo feature.
fn arch() -> Res<()> {
    // The middle column is as wide as it has to be: `llama` answers to
    // five `model_type`s now, and a fixed width that one row overflows
    // pushes that row's description out of line with every other.
    let types: Vec<String> =
        kvad::model::arch::registry().iter().map(|a| a.model_types().join(", ")).collect();
    let w = types.iter().map(String::len).chain([17]).max().unwrap_or(17);
    println!("{:<14} {:<w$} WHAT IT IS", "ARCHITECTURE", "CONFIG model_type");
    for (a, t) in kvad::model::arch::registry().iter().zip(&types) {
        println!("{:<14} {t:<w$} {}", a.id(), a.about());
    }
    println!(
        "\n{} of them, chosen at build time; see `arch-*` in crates/llm/Cargo.toml.",
        kvad::model::arch::registry().len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/// `kvad crawl` — the other way to get something to train on.
///
/// Writes two files: the text, and the manifest beside it. It does not make a
/// dataset, because datasets are rows in the server's database and this
/// binary has no business opening it; what it makes is a file, which is what
/// `kvad train --data` wanted all along.
fn crawl_site(args: Args) -> Res<()> {
    let Some(url) = args.target.clone() else {
        eprintln!("usage: kvad crawl URL [--out FILE]");
        eprintln!("       kvad crawl https://doc.rust-lang.org/book/");
        std::process::exit(2);
    };

    let out = PathBuf::from(args.out.clone().unwrap_or_else(|| {
        match crawl::suggested_name(&url).as_str() {
            "" => "corpus.txt".to_string(),
            name => format!("{name}.txt"),
        }
    }));
    // The manifest is named after the file and not after the site, so that
    // two crawls into one directory cannot quietly share one record.
    let manifest = PathBuf::from(format!("{}.crawl.json", out.display()));

    let defaults = crawl::Request::new(&url);
    let request = crawl::Request {
        name: out.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned()),
        same_host: args.same_host,
        max_pages: args.pages.unwrap_or(defaults.max_pages),
        max_bytes: args
            .megabytes
            .map_or(defaults.max_bytes, |mb| (mb * 1024.0 * 1024.0).max(1024.0) as usize),
        delay_ms: args.pause.unwrap_or(defaults.delay_ms),
        drop_rare: args.drop_rare.unwrap_or(defaults.drop_rare),
        ..defaults
    }
    .sane();

    // Before a single request: an address that cannot be fetched is worth
    // saying now rather than after the first connection times out.
    crawl::check(&request)?;
    if out.exists() {
        println!("overwriting {}", out.display());
    }

    let mut last = 0;
    let mut say = |note: crawl::Note| match note {
        crawl::Note::Say(message) => println!("{message}"),
        crawl::Note::Page { title, done, total, .. } => {
            // The total is a guess that grows as links turn up, so it is
            // printed every time rather than remembered from the first page.
            let width = total.to_string().len();
            println!("  {done:>width$}/{total}  {title}");
            last = done;
        }
    };

    let crawled = crawl::run(&request, &mut say, &std::sync::atomic::AtomicBool::new(false))?;
    let manifest_json = serde_json::to_vec_pretty(&crawled.manifest)?;
    std::fs::write(&out, &crawled.text)
        .map_err(|e| format!("could not write {}: {e}", out.display()))?;
    std::fs::write(&manifest, manifest_json)
        .map_err(|e| format!("could not write {}: {e}", manifest.display()))?;

    let m = &crawled.manifest;
    println!();
    println!("wrote {} ({})", out.display(), hub::human_bytes(crawled.text.len() as u64));
    println!("  {} pages, {} characters, {} distinct", m.pages.len(), m.characters, m.distinct);
    if !m.skipped.is_empty() {
        println!("  {} skipped, and why is in the manifest", m.skipped.len());
    }
    if let Some(stopped) = &m.stopped {
        println!("  stopped at {stopped}");
    }
    let mapped: usize = m.mapped.iter().map(|c| c.count).sum();
    if mapped > 0 || !m.dropped.is_empty() {
        println!(
            "  {mapped} characters mapped onto ASCII, {} dropped as too rare",
            m.dropped.len()
        );
    }
    println!("  {}", manifest.display());
    println!();
    // The whole point of the file. `--name` is left blank on purpose: naming
    // a model is a decision, and a suggested one would be taken.
    println!("  kvad train --data {} --name NAME", out.display());
    Ok(())
}

/// `kvad train` — the one command that makes a model instead of fetching one.
fn train_model(args: Args) -> Res<()> {
    let Some(data) = args.data.as_deref() else {
        eprintln!("usage: kvad train --data FILE --name NAME");
        eprintln!("       kvad train --data FILE --from NAME     (train an existing model further)");
        std::process::exit(2);
    };

    let size = match args.size.as_deref() {
        None => &train::SIZES[train::DEFAULT_SIZE],
        Some(name) => train::size(name).unwrap_or_else(|| {
            let names: Vec<_> = train::SIZES.iter().map(|s| s.name).collect();
            eprintln!("--size expects one of {}, got `{name}`", names.join(", "));
            std::process::exit(2);
        }),
    };

    let d = nervus::text::Training::default();
    let training = nervus::text::Training {
        steps: args.steps.unwrap_or(d.steps),
        batch: args.batch.unwrap_or(d.batch),
        lr: args.lr.unwrap_or(d.lr),
        eval_every: args.eval_every.unwrap_or(d.eval_every),
        threads: args.threads.unwrap_or(d.threads),
        // `--warmup` unset means a tenth of the run; see `Training::schedule`.
        warmup: args.warmup.or(d.warmup),
        decay_to: args.decay_to.unwrap_or(d.decay_to),
        // Zero is not a clip anyone would want, so it means "none".
        clip: args.clip.map_or(d.clip, |c| Some(c).filter(|&c| c > 0.0)),
        ..d
    };

    let opts = train::Options {
        data: data.into(),
        from: args.from.clone(),
        name: args.name.clone(),
        size,
        training,
        seed: args.seed,
        sample: args.sample.unwrap_or(160),
        temperature: args.temperature,
        // A command line already has a way to stop a run: Ctrl-C.
        cancel: None,
    };

    // Say where it is going before it starts, because it is about to take a
    // while and overwriting a model is not undoable.
    match (&opts.name, &opts.from) {
        (Some(name), _) if !weights::is_model_name(name) => {
            eprintln!("`{name}` is not a model name: it has to be one word, with no `/` in it");
            std::process::exit(2);
        }
        (Some(name), _) => {
            let dir = train::would_write(name);
            match train::holds_a_model(&dir) {
                true => println!("training `{name}`, replacing the model already in {}", dir.display()),
                false => println!("training `{name}` into {}", dir.display()),
            }
        }
        (None, Some(from)) => {
            println!("training `{from}` further, in place — pass --name to keep the old one as well");
        }
        (None, None) => {
            eprintln!("a new model needs a name:  kvad train --data {data} --name NAME");
            std::process::exit(2);
        }
    }
    // Only for a new model. With `--from` the shape comes from the
    // checkpoint, so say plainly that `--size` is being ignored rather than
    // let someone believe they resized a model by asking.
    match (opts.from.is_some(), args.size.is_some()) {
        (false, _) => println!("size {}: {}", size.name, size.shape()),
        (true, true) => eprintln!("ignoring --size: a model's shape is fixed when it is first trained"),
        (true, false) => {}
    }

    let summary = train::run(&opts, &mut |line| println!("{line}"))?;

    // Step 0 is "nothing was written": a continuation where no checkpoint
    // beat the model it started from. Reporting a kept step there would name
    // a step whose model is not on disk and never was.
    if !summary.improved() {
        let reached = match summary.reached.is_finite() {
            true => format!("the best checkpoint here reached {:.3}", summary.reached),
            false => "no checkpoint was measured".to_string(),
        };
        println!(
            "\nnothing beat the model this run started from: it scores {:.3}, and {reached}.",
            summary.best_val
        );
        let from = opts.from.as_deref().unwrap_or(&summary.handle);
        match &opts.name {
            Some(name) => println!("Nothing was written: `{name}` was not created and `{from}` is untouched."),
            None => println!("Nothing was written, and `{from}` is untouched."),
        }
        println!("\nrun it with:  kvad run --model {from} --prompt \"...\"");
        return Ok(());
    }

    println!(
        "\nkept step {} of {}: validation loss {:.3} (the last step measured {:.3})",
        summary.best_step, opts.training.steps, summary.best_val, summary.last_val
    );
    println!(
        "{} parameters, {} in {}",
        summary.params,
        hub::human_bytes(summary.dir.join("model.safetensors").metadata().map(|m| m.len()).unwrap_or(0)),
        summary.dir.display()
    );
    println!("\nrun it with:  kvad run --model {} --prompt \"...\"", summary.handle);
    Ok(())
}

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
        "{:<40} {:>10} {:>9}  {:<11} {}",
        "MODEL", "DOWNLOADS", "SIZE", "ARCH", "STATUS"
    );
    for m in &results {
        // The size is the download; the fit is what it costs *here*, which is
        // a different number because the weights are quantised on load.
        let size = m
            .download_bytes
            .map(hub::human_bytes)
            .unwrap_or_else(|| "-".into());
        let status = match m.blocker() {
            Some(reason) => reason,
            None if local.contains(&m.id) => "downloaded".into(),
            None if m.looks_instruct => format!("chat · {}", m.fit()),
            None => format!("completion · {}", m.fit()),
        };
        // Names come from the Hub; print them, never act on them.
        println!(
            "{:<40} {:>10} {:>9}  {:<11} {}",
            truncate(&m.id, 40),
            m.downloads,
            size,
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

/// `kvad ls` — everything runnable on this machine, in two sections.
///
/// Downloaded and trained models are listed apart because they are different
/// kinds of thing: one can be fetched again, and the other is the only copy
/// there is. Both are named the way `--model` wants them.
fn list_local() -> Res<()> {
    let downloaded = hub::local_models();
    let trained = hub::trained_models();
    if downloaded.is_empty() && trained.is_empty() {
        println!("nothing here yet. try:  kvad search smollm");
        println!("               or:  kvad train --data some.txt --name mine");
        return Ok(());
    }

    let active = State::active();
    let mut total = 0;
    let mut row = |m: &hub::LocalModel, note: &str| {
        total += m.bytes;
        let marker = if active.as_deref() == Some(m.id.as_str()) { " *" } else { "" };
        println!(
            "{:<46} {:<11} {:>9}{}{}",
            truncate(&m.id, 46),
            m.arch.map(|a| a.to_string()).unwrap_or_else(|| "?".into()),
            hub::human_bytes(m.bytes),
            marker,
            note
        );
    };

    if !downloaded.is_empty() {
        println!("{:<46} {:<11} {:>9}", "DOWNLOADED", "ARCH", "SIZE");
        for m in &downloaded {
            row(m, if m.complete { "" } else { "  (config only)" });
        }
    }
    if !trained.is_empty() {
        if !downloaded.is_empty() {
            println!();
        }
        println!("{:<46} {:<11} {:>9}", "TRAINED HERE", "ARCH", "SIZE");
        for m in &trained {
            row(m, if m.complete { "" } else { "  (unfinished — no weights)" });
        }
    }

    println!("\n{} models, {}", downloaded.len() + trained.len(), hub::human_bytes(total));
    if let Some(a) = active {
        println!("* active: {a}");
    } else {
        println!("no active model set (using {DEFAULT_MODEL})");
    }
    println!("cache: {}", hub::cache_dir().display());
    if !trained.is_empty() {
        println!("trained: {}", weights::models_dir().display());
    }
    Ok(())
}

fn use_model(args: Args) -> Res<()> {
    let Some(repo) = args.target else {
        eprintln!("usage: kvad use REPO");
        std::process::exit(2);
    };
    // A directory is remembered by its absolute path: the active model has to
    // mean the same thing from whichever directory `kvad` is next run in.
    let repo = weights::model_id(&repo);
    if !weights::is_local(&repo) && hub::find_local(&repo).is_none() {
        match (weights::looks_like_path(&repo), repo.contains('/')) {
            (true, _) => eprintln!("`{repo}` looks like a path, and there is no such directory"),
            (_, true) => eprintln!("`{repo}` is not downloaded. Run:  kvad pull {repo}"),
            // No slash, so not a repo id: they meant a model trained here.
            (_, false) => eprintln!("`{repo}` is not a model trained here. Run `kvad ls` to see what is."),
        }
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
    // Trained first: `kvad ls` shows both, so `kvad rm` has to know both.
    let trained = hub::find_trained(&repo);
    let Some(local) = trained.clone().or_else(|| hub::find_local(&repo)) else {
        eprintln!("`{repo}` is not a model on this machine. Run `kvad ls` to see what is.");
        std::process::exit(1);
    };

    // Deleting is irreversible and the download may have been slow, so say
    // exactly what will go and require a typed yes.
    println!("about to delete:");
    println!("  {}", local.path.display());
    println!("  {} ({})", local.id, hub::human_bytes(local.bytes));
    match trained.is_some() {
        // There is nowhere to fetch a trained model back from. Say so
        // plainly: this is the one deletion in `kvad` that cannot be undone
        // by waiting for a download.
        true => print!("\nit was trained here and is not on the Hub; deleting it is final. delete? [y/N] "),
        false => print!("\nre-download would be needed to use it again. delete? [y/N] "),
    }
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
            // A directory is filed under its absolute path, however it was typed.
            println!("deleted {} file(s) for {repo}", qcache::forget(&weights::model_id(repo)));
            return Ok(());
        }
        None => {}
    }

    let entries = qcache::entries();
    if entries.is_empty() {
        println!("nothing pre-quantised yet.");
        println!("the first `kvad run --quant q8` writes a file here; later runs map it.");
    } else {
        println!("{:<44} {:<8} {:>10}", "MODEL", "QUANT", "SIZE");
        let mut total = 0;
        for (_, repo, precision, bytes) in &entries {
            let size = hub::human_bytes(*bytes);
            println!("{:<44} {:<8} {:>10}", truncate(repo, 43), precision, size);
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
        let ids = llm.encode_chat(&messages, &[])?;

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
