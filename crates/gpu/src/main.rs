//! The same models, on the GPU.
//!
//!     kvad-gpu run  [--model REPO] [--prompt TEXT] [--device metal] [--dtype bf16]
//!     kvad-gpu chat [--model REPO] [--system TEXT]
//!     kvad-gpu tune [MODEL] --data DIR --name NAME      (a LoRA for SDXL)
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
        "usage: kvad-gpu <run|chat> [options]\n       kvad-gpu tune --help\n\n\
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
    // What the quantised-weight cache files this model under.
    let id = kvad::weights::model_id(&repo);
    let llm =
        Llm::load_custom(&repo, &mut |m| eprintln!("  {m}"), &watch, &mut |files, spec, progress| {
            model::session(&id, &files.weights, spec, dtype, quant, device.clone(), progress)
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
    // `tune` has options of its own, and reads them itself.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("tune") {
        if let Err(e) = tune(&argv[1..]) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
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

const TUNE_USAGE: &str = "usage: kvad-gpu tune [MODEL] --data DIR --name NAME [options]

Train a LoRA for SDXL on a folder of pictures, each with its caption in a
.txt of the same name beside it. MODEL is SDXL's repo unless given.

options:
  --data DIR          the folder of pictures and captions
  --name NAME         what to call the LoRA; it is written to the data
                      directory's loras/NAME.safetensors
  --out FILE          or the file to write it to
  --from FILE         go on from a LoRA this wrote
  --caption TEXT      the caption of every picture that has none
  --size N            pixels a side, a multiple of 64 (default 1024)
  --rank N            (default 16)
  --alpha F           the LoRA is scaled by alpha / rank (default: the rank)
  --steps N           (default 1000)
  --lr F              (default 1e-4)
  --eval-every N      steps between validation measurements (default 100)
  --holdout N         pictures kept out of training to measure on
                      (default: a tenth, from one to four; none of under 5)
  --sample TEXT       a prompt to draw before the first step and at every
                      measurement, from the same seed each time; may be
                      given more than once
  --sample-size N     pixels a side of a sample (default: --size)
  --sample-steps N    its denoising steps (default 20)
  --samples DIR       where they are written (default: the LoRA's file's
                      name, ending .samples)
  --seed N            (default 1337)
  --ffmpeg FILE       the ffmpeg that decodes the pictures (default: found)
  --cap GB            end the run if its memory passes this (default: three
                      quarters of the machine's)";

/// `kvad-gpu tune`: [`kvad_gpu::image::tune::run`] from a command line.
fn tune(argv: &[String]) -> Res<()> {
    use kvad_gpu::image::tune;
    if argv.is_empty() || argv.iter().any(|a| matches!(a.as_str(), "-h" | "--help")) {
        eprintln!("{TUNE_USAGE}");
        std::process::exit(2);
    }
    let (mut model, mut flags, mut samples) = (None, std::collections::HashMap::new(), Vec::new());
    let mut i = 0;
    while i < argv.len() {
        match argv[i].strip_prefix("--") {
            Some(flag) => {
                let value = argv.get(i + 1).ok_or_else(|| format!("--{flag} needs a value"))?;
                // The one flag that may be given more than once.
                match flag {
                    "sample" => samples.push(value.clone()),
                    _ => drop(flags.insert(flag.to_string(), value.clone())),
                }
                i += 2;
            }
            None if model.is_none() => {
                model = Some(argv[i].clone());
                i += 1;
            }
            None => return Err(format!("`{}` is one model too many\n\n{TUNE_USAGE}", argv[i]).into()),
        }
    }
    let mut take = |flag: &str| flags.remove(flag);
    fn number<T: std::str::FromStr>(flag: &str, value: Option<String>) -> Res<Option<T>> {
        value.map(|v| v.parse::<T>().map_err(|_| format!("--{flag} expects a number, got `{v}`").into())).transpose()
    }
    let data = std::path::PathBuf::from(take("data").ok_or_else(|| format!("which pictures? --data DIR\n\n{TUNE_USAGE}"))?);
    let out = match (take("out"), take("name")) {
        (Some(file), _) => std::path::PathBuf::from(file),
        (None, Some(name)) => {
            if !kvad::weights::is_model_name(&name) {
                return Err(format!("`{name}` is not a name: it has to be one word, with no `/` in it").into());
            }
            kvad::weights::data_dir().join("loras").join(format!("{name}.safetensors"))
        }
        (None, None) => return Err(format!("a LoRA needs a name: --name NAME\n\n{TUNE_USAGE}").into()),
    };
    let ffmpeg = match take("ffmpeg") {
        Some(file) => std::path::PathBuf::from(file),
        None => {
            let path = std::env::var_os("PATH").unwrap_or_default();
            kvad::video::ffmpeg_on(&std::env::split_paths(&path).collect::<Vec<_>>()).ok_or("the pictures are decoded by ffmpeg, and none was found: install it, or name one with --ffmpeg FILE")?
        }
    };
    let mut opts = tune::Options::new(model.as_deref().unwrap_or(kvad_gpu::image::sdxl::REPO), &data, &out, &ffmpeg);
    opts.from = take("from").map(std::path::PathBuf::from);
    opts.caption = take("caption");
    opts.alpha = number("alpha", take("alpha"))?;
    opts.holdout = number("holdout", take("holdout"))?;
    opts.samples = samples;
    opts.sample_dir = take("samples").map(std::path::PathBuf::from);
    opts.sample_size = number("sample-size", take("sample-size"))?;
    for (flag, into) in [("size", &mut opts.size), ("rank", &mut opts.rank), ("steps", &mut opts.steps), ("eval-every", &mut opts.eval_every), ("sample-steps", &mut opts.sample_steps)] {
        if let Some(n) = number(flag, take(flag))? {
            *into = n;
        }
    }
    if let Some(lr) = number("lr", take("lr"))? {
        opts.lr = lr;
    }
    if let Some(seed) = number("seed", take("seed"))? {
        opts.seed = seed;
    }
    let cap = number::<f64>("cap", take("cap"))?.or_else(|| kvad::machine::total_memory().map(|b| b as f64 / 1e9 * 0.75));
    if let Some(flag) = flags.keys().next() {
        return Err(format!("unknown flag --{flag}\n\n{TUNE_USAGE}").into());
    }
    // A backward pass that does not fit is not refused by the system: it
    // takes the machine down. The run ends itself first.
    if let Some(cap) = cap {
        kvad_gpu::cap::at(cap);
    }
    let device = model::pick_device(None)?;
    if !device.is_metal() && !device.is_cuda() {
        return Err("an image model is trained on the GPU and nowhere else, and this machine has none that candle can use".into());
    }

    // Ctrl-C ends the run after the step it is in, with its last step
    // written beside its best. A second one ends the process.
    static STOP: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicBool>> = std::sync::OnceLock::new();
    extern "C" fn stop(_: libc::c_int) {
        if let Some(flag) = STOP.get() {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: `signal` is async-signal-safe, and puts back the default.
        unsafe { libc::signal(libc::SIGINT, libc::SIG_DFL) };
    }
    opts.cancel = Some(STOP.get_or_init(Default::default).clone());
    // SAFETY: the handler stores to an atomic and calls `signal`, both of
    // which a signal handler may.
    unsafe { libc::signal(libc::SIGINT, stop as extern "C" fn(libc::c_int) as libc::sighandler_t) };

    let s = tune::run(&opts, &device, &mut |line| eprintln!("  {line}"), &mut |_| {})?;
    eprintln!();
    if s.best_step == 0 {
        eprintln!("no step was taken, and nothing was written");
        return Ok(());
    }
    eprintln!(
        "{} steps on {} pictures in {:.0} min{}",
        s.steps,
        s.pictures,
        s.elapsed_secs / 60.0,
        if s.stopped { ", stopped early" } else { "" }
    );
    eprintln!("validation loss {:.4} at step {}, from the model's own {:.4}", s.best_val, s.best_step, s.base_val);
    if s.best_val >= s.base_val {
        eprintln!("which is no better than the model without the LoRA: on pictures it did not train on, it learned nothing that carries over");
    }
    eprintln!("the LoRA, {:.1} M numbers on {} layers: {}", s.trained as f64 / 1e6, s.layers, s.out.display());
    if let Some(last) = &s.last {
        eprintln!("the last step's, validation loss {:.4}: {}", s.last_val, last.display());
    }
    if let Some(dir) = &s.samples {
        eprintln!("the samples, N-STEP.png for prompt N: {}", dir.display());
    }
    eprintln!("\ndraw with it:  kvad images make \"...\" --model {} --lora {}", opts.repo, s.out.display());
    Ok(())
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
