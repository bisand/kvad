//! Generate text with GPT-2.
//!
//!     cargo run --release -p gpt2 -- --prompt "The city of Bergen is"
//!
//! Flags: --model REPO --prompt TEXT --max-tokens N --temperature F
//!        --top-k N --top-p F --seed N --greedy

use gpt2::model::{Cache, Config, Model};
use gpt2::sampler::Sampler;
use gpt2::weights;
use std::io::Write;
use std::time::Instant;
use tokenizers::Tokenizer;

struct Args {
    model: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            model: "openai-community/gpt2".into(),
            prompt: "The first time I saw the sea,".into(),
            max_tokens: 64,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: 7,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        if flag == "--greedy" {
            a.temperature = 0.0;
            i += 1;
            continue;
        }
        let Some(value) = argv.get(i + 1).cloned() else {
            eprintln!("missing value for {flag}");
            std::process::exit(2);
        };
        let num = || -> f64 {
            value.parse().unwrap_or_else(|_| {
                eprintln!("{flag} expects a number, got `{value}`");
                std::process::exit(2);
            })
        };
        match flag.as_str() {
            "--model" => a.model = value.clone(),
            "--prompt" => a.prompt = value.clone(),
            "--max-tokens" => a.max_tokens = num() as usize,
            "--temperature" => a.temperature = num() as f32,
            "--top-k" => a.top_k = num() as usize,
            "--top-p" => a.top_p = num() as f32,
            "--seed" => a.seed = num() as u64,
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    a
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    eprintln!("model: {}", args.model);
    let files = weights::fetch(&args.model)?;

    let config = Config::from_json(&files.config)?;
    eprintln!(
        "  {} layers, {} heads, {} embedding dim, {} context, {} vocab",
        config.n_layer, config.n_head, config.n_embd, config.n_ctx, config.vocab_size
    );

    let t0 = Instant::now();
    let model = Model::load(&files.weights, config.clone())?;
    eprintln!(
        "  {:.1}M parameters loaded in {:.1}s  (KV cache at full context: {:.0} MB)",
        model.param_count() as f64 / 1e6,
        t0.elapsed().as_secs_f32(),
        Cache::max_bytes(&config) as f64 / 1e6
    );

    let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|e| e.to_string())?;
    let encoding = tokenizer.encode(args.prompt.as_str(), false).map_err(|e| e.to_string())?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    eprintln!("  prompt is {} tokens\n", ids.len());

    let mut cache = Cache::new(&config);
    let mut sampler = Sampler::new(args.temperature, args.top_k, args.top_p, args.seed);

    // Feed the prompt through to populate the cache. We only keep the logits
    // from the final prompt token -- the earlier ones predict tokens we
    // already have.
    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let prefill = Instant::now();
    let mut logits = Vec::new();
    for &id in &ids {
        logits = model.forward(id, &mut cache);
    }
    let prefill_time = prefill.elapsed();

    // Then generate, one token at a time, feeding each choice back in.
    let decode = Instant::now();
    let mut generated = 0usize;
    let budget = args.max_tokens.min(config.n_ctx.saturating_sub(ids.len()));
    for _ in 0..budget {
        let next = sampler.sample(&logits);
        ids.push(next);
        generated += 1;

        // Decode the whole sequence and print only what is new. Byte-level BPE
        // tokens can be fragments of a UTF-8 character, so decoding them
        // individually would emit garbage.
        let text = tokenizer.decode(&ids, true).map_err(|e| e.to_string())?;
        let shown = tokenizer
            .decode(&ids[..ids.len() - 1], true)
            .map_err(|e| e.to_string())?;
        print!("{}", &text[shown.len()..]);
        std::io::stdout().flush()?;

        // 50256 is <|endoftext|>, the only special token GPT-2 has.
        if next == 50256 {
            break;
        }
        logits = model.forward(next, &mut cache);
    }

    let dt = decode.elapsed().as_secs_f32();
    eprintln!(
        "\n\n[prefill {} tokens in {:.2}s, generated {} tokens in {:.2}s = {:.1} tok/s]",
        ids.len() - generated,
        prefill_time.as_secs_f32(),
        generated,
        dt,
        generated as f32 / dt.max(1e-6)
    );
    Ok(())
}
