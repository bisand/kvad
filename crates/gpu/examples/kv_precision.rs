//! What keeping a quantised model's attention cache in f16 does to what it
//! says.
//!
//! `common::kv_store` keeps a quantised model's cache in f16, not the f32 it
//! computes in, which halves what attention reads at long context. That
//! changes numbers, so this measures what it changes. It loads the same model
//! twice, one with each cache (`KVAD_GPU_KV_F32=1` for the f32 one), and
//! feeds both the same text a token at a time, after prompts of several
//! lengths. At every step it compares:
//! - **perplexity** on the text's actual next token, for each cache;
//! - **top-1:** how often the two pick the same most likely token;
//! - **logits:** the largest difference anywhere in the vocabulary;
//! - **KL divergence** of the f16 cache's distribution from the f32 one's.
//!
//! Then both generate greedily from the same prompt, and it says how long
//! they stay word for word the same.
//!
//!     cargo run --release -p kvad-gpu --example kv_precision
//!
//! Needs Qwen/Qwen2.5-1.5B-Instruct in the Hub cache (`kvad pull`).
use candle_core::{quantized::GgmlDType, DType, Device};
use kvad::runtime::Llm;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const REPO: &str = "Qwen/Qwen2.5-1.5B-Instruct";
/// Tokens fed one at a time after each prompt.
const STEPS: usize = 256;

fn load(f32_cache: bool) -> Res<Llm> {
    // Read by `kv_store` as each model loads, so one process can hold both.
    match f32_cache {
        true => std::env::set_var("KVAD_GPU_KV_F32", "1"),
        false => std::env::remove_var("KVAD_GPU_KV_F32"),
    }
    let id = kvad::weights::model_id(REPO);
    let dev = Device::new_metal(0)?;
    Llm::load_custom(REPO, &mut |_| {}, &kvad::weights::Watcher::none(), &mut |files, spec, progress| {
        kvad_gpu::model::session(&id, &files.weights, spec, DType::BF16, Some(GgmlDType::Q8_0), dev.clone(), progress)
    })
}

/// `log_softmax(logits)` at every index, in f64.
fn log_probs(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&x| x as f64 - lse).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

fn main() -> Res<()> {
    let mut exact = load(true)?;
    let mut half = load(false)?;
    println!("{REPO}: {} against {}\n", half.session.label(), exact.session.label());

    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let mut ids = Vec::new();
    while ids.len() < 8192 + STEPS + 1 {
        ids.extend(exact.encode(&text)?);
    }

    println!("teacher-forced, {STEPS} tokens after each prompt:\n");
    for prompt in [128usize, 2048, 8192] {
        exact.session.truncate(0)?;
        half.session.truncate(0)?;
        let (mut a, mut b) = (exact.session.forward(&ids[..prompt])?, half.session.forward(&ids[..prompt])?);
        let (mut nll_a, mut nll_b, mut same, mut kl) = (0.0, 0.0, 0usize, 0.0);
        let mut worst = Vec::new();
        for t in 0..STEPS {
            let next = ids[prompt + t] as usize;
            let (la, lb) = (log_probs(&a), log_probs(&b));
            nll_a -= la[next];
            nll_b -= lb[next];
            same += usize::from(argmax(&a) == argmax(&b));
            kl += la.iter().zip(&lb).map(|(p, q)| p.exp() * (p - q)).sum::<f64>();
            worst.push(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
            a = exact.session.forward(&[next as u32])?;
            b = half.session.forward(&[next as u32])?;
        }
        worst.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let n = STEPS as f64;
        println!(
            "  prompt {prompt:5}: perplexity f32 {:.4}  f16 {:.4}   top-1 same {}/{STEPS}   \
             KL {:.2e}   max |Δlogit| median {:.4}, worst {:.4}",
            (nll_a / n).exp(),
            (nll_b / n).exp(),
            same,
            kl / n,
            worst[STEPS / 2],
            worst[STEPS - 1]
        );
    }

    println!("\ngreedy, 256 tokens after a 2048-token prompt:\n");
    let greedy = |llm: &mut Llm| -> Res<Vec<u32>> {
        llm.session.truncate(0)?;
        let mut logits = llm.session.forward(&ids[..2048])?;
        let mut out = Vec::new();
        for _ in 0..256 {
            let next = argmax(&logits) as u32;
            out.push(next);
            logits = llm.session.forward(&[next])?;
        }
        Ok(out)
    };
    let (ga, gb) = (greedy(&mut exact)?, greedy(&mut half)?);
    match ga.iter().zip(&gb).position(|(x, y)| x != y) {
        None => println!("  the same 256 tokens"),
        Some(i) => {
            println!("  the same for {i} tokens, then:");
            println!("    f32: {:?}", exact.decode(&ga[i.saturating_sub(8)..(i + 12).min(256)])?);
            println!("    f16: {:?}", exact.decode(&gb[i.saturating_sub(8)..(i + 12).min(256)])?);
        }
    }
    Ok(())
}
