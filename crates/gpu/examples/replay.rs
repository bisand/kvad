//! Is a prompt answered from its cache the prompt answered fresh?
//!
//! The second time a server is sent a conversation it does not read the
//! prompt again: it keeps the cache, drops the last token and runs that
//! one alone, for its logits. The first time, that token was the last row
//! of a pass over the whole prompt. Same token, same position, same
//! weights, and another kernel: one row where there were forty.
//!
//! At temperature 0 the two replies were seen to part after a few hundred
//! characters. This asks whether that is rounding or a fault. For each
//! engine it reads a prompt fresh and writes greedily, then rewinds to the
//! prompt less a token and does it again, and says how far apart the two
//! sets of logits are, where the replies part, and how near a tie the
//! token they part at was.
//!
//! It is rounding, each road gives the same answer every time, and the
//! cure is to take one road: the last token of a prompt read by itself
//! whether the rest came from the cache or from a pass. `Llm::prefill`
//! does that now. This still drives a session by hand both ways, and
//! times what the cure costs.
//!
//!     KVAD_QUANT_CACHE=/tmp/q cargo run --release -p kvad-gpu --example replay -- \
//!         HuggingFaceTB/SmolLM2-135M-Instruct

use kvad::model::Session;
use kvad::quant::Precision;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const WRITE: usize = 400;

/// The token greedy decoding takes, and how far ahead of the next it was.
fn top(logits: &[f32]) -> (u32, f32) {
    let (mut best, mut next) = ((0, f32::MIN), f32::MIN);
    for (i, &l) in logits.iter().enumerate() {
        if l > best.1 {
            (next, best) = (best.1, (i, l));
        } else if l > next {
            next = l;
        }
    }
    (best.0 as u32, best.1 - next)
}

/// Write greedily from `logits`: the tokens, and each one's lead.
fn write(session: &mut dyn Session, mut logits: Vec<f32>) -> Res<Vec<(u32, f32)>> {
    let mut out = Vec::with_capacity(WRITE);
    for _ in 0..WRITE {
        let (token, lead) = top(&logits);
        out.push((token, lead));
        logits = session.forward(&[token])?;
    }
    Ok(out)
}

fn compare(name: &str, session: &mut dyn Session, prompt: &[u32]) -> Res<()> {
    let differ = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let parts = |a: &[(u32, f32)], b: &[(u32, f32)]| a.iter().zip(b).position(|(a, b)| a.0 != b.0);

    session.truncate(0)?;
    let fresh = session.forward(prompt)?;
    let first = write(session, fresh.clone())?;

    // As a server does with a conversation it has seen.
    session.truncate(prompt.len() - 1)?;
    let cached = session.forward(&prompt[prompt.len() - 1..])?;
    let second = write(session, cached.clone())?;

    // Each again, to know what the same road gives twice.
    session.truncate(prompt.len() - 1)?;
    let cached_again = session.forward(&prompt[prompt.len() - 1..])?;
    let third = write(session, cached_again.clone())?;
    session.truncate(0)?;
    let fresh_again = session.forward(prompt)?;
    let fourth = write(session, fresh_again.clone())?;

    // And the cure: the last token of a fresh prompt read by itself, as
    // it is from the cache, after the rest in a pass.
    session.truncate(0)?;
    session.forward(&prompt[..prompt.len() - 1])?;
    let apart = session.forward(&prompt[prompt.len() - 1..])?;
    let fifth = write(session, apart.clone())?;
    session.truncate(prompt.len() - 1)?;
    let apart_cached = session.forward(&prompt[prompt.len() - 1..])?;
    let sixth = write(session, apart_cached.clone())?;

    // What the cure costs: a long prompt in one call, against all but its
    // last token and then that. Interleaved, the median of seven. At two
    // lengths, because a GPU reads a prompt in tiles of 32 and a length
    // that is a whole number of them is the one the cure spoils.
    let mut cost = Vec::new();
    for length in [500, 512] {
        let long: Vec<u32> = prompt.iter().cycle().take(length).copied().collect();
        let (mut whole, mut split) = (Vec::new(), Vec::new());
        for _ in 0..7 {
            session.truncate(0)?;
            let start = std::time::Instant::now();
            session.forward(&long)?;
            whole.push(start.elapsed().as_secs_f64() * 1e3);
            session.truncate(0)?;
            let start = std::time::Instant::now();
            session.forward(&long[..long.len() - 1])?;
            session.forward(&long[long.len() - 1..])?;
            split.push(start.elapsed().as_secs_f64() * 1e3);
        }
        whole.sort_by(f64::total_cmp);
        split.sort_by(f64::total_cmp);
        cost.push((length, whole[3], split[3]));
    }

    let scale = fresh.iter().fold(0f32, |m, l| m.max(l.abs()));
    println!("## {name}\n");
    println!("- the prompt's last logits, fresh against cached: differ by at most {:.2e}, of logits up to {scale:.1}", differ(&fresh, &cached));
    println!("- the same road twice: fresh {:.2e}, cached {:.2e}", differ(&fresh, &fresh_again), differ(&cached, &cached_again));
    println!("- replies by the same road: fresh twice {}, cached twice {}", same(parts(&first, &fourth)), same(parts(&second, &third)));
    match parts(&first, &second) {
        None => println!("- fresh and cached write the same {WRITE} tokens"),
        Some(at) => {
            println!("- fresh and cached part at token {at} of {WRITE}: the fresh one took its token by {:.4}, the cached one by {:.4}", first[at].1, second[at].1);
            let mut leads: Vec<f32> = first.iter().map(|t| t.1).collect();
            leads.sort_by(f32::total_cmp);
            println!("- a token's lead over the next in this reply: median {:.2}, and {} of {WRITE} under 0.01", leads[leads.len() / 2], leads.iter().filter(|l| **l < 0.01).count());
        }
    }
    println!(
        "- with the last token read by itself: fresh against cached differ by {:.2e}, and the replies {}",
        differ(&apart, &apart_cached),
        same(parts(&fifth, &sixth))
    );
    for (length, whole, split) in cost {
        println!("- a prompt of {length} tokens: {whole:.1} ms in one call, {split:.1} ms with its last token apart");
    }
    println!();
    Ok(())
}

fn same(parts: Option<usize>) -> String {
    match parts {
        None => "the same".into(),
        Some(at) => format!("part at token {at}"),
    }
}

fn main() -> Res<()> {
    let repo = std::env::args().nth(1).unwrap_or_else(|| "HuggingFaceTB/SmolLM2-135M-Instruct".into());
    let files = kvad::weights::fetch_with(&repo, &mut |_| {})?;
    let spec = kvad::model::Spec::from_json(&files.config)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer).map_err(|e| e.to_string())?;
    let text = "<|im_start|>user\nWrite a long story about the sea, at least six hundred words.<|im_end|>\n<|im_start|>assistant\n";
    let prompt: Vec<u32> = tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    println!("{repo}, a prompt of {} tokens, {WRITE} written greedily after it\n", prompt.len());

    let quiet = &mut |_: &str| {};
    let dtype = kvad_gpu::model::parse_dtype("bf16").ok_or("dtype")?;
    for (name, quant) in [("GPU, q8", "q8"), ("GPU, bf16", "none")] {
        let device = kvad_gpu::model::pick_device(None)?;
        let quant = kvad_gpu::model::parse_quant(quant).ok_or("quant")?;
        let mut session = kvad_gpu::model::session(&repo, &files.weights, &spec, dtype, quant, device, quiet)?;
        compare(name, session.as_mut(), &prompt)?;
    }
    // The same model a stage at a time, as it is across machines: the
    // embedding and the head on the CPU, the layers on the GPU in two runs.
    {
        use kvad::cluster::stage::Stage;
        use kvad::model::Shard;
        let device = kvad_gpu::model::pick_device(None)?;
        let quant = kvad_gpu::model::parse_quant("q8").ok_or("quant")?;
        let ends = kvad::qcache::load_shard(&repo, &files, &spec, Precision::Q8, &Shard { layers: 0..0, ends: true }, quiet)?;
        let half = spec.n_layer / 2;
        let mut stages: Vec<Box<dyn Stage>> = Vec::new();
        for layers in [0..half, half..spec.n_layer] {
            stages.push(Box::new(kvad_gpu::model::stage::GpuStage::load(&repo, &files.weights, spec.clone(), dtype, quant, device.clone(), layers, quiet)?));
        }
        // The stages by themselves first: the prompt's rows in one pass,
        // and its last row again against the cache, layer by layer of depth.
        {
            use kvad::model::Transformer;
            let e = spec.n_embd;
            let mut rows = ends.embed(&prompt, 0);
            let m = prompt.len();
            println!("## A stage alone: the last row of a pass of {m}, against that row run alone on the cache\n");
            for stage in stages.iter_mut() {
                let fresh = stage.run(7, 0, m, rows.clone()).map_err(|e| e.to_string())?;
                stage.truncate(7, m - 1).map_err(|e| e.to_string())?;
                let cached = stage.run(7, m - 1, 1, rows[(m - 1) * e..].to_vec()).map_err(|e| e.to_string())?;
                let last = &fresh[(m - 1) * e..];
                let worst = last.iter().zip(&cached).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let scale = last.iter().fold(0f32, |s, x| s.max(x.abs()));
                println!("- {}: differ by at most {worst:.2e}, of rows up to {scale:.1}", stage.shard());
                stage.truncate(7, 0).map_err(|e| e.to_string())?;
                rows = fresh;
            }
            println!();
        }
        let mut session = kvad::cluster::session::PipelineSession::new(std::sync::Arc::from(ends), stages, Precision::Q8).map_err(|e| e.to_string())?;
        compare("GPU, q8, in two stages with the ends on the CPU", &mut session, &prompt)?;
    }
    for precision in [Precision::Q8, Precision::F32] {
        let model = kvad::qcache::load(&repo, &files, &spec, precision, quiet)?;
        let mut session = kvad::model::CpuSession::new(model, precision);
        compare(&format!("CPU, {precision}"), &mut session, &prompt)?;
    }
    Ok(())
}
