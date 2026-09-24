//! Where the rest of a decode token goes: everything `decode_cost` found
//! the matmuls do not account for.
//!
//! `decode_cost` measured one token's matrix-vector products at 77–90% of
//! what this machine's memory delivers, and about 1.7 ms per token beside
//! them, at every precision. This splits that 1.7 ms in two ways.
//!
//! **Around the model.** The real model and the runtime's own decode loop,
//! with each part of a step timed where `Llm::generate` runs it:
//! - `forward`: the GPU's step, ending in a sync and the logits copied back
//!   to the CPU;
//! - `sample`: picking the next token from 151,936 logits, on the CPU;
//! - `detokenise`: `generate` decodes the **whole** sequence twice every
//!   step, to find the new text. That grows with the conversation.
//!
//! **Inside `forward`.** The small ops between one token's matmuls, rebuilt
//! at Qwen2.5-1.5B's decode shapes on random tensors, and timed as a chain
//! with one sync, as `forward` runs them:
//! - per layer: 2 RMS norms, 3 bias adds, 2 rotations, 3 head transposes,
//!   the KV cache's 2 `cat`s, fused attention, 2 residual adds, SiLU and a
//!   multiply;
//! - per token: the embedding row, the rotary tables' row, the final norm,
//!   and the logits cast and read back.
//!
//! The `cat`s are timed apart, and at two context lengths, because each one
//! copies the whole cache so far.
//!
//!     cargo run --release -p kvad-gpu --example decode_breakdown -- [--quant q8] [--context 2048]
//!
//! Needs Qwen/Qwen2.5-1.5B-Instruct in the Hub cache (`kvad pull`).
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, rotary_emb};
use kvad::sampler::Sampler;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const REPO: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const LAYERS: usize = 28;
const E: usize = 1536;
const HEADS: usize = 12;
const KV_HEADS: usize = 2;
const HD: usize = 128;
const FFN: usize = 8960;
const VOCAB: usize = 151936;
const STEPS: usize = 128;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Res<()> {
    let mut quant = "q8".to_string();
    let mut context = 128usize;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    for pair in argv.chunks(2) {
        match (pair[0].as_str(), pair.get(1)) {
            ("--quant", Some(v)) => quant = v.clone(),
            ("--context", Some(v)) => context = v.parse()?,
            (other, _) => return Err(format!("unknown option {other}").into()),
        }
    }
    let gd = kvad_gpu::model::parse_quant(&quant).ok_or("--quant is none, q8, q4, q4k or q6k")?;
    // What `kvad-gpu run` computes in: f32 around quantised weights, else bf16.
    let act = if gd.is_some() { DType::F32 } else { DType::BF16 };
    let dev = Device::new_metal(0)?;

    around_the_model(gd, context, &dev)?;
    inside_forward(act, context, &dev)?;
    Ok(())
}

/// The real model, stepped the way `Llm::generate` steps it.
fn around_the_model(gd: Option<candle_core::quantized::GgmlDType>, context: usize, dev: &Device) -> Res<()> {
    let id = kvad::weights::model_id(REPO);
    let mut llm = kvad::runtime::Llm::load_custom(
        REPO,
        &mut |_| {},
        &kvad::weights::Watcher::none(),
        &mut |files, spec, progress| {
            kvad_gpu::model::session(&id, &files.weights, spec, DType::BF16, gd, dev.clone(), progress)
        },
    )?;
    println!("{REPO}, {}, context {context}\n", llm.session.label());

    // A prompt of exactly `context` tokens: the README, repeated.
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../README.md"))?;
    let mut prompt = Vec::new();
    while prompt.len() < context {
        prompt.extend(llm.encode(&text)?);
    }
    prompt.truncate(context);

    for (what, mut sampler) in [("greedy", Sampler::new(0.0, 0, 1.0, 7)), ("t 0.7, top-p 0.9", Sampler::new(0.7, 40, 0.9, 7))] {
        llm.session.truncate(0)?;
        let mut logits = llm.session.forward(&prompt)?;
        let mut ids = prompt.clone();
        let (mut fwd, mut samp, mut detok) = (vec![], vec![], vec![]);
        for _ in 0..STEPS {
            let t = Instant::now();
            let next = sampler.sample(&logits);
            samp.push(t.elapsed().as_secs_f64() * 1e3);
            ids.push(next);

            let t = Instant::now();
            let text = llm.decode(&ids)?;
            let previous = llm.decode(&ids[..ids.len() - 1])?;
            std::hint::black_box((text, previous));
            detok.push(t.elapsed().as_secs_f64() * 1e3);

            let t = Instant::now();
            logits = llm.session.forward(&[next])?;
            fwd.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (f, s, d) = (median(fwd), median(samp), median(detok));
        println!(
            "  {what:17} forward {f:6.2} ms   sample {s:5.3} ms   detokenise x2 {d:5.3} ms   \
             = {:6.2} ms, {:5.1} tok/s",
            f + s + d,
            1000.0 / (f + s + d)
        );
    }
    Ok(())
}

/// The small ops of one token's `forward`, without its matmuls.
fn inside_forward(act: DType, context: usize, dev: &Device) -> Res<()> {
    let rand = |shape: &[usize]| -> candle_core::Result<Tensor> {
        Tensor::rand(0f32, 1f32, shape, dev)?.to_dtype(act)
    };
    let x = rand(&[1, E])?;
    let norm_w = rand(&[E])?;
    let bias_q = rand(&[HEADS * HD])?;
    let bias_kv = rand(&[KV_HEADS * HD])?;
    let (q_out, kv_out) = (rand(&[1, HEADS * HD])?, rand(&[1, KV_HEADS * HD])?);
    let (gate, up) = (rand(&[1, FFN])?, rand(&[1, FFN])?);
    let cos_table = rand(&[context + STEPS, HD / 2])?;
    let sin_table = rand(&[context + STEPS, HD / 2])?;
    let embed = rand(&[VOCAB, E])?;
    let ids = Tensor::from_slice(&[42u32], (1,), dev)?;
    let logits = rand(&[1, VOCAB])?;
    let caches: Vec<(Tensor, Tensor)> = (0..LAYERS)
        .map(|_| Ok((rand(&[1, KV_HEADS, context, HD])?, rand(&[1, KV_HEADS, context, HD])?)))
        .collect::<candle_core::Result<_>>()?;

    // One layer's ops as `GpuLlama::run` does them at m = 1, the matmuls'
    // outputs stood in for by tensors of their shape. `cat` says whether
    // the cache is extended, which is the one op whose cost grows.
    let layer = |i: usize, cat: bool| -> candle_core::Result<Tensor> {
        let h = ops::rms_norm(&x, &norm_w, 1e-6)?;
        let q = q_out.broadcast_add(&bias_q)?;
        let k = kv_out.broadcast_add(&bias_kv)?;
        let v = kv_out.broadcast_add(&bias_kv)?;
        let q = q.reshape((1, 1, HEADS, HD))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((1, 1, KV_HEADS, HD))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((1, 1, KV_HEADS, HD))?.transpose(1, 2)?.contiguous()?;
        let cos = cos_table.narrow(0, context, 1)?.contiguous()?;
        let sin = sin_table.narrow(0, context, 1)?.contiguous()?;
        let q = rotary_emb::rope(&q, &cos, &sin)?;
        let k = rotary_emb::rope(&k, &cos, &sin)?;
        let (pk, pv) = &caches[i];
        let (k, v) = match cat {
            true => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            false => (pk.clone(), pv.clone()),
        };
        let out = ops::sdpa(&q, &k, &v, None, false, (1.0 / (HD as f64).sqrt()) as f32, 1.0)?;
        let out = out.transpose(1, 2)?.reshape((1, HEADS * HD))?;
        let x2 = (&x + &out.narrow(D::Minus1, 0, E)?)?;
        let h2 = ops::rms_norm(&x2, &norm_w, 1e-6)?;
        let ffn = (ops::silu(&gate)? * &up)?;
        std::hint::black_box((h, h2, ffn.clone()));
        &x2 + &ffn.narrow(D::Minus1, 0, E)?
    };
    let tail = || -> Res<()> {
        let _ = embed.i(&ids)?;
        let _ = ops::rms_norm(&x, &norm_w, 1e-6)?;
        let l = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        std::hint::black_box(l);
        Ok(())
    };

    let time = |f: &dyn Fn() -> Res<()>| -> Res<f64> {
        f()?;
        dev.synchronize()?;
        let mut ms = Vec::new();
        for _ in 0..15 {
            let t = Instant::now();
            f()?;
            dev.synchronize()?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(median(ms))
    };
    let layers = |cat: bool| -> Res<()> {
        for i in 0..LAYERS {
            layer(i, cat)?;
        }
        Ok(())
    };
    let with_cat = time(&|| layers(true))?;
    let without = time(&|| layers(false))?;
    let tail_ms = time(&tail)?;
    let sync = time(&|| Ok(()))?;
    let ops_per_layer = 23;
    println!("\ninside forward, {act:?} activations, the matmuls left out:\n");
    println!("  {LAYERS} layers of small ops, cache extended   {with_cat:6.2} ms");
    println!("  the same, cache not extended           {without:6.2} ms   so the cats are {:.2} ms", with_cat - without);
    println!("  the tail: embedding row, final norm,");
    println!("    logits cast and read back            {tail_ms:6.2} ms");
    println!("  a sync with nothing to wait for        {sync:6.3} ms");
    println!(
        "\n  about {} small ops a token: {:.1} µs each, cats and tail aside",
        LAYERS * ops_per_layer,
        without * 1000.0 / (LAYERS * ops_per_layer) as f64
    );
    Ok(())
}
