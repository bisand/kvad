//! What the KV cache costs, and what the score loop's shape is worth.
//!
//!     cargo run --release -p kvad --example bench_attend
//!
//! `attend` is the only part of a decode step whose work grows with the
//! conversation: every other matmul is the same size at position 10 and at
//! position 4000. Nothing measured it until this existed, which is why the
//! score loop kept a shape that cost 1.3x for as long as it did.
//!
//! Shapes are synthetic so this needs no model. Both versions run in one
//! process, alternating round by round: the difference is a few per cent of a
//! decode step, which two binaries on a machine doing anything else cannot
//! resolve.
use kvad::model::{attend, Json, Spec};
use kvad::tensor::softmax_inplace;
use nervus::rng::Rng;
use rayon::prelude::*;
use std::time::Instant;

/// `attend` as it was: the score dot written out as one running sum.
///
/// A copy rather than a flag, so both paths compile exactly as they would on
/// their own and neither pays for the other's existence.
fn attend_serial(
    spec: &Spec,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_positions: usize,
) -> Vec<f32> {
    let hd = spec.head_dim;
    let kv_dim = spec.kv_dim();
    let group = spec.group_size();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; spec.n_head * hd];
    let per_task = spec.n_head.div_ceil(rayon::current_num_threads().max(1)).max(1);
    out.par_chunks_mut(per_task * hd).enumerate().for_each(|(task, dst)| {
        let mut scores = Vec::with_capacity(n_positions);
        for (j, slot) in dst.chunks_mut(hd).enumerate() {
            let head = task * per_task + j;
            let q_head = &q[head * hd..(head + 1) * hd];
            let kv_off = (head / group) * hd;
            scores.clear();
            for t in 0..n_positions {
                let base = t * kv_dim + kv_off;
                let k_head = &k_cache[base..base + hd];
                let mut dot = 0.0f32;
                for i in 0..hd {
                    dot += q_head[i] * k_head[i];
                }
                scores.push(dot * scale);
            }
            softmax_inplace(&mut scores);
            for (t, &w) in scores.iter().enumerate() {
                if w < 1e-8 {
                    continue;
                }
                let base = t * kv_dim + kv_off;
                let v_head = &v_cache[base..base + hd];
                for i in 0..hd {
                    slot[i] += w * v_head[i];
                }
            }
        }
    });
    out
}

fn spec(n_head: usize, n_kv_head: usize, head_dim: usize) -> Spec {
    let cfg = serde_json::json!({
        "model_type": "llama",
        "hidden_size": n_head * head_dim,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_head,
        "head_dim": head_dim,
        "num_hidden_layers": 24,
        "vocab_size": 32000,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
    });
    Spec::from_config(Json::new(cfg)).expect("spec")
}

fn main() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(kvad::model::threads())
        .build()
        .unwrap();
    pool.install(run);
}

fn run() {
    println!("threads {}\n", rayon::current_num_threads());
    for (label, nh, nkv, hd, layers) in [
        ("qwen2.5-0.5b", 14usize, 2usize, 64usize, 24usize),
        ("llama-8b-ish", 32, 8, 128, 32),
    ] {
        let s = spec(nh, nkv, hd);
        let kv_dim = s.kv_dim();
        println!("{label}: {nh} heads, {nkv} kv, head_dim {hd}, {layers} layers");
        println!(
            "  {:>6} {:>10} {:>10} {:>7} {:>11} {:>10}",
            "ctx", "serial us", "8-lane us", "ratio", "MB/layer", "ms/token"
        );
        let mut rng = Rng::new(1);
        for ctx in [128usize, 512, 2048, 8192] {
            let q: Vec<f32> = (0..nh * hd).map(|_| rng.normal()).collect();
            let k: Vec<f32> = (0..ctx * kv_dim).map(|_| rng.normal()).collect();
            let v: Vec<f32> = (0..ctx * kv_dim).map(|_| rng.normal()).collect();

            let a = attend_serial(&s, &q, &k, &v, ctx);
            let b = attend(&s, &q, &k, &v, ctx);
            let worst = a.iter().zip(b.iter()).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(worst <= 1e-4 * scale.max(1e-3), "disagree by {worst}");

            for _ in 0..5 {
                std::hint::black_box(attend_serial(&s, &q, &k, &v, ctx));
                std::hint::black_box(attend(&s, &q, &k, &v, ctx));
            }
            let probe = Instant::now();
            std::hint::black_box(attend(&s, &q, &k, &v, ctx));
            let iters =
                ((0.05 / probe.elapsed().as_secs_f64().max(1e-9)) as usize).clamp(20, 50_000);
            let mut best = [f64::MAX; 2];
            // Alternate round by round so drift lands on both.
            for _ in 0..7 {
                let t0 = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(attend_serial(&s, &q, &k, &v, ctx));
                }
                best[0] = best[0].min(t0.elapsed().as_secs_f64() / iters as f64);
                let t0 = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(attend(&s, &q, &k, &v, ctx));
                }
                best[1] = best[1].min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            println!(
                "  {ctx:>6} {:>10.1} {:>10.1} {:>6.2}x {:>11.2} {:>10.2}",
                best[0] * 1e6,
                best[1] * 1e6,
                best[0] / best[1],
                (ctx * kv_dim * 2 * 4) as f64 / 1e6,
                best[1] * layers as f64 * 1e3
            );
        }
        println!();
    }
}
