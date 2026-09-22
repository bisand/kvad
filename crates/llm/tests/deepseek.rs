//! DeepSeek V2 and V3, checked against a second implementation.
//!
//! The engine's attention is *absorbed*: the per-head matrices that would turn
//! a cached vector into that head's key and value are folded into the query
//! and into the output instead, so the cache holds one compressed vector per
//! position and nothing is ever decompressed. That is an algebraic identity —
//! `q · (W c) = (Wᵀ q) · c` — and an identity is exactly the kind of thing
//! that is easy to get subtly wrong and impossible to notice, because the
//! wrong version still produces plausible-looking numbers.
//!
//! So this file contains a second DeepSeek: the obvious one, translated
//! line by line from `modeling_deepseek.py`, which builds every head's full
//! key and value, concatenates the rotary part, masks, and takes a softmax.
//! It is slow and it allocates constantly and it is not clever anywhere, and
//! that is the point — the two implementations share no code below the weight
//! files, so agreement between them is evidence about both.
//!
//! The rotary embedding is deliberately translated the long way round, with
//! the reference's interleave-then-halve permutation written out, because the
//! engine's claim that the permutation cancels inside the dot product is one
//! of the things being checked.

use kvad::model::{deepseek, Json, KvCache, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::tensor::rms_norm;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::f32::consts::PI;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// A checkpoint, built here
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random numbers, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 8_388_608.0 - 1.0) * 0.25
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: (0..rows * cols).map(|_| self.next()).collect(),
        }
    }

    /// Norm weights, which are near one rather than near zero.
    fn ones(&mut self, n: usize) -> Mat {
        Mat {
            rows: 1,
            cols: n,
            data: (0..n).map(|_| 1.0 + self.next()).collect(),
        }
    }
}

#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// `y = x @ selfᵀ`, the `nn.Linear` convention the checkpoint is in.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|r| self.row(r).iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }
}

/// The smallest safetensors writer that will do: a JSON header, its length,
/// and the arrays. Writing one here keeps this test free of anything that
/// could also be wrong in the engine.
fn write_safetensors(path: &PathBuf, tensors: &BTreeMap<String, Mat>) {
    let mut header = String::from("{");
    let mut blob: Vec<u8> = Vec::new();
    for (i, (name, m)) in tensors.iter().enumerate() {
        let start = blob.len();
        for v in &m.data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let shape = match m.rows {
            1 => format!("[{}]", m.cols),
            r => format!("[{r},{}]", m.cols),
        };
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape},\"data_offsets\":[{start},{}]}}",
            blob.len()
        ));
    }
    header.push('}');
    // The data section has to start 8-byte aligned, so the header is padded
    // with the one character JSON does not mind.
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&blob);
    std::fs::write(path, out).unwrap();
}

/// Everything about the tiny model both implementations read.
struct Tiny {
    config: serde_json::Value,
    tensors: BTreeMap<String, Mat>,
    n_layer: usize,
    n_head: usize,
    nope: usize,
    rope: usize,
    v_head: usize,
    lat: usize,
    q_lora: Option<usize>,
    hidden: usize,
    vocab: usize,
    n_experts: usize,
    first_dense: usize,
}

/// Which of the two architectures to build. They differ in the router and in
/// whether the query is compressed, so one of each covers every branch.
#[derive(Clone, Copy, PartialEq)]
enum Flavour {
    /// V2-Lite's shape: a direct query projection, softmax scoring, plain
    /// top-k, one group, no normalisation.
    V2,
    /// V3's: a compressed query, sigmoid scoring, a learned choice bias,
    /// two groups of experts, normalised weights and a scaling factor.
    V3,
}

fn tiny(flavour: Flavour) -> Tiny {
    // Every matrix the quantiser sees has a row length that is a multiple of
    // 32, because anything narrower it leaves as f32 — and a tiny model that
    // quietly skipped quantisation would test the wrong loader.
    let (hidden, n_head, nope, rope, v_head, lat) = (64usize, 2usize, 32usize, 16usize, 32, 64);
    let (n_layer, vocab, n_experts, first_dense) = (3usize, 32usize, 4usize, 1usize);
    let (moe_inter, inter, n_shared) = (32usize, 64usize, 1usize);
    let q_lora = match flavour {
        Flavour::V2 => None,
        Flavour::V3 => Some(32),
    };
    let q_head = nope + rope;

    let mut r = Rng(0x1234_5678_9abc_def0);
    let mut t: BTreeMap<String, Mat> = BTreeMap::new();
    t.insert("model.embed_tokens.weight".into(), r.matrix(vocab, hidden));
    t.insert("model.norm.weight".into(), r.ones(hidden));
    t.insert("lm_head.weight".into(), r.matrix(vocab, hidden));

    for l in 0..n_layer {
        let p = format!("model.layers.{l}");
        t.insert(format!("{p}.input_layernorm.weight"), r.ones(hidden));
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            r.ones(hidden),
        );
        match q_lora {
            None => {
                t.insert(
                    format!("{p}.self_attn.q_proj.weight"),
                    r.matrix(n_head * q_head, hidden),
                );
            }
            Some(rank) => {
                t.insert(
                    format!("{p}.self_attn.q_a_proj.weight"),
                    r.matrix(rank, hidden),
                );
                t.insert(format!("{p}.self_attn.q_a_layernorm.weight"), r.ones(rank));
                t.insert(
                    format!("{p}.self_attn.q_b_proj.weight"),
                    r.matrix(n_head * q_head, rank),
                );
            }
        }
        t.insert(
            format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
            r.matrix(lat + rope, hidden),
        );
        t.insert(format!("{p}.self_attn.kv_a_layernorm.weight"), r.ones(lat));
        t.insert(
            format!("{p}.self_attn.kv_b_proj.weight"),
            r.matrix(n_head * (nope + v_head), lat),
        );
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            r.matrix(hidden, n_head * v_head),
        );

        if l < first_dense {
            t.insert(format!("{p}.mlp.gate_proj.weight"), r.matrix(inter, hidden));
            t.insert(format!("{p}.mlp.up_proj.weight"), r.matrix(inter, hidden));
            t.insert(format!("{p}.mlp.down_proj.weight"), r.matrix(hidden, inter));
        } else {
            t.insert(format!("{p}.mlp.gate.weight"), r.matrix(n_experts, hidden));
            if flavour == Flavour::V3 {
                t.insert(
                    format!("{p}.mlp.gate.e_score_correction_bias"),
                    r.ones(n_experts),
                );
            }
            for e in 0..n_experts {
                let q = format!("{p}.mlp.experts.{e}");
                t.insert(format!("{q}.gate_proj.weight"), r.matrix(moe_inter, hidden));
                t.insert(format!("{q}.up_proj.weight"), r.matrix(moe_inter, hidden));
                t.insert(format!("{q}.down_proj.weight"), r.matrix(hidden, moe_inter));
            }
            let q = format!("{p}.mlp.shared_experts");
            t.insert(
                format!("{q}.gate_proj.weight"),
                r.matrix(moe_inter * n_shared, hidden),
            );
            t.insert(
                format!("{q}.up_proj.weight"),
                r.matrix(moe_inter * n_shared, hidden),
            );
            t.insert(
                format!("{q}.down_proj.weight"),
                r.matrix(hidden, moe_inter * n_shared),
            );
        }
    }

    let router = match flavour {
        Flavour::V2 => serde_json::json!({
            "model_type": "deepseek_v2",
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": false,
            "routed_scaling_factor": 1.0,
            "q_lora_rank": serde_json::Value::Null,
        }),
        Flavour::V3 => serde_json::json!({
            "model_type": "deepseek_v3",
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "n_group": 2,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "q_lora_rank": q_lora.unwrap(),
        }),
    };
    let mut config = serde_json::json!({
        "hidden_size": hidden,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,
        "num_hidden_layers": n_layer,
        "vocab_size": vocab,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "n_routed_experts": n_experts,
        "num_experts_per_tok": 2,
        "n_shared_experts": n_shared,
        "first_k_dense_replace": first_dense,
        "moe_layer_freq": 1,
        "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope,
        "v_head_dim": v_head,
        "kv_lora_rank": lat,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": false,
        // Small numbers, but the same shape of YaRN the real configs use —
        // a factor, a shorter trained length, and the two mscales.
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        },
    });
    for (k, v) in router.as_object().unwrap() {
        config[k] = v.clone();
    }

    Tiny {
        config,
        tensors: t,
        n_layer,
        n_head,
        nope,
        rope,
        v_head,
        lat,
        q_lora,
        hidden,
        vocab,
        n_experts,
        first_dense,
    }
}

/// Load the tiny model through the engine's ordinary path.
fn engine(tiny: &Tiny, tag: &str) -> (Box<dyn Transformer>, Spec) {
    at(tiny, tag, Precision::F32)
}

fn at(tiny: &Tiny, tag: &str, precision: Precision) -> (Box<dyn Transformer>, Spec) {
    let path =
        std::env::temp_dir().join(format!("kvad-ds-{}-{tag}.safetensors", std::process::id()));
    write_safetensors(&path, &tiny.tensors);
    let ckpt = Checkpoint::open(std::slice::from_ref(&path)).unwrap();
    let spec = Spec::from_config(Json::new(tiny.config.clone())).unwrap();
    let live = Live::new(&ckpt, precision);
    let model = deepseek::Model::load(&live, spec.clone()).unwrap();

    // Every fixture, every flavour: nothing in the file goes unread. Checking it
    // here rather than in a test of its own means each new shape this file
    // learns to build — V2, V3, the compressed query, the fp8 variant — is
    // covered the day it is added. A weight nobody reads is a piece of the model
    // that is not running, and this is the largest architecture here, so it is
    // the one with the most places to lose one.
    let unread = live.unread();
    assert!(unread.is_empty(), "{tag}: the loader never read {unread:?}");

    let _ = std::fs::remove_file(&path);
    (Box::new(model), spec)
}

// ---------------------------------------------------------------------------
// The second implementation
// ---------------------------------------------------------------------------

/// YaRN's `0.1 m ln(s) + 1`.
fn mscale(scale: f32, m: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * m * scale.ln() + 1.0
    }
}

/// Cosine and sine tables, as `_set_cos_sin_cache` builds them: `dim` wide,
/// the first half repeated.
fn yarn_tables(dim: usize, max_pos: usize, base: f32, cfg: &serde_json::Value) -> (Mat, Mat) {
    let s = cfg["rope_scaling"]["factor"].as_f64().unwrap() as f32;
    let trained = cfg["rope_scaling"]["original_max_position_embeddings"]
        .as_f64()
        .unwrap() as f32;
    let beta_fast = cfg["rope_scaling"]["beta_fast"].as_f64().unwrap() as f32;
    let beta_slow = cfg["rope_scaling"]["beta_slow"].as_f64().unwrap() as f32;
    let m_rope = cfg["rope_scaling"]["mscale"].as_f64().unwrap() as f32;
    let m_all = cfg["rope_scaling"]["mscale_all_dim"].as_f64().unwrap() as f32;

    let extra: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();
    let inter: Vec<f32> = extra.iter().map(|f| f / s).collect();

    let corr = |rot: f32| (dim as f32 * (trained / (rot * 2.0 * PI)).ln()) / (2.0 * base.ln());
    let low = corr(beta_fast).floor().max(0.0);
    let high = corr(beta_slow).ceil().min(dim as f32 - 1.0);
    let high = if (high - low).abs() < f32::EPSILON {
        high + 0.001
    } else {
        high
    };

    let inv: Vec<f32> = (0..dim / 2)
        .map(|i| {
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            inter[i] * (1.0 - mask) + extra[i] * mask
        })
        .collect();

    let amp = mscale(s, m_rope) / mscale(s, m_all);
    let mut cos = Mat {
        rows: max_pos,
        cols: dim,
        data: vec![0.0; max_pos * dim],
    };
    let mut sin = cos.clone();
    for pos in 0..max_pos {
        for (i, f) in inv.iter().enumerate() {
            let a = pos as f32 * f;
            for half in [0, dim / 2] {
                cos.data[pos * dim + half + i] = a.cos() * amp;
                sin.data[pos * dim + half + i] = a.sin() * amp;
            }
        }
    }
    (cos, sin)
}

/// `apply_rotary_pos_emb`, written out: permute pairs into the half-split
/// layout, then `x * cos + rotate_half(x) * sin`.
fn apply_rope_reference(x: &[f32], pos: usize, cos: &Mat, sin: &Mat) -> Vec<f32> {
    let d = x.len();
    // view(d/2, 2).transpose(-1, -2).reshape(d)
    let permuted: Vec<f32> = (0..d)
        .map(|j| {
            if j < d / 2 {
                x[2 * j]
            } else {
                x[2 * (j - d / 2) + 1]
            }
        })
        .collect();
    let (c, s) = (cos.row(pos), sin.row(pos));
    (0..d)
        .map(|j| {
            // rotate_half: (-x2, x1)
            let rotated = if j < d / 2 {
                -permuted[j + d / 2]
            } else {
                permuted[j - d / 2]
            };
            permuted[j] * c[j] + rotated * s[j]
        })
        .collect()
}

fn softmax(v: &mut [f32]) {
    let top = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for x in v.iter_mut() {
        *x = (*x - top).exp();
        total += *x;
    }
    for x in v {
        *x /= total;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One SwiGLU MLP, from the three stored matrices.
fn ffn(t: &BTreeMap<String, Mat>, prefix: &str, x: &[f32]) -> Vec<f32> {
    let gate = t[&format!("{prefix}.gate_proj.weight")].apply(x);
    let up = t[&format!("{prefix}.up_proj.weight")].apply(x);
    let hidden: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    t[&format!("{prefix}.down_proj.weight")].apply(&hidden)
}

/// The whole model, the obvious way: full keys and values per head, an
/// explicit causal mask, nothing absorbed and nothing cached.
fn reference(tiny: &Tiny, tokens: &[u32]) -> Vec<Vec<f32>> {
    let t = &tiny.tensors;
    let cfg = &tiny.config;
    let eps = cfg["rms_norm_eps"].as_f64().unwrap() as f32;
    let (n_head, nope, rope_d, v_head, lat) =
        (tiny.n_head, tiny.nope, tiny.rope, tiny.v_head, tiny.lat);
    let q_head = nope + rope_d;
    let m = tokens.len();

    let (cos, sin) = yarn_tables(rope_d, 64, cfg["rope_theta"].as_f64().unwrap() as f32, cfg);
    let mut scale = 1.0 / (q_head as f32).sqrt();
    let s = cfg["rope_scaling"]["factor"].as_f64().unwrap() as f32;
    let all = cfg["rope_scaling"]["mscale_all_dim"].as_f64().unwrap() as f32;
    let ms = mscale(s, all);
    scale *= ms * ms;

    let embed = &t["model.embed_tokens.weight"];
    let ones = vec![1.0f32; tiny.hidden];
    let _ = &ones;
    let mut xs: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&tok| embed.row(tok as usize).to_vec())
        .collect();

    for l in 0..tiny.n_layer {
        let p = format!("model.layers.{l}");
        let hs: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| rms_norm(x, &t[&format!("{p}.input_layernorm.weight")].data, eps))
            .collect();

        // Queries, compressed or not.
        let qs: Vec<Vec<f32>> = hs
            .iter()
            .map(|h| match tiny.q_lora {
                None => t[&format!("{p}.self_attn.q_proj.weight")].apply(h),
                Some(_) => {
                    let mid = t[&format!("{p}.self_attn.q_a_proj.weight")].apply(h);
                    let mid = rms_norm(
                        &mid,
                        &t[&format!("{p}.self_attn.q_a_layernorm.weight")].data,
                        eps,
                    );
                    t[&format!("{p}.self_attn.q_b_proj.weight")].apply(&mid)
                }
            })
            .collect();

        // Keys and values, built in full from the compressed vector.
        let mut k_pe: Vec<Vec<f32>> = Vec::with_capacity(m);
        let mut k_nope: Vec<Vec<f32>> = Vec::with_capacity(m);
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(m);
        for (i, h) in hs.iter().enumerate() {
            let kv = t[&format!("{p}.self_attn.kv_a_proj_with_mqa.weight")].apply(h);
            let c = rms_norm(
                &kv[..lat],
                &t[&format!("{p}.self_attn.kv_a_layernorm.weight")].data,
                eps,
            );
            k_pe.push(apply_rope_reference(&kv[lat..], i, &cos, &sin));
            let full = t[&format!("{p}.self_attn.kv_b_proj.weight")].apply(&c);
            let mut kn = Vec::with_capacity(n_head * nope);
            let mut vv = Vec::with_capacity(n_head * v_head);
            for head in 0..n_head {
                let at = head * (nope + v_head);
                kn.extend_from_slice(&full[at..at + nope]);
                vv.extend_from_slice(&full[at + nope..at + nope + v_head]);
            }
            k_nope.push(kn);
            values.push(vv);
        }

        let mut attn: Vec<Vec<f32>> = vec![vec![0.0; n_head * v_head]; m];
        for (i, q) in qs.iter().enumerate() {
            for head in 0..n_head {
                let q_nope = &q[head * q_head..head * q_head + nope];
                let q_pe = apply_rope_reference(
                    &q[head * q_head + nope..(head + 1) * q_head],
                    i,
                    &cos,
                    &sin,
                );
                // Causal by construction: only positions up to i.
                let mut scores: Vec<f32> = (0..=i)
                    .map(|j| {
                        let kn = &k_nope[j][head * nope..(head + 1) * nope];
                        let mut dot: f32 = q_nope.iter().zip(kn).map(|(a, b)| a * b).sum();
                        dot += q_pe.iter().zip(&k_pe[j]).map(|(a, b)| a * b).sum::<f32>();
                        dot * scale
                    })
                    .collect();
                softmax(&mut scores);
                for (j, w) in scores.iter().enumerate() {
                    let v = &values[j][head * v_head..(head + 1) * v_head];
                    for (o, val) in attn[i][head * v_head..(head + 1) * v_head]
                        .iter_mut()
                        .zip(v)
                    {
                        *o += w * val;
                    }
                }
            }
        }
        for (x, a) in xs.iter_mut().zip(&attn) {
            let proj = t[&format!("{p}.self_attn.o_proj.weight")].apply(a);
            for (xi, pv) in x.iter_mut().zip(&proj) {
                *xi += pv;
            }
        }

        // The MLP: dense for the first layers, a mixture after them.
        for x in xs.iter_mut() {
            let h = rms_norm(
                x,
                &t[&format!("{p}.post_attention_layernorm.weight")].data,
                eps,
            );
            let out = if l < tiny.first_dense {
                ffn(t, &format!("{p}.mlp"), &h)
            } else {
                let mut scores = t[&format!("{p}.mlp.gate.weight")].apply(&h);
                let sigmoid = cfg["scoring_func"] == "sigmoid";
                if sigmoid {
                    for s in &mut scores {
                        *s = 1.0 / (1.0 + (-*s).exp());
                    }
                } else {
                    softmax(&mut scores);
                }
                let mut choice = scores.clone();
                if let Some(bias) = t.get(&format!("{p}.mlp.gate.e_score_correction_bias")) {
                    for (c, b) in choice.iter_mut().zip(&bias.data) {
                        *c += b;
                    }
                }
                let n_group = cfg["n_group"].as_u64().unwrap() as usize;
                let topk_group = cfg["topk_group"].as_u64().unwrap() as usize;
                if n_group > 1 {
                    let per = tiny.n_experts / n_group;
                    // V3 scores a group by its best two; V2's group-limited
                    // variant by its best one. Only V3 is built here.
                    let mut strength: Vec<(usize, f32)> = (0..n_group)
                        .map(|g| {
                            let mut vs = choice[g * per..(g + 1) * per].to_vec();
                            vs.sort_by(|a, b| b.total_cmp(a));
                            (g, vs.iter().take(2).sum())
                        })
                        .collect();
                    strength.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for &(g, _) in &strength[topk_group..] {
                        for v in &mut choice[g * per..(g + 1) * per] {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }
                let top_k = cfg["num_experts_per_tok"].as_u64().unwrap() as usize;
                let mut order: Vec<usize> = (0..tiny.n_experts).collect();
                order.sort_by(|&a, &b| choice[b].total_cmp(&choice[a]));
                let picked: Vec<usize> = order.into_iter().take(top_k).collect();
                let mut weights: Vec<f32> = picked.iter().map(|&e| scores[e]).collect();

                let norm = cfg["norm_topk_prob"].as_bool().unwrap() && top_k > 1;
                if norm {
                    let total: f32 = weights.iter().sum::<f32>() + 1e-20;
                    for w in &mut weights {
                        *w /= total;
                    }
                }
                let factor = cfg["routed_scaling_factor"].as_f64().unwrap() as f32;
                if sigmoid || !norm {
                    for w in &mut weights {
                        *w *= factor;
                    }
                }

                let mut out = ffn(t, &format!("{p}.mlp.shared_experts"), &h);
                for (&e, &w) in picked.iter().zip(&weights) {
                    let y = ffn(t, &format!("{p}.mlp.experts.{e}"), &h);
                    for (o, v) in out.iter_mut().zip(&y) {
                        *o += w * v;
                    }
                }
                out
            };
            for (xi, v) in x.iter_mut().zip(&out) {
                *xi += v;
            }
        }
    }

    xs.iter()
        .map(|x| {
            let h = rms_norm(x, &t["model.norm.weight"].data, eps);
            t["lm_head.weight"].apply(&h)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// How far apart two logits may be and still be the same number.
///
/// Both implementations are f32 and sum in different orders, so exact
/// equality is not available. A wrong permutation, a transposed matrix or a
/// missing normalisation moves logits by whole units; this catches all of
/// them with room to spare.
fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "{what}: worst disagreement {worst}");
    assert!(a.iter().all(|v| v.is_finite()), "{what}: not finite");
}

/// The claim the whole module rests on: scoring against the compressed vector
/// gives the same answer as reconstructing every head's key and value.
#[test]
fn absorbed_attention_agrees_with_the_obvious_implementation() {
    for (flavour, tag) in [(Flavour::V2, "v2"), (Flavour::V3, "v3")] {
        let t = tiny(flavour);
        let (model, spec) = engine(&t, tag);
        let tokens: Vec<u32> = vec![3, 17, 8, 0, 29, 11, 5];

        let mut cache = KvCache::new(&spec);
        let got = model.forward_batch_all(&tokens, &mut cache);
        let want = reference(&t, &tokens);

        assert_eq!(got.len(), tokens.len() * t.vocab);
        for (i, row) in want.iter().enumerate() {
            close(
                &got[i * t.vocab..(i + 1) * t.vocab],
                row,
                &format!("{tag} position {i}"),
            );
        }
    }
}

/// Prefill and decode must be the same model.
///
/// They share `run_batch`, but not the path into it: one pushes seven
/// positions into the cache in a single pass and the other grows it a
/// position at a time, and the second is the one that has to keep working
/// after a conversation has been going for a while.
#[test]
fn feeding_tokens_one_at_a_time_gives_the_same_answer() {
    for (flavour, tag) in [(Flavour::V2, "one-v2"), (Flavour::V3, "one-v3")] {
        let t = tiny(flavour);
        let (model, spec) = engine(&t, tag);
        let tokens: Vec<u32> = vec![3, 17, 8, 0, 29];

        let mut batched = KvCache::new(&spec);
        let all_at_once = model.forward_batch(&tokens, &mut batched);

        let mut stepped = KvCache::new(&spec);
        let mut last = Vec::new();
        for &tok in &tokens {
            last = model.forward(tok, &mut stepped);
        }

        assert_eq!(batched.len, stepped.len);
        close(&all_at_once, &last, tag);
    }
}

/// The cache holds the compressed vector and the shared rotary key, and
/// nothing else. If that ever stops being true the memory claim in the module
/// header stops being true with it.
#[test]
fn the_cache_holds_the_latent_and_not_the_keys() {
    let t = tiny(Flavour::V2);
    let (model, spec) = engine(&t, "cache");
    assert_eq!(spec.cache.k, t.rope);
    assert_eq!(spec.cache.v, t.lat);

    // What multi-head attention would have stored, for the same model.
    let plain = t.n_head * (t.nope + t.rope + t.v_head);
    assert!(
        spec.cache.k + spec.cache.v < plain,
        "latent cache {} is not smaller than {plain}",
        spec.cache.k + spec.cache.v
    );

    let mut cache = KvCache::new(&spec);
    model.forward_batch(&[1, 2, 3], &mut cache);
    assert_eq!(cache.len, 3);
    assert_eq!(cache.keys(0).len(), 3 * t.rope);
    assert_eq!(cache.values(0).len(), 3 * t.lat);
    assert_eq!(cache.bytes(), 3 * (t.rope + t.lat) * t.n_layer * 4);
}

/// Truncation is what makes a second chat turn cheap. Under MLA the two
/// streams have different widths, which is exactly the sort of thing a
/// `truncate` written for one width gets wrong.
#[test]
fn a_truncated_cache_continues_correctly() {
    let t = tiny(Flavour::V2);
    let (model, spec) = engine(&t, "truncate");
    let tokens: Vec<u32> = vec![3, 17, 8, 0, 29];

    let mut fresh = KvCache::new(&spec);
    let want = model.forward_batch(&tokens, &mut fresh);

    let mut reused = KvCache::new(&spec);
    model.forward_batch(&[3, 17, 8, 31, 4, 9], &mut reused);
    assert_eq!(reused.truncate(3), 3, "MLA caches rows, so it rewinds exactly");
    assert_eq!(reused.keys(0).len(), 3 * t.rope);
    assert_eq!(reused.values(0).len(), 3 * t.lat);
    let got = model.forward_batch(&tokens[3..], &mut reused);
    close(&got, &want, "after truncation");
}

/// V3's multi-token-prediction head must be passed over in silence, not
/// stumbled over.
///
/// It is a training-time device the module header says is not implemented, and
/// its weights sit in the checkpoint all the same: a whole extra block, an
/// embedding table and two norms, filed at `layers.{n_layer}`. The check that
/// names tensors nobody read cannot tell *skipped on purpose* from *forgotten*
/// by itself, so the loader has to say which — and if it stops saying so, every
/// real V3 checkpoint fails to load rather than running as it does today.
#[test]
fn v3s_extra_prediction_head_is_skipped_on_purpose() {
    let mut t = tiny(Flavour::V3);
    let mtp = t.n_layer; // the head's block sits one past the last real layer.
    t.config["num_nextn_predict_layers"] = serde_json::json!(1);

    // The shapes do not matter — nothing reads them. Their presence does.
    let hidden = t.hidden;
    let p = format!("model.layers.{mtp}");
    let mut r = Rng(0x9999_8888_7777_6666);
    for (name, m) in [
        (format!("{p}.embed_tokens.weight"), r.matrix(t.vocab, hidden)),
        (format!("{p}.enorm.weight"), r.matrix(1, hidden)),
        (format!("{p}.hnorm.weight"), r.matrix(1, hidden)),
        (format!("{p}.eh_proj.weight"), r.matrix(hidden, 2 * hidden)),
        (format!("{p}.shared_head.norm.weight"), r.matrix(1, hidden)),
        (format!("{p}.shared_head.head.weight"), r.matrix(t.vocab, hidden)),
        (format!("{p}.input_layernorm.weight"), r.matrix(1, hidden)),
    ] {
        t.tensors.insert(name, m);
    }

    // `at` asserts that nothing was left unread, which is the whole point.
    let (model, spec) = at(&t, "mtp", Precision::F32);
    assert_eq!(model.spec().n_layer, spec.n_layer, "the extra block is not a layer");
}

/// fp8 checkpoints are refused where somebody can still do something about
/// it, rather than thirty gigabytes later.
#[test]
fn an_fp8_checkpoint_is_refused_by_name() {
    let mut config = tiny(Flavour::V3).config;
    config["quantization_config"] = serde_json::json!({"quant_method": "fp8", "fmt": "e4m3"});
    let err = Spec::from_config(Json::new(config))
        .unwrap_err()
        .to_string();
    assert!(err.contains("fp8"), "{err}");
    assert!(err.contains("bf16"), "{err}");
}

/// The quantised path is the one anybody will actually run, and MLA gives it
/// something new to get wrong: the per-head slices of `kv_b_proj` are
/// quantised as matrices of their own, one of them transposed first.
///
/// This does not check that q8 is *accurate* — `quant.rs` has tests for that.
/// It checks that the same model, loaded the other way, still predicts the
/// same tokens.
#[test]
fn the_quantised_path_predicts_the_same_tokens() {
    for (flavour, tag) in [(Flavour::V2, "q8-v2"), (Flavour::V3, "q8-v3")] {
        let t = tiny(flavour);
        let tokens: Vec<u32> = vec![3, 17, 8, 0, 29];

        let (exact, spec) = at(&t, tag, Precision::F32);
        let (rough, _) = at(&t, tag, Precision::Q8);

        let mut a = KvCache::new(&spec);
        let mut b = KvCache::new(&spec);
        let want = exact.forward_batch(&tokens, &mut a);
        let got = rough.forward_batch(&tokens, &mut b);

        // Weights this small and this random are the worst case for a block
        // quantiser, so the logits are compared by their order, not their
        // values: the best token, and the direction of every gap.
        let best = |v: &[f32]| {
            let pick = |m: (usize, f32), (i, &x): (usize, &f32)| if x > m.1 { (i, x) } else { m };
            v.iter().enumerate().fold((0, f32::NEG_INFINITY), pick).0
        };
        assert_eq!(best(&got), best(&want), "{tag}: different argmax");
        assert!(got.iter().all(|v| v.is_finite()), "{tag}: not finite");

        let spread = want.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - want.iter().copied().fold(f32::INFINITY, f32::min);
        let worst = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < spread / 10.0, "{tag}: q8 moved a logit by {worst} of {spread}");
    }
}

/// Write the tiny model, and this engine's logits for it, where somebody can
/// check them against `transformers`.
///
/// The second implementation in this file is a careful reading of
/// `modeling_deepseek.py`, and a careful reading can still be a wrong one —
/// the same misunderstanding would be in both. The only cure is to run the
/// actual reference, which needs Python, PyTorch and `trust_remote_code`, and
/// which therefore cannot live in `cargo test`.
///
/// So this is the seam. Set the variable and it writes a directory that
/// `AutoModelForCausalLM.from_pretrained` will load, plus `logits.json` — what
/// this engine says every position's distribution is. `scripts/check-deepseek.py`
/// loads both and compares.
///
///     KVAD_DEEPSEEK_FIXTURE=/tmp/ds cargo test -p kvad --test deepseek -- --ignored
#[test]
#[ignore = "writes a fixture for the Python reference; needs a directory to write to"]
fn write_a_fixture_for_the_reference_implementation() {
    let Ok(dir) = std::env::var("KVAD_DEEPSEEK_FIXTURE") else {
        panic!("set KVAD_DEEPSEEK_FIXTURE to a directory to write");
    };
    for (flavour, name) in [(Flavour::V2, "v2"), (Flavour::V3, "v3")] {
        let t = tiny(flavour);
        let out = std::path::Path::new(&dir).join(name);
        std::fs::create_dir_all(&out).unwrap();
        write_safetensors(&out.join("model.safetensors"), &t.tensors);
        std::fs::write(
            out.join("config.json"),
            serde_json::to_string_pretty(&t.config).unwrap(),
        )
        .unwrap();

        let tokens: Vec<u32> = vec![3, 17, 8, 0, 29, 11, 5];
        let (model, spec) = engine(&t, &format!("fixture-{name}"));
        let mut cache = KvCache::new(&spec);
        let logits = model.forward_batch_all(&tokens, &mut cache);
        let rows: Vec<Vec<f32>> =
            logits.chunks(t.vocab).map(<[f32]>::to_vec).collect();
        std::fs::write(
            out.join("logits.json"),
            serde_json::to_string(&serde_json::json!({"tokens": tokens, "logits": rows})).unwrap(),
        )
        .unwrap();
        println!("wrote {}", out.display());
    }
}
