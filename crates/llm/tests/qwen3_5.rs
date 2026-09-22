//! Qwen3.5/3.8 against a second implementation of the same arithmetic.
//!
//! The engine's version of a gated delta net is fast and shares code with the
//! rest of the engine. This file's is slow, allocates freely, and shares
//! nothing — it is written straight from the recurrence:
//!
//! ```text
//!     S <- S · e^g ;  δ = (v − Sᵀk)·β ;  S <- S + k⊗δ ;  y = Sᵀq
//! ```
//!
//! Two implementations of one formula, agreeing on random weights, is the only
//! check available before the 27B checkpoint exists on this machine. A linear
//! layer that decays with the wrong sign, or normalises the key after scaling
//! the query instead of before, produces numbers either way — it does not
//! produce *these* numbers.
//!
//! The last test is the one the issue asks for by name: feeding a prompt whole
//! must equal feeding it in two pieces. For an attention model that is nearly
//! a tautology. For this one it is the whole question, because the state
//! carried between the two calls is the only thing tying them together.

use kvad::model::{Json, KvCache, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift64*, so a failure reproduces exactly.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32 / 16777216.0 - 0.5) * 0.4
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Mat {
        let data = (0..rows * cols).map(|_| self.next()).collect();
        Mat { rows, cols, data }
    }

    /// A plain norm's weight, which this family stores as an offset from one
    /// and trains from zero.
    fn offsets(&mut self, n: usize) -> Mat {
        let data = (0..n).map(|_| self.next() * 0.2).collect();
        Mat { rows: 1, cols: n, data }
    }

    /// The gated norm's, and the delta net's scalars, which are scales rather
    /// than offsets and sit near one.
    fn ones(&mut self, n: usize) -> Mat {
        let data = (0..n).map(|_| 1.0 + self.next() * 0.1).collect();
        Mat { rows: 1, cols: n, data }
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

    /// `y = W x`, with `W` stored `[out, in]`.
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.rows).map(|r| self.row(r).iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }
}

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
    hidden: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    rope_dim: usize,
    n_layer: usize,
    vocab: usize,
    // the linear half
    n_k_head: usize,
    n_v_head: usize,
    k_head: usize,
    v_head: usize,
    conv: usize,
    eps: f32,
    theta: f32,
}

/// A Qwen3.8 small enough to build from a seeded generator, with both kinds of
/// layer and every width a multiple of 32 so the quantisers will take it.
///
/// Four layers so that `layer_types` has three linear and one full, which is
/// the published pattern rather than a shape invented for the test.
fn tiny() -> Tiny {
    tiny_with(None)
}

/// The same model with a chosen layer pattern, for isolating one mixer from
/// the other when the two implementations disagree.
fn tiny_with(pattern: Option<&[&str]>) -> Tiny {
    let (hidden, n_head, n_kv, head_dim) = (64usize, 4usize, 2usize, 32usize);
    let (n_k_head, n_v_head, k_head, v_head, conv) = (2usize, 4usize, 32usize, 32usize, 4usize);
    let (n_layer, vocab, inter) = (pattern.map_or(4, <[&str]>::len), 64usize, 128usize);
    let (eps, theta) = (1e-6f32, 1_000_000.0f32);
    let key_dim = n_k_head * k_head;
    let value_dim = n_v_head * v_head;
    let conv_dim = key_dim * 2 + value_dim;
    let rope_dim = head_dim / 2; // partial_rotary_factor 0.5

    let mut r = Rng(0x0BAD_C0DE_F00D_1234);
    let mut t: BTreeMap<String, Mat> = BTreeMap::new();
    t.insert("model.language_model.embed_tokens.weight".into(), r.matrix(vocab, hidden));
    t.insert("model.language_model.norm.weight".into(), r.offsets(hidden));
    t.insert("lm_head.weight".into(), r.matrix(vocab, hidden));

    let kinds: Vec<&str> = match pattern {
        Some(p) => p.to_vec(),
        None => (0..n_layer)
            .map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" })
            .collect(),
    };

    for (l, kind) in kinds.iter().enumerate() {
        let p = format!("model.language_model.layers.{l}");
        t.insert(format!("{p}.input_layernorm.weight"), r.offsets(hidden));
        t.insert(format!("{p}.post_attention_layernorm.weight"), r.offsets(hidden));
        t.insert(format!("{p}.mlp.gate_proj.weight"), r.matrix(inter, hidden));
        t.insert(format!("{p}.mlp.up_proj.weight"), r.matrix(inter, hidden));
        t.insert(format!("{p}.mlp.down_proj.weight"), r.matrix(hidden, inter));
        match *kind {
            "full_attention" => {
                let q = format!("{p}.self_attn");
                // Twice as wide as the queries: query then gate, per head.
                t.insert(format!("{q}.q_proj.weight"), r.matrix(n_head * head_dim * 2, hidden));
                t.insert(format!("{q}.k_proj.weight"), r.matrix(n_kv * head_dim, hidden));
                t.insert(format!("{q}.v_proj.weight"), r.matrix(n_kv * head_dim, hidden));
                t.insert(format!("{q}.o_proj.weight"), r.matrix(hidden, n_head * head_dim));
                t.insert(format!("{q}.q_norm.weight"), r.offsets(head_dim));
                t.insert(format!("{q}.k_norm.weight"), r.offsets(head_dim));
            }
            _ => {
                let q = format!("{p}.linear_attn");
                t.insert(format!("{q}.in_proj_qkv.weight"), r.matrix(conv_dim, hidden));
                t.insert(format!("{q}.in_proj_z.weight"), r.matrix(value_dim, hidden));
                t.insert(format!("{q}.in_proj_b.weight"), r.matrix(n_v_head, hidden));
                t.insert(format!("{q}.in_proj_a.weight"), r.matrix(n_v_head, hidden));
                t.insert(format!("{q}.conv1d.weight"), r.matrix(conv_dim, conv));
                t.insert(format!("{q}.dt_bias"), r.ones(n_v_head));
                t.insert(format!("{q}.A_log"), r.ones(n_v_head));
                t.insert(format!("{q}.norm.weight"), r.ones(v_head));
                t.insert(format!("{q}.out_proj.weight"), r.matrix(hidden, value_dim));
            }
        }
    }

    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": n_layer,
            "num_attention_heads": n_head,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
            "hidden_size": hidden,
            "intermediate_size": inter,
            "vocab_size": vocab,
            "max_position_embeddings": 64,
            "rms_norm_eps": eps,
            "tie_word_embeddings": false,
            "attn_output_gate": true,
            "layer_types": kinds,
            "full_attention_interval": 4,
            "linear_num_key_heads": n_k_head,
            "linear_num_value_heads": n_v_head,
            "linear_key_head_dim": k_head,
            "linear_value_head_dim": v_head,
            "linear_conv_kernel_dim": conv,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": theta,
                "partial_rotary_factor": 0.5,
            },
        },
    });

    Tiny {
        config,
        tensors: t,
        hidden,
        n_head,
        n_kv,
        head_dim,
        rope_dim,
        n_layer,
        vocab,
        n_k_head,
        n_v_head,
        k_head,
        v_head,
        conv,
        eps,
        theta,
    }
}

fn engine(tiny: &Tiny, tag: &str) -> (Box<dyn Transformer>, Spec) {
    let path =
        std::env::temp_dir().join(format!("kvad-q35-{}-{tag}.safetensors", std::process::id()));
    write_safetensors(&path, &tiny.tensors);
    let ckpt = Checkpoint::open(std::slice::from_ref(&path)).unwrap();
    let spec = Spec::from_config(Json::new(tiny.config.clone())).unwrap();
    let src = Live::new(&ckpt, Precision::F32);
    let model = kvad::model::qwen3_5::Model::load(&src, spec.clone()).unwrap();
    let left = src.unread();
    assert!(left.is_empty(), "the loader left tensors unread: {left:?}");
    let _ = std::fs::remove_file(&path);
    (Box::new(model), spec)
}

// ---------------------------------------------------------------------------
// The second implementation
// ---------------------------------------------------------------------------

/// This family's plain RMSNorm: `x_norm · (1 + w)`.
///
/// The stored weight is an offset from one, initialised to zeros. The engine
/// folds the one in at load; this does it here, so the two arrive at the same
/// answer by different routes rather than by sharing a mistake.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * (1.0 + g)).collect()
}

/// The *gated* norm's, which scales by `w` itself. Same file, other
/// convention.
fn rms_norm_gated(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / n).collect()
}

/// Partial rotary embedding, `rotate_half`, over the first `dim` coordinates.
fn rope(x: &mut [f32], pos: usize, dim: usize, theta: f32) {
    let half = dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
        let (c, s) = ((pos as f32 * freq).cos(), (pos as f32 * freq).sin());
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

/// The whole forward pass, again, from the formulas.
///
/// Returns logits for every position, so a caller can compare any of them.
fn reference(t: &Tiny, tokens: &[u32]) -> Vec<Vec<f32>> {
    let w = &t.tensors;
    let get = |n: &str| w.get(n).unwrap_or_else(|| panic!("no tensor {n}"));
    let embed = get("model.language_model.embed_tokens.weight");

    let key_dim = t.n_k_head * t.k_head;
    let value_dim = t.n_v_head * t.v_head;
    let conv_dim = key_dim * 2 + value_dim;
    let group = t.n_v_head / t.n_k_head;

    // Per linear layer: the convolution window, and the state matrices.
    let mut conv_state: Vec<Vec<f32>> = vec![vec![0.0; conv_dim * (t.conv - 1)]; t.n_layer];
    let mut s_state: Vec<Vec<f32>> =
        vec![vec![0.0; t.n_v_head * t.k_head * t.v_head]; t.n_layer];
    // Per full layer: every key and value so far.
    let mut keys: Vec<Vec<Vec<f32>>> = vec![Vec::new(); t.n_layer];
    let mut vals: Vec<Vec<Vec<f32>>> = vec![Vec::new(); t.n_layer];

    let mut all = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut x = embed.row(tok as usize).to_vec();

        for l in 0..t.n_layer {
            let p = format!("model.language_model.layers.{l}");
            let h = rms_norm(&x, get(&format!("{p}.input_layernorm.weight")).row(0), t.eps);
            let linear = (l + 1) % 4 != 0;

            let mixed = if linear {
                let q = format!("{p}.linear_attn");
                let mut qkv = get(&format!("{q}.in_proj_qkv.weight")).apply(&h);
                // depthwise causal convolution, then SiLU
                let filters = get(&format!("{q}.conv1d.weight"));
                let win = &mut conv_state[l];
                let k = t.conv;
                let mut conved = vec![0.0f32; conv_dim];
                for c in 0..conv_dim {
                    let f = filters.row(c);
                    let past = &win[c * (k - 1)..(c + 1) * (k - 1)];
                    let mut acc = f[k - 1] * qkv[c];
                    for j in 0..k - 1 {
                        acc += f[j] * past[j];
                    }
                    conved[c] = silu(acc);
                }
                for c in 0..conv_dim {
                    let past = &mut win[c * (k - 1)..(c + 1) * (k - 1)];
                    for j in 0..k - 2 {
                        past[j] = past[j + 1];
                    }
                    past[k - 2] = qkv[c];
                }
                qkv = conved;

                let z = get(&format!("{q}.in_proj_z.weight")).apply(&h);
                let b = get(&format!("{q}.in_proj_b.weight")).apply(&h);
                let a = get(&format!("{q}.in_proj_a.weight")).apply(&h);
                let dt = get(&format!("{q}.dt_bias")).row(0).to_vec();
                let a_log = get(&format!("{q}.A_log")).row(0).to_vec();
                let nw = get(&format!("{q}.norm.weight")).row(0).to_vec();

                let mut out = vec![0.0f32; value_dim];
                for hh in 0..t.n_v_head {
                    let kh = hh / group;
                    let qh = l2(&qkv[kh * t.k_head..(kh + 1) * t.k_head]);
                    let qh: Vec<f32> =
                        qh.iter().map(|v| v / (t.k_head as f32).sqrt()).collect();
                    let kk = l2(&qkv[key_dim + kh * t.k_head..key_dim + (kh + 1) * t.k_head]);
                    let vv =
                        &qkv[2 * key_dim + hh * t.v_head..2 * key_dim + (hh + 1) * t.v_head];

                    let beta = sigmoid(b[hh]);
                    let sp = (1.0 + (a[hh] + dt[hh]).exp()).ln();
                    let decay = (-a_log[hh].exp() * sp).exp();

                    let per = t.k_head * t.v_head;
                    let s = &mut s_state[l][hh * per..(hh + 1) * per];
                    for v in s.iter_mut() {
                        *v *= decay;
                    }
                    let mut mem = vec![0.0f32; t.v_head];
                    for dk in 0..t.k_head {
                        for dv in 0..t.v_head {
                            mem[dv] += s[dk * t.v_head + dv] * kk[dk];
                        }
                    }
                    for dk in 0..t.k_head {
                        for dv in 0..t.v_head {
                            s[dk * t.v_head + dv] += kk[dk] * (vv[dv] - mem[dv]) * beta;
                        }
                    }
                    for dk in 0..t.k_head {
                        for dv in 0..t.v_head {
                            out[hh * t.v_head + dv] += s[dk * t.v_head + dv] * qh[dk];
                        }
                    }
                }
                // gated norm, per head
                let mut gated = vec![0.0f32; value_dim];
                for hh in 0..t.n_v_head {
                    let slice = &out[hh * t.v_head..(hh + 1) * t.v_head];
                    let n = rms_norm_gated(slice, &nw, t.eps);
                    for dv in 0..t.v_head {
                        gated[hh * t.v_head + dv] = n[dv] * silu(z[hh * t.v_head + dv]);
                    }
                }
                get(&format!("{q}.out_proj.weight")).apply(&gated)
            } else {
                let q = format!("{p}.self_attn");
                let qg = get(&format!("{q}.q_proj.weight")).apply(&h);
                let qn = get(&format!("{q}.q_norm.weight")).row(0).to_vec();
                let kn = get(&format!("{q}.k_norm.weight")).row(0).to_vec();

                let mut qh = Vec::new();
                let mut gate = Vec::new();
                for head in 0..t.n_head {
                    let s = &qg[head * t.head_dim * 2..(head + 1) * t.head_dim * 2];
                    let mut qq = rms_norm(&s[..t.head_dim], &qn, t.eps);
                    rope(&mut qq, pos, t.rope_dim, t.theta);
                    qh.push(qq);
                    gate.extend_from_slice(&s[t.head_dim..]);
                }
                let kraw = get(&format!("{q}.k_proj.weight")).apply(&h);
                let vraw = get(&format!("{q}.v_proj.weight")).apply(&h);
                let mut krow = Vec::new();
                for head in 0..t.n_kv {
                    let mut kk =
                        rms_norm(&kraw[head * t.head_dim..(head + 1) * t.head_dim], &kn, t.eps);
                    rope(&mut kk, pos, t.rope_dim, t.theta);
                    krow.extend_from_slice(&kk);
                }
                keys[l].push(krow);
                vals[l].push(vraw);

                let scale = 1.0 / (t.head_dim as f32).sqrt();
                let group_q = t.n_head / t.n_kv;
                let mut attended = vec![0.0f32; t.n_head * t.head_dim];
                for head in 0..t.n_head {
                    let kv = head / group_q;
                    let mut scores: Vec<f32> = keys[l]
                        .iter()
                        .map(|row| {
                            let kk = &row[kv * t.head_dim..(kv + 1) * t.head_dim];
                            qh[head].iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (i, s) in scores.iter().enumerate() {
                        let vv = &vals[l][i][kv * t.head_dim..(kv + 1) * t.head_dim];
                        for d in 0..t.head_dim {
                            attended[head * t.head_dim + d] += (s / sum) * vv[d];
                        }
                    }
                }
                for (o, g) in attended.iter_mut().zip(&gate) {
                    *o *= sigmoid(*g);
                }
                get(&format!("{q}.o_proj.weight")).apply(&attended)
            };

            for (xi, m) in x.iter_mut().zip(&mixed) {
                *xi += m;
            }

            let h = rms_norm(&x, get(&format!("{p}.post_attention_layernorm.weight")).row(0), t.eps);
            let g = get(&format!("{p}.mlp.gate_proj.weight")).apply(&h);
            let u = get(&format!("{p}.mlp.up_proj.weight")).apply(&h);
            let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
            let down = get(&format!("{p}.mlp.down_proj.weight")).apply(&act);
            for (xi, m) in x.iter_mut().zip(&down) {
                *xi += m;
            }
        }

        let h = rms_norm(&x, get("model.language_model.norm.weight").row(0), t.eps);
        all.push(get("lm_head.weight").apply(&h));
    }
    all
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let scale = a.iter().map(|x| x.abs()).fold(1f32, f32::max);
    assert!(worst < 2e-4 * scale, "{what}: worst disagreement {worst} (scale {scale})");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The recurrence, against the recurrence.
#[test]
fn the_gated_delta_net_agrees_with_the_formulas() {
    let t = tiny();
    let tokens = [3u32, 17, 8, 31, 4];
    let want = reference(&t, &tokens);

    let (model, spec) = engine(&t, "agree");
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch(&tokens, &mut cache);
    close(&got, want.last().unwrap(), "logits after five tokens");
}

/// Three layers in four hold a state and no keys; the fourth holds keys and no
/// state. If that is the wrong way round the model still runs.
#[test]
fn the_cache_holds_a_state_for_linear_layers_and_rows_for_full_ones() {
    let t = tiny();
    let (_, spec) = engine(&t, "layout");

    let key_dim = t.n_k_head * t.k_head;
    let conv_dim = key_dim * 2 + t.n_v_head * t.v_head;
    let state = conv_dim * (t.conv - 1) + t.n_v_head * t.k_head * t.v_head;

    for l in 0..t.n_layer {
        let shape = spec.cache.at(l);
        match (l + 1) % 4 == 0 {
            true => {
                assert_eq!(shape.k, t.n_kv * t.head_dim, "layer {l} is full attention");
                assert_eq!(shape.state, 0, "layer {l} should hold no state");
            }
            false => {
                assert_eq!(shape.state, state, "layer {l} is linear");
                assert_eq!(shape.k + shape.v, 0, "layer {l} should cache no rows");
            }
        }
    }
    assert!(spec.cache.any_recurrent());
}

/// The cost of the linear layers does not move with context, and the cost of
/// the full ones does. That is the entire argument for the architecture, as a
/// number.
#[test]
fn only_the_attention_layers_get_more_expensive_with_context() {
    let t = tiny();
    let (_, spec) = engine(&t, "bytes");
    let at = |n: usize| spec.cache.bytes(t.n_layer, n);

    let growth = at(100) - at(99);
    let full_layers = t.n_layer / 4;
    let per_token = full_layers * 2 * t.n_kv * t.head_dim * 4;
    assert_eq!(growth, per_token, "only the full-attention layers should grow");
    assert!(at(0) > 0, "the linear layers cost something before any token arrives");
}

/// The issue's own done-when: a prompt fed whole must equal the same prompt
/// fed in two pieces.
///
/// For an attention model this is nearly a tautology — the cache holds rows
/// and rows do not care how they arrived. Here the only thing connecting the
/// two calls is the recurrent state, so this is the test that the state is
/// actually carried and actually correct.
#[test]
fn a_prompt_in_two_pieces_is_the_same_prompt() {
    let t = tiny();
    let tokens = [5u32, 12, 30, 7, 19, 2, 44];

    let (whole, spec) = engine(&t, "whole");
    let mut cache = KvCache::new(&spec);
    let want = whole.forward_batch(&tokens, &mut cache);

    let (split, spec2) = engine(&t, "split");
    let mut cache2 = KvCache::new(&spec2);
    split.forward_batch(&tokens[..3], &mut cache2);
    let got = split.forward_batch(&tokens[3..], &mut cache2);

    close(&got, &want, "seven tokens at once against three then four");
}

/// Every position, not just the last: a state that decays with the wrong sign
/// can still agree at position zero.
#[test]
fn every_position_agrees_and_not_only_the_last() {
    let t = tiny();
    let tokens = [9u32, 1, 25, 13];
    let want = reference(&t, &tokens);

    let (model, spec) = engine(&t, "each");
    let mut cache = KvCache::new(&spec);
    for (i, &tok) in tokens.iter().enumerate() {
        let got = model.forward(tok, &mut cache);
        close(&got, &want[i], &format!("position {i}"));
    }
    assert_eq!(cache.len, tokens.len());
}

/// The vocabulary comes back whole, which catches a head read at the wrong
/// width before any of the above can be trusted.
#[test]
fn the_output_head_is_the_full_vocabulary() {
    let t = tiny();
    let (model, spec) = engine(&t, "vocab");
    let mut cache = KvCache::new(&spec);
    assert_eq!(model.forward(1, &mut cache).len(), t.vocab);
    assert_eq!(spec.n_embd, t.hidden);
}

/// Write this model and this engine's logits out, for `scripts/check-qwen3-5.py`
/// to run the real implementation against.
///
/// Ignored because it needs somewhere to write, and because the thing that
/// reads it needs PyTorch. The Rust reference above catches algebra mistakes;
/// it cannot catch a *misreading*, because a wrong understanding of the
/// reference sits in both versions equally and they agree all the way to the
/// wrong answer. Only the published implementation settles that.
#[test]
#[ignore = "writes a fixture for the Python reference; needs a directory to write to"]
fn write_a_fixture_for_the_reference_implementation() {
    let Ok(dir) = std::env::var("KVAD_QWEN35_FIXTURE") else {
        panic!("set KVAD_QWEN35_FIXTURE to a directory to write");
    };
    // `KVAD_QWEN35_LAYERS=l,f` writes a model with exactly those layers, so a
    // disagreement can be pinned on one mixer instead of the pair.
    let spelled = std::env::var("KVAD_QWEN35_LAYERS").unwrap_or_default();
    let pattern: Vec<&str> = spelled
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| match s.trim() {
            "l" => "linear_attention",
            "f" => "full_attention",
            other => panic!("KVAD_QWEN35_LAYERS takes `l` and `f`, not {other:?}"),
        })
        .collect();
    let t = match pattern.is_empty() {
        true => tiny(),
        false => tiny_with(Some(&pattern)),
    };
    let out = std::path::Path::new(&dir);
    std::fs::create_dir_all(out).unwrap();
    write_safetensors(&out.join("model.safetensors"), &t.tensors);
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&t.config).unwrap())
        .unwrap();

    let tokens: Vec<u32> = vec![3, 17, 8, 0, 29, 11, 5];
    let (model, spec) = engine(&t, "fixture");
    let mut cache = KvCache::new(&spec);
    let logits = model.forward_batch_all(&tokens, &mut cache);
    let rows: Vec<Vec<f32>> = logits.chunks(t.vocab).map(<[f32]>::to_vec).collect();
    std::fs::write(
        out.join("logits.json"),
        serde_json::to_string(&serde_json::json!({"tokens": tokens, "logits": rows})).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out.display());
}

/// Qwen3.8-27B's own numbers, from its published `config.json`.
///
/// Not the whole file — the shape of it, so that the arithmetic this README
/// quotes is checked rather than asserted. A recurrent state is a fixed cost
/// and the claim that it is *157 MB whatever the context* is the entire
/// argument for the architecture; a claim like that should fail a test when it
/// stops being true.
#[test]
fn the_published_27b_has_the_state_and_the_cache_we_say_it_does() {
    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    });
    let spec = Spec::from_config(Json::new(config)).unwrap();

    // Read from `text_config`, a level down from where every other family
    // keeps it.
    assert_eq!(spec.n_layer, 64);
    assert_eq!(spec.n_embd, 5120);
    assert_eq!(spec.head_dim, 256);
    // `rope_theta` is nested inside `rope_parameters` here, and defaulting it
    // to 10000 would be a model that runs and is wrong.
    assert_eq!(spec.rope_theta, 10_000_000.0);

    // Sixteen layers cache, forty-eight do not.
    let full = (0..64).filter(|&i| spec.cache.at(i).k > 0).count();
    let linear = (0..64).filter(|&i| spec.cache.at(i).state > 0).count();
    assert_eq!((full, linear), (16, 48));
    assert!(spec.cache.any_recurrent());

    // The state: 48 layers of a [48, 128, 128] matrix plus a 10240-channel
    // convolution window three deep.
    let per_layer = 10240 * 3 + 48 * 128 * 128;
    let fixed = 48 * per_layer * 4;
    assert_eq!(fixed, 156_893_184, "the state is not the 157 MB we claim");
    assert_eq!(spec.cache.bytes(64, 0), fixed, "at zero tokens, the state is the whole cost");

    // And it does not move with context, while the sixteen attention layers do.
    let grew = spec.cache.bytes(64, 1000) - spec.cache.bytes(64, 0);
    assert_eq!(grew, 16 * 2 * 4 * 256 * 1000 * 4);
    assert_eq!(spec.cache.bytes_per_token(64), 16 * 2 * 4 * 256 * 4);
}
