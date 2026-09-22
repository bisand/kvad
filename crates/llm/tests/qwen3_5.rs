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

use kvad::model::ffn::{Layout, Router, Shared};
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
        Mat { rows, cols, data, vector: false }
    }

    /// A plain norm's weight, which this family stores as an offset from one
    /// and trains from zero.
    fn offsets(&mut self, n: usize) -> Mat {
        let data = (0..n).map(|_| self.next() * 0.2).collect();
        Mat { rows: 1, cols: n, data, vector: true }
    }

    /// The gated norm's, and the delta net's scalars, which are scales rather
    /// than offsets and sit near one.
    fn ones(&mut self, n: usize) -> Mat {
        let data = (0..n).map(|_| 1.0 + self.next() * 0.1).collect();
        Mat { rows: 1, cols: n, data, vector: true }
    }
}

#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    /// Whether the checkpoint stores this as a 1-D tensor.
    ///
    /// Not the same question as `rows == 1`. Norms and biases are genuinely
    /// one-dimensional; `shared_expert_gate.weight` is a matrix that happens
    /// to have one row, and the published checkpoints store it as `[1,
    /// hidden]`. Writing it flat would hand the loaders a shape the Hub never
    /// ships.
    vector: bool,
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
        let shape = match m.vector {
            true => format!("[{}]", m.cols),
            false => format!("[{},{}]", m.rows, m.cols),
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
    /// Where the decoder lives: `model.language_model.` for Qwen3.5, which
    /// wraps its text model in a multimodal shell, and `model.` for
    /// Qwen3-Next, which has nothing to wrap it in.
    root: String,
    /// Whether the delta net's inputs are written fused, one key head at a
    /// time, as Qwen3-Next writes them.
    fused: bool,
    /// `(n_experts, top_k, expert_width, shared_width)`, or `None` for a dense
    /// feed-forward.
    moe: Option<(usize, usize, usize, usize)>,
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
        root: "model.language_model.".into(),
        fused: false,
        moe: None,
    }
}

/// Qwen3-Next, at the same widths.
///
/// The same mixer under a different spelling — two fused input projections
/// instead of four, laid out one key head at a time — and a mixture where
/// Qwen3.8 has one MLP. Eight experts of two, with a gated shared one, which
/// is the published shape at a size a test can hold.
fn tiny_next() -> Tiny {
    let mut t = tiny();
    let (n_experts, top_k, expert, shared) = (8usize, 2usize, 32usize, 32usize);
    let (hidden, key_dim, value_dim) = (t.hidden, t.n_k_head * t.k_head, t.n_v_head * t.v_head);
    let group = t.n_v_head / t.n_k_head;

    let mut r = Rng(0xFEED_BEEF_5EED_0001);
    let mut w: BTreeMap<String, Mat> = BTreeMap::new();
    w.insert("model.embed_tokens.weight".into(), r.matrix(t.vocab, hidden));
    w.insert("model.norm.weight".into(), r.offsets(hidden));
    w.insert("lm_head.weight".into(), r.matrix(t.vocab, hidden));

    for l in 0..t.n_layer {
        let p = format!("model.layers.{l}");
        w.insert(format!("{p}.input_layernorm.weight"), r.offsets(hidden));
        w.insert(format!("{p}.post_attention_layernorm.weight"), r.offsets(hidden));

        // Every layer routes: `decoder_sparse_step` 1 and no `mlp_only_layers`,
        // which is what both published Next checkpoints say.
        w.insert(format!("{p}.mlp.gate.weight"), r.matrix(n_experts, hidden));
        for x in 0..n_experts {
            let e = format!("{p}.mlp.experts.{x}");
            w.insert(format!("{e}.gate_proj.weight"), r.matrix(expert, hidden));
            w.insert(format!("{e}.up_proj.weight"), r.matrix(expert, hidden));
            w.insert(format!("{e}.down_proj.weight"), r.matrix(hidden, expert));
        }
        let e = format!("{p}.mlp.shared_expert");
        w.insert(format!("{e}.gate_proj.weight"), r.matrix(shared, hidden));
        w.insert(format!("{e}.up_proj.weight"), r.matrix(shared, hidden));
        w.insert(format!("{e}.down_proj.weight"), r.matrix(hidden, shared));
        w.insert(format!("{p}.mlp.shared_expert_gate.weight"), r.matrix(1, hidden));

        match (l + 1) % 4 == 0 {
            true => {
                let q = format!("{p}.self_attn");
                w.insert(
                    format!("{q}.q_proj.weight"),
                    r.matrix(t.n_head * t.head_dim * 2, hidden),
                );
                w.insert(format!("{q}.k_proj.weight"), r.matrix(t.n_kv * t.head_dim, hidden));
                w.insert(format!("{q}.v_proj.weight"), r.matrix(t.n_kv * t.head_dim, hidden));
                w.insert(format!("{q}.o_proj.weight"), r.matrix(hidden, t.n_head * t.head_dim));
                w.insert(format!("{q}.q_norm.weight"), r.offsets(t.head_dim));
                w.insert(format!("{q}.k_norm.weight"), r.offsets(t.head_dim));
            }
            false => {
                let q = format!("{p}.linear_attn");
                w.insert(
                    format!("{q}.in_proj_qkvz.weight"),
                    r.matrix(key_dim * 2 + value_dim * 2, hidden),
                );
                w.insert(format!("{q}.in_proj_ba.weight"), r.matrix(t.n_v_head * 2, hidden));
                w.insert(format!("{q}.conv1d.weight"), r.matrix(key_dim * 2 + value_dim, t.conv));
                w.insert(format!("{q}.dt_bias"), r.ones(t.n_v_head));
                w.insert(format!("{q}.A_log"), r.ones(t.n_v_head));
                w.insert(format!("{q}.norm.weight"), r.ones(t.v_head));
                w.insert(format!("{q}.out_proj.weight"), r.matrix(hidden, value_dim));
            }
        }
    }
    let _ = group;

    t.config = serde_json::json!({
        "model_type": "qwen3_next",
        "num_hidden_layers": t.n_layer,
        "num_attention_heads": t.n_head,
        "num_key_value_heads": t.n_kv,
        "head_dim": t.head_dim,
        "hidden_size": hidden,
        "intermediate_size": 128,
        "vocab_size": t.vocab,
        "max_position_embeddings": 64,
        "rms_norm_eps": t.eps,
        "rope_theta": t.theta,
        "partial_rotary_factor": 0.5,
        "tie_word_embeddings": false,
        "full_attention_interval": 4,
        "linear_num_key_heads": t.n_k_head,
        "linear_num_value_heads": t.n_v_head,
        "linear_key_head_dim": t.k_head,
        "linear_value_head_dim": t.v_head,
        "linear_conv_kernel_dim": t.conv,
        "num_experts": n_experts,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": expert,
        "shared_expert_intermediate_size": shared,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    });
    t.tensors = w;
    t.root = "model.".into();
    t.fused = true;
    t.moe = Some((n_experts, top_k, expert, shared));
    t
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

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
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
    let embed = get(&format!("{}embed_tokens.weight", t.root));

    // One SwiGLU MLP: a dense layer, one expert, or the shared one.
    let ffn = |prefix: &str, h: &[f32]| -> Vec<f32> {
        let g = get(&format!("{prefix}.gate_proj.weight")).apply(h);
        let u = get(&format!("{prefix}.up_proj.weight")).apply(h);
        let act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        get(&format!("{prefix}.down_proj.weight")).apply(&act)
    };

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
            let p = format!("{}layers.{l}", t.root);
            let h = rms_norm(&x, get(&format!("{p}.input_layernorm.weight")).row(0), t.eps);
            let linear = (l + 1) % 4 != 0;

            let mixed = if linear {
                let q = format!("{p}.linear_attn");
                // Qwen3-Next writes q, k, v and z one *key* head at a time —
                // each key head followed by the value heads that share it —
                // where Qwen3.5 writes four matrices already in head order.
                // Taken apart here from the reference's own split list, not
                // from the engine's.
                let (mut qkv, z, b, a) = match t.fused {
                    false => (
                        get(&format!("{q}.in_proj_qkv.weight")).apply(&h),
                        get(&format!("{q}.in_proj_z.weight")).apply(&h),
                        get(&format!("{q}.in_proj_b.weight")).apply(&h),
                        get(&format!("{q}.in_proj_a.weight")).apply(&h),
                    ),
                    true => {
                        let wide = group * t.v_head;
                        let stride = 2 * t.k_head + 2 * wide;
                        let mixed = get(&format!("{q}.in_proj_qkvz.weight")).apply(&h);
                        let mut qkv = vec![0.0f32; conv_dim];
                        let mut z = vec![0.0f32; value_dim];
                        for g in 0..t.n_k_head {
                            let row = &mixed[g * stride..(g + 1) * stride];
                            let at = |n: usize| g * n..(g + 1) * n;
                            qkv[at(t.k_head)].copy_from_slice(&row[..t.k_head]);
                            let kat = key_dim + g * t.k_head..key_dim + (g + 1) * t.k_head;
                            qkv[kat].copy_from_slice(&row[t.k_head..2 * t.k_head]);
                            let vat = 2 * key_dim + g * wide..2 * key_dim + (g + 1) * wide;
                            qkv[vat].copy_from_slice(&row[2 * t.k_head..2 * t.k_head + wide]);
                            z[at(wide)].copy_from_slice(&row[2 * t.k_head + wide..]);
                        }
                        let mixed = get(&format!("{q}.in_proj_ba.weight")).apply(&h);
                        let mut b = vec![0.0f32; t.n_v_head];
                        let mut a = vec![0.0f32; t.n_v_head];
                        for g in 0..t.n_k_head {
                            let row = &mixed[g * 2 * group..(g + 1) * 2 * group];
                            b[g * group..(g + 1) * group].copy_from_slice(&row[..group]);
                            a[g * group..(g + 1) * group].copy_from_slice(&row[group..]);
                        }
                        (qkv, z, b, a)
                    }
                };
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
            let down = match t.moe {
                None => ffn(&format!("{p}.mlp"), &h),
                // Softmax over every expert, the best `top_k`, those
                // renormalised to sum to one — and a shared expert that runs
                // whatever the router said, scaled by its own one-row gate.
                Some((n_experts, top_k, _, _)) => {
                    let probs = softmax(&get(&format!("{p}.mlp.gate.weight")).apply(&h));
                    let mut order: Vec<usize> = (0..n_experts).collect();
                    order.sort_by(|&i, &j| probs[j].total_cmp(&probs[i]));
                    let chosen = &order[..top_k];
                    let total: f32 = chosen.iter().map(|&i| probs[i]).sum();

                    let gate = get(&format!("{p}.mlp.shared_expert_gate.weight")).apply(&h);
                    let gate = sigmoid(gate[0]);
                    let mut out = ffn(&format!("{p}.mlp.shared_expert"), &h);
                    for o in out.iter_mut() {
                        *o *= gate;
                    }
                    for &i in chosen {
                        let y = ffn(&format!("{p}.mlp.experts.{i}"), &h);
                        let weight = probs[i] / total;
                        for (o, v) in out.iter_mut().zip(&y) {
                            *o += weight * v;
                        }
                    }
                    out
                }
            };
            for (xi, m) in x.iter_mut().zip(&down) {
                *xi += m;
            }
        }

        let h = rms_norm(&x, get(&format!("{}norm.weight", t.root)).row(0), t.eps);
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

/// Qwen3-Next's spelling of the same mixer, against the formulas.
///
/// Two fused input projections instead of four, laid out one key head at a
/// time, and a mixture behind them instead of an MLP. Everything between is
/// the arithmetic the test above already checks, which is the claim this
/// module is built on and therefore the one worth checking twice.
#[test]
fn qwen3_next_agrees_with_the_formulas_too() {
    let t = tiny_next();
    let tokens = [3u32, 17, 8, 31, 4];
    let want = reference(&t, &tokens);

    let (model, spec) = engine(&t, "next-agree");
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch(&tokens, &mut cache);
    close(&got, want.last().unwrap(), "Qwen3-Next logits after five tokens");
}

/// Every position, because an interleaving read the wrong way round can still
/// agree at position zero: the convolution has no history yet and the state is
/// empty, so several of the ways to get this wrong are invisible there.
#[test]
fn every_qwen3_next_position_agrees_and_not_only_the_last() {
    let t = tiny_next();
    let tokens = [9u32, 1, 25, 13];
    let want = reference(&t, &tokens);

    let (model, spec) = engine(&t, "next-each");
    let mut cache = KvCache::new(&spec);
    for (i, &tok) in tokens.iter().enumerate() {
        let got = model.forward(tok, &mut cache);
        close(&got, &want[i], &format!("Qwen3-Next position {i}"));
    }
}

/// The recurrent state carries across calls here too, with a mixture in the
/// way. Worth its own run: the router reads the residual stream, so a state
/// carried wrongly changes *which experts run*, not only by how much.
#[test]
fn a_qwen3_next_prompt_in_two_pieces_is_the_same_prompt() {
    let t = tiny_next();
    let tokens = [5u32, 12, 30, 7, 19, 2, 44];

    let (whole, spec) = engine(&t, "next-whole");
    let mut cache = KvCache::new(&spec);
    let want = whole.forward_batch(&tokens, &mut cache);

    let (split, spec2) = engine(&t, "next-split");
    let mut cache2 = KvCache::new(&spec2);
    split.forward_batch(&tokens[..3], &mut cache2);
    let got = split.forward_batch(&tokens[3..], &mut cache2);

    close(&got, &want, "seven tokens at once against three then four");
}

/// Qwen3-Next is this module's other family, and `Spec` should say so without
/// being told twice.
///
/// The cache is the thing worth pinning: the two families have the same mixer,
/// so they have the same shape of state, and a checkpoint that routed its way
/// into the wrong layer pattern would still load.
#[test]
fn qwen3_next_is_the_same_hybrid_under_its_own_name() {
    let t = tiny_next();
    let (model, spec) = engine(&t, "next-shape");
    assert!(spec.arch.is("qwen3_next"), "loaded as {}", spec.arch.id());

    let key_dim = t.n_k_head * t.k_head;
    let conv_dim = key_dim * 2 + t.n_v_head * t.v_head;
    let state = conv_dim * (t.conv - 1) + t.n_v_head * t.k_head * t.v_head;
    for l in 0..t.n_layer {
        let shape = spec.cache.at(l);
        match (l + 1) % 4 == 0 {
            true => assert_eq!((shape.k, shape.state), (t.n_kv * t.head_dim, 0), "layer {l}"),
            false => assert_eq!((shape.k + shape.v, shape.state), (0, state), "layer {l}"),
        }
    }

    // Eight experts and a shared one per layer, which the dense sibling at the
    // same widths does not have. If the mixture had quietly loaded as one MLP
    // the model would still run.
    let (n_experts, _, expert, shared) = t.moe.unwrap();
    let per_layer = (n_experts * expert + shared) * t.hidden * 3
        + n_experts * t.hidden
        + t.hidden;
    assert!(
        model.param_count() > t.n_layer * per_layer,
        "the experts do not seem to be loaded: {} parameters",
        model.param_count()
    );
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
    // `KVAD_QWEN35_FAMILY=next` writes the Qwen3-Next spelling instead, which
    // the Python side runs through `Qwen3NextForCausalLM`.
    let next = std::env::var("KVAD_QWEN35_FAMILY").is_ok_and(|v| v == "next");
    let t = match (next, pattern.is_empty()) {
        (true, _) => tiny_next(),
        (false, true) => tiny(),
        (false, false) => tiny_with(Some(&pattern)),
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

/// Qwen3-Next-80B-A3B's and Qwen3-Coder-Next's own numbers, which are the same
/// numbers: the two checkpoints differ in `rope_theta` and in whether they
/// carry a multi-token-prediction head, and in nothing else this reads.
///
/// The config states `full_attention_interval` and no `layer_types`, so this
/// is also the test that the rule and the list agree: 48 layers at an interval
/// of four is 36 linear and 12 full, counting from one.
#[test]
fn the_published_80b_is_the_same_hybrid_with_a_mixture_behind_it() {
    let config = serde_json::json!({
        "model_type": "qwen3_next",
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "hidden_size": 2048,
        "intermediate_size": 5120,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "tie_word_embeddings": false,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    });
    let spec = Spec::from_config(Json::new(config.clone())).unwrap();
    assert!(spec.arch.is("qwen3_next"), "loaded as {}", spec.arch.id());
    // Flat, where Qwen3.5 keeps all of this under `text_config`.
    assert_eq!((spec.n_layer, spec.n_embd, spec.head_dim), (48, 2048, 256));
    assert_eq!(spec.rope_theta, 10_000_000.0);
    // A quarter of a 256-wide head rotates; the other 192 carry no position.
    assert_eq!(kvad::model::qwen3_5::rope_dim(&spec), 64);

    let full = (0..48).filter(|&i| spec.cache.at(i).k > 0).count();
    let linear = (0..48).filter(|&i| spec.cache.at(i).state > 0).count();
    assert_eq!((full, linear), (12, 36));

    // The state: 36 layers of a [32, 128, 128] matrix plus an 8192-channel
    // convolution window three deep. Half Qwen3.8's, on a model three times
    // the size, because the state is a property of the mixer and not of the
    // parameter count.
    let fixed = 36 * (8192 * 3 + 32 * 128 * 128) * 4;
    assert_eq!(fixed, 79_036_416);
    assert_eq!(spec.cache.bytes(48, 0), fixed);
    assert_eq!(spec.cache.bytes_per_token(48), 12 * 2 * 2 * 256 * 4);

    // Every layer routes, to ten of five hundred and twelve, and every layer
    // also runs the shared expert whatever the router said.
    let router = Router::read(&spec.config).unwrap();
    let layout = Layout::read(&spec.config);
    assert_eq!((router.n_experts, router.top_k), (512, 10));
    assert!((0..48).all(|i| layout.is_moe(i, router.n_experts)));
    assert_eq!(layout.shared, Shared::Gated(512));
}
