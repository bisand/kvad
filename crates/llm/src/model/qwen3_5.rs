//! Qwen3.5 and Qwen3.8: three layers in four keep a state instead of a cache.
//!
//! Read this as a diff against [`super::llama`]. The skeleton is unchanged —
//! embed, then per layer `x = x + Mix(RMSNorm(x))` and
//! `x = x + SwiGLU(RMSNorm(x))`, then a final norm and the output head — and
//! the feed-forward half is the same dense SwiGLU. What changed is the *mixer*,
//! and it changed on only three quarters of the layers.
//!
//! # Two kinds of layer
//!
//! `layer_types` says which is which, one entry per layer, and Qwen3.8's reads
//! `linear, linear, linear, full` sixteen times over. The full-attention layers
//! are the grouped-query attention this engine already had, with two additions
//! noted below. The other forty-eight are a **gated delta net**, and they hold
//! no keys and no values at all.
//!
//! # What a gated delta net does
//!
//! Attention keeps every past key and value and looks at all of them. A linear
//! layer keeps one matrix `S` of `key_dim × value_dim` per head — an
//! associative memory — and reads it with the query:
//!
//! ```text
//!     S  <-  S · e^g                       decay what is already there
//!     δ  =  (v − Sᵀk) · β                  how wrong S is about this key
//!     S  <-  S + k ⊗ δ                     write the correction
//!     y  =  Sᵀq                            read it back
//! ```
//!
//! That is the *delta rule*: rather than appending `k ⊗ v` and hoping, it
//! first asks what `S` already returns for this key and stores only the
//! difference, scaled by a learned per-head `β`. `g` is a learned decay, and
//! both are projections of the token — `β = σ(W_b x)` and
//! `g = −e^{A} · softplus(W_a x + b)` — so the layer chooses per token how
//! much to forget and how hard to write.
//!
//! The cost is constant. `S` is the same size at position one and position two
//! hundred thousand, which is the whole argument: 48 layers of Qwen3.8 hold
//! 157 MB of state whatever the context length, where 64 layers of ordinary
//! attention would be paying per token for all of it.
//!
//! The price is that **the state cannot be rewound**. See
//! [`KvCache::truncate`](super::KvCache::truncate), which is where this engine
//! records what it does about that.
//!
//! # The other pieces
//!
//! A **short causal convolution** runs in front of the recurrence — depthwise,
//! kernel 4, over the concatenated q, k and v — so each of them sees a little
//! local context before the state does. Its window lives in the cache too.
//!
//! The full-attention layers add an **output gate**: `q_proj` is twice as wide
//! as it needs to be and the second half gates the attention output through a
//! sigmoid. And their RoPE is **partial** — `partial_rotary_factor` 0.25, so
//! only the first 64 of each 256-wide head is rotated and the rest carries no
//! position at all.
//!
//! # What is not here
//!
//! The vision tower, and the multi-token-prediction head. The checkpoint is a
//! `Qwen3_5ForConditionalGeneration` and carries both; this reads
//! `text_config` and skips the rest by name, because *deliberately not read*
//! and *forgotten* look identical from outside a loader.

use super::ffn::{Ffn, Mlp};
use super::{attend, Architecture, CacheLayout, CacheShape, KvCache, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::{rms_norm, Rope};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Qwen3.5 / Qwen3.8: hybrid linear attention.
pub static ARCH: Architecture = Architecture {
    id: "qwen3_5",
    model_types: &["qwen3_5"],
    about: "Qwen3.5/3.8: gated delta net on three layers in four, gated attention on the fourth",
    configure,
    load: |src, spec| Ok(Box::new(Model::load(src, spec)?)),
};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// Which mixer a layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Linear,
    Full,
}

/// The dimensions a gated delta net has and attention does not.
#[derive(Debug, Clone, Copy)]
pub struct Delta {
    pub n_k_head: usize,
    pub n_v_head: usize,
    pub k_head: usize,
    pub v_head: usize,
    /// Width of the depthwise causal convolution in front of the recurrence.
    pub conv: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// `key_dim * 2 + value_dim`: what the convolution runs over.
    pub conv_dim: usize,
}

impl Delta {
    pub fn read(config: &super::Json) -> Res<Self> {
        let n_k_head = config.need("linear_num_key_heads")?;
        let n_v_head = config.need("linear_num_value_heads")?;
        let k_head = config.need("linear_key_head_dim")?;
        let v_head = config.need("linear_value_head_dim")?;
        if n_v_head % n_k_head != 0 {
            return Err(format!(
                "linear_num_value_heads ({n_v_head}) is not a multiple of \
                 linear_num_key_heads ({n_k_head}); the query and key heads are \
                 repeated to meet the value heads and cannot be"
            )
            .into());
        }
        let key_dim = n_k_head * k_head;
        let value_dim = n_v_head * v_head;
        Ok(Delta {
            n_k_head,
            n_v_head,
            k_head,
            v_head,
            conv: config.num(&["linear_conv_kernel_dim"]).unwrap_or(4),
            key_dim,
            value_dim,
            conv_dim: key_dim * 2 + value_dim,
        })
    }

    /// How many query and key heads one value head shares.
    pub fn group(&self) -> usize {
        self.n_v_head / self.n_k_head
    }

    /// Floats this layer keeps between tokens: the convolution's window, then
    /// the recurrent state.
    ///
    /// One allocation holding both, because a layer's state is one thing as
    /// far as the cache is concerned and splitting it would mean two parallel
    /// vectors indexed the same way.
    pub fn state_len(&self) -> usize {
        self.conv_window() + self.n_v_head * self.k_head * self.v_head
    }

    /// The convolution needs the previous `kernel - 1` inputs per channel.
    pub fn conv_window(&self) -> usize {
        self.conv_dim * (self.conv - 1)
    }
}

/// Which layers are linear and which are full attention.
///
/// Read from `layer_types` when the config states it, which Qwen3.8 does for
/// all sixty-four. `full_attention_interval` is the rule behind that list and
/// is the fallback for a config that gives only the rule.
pub fn layer_kinds(config: &super::Json, n_layer: usize) -> Res<Vec<Kind>> {
    if let Some(list) = config.get("layer_types").and_then(|v| v.as_array()) {
        if list.len() != n_layer {
            return Err(format!(
                "config: `layer_types` has {} entries for {n_layer} layers",
                list.len()
            )
            .into());
        }
        return list
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(Kind::Linear),
                Some("full_attention") => Ok(Kind::Full),
                other => Err(format!("config: unknown layer type {other:?}").into()),
            })
            .collect();
    }
    // The rule, for a config that states it rather than the list. Counting
    // from one: layer 3 is the first full one at an interval of 4, which is
    // what the published `layer_types` says.
    let every = config.num(&["full_attention_interval"]).unwrap_or(1).max(1);
    Ok((0..n_layer)
        .map(|i| match (i + 1) % every == 0 {
            true => Kind::Full,
            false => Kind::Linear,
        })
        .collect())
}

/// Everything the shared [`Spec`] cannot work out by itself.
fn configure(spec: &mut Spec) -> Res<()> {
    let config = spec.config.clone();
    let delta = Delta::read(&config)?;
    let kinds = layer_kinds(&config, spec.n_layer)?;

    // `rope_theta` is nested here, where every other family has it at the top.
    if let Some(rope) = config.get("rope_parameters") {
        if let Some(theta) = rope.get("rope_theta").and_then(|v| v.as_f64()) {
            spec.rope_theta = theta as f32;
        }
    }

    // A cache row for the attention layers, a state for the linear ones, and
    // nothing of the other's on either.
    let kv = spec.n_kv_head * spec.head_dim;
    spec.cache = CacheLayout::per_layer(
        kinds
            .iter()
            .map(|k| match k {
                Kind::Full => CacheShape::kv(kv, kv),
                Kind::Linear => CacheShape::recurrent(delta.state_len()),
            })
            .collect(),
    );
    Ok(())
}

/// How much of a head RoPE touches.
///
/// `partial_rotary_factor` 0.25 over a 256-wide head means the first 64
/// coordinates rotate and the remaining 192 carry no position at all. Getting
/// this wrong produces a model that runs and is wrong, which is why it is
/// read rather than assumed.
pub fn rope_dim(spec: &Spec) -> usize {
    let factor = spec
        .config
        .get("rope_parameters")
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .or_else(|| spec.config.float(&["partial_rotary_factor"]).map(f64::from))
        .unwrap_or(1.0);
    // Even, because RoPE pairs coordinates.
    ((spec.head_dim as f64 * factor) as usize) / 2 * 2
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// One of this family's plain RMSNorm weights, with the one folded in.
///
/// Qwen3.5 scales by `1 + w` and not by `w`: the stored vector is an *offset
/// from one*, initialised to zeros, which is Gemma's convention and not the
/// one every other architecture here uses. Adding the one at load keeps the
/// hot path a plain [`rms_norm`] and puts the difference in the single place
/// a reader goes looking for it.
///
/// `linear_attn.norm` is deliberately **not** read through this. The gated
/// norm in the same reference file scales by `w` directly and is initialised
/// to ones — two conventions in one checkpoint, and getting them the wrong
/// way round produces a model that runs and is wrong.
fn offset_norm(src: &dyn Source, name: &str) -> Res<Vec<f32>> {
    let mut v = src.vector(name)?;
    for x in v.iter_mut() {
        *x += 1.0;
    }
    Ok(v)
}

/// A gated delta net's weights.
struct DeltaNet {
    /// `[conv_dim, hidden]`: query, key and value in one projection, because
    /// one convolution runs over all three.
    in_qkv: Weight,
    /// `[value_dim, hidden]`: the gate the output norm is multiplied by.
    in_z: Weight,
    /// `[n_v_head, hidden]` each: the delta rule's write strength and the
    /// state's decay, one number per head per token.
    in_b: Weight,
    in_a: Weight,
    /// `[conv_dim, kernel]`, depthwise: one filter per channel, no mixing.
    conv: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    /// `[v_head]`: the gated RMSNorm's scale, shared across heads.
    norm: Vec<f32>,
    out: Weight,
}

/// A full-attention layer's weights.
///
/// The Llama block's, plus the output gate folded into `q`.
struct Attn {
    /// `[n_head * head_dim * 2, hidden]`. The second half of each head's slice
    /// is the gate, not more query.
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

enum Mixer {
    Linear(Box<DeltaNet>),
    Full(Box<Attn>),
}

struct Block {
    attn_norm: Vec<f32>,
    mixer: Mixer,
    mlp_norm: Vec<f32>,
    mlp: Mlp,
}

pub struct Model {
    spec: Spec,
    delta: Delta,
    kinds: Vec<Kind>,
    embed: Weight,
    lm_head: Option<Weight>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    rope: Rope,
    rope_dim: usize,
}

impl Model {
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let delta = Delta::read(&spec.config)?;
        let kinds = layer_kinds(&spec.config, spec.n_layer)?;
        let (hd, n_head, n_kv) = (spec.head_dim, spec.n_head, spec.n_kv_head);

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for (i, kind) in kinds.iter().enumerate() {
            // One level deeper than every other family: the text model is a
            // part of a multimodal wrapper, and the checkpoint says so.
            let p = |s: &str| format!("language_model.layers.{i}.{s}");
            let mixer = match kind {
                Kind::Full => Mixer::Full(Box::new(Attn {
                    q: src.matrix(&p("self_attn.q_proj.weight"))?,
                    k: src.matrix(&p("self_attn.k_proj.weight"))?,
                    v: src.matrix(&p("self_attn.v_proj.weight"))?,
                    o: src.matrix(&p("self_attn.o_proj.weight"))?,
                    q_norm: offset_norm(src, &p("self_attn.q_norm.weight"))?,
                    k_norm: offset_norm(src, &p("self_attn.k_norm.weight"))?,
                })),
                Kind::Linear => Mixer::Linear(Box::new(DeltaNet {
                    in_qkv: src.matrix(&p("linear_attn.in_proj_qkv.weight"))?,
                    in_z: src.matrix(&p("linear_attn.in_proj_z.weight"))?,
                    in_b: src.matrix(&p("linear_attn.in_proj_b.weight"))?,
                    in_a: src.matrix(&p("linear_attn.in_proj_a.weight"))?,
                    // Stored `[conv_dim, 1, kernel]` — depthwise, so the
                    // middle axis is 1 and the flat read is the filters back
                    // to back.
                    conv: src.vector(&p("linear_attn.conv1d.weight"))?,
                    dt_bias: src.vector(&p("linear_attn.dt_bias"))?,
                    a_log: src.vector(&p("linear_attn.A_log"))?,
                    norm: src.vector(&p("linear_attn.norm.weight"))?,
                    out: src.matrix(&p("linear_attn.out_proj.weight"))?,
                })),
            };
            blocks.push(Block {
                attn_norm: offset_norm(src, &p("input_layernorm.weight"))?,
                mixer,
                mlp_norm: offset_norm(src, &p("post_attention_layernorm.weight"))?,
                mlp: Mlp::Dense(Ffn::load(src, &p("mlp"))?),
            });
        }

        // The two halves of this checkpoint that are deliberately not run.
        // Saying so is not decoration: the check that follows a load would
        // otherwise refuse every Qwen3.8 for carrying tensors nobody wanted,
        // and a reader cannot tell "skipped" from "forgotten" without it.
        src.skip_under("visual.");
        src.skip_under("model.visual.");
        src.skip_under("mtp.");

        let lm_head = head(src, &spec, "lm_head.weight")?;
        let rope_dim = rope_dim(&spec);
        let rope = Rope::new(rope_dim, spec.n_ctx, spec.rope_theta);

        let _ = (hd, n_head, n_kv);
        Ok(Model {
            embed: src.matrix("language_model.embed_tokens.weight")?,
            lm_head,
            blocks,
            final_norm: offset_norm(src, "language_model.norm.weight")?,
            delta,
            kinds,
            rope,
            rope_dim,
            spec,
        })
    }

    /// One head's vector, L2-normalised. The delta rule wants unit keys and
    /// queries so that `Sᵀk` is a read and not a rescaling.
    fn l2norm(v: &mut [f32]) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    /// The gated delta net, for one token.
    ///
    /// Reads and writes `state`, which holds the convolution's window followed
    /// by the recurrent matrices — see [`Delta::state_len`].
    fn linear_step(&self, net: &DeltaNet, h: &[f32], state: &mut [f32]) -> Vec<f32> {
        let d = &self.delta;
        let mut qkv = net.in_qkv.matvec_bt(h, None);

        // ---- the depthwise causal convolution ---------------------------
        //
        // `y[c] = Σ_j w[c][j] · x[c][t − (K−1) + j]`, so the window is the
        // previous `K−1` inputs and this one. The window is then shifted by
        // one, which is the whole of the convolution's memory.
        let k = d.conv;
        let window = d.conv_window();
        {
            let conv_state = &mut state[..window];
            for c in 0..d.conv_dim {
                let past = &mut conv_state[c * (k - 1)..(c + 1) * (k - 1)];
                let filter = &net.conv[c * k..(c + 1) * k];
                let mut acc = filter[k - 1] * qkv[c];
                for j in 0..k - 1 {
                    acc += filter[j] * past[j];
                }
                // Shift this channel's window while the input is still in
                // hand: drop the oldest, admit this token's *input*. Writing
                // the output back into `qkv[c]` first would memorise the
                // convolution's own answer, and every later token would then
                // be convolved with the wrong history — a model that runs and
                // drifts rather than one that fails.
                for j in 0..k.saturating_sub(2) {
                    past[j] = past[j + 1];
                }
                if k >= 2 {
                    past[k - 2] = qkv[c];
                }
                qkv[c] = acc;
            }
        }
        // SiLU, which the reference passes to the convolution as its
        // `activation` and applies to the output.
        for x in qkv.iter_mut() {
            *x = *x / (1.0 + (-*x).exp());
        }

        let (q_all, rest) = qkv.split_at(d.key_dim);
        let (k_all, v_all) = rest.split_at(d.key_dim);

        let z = net.in_z.matvec_bt(h, None);
        let b = net.in_b.matvec_bt(h, None);
        let a = net.in_a.matvec_bt(h, None);

        let recurrent = &mut state[window..];
        let per_head = d.k_head * d.v_head;
        let scale = 1.0 / (d.k_head as f32).sqrt();
        let mut out = vec![0.0f32; d.value_dim];

        for hh in 0..d.n_v_head {
            // Query and key heads are shared: `group` value heads read the
            // same one, which is `repeat_interleave` in the reference.
            let kh = hh / d.group();
            let mut q: Vec<f32> = q_all[kh * d.k_head..(kh + 1) * d.k_head].to_vec();
            let mut kk: Vec<f32> = k_all[kh * d.k_head..(kh + 1) * d.k_head].to_vec();
            Self::l2norm(&mut q);
            Self::l2norm(&mut kk);
            for x in q.iter_mut() {
                *x *= scale;
            }
            let v = &v_all[hh * d.v_head..(hh + 1) * d.v_head];

            let beta = 1.0 / (1.0 + (-b[hh]).exp());
            // `softplus`, written so a large argument does not overflow the
            // exponential on its way to being ~x.
            let t = a[hh] + net.dt_bias[hh];
            let softplus = if t > 20.0 { t } else { (1.0 + t.exp()).ln() };
            let decay = (-net.a_log[hh].exp() * softplus).exp();

            let s = &mut recurrent[hh * per_head..(hh + 1) * per_head];
            for x in s.iter_mut() {
                *x *= decay;
            }
            // What the state already returns for this key.
            let mut mem = vec![0.0f32; d.v_head];
            for (dk, &kv) in kk.iter().enumerate() {
                if kv == 0.0 {
                    continue;
                }
                let row = &s[dk * d.v_head..(dk + 1) * d.v_head];
                for (m, &r) in mem.iter_mut().zip(row) {
                    *m += r * kv;
                }
            }
            // Store only the difference, scaled by this token's write strength.
            for (dk, &kv) in kk.iter().enumerate() {
                let row = &mut s[dk * d.v_head..(dk + 1) * d.v_head];
                for dv in 0..d.v_head {
                    row[dv] += kv * (v[dv] - mem[dv]) * beta;
                }
            }
            // And read it back with the query.
            let o = &mut out[hh * d.v_head..(hh + 1) * d.v_head];
            for (dk, &qv) in q.iter().enumerate() {
                if qv == 0.0 {
                    continue;
                }
                let row = &s[dk * d.v_head..(dk + 1) * d.v_head];
                for (oo, &r) in o.iter_mut().zip(row) {
                    *oo += r * qv;
                }
            }
        }

        // Gated RMSNorm, per head, then the gate — normalise first, scale by
        // the learned weight, and only then multiply by `silu(z)`.
        for hh in 0..d.n_v_head {
            let slice = &mut out[hh * d.v_head..(hh + 1) * d.v_head];
            let normed = rms_norm(slice, &net.norm, self.spec.eps);
            let gate = &z[hh * d.v_head..(hh + 1) * d.v_head];
            for (o, (n, g)) in slice.iter_mut().zip(normed.iter().zip(gate)) {
                *o = n * (g / (1.0 + (-g).exp()));
            }
        }
        net.out.matvec_bt(&out, None)
    }

    /// A full-attention layer, for one token: the Llama block with a gate on
    /// the output and RoPE over only part of each head.
    fn full_step(&self, attn: &Attn, h: &[f32], layer: usize, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let (hd, n_head) = (spec.head_dim, spec.n_head);
        let kv_dim = spec.kv_dim();
        let pos = cache.len;

        // `q_proj` is twice as wide as the queries: each head's slice is the
        // query followed by its gate.
        let qg = attn.q.matvec_bt(h, None);
        let mut q = vec![0.0f32; n_head * hd];
        let mut gate = vec![0.0f32; n_head * hd];
        for head in 0..n_head {
            let src = &qg[head * hd * 2..(head + 1) * hd * 2];
            q[head * hd..(head + 1) * hd].copy_from_slice(&src[..hd]);
            gate[head * hd..(head + 1) * hd].copy_from_slice(&src[hd..]);
        }
        let mut k = attn.k.matvec_bt(h, None);
        let v = attn.v.matvec_bt(h, None);

        for head in q.chunks_mut(hd) {
            head.copy_from_slice(&rms_norm(head, &attn.q_norm, spec.eps));
            self.rope.apply(&mut head[..self.rope_dim], pos);
        }
        for head in k.chunks_mut(hd) {
            head.copy_from_slice(&rms_norm(head, &attn.k_norm, spec.eps));
            self.rope.apply(&mut head[..self.rope_dim], pos);
        }

        cache.push(layer, &k, &v);
        let _ = kv_dim;
        let attended = attend(spec, &q, cache.keys(layer), cache.values(layer), pos + 1);

        // The gate, which is what `attn_output_gate` names.
        let mut gated = attended;
        for (o, g) in gated.iter_mut().zip(&gate) {
            *o *= 1.0 / (1.0 + (-g).exp());
        }
        attn.o.matvec_bt(&gated, None)
    }
}

impl Transformer for Model {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn param_count(&self) -> usize {
        let opt = |v: &Option<Weight>| v.as_ref().map_or(0, |w| w.param_count());
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let mixer = match &b.mixer {
                    Mixer::Full(a) => {
                        a.q.param_count()
                            + a.k.param_count()
                            + a.v.param_count()
                            + a.o.param_count()
                            + a.q_norm.len()
                            + a.k_norm.len()
                    }
                    Mixer::Linear(n) => {
                        n.in_qkv.param_count()
                            + n.in_z.param_count()
                            + n.in_b.param_count()
                            + n.in_a.param_count()
                            + n.conv.len()
                            + n.dt_bias.len()
                            + n.a_log.len()
                            + n.norm.len()
                            + n.out.param_count()
                    }
                };
                mixer + b.mlp.param_count() + b.attn_norm.len() + b.mlp_norm.len()
            })
            .sum();
        blocks + self.embed.param_count() + opt(&self.lm_head) + self.final_norm.len()
    }

    fn memory_bytes(&self) -> usize {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let mixer = match &b.mixer {
                    Mixer::Full(a) => a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes(),
                    Mixer::Linear(n) => {
                        n.in_qkv.bytes()
                            + n.in_z.bytes()
                            + n.in_b.bytes()
                            + n.in_a.bytes()
                            + n.out.bytes()
                    }
                };
                mixer + b.mlp.bytes()
            })
            .sum();
        self.embed.bytes() + self.lm_head.as_ref().map_or(0, |h| h.bytes()) + blocks
    }

    /// One token.
    ///
    /// There is no batched prefill here, and the reason is the architecture
    /// rather than an omission. A recurrent layer's state at position `t`
    /// depends on its state at `t − 1`, so a batch cannot be one matmul the
    /// way attention's can — the reference implementation has a chunked
    /// formulation for exactly this, and writing it is worth its own session
    /// with a model to check against. Until then the default
    /// [`Transformer::forward_batch`] loop is correct and slow, and slow is
    /// the half that can be fixed later.
    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let e = spec.n_embd;
        let mut x = self.embed.row(token as usize).to_vec();

        for (l, block) in self.blocks.iter().enumerate() {
            let h = rms_norm(&x, &block.attn_norm, spec.eps);
            let mixed = match &block.mixer {
                Mixer::Full(attn) => self.full_step(attn, &h, l, cache),
                Mixer::Linear(net) => self.linear_step(net, &h, cache.state_mut(l)),
            };
            for (xi, m) in x.iter_mut().zip(mixed.iter()) {
                *xi += m;
            }

            let h = rms_norm(&x, &block.mlp_norm, spec.eps);
            let mlp = block.mlp.run_one(&h, e);
            for (xi, m) in x.iter_mut().zip(mlp.iter()) {
                *xi += m;
            }
        }
        cache.len += 1;

        let h = rms_norm(&x, &self.final_norm, spec.eps);
        match &self.lm_head {
            Some(w) => w.matvec_bt(&h, None),
            None => self.embed.matvec_bt(&h, None),
        }
    }
}

/// Which layers are which, for anything that needs to know from outside.
impl Model {
    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }
}
