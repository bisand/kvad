//! Qwen3.5, Qwen3.8, Qwen3-Next: three layers in four keep a state, not a cache.
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
//! attention would be paying per token for all of it. Qwen3-Next holds 79 MB
//! on a model three times the size, because the state is a property of the
//! mixer and not of the parameter count.
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
//! # Two families, one mixer
//!
//! `qwen3_next` is here rather than in a file of its own, and that is a claim
//! worth stating plainly: the two published implementations are identical
//! across the convolution, the delta rule, the gated norm and the gated
//! attention — every line of the arithmetic above. They differ in three
//! places and nowhere else.
//!
//! - **Where the decoder lives.** Qwen3.5 is a
//!   `Qwen3_5ForConditionalGeneration` and keeps its text model under
//!   `language_model`, beside a vision tower. Qwen3-Next is text alone.
//! - **How the inputs are spelled.** Qwen3.5 writes four projections in head
//!   order. Qwen3-Next writes two, laid out one *key* head at a time — its
//!   query, its key, then the values and gates of the value heads that share
//!   them. See [`Inputs`].
//! - **What follows the mixer.** Qwen3.8 is dense. Qwen3-Next routes every
//!   layer to ten of five hundred and twelve, with a shared expert that runs
//!   whatever the router said — and, unlike DeepSeek's, one that a token can
//!   decline through a sigmoid gate.
//!
//! [`Family`] is that list, and the only branch either spelling costs the
//! other is one match in the loader and one in the projection.
//!
//! # And a third that adds rather than respells
//!
//! Qwen3.8-Flash-Next (`qwen4_exp`) is Next's mixer and mixture with three
//! things added — four residual streams, hashed n-gram embeddings, and a
//! block selection in front of attention — which [`Streams`] describes. It
//! is here because everything those three wrap is this module's arithmetic
//! unchanged.
//!
//! # What is not here
//!
//! The vision tower, and the multi-token-prediction head. Qwen3.5's checkpoint
//! carries both and Qwen3-Next-80B carries the second; this reads
//! `text_config` where there is one and skips the rest by name, because
//! *deliberately not read* and *forgotten* look identical from outside a
//! loader.

use super::ffn::{Ffn, Layout, Mlp, Moe, Router};
use super::{attend, Architecture, CacheLayout, CacheShape, KvCache, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::{dot, rms_norm, Rope};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Qwen3.5 / Qwen3.8: hybrid linear attention.
pub static ARCH: Architecture = Architecture {
    id: "qwen3_5",
    model_types: &["qwen3_5"],
    about: "Qwen3.5/3.8: gated delta net on three layers in four, gated attention on the fourth",
    configure,
    splits: false,
    load: whole,
};

/// Qwen3-Next, and Qwen3-Coder-Next: the same mixer, with a mixture behind it.
pub static NEXT: Architecture = Architecture {
    id: "qwen3_next",
    model_types: &["qwen3_next"],
    about: "Qwen3-Next/Coder-Next: the same hybrid, 512 experts and a gated shared one",
    configure,
    splits: false,
    load: whole,
};

/// Qwen3.8-Flash-Next: the same again, on four residual streams, with
/// hashed n-gram embeddings and a block selection in front of attention.
/// See [`Streams`], [`Ple`] and [`Indexer`].
pub static EXP: Architecture = Architecture {
    id: "qwen4_exp",
    model_types: &["qwen4_exp"],
    about: "Qwen3.8-Flash-Next: the Next hybrid on four residual streams, n-gram embeddings, top-k attention",
    configure,
    splits: false,
    load: whole,
};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// Which of the two families in this module a checkpoint belongs to.
///
/// They differ in three places and nowhere else: where the decoder lives, how
/// the delta net's inputs are spelled, and whether the feed-forward is one MLP
/// or five hundred. Everything between — the convolution, the delta rule, the
/// gated norm, the gated attention over a quarter-rotated head — is the same
/// arithmetic, and the two reference implementations are identical across all
/// of it, line for line. That is the reason this is an enum in one module and
/// not a second file.
///
/// Public because the GPU backend has to reach the same answer, and two
/// loaders deciding separately which family a checkpoint is is exactly how
/// they come to disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Qwen3.5 and Qwen3.8: a vision tower, a dense feed-forward, and four
    /// separate input projections.
    Qwen35,
    /// Qwen3-Next and Qwen3-Coder-Next: text alone, a mixture, and two.
    Next,
    /// Qwen3.8-Flash-Next (`qwen4_exp`): Qwen3.5's wrapper and four input
    /// projections, Next's mixture with its experts packed, and the three
    /// additions [`Streams`] describes.
    Exp,
}

impl Family {
    pub fn of(spec: &Spec) -> Self {
        match () {
            _ if spec.arch.is(NEXT.id) => Family::Next,
            _ if spec.arch.is(EXP.id) => Family::Exp,
            _ => Family::Qwen35,
        }
    }

    /// Where the decoder lives in the checkpoint.
    ///
    /// Qwen3.5 ships as a `Qwen3_5ForConditionalGeneration`, so its text model
    /// sits a level down under `language_model`, beside the vision tower.
    /// Qwen3-Next is text and nothing else and puts its layers where every
    /// Llama does — which is spelled as nothing at all here, because the
    /// checkpoint reader already tries `model.` in front of whatever it is
    /// asked for.
    pub fn prefix(self) -> &'static str {
        match self {
            Family::Qwen35 | Family::Exp => "language_model.",
            Family::Next => "",
        }
    }
}

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
    /// Whether the output gate is `sigmoid(z)` rather than `silu(z)`.
    /// Qwen3.8-Flash-Next's `output_gate_type` says sigmoid; every earlier
    /// checkpoint gates through its `hidden_act`, which is SiLU.
    pub sigmoid_gate: bool,
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
            sigmoid_gate: match config.text("output_gate_type").or(config.text("hidden_act")) {
                None | Some("silu") => false,
                Some("sigmoid") => true,
                Some(other) => return Err(format!("config: unknown output gate `{other}`").into()),
            },
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
                // Qwen3.8-Flash-Next's config says `full_attention` and its
                // reference renames it `indexed_attention`: the same layer
                // with an [`Indexer`] in front, which [`Streams`] decides.
                Some("full_attention" | "indexed_attention") => Ok(Kind::Full),
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

    // A router that renormalises its top k unless told otherwise, as the
    // reference class does: Qwen3.8-Flash-Next's config leaves
    // `norm_topk_prob` out, and the engine's default for a config that is
    // silent is the other one.
    if spec.arch.is(EXP.id) && config.get("norm_topk_prob").is_none() {
        let mut v = config.value().clone();
        v["norm_topk_prob"] = serde_json::Value::Bool(true);
        spec.config = super::Json::new(v);
    }
    let streams = Streams::read(&config, spec)?;

    // A cache row for the attention layers, a state for the linear ones, and
    // nothing of the other's on either. Qwen3.8-Flash-Next's attention rows
    // carry the indexer's key as well, and its n-gram layers a little more
    // state; see [`Streams::index_dim`] and [`Ple::state_len`].
    let kv = spec.n_kv_head * spec.head_dim;
    let index = streams.as_ref().map_or(0, Streams::index_dim);
    spec.cache = CacheLayout::per_layer(
        kinds
            .iter()
            .enumerate()
            .map(|(i, k)| match k {
                Kind::Full => CacheShape::kv(kv + index, kv),
                Kind::Linear => CacheShape::recurrent(
                    delta.state_len() + streams.as_ref().filter(|s| s.has_ple(i)).map_or(0, |s| s.ple_state_len(spec.n_embd)),
                ),
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
// Qwen3.8-Flash-Next's additions
// ---------------------------------------------------------------------------

/// What Qwen3.8-Flash-Next (`qwen4_exp`) adds to Qwen3-Next, as numbers.
///
/// Three things, read from its reference implementation, and nothing else:
/// the delta net, the gated attention and the mixture with its gated shared
/// expert are Next's, line for line, with the output gate a sigmoid.
///
/// **Four residual streams** (`hc_count`, "hyper-connections"). The state
/// carried from layer to layer is four copies of the hidden vector, not one.
/// The embedding starts all four the same. Before each mixer and each MLP a
/// [`Hyper`] normalises each stream, weighs every number of every stream by
/// a learned low-rank function of all four, and averages them down to the
/// one vector the sublayer reads; the sublayer's output is then added back
/// to each stream scaled by a per-stream weight in `(0, 2)`. That replaces
/// both RMSNorms of every block, and a last [`Hyper`] replaces the final
/// norm. The streams cost 4 × 2560 floats a token and two small matmuls a
/// sublayer — 3.3 M parameters each, against 1.7 G in a layer's experts.
///
/// **Per-layer embeddings** (`ple_layer_ids`, one-indexed — layer 2 in the
/// published config, index 1 here). A table of hashed bigrams and trigrams
/// read on that layer and gated into the streams; see [`Ple`].
///
/// **Query-selected attention** (`indexer_*`). In front of each attention
/// layer a small indexer picks which of the past tokens a query may look at:
/// whole blocks of `indexer_compress_ratio` tokens, the best
/// `indexer_budget / ratio` of them by a cheap score, plus the incomplete
/// block at the end. With the published budget of 2048 that changes nothing
/// until a query sees more than 2051 tokens. See [`Indexer`].
#[derive(Debug, Clone)]
pub struct Streams {
    pub hc: usize,
    /// Zero-based layers carrying a [`Ple`], in the order of `ple_layer_ids`.
    pub ple_layers: Vec<usize>,
    pub ple_dim: usize,
    pub ple_kernel: usize,
    pub ngram: usize,
    pub per_ngram: usize,
    pub ngram_base: u64,
    pub seed: u64,
    pub eos: u32,
    pub vocab: usize,
    /// `(heads, head_dim, budget, ratio)`, when attention is selected.
    pub indexer: Option<(usize, usize, usize, usize)>,
}

impl Streams {
    /// `None` for a checkpoint without hyper-connections, which is every one
    /// but Qwen3.8-Flash-Next.
    pub fn read(config: &super::Json, spec: &Spec) -> Res<Option<Self>> {
        if !spec.arch.is(EXP.id) {
            return Ok(None);
        }
        let hc = config.need("hc_count")?;
        if hc < 2 {
            return Err(format!("config: `hc_count` is {hc}; the reference requires two or more").into());
        }
        let ple_layers: Vec<usize> = config
            .get("ple_layer_ids")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|n| n as usize).collect())
            .unwrap_or_default();
        if let Some(bad) = ple_layers.iter().find(|&&l| l == 0 || l > spec.n_layer) {
            return Err(format!("config: `ple_layer_ids` counts from one, and has {bad}").into());
        }
        let (ngram, per_ngram) = (config.num(&["ngram_size"]).unwrap_or(3), config.num(&["heads_per_ngram"]).unwrap_or(8));
        let ple_dim = config.num(&["ple_embed_dim"]).unwrap_or(spec.n_embd);
        if !ple_layers.is_empty() && (ngram < 2 || ple_dim % ((ngram - 1) * per_ngram) != 0) {
            return Err(format!("config: `ple_embed_dim` {ple_dim} does not divide among {} n-gram heads", (ngram - 1) * per_ngram).into());
        }
        let eos = match config.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => a.first().and_then(|v| v.as_u64()),
            Some(v) => v.as_u64(),
            None => None,
        };
        let indexer = match config.num(&["indexer_n_heads"]) {
            None => None,
            Some(heads) => {
                let (dim, budget, ratio) =
                    (config.need("indexer_head_dim")?, config.need("indexer_budget")?, config.need("indexer_compress_ratio")?);
                if config.num(&["indexer_kv_heads"]) != Some(1) || ratio == 0 || budget % ratio != 0 {
                    return Err("config: the indexer wants one key head and a budget of whole blocks".into());
                }
                if rope_dim(spec) > dim {
                    return Err(format!("config: RoPE turns {} numbers and an index head has {dim}", rope_dim(spec)).into());
                }
                Some((heads, dim, budget, ratio))
            }
        };
        Ok(Some(Streams {
            hc,
            ple_layers: ple_layers.iter().map(|l| l - 1).collect(),
            ple_dim,
            ple_kernel: config.num(&["ple_conv_kernel_size"]).unwrap_or(4),
            ngram,
            per_ngram,
            ngram_base: config.num(&["ngram_vocab_size_base"]).unwrap_or(20_000_000) as u64,
            seed: config.num(&["seed"]).unwrap_or(1234) as u64,
            eos: match (eos, ple_layers.is_empty()) {
                (Some(e), _) => e as u32,
                (None, true) => 0,
                (None, false) => return Err("config: n-gram embeddings need `eos_token_id`".into()),
            },
            vocab: spec.vocab_size,
            indexer,
        }))
    }

    pub fn has_ple(&self, layer: usize) -> bool {
        self.ple_layers.contains(&layer)
    }

    /// The indexer's key, cached beside each attention key.
    pub fn index_dim(&self) -> usize {
        self.indexer.map_or(0, |(_, dim, _, _)| dim)
    }

    /// An n-gram layer's state: the last `ngram − 1` tokens, then its
    /// convolution's window, `(kernel − 1) × ngram` past inputs on each of
    /// `hc × hidden` channels.
    pub fn ple_state_len(&self, e: usize) -> usize {
        (self.ngram - 1) + self.hc * e * self.ple_window()
    }

    /// The convolution's dilation is the n-gram size, so it reaches back
    /// `(kernel − 1) × ngram` tokens.
    pub fn ple_window(&self) -> usize {
        (self.ple_kernel - 1) * self.ngram
    }

    /// The n-gram heads of the `nth` layer in `ple_layer_ids`: how many rows
    /// each has, and where they start in the concatenated table.
    ///
    /// Each head's row count is a different prime — the first ones at or
    /// above `ngram_vocab_size_base`, counted on across heads and layers —
    /// so that two heads hashing the same n-gram collide with different
    /// n-grams. Derived here rather than read from the checkpoint's
    /// `ngram_heads_vocab_sizes`, because that is an int64 buffer and the
    /// cache file keeps floats; the test against a real file checks the
    /// derivation against it.
    pub fn heads(&self, nth: usize) -> Vec<(u64, u64)> {
        let n = (self.ngram - 1) * self.per_ngram;
        let mut p = self.ngram_base.saturating_sub(1);
        for _ in 0..nth * n {
            p = next_prime(p);
        }
        let mut at = 0;
        (0..n)
            .map(|_| {
                p = next_prime(p);
                at += p;
                (p, at - p)
            })
            .collect()
    }

    /// The `nth` n-gram layer's odd hash multipliers, one per position in
    /// the n-gram: SplitMix64 from `seed + 10007 × nth`, reduced so that a
    /// token id times one cannot overflow an `i64`.
    pub fn multipliers(&self, nth: usize) -> Vec<u64> {
        let half = ((i64::MAX as u64) / self.vocab.max(1) as u64 / 2).max(1);
        let base = self.seed.wrapping_add(10007u64.wrapping_mul(nth as u64));
        (0..self.ngram as u64).map(|i| 2 * (splitmix64(base.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half) + 1).collect()
    }
}

impl Streams {
    /// The tokens an n-gram ending here reaches back to, nearest first, from
    /// the last `ngram − 1` seen (nearest first, EOS for before the start).
    ///
    /// An n-gram does not cross an EOS: once one is met, everything further
    /// back reads as EOS too.
    pub fn before(&self, history: &[u32]) -> Vec<u32> {
        let mut cut = false;
        history
            .iter()
            .map(|&t| {
                cut |= t == self.eos;
                if cut { self.eos } else { t }
            })
            .collect()
    }

    /// The table's rows for `token`: one per head, bigram heads first.
    /// `heads` and `mult` are [`Streams::heads`] and [`Streams::multipliers`]
    /// for this layer.
    pub fn rows(&self, heads: &[(u64, u64)], mult: &[u64], token: u32, before: &[u32]) -> Vec<u64> {
        let mut out = Vec::with_capacity(heads.len());
        let mut mixed = token as u64 * mult[0];
        for order in 2..=self.ngram {
            mixed ^= before[order - 2] as u64 * mult[order - 1];
            for &(rows, at) in &heads[(order - 2) * self.per_ngram..(order - 1) * self.per_ngram] {
                out.push(mixed % rows + at);
            }
        }
        out
    }
}

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

/// The first prime above `n`.
fn next_prime(mut n: u64) -> u64 {
    loop {
        n += 1;
        if n >= 2 && (n == 2 || n % 2 == 1) && (3..).step_by(2).take_while(|d| d * d <= n).all(|d| n % d != 0) {
            return n;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// RMSNorm over each `e`-wide group of `x` separately, each with its own
/// slice of `w`: how this family normalises four streams at once.
fn group_norm(x: &[f32], w: &[f32], e: usize, eps: f32) -> Vec<f32> {
    x.chunks(e).zip(w.chunks(e)).flat_map(|(x, w)| rms_norm(x, w, eps)).collect()
}

/// One hyper-connection: four streams in, one vector out for a sublayer,
/// and how much of the sublayer's answer each stream gets back.
///
/// ```text
///     n  = RMSNorm(stream) per stream                       [hc, e]
///     w  = σ(up(silu(down(n) / hc)))                        [hc, e]
///     in = mean over streams of w · n                       [e]
///     g  = 2 σ(inject(n) / hc)                              [hc]
///     stream_s  <-  stream_s + g_s · sublayer(in)
/// ```
///
/// The final one has no `inject`: it only mixes down, for the output head.
struct Hyper {
    norm: Vec<f32>,
    down: Weight,
    up: Weight,
    inject: Option<Weight>,
}

impl Hyper {
    fn load(src: &dyn Source, prefix: &str, combine: bool) -> Res<Self> {
        let n = |s: &str| format!("{prefix}.{s}.weight");
        Ok(Hyper {
            norm: offset_norm(src, &n("hc_norm"))?,
            down: src.matrix(&n("input_mix_weight_down"))?,
            up: src.matrix(&n("input_mix_weight_up"))?,
            inject: if combine { Some(src.matrix(&n("block_inject_weight"))?) } else { None },
        })
    }

    /// `(in, g)`: the sublayer's input, and each stream's share of its output.
    fn mix(&self, x: &[f32], e: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let hc = (x.len() / e) as f32;
        let n = group_norm(x, &self.norm, e, eps);
        let low: Vec<f32> = self.down.matvec_bt(&n, None).into_iter().map(|v| silu(v / hc)).collect();
        let w = self.up.matvec_bt(&low, None);
        let mut mixed = vec![0.0f32; e];
        for (w, n) in w.chunks(e).zip(n.chunks(e)) {
            for (m, (w, n)) in mixed.iter_mut().zip(w.iter().zip(n)) {
                *m += sigmoid(*w) * n;
            }
        }
        mixed.iter_mut().for_each(|m| *m /= hc);
        let share = self
            .inject
            .as_ref()
            .map(|m| m.matvec_bt(&n, None).into_iter().map(|v| 2.0 * sigmoid(v / hc)).collect())
            .unwrap_or_default();
        (mixed, share)
    }

    fn param_count(&self) -> usize {
        self.norm.len() + self.down.param_count() + self.up.param_count() + self.inject.as_ref().map_or(0, Weight::param_count)
    }

    fn bytes(&self) -> usize {
        self.down.bytes() + self.up.bytes() + self.inject.as_ref().map_or(0, Weight::bytes)
    }
}

/// Add `y`, scaled per stream by `share`, to every stream of `x`.
fn inject(x: &mut [f32], y: &[f32], share: &[f32]) {
    for (stream, g) in x.chunks_mut(y.len()).zip(share) {
        for (x, y) in stream.iter_mut().zip(y) {
            *x += g * y;
        }
    }
}

/// Per-layer embeddings: what the last few tokens were, looked up in a
/// hashed table and gated into the streams.
///
/// For each token, the bigram ending at it and the trigram ending at it are
/// hashed eight ways each — `t₀m₀ ⊕ t₁m₁ (⊕ t₂m₂)`, modulo a different prime
/// per head — and the sixteen rows found are concatenated into one
/// `ple_embed_dim` vector. An n-gram never reaches back across an EOS: a
/// position before one, or before the start, reads as EOS.
///
/// That vector is projected to one value and to one key per stream; each
/// stream's normalised state is the query, and their scaled dot product,
/// square-rooted with its sign kept, gates through a sigmoid how much of the
/// value that stream receives. Normalised, it then runs through a depthwise
/// convolution with dilation `ngram_size` — so it sees this token and the
/// ones 3, 6 and 9 back — and the stream gets the gated value plus the
/// convolution's SiLU.
///
/// The table is what makes this layer unusual: 51 G parameters in the
/// published checkpoint, read in place; see [`crate::weights::Table`].
pub struct Ple {
    table: crate::weights::Table,
    /// `(rows, first row)` per head.
    heads: Vec<(u64, u64)>,
    mult: Vec<u64>,
    key: Weight,
    value: Weight,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    /// `[hc × hidden, kernel]`, depthwise.
    conv: Vec<f32>,
}

impl Ple {
    fn load(src: &dyn Source, prefix: &str, streams: &Streams, nth: usize) -> Res<Self> {
        let n = |s: &str| format!("{prefix}.{s}");
        let stem = n("ple_embedding.ngram_embedding");
        let table = crate::weights::Table::open(&src.files(), &format!("model.{stem}"))
            .or_else(|_| crate::weights::Table::open(&src.files(), &stem))?;
        src.skip_under(&format!("{stem}."));
        src.skip_under(&format!("model.{stem}."));
        // The hash's constants are derived (see [`Streams::heads`]); the
        // buffers that restate them are int64 and deliberately not read.
        for b in ["layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"] {
            src.skip(&n(&format!("ple_embedding.{b}")));
        }
        let heads = streams.heads(nth);
        let width = streams.ple_dim / heads.len();
        let end = heads.last().map_or(0, |&(rows, at)| rows + at);
        if table.cols != width || (table.rows as u64) < end {
            return Err(format!(
                "`{stem}` is {}x{}; {} heads of {width} want {end} rows",
                table.rows,
                table.cols,
                heads.len()
            )
            .into());
        }
        Ok(Ple {
            table,
            mult: streams.multipliers(nth),
            heads,
            key: src.matrix(&n("key_proj.weight"))?,
            value: src.matrix(&n("value_proj.weight"))?,
            norm_key: offset_norm(src, &n("norm_key.weight"))?,
            norm_query: offset_norm(src, &n("norm_query.weight"))?,
            norm_conv: offset_norm(src, &n("norm_conv.weight"))?,
            conv: src.vector(&n("conv1d.weight"))?,
        })
    }

    /// What this layer adds to the streams for `token`. `state` holds the
    /// last tokens and the convolution's window; see [`Streams::ple_state_len`].
    fn step(&self, streams: &Streams, x: &[f32], token: u32, state: &mut [f32], eps: f32) -> Vec<f32> {
        let (e, hc, ctx) = (x.len() / streams.hc, streams.hc, streams.ngram - 1);
        let (history, window) = state.split_at_mut(ctx);

        // Tokens are kept as `id + 1`, so a fresh state's zeros read as "before
        // the start", which is EOS. Nearest first.
        let seen: Vec<u32> = history.iter().map(|&h| if h == 0.0 { streams.eos } else { h as u32 - 1 }).collect();
        let before = streams.before(&seen);
        history.copy_within(0..ctx - 1, 1);
        history[0] = (token + 1) as f32;

        let width = self.table.cols;
        let mut emb = vec![0.0f32; self.heads.len() * width];
        for (row, out) in streams.rows(&self.heads, &self.mult, token, &before).into_iter().zip(emb.chunks_mut(width)) {
            self.table.row_into(row as usize, out);
        }

        let key = group_norm(&self.key.matvec_bt(&emb, None), &self.norm_key, e, eps);
        let value = self.value.matvec_bt(&emb, None);
        let query = group_norm(x, &self.norm_query, e, eps);
        let mut gated = vec![0.0f32; hc * e];
        for s in 0..hc {
            let dot: f32 = key[s * e..(s + 1) * e].iter().zip(&query[s * e..(s + 1) * e]).map(|(k, q)| k * q).sum();
            let g = dot / (e as f32).sqrt();
            let g = g.abs().max(1e-6).sqrt() * g.signum() * f32::from(g != 0.0);
            let g = sigmoid(g);
            for (o, v) in gated[s * e..(s + 1) * e].iter_mut().zip(&value) {
                *o = g * v;
            }
        }
        let normed = group_norm(&gated, &self.norm_conv, e, eps);

        // Depthwise, dilated: the input `(kernel − 1 − j) × ngram` tokens back
        // meets filter tap `j`. The window holds the last `(kernel − 1) ×
        // ngram` inputs per channel, oldest first.
        let (k, dil, len) = (streams.ple_kernel, streams.ngram, streams.ple_window());
        for (c, out) in gated.iter_mut().enumerate() {
            let past = &mut window[c * len..(c + 1) * len];
            let filter = &self.conv[c * k..(c + 1) * k];
            let mut acc = filter[k - 1] * normed[c];
            for (j, w) in filter[..k - 1].iter().enumerate() {
                acc += w * past[len - (k - 1 - j) * dil];
            }
            past.copy_within(1.., 0);
            past[len - 1] = normed[c];
            *out += silu(acc);
        }
        gated
    }

    fn param_count(&self) -> usize {
        self.table.rows * self.table.cols
            + self.key.param_count()
            + self.value.param_count()
            + self.norm_key.len()
            + self.norm_query.len()
            + self.norm_conv.len()
            + self.conv.len()
    }

    fn bytes(&self) -> usize {
        self.key.bytes() + self.value.bytes()
    }
}

/// The block selection in front of an attention layer.
///
/// One small projection gives `heads` index queries and a single index key
/// per token, both `dim` wide. The keys are cached raw. For a query that sees
/// `n` tokens, every complete block of `ratio` of them is reduced to one key
/// — the mean of its raw keys, normalised, then rotated to the block's first
/// position — and scored as
///
/// ```text
///     score(block) = Σ_heads relu(q_h · k_block) / √dim
/// ```
///
/// The best `budget / ratio` blocks, and whatever incomplete block is left
/// at the end, are the tokens this query attends to. Up to `budget + ratio −
/// 1` tokens: below that, every token is selected and this is ordinary
/// attention, which is where every prompt under two thousand tokens stays.
struct Indexer {
    qk: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl Indexer {
    /// The positions a query at `n − 1` attends to, given every index key
    /// so far: the `dim` numbers after `offset` in each `stride`-wide row.
    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        iq: &[f32],
        keys: &[f32],
        stride: usize,
        offset: usize,
        dim: usize,
        n: usize,
        budget: usize,
        ratio: usize,
        rope: &Rope,
        rope_dim: usize,
        eps: f32,
    ) -> Vec<usize> {
        let (blocks, keep) = (n / ratio, budget / ratio);
        if blocks <= keep {
            return (0..n).collect();
        }
        let scale = 1.0 / (dim as f32).sqrt();
        let mut scored: Vec<(f32, usize)> = (0..blocks)
            .map(|b| {
                let mut pooled = vec![0.0f32; dim];
                for t in b * ratio..(b + 1) * ratio {
                    for (p, k) in pooled.iter_mut().zip(&keys[t * stride + offset..t * stride + offset + dim]) {
                        *p += k;
                    }
                }
                pooled.iter_mut().for_each(|p| *p /= ratio as f32);
                let mut key = rms_norm(&pooled, &self.k_norm, eps);
                rope.apply(&mut key[..rope_dim], b * ratio);
                let score: f32 = iq.chunks(dim).map(|q| dot(q, &key).max(0.0)).sum();
                (score * scale, b)
            })
            .collect();
        // Best first; a tie goes to the earlier block, which is a choice the
        // reference leaves to `topk` and random weights never reach.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut chosen: Vec<usize> = scored[..keep].iter().flat_map(|&(_, b)| b * ratio..(b + 1) * ratio).collect();
        chosen.extend(blocks * ratio..n);
        chosen
    }

    fn param_count(&self) -> usize {
        self.qk.param_count() + self.q_norm.len() + self.k_norm.len()
    }
}

/// Attention over some of the cached positions rather than all of them, with
/// keys `stride` apart: [`attend`] for a selection, on one thread, written
/// out because the rows here carry an index key the shared kernel does not
/// know to step over.
fn attend_some(spec: &Spec, q: &[f32], keys: &[f32], stride: usize, values: &[f32], chosen: &[usize]) -> Vec<f32> {
    let (hd, group, vstride) = (spec.head_dim, spec.group_size(), spec.kv_dim());
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; spec.n_head * hd];
    for (head, o) in out.chunks_mut(hd).enumerate() {
        let (qh, kv) = (&q[head * hd..(head + 1) * hd], (head / group) * hd);
        let mut w: Vec<f32> = chosen.iter().map(|&t| dot(qh, &keys[t * stride + kv..t * stride + kv + hd]) * scale).collect();
        let top = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        w.iter_mut().for_each(|x| *x = (*x - top).exp());
        let sum: f32 = w.iter().sum();
        for (&t, p) in chosen.iter().zip(&w) {
            for (o, v) in o.iter_mut().zip(&values[t * vstride + kv..t * vstride + kv + hd]) {
                *o += p / sum * v;
            }
        }
    }
    out
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

/// How a checkpoint spells the six things a delta net projects out of the
/// token: query, key, value, the output gate `z`, the write strength `b` and
/// the decay `a`.
///
/// Whichever spelling, this hands back the same four vectors — `q k v`
/// concatenated in head order for the convolution to run over, then `z`, `b`
/// and `a` — and the rest of the layer never learns which it was.
enum Inputs {
    /// Qwen3.5's: one matrix each, each already in head order.
    Split {
        /// `[conv_dim, hidden]`: query, key and value together, because one
        /// convolution runs over all three.
        qkv: Weight,
        /// `[value_dim, hidden]`: the gate the output norm is multiplied by.
        z: Weight,
        /// `[n_v_head, hidden]` each, one number per head per token.
        b: Weight,
        a: Weight,
    },
    /// Qwen3-Next's: two matrices, laid out one *key* head at a time —
    /// `q, k, v, z` for the first key head and the two value heads that share
    /// it, then the second key head's, and `b, a` the same way.
    ///
    /// Taken apart per token rather than permuted once at load. The rows being
    /// permuted belong to a matrix that may be quantised, and slicing those
    /// apart is a kernel this engine does not have; the alternative costs one
    /// pass over twelve thousand floats, against the `[12288, 2048]`
    /// matrix-vector product that just produced them.
    Fused {
        /// `[key_dim * 2 + value_dim * 2, hidden]`.
        qkvz: Weight,
        /// `[n_v_head * 2, hidden]`.
        ba: Weight,
    },
}

impl Inputs {
    fn load(src: &dyn Source, prefix: &str, family: Family) -> Res<Self> {
        let n = |s: &str| format!("{prefix}.{s}.weight");
        Ok(match family {
            Family::Qwen35 | Family::Exp => Inputs::Split {
                qkv: src.matrix(&n("in_proj_qkv"))?,
                z: src.matrix(&n("in_proj_z"))?,
                b: src.matrix(&n("in_proj_b"))?,
                a: src.matrix(&n("in_proj_a"))?,
            },
            Family::Next => Inputs::Fused {
                qkvz: src.matrix(&n("in_proj_qkvz"))?,
                ba: src.matrix(&n("in_proj_ba"))?,
            },
        })
    }

    /// `(qkv, z, b, a)` for one token, in head order whatever the checkpoint
    /// did.
    fn project(&self, h: &[f32], d: &Delta) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        match self {
            Inputs::Split { qkv, z, b, a } => (
                qkv.matvec_bt(h, None),
                z.matvec_bt(h, None),
                b.matvec_bt(h, None),
                a.matvec_bt(h, None),
            ),
            Inputs::Fused { qkvz, ba } => {
                let (group, fused) = (d.group(), qkvz.matvec_bt(h, None));
                // One key head's share: its query, its key, and the values and
                // gates of the `group` value heads that read them.
                let stride = 2 * d.k_head + 2 * group * d.v_head;
                let mut qkv = vec![0.0f32; d.conv_dim];
                let mut z = vec![0.0f32; d.value_dim];
                let (q_all, rest) = qkv.split_at_mut(d.key_dim);
                let (k_all, v_all) = rest.split_at_mut(d.key_dim);
                for kh in 0..d.n_k_head {
                    let src = &fused[kh * stride..(kh + 1) * stride];
                    let (q, src) = src.split_at(d.k_head);
                    let (k, src) = src.split_at(d.k_head);
                    let (v, gate) = src.split_at(group * d.v_head);
                    q_all[kh * d.k_head..(kh + 1) * d.k_head].copy_from_slice(q);
                    k_all[kh * d.k_head..(kh + 1) * d.k_head].copy_from_slice(k);
                    // The value heads a key head owns are consecutive, so its
                    // block lands whole.
                    let (from, to) = (kh * group * d.v_head, (kh + 1) * group * d.v_head);
                    v_all[from..to].copy_from_slice(v);
                    z[from..to].copy_from_slice(gate);
                }

                let fused = ba.matvec_bt(h, None);
                let mut b = vec![0.0f32; d.n_v_head];
                let mut a = vec![0.0f32; d.n_v_head];
                for kh in 0..d.n_k_head {
                    let src = &fused[kh * 2 * group..(kh + 1) * 2 * group];
                    b[kh * group..(kh + 1) * group].copy_from_slice(&src[..group]);
                    a[kh * group..(kh + 1) * group].copy_from_slice(&src[group..]);
                }
                (qkv, z, b, a)
            }
        }
    }

    fn param_count(&self) -> usize {
        match self {
            Inputs::Split { qkv, z, b, a } => {
                qkv.param_count() + z.param_count() + b.param_count() + a.param_count()
            }
            Inputs::Fused { qkvz, ba } => qkvz.param_count() + ba.param_count(),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Inputs::Split { qkv, z, b, a } => qkv.bytes() + z.bytes() + b.bytes() + a.bytes(),
            Inputs::Fused { qkvz, ba } => qkvz.bytes() + ba.bytes(),
        }
    }
}

/// A gated delta net's weights.
struct DeltaNet {
    inputs: Inputs,
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
    /// Qwen3.8-Flash-Next's block selection; `None` everywhere else.
    indexer: Option<Indexer>,
}

enum Mixer {
    Linear(Box<DeltaNet>),
    Full(Box<Attn>),
}

/// How a block's sublayers read and write the residual.
enum Residual {
    /// `x = x + f(RMSNorm(x))`, twice: every family but one.
    Plain { attn_norm: Vec<f32>, mlp_norm: Vec<f32> },
    /// Through four streams; see [`Streams`].
    Hyper { attn: Hyper, mlp: Hyper, ple: Option<Box<Ple>> },
}

struct Block {
    residual: Residual,
    mixer: Mixer,
    mlp: Mlp,
}

pub struct Model {
    spec: Spec,
    delta: Delta,
    kinds: Vec<Kind>,
    embed: Weight,
    lm_head: Option<Weight>,
    blocks: Vec<Block>,
    /// Empty when [`Model::streams`] is set: the last [`Hyper`] is the norm.
    final_norm: Vec<f32>,
    rope: Rope,
    rope_dim: usize,
    /// Qwen3.8-Flash-Next's additions, and the hyper-connection that mixes
    /// its streams down for the output head.
    streams: Option<(Streams, Hyper)>,
}

/// The whole model or nothing. A linear layer's recurrent state is one more
/// thing a shard would hold, and nothing about it is in the way; what is
/// missing is the pass in three pieces, which this architecture has yet to
/// be given.
fn whole(src: &dyn Source, spec: Spec, shard: &super::Shard) -> Res<Box<dyn Transformer>> {
    if !shard.is_whole(&spec) {
        return Err(format!("the `{}` architecture runs whole and does not split yet", spec.arch).into());
    }
    Ok(Box::new(Model::load(src, spec)?))
}

impl Model {
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let delta = Delta::read(&spec.config)?;
        let kinds = layer_kinds(&spec.config, spec.n_layer)?;
        let family = Family::of(&spec);
        let root = family.prefix();

        // Qwen3.8 is dense and Qwen3-Next routes; both of them are this
        // module. `Router::count` is the question that can be asked of either,
        // and the one that says which.
        let routed = match Router::count(&spec.config) {
            None => None,
            Some(_) => Some((Router::read(&spec.config)?, Layout::read(&spec.config))),
        };

        let streams = Streams::read(&spec.config, &spec)?;

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for (i, kind) in kinds.iter().enumerate() {
            let p = |s: &str| format!("{root}layers.{i}.{s}");
            let mixer = match kind {
                Kind::Full => Mixer::Full(Box::new(Attn {
                    q: src.matrix(&p("self_attn.q_proj.weight"))?,
                    k: src.matrix(&p("self_attn.k_proj.weight"))?,
                    v: src.matrix(&p("self_attn.v_proj.weight"))?,
                    o: src.matrix(&p("self_attn.o_proj.weight"))?,
                    q_norm: offset_norm(src, &p("self_attn.q_norm.weight"))?,
                    k_norm: offset_norm(src, &p("self_attn.k_norm.weight"))?,
                    indexer: match streams.as_ref().and_then(|s| s.indexer) {
                        None => None,
                        Some(_) => Some(Indexer {
                            qk: src.matrix(&p("self_attn.indexer.index_qk_proj.weight"))?,
                            q_norm: offset_norm(src, &p("self_attn.indexer.q_layernorm.weight"))?,
                            k_norm: offset_norm(src, &p("self_attn.indexer.k_layernorm.weight"))?,
                        }),
                    },
                })),
                Kind::Linear => Mixer::Linear(Box::new(DeltaNet {
                    inputs: Inputs::load(src, &p("linear_attn"), family)?,
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
            let residual = match &streams {
                None => Residual::Plain {
                    attn_norm: offset_norm(src, &p("input_layernorm.weight"))?,
                    mlp_norm: offset_norm(src, &p("post_attention_layernorm.weight"))?,
                },
                Some(st) => Residual::Hyper {
                    attn: Hyper::load(src, &p("attn_hyper_connection"), true)?,
                    mlp: Hyper::load(src, &p("mlp_hyper_connection"), true)?,
                    ple: match st.ple_layers.iter().position(|&l| l == i) {
                        None => None,
                        Some(nth) => Some(Box::new(Ple::load(src, &p("ple"), st, nth)?)),
                    },
                },
            };
            blocks.push(Block {
                residual,
                mixer,
                mlp: match &routed {
                    Some((router, layout)) if layout.is_moe(i, router.n_experts) => {
                        Mlp::Moe(Box::new(Moe::load(src, &p("mlp"), router, layout.shared)?))
                    }
                    _ => Mlp::Dense(Ffn::load(src, &p("mlp"))?),
                },
            });
        }

        // The parts of this checkpoint that are deliberately not run. Saying
        // so is not decoration: the check that follows a load would otherwise
        // refuse every one of these for carrying tensors nobody wanted, and a
        // reader cannot tell "skipped" from "forgotten" without it.
        //
        // The multi-token-prediction head is a training-time device and both
        // families ship one. The vision tower is Qwen3.5's alone, and naming
        // it under Qwen3-Next would claim to have decided about something that
        // is not there.
        src.skip_under("mtp.");
        if family != Family::Next {
            src.skip_under("visual.");
            src.skip_under("model.visual.");
        }

        let lm_head = head(src, &spec, "lm_head.weight")?;
        let rope_dim = rope_dim(&spec);
        let rope = Rope::new(rope_dim, spec.n_ctx, spec.rope_theta);

        let (final_norm, streams) = match streams {
            None => (offset_norm(src, &format!("{root}norm.weight"))?, None),
            Some(st) => (Vec::new(), Some((st, Hyper::load(src, &format!("{root}hyper_connection_mixer"), false)?))),
        };

        Ok(Model {
            embed: src.matrix(&format!("{root}embed_tokens.weight"))?,
            lm_head,
            blocks,
            final_norm,
            streams,
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
        let (mut qkv, z, b, a) = net.inputs.project(h, d);

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
                *o = n * if d.sigmoid_gate { sigmoid(*g) } else { silu(*g) };
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

        let index = self.streams.as_ref().and_then(|(s, _)| s.indexer);
        let attended = match (&attn.indexer, index) {
            (Some(ix), Some((heads, dim, budget, ratio))) => {
                let qk = ix.qk.matvec_bt(h, None);
                let (iq, ik) = qk.split_at(heads * dim);
                let mut iq = iq.to_vec();
                for head in iq.chunks_mut(dim) {
                    head.copy_from_slice(&rms_norm(head, &ix.q_norm, spec.eps));
                    self.rope.apply(&mut head[..self.rope_dim], pos);
                }
                // The index key rides at the end of the attention key's row.
                let mut row = k;
                row.extend_from_slice(ik);
                cache.push(layer, &row, &v);
                let stride = kv_dim + dim;
                let keys = cache.keys(layer);
                let chosen = ix.select(&iq, keys, stride, kv_dim, dim, pos + 1, budget, ratio, &self.rope, self.rope_dim, spec.eps);
                attend_some(spec, &q, keys, stride, cache.values(layer), &chosen)
            }
            _ => {
                cache.push(layer, &k, &v);
                attend(spec, &q, cache.keys(layer), cache.values(layer), pos + 1)
            }
        };

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
                            + a.indexer.as_ref().map_or(0, Indexer::param_count)
                    }
                    Mixer::Linear(n) => {
                        n.inputs.param_count()
                            + n.conv.len()
                            + n.dt_bias.len()
                            + n.a_log.len()
                            + n.norm.len()
                            + n.out.param_count()
                    }
                };
                let residual = match &b.residual {
                    Residual::Plain { attn_norm, mlp_norm } => attn_norm.len() + mlp_norm.len(),
                    Residual::Hyper { attn, mlp, ple } => {
                        attn.param_count() + mlp.param_count() + ple.as_ref().map_or(0, |p| p.param_count())
                    }
                };
                mixer + b.mlp.param_count() + residual
            })
            .sum();
        let last = self.streams.as_ref().map_or(0, |(_, h)| h.param_count());
        blocks + self.embed.param_count() + opt(&self.lm_head) + self.final_norm.len() + last
    }

    fn memory_bytes(&self) -> usize {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let mixer = match &b.mixer {
                    Mixer::Full(a) => {
                        a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes() + a.indexer.as_ref().map_or(0, |i| i.qk.bytes())
                    }
                    Mixer::Linear(n) => n.inputs.bytes() + n.out.bytes(),
                };
                // The n-gram table is not here: it is read where it lies.
                let residual = match &b.residual {
                    Residual::Plain { .. } => 0,
                    Residual::Hyper { attn, mlp, ple } => attn.bytes() + mlp.bytes() + ple.as_ref().map_or(0, |p| p.bytes()),
                };
                mixer + b.mlp.bytes() + residual
            })
            .sum::<usize>()
            + self.streams.as_ref().map_or(0, |(_, h)| h.bytes());
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
        let row = self.embed.row(token as usize).to_vec();
        // Every stream starts as the embedding; one stream is the plain case.
        let hc = self.streams.as_ref().map_or(1, |(s, _)| s.hc);
        let mut x = row.repeat(hc);
        let state_len = self.delta.state_len();

        for (l, block) in self.blocks.iter().enumerate() {
            let mix = |x: &[f32], h: &Hyper| h.mix(x, e, spec.eps);
            let (h, share) = match &block.residual {
                Residual::Plain { attn_norm, .. } => (rms_norm(&x, attn_norm, spec.eps), vec![1.0]),
                Residual::Hyper { attn, ple, .. } => {
                    if let (Some(ple), Some((st, _))) = (ple, &self.streams) {
                        let add = ple.step(st, &x, token, &mut cache.state_mut(l)[state_len..], spec.eps);
                        x.iter_mut().zip(add).for_each(|(x, a)| *x += a);
                    }
                    mix(&x, attn)
                }
            };
            let mixed = match &block.mixer {
                Mixer::Full(attn) => self.full_step(attn, &h, l, cache),
                Mixer::Linear(net) => self.linear_step(net, &h, &mut cache.state_mut(l)[..state_len]),
            };
            inject(&mut x, &mixed, &share);

            let (h, share) = match &block.residual {
                Residual::Plain { mlp_norm, .. } => (rms_norm(&x, mlp_norm, spec.eps), vec![1.0]),
                Residual::Hyper { mlp, .. } => mix(&x, mlp),
            };
            let out = block.mlp.run_one(&h, e, self.blocks.get(l + 1).map(|b| &b.mlp));
            inject(&mut x, &out, &share);
        }
        cache.len += 1;

        let h = match &self.streams {
            None => rms_norm(&x, &self.final_norm, spec.eps),
            Some((_, last)) => last.mix(&x, e, spec.eps).0,
        };
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
