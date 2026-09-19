//! Model architectures, and the parts they have in common.
//!
//! # What is actually shared
//!
//! GPT-2 (2019) and the Llama family (2023 onwards) look different in every
//! sub-component, and yet:
//!
//! ```text
//!   x = embed(token)
//!   repeat n_layer times:
//!       x = x + Attention(Norm(x))
//!       x = x + MLP(Norm(x))
//!   logits = Norm(x) @ embedding_matrixᵀ
//! ```
//!
//! is still exactly the shape of both. The residual stream, the alternation of
//! "tokens talk to each other" and "each token thinks alone", the final
//! projection back to vocabulary — none of it moved. What changed was the
//! *contents* of `Norm`, `MLP`, and how position gets into `Attention`.
//!
//! So this module holds the skeleton — [`Spec`], [`KvCache`], [`attend`] — and
//! [`gpt2`] and [`llama`] supply the rest. Reading `gpt2.rs` first is
//! recommended: it is the simpler of the two, and `llama.rs` is written to be
//! read as a diff against it.

pub mod gpt2;
pub mod llama;

use crate::tensor::softmax_inplace;
use crate::weights::read_json;
use rayon::prelude::*;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Gpt2,
    /// Llama and everything shaped like it: Llama 2/3, Mistral, Qwen2/2.5,
    /// SmolLM2, TinyLlama. The differences between those are configuration,
    /// not code.
    Llama,
}

impl Arch {
    /// Map a HuggingFace `model_type` (or `architectures[0]`) onto an
    /// implementation, or `None` if we cannot run it.
    ///
    /// Used both when loading a checkpoint and when searching the Hub, so the
    /// search can say up front which results are actually runnable.
    pub fn from_model_type(model_type: &str) -> Option<Arch> {
        let t = model_type.to_ascii_lowercase();
        match t.as_str() {
            t if t.contains("gpt2") => Some(Arch::Gpt2),
            // These all share one implementation. If you hit an unsupported
            // model_type, checking whether it is Llama-shaped is usually a
            // matter of looking for rms_norm_eps and rope_theta in its config.
            t if t.contains("llama")
                || t.contains("qwen2")
                || t.contains("mistral")
                || t.contains("smollm") =>
            {
                Some(Arch::Llama)
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Arch::Gpt2 => "gpt2",
            Arch::Llama => "llama",
        })
    }
}

/// Everything the runtime needs to know about a model, normalised across
/// architectures so the rest of the program does not have to branch.
#[derive(Debug, Clone)]
pub struct Spec {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_head: usize,
    /// Number of *key/value* heads. Equal to `n_head` for ordinary multi-head
    /// attention; smaller under grouped-query attention.
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub head_dim: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    /// Width of the MLP's hidden layer.
    ///
    /// The hand-written loader never needed this — it reads each matrix's
    /// shape straight from the checkpoint. A framework that allocates tensors
    /// up front has to be told.
    pub intermediate: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub tie_embeddings: bool,
}

impl Spec {
    /// Width of one position's worth of cached keys (and of values).
    ///
    /// This is the number that grouped-query attention shrinks. Qwen2.5-0.5B
    /// has 14 query heads but only 2 KV heads, so its cache is one seventh the
    /// size it would otherwise be — which at 32k context is the difference
    /// between 3.4 GB and 0.5 GB.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    /// How many query heads share each KV head.
    pub fn group_size(&self) -> usize {
        self.n_head / self.n_kv_head
    }

    pub fn from_json(path: &Path) -> Res<Self> {
        let v = read_json(path)?;
        let num = |keys: &[&str]| -> Option<usize> {
            keys.iter().find_map(|k| v.get(*k)?.as_u64()).map(|n| n as usize)
        };
        let float = |keys: &[&str]| -> Option<f32> {
            keys.iter().find_map(|k| v.get(*k)?.as_f64()).map(|n| n as f32)
        };

        // `model_type` is the reliable discriminator; `architectures` is a
        // fallback for the handful of configs that omit it.
        let model_type = v
            .get("model_type")
            .and_then(|m| m.as_str())
            .map(str::to_ascii_lowercase)
            .or_else(|| {
                v.get("architectures")?
                    .as_array()?
                    .first()?
                    .as_str()
                    .map(str::to_ascii_lowercase)
            })
            .ok_or("config.json has neither model_type nor architectures")?;

        let arch = Arch::from_model_type(&model_type).ok_or_else(|| {
            format!(
                "unsupported architecture `{model_type}`.\n\
                 This engine implements two: gpt2 and llama (which covers \
                 Llama 2/3, Mistral, Qwen2/2.5, SmolLM2, TinyLlama)."
            )
        })?;

        let n_embd = num(&["n_embd", "hidden_size"]).ok_or("config: no hidden size")?;
        let n_head = num(&["n_head", "num_attention_heads"]).ok_or("config: no head count")?;
        // Llama 3.2 states head_dim explicitly; everyone else implies it.
        let head_dim = num(&["head_dim"]).unwrap_or(n_embd / n_head);

        Ok(Spec {
            arch,
            n_layer: num(&["n_layer", "num_hidden_layers"]).ok_or("config: no layer count")?,
            n_head,
            // Absent means ordinary multi-head attention.
            n_kv_head: num(&["num_key_value_heads"]).unwrap_or(n_head),
            n_embd,
            head_dim,
            n_ctx: num(&["n_positions", "n_ctx", "max_position_embeddings"]).unwrap_or(1024),
            vocab_size: num(&["vocab_size"]).ok_or("config: no vocab_size")?,
            // GPT-2 does not state it; its MLP widens by 4x by construction.
            intermediate: num(&["intermediate_size", "n_inner"]).unwrap_or(4 * n_embd),
            eps: float(&["layer_norm_epsilon", "rms_norm_eps"]).unwrap_or(1e-5),
            rope_theta: float(&["rope_theta"]).unwrap_or(10000.0),
            tie_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|t| t.as_bool())
                // GPT-2 ties unconditionally and does not say so in its config.
                .unwrap_or(arch == Arch::Gpt2),
        })
    }

    pub fn summary(&self) -> String {
        let gqa = if self.n_kv_head == self.n_head {
            String::new()
        } else {
            format!(" ({} KV heads, {}x grouped)", self.n_kv_head, self.group_size())
        };
        format!(
            "{} · {} layers · {} heads{} · {} embd · {} ctx · {} vocab",
            self.arch, self.n_layer, self.n_head, gqa, self.n_embd, self.n_ctx, self.vocab_size
        )
    }
}

/// The interface the runtime talks to. Two implementations, one shape.
pub trait Transformer: Send + Sync {
    fn spec(&self) -> &Spec;
    /// Run one token and return logits over the vocabulary.
    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32>;

    /// Run several tokens in one pass, returning logits for the **last** one.
    ///
    /// This is prefill. Feeding the prompt a token at a time re-reads every
    /// weight matrix once per token; feeding them together reads each weight
    /// once and reuses it across the batch, which is the difference between a
    /// memory-bound and a compute-bound operation.
    ///
    /// Causal masking comes for free. All the batch's keys and values go into
    /// the cache first, then query `i` attends over exactly `pos0 + i + 1`
    /// positions — so it never sees a token that comes after it, without an
    /// explicit mask anywhere.
    ///
    /// The default is the equivalent loop, which is also what the batched
    /// implementations are tested against.
    fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.forward(t, cache);
        }
        logits
    }
    fn param_count(&self) -> usize;
    /// Bytes the weight matrices occupy in memory, after quantisation.
    fn memory_bytes(&self) -> usize;
}

/// Per-layer key and value history.
///
/// Without this, generating token N means recomputing the whole prefix — O(N²)
/// work across a sequence instead of O(N). It is not an optimisation you add
/// later; it is why inference is tractable at all, and why memory use climbs
/// as you fill the context window.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
    pub len: usize,
}

impl KvCache {
    pub fn new(spec: &Spec) -> Self {
        // Deliberately not pre-allocated: a 32k-context model would reserve
        // gigabytes up front for a conversation that may run to fifty tokens.
        KvCache {
            k: (0..spec.n_layer).map(|_| Vec::new()).collect(),
            v: (0..spec.n_layer).map(|_| Vec::new()).collect(),
            kv_dim: spec.kv_dim(),
            len: 0,
        }
    }

    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.k[layer].extend_from_slice(k);
        self.v[layer].extend_from_slice(v);
    }

    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }

    pub fn values(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    /// Drop everything after `len` positions.
    ///
    /// This is what makes *prefix caching* possible. Two turns of a
    /// conversation share a long common prefix — the entire history — so
    /// rather than rebuilding the cache from scratch each turn, keep the part
    /// that still matches and recompute only the tail. On a long chat this is
    /// the difference between re-reading the whole transcript every time and
    /// processing just the new message.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            k.truncate(len * self.kv_dim);
            v.truncate(len * self.kv_dim);
        }
        self.len = len;
    }

    pub fn clear(&mut self) {
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            k.clear();
            v.clear();
        }
        self.len = 0;
    }

    /// Bytes currently held.
    pub fn bytes(&self) -> usize {
        2 * self.len * self.kv_dim * self.k.len() * std::mem::size_of::<f32>()
    }

    /// Bytes this cache would hold at full context.
    pub fn max_bytes(spec: &Spec) -> usize {
        2 * spec.n_layer * spec.n_ctx * spec.kv_dim() * std::mem::size_of::<f32>()
    }
}

/// Multi-head causal self-attention for a single query position.
///
/// Shared by both architectures, because attention itself never changed. For
/// each head: score this token's query against the key of every token so far,
/// softmax those scores into weights, and return the correspondingly weighted
/// average of the values.
///
/// "Causal" is free here — the cache only holds earlier positions, so there is
/// nothing in the future to mask. (Processing a whole prompt in one pass *does*
/// need an explicit triangular mask, which is the usual first bug.)
///
/// # Grouped-query attention
///
/// When `n_kv_head < n_head`, several query heads read the same cached key and
/// value. That is the whole trick: the expensive thing to store is the KV
/// cache, the cheap thing is queries, so keep all the query heads and share the
/// keys. Quality barely moves; memory drops by the group factor.
pub fn attend(spec: &Spec, q: &[f32], k_cache: &[f32], v_cache: &[f32], n_positions: usize) -> Vec<f32> {
    let hd = spec.head_dim;
    let kv_dim = spec.kv_dim();
    let group = spec.group_size();
    // Divide by sqrt(head_dim) before the softmax. The dot product of two
    // random vectors of dimension d has standard deviation sqrt(d); left
    // unscaled, the softmax saturates towards one-hot as d grows and gradients
    // vanish. This constant is the "scaled" in "scaled dot-product attention".
    let scale = 1.0 / (hd as f32).sqrt();

    let heads: Vec<Vec<f32>> = (0..spec.n_head)
        .into_par_iter()
        .map(|head| {
            let q_head = &q[head * hd..(head + 1) * hd];
            // Which KV head this query head reads from.
            let kv_off = (head / group) * hd;

            let mut scores = Vec::with_capacity(n_positions);
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

            let mut out = vec![0.0f32; hd];
            for (t, &w) in scores.iter().enumerate() {
                if w < 1e-8 {
                    continue;
                }
                let base = t * kv_dim + kv_off;
                let v_head = &v_cache[base..base + hd];
                for i in 0..hd {
                    out[i] += w * v_head[i];
                }
            }
            out
        })
        .collect();

    heads.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec() -> Spec {
        Spec {
            arch: Arch::Llama,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 2,
            n_embd: 8,
            head_dim: 2,
            n_ctx: 16,
            vocab_size: 32,
            intermediate: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
            tie_embeddings: true,
        }
    }

    #[test]
    fn grouped_query_attention_shrinks_the_cache() {
        let spec = test_spec();
        // 4 query heads over 2 KV heads: each KV head serves two queries, so
        // only half as much has to be stored per position.
        assert_eq!(spec.group_size(), 2);
        assert_eq!(spec.kv_dim(), 4);
        assert_eq!(spec.kv_dim() * 2, spec.n_head * spec.head_dim);
    }

    #[test]
    fn cache_truncation_keeps_the_prefix_intact() {
        let spec = test_spec();
        let mut cache = KvCache::new(&spec);
        for pos in 0..5 {
            let k: Vec<f32> = (0..spec.kv_dim()).map(|i| (pos * 10 + i) as f32).collect();
            cache.push(0, &k, &k);
            cache.push(1, &k, &k);
            cache.len += 1;
        }
        assert_eq!(cache.len, 5);
        assert_eq!(cache.keys(0).len(), 5 * spec.kv_dim());

        cache.truncate(3);
        assert_eq!(cache.len, 3);
        assert_eq!(cache.keys(0).len(), 3 * spec.kv_dim());
        assert_eq!(cache.keys(1).len(), 3 * spec.kv_dim());
        // Position 2 must still hold exactly what it held before.
        assert_eq!(cache.keys(0)[2 * spec.kv_dim()], 20.0);

        // Truncating upwards is a no-op, not an extension.
        cache.truncate(99);
        assert_eq!(cache.len, 3);
    }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// A loaded model plus its KV cache, behind one interface.
///
/// [`Transformer`] deliberately keeps the cache outside the model, because on
/// the CPU it is a plain `Vec<f32>` the caller can own. A GPU backend cannot
/// work that way — its cache lives in device memory and must never round-trip
/// through the host between layers. So the cache moves inside, and everything
/// above this line (prefix reuse, sampling, streaming) stops caring which
/// backend it is talking to.
pub trait Session: Send {
    fn spec(&self) -> &Spec;
    /// Run `tokens` and return logits for the **last** one.
    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>>;
    /// Tokens currently held in the cache.
    fn cached(&self) -> usize;
    /// Drop everything after `len` positions, for reuse across chat turns.
    fn truncate(&mut self, len: usize) -> Res<()>;
    /// Short description of where this runs, e.g. `cpu q8` or `metal bf16`.
    fn label(&self) -> String;
    fn param_count(&self) -> usize;
    fn weight_bytes(&self) -> usize;
}

/// The hand-written engine, as a [`Session`].
pub struct CpuSession {
    model: Box<dyn Transformer>,
    cache: KvCache,
    precision: crate::quant::Precision,
    /// Prompt tokens per batched prefill pass.
    chunk: usize,
}

impl CpuSession {
    pub fn new(model: Box<dyn Transformer>, precision: crate::quant::Precision) -> Self {
        let cache = KvCache::new(model.spec());
        // Tunable so the effect of batching stays measurable; 1 gives the old
        // token-at-a-time behaviour.
        let chunk = std::env::var("LLM_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(64);
        CpuSession { model, cache, precision, chunk }
    }
}

impl Session for CpuSession {
    fn spec(&self) -> &Spec {
        self.model.spec()
    }

    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let mut logits = Vec::new();
        for part in tokens.chunks(self.chunk) {
            logits = self.model.forward_batch(part, &mut self.cache);
        }
        Ok(logits)
    }

    fn cached(&self) -> usize {
        self.cache.len
    }

    fn truncate(&mut self, len: usize) -> Res<()> {
        self.cache.truncate(len);
        Ok(())
    }

    fn label(&self) -> String {
        format!("cpu {}", self.precision)
    }

    fn param_count(&self) -> usize {
        self.model.param_count()
    }

    fn weight_bytes(&self) -> usize {
        self.model.memory_bytes()
    }
}
