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
//! the architecture modules supply the rest. Reading `gpt2.rs` first is
//! recommended: it is the simplest of them, and `llama.rs` is written to be
//! read as a diff against it. `deepseek.rs` is a diff against *that*, and is
//! where the skeleton stops being enough.
//!
//! Which of them a build contains is a Cargo feature; see [`arch`], which is
//! also where to look when adding one.

pub mod arch;
#[cfg(feature = "arch-deepseek")]
pub mod deepseek;

/// What a block does after it has attended, shared by every architecture
/// that has experts.
pub mod ffn;
#[cfg(feature = "arch-gpt2")]
pub mod gpt2;
#[cfg(feature = "arch-llama")]
pub mod llama;

pub use arch::{Arch, Architecture};

use crate::tensor::{dot, softmax_inplace};
use crate::weights::read_json;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

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
    /// How wide one position's worth of cache is, in each of the two streams.
    ///
    /// Ordinary attention stores a key and a value, both [`Spec::kv_dim`]
    /// wide, and for every architecture here but one that is what this says.
    /// DeepSeek's latent attention stores something else, in two streams of
    /// different widths, which is why this is a field the architecture sets
    /// rather than a number the skeleton computes.
    pub cache: CacheShape,
    /// The model's `config.json`, as it was read.
    ///
    /// Everything above this line is a field most of the Hub agrees on.
    /// Everything an architecture needs and nobody else has heard of —
    /// `kv_lora_rank`, `n_routed_experts`, `first_k_dense_replace` — is read
    /// from here, by the module that knows what it means.
    pub config: Json,
}

/// The width of one cached row in each of the KV cache's two streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheShape {
    pub k: usize,
    pub v: usize,
}

/// A model's `config.json`, with the lookups every architecture needs.
///
/// Shared behind an `Arc` because [`Spec`] is cloned freely and a config is a
/// few kilobytes of `serde_json` tree that nobody mutates.
#[derive(Clone, Default)]
pub struct Json(Arc<serde_json::Value>);

impl Json {
    pub fn new(v: serde_json::Value) -> Self {
        Json(Arc::new(v))
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    /// The first of `keys` that is present and a number.
    ///
    /// Several keys because the same quantity has different names in
    /// different eras: `n_layer` became `num_hidden_layers`, `n_embd` became
    /// `hidden_size`.
    pub fn num(&self, keys: &[&str]) -> Option<usize> {
        keys.iter().find_map(|k| self.0.get(*k)?.as_u64()).map(|n| n as usize)
    }

    pub fn float(&self, keys: &[&str]) -> Option<f32> {
        keys.iter().find_map(|k| self.0.get(*k)?.as_f64()).map(|n| n as f32)
    }

    pub fn flag(&self, key: &str) -> Option<bool> {
        self.0.get(key)?.as_bool()
    }

    pub fn text(&self, key: &str) -> Option<&str> {
        self.0.get(key)?.as_str()
    }

    /// A number the architecture cannot run without.
    pub fn need(&self, key: &str) -> Res<usize> {
        self.num(&[key]).ok_or_else(|| format!("config: no `{key}`").into())
    }
}

/// Terse on purpose: a `Spec` is printed in logs and in error messages, and
/// the config behind it is a few hundred lines.
impl std::fmt::Debug for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("config.json")
    }
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
        Spec::from_config(Json::new(read_json(path)?))
    }

    /// As [`Spec::from_json`], from a config already in hand.
    ///
    /// The shared fields are read here; then the architecture is handed the
    /// half-built `Spec` to finish, because the rest of the file is written in
    /// its vocabulary and not ours.
    pub fn from_config(config: Json) -> Res<Self> {
        // `model_type` is the reliable discriminator; `architectures` is a
        // fallback for the handful of configs that omit it.
        let model_type = config
            .text("model_type")
            .map(str::to_ascii_lowercase)
            .or_else(|| {
                config.get("architectures")?.as_array()?.first()?.as_str().map(str::to_ascii_lowercase)
            })
            .ok_or("config.json has neither model_type nor architectures")?;

        let arch = Arch::from_model_type(&model_type).ok_or_else(|| {
            format!("unsupported architecture `{model_type}`.\nThis build runs: {}.", arch::supported())
        })?;

        let n_embd = config.num(&["n_embd", "hidden_size"]).ok_or("config: no hidden size")?;
        let n_head = config.num(&["n_head", "num_attention_heads"]).ok_or("config: no head count")?;
        // Llama 3.2 states head_dim explicitly; everyone else implies it.
        let head_dim = config.num(&["head_dim"]).unwrap_or(n_embd / n_head);
        // Absent means ordinary multi-head attention.
        let n_kv_head = config.num(&["num_key_value_heads"]).unwrap_or(n_head);

        let mut spec = Spec {
            arch,
            n_layer: config.num(&["n_layer", "num_hidden_layers"]).ok_or("config: no layer count")?,
            n_head,
            n_kv_head,
            n_embd,
            head_dim,
            n_ctx: config.num(&["n_positions", "n_ctx", "max_position_embeddings"]).unwrap_or(1024),
            vocab_size: config.num(&["vocab_size"]).ok_or("config: no vocab_size")?,
            // GPT-2 does not state it; its MLP widens by 4x by construction.
            intermediate: config.num(&["intermediate_size", "n_inner"]).unwrap_or(4 * n_embd),
            eps: config.float(&["layer_norm_epsilon", "rms_norm_eps"]).unwrap_or(1e-5),
            rope_theta: config.float(&["rope_theta"]).unwrap_or(10000.0),
            // GPT-2 ties unconditionally and does not say so in its config.
            tie_embeddings: config.flag("tie_word_embeddings").unwrap_or(arch.is("gpt2")),
            // The ordinary answer. An architecture that caches something else
            // overwrites this in `configure`.
            cache: CacheShape { k: n_kv_head * head_dim, v: n_kv_head * head_dim },
            config,
        };
        arch.configure(&mut spec)?;
        Ok(spec)
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
    /// As [`Transformer::forward_batch`], returning logits for **every**
    /// position: `tokens.len()` rows of `vocab_size`.
    ///
    /// This is what scoring text needs. Generation only ever wants the last
    /// row, so the fast path throws the rest away; perplexity wants all of
    /// them, and getting them from `m` separate forward passes would cost `m`
    /// times the memory traffic for the same arithmetic.
    ///
    /// The default is that slow, correct loop, which is also what the batched
    /// implementations are tested against.
    fn forward_batch_all(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let mut all = Vec::new();
        for &t in tokens {
            all.extend(self.forward(t, cache));
        }
        all
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
    shape: CacheShape,
    pub len: usize,
}

impl KvCache {
    pub fn new(spec: &Spec) -> Self {
        // Deliberately not pre-allocated: a 32k-context model would reserve
        // gigabytes up front for a conversation that may run to fifty tokens.
        KvCache {
            k: (0..spec.n_layer).map(|_| Vec::new()).collect(),
            v: (0..spec.n_layer).map(|_| Vec::new()).collect(),
            shape: spec.cache,
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
            k.truncate(len * self.shape.k);
            v.truncate(len * self.shape.v);
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
        let per_pos = self.shape.k + self.shape.v;
        self.len * per_pos * self.k.len() * std::mem::size_of::<f32>()
    }

    /// Bytes this cache would hold at full context.
    pub fn max_bytes(spec: &Spec) -> usize {
        let per_pos = spec.cache.k + spec.cache.v;
        spec.n_layer * spec.n_ctx * per_pos * std::mem::size_of::<f32>()
    }
}

/// Multi-head causal self-attention for a single query position.
///
/// Shared by GPT-2 and the Llama family, because attention itself did not
/// change between them. (DeepSeek's does not call this; see `deepseek.rs`.) For
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

    // One contiguous output, sliced per task, rather than a `Vec` per head
    // and a `concat` to join them: heads are small and there are a lot of
    // layers, so those allocations are not free at 169 matmuls a token.
    let mut out = vec![0.0f32; spec.n_head * hd];
    let per_task = spec.n_head.div_ceil(rayon::current_num_threads().max(1)).max(1);
    out.par_chunks_mut(per_task * hd).enumerate().for_each(|(task, dst)| {
        // Reused across every head this task handles.
        let mut scores = Vec::with_capacity(n_positions);

        for (j, slot) in dst.chunks_mut(hd).enumerate() {
            let head = task * per_task + j;
            let q_head = &q[head * hd..(head + 1) * hd];
            // Which KV head this query head reads from.
            let kv_off = (head / group) * hd;

            scores.clear();
            for t in 0..n_positions {
                let base = t * kv_dim + kv_off;
                // [`crate::tensor::dot`] rather than a running sum written
                // out here. This is the only loop in a decode step whose
                // length grows with the conversation, so the difference
                // between FMA latency and FMA throughput is the difference
                // between a long chat staying fast and not.
                scores.push(dot(q_head, &k_cache[base..base + hd]) * scale);
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

    /// Run `tokens` and return logits for **every** one of them, in order.
    ///
    /// Scoring text needs a distribution per position; generation needs only
    /// the last. The default here is the one-token-at-a-time loop, which is
    /// correct on any backend and slow on all of them — a session that can do
    /// better says so by overriding it.
    fn forward_all(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let mut all = Vec::new();
        for &t in tokens {
            all.extend(self.forward(&[t])?);
        }
        Ok(all)
    }
}

/// The hand-written engine, as a [`Session`].
pub struct CpuSession {
    model: Box<dyn Transformer>,
    cache: KvCache,
    precision: crate::quant::Precision,
    /// Prompt tokens per batched prefill pass.
    chunk: usize,
    /// The threads every matmul runs on.
    ///
    /// Owning a pool rather than using rayon's global one is not about
    /// isolation — it is so the forward pass can run *inside* it. See
    /// [`CpuSession::forward`].
    pool: rayon::ThreadPool,
}

impl CpuSession {
    pub fn new(model: Box<dyn Transformer>, precision: crate::quant::Precision) -> Self {
        let cache = KvCache::new(model.spec());
        // Tunable so the effect of batching stays measurable; 1 gives the old
        // token-at-a-time behaviour.
        let chunk = std::env::var("KVAD_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(64);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads())
            .thread_name(|i| format!("kvad-{i}"))
            .build()
            .expect("building a thread pool");
        CpuSession { model, cache, precision, chunk, pool }
    }
}

/// How many threads to decode on.
///
/// Not `num_cpus`. This machine reports 18 logical CPUs, but six of them are
/// performance cores and twelve are efficiency cores, and rayon splits a matmul
/// into equal pieces regardless. Every parallel section then ends when the
/// slowest piece does, so the E-cores set the pace and adding them past a point
/// makes decoding *slower* — measured at 35 tok/s on all 18 against 50 on
/// eleven.
///
/// So: all the performance cores, plus a few efficiency cores to absorb the
/// tail, and never the whole machine. `KVAD_THREADS` overrides it.
pub fn threads() -> usize {
    if let Some(n) = std::env::var("KVAD_THREADS").ok().and_then(|v| v.parse().ok()) {
        return n;
    }
    let total = std::thread::available_parallelism().map_or(4, |n| n.get());
    match perf_cores() {
        // Apple silicon and other big.LITTLE parts. Two thirds of the slow
        // cores is where this machine measures best (69 tok/s at 14-15 of 18);
        // the last few add throughput worth less than the barrier they extend.
        Some(fast) if fast < total => (fast + 2 * (total - fast) / 3).clamp(1, total),
        _ => total,
    }
}

/// Performance-core count, where the OS will say.
#[cfg(target_os = "macos")]
fn perf_cores() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.logicalcpu"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(not(target_os = "macos"))]
fn perf_cores() -> Option<usize> {
    None
}

impl Session for CpuSession {
    fn spec(&self) -> &Spec {
        self.model.spec()
    }

    /// Run the whole pass *inside* the pool, not from outside it.
    ///
    /// This one line was worth more than every kernel in `quant.rs` put
    /// together, and the reason is worth understanding.
    ///
    /// A `par_iter()` called from a thread that is not a pool worker takes
    /// rayon's cold path: push the job onto an injection queue, wake the
    /// workers, then block the caller on a condition variable until they
    /// finish. Two kernel transitions and a scheduler round trip, per matmul.
    /// Decoding one token runs 169 matmuls, so that is 169 sleeps and 169
    /// wake-ups, each one costing more than the arithmetic it is waiting for.
    /// A profile of the old code found 20% of CPU time in the kernels and the
    /// rest in `swtch_pri` and `psynch_cvwait` — the pool thrashing.
    ///
    /// `install` moves the whole pass onto a pool thread and blocks the caller
    /// once. Every `par_iter` inside is then being called *from* a worker, so
    /// it takes the hot path — push onto that worker's own deque, and let the
    /// other workers steal — with no injection queue and no condvar. The cold
    /// path is paid once per token instead of 169 times.
    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let CpuSession { model, cache, chunk, pool, .. } = self;
        Ok(pool.install(|| {
            let mut logits = Vec::new();
            for part in tokens.chunks(*chunk) {
                logits = model.forward_batch(part, cache);
            }
            logits
        }))
    }

    fn forward_all(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let CpuSession { model, cache, chunk, pool, .. } = self;
        // The same chunking as `forward`, and for the same reason; the only
        // difference is which rows of the result survive. A chunk's worth of
        // logits is `chunk × vocab` floats, which is why the chunk stays
        // small rather than being the whole file.
        Ok(pool.install(|| {
            let mut all = Vec::new();
            for part in tokens.chunks(*chunk) {
                all.extend(model.forward_batch_all(part, cache));
            }
            all
        }))
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

#[cfg(test)]
mod tests {

    use super::*;

    /// The families this engine has actually been run against, and the
    /// near-misses that a substring test used to accept.
    #[test]
    fn only_architectures_that_have_been_run_here_are_claimed() {
        let llama = Arch::require("llama");
        for t in ["llama", "mistral", "qwen2", "qwen3", "LlamaForCausalLM", "Qwen3ForCausalLM"] {
            assert_eq!(Arch::from_model_type(t), Some(llama), "{t}");
        }
        let gpt2 = Arch::require("gpt2");
        assert_eq!(Arch::from_model_type("gpt2"), Some(gpt2));
        assert_eq!(Arch::from_model_type("GPT2LMHeadModel"), Some(gpt2));

        // Each of these is a real model_type on the Hub, and each one a
        // substring test said yes to. A mixture of experts, and a model that
        // skips RoPE on every fourth layer.
        // DeepSeek is its own module now, and answers to its own names.
        assert!(Arch::from_model_type("deepseek_v2").unwrap().is("deepseek_v2"));
        assert!(Arch::from_model_type("DeepseekV3ForCausalLM").unwrap().is("deepseek_v3"));

        for t in ["qwen3_5_moe", "qwen2_moe", "smollm3", "deepseek_v4", "gemma2", "phi3"] {
            assert_eq!(Arch::from_model_type(t), None, "claimed to run `{t}`");
        }
    }

    /// No two modules may answer to the same `model_type`, or which one runs a
    /// checkpoint would depend on the order of the registry.
    #[test]
    fn no_two_architectures_claim_the_same_model_type() {
        let mut seen: Vec<&str> =
            arch::registry().iter().flat_map(|a| a.model_types()).copied().collect();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), before, "two architectures claim the same model_type");
    }

    fn test_spec() -> Spec {
        let (n_kv_head, head_dim) = (2, 2);
        Spec {
            arch: Arch::require("llama"),
            n_layer: 2,
            n_head: 4,
            n_kv_head,
            n_embd: 8,
            head_dim,
            n_ctx: 16,
            vocab_size: 32,
            intermediate: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
            tie_embeddings: true,
            cache: CacheShape { k: n_kv_head * head_dim, v: n_kv_head * head_dim },
            config: Json::default(),
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
