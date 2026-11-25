//! Nemotron-H: Mamba-2 on most layers, attention on a few, and every layer one
//! thing.
//!
//! NVIDIA's hybrid, and what Nemotron 3 Nano and Super and their fine-tunes
//! (Munin 2.0 among them) are built from. Read this as a diff against
//! [`super::qwen3_5`], the other hybrid here, and the first difference is the
//! skeleton itself: a Llama block is two halves, mix then feed forward, each
//! behind its own norm. A Nemotron-H layer is **one mixer** behind one norm,
//!
//! ```text
//!     x  <-  x + mixer(RMSNorm(x))
//! ```
//!
//! and `hybrid_override_pattern` says which mixer, a character a layer: `M`
//! Mamba-2, `*` attention, `-` an MLP, `E` a mixture of them. Munin's reads
//! `MEMEMEM*EMEMEMEM*E...`: forty Mamba layers, forty mixtures, eight
//! attention layers. An MLP is a layer of its own, not half of one.
//!
//! # Mamba-2
//!
//! A selective state space: per head `h`, a state `S` of `head_dim × state`
//! numbers, decayed and written once a token and read back with `C`:
//!
//! ```text
//!     S  <-  S · e^(dt·A)  +  dt · x ⊗ B       A = −e^A_log, one per head
//!     y   =  S · C  +  D · x
//! ```
//!
//! `dt`, `B` and `C` are projections of the token, which is what makes it
//! *selective*: the token decides how much of the state to keep (`dt`) and
//! where to write and read (`B`, `C`). `B` and `C` are shared by a group of
//! heads, as grouped-query attention shares keys. Against Qwen3.5's gated
//! delta net it is simpler, with no delta rule and no normalised keys, and
//! wider: Munin's 128 heads of 64 × 128 are 4 MB of state a layer.
//!
//! Around the recurrence: the same short causal convolution in front (over
//! `x`, `B` and `C` together, kernel 4, with a bias), and at the back a
//! **grouped** RMSNorm of `y · silu(z)`, the gate applied *before* the norm,
//! where Qwen3.5 gates after.
//!
//! # Attention
//!
//! Grouped-query attention with **no position encoding at all**. No RoPE, no
//! ALiBi: the Mamba layers' convolution and recurrence carry order, and the
//! attention layers only look things up.
//!
//! # MLPs and the mixture
//!
//! `down(relu(up(x))²)`, with no gate, for the dense layers, the routed
//! experts and the shared expert alike: [`Ffn::load_ungated`]. The router is
//! DeepSeek-V3's rule (a sigmoid, a balancing bias that picks but does not
//! weigh, normalise then scale) without the config saying so, which
//! [`configure`] supplies. Nemotron 3 Super adds a **latent** projection: the
//! routed experts run in 1024 dimensions rather than 4096, between
//! `fc1_latent_proj` and `fc2_latent_proj`, and the shared expert reads the
//! full width. [`Moe`] carries that, so the expert cache and read-ahead work
//! here as they do for Qwen3 and DeepSeek.
//!
//! # A prompt is one pass
//!
//! [`Model::pass`] runs every token of a batch through one layer before the
//! next, so each weight is read once a pass rather than once a token. For a
//! model whose experts are read from the SSD that is most of the prompt's
//! cost. The Mamba recurrence inside it still steps token by token; the
//! reference's chunked scan is a faster way to the same numbers, and is left
//! for when the prompt's arithmetic, not its reads, is what a prompt waits on.
//!
//! # What is not here
//!
//! The multi-token-prediction head (`mtp.*`), skipped by name, as Qwen3-Next's
//! is. And the remote code's pure-PyTorch fallback clamps `dt` below at
//! `time_step_min`; its CUDA path and `transformers` clamp to
//! `time_step_limit`, which this follows. See `docs/nemotron-h-plan.md`.

use super::ffn::{Ffn, Layout, Mlp, Moe, Router};
use super::{attend, Architecture, CacheLayout, CacheShape, KvCache, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::rms_norm;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub static ARCH: Architecture = Architecture {
    id: "nemotron_h",
    model_types: &["nemotron_h"],
    about: "Nemotron-H / Nemotron 3: Mamba-2, attention without positions, ReLU² MLPs and latent mixtures",
    configure,
    splits: false,
    load: whole,
};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// What a layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Mamba,
    Attention,
    Mlp,
    Moe,
}

/// The pattern, one entry a layer.
///
/// `hybrid_override_pattern` is how every published config says it.
/// `transformers` turns that into `layers_block_type` and writes the list back
/// out, so a config saved by it has the list instead; both are read.
pub fn layer_kinds(config: &super::Json, n_layer: usize) -> Res<Vec<Kind>> {
    let kinds: Vec<Kind> = match (config.text("hybrid_override_pattern"), config.get("layers_block_type")) {
        (Some(pattern), _) => pattern
            .chars()
            .map(|c| match c {
                'M' => Ok(Kind::Mamba),
                '*' => Ok(Kind::Attention),
                '-' => Ok(Kind::Mlp),
                'E' => Ok(Kind::Moe),
                other => Err(format!("config: unknown layer `{other}` in hybrid_override_pattern")),
            })
            .collect::<Result<_, _>>()?,
        (None, Some(list)) => list
            .as_array()
            .ok_or("config: `layers_block_type` is not a list")?
            .iter()
            .map(|v| match v.as_str() {
                Some("mamba") | Some("linear_attention") => Ok(Kind::Mamba),
                Some("attention") | Some("full_attention") => Ok(Kind::Attention),
                Some("mlp") => Ok(Kind::Mlp),
                Some("moe") => Ok(Kind::Moe),
                other => Err(format!("config: unknown layer type {other:?}")),
            })
            .collect::<Result<_, _>>()?,
        (None, None) => return Err("config: no `hybrid_override_pattern`".into()),
    };
    if kinds.len() != n_layer {
        return Err(format!("config: the pattern names {} layers and the model has {n_layer}", kinds.len()).into());
    }
    Ok(kinds)
}

/// A Mamba-2 layer's dimensions.
#[derive(Debug, Clone, Copy)]
pub struct Ssm {
    /// `mamba_num_heads`.
    pub heads: usize,
    /// `mamba_head_dim`: the `x` each head reads, and the rows of its state.
    pub head_dim: usize,
    /// `n_groups`: how many `B` and `C` the heads share between them.
    pub groups: usize,
    /// `ssm_state_size`: the columns of a head's state.
    pub state: usize,
    /// `conv_kernel`.
    pub conv: usize,
    /// `heads × head_dim`.
    pub inner: usize,
    /// `inner + 2 · groups · state`: `x`, `B` and `C`, which one convolution
    /// runs over.
    pub conv_dim: usize,
    /// `time_step_limit`: what `dt` is clamped to after its softplus.
    pub dt_limit: (f32, f32),
}

impl Ssm {
    pub fn read(config: &super::Json) -> Res<Self> {
        let heads = config.need("mamba_num_heads")?;
        let head_dim = config.need("mamba_head_dim")?;
        let groups = config.need("n_groups")?;
        let state = config.need("ssm_state_size")?;
        if heads % groups != 0 {
            return Err(format!("config: {heads} Mamba heads do not divide into {groups} groups").into());
        }
        let conv = config.num(&["conv_kernel"]).unwrap_or(4);
        if conv < 2 {
            return Err(format!("config: a convolution of width {conv} has no window to keep").into());
        }
        let limit = config.get("time_step_limit").and_then(|v| v.as_array());
        let bound = |i: usize, default: f32| {
            limit.and_then(|l| l.get(i)).and_then(|v| v.as_f64()).map_or(default, |v| v as f32)
        };
        let inner = heads * head_dim;
        Ok(Ssm {
            heads,
            head_dim,
            groups,
            state,
            conv,
            inner,
            conv_dim: inner + 2 * groups * state,
            dt_limit: (bound(0, 0.0), bound(1, f32::INFINITY)),
        })
    }

    /// Floats a layer keeps between tokens: the convolution's window, then
    /// every head's state.
    pub fn state_len(&self) -> usize {
        self.conv_window() + self.heads * self.head_dim * self.state
    }

    /// The convolution's previous `kernel − 1` inputs, a channel at a time.
    pub fn conv_window(&self) -> usize {
        self.conv_dim * (self.conv - 1)
    }
}

/// Refuse what this does not implement, rather than run without it.
fn refuse_unread(config: &super::Json) -> Res<()> {
    for (key, plain) in [
        ("attention_bias", false),
        ("mlp_bias", false),
        ("mamba_proj_bias", false),
        ("use_bias", false),
        ("use_conv_bias", true),
        ("residual_in_fp32", false),
    ] {
        if config.flag(key).is_some_and(|v| v != plain) {
            return Err(format!("config: `{key}: {}` is not implemented here", !plain).into());
        }
    }
    for (key, act) in [("mlp_hidden_act", "relu2"), ("mamba_hidden_act", "silu")] {
        if config.text(key).is_some_and(|v| v != act) {
            return Err(format!("config: `{key}` is `{}`, and only `{act}` is implemented", config.text(key).unwrap_or("")).into());
        }
    }
    Ok(())
}

/// Everything the shared [`Spec`] cannot work out by itself.
fn configure(spec: &mut Spec) -> Res<()> {
    refuse_unread(&spec.config)?;
    let kinds = layer_kinds(&spec.config, spec.n_layer)?;
    let ssm = Ssm::read(&spec.config)?;
    // The first Nemotron-H configs (4B, 8B, 47B) spell it the old way, and
    // `hidden / heads` is not it: 3072 / 32 is 96, and their heads are 128.
    if let Some(hd) = spec.config.num(&["head_dim", "attention_head_dim"]) {
        spec.head_dim = hd;
    }

    // Keys and values for the attention layers, a state for the Mamba ones,
    // and nothing for the MLPs and mixtures.
    let kv = spec.n_kv_head * spec.head_dim;
    spec.cache = CacheLayout::per_layer(
        kinds
            .iter()
            .map(|k| match k {
                Kind::Attention => CacheShape::kv(kv, kv),
                Kind::Mamba => CacheShape::recurrent(ssm.state_len()),
                Kind::Mlp | Kind::Moe => CacheShape::kv(0, 0),
            })
            .collect(),
    );

    // The routing rule, which the config does not name. It is DeepSeek-V3's
    // `noaux_tc` with one group: sigmoid scores, the correction bias added
    // only to choose, normalised, then always scaled by
    // `routed_scaling_factor`. Under the defaults `Router` would softmax, and
    // skip the scale after normalising: a model that runs, five times quiet.
    if Router::count(&spec.config).is_some() {
        let mut config = spec.config.value().clone();
        if let Some(map) = config.as_object_mut() {
            map.entry("scoring_func").or_insert("sigmoid".into());
            map.entry("topk_method").or_insert("noaux_tc".into());
        }
        spec.config = super::Json::new(config);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

struct Mamba {
    /// `[inner + conv_dim + heads, hidden]`: the gate `z`, then `x B C`, then `dt`.
    in_proj: Weight,
    /// `[conv_dim, kernel]`, depthwise. Stored `[conv_dim, 1, kernel]`, so the
    /// flat read is the filters back to back.
    conv: Vec<f32>,
    conv_bias: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    d: Vec<f32>,
    /// `[inner]`: the grouped norm's scale.
    norm: Vec<f32>,
    out_proj: Weight,
}

struct Attn {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
}

enum Mixer {
    Mamba(Box<Mamba>),
    Attention(Box<Attn>),
    /// A dense MLP, or a mixture: [`Mlp`] is both already.
    Mlp(Mlp),
}

struct Layer {
    norm: Vec<f32>,
    mixer: Mixer,
}

pub struct Model {
    spec: Spec,
    ssm: Ssm,
    kinds: Vec<Kind>,
    embed: Weight,
    lm_head: Option<Weight>,
    layers: Vec<Layer>,
    norm_f: Vec<f32>,
    /// For each layer, the next mixture above it, for a mixture to guess that
    /// one's experts. Not the next layer: Mamba layers sit between them.
    next_moe: Vec<Option<usize>>,
    /// Every layer but the mixtures, read off the disk a layer at a time, for
    /// a model too big for its mapping to stay in memory. The mixtures' experts
    /// have [`crate::experts`] instead. See [`crate::layers`].
    stream: Option<std::sync::Arc<crate::layers::Layers>>,
}

fn whole(src: &dyn Source, spec: Spec, shard: &super::Shard) -> Res<Box<dyn Transformer>> {
    if !shard.is_whole(&spec) {
        return Err(format!("the `{}` architecture runs whole and does not split yet", spec.arch).into());
    }
    Ok(Box::new(Model::load(src, spec)?))
}

/// One token through the matrix-vector kernel, more than one through the
/// matrix one: decode's hot path stays the kernel it always was.
fn project(w: &Weight, xs: &[f32], m: usize) -> Vec<f32> {
    match m {
        1 => w.matvec_bt(xs, None),
        _ => w.matmul_bt(xs, m, None),
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Model {
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let kinds = layer_kinds(&spec.config, spec.n_layer)?;
        let ssm = Ssm::read(&spec.config)?;
        let routed = match Router::count(&spec.config) {
            Some(_) => Some((Router::read(&spec.config)?, Layout::read(&spec.config))),
            None => None,
        };

        let mut layers = Vec::with_capacity(spec.n_layer);
        for (i, kind) in kinds.iter().enumerate() {
            let p = |s: &str| format!("backbone.layers.{i}.{s}");
            let mixer = match kind {
                Kind::Mamba => Mixer::Mamba(Box::new(Mamba {
                    in_proj: src.matrix(&p("mixer.in_proj.weight"))?,
                    conv: src.vector(&p("mixer.conv1d.weight"))?,
                    conv_bias: src.vector(&p("mixer.conv1d.bias"))?,
                    dt_bias: src.vector(&p("mixer.dt_bias"))?,
                    a_log: src.vector(&p("mixer.A_log"))?,
                    d: src.vector(&p("mixer.D"))?,
                    norm: src.vector(&p("mixer.norm.weight"))?,
                    out_proj: src.matrix(&p("mixer.out_proj.weight"))?,
                })),
                Kind::Attention => Mixer::Attention(Box::new(Attn {
                    q: src.matrix(&p("mixer.q_proj.weight"))?,
                    k: src.matrix(&p("mixer.k_proj.weight"))?,
                    v: src.matrix(&p("mixer.v_proj.weight"))?,
                    o: src.matrix(&p("mixer.o_proj.weight"))?,
                })),
                Kind::Mlp => Mixer::Mlp(Mlp::Dense(Ffn::load_ungated(src, &p("mixer"))?)),
                Kind::Moe => {
                    let (router, layout) =
                        routed.as_ref().ok_or("an `E` layer, and no `n_routed_experts` in the config")?;
                    Mixer::Mlp(Mlp::Moe(Box::new(Moe::load_as(src, &p("mixer"), router, layout.shared, false)?)))
                }
            };
            layers.push(Layer { norm: src.vector(&p("norm.weight"))?, mixer });
        }
        // Deliberately not run: the multi-token-prediction head, a training
        // device that speculative decoding could use and this does not.
        src.skip_under("mtp.");

        let dense: Vec<usize> = (0..kinds.len()).filter(|&i| kinds[i] != Kind::Moe).collect();
        let stream = src.layers(&dense);
        let next_moe = (0..kinds.len())
            .map(|i| (i + 1..kinds.len()).find(|&j| kinds[j] == Kind::Moe))
            .collect();
        Ok(Model {
            embed: src.matrix("backbone.embeddings.weight")?,
            lm_head: head(src, &spec, "lm_head.weight")?,
            norm_f: src.vector("backbone.norm_f.weight")?,
            layers,
            next_moe,
            stream,
            kinds,
            ssm,
            spec,
        })
    }

    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }

    /// When its layers are streamed: bytes read so far, and weights that ran
    /// from the mapping instead of their slab.
    pub fn streamed(&self) -> Option<(u64, u64)> {
        self.stream.as_ref().map(|s| s.counts())
    }

    /// A Mamba-2 layer over `m` tokens, in order, carrying `state` from each to
    /// the next.
    ///
    /// `in_proj` and `out_proj` are `net`'s, or the same read into a slab.
    fn mamba(&self, net: &Mamba, (in_proj, out_proj): (&Weight, &Weight), hs: &[f32], m: usize, state: &mut [f32]) -> Vec<f32> {
        let s = &self.ssm;
        let (k, window) = (s.conv, s.conv_window());
        let width = s.inner + s.conv_dim + s.heads;
        let proj = project(in_proj, hs, m);
        let per_group = s.heads / s.groups;
        let mut ys = vec![0.0f32; m * s.inner];

        for t in 0..m {
            let row = &proj[t * width..(t + 1) * width];
            let (z, rest) = row.split_at(s.inner);
            let (xbc_in, dt) = rest.split_at(s.conv_dim);

            // ---- the depthwise causal convolution, then SiLU -------------
            //
            // As in `qwen3_5::Model::linear_step`: the window holds the last
            // `k − 1` *inputs*, and this token's input joins it after its
            // own output is computed.
            let mut xbc = vec![0.0f32; s.conv_dim];
            let (conv_state, ssm_state) = state.split_at_mut(window);
            for c in 0..s.conv_dim {
                let past = &mut conv_state[c * (k - 1)..(c + 1) * (k - 1)];
                let filter = &net.conv[c * k..(c + 1) * k];
                let mut acc = net.conv_bias[c] + filter[k - 1] * xbc_in[c];
                for j in 0..k - 1 {
                    acc += filter[j] * past[j];
                }
                past.copy_within(1.., 0);
                past[k - 2] = xbc_in[c];
                xbc[c] = silu(acc);
            }
            let (x, bc) = xbc.split_at(s.inner);
            let (b, c) = bc.split_at(s.groups * s.state);

            // ---- the selective scan, one step ----------------------------
            let y = &mut ys[t * s.inner..(t + 1) * s.inner];
            for h in 0..s.heads {
                let g = h / per_group;
                let (b, c) = (&b[g * s.state..(g + 1) * s.state], &c[g * s.state..(g + 1) * s.state]);
                // `softplus`, written so a large argument does not overflow
                // the exponential on its way to being ~x, as torch's does.
                let raw = dt[h] + net.dt_bias[h];
                let dt = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
                let dt = dt.clamp(s.dt_limit.0, s.dt_limit.1);
                let decay = (-dt * net.a_log[h].exp()).exp();
                for p in 0..s.head_dim {
                    let xv = x[h * s.head_dim + p];
                    let at = (h * s.head_dim + p) * s.state;
                    let row = &mut ssm_state[at..at + s.state];
                    let mut out = 0.0f32;
                    for n in 0..s.state {
                        row[n] = row[n] * decay + dt * b[n] * xv;
                        out += row[n] * c[n];
                    }
                    y[h * s.head_dim + p] = out + net.d[h] * xv;
                }
            }

            // ---- gate, then the grouped norm -----------------------------
            for (v, &g) in y.iter_mut().zip(z) {
                *v *= silu(g);
            }
            let group = s.inner / s.groups;
            for (chunk, w) in y.chunks_mut(group).zip(net.norm.chunks(group)) {
                let normed = rms_norm(chunk, w, self.spec.eps);
                chunk.copy_from_slice(&normed);
            }
        }
        project(out_proj, &ys, m)
    }

    /// An attention layer over `m` tokens starting at position `pos0`. Each
    /// token sees the cache up to and including itself.
    fn attention(&self, attn: &Attn, hs: &[f32], m: usize, layer: usize, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let (q_dim, kv_dim) = (spec.n_head * spec.head_dim, spec.kv_dim());
        let (q, k, v) = (project(&attn.q, hs, m), project(&attn.k, hs, m), project(&attn.v, hs, m));
        let pos0 = cache.len;
        let mut out = Vec::with_capacity(m * q_dim);
        for t in 0..m {
            cache.push(layer, &k[t * kv_dim..(t + 1) * kv_dim], &v[t * kv_dim..(t + 1) * kv_dim]);
            out.extend(attend(spec, &q[t * q_dim..(t + 1) * q_dim], cache.keys(layer), cache.values(layer), pos0 + t + 1));
        }
        project(&attn.o, &out, m)
    }

    /// `m` tokens through every layer, one layer at a time. Returns the final
    /// hidden states, normed, one row a token.
    fn pass(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let (e, m) = (self.spec.n_embd, tokens.len());
        let mut xs: Vec<f32> = tokens.iter().flat_map(|&t| self.embed.row(t as usize).to_vec()).collect();
        for (l, layer) in self.layers.iter().enumerate() {
            let hs: Vec<f32> = xs.chunks(e).flat_map(|x| rms_norm(x, &layer.norm, self.spec.eps)).collect();
            // This layer's matrices read into a slab, when it is streamed: the
            // same bytes, so the same answer, from memory of our own.
            let stream = self.stream.as_ref().filter(|_| self.kinds[l] != Kind::Moe);
            let held = stream.and_then(|s| s.get(l));
            let on = |w: &Weight| {
                let view = held.as_ref().and_then(|(lo, slab)| w.rebased(*lo, slab));
                if let (Some(s), None) = (stream, &view) {
                    s.fell_back();
                }
                view
            };
            let ys = match &layer.mixer {
                Mixer::Mamba(net) => {
                    let (i, o) = (on(&net.in_proj), on(&net.out_proj));
                    let w = (i.as_ref().unwrap_or(&net.in_proj), o.as_ref().unwrap_or(&net.out_proj));
                    self.mamba(net, w, &hs, m, cache.state_mut(l))
                }
                Mixer::Attention(attn) => match (on(&attn.q), on(&attn.k), on(&attn.v), on(&attn.o)) {
                    (Some(q), Some(k), Some(v), Some(o)) => self.attention(&Attn { q, k, v, o }, &hs, m, l, cache),
                    _ => self.attention(attn, &hs, m, l, cache),
                },
                Mixer::Mlp(Mlp::Dense(ffn)) if stream.is_some() => {
                    let view = held.as_ref().and_then(|(lo, slab)| ffn.rebased(*lo, slab));
                    if view.is_none() {
                        stream.into_iter().for_each(|s| s.fell_back());
                    }
                    let f = view.as_ref().unwrap_or(ffn);
                    match m {
                        1 => f.run_one(&hs),
                        _ => f.run(&hs, m),
                    }
                }
                Mixer::Mlp(mlp) => {
                    let next = self.next_moe[l].map(|j| match &self.layers[j].mixer {
                        Mixer::Mlp(next) => next,
                        _ => unreachable!("next_moe names mixtures only"),
                    });
                    match m {
                        1 => mlp.run_one(&hs, e, next),
                        _ => mlp.run(&hs, m, e, next),
                    }
                }
            };
            for (x, y) in xs.iter_mut().zip(&ys) {
                *x += y;
            }
            // Every view of the slab is gone with the match above.
            drop(held);
            if let Some(s) = stream {
                s.done(l);
            }
        }
        cache.len += m;
        xs.chunks(e).flat_map(|x| rms_norm(x, &self.norm_f, self.spec.eps)).collect()
    }

    fn logits(&self, hs: &[f32], m: usize) -> Vec<f32> {
        project(self.lm_head.as_ref().unwrap_or(&self.embed), hs, m)
    }
}

impl Transformer for Model {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let h = self.pass(&[token], cache);
        self.logits(&h, 1)
    }

    fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let e = self.spec.n_embd;
        let h = self.pass(tokens, cache);
        self.logits(&h[h.len() - e..], 1)
    }

    fn forward_batch_all(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let h = self.pass(tokens, cache);
        self.logits(&h, tokens.len())
    }

    fn param_count(&self) -> usize {
        let opt = |v: &Option<Weight>| v.as_ref().map_or(0, |w| w.param_count());
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                l.norm.len()
                    + match &l.mixer {
                        Mixer::Mamba(n) => {
                            n.in_proj.param_count()
                                + n.out_proj.param_count()
                                + n.conv.len()
                                + n.conv_bias.len()
                                + n.dt_bias.len()
                                + n.a_log.len()
                                + n.d.len()
                                + n.norm.len()
                        }
                        Mixer::Attention(a) => {
                            a.q.param_count() + a.k.param_count() + a.v.param_count() + a.o.param_count()
                        }
                        Mixer::Mlp(mlp) => mlp.param_count(),
                    }
            })
            .sum();
        layers + self.embed.param_count() + opt(&self.lm_head) + self.norm_f.len()
    }

    fn memory_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Mamba(n) => n.in_proj.bytes() + n.out_proj.bytes(),
                Mixer::Attention(a) => a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes(),
                Mixer::Mlp(mlp) => mlp.bytes(),
            })
            .sum();
        self.embed.bytes() + self.lm_head.as_ref().map_or(0, |h| h.bytes()) + layers
    }
}
