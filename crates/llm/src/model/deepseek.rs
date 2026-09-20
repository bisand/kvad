//! DeepSeek V2 and V3: latent attention, and a mixture of experts.
//!
//! Read this as a diff against [`super::llama`], which is where the shared
//! skeleton stops being enough. Two things changed, and both changed for the
//! same reason — the things that make a large model expensive to *serve* are
//! not the things that make it expensive to train.
//!
//! # Multi-head Latent Attention
//!
//! Grouped-query attention shrinks the KV cache by making query heads share
//! keys. MLA shrinks it by not storing keys at all.
//!
//! Each position is compressed once, into a single vector `c` of
//! `kv_lora_rank` numbers — 512, against the 2048 of the residual stream —
//! and every head's key and value are *reconstructed* from it by a per-head
//! matrix. So the cache holds `c` and nothing else.
//!
//! That alone would be a bad trade: reconstructing every head's key at every
//! step is more arithmetic than reading one. The trick is that you never have
//! to. Writing `W_UK[h]` for the matrix that turns `c` into head `h`'s key,
//!
//! ```text
//!     score = q · (W_UK[h] c) = (W_UK[h]ᵀ q) · c
//! ```
//!
//! and the right-hand side scores directly against the stored vector. The
//! per-head matrix moves onto the *query*, which there is only one of per
//! step, instead of onto the key, of which there are thousands. The same
//! applies at the other end: the softmax's weighted average can be taken over
//! the `c`s and un-compressed once, rather than un-compressing first.
//!
//! This is "absorption", and it is why the cache here is 576 floats per
//! position per layer where the same model as ordinary multi-head attention
//! would need 5120 — a factor of nine, which at long context is the
//! difference between fitting and not.
//!
//! One detail resists it. Rotary position embedding is applied *after* the
//! projection, and `W_UK[h]ᵀ R(pos)` is not a matrix you can precompute,
//! because it depends on the position. DeepSeek's answer is to not rotate
//! most of the vector: each head's query and key are split into a
//! `qk_nope_head_dim` part that carries content and is absorbed, and a
//! `qk_rope_head_dim` part that carries position and is stored as it is —
//! once, shared by every head. Hence the cache's two streams, of 512 and 64,
//! and hence [`CacheShape`](super::CacheShape) being a thing at all.
//!
//! # Mixture of experts
//!
//! The MLP is replaced by 64 (V2-Lite) or 256 (V3) small MLPs and a router
//! that picks a handful per token. V2-Lite has 15.7B parameters and uses 2.4B
//! of them on any given token — so it costs a 2.4B model to run and knows what
//! a 16B model knows. A couple of *shared* experts run for every token,
//! holding whatever all tokens need and leaving the routed ones free to
//! specialise.
//!
//! Routing is per token, so a batch of 64 prompt tokens generally touches most
//! of the experts. [`Moe::run`] therefore groups the batch by expert and runs
//! each one once over its own rows, rather than walking the batch and paying
//! for every expert's weights on every token.
//!
//! # V2 against V3
//!
//! The same file, because they are the same architecture. V3 is wider and
//! deeper, has eight groups of experts instead of one, scores them with a
//! sigmoid rather than a softmax, and adds a learned per-expert bias used for
//! choosing experts but not for weighting them — the "auxiliary-loss-free"
//! balancing trick. All four differences live in [`Router`], read from the
//! config.
//!
//! What is *not* implemented is V3's `num_nextn_predict_layers`: the
//! multi-token-prediction head, which is a training-time device and is not
//! needed to run the model. Its weights are skipped.

use super::{Architecture, CacheShape, Json, KvCache, Spec, Transformer};
use crate::qcache::Source;
use crate::quant::Weight;
use crate::tensor::{rms_norm, softmax_inplace, swiglu_inplace, Rope};
use rayon::prelude::*;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// DeepSeek V2 — the architecture, and the one of the two that has been run
/// here end to end.
pub static V2: Architecture = Architecture {
    id: "deepseek_v2",
    model_types: &["deepseek_v2"],
    about: "DeepSeek V2: latent attention (MLA) over a 512-wide cache, 64 routed experts",
    configure,
    load: |src, spec| Ok(Box::new(Model::load(src, spec)?)),
};

/// DeepSeek V3, and R1, which is V3 with different weights.
///
/// Same code, four config-driven differences in the router. The public
/// checkpoints are 671B parameters stored in fp8, which this engine does not
/// read; [`configure`] says so rather than failing later with a dtype error.
pub static V3: Architecture = Architecture {
    id: "deepseek_v3",
    model_types: &["deepseek_v3"],
    about: "DeepSeek V3/R1: MLA, 256 experts in 8 groups, sigmoid routing with a learned bias",
    configure,
    load: |src, spec| Ok(Box::new(Model::load(src, spec)?)),
};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The dimensions MLA has and ordinary attention does not.
#[derive(Debug, Clone, Copy)]
struct Mla {
    n_head: usize,
    /// Per head, the part of the query and key that carries content and is
    /// absorbed into the projection matrices.
    qk_nope: usize,
    /// Per head, the part that carries position and is rotated. Shared by
    /// every head on the key side — one rotated vector per position, not one
    /// per head.
    qk_rope: usize,
    v_head: usize,
    /// Width of the compressed vector that is all the cache holds.
    kv_lora: usize,
    /// V3 compresses the query too, through a rank of this width, to save
    /// parameters. V2-Lite does not (`q_lora_rank: null`).
    q_lora: Option<usize>,
    /// `1/sqrt(qk_nope + qk_rope)`, times YaRN's correction when there is one.
    softmax_scale: f32,
}

impl Mla {
    fn read(config: &Json, n_head: usize) -> Res<Self> {
        let qk_nope = config.need("qk_nope_head_dim")?;
        let qk_rope = config.need("qk_rope_head_dim")?;
        let mut softmax_scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt();
        // YaRN raises attention temperature along with the frequencies. This
        // is a plain multiplier on every score, and it is *not* optional: at
        // factor 40 it is 1.59, and without it the softmax is far too sharp
        // at every position, including position zero.
        if let Some(rope) = config.get("rope_scaling") {
            let get = |k: &str| rope.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
            if let (Some(all_dim), Some(factor)) = (get("mscale_all_dim"), get("factor")) {
                if all_dim != 0.0 {
                    let m = yarn_mscale(factor, all_dim);
                    softmax_scale *= m * m;
                }
            }
        }
        Ok(Mla {
            n_head,
            qk_nope,
            qk_rope,
            v_head: config.need("v_head_dim")?,
            kv_lora: config.need("kv_lora_rank")?,
            // Present and null in V2-Lite's config, which is not the same as
            // absent, and `num` says None to both.
            q_lora: config.num(&["q_lora_rank"]),
            softmax_scale,
        })
    }

    /// Width of one head's query: content part then position part, in that
    /// order, which is how the checkpoint stores it.
    fn q_head(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
}

/// What the shared [`Spec`] cannot work out for itself.
fn configure(spec: &mut Spec) -> Res<()> {
    // fp8 is not a dtype this engine's safetensors reader handles, and the
    // failure it would otherwise produce arrives thirty gigabytes later.
    if let Some(q) = spec.config.get("quantization_config") {
        let method = q
            .get("quant_method")
            .and_then(|m| m.as_str())
            .unwrap_or("?");
        return Err(format!(
            "this checkpoint's weights are stored as `{method}`, which this engine does not \
             read. DeepSeek publishes V3 and R1 in fp8; a bf16 conversion of the same model \
             would load."
        )
        .into());
    }
    let mla = Mla::read(&spec.config, spec.n_head)?;
    // `head_dim` means "how wide is one head's output" everywhere else, and
    // under MLA that is the value width, which is not the query width.
    spec.head_dim = mla.v_head;
    // MLA is not grouped-query attention: every head has its own key, it is
    // just never stored. Saying otherwise would make `kv_dim` a lie.
    spec.n_kv_head = spec.n_head;
    // The whole point of the architecture, as a number.
    spec.cache = CacheShape {
        k: mla.qk_rope,
        v: mla.kv_lora,
    };
    Ok(())
}

// ---------------------------------------------------------------------------
// YaRN
// ---------------------------------------------------------------------------

/// YaRN's attention-temperature correction, `0.1 m ln(s) + 1`.
fn yarn_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        return 1.0;
    }
    0.1 * mscale * scale.ln() + 1.0
}

/// The dimension whose wavelength is `rotations` full turns at the trained
/// context length — YaRN's way of asking "which frequencies has this model
/// actually seen wrap around?".
fn correction_dim(rotations: f32, dim: usize, base: f32, trained: f32) -> f32 {
    (dim as f32 * (trained / (rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * base.ln())
}

/// Build the rotation table, interpolating frequencies if the config asks.
///
/// Without `rope_scaling` this is the ordinary geometric series and matches
/// [`Rope::new`]. With it, YaRN: the fast dimensions — the ones that have
/// wrapped around many times during training, and so have seen every phase —
/// keep their frequencies, the slow ones are divided by the scaling factor so
/// the model's whole trained range is stretched over the longer context, and
/// a linear ramp blends between the two.
///
/// This is why a 4k model reads 160k: not by being told a bigger number, but
/// by moving only the frequencies that would otherwise be extrapolated past
/// anything they were trained on.
fn build_rope(dim: usize, max_positions: usize, theta: f32, config: &Json) -> Res<Rope> {
    let half = dim / 2;
    let extra: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
        .collect();

    let Some(scaling) = config.get("rope_scaling") else {
        return Ok(Rope::from_freqs(&extra, max_positions, 1.0));
    };
    let kind = scaling
        .get("type")
        .or_else(|| scaling.get("rope_type"))
        .and_then(|t| t.as_str());
    if kind != Some("yarn") {
        return Err(format!(
            "rope_scaling type `{}` is not implemented; this engine does yarn",
            kind.unwrap_or("?")
        )
        .into());
    }
    let get = |k: &str, fallback: f32| {
        scaling
            .get(k)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(fallback)
    };
    let factor = get("factor", 1.0);
    let trained = get("original_max_position_embeddings", 4096.0);
    let (beta_fast, beta_slow) = (get("beta_fast", 32.0), get("beta_slow", 1.0));

    let low = correction_dim(beta_fast, dim, theta, trained)
        .floor()
        .max(0.0);
    let high = correction_dim(beta_slow, dim, theta, trained)
        .ceil()
        .min(dim as f32 - 1.0);
    // The reference guards this singularity the same way; it only bites on a
    // config where the two betas land on one dimension.
    let high = if (high - low).abs() < f32::EPSILON {
        high + 0.001
    } else {
        high
    };

    let inv_freq: Vec<f32> = extra
        .iter()
        .enumerate()
        .map(|(i, &fast)| {
            let slow = fast / factor;
            // 1 at the fast end, 0 at the slow end.
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            slow * ramp + fast * (1.0 - ramp)
        })
        .collect();

    // `mscale` scales the rotated vector; `mscale_all_dim` is already in the
    // softmax scale. Their ratio is what is left for the table, and it is 1.0
    // for every DeepSeek config published so far.
    let amplitude =
        yarn_mscale(factor, get("mscale", 1.0)) / yarn_mscale(factor, get("mscale_all_dim", 1.0));
    Ok(Rope::from_freqs(&inv_freq, max_positions, amplitude))
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scoring {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Select {
    /// V2-Lite: the top `k` experts, and nothing else.
    Greedy,
    /// V2-236B: pick `topk_group` groups by their best expert, then the top
    /// `k` experts within those. A device holds a group, so limiting how many
    /// groups a token can reach limits how many devices its activations have
    /// to cross.
    GroupLimited,
    /// V3: as above, but a group is scored by its best *two* experts, and the
    /// choice is made on scores plus a learned per-expert bias that does not
    /// reach the weights. The bias is nudged during training towards whatever
    /// balances the experts, which is how V3 balances them without an
    /// auxiliary loss pulling against the language objective.
    NoAuxTc,
}

/// How a token's experts are chosen, and with what weights.
#[derive(Debug, Clone)]
struct Router {
    n_experts: usize,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    scoring: Scoring,
    select: Select,
    norm_topk: bool,
    scale: f32,
    /// V3 multiplies by `routed_scaling_factor` whether or not it normalised;
    /// V2 does it only when it did not. The two reference files differ in
    /// exactly this line.
    always_scale: bool,
}

impl Router {
    fn read(spec: &Spec) -> Res<Self> {
        let c = &spec.config;
        let scoring = match c.text("scoring_func").unwrap_or("softmax") {
            "softmax" => Scoring::Softmax,
            "sigmoid" => Scoring::Sigmoid,
            other => return Err(format!("unknown scoring_func `{other}`").into()),
        };
        let select = match c.text("topk_method").unwrap_or("greedy") {
            "greedy" => Select::Greedy,
            "group_limited_greedy" => Select::GroupLimited,
            "noaux_tc" => Select::NoAuxTc,
            other => return Err(format!("unknown topk_method `{other}`").into()),
        };
        Ok(Router {
            n_experts: c.need("n_routed_experts")?,
            top_k: c.need("num_experts_per_tok")?,
            n_group: c.num(&["n_group"]).unwrap_or(1),
            topk_group: c.num(&["topk_group"]).unwrap_or(1),
            scoring,
            select,
            norm_topk: c.flag("norm_topk_prob").unwrap_or(false),
            scale: c.float(&["routed_scaling_factor"]).unwrap_or(1.0),
            always_scale: select == Select::NoAuxTc,
        })
    }

    /// Which experts run for one token, and how much each one counts.
    fn route(&self, logits: &[f32], bias: Option<&[f32]>, out: &mut Vec<(usize, f32)>) {
        let mut scores = logits.to_vec();
        match self.scoring {
            Scoring::Softmax => softmax_inplace(&mut scores),
            Scoring::Sigmoid => {
                for s in &mut scores {
                    *s = 1.0 / (1.0 + (-*s).exp());
                }
            }
        }

        // What the *choice* is made on, which for V3 is not what the weights
        // are read from.
        let mut choice = scores.clone();
        if let Some(bias) = bias {
            for (c, b) in choice.iter_mut().zip(bias) {
                *c += b;
            }
        }
        if self.select != Select::Greedy && self.n_group > 1 {
            self.mask_to_best_groups(&mut choice);
        }

        out.clear();
        // `top_k` is 6 or 8 against 64 or 256 experts, so a partial selection
        // beats sorting the whole row.
        for _ in 0..self.top_k.min(self.n_experts) {
            let mut best = usize::MAX;
            for i in 0..self.n_experts {
                if choice[i] > f32::NEG_INFINITY && (best == usize::MAX || choice[i] > choice[best])
                {
                    best = i;
                }
            }
            if best == usize::MAX {
                break;
            }
            out.push((best, scores[best]));
            choice[best] = f32::NEG_INFINITY;
        }

        let normalised = self.norm_topk && self.top_k > 1;
        if normalised {
            let total: f32 = out.iter().map(|(_, w)| *w).sum::<f32>() + 1e-20;
            for (_, w) in out.iter_mut() {
                *w /= total;
            }
        }
        if self.always_scale || !normalised {
            for (_, w) in out.iter_mut() {
                *w *= self.scale;
            }
        }
    }

    /// Knock every expert outside the best `topk_group` groups out of the
    /// running.
    fn mask_to_best_groups(&self, choice: &mut [f32]) {
        let per_group = self.n_experts / self.n_group;
        let strength = |g: usize| -> f32 {
            let group = &choice[g * per_group..(g + 1) * per_group];
            match self.select {
                // V2 scores a group by its single best expert.
                Select::GroupLimited => group.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                // V3 by its best two, which stops one strong expert from
                // dragging in a group that is otherwise weak.
                _ => {
                    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for &v in group {
                        if v > a {
                            b = a;
                            a = v;
                        } else if v > b {
                            b = v;
                        }
                    }
                    a + b
                }
            }
        };
        let mut order: Vec<usize> = (0..self.n_group).collect();
        order.sort_by(|&a, &b| strength(b).total_cmp(&strength(a)));
        for &g in &order[self.topk_group.min(self.n_group)..] {
            for v in &mut choice[g * per_group..(g + 1) * per_group] {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// A SwiGLU MLP, which is what both a dense layer and one expert are.
struct Ffn {
    gate: Weight,
    up: Weight,
    down: Weight,
}

impl Ffn {
    fn load(src: &dyn Source, prefix: &str) -> Res<Self> {
        Ok(Ffn {
            gate: src.matrix(&format!("{prefix}.gate_proj.weight"))?,
            up: src.matrix(&format!("{prefix}.up_proj.weight"))?,
            down: src.matrix(&format!("{prefix}.down_proj.weight"))?,
        })
    }

    fn run(&self, xs: &[f32], m: usize) -> Vec<f32> {
        let mut gate = self.gate.matmul_bt(xs, m, None);
        let up = self.up.matmul_bt(xs, m, None);
        swiglu_inplace(&mut gate, &up);
        self.down.matmul_bt(&gate, m, None)
    }

    fn param_count(&self) -> usize {
        self.gate.param_count() + self.up.param_count() + self.down.param_count()
    }

    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

/// The query projection, which V3 compresses and V2-Lite does not.
enum Query {
    Direct(Weight),
    Compressed {
        down: Weight,
        norm: Vec<f32>,
        up: Weight,
    },
}

struct Attn {
    norm: Vec<f32>,
    q: Query,
    /// `hidden -> kv_lora_rank + qk_rope_head_dim`: the compression and the
    /// one shared rotary key, in a single matrix.
    kv_a: Weight,
    kv_a_norm: Vec<f32>,
    /// Per head, `W_UK[h]ᵀ` — stored transposed because it is applied to the
    /// query, not to the key. See the module header.
    uk: Vec<Weight>,
    /// Per head, `W_UV[h]`: turns the attention-weighted average of cached
    /// vectors into this head's output.
    uv: Vec<Weight>,
    o: Weight,
}

enum Mlp {
    /// The first `first_k_dense_replace` layers are ordinary. The router
    /// needs a residual stream that already means something, and at layer
    /// zero it does not.
    Dense(Ffn),
    Moe(Moe),
}

struct Moe {
    /// `[n_routed_experts, hidden]` — the router itself, one dot product per
    /// expert.
    gate: Weight,
    /// V3's learned balancing bias, if this is V3.
    bias: Option<Vec<f32>>,
    experts: Vec<Ffn>,
    /// Runs for every token, whatever the router says.
    shared: Option<Ffn>,
}

struct Block {
    attn: Attn,
    mlp_norm: Vec<f32>,
    mlp: Mlp,
}

pub struct Model {
    spec: Spec,
    mla: Mla,
    router: Router,
    embed: Weight,
    lm_head: Option<Weight>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    rope: Rope,
}

impl Model {
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let mla = Mla::read(&spec.config, spec.n_head)?;
        let router = Router::read(&spec)?;
        let c = &spec.config;
        let first_dense = c.num(&["first_k_dense_replace"]).unwrap_or(0);
        let moe_every = c.num(&["moe_layer_freq"]).unwrap_or(1).max(1);
        let n_shared = c.num(&["n_shared_experts"]).unwrap_or(0);

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            let attn = Attn {
                norm: src.vector(&p("input_layernorm.weight"))?,
                q: match mla.q_lora {
                    None => Query::Direct(src.matrix(&p("self_attn.q_proj.weight"))?),
                    Some(_) => Query::Compressed {
                        down: src.matrix(&p("self_attn.q_a_proj.weight"))?,
                        norm: src.vector(&p("self_attn.q_a_layernorm.weight"))?,
                        up: src.matrix(&p("self_attn.q_b_proj.weight"))?,
                    },
                },
                kv_a: src.matrix(&p("self_attn.kv_a_proj_with_mqa.weight"))?,
                kv_a_norm: src.vector(&p("self_attn.kv_a_layernorm.weight"))?,
                // One stored matrix, `[n_head * (qk_nope + v_head), kv_lora]`,
                // holding both up-projections for every head back to back.
                // They are used at opposite ends of the softmax and one of
                // them is used transposed, so they are separated here rather
                // than sliced on every token.
                uk: (0..mla.n_head)
                    .map(|h| {
                        let start = h * (mla.qk_nope + mla.v_head);
                        src.matrix_rows_t(&p("self_attn.kv_b_proj.weight"), start, mla.qk_nope)
                    })
                    .collect::<Res<Vec<_>>>()?,
                uv: (0..mla.n_head)
                    .map(|h| {
                        let start = h * (mla.qk_nope + mla.v_head) + mla.qk_nope;
                        src.matrix_rows(&p("self_attn.kv_b_proj.weight"), start, mla.v_head)
                    })
                    .collect::<Res<Vec<_>>>()?,
                o: src.matrix(&p("self_attn.o_proj.weight"))?,
            };

            let is_moe = router.n_experts > 0 && i >= first_dense && i % moe_every == 0;
            let mlp = match is_moe {
                false => Mlp::Dense(Ffn::load(src, &p("mlp"))?),
                true => Mlp::Moe(Moe {
                    gate: src.matrix(&p("mlp.gate.weight"))?,
                    bias: src.try_vector(&p("mlp.gate.e_score_correction_bias")),
                    experts: (0..router.n_experts)
                        .map(|e| Ffn::load(src, &p(&format!("mlp.experts.{e}"))))
                        .collect::<Res<Vec<_>>>()?,
                    shared: match n_shared {
                        0 => None,
                        _ => Some(Ffn::load(src, &p("mlp.shared_experts"))?),
                    },
                }),
            };

            blocks.push(Block {
                attn,
                mlp_norm: src.vector(&p("post_attention_layernorm.weight"))?,
                mlp,
            });
        }

        let lm_head = match spec.tie_embeddings {
            true => None,
            false => src.try_matrix("lm_head.weight"),
        };
        let rope = build_rope(mla.qk_rope, spec.n_ctx, spec.rope_theta, &spec.config)?;

        Ok(Model {
            embed: src.matrix("embed_tokens.weight")?,
            lm_head,
            blocks,
            final_norm: src.vector("norm.weight")?,
            mla,
            router,
            rope,
            spec,
        })
    }

    /// Every block over a batch of tokens, leaving the residual stream as it
    /// is — no output head. Both public entry points are this plus a norm and
    /// a projection, over different rows.
    fn run_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let (m, e) = (tokens.len(), spec.n_embd);
        let pos0 = cache.len;
        assert!(
            pos0 + m <= spec.n_ctx,
            "context window of {} tokens is full",
            spec.n_ctx
        );

        let mut xs = vec![0.0f32; m * e];
        for (i, &t) in tokens.iter().enumerate() {
            xs[i * e..(i + 1) * e].copy_from_slice(&self.embed.row(t as usize));
        }

        let mut hs = vec![0.0f32; m * e];
        for (l, block) in self.blocks.iter().enumerate() {
            for i in 0..m {
                hs[i * e..(i + 1) * e].copy_from_slice(&rms_norm(
                    &xs[i * e..(i + 1) * e],
                    &block.attn.norm,
                    spec.eps,
                ));
            }
            let attn = self.attend(l, &block.attn, &hs, m, pos0, cache);
            for (x, a) in xs.iter_mut().zip(attn.iter()) {
                *x += a;
            }

            for i in 0..m {
                hs[i * e..(i + 1) * e].copy_from_slice(&rms_norm(
                    &xs[i * e..(i + 1) * e],
                    &block.mlp_norm,
                    spec.eps,
                ));
            }
            let out = match &block.mlp {
                Mlp::Dense(ffn) => ffn.run(&hs, m),
                Mlp::Moe(moe) => self.moe(moe, &hs, m, e),
            };
            for (x, v) in xs.iter_mut().zip(out.iter()) {
                *x += v;
            }
        }

        cache.len += m;
        xs
    }

    /// Latent attention for a batch of positions.
    fn attend(
        &self,
        layer: usize,
        w: &Attn,
        hs: &[f32],
        m: usize,
        pos0: usize,
        cache: &mut KvCache,
    ) -> Vec<f32> {
        let mla = &self.mla;
        let (qh, nope, rope_dim) = (mla.q_head(), mla.qk_nope, mla.qk_rope);
        let (lat, vh) = (mla.kv_lora, mla.v_head);

        // ---- Queries -------------------------------------------------------
        let mut q = match &w.q {
            Query::Direct(proj) => proj.matmul_bt(hs, m, None),
            Query::Compressed { down, norm, up } => {
                let mut mid = down.matmul_bt(hs, m, None);
                let rank = down.rows();
                for i in 0..m {
                    let row = rms_norm(&mid[i * rank..(i + 1) * rank], norm, self.spec.eps);
                    mid[i * rank..(i + 1) * rank].copy_from_slice(&row);
                }
                up.matmul_bt(&mid, m, None)
            }
        };
        // Position enters only the last `qk_rope` numbers of each head.
        for i in 0..m {
            for h in 0..mla.n_head {
                let at = i * mla.n_head * qh + h * qh + nope;
                self.rope
                    .apply_interleaved(&mut q[at..at + rope_dim], pos0 + i);
            }
        }

        // ---- Compress this batch's positions into the cache -----------------
        let kv = w.kv_a.matmul_bt(hs, m, None);
        let wide = lat + rope_dim;
        for i in 0..m {
            let row = &kv[i * wide..(i + 1) * wide];
            // The norm goes on before the cache, not after: what is stored is
            // what the up-projections expect to be handed.
            let c = rms_norm(&row[..lat], &w.kv_a_norm, self.spec.eps);
            let mut k_pe = row[lat..].to_vec();
            self.rope.apply_interleaved(&mut k_pe, pos0 + i);
            cache.push(layer, &k_pe, &c);
        }
        let (k_pe, cs) = (cache.keys(layer), cache.values(layer));

        // ---- Score, one head at a time --------------------------------------
        //
        // Laid out head-major so each rayon task owns a contiguous block; the
        // transpose back into position-major is a copy of a few thousand
        // floats and costs nothing next to the matmuls.
        let mut by_head = vec![0.0f32; mla.n_head * m * vh];
        by_head
            .par_chunks_mut(m * vh)
            .enumerate()
            .for_each(|(h, dst)| {
                let mut scores = Vec::with_capacity(pos0 + m);
                for i in 0..m {
                    let head = &q[i * mla.n_head * qh + h * qh..][..qh];
                    // The absorption: the head's key matrix, applied to the query
                    // once, instead of to every cached position.
                    let q_lat = w.uk[h].matvec_bt(&head[..nope], None);
                    let q_pe = &head[nope..];

                    let n = pos0 + i + 1;
                    scores.clear();
                    for t in 0..n {
                        let c = &cs[t * lat..(t + 1) * lat];
                        let mut dot = 0.0f32;
                        for j in 0..lat {
                            dot += q_lat[j] * c[j];
                        }
                        let pe = &k_pe[t * rope_dim..(t + 1) * rope_dim];
                        for j in 0..rope_dim {
                            dot += q_pe[j] * pe[j];
                        }
                        scores.push(dot * mla.softmax_scale);
                    }
                    softmax_inplace(&mut scores);

                    // Average the *compressed* vectors, then decompress once.
                    let mut acc = vec![0.0f32; lat];
                    for (t, &weight) in scores.iter().enumerate() {
                        if weight < 1e-8 {
                            continue;
                        }
                        let c = &cs[t * lat..(t + 1) * lat];
                        for j in 0..lat {
                            acc[j] += weight * c[j];
                        }
                    }
                    dst[i * vh..(i + 1) * vh].copy_from_slice(&w.uv[h].matvec_bt(&acc, None));
                }
            });

        let mut heads = vec![0.0f32; m * mla.n_head * vh];
        for h in 0..mla.n_head {
            for i in 0..m {
                let src = &by_head[h * m * vh + i * vh..][..vh];
                heads[i * mla.n_head * vh + h * vh..][..vh].copy_from_slice(src);
            }
        }
        w.o.matmul_bt(&heads, m, None)
    }

    /// The mixture, over a batch.
    ///
    /// Grouped by expert rather than by token. Every token picks its own
    /// handful, so a batch of sixty-four touches most of the sixty-four
    /// experts however you slice it — but walking tokens would read each
    /// chosen expert's weights again for every token that chose it, and
    /// walking experts reads each one once.
    fn moe(&self, moe: &Moe, hs: &[f32], m: usize, e: usize) -> Vec<f32> {
        let n = self.router.n_experts;
        let logits = moe.gate.matmul_bt(hs, m, None);

        let mut picks = Vec::with_capacity(self.router.top_k);
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for i in 0..m {
            self.router
                .route(&logits[i * n..(i + 1) * n], moe.bias.as_deref(), &mut picks);
            for &(expert, weight) in &picks {
                by_expert[expert].push((i, weight));
            }
        }

        let mut out = match &moe.shared {
            // Every token, so it is one batched pass and needs no grouping.
            Some(shared) => shared.run(hs, m),
            None => vec![0.0f32; m * e],
        };
        let mut rows = Vec::with_capacity(m * e);
        for (expert, tokens) in by_expert.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            rows.clear();
            for &(i, _) in tokens {
                rows.extend_from_slice(&hs[i * e..(i + 1) * e]);
            }
            let y = moe.experts[expert].run(&rows, tokens.len());
            for (j, &(i, weight)) in tokens.iter().enumerate() {
                for (o, v) in out[i * e..(i + 1) * e]
                    .iter_mut()
                    .zip(&y[j * e..(j + 1) * e])
                {
                    *o += weight * v;
                }
            }
        }
        out
    }

    fn head(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }
}

impl Transformer for Model {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        self.forward_batch(&[token], cache)
    }

    fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let (m, e) = (tokens.len(), self.spec.n_embd);
        let xs = self.run_batch(tokens, cache);
        let last = rms_norm(&xs[(m - 1) * e..m * e], &self.final_norm, self.spec.eps);
        self.head().matvec_bt(&last, None)
    }

    fn forward_batch_all(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let (m, e) = (tokens.len(), self.spec.n_embd);
        let mut xs = self.run_batch(tokens, cache);
        for i in 0..m {
            let normed = rms_norm(&xs[i * e..(i + 1) * e], &self.final_norm, self.spec.eps);
            xs[i * e..(i + 1) * e].copy_from_slice(&normed);
        }
        self.head().matmul_bt(&xs, m, None)
    }

    fn param_count(&self) -> usize {
        self.embed.param_count()
            + self.lm_head.as_ref().map_or(0, |h| h.param_count())
            + self.blocks.iter().map(|b| b.param_count()).sum::<usize>()
    }

    fn memory_bytes(&self) -> usize {
        self.embed.bytes()
            + self.lm_head.as_ref().map_or(0, |h| h.bytes())
            + self.blocks.iter().map(|b| b.bytes()).sum::<usize>()
    }
}

impl Block {
    fn param_count(&self) -> usize {
        let a = &self.attn;
        let attn = match &a.q {
            Query::Direct(w) => w.param_count(),
            Query::Compressed { down, norm, up } => {
                down.param_count() + norm.len() + up.param_count()
            }
        } + a.kv_a.param_count()
            + a.kv_a_norm.len()
            + a.uk.iter().map(|w| w.param_count()).sum::<usize>()
            + a.uv.iter().map(|w| w.param_count()).sum::<usize>()
            + a.o.param_count()
            + a.norm.len();
        attn + self.mlp_norm.len() + self.mlp.param_count()
    }

    fn bytes(&self) -> usize {
        let a = &self.attn;
        let attn = match &a.q {
            Query::Direct(w) => w.bytes(),
            Query::Compressed { down, up, .. } => down.bytes() + up.bytes(),
        } + a.kv_a.bytes()
            + a.uk.iter().map(|w| w.bytes()).sum::<usize>()
            + a.uv.iter().map(|w| w.bytes()).sum::<usize>()
            + a.o.bytes();
        attn + self.mlp.bytes()
    }
}

impl Mlp {
    fn param_count(&self) -> usize {
        match self {
            Mlp::Dense(f) => f.param_count(),
            Mlp::Moe(m) => {
                m.gate.param_count()
                    + m.bias.as_ref().map_or(0, |b| b.len())
                    + m.experts.iter().map(|f| f.param_count()).sum::<usize>()
                    + m.shared.as_ref().map_or(0, |f| f.param_count())
            }
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Mlp::Dense(f) => f.bytes(),
            Mlp::Moe(m) => {
                m.gate.bytes()
                    + m.experts.iter().map(|f| f.bytes()).sum::<usize>()
                    + m.shared.as_ref().map_or(0, |f| f.bytes())
            }
        }
    }
}
