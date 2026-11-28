//! The Llama family (2023 onwards): Llama 2/3, Mistral and Ministral 3, Qwen2/2.5, SmolLM2,
//! TinyLlama, Apertus.
//!
//! Read this as a diff against [`super::gpt2`]. The skeleton is identical —
//! two residual sub-blocks per layer, attention then MLP — and every part
//! inside it was replaced:
//!
//! | GPT-2 | here | effect |
//! |---|---|---|
//! | learned position rows (`wpe`) | RoPE, applied to Q and K | no hard context ceiling |
//! | LayerNorm (centre, scale, bias) | RMSNorm (scale only) | cheaper, no bias vectors |
//! | GELU MLP, 2 matrices | SwiGLU, 3 matrices | a learned gate per channel |
//! | multi-head attention | grouped-query attention | KV cache shrinks by the group factor |
//! | one fused QKV matrix | three separate projections | Q and KV have different widths now |
//!
//! There is one more difference, invisible in any diagram: GPT-2's checkpoint
//! was written for a `Conv1D` layer, which stores weights as
//! `[in_features, out_features]`. Everything here uses `nn.Linear`, which
//! stores them the other way round. Rather than carry two matmul kernels, the
//! GPT-2 loader transposes its projection matrices on the way in, so both
//! architectures run the same code — and GPT-2 gets the quantised integer
//! path for free. Mixing the two layouts up produces fluent nonsense rather
//! than an error, which is why the counting test in the README exists.

use super::ffn::{Ffn, Layout, Mlp, Moe, Router};
use super::{attend, build_rope, llama_freqs, Architecture, KvCache, Shard, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::{rms_norm, Rope};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `y = x @ Wᵀ (+ b)`, the HuggingFace `nn.Linear` convention.
///
/// The bias is optional because most of this family dropped biases entirely —
/// they cost parameters and buy nothing once you have normalisation. Qwen2 is
/// the exception: it keeps biases on Q, K and V but nowhere else.
fn linear(x: &[f32], w: &Weight, bias: Option<&Vec<f32>>) -> Vec<f32> {
    w.matvec_bt(x, bias.map(|b| b.as_slice()))
}

/// This module's entry in the registry.
///
/// Five `model_type`s, one implementation. They differ in configuration —
/// widths, head counts, rope base — except for two things that are branches in
/// this file: Qwen3's per-head RMSNorm on Q and K, applied when the weights for
/// it are present, and `qwen3_moe`'s feed-forward, which is a router and 128
/// experts where the others have one MLP. Its attention is the `qwen3` block
/// unchanged, which is the whole reason it belongs here rather than in a file
/// of its own.
///
/// The list is exact on purpose. `qwen2_moe` has a shared expert and a
/// different gate and is not this; `smollm3` skips RoPE on every fourth layer
/// and is not this either, though a substring test happily said both were.
pub static ARCH: Architecture = Architecture {
    id: "llama",
    model_types: &["llama", "mistral", "qwen2", "qwen3", "qwen3_moe"],
    about: "Llama 2/3, Mistral, Qwen2/2.5/3, Qwen3-MoE, SmolLM2: RoPE, RMSNorm, SwiGLU, GQA",
    configure: |_| Ok(()),
    splits: true,
    load: |src, spec, shard| Ok(Box::new(Model::load_shard(src, spec, shard)?)),
};

/// Ministral 3 and Mistral 3, text only: this file's Mistral with two
/// changes, both read from `rope_parameters`.
///
/// - **YaRN** stretches the rotation over 16× the 16k it was trained on, as
///   DeepSeek's does ([`build_rope`]).
/// - **A query temperature** past that length: each query is multiplied by
///   `1 + β·ln(1 + ⌊pos / 16384⌋)`, Llama 4's trick for keeping attention
///   sharp at long range. It is 1 for the first 16k positions.
///
/// It ships as `Mistral3ForConditionalGeneration`, a Pixtral vision tower and
/// projector beside the language model, which are skipped by name.
///
/// An entry of its own rather than two more names on [`ARCH`], because the
/// GPU backend runs `llama` and knows neither change: it would run this as
/// plain Mistral and answer fluently and wrong past the first few thousand
/// positions.
pub static MISTRAL3: Architecture = Architecture {
    id: "mistral3",
    model_types: &["mistral3", "ministral3"],
    about: "Ministral 3, text only: Mistral with YaRN RoPE and Llama 4's long-range query scale",
    configure: |_| Ok(()),
    splits: true,
    load: |src, spec, shard| Ok(Box::new(Model::load_shard(src, spec, shard)?)),
};

/// Apertus (Swiss AI, 2025): Qwen3's attention, per-head norms and all,
/// under Llama 3.1's stretched rotation, and an MLP with no gate whose
/// activation is xIELU, learned per layer ([`super::ffn::Xielu`]). Its two
/// norms are named for what they precede, `attention_layernorm` and
/// `feedforward_layernorm`.
///
/// An entry of its own for the same reason as [`MISTRAL3`]: the GPU backend
/// runs `llama`, and has neither the rotation nor the activation.
pub static APERTUS: Architecture = Architecture {
    id: "apertus",
    model_types: &["apertus"],
    about: "Apertus: Llama with per-head QK norms, llama3 RoPE and an ungated xIELU MLP",
    configure: |_| Ok(()),
    splits: true,
    load: |src, spec, shard| Ok(Box::new(Model::load_shard(src, spec, shard)?)),
};

pub struct Block {
    attn_norm: Vec<f32>,
    q_w: Weight,
    q_b: Option<Vec<f32>>,
    k_w: Weight,
    k_b: Option<Vec<f32>>,
    v_w: Weight,
    v_b: Option<Vec<f32>>,
    o_w: Weight,
    /// Qwen3's per-head RMSNorm on the queries and the keys, applied after
    /// the projection and *before* RoPE. One vector of `head_dim`, shared by
    /// every head — so it normalises each head's slice independently rather
    /// than the whole projection at once.
    ///
    /// Absent in Llama and Qwen2, which is the only difference between them
    /// and Qwen3 that reaches this file.
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    mlp_norm: Vec<f32>,
    /// One MLP, or a routed hundred of them. See [`super::ffn`]: what a block
    /// feeds its attention through is a choice this family makes independently
    /// of how it attends, and `qwen3_moe` is the same block as `qwen3` with
    /// the other answer here.
    mlp: Mlp,
}

pub struct Model {
    spec: Spec,
    /// Which of the model's layers `blocks` holds, and whether `ends` is
    /// here. Everything and yes, for a model on one machine.
    shard: Shard,
    ends: Option<Ends>,
    blocks: Vec<Block>,
    rope: Rope,
    /// Ministral 3's query temperature, `(β, trained length)`. See
    /// [`MISTRAL3`] and [`query_scale`].
    q_scale: Option<(f32, usize)>,
}

/// What Ministral 3 multiplies a query at `pos` by: `1 + β·ln(1 + ⌊pos / n⌋)`.
fn query_scale(beta: f32, trained: usize, pos: usize) -> f32 {
    1.0 + beta * ((1 + pos / trained) as f32).ln()
}

/// The two ends of the model, which stay together when the blocks between
/// them are spread out: see [`Shard`].
struct Ends {
    embed: Weight,
    /// Separate output projection, when the model does not tie its embeddings.
    /// Small models almost always tie (it is a large fraction of their
    /// parameters); large ones often do not.
    lm_head: Option<Weight>,
    final_norm: Vec<f32>,
}

/// RMSNorm every head of a projection in place, if this model has the weights
/// for it.
///
/// Qwen3's addition to the Llama block, and the whole of it. The weight is one
/// vector of `head_dim` reused across heads, so this normalises each head's
/// own slice — which is the point: it bounds the magnitude of each head's
/// query and key independently before they meet in a dot product, and lets the
/// model be trained without the attention logits drifting apart between heads.
///
/// `None` is every other model in this family, where it costs one branch per
/// layer.
fn norm_heads(x: &mut [f32], weight: Option<&[f32]>, spec: &Spec) {
    let Some(weight) = weight else { return };
    for head in x.chunks_mut(spec.head_dim) {
        head.copy_from_slice(&rms_norm(head, weight, spec.eps));
    }
}

impl Model {
    /// The whole model.
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let whole = Shard::whole(&spec);
        Model::load_shard(src, spec, &whole)
    }

    /// The part of the model `shard` names, reading that and nothing else.
    /// What it leaves to the other shards it says it is leaving, so the
    /// check for weights nobody read keeps meaning what it means.
    pub fn load_shard(src: &dyn Source, spec: Spec, shard: &Shard) -> Res<Self> {
        // Dense unless the config says otherwise, and what says otherwise is an
        // expert count. `qwen3_moe` is the only `model_type` here that has one;
        // reading it this way rather than off `model_type` means the branch is
        // about what the checkpoint holds, which is the only thing the loader
        // can actually check.
        let routed = match Router::count(&spec.config) {
            None => None,
            Some(_) => Some((Router::read(&spec.config)?, Layout::read(&spec.config))),
        };

        // Mistral 3's decoder sits under `language_model.model.`, its head under
        // `language_model.`, beside a vision tower this engine does not run.
        // Asked of the checkpoint rather than the config, whose inner
        // `ministral3` is also what a text-only checkpoint calls itself.
        let wrapped = spec.arch.is("mistral3") && src.try_vector("language_model.model.norm.weight").is_some();
        let (pre, head_name) = match wrapped {
            true => {
                for stem in ["vision_tower.", "multi_modal_projector.", "model.vision_tower.", "model.multi_modal_projector."] {
                    src.skip_under(stem);
                }
                ("language_model.model.", "language_model.lm_head.weight")
            }
            false => ("", "lm_head.weight"),
        };

        let apertus = spec.arch.is("apertus");
        let (attn_norm, mlp_norm) = match apertus {
            true => ("attention_layernorm.weight", "feedforward_layernorm.weight"),
            false => ("input_layernorm.weight", "post_attention_layernorm.weight"),
        };

        let mut blocks = Vec::with_capacity(shard.layers.len());
        for i in 0..spec.n_layer {
            // Another shard's layer: known about, and deliberately not read.
            if !shard.layers.contains(&i) {
                src.skip_under(&format!("{pre}layers.{i}."));
                continue;
            }
            let p = |s: &str| format!("{pre}layers.{i}.{s}");
            // An fp8 checkpoint's static activation scales: for running the
            // matmul in fp8, which this engine does not, since it widens the
            // weight on the way in.
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                src.skip(&p(&format!("{proj}.activation_scale")));
            }
            blocks.push(Block {
                attn_norm: src.vector(&p(attn_norm))?,
                q_w: src.matrix(&p("self_attn.q_proj.weight"))?,
                q_b: src.try_vector(&p("self_attn.q_proj.bias")),
                k_w: src.matrix(&p("self_attn.k_proj.weight"))?,
                k_b: src.try_vector(&p("self_attn.k_proj.bias")),
                v_w: src.matrix(&p("self_attn.v_proj.weight"))?,
                v_b: src.try_vector(&p("self_attn.v_proj.bias")),
                o_w: src.matrix(&p("self_attn.o_proj.weight"))?,
                q_norm: src.try_vector(&p("self_attn.q_norm.weight")),
                k_norm: src.try_vector(&p("self_attn.k_norm.weight")),
                mlp_norm: src.vector(&p(mlp_norm))?,
                mlp: match &routed {
                    Some((router, layout)) if layout.is_moe(i, router.n_experts) => {
                        Mlp::Moe(Box::new(Moe::load(src, &p("mlp"), router, layout.shared)?))
                    }
                    _ if apertus => Mlp::Dense(Ffn::load_xielu(src, &p("mlp"))?),
                    _ => Mlp::Dense(Ffn::load(src, &p("mlp"))?),
                },
            });
        }

        let ends = match shard.ends {
            true => Some(Ends {
                lm_head: head(src, &spec, head_name)?,
                embed: src.matrix(&format!("{pre}embed_tokens.weight"))?,
                final_norm: src.vector(&format!("{pre}norm.weight"))?,
            }),
            false => {
                for name in [head_name.to_string(), format!("{pre}embed_tokens.weight"), format!("{pre}norm.weight")] {
                    src.skip(&name);
                }
                None
            }
        };
        let rope = match spec.arch.is("mistral3") {
            true => build_rope(spec.head_dim, spec.n_ctx, spec.rope_theta, &spec.config)?,
            // Precompute every rotation angle once. Cheap: two floats per
            // position per coordinate pair. Llama 3.1's stretch, and
            // Apertus's, when the config asks for it.
            false => Rope::from_freqs(&llama_freqs(spec.head_dim, spec.rope_theta, &spec.config), spec.n_ctx, 1.0),
        };
        let q_scale = spec.arch.is("mistral3").then(|| {
            let p = spec.config.get("rope_parameters")?;
            let beta = p.get("llama_4_scaling_beta")?.as_f64()? as f32;
            let trained = p.get("original_max_position_embeddings")?.as_u64()? as usize;
            Some((beta, trained.max(1)))
        }).flatten();

        Ok(Model {
            shard: shard.clone(),
            ends,
            blocks,
            rope,
            q_scale,
            spec,
        })
    }

    /// Ministral 3's long-range query temperature, after the rotation as the
    /// reference has it. Nothing for every other model, and nothing for
    /// Ministral 3's first 16k positions either.
    fn scale_query(&self, q: &mut [f32], pos: usize) {
        let Some((beta, trained)) = self.q_scale else { return };
        let s = query_scale(beta, trained, pos);
        if s != 1.0 {
            q.iter_mut().for_each(|v| *v *= s);
        }
    }

    /// Rotate each head of a projection in place.
    fn apply_rope(&self, x: &mut [f32], pos: usize) {
        for head in x.chunks_mut(self.spec.head_dim) {
            self.rope.apply(head, pos);
        }
    }
    /// The embedding and the head, on the shard that has them.
    fn ends(&self) -> &Ends {
        self.ends.as_ref().unwrap_or_else(|| panic!("{} of this model were asked for its ends, which another shard holds", self.shard))
    }

    /// This shard's blocks over `m` rows of the residual stream, in place,
    /// for rows that are either one sequence's next positions or several
    /// sequences' next position each; see [`Rows`].
    ///
    /// One body for prefill and for batched decode, because between the
    /// lines that touch a cache they are the same arithmetic: `m` rows
    /// through each projection and each MLP as one matmul. And no embedding
    /// before it or head after it, because what goes in and what comes out
    /// is the residual stream, wherever it came from and is going.
    fn run_rows(&self, xs: &mut [f32], m: usize, mut rows: Rows) {
        let spec = &self.spec;
        let e = spec.n_embd;
        let qdim = spec.n_head * spec.head_dim;
        let kvdim = spec.kv_dim();
        // Where each row sits in its own sequence, fixed before anything is
        // pushed: a cache's length does not move until the pass is over.
        let at: Vec<usize> = (0..m).map(|i| rows.position(i)).collect();
        assert!(
            at.iter().all(|&pos| pos < spec.n_ctx),
            "context window of {} tokens is full",
            spec.n_ctx
        );

        let mut hs = vec![0.0f32; m * e];
        for (l, block) in self.blocks.iter().enumerate() {
            // ---- Attention sub-block -------------------------------------
            for i in 0..m {
                hs[i * e..(i + 1) * e]
                    .copy_from_slice(&rms_norm(&xs[i * e..(i + 1) * e], &block.attn_norm, spec.eps));
            }
            let mut q = block.q_w.matmul_bt(&hs, m, block.q_b.as_deref());
            let mut k = block.k_w.matmul_bt(&hs, m, block.k_b.as_deref());
            let v = block.v_w.matmul_bt(&hs, m, block.v_b.as_deref());

            // Each row rotates by its own absolute position.
            for i in 0..m {
                norm_heads(&mut q[i * qdim..(i + 1) * qdim], block.q_norm.as_deref(), spec);
                norm_heads(&mut k[i * kvdim..(i + 1) * kvdim], block.k_norm.as_deref(), spec);
                self.apply_rope(&mut q[i * qdim..(i + 1) * qdim], at[i]);
                self.apply_rope(&mut k[i * kvdim..(i + 1) * kvdim], at[i]);
                self.scale_query(&mut q[i * qdim..(i + 1) * qdim], at[i]);
                rows.cache(i).push(l, &k[i * kvdim..(i + 1) * kvdim], &v[i * kvdim..(i + 1) * kvdim]);
            }

            // Each row attends over its own sequence up to itself. For one
            // sequence that is the causal mask, for nothing; for several it
            // is what keeps them apart, for nothing as well.
            let mut attn = vec![0.0f32; m * qdim];
            for i in 0..m {
                let cache = rows.cache(i);
                attn[i * qdim..(i + 1) * qdim].copy_from_slice(&attend(
                    spec,
                    &q[i * qdim..(i + 1) * qdim],
                    cache.keys(l),
                    cache.values(l),
                    at[i] + 1,
                ));
            }
            let proj = block.o_w.matmul_bt(&attn, m, None);
            for (x, a) in xs.iter_mut().zip(proj.iter()) {
                *x += a;
            }

            // ---- MLP sub-block -------------------------------------------
            for i in 0..m {
                hs[i * e..(i + 1) * e]
                    .copy_from_slice(&rms_norm(&xs[i * e..(i + 1) * e], &block.mlp_norm, spec.eps));
            }
            let mlp = block.mlp.run(&hs, m, e, self.blocks.get(l + 1).map(|b| &b.mlp));
            for (x, val) in xs.iter_mut().zip(mlp.iter()) {
                *x += val;
            }
        }

        rows.advance(m);
    }
}

/// Whose positions the rows of a pass are.
///
/// Prefill and batched decode run the same rows through the same weights and
/// differ only in this: where a row's keys and values go, and what it may
/// attend to.
enum Rows<'a, 'b> {
    /// One sequence's next positions, in order: prefill.
    Following(&'a mut KvCache),
    /// The next position of each of several sequences: decode, batched.
    Each(&'a mut [&'b mut KvCache]),
}

impl Rows<'_, '_> {
    /// The cache row `i` belongs to.
    fn cache(&mut self, i: usize) -> &mut KvCache {
        match self {
            Rows::Following(cache) => cache,
            Rows::Each(caches) => caches[i],
        }
    }

    /// Where row `i` sits in its sequence, before the pass has counted it.
    fn position(&self, i: usize) -> usize {
        match self {
            Rows::Following(cache) => cache.len + i,
            Rows::Each(caches) => caches[i].len,
        }
    }

    /// Count the pass's `m` rows as held.
    fn advance(&mut self, m: usize) {
        match self {
            Rows::Following(cache) => cache.len += m,
            Rows::Each(caches) => caches.iter_mut().for_each(|c| c.len += 1),
        }
    }
}

impl Transformer for Model {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn param_count(&self) -> usize {
        // The optional vectors count too, and they are where a family differs:
        // Qwen2 has the attention biases and Qwen3 has the per-head norms. A
        // count that skipped them would come out the same whether the model had
        // them or not — the uncounted output head again, far smaller and
        // just as wrong.
        let opt = |v: &Option<Vec<f32>>| v.as_ref().map_or(0, Vec::len);
        let per_block = self.blocks.first().map_or(0, |b| {
            b.q_w.param_count()
                + b.k_w.param_count()
                + b.v_w.param_count()
                + b.o_w.param_count()
                + b.mlp.param_count()
                + b.attn_norm.len()
                + b.mlp_norm.len()
                + opt(&b.q_b)
                + opt(&b.k_b)
                + opt(&b.v_b)
                + opt(&b.q_norm)
                + opt(&b.k_norm)
        });
        let ends = self.ends.as_ref().map_or(0, |ends| {
            ends.embed.param_count() + ends.lm_head.as_ref().map_or(0, |h| h.param_count()) + ends.final_norm.len()
        });
        ends + per_block * self.blocks.len()
    }

    fn memory_bytes(&self) -> usize {
        let per_block: usize = self.blocks.first().map_or(0, |b| {
            b.q_w.bytes()
                + b.k_w.bytes()
                + b.v_w.bytes()
                + b.o_w.bytes()
                + b.mlp.bytes()
        });
        let ends = self.ends.as_ref().map_or(0, |ends| ends.embed.bytes() + ends.lm_head.as_ref().map_or(0, |h| h.bytes()));
        ends + per_block * self.blocks.len()
    }

    fn shard(&self) -> Shard {
        self.shard.clone()
    }

    fn splits(&self) -> bool {
        true
    }

    /// A row of the table for each token, and nothing else: position
    /// arrives inside attention, as RoPE.
    fn embed(&self, tokens: &[u32], _pos0: usize) -> Vec<f32> {
        let (e, embed) = (self.spec.n_embd, &self.ends().embed);
        let mut xs = vec![0.0f32; tokens.len() * e];
        for (i, &t) in tokens.iter().enumerate() {
            xs[i * e..(i + 1) * e].copy_from_slice(&embed.row(t as usize));
        }
        xs
    }

    fn blocks(&self, xs: &mut [f32], m: usize, cache: &mut KvCache) {
        self.run_rows(xs, m, Rows::Following(cache))
    }

    /// The final norm and the projection back to the vocabulary.
    ///
    /// Generation wants the last position's logits and nothing else, so
    /// that stays a matrix-vector product. Scoring text — perplexity —
    /// wants every position's, and the residual stream already holds them
    /// all: the head over `m` rows as one matmul costs a fraction of the
    /// `m` forward passes it would otherwise take.
    fn head(&self, xs: &[f32], m: usize, all: bool) -> Vec<f32> {
        let (e, ends) = (self.spec.n_embd, self.ends());
        let table = ends.lm_head.as_ref().unwrap_or(&ends.embed);
        if !all {
            let last = rms_norm(&xs[(m - 1) * e..m * e], &ends.final_norm, self.spec.eps);
            return table.matvec_bt(&last, None);
        }
        let mut normed = vec![0.0f32; m * e];
        for i in 0..m {
            normed[i * e..(i + 1) * e].copy_from_slice(&rms_norm(&xs[i * e..(i + 1) * e], &ends.final_norm, self.spec.eps));
        }
        table.matmul_bt(&normed, m, None)
    }

    /// One matmul per projection for all of them, and per-sequence
    /// attention: the same pass as prefill, with a cache a row.
    fn forward_each(&self, tokens: &[u32], caches: &mut [&mut KvCache]) -> Vec<f32> {
        assert_eq!(tokens.len(), caches.len(), "a token for each sequence, and a sequence for each token");
        let m = tokens.len();
        match m {
            0 => return Vec::new(),
            // One sequence is the ordinary step, and the ordinary step's
            // kernels are the ones written for a single row: on a small
            // model the batched ones measured 7% slower at one.
            1 => return self.forward(tokens[0], caches[0]),
            _ => {}
        }
        let mut xs = self.embed(tokens, 0);
        self.run_rows(&mut xs, m, Rows::Each(caches));
        self.head(&xs, m, true)
    }

    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let pos = cache.len;
        assert!(pos < spec.n_ctx, "context window of {} tokens is full", spec.n_ctx);

        // No position embedding here. The token vector is all that enters the
        // residual stream; position arrives later, inside attention.
        let ends = self.ends();
        let mut x = ends.embed.row(token as usize);

        for (l, block) in self.blocks.iter().enumerate() {
            // ---- Attention sub-block -------------------------------------
            let h = rms_norm(&x, &block.attn_norm, spec.eps);

            // Three projections rather than one fused matrix: under grouped
            // query attention Q is wider than K and V, so they cannot share.
            let mut q = linear(&h, &block.q_w, block.q_b.as_ref());
            let mut k = linear(&h, &block.k_w, block.k_b.as_ref());
            let v = linear(&h, &block.v_w, block.v_b.as_ref());

            // Qwen3 normalises each head of Q and K here, between the
            // projection and the rotation. Nothing else does, and for
            // everything else this is a no-op.
            norm_heads(&mut q, block.q_norm.as_deref(), spec);
            norm_heads(&mut k, block.k_norm.as_deref(), spec);

            // Position enters here, as a rotation of Q and K -- and nowhere
            // else. V is left alone: it carries content, not location.
            //
            // The keys go into the cache already rotated, which is what makes
            // this cheap: each key is rotated once, when it is first computed,
            // not re-rotated on every later step.
            self.apply_rope(&mut q, pos);
            self.apply_rope(&mut k, pos);
            self.scale_query(&mut q, pos);

            cache.push(l, &k, &v);
            let attn = attend(spec, &q, cache.keys(l), cache.values(l), pos + 1);
            let attn = linear(&attn, &block.o_w, None);
            for (xi, a) in x.iter_mut().zip(attn.iter()) {
                *xi += a; // residual
            }

            // ---- MLP sub-block -------------------------------------------
            let h = rms_norm(&x, &block.mlp_norm, spec.eps);
            let mlp = block.mlp.run_one(&h, spec.n_embd, self.blocks.get(l + 1).map(|b| &b.mlp));
            for (xi, m) in x.iter_mut().zip(mlp.iter()) {
                *xi += m; // residual
            }
        }

        cache.len += 1;

        let x = rms_norm(&x, &ends.final_norm, spec.eps);
        ends.lm_head.as_ref().unwrap_or(&ends.embed).matvec_bt(&x, None)
    }
}
