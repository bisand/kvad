//! The Llama family (2023 onwards): Llama 2/3, Mistral, Qwen2/2.5, SmolLM2,
//! TinyLlama.
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

use super::{attend, Architecture, KvCache, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::{rms_norm, swiglu_inplace, Rope};

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
/// Four `model_type`s, one implementation. They differ in configuration —
/// widths, head counts, rope base — except for Qwen3's per-head RMSNorm on Q
/// and K, which is applied when the weights for it are present and is the only
/// branch in this file that is about *which* model is running.
///
/// The list is exact on purpose. `qwen2_moe` and `qwen3_5_moe` are mixtures of
/// experts and are not this; `smollm3` skips RoPE on every fourth layer and is
/// not this either, though a substring test happily said all three were.
pub static ARCH: Architecture = Architecture {
    id: "llama",
    model_types: &["llama", "mistral", "qwen2", "qwen3"],
    about: "Llama 2/3, Mistral, Qwen2/2.5/3, SmolLM2, TinyLlama: RoPE, RMSNorm, SwiGLU, GQA",
    configure: |_| Ok(()),
    load: |src, spec| Ok(Box::new(Model::load(src, spec)?)),
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
    /// The gate and the value are computed by two separate matrices from the
    /// same input; `down` projects their product back to `n_embd`.
    gate_w: Weight,
    up_w: Weight,
    down_w: Weight,
}

pub struct Model {
    spec: Spec,
    embed: Weight,
    /// Separate output projection, when the model does not tie its embeddings.
    /// Small models almost always tie (it is a large fraction of their
    /// parameters); large ones often do not.
    lm_head: Option<Weight>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    rope: Rope,
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
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            blocks.push(Block {
                attn_norm: src.vector(&p("input_layernorm.weight"))?,
                q_w: src.matrix(&p("self_attn.q_proj.weight"))?,
                q_b: src.try_vector(&p("self_attn.q_proj.bias")),
                k_w: src.matrix(&p("self_attn.k_proj.weight"))?,
                k_b: src.try_vector(&p("self_attn.k_proj.bias")),
                v_w: src.matrix(&p("self_attn.v_proj.weight"))?,
                v_b: src.try_vector(&p("self_attn.v_proj.bias")),
                o_w: src.matrix(&p("self_attn.o_proj.weight"))?,
                q_norm: src.try_vector(&p("self_attn.q_norm.weight")),
                k_norm: src.try_vector(&p("self_attn.k_norm.weight")),
                mlp_norm: src.vector(&p("post_attention_layernorm.weight"))?,
                gate_w: src.matrix(&p("mlp.gate_proj.weight"))?,
                up_w: src.matrix(&p("mlp.up_proj.weight"))?,
                down_w: src.matrix(&p("mlp.down_proj.weight"))?,
            });
        }

        let lm_head = head(src, &spec, "lm_head.weight")?;

        Ok(Model {
            embed: src.matrix("embed_tokens.weight")?,
            lm_head,
            blocks,
            final_norm: src.vector("norm.weight")?,
            // Precompute every rotation angle once. Cheap: two floats per
            // position per coordinate pair.
            rope: Rope::new(spec.head_dim, spec.n_ctx, spec.rope_theta),
            spec,
        })
    }

    /// Rotate each head of a projection in place.
    fn apply_rope(&self, x: &mut [f32], pos: usize) {
        for head in x.chunks_mut(self.spec.head_dim) {
            self.rope.apply(head, pos);
        }
    }
    /// Every block, over a batch of tokens, leaving the residual stream as
    /// it is — no output head.
    ///
    /// Split out because the two callers want different slices of the same
    /// work: generation needs the last position's logits, scoring needs all
    /// of them, and the thirty layers in between are identical.
    fn run_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let (m, e) = (tokens.len(), spec.n_embd);
        let qdim = spec.n_head * spec.head_dim;
        let kvdim = spec.kv_dim();
        let pos0 = cache.len;
        assert!(pos0 + m <= spec.n_ctx, "context window of {} tokens is full", spec.n_ctx);

        // No position embedding: position arrives inside attention, as RoPE.
        let mut xs = vec![0.0f32; m * e];
        for (i, &t) in tokens.iter().enumerate() {
            xs[i * e..(i + 1) * e].copy_from_slice(&self.embed.row(t as usize));
        }

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
                self.apply_rope(&mut q[i * qdim..(i + 1) * qdim], pos0 + i);
                self.apply_rope(&mut k[i * kvdim..(i + 1) * kvdim], pos0 + i);
                cache.push(l, &k[i * kvdim..(i + 1) * kvdim], &v[i * kvdim..(i + 1) * kvdim]);
            }

            let mut attn = vec![0.0f32; m * qdim];
            for i in 0..m {
                attn[i * qdim..(i + 1) * qdim].copy_from_slice(&attend(
                    spec,
                    &q[i * qdim..(i + 1) * qdim],
                    cache.keys(l),
                    cache.values(l),
                    pos0 + i + 1,
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
            let mut gate = block.gate_w.matmul_bt(&hs, m, None);
            let up = block.up_w.matmul_bt(&hs, m, None);
            swiglu_inplace(&mut gate, &up);
            let mlp = block.down_w.matmul_bt(&gate, m, None);
            for (x, val) in xs.iter_mut().zip(mlp.iter()) {
                *x += val;
            }
        }

        cache.len += m;
        xs
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
                + b.gate_w.param_count()
                + b.up_w.param_count()
                + b.down_w.param_count()
                + b.attn_norm.len()
                + b.mlp_norm.len()
                + opt(&b.q_b)
                + opt(&b.k_b)
                + opt(&b.v_b)
                + opt(&b.q_norm)
                + opt(&b.k_norm)
        });
        self.embed.param_count()
            + self.lm_head.as_ref().map_or(0, |h| h.param_count())
            + self.final_norm.len()
            + per_block * self.blocks.len()
    }

    fn memory_bytes(&self) -> usize {
        let per_block: usize = self.blocks.first().map_or(0, |b| {
            b.q_w.bytes()
                + b.k_w.bytes()
                + b.v_w.bytes()
                + b.o_w.bytes()
                + b.gate_w.bytes()
                + b.up_w.bytes()
                + b.down_w.bytes()
        });
        self.embed.bytes()
            + self.lm_head.as_ref().map_or(0, |h| h.bytes())
            + per_block * self.blocks.len()
    }

    fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let (m, e) = (tokens.len(), self.spec.n_embd);
        let xs = self.run_batch(tokens, cache);
        let last = rms_norm(&xs[(m - 1) * e..m * e], &self.final_norm, self.spec.eps);
        self.lm_head.as_ref().unwrap_or(&self.embed).matvec_bt(&last, None)
    }

    /// Logits for **every** position, not just the last.
    ///
    /// The residual stream already holds all of them; what `forward_batch`
    /// throws away is the output head applied to the other rows. Scoring text
    /// — perplexity — needs exactly those, and running the head over `m` rows
    /// as one matmul costs a fraction of the `m` forward passes the default
    /// implementation would do instead.
    fn forward_batch_all(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let (m, e) = (tokens.len(), self.spec.n_embd);
        let mut xs = self.run_batch(tokens, cache);
        for i in 0..m {
            let normed = rms_norm(&xs[i * e..(i + 1) * e], &self.final_norm, self.spec.eps);
            xs[i * e..(i + 1) * e].copy_from_slice(&normed);
        }
        self.lm_head.as_ref().unwrap_or(&self.embed).matmul_bt(&xs, m, None)
    }

    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let pos = cache.len;
        assert!(pos < spec.n_ctx, "context window of {} tokens is full", spec.n_ctx);

        // No position embedding here. The token vector is all that enters the
        // residual stream; position arrives later, inside attention.
        let mut x = self.embed.row(token as usize);

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

            cache.push(l, &k, &v);
            let attn = attend(spec, &q, cache.keys(l), cache.values(l), pos + 1);
            let attn = linear(&attn, &block.o_w, None);
            for (xi, a) in x.iter_mut().zip(attn.iter()) {
                *xi += a; // residual
            }

            // ---- MLP sub-block -------------------------------------------
            let h = rms_norm(&x, &block.mlp_norm, spec.eps);
            let mut gate = linear(&h, &block.gate_w, None);
            let up = linear(&h, &block.up_w, None);
            // gate <- silu(gate) * up
            swiglu_inplace(&mut gate, &up);
            let mlp = linear(&gate, &block.down_w, None);
            for (xi, m) in x.iter_mut().zip(mlp.iter()) {
                *xi += m; // residual
            }
        }

        cache.len += 1;

        let x = rms_norm(&x, &self.final_norm, spec.eps);
        self.lm_head.as_ref().unwrap_or(&self.embed).matvec_bt(&x, None)
    }
}
