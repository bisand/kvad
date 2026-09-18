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
//! stores them the other way round — so these matmuls are all
//! [`matvec_bt`], and GPT-2's are [`matvec`]. Mixing the two up produces
//! fluent nonsense rather than an error, which is why the counting test in the
//! README exists.

use super::{attend, KvCache, Spec, Transformer};
use crate::quant::{Precision, Weight};
use crate::tensor::{rms_norm, swiglu_inplace, Rope};
use crate::weights::Checkpoint;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `y = x @ Wᵀ (+ b)`, the HuggingFace `nn.Linear` convention.
///
/// The bias is optional because most of this family dropped biases entirely —
/// they cost parameters and buy nothing once you have normalisation. Qwen2 is
/// the exception: it keeps biases on Q, K and V but nowhere else.
fn linear(x: &[f32], w: &Weight, bias: Option<&Vec<f32>>) -> Vec<f32> {
    let mut y = w.matvec_bt(x);
    if let Some(b) = bias {
        for (v, bi) in y.iter_mut().zip(b.iter()) {
            *v += bi;
        }
    }
    y
}

pub struct Block {
    attn_norm: Vec<f32>,
    q_w: Weight,
    q_b: Option<Vec<f32>>,
    k_w: Weight,
    k_b: Option<Vec<f32>>,
    v_w: Weight,
    v_b: Option<Vec<f32>>,
    o_w: Weight,
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

impl Model {
    pub fn load(ckpt: &Checkpoint, spec: Spec, precision: Precision) -> Res<Self> {
        let w = |name: &str| -> Res<Weight> { Ok(Weight::quantize(ckpt.get(name)?, precision)) };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            blocks.push(Block {
                attn_norm: ckpt.get_flat(&p("input_layernorm.weight"))?,
                q_w: w(&p("self_attn.q_proj.weight"))?,
                q_b: ckpt.try_get_flat(&p("self_attn.q_proj.bias")),
                k_w: w(&p("self_attn.k_proj.weight"))?,
                k_b: ckpt.try_get_flat(&p("self_attn.k_proj.bias")),
                v_w: w(&p("self_attn.v_proj.weight"))?,
                v_b: ckpt.try_get_flat(&p("self_attn.v_proj.bias")),
                o_w: w(&p("self_attn.o_proj.weight"))?,
                mlp_norm: ckpt.get_flat(&p("post_attention_layernorm.weight"))?,
                gate_w: w(&p("mlp.gate_proj.weight"))?,
                up_w: w(&p("mlp.up_proj.weight"))?,
                down_w: w(&p("mlp.down_proj.weight"))?,
            });
        }

        let lm_head = match spec.tie_embeddings {
            true => None,
            false => ckpt.try_get("lm_head.weight").map(|t| Weight::quantize(t, precision)),
        };

        Ok(Model {
            embed: w("embed_tokens.weight")?,
            lm_head,
            blocks,
            final_norm: ckpt.get_flat("norm.weight")?,
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
}

impl Transformer for Model {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn param_count(&self) -> usize {
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
        });
        self.embed.param_count()
            + self.lm_head.as_ref().map_or(0, |h| h.param_count())
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
        self.lm_head.as_ref().unwrap_or(&self.embed).matvec_bt(&x)
    }
}
