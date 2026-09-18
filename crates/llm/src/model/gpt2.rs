//! GPT-2 (2019).
//!
//! The simpler of the two architectures, and the one to read first. Its block
//! is:
//!
//! ```text
//!   x = x + Attention(LayerNorm(x))
//!   x = x + MLP(LayerNorm(x))
//! ```
//!
//! with position information added once, at the very bottom, by looking up a
//! learned vector per slot in the context window.

use super::{attend, KvCache, Spec, Transformer};
use crate::tensor::{gelu_inplace, layer_norm, matvec, matvec_bt, Tensor};
use crate::weights::Checkpoint;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Block {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    /// [n_embd, 3 * n_embd] — query, key and value projections fused into one
    /// matrix, because three matmuls that all read the same input is wasteful
    /// when one will do.
    attn_w: Tensor,
    attn_b: Vec<f32>,
    attn_proj_w: Tensor,
    attn_proj_b: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    /// [n_embd, 4 * n_embd] — the MLP widens by 4x, applies GELU, and comes
    /// back down. Two thirds of the model's parameters live here.
    fc_w: Tensor,
    fc_b: Vec<f32>,
    proj_w: Tensor,
    proj_b: Vec<f32>,
}

pub struct Model {
    spec: Spec,
    /// [vocab_size, n_embd] — token embeddings, reused transposed as the
    /// output head. GPT-2 ties those weights: "the vector meaning cat" and
    /// "the direction that predicts cat" are the same thing.
    wte: Tensor,
    /// [n_ctx, n_embd] — *learned* position embeddings, one row per slot.
    /// This is why GPT-2 stops at 1024 tokens: there is no row 1025, and no
    /// way to invent one. Replacing this with RoPE is the single biggest
    /// difference in `llama.rs`.
    wpe: Tensor,
    blocks: Vec<Block>,
    lnf_g: Vec<f32>,
    lnf_b: Vec<f32>,
}

impl Model {
    pub fn load(ckpt: &Checkpoint, spec: Spec) -> Res<Self> {
        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let p = |s: &str| format!("h.{i}.{s}");
            blocks.push(Block {
                ln1_g: ckpt.get_flat(&p("ln_1.weight"))?,
                ln1_b: ckpt.get_flat(&p("ln_1.bias"))?,
                attn_w: ckpt.get(&p("attn.c_attn.weight"))?,
                attn_b: ckpt.get_flat(&p("attn.c_attn.bias"))?,
                attn_proj_w: ckpt.get(&p("attn.c_proj.weight"))?,
                attn_proj_b: ckpt.get_flat(&p("attn.c_proj.bias"))?,
                ln2_g: ckpt.get_flat(&p("ln_2.weight"))?,
                ln2_b: ckpt.get_flat(&p("ln_2.bias"))?,
                fc_w: ckpt.get(&p("mlp.c_fc.weight"))?,
                fc_b: ckpt.get_flat(&p("mlp.c_fc.bias"))?,
                proj_w: ckpt.get(&p("mlp.c_proj.weight"))?,
                proj_b: ckpt.get_flat(&p("mlp.c_proj.bias"))?,
            });
        }

        Ok(Model {
            wte: ckpt.get("wte.weight")?,
            wpe: ckpt.get("wpe.weight")?,
            blocks,
            lnf_g: ckpt.get_flat("ln_f.weight")?,
            lnf_b: ckpt.get_flat("ln_f.bias")?,
            spec,
        })
    }
}

impl Transformer for Model {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn param_count(&self) -> usize {
        let per_block = self.blocks.first().map_or(0, |b| {
            b.attn_w.data.len()
                + b.attn_b.len()
                + b.attn_proj_w.data.len()
                + b.attn_proj_b.len()
                + b.fc_w.data.len()
                + b.fc_b.len()
                + b.proj_w.data.len()
                + b.proj_b.len()
                + 4 * self.spec.n_embd
        });
        self.wte.data.len() + self.wpe.data.len() + per_block * self.blocks.len()
    }

    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let pos = cache.len;
        assert!(pos < spec.n_ctx, "context window of {} tokens is full", spec.n_ctx);

        // Embedding lookup is literally a row index, and position is a second
        // row index added on top. "Meaning" here is just which of 50257 rows
        // of 768 floats you picked.
        let mut x: Vec<f32> = self
            .wte
            .row(token as usize)
            .iter()
            .zip(self.wpe.row(pos).iter())
            .map(|(a, b)| a + b)
            .collect();

        for (l, block) in self.blocks.iter().enumerate() {
            // ---- Attention sub-block -------------------------------------
            let h = layer_norm(&x, &block.ln1_g, &block.ln1_b, spec.eps);
            // One matmul produces query, key and value back to back.
            let qkv = matvec(&h, &block.attn_w, Some(&block.attn_b));
            let (q, kv) = qkv.split_at(spec.n_embd);
            let (k, v) = kv.split_at(spec.n_embd);

            cache.push(l, k, v);
            let attn = attend(spec, q, cache.keys(l), cache.values(l), pos + 1);
            let attn = matvec(&attn, &block.attn_proj_w, Some(&block.attn_proj_b));
            for (xi, a) in x.iter_mut().zip(attn.iter()) {
                *xi += a; // residual
            }

            // ---- MLP sub-block -------------------------------------------
            let h = layer_norm(&x, &block.ln2_g, &block.ln2_b, spec.eps);
            let mut hidden = matvec(&h, &block.fc_w, Some(&block.fc_b));
            gelu_inplace(&mut hidden);
            let mlp = matvec(&hidden, &block.proj_w, Some(&block.proj_b));
            for (xi, m) in x.iter_mut().zip(mlp.iter()) {
                *xi += m; // residual
            }
        }

        cache.len += 1;

        let x = layer_norm(&x, &self.lnf_g, &self.lnf_b, spec.eps);
        matvec_bt(&x, &self.wte)
    }
}
