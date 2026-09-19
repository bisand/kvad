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
use crate::qcache::Source;
use crate::quant::Weight;
use crate::tensor::{gelu_inplace, layer_norm};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Block {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    /// [n_embd, 3 * n_embd] — query, key and value projections fused into one
    /// matrix, because three matmuls that all read the same input is wasteful
    /// when one will do.
    attn_w: Weight,
    attn_b: Vec<f32>,
    attn_proj_w: Weight,
    attn_proj_b: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    /// [n_embd, 4 * n_embd] — the MLP widens by 4x, applies GELU, and comes
    /// back down. Two thirds of the model's parameters live here.
    fc_w: Weight,
    fc_b: Vec<f32>,
    proj_w: Weight,
    proj_b: Vec<f32>,
}

pub struct Model {
    spec: Spec,
    /// [vocab_size, n_embd] — token embeddings, reused transposed as the
    /// output head. GPT-2 ties those weights: "the vector meaning cat" and
    /// "the direction that predicts cat" are the same thing.
    wte: Weight,
    /// [n_ctx, n_embd] — *learned* position embeddings, one row per slot.
    /// This is why GPT-2 stops at 1024 tokens: there is no row 1025, and no
    /// way to invent one. Replacing this with RoPE is the single biggest
    /// difference in `llama.rs`.
    wpe: Weight,
    blocks: Vec<Block>,
    lnf_g: Vec<f32>,
    lnf_b: Vec<f32>,
}

impl Model {
    /// Each matrix is quantised as it is read, so the f32 copy is transient
    /// and peak memory is the quantised model plus one tensor, not two full
    /// copies of the weights.
    ///
    /// `matrix_t` additionally transposes on the way in. GPT-2's checkpoint
    /// stores the projections `[in, out]` for a `Conv1D`; flipping them to
    /// `[out, in]` lets them share the one matmul kernel with Llama, integer
    /// path included.
    ///
    /// The embedding tables are deliberately *not* transposed: `wte` is
    /// already `[vocab, n_embd]`, which is both the layout row lookup wants
    /// and the layout the tied output head wants.
    pub fn load(src: &dyn Source, spec: Spec) -> Res<Self> {
        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let p = |s: &str| format!("h.{i}.{s}");
            blocks.push(Block {
                ln1_g: src.vector(&p("ln_1.weight"))?,
                ln1_b: src.vector(&p("ln_1.bias"))?,
                attn_w: src.matrix_t(&p("attn.c_attn.weight"))?,
                attn_b: src.vector(&p("attn.c_attn.bias"))?,
                attn_proj_w: src.matrix_t(&p("attn.c_proj.weight"))?,
                attn_proj_b: src.vector(&p("attn.c_proj.bias"))?,
                ln2_g: src.vector(&p("ln_2.weight"))?,
                ln2_b: src.vector(&p("ln_2.bias"))?,
                fc_w: src.matrix_t(&p("mlp.c_fc.weight"))?,
                fc_b: src.vector(&p("mlp.c_fc.bias"))?,
                proj_w: src.matrix_t(&p("mlp.c_proj.weight"))?,
                proj_b: src.vector(&p("mlp.c_proj.bias"))?,
            });
        }

        Ok(Model {
            wte: src.matrix("wte.weight")?,
            wpe: src.matrix("wpe.weight")?,
            blocks,
            lnf_g: src.vector("ln_f.weight")?,
            lnf_b: src.vector("ln_f.bias")?,
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
            b.attn_w.param_count()
                + b.attn_b.len()
                + b.attn_proj_w.param_count()
                + b.attn_proj_b.len()
                + b.fc_w.param_count()
                + b.fc_b.len()
                + b.proj_w.param_count()
                + b.proj_b.len()
                + 4 * self.spec.n_embd
        });
        self.wte.param_count() + self.wpe.param_count() + per_block * self.blocks.len()
    }

    fn memory_bytes(&self) -> usize {
        let per_block: usize = self.blocks.first().map_or(0, |b| {
            b.attn_w.bytes() + b.attn_proj_w.bytes() + b.fc_w.bytes() + b.proj_w.bytes()
        });
        self.wte.bytes() + self.wpe.bytes() + per_block * self.blocks.len()
    }

    fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let (m, e) = (tokens.len(), spec.n_embd);
        let pos0 = cache.len;
        assert!(pos0 + m <= spec.n_ctx, "context window of {} tokens is full", spec.n_ctx);

        // Token embedding plus position embedding, one row per token.
        let mut xs = vec![0.0f32; m * e];
        for (i, &t) in tokens.iter().enumerate() {
            let tok = self.wte.row(t as usize);
            let pos = self.wpe.row(pos0 + i);
            for j in 0..e {
                xs[i * e + j] = tok[j] + pos[j];
            }
        }

        let mut hs = vec![0.0f32; m * e];
        for (l, block) in self.blocks.iter().enumerate() {
            // ---- Attention sub-block -------------------------------------
            for i in 0..m {
                hs[i * e..(i + 1) * e]
                    .copy_from_slice(&layer_norm(&xs[i * e..(i + 1) * e], &block.ln1_g, &block.ln1_b, spec.eps));
            }
            let qkv = block.attn_w.matmul_bt(&hs, m, Some(&block.attn_b));

            // Every key and value first, so the cache is complete before any
            // query reads it.
            for i in 0..m {
                let row = &qkv[i * 3 * e..(i + 1) * 3 * e];
                cache.push(l, &row[e..2 * e], &row[2 * e..3 * e]);
            }

            let mut attn = vec![0.0f32; m * e];
            for i in 0..m {
                let q = &qkv[i * 3 * e..i * 3 * e + e];
                attn[i * e..(i + 1) * e]
                    .copy_from_slice(&attend(spec, q, cache.keys(l), cache.values(l), pos0 + i + 1));
            }
            let proj = block.attn_proj_w.matmul_bt(&attn, m, Some(&block.attn_proj_b));
            for (x, a) in xs.iter_mut().zip(proj.iter()) {
                *x += a;
            }

            // ---- MLP sub-block -------------------------------------------
            for i in 0..m {
                hs[i * e..(i + 1) * e]
                    .copy_from_slice(&layer_norm(&xs[i * e..(i + 1) * e], &block.ln2_g, &block.ln2_b, spec.eps));
            }
            let mut hidden = block.fc_w.matmul_bt(&hs, m, Some(&block.fc_b));
            gelu_inplace(&mut hidden);
            let mlp = block.proj_w.matmul_bt(&hidden, m, Some(&block.proj_b));
            for (x, v) in xs.iter_mut().zip(mlp.iter()) {
                *x += v;
            }
        }

        cache.len += m;

        // Only the last position predicts anything we need, so the output head
        // stays a matrix-vector product.
        let last = layer_norm(&xs[(m - 1) * e..m * e], &self.lnf_g, &self.lnf_b, spec.eps);
        self.wte.matvec_bt(&last, None)
    }

    fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let spec = &self.spec;
        let pos = cache.len;
        assert!(pos < spec.n_ctx, "context window of {} tokens is full", spec.n_ctx);

        // Embedding lookup is literally a row index, and position is a second
        // row index added on top. "Meaning" here is just which of 50257 rows
        // of 768 floats you picked.
        let mut x = self.wte.row(token as usize);
        for (xi, p) in x.iter_mut().zip(self.wpe.row(pos)) {
            *xi += p;
        }

        for (l, block) in self.blocks.iter().enumerate() {
            // ---- Attention sub-block -------------------------------------
            let h = layer_norm(&x, &block.ln1_g, &block.ln1_b, spec.eps);
            // One matmul produces query, key and value back to back.
            let qkv = block.attn_w.matvec_bt(&h, Some(&block.attn_b));
            let (q, kv) = qkv.split_at(spec.n_embd);
            let (k, v) = kv.split_at(spec.n_embd);

            cache.push(l, k, v);
            let attn = attend(spec, q, cache.keys(l), cache.values(l), pos + 1);
            let attn = block.attn_proj_w.matvec_bt(&attn, Some(&block.attn_proj_b));
            for (xi, a) in x.iter_mut().zip(attn.iter()) {
                *xi += a; // residual
            }

            // ---- MLP sub-block -------------------------------------------
            let h = layer_norm(&x, &block.ln2_g, &block.ln2_b, spec.eps);
            let mut hidden = block.fc_w.matvec_bt(&h, Some(&block.fc_b));
            gelu_inplace(&mut hidden);
            let mlp = block.proj_w.matvec_bt(&hidden, Some(&block.proj_b));
            for (xi, m) in x.iter_mut().zip(mlp.iter()) {
                *xi += m; // residual
            }
        }

        cache.len += 1;

        let x = layer_norm(&x, &self.lnf_g, &self.lnf_b, spec.eps);
        self.wte.matvec_bt(&x, None)
    }
}
