//! GPT-2 (2019).
//!
//! The simplest of the architectures here, and the one to read first. Its block
//! is:
//!
//! ```text
//!   x = x + Attention(LayerNorm(x))
//!   x = x + MLP(LayerNorm(x))
//! ```
//!
//! with position information added once, at the very bottom, by looking up a
//! learned vector per slot in the context window.

use super::{attend, Architecture, KvCache, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::{gelu_inplace, layer_norm};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// This module's entry in the registry.
///
/// `gpt2` is the `model_type`; `GPT2LMHeadModel` reduces to the same stem.
/// Nothing here needs configuring — GPT-2 caches a key and a value per head
/// per position, which is what [`Spec`] assumes by default.
pub static ARCH: Architecture = Architecture {
    id: "gpt2",
    model_types: &["gpt2"],
    about: "GPT-2 (2019): learned positions, LayerNorm, GELU, one fused QKV matrix",
    configure: |_| Ok(()),
    load: |src, spec| Ok(Box::new(Model::load(src, spec)?)),
};

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
    /// A separate output head, `[vocab, n_embd]`, and its bias. No GPT-2 that
    /// OpenAI released has either. A checkpoint written by this repository's
    /// own `nervus` has both, and says so with `tie_word_embeddings: false`.
    lm_head: Option<Weight>,
    lm_head_b: Option<Vec<f32>>,
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

        let lm_head = head(src, &spec, "lm_head.weight")?;
        // The bias stays optional even in an untied model, because GPT-2 itself
        // has none: `lm_head` there is the embedding table applied backwards,
        // and a table has no bias to apply with it.
        let lm_head_b = match spec.tie_embeddings {
            true => {
                src.skip("lm_head.bias");
                None
            }
            false => src.try_vector("lm_head.bias"),
        };

        Ok(Model {
            lm_head,
            lm_head_b,
            wte: src.matrix("wte.weight")?,
            wpe: src.matrix("wpe.weight")?,
            blocks,
            lnf_g: src.vector("ln_f.weight")?,
            lnf_b: src.vector("ln_f.bias")?,
            spec,
        })
    }
}

impl Model {
    /// Logits from the final hidden state: the token table read the other
    /// way round, unless the checkpoint brought a head of its own.
    fn head(&self, x: &[f32]) -> Vec<f32> {
        self.lm_head.as_ref().unwrap_or(&self.wte).matvec_bt(x, self.lm_head_b.as_deref())
    }

    /// Every block, over a batch of tokens, leaving the residual stream as
    /// it is — no output head.
    ///
    /// Split out because the two callers want different slices of the same
    /// work: generation needs the last position's logits, scoring needs all
    /// of them, and the twelve layers in between are identical.
    fn run_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
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
        xs
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
        self.wte.param_count()
            + self.wpe.param_count()
            + self.lm_head.as_ref().map_or(0, |h| h.param_count())
            + self.lm_head_b.as_ref().map_or(0, |b| b.len())
            + self.lnf_g.len()
            + self.lnf_b.len()
            + per_block * self.blocks.len()
    }

    fn memory_bytes(&self) -> usize {
        let per_block: usize = self.blocks.first().map_or(0, |b| {
            b.attn_w.bytes() + b.attn_proj_w.bytes() + b.fc_w.bytes() + b.proj_w.bytes()
        });
        self.wte.bytes()
            + self.wpe.bytes()
            + self.lm_head.as_ref().map_or(0, |h| h.bytes())
            + per_block * self.blocks.len()
    }

    fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let (m, e) = (tokens.len(), self.spec.n_embd);
        let xs = self.run_batch(tokens, cache);
        // Only the last position predicts anything generation needs, so the
        // output head stays a matrix-vector product.
        let last = layer_norm(&xs[(m - 1) * e..m * e], &self.lnf_g, &self.lnf_b, self.spec.eps);
        self.head(&last)
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
            let normed =
                layer_norm(&xs[i * e..(i + 1) * e], &self.lnf_g, &self.lnf_b, self.spec.eps);
            xs[i * e..(i + 1) * e].copy_from_slice(&normed);
        }
        self.lm_head
            .as_ref()
            .unwrap_or(&self.wte)
            .matmul_bt(&xs, m, self.lm_head_b.as_deref())
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
        self.head(&x)
    }
}
