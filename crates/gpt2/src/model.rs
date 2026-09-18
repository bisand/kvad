//! GPT-2, forwards.
//!
//! # What a language model actually does
//!
//! It takes a sequence of token ids and produces, for the *last* position, a
//! score for every token in the vocabulary — "what comes next?". That is all.
//! Chat, code completion and summarisation are that one operation run in a
//! loop, with the sampled token appended and the loop repeated.
//!
//! # The architecture
//!
//! ```text
//! token id ──► embedding lookup (wte) ──┐
//!                                       ├──► x  (the "residual stream")
//! position ──► embedding lookup (wpe) ──┘
//!
//!   repeat 12 times:
//!       x = x + Attention(LayerNorm(x))     ← tokens exchange information
//!       x = x + MLP(LayerNorm(x))           ← each token thinks on its own
//!
//!   logits = LayerNorm(x) @ wteᵀ            ← project back to vocabulary
//! ```
//!
//! Two things are worth sitting with:
//!
//! * **`x = x + f(x)`, not `x = f(x)`.** The residual stream is a running
//!   total that every block reads from and writes back into. A block can be
//!   close to a no-op if it has nothing to contribute, which is what makes
//!   very deep stacks trainable.
//!
//! * **Attention is the only place tokens see each other.** The MLP processes
//!   each position completely independently. All the "context" in a context
//!   window flows through the attention operation below, and nowhere else.

use crate::tensor::{gelu_inplace, layer_norm, matvec, matvec_bt, softmax_inplace, Tensor};
use crate::weights::{self, Weights};
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub eps: f32,
}

impl Config {
    /// Dimension of a single attention head. The embedding is split evenly
    /// across heads, so more heads means narrower ones, not more compute.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let usize_at = |keys: &[&str]| -> Option<usize> {
            keys.iter().find_map(|k| v.get(*k)?.as_u64()).map(|n| n as usize)
        };
        Ok(Config {
            n_layer: usize_at(&["n_layer", "num_hidden_layers"]).ok_or("config: no n_layer")?,
            n_head: usize_at(&["n_head", "num_attention_heads"]).ok_or("config: no n_head")?,
            n_embd: usize_at(&["n_embd", "hidden_size"]).ok_or("config: no n_embd")?,
            n_ctx: usize_at(&["n_positions", "n_ctx"]).unwrap_or(1024),
            vocab_size: usize_at(&["vocab_size"]).ok_or("config: no vocab_size")?,
            eps: v.get("layer_norm_epsilon").and_then(|e| e.as_f64()).unwrap_or(1e-5) as f32,
        })
    }
}

/// One transformer block: attention, then MLP, each wrapped in a LayerNorm
/// and a residual connection.
pub struct Block {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    /// [n_embd, 3 * n_embd] — query, key and value projections fused into one
    /// matrix, because doing three matmuls that read the same input is
    /// wasteful when you can do one.
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
    pub config: Config,
    /// [vocab_size, n_embd] — token embeddings. Also reused, transposed, as
    /// the output head: GPT-2 ties those weights, which saves 38M parameters
    /// and encodes the idea that "the vector meaning cat" and "the direction
    /// that predicts cat" should be the same thing.
    wte: Tensor,
    /// [n_ctx, n_embd] — *learned* position embeddings, one per slot in the
    /// context window. This is why GPT-2 cannot read beyond 1024 tokens: there
    /// simply is no row 1025.
    wpe: Tensor,
    blocks: Vec<Block>,
    lnf_g: Vec<f32>,
    lnf_b: Vec<f32>,
}

impl Model {
    pub fn load(weights_path: &Path, config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let file = Weights::open(weights_path)?;
        let st = file.view()?;

        let flat = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Ok(weights::get(&st, name)?.data)
        };

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let p = |s: &str| format!("h.{i}.{s}");
            blocks.push(Block {
                ln1_g: flat(&p("ln_1.weight"))?,
                ln1_b: flat(&p("ln_1.bias"))?,
                attn_w: weights::get(&st, &p("attn.c_attn.weight"))?,
                attn_b: flat(&p("attn.c_attn.bias"))?,
                attn_proj_w: weights::get(&st, &p("attn.c_proj.weight"))?,
                attn_proj_b: flat(&p("attn.c_proj.bias"))?,
                ln2_g: flat(&p("ln_2.weight"))?,
                ln2_b: flat(&p("ln_2.bias"))?,
                fc_w: weights::get(&st, &p("mlp.c_fc.weight"))?,
                fc_b: flat(&p("mlp.c_fc.bias"))?,
                proj_w: weights::get(&st, &p("mlp.c_proj.weight"))?,
                proj_b: flat(&p("mlp.c_proj.bias"))?,
            });
        }

        Ok(Model {
            wte: weights::get(&st, "wte.weight")?,
            wpe: weights::get(&st, "wpe.weight")?,
            blocks,
            lnf_g: flat("ln_f.weight")?,
            lnf_b: flat("ln_f.bias")?,
            config,
        })
    }

    pub fn param_count(&self) -> usize {
        // Counts unique parameters; wte is shared with the output head.
        let per_block = self.blocks.first().map_or(0, |b| {
            b.attn_w.data.len()
                + b.attn_b.len()
                + b.attn_proj_w.data.len()
                + b.attn_proj_b.len()
                + b.fc_w.data.len()
                + b.fc_b.len()
                + b.proj_w.data.len()
                + b.proj_b.len()
                + 4 * self.config.n_embd
        });
        self.wte.data.len() + self.wpe.data.len() + per_block * self.blocks.len()
    }

    /// Run one token through the whole stack and return logits over the
    /// vocabulary.
    ///
    /// `cache` holds the keys and values computed for every previous token.
    /// Without it, generating token N would mean recomputing the entire
    /// prefix — O(N²) work for the sequence instead of O(N). The cache is not
    /// an optimisation you add later; it is the reason inference is tractable,
    /// and it is why memory use grows as you fill the context window.
    pub fn forward(&self, token: u32, cache: &mut Cache) -> Vec<f32> {
        let cfg = &self.config;
        let pos = cache.len;
        assert!(pos < cfg.n_ctx, "context window of {} tokens is full", cfg.n_ctx);

        // Embedding lookup is literally a row index. "Meaning" here is just
        // which of 50257 rows of 768 floats you picked.
        let mut x: Vec<f32> = self
            .wte
            .row(token as usize)
            .iter()
            .zip(self.wpe.row(pos).iter())
            .map(|(a, b)| a + b)
            .collect();

        for (l, block) in self.blocks.iter().enumerate() {
            // ---- Attention sub-block -------------------------------------
            let h = layer_norm(&x, &block.ln1_g, &block.ln1_b, cfg.eps);
            let qkv = matvec(&h, &block.attn_w, Some(&block.attn_b));
            let (q, kv) = qkv.split_at(cfg.n_embd);
            let (k, v) = kv.split_at(cfg.n_embd);

            // Append this token's key and value to the running cache.
            cache.k[l].extend_from_slice(k);
            cache.v[l].extend_from_slice(v);

            let attn_out = self.attend(q, &cache.k[l], &cache.v[l], pos + 1);
            let attn_out = matvec(&attn_out, &block.attn_proj_w, Some(&block.attn_proj_b));
            for (xi, a) in x.iter_mut().zip(attn_out.iter()) {
                *xi += a; // residual
            }

            // ---- MLP sub-block -------------------------------------------
            let h = layer_norm(&x, &block.ln2_g, &block.ln2_b, cfg.eps);
            let mut hidden = matvec(&h, &block.fc_w, Some(&block.fc_b));
            gelu_inplace(&mut hidden);
            let mlp_out = matvec(&hidden, &block.proj_w, Some(&block.proj_b));
            for (xi, m) in x.iter_mut().zip(mlp_out.iter()) {
                *xi += m; // residual
            }
        }

        cache.len += 1;

        let x = layer_norm(&x, &self.lnf_g, &self.lnf_b, cfg.eps);
        matvec_bt(&x, &self.wte)
    }

    /// Multi-head causal self-attention for a single query position.
    ///
    /// For each head: score this token's query against the key of every token
    /// so far, turn those scores into weights with a softmax, and return the
    /// correspondingly weighted average of the values.
    ///
    /// "Causal" is free here — the cache only contains earlier positions, so
    /// there is nothing in the future to mask out. (When you process a whole
    /// prompt at once you *do* need an explicit triangular mask, which is the
    /// usual first bug.)
    fn attend(&self, q: &[f32], k_cache: &[f32], v_cache: &[f32], n_positions: usize) -> Vec<f32> {
        let cfg = &self.config;
        let hd = cfg.head_dim();
        // Divide by sqrt(head_dim) before the softmax. The dot product of two
        // random vectors of dimension d has standard deviation sqrt(d); left
        // unscaled, the softmax would saturate into a one-hot as d grows and
        // gradients would vanish. This single constant is the "scaled" in
        // "scaled dot-product attention".
        let scale = 1.0 / (hd as f32).sqrt();

        let heads: Vec<Vec<f32>> = (0..cfg.n_head)
            .into_par_iter()
            .map(|head| {
                let off = head * hd;
                let q_head = &q[off..off + hd];

                let mut scores = Vec::with_capacity(n_positions);
                for t in 0..n_positions {
                    let k_head = &k_cache[t * cfg.n_embd + off..t * cfg.n_embd + off + hd];
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
                    let v_head = &v_cache[t * cfg.n_embd + off..t * cfg.n_embd + off + hd];
                    for i in 0..hd {
                        out[i] += w * v_head[i];
                    }
                }
                out
            })
            .collect();

        heads.concat()
    }
}

/// Per-layer key and value history.
pub struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl Cache {
    pub fn new(config: &Config) -> Self {
        let per_layer = || Vec::with_capacity(config.n_ctx * config.n_embd);
        Cache {
            k: (0..config.n_layer).map(|_| per_layer()).collect(),
            v: (0..config.n_layer).map(|_| per_layer()).collect(),
            len: 0,
        }
    }

    /// Bytes this cache will occupy once the context window is full. Worth
    /// printing once: for long-context models it dwarfs the weights.
    pub fn max_bytes(config: &Config) -> usize {
        2 * config.n_layer * config.n_ctx * config.n_embd * std::mem::size_of::<f32>()
    }
}
