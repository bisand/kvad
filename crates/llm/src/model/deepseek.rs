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

use super::ffn::{Ffn, Layout, Mlp, Moe, Router};
use super::{Architecture, CacheShape, Json, KvCache, Spec, Transformer};
use crate::qcache::{head, Source};
use crate::quant::Weight;
use crate::tensor::{dot, rms_norm, softmax_inplace, Rope};
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
pub struct Mla {
    pub n_head: usize,
    /// Per head, the part of the query and key that carries content and is
    /// absorbed into the projection matrices.
    pub qk_nope: usize,
    /// Per head, the part that carries position and is rotated. Shared by
    /// every head on the key side — one rotated vector per position, not one
    /// per head.
    pub qk_rope: usize,
    pub v_head: usize,
    /// Width of the compressed vector that is all the cache holds.
    pub kv_lora: usize,
    /// V3 compresses the query too, through a rank of this width, to save
    /// parameters. V2-Lite does not (`q_lora_rank: null`).
    pub q_lora: Option<usize>,
    /// `1/sqrt(qk_nope + qk_rope)`, times YaRN's correction when there is one.
    pub softmax_scale: f32,
}

impl Mla {
    pub fn read(config: &Json, n_head: usize) -> Res<Self> {
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
    pub fn q_head(&self) -> usize {
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
pub fn build_rope(dim: usize, max_positions: usize, theta: f32, config: &Json) -> Res<Rope> {
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
// Weights
// ---------------------------------------------------------------------------

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

struct Block {
    attn: Attn,
    mlp_norm: Vec<f32>,
    mlp: Mlp,
}

pub struct Model {
    spec: Spec,
    mla: Mla,
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
        let Layout { first_dense, moe_every, n_shared } = Layout::read(c);

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

            let is_moe = Layout { first_dense, moe_every, n_shared }.is_moe(i, router.n_experts);
            let mlp = match is_moe {
                false => Mlp::Dense(Ffn::load(src, &p("mlp"))?),
                true => Mlp::Moe(Box::new(Moe {
                    router: router.clone(),
                    gate: src.matrix(&p("mlp.gate.weight"))?,
                    bias: src.try_vector(&p("mlp.gate.e_score_correction_bias")),
                    experts: (0..router.n_experts)
                        .map(|e| Ffn::load(src, &p(&format!("mlp.experts.{e}"))))
                        .collect::<Res<Vec<_>>>()?,
                    shared: match n_shared {
                        0 => None,
                        _ => Some(Ffn::load(src, &p("mlp.shared_experts"))?),
                    },
                })),
            };

            blocks.push(Block {
                attn,
                mlp_norm: src.vector(&p("post_attention_layernorm.weight"))?,
                mlp,
            });
        }

        // V3's multi-token-prediction head, which the module header explains is
        // not implemented: one whole extra block per predicted token, filed at
        // `layers.{n_layer}` and up. Saying so is not decoration — *deliberately
        // not read* and *forgotten* look identical from outside a loader, and
        // the check that runs after this would otherwise refuse every V3
        // checkpoint for carrying weights nobody wanted.
        let mtp = spec.config.num(&["num_nextn_predict_layers"]).unwrap_or(0);
        for i in spec.n_layer..spec.n_layer + mtp {
            src.skip_under(&format!("layers.{i}."));
        }

        let lm_head = head(src, &spec, "lm_head.weight")?;
        let rope = build_rope(mla.qk_rope, spec.n_ctx, spec.rope_theta, &spec.config)?;

        Ok(Model {
            embed: src.matrix("embed_tokens.weight")?,
            lm_head,
            blocks,
            final_norm: src.vector("norm.weight")?,
            mla,
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
            let out = block.mlp.run(&hs, m, e);
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
                        // The latent is 512 wide here, so a running sum would
                        // be a 512-long chain of dependent adds per position.
                        let c = &cs[t * lat..(t + 1) * lat];
                        let pe = &k_pe[t * rope_dim..(t + 1) * rope_dim];
                        let d = dot(&q_lat, c) + dot(q_pe, pe);
                        scores.push(d * mla.softmax_scale);
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
            + self.final_norm.len()
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
