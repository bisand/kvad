//! DeepSeek V2 and V3 on the GPU: latent attention, and a mixture of experts.
//!
//! Read [`kvad::model::deepseek`] first — its header explains what MLA is and
//! why it exists, and none of that is repeated here. This file is about what
//! changes when the same arithmetic is handed to a framework.
//!
//! # What the GPU changes, and what it must not
//!
//! The hand-written engine scores one head at a time, because a per-head loop
//! over `rayon` is how a CPU gets parallelism. Here every head is one axis of
//! one tensor, so the whole of attention is five batched matmuls and a softmax:
//!
//! ```text
//!   q_lat  = q_nope ⊗ W_UK          [heads, m, lat]   -- the absorption
//!   scores = q_lat ⊗ cᵀ + q_pe ⊗ k_peᵀ                -- content, then position
//!   acc    = softmax(scores) ⊗ c    [heads, m, lat]   -- average the latents
//!   out    = acc ⊗ W_UVᵀ            [heads, m, v]     -- decompress once
//! ```
//!
//! The absorption is the same trick either way: `W_UK` moves onto the query,
//! of which there is one per step, rather than onto the keys, of which there
//! are thousands. What the framework buys is that the per-head matrices stop
//! being a list of small matmuls and become one axis of a batched one.
//!
//! So the cache holds `c` (512 wide) and `k_pe` (64), *shared by every head*,
//! and nothing else. That is the architecture, and it is why this file never
//! makes a per-head copy of the shared key.
//!
//! # The two rotations that are not the usual one
//!
//! Position enters in exactly two places, both narrow: the last `qk_rope`
//! numbers of each head's query, and the one shared `k_pe` per position. And
//! it uses `rope_i` — the interleaved convention, pairing `2i` with `2i+1` —
//! where every other model in this backend uses `rope`. Getting that wrong
//! costs nothing visible: the model stays fluent and stops being right.
//!
//! The frequencies themselves are YaRN's, and are *not* recomputed here. They
//! come from [`kvad::model::deepseek::build_rope`], because which frequencies
//! a config implies is a property of the model rather than of an engine, and a
//! second implementation of a ramp between two correction dimensions is a
//! second thing to get subtly wrong.
//!
//! # The mixture is grouped by expert, not by token
//!
//! Routing is per token, so a batch of a hundred generally touches most of the
//! sixty-four experts however you slice it. Walking tokens would read each
//! chosen expert's weights again for every token that chose it; walking experts
//! reads each one once. [`GpuDeepSeek::mixture`] therefore gathers the rows
//! that picked an expert, runs it over exactly those, and scatters the weighted
//! result back — the same grouping the CPU engine does, in `index_select`,
//! `broadcast_mul` and `index_add` instead of three loops.
//!
//! An expert nobody picked is skipped entirely, which is where the saving
//! actually comes from at decode: one token reaches `top_k` experts, so six of
//! sixty-four run and fifty-eight are never touched.

use crate::common::{
    causal_mask, check_block, embedding, label, unread, unread_error, Embed, Loader, Proj, Reader,
    Stored,
};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, rotary_emb, VarBuilder};
use kvad::model::deepseek::{build_rope, Mla};
use kvad::model::ffn::{Layout, Router};
use kvad::model::{Session, Spec};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A SwiGLU MLP: a dense layer, a shared expert, or one routed expert.
struct Ffn {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(gate * self.up.forward(x)?)?)
    }

    fn params(&self) -> usize {
        self.gate.params() + self.up.params() + self.down.params()
    }

    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

/// The query projection, which V3 compresses and V2-Lite does not.
enum Query {
    Direct(Proj),
    Compressed { down: Proj, norm: Tensor, up: Proj },
}

struct Attn {
    norm: Tensor,
    q: Query,
    kv_a: Proj,
    kv_a_norm: Tensor,
    /// `[heads, qk_nope, kv_lora]` — every head's key up-projection, kept as
    /// one tensor so the absorption is a batched matmul rather than a loop.
    /// Dense whatever the mode: on V2-Lite this is 2M numbers a layer against
    /// the experts' 400M, and a quantised one could not be batched.
    uk: Tensor,
    /// `[heads, kv_lora, v_head]` — the value up-projection, transposed for the
    /// same matmul.
    uv: Tensor,
    o: Proj,
}

struct Moe {
    /// `[n_experts, hidden]`: one dot product per expert.
    gate: Proj,
    /// V3's learned balancing bias, which steers the choice and not the
    /// weights. Kept on the host: it is read by the router, which runs there.
    bias: Option<Vec<f32>>,
    experts: Vec<Ffn>,
    shared: Option<Ffn>,
}

enum Mlp {
    Dense(Ffn),
    Moe(Moe),
}

struct Block {
    attn: Attn,
    mlp_norm: Tensor,
    mlp: Mlp,
}

pub struct GpuDeepSeek {
    spec: Spec,
    mla: Mla,
    router: Router,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    embed: Embed,
    head: Proj,
    tied: bool,
    blocks: Vec<Block>,
    final_norm: Tensor,
    /// YaRN's tables, `[n_ctx, qk_rope / 2]`.
    cos: Tensor,
    sin: Tensor,
    /// Per layer, the compressed vector and the shared rotated key:
    /// `[1, 1, seq, kv_lora]` and `[1, 1, seq, qk_rope]`.
    cache: Vec<Option<(Tensor, Tensor)>>,
    pos: usize,
}

impl GpuDeepSeek {
    pub fn load(
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
        cache: &Vault,
    ) -> Res<Self> {
        let mla = Mla::read(&spec.config, spec.n_head)?;
        let router = Router::read(&spec)?;
        let layout = Layout::read(&spec.config);
        let load_dtype = if quant.is_some() { DType::F32 } else { dtype };
        let compute = load_dtype;
        let e = spec.n_embd;
        let (nope, rope_dim, lat, vh) = (mla.qk_nope, mla.qk_rope, mla.kv_lora, mla.v_head);
        let heads = mla.n_head;
        let inter: usize = spec.config.num(&["intermediate_size"]).unwrap_or(spec.intermediate);
        let moe_inter: usize =
            spec.config.num(&["moe_intermediate_size"]).unwrap_or(inter);

        check_block(
            quant,
            &[
                ("hidden size", e),
                ("MLP width", inter),
                ("expert width", moe_inter),
                ("latent width", lat),
            ],
        )?;

        // The checkpoint is mapped on the host in both modes, and each
        // tensor is moved to the device once it is in its final form —
        // quantised into blocks, or transposed. Reading straight onto the
        // device instead costs a second full copy of every dense weight
        // while its transpose is built, which is what put a 7B over this
        // machine's GPU budget; see `Loader::proj`.
        let load_dev = Device::Cpu;
        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, load_dtype, &load_dev)? };
        let vb = Reader::new(vb);
        let model = vb.pp("model");

        let to_dev = |t: Tensor| -> Res<Tensor> { Ok(t.to_device(&device)?) };
        // HuggingFace stores `nn.Linear` as [out, in]. Same rule as
        // `model.rs`, and now the same code.
        let ld = Loader::new(quant, device.clone(), cache);
        let load_t = |vb: &Reader<'_>, name: &str, out: usize, inp: usize| -> Res<Proj> {
            ld.proj(vb, name, out, inp, Stored::OutIn)
        };
        let ffn = |vb: &Reader<'_>, width: usize| -> Res<Ffn> {
            Ok(Ffn {
                gate: load_t(vb, "gate_proj.weight", width, e)?,
                up: load_t(vb, "up_proj.weight", width, e)?,
                down: load_t(vb, "down_proj.weight", e, width)?,
            })
        };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let l = model.pp(format!("layers.{i}"));
            let sa = l.pp("self_attn");

            // One stored matrix, `[heads * (nope + v_head), lat]`, holding both
            // up-projections for every head back to back. The CPU engine slices
            // it into 2 * heads small weights; here it is reshaped into two
            // tensors with a head axis, which is the same split done once.
            let kv_b = sa.get((heads * (nope + vh), lat), "kv_b_proj.weight")?;
            let by_head = kv_b.reshape((heads, nope + vh, lat))?;
            let uk = by_head.narrow(1, 0, nope)?.contiguous()?;
            // `W_UV` is applied the other way round, so it is transposed once
            // here rather than inside every token's matmul.
            let uv = by_head.narrow(1, nope, vh)?.transpose(1, 2)?.contiguous()?;

            let attn = Attn {
                norm: to_dev(l.get(e, "input_layernorm.weight")?)?,
                q: match mla.q_lora {
                    None => Query::Direct(load_t(&sa, "q_proj.weight", heads * mla.q_head(), e)?),
                    Some(rank) => Query::Compressed {
                        down: load_t(&sa, "q_a_proj.weight", rank, e)?,
                        norm: to_dev(sa.get(rank, "q_a_layernorm.weight")?)?,
                        up: load_t(&sa, "q_b_proj.weight", heads * mla.q_head(), rank)?,
                    },
                },
                kv_a: load_t(&sa, "kv_a_proj_with_mqa.weight", lat + rope_dim, e)?,
                kv_a_norm: to_dev(sa.get(lat, "kv_a_layernorm.weight")?)?,
                uk: to_dev(uk.to_dtype(compute)?)?,
                uv: to_dev(uv.to_dtype(compute)?)?,
                o: load_t(&sa, "o_proj.weight", e, heads * vh)?,
            };

            let mlp = match layout.is_moe(i, router.n_experts) {
                false => Mlp::Dense(ffn(&l.pp("mlp"), inter)?),
                true => {
                    let m = l.pp("mlp");
                    Mlp::Moe(Moe {
                        gate: load_t(&m, "gate.weight", router.n_experts, e)?,
                        bias: m
                            .try_get(router.n_experts, "gate.e_score_correction_bias")
                            .map(|t| t.to_dtype(DType::F32)?.to_vec1::<f32>())
                            .transpose()?,
                        experts: (0..router.n_experts)
                            .map(|x| ffn(&m.pp(format!("experts.{x}")), moe_inter))
                            .collect::<Res<Vec<_>>>()?,
                        shared: match layout.n_shared {
                            0 => None,
                            n => Some(ffn(&m.pp("shared_experts"), moe_inter * n)?),
                        },
                    })
                }
            };

            blocks.push(Block {
                attn,
                mlp_norm: to_dev(l.get(e, "post_attention_layernorm.weight")?)?,
                mlp,
            });
        }

        // V3's multi-token-prediction head: one whole extra block per predicted
        // token, filed at `layers.{n_layer}` and up. Not implemented here any
        // more than on the CPU, and said out loud for the same reason -- what a
        // loader deliberately skips and what it forgets look identical from
        // outside, and the check below would refuse every V3 checkpoint.
        let mtp = spec.config.num(&["num_nextn_predict_layers"]).unwrap_or(0);
        for i in spec.n_layer..spec.n_layer + mtp {
            model.skip_under(&format!("layers.{i}."));
        }

        let (embed, head, tied) = embedding(&ld, &vb, &model, "embed_tokens.weight", &spec)?;

        let final_norm = to_dev(model.get(e, "norm.weight")?)?;

        let left = unread(paths, &vb.seen(), &vb.skipped())?;
        if !left.is_empty() {
            return Err(unread_error(&spec.arch.to_string(), &left).into());
        }

        // YaRN's frequencies, chosen by the CPU engine's builder so that there
        // is one implementation of the ramp rather than two.
        let rope = build_rope(rope_dim, spec.n_ctx, spec.rope_theta, &spec.config)?;
        let (cos, sin, half) = rope.tables();
        let cos = Tensor::from_slice(cos, (spec.n_ctx, half), &device)?.to_dtype(compute)?;
        let sin = Tensor::from_slice(sin, (spec.n_ctx, half), &device)?.to_dtype(compute)?;

        Ok(GpuDeepSeek {
            cache: (0..spec.n_layer).map(|_| None).collect(),
            blocks,
            embed,
            head,
            tied,
            final_norm,
            cos,
            sin,
            mla,
            router,
            device,
            dtype: compute,
            quant,
            pos: 0,
            spec,
        })
    }

    pub fn device_label(&self) -> String {
        label(&self.device, self.dtype, self.quant)
    }

    fn rewind(&mut self, len: usize) -> Res<()> {
        if len >= self.pos {
            return Ok(());
        }
        if len == 0 {
            self.cache.iter_mut().for_each(|s| *s = None);
        } else {
            for slot in self.cache.iter_mut() {
                if let Some((c, pe)) = slot.take() {
                    *slot = Some((c.narrow(0, 0, len)?, pe.narrow(0, 0, len)?));
                }
            }
        }
        self.pos = len;
        Ok(())
    }

    /// Rotate the last `qk_rope` numbers of every head, interleaved.
    ///
    /// `rope_i` wants `[batch, heads, seq, dim]`, which is why the caller hands
    /// this a tensor already in that shape.
    fn rotate(&self, x: &Tensor, pos0: usize, m: usize) -> candle_core::Result<Tensor> {
        let cos = self.cos.narrow(0, pos0, m)?.contiguous()?;
        let sin = self.sin.narrow(0, pos0, m)?.contiguous()?;
        rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn run(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let m = tokens.len();
        let spec = &self.spec;
        let e = spec.n_embd;
        let mla = self.mla;
        let (nope, rope_dim, lat, vh) = (mla.qk_nope, mla.qk_rope, mla.kv_lora, mla.v_head);
        let (heads, qh) = (mla.n_head, mla.q_head());
        let pos0 = self.pos;

        if pos0 + m > spec.n_ctx {
            return Err(format!("context window of {} tokens is full", spec.n_ctx).into());
        }

        let ids = Tensor::from_slice(tokens, (m,), &self.device)?;
        let mut x = self.embed.rows(&ids)?.to_dtype(self.dtype)?;

        let mask = match m > 1 {
            true => Some(causal_mask(m, pos0, &self.device, self.dtype)?),
            false => None,
        };

        for (l, blk) in self.blocks.iter().enumerate() {
            let h = ops::rms_norm(&x, &blk.attn.norm, spec.eps)?;

            // ---- Queries ---------------------------------------------------
            let q = match &blk.attn.q {
                Query::Direct(p) => p.forward(&h)?,
                Query::Compressed { down, norm, up } => {
                    let mid = ops::rms_norm(&down.forward(&h)?.contiguous()?, norm, spec.eps)?;
                    up.forward(&mid)?
                }
            };
            // [m, heads * qh] -> [1, heads, m, qh]
            let q = q.reshape((1, m, heads, qh))?.transpose(1, 2)?.contiguous()?;
            let q_nope = q.narrow(3, 0, nope)?.contiguous()?;
            let q_pe = self.rotate(&q.narrow(3, nope, rope_dim)?, pos0, m)?;

            // ---- Compress this batch into the cache ------------------------
            let kv = blk.attn.kv_a.forward(&h)?;
            // The norm goes on *before* the cache: what is stored is what the
            // up-projections expect to be handed.
            let c = ops::rms_norm(
                &kv.narrow(1, 0, lat)?.contiguous()?,
                &blk.attn.kv_a_norm,
                spec.eps,
            )?;
            // One rotated key per position, shared by every head -- so it is
            // rotated as a single "head" and squeezed back.
            let k_pe = self
                .rotate(&kv.narrow(1, lat, rope_dim)?.reshape((1, 1, m, rope_dim))?, pos0, m)?
                .reshape((m, rope_dim))?;

            let (c, k_pe) = match self.cache[l].take() {
                None => (c, k_pe),
                Some((pc, ppe)) => (Tensor::cat(&[&pc, &c], 0)?, Tensor::cat(&[&ppe, &k_pe], 0)?),
            };
            self.cache[l] = Some((c.clone(), k_pe.clone()));
            let total = pos0 + m;

            // ---- Score -----------------------------------------------------
            //
            // The absorption, as one batched matmul: every head's key matrix
            // applied to that head's query, rather than to every cached
            // position. `[heads, m, nope] @ [heads, nope, lat]`.
            let q_lat = q_nope.squeeze(0)?.matmul(&blk.attn.uk)?;
            let content = q_lat.matmul(&c.t()?.contiguous()?.broadcast_as((heads, lat, total))?)?;
            let position = q_pe
                .squeeze(0)?
                .matmul(&k_pe.t()?.contiguous()?.broadcast_as((heads, rope_dim, total))?)?;
            let mut scores = ((content + position)? * mla.softmax_scale as f64)?;
            if let Some(msk) = &mask {
                scores = scores.broadcast_add(&msk.squeeze(0)?)?;
            }
            let att = ops::softmax_last_dim(&scores)?;

            // Average the *compressed* vectors, then decompress once.
            let acc = att.matmul(&c.broadcast_as((heads, total, lat))?)?;
            let out = acc.matmul(&blk.attn.uv)?;
            // [heads, m, vh] -> [m, heads * vh]
            let out = out.transpose(0, 1)?.reshape((m, heads * vh))?;
            x = (x + blk.attn.o.forward(&out)?)?;

            // ---- The MLP, or the mixture -----------------------------------
            let h = ops::rms_norm(&x, &blk.mlp_norm, spec.eps)?;
            let out = match &blk.mlp {
                Mlp::Dense(f) => f.forward(&h)?,
                Mlp::Moe(moe) => self.mixture(moe, &h, m, e)?,
            };
            x = (x + out)?;
        }

        self.pos += m;

        let last = x.i(m - 1)?.unsqueeze(0)?;
        let last = ops::rms_norm(&last, &self.final_norm, spec.eps)?;
        let logits = self.head.forward(&last)?.to_dtype(DType::F32)?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }

    /// The mixture, grouped by expert.
    ///
    /// The routing itself runs on the host, on `[m, n_experts]` logits, through
    /// [`Router::route`] — the CPU engine's own function, so that
    /// group-limited selection and V3's bias-steered choice have one
    /// implementation rather than two that must be kept level with each other.
    /// What comes back is a handful of (expert, weight) pairs per token, which
    /// is inverted here into a row list per expert.
    ///
    /// Then, per expert that anybody picked: gather its rows, run it once over
    /// them, scale each row by that token's weight, and add the result back
    /// where it came from. `index_add` is the scatter, and it is an add rather
    /// than a write because a token's output is the *sum* over its experts.
    fn mixture(&self, moe: &Moe, h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
        let logits = moe.gate.forward(h)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        let mut by_expert: Vec<(Vec<u32>, Vec<f32>)> =
            vec![(Vec::new(), Vec::new()); self.router.n_experts];
        let mut picks = Vec::with_capacity(self.router.top_k);
        for (i, row) in logits.iter().enumerate() {
            self.router.route(row, moe.bias.as_deref(), &mut picks);
            for &(expert, w) in &picks {
                by_expert[expert].0.push(i as u32);
                by_expert[expert].1.push(w);
            }
        }

        // The shared experts run for every token whatever the router says, so
        // they need no grouping and make a convenient accumulator.
        let mut out = match &moe.shared {
            Some(shared) => shared.forward(h)?,
            None => Tensor::zeros((m, e), self.dtype, &self.device)?,
        };
        for (i, (rows, weights)) in by_expert.iter().enumerate() {
            // Where the saving is: at decode a token reaches `top_k` experts,
            // so six of sixty-four run and the rest are never touched.
            if rows.is_empty() {
                continue;
            }
            let k = rows.len();
            let idx = Tensor::from_slice(rows, (k,), &self.device)?;
            let xs = h.index_select(&idx, 0)?;
            let w = Tensor::from_slice(weights, (k, 1), &self.device)?.to_dtype(self.dtype)?;
            let y = moe.experts[i].forward(&xs)?.broadcast_mul(&w)?;
            out = out.index_add(&idx, &y, 0)?;
        }
        Ok(out)
    }

    fn params(&self) -> usize {
        let n = |t: &Tensor| t.elem_count();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let a = &b.attn;
                let q = match &a.q {
                    Query::Direct(p) => p.params(),
                    Query::Compressed { down, norm, up } => {
                        down.params() + n(norm) + up.params()
                    }
                };
                let mlp = match &b.mlp {
                    Mlp::Dense(f) => f.params(),
                    Mlp::Moe(moe) => {
                        moe.gate.params()
                            + moe.bias.as_ref().map_or(0, |v| v.len())
                            + moe.experts.iter().map(|f| f.params()).sum::<usize>()
                            + moe.shared.as_ref().map_or(0, |f| f.params())
                    }
                };
                q + a.kv_a.params()
                    + n(&a.kv_a_norm)
                    + n(&a.uk)
                    + n(&a.uv)
                    + a.o.params()
                    + n(&a.norm)
                    + n(&b.mlp_norm)
                    + mlp
            })
            .sum();
        let head = match self.spec.tie_embeddings {
            true => 0,
            false => self.head.params(),
        };
        blocks + self.embed.params() + head + n(&self.final_norm)
    }

    fn memory_bytes(&self) -> usize {
        let per = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let a = &b.attn;
                let q = match &a.q {
                    Query::Direct(p) => p.bytes(),
                    Query::Compressed { down, up, .. } => down.bytes() + up.bytes(),
                };
                let mlp = match &b.mlp {
                    Mlp::Dense(f) => f.bytes(),
                    Mlp::Moe(moe) => {
                        moe.gate.bytes()
                            + moe.experts.iter().map(|f| f.bytes()).sum::<usize>()
                            + moe.shared.as_ref().map_or(0, |f| f.bytes())
                    }
                };
                q + a.kv_a.bytes() + per(&a.uk) + per(&a.uv) + a.o.bytes() + mlp
            })
            .sum();
        let head = if self.tied { 0 } else { self.head.bytes() };
        blocks + self.embed.bytes() + head
    }
}

/// Prompt tokens per prefill pass.
///
/// The same as the Llama backend's, and for the same reason: one matmul per
/// layer beats `m` of them, up to the point where the attention matrix stops
/// fitting. The mixture no longer has an opinion about it — grouped by
/// expert, a chunk costs roughly `top_k / n_experts` of what its size suggests,
/// whatever the size.
const PREFILL_CHUNK: usize = 512;

impl Session for GpuDeepSeek {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let mut last = Vec::new();
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            last = self.run(chunk)?;
        }
        Ok(last)
    }

    fn cached(&self) -> usize {
        self.pos
    }

    fn truncate(&mut self, len: usize) -> Res<()> {
        self.rewind(len)
    }

    fn label(&self) -> String {
        self.device_label()
    }

    fn param_count(&self) -> usize {
        self.params()
    }

    fn weight_bytes(&self) -> usize {
        self.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::model::{Json, Transformer};
    use kvad::serde_json;
    use std::collections::HashMap;

    /// A DeepSeek small enough to build from random numbers, with every width
    /// a multiple of 32 so the quantisers will take it.
    ///
    /// Two layers, because `first_k_dense_replace` is 1: the first is an
    /// ordinary MLP and the second routes, so one fixture covers both arms.
    fn tiny(cfg: serde_json::Value) -> (Spec, std::path::PathBuf, String) {
        let tag = cfg["__tag"].as_str().unwrap().to_string();
        let spec = Spec::from_config(Json::new(cfg)).unwrap();
        let path = write_tensors(&spec, &tag);
        (spec, path, tag)
    }

    fn base_config(tag: &str) -> serde_json::Value {
        serde_json::json!({
            "__tag": tag,
            "model_type": "deepseek_v2",
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 64,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 16,
            "v_head_dim": 32,
            "kv_lora_rank": 32,
            "q_lora_rank": serde_json::Value::Null,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "vocab_size": 64,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "scoring_func": "softmax",
            "topk_method": "greedy",
        })
    }

    fn write_tensors(spec: &Spec, tag: &str) -> std::path::PathBuf {
        let d = Device::Cpu;
        let mla = Mla::read(&spec.config, spec.n_head).unwrap();
        let layout = Layout::read(&spec.config);
        let router = Router::read(spec).unwrap();
        let e = spec.n_embd;
        let inter: usize = spec.config.num(&["intermediate_size"]).unwrap();
        let moe_inter: usize = spec.config.num(&["moe_intermediate_size"]).unwrap();
        let rand = |r: usize, c: usize| Tensor::randn(0f32, 0.05f32, (r, c), &d).unwrap();
        // Norm gains are random, not ones: all-ones makes the scaling step the
        // identity, so a backend that dropped a norm would still look right.
        let vec1 = |n: usize| Tensor::randn(1f32, 0.05f32, n, &d).unwrap();

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("model.embed_tokens.weight".into(), rand(spec.vocab_size, e));
        t.insert("model.norm.weight".into(), vec1(e));
        for i in 0..spec.n_layer {
            let p = format!("model.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), vec1(e));
            t.insert(format!("{p}.post_attention_layernorm.weight"), vec1(e));
            let sa = format!("{p}.self_attn");
            t.insert(format!("{sa}.q_proj.weight"), rand(mla.n_head * mla.q_head(), e));
            t.insert(
                format!("{sa}.kv_a_proj_with_mqa.weight"),
                rand(mla.kv_lora + mla.qk_rope, e),
            );
            t.insert(format!("{sa}.kv_a_layernorm.weight"), vec1(mla.kv_lora));
            t.insert(
                format!("{sa}.kv_b_proj.weight"),
                rand(mla.n_head * (mla.qk_nope + mla.v_head), mla.kv_lora),
            );
            t.insert(format!("{sa}.o_proj.weight"), rand(e, mla.n_head * mla.v_head));

            let ffn = |t: &mut HashMap<String, Tensor>, prefix: &str, width: usize| {
                t.insert(format!("{prefix}.gate_proj.weight"), rand(width, e));
                t.insert(format!("{prefix}.up_proj.weight"), rand(width, e));
                t.insert(format!("{prefix}.down_proj.weight"), rand(e, width));
            };
            match layout.is_moe(i, router.n_experts) {
                false => ffn(&mut t, &format!("{p}.mlp"), inter),
                true => {
                    t.insert(format!("{p}.mlp.gate.weight"), rand(router.n_experts, e));
                    for x in 0..router.n_experts {
                        ffn(&mut t, &format!("{p}.mlp.experts.{x}"), moe_inter);
                    }
                    if layout.n_shared > 0 {
                        ffn(&mut t, &format!("{p}.mlp.shared_experts"), moe_inter * layout.n_shared);
                    }
                }
            }
        }

        let path = std::env::temp_dir()
            .join(format!("gpu-ds-{}-{tag}.safetensors", std::process::id()));
        candle_core::safetensors::save(&t, &path).unwrap();
        path
    }

    fn cpu_model(path: &std::path::PathBuf, spec: &Spec) -> kvad::model::deepseek::Model {
        let ckpt = kvad::weights::Checkpoint::open(std::slice::from_ref(path)).unwrap();
        let src = kvad::qcache::Live::new(&ckpt, kvad::quant::Precision::F32);
        kvad::model::deepseek::Model::load(&src, spec.clone()).unwrap()
    }

    fn engines_differ_by(path: &std::path::PathBuf, spec: &Spec, tokens: &[u32]) -> f32 {
        let mut gpu = GpuDeepSeek::load(
            std::slice::from_ref(path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        )
        .unwrap();
        let mine = gpu.forward(tokens).unwrap();

        let cpu = cpu_model(path, spec);
        let mut cache = kvad::model::KvCache::new(spec);
        let theirs = cpu.forward_batch(tokens, &mut cache);

        assert_eq!(mine.len(), theirs.len());
        mine.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
    }

    /// The whole architecture against the engine that was written by hand.
    ///
    /// This is the only test that can see whether the absorption is right.
    /// `W_UK` folded onto the query and `W_UK` applied to the keys are the same
    /// arithmetic rearranged, so a backend that rearranges it wrongly — a
    /// transpose, a head axis in the wrong place, the norm on the far side of
    /// the cache — produces a model that runs at full speed and attends to the
    /// wrong thing.
    #[test]
    fn latent_attention_agrees_with_the_cpu_engine() {
        let (spec, path, _) = tiny(base_config("agree"));
        // More than one token, so the causal mask and the batched path are
        // exercised rather than the single-position case.
        let worst = engines_differ_by(&path, &spec, &[1, 2, 3, 4, 5]);
        assert!(worst < 1e-4, "logits disagree by {worst}");
        std::fs::remove_file(&path).unwrap();
    }

    /// V3's query compression, its sigmoid scoring and its bias-steered choice.
    ///
    /// The same file runs both models, and the four differences all live in the
    /// router and the query projection — so a fixture that sets them is the
    /// only way this backend's V3 path is ever executed.
    #[test]
    fn the_v3_shapes_agree_too() {
        let mut cfg = base_config("v3");
        cfg["__tag"] = "v3".into();
        cfg["q_lora_rank"] = 32.into();
        cfg["scoring_func"] = "sigmoid".into();
        cfg["topk_method"] = "noaux_tc".into();
        cfg["n_group"] = 2.into();
        cfg["topk_group"] = 1.into();
        cfg["routed_scaling_factor"] = 2.5.into();
        let spec = Spec::from_config(Json::new(cfg)).unwrap();
        let mut path = write_tensors(&spec, "v3");

        // The compressed query and V3's balancing bias, which the base fixture
        // does not carry.
        {
            let d = Device::Cpu;
            let mut extra: HashMap<String, Tensor> = HashMap::new();
            let mla = Mla::read(&spec.config, spec.n_head).unwrap();
            let rank = mla.q_lora.unwrap();
            for i in 0..spec.n_layer {
                let sa = format!("model.layers.{i}.self_attn");
                extra.insert(
                    format!("{sa}.q_a_proj.weight"),
                    Tensor::randn(0f32, 0.05f32, (rank, spec.n_embd), &d).unwrap(),
                );
                extra.insert(
                    format!("{sa}.q_a_layernorm.weight"),
                    Tensor::randn(1f32, 0.05f32, rank, &d).unwrap(),
                );
                extra.insert(
                    format!("{sa}.q_b_proj.weight"),
                    Tensor::randn(0f32, 0.05f32, (mla.n_head * mla.q_head(), rank), &d).unwrap(),
                );
            }
            extra.insert(
                "model.layers.1.mlp.gate.e_score_correction_bias".into(),
                Tensor::randn(0f32, 0.5f32, 4, &d).unwrap(),
            );
            let mut all = candle_core::safetensors::load(&path, &d).unwrap();
            all.remove("model.layers.0.self_attn.q_proj.weight");
            all.remove("model.layers.1.self_attn.q_proj.weight");
            all.extend(extra);
            path = std::env::temp_dir()
                .join(format!("gpu-ds-{}-v3-full.safetensors", std::process::id()));
            candle_core::safetensors::save(&all, &path).unwrap();
        }

        let worst = engines_differ_by(&path, &spec, &[1, 2, 3, 4, 5]);
        assert!(worst < 1e-4, "logits disagree by {worst}");
        std::fs::remove_file(&path).unwrap();
    }

    /// YaRN, which every published DeepSeek config turns on and the fixtures
    /// above leave off.
    ///
    /// Two things ride on it and neither is visible without a `rope_scaling`
    /// block: the interpolated frequencies, and the attention temperature. The
    /// second is a plain multiplier on every score — 1.59 at V2-Lite's
    /// settings — and a backend that used the textbook `1/sqrt(d)` instead
    /// would have a softmax far too sharp at every position, including
    /// position zero, while still looking like a model.
    #[test]
    fn yarn_frequencies_and_temperature_agree() {
        let mut cfg = base_config("yarn");
        cfg["rope_scaling"] = serde_json::json!({
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 8,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        });
        let spec = Spec::from_config(Json::new(cfg)).unwrap();
        // The temperature correction is the whole point of this fixture, so
        // say out loud that it is not 1.
        let mla = Mla::read(&spec.config, spec.n_head).unwrap();
        let plain = 1.0 / ((mla.qk_nope + mla.qk_rope) as f32).sqrt();
        assert!(
            (mla.softmax_scale / plain - 1.589).abs() < 0.01,
            "expected YaRN to raise the temperature by ~1.589, got {}",
            mla.softmax_scale / plain
        );

        let path = write_tensors(&spec, "yarn");
        let worst = engines_differ_by(&path, &spec, &[1, 2, 3, 4, 5]);
        assert!(worst < 1e-4, "logits disagree by {worst}");
        std::fs::remove_file(&path).unwrap();
    }

    /// Decoding one token at a time must land where one pass over the prompt
    /// does. The cache here holds the compressed vector and the shared rotated
    /// key, so this is also the test that the two streams stay in step.
    #[test]
    fn the_latent_cache_agrees_with_a_single_pass() {
        let (spec, path, _) = tiny(base_config("cache"));
        let load = || {
            GpuDeepSeek::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap()
        };

        let whole = load().forward(&[1, 2, 3, 4]).unwrap();
        let mut stepped = load();
        let mut last = Vec::new();
        for t in [1u32, 2, 3, 4] {
            last = stepped.forward(&[t]).unwrap();
        }
        let worst = whole.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-4, "the cache changes the answer by {worst}");
        assert_eq!(stepped.cached(), 4);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn both_engines_count_the_same_parameters() {
        let (spec, path, _) = tiny(base_config("params"));
        let gpu = GpuDeepSeek::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        )
        .unwrap();
        assert_eq!(gpu.param_count(), cpu_model(&path, &spec).param_count());
        std::fs::remove_file(&path).unwrap();
    }

    /// V3's multi-token-prediction head is not implemented, and *deliberately
    /// not read* has to be distinguishable from *forgotten* — otherwise this
    /// backend refuses every real V3 checkpoint for carrying weights nobody
    /// wanted. The CPU engine learned this the same way.
    #[test]
    fn the_prediction_head_is_skipped_on_purpose() {
        let mut cfg = base_config("mtp");
        cfg["num_nextn_predict_layers"] = 1.into();
        let spec = Spec::from_config(Json::new(cfg)).unwrap();
        let path = write_tensors(&spec, "mtp");

        let d = Device::Cpu;
        let mut all = candle_core::safetensors::load(&path, &d).unwrap();
        let p = format!("model.layers.{}", spec.n_layer);
        for (name, shape) in [
            ("enorm.weight", vec![spec.n_embd]),
            ("hnorm.weight", vec![spec.n_embd]),
            ("eh_proj.weight", vec![spec.n_embd, 2 * spec.n_embd]),
            ("shared_head.norm.weight", vec![spec.n_embd]),
        ] {
            all.insert(
                format!("{p}.{name}"),
                Tensor::randn(0f32, 0.05f32, shape, &d).unwrap(),
            );
        }
        let path = std::env::temp_dir()
            .join(format!("gpu-ds-{}-mtp-full.safetensors", std::process::id()));
        candle_core::safetensors::save(&all, &path).unwrap();

        assert!(
            GpuDeepSeek::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .is_ok(),
            "a head this build knowingly does not run must not refuse the load"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// And a weight nobody read still stops the load, prediction head or not.
    #[test]
    fn a_tensor_this_backend_never_reads_refuses_to_load() {
        let (spec, path, _) = tiny(base_config("unknown"));
        let d = Device::Cpu;
        let mut all = candle_core::safetensors::load(&path, &d).unwrap();
        all.insert(
            "model.layers.0.self_attn.q_norm.weight".into(),
            Tensor::randn(0f32, 0.05f32, 32, &d).unwrap(),
        );
        let path = std::env::temp_dir()
            .join(format!("gpu-ds-{}-unknown-full.safetensors", std::process::id()));
        candle_core::safetensors::save(&all, &path).unwrap();

        let error = match GpuDeepSeek::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a weight nobody reads must not load quietly"),
        };
        assert!(error.contains("q_norm"), "it should name the tensor: {error}");
        std::fs::remove_file(&path).unwrap();
    }
}
