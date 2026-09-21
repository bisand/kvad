//! The Llama forward pass again — this time on the GPU, in candle.
//!
//! # What changed, and what did not
//!
//! Read this next to [`kvad::model::llama`]. The structure is line for line the
//! same: embed, then per layer `x = x + Attention(RMSNorm(x))` and
//! `x = x + SwiGLU(RMSNorm(x))`, then a final norm and the output head. RoPE
//! still rotates Q and K, grouped-query attention still shares KV heads, the
//! cache still grows by one position per token.
//!
//! What changed is who does the arithmetic. Every `matvec_bt` we wrote by hand
//! is now `Tensor::matmul`, and the loop over heads is a batched matmul over a
//! `[1, n_head, seq, head_dim]` tensor. `candle_nn::rotary_emb::rope` is the
//! same `rotate_half` convention implemented in `tensor.rs`, and
//! `candle_nn::ops::rms_norm` is the same scale-only normalisation.
//!
//! That is the point of having written it by hand first: nothing here is
//! mysterious, it is just faster.
//!
//! # Batching is free here
//!
//! The CPU engine needed a separate `forward_batch`, because a matrix-vector
//! product and a matrix-matrix product are genuinely different kernels. Here
//! there is one `forward`, and `m = 1` is simply the narrow case. The
//! framework's matmul does not care.

use crate::common::{
    causal_mask, check_block, embedding, label, linear, unread, unread_error, Embed, Loader, Proj,
    Reader, Stored,
};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, rotary_emb, VarBuilder};
use kvad::model::{Arch, Session, Spec};

type Res<T> = Result<T, Box<dyn std::error::Error>>;





struct Block {
    attn_norm: Tensor,
    q: Proj,
    q_b: Option<Tensor>,
    k: Proj,
    k_b: Option<Tensor>,
    v: Proj,
    v_b: Option<Tensor>,
    o: Proj,
    /// Qwen3's per-head RMSNorm on the queries and the keys, applied after the
    /// projection and *before* RoPE. One vector of `head_dim`, shared by every
    /// head.
    ///
    /// Absent in Llama and Qwen2. Nothing in the checkpoint announces which
    /// one is loading, so the presence of these weights *is* the test — which
    /// also means forgetting to read them is silent, and produces a model that
    /// runs at full speed and talks nonsense.
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    mlp_norm: Tensor,
    gate: Proj,
    up: Proj,
    down: Proj,
}

pub struct GpuLlama {
    spec: Spec,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    embed: Embed,
    /// The output head. Shares the embedding's storage when the model ties
    /// them and both are quantised; a transposed copy otherwise.
    head: Proj,
    /// Whether `head` is the same allocation as `embed`, so the memory
    /// accounting does not count it twice.
    tied: bool,
    blocks: Vec<Block>,
    final_norm: Tensor,
    /// Precomputed rotations, `[n_ctx, head_dim / 2]`.
    cos: Tensor,
    sin: Tensor,
    /// Per layer, `[1, n_kv_head, seq, head_dim]` for keys and values.
    kv: Vec<Option<(Tensor, Tensor)>>,
    pos: usize,
}



/// RMSNorm every head of a projection shaped `[.., heads, head_dim]`, if this
/// model has the weights for it.
///
/// The mirror of `norm_heads` in [`kvad::model::llama`], and much shorter for
/// one reason: `rms_norm` normalises along the last axis, so putting the heads
/// on the axis before it turns a loop over heads into a single kernel call.
///
/// `None` is every other model in this family, and costs one branch per layer.
fn norm_heads(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> candle_core::Result<Tensor> {
    match weight {
        None => Ok(x.clone()),
        Some(w) => ops::rms_norm(&x.contiguous()?, w, eps),
    }
}

impl GpuLlama {
    pub fn load(
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
        cache: &Vault,
    ) -> Res<Self> {
        // Quantising reads f32 and produces blocks, so in that mode the
        // weights arrive as f32 and each one is converted and dropped in turn
        // — peak memory is the quantised model plus a single tensor, not two
        // full copies. Activations then stay f32 too, which is what candle's
        // quantised kernels expect.
        let load_dtype = if quant.is_some() { DType::F32 } else { dtype };
        let compute = load_dtype;

        // The quantiser reads from host memory, so in that mode the checkpoint
        // is mapped on the CPU and each tensor is quantised *onto* the device
        // one at a time. Without quantisation the weights go straight to the
        // device in their final form.
        check_block(
            quant,
            &[
                ("hidden size", spec.n_embd),
                ("MLP width", spec.intermediate),
                ("attention output", spec.n_head * spec.head_dim),
                ("KV width", spec.kv_dim()),
            ],
        )?;

        let load_dev = if quant.is_some() { Device::Cpu } else { device.clone() };

        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, load_dtype, &load_dev)? };
        let vb = Reader::new(vb);

        // Dense tensors (norms, embeddings) still have to make the trip.
        let to_dev = |t: Tensor| -> Res<Tensor> { Ok(t.to_device(&device)?) };

        let (e, hd) = (spec.n_embd, spec.head_dim);
        let (qd, kvd) = (spec.n_head * hd, spec.kv_dim());
        let model = vb.pp("model");

        // HuggingFace stores `nn.Linear` weights as [out, in]. Who transposes
        // and when is `Stored`'s business now; this loader only says which way
        // round the file has it.
        let ld = Loader::new(quant, device.clone(), cache);
        let load_t = |vb: &Reader<'_>, name: &str, out: usize, inp: usize| -> Res<Proj> {
            ld.proj(vb, name, out, inp, Stored::OutIn)
        };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let l = model.pp(format!("layers.{i}"));
            let attn = l.pp("self_attn");
            let mlp = l.pp("mlp");
            blocks.push(Block {
                attn_norm: to_dev(l.get(e, "input_layernorm.weight")?)?,
                q: load_t(&attn, "q_proj.weight", qd, e)?,
                q_b: attn.try_get(qd, "q_proj.bias").map(&to_dev).transpose()?,
                k: load_t(&attn, "k_proj.weight", kvd, e)?,
                k_b: attn.try_get(kvd, "k_proj.bias").map(&to_dev).transpose()?,
                v: load_t(&attn, "v_proj.weight", kvd, e)?,
                v_b: attn.try_get(kvd, "v_proj.bias").map(&to_dev).transpose()?,
                o: load_t(&attn, "o_proj.weight", e, qd)?,
                q_norm: attn.try_get(hd, "q_norm.weight").map(&to_dev).transpose()?,
                k_norm: attn.try_get(hd, "k_norm.weight").map(&to_dev).transpose()?,
                mlp_norm: to_dev(l.get(e, "post_attention_layernorm.weight")?)?,
                gate: load_t(&mlp, "gate_proj.weight", spec.intermediate, e)?,
                up: load_t(&mlp, "up_proj.weight", spec.intermediate, e)?,
                down: load_t(&mlp, "down_proj.weight", e, spec.intermediate)?,
            });
        }

        let (embed, head, tied) = embedding(&ld, &vb, &model, "embed_tokens.weight", &spec)?;

        let final_norm = to_dev(model.get(e, "norm.weight")?)?;

        // Everything has been read, so anything left in the file is a part of
        // this model that is not running.
        let left = unread(paths, &vb.seen(), &vb.skipped())?;
        if !left.is_empty() {
            return Err(unread_error("llama", &left).into());
        }

        // The same rotation table as `Rope::new`, built once on the device.
        let half = hd / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / spec.rope_theta.powf(2.0 * i as f32 / hd as f32))
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), &device)?;
        let positions: Vec<f32> = (0..spec.n_ctx).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (spec.n_ctx, 1), &device)?;
        let angles = positions.matmul(&inv)?;
        let cos = angles.cos()?.to_dtype(compute)?;
        let sin = angles.sin()?.to_dtype(compute)?;

        Ok(GpuLlama {
            kv: (0..spec.n_layer).map(|_| None).collect(),
            blocks,
            embed,
            head,
            tied,
            final_norm,
            cos,
            sin,
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

    /// Drop everything after `len` positions, for reuse across chat turns.
    fn rewind(&mut self, len: usize) -> Res<()> {
        if len >= self.pos {
            return Ok(());
        }
        if len == 0 {
            self.kv.iter_mut().for_each(|s| *s = None);
        } else {
            for slot in self.kv.iter_mut() {
                if let Some((k, v)) = slot.take() {
                    *slot = Some((k.narrow(2, 0, len)?, v.narrow(2, 0, len)?));
                }
            }
        }
        self.pos = len;
        Ok(())
    }

    /// Run `tokens` and return logits for the last one.
    fn run(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let m = tokens.len();
        let spec = &self.spec;
        let hd = spec.head_dim;
        let (n_head, n_kv) = (spec.n_head, spec.n_kv_head);
        let group = spec.group_size();
        let pos0 = self.pos;
        let scale = 1.0 / (hd as f64).sqrt();

        let ids = Tensor::from_slice(tokens, (m,), &self.device)?;
        let mut x = self.embed.rows(&ids)?.to_dtype(self.dtype)?;

        let cos = self.cos.narrow(0, pos0, m)?.contiguous()?;
        let sin = self.sin.narrow(0, pos0, m)?.contiguous()?;
        let mask = match m > 1 {
            true => Some(causal_mask(m, pos0, &self.device, self.dtype)?),
            false => None,
        };

        for (i, blk) in self.blocks.iter().enumerate() {
            let h = ops::rms_norm(&x, &blk.attn_norm, spec.eps)?;

            let q = linear(&h, &blk.q, blk.q_b.as_ref())?;
            let k = linear(&h, &blk.k, blk.k_b.as_ref())?;
            let v = linear(&h, &blk.v, blk.v_b.as_ref())?;

            // [m, heads * hd] -> [1, m, heads, hd]: the heads are split out
            // with `hd` still last, which is the layout the per-head norm
            // below wants.
            let q = q.reshape((1, m, n_head, hd))?;
            let k = k.reshape((1, m, n_kv, hd))?;

            // Qwen3 normalises each head of Q and K here, between the
            // projection and the rotation. Nothing else in this family does,
            // and for everything else this is a no-op.
            let q = norm_heads(&q, blk.q_norm.as_ref(), spec.eps)?;
            let k = norm_heads(&k, blk.k_norm.as_ref(), spec.eps)?;

            // -> [1, heads, m, hd], which is what the rotary kernel and the
            // batched attention matmuls want.
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.reshape((1, m, n_kv, hd))?.transpose(1, 2)?.contiguous()?;

            let q = rotary_emb::rope(&q, &cos, &sin)?;
            let k = rotary_emb::rope(&k, &cos, &sin)?;

            // Append to the cache along the sequence axis.
            let (k, v) = match self.kv[i].take() {
                None => (k, v),
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            };
            self.kv[i] = Some((k.clone(), v.clone()));

            // Fold the query heads onto their KV head rather than copying
            // the KV head out once per query head. `repeat_kv` was 9 ms a
            // layer at 6.5k of context and the transpose behind it another
            // 7 ms, against 0.4 ms for the matmul they were shaping data
            // for — 28 layers of that is most of a 640 ms token. Reshaping
            // Q instead costs nothing: heads are laid out `kv * group + g`,
            // which is exactly `[kv][group][m]` already, so the same bytes
            // read as the grouped rows the matmul wants.
            let seq = k.dim(2)?;
            let kt = k.transpose(2, 3)?.contiguous()?;
            let qg = q.reshape((1, n_kv, group * m, hd))?;
            let mut att = (qg.matmul(&kt)? * scale)?;
            if let Some(msk) = &mask {
                // The mask is per query row, and the rows are grouped by KV
                // head here. Back to head-major to add it, and back again.
                att = att
                    .reshape((1, n_head, m, seq))?
                    .broadcast_add(msk)?
                    .reshape((1, n_kv, group * m, seq))?;
            }
            let att = ops::softmax_last_dim(&att)?;

            let out = att.matmul(&v.contiguous()?)?;
            let out = out.reshape((1, n_head, m, hd))?;
            let out = out.transpose(1, 2)?.reshape((m, n_head * hd))?;
            x = (x + linear(&out, &blk.o, None)?)?;

            let h = ops::rms_norm(&x, &blk.mlp_norm, spec.eps)?;
            let gate = ops::silu(&linear(&h, &blk.gate, None)?)?;
            let up = linear(&h, &blk.up, None)?;
            x = (x + linear(&(gate * up)?, &blk.down, None)?)?;
        }

        self.pos += m;

        // Only the last position predicts anything we need.
        let last = x.i(m - 1)?.unsqueeze(0)?;
        let last = ops::rms_norm(&last, &self.final_norm, spec.eps)?;
        let logits = self.head.forward(&last)?.to_dtype(DType::F32)?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }

    fn params(&self) -> usize {
        let n = |t: &Tensor| t.elem_count();
        // As on the CPU: the optional vectors are exactly the ones that differ
        // between Qwen2 and Qwen3, so leaving them out would report the same
        // count for a model that has them and one that does not.
        let opt = |t: &Option<Tensor>| t.as_ref().map_or(0, n);
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.q.params()
                    + b.k.params()
                    + b.v.params()
                    + b.o.params()
                    + b.gate.params()
                    + b.up.params()
                    + b.down.params()
                    + n(&b.attn_norm)
                    + n(&b.mlp_norm)
                    + opt(&b.q_b)
                    + opt(&b.k_b)
                    + opt(&b.v_b)
                    + opt(&b.q_norm)
                    + opt(&b.k_norm)
            })
            .sum();
        // A head that is not counted is a head nobody notices is missing. It
        // is a separate parameter exactly when the model does not tie — and how
        // many copies of it *this* backend keeps is a question about
        // allocations, so `tied` is the wrong flag to ask here. Using it would
        // make a dense load and a quantised load of one model report different
        // parameter counts.
        let head = match self.spec.tie_embeddings {
            true => 0,
            false => self.head.params(),
        };
        blocks + self.embed.params() + head + n(&self.final_norm)
    }

    fn memory_bytes(&self) -> usize {
        let per = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let opt = |t: &Option<Tensor>| t.as_ref().map_or(0, per);
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.q.bytes() + b.k.bytes() + b.v.bytes() + b.o.bytes() + b.gate.bytes()
                    + b.up.bytes()
                    + b.down.bytes()
                    + per(&b.attn_norm)
                    + per(&b.mlp_norm)
                    + opt(&b.q_b)
                    + opt(&b.k_b)
                    + opt(&b.v_b)
            })
            .sum();
        // A tied head is the embedding, not a copy of it.
        let head = if self.tied { 0 } else { self.head.bytes() };
        blocks + self.embed.bytes() + head + per(&self.final_norm)
    }
}

impl Session for GpuLlama {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        // Chunked so that a long prompt does not allocate an attention matrix
        // of `[heads, m, m]` all at once.
        let mut logits = Vec::new();
        for part in tokens.chunks(PREFILL_CHUNK) {
            logits = self.run(part)?;
        }
        Ok(logits)
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

const PREFILL_CHUNK: usize = 512;


/// Weight quantisation for the GPU, in GGML's block formats.
///
/// `q8` and `q4` are the same scheme implemented by hand in
/// [`kvad::quant`]: blocks of 32 with a per-block scale. `q4k` is the
/// "k-quant" refinement — a second level of scales within a super-block, which
/// buys noticeably better quality at the same 4 bits.
pub fn parse_quant(s: &str) -> Option<Option<GgmlDType>> {
    match s.to_ascii_lowercase().as_str() {
        "none" | "off" | "dense" => Some(None),
        "q8" | "q8_0" => Some(Some(GgmlDType::Q8_0)),
        "q4" | "q4_0" => Some(Some(GgmlDType::Q4_0)),
        "q4k" | "q4_k" => Some(Some(GgmlDType::Q4K)),
        "q6k" | "q6_k" => Some(Some(GgmlDType::Q6K)),
        _ => None,
    }
}

pub fn parse_dtype(s: &str) -> Option<DType> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Some(DType::F32),
        "f16" | "fp16" | "half" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        _ => None,
    }
}

/// Build whichever architecture the config named, as a [`Session`].
///
/// The CPU engine has a registry for this and looks nothing up by hand; here
/// there is a branch per architecture, because adding one should be a visible
/// act rather than a line in a table. What they share is the vocabulary in
/// [`crate::common`] — the check that the checkpoint has nothing left in it
/// that the loader never asked for, and the cache that means the loader does
/// not quantise the same weights twice.
pub fn session(
    repo: &str,
    paths: &[std::path::PathBuf],
    spec: &Spec,
    dtype: DType,
    quant: Option<GgmlDType>,
    device: Device,
    progress: &mut dyn FnMut(&str),
) -> Res<Box<dyn Session>> {
    let arch = &spec.arch;
    if !supports(*arch) {
        return Err(format!(
            "the GPU backend implements {}; `{arch}` is not one of them.\n\
             Run it on the CPU engine instead:  kvad run",
            supported()
        )
        .into());
    }

    // Opened before the load and closed after it, because both of its
    // questions are answered by the load: a file is only whole once every
    // weight has been through it, and a mapped file is only known to be
    // missing something once the architecture has finished asking.
    let mut cache = Vault::open(repo, paths, spec, quant, progress);
    let session: Box<dyn Session> = if arch.is("llama") {
        Box::new(GpuLlama::load(paths, spec.clone(), dtype, quant, device, &cache)?)
    } else if arch.is("gpt2") {
        Box::new(crate::gpt2::GpuGpt2::load(paths, spec.clone(), dtype, quant, device, &cache)?)
    } else {
        Box::new(crate::deepseek::GpuDeepSeek::load(
            paths,
            spec.clone(),
            dtype,
            quant,
            device,
            &cache,
        )?)
    };
    cache.finish(progress);
    Ok(session)
}

/// The architectures this backend has an implementation for, by the id the
/// engine gives them.
///
/// One list, because there are now two questions about it. `session` answers
/// "can you run this?" by trying, and a caller that has to decide *before*
/// committing — the server picking a default backend — asks [`supports`].
/// Two lists would be a server that offers the GPU for a model this cannot
/// load.
///
/// `deepseek.rs` implements V3 as well, and V3 is not here: the dispatch
/// below has no arm for it, so no V3 checkpoint reaches that code. The arm
/// and this list go together when it gets one.
const IMPLEMENTED: [&str; 3] = ["llama", "gpt2", "deepseek_v2"];

/// Whether this backend can run `arch`, asked before anything is loaded.
pub fn supports(arch: Arch) -> bool {
    IMPLEMENTED.iter().any(|id| arch.is(id))
}

/// The architectures this backend can run, for an error message that does not
/// have to be kept in step by hand.
pub fn supported() -> String {
    IMPLEMENTED.join(", ")
}

/// Pick the best device available, unless one was named.
pub fn pick_device(name: Option<&str>) -> Res<Device> {
    let want = name.unwrap_or("auto").to_ascii_lowercase();
    Ok(match want.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        _ => Device::new_metal(0).or_else(|_| Device::new_cuda(0)).unwrap_or(Device::Cpu),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// What `supports` promises is what `session` has an arm for. Asked as a
    /// test because the promise is made to a caller that acts on it before
    /// there is a model to try — the server offers the GPU as a default on
    /// the strength of this answer.
    #[test]
    fn the_predicate_and_the_dispatch_name_the_same_architectures() {
        for id in IMPLEMENTED {
            assert!(supports(Arch::require(id)), "{id} is listed and not supported");
        }
        // Not an assertion about what *should* be: `deepseek.rs` runs V3 and
        // the dispatch has no arm for it, so a caller must not be told the
        // GPU will take one. This fails the day an arm is added, which is
        // the day this list changes.
        assert!(!supports(Arch::require("deepseek_v3")));
        assert!(supported().contains("deepseek_v2"));
    }

    /// A model small enough to build from random numbers, with every
    /// dimension a multiple of 32 so the quantisers will take it.
    pub(crate) fn tiny_spec() -> Spec {
        Spec {
            arch: Arch::require("llama"),
            n_layer: 1,
            n_head: 2,
            n_kv_head: 1,
            n_embd: 64,
            head_dim: 32,
            n_ctx: 16,
            vocab_size: 64,
            intermediate: 128,
            eps: 1e-5,
            rope_theta: 10000.0,
            tie_embeddings: true,
            cache: kvad::model::CacheShape { k: 32, v: 32 },
            config: kvad::model::Json::default(),
        }
    }

    /// Write a checkpoint the loader will accept, with or without its own
    /// output head.
    fn write_checkpoint(spec: &Spec, own_head: bool, tag: &str) -> std::path::PathBuf {
        write_tensors(spec, own_head, &[], tag)
    }

    /// The same, plus tensors the base Llama layout does not have: Qwen3's
    /// per-head norms, or something this backend does not implement at all.
    pub(crate) fn write_tensors(
        spec: &Spec,
        own_head: bool,
        extra: &[(String, Tensor)],
        tag: &str,
    ) -> std::path::PathBuf {
        let d = Device::Cpu;
        let (e, i) = (spec.n_embd, spec.intermediate);
        let (qd, kvd) = (spec.n_head * spec.head_dim, spec.kv_dim());
        let rand = |r: usize, c: usize| Tensor::randn(0f32, 0.02f32, (r, c), &d).unwrap();

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("model.embed_tokens.weight".into(), rand(spec.vocab_size, e));
        t.insert("model.norm.weight".into(), Tensor::ones(e, DType::F32, &d).unwrap());
        let p = "model.layers.0";
        t.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(e, DType::F32, &d).unwrap());
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::ones(e, DType::F32, &d).unwrap(),
        );
        t.insert(format!("{p}.self_attn.q_proj.weight"), rand(qd, e));
        t.insert(format!("{p}.self_attn.k_proj.weight"), rand(kvd, e));
        t.insert(format!("{p}.self_attn.v_proj.weight"), rand(kvd, e));
        t.insert(format!("{p}.self_attn.o_proj.weight"), rand(e, qd));
        t.insert(format!("{p}.mlp.gate_proj.weight"), rand(i, e));
        t.insert(format!("{p}.mlp.up_proj.weight"), rand(i, e));
        t.insert(format!("{p}.mlp.down_proj.weight"), rand(e, i));
        if own_head {
            t.insert("lm_head.weight".into(), rand(spec.vocab_size, e));
        }
        for (name, tensor) in extra {
            t.insert(name.clone(), tensor.clone());
        }

        // Unique per test as well as per process: the tests run in parallel
        // and would otherwise delete each other's checkpoints.
        let path = std::env::temp_dir().join(format!(
            "gpu-tiny-{}-{tag}-{own_head}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&t, &path).unwrap();
        path
    }

    /// The hand-written engine on the same checkpoint. Owned, so the mapped
    /// file and the quantising source it was built through can both go away.
    fn cpu_model(path: &std::path::PathBuf, spec: &Spec) -> kvad::model::llama::Model {
        let ckpt = kvad::weights::Checkpoint::open(std::slice::from_ref(path)).unwrap();
        let src = kvad::qcache::Live::new(&ckpt, kvad::quant::Precision::F32);
        kvad::model::llama::Model::load(&src, spec.clone()).unwrap()
    }

    /// The largest disagreement between the two engines on one checkpoint.
    ///
    /// Both in f32, so what is left is the order the sums happen in — anything
    /// above a rounding error means they are running different models, which is
    /// the only way one engine can tell that the other is wrong.
    fn engines_differ_by(path: &std::path::PathBuf, spec: &Spec, tokens: &[u32]) -> f32 {
        use kvad::model::Transformer;

        let mut gpu =
            GpuLlama::load(
                std::slice::from_ref(path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap();
        let theirs = gpu.forward(tokens).unwrap();

        let mut cache = kvad::model::KvCache::new(spec);
        let ours = cpu_model(path, spec).forward_batch(tokens, &mut cache);

        assert_eq!(theirs.len(), ours.len());
        theirs.iter().zip(&ours).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
    }

    /// Decoding a token at a time has to give what one pass over the whole
    /// prompt gives.
    ///
    /// GPT-2 and DeepSeek both had this test and the Llama family did not,
    /// which left the most-used path in this file checked only as a prefill:
    /// `engines_differ_by` runs one `forward` over several tokens and never
    /// asks the cache a second question. The grouped-query attention here
    /// folds query heads onto their KV head, so `m = 1` against a cache and
    /// `m = 4` in one go take visibly different routes through the same
    /// reshapes, and only one of them was covered.
    ///
    /// The spec is 2 query heads to 1 KV head, so the folding is exercised
    /// rather than being the identity it becomes at `group = 1`.
    #[test]
    fn the_cache_agrees_with_a_single_pass() {
        let spec = tiny_spec();
        assert!(spec.n_head > spec.n_kv_head, "this test is about grouped-query attention");
        let path = write_checkpoint(&spec, true, "llama-cache");
        let load = || {
            GpuLlama::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap()
        };

        let mut at_once = load();
        let whole = at_once.forward(&[1, 2, 3, 4]).unwrap();

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

    /// The bug this backend shipped with, and the only check that would have
    /// caught it.
    ///
    /// Qwen3 puts an RMSNorm on every attention head's query and key. This
    /// loader did not read the weights for it, so it ran a Llama forward pass
    /// on a Qwen3 model: full speed, right parameter count, nonsense output.
    /// Nothing inside one engine can notice that — "a Llama forward pass" is
    /// what both engines think they are doing — so the test is agreement with
    /// the hand-written CPU one, which had the norms all along.
    ///
    /// The norm weights are deliberately random. All-ones is the trap: it makes
    /// the *scale* of the normalisation the identity, so a backend that skipped
    /// the whole operation would still look close enough to pass.
    #[test]
    fn per_head_norms_agree_with_the_cpu_engine() {
        use kvad::model::Transformer;

        let spec = tiny_spec();
        let d = Device::Cpu;
        let norms: Vec<(String, Tensor)> = ["q_norm", "k_norm"]
            .iter()
            .map(|n| {
                let w = Tensor::randn(1f32, 0.3f32, spec.head_dim, &d).unwrap();
                (format!("model.layers.0.self_attn.{n}.weight"), w)
            })
            .collect();
        let path = write_tensors(&spec, false, &norms, "qk-norm");
        // The same model without them, for the count below.
        let plain = write_tensors(&spec, false, &[], "qk-norm-absent");

        let worst = engines_differ_by(&path, &spec, &[1, 2, 3]);
        assert!(worst < 1e-4, "logits disagree by {worst}");

        // The norms are parameters, and both engines have to say so. A weight
        // left out of the count is a weight nobody notices is missing — which
        // is the shape of the bug this whole test is about, one field over.
        let gpu = GpuLlama::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        )
        .unwrap();
        let cpu = cpu_model(&path, &spec);
        assert_eq!(gpu.param_count(), cpu.param_count());
        // And counted, not merely counted alike: two vectors per layer.
        assert_eq!(cpu.param_count() - cpu_model(&plain, &spec).param_count(), 2 * spec.head_dim);

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&plain).unwrap();
    }

    /// A tied model's output head is its embedding table, whatever else the
    /// file happens to contain.
    ///
    /// `tie_word_embeddings` says the two *are* one matrix, so a checkpoint
    /// carrying both is stating it twice — Qwen3-0.6B ships 297 MB of
    /// byte-identical duplicate. This backend used to believe the file and
    /// `llama.rs` the config, and they agreed only because those bytes matched.
    /// Here they deliberately do not: the head in this checkpoint is random and
    /// unrelated, so an engine that reads it lands somewhere else entirely.
    #[test]
    fn a_tied_model_ignores_a_duplicate_output_head() {
        let spec = tiny_spec();
        assert!(spec.tie_embeddings, "this test is about what tying means");

        let path = write_checkpoint(&spec, true, "tied-duplicate-head");
        let worst = engines_differ_by(&path, &spec, &[1, 2, 3]);
        assert!(worst < 1e-4, "logits disagree by {worst}");

        std::fs::remove_file(&path).unwrap();
    }

    /// An untied model with no head of its own is a broken checkpoint, and must
    /// say so rather than quietly borrowing the embedding table.
    ///
    /// The old code reached for `lm_head.weight`, shrugged when it was missing
    /// and used the table — which is a different model from the one the config
    /// describes, run at full speed without a word.
    #[test]
    fn an_untied_model_without_a_head_says_so() {
        let mut spec = tiny_spec();
        spec.tie_embeddings = false;
        let path = write_checkpoint(&spec, false, "untied-headless");

        let err = match GpuLlama::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an untied model with no `lm_head.weight` must not load"),
        };
        assert!(err.contains("lm_head.weight"), "unhelpful message: {err}");

        std::fs::remove_file(&path).unwrap();
    }

    /// A weight this backend does not read must stop the load rather than
    /// quietly change the answer.
    ///
    /// The general form of the Qwen3 bug: a `VarBuilder` answers the questions
    /// it is asked and says nothing about the rest, so *not implemented* and
    /// *implemented correctly* looked identical from outside the loader.
    #[test]
    fn a_tensor_this_backend_never_reads_refuses_to_load() {
        let spec = tiny_spec();
        let extra = vec![(
            "model.layers.0.self_attn.some_new_norm.weight".to_string(),
            Tensor::ones(spec.head_dim, DType::F32, &Device::Cpu).unwrap(),
        )];
        let path = write_tensors(&spec, false, &extra, "unread");

        // Not `expect_err`: a loaded model is not `Debug`, and the message is
        // the thing under test anyway.
        let err = match GpuLlama::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a tensor this backend never reads must not load silently"),
        };
        assert!(err.contains("some_new_norm"), "unhelpful message: {err}");

        std::fs::remove_file(&path).unwrap();
    }

    /// Derived tensors are the exception, and must stay one: Llama-2-era
    /// exports saved the RoPE frequencies as a buffer, and this loader computes
    /// them from `rope_theta`. Refusing those checkpoints would be the guard
    /// causing the harm it exists to prevent.
    #[test]
    fn a_saved_rope_table_is_not_an_unread_weight() {
        let spec = tiny_spec();
        let extra = vec![(
            "model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            Tensor::ones(spec.head_dim / 2, DType::F32, &Device::Cpu).unwrap(),
        )];
        let path = write_tensors(&spec, false, &extra, "inv-freq");

        let none = Vault::off();
        GpuLlama::load(std::slice::from_ref(&path), spec, DType::F32, None, Device::Cpu, &none)
            .unwrap();

        std::fs::remove_file(&path).unwrap();
    }

    /// Tying is the common case and the one that saves the memory, but an
    /// untied model must still load — and must *not* be reported as sharing
    /// storage it does not share.
    ///
    /// The loop is over the *config*, not over what the file contains, which is
    /// the thing this backend used to get backwards. The parameter count is
    /// checked against the hand-written engine in the same breath: a head that
    /// nobody counts is a head nobody notices is missing, and this one went
    /// uncounted in every model that has one.
    #[test]
    fn tied_and_untied_models_both_run() {
        use kvad::model::Transformer;

        for tie in [true, false] {
            let mut spec = tiny_spec();
            spec.tie_embeddings = tie;
            // An untied model carries its own head; a tied one must not need to.
            let path = write_checkpoint(&spec, !tie, "both-run");
            let mut m = GpuLlama::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                Some(GgmlDType::Q8_0),
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap();

            assert_eq!(m.tied, tie, "tying is the config's statement, not the file's");
            assert_eq!(m.param_count(), cpu_model(&path, &spec).param_count());

            let logits = m.forward(&[1, 2, 3]).unwrap();
            assert_eq!(logits.len(), spec.vocab_size);
            assert!(logits.iter().all(|v| v.is_finite()));

            std::fs::remove_file(&path).unwrap();
        }
    }

    /// The point of the exercise: quantising the table, and sharing it with
    /// the head, must actually shrink the model.
    #[test]
    fn a_quantised_table_costs_less_than_a_dense_one() {
        let spec = tiny_spec();
        let path = write_checkpoint(&spec, false, "costs-less");
        let load = |quant| {
            GpuLlama::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                quant,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap()
        };

        let dense = load(None);
        let quant = load(Some(GgmlDType::Q8_0));

        // 8 bits plus an f16 scale per 32 weights, against 32 bits — and the
        // tied head is no longer a second copy.
        let table = spec.vocab_size * spec.n_embd;
        assert_eq!(dense.embed.bytes(), table * 4);
        assert_eq!(quant.embed.bytes(), table * 34 / 32);
        assert!(quant.memory_bytes() * 3 < dense.memory_bytes());

        std::fs::remove_file(&path).unwrap();
    }
}
