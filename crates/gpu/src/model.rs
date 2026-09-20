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

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{ops, rotary_emb, VarBuilder};
use kvad::model::{Session, Spec};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One projection matrix, dense or quantised.
///
/// The two want opposite layouts. A dense matmul is cheapest if the weight is
/// pre-transposed to `[in, out]`, so the forward pass is a plain `x @ w`.
/// `QMatMul` instead keeps HuggingFace's `[out, in]` and transposes inside its
/// kernel. Hiding that behind one `forward` keeps the block code identical
/// either way.
enum Proj {
    Dense(Tensor),
    Quant(QMatMul),
}

impl Proj {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Proj::Dense(w) => x.matmul(w),
            Proj::Quant(q) => q.forward(x),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Proj::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Proj::Quant(q) => match q {
                QMatMul::QTensor(t) => t.storage_size_in_bytes(),
                _ => 0,
            },
        }
    }

    fn params(&self) -> usize {
        match self {
            Proj::Dense(t) => t.elem_count(),
            Proj::Quant(q) => match q {
                QMatMul::QTensor(t) => t.shape().elem_count(),
                _ => 0,
            },
        }
    }
}

/// The token embedding table, `[vocab, n_embd]`, read one row per token.
///
/// This was the last dense tensor in a quantised model, and on a small model
/// it is not a small one: Qwen2.5-0.5B's is 136M of its 494M parameters, 272 MB
/// in bf16 against 797 MB for everything else put together.
///
/// Quantising it needs an operation the rest of the engine never wanted —
/// *gather rows and dequantise only those*. A `QTensor` is blocks, not a
/// matrix, so row `t` is a range of blocks that has to be decoded on its own;
/// candle has a kernel for exactly this (`QTensor::embedding`, GGML's
/// `get_rows`), which is what makes this three lines rather than a Metal
/// shader.
///
/// The bigger win is not the compression. When a model ties its embeddings —
/// and small ones nearly always do — the lookup table and the output head are
/// the *same matrix*, but they were stored twice because a dense lookup and a
/// quantised matmul want different things. Quantise the lookup and they want
/// the same thing, so one `Arc<QTensor>` serves both.
enum Embed {
    Dense(Tensor),
    Quant(Arc<QTensor>),
}

impl Embed {
    /// Row `ids[i]` of the table, per element of `ids`.
    fn rows(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Embed::Dense(t) => t.index_select(ids, 0),
            Embed::Quant(q) => q.embedding(ids),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Embed::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Embed::Quant(q) => q.storage_size_in_bytes(),
        }
    }

    fn params(&self) -> usize {
        match self {
            Embed::Dense(t) => t.elem_count(),
            Embed::Quant(q) => q.shape().elem_count(),
        }
    }
}

struct Block {
    attn_norm: Tensor,
    q: Proj,
    q_b: Option<Tensor>,
    k: Proj,
    k_b: Option<Tensor>,
    v: Proj,
    v_b: Option<Tensor>,
    o: Proj,
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

/// `KVAD_GPU_DENSE_EMBED=1` restores the dense bf16 lookup table, for
/// measuring what quantising it is worth.
fn dense_embedding() -> bool {
    matches!(std::env::var("KVAD_GPU_DENSE_EMBED").as_deref(), Ok("1") | Ok("true"))
}

/// `y = proj(x) (+ b)`.
fn linear(x: &Tensor, w: &Proj, b: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let y = w.forward(x)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

/// Grouped-query attention: give every query head a copy of its KV head.
///
/// The hand-written version indexed into the shared head and never materialised
/// the copies. Here it is cheaper to expand, because the result feeds a batched
/// matmul that wants one KV head per query head.
fn repeat_kv(x: &Tensor, group: usize) -> candle_core::Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (b, kv_heads, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, group, seq, hd))?
        .reshape((b, kv_heads * group, seq, hd))
}

impl GpuLlama {
    pub fn load(
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
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
        // Every quantised format works in blocks along the contraction axis,
        // and k-quants use a 256-wide super-block. A model whose dimensions
        // are not a multiple of that simply cannot use them — Qwen2.5-0.5B is
        // 896 wide, which is fine for q8 and q4 (32) and hopeless for q4k.
        // Candle reports this per tensor, deep in the load; better to say it
        // once, up front, and name the alternative.
        if let Some(gd) = quant {
            let block = gd.block_size();
            let dims = [
                ("hidden size", spec.n_embd),
                ("MLP width", spec.intermediate),
                ("attention output", spec.n_head * spec.head_dim),
                ("KV width", spec.kv_dim()),
            ];
            if let Some((what, n)) = dims.iter().find(|(_, n)| n % block != 0) {
                return Err(format!(
                    "{} needs dimensions divisible by {block}, but this model's {what} is {n}.\n\
                     Try --quant q8 or --quant q4, whose blocks are 32.",
                    ggml_name(gd)
                )
                .into());
            }
        }

        let load_dev = if quant.is_some() { Device::Cpu } else { device.clone() };

        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, load_dtype, &load_dev)? };

        // Dense tensors (norms, embeddings) still have to make the trip.
        let to_dev = |t: Tensor| -> Res<Tensor> { Ok(t.to_device(&device)?) };

        let (e, hd) = (spec.n_embd, spec.head_dim);
        let (qd, kvd) = (spec.n_head * hd, spec.kv_dim());
        let model = vb.pp("model");

        // HuggingFace stores `nn.Linear` weights as [out, in]. Dense matmuls
        // want the transpose; quantised ones want it exactly as stored.
        let dev = device.clone();
        let load_t = move |vb: &VarBuilder, name: &str, out: usize, inp: usize| -> Res<Proj> {
            let w = vb.get((out, inp), name)?;
            Ok(match quant {
                None => Proj::Dense(w.t()?.contiguous()?),
                Some(gd) => Proj::Quant(QMatMul::from_qtensor(QTensor::quantize_onto(
                    &w, gd, &dev,
                )?)?),
            })
        };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let l = model.pp(format!("layers.{i}"));
            let attn = l.pp("self_attn");
            let mlp = l.pp("mlp");
            blocks.push(Block {
                attn_norm: to_dev(l.get(e, "input_layernorm.weight")?)?,
                q: load_t(&attn, "q_proj.weight", qd, e)?,
                q_b: attn.get(qd, "q_proj.bias").ok().map(&to_dev).transpose()?,
                k: load_t(&attn, "k_proj.weight", kvd, e)?,
                k_b: attn.get(kvd, "k_proj.bias").ok().map(&to_dev).transpose()?,
                v: load_t(&attn, "v_proj.weight", kvd, e)?,
                v_b: attn.get(kvd, "v_proj.bias").ok().map(&to_dev).transpose()?,
                o: load_t(&attn, "o_proj.weight", e, qd)?,
                mlp_norm: to_dev(l.get(e, "post_attention_layernorm.weight")?)?,
                gate: load_t(&mlp, "gate_proj.weight", spec.intermediate, e)?,
                up: load_t(&mlp, "up_proj.weight", spec.intermediate, e)?,
                down: load_t(&mlp, "down_proj.weight", e, spec.intermediate)?,
            });
        }

        // The table, and the output head — which is the same matrix again
        // unless the model says otherwise.
        let table = model.get((spec.vocab_size, e), "embed_tokens.weight")?;
        let own_head = vb.get((spec.vocab_size, e), "lm_head.weight").ok();

        let mut tied = false;
        let (embed, head) = match quant {
            None => {
                // Dense keeps two copies: `index_select` wants
                // `[vocab, n_embd]` and `matmul` wants the transpose, and
                // neither is cheap to fake from the other.
                let w = own_head.unwrap_or_else(|| table.clone());
                (Embed::Dense(to_dev(table)?), Proj::Dense(to_dev(w.t()?.contiguous()?)?))
            }
            // The arrangement this replaced, kept behind a flag so the
            // difference it makes is one environment variable wide: a dense
            // half-precision table, and the head quantised separately from it.
            Some(gd) if dense_embedding() => {
                let w = own_head.unwrap_or_else(|| table.clone());
                let head = QMatMul::from_qtensor(QTensor::quantize_onto(&w, gd, &device)?)?;
                (Embed::Dense(to_dev(table.to_dtype(DType::BF16)?)?), Proj::Quant(head))
            }
            Some(gd) => {
                let q = Arc::new(QTensor::quantize_onto(&table, gd, &device)?);
                let head = match own_head {
                    Some(w) => QMatMul::from_qtensor(QTensor::quantize_onto(&w, gd, &device)?)?,
                    // Tied, and now in one layout: one allocation, two uses.
                    None => {
                        tied = true;
                        QMatMul::from_arc(Arc::clone(&q))?
                    }
                };
                (Embed::Quant(q), Proj::Quant(head))
            }
        };

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
            final_norm: to_dev(model.get(e, "norm.weight")?)?,
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
        let kind = if self.device.is_metal() {
            "metal"
        } else if self.device.is_cuda() {
            "cuda"
        } else {
            "cpu"
        };
        match self.quant {
            None => format!("{kind} {}", dtype_name(self.dtype)),
            Some(q) => format!("{kind} {}", ggml_name(q)),
        }
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

    /// Additive causal mask, `[1, 1, m, total]`: zero where a query may attend,
    /// -inf where it may not.
    ///
    /// The CPU engine never needed this — its cache only ever contained earlier
    /// positions, so "causal" was free. Here the whole batch is one matmul, so
    /// the future has to be masked out explicitly.
    fn causal_mask(&self, m: usize, pos0: usize) -> candle_core::Result<Tensor> {
        let total = pos0 + m;
        let mut data = vec![0f32; m * total];
        for i in 0..m {
            for j in (pos0 + i + 1)..total {
                data[i * total + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(data, (1, 1, m, total), &self.device)?.to_dtype(self.dtype)
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
        let mask = if m > 1 { Some(self.causal_mask(m, pos0)?) } else { None };

        for (i, blk) in self.blocks.iter().enumerate() {
            let h = ops::rms_norm(&x, &blk.attn_norm, spec.eps)?;

            let q = linear(&h, &blk.q, blk.q_b.as_ref())?;
            let k = linear(&h, &blk.k, blk.k_b.as_ref())?;
            let v = linear(&h, &blk.v, blk.v_b.as_ref())?;

            // [m, heads * hd] -> [1, heads, m, hd], which is what the rotary
            // kernel and the batched attention matmuls want.
            let q = q.reshape((1, m, n_head, hd))?.transpose(1, 2)?.contiguous()?;
            let k = k.reshape((1, m, n_kv, hd))?.transpose(1, 2)?.contiguous()?;
            let v = v.reshape((1, m, n_kv, hd))?.transpose(1, 2)?.contiguous()?;

            let q = rotary_emb::rope(&q, &cos, &sin)?;
            let k = rotary_emb::rope(&k, &cos, &sin)?;

            // Append to the cache along the sequence axis.
            let (k, v) = match self.kv[i].take() {
                None => (k, v),
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            };
            self.kv[i] = Some((k.clone(), v.clone()));

            let kx = repeat_kv(&k, group)?;
            let vx = repeat_kv(&v, group)?;

            let mut att = (q.matmul(&kx.transpose(2, 3)?.contiguous()?)? * scale)?;
            if let Some(msk) = &mask {
                att = att.broadcast_add(msk)?;
            }
            let att = ops::softmax_last_dim(&att)?;

            let out = att.matmul(&vx)?;
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
            })
            .sum();
        blocks + self.embed.params() + n(&self.final_norm)
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

pub fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => match other {
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
            _ => "?",
        },
    }
}

pub fn ggml_name(d: GgmlDType) -> &'static str {
    match d {
        GgmlDType::Q8_0 => "q8",
        GgmlDType::Q4_0 => "q4",
        GgmlDType::Q4K => "q4k",
        GgmlDType::Q6K => "q6k",
        _ => "quant",
    }
}

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
mod tests {
    use super::*;
    use kvad::model::Arch;
    use std::collections::HashMap;

    /// A model small enough to build from random numbers, with every
    /// dimension a multiple of 32 so the quantisers will take it.
    fn tiny_spec() -> Spec {
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

        // Unique per test as well as per process: the tests run in parallel
        // and would otherwise delete each other's checkpoints.
        let path = std::env::temp_dir().join(format!(
            "gpu-tiny-{}-{tag}-{own_head}.safetensors",
            std::process::id()
        ));
        candle_core::safetensors::save(&t, &path).unwrap();
        path
    }

    /// Tying is the common case and the one that saves the memory, but an
    /// untied model must still load — and must *not* be reported as sharing
    /// storage it does not share.
    #[test]
    fn tied_and_untied_models_both_run() {
        let spec = tiny_spec();
        for own_head in [false, true] {
            let path = write_checkpoint(&spec, own_head, "both-run");
            let mut m = GpuLlama::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                Some(GgmlDType::Q8_0),
                Device::Cpu,
            )
            .unwrap();

            assert_eq!(m.tied, !own_head);
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
