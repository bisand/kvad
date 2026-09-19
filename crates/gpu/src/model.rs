//! The Llama forward pass again — this time on the GPU, in candle.
//!
//! # What changed, and what did not
//!
//! Read this next to [`llm::model::llama`]. The structure is line for line the
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

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, rotary_emb, VarBuilder};
use llm::model::{Session, Spec};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Block {
    attn_norm: Tensor,
    /// Stored already transposed, `[in, out]`, so the forward pass is a plain
    /// `x @ w` with no per-call transpose.
    q: Tensor,
    q_b: Option<Tensor>,
    k: Tensor,
    k_b: Option<Tensor>,
    v: Tensor,
    v_b: Option<Tensor>,
    o: Tensor,
    mlp_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub struct GpuLlama {
    spec: Spec,
    device: Device,
    dtype: DType,
    /// `[vocab, n_embd]`, for the embedding lookup.
    embed: Tensor,
    /// `[n_embd, vocab]`, the output head. A separate transposed copy even
    /// when the model ties its embeddings — the two uses want opposite
    /// layouts, and one extra copy of the table is cheaper than transposing on
    /// every token.
    head: Tensor,
    blocks: Vec<Block>,
    final_norm: Tensor,
    /// Precomputed rotations, `[n_ctx, head_dim / 2]`.
    cos: Tensor,
    sin: Tensor,
    /// Per layer, `[1, n_kv_head, seq, head_dim]` for keys and values.
    kv: Vec<Option<(Tensor, Tensor)>>,
    pos: usize,
}

/// `y = x @ w (+ b)`, with `w` already `[in, out]`.
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let y = x.matmul(w)?;
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
    pub fn load(paths: &[std::path::PathBuf], spec: Spec, dtype: DType, device: Device) -> Res<Self> {
        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, &device)? };

        let (e, hd) = (spec.n_embd, spec.head_dim);
        let (qd, kvd) = (spec.n_head * hd, spec.kv_dim());
        let model = vb.pp("model");

        // HuggingFace stores `nn.Linear` weights as [out, in]; transposing once
        // at load lets every forward pass be a plain matmul.
        let load_t = |vb: &VarBuilder, name: &str, out: usize, inp: usize| -> Res<Tensor> {
            Ok(vb.get((out, inp), name)?.t()?.contiguous()?)
        };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let l = model.pp(format!("layers.{i}"));
            let attn = l.pp("self_attn");
            let mlp = l.pp("mlp");
            blocks.push(Block {
                attn_norm: l.get(e, "input_layernorm.weight")?,
                q: load_t(&attn, "q_proj.weight", qd, e)?,
                q_b: attn.get(qd, "q_proj.bias").ok(),
                k: load_t(&attn, "k_proj.weight", kvd, e)?,
                k_b: attn.get(kvd, "k_proj.bias").ok(),
                v: load_t(&attn, "v_proj.weight", kvd, e)?,
                v_b: attn.get(kvd, "v_proj.bias").ok(),
                o: load_t(&attn, "o_proj.weight", e, qd)?,
                mlp_norm: l.get(e, "post_attention_layernorm.weight")?,
                gate: load_t(&mlp, "gate_proj.weight", spec.intermediate, e)?,
                up: load_t(&mlp, "up_proj.weight", spec.intermediate, e)?,
                down: load_t(&mlp, "down_proj.weight", e, spec.intermediate)?,
            });
        }

        let embed = model.get((spec.vocab_size, e), "embed_tokens.weight")?;
        let head = match vb.get((spec.vocab_size, e), "lm_head.weight") {
            Ok(w) => w.t()?.contiguous()?,
            Err(_) => embed.t()?.contiguous()?,
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
        let cos = angles.cos()?.to_dtype(dtype)?;
        let sin = angles.sin()?.to_dtype(dtype)?;

        Ok(GpuLlama {
            kv: (0..spec.n_layer).map(|_| None).collect(),
            blocks,
            embed,
            head,
            final_norm: model.get(e, "norm.weight")?,
            cos,
            sin,
            device,
            dtype,
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
        format!("{kind} {}", dtype_name(self.dtype))
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
        let mut x = self.embed.index_select(&ids, 0)?.to_dtype(self.dtype)?;

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
        let logits = last.matmul(&self.head)?.to_dtype(DType::F32)?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }

    fn params(&self) -> usize {
        let n = |t: &Tensor| t.elem_count();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                n(&b.q) + n(&b.k) + n(&b.v) + n(&b.o) + n(&b.gate) + n(&b.up) + n(&b.down)
                    + n(&b.attn_norm)
                    + n(&b.mlp_norm)
            })
            .sum();
        blocks + n(&self.embed) + n(&self.final_norm)
    }

    fn memory_bytes(&self) -> usize {
        let per = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                per(&b.q) + per(&b.k) + per(&b.v) + per(&b.o) + per(&b.gate) + per(&b.up)
                    + per(&b.down)
            })
            .sum();
        blocks + per(&self.embed) + per(&self.head)
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
