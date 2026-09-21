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
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
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

/// `KVAD_GPU_DENSE_EMBED=1` restores the dense bf16 lookup table, for
/// measuring what quantising it is worth.
fn dense_embedding() -> bool {
    matches!(std::env::var("KVAD_GPU_DENSE_EMBED").as_deref(), Ok("1") | Ok("true"))
}

// ---------------------------------------------------------------------------
// Reading the checkpoint, and noticing what was not read
// ---------------------------------------------------------------------------

/// A [`VarBuilder`] that remembers every name it was asked for.
///
/// The bug this exists to prevent was not a wrong answer but a question never
/// asked: Qwen3's `self_attn.q_norm.weight` sat in the checkpoint unread, and a
/// `VarBuilder` has no opinion about tensors nobody wants. The model loaded,
/// reported the right parameter count, ran at full speed, and talked nonsense.
///
/// So every read goes through here and the names pile up in one set, which
/// [`unread`] subtracts from the checkpoint's own list at the end of the load.
/// The set is shared by `Rc` rather than copied, so however deep the prefixes
/// nest there is one record — and the only way to add a weight to this backend
/// is to read it through a `Reader`, which registers it without being asked to.
struct Reader<'a> {
    vb: VarBuilder<'a>,
    seen: Rc<RefCell<HashSet<String>>>,
}

impl<'a> Reader<'a> {
    fn new(vb: VarBuilder<'a>) -> Self {
        Reader { vb, seen: Rc::new(RefCell::new(HashSet::new())) }
    }

    /// Descend into a prefix, keeping the shared record.
    fn pp(&self, s: impl std::fmt::Display) -> Self {
        Reader { vb: self.vb.pp(s.to_string()), seen: Rc::clone(&self.seen) }
    }

    /// The name this read is really about, prefixes and all — the spelling the
    /// checkpoint uses, and so the one worth recording.
    fn full(&self, name: &str) -> String {
        let prefix = self.vb.prefix();
        if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}.{name}")
        }
    }

    /// Note that this backend knows about `name`.
    ///
    /// Every read calls this, and it is also called on its own for a tensor
    /// this backend knows about and deliberately does not use: the duplicate
    /// `lm_head.weight` in a tied checkpoint. Reading that one to satisfy the
    /// guard would move 300 MB for nothing, and leaving it out would make the
    /// guard refuse a model that is perfectly correct — so what the set records
    /// is the *decision*, which is what it was always about.
    fn record(&self, name: &str) {
        self.seen.borrow_mut().insert(self.full(name));
    }

    fn get(
        &self,
        shape: impl Into<candle_core::Shape>,
        name: &str,
    ) -> candle_core::Result<Tensor> {
        self.record(name);
        self.vb.get(shape, name)
    }

    /// A tensor this model may not have: a bias Qwen2 carries and Llama does
    /// not, an untied output head.
    ///
    /// Recorded whether or not it is there, because the set means *names this
    /// backend knows about*, not *names it found*. A bias that is absent is
    /// absent from the checkpoint too, so recording it costs nothing — and
    /// recording only the hits would make every optional weight in every model
    /// that lacks it look unread.
    fn try_get(&self, shape: impl Into<candle_core::Shape>, name: &str) -> Option<Tensor> {
        self.get(shape, name).ok()
    }

    fn seen(&self) -> HashSet<String> {
        self.seen.borrow().clone()
    }
}

/// Tensors the checkpoint holds that nothing in [`GpuLlama::load`] asked for.
///
/// Reopening the files costs one pass over the safetensors headers and reads no
/// tensor data — a rounding error against the load itself. The names come from
/// the file rather than from a list kept in this crate, which is the whole
/// point: a list would have to be remembered, and forgetting is what went
/// wrong.
fn unread(paths: &[std::path::PathBuf], seen: &HashSet<String>) -> Res<Vec<String>> {
    let ckpt = kvad::weights::Checkpoint::open(paths)?;
    let mut left: Vec<String> = ckpt
        .names()
        .filter(|n| !seen.contains(*n) && !kvad::weights::derived(n))
        .map(str::to_string)
        .collect();
    left.sort();
    Ok(left)
}

/// `y = proj(x) (+ b)`.
fn linear(x: &Tensor, w: &Proj, b: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let y = w.forward(x)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
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
        let vb = Reader::new(vb);

        // Dense tensors (norms, embeddings) still have to make the trip.
        let to_dev = |t: Tensor| -> Res<Tensor> { Ok(t.to_device(&device)?) };

        let (e, hd) = (spec.n_embd, spec.head_dim);
        let (qd, kvd) = (spec.n_head * hd, spec.kv_dim());
        let model = vb.pp("model");

        // HuggingFace stores `nn.Linear` weights as [out, in]. Dense matmuls
        // want the transpose; quantised ones want it exactly as stored.
        let dev = device.clone();
        let load_t = move |vb: &Reader<'_>, name: &str, out: usize, inp: usize| -> Res<Proj> {
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

        // The table, and the output head — which is the same matrix again
        // unless the model says otherwise.
        let table = model.get((spec.vocab_size, e), "embed_tokens.weight")?;

        // Whether the head is its own matrix is the *config's* statement, not
        // the checkpoint's. `tie_word_embeddings` means `lm_head.weight` **is**
        // `embed_tokens.weight`, so a file that carries both is saying one
        // matrix twice — Qwen3-0.6B ships 297 MB of byte-identical duplicate,
        // and HuggingFace itself overwrites the stored copy when it ties.
        //
        // Reading whichever tensor happened to be present is how this backend
        // came to disagree with `llama.rs`, which reads the flag. They agreed on
        // Qwen3 only because the duplicate holds the same numbers; a stale
        // `lm_head` would have split them, and this side would have been wrong.
        let own_head = match spec.tie_embeddings {
            true => {
                // Skipped on purpose, and recorded so the guard does not read
                // that as an omission.
                vb.record("lm_head.weight");
                None
            }
            false => match vb.get((spec.vocab_size, e), "lm_head.weight") {
                Ok(w) => Some(w),
                Err(_) => {
                    return Err(concat!(
                        "this model does not tie its embeddings, so it needs its own ",
                        "`lm_head.weight`, and the checkpoint has none.\n",
                        "If it is meant to be tied, its config is missing ",
                        "`tie_word_embeddings: true`."
                    )
                    .into())
                }
            },
        };

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

        let final_norm = to_dev(model.get(e, "norm.weight")?)?;

        // Everything has been read, so anything left in the file is a part of
        // this model that is not running.
        let left = unread(paths, &vb.seen())?;
        if !left.is_empty() {
            return Err(format!(
                "this checkpoint holds {} tensor(s) that the GPU backend never reads:\n  {}\n\
                 A weight nobody reads is a piece of the model that is not running — a wrong\n\
                 answer at full speed rather than an error, which is how Qwen3's per-head Q/K\n\
                 norms were missed. Run it on the CPU engine instead:  kvad run",
                left.len(),
                kvad::weights::collapsed(&left).join("\n  ")
            )
            .into());
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
        write_tensors(spec, own_head, &[], tag)
    }

    /// The same, plus tensors the base Llama layout does not have: Qwen3's
    /// per-head norms, or something this backend does not implement at all.
    fn write_tensors(
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
            GpuLlama::load(std::slice::from_ref(path), spec.clone(), DType::F32, None, Device::Cpu)
                .unwrap();
        let theirs = gpu.forward(tokens).unwrap();

        let mut cache = kvad::model::KvCache::new(spec);
        let ours = cpu_model(path, spec).forward_batch(tokens, &mut cache);

        assert_eq!(theirs.len(), ours.len());
        theirs.iter().zip(&ours).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
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

        let worst = engines_differ_by(&path, &spec, &[1, 2, 3]);
        assert!(worst < 1e-4, "logits disagree by {worst}");

        std::fs::remove_file(&path).unwrap();
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

        GpuLlama::load(std::slice::from_ref(&path), spec, DType::F32, None, Device::Cpu).unwrap();

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
