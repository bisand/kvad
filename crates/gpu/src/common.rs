//! The vocabulary every architecture on this backend shares.
//!
//! Not a grab bag: each of these is a decision that must come out the same way
//! whichever block is being built. A projection has two layouts and one
//! `forward`; an embedding table is dense or quantised and is read by row
//! either way; and every checkpoint is read through a [`Reader`], which is what
//! makes [`unread`] able to say what a loader never asked for.
//!
//! That last one is the reason this file exists rather than each architecture
//! keeping its own copy. The bug it guards against — Qwen3's per-head norms
//! going unread while the model ran at full speed — is a bug about a *loader*,
//! and a second loader is a second chance to make it.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
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
pub(crate) enum Proj {
    Dense(Tensor),
    Quant(QMatMul),
}

impl Proj {    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Proj::Dense(w) => x.matmul(w),
            Proj::Quant(q) => q.forward(x),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        match self {
            Proj::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Proj::Quant(q) => match q {
                QMatMul::QTensor(t) => t.storage_size_in_bytes(),
                _ => 0,
            },
        }
    }

    pub(crate) fn params(&self) -> usize {
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
pub(crate) enum Embed {
    Dense(Tensor),
    Quant(Arc<QTensor>),
}

impl Embed {    /// Row `ids[i]` of the table, per element of `ids`.
    pub(crate) fn rows(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Embed::Dense(t) => t.index_select(ids, 0),
            Embed::Quant(q) => q.embedding(ids),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        match self {
            Embed::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Embed::Quant(q) => q.storage_size_in_bytes(),
        }
    }

    pub(crate) fn params(&self) -> usize {
        match self {
            Embed::Dense(t) => t.elem_count(),
            Embed::Quant(q) => q.shape().elem_count(),
        }
    }
}

/// `KVAD_GPU_DENSE_EMBED=1` restores the dense bf16 lookup table, for
/// measuring what quantising it is worth.
pub(crate) fn dense_embedding() -> bool {
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
pub(crate) struct Reader<'a> {
    vb: VarBuilder<'a>,
    seen: Rc<RefCell<HashSet<String>>>,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(vb: VarBuilder<'a>) -> Self {
        Reader { vb, seen: Rc::new(RefCell::new(HashSet::new())) }
    }

    /// Descend into a prefix, keeping the shared record.
    pub(crate) fn pp(&self, s: impl std::fmt::Display) -> Self {
        Reader { vb: self.vb.pp(s.to_string()), seen: Rc::clone(&self.seen) }
    }

    /// The name this read is really about, prefixes and all — the spelling the
    /// checkpoint uses, and so the one worth recording.
    pub(crate) fn full(&self, name: &str) -> String {
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
    pub(crate) fn record(&self, name: &str) {
        self.seen.borrow_mut().insert(self.full(name));
    }

    pub(crate) fn get(
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
    pub(crate) fn try_get(&self, shape: impl Into<candle_core::Shape>, name: &str) -> Option<Tensor> {
        self.get(shape, name).ok()
    }

    pub(crate) fn seen(&self) -> HashSet<String> {
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
pub(crate) fn unread(paths: &[std::path::PathBuf], seen: &HashSet<String>) -> Res<Vec<String>> {
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
pub(crate) fn linear(x: &Tensor, w: &Proj, b: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let y = w.forward(x)?;
    match b {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

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

/// Additive causal mask, `[1, 1, m, total]`: zero where a query may attend,
/// -inf where it may not.
///
/// The CPU engine never needed this — its cache only ever contained earlier
/// positions, so "causal" was free. Here the whole batch is one matmul, so the
/// future has to be masked out explicitly.
pub(crate) fn causal_mask(
    m: usize,
    pos0: usize,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    let total = pos0 + m;
    let mut data = vec![0f32; m * total];
    for i in 0..m {
        for j in (pos0 + i + 1)..total {
            data[i * total + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, m, total), device)?.to_dtype(dtype)
}

/// Refuse a quantisation whose block size does not divide the model.
///
/// Every quantised format works in blocks along the contraction axis, and
/// k-quants use a 256-wide super-block. A model whose dimensions are not a
/// multiple of that simply cannot use them — Qwen2.5-0.5B is 896 wide, which
/// is fine for q8 and q4 (32) and hopeless for q4k. Candle reports it per
/// tensor, deep in the load; better to say it once, up front, and name the
/// alternative.
pub(crate) fn check_block(quant: Option<GgmlDType>, dims: &[(&str, usize)]) -> Res<()> {
    let Some(gd) = quant else { return Ok(()) };
    let block = gd.block_size();
    if let Some((what, n)) = dims.iter().find(|(_, n)| n % block != 0) {
        return Err(format!(
            "{} needs dimensions divisible by {block}, but this model's {what} is {n}.\n\
             Try --quant q8 or --quant q4, whose blocks are 32.",
            ggml_name(gd)
        )
        .into());
    }
    Ok(())
}

/// What a loaded model calls the place it runs: `metal bf16`, `cpu q8`.
pub(crate) fn label(device: &Device, dtype: DType, quant: Option<GgmlDType>) -> String {
    let kind = if device.is_metal() {
        "metal"
    } else if device.is_cuda() {
        "cuda"
    } else {
        "cpu"
    };
    match quant {
        None => format!("{kind} {}", dtype_name(dtype)),
        Some(q) => format!("{kind} {}", ggml_name(q)),
    }
}

/// The message a load gives when the checkpoint holds weights nobody read.
pub(crate) fn unread_error(arch: &str, left: &[String]) -> String {
    format!(
        "this checkpoint holds {} tensor(s) that the GPU backend's `{arch}` loader never \
         reads:\n  {}\n\
         A weight nobody reads is a piece of the model that is not running — a wrong\n\
         answer at full speed rather than an error, which is how Qwen3's per-head Q/K\n\
         norms were missed. Run it on the CPU engine instead:  kvad run",
        left.len(),
        kvad::weights::collapsed(left).join("\n  ")
    )
}
