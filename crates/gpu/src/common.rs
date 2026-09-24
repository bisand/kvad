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

use crate::qcache::Vault;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use kvad::model::Spec;
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
///
/// `Blocks` is a third way to hold a quantised matrix, for the M5's matrix
/// units: the raw Q8_0 bytes, which `mpp`'s kernel reads itself. It is what
/// [`Loader::accelerated`] gives, and only the image pipelines ask for it;
/// `mpp` says why a language model cannot.
pub(crate) enum Proj {
    Dense(Tensor),
    Quant(QMatMul),
    #[cfg(target_os = "macos")]
    Blocks(crate::mpp::Q8),
}

impl Proj {    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            // On the M5's matrix units when it is f16 or bf16 and more than
            // a few rows; `mpp::dense` says which, and declines the rest.
            #[cfg(target_os = "macos")]
            Proj::Dense(w) => match crate::mpp::dense(x, w)? {
                Some(y) => Ok(y),
                None => x.matmul(w),
            },
            #[cfg(not(target_os = "macos"))]
            Proj::Dense(w) => x.matmul(w),
            // candle 0.11's Metal kernel for a quantised matrix-matrix product
            // reads its input from the start of the buffer whatever the
            // tensor's own start offset is: it builds a fresh contiguous
            // layout for the input and passes *that* layout's offset, which is
            // always zero. The one-row kernel honours the offset, so a decode
            // step never shows it. A `narrow` along the rows — the image half
            // of Qwen-Image's joint attention, which starts where the text
            // ends — is contiguous, so `contiguous()` does not copy it, and the
            // product silently used the text's rows for the first patches. It
            // drew noise at full speed. `copy()` is no cure either: it copies
            // the whole buffer and keeps the offset. `force_contiguous` lays
            // the rows out afresh, at zero.
            Proj::Quant(q) if x.layout().start_offset() != 0 => q.forward(&x.force_contiguous()?),
            Proj::Quant(q) => q.forward(x),
            // Casts to f16 on the way in, which lays the rows out afresh:
            // the offset bug above cannot reach it.
            #[cfg(target_os = "macos")]
            Proj::Blocks(q) => q.forward(x),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        match self {
            Proj::Dense(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Proj::Quant(q) => match q {
                QMatMul::QTensor(t) => t.storage_size_in_bytes(),
                _ => 0,
            },
            #[cfg(target_os = "macos")]
            Proj::Blocks(q) => q.bytes(),
        }
    }

    pub(crate) fn params(&self) -> usize {
        match self {
            Proj::Dense(t) => t.elem_count(),
            Proj::Quant(q) => match q {
                QMatMul::QTensor(t) => t.shape().elem_count(),
                _ => 0,
            },
            #[cfg(target_os = "macos")]
            Proj::Blocks(q) => q.params(),
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
// Turning a stored matrix into one of the two layouts above
// ---------------------------------------------------------------------------

/// Which way round the checkpoint wrote a matrix.
///
/// HuggingFace's `nn.Linear` stores `[out, in]`; GPT-2's `Conv1D` stores
/// `[in, out]`. Both end up in the same two destinations — a dense matmul
/// wants `[in, out]`, `QMatMul` wants `[out, in]` — so exactly one of the two
/// transposes, and which one is the *only* thing these two spellings disagree
/// about. Saying that once here is worth more than saying it twice: the GPT-2
/// loader had a paragraph explaining that its transpose was the opposite of
/// every other loader's, which is a comment that exists because the code was
/// two copies of one rule.
#[derive(Clone, Copy)]
pub(crate) enum Stored {
    OutIn,
    InOut,
}

/// Where the weights come from, and what is done to them on the way in.
///
/// One of these per load, carrying the three answers every matrix needs — the
/// quantisation, the device, and the cache — so that an architecture's loader
/// asks for `q_proj.weight` and says nothing about any of them.
pub(crate) struct Loader<'v> {
    pub(crate) quant: Option<GgmlDType>,
    pub(crate) device: Device,
    vault: &'v Vault,
    /// Q8_0 projections as [`Proj::Blocks`], for the M5's matrix units.
    blocks: bool,
}

impl<'v> Loader<'v> {
    pub(crate) fn new(quant: Option<GgmlDType>, device: Device, vault: &'v Vault) -> Self {
        Loader { quant, device, vault, blocks: false }
    }

    /// The same load, with its Q8_0 projections on the M5's matrix units
    /// where this machine has them, and as before where it does not.
    ///
    /// Only for a model with no decode step: a [`Proj::Blocks`] has no
    /// one-row kernel behind it. `mpp` says why it cannot also keep one.
    pub(crate) fn accelerated(mut self) -> Self {
        #[cfg(target_os = "macos")]
        {
            self.blocks = self.quant == Some(GgmlDType::Q8_0) && crate::mpp::available(&self.device);
        }
        self
    }

    /// One projection matrix, in whichever layout this load will use it.
    pub(crate) fn proj(
        &self,
        vb: &Reader<'_>,
        name: &str,
        out: usize,
        inp: usize,
        stored: Stored,
    ) -> Res<Proj> {
        let Some(_) = self.quant else {
            // Transposed on the host and moved once, never transposed on the
            // device. `.t()?.contiguous()?` allocates a second buffer the
            // size of the first, and doing that on the GPU means the source
            // and the copy are both device-resident until the source drops —
            // a peak of two full models. Qwen2.5-Coder-7B in bf16 is 15.2 GB,
            // so that peak was 30.4 GB, and the command buffer died of
            // `kIOGPUCommandBufferCallbackErrorOutOfMemory` on a machine with
            // 41 GB free. Nobody was reading command buffer status, so the
            // failed copy left its destination zeroed: the output head came
            // out all zeros, every logit was 0.0, argmax returned token 0 and
            // the server answered `!!!!!!!!` with HTTP 200.
            //
            // The quantised path never had this problem, and the comment in
            // `GpuLlama::load` says why in as many words — it reads to the
            // host and hands the device one finished tensor at a time. This
            // is the dense path doing the same.
            let w = match stored {
                Stored::OutIn => vb.get((out, inp), name)?.t()?.contiguous()?,
                Stored::InOut => vb.get((inp, out), name)?,
            };
            return Ok(Proj::Dense(w.to_device(&self.device)?));
        };
        #[cfg(target_os = "macos")]
        if self.blocks {
            let full = vb.full(name);
            if let Some(b) = self.vault.blocks(&full, (out, inp)) {
                vb.record(name);
                return Ok(Proj::Blocks(crate::mpp::Q8::new(b, out, inp, &self.device)?));
            }
            let blocks = self.quantize(vb, name, out, inp, stored)?.data()?.into_owned();
            return Ok(Proj::Blocks(crate::mpp::Q8::new(&blocks, out, inp, &self.device)?));
        }
        Ok(Proj::Quant(QMatMul::from_qtensor(self.quantized(vb, name, out, inp, stored)?)?))
    }

    /// One matrix in blocks: read back from the cache, or quantised and filed.
    ///
    /// A `QTensor` keeps `[out, in]` whichever way the checkpoint spelled it,
    /// so one name means one blob and a `Conv1D` model and a `Linear` one cache
    /// alike.
    pub(crate) fn quantized(
        &self,
        vb: &Reader<'_>,
        name: &str,
        out: usize,
        inp: usize,
        stored: Stored,
    ) -> Res<QTensor> {
        let gd = self.quant.ok_or("asked for blocks on a load that is not quantised")?;
        let full = vb.full(name);
        if let Some(q) = self.vault.get(&full, (out, inp), &self.device) {
            // The cache answered and the checkpoint still gets the credit:
            // what [`unread`] subtracts is the names this loader asked about,
            // not the ones it happened to read bytes for.
            vb.record(name);
            return Ok(q);
        }
        let cpu = self.quantize(vb, name, out, inp, stored)?;
        if self.device.is_cpu() {
            return Ok(cpu);
        }
        Ok(QTensor::new(QStorage::from_data(cpu.data()?, &self.device, gd)?, (out, inp))?)
    }

    /// A cache miss: read the matrix, quantise it on the host, and file the
    /// blocks.
    fn quantize(&self, vb: &Reader<'_>, name: &str, out: usize, inp: usize, stored: Stored) -> Res<QTensor> {
        let gd = self.quant.ok_or("asked for blocks on a load that is not quantised")?;
        let w = match stored {
            Stored::OutIn => vb.get((out, inp), name)?,
            Stored::InOut => vb.get((inp, out), name)?.t()?.contiguous()?,
        };

        // Candle quantises on the host even when the destination is a GPU, so
        // going by way of the CPU costs nothing that `quantize_onto` does not
        // also spend — and it hands over the blocks the cache wants without
        // reading them back off the device afterwards.
        let cpu = QTensor::quantize(&w, gd)?;
        self.vault.put(&vb.full(name), (out, inp), &cpu.data()?);
        Ok(cpu)
    }
}

/// The token table and the output head, which are usually one matrix.
///
/// The same forty lines stood in all three architectures here, which is two
/// chances to get tying wrong. `at` is where the table lives — under `model`
/// for Llama and DeepSeek, at the root for GPT-2 — and `root` is where an
/// untied head would be. Returns the two, and whether they share storage.
pub(crate) fn embedding(
    ld: &Loader<'_>,
    root: &Reader<'_>,
    at: &Reader<'_>,
    name: &str,
    spec: &Spec,
) -> Res<(Embed, Proj, bool)> {
    let (vocab, e) = (spec.vocab_size, spec.n_embd);

    // Whether the head is its own matrix is the *config's* statement, not the
    // checkpoint's. `tie_word_embeddings` means `lm_head.weight` **is** the
    // table, so a file that carries both is saying one matrix twice —
    // Qwen3-0.6B ships 297 MB of byte-identical duplicate, and HuggingFace
    // itself overwrites the stored copy when it ties. Reading whichever tensor
    // happened to be present is how this backend came to disagree with
    // `llama.rs`, which reads the flag.
    //
    // And an untied model with no head of its own is a broken checkpoint:
    // falling through to the table would run a different model from the one
    // the config describes, at full speed and without a word.
    let untied = !spec.tie_embeddings;
    if !untied {
        // Skipped on purpose, and recorded so the guard does not read that as
        // an omission.
        root.record("lm_head.weight");
    }

    let Some(gd) = ld.quant else {
        // Dense keeps two copies: `index_select` wants `[vocab, n_embd]` and
        // `matmul` wants the transpose, and neither is cheap to fake from the
        // other.
        // As in `Loader::proj`: the transpose happens on the host, so the
        // device is never asked to hold the table and its transpose at once.
        let table = at.get((vocab, e), name)?;
        let w = match untied {
            false => table.clone(),
            true => root.get((vocab, e), "lm_head.weight").map_err(|_| no_head())?,
        };
        let head = w.t()?.contiguous()?.to_device(&ld.device)?;
        return Ok((Embed::Dense(table.to_device(&ld.device)?), Proj::Dense(head), false));
    };

    // The arrangement this replaced, kept behind a flag so the difference it
    // makes is one environment variable wide: a dense half-precision table,
    // and the head quantised separately from it. Not cached, because a
    // measurement wants to be measuring the thing and not a file written by an
    // earlier run of it.
    if dense_embedding() {
        let table = at.get((vocab, e), name)?;
        let w = match untied {
            false => table.clone(),
            true => root.get((vocab, e), "lm_head.weight").map_err(|_| no_head())?,
        };
        let head = QTensor::quantize_onto(&w, gd, &ld.device)?;
        let table = table.to_dtype(DType::BF16)?.to_device(&ld.device)?;
        return Ok((Embed::Dense(table), Proj::Quant(QMatMul::from_qtensor(head)?), false));
    }

    // Quantised, the lookup table and the matmul want the same bytes, so a
    // tied model gets one allocation and two uses.
    let q = Arc::new(ld.quantized(at, name, vocab, e, Stored::OutIn)?);
    match untied {
        true => {
            let w = ld
                .quantized(root, "lm_head.weight", vocab, e, Stored::OutIn)
                .map_err(|_| no_head())?;
            Ok((Embed::Quant(q), Proj::Quant(QMatMul::from_qtensor(w)?), false))
        }
        false => {
            let head = QMatMul::from_arc(Arc::clone(&q))?;
            Ok((Embed::Quant(q), Proj::Quant(head), true))
        }
    }
}

fn no_head() -> String {
    concat!(
        "this model does not tie its embeddings, so it needs its own ",
        "`lm_head.weight`, and the checkpoint has none.\n",
        "If it is meant to be tied, its config is missing ",
        "`tie_word_embeddings: true`."
    )
    .to_string()
}

// ---------------------------------------------------------------------------
// The attention cache
// ---------------------------------------------------------------------------

/// The dtype a model's attention cache is kept in, if not its compute dtype.
///
/// A quantised model computes in f32, because candle's quantised matmul
/// takes nothing else, and so its cache was f32: four bytes a number where
/// a dense model's cache has two. Attention reads the whole cache every
/// token, so at long context that is a large share of what a token reads.
/// At 8,422 positions of Qwen2.5-1.5B it is about 480 MB a token, against
/// 1.64 GB of q8 weights. Kept in f16 instead, it is half that, and so is
/// the memory it holds.
///
/// f16 rather than bf16: a cached key or value is a projection's output,
/// small numbers that want the mantissa (f16 has 11 bits, bf16 8), not the
/// range bf16 buys. Attention casts the query to match, and the fused kernel
/// accumulates in f32 whatever it is given. The measured effect on what the
/// model says is in `examples/kv_precision`.
///
/// Only for quantised models on Metal, where the fused kernel is. A dense
/// model's cache is already its own two-byte dtype, or f32 because f32 was
/// asked for. `KVAD_GPU_KV_F32=1` keeps the f32 cache, for measuring what
/// this is worth.
pub(crate) fn kv_store(device: &Device, quant: Option<GgmlDType>) -> Option<DType> {
    kv_store_on(device.is_metal(), quant)
}

/// [`kv_store`] for a device that is Metal or not, for asking before there
/// is one: `model::kv_number_bytes`.
pub(crate) fn kv_store_on(metal: bool, quant: Option<GgmlDType>) -> Option<DType> {
    let off = matches!(std::env::var("KVAD_GPU_KV_F32").as_deref(), Ok("1") | Ok("true"));
    (quant.is_some() && metal && !off).then_some(DType::F16)
}

/// One layer's keys and values for every position so far, grown in place.
///
/// This was a pair of tensors rebuilt with `Tensor::cat` every step, and
/// `cat` copies both whole to add one position: the cache is copied once a
/// token, so a reply costs quadratic time in its context.
/// `examples/decode_breakdown` measured it at 0.14 ms a token at 128
/// positions of Qwen2.5-1.5B and 0.44 ms at 2,048, still growing.
///
/// Now each is a buffer with room to spare. A step writes its positions into
/// place with `slice_set`, and attention reads a `narrow` view of the part
/// in use. That view is not contiguous, and it needs not be: the fused
/// kernel takes the cache's strides, and the written-out path makes its own
/// contiguous copy of the transpose it wants anyway. When the room runs out
/// the buffer doubles, so a position is copied a handful of times over a
/// conversation rather than once per token after it, and the memory held is
/// never more than twice what is in use.
///
/// `axis` is the position axis: 2 for `[1, heads, seq, head_dim]`, 0 for
/// DeepSeek's compressed `[seq, width]`.
///
/// `store` is the dtype the positions are kept in, when it is not the one
/// they arrive in: see [`kv_store`].
pub(crate) struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
    axis: usize,
    store: Option<DType>,
}

impl KvCache {
    /// Positions the first buffer holds, so that a short exchange never grows.
    const FIRST: usize = 256;

    pub(crate) fn new(axis: usize) -> Self {
        KvCache { k: None, v: None, len: 0, axis, store: None }
    }

    /// Bytes one cached number takes, for a model that computes in
    /// `compute`: what `Session::kv_number_bytes` reports.
    pub(crate) fn number_bytes(&self, compute: DType) -> usize {
        self.store.unwrap_or(compute).size_in_bytes()
    }

    /// The same cache, keeping its positions in `store` where that is
    /// `Some`: [`kv_store`]'s answer.
    pub(crate) fn stored_as(mut self, store: Option<DType>) -> Self {
        self.store = store;
        self
    }

    /// Positions held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Add `k` and `v`'s positions after the ones held, and return all of
    /// them: views onto the buffers, not copies.
    pub(crate) fn push(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let (axis, len, store) = (self.axis, self.len, self.store);
        let m = k.dim(axis)?;
        let grow = |held: &Option<Tensor>, new: &Tensor| -> candle_core::Result<Tensor> {
            let new = match store {
                Some(dt) => new.to_dtype(dt)?,
                None => new.clone(),
            };
            let room = held.as_ref().map_or(Ok(0), |b| b.dim(axis))?;
            let buf = match held {
                Some(b) if len + m <= room => b.clone(),
                _ => {
                    let mut shape = new.dims().to_vec();
                    shape[axis] = (len + m).max(2 * room).max(Self::FIRST);
                    let bigger = Tensor::zeros(shape, new.dtype(), new.device())?;
                    if let Some(b) = held {
                        bigger.slice_set(b, axis, 0)?;
                    }
                    bigger
                }
            };
            buf.slice_set(&new.contiguous()?, axis, len)?;
            Ok(buf)
        };
        let (kb, vb) = (grow(&self.k, k)?, grow(&self.v, v)?);
        self.len = len + m;
        let views = (kb.narrow(axis, 0, self.len)?, vb.narrow(axis, 0, self.len)?);
        (self.k, self.v) = (Some(kb), Some(vb));
        Ok(views)
    }

    /// Forget everything after `len` positions. The room stays for the next
    /// turn of a conversation, except at zero, which gives the memory back.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
        if self.len == 0 {
            (self.k, self.v) = (None, None);
        }
    }
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
    /// Whole subtrees this backend knows about and does not implement, by the
    /// prefix the checkpoint files them under.
    ///
    /// By prefix rather than by name because the alternative is a list of those
    /// tensors kept in this crate, and a list that has to be remembered is what
    /// went wrong in the first place. DeepSeek V3's multi-token-prediction head
    /// is the case this exists for.
    skipped: Rc<RefCell<Vec<String>>>,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(vb: VarBuilder<'a>) -> Self {
        Reader {
            vb,
            seen: Rc::new(RefCell::new(HashSet::new())),
            skipped: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Descend into a prefix, keeping the shared record.
    pub(crate) fn pp(&self, s: impl std::fmt::Display) -> Self {
        Reader {
            vb: self.vb.pp(s.to_string()),
            seen: Rc::clone(&self.seen),
            skipped: Rc::clone(&self.skipped),
        }
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

    /// A depthwise convolution's filters, as `[channels, kernel]`.
    ///
    /// PyTorch stores a `Conv1d` with `groups = channels` as
    /// `[channels, 1, kernel]` — the middle axis is the input channels *per
    /// group*, which is one, and it carries nothing. The test fixtures write
    /// the two-dimensional spelling and the real checkpoints write the three,
    /// so both are accepted and the caller gets the shape it can use.
    pub(crate) fn conv(
        &self,
        channels: usize,
        kernel: usize,
        name: &str,
    ) -> candle_core::Result<Tensor> {
        match self.get((channels, kernel), name) {
            Ok(t) => Ok(t),
            Err(flat) => match self.vb.get((channels, 1, kernel), name) {
                Ok(t) => t.reshape((channels, kernel)),
                // The two-dimensional read is the one worth reporting: a
                // checkpoint that has neither shape has the wrong width, and
                // that is what the caller needs to be told.
                Err(_) => Err(flat),
            },
        }
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

    /// Note a whole subtree this backend deliberately does not implement.
    ///
    /// `prefix` is relative to this reader, so a prefix under `model` reads the
    /// same here as the names around it do.
    pub(crate) fn skip_under(&self, prefix: &str) {
        self.skipped.borrow_mut().push(self.full(prefix));
    }

    pub(crate) fn seen(&self) -> HashSet<String> {
        self.seen.borrow().clone()
    }

    pub(crate) fn skipped(&self) -> Vec<String> {
        self.skipped.borrow().clone()
    }
}

/// Tensors the checkpoint holds that nothing in [`GpuLlama::load`] asked for.
///
/// Reopening the files costs one pass over the safetensors headers and reads no
/// tensor data — a rounding error against the load itself. The names come from
/// the file rather than from a list kept in this crate, which is the whole
/// point: a list would have to be remembered, and forgetting is what went
/// wrong.
pub(crate) fn unread(
    paths: &[std::path::PathBuf],
    seen: &HashSet<String>,
    skipped: &[String],
) -> Res<Vec<String>> {
    let ckpt = kvad::weights::Checkpoint::open(paths)?;
    let mut left: Vec<String> = ckpt
        .names()
        .filter(|n| !seen.contains(*n) && !kvad::weights::derived(n))
        .filter(|n| !skipped.iter().any(|p| n.starts_with(p.as_str())))
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    /// A quantised projection of rows taken from the middle of a tensor gives
    /// those rows' answers, not the first rows' — see `Proj::forward` for the
    /// kernel that did the second.
    #[test]
    fn a_quantised_projection_reads_a_narrowed_input_from_where_it_starts() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; the kernel in question is Metal's");
            return;
        };
        let rand = |n: usize, seed: f32| -> Tensor {
            let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 12.9898 + seed).sin() * 43758.547).fract() - 0.5).collect();
            Tensor::from_vec(v, n, &Device::Cpu).unwrap()
        };
        let w = rand(64 * 128, 1.0).reshape((64, 128)).unwrap();
        let q = Proj::Quant(QMatMul::from_qtensor(QTensor::quantize_onto(&w, GgmlDType::Q8_0, &dev).unwrap()).unwrap());
        let x = rand(40 * 128, 2.0).reshape((40, 128)).unwrap().to_device(&dev).unwrap();
        let whole = q.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        let part = x.narrow(0, 10, 30).unwrap();
        assert_ne!(part.layout().start_offset(), 0, "the case this is about");
        let got = q.forward(&part).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(got[0], whole[10]);
        assert_eq!(got[29], whole[39]);
    }

    /// The cache holds exactly what `cat` would have: through growth, after
    /// a rewind, and from pushes of every size, on the axes both layouts use.
    #[test]
    fn the_cache_holds_what_cat_would() {
        let mut devices = vec![Device::Cpu];
        if let Ok(m) = Device::new_metal(0) {
            devices.push(m);
        }
        for dev in &devices {
            for axis in [2usize, 0] {
                let shape = |n: usize| if axis == 2 { vec![1, 2, n, 4] } else { vec![n, 6] };
                let chunk = |n: usize, seed: f32| {
                    let count: usize = shape(n).iter().product();
                    let v: Vec<f32> = (0..count).map(|i| (i as f32 * 0.37 + seed).sin()).collect();
                    Tensor::from_vec(v, shape(n), dev).unwrap()
                };
                let mut cache = KvCache::new(axis);
                let (mut ks, mut vs): (Vec<Tensor>, Vec<Tensor>) = (vec![], vec![]);
                // One big prompt, single tokens across the first growth, a
                // chunk across the second, a rewind, and more after it.
                let mut steps: Vec<usize> = vec![200];
                steps.extend(std::iter::repeat(1).take(80));
                steps.push(300);
                for (i, &n) in steps.iter().enumerate() {
                    let (k, v) = (chunk(n, i as f32), chunk(n, 100.0 + i as f32));
                    let (gk, gv) = cache.push(&k, &v).unwrap();
                    ks.push(k);
                    vs.push(v);
                    let want_k = Tensor::cat(&ks.iter().collect::<Vec<_>>(), axis).unwrap();
                    let want_v = Tensor::cat(&vs.iter().collect::<Vec<_>>(), axis).unwrap();
                    assert_eq!(cache.len(), want_k.dim(axis).unwrap());
                    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert_eq!(flat(&gk.contiguous().unwrap()), flat(&want_k), "axis {axis}, step {i}");
                    assert_eq!(flat(&gv.contiguous().unwrap()), flat(&want_v), "axis {axis}, step {i}");
                }
                // Rewind to 150, then push again: the tail is written over.
                cache.truncate(150);
                let (k, v) = (chunk(3, 7.0), chunk(3, 8.0));
                let (gk, _) = cache.push(&k, &v).unwrap();
                let whole = Tensor::cat(&ks.iter().collect::<Vec<_>>(), axis).unwrap();
                let want = Tensor::cat(&[&whole.narrow(axis, 0, 150).unwrap(), &k], axis).unwrap();
                assert_eq!(
                    gk.contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    want.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                );
                cache.truncate(0);
                assert_eq!(cache.len(), 0);
            }
        }
    }


    /// A cache kept in another dtype holds what `cat` would, cast.
    #[test]
    fn a_cache_kept_in_f16_holds_the_cast() {
        let mut devices = vec![Device::Cpu];
        if let Ok(m) = Device::new_metal(0) {
            devices.push(m);
        }
        for dev in &devices {
            let mut cache = KvCache::new(2).stored_as(Some(DType::F16));
            let mut ks = Vec::new();
            for (i, n) in [300usize, 1, 1, 40].into_iter().enumerate() {
                let v: Vec<f32> = (0..2 * n * 4).map(|j| (j as f32 * 0.11 + i as f32).cos() * 30.0).collect();
                let k = Tensor::from_vec(v, (1, 2, n, 4), dev).unwrap();
                let (gk, gv) = cache.push(&k, &k).unwrap();
                assert_eq!((gk.dtype(), gv.dtype()), (DType::F16, DType::F16));
                ks.push(k);
                let want = Tensor::cat(&ks.iter().collect::<Vec<_>>(), 2).unwrap().to_dtype(DType::F16).unwrap();
                let flat = |t: &Tensor| t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(flat(&gk.contiguous().unwrap()), flat(&want));
            }
        }
    }

}
