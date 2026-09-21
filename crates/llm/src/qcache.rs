//! Quantised weights, written to disk once and memory-mapped thereafter.
//!
//! # The problem
//!
//! [`crate::quant::Weight::quantize`] is not free. Loading Qwen2.5-0.5B at q8
//! means widening 494 million bf16 values to f32, walking them in blocks of
//! 32, finding each block's extreme, and writing out the codes. That work is
//! *identical* every time the model is loaded, and it is thrown away when the
//! process exits.
//!
//! So do it once. The first load writes the quantised arrays to a file; every
//! later load maps that file and hands the model slices into it.
//!
//! # Why mapping, not reading
//!
//! The obvious version reads the file into `Vec`s. That is already much
//! faster than re-quantising, but it still moves every byte through the CPU
//! before the model can start.
//!
//! `mmap` skips even that. The kernel wires the file's pages into the address
//! space and returns; nothing is read until a page is touched, and when it is
//! touched it comes from the page cache, which is where the file already is
//! after the first load. Startup stops scaling with model size, because
//! startup no longer does anything.
//!
//! The price is that [`Weight`] can no longer assume it owns its arrays —
//! hence [`Store`], which is either a `Vec` or a window onto a mapping, and
//! derefs to a slice either way. Every kernel in `quant.rs` is untouched by
//! this: they all take `&[i8]` and `&[f32]`, and they still do.
//!
//! # The format
//!
//! ```text
//! 0..8       magic
//! 8..64      reserved
//! 64..H      tensor data, each array padded to a 64-byte boundary
//! H..N-8     JSON header: name -> {kind, shape, offsets}
//! N-8..N     u64, little-endian: H
//! ```
//!
//! The header goes at the *end* because that lets the writer stream: append
//! each array as it is produced, remember where it went, and only then write
//! down what it did. safetensors puts the header first and can afford to,
//! because it is written by a process that already holds the whole model in
//! memory. We are quantising as we go.
//!
//! Alignment is the one thing the format has to get right. A mapping's base
//! address is page-aligned, so an array at a 64-byte offset within the file is
//! 64-byte aligned in memory, which is enough to hand out `&[f32]` — and,
//! incidentally, enough for the aligned SIMD loads in the kernels.
//!
//! # Invalidation
//!
//! A cache that can serve stale bytes is worse than no cache. Two checks:
//!
//! * [`VERSION`] is bumped whenever the *meaning* of the stored bytes
//!   changes. This is not hypothetical — the q4 scale fix in `quant.rs`
//!   changed every 4-bit code in every file, and a cache written before it
//!   would silently keep the old, worse weights alive.
//! * The source checkpoint's file names and sizes are recorded, so a
//!   re-download or a different revision does not get served from the old
//!   cache.
//!
//! Anything that fails a check is rebuilt, loudly.
//!
//! What these checks deliberately do *not* cover is bit rot: a single flipped
//! byte inside the data section passes, because catching it would mean reading
//! and hashing the whole file, which is exactly the cost this module exists to
//! avoid. safetensors makes the same trade. The checks are about provenance —
//! is this the right file, written by the right rules — not integrity.

use crate::model::{Spec, Transformer};
use crate::quant::{Parts, Precision, Weight, BLOCK};
use crate::tensor::Tensor;
use crate::weights::{Checkpoint, ModelFiles};
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::HashMap;
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const MAGIC: &[u8; 8] = b"NANOQ\x00\x00\x01";
/// Where the first array may start. Also the alignment every array gets.
const ALIGN: usize = 64;

/// Bump this when the bytes stop meaning what they used to, **or when a loader
/// starts reading a tensor it used to ignore**.
///
/// Version 1 is the first format. If the quantiser changes — a different
/// scale rule, a different packing order, a different [`BLOCK`] — this must
/// change too, or old files will be read as if they were new ones.
///
/// The second half of that rule is newer, and is the one that is easy to
/// forget. A cache holds what the build that wrote it read, so a build that
/// learns to read one more weight will not find it in an old cache — and for an
/// *optional* weight, [`Source::try_vector`] answers `None` and the model
/// quietly runs without it. That is the same silence [`Live::unread`] exists to
/// end, arriving by a route that check cannot see: it compares against the
/// checkpoint, and a mapped cache never opens one.
pub const VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Store: owned or mapped, the same either way
// ---------------------------------------------------------------------------

/// Types that may be read straight out of a memory map.
///
/// # Safety
///
/// Implementors must be fixed-size, contain no padding, have no invalid bit
/// patterns, and no `Drop`. `f32`, `i8` and `u8` all qualify: any 32 bits are
/// a valid `f32` (NaNs included), and any 8 bits are a valid `i8`. A type with
/// a niche — `bool`, `char`, a reference — would not be, because the file
/// could contain a bit pattern that is undefined behaviour to materialise.
pub unsafe trait Plain: Copy {}
unsafe impl Plain for f32 {}
unsafe impl Plain for i8 {}
unsafe impl Plain for u8 {}

/// A run of values the model can read, whoever owns the memory.
///
/// `Deref` is what makes this invisible to the kernels: `&store[..]` is a
/// plain slice, and the matmuls never learn whether the bytes were computed
/// this run or mapped from a file written last week.
pub enum Store<T: Plain> {
    Owned(Vec<T>),
    Mapped {
        /// Kept alive by every `Store` that points into it; the mapping is
        /// unmapped when the last weight is dropped.
        map: Arc<memmap2::Mmap>,
        off: usize,
        len: usize,
        _t: PhantomData<T>,
    },
}

impl<T: Plain> Store<T> {
    /// Point at `len` values starting `off` bytes into `map`.
    fn mapped(map: &Arc<memmap2::Mmap>, off: usize, len: usize) -> Res<Self> {
        let size = std::mem::size_of::<T>();
        let end = off.checked_add(len * size).ok_or("array offset overflows")?;
        if end > map.len() {
            return Err(format!("array at {off}+{} runs past the file", len * size).into());
        }
        // The format aligns every array to 64 bytes and a mapping starts on a
        // page boundary, so this should never fire. It is checked anyway,
        // because the alternative to checking is undefined behaviour.
        let addr = map.as_ptr() as usize + off;
        if addr % std::mem::align_of::<T>() != 0 {
            let t = std::any::type_name::<T>();
            return Err(format!("array at {off} is misaligned for {t}").into());
        }
        Ok(Store::Mapped { map: Arc::clone(map), off, len, _t: PhantomData })
    }
}

impl<T: Plain> std::ops::Deref for Store<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Store::Owned(v) => v,
            // SAFETY: `mapped()` checked bounds and alignment, `T: Plain`
            // makes every bit pattern valid, and the `Arc` keeps the mapping
            // alive for as long as this `Store` exists. The usual mmap
            // caveat applies: another process truncating the file underneath
            // us is undefined behaviour, the same caveat `Checkpoint` carries.
            Store::Mapped { map, off, len, .. } => unsafe {
                std::slice::from_raw_parts(map.as_ptr().add(*off) as *const T, *len)
            },
        }
    }
}

/// View a slice of plain values as the bytes that represent them.
fn as_bytes<T: Plain>(v: &[T]) -> &[u8] {
    // SAFETY: `T: Plain` is exactly the promise that this is meaningful —
    // fixed size, no padding, no pointers.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// Where a model's weights come from
// ---------------------------------------------------------------------------

/// The loaders' view of a checkpoint.
///
/// `gpt2.rs` and `llama.rs` ask for named matrices and vectors and do not care
/// whether the answer is being quantised from bf16 right now or mapped from a
/// file. That indirection is the whole reason the cache needs no changes in
/// either architecture beyond the names of the calls.
pub trait Source {
    /// A weight matrix, at whatever precision this source works in.
    fn matrix(&self, name: &str) -> Res<Weight>;
    /// As [`Source::matrix`], returning `None` for a tensor the model does not
    /// have — an untied output head, a bias this family dropped.
    fn try_matrix(&self, name: &str) -> Option<Weight>;
    /// A matrix stored `[in, out]` that the engine wants as `[out, in]`.
    ///
    /// GPT-2's checkpoint was written for a `Conv1D`; transposing on the way
    /// in lets both architectures share one matmul kernel. The cache stores
    /// the result, so this is a property of the *checkpoint*, not of the
    /// cache, and the mapped source ignores the distinction entirely.
    fn matrix_t(&self, name: &str) -> Res<Weight>;
    /// Rows `start..start + count` of a stored matrix, on their own.
    ///
    /// DeepSeek stores one matrix per layer holding every head's
    /// up-projection for keys and values back to back, and the engine wants
    /// them apart: the key half is folded into the query and the value half
    /// is applied after the softmax, so they are used at opposite ends of the
    /// attention and never together. Slicing at load time costs one pass and
    /// saves doing it on every token.
    ///
    /// The slice is cached under a name derived from this one, so the second
    /// load maps it like any other weight.
    fn matrix_rows(&self, name: &str, start: usize, count: usize) -> Res<Weight>;
    /// As [`Source::matrix_rows`], transposed: `[count, cols]` stored as
    /// `[cols, count]`.
    ///
    /// The engine has one matmul kernel and it computes `x @ Wᵀ`. A slice
    /// that has to be applied the *other* way round is transposed once here
    /// rather than given a second kernel.
    fn matrix_rows_t(&self, name: &str, start: usize, count: usize) -> Res<Weight>;
    /// A 1-D tensor: norm weights, biases. Never quantised — they are a
    /// rounding error in both size and cost.
    fn vector(&self, name: &str) -> Res<Vec<f32>>;
    fn try_vector(&self, name: &str) -> Option<Vec<f32>>;

    /// Note a tensor this architecture knows about and deliberately does not
    /// read, so that [`Live`]'s check for weights nobody wanted does not
    /// mistake a decision for an omission.
    ///
    /// The duplicate output head in a tied checkpoint is one. The default does
    /// nothing, which is right for a source reading a cache file: it contains
    /// what was recorded and nothing else.
    fn skip(&self, _name: &str) {}

    /// The same, for a whole subtree of tensors at once.
    ///
    /// DeepSeek V3's multi-token-prediction head is the case this exists for: a
    /// complete extra transformer block, an embedding table and two norms, all
    /// filed under `layers.{n_layer}` and up. It is a training-time device and
    /// nothing here needs it to run the model.
    ///
    /// By prefix rather than by name because the alternative is a list of those
    /// tensors kept in this crate, and a list that has to be remembered is what
    /// went wrong in the first place.
    fn skip_under(&self, _prefix: &str) {}
}

/// The output head: its own matrix, or `None` when the model ties it to the
/// embedding table.
///
/// One function rather than the same `match` in three architectures, because all
/// three wrote it the same way and so shared its two faults.
///
/// Tying is the *config's* statement about the model. `tie_word_embeddings`
/// means the head **is** the table, whatever else the file carries: Qwen3-0.6B
/// ships an `lm_head.weight` byte-identical to `embed_tokens.weight`, stating
/// one matrix twice, and HuggingFace overwrites the stored copy when it ties.
/// So the flag decides, and the redundant copy is skipped by name.
///
/// And an untied model with no head of its own is a broken checkpoint. This
/// used to fall through to the embedding table, which is a different model from
/// the one the config describes, run at full speed without a word.
pub fn head(src: &dyn Source, spec: &Spec, name: &str) -> Res<Option<Weight>> {
    if spec.tie_embeddings {
        src.skip(name);
        return Ok(None);
    }
    match src.try_matrix(name) {
        Some(w) => Ok(Some(w)),
        None => Err(format!(
            "this model does not tie its embeddings, so it needs its own `{name}`, and the \
             checkpoint has none.\nIf it is meant to be tied, its config is missing \
             `tie_word_embeddings: true`."
        )
        .into()),
    }
}

/// The name a sliced weight is cached under.
///
/// Derived rather than passed in, so the run that writes the cache and the run
/// that reads it cannot disagree about it.
fn slice_name(name: &str, start: usize, count: usize, transposed: bool) -> String {
    let t = if transposed { "t" } else { "" };
    format!("{name}#{start}+{count}{t}")
}

/// A contiguous band of rows, as a matrix of its own.
fn rows_of(t: &Tensor, name: &str, start: usize, count: usize) -> Res<Tensor> {
    if start + count > t.rows {
        return Err(format!("`{name}` has {} rows; asked for {count} starting at {start}", t.rows).into());
    }
    let mut data = Vec::with_capacity(count * t.cols);
    for r in start..start + count {
        data.extend_from_slice(t.row(r));
    }
    Ok(Tensor::new(count, t.cols, data))
}

// ---------------------------------------------------------------------------
// Source 1: the checkpoint, quantising as it goes (and optionally recording)
// ---------------------------------------------------------------------------

/// Quantise straight from safetensors, optionally teeing everything to a
/// cache file on the way past.
///
/// Recording during the load rather than in a separate pass matters: the f32
/// copy of each tensor exists for a moment anyway, so writing it out costs
/// one extra traversal of data already in cache, and peak memory does not
/// move.
pub struct Live<'a> {
    ckpt: &'a Checkpoint,
    precision: Precision,
    out: RefCell<Option<Writer>>,
    /// Why recording stopped, if it did.
    failure: RefCell<Option<String>>,
    /// Every checkpoint tensor an architecture asked about, by the name the
    /// file files it under rather than the name it was asked for — the two
    /// differ by a prefix, and only the former can be compared against
    /// [`Checkpoint::names`].
    ///
    /// The point is [`Live::unread`]. A weight nobody reads is a piece of the
    /// model that is not running, and nothing else notices: a loader asks for
    /// what it knows about and a checkpoint has no opinion about the rest. That
    /// is how the GPU backend ran a Llama forward pass over a Qwen3 model at
    /// full speed, and it could as easily happen here.
    seen: RefCell<HashSet<String>>,
}

impl<'a> Live<'a> {
    pub fn new(ckpt: &'a Checkpoint, precision: Precision) -> Self {
        Live {
            ckpt,
            precision,
            out: RefCell::new(None),
            failure: RefCell::new(None),
            seen: RefCell::new(HashSet::new()),
        }
    }

    /// Start recording to `path`. Writes go to a temporary file and are moved
    /// into place by [`Live::finish`], so an interrupted load leaves no
    /// half-written cache behind.
    pub fn record_to(&mut self, path: &Path) -> Res<()> {
        *self.out.get_mut() = Some(Writer::create(path)?);
        Ok(())
    }

    /// Tee one array to the cache file.
    ///
    /// A write failure — a full disk, a read-only directory — abandons the
    /// cache and is otherwise ignored. It must never change what this load
    /// returns: the cache is an optimisation, and an optimisation that can
    /// silently drop a weight matrix is a correctness bug.
    fn tee(&self, write: impl FnOnce(&mut Writer) -> Res<()>) {
        let mut out = self.out.borrow_mut();
        let failed = match out.as_mut() {
            Some(writer) => write(writer).err(),
            None => None,
        };
        if let Some(e) = failed {
            *self.failure.borrow_mut() = Some(e.to_string());
            *out = None; // drops the Writer, which removes the partial file
        }
    }

    /// Note that some architecture asked about `name`, under whichever spelling
    /// this checkpoint files it.
    ///
    /// Called by every read, and on its own from [`Source::skip`] for a tensor
    /// deliberately passed over. A name the checkpoint does not have records
    /// nothing, which is right: there is no such tensor to leave unread.
    fn record(&self, name: &str) {
        if let Some(key) = self.ckpt.resolve(name) {
            self.seen.borrow_mut().insert(key.to_string());
        }
    }

    /// Checkpoint tensors that nothing asked about, derived buffers aside.
    ///
    /// Call it after the architecture has finished loading. Anything here is a
    /// weight this build does not implement, which is worth failing over: it is
    /// a wrong answer at full speed rather than an error.
    pub fn unread(&self) -> Vec<String> {
        let seen = self.seen.borrow();
        let mut left: Vec<String> = self
            .ckpt
            .names()
            .filter(|n| !seen.contains(*n) && !crate::weights::derived(n))
            .map(str::to_string)
            .collect();
        left.sort();
        left
    }

    /// Why nothing was cached, when nothing was.
    pub fn failure(&self) -> Option<String> {
        self.failure.borrow().clone()
    }

    /// Close the cache file, stamping it with everything needed to decide
    /// later whether it is still valid. Returns the bytes written.
    pub fn finish(&mut self, repo: &str, files: &[PathBuf], spec: &Spec) -> Res<Option<u64>> {
        let Some(writer) = self.out.get_mut().take() else {
            return Ok(None);
        };
        writer.finish(header(repo, files, spec, self.precision)?).map(Some)
    }

    fn quantized(&self, t: Tensor) -> Weight {
        Weight::quantize(t, self.precision)
    }
}

impl Source for Live<'_> {
    fn matrix(&self, name: &str) -> Res<Weight> {
        self.record(name);
        let w = self.quantized(self.ckpt.get(name)?);
        self.tee(|out| out.put_weight(name, &w));
        Ok(w)
    }

    fn try_matrix(&self, name: &str) -> Option<Weight> {
        self.record(name);
        let w = self.quantized(self.ckpt.try_get(name)?);
        self.tee(|out| out.put_weight(name, &w));
        Some(w)
    }

    fn matrix_t(&self, name: &str) -> Res<Weight> {
        self.record(name);
        let w = self.quantized(self.ckpt.get(name)?.transposed());
        self.tee(|out| out.put_weight(name, &w));
        Ok(w)
    }

    fn matrix_rows(&self, name: &str, start: usize, count: usize) -> Res<Weight> {
        self.record(name);
        let w = self.quantized(rows_of(&self.ckpt.get(name)?, name, start, count)?);
        let as_name = slice_name(name, start, count, false);
        self.tee(|out| out.put_weight(&as_name, &w));
        Ok(w)
    }

    fn matrix_rows_t(&self, name: &str, start: usize, count: usize) -> Res<Weight> {
        self.record(name);
        let w = self.quantized(rows_of(&self.ckpt.get(name)?, name, start, count)?.transposed());
        let as_name = slice_name(name, start, count, true);
        self.tee(|out| out.put_weight(&as_name, &w));
        Ok(w)
    }

    fn vector(&self, name: &str) -> Res<Vec<f32>> {
        self.record(name);
        let v = self.ckpt.get_flat(name)?;
        self.tee(|out| out.put_vector(name, &v));
        Ok(v)
    }

    fn try_vector(&self, name: &str) -> Option<Vec<f32>> {
        self.record(name);
        let v = self.ckpt.try_get_flat(name)?;
        self.tee(|out| out.put_vector(name, &v));
        Some(v)
    }

    fn skip(&self, name: &str) {
        self.record(name);
    }

    fn skip_under(&self, prefix: &str) {
        let names: Vec<String> =
            self.ckpt.names_under(prefix).into_iter().map(str::to_string).collect();
        self.seen.borrow_mut().extend(names);
    }
}

// ---------------------------------------------------------------------------
// Source 2: the cache file
// ---------------------------------------------------------------------------

/// One array's location in the file: byte offset, element count.
type Span = (usize, usize);

enum Entry {
    F32 { shape: (usize, usize), data: Span },
    Q8 { shape: (usize, usize), scales: Span, qs: Span },
    Q4 { shape: (usize, usize), scales: Span, qs: Span },
}

/// A cache file, mapped.
pub struct Mapped {
    map: Arc<memmap2::Mmap>,
    entries: HashMap<String, Entry>,
}

impl Mapped {
    /// Map `path` if it exists and is still valid for this checkpoint.
    ///
    /// `Ok(None)` means there is no cache yet. `Err` means there is one and it
    /// cannot be trusted — the caller rebuilds, and says why.
    pub fn open(
        path: &Path,
        repo: &str,
        files: &[PathBuf],
        spec: &Spec,
        precision: Precision,
    ) -> Res<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let file = std::fs::File::open(path)?;
        // SAFETY: read-only, and the same caveat as every other mmap here.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < ALIGN + 8 || &map[..8] != MAGIC {
            return Err("not a quantised-weight file".into());
        }

        // The header length lives in the last eight bytes; the header sits
        // just before them.
        let n = map.len();
        let at = u64::from_le_bytes(map[n - 8..].try_into().unwrap()) as usize;
        if at < ALIGN || at > n - 8 {
            return Err("header offset is out of range".into());
        }
        let json: serde_json::Value = serde_json::from_slice(&map[at..n - 8])?;

        let want = header(repo, files, spec, precision)?;
        for key in ["version", "precision", "block", "repo", "spec", "sources"] {
            if json.get(key) != want.get(key) {
                let was = json.get(key).unwrap_or(&serde_json::Value::Null).to_string();
                let was = brief(&was);
                return Err(format!("`{key}` changed since it was written ({was})").into());
            }
        }

        let tensors = json.get("tensors").and_then(|t| t.as_object()).ok_or("no tensors")?;
        let mut entries = HashMap::with_capacity(tensors.len());
        for (name, entry) in tensors {
            entries.insert(name.clone(), parse_entry(entry)?);
        }

        Ok(Some(Mapped { map: Arc::new(map), entries }))
    }

    /// Bytes on disk.
    pub fn bytes(&self) -> usize {
        self.map.len()
    }

    fn build(&self, name: &str) -> Res<Weight> {
        let entry = self.entries.get(name).ok_or_else(|| format!("`{name}` not in the cache"))?;
        match entry {
            Entry::F32 { shape, data } => {
                let (rows, cols) = *shape;
                // The f32 path still owns its data: `Tensor` is used for
                // arithmetic elsewhere and wants a `Vec`. It is also the path
                // the cache is not built for.
                Ok(Weight::from_f32(Tensor::new(rows, cols, self.f32s(*data)?.to_vec())))
            }
            Entry::Q8 { shape, scales, qs } => Weight::from_q8(
                shape.0,
                shape.1,
                Store::mapped(&self.map, scales.0, scales.1)?,
                Store::mapped(&self.map, qs.0, qs.1)?,
            ),
            Entry::Q4 { shape, scales, qs } => Weight::from_q4(
                shape.0,
                shape.1,
                Store::mapped(&self.map, scales.0, scales.1)?,
                Store::mapped(&self.map, qs.0, qs.1)?,
            ),
        }
    }

    fn f32s(&self, span: Span) -> Res<Store<f32>> {
        Store::mapped(&self.map, span.0, span.1)
    }
}

impl Source for Mapped {
    fn matrix(&self, name: &str) -> Res<Weight> {
        self.build(name)
    }

    fn matrix_rows(&self, name: &str, start: usize, count: usize) -> Res<Weight> {
        self.build(&slice_name(name, start, count, false))
    }

    fn matrix_rows_t(&self, name: &str, start: usize, count: usize) -> Res<Weight> {
        self.build(&slice_name(name, start, count, true))
    }

    fn try_matrix(&self, name: &str) -> Option<Weight> {
        // Absent from the file means absent from the model: the run that
        // wrote it asked the same question and got nothing.
        self.build(name).ok()
    }

    fn matrix_t(&self, name: &str) -> Res<Weight> {
        // Already transposed when it was written.
        self.build(name)
    }

    fn vector(&self, name: &str) -> Res<Vec<f32>> {
        match self.entries.get(name) {
            Some(Entry::F32 { shape, data }) if shape.0 == 1 => Ok(self.f32s(*data)?.to_vec()),
            Some(_) => Err(format!("`{name}` is a quantised matrix, not a vector").into()),
            None => Err(format!("`{name}` not in the cache").into()),
        }
    }

    fn try_vector(&self, name: &str) -> Option<Vec<f32>> {
        match self.entries.get(name) {
            Some(Entry::F32 { shape, data }) if shape.0 == 1 => {
                Some(self.f32s(*data).ok()?.to_vec())
            }
            _ => None,
        }
    }
}

/// Keep a diagnostic short enough for a status line.
fn brief(s: &str) -> String {
    match s.char_indices().nth(48) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

fn parse_entry(v: &serde_json::Value) -> Res<Entry> {
    let span = |key: &str| -> Res<Span> {
        let a = v.get(key).and_then(|s| s.as_array()).ok_or_else(|| format!("no `{key}`"))?;
        let n = |i: usize| a.get(i).and_then(|x| x.as_u64()).map(|x| x as usize);
        Ok((n(0).ok_or("bad span")?, n(1).ok_or("bad span")?))
    };
    let shape = {
        let a = v.get("shape").and_then(|s| s.as_array()).ok_or("no `shape`")?;
        let n = |i: usize| a.get(i).and_then(|x| x.as_u64()).map(|x| x as usize);
        (n(0).ok_or("bad shape")?, n(1).ok_or("bad shape")?)
    };
    Ok(match v.get("kind").and_then(|k| k.as_str()).ok_or("no `kind`")? {
        "f32" => Entry::F32 { shape, data: span("data")? },
        "q8" => Entry::Q8 { shape, scales: span("scales")?, qs: span("qs")? },
        "q4" => Entry::Q4 { shape, scales: span("scales")?, qs: span("qs")? },
        other => return Err(format!("unknown kind `{other}`").into()),
    })
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

struct Writer {
    file: std::io::BufWriter<std::fs::File>,
    tmp: PathBuf,
    final_path: PathBuf,
    /// Set once the file has been renamed into place, so `Drop` knows not to
    /// delete it.
    done: bool,
    /// Bytes written so far — also the offset the next array will land at,
    /// before padding.
    at: usize,
    tensors: serde_json::Map<String, serde_json::Value>,
}

impl Writer {
    fn create(path: &Path) -> Res<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // A distinct temp name per process, so two loads racing each other
        // both produce a complete file and the rename picks a winner.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        file.write_all(MAGIC)?;
        file.write_all(&[0u8; ALIGN - 8])?;
        Ok(Writer {
            file,
            tmp,
            final_path: path.to_path_buf(),
            done: false,
            at: ALIGN,
            tensors: serde_json::Map::new(),
        })
    }

    /// Append one array, padded up to the next [`ALIGN`] boundary first.
    fn put<T: Plain>(&mut self, v: &[T]) -> Res<serde_json::Value> {
        let pad = (ALIGN - self.at % ALIGN) % ALIGN;
        self.file.write_all(&[0u8; ALIGN][..pad])?;
        self.at += pad;

        let off = self.at;
        let bytes = as_bytes(v);
        self.file.write_all(bytes)?;
        self.at += bytes.len();
        Ok(serde_json::json!([off, v.len()]))
    }

    fn put_weight(&mut self, name: &str, w: &Weight) -> Res<()> {
        if self.tensors.contains_key(name) {
            return Ok(()); // tied weights: stored once
        }
        let shape = serde_json::json!([w.rows(), w.cols()]);
        let entry = match w.parts() {
            Parts::F32(data) => {
                serde_json::json!({ "kind": "f32", "shape": shape, "data": self.put(data)? })
            }
            Parts::Q8 { scales, qs } => serde_json::json!({
                "kind": "q8", "shape": shape,
                "scales": self.put(scales)?, "qs": self.put(qs)?,
            }),
            Parts::Q4 { scales, qs } => serde_json::json!({
                "kind": "q4", "shape": shape,
                "scales": self.put(scales)?, "qs": self.put(qs)?,
            }),
        };
        self.tensors.insert(name.to_string(), entry);
        Ok(())
    }

    fn put_vector(&mut self, name: &str, v: &[f32]) -> Res<()> {
        if self.tensors.contains_key(name) {
            return Ok(());
        }
        let entry = serde_json::json!({
            "kind": "f32", "shape": [1, v.len()], "data": self.put(v)?,
        });
        self.tensors.insert(name.to_string(), entry);
        Ok(())
    }

    /// Write the header, then its offset, then move the file into place.
    fn finish(mut self, mut header: serde_json::Value) -> Res<u64> {
        header["tensors"] = serde_json::Value::Object(std::mem::take(&mut self.tensors));
        let json = serde_json::to_vec(&header)?;

        let at = self.at as u64;
        self.file.write_all(&json)?;
        self.file.write_all(&at.to_le_bytes())?;
        self.file.flush()?;
        // Ask for the bytes to actually reach the disk before the rename
        // publishes them. Without this, a crash between the two can leave a
        // correctly-named file full of zeros.
        self.file.get_ref().sync_all()?;
        let size = self.file.get_ref().metadata()?.len();

        std::fs::rename(&self.tmp, &self.final_path)?;
        self.done = true;
        Ok(size)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // Reached only when the load failed partway: clean up the partial
        // file rather than leave it to be mistaken for a finished one.
        if !self.done {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// How to tell whether a checkpoint file is still the same file.
///
/// The HuggingFace cache is content-addressed: every file in a snapshot is a
/// symlink to `blobs/<sha256 of the contents>`. So reading the link gives a
/// content hash for nothing — no bytes touched — which beats comparing sizes,
/// because a re-download at a different revision can easily land on the same
/// length.
///
/// A file that is not a symlink is a model somebody put there, and the likely
/// somebody is `nanograd`, saving over its last attempt. Size alone is
/// useless for that: the size of a checkpoint is decided by the architecture,
/// so a retrained model is the same length *to the byte*, and the cache
/// happily served the weights of the model it replaced. So it is the size and
/// the modification time, which is `make`'s answer and has `make`'s flaw — a
/// copy that preserves timestamps can defeat it — and costs nothing, where
/// hashing a hand-placed 8 GB checkpoint on every load would cost more than
/// the cache saves.
fn identify(path: &Path) -> Res<String> {
    if let Ok(target) = std::fs::read_link(path) {
        if let Some(name) = target.file_name().and_then(|n| n.to_str()) {
            return Ok(format!("sha256:{name}"));
        }
    }
    let meta = std::fs::metadata(path)?;
    let modified = meta.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    Ok(format!("bytes:{} modified:{modified}", meta.len()))
}

/// Everything that has to match for a cache file to be reusable.
fn header(
    repo: &str,
    files: &[PathBuf],
    spec: &Spec,
    precision: Precision,
) -> Res<serde_json::Value> {
    let mut sources = Vec::new();
    for path in files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        sources.push(serde_json::json!([name, identify(path)?]));
    }
    Ok(serde_json::json!({
        "format": "nanollm-quant",
        "version": VERSION,
        "precision": precision.to_string(),
        "block": BLOCK,
        "repo": repo,
        // Not for validation's sake alone: a header you can read with `tail`
        // and `jq` is worth the forty bytes.
        "spec": {
            "arch": spec.arch.to_string(),
            "n_layer": spec.n_layer,
            "n_embd": spec.n_embd,
            "vocab_size": spec.vocab_size,
            "tie_embeddings": spec.tie_embeddings,
        },
        "sources": sources,
    }))
}

// ---------------------------------------------------------------------------
// Policy: where the files live, and when to use them
// ---------------------------------------------------------------------------

/// Root of the quantised-weight cache.
///
/// Deliberately *not* inside the HuggingFace cache: nothing there is ours, and
/// `huggingface-cli delete-cache` should not have an opinion about our files.
pub fn cache_root() -> PathBuf {
    if let Ok(v) = std::env::var("KVAD_QUANT_CACHE") {
        return PathBuf::from(v);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(".cache"));
    base.join("kvad").join("quant")
}

fn home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

/// The cache file for one model at one precision.
pub fn path_for(repo: &str, precision: Precision) -> PathBuf {
    cache_root().join(file_name_for(repo, precision))
}

/// A repo id becomes its own file name, which is what makes the directory
/// readable with `ls`. A model loaded from a directory is known by its
/// absolute path (see `weights::model_id`), and a path makes a poor file name:
/// it can be longer than a file name may be, and flattening its slashes lets
/// two different paths collide. So it is filed under its last component, for
/// the human, and a hash of the whole path, for correctness. Nothing reads
/// the model's name back out of the file name; the header has it.
fn file_name_for(repo: &str, precision: Precision) -> String {
    let path = Path::new(repo);
    if !path.is_absolute() {
        return format!("{}.{precision}.nq", repo.replace('/', "--"));
    }
    // FNV-1a: five lines, and unlike the standard library's hasher, promised
    // to give the same answer after the next compiler upgrade.
    let hash = repo.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
    let last = path.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    format!("local--{last}-{hash:016x}.{precision}.nq")
}

/// The model a cache file says it belongs to, read from its header.
fn repo_of(path: &Path) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let n = file.metadata().ok()?.len();
    let mut at = [0u8; 8];
    file.seek(SeekFrom::End(-8)).ok()?;
    file.read_exact(&mut at).ok()?;
    let at = u64::from_le_bytes(at);
    // A header is a few kilobytes of JSON. Anything else is not a header, and
    // is certainly not worth allocating for.
    let len = n.checked_sub(8)?.checked_sub(at).filter(|&len| len < (1 << 24))?;
    let mut json = vec![0u8; len as usize];
    file.seek(SeekFrom::Start(at)).ok()?;
    file.read_exact(&mut json).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&json).ok()?;
    json.get("repo")?.as_str().map(String::from)
}

/// Everything currently cached: (path, repo, precision, bytes).
pub fn entries() -> Vec<(PathBuf, String, String, u64)> {
    let Ok(dir) = std::fs::read_dir(cache_root()) else {
        return Vec::new();
    };
    let mut out: Vec<_> = dir
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            let stem = path.file_name()?.to_str()?.strip_suffix(".nq")?.to_string();
            let (repo, precision) = stem.rsplit_once('.')?;
            let bytes = e.metadata().ok()?.len();
            // The header knows; the file name is a guess for a file whose
            // header cannot be read.
            let repo = repo_of(&path).unwrap_or_else(|| repo.replacen("--", "/", 1));
            Some((path, repo, precision.to_string(), bytes))
        })
        .collect();
    out.sort();
    out
}

/// Delete every cache file belonging to `repo`. Returns how many went.
pub fn forget(repo: &str) -> usize {
    entries()
        .into_iter()
        .filter(|(_, r, _, _)| r == repo)
        .filter(|(p, _, _, _)| std::fs::remove_file(p).is_ok())
        .count()
}

fn enabled() -> bool {
    !matches!(std::env::var("KVAD_NO_QCACHE").as_deref(), Ok("1") | Ok("true"))
}

/// Build the model for `spec`, using the quantised cache where it can.
///
/// The three outcomes, in the order they are tried:
///
/// 1. A valid cache file exists — map it; nothing is read or computed.
/// 2. No cache, or an invalid one — quantise from the checkpoint and write
///    one out on the way past.
/// 3. f32, or caching disabled — quantise nothing, cache nothing.
pub fn load(
    repo: &str,
    files: &ModelFiles,
    spec: &Spec,
    precision: Precision,
    progress: &mut dyn FnMut(&str),
) -> Res<Box<dyn Transformer>> {
    // There is nothing to cache at f32: the checkpoint already *is* the
    // weights, and a copy of them widened to 32 bits would be twice the size
    // of the file it came from.
    let path = (precision != Precision::F32 && enabled()).then(|| path_for(repo, precision));

    let mut rebuilding = false;
    if let Some(path) = &path {
        match Mapped::open(path, repo, &files.weights, spec, precision) {
            Ok(Some(cache)) => {
                let mb = cache.bytes() / 1_000_000;
                progress(&format!("mapping {precision} weights ({mb} MB)"));
                return build(&cache, spec);
            }
            Ok(None) => {}
            Err(e) => {
                progress(&format!("stale {precision} cache: {e}"));
                rebuilding = true;
            }
        }
    }

    let ckpt = Checkpoint::open(&files.weights)?;
    let mut live = Live::new(&ckpt, precision);
    if let Some(path) = &path {
        match live.record_to(path) {
            Ok(()) => {
                let why = if rebuilding { "rebuilding" } else { "first load" };
                progress(&format!("quantising to {precision} ({why})"));
            }
            // A read-only or full cache directory is not a reason to fail.
            Err(e) => progress(&format!("not caching: {e}")),
        }
    }

    let model = build(&live, spec)?;

    // Before `finish`, so a refused load leaves no cache behind: dropping the
    // `Writer` removes the partial file.
    let left = live.unread();
    if !left.is_empty() {
        return Err(format!(
            "this checkpoint holds {} tensor(s) that the `{}` loader never reads:\n  {}\n\
             A weight nobody reads is a piece of the model that is not running — a wrong \
             answer\nat full speed rather than an error: this build does not implement all \
             of this model.",
            left.len(),
            spec.arch,
            crate::weights::collapsed(&left).join("\n  "),
        )
        .into());
    }

    match live.finish(repo, &files.weights, spec) {
        Ok(Some(bytes)) => progress(&format!("cached for next time ({} MB)", bytes / 1_000_000)),
        Ok(None) => {
            if let Some(e) = live.failure() {
                progress(&format!("not caching: {e}"));
            }
        }
        // Failing to *write* the cache is not a reason to fail the load.
        Err(e) => progress(&format!("not caching: {e}")),
    }
    Ok(model)
}

/// Hand the weights to whichever architecture the config named.
///
/// There is no `match` here any more, and that is the point: a new
/// architecture is a module and one line in `model/arch.rs`, not an arm in
/// every loader that ever learned the old list.
fn build(src: &dyn Source, spec: &Spec) -> Res<Box<dyn Transformer>> {
    spec.arch.load(src, spec.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, CacheShape, Json};

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nq-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn spec() -> Spec {
        Spec {
            arch: Arch::require("llama"),
            n_layer: 1,
            n_head: 1,
            n_kv_head: 1,
            n_embd: 32,
            head_dim: 32,
            n_ctx: 8,
            vocab_size: 4,
            intermediate: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
            tie_embeddings: true,
            cache: CacheShape { k: 32, v: 32 },
            config: Json::default(),
        }
    }

    /// Everything a weight knows must survive the round trip: the codes, the
    /// scales, the shape, and therefore the product it computes.
    #[test]
    fn weights_survive_the_round_trip() {
        for precision in [Precision::Q8, Precision::Q4] {
            let path = tmp(&format!("roundtrip-{precision}"));
            let data: Vec<f32> =
                (0..8 * 64).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
            let before = Weight::quantize(Tensor::new(8, 64, data.clone()), precision);

            let mut w = Writer::create(&path).unwrap();
            w.put_weight("a", &before).unwrap();
            w.put_vector("norm", &[1.0, 2.0, 3.0]).unwrap();
            w.finish(header("r", &[], &spec(), precision).unwrap()).unwrap();

            let cache = Mapped::open(&path, "r", &[], &spec(), precision).unwrap().unwrap();
            let after = cache.matrix("a").unwrap();
            assert_eq!(after.precision(), precision);
            assert_eq!((after.rows(), after.cols()), (8, 64));
            assert_eq!(after.bytes(), before.bytes());
            assert_eq!(cache.vector("norm").unwrap(), vec![1.0, 2.0, 3.0]);

            // The mapped weight must give bit-identical answers, not close
            // ones: it is the same codes and the same scales.
            let x: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
            assert_eq!(after.matvec_bt(&x, None), before.matvec_bt(&x, None));
            // And row lookup, which is the embedding path.
            assert_eq!(after.row(3), before.row(3));

            std::fs::remove_file(&path).unwrap();
        }
    }

    /// The case that was broken: a model retrained and saved over itself. Its
    /// file has the same name and exactly the same length, and a cache that
    /// identified it by those served the old model's weights under the new
    /// model's name.
    #[test]
    fn a_retrained_model_is_not_the_model_it_replaced() {
        let weights = tmp("retrained.safetensors");
        let cache = tmp("retrained");
        let files = [weights.clone()];

        std::fs::write(&weights, [1u8; 64]).unwrap();
        let w = Writer::create(&cache).unwrap();
        w.finish(header("out/model", &files, &spec(), Precision::Q8).unwrap()).unwrap();
        assert!(Mapped::open(&cache, "out/model", &files, &spec(), Precision::Q8).unwrap().is_some());

        // Same length, different floats, saved a second later. The time is
        // set rather than waited for, so the test does not depend on how
        // finely this file system keeps it.
        let before = std::fs::metadata(&weights).unwrap().modified().unwrap();
        std::fs::write(&weights, [2u8; 64]).unwrap();
        let file = std::fs::File::options().write(true).open(&weights).unwrap();
        file.set_modified(before + std::time::Duration::from_secs(1)).unwrap();
        drop(file);

        let error = Mapped::open(&cache, "out/model", &files, &spec(), Precision::Q8).err().unwrap();
        assert!(error.to_string().contains("`sources` changed"), "{error}");

        // And a file nobody touched is still itself.
        assert_eq!(identify(&weights).unwrap(), identify(&weights).unwrap());
        std::fs::remove_file(&weights).unwrap();
        std::fs::remove_file(&cache).unwrap();
    }

    /// A cache that can be read after the rules changed is worse than none.
    #[test]
    fn stale_caches_are_rejected() {
        let path = tmp("stale");
        let w = Writer::create(&path).unwrap();
        w.finish(header("repo-a", &[], &spec(), Precision::Q8).unwrap()).unwrap();

        // Same everything: fine.
        assert!(Mapped::open(&path, "repo-a", &[], &spec(), Precision::Q8).unwrap().is_some());
        // Different precision, different model, different shape: not fine.
        assert!(Mapped::open(&path, "repo-a", &[], &spec(), Precision::Q4).is_err());
        assert!(Mapped::open(&path, "repo-b", &[], &spec(), Precision::Q8).is_err());
        let mut other = spec();
        other.n_layer = 2;
        assert!(Mapped::open(&path, "repo-a", &[], &other, Precision::Q8).is_err());

        // A file that is not one of ours at all.
        let junk = tmp("junk");
        std::fs::write(&junk, vec![7u8; 256]).unwrap();
        assert!(Mapped::open(&junk, "repo-a", &[], &spec(), Precision::Q8).is_err());

        // A missing file is not an error, just an empty cache.
        assert!(Mapped::open(&tmp("absent"), "repo-a", &[], &spec(), Precision::Q8)
            .unwrap()
            .is_none());

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&junk).unwrap();
    }

    /// The mapping hands out `&[f32]`, so every array must sit on an address
    /// the type is allowed to be read from.
    #[test]
    fn arrays_are_aligned() {
        let path = tmp("align");
        let mut w = Writer::create(&path).unwrap();
        // Odd-length byte arrays between the float ones: without padding, the
        // f32 that follows would land on an odd address.
        w.put_vector("a", &[1.0]).unwrap();
        w.put(&[1u8, 2, 3]).unwrap();
        w.put_vector("b", &[2.0, 3.0]).unwrap();
        w.finish(header("r", &[], &spec(), Precision::Q8).unwrap()).unwrap();

        let cache = Mapped::open(&path, "r", &[], &spec(), Precision::Q8).unwrap().unwrap();
        for name in ["a", "b"] {
            let Some(Entry::F32 { data, .. }) = cache.entries.get(name) else { panic!() };
            assert_eq!(data.0 % ALIGN, 0, "`{name}` at {} is not aligned", data.0);
        }
        assert_eq!(cache.vector("b").unwrap(), vec![2.0, 3.0]);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn cache_paths_are_derived_from_the_repo_id() {
        std::env::set_var("KVAD_QUANT_CACHE", "/tmp/qc");
        let p = path_for("Qwen/Qwen2.5-0.5B-Instruct", Precision::Q8);
        assert_eq!(p, PathBuf::from("/tmp/qc/Qwen--Qwen2.5-0.5B-Instruct.q8.nq"));
        std::env::remove_var("KVAD_QUANT_CACHE");
    }

    #[test]
    fn a_directory_is_filed_under_its_whole_path() {
        let name = |repo: &str| file_name_for(repo, Precision::Q8);
        // Readable, and a legal file name however deep the directory is.
        let deep = format!("/{}/readme", "a-long-directory-name/".repeat(30));
        assert!(name(&deep).starts_with("local--readme-") && name(&deep).len() < 64, "{}", name(&deep));
        // Flattening the slashes would have made one file of these two.
        assert_ne!(name("/out/a/b"), name("/out/a--b"));
        assert_ne!(name("/one/readme"), name("/two/readme"));
        assert_eq!(name("/one/readme"), name("/one/readme"));
        // Pinned, because a hash that changed would orphan every cache file.
        assert_eq!(name("/out/readme"), "local--readme-3ee92617cc0f7705.q8.nq");
    }

    /// `kvad cache` lists models by name, and a directory's name cannot be
    /// recovered from its file name. It is read from the header instead.
    #[test]
    fn the_listing_reads_the_name_from_the_header() {
        let path = tmp("named");
        let w = Writer::create(&path).unwrap();
        w.finish(header("/home/me/out/my--model", &[], &spec(), Precision::Q8).unwrap()).unwrap();
        assert_eq!(repo_of(&path).as_deref(), Some("/home/me/out/my--model"));

        std::fs::write(&path, b"not a cache file at all").unwrap();
        assert_eq!(repo_of(&path), None);
        std::fs::write(&path, b"short").unwrap();
        assert_eq!(repo_of(&path), None);
        std::fs::remove_file(&path).unwrap();
    }
}
