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

use crate::model::{gpt2, llama, Arch, Spec, Transformer};
use crate::quant::{Parts, Precision, Weight, BLOCK};
use crate::tensor::Tensor;
use crate::weights::{Checkpoint, ModelFiles};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const MAGIC: &[u8; 8] = b"NANOQ\x00\x00\x01";
/// Where the first array may start. Also the alignment every array gets.
const ALIGN: usize = 64;

/// Bump this when the bytes stop meaning what they used to.
///
/// Version 1 is the first format. If the quantiser changes — a different
/// scale rule, a different packing order, a different [`BLOCK`] — this must
/// change too, or old files will be read as if they were new ones.
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
    /// A 1-D tensor: norm weights, biases. Never quantised — they are a
    /// rounding error in both size and cost.
    fn vector(&self, name: &str) -> Res<Vec<f32>>;
    fn try_vector(&self, name: &str) -> Option<Vec<f32>>;
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
}

impl<'a> Live<'a> {
    pub fn new(ckpt: &'a Checkpoint, precision: Precision) -> Self {
        Live { ckpt, precision, out: RefCell::new(None), failure: RefCell::new(None) }
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
    fn record(&self, write: impl FnOnce(&mut Writer) -> Res<()>) {
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
        let w = self.quantized(self.ckpt.get(name)?);
        self.record(|out| out.put_weight(name, &w));
        Ok(w)
    }

    fn try_matrix(&self, name: &str) -> Option<Weight> {
        let w = self.quantized(self.ckpt.try_get(name)?);
        self.record(|out| out.put_weight(name, &w));
        Some(w)
    }

    fn matrix_t(&self, name: &str) -> Res<Weight> {
        let w = self.quantized(self.ckpt.get(name)?.transposed());
        self.record(|out| out.put_weight(name, &w));
        Ok(w)
    }

    fn vector(&self, name: &str) -> Res<Vec<f32>> {
        let v = self.ckpt.get_flat(name)?;
        self.record(|out| out.put_vector(name, &v));
        Ok(v)
    }

    fn try_vector(&self, name: &str) -> Option<Vec<f32>> {
        let v = self.ckpt.try_get_flat(name)?;
        self.record(|out| out.put_vector(name, &v));
        Some(v)
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
/// length. Files that are not symlinks (a hand-placed checkpoint) fall back to
/// the size, which is the best that can be had without hashing gigabytes.
fn identify(path: &Path) -> Res<String> {
    if let Ok(target) = std::fs::read_link(path) {
        if let Some(name) = target.file_name().and_then(|n| n.to_str()) {
            return Ok(format!("sha256:{name}"));
        }
    }
    Ok(format!("bytes:{}", std::fs::metadata(path)?.len()))
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

/// The cache file for one repo at one precision.
pub fn path_for(repo: &str, precision: Precision) -> PathBuf {
    cache_root().join(format!("{}.{precision}.nq", repo.replace('/', "--")))
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
            Some((path, repo.replacen("--", "/", 1), precision.to_string(), bytes))
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

fn build(src: &dyn Source, spec: &Spec) -> Res<Box<dyn Transformer>> {
    Ok(match spec.arch {
        Arch::Gpt2 => Box::new(gpt2::Model::load(src, spec.clone())?),
        Arch::Llama => Box::new(llama::Model::load(src, spec.clone())?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nq-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn spec() -> Spec {
        Spec {
            arch: Arch::Llama,
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
}
