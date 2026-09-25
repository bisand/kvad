//! Candle's quantised weights, written to disk once and read back thereafter.
//!
//! # Why this is not [`kvad::qcache`]
//!
//! It is the same idea and the same file format, and deliberately not the same
//! bytes. The CPU engine's cache holds *its* blocks — `i8` codes and `f32`
//! scales in the layout `quant.rs` reads — and this one holds GGML's, because
//! that is what a `QTensor` is made of and what candle's kernels want. Two
//! quantisers, two payloads, one container: [`kvad::qcache::Container`] and
//! [`kvad::qcache::Writer`] are shared, so `kvad cache` lists both kinds and
//! `kvad cache <repo>` forgets both.
//!
//! # What is cached, and what is not
//!
//! Only the quantised matrices. At `--quant none` the weights arrive in their
//! final form and a cache of them would be a second copy of the checkpoint,
//! which is the same reason [`kvad::qcache::load`] stores nothing at f32.
//!
//! So the checkpoint is still opened on every load, for the norms and the
//! biases and — in DeepSeek — the two dense halves of `kv_b_proj`. That sounds
//! like a concession and is in fact this cache's best property. The CPU engine
//! had to stamp the checkpoint's whole tensor list into its files, because a
//! mapped cache never reopens a checkpoint and so cannot tell a weight it was
//! written without from a weight the model does not have: the wrong answer is
//! silence. Here the checkpoint is right there. A name the file does not hold
//! is quantised from the source on the spot, correctly, and the file is thrown
//! away at the end of the load so the next one writes a complete one.
//!
//! That is also why the unread-tensor guard in [`crate::common::unread`] needs
//! no help from this module. It subtracts what the loader asked for from what
//! the *checkpoint* holds, exactly as before; a cache hit still records the
//! name it answered, so a weight served from disk counts as read and a weight
//! nobody wanted still counts as missed.

use crate::common::ggml_name;
use crate::uncached::{self, Pages};
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::Device;
use kvad::model::Spec;
use kvad::qcache::{Container, Writer};
use kvad::serde_json;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Bump this when the bytes stop meaning what they used to.
///
/// Candle's quantiser decides these bytes, so an upgrade that changes a block
/// layout changes them — and so does a loader here that starts quantising a
/// matrix it used to keep dense, since the name it writes would be the same
/// and the shape behind it would not.
///
/// The CPU engine's constant carries a third duty this one does not: there,
/// forgetting to bump it after teaching a loader to read one more weight was a
/// silent wrong answer, which is why its files stamp in the checkpoint's whole
/// tensor list. A miss here is answered from the checkpoint, so forgetting
/// costs one slow load and nothing else.
pub const VERSION: u32 = 1;

/// One array's location in the file: byte offset, byte count.
type Span = (usize, usize);

/// What `kvad cache` prints in its QUANT column, and what keeps this backend's
/// files from colliding with the CPU engine's in the same directory.
pub fn tag(quant: GgmlDType) -> String {
    format!("gpu-{}", ggml_name(quant))
}

/// A cache file that is valid for this checkpoint, open.
///
/// Read with [`uncached`], not through [`Container`]'s mapping, which is
/// dropped once the header has been checked. The blocks go to the device and
/// the file's pages would stay behind in the page cache: a second copy of the
/// model, which Qwen-Image at q8 (29.5 GB) cannot have on a 48 GB machine.
/// Mapped, its load compressed 36 GB of the weights it had already loaded,
/// and the first image took 7.8 s to encode and 11.4 s for its first step.
///
/// Read this way nothing is compressed, and it is faster besides: `pread`
/// in whole blobs runs at about 6 GB/s where page faults on a mapping ran at
/// 0.8. From a file not in the page cache, Qwen-Image loads in 5.1 s rather
/// than 71, and Qwen3-14B in 3.2 s rather than 21.2; from one that is, 4.8 s
/// rather than 12.9, and 3.3 rather than 4.2. The price is the blob being
/// uploaded, which is counted on the host until it is on the device where a
/// mapped page was not: the peak is the largest blob above what the model
/// holds once loaded, 0.84 GB for Qwen3-14B, whose token table is 826 MB
/// ([`uncached::Pages`] says why no more than that).
struct Cache {
    file: File,
    bytes: u64,
    /// Name to where its blocks are and what shape they make.
    entries: HashMap<String, (Span, (usize, usize))>,
}

/// Where a quantised weight comes from, and where it goes.
///
/// Three states in one type rather than an enum, because two of them overlap:
/// a vault that is reading may still have to quantise something the file does
/// not hold, and one that is writing is quantising everything. What decides
/// each call is whether `read` has the name and whether `write` is still open.
pub struct Vault {
    /// The file, when this load is allowed one at all.
    path: Option<PathBuf>,
    /// The header this build would write, kept from the comparison at open so
    /// that [`Vault::finish`] does not have to identify the files again.
    header: serde_json::Value,
    tag: String,
    quant: Option<GgmlDType>,
    read: Option<Cache>,
    write: RefCell<Option<Writer>>,
    /// Why recording stopped, if it did.
    failure: RefCell<Option<String>>,
    /// Names this build wanted that the file did not hold.
    missed: RefCell<Vec<String>>,
}

impl Vault {
    /// No cache: quantise from the checkpoint, keep nothing.
    pub fn off() -> Self {
        Vault {
            path: None,
            header: serde_json::Value::Null,
            tag: String::new(),
            quant: None,
            read: None,
            write: RefCell::new(None),
            failure: RefCell::new(None),
            missed: RefCell::new(Vec::new()),
        }
    }

    /// Open the cache for this model at this quantisation, or arrange to
    /// write one.
    ///
    /// Never fails. Every way this can go wrong — no directory, a file from an
    /// older format, a re-downloaded checkpoint — ends with the load going to
    /// the checkpoint, which is where it went before this module existed.
    pub fn open(
        repo: &str,
        paths: &[PathBuf],
        spec: &Spec,
        quant: Option<GgmlDType>,
        progress: &mut dyn FnMut(&str),
    ) -> Self {
        let shape = serde_json::json!({
            "arch": spec.arch.to_string(),
            "n_layer": spec.n_layer,
            "n_embd": spec.n_embd,
            "vocab_size": spec.vocab_size,
            "tie_embeddings": spec.tie_embeddings,
        });
        Vault::open_as(repo, paths, shape, quant, progress)
    }

    /// As [`Vault::open`], for weights that are not a language model and so
    /// have no [`Spec`] to be known by: one component of an image pipeline,
    /// described by whatever says what shape it is.
    ///
    /// `repo` names the file, so a pipeline with two quantised components
    /// gives each its own: `Qwen/Qwen-Image/transformer`.
    pub fn open_as(
        repo: &str,
        paths: &[PathBuf],
        shape: serde_json::Value,
        quant: Option<GgmlDType>,
        progress: &mut dyn FnMut(&str),
    ) -> Self {
        // Nothing to store at bf16: the checkpoint already is the weights.
        let Some(gd) = quant else { return Vault::off() };
        if !kvad::qcache::enabled() {
            return Vault::off();
        }
        let tag = tag(gd);
        let header = match header(repo, paths, shape, &tag) {
            Ok(h) => h,
            Err(e) => {
                progress(&format!("not caching: {e}"));
                return Vault::off();
            }
        };
        let path = kvad::qcache::path_for_tag(repo, &tag);

        let mut vault = Vault { quant, tag, header, ..Vault::off() };
        let stale = match open_valid(&path, &vault.header) {
            Ok(Some(cache)) => {
                progress(&format!(
                    "reading cached {} weights ({} MB)",
                    vault.tag,
                    cache.bytes / 1_000_000
                ));
                vault.read = Some(cache);
                vault.path = Some(path);
                return vault;
            }
            Ok(None) => None,
            Err(e) => Some(e.to_string()),
        };

        match Writer::create(&path) {
            Ok(w) => {
                let why = match &stale {
                    Some(e) => {
                        progress(&format!("stale {} cache: {e}", vault.tag));
                        "rebuilding"
                    }
                    None => "first load",
                };
                progress(&format!("quantising to {} ({why})", vault.tag));
                *vault.write.borrow_mut() = Some(w);
                vault.path = Some(path);
            }
            // A read-only or full cache directory is not a reason to fail.
            Err(e) => progress(&format!("not caching: {e}")),
        }
        vault
    }

    /// `name`, already quantised, if the file has it in the shape asked for.
    ///
    /// A shape that disagrees is treated as a miss rather than an error: the
    /// file is describing a different model under the same name, and the
    /// checkpoint is the one to believe.
    pub(crate) fn get(
        &self,
        name: &str,
        shape: (usize, usize),
        device: &Device,
    ) -> Option<QTensor> {
        let gd = self.quant?;
        let blocks = self.blocks(name, shape)?;
        let storage = QStorage::from_data(Cow::Borrowed(&blocks), device, gd);
        match storage.and_then(|s| QTensor::new(s, shape)) {
            Ok(q) => Some(q),
            // A length candle will not take: the file is damaged, and the
            // load carries on without it.
            Err(_) => {
                self.missed.borrow_mut().push(name.to_string());
                None
            }
        }
    }

    /// `name`'s blocks as the file holds them, for a caller that wants the
    /// bytes rather than a `QTensor`: `mpp`'s kernel reads Q8_0 itself.
    pub(crate) fn blocks(&self, name: &str, shape: (usize, usize)) -> Option<Pages> {
        self.quant?;
        let cache = self.read.as_ref()?;
        let miss = |v: &Self| {
            v.missed.borrow_mut().push(name.to_string());
            None
        };
        let Some(&(span, stored)) = cache.entries.get(name) else { return miss(self) };
        if stored != shape {
            return miss(self);
        }
        match uncached::read(&cache.file, span.0 as u64, span.1) {
            Ok(b) => Some(b),
            // Offsets past the end, or a read that failed: the file is
            // damaged, and the load carries on without it.
            Err(_) => miss(self),
        }
    }

    /// Tee one quantised weight to the cache file.
    ///
    /// A write failure — a full disk, a directory that went away — abandons
    /// the cache and is otherwise ignored. It must never change what this load
    /// returns: the cache is an optimisation, and an optimisation that can
    /// drop a weight matrix is a correctness bug.
    pub(crate) fn put(&self, name: &str, shape: (usize, usize), blocks: &[u8]) {
        let Some(gd) = self.quant else { return };
        let mut out = self.write.borrow_mut();
        let Some(writer) = out.as_mut() else { return };
        // Tied weights: stored once, named twice.
        if writer.has(name) {
            return;
        }
        let wrote = writer.put_bytes(blocks).map(|data| {
            writer.set(
                name,
                serde_json::json!({
                    "kind": ggml_name(gd), "shape": [shape.0, shape.1], "data": data,
                }),
            )
        });
        if let Err(e) = wrote {
            *self.failure.borrow_mut() = Some(e.to_string());
            *out = None; // drops the Writer, which removes the partial file
        }
    }

    /// Close the file, or take it away.
    ///
    /// Call it once the architecture has finished loading, which is the first
    /// moment either question can be answered: what was written is only whole
    /// now, and what was missing from a cached file is only known now.
    pub fn finish(&mut self, progress: &mut dyn FnMut(&str)) {
        let missed = std::mem::take(&mut *self.missed.borrow_mut());
        if !missed.is_empty() {
            // A cache behind the code, which is a stale cache like any other —
            // the load was served correctly from the checkpoint, and the file
            // goes so that the next load writes a complete one rather than
            // paying for these same tensors forever.
            let what = kvad::weights::collapsed(&missed).join(", ");
            progress(&format!("stale {} cache: written without {what}", self.tag));
            match self.path.as_deref().map(std::fs::remove_file) {
                Some(Ok(())) => progress("dropped it; the next load rebuilds it"),
                Some(Err(e)) => progress(&format!("could not drop it: {e}")),
                None => {}
            }
            return;
        }

        let Some(writer) = self.write.borrow_mut().take() else {
            if let Some(e) = self.failure.borrow().as_ref() {
                progress(&format!("not caching: {e}"));
            }
            return;
        };
        match writer.finish(self.header.clone()) {
            Ok(bytes) => progress(&format!("cached for next time ({} MB)", bytes / 1_000_000)),
            // Failing to write the cache is not a reason to fail the load.
            Err(e) => progress(&format!("not caching: {e}")),
        }
    }
}

/// Open `path` if it is there and is still about this checkpoint.
///
/// `Ok(None)` means there is no cache yet. `Err` means there is one and it
/// cannot be trusted, which is worth saying out loud before rebuilding.
fn open_valid(path: &Path, want: &serde_json::Value) -> Res<Option<Cache>> {
    let Some(file) = Container::open(path)? else {
        return Ok(None);
    };
    {
        let json = file.header();
        for key in ["format", "version", "precision", "repo", "spec", "sources"] {
            if json.get(key) != want.get(key) {
                let was = json.get(key).unwrap_or(&serde_json::Value::Null).to_string();
                let was = brief(&was);
                return Err(format!("`{key}` changed since it was written ({was})").into());
            }
        }
    }

    let mut entries = HashMap::new();
    {
        let tensors =
            file.header().get("tensors").and_then(|t| t.as_object()).ok_or("no tensors")?;
        entries.reserve(tensors.len());
        for (name, entry) in tensors {
            entries.insert(name.clone(), parse_entry(entry)?);
        }
    }
    let bytes = file.bytes() as u64;
    drop(file);
    Ok(Some(Cache { file: uncached::open(path)?, bytes, entries }))
}

/// One blob, handed to candle as the blocks it already is.
///
/// `from_data` copies: on Metal into a device buffer, which is the trip a
/// weight has to make however it arrived, and on the CPU into a `Vec`, which
/// the CPU engine's own cache, reading its mapping in place, does without.
/// That is the price of speaking candle's vocabulary rather than our own, and
/// it is paid against quantising the matrix from scratch.
fn parse_entry(v: &serde_json::Value) -> Res<(Span, (usize, usize))> {
    let num = |v: Option<&serde_json::Value>| -> Res<usize> {
        v.and_then(serde_json::Value::as_u64).map(|n| n as usize).ok_or_else(|| "bad entry".into())
    };
    let shape = v.get("shape").and_then(|s| s.as_array()).ok_or("entry has no shape")?;
    let data = v.get("data").and_then(|s| s.as_array()).ok_or("entry has no data")?;
    Ok((
        (num(data.first())?, num(data.get(1))?),
        (num(shape.first())?, num(shape.get(1))?),
    ))
}

/// Keep a diagnostic short enough for a status line.
fn brief(s: &str) -> String {
    match s.char_indices().nth(60) {
        None => s.to_string(),
        Some((at, _)) => format!("{}…", &s[..at]),
    }
}

/// Everything that has to match for a cache file to be reusable.
///
/// The same questions [`kvad::qcache`] asks, with `precision` naming this
/// backend rather than a [`kvad::quant::Precision`] — which is what stops a
/// `gpu-q8` file being read as a `q8` one, on top of the file names differing.
fn header(repo: &str, paths: &[PathBuf], shape: serde_json::Value, tag: &str) -> Res<serde_json::Value> {
    Ok(serde_json::json!({
        "format": "nanollm-quant-gpu",
        "version": VERSION,
        "precision": tag,
        "repo": repo,
        "spec": shape,
        "sources": kvad::qcache::sources(paths)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::sync::{Mutex, MutexGuard};

    /// `KVAD_QUANT_CACHE` is one variable for the whole process, so the tests
    /// that redirect it take turns.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// A cache directory of this test's own, gone again when the test ends.
    struct Dir {
        path: PathBuf,
        _turn: MutexGuard<'static, ()>,
    }

    impl Dir {
        fn new(tag: &str) -> Self {
            // A poisoned lock means another test panicked, not that this
            // directory is unusable.
            let turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let path =
                std::env::temp_dir().join(format!("kvad-gpu-nq-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            std::env::set_var("KVAD_QUANT_CACHE", &path);
            Dir { path, _turn: turn }
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            std::env::remove_var("KVAD_QUANT_CACHE");
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// One checkpoint per architecture that reaches this cache by a different
    /// route: Llama stores `[out, in]` and GPT-2 stores its transpose, so a
    /// round trip that only ever saw one of them would not be testing the part
    /// most likely to be wrong.
    fn fixtures(tag: &str) -> Vec<(&'static str, Spec, PathBuf)> {
        let llama = crate::model::tests::tiny_spec();
        let gpt2 = crate::gpt2::tests::tiny_spec(true);
        vec![
            ("llama", llama.clone(), crate::model::tests::write_tensors(&llama, false, &[], tag)),
            ("gpt2", gpt2.clone(), crate::gpt2::tests::write_tensors(&gpt2, &[], tag)),
        ]
    }

    /// One load through the dispatcher, with whatever the cache directory now
    /// holds. Returns the logits and everything the load said.
    fn load(
        repo: &str,
        path: &Path,
        spec: &Spec,
        quant: Option<GgmlDType>,
        tokens: &[u32],
    ) -> (Vec<f32>, String) {
        let mut log: Vec<String> = Vec::new();
        let files = [path.to_path_buf()];
        let mut session = crate::model::session(
            repo,
            &files,
            spec,
            DType::F32,
            quant,
            Device::Cpu,
            &mut |m| log.push(m.to_string()),
        )
        .unwrap();
        (session.forward(tokens).unwrap(), log.join(" | "))
    }

    /// Take one tensor back out of a cache file: exactly the file a build that
    /// never read that tensor would have written.
    ///
    /// The header may be rewritten in place at any length, because what fixes
    /// its position is the offset in the last eight bytes rather than the end
    /// of the file. The blocks stay where they are with nothing pointing at
    /// them, which is also what a real such file would look like.
    fn forget_from_cache(path: &Path, drop: &str) {
        let bytes = std::fs::read(path).unwrap();
        let n = bytes.len();
        let at = u64::from_le_bytes(bytes[n - 8..].try_into().unwrap()) as usize;
        let mut json: serde_json::Value = serde_json::from_slice(&bytes[at..n - 8]).unwrap();
        let gone = json["tensors"].as_object_mut().unwrap().remove(drop);
        assert!(gone.is_some(), "`{drop}` was not in the cache to begin with");

        let mut out = bytes[..at].to_vec();
        out.extend_from_slice(&serde_json::to_vec(&json).unwrap());
        out.extend_from_slice(&(at as u64).to_le_bytes());
        std::fs::write(path, out).unwrap();
    }

    /// A cache is worth having only if the second load is the first load.
    ///
    /// Bit-for-bit, not close: both runs quantise the same numbers with the
    /// same quantiser, so anything but equality means the blocks came back
    /// meaning something else — a shape read back transposed, a span off by an
    /// alignment boundary, a dtype the header did not pin down.
    ///
    /// And the file has to be doing the work. A miss is answered from the
    /// checkpoint, correctly and silently, so a cache that matched nothing at
    /// all would pass the equality check and fail at its job: hence the
    /// assertions that the second load said `reading cached`, said nothing about
    /// staleness, and left the file where it was.
    #[test]
    fn a_cached_model_answers_exactly_what_the_quantised_one_did() {
        let _dir = Dir::new("roundtrip");
        let tokens = [1u32, 2, 3, 4];
        for (arch, spec, path) in fixtures("roundtrip") {
            let repo = format!("test/gpu-cache-{arch}");
            let file = kvad::qcache::path_for_tag(&repo, &tag(GgmlDType::Q8_0));

            let (first, log) = load(&repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
            assert!(log.contains("quantising to gpu-q8 (first load)"), "{arch}: {log}");
            assert!(log.contains("cached for next time"), "{arch}: {log}");
            assert!(file.exists(), "{arch}: nothing was written");

            let (second, log) = load(&repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
            assert!(log.contains("reading cached gpu-q8 weights"), "{arch}: {log}");
            assert!(!log.contains("stale"), "{arch}: a cache this build wrote is not stale: {log}");
            assert!(file.exists(), "{arch}: the file went away: {log}");
            assert_eq!(second, first, "{arch}: the cached model is a different model");

            std::fs::remove_file(&path).unwrap();
        }
    }

    /// A cache missing something this build wants must not be half-used.
    ///
    /// The CPU engine answers this question with a stamp of the checkpoint's
    /// whole tensor list, because a mapped cache there never reopens a
    /// checkpoint and so cannot tell a weight it was written without from a
    /// weight the model does not have. Here the checkpoint is still open, so
    /// the miss is simply answered from it — and the file, now known to be
    /// behind the code, goes.
    #[test]
    fn a_cache_written_without_a_weight_is_answered_and_dropped() {
        let _dir = Dir::new("incomplete");
        let tokens = [1u32, 2, 3, 4];
        let spec = crate::model::tests::tiny_spec();
        let path = crate::model::tests::write_tensors(&spec, false, &[], "incomplete");
        let repo = "test/gpu-cache-incomplete";
        let file = kvad::qcache::path_for_tag(repo, &tag(GgmlDType::Q8_0));

        let (whole, _) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
        forget_from_cache(&file, "model.layers.0.self_attn.q_proj.weight");

        let (served, log) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
        assert_eq!(served, whole, "the missing weight must come from the checkpoint");
        assert!(log.contains("stale"), "{log}");
        assert!(log.contains("q_proj"), "it should say which weight: {log}");
        assert!(!file.exists(), "a file known to be behind the code must not survive: {log}");

        // And the drop is a drop: the next load writes a whole one, and the
        // one after that is quiet again.
        let (_, log) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
        assert!(log.contains("first load"), "{log}");
        let (last, log) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
        assert_eq!(last, whole);
        assert!(log.contains("reading cached") && !log.contains("stale"), "{log}");

        std::fs::remove_file(&path).unwrap();
    }

    /// A checkpoint that changed underneath a cache file invalidates it.
    ///
    /// The weights here are the same shapes and different numbers, which is
    /// what a retrain looks like from the outside: nothing about the size of
    /// the file says anything happened, and serving the old blocks would run
    /// the previous model at full speed under the new one's name.
    #[test]
    fn a_cache_for_a_checkpoint_that_moved_on_is_not_served() {
        let _dir = Dir::new("moved");
        let tokens = [1u32, 2, 3, 4];
        let spec = crate::model::tests::tiny_spec();
        let repo = "test/gpu-cache-moved";

        let path = crate::model::tests::write_tensors(&spec, false, &[], "moved");
        let (before, _) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);

        // `identify` reads a size and a modification time for a file nobody
        // symlinked, and a same-second rewrite is the case it has to catch.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let path = crate::model::tests::write_tensors(&spec, false, &[], "moved");
        let (after, log) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
        assert!(log.contains("stale") && log.contains("sources"), "{log}");
        assert!(log.contains("rebuilding"), "{log}");
        assert_ne!(after, before, "the new checkpoint is a different model");

        // And the rebuilt file is about the new one.
        let (again, log) = load(repo, &path, &spec, Some(GgmlDType::Q8_0), &tokens);
        assert!(log.contains("reading cached") && !log.contains("stale"), "{log}");
        assert_eq!(again, after);

        std::fs::remove_file(&path).unwrap();
    }

    /// Two engines, two quantisers, one directory.
    ///
    /// The CPU engine's `q8` and this backend's are both eight bits a weight
    /// and are not the same bytes: its blocks carry an `f32` scale per 32
    /// weights in its own order, GGML's carry an `f16`. Handing one to the
    /// other would be a model made of noise, so the file names have to differ
    /// before anything else does.
    #[test]
    fn the_two_engines_do_not_share_a_file() {
        let ours = kvad::qcache::path_for_tag("test/collide", &tag(GgmlDType::Q8_0));
        let theirs = kvad::qcache::path_for("test/collide", kvad::quant::Precision::Q8);
        assert_ne!(ours, theirs);
        for q in [GgmlDType::Q8_0, GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q6K] {
            let path = kvad::qcache::path_for_tag("test/collide", &tag(q));
            assert!(path.to_str().unwrap().contains(".gpu-"), "{path:?}");
            assert_ne!(path, theirs);
        }
    }

    /// There is nothing to cache at bf16: the checkpoint already is the
    /// weights, and a copy of them would be a second copy of the checkpoint.
    #[test]
    fn a_dense_load_writes_nothing() {
        let dir = Dir::new("dense");
        let spec = crate::model::tests::tiny_spec();
        let path = crate::model::tests::write_tensors(&spec, false, &[], "dense");
        let (_, log) = load("test/gpu-cache-dense", &path, &spec, None, &[1, 2, 3]);
        assert!(log.is_empty(), "a dense load has nothing to say about the cache: {log}");
        assert_eq!(std::fs::read_dir(&dir.path).unwrap().count(), 0, "it wrote something");
        std::fs::remove_file(&path).unwrap();
    }
}
