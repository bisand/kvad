//! Fetching models from the HuggingFace Hub and reading their weights.
//!
//! # safetensors
//!
//! The format is deliberately boring, which is the point — the older `.bin`
//! format was a pickled Python object graph, i.e. arbitrary code execution on
//! load. A safetensors file is:
//!
//! ```text
//! [8 bytes: header length, little-endian u64]
//! [header: JSON mapping tensor name -> {dtype, shape, byte offsets}]
//! [the raw tensor bytes, back to back]
//! ```
//!
//! So loading is: parse a small JSON blob, then slice into a memory map.
//!
//! Models above a few GB are split into shards, with a
//! `model.safetensors.index.json` mapping each tensor name to the file holding
//! it. [`Checkpoint`] hides that: open all the shards, build one name index,
//! and look tensors up without caring where they live.
//!
//! # Models that are not on the Hub
//!
//! Wherever a repo id is accepted, a directory is too: if the name given is a
//! directory that exists, it is read in place and nothing is fetched. That is
//! how a model trained by this repository's own `nervus` gets here, and the
//! rule — an existing directory wins over a repo of the same name — is the one
//! `transformers` uses, so nobody has to learn a second one. The directory
//! holds what a Hub repo would: `config.json`, `tokenizer.json`, and either
//! `model.safetensors` or a shard index.
//!
//! A directory is a poor name, though. `kvad train --name shakespeare` puts
//! its model in a home of its own — `$XDG_DATA_HOME/kvad/models/shakespeare`
//! — and from then on the bare word `shakespeare` means that model from any
//! working directory. So a model name is resolved in three steps, in this
//! order:
//!
//! 1. a directory of that name that exists, relative to where you are;
//! 2. a model of that name trained here;
//! 3. a repo id on the Hub.
//!
//! The three cannot be confused by accident. A Hub repo id always contains a
//! slash (`owner/name`) and a trained model's name never may, because it has
//! to be one path component — which is also what keeps `../../etc` from being
//! a model name. And step 1 comes first so that the `transformers` rule still
//! holds: a directory that is there wins.

use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub struct ModelFiles {
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    /// Holds the chat template, when the model has one.
    pub tokenizer_config: Option<PathBuf>,
    pub generation_config: Option<PathBuf>,
}

/// Where everything this machine cannot download again is kept.
///
/// `$XDG_DATA_HOME/kvad`, or `~/.local/share/kvad`. Data rather than cache,
/// because a model you trained has exactly one copy, and so does the server's
/// database. Settings live next door under `XDG_CONFIG_HOME`; see
/// [`crate::hub::config_dir`].
pub fn data_dir() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join(".local/share"));
    base.join("kvad")
}

/// Where models trained on this machine live: `models` under [`data_dir`].
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

fn home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

/// Whether `name` is usable as the name of a trained model: exactly one
/// ordinary path component.
///
/// This is the check that makes `models_dir().join(name)` safe to build. A
/// name with a slash in it, an absolute path, `.` or `..` would all escape
/// the models directory, and `..` is the one somebody would try.
pub fn is_model_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(std::path::Component::Normal(_))) && parts.next().is_none()
}

/// Where a model trained here by that name lives, if one does.
///
/// The path comes back resolved, as [`local_dir`]'s does, so that the two
/// answers can be compared and neither depends on how the models home was
/// reached.
pub fn trained_dir(name: &str) -> Option<PathBuf> {
    let dir = models_dir().join(name);
    (is_model_name(name) && dir.is_dir()).then(|| std::fs::canonicalize(&dir).unwrap_or(dir))
}

/// The directory a model name refers to on this machine, if any: a directory
/// that exists as typed, else a model trained here under that name.
pub fn local_dir(model: &str) -> Option<PathBuf> {
    let typed = Path::new(model);
    if typed.is_dir() {
        return Some(std::fs::canonicalize(typed).unwrap_or_else(|_| typed.to_path_buf()));
    }
    trained_dir(model)
}

/// Whether `model` names a directory on this machine rather than a Hub repo.
pub fn is_local(model: &str) -> bool {
    local_dir(model).is_some()
}

/// Whether `model` was written the way paths are and repo ids never are.
pub fn looks_like_path(model: &str) -> bool {
    model.starts_with(['.', '/', '~'])
}

/// The name a model is known by once loaded: a repo id as it is, a model
/// trained here by its bare name, any other directory by its absolute path.
///
/// Anything keyed on the name — the quantised-weight cache above all — must
/// not think `out/readme`, `./out/readme` and the same words typed from
/// another working directory are three models, or worse, one. The same goes
/// for `shakespeare` and the long path it stands for, which is why a trained
/// model resolves *back* to its name here rather than forward to its path.
pub fn model_id(model: &str) -> String {
    match local_dir(model) {
        Some(dir) => trained_name(&dir).unwrap_or_else(|| dir.display().to_string()),
        None => model.to_string(),
    }
}

/// The name of a trained model, given its directory — the reverse of
/// [`trained_dir`], and `None` for a directory that is not in the models home.
pub fn trained_name(dir: &Path) -> Option<String> {
    let root = std::fs::canonicalize(models_dir()).ok()?;
    let dir = std::fs::canonicalize(dir).ok()?;
    (dir.parent()? == root).then(|| dir.file_name()?.to_str().map(str::to_string))?
}

impl ModelFiles {
    /// The files of a model that is already in `dir`.
    pub fn from_dir(dir: &Path) -> Res<Self> {
        let need = |name: &str| -> Res<PathBuf> {
            let path = dir.join(name);
            match path.is_file() {
                true => Ok(path),
                false => Err(format!("{} has no {name}", dir.display()).into()),
            }
        };
        let maybe = |name: &str| Some(dir.join(name)).filter(|p| p.is_file());

        let weights = match maybe("model.safetensors") {
            Some(single) => vec![single],
            None => {
                let index = maybe("model.safetensors.index.json").ok_or_else(|| {
                    format!("{} has no model.safetensors, and no shard index either", dir.display())
                })?;
                shard_names(&index)?.iter().map(|s| need(s)).collect::<Res<Vec<_>>>()?
            }
        };

        Ok(ModelFiles {
            weights,
            tokenizer: need("tokenizer.json")?,
            config: need("config.json")?,
            tokenizer_config: maybe("tokenizer_config.json"),
            generation_config: maybe("generation_config.json"),
        })
    }
}

/// The distinct files a shard index points at.
fn shard_names(index: &Path) -> Res<Vec<String>> {
    let json = read_json(index)?;
    let map = json.get("weight_map").and_then(|m| m.as_object()).ok_or("shard index has no weight_map")?;

    // Many tensor names point at the same handful of files.
    let mut shards: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
    shards.sort();
    shards.dedup();
    Ok(shards)
}

/// What a fetch is doing, in numbers rather than words.
///
/// This exists *beside* the line of text a fetch already reports, not instead
/// of it. "fetching model.safetensors" is what a person reads; a progress bar
/// needs to know that 412 of 990 MB have arrived, and a bar is the only
/// honest way to show a download that takes two minutes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fetch {
    /// Nothing will be downloaded: the model is already a directory here.
    Local,
    /// The checkpoint is split, and this many shards are about to be fetched.
    Shards(usize),
    /// Bytes moved so far for one file. `total` is 0 when the size is not
    /// known — a file answered from the cache reports that it is done and
    /// never says how big it was.
    Download { file: String, bytes: u64, total: u64 },
    /// A file is on disk, whether it was downloaded now or cached earlier.
    Fetched { file: String },
}

/// Where [`Fetch`] events go.
///
/// Two things make this unlike the `&mut dyn FnMut(&str)` beside it, and both
/// come from `hf-hub`: it takes `&self`, and it is `Send + Sync`. The
/// download runs on tokio tasks of its own while this thread sits blocked
/// inside the request, so the handler is called from a thread that is not
/// this one and cannot be handed a `&mut` to anything on it.
///
/// A watcher nobody is listening to drops every event, so callers that do not
/// want progress pass [`Watcher::none`] rather than an `Option`.
#[derive(Clone, Default)]
pub struct Watcher(Option<Arc<dyn Fn(Fetch) + Send + Sync>>);

impl Watcher {
    pub fn new(f: impl Fn(Fetch) + Send + Sync + 'static) -> Self {
        Watcher(Some(Arc::new(f)))
    }

    /// A watcher that discards everything. What [`fetch_with`] uses.
    pub fn none() -> Self {
        Watcher(None)
    }

    pub fn is_listening(&self) -> bool {
        self.0.is_some()
    }

    fn emit(&self, event: Fetch) {
        if let Some(f) = &self.0 {
            f(event);
        }
    }
}

impl std::fmt::Debug for Watcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Watcher").field(&self.is_listening()).finish()
    }
}

/// Adapts one `hf-hub` download to a [`Watcher`].
///
/// Every event is reported under the filename this relay was built for rather
/// than the one `hf-hub` names, so that a caller drawing a bar sees the file
/// it asked for whichever path the download took. One `download_file` call
/// fetches exactly one file, so the two can only ever be the same name; the
/// xet batch path does not carry a name at all.
struct Relay {
    file: String,
    watch: Watcher,
}

impl hf_hub::progress::ProgressHandler for Relay {
    fn on_progress(&self, event: &hf_hub::progress::ProgressEvent) {
        use hf_hub::progress::{DownloadEvent as D, FileStatus, ProgressEvent as P};
        let file = || self.file.clone();
        match event {
            P::Download(D::Progress { files }) => {
                for f in files {
                    self.watch.emit(match f.status {
                        FileStatus::Complete => Fetch::Fetched { file: file() },
                        _ => Fetch::Download {
                            file: file(),
                            bytes: f.bytes_completed,
                            total: f.total_bytes,
                        },
                    });
                }
            }
            P::Download(D::AggregateProgress { bytes_completed, total_bytes, .. }) => {
                self.watch.emit(Fetch::Download {
                    file: file(),
                    bytes: *bytes_completed,
                    total: *total_bytes,
                });
            }
            _ => {}
        }
    }
}

/// Download (or reuse from the local cache) everything needed to run `repo_id`,
/// reporting progress to stderr. A directory is used as it is.
pub fn fetch(repo_id: &str) -> Res<ModelFiles> {
    fetch_with(repo_id, &mut |msg| eprintln!("  {msg}"))
}

/// As [`fetch`], but progress goes to a callback.
///
/// The TUI needs this: anything written straight to stderr lands on top of the
/// rendered frame and corrupts the display.
pub fn fetch_with(repo_id: &str, progress: &mut dyn FnMut(&str)) -> Res<ModelFiles> {
    fetch_watched(repo_id, progress, &Watcher::none())
}

/// As [`fetch_with`], and also reporting [`Fetch`] events to `watch`.
pub fn fetch_watched(
    repo_id: &str,
    progress: &mut dyn FnMut(&str),
    watch: &Watcher,
) -> Res<ModelFiles> {
    if let Some(dir) = local_dir(repo_id) {
        progress(match trained_name(&dir) {
            Some(_) => "a model trained here; nothing to fetch",
            None => "a directory on this machine; nothing to fetch",
        });
        watch.emit(Fetch::Local);
        return ModelFiles::from_dir(&dir);
    }
    // Typed as a path, so meant as one: say the directory is missing, rather
    // than go and ask the Hub for a repo called `./out`.
    if looks_like_path(repo_id) {
        return Err(format!("`{repo_id}` looks like a path, and there is no such directory").into());
    }

    let (owner, name) = repo_id.split_once('/').ok_or_else(|| {
        // No slash, so it cannot be a repo id and was not a trained model
        // either. Say which of the two they might have meant.
        format!(
            "`{repo_id}` is not a model trained here, and a Hub repo id looks like \
             `openai-community/gpt2`. `kvad ls` lists what is on this machine."
        )
    })?;

    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model(owner, name);

    let repo = &repo;
    let progress = std::cell::RefCell::new(progress);
    // `hf-hub` emits nothing at all when no handler is set, so a fetch nobody
    // is watching pays for none of this.
    let handler = |filename: &str| {
        watch
            .is_listening()
            .then(|| hf_hub::progress::Progress::new(Relay { file: filename.to_string(), watch: watch.clone() }))
    };
    let fetched = |filename: &str, path: PathBuf| -> PathBuf {
        watch.emit(Fetch::Fetched { file: filename.to_string() });
        path
    };
    let get = |filename: &str| -> Res<PathBuf> {
        (progress.borrow_mut())(&format!("fetching {filename}"));
        let path = repo
            .download_file()
            .filename(filename.to_string())
            .maybe_progress(handler(filename))
            .send()?;
        Ok(fetched(filename, path))
    };
    let try_get = |filename: &str| -> Option<PathBuf> {
        let path = repo
            .download_file()
            .filename(filename.to_string())
            .maybe_progress(handler(filename))
            .send()
            .ok()?;
        Some(fetched(filename, path))
    };

    // Single file, or a shard index naming several.
    let weights = match try_get("model.safetensors") {
        Some(single) => vec![single],
        None => {
            let shards = shard_names(&get("model.safetensors.index.json")?)?;
            (progress.borrow_mut())(&format!("checkpoint is split across {} shards", shards.len()));
            watch.emit(Fetch::Shards(shards.len()));
            shards.iter().map(|s| get(s)).collect::<Res<Vec<_>>>()?
        }
    };

    Ok(ModelFiles {
        weights,
        tokenizer: get("tokenizer.json")?,
        config: get("config.json")?,
        tokenizer_config: try_get("tokenizer_config.json"),
        generation_config: try_get("generation_config.json"),
    })
}

/// Names a checkpoint may hold that are not weights, so leaving them unread is
/// correct rather than a gap in the implementation.
///
/// Both are constants an engine builds for itself. `rotary_emb.inv_freq` is the
/// RoPE frequency table, which Llama-2-era exports saved as a buffer and every
/// loader here computes from `rope_theta`. GPT-2's `attn.bias` is its causal
/// mask — a lower-triangular block of ones the size of the context window,
/// stored because `torch` had nowhere else to put it.
///
/// The list is exact on purpose. Anything not here that nobody reads is a piece
/// of the model that is not running.
pub fn derived(name: &str) -> bool {
    name.ends_with("rotary_emb.inv_freq")
        || name.ends_with("attn.bias")
        || name.ends_with("attn.masked_bias")
}

/// The same names with every numeric path segment collapsed to `*`, each line
/// carrying a count when it stands for more than one tensor.
///
/// For naming what a loader did not read. Twenty-eight layers means twenty-eight
/// copies of one omission, and a message that lists them all buries the single
/// fact worth reading — DeepSeek-V2-Lite would put sixty-four experts on top of
/// that. Shared by both engines because the complaint is the same on either.
pub fn collapsed(names: &[String]) -> Vec<String> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in names {
        let key: String = name
            .split('.')
            .map(|part| match !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()) {
                true => "*",
                false => part,
            })
            .collect::<Vec<_>>()
            .join(".");
        match counts.get_mut(&key) {
            Some(n) => *n += 1,
            None => {
                counts.insert(key.clone(), 1);
                order.push(key);
            }
        }
    }
    order
        .into_iter()
        .map(|k| match counts[&k] {
            1 => k,
            n => format!("{k}  ({n} tensors)"),
        })
        .collect()
}

/// What a checkpoint holds, and which of it a loader has asked for.
///
/// A weight nobody reads is a piece of the model that is not running, and
/// nothing else notices: a loader asks for what it knows about, and a file has
/// no opinion about the rest. That is how the GPU backend ran a Llama forward
/// pass over a Qwen3 model at full speed, saying nothing. [`Audit::unread`] is
/// the subtraction that ends it — what the file holds, minus what was asked
/// for, minus the buffers in [`derived`] that no engine here reads on purpose.
///
/// It is built from a list of names rather than from a [`Checkpoint`] because
/// there are two places that list can come from, and both need the same
/// arithmetic: the checkpoint itself, when one is open, and the copy of its
/// tensor list stamped into a quantised cache file, when it is not. Without the
/// second the check has a blind spot the size of the cache — see
/// [`crate::qcache::VERSION`].
pub struct Audit {
    have: HashSet<String>,
    seen: RefCell<HashSet<String>>,
}

impl Audit {
    pub fn new(names: impl IntoIterator<Item = String>) -> Self {
        Audit { have: names.into_iter().collect(), seen: RefCell::new(HashSet::new()) }
    }

    /// Every name, sorted — for stamping one of these lists into a file, and
    /// for reading it back out with `jq`.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.have.iter().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// The spelling this list files `name` under, if it has it at all.
    ///
    /// Which spelling won is the whole point: a loader asks for
    /// `layers.0.mlp.down_proj.weight` and the file answers to
    /// `model.layers.0.mlp.down_proj.weight`, and only the latter can be
    /// subtracted from [`Audit::names`].
    fn resolve(&self, name: &str) -> Option<&str> {
        spellings(name).into_iter().find_map(|c| self.have.get(c.as_str()).map(String::as_str))
    }

    /// Note that something asked about `name`.
    ///
    /// A name this list does not hold records nothing, which is right: there is
    /// no such tensor here to leave unread.
    pub fn saw(&self, name: &str) {
        if let Some(key) = self.resolve(name) {
            self.seen.borrow_mut().insert(key.to_string());
        }
    }

    /// The same for a whole subtree at once, whatever prefix the file itself
    /// puts in front of it.
    ///
    /// For naming a part of a model this build knowingly does not run. See
    /// [`crate::qcache::Source::skip_under`].
    pub fn saw_under(&self, prefix: &str) {
        let want = spellings(prefix);
        let under: Vec<String> = self
            .have
            .iter()
            .filter(|n| want.iter().any(|w| n.starts_with(w.as_str())))
            .cloned()
            .collect();
        self.seen.borrow_mut().extend(under);
    }

    /// Everything nothing asked about, derived buffers aside.
    ///
    /// Call it once the architecture has finished loading. Anything here is a
    /// weight that is not running.
    pub fn unread(&self) -> Vec<String> {
        let seen = self.seen.borrow();
        let mut left: Vec<String> = self
            .have
            .iter()
            .filter(|n| !seen.contains(*n) && !derived(n))
            .cloned()
            .collect();
        left.sort();
        left
    }
}

/// The spellings a checkpoint might file `name` under.
///
/// GPT-2 saves `wte.weight` or `transformer.wte.weight` depending on which
/// Python class wrote it, and the Llama family puts `model.` in front of
/// everything but `lm_head.weight`.
fn spellings(name: &str) -> [String; 3] {
    [name.to_string(), format!("transformer.{name}"), format!("model.{name}")]
}

/// One or more safetensors files, presented as a single namespace.
pub struct Checkpoint {
    maps: Vec<memmap2::Mmap>,
    /// tensor name -> which shard holds it
    index: HashMap<String, usize>,
}

impl Checkpoint {
    pub fn open(paths: &[PathBuf]) -> Res<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        for path in paths {
            let file = std::fs::File::open(path)?;
            // SAFETY: we only ever read, and these are read-only cache entries.
            // A concurrent writer truncating the file would be undefined
            // behaviour, which is the standard caveat on every mmap.
            maps.push(unsafe { memmap2::Mmap::map(&file)? });
        }

        // Parse each header once and remember where every tensor lives.
        let mut index = HashMap::new();
        for (i, map) in maps.iter().enumerate() {
            let st = SafeTensors::deserialize(map)?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
        }
        Ok(Checkpoint { maps, index })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// The name this checkpoint actually files `name` under, if it has it.
    ///
    /// Checkpoints disagree about prefixes — GPT-2 saves `wte.weight` or
    /// `transformer.wte.weight` depending on which Python class wrote it — so a
    /// few spellings are tried before giving up.
    pub fn resolve(&self, name: &str) -> Option<&str> {
        spellings(name)
            .into_iter()
            .find_map(|c| self.index.get_key_value(c.as_str()).map(|(k, _)| k.as_str()))
    }

    /// Look a tensor up, converting to `f32`, or `None` if it is not there.
    pub fn try_get(&self, name: &str) -> Option<Tensor> {
        self.read(name).ok().flatten()
    }

    pub fn get(&self, name: &str) -> Res<Tensor> {
        match self.read(name)? {
            Some(t) => Ok(t),
            None => Err(format!("tensor `{name}` not found in checkpoint").into()),
        }
    }

    /// `Ok(None)` when the checkpoint does not have it, and an error when it
    /// does and this cannot read it.
    ///
    /// Two different problems, reported as the same one until a real
    /// Qwen3-Next turned up: its depthwise convolutions are stored
    /// `[channels, 1, kernel]`, this read rank three and gave up, and the
    /// error said the tensor was *missing* from a file it was sitting in.
    fn read(&self, name: &str) -> Res<Option<Tensor>> {
        let Some(key) = self.resolve(name) else { return Ok(None) };
        let st = SafeTensors::deserialize(&self.maps[self.index[key]])?;
        let view = st.tensor(key)?;

        let (rows, cols) = match view.shape() {
            [n] => (1, *n),
            [r, c] => (*r, *c),
            // A depthwise convolution's filters. PyTorch stores a `Conv1d`
            // with `groups = channels` as `[channels, 1, kernel]`: one filter
            // per channel and no mixing, so the middle axis is the input
            // channels *per group*, which is one. It carries nothing.
            [r, 1, c] => (*r, *c),
            other => {
                return Err(format!("tensor `{name}` has shape {other:?}, which this cannot read")
                    .into())
            }
        };
        let data = decode(view.data(), view.dtype()).ok_or_else(|| {
            format!("tensor `{name}` is stored as {:?}, which this cannot read", view.dtype())
        })?;
        let data = match view.dtype() {
            Dtype::F8_E4M3 => self.rescale(key, rows, cols, data)?,
            _ => data,
        };
        Ok(Some(Tensor::new(rows, cols, data)))
    }

    /// Put an fp8 tensor back on the scale it was quantised from.
    ///
    /// An fp8 checkpoint stores `X.weight` beside `X.weight_scale_inv`: one
    /// f32 for each block of the weight matrix, and the weight is the product
    /// of the two. The name says *inv*, and the multiplication is still a
    /// multiplication — that is DeepSeek's spelling, which Qwen and everyone
    /// publishing fp8 has followed, and it is the kind of thing that returns
    /// plausible text rather than an error when read the wrong way round.
    ///
    /// Nothing here reads the block size from the config, because it is
    /// already implied: a `[2048, 512]` weight under a `[16, 4]` scale is
    /// blocked 128 by 128, and deriving it cannot disagree with the file the
    /// way a separately-parsed constant could.
    fn rescale(&self, key: &str, rows: usize, cols: usize, mut data: Vec<f32>) -> Res<Vec<f32>> {
        let name = format!("{key}_scale_inv");
        let scale = self.read(&name)?.ok_or_else(|| {
            format!("`{key}` is fp8, and there is no `{name}` beside it to scale it by")
        })?;
        if scale.rows == 0 || scale.cols == 0 || scale.rows > rows || scale.cols > cols {
            return Err(format!(
                "`{key}` is {rows}x{cols} and its scales are {}x{}, which is not a blocking of it",
                scale.rows, scale.cols
            )
            .into());
        }
        let (high, wide) = (rows.div_ceil(scale.rows), cols.div_ceil(scale.cols));
        for r in 0..rows {
            let row = &mut data[r * cols..(r + 1) * cols];
            let base = (r / high) * scale.cols;
            // By block along the row, so the inner loop is a constant times a
            // contiguous run and the compiler can use the wide multiply.
            for (b, run) in row.chunks_mut(wide).enumerate() {
                let s = scale.data[base + b];
                for v in run.iter_mut() {
                    *v *= s;
                }
            }
        }
        Ok(data)
    }

    /// For 1-D tensors, where the shape is noise.
    pub fn get_flat(&self, name: &str) -> Res<Vec<f32>> {
        Ok(self.get(name)?.data)
    }

    pub fn try_get_flat(&self, name: &str) -> Option<Vec<f32>> {
        self.try_get(name).map(|t| t.data)
    }
}

fn decode(bytes: &[u8], dtype: Dtype) -> Option<Vec<f32>> {
    Some(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // Half precision: widen on load. bf16 is just f32 with the bottom 16
        // bits chopped off, which is why it is so cheap to convert and so
        // popular for training -- it keeps f32's exponent range, and range is
        // what gradients need.
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        // Raw, and not yet worth anything: an fp8 checkpoint stores a
        // separate scale for each block of weights, and these values are
        // meaningless until multiplied by it. `Checkpoint::read` does that
        // as soon as this returns, and is the only caller.
        Dtype::F8_E4M3 => bytes.iter().map(|&b| f8_e4m3_to_f32(b)).collect(),
        _ => return None,
    })
}

/// Every value an `e4m3` byte can take, by the byte.
///
/// There are two hundred and fifty-six of them, so the arithmetic is done
/// once at startup and never again — 80B weights would otherwise pay for the
/// same sixteen exponents over and over.
fn f8_e4m3_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (b, v) in t.iter_mut().enumerate() {
            let b = b as u8;
            let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
            let exp = ((b >> 3) & 0x0f) as i32;
            let man = (b & 0x07) as f32;
            *v = match exp {
                // Subnormal: no implicit leading one, and the exponent is
                // pinned at the smallest normal one.
                0 => sign * man / 8.0 * (2.0f32).powi(-6),
                // `e4m3fn`, the variant every checkpoint uses, spends no
                // encodings on infinity. All ones is the only NaN.
                0x0f if man == 7.0 => f32::NAN,
                _ => sign * (1.0 + man / 8.0) * (2.0f32).powi(exp - 7),
            };
        }
        t
    })
}

/// Whether the packing a `quantization_config` describes is one this engine
/// can read.
///
/// Kept beside [`decode`], which is what makes the answer true. Callers ask
/// it from two directions — a search result deciding whether to offer a
/// download, and a loader deciding whether to start one — and both were
/// answering "no" to everything before fp8 arrived.
///
/// `e4m3` only. `e5m2` trades mantissa for range and is published far less,
/// and guessing at it would be worse than saying no.
pub fn reads_packing(quant: &serde_json::Value) -> bool {
    let method = quant.get("quant_method").and_then(|m| m.as_str()).unwrap_or("");
    // Absent means e4m3: it is the default every fp8 publication uses, and
    // the ones that state it state that.
    let fmt = quant.get("fmt").and_then(|m| m.as_str()).unwrap_or("e4m3");
    method == "fp8" && fmt == "e4m3"
}

/// One `e4m3` byte: sign, four exponent bits biased by seven, three of
/// mantissa. Largest finite value 448, smallest subnormal 2^-9.
fn f8_e4m3_to_f32(b: u8) -> f32 {
    f8_e4m3_table()[b as usize]
}

/// IEEE 754 half -> single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            // Subnormal: there is no implicit leading 1, and the value is
            // `mant * 2^-24`. f32 has the range to store it as a normal
            // number, so renormalise: find the top set bit, make it the
            // implicit 1, and shift the rest into the fraction field.
            let top = 31 - mant.leading_zeros();
            let exp = 127 - 24 + top;
            let frac = (mant << (23 - top)) & 0x7f_ffff;
            (sign << 31) | (exp << 23) | frac
        }
        31 => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// Read a JSON file into a `serde_json::Value`.
pub fn read_json(path: &Path) -> Res<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

#[cfg(test)]
mod fp8_tests {
    use super::*;

    /// Worked out from the format rather than from this implementation, so
    /// the test can disagree with the code: sign, four exponent bits biased
    /// by seven, three of mantissa, no implicit one when the exponent is
    /// zero.
    #[test]
    fn e4m3_decodes_to_the_values_the_format_defines() {
        assert_eq!(f8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(f8_e4m3_to_f32(0x80), 0.0); // negative zero
        assert_eq!(f8_e4m3_to_f32(0x38), 1.0); // exp 7, mantissa 0
        assert_eq!(f8_e4m3_to_f32(0x3c), 1.5); // exp 7, mantissa 4 -> 1 + 4/8
        assert_eq!(f8_e4m3_to_f32(0xb8), -1.0);
        assert_eq!(f8_e4m3_to_f32(0x40), 2.0);

        // The ends. 448 is the largest finite value `e4m3fn` can hold, which
        // is the number every description of the format quotes.
        assert_eq!(f8_e4m3_to_f32(0x7e), 448.0);
        assert!(f8_e4m3_to_f32(0x7f).is_nan());
        assert!(f8_e4m3_to_f32(0xff).is_nan());

        // Smallest normal is 2^-6, and below it the mantissa carries the
        // value alone: 2^-9 is the smallest number the format has at all.
        assert_eq!(f8_e4m3_to_f32(0x08), 0.015625);
        assert_eq!(f8_e4m3_to_f32(0x01), 0.001953125);
    }

    /// Write a one-tensor-per-entry safetensors file by hand: eight bytes of
    /// header length, that much JSON, then the blob the offsets point into.
    fn write_safetensors(path: &std::path::Path, entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = String::from("{");
        let mut blob: Vec<u8> = Vec::new();
        for (i, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
            let start = blob.len();
            blob.extend_from_slice(bytes);
            let shape = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{start},{}]}}",
                blob.len()
            ));
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&blob);
        std::fs::write(path, out).unwrap();
    }

    /// The block indexing, which is the part worth doubting.
    ///
    /// A 4x8 weight under a 2x4 scale is blocked two rows by two columns, so
    /// every value in a block is multiplied by the same number and the four
    /// blocks across a row use four different ones. Getting the row stride
    /// wrong, or transposing the two, still produces a finite matrix of
    /// plausible size — which is why this checks a value per block rather
    /// than a norm over the whole thing.
    #[test]
    fn an_fp8_tensor_comes_back_multiplied_by_its_block_scales() {
        let dir = std::env::temp_dir().join(format!("kvad-fp8-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        // Every weight is 1.0, so whatever comes back *is* the scale that was
        // applied to it, named by position.
        let weights = vec![0x38u8; 4 * 8];
        let scales: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scale_bytes = scales.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>();
        write_safetensors(
            &path,
            &[
                ("w.weight", "F8_E4M3", vec![4, 8], weights),
                ("w.weight_scale_inv", "F32", vec![2, 4], scale_bytes),
            ],
        );

        let ck = Checkpoint::open(&[path]).unwrap();
        let t = ck.get("w.weight").unwrap();
        assert_eq!((t.rows, t.cols), (4, 8));
        for r in 0..4 {
            for c in 0..8 {
                let want = scales[(r / 2) * 4 + c / 2];
                assert_eq!(t.data[r * 8 + c], want, "at ({r}, {c})");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An fp8 tensor with no scales beside it is not a tensor. Reading it as
    /// raw e4m3 would hand back numbers that are wrong by whatever the scale
    /// would have been, and nothing downstream could tell.
    #[test]
    fn an_fp8_tensor_without_its_scales_is_an_error() {
        let dir = std::env::temp_dir().join(format!("kvad-fp8-bare-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        write_safetensors(&path, &[("w.weight", "F8_E4M3", vec![4, 8], vec![0x38u8; 32])]);

        let ck = Checkpoint::open(&[path]).unwrap();
        let err = ck.get("w.weight").unwrap_err().to_string();
        assert!(err.contains("weight_scale_inv"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The reader's list and the blocker's list are the same list.
    #[test]
    fn only_the_packings_that_decode_are_accepted() {
        let q = |json: &str| serde_json::from_str::<serde_json::Value>(json).unwrap();

        assert!(reads_packing(&q(r#"{"quant_method":"fp8","fmt":"e4m3"}"#)));
        // Qwen's small publications omit `fmt`; e4m3 is what they mean.
        assert!(reads_packing(&q(r#"{"quant_method":"fp8"}"#)));

        // Range in place of mantissa, published rarely, and not implemented.
        assert!(!reads_packing(&q(r#"{"quant_method":"fp8","fmt":"e5m2"}"#)));
        assert!(!reads_packing(&q(r#"{"quant_method":"awq"}"#)));
        assert!(!reads_packing(&q(r#"{"quant_method":"compressed-tensors"}"#)));
        assert!(!reads_packing(&q("{}")));
    }
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;
    use std::process;

    #[test]
    fn half_precision_conversion() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert!((f16_to_f32(0x3555) - 0.333_251).abs() < 1e-5);
        assert!(f16_to_f32(0x7c00).is_infinite());
        // Smallest positive subnormal: 2^-24.
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }

    /// A depthwise convolution's filters, as a real checkpoint stores them.
    ///
    /// PyTorch writes a `Conv1d` with `groups = channels` as
    /// `[channels, 1, kernel]`. Every fixture in this repository writes the
    /// two-dimensional spelling, because that is what the engine asks for, so
    /// nothing else here would notice this going back.
    #[test]
    fn a_depthwise_convolution_reads_as_the_matrix_it_is() {
        let path = std::env::temp_dir().join(format!("kvad-rank3-{}.safetensors", process::id()));
        write(&path, &[("conv.weight", &[3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])]);
        let ckpt = super::Checkpoint::open(std::slice::from_ref(&path)).unwrap();

        let t = ckpt.get("conv.weight").unwrap();
        assert_eq!((t.rows, t.cols), (3, 2), "the middle axis carries nothing");
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        std::fs::remove_file(&path).unwrap();
    }

    /// Absent and unreadable are different problems.
    ///
    /// They were the same message until a real Qwen3-Next arrived: rank three
    /// fell through to `None` and the error said the tensor was missing from a
    /// file it was sitting in, which sends a reader looking in the wrong
    /// place.
    #[test]
    fn a_tensor_that_is_there_and_unreadable_does_not_say_it_is_missing() {
        let path = std::env::temp_dir().join(format!("kvad-rank4-{}.safetensors", process::id()));
        write(&path, &[("odd.weight", &[2, 1, 1, 2], &[1.0, 2.0, 3.0, 4.0][..])]);
        let ckpt = super::Checkpoint::open(std::slice::from_ref(&path)).unwrap();

        let err = ckpt.get("odd.weight").unwrap_err().to_string();
        assert!(err.contains("shape"), "{err}");
        assert!(!err.contains("not found"), "{err}");
        assert!(ckpt.try_get("odd.weight").is_none(), "and still not usable");

        let missing = ckpt.get("absent.weight").unwrap_err().to_string();
        assert!(missing.contains("not found"), "{missing}");
        std::fs::remove_file(&path).unwrap();
    }

    /// The smallest safetensors writer that will do: a header of shapes and
    /// offsets, then f32 back to back.
    fn write(path: &std::path::Path, tensors: &[(&str, &[usize], &[f32])]) {
        let mut header = String::from("{");
        let mut blob: Vec<u8> = Vec::new();
        for (i, (name, shape, data)) in tensors.iter().enumerate() {
            let start = blob.len();
            for v in *data {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
                dims.join(","),
                blob.len()
            ));
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&blob);
        std::fs::write(path, out).unwrap();
    }
}
