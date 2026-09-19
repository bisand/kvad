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
//! how a model trained by this repository's own `nanograd` gets here, and the
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
use std::collections::HashMap;
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

    /// Look a tensor up, converting to `f32`.
    ///
    /// Checkpoints disagree about prefixes — GPT-2 saves `wte.weight` or
    /// `transformer.wte.weight` depending on which Python class wrote it — so
    /// a few spellings are tried before giving up.
    pub fn try_get(&self, name: &str) -> Option<Tensor> {
        let candidates = [
            name.to_string(),
            format!("transformer.{name}"),
            format!("model.{name}"),
        ];
        let key = candidates.iter().find(|c| self.index.contains_key(*c))?;
        let st = SafeTensors::deserialize(&self.maps[self.index[key]]).ok()?;
        let view = st.tensor(key).ok()?;

        let shape = view.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => return None,
        };
        Some(Tensor::new(rows, cols, decode(view.data(), view.dtype())?))
    }

    pub fn get(&self, name: &str) -> Res<Tensor> {
        self.try_get(name)
            .ok_or_else(|| format!("tensor `{name}` not found in checkpoint").into())
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
        _ => return None,
    })
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
mod tests {
    use super::f16_to_f32;

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
}
