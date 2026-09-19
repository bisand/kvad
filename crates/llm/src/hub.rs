//! Finding models on the HuggingFace Hub, and keeping track of local ones.
//!
//! # Searching
//!
//! The Hub's model index is a plain JSON endpoint — no SDK, no auth for public
//! models:
//!
//! ```text
//! GET https://huggingface.co/api/models?search=smollm&config=true&sort=downloads
//! ```
//!
//! `config=true` is the useful part: it returns each model's `model_type`
//! *before* you download several gigabytes, so the search can say which results
//! this engine can actually run. Of the roughly two million models on the Hub,
//! we handle two architecture families — being honest about that in the listing
//! is better than failing after the download.
//!
//! # Local models
//!
//! `hf-hub` stores downloads in the standard HuggingFace cache
//! (`~/.cache/huggingface/hub` unless `HF_HOME` says otherwise), laid out as
//! `models--{owner}--{name}/`. Listing what is on disk is a directory walk;
//! nothing here maintains a database of its own.

use crate::model::Arch;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A search result. Everything here comes from the Hub and is display-only
/// data — nothing in it is trusted or executed.
#[derive(Debug, Clone)]
pub struct HubModel {
    pub id: String,
    pub model_type: Option<String>,
    pub arch: Option<Arch>,
    pub downloads: u64,
    pub likes: u64,
    pub gated: bool,
    /// Heuristic, from the name. Only loading the tokenizer config can
    /// actually confirm it, which `pull` does.
    pub looks_instruct: bool,
    /// Weights, counted by the Hub from the safetensors headers. `None` for a
    /// repo that ships no safetensors — a GGUF mirror, say — which is also a
    /// repo this engine cannot load.
    pub params: Option<u64>,
    /// Bytes to download: the parameter counts multiplied by the width of the
    /// dtype each is stored in. Not the same as what it costs in memory here,
    /// which depends on the precision it is loaded at.
    pub download_bytes: Option<u64>,
}

impl HubModel {
    pub fn runnable(&self) -> bool {
        self.arch.is_some() && !self.gated
    }

    /// One-line reason a model cannot be run, if it cannot.
    pub fn blocker(&self) -> Option<String> {
        if self.gated {
            Some("gated — needs licence acceptance on huggingface.co".into())
        } else {
            match &self.model_type {
                None => Some("no config.json".into()),
                Some(t) if self.arch.is_none() => Some(format!("unsupported arch `{t}`")),
                _ => None,
            }
        }
    }

    /// What this model's weights would occupy here, at each precision.
    ///
    /// Weights only — a KV cache needs the context length, and a search result
    /// does not carry one. For a 7B model that understates the real
    /// requirement by a gigabyte or two at a long context, which is worth
    /// knowing and is still the right number to show: it is the part that is
    /// fixed, and the part that decides whether the download is worth starting.
    pub fn memory_at(&self, precision: crate::quant::Precision) -> Option<u64> {
        self.params.map(|p| precision.weight_bytes(p))
    }

    /// The cheapest precision whose weights fit in this machine's memory.
    ///
    /// `None` means either that we do not know the size, or that nothing fits.
    /// [`HubModel::fit`] tells those apart.
    pub fn best_precision(&self) -> Option<crate::quant::Precision> {
        let usable = crate::machine::usable_memory()?;
        let params = self.params?;
        // Largest first, so the answer is the *best* precision that fits
        // rather than merely the smallest.
        crate::quant::Precision::SMALLEST_FIRST
            .into_iter()
            .rev()
            .find(|p| p.weight_bytes(params) <= usable)
    }

    pub fn fit(&self) -> Fit {
        match (self.params, crate::machine::usable_memory()) {
            (None, _) | (_, None) => Fit::Unknown,
            (Some(_), Some(_)) => match self.best_precision() {
                Some(p) => Fit::At(p),
                None => Fit::TooBig,
            },
        }
    }
}

/// Whether a model will run on this machine, and how cheaply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Fits, at this precision or anything smaller.
    At(crate::quant::Precision),
    /// Does not fit even at q4.
    TooBig,
    /// The Hub did not say how big it is, or we cannot read this machine's
    /// memory. Saying nothing beats guessing.
    Unknown,
}

impl std::fmt::Display for Fit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fit::At(p) => write!(f, "fits at {p}"),
            Fit::TooBig => f.write_str("too big"),
            Fit::Unknown => f.write_str("?"),
        }
    }
}

/// Bytes a dtype takes per value, as the Hub names them.
fn dtype_bytes(name: &str) -> u64 {
    match name {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        // F8_E4M3, F8_E5M2, I8, U8, BOOL, and the 4-bit types, which the Hub
        // still counts one value per byte.
        _ => 1,
    }
}

/// Search the Hub, newest-first by download count.
pub fn search(query: &str, limit: usize) -> Res<Vec<HubModel>> {
    // `expand[]` *replaces* the default field set rather than adding to it, so
    // everything this function reads has to be named — including the fields
    // that used to arrive for free. Asking for one more thing and silently
    // losing `config` was the first version of this.
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit={}\
         &filter=text-generation\
         &expand[]=config&expand[]=downloads&expand[]=likes&expand[]=gated\
         &expand[]=safetensors",
        urlencode(query),
        limit.clamp(1, 100)
    );
    let body = ureq::get(&url).call()?.body_mut().read_to_string()?;
    let items: serde_json::Value = serde_json::from_str(&body)?;
    let items = items.as_array().ok_or("unexpected response from the Hub API")?;

    Ok(items
        .iter()
        .map(|m| {
            let id = m.get("modelId").or_else(|| m.get("id")).and_then(|v| v.as_str()).unwrap_or("?");
            let model_type = m
                .get("config")
                .and_then(|c| c.get("model_type"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let lower = id.to_ascii_lowercase();
            let safetensors = m.get("safetensors");
            let params = safetensors.and_then(|s| s.get("total")).and_then(|v| v.as_u64());
            // Each dtype's count times its width. A model stored half in bf16
            // and half in fp8 is neither one nor the other.
            let download_bytes = safetensors
                .and_then(|s| s.get("parameters"))
                .and_then(|p| p.as_object())
                .map(|by_dtype| {
                    by_dtype
                        .iter()
                        .filter_map(|(dtype, n)| Some(n.as_u64()? * dtype_bytes(dtype)))
                        .sum()
                });
            HubModel {
                arch: model_type.as_deref().and_then(Arch::from_model_type),
                model_type,
                downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                gated: !matches!(m.get("gated"), None | Some(serde_json::Value::Bool(false))),
                looks_instruct: ["instruct", "-it", "chat", "sft"]
                    .iter()
                    .any(|k| lower.contains(k)),
                params,
                download_bytes,
                id: id.to_string(),
            }
        })
        .collect())
}

/// A model already downloaded to the local cache.
#[derive(Debug, Clone)]
pub struct LocalModel {
    pub id: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// Read from the cached config.json, so this is authoritative rather than
    /// a guess.
    pub arch: Option<Arch>,
    pub complete: bool,
}

/// Root of the HuggingFace cache, honouring the usual environment variables.
pub fn cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        return PathBuf::from(v).join("hub");
    }
    dirs_home().join(".cache/huggingface/hub")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

/// Everything currently in the cache.
pub fn local_models() -> Vec<LocalModel> {
    let root = cache_dir();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out: Vec<LocalModel> = entries
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            // `models--openai-community--gpt2` -> `openai-community/gpt2`.
            let rest = name.strip_prefix("models--")?;
            let id = rest.replacen("--", "/", 1);
            let path = entry.path();
            let bytes = dir_size(&path);
            let files = snapshot_files(&path);
            let arch = find_config(&path)
                .and_then(|c| crate::weights::read_json(&c).ok())
                .and_then(|j| {
                    j.get("model_type")
                        .and_then(|m| m.as_str())
                        .and_then(Arch::from_model_type)
                });
            Some(LocalModel {
                id,
                path,
                bytes,
                arch,
                // A cache entry with a config but no weights is a half-finished
                // `info` call, not a usable model.
                complete: files.iter().any(|f| f.ends_with(".safetensors")),
            })
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn find_local(id: &str) -> Option<LocalModel> {
    local_models().into_iter().find(|m| m.id.eq_ignore_ascii_case(id))
}

/// Every model trained on this machine, by name.
///
/// Kept apart from [`local_models`] rather than folded into it, because the
/// two are not the same kind of thing. A downloaded model can be deleted and
/// fetched again; a trained one is the only copy there is. `kvad ls` lists
/// them in a section of their own for that reason, and `kvad rm` says
/// "retraining" rather than "re-download" when asked to delete one.
pub fn trained_models() -> Vec<LocalModel> {
    let root = crate::weights::models_dir();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out: Vec<LocalModel> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|entry| {
            let path = entry.path();
            let config = path.join("config.json");
            LocalModel {
                id: entry.file_name().to_string_lossy().into_owned(),
                bytes: dir_size(&path),
                arch: crate::weights::read_json(&config).ok().and_then(|j| {
                    j.get("model_type").and_then(|m| m.as_str()).and_then(Arch::from_model_type)
                }),
                // A directory left behind by a run that was stopped before
                // its first checkpoint has a tokeniser and no weights.
                complete: path.join("model.safetensors").is_file() && config.is_file(),
                path,
            }
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn find_trained(name: &str) -> Option<LocalModel> {
    trained_models().into_iter().find(|m| m.id == name)
}

fn find_config(model_dir: &Path) -> Option<PathBuf> {
    let snapshots = model_dir.join("snapshots");
    for rev in std::fs::read_dir(snapshots).ok()?.filter_map(|e| e.ok()) {
        let candidate = rev.path().join("config.json");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// The human-readable file names in a cache entry.
///
/// The cache stores each download twice over: `blobs/` holds the real files
/// under content hashes, and `snapshots/<revision>/` holds symlinks to them
/// under their proper names. So "is this model actually downloaded?" has to be
/// answered from the snapshot side — the blob side is all hashes.
fn snapshot_files(model_dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(revisions) = std::fs::read_dir(model_dir.join("snapshots")) else {
        return names;
    };
    for rev in revisions.filter_map(|e| e.ok()) {
        let Ok(entries) = std::fs::read_dir(rev.path()) else { continue };
        names.extend(entries.filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned())));
    }
    names
}

/// Bytes on disk, counting each file once.
///
/// `DirEntry::metadata` does *not* follow symlinks, which is exactly what is
/// wanted here: the links under `snapshots/` report as neither file nor
/// directory and are skipped, so only the real blobs are counted. Following
/// them would report every model at twice its true size.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.filter_map(|e| e.ok()) {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total += meta.len();
            }
        }
    }
    total
}

pub fn human_bytes(b: u64) -> String {
    // Up to TB, because a search result can now be a 1.5 TB checkpoint and
    // "1491.9 GB" is a number nobody reads as one and a half terabytes.
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

// ---------------------------------------------------------------------------
// Which model is "active"
// ---------------------------------------------------------------------------

/// The selected model, remembered between runs.
///
/// Deliberately a three-line JSON file rather than anything clever: the models
/// themselves live in the HuggingFace cache, and duplicating that state would
/// only create a second source of truth to keep in sync.
pub struct State;

/// Where settings somebody typed are kept: `$XDG_CONFIG_HOME/kvad`, or
/// `~/.config/kvad`. Config rather than data, because everything in here can
/// be written again from scratch — unlike a trained model. See
/// [`crate::weights::data_dir`] for the other side of that line.
pub fn config_dir() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join(".config"))
        .join("kvad")
}

impl State {
    pub fn path() -> PathBuf {
        config_dir().join("state.json")
    }

    pub fn active() -> Option<String> {
        let json = crate::weights::read_json(&Self::path()).ok()?;
        json.get("active")?.as_str().map(str::to_string)
    }

    pub fn set_active(id: &str) -> Res<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::json!({ "active": id }).to_string())?;
        Ok(())
    }

    pub fn clear() -> Res<()> {
        let path = Self::path();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            other => format!("%{other:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoding_handles_spaces_and_slashes() {
        assert_eq!(urlencode("smollm2 instruct"), "smollm2+instruct");
        assert_eq!(urlencode("Qwen/Qwen2.5"), "Qwen%2FQwen2.5");
    }

    fn sized(params: Option<u64>) -> HubModel {
        HubModel {
            id: "a/b".into(),
            model_type: Some("llama".into()),
            arch: Some(Arch::Llama),
            downloads: 0,
            likes: 0,
            gated: false,
            looks_instruct: false,
            params,
            download_bytes: params.map(|p| p * 2),
        }
    }

    /// The quantised formats are not whole bytes, and the arithmetic that says
    /// how big a download will be has to know it.
    #[test]
    fn a_models_size_here_depends_on_the_precision_it_is_loaded_at() {
        use crate::quant::Precision;
        let m = sized(Some(8_000_000_000));

        // 4 bytes, 9 bits and 5 bits per weight.
        assert_eq!(m.memory_at(Precision::F32), Some(32_000_000_000));
        assert_eq!(m.memory_at(Precision::Q8), Some(9_000_000_000));
        assert_eq!(m.memory_at(Precision::Q4), Some(5_000_000_000));
        // Smaller is smaller, at every step.
        assert!(m.memory_at(Precision::Q4) < m.memory_at(Precision::Q8));
        assert!(m.memory_at(Precision::Q8) < m.memory_at(Precision::F32));

        // A repo with no safetensors says nothing rather than zero.
        assert_eq!(sized(None).memory_at(Precision::Q8), None);
        assert_eq!(sized(None).fit(), Fit::Unknown);
    }

    /// The verdict has to be the *best* precision that fits, not the smallest
    /// one that does — otherwise every model would report q4.
    #[test]
    fn the_fit_is_the_best_precision_that_will_run() {
        use crate::quant::Precision;
        let Some(usable) = crate::machine::usable_memory() else { return };

        // A model whose f32 weights alone exceed memory, but whose q8 fit.
        let params = (usable as f64 / Precision::Q8.bytes_per_weight()) as u64;
        assert_eq!(sized(Some(params)).fit(), Fit::At(Precision::Q8));

        // Something that fits comfortably at full precision.
        let tiny = (usable as f64 / 4.0) as u64 / 100;
        assert_eq!(sized(Some(tiny)).fit(), Fit::At(Precision::F32));

        // And something no precision saves.
        let huge = (usable as f64 / Precision::Q4.bytes_per_weight()) as u64 * 4;
        assert_eq!(sized(Some(huge)).fit(), Fit::TooBig);
    }

    /// The Hub reports a mixed-dtype checkpoint as counts per dtype, and the
    /// download is the sum of each times its width.
    #[test]
    fn download_size_counts_each_dtype_at_its_own_width() {
        assert_eq!(dtype_bytes("BF16"), 2);
        assert_eq!(dtype_bytes("F32"), 4);
        assert_eq!(dtype_bytes("F8_E4M3"), 1);
        // Anything unrecognised counts as a byte rather than as nothing, so an
        // unknown dtype understates rather than vanishing.
        assert_eq!(dtype_bytes("SOMETHING_NEW"), 1);
    }

    #[test]
    fn byte_formatting() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024 * 1024), "5.0 GB");
        assert_eq!(human_bytes(1536 * 1024 * 1024 * 1024), "1.5 TB");
    }
}
