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
}

/// Search the Hub, newest-first by download count.
pub fn search(query: &str, limit: usize) -> Res<Vec<HubModel>> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&config=true&sort=downloads&direction=-1&limit={}&filter=text-generation",
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
            HubModel {
                arch: model_type.as_deref().and_then(Arch::from_model_type),
                model_type,
                downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                gated: !matches!(m.get("gated"), None | Some(serde_json::Value::Bool(false))),
                looks_instruct: ["instruct", "-it", "chat", "sft"]
                    .iter()
                    .any(|k| lower.contains(k)),
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
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
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

impl State {
    pub fn path() -> PathBuf {
        let base = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs_home().join(".config"));
        base.join("kvad").join("state.json")
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

    #[test]
    fn byte_formatting() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024 * 1024), "5.0 GB");
    }
}
