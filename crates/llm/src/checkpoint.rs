//! Names for SDXL checkpoints in one file, in Stability's own layout.
//!
//! A diffusers repo is a model by its name alone. A checkpoint in one file
//! is named three ways:
//!
//! - **`repo`**, when the repo's only model is one checkpoint, as Pony's is:
//!   no `config.json`, no `model_index.json`, and one `.safetensors` file at
//!   its top.
//! - **`repo:file.safetensors`**, for a repo with several, or with diffusers
//!   folders beside it, as Illustrious' and the SDXL base's have.
//! - **A path to a file on this machine**, for what the Hub does not have:
//!   Civitai's, which Kvad cannot fetch.
//!
//! A file is one of these only if its header is SDXL's: the UNet under
//! `model.diffusion_model.` and bigG under `conditioner.embedders.1.model.`.
//! A LoRA beside a checkpoint, or an SD 1.5 file, is not. On the Hub the
//! header is read with a range request, so a file that is not one is refused
//! before its gigabytes are fetched. The GPU crate reads the file and knows
//! the layout; this module finds it.

use crate::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A checkpoint on this machine, by the name it is known by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub name: String,
    pub file: PathBuf,
}

/// `repo:file.safetensors` as its two halves.
pub fn split(name: &str) -> Option<(&str, &str)> {
    if crate::weights::looks_like_path(name) {
        return None;
    }
    let (repo, file) = name.split_once(':')?;
    (repo.contains('/') && file.to_ascii_lowercase().ends_with(".safetensors") && file.len() > ".safetensors".len()).then_some((repo, file))
}

/// Whether `name` is a path to a checkpoint file, as typed.
pub fn is_path(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".safetensors") && (crate::weights::looks_like_path(name) || Path::new(name).is_file())
}

/// Whether a file with these tensors is an SDXL checkpoint in Stability's
/// layout.
pub fn is_sdxl<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    let (mut unet, mut big_g) = (false, false);
    for n in names {
        unet |= n.starts_with("model.diffusion_model.");
        big_g |= n.starts_with("conditioner.embedders.1.model.");
    }
    unet && big_g
}

/// The tensor names in a safetensors file's header.
pub fn names(path: &Path) -> Option<Vec<String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let n = u64::from_le_bytes(len);
    // A header is kilobytes; anything past a few hundred megabytes is not one.
    if n > 256 << 20 {
        return None;
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).ok()?;
    header_names(&header)
}

fn header_names(header: &[u8]) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_slice(header).ok()?;
    Some(v.as_object()?.keys().filter(|k| *k != "__metadata__").cloned().collect())
}

/// The tensor names of `file` in `repo`, from two range requests to the
/// Hub, without its data.
pub fn remote_names(repo: &str, file: &str) -> Res<Vec<String>> {
    // A path in a URL: its reserved bytes escaped, its slashes kept.
    let path: String = file
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => (b as char).to_string(),
            other => format!("%{other:02X}"),
        })
        .collect();
    let url = format!("https://huggingface.co/{repo}/resolve/main/{path}");
    let get = |from: u64, to: u64| -> Res<Vec<u8>> {
        let mut r = ureq::get(&url).header("Range", &format!("bytes={from}-{to}")).call().map_err(|e| format!("reading {file}'s header in {repo}: {e}"))?;
        Ok(r.body_mut().with_config().limit(300 << 20).read_to_vec()?)
    };
    let len = u64::from_le_bytes(get(0, 7)?.get(..8).ok_or("a file too short to be safetensors")?.try_into()?);
    if len > 256 << 20 {
        return Err(format!("{file} in {repo} is not a safetensors file").into());
    }
    header_names(&get(8, 8 + len - 1)?).ok_or_else(|| format!("{file} in {repo} has no safetensors header").into())
}

/// The SDXL checkpoints at the top of a cache entry, `dir`, of `repo`, by
/// the names they are known by: the repo's own, if the file is all it holds.
pub fn locals(dir: &Path, repo: &str) -> Vec<Local> {
    let Ok(revisions) = std::fs::read_dir(dir.join("snapshots")) else { return Vec::new() };
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for rev in revisions.filter_map(|e| e.ok()).map(|e| e.path()) {
        let Ok(entries) = std::fs::read_dir(&rev) else { continue };
        for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if name.to_ascii_lowercase().ends_with(".safetensors") && path.is_file() && !files.iter().any(|(n, _)| *n == name) && names(&path).is_some_and(|n| is_sdxl(n.iter().map(String::as_str))) {
                files.push((name, path));
            }
        }
    }
    files.sort();
    let alone = files.len() == 1 && crate::hub::model_file(dir, "config.json").is_none() && crate::hub::model_file(dir, "model_index.json").is_none();
    files
        .into_iter()
        .map(|(f, file)| Local { name: if alone { repo.to_string() } else { format!("{repo}:{f}") }, file })
        .collect()
}

/// The checkpoint `name` means, if it is on this machine.
pub fn local(name: &str) -> Option<Local> {
    if is_path(name) {
        let file = std::fs::canonicalize(name).ok()?;
        return names(&file).is_some_and(|n| is_sdxl(n.iter().map(String::as_str))).then(|| Local { name: name.to_string(), file });
    }
    let repo = split(name).map_or(name, |(r, _)| r);
    if !repo.contains('/') || crate::gguf::split(name).is_some() {
        return None;
    }
    let dir = crate::hub::cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    locals(&dir, repo).into_iter().find(|l| l.name == name)
}

/// A checkpoint on the Hub: the file `name` means in its repo, found
/// without downloading it, and checked to be SDXL's.
#[derive(Debug, Clone)]
pub struct Found {
    pub repo: String,
    pub file: String,
}

/// A repo named alone, on the Hub: the `.safetensors` files at its top, if
/// nothing in it makes it a model of its own (a `config.json`, a
/// `model_index.json`); `None` if something does.
pub fn candidates(repo: &str) -> Res<Option<Vec<String>>> {
    let files = crate::hub::repo_files(repo)?;
    if files.iter().any(|f| f == "config.json" || f == "model_index.json") {
        return Ok(None);
    }
    Ok(Some(files.into_iter().filter(|f| !f.contains('/') && f.to_ascii_lowercase().ends_with(".safetensors")).collect()))
}

/// Find the checkpoint `name` means on the Hub.
pub fn find(name: &str) -> Res<Found> {
    let (repo, file) = match split(name) {
        Some((repo, file)) => (repo.to_string(), file.to_string()),
        None if name.contains('/') && !crate::weights::looks_like_path(name) => {
            let tops = candidates(name)?.ok_or_else(|| format!("{name} is a model of its own, not one checkpoint in a file"))?;
            match tops.as_slice() {
                [one] => (name.to_string(), one.to_string()),
                [] => return Err(format!("{name} has no checkpoint file").into()),
                many => {
                    let list: Vec<String> = many.iter().map(|f| format!("{name}:{f}")).collect();
                    return Err(format!("{name} has {} .safetensors files at its top; name the checkpoint: {}", many.len(), list.join(", ")).into());
                }
            }
        }
        None => return Err(format!("`{name}` is not a checkpoint's name").into()),
    };
    let names = remote_names(&repo, &file)?;
    if !is_sdxl(names.iter().map(String::as_str)) {
        return Err(format!("{file} in {repo} is not an SDXL checkpoint in Stability's layout, which is the one kind of single file read here").into());
    }
    Ok(Found { repo, file })
}

/// Download what [`find`] found, or take it from the cache.
pub fn fetch(found: &Found, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<PathBuf> {
    progress(&format!("fetching {} from {}", found.file, found.repo));
    fetch_file(&found.repo, &found.file, watch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_split_only_at_a_checkpoint_file() {
        assert_eq!(split("OnomaAIResearch/Illustrious-xl-early-release-v0:Illustrious-XL-v0.1.safetensors"), Some(("OnomaAIResearch/Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors")));
        assert_eq!(split("a/b:model.SafeTensors"), Some(("a/b", "model.SafeTensors")));
        assert_eq!(split("a/b:Q4_K_S"), None);
        assert_eq!(split("a/b"), None);
        assert_eq!(split("nobody:x.safetensors"), None);
        assert_eq!(split("a/b:.safetensors"), None);
        assert_eq!(split("./dir/x:y.safetensors"), None, "a path is not a repo");
    }

    #[test]
    fn an_sdxl_file_has_a_unet_and_big_g() {
        assert!(is_sdxl(["model.diffusion_model.out.2.weight", "conditioner.embedders.1.model.ln_final.weight"]));
        assert!(!is_sdxl(["model.diffusion_model.out.2.weight", "cond_stage_model.transformer.text_model.final_layer_norm.weight"]), "SD 1.5");
        assert!(!is_sdxl(["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]), "a LoRA");
    }

    /// A repo whose only model is one checkpoint is named by the repo; one
    /// with a checkpoint beside its diffusers folders, by the file; a LoRA
    /// beside it is no checkpoint at all.
    #[test]
    fn a_cache_entry_names_its_checkpoints() {
        let root = std::env::temp_dir().join(format!("kvad-checkpoint-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let file = |path: &Path, names: &[&str]| {
            let header: serde_json::Map<String, serde_json::Value> =
                names.iter().map(|n| (n.to_string(), serde_json::json!({ "dtype": "F16", "shape": [1], "data_offsets": [0, 2] }))).collect();
            let h = serde_json::to_vec(&header).unwrap();
            let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
            bytes.extend(h);
            bytes.extend([0, 0]);
            std::fs::write(path, bytes).unwrap();
        };
        let sdxl = ["model.diffusion_model.out.2.weight", "conditioner.embedders.1.model.ln_final.weight"];
        let alone = root.join("models--o--pony/snapshots/abc");
        std::fs::create_dir_all(&alone).unwrap();
        file(&alone.join("pony.safetensors"), &sdxl);
        let got = locals(&root.join("models--o--pony"), "o/pony");
        assert_eq!(got.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["o/pony"]);

        let beside = root.join("models--o--base/snapshots/abc");
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::write(beside.join("model_index.json"), "{}").unwrap();
        file(&beside.join("base.safetensors"), &sdxl);
        file(&beside.join("offset-lora.safetensors"), &["lora_unet_x.lora_down.weight"]);
        let got = locals(&root.join("models--o--base"), "o/base");
        assert_eq!(got.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["o/base:base.safetensors"]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
