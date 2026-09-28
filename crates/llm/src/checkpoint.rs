//! Names for SDXL and SD 1.5 checkpoints in one file, in Stability's own
//! layout.
//!
//! A diffusers repo is a model by its name alone. A checkpoint in one file
//! is named three ways:
//!
//! - **`repo`**, when the repo's only model is one checkpoint, as Pony's is:
//!   no `config.json`, no `model_index.json`, and one `.safetensors` file at
//!   its top. Only the Hub can say that a repo holds nothing else; the
//!   cache holds what was fetched, and one file of `Lykon/DreamShaper`'s 37,
//!   beside no model index, would look like the whole repo. So a pull that
//!   found it so by the repo's name records it in the cache entry
//!   ([`ONLY`]), and a checkpoint is listed by its repo's name only then.
//! - **`repo:file.safetensors`**, for a repo with several, or with diffusers
//!   folders beside it, as Illustrious' and the SDXL base's have.
//! - **A path to a file on this machine**, for what the Hub does not have:
//!   Civitai's, which Kvad cannot fetch.
//!
//! A file is one of these only if its header is one of the two [`Kind`]s:
//! the UNet under `model.diffusion_model.`, and SDXL's bigG under
//! `conditioner.embedders.1.model.` or SD 1.5's CLIP-L under
//! `cond_stage_model.transformer.`. A LoRA beside a checkpoint is neither,
//! and nor is SD 2's, whose text encoder is OpenCLIP's under
//! `cond_stage_model.model.`. On the Hub the header is read with a range
//! request, so a file that is not one is refused before its gigabytes are
//! fetched. The GPU crate reads the file and knows the layout; this module
//! finds it, and says which it is.

use crate::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Which model a checkpoint in one file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Sdxl,
    Sd15,
}

impl Kind {
    /// The diffusers pipeline its models are, and so the one that runs it.
    pub fn pipeline(self) -> &'static str {
        match self {
            Kind::Sdxl => "StableDiffusionXLPipeline",
            Kind::Sd15 => "StableDiffusionPipeline",
        }
    }
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Kind::Sdxl => "SDXL",
            Kind::Sd15 => "SD 1.5",
        })
    }
}

/// The file, at the top of a cache entry, that names the checkpoint a pull
/// by the repo's name found to be the repo's only model: the one it is
/// listed as the repo for. Beside `blobs/` and `snapshots/`, where `hf-hub`
/// reads nothing, and deleted with the entry.
pub const ONLY: &str = "kvad-only-checkpoint";

/// A checkpoint on this machine, by the name it is known by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub name: String,
    pub file: PathBuf,
    pub kind: Kind,
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

/// Which checkpoint in Stability's layout a file with these tensors is, if
/// it is one: a UNet and one of the two text encoders, and not both.
pub fn kind<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<Kind> {
    let (mut unet, mut big_g, mut clip_l) = (false, false, false);
    for n in names {
        unet |= n.starts_with("model.diffusion_model.");
        big_g |= n.starts_with("conditioner.embedders.1.model.");
        clip_l |= n.starts_with("cond_stage_model.transformer.");
    }
    match (unet, big_g, clip_l) {
        (true, true, false) => Some(Kind::Sdxl),
        (true, false, true) => Some(Kind::Sd15),
        _ => None,
    }
}

/// The kind of the file at `path`, from its header.
fn kind_of(path: &Path) -> Option<Kind> {
    kind(names(path)?.iter().map(String::as_str))
}

/// The tensor names in a safetensors file's header.
pub fn names(path: &Path) -> Option<Vec<String>> {
    Some(header(path)?.into_iter().map(|(k, _)| k).collect())
}

/// A safetensors file's header: each tensor's entry, by its name, without
/// `__metadata__`.
pub(crate) fn header(path: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
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
    let mut v: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header).ok()?;
    v.remove("__metadata__");
    Some(v)
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

/// The `.safetensors` files at the top of a cache entry, `dir`, by their
/// names, in order: each name once, whichever revision holds it.
pub(crate) fn tops(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(revisions) = std::fs::read_dir(dir.join("snapshots")) else { return Vec::new() };
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for rev in revisions.filter_map(|e| e.ok()).map(|e| e.path()) {
        let Ok(entries) = std::fs::read_dir(&rev) else { continue };
        for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if name.to_ascii_lowercase().ends_with(".safetensors") && path.is_file() && !files.iter().any(|(n, _)| *n == name) {
                files.push((name, path));
            }
        }
    }
    files.sort();
    files
}

/// `file`'s name in `repo`: the repo's own when `dir`'s record `only` names
/// it as the repo's one model, `repo:file` otherwise.
pub(crate) fn name_in(dir: &Path, only: &str, repo: &str, file: &str) -> String {
    match std::fs::read_to_string(dir.join(only)).ok().as_deref() == Some(file) {
        true => repo.to_string(),
        false => format!("{repo}:{file}"),
    }
}

/// Record in the cache entry `path` is in that `file` is `repo`'s one model,
/// in the record `only`. Nothing for a directory standing in for the repo,
/// whose files are named by their paths.
pub(crate) fn record_only(path: &Path, repo: &str, only: &str, file: &str) -> Res<()> {
    let entry = format!("models--{}", repo.replace('/', "--"));
    if let Some(dir) = path.ancestors().find(|a| a.file_name().is_some_and(|n| n.to_string_lossy() == entry)) {
        std::fs::write(dir.join(only), file)?;
    }
    Ok(())
}

/// The checkpoints at the top of a cache entry, `dir`, of `repo`, by the
/// names they are known by: the repo's own, if a pull found it to be all
/// the repo holds.
pub fn locals(dir: &Path, repo: &str) -> Vec<Local> {
    tops(dir)
        .into_iter()
        .filter_map(|(f, file)| Some(Local { name: name_in(dir, ONLY, repo, &f), kind: kind_of(&file)?, file }))
        .collect()
}

/// The checkpoint `name` means, if it is on this machine.
pub fn local(name: &str) -> Option<Local> {
    if is_path(name) {
        let file = std::fs::canonicalize(name).ok()?;
        return kind_of(&file).map(|kind| Local { name: name.to_string(), file, kind });
    }
    let repo = split(name).map_or(name, |(r, _)| r);
    if !repo.contains('/') || crate::gguf::split(name).is_some() {
        return None;
    }
    let dir = crate::hub::cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    locals(&dir, repo).into_iter().find(|l| l.name == name)
}

/// A checkpoint on the Hub: the file `name` means in its repo, found
/// without downloading it, and which kind it is.
#[derive(Debug, Clone)]
pub struct Found {
    pub repo: String,
    pub file: String,
    pub kind: Kind,
    /// Named by the repo alone, and so the repo's only model.
    pub only: bool,
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

/// The most files at a repo's top whose headers [`find`] reads to count
/// its checkpoints; past it, the repo is a collection to name a file of.
pub(crate) const PROBED: usize = 8;

/// Find the checkpoint `name` means on the Hub.
pub fn find(name: &str) -> Res<Found> {
    let not_one = |file: &str, repo: &str| format!("{file} in {repo} is neither an SDXL nor an SD 1.5 checkpoint in Stability's layout, the kinds of single file read here");
    match split(name) {
        Some((repo, file)) => {
            let kind = kind(remote_names(repo, file)?.iter().map(String::as_str)).ok_or_else(|| not_one(file, repo))?;
            Ok(Found { repo: repo.to_string(), file: file.to_string(), kind, only: false })
        }
        None if name.contains('/') && !crate::weights::looks_like_path(name) => {
            let tops = candidates(name)?.ok_or_else(|| format!("{name} is a model of its own, not one checkpoint in a file"))?;
            let list = |files: &[String]| files.iter().map(|f| format!("{name}:{f}")).collect::<Vec<_>>().join(", ");
            if tops.len() > PROBED {
                return Err(format!("{name} has {} .safetensors files at its top; name the checkpoint: {}", tops.len(), list(&tops)).into());
            }
            // Only a checkpoint counts: Pony's repo keeps SDXL's VAE beside
            // its one model, which is no second one.
            let mut found = Vec::new();
            for f in &tops {
                if let Some(k) = kind(remote_names(name, f)?.iter().map(String::as_str)) {
                    found.push((f.clone(), k));
                }
            }
            match found.as_slice() {
                [(file, kind)] => Ok(Found { repo: name.to_string(), file: file.clone(), kind: *kind, only: true }),
                [] if tops.len() == 1 => Err(not_one(&tops[0], name).into()),
                [] => Err(format!("{name} has no SDXL or SD 1.5 checkpoint in Stability's layout among its {} .safetensors files", tops.len()).into()),
                many => {
                    let files: Vec<String> = many.iter().map(|(f, _)| f.clone()).collect();
                    Err(format!("{name} has {} checkpoints at its top; name one: {}", many.len(), list(&files)).into())
                }
            }
        }
        None => Err(format!("`{name}` is not a checkpoint's name").into()),
    }
}

/// Download what [`find`] found, or take it from the cache; and record, for
/// one found as its repo's only model, that it is ([`ONLY`]).
pub fn fetch(found: &Found, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<PathBuf> {
    progress(&format!("fetching {} from {}", found.file, found.repo));
    let path = fetch_file(&found.repo, &found.file, watch)?;
    if found.only {
        record_only(&path, &found.repo, ONLY, &found.file)?;
    }
    Ok(path)
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
    fn a_file_is_known_by_its_unet_and_text_encoder() {
        let unet = "model.diffusion_model.out.2.weight";
        assert_eq!(kind([unet, "conditioner.embedders.1.model.ln_final.weight"]), Some(Kind::Sdxl));
        assert_eq!(kind([unet, "cond_stage_model.transformer.text_model.final_layer_norm.weight"]), Some(Kind::Sd15));
        assert_eq!(kind([unet, "cond_stage_model.model.ln_final.weight"]), None, "SD 2's OpenCLIP");
        assert_eq!(kind(["cond_stage_model.transformer.text_model.final_layer_norm.weight"]), None, "no UNet");
        assert_eq!(kind(["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]), None, "a LoRA");
    }

    /// A repo a pull found to have one checkpoint and nothing else is named
    /// by the repo; one with a checkpoint beside its diffusers folders, or
    /// one file of several fetched and nothing to say there are no others,
    /// by the file; a LoRA beside it is no checkpoint at all.
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
        std::fs::write(root.join("models--o--pony").join(ONLY), "pony.safetensors").unwrap();
        let got = locals(&root.join("models--o--pony"), "o/pony");
        assert_eq!(got.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["o/pony"]);

        // One of a repo's several files, fetched by its name: the cache
        // cannot tell it from a repo's only one, and the record is not there.
        let one_of = root.join("models--o--shaper/snapshots/abc");
        std::fs::create_dir_all(&one_of).unwrap();
        file(&one_of.join("shaper_8.safetensors"), &sdxl);
        let got = locals(&root.join("models--o--shaper"), "o/shaper");
        assert_eq!(got.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["o/shaper:shaper_8.safetensors"]);

        let beside = root.join("models--o--base/snapshots/abc");
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::write(beside.join("model_index.json"), "{}").unwrap();
        file(&beside.join("base.safetensors"), &sdxl);
        file(&beside.join("offset-lora.safetensors"), &["lora_unet_x.lora_down.weight"]);
        let got = locals(&root.join("models--o--base"), "o/base");
        assert_eq!(got.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["o/base:base.safetensors"]);
        assert_eq!(got[0].kind, Kind::Sdxl);

        // SD 1.5's base: its file beside its folders, and the schedule's
        // tables beside its models.
        let sd15 = root.join("models--o--sd15/snapshots/abc");
        std::fs::create_dir_all(&sd15).unwrap();
        std::fs::write(sd15.join("model_index.json"), "{}").unwrap();
        file(&sd15.join("v1-5.safetensors"), &["model.diffusion_model.out.2.weight", "cond_stage_model.transformer.text_model.final_layer_norm.weight", "alphas_cumprod"]);
        let got = locals(&root.join("models--o--sd15"), "o/sd15");
        assert_eq!(got.iter().map(|l| (l.name.as_str(), l.kind)).collect::<Vec<_>>(), [("o/sd15:v1-5.safetensors", Kind::Sd15)]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
