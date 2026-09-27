//! Names for the community's GGUFs of a denoiser: `repo:QUANT`.
//!
//! A GGUF repo holds one file per quantisation, `qwen-image-Q4_K_S.gguf`
//! beside `qwen-image-Q8_0.gguf`, and each file is a model of its own: the
//! Q4 is half the memory of the Q8 and draws a different picture. So a
//! model here is named by the repo and the quantisation together,
//! `city96/Qwen-Image-gguf:Q4_K_S`, as Ollama and the Hub already name them.
//!
//! The file is only the denoiser. The text encoder, the VAE and the
//! scheduler come from the repo it was made from, which its model card
//! names as `base_model`. So a GGUF model is two repos: this module finds
//! the file and the base, and the GPU crate, which knows what each pipeline
//! reads, fetches the rest of the base and loads the two together.
//!
//! Nothing here parses a GGUF. The name, the file and the base are all this
//! crate needs to list, pull and delete one.

use crate::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One GGUF on this machine: the file, and where the rest of its model is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub repo: String,
    /// As the file spells it, which is what the listing shows.
    pub quant: String,
    pub file: PathBuf,
    /// The model card's `base_model`, when its card is here and names one.
    pub base: Option<String>,
}

impl Local {
    /// The model's name: `repo:QUANT`.
    pub fn name(&self) -> String {
        format!("{}:{}", self.repo, self.quant)
    }
}

/// `repo:QUANT` as its two halves, or `None` for any other name.
///
/// The quantisation has to look like one, so that nothing else with a
/// colon in it is taken for a GGUF.
pub fn split(name: &str) -> Option<(&str, &str)> {
    let (repo, quant) = name.rsplit_once(':')?;
    (repo.contains('/') && is_quant(quant)).then_some((repo, quant))
}

/// Whether `s` is a quantisation as GGUF files name them: `Q8_0`, `Q4_K_S`,
/// `IQ4_XS`, `BF16`, `F16`, `F32`, in any case.
fn is_quant(s: &str) -> bool {
    let s = s.to_ascii_uppercase();
    if matches!(s.as_str(), "BF16" | "F16" | "F32" | "FP16" | "FP32") {
        return true;
    }
    let rest = s.strip_prefix("IQ").or_else(|| s.strip_prefix('Q'));
    let Some(rest) = rest else { return false };
    let mut parts = rest.split('_');
    let bits = parts.next().unwrap_or_default();
    !bits.is_empty()
        && bits.chars().all(|c| c.is_ascii_digit())
        && parts.all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric()))
}

/// The quantisation a GGUF file's name ends in: `Q4_K_S` for
/// `qwen-image-Q4_K_S.gguf` and for `Qwen_Image-Q4_K_S.gguf`.
///
/// The longest tail after a `-`, `_` or `.` that is a quantisation, so
/// `Q4_K_S` rather than `K_S`. A file split in parts, `…-00001-of-00003.gguf`,
/// has none: it is not a model on its own, and nothing here joins the parts.
pub fn quant_of(file: &str) -> Option<String> {
    let stem = file.rsplit('/').next()?.strip_suffix(".gguf")?;
    if stem.contains("-of-") {
        return None;
    }
    stem.char_indices()
        .filter(|&(_, c)| matches!(c, '-' | '_' | '.'))
        .map(|(i, _)| &stem[i + 1..])
        .find(|tail| is_quant(tail))
        .map(str::to_string)
}

/// The one file among `files`, paths in a repo, that is `quant`.
pub fn file_for<'a>(files: impl IntoIterator<Item = &'a str>, repo: &str, quant: &str) -> Res<&'a str> {
    let all: Vec<&str> = files.into_iter().filter(|f| f.ends_with(".gguf")).collect();
    let hits: Vec<&str> = all.iter().copied().filter(|f| quant_of(f).is_some_and(|q| q.eq_ignore_ascii_case(quant))).collect();
    match hits.as_slice() {
        [one] => Ok(one),
        [] => {
            let mut have: Vec<String> = all.iter().filter_map(|f| quant_of(f)).collect();
            have.sort();
            have.dedup();
            Err(match have.is_empty() {
                true => format!("{repo} has no GGUF files this can read"),
                false => format!("{repo} has no {quant}; it has {}", have.join(", ")),
            }
            .into())
        }
        many => Err(format!("{repo} has {} files that are {quant}: {}; name one of them some other way", many.len(), many.join(", ")).into()),
    }
}

/// The `base_model` a model card's front matter names: a string, or the
/// first of a list.
pub fn base_model(card: &str) -> Option<String> {
    let front = card.strip_prefix("---")?.split("\n---").next()?;
    let mut lines = front.lines();
    while let Some(line) = lines.next() {
        let Some(value) = line.strip_prefix("base_model:") else { continue };
        let value = value.trim();
        let unquote = |s: &str| s.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        if let Some(list) = value.strip_prefix('[') {
            return list.trim_end_matches(']').split(',').map(unquote).find(|s| !s.is_empty());
        }
        if !value.is_empty() {
            return Some(unquote(value));
        }
        // A block list on the lines after.
        return lines.map(str::trim).take_while(|l| l.starts_with('-')).map(|l| unquote(&l[1..])).find(|s| !s.is_empty());
    }
    None
}

/// Every GGUF in one cache entry, `dir`, of `repo`.
pub fn locals(dir: &Path, repo: &str) -> Vec<Local> {
    let Ok(revisions) = std::fs::read_dir(dir.join("snapshots")) else { return Vec::new() };
    let mut out = Vec::new();
    for rev in revisions.filter_map(|e| e.ok()).map(|e| e.path()) {
        let base = std::fs::read_to_string(rev.join("README.md")).ok().as_deref().and_then(base_model);
        let mut found = Vec::new();
        collect(&rev, &rev, &mut found);
        for (rel, file) in found {
            if let Some(quant) = quant_of(&rel) {
                out.push(Local { repo: repo.to_string(), quant, file, base: base.clone() });
            }
        }
    }
    out.sort_by(|a, b| a.quant.cmp(&b.quant));
    out.dedup_by(|a, b| a.quant.eq_ignore_ascii_case(&b.quant));
    out
}

/// The `.gguf` files under `dir`, with their paths relative to `root`. A
/// snapshot's files are links to blobs, and a link to a blob that is gone
/// is not a file.
fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
        if path.is_dir() {
            collect(root, &path, out);
        } else if path.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) && path.is_file() {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push((rel.to_string_lossy().into_owned(), path));
            }
        }
    }
}

/// The GGUF `name` means, if it is on this machine.
pub fn local(name: &str) -> Option<Local> {
    let (repo, quant) = split(name)?;
    let dir = crate::hub::cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    locals(&dir, repo).into_iter().find(|l| l.quant.eq_ignore_ascii_case(quant))
}

/// A GGUF on the Hub: which file of its repo `name` means, and its base,
/// found without downloading the file.
#[derive(Debug, Clone)]
pub struct Found {
    pub repo: String,
    /// Its path in the repo.
    pub file: String,
    pub quant: String,
    pub base: String,
}

/// Find the GGUF `name` means: the repo's file list from the Hub's API,
/// since the quantisation names a file only by the end of its name, and
/// the base from its model card, which is small and is fetched.
pub fn find(name: &str, watch: &Watcher) -> Res<Found> {
    let (repo, quant) = split(name).ok_or_else(|| format!("`{name}` is not a GGUF's name, which looks like `city96/Qwen-Image-gguf:Q4_K_S`"))?;
    let base = |card: &Path| -> Res<String> {
        base_model(&std::fs::read_to_string(card)?)
            .ok_or_else(|| format!("{repo}'s model card names no `base_model`, so there is no telling which model's text encoder and VAE go with it").into())
    };
    // Here already, card and all: nothing to ask.
    if let Some(l) = local(name) {
        if let (Some(b), Some(rel)) = (l.base.clone(), relative(&l.file)) {
            return Ok(Found { repo: repo.to_string(), file: rel, quant: l.quant, base: b });
        }
    }
    let files = crate::hub::repo_files(repo)?;
    let file = file_for(files.iter().map(String::as_str), repo, quant)?.to_string();
    let card = fetch_file(repo, "README.md", watch).map_err(|e| format!("{repo}'s model card: {e}"))?;
    Ok(Found { repo: repo.to_string(), quant: quant_of(&file).unwrap_or_else(|| quant.to_string()), file, base: base(&card)? })
}

/// A snapshot file's path in its repo: what follows `snapshots/<revision>/`.
fn relative(file: &Path) -> Option<String> {
    let s = file.to_string_lossy();
    let (_, after) = s.split_once("/snapshots/")?;
    Some(after.split_once('/')?.1.to_string())
}

/// Download what [`find`] found, or take it from the cache.
pub fn fetch(found: &Found, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Local> {
    progress(&format!("{}, a {} of {}'s denoiser", found.file, found.quant, found.base));
    let path = fetch_file(&found.repo, &found.file, watch)?;
    Ok(Local { repo: found.repo.clone(), quant: found.quant.clone(), file: path, base: Some(found.base.clone()) })
}

/// Delete one GGUF: its link in the snapshot and the blob it points at,
/// and the whole cache entry once no GGUF or anything else is left in it.
pub fn remove(l: &Local) -> std::io::Result<()> {
    let blob = std::fs::canonicalize(&l.file)?;
    std::fs::remove_file(&l.file)?;
    if blob != l.file {
        std::fs::remove_file(&blob)?;
    }
    // The cache entry is the file's ancestor named for the repo.
    let entry = format!("models--{}", l.repo.replace('/', "--"));
    let Some(dir) = l.file.ancestors().find(|a| a.file_name().is_some_and(|n| n.to_string_lossy() == entry)) else { return Ok(()) };
    if locals(dir, &l.repo).is_empty() && crate::hub::model_file(dir, "config.json").is_none() && crate::hub::model_file(dir, "model_index.json").is_none() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_split_only_when_the_tail_is_a_quantisation() {
        assert_eq!(split("city96/Qwen-Image-gguf:Q4_K_S"), Some(("city96/Qwen-Image-gguf", "Q4_K_S")));
        assert_eq!(split("unsloth/Qwen-Image-2512-GGUF:q8_0"), Some(("unsloth/Qwen-Image-2512-GGUF", "q8_0")));
        assert_eq!(split("a/b:BF16"), Some(("a/b", "BF16")));
        assert_eq!(split("a/b:IQ4_XS"), Some(("a/b", "IQ4_XS")));
        assert_eq!(split("Qwen/Qwen-Image"), None);
        assert_eq!(split("a/b:main"), None);
        assert_eq!(split("nobody:Q4_K_S"), None);
        assert_eq!(split("a/b:Q"), None);
    }

    /// Every spelling of the repos read for #42.
    #[test]
    fn a_files_quantisation_is_the_longest_tail_that_is_one() {
        let q = |f: &str| quant_of(f);
        assert_eq!(q("qwen-image-Q4_K_S.gguf").as_deref(), Some("Q4_K_S"));
        assert_eq!(q("Qwen_Image-Q4_K_S.gguf").as_deref(), Some("Q4_K_S"));
        assert_eq!(q("qwen-image-2512-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(q("flux1-schnell-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(q("LTX-2.5-Distilled-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(q("sub/dir/model.Q5_K_M.gguf").as_deref(), Some("Q5_K_M"));
        assert_eq!(q("big-Q8_0-00001-of-00002.gguf"), None);
        assert_eq!(q("model.safetensors"), None);
        assert_eq!(q("model.gguf"), None);
    }

    #[test]
    fn a_quantisation_names_one_file() {
        let files = ["README.md", "qwen-image-Q4_K_S.gguf", "qwen-image-Q4_K_M.gguf", "qwen-image-Q8_0.gguf"];
        assert_eq!(file_for(files, "r/x", "q4_k_s").unwrap(), "qwen-image-Q4_K_S.gguf");
        let missing = file_for(files, "r/x", "Q2_K").unwrap_err().to_string();
        assert!(missing.contains("has no Q2_K; it has Q4_K_M, Q4_K_S, Q8_0"), "{missing}");
        let twice = ["a-Q8_0.gguf", "b-Q8_0.gguf"];
        assert!(file_for(twice, "r/x", "Q8_0").is_err());
    }

    #[test]
    fn the_base_is_read_from_the_card_however_it_is_written() {
        assert_eq!(base_model("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen-Image\n---\n# x").as_deref(), Some("Qwen/Qwen-Image"));
        assert_eq!(base_model("---\nbase_model:\n- Qwen/Qwen-Image\n- other/one\ntags:\n---\n").as_deref(), Some("Qwen/Qwen-Image"));
        assert_eq!(base_model("---\nbase_model: ['Qwen/Qwen-Image']\n---\n").as_deref(), Some("Qwen/Qwen-Image"));
        assert_eq!(base_model("---\nbase_model: \"black-forest-labs/FLUX.1-dev\"\n---\n").as_deref(), Some("black-forest-labs/FLUX.1-dev"));
        assert_eq!(base_model("# no front matter\nbase_model: a/b\n"), None);
        assert_eq!(base_model("---\nlicense: mit\n---\nbase_model: a/b\n"), None);
    }

    /// A cache entry laid out as `hf-hub` lays one out: links in a
    /// snapshot to blobs, and the card beside them.
    #[test]
    fn a_cache_entry_lists_each_gguf_and_deleting_one_keeps_the_rest() {
        let root = std::env::temp_dir().join(format!("kvad-gguf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("models--o--r-gguf");
        let (blobs, rev) = (dir.join("blobs"), dir.join("snapshots/abc"));
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&rev).unwrap();
        for (name, blob) in [("r-Q8_0.gguf", "b1"), ("r-Q4_K_S.gguf", "b2"), ("README.md", "b3")] {
            let text = if name == "README.md" { "---\nbase_model: o/base\n---\n" } else { "GGUF" };
            std::fs::write(blobs.join(blob), text).unwrap();
            std::os::unix::fs::symlink(blobs.join(blob), rev.join(name)).unwrap();
        }
        let found = locals(&dir, "o/r-gguf");
        assert_eq!(found.iter().map(|l| l.name()).collect::<Vec<_>>(), ["o/r-gguf:Q4_K_S", "o/r-gguf:Q8_0"]);
        assert!(found.iter().all(|l| l.base.as_deref() == Some("o/base")));

        remove(&found[0]).unwrap();
        assert!(!blobs.join("b2").exists(), "the blob went with the link");
        assert_eq!(locals(&dir, "o/r-gguf").iter().map(|l| l.name()).collect::<Vec<_>>(), ["o/r-gguf:Q8_0"]);
        remove(&found[1]).unwrap();
        assert!(!dir.exists(), "the entry went with its last GGUF");
        let _ = std::fs::remove_dir_all(&root);
    }
}
