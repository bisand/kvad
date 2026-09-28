//! Names for LoRAs, as for checkpoints in one file.
//!
//! A LoRA is one `.safetensors` file of thin factor pairs, and is named as a
//! checkpoint in one file is ([`crate::checkpoint`]):
//!
//! - **`repo`**, when the repo's only LoRA is one file, as most LoRA repos'
//!   is. As for a checkpoint, only a pull by that name asks the Hub whether
//!   the repo holds nothing else, and records the answer in the cache entry
//!   ([`ONLY`]); the name is used only then.
//! - **`repo:file.safetensors`**, for a repo of several, as
//!   `lightx2v/Qwen-Image-Lightning`'s eleven.
//! - **A path to a file on this machine**, for Civitai's.
//!
//! A file is a LoRA if every tensor in its header is one half of a pair, in
//! one of the [`SPELLINGS`], or the `alpha` beside one. On the Hub the header
//! is read with a range request before anything is fetched. Which model a
//! LoRA adapts is not in the file; a model it does not fit refuses it when a
//! request sets it (`kvad_gpu::image::lora`).

use crate::checkpoint::{candidates, header, is_path, name_in, record_only, remote_names, split, tops, PROBED};
use crate::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The endings a pair's two factors are spelled with, `A` then `B`: PEFT's,
/// kohya's, and diffusers' two older ones, the second its
/// `LoRAAttnProcessor`'s (`attn1.processor.to_q_lora.down.weight`).
pub const SPELLINGS: [(&str, &str); 4] =
    [(".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight"), (".lora.down.weight", ".lora.up.weight"), ("_lora.down.weight", "_lora.up.weight")];

/// The file, at the top of a cache entry, that names the LoRA a pull by the
/// repo's name found to be the repo's only one. As
/// [`crate::checkpoint::ONLY`], and apart from it.
pub const ONLY: &str = "kvad-only-lora";

/// A LoRA on this machine, by the name it is known by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub name: String,
    pub file: PathBuf,
    /// The pipeline it was made for, as far as its names say ([`adapts`]).
    pub adapts: Option<&'static str>,
}

/// The pipeline a LoRA with these tensors was made for, as far as its
/// layers' names say, or `None` where they do not. The file does not say,
/// and a model it does not fit refuses it whatever this answers; this is
/// for offering a model the LoRAs likely to fit it. In order:
///
/// - a UNet's blocks, in `ldm`'s names or diffusers', are SD 1.5's or
///   SDXL's: SDXL's where there is a second text encoder or a transformer
///   deeper than one layer, SD 1.5's where the top level attends, which
///   SDXL's does not;
/// - FLUX's have single-stream blocks, in diffusers' names or Black Forest
///   Labs', whose double blocks name `img_mlp` and `img_mod` as Qwen-Image's
///   do, so they are asked about first;
/// - Qwen-Image's double blocks name each stream's MLP and modulation;
/// - what is left with `attn1` or an audio stream under `transformer_blocks`
///   is LTX-2.5's.
pub fn adapts<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<&'static str> {
    let names: Vec<&str> = names.into_iter().collect();
    let any = |marks: &[&str]| names.iter().any(|n| marks.iter().any(|m| n.contains(m)));
    if any(&["input_blocks", "output_blocks", "middle_block", "down_blocks", "up_blocks", "mid_block"]) {
        if any(&["lora_te2_", "text_encoder_2", "transformer_blocks_1_", "transformer_blocks.1."]) {
            return Some("StableDiffusionXLPipeline");
        }
        if any(&["input_blocks_1_1_", "input_blocks.1.1.", "down_blocks_0_attentions", "down_blocks.0.attentions"]) {
            return Some("StableDiffusionPipeline");
        }
        return None;
    }
    if any(&["single_transformer_blocks", "single_blocks", "double_blocks"]) {
        return Some("FluxPipeline");
    }
    if any(&["img_mlp", "txt_mlp", "img_mod", "txt_mod"]) {
        return Some("QwenImagePipeline");
    }
    if any(&["transformer_blocks"]) && any(&["attn1", "audio"]) {
        return Some(crate::video::LTX_PIPELINE);
    }
    None
}

/// Whether a file with these tensors is a LoRA: every one a factor or an
/// `alpha`, and at least one pair.
pub fn is_lora<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    let mut pairs = 0;
    for n in names {
        if SPELLINGS.iter().any(|(a, _)| n.ends_with(a)) {
            pairs += 1;
        } else if !(SPELLINGS.iter().any(|(_, b)| n.ends_with(b)) || n.ends_with(".alpha")) {
            return false;
        }
    }
    pairs > 0
}

/// The LoRA at `path`, if it is one, and what it adapts.
fn read(path: &Path) -> Option<Option<&'static str>> {
    let h = header(path)?;
    is_lora(h.keys().map(String::as_str)).then(|| adapts(h.keys().map(String::as_str)))
}

/// What the LoRA at `path` holds on the device while it is set: its factors
/// in half precision, which is what a pipeline keeps them in.
pub fn device_bytes(path: &Path) -> Option<u64> {
    let h = header(path)?;
    let factor = |n: &str| SPELLINGS.iter().any(|(a, b)| n.ends_with(a) || n.ends_with(b));
    let elements = |v: &serde_json::Value| -> u64 { v["shape"].as_array().map_or(0, |s| s.iter().filter_map(|d| d.as_u64()).product()) };
    Some(h.iter().filter(|(n, _)| factor(n)).map(|(_, v)| elements(v) * 2).sum())
}

/// The LoRAs at the top of a cache entry, `dir`, of `repo`, by the names
/// they are known by.
pub fn locals(dir: &Path, repo: &str) -> Vec<Local> {
    tops(dir).into_iter().filter_map(|(f, file)| Some(Local { name: name_in(dir, ONLY, repo, &f), adapts: read(&file)?, file })).collect()
}

/// The LoRA `name` means, if it is on this machine. Never asks the Hub.
pub fn local(name: &str) -> Option<Local> {
    if is_path(name) {
        let file = std::fs::canonicalize(name).ok()?;
        return Some(Local { name: name.to_string(), adapts: read(&file)?, file });
    }
    let repo = split(name).map_or(name, |(r, _)| r);
    if !repo.contains('/') || crate::gguf::split(name).is_some() {
        return None;
    }
    let dir = crate::hub::cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    locals(&dir, repo).into_iter().find(|l| l.name == name)
}

/// A LoRA on the Hub: the file `name` means in its repo, found without
/// downloading it.
#[derive(Debug, Clone)]
pub struct Found {
    pub repo: String,
    pub file: String,
    /// Named by the repo alone, and so the repo's only LoRA.
    pub only: bool,
}

/// Find the LoRA `name` means on the Hub.
pub fn find(name: &str) -> Res<Found> {
    match split(name) {
        Some((repo, file)) => {
            if !is_lora(remote_names(repo, file)?.iter().map(String::as_str)) {
                return Err(format!("{file} in {repo} is not a LoRA: not every tensor in it is a factor of a pair or an alpha").into());
            }
            Ok(Found { repo: repo.to_string(), file: file.to_string(), only: false })
        }
        None if name.contains('/') && !crate::weights::looks_like_path(name) => {
            let tops = candidates(name)?.ok_or_else(|| format!("{name} is a model of its own, not a LoRA"))?;
            let list = |files: &[String]| files.iter().map(|f| format!("{name}:{f}")).collect::<Vec<_>>().join(", ");
            if tops.len() > PROBED {
                return Err(format!("{name} has {} .safetensors files at its top; name the LoRA: {}", tops.len(), list(&tops)).into());
            }
            let mut found = Vec::new();
            for f in &tops {
                if is_lora(remote_names(name, f)?.iter().map(String::as_str)) {
                    found.push(f.clone());
                }
            }
            match found.as_slice() {
                [file] => Ok(Found { repo: name.to_string(), file: file.clone(), only: true }),
                [] => Err(format!("{name} has no LoRA among its {} .safetensors files", tops.len()).into()),
                many => Err(format!("{name} has {} LoRAs at its top; name one: {}", many.len(), list(many)).into()),
            }
        }
        None => Err(format!("`{name}` is not a LoRA's name").into()),
    }
}

/// Download what [`find`] found, or take it from the cache; and record, for
/// one found as its repo's only LoRA, that it is ([`ONLY`]).
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

    /// Each kind's names, as the most downloaded LoRAs spell them.
    #[test]
    fn a_lora_says_what_it_adapts_by_its_names() {
        let a = |n: &[&str]| adapts(n.iter().copied());
        assert_eq!(a(&["transformer_blocks.0.img_mlp.net.2.lora_down.weight"]), Some("QwenImagePipeline"), "Lightning");
        assert_eq!(a(&["lora_unet_input_blocks_4_1_proj_in.lora_down.weight", "lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight"]), Some("StableDiffusionXLPipeline"), "pixel-art-xl");
        assert_eq!(a(&["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"]), Some("StableDiffusionPipeline"));
        assert_eq!(a(&["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"]), Some("FluxPipeline"));
        assert_eq!(a(&["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight"]), Some("FluxPipeline"));
        assert_eq!(a(&["lora_unet_double_blocks_0_img_mlp_0.lora_down.weight", "lora_unet_double_blocks_0_img_mod_lin.lora_down.weight"]), Some("FluxPipeline"), "kohya's FLUX, with Qwen-Image's words in it");
        assert_eq!(a(&["diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight"]), Some(crate::video::LTX_PIPELINE), "LTX's IC-LoRA");
        assert_eq!(a(&["blocks.0.self_attn.q.lora_A.weight"]), None);
    }

    #[test]
    fn a_lora_is_pairs_and_alphas_and_nothing_else() {
        assert!(is_lora(["a.to_q.lora_A.weight", "a.to_q.lora_B.weight"]));
        assert!(is_lora(["lora_unet_a.lora_down.weight", "lora_unet_a.lora_up.weight", "lora_unet_a.alpha"]));
        assert!(is_lora(["a.lora.down.weight", "a.lora.up.weight"]));
        assert!(!is_lora(["a.alpha"]), "no pair");
        assert!(!is_lora(["a.lora_A.weight", "a.lora_B.weight", "a.hada_w1_a"]), "a LoHa's");
        assert!(!is_lora(["model.diffusion_model.out.2.weight"]), "a checkpoint");
    }

    /// A repo's LoRA is named by the repo only where a pull recorded it as
    /// the repo's one; a checkpoint beside it is not a LoRA; and what one
    /// holds on the device is its factors in half precision, not its alphas.
    #[test]
    fn a_cache_entry_names_its_loras() {
        let root = std::env::temp_dir().join(format!("kvad-lora-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let file = |path: &Path, tensors: &[(&str, &[u64])]| {
            let header: serde_json::Map<String, serde_json::Value> = tensors
                .iter()
                .map(|(n, shape)| (n.to_string(), serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [0, 2] })))
                .collect();
            let h = serde_json::to_vec(&header).unwrap();
            let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
            bytes.extend(h);
            bytes.extend([0, 0]);
            std::fs::write(path, bytes).unwrap();
        };
        let pair: &[(&str, &[u64])] = &[("x.lora_down.weight", &[4, 16]), ("x.lora_up.weight", &[8, 4]), ("x.alpha", &[])];
        let snap = root.join("models--o--style/snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        file(&snap.join("style.safetensors"), pair);
        file(&snap.join("model.safetensors"), &[("model.diffusion_model.out.2.weight", &[1])]);
        let dir = root.join("models--o--style");
        let names = |l: Vec<Local>| l.into_iter().map(|l| l.name).collect::<Vec<_>>();
        assert_eq!(names(locals(&dir, "o/style")), ["o/style:style.safetensors"], "not recorded, so named by its file");
        std::fs::write(dir.join(ONLY), "style.safetensors").unwrap();
        assert_eq!(names(locals(&dir, "o/style")), ["o/style"]);
        assert_eq!(device_bytes(&snap.join("style.safetensors")), Some((4 * 16 + 8 * 4) * 2));
        let _ = std::fs::remove_dir_all(&root);
    }
}
