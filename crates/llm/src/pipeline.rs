//! What an image pipeline in diffusers' layout reads from the Hub.
//!
//! A pipeline's repo is a directory to a model, `unet/` beside
//! `text_encoder/` beside `vae/`, with more in it than any one loader reads:
//! Stability ships SDXL's weights in f32 and again in f16, a VAE the f16
//! pipeline cannot use, and the whole model once more as one file. So which
//! files a model *is* depends on who loads it, and the lists here are the
//! GPU crate's loaders' — it reads its weights through [`weights`] and
//! [`component`], so that what a pull fetches and what a load opens are
//! chosen by the same lines.
//!
//! They live in this crate, which loads none of it, because a pull is not a
//! load: `kvad pull` with no server running has no GPU backend, and fetches
//! a pipeline all the same.

use crate::weights::{fetch_file, read_json, Cached, Watcher};
use serde_json::Value;
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A component's weights, `dir/stem`: the `.fp16` variant where the repo
/// ships one, as Stability's does beside its f32 files, and the plain file
/// otherwise, as nearly every fine-tune does, in f16 already. Either is read
/// in f16, whatever it is stored in.
///
/// The cache is asked first, so a repo that is here asks the Hub nothing;
/// then the Hub, `.fp16` first, so that a repo with both never downloads its
/// f32 file.
pub fn weights(repo: &str, dir: &str, stem: &str, watch: &Watcher) -> Res<PathBuf> {
    let (fp16, plain) = (format!("{dir}/{stem}.fp16.safetensors"), format!("{dir}/{stem}.safetensors"));
    for f in [&fp16, &plain] {
        if let Cached::Here(p) = crate::weights::cached(repo, f) {
            return Ok(p);
        }
    }
    fetch_file(repo, &fp16, watch).or_else(|_| fetch_file(repo, &plain, watch)).map_err(|e| format!("{repo} has neither {fp16} nor {plain}: {e}").into())
}

/// A component's config and weights: a shard index's worth, or one file.
pub fn component(repo: &str, dir: &str, weights: &str, watch: &Watcher) -> Res<(Value, Vec<PathBuf>)> {
    let config = read_json(&fetch_file(repo, &format!("{dir}/config.json"), watch)?)?;
    let paths = match fetch_file(repo, &format!("{dir}/{weights}.safetensors.index.json"), watch) {
        Ok(index) => {
            let map = read_json(&index)?;
            let mut shards: Vec<String> =
                map["weight_map"].as_object().ok_or("a shard index with no weight_map")?.values().filter_map(Value::as_str).map(str::to_string).collect();
            shards.sort();
            shards.dedup();
            shards.iter().map(|s| fetch_file(repo, &format!("{dir}/{s}"), watch)).collect::<Res<Vec<_>>>()?
        }
        Err(_) => vec![fetch_file(repo, &format!("{dir}/{weights}.safetensors"), watch)?],
    };
    Ok((config, paths))
}

/// The base repo ships CLIP's vocabulary as `vocab.json` and `merges.txt`
/// only; this repo has the same vocabulary as a `tokenizer.json`. SDXL's,
/// SD 1.5's and FLUX's CLIP all read it.
pub const CLIP_TOKENIZER_REPO: &str = "openai/clip-vit-large-patch14";

/// SDXL: two text encoders and a UNet of the repo's own, and a VAE that is
/// not.
pub mod sdxl {
    use super::*;

    pub const REPO: &str = "stabilityai/stable-diffusion-xl-base-1.0";

    /// SDXL's own VAE overflows f16 in its decoder, which is why its config says
    /// `force_upcast`. This is the same decoder retrained to stay in range, so
    /// the whole pipeline can run in one dtype. See the plan.
    pub const VAE_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";

    /// What a load reads that is not a model of `repo`'s own: the tokenizer,
    /// `repo`'s configs and scheduler, and the VAE. All a checkpoint in one
    /// file reads besides the file, with [`REPO`] for `repo`.
    pub fn beside(repo: &str, watch: &Watcher) -> Res<Vec<PathBuf>> {
        let mut files = vec![fetch_file(CLIP_TOKENIZER_REPO, "tokenizer.json", watch)?];
        for f in ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "text_encoder_2/config.json", "unet/config.json"] {
            files.push(fetch_file(repo, f, watch)?);
        }
        for f in ["config.json", "diffusion_pytorch_model.safetensors"] {
            files.push(fetch_file(VAE_REPO, f, watch)?);
        }
        Ok(files)
    }

    /// Every file a load reads for `repo`, fetched and not loaded: what its
    /// pull brings. The `.fp16` weights where the repo ships them, as
    /// [`weights`] chooses, and never the repo's own VAE, which the load
    /// does not read.
    pub fn fetch(repo: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Vec<PathBuf>> {
        progress(&format!("fetching {repo}'s configs and {VAE_REPO}'s VAE"));
        let mut files = beside(repo, watch)?;
        for (dir, stem, what) in [("text_encoder", "model", "text encoder"), ("text_encoder_2", "model", "second text encoder"), ("unet", "diffusion_pytorch_model", "UNet")] {
            progress(&format!("fetching the {what}"));
            files.push(weights(repo, dir, stem, watch)?);
        }
        Ok(files)
    }
}

/// SD 1.5: a text encoder, a UNet and a VAE, all the repo's own.
pub mod sd15 {
    use super::*;

    pub const REPO: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";

    /// What a load reads that is not a model: the tokenizer, and `repo`'s
    /// configs and scheduler. All a checkpoint in one file reads besides
    /// the file, with [`REPO`] for `repo`.
    pub fn beside(repo: &str, watch: &Watcher) -> Res<Vec<PathBuf>> {
        let mut files = vec![fetch_file(CLIP_TOKENIZER_REPO, "tokenizer.json", watch)?];
        for f in ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "unet/config.json", "vae/config.json"] {
            files.push(fetch_file(repo, f, watch)?);
        }
        Ok(files)
    }

    /// Every file a load reads for `repo`, fetched and not loaded: what its
    /// pull brings. The `.fp16` weights where the repo ships them, as
    /// [`weights`] chooses.
    pub fn fetch(repo: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Vec<PathBuf>> {
        progress(&format!("fetching {repo}'s configs"));
        let mut files = beside(repo, watch)?;
        for (dir, stem, what) in [("text_encoder", "model", "text encoder"), ("unet", "diffusion_pytorch_model", "UNet"), ("vae", "diffusion_pytorch_model", "VAE")] {
            progress(&format!("fetching the {what}"));
            files.push(weights(repo, dir, stem, watch)?);
        }
        Ok(files)
    }
}

/// FLUX: CLIP, T5, a transformer and a VAE.
pub mod flux {
    use super::*;

    /// Every file of `repo` a load reads but its transformer's weights:
    /// what the pull of a GGUF of its transformer brings of the base. A
    /// component's shard index is read and not listed: which shards there
    /// are is all it says.
    pub fn beside(repo: &str, watch: &Watcher) -> Res<Vec<PathBuf>> {
        let mut files = vec![fetch_file(CLIP_TOKENIZER_REPO, "tokenizer.json", watch)?];
        for f in ["model_index.json", "tokenizer_2/tokenizer.json", "scheduler/scheduler_config.json", "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"] {
            files.push(fetch_file(repo, f, watch)?);
        }
        for dir in ["text_encoder", "text_encoder_2"] {
            files.push(fetch_file(repo, &format!("{dir}/config.json"), watch)?);
            files.extend(component(repo, dir, "model", watch)?.1);
        }
        Ok(files)
    }

    /// Every file a load reads for `repo`, fetched and not loaded: what its
    /// pull brings. The transformer in the repo's own bf16, whatever
    /// quantisation a load then makes of it.
    pub fn fetch(repo: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Vec<PathBuf>> {
        progress(&format!("fetching {repo}'s text encoders and VAE"));
        let mut files = beside(repo, watch)?;
        progress("fetching the transformer");
        files.extend(component(repo, "transformer", "diffusion_pytorch_model", watch)?.1);
        Ok(files)
    }
}

/// Qwen-Image: a language model for a text encoder, a transformer and a VAE.
pub mod qwen {
    use super::*;

    /// The repo ships the tokenizer as `vocab.json` and `merges.txt`; this one
    /// has the same vocabulary as a `tokenizer.json`.
    pub const TOKENIZER_REPO: &str = "Qwen/Qwen2.5-VL-7B-Instruct";

    /// Every file of `repo` a load reads but its transformer's weights:
    /// what the pull of a GGUF of its transformer brings of the base, one
    /// of them the transformer's config, which a GGUF does not carry. A
    /// component's shard index is read and not listed.
    pub fn beside(repo: &str, watch: &Watcher) -> Res<Vec<PathBuf>> {
        let mut files = vec![fetch_file(TOKENIZER_REPO, "tokenizer.json", watch)?];
        for f in ["model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "text_encoder/config.json"] {
            files.push(fetch_file(repo, f, watch)?);
        }
        files.extend(component(repo, "text_encoder", "model", watch)?.1);
        Ok(files)
    }

    /// Every file a load reads for `repo`, fetched and not loaded: what its
    /// pull brings. The transformer in the repo's own bf16, whatever
    /// quantisation a load then makes of it.
    pub fn fetch(repo: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Vec<PathBuf>> {
        progress(&format!("fetching {repo}'s text encoder and VAE"));
        let mut files = beside(repo, watch)?;
        progress("fetching the transformer — twenty billion parameters, in bf16");
        files.extend(component(repo, "transformer", "diffusion_pytorch_model", watch)?.1);
        Ok(files)
    }
}

/// The pipelines there is a pull for, by the `_class_name` their repos'
/// `model_index.json` gives: the ones the GPU crate loads, which names
/// them from here.
pub const PIPELINES: [&str; 4] = ["StableDiffusionXLPipeline", "QwenImagePipeline", "FluxPipeline", "StableDiffusionPipeline"];

/// Fetch every file the load of `repo` reads, a pipeline in diffusers'
/// layout, and load none of it: a pull. The files, its own and the ones its
/// pipeline borrows from other repos; or `None` for a repo with no
/// `model_index.json`, which is not one, and has been asked nothing else.
///
/// A language model's pull asks such a repo for a `config.json` it does not
/// have, which is why this is asked first. For a language model the
/// question is one small request, and none once the cache holds the answer.
pub fn pull(repo: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Option<Vec<PathBuf>>> {
    let Ok(index) = fetch_file(repo, "model_index.json", watch) else { return Ok(None) };
    let v = read_json(&index)?;
    // A pipeline with no implementation is refused, as its load would be:
    // it is not worth its gigabytes.
    let files = match v.get("_class_name").and_then(Value::as_str).unwrap_or("?") {
        "StableDiffusionXLPipeline" => sdxl::fetch(repo, progress, watch)?,
        "StableDiffusionPipeline" => sd15::fetch(repo, progress, watch)?,
        "QwenImagePipeline" => qwen::fetch(repo, progress, watch)?,
        "FluxPipeline" => flux::fetch(repo, progress, watch)?,
        other => return Err(format!("`{repo}` is a {other}; the pipelines implemented here are {}", PIPELINES.join(" and ")).into()),
    };
    Ok(Some(files))
}
