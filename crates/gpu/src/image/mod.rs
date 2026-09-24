//! Text to image, on the GPU.
//!
//! Read `docs/image-plan.md` first: it has every shape used here, read off
//! the checkpoints, and the reasons this lives on candle and nowhere else.
//! Then, in the order a prompt meets them:
//!
//! 1. [`nn`] — the two-dimensional vocabulary a language model never needed.
//! 2. [`clip`] — the text encoders.
//! 3. [`schedule`] — the arithmetic between denoiser calls, which is where
//!    "diffusion" actually happens.
//! 4. [`unet`] — SDXL's denoiser.
//! 5. [`vae`] — latents to pixels.
//! 6. [`sdxl`] — the four put together, as a [`Painter`].
//! 7. [`qwen`] — the same story at twenty billion parameters: an LLM as the
//!    text encoder, a transformer as the denoiser ([`mmdit`]), a video VAE
//!    as the decoder.
//! 8. [`flux`] — the same transformer block again, with [`t5`] and CLIP as
//!    its encoders and a second, single-stream kind of block after it.
//!
//! [`load`] picks the pipeline from the repo's `model_index.json`, the way
//! [`crate::model::session`] picks an architecture from `config.json`.

pub mod clip;
pub mod flux;
pub mod mmdit;
pub mod nn;
pub mod qwen;
pub mod schedule;
pub mod sdxl;
pub mod t5;
pub mod unet;
pub mod vae;

use crate::common::{unread, Reader};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use kvad::image::Painter;
use kvad::serde_json::Value;
use kvad::weights::{fetch_file, Cached, Watcher};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The pipelines this backend implements, by the `_class_name` their repos'
/// `model_index.json` gives.
pub const PIPELINES: [&str; 3] = ["StableDiffusionXLPipeline", "QwenImagePipeline", "FluxPipeline"];

/// What kind of model `repo` is, from its `model_index.json`, if it has one on
/// this machine. `None` for a language model, a missing repo, or a pipeline
/// with no implementation here.
///
/// Asked without downloading anything, because the server asks it about every
/// model on disk each time it lists them.
pub fn pipeline_of(repo: &str) -> Option<&'static str> {
    let index = local_file(repo, "model_index.json")?;
    let v: Value = kvad::serde_json::from_str(&std::fs::read_to_string(index).ok()?).ok()?;
    let class = v.get("_class_name")?.as_str()?;
    PIPELINES.into_iter().find(|p| *p == class)
}

/// Whether `repo` is an image pipeline this backend implements, asking the
/// Hub only when this machine cannot say.
///
/// A language model's repo has no `model_index.json`, so for one that is not
/// here yet the Hub's answer is a 404 — one small request before a download
/// of gigabytes that would have happened anyway. For one that is here, the
/// request is not small: with DNS for the Hub timing out, `hf-hub` retried
/// it for three minutes before giving up. The cache answers instead, when it can:
/// the file marked as missing, or a language model's files all present,
/// which is a repo that is not a pipeline.
pub fn is_pipeline(repo: &str, watch: &Watcher) -> bool {
    if pipeline_of(repo).is_some() {
        return true;
    }
    if kvad::weights::local_dir(repo).is_some() || !repo.contains('/') {
        return false;
    }
    match kvad::weights::cached(repo, "model_index.json") {
        // Here, and `pipeline_of` did not recognise it: a pipeline, but not
        // one implemented here — which is the same answer as a language model.
        Cached::Here(_) | Cached::Absent => return false,
        Cached::Unknown if kvad::weights::in_cache(repo).is_some() => return false,
        Cached::Unknown => {}
    }
    fetch_file(repo, "model_index.json", watch)
        .ok()
        .and_then(|p| read_json(&p).ok())
        .and_then(|v| v.get("_class_name")?.as_str().map(str::to_string))
        .is_some_and(|class| PIPELINES.contains(&class.as_str()))
}

/// What loading `repo` at `quant` will take, from the files on this machine,
/// before anything is loaded — for the server's admission check.
///
/// Each pipeline reads only some of what its repo holds, and in its own
/// precision, so the repo's size on disk is the wrong answer both ways: SDXL
/// is charged for the VAE it borrows from another repo, and Qwen-Image's 58 GB
/// of bf16 is about half that at q8. `None` when the files are not here.
pub fn weight_bytes(repo: &str, quant: Option<candle_core::quantized::GgmlDType>) -> Option<u64> {
    let size = |r: &str, f: &str| local_file(r, f).and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
    match pipeline_of(repo)? {
        "StableDiffusionXLPipeline" => {
            let own = ["text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors"]
                .iter()
                .map(|f| size(repo, f))
                .sum::<Option<u64>>()?;
            // The VAE is f32 on disk and held in f16. Not downloaded yet is
            // not a reason to refuse; it is small.
            let vae = size(sdxl::VAE_REPO, "diffusion_pytorch_model.safetensors").unwrap_or(335_000_000) / 2;
            Some(own + vae)
        }
        "QwenImagePipeline" => qwen::weight_bytes(repo, quant, &size),
        "FluxPipeline" => flux::weight_bytes(repo, quant, &size),
        _ => None,
    }
}

/// `file` from `repo` if it is already on this machine: in a directory
/// standing in for the repo, or in the Hub cache.
pub(crate) fn local_file(repo: &str, file: &str) -> Option<PathBuf> {
    if let Some(dir) = kvad::weights::local_dir(repo) {
        return Some(dir.join(file)).filter(|p| p.is_file());
    }
    let hub = kvad::hub::cache_dir();
    let snapshots = hub.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    std::fs::read_dir(snapshots).ok()?.flatten().map(|e| e.path().join(file)).find(|p| p.is_file())
}

/// Load whichever pipeline `repo` is.
///
/// `quant` is for the pipelines large enough to need it; SDXL ignores it and
/// runs in f16 whatever it says, and says so through `progress`.
pub fn load(
    repo: &str,
    quant: Option<candle_core::quantized::GgmlDType>,
    progress: &mut dyn FnMut(&str),
    watch: &Watcher,
) -> Res<Box<dyn Painter>> {
    let index = fetch_file(repo, "model_index.json", watch)?;
    let v = read_json(&index)?;
    let class = v.get("_class_name").and_then(Value::as_str).unwrap_or("?");
    let device = crate::model::pick_device(None)?;
    if !device.is_metal() && !device.is_cuda() {
        return Err(concat!(
            "text-to-image runs on the GPU and nowhere else, and this machine has none \
             that candle can use. docs/image-plan.md says why there is no CPU path."
        )
        .into());
    }
    match class {
        "StableDiffusionXLPipeline" => {
            if quant.is_some() {
                progress("SDXL runs in f16; ignoring the quantisation asked for");
            }
            Ok(Box::new(sdxl::Sdxl::load(repo, device, progress, watch)?))
        }
        "QwenImagePipeline" => Ok(Box::new(qwen::QwenImage::load(repo, quant, device, progress, watch)?)),
        "FluxPipeline" => Ok(Box::new(flux::Flux::load(repo, quant, device, progress, watch)?)),
        other => Err(format!(
            "`{repo}` is a {other}; the pipelines implemented here are {}",
            PIPELINES.join(" and ")
        )
        .into()),
    }
}

pub(crate) fn read_json(path: &Path) -> Res<Value> {
    Ok(kvad::serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// A [`Reader`] over some safetensors files, mapped on the host and read in
/// `dtype`. Every tensor is moved to the device once it is in its final
/// form, as the text backend does and for the same reason (`Loader::proj`).
pub(crate) fn open(paths: &[PathBuf], dtype: DType) -> Res<Reader<'static>> {
    // SAFETY: candle memory-maps the checkpoints; they are read-only cache
    // entries that nothing else writes while we hold them.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, &Device::Cpu)? };
    Ok(Reader::new(vb))
}

/// Refuse a component whose checkpoint holds weights nothing read, and count
/// the parameters of the ones that were.
///
/// The same guard as the text models'. It matters more here, if anything:
/// a UNet missing one attention block's cross-attention draws a picture that
/// ignores part of the prompt, and nothing about that looks like an error.
pub(crate) fn finish(what: &str, paths: &[PathBuf], r: &Reader<'_>) -> Res<usize> {
    let left = unread(paths, &r.seen(), &r.skipped())?;
    if !left.is_empty() {
        return Err(format!(
            "{what}: the checkpoint holds {} tensor(s) that this loader never reads:\n  {}\n\
             A weight nobody reads is a piece of the model that is not running.",
            left.len(),
            kvad::weights::collapsed(&left).join("\n  ")
        )
        .into());
    }
    // SAFETY: as in `open`.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
    let seen = r.seen();
    Ok(st.tensors().iter().filter(|(n, _)| seen.contains(n)).map(|(_, v)| v.shape().iter().product::<usize>()).sum())
}
