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
pub mod dataset;
pub(crate) mod edit;
pub mod flux;
pub mod mmdit;
pub mod nn;
pub mod qwen;
pub mod schedule;
pub mod sd15;
pub mod sdxl;
pub mod lora;
pub(crate) mod single;
pub mod t5;
pub mod tune;
pub mod unet;
pub mod vae;

use crate::common::{unread, Reader};
use crate::gguf::Gguf;
use crate::uncached;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use kvad::checkpoint::Kind;
use kvad::image::Painter;
use kvad::serde_json::Value;
use kvad::weights::{fetch_file, Cached, Watcher};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The pipelines this backend implements, by the `_class_name` their repos'
/// `model_index.json` gives.
pub const PIPELINES: [&str; 4] = ["StableDiffusionXLPipeline", "QwenImagePipeline", "FluxPipeline", "StableDiffusionPipeline"];

/// The pipelines whose denoiser can be read from a community GGUF.
pub const GGUF_PIPELINES: [&str; 2] = ["QwenImagePipeline", "FluxPipeline"];

/// What kind of model `repo` is, from its `model_index.json`, if it has one on
/// this machine. `None` for a language model, a missing repo, or a pipeline
/// with no implementation here.
///
/// Asked without downloading anything, because the server asks it about every
/// model on disk each time it lists them.
pub fn pipeline_of(repo: &str) -> Option<&'static str> {
    // A checkpoint in one file: its header says which.
    if let Some(c) = kvad::checkpoint::local(repo) {
        return Some(c.kind.pipeline());
    }
    // A GGUF of a denoiser is its base's pipeline.
    if kvad::gguf::split(repo).is_some() {
        return pipeline_of(&kvad::gguf::local(repo)?.base?);
    }
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
    // A checkpoint in one file, not here: its header on the Hub says.
    if kvad::checkpoint::split(repo).is_some() {
        return kvad::checkpoint::find(repo).is_ok();
    }
    // A GGUF not here, or whose base is not: its card names the base, and
    // the base says.
    if let Some((gguf_repo, _)) = kvad::gguf::split(repo) {
        let base = kvad::gguf::local(repo).and_then(|l| l.base).or_else(|| {
            let card = fetch_file(gguf_repo, "README.md", watch).ok()?;
            kvad::gguf::base_model(&std::fs::read_to_string(card).ok()?)
        });
        return base.is_some_and(|b| is_pipeline(&b, watch));
    }
    if kvad::weights::local_dir(repo).is_some() || !repo.contains('/') {
        return false;
    }
    match kvad::weights::cached(repo, "model_index.json") {
        // Here, and `pipeline_of` did not recognise it: a pipeline, but not
        // one implemented here — which is the same answer as a language model.
        Cached::Here(_) => return false,
        // No index, and no config here either: a repo whose only model is
        // one checkpoint, not yet pulled, or nothing. A language model has
        // its config, and never pays for the question.
        Cached::Absent if matches!(kvad::weights::cached(repo, "config.json"), Cached::Here(_)) => return false,
        Cached::Absent => return kvad::checkpoint::find(repo).is_ok(),
        Cached::Unknown if kvad::weights::in_cache(repo).is_some() => return false,
        Cached::Unknown => {}
    }
    match fetch_file(repo, "model_index.json", watch) {
        Ok(p) => read_json(&p)
            .ok()
            .and_then(|v| v.get("_class_name")?.as_str().map(str::to_string))
            .is_some_and(|class| PIPELINES.contains(&class.as_str())),
        // No model index: a repo whose only model is one checkpoint, or
        // not a pipeline at all. One request to the Hub for its files says.
        Err(_) => kvad::checkpoint::find(repo).is_ok(),
    }
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
    if let Some(c) = kvad::checkpoint::local(repo) {
        return match c.kind {
            Kind::Sdxl => sdxl::single_weight_bytes(&c.file, &size),
            Kind::Sd15 => sd15::single_weight_bytes(&c.file),
        };
    }
    if kvad::gguf::split(repo).is_some() {
        let g = kvad::gguf::local(repo)?;
        let base = g.base?;
        return match pipeline_of(&base)? {
            "QwenImagePipeline" => qwen::weight_bytes(&base, quant, &size, Some(&g.file)),
            "FluxPipeline" => flux::weight_bytes(&base, quant, &size, Some(&g.file)),
            _ => None,
        };
    }
    match pipeline_of(repo)? {
        "StableDiffusionXLPipeline" => sdxl::weight_bytes(repo, &size),
        "StableDiffusionPipeline" => sd15::weight_bytes(repo),
        "QwenImagePipeline" => qwen::weight_bytes(repo, quant, &size, None),
        "FluxPipeline" => flux::weight_bytes(repo, quant, &size, None),
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
    load_with(repo, None, quant, progress, watch)
}

/// [`load`], with the denoiser read from `gguf`, a community quantisation
/// of it, and everything else from `repo`.
pub fn load_with(
    repo: &str,
    gguf: Option<&Path>,
    quant: Option<candle_core::quantized::GgmlDType>,
    progress: &mut dyn FnMut(&str),
    watch: &Watcher,
) -> Res<Box<dyn Painter>> {
    // A GGUF by its name, `repo:QUANT`: the file, and its base for the rest.
    if kvad::gguf::split(repo).is_some() {
        if gguf.is_some() {
            return Err(format!("`{repo}` is a GGUF already, and another was given beside it").into());
        }
        let found = kvad::gguf::find(repo, watch)?;
        gguf_pipeline(repo, &found.base, watch)?;
        let g = kvad::gguf::fetch(&found, progress, watch)?;
        progress(&format!("the rest is {}'s", found.base));
        return load_with(&found.base, Some(&g.file), quant, progress, watch);
    }
    // A checkpoint in one file: `repo`, `repo:file.safetensors`, or a path.
    // Its kind's base gives its configs.
    let single = match kvad::checkpoint::local(repo) {
        Some(c) => Some((c.file, c.kind)),
        None if kvad::checkpoint::split(repo).is_some() || kvad::checkpoint::is_path(repo) => {
            let found = kvad::checkpoint::find(repo)?;
            Some((kvad::checkpoint::fetch(&found, progress, watch)?, found.kind))
        }
        None => None,
    };
    let index = match (single, fetch_file(repo, "model_index.json", watch)) {
        (None, Ok(index)) => index,
        (Some((file, kind)), _) => return load_single(&file, kind, progress, watch),
        // No model index: one checkpoint, if the repo's files say so.
        (None, Err(e)) => match kvad::checkpoint::find(repo) {
            Ok(found) => return load_single(&kvad::checkpoint::fetch(&found, progress, watch)?, found.kind, progress, watch),
            Err(_) => return Err(e),
        },
    };
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
    if gguf.is_some() && !GGUF_PIPELINES.contains(&class) {
        return Err(format!("`{repo}` is a {class}, and a GGUF's denoiser is read for {} only, so far", GGUF_PIPELINES.join(" and ")).into());
    }
    match class {
        "StableDiffusionXLPipeline" => {
            if quant.is_some() {
                progress("SDXL runs in f16; ignoring the quantisation asked for");
            }
            Ok(Box::new(sdxl::Sdxl::load(repo, device, progress, watch)?))
        }
        "StableDiffusionPipeline" => {
            if quant.is_some() {
                progress("SD 1.5 runs in f16; ignoring the quantisation asked for");
            }
            Ok(Box::new(sd15::Sd15::load(repo, device, progress, watch)?))
        }
        "QwenImagePipeline" => Ok(Box::new(qwen::QwenImage::load_with(repo, gguf, quant, device, progress, watch)?)),
        "FluxPipeline" => Ok(Box::new(flux::Flux::load_with(repo, gguf, quant, device, progress, watch)?)),
        other => Err(format!(
            "`{repo}` is a {other}; the pipelines implemented here are {}",
            PIPELINES.join(" and ")
        )
        .into()),
    }
}

/// A checkpoint in one file, on this machine, by its kind's pipeline, with
/// the configs of that kind's base.
fn load_single(file: &Path, kind: Kind, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Box<dyn Painter>> {
    let device = crate::model::pick_device(None)?;
    Ok(match kind {
        Kind::Sdxl => Box::new(sdxl::Sdxl::load_with(sdxl::REPO, Some(file), device, progress, watch)?),
        Kind::Sd15 => Box::new(sd15::Sd15::load_with(sd15::REPO, Some(file), device, progress, watch)?),
    })
}

/// What a checkpoint of `kind` reads besides its file.
fn fetch_base(kind: Kind, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<()> {
    match kind {
        Kind::Sdxl => sdxl::fetch_base(progress, watch),
        Kind::Sd15 => sd15::fetch_base(progress, watch),
    }
}

/// Fetch a checkpoint in one file, and what it reads beside it, or a LoRA,
/// without loading either: a pull. The header is read on the Hub first, so
/// a file that is neither costs a few small requests.
pub fn pull_single(name: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<()> {
    // Here already, or a file on this machine: only what goes beside it.
    if let Some(c) = kvad::checkpoint::local(name) {
        return fetch_base(c.kind, progress, watch);
    }
    if kvad::lora::local(name).is_some() {
        progress("a LoRA, on this machine already; nothing to fetch");
        return Ok(());
    }
    if kvad::checkpoint::is_path(name) {
        return Err(format!("{name} is neither an SDXL or SD 1.5 checkpoint in Stability's layout nor a LoRA").into());
    }
    // A checkpoint, and if its header says it is none, a LoRA: which needs
    // nothing beside it, the model it is applied to having its own.
    match kvad::checkpoint::find(name) {
        Ok(found) => {
            kvad::checkpoint::fetch(&found, progress, watch)?;
            fetch_base(found.kind, progress, watch)
        }
        Err(not_checkpoint) => match kvad::lora::find(name) {
            Ok(found) => kvad::lora::fetch(&found, progress, watch).map(|_| ()),
            Err(not_lora) => Err(format!("{not_checkpoint}; and as a LoRA: {not_lora}").into()),
        },
    }
}

/// Fetch every file the load of `repo` reads, a pipeline in diffusers'
/// layout, a directory to a model, and load none of it: a pull. The files,
/// its own and the ones its pipeline borrows from other repos; or `None`
/// for a repo with no `model_index.json`, which is not one, and has been
/// asked nothing else.
///
/// A language model's pull asks such a repo for a `config.json` it does not
/// have, which is why this is asked first. For a language model the
/// question is one small request, and none once the cache holds the answer.
pub fn pull_pipeline(repo: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Option<Vec<PathBuf>>> {
    let Ok(index) = fetch_file(repo, "model_index.json", watch) else { return Ok(None) };
    let v = read_json(&index)?;
    // The same four arms as [`load_with`]'s, and the same refusal: a
    // pipeline with no implementation here is not worth its gigabytes.
    let files = match v.get("_class_name").and_then(Value::as_str).unwrap_or("?") {
        "StableDiffusionXLPipeline" => sdxl::fetch(repo, progress, watch)?,
        "StableDiffusionPipeline" => sd15::fetch(repo, progress, watch)?,
        "QwenImagePipeline" => qwen::fetch(repo, progress, watch)?,
        "FluxPipeline" => flux::fetch(repo, progress, watch)?,
        other => return Err(format!("`{repo}` is a {other}; the pipelines implemented here are {}", PIPELINES.join(" and ")).into()),
    };
    Ok(Some(files))
}

/// Fetch a GGUF, `repo:QUANT`, and everything else its model reads from
/// its base, without loading any of it: a pull.
///
/// The base is asked what it is before anything large is fetched, so a GGUF
/// of a model not implemented here costs a model card, not gigabytes.
pub fn pull(name: &str, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<()> {
    let found = kvad::gguf::find(name, watch)?;
    // LTX-2.5 is no diffusers pipeline, and is known by its name.
    if found.base.eq_ignore_ascii_case(crate::video::LTX_REPO) {
        kvad::gguf::fetch(&found, progress, watch)?;
        return crate::video::ltx::fetch_base(progress, watch);
    }
    let pipeline = gguf_pipeline(name, &found.base, watch)?;
    kvad::gguf::fetch(&found, progress, watch)?;
    match pipeline {
        "QwenImagePipeline" => qwen::fetch_base(&found.base, progress, watch),
        "FluxPipeline" => flux::fetch_base(&found.base, progress, watch),
        other => Err(format!("no pull is written for a GGUF of a {other}").into()),
    }
}

/// A GGUF of `base`'s denoiser, under the names `base`'s loader asks for:
/// FLUX's mapped from Black Forest Labs' layout, LTX-2.5's under its DiT
/// file's prefix, Qwen-Image's as it is.
pub fn open_gguf(path: &Path, base: &str, watch: &Watcher) -> Res<Gguf> {
    let file = Gguf::open(path)?;
    match file.text("general.architecture") {
        Some("flux") => flux::open_gguf_for(base, path, watch),
        Some("ltxv") => crate::video::open_dit_gguf(path),
        _ => Ok(file),
    }
}

/// The pipeline a GGUF's base is, if it is one whose denoiser can be read
/// from a GGUF here.
fn gguf_pipeline(name: &str, base: &str, watch: &Watcher) -> Res<&'static str> {
    let index = fetch_file(base, "model_index.json", watch).map_err(|e| format!("{name} is a GGUF of {base}, and {base} is not an image pipeline: {e}"))?;
    let class = read_json(&index)?.get("_class_name").and_then(Value::as_str).map(str::to_string).unwrap_or_default();
    match GGUF_PIPELINES.iter().find(|p| **p == class) {
        Some(&"FluxPipeline") => {
            flux::runs(base, watch).map_err(|e| format!("{name} is a GGUF of {base}: {e}"))?;
            Ok("FluxPipeline")
        }
        Some(p) => Ok(p),
        None => Err(format!("{name} is a GGUF of {base}, a {class}, and a GGUF's denoiser is read for {} only, so far", GGUF_PIPELINES.join(" and ")).into()),
    }
}

pub(crate) fn read_json(path: &Path) -> Res<Value> {
    Ok(kvad::serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// A [`Reader`] over some safetensors files, read on the host in `dtype`
/// ([`Uncached`] says how). Every tensor is moved to the device once it is
/// in its final form, as the text backend does and for the same reason
/// (`Loader::proj`).
///
/// `dtype` is the one the weights are kept in: the compute dtype. A dense
/// matrix read in any other has to be cast after `Loader::proj` has uploaded
/// it, and on Metal that costs more than twice the model. candle 0.11 never
/// reuses an upload's buffer, and frees a dropped one only at
/// `synchronize()`, so every f32 upload stays resident until the load is
/// over; and the cast's output is a pool buffer, rounded up to a power of
/// two. FLUX's T5, 9.5 GB in bf16, peaked at 32.5 GB read as f32 and at
/// 10.0 GB read as bf16; the whole bf16 pipeline, 33.7 GB of weights, could
/// not load on a 48 GB machine and now peaks at 35.0 GB. Quantising needs
/// no f32 reader either: `QTensor::quantize` widens its input itself.
pub(crate) fn open(paths: &[PathBuf], dtype: DType) -> Res<Reader<'static>> {
    let vb = VarBuilder::from_backend(Box::new(Uncached::open(paths)?), dtype, Device::Cpu);
    Ok(Reader::new(vb))
}

/// One file in another layout, read under the names a loader asks for: each
/// name a tensor of the file, some of its rows, or its transpose
/// ([`single`]). The file is opened once and shared by every reader of it.
pub(crate) fn open_file(path: &Path) -> Res<std::sync::Arc<Uncached>> {
    Ok(std::sync::Arc::new(Uncached::open(&[path.to_path_buf()])?))
}

/// A [`Reader`] over `file`, under `map`'s names, in `dtype`.
pub(crate) fn open_mapped(file: &std::sync::Arc<Uncached>, map: single::Map, dtype: DType) -> Reader<'static> {
    let vb = VarBuilder::from_backend(Box::new(Mapped { file: std::sync::Arc::clone(file), map }), dtype, Device::Cpu);
    Reader::new(vb)
}

/// [`finish`], for a file read through maps: every tensor in it read
/// through one of `parts`, or listed in `unread` as deliberately not. The
/// parameters of what was read.
pub(crate) fn finish_mapped(what: &str, file: &Uncached, parts: &[(&single::Map, &Reader<'_>)], unread: &[String]) -> Res<usize> {
    // What each loader read, and what it knows of and leaves under a
    // prefix it skips: CLIP-L's last layer, which SDXL does not read.
    // Only what was read is counted, as `finish` counts it.
    let (mut read, mut known) = (std::collections::HashSet::new(), std::collections::HashSet::new());
    for (map, r) in parts {
        let (seen, skipped) = (r.seen(), r.skipped());
        for (name, src) in map.iter() {
            if seen.contains(name) {
                read.insert(src.name.clone());
                known.insert(src.name.clone());
            } else if skipped.iter().any(|p| name.starts_with(p.as_str())) {
                known.insert(src.name.clone());
            }
        }
    }
    let mut left: Vec<String> = file.tensors.keys().filter(|n| !known.contains(*n) && !unread.contains(n)).cloned().collect();
    left.sort();
    refuse_unread(what, left)?;
    Ok(read.iter().filter_map(|n| file.tensors.get(n)).map(|t| t.shape.iter().product::<usize>()).sum())
}

struct Mapped {
    file: std::sync::Arc<Uncached>,
    map: single::Map,
}

impl SimpleBackend for Mapped {
    fn get(&self, s: Shape, name: &str, _: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &s {
            let msg = format!("shape mismatch for {name}");
            return Err(candle_core::Error::UnexpectedShape { msg, expected: s, got: t.shape().clone() }.bt());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let src = self.map.get(name).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        let mut t = self.file.load(&src.name)?;
        if let Some(rows) = &src.rows {
            t = t.narrow(0, rows.start, rows.len())?;
        }
        if src.transpose {
            t = t.t()?;
        }
        // Squeezing leaves a dimension that is not 1 as it is, and `get`'s
        // check of the shape then says so.
        if src.matrix {
            t = t.squeeze(3)?.squeeze(2)?;
        }
        t.contiguous()?.to_dtype(dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}

/// Safetensors files read a tensor at a time, past the page cache.
///
/// [`crate::uncached`] says why: read through candle's memory map, FLUX at
/// bf16 loaded in 94 s, compressed 41 GB of its own weights to make room for
/// the file pages, and took 29 s to encode its first image and 31 s for the
/// first step. Read this way it loads in 44 s, nothing is compressed, and
/// those take 0.33 and 1.96 s; the price is a tensor's buffers on the host,
/// 0.3 GB at the peak. The headers are parsed here rather than by
/// `safetensors`, which wants the whole file in memory to read one.
pub(crate) struct Uncached {
    files: Vec<File>,
    tensors: HashMap<String, Stored>,
}

/// Where one tensor's bytes are, and what they are.
struct Stored {
    file: usize,
    dtype: DType,
    shape: Vec<usize>,
    at: u64,
    len: usize,
}

impl Uncached {
    fn open(paths: &[PathBuf]) -> Res<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for (i, path) in paths.iter().enumerate() {
            let file = uncached::open(path)?;
            // An eight-byte little-endian header length, the header as JSON,
            // then the data, which the header's offsets count from.
            let n = u64::from_le_bytes(uncached::read(&file, 0, 8)?[..].try_into()?);
            let header: Value = kvad::serde_json::from_slice(&uncached::read(&file, 8, usize::try_from(n)?)?)?;
            let bad = |what: &str| format!("{}: a safetensors header with {what}", path.display());
            for (name, t) in header.as_object().ok_or_else(|| bad("no tensors"))? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match t["dtype"].as_str() {
                    Some("BF16") => DType::BF16,
                    Some("F16") => DType::F16,
                    Some("F32") => DType::F32,
                    Some("F64") => DType::F64,
                    Some("U8") => DType::U8,
                    Some("U32") => DType::U32,
                    Some("I32") => DType::I32,
                    Some("I64") => DType::I64,
                    other => return Err(bad(&format!("`{name}` in {other:?}, which is not read here")).into()),
                };
                let shape = t["shape"].as_array().ok_or_else(|| bad(&format!("no shape for `{name}`")))?;
                let shape = shape.iter().map(|d| d.as_u64().map(|d| d as usize)).collect::<Option<Vec<_>>>();
                let offsets = t["data_offsets"].as_array().map(|o| o.iter().filter_map(Value::as_u64).collect::<Vec<_>>());
                let (Some(shape), Some(&[start, end])) = (shape, offsets.as_deref()) else {
                    return Err(bad(&format!("a malformed entry for `{name}`")).into());
                };
                let len = (end - start) as usize;
                if len != shape.iter().product::<usize>() * dtype.size_in_bytes() {
                    return Err(bad(&format!("`{name}` {len} bytes long, which is not its shape")).into());
                }
                tensors.insert(name.clone(), Stored { file: i, dtype, shape, at: 8 + n + start, len });
            }
            files.push(file);
        }
        Ok(Uncached { files, tensors })
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The bytes `name` is stored in, or none if the file has no such tensor.
    fn bytes_of(&self, name: &str) -> usize {
        self.tensors.get(name).map_or(0, |t| t.len)
    }

    fn load(&self, name: &str) -> candle_core::Result<Tensor> {
        let t = self.tensors.get(name).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        let bytes = uncached::read(&self.files[t.file], t.at, t.len)?;
        Tensor::from_raw_buffer(&bytes, t.dtype, &t.shape, &Device::Cpu)
    }
}

impl SimpleBackend for Uncached {
    fn get(&self, s: Shape, name: &str, _: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &s {
            let msg = format!("shape mismatch for {name}");
            return Err(candle_core::Error::UnexpectedShape { msg, expected: s, got: t.shape().clone() }.bt());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name)?.to_dtype(dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// Refuse a component whose checkpoint holds weights nothing read, and count
/// the parameters of the ones that were.
///
/// The same guard as the text models'. It matters more here, if anything:
/// a UNet missing one attention block's cross-attention draws a picture that
/// ignores part of the prompt, and nothing about that looks like an error.
pub(crate) fn finish(what: &str, paths: &[PathBuf], r: &Reader<'_>) -> Res<usize> {
    refuse_unread(what, unread(paths, &r.seen(), &r.skipped())?)?;
    let st = Uncached::open(paths)?;
    let seen = r.seen();
    Ok(st.tensors.iter().filter(|(n, _)| seen.contains(*n)).map(|(_, t)| t.shape.iter().product::<usize>()).sum())
}

/// [`finish`], for a component read from a GGUF.
pub(crate) fn finish_gguf(what: &str, file: &Gguf, r: &Reader<'_>) -> Res<usize> {
    let (seen, skipped) = (r.seen(), r.skipped());
    let mut left: Vec<String> = file
        .names()
        .filter(|n| !seen.contains(*n) && !kvad::weights::derived(n))
        .filter(|n| !skipped.iter().any(|p| n.starts_with(p.as_str())))
        .map(str::to_string)
        .collect();
    left.sort();
    refuse_unread(what, left)?;
    Ok(file.names().filter(|n| seen.contains(*n)).filter_map(|n| file.stored(n)).map(|t| t.elems()).sum())
}

fn refuse_unread(what: &str, left: Vec<String>) -> Res<()> {
    if left.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{what}: the checkpoint holds {} tensor(s) that this loader never reads:\n  {}\n\
         A weight nobody reads is a piece of the model that is not running.",
        left.len(),
        kvad::weights::collapsed(&left).join("\n  ")
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tensor read past the cache is the one candle's mapped reader
    /// gives, in the file's dtype and converted, across two shards, with
    /// data at offsets that are not page-aligned and a tensor longer than a
    /// page.
    #[test]
    fn the_uncached_reader_reads_what_the_mapped_one_does() {
        let dir = std::env::temp_dir().join(format!("kvad-gpu-uncached-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ramp = |n: usize, dtype: DType| {
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
            Tensor::from_vec(v, n, &Device::Cpu).unwrap().to_dtype(dtype).unwrap()
        };
        let a: HashMap<String, Tensor> = [
            ("odd".to_string(), ramp(15, DType::BF16).reshape((3, 5)).unwrap()),
            ("wide".to_string(), ramp(30_011, DType::BF16)),
            ("full".to_string(), ramp(7, DType::F32)),
        ]
        .into();
        let b: HashMap<String, Tensor> = [
            ("half".to_string(), ramp(9, DType::F16)),
            ("ids".to_string(), Tensor::new(&[3u32, 1, 4, 1, 5], &Device::Cpu).unwrap()),
        ]
        .into();
        let paths = vec![dir.join("a.safetensors"), dir.join("b.safetensors")];
        candle_core::safetensors::save(&a, &paths[0]).unwrap();
        candle_core::safetensors::save(&b, &paths[1]).unwrap();

        for dtype in [DType::BF16, DType::F32] {
            let ours = open(&paths, dtype).unwrap();
            // SAFETY: files this test wrote and nothing else touches.
            let theirs = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &Device::Cpu).unwrap() };
            let names = [("odd", vec![3, 5]), ("wide", vec![30_011]), ("full", vec![7]), ("half", vec![9]), ("ids", vec![5])];
            for (name, shape) in names {
                let x = ours.get(shape.as_slice(), name).unwrap();
                let y = theirs.get(shape.as_slice(), name).unwrap();
                assert_eq!(x.dtype(), dtype, "{name}");
                // Compared as bits: widening any of these dtypes to f32 is
                // exact, so equal bits mean equal tensors.
                let bits = |t: &Tensor| -> Vec<u32> {
                    let v = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    v.iter().map(|f| f.to_bits()).collect()
                };
                assert_eq!(bits(&x), bits(&y), "{name} at {dtype:?}");
            }
            assert!(ours.get((5, 3), "odd").is_err(), "a wrong shape is refused");
            assert!(ours.get(1, "absent").is_err(), "a missing tensor is refused");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
