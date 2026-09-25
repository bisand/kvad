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
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use kvad::image::Painter;
use kvad::serde_json::Value;
use kvad::weights::{fetch_file, Cached, Watcher};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
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

/// Safetensors files read a tensor at a time, past the page cache.
///
/// candle's reader memory-maps the files, and every page it touches stays in
/// the page cache after the tensor has been copied out of it. An image model
/// is read once and kept whole on the device, so a load held two copies: the
/// weights, and the file pages they came from. For FLUX at bf16 that is
/// 33.7 GB of each, on a 48 GB machine, and macOS made room by compressing
/// and swapping the weights already loaded rather than dropping the clean
/// file pages. Nothing looked wrong until the first image: the encoder and
/// the first step each touch a model's worth of compressed pages, and took
/// 29 and 31 s where the second image's took 0.17 and 1.8 s.
///
/// So each tensor is `pread` into a buffer of its own, which is dropped once
/// the tensor is converted; on macOS with `F_NOCACHE`, which keeps the read
/// out of the cache. The headers are parsed here rather than by
/// `safetensors`, which wants the whole file in memory to read one. Read
/// this way FLUX loads in 44 s rather than 94, nothing is compressed, and
/// the first image's encoder and first step take 0.33 and 1.96 s; the price
/// is a tensor's buffers on the host, 0.3 GB at the peak.
struct Uncached {
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
            let file = File::open(path)?;
            #[cfg(target_os = "macos")]
            uncache(&file).map_err(|e| format!("{}: {e}", path.display()))?;
            // An eight-byte little-endian header length, the header as JSON,
            // then the data, which the header's offsets count from.
            let mut n = [0u8; 8];
            file.read_exact_at(&mut n, 0)?;
            let n = u64::from_le_bytes(n);
            let mut header = vec![0u8; usize::try_from(n)?];
            file.read_exact_at(&mut header, 8)?;
            let header: Value = kvad::serde_json::from_slice(&header)?;
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

    fn load(&self, name: &str) -> candle_core::Result<Tensor> {
        let t = self.tensors.get(name).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        // Page-aligned at both ends, in the file and in memory: an uncached
        // read that is not falls back to the cache.
        const PAGE: u64 = 16384;
        let file = &self.files[t.file];
        let from = t.at / PAGE * PAGE;
        let to = ((t.at + t.len as u64).div_ceil(PAGE) * PAGE).min(file.metadata()?.len());
        let span = (to - from) as usize;
        let mut buf = vec![0u8; span + PAGE as usize];
        let off = buf.as_ptr().align_offset(PAGE as usize);
        file.read_exact_at(&mut buf[off..off + span], from)?;
        let skip = off + (t.at - from) as usize;
        Tensor::from_raw_buffer(&buf[skip..skip + t.len], t.dtype, &t.shape, &Device::Cpu)
    }
}

/// Read `file` past the page cache from now on, and drop what the cache
/// already holds of it.
///
/// The second half matters as much as the first. An uncached read of a page
/// the cache already has is served from the cache, and marks the page used,
/// so a checkpoint just downloaded, or read by any earlier load, is kept
/// through the whole load. With 12.5 GB of FLUX's files cached that way, the
/// load compressed 35 GB of weights and the first image was as slow as with
/// no `F_NOCACHE` at all. `msync(MS_INVALIDATE)` over a mapping of the file
/// drops its clean pages, and a checkpoint has no other kind.
#[cfg(target_os = "macos")]
fn uncache(file: &File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    let len = file.metadata()?.len() as usize;
    // SAFETY: calls on a descriptor this function borrows, and a read-only
    // mapping that nothing reads and that is unmapped before returning.
    unsafe {
        if libc::fcntl(fd, libc::F_NOCACHE, 1) == -1 {
            return Err(std::io::Error::last_os_error());
        }
        if len == 0 {
            return Ok(());
        }
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, fd, 0);
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        let r = libc::msync(p, len, libc::MS_INVALIDATE);
        let e = std::io::Error::last_os_error();
        libc::munmap(p, len);
        if r == -1 {
            return Err(e);
        }
    }
    Ok(())
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
    let st = Uncached::open(paths)?;
    let seen = r.seen();
    Ok(st.tensors.iter().filter(|(n, _)| seen.contains(*n)).map(|(_, t)| t.shape.iter().product::<usize>()).sum())
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
