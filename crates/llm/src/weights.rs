//! Fetching models from the HuggingFace Hub and reading their weights.
//!
//! # safetensors
//!
//! The format is deliberately boring, which is the point — the older `.bin`
//! format was a pickled Python object graph, i.e. arbitrary code execution on
//! load. A safetensors file is:
//!
//! ```text
//! [8 bytes: header length, little-endian u64]
//! [header: JSON mapping tensor name -> {dtype, shape, byte offsets}]
//! [the raw tensor bytes, back to back]
//! ```
//!
//! So loading is: parse a small JSON blob, then slice into a memory map.
//!
//! Models above a few GB are split into shards, with a
//! `model.safetensors.index.json` mapping each tensor name to the file holding
//! it. [`Checkpoint`] hides that: open all the shards, build one name index,
//! and look tensors up without caring where they live.
//!
//! # Models that are not on the Hub
//!
//! Wherever a repo id is accepted, a directory is too: if the name given is a
//! directory that exists, it is read in place and nothing is fetched. That is
//! how a model trained by this repository's own `nanograd` gets here, and the
//! rule — an existing directory wins over a repo of the same name — is the one
//! `transformers` uses, so nobody has to learn a second one. The directory
//! holds what a Hub repo would: `config.json`, `tokenizer.json`, and either
//! `model.safetensors` or a shard index.

use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub struct ModelFiles {
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    /// Holds the chat template, when the model has one.
    pub tokenizer_config: Option<PathBuf>,
    pub generation_config: Option<PathBuf>,
}

/// Whether `model` names a directory on this machine rather than a Hub repo.
pub fn is_local(model: &str) -> bool {
    Path::new(model).is_dir()
}

/// Whether `model` was written the way paths are and repo ids never are.
pub fn looks_like_path(model: &str) -> bool {
    model.starts_with(['.', '/', '~'])
}

/// The name a model is known by once loaded: a repo id as it is, a directory
/// as its absolute path.
///
/// Anything keyed on the name — the quantised-weight cache above all — must
/// not think `out/readme`, `./out/readme` and the same words typed from
/// another working directory are three models, or worse, one.
pub fn model_id(model: &str) -> String {
    match is_local(model) {
        true => std::fs::canonicalize(model).map_or_else(|_| model.to_string(), |p| p.display().to_string()),
        false => model.to_string(),
    }
}

impl ModelFiles {
    /// The files of a model that is already in `dir`.
    pub fn from_dir(dir: &Path) -> Res<Self> {
        let need = |name: &str| -> Res<PathBuf> {
            let path = dir.join(name);
            match path.is_file() {
                true => Ok(path),
                false => Err(format!("{} has no {name}", dir.display()).into()),
            }
        };
        let maybe = |name: &str| Some(dir.join(name)).filter(|p| p.is_file());

        let weights = match maybe("model.safetensors") {
            Some(single) => vec![single],
            None => {
                let index = maybe("model.safetensors.index.json").ok_or_else(|| {
                    format!("{} has no model.safetensors, and no shard index either", dir.display())
                })?;
                shard_names(&index)?.iter().map(|s| need(s)).collect::<Res<Vec<_>>>()?
            }
        };

        Ok(ModelFiles {
            weights,
            tokenizer: need("tokenizer.json")?,
            config: need("config.json")?,
            tokenizer_config: maybe("tokenizer_config.json"),
            generation_config: maybe("generation_config.json"),
        })
    }
}

/// The distinct files a shard index points at.
fn shard_names(index: &Path) -> Res<Vec<String>> {
    let json = read_json(index)?;
    let map = json.get("weight_map").and_then(|m| m.as_object()).ok_or("shard index has no weight_map")?;

    // Many tensor names point at the same handful of files.
    let mut shards: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
    shards.sort();
    shards.dedup();
    Ok(shards)
}

/// Download (or reuse from the local cache) everything needed to run `repo_id`,
/// reporting progress to stderr. A directory is used as it is.
pub fn fetch(repo_id: &str) -> Res<ModelFiles> {
    fetch_with(repo_id, &mut |msg| eprintln!("  {msg}"))
}

/// As [`fetch`], but progress goes to a callback.
///
/// The TUI needs this: anything written straight to stderr lands on top of the
/// rendered frame and corrupts the display.
pub fn fetch_with(repo_id: &str, progress: &mut dyn FnMut(&str)) -> Res<ModelFiles> {
    if is_local(repo_id) {
        progress("a directory on this machine; nothing to fetch");
        return ModelFiles::from_dir(Path::new(repo_id));
    }
    // Typed as a path, so meant as one: say the directory is missing, rather
    // than go and ask the Hub for a repo called `./out`.
    if looks_like_path(repo_id) {
        return Err(format!("`{repo_id}` looks like a path, and there is no such directory").into());
    }

    let (owner, name) = repo_id.split_once('/').ok_or_else(|| {
        format!("expected a repo id like `openai-community/gpt2` or a directory, got `{repo_id}`")
    })?;

    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model(owner, name);

    let repo = &repo;
    let progress = std::cell::RefCell::new(progress);
    let get = |filename: &str| -> Res<PathBuf> {
        (progress.borrow_mut())(&format!("fetching {filename}"));
        Ok(repo.download_file().filename(filename.to_string()).send()?)
    };
    let try_get = |filename: &str| -> Option<PathBuf> {
        repo.download_file().filename(filename.to_string()).send().ok()
    };

    // Single file, or a shard index naming several.
    let weights = match try_get("model.safetensors") {
        Some(single) => vec![single],
        None => {
            let shards = shard_names(&get("model.safetensors.index.json")?)?;
            (progress.borrow_mut())(&format!("checkpoint is split across {} shards", shards.len()));
            shards.iter().map(|s| get(s)).collect::<Res<Vec<_>>>()?
        }
    };

    Ok(ModelFiles {
        weights,
        tokenizer: get("tokenizer.json")?,
        config: get("config.json")?,
        tokenizer_config: try_get("tokenizer_config.json"),
        generation_config: try_get("generation_config.json"),
    })
}

/// One or more safetensors files, presented as a single namespace.
pub struct Checkpoint {
    maps: Vec<memmap2::Mmap>,
    /// tensor name -> which shard holds it
    index: HashMap<String, usize>,
}

impl Checkpoint {
    pub fn open(paths: &[PathBuf]) -> Res<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        for path in paths {
            let file = std::fs::File::open(path)?;
            // SAFETY: we only ever read, and these are read-only cache entries.
            // A concurrent writer truncating the file would be undefined
            // behaviour, which is the standard caveat on every mmap.
            maps.push(unsafe { memmap2::Mmap::map(&file)? });
        }

        // Parse each header once and remember where every tensor lives.
        let mut index = HashMap::new();
        for (i, map) in maps.iter().enumerate() {
            let st = SafeTensors::deserialize(map)?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
        }
        Ok(Checkpoint { maps, index })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Look a tensor up, converting to `f32`.
    ///
    /// Checkpoints disagree about prefixes — GPT-2 saves `wte.weight` or
    /// `transformer.wte.weight` depending on which Python class wrote it — so
    /// a few spellings are tried before giving up.
    pub fn try_get(&self, name: &str) -> Option<Tensor> {
        let candidates = [
            name.to_string(),
            format!("transformer.{name}"),
            format!("model.{name}"),
        ];
        let key = candidates.iter().find(|c| self.index.contains_key(*c))?;
        let st = SafeTensors::deserialize(&self.maps[self.index[key]]).ok()?;
        let view = st.tensor(key).ok()?;

        let shape = view.shape();
        let (rows, cols) = match shape.len() {
            1 => (1, shape[0]),
            2 => (shape[0], shape[1]),
            _ => return None,
        };
        Some(Tensor::new(rows, cols, decode(view.data(), view.dtype())?))
    }

    pub fn get(&self, name: &str) -> Res<Tensor> {
        self.try_get(name)
            .ok_or_else(|| format!("tensor `{name}` not found in checkpoint").into())
    }

    /// For 1-D tensors, where the shape is noise.
    pub fn get_flat(&self, name: &str) -> Res<Vec<f32>> {
        Ok(self.get(name)?.data)
    }

    pub fn try_get_flat(&self, name: &str) -> Option<Vec<f32>> {
        self.try_get(name).map(|t| t.data)
    }
}

fn decode(bytes: &[u8], dtype: Dtype) -> Option<Vec<f32>> {
    Some(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // Half precision: widen on load. bf16 is just f32 with the bottom 16
        // bits chopped off, which is why it is so cheap to convert and so
        // popular for training -- it keeps f32's exponent range, and range is
        // what gradients need.
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => return None,
    })
}

/// IEEE 754 half -> single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            // Subnormal: there is no implicit leading 1, and the value is
            // `mant * 2^-24`. f32 has the range to store it as a normal
            // number, so renormalise: find the top set bit, make it the
            // implicit 1, and shift the rest into the fraction field.
            let top = 31 - mant.leading_zeros();
            let exp = 127 - 24 + top;
            let frac = (mant << (23 - top)) & 0x7f_ffff;
            (sign << 31) | (exp << 23) | frac
        }
        31 => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// Read a JSON file into a `serde_json::Value`.
pub fn read_json(path: &Path) -> Res<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn half_precision_conversion() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert!((f16_to_f32(0x3555) - 0.333_251).abs() < 1e-5);
        assert!(f16_to_f32(0x7c00).is_infinite());
        // Smallest positive subnormal: 2^-24.
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }
}
