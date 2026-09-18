//! Fetching a model from the HuggingFace Hub and reading its weights.
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
//! So loading is: parse a small JSON blob, then slice into a memory map. No
//! deserialisation of the weights themselves.

use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
use std::path::PathBuf;

pub struct ModelFiles {
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    pub config: PathBuf,
}

/// Download (or reuse from the local cache) the three files we need.
pub fn fetch(repo_id: &str) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    let (owner, name) = repo_id
        .split_once('/')
        .ok_or_else(|| format!("expected a repo id like `openai-community/gpt2`, got `{repo_id}`"))?;

    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model(owner, name);

    let get = |filename: &str| -> Result<PathBuf, Box<dyn std::error::Error>> {
        eprintln!("  fetching {repo_id}/{filename}");
        Ok(repo.download_file().filename(filename.to_string()).send()?)
    };

    Ok(ModelFiles {
        weights: get("model.safetensors")?,
        tokenizer: get("tokenizer.json")?,
        config: get("config.json")?,
    })
}

/// A loaded safetensors file, with the raw bytes kept alive by an mmap.
pub struct Weights {
    mmap: memmap2::Mmap,
}

impl Weights {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: we only read, and we never mutate the file while mapped.
        // (A concurrent writer truncating the file would be UB; in practice
        // this is a read-only cache entry.)
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Weights { mmap })
    }

    pub fn view(&self) -> Result<SafeTensors<'_>, safetensors::SafeTensorError> {
        SafeTensors::deserialize(&self.mmap)
    }
}

/// Pull one tensor out by name, converting to `f32`.
///
/// `prefix` handles a wart: some GPT-2 checkpoints name their tensors
/// `wte.weight`, others `transformer.wte.weight`, depending on which Python
/// class saved them. We try both rather than making you care.
pub fn get(st: &SafeTensors, name: &str) -> Result<Tensor, Box<dyn std::error::Error>> {
    let view = st
        .tensor(name)
        .or_else(|_| st.tensor(&format!("transformer.{name}")))
        .map_err(|_| format!("tensor `{name}` not found in checkpoint"))?;

    let shape = view.shape();
    let (rows, cols) = match shape.len() {
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        n => return Err(format!("tensor `{name}` has rank {n}, expected 1 or 2").into()),
    };

    let bytes = view.data();
    let data: Vec<f32> = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // Half precision: widen on load. f16 and bf16 differ in where the
        // bits go -- bf16 is just f32 with the bottom 16 bits chopped off,
        // which is why it is so cheap to convert and so popular for training.
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        other => return Err(format!("tensor `{name}` has unsupported dtype {other:?}").into()),
    };

    Ok(Tensor::new(rows, cols, data))
}

/// IEEE 754 half -> single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        // Zero or subnormal.
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
        // Inf / NaN.
        31 => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
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
