//! Video models: a prompt to frames and sound.
//!
//! So far this is LTX-2.5 (`docs/video-plan.md`), its distilled model in two
//! stages:
//!
//! - [`ltx_text`]: the prompt to two contexts, through Gemma 4 ([`gemma`]),
//!   the aggregate projections and the connectors;
//! - [`ltx_dit`]: the 48-block DiT that denoises a video and its sound
//!   together, and [`ltx_sample`], the steps that drive it: eight at half
//!   size, then three at full size;
//! - [`ltx_duration`]: how long a prompt's clip wants to be, from the same
//!   contexts, when a request does not say;
//! - [`ltx_upsample`]: the latent upsamplers, spatial between the two
//!   stages, and temporal for DFR's rounds;
//! - [`ltx_cond`]: the tokens DFR appends to a video, its keyframes and
//!   reference latent, each at its own σ and place, and [`ltx_dfr`], DFR's
//!   canvas, stages and temporal rounds, which make them;
//! - [`ltx_diffvae`]: the diffusion video decoder, a neighbourhood-attention
//!   transformer, the default, and the conv decoder's alternative;
//! - [`ltx_vae`]: the convolutional video decoder, and [`ltx_audio`]: the
//!   audio decoder, vocoder and bandwidth extension.
//!
//! [`ltx`] puts them together as a [`kvad::video::Director`], for the
//! server.
//!
//! They are written on candle for the same reason the image models are
//! (`docs/image-plan.md`): the decoders are convolution stacks, and `kvad`'s
//! own `tensor.rs` has no convolutions.
//!
//! The one thing here that the image models never needed is the third
//! dimension. candle has no 3D convolution, and [`conv3d`] builds one out of
//! 2D ones, exactly.

pub mod conv3d;
pub mod gemma;
pub mod ltx;
pub mod ltx_audio;
pub mod ltx_cond;
pub mod ltx_dfr;
pub mod ltx_diffvae;
pub mod ltx_dit;
pub mod ltx_duration;
pub(crate) mod ltx_fused;
pub mod ltx_sample;
pub mod ltx_tile;
pub(crate) mod ltx_nn;
pub mod ltx_text;
pub mod ltx_upsample;
pub mod ltx_vae;

use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The repo LTX-2.5 is published in: one safetensors file per component,
/// each with its config in the file's own metadata.
pub const LTX_REPO: &str = kvad::video::LTX_REPO;

/// A safetensors file's `__metadata__`, its strings as they are.
pub(crate) fn metadata_raw(path: &Path) -> Res<std::collections::HashMap<String, String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let header: kvad::serde_json::Value = kvad::serde_json::from_slice(&header)?;
    Ok(header["__metadata__"].as_object().map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect()).unwrap_or_default())
}

/// Whether `path` is a GGUF, which LTX-2.5's DiT can be read from.
pub(crate) fn is_gguf(path: &Path) -> bool {
    path.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
}

/// A GGUF of LTX-2.5's DiT under the names its own file uses.
pub(crate) fn open_dit_gguf(path: &Path) -> Res<crate::gguf::Gguf> {
    let file = crate::gguf::Gguf::open(path)?;
    match file.text("general.architecture") {
        Some("ltxv") => file.prefixed("model.diffusion_model."),
        other => Err(format!("{} is a GGUF of {}, not of LTX-2.5's DiT", path.display(), other.unwrap_or("an unnamed architecture")).into()),
    }
}

/// A string from a safetensors file's `__metadata__`, parsed as JSON.
///
/// LTX's split checkpoints carry their configs there rather than in a
/// `config.json` beside them. Reading it is the first 8 bytes (the header's
/// length) and the header, never the tensors.
pub(crate) fn metadata(path: &Path, key: &str) -> Res<kvad::serde_json::Value> {
    use std::io::Read;
    // A GGUF of the DiT carries the same strings in its own header.
    if is_gguf(path) {
        let file = crate::gguf::Gguf::open(path)?;
        let text = file.text(key).ok_or_else(|| format!("{}: no `{key}` in the GGUF's header", path.display()))?;
        return Ok(kvad::serde_json::from_str(text)?);
    }
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let header: kvad::serde_json::Value = kvad::serde_json::from_slice(&header)?;
    let text = header["__metadata__"][key]
        .as_str()
        .ok_or_else(|| format!("{}: no `{key}` in the safetensors metadata", path.display()))?;
    Ok(kvad::serde_json::from_str(text)?)
}
