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
//! - [`ltx_upsample`]: the latent upsampler between the two stages;
//! - [`ltx_vae`]: the convolutional video decoder, and [`ltx_audio`]: the
//!   audio decoder, vocoder and bandwidth extension.
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
pub mod ltx_audio;
pub mod ltx_dit;
pub mod ltx_sample;
pub(crate) mod ltx_nn;
pub mod ltx_text;
pub mod ltx_upsample;
pub mod ltx_vae;

use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The repo LTX-2.5 is published in: one safetensors file per component,
/// each with its config in the file's own metadata.
pub const LTX_REPO: &str = "Lightricks/LTX-2.5";

/// A string from a safetensors file's `__metadata__`, parsed as JSON.
///
/// LTX's split checkpoints carry their configs there rather than in a
/// `config.json` beside them. Reading it is the first 8 bytes (the header's
/// length) and the header, never the tensors.
pub(crate) fn metadata(path: &Path, key: &str) -> Res<kvad::serde_json::Value> {
    use std::io::Read;
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
