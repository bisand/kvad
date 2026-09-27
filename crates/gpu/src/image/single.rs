//! Stability's own layout: an SDXL checkpoint in one file.
//!
//! Most fine-tunes that are not diffusers folders are this: Pony on the Hub,
//! nearly everything on Civitai, and `sd_xl_base_1.0.safetensors` beside the
//! base's own folders. One file holds four models under four prefixes, in
//! the names the original code gave them:
//!
//! - `model.diffusion_model.`: the UNet, in `ldm`'s names. Its blocks are
//!   numbered as one list (`input_blocks.N.M`, `middle_block.M`,
//!   `output_blocks.N.M`) where diffusers nests them by level; inside a
//!   resnet the layers are numbered where diffusers names them; the
//!   transformers inside are diffusers' already.
//! - `conditioner.embedders.0.transformer.`: CLIP-L, in Hugging Face's names.
//! - `conditioner.embedders.1.model.`: OpenCLIP's bigG, in its own. Each
//!   layer's q, k and v are one `in_proj` matrix, whose thirds by rows are
//!   diffusers' three, and `text_projection` is the transpose of the linear
//!   layer diffusers makes of it.
//! - `first_stage_model.`: the VAE, left unread: Kvad decodes with
//!   madebyollin's fp16-fix, as it does every SDXL, in the same latent space
//!   and without SDXL's own VAE's overflow in f16.
//!
//! [`sdxl`] maps each name the three loaders ask for to where it is in the
//! file, from the file's own names; `docs/checkpoint-plan.md` has what that
//! was checked against.

use std::collections::HashMap;
use std::ops::Range;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Where one tensor a loader asks for is in the file: a tensor, some of its
/// rows, or its transpose.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Src {
    pub(crate) name: String,
    pub(crate) rows: Option<Range<usize>>,
    pub(crate) transpose: bool,
}

impl Src {
    fn whole(name: &str) -> Self {
        Src { name: name.to_string(), rows: None, transpose: false }
    }
}

/// A loader's names, each to its [`Src`].
pub(crate) type Map = HashMap<String, Src>;

/// The three maps, and the file's tensors deliberately read by none of them.
pub(crate) struct Sdxl {
    pub(crate) unet: Map,
    pub(crate) clip_l: Map,
    pub(crate) clip_g: Map,
    pub(crate) unread: Vec<String>,
}

const UNET: &str = "model.diffusion_model.";
const CLIP_L: &str = "conditioner.embedders.0.transformer.";
const CLIP_G: &str = "conditioner.embedders.1.model.";
const VAE: &str = "first_stage_model.";

/// bigG's width, and so the height of each third of its `in_proj`.
const G_WIDTH: usize = 1280;

/// The maps for an SDXL file whose tensors are `names`.
pub(crate) fn sdxl<'a>(names: impl IntoIterator<Item = &'a str>) -> Res<Sdxl> {
    let mut out = Sdxl { unet: Map::new(), clip_l: Map::new(), clip_g: Map::new(), unread: Vec::new() };
    let mut unknown = Vec::new();
    for name in names {
        if let Some(k) = name.strip_prefix(UNET) {
            out.unet.insert(unet(k).ok_or_else(|| format!("`{name}` is no UNet weight SDXL has"))?, Src::whole(name));
        } else if let Some(k) = name.strip_prefix(CLIP_L) {
            // A buffer of 0 … 76, which the loader builds itself.
            match k.ends_with("embeddings.position_ids") {
                true => out.unread.push(name.to_string()),
                false => drop(out.clip_l.insert(k.to_string(), Src::whole(name))),
            }
        } else if let Some(k) = name.strip_prefix(CLIP_G) {
            match clip_g(k, name) {
                Some(srcs) if srcs.is_empty() => out.unread.push(name.to_string()),
                Some(srcs) => out.clip_g.extend(srcs),
                None => unknown.push(name.to_string()),
            }
        } else if name.starts_with(VAE) {
            out.unread.push(name.to_string());
        } else {
            unknown.push(name.to_string());
        }
    }
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!(
            "{} tensor(s) are none of SDXL's four parts:\n  {}",
            unknown.len(),
            kvad::weights::collapsed(&unknown).join("\n  ")
        )
        .into());
    }
    if out.unet.is_empty() || out.clip_l.is_empty() || out.clip_g.is_empty() {
        return Err("this is not an SDXL checkpoint in Stability's layout: it lacks the UNet or one of the two text encoders".into());
    }
    Ok(out)
}

/// A UNet name of `ldm`'s, as diffusers names it, for SDXL's three levels:
/// two resnets a level on the way down and three on the way up.
fn unet(k: &str) -> Option<String> {
    for (from, to) in [
        ("time_embed.0.", "time_embedding.linear_1."),
        ("time_embed.2.", "time_embedding.linear_2."),
        ("label_emb.0.0.", "add_embedding.linear_1."),
        ("label_emb.0.2.", "add_embedding.linear_2."),
        ("input_blocks.0.0.", "conv_in."),
        ("out.0.", "conv_norm_out."),
        ("out.2.", "conv_out."),
    ] {
        if let Some(rest) = k.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    // `input_blocks.N.M.rest`, `middle_block.M.rest`, `output_blocks.N.M.rest`.
    let mut parts = k.splitn(4, '.');
    let (block, first) = (parts.next()?, parts.next()?);
    let num = |s: &str| s.parse::<usize>().ok();
    match block {
        "input_blocks" => {
            let (n, m, rest) = (num(first)?, num(parts.next()?)?, parts.next()?);
            let (level, i) = ((n - 1) / 3, (n - 1) % 3);
            Some(match (i, m) {
                // The third of each level's three is its downsampler.
                (2, 0) => format!("down_blocks.{level}.downsamplers.0.conv.{}", rest.strip_prefix("op.")?),
                (_, 0) => format!("down_blocks.{level}.resnets.{i}.{}", resnet(rest)?),
                (_, 1) => format!("down_blocks.{level}.attentions.{i}.{rest}"),
                _ => return None,
            })
        }
        "middle_block" => {
            let (m, rest) = (num(first)?, k.splitn(3, '.').nth(2)?);
            Some(match m {
                0 => format!("mid_block.resnets.0.{}", resnet(rest)?),
                1 => format!("mid_block.attentions.0.{rest}"),
                2 => format!("mid_block.resnets.1.{}", resnet(rest)?),
                _ => return None,
            })
        }
        "output_blocks" => {
            let (n, m, rest) = (num(first)?, num(parts.next()?)?, parts.next()?);
            let (level, i) = (n / 3, n % 3);
            Some(match m {
                0 => format!("up_blocks.{level}.resnets.{i}.{}", resnet(rest)?),
                // An upsampler follows the resnet, or the attention where
                // the level has one.
                _ if rest.starts_with("conv.") => format!("up_blocks.{level}.upsamplers.0.{rest}"),
                1 => format!("up_blocks.{level}.attentions.{i}.{rest}"),
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Inside a resnet, `ldm`'s numbered layers by diffusers' names.
fn resnet(rest: &str) -> Option<String> {
    for (from, to) in [
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("emb_layers.1.", "time_emb_proj."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("skip_connection.", "conv_shortcut."),
    ] {
        if let Some(r) = rest.strip_prefix(from) {
            return Some(format!("{to}{r}"));
        }
    }
    None
}

/// An OpenCLIP name of bigG's, as the one or more diffusers names it
/// becomes: empty for what diffusers has no use for, `None` for what bigG
/// does not have.
fn clip_g(k: &str, full: &str) -> Option<Vec<(String, Src)>> {
    let one = |to: String| Some(vec![(to, Src::whole(full))]);
    match k {
        "token_embedding.weight" => return one("text_model.embeddings.token_embedding.weight".into()),
        "positional_embedding" => return one("text_model.embeddings.position_embedding.weight".into()),
        "text_projection" => {
            return Some(vec![("text_projection.weight".into(), Src { name: full.to_string(), rows: None, transpose: true })])
        }
        // CLIP's contrastive temperature, which only training reads.
        "logit_scale" => return Some(Vec::new()),
        _ => {}
    }
    if let Some(rest) = k.strip_prefix("ln_final.") {
        return one(format!("text_model.final_layer_norm.{rest}"));
    }
    let rest = k.strip_prefix("transformer.resblocks.")?;
    let (n, rest) = rest.split_once('.')?;
    let layer = format!("text_model.encoder.layers.{n}.");
    if let Some(kind) = rest.strip_prefix("attn.in_proj_") {
        return Some(
            ["q", "k", "v"]
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let src = Src { name: full.to_string(), rows: Some(i * G_WIDTH..(i + 1) * G_WIDTH), transpose: false };
                    (format!("{layer}self_attn.{x}_proj.{kind}"), src)
                })
                .collect(),
        );
    }
    for (from, to) in [("attn.out_proj.", "self_attn.out_proj."), ("ln_1.", "layer_norm1."), ("ln_2.", "layer_norm2."), ("mlp.c_fc.", "mlp.fc1."), ("mlp.c_proj.", "mlp.fc2.")] {
        if let Some(r) = rest.strip_prefix(from) {
            return one(format!("{layer}{to}{r}"));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names `diffusers`' own conversion gives, for one of each kind.
    #[test]
    fn unet_names_are_diffusers() {
        let cases = [
            ("time_embed.0.weight", "time_embedding.linear_1.weight"),
            ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
            ("input_blocks.0.0.weight", "conv_in.weight"),
            ("input_blocks.1.0.in_layers.2.weight", "down_blocks.0.resnets.0.conv1.weight"),
            ("input_blocks.2.0.emb_layers.1.bias", "down_blocks.0.resnets.1.time_emb_proj.bias"),
            ("input_blocks.3.0.op.weight", "down_blocks.0.downsamplers.0.conv.weight"),
            ("input_blocks.4.0.skip_connection.weight", "down_blocks.1.resnets.0.conv_shortcut.weight"),
            ("input_blocks.5.1.proj_in.weight", "down_blocks.1.attentions.1.proj_in.weight"),
            ("input_blocks.8.1.transformer_blocks.9.attn2.to_k.weight", "down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_k.weight"),
            ("middle_block.0.out_layers.3.bias", "mid_block.resnets.0.conv2.bias"),
            ("middle_block.1.norm.weight", "mid_block.attentions.0.norm.weight"),
            ("middle_block.2.in_layers.0.weight", "mid_block.resnets.1.norm1.weight"),
            ("output_blocks.0.0.out_layers.0.bias", "up_blocks.0.resnets.0.norm2.bias"),
            ("output_blocks.2.1.proj_out.bias", "up_blocks.0.attentions.2.proj_out.bias"),
            ("output_blocks.2.2.conv.weight", "up_blocks.0.upsamplers.0.conv.weight"),
            ("output_blocks.5.2.conv.bias", "up_blocks.1.upsamplers.0.conv.bias"),
            ("output_blocks.8.0.skip_connection.bias", "up_blocks.2.resnets.2.conv_shortcut.bias"),
            ("out.2.weight", "conv_out.weight"),
        ];
        for (from, to) in cases {
            assert_eq!(unet(from).as_deref(), Some(to), "{from}");
        }
        assert_eq!(unet("input_blocks.1.2.weight"), None);
        assert_eq!(unet("zero_block.0.weight"), None);
    }

    #[test]
    fn big_g_splits_its_projections_and_transposes_its_head() {
        let full = |k: &str| format!("{CLIP_G}{k}");
        let q = clip_g("transformer.resblocks.3.attn.in_proj_weight", &full("transformer.resblocks.3.attn.in_proj_weight")).unwrap();
        let names: Vec<&str> = q.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["text_model.encoder.layers.3.self_attn.q_proj.weight", "text_model.encoder.layers.3.self_attn.k_proj.weight", "text_model.encoder.layers.3.self_attn.v_proj.weight"]);
        assert_eq!(q[2].1.rows, Some(2560..3840));
        let head = clip_g("text_projection", &full("text_projection")).unwrap();
        assert_eq!((head[0].0.as_str(), head[0].1.transpose), ("text_projection.weight", true));
        assert_eq!(clip_g("logit_scale", &full("logit_scale")), Some(Vec::new()));
        assert_eq!(clip_g("visual.proj", &full("visual.proj")), None);
    }

    /// Every tensor the three loaders read from Stability's own
    /// `sd_xl_base_1.0.safetensors`, through the maps, against the same name
    /// in the base's own diffusers files, bit for bit, and every name those
    /// files have, found. The map's real check. Needs both on this machine:
    ///
    ///     cargo test --release -p kvad-gpu single::tests::the_base -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_base_file_is_its_own_folders_tensor_for_tensor() {
        use crate::image::{local_file, open_file, open_mapped, sdxl::REPO};
        use candle_core::{DType, Device};
        let path = local_file(REPO, "sd_xl_base_1.0.safetensors").expect("fetch sd_xl_base_1.0.safetensors first");
        let file = open_file(&path).unwrap();
        let maps = sdxl(file.names()).unwrap();
        let (mut same, mut total) = (0, 0);
        for (map, part) in [
            (&maps.clip_l, "text_encoder/model.fp16.safetensors"),
            (&maps.clip_g, "text_encoder_2/model.fp16.safetensors"),
            (&maps.unet, "unet/diffusion_pytorch_model.fp16.safetensors"),
        ] {
            // SAFETY: a read-only file in the Hub's cache.
            let theirs = unsafe { candle_core::safetensors::MmapedSafetensors::new(local_file(REPO, part).unwrap()).unwrap() };
            let ours = open_mapped(&file, map.clone(), DType::F16);
            let names: Vec<String> = theirs.tensors().into_iter().map(|(n, _)| n).collect();
            assert_eq!(names.len(), map.len(), "{part}: {} names, the map {}", names.len(), map.len());
            for n in &names {
                let want = theirs.load(n, &Device::Cpu).unwrap().to_dtype(DType::F16).unwrap();
                let got = ours.get(want.dims(), n).unwrap_or_else(|e| panic!("{part} {n}: {e}"));
                let bits = |t: &candle_core::Tensor| t.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
                total += 1;
                if bits(&got) == bits(&want) {
                    same += 1;
                } else {
                    eprintln!("{part} {n}: different numbers");
                }
            }
        }
        eprintln!("{same} of {total} tensors bit for bit; {} of the file's left unread", maps.unread.len());
        assert_eq!(same, total);
    }

    #[test]
    fn a_file_that_is_not_sdxl_says_so() {
        let names = ["model.diffusion_model.input_blocks.0.0.weight", "cond_stage_model.transformer.text_model.final_layer_norm.weight"];
        let e = sdxl(names).err().unwrap().to_string();
        assert!(e.contains("cond_stage_model"), "{e}");
    }
}
