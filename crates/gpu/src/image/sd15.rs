//! Stable Diffusion 1.5: [`super::sdxl`] at a smaller size, with less
//! around it.
//!
//! One text encoder, CLIP-L, read to its last layer and through its final
//! norm, where SDXL reads two encoders' second-to-last layers; no pooled
//! vector and no size conditioning; a UNet of four levels, whose transformers
//! project in and out with 1×1 convolutions; and the repo's own VAE, in f16,
//! which SD 1.5's, unlike SDXL's, stays in range for. Or all three from one
//! file in Stability's layout ([`super::single`]), its VAE among them, since
//! a fine-tune's is often its own. `docs/sd15-plan.md` has what each part was
//! checked against.
//!
//! The scheduler is Euler on the repo's noise levels, whatever class its
//! config names: PNDM in the base, DEIS in DreamShaper's. That is what most
//! people swap SD 1.5 to anyway, and what `schedule::euler` implements.

use super::clip::{self, Clip, ClipConfig, Pooled};
use super::nn::{check_latent, latent_preview, noise, to_rgb8, Ctx};
use super::edit::Edited;
use super::schedule;
use super::sdxl::{weights, TOKENIZER_REPO};
use super::unet::{Unet, UnetConfig};
use super::vae::{Decoder, Encoder, VaeConfig};
use super::sdxl::local_weights;
use super::lora::{self, Adapters};
use super::{finish, finish_mapped, open, open_file, open_mapped, read_json, single};
use crate::common::{settle, Loader, Reader};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor};
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::Value;
use kvad::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub const REPO: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";

pub struct Sd15 {
    tok: tokenizers::Tokenizer,
    clip: Clip,
    unet: Unet,
    vae: Decoder,
    /// The VAE's other half, for an image made from a picture ([`edit`]).
    encoder: Encoder,
    scheduler: Value,
    device: Device,
    dtype: DType,
    /// The text encoder's and the UNet's layers, for LoRAs ([`lora`]).
    adapters: Adapters,
    params: usize,
}

/// What a LoRA's names for SD 1.5 may start with, and the part each is in:
/// kohya's `lora_unet_` and `lora_te_`, diffusers' `unet.` and
/// `text_encoder.`, and a UNet's layers named bare.
pub(crate) const PREFIXES: [(&str, &str); 6] = [("lora_unet_", "unet"), ("unet.", "unet"), ("lora_te_", "te1"), ("lora_te1_", "te1"), ("text_encoder.", "te1"), ("", "unet")];

impl Sd15 {
    pub fn load(repo: &str, device: Device, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Self> {
        Self::load_with(repo, None, device, progress, watch)
    }

    /// [`Sd15::load`], with all three models read from `single`, one file in
    /// Stability's layout ([`single`]), its VAE among them. `repo` then gives
    /// only the components' configs and the scheduler, which a single file
    /// does not carry.
    pub fn load_with(repo: &str, single: Option<&Path>, device: Device, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Self> {
        let dtype = DType::F16;
        let get = |f: &str| fetch_file(repo, f, watch);
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };

        progress("reading the tokenizer");
        // The repo's CLIP tokenizer is `vocab.json` and `merges.txt`; the
        // same vocabulary as a `tokenizer.json` is where SDXL gets it.
        let tok = tokenizers::Tokenizer::from_file(fetch_file(TOKENIZER_REPO, "tokenizer.json", watch)?).map_err(|e| e.to_string())?;
        let scheduler = read_json(&get("scheduler/scheduler_config.json")?)?;
        // Checked now rather than at the first request.
        schedule::euler(&scheduler, 25)?;

        let mut params = 0;
        let config = |dir: &str| -> Res<Value> { read_json(&get(&format!("{dir}/config.json"))?) };
        // One file for all three, read through a map each, and their unread
        // weights counted together at the end; or each component's own file.
        let file = single.map(open_file).transpose()?;
        let maps = file.as_ref().map(|f| single::sd15(f.names())).transpose()?;
        let reader = |dir: &str, stem: &str, map: Option<&single::Map>| -> Res<(Reader<'static>, Vec<PathBuf>)> {
            match (&file, map) {
                (Some(f), Some(m)) => Ok((open_mapped(f, m.clone(), dtype), Vec::new())),
                _ => {
                    let paths = vec![weights(repo, dir, stem, watch)?];
                    Ok((open(&paths, dtype)?, paths))
                }
            }
        };

        let adapters = Adapters::new(&PREFIXES);
        progress("loading the text encoder");
        let (r_t, paths) = reader("text_encoder", "model", maps.as_ref().map(|m| &m.clip))?;
        let r_t = r_t.with_adapters(adapters.part("te1"));
        let clip = Clip::load(&cx, &r_t, ClipConfig::from_json(&config("text_encoder")?)?, Pooled::Last)?;
        if file.is_none() {
            params += finish("text encoder", &paths, &r_t)?;
        }

        progress("loading the UNet");
        let (r_u, paths) = reader("unet", "diffusion_pytorch_model", maps.as_ref().map(|m| &m.unet))?;
        let r_u = r_u.with_adapters(adapters.part("unet"));
        let ucfg = UnetConfig::from_json(&config("unet")?)?;
        if ucfg.added.is_some() || ucfg.context != clip.width() {
            return Err(format!("this UNet attends to {}-wide text with size conditioning {:?}: not SD 1.5's", ucfg.context, ucfg.added).into());
        }
        // An inpainting model's UNet reads the masked image and the mask
        // beside the latent: nine channels in, where the base's config says
        // four. Said here, rather than as a shape the loader did not expect.
        if let Some(t) = file.as_ref().and_then(|f| f.tensors.get("model.diffusion_model.input_blocks.0.0.weight")) {
            if t.shape.get(1) != Some(&ucfg.in_channels) {
                return Err(format!("this UNet takes {:?} channels in, where SD 1.5's text-to-image takes {}: an inpainting model, which is not implemented here", t.shape.get(1), ucfg.in_channels).into());
            }
        }
        let unet = Unet::load(&cx, &r_u, ucfg)?;
        adapters.alias_ldm("unet");
        if file.is_none() {
            params += finish("UNet", &paths, &r_u)?;
        }

        progress("loading the VAE");
        let (r_v, paths) = reader("vae", "diffusion_pytorch_model", maps.as_ref().map(|m| &m.vae))?;
        let vae = Decoder::load(&cx, &r_v, VaeConfig::from_json(&config("vae")?)?)?;
        let encoder = Encoder::load(&cx, &r_v, VaeConfig::from_json(&config("vae")?)?)?;
        match (&file, &maps) {
            (Some(f), Some(m)) => {
                let parts = [(&m.clip, &r_t), (&m.unet, &r_u), (&m.vae, &r_v)];
                let what = single.map(|p| p.display().to_string()).unwrap_or_default();
                params += finish_mapped(&what, f, &parts, &m.unread)?;
            }
            _ => params += finish("VAE", &paths, &r_v)?,
        }

        settle(&device)?;
        progress(&format!("loaded SD 1.5: {:.2} B parameters in f16", params as f64 / 1e9));
        Ok(Sd15 { tok, clip, unet, vae, encoder, scheduler, device, dtype, adapters, params })
    }

    /// The prompt as the UNet reads it: `[1, 77, 768]`.
    fn encode(&self, text: &str) -> Res<Tensor> {
        let (ids, _) = clip::tokenize(&self.tok, text, clip::END)?;
        Ok(self.clip.encode(&ids, 0)?.0)
    }
}

impl Sd15 {
    /// One image, with whatever LoRAs the adapters hold.
    fn draw(&mut self, asked: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let req = asked.resolved(&self.defaults())?;
        let t0 = Instant::now();

        // The prompt, and what guidance steers away from: the negative
        // prompt, or the empty one, encoded, as SD 1.5's pipeline does. (It
        // is SDXL's that uses zeros.)
        let ctx = self.encode(&req.prompt)?;
        let guided = req.guidance > 1.0;
        let ctx = match guided {
            true => Tensor::cat(&[&self.encode(req.negative_prompt.as_deref().unwrap_or(""))?, &ctx], 0)?,
            false => ctx,
        };
        settle(&self.device)?;
        let encode_secs = t0.elapsed().as_secs_f64();

        let sched = schedule::euler(&self.scheduler, req.steps)?;
        let f = self.vae.config().factor();
        let shape = [1, 4, req.height / f, req.width / f];
        // In f32 between steps, as SDXL's.
        let eps = noise(req.seed, &shape, &self.device, DType::F32)?;
        // From noise; or from a picture, noised part of the way ([`edit`]).
        let (edited, mut x) = match &asked.edit {
            None => (None, (eps * sched.init_scale)?),
            Some(e) => {
                let (edited, x) = Edited::begin(e, &req, &sched, f, eps, |pixels| {
                    let seen = self.encoder.encode(&pixels.to_dtype(self.dtype)?.to_device(&self.device)?)?;
                    Ok(self.encoder.to_denoiser(&seen.mean.to_dtype(DType::F32)?)?)
                })?;
                settle(&self.device)?;
                (Some(edited), x)
            }
        };
        let first = edited.as_ref().map_or(0, |e| e.first);

        let t1 = Instant::now();
        for i in first..sched.steps() {
            let xin = (&x * sched.input_scale(i))?.to_dtype(self.dtype)?;
            let xin = if guided { Tensor::cat(&[&xin, &xin], 0)? } else { xin };
            let eps = self.unet.forward(&xin, sched.timesteps[i], &ctx, None)?.to_dtype(DType::F32)?;
            let eps = match guided {
                true => {
                    let (u, c) = (eps.narrow(0, 0, 1)?, eps.narrow(0, 1, 1)?);
                    (&u + ((c - &u)? * req.guidance as f64)?)?
                }
                false => eps,
            };
            let preview = match req.preview {
                true => {
                    let clean = (&x - (&eps * sched.sigmas[i])?)?;
                    Some(latent_preview(&(clean / self.vae.config().scaling)?, &PREVIEW, PREVIEW_BIAS)?)
                }
                false => {
                    settle(&self.device)?;
                    None
                }
            };
            x = sched.stepped(&x, &eps, i, req.seed)?;
            if let Some(e) = &edited {
                x = e.hold(x, &sched, i)?;
            }
            if !on_step(Step { done: i + 1 - first, total: sched.steps() - first, preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        check_latent(&x)?;
        let t2 = Instant::now();
        let pixels = self.vae.decode(&x.to_dtype(self.dtype)?)?;
        let image = to_rgb8(&pixels)?;
        let image = match &edited {
            Some(e) => e.paste(image),
            None => image,
        };
        let decode_secs = t2.elapsed().as_secs_f64();
        Ok(Painted { image, request: req, encode_secs, denoise_secs, decode_secs })
    }
}

impl Painter for Sd15 {
    /// With the request's LoRAs set for it and taken off after.
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let (adapters, device, dtype) = (self.adapters.clone(), self.device.clone(), self.dtype);
        lora::painting(&adapters, req, &device, dtype, || self.draw(req, on_step))
    }

    fn defaults(&self) -> Defaults {
        // The model was trained at 512²; 25 steps and guidance 7.5 are what
        // its examples use.
        Defaults { width: 512, height: 512, steps: 25, guidance: 7.5, multiple: 8, takes_guidance: true, takes_negative: true, takes_loras: true, edits: true }
    }

    fn summary(&self) -> String {
        format!("Stable Diffusion 1.5, {:.2} B parameters", self.params as f64 / 1e9)
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        self.params * self.dtype.size_in_bytes()
    }

    fn backend(&self) -> String {
        crate::common::label(&self.device, self.dtype, None)
    }
}

/// What a checkpoint in one file reads besides the file: the tokenizer, and
/// the base's configs and scheduler. What its pull fetches.
pub(crate) fn fetch_base(progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<()> {
    progress(&format!("fetching {REPO}'s configs"));
    fetch_file(TOKENIZER_REPO, "tokenizer.json", watch)?;
    for f in ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "unet/config.json", "vae/config.json"] {
        fetch_file(REPO, f, watch)?;
    }
    Ok(())
}

/// [`weight_bytes`] for a checkpoint in one file: what the three loaders
/// read of it, in f16. Not the VAE's encoder, `ldm`'s training state or an
/// EMA's copy of the UNet, which in a file that kept it doubles the UNet.
pub(crate) fn single_weight_bytes(file: &Path) -> Option<u64> {
    // SAFETY: a read-only file, its header only.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file).ok()? };
    let read = |n: &str| {
        (n.starts_with("model.diffusion_model.") || n.starts_with("cond_stage_model.") || n.starts_with("first_stage_model.decoder.") || n.starts_with("first_stage_model.post_quant_conv."))
            && !n.ends_with("position_ids")
    };
    Some(st.tensors().iter().filter(|(n, _)| read(n)).map(|(_, v)| v.shape().iter().product::<usize>() as u64 * 2).sum())
}

/// What the pipeline will hold, from the files on this machine: the text
/// encoder, the UNet and the VAE, each in f16 from its header, whatever it
/// is stored in. `None` until each is here.
pub(crate) fn weight_bytes(repo: &str) -> Option<u64> {
    let f16 = |dir: &str| -> Option<u64> {
        let stem = if dir == "text_encoder" { "model" } else { "diffusion_pytorch_model" };
        // SAFETY: a read-only cache file, its header only.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(local_weights(repo, dir, stem)?).ok()? };
        Some(st.tensors().iter().map(|(_, v)| v.shape().iter().product::<usize>() as u64 * 2).sum())
    };
    Some(f16("text_encoder")? + f16("unet")? + f16("vae")?)
}

/// SD 1.5's four latent channels as colour, roughly: rows are channels (of
/// the latent divided by the VAE's scaling factor), columns R, G, B in
/// `[−1, 1]`. Fitted as SDXL's were (see `sdxl.rs`), by
/// `scripts/sd15-fixtures.py --preview`: one 512² image, 25 Euler steps,
/// seed 7, "a busy market street in marrakech, vivid colours, photograph",
/// each final latent pixel against the mean of the 8×8 pixels it decoded
/// to. It explains 74%, 72% and 62% of the variance in R, G and B.
const PREVIEW: [[f32; 3]; 4] = [[0.0684, 0.0437, 0.0456], [0.0354, 0.0630, 0.0220], [-0.0443, 0.0448, 0.0349], [-0.0431, -0.0607, -0.0941]];
const PREVIEW_BIAS: [f32; 3] = [0.0155, -0.1923, -0.3309];

#[cfg(test)]
mod tests {
    use super::*;

    /// A decode with any one of the VAE's stages left as zeros, as a failed
    /// Metal command buffer leaves it, and the rest run on them, is refused
    /// rather than saved; the decode that did not fail is not. In f16 on
    /// Metal as the pipeline runs it, at 512², from a latent of noise.
    ///
    ///     cargo test --release -p kvad-gpu sd15::tests::a_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_decode_that_fails_part_way_is_refused() {
        let w = Watcher::none();
        let config = read_json(&fetch_file(REPO, "vae/config.json", &w).unwrap()).unwrap();
        let file = vec![weights(REPO, "vae", "diffusion_pytorch_model", &w).unwrap()];
        let device = Device::new_metal(0).unwrap();
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype: DType::F16 };
        let vae = Decoder::load(&cx, &open(&file, DType::F16).unwrap(), VaeConfig::from_json(&config).unwrap()).unwrap();
        let z = noise(5, &[1, 4, 64, 64], &device, DType::F16).unwrap();

        to_rgb8(&vae.decode(&z).unwrap()).unwrap();
        for stage in 0..vae.stages() {
            let e = to_rgb8(&vae.decode_failing(&z, Some(stage)).unwrap()).err().map(|e| e.to_string());
            eprintln!("stage {stage}: {}", e.as_deref().unwrap_or("passed"));
            assert!(e.is_some(), "stage {stage} failed and was not refused");
        }
    }

    /// kvad's text encoder, UNet and VAE against diffusers' own, on the
    /// same weights and inputs: `scripts/sd15-fixtures.py` writes theirs,
    /// in f32. Each part runs from the reference's own input, in f32 on the
    /// CPU and in f16 on Metal, as the pipeline runs it.
    ///
    ///     KVAD_SD15_FIXTURES=/tmp/sd15-fx cargo test --release -p kvad-gpu sd15::tests -- --ignored --nocapture
    #[test]
    #[ignore]
    fn agrees_with_diffusers() {
        let dir = std::env::var("KVAD_SD15_FIXTURES").expect("KVAD_SD15_FIXTURES names the fixtures' directory");
        let fx = candle_core::safetensors::load(format!("{dir}/sd15.safetensors"), &Device::Cpu).unwrap();
        let get = |k: &str| fx.get(k).unwrap_or_else(|| panic!("no `{k}`")).clone();
        let db = |got: &Tensor, want: &Tensor| -> f32 {
            let (g, w) = (got.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap(), want.to_dtype(DType::F32).unwrap());
            let err = (&g - &w).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let sig = w.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            10.0 * (sig / err.max(1e-30)).log10()
        };
        let w = Watcher::none();
        let config = |dir: &str| read_json(&fetch_file(REPO, &format!("{dir}/config.json"), &w).unwrap()).unwrap();
        let file = |dir: &str, stem: &str| vec![weights(REPO, dir, stem, &w).unwrap()];

        // The tokens, as kvad's tokenizer makes them.
        let tok = tokenizers::Tokenizer::from_file(fetch_file(TOKENIZER_REPO, "tokenizer.json", &w).unwrap()).unwrap();
        let (ids, _) = clip::tokenize(&tok, "a red fox sitting in fresh snow, photograph", clip::END).unwrap();
        let theirs: Vec<u32> = get("ids").flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|&v| v as u32).collect();
        assert_eq!(ids, theirs, "the tokens");

        let metal = Device::new_metal(0).ok();
        let devices = [(Device::Cpu, DType::F32)].into_iter().chain(metal.map(|m| (m, DType::F16)));
        for (device, dtype) in devices {
            let vault = Vault::off();
            let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
            let r = open(&file("text_encoder", "model"), dtype).unwrap();
            let clip = Clip::load(&cx, &r, ClipConfig::from_json(&config("text_encoder")).unwrap(), Pooled::Last).unwrap();
            let hidden = clip.encode(&ids, 0).unwrap().0;

            let r = open(&file("unet", "diffusion_pytorch_model"), dtype).unwrap();
            let unet = Unet::load(&cx, &r, UnetConfig::from_json(&config("unet")).unwrap()).unwrap();
            let x = get("x").to_device(&device).unwrap().to_dtype(dtype).unwrap();
            let ctx = get("hidden").to_device(&device).unwrap().to_dtype(dtype).unwrap();
            let t = get("t").to_vec1::<f32>().unwrap()[0] as f64;
            let eps = unet.forward(&x, t, &ctx, None).unwrap();

            let r = open(&file("vae", "diffusion_pytorch_model"), dtype).unwrap();
            let vae = Decoder::load(&cx, &r, VaeConfig::from_json(&config("vae")).unwrap()).unwrap();
            let pixels = vae.decode(&get("z").to_device(&device).unwrap().to_dtype(dtype).unwrap()).unwrap();

            let (h, e, p) = (db(&hidden, &get("hidden")), db(&eps, &get("eps")), db(&pixels, &get("pixels")));
            eprintln!("{dtype:?} on {:?}: text encoder {h:.1} dB, UNet {e:.1} dB, VAE {p:.1} dB", device.location());
            let least = if dtype == DType::F32 { 80.0 } else { 30.0 };
            assert!(h > least && e > least && p > least, "{dtype:?}: {h:.1}, {e:.1}, {p:.1} dB");
        }
    }
}
