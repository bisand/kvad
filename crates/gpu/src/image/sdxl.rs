//! Stable Diffusion XL: two CLIPs, a UNet, a VAE and an Euler scheduler.
//!
//! [`Sdxl::paint`] is the whole of text-to-image in about sixty lines, and is
//! the place to start reading: tokenize, encode, draw noise, run the loop,
//! decode. Everything it calls is one of the other files in this directory.

use super::clip::{self, Clip, ClipConfig, Pooled};
use super::nn::{check_latent, latent_preview, noise, to_rgb8, Ctx};
use super::edit::Edited;
use super::schedule;
use super::unet::{Unet, UnetConfig};
use super::vae::{Decoder, Encoder, VaeConfig};
use super::lora::{self, Adapters};
use super::{finish, finish_mapped, local_file, open, open_file, open_mapped, read_json, single};
use crate::common::{settle, Loader, Reader};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor};
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::Value;
use kvad::weights::{fetch_file, Cached, Watcher};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub const REPO: &str = "stabilityai/stable-diffusion-xl-base-1.0";

/// SDXL's own VAE overflows f16 in its decoder, which is why its config says
/// `force_upcast`. This is the same decoder retrained to stay in range, so
/// the whole pipeline can run in one dtype. See the plan.
pub const VAE_REPO: &str = "madebyollin/sdxl-vae-fp16-fix";

/// The base repo ships CLIP's vocabulary as `vocab.json` and `merges.txt`
/// only; this repo has the same vocabulary as a `tokenizer.json`.
pub const TOKENIZER_REPO: &str = "openai/clip-vit-large-patch14";

/// How SDXL's four latent channels look, roughly, as colour: rows are
/// channels (of the latent divided by the VAE's scaling factor), columns R, G,
/// B in `[−1, 1]`.
///
/// Measured rather than remembered. One 1024² image (20 steps, seed 7, "a
/// busy market street in marrakech, vivid colours, photograph") was
/// generated, and each of its 128² final latent pixels was fitted by least
/// squares against the mean of the 8×8 block of pixels it decoded to. The
/// fit explains 83%, 81% and 76% of the variance in R, G and B — a blurry,
/// slightly wrong-coloured picture, which is what a preview is for.
pub(crate) const PREVIEW: [[f32; 3]; 4] =
    [[0.0550, 0.0538, 0.0513], [-0.0319, -0.0023, 0.0079], [0.0157, 0.0059, -0.0009], [-0.0416, -0.0268, -0.0241]];
pub(crate) const PREVIEW_BIAS: [f32; 3] = [0.0897, -0.1454, -0.1718];

/// A component's weights, `dir/stem`: the `.fp16` variant where the repo
/// ships one, as Stability's does beside its f32 files, and the plain file
/// otherwise, as nearly every fine-tune does, in f16 already. Either is read
/// in f16, whatever it is stored in.
///
/// The cache is asked first, so a repo that is here asks the Hub nothing;
/// then the Hub, `.fp16` first, so that a repo with both never downloads its
/// f32 file.
pub(crate) fn weights(repo: &str, dir: &str, stem: &str, watch: &Watcher) -> Res<PathBuf> {
    let (fp16, plain) = (format!("{dir}/{stem}.fp16.safetensors"), format!("{dir}/{stem}.safetensors"));
    for f in [&fp16, &plain] {
        if let Cached::Here(p) = kvad::weights::cached(repo, f) {
            return Ok(p);
        }
    }
    fetch_file(repo, &fp16, watch).or_else(|_| fetch_file(repo, &plain, watch)).map_err(|e| format!("{repo} has neither {fp16} nor {plain}: {e}").into())
}

/// [`weights`], asked of this machine only.
pub(crate) fn local_weights(repo: &str, dir: &str, stem: &str) -> Option<PathBuf> {
    local_file(repo, &format!("{dir}/{stem}.fp16.safetensors")).or_else(|| local_file(repo, &format!("{dir}/{stem}.safetensors")))
}

/// What the pipeline will hold, from the files on this machine: both text
/// encoders and the UNet in f16, from their headers, whatever they are
/// stored in, and the VAE, which is f32 on disk, in f16 too. Not
/// downloaded yet is not a reason to refuse for the VAE; it is small.
pub(crate) fn weight_bytes(repo: &str, size: &dyn Fn(&str, &str) -> Option<u64>) -> Option<u64> {
    let f16 = |dir: &str, stem: &str| -> Option<u64> {
        let path = local_weights(repo, dir, stem)?;
        // SAFETY: a read-only cache file, its header only.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path).ok()? };
        Some(st.tensors().iter().map(|(_, v)| v.shape().iter().product::<usize>() as u64 * 2).sum())
    };
    let own = f16("text_encoder", "model")? + f16("text_encoder_2", "model")? + f16("unet", "diffusion_pytorch_model")?;
    let vae = size(VAE_REPO, "diffusion_pytorch_model.safetensors").unwrap_or(335_000_000) / 2;
    Some(own + vae)
}

/// What a checkpoint in one file reads besides the file: the tokenizer, the
/// base's configs and scheduler, and the VAE. What its pull fetches.
pub(crate) fn fetch_base(progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<()> {
    progress(&format!("fetching {REPO}'s configs and {VAE_REPO}'s VAE"));
    fetch_file(TOKENIZER_REPO, "tokenizer.json", watch)?;
    for f in ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "text_encoder_2/config.json", "unet/config.json"] {
        fetch_file(REPO, f, watch)?;
    }
    for f in ["config.json", "diffusion_pytorch_model.safetensors"] {
        fetch_file(VAE_REPO, f, watch)?;
    }
    Ok(())
}

/// [`weight_bytes`] for a checkpoint in one file: everything in it but its
/// VAE, in f16, and the fp16-fix VAE.
pub(crate) fn single_weight_bytes(file: &Path, size: &dyn Fn(&str, &str) -> Option<u64>) -> Option<u64> {
    // SAFETY: a read-only file, its header only.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file).ok()? };
    let own: u64 = st.tensors().iter().filter(|(n, _)| !n.starts_with("first_stage_model.")).map(|(_, v)| v.shape().iter().product::<usize>() as u64 * 2).sum();
    let vae = size(VAE_REPO, "diffusion_pytorch_model.safetensors").unwrap_or(335_000_000) / 2;
    Some(own + vae)
}

pub struct Sdxl {
    tok: tokenizers::Tokenizer,
    clip_l: Clip,
    clip_g: Clip,
    unet: Unet,
    vae: Decoder,
    /// The VAE's other half, for an image made from a picture ([`edit`]).
    encoder: Encoder,
    scheduler: Value,
    device: Device,
    dtype: DType,
    /// The text encoders' and the UNet's layers, for LoRAs ([`lora`]).
    adapters: Adapters,
    params: usize,
}

/// What a LoRA's names for SDXL may start with, and the part each is in:
/// kohya's `lora_unet_`, `lora_te1_` and `lora_te2_`, diffusers' `unet.`,
/// `text_encoder.` and `text_encoder_2.`, and a UNet's layers named bare.
pub(crate) const PREFIXES: [(&str, &str); 7] = [
    ("lora_unet_", "unet"),
    ("unet.", "unet"),
    ("lora_te1_", "te1"),
    ("text_encoder.", "te1"),
    ("lora_te2_", "te2"),
    ("text_encoder_2.", "te2"),
    ("", "unet"),
];

impl Sdxl {
    pub fn load(repo: &str, device: Device, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Self> {
        Self::load_with(repo, None, device, progress, watch)
    }

    /// [`Sdxl::load`], with the UNet and both text encoders read from
    /// `single`, one file in Stability's layout ([`single`]). `repo` then
    /// gives only the tokenizer's settings, the components' configs and the
    /// scheduler, which a single file does not carry.
    pub fn load_with(repo: &str, single: Option<&Path>, device: Device, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Self> {
        let dtype = DType::F16;
        let get = |r: &str, f: &str| fetch_file(r, f, watch);
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };

        progress("reading the tokenizer");
        let tok = tokenizers::Tokenizer::from_file(get(TOKENIZER_REPO, "tokenizer.json")?).map_err(|e| e.to_string())?;
        let scheduler = read_json(&get(repo, "scheduler/scheduler_config.json")?)?;
        // Checked now rather than at the first request.
        schedule::euler(&scheduler, 30)?;

        let mut params = 0;
        let config = |dir: &str| -> Res<Value> { read_json(&get(repo, &format!("{dir}/config.json"))?) };
        // One file for all three, read through a map each, and their unread
        // weights counted together at the end; or each component's own file.
        let file = single.map(open_file).transpose()?;
        let maps = file.as_ref().map(|f| single::sdxl(f.names())).transpose()?;
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
        progress("loading the text encoders");
        let (r_l, paths) = reader("text_encoder", "model", maps.as_ref().map(|m| &m.clip_l))?;
        let r_l = r_l.with_adapters(adapters.part("te1"));
        let clip_l = Clip::load(&cx, &r_l, ClipConfig::from_json(&config("text_encoder")?)?, Pooled::No)?;
        if file.is_none() {
            params += finish("text encoder", &paths, &r_l)?;
        }

        let (r_g, paths) = reader("text_encoder_2", "model", maps.as_ref().map(|m| &m.clip_g))?;
        let r_g = r_g.with_adapters(adapters.part("te2"));
        let clip_g = Clip::load(&cx, &r_g, ClipConfig::from_json(&config("text_encoder_2")?)?, Pooled::Projected)?;
        if file.is_none() {
            params += finish("second text encoder", &paths, &r_g)?;
        }

        progress("loading the UNet");
        let (r, paths) = reader("unet", "diffusion_pytorch_model", maps.as_ref().map(|m| &m.unet))?;
        let r = r.with_adapters(adapters.part("unet"));
        let ucfg = UnetConfig::from_json(&config("unet")?)?;
        if ucfg.context != clip_l.width() + clip_g.width() {
            return Err(format!(
                "the UNet attends to {}-wide text and the two encoders give {} + {}",
                ucfg.context,
                clip_l.width(),
                clip_g.width()
            )
            .into());
        }
        let unet = Unet::load(&cx, &r, ucfg)?;
        adapters.alias_ldm("unet");
        match (&file, &maps) {
            (Some(f), Some(m)) => {
                let parts = [(&m.clip_l, &r_l), (&m.clip_g, &r_g), (&m.unet, &r)];
                let what = single.map(|p| p.display().to_string()).unwrap_or_default();
                params += finish_mapped(&what, f, &parts, &m.unread)?;
                progress(&format!("{} of the file's tensors left unread, its VAE's among them: the VAE is madebyollin's fp16-fix", m.unread.len()));
            }
            _ => params += finish("UNet", &paths, &r)?,
        }

        progress("loading the VAE");
        let (c, paths) = (read_json(&get(VAE_REPO, "config.json")?)?, vec![get(VAE_REPO, "diffusion_pytorch_model.safetensors")?]);
        let r = open(&paths, dtype)?;
        let vae = Decoder::load(&cx, &r, VaeConfig::from_json(&c)?)?;
        let encoder = Encoder::load(&cx, &r, VaeConfig::from_json(&c)?)?;
        params += finish("VAE", &paths, &r)?;

        settle(&device)?;
        progress(&format!("loaded SDXL: {:.2} B parameters in f16", params as f64 / 1e9));
        Ok(Sdxl { tok, clip_l, clip_g, unet, vae, encoder, scheduler, device, dtype, adapters, params })
    }

    /// The prompt as the UNet reads it: `[1, 77, 2048]` per token and
    /// `[1, 1280]` pooled.
    fn encode(&self, text: &str) -> Res<(Tensor, Tensor)> {
        read_prompt(&self.tok, &self.clip_l, &self.clip_g, text)
    }
}

/// `text` as the two encoders read it between them: every token from both,
/// side by side, and the second's pooled summary.
fn read_prompt(tok: &tokenizers::Tokenizer, clip_l: &Clip, clip_g: &Clip, text: &str) -> Res<(Tensor, Tensor)> {
    let (ids, _) = clip::tokenize(tok, text, clip::END)?;
    let (l, _) = clip_l.encode(&ids, 0)?;
    // The second tokenizer pads with `!`, id 0, where the first pads with
    // the end marker. Same vocabulary otherwise.
    let (ids, end) = clip::tokenize(tok, text, 0)?;
    let (g, pooled) = clip_g.encode(&ids, end)?;
    Ok((Tensor::cat(&[&l, &g], 2)?, pooled.expect("the second encoder is loaded pooled")))
}

/// What training reads a picture and its caption with (#75): both text
/// encoders and the VAE's encoding half, and no UNet.
///
/// Training reads each picture and each caption once, before its first
/// step, and lets these go: the answers do not change while a LoRA on the
/// UNet is trained, and the UNet then has the memory to itself.
pub(crate) struct Readers {
    tok: tokenizers::Tokenizer,
    clip_l: Clip,
    clip_g: Clip,
    vae: Encoder,
    device: Device,
    dtype: DType,
}

impl Readers {
    /// From `repo`, a pipeline in diffusers' layout; the VAE is the one
    /// drawing decodes with ([`VAE_REPO`]).
    pub(crate) fn load(repo: &str, device: &Device, watch: &Watcher) -> Res<Self> {
        let dtype = DType::F16;
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let tok = tokenizers::Tokenizer::from_file(fetch_file(TOKENIZER_REPO, "tokenizer.json", watch)?).map_err(|e| e.to_string())?;
        let clip = |dir: &str, what: &str, pooled: Pooled| -> Res<Clip> {
            let paths = vec![weights(repo, dir, "model", watch)?];
            let r = open(&paths, dtype)?;
            let cfg = ClipConfig::from_json(&read_json(&fetch_file(repo, &format!("{dir}/config.json"), watch)?)?)?;
            let clip = Clip::load(&cx, &r, cfg, pooled)?;
            finish(what, &paths, &r)?;
            Ok(clip)
        };
        let (clip_l, clip_g) = (clip("text_encoder", "text encoder", Pooled::No)?, clip("text_encoder_2", "second text encoder", Pooled::Projected)?);
        let (c, paths) = (read_json(&fetch_file(VAE_REPO, "config.json", watch)?)?, vec![fetch_file(VAE_REPO, "diffusion_pytorch_model.safetensors", watch)?]);
        let r = open(&paths, dtype)?;
        let vae = Encoder::load(&cx, &r, VaeConfig::from_json(&c)?)?;
        finish("VAE", &paths, &r)?;
        settle(device)?;
        Ok(Readers { tok, clip_l, clip_g, vae, device: device.clone(), dtype })
    }

    /// A caption as the UNet reads it: `[1, 77, 2048]` a token, and
    /// `[1, 1280]` pooled.
    pub(crate) fn caption(&self, text: &str) -> Res<(Tensor, Tensor)> {
        let read = read_prompt(&self.tok, &self.clip_l, &self.clip_g, text)?;
        settle(&self.device)?;
        Ok(read)
    }

    /// A picture, `[1, 3, H, W]` in `[−1, 1]` on the host, as the Gaussian
    /// over its latents the VAE says it is, in the UNet's units and in
    /// f32: the mean, and the spread in each number.
    pub(crate) fn picture(&self, pixels: &Tensor) -> Res<(Tensor, Tensor)> {
        let seen = self.vae.encode(&pixels.to_dtype(self.dtype)?.to_device(&self.device)?)?;
        let scaling = self.vae.config().scaling;
        let mean = self.vae.to_denoiser(&seen.mean.to_dtype(DType::F32)?)?;
        let spread = (seen.logvar.to_dtype(DType::F32)?.affine(0.5, 0.0)?.exp()? * scaling)?;
        settle(&self.device)?;
        Ok((mean, spread))
    }
}

impl Sdxl {
    /// One image, with whatever LoRAs the adapters hold.
    fn draw(&mut self, asked: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let req = asked.resolved(&self.defaults())?;
        let t0 = Instant::now();

        // The prompt, and what guidance steers away from. SDXL's pipeline
        // does not encode an empty negative prompt: it uses zeros
        // (`force_zeros_for_empty_prompt`), and so does this.
        let (ctx, pooled) = self.encode(&req.prompt)?;
        let guided = req.guidance > 1.0;
        let (ctx, pooled) = match (guided, &req.negative_prompt) {
            (false, _) => (ctx, pooled),
            (true, Some(neg)) => {
                let (nc, np) = self.encode(neg)?;
                (Tensor::cat(&[&nc, &ctx], 0)?, Tensor::cat(&[&np, &pooled], 0)?)
            }
            (true, None) => (Tensor::cat(&[&ctx.zeros_like()?, &ctx], 0)?, Tensor::cat(&[&pooled.zeros_like()?, &pooled], 0)?),
        };
        settle(&self.device)?;
        let encode_secs = t0.elapsed().as_secs_f64();

        // Original size, crop origin, target size: the conditioning SDXL was
        // trained with to stop it drawing the crops it was trained on.
        let (h, w) = (req.height as f64, req.width as f64);
        let time_ids = [h, w, 0.0, 0.0, h, w];

        let sched = schedule::euler(&self.scheduler, req.steps)?;
        let f = self.vae.config().factor();
        let shape = [1, 4, req.height / f, req.width / f];
        // The latent is kept in f32 between steps: the UNet's answer is f16,
        // but thirty small updates accumulated in f16 lose the last of them.
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
            let eps = self.unet.forward(&xin, sched.timesteps[i], &ctx, Some((&pooled, &time_ids)))?.to_dtype(DType::F32)?;
            let eps = match guided {
                true => {
                    let (u, c) = (eps.narrow(0, 0, 1)?, eps.narrow(0, 1, 1)?);
                    (&u + ((c - &u)? * req.guidance as f64)?)?
                }
                false => eps,
            };
            // What the model thinks the finished latent is from here: the
            // current one minus all the noise it predicts is in it. That is
            // what the preview shows, rather than the noisy latent itself.
            let preview = match req.preview {
                true => {
                    let clean = (&x - (&eps * sched.sigmas[i])?)?;
                    Some(latent_preview(&(clean / self.vae.config().scaling)?, &PREVIEW, PREVIEW_BIAS)?)
                }
                false => {
                    // Without a preview nothing reads the step back, and the
                    // GPU would run ahead of the progress report.
                    settle(&self.device)?;
                    None
                }
            };
            x = (&x + (eps * sched.dt(i))?)?;
            if let Some(e) = &edited {
                x = e.hold(x, &sched, i)?;
            }
            if !on_step(Step { done: i + 1 - first, total: sched.steps() - first, preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        // A failed command buffer on Metal leaves zeros behind rather than an
        // error, and an overflow leaves NaN. Either one is worth a sentence
        // rather than a black square.
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

impl Painter for Sdxl {
    /// With the request's LoRAs set for it and taken off after.
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let (adapters, device, dtype) = (self.adapters.clone(), self.device.clone(), self.dtype);
        lora::painting(&adapters, req, &device, dtype, || self.draw(req, on_step))
    }

    fn defaults(&self) -> Defaults {
        // 30 steps and guidance 5: the middle of what Stability's own
        // examples use. The model was trained at 1024², and the VAE needs
        // multiples of 8.
        Defaults { width: 1024, height: 1024, steps: 30, guidance: 5.0, multiple: 8, takes_guidance: true, takes_loras: true, edits: true }
    }

    fn summary(&self) -> String {
        format!("SDXL, {:.2} B parameters", self.params as f64 / 1e9)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    /// SDXL's two text encoders, whole and with their real weights, as
    /// functions of the embeddings they are given and of a LoRA on two
    /// layers of each: the gradient `backward` finds is the function's own.
    /// The embeddings are where a learned token goes in (textual
    /// inversion), and the LoRA is what `lora_te1_`/`lora_te2_` train.
    ///
    /// ViT-L is read at its penultimate layer, and bigG there and at its
    /// pooled, projected end-of-text row; the loss weighs every number of
    /// both, so each layer, the final norm and the projection are on the
    /// way. Certified in f64 on the CPU and held to that after, as
    /// `flux::tests::a_real_block_has_a_whole_gradient` is, which has the
    /// reasoning.
    ///
    ///     cargo test --release -p kvad-gpu sdxl::tests::the_text_encoders -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_text_encoders_have_whole_gradients() {
        crate::cap::at(24.0);
        let w = Watcher::none();
        let config = |dir: &str| read_json(&fetch_file(REPO, &format!("{dir}/config.json"), &w).unwrap()).unwrap();
        let mut runs = vec![(Device::Cpu, DType::F64, 1e-4, 0.0, "the CPU, f64"), (Device::Cpu, DType::F32, 1e-2, 1e-3, "the CPU, f32")];
        if let Ok(metal) = Device::new_metal(0) {
            runs.push((metal.clone(), DType::F32, 1e-2, 1e-3, "Metal, f32"));
            // As the pipeline runs them.
            runs.push((metal, DType::F16, 1e-2, 2e-2, "Metal, f16"));
        }
        // The start marker, eight tokens of a prompt, the end marker.
        let mut ids: Vec<u32> = vec![clip::START, 320, 1125, 2368, 4919, 530, 3293, 2583, 1215, clip::END];
        let end = ids.len() - 1;
        for (dir, part, pooled, pad, layers) in [
            ("text_encoder", "te1", Pooled::No, clip::END, ["text_model.encoder.layers.0.self_attn.q_proj", "text_model.encoder.layers.5.mlp.fc1"]),
            ("text_encoder_2", "te2", Pooled::Projected, 0, ["text_model.encoder.layers.0.self_attn.v_proj", "text_model.encoder.layers.31.mlp.fc2"]),
        ] {
            ids.truncate(end + 1);
            ids.resize(clip::CONTEXT, pad);
            let cfg = ClipConfig::from_json(&config(dir)).unwrap();
            let paths = vec![weights(REPO, dir, "model", &w).unwrap()];
            let mut exact = crate::grad::real::Exact::default();
            for (dev, dtype, step, tolerance, what) in &runs {
                let (dtype, what) = (*dtype, format!("{dir}, {what}"));
                let mut run = exact.run(&what, *tolerance);
                let vault = Vault::off();
                let cx = Ctx { ld: Loader::new(None, dev.clone(), &vault), dtype };
                let adapters = Adapters::new(&PREFIXES);
                let r = open(&paths, dtype).unwrap().with_adapters(adapters.part(part));
                let model = Clip::load(&cx, &r, cfg, pooled).unwrap();

                let seed = std::cell::Cell::new(74u64);
                let randn = |shape: &[usize], std: f32| {
                    seed.set(seed.get() + 1);
                    (noise(seed.get(), shape, dev, DType::F32).unwrap() * std as f64).unwrap().to_dtype(dtype).unwrap()
                };
                let x = model.embed(&ids).unwrap();
                let scale = x.to_dtype(DType::F32).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
                let (weigh, weigh_pooled) = (randn(&[1, clip::CONTEXT, cfg.width], 1.0), randn(&[1, cfg.width], 1.0));
                let loss = |x: &Tensor| -> candle_core::Result<Tensor> {
                    let (hidden, pooled) = model.read(x, end).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    let l = (hidden * &weigh)?.sum_all()?;
                    match pooled {
                        Some(p) => l + (p * &weigh_pooled)?.sum_all()?,
                        None => Ok(l),
                    }
                };

                // Rank 4, neither factor zero, so that both get a gradient.
                const SCALE: f32 = 0.05;
                let factors: Vec<(Var, Var)> = layers
                    .iter()
                    .map(|name| {
                        let (inp, out) = if name.ends_with("fc1") { (cfg.width, cfg.inter) } else if name.ends_with("fc2") { (cfg.inter, cfg.width) } else { (cfg.width, cfg.width) };
                        let (a, b) = (Var::from_tensor(&randn(&[inp, 4], SCALE)).unwrap(), Var::from_tensor(&randn(&[4, out], SCALE)).unwrap());
                        adapters.place(part, name, a.as_tensor(), b.as_tensor()).unwrap();
                        (a, b)
                    })
                    .collect();

                run.take("the embeddings", crate::grad::directional(&loss, &x, step * scale, 11).unwrap());
                for (name, (a, b)) in layers.iter().zip(&factors) {
                    run.factors(&adapters, part, name, (a, b), &|| loss(&x), step * SCALE as f64);
                }
            }
        }
    }
}
