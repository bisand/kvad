//! Stable Diffusion XL: two CLIPs, a UNet, a VAE and an Euler scheduler.
//!
//! [`Sdxl::paint`] is the whole of text-to-image in about sixty lines, and is
//! the place to start reading: tokenize, encode, draw noise, run the loop,
//! decode. Everything it calls is one of the other files in this directory.

use super::clip::{self, Clip, ClipConfig};
use super::nn::{latent_preview, noise, to_rgb8, Ctx};
use super::schedule;
use super::unet::{Unet, UnetConfig};
use super::vae::{Decoder, VaeConfig};
use super::{finish, open, read_json};
use crate::common::Loader;
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor};
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::Value;
use kvad::weights::{fetch_file, Watcher};
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
const PREVIEW: [[f32; 3]; 4] =
    [[0.0550, 0.0538, 0.0513], [-0.0319, -0.0023, 0.0079], [0.0157, 0.0059, -0.0009], [-0.0416, -0.0268, -0.0241]];
const PREVIEW_BIAS: [f32; 3] = [0.0897, -0.1454, -0.1718];

pub struct Sdxl {
    tok: tokenizers::Tokenizer,
    clip_l: Clip,
    clip_g: Clip,
    unet: Unet,
    vae: Decoder,
    scheduler: Value,
    device: Device,
    dtype: DType,
    params: usize,
}

impl Sdxl {
    pub fn load(repo: &str, device: Device, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Self> {
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
        let part = |dir_repo: &str, config: &str, weights: &str| -> Res<(Value, Vec<std::path::PathBuf>)> {
            Ok((read_json(&get(dir_repo, config)?)?, vec![get(dir_repo, weights)?]))
        };

        progress("loading the text encoders");
        let (c, paths) = part(repo, "text_encoder/config.json", "text_encoder/model.fp16.safetensors")?;
        let r = open(&paths, dtype)?;
        let clip_l = Clip::load(&cx, &r, ClipConfig::from_json(&c)?, false)?;
        params += finish("text encoder", &paths, &r)?;

        let (c, paths) = part(repo, "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors")?;
        let r = open(&paths, dtype)?;
        let clip_g = Clip::load(&cx, &r, ClipConfig::from_json(&c)?, true)?;
        params += finish("second text encoder", &paths, &r)?;

        progress("loading the UNet");
        let (c, paths) = part(repo, "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors")?;
        let r = open(&paths, dtype)?;
        let ucfg = UnetConfig::from_json(&c)?;
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
        params += finish("UNet", &paths, &r)?;

        progress("loading the VAE");
        let (c, paths) = part(VAE_REPO, "config.json", "diffusion_pytorch_model.safetensors")?;
        let r = open(&paths, dtype)?;
        let vae = Decoder::load(&cx, &r, VaeConfig::from_json(&c)?)?;
        params += finish("VAE", &paths, &r)?;

        device.synchronize()?;
        progress(&format!("loaded SDXL: {:.2} B parameters in f16", params as f64 / 1e9));
        Ok(Sdxl { tok, clip_l, clip_g, unet, vae, scheduler, device, dtype, params })
    }

    /// The prompt as the UNet reads it: `[1, 77, 2048]` per token and
    /// `[1, 1280]` pooled.
    fn encode(&self, text: &str) -> Res<(Tensor, Tensor)> {
        let (ids, _) = clip::tokenize(&self.tok, text, clip::END)?;
        let (l, _) = self.clip_l.encode(&ids, 0)?;
        // The second tokenizer pads with `!`, id 0, where the first pads with
        // the end marker. Same vocabulary otherwise.
        let (ids, end) = clip::tokenize(&self.tok, text, 0)?;
        let (g, pooled) = self.clip_g.encode(&ids, end)?;
        Ok((Tensor::cat(&[&l, &g], 2)?, pooled.expect("the second encoder is loaded pooled")))
    }
}

impl Painter for Sdxl {
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let req = req.resolved(&self.defaults())?;
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
        self.device.synchronize()?;
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
        let mut x = (noise(req.seed, &shape, &self.device, DType::F32)? * sched.init_scale)?;

        let t1 = Instant::now();
        for i in 0..sched.steps() {
            let xin = (&x * sched.input_scale(i))?.to_dtype(self.dtype)?;
            let xin = if guided { Tensor::cat(&[&xin, &xin], 0)? } else { xin };
            let eps = self.unet.forward(&xin, sched.timesteps[i], &ctx, &pooled, &time_ids)?.to_dtype(DType::F32)?;
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
                    self.device.synchronize()?;
                    None
                }
            };
            x = (&x + (eps * sched.dt(i))?)?;
            if !on_step(Step { done: i + 1, total: sched.steps(), preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        // A failed command buffer on Metal leaves zeros behind rather than an
        // error, and an overflow leaves NaN. Either one is worth a sentence
        // rather than a black square.
        let worst = x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        if !worst.is_finite() || worst == 0.0 {
            return Err(format!("the denoiser's result is {worst} everywhere it is largest; not decoding it").into());
        }

        let t2 = Instant::now();
        let pixels = self.vae.decode(&x.to_dtype(self.dtype)?)?;
        let image = to_rgb8(&pixels)?;
        let decode_secs = t2.elapsed().as_secs_f64();
        Ok(Painted { image, request: req, encode_secs, denoise_secs, decode_secs })
    }

    fn defaults(&self) -> Defaults {
        // 30 steps and guidance 5: the middle of what Stability's own
        // examples use. The model was trained at 1024², and the VAE needs
        // multiples of 8.
        Defaults { width: 1024, height: 1024, steps: 30, guidance: 5.0, multiple: 8 }
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
