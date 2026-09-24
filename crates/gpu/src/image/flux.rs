//! FLUX.1-schnell: an MMDiT like Qwen-Image's, with two text encoders and a
//! second kind of block.
//!
//! - **Two text encoders, for two jobs.** T5-XXL reads the prompt as a
//!   sentence, 256 tokens of it, and its hidden states are the text stream
//!   the transformer attends to. CLIP-L reads it as 77 tokens and contributes
//!   one pooled vector, which is added to the time embedding: CLIP says what
//!   the picture is *of*, T5 says what the prompt *says*.
//! - **Nineteen double blocks, then thirty-eight single ones.** The double
//!   blocks are Qwen-Image's, weight for weight in shape ([`super::mmdit`]).
//!   After them the two streams are one sequence, and each single block runs
//!   attention and MLP side by side on it.
//! - **Positions are not centred.** A patch's RoPE position is its row and
//!   column counted from the top left, and every text token is at the origin.
//!   Qwen-Image centred its image and put its text on the diagonal after it;
//!   FLUX, the older model, did neither.
//! - **No guidance.** Schnell is distilled to make an image in one to four
//!   steps without it, so each step is one forward pass. (FLUX.1-dev instead
//!   takes the guidance scale as an input to the time embedding; that variant
//!   is refused here rather than run without its input.)

use super::clip::{self, Clip, ClipConfig, Pooled};
use super::mmdit::{norm_out, Double, Names, Shape, Single};
use super::nn::{latent_preview, noise, timestep_embedding, to_rgb8, Ctx, Linear};
use super::schedule;
use super::t5::T5;
use super::vae::{Decoder, VaeConfig};
use super::{finish, local_file, open, read_json};
use crate::common::{Loader, Reader};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::{json, Value};
use kvad::weights::{fetch_file, Watcher};
use std::path::PathBuf;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The prompt as T5 sees it is always this many tokens, padded if shorter:
/// the reference pads to its maximum and never masks the padding, so the
/// model was trained attending to it.
const T5_TOKENS: usize = 256;

/// Latent channels.
const Z: usize = 16;

struct Config {
    double: usize,
    single: usize,
    shape: Shape,
    in_channels: usize,
    joint: usize,
    pooled: usize,
    axes: [usize; 3],
}

impl Config {
    fn from_json(v: &Value) -> Res<Self> {
        let n = |k: &str| -> Res<usize> {
            v.get(k).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("transformer config has no `{k}`").into())
        };
        if v.get("guidance_embeds").and_then(Value::as_bool) == Some(true) {
            return Err(concat!(
                "this is a guidance-distilled FLUX (FLUX.1-dev), which takes the guidance scale as an input; ",
                "only FLUX.1-schnell is implemented"
            )
            .into());
        }
        if n("patch_size")? != 1 {
            return Err("FLUX with a transformer patch size other than 1 is not implemented".into());
        }
        let axes: Vec<usize> = v
            .get("axes_dims_rope")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect())
            .unwrap_or_else(|| vec![16, 56, 56]);
        Ok(Config {
            double: n("num_layers")?,
            single: n("num_single_layers")?,
            shape: Shape { heads: n("num_attention_heads")?, head_dim: n("attention_head_dim")? },
            in_channels: n("in_channels")?,
            joint: n("joint_attention_dim")?,
            pooled: n("pooled_projection_dim")?,
            axes: axes.try_into().map_err(|_| "`axes_dims_rope` should have three entries")?,
        })
    }
}

struct Transformer {
    cfg: Config,
    x_in: Linear,
    context_in: Linear,
    time1: Linear,
    time2: Linear,
    text1: Linear,
    text2: Linear,
    double: Vec<Double>,
    single: Vec<Single>,
    norm_out: Linear,
    proj_out: Linear,
}

impl Transformer {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: Config) -> Res<Self> {
        let (w, s) = (cfg.shape.width(), cfg.shape);
        let img = Names { modulate: "norm1.linear", mlp_in: "ff.net.0.proj", mlp_out: "ff.net.2" };
        let txt = Names { modulate: "norm1_context.linear", mlp_in: "ff_context.net.0.proj", mlp_out: "ff_context.net.2" };
        let double = (0..cfg.double)
            .map(|i| Double::load(cx, &r.pp(format!("transformer_blocks.{i}")), s, &img, &txt))
            .collect::<Res<Vec<_>>>()?;
        let single = (0..cfg.single)
            .map(|i| Single::load(cx, &r.pp(format!("single_transformer_blocks.{i}")), s))
            .collect::<Res<Vec<_>>>()?;
        let t = r.pp("time_text_embed");
        Ok(Transformer {
            x_in: Linear::load(cx, r, "x_embedder", cfg.in_channels, w, true)?,
            context_in: Linear::load(cx, r, "context_embedder", cfg.joint, w, true)?,
            time1: Linear::load(cx, &t, "timestep_embedder.linear_1", 256, w, true)?,
            time2: Linear::load(cx, &t, "timestep_embedder.linear_2", w, w, true)?,
            text1: Linear::load(cx, &t, "text_embedder.linear_1", cfg.pooled, w, true)?,
            text2: Linear::load(cx, &t, "text_embedder.linear_2", w, w, true)?,
            norm_out: Linear::load(cx, r, "norm_out.linear", w, 2 * w, true)?,
            proj_out: Linear::load(cx, r, "proj_out", w, cfg.in_channels, true)?,
            double,
            single,
            cfg,
        })
    }

    /// The velocity at noise level `sigma` for packed latents `x`
    /// (`[1, rows·cols, 64]`), T5's hidden states `txt` and CLIP's `pooled`.
    fn forward(&self, x: &Tensor, txt: &Tensor, pooled: &Tensor, sigma: f64, rows: usize, cols: usize) -> Res<Tensor> {
        let dev = x.device();
        let dtype = x.dtype();
        let (n_img, n_txt) = (x.dim(1)?, txt.dim(1)?);
        let mut img = self.x_in.forward(x)?;
        let mut txt = self.context_in.forward(txt)?;

        // The time embedding (σ as a timestep, ×1000) plus CLIP's summary of
        // the prompt, each through its own two-layer MLP.
        let t = timestep_embedding(&[sigma * 1000.0], 256, true, 0.0, dev)?.to_dtype(dtype)?;
        let temb = (self.time2.forward(&self.time1.forward(&t)?.silu()?)?
            + self.text2.forward(&self.text1.forward(pooled)?.silu()?)?)?;
        let temb = temb.silu()?;

        let half = self.cfg.shape.head_dim / 2;
        let angles = Tensor::from_vec(angles(&self.cfg, rows, cols, n_txt), (n_txt + n_img, half), dev)?;
        let (cos, sin) = (angles.cos()?.to_dtype(dtype)?, angles.sin()?.to_dtype(dtype)?);
        let (cos_txt, sin_txt) = (cos.narrow(0, 0, n_txt)?, sin.narrow(0, 0, n_txt)?);
        let (cos_img, sin_img) = (cos.narrow(0, n_txt, n_img)?, sin.narrow(0, n_txt, n_img)?);

        let s = self.cfg.shape;
        for block in &self.double {
            (img, txt) = block.forward(s, &img, &txt, &temb, (&cos_img, &sin_img), (&cos_txt, &sin_txt))?;
        }
        // One stream from here on, text first, as the RoPE table is.
        let mut joint = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single {
            joint = block.forward(s, &joint, &temb, (&cos, &sin))?;
        }
        let img = joint.narrow(1, n_txt, n_img)?;
        let img = norm_out(&self.norm_out, &img, &temb, s.width())?;
        Ok(self.proj_out.forward(&img)?)
    }
}

/// Rotation angles, `[text + patches, head_dim / 2]`, text first.
///
/// Each position is three numbers — frame, row, column — and each owns a
/// slice of the head (16, 56 and 56 wide). Text tokens are all `(0, 0, 0)`, so
/// they are not rotated at all; a patch is `(0, row, col)` from the top left.
fn angles(cfg: &Config, rows: usize, cols: usize, text: usize) -> Vec<f32> {
    let freqs = |dim: usize| -> Vec<f64> { (0..dim / 2).map(|i| 1.0 / 10000f64.powf(2.0 * i as f64 / dim as f64)).collect() };
    let (ft, fh, fw) = (freqs(cfg.axes[0]), freqs(cfg.axes[1]), freqs(cfg.axes[2]));
    let half = cfg.shape.head_dim / 2;
    let mut out = vec![0f32; text * half];
    out.reserve(rows * cols * half);
    for r in 0..rows {
        for c in 0..cols {
            out.extend(ft.iter().map(|_| 0f32));
            out.extend(fh.iter().map(|f| (r as f64 * f) as f32));
            out.extend(fw.iter().map(|f| (c as f64 * f) as f32));
        }
    }
    out
}

pub struct Flux {
    clip_tok: tokenizers::Tokenizer,
    t5_tok: tokenizers::Tokenizer,
    clip: Clip,
    t5: T5,
    dit: Transformer,
    vae: Decoder,
    scheduler: Value,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    params: usize,
    bytes: usize,
}

/// A component's config and weights: a shard index's worth, or one file.
fn component(repo: &str, dir: &str, weights: &str, watch: &Watcher) -> Res<(Value, Vec<PathBuf>)> {
    let config = read_json(&fetch_file(repo, &format!("{dir}/config.json"), watch)?)?;
    let paths = match fetch_file(repo, &format!("{dir}/{weights}.safetensors.index.json"), watch) {
        Ok(index) => {
            let map = read_json(&index)?;
            let mut shards: Vec<String> =
                map["weight_map"].as_object().ok_or("a shard index with no weight_map")?.values().filter_map(Value::as_str).map(str::to_string).collect();
            shards.sort();
            shards.dedup();
            shards.iter().map(|s| fetch_file(repo, &format!("{dir}/{s}"), watch)).collect::<Res<Vec<_>>>()?
        }
        Err(_) => vec![fetch_file(repo, &format!("{dir}/{weights}.safetensors"), watch)?],
    };
    Ok((config, paths))
}

/// Bytes for `params` weights at a quantisation, as candle stores them.
fn at(params: usize, quant: Option<GgmlDType>) -> u64 {
    match quant {
        None => params as u64 * 2,
        Some(q) => params as u64 * q.type_size() as u64 / q.block_size() as u64,
    }
}

impl Flux {
    pub fn load(
        repo: &str,
        quant: Option<GgmlDType>,
        device: Device,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Res<Self> {
        let dtype = if quant.is_some() { DType::F32 } else { DType::BF16 };
        let label = quant.map(crate::common::ggml_name).unwrap_or("bf16");

        progress("reading the tokenizers");
        // The repo's CLIP tokenizer is `vocab.json` and `merges.txt`; the
        // same vocabulary as a `tokenizer.json` is where SDXL gets it.
        let clip_tok = tokenizers::Tokenizer::from_file(fetch_file(super::sdxl::TOKENIZER_REPO, "tokenizer.json", watch)?)
            .map_err(|e| e.to_string())?;
        let t5_tok = tokenizers::Tokenizer::from_file(fetch_file(repo, "tokenizer_2/tokenizer.json", watch)?).map_err(|e| e.to_string())?;
        let scheduler = read_json(&fetch_file(repo, "scheduler/scheduler_config.json", watch)?)?;
        schedule::flow(&scheduler, 4, 4096)?;

        let mut params = 0;
        let mut bytes = 0u64;

        // CLIP is small, and is kept dense.
        progress("loading CLIP");
        let (config, paths) = component(repo, "text_encoder", "model", watch)?;
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let r = open(&paths, dtype)?;
        let clip = Clip::load(&cx, &r, ClipConfig::from_json(&config)?, Pooled::Normed)?;
        let n = finish("CLIP", &paths, &r)?;
        (params, bytes) = (params + n, bytes + (n * dtype.size_in_bytes()) as u64);

        progress(&format!("loading T5 at {label}"));
        let (config, paths) = component(repo, "text_encoder_2", "model", watch)?;
        let mut vault = Vault::open_as(&format!("{repo}/text_encoder_2"), &paths, json!({ "component": "t5" }), quant, progress);
        let t5 = {
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault), dtype };
            let r = open(&paths, DType::F32)?;
            let t5 = T5::load(&cx, &r, &config)?;
            let n = finish("T5", &paths, &r)?;
            (params, bytes) = (params + n, bytes + at(n, quant));
            t5
        };
        vault.finish(progress);

        progress(&format!("loading the transformer at {label}"));
        let (config, paths) = component(repo, "transformer", "diffusion_pytorch_model", watch)?;
        let cfg = Config::from_json(&config)?;
        let shape = json!({ "component": "transformer", "double": cfg.double, "single": cfg.single, "width": cfg.shape.width() });
        let mut vault = Vault::open_as(&format!("{repo}/transformer"), &paths, shape, quant, progress);
        let dit = {
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault), dtype };
            let r = open(&paths, DType::F32)?;
            let dit = Transformer::load(&cx, &r, cfg)?;
            let n = finish("transformer", &paths, &r)?;
            (params, bytes) = (params + n, bytes + at(n, quant));
            dit
        };
        vault.finish(progress);

        // The decoder is the one stage at full resolution, where f32
        // activations would be gigabytes; it keeps bf16, whose range is f32's.
        progress("loading the VAE");
        let config = read_json(&fetch_file(repo, "vae/config.json", watch)?)?;
        let paths = vec![fetch_file(repo, "vae/diffusion_pytorch_model.safetensors", watch)?];
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype: DType::BF16 };
        let r = open(&paths, DType::BF16)?;
        let vae = Decoder::load(&cx, &r, VaeConfig::from_json(&config)?)?;
        let n = finish("VAE", &paths, &r)?;
        (params, bytes) = (params + n, bytes + 2 * n as u64);

        device.synchronize()?;
        progress(&format!("loaded FLUX.1-schnell: {:.1} B parameters at {label}", params as f64 / 1e9));
        Ok(Flux { clip_tok, t5_tok, clip, t5, dit, vae, scheduler, device, dtype, quant, params, bytes: bytes as usize })
    }

    /// T5's hidden states, `[1, 256, 4096]`, and CLIP's pooled vector,
    /// `[1, 768]`.
    fn encode(&self, prompt: &str) -> Res<(Tensor, Tensor)> {
        let enc = self.t5_tok.encode(prompt, true).map_err(|e| e.to_string())?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        // Truncated keeping the end marker, then padded with id 0.
        if ids.len() > T5_TOKENS {
            let end = *ids.last().unwrap_or(&1);
            ids.truncate(T5_TOKENS - 1);
            ids.push(end);
        }
        ids.resize(T5_TOKENS, 0);
        let t5 = self.t5.forward(&ids, &self.device, self.dtype)?;

        let (ids, end) = clip::tokenize(&self.clip_tok, prompt, clip::END)?;
        let (_, pooled) = self.clip.encode(&ids, end)?;
        Ok((t5, pooled.expect("CLIP is loaded pooled").to_dtype(self.dtype)?))
    }
}

/// What the pipeline will hold at `quant`, from the checkpoint headers on the
/// disk. `None` until every shard is here.
pub(crate) fn weight_bytes(repo: &str, quant: Option<GgmlDType>, size: &dyn Fn(&str, &str) -> Option<u64>) -> Option<u64> {
    let params = |dir: &str, weights: &str| -> Option<usize> {
        let index = read_json(&local_file(repo, &format!("{dir}/{weights}.safetensors.index.json"))?).ok()?;
        let mut shards: Vec<&str> = index["weight_map"].as_object()?.values().filter_map(Value::as_str).collect();
        shards.sort();
        shards.dedup();
        let paths: Vec<PathBuf> = shards.iter().map(|s| local_file(repo, &format!("{dir}/{s}"))).collect::<Option<_>>()?;
        // SAFETY: read-only cache files, headers only.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&paths).ok()? };
        Some(st.tensors().iter().map(|(_, v)| v.shape().iter().product::<usize>()).sum())
    };
    let quantised = params("text_encoder_2", "model")? + params("transformer", "diffusion_pytorch_model")?;
    // CLIP and the VAE are small and held as they are; their files' sizes
    // are close enough.
    let small = size(repo, "text_encoder/model.safetensors")? + size(repo, "vae/diffusion_pytorch_model.safetensors")?;
    Some(at(quantised, quant) + small)
}

impl Painter for Flux {
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        // A guidance scale or a negative prompt is refused here, because the
        // defaults say this model takes neither.
        let req = req.resolved(&self.defaults())?;

        let t0 = Instant::now();
        let (txt, pooled) = self.encode(&req.prompt)?;
        self.device.synchronize()?;
        let encode_secs = t0.elapsed().as_secs_f64();

        let (rows, cols) = (req.height / 16, req.width / 16);
        let sched = schedule::flow(&self.scheduler, req.steps, rows * cols)?;
        let mut x = noise(req.seed, &[1, rows * cols, self.dit.cfg.in_channels], &self.device, DType::F32)?;

        let t1 = Instant::now();
        for i in 0..sched.steps() {
            let sigma = sched.timesteps[i];
            let v = self.dit.forward(&x.to_dtype(self.dtype)?, &txt, &pooled, sigma, rows, cols)?.to_dtype(DType::F32)?;
            let preview = match req.preview {
                true => {
                    let clean = (&x - (&v * sigma)?)?;
                    Some(latent_preview(&unpack(&clean, rows, cols)?, &PREVIEW, PREVIEW_BIAS)?)
                }
                false => {
                    self.device.synchronize()?;
                    None
                }
            };
            x = (&x + (v * sched.dt(i))?)?;
            if !on_step(Step { done: i + 1, total: sched.steps(), preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        let worst = x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        if !worst.is_finite() || worst == 0.0 {
            return Err(format!("the denoiser's result is {worst} everywhere it is largest; not decoding it").into());
        }
        let t2 = Instant::now();
        let pixels = self.vae.decode(&unpack(&x, rows, cols)?.to_dtype(DType::BF16)?)?;
        let image = to_rgb8(&pixels)?;
        let decode_secs = t2.elapsed().as_secs_f64();
        Ok(Painted { image, request: req, encode_secs, denoise_secs, decode_secs })
    }

    fn defaults(&self) -> Defaults {
        // Black Forest Labs' own settings for schnell: four steps, no
        // guidance, a megapixel.
        Defaults { width: 1024, height: 1024, steps: 4, guidance: 0.0, multiple: 16, takes_guidance: false }
    }

    fn summary(&self) -> String {
        format!("FLUX.1-schnell, {:.1} B parameters", self.params as f64 / 1e9)
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        self.bytes
    }

    fn backend(&self) -> String {
        crate::common::label(&self.device, self.dtype, self.quant)
    }
}

/// Packed latents back to a `[1, 16, 2·rows, 2·cols]` grid; the same packing
/// as Qwen-Image's.
fn unpack(x: &Tensor, rows: usize, cols: usize) -> candle_core::Result<Tensor> {
    x.reshape((1, rows, cols, Z, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((1, Z, rows * 2, cols * 2))
}

/// FLUX's sixteen latent channels (as the transformer sees them, before the
/// VAE's scale and shift) as colour, roughly: fitted by least squares the same
/// way as SDXL's (see `sdxl.rs`), from one 1024² image — 4 steps, seed 3, "a
/// red fox sitting in fresh snow next to a wooden sign that says "FLUX",
/// photograph". It explains 97%, 99% and 99% of the variance in R, G and B on
/// that image.
const PREVIEW: [[f32; 3]; Z] = [
    [-0.0045, 0.0508, 0.0870],
    [0.0170, 0.0428, 0.0794],
    [0.0414, -0.0170, -0.0216],
    [-0.0065, 0.0277, 0.0632],
    [0.0487, 0.0339, 0.0082],
    [0.0115, 0.0401, 0.0281],
    [0.0244, 0.0448, 0.0508],
    [-0.0390, -0.0411, -0.0519],
    [-0.0384, 0.0147, 0.0980],
    [0.0933, 0.0512, -0.0320],
    [0.0022, 0.0400, 0.0273],
    [0.0621, 0.0317, 0.0229],
    [0.0635, 0.0497, 0.0443],
    [-0.1258, -0.0943, -0.1175],
    [-0.0158, -0.0503, -0.0323],
    [-0.1037, -0.0788, -0.0493],
];
const PREVIEW_BIAS: [f32; 3] = [-0.0046, -0.0804, -0.1004];

#[cfg(test)]
mod tests {
    use super::*;

    /// The positions diffusers' `_prepare_latent_image_ids` gives, and text at
    /// the origin: rows and columns from the top left, not centred.
    #[test]
    fn text_is_at_the_origin_and_patches_count_from_the_corner() {
        let cfg = Config {
            double: 0,
            single: 0,
            shape: Shape { heads: 24, head_dim: 128 },
            in_channels: 64,
            joint: 4096,
            pooled: 768,
            axes: [16, 56, 56],
        };
        let (rows, cols, text) = (3, 5, 2);
        let a = angles(&cfg, rows, cols, text);
        let half = 64;
        assert_eq!(a.len(), (text + rows * cols) * half);
        assert!(a[..text * half].iter().all(|&x| x == 0.0), "text is not rotated");
        // Pair 0 of each axis has frequency 1, so its angle is the position.
        let at = |token: usize, pair: usize| a[(text + token) * half + pair];
        let (row, col) = (8, 8 + 28);
        assert_eq!((at(0, row), at(0, col)), (0.0, 0.0));
        assert_eq!((at(rows * cols - 1, row), at(rows * cols - 1, col)), (2.0, 4.0));
        assert_eq!(at(cols, row), 1.0, "the second row starts one row down");
    }
}
