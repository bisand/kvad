//! FLUX.1, schnell and dev: an MMDiT like Qwen-Image's, with two text
//! encoders and a second kind of block.
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
//! - **One pass a step, in both.** Schnell is distilled to make an image in
//!   one to four steps with no guidance at all. Dev is distilled another
//!   way, from a model guided by running it twice: the guidance scale is
//!   now a number it *reads*, embedded as the timestep is and added to it,
//!   so guidance 3.5 costs what guidance 1 does and there is nothing for a
//!   negative prompt to do. Which of the two a transformer is, its config
//!   says: `guidance_embeds`.

use super::clip::{self, Clip, ClipConfig, Pooled};
use super::lora::{self, Adapters};
use super::mmdit::{norm_out, Double, Names, Shape, Single};
use super::nn::{check_latent, latent_preview, noise, timestep_embedding, to_rgb8, Ctx, Linear};
use super::schedule;
use super::t5::T5;
use super::vae::{Decoder, VaeConfig};
use super::{finish, finish_gguf, local_file, open, read_json};
use crate::common::{settle, Loader, Reader};
use crate::gguf::{Gguf, Part};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use kvad::image::{Defaults, ImageRequest, Painted, Painter, Step};
use kvad::serde_json::{json, Value};
use kvad::pipeline::component;
use kvad::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
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
    /// Whether the guidance scale is an input: dev's, and not schnell's.
    guided: bool,
}

impl Config {
    fn from_json(v: &Value) -> Res<Self> {
        let n = |k: &str| -> Res<usize> {
            v.get(k).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("transformer config has no `{k}`").into())
        };
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
            guided: v.get("guidance_embeds").and_then(Value::as_bool) == Some(true),
        })
    }
}

/// Black Forest Labs' own layout, which city96's GGUFs keep, under the
/// names diffusers gives FLUX, which the loader asks for: diffusers' own
/// conversion (`convert_flux_to_diffusers.py`), backwards.
///
/// Three things differ besides the names:
/// - **q, k and v are one matrix**, `qkv`, in a double block's two streams,
///   and diffusers' three are its thirds, by rows.
/// - **A single block's `linear1` is four**: q, k and v, then the MLP's
///   input, `proj_mlp`, at four times the width.
/// - **The last modulation's halves are the other way round**: Black Forest
///   Labs' `adaLN_modulation` gives shift then scale, and diffusers'
///   `norm_out` scale then shift.
fn gguf_map(cfg: &Config) -> Vec<(String, Vec<Part>)> {
    let w = cfg.shape.width();
    let mut m: Vec<(String, Vec<Part>)> = Vec::new();
    // A linear layer's weight and bias, each made of the same rows.
    let mut lin = |to: &str, parts: &[(&str, std::ops::Range<usize>)]| {
        for kind in ["weight", "bias"] {
            m.push((format!("{to}.{kind}"), parts.iter().map(|(from, rows)| Part::rows(format!("{from}.{kind}"), rows.clone())).collect()));
        }
    };
    for i in 0..cfg.double {
        let (b, d) = (format!("transformer_blocks.{i}"), format!("double_blocks.{i}"));
        for (stream, modulate, qkv, out, mlp) in [
            ("img", "norm1.linear", ["attn.to_q", "attn.to_k", "attn.to_v"], "attn.to_out.0", "ff"),
            ("txt", "norm1_context.linear", ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"], "attn.to_add_out", "ff_context"),
        ] {
            lin(&format!("{b}.{modulate}"), &[(&format!("{d}.{stream}_mod.lin"), 0..6 * w)]);
            for (j, to) in qkv.iter().enumerate() {
                lin(&format!("{b}.{to}"), &[(&format!("{d}.{stream}_attn.qkv"), j * w..(j + 1) * w)]);
            }
            lin(&format!("{b}.{out}"), &[(&format!("{d}.{stream}_attn.proj"), 0..w)]);
            lin(&format!("{b}.{mlp}.net.0.proj"), &[(&format!("{d}.{stream}_mlp.0"), 0..4 * w)]);
            lin(&format!("{b}.{mlp}.net.2"), &[(&format!("{d}.{stream}_mlp.2"), 0..w)]);
        }
    }
    for i in 0..cfg.single {
        let (b, s) = (format!("single_transformer_blocks.{i}"), format!("single_blocks.{i}"));
        lin(&format!("{b}.norm.linear"), &[(&format!("{s}.modulation.lin"), 0..3 * w)]);
        for (j, to) in ["attn.to_q", "attn.to_k", "attn.to_v"].iter().enumerate() {
            lin(&format!("{b}.{to}"), &[(&format!("{s}.linear1"), j * w..(j + 1) * w)]);
        }
        lin(&format!("{b}.proj_mlp"), &[(&format!("{s}.linear1"), 3 * w..7 * w)]);
        lin(&format!("{b}.proj_out"), &[(&format!("{s}.linear2"), 0..w)]);
    }
    let t = "time_text_embed";
    lin("x_embedder", &[("img_in", 0..w)]);
    lin("context_embedder", &[("txt_in", 0..w)]);
    lin(&format!("{t}.timestep_embedder.linear_1"), &[("time_in.in_layer", 0..w)]);
    lin(&format!("{t}.timestep_embedder.linear_2"), &[("time_in.out_layer", 0..w)]);
    lin(&format!("{t}.text_embedder.linear_1"), &[("vector_in.in_layer", 0..w)]);
    lin(&format!("{t}.text_embedder.linear_2"), &[("vector_in.out_layer", 0..w)]);
    if cfg.guided {
        lin(&format!("{t}.guidance_embedder.linear_1"), &[("guidance_in.in_layer", 0..w)]);
        lin(&format!("{t}.guidance_embedder.linear_2"), &[("guidance_in.out_layer", 0..w)]);
    }
    lin("norm_out.linear", &[("final_layer.adaLN_modulation.1", w..2 * w), ("final_layer.adaLN_modulation.1", 0..w)]);
    lin("proj_out", &[("final_layer.linear", 0..cfg.in_channels)]);
    // The per-head norms have one name each way.
    let hd = cfg.shape.head_dim;
    for i in 0..cfg.double {
        let (b, d) = (format!("transformer_blocks.{i}.attn"), format!("double_blocks.{i}"));
        for (to, from) in [("norm_q", "img_attn.norm.query_norm"), ("norm_k", "img_attn.norm.key_norm"), ("norm_added_q", "txt_attn.norm.query_norm"), ("norm_added_k", "txt_attn.norm.key_norm")] {
            m.push((format!("{b}.{to}.weight"), vec![Part::all(format!("{d}.{from}.scale"), hd)]));
        }
    }
    for i in 0..cfg.single {
        let (b, s) = (format!("single_transformer_blocks.{i}.attn"), format!("single_blocks.{i}"));
        for (to, from) in [("norm_q", "norm.query_norm"), ("norm_k", "norm.key_norm")] {
            m.push((format!("{b}.{to}.weight"), vec![Part::all(format!("{s}.{from}.scale"), hd)]));
        }
    }
    m
}

/// What a LoRA's names for FLUX may start with, and the part each is in:
/// diffusers' `transformer.` and `text_encoder.`, ComfyUI's
/// `diffusion_model.`, kohya's `lora_unet_`, `lora_transformer_` and
/// `lora_te1_`, and the transformer's layers named bare. Under
/// `diffusion_model.` and `lora_unet_` the names are Black Forest Labs'
/// ([`bfl_loras`]).
pub(crate) const PREFIXES: [(&str, &str); 7] = [
    ("transformer.", "transformer"),
    ("diffusion_model.", "transformer"),
    ("lora_unet_", "transformer"),
    ("lora_transformer_", "transformer"),
    ("lora_te1_", "te1"),
    ("text_encoder.", "te1"),
    ("", "transformer"),
];

/// The transformer's layers also by Black Forest Labs' names, in which
/// kohya's FLUX LoRAs are made: [`gguf_map`]'s rows, so that a LoRA's pair
/// for a fused `qkv` or `linear1` is each of its layers', `B` cut as the
/// weights are, and one for the last modulation has its halves the other
/// way round as the weight does. Norms take no LoRA.
fn bfl_loras(adapters: &Adapters, map: &[(String, Vec<Part>)]) {
    for (to, parts) in map {
        let (Some(to), Some(first)) = (to.strip_suffix(".weight"), parts.first()) else { continue };
        let Some(from) = first.name.strip_suffix(".weight") else { continue };
        if parts.iter().all(|p| p.name == first.name) {
            adapters.fused("transformer", from, to, parts.iter().map(|p| p.rows.clone()).collect());
        }
    }
}

/// A GGUF of FLUX's transformer, under diffusers' names.
fn open_gguf(path: &Path, cfg: &Config) -> Res<Gguf> {
    let file = Gguf::open(path)?;
    match file.text("general.architecture") {
        Some("flux") => file.mapped(gguf_map(cfg)),
        other => Err(format!("{} is a GGUF of {}, not of FLUX's transformer", path.display(), other.unwrap_or("an unnamed architecture")).into()),
    }
}

/// Whether `repo`'s transformer is one this pipeline runs, from its config
/// alone: asked before a GGUF of it is downloaded, rather than after, at
/// its load.
pub(crate) fn runs(repo: &str, watch: &Watcher) -> Res<()> {
    Config::from_json(&read_json(&fetch_file(repo, "transformer/config.json", watch)?)?).map(|_| ())
}

/// [`open_gguf`], with the config read from `repo`, the file's base.
pub(crate) fn open_gguf_for(repo: &str, path: &Path, watch: &Watcher) -> Res<Gguf> {
    open_gguf(path, &Config::from_json(&read_json(&fetch_file(repo, "transformer/config.json", watch)?)?)?)
}

struct Transformer {
    cfg: Config,
    x_in: Linear,
    context_in: Linear,
    time1: Linear,
    time2: Linear,
    text1: Linear,
    text2: Linear,
    /// The guidance scale's own two layers, in a model that reads one.
    guidance: Option<(Linear, Linear)>,
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
            guidance: match cfg.guided {
                true => Some((Linear::load(cx, &t, "guidance_embedder.linear_1", 256, w, true)?, Linear::load(cx, &t, "guidance_embedder.linear_2", w, w, true)?)),
                false => None,
            },
            norm_out: Linear::load(cx, r, "norm_out.linear", w, 2 * w, true)?,
            proj_out: Linear::load(cx, r, "proj_out", w, cfg.in_channels, true)?,
            double,
            single,
            cfg,
        })
    }

    /// The velocity at noise level `sigma` for packed latents `x`
    /// (`[1, rows·cols, 64]`), T5's hidden states `txt` and CLIP's `pooled`,
    /// at the guidance scale `guidance`, which only a model that reads one
    /// (dev) does anything with.
    fn forward(&self, x: &Tensor, txt: &Tensor, pooled: &Tensor, sigma: f64, guidance: f64, rows: usize, cols: usize) -> Res<Tensor> {
        let dev = x.device();
        let dtype = x.dtype();
        let (n_img, n_txt) = (x.dim(1)?, txt.dim(1)?);
        let mut img = self.x_in.forward(x)?;
        let mut txt = self.context_in.forward(txt)?;

        // The time embedding (σ as a timestep, ×1000) plus CLIP's summary of
        // the prompt, each through its own two-layer MLP.
        let embedded = |n: f64| -> Res<Tensor> { Ok(timestep_embedding(&[n * 1000.0], 256, true, 0.0, dev)?.to_dtype(dtype)?) };
        let mut temb = self.time2.forward(&self.time1.forward(&embedded(sigma)?)?.silu()?)?;
        // The guidance scale the same way: the same sinusoids of it, ×1000
        // as σ is, through two layers of its own, added to the time's.
        if let Some((one, two)) = &self.guidance {
            temb = (temb + two.forward(&one.forward(&embedded(guidance)?)?.silu()?)?)?;
        }
        let temb = (temb + self.text2.forward(&self.text1.forward(pooled)?.silu()?)?)?;
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
    /// The GGUF the transformer came from, and what it is made of.
    gguf: Option<(String, String)>,
    /// CLIP's and the transformer's layers, for LoRAs ([`lora`]).
    adapters: Adapters,
    params: usize,
    bytes: usize,
}

/// What a transformer of this shape is called.
fn name_of(cfg: &Config) -> &'static str {
    match cfg.guided {
        true => "FLUX.1-dev",
        false => "FLUX.1-schnell",
    }
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
        Self::load_with(repo, None, quant, device, progress, watch)
    }

    /// [`Flux::load`], with the transformer read from `gguf`, in Black
    /// Forest Labs' layout ([`gguf_map`]); everything else is `repo`'s, T5
    /// at `quant`, or q8 when nothing is asked, as beside Qwen-Image's.
    pub fn load_with(
        repo: &str,
        gguf: Option<&Path>,
        quant: Option<GgmlDType>,
        device: Device,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Res<Self> {
        let quant = match (gguf, quant) {
            (Some(_), None) => {
                progress("T5 runs at q8 beside a GGUF's transformer");
                Some(GgmlDType::Q8_0)
            }
            _ => quant,
        };
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
        let adapters = Adapters::new(&PREFIXES);
        let r = open(&paths, dtype)?.with_adapters(adapters.part("te1"));
        let clip = Clip::load(&cx, &r, ClipConfig::from_json(&config)?, Pooled::Normed)?;
        let n = finish("CLIP", &paths, &r)?;
        (params, bytes) = (params + n, bytes + (n * dtype.size_in_bytes()) as u64);

        progress(&format!("loading T5 at {label}"));
        let (config, paths) = component(repo, "text_encoder_2", "model", watch)?;
        let mut vault = Vault::open_as(&format!("{repo}/text_encoder_2"), &paths, json!({ "component": "t5" }), quant, progress);
        let t5 = {
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
            let r = open(&paths, dtype)?;
            let t5 = T5::load(&cx, &r, &config)?;
            let n = finish("T5", &paths, &r)?;
            (params, bytes) = (params + n, bytes + at(n, quant));
            t5
        };
        vault.finish(progress);

        let (dit, made) = match gguf {
            Some(path) => {
                let cfg = Config::from_json(&read_json(&fetch_file(repo, "transformer/config.json", watch)?)?)?;
                let file = Arc::new(open_gguf(path, &cfg)?);
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                progress(&format!("loading the transformer from {name}: {}", file.make_up()));
                let vault = Vault::off();
                let cx = Ctx { ld: Loader::new(None, device.clone(), &vault).accelerated(), dtype };
                let r = Reader::gguf(Arc::clone(&file), dtype).with_adapters(adapters.part("transformer"));
                let map = gguf_map(&cfg);
                let dit = Transformer::load(&cx, &r, cfg)?;
                bfl_loras(&adapters, &map);
                let n = finish_gguf("transformer", &file, &r)?;
                (params, bytes) = (params + n, bytes + file.device_bytes(dtype) as u64);
                (dit, Some((name, file.make_up())))
            }
            None => {
                progress(&format!("loading the transformer at {label}"));
                let (config, paths) = component(repo, "transformer", "diffusion_pytorch_model", watch)?;
                let cfg = Config::from_json(&config)?;
                let shape = json!({ "component": "transformer", "double": cfg.double, "single": cfg.single, "width": cfg.shape.width() });
                let mut vault = Vault::open_as(&format!("{repo}/transformer"), &paths, shape, quant, progress);
                let dit = {
                    let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
                    let r = open(&paths, dtype)?.with_adapters(adapters.part("transformer"));
                    let map = gguf_map(&cfg);
                    let dit = Transformer::load(&cx, &r, cfg)?;
                    bfl_loras(&adapters, &map);
                    let n = finish("transformer", &paths, &r)?;
                    (params, bytes) = (params + n, bytes + at(n, quant));
                    dit
                };
                vault.finish(progress);
                (dit, None)
            }
        };

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

        settle(&device)?;
        let from = made.as_ref().map(|(name, _)| format!(", the transformer from {name}")).unwrap_or_default();
        progress(&format!("loaded {}: {:.1} B parameters at {label}{from}", name_of(&dit.cfg), params as f64 / 1e9));
        Ok(Flux { clip_tok, t5_tok, clip, t5, dit, vae, scheduler, device, dtype, quant, gguf: made, adapters, params, bytes: bytes as usize })
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
///
/// With `gguf`, the transformer is that file's, its plain tensors widened to
/// f32, and T5 is q8 if nothing else is asked, as [`Flux::load_with`] loads
/// them.
pub(crate) fn weight_bytes(repo: &str, quant: Option<GgmlDType>, size: &dyn Fn(&str, &str) -> Option<u64>, gguf: Option<&Path>) -> Option<u64> {
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
    // CLIP and the VAE are small and held as they are; their files' sizes
    // are close enough.
    let small = size(repo, "text_encoder/model.safetensors")? + size(repo, "vae/diffusion_pytorch_model.safetensors")?;
    match gguf {
        Some(file) => {
            let dit = Gguf::open(file).ok()?.device_bytes(DType::F32) as u64;
            Some(at(params("text_encoder_2", "model")?, quant.or(Some(GgmlDType::Q8_0))) + dit + small)
        }
        None => {
            let quantised = params("text_encoder_2", "model")? + params("transformer", "diffusion_pytorch_model")?;
            Some(at(quantised, quant) + small)
        }
    }
}

impl Flux {
    /// One image, with whatever LoRAs the adapters hold.
    fn draw(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        // What the defaults say this model does not take is refused here:
        // a guidance scale by schnell, a negative prompt by both.
        let req = req.resolved(&self.defaults())?;

        let t0 = Instant::now();
        let (txt, pooled) = self.encode(&req.prompt)?;
        settle(&self.device)?;
        let encode_secs = t0.elapsed().as_secs_f64();

        let (rows, cols) = (req.height / 16, req.width / 16);
        let sched = schedule::flow(&self.scheduler, req.steps, rows * cols)?;
        let mut x = noise(req.seed, &[1, rows * cols, self.dit.cfg.in_channels], &self.device, DType::F32)?;

        let t1 = Instant::now();
        for i in 0..sched.steps() {
            let sigma = sched.timesteps[i];
            let v = self.dit.forward(&x.to_dtype(self.dtype)?, &txt, &pooled, sigma, req.guidance as f64, rows, cols)?.to_dtype(DType::F32)?;
            let preview = match req.preview {
                true => {
                    let clean = (&x - (&v * sigma)?)?;
                    Some(latent_preview(&unpack(&clean, rows, cols)?, &PREVIEW, PREVIEW_BIAS)?)
                }
                false => {
                    settle(&self.device)?;
                    None
                }
            };
            x = (&x + (v * sched.dt(i))?)?;
            if !on_step(Step { done: i + 1, total: sched.steps(), preview }) {
                return Err("cancelled".into());
            }
        }
        let denoise_secs = t1.elapsed().as_secs_f64();

        check_latent(&x)?;
        let t2 = Instant::now();
        let pixels = self.vae.decode(&unpack(&x, rows, cols)?.to_dtype(DType::BF16)?)?;
        let image = to_rgb8(&pixels)?;
        let decode_secs = t2.elapsed().as_secs_f64();
        Ok(Painted { image, request: req, encode_secs, denoise_secs, decode_secs })
    }
}

impl Painter for Flux {
    /// With the request's LoRAs set for it and taken off after: their
    /// factors in the pipeline's dtype, or bf16 beside quantised weights.
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let factors = if self.dtype == DType::F32 { DType::BF16 } else { self.dtype };
        let (adapters, device) = (self.adapters.clone(), self.device.clone());
        lora::painting(&adapters, req, &device, factors, || self.draw(req, on_step))
    }

    fn defaults(&self) -> Defaults {
        // Black Forest Labs' own settings for schnell: four steps, no
        // guidance, a megapixel.
        let schnell = Defaults { width: 1024, height: 1024, steps: 4, guidance: 0.0, multiple: 16, takes_guidance: false, takes_negative: false, takes_loras: true, edits: false };
        match self.dit.cfg.guided {
            false => schnell,
            // And diffusers' for dev: 28 steps at guidance 3.5.
            true => Defaults { steps: 28, guidance: 3.5, takes_guidance: true, ..schnell },
        }
    }

    fn summary(&self) -> String {
        let from = self.gguf.as_ref().map(|(name, _)| format!(", the transformer from {name}")).unwrap_or_default();
        format!("{}, {:.1} B parameters{from}", name_of(&self.dit.cfg), self.params as f64 / 1e9)
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        self.bytes
    }

    fn backend(&self) -> String {
        let label = crate::common::label(&self.device, self.dtype, self.quant);
        match &self.gguf {
            Some((_, made)) => format!("{label}, transformer {made}"),
            None => label,
        }
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

    /// The block a LoRA's layer is in, in diffusers' names or Black Forest
    /// Labs', dotted or kohya's: `(single, index)`, or `None` outside them.
    fn block_of(m: &str) -> Option<(bool, usize)> {
        let m = m.replace('_', ".");
        for (mark, single) in [("single.transformer.blocks.", true), ("transformer.blocks.", false), ("double.blocks.", false), ("single.blocks.", true)] {
            if let Some(at) = m.find(mark) {
                let rest = &m[at + mark.len()..];
                return rest.split('.').next().and_then(|n| n.parse().ok()).map(|n| (single, n));
            }
        }
        None
    }

    /// FLUX's first double and single blocks with a LoRA, against diffusers
    /// with PEFT (`scripts/lora-fixtures.py --pipeline flux --blocks 1`):
    /// without it, with it and at half its strength, and the LoRA's own part.
    /// One fixture a LoRA, found by the name in its file's: PEFT's names,
    /// and kohya's in Black Forest Labs' layout, whose fused `qkv` and
    /// `linear1` diffusers splits as [`bfl_loras`] does. CLIP's pairs are
    /// left out, for this builds the transformer alone. On the CPU in f32,
    /// and on Metal at q8, as Kvad runs FLUX:
    ///
    ///     KVAD_LORA_FIXTURES=/tmp/lora-fx cargo test --release -p kvad-gpu flux::tests::a_lora -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_lora_agrees_with_peft() {
        let dir = std::env::var("KVAD_LORA_FIXTURES").expect("KVAD_LORA_FIXTURES, from scripts/lora-fixtures.py");
        let (config, paths) = component("black-forest-labs/FLUX.1-schnell", "transformer", "diffusion_pytorch_model", &Watcher::none()).unwrap();
        let db = |want: &Tensor, got: &Tensor| -> f64 {
            let e = (want - got).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
            10.0 * (want.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64 / e).log10()
        };
        let mut fixtures: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
        fixtures.sort();
        let mut seen = 0;
        for path in fixtures {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let Some(name) = stem.strip_prefix("flux_") else { continue };
            let name = name.replace("--", "/").replace("@@", ":");
            let fx = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
            let here = kvad::lora::local(&name).unwrap_or_else(|| panic!("{name} is not on this machine"));
            let file = lora::File::open(&here.file)
                .unwrap()
                .named(&name)
                .only(|m| !m.starts_with("lora_te") && !m.starts_with("text_encoder.") && block_of(m).is_none_or(|(_, i)| i < 1));
            let sigma = fx["sigma"].to_vec1::<f32>().unwrap()[0] as f64;
            let (plain_ref, adapted_ref, half_ref) = (&fx["plain"], &fx["adapted"], &fx["half"]);
            let part_ref = (adapted_ref - plain_ref).unwrap();
            let mut runs = vec![(Device::Cpu, None, DType::F32, "the CPU, f32")];
            if let Ok(metal) = Device::new_metal(0) {
                runs.push((metal, Some(GgmlDType::Q8_0), DType::BF16, "Metal, q8, bf16 factors"));
            }
            for (dev, quant, factors, what) in runs {
                let vault = Vault::off();
                let cx = Ctx { ld: Loader::new(quant, dev.clone(), &vault).accelerated(), dtype: DType::F32 };
                let mut cfg = Config::from_json(&config).unwrap();
                (cfg.double, cfg.single) = (1, 1);
                let adapters = Adapters::new(&PREFIXES);
                let r = open(&paths, DType::F32).unwrap().with_adapters(adapters.part("transformer"));
                let map = gguf_map(&cfg);
                let dit = Transformer::load(&cx, &r, cfg).unwrap();
                bfl_loras(&adapters, &map);
                let on = |k: &str| fx[k].to_device(&dev).unwrap();
                let run = || dit.forward(&on("x"), &on("txt"), &on("pooled"), sigma, 0.0, 8, 8).unwrap().to_device(&Device::Cpu).unwrap();
                let plain = run();
                let n = adapters.set(&[(&file, 1.0)], &dev, factors).unwrap();
                let adapted = run();
                adapters.set(&[(&file, 0.5)], &dev, factors).unwrap();
                let half = run();
                let part = (&adapted - &plain).unwrap();
                let (p, a, h, d) = (db(plain_ref, &plain), db(adapted_ref, &adapted), db(half_ref, &half), db(&part_ref, &part));
                let size = db(plain_ref, adapted_ref);
                eprintln!("{name}, {what}: plain {p:.1} dB, adapted {a:.1}, half {h:.1}; its own part, {size:.1} dB below the output, {d:.1} dB, on {n} layers");
                let floor = if quant.is_some() { 25.0 } else { 80.0 };
                assert!(a > floor && h > floor, "{name}, {what}");
                assert!(d > p - size - 6.0, "{name}, {what}: the LoRA's part is further from the reference's than the model's own rounding explains");
            }
            seen += 1;
        }
        assert!(seen > 0, "no flux_ fixtures in {dir}");
    }

    /// FLUX's first double block and first single block, with their real
    /// weights and a LoRA on two of their layers: the gradient `backward`
    /// finds is the function's own, through the latents the model is given
    /// and through each LoRA factor.
    ///
    /// In two steps. In f64 on the CPU, [`crate::grad::directional`]
    /// measures the slope with eight digits to agree in, and backward's
    /// agrees: that gradient is whole. Then every other way the block runs
    /// (f32 on the CPU and on Metal, and on Metal with the weights at q8 as
    /// candle's quantised product reads them) must find the same slope as
    /// that one did, from the same seeded tensors along the same direction.
    /// A slope measured in f32 is not asked: it is 0.5–3% of a typical slope
    /// out here, where backward's own is within 1e-5 of the f64 one.
    ///
    /// The LoRA sits on the first block's `to_q` and on the single block's
    /// `proj_mlp`, so its gradient has the whole of both blocks to come back
    /// through: the norms, the rotation, the attention, every frozen
    /// projection after it, and the other LoRA's side path.
    ///
    ///     cargo test --release -p kvad-gpu flux::tests::a_real_block -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_real_block_has_a_whole_gradient() {
        use candle_core::Var;
        let (config, paths) = component("black-forest-labs/FLUX.1-schnell", "transformer", "diffusion_pytorch_model", &Watcher::none()).unwrap();
        // (device, weights, the arithmetic, the step as a share of each
        // tensor's own scale, and how far backward's slope may be from the
        // f64 run's: f32's rounding, and at q8 the weights' own, which is a
        // slightly different function)
        let mut runs = vec![(Device::Cpu, None, DType::F64, DType::F64, 1e-5, 0.0, "the CPU, f64"), (Device::Cpu, None, DType::F32, DType::F32, 1e-2, 1e-3, "the CPU, f32")];
        if let Ok(metal) = Device::new_metal(0) {
            runs.push((metal.clone(), None, DType::F32, DType::F32, 1e-2, 1e-3, "Metal, f32"));
            runs.push((metal.clone(), Some(GgmlDType::Q8_0), DType::F32, DType::F32, 1e-2, 1e-1, "Metal, q8 weights"));
            // Half precision, as the pipeline runs, with the factors in it
            // too and then in f32, as #74 means to train them. Here the
            // measured slope is rounding and nothing else (0, or ±64000),
            // and backward's is 0.1–7% of a typical slope from f64's.
            runs.push((metal.clone(), None, DType::BF16, DType::BF16, 1e-2, 0.2, "Metal, bf16"));
            runs.push((metal, None, DType::BF16, DType::F32, 1e-2, 0.2, "Metal, bf16 with f32 factors"));
        }
        // Backward's slopes in f64, in the order they are taken.
        let mut exact: Vec<f64> = Vec::new();
        for (dev, quant, dtype, factor, step, tolerance, what) in runs {
            let first = exact.is_empty();
            let mut taken = 0;
            let vault = Vault::off();
            let cx = Ctx { ld: Loader::new(quant, dev.clone(), &vault), dtype };
            let mut cfg = Config::from_json(&config).unwrap();
            (cfg.double, cfg.single) = (1, 1);
            let (width, joint, pooled_width, channels) = (cfg.shape.width(), cfg.joint, cfg.pooled, cfg.in_channels);
            let adapters = Adapters::new(&PREFIXES);
            let r = open(&paths, dtype).unwrap().with_adapters(adapters.part("transformer"));
            let dit = Transformer::load(&cx, &r, cfg).unwrap();

            // Seeded, each tensor from its own, so that a run can be had again.
            let seed = std::cell::Cell::new(74u64);
            let randn_in = |shape: &[usize], std: f32, dt: DType| {
                seed.set(seed.get() + 1);
                (noise(seed.get(), shape, &dev, DType::F32).unwrap() * std as f64).unwrap().to_dtype(dt).unwrap()
            };
            let randn = |shape: &[usize], std: f32| randn_in(shape, std, dtype);
            // 8×8 patches and 12 text tokens, at σ = 0.6.
            let (x, txt, pooled) = (randn(&[1, 64, channels], 1.0), randn(&[1, 12, joint], 1.0), randn(&[1, pooled_width], 1.0));
            let weigh = randn(&[1, 64, channels], 1.0);
            let loss = |x: &Tensor| -> candle_core::Result<Tensor> {
                let v = dit.forward(x, &txt, &pooled, 0.6, 0.0, 8, 8).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                (v * &weigh)?.sum_all()
            };

            // Rank 4 on each, neither factor zero, so that both get a
            // gradient (a LoRA starts with `B` at zero, and `A`'s is then
            // zero too).
            const SCALE: f32 = 0.05;
            let layers = [("transformer_blocks.0.attn.to_q", width, width), ("single_transformer_blocks.0.proj_mlp", width, 4 * width)];
            let factors: Vec<(Var, Var)> = layers
                .iter()
                .map(|&(_, inp, out)| (Var::from_tensor(&randn_in(&[inp, 4], SCALE, factor)).unwrap(), Var::from_tensor(&randn_in(&[4, out], SCALE, factor)).unwrap()))
                .collect();
            let place = |i: usize, a: &Tensor, b: &Tensor| adapters.place("transformer", layers[i].0, a, b).unwrap();
            for (i, (a, b)) in factors.iter().enumerate() {
                place(i, a.as_tensor(), b.as_tensor());
            }

            let mut report = |name: &str, s: crate::grad::Slopes| {
                if first {
                    eprintln!("{what}: {name}: backward {:.6}, measured {:.6}, {:.1e} apart (a slope here is typically {:.1})", s.by_backward, s.measured, s.apart(), s.typical);
                    assert!(s.apart() < 1e-6, "{what}: the gradient through {name} is not the function's: {s:?}");
                    exact.push(s.by_backward);
                } else {
                    let off = (s.by_backward - exact[taken]).abs() / s.typical;
                    eprintln!("{what}: {name}: backward {:.6}, {off:.1e} from f64's; measured {:.6}, {:.1e} apart", s.by_backward, s.measured, s.apart());
                    assert!(off < tolerance, "{what}: the gradient through {name} is not f64's: {s:?} against {}", exact[taken]);
                }
                taken += 1;
            };
            report("the latents", crate::grad::directional(&loss, &x, step, 11).unwrap());
            for (i, (a, b)) in factors.iter().enumerate() {
                // The factor under test is placed afresh for every pass, and
                // the one beside it stays.
                let by_a = |t: &Tensor| {
                    place(i, t, b.as_tensor());
                    loss(&x)
                };
                report(&format!("{}'s A", layers[i].0), crate::grad::directional(&by_a, a.as_tensor(), step * SCALE as f64, 12).unwrap());
                place(i, a.as_tensor(), b.as_tensor());
                let by_b = |t: &Tensor| {
                    place(i, a.as_tensor(), t);
                    loss(&x)
                };
                report(&format!("{}'s B", layers[i].0), crate::grad::directional(&by_b, b.as_tensor(), step * SCALE as f64, 13).unwrap());
                place(i, a.as_tensor(), b.as_tensor());
            }
        }
    }

    /// T5's first two layers and its final norm, with their real weights,
    /// as a function of the embeddings they are given and of a LoRA on two
    /// of their layers: the gradient `backward` finds is the function's
    /// own. Two layers and not twenty-four, for the whole of it in f64 is
    /// 38 GB; every layer is the same code, and the position bias, which
    /// the first layer alone owns, is in.
    ///
    /// Certified in f64 on the CPU and held to that after, as
    /// [`a_real_block_has_a_whole_gradient`] is. The q8 weights are read
    /// by candle's quantised product, which has a backward
    /// (`crate::grad::Frozen`); the M5's matrix units, which the pipeline
    /// loads them for, refuse a tensor that is being differentiated.
    ///
    ///     cargo test --release -p kvad-gpu flux::tests::t5_has -- --ignored --nocapture
    #[test]
    #[ignore]
    fn t5_has_a_whole_gradient() {
        crate::cap::at(24.0);
        use candle_core::Var;
        let (mut config, paths) = component("black-forest-labs/FLUX.1-schnell", "text_encoder_2", "model", &Watcher::none()).unwrap();
        config["num_layers"] = json!(2);
        let (width, ff, inner) = (4096, 10240, 4096);
        let mut runs = vec![(Device::Cpu, None, DType::F64, 1e-5, 0.0, "the CPU, f64"), (Device::Cpu, None, DType::F32, 1e-2, 1e-3, "the CPU, f32")];
        if let Ok(metal) = Device::new_metal(0) {
            runs.push((metal.clone(), None, DType::F32, 1e-2, 1e-3, "Metal, f32"));
            runs.push((metal.clone(), Some(GgmlDType::Q8_0), DType::F32, 1e-2, 1e-1, "Metal, q8 weights"));
            runs.push((metal, None, DType::BF16, 1e-2, 0.2, "Metal, bf16"));
        }
        // "a red fox sitting in fresh snow", and the end marker.
        let ids: [u32; 12] = [3, 9, 1131, 3, 20400, 3823, 16, 1434, 4170, 3, 2, 1];
        let layers = [("encoder.block.0.layer.0.SelfAttention.q", width, inner), ("encoder.block.1.layer.1.DenseReluDense.wi_1", width, ff)];
        let mut exact = crate::grad::real::Exact::default();
        // The embeddings the f64 run read, for every run after it: a
        // quantised table's rows are other numbers, and the check is of
        // the layers.
        let mut read: Option<Tensor> = None;
        for (dev, quant, dtype, step, tolerance, what) in runs {
            let mut run = exact.run(what, tolerance);
            let vault = Vault::off();
            let cx = Ctx { ld: Loader::new(quant, dev.clone(), &vault), dtype };
            let adapters = Adapters::new(&PREFIXES);
            let r = open(&paths, dtype).unwrap().with_adapters(adapters.part("te2"));
            let t5 = T5::load(&cx, &r, &config).unwrap();

            let seed = std::cell::Cell::new(74u64);
            let randn = |shape: &[usize], std: f32| {
                seed.set(seed.get() + 1);
                (noise(seed.get(), shape, &dev, DType::F32).unwrap() * std as f64).unwrap().to_dtype(dtype).unwrap()
            };
            let x = match &read {
                Some(x) => x.to_dtype(dtype).unwrap().to_device(&dev).unwrap(),
                None => t5.embed(&ids, &dev, dtype).unwrap(),
            };
            read.get_or_insert_with(|| x.clone());
            let scale = x.to_dtype(DType::F32).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64;
            let weigh = randn(&[1, ids.len(), width], 1.0);
            let loss = |x: &Tensor| -> candle_core::Result<Tensor> {
                let h = t5.read(x).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                (h * &weigh)?.sum_all()
            };

            const SCALE: f32 = 0.05;
            let factors: Vec<(Var, Var)> = layers
                .iter()
                .map(|&(name, inp, out)| {
                    let (a, b) = (Var::from_tensor(&randn(&[inp, 4], SCALE)).unwrap(), Var::from_tensor(&randn(&[4, out], SCALE)).unwrap());
                    adapters.place("te2", name, a.as_tensor(), b.as_tensor()).unwrap();
                    (a, b)
                })
                .collect();
            run.take("the embeddings", crate::grad::directional(&loss, &x, step * scale, 11).unwrap());
            for (&(name, ..), (a, b)) in layers.iter().zip(&factors) {
                run.factors(&adapters, "te2", name, (a, b), &|| loss(&x), step * SCALE as f64);
            }
        }
    }

    /// The positions diffusers' `_prepare_latent_image_ids` gives, and text at
    /// the origin: rows and columns from the top left, not centred.
    /// Dev's config and schnell's differ in one line, and so do the
    /// layers each is read with: the guidance scale's two, under Black
    /// Forest Labs' name for them in a GGUF.
    #[test]
    fn only_a_guided_transformer_has_the_guidance_layers() {
        let config = |guided: bool| {
            json!({
                "patch_size": 1, "num_layers": 1, "num_single_layers": 1, "num_attention_heads": 24, "attention_head_dim": 128,
                "in_channels": 64, "joint_attention_dim": 4096, "pooled_projection_dim": 768, "guidance_embeds": guided
            })
        };
        let (dev, schnell) = (Config::from_json(&config(true)).unwrap(), Config::from_json(&config(false)).unwrap());
        assert!(dev.guided && !schnell.guided);
        assert_eq!((name_of(&dev), name_of(&schnell)), ("FLUX.1-dev", "FLUX.1-schnell"));
        let guidance = |cfg: &Config| -> Vec<(String, String)> {
            gguf_map(cfg).into_iter().filter(|(to, _)| to.contains("guidance")).map(|(to, parts)| (to, parts[0].name.clone())).collect()
        };
        assert!(guidance(&schnell).is_empty());
        assert_eq!(
            guidance(&dev),
            [
                ("time_text_embed.guidance_embedder.linear_1.weight", "guidance_in.in_layer.weight"),
                ("time_text_embed.guidance_embedder.linear_1.bias", "guidance_in.in_layer.bias"),
                ("time_text_embed.guidance_embedder.linear_2.weight", "guidance_in.out_layer.weight"),
                ("time_text_embed.guidance_embedder.linear_2.bias", "guidance_in.out_layer.bias"),
            ]
            .map(|(a, b)| (a.to_string(), b.to_string()))
        );
        // A config that does not say is schnell's.
        let mut bare = config(false);
        bare.as_object_mut().unwrap().remove("guidance_embeds");
        assert!(!Config::from_json(&bare).unwrap().guided);
    }

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
            guided: false,
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
