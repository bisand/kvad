//! LTX-2.5's text path: a prompt to the two contexts the DiT attends to.
//!
//! ```text
//! prompt ─ tokens ─ Gemma 4 ─ 49 hidden states
//!        ─ each RMS-normed per token, interleaved into one 188 160-wide row
//!        ─ two projections: to 4096 (video) and to 2048 (audio)
//!        ─ two 8-block connectors, which fill the prompt out to 1024 rows
//!          with learned registers ─ video [1024, 4096], audio [1024, 2048]
//! ```
//!
//! The tokenizer and the projections are in the text encoder's file; the
//! connectors are in the DiT's. `docs/video-plan.md` has each step, and
//! where the reference was read for it.

use super::gemma::Gemma;
use super::ltx_nn::{gelu, rms, GatedAttention, Rope};
use super::metadata;
use crate::common::{Loader, Reader};
use crate::image::nn::{Ctx, Linear};
use crate::image::{finish, open};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use kvad::serde_json::{json, Value};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The text encoder's file in [`super::LTX_REPO`].
pub const TEXT_FILE: &str = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors";
/// The distilled DiT's file, which holds the connectors.
pub const DIT_FILE: &str = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors";

/// Every prompt is this many tokens by the time the DiT sees it, padded if
/// shorter and cut if longer.
pub const LENGTH: usize = 1024;

/// Gemma's beginning-of-sequence and padding tokens.
const BOS: u32 = 2;
const PAD: u32 = 0;
/// Gemma's image, audio and video placeholders. A prompt that spells one out
/// gets padding in its place, as the reference does, rather than a
/// placeholder with nothing behind it.
const PLACEHOLDERS: [u32; 3] = [258880, 258881, 258884];

/// Gemma's tokenizer, as the LTX text encoder file carries it: a tensor of
/// bytes holding a `tokenizer.json`.
pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    /// The tokenizer stored in the text encoder file at `path`.
    pub fn load(path: &Path) -> Res<Self> {
        // Read raw: it is bytes, not numbers, and the reader would convert it.
        let raw = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        let t = tokenizers::Tokenizer::from_bytes(raw.get("tokenizer_json")?.data()).map_err(|e| format!("the embedded tokenizer: {e}"))?;
        Ok(Tokenizer(t))
    }

    /// The prompt as the tokens Gemma reads: trimmed, tokenised with nothing
    /// added, `<bos>` in front, at most [`LENGTH`] in all.
    pub fn tokens(&self, prompt: &str) -> Res<Vec<u32>> {
        let enc = self.0.encode(prompt.trim(), false).map_err(|e| format!("tokenising: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().iter().take(LENGTH).map(|&t| if PLACEHOLDERS.contains(&t) { PAD } else { t }).collect();
        if ids.first() != Some(&BOS) {
            ids.insert(0, BOS);
        }
        ids.truncate(LENGTH);
        Ok(ids)
    }
}

/// What the DiT attends to: one context for the video stream and one for the
/// audio stream, [`LENGTH`] rows each.
pub struct Contexts {
    pub video: Tensor,
    pub audio: Tensor,
}

/// A connector: eight blocks of self-attention and a feed-forward over the
/// whole [`LENGTH`] rows, with the rows the prompt did not fill taken from
/// learned registers.
struct Connector {
    blocks: Vec<(GatedAttention, Linear, Linear)>,
    /// `[registers, width]`.
    registers: Tensor,
    rope: Rope,
}

impl Connector {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: &Value, audio: bool) -> Res<Self> {
        let key = |k: &str| match audio {
            true => cfg.get(format!("audio_{k}")).filter(|v| !v.is_null()).unwrap_or(&cfg[k]),
            false => &cfg[k],
        };
        let num = |k: &str| key(k).as_u64().map(|v| v as usize).ok_or_else(|| format!("connector: no `{k}`"));
        let (heads, hd, layers, regs) = (num("connector_num_attention_heads")?, num("connector_attention_head_dim")?, num("connector_num_layers")?, num("connector_num_learnable_registers")?);
        let width = heads * hd;
        let gated = cfg["connector_apply_gated_attention"].as_bool().unwrap_or(false);
        if cfg["rope_type"] != "split" {
            return Err(format!("connector: rope_type {} is not implemented", cfg["rope_type"]).into());
        }
        let max_pos = cfg["connector_positional_embedding_max_pos"][0].as_f64().ok_or("connector: no max_pos")? as f32;
        let theta = cfg["positional_embedding_theta"].as_f64().unwrap_or(10000.0);

        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = r.pp(format!("transformer_1d_blocks.{i}"));
            blocks.push((
                GatedAttention::load(cx, &b.pp("attn1"), (width, width), heads, hd, gated)?,
                Linear::load(cx, &b.pp("ff.net.0"), "proj", width, 4 * width, true)?,
                Linear::load(cx, &b.pp("ff.net"), "2", 4 * width, width, true)?,
            ));
        }
        let positions = vec![(0..LENGTH).map(|p| p as f32).collect::<Vec<_>>()];
        Ok(Connector {
            blocks,
            registers: cx.get(r, (regs, width), "learnable_registers")?,
            rope: Rope::split(&positions, &[max_pos], width, heads, theta, cx.device(), cx.dtype)?,
        })
    }

    /// The prompt's rows `[n, width]` to `[LENGTH, width]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let n = x.dim(0)?;
        let regs = self.registers.dim(0)?;
        // Row p, for every p the prompt leaves empty, is register p % 128:
        // the registers repeated to the full length, then its tail.
        let filled = self.registers.repeat((LENGTH / regs, 1))?.narrow(0, n, LENGTH - n)?;
        let mut x = Tensor::cat(&[x, &filled.to_dtype(x.dtype())?], 0)?;
        for (attn, ff_in, ff_out) in &self.blocks {
            x = (&x + attn.forward(&rms(&x, 1e-6)?, None, Some(&self.rope), Some(&self.rope))?)?;
            x = (&x + ff_out.forward(&gelu(&ff_in.forward(&rms(&x, 1e-6)?)?)?)?)?;
        }
        rms(&x, 1e-6)
    }
}

pub struct TextEncoder {
    tokenizer: Tokenizer,
    pub gemma: Gemma,
    video_proj: Linear,
    audio_proj: Linear,
    video: Connector,
    audio: Connector,
    params: usize,
}

impl TextEncoder {
    /// Load Gemma, the projections and the tokenizer from the text encoder
    /// file `text`, and the two connectors from the DiT file `dit`, computing
    /// in `dtype` on `device`.
    ///
    /// `quant` quantises Gemma's matrices and the projections (12.4 of the
    /// path's 15.1 B parameters), and caches them so that the next load maps
    /// them instead. The connectors stay in `dtype`.
    pub fn load(text: &Path, dit: &Path, device: &Device, dtype: DType, quant: Option<GgmlDType>, progress: &mut dyn FnMut(&str)) -> Res<Self> {
        Self::load_with(text, dit, device, dtype, None, quant, progress)
    }

    /// [`TextEncoder::load`] with only Gemma's first `layers` layers, for a
    /// check that feeds [`TextEncoder::project`] hidden states from
    /// elsewhere and cannot afford the whole tower (48 GB in f32).
    pub fn load_with(text: &Path, dit: &Path, device: &Device, dtype: DType, layers: Option<usize>, quant: Option<GgmlDType>, progress: &mut dyn FnMut(&str)) -> Res<Self> {
        let gemma_cfg = metadata(text, "gemma_config")?;
        let dit_cfg = metadata(dit, "config")?;
        // The DiT names the text encoder it was trained against.
        let wanted = metadata(dit, "gemma_source_checkpoint")?;
        if wanted["gemma_version"] != gemma_cfg["gemma_version"] {
            return Err(format!("the DiT wants Gemma {}, and the text encoder is {}", wanted["gemma_version"], gemma_cfg["gemma_version"]).into());
        }
        let paths = [text.to_path_buf()];
        let mut vault = Vault::open_as(&format!("{}/text_encoder", super::LTX_REPO), &paths, json!({ "component": "text_encoder", "layers": layers }), quant, progress);
        let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
        let r = open(&paths, DType::BF16)?;
        for p in ["vision_model.", "multi_modal_projector.", "audio_projector.", "hf_asset__"] {
            r.skip_under(p);
        }
        r.record("tokenizer_json");
        let tokenizer = Tokenizer::load(text)?;

        let gemma = Gemma::load(&cx, &r, &gemma_cfg, layers)?;
        let width = gemma_cfg["text_config"]["hidden_size"].as_u64().ok_or("Gemma: no hidden_size")? as usize;
        // Every hidden state goes into the projection: the embeddings and one
        // per layer, whatever part of the tower was loaded.
        let states = gemma_cfg["text_config"]["num_hidden_layers"].as_u64().ok_or("Gemma: no num_hidden_layers")? as usize + 1;
        let flat = width * states;
        let t = &dit_cfg["transformer"];
        let video_width = t["cross_attention_dim"].as_u64().ok_or("DiT: no cross_attention_dim")? as usize;
        let audio_width = t["audio_cross_attention_dim"].as_u64().ok_or("DiT: no audio_cross_attention_dim")? as usize;
        let p = r.pp("text_embedding_projection");
        let video_proj = Linear::load(&cx, &p, "video_aggregate_embed", flat, video_width, true)?;
        let audio_proj = Linear::load(&cx, &p, "audio_aggregate_embed", flat, audio_width, true)?;
        let mut params = finish("LTX text encoder", &paths, &r)?;
        drop(cx);
        vault.finish(progress);

        // From the DiT's file, only the connectors: the rest is the DiT's.
        let off = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &off), dtype };
        let dit_paths = [dit.to_path_buf()];
        let d = open(&dit_paths, DType::BF16)?;
        let m = d.pp("model.diffusion_model");
        let video = Connector::load(&cx, &m.pp("video_embeddings_connector"), t, false)?;
        let audio = Connector::load(&cx, &m.pp("audio_embeddings_connector"), t, true)?;
        let left: Vec<String> = crate::common::unread(&dit_paths, &d.seen(), &d.skipped())?.into_iter().filter(|n| n.contains("_embeddings_connector.")).collect();
        if !left.is_empty() {
            return Err(format!("LTX connectors: {} tensor(s) this loader never reads:\n  {}", left.len(), left.join("\n  ")).into());
        }
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(dit)? };
        let seen = d.seen();
        params += st.tensors().iter().filter(|(n, _)| seen.contains(n)).map(|(_, v)| v.shape().iter().product::<usize>()).sum::<usize>();

        Ok(TextEncoder { tokenizer, gemma, video_proj, audio_proj, video, audio, params })
    }

    pub fn params(&self) -> usize {
        self.params
    }

    /// See [`Tokenizer::tokens`].
    pub fn tokens(&self, prompt: &str) -> Res<Vec<u32>> {
        self.tokenizer.tokens(prompt)
    }

    /// A prompt to the DiT's two contexts.
    pub fn encode(&self, prompt: &str) -> Res<Contexts> {
        let ids = self.tokens(prompt)?;
        let states = self.gemma.hidden_states(&ids, LENGTH - ids.len())?;
        Ok(self.project(&states)?)
    }

    /// Gemma's hidden states for a prompt, `[n, width]` each, to the two
    /// contexts: the part after Gemma, callable on its own so that it can be
    /// checked against the reference's own hidden states.
    pub fn project(&self, states: &[Tensor]) -> candle_core::Result<Contexts> {
        // Each state RMS-normed over its width, token by token, then the 49
        // interleaved: column d·49 + l is dimension d of state l.
        let normed = states.iter().map(|s| rms(&s.to_dtype(DType::F32)?, 1e-6)).collect::<candle_core::Result<Vec<_>>>()?;
        let (n, width) = normed[0].dims2()?;
        let flat = Tensor::stack(&normed, 2)?.reshape((n, width * normed.len()))?;
        let dtype = states[0].dtype();
        // Rescaled so the projection sees the norm it was trained with.
        let project = |proj: &Linear, out: usize| -> candle_core::Result<Tensor> {
            proj.forward(&(&flat * (out as f64 / width as f64).sqrt())?.to_dtype(dtype)?)?.to_dtype(dtype)
        };
        let video = project(&self.video_proj, self.video.registers.dim(1)?)?;
        let audio = project(&self.audio_proj, self.audio.registers.dim(1)?)?;
        Ok(Contexts { video: self.video.forward(&video)?, audio: self.audio.forward(&audio)? })
    }
}
