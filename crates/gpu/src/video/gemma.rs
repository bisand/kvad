//! Gemma 4's text tower, as LTX-2.5 uses it: to encode, never to generate.
//!
//! The checkpoint is `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16`, a
//! `gemma4_unified` model LTX tuned and ships with its own projections (see
//! [`super::ltx_text`]). Its vision and audio towers are never used for text
//! to video and are not read. `docs/video-plan.md` has the architecture; what
//! makes it unlike the Llama-shaped models in `model.rs`:
//!
//! - **Norms multiply by `w`, not `1 + w`**, and every layer is normalised
//!   four times: before and after attention, before and after the MLP.
//! - **Each layer ends by scaling the whole residual stream** by a learned
//!   scalar, from 0.0045 to 0.92. It is one number per layer and easy to miss.
//! - **Two kinds of attention.** Five layers in six are "sliding": 16 query
//!   heads and 8 key-value heads of 256, rotated in full. Every sixth is
//!   global: 16 query heads of 512 against a *single* key-value head, whose
//!   values are its keys taken before their norm and rotation, and whose
//!   rotation turns only the first 64 of its 256 frequencies.
//! - **Queries and keys are RMS-normed per head, and values too** (with no
//!   weight). The attention scale is 1: the norms' weights carry it.
//!
//! What LTX wants is not the last layer but **all 49 hidden states**: the
//! scaled embeddings, every layer's output but the last, and the last one
//! through the final norm.

use super::ltx_nn::{rms, RmsNorm, Rope};
use crate::common::{Loader, Reader};
use crate::image::nn::{Ctx, Linear};
use crate::image::{finish, open};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use kvad::serde_json::{json, Value};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const EPS: f64 = 1e-6;

struct Layer {
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
    q: Linear,
    k: Linear,
    /// Absent on global layers, whose values are their keys.
    v: Option<Linear>,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    scalar: f64,
    global: bool,
    head_dim: usize,
    kv_heads: usize,
}

pub struct Gemma {
    /// `[vocab, width]`, left on the CPU: a prompt needs a few hundred of its
    /// 262 144 rows, and copying 2 GB to the GPU for them would be waste.
    embed: Tensor,
    /// `√width` rounded to the compute dtype, as the reference multiplies by
    /// it: 62.0 in bf16, 61.97 in f32.
    scale: f64,
    layers: Vec<Layer>,
    norm: RmsNorm,
    heads: usize,
    sliding_freq: Vec<f32>,
    global_freq: Vec<f32>,
    device: Device,
    dtype: DType,
}

impl Gemma {
    /// Load the tower from `r` (the text encoder file's reader) with the
    /// `text_config` in `config`; only its first `layers` layers when that is
    /// given, which is what a test that has to fit in f32 wants.
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, config: &Value, layers: Option<usize>) -> Res<Self> {
        let c = &config["text_config"];
        for (key, want) in [
            ("model_type", Value::from("gemma4_unified_text")),
            ("attention_k_eq_v", Value::Bool(true)),
            ("hidden_activation", Value::from("gelu_pytorch_tanh")),
            ("attention_bias", Value::Bool(false)),
            ("num_kv_shared_layers", Value::from(0)),
            ("enable_moe_block", Value::Bool(false)),
            ("hidden_size_per_layer_input", Value::from(0)),
            ("use_double_wide_mlp", Value::Bool(false)),
        ] {
            if c[key] != want {
                return Err(format!("Gemma: `{key}` is {}, and this tower is written for {want}", c[key]).into());
            }
        }
        let num = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("Gemma: no `{k}`"));
        let (width, heads, ffn, vocab) = (num("hidden_size")?, num("num_attention_heads")?, num("intermediate_size")?, num("vocab_size")?);
        let (hd, ghd, kvh, gkvh) = (num("head_dim")?, num("global_head_dim")?, num("num_key_value_heads")?, num("num_global_key_value_heads")?);
        let total = num("num_hidden_layers")?;
        let types: Vec<&str> = c["layer_types"].as_array().ok_or("Gemma: no layer_types")?.iter().filter_map(Value::as_str).collect();
        let n = layers.unwrap_or(total).min(total);

        let rope = &c["rope_parameters"];
        let theta = |t: &str| rope[t]["rope_theta"].as_f64().ok_or_else(|| format!("Gemma: no rope_theta for {t}"));
        if rope["sliding_attention"]["rope_type"] != "default" || rope["full_attention"]["rope_type"] != "proportional" {
            return Err(format!("Gemma: RoPE types {rope} are not the ones implemented").into());
        }
        // Sliding: every pair of a 256-wide head turns, θ = 10⁴.
        let base = theta("sliding_attention")? as f32;
        let sliding_freq = (0..hd / 2).map(|i| 1.0 / base.powf((2 * i) as f32 / hd as f32)).collect();
        // Global, "proportional": a quarter of the 512-wide head's pairs turn,
        // at frequencies spaced as if all of them did; the rest stand still.
        let base = theta("full_attention")? as f32;
        let partial = rope["full_attention"]["partial_rotary_factor"].as_f64().unwrap_or(1.0);
        let turning = (partial * ghd as f64 / 2.0) as usize;
        let global_freq = (0..ghd / 2).map(|i| if i < turning { 1.0 / base.powf((2 * i) as f32 / ghd as f32) } else { 0.0 }).collect();

        let m = r.pp("model");
        let mut list = Vec::with_capacity(n);
        for i in 0..n {
            let l = m.pp(format!("layers.{i}"));
            let global = types.get(i) == Some(&"full_attention");
            let (d, kv) = if global { (ghd, gkvh) } else { (hd, kvh) };
            let a = l.pp("self_attn");
            list.push(Layer {
                input_ln: RmsNorm::load(cx, &l, "input_layernorm.weight", width, EPS)?,
                post_attn_ln: RmsNorm::load(cx, &l, "post_attention_layernorm.weight", width, EPS)?,
                pre_ff_ln: RmsNorm::load(cx, &l, "pre_feedforward_layernorm.weight", width, EPS)?,
                post_ff_ln: RmsNorm::load(cx, &l, "post_feedforward_layernorm.weight", width, EPS)?,
                q: Linear::load(cx, &a, "q_proj", width, heads * d, false)?,
                k: Linear::load(cx, &a, "k_proj", width, kv * d, false)?,
                v: match global {
                    true => None,
                    false => Some(Linear::load(cx, &a, "v_proj", width, kv * d, false)?),
                },
                o: Linear::load(cx, &a, "o_proj", heads * d, width, false)?,
                q_norm: RmsNorm::load(cx, &a, "q_norm.weight", d, EPS)?,
                k_norm: RmsNorm::load(cx, &a, "k_norm.weight", d, EPS)?,
                gate: Linear::load(cx, &l.pp("mlp"), "gate_proj", width, ffn, false)?,
                up: Linear::load(cx, &l.pp("mlp"), "up_proj", width, ffn, false)?,
                down: Linear::load(cx, &l.pp("mlp"), "down_proj", ffn, width, false)?,
                scalar: l.get(1, "layer_scalar")?.to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as f64,
                global,
                head_dim: d,
                kv_heads: kv,
            });
        }
        for i in n..total {
            m.skip_under(&format!("layers.{i}."));
        }
        let dtype = cx.dtype;
        let scale = Tensor::new(&[(width as f32).sqrt()], &Device::Cpu)?.to_dtype(dtype)?.to_dtype(DType::F32)?.to_vec1::<f32>()?[0] as f64;
        Ok(Gemma {
            embed: m.get((vocab, width), "embed_tokens.weight")?,
            scale,
            layers: list,
            norm: RmsNorm::load(cx, &m, "norm.weight", width, EPS)?,
            heads,
            sliding_freq,
            global_freq,
            device: cx.device().clone(),
            dtype,
        })
    }

    /// Load just the tower from the text encoder file at `path`, marking
    /// everything else in it as deliberately unread: for tests and fixtures.
    /// `quant` quantises its matrices, as [`super::ltx_text::TextEncoder`]
    /// does, and caches them where that would.
    pub fn load_file(path: &Path, device: &Device, dtype: DType, layers: Option<usize>, quant: Option<GgmlDType>, progress: &mut dyn FnMut(&str)) -> Res<Self> {
        let config = super::metadata(path, "gemma_config")?;
        let paths = [path.to_path_buf()];
        let mut vault = Vault::open_as(&format!("{}/text_encoder", super::LTX_REPO), &paths, json!({ "component": "gemma", "layers": layers }), quant, progress);
        let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
        let r = open(&paths, DType::BF16)?;
        for p in ["vision_model.", "multi_modal_projector.", "audio_projector.", "hf_asset__", "tokenizer_json", "text_embedding_projection."] {
            r.skip_under(p);
        }
        let g = Gemma::load(&cx, &r, &config, layers)?;
        finish("Gemma 4", &paths, &r)?;
        drop(cx);
        vault.finish(progress);
        Ok(g)
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Every hidden state for the tokens `ids`, `[n, width]` each: the
    /// embeddings, each layer's output but the last, and the last one through
    /// the final norm — one more than there are layers.
    ///
    /// `first` is the first token's position. LTX pads every prompt on the
    /// left to 1024 tokens, so a prompt of `n` starts at `1024 − n`; running
    /// only the real tokens at the positions they would have had gives the
    /// same numbers, because attention is causal and the padding masked out.
    pub fn hidden_states(&self, ids: &[u32], first: usize) -> candle_core::Result<Vec<Tensor>> {
        let n = ids.len();
        let idx = Tensor::from_vec(ids.to_vec(), n, &Device::Cpu)?;
        let mut x = (self.embed.index_select(&idx, 0)?.to_device(&self.device)?.to_dtype(self.dtype)? * self.scale)?;

        let positions: Vec<f32> = (0..n).map(|i| (first + i) as f32).collect();
        let rope = |freq: &[f32]| Rope::standard(&positions, freq, &self.device, self.dtype).map_err(|e| candle_core::Error::Msg(e.to_string()));
        let (sliding, global) = (rope(&self.sliding_freq)?, rope(&self.global_freq)?);
        // Causal: row i sees columns 0..=i. The sliding window is 1024 and a
        // prompt never longer, so the sliding layers are causal too.
        let mask: Vec<f32> = (0..n * n).map(|k| if k % n > k / n { f32::NEG_INFINITY } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask, (n, n), &self.device)?;

        let mut states = Vec::with_capacity(self.layers.len() + 1);
        for layer in &self.layers {
            states.push(x.clone());
            let rope = if layer.global { &global } else { &sliding };
            x = self.layer(layer, &x, rope, &mask)?;
        }
        states.push(self.norm.forward(&x)?);
        Ok(states)
    }

    /// A projection, in the compute dtype whatever the weights' kind: a Q8_0
    /// matrix answers in f32 even when it was asked in bf16.
    fn lin(&self, l: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
        l.forward(x)?.to_dtype(self.dtype)
    }

    fn layer(&self, l: &Layer, x: &Tensor, rope: &Rope, mask: &Tensor) -> candle_core::Result<Tensor> {
        let n = x.dim(0)?;
        let (h, d, kv) = (self.heads, l.head_dim, l.kv_heads);
        let a = l.input_ln.forward(x)?;
        let q = rope.rotate(&l.q_norm.forward(&self.lin(&l.q, &a)?.reshape((n, h, d))?)?)?;
        let raw_k = self.lin(&l.k, &a)?.reshape((n, kv, d))?;
        let v = match &l.v {
            Some(v) => self.lin(v, &a)?.reshape((n, kv, d))?,
            None => raw_k.clone(),
        };
        let k = rope.rotate(&l.k_norm.forward(&raw_k)?)?;
        let v = rms(&v, EPS)?;

        // [n, heads, d] → [heads, n, d], each key-value head shared by
        // heads / kv query heads in a row.
        let q = q.transpose(0, 1)?.contiguous()?;
        let share = |t: Tensor| -> candle_core::Result<Tensor> {
            let t = t.transpose(0, 1)?.contiguous()?;
            t.unsqueeze(1)?.expand((kv, h / kv, n, d))?.contiguous()?.reshape((h, n, d))
        };
        let (k, v) = (share(k)?, share(v)?);
        // Scale 1: the per-head norms' weights set the temperature.
        let scores = q.to_dtype(DType::F32)?.matmul(&k.to_dtype(DType::F32)?.t()?.contiguous()?)?.broadcast_add(mask)?;
        let att = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        let o = att.matmul(&v)?.transpose(0, 1)?.contiguous()?.reshape((n, h * d))?;
        let x = (x + l.post_attn_ln.forward(&self.lin(&l.o, &o)?)?)?;

        let f = l.pre_ff_ln.forward(&x)?;
        let f = self.lin(&l.down, &(self.lin(&l.gate, &f)?.gelu()? * self.lin(&l.up, &f)?)?)?;
        (x + l.post_ff_ln.forward(&f)?)? * l.scalar
    }
}
