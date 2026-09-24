//! CLIP's text encoder: the prompt as the denoiser reads it.
//!
//! CLIP was trained to put a caption and its image at the same point in one
//! space. Its text half is a small GPT-2-shaped transformer — pre-norm,
//! learned positions, a causal mask — and the diffusion models that use it
//! throw its training objective away and keep its hidden states: a vector per
//! token that already "means" something visual.
//!
//! SDXL uses two of them, OpenAI's ViT-L (12 layers, 768 wide) and
//! OpenCLIP's bigG (32 layers, 1280 wide), and reads both at the
//! **penultimate** layer. The last layer of a CLIP text model is shaped by the
//! contrastive loss towards the one pooled vector that loss compares, and the
//! layer before it keeps more of the per-token detail a denoiser can use. So
//! for ViT-L the last layer is never run at all, and its weights are skipped
//! by name rather than loaded for nothing.

use super::nn::{Ctx, LayerNorm, Linear};
use crate::common::Reader;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use kvad::serde_json::Value;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// How many tokens a CLIP text model sees: 75 of prompt and the two markers.
pub(crate) const CONTEXT: usize = 77;
pub(crate) const START: u32 = 49406;
pub(crate) const END: u32 = 49407;

/// The shape of one CLIP text model, from its `config.json`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClipConfig {
    pub(crate) width: usize,
    pub(crate) heads: usize,
    pub(crate) layers: usize,
    pub(crate) inter: usize,
    pub(crate) act: Act,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Act {
    /// `x·σ(1.702x)`, OpenAI's cheap stand-in for GELU from before GELU was
    /// cheap. ViT-L was trained with it, so ViT-L must be run with it.
    QuickGelu,
    Gelu,
}

impl ClipConfig {
    pub(crate) fn from_json(v: &Value) -> Res<Self> {
        let n = |k: &str| -> Res<usize> {
            v.get(k).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| format!("CLIP config has no `{k}`").into())
        };
        let act = match v.get("hidden_act").and_then(Value::as_str) {
            Some("quick_gelu") => Act::QuickGelu,
            Some("gelu") => Act::Gelu,
            other => return Err(format!("CLIP activation {other:?} is not one this encoder knows").into()),
        };
        Ok(ClipConfig {
            width: n("hidden_size")?,
            heads: n("num_attention_heads")?,
            layers: n("num_hidden_layers")?,
            inter: n("intermediate_size")?,
            act,
        })
    }
}

struct Layer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// One CLIP text model, loaded as far as the pipeline needs it.
pub(crate) struct Clip {
    cfg: ClipConfig,
    tokens: Tensor,
    positions: Tensor,
    layers: Vec<Layer>,
    /// Only when the pooled vector is wanted: bigG's final norm and its
    /// projection into the joint space.
    pooled: Option<(LayerNorm, Linear)>,
}

impl Clip {
    /// Load enough of the model to give its penultimate hidden state, and,
    /// if `pooled`, the whole of it and the projection as well.
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, cfg: ClipConfig, pooled: bool) -> Res<Self> {
        let tm = r.pp("text_model");
        let emb = tm.pp("embeddings");
        let w = cfg.width;
        let tokens = cx.get(&emb, (49408, w), "token_embedding.weight")?;
        let positions = cx.get(&emb, (CONTEXT, w), "position_embedding.weight")?;

        // Without the pooled vector the last layer's output is never read.
        let run = if pooled { cfg.layers } else { cfg.layers - 1 };
        let mut layers = Vec::with_capacity(run);
        for i in 0..run {
            let l = tm.pp(format!("encoder.layers.{i}"));
            let a = l.pp("self_attn");
            layers.push(Layer {
                ln1: LayerNorm::load(cx, &l, "layer_norm1", w, 1e-5)?,
                q: Linear::load(cx, &a, "q_proj", w, w, true)?,
                k: Linear::load(cx, &a, "k_proj", w, w, true)?,
                v: Linear::load(cx, &a, "v_proj", w, w, true)?,
                out: Linear::load(cx, &a, "out_proj", w, w, true)?,
                ln2: LayerNorm::load(cx, &l, "layer_norm2", w, 1e-5)?,
                fc1: Linear::load(cx, &l, "mlp.fc1", w, cfg.inter, true)?,
                fc2: Linear::load(cx, &l, "mlp.fc2", cfg.inter, w, true)?,
            });
        }
        let pooled = match pooled {
            true => Some((
                LayerNorm::load(cx, &tm, "final_layer_norm", w, 1e-5)?,
                Linear::load(cx, r, "text_projection", w, w, false)?,
            )),
            false => {
                tm.skip_under(&format!("encoder.layers.{}.", cfg.layers - 1));
                tm.skip_under("final_layer_norm");
                None
            }
        };
        Ok(Clip { cfg, tokens, positions, layers, pooled })
    }

    pub(crate) fn width(&self) -> usize {
        self.cfg.width
    }

    /// The penultimate hidden state, `[1, 77, width]`, and — when this model
    /// was loaded for it — the pooled vector, `[1, width]`, taken at `end`,
    /// the position of the end-of-text marker.
    pub(crate) fn encode(&self, ids: &[u32], end: usize) -> Res<(Tensor, Option<Tensor>)> {
        let dev = self.tokens.device();
        let ids_t = Tensor::new(ids, dev)?;
        let mut x = self.tokens.index_select(&ids_t, 0)?.broadcast_add(&self.positions)?.unsqueeze(0)?;
        let mask = causal_mask(ids.len(), dev)?;

        let penult_at = self.cfg.layers - 1;
        let mut penultimate = None;
        for (i, l) in self.layers.iter().enumerate() {
            if i == penult_at {
                penultimate = Some(x.clone());
            }
            x = self.layer(l, &x, &mask)?;
        }
        let penultimate = match penultimate {
            Some(p) => p,
            None => x.clone(),
        };
        let pooled = match &self.pooled {
            Some((ln, proj)) => {
                let last = ln.forward(&x)?;
                Some(proj.forward(&last.narrow(1, end, 1)?.squeeze(1)?)?)
            }
            None => None,
        };
        Ok((penultimate, pooled))
    }

    fn layer(&self, l: &Layer, x: &Tensor, mask: &Tensor) -> Res<Tensor> {
        let (b, n, w) = x.dims3()?;
        let heads = self.cfg.heads;
        let d = w / heads;
        let h = l.ln1.forward(x)?;
        let split = |t: Tensor| -> candle_core::Result<Tensor> { t.reshape((b, n, heads, d))?.transpose(1, 2)?.contiguous() };
        let q = split(l.q.forward(&h)?)?;
        let k = split(l.k.forward(&h)?)?;
        let v = split(l.v.forward(&h)?)?;
        // Seventy-seven tokens: the score matrix is tiny and written out, and
        // the causal mask goes on explicitly. The fused kernel's causal path
        // wants a multiple of 32 queries (see `model.rs`), and 77 is not one.
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)?.to_dtype(DType::F32)? * (1.0 / (d as f64).sqrt()))?;
        let att = ops::softmax_last_dim(&att.broadcast_add(mask)?)?.to_dtype(v.dtype())?;
        let a = att.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, n, w))?;
        let x = (x + l.out.forward(&a)?)?;

        let h = l.fc1.forward(&l.ln2.forward(&x)?)?;
        let h = match self.cfg.act {
            Act::QuickGelu => (&h * ops::sigmoid(&(&h * 1.702)?)?)?,
            Act::Gelu => h.gelu_erf()?,
        };
        Ok((x + l.fc2.forward(&h)?)?)
    }
}

/// `-inf` above the diagonal, in f32 so it adds to f32 scores.
fn causal_mask(n: usize, dev: &Device) -> Res<Tensor> {
    let data: Vec<f32> = (0..n * n).map(|i| if i % n > i / n { f32::NEG_INFINITY } else { 0.0 }).collect();
    Ok(Tensor::from_vec(data, (1, 1, n, n), dev)?)
}

/// A prompt as CLIP token ids: the start marker, at most 75 tokens of text,
/// the end marker, then `pad` to 77. Returns the ids and where the end
/// marker is, which is where the pooled vector is read.
///
/// The configs say `eos_token_id: 2`, which is wrong (a conversion leftover
/// that every pipeline works around); the end marker is 49407, the largest id
/// in the vocabulary, and `transformers` finds it with an `argmax`.
pub(crate) fn tokenize(tok: &tokenizers::Tokenizer, text: &str, pad: u32) -> Res<(Vec<u32>, usize)> {
    let enc = tok.encode(text, false).map_err(|e| e.to_string())?;
    let body: Vec<u32> = enc.get_ids().iter().copied().filter(|&t| t != START && t != END).take(CONTEXT - 2).collect();
    let mut ids = Vec::with_capacity(CONTEXT);
    ids.push(START);
    ids.extend(&body);
    let end = ids.len();
    ids.push(END);
    ids.resize(CONTEXT, pad);
    Ok((ids, end))
}
