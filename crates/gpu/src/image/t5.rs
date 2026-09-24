//! T5's encoder: the prompt as FLUX reads it.
//!
//! T5 is an encoder-decoder translation model from 2019, and FLUX keeps only
//! the encoder — 24 bidirectional layers, 4096 wide, 4.8 billion parameters of
//! reading comprehension with no idea that it is being asked about pictures.
//! It is the reason FLUX can spell: CLIP's 77 tokens see a prompt as a bag of
//! visual concepts, and T5's 256 see it as a sentence.
//!
//! Three things make it unlike every other transformer in this repository:
//!
//! - **No positions are added to the tokens.** Instead every layer's attention
//!   scores get a learned bias that depends only on how far apart the two
//!   tokens are, bucketed so that near distances are exact and far ones share
//!   a bucket. It is computed once, from a table in the first layer, and every
//!   layer reuses it.
//! - **No scaling of the scores by `1/√d`.** T5 folded it into the weights'
//!   initialisation instead, so the scores are used as they come.
//! - **A gated GELU MLP**, `wo(gelu(wi_0·x) · wi_1·x)`, and RMSNorm without a
//!   bias, which T5 called a layer norm.

use super::nn::{Ctx, Linear};
use crate::common::{Reader, Stored};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use kvad::serde_json::Value;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Layer {
    ln1: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: Tensor,
    wi0: Linear,
    wi1: Linear,
    wo: Linear,
}

pub(crate) struct T5 {
    embed: Embed,
    /// `[buckets, heads]`: the bias each head adds for a distance bucket.
    bias: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    heads: usize,
    d_kv: usize,
    buckets: usize,
    max_distance: usize,
    eps: f32,
}

enum Embed {
    Dense(Tensor),
    Quant(Arc<QTensor>),
}

impl T5 {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, c: &Value) -> Res<Self> {
        let n = |k: &str| -> Res<usize> {
            c.get(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("T5 config has no `{k}`").into())
        };
        let (d, ff, heads, d_kv, layers, vocab) =
            (n("d_model")?, n("d_ff")?, n("num_heads")?, n("d_kv")?, n("num_layers")?, n("vocab_size")?);
        if c.get("feed_forward_proj").and_then(Value::as_str) != Some("gated-gelu") {
            return Err("this T5 encoder implements the gated-GELU feed-forward (T5 v1.1) only".into());
        }
        let inner = heads * d_kv;
        let embed = match cx.ld.quant {
            Some(_) => Embed::Quant(Arc::new(cx.ld.quantized(r, "shared.weight", vocab, d, Stored::OutIn)?)),
            None => Embed::Dense(cx.get(r, (vocab, d), "shared.weight")?),
        };
        let e = r.pp("encoder");
        let buckets = n("relative_attention_num_buckets")?;
        let bias = cx.get(&e.pp("block.0.layer.0.SelfAttention"), (buckets, heads), "relative_attention_bias.weight")?;
        let mut out = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = e.pp(format!("block.{i}"));
            let a = b.pp("layer.0.SelfAttention");
            let f = b.pp("layer.1.DenseReluDense");
            out.push(Layer {
                ln1: cx.get(&b, d, "layer.0.layer_norm.weight")?,
                q: Linear::load(cx, &a, "q", d, inner, false)?,
                k: Linear::load(cx, &a, "k", d, inner, false)?,
                v: Linear::load(cx, &a, "v", d, inner, false)?,
                o: Linear::load(cx, &a, "o", inner, d, false)?,
                ln2: cx.get(&b, d, "layer.1.layer_norm.weight")?,
                wi0: Linear::load(cx, &f, "wi_0", d, ff, false)?,
                wi1: Linear::load(cx, &f, "wi_1", d, ff, false)?,
                wo: Linear::load(cx, &f, "wo", ff, d, false)?,
            });
        }
        Ok(T5 {
            embed,
            bias,
            layers: out,
            norm: cx.get(&e, d, "final_layer_norm.weight")?,
            heads,
            d_kv,
            buckets,
            max_distance: n("relative_attention_max_distance")?,
            eps: c.get("layer_norm_epsilon").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
        })
    }

    /// The last layer's output after the final norm, `[1, tokens, d_model]`.
    pub(crate) fn forward(&self, ids: &[u32], device: &Device, dtype: DType) -> Res<Tensor> {
        let l = ids.len();
        let ids_t = Tensor::new(ids, device)?;
        let mut x = match &self.embed {
            Embed::Dense(t) => t.index_select(&ids_t, 0)?,
            Embed::Quant(q) => q.embedding(&ids_t)?,
        }
        .to_dtype(dtype)?
        .unsqueeze(0)?;

        // The position bias, `[1, heads, l, l]`, once for every layer.
        let index: Vec<u32> = (0..l)
            .flat_map(|q| (0..l).map(move |k| bucket(k as i64 - q as i64, self.buckets, self.max_distance) as u32))
            .collect();
        let bias = self
            .bias
            .index_select(&Tensor::new(index, device)?, 0)?
            .reshape((l, l, self.heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        let inner = self.heads * self.d_kv;
        for layer in &self.layers {
            let h = ops::rms_norm(&x, &layer.ln1, self.eps)?;
            let split = |t: Tensor| -> candle_core::Result<Tensor> {
                t.reshape((1, l, self.heads, self.d_kv))?.transpose(1, 2)?.contiguous()
            };
            let (q, k, v) = (split(layer.q.forward(&h)?)?, split(layer.k.forward(&h)?)?, split(layer.v.forward(&h)?)?);
            let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.to_dtype(DType::F32)?.broadcast_add(&bias)?;
            let att = ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
            let a = att.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((1, l, inner))?;
            x = (x + layer.o.forward(&a)?)?;

            let h = ops::rms_norm(&x, &layer.ln2, self.eps)?;
            let g = (layer.wi0.forward(&h)?.gelu()? * layer.wi1.forward(&h)?)?;
            x = (x + layer.wo.forward(&g)?)?;
        }
        Ok(ops::rms_norm(&x, &self.norm, self.eps)?)
    }
}

/// Which bucket a distance falls in, for an encoder (attention both ways).
///
/// Half the buckets are for keys after the query and half for keys before it.
/// Within each half, the first half of the buckets are exact distances 0…7,
/// and the rest are spaced logarithmically out to `max_distance`, beyond which
/// every distance shares the last bucket. Transcribed from `transformers`'
/// `T5Attention._relative_position_bucket`, and checked against values it
/// gives in the test below.
fn bucket(relative: i64, buckets: usize, max_distance: usize) -> usize {
    let half = buckets / 2;
    let side = if relative > 0 { half } else { 0 };
    let n = relative.unsigned_abs() as usize;
    let exact = half / 2;
    if n < exact {
        return side + n;
    }
    let far = exact as f64 + ((n as f64 / exact as f64).ln() / (max_distance as f64 / exact as f64).ln() * (half - exact) as f64);
    side + (far as usize).min(half - 1)
}

#[cfg(test)]
mod tests {
    use super::bucket;

    /// Values of `T5Attention._relative_position_bucket(rel, True, 32, 128)`,
    /// worked through by hand from its formula: exact to 7, logarithmic to
    /// 128, one bucket beyond, and the far side offset by 16.
    #[test]
    fn distances_fall_in_t5_s_buckets() {
        let b = |r: i64| bucket(r, 32, 128);
        assert_eq!([b(0), b(-1), b(-7)], [0, 1, 7]);
        assert_eq!([b(1), b(7)], [17, 23]);
        // 8 + ln(n/8)/ln(16)·8, floored.
        assert_eq!(b(-8), 8);
        assert_eq!(b(-16), 10);
        assert_eq!(b(-32), 12);
        assert_eq!(b(-127), 15);
        assert_eq!([b(-128), b(-1000)], [15, 15]);
        assert_eq!([b(16), b(500)], [26, 31]);
    }
}
