//! The MMDiT block, which Qwen-Image and FLUX share.
//!
//! An MMDiT ("multimodal diffusion transformer") keeps the image and the text
//! as two separate streams of tokens with separate weights, and lets them meet
//! in one place: a joint attention over both, text first. Everything else in a
//! block — the LayerNorms, the time modulation, the MLP — happens to each
//! stream on its own.
//!
//! Qwen-Image's sixty blocks and FLUX's first nineteen are the same block. The
//! checkpoints spell it differently (`img_mod.1` against `norm1.linear`,
//! `img_mlp` against `ff`), so [`Names`] is how a model says where its weights
//! are, and the arithmetic is written once.
//!
//! FLUX then adds a second kind, the [`Single`] block: one stream of text and
//! image tokens concatenated, with the attention and the MLP run side by side
//! from the same normalised input rather than one after the other. It is what
//! FLUX spends two thirds of its blocks on.

use super::nn::{layer_norm_plain, Ctx, Linear};
use crate::common::Reader;
use candle_core::Tensor;
use candle_nn::ops;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Where one stream of a double block keeps its weights, relative to the
/// block. The attention projections are named alike in every checkpoint seen
/// so far; the modulation and MLP are not.
pub(crate) struct Names {
    pub(crate) modulate: &'static str,
    pub(crate) mlp_in: &'static str,
    pub(crate) mlp_out: &'static str,
}

/// One stream's half of a double block: its modulation, its attention
/// projections and norms, and its MLP.
pub(crate) struct Stream {
    modulate: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    mlp_in: Linear,
    mlp_out: Linear,
}

/// The image stream and the text stream.
pub(crate) struct Double {
    img: Stream,
    txt: Stream,
}

/// The shape every block shares.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
}

impl Shape {
    pub(crate) fn width(&self) -> usize {
        self.heads * self.head_dim
    }
}

impl Double {
    /// `img` and `txt` say where each stream's modulation and MLP live; the
    /// attention's projections are `to_q` … `to_out.0` for the image and
    /// `add_q_proj` … `to_add_out` for the text in every checkpoint here.
    pub(crate) fn load(cx: &Ctx<'_>, b: &Reader<'_>, s: Shape, img: &Names, txt: &Names) -> Res<Self> {
        let w = s.width();
        let a = b.pp("attn");
        let stream = |n: &Names, q: &str, k: &str, v: &str, out: &str, nq: &str, nk: &str| -> Res<Stream> {
            Ok(Stream {
                modulate: Linear::load(cx, b, n.modulate, w, 6 * w, true)?,
                q: Linear::load(cx, &a, q, w, w, true)?,
                k: Linear::load(cx, &a, k, w, w, true)?,
                v: Linear::load(cx, &a, v, w, w, true)?,
                out: Linear::load(cx, &a, out, w, w, true)?,
                norm_q: cx.get(&a, s.head_dim, &format!("{nq}.weight"))?,
                norm_k: cx.get(&a, s.head_dim, &format!("{nk}.weight"))?,
                mlp_in: Linear::load(cx, b, n.mlp_in, w, 4 * w, true)?,
                mlp_out: Linear::load(cx, b, n.mlp_out, 4 * w, w, true)?,
            })
        };
        Ok(Double {
            img: stream(img, "to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k")?,
            txt: stream(txt, "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "norm_added_q", "norm_added_k")?,
        })
    }

    /// One block. `temb` has had its SiLU already. `rope_img` and `rope_txt`
    /// are each stream's `(cos, sin)`, `[tokens, head_dim / 2]`, for rotating
    /// adjacent pairs.
    pub(crate) fn forward(
        &self,
        s: Shape,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        rope_img: (&Tensor, &Tensor),
        rope_txt: (&Tensor, &Tensor),
    ) -> Res<(Tensor, Tensor)> {
        // Six vectors per stream from the time embedding: shift, scale and
        // gate for the attention half, and the same for the MLP half.
        let (mi, mt) = (six(&self.img.modulate, temb, s.width())?, six(&self.txt.modulate, temb, s.width())?);

        // Each stream projects its own queries, keys and values, normalises
        // and rotates them; then one attention runs over the two
        // concatenated, text first.
        let [qi, ki, vi] = qkv(s, &self.img, &modulate(img, &mi[0], &mi[1])?, rope_img)?;
        let [qt, kt, vt] = qkv(s, &self.txt, &modulate(txt, &mt[0], &mt[1])?, rope_txt)?;
        let a = attend(s, &Tensor::cat(&[&qt, &qi], 2)?, &Tensor::cat(&[&kt, &ki], 2)?, &Tensor::cat(&[&vt, &vi], 2)?)?;
        let (n_txt, n_img) = (qt.dim(2)?, qi.dim(2)?);
        let (at, ai) = (a.narrow(1, 0, n_txt)?, a.narrow(1, n_txt, n_img)?);

        let img = (img + self.img.out.forward(&ai)?.broadcast_mul(&mi[2])?)?;
        let txt = (txt + self.txt.out.forward(&at)?.broadcast_mul(&mt[2])?)?;

        let mlp = |st: &Stream, x: &Tensor, m: &[Tensor]| -> candle_core::Result<Tensor> {
            let h = st.mlp_in.forward(&modulate(x, &m[3], &m[4])?)?.gelu()?;
            x + st.mlp_out.forward(&h)?.broadcast_mul(&m[5])?
        };
        Ok((mlp(&self.img, &img, &mi)?, mlp(&self.txt, &txt, &mt)?))
    }
}

/// FLUX's single-stream block: text and image as one sequence.
///
/// One modulation (shift, scale, gate), one LayerNorm, and from that one
/// normalised input both an attention and an MLP, whose outputs are
/// concatenated and projected back together by `proj_out`. Running the two
/// in parallel rather than in series is the whole difference from a double
/// block, and it is cheaper: one projection where there were two.
pub(crate) struct Single {
    modulate: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    mlp: Linear,
    out: Linear,
}

impl Single {
    pub(crate) fn load(cx: &Ctx<'_>, b: &Reader<'_>, s: Shape) -> Res<Self> {
        let w = s.width();
        let a = b.pp("attn");
        Ok(Single {
            modulate: Linear::load(cx, b, "norm.linear", w, 3 * w, true)?,
            q: Linear::load(cx, &a, "to_q", w, w, true)?,
            k: Linear::load(cx, &a, "to_k", w, w, true)?,
            v: Linear::load(cx, &a, "to_v", w, w, true)?,
            norm_q: cx.get(&a, s.head_dim, "norm_q.weight")?,
            norm_k: cx.get(&a, s.head_dim, "norm_k.weight")?,
            mlp: Linear::load(cx, b, "proj_mlp", w, 4 * w, true)?,
            out: Linear::load(cx, b, "proj_out", 5 * w, w, true)?,
        })
    }

    /// `x` is text then image, and `rope` covers the same tokens in the same
    /// order.
    pub(crate) fn forward(&self, s: Shape, x: &Tensor, temb: &Tensor, rope: (&Tensor, &Tensor)) -> Res<Tensor> {
        let w = s.width();
        let m = self.modulate.forward(temb)?;
        let part = |i: usize| m.narrow(1, i * w, w)?.unsqueeze(1);
        let (shift, scale, gate) = (part(0)?, part(1)?, part(2)?);
        let h = modulate(x, &shift, &scale)?;

        let heads_of = |t: Tensor| -> candle_core::Result<Tensor> { t.reshape((1, x.dim(1)?, s.heads, s.head_dim)) };
        let q = rotate(ops::rms_norm(&heads_of(self.q.forward(&h)?)?.contiguous()?, &self.norm_q, 1e-6)?, rope)?;
        let k = rotate(ops::rms_norm(&heads_of(self.k.forward(&h)?)?.contiguous()?, &self.norm_k, 1e-6)?, rope)?;
        let v = heads_of(self.v.forward(&h)?)?.transpose(1, 2)?.contiguous()?;
        let a = attend(s, &q, &k, &v)?;
        let mlp = self.mlp.forward(&h)?.gelu()?;
        Ok((x + self.out.forward(&Tensor::cat(&[&a, &mlp], 2)?)?.broadcast_mul(&gate)?)?)
    }
}

/// The adaptive norm both models end with: a LayerNorm scaled and shifted by
/// the time embedding — and note the order, *scale* then shift, the opposite
/// of the blocks'.
pub(crate) fn norm_out(linear: &Linear, x: &Tensor, temb: &Tensor, width: usize) -> Res<Tensor> {
    let m = linear.forward(temb)?;
    let (scale, shift) = (m.narrow(1, 0, width)?.unsqueeze(1)?, m.narrow(1, width, width)?.unsqueeze(1)?);
    Ok(layer_norm_plain(x, 1e-6)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?)
}

/// `n` vectors of `width` from one modulation projection, as `[1, 1, width]`.
fn six(linear: &Linear, temb: &Tensor, width: usize) -> candle_core::Result<Vec<Tensor>> {
    let m = linear.forward(temb)?;
    (0..6).map(|i| m.narrow(1, i * width, width)?.unsqueeze(1)).collect()
}

/// LayerNorm without weights, then the time embedding's scale and shift.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    layer_norm_plain(x, 1e-6)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// One stream's queries, keys and values as `[1, heads, tokens, head_dim]`,
/// queries and keys normalised per head and rotated.
fn qkv(s: Shape, st: &Stream, x: &Tensor, rope: (&Tensor, &Tensor)) -> candle_core::Result<[Tensor; 3]> {
    let n = x.dim(1)?;
    let heads_of = |t: Tensor| t.reshape((1, n, s.heads, s.head_dim));
    let q = ops::rms_norm(&heads_of(st.q.forward(x)?)?.contiguous()?, &st.norm_q, 1e-6)?;
    let k = ops::rms_norm(&heads_of(st.k.forward(x)?)?.contiguous()?, &st.norm_k, 1e-6)?;
    let v = heads_of(st.v.forward(x)?)?;
    Ok([rotate(q, rope)?, rotate(k, rope)?, v.transpose(1, 2)?.contiguous()?])
}

/// `[1, tokens, heads, d]` to `[1, heads, tokens, d]`, rotating adjacent
/// pairs — `(x₀, x₁)`, `(x₂, x₃)` … — which is both models' convention and
/// not the halves convention the Llama family uses.
fn rotate(t: Tensor, (cos, sin): (&Tensor, &Tensor)) -> candle_core::Result<Tensor> {
    candle_nn::rotary_emb::rope_i(&t.transpose(1, 2)?.contiguous()?, cos, sin)
}

/// Unmasked attention over `[1, heads, tokens, d]`, back to `[1, tokens,
/// heads·d]`.
fn attend(s: Shape, q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let scale = 1.0 / (s.head_dim as f64).sqrt();
    let a = match q.device().is_metal() {
        true => ops::sdpa(q, k, v, None, false, scale as f32, 1.0)?,
        false => super::nn::written_out(q, k, v, scale)?,
    };
    let n = q.dim(2)?;
    a.transpose(1, 2)?.contiguous()?.reshape((1, n, s.width()))
}
