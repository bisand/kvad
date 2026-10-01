//! A diffusion transformer: the GPT, taught to draw.
//!
//! # What changes, and what does not
//!
//! The models kvad runs for images (FLUX, Qwen-Image) and video (LTX) are
//! transformers, and not in a loose sense: take the blocks in [`block`], and
//! four changes turn a GPT into one of them.
//!
//! 1. **Patches, not tokens.** A 28×28 digit cut into 4×4 squares is a
//!    sequence of 49 of them, and one `Linear` turns each square's 16 pixels
//!    into a `d_model` vector. A token embedding looked a vector up; this one
//!    computes it, which is the only difference between the two.
//! 2. **No causal mask.** Nothing is predicted left to right, so every patch
//!    may read every other one ([`SelfAttention::bidirectional`]).
//! 3. **Conditioning.** The model is told *how noisy* its input is (a number
//!    `t` between 0 and 1) and *what to draw* (a digit, standing in for the
//!    prompt a text encoder would supply). How that reaches the blocks is the
//!    one new idea in this file; see below.
//! 4. **The output is an image, not a distribution over tokens.** One
//!    `Linear` per patch turns each vector back into 16 numbers, and they are
//!    put back where the patch came from. What those numbers *mean*, and the
//!    loss that trains them, is [`flow`]'s business.
//!
//! # The one idea in this file: adaLN-Zero
//!
//! A GPT's `LayerNorm` ends with a learned scale and shift, `gamma * x̂ + beta`,
//! the same for every input. The DiT paper made them depend on the
//! conditioning instead: a `Linear` reads the conditioning vector `c` and
//! produces a scale and a shift for each norm, so the same weights normalise
//! a nearly clean image one way and pure noise another. It also produces a
//! *gate* for each branch of the block:
//!
//! ```text
//! shift, scale, gate   (×2, one set per branch)  =  Linear(silu(c))
//!
//! x = x + gate₁ * attention( norm(x) * (1 + scale₁) + shift₁ )
//! x = x + gate₂ * mlp      ( norm(x) * (1 + scale₂) + shift₂ )
//! ```
//!
//! The "Zero" is the initialisation. The modulation `Linear` starts at zero,
//! so every gate is 0 and every block starts as the identity. A 28-block
//! model begins life as a single `Linear` from patches to output, and the
//! blocks are switched on one gate at a time as training finds them useful.
//! The paper measured this against the obvious alternatives (feeding `c` in
//! as an extra token, or through cross-attention) and it won, for less
//! compute.
//!
//! Its derivatives need no new calculus. A scale and shift shared by every
//! row is a bias with a multiplier, and a bias shared by every row collects
//! the gradient of every row: `dshift = Σ dy`, `dscale = Σ dy * x̂`. The gate is
//! the same shape of thing again. See [`modulate`] and [`gate`].
//!
//! # The layout is DiT's
//!
//! The pieces are laid out as in Facebook's DiT (`facebook/DiT-XL-2-256`,
//! diffusers' `DiTTransformer2DModel`): the same sinusoidal timestep and
//! 2-D position embeddings, the same chunk order in the modulation, the same
//! patch order. What differs is the objective, which is flow matching rather
//! than DDPM — that makes this SiT, the paper that tried DiT's layout with
//! flow matching and found it better — and the output, which is one number
//! per pixel rather than DiT's two (DiT also learns a variance, which flow
//! matching has no use for).
//!
//! [`block`]: crate::block
//! [`flow`]: crate::flow

use crate::attention::SelfAttention;
use crate::checkpoint::{invalid, read_safetensors, write_safetensors, Tensor};
use crate::embedding::Embedding;
use crate::json::{object, Json};
use crate::matrix::Matrix;
use crate::nn::{prefixed, Gelu, Layer, Linear, Param, Silu};
use crate::norm::LayerNorm;
use crate::rng::Rng;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitConfig {
    /// Side of the square image, in pixels.
    pub image: usize,
    pub channels: usize,
    /// Side of a patch, in pixels. `image` has to be a multiple of it.
    pub patch: usize,
    /// How many labels there are. One more embedding than this is kept: the
    /// label "none", which classifier-free guidance needs (see `flow`).
    pub classes: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
}

impl DitConfig {
    /// Patches along one side.
    pub fn grid(&self) -> usize {
        self.image / self.patch
    }

    /// The sequence length: one token per patch.
    pub fn tokens(&self) -> usize {
        self.grid() * self.grid()
    }

    /// Numbers in one patch.
    pub fn patch_len(&self) -> usize {
        self.patch * self.patch * self.channels
    }

    /// Numbers in one image.
    pub fn pixels(&self) -> usize {
        self.channels * self.image * self.image
    }

    /// The label that means "no label".
    pub fn unconditional(&self) -> usize {
        self.classes
    }
}

/// Width of the sinusoidal timestep features, before the MLP widens or
/// narrows them to `d_model`. DiT's number.
pub const FREQUENCIES: usize = 256;

/// `t` runs from 0 to 1 here, and the sinusoids below were designed for the
/// 0..1000 of a DDPM schedule: their fastest wave has a period of 2π, so on
/// 0..1 the slow ones barely move and the fast ones hardly complete a turn.
/// Multiplying by 1000 puts `t` back on the scale the frequencies expect. FLUX
/// does the same, and so does diffusers' `FlowMatchEulerDiscreteScheduler`,
/// whose `num_train_timesteps` of 1000 is this number written down.
pub const TIMESTEP_SCALE: f32 = 1000.0;

/// A number as a vector of waves, so that a network can read it.
///
/// A single input `t` is a poor thing to hand a `Linear`: all it can do is
/// multiply it. Handed `cos(t * f)` and `sin(t * f)` at 128 frequencies from
/// fast to slow, it can tell nearby values apart (the fast waves differ) and
/// see how far apart distant ones are (the slow ones do). It is the
/// Transformer's position encoding, applied to a noise level instead.
///
/// The frequencies run from 1 down to 1/10000 in `half` steps — and exactly
/// down to it, because the exponent is divided by `half - 1`. Facebook's DiT
/// divides by `half` and stops one step short; diffusers' DiT divides by
/// `half - 1` (`downscale_freq_shift=1`), and its layout is the one this
/// model is saved in, so its arithmetic is the one used. The fastest wave is
/// the same in both and the slowest differs by 7%, which a model trained
/// with one notices when run with the other.
pub fn timestep_features(t: f32) -> Vec<f32> {
    let half = FREQUENCIES / 2;
    let t = t * TIMESTEP_SCALE;
    let mut out = vec![0.0; FREQUENCIES];
    for k in 0..half {
        let freq = (-(10_000f32).ln() * k as f32 / (half - 1) as f32).exp();
        out[k] = (t * freq).cos();
        out[half + k] = (t * freq).sin();
    }
    out
}

/// Where each patch is, as a fixed pattern of waves: `[tokens, d_model]`.
///
/// The GPT learned its position vectors. DiT computes them, half the width
/// for the column and half for the row, with the same sinusoids as the
/// timestep. They have no parameters, so there is nothing to train and nothing
/// to save. (Which half is which follows DiT's code, where a `meshgrid`
/// argument order puts the column first.)
pub fn position_table(d_model: usize, grid: usize) -> Matrix {
    assert_eq!(d_model % 4, 0, "d_model {d_model} does not split into four sinusoid bands");
    let quarter = d_model / 4;
    let wave = |pos: usize, out: &mut [f32]| {
        for k in 0..quarter {
            let freq = 1.0 / (10_000f32).powf(k as f32 / quarter as f32);
            out[k] = (pos as f32 * freq).sin();
            out[quarter + k] = (pos as f32 * freq).cos();
        }
    };
    let mut table = Matrix::zeros(grid * grid, d_model);
    for row in 0..grid {
        for col in 0..grid {
            let (first, second) = table.row_mut(row * grid + col).split_at_mut(d_model / 2);
            wave(col, first);
            wave(row, second);
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Patches
// ---------------------------------------------------------------------------

/// Cut an image, `[channels, image, image]` flattened, into `[tokens, patch_len]`.
///
/// Within a patch the order is channel, then row, then column: the order of a
/// `Conv2d(channels, d_model, kernel = patch, stride = patch)` weight, which
/// is how DiT writes its patch embedding. (A convolution whose stride equals
/// its kernel never overlaps itself, so it *is* a `Linear` on each patch.)
pub fn patchify(config: &DitConfig, image: &[f32]) -> Matrix {
    let mut out = Matrix::zeros(config.tokens(), config.patch_len());
    each_pixel(config, |token, pixel, _, at| out.data[token * config.patch_len() + at] = image[pixel]);
    out
}

/// The model's output, `[tokens, patch_len]`, put back together as an image.
///
/// The order within an output patch is row, column, *then* channel, which is
/// not the order `patchify` uses. That is DiT's unpatchify, and a checkpoint
/// trained with one order and read with the other draws scrambled colour. For
/// one channel the two orders are the same, which is exactly why it has to be
/// written down: nothing here would notice.
pub fn unpatchify(config: &DitConfig, patches: &Matrix) -> Vec<f32> {
    let mut image = vec![0.0; config.pixels()];
    each_pixel(config, |token, pixel, at, _| image[pixel] = patches.data[token * config.patch_len() + at]);
    image
}

/// The inverse of [`unpatchify`], which is also its backward pass: the
/// gradient of an image arrives here, and leaves as the gradient of the
/// patches that were put together to make it.
fn unpatchify_backward(config: &DitConfig, dimage: &[f32]) -> Matrix {
    let mut out = Matrix::zeros(config.tokens(), config.patch_len());
    each_pixel(config, |token, pixel, at, _| out.data[token * config.patch_len() + at] = dimage[pixel]);
    out
}

/// Call `f(token, pixel, output_at, input_at)` for every pixel: which patch
/// it lands in, where it is in the image, and where it sits within its patch
/// in the output order and in the input order.
fn each_pixel(config: &DitConfig, mut f: impl FnMut(usize, usize, usize, usize)) {
    let (p, n, c_all) = (config.patch, config.image, config.channels);
    for c in 0..c_all {
        for y in 0..n {
            for x in 0..n {
                let token = (y / p) * config.grid() + x / p;
                let (dy, dx) = (y % p, x % p);
                let pixel = (c * n + y) * n + x;
                f(token, pixel, (dy * p + dx) * c_all + c, (c * p + dy) * p + dx);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Modulation
// ---------------------------------------------------------------------------

/// `x * (1 + scale) + shift`, with one `scale` and one `shift` for every row.
pub fn modulate(x: &Matrix, shift: &[f32], scale: &[f32]) -> Matrix {
    let mut out = x.clone();
    for r in 0..out.rows {
        for (j, v) in out.row_mut(r).iter_mut().enumerate() {
            *v = *v * (1.0 + scale[j]) + shift[j];
        }
    }
    out
}

/// The backward pass of [`modulate`]: `(dx, dshift, dscale)`.
///
/// `shift` was added to every row, so it is blamed for all of them: a sum
/// over rows, like a `Linear`'s bias. `scale` multiplied `x`, so it collects
/// `dy * x`, summed the same way.
fn modulate_backward(dy: &Matrix, x: &Matrix, scale: &[f32]) -> (Matrix, Vec<f32>, Vec<f32>) {
    let mut dx = Matrix::zeros(dy.rows, dy.cols);
    let (mut dshift, mut dscale) = (vec![0.0; dy.cols], vec![0.0; dy.cols]);
    for r in 0..dy.rows {
        let (g, xr) = (dy.row(r), x.row(r));
        for (j, d) in dx.row_mut(r).iter_mut().enumerate() {
            *d = g[j] * (1.0 + scale[j]);
            dshift[j] += g[j];
            dscale[j] += g[j] * xr[j];
        }
    }
    (dx, dshift, dscale)
}

/// `x + gate * branch`, one gate per column.
pub fn gate(x: &Matrix, gate: &[f32], branch: &Matrix) -> Matrix {
    let mut out = x.clone();
    for r in 0..out.rows {
        for (j, v) in out.row_mut(r).iter_mut().enumerate() {
            *v += gate[j] * branch.get(r, j);
        }
    }
    out
}

/// The backward pass of [`gate`], less the `dx = dy` that the residual
/// connection hands down unchanged: `(dbranch, dgate)`.
fn gate_backward(dy: &Matrix, gate: &[f32], branch: &Matrix) -> (Matrix, Vec<f32>) {
    let mut dbranch = Matrix::zeros(dy.rows, dy.cols);
    let mut dgate = vec![0.0; dy.cols];
    for r in 0..dy.rows {
        let (g, b) = (dy.row(r), branch.row(r));
        for (j, d) in dbranch.row_mut(r).iter_mut().enumerate() {
            *d = g[j] * gate[j];
            dgate[j] += g[j] * b[j];
        }
    }
    (dbranch, dgate)
}

// ---------------------------------------------------------------------------
// The block
// ---------------------------------------------------------------------------

/// The norm before attention, and the one before the output: diffusers
/// writes 1e-6 into its code for both.
pub const NORM_EPS_ATTENTION: f32 = 1e-6;
/// The norm before the MLP, which diffusers takes from `norm_eps` in the
/// config, where DiT's is 1e-5.
///
/// Matched because they are the layout's, not because they are large: on the
/// digits model, running the attention norm at 1e-5 instead of 1e-6 moved
/// the output by 7e-6, against outputs of about 6, and the comparison with
/// diffusers (`agrees_with_diffusers`) cannot tell. A row whose variance is
/// near the eps would feel it; these never are.
pub const NORM_EPS: f32 = 1e-5;

/// One DiT block: a transformer [`Block`](crate::block::Block) whose norms and
/// residual branches are steered by the conditioning.
///
/// Not a [`Layer`]: a layer has one input and this has two, the sequence and
/// the conditioning. So `backward` returns two gradients as well.
pub struct DitBlock {
    /// `silu(c)` to the six vectors: shift, scale and gate for attention, then
    /// the same three for the MLP. DiT's order.
    modulation: Linear,
    norm1: LayerNorm,
    attn: SelfAttention,
    norm2: LayerNorm,
    mlp: Vec<Box<dyn Layer>>,
    // What the forward pass leaves for the backward one.
    m: Vec<f32>,
    n1: Matrix,
    a: Matrix,
    n2: Matrix,
    f: Matrix,
}

impl DitBlock {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        DitBlock {
            modulation: Linear::new(d_model, 6 * d_model, rng),
            norm1: LayerNorm::plain(d_model, NORM_EPS_ATTENTION),
            attn: SelfAttention::bidirectional(d_model, n_heads, rng),
            norm2: LayerNorm::plain(d_model, NORM_EPS),
            mlp: vec![
                Box::new(Linear::new(d_model, 4 * d_model, rng)),
                Box::new(Gelu::default()),
                Box::new(Linear::new(4 * d_model, d_model, rng)),
            ],
            m: Vec::new(),
            n1: Matrix::zeros(0, 0),
            a: Matrix::zeros(0, 0),
            n2: Matrix::zeros(0, 0),
            f: Matrix::zeros(0, 0),
        }
    }

    /// The `k`-th of the six vectors the modulation produced.
    fn chunk(&self, k: usize) -> &[f32] {
        let d = self.m.len() / 6;
        &self.m[k * d..(k + 1) * d]
    }

    /// `x` is `[tokens, d_model]`; `c` is `silu` of the conditioning, `[1, d_model]`.
    pub fn forward(&mut self, x: &Matrix, c: &Matrix) -> Matrix {
        self.m = self.modulation.forward(c).data;

        self.n1 = self.norm1.forward(x);
        let h = modulate(&self.n1, self.chunk(0), self.chunk(1));
        self.a = self.attn.forward(&h);
        let x = gate(x, self.chunk(2), &self.a);

        self.n2 = self.norm2.forward(&x);
        let mut h = modulate(&self.n2, self.chunk(3), self.chunk(4));
        for layer in self.mlp.iter_mut() {
            h = layer.forward(&h);
        }
        self.f = h;
        gate(&x, self.chunk(5), &self.f)
    }

    /// Returns `(dx, dc)`: the gradient for the sequence below, and this
    /// block's share of the gradient for the conditioning.
    pub fn backward(&mut self, dy: &Matrix) -> (Matrix, Matrix) {
        // Forward, read bottom to top. Each branch: the gate, the branch
        // itself, the modulation, the norm; and the residual's dy on top.
        let (df, dgate2) = gate_backward(dy, self.chunk(5), &self.f);
        let mut dh = df;
        for layer in self.mlp.iter_mut().rev() {
            dh = layer.backward(&dh);
        }
        let (dn2, dshift2, dscale2) = modulate_backward(&dh, &self.n2, self.chunk(4));
        let mut dx = dy.clone();
        dx.add_in_place(&self.norm2.backward(&dn2));

        let (da, dgate1) = gate_backward(&dx, self.chunk(2), &self.a);
        let dh = self.attn.backward(&da);
        let (dn1, dshift1, dscale1) = modulate_backward(&dh, &self.n1, self.chunk(1));
        dx.add_in_place(&self.norm1.backward(&dn1));

        let dm = [dshift1, dscale1, dgate1, dshift2, dscale2, dgate2].concat();
        let dc = self.modulation.backward(&Matrix::from_vec(1, dm.len(), dm));
        (dx, dc)
    }

    pub fn zero_grad(&mut self) {
        self.modulation.zero_grad();
        self.attn.zero_grad();
        self.mlp.iter_mut().for_each(|l| l.zero_grad());
    }

    pub fn params(&mut self) -> Vec<Param<'_>> {
        let mut all = prefixed("modulation", self.modulation.params());
        all.extend(prefixed("attn", self.attn.params()));
        for (i, layer) in self.mlp.iter_mut().enumerate() {
            all.extend(prefixed(&format!("mlp.{i}"), layer.params()));
        }
        all
    }
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

pub struct Dit {
    config: DitConfig,
    /// Patch to vector: `[patch_len, d_model]`.
    patches: Linear,
    /// Fixed, not learned. See [`position_table`].
    positions: Matrix,
    /// Timestep features to a vector: Linear, SiLU, Linear.
    time: Vec<Box<dyn Layer>>,
    /// One row per label, and one more for "none".
    labels: Embedding,
    /// Every modulation reads `silu(c)`, never `c` itself. DiT applies the
    /// SiLU inside each block; it is the same number every time, so it is
    /// computed once here and its gradients are gathered once.
    cond_act: Silu,
    blocks: Vec<DitBlock>,
    /// The last layer is modulated too, with a shift and a scale but no gate.
    final_norm: LayerNorm,
    final_modulation: Linear,
    /// Vector back to patch: `[d_model, patch_len]`.
    out: Linear,
    final_m: Vec<f32>,
    final_n: Matrix,
}

impl Dit {
    pub fn new(config: DitConfig, rng: &mut Rng) -> Self {
        let DitConfig { d_model, n_heads, n_layers, classes, .. } = config;
        assert_eq!(config.image % config.patch, 0, "a {}-pixel image does not cut into {}-pixel patches", config.image, config.patch);
        let mut model = Dit {
            config,
            patches: Linear::new(config.patch_len(), d_model, rng),
            positions: position_table(d_model, config.grid()),
            time: vec![
                Box::new(Linear::new(FREQUENCIES, d_model, rng)),
                Box::new(Silu::default()),
                Box::new(Linear::new(d_model, d_model, rng)),
            ],
            labels: Embedding::new(classes + 1, d_model, rng),
            cond_act: Silu::default(),
            blocks: (0..n_layers).map(|_| DitBlock::new(d_model, n_heads, rng)).collect(),
            final_norm: LayerNorm::plain(d_model, NORM_EPS_ATTENTION),
            final_modulation: Linear::new(d_model, 2 * d_model, rng),
            out: Linear::new(d_model, config.patch_len(), rng),
            final_m: Vec::new(),
            final_n: Matrix::zeros(0, 0),
        };
        model.init(rng);
        model
    }

    /// DiT's initialisation, over the top of what the layers chose.
    ///
    /// Xavier-uniform for every `Linear`, N(0, 0.02) for the timestep MLP and
    /// the label table, and **zero** for everything that modulates and for the
    /// output layer. The zeros are the "Zero" in adaLN-Zero: every block
    /// starts as the identity, and the model's first prediction is exactly 0
    /// everywhere.
    fn init(&mut self, rng: &mut Rng) {
        let fan = |name: &str, c: &DitConfig| -> (usize, usize) {
            let d = c.d_model;
            match name {
                "patches.weight" => (c.patch_len(), d),
                n if n.ends_with("mlp.0.weight") => (d, 4 * d),
                n if n.ends_with("mlp.2.weight") => (4 * d, d),
                _ => (d, d),
            }
        };
        let config = self.config;
        for p in self.params() {
            let name = p.name.as_str();
            if name.contains("modulation") || name.starts_with("out.") {
                p.value.fill(0.0);
            } else if name.starts_with("time.") && name.ends_with("weight") || name == "labels.table" {
                p.value.iter_mut().for_each(|v| *v = 0.02 * rng.normal());
            } else if name.ends_with("weight") {
                let (fan_in, fan_out) = fan(name, &config);
                let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
                p.value.iter_mut().for_each(|v| *v = bound * (2.0 * rng.uniform() - 1.0));
            } else {
                p.value.fill(0.0);
            }
        }
    }

    /// What the model says about a noisy image `x` at noise level `t`, asked
    /// to draw `label`. One number per pixel, in the image's own layout.
    pub fn forward(&mut self, x: &[f32], t: f32, label: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.config.pixels(), "an image of {} numbers, not {}", x.len(), self.config.pixels());
        assert!(label <= self.config.classes, "label {label} of {} classes", self.config.classes);

        let mut h = self.patches.forward(&patchify(&self.config, x));
        h.add_in_place(&self.positions);

        // The conditioning: when, plus what. Added, as token and position
        // were in the GPT.
        let mut c = Matrix::from_vec(1, FREQUENCIES, timestep_features(t));
        for layer in self.time.iter_mut() {
            c = layer.forward(&c);
        }
        c.add_in_place(&self.labels.forward(&[label]));
        let c = self.cond_act.forward(&c);

        for block in self.blocks.iter_mut() {
            h = block.forward(&h, &c);
        }

        self.final_m = self.final_modulation.forward(&c).data;
        self.final_n = self.final_norm.forward(&h);
        let d = self.config.d_model;
        let h = modulate(&self.final_n, &self.final_m[..d], &self.final_m[d..]);
        unpatchify(&self.config, &self.out.forward(&h))
    }

    /// Given dLoss/dOutput, one number per pixel, accumulate every gradient.
    pub fn backward(&mut self, doutput: &[f32]) {
        let d = self.config.d_model;
        let dh = self.out.backward(&unpatchify_backward(&self.config, doutput));
        let (dn, dshift, dscale) = modulate_backward(&dh, &self.final_n, &self.final_m[d..]);
        let mut dh = self.final_norm.backward(&dn);
        let dm = [dshift, dscale].concat();
        let mut dc = self.final_modulation.backward(&Matrix::from_vec(1, 2 * d, dm));

        // Every block read the same `c`, so every block's blame for it adds.
        for block in self.blocks.iter_mut().rev() {
            let (dx, dcb) = block.backward(&dh);
            dh = dx;
            dc.add_in_place(&dcb);
        }
        // The positions are fixed, so the sequence's gradient stops at the
        // patch embedding, and the image below it is not ours to change.
        self.patches.backward(&dh);

        let mut dc = self.cond_act.backward(&dc);
        self.labels.backward(&dc);
        for layer in self.time.iter_mut().rev() {
            dc = layer.backward(&dc);
        }
    }

    pub fn zero_grad(&mut self) {
        self.patches.zero_grad();
        self.time.iter_mut().for_each(|l| l.zero_grad());
        self.labels.zero_grad();
        self.blocks.iter_mut().for_each(|b| b.zero_grad());
        self.final_modulation.zero_grad();
        self.out.zero_grad();
    }

    pub fn params(&mut self) -> Vec<Param<'_>> {
        let mut all = prefixed("patches", self.patches.params());
        for (i, layer) in self.time.iter_mut().enumerate() {
            all.extend(prefixed(&format!("time.{i}"), layer.params()));
        }
        all.extend(prefixed("labels", self.labels.params()));
        for (i, block) in self.blocks.iter_mut().enumerate() {
            all.extend(prefixed(&format!("blocks.{i}"), block.params()));
        }
        all.extend(prefixed("final.modulation", self.final_modulation.params()));
        all.extend(prefixed("out", self.out.params()));
        all
    }

    pub fn config(&self) -> DitConfig {
        self.config
    }

    pub fn param_count(&mut self) -> usize {
        self.params().iter().map(|p| p.value.len()).sum()
    }

    pub fn summary(&mut self) -> String {
        let DitConfig { image, patch, d_model, n_heads, n_layers, .. } = self.config;
        format!(
            "Dit({n_layers} layers, {n_heads} heads, d_model {d_model}, {image}x{image} in {patch}x{patch} patches = {} tokens, {} params)",
            self.config.tokens(),
            self.param_count()
        )
    }

    /// Every weight copied from `other`, a model of the same shape.
    pub fn copy_from(&mut self, other: &mut Dit) {
        for (ours, theirs) in self.params().into_iter().zip(other.params()) {
            ours.value.copy_from_slice(theirs.value);
        }
    }
}

// ---------------------------------------------------------------------------
// On disk
// ---------------------------------------------------------------------------
//
// # The layout is diffusers'
//
// A trained model is written as a diffusers pipeline directory, so that
// what `train_digits` makes is a `DiTTransformer2DModel` any DiT code can
// read, and `kvad` loads it by the same rules as every other image model:
//
// ```text
// model_index.json                             which pipeline, and the labels
// transformer/config.json                      DiT's config
// transformer/diffusion_pytorch_model.safetensors
// scheduler/scheduler_config.json              FlowMatchEulerDiscreteScheduler
// ```
//
// Two things about it are not DiT's, and both are written down rather than
// assumed. There is no VAE: this model draws pixels in [-1, 1], not latents,
// so there is nothing to decode and `model_index.json` names no `vae`. And
// the scheduler is flow matching, where DiT's is DDPM. diffusers'
// `FlowMatchEulerDiscreteScheduler` is exactly the sampler in `flow` —
// `xₜ = (1 − σ)x₀ + σε`, the model predicting `ε − x₀`, the timestep `σ·1000`
// — so the objective is recorded by naming it, and a loader that finds any
// other scheduler refuses.
//
// # One timestep embedder, written out once per block
//
// diffusers gives every block its own timestep MLP and label table
// (`transformer_blocks.N.norm1.emb`), and reads the output layer's
// conditioning from block 0's. Facebook's DiT had one of each, and its
// conversion copied that one into every block. So does `save`. `load` takes
// block 0's, and refuses a checkpoint whose blocks disagree: that is a model
// this one cannot represent, not one to be run with the difference ignored.

/// `_class_name` in `model_index.json`. Not `DiTPipeline`, because that
/// pipeline has a VAE and a DDPM scheduler, and code that saw the name would
/// run this model with both.
pub const PIPELINE: &str = "NervusDiTPipeline";
pub const TRANSFORMER: &str = "DiTTransformer2DModel";
pub const SCHEDULER: &str = "FlowMatchEulerDiscreteScheduler";
pub const WEIGHTS: &str = "transformer/diffusion_pytorch_model.safetensors";

/// Where one of this model's tensors goes in the file.
struct Place {
    /// Every name it is written under: one, or one per block for the
    /// conditioning's embedders.
    names: Vec<String>,
    shape: Vec<usize>,
    /// `(in, out)` for a `Linear` weight, which is stored `[in, out]` here and
    /// `[out, in]` by PyTorch, so it is transposed on the way.
    linear: Option<(usize, usize)>,
}

fn place(ours: &str, c: &DitConfig) -> Place {
    let d = c.d_model;
    let every_block = |sub: &str| -> Vec<String> {
        (0..c.n_layers).map(|i| format!("transformer_blocks.{i}.norm1.emb.{sub}")).collect()
    };
    if ours == "labels.table" {
        return Place { names: every_block("class_embedder.embedding_table.weight"), shape: vec![c.classes + 1, d], linear: None };
    }
    let (stem, what) = ours.rsplit_once('.').expect("a parameter's name has a dot in it");
    let (theirs, fan_in, fan_out) = match stem {
        "patches" => (vec!["pos_embed.proj".to_string()], c.patch_len(), d),
        "time.0" => (every_block("timestep_embedder.linear_1"), FREQUENCIES, d),
        "time.2" => (every_block("timestep_embedder.linear_2"), d, d),
        "final.modulation" => (vec!["proj_out_1".to_string()], d, 2 * d),
        "out" => (vec!["proj_out_2".to_string()], d, c.patch_len()),
        _ => {
            let (i, part) = stem.strip_prefix("blocks.").and_then(|r| r.split_once('.')).unwrap_or_else(|| panic!("no place for {ours}"));
            let (sub, fan_in, fan_out) = match part {
                "modulation" => ("norm1.linear", d, 6 * d),
                "attn.wq" => ("attn1.to_q", d, d),
                "attn.wk" => ("attn1.to_k", d, d),
                "attn.wv" => ("attn1.to_v", d, d),
                "attn.wo" => ("attn1.to_out.0", d, d),
                "mlp.0" => ("ff.net.0.proj", d, 4 * d),
                "mlp.2" => ("ff.net.2", 4 * d, d),
                _ => panic!("no place for {ours}"),
            };
            (vec![format!("transformer_blocks.{i}.{sub}")], fan_in, fan_out)
        }
    };
    let names = theirs.into_iter().map(|t| format!("{t}.{what}")).collect();
    match what {
        "bias" => Place { names, shape: vec![fan_out], linear: None },
        // The patch embedding is a convolution there, `[d, C, p, p]`, whose
        // flattened rows are in exactly the order `patchify` reads a patch.
        _ if stem == "patches" => Place { names, shape: vec![d, c.channels, c.patch, c.patch], linear: Some((fan_in, fan_out)) },
        _ => Place { names, shape: vec![fan_out, fan_in], linear: Some((fan_in, fan_out)) },
    }
}

/// `[rows, cols]` to `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Write `model` into `dir` as a diffusers pipeline. `labels` names each
/// class, in order: the words a prompt may use to ask for one.
pub fn save(dir: &Path, model: &mut Dit, labels: &[&str]) -> io::Result<()> {
    let c = model.config();
    assert_eq!(labels.len(), c.classes, "{} labels for {} classes", labels.len(), c.classes);
    for sub in ["transformer", "scheduler"] {
        std::fs::create_dir_all(dir.join(sub))?;
    }
    let write = |path: &str, json: Json| std::fs::write(dir.join(path), format!("{json}\n"));

    let labels = labels.iter().enumerate().map(|(i, &l)| (i.to_string(), Json::from(l))).collect();
    write(
        "model_index.json",
        object([
            ("_class_name", PIPELINE.into()),
            ("transformer", Json::Array(vec!["diffusers".into(), TRANSFORMER.into()])),
            ("scheduler", Json::Array(vec!["diffusers".into(), SCHEDULER.into()])),
            ("id2label", Json::Object(labels)),
        ]),
    )?;
    write(
        "transformer/config.json",
        object([
            ("_class_name", TRANSFORMER.into()),
            ("activation_fn", "gelu-approximate".into()),
            ("attention_bias", Json::Bool(true)),
            ("attention_head_dim", (c.d_model / c.n_heads).into()),
            ("dropout", Json::Number(0.0)),
            ("in_channels", c.channels.into()),
            ("norm_elementwise_affine", Json::Bool(false)),
            ("norm_eps", Json::Number(NORM_EPS as f64)),
            ("norm_num_groups", 32usize.into()),
            ("norm_type", "ada_norm_zero".into()),
            ("num_attention_heads", c.n_heads.into()),
            ("num_embeds_ada_norm", c.classes.into()),
            ("num_layers", c.n_layers.into()),
            // One number per pixel: a velocity. DiT's is twice that, because
            // it also learns a variance.
            ("out_channels", c.channels.into()),
            ("patch_size", c.patch.into()),
            ("sample_size", c.image.into()),
            ("upcast_attention", Json::Bool(false)),
        ]),
    )?;
    write(
        "scheduler/scheduler_config.json",
        object([
            ("_class_name", SCHEDULER.into()),
            ("num_train_timesteps", (TIMESTEP_SCALE as usize).into()),
            ("shift", Json::Number(1.0)),
        ]),
    )?;

    let mut tensors = Vec::new();
    for p in model.params() {
        let place = place(&p.name, &c);
        let data = match place.linear {
            Some((fan_in, fan_out)) => transpose(p.value, fan_in, fan_out),
            None => p.value.to_vec(),
        };
        for name in place.names {
            tensors.push(Tensor { name, shape: place.shape.clone(), data: data.clone() });
        }
    }
    write_safetensors(&dir.join(WEIGHTS), &tensors)
}

/// Read back what [`save`] wrote: the model, and its labels in class order.
pub fn load(dir: &Path) -> io::Result<(Dit, Vec<String>)> {
    let read = |path: &str| -> io::Result<Json> {
        let path = dir.join(path);
        let text = std::fs::read_to_string(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Json::parse(&text).map_err(|e| invalid(format!("{}: {e}", path.display())))
    };
    let index = read("model_index.json")?;
    let transformer = read("transformer/config.json")?;
    let scheduler = read("scheduler/scheduler_config.json")?;
    let bad = |what: String| invalid(format!("{}: {what}", dir.display()));

    let class = |json: &Json| json.get("_class_name").and_then(Json::as_str).map(str::to_string);
    for (json, wanted) in [(&index, PIPELINE), (&transformer, TRANSFORMER), (&scheduler, SCHEDULER)] {
        if class(json).as_deref() != Some(wanted) {
            return Err(bad(format!("{:?} where {wanted} was expected", class(json))));
        }
    }
    let number = |json: &Json, key: &str| match json.get(key) {
        Some(Json::Number(n)) => Some(*n),
        _ => None,
    };
    if number(&scheduler, "num_train_timesteps") != Some(TIMESTEP_SCALE as f64) || number(&scheduler, "shift") != Some(1.0) {
        return Err(bad("a scheduler with other timesteps or a shift; only 1000 timesteps and a shift of 1 are drawn here".into()));
    }
    let text = |key: &str| transformer.get(key).and_then(Json::as_str);
    let flag = |key: &str| transformer.get(key).cloned();
    if text("norm_type") != Some("ada_norm_zero")
        || text("activation_fn") != Some("gelu-approximate")
        || flag("attention_bias") != Some(Json::Bool(true))
        || flag("norm_elementwise_affine") != Some(Json::Bool(false))
        || number(&transformer, "norm_eps").is_none_or(|e| (e - NORM_EPS as f64).abs() > 1e-12)
    {
        return Err(bad("a DiT built differently from this one (its norms, activation or attention bias)".into()));
    }
    let size = |key: &str| transformer.get(key).and_then(Json::as_usize).ok_or_else(|| bad(format!("no `{key}` in the transformer's config")));
    let channels = size("in_channels")?;
    if size("out_channels")? != channels {
        return Err(bad(format!(
            "{} output channels for {channels} in: a DiT that also learns a variance, which this one does not",
            size("out_channels")?
        )));
    }
    let heads = size("num_attention_heads")?;
    let config = DitConfig {
        image: size("sample_size")?,
        channels,
        patch: size("patch_size")?,
        classes: size("num_embeds_ada_norm")?,
        d_model: heads * size("attention_head_dim")?,
        n_heads: heads,
        n_layers: size("num_layers")?,
    };
    let labels: Vec<String> = (0..config.classes)
        .map(|i| index.get("id2label").and_then(|l| l.get(&i.to_string())).and_then(Json::as_str).map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| bad(format!("`id2label` does not name all {} classes", config.classes)))?;

    let mut model = Dit::new(config, &mut Rng::new(0));
    let mut tensors: std::collections::HashMap<String, Tensor> =
        read_safetensors(&dir.join(WEIGHTS))?.into_iter().map(|t| (t.name.clone(), t)).collect();
    for p in model.params() {
        let place = place(&p.name, &config);
        let mut found = Vec::new();
        for name in &place.names {
            let t = tensors.remove(name).ok_or_else(|| bad(format!("no tensor `{name}`")))?;
            if t.shape != place.shape {
                return Err(bad(format!("`{name}` is {:?}, and the config makes it {:?}", t.shape, place.shape)));
            }
            found.push(t);
        }
        if let Some(other) = found.iter().position(|t| t.data != found[0].data) {
            return Err(bad(format!(
                "`{}` differs from `{}`: every block has an embedder of its own, and this model shares one",
                place.names[other], place.names[0]
            )));
        }
        let data = match place.linear {
            Some((fan_in, fan_out)) => transpose(&found[0].data, fan_out, fan_in),
            None => found.swap_remove(0).data,
        };
        p.value.copy_from_slice(&data);
    }
    if let Some(name) = tensors.keys().next() {
        return Err(bad(format!("tensor `{name}` belongs to nothing in this model")));
    }
    Ok((model, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{check_params, relative_error, scramble};

    const CONFIG: DitConfig = DitConfig { image: 8, channels: 2, patch: 4, classes: 3, d_model: 8, n_heads: 2, n_layers: 2 };

    fn random(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| rng.normal()).collect()
    }

    /// Mean squared error against a fixed target, the loss `flow` trains with.
    fn mse(output: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
        let n = output.len() as f32;
        let loss = output.iter().zip(target).map(|(o, t)| (o - t) * (o - t)).sum::<f32>() / n;
        (loss, output.iter().zip(target).map(|(o, t)| 2.0 * (o - t) / n).collect())
    }

    /// Two channels, so the patch orders in and out really differ. With one,
    /// a mix-up between them would pass everything.
    #[test]
    fn unpatchify_undoes_the_cut_and_its_backward_is_its_adjoint() {
        let mut rng = Rng::new(71);
        let image = random(CONFIG.pixels(), &mut rng);

        // In and out use different orders within a patch, so going round
        // needs the permutation between them: reorder input order to output.
        let cut = patchify(&CONFIG, &image);
        let mut reordered = Matrix::zeros(cut.rows, cut.cols);
        each_pixel(&CONFIG, |token, _, out_at, in_at| {
            reordered.data[token * CONFIG.patch_len() + out_at] = cut.data[token * CONFIG.patch_len() + in_at]
        });
        assert_eq!(unpatchify(&CONFIG, &reordered), image);
        assert_ne!(cut, reordered, "with two channels the orders should differ");

        // <unpatchify(a), b> = <a, backward(b)>: the definition of a backward
        // pass through something linear.
        let a = Matrix::from_vec(CONFIG.tokens(), CONFIG.patch_len(), random(CONFIG.pixels(), &mut rng));
        let b = random(CONFIG.pixels(), &mut rng);
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f32>();
        let left = dot(&unpatchify(&CONFIG, &a), &b);
        let right = dot(&a.data, &unpatchify_backward(&CONFIG, &b).data);
        assert!((left - right).abs() < 1e-4, "{left} against {right}");
    }

    /// The patch order DiT's convolution weight implies: channel, row, column.
    #[test]
    fn a_patch_is_read_in_the_order_of_a_convolution_weight() {
        let config = DitConfig { image: 4, channels: 2, patch: 2, ..CONFIG };
        let image: Vec<f32> = (0..config.pixels()).map(|i| i as f32).collect();
        let cut = patchify(&config, &image);
        // Top-left patch: channel 0's (0,0) (0,1) (1,0) (1,1), then channel 1's.
        assert_eq!(cut.row(0), &[0.0, 1.0, 4.0, 5.0, 16.0, 17.0, 20.0, 21.0]);
        // The one to its right starts two columns along.
        assert_eq!(cut.row(1)[0], 2.0);
    }

    #[test]
    fn a_fresh_model_predicts_nothing_and_its_blocks_are_the_identity() {
        let mut rng = Rng::new(72);
        let mut model = Dit::new(CONFIG, &mut rng);
        let out = model.forward(&random(CONFIG.pixels(), &mut rng), 0.3, 1);
        assert!(out.iter().all(|&v| v == 0.0), "adaLN-Zero should start at exactly zero");

        let x = Matrix::from_vec(CONFIG.tokens(), CONFIG.d_model, random(CONFIG.tokens() * CONFIG.d_model, &mut rng));
        let c = Matrix::from_vec(1, CONFIG.d_model, random(CONFIG.d_model, &mut rng));
        assert_eq!(model.blocks[0].forward(&x, &c), x);
    }

    /// One block, checked on both of its inputs: the sequence, and the
    /// conditioning, whose gradient arrives through six chunks at once.
    #[test]
    fn block_gradient_matches_numerical() {
        let mut rng = Rng::new(73);
        let mut block = DitBlock::new(CONFIG.d_model, CONFIG.n_heads, &mut rng);
        // Off zero, or the gates hide everything behind them.
        scramble(block.params(), &mut rng);
        let mut x = Matrix::from_vec(CONFIG.tokens(), CONFIG.d_model, random(CONFIG.tokens() * CONFIG.d_model, &mut rng));
        let mut c = Matrix::from_vec(1, CONFIG.d_model, random(CONFIG.d_model, &mut rng));
        let target = random(x.data.len(), &mut rng);
        let loss = |b: &mut DitBlock, x: &Matrix, c: &Matrix| mse(&b.forward(x, c).data, &target).0;

        block.zero_grad();
        let (_, dy) = mse(&block.forward(&x, &c).data, &target);
        let (dx, dc) = block.backward(&Matrix::from_vec(x.rows, x.cols, dy));

        let report = check_params(&mut block, DitBlock::params, |b| loss(b, &x, &c), NUDGE);
        assert_eq!(report.len(), 2 + 8 + 4);
        for r in report {
            if r.name.ends_with("wk.bias") {
                // Zero by construction (see the attention tests), so this is
                // rounding: 1e-5 measured here, against gradients near 1.
                assert!(r.analytic_norm < 1e-4, "{}: {:e}", r.name, r.analytic_norm);
                continue;
            }
            assert!(r.rel < TOLERANCE, "{}: analytic and numerical gradients differ (rel {:.4})", r.name, r.rel);
        }

        let numerical = numerical_input(&mut x, |x| loss(&mut block, x, &c));
        let rel = relative_error(&dx.data, &numerical);
        assert!(rel < TOLERANCE, "dx: analytic and numerical gradients differ (rel {rel:.4})");
        let numerical = numerical_input(&mut c, |c| loss(&mut block, &x, c));
        let rel = relative_error(&dc.data, &numerical);
        assert!(rel < TOLERANCE, "dc: analytic and numerical gradients differ (rel {rel:.4})");
    }

    /// dLoss/dInput, measured by nudging each element of `input` in turn.
    fn numerical_input(input: &mut Matrix, mut loss: impl FnMut(&Matrix) -> f32) -> Vec<f32> {
        let mut numerical = vec![0.0; input.data.len()];
        for (i, slot) in numerical.iter_mut().enumerate() {
            input.data[i] += NUDGE;
            let up = loss(input);
            input.data[i] -= 2.0 * NUDGE;
            let down = loss(input);
            input.data[i] += NUDGE;
            *slot = (up - down) / (2.0 * NUDGE);
        }
        numerical
    }

    /// Every tensor in the model, through patches, positions, the timestep
    /// MLP, the label table, both blocks and the modulated output layer.
    #[test]
    fn analytic_gradient_matches_numerical() {
        let mut rng = Rng::new(74);
        let mut model = Dit::new(CONFIG, &mut rng);
        scramble(model.params(), &mut rng);
        let x = random(CONFIG.pixels(), &mut rng);
        let target = random(CONFIG.pixels(), &mut rng);
        let (t, label) = (0.37, 2);

        model.zero_grad();
        let (_, dy) = mse(&model.forward(&x, t, label), &target);
        model.backward(&dy);

        // Only the label that was asked for is to blame.
        let table = model.params().into_iter().find(|p| p.name == "labels.table").unwrap();
        assert!(table.grad.iter().enumerate().all(|(i, &g)| i / CONFIG.d_model == label || g == 0.0));

        let report = check_params(&mut model, Dit::params, |m| mse(&m.forward(&x, t, label), &target).0, NUDGE);
        // patches 2, time 4, labels 1, 14 per block, final modulation 2, out 2.
        assert_eq!(report.len(), 2 + 4 + 1 + 14 * CONFIG.n_layers + 2 + 2);
        for r in report {
            if r.name.ends_with("wk.bias") {
                // Zero by construction (see the attention tests), so this is
                // rounding: 1e-5 measured here, against gradients near 1.
                assert!(r.analytic_norm < 1e-4, "{}: {:e}", r.name, r.analytic_norm);
                continue;
            }
            assert!(r.rel < TOLERANCE, "{}: analytic and numerical gradients differ (rel {:.4})", r.name, r.rel);
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nervus-dit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    const LABELS: [&str; 3] = ["cat", "dog", "fish"];

    #[test]
    fn a_saved_model_loads_and_draws_the_same() {
        let dir = scratch("roundtrip");
        let mut rng = Rng::new(75);
        let mut model = Dit::new(CONFIG, &mut rng);
        scramble(model.params(), &mut rng);
        save(&dir, &mut model, &LABELS).unwrap();
        let (mut loaded, labels) = load(&dir).unwrap();
        assert_eq!(loaded.config(), CONFIG);
        assert_eq!(labels, LABELS);

        let x = random(CONFIG.pixels(), &mut rng);
        assert_eq!(model.forward(&x, 0.5, 1), loaded.forward(&x, 0.5, 1));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The contract with diffusers: every name and shape a
    /// `DiTTransformer2DModel` with this config has, and nothing else. Written
    /// out by hand from diffusers' modules, not derived from `place`, so that
    /// the two can disagree.
    #[test]
    fn the_file_holds_what_diffusers_expects_and_nothing_else() {
        let dir = scratch("layout");
        let mut model = Dit::new(CONFIG, &mut Rng::new(76));
        save(&dir, &mut model, &LABELS).unwrap();
        let tensors = read_safetensors(&dir.join(WEIGHTS)).unwrap();
        let found: std::collections::BTreeMap<String, Vec<usize>> = tensors.into_iter().map(|t| (t.name, t.shape)).collect();

        let (d, pl, c, p) = (CONFIG.d_model, CONFIG.patch_len(), CONFIG.channels, CONFIG.patch);
        let mut wanted = std::collections::BTreeMap::new();
        let mut linear = |name: String, fan_in: usize, fan_out: usize| {
            wanted.insert(format!("{name}.weight"), vec![fan_out, fan_in]);
            wanted.insert(format!("{name}.bias"), vec![fan_out]);
        };
        linear("proj_out_1".into(), d, 2 * d);
        linear("proj_out_2".into(), d, pl);
        for i in 0..CONFIG.n_layers {
            let b = format!("transformer_blocks.{i}");
            linear(format!("{b}.norm1.emb.timestep_embedder.linear_1"), FREQUENCIES, d);
            linear(format!("{b}.norm1.emb.timestep_embedder.linear_2"), d, d);
            linear(format!("{b}.norm1.linear"), d, 6 * d);
            for q in ["to_q", "to_k", "to_v", "to_out.0"] {
                linear(format!("{b}.attn1.{q}"), d, d);
            }
            linear(format!("{b}.ff.net.0.proj"), d, 4 * d);
            linear(format!("{b}.ff.net.2"), 4 * d, d);
        }
        for i in 0..CONFIG.n_layers {
            wanted.insert(format!("transformer_blocks.{i}.norm1.emb.class_embedder.embedding_table.weight"), vec![CONFIG.classes + 1, d]);
        }
        wanted.insert("pos_embed.proj.weight".into(), vec![d, c, p, p]);
        wanted.insert("pos_embed.proj.bias".into(), vec![d]);
        assert_eq!(found, wanted);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `[in, out]` here, `[out, in]` there: element `[i][o]` of ours is
    /// element `[o][i]` of theirs. A save and load that both forgot would
    /// round-trip perfectly, so check one weight against the file directly.
    #[test]
    fn a_weight_is_stored_the_way_pytorch_stores_it() {
        let dir = scratch("transposed");
        let mut model = Dit::new(CONFIG, &mut Rng::new(77));
        scramble(model.params(), &mut Rng::new(78));
        let ours = model.params().into_iter().find(|p| p.name == "blocks.1.mlp.0.weight").unwrap().value.to_vec();
        save(&dir, &mut model, &LABELS).unwrap();
        let theirs = read_safetensors(&dir.join(WEIGHTS)).unwrap().into_iter().find(|t| t.name == "transformer_blocks.1.ff.net.0.proj.weight").unwrap();
        let (fan_in, fan_out) = (CONFIG.d_model, 4 * CONFIG.d_model);
        assert_eq!(theirs.shape, vec![fan_out, fan_in]);
        for (i, o) in [(0, 1), (3, 17), (fan_in - 1, fan_out - 1)] {
            assert_eq!(ours[i * fan_out + o], theirs.data[o * fan_in + i]);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// What `load` refuses, each by name: another scheduler (so another
    /// objective), a DiT that learns a variance, and blocks with embedders
    /// of their own.
    #[test]
    fn a_model_this_one_cannot_be_is_refused_by_name() {
        let dir = scratch("refused");
        let mut model = Dit::new(CONFIG, &mut Rng::new(79));
        let refusal = |edit: &dyn Fn(&Path)| -> String {
            let _ = std::fs::remove_dir_all(&dir);
            save(&dir, &mut Dit::new(CONFIG, &mut Rng::new(79)), &LABELS).unwrap();
            edit(&dir);
            load(&dir).err().expect("loaded a model it should have refused").to_string()
        };
        let rewrite = |path: &Path, from: &str, to: &str| {
            let text = std::fs::read_to_string(path).unwrap();
            assert!(text.contains(from), "{from} is not in {}", path.display());
            std::fs::write(path, text.replace(from, to)).unwrap();
        };

        let err = refusal(&|d| rewrite(&d.join("scheduler/scheduler_config.json"), SCHEDULER, "DDPMScheduler"));
        assert!(err.contains("DDPMScheduler"), "{err}");
        let err = refusal(&|d| rewrite(&d.join("transformer/config.json"), "\"out_channels\":2", "\"out_channels\":4"));
        assert!(err.contains("variance"), "{err}");

        let err = refusal(&|d| {
            let path = d.join(WEIGHTS);
            let mut tensors = read_safetensors(&path).unwrap();
            let t = tensors.iter_mut().find(|t| t.name == "transformer_blocks.1.norm1.emb.timestep_embedder.linear_1.bias").unwrap();
            t.data[0] += 1.0;
            write_safetensors(&path, &tensors).unwrap();
        });
        assert!(err.contains("embedder of its own"), "{err}");

        // And the one it can: untouched, it loads.
        save(&dir, &mut model, &LABELS).unwrap();
        assert!(load(&dir).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// nervus's DiT against diffusers' on the same saved model: the file
    /// layout, the timestep's frequencies, the positions, the norms' eps and
    /// the patch orders all have to be right for these to agree. See
    /// `scripts/dit-fixtures.py`, which writes the reference.
    #[test]
    #[ignore]
    fn agrees_with_diffusers() {
        let (Some(model), Some(fixtures)) = (std::env::var_os("KVAD_DIT_MODEL"), std::env::var_os("KVAD_DIT_FIXTURES")) else {
            panic!("set KVAD_DIT_MODEL and KVAD_DIT_FIXTURES; see scripts/dit-fixtures.py");
        };
        let (mut dit, _) = load(Path::new(&model)).unwrap();
        let fx: std::collections::HashMap<String, Vec<f32>> =
            read_safetensors(Path::new(&fixtures)).unwrap().into_iter().map(|t| (t.name, t.data)).collect();
        let x = &fx["x"];
        let mut case = 0;
        while let Some(expected) = fx.get(&format!("case.{case}.out")) {
            let t = fx[&format!("case.{case}.t")][0];
            let label = fx[&format!("case.{case}.label")][0] as usize;
            let ours = dit.forward(x, t, label);
            let worst = ours.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            let size = expected.iter().map(|v| v.abs()).fold(0.0, f32::max);
            println!("case {case}: t {t}, label {label}: worst difference {worst:e}, largest output {size}");
            assert!(worst < 1e-4 * size.max(1.0), "case {case}: nervus and diffusers differ by {worst}");
            case += 1;
        }
        assert!(case > 0, "no cases in {}", fixtures.to_string_lossy());
    }

    const NUDGE: f32 = 1e-2;
    const TOLERANCE: f32 = 5e-3;
}
