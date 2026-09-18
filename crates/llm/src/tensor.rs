//! The five operations a transformer is made of.
//!
//! There is no tensor library here and no autodiff: inference only ever runs
//! forward, so all we need are matrix products and three elementwise
//! functions. A "tensor" is a flat `Vec<f32>` plus a shape, row-major.
//!
//! Every weight matrix is stored `[in_features, out_features]`, which is the
//! layout HuggingFace's GPT-2 checkpoint already uses (it was written for a
//! `Conv1D` layer rather than a `Linear`, so no transposes are needed).

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "shape {rows}x{cols} != {} values", data.len());
        Tensor { rows, cols, data }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Swap the axes, producing `[cols, rows]`.
    ///
    /// Used once per GPT-2 weight at load time. Its checkpoint was written for
    /// a `Conv1D` layer, which stores `[in, out]`; everything else in the
    /// world uses `nn.Linear`'s `[out, in]`. Normalising at load costs one
    /// pass over each matrix and means the whole engine has exactly one matmul
    /// kernel — which in turn means GPT-2 gets the quantised integer path for
    /// free.
    pub fn transposed(&self) -> Tensor {
        let mut data = vec![0.0f32; self.data.len()];
        for r in 0..self.rows {
            let row = self.row(r);
            for c in 0..self.cols {
                data[c * self.rows + r] = row[c];
            }
        }
        Tensor { rows: self.cols, cols: self.rows, data }
    }
}

/// `y = x @ wᵀ`, where `w` is `[out_features, in_features]`.
///
/// Used for the output head, where the weights are the token embedding matrix
/// re-read in the other direction. Here each output is one dot product against
/// a contiguous row, so we parallelise over outputs directly.
pub fn matvec_bt(x: &[f32], w: &Tensor) -> Vec<f32> {
    assert_eq!(x.len(), w.cols, "matvec_bt shape mismatch");
    (0..w.rows)
        .into_par_iter()
        .map(|r| {
            let row = w.row(r);
            let mut acc = 0.0f32;
            for i in 0..row.len() {
                acc += x[i] * row[i];
            }
            acc
        })
        .collect()
}

/// Layer normalisation: centre, scale to unit variance, then apply a learned
/// gain and bias.
///
/// This is not a nicety. Residual streams accumulate: every block adds its
/// output back into `x`, so without renormalising before each block the
/// magnitudes grow layer over layer until the softmax saturates. Normalising
/// *per token vector* (not across the batch) is what makes transformers train
/// stably at depth.
pub fn layer_norm(x: &[f32], gain: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(gain.iter())
        .zip(bias.iter())
        .map(|((&v, &g), &b)| (v - mean) * inv * g + b)
        .collect()
}

/// GELU, in the tanh approximation that GPT-2 was actually trained with.
///
/// Unlike ReLU, GELU is smooth and lets slightly-negative values through
/// attenuated rather than zeroing them. The exact form uses the Gaussian CDF;
/// OpenAI used this cheaper approximation, so we must too — matching the
/// training-time nonlinearity exactly is part of reproducing the model.
pub fn gelu_inplace(x: &mut [f32]) {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (C * (t + 0.044715 * t * t * t)).tanh());
    }
}

/// Softmax, in place, with the max subtracted first for numerical stability.
pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

// ---------------------------------------------------------------------------
// The Llama-family replacements.
//
// Between GPT-2 (2019) and Llama (2023) every sub-component of the block was
// swapped out, but the skeleton -- residual stream, attention, MLP -- did not
// change at all. These four functions are the entire difference.
// ---------------------------------------------------------------------------

/// RMSNorm: like LayerNorm, but without the centring step and without a bias.
///
/// LayerNorm subtracts the mean, then divides by the standard deviation.
/// Someone eventually checked whether the subtraction was doing any work, and
/// it was not: only the rescaling matters. Dropping it saves a pass over the
/// data and a whole bias vector per normalisation, of which a transformer has
/// two per layer.
///
/// This is worth noticing as a pattern. A lot of "architecture progress" is
/// finding out that a piece everyone inherited from the previous paper was
/// never load-bearing.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_square + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&v, &w)| v * inv * w).collect()
}

/// SwiGLU: `down( silu(gate(x)) * up(x) )`, the multiplication elementwise.
///
/// GPT-2's MLP is `down(gelu(up(x)))` -- two matrices. This is three: one
/// produces values, another produces a *gate* that scales them, and only then
/// does the projection back down. The network learns not just what to compute
/// but how much of it to let through, per channel.
///
/// `silu(z) = z * sigmoid(z)` -- like GELU, a smooth almost-ReLU, and cheaper.
/// This function does the gating in place on `gate`.
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        let silu = *g / (1.0 + (-*g).exp());
        *g = silu * u;
    }
}

/// Precomputed rotation angles for rotary position embedding.
///
/// # Why RoPE replaced learned position embeddings
///
/// GPT-2 has a `wpe` matrix with one learned row per position, which is why it
/// stops dead at 1024 tokens: there is no row 1025, and no way to invent one.
///
/// RoPE encodes position differently. Split each head's vector into pairs of
/// coordinates, treat each pair as a point in a 2D plane, and *rotate* it by an
/// angle proportional to the token's position. Low-index pairs rotate fast,
/// high-index pairs rotate slowly -- like the hands of a clock, giving a
/// representation that is unique over an enormous range.
///
/// The elegant part is what happens in attention. The dot product of two
/// rotated vectors depends only on the *difference* of their angles, so a query
/// at position 900 and a key at position 850 interact exactly as one at 50 and
/// one at 0. Position enters the model as relative distance, for free, with no
/// parameters at all -- and nothing breaks structurally if you go past the
/// training length.
pub struct Rope {
    /// [max_positions, head_dim / 2]
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Rope {
    pub fn new(head_dim: usize, max_positions: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_positions * half);
        let mut sin = Vec::with_capacity(max_positions * half);
        for pos in 0..max_positions {
            for i in 0..half {
                // Each pair gets its own frequency, geometrically spaced from
                // 1 down to 1/theta. `theta` (10000 originally, 100000+ in
                // recent models) sets how slow the slowest pair is, and so how
                // far the encoding stays unambiguous.
                let inv_freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * inv_freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Rope { cos, sin, half }
    }

    /// Rotate one head's vector in place, for a token at `pos`.
    ///
    /// HuggingFace pairs coordinate `i` with `i + head_dim/2` rather than
    /// `2i` with `2i+1`. The two conventions are a permutation apart and give
    /// identical results *as long as the weights were trained with the same
    /// one* -- get it wrong and output degrades to fluent nonsense rather
    /// than failing loudly.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        debug_assert_eq!(x.len(), self.half * 2);
        let base = pos * self.half;
        for i in 0..self.half {
            let (c, s) = (self.cos[base + i], self.sin[base + i]);
            let (a, b) = (x[i], x[i + self.half]);
            x[i] = a * c - b * s;
            x[i + self.half] = b * c + a * s;
        }
    }

    pub fn max_positions(&self) -> usize {
        if self.half == 0 {
            0
        } else {
            self.cos.len() / self.half
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_swaps_axes_and_is_its_own_inverse() {
        let t = Tensor::new(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let tt = t.transposed();
        assert_eq!((tt.rows, tt.cols), (3, 2));
        assert_eq!(tt.data, vec![1., 4., 2., 5., 3., 6.]);
        assert_eq!(tt.transposed().data, t.data);
    }

    #[test]
    fn transpose_turns_matvec_into_matvec_bt() {
        // The identity GPT-2 relies on: x @ W == x @ (Wᵀ)ᵀ.
        let w = Tensor::new(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let x = [1.0f32, 2.0];
        assert_eq!(matvec_bt(&x, &w.transposed()), vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn matvec_bt_matches_hand_computation() {
        // w rows are the vectors we dot against.
        let w = Tensor::new(3, 2, vec![1., 0., 0., 1., 1., 1.]);
        assert_eq!(matvec_bt(&[3.0, 5.0], &w), vec![3.0, 5.0, 8.0]);
    }

    #[test]
    fn layer_norm_produces_zero_mean_unit_variance() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let g = [1.0f32; 4];
        let b = [0.0f32; 4];
        let y = layer_norm(&x, &g, &b, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        let var: f32 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = vec![1000.0f32, 1000.0, 1000.0];
        softmax_inplace(&mut x);
        assert!(x.iter().all(|v| (v - 1.0 / 3.0).abs() < 1e-6), "{x:?}");

        let mut y = vec![0.0f32, 1.0, 2.0];
        softmax_inplace(&mut y);
        assert!((y.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(y[2] > y[1] && y[1] > y[0]);
    }

    #[test]
    fn gelu_shape_is_right() {
        let mut x = vec![-3.0f32, 0.0, 3.0];
        gelu_inplace(&mut x);
        assert!(x[0] < 0.0 && x[0] > -0.01, "gelu(-3) = {}", x[0]);
        assert!(x[1].abs() < 1e-6);
        assert!((x[2] - 2.9959).abs() < 1e-3, "gelu(3) = {}", x[2]);
    }

    #[test]
    fn rms_norm_rescales_without_centring() {
        // Unlike LayerNorm, the mean is left alone: an all-ones input stays
        // all ones rather than collapsing to zero.
        let x = [1.0f32; 8];
        let w = [1.0f32; 8];
        let y = rms_norm(&x, &w, 1e-6);
        assert!(y.iter().all(|v| (v - 1.0).abs() < 1e-4), "{y:?}");

        // Root mean square of the output should be 1.
        let x = [3.0f32, -4.0, 0.0, 5.0];
        let y = rms_norm(&x, &[1.0; 4], 1e-6);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms {rms}");
    }

    #[test]
    fn swiglu_gates() {
        let mut gate = vec![0.0f32, 10.0, -10.0];
        let up = vec![2.0f32, 2.0, 2.0];
        swiglu_inplace(&mut gate, &up);
        // silu(0) = 0, so the gate is shut regardless of `up`.
        assert!(gate[0].abs() < 1e-6);
        // silu(10) ~= 10, so ~10 * 2.
        assert!((gate[1] - 20.0).abs() < 0.01, "{}", gate[1]);
        // silu(-10) ~= 0, gate shut again.
        assert!(gate[2].abs() < 0.01, "{}", gate[2]);
    }

    #[test]
    fn rope_preserves_length_and_encodes_relative_position() {
        let rope = Rope::new(8, 64, 10000.0);

        // A rotation cannot change a vector's length.
        let mut x = vec![1.0f32, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25, 1.5];
        let before: f32 = x.iter().map(|v| v * v).sum();
        rope.apply(&mut x, 37);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-3, "{before} vs {after}");

        // Position 0 is the identity: angle 0, so cos=1 and sin=0.
        let original = vec![1.0f32, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25, 1.5];
        let mut at_zero = original.clone();
        rope.apply(&mut at_zero, 0);
        assert!(at_zero.iter().zip(original.iter()).all(|(a, b)| (a - b).abs() < 1e-6));

        // The property that makes RoPE work: the dot product of a rotated
        // query and a rotated key depends only on the distance between them.
        let q = vec![0.3f32, -1.2, 0.7, 2.0, -0.5, 1.1, 0.9, -0.4];
        let k = vec![1.0f32, 0.4, -0.8, 0.2, 1.7, -1.3, 0.6, 0.1];
        let dot_at = |qp: usize, kp: usize| -> f32 {
            let (mut a, mut b) = (q.clone(), k.clone());
            rope.apply(&mut a, qp);
            rope.apply(&mut b, kp);
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };
        // Same gap of 10, four very different absolute positions.
        let reference = dot_at(10, 0);
        for start in [5usize, 20, 40, 50] {
            let shifted = dot_at(start + 10, start);
            assert!(
                (shifted - reference).abs() < 1e-3,
                "positions {}/{} gave {shifted}, expected {reference}",
                start + 10,
                start
            );
        }
    }
}
