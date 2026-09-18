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
}

/// `y = x @ w + b`, where `x` is a single vector of length `w.rows`.
///
/// Generating one token at a time means every matmul in the model is really a
/// matrix-*vector* product, which is memory-bandwidth bound rather than
/// compute bound: for each output we read a whole column of weights and do one
/// multiply-add. That is why LLM inference speed tracks memory bandwidth, and
/// why quantisation (fewer bytes per weight) speeds things up so much.
///
/// We split the output range across threads so each thread reads a contiguous
/// slice of every weight row.
pub fn matvec(x: &[f32], w: &Tensor, bias: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), w.rows, "matvec shape mismatch");
    let n = w.cols;
    let mut out = match bias {
        Some(b) => b.to_vec(),
        None => vec![0.0; n],
    };

    // Chunk size chosen so each thread's working set stays in L2.
    let chunk = (n / rayon::current_num_threads().max(1)).max(64);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, out_chunk)| {
        let j0 = ci * chunk;
        let width = out_chunk.len();
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let w_row = &w.data[i * n + j0..i * n + j0 + width];
            for j in 0..width {
                out_chunk[j] += xi * w_row[j];
            }
        }
    });
    out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_hand_computation() {
        // x = [1, 2]; w = [[1, 2, 3], [4, 5, 6]]  ->  [9, 12, 15]
        let w = Tensor::new(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(matvec(&[1.0, 2.0], &w, None), vec![9.0, 12.0, 15.0]);
        assert_eq!(matvec(&[1.0, 2.0], &w, Some(&[1., 1., 1.])), vec![10.0, 13.0, 16.0]);
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
}
