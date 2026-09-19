//! Layer normalisation and RMS normalisation, forward and backward.
//!
//! # Why a transformer needs this at all
//!
//! A transformer block *adds* its output to its input, and there are dozens of
//! blocks. Nothing in that sum keeps the numbers the same size from one layer
//! to the next, and He initialisation — the trick that did that job in the
//! MLP — only holds at the first step of training. So each block first
//! rescales its input, row by row, to a standard size, and the problem is
//! fixed by construction rather than by hoping the weights behave.
//!
//! # The one idea in this file
//!
//! Forward is forgettable: subtract the row's mean, divide by its standard
//! deviation. Backward is where people go wrong, and there is a way to see the
//! answer without grinding through the algebra.
//!
//! **A normalised row cannot see some changes to its input.** Add 5 to every
//! element and the mean absorbs it; the output does not move. Stretch the row
//! about its mean and the standard deviation absorbs it; the output does not
//! move. If the output cannot see a change, the loss cannot either, so the
//! gradient must have *no component* along those two directions:
//!
//! ```text
//! dx = ( dx̂  -  mean(dx̂)  -  x̂ * mean(dx̂ * x̂) ) / std
//!              ^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
//!              remove the    remove the "stretch"
//!              "shift" part  part (x̂ is that direction)
//! ```
//!
//! Each subtraction is a projection: it removes from `dx̂` whatever points
//! along a direction the layer ignores. Forget either one and the network
//! still trains — towards the wrong place.
//!
//! [`RmsNorm`] makes the point from the other side. It divides by the root
//! mean square and never subtracts the mean, so it ignores stretching but
//! *does* see a shift. One invariance, one projection: its backward is the
//! line above with the `mean(dx̂)` term deleted. It is what Llama and Qwen
//! use — the mean subtraction turned out not to be earning its cost.

use crate::matrix::Matrix;
use crate::nn::{sgd, Layer, Param};

/// Added to the variance before the square root, so a constant row divides by
/// something small rather than by zero.
const EPS: f32 = 1e-5;

/// `y = gamma * (x - mean) / std + beta`, each row on its own.
///
/// `gamma` and `beta` give back the freedom normalising took away: if the
/// network would rather a feature were large, or off-centre, it can learn that.
pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    dgamma: Vec<f32>,
    dbeta: Vec<f32>,
    vgamma: Vec<f32>,
    vbeta: Vec<f32>,
    /// The normalised rows, before gamma and beta. Backward needs these and,
    /// conveniently, not the original input.
    x_hat: Matrix,
    /// 1/std of each row.
    inv_std: Vec<f32>,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        LayerNorm {
            // Start as a plain normalisation: scale by 1, shift by 0.
            gamma: vec![1.0; features],
            beta: vec![0.0; features],
            dgamma: vec![0.0; features],
            dbeta: vec![0.0; features],
            vgamma: vec![0.0; features],
            vbeta: vec![0.0; features],
            x_hat: Matrix::zeros(0, features),
            inv_std: Vec::new(),
        }
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let n = x.cols as f32;
        let mut y = Matrix::zeros(x.rows, x.cols);
        self.x_hat = Matrix::zeros(x.rows, x.cols);
        self.inv_std.clear();

        for r in 0..x.rows {
            let row = x.row(r);
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv_std = 1.0 / (var + EPS).sqrt();

            let x_hat = self.x_hat.row_mut(r);
            for (j, out) in y.row_mut(r).iter_mut().enumerate() {
                x_hat[j] = (row[j] - mean) * inv_std;
                *out = self.gamma[j] * x_hat[j] + self.beta[j];
            }
            self.inv_std.push(inv_std);
        }
        y
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        let n = dy.cols as f32;
        let mut dx = Matrix::zeros(dy.rows, dy.cols);

        for r in 0..dy.rows {
            let dy_row = dy.row(r);
            let x_hat = self.x_hat.row(r);

            // y = gamma * x̂ + beta is a Linear layer with a diagonal weight,
            // and these are Linear's rules: the bias collects dy, the weight
            // collects input * dy, and the input gets dy * weight.
            let mut dx_hat = vec![0.0; dy.cols];
            for j in 0..dy.cols {
                self.dbeta[j] += dy_row[j];
                self.dgamma[j] += dy_row[j] * x_hat[j];
                dx_hat[j] = dy_row[j] * self.gamma[j];
            }

            // The two projections from the top of this file.
            let shift = dx_hat.iter().sum::<f32>() / n;
            let stretch = dx_hat.iter().zip(x_hat).map(|(g, h)| g * h).sum::<f32>() / n;
            for (j, out) in dx.row_mut(r).iter_mut().enumerate() {
                *out = (dx_hat[j] - shift - x_hat[j] * stretch) * self.inv_std[r];
            }
        }
        dx
    }

    fn step(&mut self, lr: f32, momentum: f32) {
        sgd(&mut self.gamma, &self.dgamma, &mut self.vgamma, lr, momentum);
        sgd(&mut self.beta, &self.dbeta, &mut self.vbeta, lr, momentum);
    }

    fn zero_grad(&mut self) {
        self.dgamma.fill(0.0);
        self.dbeta.fill(0.0);
    }

    fn describe(&self) -> String {
        format!("LayerNorm({}, {} params)", self.gamma.len(), 2 * self.gamma.len())
    }

    fn params(&mut self) -> Vec<Param<'_>> {
        vec![
            Param::new("gamma", &mut self.gamma, &self.dgamma),
            Param::new("beta", &mut self.beta, &self.dbeta),
        ]
    }
}

/// `y = gamma * x / rms(x)`, each row on its own. No mean, no beta.
pub struct RmsNorm {
    gamma: Vec<f32>,
    dgamma: Vec<f32>,
    vgamma: Vec<f32>,
    x_hat: Matrix,
    /// 1/rms of each row.
    inv_rms: Vec<f32>,
}

impl RmsNorm {
    pub fn new(features: usize) -> Self {
        RmsNorm {
            gamma: vec![1.0; features],
            dgamma: vec![0.0; features],
            vgamma: vec![0.0; features],
            x_hat: Matrix::zeros(0, features),
            inv_rms: Vec::new(),
        }
    }
}

impl Layer for RmsNorm {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let n = x.cols as f32;
        let mut y = Matrix::zeros(x.rows, x.cols);
        self.x_hat = Matrix::zeros(x.rows, x.cols);
        self.inv_rms.clear();

        for r in 0..x.rows {
            let row = x.row(r);
            let mean_square = row.iter().map(|v| v * v).sum::<f32>() / n;
            let inv_rms = 1.0 / (mean_square + EPS).sqrt();

            let x_hat = self.x_hat.row_mut(r);
            for (j, out) in y.row_mut(r).iter_mut().enumerate() {
                x_hat[j] = row[j] * inv_rms;
                *out = self.gamma[j] * x_hat[j];
            }
            self.inv_rms.push(inv_rms);
        }
        y
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        let n = dy.cols as f32;
        let mut dx = Matrix::zeros(dy.rows, dy.cols);

        for r in 0..dy.rows {
            let dy_row = dy.row(r);
            let x_hat = self.x_hat.row(r);

            let mut dx_hat = vec![0.0; dy.cols];
            for j in 0..dy.cols {
                self.dgamma[j] += dy_row[j] * x_hat[j];
                dx_hat[j] = dy_row[j] * self.gamma[j];
            }

            // Only the stretch is invisible to this layer, so only the
            // stretch is projected out. A shift is seen, and is blamed.
            let stretch = dx_hat.iter().zip(x_hat).map(|(g, h)| g * h).sum::<f32>() / n;
            for (j, out) in dx.row_mut(r).iter_mut().enumerate() {
                *out = (dx_hat[j] - x_hat[j] * stretch) * self.inv_rms[r];
            }
        }
        dx
    }

    fn step(&mut self, lr: f32, momentum: f32) {
        sgd(&mut self.gamma, &self.dgamma, &mut self.vgamma, lr, momentum);
    }

    fn zero_grad(&mut self) {
        self.dgamma.fill(0.0);
    }

    fn describe(&self) -> String {
        format!("RmsNorm({}, {} params)", self.gamma.len(), self.gamma.len())
    }

    fn params(&mut self) -> Vec<Param<'_>> {
        vec![Param::new("gamma", &mut self.gamma, &self.dgamma)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::relative_error;
    use crate::nn::softmax_cross_entropy;
    use crate::rng::Rng;

    const ROWS: usize = 4;
    const FEATURES: usize = 6;
    const TARGETS: [usize; ROWS] = [2, 0, 5, 3];

    /// Inputs that are deliberately *not* already normalised, and a gamma that
    /// is deliberately not 1. With x ~ N(0, 1) and a fresh layer, std ≈ 1 and
    /// gamma = 1, so a backward that forgot either factor would still pass.
    fn input(rng: &mut Rng) -> Matrix {
        Matrix::from_vec(ROWS, FEATURES, (0..ROWS * FEATURES).map(|_| 3.0 * rng.normal() + 2.0).collect())
    }

    fn scramble(params: &mut [f32], rng: &mut Rng) {
        params.iter_mut().for_each(|p| *p = 1.0 + 0.5 * rng.normal());
    }

    /// Ten times the MLP check's nudge. The inputs here are several times
    /// larger, and what limits this estimate is f32 rounding in the loss, not
    /// curvature: measured, the correct gradients disagree by 0.0009 at 1e-3
    /// and 0.0001 at 1e-2.
    const NUDGE: f32 = 1e-2;
    const TOLERANCE: f32 = 2e-3;

    fn loss(layer: &mut dyn Layer, x: &Matrix) -> f32 {
        softmax_cross_entropy(&layer.forward(x), &TARGETS).0
    }

    /// Measure dLoss/dx by nudging every input element. Returns (analytic, numerical).
    fn input_gradients(layer: &mut dyn Layer, x: &mut Matrix) -> (Vec<f32>, Vec<f32>) {
        layer.zero_grad();
        let (_, dlogits) = softmax_cross_entropy(&layer.forward(x), &TARGETS);
        let analytic = layer.backward(&dlogits).data;

        let mut numerical = vec![0.0; x.data.len()];
        for (idx, slot) in numerical.iter_mut().enumerate() {
            x.data[idx] += NUDGE;
            let up = loss(layer, x);
            x.data[idx] -= 2.0 * NUDGE;
            let down = loss(layer, x);
            x.data[idx] += NUDGE;
            *slot = (up - down) / (2.0 * NUDGE);
        }
        (analytic, numerical)
    }

    /// The same for one parameter vector, reached through `pick` so the layer
    /// is free to be borrowed again for the forward pass in between.
    fn param_gradient<L: Layer>(layer: &mut L, x: &Matrix, pick: fn(&mut L) -> &mut Vec<f32>) -> Vec<f32> {
        let mut numerical = vec![0.0; pick(layer).len()];
        for (idx, slot) in numerical.iter_mut().enumerate() {
            pick(layer)[idx] += NUDGE;
            let up = loss(layer, x);
            pick(layer)[idx] -= 2.0 * NUDGE;
            let down = loss(layer, x);
            pick(layer)[idx] += NUDGE;
            *slot = (up - down) / (2.0 * NUDGE);
        }
        numerical
    }

    #[test]
    fn layer_norm_gradient_matches_numerical() {
        let mut rng = Rng::new(11);
        let mut norm = LayerNorm::new(FEATURES);
        scramble(&mut norm.gamma, &mut rng);
        scramble(&mut norm.beta, &mut rng);
        let mut x = input(&mut rng);

        let (analytic, numerical) = input_gradients(&mut norm, &mut x);
        let rel = relative_error(&analytic, &numerical);
        assert!(rel < TOLERANCE, "dx: analytic and numerical gradients differ (rel {rel:.4})");

        // input_gradients left dgamma and dbeta filled in by its one backward.
        let (dgamma, dbeta) = (norm.dgamma.clone(), norm.dbeta.clone());
        let rel = relative_error(&dgamma, &param_gradient(&mut norm, &x, |n| &mut n.gamma));
        assert!(rel < TOLERANCE, "gamma: analytic and numerical gradients differ (rel {rel:.4})");
        let rel = relative_error(&dbeta, &param_gradient(&mut norm, &x, |n| &mut n.beta));
        assert!(rel < TOLERANCE, "beta: analytic and numerical gradients differ (rel {rel:.4})");
    }

    #[test]
    fn rms_norm_gradient_matches_numerical() {
        let mut rng = Rng::new(12);
        let mut norm = RmsNorm::new(FEATURES);
        scramble(&mut norm.gamma, &mut rng);
        let mut x = input(&mut rng);

        let (analytic, numerical) = input_gradients(&mut norm, &mut x);
        let rel = relative_error(&analytic, &numerical);
        assert!(rel < TOLERANCE, "dx: analytic and numerical gradients differ (rel {rel:.4})");

        let dgamma = norm.dgamma.clone();
        let rel = relative_error(&dgamma, &param_gradient(&mut norm, &x, |n| &mut n.gamma));
        assert!(rel < TOLERANCE, "gamma: analytic and numerical gradients differ (rel {rel:.4})");
    }

    /// The idea at the top of the file, checked directly rather than through a
    /// loss: the gradient has no component along a direction the layer ignores.
    #[test]
    fn gradient_is_blind_where_the_layer_is() {
        let mut rng = Rng::new(13);
        let x = input(&mut rng);
        let dy = Matrix::from_vec(ROWS, FEATURES, (0..ROWS * FEATURES).map(|_| rng.normal()).collect());
        // How much of `g` points along `direction`, from -1 to 1. A raw dot
        // product would do, but its rounding noise grows with the size of the
        // gradient; a cosine means the same thing at any scale.
        let along = |g: &[f32], direction: &[f32]| {
            let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
            dot(g, direction) / (dot(g, g) * dot(direction, direction)).sqrt()
        };
        let ones = [1.0; FEATURES];

        let mut layer_norm = LayerNorm::new(FEATURES);
        scramble(&mut layer_norm.gamma, &mut rng);
        layer_norm.forward(&x);
        let dx = layer_norm.backward(&dy);
        for r in 0..ROWS {
            let shift = along(dx.row(r), &ones);
            let stretch = along(dx.row(r), layer_norm.x_hat.row(r));
            assert!(shift.abs() < 1e-4, "row {r}: LayerNorm blamed a shift ({shift})");
            assert!(stretch.abs() < 1e-4, "row {r}: LayerNorm blamed a stretch ({stretch})");
        }

        let mut rms_norm = RmsNorm::new(FEATURES);
        scramble(&mut rms_norm.gamma, &mut rng);
        rms_norm.forward(&x);
        let dx = rms_norm.backward(&dy);
        for r in 0..ROWS {
            let shift = along(dx.row(r), &ones);
            let stretch = along(dx.row(r), rms_norm.x_hat.row(r));
            assert!(stretch.abs() < 1e-4, "row {r}: RmsNorm blamed a stretch ({stretch})");
            // RmsNorm can see a shift, and must say so.
            assert!(shift.abs() > 1e-2, "row {r}: RmsNorm ignored a shift ({shift})");
        }
    }

    #[test]
    fn layer_norm_output_is_normalised() {
        let mut rng = Rng::new(14);
        let y = LayerNorm::new(FEATURES).forward(&input(&mut rng));
        for r in 0..ROWS {
            let mean = y.row(r).iter().sum::<f32>() / FEATURES as f32;
            let var = y.row(r).iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / FEATURES as f32;
            assert!(mean.abs() < 1e-5, "row {r}: mean {mean}");
            assert!((var - 1.0).abs() < 1e-3, "row {r}: variance {var}");
        }
    }
}
