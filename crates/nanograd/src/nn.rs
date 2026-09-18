//! Layers, the loss, and the optimiser.
//!
//! # The one idea in this file
//!
//! A network is a chain of functions. Training needs dLoss/dParam for every
//! parameter, and the chain rule says you can get it by walking the chain
//! backwards, carrying one quantity with you: the gradient of the loss with
//! respect to the *current* layer's output.
//!
//! So every layer implements the same two-sided contract:
//!
//! * `forward(x) -> y`   — and stashes whatever it will need later.
//! * `backward(dy) -> dx` — given dLoss/dy, produce dLoss/dx, and on the way
//!                          accumulate dLoss/dW into its own gradient buffers.
//!
//! That is backpropagation, in full. Everything else — GPUs, autodiff graphs,
//! transformers — is engineering on top of these two lines.

use crate::matrix::Matrix;
use crate::rng::Rng;

pub trait Layer {
    fn forward(&mut self, x: &Matrix) -> Matrix;
    fn backward(&mut self, dy: &Matrix) -> Matrix;
    /// Apply accumulated gradients. Layers with no parameters do nothing.
    fn step(&mut self, _lr: f32, _momentum: f32) {}
    fn zero_grad(&mut self) {}
    fn describe(&self) -> String;

    /// Flat view of this layer's weights, if it has any. Used by the gradient
    /// check, and handy when you want to inspect or serialise a model.
    fn weights_mut(&mut self) -> Option<&mut [f32]> {
        None
    }
    /// Flat view of the gradients accumulated for those same weights.
    fn weight_grads(&self) -> Option<&[f32]> {
        None
    }
}

/// A fully connected layer: `y = x @ W + b`.
pub struct Linear {
    /// [in_features, out_features]
    w: Matrix,
    b: Vec<f32>,
    dw: Matrix,
    db: Vec<f32>,
    /// Momentum buffers ("velocity").
    vw: Matrix,
    vb: Vec<f32>,
    /// The input we were given, needed to compute dW = xᵀ @ dy.
    x: Matrix,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Self {
        // Kaiming/He initialisation: draw from N(0, 2/fan_in).
        //
        // Why this number: a ReLU zeroes roughly half its inputs, halving the
        // variance of the signal. Scaling by sqrt(2/fan_in) puts it back, so
        // activations neither explode nor decay to nothing as depth grows.
        // Initialisation is not a detail — get it wrong and a deep net simply
        // will not train.
        let std = (2.0 / in_features as f32).sqrt();
        let data = (0..in_features * out_features).map(|_| rng.normal() * std).collect();
        Linear {
            w: Matrix::from_vec(in_features, out_features, data),
            b: vec![0.0; out_features],
            dw: Matrix::zeros(in_features, out_features),
            db: vec![0.0; out_features],
            vw: Matrix::zeros(in_features, out_features),
            vb: vec![0.0; out_features],
            x: Matrix::zeros(0, in_features),
        }
    }

    pub fn param_count(&self) -> usize {
        self.w.data.len() + self.b.len()
    }
}

impl Layer for Linear {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        self.x = x.clone();
        let mut y = x.matmul(&self.w);
        for r in 0..y.rows {
            for (j, v) in y.row_mut(r).iter_mut().enumerate() {
                *v += self.b[j];
            }
        }
        y
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        // dW = xᵀ @ dy  — how much each weight contributed to the error.
        let dw = self.x.matmul_at_b(dy);
        for (acc, g) in self.dw.data.iter_mut().zip(dw.data.iter()) {
            *acc += g;
        }
        // db = sum of dy over the batch — the bias was added to every row.
        for r in 0..dy.rows {
            for (j, g) in dy.row(r).iter().enumerate() {
                self.db[j] += g;
            }
        }
        // dx = dy @ Wᵀ — the error signal handed to the layer below.
        dy.matmul_a_bt(&self.w)
    }

    fn step(&mut self, lr: f32, momentum: f32) {
        for i in 0..self.w.data.len() {
            self.vw.data[i] = momentum * self.vw.data[i] - lr * self.dw.data[i];
            self.w.data[i] += self.vw.data[i];
        }
        for i in 0..self.b.len() {
            self.vb[i] = momentum * self.vb[i] - lr * self.db[i];
            self.b[i] += self.vb[i];
        }
    }

    fn zero_grad(&mut self) {
        self.dw.fill(0.0);
        self.db.iter_mut().for_each(|v| *v = 0.0);
    }

    fn describe(&self) -> String {
        format!("Linear({} -> {}, {} params)", self.w.rows, self.w.cols, self.param_count())
    }

    fn weights_mut(&mut self) -> Option<&mut [f32]> {
        Some(&mut self.w.data)
    }

    fn weight_grads(&self) -> Option<&[f32]> {
        Some(&self.dw.data)
    }
}

/// `y = max(0, x)`.
///
/// The derivative is 1 where the input was positive and 0 where it was not,
/// so backward is just a mask. Nonlinearity is the entire point: stack two
/// Linear layers with nothing between them and you get... one Linear layer.
#[derive(Default)]
pub struct Relu {
    positive: Vec<bool>,
}

impl Layer for Relu {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        self.positive = x.data.iter().map(|&v| v > 0.0).collect();
        let data = x.data.iter().map(|&v| v.max(0.0)).collect();
        Matrix::from_vec(x.rows, x.cols, data)
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        let data = dy
            .data
            .iter()
            .zip(self.positive.iter())
            .map(|(&g, &p)| if p { g } else { 0.0 })
            .collect();
        Matrix::from_vec(dy.rows, dy.cols, data)
    }

    fn describe(&self) -> String {
        "ReLU".to_string()
    }
}

/// Softmax followed by cross-entropy loss, fused into one function.
///
/// Returns `(mean_loss, dLoss/dLogits)`.
///
/// Fusing them is not just an optimisation. Differentiating softmax and
/// cross-entropy separately gives you a Jacobian and a division that very
/// nearly cancel; compose them first and the gradient collapses to
/// `(probability - target) / batch_size`. Predicted 0.9 for the right class?
/// Gradient -0.1. That is the whole signal.
pub fn softmax_cross_entropy(logits: &Matrix, targets: &[usize]) -> (f32, Matrix) {
    assert_eq!(logits.rows, targets.len());
    let n = logits.rows;
    let mut grad = Matrix::zeros(logits.rows, logits.cols);
    let mut total = 0.0;

    for r in 0..n {
        let row = logits.row(r);
        // Subtract the max before exponentiating. exp(800) overflows to inf;
        // exp(800 - 800) does not. The result is mathematically identical.
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for &v in row {
            sum += (v - max).exp();
        }
        let log_sum = sum.ln();

        // loss = -log p[target] = -(logit[target] - max - log_sum)
        total += -(row[targets[r]] - max - log_sum);

        let g = grad.row_mut(r);
        for (j, &v) in row.iter().enumerate() {
            g[j] = (v - max).exp() / sum;
        }
        g[targets[r]] -= 1.0;
        // Average over the batch here, once, so every downstream gradient is
        // already a per-example mean.
        for v in g.iter_mut() {
            *v /= n as f32;
        }
    }

    (total / n as f32, grad)
}

/// A stack of layers applied in order.
pub struct Mlp {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Mlp {
    pub fn new(sizes: &[usize], rng: &mut Rng) -> Self {
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Box::new(Linear::new(sizes[i], sizes[i + 1], rng)));
            // No activation after the final layer: it produces logits, and
            // softmax_cross_entropy handles the rest.
            if i + 2 < sizes.len() {
                layers.push(Box::new(Relu::default()));
            }
        }
        Mlp { layers }
    }

    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for layer in self.layers.iter_mut() {
            out = layer.forward(&out);
        }
        out
    }

    /// Walk the layers in reverse, threading dLoss/dOutput back to the input.
    pub fn backward(&mut self, dy: &Matrix) {
        let mut grad = dy.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    pub fn step(&mut self, lr: f32, momentum: f32) {
        for layer in self.layers.iter_mut() {
            layer.step(lr, momentum);
        }
    }

    pub fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }

    pub fn summary(&self) -> String {
        self.layers.iter().map(|l| l.describe()).collect::<Vec<_>>().join("\n  ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_cross_entropy_is_sane() {
        // A confident correct prediction should have near-zero loss.
        let logits = Matrix::from_vec(1, 3, vec![10.0, 0.0, 0.0]);
        let (loss, grad) = softmax_cross_entropy(&logits, &[0]);
        assert!(loss < 1e-3, "loss was {loss}");
        assert!(grad.data[0] < 0.0, "gradient should push the correct logit up");

        // A uniform prediction over 3 classes should cost ln(3).
        let logits = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]);
        let (loss, _) = softmax_cross_entropy(&logits, &[1]);
        assert!((loss - 3f32.ln()).abs() < 1e-5, "loss was {loss}");
    }

    /// The test that matters.
    ///
    /// Nudge one weight by ±eps, measure how the loss actually changes, and
    /// compare against what backward() claimed the gradient was. If your
    /// backprop has a sign error or a missing transpose, this catches it —
    /// and nothing else will, because a subtly wrong gradient still trains,
    /// just badly.
    #[test]
    fn analytic_gradient_matches_numerical() {
        let mut rng = Rng::new(42);
        let mut net = Mlp::new(&[4, 6, 3], &mut rng);
        let x = Matrix::from_vec(
            2,
            4,
            (0..8).map(|_| rng.normal()).collect::<Vec<f32>>(),
        );
        let targets = [2usize, 0];

        let loss_at = |net: &mut Mlp| {
            let logits = net.forward(&x);
            softmax_cross_entropy(&logits, &targets).0
        };

        net.zero_grad();
        let logits = net.forward(&x);
        let (_, dlogits) = softmax_cross_entropy(&logits, &targets);
        net.backward(&dlogits);

        // Probe a handful of weights in the first Linear layer.
        let eps = 1e-3;
        for idx in [0usize, 5, 11, 17, 23] {
            let analytic = net.layers[0].weight_grads().unwrap()[idx];

            let nudge = |net: &mut Mlp, d: f32| {
                net.layers[0].weights_mut().unwrap()[idx] += d;
            };

            nudge(&mut net, eps);
            let up = loss_at(&mut net);
            nudge(&mut net, -2.0 * eps);
            let down = loss_at(&mut net);
            nudge(&mut net, eps); // restore

            // The centred difference: (f(w+e) - f(w-e)) / 2e. This is what the
            // gradient *means* — how much the loss moves when this one weight
            // moves — computed without any calculus at all.
            let numerical = (up - down) / (2.0 * eps);
            let denom = analytic.abs().max(numerical.abs()).max(1e-6);
            let rel = (analytic - numerical).abs() / denom;
            assert!(
                rel < 2e-2,
                "weight {idx}: analytic {analytic:.6} vs numerical {numerical:.6} (rel {rel:.4})"
            );
        }
    }
}
