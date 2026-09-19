//! AdamW: the optimiser transformers are actually trained with.
//!
//! # What is wrong with SGD
//!
//! SGD moves every parameter by `lr * gradient`, with one `lr` for the whole
//! model. That is fine when every gradient is about the same size, as in the
//! MNIST network. In a transformer they are not remotely: an embedding row
//! gets a gradient only when its token shows up, a norm's gain sees every
//! position at once, and the tensors differ by orders of magnitude. Any single
//! `lr` is too big for the steepest parameter or too small for the rest — and
//! it has to be set for the steepest, or training blows up.
//!
//! # The one idea in this file
//!
//! Adam gives every parameter its own step size, by dividing its gradient by
//! how big that parameter's gradients *usually* are:
//!
//! ```text
//! m = running average of g          which way, lately?     (momentum)
//! v = running average of g*g        how big, lately?
//! w -= lr * m / sqrt(v)
//! ```
//!
//! Look at what the division does. Multiply every gradient by 1000 and `m`
//! grows 1000-fold, `sqrt(v)` grows 1000-fold, and the step does not change at
//! all. The *size* of the gradient has been cancelled out; what is left is its
//! sign and its consistency. A gradient that always points the same way gives
//! `m / sqrt(v)` near ±1, and a step of about `lr`. One that flips sign every
//! step averages to `m` near 0, and the parameter barely moves. So `lr` stops
//! meaning "a multiplier on the gradient" and becomes "about how far a
//! parameter may move per step", which is a thing you can reason about.
//!
//! # Two corrections
//!
//! **Bias correction.** `m` and `v` start at zero, so for the first several
//! steps they are averages of mostly zeros, and far too small. After `t` steps
//! an average with decay `beta` has taken in only `1 - beta^t` of its weight,
//! so dividing by that repairs it exactly. With it, the very first step is
//! `lr * g / |g|`: every parameter moves by `lr`, in the direction of its
//! sign. The tests check that to the digit.
//!
//! **Decoupled weight decay** — the W. Weight decay shrinks weights a little
//! each step, so that a weight has to keep earning its size. The classic way
//! is to add `decay * w` to the gradient. Under Adam that goes through the
//! division like everything else, and gets cancelled like everything else:
//! the parameters with the largest gradients end up decayed the least. AdamW
//! takes the decay out of the gradient and applies it to the weight directly.
//! Only to matrices, though — a bias is not where overfitting lives, and a
//! norm's gain belongs at 1, not 0.

use crate::nn::Param;

pub struct AdamW {
    pub lr: f32,
    /// Decay of the running average of gradients. 0.9 is "the last ten or so".
    pub beta1: f32,
    /// Decay of the running average of squared gradients: the last thousand.
    pub beta2: f32,
    /// Keeps the division finite for a parameter whose gradient is zero.
    pub eps: f32,
    pub weight_decay: f32,
    /// Steps taken so far, for the bias correction.
    t: i32,
    /// One `m` and one `v` per parameter tensor, in the order `params()` gives
    /// them. This is the memory cost of Adam: two more floats for every weight.
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        AdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01, t: 0, m: Vec::new(), v: Vec::new() }
    }

    /// Apply the gradients in `params` to the values in `params`.
    ///
    /// Pass the same model's `params()` every time: the running averages are
    /// matched to tensors by position.
    pub fn step(&mut self, params: Vec<Param<'_>>) {
        if self.m.is_empty() {
            self.m = params.iter().map(|p| vec![0.0; p.value.len()]).collect();
            self.v = self.m.clone();
        }
        assert_eq!(params.len(), self.m.len(), "the optimiser was given a different model");

        self.t += 1;
        let m_correction = 1.0 - self.beta1.powi(self.t);
        let v_correction = 1.0 - self.beta2.powi(self.t);

        for ((p, m), v) in params.into_iter().zip(self.m.iter_mut()).zip(self.v.iter_mut()) {
            assert_eq!(p.value.len(), m.len(), "{} changed size", p.name);
            let decay = if is_matrix(&p.name) { self.lr * self.weight_decay } else { 0.0 };

            for i in 0..p.value.len() {
                let g = p.grad[i];
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

                let m_hat = m[i] / m_correction;
                let v_hat = v[i] / v_correction;

                // The decay first, on the weight as it was; then the step.
                p.value[i] -= decay * p.value[i];
                p.value[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

/// Weight matrices and embedding tables are decayed; biases and gains are not.
fn is_matrix(name: &str) -> bool {
    name.ends_with("weight") || name.ends_with("table")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Gpt, GptConfig};
    use crate::nn::{sgd, softmax_cross_entropy};
    use crate::rng::Rng;

    /// A stand-in for a model: some values, and whatever gradients we say.
    struct Toy {
        names: Vec<&'static str>,
        values: Vec<Vec<f32>>,
        grads: Vec<Vec<f32>>,
    }

    impl Toy {
        fn new(names: &[&'static str], values: &[f32]) -> Self {
            Toy {
                names: names.to_vec(),
                values: names.iter().map(|_| values.to_vec()).collect(),
                grads: names.iter().map(|_| vec![0.0; values.len()]).collect(),
            }
        }

        fn params(&mut self) -> Vec<Param<'_>> {
            let tensors = self.values.iter_mut().zip(self.grads.iter_mut());
            self.names.iter().zip(tensors).map(|(n, (v, g))| Param::new(n, v, g)).collect()
        }
    }

    const GRADS: [f32; 4] = [3.0, -0.002, 500.0, -1e-5];

    fn no_decay(lr: f32) -> AdamW {
        AdamW { weight_decay: 0.0, ..AdamW::new(lr) }
    }

    /// With bias correction, step one is `lr * g / (|g| + eps)`: gradients
    /// spanning eight orders of magnitude all move their parameter by `lr`.
    /// Without it the step would be `lr * 0.1 / sqrt(0.001)`, over three times
    /// too far.
    ///
    /// Not *quite* `lr` for the smallest one, and the shortfall is what `eps`
    /// means. At 1e-5 that gradient is only a thousand times `eps`, so it
    /// moves 0.1% short. `eps` is the size below which Adam stops amplifying a
    /// gradient and starts treating it as nothing — without it, a parameter
    /// whose gradient is rounding noise would be marched about at full speed.
    /// This crate has such a parameter: attention's key bias, whose true
    /// gradient is zero. Over the 100 steps of `it_trains_the_gpt` it drifts
    /// 0.00003 from where it started, while the biases either side of it
    /// travel 0.015 to 0.058.
    #[test]
    fn the_first_step_is_lr_in_the_direction_of_the_sign() {
        let mut toy = Toy::new(&["w.bias"], &[0.0; 4]);
        toy.grads[0] = GRADS.to_vec();
        let mut opt = no_decay(0.01);
        opt.step(toy.params());

        for (&moved, g) in toy.values[0].iter().zip(GRADS) {
            let exact = -0.01 * g / (g.abs() + opt.eps);
            assert!((moved - exact).abs() < 1e-7, "gradient {g} moved its weight by {moved}, not {exact}");
            assert!((moved + 0.01 * g.signum()).abs() < 2e-5, "gradient {g} moved its weight by {moved}");
        }
    }

    /// The same holds for as long as the gradient stays put: a constant
    /// gradient makes both running averages exact, so every step is `lr`.
    #[test]
    fn a_steady_gradient_moves_lr_per_step_whatever_its_size() {
        let mut toy = Toy::new(&["w.bias"], &[0.0; 4]);
        toy.grads[0] = GRADS.to_vec();
        let mut opt = no_decay(0.01);
        for _ in 0..50 {
            opt.step(toy.params());
        }
        for (w, g) in toy.values[0].iter().zip(GRADS) {
            // Exactly, `eps` and all: measured, these agree to 5e-6.
            let exact = -50.0 * 0.01 * g / (g.abs() + opt.eps);
            assert!((w - exact).abs() < 5e-5, "gradient {g} ended at {w}, not {exact}");
        }
    }

    /// The idea at the top of the file: scale every gradient by 1000 and the
    /// trajectory does not change. Under SGD it would be 1000 times longer.
    #[test]
    fn the_size_of_the_gradient_cancels_out() {
        let mut rng = Rng::new(61);
        let noisy: Vec<Vec<f32>> = (0..20).map(|_| (0..4).map(|_| rng.normal() + 0.3).collect()).collect();

        let run = |scale: f32| {
            let mut toy = Toy::new(&["w.bias"], &[0.0; 4]);
            let mut opt = no_decay(0.01);
            for g in &noisy {
                toy.grads[0] = g.iter().map(|g| g * scale).collect();
                opt.step(toy.params());
            }
            toy.values[0].clone()
        };

        for (small, large) in run(1.0).iter().zip(run(1000.0)) {
            assert!((small - large).abs() < 1e-5, "{small} at scale 1, {large} at scale 1000");
        }
    }

    /// And a gradient that cannot make up its mind goes nowhere, however large.
    ///
    /// Nowhere *much*: it ends 0.0176 from the start, under two steps' worth.
    /// That much is built in — bias correction makes the first step a full
    /// `lr` whatever happens next, and the averages take a few steps to notice
    /// the gradient is not going anywhere.
    #[test]
    fn a_gradient_that_keeps_changing_sign_barely_moves() {
        let mut toy = Toy::new(&["w.bias"], &[0.0]);
        let mut opt = no_decay(0.01);
        for step in 0..200 {
            toy.grads[0] = vec![if step % 2 == 0 { 100.0 } else { -100.0 }];
            opt.step(toy.params());
        }
        // 200 steps of lr = 0.01 could have covered 2.0.
        assert!(toy.values[0][0].abs() < 0.05, "ended at {}", toy.values[0][0]);
    }

    /// With no gradient at all, the only thing acting is the decay: matrices
    /// shrink by `1 - lr * weight_decay` a step, and nothing else is touched.
    #[test]
    fn decay_shrinks_matrices_and_leaves_the_rest() {
        let mut toy = Toy::new(&["head.weight", "tokens.table", "head.bias", "norm.gamma"], &[2.0, -1.0]);
        let mut opt = AdamW { weight_decay: 0.1, ..AdamW::new(0.5) };
        for _ in 0..3 {
            opt.step(toy.params());
        }
        let kept = (1.0f32 - 0.5 * 0.1).powi(3);
        for matrix in [0, 1] {
            assert!((toy.values[matrix][0] - 2.0 * kept).abs() < 1e-6);
            assert!((toy.values[matrix][1] + 1.0 * kept).abs() < 1e-6);
        }
        for other in [2, 3] {
            assert_eq!(toy.values[other], vec![2.0, -1.0], "{} was decayed", toy.names[other]);
        }
    }

    /// Why any of this matters. A valley a million times steeper one way than
    /// the other: `loss = (1000 x^2 + 0.001 y^2) / 2`, starting at (1, 1).
    ///
    /// SGD's `lr` must stay under 2/1000 or x diverges, and at that `lr` each
    /// step moves y by two parts in a million. Adam moves both at the same
    /// speed, because it never sees the steepness.
    #[test]
    fn adam_does_not_care_how_steep_each_direction_is() {
        const STEPS: usize = 300;
        let grad = |w: &[f32]| vec![1000.0 * w[0], 0.001 * w[1]];

        let mut adam = Toy::new(&["w.bias"], &[1.0, 1.0]);
        let mut opt = no_decay(0.01);
        for _ in 0..STEPS {
            adam.grads[0] = grad(&adam.values[0]);
            opt.step(adam.params());
        }

        let mut plain = vec![1.0, 1.0];
        let mut velocity = vec![0.0, 0.0];
        for _ in 0..STEPS {
            let g = grad(&plain);
            sgd(&mut plain, &g, &mut velocity, 0.0019, 0.0);
        }

        let [x, y] = adam.values[0][..] else { unreachable!() };
        assert!(x.abs() < 0.05 && y.abs() < 0.05, "Adam ended at ({x}, {y})");
        assert!(plain[1] > 0.999, "SGD got y to {}", plain[1]);
    }

    /// The optimiser and the model, together: the same memorisation check the
    /// model passes with SGD.
    #[test]
    fn it_trains_the_gpt() {
        const IDS: [usize; 5] = [4, 9, 4, 0, 7];
        const NEXT: [usize; 5] = [9, 4, 0, 7, 2];
        let config = GptConfig { vocab: 11, context: 6, d_model: 8, n_heads: 2, n_layers: 2 };
        let mut model = Gpt::new(config, &mut Rng::new(62));
        let mut opt = AdamW::new(LR);

        let initial = softmax_cross_entropy(&model.forward(&IDS), &NEXT).0;
        for _ in 0..STEPS {
            model.zero_grad();
            let (_, dlogits) = softmax_cross_entropy(&model.forward(&IDS), &NEXT);
            model.backward(&dlogits);
            opt.step(model.params());
        }
        let trained = softmax_cross_entropy(&model.forward(&IDS), &NEXT).0;
        assert!(trained < 0.05, "loss went from {initial} to only {trained}");
    }

    const STEPS: usize = 100;
    const LR: f32 = 0.01;
}
