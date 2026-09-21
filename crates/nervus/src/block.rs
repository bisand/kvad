//! The residual connection, and the transformer block built from it.
//!
//! # The one idea in this file
//!
//! Every layer so far *replaces* its input: `y = f(x)`. A residual connection
//! *adds to* it instead:
//!
//! ```text
//! y = x + f(x)
//! ```
//!
//! That one `+` is why networks can be a hundred layers deep. Differentiate it
//! and the derivative is `1 + f'(x)`, so the backward pass is
//!
//! ```text
//! dx = dy + f.backward(dy)
//! ```
//!
//! The first term is the point. Whatever `f` does to the gradient — shrinks
//! it, scrambles it, zeroes it — `dy` also arrives at the layer below
//! *untouched*. Stack fifty of these and the loss's gradient still reaches the
//! bottom at full strength, along an unbroken chain of additions usually
//! called the residual stream. Without it, the gradient is a product of fifty
//! Jacobians, and a product of fifty numbers is either nothing or infinity.
//!
//! There is no new calculus here. `x` is used in two places, so its gradients
//! add — the rule attention already used for its three projections.
//!
//! # The block
//!
//! A transformer block is two of these in a row:
//!
//! ```text
//! x = x + attention(norm(x))      positions exchange information
//! x = x + mlp(norm(x))            each position thinks about what it received
//! ```
//!
//! Note where the norm sits: *inside* the branch. The stream itself is never
//! normalised, so the untouched path stays untouched from the last block to
//! the first. The original Transformer normalised after the addition, which
//! puts a norm on the highway; it needed careful learning-rate warm-up to
//! train at all, and everything since GPT-2 does it this way round.

use crate::attention::CausalSelfAttention;
use crate::matrix::Matrix;
use crate::nn::{prefixed, Gelu, Layer, Linear, Param};
use crate::norm::LayerNorm;
use crate::rng::Rng;

/// `y = x + branch(x)`, where the branch is a chain of layers that ends at the
/// same shape it started from.
pub struct Residual {
    branch: Vec<Box<dyn Layer>>,
}

impl Residual {
    pub fn new(branch: Vec<Box<dyn Layer>>) -> Self {
        Residual { branch }
    }
}

impl Layer for Residual {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for layer in self.branch.iter_mut() {
            out = layer.forward(&out);
        }
        out.add_in_place(x);
        out
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        let mut dx = dy.clone();
        for layer in self.branch.iter_mut().rev() {
            dx = layer.backward(&dx);
        }
        // The highway: dy reaches the layer below whatever the branch did.
        dx.add_in_place(dy);
        dx
    }

    fn step(&mut self, lr: f32, momentum: f32) {
        self.branch.iter_mut().for_each(|l| l.step(lr, momentum));
    }

    fn zero_grad(&mut self) {
        self.branch.iter_mut().for_each(|l| l.zero_grad());
    }

    fn params(&mut self) -> Vec<Param<'_>> {
        let mut all = Vec::new();
        for (i, layer) in self.branch.iter_mut().enumerate() {
            all.extend(prefixed(&i.to_string(), layer.params()));
        }
        all
    }

    fn describe(&self) -> String {
        let branch: Vec<String> = self.branch.iter().map(|l| l.describe()).collect();
        format!("x + [{}]", branch.join(" -> "))
    }
}

/// One pre-norm transformer block: `[seq, d_model]` in, the same shape out.
pub struct Block {
    attn: Residual,
    mlp: Residual,
}

impl Block {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        // The MLP widens to 4x and comes back. Attention only ever *averages*
        // value vectors; this is where a position gets to compute something
        // from what it gathered, and most of a transformer's parameters live
        // here. GELU, as in GPT-2; see `Gelu` for why not ReLU.
        let hidden = 4 * d_model;
        Block {
            attn: Residual::new(vec![
                Box::new(LayerNorm::new(d_model)),
                Box::new(CausalSelfAttention::new(d_model, n_heads, rng)),
            ]),
            mlp: Residual::new(vec![
                Box::new(LayerNorm::new(d_model)),
                Box::new(Linear::new(d_model, hidden, rng)),
                Box::new(Gelu::default()),
                Box::new(Linear::new(hidden, d_model, rng)),
            ]),
        }
    }
}

impl Layer for Block {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let x = self.attn.forward(x);
        self.mlp.forward(&x)
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        let dy = self.mlp.backward(dy);
        self.attn.backward(&dy)
    }

    fn step(&mut self, lr: f32, momentum: f32) {
        self.attn.step(lr, momentum);
        self.mlp.step(lr, momentum);
    }

    fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.mlp.zero_grad();
    }

    fn params(&mut self) -> Vec<Param<'_>> {
        let mut all = prefixed("attn", self.attn.params());
        all.extend(prefixed("mlp", self.mlp.params()));
        all
    }

    fn describe(&self) -> String {
        format!("Block\n    {}\n    {}", self.attn.describe(), self.mlp.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{check_layer, scramble};

    const SEQ: usize = 5;
    const D_MODEL: usize = 8;
    const TARGETS: [usize; SEQ] = [3, 0, 7, 1, 4];

    fn random(rows: usize, cols: usize, rng: &mut Rng) -> Matrix {
        Matrix::from_vec(rows, cols, (0..rows * cols).map(|_| rng.normal()).collect())
    }

    #[test]
    fn analytic_gradient_matches_numerical() {
        let mut rng = Rng::new(31);
        let mut block = Block::new(D_MODEL, 2, &mut rng);
        scramble(block.params(), &mut rng);
        let x = random(SEQ, D_MODEL, &mut rng);

        let report = check_layer(&mut block, &x, &TARGETS, NUDGE);
        // Two norms and two Linears at 2 tensors each, attention's 8, and dx.
        assert_eq!(report.len(), 4 + 4 + 8 + 1);
        for c in report {
            if c.name == "attn.1.wk.bias" {
                // Exactly zero by construction — see the attention tests — so
                // there is no ratio to take. Both must simply be nothing.
                assert!(c.analytic_norm < 1e-6, "key bias gradient {:e}", c.analytic_norm);
                assert!(c.numerical_norm < 1e-4, "key bias numerical gradient {:e}", c.numerical_norm);
                continue;
            }
            assert!(c.rel < TOLERANCE, "{}: analytic and numerical gradients differ (rel {:.4})", c.name, c.rel);
        }
    }

    const NUDGE: f32 = 1e-2;
    const TOLERANCE: f32 = 2e-3;

    /// The highway, isolated. Silence the branch and a residual layer is the
    /// identity in both directions — exactly, since adding 0.0 changes no bits.
    #[test]
    fn a_silent_branch_leaves_the_identity() {
        let mut rng = Rng::new(32);
        let mut silent = Linear::new(D_MODEL, D_MODEL, &mut rng);
        silent.params().into_iter().for_each(|p| p.value.fill(0.0));
        let mut residual = Residual::new(vec![Box::new(silent)]);

        let x = random(SEQ, D_MODEL, &mut rng);
        let dy = random(SEQ, D_MODEL, &mut rng);
        assert_eq!(residual.forward(&x), x);
        assert_eq!(residual.backward(&dy), dy);
    }

    /// Why that matters. Stack blocks whose branches *destroy* the gradient,
    /// and it still reaches the bottom; chain the same branches without the
    /// `x +` and nothing does.
    #[test]
    fn the_gradient_survives_depth_only_with_the_highway() {
        const DEPTH: usize = 24;
        let mut rng = Rng::new(33);
        // A Linear layer scaled down to a tenth: each one passes back about a
        // tenth of the gradient it is given.
        let weak = |rng: &mut Rng| -> Box<dyn Layer> {
            let mut l = Linear::new(D_MODEL, D_MODEL, rng);
            l.params().into_iter().for_each(|p| p.value.iter_mut().for_each(|v| *v *= 0.1));
            Box::new(l)
        };
        let mut plain: Vec<Box<dyn Layer>> = (0..DEPTH).map(|_| weak(&mut rng)).collect();
        let mut residual: Vec<Box<dyn Layer>> =
            (0..DEPTH).map(|_| Box::new(Residual::new(vec![weak(&mut rng)])) as Box<dyn Layer>).collect();

        let x = random(SEQ, D_MODEL, &mut rng);
        let dy = random(SEQ, D_MODEL, &mut rng);
        let size = |m: &Matrix| m.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        let through = |stack: &mut Vec<Box<dyn Layer>>| {
            let mut h = x.clone();
            for layer in stack.iter_mut() {
                h = layer.forward(&h);
            }
            let mut g = dy.clone();
            for layer in stack.iter_mut().rev() {
                g = layer.backward(&g);
            }
            size(&g) / size(&dy)
        };

        let plain = through(&mut plain);
        let residual = through(&mut residual);
        assert!(plain < 1e-12, "24 weak layers in a chain passed back {plain:e} of the gradient");
        assert!((0.1..10.0).contains(&residual), "24 residual layers passed back {residual:e}");
    }
}
