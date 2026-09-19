//! Causal self-attention, forward and backward.
//!
//! # What is new here, and what is not
//!
//! Attention looks like a different kind of object from the layers in [`nn`],
//! but take it apart and almost none of it is new:
//!
//! ```text
//! Q = x @ Wq      K = x @ Wk      V = x @ Wv        three Linear layers
//! S = Q @ Kᵀ / sqrt(head_dim)                       a matrix product
//! P = softmax of each row of S, future masked out   <- the only new function
//! out = (P @ V) @ Wo                                a product, and a Linear
//! ```
//!
//! [`nn`] already knows how to go backwards through a `Linear`, and [`matrix`]
//! already has the three products a matmul needs to be differentiated. The one
//! derivative this file has to supply is the softmax's — see
//! `softmax_backward` below. Everything else is those same pieces run in reverse.
//!
//! The rows of `x` mean something different here. In the MNIST network each
//! row was an independent example. Here each row is a *position in one
//! sequence*, and attention is the only layer in a transformer where rows talk
//! to each other. `P[i][j]` is how much position `i` reads from position `j`.
//!
//! One sequence per call. A batch of sequences is a loop around this: the
//! gradient buffers accumulate until `zero_grad`, which is what they were
//! for all along.
//!
//! [`nn`]: crate::nn
//! [`matrix`]: crate::matrix

use crate::matrix::Matrix;
use crate::nn::{Layer, Linear};
use crate::rng::Rng;

/// Multi-head causal self-attention over one sequence: `[seq, d_model]` in,
/// `[seq, d_model]` out.
pub struct CausalSelfAttention {
    n_heads: usize,
    head_dim: usize,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    /// What each head computed on the way forward, kept for the way back.
    heads: Vec<HeadCache>,
}

/// One head's share of the forward pass.
///
/// `probs` is `[seq, seq]`, per head, per layer — the quadratic memory cost of
/// attention, sitting right here in a struct. Inference throws it away at
/// once; training has to keep it until backward, and avoiding exactly that is
/// what FlashAttention is for.
struct HeadCache {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    probs: Matrix,
}

impl CausalSelfAttention {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model {d_model} does not split into {n_heads} heads");
        CausalSelfAttention {
            n_heads,
            head_dim: d_model / n_heads,
            wq: Linear::new(d_model, d_model, rng),
            wk: Linear::new(d_model, d_model, rng),
            wv: Linear::new(d_model, d_model, rng),
            wo: Linear::new(d_model, d_model, rng),
            heads: Vec::new(),
        }
    }

    pub fn param_count(&self) -> usize {
        self.wq.param_count() + self.wk.param_count() + self.wv.param_count() + self.wo.param_count()
    }

    /// Dividing by sqrt(head_dim) keeps the scores' variance near 1 however
    /// wide the head is. A dot product of `n` unit-variance terms has variance
    /// `n`; unscaled, a wide head produces huge scores, the softmax saturates
    /// to one-hot, and a saturated softmax has almost no gradient.
    fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

impl Layer for CausalSelfAttention {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        let scale = self.scale();
        let mut mixed = Matrix::zeros(x.rows, x.cols);
        self.heads.clear();
        for h in 0..self.n_heads {
            // A head is nothing more than a slice of the columns. The heads
            // never interact until Wo mixes them back together.
            let qh = take_head(&q, h, self.head_dim);
            let kh = take_head(&k, h, self.head_dim);
            let vh = take_head(&v, h, self.head_dim);

            // scores[i][j] = q_i · k_j — how well position i's question
            // matches position j's label.
            let mut probs = qh.matmul_a_bt(&kh);
            probs.data.iter_mut().for_each(|s| *s *= scale);
            causal_softmax_rows(&mut probs);

            // Each output row is a weighted average of the value rows.
            let out = probs.matmul(&vh);
            put_head(&mut mixed, h, &out);

            self.heads.push(HeadCache { q: qh, k: kh, v: vh, probs });
        }

        self.wo.forward(&mixed)
    }

    fn backward(&mut self, dy: &Matrix) -> Matrix {
        // Forward, read bottom to top.
        let dmixed = self.wo.backward(dy);

        let scale = self.scale();
        let mut dq = Matrix::zeros(dy.rows, dy.cols);
        let mut dk = Matrix::zeros(dy.rows, dy.cols);
        let mut dv = Matrix::zeros(dy.rows, dy.cols);
        for (h, head) in self.heads.iter().enumerate() {
            let dout = take_head(&dmixed, h, self.head_dim);

            // out = P @ V. The same two rules as Linear, where P plays the
            // input and V plays the weight: dV = Pᵀ @ dout, dP = dout @ Vᵀ.
            let dvh = head.probs.matmul_at_b(&dout);
            let dprobs = dout.matmul_a_bt(&head.v);

            // P = softmax(S), then S = (Q @ Kᵀ) * scale.
            let mut dscores = softmax_backward(&head.probs, &dprobs);
            dscores.data.iter_mut().for_each(|g| *g *= scale);

            // S = Q @ Kᵀ is a matmul whose "weight" is Kᵀ, so the rules come
            // out transposed: dQ = dS @ K, dK = dSᵀ @ Q.
            let dqh = dscores.matmul(&head.k);
            let dkh = dscores.matmul_at_b(&head.q);

            put_head(&mut dq, h, &dqh);
            put_head(&mut dk, h, &dkh);
            put_head(&mut dv, h, &dvh);
        }

        // x fed three projections, so it is to blame through all three. When a
        // value is used in several places, its gradients add.
        let mut dx = self.wq.backward(&dq);
        add_into(&mut dx, &self.wk.backward(&dk));
        add_into(&mut dx, &self.wv.backward(&dv));
        dx
    }

    fn step(&mut self, lr: f32, momentum: f32) {
        self.wq.step(lr, momentum);
        self.wk.step(lr, momentum);
        self.wv.step(lr, momentum);
        self.wo.step(lr, momentum);
    }

    fn zero_grad(&mut self) {
        self.wq.zero_grad();
        self.wk.zero_grad();
        self.wv.zero_grad();
        self.wo.zero_grad();
    }

    fn describe(&self) -> String {
        format!(
            "CausalSelfAttention({} heads x {}, {} params)",
            self.n_heads,
            self.head_dim,
            self.param_count()
        )
    }
}

/// Softmax each row in place, with row `i` allowed to see only columns `0..=i`.
///
/// The usual description of the causal mask is "set the future scores to -inf
/// before the softmax". Leaving them out of the sum is the same thing, since
/// exp(-inf) = 0, and needs no infinities.
fn causal_softmax_rows(scores: &mut Matrix) {
    for i in 0..scores.rows {
        let row = scores.row_mut(i);
        let (seen, future) = row.split_at_mut(i + 1);

        // Subtract the max first, for the same reason as in the loss.
        let max = seen.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for s in seen.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        seen.iter_mut().for_each(|s| *s /= sum);
        future.fill(0.0);
    }
}

/// Given `P = softmax(S)` row by row, turn dLoss/dP into dLoss/dS.
///
/// In the loss, softmax was fused with cross-entropy and its derivative
/// cancelled away. Here it stands alone, so this is the real thing.
///
/// Raising one score `s_k` raises `p_k` and, because the row must still sum to
/// 1, lowers every other `p_j` in proportion to its size:
///
/// ```text
/// dp_j/ds_k = p_j * ([j == k] - p_k)
/// ```
///
/// Sum that over `j`, weighted by the incoming gradient, and it collapses to
///
/// ```text
/// ds_k = p_k * (dp_k - sum_j(dp_j * p_j))
/// ```
///
/// Read it as: take each probability's gradient *relative to the row's
/// average gradient*. If every `dp_j` is the same, nothing can improve by
/// moving probability around, and `ds` is zero.
///
/// The causal mask needs no code here. A masked entry has `p = 0`, so it
/// contributes nothing to the sum and receives `ds = 0`.
fn softmax_backward(probs: &Matrix, dprobs: &Matrix) -> Matrix {
    let mut dscores = Matrix::zeros(probs.rows, probs.cols);
    for r in 0..probs.rows {
        let p = probs.row(r);
        let dp = dprobs.row(r);
        let avg: f32 = p.iter().zip(dp).map(|(p, dp)| p * dp).sum();
        for (j, ds) in dscores.row_mut(r).iter_mut().enumerate() {
            *ds = p[j] * (dp[j] - avg);
        }
    }
    dscores
}

/// Copy head `h`'s columns out of a `[seq, d_model]` matrix.
fn take_head(m: &Matrix, h: usize, head_dim: usize) -> Matrix {
    let mut out = Matrix::zeros(m.rows, head_dim);
    for r in 0..m.rows {
        out.row_mut(r).copy_from_slice(&m.row(r)[h * head_dim..(h + 1) * head_dim]);
    }
    out
}

/// The inverse of [`take_head`]: write `[seq, head_dim]` back into its columns.
fn put_head(m: &mut Matrix, h: usize, head: &Matrix) {
    let head_dim = head.cols;
    for r in 0..m.rows {
        m.row_mut(r)[h * head_dim..(h + 1) * head_dim].copy_from_slice(head.row(r));
    }
}

fn add_into(acc: &mut Matrix, g: &Matrix) {
    for (a, g) in acc.data.iter_mut().zip(g.data.iter()) {
        *a += g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::relative_error;
    use crate::nn::softmax_cross_entropy;

    const SEQ: usize = 5;
    const D_MODEL: usize = 8;

    fn setup() -> (CausalSelfAttention, Matrix, Vec<usize>) {
        let mut rng = Rng::new(7);
        let attn = CausalSelfAttention::new(D_MODEL, 2, &mut rng);
        let x = Matrix::from_vec(SEQ, D_MODEL, (0..SEQ * D_MODEL).map(|_| rng.normal()).collect());
        // One target per position, as in next-token prediction.
        let targets = vec![3, 0, 7, 1, 4];
        (attn, x, targets)
    }

    fn loss(attn: &mut CausalSelfAttention, x: &Matrix, targets: &[usize]) -> f32 {
        softmax_cross_entropy(&attn.forward(x), targets).0
    }

    fn projection(attn: &mut CausalSelfAttention, i: usize) -> &mut Linear {
        match i {
            0 => &mut attn.wq,
            1 => &mut attn.wk,
            2 => &mut attn.wv,
            _ => &mut attn.wo,
        }
    }

    #[test]
    fn analytic_gradient_matches_numerical() {
        let (mut attn, mut x, targets) = setup();
        let eps = 1e-3;

        attn.zero_grad();
        let (_, dlogits) = softmax_cross_entropy(&attn.forward(&x), &targets);
        let dx = attn.backward(&dlogits);

        // Every weight of every projection. Wk is the one most worth
        // checking: its gradient arrives through a transposed product.
        for (i, name) in ["Wq", "Wk", "Wv", "Wo"].into_iter().enumerate() {
            let analytic = projection(&mut attn, i).weight_grads().unwrap().to_vec();
            let mut numerical = vec![0.0; analytic.len()];
            for (idx, slot) in numerical.iter_mut().enumerate() {
                projection(&mut attn, i).weights_mut().unwrap()[idx] += eps;
                let up = loss(&mut attn, &x, &targets);
                projection(&mut attn, i).weights_mut().unwrap()[idx] -= 2.0 * eps;
                let down = loss(&mut attn, &x, &targets);
                projection(&mut attn, i).weights_mut().unwrap()[idx] += eps;
                *slot = (up - down) / (2.0 * eps);
            }
            let rel = relative_error(&analytic, &numerical);
            assert!(rel < 1e-2, "{name}: analytic and numerical gradients differ (rel {rel:.4})");
        }

        // And the gradient handed to the layer below. In a transformer this
        // is the one every earlier layer depends on.
        let mut numerical = vec![0.0; x.data.len()];
        for (idx, slot) in numerical.iter_mut().enumerate() {
            x.data[idx] += eps;
            let up = loss(&mut attn, &x, &targets);
            x.data[idx] -= 2.0 * eps;
            let down = loss(&mut attn, &x, &targets);
            x.data[idx] += eps;
            *slot = (up - down) / (2.0 * eps);
        }
        let rel = relative_error(&dx.data, &numerical);
        assert!(rel < 1e-2, "dx: analytic and numerical gradients differ (rel {rel:.4})");
    }

    /// Changing a token must not change anything before it.
    #[test]
    fn the_future_cannot_leak_into_the_past() {
        let (mut attn, mut x, _) = setup();
        let before = attn.forward(&x);

        x.row_mut(SEQ - 1).iter_mut().for_each(|v| *v += 1.0);
        let after = attn.forward(&x);

        for r in 0..SEQ - 1 {
            assert_eq!(before.row(r), after.row(r), "position {r} saw the last token");
        }
        assert_ne!(before.row(SEQ - 1), after.row(SEQ - 1));
    }

    /// The same, seen from the gradient's side: a loss on the first position
    /// alone has no business blaming any later input.
    #[test]
    fn the_past_is_not_blamed_on_the_future() {
        let (mut attn, x, _) = setup();
        attn.forward(&x);

        let mut dy = Matrix::zeros(SEQ, D_MODEL);
        dy.row_mut(0).fill(1.0);
        let dx = attn.backward(&dy);

        assert!(dx.row(0).iter().any(|&g| g != 0.0));
        for r in 1..SEQ {
            assert!(dx.row(r).iter().all(|&g| g == 0.0), "position {r} was blamed");
        }
    }
}
