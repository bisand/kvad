//! Self-attention, forward and backward, with the causal mask or without it.
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
//! # A parameter that does nothing
//!
//! `Wk` is a `Linear`, and a `Linear` has a bias, so every key gets the same
//! vector `b` added to it. Follow that through: `q_i · (k_j + b)` is
//! `q_i · k_j + q_i · b`, and the second term does not depend on `j`. Every
//! score in row `i` moves by the same amount, and softmax does not care — it
//! only sees differences within a row. The key bias cannot change the output,
//! its gradient is exactly zero, and it never trains. GPT-2 ships one in every
//! layer anyway. The gradient check found this, not the algebra: it reported a
//! 100% disagreement on one tensor, which turned out to be rounding noise
//! compared with rounding noise. (With rotary position embeddings the keys are
//! rotated by position after the bias is added, the term stops being constant
//! along the row, and the bias starts to matter — which is why Qwen has one.)
//!
//! One sequence per call. A batch of sequences is a loop around this: the
//! gradient buffers accumulate until `zero_grad`, which is what they were
//! for all along.
//!
//! # With the mask and without it
//!
//! A GPT masks the future because it is trained to predict it: a position
//! that could read the next token would be copying the answer. A diffusion
//! model has nothing to predict in order. Every patch of a noisy image is
//! equally given, and the patch in the corner should be allowed to look at
//! the one in the middle. So [`SelfAttention::bidirectional`] is the same
//! layer with the mask switched off, and the backward pass needs no change
//! at all: it never mentioned the mask in the first place (see
//! `softmax_backward`).
//!
//! [`nn`]: crate::nn
//! [`matrix`]: crate::matrix

use crate::matrix::Matrix;
use crate::nn::{prefixed, Layer, Linear, Param};
use crate::rng::Rng;

/// Which positions each position may read.
///
/// A video is a sequence too, `frames × patches` long, and attention over all
/// of it costs the square of that. The cheaper option attends within groups:
/// within one frame (every patch sees its neighbours, none of the other
/// frames), or across frames at one place (a patch sees itself earlier and
/// later, nothing else). Alternate the two and information still gets from
/// anywhere to anywhere, in two hops rather than one. See `dit` for which
/// blocks use which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every position reads every position.
    All,
    /// Position `i` reads `0..=i`.
    Causal,
    /// The rows in runs of this many, each run reading only itself: one
    /// frame's patches, with the frames one after another.
    Runs(usize),
    /// Rows `i, i + n, i + 2n, ...` reading only each other: one patch
    /// through every frame, for a stride of one frame's patches.
    Strided(usize),
}

impl Scope {
    /// The groups a sequence of `rows` splits into, each a list of rows.
    pub fn groups(self, rows: usize) -> Vec<Vec<usize>> {
        match self {
            Scope::All | Scope::Causal => vec![(0..rows).collect()],
            Scope::Runs(n) => {
                assert!(n > 0 && rows % n == 0, "{rows} rows do not split into runs of {n}");
                (0..rows / n).map(|g| (g * n..(g + 1) * n).collect()).collect()
            }
            Scope::Strided(n) => {
                assert!(n > 0 && rows % n == 0, "{rows} rows do not split with a stride of {n}");
                (0..n).map(|g| (g..rows).step_by(n).collect()).collect()
            }
        }
    }
}

/// Multi-head self-attention over one sequence: `[seq, d_model]` in,
/// `[seq, d_model]` out.
pub struct SelfAttention {
    scope: Scope,
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
///
/// One per head *and group*: attending within groups keeps `g` squares of
/// `seq / g` rows rather than one of `seq`, which is the whole saving.
struct HeadCache {
    head: usize,
    rows: Vec<usize>,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    probs: Matrix,
}

impl SelfAttention {
    /// Each position reads itself and what came before it: a GPT's attention.
    pub fn causal(d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        Self::new(Scope::Causal, d_model, n_heads, rng)
    }

    /// Every position reads every position: a diffusion transformer's.
    pub fn bidirectional(d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        Self::new(Scope::All, d_model, n_heads, rng)
    }

    pub fn new(scope: Scope, d_model: usize, n_heads: usize, rng: &mut Rng) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model {d_model} does not split into {n_heads} heads");
        SelfAttention {
            scope,
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

impl Layer for SelfAttention {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        let scale = self.scale();
        let mut mixed = Matrix::zeros(x.rows, x.cols);
        self.heads.clear();
        let groups = self.scope.groups(x.rows);
        for h in 0..self.n_heads {
            for rows in &groups {
                // A head is nothing more than a slice of the columns, and a
                // group a selection of the rows. The heads never interact
                // until Wo mixes them back together; the groups never do.
                let qh = take_head(&q, h, self.head_dim, rows);
                let kh = take_head(&k, h, self.head_dim, rows);
                let vh = take_head(&v, h, self.head_dim, rows);

                // scores[i][j] = q_i · k_j — how well position i's question
                // matches position j's label.
                let mut probs = qh.matmul_a_bt(&kh);
                probs.data.iter_mut().for_each(|s| *s *= scale);
                softmax_rows(&mut probs, self.scope == Scope::Causal);

                // Each output row is a weighted average of the value rows.
                let out = probs.matmul(&vh);
                put_head(&mut mixed, h, rows, &out);

                self.heads.push(HeadCache { head: h, rows: rows.clone(), q: qh, k: kh, v: vh, probs });
            }
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
        for head in &self.heads {
            let (h, rows) = (head.head, &head.rows);
            let dout = take_head(&dmixed, h, self.head_dim, rows);

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

            put_head(&mut dq, h, rows, &dqh);
            put_head(&mut dk, h, rows, &dkh);
            put_head(&mut dv, h, rows, &dvh);
        }

        // x fed three projections, so it is to blame through all three. When a
        // value is used in several places, its gradients add.
        let mut dx = self.wq.backward(&dq);
        dx.add_in_place(&self.wk.backward(&dk));
        dx.add_in_place(&self.wv.backward(&dv));
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

    fn params(&mut self) -> Vec<Param<'_>> {
        let mut all = prefixed("wq", self.wq.params());
        all.extend(prefixed("wk", self.wk.params()));
        all.extend(prefixed("wv", self.wv.params()));
        all.extend(prefixed("wo", self.wo.params()));
        all
    }

    fn describe(&self) -> String {
        format!(
            "SelfAttention({:?}, {} heads x {}, {} params)",
            self.scope,
            self.n_heads,
            self.head_dim,
            self.param_count()
        )
    }
}

/// Softmax each row in place. With `causal`, row `i` may see only columns
/// `0..=i`.
///
/// The usual description of the causal mask is "set the future scores to -inf
/// before the softmax". Leaving them out of the sum is the same thing, since
/// exp(-inf) = 0, and needs no infinities.
fn softmax_rows(scores: &mut Matrix, causal: bool) {
    for i in 0..scores.rows {
        let row = scores.row_mut(i);
        let seen = if causal { i + 1 } else { row.len() };
        let (seen, future) = row.split_at_mut(seen);

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

/// Copy head `h`'s columns of the given rows out of a `[seq, d_model]` matrix.
fn take_head(m: &Matrix, h: usize, head_dim: usize, rows: &[usize]) -> Matrix {
    let mut out = Matrix::zeros(rows.len(), head_dim);
    for (i, &r) in rows.iter().enumerate() {
        out.row_mut(i).copy_from_slice(&m.row(r)[h * head_dim..(h + 1) * head_dim]);
    }
    out
}

/// The inverse of [`take_head`]: write `[rows, head_dim]` back where it came from.
fn put_head(m: &mut Matrix, h: usize, rows: &[usize], head: &Matrix) {
    let head_dim = head.cols;
    for (i, &r) in rows.iter().enumerate() {
        m.row_mut(r)[h * head_dim..(h + 1) * head_dim].copy_from_slice(head.row(i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::relative_error;
    use crate::nn::softmax_cross_entropy;

    const SEQ: usize = 5;
    const D_MODEL: usize = 8;

    fn setup() -> (SelfAttention, Matrix, Vec<usize>) {
        setup_with(SelfAttention::causal)
    }

    fn setup_with(build: fn(usize, usize, &mut Rng) -> SelfAttention) -> (SelfAttention, Matrix, Vec<usize>) {
        let mut rng = Rng::new(7);
        let attn = build(D_MODEL, 2, &mut rng);
        let x = Matrix::from_vec(SEQ, D_MODEL, (0..SEQ * D_MODEL).map(|_| rng.normal()).collect());
        // One target per position, as in next-token prediction.
        let targets = vec![3, 0, 7, 1, 4];
        (attn, x, targets)
    }

    fn loss(attn: &mut SelfAttention, x: &Matrix, targets: &[usize]) -> f32 {
        softmax_cross_entropy(&attn.forward(x), targets).0
    }

    fn projection(attn: &mut SelfAttention, i: usize) -> &mut Linear {
        match i {
            0 => &mut attn.wq,
            1 => &mut attn.wk,
            2 => &mut attn.wv,
            _ => &mut attn.wo,
        }
    }

    #[test]
    fn analytic_gradient_matches_numerical() {
        check_gradients(setup());
    }

    /// The same check with the mask off. Nothing in backward changed, and this
    /// is what says that nothing had to: the masked entries were zeros that
    /// the backward pass multiplied through, not a case it handled.
    #[test]
    fn bidirectional_gradient_matches_numerical() {
        check_gradients(setup_with(SelfAttention::bidirectional));
    }

    fn check_gradients((mut attn, mut x, targets): (SelfAttention, Matrix, Vec<usize>)) {
        let eps = 1e-3;

        attn.zero_grad();
        let (_, dlogits) = softmax_cross_entropy(&attn.forward(&x), &targets);
        let dx = attn.backward(&dlogits);

        // Every weight of every projection. Wk is the one most worth
        // checking: its gradient arrives through a transposed product.
        for (i, name) in ["Wq", "Wk", "Wv", "Wo"].into_iter().enumerate() {
            let analytic = projection(&mut attn, i).params()[0].grad.to_vec();
            let mut numerical = vec![0.0; analytic.len()];
            for (idx, slot) in numerical.iter_mut().enumerate() {
                projection(&mut attn, i).params()[0].value[idx] += eps;
                let up = loss(&mut attn, &x, &targets);
                projection(&mut attn, i).params()[0].value[idx] -= 2.0 * eps;
                let down = loss(&mut attn, &x, &targets);
                projection(&mut attn, i).params()[0].value[idx] += eps;
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

    /// See "A parameter that does nothing" at the top of the file.
    #[test]
    fn the_key_bias_cannot_change_anything() {
        let (mut attn, x, targets) = setup();
        crate::gradcheck::scramble(attn.params(), &mut Rng::new(5));
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

        attn.zero_grad();
        let (_, dlogits) = softmax_cross_entropy(&attn.forward(&x), &targets);
        attn.backward(&dlogits);
        for p in attn.params() {
            match p.name.as_str() {
                "wk.bias" => assert!(norm(p.grad) < 1e-6, "key bias gradient {:e}", norm(p.grad)),
                _ => assert!(norm(p.grad) > 1e-2, "{} gradient {:e}", p.name, norm(p.grad)),
            }
        }

        // Not merely a small gradient at this point: move it a long way and
        // the output stays where it was.
        let before = attn.forward(&x);
        for p in attn.params() {
            if p.name == "wk.bias" {
                p.value.iter_mut().for_each(|v| *v += 5.0);
            }
        }
        let after = attn.forward(&x);
        let moved = before.data.iter().zip(&after.data).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(moved < 1e-5, "output moved by {moved:e}");
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

    /// And with the mask off, it must leak everywhere: that is the point of
    /// taking it off.
    #[test]
    fn without_the_mask_every_position_sees_every_other() {
        let (mut attn, mut x, _) = setup_with(SelfAttention::bidirectional);
        let before = attn.forward(&x);

        x.row_mut(SEQ - 1).iter_mut().for_each(|v| *v += 1.0);
        let after = attn.forward(&x);

        for r in 0..SEQ {
            assert_ne!(before.row(r), after.row(r), "position {r} did not see the last one");
        }
    }

    // -----------------------------------------------------------------------
    // Attention within groups
    // -----------------------------------------------------------------------

    /// Three frames of two patches: rows 0-1, 2-3, 4-5.
    const FRAMES: usize = 3;
    const PATCHES: usize = 2;

    fn grouped(scope: Scope, seed: u64) -> (SelfAttention, Matrix) {
        let mut rng = Rng::new(seed);
        let attn = SelfAttention::new(scope, D_MODEL, 2, &mut rng);
        let rows = FRAMES * PATCHES;
        (attn, Matrix::from_vec(rows, D_MODEL, (0..rows * D_MODEL).map(|_| rng.normal()).collect()))
    }

    #[test]
    fn groups_are_frames_or_places() {
        assert_eq!(Scope::Runs(2).groups(6), vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
        assert_eq!(Scope::Strided(2).groups(6), vec![vec![0, 2, 4], vec![1, 3, 5]]);
        assert_eq!(Scope::All.groups(3), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn grouped_gradient_matches_numerical() {
        for scope in [Scope::Runs(PATCHES), Scope::Strided(PATCHES)] {
            let (mut attn, x) = grouped(scope, 9);
            crate::gradcheck::scramble(attn.params(), &mut Rng::new(10));
            for c in crate::gradcheck::check_layer(&mut attn, &x, &[3, 0, 7, 1, 4, 6], 1e-2) {
                if c.name == "wk.bias" {
                    continue; // zero by construction; see above
                }
                assert!(c.rel < 5e-3, "{scope:?} {}: analytic and numerical gradients differ (rel {:.4})", c.name, c.rel);
            }
        }
    }

    /// Attending within groups is exactly attending to each group alone, with
    /// the same weights: nothing is masked, nothing leaks, and nothing else is
    /// computed. So run every group through a plain bidirectional layer with
    /// copied weights and compare.
    #[test]
    fn a_group_attends_as_if_it_were_the_whole_sequence() {
        for scope in [Scope::Runs(PATCHES), Scope::Strided(PATCHES)] {
            let (mut attn, x) = grouped(scope, 11);
            crate::gradcheck::scramble(attn.params(), &mut Rng::new(12));
            let mut alone = SelfAttention::bidirectional(D_MODEL, 2, &mut Rng::new(0));
            for (theirs, ours) in alone.params().into_iter().zip(attn.params()) {
                theirs.value.copy_from_slice(ours.value);
            }

            let together = attn.forward(&x);
            for rows in scope.groups(x.rows) {
                let mut part = Matrix::zeros(rows.len(), D_MODEL);
                for (i, &r) in rows.iter().enumerate() {
                    part.row_mut(i).copy_from_slice(x.row(r));
                }
                let out = alone.forward(&part);
                for (i, &r) in rows.iter().enumerate() {
                    let gap = out.row(i).iter().zip(together.row(r)).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                    assert!(gap < 1e-5, "{scope:?}: row {r} differs by {gap:e}");
                }
            }
        }
    }

    /// A change in one frame reaches only that frame under `Runs`, and only
    /// that place in every frame under `Strided`.
    #[test]
    fn a_change_stays_inside_its_group() {
        for scope in [Scope::Runs(PATCHES), Scope::Strided(PATCHES)] {
            let (mut attn, mut x) = grouped(scope, 13);
            let before = attn.forward(&x);
            x.row_mut(4).iter_mut().for_each(|v| *v += 1.0);
            let after = attn.forward(&x);
            let group: Vec<usize> = scope.groups(x.rows).into_iter().find(|g| g.contains(&4)).unwrap();
            for r in 0..x.rows {
                match group.contains(&r) {
                    true => assert_ne!(before.row(r), after.row(r), "{scope:?}: row {r} did not see row 4"),
                    false => assert_eq!(before.row(r), after.row(r), "{scope:?}: row {r} saw row 4"),
                }
            }
        }
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
