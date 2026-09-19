//! The embedding table: token ids in, vectors out.
//!
//! # The one idea in this file
//!
//! An embedding looks like a new kind of thing — a lookup, not arithmetic —
//! and a lookup does not obviously have a derivative. But it is a layer you
//! already have. Write token 3 of a 5-token vocabulary as a *one-hot* row,
//! `[0, 0, 0, 1, 0]`, feed it to a bias-free `Linear`, and the product
//! `one_hot @ W` multiplies every row of `W` by zero except row 3, which it
//! multiplies by one. It returns row 3. An embedding is that `Linear` layer
//! with the multiplications by zero skipped.
//!
//! So the backward pass needs no new calculus, only the same shortcut applied
//! to `Linear`'s rule. `dW = xᵀ @ dy` with one-hot `x` says: *add each row of
//! `dy` into the row of `dW` that its token selected.* That is all of it. The
//! tests check the shortcut against the real one-hot matmul, to the bit.
//!
//! Two consequences worth seeing in the code:
//!
//! * **Add, do not assign.** A token that appears three times in the sequence
//!   was used three times, and is to blame three times. Writing `=` where
//!   `+=` belongs is the classic embedding bug, and it still trains.
//! * **Most of the gradient is zero.** Only rows whose tokens actually
//!   appeared receive anything. A rare token learns only on the rare steps
//!   where it shows up — one reason tokenisers work to avoid rare tokens.
//!
//! # Why this is not a `Layer`
//!
//! [`Layer`](crate::nn::Layer) maps a matrix to a matrix and hands a gradient
//! to whatever is below it. An embedding takes ids, and nothing is below it:
//! a token id is a name, not a quantity, and there is no such thing as
//! nudging token 3 a little towards token 4. The chain rule stops here, so
//! `backward` returns nothing.
//!
//! A learned *positional* embedding, as in GPT-2, is this same struct looked
//! up with `0, 1, 2, ...` instead of token ids, and added to the result.

use crate::matrix::Matrix;
use crate::nn::sgd;
use crate::rng::Rng;

pub struct Embedding {
    /// [vocab, dim] — one row per token.
    table: Matrix,
    dtable: Matrix,
    vtable: Matrix,
    /// The ids we were given, needed to route the gradient back.
    ids: Vec<usize>,
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize, rng: &mut Rng) -> Self {
        // Not He initialisation. That rule exists because a Linear layer sums
        // fan_in terms and the sum's variance has to be tamed; a lookup sums
        // exactly one. Small values, as GPT-2 uses, so that no token starts
        // out shouting.
        let data = (0..vocab * dim).map(|_| rng.normal() * 0.02).collect();
        Embedding {
            table: Matrix::from_vec(vocab, dim, data),
            dtable: Matrix::zeros(vocab, dim),
            vtable: Matrix::zeros(vocab, dim),
            ids: Vec::new(),
        }
    }

    pub fn param_count(&self) -> usize {
        self.table.data.len()
    }

    /// One row of the table per id: `[ids.len(), dim]`.
    pub fn forward(&mut self, ids: &[usize]) -> Matrix {
        let mut out = Matrix::zeros(ids.len(), self.table.cols);
        for (r, &id) in ids.iter().enumerate() {
            assert!(id < self.table.rows, "token id {id} is outside a vocabulary of {}", self.table.rows);
            out.row_mut(r).copy_from_slice(self.table.row(id));
        }
        self.ids = ids.to_vec();
        out
    }

    /// Route each row of `dy` back to the table row it was copied from.
    pub fn backward(&mut self, dy: &Matrix) {
        assert_eq!(dy.rows, self.ids.len(), "backward got {} rows for {} tokens", dy.rows, self.ids.len());
        for (r, &id) in self.ids.iter().enumerate() {
            for (acc, g) in self.dtable.row_mut(id).iter_mut().zip(dy.row(r)) {
                *acc += g;
            }
        }
    }

    pub fn step(&mut self, lr: f32, momentum: f32) {
        sgd(&mut self.table.data, &self.dtable.data, &mut self.vtable.data, lr, momentum);
    }

    pub fn zero_grad(&mut self) {
        self.dtable.fill(0.0);
    }

    pub fn describe(&self) -> String {
        format!("Embedding({} tokens x {}, {} params)", self.table.rows, self.table.cols, self.param_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::relative_error;
    use crate::nn::softmax_cross_entropy;

    const VOCAB: usize = 7;
    const DIM: usize = 5;
    /// Token 2 appears three times and tokens 0, 3 and 6 never do, so both
    /// consequences from the top of the file are exercised.
    const IDS: [usize; 6] = [2, 5, 2, 1, 4, 2];

    fn random(rows: usize, cols: usize, rng: &mut Rng) -> Matrix {
        Matrix::from_vec(rows, cols, (0..rows * cols).map(|_| rng.normal()).collect())
    }

    fn one_hot(ids: &[usize]) -> Matrix {
        let mut m = Matrix::zeros(ids.len(), VOCAB);
        for (r, &id) in ids.iter().enumerate() {
            m.row_mut(r)[id] = 1.0;
        }
        m
    }

    /// The claim this file rests on, checked exactly: the lookup and the
    /// scatter-add are `Linear`'s forward and backward with the zeros skipped.
    /// Not approximately — the same additions in the same order, so the same
    /// bits.
    #[test]
    fn lookup_is_a_one_hot_matmul() {
        let mut rng = Rng::new(21);
        let mut emb = Embedding::new(VOCAB, DIM, &mut rng);
        let dy = random(IDS.len(), DIM, &mut rng);
        let x = one_hot(&IDS);

        assert_eq!(emb.forward(&IDS), x.matmul(&emb.table));

        emb.backward(&dy);
        assert_eq!(emb.dtable, x.matmul_at_b(&dy));
    }

    #[test]
    fn a_token_is_blamed_once_per_use_and_not_otherwise() {
        let mut rng = Rng::new(22);
        let mut emb = Embedding::new(VOCAB, DIM, &mut rng);
        let dy = random(IDS.len(), DIM, &mut rng);
        emb.forward(&IDS);
        emb.backward(&dy);

        // Token 2 sat at positions 0, 2 and 5.
        for j in 0..DIM {
            let expected = dy.get(0, j) + dy.get(2, j) + dy.get(5, j);
            assert_eq!(emb.dtable.get(2, j), expected);
        }
        for unused in [0, 3, 6] {
            assert!(emb.dtable.row(unused).iter().all(|&g| g == 0.0), "token {unused} was never used");
        }
    }

    /// And through a loss, like every other layer — it costs nothing, and it
    /// is the check that would still work if the one-hot argument were wrong.
    #[test]
    fn analytic_gradient_matches_numerical() {
        let mut rng = Rng::new(23);
        let mut emb = Embedding::new(VOCAB, DIM, &mut rng);
        // Away from the near-zero initial table, where every row gives almost
        // the same uniform prediction and the gradients all look alike.
        emb.table = random(VOCAB, DIM, &mut rng);
        let targets = [4usize, 0, 3, 1, 2, 0];
        let eps = 1e-2;

        let (_, dlogits) = softmax_cross_entropy(&emb.forward(&IDS), &targets);
        emb.backward(&dlogits);

        let mut numerical = vec![0.0; emb.table.data.len()];
        for (idx, slot) in numerical.iter_mut().enumerate() {
            emb.table.data[idx] += eps;
            let up = softmax_cross_entropy(&emb.forward(&IDS), &targets).0;
            emb.table.data[idx] -= 2.0 * eps;
            let down = softmax_cross_entropy(&emb.forward(&IDS), &targets).0;
            emb.table.data[idx] += eps;
            *slot = (up - down) / (2.0 * eps);
        }
        let rel = relative_error(&emb.dtable.data, &numerical);
        assert!(rel < 2e-3, "table: analytic and numerical gradients differ (rel {rel:.4})");
    }

    #[test]
    #[should_panic(expected = "outside a vocabulary")]
    fn an_unknown_token_is_an_error_not_a_crash_somewhere_else() {
        let mut rng = Rng::new(24);
        Embedding::new(VOCAB, DIM, &mut rng).forward(&[VOCAB]);
    }
}
