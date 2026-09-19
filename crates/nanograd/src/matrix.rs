//! A row-major 2D matrix of `f32`, and the three matrix products that
//! backpropagation through a linear layer requires.
//!
//! This is the whole "tensor library". Real frameworks generalise to N
//! dimensions and dispatch to BLAS or a GPU, but the arithmetic below is
//! what they are ultimately doing.

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "data length does not match {rows}x{cols}");
        Matrix { rows, cols, data }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn fill(&mut self, v: f32) {
        self.data.iter_mut().for_each(|x| *x = v);
    }

    /// `self += other`, element by element.
    pub fn add_in_place(&mut self, other: &Matrix) {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "add shape mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// `self @ b`, where self is [m, k] and b is [k, n]. Result is [m, n].
    ///
    /// The loop order is i-k-j rather than the textbook i-j-k: it lets the
    /// inner loop walk `b`'s row and the output row contiguously, which is
    /// several times faster for the same number of multiplications. Memory
    /// layout, not flop count, is what makes matmul fast.
    pub fn matmul(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.cols, b.rows, "matmul shape mismatch");
        let (m, k, n) = (self.rows, self.cols, b.cols);
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            let a_row = self.row(i);
            let out_row = out.row_mut(i);
            for p in 0..k {
                let a_ip = a_row[p];
                if a_ip == 0.0 {
                    continue;
                }
                let b_row = &b.data[p * n..(p + 1) * n];
                for j in 0..n {
                    out_row[j] += a_ip * b_row[j];
                }
            }
        }
        out
    }

    /// `selfᵀ @ b`, where self is [m, k] and b is [m, n]. Result is [k, n].
    ///
    /// This is the shape that gradients-with-respect-to-weights take:
    /// dW = xᵀ @ dy.
    pub fn matmul_at_b(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.rows, b.rows, "matmul_at_b shape mismatch");
        let (m, k, n) = (self.rows, self.cols, b.cols);
        let mut out = Matrix::zeros(k, n);
        for i in 0..m {
            let a_row = self.row(i);
            let b_row = b.row(i);
            for p in 0..k {
                let a_ip = a_row[p];
                if a_ip == 0.0 {
                    continue;
                }
                let out_row = &mut out.data[p * n..(p + 1) * n];
                for j in 0..n {
                    out_row[j] += a_ip * b_row[j];
                }
            }
        }
        out
    }

    /// `self @ bᵀ`, where self is [m, k] and b is [n, k]. Result is [m, n].
    ///
    /// This is the shape that gradients-with-respect-to-inputs take:
    /// dx = dy @ Wᵀ.
    pub fn matmul_a_bt(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.cols, b.cols, "matmul_a_bt shape mismatch");
        let (m, n) = (self.rows, b.rows);
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            let a_row = self.row(i);
            let out_row = out.row_mut(i);
            for (j, out) in out_row.iter_mut().enumerate() {
                *out = dot(a_row, b.row(j));
            }
        }
        out
    }
}

/// How many running sums [`dot`] keeps.
const LANES: usize = 8;

/// The dot product, summed in eight lanes rather than one.
///
/// The obvious loop, `acc += a[p] * b[p]`, was half of all the time spent
/// training a transformer in this crate, and the reason is a rule of
/// arithmetic, not of hardware. Floating-point addition is not associative:
/// `(x + y) + z` and `x + (y + z)` round differently. A compiler may not
/// change what a program computes, so it may not reorder that sum, and a sum
/// that must be done in order is done one number at a time — each addition
/// waiting for the one before — while a SIMD register that holds four floats
/// sits idle. (The compiler did vectorise the *multiplications*. Then it
/// added the products up singly.)
///
/// Eight independent sums say, in the code, that the order is ours to choose:
/// lane `l` takes elements `l, l + 8, l + 16, ...`, and the lanes meet at the
/// end. Now the additions are two 4-wide vector adds with no dependence on
/// each other. Measured on `train_text`, this one function made training 1.5x
/// faster end to end; 4 lanes did nothing, and 16 and 32 were slower than 8,
/// because attention's vectors are 16 long and a chunk that does not fill is
/// handled by the slow loop.
///
/// The answer differs from the single sum's in the last bits. Neither is
/// more correct; they are two roundings of the same number.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (a_chunks, b_chunks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = a_chunks.remainder().iter().zip(b_chunks.remainder()).map(|(x, y)| x * y).sum();
    for (ca, cb) in a_chunks.zip(b_chunks) {
        for l in 0..LANES {
            acc[l] += ca[l] * cb[l];
        }
    }
    acc.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(rows: usize, cols: usize, d: &[f32]) -> Matrix {
        Matrix::from_vec(rows, cols, d.to_vec())
    }

    /// Reference implementation: the definition, written as literally as
    /// possible. The optimised kernels above must agree with it.
    fn naive_matmul(a: &Matrix, b: &Matrix) -> Matrix {
        let mut out = Matrix::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut acc = 0.0;
                for p in 0..a.cols {
                    acc += a.get(i, p) * b.get(p, j);
                }
                out.data[i * b.cols + j] = acc;
            }
        }
        out
    }

    fn transpose(a: &Matrix) -> Matrix {
        let mut out = Matrix::zeros(a.cols, a.rows);
        for i in 0..a.rows {
            for j in 0..a.cols {
                out.data[j * a.rows + i] = a.get(i, j);
            }
        }
        out
    }

    #[test]
    fn matmul_matches_definition() {
        let a = m(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let b = m(3, 2, &[7., 8., 9., 10., 11., 12.]);
        assert_eq!(a.matmul(&b), naive_matmul(&a, &b));
        assert_eq!(a.matmul(&b), m(2, 2, &[58., 64., 139., 154.]));
    }

    #[test]
    fn transposed_variants_match_definition() {
        let a = m(3, 2, &[1., 2., 3., 4., 5., 6.]);
        let b = m(3, 4, &[1., 0., -1., 2., 3., 1., 0., -2., 0.5, 2., 1., 1.]);
        assert_eq!(a.matmul_at_b(&b), naive_matmul(&transpose(&a), &b));

        let c = m(2, 4, &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let d = m(3, 4, &[1., 0., -1., 2., 3., 1., 0., -2., 0.5, 2., 1., 1.]);
        assert_eq!(c.matmul_a_bt(&d), naive_matmul(&c, &transpose(&d)));
    }

    /// Wide enough to fill the lanes of `dot` four times over and leave five
    /// elements for the remainder, which the small cases above never reach.
    /// The values are small whole numbers, so every partial sum is exact and
    /// the order of addition cannot excuse a difference.
    #[test]
    fn a_bt_matches_definition_beyond_one_chunk() {
        let (rows, k, n) = (3, 4 * LANES + 5, 6);
        let values = |count: usize, salt: usize| (0..count).map(|i| ((i * 7 + salt) % 11) as f32 - 5.0).collect();
        let a = Matrix::from_vec(rows, k, values(rows * k, 1));
        let b = Matrix::from_vec(n, k, values(n * k, 4));
        assert_eq!(a.matmul_a_bt(&b), naive_matmul(&a, &transpose(&b)));
    }
}
