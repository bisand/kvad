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
        let (m, k, n) = (self.rows, self.cols, b.rows);
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            let a_row = self.row(i);
            let out_row = out.row_mut(i);
            for j in 0..n {
                let b_row = b.row(j);
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a_row[p] * b_row[p];
                }
                out_row[j] = acc;
            }
        }
        out
    }
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
}
