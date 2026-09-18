//! Block-wise quantisation.
//!
//! # Why this helps at all
//!
//! Generating one token at a time makes every matmul in the model a
//! matrix-*vector* product: each weight is read from memory, used for exactly
//! one multiply-add, and thrown away. Arithmetic is not the bottleneck —
//! moving the bytes is. So the way to go faster is to make the weights
//! smaller, and accept some arithmetic to unpack them.
//!
//! # Why *block-wise*
//!
//! The naive scheme is one scale for the whole tensor: `q = round(x / scale)`
//! with `scale = max|x| / 127`. It fails badly, because weight matrices have
//! outliers. One weight of 4.0 in a tensor whose values are otherwise under
//! 0.05 forces a scale that rounds almost everything to zero.
//!
//! Instead, chop each row into blocks of [`BLOCK`] weights and give every block
//! its own scale. An outlier then only ruins its own 32 neighbours, and the
//! rest of the row is quantised against its own much smaller range. This is the
//! central idea behind the `Q8_0` / `Q4_0` formats that llama.cpp popularised.
//!
//! The scale costs 4 bytes per block, which is 1 bit per weight at `BLOCK` 32:
//!
//! | format | bits/weight | vs f32 |
//! |---|---|---|
//! | f32 | 32 | 1.0x |
//! | q8  | 8 + 1 = 9 | 3.6x smaller |
//! | q4  | 4 + 1 = 5 | 6.4x smaller |
//!
//! (Storing the scale as f16 would give 8.5 and 4.5 bits, which is what
//! production formats do. f32 is kept here because it is one less thing in the
//! way.)

use crate::tensor::{matvec, matvec_bt, Tensor};
use rayon::prelude::*;

/// Weights per block. 32 is the usual choice: small enough to isolate
/// outliers, large enough that the per-block scale is not itself the cost.
pub const BLOCK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Q8,
    Q4,
}

impl Precision {
    pub fn parse(s: &str) -> Option<Precision> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "none" => Some(Precision::F32),
            "q8" | "int8" | "8" => Some(Precision::Q8),
            "q4" | "int4" | "4" => Some(Precision::Q4),
            _ => None,
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Precision::F32 => "f32",
            Precision::Q8 => "q8",
            Precision::Q4 => "q4",
        })
    }
}

enum Data {
    F32(Tensor),
    /// One `i8` per weight, one `f32` scale per block.
    Q8 { scales: Vec<f32>, qs: Vec<i8> },
    /// Two 4-bit values per byte, one `f32` scale per block.
    Q4 { scales: Vec<f32>, qs: Vec<u8> },
}

/// A weight matrix, in whichever precision it was loaded at.
///
/// The model code never branches on precision: it calls [`Weight::matvec`] or
/// [`Weight::matvec_bt`] and the right kernel runs.
pub struct Weight {
    rows: usize,
    cols: usize,
    data: Data,
}

impl Weight {
    /// Quantise a tensor, or keep it as f32.
    ///
    /// Tensors whose row length is not a multiple of [`BLOCK`] are left alone.
    /// Every large matrix in a real transformer has a power-of-two-ish width,
    /// so in practice this only exempts the 1-D norm and bias vectors — and
    /// skipping the padding logic keeps the inner loops tight enough for the
    /// compiler to vectorise.
    pub fn quantize(t: Tensor, precision: Precision) -> Weight {
        let (rows, cols) = (t.rows, t.cols);
        if precision == Precision::F32 || cols % BLOCK != 0 || cols < BLOCK {
            return Weight { rows, cols, data: Data::F32(t) };
        }
        let blocks = rows * cols / BLOCK;
        let mut scales = Vec::with_capacity(blocks);

        match precision {
            Precision::Q8 => {
                let mut qs = Vec::with_capacity(rows * cols);
                for block in t.data.chunks_exact(BLOCK) {
                    // Symmetric: no zero point, so exact zero stays exact.
                    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = amax / 127.0;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    scales.push(scale);
                    qs.extend(block.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8));
                }
                Weight { rows, cols, data: Data::Q8 { scales, qs } }
            }
            Precision::Q4 => {
                let mut qs = Vec::with_capacity(rows * cols / 2);
                for block in t.data.chunks_exact(BLOCK) {
                    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    // 4 bits signed spans -8..7. Scaling by 7 keeps the mapping
                    // symmetric and wastes only the single code -8, which is
                    // worth it for not having to think about asymmetry.
                    let scale = amax / 7.0;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    scales.push(scale);
                    // Split packing: byte k holds weight k in the low nibble
                    // and weight k+16 in the high nibble.
                    //
                    // The obvious alternative -- weights 2k and 2k+1 in one
                    // byte -- forces the reader to interleave two different
                    // extractions across adjacent outputs, which does not
                    // vectorise. This way 16 bytes unpack into two runs of 16
                    // consecutive weights, each a uniform mask or shift across
                    // the whole vector. Production 4-bit formats split for the
                    // same reason.
                    let (lo, hi) = block.split_at(BLOCK / 2);
                    for (&l, &h) in lo.iter().zip(hi.iter()) {
                        let a = (l * inv).round().clamp(-8.0, 7.0) as i32 + 8;
                        let b = (h * inv).round().clamp(-8.0, 7.0) as i32 + 8;
                        qs.push((a as u8 & 0x0f) | ((b as u8 & 0x0f) << 4));
                    }
                }
                Weight { rows, cols, data: Data::Q4 { scales, qs } }
            }
            Precision::F32 => unreachable!(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn precision(&self) -> Precision {
        match self.data {
            Data::F32(_) => Precision::F32,
            Data::Q8 { .. } => Precision::Q8,
            Data::Q4 { .. } => Precision::Q4,
        }
    }

    pub fn param_count(&self) -> usize {
        self.rows * self.cols
    }

    /// Bytes actually held, scales included.
    pub fn bytes(&self) -> usize {
        match &self.data {
            Data::F32(t) => t.data.len() * 4,
            Data::Q8 { scales, qs } => scales.len() * 4 + qs.len(),
            Data::Q4 { scales, qs } => scales.len() * 4 + qs.len(),
        }
    }

    /// Reconstruct one row — the embedding lookup.
    pub fn row(&self, r: usize) -> Vec<f32> {
        let (start, end) = (r * self.cols, (r + 1) * self.cols);
        match &self.data {
            Data::F32(t) => t.row(r).to_vec(),
            Data::Q8 { scales, qs } => {
                let base = r * self.cols / BLOCK;
                qs[start..end]
                    .chunks_exact(BLOCK)
                    .enumerate()
                    .flat_map(|(b, chunk)| {
                        let s = scales[base + b];
                        chunk.iter().map(move |&q| q as f32 * s)
                    })
                    .collect()
            }
            Data::Q4 { scales, qs } => {
                let base = r * self.cols / BLOCK;
                let mut out = vec![0.0f32; self.cols];
                for (b, chunk) in qs[start / 2..end / 2].chunks_exact(BLOCK / 2).enumerate() {
                    let s = scales[base + b];
                    for (k, &byte) in chunk.iter().enumerate() {
                        out[b * BLOCK + k] = ((byte & 0x0f) as i32 - 8) as f32 * s;
                        out[b * BLOCK + BLOCK / 2 + k] = ((byte >> 4) as i32 - 8) as f32 * s;
                    }
                }
                out
            }
        }
    }

    /// `y = x @ W (+ b)`, with `W` stored `[in_features, out_features]`.
    ///
    /// Blocks run along the output axis here, so each `(row, block)` pair
    /// contributes `x[i] * scale` times 32 consecutive integers. Folding the
    /// activation into the scale means one multiply per block rather than one
    /// per weight.
    pub fn matvec(&self, x: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.rows);
        if let Data::F32(t) = &self.data {
            return matvec(x, t, bias);
        }

        let n = self.cols;
        let blocks_per_row = n / BLOCK;
        let mut out = match bias {
            Some(b) => b.to_vec(),
            None => vec![0.0; n],
        };

        // Split the output into whole blocks so every thread owns entire
        // scales; no two threads touch the same accumulator.
        let per_thread = (blocks_per_row / rayon::current_num_threads().max(1)).max(1);
        out.par_chunks_mut(per_thread * BLOCK).enumerate().for_each(|(ci, out_chunk)| {
            let b0 = ci * per_thread;

            // One output block at a time, accumulated in a register-sized
            // array across the whole contraction, and written out once.
            //
            // The obvious loop order -- rows outside, blocks inside -- reads
            // and writes all 32 outputs on every row. That is 256 bytes of
            // output traffic per 32 bytes of q8 weights, so the *output*
            // becomes the bottleneck and quantising makes the kernel slower
            // rather than faster. Hoisting the accumulator out of the row loop
            // removes that traffic entirely.
            for (b, dst) in out_chunk.chunks_exact_mut(BLOCK).enumerate() {
                let bb = b0 + b;
                let mut acc = [0.0f32; BLOCK];

                match &self.data {
                    Data::Q8 { scales, qs } => {
                        for i in 0..self.rows {
                            let xi = x[i];
                            if xi == 0.0 {
                                continue;
                            }
                            let s = xi * scales[i * blocks_per_row + bb];
                            let src = i * n + bb * BLOCK;
                            let q = &qs[src..src + BLOCK];
                            // 32 independent accumulator chains: nothing here
                            // depends on anything else, so this vectorises.
                            for k in 0..BLOCK {
                                acc[k] += s * q[k] as f32;
                            }
                        }
                    }
                    Data::Q4 { scales, qs } => {
                        // The stored nibble is `q + 8`, so the true weight is
                        // `scale * (nibble - 8)`. Rather than subtracting 8
                        // from every nibble, accumulate the scales separately
                        // and correct all 32 outputs once at the end:
                        //   sum s*(nibble - 8) = sum s*nibble - 8 * sum s
                        let mut scale_sum = 0.0f32;
                        for i in 0..self.rows {
                            let xi = x[i];
                            if xi == 0.0 {
                                continue;
                            }
                            let s = xi * scales[i * blocks_per_row + bb];
                            scale_sum += s;
                            let src = (i * n + bb * BLOCK) / 2;
                            let q = &qs[src..src + BLOCK / 2];
                            // Two uniform 16-wide streams.
                            for k in 0..BLOCK / 2 {
                                let byte = q[k];
                                acc[k] += s * (byte & 0x0f) as f32;
                                acc[BLOCK / 2 + k] += s * (byte >> 4) as f32;
                            }
                        }
                        let correction = 8.0 * scale_sum;
                        for v in acc.iter_mut() {
                            *v -= correction;
                        }
                    }
                    Data::F32(_) => unreachable!(),
                }

                for k in 0..BLOCK {
                    dst[k] += acc[k];
                }
            }
        });
        out
    }

    /// `y = x @ Wᵀ`, with `W` stored `[out_features, in_features]`.
    ///
    /// Blocks run along the contraction axis here, so each output is a sum of
    /// per-block dot products, each scaled once at the end. This is the shape
    /// the output head takes, over the whole vocabulary.
    pub fn matvec_bt(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        if let Data::F32(t) = &self.data {
            return matvec_bt(x, t);
        }

        let n = self.cols;
        let blocks_per_row = n / BLOCK;

        // Sum of the activations in each block, precomputed once.
        //
        // The 4-bit path needs this to undo the +8 offset, and it depends only
        // on x -- not on which row is being multiplied. Computing it inside the
        // row loop, as the obvious version does, repeats the whole thing once
        // per output: for a 151936-row output head that is 150k redundant
        // passes over x, which roughly doubles the kernel's work.
        let xsums: Vec<f32> = match self.data {
            Data::Q4 { .. } => x.chunks_exact(BLOCK).map(|b| b.iter().sum()).collect(),
            _ => Vec::new(),
        };

        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let base = r * blocks_per_row;
                // Four partial sums rather than one.
                //
                // Floating-point addition is not associative, so the compiler
                // may not reorder a single accumulator chain -- every add waits
                // for the previous one, and the loop runs at the latency of an
                // FMA rather than its throughput. Splitting into independent
                // lanes is the standard fix, and it is also what lets this
                // vectorise.
                let mut sums = [0.0f32; 4];

                match &self.data {
                    Data::Q8 { scales, qs } => {
                        let row = &qs[r * n..(r + 1) * n];
                        for (b, (qb, xb)) in
                            row.chunks_exact(BLOCK).zip(x.chunks_exact(BLOCK)).enumerate()
                        {
                            let mut inner = [0.0f32; 4];
                            for (qc, xc) in qb.chunks_exact(4).zip(xb.chunks_exact(4)) {
                                inner[0] += xc[0] * qc[0] as f32;
                                inner[1] += xc[1] * qc[1] as f32;
                                inner[2] += xc[2] * qc[2] as f32;
                                inner[3] += xc[3] * qc[3] as f32;
                            }
                            // One scale multiply per block, not per weight.
                            let s = scales[base + b];
                            for j in 0..4 {
                                sums[j] += s * inner[j];
                            }
                        }
                    }
                    Data::Q4 { scales, qs } => {
                        let row = &qs[r * n / 2..(r + 1) * n / 2];
                        for (b, (qb, xb)) in row
                            .chunks_exact(BLOCK / 2)
                            .zip(x.chunks_exact(BLOCK))
                            .enumerate()
                        {
                            // Split packing means the low nibbles are the
                            // first half of the block and the high nibbles the
                            // second, so each is a straight run against its own
                            // half of x -- no interleaving.
                            let (xlo, xhi) = xb.split_at(BLOCK / 2);
                            debug_assert_eq!(xsums.len(), blocks_per_row);
                            let mut lo = [0.0f32; 4];
                            let mut hi = [0.0f32; 4];
                            for ((qc, xl), xh) in qb
                                .chunks_exact(4)
                                .zip(xlo.chunks_exact(4))
                                .zip(xhi.chunks_exact(4))
                            {
                                for j in 0..4 {
                                    lo[j] += xl[j] * (qc[j] & 0x0f) as f32;
                                    hi[j] += xh[j] * (qc[j] >> 4) as f32;
                                }
                            }
                            // Undo the +8 offset once per block, from the
                            // precomputed activation sum.
                            let dot = (lo[0] + lo[1]) + (lo[2] + lo[3]) + (hi[0] + hi[1])
                                + (hi[2] + hi[3])
                                - 8.0 * xsums[b];
                            sums[0] += scales[base + b] * dot;
                        }
                    }
                    Data::F32(_) => unreachable!(),
                }

                (sums[0] + sums[1]) + (sums[2] + sums[3])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanograd::rng::Rng;

    fn random_tensor(rows: usize, cols: usize, seed: u64) -> Tensor {
        let mut rng = Rng::new(seed);
        Tensor::new(rows, cols, (0..rows * cols).map(|_| rng.normal() * 0.1).collect())
    }

    /// Relative error of a round trip through the quantiser.
    fn round_trip_error(precision: Precision) -> f32 {
        let t = random_tensor(8, 256, 1);
        let original = t.data.clone();
        let w = Weight::quantize(t, precision);

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in 0..8 {
            for (i, v) in w.row(r).iter().enumerate() {
                let want = original[r * 256 + i];
                num += ((v - want) as f64).powi(2);
                den += (want as f64).powi(2);
            }
        }
        (num / den).sqrt() as f32
    }

    #[test]
    fn quantisation_error_is_within_expectations() {
        // Rough expectation: uniform rounding to N levels over a block whose
        // extreme is `amax` gives a relative error around 1/(levels * spread).
        let q8 = round_trip_error(Precision::Q8);
        let q4 = round_trip_error(Precision::Q4);
        assert!(q8 < 0.01, "q8 relative error {q8}");
        assert!(q4 < 0.15, "q4 relative error {q4}");
        // And the ordering must hold: more bits, less error.
        assert!(q8 < q4, "q8 {q8} should beat q4 {q4}");
    }

    #[test]
    fn a_single_outlier_does_not_destroy_the_row() {
        // The case that kills per-tensor quantisation: one huge weight among
        // small ones. With per-block scales only the outlier's own block
        // suffers.
        let mut data = vec![0.01f32; 256];
        data[0] = 40.0;
        let w = Weight::quantize(Tensor::new(1, 256, data), Precision::Q8);
        let back = w.row(0);

        assert!((back[0] - 40.0).abs() < 0.2, "outlier itself: {}", back[0]);
        // A weight in a later block is unaffected by the outlier entirely.
        assert!(
            (back[200] - 0.01).abs() < 1e-4,
            "far-away weight was {} (should be ~0.01)",
            back[200]
        );
        // Even inside the outlier's own block, values survive to ~amax/127.
        assert!(back[5].abs() < 0.4, "same-block weight blew up: {}", back[5]);
    }

    #[test]
    fn exact_zero_stays_exact() {
        // Symmetric quantisation has no zero point, so zeros round-trip
        // perfectly. Asymmetric schemes do not, which matters for padding.
        let w = Weight::quantize(Tensor::new(1, 32, vec![0.0; 32]), Precision::Q4);
        assert!(w.row(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn quantised_matvec_agrees_with_f32() {
        let t = random_tensor(64, 128, 7);
        let mut rng = Rng::new(99);
        let x: Vec<f32> = (0..64).map(|_| rng.normal()).collect();
        let bias: Vec<f32> = (0..128).map(|_| rng.normal() * 0.01).collect();

        let reference = Weight::quantize(t.clone(), Precision::F32).matvec(&x, Some(&bias));
        for (precision, tolerance) in [(Precision::Q8, 0.02f32), (Precision::Q4, 0.3)] {
            let got = Weight::quantize(t.clone(), precision).matvec(&x, Some(&bias));
            let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tolerance * scale,
                    "{precision} output {i}: {b} vs {a}"
                );
            }
        }
    }

    #[test]
    fn quantised_matvec_bt_agrees_with_f32() {
        let t = random_tensor(96, 64, 11);
        let mut rng = Rng::new(5);
        let x: Vec<f32> = (0..64).map(|_| rng.normal()).collect();

        let reference = Weight::quantize(t.clone(), Precision::F32).matvec_bt(&x);
        for (precision, tolerance) in [(Precision::Q8, 0.02f32), (Precision::Q4, 0.3)] {
            let got = Weight::quantize(t.clone(), precision).matvec_bt(&x);
            let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, (a, b)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tolerance * scale,
                    "{precision} output {i}: {b} vs {a}"
                );
            }
        }
    }

    #[test]
    fn storage_matches_the_advertised_bits_per_weight() {
        let t = random_tensor(32, 256, 3);
        let n = 32 * 256;
        assert_eq!(Weight::quantize(t.clone(), Precision::F32).bytes(), n * 4);
        // 8 bits of payload + one f32 scale per 32 weights = 9 bits.
        assert_eq!(Weight::quantize(t.clone(), Precision::Q8).bytes(), n + n / BLOCK * 4);
        // 4 bits of payload + the same scale = 5 bits.
        assert_eq!(Weight::quantize(t, Precision::Q4).bytes(), n / 2 + n / BLOCK * 4);
    }

    #[test]
    fn odd_widths_fall_back_to_f32() {
        // 100 is not a multiple of BLOCK, so this stays f32 rather than
        // silently corrupting the tail.
        let w = Weight::quantize(random_tensor(4, 100, 2), Precision::Q4);
        assert_eq!(w.precision(), Precision::F32);
    }
}
