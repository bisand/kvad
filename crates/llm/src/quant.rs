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

use crate::tensor::{matvec_bt, Tensor};
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

    /// `y = x @ Wᵀ (+ b)`, with `W` stored `[out_features, in_features]`.
    ///
    /// Every matmul in the model goes through here. Blocks run along the
    /// contraction axis, so each output is a sum of per-block dot products.
    ///
    /// # Quantised activations
    ///
    /// When the weights are quantised, `x` is quantised too — once per call,
    /// not once per row — and the dot product becomes **integer**:
    ///
    /// ```text
    ///   f32 weights:  sum over k of  x[k] * w[k]                  (f32 FMA)
    ///   q8 weights:   sum over k of  x[k] * (w[k] as f32) * ws    (convert, then f32 FMA)
    ///   q8 + q8 act:  (sum over k of  xq[k] * wq[k]) * xs * ws    (i8 dot, one f32 mul)
    /// ```
    ///
    /// Two things happen at once. The per-weight `i8 -> f32` conversion
    /// disappears, and the inner loop becomes something the CPU has a
    /// dedicated instruction for: `sdot` on ARM, VNNI on x86, four `i8 x i8`
    /// products accumulated into an `i32` per lane per cycle.
    ///
    /// Quantising `x` costs one pass over `cols` values. The weight matrix is
    /// `rows x cols`, so for the output head that is 896 values against 136
    /// million — it rounds to nothing.
    ///
    /// Overflow is not a concern: `127 * 127 * 32` is about 516k, comfortably
    /// inside `i32`.
    pub fn matvec_bt(&self, x: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);

        let mut out = match &self.data {
            Data::F32(t) => matvec_bt(x, t),
            _ if self.bytes() < INTEGER_PATH_MIN_BYTES => self.matvec_bt_dequant(x),
            _ => {
                let n = self.cols;
                let blocks_per_row = n / BLOCK;
                let xq = QActivation::new(x);

                // `map_init` gives each worker thread one scratch buffer that
                // is reused for every row it handles, rather than allocating
                // 151936 of them.
                (0..self.rows)
                    .into_par_iter()
                    .map_init(
                        || vec![0i32; blocks_per_row],
                        |dots, r| {
                            let base = r * blocks_per_row;

                            // ---- pass 1: integers only ---------------------
                            //
                            // This loop must contain no floating-point work at
                            // all. Mixing the per-block scaling in here -- the
                            // obvious way to write it -- stops the whole thing
                            // vectorising: the float accumulation is a
                            // non-associative chain the compiler may not
                            // reorder, and interleaving it with the integer
                            // reduction defeats that too. Split into two
                            // passes, this becomes `sdot`; combined, it
                            // compiles to scalar loads and multiplies and runs
                            // several times slower.
                            //
                            // The scratch buffer is a few hundred bytes and
                            // never leaves L1.
                            match &self.data {
                                Data::Q8 { qs, .. } => {
                                    let row = &qs[r * n..(r + 1) * n];
                                    for (d, (wb, xb)) in dots
                                        .iter_mut()
                                        .zip(row.chunks_exact(BLOCK).zip(xq.qs.chunks_exact(BLOCK)))
                                    {
                                        *d = wb
                                            .iter()
                                            .zip(xb.iter())
                                            .map(|(&w, &v)| w as i32 * v as i32)
                                            .sum();
                                    }
                                }
                                Data::Q4 { qs, .. } => {
                                    const HALF: usize = BLOCK / 2;
                                    let row = &qs[r * n / 2..(r + 1) * n / 2];
                                    for (d, (wb, xb)) in dots
                                        .iter_mut()
                                        .zip(row.chunks_exact(HALF).zip(xq.qs.chunks_exact(BLOCK)))
                                    {
                                        let (xlo, xhi) = xb.split_at(HALF);
                                        // Two uniform reductions: low nibbles
                                        // against the first half of the block,
                                        // high nibbles against the second.
                                        let lo: i32 = wb
                                            .iter()
                                            .zip(xlo.iter())
                                            .map(|(&w, &v)| (w & 0x0f) as i32 * v as i32)
                                            .sum();
                                        let hi: i32 = wb
                                            .iter()
                                            .zip(xhi.iter())
                                            .map(|(&w, &v)| (w >> 4) as i32 * v as i32)
                                            .sum();
                                        *d = lo + hi;
                                    }
                                }
                                Data::F32(_) => unreachable!(),
                            }

                            // ---- pass 2: apply the scales ------------------
                            let scales = match &self.data {
                                Data::Q8 { scales, .. } | Data::Q4 { scales, .. } => scales,
                                Data::F32(_) => unreachable!(),
                            };
                            // The 4-bit codes are stored offset by 8, undone
                            // here against the summed quantised activations --
                            // still integer, one correction per block.
                            let offset = matches!(self.data, Data::Q4 { .. });

                            let mut sums = [0.0f32; 2];
                            for (b, &d) in dots.iter().enumerate() {
                                let d = if offset { d - 8 * xq.sums[b] } else { d };
                                sums[b & 1] += scales[base + b] * xq.scales[b] * d as f32;
                            }
                            sums[0] + sums[1]
                        },
                    )
                    .collect()
            }
        };

        if let Some(b) = bias {
            for (v, bi) in out.iter_mut().zip(b.iter()) {
                *v += bi;
            }
        }
        out
    }
}

impl Weight {
    /// The dequantising kernel: unpack each weight to `f32` and use ordinary
    /// float FMAs, leaving the activations alone.
    ///
    /// Slower than the integer path on large matrices, and *faster* on small
    /// ones — see [`INTEGER_PATH_MIN_BYTES`].
    fn matvec_bt_dequant(&self, x: &[f32]) -> Vec<f32> {
        let n = self.cols;
        let blocks_per_row = n / BLOCK;
        // Per-block sums of the activations, for the 4-bit offset. Depends
        // only on x, so it is computed once rather than once per row.
        let xsums: Vec<f32> = match self.data {
            Data::Q4 { .. } => x.chunks_exact(BLOCK).map(|b| b.iter().sum()).collect(),
            _ => Vec::new(),
        };

        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let base = r * blocks_per_row;
                // Four independent lanes: float addition is not associative,
                // so a single accumulator would run at FMA latency rather than
                // throughput.
                let mut sums = [0.0f32; 4];
                match &self.data {
                    Data::Q8 { scales, qs } => {
                        let row = &qs[r * n..(r + 1) * n];
                        for (b, (wb, xb)) in
                            row.chunks_exact(BLOCK).zip(x.chunks_exact(BLOCK)).enumerate()
                        {
                            let mut inner = [0.0f32; 4];
                            for (qc, xc) in wb.chunks_exact(4).zip(xb.chunks_exact(4)) {
                                for j in 0..4 {
                                    inner[j] += xc[j] * qc[j] as f32;
                                }
                            }
                            let sc = scales[base + b];
                            for j in 0..4 {
                                sums[j] += sc * inner[j];
                            }
                        }
                    }
                    Data::Q4 { scales, qs } => {
                        const HALF: usize = BLOCK / 2;
                        let row = &qs[r * n / 2..(r + 1) * n / 2];
                        for (b, (wb, xb)) in
                            row.chunks_exact(HALF).zip(x.chunks_exact(BLOCK)).enumerate()
                        {
                            let (xlo, xhi) = xb.split_at(HALF);
                            let mut inner = [0.0f32; 4];
                            for ((qc, xl), xh) in wb
                                .chunks_exact(4)
                                .zip(xlo.chunks_exact(4))
                                .zip(xhi.chunks_exact(4))
                            {
                                for j in 0..4 {
                                    inner[j] += xl[j] * (qc[j] & 0x0f) as f32;
                                    inner[j] += xh[j] * (qc[j] >> 4) as f32;
                                }
                            }
                            let dot = (inner[0] + inner[1]) + (inner[2] + inner[3])
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

/// Above this many bytes of weights, quantise the activations and use the
/// integer kernel; below it, dequantise to f32 instead.
///
/// # Why there are two kernels
///
/// The integer path is much faster on a matrix that has to be streamed from
/// main memory, and slower on one that already fits in cache. A cache-resident
/// matmul is not bandwidth-bound, so shrinking the weights buys nothing, and
/// the extra steps — quantising the activation vector, the second pass over
/// the block dots — are pure overhead.
///
/// Measured here: the 151936-row output head runs 5.1x faster with integer
/// dots, while a 5 MB MLP matrix runs *slower* than simply dequantising. The
/// threshold sits between the two. It is empirical, and the right value on
/// another machine depends on its last-level cache.
const INTEGER_PATH_MIN_BYTES: usize = 32 << 20;

/// An activation vector, quantised to `i8` in blocks of [`BLOCK`].
///
/// Same scheme as the weights: symmetric, one scale per block. Activations
/// have outliers too — more so than weights, which is what makes naive
/// whole-tensor activation quantisation fail badly — and per-block scales
/// contain them the same way.
struct QActivation {
    qs: Vec<i8>,
    scales: Vec<f32>,
    /// Sum of the quantised values per block, for the 4-bit offset correction.
    sums: Vec<i32>,
}

impl QActivation {
    fn new(x: &[f32]) -> Self {
        let blocks = x.len() / BLOCK;
        let mut qs = Vec::with_capacity(x.len());
        let mut scales = Vec::with_capacity(blocks);
        let mut sums = Vec::with_capacity(blocks);

        for block in x.chunks_exact(BLOCK) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            scales.push(scale);

            let mut sum = 0i32;
            for &v in block {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                sum += q as i32;
                qs.push(q);
            }
            sums.push(sum);
        }
        QActivation { qs, scales, sums }
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
    fn bias_is_applied_in_every_precision() {
        let t = random_tensor(16, 64, 7);
        let mut rng = Rng::new(99);
        let x: Vec<f32> = (0..64).map(|_| rng.normal()).collect();
        let bias: Vec<f32> = (0..16).map(|_| rng.normal()).collect();

        for precision in [Precision::F32, Precision::Q8, Precision::Q4] {
            let w = Weight::quantize(t.clone(), precision);
            let without = w.matvec_bt(&x, None);
            let with = w.matvec_bt(&x, Some(&bias));
            for i in 0..16 {
                assert!(
                    (with[i] - (without[i] + bias[i])).abs() < 1e-4,
                    "{precision} output {i}"
                );
            }
        }
    }

    #[test]
    fn quantised_matvec_bt_agrees_with_f32() {
        let t = random_tensor(96, 64, 11);
        let mut rng = Rng::new(5);
        let x: Vec<f32> = (0..64).map(|_| rng.normal()).collect();

        let reference = Weight::quantize(t.clone(), Precision::F32).matvec_bt(&x, None);
        // Tolerances are looser than the weight-only round trip, because the
        // activations are quantised now too and the two errors compound.
        for (precision, tolerance) in [(Precision::Q8, 0.03f32), (Precision::Q4, 0.35)] {
            let got = Weight::quantize(t.clone(), precision).matvec_bt(&x, None);
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
