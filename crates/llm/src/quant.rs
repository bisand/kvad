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

use crate::qcache::Store;
use crate::tensor::{gemm_bt, matvec_bt, Tensor};
use rayon::prelude::*;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Rows one task should handle, so that a parallel section is one job per
/// thread instead of a tree of them.
///
/// `into_par_iter().map(..).collect()` over 896 rows looks innocent and is
/// not: rayon splits it adaptively, which means a binary tree of `join`s,
/// steal attempts between them, an epoch-based reclaim, and an allocation for
/// the collected result — all to hand each thread a few microseconds of work.
/// Decoding one token does this 169 times. Slicing the output into exactly one
/// chunk per thread up front costs none of it.
#[inline]
fn rows_per_task(rows: usize) -> usize {
    rows.div_ceil(rayon::current_num_threads().max(1)).max(1)
}

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

/// The arrays a weight is made of.
///
/// [`Store`] rather than `Vec` because these may be windows onto a
/// memory-mapped cache file instead of memory this process filled in — see
/// [`crate::qcache`]. It derefs to a slice, so every kernel below is written
/// as if these were still plain `Vec`s.
enum Data {
    F32(Tensor),
    /// One `i8` per weight, one `f32` scale per block.
    Q8 { scales: Store<f32>, qs: Store<i8> },
    /// Two 4-bit values per byte, one `f32` scale per block.
    Q4 { scales: Store<f32>, qs: Store<u8> },
}

/// The shape checks a quantised weight has to satisfy: one scale per block,
/// and `per_byte` weights packed into each stored byte.
fn check(rows: usize, cols: usize, scales: usize, qs: usize, per_byte: usize) -> Res<()> {
    let n = rows * cols;
    if cols % BLOCK != 0 || scales != n / BLOCK || qs != n / per_byte {
        return Err(format!(
            "{rows}x{cols} needs {} scales and {} bytes, got {scales} and {qs}",
            n / BLOCK,
            n / per_byte
        )
        .into());
    }
    Ok(())
}

/// A borrowed view of those arrays, for writing a weight out.
pub enum Parts<'a> {
    F32(&'a [f32]),
    Q8 { scales: &'a [f32], qs: &'a [i8] },
    Q4 { scales: &'a [f32], qs: &'a [u8] },
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
                let (scales, qs) = (Store::Owned(scales), Store::Owned(qs));
                Weight { rows, cols, data: Data::Q8 { scales, qs } }
            }
            Precision::Q4 => {
                let mut qs = Vec::with_capacity(rows * cols / 2);
                for block in t.data.chunks_exact(BLOCK) {
                    // Four bits span the sixteen codes -8..7, and the scale
                    // must map the block's extreme value onto one end of that
                    // range. Dividing the *magnitude* by 7 looks symmetric and
                    // reads more naturally, but it never produces -8: one code
                    // in sixteen is wasted and every step is ~12% coarser than
                    // it needs to be.
                    //
                    // Dividing the *signed* extreme by -8 uses all sixteen.
                    // The sign is what makes it work: whichever end the
                    // extreme value sits at, it lands on -8 and the rest of
                    // the block spreads across the remaining codes.
                    //
                    // This cost real accuracy — the coarser version turned
                    // `17 + 25 = 42` into 40 on SmolLM2, which the fixed one
                    // gets right.
                    let mut extreme = 0.0f32;
                    for &v in block {
                        if v.abs() > extreme.abs() {
                            extreme = v;
                        }
                    }
                    let scale = extreme / -8.0;
                    let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
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
                        // `+ 8.5` rounds and applies the offset in one step.
                        let a = (l * inv + 8.5).clamp(0.0, 15.0) as u8;
                        let b = (h * inv + 8.5).clamp(0.0, 15.0) as u8;
                        qs.push((a & 0x0f) | ((b & 0x0f) << 4));
                    }
                }
                let (scales, qs) = (Store::Owned(scales), Store::Owned(qs));
                Weight { rows, cols, data: Data::Q4 { scales, qs } }
            }
            Precision::F32 => unreachable!(),
        }
    }

    /// Rebuild a weight from arrays that are already quantised.
    ///
    /// The counterpart to [`Weight::quantize`]: no rounding happens here, and
    /// the checks are the ones the file format cannot make for itself.
    pub fn from_q8(rows: usize, cols: usize, scales: Store<f32>, qs: Store<i8>) -> Res<Weight> {
        check(rows, cols, scales.len(), qs.len(), 1)?;
        Ok(Weight { rows, cols, data: Data::Q8 { scales, qs } })
    }

    pub fn from_q4(rows: usize, cols: usize, scales: Store<f32>, qs: Store<u8>) -> Res<Weight> {
        check(rows, cols, scales.len(), qs.len(), 2)?;
        Ok(Weight { rows, cols, data: Data::Q4 { scales, qs } })
    }

    pub fn from_f32(t: Tensor) -> Weight {
        Weight { rows: t.rows, cols: t.cols, data: Data::F32(t) }
    }

    /// The arrays behind this weight, ready to be written to disk.
    pub fn parts(&self) -> Parts<'_> {
        match &self.data {
            Data::F32(t) => Parts::F32(&t.data),
            Data::Q8 { scales, qs } => Parts::Q8 { scales, qs },
            Data::Q4 { scales, qs } => Parts::Q4 { scales, qs },
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
            _ if dequant_kernel() => self.matvec_bt_dequant(x),
            _ => {
                let n = self.cols;
                let blocks_per_row = n / BLOCK;
                let xq = QActivation::new(x);

                let mut out = vec![0.0f32; self.rows];
                let per = rows_per_task(self.rows);
                out.par_chunks_mut(per).enumerate().for_each(|(task, dst)| {
                    // One scratch buffer per task, reused for every row it
                    // handles, rather than 151936 of them.
                    let mut dots = vec![0i32; blocks_per_row];
                    for (j, slot) in dst.iter_mut().enumerate() {
                        let r = task * per + j;
                        {
                            let dots = &mut dots;
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
                            *slot = sums[0] + sums[1];
                        }
                    }
                });
                out
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
    /// No longer the default at any size — see [`dequant_kernel`] for the
    /// threshold that used to select it and why it was wrong. Kept as the
    /// baseline the integer path is measured against.
    fn matvec_bt_dequant(&self, x: &[f32]) -> Vec<f32> {
        let n = self.cols;
        let blocks_per_row = n / BLOCK;
        // Per-block sums of the activations, for the 4-bit offset. Depends
        // only on x, so it is computed once rather than once per row.
        let xsums: Vec<f32> = match self.data {
            Data::Q4 { .. } => x.chunks_exact(BLOCK).map(|b| b.iter().sum()).collect(),
            _ => Vec::new(),
        };

        let mut out = vec![0.0f32; self.rows];
        let per = rows_per_task(self.rows);
        out.par_chunks_mut(per).enumerate().for_each(|(task, dst)| {
            for (j, slot) in dst.iter_mut().enumerate() {
                let r = task * per + j;
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
                *slot = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            }
        });
        out
    }
}

/// Which kernel a quantised weight uses.
///
/// There used to be a size threshold here — integer dots above 32 MB of
/// weights, dequantise to f32 below it — on the measured grounds that the
/// integer path won 5x on the output head and *lost* on a 5 MB MLP matrix.
///
/// That threshold was measuring rayon, not arithmetic. Every call in that
/// benchmark paid ~0.17 ms of cold-path dispatch (see
/// [`crate::model::CpuSession::forward`]), which is invisible next to a 153 MB
/// matmul and is the entire runtime of a 5 MB one. With the dispatch gone, the
/// integer path wins everywhere it applies: 1.13x to 1.70x end to end across
/// three models and both precisions, and nothing measured slower.
///
/// So there is one kernel now. The dequantising one is kept because it is the
/// honest baseline for what the integer path buys, and `KVAD_DEQUANT=1`
/// selects it.
fn dequant_kernel() -> bool {
    matches!(std::env::var("KVAD_DEQUANT").as_deref(), Ok("1") | Ok("true"))
}


// ---------------------------------------------------------------------------
// Batched matmul: the prefill path
// ---------------------------------------------------------------------------

/// A batch of `m` activation vectors, quantised per block.
///
/// `packed` additionally holds the rows interleaved in pairs, which is the
/// operand shape `SMMLA` wants. It is built only when that kernel will run.
struct QActBatch {
    qs: Vec<i8>,
    scales: Vec<f32>,
    sums: Vec<i32>,
    packed: Vec<i8>,
}

impl QActBatch {
    fn new(xs: &[f32], m: usize, cols: usize, pack: bool) -> Self {
        let blocks = cols / BLOCK;
        let mut qs = vec![0i8; m * cols];
        let mut scales = vec![0.0f32; m * blocks];
        let mut sums = vec![0i32; m * blocks];

        for i in 0..m {
            for (b, block) in xs[i * cols..(i + 1) * cols].chunks_exact(BLOCK).enumerate() {
                let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                let scale = amax / 127.0;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                scales[i * blocks + b] = scale;

                let mut sum = 0i32;
                for (k, &v) in block.iter().enumerate() {
                    let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                    sum += q as i32;
                    qs[i * cols + b * BLOCK + k] = q;
                }
                sums[i * blocks + b] = sum;
            }
        }

        // Interleave row 2p with row 2p+1 in runs of 8, which is exactly one
        // SMMLA operand. Cheap: the batch is a few dozen rows, against weight
        // matrices of tens of thousands.
        let mut packed = Vec::new();
        if pack {
            packed = vec![0i8; (m / 2) * 2 * cols];
            for pr in 0..m / 2 {
                for b in 0..blocks {
                    for g in 0..BLOCK / 8 {
                        let dst = ((pr * blocks + b) * (BLOCK / 8) + g) * 16;
                        let src = b * BLOCK + g * 8;
                        packed[dst..dst + 8]
                            .copy_from_slice(&qs[(2 * pr) * cols + src..(2 * pr) * cols + src + 8]);
                        packed[dst + 8..dst + 16].copy_from_slice(
                            &qs[(2 * pr + 1) * cols + src..(2 * pr + 1) * cols + src + 8],
                        );
                    }
                }
            }
        }
        QActBatch { qs, scales, sums, packed }
    }
}

impl Weight {
    /// `Y = X @ Wᵀ (+ b)` for a whole batch: `xs` is `[m, cols]`, the result is
    /// `[m, rows]`.
    ///
    /// # Why batching matters more than the instruction
    ///
    /// Running `m` tokens one at a time reads the entire weight matrix `m`
    /// times. Running them together reads it once and reuses each weight `m`
    /// times, which turns a memory-bound operation into a compute-bound one.
    /// That is the bulk of the win here; `SMMLA` is a bonus on top, and only
    /// applies once the operation is a real matrix-matrix product.
    pub fn matmul_bt(&self, xs: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        self.matmul_bt_with(xs, m, bias, true)
    }

    /// As [`Weight::matmul_bt`], with the `SMMLA` kernel selectable.
    ///
    /// Passing `false` forces the portable path, which is how the tests check
    /// the hand-written assembly against something independent.
    pub fn matmul_bt_with(
        &self,
        xs: &[f32],
        m: usize,
        bias: Option<&[f32]>,
        allow_smmla: bool,
    ) -> Vec<f32> {
        debug_assert_eq!(xs.len(), m * self.cols);
        if m == 1 {
            return self.matvec_bt(xs, bias);
        }

        let (n, rows) = (self.cols, self.rows);
        let blocks = n / BLOCK;
        let use_smmla =
            allow_smmla && crate::simd::has_i8mm() && matches!(self.data, Data::Q8 { .. }) && m >= 2;

        // Both precisions fill the same transposed buffer, so the layout fix-up
        // and the bias are written once rather than per kernel.
        let mut out_t = vec![0.0f32; rows * m];
        if let Data::F32(t) = &self.data {
            gemm_bt(xs, m, t, &mut out_t);
            return Self::finish(out_t, m, rows, bias);
        }

        let act = QActBatch::new(xs, m, n, use_smmla);

        // Computed transposed — `[rows, m]` — so each thread owns a contiguous
        // run of output rows and reads each weight row exactly once.
        out_t
            .par_chunks_mut(2 * m)
            .enumerate()
            .for_each_init(
                || vec![0i32; blocks],
                |scratch, (rp, chunk)| {
                let r0 = rp * 2;
                let have_pair = chunk.len() == 2 * m;

                if use_smmla && have_pair {
                    // SAFETY: `use_smmla` checked the CPU feature, and the
                    // index arithmetic below stays inside `qs` / `packed`,
                    // whose sizes are fixed by `rows`, `cols` and `m`.
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        self.smmla_row_pair(&act, r0, m, blocks, chunk);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    unreachable!();
                    return;
                }

                // Fallback: one row at a time, in the same two passes as
                // `matvec_bt` -- integers first, scaling second. Folding the
                // scaling into the integer loop stops it vectorising, and the
                // portable path would then be compared against scalar code
                // rather than against SDOT.
                for (local, dst) in chunk.chunks_exact_mut(m).enumerate() {
                    let r = r0 + local;
                    for i in 0..m {
                        dst[i] = self.row_dot(&act, r, i, blocks, scratch);
                    }
                }
                },
            );

        Self::finish(out_t, m, rows, bias)
    }

    /// Turn the `[rows, m]` working buffer into the `[m, rows]` layout callers
    /// want, applying the bias on the way through.
    fn finish(out_t: Vec<f32>, m: usize, rows: usize, bias: Option<&[f32]>) -> Vec<f32> {
        let mut out = vec![0.0f32; m * rows];
        for r in 0..rows {
            let add = bias.map_or(0.0, |b| b[r]);
            for i in 0..m {
                out[i * rows + r] = out_t[r * m + i] + add;
            }
        }
        out
    }

    /// One output element: weight row `r` against activation row `i`.
    ///
    /// Two passes, for the reason given in [`Weight::matvec_bt`]: the integer
    /// reduction only vectorises if no floating-point work is interleaved
    /// with it.
    fn row_dot(
        &self,
        act: &QActBatch,
        r: usize,
        i: usize,
        blocks: usize,
        scratch: &mut [i32],
    ) -> f32 {
        let n = self.cols;

        // ---- pass 1: integers only ----
        match &self.data {
            Data::Q8 { qs, .. } => {
                let wrow = &qs[r * n..(r + 1) * n];
                let arow = &act.qs[i * n..(i + 1) * n];
                for (d, (wb, ab)) in scratch
                    .iter_mut()
                    .zip(wrow.chunks_exact(BLOCK).zip(arow.chunks_exact(BLOCK)))
                {
                    *d = wb.iter().zip(ab.iter()).map(|(&w, &v)| w as i32 * v as i32).sum();
                }
            }
            Data::Q4 { qs, .. } => {
                const HALF: usize = BLOCK / 2;
                let wrow = &qs[r * n / 2..(r + 1) * n / 2];
                let arow = &act.qs[i * n..(i + 1) * n];
                for (d, (wb, ab)) in scratch
                    .iter_mut()
                    .zip(wrow.chunks_exact(HALF).zip(arow.chunks_exact(BLOCK)))
                {
                    let (alo, ahi) = ab.split_at(HALF);
                    let lo: i32 =
                        wb.iter().zip(alo.iter()).map(|(&w, &v)| (w & 0x0f) as i32 * v as i32).sum();
                    let hi: i32 =
                        wb.iter().zip(ahi.iter()).map(|(&w, &v)| (w >> 4) as i32 * v as i32).sum();
                    *d = lo + hi;
                }
            }
            Data::F32(_) => unreachable!(),
        }

        // ---- pass 2: apply the scales ----
        let scales = match &self.data {
            Data::Q8 { scales, .. } | Data::Q4 { scales, .. } => scales,
            Data::F32(_) => unreachable!(),
        };
        let offset = matches!(self.data, Data::Q4 { .. });
        let (wbase, abase) = (r * blocks, i * blocks);

        let mut total = [0.0f32; 2];
        for (b, &d) in scratch.iter().enumerate() {
            let d = if offset { d - 8 * act.sums[abase + b] } else { d };
            total[b & 1] += scales[wbase + b] * act.scales[abase + b] * d as f32;
        }
        total[0] + total[1]
    }

    /// Two output rows at once, via `SMMLA`.
    ///
    /// Each instruction handles a 2x2 tile: weight rows `r0`/`r0+1` against
    /// activation rows `2p`/`2p+1`. Four independent accumulators cover the 32
    /// weights of one quantisation block — with a single accumulator the chain
    /// is latency-bound and the whole thing runs *slower* than `SDOT`.
    ///
    /// # Safety
    /// Requires `i8mm`; `dst` must be `2 * m` long.
    ///
    /// The `target_feature` attribute is load-bearing, not decoration. A
    /// `#[target_feature]` function can only be inlined into a caller that
    /// declares at least the same features — so without it here, every single
    /// `smmla` becomes a real function call and the kernel runs several times
    /// slower than the plain `SDOT` path it was meant to beat. Nothing warns
    /// about this; it just quietly loses.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "i8mm")]
    unsafe fn smmla_row_pair(
        &self,
        act: &QActBatch,
        r0: usize,
        m: usize,
        blocks: usize,
        dst: &mut [f32],
    ) {
        use crate::simd::*;
        let Data::Q8 { scales, qs } = &self.data else { unreachable!() };
        let n = self.cols;
        let w0 = qs.as_ptr().add(r0 * n);
        let w1 = qs.as_ptr().add((r0 + 1) * n);
        let (ws0, ws1) = (r0 * blocks, (r0 + 1) * blocks);

        dst.fill(0.0);
        for pr in 0..m / 2 {
            let (i0, i1) = (2 * pr, 2 * pr + 1);
            let (mut acc00, mut acc01, mut acc10, mut acc11) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

            for b in 0..blocks {
                let mut lanes4 = [zero(); BLOCK / 8];
                for (g, a) in lanes4.iter_mut().enumerate() {
                    let off = b * BLOCK + g * 8;
                    let wv = combine_rows(w0.add(off), w1.add(off));
                    let bv = load16(
                        act.packed.as_ptr().add(((pr * blocks + b) * (BLOCK / 8) + g) * 16),
                    );
                    *a = smmla(*a, wv, bv);
                }
                let total = add4(add4(lanes4[0], lanes4[1]), add4(lanes4[2], lanes4[3]));
                // [w0·x0, w0·x1, w1·x0, w1·x1]
                let l = lanes(total);
                let (xs0, xs1) =
                    (act.scales[i0 * blocks + b], act.scales[i1 * blocks + b]);
                let (a0, a1) = (scales[ws0 + b], scales[ws1 + b]);
                acc00 += a0 * xs0 * l[0] as f32;
                acc01 += a0 * xs1 * l[1] as f32;
                acc10 += a1 * xs0 * l[2] as f32;
                acc11 += a1 * xs1 * l[3] as f32;
            }
            dst[i0] = acc00;
            dst[i1] = acc01;
            dst[m + i0] = acc10;
            dst[m + i1] = acc11;
        }

        // Odd batch: the last activation row has no partner.
        if m % 2 == 1 {
            let i = m - 1;
            let mut scratch = vec![0i32; blocks];
            dst[i] = self.row_dot(act, r0, i, blocks, &mut scratch);
            dst[m + i] = self.row_dot(act, r0 + 1, i, blocks, &mut scratch);
        }
    }
}

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

    /// The hand-written SMMLA kernel must agree with the portable one.
    ///
    /// Nothing else checks that assembly: a swapped result lane or a mismatched
    /// scale produces numbers that look entirely plausible. Both paths quantise
    /// identically, so agreement here should be near-exact rather than
    /// approximate.
    #[test]
    fn smmla_kernel_matches_the_portable_path() {
        let mut rng = Rng::new(31337);
        // Odd batch sizes and an odd row count exercise the leftover handling.
        for m in [2usize, 3, 8, 9, 16] {
            let t = random_tensor(71, 128, 5);
            let xs: Vec<f32> = (0..m * 128).map(|_| rng.normal()).collect();
            let bias: Vec<f32> = (0..71).map(|_| rng.normal() * 0.1).collect();
            let w = Weight::quantize(t, Precision::Q8);

            let fast = w.matmul_bt_with(&xs, m, Some(&bias), true);
            let slow = w.matmul_bt_with(&xs, m, Some(&bias), false);
            for (i, (a, b)) in slow.iter().zip(fast.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "m={m} element {i}: smmla {b} vs portable {a}"
                );
            }
        }
    }

    /// Batching must not change the answer beyond quantisation noise.
    ///
    /// The tolerance is loose on purpose: below the size threshold a single
    /// row takes the dequantising kernel, which leaves the activations in f32,
    /// while the batched path quantises them. The two genuinely differ by
    /// about a percent — that is the cost of activation quantisation, not a
    /// bug.
    #[test]
    fn batched_matmul_agrees_with_row_by_row() {
        let mut rng = Rng::new(4242);
        for m in [1usize, 2, 5] {
            let t = random_tensor(70, 128, 5);
            let xs: Vec<f32> = (0..m * 128).map(|_| rng.normal()).collect();

            for (precision, tol) in
                [(Precision::F32, 1e-4f32), (Precision::Q8, 0.05), (Precision::Q4, 0.4)]
            {
                let w = Weight::quantize(t.clone(), precision);
                let batched = w.matmul_bt(&xs, m, None);
                for i in 0..m {
                    let single = w.matvec_bt(&xs[i * 128..(i + 1) * 128], None);
                    let scale = single.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                    for r in 0..70 {
                        let (a, b) = (single[r], batched[i * 70 + r]);
                        assert!(
                            (a - b).abs() <= tol * scale.max(1e-3),
                            "{precision} m={m} row {i} out {r}: batched {b} vs single {a}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn odd_widths_fall_back_to_f32() {
        // 100 is not a multiple of BLOCK, so this stays f32 rather than
        // silently corrupting the tail.
        let w = Weight::quantize(random_tensor(4, 100, 2), Precision::Q4);
        assert_eq!(w.precision(), Precision::F32);
    }
}
