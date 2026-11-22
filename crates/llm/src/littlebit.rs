//! LittleBit: a weight matrix as two matrices of signs and four vectors.
//!
//! # The idea
//!
//! [`crate::quant`] makes a weight smaller by storing each number in fewer
//! bits. That runs out at one bit: a sign, and nothing else. LittleBit
//! (Lee, Kim, You & Kim, *LittleBit: Ultra Low-Bit Quantization via Latent
//! Factorization*, NeurIPS 2025, arXiv 2506.13771) gets *below* one bit per
//! weight by not storing the matrix at all. It stores a low-rank factorisation
//! of it, and quantises the factors to signs:
//!
//! ```text
//!   W  ≈  diag(u1) · U · diag(u2) · diag(v1) · V · diag(v2)
//!
//!   W   [out, in]   the weight the model was trained with
//!   U   [out, s]    ±1, one bit each
//!   V   [s,  in]    ±1, one bit each
//!   u1  [out]       a scale per output row
//!   u2, v1  [s]     a scale per latent dimension, from each side
//!   v2  [in]        a scale per input column
//! ```
//!
//! `s` is the rank, and it is the knob. A 4096 x 4096 projection at rank 544
//! holds `544 * (4096 + 4096)` sign bits — 4.5 million — against 16.8 million
//! weights: about a quarter of a bit each, before the vectors. The
//! checkpoints published so far add a second, *residual* factorisation of
//! what the first one missed (`U_R`, `V_R`, `u1_R`, ...) and sum the two, at
//! the same rank, for twice the bits.
//!
//! The four vectors are where the magnitudes went. A sign matrix knows only
//! direction; `v2` restores how much each input column matters, `u1` how loud
//! each output row is, and `u2 · v1` how much each of the `s` latent
//! directions carries. They are a rounding error in size — `out + 2s + in`
//! numbers against `s * (out + in)` bits — and they are what makes a matrix
//! of signs usable at all.
//!
//! # Running it
//!
//! Never rebuild `W`. Applied right to left, the factorisation is a chain of
//! cheap steps, and the expensive two are sums of signed values:
//!
//! ```text
//!   t = x ⊙ v2                   in     multiplies
//!   z = V · t                    s·in   signed adds      ← no multiplies
//!   z = z ⊙ (v1 ⊙ u2)            s      multiplies
//!   y = U · z                    out·s  signed adds      ← no multiplies
//!   y = y ⊙ u1                   out    multiplies
//! ```
//!
//! A product with a ±1 is not a multiplication: it is the value with its sign
//! bit flipped, or not. [`signed_sum`] does exactly that — an XOR into the
//! float's top bit, then an add — which is cheaper than any of the integer
//! dot products in `quant.rs` and needs no quantised copy of the activation.
//!
//! # What this does to the cost of a token
//!
//! Decoding one token through a q8 matrix is memory-bound: every weight is
//! read for one multiply-add (see the top of [`crate::quant`]). At rank `s`
//! the same projection reads `s * (in + out) / 8` bytes and does
//! `s * (in + out)` adds. For Qwen3-8B's MLP at rank 824 that is 1.7 MB and
//! 13.5 million adds per path, against 57 MB and 50 million multiply-adds for
//! the dense matrix at q8. Two paths, and the bytes still fall sixteen-fold;
//! the arithmetic falls by less than two. A LittleBit model is compute-bound on hardware where the dense
//! one was waiting on memory, which is a different machine to optimise for.
//!
//! # What is here, and what is not
//!
//! This module *runs* LittleBit weights. Making them is a separate job, and
//! most of it is training: an SVD of every weight to start from, the
//! LittleBit-2 rotation (Joint-ITQ, arXiv 2603.00042) that aligns those
//! factors with the corners of the hypercube before they are rounded to signs,
//! and then quantisation-aware distillation from the original model, for
//! epochs, on a GPU. Without that last step the factors are a poor
//! approximation; it is the step the method is mostly made of. See
//! `docs/littlebit-plan.md`.
//!
//! The reference implementation is <https://github.com/SamsungLabs/LittleBit>,
//! licensed CC BY-NC 4.0. Nothing here is copied from it — this is the
//! arithmetic in the paper, written against the file format its `state_dict`
//! produces — but the checkpoints it trains carry that licence, so a
//! LittleBit model is a non-commercial model whatever runs it.

use crate::qcache::Store;
use crate::weights::Checkpoint;
use rayon::prelude::*;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Signs per stored word.
///
/// The reference packs 32 to an `int32`, least significant bit first, which is
/// the order a shift-and-mask walks them in. Kept as the in-memory layout too:
/// a checkpoint's words are this engine's words, and loading one is a copy.
pub const WORD: usize = 32;

/// Rows one task should handle — one chunk per thread, as in
/// [`crate::quant`], and for the same reason.
#[inline]
fn rows_per_task(rows: usize) -> usize {
    rows.div_ceil(rayon::current_num_threads().max(1)).max(1)
}

// ---------------------------------------------------------------------------
// A matrix of signs
// ---------------------------------------------------------------------------

/// A `rows x cols` matrix of ±1, one bit each.
///
/// A set bit is **−1** and a clear bit is +1. That is the reference's
/// convention, and it is the convenient one: zero is the identity for XOR, so
/// a clear bit leaves a float alone. A row whose length is not a multiple of
/// [`WORD`] is padded with clear bits, which [`signed_sum`] multiplies by
/// zero-padded activations, so padding never needs a special case.
pub struct Signs {
    rows: usize,
    cols: usize,
    bits: Store<u32>,
}

impl Signs {
    pub fn new(rows: usize, cols: usize, bits: Store<u32>) -> Res<Signs> {
        let want = rows * cols.div_ceil(WORD);
        if bits.len() != want {
            return Err(format!(
                "a {rows}x{cols} sign matrix is {want} words, and {} were given",
                bits.len()
            )
            .into());
        }
        Ok(Signs { rows, cols, bits })
    }

    /// Pack signs from a predicate: `negative(r, c)` is true where the entry
    /// is −1.
    pub fn pack(rows: usize, cols: usize, negative: impl Fn(usize, usize) -> bool) -> Signs {
        let words = cols.div_ceil(WORD);
        let mut bits = vec![0u32; rows * words];
        for r in 0..rows {
            for c in 0..cols {
                if negative(r, c) {
                    bits[r * words + c / WORD] |= 1 << (c % WORD);
                }
            }
        }
        Signs { rows, cols, bits: Store::Owned(bits) }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Words per row.
    pub fn words(&self) -> usize {
        self.cols.div_ceil(WORD)
    }

    /// Every word, row after row — for writing to disk.
    pub fn bits(&self) -> &[u32] {
        &self.bits
    }

    fn row(&self, r: usize) -> &[u32] {
        let w = self.words();
        &self.bits[r * w..(r + 1) * w]
    }

    /// One entry, as the ±1 it stands for.
    pub fn sign(&self, r: usize, c: usize) -> f32 {
        if self.row(r)[c / WORD] >> (c % WORD) & 1 == 1 {
            -1.0
        } else {
            1.0
        }
    }

    pub fn bytes(&self) -> usize {
        self.bits.len() * 4
    }

    fn will_need(&self) {
        self.bits.will_need();
    }
}

/// `Σ s_k · x_k` for one row of signs, where `x` is padded with zeros to a
/// whole number of words.
///
/// The multiply is gone. Flipping a float's sign is XOR-ing its top bit, so
/// each term is the activation's bits XOR the weight's bit moved up to bit 31,
/// and the sum is an ordinary float sum of those.
///
/// Thirty-two accumulators, one per bit position, for two reasons. A single
/// running sum is a dependency chain the CPU cannot overlap: every add waits
/// for the one before. And keeping lane `j` on bit `j` of every word makes the
/// bit each lane wants a constant, which is what lets the compiler turn the
/// inner loop into a few vector instructions on 4, 8 or 16 lanes at once.
///
/// How the bit becomes a sign is the one thing that differs by machine, and
/// it was measured rather than guessed (824 rows of 4096, one core):
///
/// | form | x86-64 baseline (SSE2) | x86-64 with AVX-512 |
/// |---|---|---|
/// | shift: `(w >> j & 1) << 31` | 3.1 G adds/s | 24.9 G adds/s |
/// | mask: `w & (1 << j) != 0` | 5.1 G adds/s | 22.5 G adds/s |
///
/// SSE2 has no per-lane shift, so the shift form is emulated and the mask
/// form — AND with a constant, compare, select — wins by 1.6x. NEON has one
/// (`ushl`), and there the shift form is four vector instructions per four
/// lanes to the mask form's five; aarch64 gets the shift.
#[inline]
pub fn signed_sum(words: &[u32], x: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), words.len() * WORD);
    let mut acc = [0.0f32; WORD];
    for (&w, xs) in words.iter().zip(x.chunks_exact(WORD)) {
        for j in 0..WORD {
            // Bit j of the word, moved to the float's sign position.
            #[cfg(target_arch = "aarch64")]
            let flip = (w >> j & 1) << 31;
            #[cfg(not(target_arch = "aarch64"))]
            let flip = if w & (1 << j) != 0 { 1 << 31 } else { 0 };
            acc[j] += f32::from_bits(xs[j].to_bits() ^ flip);
        }
    }
    // Pairwise, so the tail of a fast loop is not a slow one.
    let mut width = WORD;
    while width > 1 {
        width /= 2;
        for j in 0..width {
            acc[j] += acc[j + width];
        }
    }
    acc[0]
}

// ---------------------------------------------------------------------------
// One factorisation
// ---------------------------------------------------------------------------

/// One rank-`s` factorisation: `diag(u1) · U · diag(mid) · V · diag(v2)`.
///
/// `mid` is `u2 ⊙ v1`. The checkpoint stores the two separately because
/// training learns them separately — one belongs to each factor — but at
/// inference they are always applied together, so they are multiplied once
/// here and never again.
pub struct Path {
    u: Signs,
    v: Signs,
    u1: Store<f32>,
    mid: Store<f32>,
    v2: Store<f32>,
}

impl Path {
    /// From the arrays as a checkpoint stores them.
    pub fn new(u: Signs, v: Signs, u1: Vec<f32>, u2: &[f32], v1: &[f32], v2: Vec<f32>) -> Res<Path> {
        if u2.len() != v1.len() {
            return Err(format!("u2 has {} entries and v1 has {}", u2.len(), v1.len()).into());
        }
        let mid = u2.iter().zip(v1).map(|(a, b)| a * b).collect();
        Path::from_parts(u, v, Store::Owned(u1), Store::Owned(mid), Store::Owned(v2))
    }

    /// From the arrays as this engine keeps them, `mid` already multiplied
    /// out — the form [`crate::qcache`] writes and maps.
    pub fn from_parts(u: Signs, v: Signs, u1: Store<f32>, mid: Store<f32>, v2: Store<f32>) -> Res<Path> {
        let (out, s, inp) = (u.rows, u.cols, v.cols);
        let fits = v.rows == s && u1.len() == out && mid.len() == s && v2.len() == inp;
        if !fits {
            return Err(format!(
                "U is {out}x{s} and V is {}x{inp}, with scales of {}, {} and {}: \
                 they do not make one factorisation",
                v.rows,
                u1.len(),
                mid.len(),
                v2.len()
            )
            .into());
        }
        Ok(Path { u, v, u1, mid, v2 })
    }

    /// The rank, `s`.
    pub fn rank(&self) -> usize {
        self.u.cols
    }

    pub fn u(&self) -> &Signs {
        &self.u
    }

    pub fn v(&self) -> &Signs {
        &self.v
    }

    pub fn u1(&self) -> &[f32] {
        &self.u1
    }

    pub fn mid(&self) -> &[f32] {
        &self.mid
    }

    pub fn v2(&self) -> &[f32] {
        &self.v2
    }

    fn bytes(&self) -> usize {
        self.u.bytes() + self.v.bytes() + 4 * (self.u1.len() + self.mid.len() + self.v2.len())
    }

    fn will_need(&self) {
        self.u.will_need();
        self.v.will_need();
        self.u1.will_need();
        self.mid.will_need();
        self.v2.will_need();
    }
}

// ---------------------------------------------------------------------------
// A weight: one factorisation, or a sum of them
// ---------------------------------------------------------------------------

/// A weight matrix stored as LittleBit factors: the sum of one or more
/// [`Path`]s, all the same shape.
///
/// One path is the method as first published. Two is "residual" LittleBit,
/// where the second factorises what the first left over — every checkpoint
/// published so far has it.
pub struct Factored {
    rows: usize,
    cols: usize,
    paths: Vec<Path>,
}

impl Factored {
    pub fn new(paths: Vec<Path>) -> Res<Factored> {
        let first = paths.first().ok_or("a factored weight needs at least one factorisation")?;
        let (rows, cols) = (first.u.rows, first.v.cols);
        if let Some(p) = paths.iter().find(|p| (p.u.rows, p.v.cols) != (rows, cols)) {
            return Err(format!(
                "a {rows}x{cols} weight cannot be summed with a {}x{} one",
                p.u.rows, p.v.cols
            )
            .into());
        }
        Ok(Factored { rows, cols, paths })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn paths(&self) -> &[Path] {
        &self.paths
    }

    /// Bytes held: the sign words and the four vectors of every path.
    pub fn bytes(&self) -> usize {
        self.paths.iter().map(Path::bytes).sum()
    }

    /// What this weight costs per entry of the matrix it stands for, counting
    /// the vectors at the 16 bits a checkpoint stores them in, and `u2 · v1`
    /// once, as the single vector it becomes.
    ///
    /// The paper's accounting, so a checkpoint's own `eff_bit` can be checked
    /// against it.
    pub fn bits_per_weight(&self) -> f64 {
        let bits: usize = self
            .paths
            .iter()
            .map(|p| p.rank() * (self.rows + self.cols) + 16 * (self.rows + p.rank() + self.cols))
            .sum();
        bits as f64 / (self.rows * self.cols) as f64
    }

    pub fn will_need(&self) {
        self.paths.iter().for_each(Path::will_need);
    }

    /// `y = W x`.
    ///
    /// Two parallel sections, whatever the number of paths: every path's `V`
    /// rows in the first, every output row — summed across paths — in the
    /// second. Each section costs a dispatch, and a residual model would
    /// otherwise pay four per projection.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);

        // `x ⊙ v2` per path, padded to whole words.
        let ts: Vec<Vec<f32>> = self
            .paths
            .iter()
            .map(|p| {
                let mut t = vec![0.0f32; p.v.words() * WORD];
                for ((t, &x), &s) in t.iter_mut().zip(x).zip(p.v2.iter()) {
                    *t = x * s;
                }
                t
            })
            .collect();

        // z = (V t) ⊙ mid, every path's rows in one index space.
        let starts: Vec<usize> = self
            .paths
            .iter()
            .scan(0, |at, p| {
                let s = *at;
                *at += p.rank();
                Some(s)
            })
            .collect();
        let total: usize = self.paths.iter().map(Path::rank).sum();
        let mut flat = vec![0.0f32; total];
        let per = rows_per_task(total);
        flat.par_chunks_mut(per).enumerate().for_each(|(task, dst)| {
            for (j, slot) in dst.iter_mut().enumerate() {
                let g = task * per + j;
                let k = starts.partition_point(|&s| s <= g) - 1;
                let (p, r) = (&self.paths[k], g - starts[k]);
                *slot = signed_sum(p.v.row(r), &ts[k]) * p.mid[r];
            }
        });
        let zs: Vec<Vec<f32>> = self
            .paths
            .iter()
            .zip(&starts)
            .map(|(p, &s)| {
                let mut z = vec![0.0f32; p.u.words() * WORD];
                z[..p.rank()].copy_from_slice(&flat[s..s + p.rank()]);
                z
            })
            .collect();

        // y = Σ (U z) ⊙ u1.
        let mut y = vec![0.0f32; self.rows];
        let per = rows_per_task(self.rows);
        y.par_chunks_mut(per).enumerate().for_each(|(task, dst)| {
            for (j, slot) in dst.iter_mut().enumerate() {
                let r = task * per + j;
                *slot = self
                    .paths
                    .iter()
                    .zip(&zs)
                    .map(|(p, z)| signed_sum(p.u.row(r), z) * p.u1[r])
                    .sum();
            }
        });
        y
    }

    /// `Y = X Wᵀ` for `m` rows of `X` — the prefill path. `xs` is `[m, cols]`
    /// and the result `[m, rows]`.
    ///
    /// The same chain as [`Factored::matvec`], a matrix at a time, and the
    /// sign products become ordinary float dot products against a row of
    /// signs unpacked to `±1.0` once and reused for all `m` activations —
    /// which is what batching buys everywhere: each weight read once, used
    /// `m` times.
    pub fn matmul(&self, xs: &[f32], m: usize) -> Vec<f32> {
        debug_assert_eq!(xs.len(), m * self.cols);
        let mut ys = vec![0.0f32; m * self.rows];
        for p in &self.paths {
            // T = X ⊙ v2, each row padded with zeros to whole tiles of V.
            let vw = tile_width(&p.v);
            let mut t = vec![0.0f32; m * vw];
            for (dst, src) in t.chunks_exact_mut(vw).zip(xs.chunks_exact(self.cols)) {
                for ((d, &x), &s) in dst.iter_mut().zip(src).zip(p.v2.iter()) {
                    *d = x * s;
                }
            }

            // Z = (T Vᵀ) ⊙ mid, rows padded to whole tiles of U.
            let z_t = sign_gemm(&p.v, &t, m, &p.mid);
            let uw = tile_width(&p.u);
            let mut z = vec![0.0f32; m * uw];
            for k in 0..p.rank() {
                for i in 0..m {
                    z[i * uw + k] = z_t[k * m + i];
                }
            }

            // Y += (Z Uᵀ) ⊙ u1.
            let y_t = sign_gemm(&p.u, &z, m, &p.u1);
            for r in 0..self.rows {
                for i in 0..m {
                    ys[i * self.rows + r] += y_t[r * m + i];
                }
            }
        }
        ys
    }

    /// Row `r` of the matrix these factors stand for.
    ///
    /// What an embedding lookup needs, and what the tests compare the fast
    /// paths against: `W[r, :] = u1[r] · Σ_k U[r, k] · mid[k] · V[k, :] ⊙ v2`.
    /// `s * cols` work, so a row and not a matrix at a time.
    pub fn row(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.cols];
        for p in &self.paths {
            let mut acc = vec![0.0f32; self.cols];
            for k in 0..p.rank() {
                let a = p.u.sign(r, k) * p.mid[k];
                for (c, v) in acc.iter_mut().enumerate() {
                    *v += a * p.v.sign(k, c);
                }
            }
            for ((o, a), s) in out.iter_mut().zip(&acc).zip(p.v2.iter()) {
                *o += p.u1[r] * a * s;
            }
        }
        out
    }
}

/// Sign rows sharing each load of an activation in [`sign_gemm`].
const TILE_ROWS: usize = 4;
/// Columns per tile in [`sign_gemm`]: a multiple of [`WORD`], so a tile is
/// whole words.
const TILE_COLS: usize = 512;
/// Accumulator lanes per row in [`sign_gemm`].
const LANES: usize = 8;

/// How wide [`sign_gemm`] wants each activation row for `signs`: its columns,
/// rounded up to a whole number of tiles and padded with zeros.
fn tile_width(signs: &Signs) -> usize {
    signs.cols.div_ceil(TILE_COLS) * TILE_COLS
}

/// `out_t[r, i] = scale[r] · Σ_c signs[r, c] · xs[i, c]` — a batch through a
/// sign matrix, transposed so that each thread owns whole weight rows.
///
/// `xs` is `[m, tile_width(signs)]`, zero beyond the matrix's columns.
///
/// # How it is shaped, and why
///
/// The obvious loop — unpack a row of signs to `±1.0`, call
/// [`crate::tensor::dot`] against each of the `m` activations, next row —
/// runs at 3.3 billion multiply-adds a second on one core here. Each dot
/// product has eight lanes of accumulator, which is two SSE registers or two
/// NEON ones, and an add takes four cycles to come back: two chains in flight
/// cannot keep the adders busy.
///
/// So [`TILE_ROWS`] sign rows go through together. A [`TILE_COLS`]-wide slice
/// of each is unpacked into a tile that stays in L1, and every activation's
/// matching slice is run against all of them at once, [`LANES`] lanes each:
/// eight independent chains, and every load of `x` feeds four rows.
///
/// Measured on one core of this machine (x86-64, SSE2), at the shapes of
/// Qwen3-8B's projections and 20 to 64 tokens:
///
/// | | multiply-adds a second |
/// |---|---|
/// | row by row, through `dot` | 3.0 – 3.8 G |
/// | tiled for cache, still through `dot` | 3.4 – 3.8 G |
/// | 4 rows x 8 lanes, tiles of runtime width | 3.2 – 5.9 G |
/// | 4 rows x 8 lanes, tiles of fixed width, as here | 14.8 – 19.6 G |
///
/// The second row was the obvious fix — the activations were being streamed
/// past every weight row — and bought nothing: the loads were not what the
/// loop waited for. The third and fourth rows are the same arithmetic. The
/// difference is whether the compiler can see that every index is in bounds:
/// with a runtime width it could not always, and then it either kept the
/// checks or vectorised across the four *rows* instead of along them —
/// shuffling four loads into one register for every column, which is slower
/// than not vectorising at all. Which of those it did changed with the code
/// *around* the loop. Fixed-size arrays take the decision away from it: the
/// activations are padded to whole tiles, the padding is zero, and the loop
/// has no lengths left to reason about.
fn sign_gemm(signs: &Signs, xs: &[f32], m: usize, scale: &[f32]) -> Vec<f32> {
    let width = tile_width(signs);
    let words = signs.words();
    debug_assert_eq!(xs.len(), m * width);
    let mut out_t = vec![0.0f32; signs.rows * m];
    // Whole blocks of rows per task, so no block straddles two threads.
    let per = rows_per_task(signs.rows).div_ceil(TILE_ROWS) * TILE_ROWS;
    out_t.par_chunks_mut(per * m).enumerate().for_each(|(task, dst)| {
        let mut tile = [[0.0f32; TILE_COLS]; TILE_ROWS];
        for (b, outs) in dst.chunks_mut(TILE_ROWS * m).enumerate() {
            let r0 = task * per + b * TILE_ROWS;
            let rows = outs.len() / m;
            for c0 in (0..width).step_by(TILE_COLS) {
                // Unpack this slice of each row. Past the last row or the
                // last word, zeros: they contribute nothing, and they keep
                // the inner loop one shape.
                for (k, dst) in tile.iter_mut().enumerate() {
                    for (wi, d) in dst.chunks_exact_mut(WORD).enumerate() {
                        let at = c0 / WORD + wi;
                        if k >= rows || at >= words {
                            d.fill(0.0);
                            continue;
                        }
                        let w = signs.row(r0 + k)[at];
                        for (j, v) in d.iter_mut().enumerate() {
                            *v = if w >> j & 1 == 1 { -1.0 } else { 1.0 };
                        }
                    }
                }
                for i in 0..m {
                    let x: &[f32; TILE_COLS] =
                        xs[i * width + c0..i * width + c0 + TILE_COLS].try_into().expect("a whole tile");
                    let sums = dot_tile(&tile, x);
                    for (k, s) in sums.iter().enumerate().take(rows) {
                        outs[k * m + i] += s;
                    }
                }
            }
        }
        for (k, outs) in dst.chunks_mut(m).enumerate() {
            let s = scale[task * per + k];
            outs.iter_mut().for_each(|o| *o *= s);
        }
    });
    out_t
}

/// One activation slice against a tile of sign rows: [`TILE_ROWS`] dot
/// products of [`TILE_COLS`], [`LANES`] lanes each.
#[inline(always)]
fn dot_tile(tile: &[[f32; TILE_COLS]; TILE_ROWS], x: &[f32; TILE_COLS]) -> [f32; TILE_ROWS] {
    let mut acc = [[0.0f32; LANES]; TILE_ROWS];
    for c in (0..TILE_COLS).step_by(LANES) {
        for l in 0..LANES {
            let xv = x[c + l];
            for (a, t) in acc.iter_mut().zip(tile) {
                a[l] += t[c + l] * xv;
            }
        }
    }
    acc.map(|a| ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])))
}

// ---------------------------------------------------------------------------
// Reading a checkpoint
// ---------------------------------------------------------------------------

/// The factors of the weight that would have been called `{stem}.weight`, if
/// the checkpoint holds them; `Ok(None)` if it does not.
///
/// The reference's `state_dict` replaces every quantised `nn.Linear`'s
/// `weight` with, per path (the residual path's names end in `_R`):
///
/// ```text
///   {stem}.U_packed   int32 [out, ⌈s/32⌉]    U's signs, 32 to a word
///   {stem}.U_shape    int64 [2]             [out, s]
///   {stem}.V_packed   int32 [s, ⌈in/32⌉]
///   {stem}.V_shape    int64 [2]             [s, in]
///   {stem}.u1  [1, out]   {stem}.u2  [1, s]   {stem}.v1  [1, s]   {stem}.v2  [1, in]
/// ```
///
/// plus three bookkeeping scalars — the target and achieved bits per weight
/// and the rank — that the factors already imply and nothing here reads.
///
/// Older checkpoints, from before the reference learned to pack, store `U`
/// and `V` as the full-precision latent matrices training left behind, and
/// apply `sign()` at every forward pass. Those are read too, by taking the
/// sign once: zero counts as +1, as it does there.
pub fn read(ckpt: &Checkpoint, stem: &str) -> Res<Option<Factored>> {
    let mut paths = Vec::new();
    for suffix in ["", "_R"] {
        match read_path(ckpt, stem, suffix)? {
            Some(p) => paths.push(p),
            // A residual path without a main one would be a checkpoint this
            // does not understand; the main path alone is the original method.
            None if suffix.is_empty() => return Ok(None),
            None => break,
        }
    }
    Factored::new(paths).map(Some)
}

/// Whether `{stem}` is stored as LittleBit factors, without reading them.
pub fn present(ckpt: &Checkpoint, stem: &str) -> bool {
    ckpt.resolve(&format!("{stem}.U_packed")).is_some() || ckpt.resolve(&format!("{stem}.U")).is_some()
}

fn read_path(ckpt: &Checkpoint, stem: &str, suffix: &str) -> Res<Option<Path>> {
    let name = |what: &str| format!("{stem}.{what}{suffix}");
    let Some(u) = signs(ckpt, &name("U"))? else { return Ok(None) };
    let v = signs(ckpt, &name("V"))?.ok_or_else(|| format!("`{}` has a U and no V", name("")))?;
    let vector = |what: &str| ckpt.get_flat(&name(what));
    Path::new(u, v, vector("u1")?, &vector("u2")?, &vector("v1")?, vector("v2")?)
        .map(Some)
        .map_err(|e| format!("`{stem}`: {e}").into())
}

/// A sign matrix, packed or not.
///
/// `name` is `{stem}.U` or `{stem}.U_R`; the packed form is filed as
/// `{stem}.U_packed` and `{stem}.U_R_packed`, so the suffix goes after it.
fn signs(ckpt: &Checkpoint, name: &str) -> Res<Option<Signs>> {
    let packed = format!("{name}_packed");
    if let Some((_, words)) = ckpt.read_words(&packed)? {
        let shape = format!("{name}_shape");
        let (_, dims) = ckpt.read_words(&shape)?.ok_or_else(|| format!("`{packed}` has no `{shape}`"))?;
        // An int64 [2] arrives as four 32-bit words, low half first.
        let [rows, _, cols, _] = dims[..] else {
            return Err(format!("`{shape}` should be two int64s").into());
        };
        return Signs::new(rows as usize, cols as usize, Store::Owned(words))
            .map(Some)
            .map_err(|e| format!("`{packed}`: {e}").into());
    }
    // The unpacked form: the latent matrix itself.
    let Some(t) = ckpt.try_get(name) else { return Ok(None) };
    Ok(Some(Signs::pack(t.rows, t.cols, |r, c| t.data[r * t.cols + c] < 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator, so a failure reproduces.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn path(rng: &mut Lcg, out: usize, s: usize, inp: usize) -> Path {
        let u: Vec<f32> = rng.vec(out * s);
        let v: Vec<f32> = rng.vec(s * inp);
        Path::new(
            Signs::pack(out, s, |r, c| u[r * s + c] < 0.0),
            Signs::pack(s, inp, |r, c| v[r * inp + c] < 0.0),
            rng.vec(out),
            &rng.vec(s),
            &rng.vec(s),
            rng.vec(inp),
        )
        .unwrap()
    }

    /// The matrix the factors stand for, built the slow, obvious way: every
    /// entry as the sum it is defined to be.
    fn dense(f: &Factored) -> Vec<f32> {
        let mut w = vec![0.0f32; f.rows * f.cols];
        for p in &f.paths {
            for r in 0..f.rows {
                for c in 0..f.cols {
                    let mut sum = 0.0f64;
                    for k in 0..p.rank() {
                        sum += (p.u.sign(r, k) * p.mid[k] * p.v.sign(k, c)) as f64;
                    }
                    w[r * f.cols + c] += (p.u1[r] as f64 * sum * p.v2[c] as f64) as f32;
                }
            }
        }
        w
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()), "at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn a_set_bit_is_minus_one_and_the_first_column_is_the_lowest_bit() {
        // Row 0 is -1 at columns 0 and 33; row 1 only at column 31.
        let s = Signs::new(2, 40, Store::Owned(vec![1, 1 << 1, 1 << 31, 0])).unwrap();
        assert_eq!(s.sign(0, 0), -1.0);
        assert_eq!(s.sign(0, 1), 1.0);
        assert_eq!(s.sign(0, 33), -1.0);
        assert_eq!(s.sign(1, 31), -1.0);
        assert_eq!(s.sign(1, 0), 1.0);
        assert!(Signs::new(2, 40, Store::Owned(vec![0; 3])).is_err());
    }

    #[test]
    fn a_signed_sum_is_the_dot_product_with_the_signs() {
        let mut rng = Lcg(7);
        for cols in [32, 64, 96, 208, 544] {
            let signs = Signs::pack(1, cols, |_, c| (c * 7 + 3) % 5 < 2);
            let mut x = rng.vec(signs.words() * WORD);
            x[cols..].iter_mut().for_each(|v| *v = 0.0);
            let want: f32 = (0..cols).map(|c| signs.sign(0, c) * x[c]).sum();
            let got = signed_sum(signs.row(0), &x);
            assert!((got - want).abs() < 1e-4, "{cols}: {got} vs {want}");
        }
    }

    /// The fast paths against the definition, at ranks that are not a whole
    /// number of words — 208 and 824 are what Qwen3-8B's checkpoint uses, and
    /// the padding is where an off-by-one would hide.
    #[test]
    fn matvec_and_matmul_compute_the_matrix_the_factors_stand_for() {
        let mut rng = Lcg(42);
        for (out, s, inp, residual) in [(40, 13, 64, false), (33, 45, 96, true), (64, 72, 70, true)] {
            let mut paths = vec![path(&mut rng, out, s, inp)];
            if residual {
                paths.push(path(&mut rng, out, s, inp));
            }
            let f = Factored::new(paths).unwrap();
            let w = dense(&f);

            let x = rng.vec(inp);
            let want: Vec<f32> =
                (0..out).map(|r| (0..inp).map(|c| w[r * inp + c] * x[c]).sum()).collect();
            close(&f.matvec(&x), &want);

            for r in [0, out / 2, out - 1] {
                close(&f.row(r), &w[r * inp..(r + 1) * inp]);
            }

            let m = 5;
            let xs = rng.vec(m * inp);
            let ys = f.matmul(&xs, m);
            for i in 0..m {
                close(&ys[i * out..(i + 1) * out], &f.matvec(&xs[i * inp..(i + 1) * inp]));
            }
        }
    }

    #[test]
    fn bits_per_weight_is_the_papers_accounting() {
        // The reference's formula for a residual layer, with Qwen3-8B's
        // q_proj: rank 544 on a 4096 x 4096 matrix.
        let (a, b, s) = (4096usize, 4096usize, 544usize);
        let theirs = (s * 2 * (a + b + 16) + 32 * (a + b)) as f64 / (a * b) as f64;
        let mut rng = Lcg(1);
        let small = |rng: &mut Lcg| {
            Path::from_parts(
                Signs::pack(b, s, |_, _| false),
                Signs::pack(s, a, |_, _| false),
                Store::Owned(rng.vec(b)),
                Store::Owned(rng.vec(s)),
                Store::Owned(rng.vec(a)),
            )
            .unwrap()
        };
        let f = Factored::new(vec![small(&mut rng), small(&mut rng)]).unwrap();
        assert!((f.bits_per_weight() - theirs).abs() < 1e-9);
        // The checkpoint stores it as `_eff_bit_actual`, in bf16: 0.5479
        // with its mantissa cut to seven bits.
        let bf16 = f32::from_bits((theirs as f32).to_bits() & 0xffff_0000);
        assert_eq!(bf16, 0.546875);
    }

    #[test]
    fn mismatched_factors_are_refused() {
        let mut rng = Lcg(3);
        let a = path(&mut rng, 8, 4, 32);
        let b = path(&mut rng, 8, 4, 64);
        assert!(Factored::new(vec![a, b]).is_err());
        assert!(Path::from_parts(
            Signs::pack(8, 4, |_, _| false),
            Signs::pack(5, 32, |_, _| false),
            Store::Owned(vec![0.0; 8]),
            Store::Owned(vec![0.0; 4]),
            Store::Owned(vec![0.0; 32]),
        )
        .is_err());
    }
}
