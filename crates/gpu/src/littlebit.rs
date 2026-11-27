//! LittleBit weights on Metal: two matrices of signs and four vectors per
//! factorisation, multiplied without ever rebuilding the matrix.
//!
//! The format, the arithmetic and why it works are in `kvad::littlebit`; this
//! is the same chain on the GPU,
//!
//! ```text
//!   z = (V · (x ⊙ v2)) ⊙ mid          one kernel
//!   y = (U · z) ⊙ u1  (+ y so far)     one kernel
//! ```
//!
//! per factorisation, so a residual checkpoint's projection is four
//! dispatches: `v2` is applied as the activations are loaded, and the second
//! path's last kernel adds the first path's answer as it writes its own.
//!
//! # The kernel
//!
//! [`sign_matmul`]: `y[i, r] = scale[r] · Σ_c sign(r, c) · x[i, c]`, one
//! simdgroup an output row. Each lane takes a word of 32 signs at a time and
//! the 32 activations under it, four to a vector, flips each one's sign bit by
//! XOR where the weight is −1, and adds. `simd_sum` joins the lanes. There is
//! no multiply in the loop, as there is none on the CPU (`signed_sum`).
//!
//! That kernel takes the rows of `x` — a prompt's tokens — one at a time,
//! so a prefill costs a decode step a row. From [`TILED_ROWS`] rows,
//! `sign_matmul_tiled` takes the product as a matrix product: a 32 x 32 tile
//! of `y` a threadgroup, a word of each of 32 sign rows spread into ±1 floats
//! in threadgroup memory and multiplied on f32 simdgroup matrices. A short
//! prompt against rank 544 is a few dozen tiles, too few for the GPU, so the
//! words are split across up to [`GROUPS`] threadgroups and
//! `sign_matmul_sum` adds the parts in order.
//!
//! # Measured
//!
//! On an M5 Pro, the 252 projections of a Qwen3-8B token at 0.55 bits a
//! weight (`bench` below, random factors at the real shapes): 21 ms for one
//! row on the row kernel, against about 90 ms for the CPU engine's whole
//! token, and 12 ms a row however many. The tiled kernel costs 58 ms for one
//! to eight rows, 64 for 27, 200 for 128 and 817 for 512: 1.6 ms a row, about
//! 6 TFLOP/s of f32. They cross at four rows. On the real checkpoint a
//! 19-token prompt reads in 0.09 s instead of 0.24, and 501 tokens in 0.80 s
//! instead of 5.66, with the same text.
//!
//! The activations are f32, whatever the load's dtype: the CPU engine's
//! kernels are f32, and so are the quantised loads' activations here. A bf16
//! load casts on the way in and out.
//!
//! The reference implementation is CC BY-NC 4.0; nothing here is from it.

use crate::fused::metal::{buffer, output};
use candle_core::backend::BackendStorage;
use candle_core::{CustomOp1, DType, Device, Layout, MetalStorage, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder};
use kvad::littlebit::{Factored, WORD};
use objc2_metal::MTLSize;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params { uint rows; uint cols; uint flags; };

// Four activations from `c`, zero past the end of the row; times `pre` there
// when there is one.
static inline float4 load4(device const float *x, device const float *pre, uint c, uint cols, bool has_pre) {
    float4 v = 0;
    if (c + 4 <= cols) {
        v = float4(*(device const packed_float4 *)(x + c));
        if (has_pre) v *= float4(*(device const packed_float4 *)(pre + c));
    } else {
        for (uint j = 0; c + j < cols; j++) v[j] = x[c + j] * (has_pre ? pre[c + j] : 1.0f);
    }
    return v;
}

// One simdgroup an output row, one threadgroup row of the grid a row of x.
kernel void sign_matmul(
    device const uint *signs [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device const float *pre [[buffer(3)]],
    device const float *add [[buffer(4)]],
    device float *y [[buffer(5)]],
    constant Params &p [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint r = tg.x * sgs + sg;
    if (r >= p.rows) return;
    bool has_pre = p.flags & 1;
    uint words = (p.cols + 31) / 32;
    device const uint *row = signs + r * words;
    device const float *xi = x + tg.y * p.cols;
    float4 acc = 0;
    for (uint w = lane; w < words; w += 32) {
        uint b = row[w];
        for (uint k = 0; k < 32; k += 4) {
            uint c = w * 32 + k;
            if (c >= p.cols) break;
            float4 v = load4(xi, pre, c, p.cols, has_pre);
            // A set bit is -1: move it up to the float's sign bit.
            uint4 bit = (uint4(b) >> (uint4(0, 1, 2, 3) + k)) & 1u;
            acc += as_type<float4>(as_type<uint4>(v) ^ (bit << 31));
        }
    }
    float s = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0) {
        uint o = tg.y * p.rows + r;
        y[o] = s * scale[r] + ((p.flags & 2) ? add[o] : 0.0f);
    }
}

// The same product for many rows of x: a 32 x 32 tile of y a threadgroup, 32
// rows of x against 32 rows of signs, a word of each sign row at a time. The
// word is spread into +-1 floats in threadgroup memory and multiplied on the
// simdgroup matrices; four simdgroups take a 16 x 16 quarter each.
//
// The grid's third axis splits the words, when there are too few tiles to
// keep the GPU busy: each split writes its sums to `sums`, and
// `sign_matmul_sum` adds them, in order, so the answer is the same every run.
#define T 32
struct Tiled { uint m; uint per_split; uint splits; };
kernel void sign_matmul_tiled(
    device const uint *signs [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device const float *pre [[buffer(3)]],
    device const float *add [[buffer(4)]],
    device float *y [[buffer(5)]],
    constant Params &p [[buffer(6)]],
    constant Tiled &tl [[buffer(7)]],
    device float *sums [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[T * T];   // [row of x][column]
    threadgroup float ss[T * T];   // [column][row of signs], so B as it is multiplied
    bool has_pre = p.flags & 1;
    uint m = tl.m;
    uint words = (p.cols + 31) / 32;
    uint w0 = tg.z * tl.per_split, w1 = min(words, w0 + tl.per_split);
    uint r0 = tg.x * T, i0 = tg.y * T;
    // Each thread loads eight of the tile's 1024 activations and spreads
    // eight bits of one row's word.
    uint row = t / 4, part = (t % 4) * 8;
    uint i = i0 + row, r = r0 + row;
    device const float *xi = x + i * p.cols;
    uint sy = (sg / 2) * 16, sx = (sg % 2) * 16;
    // Constant indices only: a fragment array indexed by a variable spills.
    simdgroup_float8x8 acc[4] = { make_filled_simdgroup_matrix<float, 8, 8>(0.0f), make_filled_simdgroup_matrix<float, 8, 8>(0.0f),
                                  make_filled_simdgroup_matrix<float, 8, 8>(0.0f), make_filled_simdgroup_matrix<float, 8, 8>(0.0f) };

    for (uint w = w0; w < w1; w++) {
        uint c = w * 32 + part;
        float4 a = 0, b = 0;
        if (i < m) {
            a = load4(xi, pre, c, p.cols, has_pre);
            b = load4(xi, pre, c + 4, p.cols, has_pre);
        }
        *(threadgroup float4 *)(xs + row * T + part) = a;
        *(threadgroup float4 *)(xs + row * T + part + 4) = b;
        uint bits = r < p.rows ? signs[r * words + w] >> part : 0u;
        for (uint k = 0; k < 8; k++) ss[(part + k) * T + row] = (bits >> k) & 1u ? -1.0f : 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < T; k += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, xs + sy * T + k, T);
            simdgroup_load(a1, xs + (sy + 8) * T + k, T);
            simdgroup_load(b0, ss + k * T + sx, T);
            simdgroup_load(b1, ss + k * T + sx + 8, T);
            simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
            simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Through threadgroup memory, so the edges and the epilogue are plain.
    simdgroup_store(acc[0], xs + sy * T + sx, T);
    simdgroup_store(acc[1], xs + sy * T + sx + 8, T);
    simdgroup_store(acc[2], xs + (sy + 8) * T + sx, T);
    simdgroup_store(acc[3], xs + (sy + 8) * T + sx + 8, T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= m) return;
    for (uint k = 0; k < 8; k++) {
        uint rr = r0 + part + k;
        if (rr >= p.rows) break;
        uint o = i * p.rows + rr;
        float s = xs[row * T + part + k];
        if (tl.splits > 1) sums[tg.z * m * p.rows + o] = s;
        else y[o] = s * scale[rr] + ((p.flags & 2) ? add[o] : 0.0f);
    }
}

kernel void sign_matmul_sum(
    device const float *scale [[buffer(2)]],
    device const float *add [[buffer(4)]],
    device float *y [[buffer(5)]],
    constant Params &p [[buffer(6)]],
    constant Tiled &tl [[buffer(7)]],
    device const float *sums [[buffer(8)]],
    uint o [[thread_position_in_grid]])
{
    uint n = tl.m * p.rows;
    if (o >= n) return;
    float s = 0;
    for (uint z = 0; z < tl.splits; z++) s += sums[z * n + o];
    y[o] = s * scale[o % p.rows] + ((p.flags & 2) ? add[o] : 0.0f);
}
"#;

/// Simdgroups, so output rows, per threadgroup.
const ROWS_PER_GROUP: usize = 8;

/// The rows of `x` from which `sign_matmul_tiled` takes the product.
const TILED_ROWS: usize = 4;

/// `sign_matmul_tiled`'s tile, rows of `x` by rows of signs.
const TILE: usize = 32;

/// Threadgroups `sign_matmul_tiled` splits the words to reach, and the
/// fewest words a split takes.
const GROUPS: usize = 256;
const SPLIT_WORDS: usize = 4;

#[repr(C)]
struct Tiled {
    m: u32,
    per_split: u32,
    splits: u32,
}

/// One factorisation on the device: the sign words as `u32`, the vectors as
/// f32, `mid` already `u2 ⊙ v1`.
struct Path {
    u: Tensor,
    v: Tensor,
    u1: Tensor,
    mid: Tensor,
    v2: Tensor,
}

/// A weight matrix held as LittleBit factors, on Metal.
pub(crate) struct LittleBit {
    rows: usize,
    cols: usize,
    paths: Vec<Path>,
}

impl LittleBit {
    /// `f` on `device`, which must be `out x inp` and a Metal device whose
    /// kernels built.
    pub(crate) fn new(f: &Factored, out: usize, inp: usize, device: &Device) -> Res<Self> {
        if (f.rows(), f.cols()) != (out, inp) {
            return Err(format!("LittleBit factors for a {}x{} matrix where {out}x{inp} was expected", f.rows(), f.cols()).into());
        }
        if kernels(device).is_none() {
            return Err("LittleBit weights need kvad-gpu's Metal kernels, and they are not available here".into());
        }
        let signs = |s: &kvad::littlebit::Signs| Tensor::from_slice(s.bits(), (s.rows(), s.words()), device);
        let vector = |v: &[f32]| Tensor::from_slice(v, v.len(), device);
        let paths = f
            .paths()
            .iter()
            .map(|p| {
                Ok(Path { u: signs(p.u())?, v: signs(p.v())?, u1: vector(p.u1())?, mid: vector(p.mid())?, v2: vector(p.v2())? })
            })
            .collect::<candle_core::Result<_>>()?;
        Ok(LittleBit { rows: out, cols: inp, paths })
    }

    /// `x · Wᵀ`, for `x` of any number of leading axes.
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_from(x, TILED_ROWS)
    }

    /// [`forward`](Self::forward), tiled from `tiled_from` rows of `x`.
    fn forward_from(&self, x: &Tensor, tiled_from: usize) -> candle_core::Result<Tensor> {
        let dt = x.dtype();
        let mut dims = x.dims().to_vec();
        let xs = x.to_dtype(DType::F32)?.reshape(((), self.cols))?.contiguous()?;
        let mut y: Option<Tensor> = None;
        for p in &self.paths {
            let z = sign_matmul_from(&xs, &p.v, &p.mid, Some(&p.v2), None, tiled_from)?;
            y = Some(sign_matmul_from(&z, &p.u, &p.u1, None, y.as_ref(), tiled_from)?);
        }
        *dims.last_mut().unwrap() = self.rows;
        y.expect("at least one path").reshape(dims)?.to_dtype(dt)
    }

    pub(crate) fn bytes(&self) -> usize {
        let per = |t: &Tensor| t.elem_count() * 4;
        self.paths.iter().map(|p| per(&p.u) + per(&p.v) + per(&p.u1) + per(&p.mid) + per(&p.v2)).sum()
    }

    /// The entries of the matrix these factors stand for.
    pub(crate) fn params(&self) -> usize {
        self.rows * self.cols
    }
}

fn kernels(device: &Device) -> Option<&'static crate::fused::metal::Kernels> {
    // Not `library`: there is no candle chain to fall back on, so
    // `KVAD_GPU_FUSED=0` does not turn this off.
    crate::fused::metal::build(device, "littlebit", SOURCE, None)
}

/// `y[i, r] = scale[r] · Σ_c signs[r, c] · (x[i, c] · pre[c]) + add[i, r]`,
/// for `x` `[m, cols]`, contiguous f32; on `sign_matmul_tiled` from
/// `tiled_from` rows of `x`.
fn sign_matmul_from(x: &Tensor, signs: &Tensor, scale: &Tensor, pre: Option<&Tensor>, add: Option<&Tensor>, tiled_from: usize) -> candle_core::Result<Tensor> {
    let op = SignMatmul { signs: signs.clone(), scale: scale.clone(), pre: pre.cloned(), add: add.cloned(), tiled_from };
    x.apply_op1_no_bwd(&op)
}

struct SignMatmul {
    tiled_from: usize,
    signs: Tensor,
    scale: Tensor,
    pre: Option<Tensor>,
    add: Option<Tensor>,
}

#[repr(C)]
struct Params {
    rows: u32,
    cols: u32,
    flags: u32,
}

impl CustomOp1 for SignMatmul {
    fn name(&self) -> &'static str {
        "sign_matmul"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("sign_matmul runs on Metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        let (m, cols) = lx.shape().dims2()?;
        let rows = self.signs.dim(0)?;
        if x.dtype() != DType::F32 || !lx.is_contiguous() || self.signs.dim(1)? != cols.div_ceil(WORD) {
            candle_core::bail!("sign_matmul: {:?} {:?} against {:?} signs", x.dtype(), lx.shape(), self.signs.shape());
        }
        let dev = x.device();
        let tiled = m >= self.tiled_from;
        // How `sign_matmul_tiled` splits the words, decided and allocated
        // before the encoder is taken: `unpooled` takes one of its own
        // under test, and two at once wait on each other for ever.
        let words = cols.div_ceil(WORD);
        let tiles = rows.div_ceil(TILE) * m.div_ceil(TILE);
        let per_split = words.div_ceil(GROUPS.div_ceil(tiles)).max(SPLIT_WORDS).min(words);
        let splits = if tiled { words.div_ceil(per_split) } else { 1 };
        let sums = match splits {
            1 => None,
            _ => Some(crate::fused::metal::unpooled(dev, splits * m * rows * 4)?),
        };
        let name = if tiled { "sign_matmul_tiled" } else { "sign_matmul" };
        let pipe = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no littlebit kernels".into()))?.pipe(name)?;
        let out = output(dev, m * rows * 4)?;

        // Separate storages from `x`, so their locks cannot wait on candle's.
        let metal = |t: &Tensor| -> candle_core::Result<(Buffer, usize)> {
            let (s, l) = t.storage_and_layout();
            match &*s {
                Storage::Metal(s) => Ok(buffer(s, l)),
                _ => candle_core::bail!("sign_matmul: a tensor is not on Metal"),
            }
        };
        let (xb, xo) = buffer(x, lx);
        let (sb, so) = metal(&self.signs)?;
        let (cb, co) = metal(&self.scale)?;
        // An absent input is bound to `x`, which the kernel then never reads.
        let pre = self.pre.as_ref().map(metal).transpose()?;
        let add = self.add.as_ref().map(metal).transpose()?;
        let params = Params { rows: rows as u32, cols: cols as u32, flags: pre.is_some() as u32 | (add.is_some() as u32) << 1 };
        let (pb, po) = pre.unwrap_or_else(|| (xb.clone(), xo));
        let (ab, ao) = add.unwrap_or_else(|| (xb.clone(), xo));

        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label(name);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_input_buffer(0, Some(&sb), so);
        enc.set_input_buffer(1, Some(&xb), xo);
        enc.set_input_buffer(2, Some(&cb), co);
        enc.set_input_buffer(3, Some(&pb), po);
        enc.set_input_buffer(4, Some(&ab), ao);
        enc.set_output_buffer(5, Some(&out), 0);
        enc.set_bytes(6, &params);
        if tiled {
            let tl = Tiled { m: m as u32, per_split: per_split as u32, splits: splits as u32 };
            // With one split there are no sums, and `out` stands in unread.
            let part = sums.as_ref().unwrap_or(&out);
            enc.set_bytes(7, &tl);
            enc.set_output_buffer(8, Some(part), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: rows.div_ceil(TILE), height: m.div_ceil(TILE), depth: splits },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            if splits > 1 {
                let kernels = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no littlebit kernels".into()))?;
                enc.set_compute_pipeline_state(&kernels.pipe("sign_matmul_sum")?);
                // Every buffer named again, though bound already: candle
                // tracks hazards by what each dispatch names, and the barrier
                // `sums` brings forgets the rest. A `y` written here unnamed
                // was read by the next product before it was written: wrong
                // text from the real Qwen3-8B, and different every run.
                enc.set_input_buffer(8, Some(part), 0);
                enc.set_input_buffer(2, Some(&cb), co);
                enc.set_input_buffer(4, Some(&ab), ao);
                enc.set_output_buffer(5, Some(&out), 0);
                enc.dispatch_thread_groups(
                    MTLSize { width: (m * rows).div_ceil(256), height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
        } else {
            enc.dispatch_thread_groups(
                MTLSize { width: rows.div_ceil(ROWS_PER_GROUP), height: m, depth: 1 },
                MTLSize { width: ROWS_PER_GROUP * 32, height: 1, depth: 1 },
            );
        }
        Ok((MetalStorage::new(out, dev.clone(), m * rows, DType::F32), Shape::from((m, rows))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::littlebit::{Path as Factors, Signs};

    /// A random factorisation, with ranks and widths that are not whole words
    /// or whole vectors of four, and a residual path.
    pub(super) fn random(out: usize, inp: usize, s: usize, seed: u64) -> Factored {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let paths = (0..2)
            .map(|_| {
                let mut bits = |rows: usize, cols: usize| {
                    let words: Vec<bool> = (0..rows * cols).map(|_| next() & 1 == 1).collect();
                    Signs::pack(rows, cols, |r, c| words[r * cols + c])
                };
                let (u, v) = (bits(out, s), bits(s, inp));
                let mut vec = |n: usize| (0..n).map(|_| next() as f32 / u32::MAX as f32 + 0.5).collect::<Vec<f32>>();
                let (u1, u2, v1, v2) = (vec(out), vec(s), vec(s), vec(inp));
                Factors::new(u, v, u1, &u2, &v1, v2).unwrap()
            })
            .collect();
        Factored::new(paths).unwrap()
    }

    #[test]
    fn the_kernel_agrees_with_the_cpu_engine() {
        let Ok(dev) = Device::new_metal(0) else { return };
        // Shapes off the tiles and the words, both kernels on every one.
        let shapes = [(70, 45, 37, 1), (64, 128, 32, 5), (33, 203, 61, 3), (97, 70, 40, 33), (32, 64, 64, 64), (40, 300, 90, 71), (64, 1000, 100, 9)];
        for (out, inp, s, m) in shapes {
            let f = random(out, inp, s, (out * inp) as u64);
            let lb = LittleBit::new(&f, out, inp, &dev).unwrap();
            let xs: Vec<f32> = (0..m * inp).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
            let want = f.matmul(&xs, m);
            for tiled_from in [1, usize::MAX] {
                let before = crate::fused::tests_ran();
                let got = lb.forward_from(&Tensor::from_slice(&xs, (m, inp), &dev).unwrap(), tiled_from).unwrap();
                assert_eq!(crate::fused::tests_ran() - before, 4, "two kernels a path");
                let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{out}x{inp} rank {s}, m {m}, tiled from {tiled_from}: [{i}] {g} against {w}");
                }
            }
        }
    }
}

/// The projections of a Qwen3-8B token at 0.55 bits a weight, from random
/// factors: rank 544 for attention and 824 for the MLP, two paths each, 36
/// layers.
///
/// ```text
/// cargo test --release -p kvad-gpu --lib littlebit::bench -- --ignored --nocapture
/// ```
#[cfg(test)]
mod bench {
    use super::tests::random;
    use super::*;

    #[test]
    #[ignore = "a measurement, not a check"]
    fn qwen3_8b_projections() {
        let dev = Device::new_metal(0).unwrap();
        let (e, kv, ff) = (4096, 1024, 12288);
        let shapes = [(e, e, 544), (kv, e, 544), (kv, e, 544), (e, e, 544), (ff, e, 824), (ff, e, 824), (e, ff, 824)];
        let layers: Vec<Vec<LittleBit>> = (0..36)
            .map(|l| shapes.iter().map(|&(o, i, s)| LittleBit::new(&random(o, i, s, l as u64 + o as u64), o, i, &dev).unwrap()).collect())
            .collect();
        for m in [1, 2, 3, 4, 8, 16, 27, 64, 128, 512] {
            let xs = [e, ff].map(|w| Tensor::rand(-1f32, 1f32, (m, w), &dev).unwrap());
            for (kernel, tiled_from) in [("row", usize::MAX), ("tiled", 1)] {
                let token = || {
                    for layer in &layers {
                        for (p, &(_, i, _)) in layer.iter().zip(&shapes) {
                            p.forward_from(&xs[(i == ff) as usize], tiled_from).unwrap();
                        }
                    }
                    dev.synchronize().unwrap();
                };
                token();
                let n = if m >= 128 { 3 } else { 10 };
                let t = std::time::Instant::now();
                (0..n).for_each(|_| token());
                let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
                println!("{m:>4} rows, {kernel:>5}: {ms:8.2} ms a pass of 252 projections, {:.2} ms a row", ms / m as f64);
            }
        }
    }
}

