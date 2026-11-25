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
//! The rows of `x` — a prompt's tokens — are the grid's second axis, so a
//! prefill rereads each row of signs once a token, from cache.
//!
//! # Measured
//!
//! On an M5 Pro, the 252 projections of a Qwen3-8B token at 0.55 bits a
//! weight (`bench` below, random factors at the real shapes): 19.6 ms for one
//! row, against about 90 ms for the CPU engine's whole token. For 27 rows,
//! 320 ms, which is 12 ms a row. A prompt barely shares any work between its
//! rows.
//!
//! shortcut: a prefill is one decode step a row, sharing nothing but the
//! cache; holding a word's signs against several rows of `x` would be the
//! next step if long prompts matter.
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
"#;

/// Simdgroups, so output rows, per threadgroup.
const ROWS_PER_GROUP: usize = 8;

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
        let dt = x.dtype();
        let mut dims = x.dims().to_vec();
        let xs = x.to_dtype(DType::F32)?.reshape(((), self.cols))?.contiguous()?;
        let mut y: Option<Tensor> = None;
        for p in &self.paths {
            let z = sign_matmul(&xs, &p.v, &p.mid, Some(&p.v2), None)?;
            y = Some(sign_matmul(&z, &p.u, &p.u1, None, y.as_ref())?);
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
/// for `x` `[m, cols]`, contiguous f32.
fn sign_matmul(x: &Tensor, signs: &Tensor, scale: &Tensor, pre: Option<&Tensor>, add: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let op = SignMatmul { signs: signs.clone(), scale: scale.clone(), pre: pre.cloned(), add: add.cloned() };
    x.apply_op1_no_bwd(&op)
}

struct SignMatmul {
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
        let pipe = kernels(&Device::Metal(dev.clone())).ok_or_else(|| candle_core::Error::Msg("no littlebit kernels".into()))?.pipe("sign_matmul")?;
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
        enc.set_label("sign_matmul");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_input_buffer(0, Some(&sb), so);
        enc.set_input_buffer(1, Some(&xb), xo);
        enc.set_input_buffer(2, Some(&cb), co);
        enc.set_input_buffer(3, Some(&pb), po);
        enc.set_input_buffer(4, Some(&ab), ao);
        enc.set_output_buffer(5, Some(&out), 0);
        enc.set_bytes(6, &params);
        enc.dispatch_thread_groups(
            MTLSize { width: rows.div_ceil(ROWS_PER_GROUP), height: m, depth: 1 },
            MTLSize { width: ROWS_PER_GROUP * 32, height: 1, depth: 1 },
        );
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
        for (out, inp, s, m) in [(70, 45, 37, 1), (64, 128, 32, 5), (33, 203, 61, 3)] {
            let f = random(out, inp, s, (out * inp) as u64);
            let lb = LittleBit::new(&f, out, inp, &dev).unwrap();
            let xs: Vec<f32> = (0..m * inp).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
            let before = crate::fused::tests_ran();
            let got = lb.forward(&Tensor::from_slice(&xs, (m, inp), &dev).unwrap()).unwrap();
            assert_eq!(crate::fused::tests_ran() - before, 4, "two kernels a path");
            let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let want = f.matmul(&xs, m);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{out}x{inp} rank {s}, m {m}: [{i}] {g} against {w}");
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
        for m in [1, 27] {
            let xs = [e, ff].map(|w| Tensor::rand(-1f32, 1f32, (m, w), &dev).unwrap());
            let token = || {
                for layer in &layers {
                    for (p, &(_, i, _)) in layer.iter().zip(&shapes) {
                        p.forward(&xs[(i == ff) as usize]).unwrap();
                    }
                }
                dev.synchronize().unwrap();
            };
            token();
            let n = 20;
            let t = std::time::Instant::now();
            (0..n).for_each(|_| token());
            let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            println!("{m} rows: {ms:.2} ms a pass of 252 projections");
        }
    }
}
