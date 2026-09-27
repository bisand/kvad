//! LTX-2.5's diffusion video decoder ("DiffVAE"): latents to frames by a
//! neighbourhood-attention transformer and one step of diffusion.
//!
//! `vae/ltx-2.5-video-vae-bf16.safetensors`, whose decoder is an
//! `NADiffusionDecoder` (`docs/video-plan.md`, "The DiffVAE decoder"):
//!
//! ```text
//! latent [128, F, h, w] ─ un-normalise ─ linear to 2048
//!   ─ stage 1: 4 blocks, 2048 wide, windows 3×7×7 ─ ×2 in space   ─ 1024
//!   ─ stage 2: 6 blocks, 1024,      3×7×7        ─ ×2 in time    ─ 512
//!   ─ stage 3: 4 blocks, 512,       3×5×5        ─ ×2 in all     ─ 512
//!   ─ stage 4: 2 blocks, 512,       3×5×5        ─ ×2 in all     ─ 256: the context
//! noise [3, 8(F−1)+1, 32h, 32w] ─ 4×4 patches, 48 ─ linear to 256
//!   ─ stage 5: 8 blocks, 256, windows 11×11×11, each adding the context
//!     and modulated by the step ─ norm, linear to 48 ─ frames in [−1, 1]
//! ```
//!
//! A block is pre-norm neighbourhood attention and a SwiGLU feed-forward.
//! Everything is channels-last tokens `[t·h·w, C]`, row by row: the linear
//! layers act on rows, and a token's neighbours are a fixed offset away.
//!
//! **Neighbourhood attention** is NATTEN's `na3d`: every token attends to
//! exactly the window around it, shifted inward at the edges rather than
//! cut, so each axis must be at least its window ([`window_start`]).
//! [`neighbourhood`] runs it as flash attention over each patch of queries'
//! box of keys on the M5's matrix units, and [`plain`] writes it out, one
//! gather for each offset in the window.
//!
//! **Stage 5 is one step of diffusion**, from noise at `t = 1` straight to the
//! clean frames, so the frames depend on the noise: the seed.
//!
//! **With keyframes** ([`DiffDecoder::decode_keyed`]), as DFR decodes, a
//! second stream of *planes* runs beside the video: one latent frame each,
//! tagged with the file's `type_emb` and then through the same weights, and
//! upsampled each as a clip of one frame, so in space only. Each plane has
//! a place in each stage's time ([`DiffDecoder::times`]). The streams meet
//! only in the attention, which is then joint ([`joint`]): a frame's query
//! sees its own window and the same rows and columns on its two nearest
//! planes, a plane's its own and the same on its two nearest frames, all in
//! one softmax, and windows are cut at the edges rather than shifted, as
//! the reference's joint kernels have them. At stage 5 the planes are
//! pixels of their own noise, denoised with the video and dropped. Each
//! tile carries the planes near it ([`planes_for_tile`]).

use super::metadata;
use super::ltx_nn::RmsNorm;
use crate::common::Loader;
use crate::image::nn::{noise_block, Ctx, Linear};
use crate::image::{finish, open};
use crate::prof::span;
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor, D};
use kvad::serde_json::Value;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The DiffVAE's file in [`super::LTX_REPO`].
pub const FILE: &str = "vae/ltx-2.5-video-vae-bf16.safetensors";

/// Norms' epsilon, everywhere in the decoder.
const EPS: f64 = 1e-6;

/// Where each window starts along an axis of `n`, for a window of `k`:
/// centred on the index, shifted inward at the edges, `clamp(i − ⌊k/2⌋, 0,
/// n − k)`. NATTEN's rule, which the reference's eager path writes out.
pub fn window_start(n: usize, k: usize) -> Vec<usize> {
    (0..n).map(|i| i.saturating_sub(k / 2).min(n - k)).collect()
}

/// A token grid, (frames, rows, columns).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Grid {
    pub fn tokens(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// Neighbourhood attention over tokens `[n, heads·d]` on `grid`, each
/// query attending to the `kernel` window around it: `softmax(q·kᵀ)·v`
/// within the window, q already scaled; in q's dtype.
///
/// On the M5's matrix units in bf16 or f16 (`crate::mpp_neighbourhood`),
/// and otherwise [`plain`].
pub fn neighbourhood(q: &Tensor, k: &Tensor, v: &Tensor, grid: Grid, kernel: [usize; 3], heads: usize) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    if let Some(o) = crate::mpp_neighbourhood::neighbourhood(q, k, v, [grid.t, grid.h, grid.w], kernel, heads, 1.0)? {
        return Ok(o);
    }
    plain(q, k, v, grid, kernel, heads)?.to_dtype(q.dtype())
}

/// [`neighbourhood`] written plainly, in f32: once for each of the window's
/// offsets `(a, b, c)`, every query's key is its window's corner plus that
/// offset, so one gather fetches them all, and an online softmax (running
/// maximum, sum and weighted values) takes them in. Exact, whatever order
/// the offsets come in, and slow: 1331 gathers of every key at stage 5.
pub fn plain(q: &Tensor, k: &Tensor, v: &Tensor, grid: Grid, kernel: [usize; 3], heads: usize) -> candle_core::Result<Tensor> {
    let (n, width) = q.dims2()?;
    let [kt, kh, kw] = kernel;
    if grid.tokens() != n || grid.t < kt || grid.h < kh || grid.w < kw {
        candle_core::bail!("neighbourhood attention: {n} tokens on {grid:?}, window {kernel:?}");
    }
    let d = width / heads;
    let dev = q.device();
    // Each query's window corner, as a token index.
    let (st, sh, sw) = (window_start(grid.t, kt), window_start(grid.h, kh), window_start(grid.w, kw));
    let mut corner = Vec::with_capacity(n);
    for &t0 in &st {
        for &h0 in &sh {
            for &w0 in &sw {
                corner.push(((t0 * grid.h + h0) * grid.w + w0) as u32);
            }
        }
    }
    let corner = Tensor::from_vec(corner, n, dev)?;
    let f = |t: &Tensor| t.to_dtype(DType::F32)?.reshape((n, heads, d));
    let (q, k, v) = (f(q)?, f(k)?.reshape((n, width))?, f(v)?.reshape((n, width))?);
    let mut best = Tensor::full(f32::NEG_INFINITY, (n, heads, 1), dev)?;
    let mut total = Tensor::zeros((n, heads, 1), DType::F32, dev)?;
    let mut acc = Tensor::zeros((n, heads, d), DType::F32, dev)?;
    for a in 0..kt {
        for b in 0..kh {
            for c in 0..kw {
                let offset = ((a * grid.h + b) * grid.w + c) as f64;
                let at = (corner.to_dtype(DType::F32)? + offset)?.to_dtype(DType::U32)?;
                let kk = k.index_select(&at, 0)?.reshape((n, heads, d))?;
                let vv = v.index_select(&at, 0)?.reshape((n, heads, d))?;
                let s = (&q * &kk)?.sum_keepdim(D::Minus1)?;
                let next = best.maximum(&s)?;
                let fade = (&best - &next)?.exp()?;
                let p = (&s - &next)?.exp()?;
                total = ((total * &fade)? + &p)?;
                acc = (acc.broadcast_mul(&fade)? + vv.broadcast_mul(&p)?)?;
                best = next;
            }
        }
    }
    acc.broadcast_div(&total)?.reshape((n, width))
}

// ---------------------------------------------------------------------------
// Keyframes: joint attention
// ---------------------------------------------------------------------------

/// How many planes a frame's queries see, and frames a plane's: the
/// reference's `KEYFRAME_CONTEXT_SLOTS`.
pub const SLOTS: usize = 2;

/// Each of `frames` frames' nearest keyframe planes, and each plane's
/// nearest frames, [`SLOTS`] of each: by distance in the stage's time, the
/// lower index on a tie, −1 where there are too few. The reference's
/// `video_keyframe_slots` and `keyframe_video_slots`, which rank by a
/// stable sort of `|Δt|` in f32.
pub fn slots(times: &[f32], frames: usize) -> (Vec<[i32; SLOTS]>, Vec<[i32; SLOTS]>) {
    let nearest = |q: f32, candidates: &mut dyn Iterator<Item = f32>| -> [i32; SLOTS] {
        let mut d: Vec<(f32, usize)> = candidates.enumerate().map(|(i, c)| ((q - c).abs(), i)).collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        std::array::from_fn(|j| d.get(j).map_or(-1, |x| x.1 as i32))
    };
    let video = (0..frames).map(|t| nearest(t as f32, &mut times.iter().copied())).collect();
    let planes = times.iter().map(|&p| nearest(p, &mut (0..frames).map(|t| t as f32))).collect();
    (video, planes)
}

/// One part of a joint attention: keys and values `[frames·h·w, width]` on
/// the queries' rows and columns, and for each query frame the key frame
/// its window is centred on, −1 for none. A window here is cut at the
/// edges rather than shifted inward: `(c + a − ⌊kt/2⌋, h + b − ⌊kh/2⌋, w + c −
/// ⌊kw/2⌋)` for its offsets, those off the grid left out, as the
/// reference's joint attention has it.
pub(crate) struct Part<'a> {
    pub(crate) k: &'a Tensor,
    pub(crate) v: &'a Tensor,
    pub(crate) frames: usize,
    pub(crate) centres: Vec<i32>,
    pub(crate) kernel: [usize; 3],
}

/// A part's attention for queries `q` `[n, heads·d]` on `grid`, written
/// plainly as [`plain`] is, in f32: the answer `[n, heads, d]` and each
/// row's log-sum-exp of scores `[n, heads, 1]`. An offset at a time, every
/// query's key gathered, those off the grid kept out of the softmax; a row
/// with no keys answers 0, at −∞.
pub(crate) fn plain_part(q: &Tensor, grid: Grid, part: &Part<'_>, heads: usize) -> candle_core::Result<(Tensor, Tensor)> {
    let (n, width) = q.dims2()?;
    let [kt, kh, kw] = part.kernel;
    if grid.tokens() != n || part.centres.len() != grid.t || part.k.dim(0)? != part.frames * grid.h * grid.w {
        candle_core::bail!("joint attention: {n} queries on {grid:?}, keys {:?} in {} frames", part.k.dims(), part.frames);
    }
    let d = width / heads;
    let dev = q.device();
    let f = |t: &Tensor| t.to_dtype(DType::F32);
    let (q, k, v) = (f(q)?.reshape((n, heads, d))?, f(part.k)?, f(part.v)?);
    // −1e30 rather than −∞ for a key left out, so that no difference of
    // two of them is a NaN; its weight is zeroed besides.
    let mut best = Tensor::full(-1e30f32, (n, heads, 1), dev)?;
    let mut total = Tensor::zeros((n, heads, 1), DType::F32, dev)?;
    let mut acc = Tensor::zeros((n, heads, d), DType::F32, dev)?;
    let (gh, gw) = (grid.h as i64, grid.w as i64);
    for a in 0..kt {
        for b in 0..kh {
            for c in 0..kw {
                let mut at = Vec::with_capacity(n);
                let mut ok = Vec::with_capacity(n);
                for &centre in &part.centres {
                    let ft = centre as i64 + a as i64 - (kt / 2) as i64;
                    let frame_ok = centre >= 0 && (0..part.frames as i64).contains(&ft);
                    for h in 0..gh {
                        let fh = h + b as i64 - (kh / 2) as i64;
                        for w in 0..gw {
                            let fw = w + c as i64 - (kw / 2) as i64;
                            let valid = frame_ok && (0..gh).contains(&fh) && (0..gw).contains(&fw);
                            at.push(if valid { ((ft * gh + fh) * gw + fw) as u32 } else { 0 });
                            ok.push(if valid { 1f32 } else { 0.0 });
                        }
                    }
                }
                let at = Tensor::from_vec(at, n, dev)?;
                let ok = Tensor::from_vec(ok, (n, 1, 1), dev)?;
                let kk = k.index_select(&at, 0)?.reshape((n, heads, d))?;
                let vv = v.index_select(&at, 0)?.reshape((n, heads, d))?;
                let s = (&q * &kk)?.sum_keepdim(D::Minus1)?;
                let s = (s.broadcast_mul(&ok)? + ((ok.clone() - 1.0)? * 1e30)?.broadcast_as(s.shape())?)?;
                let next = best.maximum(&s)?;
                let fade = (&best - &next)?.exp()?;
                let p = (&s - &next)?.exp()?.broadcast_mul(&ok)?;
                total = ((total * &fade)? + &p)?;
                acc = (acc.broadcast_mul(&fade)? + vv.broadcast_mul(&p)?)?;
                best = next;
            }
        }
    }
    let lse = (&best + total.log()?)?;
    Ok((acc.broadcast_div(&total.maximum(1e-30)?)?, lse))
}

/// Parts' answers `[n, heads, d]` put together as one softmax over all
/// their keys, into `[n, heads·d]` of `dtype`: each weighted by its share of
/// the whole, `exp(lse − max)`. A run of rows at a time, so that its
/// temporaries stay small beside the parts, which at stage 5 are whole
/// activations in f32.
pub(crate) fn merge(parts: &[(Tensor, Tensor)], dtype: DType) -> candle_core::Result<Tensor> {
    let (n, heads, d) = parts[0].0.dims3()?;
    let dev = parts[0].0.device().clone();
    let step = (CHUNK_BYTES / (4 * heads * d)).max(1);
    let spans: Vec<(usize, usize)> = (0..n).step_by(step).map(|r| (r, (r + step).min(n))).collect();
    by_runs(n, heads * d, dtype, &dev, &spans, 1, |r0, len, _, _| {
        let rows = |t: &Tensor| t.narrow(0, r0, len);
        let mut top = rows(&parts[0].1)?;
        for (_, l) in &parts[1..] {
            top = top.maximum(&rows(l)?)?;
        }
        let (mut num, mut den) = (None::<Tensor>, None::<Tensor>);
        for (o, l) in parts {
            let w = (rows(l)? - &top)?.exp()?;
            let y = rows(o)?.broadcast_mul(&w)?;
            num = Some(match num {
                Some(a) => (a + y)?,
                None => y,
            });
            den = Some(match den {
                Some(a) => (a + w)?,
                None => w,
            });
        }
        num.unwrap().broadcast_div(&den.unwrap())?.reshape((len, heads * d))
    })
}

/// Joint neighbourhood attention over a video `q, k, v` `[n, heads·d]` on
/// `grid` and keyframe planes `pq, pk, pv` on `planes`, the planes at stage
/// times `times`, q already scaled: the reference's `joint_na3d`, one softmax
/// per query over
///
/// - for a frame's query, its `kernel` window, cut at the edges, and the
///   same rows and columns on each of its [`SLOTS`] nearest planes;
/// - for a plane's, its window on its own plane, and the same on each of
///   its nearest frames.
///
/// On the M5 each side is one kernel
/// (`crate::mpp_neighbourhood::neighbourhood_joint`), which walks the other
/// stream's rows after its own window's. Otherwise each part is its own
/// attention with its log-sum-exp ([`plain_part`]), and [`merge`] makes them
/// one. The answers `[n, heads·d]` and `[P·h·w, heads·d]`,
/// in q's dtype.
#[allow(clippy::too_many_arguments)]
pub fn joint(q: &Tensor, k: &Tensor, v: &Tensor, grid: Grid, pq: &Tensor, pk: &Tensor, pv: &Tensor, planes: Grid, times: &[f32], kernel: [usize; 3], heads: usize)
 -> candle_core::Result<(Tensor, Tensor)> {
    let (video, plane) = slots(times, grid.t);
    let flat = [1, kernel[1], kernel[2]];
    let own = |k, v, frames, n: usize, kernel| Part { k, v, frames, centres: (0..n as i32).collect(), kernel };
    let slot = |k, v, frames, table: &[[i32; SLOTS]], j: usize| Part { k, v, frames, centres: table.iter().map(|s| s[j]).collect(), kernel: flat };
    let mut vp = vec![own(k, v, grid.t, grid.t, kernel)];
    let mut pp = vec![own(pk, pv, planes.t, planes.t, flat)];
    for j in 0..SLOTS {
        vp.push(slot(pk, pv, planes.t, &video, j));
        pp.push(slot(k, v, grid.t, &plane, j));
    }
    let run = |q: &Tensor, g: Grid, parts: &[Part<'_>]| -> candle_core::Result<Tensor> {
        let answers = parts.iter().map(|p| plain_part(q, g, p, heads)).collect::<candle_core::Result<Vec<_>>>()?;
        merge(&answers, q.dtype())
    };
    // Each side in one kernel where there is one: the video's parts at
    // stage 5 would be whole activations in f32, three of them.
    #[cfg(target_os = "macos")]
    {
        use crate::mpp_neighbourhood::neighbourhood_joint as fused;
        // A plane's side is the same with the roles turned: its own plane's
        // window, then the same rows and columns on its frames.
        let o = fused(q, k, v, [grid.t, grid.h, grid.w], pk, pv, &video, kernel, heads, 1.0)?;
        let po = fused(pq, pk, pv, [planes.t, planes.h, planes.w], k, v, &plane, flat, heads, 1.0)?;
        if let (Some(o), Some(po)) = (o, po) {
            return Ok((o, po));
        }
    }
    Ok((run(q, grid, &vp)?, run(pq, planes, &pp)?))
}

/// Absolute 3D rotary positions for `[n, heads, 64]`: 16 of a head's
/// dimensions for time, 24 for rows, 24 for columns, each rotating adjacent
/// pairs `(2i, 2i + 1)` by the position times `10000^(−2i/d)`, positions
/// 0, 1, 2 … on each axis. Angles in f32 on the host, as the reference's
/// `rope_math` makes them.
///
/// Kept per axis, `[length, pairs]`, and put together on the device for a
/// run of frames at a time ([`Rope::frames`]): the whole table of a real
/// clip's stage 5 would be 760 MB.
///
/// The rotation is relative in every product of a query and a key, so a
/// tile can count its positions from its own corner; keyframe planes, which
/// join a frame's keys, take their places in that count ([`Rope::at`]).
struct Rope {
    /// cos and sin, for time, rows, columns.
    axes: [(Tensor, Tensor); 3],
    /// The same, cos then sin, flattened into one, for
    /// `ltx_fused::head_norm_rope`.
    packed: Tensor,
    pairs: [usize; 3],
    grid: Grid,
}

impl Rope {
    fn new(grid: Grid, split: [usize; 3], device: &Device) -> candle_core::Result<Self> {
        let frames: Vec<f32> = (0..grid.t).map(|t| t as f32).collect();
        Self::at(&frames, grid, split, device)
    }

    /// With frame `i` of `grid` at time `times[i]` rather than at `i`: keyframe
    /// planes, each at its own place in the stage's time, fractional as the
    /// reference's `keyframe_stage_times` makes it.
    fn at(times: &[f32], grid: Grid, split: [usize; 3], device: &Device) -> candle_core::Result<Self> {
        let axis = |positions: &[f32], d: usize| -> candle_core::Result<(Tensor, Tensor)> {
            let inv: Vec<f32> = (0..d / 2).map(|i| (1.0 / 10000f64.powf((2 * i) as f64 / d as f64)) as f32).collect();
            let angles: Vec<f32> = positions.iter().flat_map(|&p| inv.iter().map(move |f| p * f)).collect();
            let cos = angles.iter().map(|a| a.cos()).collect();
            let sin = angles.iter().map(|a| a.sin()).collect();
            Ok((Tensor::from_vec(cos, (positions.len(), d / 2), device)?, Tensor::from_vec(sin, (positions.len(), d / 2), device)?))
        };
        let count = |n: usize| (0..n).map(|i| i as f32).collect::<Vec<_>>();
        if times.len() != grid.t {
            candle_core::bail!("{} times for {} frames", times.len(), grid.t);
        }
        let axes = [axis(times, split[0])?, axis(&count(grid.h), split[1])?, axis(&count(grid.w), split[2])?];
        let flat = |i: usize| axes.iter().map(|a| if i == 0 { a.0.flatten_all() } else { a.1.flatten_all() }).collect::<candle_core::Result<Vec<_>>>();
        let packed = Tensor::cat(&[flat(0)?, flat(1)?].concat(), 0)?;
        Ok(Rope { axes, packed, pairs: split.map(|d| d / 2), grid })
    }

    /// cos and sin for the tokens of frames `t0 .. t1`, each
    /// `[(t1 − t0)·h·w, 1, 32, 1]`.
    fn frames(&self, t0: usize, t1: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let (ft, h, w) = (t1 - t0, self.grid.h, self.grid.w);
        let table = |i: usize| -> candle_core::Result<Tensor> {
            let pick = |a: usize| if i == 0 { &self.axes[a].0 } else { &self.axes[a].1 };
            let (pt, ph, pw) = (pick(0).dim(1)?, pick(1).dim(1)?, pick(2).dim(1)?);
            let t = pick(0).narrow(0, t0, ft)?.reshape((ft, 1, 1, pt))?.broadcast_as((ft, h, w, pt))?.contiguous()?;
            let r = pick(1).reshape((1, h, 1, ph))?.broadcast_as((ft, h, w, ph))?.contiguous()?;
            let c = pick(2).reshape((1, 1, w, pw))?.broadcast_as((ft, h, w, pw))?.contiguous()?;
            Tensor::cat(&[t, r, c], 3)?.reshape((ft * h * w, 1, pt + ph + pw, 1))
        };
        Ok((table(0)?, table(1)?))
    }
}

/// `x` `[n, heads, d]` rotated by `cos` and `sin` from [`Rope::frames`], in
/// its own dtype; computed in f32.
fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (n, heads, d) = x.dims3()?;
    let p = x.to_dtype(DType::F32)?.reshape((n, heads, d / 2, 2))?;
    let (xe, xo) = (p.narrow(3, 0, 1)?, p.narrow(3, 1, 1)?);
    let re = (xe.broadcast_mul(cos)? - xo.broadcast_mul(sin)?)?;
    let ro = (xe.broadcast_mul(sin)? + xo.broadcast_mul(cos)?)?;
    Tensor::cat(&[re, ro], 3)?.reshape((n, heads, d))?.to_dtype(x.dtype())
}

/// The largest f32 temporary a run of frames may make, in bytes: the
/// feed-forward's `[tokens, 4·width]`. What one block holds whole is its
/// input, the attention's q, k, v and answer, and its own answer.
const CHUNK_BYTES: usize = 1 << 30;

/// The runs of frames a stage at `width` on `grid` does its token-wise work
/// in: as many frames as keep [`CHUNK_BYTES`], and at least one.
fn runs(grid: Grid, width: usize) -> Vec<(usize, usize)> {
    let plane = grid.h * grid.w;
    let each = (CHUNK_BYTES / (plane * 16 * width).max(1)).max(1);
    (0..grid.t).step_by(each).map(|t0| (t0, (t0 + each).min(grid.t))).collect()
}

/// A new tensor for a whole grid, every element of which the caller writes:
/// on Metal of exactly its size and outside candle's pool
/// (`fused::metal::exact`), where a 1.52 GB activation would otherwise take
/// 2.15 GB and stay until the next synchronise.
fn blank(shape: &[usize], dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    if device.is_metal() {
        return crate::fused::metal::exact(shape, dtype, device);
    }
    Tensor::zeros(shape, dtype, device)
}

/// A new `[n, width]` of `dtype`, filled run by run: `f(r0, len)` is rows
/// `r0 .. r0 + len` of it.
fn by_runs(n: usize, width: usize, dtype: DType, device: &Device, spans: &[(usize, usize)], plane: usize,
           mut f: impl FnMut(usize, usize, usize, usize) -> candle_core::Result<Tensor>) -> candle_core::Result<Tensor> {
    let out = blank(&[n, width], dtype, device)?;
    for &(t0, t1) in spans {
        let y = f(t0 * plane, (t1 - t0) * plane, t0, t1)?;
        out.slice_set(&y.to_dtype(dtype)?.contiguous()?, 0, t0 * plane)?;
    }
    Ok(out)
}

/// Neighbourhood attention's projections: a fused `qkv`, whose answer is q,
/// k and v in that order; RMS norms over each head of q and k; q scaled by
/// `1/√64` before the rotation.
struct Attention {
    qkv: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    proj: Linear,
    heads: usize,
    kernel: [usize; 3],
}

impl Attention {
    fn load(cx: &Ctx<'_>, r: &crate::common::Reader<'_>, width: usize, head_dim: usize, kernel: [usize; 3]) -> Res<Self> {
        let a = r.pp("attn");
        Ok(Attention {
            qkv: Linear::load(cx, &a, "qkv", width, 3 * width, true)?,
            q_norm: RmsNorm::load(cx, &a, "q_norm.weight", head_dim, EPS)?,
            k_norm: RmsNorm::load(cx, &a, "k_norm.weight", head_dim, EPS)?,
            proj: Linear::load(cx, &a, "proj", width, width, true)?,
            heads: width / head_dim,
            kernel,
        })
    }

    /// q, k and v `[n, width]` from the fused projection's answer `qkv`
    /// `[n, 3·width]`, q and k normed and rotated by `cos` and `sin`: what
    /// `ltx_fused::head_norm_rope` does in one kernel.
    fn project(&self, qkv: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<[Tensor; 3]> {
        let (n, width) = (qkv.dim(0)?, qkv.dim(1)? / 3);
        let d = width / self.heads;
        let part = |i: usize| qkv.narrow(1, i * width, width)?.contiguous();
        let heads = |t: Tensor| t.reshape((n, self.heads, d));
        let q = (self.q_norm.forward(&heads(part(0)?)?)? * (1.0 / (d as f64).sqrt()))?;
        let k = self.k_norm.forward(&heads(part(1)?)?)?;
        let dt = qkv.dtype();
        let q = rotate(&q, cos, sin)?.reshape((n, width))?.to_dtype(dt)?;
        let k = rotate(&k, cos, sin)?.reshape((n, width))?.to_dtype(dt)?;
        Ok([q, k, part(2)?.to_dtype(dt)?])
    }
}

/// `w_down(silu(w_gate·x) · w_up·x)`, no biases.
struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGlu {
    fn load(cx: &Ctx<'_>, r: &crate::common::Reader<'_>, width: usize) -> Res<Self> {
        let hidden = (width * 4).div_ceil(16) * 16;
        let m = r.pp("mlp");
        Ok(SwiGlu {
            gate: Linear::load(cx, &m, "w_gate", width, hidden, false)?,
            up: Linear::load(cx, &m, "w_up", width, hidden, false)?,
            down: Linear::load(cx, &m, "w_down", hidden, width, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (g, u) = (self.gate.forward(x)?, self.up.forward(x)?);
        let h = match super::ltx_fused::swiglu(&g, &u)? {
            Some(h) => h,
            None => (g.to_dtype(DType::F32)?.silu()? * u.to_dtype(DType::F32)?)?,
        };
        self.down.forward(&h.to_dtype(x.dtype())?)
    }
}

/// Stage 5's conditioning of a block: the context's projection, added
/// first, and the block's table, added to the step's rows.
struct Cond {
    context: Linear,
    table: Tensor,
}

/// A block: `x + NA(rms(x))`, then `x + SwiGLU(rms(x))`. At stage 5 the
/// context is added before, and each norm is scaled and shifted by the
/// step's rows plus the block's table (`(1 + scale)·rms(x) + shift`; the
/// table's gate rows are unused).
struct Block {
    norm1: RmsNorm,
    attn: Attention,
    norm2: RmsNorm,
    mlp: SwiGlu,
    cond: Option<Cond>,
}

impl Block {
    fn load(cx: &Ctx<'_>, r: &crate::common::Reader<'_>, width: usize, head_dim: usize, kernel: [usize; 3]) -> Res<Self> {
        Ok(Block {
            norm1: RmsNorm::load(cx, r, "norm1.weight", width, EPS)?,
            attn: Attention::load(cx, r, width, head_dim, kernel)?,
            norm2: RmsNorm::load(cx, r, "norm2.weight", width, EPS)?,
            mlp: SwiGlu::load(cx, r, width)?,
            cond: None,
        })
    }

    /// The block on tokens `x` `[n, width]` of `grid`, with stage 5's
    /// context `[n, width]` and rows `[7, width]` if it has a [`Cond`].
    ///
    /// Everything but the attention goes a run of frames at a time
    /// ([`runs`]); `x` is taken, so that it goes as soon as it is used.
    /// On Metal the norms, the q and k norms with their rotation, and the
    /// SwiGLU's product are one kernel each (`ltx_fused`).
    fn forward(&self, x: Tensor, grid: Grid, rope: &Rope, ctx: Option<(&Tensor, &Tensor)>) -> candle_core::Result<Tensor> {
        Ok(self.forward_keyed(x, grid, rope, ctx, None)?.0)
    }

    /// [`Block::forward`], and with `keys` the keyframe planes beside the
    /// video: the same weights and the same step's rows for both, each with
    /// its own context, meeting only in the attention ([`joint`]). The
    /// planes' answer is the second.
    fn forward_keyed(&self, x: Tensor, grid: Grid, rope: &Rope, ctx: Option<(&Tensor, &Tensor)>, keys: Option<Keyed<'_>>)
     -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let width = x.dim(1)?;
        let (dt, dev) = (x.dtype(), x.device().clone());
        let with_context = |x: Tensor, grid: Grid, context: Option<&Tensor>| -> candle_core::Result<Tensor> {
            match (&self.cond, context) {
                (Some(c), Some(context)) => span(|| format!("{width}: context"), &dev, || {
                    by_runs(grid.tokens(), width, dt, &dev, &runs(grid, width), grid.h * grid.w, |r0, len, _, _| {
                        x.narrow(0, r0, len)? + c.context.forward(&context.narrow(0, r0, len)?)?
                    })
                }),
                _ => Ok(x),
            }
        };
        let m = match (&self.cond, ctx) {
            (Some(c), Some((_, rows))) => Some((rows + &c.table)?),
            _ => None,
        };
        let x = with_context(x, grid, ctx.map(|c| c.0))?;
        let keys = match keys {
            Some(k) => Some(Keyed { x: with_context(k.x, k.grid, k.context)?, ..k }),
            None => None,
        };
        // Each norm as `rms(x)·a + b`: its weight, times `1 + scale` and
        // plus the shift where the step modulates it, in f32.
        let affine = |norm: &RmsNorm, scale: usize, shift: usize| -> candle_core::Result<(Tensor, Tensor)> {
            let w = norm.weight();
            match &m {
                Some(m) => Ok((w.broadcast_mul(&(m.narrow(0, scale, 1)? + 1.0)?.to_dtype(DType::F32)?.flatten_all()?)?, m.narrow(0, shift, 1)?.to_dtype(DType::F32)?.flatten_all()?)),
                None => Ok((w.clone(), w.zeros_like()?)),
            }
        };
        let (a1, b1) = affine(&self.norm1, 0, 1)?;
        let (a2, b2) = affine(&self.norm2, 3, 4)?;
        let normed = |norm: &RmsNorm, h: &Tensor, a: &Tensor, b: &Tensor, scale: usize, shift: usize| -> candle_core::Result<Tensor> {
            if let Some(y) = super::ltx_fused::norm_affine(h, a, b, EPS as f32)? {
                return Ok(y);
            }
            let y = norm.forward(h)?;
            match &m {
                Some(m) => y.broadcast_mul(&(m.narrow(0, scale, 1)? + 1.0)?)?.broadcast_add(&m.narrow(0, shift, 1)?),
                None => Ok(y),
            }
        };
        // q, k and v `[3, n, width]` of tokens `x` on `grid`, rotated by `rope`.
        let qkv_of = |x: &Tensor, grid: Grid, rope: &Rope| -> candle_core::Result<Tensor> {
            span(|| format!("{width}: norm, qkv, rope"), &dev, || {
                let (n, plane) = (grid.tokens(), grid.h * grid.w);
                let all = blank(&[3, n, width], dt, &dev)?;
                let d = width / self.attn.heads;
                for &(t0, t1) in &runs(grid, width) {
                    let (r0, len) = (t0 * plane, (t1 - t0) * plane);
                    let h = normed(&self.norm1, &x.narrow(0, r0, len)?, &a1, &b1, 0, 1)?;
                    let fused = self.attn.qkv.forward(&h)?.to_dtype(dt)?;
                    let place = super::ltx_fused::HeadRope { h: grid.h, w: grid.w, row0: r0, tables: &rope.packed, pairs: rope.pairs, lens: [grid.t, grid.h, grid.w] };
                    let (qn, kn) = (self.attn.q_norm.weight(), self.attn.k_norm.weight());
                    if d == 64 && super::ltx_fused::head_norm_rope(&fused, qn, kn, &place, EPS as f32, 1.0 / (d as f32).sqrt(), &all)? {
                        continue;
                    }
                    let (cos, sin) = rope.frames(t0, t1)?;
                    for (i, part) in self.attn.project(&fused, &cos, &sin)?.into_iter().enumerate() {
                        all.get(i)?.slice_set(&part, 0, r0)?;
                    }
                }
                Ok(all)
            })
        };
        // `x + proj(o)`, then its feed-forward.
        let finish = |x: &Tensor, o: &Tensor, grid: Grid| -> candle_core::Result<Tensor> {
            span(|| format!("{width}: proj, feed-forward"), &dev, || by_runs(grid.tokens(), width, dt, &dev, &runs(grid, width), grid.h * grid.w, |r0, len, _, _| {
                let y = (x.narrow(0, r0, len)? + self.attn.proj.forward(&o.narrow(0, r0, len)?)?.to_dtype(dt)?)?;
                let h = normed(&self.norm2, &y, &a2, &b2, 3, 4)?;
                &y + self.mlp.forward(&h)?.to_dtype(dt)?
            }))
        };
        let qkv = qkv_of(&x, grid, rope)?;
        let (kernel, heads) = (self.attn.kernel, self.attn.heads);
        match keys {
            None => {
                let o = span(|| format!("{width}: attention"), &dev, || neighbourhood(&qkv.get(0)?, &qkv.get(1)?, &qkv.get(2)?, grid, kernel, heads))?;
                drop(qkv);
                Ok((finish(&x, &o, grid)?, None))
            }
            Some(k) => {
                let pqkv = qkv_of(&k.x, k.grid, k.rope)?;
                let (o, po) = span(|| format!("{width}: joint attention"), &dev, || {
                    joint(&qkv.get(0)?, &qkv.get(1)?, &qkv.get(2)?, grid, &pqkv.get(0)?, &pqkv.get(1)?, &pqkv.get(2)?, k.grid, k.times, kernel, heads)
                })?;
                drop((qkv, pqkv));
                Ok((finish(&x, &o, grid)?, Some(finish(&k.x, &po, k.grid)?)))
            }
        }
    }
}

/// The keyframe planes' side of a block: their tokens on their grid
/// (planes, rows, columns), their rotation at their `times`, and at stage 5
/// their context.
struct Keyed<'a> {
    x: Tensor,
    grid: Grid,
    rope: &'a Rope,
    times: &'a [f32],
    context: Option<&'a Tensor>,
}

/// An upsampling: a linear layer to `p₁·p₂·p₃·c` channels, then each token's
/// channels unfolded into a block of `p₁×p₂×p₃` tokens of `c`, in the order
/// `((c·p₁ + i)·p₂ + j)·p₃ + k`. Doubling time drops the first frame.
struct Up {
    proj: Linear,
    stride: [usize; 3],
    out: usize,
}

impl Up {
    /// A run of frames at a time, as a block's token-wise work goes.
    fn forward(&self, x: &Tensor, grid: Grid, drop_first: bool) -> candle_core::Result<(Tensor, Grid)> {
        let [p1, p2, p3] = self.stride;
        let c = self.out;
        let g = Grid { t: grid.t * p1, h: grid.h * p2, w: grid.w * p3 };
        let plane = grid.h * grid.w;
        let (dt, dev) = (x.dtype(), x.device());
        let y = blank(&[g.tokens(), c], dt, dev)?;
        for (t0, t1) in runs(grid, c * p1 * p2 * p3 / 4) {
            let ft = t1 - t0;
            let part = self.proj.forward(&x.narrow(0, t0 * plane, ft * plane)?)?
                .reshape(&[ft, grid.h, grid.w, c, p1, p2, p3][..])?
                // [t, p1, h, p2, w, p3, c]
                .permute(&[0, 4, 1, 5, 2, 6, 3][..])?
                .contiguous()?
                .reshape((ft * p1 * g.h * g.w, c))?;
            y.slice_set(&part.to_dtype(dt)?, 0, t0 * p1 * g.h * g.w)?;
        }
        match (p1 == 2 && drop_first, g.h * g.w) {
            (true, plane) => Ok((y.narrow(0, plane, y.dim(0)? - plane)?, Grid { t: g.t - 1, ..g })),
            _ => Ok((y, g)),
        }
    }

    /// Keyframe planes `[P·h·w, C]` on `grid` (planes, rows, columns),
    /// each upsampled as a clip of one frame with its first dropped, the
    /// reference's `upsample_keyframe_planes`: in space only, since a
    /// doubling in time makes a plane two frames and keeps the second.
    fn planes(&self, x: &Tensor, grid: Grid) -> candle_core::Result<(Tensor, Grid)> {
        let p1 = self.stride[0];
        if p1 > 2 {
            candle_core::bail!("keyframe planes through a stride of {p1} in time");
        }
        let (y, g) = self.forward(x, grid, false)?;
        let plane = g.h * g.w;
        let y = y.reshape((grid.t, p1, plane, self.out))?.narrow(1, p1 - 1, 1)?.contiguous()?;
        Ok((y.reshape((grid.t * plane, self.out))?, Grid { t: grid.t, ..g }))
    }
}

/// The decoder: stages 1 to 4 turn a latent into a context, stage 5 turns
/// noise and the context into frames.
pub struct DiffDecoder {
    conv_in: Linear,
    /// The keyframe stream's tag, `[1, 128]` in f32: added to a plane's
    /// latent before `conv_in`, and nowhere else.
    type_emb: Tensor,
    /// The temporal upsampling still to come at each stage's input, and at
    /// stage 5, 1: the reference's `remaining_time_strides`, (8, 8, 4, 2, 1).
    time_strides: [usize; 5],
    stages: Vec<(Vec<Block>, Up)>,
    t1: Linear,
    t2: Linear,
    adaln: Linear,
    x_in: Linear,
    blocks: Vec<Block>,
    norm_out: RmsNorm,
    conv_out: Linear,
    split: [usize; 3],
    stage5_kernel: [usize; 3],
    /// Stage 4's stride, and how far, in stage-4 tokens, an edge reaches in
    /// through stages 4 and 5: a tile's overlap with the next.
    stride4: [usize; 3],
    halo: [usize; 3],
    /// The smallest tile, in stage-4 tokens, that every window fits.
    min_tile: [usize; 3],
    /// The latent's per-channel statistics, `[1, 128]`.
    mean: Tensor,
    std: Tensor,
    patch: usize,
    dtype: DType,
    device: Device,
    params: usize,
}

/// A config field as a list of numbers.
fn nums(v: &Value) -> Res<Vec<usize>> {
    v.as_array().ok_or("not a list")?.iter().map(|x| x.as_u64().map(|x| x as usize).ok_or_else(|| "not a number".into())).collect()
}

/// The most stage-5 tokens a tile may have, unless the caller says. A
/// 768 × 512 clip of 121 frames is 2.97 M, and decodes whole: in 11.5 s, at
/// a peak footprint of 15.9 GB, about 5.3 KB a token all told.
pub const BUDGET: usize = 3 << 20;

/// The seed of stage 5's noise for a generation of `seed`: turned, so that
/// it is not the stream the DiT's noise came from.
pub fn noise_seed(seed: u64) -> u64 {
    seed ^ 0x6469_6666_7661_6500
}

/// The seed of the keyframe planes' stage-5 noise, turned again.
fn plane_seed(seed: u64) -> u64 {
    noise_seed(seed) ^ 0x706c_616e_6573_0000
}

/// The two latent frames' worth of stage-4 frames that the decoder's input
/// repeats at its end and stage 4's answer loses: 2 latent frames, ×4 by
/// stage 4.
const GHOST_S4: usize = 8;

/// Keyframe planes decoded beside a video: their tokens `[P·h·w, C]` so
/// far on (planes, rows, columns), and each plane's pixel frame in the clip.
pub struct Planes {
    pub x: Tensor,
    pub grid: Grid,
    pub frames: Vec<usize>,
}

/// [`DiffDecoder::times`] for a stage whose upsampling in time to come is `r`.
fn stage_times(frames: &[usize], r: usize, origin: f32) -> Vec<f32> {
    let r = r as f32;
    frames.iter().map(|&f| if f == 0 { 0.0 } else { (f as f32 + (r - 1.0) / 2.0) / r } - origin).collect()
}

/// Which of the planes at pixel frames `frames` a tile of frames `lo ..=
/// hi` carries: those inside it, and the nearest outside it on each side,
/// the reference's `planes_for_tile`. Without those two a frame near the
/// tile's edge would see only the planes inside, and not the ones it sees
/// in a whole decode. Indices into `frames`, in its order.
pub fn planes_for_tile(frames: &[usize], lo: usize, hi: usize) -> Vec<usize> {
    let before = (0..frames.len()).filter(|&i| frames[i] < lo).max_by(|&a, &b| frames[a].cmp(&frames[b]).then(b.cmp(&a)));
    let after = (0..frames.len()).filter(|&i| frames[i] > hi).min_by(|&a, &b| frames[a].cmp(&frames[b]).then(a.cmp(&b)));
    (0..frames.len()).filter(|&i| (lo..=hi).contains(&frames[i]) || Some(i) == before || Some(i) == after).collect()
}

/// How a decode was laid out, and how long it took.
#[derive(Debug, Default)]
pub struct Report {
    /// Tiles along time, rows and columns.
    pub tiles: [usize; 3],
    /// Stage-5 tokens, in all tiles and in the largest.
    pub tokens: usize,
    pub largest: usize,
    /// Seconds in stages 1–3, and in stages 4–5 with the blending.
    pub stages_1_to_3: f64,
    pub stages_4_to_5: f64,
}

impl DiffDecoder {
    /// The decoder half of the DiffVAE's file at `path`, computing in
    /// `dtype` on `device`.
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Res<Self> {
        let config = metadata(path, "config")?;
        let vae = &config["vae"];
        let d = &vae["decoder"];
        if d["_class_name"] != "NADiffusionDecoder" {
            return Err(format!("{}: not LTX's diffusion decoder ({})", path.display(), d["_class_name"]).into());
        }
        // What the code below implements, checked rather than hoped.
        if vae["model_output_type"] != "x0" || d["default_num_inference_steps"] != 1 || d["resampler_kind"] != "linear" {
            return Err(format!("{}: a decoder of {} steps predicting {}; one step predicting x0 is implemented", path.display(), d["default_num_inference_steps"], vae["model_output_type"]).into());
        }
        let head_dim = d["head_dim"].as_u64().ok_or("no head_dim")? as usize;
        let channels = nums(&d["stage_channels"])?;
        let depths = nums(&d["stage_depths"])?;
        let kernels: Vec<[usize; 3]> = d["stage_kernels"].as_array().ok_or("no stage_kernels")?.iter().map(|k| nums(k).map(|k| [k[0], k[1], k[2]])).collect::<Res<_>>()?;
        let ups = d["upsamples"].as_array().ok_or("no upsamples")?;
        let s5 = nums(&d["stage5_kernel"])?;
        let s5 = [s5[0], s5[1], s5[2]];
        let patch = d["patch_size"].as_u64().ok_or("no patch_size")? as usize;
        if d["timestep_scale_multiplier"].as_f64() != Some(1000.0) {
            return Err(format!("{}: a timestep multiplier of {}", path.display(), d["timestep_scale_multiplier"]).into());
        }
        if channels.len() != 5 || depths.len() != 5 || kernels.len() < 4 || ups.len() != 4 {
            return Err(format!("{}: {} stages; four and a diffusion stage are implemented", path.display(), channels.len()).into());
        }

        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let paths = [path.to_path_buf()];
        let r = open(&paths, if dtype == DType::F32 { DType::F32 } else { DType::BF16 })?;
        r.skip_under("encoder");
        let m = r.pp("decoder");
        let latent = d["in_channels"].as_u64().unwrap_or(128) as usize;
        // The keyframe stream's tag; a checkpoint from before the keyframe
        // training has none, which the reference reads as zeros.
        let type_emb = match m.try_get(latent, "type_emb") {
            Some(t) => t.to_device(device)?.to_dtype(DType::F32)?.reshape((1, latent))?,
            None => Tensor::zeros((1, latent), DType::F32, device)?,
        };
        let conv_in = Linear::load(&cx, &m, "conv_in", latent, channels[0], true)?;
        let mut stages = Vec::new();
        let mut strides = Vec::new();
        for i in 0..channels.len() - 1 {
            let c = channels[i];
            let blocks = (0..depths[i]).map(|j| Block::load(&cx, &m.pp(format!("det_stages.{i}.{j}")), c, head_dim, kernels[i])).collect::<Res<Vec<_>>>()?;
            let stride = nums(&ups[i][0])?;
            let reduction = ups[i][1].as_u64().ok_or("no reduction")? as usize;
            let stride = [stride[0], stride[1], stride[2]];
            let n: usize = stride.iter().product();
            let up = Up { proj: Linear::load(&cx, &m.pp(format!("upsamples.{i}")), "proj", c, n * c / reduction, true)?, stride, out: c / reduction };
            stages.push((blocks, up));
            strides.push(stride);
        }
        let c5 = *channels.last().unwrap();
        let t_dim = 384;
        let blocks = (0..depths[4])
            .map(|j| {
                let b = m.pp(format!("diff_blocks.{j}"));
                let cond = Cond { context: Linear::load(&cx, &b, "context_proj", c5, c5, true)?, table: cx.get(&b, (7, c5), "scale_shift_table")? };
                Ok(Block { cond: Some(cond), ..Block::load(&cx, &b, c5, head_dim, s5)? })
            })
            .collect::<Res<Vec<_>>>()?;
        let pixels = 3 * patch * patch;
        let stats = r.pp("per_channel_statistics");
        let stat = |n: &str| -> Res<Tensor> { Ok(cx.get(&stats, latent, n)?.to_dtype(DType::F32)?.reshape((1, latent))?) };
        // The reference's `compute_tile_halos` and `compute_tile_min_size`:
        // an edge's error reaches in half a window a block, through stage 4's
        // blocks and, a stride coarser, stage 5's.
        let stride4 = strides[3];
        let halo = [0, 1, 2].map(|a| (depths[3] * (kernels[3][a] / 2)).max((depths[4] * (s5[a] / 2)).div_ceil(stride4[a])));
        let min_tile = [0, 1, 2].map(|a| kernels[3][a].max(s5[a].div_ceil(stride4[a])));
        let mut time_strides = [1; 5];
        for i in (0..4).rev() {
            time_strides[i] = time_strides[i + 1] * strides[i][0];
        }
        let dec = DiffDecoder {
            conv_in,
            type_emb,
            time_strides,
            stages,
            t1: Linear::load(&cx, &m.pp("t_embedder.mlp"), "0", 256, t_dim, true)?,
            t2: Linear::load(&cx, &m.pp("t_embedder.mlp"), "2", t_dim, t_dim, true)?,
            adaln: Linear::load(&cx, &m.pp("shared_adaln"), "proj", t_dim, 7 * c5, true)?,
            x_in: Linear::load(&cx, &m, "conv_in_x_t", pixels, c5, true)?,
            blocks,
            norm_out: RmsNorm::load(&cx, &m, "norm_out.weight", c5, EPS)?,
            conv_out: Linear::load(&cx, &m, "conv_out", c5, pixels, true)?,
            split: rope_split(head_dim),
            stage5_kernel: s5,
            stride4,
            halo,
            min_tile,
            mean: stat("mean-of-means")?,
            std: stat("std-of-means")?,
            patch,
            dtype,
            device: device.clone(),
            params: 0,
        };
        let params = finish("LTX diffusion decoder", &paths, &r)?;
        Ok(DiffDecoder { params, ..dec })
    }

    pub fn params(&self) -> usize {
        self.params
    }

    /// Stages 1 to 3 on a latent `[128, F, h, w]`, already padded: the
    /// stage-4 input as tokens, and its grid.
    pub fn stages_1_to_3(&self, latent: &Tensor) -> candle_core::Result<(Tensor, Grid)> {
        let (c, t, h, w) = latent.dims4()?;
        let g = Grid { t, h, w };
        let z = latent.to_dtype(DType::F32)?.reshape((c, g.tokens()))?.t()?.contiguous()?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?.to_dtype(self.dtype)?;
        let mut x = self.conv_in.forward(&z)?.to_dtype(self.dtype)?;
        let mut g = g;
        for (blocks, up) in &self.stages[..3] {
            (x, g) = self.stage(x, g, blocks, up, true)?;
        }
        Ok((x, g))
    }

    fn stage(&self, x: Tensor, g: Grid, blocks: &[Block], up: &Up, drop_first: bool) -> candle_core::Result<(Tensor, Grid)> {
        let rope = Rope::new(g, self.split, &self.device)?;
        let mut x = x;
        for b in blocks {
            x = b.forward(x, g, &rope, None)?;
            // As in stage 5: the pool lets the block's buffers go only here.
            self.device.synchronize()?;
        }
        let (y, g) = up.forward(&x, g, drop_first)?;
        self.device.synchronize()?;
        Ok((y.to_dtype(self.dtype)?, g))
    }

    /// Planes' places in stage `stage`'s time (0 to 3, and 4 for stage 5),
    /// less `origin` in its units: the reference's `keyframe_clip_times`
    /// for a decode from frame 0. With `r` the upsampling in time still to
    /// come, a stage's cells hold `r` pixel frames, but for the first, which
    /// holds frame 0 alone; frame `f` is at `(f + (r − 1)/2)/r`, the middle
    /// of its cell, and frame 0 at 0. In f32, in the reference's order.
    pub fn times(&self, frames: &[usize], stage: usize, origin: f32) -> Vec<f32> {
        stage_times(frames, self.time_strides[stage], origin)
    }

    /// Keyframe latents `[128, P, h, w]`, each one plane, at pixel frames
    /// `frames`, as the keyframe stream's first tokens: un-normalised,
    /// tagged with `type_emb`, and through the video's own `conv_in`.
    pub fn planes(&self, latents: &Tensor, frames: &[usize]) -> candle_core::Result<Planes> {
        let (c, p, h, w) = latents.dims4()?;
        if p != frames.len() {
            candle_core::bail!("{p} keyframe planes at {} frames", frames.len());
        }
        let z = latents.to_dtype(DType::F32)?.reshape((c, p * h * w))?.t()?.contiguous()?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?.broadcast_add(&self.type_emb)?.to_dtype(self.dtype)?;
        Ok(Planes { x: self.conv_in.forward(&z)?.to_dtype(self.dtype)?, grid: Grid { t: p, h, w }, frames: frames.to_vec() })
    }

    /// [`DiffDecoder::stages_1_to_3`] with keyframe planes beside the
    /// video, both streams through every block, the planes' times global.
    pub fn stages_1_to_3_keyed(&self, latent: &Tensor, planes: Planes) -> candle_core::Result<(Tensor, Grid, Planes)> {
        let (c, t, h, w) = latent.dims4()?;
        let g = Grid { t, h, w };
        let z = latent.to_dtype(DType::F32)?.reshape((c, g.tokens()))?.t()?.contiguous()?;
        let z = z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?.to_dtype(self.dtype)?;
        let (mut x, mut g, mut p) = (self.conv_in.forward(&z)?.to_dtype(self.dtype)?, g, planes);
        for i in 0..3 {
            (x, g, p) = self.stage_keyed(x, g, p, i, true, 0.0)?;
        }
        Ok((x, g, p))
    }

    /// Stage `i` on both streams, the planes' times less `origin` in its
    /// units; the planes upsampled each on its own ([`Up::planes`]).
    fn stage_keyed(&self, x: Tensor, g: Grid, planes: Planes, i: usize, drop_first: bool, origin: f32) -> candle_core::Result<(Tensor, Grid, Planes)> {
        let (blocks, up) = &self.stages[i];
        let times = self.times(&planes.frames, i, origin);
        let rope = Rope::new(g, self.split, &self.device)?;
        let prope = Rope::at(&times, planes.grid, self.split, &self.device)?;
        let (mut x, mut px) = (x, planes.x);
        for b in blocks {
            let keys = Keyed { x: px, grid: planes.grid, rope: &prope, times: &times, context: None };
            let (y, py) = b.forward_keyed(x, g, &rope, None, Some(keys))?;
            (x, px) = (y, py.ok_or_else(|| candle_core::Error::Msg("a keyed block without its planes".into()))?);
            self.device.synchronize()?;
        }
        let (y, g) = up.forward(&x, g, drop_first)?;
        let (py, pg) = up.planes(&px, planes.grid)?;
        self.device.synchronize()?;
        Ok((y.to_dtype(self.dtype)?, g, Planes { x: py.to_dtype(self.dtype)?, grid: pg, frames: planes.frames }))
    }

    /// [`DiffDecoder::stage_4`] with keyframe planes: the context and the
    /// planes' own, which stage 5 reads beside it.
    pub fn stage_4_keyed(&self, feat: &Tensor, grid: Grid, ghost: usize, planes: Planes) -> candle_core::Result<(Tensor, Grid, Planes)> {
        let (x, g, p) = self.stage_keyed(feat.clone(), grid, planes, 3, true, 0.0)?;
        let content = g.t.saturating_sub(ghost * 8).max(1);
        let keep = g.t.min(content.max(self.stage5_kernel[0]));
        Ok((x.narrow(0, 0, keep * g.h * g.w)?, Grid { t: keep, ..g }, p))
    }

    /// Stage 4, whole: the context, with the two repeated latent frames'
    /// worth (16 frames) cut from its end, to no fewer than stage 5's window.
    pub fn stage_4(&self, feat: &Tensor, grid: Grid, ghost: usize) -> candle_core::Result<(Tensor, Grid)> {
        let (x, g) = self.stage_4_tile(feat, grid, true)?;
        let content = g.t.saturating_sub(ghost * 8).max(1);
        let keep = g.t.min(content.max(self.stage5_kernel[0]));
        let plane = g.h * g.w;
        Ok((x.narrow(0, 0, keep * plane)?, Grid { t: keep, ..g }))
    }

    /// Stage 4 on a tile of its input: `origin`, the clip's first frames,
    /// drops the first upsampled frame as the whole clip does.
    fn stage_4_tile(&self, feat: &Tensor, grid: Grid, origin: bool) -> candle_core::Result<(Tensor, Grid)> {
        let (blocks, up) = &self.stages[3];
        self.stage(feat.clone(), grid, blocks, up, origin)
    }

    /// The step's seven modulation rows, `[7, 256]`, at `t = 1` as the
    /// one-step decoder runs it (`× 1000` into the sinusoid).
    fn rows(&self) -> candle_core::Result<Tensor> {
        let half = 128;
        let t = 1000f32;
        let angle = |i: usize| t * (-(10000f64.ln() as f32) * i as f32 / half as f32).exp();
        let e: Vec<f32> = (0..half).map(|i| angle(i).cos()).chain((0..half).map(|i| angle(i).sin())).collect();
        let e = Tensor::from_vec(e, (1, 2 * half), &self.device)?.to_dtype(self.dtype)?;
        let e = self.t2.forward(&self.t1.forward(&e)?.silu()?)?;
        let rows = self.adaln.forward(&e.silu()?)?;
        rows.reshape((7, rows.dim(1)? / 7))?.to_dtype(self.dtype)
    }

    /// Stage 5: the context `[n, 256]` on `grid` and noise `[T, 3, H, W]`
    /// (frames first, the grid's size in pixels), to the clean frames
    /// `[T, 3, H, W]` in `[−1, 1]`.
    pub fn stage_5(&self, context: &Tensor, grid: Grid, noise: &Tensor) -> candle_core::Result<Tensor> {
        self.stage_5_keyed(context, grid, noise, None)
    }

    /// [`DiffDecoder::stage_5`], and with `keys` keyframe planes beside the
    /// video: their context from [`DiffDecoder::stage_4_keyed`], their own
    /// noise `[P, 3, H, W]`, and the pixel frame the video's first is, their
    /// times' origin. The planes are a second stream of pixels, denoised
    /// with the video so that what the joint attention reads of them is at
    /// the noise it was trained at, and then dropped.
    pub fn stage_5_keyed(&self, context: &Tensor, grid: Grid, noise: &Tensor, keys: Option<(&Planes, &Tensor, f32)>) -> candle_core::Result<Tensor> {
        let p = self.patch;
        let (c5, plane) = (self.norm_out.weight().elem_count(), grid.h * grid.w);
        // Noise `[T, 3, H, W]` to its first tokens on `g`, `[T·h·w, 256]`.
        let x_in = |noise: &Tensor, g: Grid| -> candle_core::Result<Tensor> {
            let x = super::ltx_vae::patchify(&noise.to_dtype(self.dtype)?, p)?;
            let x = x.permute((0, 2, 3, 1))?.contiguous()?.reshape((g.tokens(), 3 * p * p))?;
            span(|| "stage 5 in", &self.device, || by_runs(g.tokens(), c5, self.dtype, &self.device, &runs(g, c5), g.h * g.w, |r0, len, _, _| {
                self.x_in.forward(&x.narrow(0, r0, len)?)
            }))
        };
        let spans = runs(grid, c5);
        let mut x = x_in(noise, grid)?;
        let rows = self.rows()?;
        let rope = Rope::new(grid, self.split, &self.device)?;
        let keyed = match keys {
            Some((planes, pnoise, origin)) => {
                let times = self.times(&planes.frames, 4, origin);
                let prope = Rope::at(&times, planes.grid, self.split, &self.device)?;
                Some((planes, x_in(pnoise, planes.grid)?, times, prope))
            }
            None => None,
        };
        let mut px = keyed.as_ref().map(|k| k.1.clone());
        for b in &self.blocks {
            let keys = keyed.as_ref().map(|(planes, _, times, prope)| Keyed {
                x: px.take().unwrap_or_else(|| planes.x.clone()),
                grid: planes.grid,
                rope: prope,
                times,
                context: Some(&planes.x),
            });
            let (y, py) = b.forward_keyed(x, grid, &rope, Some((context, &rows)), keys)?;
            (x, px) = (y, py);
            // Let the pool have the block's buffers back before the next.
            self.device.synchronize()?;
        }
        drop((px, keyed));
        span(|| "stage 5 out", &self.device, || {
            let y = by_runs(grid.tokens(), 3 * p * p, self.dtype, &self.device, &spans, plane, |r0, len, _, _| {
                self.conv_out.forward(&self.norm_out.forward(&x.narrow(0, r0, len)?)?)
            })?;
            let y = y.reshape((grid.t, grid.h, grid.w, 3 * p * p))?.permute((0, 3, 1, 2))?.contiguous()?;
            super::ltx_vae::unpatchify(&y, p)
        })
    }

    /// A latent `[128, F, h, w]` to its frames, `[8(F − 1) + 1, 3, 32h, 32w]`
    /// in `[0, 1]`, in f32 on the host, as `ltx_vae::VideoDecoder::decode`
    /// answers; stage 5 starting from the noise of `seed`. `progress` hears
    /// of each tile done, and of how many there are, and an `Err` from it
    /// stops the decode there.
    ///
    /// Stages 1 to 3 run on the whole clip. Stages 4 and 5 run on tiles of
    /// stage 4's input of at most `budget` stage-5 tokens, as the reference's
    /// tiled decode does: each tile overlaps the next by the halo, the
    /// distance an edge's error reaches in, and the overlaps are blended
    /// with complementary linear ramps. A clip that fits is one tile, and
    /// exact. Unlike the reference's one-step decode, which draws each
    /// tile's noise afresh, every tile here reads its block of one field of
    /// noise over the whole clip ([`noise_block`]), so tiles agree on the
    /// noise where they overlap and a tiled decode differs from the whole
    /// one only by the edges' error.
    pub fn decode(&self, latent: &Tensor, seed: u64, budget: usize, progress: &mut dyn FnMut(usize, usize) -> Res<()>) -> Res<(Tensor, Report)> {
        self.decode_keyed(latent, None, seed, budget, progress)
    }

    /// [`DiffDecoder::decode`] with keyframes: `keys`, latents `[128, P, h,
    /// w]`, one plane each, and their pixel frames, which every frame's
    /// attention reads beside its own window ([`joint`]), as DFR decodes.
    /// Each tile carries the planes [`planes_for_tile`] gives it, at times
    /// counted from its own first frame, and its planes' stage-5 noise from
    /// a field of its own, a frame for each plane.
    pub fn decode_keyed(&self, latent: &Tensor, keys: Option<(&Tensor, &[usize])>, seed: u64, budget: usize, progress: &mut dyn FnMut(usize, usize) -> Res<()>)
     -> Res<(Tensor, Report)> {
        let (_, f, lh, lw) = latent.dims4()?;
        let t0 = std::time::Instant::now();
        let last = latent.narrow(1, f - 1, 1)?;
        let padded = Tensor::cat(&[latent, &last, &last], 1)?;
        let (feat, g4, planes) = match keys {
            Some((k, frames)) => {
                let (x, g, p) = self.stages_1_to_3_keyed(&padded, self.planes(&k.to_device(&self.device)?, frames)?)?;
                (x, g, Some(p))
            }
            None => {
                let (x, g) = self.stages_1_to_3(&padded)?;
                (x, g, None)
            }
        };
        self.device.synchronize()?;
        let mut report = Report { stages_1_to_3: t0.elapsed().as_secs_f64(), ..Report::default() };
        let t0 = std::time::Instant::now();

        let content = g4.t - GHOST_S4;
        let frames = 8 * (f - 1) + 1;
        let (height, width) = (32 * lh, 32 * lw);
        let px = 2 * self.patch;
        // The noise field: the frames, or as many as stage 5's window wants.
        let field = [frames.max(self.stage5_kernel[0]), 3, height, width];
        let layout = self.tiles([content, g4.h, g4.w], budget)?;
        report.tiles = [layout[0].len(), layout[1].len(), layout[2].len()];
        let plane = height * width;
        let mut acc = vec![0f32; frames * 3 * plane];
        let feat = feat.reshape((g4.t, g4.h, g4.w, feat.dim(1)?))?;
        let total = report.tiles.iter().product();
        let mut done = 0;
        for (it, &(a, b)) in layout[0].iter().enumerate() {
            let (origin, trailing) = (a == 0, b == content);
            // The last tile carries the repeated frames, as the whole clip
            // does, and stage 4's answer loses them.
            let end = if trailing { g4.t } else { b };
            // The tile's frames at stage 5, and where they start in the clip.
            let own = 2 * (b - a) - usize::from(origin);
            let first = 2 * a - usize::from(!origin);
            let ramp_t = ramps(&layout[0], it, |x| 2 * x - usize::from(x > 0), |x| 2 * x - 1);
            for (ih, &(c, d)) in layout[1].iter().enumerate() {
                let ramp_h = ramps(&layout[1], ih, |x| px * x, |x| px * x);
                for (iw, &(e, g)) in layout[2].iter().enumerate() {
                    let ramp_w = ramps(&layout[2], iw, |x| px * x, |x| px * x);
                    let tile = feat.narrow(0, a, end - a)?.narrow(1, c, d - c)?.narrow(2, e, g - e)?.contiguous()?;
                    let grid = Grid { t: end - a, h: d - c, w: g - e };
                    let tile = tile.reshape((grid.tokens(), tile.dim(3)?))?;
                    // The tile's planes, cut to its rows and columns.
                    let (ctx, g5, tile_planes) = match &planes {
                        None => {
                            let (x, g) = span(|| "stage 4", &self.device, || self.stage_4_tile(&tile, grid, origin))?;
                            (x, g, None)
                        }
                        Some(p) => {
                            let pick = planes_for_tile(&p.frames, first, first + own.min(frames - first) - 1);
                            let at = Tensor::from_vec(pick.iter().map(|&i| i as u32).collect::<Vec<_>>(), pick.len(), &self.device)?;
                            let width = p.x.dim(1)?;
                            let px = p.x.reshape((p.grid.t, g4.h, g4.w, width))?.index_select(&at, 0)?.narrow(1, c, d - c)?.narrow(2, e, g - e)?.contiguous()?;
                            let pg = Grid { t: pick.len(), h: d - c, w: g - e };
                            let tp = Planes { x: px.reshape((pg.tokens(), width))?, grid: pg, frames: pick.iter().map(|&i| p.frames[i]).collect() };
                            let (x, g, tp) = span(|| "stage 4", &self.device, || self.stage_keyed(tile.clone(), grid, tp, 3, origin, a as f32))?;
                            (x, g, Some((tp, pick)))
                        }
                    };
                    drop(tile);
                    // The tile's frames, or stage 5's window of them.
                    let keep = g5.t.min(own.max(self.stage5_kernel[0]));
                    let ctx = ctx.narrow(0, 0, keep * g5.h * g5.w)?;
                    let g5 = Grid { t: keep, ..g5 };
                    report.tokens += g5.tokens();
                    report.largest = report.largest.max(g5.tokens());
                    let (h0, w0, th, tw) = (px * c, px * e, px * (d - c), px * (g - e));
                    let noise = span(|| "noise", &self.device, || {
                        Tensor::from_vec(noise_block(seed, &field, &[first, 0, h0, w0], &[keep, 3, th, tw]), (keep, 3, th, tw), &self.device)
                    })?;
                    let pixels = match &tile_planes {
                        None => self.stage_5(&ctx, g5, &noise)?,
                        Some((tp, pick)) => {
                            // A frame of their own field for each plane.
                            let all = planes.as_ref().map_or(0, |p| p.frames.len());
                            let pn = pick.iter().map(|&i| Tensor::from_vec(noise_block(plane_seed(seed), &[all, 3, height, width], &[i, 0, h0, w0], &[1, 3, th, tw]), (1, 3, th, tw), &self.device))
                                .collect::<candle_core::Result<Vec<_>>>()?;
                            self.stage_5_keyed(&ctx, g5, &noise, Some((tp, &Tensor::cat(&pn, 0)?, first as f32)))?
                        }
                    };
                    drop((ctx, noise, tile_planes));
                    let own = own.min(frames - first);
                    span(|| "blend", &self.device, || {
                        let pixels = pixels.narrow(0, 0, own)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        blend(&mut acc, &pixels, [first, h0, w0], [own, th, tw], [height, width], [&ramp_t, &ramp_h, &ramp_w]);
                        Ok(())
                    })?;
                    done += 1;
                    // An `Err` here, a cancel, stops before the next tile.
                    progress(done, total)?;
                }
            }
        }
        // [−1, 1] to [0, 1], as the reference's `to_rgb`, in place: the
        // frames of a 1536 × 1024 clip are 2.3 GB in f32.
        for v in acc.iter_mut() {
            *v = ((*v + 1.0) * 0.5).clamp(0.0, 1.0);
        }
        report.stages_4_to_5 = t0.elapsed().as_secs_f64();
        Ok((Tensor::from_vec(acc, (frames, 3, height, width), &Device::Cpu)?, report))
    }

    /// [`layout`] with this decoder's halo, windows and stride.
    fn tiles(&self, dims: [usize; 3], budget: usize) -> Res<[Vec<(usize, usize)>; 3]> {
        layout(dims, budget, self.halo, self.min_tile, self.stride4, self.stage5_kernel[0])
            .ok_or_else(|| format!("no tiling of {dims:?} keeps each tile under {budget} tokens").into())
    }
}

/// The tiles of stage 4's input `[t, h, w]` (content frames only) along
/// each axis, as `[start, end)` in stage-4 tokens: the fewest stage-5
/// tokens in all, with no tile over `budget`. Neighbours overlap by `halo`
/// at least; every tile is at least `min_tile` and twice the halo, so no
/// point is in more than two tiles along an axis. `stride` is stage 4's,
/// and stage 5 takes at least `frames5` frames.
fn layout(dims: [usize; 3], budget: usize, halo: [usize; 3], min_tile: [usize; 3], stride: [usize; 3], frames5: usize)
 -> Option<[Vec<(usize, usize)>; 3]> {
    let axis = |a: usize, n: usize| -> Option<Vec<(usize, usize)>> {
        let (len, o) = (dims[a], halo[a]);
        if n == 1 {
            return Some(vec![(0, len)]);
        }
        let size = (len + (n - 1) * o).div_ceil(n);
        if size < min_tile[a] || size < 2 * o || size >= len {
            return None;
        }
        let starts: Vec<usize> = (0..n).map(|k| if k + 1 == n { len - size } else { k * (size - o) }).collect();
        // Neighbours overlap by the halo at least, and two tiles apart
        // never meet.
        let ok = starts.windows(2).all(|s| s[1] > s[0] && s[1] + o <= s[0] + size)
            && starts.windows(3).all(|s| s[2] >= s[0] + size);
        ok.then(|| starts.iter().map(|&s| (s, s + size)).collect())
    };
    let stage5 = |t: usize, h: usize, w: usize, origin: bool| (stride[0] * t - usize::from(origin)).max(frames5) * stride[1] * h * stride[2] * w;
    let mut best: Option<(usize, [Vec<(usize, usize)>; 3])> = None;
    for nt in 1..=16 {
        let Some(lt) = axis(0, nt) else { continue };
        for nh in 1..=8 {
            let Some(lh) = axis(1, nh) else { continue };
            for nw in 1..=8 {
                let Some(lw) = axis(2, nw) else { continue };
                let (tt, hh, ww) = (lt[0].1 - lt[0].0, lh[0].1 - lh[0].0, lw[0].1 - lw[0].0);
                // Tiles along an axis are all one size; the clip's first
                // frame is dropped only where one tile has them all.
                let largest = stage5(tt, hh, ww, nt == 1);
                let all = nt * nh * nw * largest;
                if largest <= budget && best.as_ref().is_none_or(|(b, _)| all < *b) {
                    best = Some((all, [lt.clone(), lh.clone(), lw]));
                }
            }
        }
    }
    best.map(|(_, l)| l)
}

/// A tile's weights along one axis, from its `[start, end)` in stage-4
/// tokens: 1 inside, rising linearly across its overlap with the tile
/// before, falling across its overlap with the tile after, so that two
/// tiles' weights sum to 1 wherever they meet. `lo` and `hi` map a tile's
/// start and end to the output's units.
fn ramps(tiles: &[(usize, usize)], i: usize, lo: impl Fn(usize) -> usize, hi: impl Fn(usize) -> usize) -> Vec<f32> {
    let (s, e) = (lo(tiles[i].0), hi(tiles[i].1));
    let before = i.checked_sub(1).map(|p| hi(tiles[p].1));
    let after = tiles.get(i + 1).map(|&(n, _)| lo(n));
    (s..e)
        .map(|x| {
            let rise = match before {
                Some(end) if x < end => ((x - s) as f32 + 0.5) / (end - s) as f32,
                _ => 1.0,
            };
            let fall = match after {
                Some(n) if x >= n => ((e - x) as f32 - 0.5) / (e - n) as f32,
                _ => 1.0,
            };
            rise * fall
        })
        .collect()
}

/// `pixels` `[own, 3, th, tw]`, weighted by the product of each axis's
/// ramp, added into `acc` `[T, 3, height, width]` at `at` (frame, row,
/// column). A frame at a time, over the machine's cores.
fn blend(acc: &mut [f32], pixels: &[f32], at: [usize; 3], size: [usize; 3], dims: [usize; 2], ramps: [&[f32]; 3]) {
    let [own, th, tw] = size;
    let [height, width] = dims;
    let frame = 3 * height * width;
    let span = &mut acc[at[0] * frame..(at[0] + own) * frame];
    std::thread::scope(|sc| {
        for (f, out) in span.chunks_mut(frame).enumerate() {
            sc.spawn(move || {
                for ch in 0..3 {
                    for y in 0..th {
                        let wy = ramps[0][f] * ramps[1][y];
                        let src = &pixels[((f * 3 + ch) * th + y) * tw..][..tw];
                        let dst = &mut out[(ch * height + at[1] + y) * width + at[2]..][..tw];
                        for x in 0..tw {
                            dst[x] += wy * ramps[2][x] * src[x];
                        }
                    }
                }
            });
        }
    });
}

/// How a head's 64 dimensions are shared among time, rows and columns: a
/// quarter, even, for time, and the rest halved, as the reference's
/// `default_rope_dim_split`: 16, 24, 24.
fn rope_split(head_dim: usize) -> [usize; 3] {
    let mut t = head_dim / 4 / 2 * 2;
    let mut hw = (head_dim - t) / 2;
    if hw % 2 != 0 {
        t -= 2;
        hw = (head_dim - t) / 2;
    }
    [t, hw, hw]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_shift_inward_at_the_edges() {
        // 7 wide on 10: centred in the middle, held at 0 and 3 at the ends.
        assert_eq!(window_start(10, 7), [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]);
        assert_eq!(window_start(3, 3), [0, 0, 0]);
        assert_eq!(rope_split(64), [16, 24, 24]);
    }

    /// The decoder's own numbers: halo, smallest tile, stage 4's stride,
    /// stage 5's frames.
    const HALO: [usize; 3] = [20, 20, 20];
    const MIN: [usize; 3] = [6, 6, 6];
    const STRIDE: [usize; 3] = [2, 2, 2];

    /// Every axis's tiles cover it, start and end inside it, overlap their
    /// neighbours by the halo at least, and never three deep; none is over
    /// the budget; and the answer is whole when it fits.
    #[test]
    fn tiles_cover_and_overlap_by_the_halo() {
        // 768 × 512 and 1536 × 1024, 121 frames; 20 s of 768 × 512.
        let whole = layout([61, 64, 96], 3 << 20, HALO, MIN, STRIDE, 11).unwrap();
        assert_eq!(whole, [vec![(0, 61)], vec![(0, 64)], vec![(0, 96)]]);
        for (dims, budget) in [([61, 128, 192], 3 << 20), ([61, 64, 96], 700_000), ([237, 64, 96], 3 << 20)] {
            let l = layout(dims, budget, HALO, MIN, STRIDE, 11).unwrap_or_else(|| panic!("no layout of {dims:?}"));
            for (a, tiles) in l.iter().enumerate() {
                assert_eq!((tiles[0].0, tiles.last().unwrap().1), (0, dims[a]), "{dims:?} axis {a}: {tiles:?}");
                for p in tiles.windows(2) {
                    assert!(p[1].0 > p[0].0 && p[1].0 + HALO[a] <= p[0].1, "{dims:?} axis {a}: {tiles:?}");
                }
                for p in tiles.windows(3) {
                    assert!(p[2].0 >= p[0].1, "{dims:?} axis {a}: three deep, {tiles:?}");
                }
            }
            let size = |a: usize| l[a][0].1 - l[a][0].0;
            let frames = (2 * size(0) - usize::from(l[0].len() == 1)).max(11);
            assert!(frames * 2 * size(1) * 2 * size(2) <= budget, "{dims:?}: {l:?} over {budget}");
        }
        // Nothing fits a budget below the smallest tile.
        assert!(layout([61, 64, 96], 1000, HALO, MIN, STRIDE, 11).is_none());
    }

    /// Where tiles meet, their weights sum to one: in pixels along rows and
    /// columns, and in frames along time, where a tile past the first
    /// starts a frame early.
    #[test]
    fn ramps_sum_to_one() {
        let tiles = [(0, 41), (20, 61)];
        let space = |x: usize| 8 * x;
        let (lo, hi) = (|x: usize| 2 * x - usize::from(x > 0), |x: usize| 2 * x - 1);
        for (len, starts, weights) in [
            (8 * 61, tiles.map(|t| space(t.0)), [ramps(&tiles, 0, space, space), ramps(&tiles, 1, space, space)]),
            (2 * 61 - 1, tiles.map(|t| lo(t.0)), [ramps(&tiles, 0, lo, hi), ramps(&tiles, 1, lo, hi)]),
        ] {
            let mut sum = vec![0f32; len];
            for (s, w) in starts.iter().zip(&weights) {
                for (i, v) in w.iter().enumerate() {
                    sum[s + i] += v;
                }
            }
            assert!(sum.iter().all(|&v| (v - 1.0).abs() < 1e-6), "{sum:?}");
        }
    }

    /// Against attention written out: every query's window listed, its
    /// scores softmaxed, its values weighed.
    #[test]
    fn neighbourhood_attention_is_attention_within_each_window() {
        let dev = Device::Cpu;
        let g = Grid { t: 3, h: 5, w: 4 };
        let (heads, d, kernel) = (2, 4, [3, 3, 3]);
        let n = g.tokens();
        let r = |s: u64| crate::image::nn::noise(s, &[n, heads * d], &dev, DType::F32).unwrap();
        let (q, k, v) = (r(1), r(2), r(3));
        let got = plain(&q, &k, &v, g, kernel, heads).unwrap().to_vec2::<f32>().unwrap();
        let (qv, kv, vv) = (q.to_vec2::<f32>().unwrap(), k.to_vec2::<f32>().unwrap(), v.to_vec2::<f32>().unwrap());
        let (st, sh, sw) = (window_start(3, 3), window_start(5, 3), window_start(4, 3));
        for t in 0..g.t {
            for h in 0..g.h {
                for w in 0..g.w {
                    let i = (t * g.h + h) * g.w + w;
                    let keys: Vec<usize> = (0..27).map(|o| ((st[t] + o / 9) * g.h + sh[h] + o / 3 % 3) * g.w + sw[w] + o % 3).collect();
                    for hd in 0..heads {
                        let s: Vec<f32> = keys.iter().map(|&j| (0..d).map(|x| qv[i][hd * d + x] * kv[j][hd * d + x]).sum()).collect();
                        let m = s.iter().cloned().fold(f32::MIN, f32::max);
                        let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                        let z: f32 = e.iter().sum();
                        for x in 0..d {
                            let want: f32 = keys.iter().zip(&e).map(|(&j, p)| p / z * vv[j][hd * d + x]).sum();
                            assert!((got[i][hd * d + x] - want).abs() < 1e-5, "token {i} head {hd}");
                        }
                    }
                }
            }
        }
    }

    /// Planes in the stages' time: frame 0 at 0, and any other frame in
    /// the middle of the cell that holds it; less a tile's origin.
    #[test]
    fn planes_sit_in_the_middle_of_their_cells() {
        assert_eq!(stage_times(&[0, 8, 16], 8, 0.0), vec![0.0, 1.4375, 2.4375]);
        assert_eq!(stage_times(&[8, 16], 2, 0.0), vec![4.25, 8.25]);
        assert_eq!(stage_times(&[8, 16], 1, 5.0), vec![3.0, 11.0]);
    }

    /// The two nearest, the lower index on a tie, −1 when there are fewer.
    #[test]
    fn slots_rank_by_distance_then_index() {
        let (video, planes) = slots(&[1.5, 3.0], 5);
        assert_eq!(video, vec![[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]]);
        // Plane 0 at 1.5 is as near frames 1 and 2: 1 first.
        assert_eq!(planes, vec![[1, 2], [3, 2]]);
        let (video, planes) = slots(&[2.0], 1);
        assert_eq!((video, planes), (vec![[0, -1]], vec![[0, -1]]));
    }

    /// A tile's planes are those inside it and the nearest on each side.
    #[test]
    fn a_tile_carries_its_planes_and_their_neighbours() {
        let frames = [24, 48, 72, 96, 120];
        assert_eq!(planes_for_tile(&frames, 50, 90), vec![1, 2, 3]);
        assert_eq!(planes_for_tile(&frames, 0, 30), vec![0, 1]);
        assert_eq!(planes_for_tile(&frames, 100, 130), vec![3, 4]);
        // None inside: the two around it.
        assert_eq!(planes_for_tile(&frames, 50, 60), vec![1, 2]);
    }
}
