//! LTX's 3×3×3 convolutions on the M5 GPU's neural accelerators (#56).
//!
//! `video::conv3d` builds a 3D convolution out of candle's `conv2d`, which
//! copies every 3×3 neighbourhood out into a row of its own (`im2col`)
//! before it multiplies. #56 measured three quarters of each convolution in
//! that copy and the layout changes around it, and the decoder's residual
//! stages at 0.9–2.6 TFLOP/s. At 1536×1024 the decode is 284 s, more than
//! stage 2 of the DiT.
//!
//! This is an implicit-GEMM convolution: the same matrix product, with the
//! neighbourhoods gathered as it goes and never written out.
//!
//! # The product
//!
//! For one output frame, `Y[co, p] = b[co] + Σₖ W[co, k] · X[k, p]`, where
//! `p` runs over the frame's pixels and `k` over the 27 taps × the input
//! channels. `W` is the kernel stored `[out, 27·in]`, tap major, taps in
//! `(t, y, x)` order: [`taps`] lays it out once, at load. `X[k, p]` is the
//! input at pixel `p` shifted by tap `k`'s offset, from the frame before,
//! the frame itself or the frame after.
//!
//! A threadgroup owns a tile of `BM` output channels × `BN` pixels of one
//! output frame, and walks `k` 32 at a time: one tap and 32 of its channels.
//! At each step its threads gather that `32 × BN` slab of `X` into
//! threadgroup memory, and `matmul2d` multiplies the `BM × 32` slab of `W`,
//! read straight from device memory, by it, adding into a cooperative
//! tensor in f32. As in `mpp`'s Q8_0 kernel, there are two slabs, and the
//! next step's gather goes out before this step's multiply.
//!
//! Each thread keeps one column of the slab, so one pixel, for the whole
//! walk: its coordinates are worked out once, and at each step it reads a
//! channel after channel at one shifted position. Neighbouring threads read
//! neighbouring pixels.
//!
//! # Edges
//!
//! Space is padded with zeros. Time is padded as the caller says: the
//! decoder repeats the clip's first and last frames, the latent upsampler
//! reads zeros. As with [`crate::video::conv3d::Conv3d::frames`], the input
//! may be a slice of the clip, and only the clip's own first and last frames
//! are padded. Pixels past the frame's end and output channels past the
//! last are computed on zeros and not written.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;

const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Params {
    int cin, cout, h, w;
    // `x` holds `frames` frames of the clip, from frame `start`, of `total`.
    int frames, start, total;
    // The first output frame, in the clip.
    int lo;
    // Time padding: zeros, or the edge frame again.
    int zeros;
};

// `f(0)` to `f(N - 1)`, each index a compile-time constant, so that the
// arrays indexed by it stay in registers (see `mpp_attention`).
template <int I, int N>
struct each {
    template <typename F>
    static __attribute__((always_inline)) void run(thread const F &f) {
        f(integral_constant<int, I>());
        each<I + 1, N>::run(f);
    }
};
template <int N>
struct each<N, N> {
    template <typename F>
    static __attribute__((always_inline)) void run(thread const F &) {}
};
#define EACH(N, I, ...) each<0, N>::run([&](auto I##_) { constexpr int I = decltype(I##_)::value; __VA_ARGS__ })

// Channels a step.
constant constexpr int BK = 32;

template <typename T, int BM, int BN, int NSG>
[[kernel, max_total_threads_per_threadgroup(32 * NSG)]] void conv3d(
        device const T *x [[buffer(0)]],
        device const T *wt [[buffer(1)]],
        device const T *bias [[buffer(2)]],
        device T *y [[buffer(3)]],
        constant Params &p [[buffer(4)]],
        threadgroup T *slab [[threadgroup(0)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int THREADS = 32 * NSG;
    // Each thread's share of a slab: PER channels of one pixel, STRIDE
    // channels apart.
    constexpr int STRIDE = THREADS / BN;
    constexpr int PER = BK / STRIDE;
    static_assert(STRIDE * BN == THREADS && PER * STRIDE == BK, "every thread gathers the same share");
    const int hw = p.h * p.w;
    const int K = 27 * p.cin;
    const int p0 = int(tg.x) * BN;
    const int co0 = int(tg.y) * BM;
    const int f = p.lo + int(tg.z);

    // This thread's column: one pixel for the whole walk.
    const int col = tid % BN;
    const int row0 = tid / BN;
    const int px = p0 + col;
    const bool inside = px < hw;
    const int py = inside ? px / p.w : 0;
    const int pxx = inside ? px % p.w : 0;

    tensor<device T, dextents<int32_t, 2>, tensor_inline> ta((device T *)wt, dextents<int32_t, 2>(K, p.cout));
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> tb0(slab, dextents<int32_t, 2>(BN, BK));
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> tb1(slab + BK * BN, dextents<int32_t, 2>(BN, BK));
    constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto ma = ta.slice(0, co0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(ma), decltype(tb0), float>();
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }

    // Step s is tap s / blocks, channels (s % blocks)·32 on: the slab of W
    // at column s·32, since W is tap major.
    const int blocks = p.cin / BK;
    const int steps = 27 * blocks;
    T v[PER];
    // Step s's share of the gather, into registers. Where the tap reads
    // outside the clip or the frame, zeros; the address is then kept inside
    // the buffer, so no read strays past it.
    auto fetch = [&](int s) {
        const int tap = s / blocks;
        const int c0 = (s % blocks) * BK;
        const int kt = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
        int g = f + kt - 1;
        bool ok = inside;
        if (g < 0) {
            ok = ok && !p.zeros;
            g = 0;
        }
        if (g >= p.total) {
            ok = ok && !p.zeros;
            g = p.total - 1;
        }
        const int sy = py + ky - 1, sx = pxx + kx - 1;
        ok = ok && sy >= 0 && sy < p.h && sx >= 0 && sx < p.w;
        const long at = ok ? (long(g - p.start) * p.cin + c0 + row0) * hw + sy * p.w + sx : 0;
        EACH(PER, k, v[k] = ok ? x[at + long(k * STRIDE) * hw] : T(0););
    };
    auto stash = [&](threadgroup T *dst) {
        EACH(PER, k, dst[(row0 + k * STRIDE) * BN + col] = v[k];);
    };

    fetch(0);
    stash(slab);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 0; s < steps; ++s) {
        const bool more = s + 1 < steps;
        // The next step's reads go out before this step's multiply.
        if (more) {
            fetch(s + 1);
        }
        auto sa = ta.slice(s * BK, co0);
        if (s & 1) {
            op.run(sa, tb1, acc);
        } else {
            op.run(sa, tb0, acc);
        }
        if (more) {
            stash(slab + ((s + 1) & 1) * BK * BN);
        }
        // The slab just written is whole before anyone multiplies from it,
        // and the one it will overwrite next has been read by everyone.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto at = acc.get_multidimensional_index(i);
            const int pp = p0 + at[0], co = co0 + at[1];
            if (pp < hw && co < p.cout) {
                y[(long(tg.z) * p.cout + co) * hw + pp] = T(acc[i] + float(bias[co]));
            }
        }
    }
}

#define CONV(T, TN, BM, BN, NSG) \
    template [[host_name("conv3d_" #TN "_" #BM "x" #BN)]] [[kernel]] \
    decltype(conv3d<T, BM, BN, NSG>) conv3d<T, BM, BN, NSG>;
CONV(half, f16, 64, 64, 4)
CONV(half, f16, 128, 128, 8)
CONV(bfloat, bf16, 64, 64, 4)
CONV(bfloat, bf16, 128, 128, 8)
"#;

/// Output channels × pixels a threadgroup, and its SIMD groups.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tile {
    pub(crate) bm: usize,
    pub(crate) bn: usize,
    pub(crate) sg: usize,
}

pub(crate) const TILES: [Tile; 2] = [Tile { bm: 64, bn: 64, sg: 4 }, Tile { bm: 128, bn: 128, sg: 8 }];

/// 128 × 128 over eight SIMD groups: the faster of the two at every level of
/// the decoder `conv3d_race` tried, 6.9–16.6× candle's convolution against
/// 64 × 64's 6.5–14×.
const TILE: Tile = TILES[1];

#[repr(C)]
struct Params {
    cin: i32,
    cout: i32,
    h: i32,
    w: i32,
    frames: i32,
    start: i32,
    total: i32,
    lo: i32,
    zeros: i32,
}

/// The kernels: `None` where `mpp` does not run, or where
/// `KVAD_GPU_MPP_CONV=0` leaves the convolutions to candle.
fn kernels(device: &candle_core::Device) -> Option<&'static crate::fused::metal::Kernels> {
    if matches!(std::env::var("KVAD_GPU_MPP_CONV").as_deref(), Ok("0") | Ok("false")) {
        return None;
    }
    crate::fused::metal::tensor_library(device, "conv3d", SOURCE)
}

/// Whether [`conv3d`] runs on `device` in `dtype` with `cin` input
/// channels: f16 or bf16, on an M5, with channels in whole steps of 32.
pub(crate) fn runs(device: &candle_core::Device, dtype: DType, cin: usize) -> bool {
    matches!(dtype, DType::F16 | DType::BF16) && cin % 32 == 0 && kernels(device).is_some()
}

/// `[out, in, 3, 3, 3]` to the `[out, 27·in]` [`conv3d`] reads: tap major,
/// taps in `(t, y, x)` order, channels within each.
pub(crate) fn taps(w: &Tensor) -> candle_core::Result<Tensor> {
    let (o, i, kt, kh, kw) = w.dims5()?;
    w.permute((0, 2, 3, 4, 1))?.contiguous()?.reshape((o, kt * kh * kw * i))
}

/// Output frames `lo .. hi` of a clip `total` frames long, from `x`
/// (`[frames, in, h, w]`, the clip's frames from `start` on), with the
/// kernel `w` laid out by [`taps`] and the bias `b` (`[out]`). `zeros` pads
/// time with zeros instead of the edge frames. `[hi − lo, out, h, w]`.
pub(crate) fn conv3d(x: &Tensor, w: &Tensor, b: &Tensor, (start, total): (usize, usize), (lo, hi): (usize, usize), zeros: bool)
 -> candle_core::Result<Tensor> {
    conv3d_with(x, w, b, (start, total), (lo, hi), zeros, TILE)
}

/// [`conv3d`] with a given [`Tile`]: for the tests, and for measuring.
pub(crate) fn conv3d_with(x: &Tensor, w: &Tensor, b: &Tensor, clip: (usize, usize), (lo, hi): (usize, usize), zeros: bool, tile: Tile)
 -> candle_core::Result<Tensor> {
    let (frames, cin, _, _) = x.dims4()?;
    let (start, total) = clip;
    if lo >= hi || hi > total || lo.saturating_sub(1) < start || (hi + 1).min(total) > start + frames {
        candle_core::bail!("conv3d: output frames {lo}..{hi} of {total} need frames the {frames} from {start} do not hold");
    }
    if w.dim(1)? != 27 * cin {
        candle_core::bail!("conv3d: a [{}, {}] kernel for {cin} input channels", w.dim(0)?, w.dim(1)?);
    }
    let op = Conv { clip, lo, n: hi - lo, zeros, tile };
    x.contiguous()?.apply_op3_no_bwd(&w.contiguous()?, &b.contiguous()?, &op)
}

struct Conv {
    clip: (usize, usize),
    lo: usize,
    n: usize,
    zeros: bool,
    tile: Tile,
}

impl CustomOp3 for Conv {
    fn name(&self) -> &'static str {
        "mpp_conv3d"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout)
     -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("mpp_conv3d runs on Metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, w: &MetalStorage, lw: &Layout, b: &MetalStorage, lb: &Layout)
     -> candle_core::Result<(MetalStorage, Shape)> {
        use crate::fused::metal::{buffer, output};
        let dt = x.dtype();
        let dev = x.device();
        let (frames, cin, h, wd) = lx.shape().dims4()?;
        let cout = lw.shape().dims2()?.0;
        if w.dtype() != dt || b.dtype() != dt || !matches!(dt, DType::F16 | DType::BF16) || cin % 32 != 0 {
            candle_core::bail!("mpp_conv3d: {dt:?} input, {:?} kernel, {:?} bias, {cin} channels", w.dtype(), b.dtype());
        }
        let Some(lib) = kernels(&candle_core::Device::Metal(dev.clone())) else {
            candle_core::bail!("mpp_conv3d: this device cannot run it");
        };
        let Tile { bm, bn, sg } = self.tile;
        let tn = if dt == DType::F16 { "f16" } else { "bf16" };
        let pipe = lib.pipe(&format!("conv3d_{tn}_{bm}x{bn}"))?;
        let elems = self.n * cout * h * wd;
        let out = output(dev, elems * dt.size_in_bytes())?;
        let params = Params {
            cin: cin as i32,
            cout: cout as i32,
            h: h as i32,
            w: wd as i32,
            frames: frames as i32,
            start: self.clip.0 as i32,
            total: self.clip.1 as i32,
            lo: self.lo as i32,
            zeros: self.zeros as i32,
        };
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_label("mpp_conv3d");
        enc.set_compute_pipeline_state(&pipe);
        for (i, (s, l)) in [(x, lx), (w, lw), (b, lb)].into_iter().enumerate() {
            let (buf, at) = buffer(s, l);
            enc.set_input_buffer(i, Some(&buf), at);
        }
        enc.set_output_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &params);
        // Two slabs of 32 × BN.
        enc.set_threadgroup_memory_length(0, 2 * 32 * bn * dt.size_in_bytes());
        enc.dispatch_thread_groups(
            MTLSize { width: (h * wd).div_ceil(bn), height: cout.div_ceil(bm), depth: self.n },
            // 32 × SIMD groups, as `mpp` found faster than a row of threads.
            MTLSize { width: 32, height: sg, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), elems, dt), Shape::from((self.n, cout, h, wd))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::conv3d::Time;
    use candle_core::Device;

    /// The device, where it has matrix units. There the kernels must build.
    fn gpu() -> Option<Device> {
        let dev = Device::new_metal(0).ok()?;
        if !crate::mpp::available(&dev) {
            return None;
        }
        assert!(kernels(&dev).is_some(), "the conv3d kernels did not build");
        Some(dev)
    }

    fn db(got: &Tensor, want: &Tensor) -> f32 {
        let got = got.to_dtype(DType::F32).unwrap();
        assert!(got.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite(), "a NaN or an infinity: not all written");
        let err = (&got - want).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let sig = want.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        10.0 * (sig / err.max(1e-30)).log10()
    }

    /// The kernel against the folded `conv2d` convolution, which agrees
    /// with Lightricks' reference to 122 dB, run in f32 on the same rounded
    /// numbers. Both tiles, both time paddings; frames and channels that
    /// fill no tile, a frame narrower than a tile, and a slice of a clip.
    #[test]
    fn agrees_with_the_folded_convolution() {
        let Some(dev) = gpu() else {
            eprintln!("no matrix units for matmul2d; nothing to test");
            return;
        };
        // (frames, cin, cout, h, w)
        for (t, cin, cout, h, w) in [(3, 32, 48, 5, 7), (4, 64, 128, 16, 24), (2, 96, 200, 9, 70)] {
            for dt in [DType::BF16, DType::F16] {
                let x = Tensor::randn(0f32, 1.0, (t, cin, h, w), &dev).unwrap().to_dtype(dt).unwrap();
                let k = (Tensor::randn(0f32, 1.0, (cout, cin, 3, 3, 3), &dev).unwrap() / (27.0 * cin as f64).sqrt()).unwrap().to_dtype(dt).unwrap();
                let b = Tensor::randn(0f32, 1.0, cout, &dev).unwrap().to_dtype(dt).unwrap();
                let wt = taps(&k).unwrap();
                for time in [Time::Replicate, Time::Zeros] {
                    let f32 = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
                    let folded = crate::video::conv3d::Conv3d::folded(&f32(&k), f32(&b), time).unwrap();
                    let want = folded.forward(&f32(&x)).unwrap();
                    for tile in TILES {
                        let before = crate::fused::tests_ran();
                        let got = conv3d_with(&x, &wt, &b, (0, t), (0, t), time == Time::Zeros, tile).unwrap();
                        assert_eq!(crate::fused::tests_ran(), before + 1, "the kernel did not run");
                        assert_eq!(got.dims(), &[t, cout, h, w]);
                        let d = db(&got, &want);
                        // Rounding the output to bf16 alone costs about 50 dB.
                        let floor = if dt == DType::BF16 { 45.0 } else { 60.0 };
                        assert!(d > floor, "{dt:?} {time:?} {tile:?} [{t}, {cin}, {h}, {w}] → {cout}: {d:.1} dB");
                        // A slice: frames 1 .. t − 1 of the clip, from the
                        // frames they read.
                        if t >= 3 {
                            let (lo, hi) = (1, t - 1);
                            let got = conv3d_with(&x.narrow(0, lo - 1, hi - lo + 2).unwrap(), &wt, &b, (lo - 1, t), (lo, hi), time == Time::Zeros, tile).unwrap();
                            let d = db(&got, &want.narrow(0, lo, hi - lo).unwrap());
                            assert!(d > floor, "slice {dt:?} {time:?} {tile:?}: {d:.1} dB");
                        }
                    }
                }
            }
        }
    }

    /// Not a test, a measurement: the kernel's tiles against the folded
    /// convolution at the decoder's costliest levels, on a few frames.
    ///
    ///     cargo test --release -p kvad-gpu conv3d_race -- --ignored --nocapture
    #[test]
    #[ignore]
    fn conv3d_race() {
        let dev = gpu().expect("no matrix units");
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        // (what, channels, h, w), 6 output frames each: the residual stages
        // #84 profiled at 1536×1024, and the 768×512 decode's widest.
        for (what, c, h, w) in [("512 × 128×192", 512, 128, 192), ("256 × 128×192", 256, 128, 192), ("128 × 256×384", 128, 256, 384), ("256 × 64×96", 256, 64, 96)] {
            let t = 6;
            let x = Tensor::randn(0f32, 1.0, (t + 2, c, h, w), &dev).unwrap().to_dtype(DType::BF16).unwrap();
            let k = (Tensor::randn(0f32, 1.0, (c, c, 3, 3, 3), &dev).unwrap() / (27.0 * c as f64).sqrt()).unwrap().to_dtype(DType::BF16).unwrap();
            let b = Tensor::zeros(c, DType::BF16, &dev).unwrap();
            let wt = taps(&k).unwrap();
            let folded = crate::video::conv3d::Conv3d::folded(&k, b.clone(), Time::Replicate).unwrap();
            let flops = 2.0 * 27.0 * (c * c * h * w * t) as f64;
            let time = |f: &dyn Fn()| {
                dev.synchronize().unwrap();
                let s = std::time::Instant::now();
                f();
                dev.synchronize().unwrap();
                s.elapsed().as_secs_f64()
            };
            let candle = || drop(folded.frames(&x, (0, t + 2), (1, t + 1)).unwrap());
            let ours: Vec<Box<dyn Fn()>> = TILES.iter().map(|&tile| -> Box<dyn Fn()> {
                let (x, wt, b) = (x.clone(), wt.clone(), b.clone());
                Box::new(move || drop(conv3d_with(&x, &wt, &b, (0, t + 2), (1, t + 1), false, tile).unwrap()))
            }).collect();
            time(&candle);
            ours.iter().for_each(|f| {
                time(f.as_ref());
            });
            let mut base = vec![];
            let mut each = vec![vec![]; TILES.len()];
            for _ in 0..5 {
                base.push(time(&candle));
                for (f, e) in ours.iter().zip(&mut each) {
                    e.push(time(f.as_ref()));
                }
            }
            let bt = median(base);
            print!("{what:<16} candle {:8.1} ms {:5.1} TFLOP/s", bt * 1e3, flops / bt / 1e12);
            for (tile, e) in TILES.iter().zip(each) {
                let s = median(e);
                print!(" | {}x{} {:5.1} TFLOP/s {:5.2}x", tile.bm, tile.bn, flops / s / 1e12, bt / s);
            }
            println!();
        }
    }
}
