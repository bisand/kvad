//! Gradients through this crate: where they used to stop, and the check
//! that says whether they still do (#74).
//!
//! candle differentiates a computation by remembering, for each tensor, the
//! operation that made it and what it was made from; `backward` walks that
//! record from the loss to the parameters. This crate's fast paths are
//! Metal kernels of its own, entered with `apply_op*_no_bwd`, and a
//! `_no_bwd` operation records nothing: to `backward`, its answer is a
//! constant that came from nowhere.
//!
//! That is not an error. Take a LoRA on one layer, `y = W·h + B·A·h`, with
//! `W·h` from such a kernel. `A` and `B` still get a gradient, through the
//! side path. What is lost is every term that should have gone back
//! *through* `W·h` to `h`, and from `h` to every LoRA below this layer. So
//! the gradient is incomplete but not zero, the loss still falls, and
//! nothing says anything. A gradient that is subtly wrong still trains.
//!
//! # The rule
//!
//! Every such kernel now asks first whether any tensor it was handed is
//! being differentiated ([`tracked`]), and then does one of three things:
//!
//! - **It declines**, where candle's own operations can do the same
//!   arithmetic, which is nearly everywhere: each kernel already had a test
//!   for what it takes and a chain of candle operations for what it does
//!   not. Those have a backward. This is slower, by what the kernel won.
//! - **It carries its own backward**, for a quantised matrix
//!   ([`Frozen`]): candle's product with one has no backward, and there is
//!   nothing to fall back to.
//! - **It refuses** ([`refuse`]), where there is neither yet: the Q8_0
//!   product on the M5's matrix units, and the 3D convolution there.
//!   `backward is not supported` at the forward pass is the better failure.
//!
//! Nothing changes for inference: no tensor there is tracked.
//!
//! # The check
//!
//! A rule is only as good as its coverage, and a kernel added later can
//! forget it. [`directional`] does not trust the rule: it compares what
//! `backward` says with what the function does. Move the input a little
//! way along a random direction `d`, and the loss changes by `⟨∇f, d⟩`
//! times how far; `backward` gives the left side and two forward passes
//! the right. A path that cut the gradient shows as a `backward` that is
//! too small. It is `nervus::gradcheck` for a function too large to
//! perturb one number at a time, and [`complete`] is the guard to run
//! before a first training step.

use candle_core::quantized::QTensor;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Shape, Tensor, Var};
use candle_nn::{ops, rotary_emb};
use std::sync::Arc;

/// Whether `backward` will want to go through any of `ts`: it is a
/// variable, or was made from one.
pub(crate) fn tracked(ts: &[&Tensor]) -> bool {
    ts.iter().any(|t| t.track_op())
}

/// The error of a kernel that has no backward and nothing to fall back to,
/// asked to take a tensor that is being differentiated.
pub(crate) fn refuse(op: &'static str) -> candle_core::Error {
    candle_core::Error::BackwardNotSupported { op }.bt()
}

/// `x · Wᵀ` for a quantised `W` that is not being trained, with a backward.
///
/// The forward is candle's own. Going back, a frozen weight needs no
/// gradient of its own, only to pass the one it is given on to its input:
/// `∂L/∂x = ∂L/∂y · W`. `W` is dequantised for that, each time; what a
/// step of training costs through one of these is for #74's measurements.
pub(crate) struct Frozen(pub(crate) Arc<QTensor>);

impl CustomOp1 for Frozen {
    fn name(&self) -> &'static str {
        "qmatmul-frozen"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        CustomOp1::cpu_fwd(self.0.as_ref(), s, l)
    }

    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        CustomOp1::metal_fwd(self.0.as_ref(), s, l)
    }

    fn bwd(&self, arg: &Tensor, _: &Tensor, grad: &Tensor) -> candle_core::Result<Option<Tensor>> {
        // [out, in], f32.
        let w = self.0.dequantize(arg.device())?;
        Ok(Some(grad.to_dtype(DType::F32)?.broadcast_matmul(&w)?.to_dtype(arg.dtype())?))
    }
}

// ---------------------------------------------------------------------------
// candle's own
// ---------------------------------------------------------------------------
//
// candle-nn's fast normalisations, softmax, rotations and attention are
// `_no_bwd` kernels as well, and as silent: `backward` through
// `ops::rms_norm` finds a slope of exactly 0. For each, candle keeps a
// slower twin made of operations that have a backward. These pick the twin
// for a tensor that is being differentiated and the kernel otherwise, and
// the crate calls these and never the kernels. (`ops::sdpa` has no twin;
// its callers each have a written-out attention, and choose it.)
//
// The twin also takes f64, which the kernels were not written for: a
// gradient is checked in it (`directional`), and nothing else runs in it.

/// Whether candle's kernel will not do for `ts`: one is being
/// differentiated, or is f64.
fn slow(ts: &[&Tensor]) -> bool {
    tracked(ts) || ts[0].dtype() == DType::F64
}

/// `ops::rms_norm`, with a backward where one is wanted.
pub(crate) fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    match slow(&[x, w]) {
        true => ops::rms_norm_slow(x, w, eps),
        false => ops::rms_norm(x, w, eps),
    }
}

/// `ops::layer_norm`, with a backward where one is wanted.
pub(crate) fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    match slow(&[x, w, b]) {
        true => ops::layer_norm_slow(x, w, b, eps),
        false => ops::layer_norm(x, w, b, eps),
    }
}

/// `ops::softmax_last_dim`, with a backward where one is wanted.
pub(crate) fn softmax_last_dim(x: &Tensor) -> candle_core::Result<Tensor> {
    match slow(&[x]) {
        true => ops::softmax(x, candle_core::D::Minus1),
        false => ops::softmax_last_dim(x),
    }
}

/// `rotary_emb::rope`, with a backward where one is wanted.
pub(crate) fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    match slow(&[x, cos, sin]) {
        true => rotary_emb::rope_slow(x, cos, sin),
        false => rotary_emb::rope(x, cos, sin),
    }
}

/// `rotary_emb::rope_i`, with a backward where one is wanted.
pub(crate) fn rope_i(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    match slow(&[x, cos, sin]) {
        true => rotary_emb::rope_i_slow(x, cos, sin),
        false => rotary_emb::rope_i(x, cos, sin),
    }
}

/// The slope of a function along one direction, two ways, and how large a
/// slope its gradient gives a direction drawn like this one.
#[derive(Clone, Copy, Debug)]
pub struct Slopes {
    /// `⟨∇f, d⟩` with `backward`'s gradient; 0 if it reached none.
    pub by_backward: f64,
    /// `(f(x + εd) − f(x − εd)) / 2ε`.
    pub measured: f64,
    /// The length of `backward`'s gradient. Along a direction of standard
    /// normals the slope is normal too, with this as its standard deviation:
    /// the size a slope here typically is.
    pub typical: f64,
}

impl Slopes {
    /// How far apart the two slopes are, as a share of the typical slope or
    /// of the measured one, whichever is larger.
    ///
    /// Not as a share of the slope alone: along one random direction it can
    /// fall near zero by chance, and what is left of it is then the
    /// measurement's rounding. FLUX's first blocks gave slopes of 350 and of
    /// 0.59 through the same LoRA factor from two draws of the inputs, and
    /// the second read as 19% wrong.
    pub fn apart(&self) -> f64 {
        (self.by_backward - self.measured).abs() / self.typical.max(self.measured.abs()).max(1e-30)
    }
}

/// The slope of `f` at `x` along a random direction, two ways.
///
/// `f` takes `x` to one number. The direction `d` is drawn from `seed`,
/// with one standard normal for each element of `x`. Backward's slope is
/// `⟨∇f(x), d⟩`; the measured one is `(f(x + εd) − f(x − εd)) / 2ε`. Where
/// `backward` reaches no gradient for `x` at all, its slope is 0.
///
/// `ε` is a trade. Too large, and the function curves between the two
/// points; too small, and their difference is lost in the rounding of two
/// sums of the function's own size. In f64 there is room between the two
/// for nine digits; in f32, for three or four, with `ε` near 1e-2 of `x`'s
/// scale; in half precision the measured slope is mostly rounding, and
/// this is not the tool.
pub fn directional(f: &dyn Fn(&Tensor) -> candle_core::Result<Tensor>, x: &Tensor, eps: f64, seed: u64) -> candle_core::Result<Slopes> {
    // In the tensors' own precision where that is f64.
    let number = |t: Tensor| -> candle_core::Result<f64> {
        match t.dtype() {
            DType::F64 => t.sum_all()?.to_scalar::<f64>(),
            _ => Ok(t.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()? as f64),
        }
    };
    // On the host and seeded, so that a failure can be run again.
    let mut state = seed | 1;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    };
    let d: Vec<f32> = (0..x.elem_count()).map(|_| (-2.0 * uniform().ln()).sqrt() * (std::f32::consts::TAU * uniform()).cos()).collect();
    let d = Tensor::from_vec(d, x.shape(), x.device())?.to_dtype(x.dtype())?;

    let var = Var::from_tensor(x)?;
    let grads = f(var.as_tensor())?.backward()?;
    let (by_backward, typical) = match grads.get(var.as_tensor()) {
        Some(g) => (number((g * &d)?)?, number(g.sqr()?)?.sqrt()),
        None => (0.0, 0.0),
    };
    let at = |sign: f64| -> candle_core::Result<f64> { number(f(&(x + (&d * (sign * eps))?)?)?) };
    Ok(Slopes { by_backward, measured: (at(1.0)? - at(-1.0)?) / (2.0 * eps), typical })
}

/// Refuse a function whose gradient `backward` does not find whole: the
/// two slopes of [`directional`] further [`Slopes::apart`] than
/// `tolerance`. The guard to run on a model before its first step, in f32
/// or better.
pub fn complete(f: &dyn Fn(&Tensor) -> candle_core::Result<Tensor>, x: &Tensor, eps: f64, tolerance: f64) -> candle_core::Result<()> {
    let s = directional(f, x, eps, 0x5eed)?;
    if s.apart() > tolerance {
        candle_core::bail!("the gradient is incomplete: backward finds a slope of {:.6}, and the function's is {:.6}", s.by_backward, s.measured);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Proj;
    use candle_core::quantized::{GgmlDType, QMatMul};
    use candle_core::Device;

    fn metal() -> Option<Device> {
        Device::new_metal(0).ok()
    }

    fn randn(shape: &[usize], dev: &Device) -> Tensor {
        Tensor::randn(0f32, 1.0, shape, dev).unwrap()
    }

    /// The largest difference between `got` and `want`, as a share of
    /// `want`'s largest value.
    fn off(got: &Tensor, want: &Tensor) -> f32 {
        let f = |t: &Tensor| t.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
        let (got, want) = (f(got), f(want));
        let max = |t: &Tensor| t.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        max(&(got - &want).unwrap()) / max(&want)
    }

    /// The check sees a cut: with half of `f`'s dependence on `x` hidden
    /// from `backward`, as a `_no_bwd` kernel hides it, backward's slope is
    /// half the function's, and `complete` refuses it. Whole, it passes.
    #[test]
    fn the_check_sees_a_gradient_that_stops() {
        let x = randn(&[6, 5], &Device::Cpu);
        let whole = |x: &Tensor| x.sqr()?.sum_all();
        let cut = |x: &Tensor| (x * x.detach())?.sum_all();
        let Slopes { by_backward: b, measured: m, .. } = directional(&whole, &x, 1e-2, 1).unwrap();
        assert!((b - m).abs() < 1e-3 * m.abs(), "whole: {b} against {m}");
        let Slopes { by_backward: b, measured: m, .. } = directional(&cut, &x, 1e-2, 1).unwrap();
        assert!((2.0 * b - m).abs() < 1e-3 * m.abs(), "cut: {b} should be half of {m}");
        assert!(complete(&whole, &x, 1e-2, 1e-2).is_ok());
        let e = complete(&cut, &x, 1e-2, 1e-2).unwrap_err().to_string();
        assert!(e.contains("incomplete"), "{e}");
        // And one that reaches nothing at all.
        let none = |x: &Tensor| x.detach().sqr()?.sum_all();
        assert!(complete(&none, &x, 1e-2, 1e-2).is_err());
    }

    /// A dense projection in half precision on Metal, which the M5's matrix
    /// units take when nothing is tracked: the gradient reaches its input,
    /// and is `∂L/∂y · Wᵀ`.
    #[test]
    fn a_dense_projection_passes_the_gradient_to_its_input() {
        let Some(dev) = metal() else { return };
        for dt in [DType::F16, DType::BF16] {
            let (x, w, r) = (randn(&[512, 64], &dev), (randn(&[64, 96], &dev) * 0.1).unwrap(), randn(&[512, 96], &dev));
            let proj = Proj::Dense(w.to_dtype(dt).unwrap());
            let var = Var::from_tensor(&x.to_dtype(dt).unwrap()).unwrap();
            let y = proj.forward(var.as_tensor()).unwrap();
            let grads = (y.to_dtype(DType::F32).unwrap() * &r).unwrap().sum_all().unwrap().backward().unwrap();
            let got = grads.get(var.as_tensor()).expect("no gradient reached the projection's input");
            let want = r.matmul(&w.t().unwrap()).unwrap();
            assert!(off(got, &want) < 2e-2, "{dt:?}: {}", off(got, &want));
        }
    }

    /// A quantised projection, on the CPU and on Metal: candle's product has
    /// no backward, and [`Frozen`] gives it one. Checked against
    /// `∂L/∂y · W` with the weights as they dequantise, and by the slopes.
    ///
    /// The slopes are measured a whole unit apart. On the CPU candle's
    /// product rounds its *input* to eight bits a block as well, so the
    /// function is a staircase, and a step of 1e-2 measures the stairs: the
    /// slopes were 8% apart there and 6e-4 apart here. The projection is
    /// linear, so any step measures the same slope. Backward's is the slope
    /// of the product without that rounding, which is the one to train on.
    #[test]
    fn a_quantised_projection_passes_the_gradient_to_its_input() {
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            // Seeded, where the other tests here draw afresh each run: the
            // staircase leaves the two slopes about 0.4% of a typical slope
            // apart whatever the step, and they are judged as a share of
            // the slope itself, which along one random direction now and
            // then comes out small. A draw of 4.47 against 4.56 failed this.
            let seeded = |seed: u64, shape: &[usize], dev: &Device| crate::image::nn::noise(seed, shape, dev, DType::F32).unwrap();
            let w = (seeded(1, &[96, 64], &Device::Cpu) * 0.1).unwrap();
            // Quantised on the device it will run on.
            let q = Arc::new(QTensor::quantize(&w.to_device(&dev).unwrap(), GgmlDType::Q8_0).unwrap());
            let rounded = q.dequantize(&dev).unwrap();
            let proj = Proj::Quant(QMatMul::from_arc(q).unwrap());
            let (x, r) = (seeded(2, &[7, 64], &dev), seeded(3, &[7, 96], &dev));
            let f = |x: &Tensor| (proj.forward(x)? * &r)?.sum_all();
            complete(&f, &x, 1.0, 1e-2).unwrap();

            let var = Var::from_tensor(&x).unwrap();
            let grads = f(var.as_tensor()).unwrap().backward().unwrap();
            let got = grads.get(var.as_tensor()).expect("no gradient reached the projection's input");
            assert!(off(got, &r.matmul(&rounded).unwrap()) < 1e-4);
            // A row that does not start its buffer takes the path that lays
            // it out afresh first, and the gradient still finds it.
            let two = Var::from_tensor(&randn(&[9, 64], &dev)).unwrap();
            let grads = (proj.forward(&two.as_tensor().narrow(0, 2, 7).unwrap()).unwrap() * &r).unwrap().sum_all().unwrap().backward().unwrap();
            let got = grads.get(two.as_tensor()).expect("no gradient through the narrowed rows");
            assert!(off(&got.narrow(0, 2, 7).unwrap(), &r.matmul(&rounded).unwrap()) < 1e-4);
        }
    }

    /// What has no backward and no fallback says so at the forward pass:
    /// the Q8_0 product on the matrix units, and the 3D convolution there.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_kernel_with_no_backward_refuses_a_tracked_tensor() {
        let Some(dev) = metal() else { return };
        if !crate::mpp::available(&dev) {
            return;
        }
        let refused = |r: candle_core::Result<Tensor>| r.err().is_some_and(|e| e.to_string().contains("backward is not supported"));

        let w = randn(&[96, 64], &Device::Cpu);
        let q = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        let blocks = crate::mpp::Blocks::new(GgmlDType::Q8_0, &q.data().unwrap(), 96, 64, &dev).unwrap();
        let x = randn(&[512, 64], &dev);
        assert!(blocks.forward(&x).is_ok());
        assert!(refused(blocks.forward(Var::from_tensor(&x).unwrap().as_tensor())));

        let x = randn(&[3, 32, 5, 7], &dev).to_dtype(DType::BF16).unwrap();
        let k = crate::mpp_conv3d::taps(&randn(&[32, 32, 3, 3, 3], &dev).to_dtype(DType::BF16).unwrap()).unwrap();
        let b = Tensor::zeros(32, DType::BF16, &dev).unwrap();
        let time = crate::video::conv3d::Time::Replicate;
        assert!(crate::mpp_conv3d::conv3d(&x, &k, &b, (0, 3), (0, 3), time).is_ok());
        assert!(refused(crate::mpp_conv3d::conv3d(Var::from_tensor(&x).unwrap().as_tensor(), &k, &b, (0, 3), (0, 3), time)));
    }

    /// The kernels that fall back decline a tracked tensor, and take the
    /// same tensor untracked: attention on the matrix units, and the two
    /// fused kernels that answer `None`.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_kernel_with_a_fallback_declines_a_tracked_tensor() {
        let Some(dev) = metal() else { return };
        if !crate::mpp::available(&dev) {
            return;
        }
        let half = |shape: &[usize]| randn(shape, &dev).to_dtype(DType::BF16).unwrap();
        let var = |t: &Tensor| Var::from_tensor(t).unwrap().as_tensor().clone();

        // Rows by heads · d: two heads of 64.
        let (q, k, v) = (half(&[64, 128]), half(&[64, 128]), half(&[64, 128]));
        assert!(crate::mpp_attention::attention(&q, &k, &v, 2, None).unwrap().is_some());
        for i in 0..3 {
            let t = [var(&q), var(&k), var(&v)];
            let pick = |j: usize, plain: &Tensor| if i == j { t[j].clone() } else { plain.clone() };
            assert!(crate::mpp_attention::attention(&pick(0, &q), &pick(1, &k), &pick(2, &v), 2, None).unwrap().is_none(), "input {i}");
        }

        let (x, w) = (half(&[512, 64]), half(&[64, 96]));
        assert!(crate::mpp::dense(&x, &w).unwrap().is_some());
        assert!(crate::mpp::dense(&var(&x), &w).unwrap().is_none());
        assert!(crate::mpp::dense(&x, &var(&w)).unwrap().is_none());

        let (g, u) = (half(&[8, 64]), half(&[8, 64]));
        assert!(crate::video::ltx_fused::swiglu(&g, &u).unwrap().is_some());
        assert!(crate::video::ltx_fused::swiglu(&var(&g), &u).unwrap().is_none());
        assert!(crate::video::ltx_fused::swiglu(&g, &var(&u)).unwrap().is_none());
    }

    /// The fused kernels that answer either way, in f32 on Metal: the
    /// gradient through each is whole, by the slopes.
    #[test]
    fn the_fused_steps_have_a_whole_gradient() {
        let Some(dev) = metal() else { return };
        let r = randn(&[6, 32], &dev);
        let (d, w) = (randn(&[6, 32], &dev), randn(&[32], &dev));
        let norm = |x: &Tensor| {
            let (n, sum) = crate::fused::add_rms_norm(x, &d, &w, 1e-6)?;
            ((n * &r)? + sum.sqr()?)?.sum_all()
        };
        complete(&norm, &randn(&[6, 32], &dev), 1e-2, 1e-2).unwrap();

        let r16 = randn(&[6, 16], &dev);
        let silu = |x: &Tensor| (crate::fused::silu_mul(x, 16)? * &r16)?.sum_all();
        complete(&silu, &randn(&[6, 32], &dev), 1e-2, 1e-2).unwrap();

        let held = crate::video::ltx_fused::Held::ALIKE;
        let (scale, shift) = ((randn(&[1, 32], &dev) * 0.1).unwrap(), randn(&[1, 32], &dev));
        let modulate = |x: &Tensor| (crate::video::ltx_fused::modulate(x, &scale, &shift, &held, 1e-6)? * &r)?.sum_all();
        complete(&modulate, &randn(&[6, 32], &dev), 1e-2, 1e-2).unwrap();

        let gated = |x: &Tensor| (crate::video::ltx_fused::gated_add(x, &d, &shift, &held)? * &r)?.sum_all();
        complete(&gated, &randn(&[6, 32], &dev), 1e-2, 1e-2).unwrap();
        // And through the kernel's other two inputs.
        let x = randn(&[6, 32], &dev);
        let gated = |y: &Tensor| (crate::video::ltx_fused::gated_add(&x, y, &shift, &held)? * &r)?.sum_all();
        complete(&gated, &d, 1e-2, 1e-2).unwrap();

        let gelu = |x: &Tensor| (crate::video::ltx_fused::gelu(x, DType::F32)? * &r)?.sum_all();
        complete(&gelu, &randn(&[6, 32], &dev), 1e-2, 1e-2).unwrap();
    }

    /// candle's own kernels, through the wrappers, on the CPU and on Metal:
    /// a tracked tensor gets the same numbers as the kernel gives an
    /// untracked one, and a gradient that is whole. Through the kernels
    /// themselves, backward's slope is 0.
    #[test]
    fn candles_kernels_have_a_whole_gradient_through_the_wrappers() {
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            let x = randn(&[2, 4, 6, 32], &dev);
            let r = randn(&[2, 4, 6, 32], &dev);
            let (w, b) = (randn(&[32], &dev), randn(&[32], &dev));
            // [6, 16] tables of angles, one row a position.
            let angle = randn(&[6, 16], &dev);
            let (cos, sin) = (angle.cos().unwrap(), angle.sin().unwrap());
            type F<'a> = &'a dyn Fn(&Tensor) -> candle_core::Result<Tensor>;
            let rms = |x: &Tensor| rms_norm(x, &w, 1e-6);
            let layer = |x: &Tensor| layer_norm(x, &w, &b, 1e-6);
            let soft = |x: &Tensor| softmax_last_dim(x);
            let turn = |x: &Tensor| rope(x, &cos, &sin);
            let turn_i = |x: &Tensor| rope_i(x, &cos, &sin);
            for (name, f) in [("rms_norm", &rms as F), ("layer_norm", &layer), ("softmax", &soft), ("rope", &turn), ("rope_i", &turn_i)] {
                let fast = f(&x).unwrap();
                let slow = f(Var::from_tensor(&x).unwrap().as_tensor()).unwrap();
                assert!(off(&slow, &fast) < 1e-4, "{name}: tracked, the numbers differ by {}", off(&slow, &fast));
                let loss = |x: &Tensor| (f(x)? * &r)?.sum_all();
                let s = directional(&loss, &x, 1e-2, 5).unwrap();
                assert!(s.apart() < 1e-2, "{name} on {:?}: {s:?}", dev.location());
            }
            // What the wrappers are for: the kernel itself, tracked.
            let bare = |x: &Tensor| (ops::rms_norm(x, &w, 1e-6)? * &r)?.sum_all();
            let s = directional(&bare, &x, 1e-2, 5).unwrap();
            assert!(s.by_backward == 0.0 && s.measured.abs() > 1e-3, "candle's rms_norm now has a backward: {s:?}");
        }
    }

    /// Attention as the image models call it, tracked, on Metal in f32 and
    /// on the CPU: whole through the queries, the keys and the values.
    #[test]
    fn attention_has_a_whole_gradient() {
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            let (q, k, v) = (randn(&[1, 10, 128], &dev), randn(&[1, 12, 128], &dev), randn(&[1, 12, 128], &dev));
            let r = randn(&[1, 10, 128], &dev);
            let by_q = |q: &Tensor| (crate::image::nn::attention(q, &k, &v, 2)? * &r)?.sum_all();
            let by_k = |k: &Tensor| (crate::image::nn::attention(&q, k, &v, 2)? * &r)?.sum_all();
            let by_v = |v: &Tensor| (crate::image::nn::attention(&q, &k, v, 2)? * &r)?.sum_all();
            complete(&by_q, &q, 1e-2, 2e-2).unwrap();
            complete(&by_k, &k, 1e-2, 2e-2).unwrap();
            complete(&by_v, &v, 1e-2, 2e-2).unwrap();
        }
    }
}
