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
//!   product on the M5's matrix units, the 3D convolution there, and a
//!   write into a tensor that already exists ([`slice_set`]): the video
//!   decoders' chunks and the language models' cache.
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
// Writing in place
// ---------------------------------------------------------------------------

/// `Tensor::slice_set`, refused for a tensor that is being differentiated.
///
/// Writing into a tensor that already exists is how the work that is too
/// large to do at once is put together here: the video decoders fill a
/// clip a few frames at a time, and the language models' cache takes each
/// new token's keys and values. candle says of `slice_set` that it "is not
/// compatible with back-propagation", and does not check: the numbers are
/// written and nothing is recorded, so what `backward` then finds for the
/// tensor written is nothing, with no error.
///
/// Every such write in this crate comes through here. None of them has a
/// place in training as it stands (a clip is decoded, and a cache kept, to
/// draw and to write, not to learn from), so this is the rule's third
/// answer: it says so.
pub(crate) fn slice_set(into: &Tensor, from: &Tensor, dim: usize, offset: usize) -> candle_core::Result<()> {
    if tracked(&[into, from]) {
        return Err(refuse("slice_set"));
    }
    into.slice_set(from, dim, offset)
}

// ---------------------------------------------------------------------------
// Frozen layers
// ---------------------------------------------------------------------------

/// What takes a result's gradient back to its input's: `(x, ∂L/∂y)` to
/// `∂L/∂x`.
type Back = Box<dyn Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor> + Send + Sync>;

/// `y`, which was computed from `x` out of `backward`'s sight, put back in
/// it: as far as `backward` knows, `y` came from `x` by one operation, and
/// `back` is how its gradient returns.
///
/// For a layer whose weights are not being trained. candle's `backward`
/// works out a gradient for every input of every operation, and a layer's
/// weight is one: through `x.matmul(w)` it computes `xᵀ·∂L/∂y`, as large as
/// `w`, whether or not anything will ever read it, and keeps it until it
/// is done. On SDXL's UNet that was a second copy of all 2.6 B weights
/// made and thrown away at each step: 5.2 s a step and 31.8 GB at 256×256,
/// before any of it went on the LoRA. A frozen layer owes `backward` one
/// thing, its input's gradient, and `back` computes only that.
///
/// It also gives the forward pass back its kernels. `y` is made from a
/// detached `x`, so nothing in it is tracked, and the M5's matrix units
/// take it as they do when drawing.
pub(crate) fn attach(x: &Tensor, y: Tensor, back: impl Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor> + Send + Sync + 'static) -> candle_core::Result<Tensor> {
    // The storage is handed over as it is, to be read from its start.
    let y = match y.is_contiguous() && y.layout().start_offset() == 0 {
        true => y,
        false => y.force_contiguous()?,
    };
    x.apply_op2(&y, Attached { back: Box::new(back) })
}

/// The operation [`attach`] records: of `x` and the finished `y`, whose
/// storage is its answer.
struct Attached {
    back: Back,
}

impl candle_core::CustomOp2 for Attached {
    fn name(&self) -> &'static str {
        "attached"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, y: &CpuStorage, l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        Ok((y.clone(), l.shape().clone()))
    }

    fn metal_fwd(&self, _: &MetalStorage, _: &Layout, y: &MetalStorage, l: &Layout) -> candle_core::Result<(MetalStorage, Shape)> {
        // The same buffer, not a copy of it.
        Ok((y.clone(), l.shape().clone()))
    }

    fn bwd(&self, x: &Tensor, _: &Tensor, _: &Tensor, grad: &Tensor) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some((self.back)(x, grad)?), None))
    }
}

/// `∂L/∂x` for `y = x · w` over the last axis, `w` `[in, out]`: `∂L/∂y · wᵀ`,
/// in `x`'s shape and dtype.
///
/// Computed as `(w · ∂L/∂yᵀ)ᵀ`. The product wants its operands laid out in
/// rows, and a transposed one is copied to be: this way round it is the
/// gradient that is copied, twice, and not the weight, which is the larger
/// by far wherever there are fewer rows than the layer is wide.
pub(crate) fn back_through(w: &Tensor, x: &Tensor, grad: &Tensor) -> candle_core::Result<Tensor> {
    let out = w.dim(1)?;
    let g = grad.reshape((grad.elem_count() / out, out))?.to_dtype(w.dtype())?;
    w.matmul(&g.t()?.contiguous()?)?.t()?.contiguous()?.reshape(x.dims())?.to_dtype(x.dtype())
}

/// `[B, C, H, W]` to `[B, C, 2H, 2W]`, each value written four times, with
/// a backward that is the sum of each four.
///
/// candle's own backward for a nearest-neighbour upsample is a convolution
/// with a group for every channel, which it runs as one convolution a
/// channel, 640 and 1280 of them for SDXL's two. This is the same sum in
/// two reductions. (It was written on the guess that those convolutions
/// were what a training step waited on. They were not: the step's time
/// and memory did not move.)
pub(crate) fn upsample_twice(x: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    if !x.track_op() {
        return x.upsample_nearest2d(2 * h, 2 * w);
    }
    attach(x, x.detach().upsample_nearest2d(2 * h, 2 * w)?, |x, g| {
        let (b, c, h, w) = x.dims4()?;
        // [b, c, h, 2, w, 2]: summed over the two axes of twos, one at a
        // time and four axes at most, as Metal's reductions want.
        let rows = g.reshape((b * c, h, 2, 2 * w))?.sum(2)?;
        rows.reshape((b * c, h, w, 2))?.sum(3)?.reshape((b, c, h, w))?.to_dtype(x.dtype())
    })
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// The most scores [`attention_back`] holds in one tensor: 32 M, 128 MB in
/// f32. It holds about six that size at once.
const SCORES: usize = 1 << 25;

/// Unmasked attention, `softmax(scale · q·kᵀ) · v` over `[B, heads, L, d]`,
/// by `inner`, with a backward of its own where one is wanted.
///
/// Recorded step by step, attention is the most a model keeps for
/// `backward`. Its scores are a number for every pair of tokens and every
/// head, 671 MB in f32 for SDXL at 4096 tokens, and the record holds them
/// at every step from the product to the softmax's answer, and `backward`
/// then makes a gradient as large for each: 17 GB for one transformer
/// block, the peak of a whole training step at 1024×1024.
///
/// Nothing of that needs keeping. The scores can be made again from `q`
/// and `k`, which are small. So the forward pass is `inner`'s, out of
/// `backward`'s sight on whatever kernel draws, and the way back
/// ([`attention_back`]) makes the scores again a few rows of queries at a
/// time and lets each batch go.
///
/// `backward` is told of one input, and there are three, so the three are
/// laid end to end as one tensor for it, and their gradients come back the
/// same way and are taken apart by the record of that laying.
pub(crate) fn attended(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    inner: impl Fn(&Tensor, &Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    if !tracked(&[q, k, v]) || k.dtype() != q.dtype() || v.dtype() != q.dtype() {
        return inner(q, k, v);
    }
    let (qd, kd, vd) = (q.detach(), k.detach(), v.detach());
    let y = inner(&qd, &kd, &vd)?;
    let all = Tensor::cat(&[q.flatten_all()?, k.flatten_all()?, v.flatten_all()?], 0)?;
    attach(&all, y, move |_, g| attention_back(&qd, &kd, &vd, scale, g))
}

/// The gradients of [`attended`]'s `q`, `k` and `v`, each flattened and the
/// three end to end, from its answer's `g`.
///
/// With `P = softmax(S)` and `S = scale · q·kᵀ`, the answer is `P·v`, so
///
/// ```text
/// ∂v = Pᵀ·g                       each value, by how much each query took of it
/// ∂P = g·vᵀ
/// ∂S = P ⊙ (∂P − Σⱼ ∂P ⊙ P)       the softmax, a row at a time
/// ∂q = scale · ∂S·k
/// ∂k = scale · ∂Sᵀ·q
/// ```
///
/// A row of `S` belongs to one query, and nothing in these sums crosses
/// rows but `∂k` and `∂v`, which add up over them. So the queries are
/// taken a batch at a time: the batch's scores made, used and dropped, and
/// `∂k` and `∂v` added to. In f32 whatever the model's dtype, as the
/// forward pass takes its softmax: these are sums of small numbers.
fn attention_back(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, g: &Tensor) -> candle_core::Result<Tensor> {
    attention_back_in(q, k, v, scale, g, SCORES)
}

/// [`attention_back`] with `scores` of them at most in one tensor.
fn attention_back_in(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, g: &Tensor, scores: usize) -> candle_core::Result<Tensor> {
    use candle_core::D;
    let (b, h, lq, _) = q.dims4()?;
    let lk = k.dim(2)?;
    let wide = crate::image::nn::wide(q.dtype());
    let (kw, vw) = (k.to_dtype(wide)?, v.to_dtype(wide)?);
    let (kt, vt) = (kw.transpose(2, 3)?.contiguous()?, vw.transpose(2, 3)?.contiguous()?);
    let rows = (scores / (b * h * lk)).clamp(1, lq);
    let mut dq = Vec::with_capacity(lq.div_ceil(rows));
    let (mut dk, mut dv) = (kw.zeros_like()?, vw.zeros_like()?);
    let mut start = 0;
    while start < lq {
        let n = rows.min(lq - start);
        let qs = q.narrow(2, start, n)?.to_dtype(wide)?.contiguous()?;
        let gs = g.narrow(2, start, n)?.to_dtype(wide)?.contiguous()?;
        let p = softmax_last_dim(&(qs.matmul(&kt)? * scale)?)?;
        let dp = gs.matmul(&vt)?;
        let ds = ((&p * dp.broadcast_sub(&(&dp * &p)?.sum_keepdim(D::Minus1)?)?)? * scale)?;
        dq.push(ds.matmul(&kw)?);
        // `∂Sᵀ·q` as `(qᵀ·∂S)ᵀ`, and `Pᵀ·g` likewise: the product wants
        // its operands laid out in rows, and this way round it is `q` and
        // `g` that are copied to be, not the scores.
        let across = |t: &Tensor| t.transpose(2, 3)?.contiguous();
        dk = (dk + across(&across(&qs)?.matmul(&ds)?)?)?;
        dv = (dv + across(&across(&gs)?.matmul(&p)?)?)?;
        start += n;
    }
    let dq = Tensor::cat(&dq, 2)?;
    Tensor::cat(&[dq.flatten_all()?, dk.flatten_all()?, dv.flatten_all()?], 0)?.to_dtype(q.dtype())
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

/// One stretch of a model: from the state the stretch before it left to the
/// state the next one reads, each a list of tensors.
pub(crate) type Stretch<'a> = Box<dyn Fn(&[Tensor]) -> candle_core::Result<Vec<Tensor>> + 'a>;

/// The loss of a model that is a row of [`Stretch`]es, and its gradient for
/// each of `vars`, without ever holding more than one stretch's record.
///
/// `backward` needs every tensor the forward pass made, and a forward pass
/// that is being recorded keeps them all until it has. For a denoiser that
/// is every activation of every block at once, which at a useful size is
/// more than the machine has. Checkpointing trades that memory for a second
/// forward pass:
///
/// 1. **Forward, unrecorded.** `recording(false)` takes the trained tensors
///    out of sight, so nothing is tracked and each stretch runs as it does
///    when drawing, on the fast kernels, keeping nothing. Only the states
///    between stretches are kept: the checkpoints.
/// 2. **The loss**, from the last state, recorded: its gradient for that
///    state is where the walk back starts.
/// 3. **Back, a stretch at a time**, last first. The stretch is run again
///    from its checkpoint, recorded this time. What `backward` is given is
///    not the loss but `Σ⟨out, g⟩`, the stretch's outputs each weighted by
///    the gradient already known for it, held constant: by the chain rule
///    that sum's gradient is the loss's, for the stretch's trained tensors
///    and for its input, which is the `g` the stretch before it needs. Then
///    the record is dropped.
///
/// So the most held at once is the checkpoints and one stretch's record,
/// and the price is each stretch's forward pass twice.
///
/// `settle` is called after each stretch, for a device that frees what was
/// dropped only when asked.
pub(crate) fn checkpointed(
    stretches: &[Stretch<'_>],
    input: Vec<Tensor>,
    loss: &dyn Fn(&[Tensor]) -> candle_core::Result<Tensor>,
    vars: &[Var],
    recording: &dyn Fn(bool),
    settle: &dyn Fn() -> candle_core::Result<()>,
) -> candle_core::Result<(Tensor, candle_core::backprop::GradStore)> {
    recording(false);
    let mut states = vec![input];
    for stretch in stretches {
        let next = stretch(states.last().expect("the input"))?;
        if next.iter().any(|t| t.track_op()) {
            recording(true);
            candle_core::bail!("checkpointing: a stretch's answer is being recorded with recording off; something trained is not behind the switch");
        }
        states.push(next);
        settle()?;
    }
    recording(true);

    // A state as leaves `backward` reports on, and the gradient found for
    // each, zero for one nothing read. Detached: a gradient candle hands
    // back still carries the record of how it was made, and that record
    // runs through the trained tensors of the stretch it came from. Used
    // as it is to weigh the stretch before, `backward` walked on through
    // it and counted those tensors' gradients a second time, exactly
    // doubling them.
    let leaves = |state: &[Tensor]| state.iter().map(Var::from_tensor).collect::<candle_core::Result<Vec<_>>>();
    let found = |leaves: &[Var], store: &candle_core::backprop::GradStore| -> candle_core::Result<Vec<Tensor>> {
        leaves.iter().map(|v| store.get(v.as_tensor()).map_or_else(|| v.zeros_like(), |g| Ok(g.detach()))).collect()
    };
    let tensors = |leaves: &[Var]| leaves.iter().map(|v| v.as_tensor().clone()).collect::<Vec<_>>();

    let last = leaves(&states.pop().expect("the last state"))?;
    let value = loss(&tensors(&last))?;
    let mut total = value.backward()?;
    let g = found(&last, &total)?;
    // The loss as a number, and nothing of how it was made.
    let value = value.detach();
    for v in &last {
        total.remove(v.as_tensor());
    }
    drop(last);
    settle()?;
    let mut g = rehomed(g)?;

    for stretch in stretches.iter().rev() {
        let from = leaves(&states.pop().expect("a state for each stretch"))?;
        let out = stretch(&tensors(&from))?;
        if out.len() != g.len() {
            candle_core::bail!("checkpointing: a stretch left {} tensors this time and {} the first", out.len(), g.len());
        }
        // A tensor the stretch handed on untouched, a UNet's waiting skip,
        // takes its gradient as it is. The rest are weighed, in f32: a sum
        // over a whole feature map of half-precision products is past what
        // half precision holds.
        let mut passed: Vec<Option<Tensor>> = vec![None; from.len()];
        let mut weighed = Tensor::zeros((), DType::F32, value.device())?;
        for (o, g) in out.iter().zip(&g) {
            match from.iter().position(|v| v.as_tensor().id() == o.id()) {
                Some(at) => {
                    passed[at] = Some(match passed[at].take() {
                        Some(p) => (p + g)?,
                        None => g.clone(),
                    })
                }
                None => weighed = (weighed + (o.to_dtype(DType::F32)? * g.to_dtype(DType::F32)?)?.sum_all()?)?,
            }
        }
        let store = weighed.backward()?;
        let (which, mut kept): (Vec<usize>, Vec<Tensor>) = vars.iter().enumerate().filter_map(|(i, v)| store.get(v.as_tensor()).map(|g| (i, g.detach()))).unzip();
        for (g, p) in found(&from, &store)?.into_iter().zip(passed) {
            kept.push(match p {
                Some(p) => (g + p)?,
                None => g,
            });
        }
        // The record goes, all of it: `weighed` is its root, and holds
        // every tensor the stretch made for as long as it lives.
        drop((store, weighed, out, from));
        settle()?;
        let mut kept = rehomed(kept)?;
        g = kept.split_off(which.len());
        for (i, more) in which.into_iter().zip(kept) {
            let v = vars[i].as_tensor();
            let sum = match total.get(v) {
                Some(have) => (have + more)?,
                None => more,
            };
            total.insert(v, sum);
        }
    }
    Ok((value, total))
}

/// `kept`, each copied into a buffer of its own size, and the originals
/// let go. To be called straight after `settle`.
///
/// What `backward` leaves is small and lives long: a LoRA factor's gradient
/// is some kilobytes, and is wanted until the optimiser has read it. But
/// candle's Metal allocator hands a new tensor the smallest *free* buffer
/// that will hold it, and during `backward` the free ones are the last
/// activations', megabytes each. So each kept gradient sat in one of those
/// and held it: SDXL's 1120 held 4 GB between them at 480×480, by the end
/// of a step, where they are 93 MB. Straight after a `settle` nothing is
/// free, so a copy made then is given a buffer of its own; and once every
/// copy is made the originals go, and the buffers they held are free for
/// the next stretch to use, and let go at its `settle`.
fn rehomed(kept: Vec<Tensor>) -> candle_core::Result<Vec<Tensor>> {
    // `affine(1, 0)` writes a new buffer; `copy` on Metal shares the old.
    kept.iter().map(|t| t.affine(1.0, 0.0)).collect()
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

/// `ops::rms_norm`, with a backward where one is wanted: its own
/// ([`norm_back`]) where only `x` is being differentiated, which is every
/// norm of a model whose weights are frozen, and candle's twin's otherwise.
pub(crate) fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    if x.track_op() && !w.track_op() {
        let w = w.clone();
        return attach(x, rms_norm(&x.detach(), &w, eps)?, move |x, g| norm_back(x, &g.broadcast_mul(&w.to_dtype(g.dtype())?)?, eps as f64, false)?.to_dtype(x.dtype()));
    }
    match slow(&[x, w]) {
        true => ops::rms_norm_slow(x, w, eps),
        false => ops::rms_norm(x, w, eps),
    }
}

/// `ops::layer_norm`, with a backward where one is wanted, as
/// [`rms_norm`] has.
pub(crate) fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    if x.track_op() && !tracked(&[w, b]) {
        let w = w.clone();
        return attach(x, layer_norm(&x.detach(), &w, b, eps)?, move |x, g| norm_back(x, &g.broadcast_mul(&w.to_dtype(g.dtype())?)?, eps as f64, true)?.to_dtype(x.dtype()));
    }
    match slow(&[x, w, b]) {
        true => ops::layer_norm_slow(x, w, b, eps),
        false => ops::layer_norm(x, w, b, eps),
    }
}

/// `∂L/∂x` for a norm over the last axis, `y = x̂ = (x − μ) / σ` with
/// `σ = √(mean((x − μ)²) + ε)`, from `g = ∂L/∂x̂`; `μ` is the mean if
/// `centred` (a layer norm, a group norm) and 0 if not (an RMS norm). A
/// norm's weight is the caller's to multiply into `g` first.
///
/// Moving one `x` moves its own `x̂`, and through `σ` (and `μ`) every other
/// one on the axis:
///
/// ```text
/// ∂L/∂x = ( g − mean(g) − x̂ · mean(g · x̂) ) / σ        `mean(g)` only if centred
/// ```
///
/// The middle term is what the centring takes back, and the last what the
/// scaling does: a norm's answer cannot move along `x̂` itself, so that
/// much of the gradient is removed.
///
/// A norm recorded step by step is ten small operations, and `backward`
/// makes several more for each; a transformer block has three and a resnet
/// two. This is one operation in the record, and `x̂` and `σ` are made
/// again here from `x`, in f32 as the norm makes them.
pub(crate) fn norm_back(x: &Tensor, g: &Tensor, eps: f64, centred: bool) -> candle_core::Result<Tensor> {
    use candle_core::D;
    let wide = crate::image::nn::wide(x.dtype());
    let (x, g) = (x.detach().to_dtype(wide)?, g.to_dtype(wide)?);
    let x = match centred {
        true => x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?,
        false => x,
    };
    let sigma = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    let hat = x.broadcast_div(&sigma)?;
    let along = hat.broadcast_mul(&(&g * &hat)?.mean_keepdim(D::Minus1)?)?;
    let back = match centred {
        true => g.broadcast_sub(&g.mean_keepdim(D::Minus1)?)? - along,
        false => g - along,
    }?;
    back.broadcast_div(&sigma)
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

    /// Checkpointing finds the gradients `backward` finds through the whole
    /// model at once: a row of three stretches over a state of two tensors,
    /// one of which a stretch passes on untouched and a later one reads,
    /// as a UNet's skips are; each stretch with a trained matrix that the
    /// switch takes out of sight.
    #[test]
    fn checkpointing_finds_the_whole_models_gradients() {
        let dev = Device::Cpu;
        let vars: Vec<Var> = (0..3).map(|_| Var::from_tensor(&(randn(&[8, 8], &dev) * 0.3).unwrap()).unwrap()).collect();
        let on = std::cell::Cell::new(true);
        // A trained matrix as a stretch reads it: itself, or out of sight.
        let seen = |i: usize| if on.get() { vars[i].as_tensor().clone() } else { vars[i].as_tensor().detach() };
        let stretches: Vec<Stretch<'_>> = vec![
            // [x] to [h, skip]
            Box::new(|s| {
                let h = s[0].matmul(&seen(0))?.tanh()?;
                Ok(vec![h.clone(), h])
            }),
            // h changes, the skip waits
            Box::new(|s| Ok(vec![s[0].matmul(&seen(1))?.tanh()?, s[1].clone()])),
            // the skip is taken back up
            Box::new(|s| Ok(vec![(s[0].matmul(&seen(2))? + &s[1])?.sqr()?])),
        ];
        let x = randn(&[5, 8], &dev);
        let target = randn(&[5, 8], &dev);
        let loss = |s: &[Tensor]| (&s[0] - &target)?.sqr()?.mean_all();

        let mut state = vec![x.clone()];
        for stretch in &stretches {
            state = stretch(&state).unwrap();
        }
        let whole_loss = loss(&state).unwrap();
        let whole = whole_loss.backward().unwrap();

        let (value, grads) = checkpointed(&stretches, vec![x], &loss, &vars, &|live| on.set(live), &|| Ok(())).unwrap();
        assert!(on.get(), "recording is left on");
        assert_eq!(value.to_scalar::<f32>().unwrap(), whole_loss.to_scalar::<f32>().unwrap());
        for (i, v) in vars.iter().enumerate() {
            let (got, want) = (grads.get(v.as_tensor()).expect("a gradient for each"), whole.get(v.as_tensor()).unwrap());
            assert!(off(got, want) < 1e-5, "stretch {i}'s matrix: {} apart", off(got, want));
        }
        // A trained tensor that is not behind the switch is refused, not
        // quietly recorded through the first pass.
        let exposed: Vec<Stretch<'_>> = vec![Box::new(|s| Ok(vec![s[0].matmul(vars[0].as_tensor())?]))];
        let e = checkpointed(&exposed, vec![randn(&[5, 8], &dev)], &loss, &vars, &|_| {}, &|| Ok(())).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("not behind the switch"), "{e}");
    }

    /// The upsample's own backward is candle's, which is a convolution a
    /// channel: the same gradient, on the CPU and on Metal, and whole by
    /// the slopes.
    #[test]
    fn the_upsamples_backward_is_candles() {
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            let x = randn(&[2, 6, 5, 7], &dev);
            let r = randn(&[2, 6, 10, 14], &dev);
            let ours = |x: &Tensor| (upsample_twice(x)? * &r)?.sum_all();
            let theirs = |x: &Tensor| (x.upsample_nearest2d(10, 14)? * &r)?.sum_all();
            complete(&ours, &x, 1e-2, 1e-2).unwrap();
            let grad = |f: &dyn Fn(&Tensor) -> candle_core::Result<Tensor>| {
                let var = Var::from_tensor(&x).unwrap();
                f(var.as_tensor()).unwrap().backward().unwrap().get(var.as_tensor()).unwrap().clone()
            };
            assert!(off(&grad(&ours), &grad(&theirs)) < 1e-6);
            assert!(off(&upsample_twice(&x).unwrap(), &x.upsample_nearest2d(10, 14).unwrap()) == 0.0);
        }
    }

    /// A frozen convolution's backward is candle's, for a 3×3 and a 1×1 at
    /// stride 1, which go by an ordinary convolution of the turned kernel,
    /// and a 3×3 at stride 2, which keeps the transposed one: the same
    /// gradient as `conv2d`'s own backward gives its input, on the CPU and
    /// on Metal.
    #[test]
    fn a_convolutions_backward_is_candles() {
        use crate::image::nn::back_through_conv;
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            // (kernel, padding, stride, rows, columns)
            for (k, pad, stride, h, wd) in [(3, 1, 1, 6, 7), (1, 0, 1, 6, 7), (3, 1, 2, 6, 6), (3, 1, 2, 7, 7), (3, 0, 2, 7, 7), (3, 1, 2, 6, 7), (3, 0, 2, 8, 7)] {
                let x = randn(&[2, 5, h, wd], &dev);
                let w = randn(&[4, 5, k, k], &dev);
                let r = randn(x.conv2d(&w, pad, stride, 1, 1).unwrap().dims(), &dev);
                let got = back_through_conv(&w, pad, stride, &x, &r).unwrap();
                assert_eq!(got.dims(), x.dims());
                let what = format!("{k}×{k}, pad {pad}, stride {stride}, {h}×{wd} on {:?}", dev.location());
                // By the slopes, attached as a frozen convolution is.
                let f = |x: &Tensor| (x.conv2d(&w, pad, stride, 1, 1)? * &r)?.sum_all();
                let attached = |x: &Tensor| {
                    let w = w.clone();
                    let y = attach(x, x.detach().conv2d(&w, pad, stride, 1, 1)?, move |x, g| back_through_conv(&w, pad, stride, x, g))?;
                    (y * &r)?.sum_all()
                };
                let s = directional(&attached, &x, 1e-2, 9).unwrap();
                assert!(s.apart() < 1e-2, "{what}: {s:?}");
                // And against candle's own backward, where it has one: at
                // stride 2 it takes the rows' leftover for the columns too,
                // and fails where they differ.
                if stride == 1 || (h - wd) % 2 == 0 {
                    let var = Var::from_tensor(&x).unwrap();
                    let want = f(var.as_tensor()).unwrap().backward().unwrap().get(var.as_tensor()).unwrap().clone();
                    assert!(off(&got, &want) < 1e-5, "{what}: {} apart", off(&got, &want));
                }
            }
        }
    }

    /// Attention's own backward is the one candle finds through the
    /// written-out attention: for the queries, the keys and the values,
    /// self-attention and cross-attention (fewer keys than queries), on the
    /// CPU and on Metal; and in batches of query rows as in one.
    #[test]
    fn attentions_backward_is_candles() {
        use crate::image::nn::written_out;
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            // (queries, keys): self-attention, cross-attention, and enough
            // queries for three batches of rows at this test's budget.
            for (lq, lk) in [(12, 12), (12, 5), (40, 40)] {
                let (q, k, v) = (randn(&[1, 2, lq, 8], &dev), randn(&[1, 2, lk, 8], &dev), randn(&[1, 2, lk, 8], &dev));
                let r = randn(&[1, 2, lq, 8], &dev);
                let scale = 1.0 / 8f64.sqrt();
                let vars = [Var::from_tensor(&q).unwrap(), Var::from_tensor(&k).unwrap(), Var::from_tensor(&v).unwrap()];
                let (qv, kv, vv) = (vars[0].as_tensor(), vars[1].as_tensor(), vars[2].as_tensor());
                let theirs = (written_out(qv, kv, vv, scale).unwrap() * &r).unwrap().sum_all().unwrap().backward().unwrap();
                let y = attended(qv, kv, vv, scale, |q, k, v| written_out(q, k, v, scale)).unwrap();
                assert!(off(&y, &written_out(&q, &k, &v, scale).unwrap()) < 1e-6);
                let ours = (y * &r).unwrap().sum_all().unwrap().backward().unwrap();
                for (name, var) in ["q", "k", "v"].iter().zip(&vars) {
                    let (got, want) = (ours.get(var.as_tensor()).expect("a gradient for each"), theirs.get(var.as_tensor()).unwrap());
                    assert!(off(got, want) < 1e-4, "{lq} queries, {lk} keys, {name} on {:?}: {} apart", dev.location(), off(got, want));
                }
                // The same in batches of 16 rows' scores, three rows at a
                // time for the last shape.
                let whole = attention_back(&q, &k, &v, scale, &r).unwrap();
                let batched = attention_back_in(&q, &k, &v, scale, &r, 2 * 16 * 3).unwrap();
                assert!(off(&batched, &whole) < 1e-5, "{lq} queries in batches: {} apart", off(&batched, &whole));
            }
        }
    }

    /// A norm's own backward is candle's through the norm written out: the
    /// RMS norm and the layer norm, with weights, and the group norm and
    /// the plain layer norm of the image models; on the CPU and on Metal.
    #[test]
    fn a_norms_backward_is_candles() {
        use crate::image::nn::layer_norm_plain;
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            let x = randn(&[2, 5, 24], &dev);
            let r = randn(&[2, 5, 24], &dev);
            let (w, b) = (randn(&[24], &dev), randn(&[24], &dev));
            type F<'a> = &'a dyn Fn(&Tensor) -> candle_core::Result<Tensor>;
            let grad = |f: F| {
                let var = Var::from_tensor(&x).unwrap();
                let y = f(var.as_tensor()).unwrap();
                ((&y * &r).unwrap().sum_all().unwrap().backward().unwrap().get(var.as_tensor()).expect("a gradient").clone(), y)
            };
            let pairs: [(&str, F, F); 3] = [
                ("rms_norm", &|x| rms_norm(x, &w, 1e-6), &|x| ops::rms_norm_slow(x, &w, 1e-6)),
                ("layer_norm", &|x| layer_norm(x, &w, &b, 1e-5), &|x| ops::layer_norm_slow(x, &w, &b, 1e-5)),
                // Written out as it is for an untracked tensor.
                ("layer_norm_plain", &|x| layer_norm_plain(x, 1e-6), &|x| {
                    let c = x.broadcast_sub(&x.mean_keepdim(candle_core::D::Minus1)?)?;
                    c.broadcast_div(&(c.sqr()?.mean_keepdim(candle_core::D::Minus1)? + 1e-6)?.sqrt()?)
                }),
            ];
            for (name, ours, theirs) in pairs {
                let ((got, y), (want, y_want)) = (grad(ours), grad(theirs));
                assert!(off(&y, &y_want) < 1e-5, "{name}: the answers are {} apart", off(&y, &y_want));
                assert!(off(&got, &want) < 1e-4, "{name} on {:?}: {} apart", dev.location(), off(&got, &want));
            }
        }
    }

    /// A write in place of a tensor that is being differentiated is
    /// refused, where candle's own makes it and loses the gradient without
    /// a word: `backward` then finds nothing for what was written. The same
    /// write of a tensor that is not is made.
    #[test]
    fn a_write_in_place_refuses_a_tracked_tensor() {
        for dev in [Some(Device::Cpu), metal()].into_iter().flatten() {
            let part = randn(&[2, 4], &dev);
            let whole = Tensor::zeros((5, 4), DType::F32, &dev).unwrap();
            slice_set(&whole, &part, 0, 1).unwrap();
            assert!(off(&whole.narrow(0, 1, 2).unwrap(), &part) == 0.0);

            let var = Var::from_tensor(&part).unwrap();
            let made = (var.as_tensor() * 2.0).unwrap();
            let e = slice_set(&whole, &made, 0, 1).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("backward is not supported"), "{e}");
            let e = slice_set(var.as_tensor(), &part, 0, 0).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("backward is not supported"), "{e}");

            // What it is refused for: candle's, with the same tensors.
            whole.slice_set(&made, 0, 1).unwrap();
            let grads = whole.sum_all().unwrap().backward().unwrap();
            assert!(grads.get(var.as_tensor()).is_none(), "candle's slice_set now records what it writes");
        }
    }

    /// The three kernels that add into a buffer in place decline a tensor
    /// that is being differentiated, whichever of their operands it is, and
    /// take the same ones untracked.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_kernel_that_writes_in_place_declines_a_tracked_tensor() {
        let Some(dev) = metal() else { return };
        if !crate::mpp::available(&dev) {
            return;
        }
        let half = |shape: &[usize]| randn(shape, &dev).to_dtype(DType::BF16).unwrap();
        let var = |t: &Tensor| Var::from_tensor(t).unwrap().as_tensor().clone();
        let (x, w, b) = (half(&[512, 64]), half(&[64, 96]), half(&[96]));
        assert!(crate::mpp::dense_bias(&x, &w, &b).unwrap().is_some());
        for (i, (x, w, b)) in [(var(&x), w.clone(), b.clone()), (x.clone(), var(&w), b.clone()), (x.clone(), w.clone(), var(&b))].iter().enumerate() {
            assert!(crate::mpp::dense_bias(x, w, b).unwrap().is_none(), "dense_bias, operand {i}");
        }
        let answer = || half(&[512, 96]);
        assert!(crate::mpp::dense_acc(&answer(), &x, &w).unwrap());
        assert!(!crate::mpp::dense_acc(&var(&answer()), &x, &w).unwrap());
        assert!(!crate::mpp::dense_acc(&answer(), &var(&x), &w).unwrap());
        assert!(!crate::mpp::dense_acc(&answer(), &x, &var(&w)).unwrap());
    }
}
