//! The two-dimensional vocabulary: what an image model is built from that a
//! language model never needed.
//!
//! A language model is a stack of matrix multiplies over a sequence. An image
//! model is that too, in its attention layers, but around them sit things
//! that only make sense on a grid: a **convolution**, which mixes each pixel
//! with its neighbours through a small learned kernel; a **group norm**, which
//! normalises over space as well as over channels; and **upsampling**, which
//! makes the grid bigger. Everything here is one of those, or an adapter
//! between a grid (`[B, C, H, W]`) and a sequence (`[B, H·W, C]`), which is
//! what an attention layer inside a UNet has to do on the way in and out.
//!
//! Every weight is read through a [`Reader`], so the unread-tensor guard that
//! protects the text models protects these too.

use super::lora::Slot;
use crate::common::{settle, Loader, Proj, Reader, Stored};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What every loader here needs besides the reader: where the weights go and
/// in what precision they compute.
pub(crate) struct Ctx<'v> {
    pub(crate) ld: Loader<'v>,
    pub(crate) dtype: DType,
}

impl Ctx<'_> {
    pub(crate) fn device(&self) -> &Device {
        &self.ld.device
    }

    /// A tensor as the checkpoint has it, in the compute dtype, on the device.
    pub(crate) fn get(&self, r: &Reader<'_>, shape: impl Into<candle_core::Shape>, name: &str) -> Res<Tensor> {
        Ok(r.get(shape, name)?.to_dtype(self.dtype)?.to_device(self.device())?)
    }
}

/// `y = x·Wᵀ + b` over the last axis, whatever the axes before it.
pub(crate) struct Linear {
    w: Proj,
    b: Option<Tensor>,
    out: usize,
    /// Its place among the layers a LoRA may adapt, when it was loaded
    /// through a reader that registers them ([`super::lora`]).
    slot: Option<Slot>,
}

impl Linear {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, inp: usize, out: usize, bias: bool) -> Res<Self> {
        let slot = r.adapters().map(|a| a.linear(&r.full(name), inp, out));
        let r = r.pp(name);
        let w = cx.ld.proj(&r, "weight", out, inp, Stored::OutIn)?;
        let w = match w {
            // The loader keeps a dense matrix in the dtype it was read in;
            // the pipeline computes in one dtype throughout.
            Proj::Dense(t) => Proj::Dense(t.to_dtype(cx.dtype)?),
            q => q,
        };
        let b = match bias {
            true => Some(cx.get(&r, out, "bias")?),
            false => None,
        };
        Ok(Linear { w, b, out, slot })
    }

    /// A 1×1 convolution as the linear layer it is: `[out, in, 1, 1]` is
    /// `[out, in]` with two axes of one. SD 1.5's transformers project in
    /// and out this way, over a grid Kvad has already made a sequence.
    pub(crate) fn load_1x1(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, inp: usize, out: usize) -> Res<Self> {
        let slot = r.adapters().map(|a| a.linear(&r.full(name), inp, out));
        let r = r.pp(name);
        let w = r.get((out, inp, 1, 1), "weight")?.reshape((out, inp))?.t()?.contiguous()?;
        let w = Proj::Dense(w.to_dtype(cx.dtype)?.to_device(cx.device())?);
        Ok(Linear { w, b: Some(cx.get(&r, out, "bias")?), out, slot })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.frozen(x)?;
        match &self.slot {
            Some(s) => s.add(x, y),
            None => Ok(y),
        }
    }

    /// [`Linear::plain`], for an `x` that may be being differentiated: the
    /// layer's own weights are not trained, so its answer is made out of
    /// `backward`'s sight, bias and all, and attached with the one gradient
    /// it owes, its input's (`crate::grad::attach`).
    fn frozen(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match &self.w {
            Proj::Dense(w) if x.track_op() && !w.track_op() => {
                let w = w.clone();
                crate::grad::attach(x, self.plain(&x.detach())?, move |x, g| crate::grad::back_through(&w, x, g))
            }
            _ => self.plain(x),
        }
    }

    /// Whether a LoRA is set on this layer: its answer is then the layer's,
    /// bias and all, and the LoRA's side path added to it, in that order,
    /// so nothing may be fused after the bias.
    fn adapted(&self) -> bool {
        self.slot.as_ref().is_some_and(Slot::active)
    }

    fn plain(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // The bias is added to the kernel's f32 sums as they are stored.
        #[cfg(target_os = "macos")]
        if let Proj::Blocks(q) = &self.w {
            return q.linear(x, self.b.as_ref(), DType::F32, false);
        }
        self.written_out(x)
    }

    /// [`Linear::forward`], answered in `dt`. A Q8_0 matrix on the M5's
    /// matrix units rounds each f32 sum, bias and all, straight to `dt` as
    /// it stores it: the same numbers as `forward(x)?.to_dtype(dt)`, without
    /// writing the f32 answer and reading it back twice.
    pub(crate) fn forward_in(&self, x: &Tensor, dt: DType) -> candle_core::Result<Tensor> {
        if self.adapted() {
            return self.forward(x)?.to_dtype(dt);
        }
        #[cfg(target_os = "macos")]
        if let Proj::Blocks(q) = &self.w {
            return q.linear(x, self.b.as_ref(), dt, false);
        }
        self.written_out(x)?.to_dtype(dt)
    }

    /// The tanh-GELU of [`Linear::forward`], in `dt`, where the matrix's
    /// kernel can take it in its store: from the f32 sum, rounded once.
    /// `None` where it cannot, for the caller to apply its own.
    pub(crate) fn gelu_in(&self, x: &Tensor, dt: DType) -> candle_core::Result<Option<Tensor>> {
        if self.adapted() {
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        if let Proj::Blocks(q) = &self.w {
            return q.linear(x, self.b.as_ref(), dt, true).map(Some);
        }
        let _ = (x, dt);
        Ok(None)
    }

    fn written_out(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Flattened to two axes for the multiply: a dense `matmul` wants its
        // operands' batch axes to agree, and a weight has none.
        let dims = x.dims().to_vec();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((rows, dims[dims.len() - 1]))?;
        let mut shape = dims;
        *shape.last_mut().unwrap() = self.out;
        // A dense matrix with a bias, on the M5's matrix units: the bias
        // joins the f32 sums and the answer is rounded once, as PyTorch's
        // `addmm` does, rather than rounded, added to and rounded again.
        #[cfg(target_os = "macos")]
        if let (Proj::Dense(w), Some(b)) = (&self.w, &self.b) {
            if let Some(y) = crate::mpp::dense_bias(&x2, w, b)? {
                return y.reshape(shape);
            }
        }
        let mut y = self.w.forward(&x2)?;
        if let Some(b) = &self.b {
            // A Q8_0 matrix answers in f32 even when asked in bf16, and the
            // bias is in the pipeline's dtype; add it in the answer's.
            y = y.broadcast_add(&b.to_dtype(y.dtype())?)?;
        }
        y.reshape(shape)
    }
}

/// A 2D convolution: each output pixel is a weighted sum over a `k×k`
/// neighbourhood of every input channel.
///
/// Stored `[out, in, k, k]`, which is what candle wants as it is. candle does
/// it as `im2col` — copy every neighbourhood out into a row of its own — then
/// one matrix multiply, which spends memory to turn the one operation a GPU
/// is not built for into the one it is.
pub(crate) struct Conv2d {
    w: Tensor,
    b: Tensor,
    stride: usize,
    pad: usize,
    /// Its place among the layers a LoRA may adapt ([`super::lora`]).
    slot: Option<Slot>,
}

impl Conv2d {
    pub(crate) fn load(
        cx: &Ctx<'_>,
        r: &Reader<'_>,
        name: &str,
        (cin, cout, k): (usize, usize, usize),
        stride: usize,
    ) -> Res<Self> {
        // "Same" padding: a 3×3 kernel keeps the grid its size.
        let pad = k / 2;
        let slot = r.adapters().map(|a| a.conv(&r.full(name), (cin, cout, k), stride, pad));
        let r = r.pp(name);
        Ok(Conv2d { w: cx.get(&r, (cout, cin, k, k), "weight")?, b: cx.get(&r, cout, "bias")?.reshape((1, cout, 1, 1))?, stride, pad, slot })
    }

    /// From a weight already in hand, for a checkpoint that stores it some
    /// other way (see the Qwen-Image VAE, whose kernels are three-dimensional).
    pub(crate) fn from_parts(w: Tensor, b: Tensor, stride: usize) -> Res<Self> {
        let (cout, _, k, _) = w.dims4()?;
        Ok(Conv2d { b: b.reshape((1, cout, 1, 1))?, w, stride, pad: k / 2, slot: None })
    }

    /// With no padding of its own, for a caller that pads first: the VAE
    /// encoder's downsampler pads on one side only, which a symmetric pad
    /// cannot say.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn unpadded(self) -> Self {
        Conv2d { pad: 0, ..self }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let plain = |x: &Tensor| self.banded(x, BAND);
        // A kernel that is not being trained owes `backward` its input's
        // gradient and no more (`crate::grad::attach`): the transposed
        // convolution of the answer's, as candle's own backward makes it,
        // without the kernel's, which is a convolution again as large.
        let y = match x.track_op() && !self.w.track_op() {
            true => {
                let (w, pad, stride) = (self.w.clone(), self.pad, self.stride);
                crate::grad::attach(x, plain(&x.detach())?, move |x, g| back_through_conv(&w, pad, stride, x, g))?
            }
            false => plain(x)?,
        };
        match &self.slot {
            Some(s) => s.add_conv(x, y),
            None => Ok(y),
        }
    }
}

/// The most a convolution or a norm may gather at once, in bytes, before it
/// is done a piece at a time ([`Conv2d::banded`], [`GroupNorm::grouped`]).
///
/// A VAE works at the picture's own size, and two of its operations make
/// far more than they are given. candle's convolution on Metal is a matrix
/// product over gathered neighbourhoods: every input number nine times
/// over for a 3×3 kernel, 4.8 GB for 256 channels at 1024² in f16. A group
/// norm sums in f32 and makes half a dozen tensors its input's size on the
/// way, a gigabyte each there. And candle gives a dropped buffer back only
/// when the device is waited for, so all of them were held at once.
///
/// Both are sums over a neighbourhood or a group and nothing else, so a
/// piece at a time is the same arithmetic on the same numbers, and the
/// answer is the same to the bit (`tests::a_piece_at_a_time_is_the_whole`).
/// SDXL's VAE alone, by `vae::tests::one_encode`, on an M5 Pro:
///
/// | | whole | a piece at a time |
/// |---|---|---|
/// | decode, 512² | 2.1 s, 5.7 GB | 2.9 s, 2.9 GB |
/// | decode, 768² | 5.1 s, 12.7 GB | 7.0 s, 3.3 GB |
/// | decode, 1024² | not run alone | 12.4 s, 3.6 GB |
/// | encode, 1024² | 5.1 s, 12.0 GB | 6.4 s, 4.7 GB |
///
/// What it costs in time is the waits: one a piece, and a wait lets the
/// buffer pool go, so the next piece writes fresh ones. At 512 MB a 1024²
/// decode was as slow, 12.4 s, and reached 6.0 GB.
pub(crate) const BAND: usize = 256 << 20;

impl Conv2d {
    /// The convolution and its bias, in bands of rows where gathering the
    /// whole input's neighbourhoods would pass `budget` bytes ([`BAND`]):
    /// each band is convolved with `pad` rows of its neighbours either
    /// side, which are then cut from its answer, and the device is waited
    /// for before the next. A stride of 1 only, and never a tensor
    /// `backward` is to walk: a record is not made in pieces.
    fn banded(&self, x: &Tensor, budget: usize) -> candle_core::Result<Tensor> {
        let (n, c, h, w) = x.dims4()?;
        let k = self.w.dim(2)?;
        let rows = (budget / (n * c * k * k * w * x.dtype().size_in_bytes()).max(1)).max(1);
        if self.stride != 1 || k == 1 || rows >= h || x.track_op() {
            return x.conv2d(&self.w, self.pad, self.stride, 1, 1)?.broadcast_add(&self.b);
        }
        let mut bands = Vec::with_capacity(h.div_ceil(rows));
        for r0 in (0..h).step_by(rows) {
            let take = rows.min(h - r0);
            let (from, to) = (r0.saturating_sub(self.pad), (r0 + take + self.pad).min(h));
            bands.push(x.narrow(2, from, to - from)?.conv2d(&self.w, self.pad, 1, 1, 1)?.narrow(2, r0 - from, take)?.broadcast_add(&self.b)?);
            settle(x.device())?;
        }
        Tensor::cat(&bands, 2)
    }
}

/// `∂L/∂x` for `y = conv2d(x, w)` with `w` `[out, in, k, k]`: the transposed
/// convolution of `∂L/∂y`, in `x`'s shape and dtype.
///
/// At stride 1 it is computed as an ordinary convolution, by the kernel
/// turned half round and its two channel axes exchanged, padded `k − 1 −
/// pad`: the same sums. candle's Metal convolution is a matrix product over
/// gathered neighbourhoods; its transposed one is a thread for every output
/// number, looping over every input channel and tap, and a UNet's
/// convolutions are 1280 channels wide. A stride of 2 keeps the transposed
/// one: there are three in SDXL.
pub(crate) fn back_through_conv(w: &Tensor, pad: usize, stride: usize, x: &Tensor, grad: &Tensor) -> candle_core::Result<Tensor> {
    let k = w.dim(2)?;
    let g = grad.to_dtype(w.dtype())?;
    let back = match stride {
        1 if folds(w, &g) => back_folded(w, pad, &g)?,
        1 => g.conv2d(&w.flip(&[2, 3])?.transpose(0, 1)?.contiguous()?, k - 1 - pad, 1, 1, 1)?,
        _ => {
            // What the stride's rounding left off the far edge, which need
            // not be the same down as across: 6 rows and 7 columns at
            // stride 2 leave one row and no column. The transposed
            // convolution takes one number for both, so it is given the
            // larger and the answer cut to `x`'s size. (candle's own
            // backward gives both the rows' and fails on such a shape.)
            let (_, _, h, wd) = x.dims4()?;
            let left = |n: usize, gn: usize| n - ((gn - 1) * stride + k - 2 * pad);
            let (rows, cols) = (left(h, g.dim(2)?), left(wd, g.dim(3)?));
            g.conv_transpose2d(w, pad, rows.max(cols), stride, 1)?.narrow(2, 0, h)?.narrow(3, 0, wd)?.contiguous()?
        }
    };
    back.to_dtype(x.dtype())
}

/// Whether [`back_folded`] is the way back through this kernel: where it
/// has more than a twentieth as many output channels as the grid has
/// cells. Measured on SDXL's shapes in f16, the two are level at a
/// twelfth (320 channels on 64×64, 9 ms each); at 5 channels a cell the
/// fold is 2.5 ms where turning the kernel is 42, and at a fiftieth (320
/// on 128×128) it is 40 where turning is 32, for its shares are nine times
/// the grid.
fn folds(w: &Tensor, g: &Tensor) -> bool {
    let (Ok(out), Ok(cells)) = (w.dim(0), g.dims4().map(|(_, _, h, w)| h * w)) else { return false };
    20 * out > cells
}

/// [`back_through_conv`] at stride 1 with the kernel read as it is stored.
///
/// The other way turns the kernel round first, and that is a copy of it:
/// 29 MB for a 1280-channel 3×3, 38 ms where the convolution itself is 3.5
/// on a 16×16 grid, at every step. This one takes the answer's gradient
/// through the kernel as a matrix, `[out, in·k·k]`, the layout it has, which
/// gives every input cell's share of each of the `k·k` taps it was read by,
/// and then adds each tap's shares into the cell they came from: the
/// gathering a convolution starts with, run backwards.
pub(crate) fn back_folded(w: &Tensor, pad: usize, g: &Tensor) -> candle_core::Result<Tensor> {
    let (o, c, k, _) = w.dims4()?;
    let (b, _, gh, gw) = g.dims4()?;
    let rows = g.permute((0, 2, 3, 1))?.contiguous()?.reshape((b * gh * gw, o))?;
    let kernel = w.reshape((o, c * k * k))?;
    #[cfg(target_os = "macos")]
    let shares = match crate::mpp::dense(&rows, &kernel)? {
        Some(y) => y,
        None => rows.matmul(&kernel)?,
    };
    #[cfg(not(target_os = "macos"))]
    let shares = rows.matmul(&kernel)?;
    let shares = shares.reshape((b, gh, gw, c, k * k))?;
    // The input as it was padded, channels last: tap (ky, kx) of the
    // answer's cell (y, x) read cell (y + ky, x + kx).
    let mut sum: Option<Tensor> = None;
    for tap in 0..k * k {
        let (ky, kx) = (tap / k, tap % k);
        let placed = shares.narrow(4, tap, 1)?.reshape((b, gh, gw, c))?.pad_with_zeros(1, ky, k - 1 - ky)?.pad_with_zeros(2, kx, k - 1 - kx)?;
        sum = Some(match sum {
            Some(s) => (s + placed)?,
            None => placed,
        });
    }
    let (h, wd) = (gh + k - 1 - 2 * pad, gw + k - 1 - 2 * pad);
    sum.expect("a kernel has a tap").narrow(1, pad, h)?.narrow(2, pad, wd)?.permute((0, 3, 1, 2))?.contiguous()
}

/// Group norm: split the channels into `groups` groups and normalise each
/// group over all its channels *and every pixel*, then scale and shift per
/// channel.
///
/// Why not layer norm, which the text models use: a layer norm normalises one
/// position's channels, and in a convolutional network that would erase the
/// one thing a pixel's feature vector is for, its size relative to its
/// neighbours'. Why not batch norm: it depends on the other images in the
/// batch, which at inference is one image and its unconditional twin. Group
/// norm is the one that is neither, and every diffusion model uses it.
///
/// Computed in f32 whatever the pipeline's dtype: a group at 1024×1024 is
/// four million numbers, and their sum of squares in f16 is noise.
pub(crate) struct GroupNorm {
    groups: usize,
    eps: f64,
    w: Tensor,
    b: Tensor,
}

impl GroupNorm {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, channels: usize, groups: usize, eps: f64) -> Res<Self> {
        let r = r.pp(name);
        Ok(GroupNorm {
            groups,
            eps,
            w: cx.get(&r, channels, "weight")?.reshape((1, channels, 1, 1))?,
            b: cx.get(&r, channels, "bias")?.reshape((1, channels, 1, 1))?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        // Its own backward for a tensor that wants one, where the norm's
        // weights are not being trained: `grad::norm_back` over each
        // group's channels and pixels as one axis, the weight multiplied
        // into the gradient first.
        if x.track_op() && !crate::grad::tracked(&[&self.w, &self.b]) {
            let (weight, eps, groups) = (self.w.clone(), self.eps, self.groups);
            return crate::grad::attach(x, self.forward(&x.detach())?, move |x, g| {
                let grouped = (b, groups, (c / groups) * h * w);
                let g = g.broadcast_mul(&weight.to_dtype(g.dtype())?)?.reshape(grouped)?;
                crate::grad::norm_back(&x.reshape(grouped)?, &g, eps, true)?.reshape((b, c, h, w))?.to_dtype(x.dtype())
            });
        }
        self.grouped(x, BAND)
    }

    /// The norm, a few groups at a time where the whole input in the
    /// precision it is summed in would pass `budget` bytes ([`BAND`]), with
    /// the device waited for after each: a group's numbers are normalised
    /// by that group's own mean and spread, so its neighbours need not be
    /// there.
    fn grouped(&self, x: &Tensor, budget: usize) -> candle_core::Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let per = c / self.groups;
        let at_once = (budget / (b * per * h * w * wide(x.dtype()).size_in_bytes()).max(1)).max(1);
        if at_once >= self.groups || x.track_op() {
            return self.whole(x);
        }
        let mut parts = Vec::with_capacity(self.groups.div_ceil(at_once));
        for g0 in (0..self.groups).step_by(at_once) {
            let n = at_once.min(self.groups - g0);
            let (from, len) = (g0 * per, n * per);
            let part = GroupNorm { groups: n, eps: self.eps, w: self.w.narrow(1, from, len)?, b: self.b.narrow(1, from, len)? };
            parts.push(part.whole(&x.narrow(1, from, len)?)?);
            settle(x.device())?;
        }
        Tensor::cat(&parts, 1)
    }

    fn whole(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let dtype = x.dtype();
        let g = x.to_dtype(wide(dtype))?.reshape((b, self.groups, (c / self.groups) * h * w))?;
        let mean = g.mean_keepdim(D::Minus1)?;
        let g = g.broadcast_sub(&mean)?;
        let var = g.sqr()?.mean_keepdim(D::Minus1)?;
        let g = g.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        g.reshape((b, c, h, w))?.to_dtype(dtype)?.broadcast_mul(&self.w)?.broadcast_add(&self.b)
    }
}

/// Layer norm over the last axis, with weight and bias.
pub(crate) struct LayerNorm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub(crate) fn load(cx: &Ctx<'_>, r: &Reader<'_>, name: &str, width: usize, eps: f32) -> Res<Self> {
        let r = r.pp(name);
        Ok(LayerNorm { w: cx.get(&r, width, "weight")?, b: cx.get(&r, width, "bias")?, eps })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        crate::grad::layer_norm(&x.contiguous()?, &self.w, &self.b, self.eps)
    }
}

/// The precision a sum over many values is taken in: f32 for the half
/// precisions, whose sums overflow or lose their small terms, and the
/// tensor's own otherwise. Not f32 whatever comes: a gradient is checked in
/// f64 (`crate::grad`), and a step that rounded to f32 on the way would
/// leave the check nothing finer than f32 to measure.
pub(crate) fn wide(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    }
}

/// Layer norm with no weight or bias of its own, which a DiT's blocks use
/// because the time embedding supplies scale and shift instead.
pub(crate) fn layer_norm_plain(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    // Its own backward for a tensor that wants one (`grad::norm_back`).
    if x.track_op() {
        return crate::grad::attach(x, layer_norm_plain(&x.detach(), eps)?, move |x, g| crate::grad::norm_back(x, g, eps, true)?.to_dtype(x.dtype()));
    }
    let dtype = x.dtype();
    let x = x.to_dtype(wide(dtype))?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dtype)
}

/// `[B, C, H, W]` to `[B, H·W, C]`: from a grid to a sequence of pixels.
pub(crate) fn to_seq(x: &Tensor) -> candle_core::Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()
}

/// `[B, H·W, C]` back to `[B, C, H, W]`.
pub(crate) fn to_grid(x: &Tensor, h: usize, w: usize) -> candle_core::Result<Tensor> {
    let (b, _, c) = x.dims3()?;
    x.transpose(1, 2)?.contiguous()?.reshape((b, c, h, w))
}

/// Multi-head attention with no mask: `q` is `[B, Lq, heads·d]`, `k` and `v`
/// `[B, Lk, heads·d]`, and the answer is shaped like `q`.
///
/// Nothing in an image is causal. Every pixel sees every other pixel and every
/// prompt token, so there is no mask to build and no future to hide — the
/// thing that made the fused kernel delicate for the text models
/// (`model.rs`, `fused`) does not arise.
///
/// What does arise is size. SDXL's second level is 4096 positions, so a score
/// matrix is 4096² per head, per image, per guidance branch: 335 M numbers
/// for ten heads and two branches. The fused kernel never writes it down, and
/// where it cannot be used — the CPU, or a head width it was not compiled
/// for — the written-out version does the queries a slice at a time, so the
/// matrix that does exist stays under [`SCORES`].
pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> candle_core::Result<Tensor> {
    let (b, lq, width) = q.dims3()?;
    let lk = k.dim(1)?;
    let d = width / heads;
    let split = |x: &Tensor, l: usize| -> candle_core::Result<Tensor> {
        x.reshape((b, l, heads, d))?.transpose(1, 2)?.contiguous()
    };
    let (q, k, v) = (split(q, lq)?, split(k, lk)?, split(v, lk)?);
    let scale = 1.0 / (d as f64).sqrt();
    // With a backward of its own for tensors that want one
    // (`grad::attended`), so that candle's kernel, which has none, only
    // ever sees tensors that do not; if one reaches here still tracked,
    // the written-out attention has candle's.
    let out = crate::grad::attended(&q, &k, &v, scale, |q, k, v| match fused(q.device(), q.dtype(), d, lq) && !crate::grad::tracked(&[q, k, v]) {
        true => ops::sdpa(q, k, v, None, false, scale as f32, 1.0),
        false => written_out(q, k, v, scale),
    })?;
    out.transpose(1, 2)?.contiguous()?.reshape((b, lq, width))
}

/// The most score-matrix entries [`written_out`] builds at once: 128 M, which
/// is 512 MB in f32.
const SCORES: usize = 1 << 27;

/// Whether MLX's fused kernel will take this attention.
///
/// The head widths are the ones it is compiled for, and f32 at 512 overflows
/// a threadgroup's memory. It is never asked for a mask here, which is the
/// part of it that is wrong on partial tiles (see `model.rs`), and one query
/// row is a different kernel altogether that this never needs.
fn fused(device: &Device, dtype: DType, head_dim: usize, lq: usize) -> bool {
    device.is_metal()
        && lq > 1
        && matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
        && !(head_dim == 512 && dtype == DType::F32)
}

/// Attention with the scores written down, a slice of queries at a time.
pub(crate) fn written_out(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
    let (b, h, lq, _) = q.dims4()?;
    let lk = k.dim(2)?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let rows = (SCORES / (b * h * lk)).clamp(1, lq);
    let mut parts = Vec::with_capacity(lq.div_ceil(rows));
    let mut start = 0;
    while start < lq {
        let n = rows.min(lq - start);
        let qs = q.narrow(2, start, n)?.contiguous()?;
        // Softmax in f32: a row of 16384 f16 exponentials sums past f16's
        // largest number long before it is done.
        let att = (qs.matmul(&kt)?.to_dtype(wide(q.dtype()))? * scale)?;
        let att = crate::grad::softmax_last_dim(&att)?.to_dtype(v.dtype())?;
        parts.push(att.matmul(v)?);
        start += n;
    }
    Tensor::cat(&parts, 2)
}

/// A timestep as a vector: sines and cosines of it at geometrically spaced
/// frequencies, the transformer's position encoding applied to *time*.
///
/// `t` is one number per image. The frequencies are
/// `exp(−ln(10000) · i / (half − shift))` for `i < half`; `cos_first` is
/// diffusers' `flip_sin_to_cos`, which every model here sets.
pub(crate) fn timestep_embedding(t: &[f64], dim: usize, cos_first: bool, shift: f64, device: &Device) -> Res<Tensor> {
    let half = dim / 2;
    let mut data: Vec<f32> = Vec::with_capacity(t.len() * dim);
    for &t in t {
        let angle = |i: usize| t * (-(10000f64.ln()) * i as f64 / (half as f64 - shift)).exp();
        let (sin, cos): (Vec<f32>, Vec<f32>) = (0..half).map(|i| (angle(i).sin() as f32, angle(i).cos() as f32)).unzip();
        match cos_first {
            true => data.extend(cos.iter().chain(&sin)),
            false => data.extend(sin.iter().chain(&cos)),
        }
    }
    Ok(Tensor::from_vec(data, (t.len(), dim), device)?)
}

/// What is wrong with numbers a failed step left behind, if anything: zero
/// everywhere, or anything in them not finite.
///
/// On Metal, candle serves a command buffer that failed as zeros and raises
/// nothing; an overflow leaves NaN or infinity. Neither a latent nor a VAE's
/// output is ever exactly zero everywhere, so the test costs nothing real.
///
/// It reads the numbers on the host rather than as a reduction on the
/// device: candle's `max` there skips NaN, so a latent part NaN and part
/// finite used to pass the check before decoding. In chunks, so each test is
/// a fold the autovectoriser can take: a 1536 × 1024 clip's 2.3 GB of frames
/// in about 30 ms, where `all` short of a first failure takes 140.
fn broken(values: &[f32]) -> Option<&'static str> {
    let (finite, zero) = values.chunks(1024).fold((true, true), |(finite, zero), c| {
        (finite && c.iter().fold(true, |a, v| a & v.is_finite()), zero && c.iter().fold(true, |a, &v| a & (v == 0.0)))
    });
    match (finite, zero) {
        (false, _) => Some("with NaN or infinity in it"),
        (true, true) => Some("zero everywhere"),
        (true, false) => None,
    }
}

/// Refuses a denoised latent that [`broken`] finds fault with, before the
/// decode spends seconds on it. Copying it to the host is cheap: a 1024²
/// image's latent is a megabyte in f32.
pub(crate) fn check_latent(x: &Tensor) -> Res<()> {
    let values = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    // A failure that left a stale tensor's numbers, not zeros, shows only
    // as the error `settle` reads.
    settle(x.device())?;
    let Some(what) = broken(&values) else { return Ok(()) };
    Err(format!(
        "the denoiser's result came back {what}; not decoding it. A failed command buffer on Metal reads as zeros \
         rather than an error, most often for want of GPU memory, and an overflow leaves NaN"
    )
    .into())
}

/// Refuses what a failed decode leaves behind: pixels that [`broken`] finds
/// fault with.
///
/// The likeliest reason a command buffer fails at the decode is memory: the
/// VAE's last layers are the pipeline's one stage at full resolution. FLUX
/// decoding 41 GB into swap, beside a second FLUX in another process, took
/// 193 s rather than 9 and came back black. Zero is mid-grey in `[−1, 1]`
/// and pure black in `[0, 1]`, and a real picture is neither to the last
/// bit. The pixels are on the host by now anyway, and a check on the device
/// would not see a conversion after it fail.
pub(crate) fn check_decoded(pixels: &[f32]) -> Res<()> {
    let Some(what) = broken(pixels) else { return Ok(()) };
    Err(format!(
        "the decoded image came back {what}: the decode failed, most likely for want of GPU memory, and a failed \
         command buffer on Metal reads as zeros rather than an error. Nothing was saved; free some memory (another \
         model loaded?) and try again"
    )
    .into())
}

/// Refuses an 8-bit image that is one flat colour but for a frame: what a
/// decode that failed part-way and carried on draws.
///
/// A stage of the VAE left as zeros is not zeros at the end, because the
/// layers after it add their biases. Its edges are not flat either: every
/// 3×3 convolution after the failure pads with zeros, and each reaches a
/// pixel further in at its own scale, so the frame of structure they leave
/// can be as deep as the convolutions after the failure, each counted at
/// its scale — about 130 pixels in both VAEs, whatever the image's size.
/// So the count is of the middle, inside a margin of a side's eighth and
/// never under 128 pixels. There, with each of their 21 stages in turn
/// zeroed on Metal, both SD 1.5's VAE and Qwen-Image's Wan VAE left one
/// colour, all 42 times; a 64-pixel margin let early failures' frames in,
/// with up to 1,554 colours.
///
/// A real picture has more, since the VAE's own grain alone makes them,
/// though a clean model asked for a flat colour draws little else: of
/// Qwen-Image's plain swatches, walls and black at 512², the fewest was
/// 121, a black. SD 1.5's fewest, a grey swatch at 512², was 943; the 27
/// FLUX.1-schnell images in the gallery 4,856, a logo on a plain ground.
/// The count stops at [`FLAT`], so a real picture costs a few pixels of
/// it. Smaller than 384 pixels a side, there is too little middle to judge,
/// and nothing is refused.
pub(crate) fn check_flat(rgb: &[u8], width: usize, height: usize) -> Res<()> {
    if width < 384 || height < 384 {
        return Ok(());
    }
    let (mx, my) = ((width / 8).max(128), (height / 8).max(128));
    let mut seen = std::collections::HashSet::with_capacity(FLAT);
    for y in my..height - my {
        for p in rgb[(y * width + mx) * 3..(y * width + width - mx) * 3].chunks_exact(3) {
            if seen.insert(p) && seen.len() >= FLAT {
                return Ok(());
            }
        }
    }
    Err(format!(
        "the decoded image is flat: {} in all of its middle, where even a plain one has a hundred. The decode failed \
         part-way, most likely for want of GPU memory, and the layers after the failure drew only their biases. \
         Nothing was saved; free some memory (another model loaded?) and try again",
        match seen.len() {
            1 => "one colour".to_string(),
            n => format!("{n} colours"),
        }
    )
    .into())
}

/// As few colours as a picture's middle may have and pass [`check_flat`]:
/// room for a failure's frame to reach a little further in than measured,
/// and 15 times under the fewest a real picture had. Nearer the failures
/// than the pictures on purpose: refusing a real picture blames memory for
/// nothing, where missing a failure saves what was saved before the check.
const FLAT: usize = 8;

/// Pixels in `[−1, 1]`, `[1, 3, H, W]`, to 8-bit RGB, once
/// [`check_decoded`] has passed them and [`check_flat`] the bytes.
///
/// The arithmetic is done here on the host, after the check, because on the
/// device it would be three more command buffers that could fail unseen:
/// the black 1024² FLUX image that prompted the check was that, or NaN,
/// which Metal's clamp turns into −1. A VAE's zeros alone would be grey.
pub(crate) fn to_rgb8(x: &Tensor) -> Res<kvad::image::Image> {
    let (_, c, h, w) = x.dims4()?;
    if c != 3 {
        return Err(format!("a decoded image should have 3 channels, not {c}").into());
    }
    let planes = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    // A failure that left a stale tensor's numbers, not zeros, shows only
    // as the error `settle` reads.
    settle(x.device())?;
    check_decoded(&planes)?;
    // Channels first to channels last: the same f32 steps the device took,
    // so the same bytes out.
    let n = h * w;
    let rgb: Vec<u8> = (0..n * 3).map(|i| ((planes[(i % 3) * n + i / 3].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8).collect();
    check_flat(&rgb, w, h)?;
    Ok(kvad::image::Image { width: w, height: h, rgb })
}

/// A preview: the latent's channels mixed into RGB by a fixed matrix.
///
/// `factors` is `[latent channels][3]` and `bias` is per colour. The result is
/// at the latent's resolution, an eighth of the image's in each direction.
pub(crate) fn latent_preview(latent: &Tensor, factors: &[[f32; 3]], bias: [f32; 3]) -> Res<kvad::image::Image> {
    let (_, c, h, w) = latent.dims4()?;
    let lat = latent.to_dtype(DType::F32)?.squeeze(0)?.flatten_from(1)?.to_vec2::<f32>()?;
    let mut rgb = vec![0u8; h * w * 3];
    for p in 0..h * w {
        for (col, b) in bias.iter().enumerate() {
            let v: f32 = (0..c).map(|ch| lat[ch][p] * factors[ch][col]).sum::<f32>() + b;
            rgb[p * 3 + col] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
        }
    }
    Ok(kvad::image::Image { width: w, height: h, rgb })
}

/// A random normal tensor from a seed, the same on every device.
///
/// candle's own `randn` draws on the device, and Metal's generator is not the
/// CPU's, so the same seed would paint different images on different
/// backends. The noise is drawn here instead, from a small generator written
/// out in full, and copied over.
pub(crate) fn noise(seed: u64, shape: &[usize], device: &Device, dtype: DType) -> Res<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = SplitMix(seed);
    let mut out = Vec::with_capacity(n + 1);
    // Box–Muller: two uniforms in, two independent normals out.
    while out.len() < n {
        let u1 = rng.unit().max(f64::MIN_POSITIVE);
        let u2 = rng.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out.push((r * a.cos()) as f32);
        out.push((r * a.sin()) as f32);
    }
    out.truncate(n);
    Ok(Tensor::from_vec(out, shape, &Device::Cpu)?.to_dtype(dtype)?.to_device(device)?)
}

/// A tensor of numbers drawn evenly from `[−bound, bound)`, in f32, from a
/// seed, the same on every device as [`noise`] is.
pub(crate) fn uniform(seed: u64, shape: &[usize], bound: f64, device: &Device) -> Res<Tensor> {
    let mut rng = SplitMix(seed);
    let out: Vec<f32> = (0..shape.iter().product()).map(|_| ((2.0 * rng.unit() - 1.0) * bound) as f32).collect();
    Ok(Tensor::from_vec(out, shape, &Device::Cpu)?.to_device(device)?)
}

/// The block of [`noise`]'s field of `shape` that starts at `at` and is
/// `size` long on each axis, row-major, in f32 on the host: the numbers
/// `noise(seed, shape, …)` has there, without drawing the rest.
///
/// [`SplitMix`] is a counter, so its `j`th draw is a function of `seed + j·γ`
/// alone, and element `k` of the field is the cosine or the sine of the
/// Box–Muller pair from draws `k − k mod 2 + 1` and `+ 2`. A tile of a large
/// canvas draws its own noise this way and still agrees with every other
/// tile where they overlap. Rows are shared among the machine's cores.
pub(crate) fn noise_block(seed: u64, shape: &[usize], at: &[usize], size: &[usize]) -> Vec<f32> {
    let rank = shape.len();
    let n: usize = size.iter().product();
    let mut out = vec![0f32; n];
    let row = *size.last().unwrap_or(&1);
    if n == 0 || row == 0 {
        return out;
    }
    // Where each of the block's rows starts in the field.
    let first = |r: usize| -> usize {
        let (mut rest, mut k, mut stride) = (r, 0, 1);
        for a in (0..rank).rev() {
            let i = if a == rank - 1 { at[a] } else {
                let i = rest % size[a];
                rest /= size[a];
                at[a] + i
            };
            k += i * stride;
            stride *= shape[a];
        }
        k
    };
    let draw = |j: u64| -> f64 { (SplitMix::mix(seed.wrapping_add(j.wrapping_mul(SplitMix::GAMMA))) >> 11) as f64 / (1u64 << 53) as f64 };
    let value = |k: usize| -> f32 {
        let p = (k / 2) as u64;
        let u1 = draw(2 * p + 1).max(f64::MIN_POSITIVE);
        let u2 = draw(2 * p + 2);
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        (if k % 2 == 0 { r * a.cos() } else { r * a.sin() }) as f32
    };
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(n / row);
    let rows_each = (n / row).div_ceil(threads.max(1));
    std::thread::scope(|sc| {
        for (c, part) in out.chunks_mut(rows_each * row).enumerate() {
            sc.spawn(move || {
                for (i, r) in part.chunks_mut(row).enumerate() {
                    let k0 = first(c * rows_each + i);
                    for (x, v) in r.iter_mut().enumerate() {
                        *v = value(k0 + x);
                    }
                }
            });
        }
    });
    out
}

/// SplitMix64: a 64-bit counter pushed through a mixing function. Tiny,
/// fast, and good enough that nothing about an image depends on its flaws.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(Self::GAMMA);
        Self::mix(self.0)
    }

    /// Uniform in `[0, 1)`, from the top 53 bits.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A convolution in bands of rows and a group norm a few groups at a
    /// time are the whole of each, to the bit: on the CPU and on Metal, in
    /// f32 and in f16, with budgets that cut at one row or group, at a few,
    /// and unevenly, and a batch of two.
    #[test]
    fn a_piece_at_a_time_is_the_whole() {
        let mut devices = vec![Device::Cpu];
        if let Ok(metal) = Device::new_metal(0) {
            devices.push(metal);
        }
        for dev in devices {
            for dtype in [DType::F32, DType::F16] {
                let draw = |seed: u64, shape: &[usize]| noise(seed, shape, &dev, DType::F32).unwrap().to_dtype(dtype).unwrap();
                let (n, c, out, h, w) = (2, 12, 5, 23, 17);
                let x = draw(1, &[n, c, h, w]);
                let same = |a: &Tensor, b: &Tensor, what: &str| {
                    let apart = (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap()).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
                    assert_eq!(a.dims(), b.dims(), "{what}");
                    assert!(apart == 0.0, "{what}, {dtype:?} on {:?}: {apart} apart", dev.location());
                };
                let row = n * c * 9 * w * dtype.size_in_bytes();
                let conv = Conv2d::from_parts(draw(2, &[out, c, 3, 3]), draw(3, &[out]), 1).unwrap();
                let whole = conv.banded(&x, usize::MAX).unwrap();
                for rows in [1, 4, 22] {
                    same(&conv.banded(&x, rows * row).unwrap(), &whole, &format!("a convolution {rows} rows at a time"));
                }
                let norm = GroupNorm { groups: 6, eps: 1e-6, w: draw(4, &[1, c, 1, 1]), b: draw(5, &[1, c, 1, 1]) };
                let whole = norm.grouped(&x, usize::MAX).unwrap();
                let group = n * (c / 6) * h * w * wide(dtype).size_in_bytes();
                for groups in [1, 4, 5] {
                    same(&norm.grouped(&x, groups * group).unwrap(), &whole, &format!("a norm {groups} groups at a time"));
                }
            }
        }
    }

    fn close(a: &Tensor, b: &Tensor, tol: f32) -> f32 {
        let d = (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap()).unwrap();
        let worst = d.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(worst < tol, "differ by {worst}");
        worst
    }

    /// A failed decode's zeros, and NaN or infinity anywhere, are refused,
    /// in the latent before the decode and in the pixels after it —
    /// in any chunk, not only the first — while a real image passes, black
    /// ones included, and comes out as the bytes the device's arithmetic made.
    #[test]
    fn blank_or_not_finite_latents_and_pixels_are_refused() {
        let dev = Device::Cpu;
        let (h, w) = (5, 7);
        let image = (Tensor::arange(0f32, (3 * h * w) as f32, &dev).unwrap().reshape((1, 3, h, w)).unwrap() * 0.29)
            .unwrap()
            .sin()
            .unwrap()
            * 1.3;
        let image = image.unwrap();
        let refused = |x: &Tensor| to_rgb8(x).err().map(|e| e.to_string()).unwrap_or_default();

        assert!(refused(&image.zeros_like().unwrap()).contains("zero everywhere"));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut v = image.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            v[3 * h * w - 1] = bad;
            let x = Tensor::from_vec(v, (1, 3, h, w), &dev).unwrap();
            assert!(refused(&x).contains("NaN or infinity"), "{bad} passed");
        }
        // Past the first chunk: a NaN there is found, and a zero first
        // chunk is not a zero image.
        let mut v = vec![0f32; 3000];
        assert!(check_decoded(&v).is_err());
        v[2999] = 0.5;
        assert!(check_decoded(&v).is_ok());
        v[2500] = f32::NAN;
        assert!(check_decoded(&v).is_err());

        // A latent part NaN and part finite: what the device's `max`, which
        // skips NaN, used to let through to the decode.
        let mut v = image.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        v[17] = f32::NAN;
        let latent = Tensor::from_vec(v, (1, 3, h, w), &dev).unwrap().to_dtype(DType::BF16).unwrap();
        assert!(check_latent(&latent).unwrap_err().to_string().contains("not decoding it"));
        assert!(check_latent(&latent.zeros_like().unwrap()).is_err());
        assert!(check_latent(&image.to_dtype(DType::BF16).unwrap()).is_ok());

        let black = Tensor::full(-1f32, (1, 3, h, w), &dev).unwrap();
        assert_eq!(to_rgb8(&black).unwrap().rgb, vec![0u8; 3 * h * w]);
        // What `to_rgb8` did on the device before it did it here.
        let x = ((image.clamp(-1f32, 1f32).unwrap() + 1.0).unwrap() * 127.5).unwrap();
        let x = x.squeeze(0).unwrap().permute((1, 2, 0)).unwrap().contiguous().unwrap();
        let want: Vec<u8> = x.flatten_all().unwrap().to_vec1::<f32>().unwrap().into_iter().map(|v| v.round() as u8).collect();
        let got = to_rgb8(&image).unwrap();
        assert_eq!((got.width, got.height), (w, h));
        assert_eq!(got.rgb, want);
    }

    /// A flat middle inside a frame of structure, as a decode that failed
    /// part-way draws it, is refused; a logo's plain ground with something
    /// in its middle passes, so does a plain colour with a little grain, and
    /// so does anything too small to judge.
    #[test]
    fn flat_images_are_refused() {
        let (w, h) = (512, 512);
        let mut rng = SplitMix(3);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                // As deep as the Wan VAE's frame from a failed `conv_in`.
                let edge = x.min(y).min(w - 1 - x).min(h - 1 - y) < 120;
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = if edge { (rng.next() % 256) as u8 } else { [126, 118, 109][c] };
                }
            }
        }
        let e = check_flat(&rgb, w, h).unwrap_err().to_string();
        assert!(e.contains("flat: one colour in"), "{e}");
        assert!(check_flat(&vec![200; w * h * 3], w, h).is_err());
        assert!(check_flat(&vec![200; 320 * 1024 * 3], 320, 1024).is_ok(), "too narrow to judge");

        // A logo: 238 grey everywhere but a 100² mark in the middle.
        let mut logo = vec![238u8; w * h * 3];
        for y in 206..306 {
            for x in 206..306 {
                logo[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[(x * 2) as u8, (y * 2) as u8, 40]);
            }
        }
        check_flat(&logo, w, h).unwrap();

        // A black with the grain a clean model leaves on a flat colour: one
        // step either way per channel, 27 colours, fewer than any real one
        // had, and still a picture.
        let grain: Vec<u8> = (0..w * h * 3).map(|_| 10 + (rng.next() % 3) as u8).collect();
        check_flat(&grain, w, h).unwrap();

        // Through `to_rgb8`, from what the last layer's bias alone makes.
        let bias = Tensor::new(&[0.1f32, -0.05, -0.2], &Device::Cpu).unwrap().reshape((1, 3, 1, 1)).unwrap();
        let x = bias.broadcast_as((1, 3, 384, 448)).unwrap().contiguous().unwrap();
        assert!(to_rgb8(&x).err().map(|e| e.to_string()).unwrap_or_default().contains("flat"));
    }

    /// A block of the field is the same numbers as the whole field has
    /// there: odd and even starts, a single element, the whole of it.
    #[test]
    fn noise_blocks_are_slices_of_the_field() {
        let shape = [5, 3, 7, 9];
        let whole = noise(11, &shape, &Device::Cpu, DType::F32).unwrap();
        for (at, size) in [([1, 0, 2, 3], [3, 3, 4, 5]), ([0, 1, 6, 8], [1, 1, 1, 1]), ([0, 0, 0, 0], [5, 3, 7, 9]), ([4, 2, 1, 0], [1, 1, 5, 9])] {
            let want = (0..4).fold(whole.clone(), |t, a| t.narrow(a, at[a], size[a]).unwrap());
            let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(noise_block(11, &shape, &at, &size), want, "at {at:?}, {size:?}");
        }
    }

    #[test]
    fn group_norm_leaves_each_group_with_zero_mean_and_unit_variance() {
        let dev = Device::Cpu;
        let x = (Tensor::arange(0f32, 2.0 * 8.0 * 3.0 * 3.0, &dev).unwrap().reshape((2, 8, 3, 3)).unwrap() * 0.37)
            .unwrap()
            .sin()
            .unwrap();
        let gn = GroupNorm {
            groups: 4,
            eps: 1e-6,
            w: Tensor::ones((1, 8, 1, 1), DType::F32, &dev).unwrap(),
            b: Tensor::zeros((1, 8, 1, 1), DType::F32, &dev).unwrap(),
        };
        let y = gn.forward(&x).unwrap().reshape((2, 4, 18)).unwrap();
        let mean = y.mean_keepdim(2).unwrap();
        let var = y.sqr().unwrap().mean_keepdim(2).unwrap();
        close(&mean, &Tensor::zeros((2, 4, 1), DType::F32, &dev).unwrap(), 1e-5);
        close(&var, &Tensor::ones((2, 4, 1), DType::F32, &dev).unwrap(), 1e-3);
    }

    /// A group norm's own backward, for a tensor being differentiated
    /// through frozen weights, is candle's through the norm written out,
    /// which is the path it takes when its weights are variables: the same
    /// answer and the same gradient for its input, on the CPU and on Metal.
    #[test]
    fn a_group_norms_backward_is_candles() {
        use candle_core::Var;
        for dev in [Some(Device::Cpu), Device::new_metal(0).ok()].into_iter().flatten() {
            let (x, r) = (noise(1, &[2, 8, 5, 3], &dev, DType::F32).unwrap(), noise(2, &[2, 8, 5, 3], &dev, DType::F32).unwrap());
            let (w, b) = (noise(3, &[1, 8, 1, 1], &dev, DType::F32).unwrap(), noise(4, &[1, 8, 1, 1], &dev, DType::F32).unwrap());
            let frozen = GroupNorm { groups: 4, eps: 1e-5, w: w.clone(), b: b.clone() };
            let live = GroupNorm { groups: 4, eps: 1e-5, w: Var::from_tensor(&w).unwrap().as_tensor().clone(), b: Var::from_tensor(&b).unwrap().as_tensor().clone() };
            let grad = |gn: &GroupNorm| {
                let var = Var::from_tensor(&x).unwrap();
                let y = gn.forward(var.as_tensor()).unwrap();
                ((&y * &r).unwrap().sum_all().unwrap().backward().unwrap().get(var.as_tensor()).expect("a gradient").clone(), y)
            };
            let ((got, y), (want, y_want)) = (grad(&frozen), grad(&live));
            close(&y, &y_want, 1e-5);
            close(&got, &want, 1e-4);
        }
    }

    /// The fused kernel and the written-out slices are the same function on
    /// every shape the pipelines here ask for: self-attention on a latent
    /// grid, cross-attention to 77 prompt tokens, a single-head VAE attention
    /// at 512 wide — including lengths that are not a multiple of any tile.
    #[test]
    fn the_fused_kernel_agrees_with_the_written_out_one_without_a_mask() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no Metal device; nothing to compare");
            return;
        };
        // f16 is SDXL's; f32 is what a quantised model's activations are.
        let shapes = [(256, 256, 5, 64), (1024, 77, 10, 64), (100, 77, 20, 64), (64, 64, 1, 512), (300, 45, 24, 128)];
        let f32s = [(300, 300, 24, 128), (1047, 1047, 24, 128)];
        for (dtype, (lq, lk, heads, d)) in shapes.map(|s| (DType::F16, s)).into_iter().chain(f32s.map(|s| (DType::F32, s))) {
            let mk = |l: usize, s: u64| noise(s, &[1, l, heads * d], &dev, dtype).unwrap();
            let (q, k, v) = (mk(lq, 1), mk(lk, 2), mk(lk, 3));
            let fused = attention(&q, &k, &v, heads).unwrap();
            let split = |x: &Tensor, l: usize| x.reshape((1, l, heads, d)).unwrap().transpose(1, 2).unwrap().contiguous().unwrap();
            let slow = written_out(&split(&q, lq), &split(&k, lk), &split(&v, lk), 1.0 / (d as f64).sqrt())
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((1, lq, heads * d))
                .unwrap();
            close(&fused, &slow, 5e-3);
        }
    }

    #[test]
    fn the_written_out_path_is_the_same_in_slices_as_whole() {
        let dev = Device::Cpu;
        let mk = |s: u64, l: usize| noise(s, &[1, 2, l, 8], &dev, DType::F32).unwrap();
        let (q, k, v) = (mk(1, 50), mk(2, 30), mk(3, 30));
        let whole = {
            let att = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * 0.35).unwrap();
            crate::grad::softmax_last_dim(&att).unwrap().matmul(&v).unwrap()
        };
        // Force slicing by making every row its own slice.
        let mut parts = Vec::new();
        for i in 0..50 {
            parts.push(written_out(&q.narrow(2, i, 1).unwrap(), &k, &v, 0.35).unwrap());
        }
        close(&Tensor::cat(&parts, 2).unwrap(), &whole, 1e-5);
        close(&written_out(&q, &k, &v, 0.35).unwrap(), &whole, 1e-5);
    }

    #[test]
    fn noise_is_standard_normal_and_the_same_for_the_same_seed() {
        let a = noise(42, &[4, 64, 64], &Device::Cpu, DType::F32).unwrap();
        let b = noise(42, &[4, 64, 64], &Device::Cpu, DType::F32).unwrap();
        close(&a, &b, 0.0 + f32::MIN_POSITIVE);
        let v = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.03, "mean {mean}, var {var}");
    }

    #[test]
    fn the_timestep_embedding_puts_cosines_first_when_asked() {
        let e = timestep_embedding(&[0.0, 10.0], 8, true, 0.0, &Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
        // At t = 0 every cosine is 1 and every sine 0.
        assert_eq!(e[0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // The first frequency is 1: cos(10) and sin(10).
        assert!((e[1][0] - 10f32.cos()).abs() < 1e-6 && (e[1][4] - 10f32.sin()).abs() < 1e-6);
    }
}

