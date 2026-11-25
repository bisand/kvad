//! What a block does after it has attended, on the GPU: a dense MLP, or a
//! mixture of them.
//!
//! The mirror of [`kvad::model::ffn`], and separate from either backend for the
//! same reason it is there. A block is two independent choices — how it
//! attends, and what it feeds the result through — and only the first of them
//! is what makes DeepSeek DeepSeek and Llama Llama. `deepseek.rs` wrote this
//! first and had it to itself; `qwen3_moe` is the Llama block with the other
//! answer on the second axis, and reaching it from `model.rs` should not mean
//! importing latent attention along with it.
//!
//! # The mixture is grouped by expert, not by token
//!
//! Routing is per token, so a batch of prompt tokens generally touches most of
//! the experts however you slice it. Walking tokens would read each chosen
//! expert's weights again for every token that chose it; walking experts reads
//! each one once, over exactly the rows that picked it, and scatters the
//! weighted result back.
//!
//! An expert nobody picked is skipped entirely, which is where the saving
//! actually comes from at decode: one token reaches `top_k` experts, so eight
//! of a hundred and twenty-eight run and the rest are never touched.
//!
//! # The routing itself runs on the host
//!
//! [`Router::route`] is the CPU engine's own function, called here on logits
//! read back from the device. It is a partial selection over a `[m, n_experts]`
//! row — microseconds against the matmuls either side of it — and running it
//! there means group-limited selection, V3's bias-steered choice and Qwen3's
//! renormalised top-k have one implementation rather than two that must be
//! kept level with each other.

use crate::common::{Loader, Proj, Reader, Stored};
use crate::fused;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Tensor};
use candle_nn::ops;
use kvad::model::ffn::{Router, Shared};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A SwiGLU MLP: a dense layer, a shared expert, or one routed expert.
pub(crate) struct Ffn {
    /// The gate and up projections as one, the gate's outputs first: see
    /// [`Loader::proj_cat`].
    gate_up: Proj,
    down: Proj,
    width: usize,
}

impl Ffn {
    /// `width` is the MLP's inner dimension, which for an expert is
    /// `moe_intermediate_size` and not the dense layers' `intermediate_size`.
    /// Qwen3-30B-A3B's are 768 and 6144, so getting this wrong is a shape
    /// error rather than a silent one — but only because candle is told the
    /// shape it expects, which is the whole reason these widths are passed in.
    pub(crate) fn load(ld: &Loader, vb: &Reader<'_>, e: usize, width: usize) -> Res<Self> {
        let parts = [("gate_proj.weight", width), ("up_proj.weight", width)];
        Ok(Ffn {
            gate_up: ld.proj_cat(vb, &parts, e, Stored::OutIn)?,
            down: ld.proj(vb, "down_proj.weight", e, width, Stored::OutIn)?,
            width,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.down.forward(&fused::silu_mul(&self.gate_up.forward(x)?, self.width)?)
    }

    pub(crate) fn params(&self) -> usize {
        self.gate_up.params() + self.down.params()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.gate_up.bytes() + self.down.bytes()
    }
}

/// The shared expert, and whatever decides how much of it counts.
///
/// The mirror of [`kvad::model::ffn::SharedExpert`]: DeepSeek adds its shared
/// experts whole, Qwen3-Next scales its one by `sigmoid(x · w)` first.
pub(crate) struct SharedExpert {
    ffn: Ffn,
    /// `[1, hidden]`, and `None` for the family that has no such thing.
    gate: Option<Proj>,
}

impl SharedExpert {
    fn load(ld: &Loader, vb: &Reader<'_>, e: usize, width: usize, shared: Shared) -> Res<Option<Self>> {
        let name = match shared {
            Shared::None => return Ok(None),
            // Stored fused: `n` experts side by side in one matrix, so the
            // width is `n` times an expert's.
            Shared::Fused(_) => "shared_experts",
            Shared::Gated(_) => "shared_expert",
        };
        Ok(Some(SharedExpert {
            ffn: Ffn::load(ld, &vb.pp(name), e, shared.width(width))?,
            gate: match shared {
                Shared::Gated(_) => Some(ld.proj(vb, "shared_expert_gate.weight", 1, e, Stored::OutIn)?),
                _ => None,
            },
        }))
    }

    /// Every token, so no grouping — and a convenient accumulator for the
    /// routed experts to add into.
    fn forward(&self, h: &Tensor) -> Res<Tensor> {
        let out = self.ffn.forward(h)?;
        match &self.gate {
            // `[m, 1]`, broadcast back over the width it scales.
            Some(gate) => Ok(out.broadcast_mul(&ops::sigmoid(&gate.forward(h)?)?)?),
            None => Ok(out),
        }
    }

    fn params(&self) -> usize {
        self.ffn.params() + self.gate.as_ref().map_or(0, Proj::params)
    }

    fn bytes(&self) -> usize {
        self.ffn.bytes() + self.gate.as_ref().map_or(0, Proj::bytes)
    }
}

pub(crate) struct Moe {
    /// How this family picks. A clone per routed layer, which is a few dozen
    /// bytes against the layer's experts.
    router: Router,
    /// `[n_experts, hidden]`: one dot product per expert.
    gate: Proj,
    /// V3's learned balancing bias, which steers the choice and not the
    /// weights. Kept on the host: it is read by the router, which runs there.
    bias: Option<Vec<f32>>,
    experts: Vec<Ffn>,
    /// Runs for every token, whatever the router says. DeepSeek has one and
    /// Qwen3-Next has one with a gate; Qwen3's own mixture does not.
    shared: Option<SharedExpert>,
}

impl Moe {
    pub(crate) fn load(
        ld: &Loader,
        vb: &Reader<'_>,
        e: usize,
        width: usize,
        router: &Router,
        shared: Shared,
    ) -> Res<Self> {
        Ok(Moe {
            router: router.clone(),
            gate: ld.proj(vb, "gate.weight", router.n_experts, e, Stored::OutIn)?,
            bias: vb
                .try_get(router.n_experts, "gate.e_score_correction_bias")
                .map(|t| t.to_dtype(DType::F32)?.to_vec1::<f32>())
                .transpose()?,
            experts: (0..router.n_experts)
                .map(|x| Ffn::load(ld, &vb.pp(format!("experts.{x}")), e, width))
                .collect::<Res<Vec<_>>>()?,
            shared: SharedExpert::load(ld, vb, e, width, shared)?,
        })
    }

    /// The mixture, grouped by expert.
    ///
    /// Route on the host, invert the per-token picks into a row list per
    /// expert, then, for each expert anybody picked: gather its rows, run it
    /// once over them, scale each row by that token's weight, and add the
    /// result back where it came from. `index_add` is the scatter, and it is
    /// an add rather than a write because a token's output is the *sum* over
    /// its experts.
    ///
    /// The device and the dtype come from `h` rather than from a field. They
    /// are the activations' own, which is the only answer that can be right,
    /// and one fewer thing for a loader to pass in wrongly.
    pub(crate) fn forward(&self, h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
        let logits = self.gate.forward(h)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let (device, dtype) = (h.device(), h.dtype());

        let mut by_expert: Vec<(Vec<u32>, Vec<f32>)> =
            vec![(Vec::new(), Vec::new()); self.router.n_experts];
        let mut picks = Vec::with_capacity(self.router.top_k);
        for (i, row) in logits.iter().enumerate() {
            self.router.route(row, self.bias.as_deref(), &mut picks);
            for &(expert, w) in &picks {
                by_expert[expert].0.push(i as u32);
                by_expert[expert].1.push(w);
            }
        }

        // The shared experts run for every token whatever the router says, so
        // they need no grouping and make a convenient accumulator.
        let mut out = match &self.shared {
            Some(shared) => shared.forward(h)?,
            None => Tensor::zeros((m, e), dtype, device)?,
        };
        for (i, (rows, weights)) in by_expert.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let k = rows.len();
            let idx = Tensor::from_slice(rows, (k,), device)?;
            let xs = h.index_select(&idx, 0)?;
            let w = Tensor::from_slice(weights, (k, 1), device)?.to_dtype(dtype)?;
            let y = self.experts[i].forward(&xs)?.broadcast_mul(&w)?;
            out = out.index_add(&idx, &y, 0)?;
        }
        Ok(out)
    }

    pub(crate) fn params(&self) -> usize {
        self.gate.params()
            + self.bias.as_ref().map_or(0, |v| v.len())
            + self.experts.iter().map(Ffn::params).sum::<usize>()
            + self.shared.as_ref().map_or(0, SharedExpert::params)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.gate.bytes()
            + self.experts.iter().map(Ffn::bytes).sum::<usize>()
            + self.shared.as_ref().map_or(0, SharedExpert::bytes)
    }
}

/// A mixture layer that stays off the device: the CPU engine's own, reading
/// its experts through the expert cache, so that the GPU holds attention and
/// the dense layers and a model far larger than its memory can still run.
///
/// A prompt pass is different: many rows per expert is what the device is good
/// at, so from [`ON_DEVICE_ROWS`] rows the experts are lent to it, read through
/// the same cache and dropped again once they have run.
///
/// # Where a decode token goes, and what could overlap
///
/// Qwen3-30B-A3B at q4 on an M5 Pro, experts page-cached, timed inside
/// [`Hosted::forward`] with the wait and the copy back split apart (µs, one
/// layer of 48):
///
/// ```text
/// encoding the next attention        ~75
/// waiting for the device            ~350   attention, and a round trip
/// copying the row back              ~155   a second round trip
/// the CPU's experts                 ~660   ~170 fixed, ~60 an expert
/// copying the result up               ~8
/// ```
///
/// Untimed, the copy back shares the wait before it, so the profiled run
/// reads 16.8 tok/s and the real one 19.6.
///
/// Nothing in one token of this model can run beside anything else. The next
/// layer's attention reads this layer's output, and Qwen3's mixture has no
/// shared expert, so the only work the device can take from the CPU is some of
/// the experts themselves, which needs them on the device. Six interleaved
/// rounds each, medians:
///
/// - **Keeping both processors busy** instead, a CPU thread spinning while the
///   device works ([`kvad::model::linger`]) and [`crate::model::stage`]'s
///   warming kernel while the CPU does: 18.3 tok/s against 19.6. The gaps are
///   a millisecond or less, too short to go cold in, and the spinners take a
///   core and bandwidth from the work.
/// - **A hot set on the device**, Strata's way, is [`Hot`]:
///   `KVAD_GPU_HOT_EXPERTS` experts of each layer, the ones this session's
///   first [`HOT_AFTER`] tokens picked most, lent once and kept. With a
///   102-token prompt choosing them:
///
/// ```text
/// a layer   on the device   decode        peak footprint
///      0        0 GB        19.9 tok/s       3.15 GB
///     13      1.6 GB        22.0 (+11%)      3.15 GB
///     25      3.2 GB        23.7 (+19%)      4.50 GB
///     38      4.8 GB        24.6 (+24%)      6.16 GB
/// ```
///
///   At 25 a layer, the CPU was left 0–2 of a token's 8 picks. The device does
///   not run them beside the CPU's: candle commits only when it waits, so they
///   run before the next attention. They pay anyway, because an expert costs
///   the device far less than the CPU, once a row needs no gather, scatter or
///   upload ([`add_expert`]'s one-row path; without it, 38 a layer was +5%).
///
///   An earlier version of this note put the ceiling at +21% for 3.5 GB, from
///   running five experts of eight on the CPU and none on the device. The
///   set chosen on its own conversation catches more than that: from another
///   conversation's counts (`expert_residency --pin-from`, a story against
///   Rust) a set catches only 17%, 35% and 47% of the picks at 1.6, 3.2 and
///   4.8 GB, and routing is too even for better (see [`kvad::residency`]).
///   The set is chosen once per load, so a server whose conversations wander
///   should expect less than the table. Kept by LRU it would follow them, but
///   only by lending 0.3–0.6 GB of misses to the device every token.
///
/// shortcut: every layer waits for the device and copies its row both ways,
/// which nothing in one token can hide; the hot set is chosen once and never
/// revised, upgrade by re-choosing it when a conversation's picks drift.
pub(crate) struct Hosted {
    mlp: kvad::model::ffn::Mlp,
    hot: std::sync::Mutex<Hot>,
}

/// The experts of one layer kept on the device: see [`hot_experts`].
#[derive(Default)]
pub(crate) struct Hot {
    /// How often each expert was picked, until the set is chosen.
    counts: Vec<u64>,
    /// Rows routed so far.
    rows: usize,
    /// `None` until chosen; then one entry an expert, `Some` for those kept.
    pub(crate) kept: Option<Vec<Option<Ffn>>>,
}

impl Hosted {
    pub(crate) fn load(src: &dyn kvad::qcache::Source, vb: &Reader<'_>, prefix: &str, router: &Router, shared: Shared) -> Res<Self> {
        // Everything under `mlp` is the CPU engine's now, and the guard
        // against unread weights should hear that from here.
        for part in ["gate", "experts", "shared_expert"] {
            vb.skip_under(part);
        }
        let moe = kvad::model::ffn::Moe::load(src, prefix, router, shared)?;
        Ok(Hosted { mlp: kvad::model::ffn::Mlp::Moe(Box::new(moe)), hot: Default::default() })
    }

    fn forward(&self, h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
        let xs = h.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let kvad::model::ffn::Mlp::Moe(moe) = &self.mlp else { return self.on_host(&xs, h, m, e) };
        let lendable = moe.shared.is_none() && lendable(&moe.experts[0]);
        let prompt = m >= ON_DEVICE_ROWS && lendable && prompt_on_device();
        let keep = if lendable { hot_experts() } else { 0 };
        if !prompt && keep == 0 {
            return self.on_host(&xs, h, m, e);
        }
        let by_expert = kvad::model::in_shared_pool(|| moe.routed(&xs, m));
        let mut hot = self.hot.lock().unwrap_or_else(|p| p.into_inner());
        if keep > 0 {
            hot.count(moe, &by_expert, m, keep, h.device());
        }
        if prompt {
            return on_device(moe, &by_expert, h, m, e);
        }
        mixed(moe, hot.kept.as_deref().unwrap_or(&[]), &by_expert, &xs, h, m, e)
    }

    /// Every expert on the CPU, as the engine runs them.
    fn on_host(&self, xs: &[f32], h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
        let ys = kvad::model::in_shared_pool(|| self.mlp.run(xs, m, e, None));
        Ok(Tensor::from_vec(ys, (m, e), h.device())?.to_dtype(h.dtype())?)
    }
}

impl Hot {
    /// Count this pass's picks, and once [`HOT_AFTER`] rows have been routed,
    /// lend the `keep` busiest experts to the device for good.
    pub(crate) fn count(&mut self, moe: &kvad::model::ffn::Moe, by_expert: &[Vec<(usize, f32)>], m: usize, keep: usize, device: &candle_core::Device) {
        if self.kept.is_some() {
            return;
        }
        self.counts.resize(by_expert.len(), 0);
        for (c, picks) in self.counts.iter_mut().zip(by_expert) {
            *c += picks.len() as u64;
        }
        self.rows += m;
        if self.rows < HOT_AFTER {
            return;
        }
        let mut busiest: Vec<usize> = (0..self.counts.len()).collect();
        busiest.sort_by_key(|&x| std::cmp::Reverse(self.counts[x]));
        busiest.truncate(keep);
        busiest.sort_unstable();
        let mut kept: Vec<Option<Ffn>> = (0..self.counts.len()).map(|_| None).collect();
        let mut failed = None;
        moe.with_experts(&busiest, |x, ffn| match lent(ffn, device) {
            Ok(on_device) => kept[x] = Some(on_device),
            Err(err) => failed = Some(err),
        });
        if let Some(err) = failed {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| eprintln!("hot experts: some stay on the host: {err}"));
        }
        self.kept = Some(kept);
    }
}

/// The experts kept on the device run there, the rest on the CPU, and the
/// two added.
pub(crate) fn mixed(moe: &kvad::model::ffn::Moe, kept: &[Option<Ffn>], by_expert: &[Vec<(usize, f32)>], xs: &[f32], h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
    let chosen: Vec<usize> = (0..by_expert.len()).filter(|&x| !by_expert[x].is_empty()).collect();
    let on_cpu: Vec<usize> = chosen.iter().copied().filter(|&x| kept.get(x).is_none_or(Option::is_none)).collect();
    let ys = kvad::model::in_shared_pool(|| moe.run_some(xs, m, e, by_expert, &on_cpu));
    let mut out = Tensor::from_vec(ys, (m, e), h.device())?.to_dtype(h.dtype())?;
    for &x in &chosen {
        if let Some(Some(ffn)) = kept.get(x) {
            out = add_expert(out, ffn, &by_expert[x], h)?;
        }
    }
    Ok(out)
}

/// Rows a layer routes before its hot set is chosen from what they picked.
const HOT_AFTER: usize = 64;

/// `KVAD_GPU_HOT_EXPERTS`: how many experts of each hosted layer to keep on
/// the device, chosen by how often this session's first [`HOT_AFTER`] tokens
/// picked them. q4 only, like lending; 0 or unset keeps them all on the host.
fn hot_experts() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("KVAD_GPU_HOT_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// `out` plus one expert run on the device over the rows that picked it,
/// each scaled by its weight.
fn add_expert(out: Tensor, ffn: &Ffn, picks: &[(usize, f32)], h: &Tensor) -> Res<Tensor> {
    let (device, dtype) = (h.device(), h.dtype());
    // One row, as at decode: no rows to gather or scatter, and the weight is
    // a scalar rather than an upload.
    if let [(0, w)] = picks {
        if h.dim(0)? == 1 {
            return Ok((out + ffn.forward(h)?.to_dtype(dtype)?.affine(f64::from(*w), 0.0)?)?);
        }
    }
    let (rows, weights): (Vec<u32>, Vec<f32>) = picks.iter().map(|&(i, w)| (i as u32, w)).unzip();
    let k = rows.len();
    let idx = Tensor::from_vec(rows, (k,), device)?;
    let w = Tensor::from_vec(weights, (k, 1), device)?.to_dtype(dtype)?;
    let y = ffn.forward(&h.index_select(&idx, 0)?)?.to_dtype(dtype)?.broadcast_mul(&w)?;
    Ok(out.index_add(&idx, &y, 0)?)
}

/// The fewest rows in a pass for its experts to go to the device.
const ON_DEVICE_ROWS: usize = 32;

/// Buffers a dropped expert holds until the device is waited for, at most.
const LENT_BYTES: usize = 256 << 20;

/// On unless `KVAD_GPU_PROMPT_EXPERTS=0`, which is there for the A/B.
fn prompt_on_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("KVAD_GPU_PROMPT_EXPERTS").as_deref() != Ok("0"))
}

/// Run each chosen expert on the device, lent for this pass alone.
pub(crate) fn on_device(moe: &kvad::model::ffn::Moe, by_expert: &[Vec<(usize, f32)>], h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
    let chosen: Vec<usize> = (0..by_expert.len()).filter(|&x| !by_expert[x].is_empty()).collect();
    let device = h.device();
    let mut out = Some(Tensor::zeros((m, e), h.dtype(), device)?);
    let (mut held, mut failed) = (0, None);
    moe.with_experts(&chosen, |x, ffn| {
        if failed.is_some() {
            return;
        }
        let mut step = || -> Res<()> {
            let ffn_on_device = lent(ffn, device)?;
            out = Some(add_expert(out.take().expect("kept between experts"), &ffn_on_device, &by_expert[x], h)?);
            held += ffn_on_device.bytes();
            if held >= LENT_BYTES {
                drop(ffn_on_device);
                device.synchronize()?;
                held = 0;
            }
            Ok(())
        };
        failed = step().err();
    });
    match failed {
        Some(err) => Err(err),
        None => Ok(out.expect("kept between experts")),
    }
}

/// Whether an expert can be lent to the device. q4 only: on Qwen3-30B under
/// 8 GB a 1132-token prompt took 14.6 s against 20.5 at q4, but 32.8 against
/// 23.3 at q8, three rounds each.
fn lendable(ffn: &kvad::model::ffn::Ffn) -> bool {
    ffn.weights().iter().all(|w| matches!(w.parts(), kvad::quant::Parts::Q4 { .. }))
}

/// A CPU expert as device matrices.
fn lent(ffn: &kvad::model::ffn::Ffn, device: &candle_core::Device) -> Res<Ffn> {
    let [gate, up, down] = ffn.weights();
    let mut gate_up = blocks(gate).ok_or("an expert is not in q4 blocks")?;
    gate_up.extend(blocks(up).ok_or("an expert is not in q4 blocks")?);
    let down_blocks = blocks(down).ok_or("an expert is not in q4 blocks")?;
    Ok(Ffn {
        gate_up: device_proj(&gate_up, gate.rows() + up.rows(), gate.cols(), device)?,
        down: device_proj(&down_blocks, down.rows(), down.cols(), device)?,
        width: gate.rows(),
    })
}

fn device_proj(bytes: &[u8], out: usize, inp: usize, device: &candle_core::Device) -> Res<Proj> {
    let q = QTensor::new(QStorage::from_data(std::borrow::Cow::Borrowed(bytes), device, GgmlDType::Q4_0)?, (out, inp))?;
    Ok(Proj::Quant(QMatMul::from_qtensor(q)?))
}

/// `x` as IEEE half precision, rounded to nearest, ties to even.
fn to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let man = bits & 0x7f_ffff;
    let round = |v: u32, rem: u32, half: u32| v + u32::from(rem > half || (rem == half && v & 1 == 1));
    match exp {
        128 => sign | 0x7c00 | (u16::from(man != 0) << 9),
        16.. => sign | 0x7c00,
        -14.. => sign | round(((exp + 15) as u32) << 10 | man >> 13, man & 0x1fff, 0x1000) as u16,
        -25.. => {
            let (full, shift) = (man | 0x80_0000, (-exp - 1) as u32);
            sign | round(full >> shift, full & ((1 << shift) - 1), 1 << (shift - 1)) as u16
        }
        _ => sign,
    }
}

/// The CPU engine's q4 arrays as candle's blocks: its f32 scale narrowed to
/// f16, the codes as they are, since both split a block's nibbles the same way.
fn blocks(w: &kvad::quant::Weight) -> Option<Vec<u8>> {
    use kvad::quant::{Parts, BLOCK};
    let Parts::Q4 { scales, qs } = w.parts() else { return None };
    let mut out = Vec::with_capacity(scales.len() * (2 + BLOCK / 2));
    for (&s, q) in scales.iter().zip(qs.chunks_exact(BLOCK / 2)) {
        out.extend(to_f16(s).to_le_bytes());
        out.extend_from_slice(q);
    }
    Some(out)
}

pub(crate) enum Mlp {
    Dense(Ffn),
    /// Boxed, as on the CPU: an unboxed mixture makes every variant as wide as
    /// a hundred experts' worth of bookkeeping, so a dense block pays for the
    /// mixture it has not got.
    Moe(Box<Moe>),
    Hosted(Box<Hosted>),
}

impl Mlp {
    pub(crate) fn forward(&self, h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
        match self {
            Mlp::Dense(f) => Ok(f.forward(h)?),
            Mlp::Moe(moe) => moe.forward(h, m, e),
            Mlp::Hosted(moe) => moe.forward(h, m, e),
        }
    }

    pub(crate) fn params(&self) -> usize {
        match self {
            Mlp::Dense(f) => f.params(),
            Mlp::Moe(moe) => moe.params(),
            Mlp::Hosted(moe) => moe.mlp.param_count(),
        }
    }

    /// On the device, which a hosted layer is not.
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Mlp::Dense(f) => f.bytes(),
            Mlp::Moe(moe) => moe.bytes(),
            Mlp::Hosted(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_f16_rounds_as_candles_cast_does() {
        let mut xs = vec![0.0f32, -0.0, 1.0, -1.5, 65504.0, 65520.0, 1e6, 6.1e-5, 3e-8, 2.9e-8, 1e-12, f32::INFINITY];
        let mut v = 0x1234_5678u32;
        for _ in 0..20_000 {
            v = v.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = f32::from_bits(v);
            if x.is_finite() {
                xs.push(x);
            }
        }
        let bytes: Vec<u8> = xs.iter().flat_map(|&x| to_f16(x).to_le_bytes()).collect();
        let d = candle_core::Device::Cpu;
        let ours = Tensor::from_raw_buffer(&bytes, DType::F16, &[xs.len()], &d).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        let theirs = Tensor::from_slice(&xs, xs.len(), &d).unwrap().to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        for ((a, b), x) in ours.iter().zip(&theirs).zip(&xs) {
            assert_eq!(a.to_bits(), b.to_bits(), "{x}");
        }
    }
}
