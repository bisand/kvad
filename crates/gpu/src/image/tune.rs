//! What one step of LoRA training costs: the measurement #74 asks for
//! before anything promises a model or a size.
//!
//! A step is what `kvad tune` (#75) will do many of: the denoiser run
//! forward on a noised latent with a LoRA on its attention layers, a loss
//! against the noise that was added, `backward`, and AdamW on the LoRA's
//! factors. Nothing here learns anything; the latents, the text and the
//! noise are all drawn at random, because the cost does not depend on what
//! they are.
//!
//! Only the denoiser is loaded. Training encodes each picture and each
//! caption once, before it starts, and needs neither the VAE nor the text
//! encoders in memory while it runs.
//!
//! **What makes it expensive.** Inference keeps one activation at a time.
//! `backward` needs all of them, every tensor the forward pass made, until
//! it has walked back past each. Three things in `crate::grad` bring that
//! down to something a machine can hold:
//!
//! - a frozen layer hands `backward` its input's gradient and nothing for
//!   its own weights (`attach`). Without it candle makes a gradient for all
//!   2.6 B of them at every step: 31.8 GB at 256×256, where this is 14.4;
//! - the model is walked back through one stretch at a time
//!   (`checkpointed`), so one stage's record is held and not all of them;
//! - and the first of the two forward passes that costs is unrecorded, so
//!   it runs on the kernels that draw.
//!
//! [`Rig::gradients`] can also do it whole, the way that does not fit, for
//! the two to be compared where both do.
//!
//! # What was measured
//!
//! SDXL's UNet in f16, 2.57 B frozen parameters; a rank-16 LoRA on its 560
//! attention projections, 23.2 M numbers, in f32; one picture a step; an
//! M5 Pro with 48 GB. Peak footprint by `/usr/bin/time -l` on
//! `examples/tune_budget`, seconds from the second step on:
//!
//! | | a step | peak | one forward pass, drawing |
//! |---|---|---|---|
//! | 256², the whole UNet recorded, as first written | 5.2 s | 31.8 GB | 0.10 s |
//! | 256², the whole UNet recorded, frozen layers attached | 3.4 s | 14.4 GB | |
//! | 480², 22 stretches, a stage each | 3.1 s | 13.8 GB | 0.33 s |
//! | 480², 103 stretches, a transformer block each | 3.4 s | 7.4 GB | |
//! | 480², attention with a backward of its own | 3.2 s | 7.0 GB | |
//! | 512², the same | 3.2 s | 7.2 GB | 0.35 s |
//! | 1024², the same | 12.4 s | 10.8 GB | 1.38 s |
//! | 512², norms attached, one AdamW, see below | 2.8 s | 7.1 GB | 0.35 s |
//! | 768² | 5.6 s | 8.7 GB | 0.78 s |
//! | 1024² | 10.3 s | 10.5 GB | 1.39 s |
//! | 512², see "what took it from 10.3 s" | 2.1 s | 7.0 GB | 0.36 s |
//! | 768² | 4.6 s | 8.2 GB | 0.80 s |
//! | 1024² | 8.4 s | 10.3 GB | 1.43 s |
//! | 512², frozen layers' way back on the matrix units | 1.9 s | 7.0 GB | 0.36 s |
//! | 768² | 4.2 s | 8.2 GB | 0.81 s |
//! | 1024² | 7.7 s | 10.3 GB | 1.43 s |
//!
//! Recorded whole, 512² took the machine down. 5.1 GB of every figure is
//! the weights.
//!
//! - **What the stage-sized stretches held** was not activations, which
//!   are megabytes. It was `backward`'s small leavings in large buffers
//!   (`crate::grad::rehomed`), each stretch's record alive through the
//!   next one's, and ten transformer blocks' worth of both at once.
//! - **What a block then held was its attention's scores**, a number for
//!   every pair of tokens and every head: 671 MB in f32 at 1024², where
//!   the upper attention level is 4096 tokens, kept at every step of the
//!   softmax and again for each one's gradient. Before
//!   `crate::grad::attended`, 1024² was 23.4 GB and 20.7 s, 17 GB of it
//!   one block; it now makes the scores again on the way back, a batch of
//!   rows at a time.
//! - **A step is six forward passes.** At 1024², of 10.1 s as it stood
//!   before the last row: the unrecorded pass 1.8 s; each stretch run
//!   again, recorded, 2.3; and `backward` 6.0. Found by leaving each out
//!   in turn and timing the step, not by timing each in place: a clock on
//!   one piece has to wait for the GPU before and after, a wait lets every
//!   dropped buffer go, and the piece then pays for fresh ones, which is
//!   most of what it measures (below). Of `backward`'s 6.0, attention's
//!   own was 1.9, the frozen linear layers' 0.4, the norms' 0.2, the
//!   convolutions' and the rest of the hand-written ones' 2.2, and candle's
//!   own operations and bookkeeping 1.3.
//! - **What took it from 12.4 s to 10.3**: attention's backward no longer
//!   copies its scores to transpose them (1.2 s); the norms have a backward
//!   of their own, one operation in the record where there were ten
//!   (`crate::grad::norm_back`, 0.2 s); and the optimiser is one AdamW
//!   over the factors end to end (`crate::adam`), 0.06 s where candle's
//!   took 0.34 at any size.
//! - **What took it from 10.3 s to 8.4**, and 512² from 2.8 to 2.1:
//!   - *Attention's backward writes four tensors the size of the scores
//!     where it wrote eight* (`crate::grad::attention_back`): 104 ms to 48
//!     for a block at 4096 tokens, 14 to 6 at 1024. Writing 671 MB is what
//!     costs there, not the products that read it.
//!   - *The device is asked to let go only where the state changes shape*,
//!     not after every stretch. A buffer candle has let go is made afresh
//!     when next wanted, and a fresh buffer is slow to write the first
//!     time: one product into a new 671 MB takes 107 ms, and four into
//!     the same one 122 between them. A transformer's ten blocks want the
//!     same buffers one after another. What is kept across them moves out
//!     of the buffers it was born in all the same (`crate::grad::rehomed`),
//!     or sixteen small gradients a block hold sixteen large buffers:
//!     1.7 GB at 1024² when that was left out.
//!   - *A wide convolution's backward reads its kernel as it is stored*
//!     (`super::nn::back_folded`). Turning a 1280-channel kernel round was
//!     a 29 MB copy, 38 ms where the convolution is 3.5, every step.
//! - **What took it from 8.4 s to 7.7**, and 512² from 2.1 to 1.9: a
//!   frozen linear layer's gradient goes back on the M5's matrix units,
//!   as its answer came forward (`crate::mpp::dense_turned`, which reads
//!   the weight as it lies and takes its transpose). On candle's product
//!   it was 5–7 times slower than the layer's forward pass: 11 ms where
//!   that is 2, for a feed-forward's 1280 to 10 240 over 1024 tokens.
//!   (At 1024² the two builds were raced on a busy machine, 9.1 s against
//!   8.3 at best; 7.7 is a run alone.)
//! - **The recorded pass is a forward pass and little more**: run with
//!   nothing tracked it is 1.75 s at 1024², and recorded, 2.0.
//! - **What it is not.** An earlier note here put 3.4 s on candle's making
//!   a zeroed gradient for every operation in the record. Measured, an
//!   operation that size is 4 µs and with its zeroed gradient 10 to 18:
//!   half a second for every operation in a step at the most.
//!
//! So what #75 can promise from this, on this machine: SDXL's own 1024² in
//! 10.3 GB at under 8 s a step, or 512² in 7 GB at 1.9. (The last six
//! rows are the least of twelve steps, on a machine that was in use.)

use super::dataset::Seen;
use super::lora::Adapters;
use super::nn::{noise, uniform, Ctx, SplitMix};
use super::schedule::Noising;
use super::sdxl::{weights, PREFIXES};
use super::unet::{Unet, UnetConfig};
use super::{finish, open, read_json};
use crate::common::{pooled, settle, Loader};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor, Var};
use candle_core::backprop::GradStore;
use candle_nn::ParamsAdamW;
use kvad::weights::{fetch_file, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What [`sdxl`] measured.
pub struct Budget {
    /// The layers the LoRA is on, and the numbers it trains.
    pub layers: usize,
    pub trained: usize,
    /// The frozen UNet's parameters.
    pub frozen: usize,
    /// Each step's seconds, the first included, which also compiles the
    /// kernels and fills the buffer pool.
    pub secs: Vec<f64>,
    /// Of each step's seconds, those in the loss and its gradients, and
    /// those in the optimiser.
    pub parts: Vec<[f64; 2]>,
    /// Each step's loss, to see that it is a number.
    pub loss: Vec<f32>,
    /// Seconds of one forward pass alone with nothing tracked, as drawing
    /// runs it: the yardstick.
    pub drawing: f64,
    /// The last step, a stretch at a time, in the order they ran: forward
    /// through each, the loss, then back through each, last first. Each
    /// one's seconds, the footprint in bytes it reached, where
    /// `crate::cap::at` is watching, and the footprint it left.
    pub trace: Vec<(f64, u64, u64)>,
}

/// Which of a UNet's linear layers a LoRA goes on: the attention
/// projections, `to_q`, `to_k`, `to_v` and `to_out.0`, of every transformer
/// block. It is where LoRAs for these models are usually put.
fn adapted(name: &str) -> bool {
    ["to_q", "to_k", "to_v", "to_out.0"].iter().any(|p| name.ends_with(p)) && name.contains(".attn")
}

/// What the UNet is told about one picture beside its noised latent: the
/// caption as the two text encoders read it, `[1, 77, 2048]` a token and
/// `[1, 1280]` pooled, and SDXL's six sizes.
pub(crate) struct Cond {
    pub(crate) ctx: Tensor,
    pub(crate) pooled: Tensor,
    pub(crate) ids: [f64; 6],
}

/// SDXL's UNet with a LoRA to train on it.
pub(crate) struct Rig {
    unet: Unet,
    adapters: Adapters,
    /// Each layer's `A` then its `B`.
    vars: Vec<Var>,
    layers: usize,
    trained: usize,
    frozen: usize,
    /// The adapted layers' names, in the order of `vars`' pairs.
    names: Vec<String>,
    device: Device,
    dtype: DType,
    /// The last checkpointed step's stretches, in the order they ran:
    /// forward through each, the loss, then back through each, last first;
    /// the seconds each took and the footprint it reached.
    trace: std::cell::RefCell<Vec<(f64, u64, u64)>>,
}

impl Rig {
    /// The UNet of `repo`, and a rank-`rank` LoRA on its attention
    /// projections, drawn from `seed`.
    ///
    /// `A` starts as PEFT and kohya start it, uniform within `±1/√in`,
    /// which keeps what it hands `B` about the size of what it read. `B`
    /// starts at a standard deviation of `b`: 0 as training starts it, so
    /// that the first step changes nothing the model draws.
    fn load(repo: &str, device: &Device, dtype: DType, rank: usize, b: f64, seed: u64) -> Res<Self> {
        let w = Watcher::none();
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let adapters = Adapters::new(&PREFIXES);
        let paths = vec![weights(repo, "unet", "diffusion_pytorch_model", &w)?];
        let r = open(&paths, dtype)?.with_adapters(adapters.part("unet"));
        let cfg = UnetConfig::from_json(&read_json(&fetch_file(repo, "unet/config.json", &w)?)?)?;
        let unet = Unet::load(&cx, &r, cfg)?;
        let frozen = finish("UNet", &paths, &r)?;
        settle(device)?;
        let mut rig = Rig { unet, adapters, vars: Vec::new(), names: Vec::new(), layers: 0, trained: 0, frozen, device: device.clone(), dtype, trace: Default::default() };
        let layers: Vec<_> = rig.adapters.linears("unet").into_iter().filter(|(n, ..)| adapted(n)).collect();
        for (i, (name, inp, out)) in layers.iter().enumerate() {
            let bound = 1.0 / (*inp as f64).sqrt();
            let a = Var::from_tensor(&uniform(seed + 2 * i as u64, &[*inp, rank], bound, device)?)?;
            let up = Var::from_tensor(&(noise(seed + 2 * i as u64 + 1, &[rank, *out], device, DType::F32)? * b)?)?;
            rig.adapters.place("unet", name, a.as_tensor(), up.as_tensor())?;
            rig.trained += rank * (inp + out);
            rig.vars.extend([a, up]);
            rig.names.push(name.clone());
        }
        rig.layers = layers.len();
        Ok(rig)
    }

    fn draw(&self, seed: u64, shape: &[usize], dt: DType) -> Res<Tensor> {
        Ok(noise(seed, shape, &self.device, DType::F32)?.to_dtype(dt)?)
    }

    /// A caption's worth of random numbers, and the sizes of an uncropped
    /// picture `side` wide: what a step costs does not depend on what they
    /// say.
    fn any_cond(&self, side: usize) -> Res<Cond> {
        let s = side as f64;
        Ok(Cond { ctx: self.draw(1, &[1, 77, self.unet.context()], self.dtype)?, pooled: self.draw(2, &[1, 1280], self.dtype)?, ids: [s, s, 0.0, 0.0, s, s] })
    }

    /// A noised latent `side` pixels wide and the noise in it, from `seed`.
    fn picture(&self, seed: u64, side: usize) -> Res<(Tensor, Tensor)> {
        let shape = [1, 4, side / 8, side / 8];
        Ok((self.draw(seed, &shape, self.dtype)?, self.draw(seed + 1, &shape, DType::F32)?))
    }

    /// The UNet's answer to `x` at timestep `t`, as drawing runs it: with
    /// the LoRA applied and nothing recorded.
    fn draws(&self, cond: &Cond, x: &Tensor, t: f64) -> Res<Tensor> {
        self.adapters.recording(false);
        let out = self.unet.forward(x, t, &cond.ctx, Some((&cond.pooled, &cond.ids)));
        self.adapters.recording(true);
        settle(&self.device)?;
        Ok(out?)
    }

    /// The loss on one picture, the mean square of what is left between
    /// the UNet's answer to `x` at `t` and `target`, and its gradient for
    /// every factor: through the whole UNet at once, or a stretch at a
    /// time.
    ///
    /// **`scale`** is what `backward` is given the loss multiplied by, and
    /// what the gradients it finds are divided by after: [`whole_scale`]
    /// for a run, and the note there says why.
    fn gradients(&self, cond: &Cond, x: &Tensor, t: f64, target: &Tensor, scale: f64, whole: bool) -> Res<(f32, GradStore)> {
        // The loss is taken in f32.
        let loss = |state: &[Tensor]| (state[0].to_dtype(DType::F32)? - target)?.sqr()?.mean_all()? * scale;
        let added = Some((&cond.pooled, &cond.ids));
        let (value, mut grads) = match whole {
            true => {
                let value = loss(&[self.unet.forward(x, t, &cond.ctx, added)?])?;
                let grads = value.backward()?;
                (value, grads)
            }
            false => {
                let temb = self.unet.embed(x, t, added)?;
                let stretches = self.unet.stretches(&temb, &cond.ctx);
                // After each stretch, each way: what it took and reached.
                let (clock, at) = (Instant::now(), std::cell::Cell::new(0.0));
                // The device is asked to let go where the state changes
                // shape, and not between stretches that leave it as it
                // was: a transformer's blocks, whose buffers the next of
                // them takes up as they are (see the module's notes).
                let last = std::cell::RefCell::new(Vec::<Vec<usize>>::new());
                let settled = |state: &[Tensor]| {
                    let shapes: Vec<Vec<usize>> = state.iter().map(|t| t.dims().to_vec()).collect();
                    let waits = *last.borrow() != shapes;
                    *last.borrow_mut() = shapes;
                    if waits {
                        settle(&self.device)?;
                    }
                    let now = clock.elapsed().as_secs_f64();
                    self.trace.borrow_mut().push((now - at.get(), crate::cap::peak(), crate::cap::footprint()));
                    at.set(now);
                    Ok(waits)
                };
                self.trace.borrow_mut().clear();
                crate::cap::peak();
                crate::grad::checkpointed(&stretches, vec![x.clone()], &loss, &self.vars, &|on| self.adapters.recording(on), &settled)?
            }
        };
        let value = (value.to_scalar::<f32>()? as f64 / scale) as f32;
        if scale != 1.0 {
            for v in &self.vars {
                if let Some(g) = grads.remove(v.as_tensor()) {
                    grads.insert(v.as_tensor(), (g / scale)?);
                }
            }
        }
        settle(&self.device)?;
        Ok((value, grads))
    }
}

/// What a loss over `numbers` numbers is multiplied by on its way into
/// `backward`: `numbers`, which makes the mean a sum.
///
/// The mean's gradient for each number of the UNet's answer is the error
/// there, twice, *divided by how many numbers there are*: 65 536 of them
/// at 1024², so about 1e-5 each. The UNet is in half precision, and so is
/// the gradient that passes back between its stages. Half precision keeps
/// three digits down to 6e-5 and nothing under 6e-8; a gradient that
/// starts at 1e-5 and is spread thinner at every layer is rounded to a few
/// levels, or to zero. The loss still falls, on what is left.
///
/// A gradient is the loss's, times whatever the loss was multiplied by. So
/// the loss goes in as the sum, whose gradient for each number is of the
/// error's own size, and each factor's gradient, which is in f32 and holds
/// either, is divided back. `tests::a_small_loss_keeps_its_gradient_in_half_precision`
/// has what it is worth.
fn whole_scale(numbers: usize) -> f64 {
    numbers as f64
}

/// `steps` training steps of a rank-`rank` LoRA on SDXL's UNet at
/// `side`×`side` pixels, one picture a step, with the UNet in `dtype` and
/// the LoRA's factors in f32; through the whole UNet at once if `whole`,
/// and otherwise a stretch at a time.
pub fn sdxl(repo: &str, device: &Device, dtype: DType, side: usize, rank: usize, steps: usize, whole: bool) -> Res<Budget> {
    let rig = Rig::load(repo, device, dtype, rank, 0.0, 100)?;
    let cond = rig.any_cond(side)?;
    // The yardstick: twice, the second timed.
    let (x, eps) = rig.picture(3, side)?;
    let scale = whole_scale(eps.elem_count());
    let mut drawing = 0.0;
    for _ in 0..2 {
        let t = Instant::now();
        rig.draws(&cond, &x, 500.0)?;
        drawing = t.elapsed().as_secs_f64();
    }
    let mut opt = crate::adam::Adam::new(rig.vars.clone(), ParamsAdamW { lr: 1e-4, ..Default::default() })?;
    let (mut secs, mut loss, mut parts) = (Vec::new(), Vec::new(), Vec::new());
    for step in 0..steps {
        let t = Instant::now();
        let (x, eps) = rig.picture(1000 + 2 * step as u64, side)?;
        // A pool a step: see `pooled`.
        let (value, found) = pooled(|| -> Res<(f32, f64)> {
            let (value, grads) = rig.gradients(&cond, &x, 500.0, &eps, scale, whole)?;
            let found = t.elapsed().as_secs_f64();
            opt.step(&grads)?;
            drop(grads);
            settle(device)?;
            Ok((value, found))
        })?;
        secs.push(t.elapsed().as_secs_f64());
        parts.push([found, secs[step] - found]);
        loss.push(value);
    }
    let trace = rig.trace.borrow().clone();
    Ok(Budget { layers: rig.layers, trained: rig.trained, frozen: rig.frozen, secs, parts, loss, drawing, trace })
}

// ---------------------------------------------------------------------------
// A run
// ---------------------------------------------------------------------------

/// What a run is asked for.
pub struct Options {
    /// The model: SDXL, or a fine-tune of it in diffusers' layout.
    pub repo: String,
    /// The folder of pictures and captions ([`super::dataset`]).
    pub data: PathBuf,
    /// The file the LoRA is written to.
    pub out: PathBuf,
    /// A LoRA this trainer wrote, to go on from.
    pub from: Option<PathBuf>,
    /// The caption of every picture that has none beside it.
    pub caption: Option<String>,
    /// Pixels a side. Every picture is fitted to a square of it.
    pub size: usize,
    /// How thin the factors are, and what their product is multiplied by,
    /// as a share of the rank: `alpha / rank`. Without one, 1.
    pub rank: usize,
    pub alpha: Option<f64>,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    /// Steps between two measurements of the validation loss.
    pub eval_every: usize,
    /// Pictures held out of training to measure it on; without a number,
    /// see [`held_out`].
    pub holdout: Option<usize>,
    pub ffmpeg: PathBuf,
    /// Raised from another thread to end the run early.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Options {
    pub fn new(repo: &str, data: &Path, out: &Path, ffmpeg: &Path) -> Self {
        Options {
            repo: repo.to_string(),
            data: data.to_path_buf(),
            out: out.to_path_buf(),
            from: None,
            caption: None,
            size: 1024,
            rank: 16,
            alpha: None,
            steps: 1000,
            lr: 1e-4,
            seed: 1337,
            eval_every: 100,
            holdout: None,
            ffmpeg: ffmpeg.to_path_buf(),
            cancel: None,
        }
    }
}

/// What a run is doing, in numbers, beside the lines it says: as
/// `kvad::train::Event` is to `kvad train`.
#[derive(Debug, Clone)]
pub enum Event {
    /// One training step: its loss, which says little (see [`run`]), and
    /// the seconds it took.
    Step { step: usize, steps: usize, loss: f32, secs: f64 },
    /// The validation loss, measured after `step` steps; step 0 is the
    /// model without the LoRA.
    Measured { step: usize, val_loss: f32 },
    /// The LoRA was written, because this step is the best so far.
    Saved { step: usize, val_loss: f32 },
}

#[derive(Debug)]
pub struct Summary {
    /// The file the best step's LoRA is in.
    pub out: PathBuf,
    /// The last step's, where that is another.
    pub last: Option<PathBuf>,
    pub layers: usize,
    pub trained: usize,
    pub pictures: usize,
    pub held_out: usize,
    /// The model's validation loss before the first step.
    pub base_val: f32,
    pub best_val: f32,
    pub best_step: usize,
    pub last_val: f32,
    pub steps: usize,
    pub elapsed_secs: f64,
    /// Whether [`Options::cancel`] ended it early.
    pub stopped: bool,
}

/// The noise levels the validation loss is measured at: the middle of each
/// quarter of the thousand.
const MEASURED_AT: [usize; 4] = [125, 375, 625, 875];

/// How many of `n` pictures are held out of training when nobody says: a
/// tenth, at least one and at most four, and none of fewer than five, of
/// which each is too much of the set to do without.
pub fn held_out(n: usize) -> usize {
    match n {
        0..=4 => 0,
        _ => (n / 10).clamp(1, 4),
    }
}

/// `0..n` in an order drawn from `rng`: Fisher and Yates' shuffle.
fn shuffled(n: usize, rng: &mut SplitMix) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (rng.unit() * (i + 1) as f64) as usize);
    }
    order
}

/// Refuses a model this cannot train, by name and with why.
fn trainable(repo: &str) -> Res<()> {
    if kvad::checkpoint::local(repo).is_some() || kvad::checkpoint::split(repo).is_some() || kvad::checkpoint::is_path(repo) {
        return Err(format!("`{repo}` is a checkpoint in one file. Training reads a UNet from a repo in diffusers' layout, so far: `{}` or a fine-tune published as it is.", super::sdxl::REPO).into());
    }
    if kvad::gguf::split(repo).is_some() {
        return Err(format!("`{repo}` is a GGUF, and a quantised denoiser has no way back for a gradient yet").into());
    }
    let index = fetch_file(repo, "model_index.json", &Watcher::none()).map_err(|e| format!("`{repo}` is not an image model on this machine or the Hub: {e}"))?;
    let class = read_json(&index)?.get("_class_name").and_then(|c| c.as_str().map(str::to_string)).unwrap_or_default();
    match class.as_str() {
        "StableDiffusionXLPipeline" => Ok(()),
        "FluxPipeline" | "QwenImagePipeline" => Err(format!(
            "`{repo}` is a {class}. Its blocks' gradients are checked, and a training step through all of them has not been made to fit or been measured; only SDXL's has (docs/tune.md)."
        )
        .into()),
        other => Err(format!("`{repo}` is a {other}, and only SDXL is trained here so far").into()),
    }
}

impl Rig {
    /// A picture as the UNet is to see it at noise level `t`, and the noise
    /// in it, which is what the UNet is asked for. The latent is drawn from
    /// the picture's Gaussian for a training step, and is its mean for a
    /// measurement, which must be the same every time.
    fn noised(&self, seen: &Seen, noising: &Noising, t: usize, seed: u64, drawn: bool) -> Res<(Tensor, Tensor)> {
        let cpu = Device::Cpu;
        let clean = match drawn {
            true => (&seen.mean + (&seen.spread * noise(seed, seen.mean.dims(), &cpu, DType::F32)?)?)?,
            false => seen.mean.clone(),
        };
        let eps = noise(seed + 1, seen.mean.dims(), &cpu, DType::F32)?;
        let (signal, noisy) = noising.mix(t);
        let x = ((clean * signal)? + (&eps * noisy)?)?;
        // Cast on the host: an upload is a buffer of its own either way.
        Ok((x.to_dtype(self.dtype)?.to_device(&self.device)?, eps.to_device(&self.device)?))
    }

    fn cond(&self, seen: &Seen) -> Res<Cond> {
        Ok(Cond { ctx: seen.ctx.to_dtype(self.dtype)?.to_device(&self.device)?, pooled: seen.pooled.to_dtype(self.dtype)?.to_device(&self.device)?, ids: seen.ids })
    }

    /// The loss on `x` at `t`, with nothing recorded.
    fn loss(&self, cond: &Cond, x: &Tensor, t: usize, eps: &Tensor) -> Res<f32> {
        Ok((self.draws(cond, x, t as f64)?.to_dtype(DType::F32)? - eps)?.sqr()?.mean_all()?.to_scalar::<f32>()?)
    }

    /// The validation loss: the mean, over `held` and the four levels of
    /// [`MEASURED_AT`], of the loss on each picture's mean latent with
    /// noise that is the same at every measurement.
    fn measure(&self, held: &[&Seen], noising: &Noising) -> Res<f32> {
        let mut sum = 0.0;
        for (i, seen) in held.iter().enumerate() {
            let cond = self.cond(seen)?;
            for (k, &t) in MEASURED_AT.iter().enumerate() {
                sum += pooled(|| -> Res<f32> {
                    let (x, eps) = self.noised(seen, noising, t, 0x5eed + 16 * i as u64 + 2 * k as u64, false)?;
                    self.loss(&cond, &x, t, &eps)
                })? as f64;
            }
        }
        Ok((sum / (held.len() * MEASURED_AT.len()) as f64) as f32)
    }

    /// Put each layer's factors on it again, `A` multiplied by `scale`:
    /// what the layer adds is then `scale · (x·A)·B`, and the gradient
    /// passes through the multiplication to `A` itself. For a `scale` of 1
    /// the factors are on the layers as they are, and stay.
    fn place(&self, scale: f64) -> Res<()> {
        if scale == 1.0 {
            return Ok(());
        }
        for (name, pair) in self.names.iter().zip(self.vars.chunks(2)) {
            self.adapters.place("unet", name, &(pair[0].as_tensor() * scale)?, pair[1].as_tensor())?;
        }
        Ok(())
    }

    /// The first step's guard (#74): that every layer's `B` has a gradient
    /// that is a number and is not nothing. With `B` at zero the `A`s have
    /// none yet, and are not asked. The gradient's length, all the `B`s'
    /// as one.
    ///
    /// What it cannot tell is a gradient that is there and wrong: one cut
    /// short at a frozen layer, or counted twice. [`Rig::slopes`] can, and
    /// only in f32, so it is a test and not a part of every run.
    fn alive(&self, grads: &GradStore) -> Res<f64> {
        let g = self.vars.iter().skip(1).step_by(2).map(|v| grads.get(v.as_tensor()).ok_or("a factor has no gradient")).collect::<Result<Vec<_>, _>>()?;
        let squares = Tensor::stack(&g.iter().map(|g| g.sqr()?.sum_all()).collect::<candle_core::Result<Vec<_>>>()?, 0)?.to_vec1::<f32>()?;
        let dead: Vec<&str> = squares.iter().zip(&self.names).filter(|(s, _)| !s.is_finite() || **s == 0.0).map(|(_, n)| n.as_str()).collect();
        if !dead.is_empty() {
            return Err(format!("not training: {} of {} layers' factors got no gradient, or one that is not a number, at the first step: {}", dead.len(), g.len(), dead.iter().take(4).cloned().collect::<Vec<_>>().join(", ")).into());
        }
        Ok(squares.iter().map(|&s| s as f64).sum::<f64>().sqrt())
    }

    /// How steeply the loss rises along `grads`, the `B`s' gradient, by
    /// two accounts: the gradient's own length, which is what `backward`
    /// says, and the loss taken a little way along it and the same way
    /// back, the difference for the distance. `far` is how far, as the
    /// share of itself the loss should move by.
    ///
    /// A gradient cut short at some frozen layer is too short, and one
    /// counted twice too long, and neither stops the loss falling; this
    /// tells both. In f32: in half precision a nudge small enough for the
    /// slope to hold is mostly lost to rounding, and the measured slope
    /// came out anywhere from 0.35 to 0.98 of the true one.
    #[cfg(test)]
    fn slopes(&self, cond: &Cond, x: &Tensor, t: usize, eps: &Tensor, value: f32, grads: &GradStore, far: f64) -> Res<(f64, f64)> {
        let length = self.alive(grads)?;
        let ups: Vec<&Var> = self.vars.iter().skip(1).step_by(2).collect();
        let h = far * value as f64 / length;
        // `affine(1, 0)` writes a new buffer; `copy` on Metal shares the
        // old, and what was kept would move with what it was kept from.
        let before: Vec<Tensor> = ups.iter().map(|v| v.as_tensor().affine(1.0, 0.0)).collect::<candle_core::Result<_>>()?;
        let mut at = [0.0; 2];
        for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
            for (v, b) in ups.iter().zip(&before) {
                let g = grads.get(v.as_tensor()).expect("checked alive");
                v.set(&(b + (g * (sign * h / length))?)?)?;
            }
            at[side] = self.loss(cond, x, t, eps)? as f64;
        }
        for (v, b) in ups.iter().zip(&before) {
            v.set(b)?;
        }
        Ok((length, (at[0] - at[1]) / (2.0 * h)))
    }

    /// The LoRA as a file a request can name, in PEFT's spelling under
    /// diffusers' `unet.`: each layer's `lora_A.weight`, `[rank, in]`, and
    /// `lora_B.weight`, `[out, rank]`, with `scale` multiplied into `B`, so
    /// that the file needs no `alpha` to mean what was trained.
    fn save(&self, scale: f64, path: &Path) -> Res<()> {
        // In a pool of its own: 1120 tensors read back from the device.
        let tensors = pooled(|| -> Res<HashMap<String, Tensor>> {
            let mut tensors = HashMap::new();
            for (name, pair) in self.names.iter().zip(self.vars.chunks(2)) {
                let host = |t: Tensor| t.t()?.contiguous()?.to_device(&Device::Cpu);
                tensors.insert(format!("unet.{name}.lora_A.weight"), host(pair[0].as_tensor().detach())?);
                tensors.insert(format!("unet.{name}.lora_B.weight"), host((pair[1].as_tensor().detach() * scale)?)?);
            }
            Ok(tensors)
        })?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let aside = path.with_extension(format!("{}.part", std::process::id()));
        candle_core::safetensors::save(&tensors, &aside)?;
        Ok(std::fs::rename(&aside, path)?)
    }

    /// Take up the factors of a file [`Rig::save`] wrote, with `scale`
    /// divided back out of `B`.
    fn resume(&self, path: &Path, scale: f64) -> Res<()> {
        let mut file = candle_core::safetensors::load(path, &Device::Cpu).map_err(|e| format!("could not read {}: {e}", path.display()))?;
        for (name, pair) in self.names.iter().zip(self.vars.chunks(2)) {
            let mut take = |which: &str, v: &Var, by: f64| -> Res<()> {
                let key = format!("unet.{name}.{which}.weight");
                let t = file.remove(&key).ok_or_else(|| format!("{} has no `{key}`: it is not a LoRA this trainer wrote for this model", path.display()))?;
                let t = (t.to_dtype(DType::F32)?.t()?.contiguous()? * by)?;
                if t.dims() != v.dims() {
                    return Err(format!("{}: `{key}` is {:?} turned, and this run's is {:?}; `--rank` must be the file's", path.display(), t.dims(), v.dims()).into());
                }
                Ok(v.set(&t.to_device(&self.device)?)?)
            };
            take("lora_A", &pair[0], 1.0)?;
            take("lora_B", &pair[1], 1.0 / scale)?;
        }
        match file.len() {
            0 => Ok(()),
            n => Err(format!("{} holds {n} tensors that are no layer's of this run", path.display()).into()),
        }
    }
}

/// Train a LoRA for SDXL on a folder of captioned pictures, saving the
/// best as it goes, and say what happened.
///
/// 1. **Read once.** Each picture goes through the VAE's encoder and each
///    caption through the text encoders ([`super::dataset::read_all`]),
///    and the encoders are let go before the UNet is loaded.
/// 2. **A step** takes the next picture of a shuffled pass over them,
///    draws its latent, a noise level out of the thousand and the noise,
///    mixes them ([`Noising::mix`]), and asks the UNet, with the LoRA on
///    its attention projections, what noise is in the mix. The loss is
///    the mean square of what it got wrong; `backward`, a stretch at a
///    time; AdamW on the factors.
/// 3. **Every `eval_every` steps** the validation loss is measured
///    ([`Rig::measure`]) and the LoRA is written if it is the best yet.
///
/// **The training loss says almost nothing.** How much noise there is to
/// find decides most of it: at level 900 the picture is nearly all noise
/// and the answer nearly the input, at 100 the noise is a faint grain to
/// pick out of a picture. A step's loss is mostly which level it drew.
/// The validation loss is the same pictures at the same four levels with
/// the same noise each time, so two measurements differ by what the LoRA
/// learned between them and by nothing else; it is on pictures the run
/// never trained on, where the set has five or more.
pub fn run(opts: &Options, device: &Device, out: &mut dyn FnMut(&str), watch: &mut dyn FnMut(Event)) -> Res<Summary> {
    let started = Instant::now();
    trainable(&opts.repo)?;
    if opts.size % 64 != 0 || !(256..=1536).contains(&opts.size) {
        return Err(format!("--size {} is not a multiple of 64 from 256 to 1536", opts.size).into());
    }
    if opts.rank == 0 || opts.steps == 0 || opts.eval_every == 0 {
        return Err("--rank, --steps and --eval-every are each at least 1".into());
    }
    let scale = opts.alpha.map_or(1.0, |a| a / opts.rank as f64);
    let entries = super::dataset::folder(&opts.data, opts.caption.as_deref())?;
    out(&format!("{} pictures in {}, each to {}×{}", entries.len(), opts.data.display(), opts.size, opts.size));
    let seen = super::dataset::read_all(&entries, &opts.repo, opts.size, &opts.ffmpeg, device, &super::dataset::cache_dir(), out)?;

    // Which pictures are held out is drawn from the seed, and so is the
    // same for two runs to be compared.
    let mut rng = SplitMix(opts.seed);
    let order = shuffled(seen.len(), &mut rng);
    let holdout = opts.holdout.unwrap_or_else(|| held_out(seen.len()));
    if holdout >= seen.len() {
        return Err(format!("--holdout {holdout} leaves none of {} pictures to train on", seen.len()).into());
    }
    let (held, train): (Vec<&Seen>, Vec<&Seen>) = (order[..holdout].iter().map(|&i| &seen[i]).collect(), order[holdout..].iter().map(|&i| &seen[i]).collect());
    // With none held out the loss is measured on up to four of the
    // pictures trained on: it then says how well they are fitted, and
    // nothing of any other picture.
    let measured: Vec<&Seen> = if held.is_empty() { train.iter().take(4).copied().collect() } else { held.clone() };
    match held.is_empty() {
        false => out(&format!("{} held out to measure on, and never trained on: {}", held.len(), held.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "))),
        true => out(&format!("none held out of {}: the validation loss is on {} of the pictures trained on, and says how well those are fitted, not how any other would be", seen.len(), measured.len())),
    }

    out("loading the UNet");
    let rig = Rig::load(&opts.repo, device, DType::F16, opts.rank, 0.0, opts.seed)?;
    if let Some(from) = &opts.from {
        rig.resume(from, scale)?;
        out(&format!("going on from {}", from.display()));
    }
    rig.place(scale)?;
    let noising = Noising::ddpm(&read_json(&fetch_file(&opts.repo, "scheduler/scheduler_config.json", &Watcher::none())?)?)?;
    out(&format!(
        "SDXL's UNet, {:.2} B frozen parameters in f16; a rank-{} LoRA on {} layers, {:.1} M trained, in f32",
        rig.frozen as f64 / 1e9,
        opts.rank,
        rig.layers,
        rig.trained as f64 / 1e6
    ));

    let base_val = rig.measure(&measured, &noising)?;
    watch(Event::Measured { step: 0, val_loss: base_val });
    out(&format!("before the first step: validation loss {base_val:.4}"));

    let mut opt = crate::adam::Adam::new(rig.vars.clone(), ParamsAdamW { lr: opts.lr, ..Default::default() })?;
    let numbers = seen[0].mean.elem_count();
    let last_path = opts.out.with_extension("last.safetensors");
    let (mut best_val, mut best_step, mut last_val) = (f32::INFINITY, 0, base_val);
    let (mut pass, mut stopped, mut taken) = (Vec::new(), false, 0);
    let (mut since, mut since_secs, mut said_pace) = (Vec::new(), 0.0, false);
    for step in 1..=opts.steps {
        if opts.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
            stopped = true;
            break;
        }
        let t0 = Instant::now();
        if pass.is_empty() {
            pass = shuffled(train.len(), &mut rng);
        }
        let picture = train[pass.pop().expect("just filled")];
        let t = (rng.unit() * noising.levels() as f64) as usize;
        let seed = rng.next();
        // A pool a step: see `pooled`.
        let value = pooled(|| -> Res<f32> {
            let cond = rig.cond(picture)?;
            let (x, eps) = rig.noised(picture, &noising, t, seed, true)?;
            let (value, grads) = rig.gradients(&cond, &x, t as f64, &eps, whole_scale(numbers), false)?;
            if !value.is_finite() {
                return Err(format!("step {step}: the loss is not a number, on {} at noise level {t}", picture.name).into());
            }
            if step == 1 {
                rig.alive(&grads)?;
            }
            opt.step(&grads)?;
            drop(grads);
            rig.place(scale)?;
            settle(device)?;
            Ok(value)
        })?;
        taken = step;
        let secs = t0.elapsed().as_secs_f64();
        watch(Event::Step { step, steps: opts.steps, loss: value, secs });
        since.push(value);
        since_secs += secs;
        // The second step on is the pace: the first compiles the kernels.
        if step == 3 && !said_pace {
            said_pace = true;
            let left = (opts.steps - step) as f64 * secs + (opts.steps / opts.eval_every) as f64 * (measured.len() * MEASURED_AT.len()) as f64 * secs / 5.0;
            out(&format!("{secs:.1} s a step: about {} for the {} steps left and their measurements", human(left), opts.steps - step));
        }

        if step % opts.eval_every == 0 || step == opts.steps {
            last_val = rig.measure(&measured, &noising)?;
            watch(Event::Measured { step, val_loss: last_val });
            let mean = since.iter().sum::<f32>() / since.len() as f32;
            let better = last_val < best_val;
            out(&format!(
                "step {step}/{}: validation loss {last_val:.4} ({:+.1}% of the model's own){}; training loss {mean:.4} over the last {} steps, {:.1} s each; {:.1} GB",
                opts.steps,
                (last_val / base_val - 1.0) * 100.0,
                if better { ", the best: saved" } else { "" },
                since.len(),
                since_secs / since.len() as f64,
                crate::cap::footprint() as f64 / 1e9
            ));
            (since, since_secs) = (Vec::new(), 0.0);
            if better {
                (best_val, best_step) = (last_val, step);
                rig.save(scale, &opts.out)?;
                watch(Event::Saved { step, val_loss: last_val });
            }
        }
    }
    // A run that was stopped, or whose last measurement was not its best,
    // leaves its last step beside its best one.
    let last = match taken > best_step && taken > 0 {
        true => {
            if taken % opts.eval_every != 0 && taken != opts.steps {
                last_val = rig.measure(&measured, &noising)?;
                watch(Event::Measured { step: taken, val_loss: last_val });
            }
            if best_step == 0 {
                (best_val, best_step) = (last_val, taken);
                rig.save(scale, &opts.out)?;
                None
            } else {
                rig.save(scale, &last_path)?;
                Some(last_path)
            }
        }
        false => None,
    };
    Ok(Summary { out: opts.out.clone(), last, layers: rig.layers, trained: rig.trained, pictures: train.len(), held_out: held.len(), base_val, best_val, best_step, last_val, steps: taken, elapsed_secs: started.elapsed().as_secs_f64(), stopped })
}

/// Seconds, as a person says them.
fn human(secs: f64) -> String {
    match secs {
        s if s < 90.0 => format!("{s:.0} s"),
        s if s < 5400.0 => format!("{:.0} min", s / 60.0),
        s => format!("{:.1} h", s / 3600.0),
    }
}

/// Every factor's gradient in `a` and in `b`, each set end to end as one
/// long vector: their cosine, their two lengths, and the least cosine any
/// one factor's pair has.
#[cfg(test)]
fn agreement(vars: &[Var], a: &GradStore, b: &GradStore) -> (f64, f64, f64, f64) {
    let (mut dot, mut a2, mut b2, mut worst) = (0f64, 0f64, 0f64, 1f64);
    for v in vars {
        let (g, w) = (a.get(v.as_tensor()).expect("a gradient for every factor"), b.get(v.as_tensor()).expect("a gradient for every factor"));
        let n = |t: Tensor| t.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
        let (d, x, y) = (n((g * w).unwrap()), n(g.sqr().unwrap()), n(w.sqr().unwrap()));
        assert!(d.is_finite() && x.is_finite() && y.is_finite(), "a gradient is not a number");
        dot += d;
        a2 += x;
        b2 += y;
        if x > 0.0 && y > 0.0 {
            worst = worst.min(d / (x * y).sqrt());
        }
    }
    (dot / (a2 * b2).sqrt(), a2.sqrt(), b2.sqrt(), worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On SDXL's real UNet, a stretch at a time finds the gradients the
    /// whole UNet at once does, for every one of the LoRA's 1120 factors,
    /// with `B` not zero so that `A` has one too. At 64×64, where both fit
    /// in little: in f32, where the two do the same arithmetic, and in f16
    /// as it runs, where the unrecorded pass is on other kernels and the
    /// checkpoints differ by their rounding.
    ///
    ///     cargo test --release -p kvad-gpu tune::tests -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_stretch_at_a_time_finds_the_whole_unets_gradients() {
        crate::cap::at(24.0);
        let device = Device::new_metal(0).unwrap();
        for (dtype, least) in [(DType::F32, 0.9999), (DType::F16, 0.99)] {
            let rig = Rig::load(super::super::sdxl::REPO, &device, dtype, 4, 0.05, 100).unwrap();
            let cond = rig.any_cond(64).unwrap();
            let (x, eps) = rig.picture(7, 64).unwrap();
            let (whole_loss, whole) = rig.gradients(&cond, &x, 500.0, &eps, 1.0, true).unwrap();
            let (loss, grads) = rig.gradients(&cond, &x, 500.0, &eps, 1.0, false).unwrap();
            let (cosine, a, b, worst) = agreement(&rig.vars, &grads, &whole);
            eprintln!("{dtype:?}: loss {loss} against {whole_loss}; all factors' gradients cosine {cosine:.6}, lengths {a:.4e} and {b:.4e}; the worst single factor {worst:.4}");
            assert!(cosine > least, "{dtype:?}: the gradients' cosine is {cosine}");
            assert!((a / b - 1.0).abs() < 1.0 - least + 0.02, "{dtype:?}: lengths {a} and {b}");
        }
    }

    /// The gradient a step finds is the loss's own slope, on SDXL's real
    /// UNet with `B` at zero as a run starts it: `backward`'s account of
    /// how steeply the loss rises along the gradient against the loss
    /// itself, moved along it ([`Rig::slopes`]). At a low, a middle and a
    /// high noise level, in f32 and at 64×64, with the loss scaled as a
    /// run scales it.
    ///
    ///     cargo test --release -p kvad-gpu tune::tests::a_step_s_gradient -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_step_s_gradient_is_the_loss_s_slope() {
        crate::cap::at(24.0);
        let device = Device::new_metal(0).unwrap();
        let rig = Rig::load(super::super::sdxl::REPO, &device, DType::F32, 16, 0.0, 100).unwrap();
        let cond = rig.any_cond(64).unwrap();
        for t in [100usize, 500, 900] {
            let (x, eps) = rig.picture(7 + t as u64, 64).unwrap();
            let (value, grads) = rig.gradients(&cond, &x, t as f64, &eps, whole_scale(eps.elem_count()), false).unwrap();
            let (said, found) = rig.slopes(&cond, &x, t, &eps, value, &grads, 0.002).unwrap();
            eprintln!("noise level {t}: loss {value:.4}; backward says {said:.5}, measured {found:.5}, {:.4} of it", found / said);
            assert!((found / said - 1.0).abs() < 0.01, "noise level {t}: backward says {said} and the loss says {found}");
        }
    }

    /// What was trained is what a request gets. The UNet's answer with the
    /// factors on it as a run holds them, scaled by `alpha / rank`, against
    /// its answer with the file [`Rig::save`] wrote set on it as a request
    /// sets a LoRA (`lora::File`, `Adapters::set`), by the name every layer
    /// has in the file; and the file read back gives a run its factors
    /// again ([`Rig::resume`]). In f32, at 64×64.
    ///
    ///     cargo test --release -p kvad-gpu tune::tests::what_was_trained -- --ignored --nocapture
    #[test]
    #[ignore]
    fn what_was_trained_is_what_a_request_gets() {
        crate::cap::at(24.0);
        let device = Device::new_metal(0).unwrap();
        let (rank, scale) = (4, 0.5);
        let rig = Rig::load(super::super::sdxl::REPO, &device, DType::F32, rank, 0.05, 100).unwrap();
        let cond = rig.any_cond(64).unwrap();
        let (x, _) = rig.picture(7, 64).unwrap();
        let bare = {
            rig.adapters.clear();
            rig.draws(&cond, &x, 500.0).unwrap()
        };
        for (name, pair) in rig.names.iter().zip(rig.vars.chunks(2)) {
            rig.adapters.place("unet", name, &(pair[0].as_tensor() * scale).unwrap(), pair[1].as_tensor()).unwrap();
        }
        let trained = rig.draws(&cond, &x, 500.0).unwrap();

        let path = std::env::temp_dir().join(format!("kvad-tune-{}.safetensors", std::process::id()));
        rig.save(scale, &path).unwrap();
        let file = super::super::lora::File::open(&path).unwrap();
        assert_eq!(file.layers(), rig.layers);
        assert_eq!(kvad::lora::local(path.to_str().unwrap()).unwrap().adapts, Some("StableDiffusionXLPipeline"));
        assert_eq!(rig.adapters.set(&[(&file, 1.0)], &device, DType::F32).unwrap(), rig.layers);
        let asked = rig.draws(&cond, &x, 500.0).unwrap();

        let largest = |t: &Tensor| t.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let (moved, apart) = (largest(&(&trained - &bare).unwrap()), largest(&(&trained - &asked).unwrap()));
        eprintln!("the LoRA moves the answer by {moved:.4} at most, and the file's by {apart:.2e} from that");
        assert!(moved > 0.01 && apart < 1e-4 * moved.max(1.0), "moved {moved}, apart {apart}");

        // And back into a run: every factor as it was.
        let before: Vec<Tensor> = rig.vars.iter().map(|v| v.as_tensor().affine(1.0, 0.0).unwrap()).collect();
        for v in &rig.vars {
            v.set(&v.zeros_like().unwrap()).unwrap();
        }
        rig.resume(&path, scale).unwrap();
        for (v, b) in rig.vars.iter().zip(&before) {
            assert!(largest(&(v.as_tensor() - b).unwrap()) < 1e-6);
        }
        assert!(rig.resume(&path, scale).is_ok());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_tenth_is_held_out_and_a_small_set_is_not_cut() {
        assert_eq!([1, 4, 5, 12, 25, 40, 400].map(held_out), [0, 0, 1, 1, 2, 4, 4]);
        // A shuffle is every index once, and the seed's own.
        let (a, b) = (shuffled(50, &mut SplitMix(3)), shuffled(50, &mut SplitMix(3)));
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert!(a == b && a != shuffled(50, &mut SplitMix(4)) && a != sorted);
        assert_eq!([40.0, 600.0, 7200.0].map(human), ["40 s", "10 min", "2.0 h"]);
    }

    /// What [`whole_scale`] is for. A 64×64 picture's loss is a mean over
    /// 256 numbers, and a 1024² picture's over 65 536: the same loss, 256
    /// times less of it for each number. So the 64×64 loss divided by 256
    /// has a 1024² loss's gradients, at a size that takes a moment, and
    /// against the gradients the UNet finds in f32, where nothing is too
    /// small to hold:
    ///
    /// - the loss as it is, in f16;
    /// - the loss divided by 256, in f16: what 1024² would be handed;
    /// - that again multiplied by 65 536, which is what a run does to it.
    ///
    ///     cargo test --release -p kvad-gpu tune::tests::a_small_loss -- --ignored --nocapture
    #[test]
    #[ignore]
    fn a_small_loss_keeps_its_gradient_in_half_precision() {
        crate::cap::at(24.0);
        let device = Device::new_metal(0).unwrap();
        let of = |dtype: DType, scales: &[f64]| -> Vec<Vec<Tensor>> {
            let rig = Rig::load(super::super::sdxl::REPO, &device, dtype, 4, 0.05, 100).unwrap();
            let cond = rig.any_cond(64).unwrap();
            let (x, eps) = rig.picture(7, 64).unwrap();
            // Out of the rig, so that one UNet is held at a time.
            let host = |g: GradStore| rig.vars.iter().map(|v| g.get(v.as_tensor()).unwrap().to_device(&Device::Cpu).unwrap()).collect::<Vec<_>>();
            scales.iter().map(|s| host(rig.gradients(&cond, &x, 500.0, &eps, *s, false).unwrap().1)).collect()
        };
        let exact = of(DType::F32, &[1.0]).remove(0);
        let cosine = |a: &[Tensor], b: &[Tensor]| {
            let n = |t: Tensor| t.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
            let (mut dot, mut a2, mut b2) = (0f64, 0f64, 0f64);
            for (g, w) in a.iter().zip(b) {
                dot += n((g * w).unwrap());
                a2 += n(g.sqr().unwrap());
                b2 += n(w.sqr().unwrap());
            }
            (dot / (a2 * b2).sqrt(), (a2 / b2).sqrt())
        };
        let small = 1.0 / 256.0;
        let half = of(DType::F16, &[1.0, small, small * whole_scale(65_536)]);
        let said = ["as it is", "divided by 256, as 1024² is", "divided by 256 and scaled as a run scales it"];
        let found: Vec<(f64, f64)> = half.iter().map(|g| cosine(g, &exact)).collect();
        for (what, (c, len)) in said.iter().zip(&found) {
            eprintln!("f16, the loss {what}: cosine {c:.6} with f32's gradients, {len:.4} of their length");
        }
        assert!(found[2].0 > 0.99, "scaled, the gradients' cosine with f32's is {}", found[2].0);
        assert!(found[2].0 >= found[1].0, "scaling made it worse: {} from {}", found[2].0, found[1].0);
    }
}
