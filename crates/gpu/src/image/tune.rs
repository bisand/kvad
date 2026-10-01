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

use super::lora::Adapters;
use super::nn::{noise, Ctx};
use super::sdxl::{weights, PREFIXES};
use super::unet::{Unet, UnetConfig};
use super::{finish, open, read_json};
use crate::common::{settle, Loader};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor, Var};
use candle_core::backprop::GradStore;
use candle_nn::ParamsAdamW;
use kvad::weights::{fetch_file, Watcher};
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

/// SDXL's UNet with a LoRA to train on it, and one picture's conditioning.
struct Rig {
    unet: Unet,
    adapters: Adapters,
    /// Each layer's `A` then its `B`.
    vars: Vec<Var>,
    layers: usize,
    trained: usize,
    frozen: usize,
    device: Device,
    dtype: DType,
    side: usize,
    ctx: Tensor,
    pooled: Tensor,
    /// The last checkpointed step's stretches, in the order they ran:
    /// forward through each, the loss, then back through each, last first;
    /// the seconds each took and the footprint it reached.
    trace: std::cell::RefCell<Vec<(f64, u64, u64)>>,
}

impl Rig {
    /// `b` is the standard deviation `B` starts at: 0 as training starts
    /// it, so that the first step changes nothing the model draws.
    fn load(repo: &str, device: &Device, dtype: DType, side: usize, rank: usize, b: f64) -> Res<Self> {
        let w = Watcher::none();
        let vault = Vault::off();
        let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
        let adapters = Adapters::new(&PREFIXES);
        let paths = vec![weights(repo, "unet", "diffusion_pytorch_model", &w)?];
        let r = open(&paths, dtype)?.with_adapters(adapters.part("unet"));
        let cfg = UnetConfig::from_json(&read_json(&fetch_file(repo, "unet/config.json", &w)?)?)?;
        let context = cfg.context;
        let unet = Unet::load(&cx, &r, cfg)?;
        let frozen = finish("UNet", &paths, &r)?;
        settle(device)?;
        let mut rig = Rig { unet, adapters, vars: Vec::new(), layers: 0, trained: 0, frozen, device: device.clone(), dtype, side, ctx: Tensor::zeros(1, dtype, device)?, pooled: Tensor::zeros(1, dtype, device)?, trace: Default::default() };
        // The prompt's 77 tokens and its pooled summary.
        rig.ctx = rig.draw(1, &[1, 77, context], dtype)?;
        rig.pooled = rig.draw(2, &[1, 1280], dtype)?;
        let layers: Vec<_> = rig.adapters.linears("unet").into_iter().filter(|(n, ..)| adapted(n)).collect();
        for (i, (name, inp, out)) in layers.iter().enumerate() {
            let a = Var::from_tensor(&(rig.draw(100 + 2 * i as u64, &[*inp, rank], DType::F32)? * (1.0 / rank as f64))?)?;
            let up = Var::from_tensor(&(rig.draw(101 + 2 * i as u64, &[rank, *out], DType::F32)? * b)?)?;
            rig.adapters.place("unet", name, a.as_tensor(), up.as_tensor())?;
            rig.trained += rank * (inp + out);
            rig.vars.extend([a, up]);
        }
        rig.layers = layers.len();
        Ok(rig)
    }

    fn draw(&self, seed: u64, shape: &[usize], dt: DType) -> Res<Tensor> {
        Ok(noise(seed, shape, &self.device, DType::F32)?.to_dtype(dt)?)
    }

    /// A noised latent and the noise in it, from `seed`.
    fn picture(&self, seed: u64) -> Res<(Tensor, Tensor)> {
        let shape = [1, 4, self.side / 8, self.side / 8];
        Ok((self.draw(seed, &shape, self.dtype)?, self.draw(seed + 1, &shape, DType::F32)?))
    }

    /// SDXL's size conditioning for a picture this size, uncropped.
    fn ids(&self) -> [f64; 6] {
        let s = self.side as f64;
        [s, s, 0.0, 0.0, s, s]
    }

    /// One forward pass as drawing runs it, with recording off.
    fn draws(&self, x: &Tensor) -> Res<Tensor> {
        self.adapters.recording(false);
        let out = self.unet.forward(x, 500.0, &self.ctx, Some((&self.pooled, &self.ids())));
        self.adapters.recording(true);
        settle(&self.device)?;
        Ok(out?)
    }

    /// The loss on one picture and its gradient for every factor: through
    /// the whole UNet at once, or a stretch at a time.
    fn gradients(&self, x: &Tensor, eps: &Tensor, whole: bool) -> Res<(f32, GradStore)> {
        // SDXL predicts the noise; the loss is taken in f32.
        let loss = |state: &[Tensor]| (state[0].to_dtype(DType::F32)? - eps)?.sqr()?.mean_all();
        let ids = self.ids();
        let added = Some((&self.pooled, &ids));
        let (value, grads) = match whole {
            true => {
                let value = loss(&[self.unet.forward(x, 500.0, &self.ctx, added)?])?;
                let grads = value.backward()?;
                (value, grads)
            }
            false => {
                let temb = self.unet.embed(x, 500.0, added)?;
                let stretches = self.unet.stretches(&temb, &self.ctx);
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
        let value = value.to_scalar::<f32>()?;
        settle(&self.device)?;
        Ok((value, grads))
    }
}

/// `steps` training steps of a rank-`rank` LoRA on SDXL's UNet at
/// `side`×`side` pixels, one picture a step, with the UNet in `dtype` and
/// the LoRA's factors in f32; through the whole UNet at once if `whole`,
/// and otherwise a stretch at a time.
pub fn sdxl(repo: &str, device: &Device, dtype: DType, side: usize, rank: usize, steps: usize, whole: bool) -> Res<Budget> {
    let rig = Rig::load(repo, device, dtype, side, rank, 0.0)?;
    // The yardstick: twice, the second timed.
    let (x, _) = rig.picture(3)?;
    let mut drawing = 0.0;
    for _ in 0..2 {
        let t = Instant::now();
        rig.draws(&x)?;
        drawing = t.elapsed().as_secs_f64();
    }
    let mut opt = crate::adam::Adam::new(rig.vars.clone(), ParamsAdamW { lr: 1e-4, ..Default::default() })?;
    let (mut secs, mut loss, mut parts) = (Vec::new(), Vec::new(), Vec::new());
    for step in 0..steps {
        let t = Instant::now();
        let (x, eps) = rig.picture(1000 + 2 * step as u64)?;
        let (value, grads) = rig.gradients(&x, &eps, whole)?;
        let found = t.elapsed().as_secs_f64();
        opt.step(&grads)?;
        drop(grads);
        settle(device)?;
        secs.push(t.elapsed().as_secs_f64());
        parts.push([found, secs[step] - found]);
        loss.push(value);
    }
    let trace = rig.trace.borrow().clone();
    Ok(Budget { layers: rig.layers, trained: rig.trained, frozen: rig.frozen, secs, parts, loss, drawing, trace })
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
            let rig = Rig::load(super::super::sdxl::REPO, &device, dtype, 64, 4, 0.05).unwrap();
            let (x, eps) = rig.picture(7).unwrap();
            let (whole_loss, whole) = rig.gradients(&x, &eps, true).unwrap();
            let (loss, grads) = rig.gradients(&x, &eps, false).unwrap();
            // Every factor's two gradients as one long vector each: their
            // cosine, and their lengths.
            let (mut dot, mut a2, mut b2, mut worst) = (0f64, 0f64, 0f64, 1f64);
            for v in &rig.vars {
                let (g, w) = (grads.get(v.as_tensor()).expect("a gradient for every factor"), whole.get(v.as_tensor()).unwrap());
                let n = |t: Tensor| t.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
                let (d, a, b) = (n((g * w).unwrap()), n(g.sqr().unwrap()), n(w.sqr().unwrap()));
                assert!(d.is_finite() && a.is_finite(), "{dtype:?}: a gradient is not a number");
                dot += d;
                a2 += a;
                b2 += b;
                if a > 0.0 && b > 0.0 {
                    worst = worst.min(d / (a * b).sqrt());
                }
            }
            let cosine = dot / (a2 * b2).sqrt();
            eprintln!("{dtype:?}: loss {loss} against {whole_loss}; all factors' gradients cosine {cosine:.6}, lengths {:.4e} and {:.4e}; the worst single factor {worst:.4}", a2.sqrt(), b2.sqrt());
            assert!(cosine > least, "{dtype:?}: the gradients' cosine is {cosine}");
            assert!((a2.sqrt() / b2.sqrt() - 1.0).abs() < 1.0 - least + 0.02, "{dtype:?}: lengths {} and {}", a2.sqrt(), b2.sqrt());
        }
    }
}
