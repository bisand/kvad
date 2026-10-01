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
//! | 256², a stretch at a time | 1.7 s | 7.7 GB | 0.14 s |
//! | 352² | 2.5 s | 9.9 GB | 0.20 s |
//! | 480² | 3.1 s | 13.8 GB | 0.33 s |
//! | 512² | 3.3 s | 15.3 GB | 0.35 s |
//! | 640² | 5.3 s | 19.8 GB | 0.55 s |
//!
//! (352² was measured before a convolution's backward went by the turned
//! kernel, which took 480² from 4.1 s to 3.1.)
//!
//! - **Memory is 5.1 GB of weights and about 37 GB a megapixel**, a
//!   straight line through all five sizes. By it 768² is 27 GB and 1024²
//!   44 GB, which this machine does not have to spare, and neither was
//!   run. Recorded whole, 512² took the machine down.
//! - **A megabyte is not where that goes.** The activations at these sizes
//!   are megabytes each; the stretches that reach the peak are the ten
//!   transformer blocks at the lowest level, at 15×15 tokens, and what
//!   they hold there is not yet found. It is not the LoRA: at rank 1 the
//!   peak is the same.
//! - **A step is nine times a forward pass**, most of it coming back
//!   through the transformer blocks, and 0.35 s of it is the optimiser,
//!   the same at any size: 1120 small tensors, each a few operations.
//!
//! So what #75 can promise from this is 512² on this machine, in about
//! 3 s a step and 15 GB; and 1024², SDXL's own size, not before the
//! memory above is found.

use super::lora::Adapters;
use super::nn::{noise, Ctx};
use super::sdxl::{weights, PREFIXES};
use super::unet::{Unet, UnetConfig};
use super::{finish, open, read_json};
use crate::common::{settle, Loader};
use crate::qcache::Vault;
use candle_core::{DType, Device, Tensor, Var};
use candle_core::backprop::GradStore;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
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
    /// one's seconds, and the footprint in bytes it reached, where
    /// `crate::cap::at` is watching.
    pub trace: Vec<(f64, u64)>,
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
    trace: std::cell::RefCell<Vec<(f64, u64)>>,
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
                let settled = || {
                    settle(&self.device)?;
                    let now = clock.elapsed().as_secs_f64();
                    self.trace.borrow_mut().push((now - at.get(), crate::cap::peak()));
                    at.set(now);
                    Ok(())
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
    let mut opt = AdamW::new(rig.vars.clone(), ParamsAdamW { lr: 1e-4, ..Default::default() })?;
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
