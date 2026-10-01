//! AdamW over many small tensors as one long one.
//!
//! A LoRA is many small tensors: SDXL's is 1120, two factors for each of
//! 560 layers. candle's `AdamW` takes each in turn, and a step of it is
//! some eighteen operations a tensor: the two moments, their corrections,
//! the decay, the update, and three copies back. On a GPU an operation has
//! a price whatever its size, so twenty thousand of them were 0.34 s of
//! every step whether the picture was 256 pixels wide or 1024.
//!
//! The arithmetic is the same for every number in every tensor. So the
//! tensors and their gradients are laid end to end, the eighteen operations
//! are done once on the long vectors, and each tensor is handed its piece
//! of the answer. The moments are kept that way from the start.
//!
//! The update is candle's, to the letter:
//!
//! ```text
//! m ← β₁·m + (1 − β₁)·g                 the gradient, smoothed
//! v ← β₂·v + (1 − β₂)·g²                its square, smoothed
//! m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)   both started at zero; undo that
//! θ ← θ·(1 − lr·λ) − lr · m̂ / (√v̂ + ε)
//! ```

use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};
use candle_nn::ParamsAdamW;

pub(crate) struct Adam {
    vars: Vec<Var>,
    /// Both moments, each as long as all of `vars` end to end, in f32.
    m: Tensor,
    v: Tensor,
    t: i32,
    params: ParamsAdamW,
}

impl Adam {
    /// For `vars`, which are f32 and on one device.
    pub(crate) fn new(vars: Vec<Var>, params: ParamsAdamW) -> candle_core::Result<Self> {
        let Some(first) = vars.first() else { candle_core::bail!("AdamW of nothing") };
        if let Some(v) = vars.iter().find(|v| v.dtype() != DType::F32) {
            candle_core::bail!("AdamW here is for f32 tensors, and one is {:?}", v.dtype());
        }
        let n: usize = vars.iter().map(|v| v.elem_count()).sum();
        let zeros = Tensor::zeros(n, DType::F32, first.device())?;
        Ok(Adam { m: zeros.clone(), v: zeros, vars, t: 0, params })
    }

    /// One step. A tensor `grads` has nothing for is taken to have a
    /// gradient of zero: its moments decay and it is still decayed, where
    /// candle leaves such a tensor as it is.
    pub(crate) fn step(&mut self, grads: &GradStore) -> candle_core::Result<()> {
        self.t += 1;
        let ParamsAdamW { lr, beta1, beta2, eps, weight_decay } = self.params;
        // Detached: what comes back from `backward`, and a variable itself,
        // would have every operation below recorded.
        let flat = |t: &Tensor| t.detach().flatten_all();
        let theta = Tensor::cat(&self.vars.iter().map(|v| flat(v.as_tensor())).collect::<candle_core::Result<Vec<_>>>()?, 0)?;
        let g = self.vars.iter().map(|v| match grads.get(v.as_tensor()) {
            Some(g) => flat(g),
            None => v.zeros_like()?.flatten_all(),
        });
        let g = Tensor::cat(&g.collect::<candle_core::Result<Vec<_>>>()?, 0)?;

        self.m = ((&self.m * beta1)? + (&g * (1.0 - beta1))?)?;
        self.v = ((&self.v * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
        let m_hat = (&self.m * (1.0 / (1.0 - beta1.powi(self.t))))?;
        let v_hat = (&self.v * (1.0 / (1.0 - beta2.powi(self.t))))?;
        let next = ((theta * (1.0 - lr * weight_decay))? - ((m_hat / (v_hat.sqrt()? + eps)?)? * lr)?)?;

        let mut at = 0;
        for v in &self.vars {
            let n = v.elem_count();
            v.set(&next.narrow(0, at, n)?.reshape(v.shape())?)?;
            at += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{AdamW, Optimizer};

    /// The same numbers as candle's `AdamW` gives the same tensors from the
    /// same gradients, over several steps, so that the moments and their
    /// corrections are in it; on the CPU and on Metal.
    #[test]
    fn it_is_candles_adamw() {
        for dev in [Some(Device::Cpu), Device::new_metal(0).ok()].into_iter().flatten() {
            let shapes: [&[usize]; 4] = [&[5, 3], &[3, 7], &[11], &[2, 2, 2]];
            let make = || shapes.iter().enumerate().map(|(i, s)| Var::from_tensor(&crate::image::nn::noise(40 + i as u64, s, &dev, DType::F32).unwrap()).unwrap()).collect::<Vec<_>>();
            let (ours, theirs) = (make(), make());
            let params = ParamsAdamW { lr: 0.05, ..Default::default() };
            let mut a = Adam::new(ours.clone(), params.clone()).unwrap();
            let mut b = AdamW::new(theirs.clone(), params).unwrap();
            for step in 0..5 {
                // A loss that pulls each tensor somewhere of its own.
                let loss = |vars: &[Var]| {
                    let mut l = Tensor::zeros((), DType::F32, &dev).unwrap();
                    for (i, v) in vars.iter().enumerate() {
                        l = (l + ((v.as_tensor() * (i as f64 + 1.0)).unwrap() - 0.3 * step as f64).unwrap().sqr().unwrap().sum_all().unwrap()).unwrap();
                    }
                    l
                };
                a.step(&loss(&ours).backward().unwrap()).unwrap();
                b.step(&loss(&theirs).backward().unwrap()).unwrap();
                for (x, y) in ours.iter().zip(&theirs) {
                    let d = (x.as_tensor() - y.as_tensor()).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
                    assert!(d < 1e-6, "step {step} on {:?}: {d} apart", dev.location());
                }
            }
        }
    }
}
