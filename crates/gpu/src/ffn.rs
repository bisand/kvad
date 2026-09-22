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
use candle_core::{DType, Tensor};
use candle_nn::ops;
use kvad::model::ffn::Router;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A SwiGLU MLP: a dense layer, a shared expert, or one routed expert.
pub(crate) struct Ffn {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Ffn {
    /// `width` is the MLP's inner dimension, which for an expert is
    /// `moe_intermediate_size` and not the dense layers' `intermediate_size`.
    /// Qwen3-30B-A3B's are 768 and 6144, so getting this wrong is a shape
    /// error rather than a silent one — but only because candle is told the
    /// shape it expects, which is the whole reason these widths are passed in.
    pub(crate) fn load(ld: &Loader, vb: &Reader<'_>, e: usize, width: usize) -> Res<Self> {
        Ok(Ffn {
            gate: ld.proj(vb, "gate_proj.weight", width, e, Stored::OutIn)?,
            up: ld.proj(vb, "up_proj.weight", width, e, Stored::OutIn)?,
            down: ld.proj(vb, "down_proj.weight", e, width, Stored::OutIn)?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(gate * self.up.forward(x)?)?)
    }

    pub(crate) fn params(&self) -> usize {
        self.gate.params() + self.up.params() + self.down.params()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
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
    /// Runs for every token, whatever the router says. DeepSeek has one;
    /// Qwen3 does not.
    shared: Option<Ffn>,
}

impl Moe {
    pub(crate) fn load(
        ld: &Loader,
        vb: &Reader<'_>,
        e: usize,
        width: usize,
        router: &Router,
        n_shared: usize,
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
            // The shared experts are stored fused: `n` of them side by side in
            // one matrix, so the width is `n` times an expert's.
            shared: match n_shared {
                0 => None,
                n => Some(Ffn::load(ld, &vb.pp("shared_experts"), e, width * n)?),
            },
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
            + self.shared.as_ref().map_or(0, Ffn::params)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.gate.bytes()
            + self.experts.iter().map(Ffn::bytes).sum::<usize>()
            + self.shared.as_ref().map_or(0, Ffn::bytes)
    }
}

pub(crate) enum Mlp {
    Dense(Ffn),
    /// Boxed, as on the CPU: an unboxed mixture makes every variant as wide as
    /// a hundred experts' worth of bookkeeping, so a dense block pays for the
    /// mixture it has not got.
    Moe(Box<Moe>),
}

impl Mlp {
    pub(crate) fn forward(&self, h: &Tensor, m: usize, e: usize) -> Res<Tensor> {
        match self {
            Mlp::Dense(f) => Ok(f.forward(h)?),
            Mlp::Moe(moe) => moe.forward(h, m, e),
        }
    }

    pub(crate) fn params(&self) -> usize {
        match self {
            Mlp::Dense(f) => f.params(),
            Mlp::Moe(moe) => moe.params(),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        match self {
            Mlp::Dense(f) => f.bytes(),
            Mlp::Moe(moe) => moe.bytes(),
        }
    }
}
