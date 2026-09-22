//! What a block does after it has attended: a dense MLP, or a mixture of them.
//!
//! A transformer block is two choices, and they are independent. How it
//! attends — grouped-query over a KV cache, DeepSeek's latent compression,
//! and, in every architecture released since, a recurrent state for three
//! layers in four — is one axis. What it feeds the result through is the
//! other, and there are only two answers: one MLP, or a router and a hundred
//! of them.
//!
//! This module is the second axis, alone, so that a new architecture is a new
//! point in the grid rather than a new file that says all of this again.
//! `deepseek` reached it first and had it to itself; Qwen3's mixture has the
//! same feed-forward and none of the same attention, which is what made the
//! entanglement worth undoing.
//!
//! The routing differences between families are configuration, not code: see
//! [`Router::read`], which reads scoring, grouping, normalisation and scaling
//! from the checkpoint's own config and defaults every one of them to the
//! plain answer — softmax over a flat list, top `k`, no rescaling. A family
//! that wants exactly that, as Qwen3 does, adds nothing here.

use super::{Json, Spec};
use crate::qcache::Source;
use crate::quant::Weight;
use crate::tensor::{softmax_inplace, swiglu_inplace};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Which layers route
// ---------------------------------------------------------------------------

/// Which layers are a mixture and which are an ordinary MLP.
///
/// Public because a second backend has to reach the same answer, and reading
/// three config keys in two places is how two backends come to disagree about
/// what a checkpoint contains.
pub struct Layout {
    /// The first few layers are ordinary. The router needs a residual stream
    /// that already means something, and at layer zero it does not.
    pub first_dense: usize,
    pub moe_every: usize,
    pub n_shared: usize,
}

impl Layout {
    pub fn read(config: &Json) -> Self {
        Layout {
            first_dense: config.num(&["first_k_dense_replace"]).unwrap_or(0),
            moe_every: config.num(&["moe_layer_freq"]).unwrap_or(1).max(1),
            n_shared: config.num(&["n_shared_experts"]).unwrap_or(0),
        }
    }

    /// Whether layer `i` routes.
    pub fn is_moe(&self, i: usize, n_experts: usize) -> bool {
        n_experts > 0 && i >= self.first_dense && i % self.moe_every == 0
    }
}

// ---------------------------------------------------------------------------
// How a token picks its experts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scoring {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Select {
    /// V2-Lite: the top `k` experts, and nothing else.
    Greedy,
    /// V2-236B: pick `topk_group` groups by their best expert, then the top
    /// `k` experts within those. A device holds a group, so limiting how many
    /// groups a token can reach limits how many devices its activations have
    /// to cross.
    GroupLimited,
    /// V3: as above, but a group is scored by its best *two* experts, and the
    /// choice is made on scores plus a learned per-expert bias that does not
    /// reach the weights. The bias is nudged during training towards whatever
    /// balances the experts, which is how V3 balances them without an
    /// auxiliary loss pulling against the language objective.
    NoAuxTc,
}

/// How a token's experts are chosen, and with what weights.
#[derive(Debug, Clone)]
pub struct Router {
    pub n_experts: usize,
    pub top_k: usize,
    n_group: usize,
    topk_group: usize,
    scoring: Scoring,
    select: Select,
    norm_topk: bool,
    scale: f32,
    /// V3 multiplies by `routed_scaling_factor` whether or not it normalised;
    /// V2 does it only when it did not. The two reference files differ in
    /// exactly this line.
    always_scale: bool,
}

impl Router {
    pub fn read(spec: &Spec) -> Res<Self> {
        let c = &spec.config;
        let scoring = match c.text("scoring_func").unwrap_or("softmax") {
            "softmax" => Scoring::Softmax,
            "sigmoid" => Scoring::Sigmoid,
            other => return Err(format!("unknown scoring_func `{other}`").into()),
        };
        let select = match c.text("topk_method").unwrap_or("greedy") {
            "greedy" => Select::Greedy,
            "group_limited_greedy" => Select::GroupLimited,
            "noaux_tc" => Select::NoAuxTc,
            other => return Err(format!("unknown topk_method `{other}`").into()),
        };
        Ok(Router {
            n_experts: c.need("n_routed_experts")?,
            top_k: c.need("num_experts_per_tok")?,
            n_group: c.num(&["n_group"]).unwrap_or(1),
            topk_group: c.num(&["topk_group"]).unwrap_or(1),
            scoring,
            select,
            norm_topk: c.flag("norm_topk_prob").unwrap_or(false),
            scale: c.float(&["routed_scaling_factor"]).unwrap_or(1.0),
            always_scale: select == Select::NoAuxTc,
        })
    }

    /// Which experts run for one token, and how much each one counts.
    pub fn route(&self, logits: &[f32], bias: Option<&[f32]>, out: &mut Vec<(usize, f32)>) {
        let mut scores = logits.to_vec();
        match self.scoring {
            Scoring::Softmax => softmax_inplace(&mut scores),
            Scoring::Sigmoid => {
                for s in &mut scores {
                    *s = 1.0 / (1.0 + (-*s).exp());
                }
            }
        }

        // What the *choice* is made on, which for V3 is not what the weights
        // are read from.
        let mut choice = scores.clone();
        if let Some(bias) = bias {
            for (c, b) in choice.iter_mut().zip(bias) {
                *c += b;
            }
        }
        if self.select != Select::Greedy && self.n_group > 1 {
            self.mask_to_best_groups(&mut choice);
        }

        out.clear();
        // `top_k` is 6 or 8 against 64 or 256 experts, so a partial selection
        // beats sorting the whole row.
        for _ in 0..self.top_k.min(self.n_experts) {
            let mut best = usize::MAX;
            for i in 0..self.n_experts {
                if choice[i] > f32::NEG_INFINITY && (best == usize::MAX || choice[i] > choice[best])
                {
                    best = i;
                }
            }
            if best == usize::MAX {
                break;
            }
            out.push((best, scores[best]));
            choice[best] = f32::NEG_INFINITY;
        }

        let normalised = self.norm_topk && self.top_k > 1;
        if normalised {
            let total: f32 = out.iter().map(|(_, w)| *w).sum::<f32>() + 1e-20;
            for (_, w) in out.iter_mut() {
                *w /= total;
            }
        }
        if self.always_scale || !normalised {
            for (_, w) in out.iter_mut() {
                *w *= self.scale;
            }
        }
    }

    /// Knock every expert outside the best `topk_group` groups out of the
    /// running.
    fn mask_to_best_groups(&self, choice: &mut [f32]) {
        let per_group = self.n_experts / self.n_group;
        let strength = |g: usize| -> f32 {
            let group = &choice[g * per_group..(g + 1) * per_group];
            match self.select {
                // V2 scores a group by its single best expert. (It also masks
                // the losers to zero rather than to minus infinity, which can
                // pick a masked expert when fewer than `top_k` survive; that
                // only happens if a whole group scores zero under a softmax,
                // and this refuses to instead.)
                Select::GroupLimited => group.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                // V3 by its best two, which stops one strong expert from
                // dragging in a group that is otherwise weak.
                _ => {
                    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for &v in group {
                        if v > a {
                            b = a;
                            a = v;
                        } else if v > b {
                            b = v;
                        }
                    }
                    a + b
                }
            }
        };
        let mut order: Vec<usize> = (0..self.n_group).collect();
        order.sort_by(|&a, &b| strength(b).total_cmp(&strength(a)));
        for &g in &order[self.topk_group.min(self.n_group)..] {
            for v in &mut choice[g * per_group..(g + 1) * per_group] {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The weights
// ---------------------------------------------------------------------------

/// A SwiGLU MLP, which is what both a dense layer and one expert are.
pub struct Ffn {
    gate: Weight,
    up: Weight,
    down: Weight,
}

impl Ffn {
    pub fn load(src: &dyn Source, prefix: &str) -> Res<Self> {
        Ok(Ffn {
            gate: src.matrix(&format!("{prefix}.gate_proj.weight"))?,
            up: src.matrix(&format!("{prefix}.up_proj.weight"))?,
            down: src.matrix(&format!("{prefix}.down_proj.weight"))?,
        })
    }

    pub fn run(&self, xs: &[f32], m: usize) -> Vec<f32> {
        let mut gate = self.gate.matmul_bt(xs, m, None);
        let up = self.up.matmul_bt(xs, m, None);
        swiglu_inplace(&mut gate, &up);
        self.down.matmul_bt(&gate, m, None)
    }

    /// One token, through the matrix-*vector* kernel.
    ///
    /// Not `run` with a batch of one. `matvec_bt` and `matmul_bt` are separate
    /// kernels, and this is the path every generated token takes: folding it
    /// into the batched spelling for tidiness would be a change to the hottest
    /// loop in the engine, measured in neither direction.
    pub fn run_one(&self, x: &[f32]) -> Vec<f32> {
        let mut gate = self.gate.matvec_bt(x, None);
        let up = self.up.matvec_bt(x, None);
        swiglu_inplace(&mut gate, &up);
        self.down.matvec_bt(&gate, None)
    }

    pub fn param_count(&self) -> usize {
        self.gate.param_count() + self.up.param_count() + self.down.param_count()
    }

    pub fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

pub enum Mlp {
    /// The first `first_k_dense_replace` layers are ordinary. The router
    /// needs a residual stream that already means something, and at layer
    /// zero it does not.
    Dense(Ffn),
    /// Boxed, and not for recursion: an unboxed mixture makes every variant as
    /// wide as a hundred experts' worth of bookkeeping, so a dense block pays
    /// for the mixture it does not have. It cost 9% of decode on Qwen2.5-0.5B
    /// when this enum first replaced three fields on the block.
    Moe(Box<Moe>),
}

pub struct Moe {
    /// How this family picks. Held here rather than on the model, so that a
    /// mixture without a routing policy cannot be built at all.
    pub router: Router,
    /// `[n_routed_experts, hidden]` — the router itself, one dot product per
    /// expert.
    pub gate: Weight,
    /// V3's learned balancing bias, if this is V3.
    pub bias: Option<Vec<f32>>,
    pub experts: Vec<Ffn>,
    /// Runs for every token, whatever the router says.
    pub shared: Option<Ffn>,
}

impl Moe {
    /// The mixture, over a batch.
    ///
    /// Grouped by expert rather than by token. Every token picks its own
    /// handful, so a batch of sixty-four touches most of the sixty-four
    /// experts however you slice it — but walking tokens would read each
    /// chosen expert's weights again for every token that chose it, and
    /// walking experts reads each one once.
    pub fn run(&self, hs: &[f32], m: usize, e: usize) -> Vec<f32> {
        let router = &self.router;
        let n = router.n_experts;
        let logits = self.gate.matmul_bt(hs, m, None);

        let mut picks = Vec::with_capacity(router.top_k);
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for i in 0..m {
            router
                .route(&logits[i * n..(i + 1) * n], self.bias.as_deref(), &mut picks);
            for &(expert, weight) in &picks {
                by_expert[expert].push((i, weight));
            }
        }

        let mut out = match &self.shared {
            // Every token, so it is one batched pass and needs no grouping.
            Some(shared) => shared.run(hs, m),
            None => vec![0.0f32; m * e],
        };
        let mut rows = Vec::with_capacity(m * e);
        for (expert, tokens) in by_expert.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            rows.clear();
            for &(i, _) in tokens {
                rows.extend_from_slice(&hs[i * e..(i + 1) * e]);
            }
            let y = self.experts[expert].run(&rows, tokens.len());
            for (j, &(i, weight)) in tokens.iter().enumerate() {
                for (o, v) in out[i * e..(i + 1) * e]
                    .iter_mut()
                    .zip(&y[j * e..(j + 1) * e])
                {
                    *o += weight * v;
                }
            }
        }
        out
    }
}


impl Mlp {
    /// A batch of tokens. Prefill, and scoring.
    pub fn run(&self, hs: &[f32], m: usize, e: usize) -> Vec<f32> {
        match self {
            Mlp::Dense(ffn) => ffn.run(hs, m),
            Mlp::Moe(moe) => moe.run(hs, m, e),
        }
    }

    /// One token. See [`Ffn::run_one`] for why this is not the batch of one.
    ///
    /// A mixture has no such kernel yet: it routes one token through the
    /// batched expert path, which is a batch of one per chosen expert however
    /// it is spelled. Worth revisiting when a mixture is the thing being
    /// decoded rather than the thing being loaded.
    pub fn run_one(&self, x: &[f32], e: usize) -> Vec<f32> {
        match self {
            Mlp::Dense(ffn) => ffn.run_one(x),
            Mlp::Moe(moe) => moe.run(x, 1, e),
        }
    }

    pub fn param_count(&self) -> usize {
        match self {
            Mlp::Dense(ffn) => ffn.param_count(),
            Mlp::Moe(moe) => {
                moe.gate.param_count()
                    + moe.bias.as_ref().map_or(0, Vec::len)
                    + moe.experts.iter().map(Ffn::param_count).sum::<usize>()
                    + moe.shared.as_ref().map_or(0, Ffn::param_count)
            }
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mlp::Dense(ffn) => ffn.bytes(),
            Mlp::Moe(moe) => {
                moe.gate.bytes()
                    + moe.experts.iter().map(Ffn::bytes).sum::<usize>()
                    + moe.shared.as_ref().map_or(0, Ffn::bytes)
            }
        }
    }
}
