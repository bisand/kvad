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
//!
//! What *is* code is the expert that runs for every token. There are three
//! answers — none, DeepSeek's, and Qwen3-Next's, which a token can decline
//! through a sigmoid gate — and [`Shared`] is all three, because the middle
//! one and the last one are near enough alike that running either under the
//! other's rule would load, run, and be wrong by an amount that varies per
//! token.

use super::Json;
use crate::qcache::Source;
use crate::residency;
use crate::quant::Weight;
use crate::tensor::{softmax_inplace, swiglu_inplace};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Which layers route
// ---------------------------------------------------------------------------

/// The block number out of a weight prefix: `model.layers.3.mlp` is 3.
///
/// Every family spells it the same way, because every family's checkpoint
/// was written by the same PyTorch idiom. A prefix that does not say is not
/// worth failing a load over — it costs a residency trace its layer
/// numbering and nothing else — so this answers zero and says nothing.
fn layer_of(prefix: &str) -> usize {
    prefix
        .split('.')
        .skip_while(|part| *part != "layers")
        .nth(1)
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// The expert that runs for every token, whatever the router says.
///
/// Two families have one and they do not agree about it. DeepSeek stores `n`
/// experts side by side in a single matrix and adds the result as it comes;
/// Qwen3-Next stores one MLP of its own width and multiplies it by
/// `sigmoid(x · w)` first, so a token can decline it. An ungated shared expert
/// is this one with the sigmoid nailed to one, and the two are close enough
/// that running either under the other's rule would load, run, and be wrong
/// by a factor that varies per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shared {
    /// Nobody runs unconditionally. Qwen3's mixture, and Qwen3-Coder's.
    None,
    /// DeepSeek's, under `mlp.shared_experts`: `n` experts' worth of width in
    /// one matrix, added whole.
    Fused(usize),
    /// Qwen3-Next's, under `mlp.shared_expert`, of the width given, with
    /// `mlp.shared_expert_gate` deciding per token how much of it counts.
    Gated(usize),
}

impl Shared {
    /// What this is called in a checkpoint. Singular or plural, and the
    /// difference is not cosmetic: asking for the wrong one is a load error,
    /// which is the outcome to want.
    fn name(self) -> &'static str {
        match self {
            Shared::Fused(_) => "shared_experts",
            _ => "shared_expert",
        }
    }

    /// How wide the shared expert is, given one routed expert's width.
    ///
    /// Only the framework backend needs this — the hand-written loader reads
    /// every shape from the checkpoint — but both have to reach the same
    /// number, so it is worked out once here.
    pub fn width(self, expert: usize) -> usize {
        match self {
            Shared::None => 0,
            Shared::Fused(n) => expert * n,
            Shared::Gated(w) => w,
        }
    }
}

/// Which layers are a mixture and which are an ordinary MLP.
///
/// Public because a second backend has to reach the same answer, and reading
/// these config keys in two places is how two backends come to disagree about
/// what a checkpoint contains.
pub struct Layout {
    /// The first few layers are ordinary. The router needs a residual stream
    /// that already means something, and at layer zero it does not.
    ///
    /// DeepSeek's way of saying it. Qwen3 says the same thing the other way
    /// round, by naming the dense layers in `mlp_only_layers`.
    first_dense: usize,
    /// One layer in `every` routes.
    every: usize,
    /// Whether the layers are counted from one rather than from zero.
    ///
    /// DeepSeek asks `layer % moe_layer_freq == 0` and Qwen3 asks
    /// `(layer + 1) % decoder_sparse_step == 0`. The two agree whenever the
    /// period is 1 — which it is in every checkpoint either family has
    /// published — and disagree about which layers route the moment it is
    /// not. Carrying the offset costs a `usize::from` and means neither
    /// family is being run by the other's rule.
    from_one: bool,
    /// The expert that runs unconditionally, if this family has one.
    pub shared: Shared,
    /// Qwen3's explicit list of layers that are an ordinary MLP whatever the
    /// step says. Empty in the published checkpoints, and the only thing here
    /// that is a list rather than a rule.
    dense: Vec<usize>,
}

impl Layout {
    pub fn read(config: &Json) -> Self {
        // Qwen3's spelling first: a config that has it is not DeepSeek's, so
        // the count that comes with it is the one to obey.
        let step = config.num(&["decoder_sparse_step"]);
        Layout {
            first_dense: config.num(&["first_k_dense_replace"]).unwrap_or(0),
            every: step.or_else(|| config.num(&["moe_layer_freq"])).unwrap_or(1).max(1),
            from_one: step.is_some(),
            // Qwen3-Next's spelling first, and it names a width rather than
            // a count: it has exactly one shared expert and says how wide.
            shared: match config.num(&["shared_expert_intermediate_size"]) {
                Some(width) => Shared::Gated(width),
                None => match config.num(&["n_shared_experts"]).unwrap_or(0) {
                    0 => Shared::None,
                    n => Shared::Fused(n),
                },
            },
            dense: config
                .get("mlp_only_layers")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|n| n as usize).collect())
                .unwrap_or_default(),
        }
    }

    /// Whether layer `i` routes.
    pub fn is_moe(&self, i: usize, n_experts: usize) -> bool {
        n_experts > 0
            && i >= self.first_dense
            && !self.dense.contains(&i)
            && (i + usize::from(self.from_one)) % self.every == 0
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
    /// How many routed experts this config has, or `None` for a dense model.
    ///
    /// The two spellings are the same quantity: `n_routed_experts` is
    /// DeepSeek's and `num_experts` is Qwen3's. Separate from [`Router::read`]
    /// because an architecture that is dense *or* routed — `llama`, once
    /// `qwen3_moe` joined it — has to ask the question before it can know
    /// whether there is a router to read at all.
    pub fn count(config: &Json) -> Option<usize> {
        config.num(&["n_routed_experts", "num_experts"])
    }

    /// The config's routing policy, whole.
    ///
    /// Takes the config and not the `Spec` that holds it, as [`Layout::read`]
    /// does: nothing here is about widths or head counts, and a test that wants
    /// to ask what a config routes should not have to build a model first.
    pub fn read(config: &Json) -> Res<Self> {
        let scoring = match config.text("scoring_func").unwrap_or("softmax") {
            "softmax" => Scoring::Softmax,
            "sigmoid" => Scoring::Sigmoid,
            other => return Err(format!("unknown scoring_func `{other}`").into()),
        };
        let select = match config.text("topk_method").unwrap_or("greedy") {
            "greedy" => Select::Greedy,
            "group_limited_greedy" => Select::GroupLimited,
            "noaux_tc" => Select::NoAuxTc,
            other => return Err(format!("unknown topk_method `{other}`").into()),
        };
        Ok(Router {
            n_experts: Router::count(config)
                .ok_or("config: no `n_routed_experts` and no `num_experts`")?,
            top_k: config.need("num_experts_per_tok")?,
            n_group: config.num(&["n_group"]).unwrap_or(1),
            topk_group: config.num(&["topk_group"]).unwrap_or(1),
            scoring,
            select,
            norm_topk: config.flag("norm_topk_prob").unwrap_or(false),
            scale: config.float(&["routed_scaling_factor"]).unwrap_or(1.0),
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

    /// This MLP with its weights read from `slab`. See [`Weight::rebased`].
    fn rebased(&self, lo: u64, slab: &std::sync::Arc<crate::qcache::Slab>) -> Option<Ffn> {
        Some(Ffn {
            gate: self.gate.rebased(lo, slab)?,
            up: self.up.rebased(lo, slab)?,
            down: self.down.rebased(lo, slab)?,
        })
    }

    pub fn param_count(&self) -> usize {
        self.gate.param_count() + self.up.param_count() + self.down.param_count()
    }

    pub fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

/// The shared expert, and whatever decides how much of it counts.
///
/// One struct rather than two fields on [`Moe`], because an `Option<Ffn>` and
/// an `Option<Weight>` that have to agree about whether they are present is
/// two chances to disagree.
pub struct SharedExpert {
    ffn: Ffn,
    /// `[1, hidden]`: Qwen3-Next's per-token sigmoid. `None` for DeepSeek,
    /// which adds its shared experts whole.
    gate: Option<Weight>,
}

impl SharedExpert {
    fn load(src: &dyn Source, prefix: &str, shared: Shared) -> Res<Option<Self>> {
        if shared == Shared::None {
            return Ok(None);
        }
        Ok(Some(SharedExpert {
            ffn: Ffn::load(src, &format!("{prefix}.{}", shared.name()))?,
            gate: match shared {
                Shared::Gated(_) => Some(src.matrix(&format!("{prefix}.shared_expert_gate.weight"))?),
                _ => None,
            },
        }))
    }

    /// A batch, which is every batch: the shared expert runs for every token
    /// and so needs no grouping. Doubles as the accumulator the routed
    /// experts add into.
    fn run(&self, hs: &[f32], m: usize) -> Vec<f32> {
        let mut out = self.ffn.run(hs, m);
        if let Some(gate) = &self.gate {
            // One row, so one number per token.
            let scores = gate.matmul_bt(hs, m, None);
            let e = out.len() / m;
            for (i, score) in scores.iter().enumerate().take(m) {
                let g = 1.0 / (1.0 + (-score).exp());
                for o in &mut out[i * e..(i + 1) * e] {
                    *o *= g;
                }
            }
        }
        out
    }

    fn param_count(&self) -> usize {
        self.ffn.param_count() + self.gate.as_ref().map_or(0, Weight::param_count)
    }

    fn bytes(&self) -> usize {
        self.ffn.bytes() + self.gate.as_ref().map_or(0, Weight::bytes)
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
    pub shared: Option<SharedExpert>,
    /// Which block this is.
    ///
    /// Not used by the arithmetic, and kept anyway: this block's expert 5
    /// and the one above's expert 5 are different weights that share a
    /// number, so anything reasoning about experts as *storage* — which of
    /// them is resident, which would have to be read back — has to be able
    /// to tell them apart. See [`crate::residency`].
    pub layer: usize,
    /// Where to read the routed experts from instead of the mapping, when a
    /// cache for them was asked for. Shared by every layer of the model.
    pub resident: Option<std::sync::Arc<crate::experts::Resident>>,
}

impl Moe {
    /// Read a routed layer: the router's own matrix, the balancing bias if
    /// this family has one, the experts, and the shared expert if there is
    /// one.
    ///
    /// Every family that reaches here spells the routed weights identically —
    /// `mlp.gate.weight` and `mlp.experts.N.{gate,up,down}_proj.weight` — so
    /// what tells them apart is what is *around* them: Qwen3 has neither the
    /// bias nor a shared expert and asks for neither, DeepSeek has both, and
    /// Qwen3-Next has a shared expert with a mind of its own.
    pub fn load(src: &dyn Source, prefix: &str, router: &Router, shared: Shared) -> Res<Self> {
        Ok(Moe {
            router: router.clone(),
            gate: src.matrix(&format!("{prefix}.gate.weight"))?,
            bias: src.try_vector(&format!("{prefix}.gate.e_score_correction_bias")),
            experts: (0..router.n_experts)
                .map(|e| Ffn::load(src, &format!("{prefix}.experts.{e}")))
                .collect::<Res<Vec<_>>>()?,
            shared: SharedExpert::load(src, prefix, shared)?,
            layer: layer_of(prefix),
            resident: src.experts(),
        })
    }

    /// The router's logits for one token and, when this family has a shared
    /// expert, that expert up to its down projection -- as one section.
    ///
    /// `None` when a weight is not quantised, and the caller does both the
    /// ordinary way.
    fn early(&self, x: &[f32]) -> Option<(Vec<f32>, Option<Pending<'_>>)> {
        let mut jobs: Vec<(&Weight, &[f32])> = vec![(&self.gate, x)];
        if let Some(s) = &self.shared {
            jobs.push((&s.ffn.gate, x));
            jobs.push((&s.ffn.up, x));
            if let Some(gate) = &s.gate {
                jobs.push((gate, x));
            }
        }
        let mut ys = crate::quant::matvec_many(&jobs)?.into_iter();
        let logits = ys.next()?;
        let shared = match &self.shared {
            None => None,
            Some(s) => {
                let mut hidden = ys.next()?;
                swiglu_inplace(&mut hidden, &ys.next()?);
                let score = match &s.gate {
                    Some(_) => Some(ys.next()?[0]),
                    None => None,
                };
                Some(Pending { down: &s.ffn.down, hidden, score })
            }
        };
        Some((logits, shared))
    }

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
        // One token: the router and the shared expert read the same input
        // and neither needs the other, so they are one parallel section.
        let (logits, mut shared) = match (m == 1 && together()).then(|| self.early(hs)).flatten() {
            Some((logits, shared)) => (logits, shared),
            None => (self.gate.matmul_bt(hs, m, None), None),
        };

        let mut picks = Vec::with_capacity(router.top_k);
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        // Off unless `KVAD_EXPERT_TRACE` is set, and a branch per token
        // when it is not. The router is the only place that knows which
        // experts were wanted, so it is the only place to ask.
        let mut trace = residency::Trace::start(m);
        for i in 0..m {
            router
                .route(&logits[i * n..(i + 1) * n], self.bias.as_deref(), &mut picks);
            trace.token(&picks);
            for &(expert, weight) in &picks {
                by_expert[expert].push((i, weight));
            }
        }
        trace.finish(self.layer, n, self.experts.first().map_or(0, Ffn::bytes));

        let mut out = match &self.shared {
            // Waiting for its down projection, which `mix` runs with the
            // routed experts' and lands before any of theirs is added.
            Some(_) if shared.is_some() => vec![0.0f32; m * e],
            // Every token, so it is one batched pass and needs no grouping.
            Some(expert) => expert.run(hs, m),
            None => vec![0.0f32; m * e],
        };
        let chosen: Vec<usize> = (0..n).filter(|&x| !by_expert[x].is_empty()).collect();
        let mut done = vec![false; n];
        if let Some(resident) = &self.resident {
            // The same arithmetic on the same bytes, read by us rather than
            // paged in. A weight that cannot be rebased runs as mapped.
            let read = resident.with(self.layer, &chosen, |group| {
                let views: Vec<Option<Ffn>> =
                    group.iter().map(|&(x, slab, lo)| self.experts[x].rebased(lo, slab)).collect();
                let pairs: Vec<(usize, &Ffn)> = group
                    .iter()
                    .zip(&views)
                    .map(|(&(x, ..), view)| match view {
                        Some(ffn) => (x, ffn),
                        None => {
                            resident.fell_back();
                            (x, &self.experts[x])
                        }
                    })
                    .collect();
                mix(hs, m, e, &by_expert, &pairs, &mut shared, &mut out);
                for &(x, ..) in group {
                    done[x] = true;
                }
            });
            // A failed read costs speed and not the answer: whatever it did
            // not reach runs from the mapping below.
            if let Err(err) = read {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| eprintln!("expert cache: falling back to the mapping: {err}"));
            }
        }
        let rest: Vec<(usize, &Ffn)> =
            chosen.iter().filter(|&&x| !done[x]).map(|&x| (x, &self.experts[x])).collect();
        mix(hs, m, e, &by_expert, &rest, &mut shared, &mut out);
        debug_assert!(shared.is_none(), "the shared expert never landed");
        out
    }
}


/// A token's shared expert with only its down projection left to run.
///
/// Carried into [`mix`] so the down projection can share a section with the
/// routed experts' -- and so that, whichever way `mix` runs, the shared
/// output is what `out` starts as, before any routed expert is added.
struct Pending<'a> {
    down: &'a Weight,
    hidden: Vec<f32>,
    /// Qwen3-Next's per-token gate, before its sigmoid.
    score: Option<f32>,
}

impl Pending<'_> {
    /// Put the finished shared expert in `out`, as `SharedExpert::run` would.
    fn land(self, y: Vec<f32>, out: &mut [f32]) {
        out.copy_from_slice(&y);
        if let Some(score) = self.score {
            let g = 1.0 / (1.0 + (-score).exp());
            for o in out.iter_mut() {
                *o *= g;
            }
        }
    }
}

/// Run `group` -- each expert with the weights to run it from -- and add
/// each one's output into `out`, weighted, in the order given. `shared`, if
/// it is still waiting, lands first.
///
/// The order is the answer's: floating-point addition does not reassociate,
/// so the experts are summed in the order the unbatched path always used,
/// after the shared expert, as they always were.
fn mix(
    hs: &[f32],
    m: usize,
    e: usize,
    by_expert: &[Vec<(usize, f32)>],
    group: &[(usize, &Ffn)],
    shared: &mut Option<Pending<'_>>,
    out: &mut [f32],
) {
    if group.is_empty() {
        if let Some(p) = shared.take() {
            let y = p.down.matvec_bt(&p.hidden, None);
            p.land(y, out);
        }
        return;
    }
    if m == 1 && together() {
        // Every chosen expert's reads started at once, before any thread
        // faults on one. Batching without this is a bet that the experts
        // are already resident: measured cold, it lost to running them one
        // at a time (8.8 against 11.8 tok/s on Qwen3-30B at q8), and with it
        // it won (13.8). A no-op for experts read into slabs.
        if hint() {
            for &(_, ffn) in group {
                ffn.gate.will_need();
                ffn.up.will_need();
                ffn.down.will_need();
            }
        }
        let tail = shared.as_ref().map(|p| (p.down, p.hidden.as_slice()));
        if let Some((ys, tail)) = decode_together(hs, group, tail) {
            if let (Some(p), Some(y)) = (shared.take(), tail) {
                p.land(y, out);
            }
            for (&(expert, _), y) in group.iter().zip(ys) {
                // One token, which chose each of these experts exactly once.
                let weight = by_expert[expert][0].1;
                for (o, v) in out.iter_mut().zip(&y) {
                    *o += weight * v;
                }
            }
            return;
        }
    }
    if let Some(p) = shared.take() {
        let y = p.down.matvec_bt(&p.hidden, None);
        p.land(y, out);
    }
    let mut rows = Vec::with_capacity(m * e);
    for &(expert, ffn) in group {
        let tokens = &by_expert[expert];
        rows.clear();
        for &(i, _) in tokens {
            rows.extend_from_slice(&hs[i * e..(i + 1) * e]);
        }
        let y = ffn.run(&rows, tokens.len());
        for (j, &(i, weight)) in tokens.iter().enumerate() {
            for (o, v) in out[i * e..(i + 1) * e].iter_mut().zip(&y[j * e..(j + 1) * e]) {
                *o += weight * v;
            }
        }
    }
}

/// One token through all of its experts at once: every gate and up
/// projection as one parallel section, then every down projection as
/// another. See [`crate::quant::matvec_many`].
///
/// Through [`Ffn::run`] that was three sections an expert -- thirty a layer
/// on Qwen3-Next, each a 0.66 MB matrix split fourteen ways -- and the
/// profile of that was a thread pool mostly yielding. Here it is two.
/// `None` when the weights are not quantised; the caller runs them singly.
///
/// `tail` is one more down projection to run in the second section -- the
/// shared expert's -- and its result comes back beside the experts'.
#[allow(clippy::type_complexity)]
fn decode_together(
    x: &[f32],
    group: &[(usize, &Ffn)],
    tail: Option<(&Weight, &[f32])>,
) -> Option<(Vec<Vec<f32>>, Option<Vec<f32>>)> {
    let mut jobs = Vec::with_capacity(2 * group.len());
    for &(_, ffn) in group {
        jobs.push((&ffn.gate, x));
        jobs.push((&ffn.up, x));
    }
    let mut both = crate::quant::matvec_many(&jobs)?.into_iter();
    let mut hidden = Vec::with_capacity(group.len());
    while let (Some(mut gate), Some(up)) = (both.next(), both.next()) {
        swiglu_inplace(&mut gate, &up);
        hidden.push(gate);
    }
    let mut jobs: Vec<(&Weight, &[f32])> =
        group.iter().zip(&hidden).map(|(&(_, ffn), h)| (&ffn.down, h.as_slice())).collect();
    jobs.extend(tail);
    let mut ys = crate::quant::matvec_many(&jobs)?;
    let tail = tail.and_then(|_| ys.pop());
    Some((ys, tail))
}

/// Whether to tell the kernel which experts a token chose before running
/// them. On unless `KVAD_EXPERT_WILLNEED=0`, which is there for the A/B.
fn hint() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("KVAD_EXPERT_WILLNEED").as_deref(), Ok("0") | Ok("false")))
}

/// Whether a token's experts run as one batch. `KVAD_EXPERTS_ONE_BY_ONE=1`
/// puts back the path this replaced, so the two are one flag apart.
fn together() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(std::env::var("KVAD_EXPERTS_ONE_BY_ONE").as_deref(), Ok("1") | Ok("true"))
    })
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
                    + moe.shared.as_ref().map_or(0, SharedExpert::param_count)
            }
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mlp::Dense(ffn) => ffn.bytes(),
            Mlp::Moe(moe) => {
                moe.gate.bytes()
                    + moe.experts.iter().map(Ffn::bytes).sum::<usize>()
                    + moe.shared.as_ref().map_or(0, SharedExpert::bytes)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// What a config says
// ---------------------------------------------------------------------------

/// The config reading, which is the whole of what separates one family's
/// mixture from another's.
///
/// Worth testing here rather than through a model, because these are the
/// questions a checkpoint answers wrongly in silence: a layer that should have
/// routed and ran a dense MLP instead loads, runs, and talks nonsense.
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(v: serde_json::Value) -> Json {
        Json::new(v)
    }

    /// One token through a whole mixture -- router, shared expert and all --
    /// against the old path written out longhand: route, run the shared
    /// expert, then add each chosen expert's output in expert order. Every
    /// kind of shared expert, because each lands in a different way.
    #[test]
    fn a_decoded_token_equals_the_mixture_run_the_old_way() {
        use crate::quant::Precision;
        use crate::tensor::Tensor;
        let mut rng = nervus::rng::Rng::new(29);
        let mut mat = |rows: usize, cols: usize| {
            Tensor::new(rows, cols, (0..rows * cols).map(|_| rng.normal() * 0.1).collect())
        };
        let (hidden, width, n) = (64, 32, 6);
        for (kind, shared) in [("none", Shared::None), ("fused", Shared::Fused(2)), ("gated", Shared::Gated(64))] {
            let precision = Precision::Q8;
            let mut ffn = |w: usize| Ffn {
                gate: Weight::quantize(mat(w, hidden), precision),
                up: Weight::quantize(mat(w, hidden), precision),
                down: Weight::quantize(mat(hidden, w), precision),
            };
            let experts: Vec<Ffn> = (0..n).map(|_| ffn(width)).collect();
            let shared_ffn = match shared {
                Shared::None => None,
                Shared::Fused(k) => Some(ffn(width * k)),
                Shared::Gated(w) => Some(ffn(w)),
            };
            let router = Router::read(&config(json!({
                "n_routed_experts": n, "num_experts_per_tok": 2, "norm_topk_prob": true
            })))
            .unwrap();
            let moe = Moe {
                router,
                gate: Weight::quantize(mat(n, hidden), precision),
                bias: None,
                experts,
                shared: shared_ffn.map(|ffn| SharedExpert {
                    ffn,
                    gate: matches!(shared, Shared::Gated(_))
                        .then(|| Weight::quantize(mat(1, hidden), precision)),
                }),
                layer: 0,
                resident: None,
            };
            let x = mat(1, hidden).data;

            let logits = moe.gate.matmul_bt(&x, 1, None);
            let mut picks = Vec::new();
            moe.router.route(&logits, None, &mut picks);
            picks.sort_by_key(|&(expert, _)| expert);
            let mut want = match &moe.shared {
                Some(s) => s.run(&x, 1),
                None => vec![0.0; hidden],
            };
            for &(expert, weight) in &picks {
                for (o, v) in want.iter_mut().zip(&moe.experts[expert].run(&x, 1)) {
                    *o += weight * v;
                }
            }
            // Not vacuous: the folded path is the one that ran, and it carried
            // the shared expert out with it where there is one.
            let (_, pending) = moe
                .early(&x)
                .unwrap_or_else(|| panic!("{kind}: the router and shared expert did not batch"));
            assert_eq!(pending.is_some(), moe.shared.is_some(), "shared expert: {kind}");
            assert_eq!(moe.run(&x, 1, hidden), want, "shared expert: {kind}");
        }
    }

    /// A token's experts run as one batch give each expert's output exactly
    /// as it came from running that expert alone -- the property that lets
    /// the batched decode path replace the old one without changing a token.
    #[test]
    fn a_tokens_experts_together_equal_them_one_by_one() {
        use crate::quant::Precision;
        use crate::tensor::Tensor;
        let mut rng = nervus::rng::Rng::new(11);
        let mut mat = |rows: usize, cols: usize| {
            Tensor::new(rows, cols, (0..rows * cols).map(|_| rng.normal() * 0.1).collect())
        };
        for precision in [Precision::Q8, Precision::Q4] {
            let (hidden, width) = (64, 96);
            let experts: Vec<Ffn> = (0..3)
                .map(|_| Ffn {
                    gate: Weight::quantize(mat(width, hidden), precision),
                    up: Weight::quantize(mat(width, hidden), precision),
                    down: Weight::quantize(mat(hidden, width), precision),
                })
                .collect();
            let x = mat(1, hidden).data;
            let group: Vec<(usize, &Ffn)> = experts.iter().enumerate().collect();
            let (together, _) = decode_together(&x, &group, None).expect("quantised experts batch");
            for (k, (ffn, got)) in experts.iter().zip(&together).enumerate() {
                assert_eq!(got, &ffn.run(&x, 1), "{precision}: expert {k} differs");
            }
        }
    }

    /// The same quantity, two spellings, and a dense model has neither.
    #[test]
    fn both_families_say_how_many_experts_they_have() {
        assert_eq!(Router::count(&config(json!({ "n_routed_experts": 64 }))), Some(64));
        assert_eq!(Router::count(&config(json!({ "num_experts": 128 }))), Some(128));
        assert_eq!(Router::count(&config(json!({ "hidden_size": 2048 }))), None);
    }

    /// DeepSeek asks `layer % freq` and Qwen3 asks `(layer + 1) % step`, so at
    /// a period of two they route on opposite layers.
    ///
    /// Both published families use a period of one, where the two rules agree
    /// and this distinction is invisible. That is exactly why it is written
    /// down: the first checkpoint to ship a period of two would otherwise run
    /// half its layers through the wrong feed-forward.
    #[test]
    fn qwen3_counts_layers_from_one_and_deepseek_from_zero() {
        let qwen = Layout::read(&config(json!({ "decoder_sparse_step": 2 })));
        let deep = Layout::read(&config(json!({ "moe_layer_freq": 2 })));
        let routes = |l: &Layout| (0..4).map(|i| l.is_moe(i, 8)).collect::<Vec<_>>();
        assert_eq!(routes(&qwen), [false, true, false, true]);
        assert_eq!(routes(&deep), [true, false, true, false]);
    }

    /// Qwen3 names its dense layers; DeepSeek counts them from the front. A
    /// layer named in `mlp_only_layers` is dense whatever the step says.
    #[test]
    fn the_two_ways_of_saying_a_layer_is_dense_both_work() {
        let named = Layout::read(&config(json!({
            "decoder_sparse_step": 1,
            "mlp_only_layers": [0, 2],
        })));
        let counted = Layout::read(&config(json!({ "first_k_dense_replace": 2 })));
        let routes = |l: &Layout| (0..4).map(|i| l.is_moe(i, 8)).collect::<Vec<_>>();
        assert_eq!(routes(&named), [false, true, false, true]);
        assert_eq!(routes(&counted), [false, false, true, true]);
    }

    /// The three answers to "what runs for every token", from the three
    /// families' own configs.
    ///
    /// Worth pinning because the failure is quiet in both directions. Reading
    /// Qwen3-Next's shared expert as DeepSeek's asks for `mlp.shared_experts`
    /// and fails loudly, which is fine — but reading it as *absent* would load
    /// a model that runs with a whole expert missing from every layer.
    #[test]
    fn each_family_says_what_runs_for_every_token() {
        let read = |v: serde_json::Value| Layout::read(&config(v)).shared;
        assert_eq!(read(json!({ "num_experts": 128 })), Shared::None);
        assert_eq!(read(json!({ "n_shared_experts": 2 })), Shared::Fused(2));
        assert_eq!(
            read(json!({ "shared_expert_intermediate_size": 512 })),
            Shared::Gated(512)
        );

        // DeepSeek's is `n` experts side by side in one matrix; Qwen3-Next's
        // names its own width and has nothing to multiply.
        assert_eq!(Shared::Fused(2).width(768), 1536);
        assert_eq!(Shared::Gated(512).width(768), 512);
        assert_eq!(Shared::None.width(768), 0);
    }

    /// Qwen3-30B-A3B's own routing: softmax over a flat 128, the best eight,
    /// and those eight renormalised to sum to one.
    ///
    /// `norm_topk_prob` is the difference that would be quietly survivable —
    /// unnormalised softmax weights over the top eight sum to rather less than
    /// one, so the mixture's whole output comes out scaled down and the model
    /// degrades instead of failing.
    #[test]
    fn qwen3s_chosen_weights_are_renormalised_to_sum_to_one() {
        let router = Router::read(&config(json!({
            "num_experts": 8,
            "num_experts_per_tok": 4,
            "norm_topk_prob": true,
        })))
        .unwrap();

        let logits = [0.0, 3.0, 1.0, -1.0, 2.0, 0.5, -2.0, 4.0];
        let mut picks = Vec::new();
        router.route(&logits, None, &mut picks);

        assert_eq!(picks.iter().map(|&(e, _)| e).collect::<Vec<_>>(), [7, 1, 4, 2]);
        let total: f32 = picks.iter().map(|&(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-6, "the four weights sum to {total}, not 1");
        // Renormalising is a common divisor, so the softmax's ratios survive
        // it: expert 7 beats expert 1 by one logit, which is a factor of e.
        let ratio = picks[0].1 / picks[1].1;
        assert!((ratio - std::f32::consts::E).abs() < 1e-5, "ratio {ratio}");
    }

    /// The same config without `norm_topk_prob`, which is DeepSeek V2-Lite's,
    /// leaves the softmax weights as they are.
    #[test]
    fn without_normalisation_the_weights_are_the_softmaxs_own() {
        let router = Router::read(&config(json!({
            "n_routed_experts": 8,
            "num_experts_per_tok": 4,
        })))
        .unwrap();

        let logits = [0.0, 3.0, 1.0, -1.0, 2.0, 0.5, -2.0, 4.0];
        let mut picks = Vec::new();
        router.route(&logits, None, &mut picks);

        let mut softmax = logits.to_vec();
        softmax_inplace(&mut softmax);
        for &(expert, w) in &picks {
            assert!((w - softmax[expert]).abs() < 1e-6, "expert {expert}: {w}");
        }
    }
}
