//! Which machine holds which layers.
//!
//! One conversation runs a split model's stages one after another, so the
//! time a token takes is a sum:
//!
//! ```text
//!   Σ layers on a machine × what a layer costs there  +  a round trip for each worker
//! ```
//!
//! A sum, and not a maximum: the slowest machine does not set the pace, it
//! only costs what its own layers cost. That makes placing them simple.
//! Every layer goes on the fastest machine that still has room for it, and
//! a machine that ends up with nothing is left out, because all it could
//! add is a hop.
//!
//! Room is weights and cache both. A layer's keys and values at the context
//! asked for are not small — 655 KB a token for a 70B in f32, 5 GB at 8k —
//! and a plan that fits the weights and not the conversation is a plan
//! that fails an hour in.
//!
//! The numbers here are estimates made before anything is downloaded: a
//! layer's weights are the model's parameters, less the embedding and the
//! head, divided by its layers. For a model whose layers differ in size it
//! is wrong by however much they differ, which is why what is left over on
//! each machine is printed beside what is used.

use crate::model::Spec;
use crate::quant::{Precision, Weight};
use crate::tensor::Tensor;
use std::ops::Range;
use std::time::Instant;

/// A machine layers could go on.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    /// Bytes it would give a model: see [`crate::machine::usable_memory`].
    pub memory: u64,
    /// Milliseconds one layer of this model takes there, measured.
    pub layer_ms: f64,
    /// Milliseconds there and back for a decode hop. Nothing, for the
    /// machine the tokens are on.
    pub hop_ms: f64,
}

/// What the pieces of a model weigh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    pub n_layer: usize,
    /// The embedding, the final norm and the head: the coordinator's,
    /// whatever else it holds.
    pub ends: u64,
    /// One layer's weights.
    pub layer: u64,
    /// One layer's cache at the context asked for.
    pub cache: u64,
}

impl Sizes {
    /// From a model's shape and its parameter count, at a precision and for
    /// conversations of up to `context` tokens.
    pub fn of(spec: &Spec, params: u64, precision: Precision, context: usize) -> Sizes {
        // The table, and the head where it is not the table again.
        let table = (spec.vocab_size * spec.n_embd) as u64;
        let ends = (table * if spec.tie_embeddings { 1 } else { 2 }).min(params);
        let n_layer = spec.n_layer.max(1);
        Sizes {
            n_layer: spec.n_layer,
            ends: precision.weight_bytes(ends),
            layer: precision.weight_bytes((params - ends) / n_layer as u64),
            cache: (spec.cache.bytes(spec.n_layer, context) / n_layer) as u64,
        }
    }

    /// Bytes `layers` layers take on a machine, the ends too if it has them.
    pub fn of_share(&self, layers: usize, ends: bool) -> u64 {
        layers as u64 * (self.layer + self.cache) + if ends { self.ends } else { 0 }
    }

    /// The whole model, on one machine.
    pub fn whole(&self) -> u64 {
        self.of_share(self.n_layer, true)
    }
}

/// One machine's part of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct Share {
    pub layers: Range<usize>,
    /// Bytes of weights, the ends among them on the first machine.
    pub weights: u64,
    /// Bytes of cache at the context asked for.
    pub cache: u64,
    /// What that leaves of the machine's memory. Negative is a share that
    /// does not fit, which a plan made by hand can have.
    pub left: i64,
}

/// Layers on machines, the coordinator first.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub shares: Vec<Share>,
    /// What a token should take: the sum this module opens with.
    pub ms_per_token: f64,
}

impl Plan {
    /// As `--split` is written.
    pub fn split(&self) -> String {
        self.shares.iter().map(|s| format!("{}..{}", s.layers.start, s.layers.end)).collect::<Vec<_>>().join(",")
    }
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

/// The plan that `counts` layers a machine makes, in the order the
/// machines are given.
fn laid_out(nodes: &[Node], sizes: &Sizes, counts: &[usize]) -> Plan {
    let (mut next, mut ms) = (0, 0.0);
    let mut shares = Vec::with_capacity(nodes.len());
    for (i, (node, &count)) in nodes.iter().zip(counts).enumerate() {
        let weights = count as u64 * sizes.layer + if i == 0 { sizes.ends } else { 0 };
        let cache = count as u64 * sizes.cache;
        shares.push(Share { layers: next..next + count, weights, cache, left: node.memory as i64 - (weights + cache) as i64 });
        ms += count as f64 * node.layer_ms + if count > 0 { node.hop_ms } else { 0.0 };
        next += count;
    }
    Plan { shares, ms_per_token: ms }
}

/// Place a model's layers: each on the fastest machine with room left,
/// and the ends on the first machine, which is the one with the tokens.
///
/// `nodes[0]` is the coordinator. The plan's shares are in the order of
/// `nodes`; a machine given nothing has an empty run of layers.
pub fn place(nodes: &[Node], sizes: &Sizes) -> Result<Plan, String> {
    let Some(here) = nodes.first() else { return Err("no machines to place a model on".into()) };
    if here.memory < sizes.ends {
        return Err(format!(
            "{} has {} for a model, and this one's embedding and output head are {}: they stay with the tokens, so this model wants a larger machine to run it from",
            here.name,
            gb(here.memory),
            gb(sizes.ends)
        ));
    }
    let a_layer = (sizes.layer + sizes.cache).max(1);
    let room = |i: usize| {
        let ends = if i == 0 { sizes.ends } else { 0 };
        (nodes[i].memory.saturating_sub(ends) / a_layer) as usize
    };

    // Fastest first. A tie goes to the machine named first, and the
    // coordinator is named first of all: a layer here costs no hop.
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by(|&a, &b| nodes[a].layer_ms.total_cmp(&nodes[b].layer_ms).then(a.cmp(&b)));

    let mut counts = vec![0; nodes.len()];
    let mut left = sizes.n_layer;
    for i in order {
        counts[i] = room(i).min(left);
        left -= counts[i];
    }
    if left > 0 {
        let held: u64 = nodes.iter().map(|n| n.memory).sum();
        return Err(format!(
            "does not fit: the model is {} at this precision and context, and {} machine(s) have {} for it between them, which leaves {left} of its {} layers with nowhere to go",
            gb(sizes.whole()),
            nodes.len(),
            gb(held),
            sizes.n_layer
        ));
    }
    Ok(laid_out(nodes, sizes, &counts))
}

/// A split made by hand, held to the same memory: the plan it is, or which
/// machine it asks too much of.
pub fn check(nodes: &[Node], sizes: &Sizes, split: &[Range<usize>]) -> Result<Plan, String> {
    let counts: Vec<usize> = split.iter().map(|layers| layers.len()).collect();
    let plan = laid_out(nodes, sizes, &counts);
    for (node, share) in nodes.iter().zip(&plan.shares) {
        if share.left < 0 {
            return Err(format!(
                "{} is given layers {}..{}, which is {} of weights and {} of cache at this context, and it has {} for a model",
                node.name,
                share.layers.start,
                share.layers.end,
                gb(share.weights),
                gb(share.cache),
                gb(node.memory)
            ));
        }
    }
    Ok(plan)
}

/// The plan as a table, a line a machine, for whoever asked.
pub fn table(nodes: &[Node], plan: &Plan) -> Vec<String> {
    let width = nodes.iter().map(|n| n.name.len()).max().unwrap_or(0).max(7);
    let mut lines = vec![format!("{:<width$}  {:<9} {:>9} {:>9} {:>9} {:>9}  {:>10}", "MACHINE", "LAYERS", "WEIGHTS", "CACHE", "OF", "LEFT", "MS A LAYER")];
    for (node, share) in nodes.iter().zip(&plan.shares) {
        let layers = match share.layers.is_empty() {
            true => "none".to_string(),
            false => format!("{}..{}", share.layers.start, share.layers.end),
        };
        let left = if share.left < 0 { format!("-{}", gb(share.left.unsigned_abs())) } else { gb(share.left as u64) };
        lines.push(format!(
            "{:<width$}  {layers:<9} {:>9} {:>9} {:>9} {left:>9}  {:>10.3}",
            node.name,
            gb(share.weights),
            gb(share.cache),
            gb(node.memory),
            node.layer_ms
        ));
    }
    let hops = nodes.iter().zip(&plan.shares).skip(1).filter(|(_, s)| !s.layers.is_empty()).count();
    lines.push(format!(
        "about {:.1} ms a token, {:.1} tok/s: the layers' own time and {hops} round trip(s). An estimate; the first measurement is worth more.",
        plan.ms_per_token,
        1e3 / plan.ms_per_token.max(1e-9)
    ));
    lines
}

/// How long one layer of a model this shape takes on this machine, in
/// milliseconds: its seven matrix products for one token, timed.
///
/// Not a layer — there is no model yet, which is the point of asking
/// before one is loaded — but the arithmetic a layer is made of, at the
/// widths this model has and the precision it will be loaded at: four
/// products the width of the model for attention, three the width of the
/// MLP. A wide MLP is timed at a slice of its rows and scaled, so that
/// asking a small machine about a large model does not take the memory the
/// question is about.
pub fn bench(n_embd: usize, intermediate: usize, precision: Precision) -> f64 {
    let (n_embd, intermediate) = (n_embd.max(32), intermediate.max(32));
    let matrix = |rows: usize| {
        // Any numbers will do: a product costs the same whatever it is of.
        let mut state = 0x6b76_6164_u64;
        let data = (0..rows * n_embd)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect();
        Weight::quantize(Tensor::new(rows, n_embd, data), precision)
    };
    let (wide_rows, square_rows) = (intermediate.min(4096), n_embd.min(4096));
    let (wide, square) = (matrix(wide_rows), matrix(square_rows));
    let x = vec![0.5f32; n_embd];
    let best = |w: &Weight| {
        (0..12)
            .map(|_| {
                let started = Instant::now();
                std::hint::black_box(w.matvec_bt(std::hint::black_box(&x), None));
                started.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::INFINITY, f64::min)
    };
    let pool = crate::model::pool();
    pool.install(|| 3.0 * best(&wide) * (intermediate as f64 / wide_rows as f64) + 4.0 * best(&square) * (n_embd as f64 / square_rows as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn node(name: &str, gb: u64, layer_ms: f64, hop_ms: f64) -> Node {
        Node { name: name.into(), memory: gb * GB, layer_ms, hop_ms }
    }

    /// A 70B's shape in round numbers: 80 layers of 0.8 GB, 5 GB of ends,
    /// and a tenth of a gigabyte of cache a layer.
    fn seventy() -> Sizes {
        Sizes { n_layer: 80, ends: 5 * GB, layer: 800_000_000, cache: 100_000_000 }
    }

    fn runs(plan: &Plan) -> Vec<Range<usize>> {
        plan.shares.iter().map(|s| s.layers.clone()).collect()
    }

    #[test]
    fn a_model_that_fits_here_stays_here() {
        let nodes = [node("here", 96, 1.0, 0.0), node("air", 6, 2.0, 2.5)];
        let plan = place(&nodes, &seventy()).unwrap();
        assert_eq!(runs(&plan), [0..80, 80..80], "a worker is a hop, and a hop buys nothing here");
        assert_eq!(plan.ms_per_token, 80.0);
        assert_eq!(plan.shares[0].weights, 5 * GB + 80 * 800_000_000);
        assert_eq!(plan.shares[0].left, (96 - 5 - 72) as i64 * GB as i64);
        assert_eq!(plan.split(), "0..80,80..80");
    }

    #[test]
    fn layers_go_to_the_fastest_machine_with_room_and_then_the_next() {
        // 48 GB here: 5 for the ends leaves room for 47 layers of 0.9.
        let nodes = [node("here", 48, 1.0, 0.0), node("studio", 64, 0.5, 0.3), node("mini", 16, 2.0, 0.3)];
        let plan = place(&nodes, &seventy()).unwrap();
        // The studio is fastest and takes all it can, 71; here takes the
        // other 9; the mini is not needed and is not used.
        assert_eq!(runs(&plan), [0..9, 9..80, 80..80]);
        assert!((plan.ms_per_token - (9.0 * 1.0 + 71.0 * 0.5 + 0.3)).abs() < 1e-9);

        // With a slower studio the order turns over: here fills first.
        let nodes = [node("here", 48, 1.0, 0.0), node("studio", 64, 1.5, 0.3), node("mini", 16, 2.0, 0.3)];
        assert_eq!(runs(&place(&nodes, &seventy()).unwrap()), [0..47, 47..80, 80..80]);
        // And a tie goes to the machine with the tokens.
        let nodes = [node("here", 48, 1.0, 0.0), node("twin", 48, 1.0, 0.3)];
        assert_eq!(runs(&place(&nodes, &seventy()).unwrap()), [0..47, 47..80]);
    }

    #[test]
    fn a_model_too_large_for_all_of_them_is_said_to_be() {
        let nodes = [node("here", 48, 1.0, 0.0), node("air", 6, 2.0, 2.5)];
        let why = place(&nodes, &seventy()).unwrap_err();
        assert!(why.contains("does not fit") && why.contains("77.0 GB") && why.contains("54.0 GB") && why.contains("27 of its 80 layers"), "{why}");
        // The ends stay with the tokens, so a small machine cannot run a
        // large model from however many large ones.
        let nodes = [node("air", 4, 2.0, 0.0), node("studio", 512, 0.5, 0.3)];
        let why = place(&nodes, &seventy()).unwrap_err();
        assert!(why.contains("air has 4.0 GB") && why.contains("embedding and output head are 5.0 GB"), "{why}");
        assert!(place(&[], &seventy()).is_err());
    }

    #[test]
    fn a_split_by_hand_is_held_to_the_same_memory() {
        let nodes = [node("here", 48, 1.0, 0.0), node("air", 6, 2.0, 2.5)];
        let small = Sizes { n_layer: 30, ends: GB / 10, layer: 10_000_000, cache: 5_000_000 };
        let plan = check(&nodes, &small, &[0..5, 5..30]).unwrap();
        assert_eq!(runs(&plan), [0..5, 5..30]);
        assert!((plan.ms_per_token - (5.0 + 50.0 + 2.5)).abs() < 1e-9, "what the operator chose, at what it costs");
        // Everything here, and the worker's round trip is not paid.
        assert_eq!(check(&nodes, &small, &[0..30, 30..30]).unwrap().ms_per_token, 30.0);

        let why = check(&nodes, &seventy(), &[0..70, 70..80]).unwrap_err();
        assert!(why.contains("here is given layers 0..70") && why.contains("48.0 GB"), "{why}");
        let why = check(&nodes, &seventy(), &[0..40, 40..80]).unwrap_err();
        assert!(why.contains("air is given layers 40..80") && why.contains("32.0 GB of weights and 4.0 GB of cache"), "{why}");
    }

    #[test]
    fn sizes_come_from_a_models_shape() {
        use crate::model::{Arch, CacheLayout, CacheShape, Json};
        let spec = Spec {
            arch: Arch::require("llama"),
            n_layer: 28,
            n_head: 12,
            n_kv_head: 2,
            n_embd: 1536,
            head_dim: 128,
            n_ctx: 32768,
            vocab_size: 151_936,
            intermediate: 8960,
            eps: 1e-6,
            rope_theta: 1e6,
            tie_embeddings: true,
            cache: CacheLayout::uniform(CacheShape::kv(256, 256)),
            config: Json::default(),
        };
        // Qwen2.5-1.5B: 1.54 billion parameters, 233 million of them the table.
        let sizes = Sizes::of(&spec, 1_543_714_304, Precision::F32, 8192);
        assert_eq!(sizes.ends, 151_936 * 1536 * 4);
        assert_eq!(sizes.layer, (1_543_714_304 - 151_936 * 1536) / 28 * 4);
        assert_eq!(sizes.cache, 8192 * 512 * 4, "keys and values for every token of the context, one layer's");
        assert_eq!(sizes.whole(), sizes.ends + 28 * (sizes.layer + sizes.cache));
        // Untied, the head is a second table; and quantised, all of it shrinks.
        let untied = Sizes::of(&Spec { tie_embeddings: false, ..spec.clone() }, 1_543_714_304, Precision::F32, 8192);
        assert_eq!(untied.ends, 2 * sizes.ends);
        assert!(Sizes::of(&spec, 1_543_714_304, Precision::Q8, 8192).layer < sizes.layer / 3);
    }

    #[test]
    fn a_plan_is_printed_a_machine_a_line() {
        let nodes = [node("here", 48, 1.0, 0.0), node("studio", 64, 0.5, 0.3), node("mini", 16, 2.0, 0.3)];
        let plan = place(&nodes, &seventy()).unwrap();
        let lines = table(&nodes, &plan);
        assert_eq!(lines.len(), 5);
        assert!(lines[1].starts_with("here ") && lines[1].contains("0..9") && lines[1].contains("48.0 GB"), "{}", lines[1]);
        assert!(lines[2].contains("9..80") && lines[3].contains("none"), "{lines:?}");
        assert!(lines[4].contains("about 44.8 ms a token") && lines[4].contains("1 round trip(s)"), "{}", lines[4]);
    }

    /// A time comes back, for every precision and for widths a slice is
    /// taken of. Which of two shapes is slower is not asserted: beside two
    /// hundred other tests it is the machine that is being timed.
    #[test]
    fn a_layer_is_timed() {
        for precision in [Precision::F32, Precision::Q8, Precision::Q4] {
            let ms = bench(256, 1024, precision);
            assert!(ms > 0.0 && ms.is_finite(), "{precision}: {ms}");
        }
        let wide = bench(64, 9000, Precision::Q8);
        assert!(wide > 0.0 && wide.is_finite(), "{wide}");
        assert!(bench(1, 1, Precision::F32).is_finite(), "nothing so small that it divides by nothing");
    }
}
