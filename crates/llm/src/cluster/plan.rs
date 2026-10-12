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
//! # A prompt is a different question
//!
//! Reading a prompt is not a sum. A prompt goes down the pipeline in
//! chunks, and every stage works on a different chunk at once, so what a
//! chunk costs is the *slowest* stage and not all of them:
//!
//! ```text
//!   max over machines of ( its layers × what a layer costs there for a chunk  +  its round trip )
//! ```
//!
//! That wants the opposite of the reply. A reply is quickest with every
//! layer it can hold on the fastest machine; a prompt is quickest with the
//! stages *equal*, which gives the slower machine fewer layers and not
//! none. Measured on an M5 Pro and an M2 Air: twenty layers on the first
//! and ten on the second read a prompt 1.4 times as fast as the M5 Pro
//! alone, and write the reply a fifth slower.
//!
//! So a plan is made for a [`Mix`]: so many tokens of prompt for so many
//! of reply, and the placing that makes that much reading and writing
//! take least. All reply is where it starts, because that is the answer
//! that never takes a model off the machine it fits on.
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
    /// Milliseconds one layer takes there for one token of a prompt, which
    /// is read a chunk at a time and costs a fraction of a token written.
    pub prompt_layer_ms: f64,
    /// Milliseconds there and back for a chunk of a prompt.
    pub prompt_hop_ms: f64,
}

/// What a plan is for: so much prompt read for so much reply written.
///
/// Only the proportion matters. `1000:100` is a long question and a short
/// answer; all reply is a model asked to write and write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mix {
    pub prompt: f64,
    pub reply: f64,
}

impl Mix {
    /// Quickest tokens written, whatever reading costs: what a plan is
    /// for unless somebody says.
    pub const REPLY: Mix = Mix { prompt: 0.0, reply: 1.0 };
    /// Quickest prompt read, whatever writing costs.
    pub const PROMPT: Mix = Mix { prompt: 1.0, reply: 0.0 };

    /// `reply`, `prompt`, or `N:M` for N tokens read to every M written.
    pub fn parse(text: &str) -> Result<Mix, String> {
        let wrong = || format!("`{text}` is not what to place layers for: reply, prompt, or tokens read to tokens written, as 2000:200");
        match text.trim() {
            "reply" => Ok(Mix::REPLY),
            "prompt" => Ok(Mix::PROMPT),
            other => {
                let (prompt, reply) = other.split_once(':').ok_or_else(wrong)?;
                let (prompt, reply): (f64, f64) = (prompt.trim().parse().map_err(|_| wrong())?, reply.trim().parse().map_err(|_| wrong())?);
                match prompt >= 0.0 && reply >= 0.0 && prompt + reply > 0.0 && (prompt + reply).is_finite() {
                    true => Ok(Mix { prompt, reply }),
                    false => Err(wrong()),
                }
            }
        }
    }

    /// Milliseconds a plan takes over this much reading and writing.
    pub fn cost(&self, plan: &Plan) -> f64 {
        self.prompt * plan.prompt_ms_per_token + self.reply * plan.ms_per_token
    }
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
    /// What a token of a prompt should take to read: the slowest stage's
    /// share of a chunk.
    pub prompt_ms_per_token: f64,
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
    let (mut next, mut ms, mut slowest) = (0, 0.0, 0.0f64);
    let chunk = crate::model::prefill_chunk() as f64;
    let mut shares = Vec::with_capacity(nodes.len());
    for (i, (node, &count)) in nodes.iter().zip(counts).enumerate() {
        let weights = count as u64 * sizes.layer + if i == 0 { sizes.ends } else { 0 };
        let cache = count as u64 * sizes.cache;
        shares.push(Share { layers: next..next + count, weights, cache, left: node.memory as i64 - (weights + cache) as i64 });
        if count > 0 {
            ms += count as f64 * node.layer_ms + node.hop_ms;
            // A stage's share of a chunk, a token: its layers, and its
            // round trip spread over the chunk's tokens.
            slowest = slowest.max(count as f64 * node.prompt_layer_ms + node.prompt_hop_ms / chunk);
        }
        next += count;
    }
    Plan { shares, ms_per_token: ms, prompt_ms_per_token: slowest }
}

/// Place a model's layers: each on the fastest machine with room left,
/// and the ends on the first machine, which is the one with the tokens.
///
/// `nodes[0]` is the coordinator. The plan's shares are in the order of
/// `nodes`; a machine given nothing has an empty run of layers.
pub fn place(nodes: &[Node], sizes: &Sizes, mix: Mix) -> Result<Plan, String> {
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
    let mut fastest: Vec<usize> = (0..nodes.len()).collect();
    fastest.sort_by(|&a, &b| nodes[a].layer_ms.total_cmp(&nodes[b].layer_ms).then(a.cmp(&b)));
    let filled = |order: &[usize]| {
        let mut counts = vec![0; nodes.len()];
        let mut left = sizes.n_layer;
        for &i in order {
            counts[i] = room(i).min(left);
            left -= counts[i];
        }
        (counts, left)
    };
    let (counts, left) = filled(&fastest);
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
    // A faster machine is only worth its round trip if its layers save
    // more than the trip costs. So the same filling with this machine
    // first is tried beside it: a worker that is a hair faster does not
    // take a model off the machine it fits on.
    let here_first: Vec<usize> = std::iter::once(0).chain(fastest.iter().copied().filter(|&i| i != 0)).collect();
    let mut starts = Vec::new();
    if let (counts, 0) = filled(&here_first) {
        starts.push(counts);
    }
    starts.push(counts);

    // From each of those, layers are moved between machines for as long
    // as moving some makes the mix quicker. For a reply nothing moves:
    // the fillings above are already the sum's least. For a prompt it is
    // how the stages come level, and several layers go at once where one
    // alone would only add a round trip and not yet pay for it.
    let rooms: Vec<usize> = (0..nodes.len()).map(room).collect();
    let cost = |counts: &[usize]| mix.cost(&laid_out(nodes, sizes, counts));
    let mut best: Option<(f64, Vec<usize>)> = None;
    for mut counts in starts {
        let mut now = cost(&counts);
        loop {
            let mut better: Option<(f64, Vec<usize>)> = None;
            for from in 0..nodes.len() {
                for to in (0..nodes.len()).filter(|&to| to != from) {
                    for moved in 1..=counts[from].min(rooms[to] - counts[to]) {
                        let mut tried = counts.clone();
                        tried[from] -= moved;
                        tried[to] += moved;
                        let costs = cost(&tried);
                        if costs < better.as_ref().map_or(now, |(least, _)| *least) - 1e-9 {
                            better = Some((costs, tried));
                        }
                    }
                }
            }
            match better {
                Some((costs, moved)) => (now, counts) = (costs, moved),
                None => break,
            }
        }
        // The first start is this machine's, and a tie stays with it.
        if best.as_ref().is_none_or(|(least, _)| now < *least - 1e-9) {
            best = Some((now, counts));
        }
    }
    let (_, counts) = best.expect("there is always the filling by speed");
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
    let mut lines =
        vec![format!("{:<width$}  {:<9} {:>9} {:>9} {:>9} {:>9}  {:>10} {:>9}", "MACHINE", "LAYERS", "WEIGHTS", "CACHE", "OF", "LEFT", "MS A LAYER", "READING")];
    let chunk = crate::model::prefill_chunk() as f64;
    for (node, share) in nodes.iter().zip(&plan.shares) {
        let layers = match share.layers.is_empty() {
            true => "none".to_string(),
            false => format!("{}..{}", share.layers.start, share.layers.end),
        };
        let left = if share.left < 0 { format!("-{}", gb(share.left.unsigned_abs())) } else { gb(share.left as u64) };
        lines.push(format!(
            "{:<width$}  {layers:<9} {:>9} {:>9} {:>9} {left:>9}  {:>10.3} {:>9.3}",
            node.name,
            gb(share.weights),
            gb(share.cache),
            gb(node.memory),
            node.layer_ms,
            node.prompt_layer_ms
        ));
    }
    let hops = nodes.iter().zip(&plan.shares).skip(1).filter(|(_, s)| !s.layers.is_empty()).count();
    lines.push(format!(
        "a reply at about {:.1} ms a token, {:.1} tok/s: the layers' own time and {hops} round trip(s)",
        plan.ms_per_token,
        1e3 / plan.ms_per_token.max(1e-9)
    ));
    // Which machine a prompt waits on, when there is more than one.
    let waits_on = nodes
        .iter()
        .zip(&plan.shares)
        .filter(|(_, share)| !share.layers.is_empty())
        .max_by(|a, b| {
            let of = |(node, share): &(&Node, &Share)| share.layers.len() as f64 * node.prompt_layer_ms + node.prompt_hop_ms / chunk;
            of(a).total_cmp(&of(b))
        })
        .map(|(node, _)| node.name.as_str())
        .unwrap_or("");
    lines.push(format!(
        "a prompt read at about {:.0} tok/s{}. Estimates; the first measurement is worth more.",
        1e3 / plan.prompt_ms_per_token.max(1e-9),
        if hops > 0 { format!(", every machine at once and {waits_on} the one it waits on") } else { String::new() }
    ));
    lines
}

/// What a layer of a model costs on a machine, in milliseconds a token:
/// written, and read as part of a prompt.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Speed {
    pub layer_ms: f64,
    /// Absent from a worker that predates it, and then taken for the same.
    #[serde(default)]
    pub prompt_layer_ms: f64,
}

/// How long one layer of a model this shape takes on this machine: its
/// seven matrix products for one token, timed, and the same for a chunk
/// of a prompt's tokens at once.
///
/// Not a layer — there is no model yet, which is the point of asking
/// before one is loaded — but the arithmetic a layer is made of, at the
/// widths this model has and the precision it will be loaded at: four
/// products the width of the model for attention, three the width of the
/// MLP. A wide MLP is timed at a slice of its rows and scaled, so that
/// asking a small machine about a large model does not take the memory the
/// question is about.
///
/// The two are timed apart because they are different arithmetic on the
/// same weights. One token reads every weight for one row of output; a
/// chunk reads them once for all its rows, and a token of it costs a
/// fraction — a different fraction on every machine.
pub fn bench(n_embd: usize, intermediate: usize, precision: Precision) -> Speed {
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
    let chunk = crate::model::prefill_chunk();
    let (x, xs) = (vec![0.5f32; n_embd], vec![0.5f32; chunk * n_embd]);
    let best = |times: usize, run: &dyn Fn()| {
        (0..times)
            .map(|_| {
                let started = Instant::now();
                run();
                started.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::INFINITY, f64::min)
    };
    let (wide_scale, square_scale) = (intermediate as f64 / wide_rows as f64, n_embd as f64 / square_rows as f64);
    let a_layer = |wide_ms: f64, square_ms: f64| 3.0 * wide_ms * wide_scale + 4.0 * square_ms * square_scale;
    let pool = crate::model::pool();
    pool.install(|| {
        // First, long enough to be awake. A machine asked this has usually
        // been doing nothing, and one that has been doing nothing runs its
        // next tenth of a second at a third of its speed (see
        // `model::linger`): timed cold, a worker looked three times slower
        // beside the machine that asked, which had been busy asking.
        let woken = Instant::now();
        while woken.elapsed() < std::time::Duration::from_millis(120) {
            drop(std::hint::black_box(square.matmul_bt(std::hint::black_box(&xs), chunk, None)));
        }
        let one = |w: &Weight| best(12, &|| drop(std::hint::black_box(w.matvec_bt(std::hint::black_box(&x), None))));
        let many = |w: &Weight| best(4, &|| drop(std::hint::black_box(w.matmul_bt(std::hint::black_box(&xs), chunk, None))));
        Speed { layer_ms: a_layer(one(&wide), one(&square)), prompt_layer_ms: a_layer(many(&wide), many(&square)) / chunk as f64 }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    /// A machine whose layers cost a quarter as much a token read as
    /// written, and whose round trip for a chunk is ten of a token's.
    fn node(name: &str, gb: u64, layer_ms: f64, hop_ms: f64) -> Node {
        Node { name: name.into(), memory: gb * GB, layer_ms, hop_ms, prompt_layer_ms: layer_ms / 4.0, prompt_hop_ms: hop_ms * 10.0 }
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
        let plan = place(&nodes, &seventy(), Mix::REPLY).unwrap();
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
        let plan = place(&nodes, &seventy(), Mix::REPLY).unwrap();
        // The studio is fastest and takes all it can, 71; here takes the
        // other 9; the mini is not needed and is not used.
        assert_eq!(runs(&plan), [0..9, 9..80, 80..80]);
        assert!((plan.ms_per_token - (9.0 * 1.0 + 71.0 * 0.5 + 0.3)).abs() < 1e-9);

        // With a slower studio the order turns over: here fills first.
        let nodes = [node("here", 48, 1.0, 0.0), node("studio", 64, 1.5, 0.3), node("mini", 16, 2.0, 0.3)];
        assert_eq!(runs(&place(&nodes, &seventy(), Mix::REPLY).unwrap()), [0..47, 47..80, 80..80]);
        // A machine a little faster is not worth a round trip to it: the
        // token is quicker with every layer it can hold kept here.
        let nodes = [node("here", 96, 1.0, 0.0), node("twin", 96, 0.999, 0.3)];
        assert_eq!(runs(&place(&nodes, &seventy(), Mix::REPLY).unwrap()), [0..80, 80..80]);
        // One a great deal faster is, hop and all.
        let nodes = [node("here", 96, 1.0, 0.0), node("studio", 96, 0.5, 0.3)];
        assert_eq!(runs(&place(&nodes, &seventy(), Mix::REPLY).unwrap()), [0..0, 0..80]);
        // And a tie goes to the machine with the tokens.
        let nodes = [node("here", 48, 1.0, 0.0), node("twin", 48, 1.0, 0.3)];
        assert_eq!(runs(&place(&nodes, &seventy(), Mix::REPLY).unwrap()), [0..47, 47..80]);
    }

    /// The rig this was measured on, in round numbers: thirty layers, a
    /// machine here and one half as fast for a prompt over a slow cable,
    /// and room for everything on either.
    fn rig() -> ([Node; 2], Sizes) {
        let here = Node { name: "here".into(), memory: 38 * GB, layer_ms: 0.19, hop_ms: 0.0, prompt_layer_ms: 0.07, prompt_hop_ms: 0.0 };
        let air = Node { name: "air".into(), memory: 6 * GB, layer_ms: 0.22, hop_ms: 0.8, prompt_layer_ms: 0.11, prompt_hop_ms: 8.0 };
        ([here, air], Sizes { n_layer: 30, ends: GB / 10, layer: 3_000_000, cache: 12_000_000 })
    }

    #[test]
    fn a_prompt_wants_the_stages_level_and_a_reply_wants_one_machine() {
        let (nodes, sizes) = rig();
        // For the reply everything stays here: a hop buys nothing.
        let reply = place(&nodes, &sizes, Mix::REPLY).unwrap();
        assert_eq!(runs(&reply), [0..30, 30..30]);
        assert!((reply.prompt_ms_per_token - 30.0 * 0.07).abs() < 1e-9, "one machine reads a prompt in a row");

        // For the prompt the slower machine is given layers, and fewer:
        // enough that neither waits long for the other.
        let prompt = place(&nodes, &sizes, Mix::PROMPT).unwrap();
        let (ours, theirs) = (prompt.shares[0].layers.len(), prompt.shares[1].layers.len());
        assert_eq!(ours + theirs, 30);
        assert!(theirs > 0 && theirs < ours, "{ours} here and {theirs} there");
        let (a, b) = (ours as f64 * 0.07, theirs as f64 * 0.11 + 8.0 / crate::model::prefill_chunk() as f64);
        assert!((prompt.prompt_ms_per_token - a.max(b)).abs() < 1e-9);
        assert!(prompt.prompt_ms_per_token < reply.prompt_ms_per_token * 0.75, "which reads a prompt a good deal quicker");
        assert!(prompt.ms_per_token > reply.ms_per_token, "and writes a reply slower: the two are different questions");
        // No placing reads it quicker: a layer moved either way is worse.
        for moved in [ours - 1, ours + 1] {
            let other = check(&nodes, &sizes, &[0..moved, moved..30]).unwrap();
            assert!(other.prompt_ms_per_token >= prompt.prompt_ms_per_token - 1e-9, "{moved} here reads at {}", other.prompt_ms_per_token);
        }
    }

    #[test]
    fn a_mix_is_placed_for_what_it_is_mostly() {
        let (nodes, sizes) = rig();
        let here_only = |mix: &str| place(&nodes, &sizes, Mix::parse(mix).unwrap()).unwrap().shares[1].layers.is_empty();
        // A long question and a short answer is placed like a prompt; a
        // short question and a long answer like a reply.
        assert!(!here_only("5000:50") && here_only("50:5000"));
        assert!(here_only("reply") && !here_only("prompt"));
        // The proportion is what is read, not the size.
        assert_eq!(place(&nodes, &sizes, Mix::parse("10:1").unwrap()).unwrap(), place(&nodes, &sizes, Mix::parse("5000:500").unwrap()).unwrap());

        assert_eq!(Mix::parse(" 2000 : 200 ").unwrap(), Mix { prompt: 2000.0, reply: 200.0 });
        for bad in ["", "fast", "0:0", "-1:5", "1:", "a:b", "1:2:3"] {
            assert!(Mix::parse(bad).is_err(), "{bad}");
        }
    }

    /// Memory still decides first: a prompt cannot be given layers a
    /// machine has no room for, however level that would make the stages.
    #[test]
    fn a_prompts_placing_stays_within_each_machines_memory() {
        let (mut nodes, sizes) = rig();
        // Room for two layers on the worker, and no more.
        nodes[1].memory = 2 * (sizes.layer + sizes.cache) + 1;
        let plan = place(&nodes, &sizes, Mix::PROMPT).unwrap();
        assert!(plan.shares[1].layers.len() <= 2, "{:?}", runs(&plan));
        assert!(plan.shares.iter().all(|share| share.left >= 0));
        // And with three machines the layers still all go somewhere once.
        let (two, sizes) = rig();
        let three = [two[0].clone(), two[1].clone(), Node { name: "mini".into(), ..two[1].clone() }];
        let plan = place(&three, &sizes, Mix::PROMPT).unwrap();
        assert_eq!(plan.shares.iter().map(|s| s.layers.len()).sum::<usize>(), 30);
        assert_eq!(plan.shares[2].layers.end, 30);
        assert!(plan.shares.iter().skip(1).all(|s| !s.layers.is_empty()), "a third machine takes a share of a prompt: {:?}", runs(&plan));
        assert!(plan.prompt_ms_per_token < place(&two, &sizes, Mix::PROMPT).unwrap().prompt_ms_per_token);
    }

    #[test]
    fn a_model_too_large_for_all_of_them_is_said_to_be() {
        let nodes = [node("here", 48, 1.0, 0.0), node("air", 6, 2.0, 2.5)];
        let why = place(&nodes, &seventy(), Mix::REPLY).unwrap_err();
        assert!(why.contains("does not fit") && why.contains("77.0 GB") && why.contains("54.0 GB") && why.contains("27 of its 80 layers"), "{why}");
        // The ends stay with the tokens, so a small machine cannot run a
        // large model from however many large ones.
        let nodes = [node("air", 4, 2.0, 0.0), node("studio", 512, 0.5, 0.3)];
        let why = place(&nodes, &seventy(), Mix::REPLY).unwrap_err();
        assert!(why.contains("air has 4.0 GB") && why.contains("embedding and output head are 5.0 GB"), "{why}");
        assert!(place(&[], &seventy(), Mix::REPLY).is_err());
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
        let plan = place(&nodes, &seventy(), Mix::REPLY).unwrap();
        let lines = table(&nodes, &plan);
        assert_eq!(lines.len(), 6);
        assert!(lines[1].starts_with("here ") && lines[1].contains("0..9") && lines[1].contains("48.0 GB"), "{}", lines[1]);
        assert!(lines[2].contains("9..80") && lines[3].contains("none"), "{lines:?}");
        assert!(lines[4].contains("a reply at about 44.8 ms a token") && lines[4].contains("1 round trip(s)"), "{}", lines[4]);
        assert!(lines[5].contains("a prompt read at about") && lines[5].contains("studio the one it waits on"), "{}", lines[5]);
    }

    /// A time comes back, for every precision and for widths a slice is
    /// taken of. Which of two shapes is slower is not asserted: beside two
    /// hundred other tests it is the machine that is being timed.
    #[test]
    fn a_layer_is_timed() {
        for precision in [Precision::F32, Precision::Q8, Precision::Q4] {
            let speed = bench(256, 1024, precision);
            assert!(speed.layer_ms > 0.0 && speed.layer_ms.is_finite(), "{precision}: {speed:?}");
            assert!(speed.prompt_layer_ms > 0.0 && speed.prompt_layer_ms.is_finite(), "{precision}: {speed:?}");
        }
        let wide = bench(64, 9000, Precision::Q8);
        assert!(wide.layer_ms > 0.0 && wide.layer_ms.is_finite(), "{wide:?}");
        assert!(bench(1, 1, Precision::F32).layer_ms.is_finite(), "nothing so small that it divides by nothing");
        // A worker that predates the second number is read without it.
        let old: Speed = serde_json::from_str(r#"{"layer_ms": 0.5}"#).unwrap();
        assert_eq!((old.layer_ms, old.prompt_layer_ms), (0.5, 0.0));
    }
}
