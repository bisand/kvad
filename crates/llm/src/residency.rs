//! Which experts a mixture actually reads, and whether they would fit in RAM.
//!
//! # The question
//!
//! A mixture-of-experts checkpoint is nearly all experts. Qwen3-30B-A3B is
//! 61 GB at bf16, and 58 GB of that is six thousand expert MLPs — 48 layers
//! of 128 — of which a token reads eight per layer. Ninety-five per cent of
//! the weights are cold on any given token, which raises the obvious
//! question: do they have to be in memory at all?
//!
//! The disk says the shape of the answer. An NVMe that reads 11.7 GB/s in
//! 4 MB blocks and 0.19 GB/s in 4 KB ones is two different devices, and one
//! expert of Qwen3-30B-A3B is 9.44 MB — which lands in the fast regime by
//! luck of arithmetic rather than by design. So *if* most of what a token
//! asks for is already resident, the rest can be fetched at a price worth
//! paying, and a model bigger than memory runs anyway.
//!
//! Everything hangs on "most", and "most" is a measurement, not an opinion.
//!
//! # Why the measurement comes first
//!
//! The machinery this would justify — an expert store, a slab cache, a
//! prefetcher, an eviction policy — is a lot of code, and all of it is
//! worthless if the hit rate is bad. It is also entirely unnecessary to
//! write any of it to find out: the router already decides which experts it
//! wants, so writing that decision down costs one branch, and the resulting
//! trace answers the question offline.
//!
//! So this module is two halves that never run at the same time. [`Trace`]
//! records, inside a real forward pass, at roughly no cost. [`Log`] reads
//! the recording back and replays it against a cache that does not exist,
//! at capacities the machine does not have.
//!
//! # Recording
//!
//! Off unless `KVAD_EXPERT_TRACE` names a file:
//!
//! ```text
//! KVAD_EXPERT_TRACE=/tmp/qwen3.trace kvad chat qwen3-30b-a3b
//! ```
//!
//! Nothing needs wiring into `main`: the first mixture to route a token
//! checks the environment once, and every entry point — the CLI, the
//! server, a test — gets the same behaviour for free.
//!
//! Recording costs about 8% of decode; not recording costs nothing
//! measurable. Both measured over three runs against the same generation on
//! a four-expert mixture — which is the pessimistic end, because a layer
//! that cheap makes the bookkeeping around it as visible as it ever gets.
//!
//! # What a line means
//!
//! ```text
//! # kvad expert trace v1
//! # n_experts=128 expert_bytes=9437184
//! 3 0 5,17,42,88,91,101,110,127
//! ```
//!
//! Layer, the token's slot within the batch that was routed, and the
//! experts it chose in the router's own ranking order. Lines appear in the
//! order the model read them, which is the only property the replay needs:
//! a cache simulation is a function of access *order*, and the file is that
//! order written down.
//!
//! The pair `(layer, expert)` is the unit throughout, never the expert
//! alone. Layer 3's expert 5 and layer 10's expert 5 are different weights
//! that share a number, and a cache that confused them would report a hit
//! rate roughly `n_layers` times too good.

use std::collections::{BinaryHeap, HashMap};
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::{self, AtomicBool};
use std::sync::{Mutex, OnceLock};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The environment variable that names the file to record into.
pub const VAR: &str = "KVAD_EXPERT_TRACE";

static ON: AtomicBool = AtomicBool::new(false);
static SINK: Mutex<Option<Sink>> = Mutex::new(None);

struct Sink {
    out: std::io::BufWriter<std::fs::File>,
    /// The header is written by whichever mixture reports first, because
    /// that is the first moment anything here knows how wide the model is.
    described: bool,
}

/// Whether a trace is being written.
pub fn recording() -> bool {
    ON.load(atomic::Ordering::Relaxed)
}

/// Record into `path`, replacing whatever was there.
pub fn record_to(path: &Path) -> Res<()> {
    let file = std::fs::File::create(path)?;
    let mut sink = SINK.lock().unwrap_or_else(|p| p.into_inner());
    *sink = Some(Sink { out: std::io::BufWriter::new(file), described: false });
    ON.store(true, atomic::Ordering::Relaxed);
    Ok(())
}

/// Consult the environment, once per process.
///
/// A failure here is reported and then ignored. A diagnostic that cannot
/// write its output should not take a generation down with it.
fn ensure_started() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if let Some(path) = std::env::var_os(VAR).filter(|p| !p.is_empty()) {
            if let Err(e) = record_to(Path::new(&path)) {
                eprintln!("expert trace: cannot write {}: {e}", Path::new(&path).display());
            }
        }
    });
}

/// One mixture's routing decisions, on their way to the log.
///
/// [`Moe::run`](crate::model::ffn::Moe::run) builds one of these whether or
/// not anything is recording; when nothing is, it holds no allocation and
/// every method is a branch that falls through.
pub struct Trace(Option<Vec<Vec<u32>>>);

impl Trace {
    /// Begin collecting for a batch of `tokens`.
    pub fn start(tokens: usize) -> Trace {
        ensure_started();
        Trace(recording().then(|| Vec::with_capacity(tokens)))
    }

    /// One token's picks, as the router ranked them.
    ///
    /// Takes the router's own `(expert, weight)` pairs rather than a tidied
    /// list, so that the call site is one line and cannot drift from what
    /// the mixture went on to read.
    pub fn token(&mut self, picks: &[(usize, f32)]) {
        if let Some(rows) = &mut self.0 {
            rows.push(picks.iter().map(|&(e, _)| e as u32).collect());
        }
    }

    /// Write the batch out, tagged with the layer it belongs to.
    ///
    /// `experts` and `expert_bytes` are what the model actually loaded, not
    /// what its config claims: the replay converts hit rates into seconds,
    /// and it can only do that if the byte figure came from a real weight.
    pub fn finish(self, layer: usize, experts: usize, expert_bytes: usize) {
        let Some(rows) = self.0 else { return };
        let mut guard = SINK.lock().unwrap_or_else(|p| p.into_inner());
        let Some(sink) = guard.as_mut() else { return };

        if !std::mem::replace(&mut sink.described, true) {
            let _ = writeln!(sink.out, "# kvad expert trace v1");
            let _ = writeln!(sink.out, "# n_experts={experts} expert_bytes={expert_bytes}");
        }
        for (slot, picks) in rows.iter().enumerate() {
            let _ = write!(sink.out, "{layer} {slot}");
            for (i, e) in picks.iter().enumerate() {
                let _ = write!(sink.out, "{}{e}", if i == 0 { ' ' } else { ',' });
            }
            let _ = writeln!(sink.out);
        }
        // Flushed per layer, so that killing a slow generation half way
        // still leaves a readable trace.
        let _ = sink.out.flush();
    }
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// One token's worth of routing at one layer.
pub struct Step {
    pub layer: u32,
    pub experts: Vec<u32>,
}

/// A recorded trace, ready to replay.
pub struct Log {
    pub n_experts: usize,
    pub expert_bytes: usize,
    pub steps: Vec<Step>,
}

impl Log {
    pub fn read(path: &Path) -> Res<Log> {
        let file = std::fs::File::open(path)?;
        let mut log = Log { n_experts: 0, expert_bytes: 0, steps: Vec::new() };

        for line in std::io::BufReader::new(file).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(header) = line.strip_prefix('#') {
                for field in header.split_whitespace() {
                    let Some((key, value)) = field.split_once('=') else { continue };
                    match key {
                        "n_experts" => log.n_experts = value.parse()?,
                        "expert_bytes" => log.expert_bytes = value.parse()?,
                        _ => {}
                    }
                }
                continue;
            }
            // `layer slot e,e,e` — the slot is recorded for readability and
            // is not replayed: within a batch the order of tokens at one
            // layer does not change which weights get touched.
            let mut fields = line.split(' ');
            let (Some(layer), Some(_slot), Some(picks)) =
                (fields.next(), fields.next(), fields.next())
            else {
                return Err(format!("expert trace: cannot read line `{line}`").into());
            };
            log.steps.push(Step {
                layer: layer.parse()?,
                experts: picks.split(',').map(str::parse).collect::<Result<_, _>>()?,
            });
        }
        if log.n_experts == 0 {
            return Err("expert trace: no header, or no experts in it".into());
        }
        Ok(log)
    }

    /// How many distinct layers route.
    ///
    /// Not every block has a mixture — the first few are dense in every
    /// family that does this — so this counts what the trace saw rather
    /// than what the model has.
    pub fn layers(&self) -> usize {
        let mut seen: Vec<bool> = Vec::new();
        for step in &self.steps {
            let i = step.layer as usize;
            if seen.len() <= i {
                seen.resize(i + 1, false);
            }
            seen[i] = true;
        }
        seen.iter().filter(|s| **s).count()
    }

    /// Tokens routed, derived rather than recorded.
    ///
    /// Every forward pass visits every routed layer exactly once, so the
    /// step count divides evenly by the layer count and the quotient is the
    /// number of tokens — whether they arrived one at a time in decode or
    /// as a batch in prefill.
    pub fn tokens(&self) -> usize {
        match self.layers() {
            0 => 0,
            n => self.steps.len() / n,
        }
    }

    /// Every `(layer, expert)` read, flattened into access order.
    ///
    /// The key packs the pair into one integer so a cache can be a plain
    /// map. `n_experts` is the stride, which is why it has to come from the
    /// header and not be guessed from the largest expert seen.
    fn accesses(&self) -> Vec<u32> {
        let stride = self.n_experts as u32;
        let mut keys = Vec::with_capacity(self.steps.len() * 8);
        for step in &self.steps {
            keys.extend(step.experts.iter().map(|e| step.layer * stride + e));
        }
        keys
    }

    /// How many distinct experts the trace ever touched.
    ///
    /// The floor on a useful cache: below this, something is always being
    /// evicted that will be wanted again.
    pub fn distinct(&self) -> usize {
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        seen.extend(self.accesses());
        seen.len()
    }

    /// Replay against a cache of `capacity` experts under `policy`.
    pub fn replay(&self, policy: Policy, capacity: usize) -> Outcome {
        let keys = self.accesses();
        let hits = match policy {
            Policy::Lru => lru(&keys, capacity),
            Policy::Pinned => pinned(&keys, capacity),
            Policy::Optimal => optimal(&keys, capacity),
        };
        Outcome {
            policy,
            capacity,
            hits,
            misses: keys.len() as u64 - hits,
            expert_bytes: self.expert_bytes as u64,
            tokens: self.tokens() as u64,
        }
    }
}

/// How a cache decides what to keep.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Policy {
    /// Evict whatever was used longest ago. What you get for free.
    Lru,
    /// Keep the globally hottest experts and never evict anything.
    ///
    /// Worth measuring because it is so much less machinery: the residency
    /// set is decided once, offline, and the runtime needs no eviction, no
    /// bookkeeping and no lock. If this is close to LRU, build this.
    Pinned,
    /// Evict whatever is needed furthest in the future — Bélády's rule,
    /// which no real cache can follow because it reads the future.
    ///
    /// The ceiling. If LRU is close to it there is nothing left to win by
    /// being cleverer, and if both are bad the model simply needs more RAM.
    Optimal,
}

impl Policy {
    pub const ALL: [Policy; 3] = [Policy::Lru, Policy::Pinned, Policy::Optimal];

    pub fn name(self) -> &'static str {
        match self {
            Policy::Lru => "LRU",
            Policy::Pinned => "pinned",
            Policy::Optimal => "optimal",
        }
    }
}

/// What a replay came to.
pub struct Outcome {
    pub policy: Policy,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub expert_bytes: u64,
    pub tokens: u64,
}

impl Outcome {
    pub fn hit_rate(&self) -> f64 {
        match self.hits + self.misses {
            0 => 0.0,
            n => self.hits as f64 / n as f64,
        }
    }

    /// Bytes this policy would pull off the disk for each token.
    ///
    /// The number that decides everything: divided by the disk's measured
    /// bandwidth it is seconds per token, and compared against the model's
    /// compute time it says whether the fetch can hide behind arithmetic.
    pub fn bytes_per_token(&self) -> f64 {
        match self.tokens {
            0 => 0.0,
            n => (self.misses * self.expert_bytes) as f64 / n as f64,
        }
    }

    /// Tokens per second, if the fetches were the only cost.
    ///
    /// Deliberately optimistic: it assumes compute is free and every fetch
    /// runs at the streaming rate. A real implementation lands below this,
    /// and if *this* number is bad there is no point measuring the real one.
    pub fn ceiling(&self, bytes_per_second: f64) -> f64 {
        match self.bytes_per_token() {
            0.0 => f64::INFINITY,
            b => bytes_per_second / b,
        }
    }
}

/// Least recently used, with lazy deletion.
///
/// The heap holds an entry per access rather than per resident expert, and
/// stale entries are recognised on the way out by their timestamp
/// disagreeing with the map. That trades memory for never having to find
/// and remove an entry in the middle, which is the part that makes an exact
/// LRU awkward without a linked list.
fn lru(keys: &[u32], capacity: usize) -> u64 {
    if capacity == 0 {
        return 0;
    }
    let mut last: HashMap<u32, u64> = HashMap::with_capacity(capacity * 2);
    let mut heap: BinaryHeap<std::cmp::Reverse<(u64, u32)>> = BinaryHeap::new();
    let mut hits = 0;

    for (now, &key) in keys.iter().enumerate() {
        let now = now as u64;
        if last.insert(key, now).is_some() {
            hits += 1;
        }
        heap.push(std::cmp::Reverse((now, key)));
        while last.len() > capacity {
            let std::cmp::Reverse((when, victim)) = heap.pop().expect("heap holds every insert");
            if last.get(&victim) == Some(&when) {
                last.remove(&victim);
            }
        }
    }
    hits
}

/// The globally hottest `capacity` experts, resident for the whole run.
///
/// This is cheating in the model's favour — the frequencies come from the
/// very trace being replayed — but only mildly: expert popularity is a
/// property of the model and its training data, so a set chosen on one
/// corpus transfers to another. Treat it as the optimistic end of "pin the
/// hot ones", and confirm it by pinning from one trace and replaying
/// another.
fn pinned(keys: &[u32], capacity: usize) -> u64 {
    let mut count: HashMap<u32, u64> = HashMap::new();
    for &key in keys {
        *count.entry(key).or_default() += 1;
    }
    let mut ranked: Vec<(u32, u64)> = count.into_iter().collect();
    // By count, then by key: ties would otherwise resolve on hash order and
    // the same trace would score differently run to run.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // The first access to a pinned expert is a miss in any honest
    // accounting — the cache starts empty and something has to load it —
    // but over thousands of tokens that is a rounding error, and counting
    // it would need the load order, which this policy does not have. Every
    // later access hits.
    ranked.iter().take(capacity).map(|(_, n)| n - 1).sum()
}

/// Bélády: evict whatever is wanted furthest in the future.
fn optimal(keys: &[u32], capacity: usize) -> u64 {
    if capacity == 0 {
        return 0;
    }
    // Walking backwards, the last place each key was seen *is* its next use
    // from where we now stand — so one pass builds the whole table.
    let mut next = vec![usize::MAX; keys.len()];
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for i in (0..keys.len()).rev() {
        if let Some(&later) = seen.get(&keys[i]) {
            next[i] = later;
        }
        seen.insert(keys[i], i);
    }

    let mut resident: HashMap<u32, usize> = HashMap::with_capacity(capacity * 2);
    let mut heap: BinaryHeap<(usize, u32)> = BinaryHeap::new();
    let mut hits = 0;

    for (i, &key) in keys.iter().enumerate() {
        if resident.insert(key, next[i]).is_some() {
            hits += 1;
        }
        heap.push((next[i], key));
        while resident.len() > capacity {
            let (when, victim) = heap.pop().expect("heap holds every insert");
            // Stale unless it still agrees with the map, same as in `lru`.
            if resident.get(&victim) == Some(&when) {
                resident.remove(&victim);
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worked by hand: capacity 2, and the access pattern that makes LRU
    /// look bad. A B C A B C ... never hits, because the one thing being
    /// evicted is always the next thing wanted.
    #[test]
    fn lru_thrashes_on_a_loop_longer_than_it_is() {
        let keys = [0, 1, 2, 0, 1, 2, 0, 1, 2];
        assert_eq!(lru(&keys, 2), 0);
        // One more slot and every access after the first three hits.
        assert_eq!(lru(&keys, 3), 6);
    }

    #[test]
    fn optimal_beats_lru_on_that_same_loop() {
        let keys = [0, 1, 2, 0, 1, 2, 0, 1, 2];
        // Keeping two of the three and giving up on the third beats
        // rotating all of them: 0 and 1 hit twice each after loading.
        assert!(optimal(&keys, 2) > lru(&keys, 2));
    }

    /// Whatever the pattern, no policy may beat Bélády and none may beat
    /// keeping everything.
    #[test]
    fn optimal_is_the_ceiling() {
        let mut keys = Vec::new();
        let mut x: u32 = 7;
        for _ in 0..4000 {
            // A cheap skewed generator: squaring the low bits clusters the
            // keys the way a real router does, which is the case that
            // matters.
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            keys.push((x >> 16) % 40 / 3);
        }
        let distinct = keys.iter().collect::<std::collections::HashSet<_>>().len();
        for capacity in [1, 2, 5, 9, 13] {
            let (l, p, o) = (lru(&keys, capacity), pinned(&keys, capacity), optimal(&keys, capacity));
            assert!(o >= l, "optimal {o} < lru {l} at capacity {capacity}");
            assert!(o >= p, "optimal {o} < pinned {p} at capacity {capacity}");
        }
        // With room for everything, only the first sight of each key misses.
        assert_eq!(lru(&keys, distinct), keys.len() as u64 - distinct as u64);
        assert_eq!(optimal(&keys, distinct), keys.len() as u64 - distinct as u64);
    }

    #[test]
    fn pinned_keeps_the_hottest() {
        // 0 appears four times, 1 twice, 2 once. One slot keeps 0, and
        // scores its three repeat visits.
        let keys = [0, 0, 1, 0, 2, 1, 0];
        assert_eq!(pinned(&keys, 1), 3);
        assert_eq!(pinned(&keys, 2), 4);
    }

    #[test]
    fn a_trace_survives_the_round_trip() {
        let dir = std::env::temp_dir().join(format!("kvad-residency-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("t.trace");

        record_to(&path).expect("record");
        for layer in 0..3usize {
            let mut trace = Trace::start(2);
            trace.token(&[(1, 0.5), (4, 0.5)]);
            trace.token(&[(4, 0.5), (7, 0.5)]);
            trace.finish(layer, 8, 1024);
        }
        ON.store(false, atomic::Ordering::Relaxed);

        let log = Log::read(&path).expect("read back");
        assert_eq!(log.n_experts, 8);
        assert_eq!(log.expert_bytes, 1024);
        assert_eq!(log.layers(), 3);
        assert_eq!(log.steps.len(), 6);
        assert_eq!(log.tokens(), 2);
        assert_eq!(log.steps[0].experts, vec![1, 4]);
        // Expert 4 of layer 0 and expert 4 of layer 1 are different weights.
        assert_eq!(log.distinct(), 9);

        // Room for everything: twelve reads, nine of them first sights.
        let all = log.replay(Policy::Lru, 9);
        assert_eq!(all.hits, 3);
        assert_eq!(all.misses, 9);
        assert_eq!(all.bytes_per_token(), 9.0 * 1024.0 / 2.0);

        std::fs::remove_dir_all(&dir).ok();
    }
}
