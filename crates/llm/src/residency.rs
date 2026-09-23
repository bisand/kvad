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
//! # What it answered
//!
//! Measured on DeepSeek-V2-Lite at q8 — 26 routed layers, 64 experts,
//! top-6 — over three thousand-token generations in different domains
//! (Rust, prose, plant biochemistry). At half the expert store resident:
//!
//! ```text
//! trace      LRU    pinned (self)   pinned (other domain)   optimal
//! code      71.2%       77.5%            60.1%               88.4%
//! prose     70.0%       75.4%            59.0%               87.8%
//! science   71.1%       71.3%            55.5%               88.2%
//! ```
//!
//! Pinning beats LRU only while it is allowed to rank experts on the very
//! traffic it is then scored against. Given a profile from another domain,
//! which is the only kind a deployment has, it falls eleven points *below*
//! LRU, on every pair. So the policy to build is the ordinary one.
//!
//! Routing is why. It is skewed, but weakly — the busiest expert runs 5.2x
//! as often as the average and the top tenth of experts take 22-27% of the
//! reads — and what skew there is belongs to the domain rather than to the
//! model: the hot tenth of a code trace and of a science trace overlap by
//! 24%, against 10% for picking at random. Over a thousand tokens
//! essentially every expert is read at least once. There is no stable hot
//! set to pin.
//!
//! Two things worth keeping. LRU is useless below a threshold and fine
//! above it, and the threshold is not mysterious: a forward pass reads
//! `top_k * layers` experts before it repeats itself, so a cache smaller
//! than that evicts every entry before its next use. Here that is 156 of
//! 1664 blobs, and the measured cliff sits exactly there — 1.9% hit rate
//! at 5% capacity, 27.6% at 10%. And Belady stays seventeen points above
//! LRU at every size, so the gap is real and something cleverer than LRU
//! could still claim it. Just not by pinning.
//!
//! Qwen3-30B-A3B — 48 layers, 128 experts, top-8 — was then run to test
//! that reading, and confirmed it. Its working set is 6.25% of its store
//! against DeepSeek's 9.4%, and everything moved the way finer granularity
//! predicts:
//!
//! ```text
//!                       LRU @10%   LRU @50%   pinned (other domain) @50%
//! DeepSeek-V2-Lite        27.6%      71.1%            ~60%
//! Qwen3-30B-A3B           51.6%      97.0%            61-77%  (LRU +21..36)
//! ```
//!
//! The cliff landed on the blob. A cache of 374 blobs hits 1.9%; one of
//! 384 — which is `top_k * layers` exactly — hits 33.2%. Nothing about
//! that is tuned, and it gives a sizing rule that needs no measurement:
//! below `top_k * layers` experts an LRU cache returns nothing, because a
//! pass evicts every entry before its next use.
//!
//! What climbs after the cliff is routing skew, and that is a second,
//! separate variable. Qwen3 is far more concentrated than DeepSeek — its
//! busiest expert runs 14x the average against 5x, its top tenth takes
//! half the reads against a quarter — which is why it reaches 97% where
//! DeepSeek reaches 71%. And yet pinning does *worse* here, not better:
//! dead level with LRU even when ranked on the trace it is scored against,
//! and 21 to 36 points behind when ranked on another domain. Skew does not
//! help a static set, because the skew is the domain's and not the
//! model's — the hot tenths of two domains overlap by 22-37%. LRU gets the
//! same concentration for free and re-learns it when the subject changes.
//!
//! A four-expert mixture said the opposite, convincingly, and was wrong:
//! at top-2 of 4 it reads half of every layer, so its working set is half
//! the store and LRU cannot win below that capacity whatever the routing
//! does. Granularity was the variable, and one model could not show it.
//!
//! # The third model, and what it cost the argument
//!
//! Qwen3-Next-80B-A3B-Instruct at q4 — 48 layers, 512 experts, top-10 — is
//! the sparsest yet: 480 blobs of 24,576, or 1.95% of the store, against
//! Qwen3's 6.25% and DeepSeek's 9.4%. Replayed against the real file with
//! real `pread`s, over the same three domains:
//!
//! ```text
//!   cache     hit (code/prose/sci)     fetch ms/token
//!   4.8 GB     75.7 / 79.8 / 73.3        14.8 / 12.1 / 12.8
//!   9.7 GB     88.2 / 91.5 / 86.8         9.3 /  5.9 /  7.1
//!  24.2 GB     96.9 / 96.6 / 97.0         2.9 /  2.3 /  1.8
//! ```
//!
//! Compared at equal memory rather than equal percentage, the larger model
//! is the cheaper one to stream: at ~9.7 GB of cache Qwen3-30B spends
//! 28.6 ms a token on disk and this spends 9.3, with 2.6x the parameters.
//! Sparser routing is a smaller hot set, and the hot set is the whole cost.
//!
//! Then the measurement that cost the argument its premise. The case for
//! building any of this rested on naive mmap being hopeless above memory —
//! DeepSeek-V2-Lite at f32, a fifth over, generated 1.5 tok/s against 24
//! resident. But this model, *two fifths* over memory at 50 GB against 36
//! usable, generates **10.7 tok/s through the ordinary mmap path** with no
//! expert cache at all (150 tokens, 14.06s, default threads).
//!
//! The same rule explains both. DeepSeek thrashed because its working set
//! is five times denser, not because paging is a bad mechanism; the kernel
//! keeps a 0.94 GB hot set resident without being told anything. So the
//! honest value of an expert-aware fetcher here looked like 10.7 to
//! roughly 15.5 tok/s — worth having, and not the order of magnitude that
//! was claimed for it before this was measured. (Then it was built and
//! measured too; see below. It is not 15.5 either.)
//!
//! The rule now holds on three models, and predicts the mmap case as well
//! as the replayed one, which is more than it was built to do.
//!
//! # The fetcher, wired in, and what it was worth
//!
//! That estimate was then tested, and it did not survive. With
//! [`crate::experts::Resident`] in `Moe::run` -- whole experts read with
//! `F_NOCACHE`, four at a time, into an LRU of its own -- the same model and
//! prompt, 400 tokens, three interleaved reps, on AC power:
//!
//! ```text
//!   plain mmap        8.6  10.6  10.6 tok/s    median 10.6
//!   16 GB cache      10.1  11.6  11.4          median 11.4   89.9% hits
//!   26 GB cache       8.3   9.7  10.0          median  9.7   93.2% hits
//! ```
//!
//! The larger cache hits more, reads a third less, and came out slower; a
//! single rerun of each put it ahead instead, 12.0 to 10.7. Neither run
//! compressed or swapped a page. So the effect is inside the noise, and at
//! most the 8% the medians suggest -- not 45%.
//!
//! The premise was wrong in a way the numbers above already said. At 89.9%
//! hits the cache reads about 100 MB a token, which is 8 ms of an 88 ms
//! token at the disk's measured rate; the mapping was already holding the
//! hot set, as the paragraph above says the kernel does unasked. What the
//! token is waiting on is the experts' matvecs: `sample` puts two thirds of
//! the driving thread in them, each a 0.66 MB matrix handed to fourteen
//! threads, and more of the pool yielding than computing. The next win for
//! this model is a decode kernel that runs a token's experts side by side,
//! not a faster way to read them.
//!
//! That kernel was then written (`quant::matvec_many`: a token's experts
//! as two parallel sections instead of thirty), and it changed what the
//! fetcher is worth, because the two were never independent. Qwen3-30B at
//! q8 is 34 GB against 36 usable -- a fit on paper -- and pages in 20 GB
//! per 200 tokens here. Three interleaved reps, 200 tokens:
//!
//! ```text
//!                            one expert at a time    a token's experts together
//!   Qwen3-30B q8, mmap          13.1  12.6  10.0          7.9   7.9   7.5
//!   Qwen3-30B q8, 16 GB cache   10.5   9.7   9.6         15.4  10.2  16.1
//!   Qwen3-Next q4, mmap         10.0   8.8  10.3          6.7   8.5   9.1
//!   Qwen3-Next q4, 16 GB cache   8.1   9.0   9.4         11.2  12.3  11.2
//! ```
//!
//! Batched compute wants its experts already in memory, and the cache is
//! what guarantees it; through a cold mapping the batch faults its way
//! through fourteen experts at once and loses. Advising the kernel which
//! experts were chosen before touching them (`madvise`, now the default)
//! recovers most of that without a cache -- 13.8 tok/s median on the 30B.
//! So the fetcher's worth depends on the kernel it feeds: inside the noise
//! for one expert at a time. With the batched path, against the old
//! default of mmap and one expert at a time, it is 1.22x on the 30B (15.4
//! against 12.6 by median) and 1.12x on the 80B (11.2 against 10.0) --
//! and about 1.1x over the batched path's own default, the `madvise` hint.
//!
//! So the fetcher is on by default for a mixture whose weights are over
//! the memory this machine gives them, at half that memory -- the size it
//! was measured at -- and off for one that fits, where it would only be a
//! second copy of what the mapping already holds. `KVAD_EXPERT_CACHE`
//! overrides both ways; see `experts::default_budget`. It gives the same
//! answer to the bit (`tests/deepseek.rs` holds it to that).
//!
//! What that rule leaves out: Qwen3-30B at q8 fits on paper, 32 GB against
//! 36, so it gets no cache -- and thrashes here all the same, which is
//! where the cache measured best. A fit with no room to spare is not a fit,
//! and the rule does not yet know where the room runs out.
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
        self.replay_pinned_from(policy, capacity, None)
    }

    /// As [`Log::replay`], but [`Policy::Pinned`] may choose its residents
    /// from a *different* trace.
    ///
    /// This is the difference between a claim and a measurement. Ranking
    /// experts by how often this very trace used them and then scoring
    /// against the same trace asks the cache to predict a future it has
    /// already seen; every real deployment picks its residents from
    /// yesterday's traffic and meets today's. Profiling on one prompt and
    /// replaying another is the honest version, and the gap between the two
    /// numbers is exactly how much of the hit rate was hindsight.
    pub fn replay_pinned_from(
        &self,
        policy: Policy,
        capacity: usize,
        profile: Option<&Log>,
    ) -> Outcome {
        let keys = self.accesses();
        let hits = match policy {
            Policy::Lru => lru(&keys, capacity),
            Policy::Pinned => {
                let ranked = profile.map_or_else(|| self.accesses(), Log::accesses);
                pinned(&keys, &ranked, capacity)
            }
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

/// The hottest `capacity` experts of `ranked_by`, resident for the whole
/// run, scored against `keys`.
///
/// Pass the same slice twice and the frequencies come from the trace being
/// replayed, which is the optimistic end: it asks the cache to predict a
/// future it has already seen. Pass a different trace and the answer is
/// honest, because that is what a deployment does — yesterday's profile
/// against today's traffic.
fn pinned(keys: &[u32], ranked_by: &[u32], capacity: usize) -> u64 {
    let mut count: HashMap<u32, u64> = HashMap::new();
    for &key in ranked_by {
        *count.entry(key).or_default() += 1;
    }
    let mut ranked: Vec<(u32, u64)> = count.into_iter().collect();
    // By count, then by key: ties would otherwise resolve on hash order and
    // the same trace would score differently run to run.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let resident: std::collections::HashSet<u32> =
        ranked.iter().take(capacity).map(|(key, _)| *key).collect();

    // The first access to a pinned expert is a miss in any honest
    // accounting — the cache starts empty and something has to load it —
    // but over thousands of tokens that is a rounding error, and a real
    // implementation would load the whole resident set at startup and pay
    // it once. Counted as a miss here, which errs against the policy this
    // is trying to make a case for.
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut hits = 0;
    for &key in keys {
        if resident.contains(&key) && !seen.insert(key) {
            hits += 1;
        }
    }
    hits
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
            let (l, p, o) =
                (lru(&keys, capacity), pinned(&keys, &keys, capacity), optimal(&keys, capacity));
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
        assert_eq!(pinned(&keys, &keys, 1), 3);
        assert_eq!(pinned(&keys, &keys, 2), 4);
    }

    /// A hot set chosen on other traffic still has to be scored against
    /// this traffic, and only the experts both agree on can hit.
    #[test]
    fn pinning_from_another_trace_scores_that_trace() {
        // The profile says 9 is the hottest thing in the world; the replay
        // never touches it, so one slot buys nothing.
        assert_eq!(pinned(&[0, 0, 0, 0], &[9, 9, 9, 1], 1), 0);
        // Two slots reach 0, whose three repeat visits then hit.
        assert_eq!(pinned(&[0, 0, 0, 0], &[9, 9, 9, 0], 2), 3);
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
