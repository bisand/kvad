# Plan: one model across several machines

Written 2026-09-20 against the working tree after `45a4621`, from a reading of
`crates/llm/src/model/{mod,arch,llama,deepseek}.rs`, `qcache.rs`, `weights.rs`,
`service.rs`, `machine.rs`, `crates/gpu/src/model.rs` and
`crates/serve/src/{engine,scheduler}.rs`. Nothing here has been measured. Every
number is arithmetic from published specifications and says so; Phase 0 exists
to replace them with measurements before any engine code is written.

## What the owner asked for

Run Kvad on several Macs (or PCs) joined by their USB-C ports, and spread a
model too large for one of them across all of them, through installable
workers that carry the engine but not the API or the web UI. A machine that
already has Kvad installed should be usable as it is. Daisy-chained Mac minis
and Mac Studios are the obvious case; old mining rigs are the hopeful one. Is
it doable, and what is the plan?

## Is it doable

Yes, and the engine is closer to it than it looks. Three reasons.

**The traffic is tiny.** Split a model by layers and the only thing that
crosses between machines is the residual stream: `n_embd` floats per token.
For Llama-3.3-70B that is 8192 × 4 = 32 KB per token per hop. Thunderbolt
moves that in microseconds; gigabit Ethernet in a quarter of a millisecond.
Against a 70B decode step that takes 100 ms or more of memory traffic, the
link is noise. Prefill sends `m` rows at once — a 64-token chunk
(`KVAD_PREFILL_CHUNK`'s default) is 2 MB — which is about 1 ms on Thunderbolt
and 17 ms on gigabit.

**Others have done it.** exo, llama.cpp's RPC backend, MLX's distributed
backends, distributed-llama and Petals all run one model across consumer
machines, most of them by exactly this layer split. None of them is an engine
you can read end to end, which is the gap Kvad is for.

**The seams already exist.**

- `Session` (`model/mod.rs`) is tokens in, logits out, with `truncate` and
  `cached`. Everything above it — `Llm`, sampling, prefix reuse, the
  scheduler, the server — is indifferent to what is behind it. A session that
  happens to span four machines is one more `impl Session`.
- `Source` (`qcache.rs`) hands out weights *by tensor name, on request*. A
  model that never asks for layer 40 never touches layer 40, so a worker
  holding layers 0–19 needs no new loading machinery, only a model that asks
  for less.
- `llama.rs::run_batch` already separates "every block" from "the output
  head", because scoring and generation wanted different slices of it.
- `Engine::spawn(loader)` already takes its loader from the binary, because
  the GPU crate depends on `kvad` and not the other way round. A worker needs
  the same trick and can use the same one.
- `machine::usable_memory()` already answers the question a placement planner
  asks first.
- `deepseek.rs` already notes that in V3's routing "a device holds a group".
  The checkpoint was designed to be split.

### What it buys, said plainly

**A cluster makes a model possible, not faster.** With one conversation in
flight, a layer split runs the stages one after another: machine B waits while
A works. Time per token is the *sum* of the stage times plus the hops, so four
Mac minis decode a 70B model at roughly the speed one Mac with four times the
memory would — and never faster than that. The memory adds up; the bandwidth
does not.

Where it gets *better* than one machine:

- **Prefill**, once chunks are pipelined: A starts chunk 2 while B runs
  chunk 1. This needs no batching work and fits the existing chunk loop.
- **Several conversations at once**, after continuous batching (README step
  4): different sequences occupy different stages at the same moment. The wire
  protocol carries a sequence id from day one so that this does not mean a
  second protocol.
- **Mixture-of-experts models.** DeepSeek-V3 is 671B parameters with 37B
  active per token: enormous to hold, cheap to run. That is exactly the shape
  a pile of Mac Studios suits, and Kvad already runs the architecture.

### The links

Arithmetic and commonly reported figures, to be replaced in Phase 0:

| link | bandwidth | round trip | verdict |
|---|---|---|---|
| Thunderbolt 4 / USB4 bridge | 40 Gb/s nominal, ~15–25 Gb/s as IP | ~0.2–0.5 ms | ideal |
| Thunderbolt 5 | 80 Gb/s nominal | same as IP; far lower with RDMA | ideal |
| 10 GbE | 10 Gb/s | ~0.1–0.2 ms | ideal |
| 1 GbE | 1 Gb/s | ~0.3 ms | fine for decode, slows long prefill |
| Wi-Fi | varies | 2–10 ms, jittery | works, not recommended |

Things worth knowing before buying cables:

- macOS presents connected Thunderbolt ports as an ordinary network interface
  (*Thunderbolt Bridge*, link-local addresses). So **the transport is TCP**,
  and Thunderbolt, Ethernet and Wi-Fi are one code path.
- Not every USB-C port is Thunderbolt. Every Apple-silicon Mac's are. Many
  PCs' are USB 3 only and cannot network host to host; those use Ethernet.
- A chain A—B—C works because B bridges, in software. Three Macs with two
  ports each can form a full triangle; beyond that it is a chain or a ring,
  and the pipeline order should follow the cables.
- macOS 26.2 added RDMA over Thunderbolt 5, which MLX uses. It matters for
  tensor parallelism (below), not for a layer split. To verify, not assume.

### PCs and mining rigs

The protocol does not care what is on the other end; the kernels do.

- **x86 CPUs.** `simd.rs` and the integer kernels in `quant.rs` are NEON,
  `dotprod` and `i8mm` — aarch64 only, with a scalar fallback elsewhere. A PC
  worker would run and be correct and be slow. AVX2 kernels are their own
  chapter and a prerequisite for a PC being worth adding.
- **NVIDIA rigs.** `kvad-gpu` is candle with the `metal` feature hard-wired.
  candle has a `cuda` feature, so this is a Cargo feature and a device
  constructor rather than a rewrite — but `GpuLlama` is Llama only. A rig's
  x1 PCIe risers, fatal for most multi-GPU work, do not matter here: only
  activations cross them. Its usual Celeron and 4–8 GB of system RAM do
  matter: loading is slow, and the CPU path is useless on it.
- **AMD rigs.** No backend in this repository can drive them. Out of scope
  until there is a Vulkan or wgpu backend, which nothing else here needs.

So: Macs first, because that is where every kernel already is. Linux and
NVIDIA second, gated on kernels rather than on anything in this plan.

## The design

### Split by layers; star topology; the coordinator keeps both ends

```text
  coordinator                         worker 1            worker 2
  ───────────                         ────────            ────────
  tokens → embed
           layers 0..19   ─ hidden →  layers 20..49
                          ← hidden ─
                          ─ hidden ───────────────────→   layers 50..79
                          ← hidden ────────────────────
           final norm, head → logits → sampler
```

**The coordinator holds the embedding *and* the output head.** The obvious
design puts the head on the last machine and sends logits back; for a 128k
vocabulary that is 513 KB per token, and `forward_all` (perplexity) would send
that per *position*. Sending the hidden state back instead is 32 KB, and with
tied embeddings the table then exists once in the cluster instead of twice.
Sampling, the explained candidates and perplexity all stay where they are.

**Star, not chain, to begin with.** Every hidden state returns to the
coordinator, which sends it on. That is two hops per worker where a chain
would need one, and costs well under a millisecond per token at these sizes.
What it buys is that a remote stage and a local one are the same trait, the
coordinator is the only machine that knows the plan, and a worker never
connects to another worker. Chain forwarding is Phase 6, if Phase 4's numbers
ask for it.

**Each stage owns the KV cache for its own layers.** Nothing about the cache
crosses the wire except `truncate(len)`, which the coordinator broadcasts.
`CacheShape` already lets DeepSeek's latent cache differ from everyone
else's, and it keeps working per stage.

**Stages need not match.** Activations cross as f32, so a Metal stage, a
`cpu q8` stage and one day a CUDA stage can sit in one pipeline. f16 on the
wire would halve 32 KB that was never the problem and cost exactness.

### What changes in the engine

One real refactor, in `crates/llm/src/model/`:

```rust
/// Which part of a model one process holds.
pub struct Shard {
    pub layers: std::ops::Range<usize>,
    /// The embedding, final norm and output head. Exactly one shard has them.
    pub ends: bool,
}

pub trait Transformer: Send + Sync {
    fn spec(&self) -> &Spec;
    fn shard(&self) -> &Shard;
    /// tokens -> [m, n_embd]. Needs `ends`.
    fn embed(&self, tokens: &[u32], pos0: usize) -> Vec<f32>;
    /// This shard's blocks, in place over [m, n_embd].
    fn blocks(&self, xs: &mut [f32], m: usize, cache: &mut KvCache);
    /// [m, n_embd] -> logits, for the last row or for all of them. Needs `ends`.
    fn head(&self, xs: &[f32], m: usize, all: bool) -> Vec<f32>;

    // forward, forward_batch and forward_batch_all become default methods
    // written in terms of the three above, and every existing caller and
    // test keeps working.
}
```

- `Architecture::load` gains the `Shard`. A whole model is
  `Shard { layers: 0..n_layer, ends: true }`, which is what every caller
  passes today.
- `KvCache::new` takes the layer count of the shard; blocks index it by
  `layer - shard.layers.start`.
- Llama and DeepSeek each have a hand-written `m = 1` path beside the batched
  one. Whether `blocks` keeps both or decode becomes `m = 1` of the batched
  path is a measurement, not a preference — see the README on thresholds
  calibrated through a bug.

Then a new module, `crates/llm/src/cluster/`, behind a `cluster` feature:

| file | what it is |
|---|---|
| `wire.rs` | The framing. `u32 length, u8 kind, payload`. Control messages are JSON (`serde_json` is already here); tensors are a fixed header — `seq, pos0, m, n_embd` — and little-endian f32. `std::net`, `TCP_NODELAY`, no async: the `kvad` crate has no tokio and does not need one. Hand-written, because 200 lines of framing is something this repository can teach and gRPC is something it would have to apologise for. |
| `stage.rs` | `trait Stage { fn run(&mut self, seq, pos0, m, xs) -> Res<Vec<f32>>; fn truncate(..); }`, with `LocalStage` (a `Transformer` shard plus its cache and pool) and `RemoteStage` (a socket). |
| `session.rs` | `PipelineSession: Session`. Embeds, walks its stages in order, applies the head. This is the only thing `Llm` ever sees. |
| `worker.rs` | `serve(listener, key, factory)`: accept a coordinator, answer `Hello`, build a stage on `Load`, run `Forward` frames until told otherwise. The `factory` comes from the binary, as `Engine::spawn`'s loader does, so a GPU-capable binary makes GPU stages. |
| `plan.rs` | Placement, below. |

Messages: `Hello` (protocol version, git revision, architectures in this
build, backends, usable memory, thread count), `Load` (repo, revision,
precision, shard), `Progress`, `Ready`, `Forward`/`Hidden`, `Truncate`,
`Unload`, `Ping`, `Error`. A coordinator refuses a worker whose revision
differs from its own: two builds with different kernels can both be right and
still not agree to the bit, and that is a bug report nobody can act on.

### The worker, and "if Kvad is already installed"

No new binary. Two entry points to the same `worker::serve`:

- `kvad worker --listen 0.0.0.0:7420` — the CLI binary, CPU stages. It has no
  HTTP server, no SQLite, no web assets; it is the thin agent asked for.
- `kvad-serve` with `[worker] enabled = true` in `kvad.toml` — an installed
  server also offers itself as a worker, and because it is the binary with
  the `gpu` feature, its stages can be Metal ones. A worker that is busy
  serving its own model says so in `Hello` rather than swapping.

On the coordinator: `kvad run --model X --workers host:port,host:port`, and a
`[cluster]` table in `kvad.toml` for the server. Discovery by mDNS
(`_kvad._tcp`) comes in Phase 5; an address list comes first, because it is
debuggable.

### Weights

Each worker fetches its own. `weights::fetch_watched` downloads every shard
file today; it gains a filter. `model.safetensors.index.json` maps tensor
names to files, and a tensor whose name carries a layer number outside the
shard's range (`.layers.N.`, or GPT-2's `.h.N.`) is not needed, nor are the
embedding and head on a shard without `ends`. Fetch only the files that still
have a wanted tensor in them.

The quantised cache already does the right half of this: `qcache.rs`
quantises a tensor when `Source::matrix` asks for it and records only what
was asked, so a shard's first load quantises its own layers and nothing else.
The other half is a trap. The cache file is per model and precision, and the
mapped source treats a name missing from the file as a tensor missing from
the model ("the run that wrote it asked the same question and got nothing").
A cache written by a worker holding layers 20–49, later mapped by a
whole-model load on the same machine, would fail on layer 0 — or worse,
`try_matrix("lm_head.weight")` would quietly answer `None` and an untied
model would run with the wrong head. **The shard has to be part of the
cache's identity**, beside the version and the checkpoint's file sizes, and
Phase 3 has a test for exactly this.

Later, and only if it is asked for: a worker pulling files from the
coordinator over the Thunderbolt link instead of from the Hub, which for a
400 GB model is the difference between minutes and a day.

### Placement

Single-stream time per token is `Σ layers_i × seconds_per_layer_i + hops ×
round_trip`. It is a sum, not a maximum, so the slowest machine does not set
the pace — it only costs what its own layers cost. That makes the planner
simple:

1. Ask each node for usable memory (`machine::usable_memory`) and time one
   block of the target shape on it.
2. Reserve the ends on the coordinator, and on every node the KV cache for
   its layers at the context the user asked for — 655 KB per token for a 70B
   in f32, which is 5 GB at 8k and the reason this is not an afterthought.
3. Fill the fastest node to its limit with contiguous layers, then the next.
   Use as few nodes as will hold the model: every further node adds a hop and
   nothing else.
4. Print it. `kvad cluster plan MODEL --workers …` shows the table — node,
   layers, weights, cache, headroom — before a byte is downloaded, and says
   *does not fit* when that is the answer.

**And a way to overrule it.** Rule 3 is right for running a model and wrong
for developing this feature: any model small enough to iterate on fits on the
coordinator, so the planner would give every worker nothing and no byte would
ever cross the wire. `--split 0..12,12..24` names the layer ranges by hand,
one per node in the order `--workers` lists them, coordinator first, and
skips the planner entirely. It is checked — contiguous, covering every layer
once — and the memory check still runs and still refuses. This is not a test
hook. It is what the loopback tests in Phase 2 use, what a development rig
uses, and what an operator who knows something the planner does not will
reach for.

### A development rig

Two Macs of any vintage and one cable. The one this plan was written beside
is an M5 Pro MacBook Pro (48 GB, Thunderbolt 5) and an M2 MacBook Air (8 GB,
Thunderbolt 3/USB4); the link negotiates to 40 Gb/s and the chips differing
is of no interest to TCP.

- **The cable has to be a Thunderbolt or USB4 one.** The USB-C charging
  cables Apple ships carry USB 2 data. With one of those the Thunderbolt
  Bridge never comes up, and it looks exactly like a software fault.
- With the right cable, System Settings → Network → Thunderbolt Bridge shows
  a self-assigned `169.254.x.x` address on each side. The firewall may ask
  about incoming connections the first time a worker listens.
- Both machines build the same revision; the handshake insists.
- The M2 has `i8mm`, so both take the same kernels. A pair where one side
  lacks it (an M1) is a different question: the integer sums are exact
  whichever instruction computes them, but the block scales are applied in
  f32 afterwards, and whether the tiled and the row-at-a-time paths do that
  in the same order has not been checked. Such a pair may agree to the bit
  or only to a rounding, and the revision check would not notice either way.
  Unverified; worth a test the day somebody has an M1 to hand.

A lopsided pair is more useful than a matched one. It is enough for all of
Phase 0. Its speed difference tests the claim that time per token is a sum
and not a maximum: move layers from the fast machine to the slow one with
`--split` and the rate should fall by what those layers cost there, and no
more. And 8 GB — about 6 GB usable under `USABLE_FRACTION` — exercises the
planner's *does not fit* and its headroom arithmetic, which two large
machines never would.

What it cannot show is Phase 4's headline. With 48 + 8 GB there is next to
nothing that fits the pair and not the larger machine alone. That needs a
second machine with real memory, and is the only phase that does.

### Security

A worker downloads what it is told to and gives a stranger its memory, so it
does not run open.

- A pre-shared cluster key, required. `kvad worker` generates one on first
  start, stores it in `data_dir()`, prints it once. The handshake is an HMAC
  challenge in both directions; the key never crosses the wire.
- `Load` accepts Hub repo ids and the worker's own model names, never a path.
- v1 sends activations in the clear and says so. They are not the prompt, but
  they are not nothing. TLS is a later option; `rustls` is already in the
  tree by way of `hf-hub`.

### Failure

A socket that dies mid-token fails the generation with an error naming the
node; the scheduler reports it as it reports any failed load. The cluster
does not heal, re-plan or retry in v1. A worker whose coordinator vanishes
drops its stage and its cache and goes back to listening.

## What is deliberately not in the first version

- **Tensor parallelism.** Splitting each matmul across machines is the only
  way several machines make *one* conversation faster, and it costs two
  all-reduces per layer: 160 synchronisations per token on a 70B. At 0.3 ms
  each over TCP that is 48 ms of pure waiting per token, which eats most of
  what it saves. It wants RDMA, which wants Thunderbolt 5 and macOS 26.2. A
  later chapter, and an honest one only after the layer split is measured.
- **Expert parallelism** for DeepSeek: experts on different machines, inside
  one layer. The right design for V3 eventually, and two hops per MoE layer
  instead of one per stage. After the layer split, which already runs V3.
- **Chain forwarding, elastic membership, mixed-revision clusters.**

## Phases

Each one ends with something that runs and a test that says it is right. In
the house style: one at a time, verified, left uncommitted with paths listed.

**Phase 0 — measure the links.** `kvad cluster ping HOST`, and the listener
for it: round trip and throughput for 32 KB and 2 MB frames, medians of
interleaved runs, over Thunderbolt Bridge, Ethernet and Wi-Fi between two
real Macs. Replace the table above. If a Thunderbolt round trip is 5 ms
rather than 0.3, this plan changes here and not in Phase 4. *This is also
where `wire.rs` gets written and tested, so it is not throwaway.*

**Phase 1 — split the model, in one process.** `Shard`, the three-method
`Transformer`, shard-aware `load` and `KvCache` for GPT-2, Llama and DeepSeek.
`LocalStage` and `PipelineSession` with every stage local. No sockets. Test:
for each architecture, a model cut into 1, 2 and 3 shards produces logits
**bit-identical** to the uncut one, through prefill, decode, `truncate` and
`forward_all`. Then the measurement: decode and prefill tok/s cut versus
uncut, five interleaved rounds — the refactor must cost nothing, and if it
does, that is found here.

**Phase 2 — a worker on loopback.** `worker.rs`, `RemoteStage`, the
handshake and the key, `kvad worker`, `--workers`, and `--split`, because
without it a small model never leaves the coordinator. Test: a worker on
port 0 in a thread, same bit-identical assertions as Phase 1; plus wrong key,
revision mismatch, worker killed mid-generation, cancel mid-prefill, and a
`--split` with a gap, an overlap or the wrong number of ranges. Then the same
thing by hand across the development rig's cable, which is the first time a
hidden state leaves a machine.

**Phase 3 — load only your share.** The fetch filter, the shard in the
quantised cache's identity, per-node memory checks, `plan.rs` and
`kvad cluster plan`. Tests: a sharded tiny checkpoint where a worker's shard
provably opens only its own files; and a shard load followed by a whole-model
load of the same checkpoint, which must not map the shard's cache.

**Phase 4 — two real machines.** A model that does not fit on either. Decode
and prefill against Phase 0's prediction of `Σ stages + hops × RTT`, and an
account of the difference. Pipelined prefill chunks, measured before and
after. The README chapter gets written here, from the numbers.

**Phase 5 — the server.** `[cluster]` and `[worker]` in `kvad.toml`, worker
mode inside `kvad-serve`, a Cluster page (nodes, memory, layers per node,
per-hop latency, a per-stage share of each token), mDNS discovery, load
progress per node over the existing SSE. GPU stages: `GpuLlama` gains the
hidden-in, hidden-out form of `run`, one host round trip of `n_embd` floats
per stage boundary.

**Phase 6 — what the numbers ask for.** Chain forwarding. Peer weight
transfer. AVX2 kernels, then Linux workers. A `cuda` feature for `kvad-gpu`.
Expert parallelism. Tensor parallelism over RDMA. And when continuous
batching lands, micro-batches that keep every stage busy — which is the point
at which a cluster stops being only about memory.

## Open questions

- Does the Thunderbolt Bridge add enough software latency on the middle
  machine of a chain to matter? Phase 0, with three machines if there are
  three.
- Is a split model really bit-identical? It should be: each output row of a
  matmul is computed by one thread whatever the split, and attention is per
  head. Phase 1's test is the answer, not this paragraph.
- Should the coordinator hold layers at all when it is also running
  `kvad-serve`, a database and a browser? Probably yes, fewer of them — the
  planner's headroom figure is where that is expressed.
- A worker with a GPU and a slow CPU (the mining rig) can hold no `ends` and
  should never be a coordinator. Does the planner need to know, or is that
  the operator's job? Start with the operator's.
