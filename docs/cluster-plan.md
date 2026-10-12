# Plan: one model across several machines

Written 2026-09-20 against the working tree after `45a4621`, from a reading of
`crates/llm/src/model/{mod,arch,llama,deepseek}.rs`, `qcache.rs`, `weights.rs`,
`service.rs`, `machine.rs`, `crates/gpu/src/model.rs` and
`crates/serve/src/{engine,scheduler}.rs`. Nothing here has been measured
except where a line gives a date and says what was measured. Every other
number is arithmetic from published specifications and says so; Phase 0
exists to replace them with measurements before any engine code is written.

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
| USB 2 NCM — a charging cable | 0.30 Gb/s measured | 1.1 ms empty, 2.7 ms for 32 KB | a trap, not a choice; see *A development rig* |

Things worth knowing before buying cables:

- macOS presents connected Thunderbolt ports as an ordinary network interface
  (*Thunderbolt Bridge*, link-local addresses). So **the transport is TCP**,
  and Thunderbolt, Ethernet and Wi-Fi are one code path.
- Not every USB-C port is Thunderbolt. Every Apple-silicon Mac's are. Many
  PCs' are USB 3 only and cannot network host to host; those use Ethernet.
- A chain A—B—C works because B bridges, in software. Three Macs with two
  ports each can form a full triangle; beyond that it is a chain or a ring,
  and the pipeline order should follow the cables. This is a property of the
  Thunderbolt Bridge specifically, which is a real layer-2 bridge: every
  Thunderbolt port is a member of `bridge0`, so B forwards frames between A
  and C, all three share one broadcast domain, and there is no routing to
  configure. It also means B bridges only while B is awake. A USB NCM link
  joins no bridge and is strictly two hosts, and macOS does not route between
  interfaces by default (`net.inet.ip.forwarding` is 0) — so a machine on the
  far end of a charging cable is an endpoint and can never be a middle.
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
  constructor rather than a rewrite. The architecture gap that used to sit
  here is closed: the backend runs the Llama family, GPT-2 and DeepSeek V2
  as of v0.2.0, and `deepseek_v3` is one dispatch arm away. A rig's
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

**As built** (Phase 1), with three differences from the sketch above, each
for a reason:

- `forward` stays a method of its own. Llama and GPT-2 each have a
  hand-written one-row pass beside the batched one, and they are what the
  batched passes are tested against. What a session calls, whole or cut, is
  the batched pass at every `m`, so the two cannot differ by which kernel a
  single row took.
- A `splits()` method says whether an architecture has the three. Qwen3.5
  does not yet — its pass is one token at a time — and loads whole or not
  at all. The defaults of `forward_batch` and `forward_batch_all` are the
  three in a row where they exist, and the old loop where they do not.
- `embed` takes `pos0` for GPT-2 alone, which adds position at the door.
  Everything since puts it in inside attention, where each shard counts it
  from its own cache.

A shard load reads its own layers and names the rest as left to others, a
layer at a time, so the loader's check for weights nobody read still means
what it meant. It quantises from the checkpoint on every load and stays out
of the cache file; that is Phase 3's, below.

The sketch, as it was written:

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
| `wire.rs` | The framing. `u32 length, u8 kind, payload`. Control messages are JSON (`serde_json` is already here); tensors are a fixed header — `seq, pos0, m, n_embd` — and little-endian f32. `std::net`, `TCP_NODELAY`, no async: the `kvad` crate has no tokio and does not need one. Hand-written, because 200 lines of framing is something this repository can teach and gRPC is something it would have to apologise for. **Written**, with `Hello`, `Error`, the tensor frames and the four kinds a measurement uses. |
| `ping.rs` | The listener that sends frames straight back and the sweep made against it: `kvad cluster listen` and `kvad cluster ping`. **Written.** |
| `link.rs` | What a link is, from the registry and not from the socket, and who is on each cable: `kvad cluster link` and `kvad cluster find`. **Written.** |
| `stage.rs` | `trait Stage { fn run(&mut self, seq, pos0, m, xs) -> Res<Vec<f32>>; fn truncate(..); }`, with `LocalStage` (a `Transformer` shard plus its cache and pool) and `RemoteStage` (a socket). **`LocalStage` is written.** It keeps a cache a conversation, refuses rows that are not at the end of what it holds, and has no pool: the pass runs inside the session's, and a worker will bring its own. |
| `session.rs` | `PipelineSession: Session`. Embeds, walks its stages in order, applies the head. This is the only thing `Llm` ever sees. **Written**, for one conversation: it does not yet set a cache aside, so a model in stages keeps one conversation where a model whole keeps several. |
| `worker.rs` | `serve(listener, key, factory)`: accept a coordinator, answer `Hello`, build a stage on `Load`, run `Forward` frames until told otherwise. The `factory` comes from the binary, as `Engine::spawn`'s loader does, so a GPU-capable binary makes GPU stages. |
| `plan.rs` | Placement, below. |

Messages: `Hello` (protocol version, git revision, architectures in this
build, backends, usable memory, thread count), `Load` (repo, revision,
precision, shard), `Progress`, `Ready`, `Forward`/`Hidden`, `Truncate`,
`Unload`, `Ping`, `Error`. A coordinator refuses a worker whose revision
differs from its own: two builds with different kernels can both be right and
still not agree to the bit, and that is a bug report nobody can act on.

As written so far, `Hello` carries the protocol version, the package version,
the machine's name, its usable memory, its thread count and its
architectures. The package version stands in for the git revision until
Phase 2, which is the phase that refuses on it; `kvad cluster ping` only says
when the two differ.

### The worker, and "if Kvad is already installed"

No new binary. Two entry points to the same `worker::serve`:

- `kvad worker --listen 0.0.0.0:7420` — the CLI binary, CPU stages. It has no
  HTTP server, no SQLite, no web assets; it is the thin agent asked for.
- `kvad-serve` with `[worker] enabled = true` in `kvad.toml` — an installed
  server also offers itself as a worker, and because it is the binary with
  the `gpu` feature, its stages can be Metal ones. A worker that is busy
  serving its own model says so in `Hello` rather than swapping.

On the coordinator: `kvad run --model X --workers host:port,host:port`, and a
`[cluster]` table in `kvad.toml` for the server. An address list stays, as
the override, because it is debuggable — but it is not enough by itself, and
the next section is why.

### Addressing and discovery

An address list is the wrong *first* step, and the development rig is why.
Over a cable the peer is `fe80::100d:68ff:f0c0:4a08%en12`. Nobody types that
twice; it changes when the cable moves or the machine reboots; and the
interface it is scoped to has a different name on each end. Static addresses
are worse, because they have to be applied by hand to a machine that is
otherwise stock, and re-applied when it is not.

Three pieces. None of them is new work.

**A machine is an identity, not an address.** A worker persists an id once,
and `Hello` carries it beside the revision it already carries. Everything
else rests on this, because one peer honestly has several addresses: a
link-local per cable, a LAN address over Wi-Fi, a tailnet address if one is
running. The same id answering on four of them is one machine with four
paths, not four workers, and the planner must not count its memory four
times.

**Discovery by mDNS, and earlier than Phase 5.** `mDNSResponder` is running
on every Mac already. A worker registers `_kvad._tcp` with its id and its
revision in TXT; the coordinator browses, and gets every address each peer
can be reached on — including a link-local on a cable plugged in a moment
ago. Nothing to configure and nothing to redo when a cable moves. It belongs
in Phase 2, with the first worker that is not on loopback, because that is
the phase where the addresses first become unspeakable.

**The coordinator measures its paths and pins the fastest.** For each id,
ping every candidate address — the Phase 0 measurement, over the Phase 0
listener — and keep the best. This is the piece that earns itself twice. It
is also the only thing that would have caught a 480 Mb/s charging cable,
which presents as a working link and is distinguishable from Thunderbolt by
nothing a socket can observe. A cluster that reports the chosen path and its
speed for each peer cannot silently record a Phase 4 number against a link
nobody identified.

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
Thunderbolt 3/USB4); the ports are good for 40 Gb/s between them and the
chips differing is of no interest to TCP.

- **Measuring anything needs a Thunderbolt or USB4 cable.** The USB-C charging
  cables Apple ships carry USB 2 data. With one of those the Thunderbolt
  Bridge never comes up, and it looks exactly like a software fault.

  Worse than that, measured on this pair on 2026-09-20: it does not look like
  a fault at all. macOS falls back to USB NCM without saying so. An ordinary
  interface appears — `en12` here, not a bridge member and with no entry in
  System Settings — takes a self-assigned `169.254.x.x`, answers `ping6` on
  its link-local address and carries TCP; SSH to the other Mac over it works
  first try. Everything a connectivity test would check passes. What it is:
  302 Mb/s one way, 63% of the 480 the nominal rate promises, and an empty
  hop of about 1.1 ms — slower than the gigabit Ethernet row above and about
  a hundredth of what the two ports could do. A worker would run on it and
  simply be slow, which is the failure this plan is least able to see. Three
  checks tell the truth where `ifconfig` does not:

  | check | charging cable | Thunderbolt cable |
  |---|---|---|
  | `system_profiler SPThunderboltDataType` | *No device connected*, on every bus | the peer Mac, by name |
  | `ifconfig bridge0` | `status: inactive` | active, with an address |
  | `ioreg -p IOService -n enN -r -t` | `AppleUSB20XHCIARMPort` → `AppleUSBNCM11Data` | `AppleThunderboltIP…` |

  The registry path is the one that cannot be misread: the port it hangs
  off is named `usb-drd1-port-hs`, and *hs* is USB 2.0 High Speed. Phase 0
  should print these three checks before it prints a number, so that no
  measurement is ever recorded against a link nobody identified.
- **Until a Thunderbolt cable is to hand, build on the USB one.** It is the
  wrong link to measure over and a perfectly good one to develop against.
  Correctness does not depend on bandwidth: Phases 1 to 3 are bit-identical
  logits, a handshake, a key, shard loading and the failure paths, and every
  one of those passes or fails the same way at 480 Mb/s as at 40 Gb/s. What
  waits for the real cable is every *number* — the table above, Phase 0's
  replacement of it, and Phase 4 entirely.

  Measured over that cable on 2026-09-20 with `scripts/bench-link.py`,
  medians of ten interleaved rounds after a discarded warm-up, every median
  within 4% of its own minimum:

  | frame | round trip | link rate |
  |---|---|---|
  | 32 KB — a decode hop | 2.71 ms | 193 Mb/s |
  | 2 MB — a prefill chunk | 108.9 ms | 308 Mb/s |
  | fitted | 1.10 ms empty + 314 Mb/s | 302 Mb/s one way over 64 MB |

  Being slow is briefly an advantage. A 2.7 ms decode hop is plainly visible
  in the tok/s of a small model, where Thunderbolt would hide it inside a
  millisecond, so a hop that happens when it should not — or twice where it
  should happen once — is loud on this cable and quiet on the good one.
  Prefill is the reverse: 109 ms the round trip and 54 each way, against the
  1 ms this plan expects of Thunderbolt and the 17 of gigabit, so pipelined
  prefill cannot be judged on this cable at all.

  Measured again on 2026-10-12 with `kvad cluster ping`, the same pair and
  the same kind of cable, ten rounds: 2.61 ms at 32 KB and 106.1 ms at 2 MB,
  fitted to 0.86 ms empty and 318 Mb/s, 315 Mb/s one way. Three runs of each
  tool in turn put `kvad cluster ping` and the script within 4% of each
  other at 32 KB and at 2 MB. At 1 KB the script is 0.1 to 0.2 ms the
  slower, on a hop of 0.7 ms, which is Python's own share of it.

  One thing worth knowing before trusting a run. The first sweep after the
  Air woke and the cable came back reported 22.6 ms at 32 KB — a median, not
  a stray sample, and above its own 128 KB row, which is impossible. It has
  not reproduced in two runs since, at any size. A freshly re-established NCM
  link wants more warm-up than the one round the tool discards; until that is
  understood, a number from the first sweep after a reconnect should be
  thrown away rather than recorded.

  It reproduced on 2026-10-12, and not after a reconnect: 34.4 ms at 32 KB
  from `kvad cluster ping` and 37.8 ms from the script in the run straight
  after it, each a median of seven with its minimum where it belongs at
  2.5 ms, and 1 KB and 2 MB untouched in both. Then gone, for the sixteen runs after it.
  So it is a stretch of some twenty seconds on the link or on one of the
  machines, not something either tool does, and it began just after a file
  was copied to the Air over the same cable. Still not understood. What
  catches it is the last column of the table — a median thirteen times its
  own minimum — and the line under the table that says such a run is worth
  repeating rather than quoting.

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

It splits in two, because the rig has no Thunderbolt cable yet. The tool,
the listener, `wire.rs` and the link identification — the three checks from
*A development rig*, printed before any number — can be built and tested now
over USB and loopback, and they are what the later phases actually depend
on. The numbers that replace the table wait for the cable; nothing between
here and Phase 4 is blocked on them. What must not happen is the table being
filled in from whichever link is to hand.

The first half is done: `kvad cluster listen` on one end, and on the other
`kvad cluster find` for who is on each cable, `kvad cluster ping` to measure
— with no address it measures the one machine `find` finds — and
`kvad cluster link IFACE` for the identification alone. `ping` works out
which interface its connection left by and identifies that, so a link is
named even when it was reached by a hostname. The Thunderbolt branch of the
identification is written from this document's description of what a
Thunderbolt link looks like and has never seen one; the day there is a
cable, check what it prints against the three checks by hand before trusting
it. The listener sends back what it is sent, up to 64 MB a frame, to anybody
who can reach its port: it has no key, because it has nothing to give, and
it is not the worker.

The tool has a floor of its own, and Thunderbolt is fast enough to stand on
it. Over loopback on the M5 Pro, where no link is crossed, a 32 KB round trip
is 0.09 ms and a 2 MB one 1.3 ms: the frame is copied through the kernel
four times and nothing else happens. The 2 MB figure is what this plan
expects of Thunderbolt itself. Over the charging cable it is one part in
eighty and does not matter; over the real cable it has to be measured beside
every number and said with it, or the table will record the tool.

`scripts/bench-link.py` was the stopgap, and still is for a machine with no
kvad on it. It needs no engine and nothing installed on either machine: `--find` to locate a peer over a cable,
`--listen` on one end and `--connect` on the other, a sweep from 1 KB to
8 MB interleaved across rounds so that no size records one stretch of a
busy machine, and the link identified before any number is printed. It
measures a round trip rather than one-way bandwidth, because a stage
boundary is a round trip and because an iperf3 figure would describe this
link's ceiling while saying nothing about what a 32 KB hop pays before its
first byte moves. `kvad cluster ping` should keep that shape.

**Phase 1 is done.** Every cut of every architecture that splits — whole,
in two, in three, the ends on a shard with no layers, the ends on the last
shard — answers to the bit as the model whole, at f32, q8 and q4, through a
prompt read in several passes, decode, a rewind and every position's logits
(`tests/split.rs`, and in `tests/deepseek.rs` for V2 and V3). On real
weights `examples/split.rs` checks every logit of a 256-token prefill and 48
decoded tokens, and then times it. Qwen2.5-1.5B at q8 on the M5 Pro, nine
rounds with whole and cut back to back, medians, cut over whole:

| shards | prefill | decode |
|---|---|---|
| 1 | 0.97 | 0.99 |
| 2 | 1.00 | 0.97 |
| 3 | 1.01 | 1.03 |
| 4 | 1.01 | 1.00 |

That is no cost that this machine can show: the best rounds of whole and cut
are within 3% of each other in every row, and the medians wander by 3% either
way on a desktop that was not idle. SmolLM2-135M says the same at f32 and
q8. Against the build before the refactor, the model whole writes the same
text at the same speed (0.97 to 1.09 over two models and two precisions,
five rounds; 1.00 on the one repeated with nine).

One thing the measurement turned up that is not about cutting. With the
whole model mapped from its cache file and the shards quantised onto the
heap, the shards were 4% faster in all eight cells, one shard included;
with both read the same way the difference went. So weights on the heap
ran about 4% faster than the same weights mapped, on one model, once. Not
chased, and worth a measurement of its own before anybody believes it.

**Phase 1 — split the model, in one process.** `Shard`, the three-method
`Transformer`, shard-aware `load` and `KvCache` for GPT-2, Llama and DeepSeek.
`LocalStage` and `PipelineSession` with every stage local. No sockets. Test:
for each architecture, a model cut into 1, 2 and 3 shards produces logits
**bit-identical** to the uncut one, through prefill, decode, `truncate` and
`forward_all`. Then the measurement: decode and prefill tok/s cut versus
uncut, five interleaved rounds — the refactor must cost nothing, and if it
does, that is found here.

**Phase 2 — a worker on loopback.** `worker.rs`, `RemoteStage`, the handshake
and the key, `kvad worker`, `--workers`, and `--split`, because without it a
small model never leaves the coordinator. Also the machine id, `_kvad._tcp`
and the path probe from *Addressing and discovery*: this is the phase where a
peer stops being `127.0.0.1` and starts being a scoped link-local address,
which is the point at which typing addresses stops working. Test: a worker on
port 0 in a thread, same bit-identical assertions as Phase 1; plus wrong key,
revision mismatch, worker killed mid-generation, cancel mid-prefill, and a
`--split` with a gap, an overlap or the wrong number of ranges. Then the same
thing by hand across the development rig's cable, which is the first time a
hidden state leaves a machine — found by discovery rather than by an address,
and with the chosen path and its speed printed, so that what it crossed is on
the record.

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
per-hop latency, a per-stage share of each token, and for each peer the chosen
path and its measured speed), load progress per node over the existing SSE.
GPU stages: `GpuLlama` gains the hidden-in, hidden-out form of `run`, one host
round trip of `n_embd` floats per stage boundary.

**Phase 6 — what the numbers ask for.** Chain forwarding. Peer weight
transfer. AVX2 kernels, then Linux workers. A `cuda` feature for `kvad-gpu`.
Expert parallelism. Tensor parallelism over RDMA. And when continuous
batching lands, micro-batches that keep every stage busy — which is the point
at which a cluster stops being only about memory.

## Open questions

- Does the Thunderbolt Bridge add enough software latency on the middle
  machine of a chain to matter? Phase 0, with three machines if there are
  three.
- ~~Is a split model really bit-identical?~~ Yes, in one process: Phase 1's
  test compares bits and passes for GPT-2, the Llama family and DeepSeek V2
  and V3 at every precision. Across two machines it is the same question
  again with a different answer possible — two CPUs, and kernels chosen by
  what each has — and Phase 2 asks it.
- Should the coordinator hold layers at all when it is also running
  `kvad-serve`, a database and a browser? Probably yes, fewer of them — the
  planner's headroom figure is where that is expressed.
- A worker with a GPU and a slow CPU (the mining rig) can hold no `ends` and
  should never be a coordinator. Does the planner need to know, or is that
  the operator's job? Start with the operator's.
- Does the Thunderbolt Bridge actually run spanning tree? `bridge0` reports
  `proto stp`, but with a zero root id, zero hello time and zero forward
  delay — which reads as configured and not running. If it is not, the full
  triangle suggested above is a broadcast storm rather than a redundant
  ring. Cheap to settle the day there are three machines and three cables;
  until then, build chains.
