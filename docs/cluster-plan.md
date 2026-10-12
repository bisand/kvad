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
  *Measured: 1.2 to 1.4 times the faster machine alone, on a pair as
  lopsided as an M5 Pro and an M2 Air over USB 2. See Phase 4.*
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
file; `weights::fetch_for` is the same with a filter, and is what a worker
calls. `model.safetensors.index.json` maps tensor
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
  about incoming connections the first time a worker listens — and until
  it is answered the worker looks, from the other machine, like one that
  takes a connection and never speaks. The allowance is for the program at
  the path it was run from: a new copy somewhere else is asked about again.
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
  challenge in both directions; the key never crosses the wire. *As built:*
  `cluster.key`, readable by its owner only; a coordinator reads the same
  file, or `KVAD_CLUSTER_KEY`, and never takes the key as an argument. A
  nonce from each end and a label for each role go under the HMAC, so a
  proof is good for one connection and one direction. A worker checks the
  coordinator's before it shows its own.
- `Load` accepts Hub repo ids and the worker's own model names, never a path.
- v1 sends activations in the clear and says so. They are not the prompt, but
  they are not nothing. TLS is a later option; `rustls` is already in the
  tree by way of `hf-hub`.

### Failure

A socket that dies mid-token fails the generation with an error naming the
node; the scheduler reports it as it reports any failed load. A pass that
fails half way down the pipeline leaves the stages before the failure ahead
of the ones after it; the session notes that and rewinds every stage to
where the conversation really is before it runs anything else, so a stage
that refused one frame costs that pass and not the conversation. The cluster
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

**Phase 2 is half done**, the half that is a worker: `worker.rs`,
`RemoteStage` (`remote.rs`), the handshake and the key (`key.rs`),
`kvad worker`, `--workers` and `--split` on `run` and `chat`. On loopback a
model with its layers on one worker, and on two with none kept by the
coordinator, answers to the bit as the model whole at f32, q8 and q4
(`tests/worker.rs`), and the same file holds the failures: the wrong key at
either end, another build, a worker that goes away in the middle of a
reply, a pass that fails half way down the pipeline, a load by path, and a
`--split` with a gap, an overlap or the wrong number of runs.

What a hop costs with no cable in it, from `examples/split.rs` with a worker
on `127.0.0.1`, seven rounds, the model whole against half its layers in
another process: decode at 0.94 and 0.96 of whole on SmolLM2-135M (q8 and
f32), 0.99 on Qwen2.5-1.5B at q8, and 1.00 with every layer on the worker.
That is a third of a millisecond a token, visible at 6 ms a token and gone
at 20. Prefill is 1.00 everywhere.

**And the second half**: the machine id, `_kvad._tcp`, the path probe, and
the commit in `Hello`.

- *A machine is an id.* `kvad worker` makes sixteen hex digits once, keeps
  them in `machine.id`, and says them in every greeting and every
  advertisement. It turned out to matter sooner than expected: between two
  runs an hour apart the Air's own name for itself went from `Mac` to
  `Andres-MacBook-Air`, because it had joined a network in between.
- *The build is the commit.* `build.rs` asks git, and `Hello` carries
  `0.18.2+8f21d2d3a6`, with `-dirty` after it for a tree with changes. A
  coordinator refuses a worker of another; two dirty builds of one commit
  compare equal, which is a development rig's own risk.
- *Discovery* (`discover.rs`). A worker registers `_kvad._tcp` through the
  system's `dns_sd` calls, so the advertisement ends with the process
  however the process ends; a coordinator reads what `dns-sd` prints. macOS
  only so far. `--workers` takes a name or the start of an id, and
  `kvad cluster find` lists the workers there are.
- *The path probe* (`path.rs`). Every address a worker was found on is
  greeted at once and then measured one after another, 32 KB and 2 MB
  there and back, and the quickest for 32 KB is used. A worker named by its
  address is measured too. Either way the link is identified and printed.
- *A worker greets everybody and holds one model.* It answered one
  connection at a time, which made five addresses five waits and let one
  address that swallowed its bytes hold up the rest; it now takes each
  connection on a thread, and what it refuses a second coordinator is a
  load, at once, by saying it is busy.

Across the rig on 2026-10-12, the worker named and not addressed
(`--workers andres-macbook-air`, and again by `689fdb39`):

```text
Andres-MacBook-Air is at 5 address(es):
  fe80::ca1:57c7:6946:68a8%en13: USB NCM, USB 2.0 High Speed: 480 Mb/s on en13; 2.58 ms for 32 KB there and back, 106.0 ms for 2 MB
  fe80::144a:8dcc:c1cf:a629%en7: USB 10/100/1000 LAN on en7; 7.20 ms for 32 KB there and back, 341.9 ms for 2 MB
  fe80::144a:8dcc:c1cf:a629%en0: Wi-Fi on en0; 14.54 ms for 32 KB there and back, 336.2 ms for 2 MB
  169.254.16.117: USB NCM, USB 2.0 High Speed: 480 Mb/s on en13; 2.49 ms for 32 KB there and back, 105.3 ms for 2 MB
  10.10.9.61: USB 10/100/1000 LAN on en7; 20.40 ms for 32 KB there and back, 341.4 ms for 2 MB
  using 169.254.16.117
```

The cable is found, measured, named for the charging cable it is, and
chosen over Wi-Fi — which is every piece of *Addressing and discovery*
doing what it was for. Two things in it are worth a second look. The two
addresses on the cable differ by a tenth of a millisecond and took turns
winning, so which is "using" is a coin toss between equals. And the routes
by this machine's wired LAN end at the Air's Wi-Fi, so their 340 ms for
2 MB is the Air's radio and not the wire's.

Not done: discovery on Linux, and an account of which of a worker's
addresses are worth trying at all — a machine with two interfaces on one
LAN is advertised at every pairing of them.

The run across the rig's cable was tried on 2026-10-12 and at first did not
get as far as a hidden state, for a reason worth knowing. The Air's
firewall is on. It had been told to allow the binary that `kvad cluster
ping` was measured with, by the path that one was copied to, and the worker
was copied to another. macOS then *completes the TCP connection* and holds
every byte of it, with a question on the Air's screen that nobody at the
coordinator can see: `nc` reports the port open, the coordinator's greeting
is sent, the worker sits in `accept` and logs nothing. The coordinator
waited out its five-minute patience and said `Resource temporarily
unavailable`. It now waits ten seconds for a greeting and says what this
looks like and where to look. Discovery will meet the same wall, and should
say the same thing.

With the firewall told, the same day: **a hidden state left a machine.**
SmolLM2-135M, layers 0 to 14 on the M5 Pro and 15 to 29 on the M2 Air, over
the charging cable (USB 2 NCM, identified as such), wrote the text the model
whole writes. `examples/split.rs` then held every logit of a 256-token
prefill and 48 decoded tokens to the whole model's for four splits at f32
and at q8, and all eight agree to the bit — across two machines, two chips
and two thread counts, which is the answer to whether a split model is the
same model when the halves are not on the same processor. Both have `i8mm`;
a pair where one does not is still unasked.

And the first number that does not fit. Decode, milliseconds a token, five
rounds, medians, against the model whole on each machine (8.3 on the M5
Pro and 19.2 on the Air at f32; 6.5 and 6.6 at q8) and a round trip of
0.8 ms for a frame this size:

| layers here / on the Air | f32 measured | Σ + round trip | q8 measured | Σ + round trip |
|---|---|---|---|---|
| 25 / 5 | 15.5 | 10.9 | 11.8 | 7.3 |
| 15 / 15 | 15.9 | 14.6 | 9.3 | 7.4 |
| 5 / 25 | 21.2 | 18.2 | 9.7 | 7.4 |
| 0 / 30 | 22.4 | 20.0 | 9.3 | 7.4 |

Every row is slower than `Σ stages + hops × RTT` says, by 1.3 to 4.6 ms, and
by most where the Air has least to do: five layers there cost 4.5 ms more
than they should at either precision, and at q8 giving the Air *more*
layers made the token faster. So a hop on this rig costs something the
ping does not see, and it grows when a machine sits idle between frames.
Two guesses, neither tested: the Air's cores and its thread pool going to
sleep between one token's work and the next, or the USB link answering a
lone frame after ten idle milliseconds more slowly than it answers a sweep
of them back to back. Phase 4 owes an account of the difference, and this
is where it starts; `kvad cluster ping` wants a mode that leaves a gap
between frames before the second guess can be told from the first.

**The account, the same day.** It was the first guess, and not as guessed.

`examples/hops.rs` takes a token apart: what each stage took as the
pipeline saw it, and how much of a worker's round trip the worker says it
spent inside its own layers (a `Spend` frame; `PipelineSession::timing`).
SmolLM2-135M at q8 over the cable:

| layers on the Air | round trip | inside its layers | a layer there | the hop |
|---|---|---|---|---|
| 5 | 5.92 ms | 4.87 ms | 0.975 ms | 1.04 ms |
| 15 | 6.40 ms | 5.55 ms | 0.370 ms | 0.84 ms |
| 30 | 7.45 ms | 6.65 ms | 0.222 ms | 0.80 ms |

The hop is what the ping said it was. The time is in the Air's layers,
which cost four times as much each when there are five of them as when
there are thirty — the less of a token a machine does, the slower it does
it. Leaving the Air idle for a set time before each token, with all thirty
layers on it, shows what that is:

| idle before each token | thirty layers at q8, on the Air | on the M5 Pro |
|---|---|---|
| none | 6.4 ms | 4.6 ms |
| 20 ms | 16.4 ms | 6.6 ms |
| 100 ms | 23.0 ms | 11.3 ms |
| 400 ms | 23.5 ms | 11.4 ms |

A machine that has done nothing for a tenth of a second does its next
burst of work at something like a third of its speed, for the first
twenty-odd milliseconds of it: at f32, where the Air's thirty layers are
17 ms of work, they took 35.6 after the same idle, the same eighteen
milliseconds lost. What it is not, each by taking it away: not the thread
pool waking (one thread on the worker loses as much as eight), not the
process's priority (zsh nices a job put in the background, which cost a
tenth and was not the cause), and not its scheduling class (asking for
user-interactive changed nothing to three figures). It is the processor,
stepped down and slow to step up, and both machines do it.

This matters more than the cable did. A layer split is a set of machines
each idle while the others work, so *every stage of every cluster* starts
each token cold, and a 70B across four machines at 100 ms a token would
lose eighteen milliseconds a machine to it. `Σ stages + hops × RTT` is
right about machines that are awake.

So they are kept awake (`model::linger`). Between one token's rows and
the next a worker spins one thread, and so does a coordinator while
another machine has the rows; one thread measured as well as all of them.
It stops when the rows come, or after half a second for the token that
never does (`KVAD_LINGER_MS`; 0 turns it off). The same rig, decode in
milliseconds a token, off and on, beside the plan's sum from each machine's
own rate and a 0.8 ms round trip:

| here / on the Air | f32 off | f32 on | sum | q8 off | q8 on | sum |
|---|---|---|---|---|---|---|
| 25 / 5 | 15.3 | 9.1 | 9.1 | 11.3 | 6.7 | 6.5 |
| 15 / 15 | 14.2 | 13.1 | 12.6 | 9.5 | 7.1 | 6.8 |
| 5 / 25 | 19.2 | 16.8 | 16.1 | 8.5 | 7.7 | 7.0 |
| 0 / 30 | 20.8 | 18.9 | 18.2 | 8.6 | 7.5 | 7.6 |

Within 0.7 ms of the sum everywhere, where it was up to 6 ms over. The
price is a core at full power on every machine of the cluster for as long
as a reply lasts, which is what a model run whole costs on all of them.

Still open: whether a burst much longer than these warms itself up part
way through, which a large model's stages would be; what this does on a
machine that is not a Mac; and that `kvad cluster plan` still times a
layer warm and says nothing of a machine that has the linger turned off.

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

**Phase 3 is done.**

- *The fetch filter* (`weights::fetch_for`, `Shard::wants`). A checkpoint
  split across files has an index from every tensor's name to its file,
  and a name says which layer it is; so a worker fetches the files its own
  layers are in and no others. By file and not by tensor: a file with one
  wanted tensor comes whole. A checkpoint in one file is that file.
- *The shard in the cache's identity.* A shard's quantised weights go in a
  file of its own, `….q8-layers-15-30.nq`, whose header says which shard;
  it is opened for that shard and refused for any other, and the whole
  model's file is the one it always was, so nothing written before is
  stale. A shard no longer quantises on every load.
- *Memory.* Every machine's share — its layers' weights, their cache at
  the context asked for, and the ends on the coordinator — is held to what
  that machine says it has, for a split made by hand as for one made here.
  `--context` says how long a conversation to leave room for.
- *The planner* (`plan.rs`). `--workers` without `--split` now places the
  layers: each on the fastest machine with room, the ends with the tokens,
  and a machine left with nothing left out. `kvad cluster plan MODEL
  --workers …` prints the table and loads nothing. How fast a machine is
  comes from timing a layer's seven matrix products at the model's widths,
  there, on a `Bench` frame.

The tests are the two this phase named: a checkpoint filed as three files,
from which a worker's machine is given the one file its layers are in and
nothing else, and loads and answers to the bit (`tests/share.rs`); and a
shard load followed by a whole-model load of the same checkpoint on the
same disk, which quantises its own file and is refused the shard's.

On the rig, 2026-10-12, with the Air a worker:

```text
$ kvad cluster plan Qwen/Qwen3-14B --quant f32 --workers 689fdb39
59.1 GB of weights and 2.7 GB of cache for a conversation of 8192 tokens
error: does not fit: the model is 61.8 GB at this precision and context, and 2 machine(s)
have 45.1 GB for it between them, which leaves 13 of its 40 layers with nowhere to go

$ kvad cluster plan Qwen/Qwen3-14B --quant q8 --context 40960 --workers 689fdb39
MACHINE             LAYERS      WEIGHTS     CACHE        OF      LEFT  MS A LAYER
bisand-macbook-pro  0..40       16.6 GB   13.4 GB   38.7 GB    8.6 GB       2.744
DESKTOP-8GF7H0F     none         0.0 GB    0.0 GB    6.4 GB    6.4 GB       8.604
```

and SmolLM2-135M at q8 by hand, 15 layers each: the first run quantised and
kept a shard on each machine, the second mapped both, and every logit
agreed with the whole model's.

What it is not yet:

- *The timing is rough.* It orders machines; it does not predict them. On
  this pair it said 0.24 and 0.34 ms a layer for SmolLM2 at q8 where the
  model whole measures 0.22 on each, and at f32 it was out by a factor of
  two one way on the M5 Pro and three the other on the Air. The table says
  it is an estimate, and it is one.
- *Sizes are averages.* A layer is the model less its ends, divided by its
  layers. A model whose layers differ — a mixture with dense first layers
  — is placed as if they did not.
- *Nothing here could be placed across two machines for real.* Everything
  that fits the pair fits the larger machine alone, so the planner's
  answer on this rig is always "here", and its other answers are unit
  tests. That is Phase 4's rig, as this document said it would be.
- *The plan knows nothing of the 1.3 to 4.6 ms a hop really costs here.*
- The Air named itself a third way this session: `DESKTOP-8GF7H0F`.

**Phase 3 — load only your share.** The fetch filter, the shard in the
quantised cache's identity, per-node memory checks, `plan.rs` and
`kvad cluster plan`. Tests: a sharded tiny checkpoint where a worker's shard
provably opens only its own files; and a shard load followed by a whole-model
load of the same checkpoint, which must not map the shard's cache.

**Pipelined prefill is done**, ahead of the rest of Phase 4, because it
needs no machine this rig has not got. A prompt of more than one chunk,
through more than one stage of which one is elsewhere, is read with a
thread a stage: each takes chunks in order from the stage before and hands
them on, with room for one chunk between stages. A stage still sees the
chunks in order, so it is the same arithmetic — thirteen chunks through the
coordinator and two workers agree with the model whole to the bit, every
position — and what changes is that stage two need not have finished a
chunk for stage one to start the next. `KVAD_PIPELINE=0` turns it off.

Across the rig's cable, SmolLM2-135M, a prompt of 1024 tokens, tokens a
second, five rounds each and each pair run twice; the model whole on the
M5 Pro did 450 at f32 and 495 at q8 beside them:

| here / on the Air | f32 a chunk at a time | f32 pipelined | q8 a chunk at a time | q8 pipelined |
|---|---|---|---|---|
| 20 / 10 | 341 | 557 | 401 | 702 |
| 15 / 15 | 309 | 449 | 377 | 585 |
| 10 / 20 | 277 | 343 | 354 | 455 |

A quarter to three quarters faster, and at 20 / 10 the two machines read
a prompt 1.2 to 1.4 times as fast as the faster of them does alone: the
first number in this document in which a cluster is quicker than its best
machine, and not only larger. It is what the arithmetic says it should be.
A chunk at a time costs each chunk the sum of the stages; pipelined it
costs the slowest of them, so the split that is best is the one that makes
the stages equal — more layers on the M5 Pro, which is the faster — and
10 / 20, which gives the slower machine the longer stage, gains least.
Placing layers for the prompt and placing them for the reply are therefore
two different questions, and the planner asks only the second.

**It asks both now.** `--place reply` is the planner as it was and still
what it does unasked; `--place prompt` places for the prompt read quickest;
and `--place 2000:200` for that much reading to that much writing. A
plan's cost for a prompt is the slowest stage's share of a chunk — its
layers at what a layer costs there *for a chunk*, which is timed apart from
what it costs for one token because the two are different arithmetic on
the same weights, and its round trip for a chunk's worth of rows. From the
two fillings it already made, the planner moves layers between machines
for as long as moving some makes the mix quicker, several at once where
one alone would only buy a round trip.

On the rig, SmolLM2-135M, what `kvad cluster plan --place prompt` chose
and what a 1024-token prompt measured at around it, tokens a second:

| layers here / on the Air | q8 | f32 |
|---|---|---|
| 27 / 3 | 563 | 484 |
| 25 / 5 | 582 | 522 |
| 23 / 7 | 636 | 574 |
| 21 / 9 | 695 | **601** |
| 20 / 10 | **704** | 571 |
| 18 / 12 | 674 | 534 |
| the model whole, here | 495 | 450 |

The planner chose 20 or 21 here at q8 over three askings, which is the
best there is, and 22 to 24 at f32, where the best is 21: within 5% of it.

It did not start there. Its first answers were 23 to 25 at q8 and 25 at
f32, a tenth and more short of the best, and the reason was the last
finding again. The layer is timed on each machine when the plan is asked
for, and a worker asked out of the blue has been idle; timed cold it
looked three times slower than the machine that asked, where it is less
than twice. The timing now runs the machine for a tenth of a second
before it starts the clock.

What the plan's numbers are still not is right in themselves. It thinks a
layer reads a token in 0.023 ms here where the model whole says 0.067,
because seven matrix products are not a layer; the placing comes out
because the machines are wrong by about the same factor. For a reply they
are not — it has the M5 Pro 2.7 times the Air's speed where the truth is
1.15 — which costs nothing while the only question for a reply is which
machine is faster, and would the day it is how much.

**Phase 4 — two real machines.** A model that does not fit on either. Decode
and prefill against Phase 0's prediction of `Σ stages + hops × RTT`, and an
account of the difference. Pipelined prefill chunks, measured before and
after. The README chapter gets written here, from the numbers.

**Phase 5 has begun**, with the part that makes the rest reachable:
`[cluster]` and `[worker]` in `kvad.toml`, a worker inside `kvad-serve`, and
`GET /api/cluster` with `kvad cluster status` in front of it. A server with
workers named spreads a CPU load over them through the same `session::across`
that `--workers` uses, and a model the plan leaves wholly here is loaded
whole and not as a pipeline of one stage — which it was, briefly, and which
would have cost every model on such a server its cache file and its
several conversations for the sake of the one model that needed workers.

Tried with two servers on one machine, one a worker and one naming it with
a split by hand: the same reply as the model whole, and the worker's log in
the first server's log.

**The Cluster page** is built, and with it the way a running model's
stages get out of the engine: not through it. A pipeline keeps an account
of itself — each stage, the machine and the link it is on, what it has
taken to write a token and to read a chunk — on a board that
`session::live()` reads from any thread, there for as long as the model is
loaded. `GET /api/cluster` carries it, `kvad cluster status` prints it, and
the page draws a token's time as a bar a stage, in two colours for a stage
on another machine: its own layers, and what is left of the round trip.

A worker is asked for its own time every fifty tokens or two seconds, not
every token, so what it has said is behind what the coordinator has seen.
The hop is the difference of two numbers and they have to be from the same
moment: the board keeps the coordinator's counts as they stood when the
worker was asked, beside the worker's answer.

**A load, a machine at a time** (PR after #229). Every machine's share is
now loaded at once: this one's in one thread, each worker's `Load` in
another, and the pipeline is made when the last of them is ready. Before,
they went in turn, and a model across three machines waited for the sum of
three loads where it need only wait for the longest.

That is also what made the progress worth doing. A `Progress` frame was a
line of text; it is now a line or a download's bytes
(`{file, bytes, total, done}`), sent from the worker's download threads
through a second handle on the socket, at most ten a second for a file and
none while the count stands still. The coordinator hands every step to a
`Watcher` as `Fetch::Part { machine, layers, step }` — asked, said,
download, fetched, ready with the bytes held, failed — this machine's own
share among them. The server passes them down the load's SSE as
`"kind": "machine"` and the Models page keeps a row for each machine.

Tried on one machine, a coordinator and a worker server with an empty Hub
cache of its own: this machine's fifteen layers were ready while the worker
was still 8 MB into a 257 MB download, and both rows showed it. Not tried
across a cable; the Air was not reachable that day.

Not yet: GPU stages; and the
scheduler's memory, which
charges a spread model as if all of it were here and does not charge a
server for the layers it holds for somebody else.

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
  what each has — and Phase 2 asked it of an M5 Pro and an M2 over the
  rig's cable: yes, at f32 and q8, for the Llama family. A pair without the
  same kernels is still open.
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
