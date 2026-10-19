[← Back to the README](../README.md)

# Crate 5: `kvad-serve` — the server, and the web UI

```bash
cd web && npm ci && npm run build && cd ..
cargo build --release -p kvad-serve
kvad serve                         # loopback on 5823, no auth
kvad serve --bind 0.0.0.0:5823     # needs an auth mode, or --insecure
```

One binary. `rust-embed` bakes `web/dist` into it, so there is nothing to copy
beside it and nothing to serve it with. `cargo build` never runs npm — a Rust
build that silently downloads a JavaScript dependency tree is not a Rust build
— so `web/dist` is built by hand, and a binary built without it serves a page
saying exactly that.

`kvad serve` is a subcommand of the CLI that hands over to this binary. It has
to be a hand-over rather than a function call: `kvad-serve` depends on the
engine crate, so the engine crate cannot depend on it back, and Cargo would be
right to refuse the cycle. On Unix it `exec`s, so Ctrl-C reaches the server and
its exit status is yours.

Twelve pages: a dashboard, model management, chat, a playground, images,
videos, training, datasets, evals, benchmarks, monitoring and settings — plus
the API's own documentation at `/api`. `/v1/images/generations` is OpenAI's
images endpoint, with steps, guidance, seed and a negative prompt beside its
fields; every picture is kept, with the settings that made it. `/v1/videos` is
OpenAI's video endpoint, a job that answers at once and is watched until it is
done, for LTX-2.5's clips with sound, from a prompt or from a picture and a
prompt, fast from its distilled model, guided from its dev model, or by DFR,
the reference's production pipeline, which also makes 48 and 96 fps and, with
its spatial epilogue, sizes its second stage cannot hold whole, and decoded
by its diffusion decoder or, when asked, its convolutional one
(`docs/video-plan.md`). `/v1/audio/transcriptions` and
`/v1/audio/translations` are OpenAI's speech endpoints, for any Whisper, on
the GPU or the CPU; a form with the recording in, the transcript out in any
of OpenAI's five formats, at a CPU backend the same bytes
`kvad transcribe --local` writes, and nothing kept (`docs/speech-plan.md`).
`/v1/realtime?intent=transcription` is OpenAI's Realtime WebSocket for a
transcription session: audio up while somebody speaks, settled words and a
replaceable partial down, about half a second behind the voice on a small
Whisper at gpu-q8; the Transcribe page's microphone uses it.
Four authentication modes (`none`, `local`, `basic`,
`oidc`), roles, API keys. SQLite for everything the filesystem cannot answer.
`docs/ui-plan.md` is the plan it was built from, and each phase in it records
what that phase measured and which of its open questions closed.

## A reasoning model's working is not its answer

Qwen3 and its relatives open a reply with `<think>`, reason in the open, close
with `</think>`, and then answer. Handed to a client as one string, which is
how it arrives from the engine, the working reads as part of the answer — and
it is not: it contradicts itself, changes its mind, and is routinely longer
than what follows. Measured on one grounded question: 2,591 characters of
working against 284 of answer.

So the server tells them apart as they stream, and sends the working as
`reasoning_content` — the field DeepSeek's API established and clients
recognise — leaving `content` the answer. It cannot be a split at the end,
because a tag is several tokens (`<`, `think`, `>`) and a client that waited
for the whole reply to divide it would lose the streaming it came for. Text
that might still become a tag is held back; everything else goes out at once.

Two rules worth stating. A `<think>` after the model has already said
something is not a trace, it is a model writing about tags — the paragraph
above would parse as one — so the opening tag counts only before any other
text. And a trace the token budget cut off mid-thought is still a trace, and
arrives as one rather than as an answer, which for a small reasoning model is
the ordinary case: at 400 tokens, Qwen3-0.6B spent all of them thinking and
never reached an answer at all.

The web UI shows it collapsed above the reply, open while there is nothing
else to show. It is worth being able to read — it is where a model can be seen
catching itself following the wrong passage — but it is not kept: the message
saved to the conversation is the answer.

## Tool calls, and the model that cannot make one

An agentic client — OpenCode, or anything else built on the OpenAI
libraries — is not asking for prose. It sends a list of functions it is
willing to run and expects the model to answer with a call, runs it, and
sends the result back for the next turn. Without that, a coding agent
pointed at this server could talk about a file and never read one.

None of it is a protocol the engine invents. The tools go into the model's
**own chat template**, which writes them into a system block in the wording
that model was fine-tuned on — Qwen's is a `<tools>` list and an instruction
to answer inside `<tool_call>` tags — and the call comes back as text like
any other token. So the work is at both ends of the template: hand it the
schemas, and parse the tags out of the reply.

Parsing them out has the same shape as separating a reasoning trace, and for
the same reason: a tag is several tokens, so text that might still become one
is held back and everything else streams. What differs is the middle. A trace
is text to forward as it arrives; a call is JSON that has to be complete
before it is even known to *be* a call, so the block is buffered whole and
sent as one delta. Two rules follow from not wanting to lose anything: a
block whose JSON does not parse comes back out as content, tags and all, and
so does one the token budget cut off before it closed.

The rule that is easy to get wrong is when to look at all. `<tool_call>` in a
reply is a call only if tools were offered — otherwise it is a model writing
about the format, which the paragraph above would do. There is no position
rule that could tell them apart the way `<think>` has one, because a real
call legitimately follows text. So the parser runs only for a request that
offered tools, and for every other request the reply is untouched.

**Two refusals rather than two pretences.** A model whose template never
mentions `tools` renders exactly the same prompt whether or not any were
offered: the model is never told they exist and answers in prose, which a
client cannot tell apart from a model that considered the tools and declined.
That is a 400 naming the model, and `/v1/models` reports a `tools` flag per
model so a client can choose one before it loads anything — of what is on
this machine, the Qwens can and SmolLM2, the DeepSeeks and the GPT-2s cannot.
The other refusal is `tool_choice`: `auto` and `none` are honest, `required`
and a named function would need the sampler constrained to the tokens that
open a call, and that is not written.

It works on a model small enough to be surprising. Qwen2.5-0.5B-Instruct, at
q8 on the CPU, reads "what is the weather in Oslo?" with one function on
offer and answers `get_weather({"city":"Oslo"})` in 20 tokens, then takes the
result back and says it is 7 degrees and raining. Pointed at the same
endpoint, OpenCode runs its own read tool and answers out of the file. That
is not a coding agent worth using — at 0.5B nothing is — but it is the whole
loop, and the protocol above it does not care how big the model is.

One gap worth naming rather than papering over: the parser reads the
Hermes-style `<tool_call>` format that Qwen and most open models adopted, and
not Llama 3.1's bare object or DeepSeek's own markers. A model whose template
documents a different format will not be understood, and adding one is a case
in this parser rather than a redesign.

## One model at a time, said out loud

An engine holds one loaded model, and one engine works at a time. Every
awkward thing about this server follows from that, and the design is mostly a
series of decisions about where to be honest about it.

A **scheduler** owns the engine thread and everything queues behind it, so
queue depth is a number on the dashboard rather than a mutex nobody can see.
The queue has an end: `[server] queue` generations may wait (16 unless set),
and the request after that is answered 503 with a `Retry-After` rather than
held. `kvad queue` and `GET /api/queue` show what is waiting, for how long,
and — to an administrator — whose it is.

**Accounts take turns.** The next generation is the earliest request of
whichever account was served longest ago, so somebody who sends ten does not
get ten times the machine, and one request is not kept behind another
account's ten. One account may have `[server] queue_per_account` waiting (4
unless set); its next is a 429. Turns are counted in generations, not in
tokens. Nothing passes a load or an unload. An account can also be held to
`[server] generations_per_minute`, which is off unless set; over it, the 429
says how many seconds until the next request will be taken.

**A few replies run at once.** On one model, up to `[server] batch` replies
(4 unless set) are written together, each a token further every step, and one
that ends gives its place to the next in line. Whose turn it is to join is
whose turn it is to run; if that is a load, or a request for another model,
nobody joins and the batch runs down first. What this buys depends on where
the model is:

| Qwen2.5-1.5B, four clients at once | total tok/s | each reply, tok/s | first token, median |
|---|---|---|---|
| `cpu-q8`, one at a time | 44 | 46 | 4.0 s |
| `cpu-q8`, together | 93 | 24 | 0.4 s |
| `gpu-q8`, one at a time | 127 | 131 | 1.4 s |
| `gpu-q8`, together | 167 | 43 | 0.1 s |

The weights are read once a step for everybody, which is the throughput. On
the GPU that is worth less than it should be at a few replies and more at
many: a product with several rows is a different kernel from a product with
one, and costs three to seven times as much whether the rows are two or
sixteen. So one pass is used only from where it wins — two replies at bf16,
three at q8, six at q4 — and below that each reply takes its own step, which
costs nothing and still means nobody waits for somebody else's whole reply
before their own begins. With `batch` raised, sixteen replies on Qwen2.5-1.5B
at `gpu-q8` are 550 tok/s between them against 128 in turn, and on Qwen3-14B
95 against 17. A reply's `kvad.beside` says how many others it was
written beside. A request that asks for no cache — a benchmark, an evaluation
— runs alone.

A reply that joins has a prompt to read first, and a long one takes seconds.
It is read a pass at a time — 64 tokens on the CPU, 512 on the GPU, where the
backend would have cut it anyway, so the arithmetic is the same — and the
replies under way write a token between passes. With a 2,300-token prompt
arriving while a reply streams from Qwen2.5-1.5B, the longest the stream goes
quiet is 0.4 s rather than 16.7 at `cpu-q8`, and 0.3 s rather than 1.3 at
`gpu-q8`. The newcomer's own first token comes when it did, or a tenth of a
second later. One prompt is read at a time; a second newcomer waits for the
first to finish reading.

At `cpu-q8` on a CPU with the `i8mm` instructions, a reply written beside
others is not always word for word the reply written alone. The kernel for
several rows adds up in a different order, and rounding activations to eight
bits turns that into differences the size of quantisation itself — 0.3 to 0.5
in a logit, the same distance from the f32 model either way — which is enough
to tip a greedy choice between two near-equal tokens. The same goes for a
quantised model on the GPU once enough replies run together for one pass,
where 3 greedy replies in 20 differed. At f32 on the CPU, and on the GPU
below that number, they are the same reply.

The same request sent twice is the same reply at temperature 0, on every
backend. It was not always: the second time the prompt is in the cache and
only its last token is run, alone, where the first time that token was the
last row of a pass over the whole prompt, and one row is not the kernel
that several are. The two agreed to rounding — one part in ten thousand on
a GPU, which a bf16 model's logits turned into 0.6 and a q8 head on the CPU
into 0.09 — and a greedy reply parted from its twin at the first near tie,
12 tokens in at bf16 and 112 for a q8 model in two GPU stages. So a
prompt's last token is always read by itself, after the rest. That costs a
fresh prompt one token's time, and on a GPU a prompt that happened to be a
whole number of 32-token tiles reads like any other length instead of twice
as fast. `crates/gpu/examples/replay.rs` measures all of it.

**Each conversation keeps its cache.** A chat prefills only its newest
message because the turn before left its keys and values in the model, and
until several were kept that held for one conversation at a time: two people
taking turns each found the other's conversation there and re-read their own
from the start, every turn. Now the conversations that are not speaking are
set aside whole — a few pointers moved, nothing copied — and the one that
speaks is found by its tokens, as whichever shares the longest prefix with
the prompt. Nothing names a conversation; the transcript an OpenAI client
sends is the key. They fit in the context the model was already charged for
(`[server] context`), and when they do not, the one that spoke longest ago
goes. A reply's `kvad.conversations` and `kvad.held_tokens` say what is kept.

**Who used it is written down.** Every chat completion's row in the
`requests` table carries the account that asked, what the prompt came to and
how much of it was cached, beside what was generated and how long the first
token took. `kvad metrics usage` and `GET /api/metrics/usage` add that up by
account over the last days: your own figures, or everybody's for an
administrator. It answers who is using the machine and who waited for it,
and it only knows about replies made since it was added.
**Comparisons are sequences**: the Playground's "side by side" and the
Benchmarks page both load each variant in turn, and both say so on the page
rather than implying a race. **Training is not queued behind chat, or chat
behind training** — that one was measured rather than assumed, and the
measurement is below.

And **a restart can put the model back**, when asked to. The engine holds
nothing until something asks, which for a service means the first request
after a reboot pays the load — tens of seconds, ninety for DeepSeek-V2-Lite.
With `[server] autoload = true` the server loads whatever was active as soon
as it is listening, in the background and as a job on the same queue: the
address is answering while it happens, the Models page shows it arriving, and
a request that turns up meanwhile waits for that load rather than starting a
second one. It is off by default, because the cost is a model in memory a
minute after boot whether or not anybody turns up, and a load queued ahead of
everything else before anyone asked for it.

## A model too large for this machine, on several

A language model can be spread over other machines, its layers divided
between them; [`cluster-plan.md`](cluster-plan.md) is the
whole of how and why. The server's part is two sections of `kvad.toml`.

```toml
[cluster]
workers = ["studio", "689fdb39"]   # by name, id or address

[worker]
enabled = true                     # and this server holds layers for others
```

With `[cluster] workers` set, a load asks those machines how fast they are
and how much they can hold, and places the layers: as many as fit on the
fastest, then the next. **A model that fits here stays here**, whole, loaded
exactly as it is with no workers named — same cache file, same several
conversations, same replies in a batch. The workers are for the model that
does not fit. `place = "prompt"` or `"2000:200"` places for the prompt read
quickest instead of the reply written quickest, which shares the work out
where the default would not.

With `[worker] enabled`, the server answers coordinators on port 7420 beside
serving its own models, and says so on the local network. It answers only a
machine with the same `cluster.key`, which is made in the data directory the
first time a worker starts there and is copied by hand to the others. Both
ends have to be the same build, to the commit.

`GET /api/cluster`, `kvad cluster status` and the Cluster page say what the
two sections came to, which workers the network says are there, and for
every model spread out now, which machine holds which layers and where a
token's time goes: each stage's layers, and for a stage on another machine
the round trip to it.

Every machine loads its own layers at once, so a model across three is
ready when the slowest of them is. The load's stream says whose each step
is: beside the lines and the bar it always had, a `progress` event of
`"kind": "machine"` carries a machine's name, its layers, and a `state` of
`asked`, `loading`, `ready` or `failed` — with a line of the machine's own,
or the bytes of a file it is downloading, or the memory its layers took. The
Models page draws a row for each while the model loads.

Memory is one budget on each machine. A model across machines is charged
here for what of it is here: this machine's layers, the embedding and the
head, and the cache of those layers. The layers a server holds for another
machine's model come out of the same budget as its own models, are refused
when they do not fit, and are given back when the coordinator lets go; the
Models page says how much of what is taken is theirs. A worker tells a
coordinator what it has left and not what the machine has, and a
coordinator places against what its own models have left. So a model too
large for what is left here is not refused out of hand when `[cluster]`
names workers: it goes to the placing, which puts on them what does not
fit.

At a GPU backend the layers go onto each machine's GPU, a stage of them at
a time: rows of the residual stream go up to the device and come back once
a stage, here and on every worker. With SmolLM2-135M at `gpu-q8`, fifteen
layers on each of two servers on one machine, the reply was the whole
model's to the letter at 417 to 444 tokens a second against 522 to 600
whole: on one GPU a split is only its cost, which is the round trips and a
head on the CPU. With the second fifteen on an M2 Air's GPU over a USB 2
cable it was the same text at 161 to 225 tokens a second: a model this
small gains nothing from a second machine, and the point is the one that
does not fit on the first.

What this does not do yet, said plainly:

- **On a GPU, the Llama family only**, and every machine alike: a model
  loaded at `gpu-q8` has its layers on each machine's GPU, so every worker
  has to be a `kvad-serve` with the GPU backend. The embedding and the head
  stay on this machine's CPU. `kvad worker` holds CPU layers only.
- **A GPU stage's cache file is its own**, named for its layers: another
  split of the same model quantises again.
- **A model in stages keeps one conversation.** A second one costs the
  first its cache, and replies are not written in a batch.
- **A worker that goes away fails the reply** and the model stays loaded
  and broken until it is unloaded. Nothing reconnects.
- **Each load asks the network again**, which is about ten seconds of
  finding and measuring before anything is read.
- What crosses between machines is not encrypted.

## The default backend is the one that is faster here

Which backend a load uses when nobody picks one used to be `cpu-q8`, from
back when the GPU ran the Llama family and nothing else. It now runs GPT-2
and DeepSeek too, so the question was worth re-asking — and worth asking the
benchmark page rather than the intuition. Medians of interleaved rounds, 64
tokens, q8 on both sides, on an M5 Pro:

| model | architecture | cpu-q8 | gpu-q8 | |
|---|---|---|---|---|
| GPT-2 medium | gpt2 | 127.9 tok/s | 292.5 tok/s | 2.29x |
| Qwen2.5-0.5B | llama | 117.0 tok/s | 209.6 tok/s | 1.79x |
| DeepSeek-V2-Lite | deepseek_v2 | 26.7 tok/s | 39.1 tok/s | 1.46x |

The last row is measured differently from the two above it, and the reason is
worth more than the row. Interleaved against each other the same way, it read
13.6 against 37.7 — a 2.77x that this README printed and a release note
published before anybody checked it. DeepSeek-V2-Lite is 17 GB on either
backend, and this machine has 48: while the GPU variant holds its copy in
Metal buffers, the CPU variant's memory-mapped weights are evicted, and the
next CPU round faults them back off the disk as it decodes. The CPU was not
being measured. It was being paged.

Measured one at a time it is 26.7 against 39.1, which agrees with the 23.4
[measured elsewhere in this README](engine.md#the-one-that-needed-its-own-file) for a model that is
loaded and left alone.

**So the protocol that exists to defeat drift can cause it.** Interleaving is
right when the variants are small enough to coexist, and wrong the moment two
of them together do not fit in memory — at which point each round is measuring
the other variant's footprint. The benchmark page does not know that, and a
run whose variants are each a third of RAM should be run as two runs.

So the default prefers the GPU — with two qualifications that the numbers
themselves put there. It asks the GPU backend whether it has an
implementation of *this* architecture, because a default that fails to load
is worse than one that is slower, and that answer is the same list the loader
dispatches on rather than a second copy of it. And it stays at q8 on both
sides, so the comparison is the same arithmetic in two places: bf16 on the
GPU is twice the memory and, on this machine, no faster than the CPU's q8.

**The model nobody has downloaded yet.** Asking which architecture a model is
only works for a model whose config is on this disk, and the first load of
anything is the case where it is not. The two ways of knowing nothing were
sharing a spelling — `None`, from an `Option<Arch>`, meant both "the picker is
asking what this build likes in general" and "somebody named a repo this
machine has never seen" — and the second was quietly getting the first's
answer. So a model about to be downloaded for the first time was being offered
the GPU on no evidence at all, which for a `gemma` is a default that cannot
load. (So it was for a `whisper`, which now loads, and on the CPU only.)

Separating them made the honest answer `cpu-q8`, and the honest answer was
annoying: every model's first load on the slow backend, then the right one
ever after. The Hub knows, though, and it will say so for the price of a
metadata request — `config.model_type`, a few hundred bytes, the same field
the search page already reads. So `hub::remote_arch` asks. Measured: about
0.9 s, on the way to a download of several gigabytes, and only when nothing
local can answer and nobody has stored a choice. A repo id that is really a
path is refused before the request rather than after it (8 ms), and every way
of not knowing — offline, no such repo, an architecture this build has no
implementation of — comes back the same and still means the CPU.

One number in that table nearly became a paragraph of its own. The first run
had a GPU round with a time to first token of **6.9 seconds** against a 22 ms
median, which read as Metal compiling its pipelines once per process — a real
effect, plausibly placed, and worth designing around: with the server loading
a model at startup, that cost would land on whoever sent the first request.

It did not reproduce. The same benchmark on an idle machine put the GPU's
worst round at 24 ms and returned both decode medians within 1% of the first
run's. What the first run actually showed was a slow round in *both* variants
— the CPU's worst was 145 ms against its own 35 ms median — and a cost
belonging to Metal does not appear on a CPU sample. Something else on the
machine was busy, and one sample from each variant caught it. The explanation
was invented to fit a single outlier, and the outlier was noise.

So the medians stand, the default stands, and the warm-up that was going to
absorb those 6.9 seconds is not written — this time on a measurement rather
than on a failure to reproduce one. Fifteen loads per backend, three
generations after each, alternating which backend led: the first generation
of the whole process and the first generation after the fifteenth load are
the same to within noise (22.2 ms against 22.8 ms on the GPU), and a
once-per-process cost is the only thing a warm-up at startup could have
removed. What is left is ~14 ms on the first generation after any load,
which is larger on the CPU than on the GPU — the KV cache filling, not Metal
compiling. Three of this README's numbers were
once wrong in exactly this way, which is why the page that produced the table
interleaves its rounds and prints the range beside the median: the range is
what said this was worth re-running.

A stored setting still wins over all of this, because somebody choosing a
backend has a reason — and it wins early enough that the Hub is never asked.

## What it measures, and what it refuses to

Three numbers on this page have been wrong before, which is why the server has
a benchmark in it rather than a shell script:

- A **round visits every variant once**, and a run is several rounds. Five of A
  then five of B blames the model for whatever changed about the machine in
  between.
- **Median and range**, from samples that are all kept. Two ranges that overlap
  are two numbers that have not been told apart, and a page that showed one
  average each would hide that.
- **Every timed generation drops the KV cache first**, or the second round
  reports a time to first token that no first run would ever see.
- A benchmark **will not start** while a training run, an eval or another
  benchmark is going, and the refusal names what is in the way.

On an 18-core M-series machine, decoding SmolLM2-135M:

| | median | range |
|---|---|---|
| idle | 70–84 tok/s | 53–92 |
| during training, all cores | 32–37 | 25–41 |
| during training, capped to 8 | 47 | 42–54 |

So chat during a training run runs at roughly half speed, not zero. The banner
says "slower", because that is what was measured.

And with nothing else running, q4 against q8 at 64 tokens, three rounds each:

| | decode median | range | time to first token |
|---|---|---|---|
| q4 | 149.5 tok/s | 149.3–150.2 | 41 ms |
| q8 | 114.6 tok/s | 97.2–118.3 | 30 ms |

q4 decoded faster and reached its first token *slower*, and the same split
showed in scoring, where q4 took twice as long as q8 for the same 519 tokens.
Decoding is bound by memory traffic, where fewer bits win — though on a model
this small that was as much luck as principle. On anything larger q4 decoded
*slower* until [its kernel was written](engine.md#the-nibble-kernel-the-compiler-would-not-write),
and this page had simply found the one model whose matrices were small enough
to hide it. Prefill is bound by
arithmetic — and q8 had a batched `SMMLA` kernel there while q4 did not, so it
fell back to a row at a time. Both are integer; neither dequantises. That was a
missing kernel rather than a price quantisation charges, and
[it has since been written](engine.md#batched-prefill-and-i8mm): q4 prefill roughly
doubled and the 41-against-30 gap closed to nothing.

This is the first thing the web UI paid for. Nobody would have run that
comparison from a shell, because it needs two model loads and ten interleaved
generations to say anything, and the number it turned up was hiding in plain
sight behind a decode rate that looked fine.

## The playground is where the logits stop being abstract

Generation computes a distribution over the whole vocabulary at every step and
throws away everything except the token it drew. The playground keeps it: click
any token and see what else was in the running, with the model's own
probabilities — a plain softmax over every logit, not the sampler's reweighting
of it, so the numbers do not move when you drag the temperature slider. What
the sampler did is a separate mark on each row saying whether top-k and top-p
left that token in play at all. At a temperature of 0.8 with top-p 0.95, a
confident step often leaves exactly one, which is worth seeing: there was no
choice being made.

Beside it, the tokeniser inspector. `"The cat sat on the mat, and it was 2026."`
is sixteen tokens, and `2026` is five of them — a space, then each digit alone.
A token is not a word and not a character, and this is where that stops being a
thing you have read and starts being a thing you have seen.

## Perplexity, and the thing the engine was missing

`forward_batch` returns logits for the **last** position only, because that is
all generation ever wants, and every intermediate row is computed and dropped.
Scoring text wants all of them. `forward_batch_all` runs the output head over
the whole batch as one matmul instead: about 430 tokens a second, against 114
for decoding — the same arithmetic, a quarter of the memory traffic. It is
checked against `nervus`'s own logits at every position, which is the same
second-implementation argument the checkpoint test rests on.

## The command line, as a client of the server

With the service running, `kvad` sends its commands to it. Without that the
CLI would be a second installation beside the service that happens to share a
model directory: `kvad chat` would load a second copy of weights the server
already holds, which on a 17 GB model is the difference between answering and
swapping. So `ls`, `pull`, `run`, `chat`, `train` and the rest go to the
server when one answers, and run in the process when none does, and the
first line of output says which:

```
$ kvad chat
kvad: http://127.0.0.1:5823 (the kvad service; --local to run in this process instead)
model: Qwen/Qwen3-14B@gpu-q8
```

The order is `--remote URL` or `--local`, then `KVAD_URL`, then `url` under
`[client]` in `kvad.toml`, then this machine's service at the address its
unit file gives it. The first three are things somebody said, so a server
they name that does not answer is an error rather than a reason to quietly do
the work here instead. The last is a guess, and it costs a connection to
loopback, which is refused in well under a millisecond when nothing is there.

Everything the API does has a verb, including what has no local meaning:
`kvad ps`, `load` and `unload` for what is in memory, `conversations`,
`jobs`, `datasets`, `evals`, `bench`, `metrics`, and `auth`, `users`,
`sessions` and `keys` for accounts. `kvad auth login` signs in and keeps an
API key for that server in `credentials.json`, readable by you only; with
`auth.mode = "none"`, the default, there is nothing to sign in to. `--json`
prints what the server sent, and a streaming command prints one event a line,
so they compose with `jq`. `kvad api` with no arguments lists every route, and
with a path calls one.

`kvad service status` asks two things and says when they disagree: launchd
or systemd, about whether the job is running, and `/api/health`, about
whether anything answers. A job the service manager calls running that does
not answer on its port is the failure people actually hit, and a status
command that only asked the service manager would call it fine.

## From anywhere, through the tunnel

A `[remote]` section in `kvad.toml` lets this server be reached from outside
the machine and its network without opening a port: an end-to-end encrypted
tunnel, direct when the two machines can reach each other and through a relay
when they cannot (`docs/relay-plan.md`, Phase 1).

```toml
[auth]
mode = "local"                       # a tunnel needs accounts; see below

[remote]
enabled = true
relays = ["https://relay.kvad.eu"]   # tools/kvad-relay, run somewhere public
```

A device is paired once. On the server's machine, signed in:

```bash
kvad remote pair
```

It shows a QR code and a link, good once, for ten minutes. An administrator
can do the same from the web UI: *Pair a device* under Devices in Settings,
with *allow admin* for a device that should act as one. The page shows the
code until it is used or expires, and closes the slot on Cancel or when it
is left. On the device:

```bash
kvad remote join 'kvad://9ce63fe0…?k=…&relay=https://relay.kvad.eu&pair=…'
```

The device is now in `kvad devices`, with a key of its own kept in its
`credentials.json`, and every command takes the server's name:

```bash
kvad chat --remote kvad://9ce63fe0…
```

`kvad remote status --remote kvad://…` says which path the connection is on:
`lan`, `direct` or `relay`. `kvad devices rm ID` unpairs one: its key is
revoked and its connection closed at once. `[remote] allow` lists ids let in
without pairing, from `kvad remote id` on each.

- **Sign-in still applies.** A tunnelled request needs a key or a session as
  any other does, so `[remote]` is refused with `auth.mode = "none"`, and
  `--insecure` does not change that: nothing stands in front of a tunnel.
- **Only paired devices get in.** Every other key is refused at the
  handshake, before any HTTP is read. While a pairing is open, an unknown
  key gets as far as the pairing exchange, one request and one answer, and
  no further.
- **A device's key is that device's.** It works only through the tunnel,
  from the device it was made for, and acts as `user` unless the device was
  paired with `kvad remote pair --admin`, whatever its account is. A device
  cannot pair other devices unless it has `admin`. A connection without the knock gets no
  answer at all, or one TLS alert sent before the server's key would be.
- **Post-quantum.** X25519MLKEM768 is the only key exchange, to the device and
  to the relay. A relay that cannot do it is refused.
- **No router ports.** Kvad never asks the router for a mapping. A direct
  path across the internet forms by hole punching, or through a port forwarded
  by hand to `[remote] port`. `direct = "off"` opens no UDP socket at all and
  goes through the relay alone.
- **Keys in owner-only files.** The server's are `remote.key` and
  `remote.knock` in the data directory; a device's is `device.key` beside
  `credentials.json`. Each is made on first use, and refused if anyone else
  can read it.
- **The CLI opens no port.** Each HTTP connection is a stream through the
  tunnel, handed to the CLI as one end of a Unix socket pair, so nothing else
  on the device can use the connection.

### Any HTTP client: `kvad tunnel`

A program that only speaks HTTP to a port, such as OpenCode, curl or the
openai SDK, reaches the server through `kvad tunnel` on the device:

```bash
kvad tunnel --remote kvad://9ce63fe0…      # 127.0.0.1:5824 unless --port N
```

It listens on `127.0.0.1` only and forwards each TCP connection as one stream
through the tunnel, byte for byte, so the program is pointed at
`http://127.0.0.1:5824/v1` and is otherwise unchanged. It says which path the
tunnel is on (`lan`, `direct` or `relay`) when it starts and whenever that
changes, connects again when the tunnel drops, and renews the relay voucher as
the CLI does. With one server paired, `--remote` can be left out.

The port is the trade-off. The CLI's own connections are socket pairs nothing
else can reach; a port can be reached by every program on the device and by
every other user of it, and through it they all reach the server as this
device. The server still wants a key with every request, so by default each
program sends one of its own, made with `kvad keys add NAME`, which can be
revoked without touching the others. A request without one is answered 401.

`kvad tunnel --with-key` adds the device's own key to every request instead,
replacing any the program sent, so nothing needs a key at all. It also means
that anything able to reach the port acts as this device, with its rights,
for as long as the command runs. It is never on by default, and it prints a
warning when it starts. Use it on a machine that is yours alone. With it on:

- A request a browser sent from a web page is refused before it reaches the
  server: one with an `Origin` other than the port's own, a `Sec-Fetch-Site`
  other than `same-origin` or `none`, or a `Host` that is not the port (what a
  page sends after pointing its own name at 127.0.0.1). A page can make
  requests to loopback, and without this it would make them as the device.
- Each request gets a connection of its own (`Connection: close`, except for
  a WebSocket's upgrade), so every request that reaches the server is one
  whose head the forwarder has read and given the key.

The relay sees which keys talk, when, and how much, but not what. Run your own
(`tools/kvad-relay/deploy/README.md`). It serves the servers on its own allow
list, and the devices they vouch for: a paired device carries a voucher its
server signed, renewed on its own through the tunnel, so the relay needs no
list of devices and nothing changes when one moves to another network.

## Testing

The API tests train a two-layer GPT on "the cat sat on the mat" *inside the
test*, save it, and then run a real prompt suite, a real benchmark and a real
perplexity job against it through the real scheduler. A second model is
trained on chat turns, `u:the cat?|a:on the mat.|`, and saved with a chat
template and `|` as its end token. The prompt suites and the chat endpoint are
pointed at instruct models, and a model with no template once let every eval
answer come back empty on a green suite. Each model trains in under a second,
with no network and no gigabytes.

Authentication got the mutation treatment: each of fifteen checks was deleted
in turn and the suite rerun. Two escaped the first pass — the `Admin` extractor
and the CSRF gate had been tested through the functions underneath them and not
through the extractors the routes are actually written against. Both have tests
now that fail when their check is removed.

And `/api/openapi.json` is generated from a table that a test compares against
the router, by reading the source of the files that register routes. A route
added and not documented fails the build. The first version of that test failed
on itself: it scanned its own source and found the string it searches *with*.

The same trick runs one step further out, to keep "the CLI can do everything
the API can" true after the day it was written. `kvad::client::COMMANDS` maps
every route to the command that reaches it, and a route with no command has to
say why not in `NOT_COMMANDS` — the OIDC callback is not something to type. The
server's tests fail when a route is in neither table, and the CLI's tests read
its own source for every path it requests and fail when that set and the table
differ. A route added to the server therefore fails three tests in turn, until
it is described, until it has a command, and until the command actually sends
it.
