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
(`docs/video-plan.md`). Four authentication modes (`none`, `local`, `basic`,
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

The engine holds one loaded model and runs one generation at a time. Every
awkward thing about this server follows from that, and the design is mostly a
series of decisions about where to be honest about it.

A **scheduler** owns the engine thread and everything queues behind it, so
queue depth is a number on the dashboard rather than a mutex nobody can see.
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
the GPU on no evidence at all, which for a `whisper` or a `gemma` is a default
that cannot load.

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
