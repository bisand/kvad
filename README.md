# Kvad

*Kvad* — Old Norse for a composed, recited poem: what a skald performs from
memory, one line at a time.

An LLM engine in Rust, written from the arithmetic up, with two jobs.

**Explain how this works.** Every matrix multiply, every derivative, every
attention head and every rotation is code in this repo, with the reasoning for
each constant next to it. Read the crates in order and you will have built a
transformer rather than configured one.

**Be worth running.** The target is vLLM and SGLang. Not as a gesture — as the
bar this has to clear before it is a real alternative rather than a nice
explanation.

Those sound opposed and mostly are not. Almost everything that makes inference
fast is arithmetic you can read: block-wise quantisation, an integer dot
product, a tiled GEMM, a cache you do not recompute. The unreadable part of a
fast engine tends to be the last stretch — hand-tuned CUDA, a dozen code paths
per operator — and the stretch before it is where the lessons are. Where
legibility and speed genuinely conflict, this repo says which it chose and
measures what it cost.

Five crates, meant to be read in order:

| Crate | What it is | Dependencies |
|---|---|---|
| [`nanograd`](crates/nanograd) | A neural network and backpropagation, from scratch. Trains on MNIST. | **none** |
| [`kvad`](crates/llm) | Transformer inference from scratch. Four architectures, real HuggingFace weights. | hub client, tokenizer, safetensors |
| [`kvad-gpu`](crates/gpu) | The same Llama forward pass on the GPU, in candle. | candle (Metal/CUDA) |
| [`kvad-tui`](crates/tui) | Terminal app: browse, download, activate, chat. | ratatui |
| [`kvad-serve`](crates/serve) | HTTP server and web UI: manage, train, score, benchmark, watch. | axum, rusqlite, Svelte |

The first two use no ML framework at all. The third is the same model handed to
one, so the two can be compared — and so the hand-written version has something
honest to be measured against. The last two are applications: no ML in either,
and the place to see what an engine has to expose before anything can be built
on it.

### Where this actually stands

Kvad runs one sequence at a time. On an M5 Pro, Qwen2.5-0.5B decodes at
**191 tok/s** on Metal at q8 and **112 tok/s** on the hand-written CPU engine —
which is also, for now, what this machine's GPU does in bf16. It has weight and
activation quantisation, an i8mm integer kernel, a hand-written 4-bit decode
kernel, a tiled f32 GEMM, batched prefill, prefix caching across chat turns,
and a memory-mapped cache of pre-quantised weights.

It runs four architectures, and the fourth is the one that says most about
where inference has gone. **DeepSeek-V2-Lite** — 15.7 billion parameters, 2.4
billion of them used on any given token — loads in 92 seconds, occupies 17.7 GB
at q8, and decodes at **23.4 tok/s** on the CPU. It needed multi-head latent
attention and a 64-expert mixture, neither of which the GPT-2-to-Llama skeleton
had any room for, which is why architectures are
[plugin modules](#six-years-of-architecture-progress-as-a-table) now rather
than arms in a `match`.

There is now an HTTP API and a web UI around it: OpenAI-compatible completions,
plus model management, training, evals, benchmarks and monitoring. None of that
makes it a fast server, and it is built not to pretend otherwise — requests
queue behind one another because the engine runs one generation at a time, and
the queue depth is a number on the dashboard rather than a mutex nobody can
see. What it did change is that the decode numbers in this README now have a
button that reproduces them: interleaved rounds, medians and ranges, and a
refusal to measure a machine that is busy doing something else.

It still has none of what makes a server *fast*: no continuous batching, no
paged KV cache, no kernels of its own on CUDA, no speculative decoding. The KV
cache is a `Vec<f32>` per layer that grows by appending, and 805 MB of it at
Qwen's full context. Nothing here has been benchmarked against vLLM or SGLang,
because a single-sequence engine and a serving engine do not yet have a number
in common.

So the second job is a direction, not a claim. [The roadmap](#where-to-go-next)
says what would have to become true first, starting with the one thing every
measurement in this repo keeps pointing at.

## Quick start

```bash
./scripts/get-mnist.sh
cargo test                                      # includes a gradient check
cargo run --release -p nanograd --bin train_mnist
cargo run --release -p nanograd --bin train_text -- --data README.md
cargo run --release -p kvad -- train --data README.md --name readme
cargo run --release -p kvad -- run --model readme --prompt "## "
cargo run --release -p kvad -- run --prompt "Why is the sky blue?"
cargo run --release -p kvad-gpu -- run --prompt "Why is the sky blue?"
cargo run --release -p kvad-tui

cd web && npm ci && npm run build && cd ..   # once, for the web UI
cargo run --release -p kvad-serve            # then http://127.0.0.1:8080
```

Verified on an M5 Pro:

```
epoch  1  loss 0.2085  test accuracy 96.32%  (2.1s)
epoch 10  loss 0.0072  test accuracy 97.98%  (21.0s)
```

```
llama · 30 layers · 9 heads (3 KV heads, 3x grouped) · 576 embd · 8192 ctx
134.5M parameters · instruction-tuned

The sky appears blue because of the way our eyes detect light. [...]
[prefill 41 tokens in 1.10s · generated 56 in 1.62s = 34.6 tok/s]
```

---

## Crate 1: `nanograd` — where the learning actually happens

Zero dependencies. Read in this order:

1. **[`matrix.rs`](crates/nanograd/src/matrix.rs)** — three matrix products.
   `A@B` forward, `Aᵀ@B` for weight gradients, `A@Bᵀ` for input gradients.
2. **[`nn.rs`](crates/nanograd/src/nn.rs)** — the important one. Every layer
   implements `forward(x) -> y` and `backward(dL/dy) -> dL/dx`. Chaining the
   second one backwards *is* backpropagation.
3. **[`bin/train_mnist.rs`](crates/nanograd/src/bin/train_mnist.rs)** — the
   training loop: predict, score, blame, adjust.
4. **[`attention.rs`](crates/nanograd/src/attention.rs)** — causal
   self-attention, forward and backward. Read it after the rest: it is three
   `Linear` layers and two of the products from step 1, and the only new
   derivative in it is the softmax's.
5. **[`norm.rs`](crates/nanograd/src/norm.rs)** — LayerNorm and RMSNorm. The
   forward pass is two lines; the file is about the backward pass, and the
   argument that gets you there without the algebra: a normalised row cannot
   see its input being shifted or stretched, so the gradient must have no
   component in those directions. LayerNorm ignores both and subtracts two
   projections. RMSNorm ignores only the stretch and subtracts one.
6. **[`embedding.rs`](crates/nanograd/src/embedding.rs)** — token ids to
   vectors. A lookup does not look differentiable, but it is a `Linear` layer
   fed one-hot rows with the multiplications by zero skipped, so its backward
   pass is `Linear`'s with the same shortcut: add each gradient row into the
   table row its token selected. Add, not assign — a token used three times is
   to blame three times.
7. **[`block.rs`](crates/nanograd/src/block.rs)** — the residual connection,
   `y = x + f(x)`, and the transformer block, which is two of them. The backward
   pass is `dx = dy + f.backward(dy)`, and the first term is the point: whatever
   the branch does to the gradient, `dy` also reaches the layer below untouched.
   One test makes it concrete — 24 layers that each pass back about a tenth of
   their gradient deliver 3e-22 of it in a chain, and 1.5 of it with the `x +`.
8. **[`model.rs`](crates/nanograd/src/model.rs)** — a GPT. Token and position
   embeddings, a stack of blocks, a final norm, a head. Almost no new code,
   because almost nothing is new; what is new is why positions are needed at
   all, and how a model should be initialised so that it starts out ignorant
   rather than confidently wrong.
9. **[`optim.rs`](crates/nanograd/src/optim.rs)** — AdamW. SGD has one learning
   rate for a model whose gradients differ by orders of magnitude between
   tensors. Adam divides each parameter's gradient by how large that
   parameter's gradients usually are, and the size cancels: scale every
   gradient by 1000 and the trajectory does not change. What is left is sign
   and consistency, and `lr` comes to mean "how far a parameter may move per
   step". It lives outside the layers — they own parameters and gradients, it
   owns its running averages — and reaches them through `params()`.
10. **[`text.rs`](crates/nanograd/src/text.rs)** and
    **[`bin/train_text.rs`](crates/nanograd/src/bin/train_text.rs)** — from a
    text file to a model that writes. Nobody labels this data: the target at
    every position is the character that comes next, so one window of text is
    a whole batch of examples and a megabyte holds a million windows. Then
    generation, which is only "predict, draw, append, ask again" — and which
    recomputes every earlier position for every new character, the waste the
    KV cache in crate 2 exists to remove.
11. **[`checkpoint.rs`](crates/nanograd/src/checkpoint.rs)** — the trained
    model on disk. A model is a list of named arrays of floats and nothing
    else, so the only decision in saving one is which names — and those are a
    contract with whoever reads the file. This crate uses GPT-2's, so what it
    writes is a GPT-2 checkpoint that crate 2 loads with the code it uses for
    OpenAI's. ([`json.rs`](crates/nanograd/src/json.rs) is there because the
    files are JSON and the crate has no dependencies. It teaches nothing about
    networks; skip it.)

### The test worth running first

```bash
cargo test -p nanograd analytic_gradient_matches_numerical
```

It nudges one weight by ±0.001, measures how the loss actually moves, and checks
that against what `backward()` claimed. A subtly wrong gradient still trains,
just badly — this is the only thing that catches it.

Attention has its own, which checks every weight of all four projections and
the gradient handed back to the layer below. With correct derivatives the
analytic and numerical gradients differ by about 0.0007; drop the row average
from the softmax derivative, forget the `1/sqrt(d)` on the way back, or miss the
transpose in `dK`, and that becomes 0.2 to 0.8. Two more tests pin the causal
mask from both sides: the future cannot change the past's output, and the past's
loss cannot blame the future's input.

The norms get the same treatment, with one trap worth knowing about. Check a
fresh LayerNorm on inputs drawn from N(0, 1) and both `gamma` and the row's
standard deviation are 1 — so a backward pass that forgot to multiply by either
one passes. The tests use a scrambled `gamma` and inputs with a standard
deviation near 3 for exactly that reason. A gradient check is only as good as
the point you run it at.

The embedding has a better test available than a numerical one, and uses it:
build the one-hot matrix for real, push it through the matmuls from step 1, and
require the lookup and its gradient to match *to the bit*. No tolerance to tune,
because there is no approximation — the shortcut performs the same additions in
the same order.

Checking the whole block turned up two things no single layer had shown.

*ReLU cannot be checked tightly.* Nudge a weight and some hidden unit crosses
zero, where the slope jumps and a centred difference is simply wrong. Correct
gradients disagreed with their estimates by up to 0.011 with ReLU in the MLP,
and by under 0.0002 with GELU. That is why the block uses GELU, and it is not
cosmetic: dropping a `3` from GELU's own derivative measures 0.011–0.014, which
a ReLU-sized tolerance would have waved through.

*Attention's key bias does nothing.* The check reported a 100% disagreement on
one tensor. Adding the same vector to every key moves every score in a row by
the same amount, and softmax only sees differences within a row — so the
gradient is exactly zero, and the check was comparing rounding noise with
rounding noise. Move that bias by 5.0 and the output moves by 1e-6. GPT-2 ships
one in every layer. (RoPE rotates the keys after the bias is added, which makes
it matter again; Qwen has one for that reason.)

The assembled model has tests of a different kind, because its bugs are of a
different kind — not wrong calculus but forgotten wiring.

*A fresh model should know nothing.* Predicting every token equally costs
`ln(vocab)`, and that is what the first loss should be. With the He
initialisation that suits the MNIST network it is 0.92 higher, averaged over 40
seeds: almost a nat of confidence with nothing behind it, which training must
first undo. With GPT-2's N(0, 0.02) it is 0.003 higher.

*Every tensor must actually train.* `step` and `zero_grad` are forwarded by
hand through each composite layer, and a forgotten line is silent — that tensor
just never moves, and the model trains around it. A model whose position
embedding never updated still memorised its test sequence. So one test takes a
step and requires every tensor to have moved.

*It can memorise one sequence.* Loss 2.41 to 0.0001 in 100 steps. The oldest
sanity check there is, and the first time forward, backward and the optimiser
all have to agree, through every layer at once.

An optimiser has no gradient to check, but Adam makes promises exact enough to
test instead. With bias correction the first step is `lr * g / (|g| + eps)`, so
gradients of 500 and of 0.00001 both move their weight by `lr` — the second one
0.1% short, which is precisely what `eps` is for. A steady gradient covers `lr`
per step whatever its size; one that flips sign every step, at ±100, gets 0.018
from home in 200 steps that could have covered 2.0. In a valley a million times
steeper one way than the other, Adam reaches the bottom along both axes in 300
steps; SGD, at the largest learning rate the steep axis allows, moves 0.0006
along the shallow one. And with no gradient at all, only weight decay acts:
matrices shrink by `1 - lr * decay` a step and biases and gains do not move —
the test that tells AdamW from Adam with L2 folded into the gradient.

### Things to try

- `--hidden 16` — how small before accuracy collapses?
- `--lr 0.5` — watch it diverge. `--lr 0.0001` — watch it crawl.
- `--momentum 0` — see how much of the speed was momentum.
- Note that training loss keeps falling after epoch 5 while test accuracy
  flatlines. That gap is overfitting, live.

**Momentum is a multiplier on your learning rate.** At steady state the velocity
converges to `lr·g/(1−μ)`, so `--lr 0.1 --momentum 0.9` really steps at ~1.0.
That is why the default is 0.02: at 0.1 this model plateaus at 95%, at 0.02 it
reaches 98%.

### Training a GPT on a text file

```bash
./scripts/get-text.sh        # tiny Shakespeare, 1.1 MB; or bring your own
cargo run --release -p nanograd --bin train_text
cargo run --release -p nanograd --bin train_text -- --data README.md
```

Any plain text works. The numbers below are from the second command — this
README, 50 KB, as the entire training set — because it was the text to hand.
Defaults: 2 layers, 4 heads, `d_model` 64, context 64, 117,221 parameters. The
run below took 91 seconds for its 2000 steps, at about 22,000 characters a
second on one core of an M5 Pro. It takes about 11 seconds now, for reasons given
under [Where the time went](#where-the-time-went); the losses and samples are
from the slower code and have not been regenerated.

```
loss to beat: 4.615 knowing nothing, 3.378 knowing only letter frequencies

step   250  train loss 2.924  validation loss 2.767
thatrir po the, the man as ivilt nth  ate  tasar at<(d wcat rach, isthath 0

step  1250  train loss 1.666  validation loss 1.909
quantisation not shat mak is it — sare prop is behad line changed that it

step  2000  train loss 1.334  validation loss 2.261
> measured gather step is all, and the same scomentum onhere that of
```

The two numbers at the top are what make the rest readable. `ln(vocab)` is a
model that knows nothing; the second is one that knows which characters are
common and nothing about their order. Everything below that was learned from
order: first that letters come in word-sized runs, then which runs, then
markdown. By the end it has opinions about quantisation.

It is also overfitting, in plain sight. Validation loss bottoms out near step
1250 and climbs from there while training loss keeps falling — 45 KB is little
enough to start memorising. The validation set is the *end* of the text rather
than a random sample, because windows overlap: sample at random and nearly every
validation window shares most of its characters with a training window, and the
number measures memory.

**AdamW earns its place here, which it did not on the toy problem.** Same
model, same text, 750 steps, each optimiser at the best of the learning rates
tried for it (AdamW: 0.001–0.03, best 0.003; SGD with momentum: 0.01–1.0, best
0.1, diverging to NaN at 1.0). Validation loss over four seeds: AdamW 2.18–2.47,
mean 2.26; SGD 2.56–2.65, mean 2.59. AdamW is ahead on every seed, though one
of its four is a good deal worse than the others. Warm-up was the guess, and
[it was right](#the-seed-that-was-worse-than-the-others).

**A mistake no training run can reveal.** A batch here is a loop — the model
takes one sequence at a time and gradients accumulate — so each window's
gradient has to be divided by the batch size. Forget to, and the gradient is a
sum rather than a mean: `batch` times too large. Under SGD that is a learning
rate `batch` times too high and you would notice. Under Adam it is *nothing*,
because cancelling the size of the gradient is the whole point of Adam; the
model trains identically. It stays wrong, waiting for the day someone changes
the optimiser. The test for it uses a text with exactly one possible window, so
that a batch of three must produce the same gradient as a batch of one.

Things to try: `--layers 1` or `--d-model 32`, to see how little is needed to
learn spelling; `--context 8`, to see what a model that cannot see a whole word
writes; `--temperature 0.2` against `1.5`; `--warmup 0 --decay-to 1 --clip 0`,
which is how every run in this repository worked before
[the schedule](#the-seed-that-was-worse-than-the-others) was measured; and
`--steps 6000` on a small file, to watch the validation loss leave.

### Keeping what it learned

```bash
alias train_text="cargo run --release -p nanograd --bin train_text --"
train_text --data README.md --save out/readme
train_text --load out/readme --steps 0 --prompt "## "     # just write
train_text --load out/readme --data README.md --steps 500 --save out/more
```

`--save` writes a directory of three files: `model.safetensors`, `config.json`
and `tokenizer.json`. They are GPT-2's names, GPT-2's tensor layout and the
HuggingFace tokeniser format, written by hand with no library — the 117K model
is 471 KB, all but 2.4 KB of it floats. The tokeniser has to travel with the
weights: id 17 means whatever character was seventeenth in *that* text, and a
loaded model refuses text containing a character it has no row for.

Saving and loading back bit for bit is tested, and proves less than it seems
to. A writer and a reader that share a misunderstanding — query and key in the
wrong thirds of GPT-2's fused matrix, say — agree with each other perfectly.
The only real test of a file format is a second implementation, and this
repository has one: [a test in crate 2](crates/llm/tests/nanograd_checkpoint.rs)
saves a `nanograd` model, loads it with the GPT-2 code written to run OpenAI's
weights, and requires the same logits at every position. They agree to 5e-7 of
their size. Of 22 mistakes made on purpose, the round trip missed the ones made
the same way in both directions; that test caught every one that touched the
layout, the smallest at 0.2.

It also had a hole of its own, found the same way. With the floats written
big-endian the engine loads garbage and produces NaN — and the test passed,
because `f32::max` prefers anything to a NaN, so the largest difference between
two rows of NaN came out as zero. A comparison that cannot fail on NaN is not a
comparison.

Two things in the file are not GPT-2's. Its output head is its own tensor with
a bias, where GPT-2 reuses the embedding table, so the config says
`tie_word_embeddings: false` and the engine's GPT-2 learned to honour that. And
the optimiser's state is not saved: a resumed run starts Adam's averages from
nothing. Measured against the same run left alone, that cost 0.06 and 0.04 of
training loss over the first 50 steps on two seeds, nothing on a third, and
nothing visible on any by step 100.

`--save` keeps the best model rather than the last — see
[One tool: `kvad train`](#one-tool-kvad-train), where the same loop gets a
name and a home.

### Where the time went

Training ran at 22,000 characters a second, and the plan was to use more cores.
A profiler was run first — this repository has
[been burned before](#the-floor-under-everything) — and eight seconds of
`sample` said something nobody had guessed:

| | share of the time |
|---|---|
| `matmul_a_bt` (`dx = dy @ Wᵀ`) | 49% |
| `matmul` (`y = x @ W`) | 18% |
| `matmul_at_b` (`dW = xᵀ @ dy`) | 13% |
| `tanh`, inside GELU | 10% |

Three matrix products do the same number of multiplications, and one of them
cost as much as the other two and everything else combined. The other two add
a multiple of one row into another, which a compiler turns into SIMD without
being asked. `matmul_a_bt` was a dot product: `acc += a[p] * b[p]`. Floating-
point addition is not associative, a compiler may not change what a program
computes, and so it may not reorder that sum; each addition waited for the one
before it. The disassembly showed the multiplications vectorised four at a
time and the additions done singly, in a chain. The fix is to say in the code
that the order is free — keep eight running sums and add them up at the end
([`matrix.rs`](crates/nanograd/src/matrix.rs)). Four sums did nothing; 16 and
32 were slower than 8, because attention's vectors are 16 long and a chunk that
does not fill falls to the slow loop. GELU was computing each `tanh` twice,
forward and again backward, and now keeps it.

Then the cores. The windows of a batch are independent until their gradients
are added, so `text::Replicas` gives each thread a copy of the model and a
share of the windows, and adds the gradients into the one model with an
optimiser: broadcast, compute, all-reduce, which is data parallelism as a GPU
cluster does it, with a core for a GPU and a `memcpy` for the network. Window
`i` always goes to replica `i mod n`, so a run is reproducible for a given
thread count; between thread counts only the order of a sum differs. Over 1000
steps on three seeds, 1 thread and 16 printed the same training and validation
loss to three decimals at every checkpoint but one, where they differed by
0.001.

| characters a second, batch 16 | range over 5 runs | median |
|---|---|---|
| as it was | 20,600–21,700 | 21,500 |
| eight-lane dot product, `tanh` kept; 1 thread | 36,100–37,300 | 36,400 |
| 6 threads | 141,400–145,700 | 143,900 |
| 12 threads | 142,900–192,300 | 177,100 |
| 16 threads (one window each) | 163,200–208,900 | 190,200 |

That is 1.7x from two changes to the arithmetic and 5.2x more from threads:
8.9x. The default 2000 steps, sampling included, take 10.5 to 12.7 seconds
rather than 91. The runs were interleaved, five of each, on a machine 70% idle
before and 87% after. An earlier attempt is worth recording: halfway through
it another job started 144 processes that spin forever, the unchanged code
measured a third of its own speed, and every ratio held while every absolute
number was wrong.

Sixteen threads buy 5.2x, not 16x, and the profiler says why. This machine has
6 fast cores and 12 slow ones; every step ends when the slowest thread does, and
the main thread spent 73% of its time waiting. Six threads, which fit on the
fast cores, get 4.0x, and repeat to within 3%; past six the range opens to 28%,
because it depends on which cores the scheduler picks. Letting threads pull windows from a shared queue, so that
fast cores take more, was tried and measured no better than noise at these
batch sizes — and it would cost reproducibility, since which replica sums which
windows would then depend on scheduling. The rest is the serial part: adding up
16 copies of the gradient, AdamW, and starting 16 threads a step, together
about a quarter of the main thread's time.

### The seed that was worse than the others

Four AdamW seeds, and one of them finished at validation 2.47 against about
2.19 for the rest. The guess written down at the time was learning-rate
warm-up. It was a guess; this is the measurement.

First, reproduce it. Eight seeds this time, 750 steps, the same 117K model on
this README, validation loss at the end (which is now measured on a fixed set
of windows, so two runs are comparable):

| `--lr` | per seed | worst | mean | spread |
|---|---|---|---|---|
| 0.001 | 2.43 2.46 2.44 2.46 2.44 2.47 2.43 2.45 | 2.47 | 2.446 | 0.037 |
| 0.002 | 2.31 2.31 2.37 2.33 2.36 2.40 2.31 2.34 | 2.40 | **2.343** | 0.094 |
| 0.003 | 2.40 2.45 2.30 2.40 2.32 2.41 2.29 2.29 | 2.45 | 2.358 | 0.167 |
| 0.006 | 2.69 2.56 2.43 2.62 2.60 2.67 2.55 2.57 | 2.69 | 2.587 | 0.258 |
| 0.01 | 2.81 2.80 2.73 2.79 2.80 2.81 2.77 2.78 | 2.81 | 2.786 | 0.082 |

There it is, and it is not really "one seed in four". It is that **the spread
grows with the learning rate** — 0.037, 0.094, 0.167, 0.258 — until at 0.006
every seed is bad. The bad seed was the first sign of a ceiling the runs were
already pressed against.

That is the shape warm-up predicts. Read
[`optim.rs`](crates/nanograd/src/optim.rs) on bias correction again: it exists
so that the *first* step moves every parameter by the full `lr`, in the
direction of a gradient estimated from one batch, and `v` is an average of one
sample. Adam is at its most confident exactly when it knows least. Nothing
diverges visibly, because the damage is a handful of early steps in a poor
direction; it shows up later as one run ending worse than its neighbours.

So: four arms, each swept over the same six learning rates and the same eight
seeds, each reported at its own best rate.

| arm | best `--lr` | per seed at that rate | mean | spread |
|---|---|---|---|---|
| flat | 0.002 | 2.31 2.31 2.37 2.33 2.36 2.40 2.31 2.34 | 2.343 | 0.094 |
| cosine decay only | 0.002 | 2.39 2.41 2.48 2.45 2.45 2.46 2.44 2.39 | 2.434 | 0.094 |
| warm-up only | 0.006 | 2.27 2.30 2.33 2.33 2.34 2.29 2.28 2.31 | 2.305 | 0.069 |
| both | 0.006 | 2.25 2.26 2.29 2.31 2.31 2.25 2.26 2.27 | **2.275** | 0.058 |

Three things worth saying out loud.

**Warm-up works, and not by making a good run better.** At 0.003 it is worth
0.025 (2.358 to 2.333) and nothing you would notice. What it does is raise the
ceiling: the flat arm falls apart at 0.006 and the warmed-up arm is at its best
there. The gain is three times the usable learning rate, and the improvement is
what the extra rate buys.

**Cosine decay alone is worse than doing nothing** — 2.434 against 2.343. Of
course it is: without a higher peak to pay for it, decaying the rate is just
training at a lower average rate. It earns its place only on top of warm-up,
where it is worth a further 0.03, and the floor barely matters (0.1 and 0.3
measured 2.275 and 2.273; even 1.0, which is no decay at all, gives 2.305).

**The warm-up has to be long enough to be one.** At `--lr 0.006` with the
cosine, over eight seeds: 10 steps of warm-up scored 2.627 and 25 scored 2.452
— *worse than no warm-up at all* — 50 gave 2.343 with one seed still at 2.70,
and 100 gave 2.275. That is why the default is a share of the run, a tenth,
rather than a number of steps: ten steps is a tenth of a hundred and a
seventy-fifth of the default run. (Half the run measured 0.025 better again, at
750 steps and at 2000, but the sign flipped on two of eight seeds, so it is
inside the spread and a tenth is what everyone uses.)

At the real default length of 2000 steps the effect is smaller and the same
shape. The current default — `--lr 0.003`, no schedule — scores 2.138 over
eight seeds. Turning warm-up, cosine decay and clipping on and changing nothing
else scores **2.087**, with the spread down from 0.052 to 0.041. And the
learning rate almost stops mattering: 0.003, 0.006, 0.01 and 0.02 land at
2.087, 2.097, 2.111 and 2.120, where the flat arm had already lost 0.45 by
0.01. That is the part that matters for a tool: the knob you are least equipped
to set is the one that now matters least.

**Gradient clipping is insurance, and the measurement says so exactly.** If the
whole gradient — every tensor laid end to end — is longer than `--clip`, every
element is scaled by one number, so the direction is untouched. At the good
setting it does nothing: 2.275 without it, 2.269 to 2.308 across thresholds
from 0.25 to 2.0, all inside the seed spread. At a bad setting it does
everything. At `--lr 0.02`, where warm-up alone is not enough, eight seeds ran
2.29 2.69 2.28 2.73 2.76 2.80 2.31 2.77 — mean 2.578, spread 0.517. With
`--clip 1` the same eight ran 2.31 2.30 2.28 2.32 2.30 2.31 2.28 2.31 — mean
2.301, spread 0.036. It cannot replace warm-up, though: on the flat arm at
0.006 it recovered 0.1 of the 0.24 and left the spread wider than it found it.

It is not free. Clipping is one extra pass over every gradient each step, and
on the 117K model that is 13% of training throughput (median of five
interleaved runs: 257,500 characters a second without, 221,700 with). On the
4.9M `large` preset it is not measurable at all — 7,139 against 7,185, inside
the noise — because there the arithmetic dwarfs it. So the cost falls entirely
on the run that takes nine seconds and not at all on the one that takes twenty
minutes, which is why it is on by default.

The obvious optimisation was tried and refused. A left-to-right sum of a
hundred thousand squares is a chain of additions each waiting on the one
before, which is exactly the problem [eight running sums
fixed](#where-the-time-went) in `matmul_a_bt`. Eight running sums here moved
the cost from 13.9% to 12.3%, which is inside the run-to-run noise, so the
simpler code stayed: the cost is the extra pass over the gradients, not the
order they are added in. (The sum is in `f64` for a different reason, and that
one is real — on a hundred thousand gradients of 1e-3, `f64` gives the correct
0.3162278 and `f32` gives 0.3160589.)

### Where the text comes from: `kvad crawl`

```bash
kvad crawl https://doc.rust-lang.org/book/        # writes doc.rust-lang.org-book.txt
kvad train --data doc.rust-lang.org-book.txt --name rustbook
```

`--data corpus.txt` assumes somebody has a corpus. `scripts/get-text.sh`
fetches tiny Shakespeare and that is the whole of the supply, so `kvad crawl`
reads a documentation site into a text file instead — and the Datasets page in
the web UI does the same thing as a job you can watch.

**The scope is the directory, not the domain.** This is the decision the
feature turns on. `doc.rust-lang.org/book/` links into `/std/` on nearly every
page, so a crawl that stayed on the domain comes back with the whole of the
standard library's rustdoc — tens of thousands of pages of generated
signatures, which is not what anyone meant by "the book". Links are followed
under the starting address's own directory: `/book/` for that address, and
`/book/` still for `/book/ch03-00-common-programming-concepts.html`.
`--same-host` widens it for the sites that are one book at their root.

The extractor is a tag scanner rather than a parser or a dependency, because a
documentation site is one program's output from one template. Headings, lists,
tables and fenced code blocks with their language survive; `<script>` is read
as raw text so that `a < b` in JavaScript is not a tag; `<nav>`, `<footer>`
and `id`/`class`/`role` furniture are dropped by token. Links are collected
from the *whole* document and the text only from the furniture-free part — a
book's table of contents is its sidebar, and a crawler that dropped the
sidebar before looking for links fetches one page and stops.

**Then the part that decides whether any of it is trainable.** A character
tokeniser gives an id to every distinct character and the model gets a row per
id, so the alphabet is a hyperparameter, and the web is bad at keeping it
small: three kinds of quotation mark, two kinds of dash, a non-breaking space
before every unit, one emoji in a warning box. Typography is mapped onto ASCII
— not NFKD, which would also take the accents off `blåbær` — and what is left
of the tail is dropped by count. Ten pages of the book:

```
10 pages, 99,684 characters, 95 distinct
277 characters mapped onto ASCII (237 of them ’ → '), one emoji dropped
```

Six pages come out at 81 distinct characters, every one of them ASCII, against
the 101 of this README. The threshold is an absolute count and not a share of
the corpus, because that is what the number means: a row of an embedding table
seen eight times is untrained whether the text around it is 5 kB or 5 MB. It
started at 3 and is 10, from measuring — at 3, a 325,334-character crawl still
kept about forty rows for the Japanese, Hindi, Hebrew and Cyrillic in one
chapter's "hello world" examples, seen three to fifteen times each. ASCII is
never dropped however rare it is.

**Pointing it at the real thing is what found the bugs.** `class="header"` was
in the furniture list, and that is exactly how mdBook and rustdoc mark every
heading's anchor: every heading on every page was being deleted, and the
orphaned `# ` then landed on the paragraph below. The same first run produced
1,319,972 characters from three pages, because `/book/` and
`/book/title-page.html` are one page at two addresses and `print.html` is the
entire book again. Pages are deduplicated by the hash of their text now, and
`print.html` and rustdoc's `all.html` are passed over by name.

Beside the text goes `<file>.crawl.json`: the start address, the scope, every
page in the order it was fetched, what was skipped and why, what the cleaning
pass changed, and the alphabet it ended with. In six months that is the
difference between "a corpus" and "this corpus, from here, on that day".

`robots.txt` is obeyed, there is a pause between requests, and the crawl stops
at a number of pages and a number of bytes — a crawler an admin session can
point at a stranger's server should be boring. Addresses resolving to
loopback, RFC1918 or link-local are refused and rechecked on every redirect,
because "fetch this URL for me" is the shape of request that otherwise reads a
cloud metadata endpoint into a file. That check is of the name and not of the
socket, which a resolver of our own would close and nothing less will.

### One tool: `kvad train`

```bash
kvad train --data corpus.txt --name shakespeare    # train a new model
kvad train --from shakespeare --data more.txt      # train it further
kvad run --model shakespeare --prompt "ROMEO:"
kvad ls                                            # it is listed, by name
```

`train_text` and `kvad` used to be two programs with a directory passed
between them. They are one now, and the joining piece is a name. `--name
shakespeare` writes into `$XDG_DATA_HOME/kvad/models/shakespeare`, and from
then on the bare word means that model wherever you are: `kvad run --model
shakespeare`, `kvad use shakespeare`, `kvad rm shakespeare`, `kvad-tui
shakespeare`. A model name is resolved in three steps — a directory of that
name that exists, then a model trained here, then a Hub repo id — and the
three cannot collide by accident, because a repo id always has a slash in it
and a name may never have one. A directory still wins, which is the rule
`transformers` uses.

The training loop itself did not move house so much as move down a floor. It
lives in [`nanograd::text::train`](crates/nanograd/src/text.rs) now, where
both front ends call it: `train_text` still takes `--layers`, `--d-model`,
`--heads` and `--context` one at a time, for seeing what each of them does,
and `kvad train` takes `--size` instead, along with `--warmup`, `--decay-to` and
`--clip`, whose defaults come from
[the measurement above](#the-seed-that-was-worse-than-the-others). Nothing
about the learning changed in the move, and tests say so: every loss a run reports is the same however many
threads it uses (to 2.6e-5 — the losses and not the weights, because replicas
sum in a different order and Adam divides by the size of the gradient, so
where a gradient is near zero a difference of 1e-7 in it is still a step of a
whole learning rate), and the training loss a checkpoint prints is the mean
since the last one, including a final interval shorter than the rest — which
is invisible whenever the steps divide evenly, as the defaults do.

**It keeps the best model, not the last.** A run on a small text overfits in
plain sight — validation loss bottoms out and then climbs while training loss
keeps falling — and the old `--save` wrote whatever step the run ended on,
which is the memoriser. Now the directory is rewritten every time validation
improves, and a run that gets worse leaves the good model where it was. This is
early stopping done by keeping the best rather than by halting: the run still
finishes and still prints, so the overfitting stays visible instead of being
hidden by a loop that quietly gave up.

That comparison only means something if the two losses are measured on the
same windows, and they were not: evaluation drew from the training generator,
so every checkpoint scored a different sample — and, worse, `--eval-every 100`
and `--eval-every 250` trained two different models, because the draws came
out of the same stream as the training windows. Evaluation has a generator of
its own now, seeded the same way every time.

**`--from` is the owner's "add new training sets to existing models",** and it
has two limits worth saying out loud, both of which are in `kvad train --help`:

* **The vocabulary is fixed at first training.** A character tokeniser gives
  ids to the characters it saw, and the model has one row per id in its
  embedding table and one in its output head. There is no row to give a
  character that was not there the first time, so text containing one is
  refused, and the message says which character it was. Training a new model
  on both texts together is the answer.
* **The optimiser's state is not saved.** AdamW keeps two running averages per
  weight and a resumed run starts them from nothing. Measured: at most 0.06 of
  training loss over the first 50 steps on two seeds, nothing on a third, and
  nothing visible on any by step 100.

`kvad ls` lists trained models in a section of their own rather than mixed in
with downloads, and `kvad rm` says something different about them, because they
are a different kind of thing: a downloaded model can be fetched again and a
trained one cannot. Twenty-nine mutations were made on purpose to see which of these
claims a test would actually defend — the last model saved instead of the best,
a bar that starts at zero so nothing is written, the tokeniser left out of the
directory, `..` accepted as a model name, a name sent to the Hub without being
looked for here, an unseen character quietly given a fresh vocabulary. Two
survived the first pass, and both were real gaps: nothing required two
checkpoints in one run to be measured on the same windows, and nothing stopped
two directories with the same last component from being one model to the
quantised-weight cache. Both have tests now.

**`--size` is measured, not guessed.** Three shapes, timed the way everything
in this repository is timed — interleaved, five runs of each, on this README
as the training text, batch 16, all 16 threads, an M5 Pro 93% idle. The
parameter counts are for that text's 101 characters: the embedding table and
the output head are as wide as the vocabulary is, so another text gives
another count.

| `--size` | shape | parameters | chars/s, 5 runs | median | default 2000 steps |
|---|---|---|---|---|---|
| `small` | 2 layers, d_model 64, ctx 64 | 117,221 | 137,100–260,600 | 228,800 | 9 s |
| `medium` | 4 layers, d_model 128, ctx 128 | 835,685 | 38,500–52,000 | 39,500 | 1m 44s |
| `large` | 6 layers, d_model 256, ctx 256 | 4,856,421 | 5,800–7,700 | 7,000 | 19m 25s |

(Sampling off and one validation pass at the end, which is why `small` comes
out at 9 seconds where [the threads table](#where-the-time-went) says 10.5 to
12.7 for the same run with sampling on. The ranges are wide for the reason
given there: 16 threads on 6 fast and 12 slow cores, and every step ends when
the slowest one does.)

Those numbers are from one machine, though, and a table in a README cannot
know yours. So the run times its own first ten steps and says what it expects
to cost here, before there is anything to regret:

```
$ kvad train --data README.md --name big --size large
size large: 6 layers, 8 heads, d_model 256, context 256
model: Gpt(6 layers, 8 heads, d_model 256, vocab 103, context 256, 4857447 params)
loss to beat: 4.635 knowing nothing, 3.341 knowing only letter frequencies
7958 characters a second here — about 17m 04s to go. Ctrl-C now if that is too long.
```

Ten steps is the whole cost of knowing, and on the slowest preset that is
about nine seconds. Which is also the shape of the next problem: five million
parameters is a fifth of the 10–30M this repository wants to reach, and it
already takes twenty minutes to do two thousand steps. The
[roadmap](#finishing-the-story) says what that costs and what would have to
change.

### Running it in the engine

```bash
kvad run --model readme --greedy --prompt "## "     # a name, or a directory
kvad use readme              # make it the default, from any directory
kvad-tui readme              # open the TUI with it loading
```

Wherever the engine takes a repo id it takes a name or a directory, by the rule
`transformers` uses: a directory that exists wins. From there it is the same
code path as a model from the Hub — the same loader, tokeniser, KV cache and
quantised kernels — so the 117K model gets `--quant q8` for free. (How fast it
is cannot honestly be said: its context holds 64 tokens, the run is over in
about 20 ms, and six of them measured anywhere from 970 to 5,600 tokens a
second.) With sampling off, the engine and `nanograd` wrote the same 61
characters for each of three trained models; [a test](crates/llm/tests/nanograd_checkpoint.rs)
makes that journey on every run, from a text to a trained model to a directory
to the engine's output.

This is a base model in the plainest sense: it continues text. It has no chat
template and no idea what a question is, and a prompt containing a character it
never saw loses that character silently, because that is what the tokeniser
library does with one.

**The bug that was waiting for this.** The quantised-weight cache has to know
whether a checkpoint is still the file it quantised. For Hub models it reads
the content hash out of the HuggingFace cache's symlink, for free. For any
other file it fell back to the size — and the size of a checkpoint is decided
by the architecture, so a model retrained and saved over itself is the same
length *to the byte*. Reproduced before it was touched: train, run with
`--quant q8`, retrain into the same directory, run again, and the engine mapped
the old cache and wrote the old model's text under the new model's name, while
`--quant f32` beside it wrote the new one's. Nothing had ever saved over a
checkpoint before, so nothing had ever hit it. The identity is now the size and
the modification time — `make`'s answer, with `make`'s flaw, that a copy which
preserves timestamps can defeat it. Hashing would close that, and would cost
more on a hand-placed 8 GB checkpoint than the cache saves.

---

## Crate 2: `kvad` — the same ideas, at scale

The examples below assume the binaries are on your `PATH`:

```bash
cargo install --path crates/llm --path crates/tui
```

Otherwise prefix each one with `cargo run --release -p kvad --` (or `-p kvad-tui`).

```bash
kvad search smollm                 # find models; says which we can run
kvad pull HuggingFaceTB/SmolLM2-360M-Instruct
kvad ls                            # what is downloaded, and how big
kvad use  Qwen/Qwen2.5-0.5B-Instruct
kvad run  --prompt "Explain backpropagation in one sentence."
kvad chat --system "You are terse."
kvad info --model openai-community/gpt2-medium   # config only, no weights
kvad arch                          # architectures this build can run
kvad cache                         # pre-quantised weight files
```

Read in this order:

1. **[`tensor.rs`](crates/llm/src/tensor.rs)** — matmul, LayerNorm, GELU,
   softmax, then RMSNorm, SwiGLU and RoPE. Nine functions, four architectures.
2. **[`weights.rs`](crates/llm/src/weights.rs)** — safetensors is a length, a
   JSON header, and raw floats. Plus shard indexes and bf16 widening.
3. **[`model/mod.rs`](crates/llm/src/model/mod.rs)** — the skeleton the
   architectures share, including attention itself.
4. **[`model/gpt2.rs`](crates/llm/src/model/gpt2.rs)** — read first, it is
   simplest. Then **[`model/llama.rs`](crates/llm/src/model/llama.rs)**,
   written to be read as a diff against it, and
   **[`model/deepseek.rs`](crates/llm/src/model/deepseek.rs)**, a diff against
   *that*. **[`model/arch.rs`](crates/llm/src/model/arch.rs)** is the registry
   they plug into, and where to look when adding a fifth.
5. **[`quant.rs`](crates/llm/src/quant.rs)** — block-wise int8/int4 weights
   and the kernels that consume them, then
   **[`simd.rs`](crates/llm/src/simd.rs)** for the one instruction the compiler
   will not reach on its own, and
   **[`qcache.rs`](crates/llm/src/qcache.rs)** for doing that work once
   instead of once per load.
6. **[`sampler.rs`](crates/llm/src/sampler.rs)**, then
   **[`chat.rs`](crates/llm/src/chat.rs)**.

### Six years of architecture progress, as a table

| GPT-2 (2019) | Llama family (2023+) | DeepSeek V2/V3 (2024+) | Why |
|---|---|---|---|
| learned position rows (`wpe`) | RoPE: rotate Q and K by angle ∝ position | RoPE on *part* of each head, YaRN-interpolated | the rest of the head has to survive being multiplied by a matrix |
| LayerNorm (centre, scale, bias) | RMSNorm (scale only) | RMSNorm, plus one on the compressed vector | the centring was never load-bearing |
| GELU MLP, 2 matrices | SwiGLU, 3 matrices | 64 or 256 SwiGLUs, a router, and 1–2 that always run | parameters you have, without arithmetic you pay for |
| multi-head attention | grouped-query attention | multi-head *latent* attention | the KV cache is what a server runs out of |
| one fused QKV matrix | three projections | two, one of which is a rank-512 bottleneck | Q and KV now have different widths, then different ranks |

The last row of that table is the interesting one, and it is worth stating as
numbers. For DeepSeek-V2-Lite — 16 heads, 128-wide values — ordinary
multi-head attention would store 5120 floats per position per layer.
Grouped-query attention would divide that by the group factor. MLA stores
**576**: one 512-wide compressed vector, and one 64-wide rotary key shared by
every head.

It gets away with it because the per-head matrix that would turn that vector
into a key can be moved onto the query instead — `q · (W c) = (Wᵀ q) · c` —
and there is one query per step against thousands of keys. That identity is
the whole architecture, and [`model/deepseek.rs`](crates/llm/src/model/deepseek.rs)
is mostly an explanation of it.

What did *not* change across all three: the residual stream, the alternation
of attention and MLP, the causal mask, scaled dot-product attention. `attend()`
in `model/mod.rs` is shared verbatim between GPT-2 and Llama — grouped-query
attention is just a smaller `kv_dim`, and GPT-2 is the `n_kv_head == n_head`
case. DeepSeek is the first one that needed its own, which is what
[`model/arch.rs`](crates/llm/src/model/arch.rs) exists for.

And the detail worth sitting with: **SmolLM2-135M is smaller than GPT-2-medium
and holds a conversation, while GPT-2 cannot.** The architecture changes above
are real but marginal. Nearly all of that gap is training data and
post-training. Running both in the same binary makes the point better than any
benchmark.

### The one that needed its own file

Everything above this point is true of GPT-2 and of Llama, and none of it was
true of the first mixture-of-experts checkpoint pointed at it.
DeepSeek-V2-Lite is the model this engine grew a registry for, and it is worth
looking at what it actually costs to run:

```
deepseek_v2 · 27 layers · 16 heads · 2048 embd · 163840 ctx · 102400 vocab
15706.5M parameters · weights 17670 MB (cpu q8) · loaded in 92.2s

def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

15.7 billion parameters, of which the router picks **2.4 billion** per token:
six experts of sixty-four, plus two that always run. It costs a 2.4B model to
decode and knows what a 16B model knows, and that is the entire argument for a
mixture of experts.

Decode, five interleaved rounds each, medians and the range across all
samples — [the protocol the benchmark page enforces](#what-it-measures-and-what-it-refuses-to),
run from a shell because one of these takes 17 GB of RAM:

| | weights | decode | prefill (13 tokens) |
|---|---|---|---|
| q8 | 16.5 GB | **23.4 tok/s** (22.8–24.7) | 0.47s (0.46–0.58) |
| q4 | 9.1 GB | **18.0 tok/s** (17.7–18.2) | 0.44s (0.44–0.49) |

**q8 decoded faster than q4 here, by 30%, while reading twice the bytes.** A
mixture of experts is the case where you would most expect the memory argument
to win — decoding touches only six experts of sixty-four, so q4 saves more than
a gigabyte of traffic per token — and it lost anyway.

The explanation this README gave was that decoding is `m = 1`, where there is
no [i8mm tile](#batched-prefill-and-i8mm) to take: `matvec_bt` walks one weight
row at a time, q8's goes straight into `sdot`, and q4's has to be masked and
shifted apart first. Reading half as much memory does not help once you are no
longer waiting on memory.

That was the right mechanism and much too large a number for it, and the
difference was a kernel nobody had written. The nibble unpacking was costing
about five times what it should, because of *which* loop LLVM chose to
vectorise; written out by hand it costs 9%, and
[the section that chases that down](#the-nibble-kernel-the-compiler-would-not-write)
is the more useful half of this story. Re-measured on the same model, five
interleaved rounds, with q8 as the control:

| | q8 (untouched) | q4 before | q4 after |
|---|---|---|---|
| decode | 24.8 tok/s (24.3–25.0) | 19.2 (18.0–19.5) | **26.8** (26.4–27.6) |

q4 goes from 22% behind q8 to **8% ahead**, on 7.4 GB less memory. The control
arm varies by 1% across ten runs, so the 40% is not the machine.

(Both tables were measured idle. The second reads a little faster on q8 too,
which is a different prompt and generation length rather than anything
changing: the q8 kernel was not touched, so take the control's agreement
between arms as the thing to trust and not the absolute rate. The first sample
of each run is discarded — faulting 16.5 GB of weights in from the page cache
costs a 6.5 s prefill against 0.7 s warm, and that is a disk measurement.)

q4 on this model is now what the name suggests — and it took two sessions and a
disassembler to stop it being a trade.

### Base models versus instruction-tuned

GPT-2 is a **base** model: pure next-token prediction, no instruction tuning.
Ask it a question and it writes more questions, because that is what its
training data looked like. `kvad ls` and `kvad search` label which is which, and
`kvad chat` warns you.

An instruction-tuned model only behaves like an assistant when wrapped in the
exact marker tokens it was trained on. Those live as a **Jinja template** in
`tokenizer_config.json`, one per model, and they genuinely differ — so
[`chat.rs`](crates/llm/src/chat.rs) renders the model's own template rather than
hardcoding one. "The model is dumb" is very often "the template is wrong".

### Quantisation

```bash
kvad run --quant q8 --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
```

`--quant q8|q4` quantises the weight matrices as they load. Each row is chopped
into blocks of 32 with its own scale, so a single outlier only ruins its own 32
neighbours instead of flattening the whole tensor — which is what per-tensor
scaling does, and why it fails.

Measured on Qwen2.5-0.5B (494M parameters, M5 Pro, 18 threads):

| | weights | tok/s | `first 8 primes` |
|---|---|---|---|
| f32 | 1976 MB | ~34 | `2, 3, 5, 7, 11, 13, 17, 19` |
| q8 | 556 MB (3.6x) | ~38 | `2, 3, 5, 7, 11, 13, 17, 19` |
| q4 | 309 MB (6.4x) | ~37 | `2, 3, 5, 7, 11, 13, 17` |

**q8 is free; q4 mostly is not.** q8 reproduces f32 exactly on every test,
including with the activations quantised as well. q4 gets GPT-2's counting and
SmolLM2's arithmetic right but still drops a prime above — fluent, confident,
quietly wrong. Half-billion-parameter models have less redundancy to spare than
the 7B+ models where q4 is usually judged.

> **A correction.** q4 used to fail *all three* tests, and this README used to
> say so. That was a bug in the scale, not a property of 4-bit weights — see
> [the scale that wasted a code](#the-scale-that-wasted-a-code).

### The scale that wasted a code

Four bits span sixteen codes, `-8..7`. The obvious scale is
`block_max_magnitude / 7`, which is symmetric and reads naturally. It also never
produces `-8`: one code in sixteen is unused and every step is 1/7 of the range
instead of 1/8.

Dividing the *signed* extreme by `-8` uses all sixteen. The sign is what makes
it work — whichever end the extreme sits at, it lands on `-8` and the rest of
the block spreads over the remaining codes. The trade is that the *opposite*
extreme now clips to 7/8 of its value, so it is not a free win; measured over
random blocks it is a **9.54% → 8.49%** mean relative error, about what the
8/7 step ratio predicts.

On real models that margin decided three test cases. Before the fix, q4 turned
`17 + 25 = 42` into `40`, sent GPT-2 into `"The jury's jury's jury's"`, and
started the primes at 11. After it, the first two are correct.

I found this by comparing against GGML's `Q4_0` through candle, which produced
better output than my own q4 from the same checkpoint. That is the argument for
porting a model to a framework even when you have written it yourself: the
framework is a second opinion, and it disagreed for a reason.

```bash
cargo run --release -p kvad --example bench_matvec
```

The kernel benchmark is where the interesting part is. On the output head
(151936 x 896, the largest single matmul per token):

| | ms/call | GB/s | vs f32 |
|---|---|---|---|
| f32 | 2.69 | 202 | 1.00x |
| q8 + quantised activations | 0.91 | 169 | **2.96x** |
| q4 + quantised activations | 1.28 | 67 | 2.11x |

That is close to the ceiling: 545 MB in 2.69 ms is 202 GB/s, and the q8 version
moves a third of the bytes at nearly the same rate. The speedup comes from
quantising the *activations* too, which turns the inner
loop into an integer dot product. On this machine that compiles to exactly what
you would hope for — six instructions per 32 weights:

```asm
ldp     q2, q3, [x10], #0x20   ; 32 weight bytes
movi.2d v4, #0
sdot.4s v4, v3, v1             ; 16 int8 multiply-accumulates
sdot.4s v4, v2, v0             ; 16 more
addv.4s s0, v4                 ; horizontal sum
```

Getting there took four fixes, each a general lesson:

- **Keep the accumulator in registers.** The first kernel read and wrote all 32
  outputs on every input row: 256 bytes of output traffic per 32 bytes of
  weights. Quantising made it *slower than f32* (0.90x).
- **Use more than one accumulator.** Float addition is not associative, so the
  compiler cannot reorder a `sum +=` chain; the loop runs at FMA latency rather
  than throughput. Integer addition *is* associative — one reason the integer
  path is easier to vectorise.
- **Check what the inner loop doesn't depend on.** The 4-bit offset correction
  needs a per-block sum of the activations, which depends only on the input.
  Computing it inside the row loop repeated it 151,936 times per call.
- **Never mix floats into an integer reduction.** This was the big one. With
  the per-block scaling inline, the loop compiled to scalar loads — no `sdot`
  at all, despite the same expression vectorising perfectly as a standalone
  function. Splitting it into an integer pass and a scaling pass was worth
  roughly 2x on its own. Worth knowing that a correct, innocuous-looking line
  can silently cost you the vector units.

### One kernel, after a threshold that measured the wrong thing

There used to be two kernels here, chosen by weight size. The integer path
ran 4.9x faster on the 153 MB output head and *slower* on a 5 MB MLP matrix,
so `matvec_bt` dispatched on `INTEGER_PATH_MIN_BYTES` and the explanation
wrote itself: a cache-resident matmul is not bandwidth-bound, so shrinking the
weights buys nothing and the unpacking is pure overhead.

Plausible, and wrong. The benchmark called the kernels from the main thread,
which is not a rayon worker, so every call paid a flat ~0.17 ms of cold-path
dispatch — invisible beside a 153 MB matmul, and the entire runtime of a 5 MB
one. The tell was sitting in the output the whole time:

```
qwen mlp.down  [896 x 4864]
  f32       17 MB     0.17 ms/call
  q8         5 MB     0.17 ms/call
  q4         3 MB     0.17 ms/call
```

Six times the bytes, the same time, three times over. That is not a kernel
being bandwidth-bound; that is a kernel that is not being measured at all.
With the harness fixed to run inside a pool ([`bench_matvec`](crates/llm/examples/bench_matvec.rs)),
the same matrix drops to 0.09 ms and the integer path wins at every size —
1.13x to 1.70x end to end across three models and both precisions, nothing
slower. So the threshold is gone and there is one kernel. The dequantising one
survives as the baseline the integer path is measured against, behind
`KVAD_DEQUANT=1`.

### The nibble kernel the compiler would not write

For most of this project q4 decoded *slower* than q8, on every model, and this
README said so in three places with an explanation attached: `m = 1` decode
walks one weight row at a time, q8's row goes straight into `sdot`, and q4's
has to be masked and shifted apart first. Reading half the bytes does not help
once you are not waiting on bytes.

The mechanism was right. The size of it was not — and the size was an accident
of what LLVM decided to vectorise.

`sdot` multiplies `i8` by `i8`. A 4-bit code is stored unsigned, `0` to `15`,
with the bias taken off afterwards; mask one out of a byte and widen it and you
get a *zero*-extend, and a zero-extended operand against a sign-extended one is
not a shape `sdot` can take. So the q4 loop got `smull`/`smlal` instead —
eight lanes an instruction where `sdot` does sixteen, four of them per block,
with a widening step in front of each.

The obvious fix is to subtract the 8 inside the loop, which makes the weight
genuinely signed. It is exactly equivalent arithmetic — `sum((n-8)v)` and
`sum(nv) - 8 sum(v)` agree over the integers — and it made things **five times
slower**. Casting through `i8` without subtracting does nothing at all: LLVM
knows the top bits are zero and folds the sign-extend straight back. Measured
on one core, a 896-wide row, block dots only:

| | GMAC/s |
|---|---|
| unsigned nibbles, bias per block *(what shipped)* | 24 |
| signed nibbles, same iterator form | 4 |
| signed, one fused accumulator | 7 |
| signed, with the block sizes as array types | 2 |
| q8, for scale | 88 |

Every attempt to make the inner loop signed made the autovectoriser abandon the
reduction and vectorise *across blocks* instead — loading weights a byte at a
time into lanes with `ld1.b`, dozens of single-byte loads where there had been
one `ldr q`. The disassembly is unambiguous about it, and no amount of
rephrasing the Rust talked it out of the idea.

So the kernel is written down, next to `SMMLA` in [`simd.rs`](crates/llm/src/simd.rs),
and it is ten lines:

```rust
let packed = vld1q_u8(w);                      // 16 bytes = 32 weights
let bias = vdupq_n_s8(-8);
let lo = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f))), bias);
let hi = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), bias);
let acc = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x));
vdotq_s32(acc, hi, vld1q_s8(x.add(16)))
```

Two ops remove the bias from thirty-two weights at once, which is the thing the
scalar loop could not express. Four blocks run per iteration so that `vpaddq`
reduces four accumulators in three instructions instead of four `addv`s. That
lands at **80 GMAC/s against the portable path's 24** — and within 9% of q8,
which is what unpacking nibbles should actually cost.

Note what this is *not*. It is not an instruction Rust cannot name; `vdotq_s32`
is stable and the compiler emits `sdot` for q8 without being asked. It is the
compiler declining to *choose* it — the same reason `SMMLA` is hand-written one
screen up, arrived at from the opposite direction.

The old path is still there for CPUs without `dotprod`, and `KVAD_NO_DOTPROD=1`
selects it, which is how everything below was measured. Both produce
bit-identical output — verified, not assumed — because the two ways of removing
the bias are the same integers.

### What the kernel is worth end to end

Interleaved rounds, `KVAD_NO_DOTPROD=1` against the kernel, same binary and
the same weights, medians with the range across all samples. **q8 is the
control**: its kernel was not touched, so whatever its two arms differ by is
what this machine's noise is worth.

| | q8 (control) | q4 before | q4 after | |
|---|---|---|---|---|
| Qwen2.5-0.5B | 110.7 tok/s (107–111) | 91.3 (84–96) | **116.4** (114–117) | 1.27x |
| DeepSeek-V2-Lite 15.7B | 24.8 (24.3–25.0) | 19.2 (18.0–19.5) | **26.8** (26.4–27.6) | 1.40x |
| SmolLM2-135M | 173.7 (169–176) | 159.2 (155–171) | **175.9** (170–182) | 1.10x |

The control's two arms agree within 1% on all three models; q4 moves 10–40%.
The gain tracks matrix size, which is what it should do — a 135M model spends
most of a decode step in dispatch, and there is no kernel for that.

**q4 is now the fastest CPU precision on every model here**, which has not been
true before: 116 against q8's 111 on Qwen, 176 against 174 on SmolLM2, 26.8
against 24.8 on DeepSeek.

And the microbenchmark, which is where it is cleanest:

```
qwen lm_head  [151936 x 896]
  f32      545 MB     2.67 ms/call   1.00x
  q8       153 MB     0.89 ms/call   2.99x
  q4        85 MB     0.74 ms/call   3.60x
```

### What it is worth end to end

| | f32 | q8 | |
|---|---|---|---|
| Qwen2.5-0.5B | 64 tok/s | 112 tok/s | 1.75x |
| SmolLM2-135M | 165 tok/s | 194 tok/s | 1.18x |
| GPT-2-medium | 67 tok/s | 125 tok/s | 1.86x |

Quantisation buys most of what the byte counts say it should, which is worth
stating because for most of this project's life it bought nothing at all —
every one of those f32 and q8 numbers used to be about 35 tok/s, whatever the
model and whatever the precision. [Removing the floor](#the-floor-under-everything)
is what let the kernels through; before that, the honest summary of this
section was "a 4.9x kernel bought under 1.4x overall", and the Amdahl's-law
moral drawn from it was measuring a scheduler.

For a long time the honest footnote here was that q4 bought none of this: at
91 tok/s it was *slower* than q8's 112, so its only advantage was memory. That
was [a missing kernel, not a price quantisation charges](#the-nibble-kernel-the-compiler-would-not-write).
q4 now decodes at 116, ahead of q8. It is still much less accurate, and that
part is real.

### Batched prefill, and i8mm

A prompt used to go through the model one token at a time, re-reading every
weight matrix once per token. Feeding tokens through together reads each weight
once and reuses it across the batch — a memory-bound operation becomes a
compute-bound one. `forward_batch` does that, in chunks of 64.

Causal masking comes for free, which is the neat part: push all the batch's
keys and values into the cache first, then let query `i` attend over exactly
`pos0 + i + 1` positions. No mask anywhere. (Every tutorial's triangular matrix
is only needed when you *don't* have a KV cache to be careful with.)

Once prefill is a real matrix-matrix product, ARM's `i8mm` extension applies.
`SMMLA` multiplies a 2x8 block of `i8` by an 8x2 block and accumulates a 2x2
`i32` result — 32 multiply-accumulates in one instruction, twice `SDOT`'s 16.

**But look at the shape it wants: two independent activation rows.** Generating
one token at a time gives you exactly one, so half the result lanes would be
duplicates and the useful throughput collapses back to `SDOT`'s. `i8mm` cannot
help decoding. Only prefill.

Prefill, 640 tokens, Qwen2.5-0.5B, q8:

| | time | |
|---|---|---|
| one token at a time (`KVAD_PREFILL_CHUNK=1`) | 6.20 s | |
| batched f32 GEMM | 1.93 s | **3.2x** from batching |
| batched q8, `SDOT` (`KVAD_NO_I8MM=1`) | 1.81 s | |
| batched q8, `SMMLA` | 1.20 s | **1.51x** more from i8mm |

Prefill drops from ~9.7 ms/token to ~1.9 ms/token.

Note where f32 lands: **1.93 s against q8's 1.81 s.** Once the weights are
reused across a batch, prefill is compute-bound, and quantisation is an
answer to a memory problem. On the smaller `q_proj` matrix the f32 GEMM is
actually *faster* than the q8 `SDOT` path, because unpacking and rescaling cost
more than the bytes they save. Quantisation is mostly a decode optimisation.

`i8mm` is worth more on a single core than across the pool, because with every
thread running memory bandwidth becomes the limit again and a faster
instruction has less to offer. Both knobs above are environment variables
precisely so this is measurable rather than asserted.

(Every figure in this table used to be two to three times larger — the
token-at-a-time row was 18.7 s. Batching was never the only thing making
prefill slow; see [the floor](#the-floor-under-everything).)

Rust exposes `SMMLA` only through an unstable intrinsic, so
[`simd.rs`](crates/llm/src/simd.rs) emits it with inline assembly — stable, and
four lines. Availability is detected at runtime, so one binary still runs on
CPUs without it.

**q4 takes this kernel too, and for a while it did not.** The gate was one
condition — `matches!(self.data, Data::Q8 { .. })` — and everything at q4
prefilled on the row-at-a-time fallback while unpacking nibbles on top. The web
UI's benchmark page is what turned it up, by measuring time to first token
beside decode rate and showing q4 losing the one while winning the other.

The fix is a format problem rather than a kernel problem. `SMMLA` wants eight
contiguous `i8`; q4 stores two weights per byte, with the *low* nibble of each
byte belonging to the first half of a block and the high nibble to the second,
and the sign carried as a bias removed later during scaling. Unpacking a row
pair into plain `i8` — subtracting the 8 on the way, which is exactly the
correction `row_dot` applies afterwards — hands the existing kernel an operand
it already understands, and costs `n` bytes of work reused across all `m`
activation rows.

Prefill, 862 tokens of this README, Qwen2.5-0.5B, five rounds interleaved:

| | median | range | |
|---|---|---|---|
| q4, row at a time (`KVAD_NO_I8MM=1`) | 4.58 s | 4.05–4.77 | |
| q4, `SMMLA` | 2.54 s | 2.02–2.74 | **1.8x** |
| q8, `SMMLA` | 2.52 s | 2.03–2.78 | |

A second run of five put q4 at 2.15 s against 4.68 s, so the gain is 1.8–2.2x
depending on the round. The absolute times are worse than the table above
because the machine was not quiet; the ratios are the claim, and interleaving
the settings is what makes them survive that. The line that matters is the
third: q4 prefill went from roughly half q8's speed to level with it, while
still decoding faster. In the server's own benchmark, time to first token at q4
moved from 41 ms to 33.8 against q8's 34.3 — a difference that has stopped
existing.

#### Checking that, after the decode kernel turned up a five-fold hole

[The 4-bit decode kernel](#the-nibble-kernel-the-compiler-would-not-write) was
5x off from what the source suggested, and the section above makes the same
kind of claim about prefill on a noisier measurement — so it is worth asking
whether prefill has a hole of its own. It does not, and the way that was
established is the point.

The first problem was that nothing measured it. `bench_batch` had rows for
`f32 gemm`, `q8 sdot` and `q8 smmla` and no q4 at all: the benchmark's blind
spot was exactly where the question lived. With q4 in it, and swept across the
batch size rather than measured at one:

| m | q8 `SMMLA` | q4 `SMMLA` | q8 `SDOT` | q4 `SDOT` |
|---|---|---|---|---|
| 2 | 77.9 | 71.6 | 71.1 | 66.7 |
| 8 | 153.6 | 151.6 | 130.5 | 131.2 |
| 32 | 285.4 | 279.7 | 141.9 | 146.1 |
| 64 | 306.3 | 306.0 | 161.0 | 161.5 |
| 128 | 319.7 | 309.6 | 207.4 | 210.8 |

GMAC/s on `mlp.down`, `[896 x 4864]`. q4 is level with q8 from `m = 8` up and
at worst 8% behind at `m = 2`, where the unpack is amortised across exactly one
pass of the kernel. `SMMLA` beats the row-at-a-time path at every batch size,
so the `m >= 2` gate is right too. End to end, 685 tokens on Qwen2.5-0.5B, six
rounds interleaved: **q4 1.34 s against q8 1.36 s.**

And the disassembly says why there was nothing to find. The nibble unpack
compiles to `ldr q` / `and.16b` / `usra.16b` / two `str q` — sixteen bytes in,
thirty-two out, fully vectorised. The split packing is the reason: the low
nibbles of sixteen bytes *are* sixteen consecutive weights, so unpacking writes
two contiguous runs and never scatters. The decode kernel's problem was never
that nibbles are awkward; it was one `zext` in a reduction the vectoriser was
looking at.

**One thing did change, in the path nobody was looking at.** A CPU without
`i8mm` prefills on `row_dot`, which calls the same block-dot the decode kernel
replaced. So the decode work lifted prefill too, on hardware that cannot reach
`SMMLA` at all — `KVAD_NO_DOTPROD=1` against it, `m = 64`:

| | q4 `SDOT` before | after | | vs q8 |
|---|---|---|---|---|
| `mlp.down` | 117.1 | **161.5** | 1.38x | 0.75x → 1.00x |
| `q_proj` | 95.3 | **138.1** | 1.45x | 0.73x → 1.02x |

On `q_proj` that also puts q4 ahead of the f32 GEMM (138.1 against 134.9),
which the section above notes was the one shape where quantisation lost on the
fallback. It no longer does.

So: no kernel written here, and the measurement that would have caught it if
there were is now in the benchmark instead of absent from it. Worth writing
down at the same length as a fix, because "we looked and there was nothing"
is only useful if you can see how hard anyone looked.

### The KV cache, and a fix that was slower than the bug

The [roadmap](#where-to-go-next) has called the KV cache the next bottleneck for
a while, on the grounds that it is the only part of a decode step that grows
with the conversation. Everything else — every matmul, all 169 of them — is the
same size at position 10 and at position 4000. So it is worth knowing what it
actually costs, and until `bench_attend` existed nothing measured it.

`attend` scores this token's query against every cached key. The loop was
written the obvious way:

```rust
let mut dot = 0.0f32;
for i in 0..hd {
    dot += q_head[i] * k_head[i];
}
```

One running sum — which [`tensor::matvec_bt`](crates/llm/src/tensor.rs), in the
next file along, already carries a comment warning against: *"four lanes, not
one running sum: a single chain would stall on FMA latency rather than run at
its throughput."* DeepSeek's latent attention had the same loop over a
512-wide vector, so its chain was 512 dependent adds long.

**Then the fix turned out to be slower than the bug.** Four lanes, the form the
repo already uses, on a 64-long dot over a 2048-entry cache, one core:

| | |
|---|---|
| one running sum *(what shipped)* | 17.6 us |
| four lanes | 22.8 us |
| **eight lanes** | **6.3 us** |

The disassembly says why, and it is not what the comment assumes. LLVM cannot
reorder float additions, but nothing stops it vectorising the *multiplies*: the
scalar loop compiles to four `FMUL.4S` and a chain of scalar `FADD`s, and
because consecutive positions are independent, the out-of-order engine overlaps
those chains across `t`. Four lanes replaces that with a single 128-bit
accumulator carried around the loop — one vector FMA per iteration, each
waiting on the last, and nothing to overlap it with. Eight lanes is two
independent chains, which is the first shape that actually beats doing nothing.

So the rule is not "lanes are better than a running sum". It is *two
accumulators or more*, and four lanes only looks like a fix because a 128-bit
vector makes it look like four.

`tensor::dot` is now eight lanes and shared, and `attend` and DeepSeek's MLA
call it instead of writing the loop out. Both versions of `attend` live in
[`bench_attend`](crates/llm/examples/bench_attend.rs), alternating in one
process and checked against each other every round:

| ctx | running sum | eight lanes | | cache/layer | attention alone |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B shape** — 14 heads, 2 KV, `head_dim` 64, 24 layers ||||||
| 512 | 22.2 us | 18.6 | 1.19x | 0.52 MB | 0.45 ms/token |
| 2048 | 74.8 | 60.3 | 1.24x | 2.10 MB | 1.45 |
| 8192 | 332.3 | 248.1 | **1.34x** | 8.39 MB | 5.95 |
| **Llama-8B shape** — 32 heads, 8 KV, `head_dim` 128, 32 layers ||||||
| 128 | 49.6 | 37.2 | **1.33x** | 1.05 MB | 1.19 |
| 512 | 148.8 | 108.3 | **1.37x** | 4.19 MB | 3.47 |
| 2048 | 505.4 | 420.3 | 1.20x | 16.78 MB | 13.45 |
| 8192 | 2293.1 | 2275.6 | **1.01x** | 67.11 MB | 72.82 |

**The last row is the more useful result.** At 67 MB of cache per layer the fix
buys nothing at all, because the loop has stopped being latency-bound and
become DRAM-bound — and the last column says why that matters: attention alone
is 73 ms per token there, which is most of the token.

**What it is worth end to end is smaller, and today unmeasurable.** Attention
is a minority of a decode step at these lengths — 1.45 ms of a ~12 ms token at
2048 on Qwen — so 1.24x of it predicts about 3%. Driving a real session at 4096
tokens of context, five rounds alternating, that is what the paired rounds
suggested (1.00x, 1.06x, 1.02x, 1.06x, 1.05x) — but the control, the same
measurement at 64 tokens of context where the fix has almost nothing to do,
swung 0.88x to 1.03x across those same rounds. The control's noise is wider than the effect, so
the honest answer is that this machine could not resolve it, and the isolated
numbers above are the claim. It gets decisive at long context on a big model,
and that is exactly where the second finding says the fix stops helping.

Which is the argument for the paged cache, arrived at from the measurement
rather than from the memory figure. Under grouped-query attention each cached
key is read **once per query head in its group** — seven times on Qwen, four on
that Llama shape — because `attend` parallelises over query heads and each one
walks the cache for itself. The bytes are shared; the reads are not. Fixing
that means restructuring attention around the KV head rather than the query
head, with the softmax split across position blocks, which is a different and
much larger change than a dot product. It is now a measured number instead of
an assumption, which is the part that was missing.

### The bug worth stealing

The first version of the `SMMLA` kernel was **slower than the `SDOT` path it
was meant to beat** — 65% slower on one core. The cause was one missing
attribute:

```rust
#[target_feature(enable = "i8mm")]   // <- this line
unsafe fn smmla_row_pair(...)
```

A `#[target_feature]` function can only be inlined into a caller declaring at
least the same features. Without it, every single `smmla` became a real
function call. It compiles, it is correct, it is tested, and it quietly throws
away everything the instruction was for.

Two measurement traps on the way there, both of which produced confident wrong
answers:

- The first comparison said `SMMLA` was **2.01x faster**. It was being compared
  against a fallback that had floats mixed into its integer loop, so the
  baseline was scalar code rather than `SDOT`. Fixing the baseline turned the
  result into a 7% *loss*.
- An isolated benchmark with a single accumulator said `SMMLA` was **2.4x
  slower** — that loop was latency-bound on its own accumulator chain. Four
  independent chains turned it into a 1.39x win.

Three different numbers for the same instruction, all measured, none of them
right until the last one. Worth remembering before quoting a speedup.

### The f32 GEMM, and register tiling

Batching only helped the quantised paths at first: with f32 weights `matmul_bt`
still called the matrix-vector kernel once per token, which reads the whole
weight matrix `m` times. [`gemm_bt`](crates/llm/src/tensor.rs) fixes that, and
the fix is the classic one.

A first attempt kept one weight row live and ran it against four activation
rows. That loads 5 vectors per 4 multiply-accumulates — the loop spends its
time fetching, not computing. The **4x4 micro-kernel** loads four weight
vectors and four activation vectors, then does sixteen multiply-accumulates
with them:

| tile | loads per 4-wide step | FMAs | GMAC/s |
|---|---|---|---|
| 1 row x 4 tokens | 5 | 4 | 78.9 |
| 4 rows x 4 tokens | 8 | 16 | **204.5** |

Same arithmetic, same memory traffic from RAM, 2.6x the throughput. The only
thing that changed is how many times each fetched value gets used before it is
thrown away. That ratio — arithmetic per byte fetched — is what decides whether
a GEMM runs at memory speed or at FPU speed, and 4x4 is sized so the 16
accumulator vectors plus 8 operand vectors fit in aarch64's 32 vector
registers.

### The optimisation that wasn't

Rust compiles `acc += a * b` to a separate `fmul` and `fadd`. It will not fuse
them on its own, because fusing changes the result: `a * b + c` rounds twice,
`fma(a, b, c)` rounds once. `f32::mul_add` asks for the fused form explicitly.

One instruction instead of two, *and* more accurate. It should be free money.
Measured across four tile shapes, it was consistently **worse** — 204 GMAC/s
became 93–116:

| | 4x4 | 4x2 | 2x4 | 2x2 |
|---|---|---|---|---|
| `mul_add` | 93.3 | 116.3 | 103.1 | 80.4 |
| `+= a * b` | **204.5** | | | |

An `fmla` needs all three operands live simultaneously. With 16 accumulators
and 8 operands already in flight, that tips the tile past the register file and
it spills to the stack. The cheaper instruction lost to the extra memory
traffic it caused.

In [`matvec_bt`](crates/llm/src/tensor.rs), where only four accumulators are
live, `mul_add` made no measurable difference at all — that kernel is
bandwidth-bound at ~220 GB/s, so its instruction count is irrelevant. Which is
the more useful half of the lesson: an optimisation is a claim about the
bottleneck, and if you have not measured the bottleneck you are guessing.

### Verifying you got it right

```bash
kvad run --model openai-community/gpt2 --greedy --prompt "1, 2, 3, 4, 5, 6,"
kvad run --model Qwen/Qwen2.5-0.5B-Instruct --greedy \
    --prompt "List the first 8 prime numbers, comma separated."
```

The first continues `7, 8, 9, ... 16`. The second answers
`2, 3, 5, 7, 11, 13, 17, 19`. A wrong transpose, a wrong RoPE convention or a
mishandled bias degrades output to *plausible-looking noise* rather than failing
loudly, so arithmetic is the sharp test. The GPT-2 one is also the regression
test for the Llama refactor, and both are the acceptance test for `--quant q8`.

### Doing the quantising once

Every `--quant q8` load re-derived the same 494 million codes from the same
bf16 checkpoint and threw them away on exit. [`qcache.rs`](crates/llm/src/qcache.rs)
writes them out instead, and maps the file back on later loads:

```bash
kvad run --quant q8 --prompt hi      # first: "quantising to q8 (first load)"
kvad run --quant q8 --prompt hi      # after: "mapping q8 weights (556 MB)"
kvad cache                           # what has been written, and where
kvad cache clear                     # it is all derived data; delete freely
```

| model | quantise | map | |
|---|---|---|---|
| SmolLM2-135M q8 | 0.30 s | 8 ms | 37x |
| Qwen2.5-0.5B q8 | 1.17 s | 20 ms | 58x |
| Qwen2.5-0.5B q4 | 1.11 s | 24 ms | 46x |
| GPT-2-medium q8 | 1.85 s | 2 ms | 925x |

The ratios are silly because the denominator is *nothing happening*. `mmap`
does not read the file; it wires the pages into the address space and returns.
The bytes arrive later, on demand, faulted in from the page cache they are
already sitting in. Startup stops scaling with model size, because startup
stops doing work — a 7B model maps in the same 2 ms as a 355M one.

That property is why the weights are stored in the layout the kernels already
want. A format that needed a fix-up pass — endian swapping, unpacking, even
just a `Vec` copy — would give most of it back.

**What it cost in the code.** `Weight` could no longer assume it owns its
arrays, so `Vec<i8>` became `Store<i8>`: either a `Vec` or a window onto a
mapping, `Deref`ing to a slice either way. Every kernel in `quant.rs` is
untouched — they all took `&[i8]` and still do. The loaders lost their
`Checkpoint` and gained a `Source` trait with two implementations, which is
also where GPT-2's transpose went: the cache stores post-transpose matrices,
so `matrix_t` is a property of the *checkpoint*, and the mapped source ignores
the distinction.

**The interesting hazard is staleness, not speed.** A cache that can serve
bytes written under different rules is worse than no cache at all — and this
is not hypothetical here. The q4 scale fix two sections up changed every
4-bit code in every file; had this existed then, the old weights would have
gone on quietly answering `17 + 25 = 40` long after the bug was fixed. So the
header records a format version, the precision, the model's shape, and the
source checkpoint's file sizes, and anything that does not match is rebuilt
with a line saying why.

**Where the time goes now.** With the quantising gone, the load breakdown is
worth reading:

```
  [ 0.752s] reading weights          <- 0.75 s of hf-hub resolving five files
  [ 0.752s] mapping q8 weights (556 MB)
  [ 0.772s] reading tokenizer        <- 20 ms, nearly all of it RoPE tables
```

Those 20 ms are not the weights. Qwen precomputes a sin/cos table for 32768
positions at load (`Rope::new`, 12.6 ms measured on its own); the rest is the
tokenizer. The dominant cost is now the Hub client resolving file paths —
0.75 s, the same whether the model is 135M or 7B, and the same offline.
Which is the usual shape of these things: remove the obvious cost and the
bottleneck moves somewhere you were not looking.

The GPU backend does not use this. candle's `quantize_onto` takes about 0.2 s
for the same model — fast enough not to be worth a second format, and its
natural on-disk form would be GGUF rather than ours.

### The floor under everything

For most of this project, CPU decode ran at about 35 tok/s. Not approximately
— *exactly* that, whatever you changed:

| | f32 | q8 | q4 |
|---|---|---|---|
| Qwen2.5-0.5B (1976 / 556 / 309 MB) | 34 | 35 | 36 |
| SmolLM2-135M, a quarter the size | ~35 | ~35 | ~35 |

Six times the bytes, the same speed. A model a quarter the size, the same
speed. Every kernel in this chapter — the `sdot` integer path, the SMMLA
tile, the tiled GEMM — measured faster in isolation and changed nothing here.
The explanations on offer were all plausible (a fixed cost per matmul; more
layers in the smaller model) and none of them were checked.

Six seconds with a sampling profiler ended it. Top of stack:

```
swtch_pri (kernel yield)                12776
the actual matvec kernel                 5186
__psynch_cvwait                          3652
__workq_kernreturn                       1585
```

One fifth of the CPU was doing arithmetic. And the main thread's stack was the
same every time:

```
matvec_bt -> Registry::in_worker_cold -> LockLatch::wait_and_reset
          -> pthread_cond_wait
```

**`in_worker_cold`.** A `par_iter()` called from a thread that is not a pool
worker cannot push onto a worker's deque; it has to inject the job into a
shared queue, wake the workers, and block the caller on a condition variable
until they are done. Two kernel transitions and a scheduler round trip. Fine
once. Decoding one token runs 169 matmuls, so it happened 169 times per token,
each one costing more than the arithmetic it was waiting for.

Three changes, measured one at a time:

| | | |
|---|---|---|
| `pool.install()` around the forward pass | 35 → 70 | caller blocks once per *token*, not per matmul |
| one job per thread instead of a split tree | 70 → 90 | no adaptive `join` tree, no steals between leaves, no `collect` |
| size the pool for big.LITTLE | — | 18 threads was slower than 11; see below |
| delete the integer/dequant threshold | 90 → 112 | [it measured this same bug](#one-kernel-after-a-threshold-that-measured-the-wrong-thing) |

**3.2x on q8, and the anomalies went with it.** SmolLM2-135M now decodes at
194 tok/s against Qwen-494M's 112, which is what a quarter the parameters
should look like. Quantisation now buys 1.75x where it used to buy nothing.
Prefill of 640 tokens went from 2.15 s to 1.16 s without being touched.

#### Not all cores are worth using

The pool is deliberately smaller than the machine. This host reports 18
logical CPUs, of which six are performance cores and twelve are efficiency
cores. Rayon splits a matmul into equal pieces regardless, and a parallel
section ends when its *slowest* piece does — so the E-cores set the pace of
every one of those 169 barriers:

| threads | 4 | 6 | 8 | 10 | 12 | 14 | 15 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| tok/s | 36 | 46 | 54 | 61 | 65 | 69 | 70 | 68 | 52 |

Using the whole machine is the worst setting above four threads. The default
is now the performance cores plus two thirds of the efficiency cores, and
`KVAD_THREADS` overrides it.

#### What this cost in credibility

Three things in this README had to be rewritten because of one bug: the
integer/dequant threshold, the "Amdahl's law" moral drawn from a 4.9x kernel
buying 1.4x, and the claim that SmolLM2's layer count explained its speed.
All three were reasonable stories told about a number that was really a
scheduler. The profiler took six seconds and would have found it at any point
in the previous four chapters.

---

## Crate 3: `kvad-gpu` — the same model, handed to a framework

```bash
kvad-gpu run  --prompt "Why is the sky blue?"
kvad-gpu run  --device metal --dtype bf16 --model Qwen/Qwen2.5-0.5B-Instruct
kvad-gpu chat
```

[`model.rs`](crates/gpu/src/model.rs) is the Llama forward pass again, in
[candle](https://github.com/huggingface/candle). Read it next to
[`llama.rs`](crates/llm/src/model/llama.rs): the structure is line for line the
same, and every operation is one you already wrote.

- `matvec_bt` becomes `Tensor::matmul`.
- The loop over attention heads becomes one batched matmul over a
  `[1, n_head, seq, head_dim]` tensor.
- `candle_nn::rotary_emb::rope` is the same `rotate_half` convention as
  `Rope::apply`, and `ops::rms_norm` the same scale-only normalisation.

Two differences are worth noticing because they run *against* the framework:

- **Batching is free.** The CPU engine needed a separate `forward_batch`,
  because a matrix-vector product and a matrix-matrix product are genuinely
  different kernels. On the GPU there is one `forward`, and a single token is
  just the narrow case.
- **The causal mask comes back.** With a KV cache the CPU engine got masking
  for free — the cache only ever held earlier positions. Here the whole batch
  is one matmul, so the future has to be masked out explicitly, with the
  triangular `-inf` matrix every tutorial shows.

### Numbers

| | CPU (q8) | Metal (bf16) | |
|---|---|---|---|
| Qwen2.5-0.5B, decode | 112 tok/s | 113 tok/s | 1.0x |
| SmolLM2-135M, decode | 194 tok/s | ~168 tok/s | 0.9x |
| Qwen2.5-0.5B, prefill (640 tokens) | 1.16 s | 0.19 s | **6.1x** |

**The GPU wins on prefill and, at this model size, not at all on decode.**
That is the same distinction as everywhere else in this repo. Prefill is
compute-bound, which is what a GPU is for. Decode is memory-bound, and there
the CPU is carrying int8 weights (556 MB) against the GPU's bf16 (1260 MB) —
quantisation claws back a whole hardware generation, and on the smaller model
the CPU is ahead.

That row used to read 35 tok/s and 3.3x, and the smaller model used to be
*slower* on the CPU than the larger one. Both were
[a scheduling bug](#the-floor-under-everything), not a property of the
hardware. It is the same lesson as the rest of the repo, learned the
expensive way: the number you are explaining may not be a number about the
thing you think it is about.

Metal `f32` runs at about two thirds of bf16's rate, for exactly double the
memory. `bf16` is the default because it is also what the checkpoints ship as.

The GPU backend covers the **Llama family only**; GPT-2 stays on the CPU engine,
and asking for it says so rather than failing obscurely.

### Quantised weights on the GPU

```bash
kvad-gpu run --quant q8   # or q4, q4k, q6k
```

candle's `QTensor` holds GGML's block formats — the same scheme as
[`quant.rs`](crates/llm/src/quant.rs): blocks with a shared scale, `Q8_0` and
`Q4_0` being 32-wide exactly like ours. `QMatMul` keeps HuggingFace's
`[out, in]` layout and transposes inside its kernel, where the dense path wants
the transpose done once at load, so [`Proj`](crates/gpu/src/model.rs) hides the
difference.

All six backends, Qwen2.5-0.5B, decode:

| backend | weights | tok/s |
|---|---|---|
| cpu f32 | 1976 MB | 64 |
| cpu q8 | 556 MB | 112 |
| cpu q4 | 309 MB | 116 |
| metal bf16 | 1260 MB | 113 |
| **metal q8** | **525 MB** | **191** |
| metal q4 | 278 MB | 228 |

Quantisation is worth **1.7x** on both now — 113 → 191 on the GPU, 64 → 112 on
the CPU. It used to be worth nothing on the CPU, and the explanation given here
(a per-matmul cost that the bytes could not touch) was correct about the
mechanism and wrong about the conclusion: that cost was
[removable](#the-floor-under-everything), not inherent. With it gone, both
backends are bandwidth-bound in decode and behave the same way.

And then prefill, where it stops being free — eventually:

| prompt | metal bf16 | metal q8 | |
|---|---|---|---|
| 140 tokens | 0.070 s | **0.040 s** | q8 1.75x faster |
| 640 tokens | 0.180 s | **0.160 s** | q8 1.12x faster |
| 1340 tokens | 0.390 s | 0.380 s | even |
| 5040 tokens | **2.590 s** | 2.750 s | q8 1.06x *slower* |

There is a crossover, at around 1300 tokens on this machine. A short prompt is
still narrow enough that reading the weights dominates, so smaller weights win;
a long one turns every matmul into real work over a wide batch, and then the
dequantisation is pure overhead. The bottleneck moves with the batch size, and
the point where it crosses is a property of this GPU rather than of
quantisation.

(The CPU engine once claimed a similar crossover against *matrix* size, and
that one turned out not to exist —
[it was measuring a scheduler](#one-kernel-after-a-threshold-that-measured-the-wrong-thing).
This one is measured across four prompt lengths with five runs each, which is
the standard the other claim failed.)

> **Correction.** An earlier version of this section claimed a flat "1.6x
> slower on prefill" from a single pair of measurements at 654 tokens (0.24 s
> against 0.39 s). That does not reproduce: five runs at each of four prompt
> lengths give the table above, with a median equal to the minimum every time.
> Re-running the old arrangement behind `KVAD_GPU_DENSE_EMBED=1` rules out the
> embedding change as the cause, so the original figure was simply a bad
> measurement on a loaded machine. The *shape* of the claim survives — there is
> a length past which quantising costs you — but it arrives much later and much
> more gently than reported. Measure the curve, not two points on it.

One practical note. The k-quants (`q4k`, `q6k`) need dimensions divisible by
256 — Qwen is 896 wide, so they are refused up front with a message naming
`q8` and `q4` instead of failing halfway through the load.

### The table that was stored twice

```bash
kvad-gpu run --quant q8                          # quantised table
KVAD_GPU_DENSE_EMBED=1 kvad-gpu run --quant q8   # the dense one, for comparison
```

The embedding table was the last dense tensor in a quantised GPU model, and on
a small model it is not a small one: 136M of Qwen2.5-0.5B's 494M parameters live
in `[151936, 896]`. At bf16 that is 272 MB.

It stayed dense because the lookup needs an operation nothing else in the engine
wants — *gather rows and dequantise only those*. A `QTensor` is a run of blocks,
not a matrix, so row `t` is a span of blocks that has to be decoded on its own.
candle turns out to ship exactly that kernel (`QTensor::embedding`, which is
GGML's `get_rows`), so the change is one method call rather than a Metal shader.

**The compression is the smaller half of it.** Qwen ties its embeddings — small
models nearly always do — so the lookup table and the output head are the *same
matrix*. They were stored twice anyway, because a dense `index_select` wants
`[vocab, n_embd]` and a matmul wants the transpose. Quantise the lookup and both
want the identical thing: one `Arc<QTensor>`, two uses.

| | before | after |
|---|---|---|
| metal q8 | 797 MB | **525 MB** |
| metal q4 | 550 MB | **278 MB** |

525 MB for 494M parameters is 8.5 bits each, which is exactly `Q8_0` — 8-bit
codes plus an f16 scale per 32. The model is now at the format's floor, with
nothing left dense but the norms, and the win over bf16 goes from 1.6x to
**2.4x** (4.5x at q4).

**And it changed the speed by nothing at all:** 196.6 → 195.7 tok/s at q8,
239.6 → 237.3 at q4, prefill identical to three decimals. Exactly as it should
be. One row of 151936 is read per token, so the table was never on the
bandwidth path — it was pure capacity. Everywhere else in this repo,
quantisation traded accuracy for speed. Here it trades accuracy for *room*, and
that is the whole point: memory is what stops you loading a bigger model, and a
bigger model is worth far more than 1% of tok/s.

The CPU engine needed no change for any of this. `Weight::row` has dequantised
one row at a time since quantisation was added, and the tied head has always
been the same `Weight` — the hand-written version got here by the path of least
resistance, because one type did both jobs. The framework version duplicated
precisely because its two operations wanted two different types.

### One trait, two backends

Adding the GPU needed a change to the CPU engine's shape.
[`Transformer`](crates/llm/src/model/mod.rs) deliberately keeps the KV cache
*outside* the model, because on the CPU it is a `Vec<f32>` the caller can own.
A GPU backend cannot work that way — its cache lives in device memory and must
never round-trip through the host between layers.

So the cache moved inside, behind a `Session` trait: `forward`, `cached`,
`truncate`. Everything above it — prefix reuse, sampling, streaming, the chat
loop, the TUI — stopped caring which backend it was talking to. The CLI keeps
its `--quant`, the GPU CLI gets `--device` and `--dtype`, and in the TUI `p`
cycles through all six: `cpu f32`, `cpu q8`, `cpu q4`, `gpu bf16`, `gpu q8`,
`gpu q4`.

That is the usual shape of this kind of work: the interesting part was not the
new backend, it was the seam the old one had to grow.

---

## Crate 4: `kvad-tui` — the app

```bash
cargo run --release -p kvad-tui
```

`/` search · `↑↓` select · `enter` download and load · `p` cycle backend ·
`d` delete · `tab` switch to chat · `esc` interrupt generation.

`p` cycles all six backends — `cpu f32`, `cpu q8`, `cpu q4`, `gpu bf16`,
`gpu q8`, `gpu q4` — and is the quickest way to feel the trade-offs: load a model at
f32, ask it something arithmetic, reload at q4 and ask again, then reload on
the GPU and watch the token rate.

Three concerns on three threads: the UI loop only draws and reads keys, the
engine thread downloads and generates, and `rayon` fans each matmul across cores
underneath. They talk over channels — except cancellation, which a channel
cannot express because the worker is busy inside `generate`, so that is a shared
`AtomicBool` the per-token callback checks.

The chat keeps its KV cache **across turns**. Each turn it re-encodes the
conversation, finds the common prefix with what is already cached
(`Llm::common_prefix`), truncates to there, and prefills only the new message.
The status bar reports how many tokens that saved — `[33.3 tok/s · 96 cached]`.
Without it, turn *N* re-reads the entire transcript.

This crate is the least educational of the three. It is `ratatui` plumbing and
state management — good Rust, no ML. Build it last.

---

## Crate 5: `kvad-serve` — the server, and the web UI

```bash
cd web && npm ci && npm run build && cd ..
cargo build --release -p kvad-serve
kvad serve                         # loopback on 8080, no auth
kvad serve --bind 0.0.0.0:8080     # needs an auth mode, or --insecure
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

Ten pages: a dashboard, model management, chat, a playground, training,
datasets, evals, benchmarks, monitoring and settings — plus the API's own
documentation at `/api`. Four authentication modes (`none`, `local`, `basic`,
`oidc`), roles, API keys. SQLite for everything the filesystem cannot answer.
`docs/ui-plan.md` is the plan it was built from, and each phase in it records
what that phase measured and which of its open questions closed.

### One model at a time, said out loud

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

### What it measures, and what it refuses to

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
*slower* until [its kernel was written](#the-nibble-kernel-the-compiler-would-not-write),
and this page had simply found the one model whose matrices were small enough
to hide it. Prefill is bound by
arithmetic — and q8 had a batched `SMMLA` kernel there while q4 did not, so it
fell back to a row at a time. Both are integer; neither dequantises. That was a
missing kernel rather than a price quantisation charges, and
[it has since been written](#batched-prefill-and-i8mm): q4 prefill roughly
doubled and the 41-against-30 gap closed to nothing.

This is the first thing the web UI paid for. Nobody would have run that
comparison from a shell, because it needs two model loads and ten interleaved
generations to say anything, and the number it turned up was hiding in plain
sight behind a decode rate that looked fine.

### The playground is where the logits stop being abstract

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

### Perplexity, and the thing the engine was missing

`forward_batch` returns logits for the **last** position only, because that is
all generation ever wants, and every intermediate row is computed and dropped.
Scoring text wants all of them. `forward_batch_all` runs the output head over
the whole batch as one matmul instead: about 430 tokens a second, against 114
for decoding — the same arithmetic, a quarter of the memory traffic. It is
checked against `nanograd`'s own logits at every position, which is the same
second-implementation argument the checkpoint test rests on.

### Testing

The API tests train a two-layer GPT on "the cat sat on the mat" *inside the
test*, save it, and then run a real prompt suite, a real benchmark and a real
perplexity job against it through the real scheduler. Three tests, under a
second, no network and no gigabytes.

Authentication got the mutation treatment: each of fifteen checks was deleted
in turn and the suite rerun. Two escaped the first pass — the `Admin` extractor
and the CSRF gate had been tested through the functions underneath them and not
through the extractors the routes are actually written against. Both have tests
now that fail when their check is removed.

And `/api/openapi.json` is generated from a table that a test compares against
the router, by reading the source of the files that register routes. A route
added and not documented fails the build. The first version of that test failed
on itself: it scanned its own source and found the string it searches *with*.

---

## Where to go next

Two tracks, one per job. They are independent: the learning track finishes the
story, the serving track is what the second half of the mission actually costs.
[`kvad-serve`](#crate-5-kvad-serve--the-server-and-the-web-ui) now sits under
both — it trains models and scores them, and it is where the batching work will
have to show that it worked.

### Finishing the story

**1. Train your own.** A character-level transformer, 10–30M parameters, on a
corpus you pick. Backprop through attention
([`attention.rs`](crates/nanograd/src/attention.rs)) and through LayerNorm and
RMSNorm ([`norm.rs`](crates/nanograd/src/norm.rs)), and the embedding
([`embedding.rs`](crates/nanograd/src/embedding.rs)), and the block that wires
them together with GELU and residual connections
([`block.rs`](crates/nanograd/src/block.rs)) are done, and so is the GPT that
stacks them ([`model.rs`](crates/nanograd/src/model.rs)): gradient-checked end
to end, and able to memorise a sequence, as are AdamW
([`optim.rs`](crates/nanograd/src/optim.rs)) and a training loop over a text
file with sampling ([`text.rs`](crates/nanograd/src/text.rs)). So this step
works, at 117 thousand parameters rather than 10 million, and the result is
saved as a GPT-2 checkpoint ([`checkpoint.rs`](crates/nanograd/src/checkpoint.rs))
from which the `kvad` engine computes the same logits. What stands between the
two sizes is speed, which is nine times what it was — about 190,000 characters
a second across the cores — and still a hand-written loop on a CPU: a 10M
parameter model is some eighty times the arithmetic per character. Nothing
stands between the two crates any more, nor between the two commands:
`kvad train --data FILE --name NAME` trains, `kvad run --model NAME` runs, and
`kvad ls` lists. Llama's SwiGLU and RoPE are not written. The gradient check from crate 1 is how
you will debug each one — extend `nanograd` (hard, most educational) or use
[`burn`](https://github.com/tracel-ai/burn).

Step 1 also has better tooling than it did. A run is a job on the server now,
with its loss curve drawn as it falls and the samples it writes at each
checkpoint beside it, and there is a perplexity number to put on the result —
so "did that corpus help?" is a measurement rather than a squint at some
generated text.

**2. Fine-tune with LoRA.** Freeze the model, train two small low-rank matrices
per weight matrix. This is what "custom model" means in practice, and unlike
full fine-tuning it fits on a laptop.

### Becoming a server

In rough dependency order. Two things are already done, and they are done for
different reasons.

Removing the [per-matmul floor](#the-floor-under-everything) was the one
blocking everything else: CPU decode went from 35 to 112 tok/s, and until it
was fixed every other optimisation was measuring rayon's scheduler rather than
the arithmetic. Step 5, the server, was built out of order — why is under its
own number below. The rest:

**3. A paged KV cache.** Today it is a `Vec<f32>` per layer that grows by
appending, which is 805 MB at Qwen's full context and cannot be shared between
sequences or reclaimed in pieces. Paging it into fixed blocks is what makes
several conversations fit in the memory of one, and it is a prerequisite for
everything below. Quantising it is a second, separate win.

This one has now been profiled rather than assumed, and the number that came
back was not the one this paragraph expected — see
[the KV cache](#the-kv-cache-and-a-fix-that-was-slower-than-the-bug). The
cheap half is done: the attention score loop was a single float accumulator
chain and is now eight, worth up to 1.37x of `attend`. The expensive half is
that grouped-query attention reads each cached key once per query head in its
group, so the traffic is several times the cache size, and past a few tens of
megabytes per layer that is the whole cost. Restructuring around the KV head
is the real work, and it is a different shape of change from anything above.

This one now has a gauge. The dashboard reports the cache twice over — what it
holds at this moment, and what the same conversation would cost at full context
— because the two are wildly different numbers and only one of them is obvious.
A 7B model at a 4096-token context is 960 KB *per token*: four gigabytes of
cache for one conversation, against 360 MB for all of SmolLM2's eight thousand.
That gap is the problem this step exists to solve, and you can now watch it
rather than read about it.

**4. Continuous batching.** Half the machinery exists: `forward_batch` already
runs many positions through one set of weights, which is the whole reason
prefill is fast. Serving needs the same thing across *different sequences* at
different positions, which means per-sequence positions in RoPE and attention,
and admitting new requests between steps instead of between batches.

The seam is written and waiting. `kvad-serve`'s scheduler owns the engine
thread and everything queues behind it; when batching lands it replaces the
inside of that queue and nothing above it changes. What is *not* written is the
measurement: the benchmark page runs one stream at a time, which is exactly the
thing batching does not improve. Generating concurrent load, and reporting
throughput against latency rather than tokens a second, is part of this step
rather than a separate one — and without it this repo would have no way to show
that the hardest change in it had worked.

**5. `kvad-serve`.** [Done](#crate-5-kvad-serve--the-server-and-the-web-ui) —
an OpenAI-compatible endpoint, and a web UI around it. It was meant to be last
for a good reason: a server around a single-sequence engine measures nothing
interesting *about throughput*. That turned out to be a claim about one number
rather than about the whole idea. Managing models, training them, scoring them
and watching the machine are all useful before batching exists, and building
the admin surface first means steps 3 and 4 land with somewhere to show their
work. What is still true is that this server will not serve two people at once
with any grace, and it says so rather than queuing quietly.

**6. Then the hardware.** Real CUDA kernels, flash attention, speculative
decoding. This is the stretch where legibility and speed start to fight, and
the point at which this README owes an honest account of the trade.

**And a kernel the benchmark page found, which is now written.** q4 decoded
faster than q8 and reached its first token *slower* — 41 ms against 30 — which
read like a cost of quantisation and was not. Both paths are integer and
neither dequantises; the batched `SMMLA` kernel was simply gated on
`Data::Q8`, and q4 fell back to a row at a time. Unpacking a row pair of
nibbles into `i8` hands it to the same kernel and
[roughly doubles q4 prefill](#batched-prefill-and-i8mm), which puts it level
with q8 while it still decodes faster. Worth writing down as a method rather
than a result: the number came from a page built to compare things, the
explanation that first suggested itself was wrong, and reading the dispatch
took less time than believing it would have cost.

### Worth reading alongside

- Karpathy, *Let's build GPT: from scratch, in code, spelled out*.
- *The Illustrated Transformer*, Jay Alammar — the diagrams.
- Vaswani et al., *Attention Is All You Need* (2017).
- Su et al., *RoFormer* (2021) — where RoPE comes from.
- Ainslie et al., *GQA* (2023) — grouped-query attention.

And for the serving track:

- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* (2023) — the vLLM paper, and step 4 above.
- Yu et al., *Orca* (2022) — where continuous batching comes from.
- Dao et al., *FlashAttention* (2022).

## Licence

MIT — see [LICENSE](LICENSE).
