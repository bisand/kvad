# ai-llm

Learning how neural networks and language models work by building them in Rust,
from the arithmetic up.

Two crates, meant to be read in order:

| Crate | What it is | Dependencies |
|---|---|---|
| [`nanograd`](crates/nanograd) | A neural network and backpropagation, from scratch. Trains on MNIST. | **none** |
| [`gpt2`](crates/gpt2) | GPT-2 inference from scratch. Real HuggingFace weights in, text out. | model download, tokenizer, safetensors |

Neither uses an ML framework. Every matrix multiply, every derivative and every
attention head is code in this repo.

## Quick start

```bash
./scripts/get-mnist.sh
cargo test                                  # includes a gradient check
cargo run --release -p nanograd --bin train_mnist
cargo run --release -p gpt2 -- --prompt "The first time I saw the sea,"
```

Verified output on an M5 Pro:

```
epoch  1  loss 0.2085  test accuracy 96.32%  (2.1s)
epoch  5  loss 0.0347  test accuracy 97.89%  (10.6s)
epoch 10  loss 0.0072  test accuracy 97.98%  (21.0s)
```

```
The first time I saw the sea, it looked like a huge, large, black blob,
like a very large, extremely huge blob, and it was about 20 feet long."
[prefill 8 tokens in 0.14s, generated 60 tokens in 1.25s = 48.2 tok/s]
```

---

## Crate 1: `nanograd` — where the learning actually happens

Read in this order:

1. **[`matrix.rs`](crates/nanograd/src/matrix.rs)** — three matrix products.
   `A@B` for the forward pass, `Aᵀ@B` for weight gradients, `A@Bᵀ` for input
   gradients. That is the entire "tensor library".
2. **[`nn.rs`](crates/nanograd/src/nn.rs)** — the important one. Every layer
   implements two methods: `forward(x) -> y`, and `backward(dL/dy) -> dL/dx`.
   Chaining the second one backwards through the network *is* backpropagation.
3. **[`bin/train_mnist.rs`](crates/nanograd/src/bin/train_mnist.rs)** — the
   training loop, which is four lines: predict, score, blame, adjust.

### The test worth running first

```bash
cargo test -p nanograd analytic_gradient_matches_numerical -- --nocapture
```

It nudges a single weight by ±0.001, measures how the loss actually moves, and
checks that against what `backward()` claimed the gradient was. A subtly wrong
gradient — one missing transpose, one sign error — still trains, just badly, so
this is the only thing that will tell you your calculus is right.

### Things to try

- `--hidden 16` — how small can the hidden layer get before accuracy collapses?
- `--lr 0.5` — watch the loss diverge. Then `--lr 0.0001` and watch it crawl.
- `--momentum 0` — see how much of the convergence speed was momentum.
- Look at the last 10 epochs above: training loss keeps falling while test
  accuracy flatlines. That gap is overfitting, live.

---

## Crate 2: `gpt2` — the same ideas, at scale

```bash
cargo run --release -p gpt2 -- --prompt "Once upon a time" --temperature 0.9
cargo run --release -p gpt2 -- --greedy --prompt "1, 2, 3, 4, 5, 6,"
cargo run --release -p gpt2 -- --model openai-community/gpt2-medium --greedy \
    --prompt "The planets of the solar system, in order, are"
```

Weights download once into `~/.cache/huggingface` (124M params ≈ 500 MB).

Read in this order:

1. **[`tensor.rs`](crates/gpt2/src/tensor.rs)** — matmul, layernorm, GELU,
   softmax. Five functions; a transformer needs nothing else to run forwards.
2. **[`weights.rs`](crates/gpt2/src/weights.rs)** — the safetensors format,
   which is a length, a JSON header, and raw floats.
3. **[`model.rs`](crates/gpt2/src/model.rs)** — the architecture. Start with
   the diagram at the top of the file.
4. **[`sampler.rs`](crates/gpt2/src/sampler.rs)** — temperature, top-k, top-p.

### The four ideas in `model.rs`

- **The residual stream.** Blocks do `x = x + f(x)`, never `x = f(x)`. The
  vector `x` is a running total that every layer reads and adds to.
- **Attention is the only place tokens see each other.** The MLP — two thirds
  of the parameters — processes each position in total isolation.
- **The KV cache.** Generating token N without one means recomputing the whole
  prefix: O(N²) for the sequence instead of O(N). This is also why memory use
  climbs as you fill the context window (`Cache::max_bytes` prints it).
- **Every matmul is really a matrix-*vector* product**, because you generate
  one token at a time. That makes inference memory-bandwidth bound, which is
  why quantisation speeds it up and why GPU VRAM bandwidth is the number that
  matters.

### Verifying you got it right

`--greedy --prompt "1, 2, 3, 4, 5, 6,"` should continue `7, 8, 9, 10, ...`.
If any transpose or head-split is wrong, the output degrades to plausible-looking
noise rather than failing loudly. Counting is a sharp test.

---

## Where to go next

Roughly in order of difficulty. Each is a crate you can add to this workspace.

**3. Make inference fast.**
Quantise the weights to int8 or int4 and dequantise on the fly. GPT-2 in f32 is
500 MB; at int4 it is ~60 MB and noticeably faster, because you are moving a
fraction of the bytes. Then process the whole prompt as a batch instead of token
by token — you will need an explicit causal mask, which is the classic first bug.

**4. Move to the GPU with [`candle`](https://github.com/huggingface/candle).**
HuggingFace's Rust ML framework, with a Metal backend for your Mac. Port the
model and compare — you will recognise every operation, because you wrote them
all by hand first. This is also the point where modern models (Llama, Qwen,
Mistral) become practical; your 48 GB of unified memory will hold a quantised
30B model.

**5. Train your own language model.**
A character-level transformer, 10–30M parameters, on a corpus you choose. You
need backprop through attention and layernorm, plus the Adam optimiser. The
gradient check from crate 1 is how you will debug it. Either extend `nanograd`
(hard, and the most educational thing on this list) or use
[`burn`](https://github.com/tracel-ai/burn), which has autodiff and a Metal
backend. Hours of training on your hardware, not days.

**6. Fine-tune with LoRA.**
Freeze a pretrained model and train two small low-rank matrices per weight
matrix instead. This is what "custom model" means in practice — and it is
tractable on a laptop in a way that full fine-tuning is not.

### Worth reading alongside

- Karpathy, *Let's build GPT: from scratch, in code, spelled out* — the video
  companion to crate 2.
- *The Illustrated Transformer*, Jay Alammar — the diagrams.
- Vaswani et al., *Attention Is All You Need* (2017) — short, and readable once
  you have implemented it.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019) —
  the GPT-2 paper, i.e. the model in crate 2.
