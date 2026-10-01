[← Back to the README](../README.md)

# Where to go next

Two tracks, one per job. They are independent: the learning track finishes the
story, the serving track is what the second half of the mission actually costs.
[`kvad-serve`](serve.md#crate-5-kvad-serve--the-server-and-the-web-ui) now sits under
both — it trains models and scores them, and it is where the batching work will
have to show that it worked.

## Finishing the story

**1. Train your own.** A character-level transformer, 10–30M parameters, on a
corpus you pick. Backprop through attention
([`attention.rs`](../crates/nervus/src/attention.rs)) and through LayerNorm and
RMSNorm ([`norm.rs`](../crates/nervus/src/norm.rs)), and the embedding
([`embedding.rs`](../crates/nervus/src/embedding.rs)), and the block that wires
them together with GELU and residual connections
([`block.rs`](../crates/nervus/src/block.rs)) are done, and so is the GPT that
stacks them ([`model.rs`](../crates/nervus/src/model.rs)): gradient-checked end
to end, and able to memorise a sequence, as are AdamW
([`optim.rs`](../crates/nervus/src/optim.rs)) and a training loop over a text
file with sampling ([`text.rs`](../crates/nervus/src/text.rs)). So this step
works, at 117 thousand parameters rather than 10 million, and the result is
saved as a GPT-2 checkpoint ([`checkpoint.rs`](../crates/nervus/src/checkpoint.rs))
from which the `kvad` engine computes the same logits. What stands between the
two sizes is speed, which is nine times what it was — about 190,000 characters
a second across the cores — and still a hand-written loop on a CPU: a 10M
parameter model is some eighty times the arithmetic per character. Nothing
stands between the two crates any more, nor between the two commands:
`kvad train --data FILE --name NAME` trains, `kvad run --model NAME` runs, and
`kvad ls` lists. Llama's SwiGLU and RoPE are not written. The gradient check from crate 1 is how
you will debug each one — extend `nervus` (hard, most educational) or use
[`burn`](https://github.com/tracel-ai/burn).

Step 1 also has better tooling than it did. A run is a job on the server now,
with its loss curve drawn as it falls and the samples it writes at each
checkpoint beside it, and there is a perplexity number to put on the result —
so "did that corpus help?" is a measurement rather than a squint at some
generated text.

**2. Fine-tune with LoRA.** Freeze the model, train two small low-rank matrices
per weight matrix. This is what "custom model" means in practice, and unlike
full fine-tuning it fits on a laptop.

## Becoming a server

In rough dependency order. Two things are already done, and they are done for
different reasons.

Removing the [per-matmul floor](engine.md#the-floor-under-everything) was the one
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
[the KV cache](engine.md#the-kv-cache-and-a-fix-that-was-slower-than-the-bug). The
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

**5. `kvad-serve`.** [Done](serve.md#crate-5-kvad-serve--the-server-and-the-web-ui) —
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
[roughly doubles q4 prefill](engine.md#batched-prefill-and-i8mm), which puts it level
with q8 while it still decodes faster. Worth writing down as a method rather
than a result: the number came from a page built to compare things, the
explanation that first suggested itself was wrong, and reading the dispatch
took less time than believing it would have cost.

## Worth reading alongside

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
