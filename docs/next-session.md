# Instructions for the next session: after `kvad train` and the rate schedule

Written 2026-09-19, by the session that built `kvad train` and then measured
the learning-rate schedule, on top of the handoff written at commit `45a323e`
by the session before it. Read this whole
file before touching anything, then read the README's "Crate 1: `nervus`"
section, which records every measurement referred to here.

## What the owner wants

André asked at the start of the previous session: *"I want Kvad to be simple
to use in terms of training new models or add new training sets to existing
models, if that's even possible."* Everything below serves that sentence. Kvad
has two jobs — teach how LLMs work from scratch, and be a serious inference
engine — and the training track belongs to the first job. Hand-written
arithmetic is preferred over frameworks; `nervus` has zero dependencies and
must keep zero.

He directs the work one step at a time ("commit this, then do X next"). Do one
step, verify it, report, and leave it uncommitted with the paths listed. Commit
only when he says so.

## Where things stand

One tool, end to end, tested:

```bash
kvad train --data corpus.txt --name shakespeare      # small, 9 s
kvad train --data corpus.txt --name big --size large # 4.9M params, 19 min
kvad train --from shakespeare --data more.txt        # further training
kvad run --model shakespeare --prompt "ROMEO:"
kvad ls ; kvad use shakespeare ; kvad rm shakespeare ; kvad-tui shakespeare
```

| Piece | File | Commit |
|---|---|---|
| attention, norms, embedding, block, GPT, all with backward | `crates/nervus/src/{attention,norm,embedding,block,model}.rs` | up to `dece24d` |
| AdamW, outside the layers | `optim.rs` | `288c559` |
| tokeniser, windows, `train_step`, `generate`, `Replicas` | `text.rs` | `09a4770`, `45a323e` |
| the training loop as a library function, best-model saving | `text.rs::train` | `510b077` |
| warm-up, cosine decay, gradient clipping | `optim.rs`, `text.rs` | uncommitted |
| safetensors + GPT-2 layout, `tokenizer.json`, `json.rs` | `checkpoint.rs`, `text.rs` | `d10c21b` |
| engine loads a directory; stale-cache fix | `crates/llm/src/{weights,qcache,runtime,main}.rs` | `b8e70af` |
| the models home, names, `kvad train` | `crates/llm/src/{weights,hub,train,main}.rs` | `510b077` |
| cross-crate proof: same logits, same greedy text | `crates/llm/tests/nervus_checkpoint.rs` | `d10c21b`, `b8e70af` |
| the same journey by name | `crates/llm/tests/trained_models.rs` | `510b077` |

Numbers to measure against (M5 Pro, 6 fast + 12 slow cores, batch 16, medians
of 5 interleaved runs). Training the default 117,221-parameter model: 21,500
chars/s originally; 36,400 on one thread now; 143,900 on 6 threads; 190,200 on
16 with sampling on, 228,800 with it off. The `--size` presets, with sampling
off: small 228,800 chars/s, medium (835,685 params) 39,500, large (4,856,421)
7,000 — which is 19 minutes for the default 2000 steps. Validation loss on the
50 KB README bottoms near 1.9 around step 1250 and then overfits.

The defaults now include a rate schedule: warm-up over a tenth of the run, a
cosine down to a tenth of the peak, and gradient clipping at 1. Over eight
seeds at 2000 steps that took validation from 2.138 to 2.087 and the spread
from 0.052 to 0.041, and it made the learning rate nearly stop mattering
(0.003 to 0.02 all land between 2.087 and 2.120, where the flat arm had lost
0.45 by 0.01). Clipping costs 13% of throughput on the 117K model and nothing
measurable on the 4.9M one. The whole measurement is in the README under
"The seed that was worse than the others".

The repository is public at https://github.com/bisand/kvad, `master` tracking
`origin/master`, under the MIT licence. Push only when asked.

## The plans after this, in the order I would do them

**1. Speed, towards 10–30M parameters.** This is the one I would do next, and
it is now the only thing standing in the way of the learning track's stated
goal. The `--size large` measurement is the argument: 4.9M parameters is a
fifth of the target and already takes twenty minutes for 2000 steps, which is
not enough steps. That is about 85x the arithmetic per character against the
default. Two things found while measuring the schedule point the same way.
Gradient clipping's whole cost is one extra serial pass over the gradients on
the main thread, and it is 13% of the 117K model's throughput and nothing at
all on the 4.9M one — the serial part of a step is a large share of a small
model. And warm-up raised the usable learning rate threefold, which buys
better loss per step but not per second. Profile first
(`sample <pid> 8`); the last profile had the three matrix products at about
three quarters of a thread's time. (`tanh` was 15% before GELU stopped
computing it twice; it was not profiled again after.) Candidates: a
register-tiled GEMM like `crates/llm/src/tensor.rs` has; persistent worker
threads (spawning 16 a step was 3.6% of the main thread); reducing gradients
in parallel. A shared work queue was tried and measured no better than noise.

**2. Weight tying.** The model has its own output head with a bias; GPT-2
reuses the embedding table. Tying is about 30 lines (the table's gradient then
arrives from two places and adds) and makes the checkpoint a strict GPT-2 one
— no `lm_head.bias`, which no other reader knows about. It changes every loss
figure in the README, so regenerate them in the same step. Whether Python
`transformers` then loads the file is worth checking and is unverified.

**3. A BPE tokeniser.** Characters waste context, and they are also what makes
`--from` refuse a new text: a character tokeniser has no room for a character
it never saw, where BPE with byte fallback has. Training BPE is a teaching
chapter of its own, and `tokenizer.json` is already the right container: the
character tokeniser is written as BPE with an empty merge list.

**4. Llama's pieces in `nervus`.** `RmsNorm` exists with its backward.
SwiGLU and RoPE do not. With them a model can be saved in Llama layout, which
the GPU backend (`crates/gpu`) can run — it covers the Llama family, GPT-2 and
DeepSeek V2 as of v0.2.0.

**5. LoRA on real models** — `kvad tune qwen2.5-0.5b --data chats.jsonl`, then
`kvad chat --adapter`. This is what "custom model" means in practice, and it
is the largest item. It needs: `read_safetensors` to read BF16 and F16 (it
reads F32 only); loading a real checkpoint into `nervus` (tied heads, Llama
layout, GQA); item 4's backward passes; low-rank adapters with frozen base
weights; the chat template applied to training data. Work out the FLOPs
honestly before promising anything — training cost is about
6 x parameters x tokens, and half a billion parameters on hand-written CPU
kernels may simply be too slow without item 1.

## Loose ends found along the way, none fixed

New in this session:

- The TUI's model list still shows Hub models only. `hub::trained_models()`
  now exists and `kvad ls` uses it; `crates/tui/src/engine.rs` sends
  `Evt::Local(hub::local_models())` and would need the second list beside it.
  `kvad-tui NAME` already works, because it goes through `weights::model_id`.
- `kvad train` has no `--save DIR` escape hatch: a model either gets a name in
  the models home or is trained in place with `--from`. `train_text --save`
  still writes anywhere. Nobody has asked for the third case.
- Nothing tests that `train` actually uses `Replicas` when asked for threads.
  A mutation that routes every run through the single-threaded path is
  invisible to every test, because the arithmetic is the same either way; only
  a timing test would see it, and a timing test would be flaky.
- Half the run as warm-up measured 0.025 better than a tenth, at both 750 and
  2000 steps, but the sign flipped on two of eight seeds, so it is inside the
  spread. A tenth is the default because it is what everyone uses. Worth
  settling with more seeds by anyone who cares.
- Cosine decay tunes a run for exactly the `--steps` it was given: the rate
  reaches its floor at the last step. A run stopped early is a run that never
  decayed, and `--from` on a model trained with a schedule restarts the whole
  shape. Neither is wrong, but neither is written down anywhere but here.
- `--from` into a different corpus saves the best step *of that run*, starting
  the bar at infinity, because a loss on one text cannot be compared with a
  loss on another. So continuing a good model on a hard text will replace it
  with something worse. It is documented in `text::Training` and in the
  README, not prevented.

Still open from before:

- `llama.rs::param_count` omits the final norm and the q/k/v biases (GPT-2's
  was fixed in `d10c21b`).
- `crates/llm/src/quant.rs:125` has a broken doc link to `Weight::matvec`, so
  `cargo doc -p kvad` fails under `-D warnings`. Pre-existing clippy warnings
  in `simd.rs`, `tensor.rs`, `quant.rs`, `model/mod.rs`, `nervus/matrix.rs`,
  `nervus/nn.rs` and `main.rs`. This session added none.
- `kvad-tui DIR` builds and goes through the normal load path, but was never
  driven interactively.
- In the engine a prompt character outside the vocabulary is dropped silently
  by the `tokenizers` library (`kvad train` and `train_text` refuse it).
- `kvad info` claims not to download weights; `weights::fetch` downloads them.
- The quantised cache identifies a non-Hub file by size and modification time.
  A copy that preserves timestamps can defeat it; hashing was judged too slow
  for multi-gigabyte files.
- The README's loss curve and samples under "Training a GPT on a text file"
  come from the code before the speed work and were not regenerated.
- `scripts/get-text.sh` (tiny Shakespeare, 1.1 MB) was written and never run.
  Downloading is the owner's decision; ask first.

## How work is done here — these are not optional

**Measure before claiming.** Three performance claims in this repository were
wrong from single runs, and two more nearly were. Interleave A and B in one
loop, five runs each, report range and median. Check real idle
(`top -l 2 -n 0 | grep "CPU usage"`) and `ps -Ao pcpu,comm -r | head`, not the
load average, which lags by minutes. This session had to wait for an antivirus
scanner to stop using 89% of three cores before the preset table could be
measured. Profile before designing. When a number in the README turns out
wrong, correct it in place and say what it was.

**Break your own code on purpose.** After every feature: measure the error of
correct code, then mutate the code a dozen ways with a script, confirm a test
fails for each, restore, and set thresholds with margin on both sides. This
has caught a test blind to NaN (`f32::max` prefers anything to a NaN, so a
fold over it calls two lists of NaN identical), a JSON parser that hid a
writer bug, a division that Adam makes invisible to every learning test, and —
this session — a validation measurement that could drift between checkpoints
without any test noticing, two directories with the same last component that
would have been one model to the quantised-weight cache, a rate schedule the
loop could have built and then ignored, and a cosine that carried on round the
curve past the end of the run and came back up. A round trip
proves little: a writer and a reader that share a mistake agree perfectly.

**Check that a run ran.** Twice a whole batch of results was void: once the
mutated code did not compile, once a rebuilt binary was the old binary. The
mutation script reports "NO BUILD" separately from "SURVIVED" for that reason,
and it caught one this session. Confirm rows hold numbers before reading a
table.

**zsh does not split words.** `kill $PIDS` and `helper "--flag 1"` both fail
quietly. Pipe through `xargs`; pass flags as separate words. And `pgrep -f
foo` matches the shell command that is grepping for `foo`, so a
`until ! pgrep -f script.py; do sleep; done` loop waits on itself forever; wait
on a pid.

**Docs teach.** Every `nervus` file opens with "the one idea in this file",
says why before how, and reports what was measured. Test names are sentences.
Match the surrounding density. `nervus` keeps zero dependencies.

**The working tree may be shared** with another session — there was an
untracked `docs/ui-plan.md` in it this time that belonged to someone else.
Stage by explicit path, never `git add -A`. Do not branch or stash in the
shared tree. Commit on `master` only when asked, ending messages with
`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
