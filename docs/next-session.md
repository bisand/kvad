# Instructions for the next session: `kvad train`, and the plans after it

Written 2026-09-19 at commit `45a323e`, by the session that built the training
track. Read this whole file before touching anything, then read the README's
"Crate 1: `nanograd`" section, which records every measurement referred to here.

## What the owner wants

André asked at the start: *"I want Kvad to be simple to use in terms of training
new models or add new training sets to existing models, if that's even
possible."* Everything below serves that sentence. Kvad has two jobs — teach how
LLMs work from scratch, and be a serious inference engine — and the training
track belongs to the first job. Hand-written arithmetic is preferred over
frameworks; `nanograd` has zero dependencies and must keep zero.

He directs the work one step at a time ("commit this, then do X next"). Do one
step, verify it, report, and leave it uncommitted with the paths listed. Commit
only when he says so.

## Where things stand

The loop is closed at toy scale. All of this works and is tested:

```bash
alias train_text="cargo run --release -p nanograd --bin train_text --"
train_text --data README.md --save out/readme      # 2000 steps, about 11 s
kvad run --model out/readme --greedy --prompt "## "
kvad use out/readme ; kvad-tui out/readme
```

| Piece | File | Commit |
|---|---|---|
| attention, norms, embedding, block, GPT, all with backward | `crates/nanograd/src/{attention,norm,embedding,block,model}.rs` | up to `dece24d` |
| AdamW, outside the layers | `optim.rs` | `288c559` |
| tokeniser, windows, `train_step`, `generate`, `Replicas` | `text.rs` | `09a4770`, `45a323e` |
| safetensors + GPT-2 layout, `tokenizer.json`, `json.rs` | `checkpoint.rs`, `text.rs` | `d10c21b` |
| engine loads a directory; stale-cache fix | `crates/llm/src/{weights,qcache,runtime,main}.rs` | `b8e70af` |
| cross-crate proof: same logits, same greedy text | `crates/llm/tests/nanograd_checkpoint.rs` | `d10c21b`, `b8e70af` |

Numbers to measure against (M5 Pro, 6 fast + 12 slow cores, default model of
117,221 parameters, batch 16, context 64, medians of 5 interleaved runs):
21,500 chars/s originally; 36,400 on one thread now; 143,900 on 6 threads;
190,200 on 16. Validation loss on the 50 KB README bottoms near 1.9 around step
1250 and then overfits.

The repository is public at https://github.com/bisand/kvad, `master` tracking
`origin/master`. Push only when asked. It has no LICENSE file.

## Task 1: `kvad train`

Today training is `cargo run -p nanograd --bin train_text` and running is
`kvad`. Make it one tool. A proposal, to be adjusted by what you find:

```bash
kvad train --data corpus.txt --name shakespeare          # new model
kvad train --from shakespeare --data more.txt            # train an existing one further
kvad run --model shakespeare --prompt "ROMEO:"
kvad ls                                                   # lists trained models too
```

1. **Move the loop into the library.** The training loop lives in
   `crates/nanograd/src/bin/train_text.rs::main`. Lift it into `nanograd::text`
   as a function that takes a config and a progress callback, so the binary and
   `kvad train` share it. `crates/llm` already depends on `nanograd`.
2. **Give trained models a home and a name.** `hub::local_models()` only walks
   the HuggingFace cache, so `kvad ls` and the TUI's list cannot see a trained
   model; it is reachable only by path. Put them under something like
   `$XDG_DATA_HOME/kvad/models/<name>/` and resolve a bare name there before
   treating it as a Hub repo. `weights::is_local` / `model_id` /
   `ModelFiles::from_dir` are the hooks; keep "an existing directory wins".
3. **`--from`** is the owner's "add new training sets to existing models". Be
   honest about two limits in the help text: the character vocabulary is fixed
   at first training (a loaded model refuses text with an unseen character, and
   says which), and optimiser state is not saved (measured cost: at most 0.06 of
   training loss over 50 steps, gone by 100).
4. **Keep the best model, not the last.** The run visibly overfits, and `--save`
   writes whatever step it ended on. Save when validation loss improves. This
   is early stopping and deserves a paragraph in the docs.
5. **Size presets need measuring, not guessing.** If you offer `--size`, time
   each preset and print an honest estimate before starting.
6. **Test it the way the journey test does**
   (`a_trained_model_runs_from_its_directory`): train tiny, save, load through
   `Llm::load_with` by name, generate, compare with `nanograd`'s own greedy text.

## The plans after that, in the order I would do them

1. **Learning-rate warm-up, cosine decay, gradient clipping.** One AdamW seed
   in four was clearly worse (validation 2.47 against about 2.19 for the others). Warm-up is
   the usual fix. That is a hypothesis; it has not been tried. Measure over
   several seeds, both arms at their best learning rate.
2. **Weight tying.** The model has its own output head with a bias; GPT-2 reuses
   the embedding table. Tying is about 30 lines (the table's gradient then
   arrives from two places and adds) and makes the checkpoint a strict GPT-2
   one — no `lm_head.bias`, which no other reader knows about. It changes every
   loss figure in the README, so regenerate them in the same step. Whether
   Python `transformers` then loads the file is worth checking and is unverified.
3. **Speed, towards 10–30M parameters.** That is about 85x the arithmetic per
   character. Profile first (`sample <pid> 8`); the last profile had the three
   matrix products at about three quarters of a thread's time. (`tanh` was 15%
   before GELU stopped computing it twice; it was not profiled again after.) Candidates: a
   register-tiled GEMM like `crates/llm/src/tensor.rs` has; persistent worker
   threads (spawning 16 a step was 3.6% of the main thread); reducing gradients
   in parallel. A shared work queue was tried and measured no better than noise.
4. **A BPE tokeniser.** Characters waste context. Training BPE is a teaching
   chapter of its own, and `tokenizer.json` is already the right container: the
   character tokeniser is written as BPE with an empty merge list.
5. **Llama's pieces in `nanograd`.** `RmsNorm` exists with its backward. SwiGLU
   and RoPE do not. With them a model can be saved in Llama layout, which the
   GPU backend (`crates/gpu`, Llama only) can run.
6. **LoRA on real models** — `kvad tune qwen2.5-0.5b --data chats.jsonl`, then
   `kvad chat --adapter`. This is what "custom model" means in practice, and it
   is the largest item. It needs: `read_safetensors` to read BF16 and F16 (it
   reads F32 only); loading a real checkpoint into `nanograd` (tied heads,
   Llama layout, GQA); item 5's backward passes; low-rank adapters with frozen
   base weights; the chat template applied to training data. Work out the
   FLOPs honestly before promising anything — training cost is about
   6 x parameters x tokens, and half a billion parameters on hand-written CPU
   kernels may simply be too slow without item 3.

## Loose ends found along the way, none fixed

- `llama.rs::param_count` omits the final norm and the q/k/v biases (GPT-2's
  was fixed in `d10c21b`).
- `crates/llm/src/quant.rs:125` has a broken doc link to `Weight::matvec`, so
  `cargo doc -p kvad` fails under `-D warnings`. Pre-existing clippy warnings
  in `simd.rs`, `tensor.rs`, `quant.rs`, `model/mod.rs`, `nanograd/matrix.rs`.
- `kvad-tui DIR` builds and goes through the normal load path, but was never
  driven interactively. The TUI's model list shows Hub models only.
- In the engine a prompt character outside the vocabulary is dropped silently
  by the `tokenizers` library (`train_text` refuses it instead).
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
wrong from single runs, and two more nearly were in the last session. Interleave
A and B in one loop, five runs each, report range and median. Check real idle
(`top -l 2 -n 0 | grep "CPU usage"`) and `ps -Ao pcpu,comm -r | head`, not the
load average, which lags by minutes. Profile before designing. When a number in
the README turns out wrong, correct it in place and say what it was.

**Break your own code on purpose.** After every feature: measure the error of
correct code, then mutate the code a dozen ways with a script, confirm a test
fails for each, restore, and set thresholds with margin on both sides. This has
caught a test blind to NaN (`f32::max` prefers anything to a NaN, so a fold
over it calls two lists of NaN identical), a JSON parser that hid a writer bug,
and a division that Adam makes invisible to every learning test. A round trip
proves little: a writer and a reader that share a mistake agree perfectly.

**Check that a run ran.** Twice a whole batch of results was void: once the
mutated code did not compile, once a rebuilt binary was the old binary. Confirm
rows hold numbers and `cmp` the binaries before reading a table.

**zsh does not split words.** `kill $PIDS` and `helper "--flag 1"` both fail
quietly. Pipe through `xargs`; pass flags as separate words.

**Docs teach.** Every `nanograd` file opens with "the one idea in this file",
says why before how, and reports what was measured. Test names are sentences.
Match the surrounding density. `nanograd` keeps zero dependencies.

**The working tree may be shared** with another session. Stage by explicit
path, never `git add -A`. Do not branch or stash in the shared tree. Commit on
`master` only when asked, ending messages with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
