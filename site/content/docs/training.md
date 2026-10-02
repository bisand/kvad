---
title: Train your own
description: Train a small GPT from a text file, on the CPU, and run it in the same engine as everything else.
---

Kvad can train a model of its own from scratch: a character-level GPT, on any
plain text you give it, by a training loop that is written out in the
repository with no framework underneath. It will not replace a model you
download. It is here because watching a loss curve fall on text you chose is
the fastest way to understand what the bigger models are.

## From a text file

```bash
kvad train --data notes.txt --name notes
kvad run --model notes --prompt "## "
kvad ls
```

A trained model is saved as a GPT-2 checkpoint, appears in `kvad ls` beside the
downloaded ones, and loads on the same engine.

## Getting text

`crawl` reads a documentation site into one text file. It follows links under
the starting address's own directory, obeys `robots.txt`, and waits between
requests.

```bash
kvad crawl https://doc.rust-lang.org/book/ --out rust-book.txt
kvad train --data rust-book.txt --name rustbook --size medium --steps 4000
```

## Options

| Option | Default | |
|---|---|---|
| `--size NAME` | `small` | The model's shape: `small`, `medium` or `large`. |
| `--steps N` | 2000 | |
| `--batch N` | 16 | Windows per step. |
| `--lr F` | 0.003 | The peak learning rate. |
| `--warmup N` | a tenth of the run | Steps spent climbing to it. |
| `--eval-every N` | 250 | Steps between checkpoints. |
| `--threads N` | every core | Replicas to split each batch across. |
| `--sample N` | | Characters to write at each checkpoint. |
| `--from MODEL` | | Train an existing model further. |

`--from` has two limits. The character vocabulary is fixed at first training,
so text with a character the model never saw is refused. And the optimiser's
state is not saved, so a resumed run starts AdamW's running averages again.

## On the server

With a server running, a training run is a job. The web UI's Training page
draws its loss curve as it falls, with the samples it writes at each
checkpoint beside it, and the Datasets page uploads text or runs a crawl.

```bash
kvad datasets add notes.txt
kvad train --dataset notes --name notes
kvad jobs                  # ls
kvad jobs watch 4
```

Chat still works during a training run, at roughly half speed. That was
measured, and the banner in the UI says "slower" rather than "unavailable".

## Did it help?

Perplexity on held-out text puts a number on a trained model, so "did that
corpus help?" is a measurement rather than a squint at generated text.

```bash
kvad evals perplexity notes-heldout notes notes-v2
```

[The first crate](/docs/internals/nervus/) explains the training loop itself,
from backpropagation up.
