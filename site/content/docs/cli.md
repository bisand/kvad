---
title: Command line
description: Every kvad command, grouped by what it is for.
---

```text
kvad <command> [options]
```

`kvad --help` prints the same list. Most commands with subcommands print
their own usage with `--help`.

## Models

| Command | |
|---|---|
| `kvad search QUERY` | Find models on Hugging Face, and say which this build can run. |
| `kvad pull REPO` | Download a model. |
| `kvad ls` | List downloaded and trained models. |
| `kvad use MODEL` | Set the default model. |
| `kvad rm MODEL` | Delete a downloaded or trained model. |
| `kvad info [MODEL]` | Show a model's config without downloading weights. |
| `kvad arch` | List the architectures this build can run. |
| `kvad cache [REPO\|clear]` | List or delete pre-quantised weight files. |

## Generating

| Command | |
|---|---|
| `kvad run --prompt TEXT` | One answer. |
| `kvad chat` | A conversation. |
| `kvad images make PROMPT` | A picture. Also `ls` and `rm ID`. |
| `kvad images edit PROMPT --image PICTURE` | A picture made from one, with `--strength` and `--mask`. |
| `kvad videos make PROMPT` | A clip. Also `ls`, `show`, `watch`, `get`, `rm`. |
| `kvad tokenize TEXT` | How the model splits a text. |
| `kvad cancel` | Stop whatever is generating. |

## Training

| Command | |
|---|---|
| `kvad train --data FILE --name NAME` | Train a model from a text file. |
| `kvad crawl URL` | Read a documentation site into a text file. |
| `kvad datasets` | `ls`, `add FILE`, `crawl URL`, `show`, `check`, `search`, `rm`. |
| `kvad-gpu tune --data DIR --name NAME` | Train an SDXL LoRA on a folder of pictures. |

## The server

| Command | |
|---|---|
| `kvad serve [...]` | Run the HTTP server and web UI in the foreground. |
| `kvad service ...` | Run it at login: `status`, `start`, `stop`, `restart`, `logs`, `install`, `uninstall`. |
| `kvad ps` | The models in memory, and what memory is left. |
| `kvad load MODEL [--backend ID]` | Put a model in memory. |
| `kvad unload [ID]` | Take one out, or all of them. |
| `kvad conversations` | `ls`, `show ID`, `edit ID`, `rm ID`. |
| `kvad jobs` | `ls`, `show ID`, `watch ID`, `cancel ID`. |
| `kvad evals` | Prompt suites and perplexity. See [Benchmarks and evals](/docs/benchmarks/). |
| `kvad bench` | Benchmarks. |
| `kvad metrics` | The machine, recent requests, and the log. |
| `kvad api [METHOD PATH [JSON]]` | Any route, raw. With no arguments, the list of them. |

## Accounts

| Command | |
|---|---|
| `kvad auth` | `status`, `login [--key]`, `logout`, `setup TOKEN`, `password`. |
| `kvad users` | `ls`, `add NAME`, `edit ID`, `rm ID`. |
| `kvad sessions` | `ls`, `rm ID`. |
| `kvad keys` | `ls`, `add NAME`, `rm ID`. |

## Where a command runs

| Option | |
|---|---|
| `--remote URL` | Send the command to the server at URL. |
| `--local` | Run it in this process, even with a server running. |
| `--json` | Print what the server sent, as JSON. A streaming command prints one event a line. |
| `-y`, `--yes` | Answer yes to the question a command would ask. |

Without `--remote` or `--local`, the order is `KVAD_URL`, then `url` under
`[client]` in `kvad.toml`, then this machine's service if it answers, then the
process itself. The first line of output says which, and why.

`--json` makes every command composable:

```bash
kvad ps --json | jq '.residents[].id'
```

## Generation options

| Option | Default | |
|---|---|---|
| `--model MODEL` | the default model | A name trained here, a directory, or a Hugging Face repository. |
| `--prompt TEXT` | | The prompt for `run`. |
| `--system TEXT` | | The system prompt for `chat`. |
| `--max-tokens N` | 256 | |
| `--temperature F` | 0.7 | 0 is greedy. |
| `--top-k N` | 40 | |
| `--top-p F` | 0.95 | |
| `--seed N` | 7 | |
| `--greedy` | | `--temperature 0`. |
| `--quant f32\|q8\|q4` | f32 | Quantise weights on load. On a server, the CPU backend at that precision. |
| `--backend ID` | | On a server: the backend to load on, such as `gpu-q8`. |
| `--raw` | | `run` on a server: continue the prompt, with no chat template. |
| `--save` | | `chat` on a server: keep the conversation there. |
| `--conversation ID` | | `chat` on a server: carry on with a kept one. |

## The other binaries

`kvad-tui` takes no arguments. See [The terminal app](/docs/terminal-app/).

`kvad-gpu` runs the GPU engine without a server:

```text
kvad-gpu <run|chat> [--model REPO] [--device metal|cpu] [--dtype bf16|f16|f32]
                    [--quant none|q8|q4|q4k|q6k] [--prompt TEXT] …
kvad-gpu tune --help
```
