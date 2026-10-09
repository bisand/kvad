---
title: Command line
description: Every kvad command, grouped by what it is for, with its built-in help and Tab completion.
---

```text
kvad <command> [words] [options]
```

## Help

`kvad help`, or `kvad` with nothing after it, prints every command in a line
each. `kvad help COMMAND` says what one command takes: its usage, each option
with what it does, and examples. `kvad COMMAND --help` is the same thing.

```bash
kvad help
kvad help videos
kvad images --help
```

A command or an option that does not exist is told the nearest one that does:

```text
$ kvad chta
`kvad chta` is not a command.
Did you mean `kvad chat`?
`kvad help` lists them.

$ kvad run --promt x
--promt is not an option of `kvad run`.
Did you mean --prompt?
`kvad run --help` says what it takes.
```

Help that was asked for goes to stdout, so `kvad help train | less` works.

## Tab completion

`kvad` completes at a Tab in zsh, bash and fish. The
[installer](/docs/install/) offers to set it up, and so does:

```bash
kvad completions install
```

That sets it up for your login shell; open a new terminal afterwards. Tab then
completes commands and their options, and what only the machine knows:

| You type | Tab offers |
|---|---|
| `kvad ` | Every command, with what it does. |
| `kvad run --model ` | The language models on disk. |
| `kvad images make --model ` | The image models. `videos make` gets the video models. |
| `kvad images make --lora ` | The LoRAs. |
| `kvad unload ` | The models in the server's memory. |
| `kvad load MODEL --backend ` | The server's backends, such as `gpu-q8`. |
| `kvad jobs show ` | The jobs, each with what it is. The same for conversations, pictures, clips and datasets. |
| `kvad videos make --` | Every option a video takes. |
| `kvad train --data ` | Files, as your shell completes them. |

zsh and fish show a description beside each candidate:

```text
$ kvad run --quant <Tab>
f32  -- weights as they were published
q8   -- 8 bits a weight: a quarter of the memory
q4   -- 4 bits a weight: an eighth, and a little worse
```

With a server running the answers are the server's, as the commands' are, and
`--remote URL` earlier on the line asks that server instead. A server that
does not answer within two seconds is given up on, and Tab offers nothing.

| Command | |
|---|---|
| `kvad completions install [SHELL...]` | Set it up for your login shell, or for the shells named: `zsh`, `bash`, `fish`. |
| `kvad completions status` | Where it is set up, and where it would go. |
| `kvad completions uninstall` | Take it out again, of every shell. |
| `kvad completions zsh` | Print a shell's script, to place yourself. |

For zsh and bash, `install` adds one line to the end of `~/.zshrc`, or
`~/.bashrc` (`~/.bash_profile` on a Mac). For fish it writes
`~/.config/fish/completions/kvad.fish`. `uninstall` removes what `install`
wrote and nothing else. The shell asks the installed `kvad` at every Tab, so
an upgrade needs nothing done again.

The fish script has not been run on a machine with fish; zsh and bash have.

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
| `kvad datasets` | `ls`, `add FILE` or `add DIR` (a folder of pictures), `crawl URL`, `show`, `check`, `search`, `put ID FILE...`, `get ID FILE`, `rm`. |
| `kvad tune --data DIR --name NAME` | Train an SDXL LoRA on a folder of pictures, as a job on the server. `kvad tune options` says what it takes. |
| `kvad-gpu tune --data DIR --name NAME` | The same training in this terminal, with no server. |

## The server

| Command | |
|---|---|
| `kvad serve [...]` | Run the HTTP server and web UI in the foreground. |
| `kvad service ...` | Run it at login: `status`, `start`, `stop`, `restart`, `logs`, `install`, `uninstall`. |
| `kvad ps` | The models in memory, and what memory is left. |
| `kvad load MODEL [--backend ID]` | Put a model in memory. |
| `kvad unload [ID]` | Take one out, or all of them. |
| `kvad conversations` | `ls`, `show ID`, `edit ID`, `rm ID`. |
| `kvad jobs` | `ls`, `show ID`, `watch ID`, `cancel ID`, `pictures ID` (the samples a LoRA run drew). |
| `kvad evals` | Prompt suites and perplexity. See [Benchmarks and evals](/docs/benchmarks/). |
| `kvad bench` | Benchmarks. |
| `kvad metrics` | The machine, recent requests, and the log. |
| `kvad api [METHOD PATH [JSON]]` | Any route, raw. With no arguments, the list of them. |

## Accounts

| Command | |
|---|---|
| `kvad auth` | `status`, `login [--key]`, `logout`, `setup TOKEN`, `password`. |
| `kvad users` | `ls`, `add NAME`, `edit ID`, `rm ID`. |
| `kvad sessions` | `ls`, `rm HASH`. |
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
