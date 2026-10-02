---
title: Troubleshooting
description: The failures people actually hit, and what each one means.
---

## `kvad: command not found`

`~/.local/bin` is not on your `PATH`. Run the installer again and say yes to
the question, or add it yourself:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

## The service is running and nothing answers

```bash
kvad service status
kvad service logs -n 50
```

`status` compares what the service manager says with what `/api/health` says.
If they disagree, the log says why. The usual reasons are a port that
something else took, or a `bind` in `kvad.toml` that is not loopback while
`auth.mode` is `none`.

## A model will not load: not enough memory

The server refuses a load that does not fit beside what is already in memory,
and the message names what is holding it.

```bash
kvad ps
kvad unload
```

Then load at a smaller precision. `gpu-q4` or `cpu-q4` is about half of q8.

## A picture or video fails with a GPU memory error

The decode at the end is the largest step. Unload other models first with
`kvad unload`, close anything else using the GPU, or make a smaller picture.
Other applications count: a virtual machine or a second copy of a model in
`kvad-tui` takes from the same unified memory.

## `404` for a model over the API

The model is not on disk. The server loads a model that is on this machine and
never downloads one because a request named it.

```bash
kvad pull Qwen/Qwen2.5-7B-Instruct
```

## `400` when sending tools

The model's chat template has no place for tools, so the server refuses
rather than let it answer in prose. `kvad ls` marks the models that can with
`tools`. See [Tool calls and agents](/docs/agents/).

## A reasoning model answers nothing

It spent its whole budget thinking. Raise `--max-tokens`, or `max_tokens` in
the request.

## The first load of a model is slow

The first load at q8 or q4 quantises the weights and writes them to
`~/.cache/kvad/quant`. Later loads map that file and are much faster.

## The first picture after a reboot is slow

The weights are being read from disk for the first time. The second is the
real speed.

## Downloads are refused or rate-limited

For a gated model, accept its terms on Hugging Face and set a token:

```bash
export HF_TOKEN=hf_…
kvad pull black-forest-labs/FLUX.1-schnell
```

## Chat is slow while something else runs

The engine runs one generation at a time, and a training run takes most of the
cores. `kvad jobs` shows what is running and `kvad jobs cancel ID` stops it.

## macOS will not open the binary

A tarball downloaded with a browser is quarantined. Install with the `curl`
command instead, or clear the flag:

```bash
xattr -d com.apple.quarantine ~/.local/bin/kvad*
```

## Something else

`kvad metrics` and `kvad service logs` are the first places to look. If it
looks like a bug, [open an issue](https://github.com/bisand/kvad/issues) with
the output of `kvad service status` and the relevant part of the log.
