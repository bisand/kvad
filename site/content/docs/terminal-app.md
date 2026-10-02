---
title: The terminal app
description: Browse, download, switch backend and chat, without leaving the terminal.
---

```bash
kvad-tui
```

`kvad-tui` is on macOS only. It loads models in its own process, so it needs
no server; if the background service is holding a large model, unload it first
or the two will compete for memory.

## Keys

| Key | |
|---|---|
| `/` | Search Hugging Face. |
| `↑` `↓` | Select a model. |
| `enter` | Download it if needed, and load it. |
| `p` | Cycle the backend. |
| `d` | Delete the selected model. |
| `tab` | Switch between the model list and the chat. |
| `esc` | Interrupt a generation. |

## Trying the backends

`p` cycles all six backends: `cpu f32`, `cpu q8`, `cpu q4`, `gpu bf16`,
`gpu q8`, `gpu q4`. It is the quickest way to feel the trade-offs. Load a
model at f32 and ask it something arithmetic, reload at q4 and ask again, then
reload on the GPU and watch the token rate.

## The status bar

After each reply the status bar reports the decode rate and how much of the
conversation came from the cache:

```text
[33.3 tok/s · 96 cached]
```

The chat keeps its KV cache across turns. Each turn the conversation is
re-encoded, compared with what is cached, and only the new message is
prefilled. Without that, turn *N* would re-read the whole transcript.
