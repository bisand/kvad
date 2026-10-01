[← Back to the README](../README.md)

# Crate 4: `kvad-tui` — the app

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
