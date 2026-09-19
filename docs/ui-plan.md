# Plan: a web UI for Kvad, and the server under it

Written 2026-09-19 against the working tree after `5e551d9`, from a reading of
`crates/llm/src/{runtime,train,weights,hub}.rs`, the TUI's actor in
`crates/tui/src/engine.rs`, and the README's "Becoming a server" section.
Nothing here is built. Where something is a guess it says so.

## What the owner asked for

A UI to manage and monitor every aspect of the server: download models, train
them, watch performance, chat with local models, test them. A Rust backend and a
lightweight frontend. An admin-dashboard look. Configurable auth (user/password,
OAuth2, Basic). SQLite for persistence. Light and dark mode with daisyUI, the
default dark theme being `dim`.

## The stack, and why

**Frontend: Svelte 5 + Vite as a plain SPA (not SvelteKit), Tailwind 4 +
daisyUI 5.** The UI is mostly live state — tokens streaming in, loss curves
growing, progress bars, gauges — and Svelte compiles that to about 40–60 KB with
no runtime framework. HTMX + Askama would be lighter still, but streaming chat
and live charts are where it gets awkward, and those are the core of this app.
SvelteKit adds routing and SSR machinery that a Rust backend does not need; a
small client-side router is enough.

**Backend: a new `kvad-serve` crate, axum + tokio.** It is the name the README
already reserves. The hand-written rule covers the engine, not the plumbing;
`hf-hub`, `tokenizers` and `ratatui` are used on the same basis.

**SQLite, via `rusqlite` with the `bundled` feature.**

- Synchronous, no compile-time query macros, no async driver. Put it behind one
  DB thread, or use `spawn_blocking`.
- WAL mode. Numbered `.sql` migrations embedded in the binary.
- The filesystem stays the source of truth for models (the HF cache and
  `models_dir()`). The DB never claims a model exists; it only stores metadata
  about what the scan finds. That keeps `kvad ls`, the TUI and the web UI
  consistent with each other.

**Live transport: SSE only, no WebSockets.** Token streams, job progress and
metrics all flow from server to client, and the OpenAI API is SSE anyway.
Cancelling a generation or a job is a plain `DELETE`.

**Charts: uPlot.** About 45 KB and built for streaming time series. Series
colours are read from daisyUI's CSS variables, so charts follow the theme.

**Theming** is one block of CSS plus a toggle:

```css
@plugin "daisyui" { themes: light --default, dim --prefersdark; }
```

The toggle writes `data-theme` to `localStorage`, and a three-line inline script
in `index.html` applies it before first paint, so there is no flash.

**Shipping:** `rust-embed` bakes `web/dist` into the binary, so there is one
file to deploy. `build.rs` does not call npm, so `cargo build` never needs Node;
if `dist` is missing the server serves a stub page saying how to build it. In
development Vite proxies `/api` and `/v1` to axum.

## The constraint that shapes everything

The engine runs one loaded model and one generation at a time, and training uses
the same cores. The README says a server around that "measures nothing
interesting". That is true for throughput, but an admin UI is useful well before
batching exists. So:

- **One `Scheduler` owns the engine thread.** It is the TUI's `Cmd`/`Evt` actor,
  lifted out and shared. Requests queue first-in, first-out, and queue depth is
  a visible metric. When continuous batching lands it replaces the inside of the
  scheduler and nothing above it changes.
- **Training takes a compute lease.** Whether chat stays usable during training
  (for example with training capped to N threads) has to be measured, not
  assumed — interleaved, five runs, median and range. Until it is measured the
  UI says "training in progress, inference queued".
- **The UI's chat calls `/v1/chat/completions` itself.** One code path,
  exercised by our own UI. Per-message stats (prefill tok/s, decode tok/s,
  cached tokens) go in an extension field.

## Pages

| Page | What it does |
|---|---|
| **Dashboard** | Loaded model and backend, tok/s and time-to-first-token sparklines, queue depth, RSS and KV-cache bytes, running jobs, disk used by models and the quantised-weights cache |
| **Models** | Local (downloaded and trained) and Hub search with the existing `runnable()`/`blocker()` verdicts, pull with progress, delete, set active, load/unload with a precision/backend picker, quantised-cache management |
| **Chat** | Persisted conversations, system prompt, sampler controls, per-message stats, stop button, export |
| **Playground** | Raw completion, the same prompt side by side across F32/Q8/Q4 or two models, a tokeniser inspector, per-token top-k probabilities |
| **Training** | New run (dataset, size preset, `--from`, steps, learning rate), live train/val loss chart with the best-step marker, samples at each checkpoint, cancel, run history, a "chat with this" button |
| **Datasets** | Upload and list text files; character count, vocabulary, and the unseen-character check against a model before `--from` fails on it |
| **Evals** | Perplexity on a held-out file; saved prompt suites with expected outputs as regression tests across models and quantisations |
| **Benchmarks** | The measurement protocol as a feature: interleaved A/B, 5 runs, median and range, idle check first, results stored so the README numbers can be reproduced |
| **Monitoring** | Request log, latency histograms, errors, server log tail |
| **Settings** | Auth, users, API keys, thread knobs, default sampler, theme |

The tokeniser inspector and the top-k view serve the teaching half of the
mission, and both are cheap because the logits are already there.

## Auth

One `AuthProvider` seam, the mode set in `kvad.toml`. Bootstrap config lives in
the file (bind address, DB path, auth mode, OIDC secret); runtime settings live
in the DB.

- **`none`** — loopback only. The server refuses to bind a non-loopback address
  without auth unless `--insecure` is passed.
- **`local`** — username and password. argon2id hashes; server-side sessions in
  SQLite; an `HttpOnly; SameSite=Lax` cookie plus an `Origin` check on mutating
  requests; first-run bootstrap through a one-time setup token printed to the
  terminal.
- **`basic`** — the same user table over HTTP Basic, for scripts and reverse
  proxies.
- **`oidc`** — the `openidconnect` crate, authorization code + PKCE, with an
  allow-list of emails or a claim-to-role mapping. OIDC rather than raw OAuth2
  so identity is standardised across Google, GitHub-via-Dex, Keycloak,
  Authentik and the like.
- **API keys** — bearer tokens for `/v1/*`. Stored hashed, shown once, revocable
  per key, usable alongside any mode above.
- **Roles** — `admin` (everything) and `user` (chat and playground only).

## Schema sketch

`users`, `sessions`, `api_keys`, `settings`, `conversations`, `messages` (with
stats columns), `jobs` (kind, state, params as JSON, timestamps, error),
`train_metrics` (job, step, train loss, val loss, chars/s), `train_samples`,
`datasets`, `eval_suites`, `eval_runs`, `bench_runs`, and `requests` —
per-request timing, pruned or rolled up after N days.

## Phases

Each ends with something usable.

**Phase 0 — make the engine servable. No HTTP yet.**

- Move the `Cmd`/`Evt` actor out of `crates/tui` into `kvad::service`, so the
  TUI and the server share it.
- `train::run` and `weights::fetch_with` report progress as `&str`. Add
  structured events beside them: `Step{n, train, val, chars_per_s}`, `Sample`,
  `Saved`, and `Download{file, bytes, total}`.
- Whether `hf-hub` exposes byte-level progress is unchecked.
- Add a cancel flag to training, and an explicit unload.
- This touches the TUI, which was never driven interactively. Its own commit.

**Phase 1 — skeleton.** `kvad-serve` with config loading, SQLite migrations, the
embedded SPA, `/api/health`, the auth extractor wired up with only `none`
implemented, and the Svelte shell: a daisyUI drawer sidebar, navbar, theme
toggle and a toast store.

**Phase 2 — models and chat.** The Models page, `/v1/chat/completions` and
`/v1/models` over SSE, and the Chat page with persisted conversations.

**Phase 3 — auth.** `local`, `basic`, API keys, roles, then OIDC.

**Phase 4 — jobs.** A job runner with downloads and training as jobs, SSE
progress, the Training and Datasets pages, and the live loss chart.

**Phase 5 — monitoring.** A metrics ring buffer in memory, rollups in SQLite,
and the Dashboard and Monitoring pages.

**Phase 6 — playground, evals and benchmarks.**

**Phase 7 — polish.** An OpenAPI docs page, `kvad serve` as a subcommand, a
README chapter, and a release build with the embedded UI.

## Testing

- API tests run against a tiny `nanograd`-trained model, as the journey test
  does, so CI needs no downloads.
- Auth gets the mutation treatment: for each mode, remove the check and confirm
  a test fails.

## Open decisions

1. **Multi-user with roles, or a single admin?** Roles cost about a day and
   matter only if other people will use the instance.
2. **GPU in the server from the start?** The `Backend` enum already covers it,
   but it pulls candle and metal into `kvad-serve`'s build. Proposed: a cargo
   feature, on by default on macOS.
3. **Layout of the frontend.** Proposed: `web/` at the repository root, beside
   `crates/`, with its own `package.json`.
