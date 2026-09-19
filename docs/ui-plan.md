# Plan: a web UI for Kvad, and the server under it

Written 2026-09-19 against the working tree after `5e551d9`, from a reading of
`crates/llm/src/{runtime,train,weights,hub}.rs`, the TUI's actor in
`crates/tui/src/engine.rs`, and the README's "Becoming a server" section.
Where something is a guess it says so. The phases at the end say what is built;
everything above them is still the plan rather than a description.

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

Each ends with something usable. A phase is marked done here when it is
committed.

**Phase 0 — make the engine servable. No HTTP yet. Done.**

- The `Cmd`/`Evt` actor moved out of `crates/tui` into `kvad::service`, and
  the TUI now spawns it. `Backend::Gpu` is a request `kvad` can describe and
  not fulfil — the GPU crate depends on `kvad`, not the other way — so
  `Engine::spawn` takes a `Loader`. `kvad::service::cpu_loader` is the one for
  a build with no GPU crate in it; the TUI's, in `crates/tui/src/gpu.rs`,
  handles both.
- `train::run_watched` and `weights::fetch_watched` report structured events
  beside the `&str`: `train::Event::{Pace, Step, Saved, Sample}` and
  `weights::Fetch::{Local, Shards, Download, Fetched}`. `run` and `fetch_with`
  still exist and are the same calls with the events dropped.
- **`hf-hub` does expose byte-level progress**, through a `ProgressHandler`
  passed to `download_file().progress(…)`. It is called from the download's
  own tokio tasks while the calling thread is blocked inside the request, so
  the handler has to be `Send + Sync` and take `&self`. That is why
  `weights::Watcher` is an `Arc<dyn Fn>` where every other progress callback
  here is a `&mut dyn FnMut`.
- Training takes a cancel flag: `train::Options::cancel`, copied into
  `nanograd::text::Training::stop` and read once a step rather than once a
  checkpoint — checkpoints are hundreds of steps apart and a stop button that
  takes a minute to answer is one nobody believes. A stopped run reports
  `stopped: true` and leaves the best model it reached on disk, because saving
  happens at every improvement rather than at the end.
- `Cmd::Unload` drops the loaded model without loading another; `u` on the
  TUI's Models tab is the key for it.

**Phase 1 — skeleton. Done.**

`crates/serve` and `web/`, and nothing of the engine yet.

- **Config** is `kvad.toml`, all of it optional, with
  `docs/kvad.example.toml` as the reference. `--config`, `--bind` and `--db`
  override it. The one refusal that matters happens before the socket opens:
  `auth.mode = "none"` off loopback needs `--insecure`, and a mode that is
  named but unbuilt stops the server rather than quietly letting everyone in.
- **SQLite** through `rusqlite`, WAL, one connection behind a mutex, numbered
  `.sql` migrations embedded with `include_str!` and tracked in
  `PRAGMA user_version` — four bytes in the file header, written by the same
  transaction as the schema it describes, so the two cannot disagree.
- **Auth** is a `Provider` trait with `Identity` and `Admin` as axum
  extractors, so a handler that needs an administrator cannot be written
  without asking for one. `NoAuth` is the only provider; the rest is Phase 3
  and changes `auth.rs` alone.
- **The SPA** is embedded with `rust-embed` from `web/dist`. `cargo build`
  never runs npm, so `web/dist/.gitkeep` is committed to give the directory
  something to exist as, and a binary built without a UI serves a page saying
  which command would fix it. Anything that is not a file falls back to
  `index.html`, which is what makes a reload of `/models` work; a missing
  `.js` is still a 404, or a broken build would answer with HTML and the
  console would report a syntax error instead of a missing file.
- **The shell** is Svelte 5 + Vite 8, Tailwind 4 and daisyUI 5: drawer
  sidebar, navbar, a three-way theme menu (system / light / dim, the system
  default left to daisyUI's `--prefersdark`), and a toast store where errors
  stay until dismissed and everything else clears itself. 52 KB of JavaScript
  and 78 KB of CSS, 20 and 13 gzipped. Every planned page is in the sidebar
  from the start, and the ones that are not built say which phase brings them.
- **The dashboard** shows the one thing this phase can honestly show: whether
  the browser and the server agree they are talking, polled every 30s.

Not settled here, because nothing needed it yet: open decision 2, whether the
GPU backend is in the server's build. `kvad-serve` has no engine and so no
`Loader` to choose one for.

Also worth knowing: the default bind is `127.0.0.1:8080`, which is a
well-contended port. `--bind` or `server.bind` moves it.

**Phase 2 — models and chat. Done.**

- **A `Scheduler` owns the engine thread.** `kvad::service::Engine` already
  runs one thing at a time, but it speaks over one pair of channels with one
  receiver, and an HTTP server has many requests each wanting their own stream
  of tokens. The scheduler is that one receiver: jobs queue first-in
  first-out, `depth()` is a number the dashboard shows, and continuous
  batching will replace the inside of its loop without changing anything
  above it. Searching, listing, deleting and pulling do *not* go through it —
  a Hub search has no business waiting behind a thirty-second generation.
- **`/v1/chat/completions` and `/v1/models`**, streaming and not. The `kvad`
  object beside `usage` carries what OpenAI's schema has nowhere to put:
  cached tokens, prefill and decode split apart, and the backend. Naming a
  model that is not loaded loads it, which is slow and is what `model` means.
- **Sampling is per request.** `service::Cmd::Chat` carries a `Sampling`, and
  its `seed` is an `Option`: `None` continues the loaded model's generator so
  two identical requests differ, `Some(n)` restarts it so they agree. That is
  what a benchmark or a bug report needs and what nobody wants by default.
- **The Models page** lists what is here with the same `runnable()`/`blocker()`
  verdicts `kvad search` prints, searches the Hub, pulls with a byte-level
  progress bar over the Phase 0 `Fetch` events, loads and unloads with a
  backend picker, sets the default model the CLI and TUI share, deletes with a
  confirmation that says whether the model can ever come back, and manages the
  quantised-weights cache.
- **The Chat page** persists conversations in SQLite and shows each reply's
  numbers under it — including cached tokens, so the KV cache is visible doing
  its job on the second turn. System prompt, sampler controls, a stop button
  that tells both the stream and the engine, and export to Markdown. A reply
  that was stopped is kept; the question is saved before the answer is asked
  for, so a generation that dies does not lose it.
- **Storage is written by the UI, not by the endpoint.** `/v1/chat/completions`
  stays stateless and identical for every client; our own UI posts what was
  said to `/api/conversations` afterwards. One code path through the engine,
  and it is the compatible one.

**Open decision 2 is settled:** the GPU backend is a cargo feature, on by
default. Cargo cannot express "default on macOS", and it does not need to —
the workspace already requires macOS to build `kvad-gpu` at all. A CPU-only
server is `--no-default-features`, and `engine::available()` offers only what
the build can honour, so the picker cannot promise a backend that is not
there.

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
2. **GPU in the server from the start?** Settled in Phase 2: a cargo feature,
   on by default. See that phase for why "on macOS" turned out not to be a
   thing cargo can say, or need to.
3. **Layout of the frontend.** Settled in Phase 1: `web/` at the repository
   root, beside `crates/`, with its own `package.json`.
