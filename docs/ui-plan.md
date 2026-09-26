# Plan: a web UI for Kvad, and the server under it

Written 2026-09-19 against the working tree after `5e551d9`, from a reading of
`crates/llm/src/{runtime,train,weights,hub}.rs`, the TUI's actor in
`crates/tui/src/engine.rs`, and the README's "Becoming a server" section.
Where something is a guess it says so. The phases at the end say what is built.

**All eight phases are done.** What is above them was the plan and is now, for
the most part, a description — but it has deliberately not been rewritten into
one. The guesses are left where they were made, so that the phases underneath
them can say which turned out wrong: the compute lease in Phase 4, the copy it
asked for in the UI, and the claim in the README that a server around a
single-sequence engine measures nothing interesting.

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
- **Training takes a compute lease.** Measured in Phase 4: chat during a
  training run decodes at roughly half idle speed on all cores, two-thirds
  with the run capped. So the two are allowed to run together and the UI says
  "slower" rather than "queued"; the numbers are in that phase.
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
| **Images** | Text to image with an image model (SDXL, Qwen-Image): prompt, negative prompt, size, steps, guidance, seed; each denoising step with a preview as it happens; a gallery of every picture kept, with its settings. See `docs/image-plan.md` |
| **Videos** | Text to video with a video model (LTX-2.5), or a picture and text: prompt, a picture to start from, size, length (or the model's own, from the prompt), frame rate, sound, seed; progress through each phase with a rough time left, since a video is a job the server runs for minutes; a gallery of players with posters, every video kept with its settings. See `docs/video-plan.md` |
| **Training** | New run (dataset, size preset, `--from`, steps, learning rate), live train/val loss chart with the best-step marker, samples at each checkpoint, cancel, run history, a "chat with this" button |
| **Datasets** | Upload and list text files; character count, vocabulary, and the unseen-character check against a model before `--from` fails on it; read a documentation site into a corpus, scoped to the starting URL's directory, with a manifest of every page kept beside the text |
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
  `nervus::text::Training::stop` and read once a step rather than once a
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

Also worth knowing: the default bind is `127.0.0.1:5823`, which is a
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

**Phase 4 — jobs. Done.**

- **Work that outlives its request.** A download takes minutes and a training
  run can take an hour, both longer than a browser tab reliably stays open, so
  the work belongs to the server and a request only watches it. Each live job
  has a `broadcast` channel; a watcher subscribes *before* the history is read
  so nothing is lost in between, which means a step can be seen twice — hence
  metrics and samples keyed by step. Restarting the server marks whatever was
  running as failed, because a row that claims to be running when nothing is
  is a run somebody waits for forever.
- **Checkpoints are written down as they happen**, not at the end, so a run
  watched in a tab yesterday is a chart today and a run that dies halfway
  still has its curve. Pulls became jobs too; loads did not, because they are
  engine work and take seconds.
- **One training run at a time**, refused rather than queued: an hour of
  waiting with no way to see why is worse than being told now. Downloads have
  no such limit — they are waiting on a network.
- **The Training page** has the live loss chart (uPlot, colours read from
  daisyUI's variables so it follows the theme), the best step marked, the
  samples each checkpoint writes, cancel, history, and "chat with this".
  Cancelling uses the flag Phase 0 put on `Training::stop`; a stopped run
  keeps the best model it reached and is `cancelled`, not `failed`.
- **The Datasets page** answers the unseen-character question before a run
  rather than after: `kvad train --from` refuses a text containing a character
  the model has no token for, but only once it has read the file and only for
  the *first* offender. The check here names all of them, so the fix is one
  edit.

**The compute lease, measured rather than assumed.** Five generations at each
setting with idle runs either side to catch drift, on an 18-core machine
decoding SmolLM2-135M at q8:

| | median | range |
|---|---|---|
| idle | 70–84 tok/s | 53–92 |
| training, 18 threads | 32–37 tok/s | 25–41 |
| training, 8 threads | 47 tok/s | 42–54 |
| training, 4 threads | 52 tok/s | 44–56 |

So chat during training runs at **roughly half idle speed** on all cores
(45% and 52% across two runs), and about two-thirds with the run capped to 8
threads; 8 against 4 is inside the noise. Inference is therefore *not* queued
behind training and should not be — a thirty-minute wait is not a queue — and
the banner says "slower", which is what was measured. This replaces the
placeholder copy this plan asked for.

**Phase 5 — monitoring. Done.**

Two stores, because two questions. A **ring buffer in memory** answers "what
is happening now" — the last 2000 requests, 500 generations and 500 log lines
— and reading it costs a mutex and no disk, which is what a page polling every
few seconds needs. A **table in SQLite** answers "what did yesterday look
like"; it is written in batches every five seconds, because a lock on the
database in the middle of every response is a lock on the server, and pruned
after seven days, because a row a second is nothing for SQLite and unreadable
for a person.

- **Percentiles come from the samples**, sorted, rather than from buckets.
  There are at most a few thousand and sorting them is microseconds, so an
  estimate would be a worse number for no saving.
- **Routes are grouped by pattern**, `/api/jobs/{id}` rather than
  `/api/jobs/17`, or a histogram would have one row per id.
- **The Dashboard** shows what the machine is doing: the loaded model, decode
  and time-to-first-token sparklines, queue depth, resident memory, the KV
  cache against what it would cost at full context, disk broken into what can
  be downloaded again and what cannot, and anything running.
- **The Monitoring page** has per-route latency, the request log with each
  generation's numbers beside it, and the server log tail — bounded, in
  memory, and gone on restart, because a log that has to outlive the process
  belongs to whatever is running it.
- **Resident memory is current, not peak.** `getrusage`'s `ru_maxrss` is the
  obvious answer and the wrong one: it never goes down, so a model that had
  been unloaded would still show as resident.

**Two bugs this phase found in itself.** The first version attached a
generation's numbers to "the most recent completion request", which for a
non-streamed reply was the *previous* one — the handler knows the numbers
before the middleware records the row. The completions handler now records its
own row, which also fixes the second: a streamed reply timed by the middleware
measured the moment the headers went out, which is the moment before all the
work. Completions are now timed end to end; every other streaming route is
still timed to its first byte, and the page says so.

**Phase 6 — playground, evals and benchmarks. Done.**

Three pages that measure the model instead of talking to it, and one fact
that shapes all three: **the engine holds one model at a time**, so every
comparison is a sequence — load, measure, load the next — and the code says so
rather than pretending otherwise.

- **The playground is nearly free**, which is the argument for having it. The
  logits were always computed and generation was throwing away everything
  except the winner; the tokeniser had already cut the prompt up and nobody
  was shown the pieces. `generate_explained` reports the candidates behind
  each token at the cost of one sort and one softmax — tens of microseconds
  against tens of milliseconds of matmul.
- **The probabilities shown are the model's, not the sampler's.** A plain
  softmax over every logit at temperature 1, so the numbers do not move when
  somebody drags the temperature slider; what the sampler did is a separate
  mark saying whether top-k and top-p left that token in play. Both halves are
  needed and they answer different questions.
- **Perplexity needed a new thing from the engine.** `forward_batch` returns
  logits for the last position only, because that is all generation wants;
  scoring wants all of them. `forward_batch_all` runs the output head over the
  whole batch as one matmul, which measured about 430 tokens a second against
  114 for decoding — the same arithmetic, a quarter of the memory traffic. It
  is checked against `nervus`'s own logits at every position, which is the
  same second-implementation argument the checkpoint test rests on.
- **A round visits every variant once**, and a run is several rounds. The
  alternative — five of A then five of B — blames the model for anything that
  changed about the machine in between, and this project has already published
  three wrong numbers that way. The price is a model load per variant per
  round, and that price is the honest one.
- **Every timed generation starts from an empty KV cache.** The same prompt
  twice would otherwise be prefilled out of the first run's cache and report a
  time to first token no first run would ever see.
- **Everything is seeded**, and suites decode greedily. Two variants that
  differ only in their random draw are not a comparison, and a regression test
  that fails one time in five is not a test.
- **The idle check is a refusal, not a footnote.** A benchmark will not start
  while a training run, an eval or another benchmark is going, and the refusal
  names what is in the way.

**What the first real run found.** SmolLM2-135M-Instruct, 3 rounds, 64 tokens:
q4 decoded at 149.5 tok/s (range 149.3–150.2) against q8's 114.6 (97.2–118.3),
and the ranges do not overlap, so the difference is real. But q8 reached its
first token in 30 ms against q4's 41, and scoring took q4 twice as long as q8
for the same 519 tokens. Chased down afterwards: `matmul_bt_with` gates the
batched `SMMLA` kernel on `Data::Q8`, so q4 prefill takes the row-at-a-time
fallback. Both paths are integer and neither dequantises, so this is a missing
kernel rather than a cost of quantisation — which is the opposite of what the
first reading of the number assumed. Since written: unpacking a row pair of
nibbles into `i8` hands q4 to the same kernel, roughly doubling q4 prefill and
closing the gap. The page found a real bug in the engine underneath it, which
is the best argument for having built it. On a two-case suite, q8 continued "The
capital of France is" with " Paris" and q4 with " the capital of the country
of France"; on held-out prose q8 scored 2.03 perplexity against q4's 2.26, and
on the same text shuffled, 514.8 against 568.1. Quantisation costs accuracy in
every one of those, measured rather than assumed.

**Migration 006 rebuilds the jobs table** to widen its `kind` check, and that
is a `DROP TABLE` — which with foreign keys enforced would run every
`ON DELETE CASCADE` first and take every training metric on the machine with
it. The pragma cannot be set from inside a migration, because it is a no-op in
a transaction and every migration runs in one, so the runner turns enforcement
off around each. There is a test that upgrades a database with rows in it.


**Phase 7 — polish. Done.**

- **`/api/openapi.json` is generated from a table, and a test compares that
  table to the router** — by reading the source of the files that register
  routes, because a `Router` has no way to list what is in it. Scanning source
  is cruder than asking, and it has the one property that matters: a route
  added and not documented fails the build. It was checked the way Phase 3
  checked the auth modes, by adding a route and confirming the test names it.
  The first version failed on itself, having scanned its own source and found
  the string it searches *with*.
- **The bodies are described in prose, not as schemas.** A hand-written schema
  for forty-odd endpoints is a second copy of the handlers, and a second copy
  drifts; the first time somebody adds a field and the schema does not, the
  document has stopped being documentation. What is enumerated is exactly what
  the test can check. A generator (`utoipa`) is the right answer for a codebase
  whose handlers carry typed extractors for every body — half of these stream,
  and the annotations would be as long as the table with the drift moved into
  attributes where no test can see it.
- **The docs page is ours rather than Swagger UI from a CDN.** The whole point
  of embedding the UI is one file that runs on a machine with no internet, and
  a documentation page that fetches 400 KB from elsewhere is not that.
- **`kvad serve` hands over to the server binary** rather than linking it.
  `kvad-serve` depends on the engine crate, so the engine crate cannot depend
  on it back — Cargo would refuse the cycle, and would be right to. On Unix it
  `exec`s rather than spawning, so Ctrl-C reaches the server and its exit
  status is the caller's.
- **A README chapter**, with the numbers from Phases 4 and 6 in it.

**A bug this phase found in Phase 6.** `kvad` with no arguments failed with
"there is no such directory", because the Phase 6 tests load models through the
real engine — and loading a model sets the active one, which lives in
`$XDG_CONFIG_HOME/kvad/state.json`. The test run had left the developer's own
CLI pointing at a temporary directory it had then deleted. The tests now point
`XDG_CONFIG_HOME` somewhere harmless. A test suite that writes to `~/.config`
is a bug whatever else it proves.

## Testing

- API tests run against a tiny `nervus`-trained model, as the journey test
  does, so CI needs no downloads. Built in Phase 6: `compare.rs` trains a
  two-layer GPT on "the cat sat on the mat", saves it, and runs a real prompt
  suite, a real benchmark and a real perplexity job against it through the
  real scheduler — three tests, 0.7 seconds, no network.
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
