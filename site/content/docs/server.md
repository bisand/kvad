---
title: The server
description: One binary that serves the API and the web UI, holds the models, and runs everything long as a job.
---

```bash
kvad serve                         # loopback on 5823, no authentication
kvad serve --bind 0.0.0.0:5823     # needs an auth mode, or --insecure
```

`kvad serve` hands over to `kvad-serve`, which is one file with the web UI
built into it. Run it by hand like this, or let the
[background service](/docs/service/) start it at login.

| Option | |
|---|---|
| `--bind ADDR` | The address to listen on, such as `127.0.0.1:5823`. |
| `--config PATH` | The configuration file. Default `~/.config/kvad/kvad.toml`. |
| `--db PATH` | The SQLite database file. |
| `--insecure` | Allow a non-loopback address with no authentication. |

## What it holds

- **Models in memory.** Several at once when they fit, each charged for its
  weights and a full context of KV cache. See
  [what is in memory](/docs/models/#what-is-in-memory).
- **A queue.** One scheduler owns the engine, and every generation waits its
  turn behind it. The queue depth is on the dashboard.
- **Jobs.** Training runs, evals, benchmarks, crawls and videos are jobs: they
  answer at once, carry on if the client goes away, and can be followed or
  cancelled.
- **A database.** SQLite, for what the filesystem cannot answer: conversations,
  the settings of every image and video, training runs, accounts and API keys.

## One generation at a time

The engine runs one generation at a time. Requests queue, and the server is
built to say so rather than hide it:

- The playground's side-by-side comparison and the benchmark page both load
  each variant in turn, and say so on the page.
- A benchmark will not start while a training run, an eval or another
  benchmark is going, and the refusal names what is in the way.
- Chat during a training run is slower, not blocked: roughly half speed,
  measured.

## Loading at startup

By default the server holds nothing until something asks, so the first request
after a reboot pays for the load: tens of seconds for most models. To have the
default model loaded as soon as the server is listening:

```toml
[server]
autoload = true
```

It is off by default because the cost is a model in memory a minute after boot
whether or not anybody turns up.

## The command line is a client

With a server running, `kvad` sends its commands to it, so the command line
and the server share one copy of the weights. The first line of output says
where a command went:

```text
$ kvad chat
kvad: http://127.0.0.1:5823 (the kvad service; --local to run in this process instead)
model: Qwen/Qwen3-14B@gpu-q8
```

A command is sent to the first of these that is set:

1. `--remote URL`, or `--local` to run it in the process.
2. `KVAD_URL` in the environment.
3. `url` under `[client]` in `kvad.toml`.
4. This machine's service, if it answers.

The first three are things you said, so a server you name that does not answer
is an error. The last is a guess: when nothing answers there, the command runs
in the process.

That makes a Mac in another room a server for a laptop:

```bash
export KVAD_URL=http://mac-studio.local:5823
kvad auth login
kvad chat
```

## Health

```bash
kvad service status
curl http://127.0.0.1:5823/api/health
```

`/api/health` says what is loaded, how deep the queue is, and which
authentication mode is in force. `kvad metrics` prints the machine's memory,
the KV cache, and recent request timings.
