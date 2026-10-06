---
title: Install
description: One command on macOS or Linux. Prebuilt binaries, checked against the release's checksums, in your home directory. Nothing needs root.
---

```bash
curl -fsSL https://kvad.eu/install.sh | sh
```

The script downloads the latest [release](https://github.com/bisand/kvad/releases)
for your machine, checks it against that release's `SHA256SUMS`, and puts the
binaries in `~/.local/bin`. Run the same command again to upgrade.

It asks three things, on `/dev/tty` so the questions survive being piped into
`sh`:

1. Whether to add `~/.local/bin` to your `PATH`.
2. Whether `kvad-serve` should start when you log in.
3. If it should, the address and the port it listens on. The defaults are
   `127.0.0.1` and `5823`.

An address it cannot serve from is refused with the reason, and asked again: a
port something else holds, or a non-loopback address while no
[authentication](/docs/authentication/) is configured.

## macOS

Apple silicon, any M-series Mac. You get all four binaries: `kvad`,
`kvad-serve`, `kvad-tui` and `kvad-gpu`.

The binaries are unsigned. A tarball fetched with `curl` carries no quarantine
flag, so they run as they are. If you download the same tarball with a browser
instead, macOS will ask before running it.

`ffmpeg` is optional. With it on your `PATH` (or in `/opt/homebrew/bin`),
videos are compressed to H.264, a video can start from a picture, and LoRA
training can read your photos:

```bash
brew install ffmpeg
```

Intel Macs are not supported. The last untested build for them was v0.6.0.

## Linux

x86-64 and arm64, glibc. You get `kvad` and a CPU-only `kvad-serve`: the GPU
engine and the terminal app link against Metal, which only exists on a Mac.
Language models run on the hand-written CPU engine. Image and video generation
need the GPU engine, so they are macOS-only for now.

The background service is a systemd user unit, managed by `kvad service` the
same way as on macOS.

## Without the questions

For a machine with nobody watching, `--yes` never opens a terminal and takes
the quiet answer to every question: on a fresh machine, binaries and nothing
else.

```bash
curl -fsSL https://kvad.eu/install.sh | sh -s -- --yes --service
```

| Flag | What it does |
|---|---|
| `--prefix DIR` | Where the binaries go. Default `~/.local/bin`. |
| `--data-dir DIR` | Where the database, images, datasets and trained models go. Default `~/.local/share/kvad`. |
| `--version TAG` | Install a named release, such as `v0.11.0`. |
| `--host ADDR`, `--port PORT` | The address the background service listens on. |
| `--bind HOST:PORT` | Both at once. |
| `--service`, `--no-service` | Install the background service, or skip it, without asking. |
| `--add-path`, `--no-add-path` | Add the install directory to `PATH`, or leave it, without asking. |
| `-y`, `--yes` | Never ask. |
| `--uninstall` | Remove the binaries and the service. Models and data stay. |

The same answers can come from the environment: `KVAD_INSTALL_DIR`,
`KVAD_VERSION`, `KVAD_HOST`, `KVAD_PORT` and `KVAD_BIND`. Set `GITHUB_TOKEN`
if you hit GitHub's rate limit.

## Upgrade

Run the install command again. An upgrade keeps the address the service is
already installed with, stops the service before replacing the binaries
underneath it, and starts it again afterwards.

The installer then checks that the server answers at its address. If it does
not within fifteen seconds, the installer says so and prints what `kvad-serve`
wrote as it stopped. `kvad service install` and `kvad service start` do the
same, and exit non-zero.

## Uninstall

```bash
curl -fsSL https://kvad.eu/install.sh | sh -s -- --uninstall
```

This removes the binaries and the service and leaves your models and
conversations alone. The data is in `~/.local/share/kvad`, and models pulled
from Hugging Face are in `~/.cache/huggingface/hub`.

## From source

You need a stable Rust toolchain, and Node 22 for the web UI.

```bash
git clone https://github.com/bisand/kvad
cd kvad
cd web && npm ci && npm run build && cd ..
cargo build --release
```

The web UI is built first because `kvad-serve` embeds `web/dist` when it
compiles. A server built without it still runs, and serves a page that says no
UI was built in.

The binaries are in `target/release`. On Linux, build only what links there:

```bash
cargo build --release -p kvad
cargo build --release -p kvad-serve --no-default-features
```

## Where things live

| What | Where |
|---|---|
| Binaries | `~/.local/bin` |
| Database, images, videos, datasets, trained models, LoRAs | `~/.local/share/kvad` |
| Models from Hugging Face | `~/.cache/huggingface/hub` |
| Configuration | `~/.config/kvad/kvad.toml` |

Choosing a data directory moves the Hugging Face models with it, into
`huggingface/hub` inside it. `HF_HUB_CACHE` and `HF_HOME` still win when they
are set. See [Configuration](/docs/configuration/).
