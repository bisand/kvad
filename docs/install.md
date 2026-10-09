[← Back to the README](../README.md)

# Install

```bash
curl -fsSL https://kvad.eu/install.sh | sh
```

Prebuilt binaries from the latest [release](https://github.com/bisand/kvad/releases),
checked against that release's `SHA256SUMS` and put in `~/.local/bin`. It asks
whether to add that directory to your `PATH`, whether `kvad-serve` should
start when you log in, and — if it should — what address and what port it
listens on, `127.0.0.1` and `5823` by default. Those are two questions rather
than one because they are two different mistakes. It asks on `/dev/tty`, so
the questions survive being piped into `sh`. Nothing needs root.

Say an address it cannot serve from and it says why and asks again, rather
than installing a service that cannot start: a port something else already
holds, or a non-loopback address, which `kvad-serve` refuses while no auth
mode is configured.

Run the same command again to upgrade. An upgrade keeps the address the
service is already installed with, rather than quietly moving it back to the
default — it says what that address is and offers to keep it. It also stops
the service before replacing the binaries underneath it, and starts it again
afterwards, whether or not the unit file itself was rewritten.

Starting the service ends with a question put to the server itself: does
anything answer at its address. If nothing does within fifteen seconds the
installer says so, prints what `kvad-serve` wrote on its way down, and does
not finish on "installed" in green. The case it was written for is a database
newer than the release being installed — a build from a checkout has migrated
it — which the server refuses to open, as it should, and which the service
manager reports as a job it has. `kvad service install` and `kvad service
start` ask the same question and exit non-zero on the same answer.

For a machine with nobody watching, `--yes` never opens a terminal and takes
the quiet answer to every question: on a fresh machine, binaries and nothing
else; on one that already runs the service, the same service pointed at the
new binaries. Ask for the rest explicitly:

```bash
curl -fsSL https://kvad.eu/install.sh | sh -s -- --yes --service
```

Once it is installed the service is `kvad service`'s to manage, and the
installer uses the same command to install and stop it:

```bash
kvad service status        # is it running, is it answering, what is loaded
kvad service restart
kvad service logs -f
kvad service install --port 5900
kvad service uninstall
```

`--prefix DIR`, `--version vX.Y.Z` and `--uninstall` do what they look like;
`--uninstall` removes the binaries and the service and leaves your models and
conversations alone. An install into a `--prefix` other than the one the
background service runs from leaves the service alone, so a second copy can
sit beside the real one; `--service` moves the service to it. `--data-dir
DIR` puts the database, images, datasets and trained models somewhere other
than `~/.local/share/kvad`, by writing `[data] dir` to `kvad.toml`, which the
CLI and the server both read. `KVAD_DATA_DIR` overrides that for one process.
A chosen data directory takes the models pulled from Hugging Face with it,
into `huggingface/hub` inside it; without one they stay in
`~/.cache/huggingface/hub`, and `HF_HUB_CACHE` or `HF_HOME` win either way.
The web UI's Settings page does the same while the server runs, and moves
what is already there: renamed on the same disk, copied to another one with
the old copy kept until you delete it there. `--host ADDR` and `--port N` answer the address
questions ahead of time, for a run that should not stop to ask — `--bind
HOST:PORT` still says both at once. `sh install.sh --help` lists the rest.

## Help, and Tab completion

`kvad help` lists every command in a line each, and `kvad help COMMAND` — or
`kvad COMMAND --help` — says what one takes, with examples. A mistyped command
or option is told the nearest one that exists.

The installer offers to set up Tab completion for your login shell — zsh,
bash or fish — and `--completions` or `--no-completions` answers ahead of
time. It asks `kvad` to do it, and so can you:

```bash
kvad completions install      # your login shell; or name them: install zsh fish
kvad completions status       # where it is set up
kvad completions uninstall    # take it out again
```

For zsh and bash that is one line at the end of `~/.zshrc`, or `~/.bashrc`
(`~/.bash_profile` on a Mac), which asks `kvad` for the script as each shell
starts. For fish it is a file, `~/.config/fish/completions/kvad.fish`.
`uninstall` removes what `install` wrote and nothing else, and the installer's
`--uninstall` runs it. `kvad completions zsh` prints a shell's script, for
somebody who would rather place it themselves.

After that Tab completes commands and their options, and what only the
machine knows: `kvad run --model <Tab>` lists the language models on disk,
`kvad images make --model <Tab>` the image models, `kvad unload <Tab>` what is
in memory, `kvad load M --backend <Tab>` the server's backends, and
`kvad jobs show <Tab>` the jobs, each with a line saying what it is. With a
server running the answers are the server's, as the commands' are; `--remote
URL` earlier on the line asks that one instead.

The shell scripts hold no list of their own. They ask the installed `kvad`
at each Tab, so an upgrade needs nothing done again.

macOS gets all four binaries. Linux gets `kvad` and a CPU-only `kvad-serve`:
`kvad-tui` and `kvad-gpu` both link candle against Metal, which is not a thing
off a Mac. On macOS that means Apple Silicon, which is what this is developed
and measured on; Intel Macs had an untested build up to v0.6.0 and none since.

The binaries are unsigned and unnotarized, which is deliberate rather than
lazy: Gatekeeper's quarantine flag is set by LaunchServices, so a tarball
fetched with `curl` never gets one and runs as-is. Download the same tarball
with a browser and macOS will want convincing.
