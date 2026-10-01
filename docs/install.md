[← Back to the README](../README.md)

# Install

```bash
curl -fsSL https://raw.githubusercontent.com/bisand/kvad/master/install.sh | sh
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

For a machine with nobody watching, `--yes` never opens a terminal and takes
the quiet answer to every question: on a fresh machine, binaries and nothing
else; on one that already runs the service, the same service pointed at the
new binaries. Ask for the rest explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/bisand/kvad/master/install.sh | sh -s -- --yes --service
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

macOS gets all four binaries. Linux gets `kvad` and a CPU-only `kvad-serve`:
`kvad-tui` and `kvad-gpu` both link candle against Metal, which is not a thing
off a Mac. On macOS that means Apple Silicon, which is what this is developed
and measured on; Intel Macs had an untested build up to v0.6.0 and none since.

The binaries are unsigned and unnotarized, which is deliberate rather than
lazy: Gatekeeper's quarantine flag is set by LaunchServices, so a tarball
fetched with `curl` never gets one and runs as-is. Download the same tarball
with a browser and macOS will want convincing.
