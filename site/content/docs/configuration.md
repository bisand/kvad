---
title: Configuration
description: The kvad.toml file, the data directory, and the environment variables that override them.
---

Every key has a default, and a missing file is not an error. With no file at
all the server binds loopback with no authentication.

The file is `~/.config/kvad/kvad.toml` (or under `XDG_CONFIG_HOME`), or
whatever `kvad-serve --config PATH` names. Only what has to be known before
the server starts belongs in it. Settings that can change while it runs live
in the database and are edited from the web UI's Settings page.

## The whole file

```toml
[server]
bind = "127.0.0.1:5823"
# autoload = false          # load the default model at startup

[data]
# dir = "/Volumes/Models/kvad"

[videos]
# ffmpeg = "auto"           # "auto", "off", or a path

[database]
# path = "/var/lib/kvad/kvad.db"

[auth]
mode = "none"               # none | local | basic | oidc

[client]
# url = "http://gpu-box.local:5823"
```

## `[server]`

`bind` is the address to listen on. Anything other than loopback needs an
[authentication](/docs/authentication/) mode.

`autoload = true` loads the default model as soon as the server is listening,
instead of at the first request.

## `[data]`

`dir` is where the database, generated images and videos, datasets, trained
models and LoRAs go. The default is `~/.local/share/kvad`, or under
`XDG_DATA_HOME`.

Setting it also moves models pulled from Hugging Face, to `huggingface/hub`
inside it. That is the layout of `~/.cache/huggingface/hub`, so an existing
cache moves with one `mv`.

Changing the key in the file moves nothing: move the old directory yourself,
with the server stopped. The web UI's Settings page does the whole thing for
you. It moves the data, writes this key, and restarts the server into the new
place. On the same disk that is a rename; to another disk it is a copy, with
the old one kept until you delete it there.

## `[videos]`

`ffmpeg` names the ffmpeg that compresses each video, reads the picture a
video starts from, and decodes pictures for LoRA training. `auto` looks on
`PATH` and in the usual Homebrew and system directories.

## `[client]`

`url` is where the `kvad` command line sends its commands, when it should not
look for a server on this machine. This section is read by the command line,
not the server.

## Environment

| Variable | |
|---|---|
| `KVAD_URL` | The server the command line talks to. Same as `--remote`. |
| `KVAD_API_KEY` | The API key the command line sends. |
| `KVAD_DATA_DIR` | The data directory, for one process. Overrides `[data] dir`. |
| `KVAD_THREADS` | How many threads the CPU engine uses. |
| `KVAD_QUANT_CACHE` | Where pre-quantised weights are kept. Default `~/.cache/kvad/quant`. |
| `KVAD_GPU_MPP=0` | Turn off the M5 matrix-unit kernels. |
| `HF_TOKEN` | A Hugging Face token, for gated models. |
| `HF_HUB_CACHE`, `HF_HOME` | Where Hugging Face models are kept. These win over `[data] dir`. |
