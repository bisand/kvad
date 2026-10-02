---
title: Background service
description: Run the server at login, under launchd on macOS or systemd on Linux, and manage it with one command.
---

The installer offers to set this up. It can also be done, changed or undone at
any time:

```bash
kvad service install
kvad service install --port 5900
kvad service uninstall
```

On macOS the service is a launchd agent named `net.kvad.serve`. On Linux it is
a systemd user unit. Either way it runs as you, needs no root, and starts when
you log in.

## Managing it

```bash
kvad service status        # is it running, is it answering, what is loaded
kvad service start
kvad service stop
kvad service restart
kvad service logs -f
```

`stop` is for now: the service starts again at the next login. `uninstall` is
for good.

## Status

```text
$ kvad service status
service   net.kvad.serve (launchd) — running, pid 25773
          /Users/ada/.local/bin/kvad-serve --bind 127.0.0.1:5823
address   http://127.0.0.1:5823 (the installed service)
answers   kvad-serve 0.11.0, up 2h 46m, auth none, signed in as local
models    stabilityai/stable-diffusion-xl-base-1.0@gpu-bf16 (6.4 GB)

Running, and answering. Commands go to it; --local runs one here instead.
```

`status` asks two things and says when they disagree: the service manager,
about whether the job is running, and `/api/health`, about whether anything
answers. A job that is running and not answering on its port is the failure
people actually hit.

Its exit status is 0 when the server answers, 1 when the service manager says
it is running and nothing answers, and 3 when it is not running.

## The address

`install` takes `--host` and `--port`. With neither, it keeps the address the
service already has, then `bind` under `[server]` in `kvad.toml`, then
`127.0.0.1:5823`.

It will not install a service that cannot start: a port something else holds,
or a non-loopback address with no [authentication](/docs/authentication/).
`--force` installs over those objections.

## A model ready after a reboot

By default the service holds no model until a request names one. To have the
default model in memory as soon as the service is up, set `autoload = true`
under `[server]` in [kvad.toml](/docs/configuration/).
