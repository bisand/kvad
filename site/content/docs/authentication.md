---
title: Authentication
description: Four modes, from none on loopback to an external identity provider, with roles and API keys.
---

With no configuration the server listens on loopback with no authentication:
anyone who can reach the socket is an administrator. That is the right answer
on a laptop, and the server enforces the loopback half of it. Binding any
other address with `mode = "none"` is refused before the socket opens.

To serve other machines, pick a mode in `kvad.toml`:

```toml
[server]
bind = "0.0.0.0:5823"

[auth]
mode = "local"
```

## Modes

| Mode | What it is |
|---|---|
| `none` | No accounts. Loopback only. |
| `local` | A name and a password, against this server's own accounts, with a session cookie. |
| `basic` | The same accounts over HTTP Basic, for scripts and for a reverse proxy that would rather forward a header. |
| `oidc` | An external identity provider, by authorization code with PKCE. |

## The first account

With `local` or `basic`, the first account is made against a one-time token
that the server prints when it starts:

```bash
kvad service logs          # find the token
kvad auth setup TOKEN
```

The web UI offers the same form on its sign-in page. That account is an
administrator, and it creates the others:

```bash
kvad users add ada
kvad users
```

## Roles

There are two, `admin` and `user`. An administrator can load and delete models,
change settings and manage accounts. A user can use what is there: chat,
images, videos, and their own conversations and keys.

```bash
kvad users add ada --role user
kvad users edit ada --role admin
```

Passwords are asked for, never taken as flags.

## API keys

API keys work alongside every mode that has accounts. A key is a bearer token
for `/v1` and for everything else, and carries its owner's role.

```bash
kvad keys add laptop
```

```bash
curl http://mac-studio.local:5823/v1/models -H "authorization: Bearer $KVAD_API_KEY"
```

From the command line, `kvad auth login` signs in and keeps a key for that
server in `credentials.json`, readable by you only.

```bash
kvad --remote http://mac-studio.local:5823 auth login
kvad auth status
kvad sessions              # where you are signed in
kvad keys                  # ls; `kvad keys rm ID` revokes one
```

## OIDC

```toml
[auth]
mode = "oidc"

[auth.oidc]
issuer = "https://accounts.google.com"
client_id = "…"
client_secret = "…"          # omit for a public client
redirect_url = "https://kvad.example.com/api/auth/oidc/callback"

# Who may sign in. At least one of these is required.
allow_emails = ["ada@example.com"]
allow_domains = ["example.com"]

# Who is an administrator. Recomputed at every sign-in.
admin_emails = ["ada@example.com"]
role_claim = "groups"
admin_roles = ["kvad-admins"]
```

An allow list is required because a provider will authenticate every account
it has. Deciding who may use this machine is not its job.

## TLS

The server speaks plain HTTP. To reach it across a network you do not trust,
put it behind a reverse proxy that terminates TLS, or a VPN such as Tailscale.
