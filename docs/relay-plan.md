# Plan: reaching your Kvad from anywhere, through a relay you host

Tracked in [#240](https://github.com/bisand/kvad/issues/240).

Written 2026-10-16 against the working tree at `c865b51`, from a reading of
`README.md`, `docs/cluster-plan.md`, `docs/serve.md` and
`site/content/docs/authentication.md`. Nothing here has been built or
measured. Numbers that appear are either published by someone else and say
so, or are questions Phase 0 exists to answer.

## What the owner asked for

People who run models locally mostly do it for privacy. They also want to use
those models from their phone, or from a laptop on someone else's network:
record a voice note on the phone, have the Mac transcribe it, get the text
back. Today that means opening a port, running a reverse proxy, or installing
Tailscale on every device (`authentication.md` says exactly that under *TLS*).

The ask: a small Rust service the owner deploys on any public server they
like, which lets their devices find each other and connect, without the
service being able to read anything that passes. Share the Mac's identity by
a link or a QR code, pair with a handshake stronger than "whoever has the
link", and make the service easy to host almost anywhere: Docker, Kubernetes,
a VPS, a home server, a hyperscaler, and as close to "edge" as is honest.

## Is it doable

Yes. The pieces are well understood, and Kvad already has several of them.

**What the service does is small.** It is a rendezvous point and, when a
direct path cannot be made, a relay of encrypted bytes. It holds no content,
no keys, and no state that matters past a restart. That is what makes it easy
to host everywhere.

**The hard part is NAT, and it is a solved hard part.** Two devices behind
NAT can usually reach each other directly once a third party tells each the
other's public address (hole punching). Carrier-grade NAT on mobile networks
and strict corporate firewalls defeat that some of the time. Every working
system answers the same way: relay the already-encrypted traffic over HTTPS
on 443, which almost every network allows. Tailscale's DERP does this; so
does iroh's relay. The relay sees ciphertext, peer identities, IP addresses
and timing. Self-hosting moves even that metadata onto the owner's own server.

**The Mac is the easy end.** It is stationary, usually behind a router its
owner controls, and the one that has to be reachable. The phone only ever
dials out. Where the Mac's router answers PCP, NAT-PMP or UPnP, or the Mac has
a global IPv6 address, the phone can often reach it directly without punching
anything. Phase 0 should find out how often that is true.

**The seams already exist in Kvad.**

- `kvad-serve` is one axum router with its own authentication: accounts,
  roles, sessions and API keys. A tunnel only has to deliver HTTP requests to
  that router. Everything above it, including who may do what, is unchanged.
- The cluster work already has a pre-shared key kept owner-only in
  `data_dir()`, a two-way challenge handshake, and `_kvad._tcp` discovery
  over mDNS. The same habits apply here: keys in files, never in arguments;
  proofs bound to one connection and one direction; the LAN found without
  typing addresses.
- The CLI already has `--remote URL` and `credentials.json`. A remote that is
  a node identity rather than a hostname is one more kind of URL.

## What it is, said plainly

Three parts.

**`kvad-relay`, the public service.** A new crate and binary with no ML in it
and nothing from the engine, so it builds as a small static binary for Linux
on amd64 and arm64. It does three jobs:

1. *Relay.* Forwards end-to-end encrypted traffic between two peers over
   HTTPS/WebSocket on 443 when no direct path exists.
2. *Address observation.* Tells a peer the address and port its packets
   arrive from, so it can tell the other side where to aim. Needs a reachable
   UDP port. Without UDP the service degrades to relay-only, which still
   works.
3. *Pairing broker.* Accepts a short-lived, single-use pairing token from the
   Mac and delivers the first message from a phone that presents it. Stores
   nothing on disk.

**The Mac side, inside `kvad-serve`.** An Ed25519 identity kept in
`data_dir()` as `remote.key`, readable by its owner only, generated on first
use. A `[remote]` section in `kvad.toml` naming one or more relays. When it is
on, the server keeps a connection to its relays, accepts tunnelled
connections from paired devices, and hands each one to the same router that
serves `127.0.0.1:5823`.

**The client side.** Three, in order of how soon they are worth having:

- `kvad --remote kvad://<node-id>` for the CLI, so `kvad chat` and `kvad api`
  work from anywhere.
- `kvad tunnel`, which listens on `127.0.0.1` on the client machine and
  forwards to the paired Mac. Any OpenAI client, OpenCode included, then
  works unmodified against a local port.
- A phone app. Out of scope for this plan except as the reason for it; the
  pairing design below is written so that the app is just another client.

## Pairing and the handshake

The Mac's node id is its public key. Knowing it is not enough to connect.

**Starting a pairing.** `kvad remote pair` in the terminal, or a button in the
web UI's settings. The Mac asks its relay for a pairing slot and shows:

- a QR code, in the terminal and in the UI,
- the same thing as a link,
- a six-digit code, shown only on the Mac.

The link carries the node id, the relay URL(s), the single-use token and a
protocol version:

```
kvad://pair?node=<ed25519-pubkey>&relay=https://relay.example.org&token=<one-time>&v=1
```

The token lives for a few minutes (ten is a reasonable start) and is burned
on first use, whether the pairing succeeds or not.

**The handshake, in layers.**

1. *The transport is bound to the key.* The encrypted session can only be
   completed with whoever holds the private key for that node id, so neither
   the relay nor anyone in the path can stand in for the Mac.
2. *The token proves the device received the QR or the link.* A guessed or
   scraped node id is not enough.
3. *A PAKE when the link travelled over an untrusted channel.* A QR code read
   off the Mac's own screen is a trusted channel; a link sent by email or chat
   is not. For links, the phone also asks for the six-digit code and both
   sides run a password-authenticated key exchange (SPAKE2 or CPace). Someone
   who intercepted the link gets one guess, and the slot is gone.
4. *A short authentication string, optional.* Both screens show a few words
   or digits derived from the session keys, and the owner confirms they
   match. Worth offering in the UI; not worth requiring.
5. *Pinning.* Each side stores the other's public key. Reconnecting is silent
   from then on.

**What pairing creates.** A row in a new `devices` table: the device's public
key, a name, the account it belongs to, when it was paired and last seen. It
also creates an ordinary API key owned by that account and bound to that
device, so a paired phone has exactly the role its owner has and nothing
more. `kvad devices` lists them; `kvad devices rm ID` revokes the pin and the
key together. The web UI shows the same list next to sessions and keys.

Kvad's own authentication still applies inside the tunnel. A paired device is
a way to *reach* the server, not a way around signing in.

## The transport: build on iroh, or write it

Two honest options. The plan picks the first and keeps the second possible.

**A. Build on iroh.** n0's iroh is Rust, addresses peers by Ed25519 public
key, does hole punching and path selection over QUIC, and falls back to
`iroh-relay`, a separately deployable relay server with an official Docker
image and ACME support. Its documentation says relays are stateless and that
clients fail over between the relays in their list. That is most of
`kvad-relay` already written and already exercised on real networks.

The cost: the wire protocol moves at their pace, and some relay
configuration is, by their own docs, only documented in the source. The
"written from the arithmetic up" rule is for the models, not for moving bytes,
so taking a networking library here is no more out of place than axum.

**B. Write it.** `quinn` and `rustls` for QUIC, `tokio`, a few hundred lines
of rendezvous and relay. Full control, at the cost of everything iroh already
learned the hard way: hole-punching timing, path racing, connection migration
when a phone moves from Wi-Fi to 5G, and relays that survive real traffic.

**The choice.** A, behind a `trait Tunnel` so that the server, the CLI and
pairing never name iroh. Pairing, the device table, the PAKE and the link
format are Kvad's own in either case. If iroh's pace or protocol ever gets in
the way, B replaces one implementation of one trait.

Whether `kvad-relay` *is* `iroh-relay` with Kvad's packaging and a pairing
endpoint beside it, or embeds it as a library, is decided in Phase 1 by
whichever keeps the relay one binary with one config file.

## Hosting

The relay is built to be portable from the start: stateless, one static
binary, configured by TOML or environment variables, logs to stdout.

**TLS, three ways:** ACME built in; a certificate and key on disk; or
terminated by an ingress in front. **UDP optional:** without it, relay only.
**Health and metrics:** `/healthz` and a Prometheus endpoint. **Self-test:**
a page and a `kvad-relay check` command that report whether TLS is valid,
whether UDP is reachable from outside, and what kind of NAT the relay sees a
client behind. That last one is the diagnosis that TURN and STUN never gave.

| Target | Packaging | Notes |
|---|---|---|
| Docker / Compose | Multi-arch image (amd64, arm64), distroless | The default. One `compose.yaml` in the repo. |
| Kubernetes | Helm chart and plain manifests | HTTPS works through any ingress. Address observation needs a Service that passes UDP. |
| VPS | `install.sh --relay`, systemd unit, cloud-init snippet | The cheapest robust option. |
| Home server, Raspberry Pi | Same binary, systemd or Docker | Needs 443 forwarded, and a UDP port for the full set. |
| Hyperscalers | Terraform modules: AWS (EC2 or ECS behind an NLB), Azure Container Apps, GCP | Load balancers must pass UDP and long-lived TCP. Serverless platforms that time out connections do not fit. |
| Fly.io and similar | `fly.toml` | Fly passes UDP; many PaaS hosts do not, and get relay-only. |
| Homelab app stores | NixOS module, Unraid and TrueNAS apps, Coolify template | Later, by demand. |
| Edge (Workers and the like) | Not in v1 | No UDP and no long-lived raw sockets. A relay-only WebSocket variant is possible but is a second implementation. Said here so nobody is surprised. |

**Relays are not open to the world.** A relay serves only node ids it has
been told about: the Mac registers with a relay secret from `kvad.toml`, and
the relay forwards only to and from registered nodes and devices they paired.
A public relay that anyone can use is a bill and an abuse report waiting to
happen.

**More than one relay.** `[remote].relays` is a list. A home server plus a
small VPS gives failover for free, and the client tries them in order.

## Failure

- *Relay down.* The client tries the next relay. If all are down and no
  direct path is known, the CLI and the UI say so and show what was tried.
- *Token expired or used.* Refused with a message that says which, and the
  Mac can show a new code.
- *Wrong PAKE code.* The slot is burned. No retry on the same slot.
- *Unknown device.* A device whose key is not in `devices` is dropped at the
  handshake, before any HTTP is read.
- *Revoked device.* Same as unknown, and its API key no longer works on the
  LAN either.
- *Version mismatch.* The handshake carries a protocol version. A mismatch is
  refused with both versions in the message.
- *Mac asleep.* The relay says the node is not connected. Waking it is not
  this plan's problem; a client can say "your Mac is not online" rather than
  time out.

## Phases

**Phase 0: measure the problem.** Before any server code: on the owner's own
networks, how often is a direct path possible? Home broadband, a Norwegian
mobile network, a corporate guest Wi-Fi, a hotel. For each: does the Mac's
router map a port (PCP, NAT-PMP, UPnP), does the phone get public IPv6, does
hole punching succeed, and what latency does a relay add. A scratch tool built
on iroh is enough. The table this produces replaces every guess above.

**Phase 1: a tunnel on one machine.** `trait Tunnel`, an iroh-backed
implementation, `kvad-serve` accepting a tunnelled connection and handing it
to the router, `kvad --remote kvad://...`. Test: the existing API tests run
through the tunnel on loopback and pass unchanged.

**Phase 2: identity and pairing.** `remote.key`, `kvad remote pair`, the
link format, token lifetime, the PAKE, the `devices` table and its API key,
`kvad devices`. Tests: a good pairing; an expired token; a used token; a
wrong code; a revoked device; a device paired to one account cannot act as
another.

**Phase 3: `kvad-relay`.** The crate, its config, access control, health,
metrics, the self-test. Docker image and Compose file first. Test: two
processes that can only reach each other through it, and a third that is
refused.

**Phase 4: the client side.** `kvad tunnel`, the web UI's pairing screen and
device list, and the CLI over a real relay from a real phone network.

**Phase 5: packaging.** Helm, systemd, `install.sh --relay`, Terraform
modules, Fly. Each target gets a short page on kvad.eu and a CI job that at
least builds it.

**Phase 6: the reason for all of it.** A phone client, and a transcription
endpoint for it to call. The second is its own plan: a Whisper-family
architecture as a plugin and `POST /v1/audio/transcriptions` in OpenAI's
shape. This plan only makes sure the first can reach the second.

## Open questions

- Is `kvad-relay` a fork-free repackaging of `iroh-relay`, or its own binary
  that embeds it? Phase 1.
- Does the pairing broker belong in the relay, or should pairing ride on the
  same relay protocol with no extra endpoint at all?
- Should a Mac be able to act as its own relay when it is reachable (a
  forwarded port or IPv6), so that a user with no server at all still gets
  remote access?
- What does the phone do while the Mac is asleep: record and sync later, or
  refuse? "Sync later" turns a networking problem into a queue, which is the
  easier problem.
- Does the relay need rate limits per node from day one, or only once
  somebody runs a shared one?

## What it buys

The TLS section of the authentication guide stops having to send people to a
reverse proxy or a VPN. A user who runs models locally for privacy can use
them from anywhere, through infrastructure they own, that cannot read what
passes through it, and pair a phone by pointing it at a QR code.
