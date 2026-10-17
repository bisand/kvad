# Plan: reaching your Kvad from anywhere, through a relay you host

Tracked in [#240](https://github.com/bisand/kvad/issues/240).

Written 2026-10-16 against the working tree at `c865b51`, from a reading of
`README.md`, `docs/cluster-plan.md`, `docs/serve.md` and
`site/content/docs/authentication.md`. Nothing here has been built or
measured. Numbers that appear are either published by someone else and say
so, or are questions Phase 0 exists to answer.

Revised 2026-10-17: the direct path between devices is now a part of the
design in its own right, beside the relay, not something left to the
transport library. See *Direct when possible, relayed when not*.

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
on, the server keeps a connection to its relays, listens on a UDP port of its
own for direct connections, advertises its node id on the LAN, accepts
tunnelled connections from paired devices by whichever path they arrive, and
hands each one to the same router that serves `127.0.0.1:5823`.

**The client side.** Three, in order of how soon they are worth having:

- `kvad --remote kvad://<node-id>` for the CLI, so `kvad chat` and `kvad api`
  work from anywhere.
- `kvad tunnel`, which listens on `127.0.0.1` on the client machine and
  forwards to the paired Mac. Any OpenAI client, OpenCode included, then
  works unmodified against a local port.
- A phone app. Out of scope for this plan except as the reason for it; the
  pairing design below is written so that the app is just another client.

## Direct when possible, relayed when not

The relay is one part of the design, not all of it. Each connection takes a
path, the path is chosen per connection, and it can change while the
connection is open. In order of preference:

1. **Same network.** The phone and the Mac are on the same Wi-Fi. That is
   the most common case, and it should not go through a VPS in another
   country. The Mac advertises its node id over mDNS, and a paired device
   that sees its Mac's id on the LAN dials it there. No relay is involved,
   and this works with the internet down.
2. **The Mac can be dialled.** The Mac has a global IPv6 address, or its
   router mapped a port for it by PCP, NAT-PMP or UPnP, or the owner
   forwarded one by hand. The phone dials it from anywhere.
3. **Hole punched.** Both ends are behind NAT. The relay's address
   observation tells each side where the other's packets come from, and both
   send at once.
4. **Relayed.** Everything else: carrier-grade NAT on both ends, UDP
   blocked, a NAT that gives every destination a new port. HTTPS or
   WebSocket on 443.

Paths 1 to 3 are QUIC over UDP between the two devices. Path 4 is the same
encrypted session, wrapped and carried by the relay.

**Connect quickly, then upgrade.** A connection opens on whichever path
answers first, usually the LAN or the relay. If that was the relay, the
tunnel keeps trying the direct paths in the background and moves onto one
when it works. QUIC can carry a connection across a change of path, so
requests in flight survive the move. When the direct path dies, for example
when the phone leaves the Wi-Fi, traffic drops back to the relay and the
search starts over. iroh's docs say it does all of this already. Phase 0
and Phase 1 check that it holds for a streaming chat reply and a voice-note
upload, not just for a ping.

**Every path is equally secure.** The encryption runs end to end and is bound
to the node key on every path. A direct path is never the weaker one, and
the LAN path is not plain HTTP. Pinning, the device table and Kvad's own
sign-in apply the same way on all four. A direct path does show each device
the other's IP address. Both devices belong to the owner, so nothing is
lost. The relay learns less: once a connection goes direct, the relay saw it
being set up but not its timing or its volume.

**Discovery on the LAN.** The cluster already registers `_kvad._tcp` for
workers through `mDNSResponder` (`crates/llm/src/cluster/discover.rs`). The
remote identity gets a record type of its own, `_kvad-remote._udp`, with the
node id and protocol version in TXT and nothing secret. It is kept apart so
that a cluster coordinator never mistakes a remotely reachable Mac for a
worker. iOS has Bonjour and Android has NSD, so the phone can browse for it
without a library.

**Address hints.** The pairing link and QR code carry the Mac's LAN
addresses and any public address it knows, beside the relay URL (see the
link format below). After pairing, each side keeps the other's last
addresses that worked. A device that comes back to the network can then
dial straight away, before mDNS or the relay has answered.

**Pairing without a relay.** When the phone reads the QR code on the same
LAN, the whole pairing can run over the LAN path. The Mac checks the token
and runs the PAKE itself. The relay's pairing broker is only needed when the
phone cannot reach the Mac directly. The token is single use whichever path
presents it.

**No relay at all.** An owner who runs no server still gets the tunnel on
the LAN, and gets it from outside too when the Mac can be dialled at a
stable address: IPv6, a forwarded port, or a mapped port behind dynamic DNS.
Without a relay they give up the fallback and the rendezvous. If the Mac's
public address changes, the phone has no way to learn the new one, and hole
punching needs a third party to observe addresses. So the relay is optional
and recommended. That also answers one of the open questions: a Mac that can
be dialled does not need to act as its own relay. Relaying through itself
would be the direct path with extra steps.

**Configuration.**

```toml
[remote]
relays = ["https://relay.example.org"]
direct = "auto"   # "auto": every path; "lan": LAN only, else relay; "off": relay only
port = 5824       # the UDP port for direct QUIC; fixed so it can be forwarded by hand
map_port = true   # ask the router for a mapping by PCP, NAT-PMP or UPnP
```

`direct = "off"` is for debugging and for an owner who wants the Mac to
answer no packet from the internet that did not come through the relay.
`"lan"` keeps the LAN path and refuses direct connections from outside.
`map_port` is on by default because a mapped port is what makes path 2 work
for most home routers. What it exposes is one UDP port that drops every
handshake not bound to a paired key before reading any HTTP. The mapping is
released when the server stops. Whether that default is right is an open
question for Phase 0.

**Seeing which path.** `kvad remote status` and the device list in the web
UI show each connected device's current path (`lan`, `direct`, `punched`,
`relay via <url>`) and its round-trip time. `kvad --remote -v` prints path
changes as they happen. Otherwise "it's slow today" cannot be diagnosed.
`trait Tunnel` exposes the current path as a value the caller can watch, so
none of this names iroh.

## Pairing and the handshake

The Mac's node id is its public key. Knowing it is not enough to connect.

**Starting a pairing.** `kvad remote pair` in the terminal, or a button in the
web UI's settings. The Mac asks its relay for a pairing slot and shows:

- a QR code, in the terminal and in the UI,
- the same thing as a link,
- a six-digit code, shown only on the Mac.

The link carries the node id, the relay URL(s), any direct address hints,
the single-use token and a protocol version:

```
kvad://pair?node=<ed25519-pubkey>&relay=https://relay.example.org&addr=192.168.1.20:5824&addr=[2001:db8::20]:5824&token=<one-time>&v=1
```

The `addr` hints are optional and so is `relay`, but a link must carry at
least one of the two.

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

- *Relay down.* The client tries the next relay, and also the direct paths
  it knows: the LAN, then the last addresses that worked. A connection that
  had already gone direct does not notice. Only when every relay and every
  direct path has failed do the CLI and the UI say so, and they show what was
  tried and how each attempt failed.
- *Direct path lost.* The phone changed networks or a NAT mapping expired.
  The connection moves back to the relay without dropping the request in
  flight, and the search for a direct path starts again. With no relay
  configured, the connection drops and the client says the Mac is out of
  direct reach.
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
mobile network, a corporate guest Wi-Fi, a hotel. For each: which of the four
paths is reached; does the Mac's router map a port (PCP, NAT-PMP, UPnP); does
the phone get public IPv6; does hole punching succeed; how long a connection
that opened on the relay takes to go direct; what latency the relay adds over
the direct path; and whether a streaming reply survives the phone moving from
Wi-Fi to mobile data. A scratch tool built on iroh is enough. The table this
produces replaces every guess above, including the `map_port` default.

**Phase 1: a tunnel on one machine.** `trait Tunnel` with the current path as
a watchable value, an iroh-backed implementation, `kvad-serve` accepting a
tunnelled connection and handing it to the router, `kvad --remote kvad://...`
with address hints. Tests: the existing API tests run through the tunnel on
loopback and pass unchanged, once with `direct = "off"` through a local relay
and once direct with no relay configured. Then a connection that opens on the
relay and goes direct, after which the relay is killed and the stream carries
on. Then the reverse: the direct path is blocked mid-stream and the reply
finishes over the relay with no token lost.

**Phase 2: identity, pairing and the LAN.** `remote.key`, `kvad remote
pair`, the link format with its hints, token lifetime, the PAKE, the
`devices` table and its API key, `kvad devices`, the `_kvad-remote._udp`
record and `kvad remote status`. Pairing is checked on the Mac itself, so it
works over the LAN with no relay. Tests: a good pairing, over the LAN and
through a relay; an expired token; a used token, including one first used on
the other path; a wrong code; a revoked device; a device paired to one
account cannot act as another.

**Phase 3: `kvad-relay`.** The crate, its config, access control, health,
metrics, the self-test. Docker image and Compose file first. Tests: two
processes that can only reach each other through it; two that can punch
through it and then keep talking after it stops; and a third that is
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
- ~~Should a Mac be able to act as its own relay when it is reachable?~~
  Answered under *No relay at all*: a Mac that can be dialled is reached
  directly, and relaying through itself would add nothing.
- Is `map_port = true` the right default, or should mapping a port on the
  owner's router be something they turn on? Phase 0's table decides how much
  path 2 is worth.
- Should the phone prefer the LAN path even when a direct internet path is
  faster? That happens rarely, for example on a busy Wi-Fi with good 5G
  beside it. Measured RTT is the obvious tiebreak, but LAN-first keeps
  working when the internet goes down.
- What does the phone do while the Mac is asleep: record and sync later, or
  refuse? "Sync later" turns a networking problem into a queue, which is the
  easier problem.
- Does the relay need rate limits per node from day one, or only once
  somebody runs a shared one?

## What it buys

The TLS section of the authentication guide stops having to send people to a
reverse proxy or a VPN. A user who runs models locally for privacy can use
them from anywhere, through infrastructure they own, that cannot read what
passes through it, and pair a phone by pointing it at a QR code. At home, and
wherever the network allows it, the phone talks straight to the Mac, and the
relay is only the fallback.
