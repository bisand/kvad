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
dials out. Where the Mac has a global IPv6 address, the phone can often reach
it directly without punching anything. Phase 0 should find out how often that
is true. Kvad never asks the router to open a port (see *No ports opened in
the router*).

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
2. **The Mac can be dialled.** The Mac has a global IPv6 address that its
   router's firewall lets through, or the owner forwarded a port by hand.
   The phone dials it from anywhere. Kvad does not create this path. It only
   uses one that already exists.
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
stable address: IPv6, or a port they forwarded themselves, behind dynamic
DNS if need be.
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
```

`direct = "off"` is for debugging and for an owner who wants the Mac to
answer no packet from the internet that did not come through the relay.
`"lan"` keeps the LAN path and refuses direct connections from outside.

**No ports opened in the router.** Kvad never asks the router for a port
mapping: no PCP, no NAT-PMP, no UPnP. There is no setting to turn it on.
The owner's router configuration is theirs, and a program that opens holes
in it without asking every time is one they cannot audit. That costs part of
path 2: a Mac behind an IPv4 NAT is dialled directly only if the owner
forwarded `port` themselves. Hole punching (path 3) is unaffected. It opens
nothing in the router's configuration. It uses the short-lived entry the NAT
creates for any outgoing packet, the same one every video call uses, which
expires when the traffic stops.

The transport library must be held to this too. iroh maps ports by default,
so the iroh-backed `Tunnel` turns that off, and a Phase 1 test checks that
no PCP, NAT-PMP or UPnP request ever leaves the machine.

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

## Security

People who run models locally for privacy also think about security. Every
choice here is made for the safest default, and no setting makes the
cryptography weaker. Where safety costs a path, the path is the one that
goes.

**Who might attack, and what they get.**

| Who | What they can see or do | What stops the rest |
|---|---|---|
| The relay's operator, or someone who took over the server | Ciphertext, node and device ids, IP addresses, timing and volume. They can drop or delay traffic. | Encryption is end to end and bound to keys the relay never holds. The relay keeps nothing worth stealing. Self-hosting keeps even the metadata with the owner. A connection that goes direct shows the relay less. |
| Someone on the path: hotel Wi-Fi, an ISP, a carrier | The same as the relay. They can block UDP. | Both ends are pinned, so a man in the middle cannot complete a handshake. Blocked UDP falls back to the relay. |
| A scanner on the internet hitting the Mac's UDP port | At most one TLS alert. Never the node's key. | See *An unauthenticated surface of zero* below. The port exists only while `direct` is not `"off"`, and Kvad never opens one in the router. |
| Someone who got hold of a pairing link | One try. | The token is single use and lives ten minutes. A link also needs the PAKE code, which only the Mac's screen shows. A QR code read off the Mac's own screen never travels at all. |
| Someone on the same LAN | That a Kvad remote is present. | The mDNS record carries no stable id (see below). Pinning stops them from posing as the Mac. |
| Whoever has a lost or stolen phone | What the device was allowed to do, until it is revoked. | The device's key is kept where the OS makes it hardest to export, and the app locks. The device's key is scoped to what it was paired for. `kvad devices rm` revokes the pin and the key together, from the Mac. |
| An adversary who records traffic today to decrypt it later on a quantum computer | Nothing, if the key exchange holds. | Every connection uses the hybrid X25519MLKEM768 exchange, peer QUIC and relay HTTPS alike. Ed25519 signatures are not post-quantum, but they only authenticate at connection time, so a recording gives nothing to break later. |
| A compromised dependency | Potentially everything. | Few dependencies, pinned by `Cargo.lock`. iroh is built without its default features, so its port mapper is not even compiled in. Audits in CI (below). |

**An unauthenticated surface of zero.** Nothing on the Mac answers an
unpaired key with anything but a refusal, and nothing is parsed before the
handshake has proved who is calling.

- *A knock before the handshake.* The ALPN that a connection must offer
  carries a value from pairing, and it changes with each pairing. When the
  first packet shows a different ALPN, the Mac sends nothing back. When the
  first packet does not show the ALPN, rustls refuses a wrong one while it
  reads the ClientHello. That is one alert, sent before a ServerHello, so the
  node's key is never sent to a stranger. (The ClientHello usually spans two
  packets, because the post-quantum key share is large. That is why the
  first packet often cannot show the ALPN. `tools/relay-probe` found this
  out.) The knock hides the Mac from scanners, not from someone on the
  path: QUIC's first packets can be decrypted by anyone who sees them, so a
  watched connection gives the knock away. Against that observer, the
  pinning holds.
- *Refused at the handshake.* A key that is not in `devices` is rejected in
  the hook that runs as the handshake completes. Its HTTP is never read.
- *Backoff.* An address whose handshake was refused is ignored for a while
  afterwards. Only a few connections are served at once, a few more than a
  household of devices could need.
- *No 0-RTT.* Data sent in 0-RTT can be replayed by anyone who recorded it,
  so the tunnel never accepts it, whatever it would save in latency.

**No fallback to anything weaker.** There is no classical key exchange
between two Kvad peers, no plain HTTP to a relay other than on loopback, and
no protocol version below the minimum. An attacker cannot steer a peer
towards a fallback that does not exist. A version mismatch is refused, as
*Failure* says, and the refusal names both versions.

**Keys.**

- `remote.key` on the Mac: owner-only file in an owner-only directory, made
  with those permissions from the start, never on argv, never logged, and
  refused if anyone else can read it or write to its directory. Keeping it
  in the macOS Keychain instead is worth doing once the CLI and the server
  can both reach it without a prompt.
- On the phone: the Secure Enclave cannot hold an Ed25519 key, and
  Android's hardware keystore can only on recent versions. Where the
  hardware cannot hold it, the device key is wrapped by a key the hardware
  does hold (P-256 in the Secure Enclave, or a Keystore key). It is kept to
  this device only, never synced and never backed up.
- Rotation: `kvad remote rotate` makes a new node key and moves every pin
  over on the next connection of each device, so that a key that might have
  leaked can be retired without pairing again.

**Least privilege for devices.** Pairing asks what the device is for, and
its API key gets that scope. A phone paired for transcription can transcribe
and nothing else. Account and device management (creating users, minting
keys, pairing further devices) is not reachable through the tunnel unless
the owner turns it on for that device. A lost phone is then a lost
microphone, not a lost admin account. A key a device makes for a program of
its own (`kvad keys add` behind `kvad tunnel`) is not a way round that: it is
the device's, with the device's limits, and goes when the device does.

**The LAN record.** `_kvad-remote._udp` does not carry the node id in TXT.
It carries a short tag derived from a secret shared at pairing and the
current hour. A paired device recognises its Mac; a stranger sees only that
some Kvad is there, and cannot follow the Mac from one network to another.

**The relay.**

- It binds what its config names and nothing else. Metrics and health go on
  loopback unless the config says otherwise. (`iroh-relay --dev` binds every
  interface, metrics port included, so the default has to be stated, not
  inherited.)
- Rate limits per IP and per node from day one, not "once somebody runs a
  shared one": a relay on the internet is attacked whether it is shared or
  not.
- Logs carry no addresses by default and no content ever, since the relay
  has none.
- The image is distroless and runs as a non-root user on a read-only
  filesystem. It needs no capabilities beyond binding its ports.

**Logs everywhere.** No token, key, pairing code or knock value is ever
written to a log or a terminal other than where the owner asked for it,
such as the QR code. Ids are shown short.

**In CI.** `cargo deny` for licences and advisories, `cargo audit` on the
lockfile, and a test that builds the tunnel and fails if `portmapper` (or
any UPnP crate) appears in the tree. Phase 1 adds tests for the refusal
paths: an unknown key, a wrong knock, an old protocol version, a 0-RTT
attempt, and a classical-only ClientHello. Each must be refused, and none
may be answered with the node's key.

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

**Relays are not open to the world.** A relay serves only the servers' node
ids on its list, and the devices those servers vouch for (see Phase 3's
*vouchers*). A public relay that anyone can use is a bill and an abuse report
waiting to happen.

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
paths is reached, with port mapping off; does the phone get public IPv6, and
does the Mac's router let inbound IPv6 through; does hole punching succeed;
how long a connection that opened on the relay takes to go direct; what
latency the relay adds over the direct path; and whether a streaming reply
survives the phone moving from Wi-Fi to mobile data. A scratch tool built on
iroh is enough: `tools/relay-probe`. The table this produces replaces every
guess above.

*So far, on one machine (2026-10-17, iroh 1.3.0, an M5 Mac with Tailscale
running, a local `iroh-relay --dev`).* These runs test the tool and the
mechanisms, not networks. The field rows are still to do.

| Run | Opened on | Then | Median RTT |
|---|---|---|---|
| No relay; ticket with direct addresses | the tailnet address (100.64/10) | LAN after 0.25 s | 0.4 ms on the LAN |
| Relay only (no UDP socket at all) | relay | stays on the relay | 0.4 ms |
| Relay, ticket with no direct addresses | relay | the tailnet address after 0.25 s | 0.9 ms |
| As above, relay killed 3 s in | relay | carried on direct for 9 more seconds with no ping lost | 0.9 ms |
| An unpaired key | refused at the handshake | — | — |
| A wrong knock | one TLS alert (120, no application protocol) | — | — |

Found on the way: with a post-quantum key share the ClientHello spans two
packets, so a knock checked on the first packet alone would refuse every
honest client (see *Security*). And iroh's path choice is not "LAN first":
in one run it stayed on the tailnet address at 0.9 ms with the LAN beside
it. Kvad's own `PathSelector` decides that, per the open question below.

*The first field run (2026-10-18).* The listener was the MacBook Pro at
home: Norwegian broadband, IPv4 only, behind the home router. The dialler
was a MacBook Air on an iPhone's Personal Hotspot over mobile data, held to
the hotspot interface with `relay-probe --bind`. The relay was
`tools/kvad-relay` at `relay2.kvad.eu`, a k3s node in a Norwegian data
centre, behind Traefik's TLS passthrough. Its TLS negotiated X25519MLKEM768
with a client that offered only that.

| Run | Opened on | Then | Median RTT | p90 | Worst | Lost |
|---|---|---|---|---|---|---|
| Relay only, 40 s | relay, in 0.85 s | stayed | 62.5 ms | 96.6 ms | 302 ms | 0 of 156 |
| All paths, 60 s | relay, in 0.20 s | **no direct path** | 57.3 ms | 68.0 ms | 94.1 ms | 0 of 238 |

- *Address discovery worked.* The home Mac learnt `81.166.239.243:5824`,
  its own port kept by the router, and the Air learnt the carrier's
  `77.18.34.152:41523`. Neither learnt whether its NAT maps each
  destination to a new port: that needs two relays to compare, and there
  was one.
- *Hole punching did not succeed* in 60 s. The likely cause is that
  Personal Hotspot is a NAT of its own, in front of the carrier's
  carrier-grade NAT, and two NATs in a row rarely punch. An app on the
  phone itself is behind the carrier's NAT alone. That is the run that
  matters. Kvad for iPhone's Network test runs the probe's dialler from the
  phone for it (`apps/phone/README.md`, *The run over mobile data*).
- *The relay is not optional for a phone on mobile data.* Here it carried
  every packet, at about 60 ms, with no ping lost.
- *IPv6 could not help.* The carrier gave the Air a global IPv6 address,
  but the home connection has none, and neither has the relay's node, so
  every path was IPv4.
- *iroh reports "behind a captive portal" falsely.* It checks for a portal
  over the relay's plain HTTP on port 80, which kvad-relay keeps on
  loopback. Kvad's own status should not repeat that line.
- *The first relay host could not be used.* At `postbud`'s provider,
  inbound traffic reached the VM on TCP 25 alone. Rules for 443 and 7842,
  an ACCEPT policy, the provider firewall switched off and a stop and start
  of the VM changed nothing, and a capture on the VM's interface saw no
  other port arrive. Before deploying to a host, a capture there while
  connecting from outside answers in a minute what the panel cannot.

**Phase 1: a tunnel on one machine.** `trait Tunnel` with the current path as
a watchable value, an iroh-backed implementation, `kvad-serve` accepting a
tunnelled connection and handing it to the router, `kvad --remote kvad://...`
with address hints. Tests: the existing API tests run through the tunnel on
loopback and pass unchanged, once with `direct = "off"` through a local relay
and once direct with no relay configured. Then a connection that opens on the
relay and goes direct, after which the relay is killed and the stream carries
on. Then the reverse: the direct path is blocked mid-stream and the reply
finishes over the relay with no token lost. And through all of it, no port
mapping request leaves the machine. A WebSocket opened through the tunnel
streams both ways at once, and its events back are not held up by an upload
filling the other direction.

*Built, 2026-10-18.* `crates/tunnel` (`kvad-tunnel`) is the tunnel:
`Server`, `Client`, `blocking::Tunnel` for the CLI, `Address` and `Path`,
with iroh inside and named nowhere outside. It is a crate with no iroh in
its signatures rather than a `trait Tunnel`, which gives the same freedom to
replace it with less ceremony. `kvad-serve` serves it as a second
`axum::serve` listener into the same router, from a `[remote]` section that
is refused with `auth.mode = "none"`, even with `--insecure`. `kvad --remote
kvad://…` dials it, and every HTTP connection is a stream handed to ureq as
one end of a Unix socket pair, so the CLI opens no port. `kvad remote id`
prints the device's id for the server's `allow`, and `kvad remote status`
the path.

What it does that the plan did not say:

- *A hello.* QUIC lets a client open a stream before it learns that the
  server refused it, so a refused device saw only an empty reply. The server
  now writes `kvad-tunnel/1` on a stream of its own once it has let a device
  in, and a client waits for it. A refusal is an error at connect, naming the
  fix, and a different protocol version is refused with both versions named.
- *No 0-RTT.* Neither side uses iroh's 0-RTT calls, so nothing replayable is
  accepted.
- *Owner checks by uid.* Key files are checked against the process's own
  uid, not `$HOME`'s owner, since a service's data directory need not be
  under a home.

Tested:

- `crates/tunnel/tests/tunnel.rs`, on loopback:
  - HTTP direct with no relay, several streams over one connection.
  - HTTP through a relay in the test process with `direct = "off"`.
  - An unlisted key refused at connect.
  - A wrong knock refused.
  - A 4-second streamed reply that opened on the relay, went direct, and
    finished whole after the relay was shut down mid-reply.
  - The lockfile holds no port-mapping crate.
- By hand, a scratch kvad-serve with `auth.mode = "local"` and the CLI:
  - `auth status`, a refused unsigned `ls`, the first account made with
    `auth setup` and a key kept with `auth login`, all through the tunnel.
  - Then `ls` and a streamed `run`, direct on the LAN and then relay only
    through `relay.kvad.eu`.

Not yet done:

- The existing API tests run through the tunnel.
- The direct path blocked mid-stream and the reply finishing over the relay.
  That needs a firewall rule in the test, or a test transport that can drop
  a path.
- A WebSocket through the tunnel, since nothing in kvad-serve speaks one yet.
- `direct = "lan"`, which needs Kvad's own path selection.

**Phase 2: identity, pairing and the LAN.** `remote.key`, `kvad remote
pair`, the link format with its hints, token lifetime, the PAKE, the
`devices` table and its API key, `kvad devices`, the `_kvad-remote._udp`
record and `kvad remote status`. Pairing is checked on the Mac itself, so it
works over the LAN with no relay. Tests: a good pairing, over the LAN and
through a relay; an expired token; a used token, including one first used on
the other path; a wrong code; a revoked device; a device paired to one
account cannot act as another.

*Built, 2026-10-18 (pairing by QR code or link, without the PAKE).*

- **The pairing lane.** The tunnel's allow list is a live `Allowed`. While
  a slot is open, the handshake lets an unknown key in to one request and
  one answer, with a hello that says so, and no HTTP. Removing a device
  closes its live connection.
- **Slots.** kvad-serve keeps them in memory only: a 256-bit token that lives
  ten minutes and is burnt on first use, right or wrong. Presenting it again
  is told "already used". A good pairing adds a row to `devices` (migration
  022) and makes an API key for the device.
- **What a device's key may do.** It works only through the tunnel and only
  from its own node, checked against the `ConnectInfo` the tunnel's listener
  attaches to each request. It acts as `user` unless the slot granted
  `admin`, and a device without `admin` cannot open a slot.
- **Commands.** `kvad remote pair` shows the link and a QR code in the
  terminal, `kvad remote join LINK` pairs and keeps the key with the
  server's whole address, and `kvad devices` lists and removes devices. The
  server's TCP port is how the first account pairs its first device.
- **Fewer refusals turned into a wait.** A refused handshake now starts a
  backoff only on the third within a minute, so that a device that tried
  before its slot was open gets its retry.

Tested:

- *The tunnel* (`crates/tunnel/tests/tunnel.rs`): an unknown key refused
  with no slot open; reaching only the pairing lane while one is; refused
  as "not paired" by a normal connect; told "already paired" once let in;
  and a removed device cut off mid-reply.
- *Slots and devices* (`remote.rs`, `devices.rs`): a good pairing, a used
  token, an unknown token, an expired one, a refused token burnt too, the
  lane closing as slots expire, admin granted only when asked, and other
  protocol versions refused.
- *Auth* (`auth/tests.rs`): a device's key accepted from its own node and
  lowered to `user`; refused on the TCP port and from another device.
- *By hand*, with two identities against a scratch server in local auth
  mode: unpaired and refused; `pair` (a 31-line QR code), `join`, then
  commands by `kvad://<node id>` alone; an admin account's device held to
  `user`; the spent link refused to a second device; the device's key
  refused on the TCP port; a device refused a slot; `devices rm`, and the
  device refused at once.

Not yet done:

- *The PAKE and the six-digit code.* A link that travels anywhere but the
  server's own screen is as good as a password for its ten minutes.
- *The `_kvad-remote._udp` record.*
- *Pairing through the relay.* The relay's allow list is a file
  (`tools/kvad-relay/deploy/allow.local`), so a newly paired device gets
  through the relay only once its id is added there. Phase 3 is the relay
  learning a node's devices from the node.
- *The web UI's pairing screen and device list*, which is Phase 4.

**Phase 3: `kvad-relay`.** The crate, its config, access control, health,
metrics, the self-test. Docker image and Compose file first. Tests: two
processes that can only reach each other through it; two that can punch
through it and then keep talking after it stops; and a third that is
refused.

*Built, 2026-10-18 (who the relay serves, wherever they are).* The crate,
its TLS and its Docker image came first, in Phase 0. What Phase 3 adds is
the relay letting a server's devices through without anybody keeping a list
of them.

- **Vouchers, not lists.** A voucher is a string a server signs with its
  node key: "this device may use the relays that serve me, until then".
  `kv1.<node>.<device>.<expires>.<signature>`, in `kvad_tunnel::Voucher`.
  A device shows it to a relay as the relay connection's auth token.
  kvad-relay lets in the node ids on its list, and any device with a valid
  voucher from one of them.
- **Statelessness.** The relay learns nothing and keeps nothing. Restarting
  it, running several, and devices or servers moving to other networks
  change nothing.
- **What a voucher's lifetime means.** A device gets one when it pairs,
  good for thirty days, and asks for a fresh one through the tunnel
  (`GET /api/remote/voucher`) on its own once half is spent. Unpairing
  shuts a device out of its server at once, as before. Its voucher still
  gets it onto the relay until it runs out, where it can reach nothing that
  would let it in.
- **Pairing through the relay.** A pairing link carries a voucher for any
  device, good for the slot's ten minutes and no more: a relay refuses an
  any-device voucher that lives longer than fifteen. A phone on mobile data
  can pair from the QR code on the Mac's screen.

Why not have devices report to the relay when their network changes: they
already do, in effect. Every device and server holds a connection to its
relay, keyed by its node id. A machine that moves reconnects under the same
key, and anything dialling it by that key reaches it through the relay
again, then finds a direct path again by exchanging its new addresses. What
an address's `addr=` hints say only saves a round trip, and stale ones cost
nothing. The one thing worth adding is on the phone: telling the endpoint as
soon as iOS reports a network change (`Endpoint::network_change`, from
`NWPathMonitor`), so that it reacts at once rather than when it notices
(`docs/phone-plan.md`, Phase 4).

Tested:

- *Vouchers* (`kvad-tunnel`): one for the device it names, while it lives;
  refused for another device, from a node the relay does not serve, after a
  forged expiry, or once expired; any-device ones capped at fifteen
  minutes.
- *A relay in the test process, with kvad-relay's rule*: a device on no list
  gets through with a voucher, a newcomer reaches the pairing lane with a
  link's, and nothing gets through without one.
- *Through `relay.kvad.eu`, by hand*:
  - A scratch server with `direct = "off"` was on the relay's list, and the
    phone's key was not.
  - The phone paired through the relay with the link's voucher, and ran
    commands through it with its own.
  - With its voucher taken away, the relay refused it, and the error says
    so.
  - Given a nearly spent voucher, the next command swapped it for a fresh
    thirty-day one.

Not yet done: the relay's health and metrics endpoints, its self-test, and
rate limits per node rather than per connection.

**Phase 4: the client side.** `kvad tunnel`, the web UI's pairing screen and
device list, and the CLI over a real relay from a real phone network.

**Phase 5: packaging.** Helm, systemd, `install.sh --relay`, Terraform
modules, Fly. Each target gets a short page on kvad.eu and a CI job that at
least builds it.

**Phase 6: the reason for all of it.** A phone client that records and sees
the words come back while it records. The app is its own plan,
`docs/phone-plan.md`: a Slint app in Rust, a client for all of Kvad. The file endpoint,
`POST /v1/audio/transcriptions`, is built (`docs/speech-plan.md`). So is
the live one, `/v1/realtime`, under *Live, from the phone*: a WebSocket with
audio going up and settled words and a replaceable partial coming down, a
word shown about 0.4 s and settled about 1.6 s after it is spoken on small
at gpu-q8.

What it asks of this plan:

- *WebSocket upgrades through the tunnel.* The tunnel carries HTTP, so it
  carries an upgrade. Phase 1 tests it, since a tunnel that buffers whole
  requests would pass every other API test and break this one.
- *A return channel that nothing holds back.* The session is one QUIC
  stream with two independent directions. The phone's audio upload, which
  can fill a slow uplink, must not delay the small events coming back.
  Phase 1 measures that: an event's latency with an upload saturating the
  link, on the relay and direct.
- *Surviving a change of path.* A sentence that starts on the Wi-Fi and ends
  on 5G must arrive whole. QUIC carries the connection across, and the
  session's resume (sample offsets, a minute's grace) covers the case where
  it does not.
- *Recording never depends on the link.* The phone records to its own
  storage first and streams from there. With no path at all, or the Mac
  asleep, it keeps the file and sends it later through the file endpoint.

## Open questions

- Is `kvad-relay` a fork-free repackaging of `iroh-relay`, or its own binary
  that embeds it? Phase 1.
- Does the pairing broker belong in the relay, or should pairing ride on the
  same relay protocol with no extra endpoint at all?
- ~~Should a Mac be able to act as its own relay when it is reachable?~~
  Answered under *No relay at all*: a Mac that can be dialled is reached
  directly, and relaying through itself would add nothing.
- ~~Should Kvad map a port on the owner's router?~~ No, decided 2026-10-17;
  see *No ports opened in the router*.
- Should the phone prefer the LAN path even when a direct internet path is
  faster? That happens rarely, for example on a busy Wi-Fi with good 5G
  beside it. Measured RTT is the obvious tiebreak, but LAN-first keeps
  working when the internet goes down.
- ~~What does the phone do while the Mac is asleep?~~ Records, and sends the
  file later: see Phase 6.
- ~~Does the relay need rate limits from day one?~~ Yes; see *Security*.

## What it buys

The TLS section of the authentication guide stops having to send people to a
reverse proxy or a VPN. A user who runs models locally for privacy can use
them from anywhere, through infrastructure they own, that cannot read what
passes through it, and pair a phone by pointing it at a QR code. At home, and
wherever the network allows it, the phone talks straight to the Mac, and the
relay is only the fallback.
