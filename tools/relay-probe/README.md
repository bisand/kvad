# relay-probe

Phase 0 of [`docs/relay-plan.md`](../../docs/relay-plan.md): on real
networks, which path two devices reach each other by (LAN, direct, a
carrier's NAT, the relay), how long the relay is needed before a direct
path takes over, and what each path costs in round-trip time.

It is a scratch tool, in a workspace of its own, so that nothing it pulls in
reaches Kvad's builds. It is built on iroh 1.3, as the plan proposes for the
tunnel.

## What it will and will not do

- It never asks the router for a port. iroh's `portmapper` is not compiled
  in: `cargo tree -i portmapper` finds nothing.
- It talks to no third party unless told to. There is no relay unless you
  name one with `--relay`, or `--n0` for n0's public relays, and no address
  lookup service at all.
- Both ends are pinned. The dialler checks the listener's key from the
  ticket. The listener refuses every key not given with `--allow`, at the
  handshake.
- The ALPN is random for each run. An attempt without it gets no answer, or
  a single TLS alert, and never the listener's key. An address that was
  refused is ignored for 30 s, and at most 4 connections are served at once.
- The only key exchange is X25519MLKEM768, so a relay must speak it too.
  `iroh-relay` does.
- Keys live in `~/.kvad-probe/secret.key`, readable only by you, in a
  directory only you can write to. They never appear on the command line or
  in the output. Use `--key FILE` to keep a second identity.
- The ticket says where the listener is: its LAN and public addresses. Send
  it only over a channel you would send that over.

## Build

```bash
cargo build --release --manifest-path tools/relay-probe/Cargo.toml
```

## A run

On the machine that will dial, usually a laptop:

```bash
tools/relay-probe/target/release/relay-probe id
```

On the Mac, with the id that printed:

```bash
tools/relay-probe/target/release/relay-probe listen --allow <id> --relay https://relay.example.org
```

It prints what the relay saw of its network, then a ticket. On the laptop,
run the next command, then paste the ticket and press return:

```bash
tools/relay-probe/target/release/relay-probe dial --secs 60
```

It prints each change of path as it happens, then a table of round-trip
times per path, the time to the first direct path, and the longest single
ping.

`--relay-only`, on both ends, uses no UDP socket at all: the relay's
baseline. With no `--relay`, only the ticket's direct addresses are tried,
which is the LAN case and the "no relay at all" case.

## Which relay

Hole punching needs a relay to see both ends' public addresses. You can run
your own on a small VPS: `cargo install iroh-relay --features server`, with
a domain name and its ACME or certificate config. Or pass `--n0` to use n0's
public relays. They would then see both machines' IP addresses and when they
talk, but not what they say. For a test on one machine, `iroh-relay --dev`
serves plain HTTP on port 3340, which the probe accepts only for localhost.
Note that it listens on every interface, its metrics port 9090 included.

## The field runs

For each network, the Mac stays at home on its usual network, and the
laptop goes elsewhere:

1. The same Wi-Fi as the Mac.
2. Mobile data: the laptop on the phone's hotspot (Telia, Telenor, Ice, or
   whichever carrier is at hand).
3. A guest or corporate Wi-Fi.
4. A hotel or café Wi-Fi.
5. A handover: start `dial --secs 120` on Wi-Fi, switch the laptop to the
   hotspot halfway, and look for the path change, and for whether the run
   ends without `stopped early`.

For each, note the table, the `network:` lines from both ends (public
addresses, and whether the NAT gives each destination its own port), and
where it opened and where it ended up. The results go in the Phase 0 table
in the plan.
