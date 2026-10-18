# relay-probe

Phase 0 of [`docs/relay-plan.md`](../../docs/relay-plan.md): on real
networks, which path two devices reach each other by (LAN, direct, a
carrier's NAT, the relay), how long the relay is needed before a direct
path takes over, and what each path costs in round-trip time.

It is a scratch tool, in a workspace of its own, so that nothing it pulls in
reaches Kvad's builds. It is built on iroh 1.3, as the plan proposes for the
tunnel. Its core is a library (`src/lib.rs`): the command is `src/main.rs`,
and Kvad for iPhone's Network test (`apps/phone`) dials with the same code.

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

Hole punching needs a relay to see both ends' public addresses. Kvad's own is
`tools/kvad-relay`, which `tools/kvad-relay/deploy/README.md` puts on a VPS,
at `https://relay.kvad.eu` for this repo's owner. The stock `iroh-relay` will
not do: it is built on ring, which has no post-quantum key exchange, and the
probe offers nothing else. For the same reason `--n0`, n0's public relays,
will refuse the probe if they run the stock binary, which has not been
tried. For a test on one machine, `kvad-relay --dev --allow FILE` serves
plain HTTP on 127.0.0.1:3340, which the probe accepts only for localhost.

## Held to one network

A machine on several networks at once (Wi-Fi, a cable, a tailnet) will find
the fastest of them, which is never the one being measured. `--bind IP`, once
for each address, binds the probe's UDP socket to those addresses alone.
Packets then leave from them only, and an answer to any other address has no
socket to arrive at, so no other path can form. For a laptop on a phone's
hotspot, bind the hotspot interface's addresses:

```bash
ipconfig getifaddr en0
```

```bash
tools/relay-probe/target/release/relay-probe dial --bind 172.20.10.2 --secs 60
```

Add `--bind` with the interface's global IPv6 address as well, if the carrier
gives one (`ifconfig en0 | grep inet6`), to see whether IPv6 makes the path
direct.

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

6. The iPhone itself on mobile data, with Wi-Fi off: the app's Network test
   is this command's dialler, and `apps/phone/README.md` has the steps.

For each, note the table, the `network:` lines from both ends (public
addresses, and whether the NAT gives each destination its own port), and
where it opened and where it ended up. The results go in the Phase 0 table
in the plan.
