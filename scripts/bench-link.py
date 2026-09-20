#!/usr/bin/env python3
"""Measure a link the way a layer split will use it, before there is an engine.

Phase 0 of `docs/cluster-plan.md` wants round trip and throughput for 32 KB
and 2 MB frames between two real machines, so that the link table stops being
arithmetic. `kvad cluster ping` will do that properly, over `wire.rs`. This is
the stopgap that runs today, on two stock Macs with nothing installed on
either of them.

    # on the far machine
    scripts/bench-link.py --listen

    # on this one
    scripts/bench-link.py --find                     # what is on the cable
    scripts/bench-link.py --connect 'fe80::1c%en12'  # then measure it

It measures a *round trip*, not one-way bandwidth, because that is what a
layer split does: the coordinator hands a stage its hidden state and waits.
iperf3 answers a different and easier question -- it would report this link's
ceiling and say nothing about the 1.3 ms a 32 KB hop pays before any of its
bytes move, which for decode is the whole cost.

It prints what the link *is* before it prints a number, because of how this
rig started. A USB-C charging cable carries USB 2; macOS falls back to USB NCM
without saying so; the result answers ping6, carries TCP and accepts SSH at a
hundredth of the speed the ports are capable of. Nothing a socket can observe
tells that apart from Thunderbolt, so a number recorded without identifying
the link is a number about nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import time

# `kvad worker --listen 0.0.0.0:7420` in the plan, so the same here.
PORT = 7420

# The two sizes Phase 0 has to report. A decode hop is `n_embd` floats, which
# is 8192 x 4 = 32 KB for a 70B; a prefill chunk is `KVAD_PREFILL_CHUNK` rows
# of that, which is 2 MB at the default of 64.
DECODE_HOP = 32 * 1024
PREFILL_CHUNK = 2 * 1024 * 1024

# A sweep rather than those two alone. Every "X is slower than Y" in this
# repository has turned out to be a crossover, and the crossover that matters
# here is where a frame stops paying latency and starts paying bandwidth.
SIZES = [1 << 10, 1 << 12, 1 << 14, DECODE_HOP, 1 << 17, 1 << 19, PREFILL_CHUNK, 1 << 23]

HEADER = struct.Struct("!cQ")


def sh(*cmd: str, timeout: float = 30.0) -> str:
    """Run a command, returning its output or "" if it fails for any reason.

    Everything this is used for is a diagnostic, and a machine that answers
    none of them is still a machine worth measuring, so nothing here is
    allowed to be fatal.
    """
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout


# --- what the link actually is ------------------------------------------------


def identify(iface: str) -> str:
    """Name the physical link behind an interface, and show the evidence.

    `ifconfig` is no use for this: a Thunderbolt bridge and a charging cable
    are both up, both `RUNNING`, both carrying TCP. These three checks
    separate them, and the registry path is the one that cannot be misread --
    the port an interface hangs off is named `...-port-hs` for USB 2.0 High
    Speed, and `hs` is 480 Mb/s no matter what the cable's connector implies.
    """
    ioreg = sh("ioreg", "-w0", "-p", "IOService", "-n", iface, "-r", "-t")
    bridge = sh("ifconfig", "bridge0")
    tb = sh("system_profiler", "SPThunderboltDataType", timeout=90)

    # The ancestry, root first. The node directly above the interface is the
    # driver that owns it, which is the whole answer: AppleThunderboltIPPort
    # or AppleUSBNCM11Data.
    chain = re.findall(r"\+-o (\S+)\s+<class (\w+)", ioreg)
    names = [name for name, _ in chain]
    parent = ""
    if iface in names and names.index(iface) > 0:
        above = chain[names.index(iface) - 1]
        parent = f"{above[0]} ({above[1]})"

    # `usb-drdN-port-hs` is USB 2.0 High Speed and `-ss` is SuperSpeed. The
    # connector is the same shape either way, which is the entire problem.
    speed = ""
    for name in names:
        if "-port-hs" in name:
            speed = ", USB 2.0 High Speed: 480 Mb/s"
        elif "-port-ss" in name:
            speed = ", USB SuperSpeed"

    if "AppleThunderboltIP" in ioreg:
        kind = "Thunderbolt-IP"
    elif "AppleUSBNCM" in ioreg:
        kind = "USB NCM" + speed
    elif ioreg:
        kind = "neither Thunderbolt-IP nor USB NCM"
    else:
        kind = "unknown -- ioreg knows no interface by that name"

    # "Link Status: 0x100" ends in "Status:" too, so match the line and not
    # the word; counting the word says six ports on a machine that has three.
    status = [l.strip() for l in tb.splitlines() if l.strip().startswith("Status:")]
    live = sum(1 for l in status if "No device connected" not in l)
    member = f"member: {iface} " in bridge

    print(f"link on {iface}: {kind}")
    print(f"  ioreg           {parent or '(no parent node; is the cable in?)'}")
    print(f"  bridge0         {'a member' if member else 'not a member'}"
          f", {'inactive' if 'status: inactive' in bridge else 'active'}")
    print(f"  Thunderbolt     {live} of {len(status)} ports with a device")
    if kind.startswith("USB NCM"):
        print("  -> not Thunderbolt. Fine to build on, wrong to quote.")
    return kind


def own_link_local(iface: str) -> set[str]:
    """This machine's own fe80:: addresses on an interface, to discount them.

    A multicast probe hears itself first, every time.
    """
    out = sh("ifconfig", iface)
    return set(re.findall(r"inet6 (fe80::[0-9a-f:]+)%", out))


def is_cable(iface: str) -> bool:
    """Does this interface look like a cable between two machines?

    The signature is the absence of a routable IPv4 address -- none at all,
    or a self-assigned 169.254 -- because there is nothing on the far end of
    a cable handing out leases. A Wi-Fi or LAN interface has a real address
    and, on this network, twenty neighbours, none of which is what anyone is
    looking for here.
    """
    v4 = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", sh("ifconfig", iface))
    return all(addr.startswith("169.254.") for addr in v4)


def find(everything: bool = False) -> int:
    """Ask every cable who else is on it.

    `ff02::1` is all-nodes: on a link between two machines exactly one thing
    answers that is not us, and it answers whether or not either side has
    been configured, because a link-local address needs no configuring. This
    is the discovery the plan wants done properly by mDNS, done crudely
    enough to need nothing installed at the other end.
    """
    names = [n for n in sh("ifconfig", "-l").split() if n.startswith(("en", "bridge"))]
    looked, found = [], 0
    for iface in names:
        mine = own_link_local(iface)
        if not mine:                      # nothing to send from
            continue
        if not (everything or is_cable(iface)):
            continue
        looked.append(iface)
        out = sh("ping6", "-c", "2", "-i", "0.3", f"ff02::1%{iface}", timeout=15)
        for peer in sorted(set(re.findall(r"from (fe80::[0-9a-f:]+)%", out)) - mine):
            print(f"{iface}\t{peer}%{iface}")
            found += 1
    print(f"probed {', '.join(looked) or '(no candidate interfaces)'}"
          f"{'' if everything else '; --all to include Wi-Fi and the LAN'}",
          file=sys.stderr)
    if not found:
        print("nothing answered. The peer must be awake, and the cable present:",
              file=sys.stderr)
        print("an unplugged USB link does not go down, it stops existing.",
              file=sys.stderr)
    return 0 if found else 1


# --- the wire ------------------------------------------------------------------


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        buf += chunk
    return bytes(buf)


def serve(port: int) -> int:
    """Echo frames back. One connection at a time, which is the shape of a stage."""
    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("::", port))
    srv.listen(1)
    print(f"listening on [::]:{port}, IPv4 too; ^C to stop")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  <- {addr[0]}")
        try:
            while True:
                mode, n = HEADER.unpack(recv_exact(conn, HEADER.size))
                payload = recv_exact(conn, n)
                # 'E' is the round trip a stage boundary is; 'S' is one-way,
                # acknowledged once, for the bandwidth ceiling alone.
                conn.sendall(payload if mode == b"E" else b"\0" * 8)
        except (ConnectionError, OSError, struct.error) as err:
            print(f"     done ({err})")
        finally:
            conn.close()


def round_trip(conn: socket.socket, payload: bytes) -> float:
    """Milliseconds for `len(payload)` bytes to cross and come back."""
    start = time.perf_counter_ns()
    conn.sendall(HEADER.pack(b"E", len(payload)) + payload)
    recv_exact(conn, len(payload))
    return (time.perf_counter_ns() - start) / 1e6


def one_way(conn: socket.socket, payload: bytes) -> float:
    start = time.perf_counter_ns()
    conn.sendall(HEADER.pack(b"S", len(payload)) + payload)
    recv_exact(conn, 8)
    return (time.perf_counter_ns() - start) / 1e6


def sweep(conn: socket.socket, rounds: int) -> dict[int, list[float]]:
    """Walk every size once per round, not every round once per size.

    Load on these machines swings enough to move absolute timings by a
    quarter, so a size measured all in one stretch of time records that
    stretch rather than the link. Interleaving spreads each size over the
    whole run. The first round is discarded: it pays for neighbour discovery,
    for the window opening, and for the buffers growing to fit.
    """
    blobs = {n: os.urandom(n) for n in SIZES}  # not zeros, in case anything compresses
    samples: dict[int, list[float]] = {n: [] for n in SIZES}
    for r in range(rounds + 1):
        for n in SIZES:
            ms = round_trip(conn, blobs[n])
            if r:
                samples[n].append(ms)
        print(f"  round {r}/{rounds}{' (warm-up, discarded)' if r == 0 else ''}",
              file=sys.stderr)
    return samples


def fit(samples: dict[int, list[float]]) -> tuple[float, float]:
    """Least squares of median round trip against size: `rtt = a + 2*bytes/B`.

    `a` is what a hop costs carrying nothing, which is the number decode
    lives on, where 32 KB is too small to matter. `B` is what the link does
    once the frame is big enough to hide `a`, which is the number prefill
    lives on. Quoting either alone describes the link wrongly, which is most
    of why the table in the plan has two columns.
    """
    xs = [float(n) for n in SIZES]
    ys = [statistics.median(samples[n]) for n in SIZES]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0          # ms per byte, both directions
    intercept = my - slope * mx                # ms
    bandwidth = (2000.0 / slope) if slope > 0 else float("inf")  # bytes/s
    return intercept, bandwidth


def measure(host: str, port: int, rounds: int, stream_mb: int) -> int:
    iface = host.partition("%")[2].partition("]")[0]
    if iface:
        identify(iface)
    else:
        print(f"link on {host}: not a scoped link-local address, so not identified")
    print(f"  load            {sh('uptime').strip()}")
    print()

    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    family, socktype, proto, _, sockaddr = infos[0]
    conn = socket.socket(family, socktype, proto)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # or Nagle times itself
    conn.settimeout(30.0)
    conn.connect(sockaddr)

    samples = sweep(conn, rounds)
    stream = one_way(conn, os.urandom(stream_mb << 20))
    conn.close()

    print()
    print("| bytes | round trip, median | min | link rate, 2x bytes / rtt |")
    print("|---|---|---|---|")
    for n in SIZES:
        med, low = statistics.median(samples[n]), min(samples[n])
        rate = 2 * n * 8 / (med / 1000) / 1e6
        note = {DECODE_HOP: " (decode hop)", PREFILL_CHUNK: " (prefill chunk)"}.get(n, "")
        label = f"{n >> 10} KB" if n < (1 << 20) else f"{n >> 20} MB"
        print(f"| {label}{note} | {med:.3f} ms | {low:.3f} ms | {rate:.0f} Mb/s |")

    a, b = fit(samples)
    ceiling = stream_mb * (1 << 20) * 8 / (stream / 1000) / 1e6
    print()
    print(f"empty-hop latency   {a:.3f} ms      (fitted intercept)")
    print(f"fitted bandwidth    {b * 8 / 1e6:.0f} Mb/s   (from the slope)")
    print(f"one-way ceiling     {ceiling:.0f} Mb/s   ({stream_mb} MB in one direction)")
    print()
    print("Medians should sit on their minimums. Where they do not, this machine")
    print("was busy and the run is worth repeating rather than quoting.")
    return 0


def main() -> int:
    # Progress goes to stderr while the table goes to stdout, so that the
    # table can be piped straight into the plan. They interleave correctly
    # only if stdout stops block-buffering itself when it is not a terminal.
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", action="store_true", help="be the far end")
    ap.add_argument("--connect", metavar="HOST", help="measure the link to HOST")
    ap.add_argument("--find", action="store_true", help="list peers on every cable")
    ap.add_argument("--all", action="store_true", help="with --find, probe Wi-Fi too")
    ap.add_argument("--link", metavar="IFACE", help="identify one interface and stop")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--rounds", type=int, default=7, help="after one discarded warm-up")
    ap.add_argument("--stream-mb", type=int, default=64)
    args = ap.parse_args()

    if args.find:
        return find(args.all)
    if args.link:
        identify(args.link)
        return 0
    if args.listen:
        return serve(args.port)
    if args.connect:
        return measure(args.connect, args.port, args.rounds, args.stream_mb)
    ap.print_help()
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
