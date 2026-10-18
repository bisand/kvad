# Deploying kvad-relay

To a single-node k3s, as on `postbud`: the image is imported straight into
k3s's containerd, and the pod runs on the host network so that the relay
binds the public address itself and sees clients' real addresses.

## Once, on the server: the firewall

The relay needs two ports open **inbound**. Outbound it needs only what a
server already has: DNS, Let's Encrypt, and replies to its clients.

```bash
ufw allow in on eth0 to 85.137.228.199 port 443 proto tcp comment 'kvad-relay'
```

```bash
ufw allow in on eth0 to 85.137.228.199 port 7842 proto udp comment 'kvad-relay address discovery'
```

Port 80 stays closed: the certificate comes by the TLS-ALPN challenge on
443, and the relay's plain-HTTP side listens on loopback only. If the
hosting provider has a firewall of its own in front of the VPS, the same two
ports need opening there too.

`relay.kvad.eu` must resolve to the server's address, `85.137.228.199`,
before the first start, or Let's Encrypt cannot validate it.

## Deploying

The allow list is `deploy/allow.local` (not committed): one endpoint id a
line, from `relay-probe id` on each machine that will use the relay. Then,
from `tools/kvad-relay`:

```bash
deploy/deploy.sh root@postbud.tarsier-bull.ts.net deploy/allow.local -i ~/.ssh/id_postbud_bogentech_no
```

It cross-compiles a static binary (`cargo zigbuild`), builds the image
(`docker`, distroless, nothing run inside), imports it over SSH, and applies
`deploy/k8s.yaml`. To change who may use the relay, edit the allow list and
run it again.

## Checking it

The logs should say how many keys are allowed, and where it listens:

```bash
ssh root@postbud.tarsier-bull.ts.net 'kubectl -n kvad-relay logs deploy/kvad-relay'
```

The first certificate takes a few seconds after the ports are open. Then
from anywhere, a probe run through it (see `tools/relay-probe/README.md`):
the listener's `network:` lines show its public address, which only the
relay's address discovery can have told it.

## What runs

- `hostNetwork`, ports 443/tcp and 7842/udp on `85.137.228.199` only.
  Tailscale keeps its own 443 on the tailnet address.
- Root inside the container, because containerd gives a non-root process on
  the host network no capabilities, but with every capability dropped except
  `NET_BIND_SERVICE`, no privilege escalation, a read-only root filesystem,
  the default seccomp profile, and no service-account token.
- `/var/lib/kvad-relay` on the host keeps the ACME account and certificate,
  so a restart does not ask Let's Encrypt for a new one.
- No metrics endpoint, and no logs beyond its start and its errors.
