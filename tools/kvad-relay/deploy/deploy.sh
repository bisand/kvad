#!/usr/bin/env bash
# Builds kvad-relay for x86_64 Linux, imports its image into the k3s on
# HOST over SSH (no registry), and applies a manifest with the keys in
# ALLOW_FILE. Run from tools/kvad-relay:
#
#   deploy/deploy.sh root@postbud.tarsier-bull.ts.net deploy/allow.local [ssh options...]
#
# RELAY_MANIFEST picks the manifest (deploy/k8s.yaml, the relay on the host
# network; deploy/k8s-traefik.yaml, behind a Traefik that holds 443), and
# RELAY_HOSTNAME and RELAY_ADDRESS fill in its name and public address.
# RELAY_SKIP_BUILD=1 applies the manifest with the image already on HOST:
# for a change to the allow list alone.
set -euo pipefail
host=$1; allow=$2; shift 2
cd "$(dirname "$0")/.."
tag=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

if [[ -z "${RELAY_SKIP_BUILD:-}" ]]; then
    cargo zigbuild --release --target x86_64-unknown-linux-musl
    cp target/x86_64-unknown-linux-musl/release/kvad-relay deploy/kvad-relay
    docker build -q --platform linux/amd64 -t "kvad-relay:$tag" deploy
    rm deploy/kvad-relay
fi

# RELAY_HOSTNAME may name several, comma-separated: one certificate for all,
# and a Traefik route that matches any of them.
names=${RELAY_HOSTNAME:-relay.kvad.eu}
sni=$(printf '%s' "$names" | tr ',' '\n' | sed 's/.*/HostSNI(`&`)/' | paste -sd '|' - | sed 's/|/ || /g')

# The allow list goes into the ConfigMap, indented under `allow: |`.
keys=$(mktemp)
trap 'rm -f "$keys"' EXIT
grep -v '^[[:space:]]*$' "$allow" | sed 's/^/    /' > "$keys"
manifest=$(sed -e "s/IMAGE_TAG/$tag/" -e "s/RELAY_SNI_MATCH/$sni/" -e "s/RELAY_HOSTNAME/$names/g" -e "s/RELAY_ADDRESS/${RELAY_ADDRESS:-}/g" "${RELAY_MANIFEST:-deploy/k8s.yaml}" \
    | sed -e '/^    ALLOWED_KEYS$/{' -e "r $keys" -e 'd' -e '}')

if [[ -z "${RELAY_SKIP_BUILD:-}" ]]; then
    docker save "kvad-relay:$tag" | ssh "$@" "$host" 'k3s ctr images import -'
fi
printf '%s\n' "$manifest" | ssh "$@" "$host" 'kubectl apply -f - && kubectl -n kvad-relay rollout restart deploy/kvad-relay && kubectl -n kvad-relay rollout status deploy/kvad-relay --timeout=60s'
ssh "$@" "$host" 'sleep 5; kubectl -n kvad-relay logs deploy/kvad-relay --tail=20'
