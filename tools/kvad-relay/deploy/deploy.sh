#!/usr/bin/env bash
# Builds kvad-relay for x86_64 Linux, imports its image into the k3s on
# HOST over SSH (no registry), and applies deploy/k8s.yaml with the keys in
# ALLOW_FILE. Run from tools/kvad-relay:
#
#   deploy/deploy.sh root@postbud.tarsier-bull.ts.net deploy/allow.local [ssh options...]
set -euo pipefail
host=$1; allow=$2; shift 2
cd "$(dirname "$0")/.."
tag=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

cargo zigbuild --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/kvad-relay deploy/kvad-relay
docker build -q --platform linux/amd64 -t "kvad-relay:$tag" deploy
rm deploy/kvad-relay

# The allow list goes into the ConfigMap, indented under `allow: |`.
keys=$(mktemp)
trap 'rm -f "$keys"' EXIT
grep -v '^[[:space:]]*$' "$allow" | sed 's/^/    /' > "$keys"
manifest=$(sed "s/IMAGE_TAG/$tag/" deploy/k8s.yaml | sed -e '/^    ALLOWED_KEYS$/{' -e "r $keys" -e 'd' -e '}')

docker save "kvad-relay:$tag" | ssh "$@" "$host" 'k3s ctr images import -'
printf '%s\n' "$manifest" | ssh "$@" "$host" 'kubectl apply -f - && kubectl -n kvad-relay rollout restart deploy/kvad-relay && kubectl -n kvad-relay rollout status deploy/kvad-relay --timeout=60s'
ssh "$@" "$host" 'sleep 5; kubectl -n kvad-relay logs deploy/kvad-relay --tail=20'
