#!/bin/bash
# The offline server bundle for "Add a server (SSH)": Nebula and its dependencies as wheels, and (with PAIR_BIN) the
# router's backend built from NVIDIA's source. Built in a throwaway container (the only network use: pip).
#   PAIR_BIN=<dir of nvpair binaries> bash greenfield/bundle.sh OUT.tgz
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?the bundle file to write}
W=$(mktemp -d)
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/|^impulse/' | tar -cf - -T -) > "$W/src.tar"
docker run --rm -v "$W:/w" python:3.11-slim-bookworm sh -c 'mkdir /src && tar -xf /w/src.tar -C /src && mkdir -p /w/nebula-bundle/wheels && pip wheel -q --wheel-dir /w/nebula-bundle/wheels "/src[gravity,monitor]" && chmod -R a+rwX /w/nebula-bundle'
if [ -n "${PAIR_BIN:-}" ]; then mkdir -p "$W/nebula-bundle/pair" && cp "$PAIR_BIN"/nvpair-* "$W/nebula-bundle/pair/"; fi
tar -czf "$OUT" -C "$W" nebula-bundle
ls -la "$OUT"
