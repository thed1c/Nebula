#!/bin/bash
# A from-scratch Nebula on a throwaway container: a stock image, a clone from GitHub, the README's install, then the
# first run and the tests with the network cut. Nothing from the host is mounted; logs come out by docker cp.
#   bash greenfield/run.sh [branch]          (REPO=..., OUT=..., DENY=<private name list> optional)
# Throw it all away after: docker rm -f nebula-greenfield; docker rmi nebula-greenfield
set -u
REPO=${REPO:-https://github.com/thed1c/Nebula}
BRANCH=${1:-main}
NAME=nebula-greenfield
OUT=${OUT:-$PWD/greenfield-run-$(date +%Y%m%d-%H%M%S)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

say "0. build the stranger's box"
docker build -q -t "$NAME" -f "$HERE/Containerfile" "$HERE" >> "$OUT/run.log" 2>&1 || { say "build failed"; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --cap-add NET_RAW --network bridge "$NAME" sleep infinity >/dev/null

say "1. online: clone + install per the README, every outbound connection captured"
docker exec -u root -d "$NAME" tcpdump -i eth0 -nn -U -w /tmp/online.pcap
sleep 1
if ! docker exec "$NAME" sh -c "git clone -q --branch '$BRANCH' '$REPO' Nebula && cd Nebula && git log --oneline -1" > "$OUT/01-clone.log" 2>&1; then
  # a private repository: a stranger cannot clone it. With BUNDLE=1 the branch as pushed is packed on this machine
  # (git bundle from a fresh fetch of the remote) and cloned from that inside the container; recorded as a finding.
  [ "${BUNDLE:-0}" = 1 ] || { say "clone failed (a private repository? BUNDLE=1 clones the pushed branch from a bundle)"; exit 1; }
  say "clone from GitHub failed (recorded); cloning the pushed $BRANCH from a bundle instead"
  T=$(mktemp -d); git clone -q --bare --branch "$BRANCH" --single-branch "$REPO" "$T/src.git" && git -C "$T/src.git" bundle create "$T/nebula.bundle" "$BRANCH" 2>>"$OUT/run.log" \
    || { say "bundle failed"; exit 1; }
  docker cp "$T/nebula.bundle" "$NAME:/tmp/nebula.bundle" >/dev/null; rm -rf "$T"
  docker exec "$NAME" sh -c "git clone -q --branch '$BRANCH' /tmp/nebula.bundle Nebula && cd Nebula && git log --oneline -1" >> "$OUT/01-clone.log" 2>&1 || { say "clone from the bundle failed"; exit 1; }
fi
docker exec "$NAME" sh -c 'cd Nebula && python3 -m venv .venv && .venv/bin/pip install -q -e ".[monitor]"' > "$OUT/02-install.log" 2>&1 || { say "install failed"; exit 1; }
docker exec -u root "$NAME" pkill -INT tcpdump; sleep 1
docker cp "$NAME:/tmp/online.pcap" "$OUT/online.pcap" >/dev/null

say "2. offline: the network cut; only the container's own loopback is left"
docker network disconnect bridge "$NAME"
docker exec -u root -d "$NAME" tcpdump -i any -nn -U -w /tmp/offline.pcap 'not host 127.0.0.1 and not host ::1'
if [ -n "${DENY:-}" ]; then docker cp "$DENY" "$NAME:/tmp/deny.txt" >/dev/null; D=/tmp/deny.txt; else D=; fi
docker exec -e NEBULA_BOUNDARY_DENY="$D" "$NAME" bash Nebula/greenfield/walk.sh > "$OUT/03-walk.log" 2>&1
fails=$?
docker exec -u root "$NAME" pkill -INT tcpdump; sleep 1
docker cp "$NAME:/tmp/offline.pcap" "$OUT/offline.pcap" >/dev/null
docker cp "$NAME:/tmp/walk" "$OUT/walk" >/dev/null

say "3. outbound: who the container talked to"
docker run --rm --network none -v "$OUT:/o" --entrypoint sh "$NAME" -c \
  'for f in online offline; do echo "== $f (new TCP connections, by host:port)"; tcpdump -nn -r /o/$f.pcap "tcp[tcpflags] & (tcp-syn|tcp-ack) == tcp-syn" 2>/dev/null | awk "{print \$5}" | sed "s/:\$//" | sort | uniq -c; echo "== $f (DNS questions)"; tcpdump -nn -r /o/$f.pcap "udp port 53" 2>/dev/null | grep -oE "A{1,4}\? [^ ]+" | sort | uniq -c; done' > "$OUT/04-outbound.log" 2>&1
cat "$OUT/04-outbound.log" >> "$OUT/run.log"
say "done: $fails failed walk step(s); logs in $OUT (container $NAME left for a look; docker rm -f $NAME to throw it away)"
exit "$fails"
