#!/bin/bash
# A real engine in a throwaway container: llama.cpp's CPU build from its GitHub release and a 1 MB real GGUF pulled
# by Nebula's own Hugging Face code; the network is then cut and the model is deployed, verified, asked and drained
# through Nebula's API. Every outbound connection is captured. No GPU, nothing on this machine touched.
#   bash greenfield/realmodel.sh            (OUT=<logs dir>)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMG=nebula-greenfield-dev
NAME=nebula-real-$$
OUT=${OUT:-$PWD/realmodel-$(date +%Y%m%d-%H%M%S)}
LLAMA_TAG=${LLAMA_TAG:-b11742}
mkdir -p "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -q -t "$IMG" -f "$ROOT/greenfield/Containerfile.dev" "$ROOT/greenfield" >/dev/null || exit 2
docker run -d --rm --name "$NAME" --cap-add NET_RAW --network bridge -u root "$IMG" sleep infinity >/dev/null || exit 2
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT
docker exec "$NAME" sh -c 'apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y tcpdump libgomp1 >/dev/null 2>&1' || true
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') | docker cp - "$NAME:/work" >/dev/null
docker exec -d "$NAME" tcpdump -i eth0 -nn -U -w /tmp/online.pcap
sleep 1
say "1. online: llama.cpp $LLAMA_TAG CPU build from its GitHub release"
docker exec -w /work "$NAME" sh -c "curl -fsSL -o llama.tgz https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz && mkdir llama && tar -xzf llama.tgz -C llama && find /work/llama -name llama-server -type f | head -1 > /tmp/llama-server-path && cat /tmp/llama-server-path" >> "$OUT/run.log" 2>&1 || { say "llama.cpp download failed"; exit 1; }
docker exec -w /work "$NAME" sh -c 'B=$(cat /tmp/llama-server-path); LD_LIBRARY_PATH=$(dirname $B) $B --version 2>&1 | head -3' >> "$OUT/run.log" 2>&1
say "2. online: Nebula, and the GGUF through its own HF pull"
docker exec -d -w /work/Nebula "$NAME" sh -c 'printf "{\"port\": 8090, \"ledger\": \"/work/n.sqlite3\", \"monitor\": false, \"lan\": {\"decided\": true, \"discoverable\": false}}" > /work/nebula.json; LD_LIBRARY_PATH=$(dirname $(cat /tmp/llama-server-path)) /opt/venv/bin/python -m nebula --config /work/nebula.json > /work/nebula.log 2>&1'
docker exec "$NAME" sh -c 'for i in $(seq 40); do curl -sf http://127.0.0.1:8090/v1/models >/dev/null && exit 0; sleep 0.5; done; exit 1' || { say "Nebula did not start"; docker exec "$NAME" cat /work/nebula.log; exit 1; }
docker exec -w /work/Nebula "$NAME" /opt/venv/bin/python greenfield/realmodel.py pull | tee -a "$OUT/run.log" || exit 1
docker exec "$NAME" pkill -INT tcpdump; sleep 1
docker cp "$NAME:/tmp/online.pcap" "$OUT/online.pcap" >/dev/null
say "3. offline: the network cut; deploy, verify, ask, drain"
docker network disconnect bridge "$NAME"
docker exec -d "$NAME" tcpdump -i any -nn -U -w /tmp/offline.pcap 'not host 127.0.0.1 and not host ::1'
docker exec -w /work/Nebula "$NAME" /opt/venv/bin/python greenfield/realmodel.py deploy | tee -a "$OUT/run.log"; r=${PIPESTATUS[0]}
docker exec "$NAME" pkill -INT tcpdump; sleep 1
docker cp "$NAME:/tmp/offline.pcap" "$OUT/offline.pcap" >/dev/null
docker cp "$NAME:/work/nebula.log" "$OUT/nebula.log" >/dev/null 2>&1
say "4. outbound"
docker image inspect nebula-greenfield >/dev/null 2>&1 || docker build -q -t nebula-greenfield -f "$ROOT/greenfield/Containerfile" "$ROOT/greenfield" >/dev/null
docker run --rm --network none -v "$OUT:/o" --entrypoint sh nebula-greenfield -c \
  'for f in online offline; do echo "== $f (new TCP connections)"; tcpdump -nn -r /o/$f.pcap "tcp[tcpflags] & (tcp-syn|tcp-ack) == tcp-syn" 2>/dev/null | awk "{print \$5}" | sed "s/:\$//" | sort | uniq -c; echo "== $f (DNS)"; tcpdump -nn -r /o/$f.pcap "udp port 53" 2>/dev/null | grep -oE "A{1,4}\? [^ ]+" | sort | uniq -c; done' | tee -a "$OUT/run.log"
say "done (exit $r); logs in $OUT"
exit $r
