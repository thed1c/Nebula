#!/bin/bash
# Gravity across networks through your own gateway, in throwaway containers: Nebula A (with a model) on one internal
# network, Nebula B on another, the gateway (iroh's relay, nebula/gateway) on both; neither network has a route out,
# and A and B cannot reach each other except through the gateway. B joins A by invite, A grants a shared model, B
# uses it. Nebula C, beside B but not on the gateway's allowlist, cannot join. Nothing leaves the internal networks.
#   bash greenfield/gatewaynet.sh            (needs the dev image: greenfield/devtest.sh builds it)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMG=nebula-greenfield-dev
GW=nebula-gateway:1
T=gwtest-$$
say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "FAIL: $*"; exit 1; }
cleanup() { docker rm -f $T-a $T-b $T-c $T-gw >/dev/null 2>&1; docker network rm $T-n1 $T-n2 >/dev/null 2>&1; }
trap cleanup EXIT
docker image inspect $IMG >/dev/null 2>&1 || docker build -q -t $IMG -f "$ROOT/greenfield/Containerfile.dev" "$ROOT/greenfield" >/dev/null || exit 2
docker image inspect $GW >/dev/null 2>&1 || docker build -q -t $GW -f "$ROOT/nebula/gateway/Containerfile" "$ROOT/nebula/gateway" >/dev/null || fail "gateway image"
docker network create --internal $T-n1 >/dev/null && docker network create --internal $T-n2 >/dev/null
for n in a b c; do
  net=$T-n1; [ $n != a ] && net=$T-n2
  docker run -d --rm --name $T-$n --network $net $IMG sleep infinity >/dev/null
  (cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') | docker cp - $T-$n:/work >/dev/null
  docker exec -u root $T-$n chown -R stranger /work
done
# A runs a pretend engine and shares it with guests
docker exec $T-a sh -c 'echo GGUF > /work/m.gguf; cd /work/Nebula && (/opt/venv/bin/python -m nebula.tests.fake_impulse -m /work/m.gguf --port 8080 > /work/engine.log 2>&1 &)'
for n in a b c; do
  tiers='[]'; [ $n = a ] && tiers='[{"name": "big", "engine": "http://127.0.0.1:8080", "shared": true}]'
  docker exec $T-$n sh -c "printf '{\"port\": 8090, \"ledger\": \"/work/l.sqlite3\", \"monitor\": false, \"tiers\": $tiers, \"lan\": {\"decided\": true, \"discoverable\": false, \"name\": \"$n\"}, \"gravity\": {\"enabled\": true, \"name\": \"node-$n\", \"relays\": [\"http://gateway:3340\"]}}' > /work/nebula.json; cd /work/Nebula && (/opt/venv/bin/python -m nebula --config /work/nebula.json > /work/nebula.log 2>&1 &)"
done
api() { docker exec $T-$1 curl -s -m 60 ${3:+-H Content-Type:application/json -d "$3"} http://127.0.0.1:8090$2; }
for n in a b c; do
  for i in $(seq 1 60); do api $n /api/yard/node | grep -q '"id"' && break; sleep 0.5; done
done
A=$(api a /api/yard/node | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
B=$(api b /api/yard/node | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
say "A $A"; say "B $B"
# the gateway lets in A and B only
CONF=$(mktemp -d)
printf 'enable_relay = true\nenable_quic_addr_discovery = false\nhttp_bind_addr = "0.0.0.0:3340"\naccess.allowlist = ["%s", "%s"]\n' "$A" "$B" > $CONF/relay.toml
docker create --name $T-gw --network $T-n1 --network-alias gateway $GW >/dev/null
docker cp $CONF/relay.toml $T-gw:/etc/nebula-gateway/relay.toml 2>/dev/null || { docker cp $CONF/. $T-gw:/etc/nebula-gateway/ >/dev/null; }
docker network connect --alias gateway $T-n2 $T-gw
docker start $T-gw >/dev/null
for i in $(seq 1 30); do docker exec $T-b curl -sf -m 2 http://gateway:3340/ping >/dev/null && break; sleep 0.5; done
docker exec $T-b curl -sf -m 2 http://gateway:3340/ping >/dev/null || fail "the gateway does not answer from B's network"
docker exec $T-b curl -sf -m 2 http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' $T-a | awk '{print $1}'):8090/api/config >/dev/null && fail "B reaches A directly: the test networks are not apart"
say "A and B cannot reach each other except through the gateway"
for n in a b c; do api $n /api/yard/node/relays '{"urls": ["http://gateway:3340"]}' >/dev/null; done      # online now
sleep 2
INV=$(api a /api/yard/node/invite '{"kind": "friend"}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["invite"])')
api b /api/yard/node/join "{\"invite\": \"$INV\"}" | tee /tmp/$T-join.json | grep -q '"peer"' || fail "B did not join A: $(cat /tmp/$T-join.json)"
say "B joined A through the gateway"
api a /api/yard/node/peers/$B/grants '{"models": ["big"]}' | grep -q '"grants"' || fail "grant"
api b /api/yard/node/peers/$A/adopt '{}' | grep -q 'node-a-big' || fail "B did not see A's granted model"
api b /v1/chat/completions '{"model": "node-a-big", "messages": [{"role": "user", "content": "hi"}]}' | grep -q '"choices"' || fail "B could not use A's model"
say "B used A's model across the gateway"
INV2=$(api a /api/yard/node/invite '{"kind": "friend"}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["invite"])')
OUT=$(api c /api/yard/node/join "{\"invite\": \"$INV2\"}")
echo "$OUT" | grep -q '"peer"' && fail "C joined although the gateway does not let it in"
say "C (not a member) could not join: $(echo "$OUT" | head -c 160)"
docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' $T-gw | grep -qv "bridge" || fail "the gateway is on a network with a route out"
say "PASS"
