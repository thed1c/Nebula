#!/bin/bash
# Add a server (SSH), end to end, in throwaway containers on an internal network (no route out):
#   main  = your Nebula (the dev image) with the router's backend (real NVIDIA PAIR, built from source)
#   box   = a bare Debian server with sshd and Python only
# main confirms box's host key, checks it, installs a headless Nebula there from the offline bundle over SSH, joins it
# (Gravity owner link) and pairs it on the household network (PAIR mTLS, PIN carried over the owner link); then deploys a
# model on it over the owner link, uses it from main, and proves the request crossed as TLS (no plaintext on the wire).
# A changed host key is refused; uninstall removes it all.
#   PAIR_BIN=<nvpair binaries> BUNDLE=<bundle.tgz> bash greenfield/household.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=hh-$$
say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "FAIL: $*"; docker exec $T-main sh -c 'tail -20 /work/main.log' 2>/dev/null; exit 1; }
cleanup() { docker rm -f $T-main $T-box $T-tap >/dev/null 2>&1; docker network rm $T-net >/dev/null 2>&1; }
trap cleanup EXIT
: "${PAIR_BIN:?PAIR_BIN}" "${BUNDLE:?BUNDLE}"
docker network create --internal $T-net >/dev/null
docker run -d --rm --name $T-box --hostname box --network $T-net nebula-sshbox >/dev/null
docker run -d --rm --name $T-main --hostname main --network $T-net nebula-greenfield-dev sleep infinity >/dev/null
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') | docker cp - $T-main:/work >/dev/null
docker exec -u root $T-main mkdir -p /pair; docker cp "$PAIR_BIN/." $T-main:/pair/ >/dev/null; docker cp "$BUNDLE" $T-main:/work/bundle.tgz >/dev/null
docker exec -u root $T-main sh -c 'chown -R stranger /work /pair'
docker exec $T-main sh -c 'mkdir -p ~/.ssh && ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519'
docker exec $T-main cat /work/.ssh/id_ed25519.pub | docker exec -i -u root $T-box sh -c 'cat > /var/owner/.ssh/authorized_keys && chown owner /var/owner/.ssh/authorized_keys && chmod 600 /var/owner/.ssh/authorized_keys'
docker exec $T-main sh -c 'printf "{\"port\": 8090, \"ledger\": \"/work/l.sqlite3\", \"monitor\": false, \"lan\": {\"decided\": true, \"discoverable\": false, \"name\": \"main\"}, \"network\": {\"enabled\": true, \"role\": \"main\", \"pair_bin\": \"/pair\"}, \"gravity\": {\"enabled\": true, \"name\": \"main\"}}" > /work/main.json; cd /work/Nebula && (/opt/venv/bin/python -m nebula --config /work/main.json > /work/main.log 2>&1 &)'
api() { docker exec $T-main curl -s -m ${3:-900} ${2:+-H Content-Type:application/json -d "$2"} http://127.0.0.1:8090$1; }
for i in $(seq 1 60); do api /api/yard/network | grep -q '"on": true' && break; sleep 1; done
api /api/yard/network | grep -q '"on": true' || fail "main's network did not come up: $(api /api/yard/network)"
FP=$(api /api/yard/servers/scan '{"host": "box"}' | python3 -c 'import json,sys;print([k["fingerprint"] for k in json.load(sys.stdin)["keys"] if k["type"]=="ssh-ed25519"][0])')
say "box's host key: $FP (confirmed in the page; here by the test)"
api /api/yard/servers/trust "{\"host\": \"box\", \"fingerprint\": \"$FP\"}" | grep -q trusted || fail trust
api /api/yard/servers/preflight '{"host": "box", "user": "owner"}' | python3 -c 'import json,sys;[print("   ", x) for x in json.load(sys.stdin)["report"]]'
say "installing over SSH (offline bundle)"
OUT=$(api /api/yard/servers/install '{"host": "box", "user": "owner", "name": "box", "bundle": "/work/bundle.tgz"}')
echo "$OUT" | python3 -c 'import json,sys;d=json.load(sys.stdin);[print("   ", s) for s in d.get("steps") or [d]]' || fail "install: $OUT"
echo "$OUT" | grep -q 'paired on your network' || fail "not paired: $OUT"
BOX=$(echo "$OUT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["server"]["gravity"])')
say "a model on the box, deployed from main over the owner link"
api /api/yard/node/peers/$BOX/admin/engines '{"name": "tiny", "kind": "command", "command": ["/var/owner/.local/share/nebula-headless/venv/bin/python", "-m", "nebula.tests.fake_impulse", "-m", "/tmp/tiny.gguf", "--port", "18310"], "port": 18310}' | grep -q '"version"' || fail "deploy on the box"
api /api/yard/node/peers/$BOX/admin/engines/tiny/start '{"wait": true}' | grep -q '"ready"' || fail "start on the box"
api /api/yard/node/peers/$BOX/admin/engines/tiny/route '{}' | grep -q '"tier"' || fail "route on the box"
for i in $(seq 1 30); do api /api/yard/network/adopt '{}' | grep -q 'net-tiny' && break; api /api/config | grep -q '"net-tiny"' && break; sleep 2; done
api /api/config | grep -q '"net-tiny"' || fail "the box's model did not reach main"
docker run -d --rm --name $T-tap -u root --network container:$T-main --cap-add NET_RAW --entrypoint sleep nebula-greenfield infinity >/dev/null
docker exec -d $T-tap tcpdump -i eth0 -nn -s0 -U -w /tmp/tap.pcap
sleep 1
for i in $(seq 1 45); do       # the box's router advertises a new model to the household within its inventory refresh
  REPLY=$(api /v1/chat/completions '{"model": "net-tiny", "messages": [{"role": "user", "content": "PLAINTEXT-CANARY-42 hello"}]}')
  echo "$REPLY" | grep -q '"choices"' && break; sleep 2
done
echo "$REPLY" | grep -q '"choices"' || fail "main could not use the box's model: $REPLY"
sleep 1; docker exec $T-tap sh -c 'pkill tcpdump'; sleep 1
docker exec $T-tap sh -c 'tcpdump -r /tmp/tap.pcap -A -nn 2>/dev/null | grep -c "PLAINTEXT-CANARY-42"' | grep -qx 0 || fail "the prompt crossed the wire in plaintext"
CONNS=$(docker exec $T-tap sh -c 'tcpdump -r /tmp/tap.pcap -nn "tcp[tcpflags] & (tcp-syn|tcp-ack) == tcp-syn" 2>/dev/null | awk "{print \$5}" | sort -u | tr "\n" " "')
say "main used the box's model; on the wire only TLS (connections: $CONNS)"
docker exec -u root $T-box sh -c 'rm -f /etc/ssh/ssh_host_* && ssh-keygen -A >/dev/null && kill -HUP $(pgrep -o sshd)'
sleep 1
api /api/yard/servers/preflight '{"host": "box", "user": "owner"}' | grep -q 'does not match' || fail "a changed host key was not refused"
say "a changed host key is refused"
docker exec -u root $T-box sh -c 'rm -f /etc/ssh/ssh_host_*_key*' ; true
# uninstall needs SSH: trust the box's new key (the owner would confirm it in the page)
docker exec $T-main sh -c 'rm -f /work/nebula-keys/known_hosts' 2>/dev/null
docker exec -u root $T-box sh -c 'ssh-keygen -A >/dev/null && kill -HUP $(pgrep -o sshd)'; sleep 1
FP=$(api /api/yard/servers/scan '{"host": "box"}' | python3 -c 'import json,sys;print([k["fingerprint"] for k in json.load(sys.stdin)["keys"] if k["type"]=="ssh-ed25519"][0])')
api /api/yard/servers/trust "{\"host\": \"box\", \"fingerprint\": \"$FP\"}" >/dev/null
api /api/yard/servers/box/uninstall '{}' | grep -q removed || fail uninstall
docker exec $T-box sh -c 'ls -A /var/owner/.local/share /var/owner/.config 2>/dev/null; pgrep -f "nebula serve" || true' | grep -q nebula && fail "uninstall left something"
say "uninstalled: nothing left on the box"
say "PASS"
