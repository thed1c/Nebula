#!/bin/bash
# The first run, offline, inside the container (run.sh starts it): the README's steps in order against the stand-ins,
# each step's command and answer logged. Exit code = the number of failed steps.
cd "$(dirname "$0")/.." || exit 99
PY=.venv/bin/python
LOG=${LOG:-/tmp/walk}
mkdir -p "$LOG"
fails=0
step() { local name=$1; shift; echo "== $name: $*"; if "$@" > "$LOG/$name.out" 2>&1; then echo "   ok"; else echo "   FAILED (see $name.out)"; tail -5 "$LOG/$name.out" | sed 's/^/   | /'; fails=$((fails+1)); fi; }
wait_http() { for _ in $(seq 1 40); do curl -sf -m2 "$1" >/dev/null && return 0; sleep 0.5; done; return 1; }
post() { curl -sf -m20 -H 'Content-Type: application/json' -d "$2" "$1"; }

$PY greenfield/standins.py > "$LOG/standins.log" 2>&1 &
step standins wait_http http://127.0.0.1:8080/health

# README: "Install and run ... against a running engine"
$PY -m nebula --engine http://127.0.0.1:8080 --port 8090 > "$LOG/nebula-8090.log" 2>&1 &
step first-run wait_http http://127.0.0.1:8090/v1/models
step openai-models curl -sf http://127.0.0.1:8090/v1/models
step openai-chat post http://127.0.0.1:8090/v1/chat/completions '{"model":"x","messages":[{"role":"user","content":"hi"}]}'
step openai-stream post http://127.0.0.1:8090/v1/chat/completions '{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}'
step anthropic-messages post http://127.0.0.1:8090/v1/messages '{"model":"x","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
step responses post http://127.0.0.1:8090/v1/responses '{"model":"x","input":"hi"}'
step page curl -sf -o /dev/null http://127.0.0.1:8090/
step page-registry curl -sf http://127.0.0.1:8090/api/registry
step page-chat post http://127.0.0.1:8090/api/chat '{"text":"hello"}'
step replay $PY -m nebula.replay --ledger nebula.sqlite3

# README: a config with tiers (the shipped example, as a stranger would copy it)
$PY -m nebula --config nebula/examples/local-engine.json > "$LOG/nebula-18190.log" 2>&1 &
step example-local-engine wait_http http://127.0.0.1:18190/v1/models
step example-agents curl -sf http://127.0.0.1:18190/api/config

# README: watching a router, read only
$PY -m nebula --config nebula/examples/watch-router.json > "$LOG/nebula-18194.log" 2>&1 &
step example-watch wait_http http://127.0.0.1:18194/api/config
sleep 3
step watch-live curl -sf http://127.0.0.1:18194/api/live
step watch-plan curl -sf http://127.0.0.1:18194/api/plan
step watch-lanes curl -sf http://127.0.0.1:18194/api/watch/lanes
step watch-decisions curl -sf http://127.0.0.1:18194/api/watch/decisions

# two peer Nebulas (Gravity's other Nebulas), on the second lane
$PY -m nebula --engine http://127.0.0.1:8081 --port 18201 --no-ledger > "$LOG/peer-18201.log" 2>&1 &
$PY -m nebula --engine http://127.0.0.1:8081 --port 18202 --no-ledger > "$LOG/peer-18202.log" 2>&1 &
step peers wait_http http://127.0.0.1:18202/v1/models

# a Gravity router on this machine: found, adopted, asked
printf '{"port": 8095, "ledger": "/tmp/g.sqlite3", "monitor": false, "gravity_local_ports": [18280], "lan": {"decided": true, "discoverable": false}}' > /tmp/g.json
$PY -m nebula --config /tmp/g.json > "$LOG/nebula-8095.log" 2>&1 &
step gravity-up wait_http http://127.0.0.1:8095/api/config
step gravity-found sh -c 'curl -sf http://127.0.0.1:8095/api/yard/gravity | grep -q 18280'
step gravity-adopt post http://127.0.0.1:8095/api/yard/gravity/adopt '{"address":"http://127.0.0.1:18280"}'
step gravity-ask post http://127.0.0.1:8095/v1/chat/completions '{"model":"gravity-chat","messages":[{"role":"user","content":"hi"}]}'

# the tests, here
step unit-tests $PY -m unittest discover -s nebula/tests -t .
step page-parse sh -c "$PY - <<'P'
import re, pathlib
h = pathlib.Path('nebula/ui/index.html').read_text()
pathlib.Path('/tmp/page.js').write_text('\n'.join(re.findall(r'<script[^>]*>(.*?)</script>', h, re.S)))
P
node --check /tmp/page.js && node --check nebula/desktop/main.js"
[ -n "${NEBULA_BOUNDARY_DENY:-}" ] && step boundary-private env NEBULA_BOUNDARY_DENY="$NEBULA_BOUNDARY_DENY" $PY -m unittest nebula.tests.test_boundary -v

step sockets ss -tunap
echo "walk done: $fails failed step(s); logs in $LOG"
exit $fails
