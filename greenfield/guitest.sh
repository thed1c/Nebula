#!/bin/bash
# The page in Chromium, in a throwaway container with no network: bash greenfield/guitest.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
docker image inspect nebula-greenfield-ui >/dev/null 2>&1 || docker build -q -t nebula-greenfield-ui -f "$ROOT/greenfield/Containerfile.ui" "$ROOT/greenfield" >/dev/null || exit 2
N=nebula-gui-$$
docker run -d --rm --name $N --network none nebula-greenfield-ui sleep infinity >/dev/null
trap 'docker rm -f $N >/dev/null 2>&1' EXIT
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') | docker cp - $N:/work >/dev/null
docker exec -u root $N chown -R stranger /work
docker exec $N /opt/venv/bin/python /work/Nebula/greenfield/guitest.py; r=$?
docker cp $N:/tmp/engines.png "${OUT:-/tmp}/nebula-engines.png" >/dev/null 2>&1
docker cp $N:/tmp/quasar.png "${OUT:-/tmp}/nebula-quasar.png" >/dev/null 2>&1
for f in halo-panel wormhole chains exchange-1 exchange-2 exchange-3 exchange-4 exchange-5 exchange-6 exchange-7 exchange-8 exchange-rest welcome join map-swarm plan map-plan holemenu swarm-colour schema smart signin-setup signin hfcache lookup-menu picker gyro assistant-menu routing-wizard engine-spec wheel variants rules attach; do docker cp $N:/tmp/$f.png "${OUT:-/tmp}/nebula-$f.png" >/dev/null 2>&1; done
docker cp $N:/tmp/firstrun.png "${OUT:-/tmp}/nebula-firstrun.png" >/dev/null 2>&1
docker cp $N:/tmp/map.png "${OUT:-/tmp}/nebula-map.png" >/dev/null 2>&1
docker cp $N:/tmp/wizard.png "${OUT:-/tmp}/nebula-wizard.png" >/dev/null 2>&1
docker cp $N:/tmp/gravity.png "${OUT:-/tmp}/nebula-gravity.png" >/dev/null 2>&1
exit $r
