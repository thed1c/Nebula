#!/bin/bash
# Three Nebulas joined by Gravity invites, recorded for the constellation GIFs: bash greenfield/constellation_demo.sh
# Frames land in ${OUT:-/tmp}/constellation/ (frames.json lists them with their captions); greenfield/gifs.py makes the GIFs.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=nebula-demo-$$
docker run -d --rm --name $N --network none nebula-greenfield-ui sleep infinity >/dev/null
trap 'docker rm -f $N >/dev/null 2>&1' EXIT
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') | docker cp - $N:/work >/dev/null
docker exec -u root $N chown -R stranger /work
docker exec $N /opt/venv/bin/python /work/Nebula/greenfield/constellation_demo.py; r=$?
rm -rf "${OUT:-/tmp}/constellation"; docker cp $N:/tmp/demo "${OUT:-/tmp}/constellation" >/dev/null 2>&1
exit $r
