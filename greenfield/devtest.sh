#!/bin/bash
# Run Nebula's tests on the working tree inside a throwaway container with no network: the tree is copied in (tar,
# no mount), the dependencies come from the dev image (built once, online). Nothing on this machine is touched.
#   bash greenfield/devtest.sh                        # the whole suite + page parse + boundary test
#   bash greenfield/devtest.sh nebula.tests.test_x    # some tests
#   DENY=<private name list> adds the boundary test's private list
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMG=nebula-greenfield-dev
NAME=nebula-dev-$$
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -q -t "$IMG" -f "$ROOT/greenfield/Containerfile.dev" "$ROOT/greenfield" >/dev/null || exit 2
docker run -d --rm --name "$NAME" --network none "$IMG" sleep infinity >/dev/null || exit 2
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') \
  | docker cp - "$NAME:/work" >/dev/null
[ -n "${DENY:-}" ] && docker cp "$DENY" "$NAME:/tmp/deny.txt" >/dev/null
docker exec -u root "$NAME" chown -R stranger /work >/dev/null
if [ $# -gt 0 ]; then
  docker exec -w /work/Nebula ${DENY:+-e NEBULA_BOUNDARY_DENY=/tmp/deny.txt} "$NAME" /opt/venv/bin/python -m unittest "$@"
  exit $?
fi
docker exec -w /work/Nebula ${DENY:+-e NEBULA_BOUNDARY_DENY=/tmp/deny.txt} "$NAME" sh -c '
  /opt/venv/bin/python -m unittest discover -s nebula/tests -t . > /tmp/tests.log 2>&1; r=$?
  /bin/grep -A20 -E "^(FAIL|ERROR):" /tmp/tests.log; tail -4 /tmp/tests.log
  /opt/venv/bin/python - <<P
import re, pathlib
h = pathlib.Path("nebula/ui/index.html").read_text()
pathlib.Path("/tmp/page.js").write_text("\n".join(re.findall(r"<script[^>]*>(.*?)</script>", h, re.S)))
P
  node --check /tmp/page.js && node --check nebula/desktop/main.js && echo "page parse ok"'
