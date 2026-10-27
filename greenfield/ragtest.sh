#!/bin/bash
# Lookup's library in a throwaway container with no network (greenfield/Containerfile.rag: Qdrant, fastembed with its
# models, Docling with its models, all fetched once at build time). The tree is copied in; nothing here is touched.
#   bash greenfield/ragtest.sh                         # test_lookup + the real run (test_lookup_real)
#   bash greenfield/ragtest.sh nebula.tests.test_x     # some tests
#   bash greenfield/ragtest.sh all                     # the whole suite with Lookup's parts installed
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMG=nebula-greenfield-rag
NAME=nebula-rag-$$
docker image inspect nebula-greenfield-dev >/dev/null 2>&1 || docker build -q -t nebula-greenfield-dev -f "$ROOT/greenfield/Containerfile.dev" "$ROOT/greenfield" >/dev/null || exit 2
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -q -t "$IMG" -f "$ROOT/greenfield/Containerfile.rag" "$ROOT/greenfield" >/dev/null || exit 2
docker run -d --rm --name "$NAME" --network none "$IMG" sleep infinity >/dev/null || exit 2
trap 'docker rm -f "$NAME" >/dev/null 2>&1' EXIT
(cd "$ROOT" && git ls-files -co --exclude-standard | /bin/grep -v -E '^nebula/desktop/node_modules/' | tar -cf - -T - --transform 's,^,Nebula/,') \
  | docker cp - "$NAME:/work" >/dev/null
docker exec -u root "$NAME" chown -R stranger /work >/dev/null
TESTS=${*:-nebula.tests.test_lookup nebula.tests.test_lookup_real}
if [ "${1:-}" = "all" ]; then TESTS="discover -s nebula/tests -t ."; fi
docker exec -w /work/Nebula -e NEBULA_RAG_REAL=1 "$NAME" /opt/venv/bin/python -m unittest $TESTS 2>&1 | tail -40
exit "${PIPESTATUS[0]}"
