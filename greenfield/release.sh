#!/bin/bash
# The release bundle: one file a friend downloads and installs (greenfield/release/install.sh inside).
#   IMPULSE_BUILD=<a portable impulse build folder> bash greenfield/release.sh OUT_DIR
# Makes OUT_DIR/nebula-eh-<version>-linux-x86_64.tar.gz: the nebula-eh wheel, impulse-server (built with
# -DIMPULSE_PORTABLE=ON for RTX 30/40/50) and the CUDA runtime libraries it needs, the install script, the licences.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?the folder to write the bundle into}
IB=${IMPULSE_BUILD:?a portable impulse build folder (IMPULSE_BUILD=…)}
VER=$(python3 -c 'import tomllib,sys; print(tomllib.load(open(sys.argv[1],"rb"))["project"]["version"])' "$ROOT/pyproject.toml")
NAME=nebula-eh-$VER-linux-x86_64
W=$(mktemp -d); B=$W/$NAME
mkdir -p "$B/wheels" "$B/impulse"
/bin/grep -q '^GGML_NATIVE:BOOL=OFF' "$IB/CMakeCache.txt" || { echo "$IB is not a portable build (GGML_NATIVE is not OFF)" >&2; exit 1; }
(cd "$ROOT" && python3 -m pip wheel -q --no-deps --wheel-dir "$B/wheels" .)
cp "$IB/bin/impulse-server" "$B/impulse/impulse-server.bin"
# the CUDA runtime it links (a friend has the driver, not the toolkit): copied beside it; the wrapper points at them
for lib in $(ldd "$IB/bin/impulse-server" | awk '/=>/ {print $3}' | /bin/grep -E '/cuda|libcudart|libcublas|libgomp' || true); do cp -L "$lib" "$B/impulse/"; done
cat > "$B/impulse/impulse-server" <<'WRAP'
#!/bin/sh
# impulse-server with the CUDA runtime it was built with (beside it)
D=$(dirname "$(readlink -f "$0")")
LD_LIBRARY_PATH="$D${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$D/impulse-server.bin" "$@"
WRAP
chmod +x "$B/impulse/impulse-server" "$B/impulse/impulse-server.bin"
cp "$ROOT/greenfield/release/install.sh" "$B/install.sh"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$B/"
cp "$ROOT/impulse/THIRD_PARTY_NOTICES.md" "$B/impulse/"
printf 'Nebula EH %s\n\nInstall: bash install.sh\nThen open "Nebula EH" from your applications.\n\nimpulse/ carries NVIDIA CUDA runtime libraries (libcudart, libcublas) and GNU libgomp, redistributed under the NVIDIA CUDA Toolkit EULA.\n' "$VER" > "$B/README.txt"
mkdir -p "$OUT"
tar -czf "$OUT/$NAME.tar.gz" -C "$W" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
rm -rf "$W"
ls -la "$OUT/$NAME.tar.gz"
