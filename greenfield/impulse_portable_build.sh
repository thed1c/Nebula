#!/bin/bash
# Impulse portable for Linux (impulse/portable: llama.cpp's server path with Impulse's hooks, CPU + Vulkan, no CUDA), for
# Halo on a machine without an NVIDIA card. Run in a clean Ubuntu 22.04 container (glibc 2.35+) with this repository at
# /src (read-only), LunarG's Vulkan SDK at /vk (Ubuntu 22.04 has no glslc) and an output folder at /out:
#   docker run --rm -u "$(id -u):$(id -g)" -v <repo>:/src:ro -v <vulkansdk>/1.4.363.0:/vk:ro -v <out>:/out ubuntu:22.04 \
#     bash /src/greenfield/impulse_portable_build.sh            (as root: it installs the compilers first)
# then: /out/impulse-portable-<version>-linux-x86_64.tar.gz (impulse-server and its libraries, one folder) + .sha256
set -eu
if [ "$(id -u)" = 0 ]; then
  apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential ninja-build git curl unzip ca-certificates libvulkan-dev python3 python3-pip >/dev/null
  python3 -m pip install -q "cmake==3.31.6"                      # 3.24+ (Ubuntu 22.04 has 3.22)
fi
export VULKAN_SDK=/vk/x86_64 PATH=/vk/x86_64/bin:$PATH
VER=$(python3 -c 'import tomllib; print(tomllib.load(open("/src/pyproject.toml","rb"))["project"]["version"])' 2>/dev/null \
      || sed -n 's/^version = "\(.*\)"/\1/p' /src/pyproject.toml | head -1)
B=/out/build-portable
cmake -S /src/impulse/portable -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$B" --target impulse-server -j "$(nproc)"
D=/out/impulse-portable; rm -rf "$D"; mkdir -p "$D"
cp -a "$B"/bin/impulse-server "$D"/
for f in "$B"/bin/*.so*; do cp -a "$f" "$D"/; done
# what it links that a plain Linux may lack (ggml's CPU backend uses OpenMP): beside it, found through its $ORIGIN rpath
for lib in $(ldd "$D"/impulse-server "$D"/*.so* | awk '/=>/ {print $3}' | grep -E '/libgomp\.so' | sort -u); do cp -L "$lib" "$D"/; done
cp /src/impulse/flashnext/third_party/ggml/LICENSE "$D/LICENSE-llama.cpp"
"$D"/impulse-server --version 2>&1 | tail -2
( cd /out && tar -czf "impulse-portable-$VER-linux-x86_64.tar.gz" -C "$D" . && sha256sum "impulse-portable-$VER-linux-x86_64.tar.gz" > "impulse-portable-$VER-linux-x86_64.tar.gz.sha256" )
ls -la /out/impulse-portable-* "$D"
