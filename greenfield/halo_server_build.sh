#!/bin/bash
# Run in a clean Ubuntu 22.04 container with the CUDA redist (greenfield/cuda_redist.py) at /cuda and this folder at /work:
#   docker run --rm -v <dir>:/work -v <cuda>:/cuda:ro -v <vulkansdk>/1.4.363.0:/vk:ro ubuntu:22.04 bash /work/halo_server_build.sh
# then: python3 greenfield/package.py --platform linux --impulse <impulse dir> --halo-server <dir>/build/bin --halo-base <LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf>
# llama-server for Halo's own model, built on Ubuntu 22.04 (glibc 2.35+) at llama.cpp b6b003d (the commit Halo's numbers
# were measured on): CPU variants chosen at run time, CUDA and Vulkan as loadable backends (Vulkan: Halo on a Radeon or
# other non-NVIDIA card; no GPU driver at all: the CPU).
set -eu
apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake ninja-build curl unzip ca-certificates libvulkan-dev >/dev/null
# Vulkan's shader compiler (glslc) from LunarG's Vulkan SDK mounted at /vk (Ubuntu 22.04 has no glslc package)
[ -x /vk/x86_64/bin/glslc ] && export VULKAN_SDK=/vk/x86_64 && export PATH=/vk/x86_64/bin:$PATH
cd /work
[ -d llama.cpp-b6b003d2cb29647d968302eb2db8da6f66303b3e ] || { curl -sL -o src.zip https://github.com/ggml-org/llama.cpp/archive/b6b003d2cb29647d968302eb2db8da6f66303b3e.zip && unzip -q src.zip; }
cmake -S llama.cpp-b6b003d2cb29647d968302eb2db8da6f66303b3e -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120a-real" -DCMAKE_CUDA_COMPILER=/cuda/bin/nvcc -DCUDAToolkit_ROOT=/cuda \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j "$(nproc)"
ls -la build/bin
