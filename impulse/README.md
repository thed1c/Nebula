# Impulse

**Experimental.** Impulse is an inference engine and server for mixture-of-experts and dense models on consumer NVIDIA cards. It has a llama-server-compatible API. It began with Strata's adaptive expert cache (https://github.com/Niko1221/Strata) and has since grown into its own engine: Gemma 4 and Qwen3.5 / 3.6 / 3.8 (dense and expert), LFM2.5, several GPUs, parallel slots, MTP drafting and images.

Status: version 0.1.1, unreleased, the first release with Nebula (`CHANGELOG.md`; it was Strata-Lite 0.2.4 before the rename). The benchmark tables for 0.1.1 are still to be measured. The numbers under `docs/` and `results/` come from earlier versions and are labelled with them.

| | |
|---|---|
| Models | Gemma 4 26B-A4B and 31B; Qwen3.6-35B-A3B, Qwen3.8-27B and other `qwen35` models; Qwen3.8 Flash-Next (`qwen4exp`); LFM2.5-24B-A2B (`docs/QWEN36.md`, `docs/LFM25.md`) |
| Expert models on one 12 GB card | each layer's experts are cached on the GPU and adapt to the conversation; misses go to a CPU expert pool in the same step; swaps run on a copy stream (`docs/HOW-IT-WORKS.md`) |
| Several GPUs | layer split (`-sm layer -ts …`), and tensor split (`-sm tensor`) for dense models and for expert models whose experts all fit on the cards |
| Serving | parallel slots (`-np`), MTP drafters, a prompt cache (in RAM too, with `--cache-ram`), the image projector placed on demand, LoRA adapters (`--lora`, Gemma 4) |
| Correctness | the same logits as llama.cpp `b6b003d`, bit for bit, on the same cards and flags (`docs/CORRECTNESS.md`); 63/63 API parity checks per model (`docs/API.md`) |
| Needs | Linux or Windows, an NVIDIA RTX 30 / 40 / 50 card, CUDA 12.8+ to build for RTX 50, a CPU with AVX2 |

## Build

```bash
cmake -S impulse -B build-impulse -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-impulse --target impulse-server
```
The build downloads llama.cpp at the pinned commit (or uses `-DIMPULSE_LLAMA_DIR=<tree>`). `-DIMPULSE_PORTABLE=ON` builds a binary for other PCs. Details: `docs/BUILD.md`.

## Run

```bash
impulse-server -m model.gguf --mtp drafter.gguf --mmproj mmproj.gguf
```
The server picks the engine from the GGUF's architecture and serves llama-server's and OpenAI's chat API on `127.0.0.1:8080`. Every flag: `docs/RUN.md`.

## Tests

`tests/` has the bit-for-bit stages against llama.cpp (`stage_*.sh`, `q35_stage_*.sh`, `twin_check.sh`), the speed grids (`grid.sh`, `grid_qwen.sh`) and the API parity check (`api_parity.py`). They need a GPU, the model files and your own prompt files. The rule for this repository: no prompt text, token ids, logits, routes or generated text are committed; results are numbers only (`CONTRIBUTING.md`).

## Licence

MIT, Copyright (c) 2026 thed1c (`LICENSE`). Strata's CPU expert pool (in `flashnext/`, Strata's tree; `third_party/strata/UPSTREAM`; MIT, Niko1221 and the Strata contributors), ggml's `ggml-common.h` and llama.cpp keep their own notices: `THIRD_PARTY_NOTICES.md`. Model weights are not included. Download each model under its own terms.
