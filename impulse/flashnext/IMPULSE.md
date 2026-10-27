# Impulse Flash-Next engine

This folder is Impulse's engine for Qwen3.8 Flash-Next MoE packs in Strata's NVFP4 format (for example the
OrcaRouter NVFP4 GPTQ pack the reference engine runs). Impulse's other engines (`../src`: Gemma 4, Qwen, LFM) run GGUF models on
llama.cpp; this one runs a Strata pack (a folder with `pack/`, `orca-nvfp4-dense.gguf`, `ple-fp8.gguf`, `mtp/`).

## Where it comes from

- Strata, https://github.com/Niko1221/Strata, MIT, Copyright (c) 2026 Niko1221 and the Strata contributors
  (`LICENSE` here is that licence, unchanged).
- The strata-nvfp4 fork, https://github.com/sergqwer/strata-nvfp4, version 0.1.44.1-nvfp4.2, commit `257727d`
  (Strata plus the NVFP4 pack path), under the same licence.
- On top, Impulse's `numa-always` commits (2026-10-23/24), in this order: Strata 0.2.2's NUMA-split CPU expert pool
  (4abac2ad..9e3e82c9, ported from upstream), the sandbox layout detection (IMPULSE_NUMA_NODES, sysfs, getcpu;
  4557e930), the NVFP4 AVX-512BW rows for CPUs without VNNI/VBMI (99cd4091) and their bit-exact rewrite (8ee6d703),
  the per-node arena placement and its page-cache fix (6d63fe1e, 9c2ed715), the checkpoint-skip fix (79fdc144), and
  opt-in experiments that stay off (helper threads, expert deferral, async host loop, STRATA_POOL_SMT,
  STRATA_POOL_FUSE). Imported from commit d7fc0e12 of that branch with `git archive` (no history); `docs/media` left out.

What the reference engine runs in production on the test server is this code at 8ee6d703 (= the test server's 5fbfb2a7; the later d7fc0e12 adds only an
opt-in switch that is off), image `nebula-flashnext:strata-nvfp4-v0.1.44.1-nvfp4.2-numa-5fbfb2a7-r768`.

## Build

As Strata: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` (CUDA 12.9+ for Blackwell), the
server `serve/` (Python) starts `build/strata`. Or inside Impulse: `cmake -S impulse -B build-impulse -DIMPULSE_FLASHNEXT=ON`
(below, One Engine P1) builds this tree as its own project into `bin/`: `libimpulse_fx.so` (the engine impulse-server
loads for a pack), `strata`, `libstrata_kq.so`. The reference image is built from a `build/strata` binary on top of the
base image `nebula-flashnext:strata-nvfp4-v0.1.44.1-nvfp4.2` (see `docs` in the Nebula repo).

## One Engine P0 (2026-10-25, branch one-engine)

The first step of putting this engine inside impulse-server (One Engine, planned in the project's private
records). Nothing here changes what the reference engine runs until a later phase is cut over.

- **One build.** (P1 changed how: see One Engine P1 below; the parent-ggml path described here stays for a project that
  adds this folder with `add_subdirectory`.) `impulse/CMakeLists.txt -DIMPULSE_FLASHNEXT=ON` added this folder with `add_subdirectory`: every target
  below is the same as in a standalone build, with this CMakeLists' own per-file flags, and the CUDA architectures are
  Impulse's list. When the ggml targets already exist (Impulse has added llama.cpp b6b003d) this CMakeLists uses that
  ggml and forces no `GGML_*` cache value; `STRATA_GGML_DIR` is Impulse's checkout, for the ggml-cuda MMQ sources.
  Built alone, it still fetches (or takes `STRATA_GGML_DIR` / `FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP`) the pinned
  `3cf03257` as before. b6b003d (2026-09-13) vs 3cf03257 (2026-09-20) in what this engine uses: `ggml-cpu`'s dots and
  quants, `ggml-common.h`, `ggml-cuda/mmq.cuh`, `quantize.cu` and the MMQ template instances are identical; `common.cuh`
  only gains `fast_bf16_hardware_available`; the other changes (ggml-cpu's F16 matmul source, ops, ggml.c ops, the
  ggml-cuda kernels this engine does not compile) are outside it. Inside Impulse, ggml-cpu is built as Impulse's
  (`IMPULSE_PORTABLE`; `STRATA_PORTABLE` / `STRATA_ISA_FLOOR` then do not change ggml), with OpenMP on.
- **The engine as a library.** `src/program/generate.cpp` built with `STRATA_ENGINE_LIBRARY` (the `strata_generate`
  library) names its `main` `strata_main` (`include/strata/program/entry.hpp`); `strata` is `src/program/strata_main.cpp`,
  which only calls it. The Metal and SYCL builds still compile generate.cpp as their own `main`. `libstrata_kq.so` goes
  beside the binaries when a runtime output folder is set (Impulse's `bin/`), where `moe_mmq.cu` looks for it.
- **One CPU expert pool.** Impulse's copy (`impulse/third_party/strata/`) is merged into this tree's
  `src/kernels/cpu/` and `include/strata/` and removed there; `impulse/third_party/strata/UPSTREAM` lists every file and
  which version is kept. Added to this tree from Impulse: `NativeAct` (SwiGLU_ggml, GeGLU: llama.cpp's CPU rounding),
  `separate_down`, per-job NVFP4 second-level scales (`ExpertJobMulti::s_gate/s_up/s_down`, on every path including the
  Q2 rows and STRATA_POOL_FUSE's mode 7), `hidden`, `kNativeMaxFF` 1536 / `kNativeHBytes` 2048 (LFM2.5's experts; the
  pool's per-expert buffers grow by ~3.5 MB), `numa_split_possible`, `IMPULSE_NVFP4_512` / `nvfp4_rows_exact`, and
  upstream's lazy statics (#2324) and AVX-only IQ4_NL decline (#2638). A pack blob (`NativeAct::SwiGLU`, the three
  slices together) keeps its NVFP4 scale tail exactly as before. Checked bit for bit against both old copies.
- `tools/check_isa_guard.py`: `nvfp4_avx512bw.cpp` (the numa-always BW twin) added to the wide-ISA files; the CPU
  ctest was failing on it.

## One Engine P1 (2026-10-25, branch one-engine-p1): the engine inside impulse-server, step 1 (the bridge)

- **libimpulse_fx.** `-DSTRATA_FX_LIBRARY=ON` builds `impulse_fx` (`libimpulse_fx.so` / `impulse_fx.dll`): strata_generate
  behind two exported functions, `impulse_fx_abi()` (1) and `impulse_fx_run(argc, argv, in, out)` (`src/program/fx_bridge.cpp`:
  strata_main with the serve loop's input and output on the caller's pipes; OS handles, not C runtime descriptors). With the
  option every target of this tree, its ggml too, is compiled position-independent with hidden visibility, and the library's
  version script (`src/program/impulse_fx.map`) exports `impulse_fx_*` only (`--exclude-libs,ALL`, `-z defs`).
- **Why a shared library (the link question P0 left).** P0 found the MMQ prompt kernels (`strata_mmq`: ggml-cuda's
  `quantize.cu`, its MMQ instances and `ggml_cuda_host.cu`, compiled with `-use_fast_math`) defining 10 symbols that
  impulse-server's ggml-cuda defines too (without it), and the CPU pool built twice (`impulse_pool` with Impulse's flags,
  `strata_kernels_cpu` with Strata's: 112 symbols). In one static link only one copy of each survives, decided by link
  order: a fast-math or plain kernel, an AVX-VNNI pool row or not, possibly different per symbol, and for CUDA kernels the
  host stub one module registers can be the other module's. Renaming one side (objcopy prefixes) would have to cover C++
  templates, CUDA host stubs and device registration, and be repeated on every upstream sync. As a library the engine is the
  same objects the standalone `strata` is linked from, only `-fPIC -fvisibility=hidden` (non-interposable, so inlining and
  FP contraction are decided as in the `-fPIE` executable); nothing of it can bind to impulse-server's copies.
- **Built as its own project.** `impulse/CMakeLists.txt` (`IMPULSE_FLASHNEXT=ON`) adds this tree with ExternalProject, its
  own configure: Strata's ggml options (static, no OpenMP, `STRATA_PORTABLE` = ggml-cpu at the AVX2 baseline as the reference engine's
  engine is built; `IMPULSE_FX_PORTABLE=OFF` for the build host's features), its own ggml-base / ggml-cpu from the same
  llama.cpp checkout (b6b003d), `IMPULSE_FX_CUDA_ARCHITECTURES` (default Impulse's list; the test server's engine is built for `120`),
  `STRATA_BUILD_TESTS` = `IMPULSE_TOOLS`. impulse-server's own ggml stays exactly as before (not position-independent), so
  the GGUF engines (P4) do not move.
- **impulse-server's fx engine** (`impulse/src/fx/`): `-m <pack folder>` (pack/index.txt + a `*dense*.gguf`) or the dense
  GGUF beside `pack/` selects it (`fx_pack.hpp`; a qwen4exp GGUF without a pack still goes to the q35 engine). It dlopens
  the library (RTLD_LOCAL; `--fx-lib`, `$IMPULSE_FX_LIB`, default beside the binary), runs the serve loop on a thread over
  two pipes (`GEN` / `T` / `PP` / `RESUME` / `DONE` / `ERR` / `STOP`, as `serve/server.py`), and serves llama-server's API:
  the dense GGUF's vocab and template through llama.cpp (vocab-only), the request's sampling fields as server.py's
  `sampling_keys`, streaming. Engine flags: `--fx-config <Strata run config>` (args, gpu + layer_split, env) or
  `--fx-args ...`; `--fx-cli` runs the engine's own command line in-process (one-shot tools such as `--dump-logits`).
- **Gate P1a (tokenizer, the dev machine, 2026-10-25):** llama.cpp's tokenizer from the reference engine's dense GGUF metadata and
  `tools/strata_tokenizer.py` from `pack/tokenizer` (as server.py builds it): 2,366 tokenizations (the round-trip corpus,
  edge cases, 2.1 MB of real text, 1.3 MB of ChatML prompts with special tokens; parse_special off and on), 3.2 M tokens,
  0 mismatches (`impulse/tests/fx_tokenizer_parity.py`).
- **Gate P1 (the test server, 2026-10-25) and the fix (branch one-engine-p1b).** In-process the engine first ran with ~430 MiB more
  VRAM free (23 + 27 more cache slots), other logits from position 2, and its first 17.5K read took 21.1 s against 10.1 s:
  impulse-server's vocab-only load had started CUDA (ggml-cuda's device init, llama's free-memory query) before the
  engine set `CUDA_MODULE_LOADING=EAGER`, so kernels loaded lazily. impulse-server now sets EAGER with the engine's
  environment and loads the vocabulary only after the engine's READY, with no device. `--expert-cache-later N` (testing,
  default off) caps a layer split's later cards so a gate can pin every card's cache; gate P1b compares today's binary,
  this tree's `strata` and impulse-server three ways.
- **Gate P1b (the test server, 2026-10-25) and the front split.** The EAGER fix brought the in-process 17.5K read to today's speed
  (1,727-1,764 against 1,736-1,775 tok/s), and this tree's standalone `strata` gave teacher-forced logits byte-identical
  to the live binary. In-process the engine still saw ~110 MiB less free VRAM per card: under EAGER, impulse-server's own
  statically linked ggml-cuda kernels were loaded into the engine's contexts too. So a pack now runs in **impulse-fx**
  (`impulse/src/fx/`, a project of its own built with ggml's CPU backend only - no CUDA code at all): impulse-server execs
  it with the same arguments, and it loads libimpulse_fx. The process that runs a pack holds the engine's CUDA code
  alone, as `strata` does; impulse-server and its GGUF engines are as before. (Chosen over moving the GGUF engines into a
  plugin library: that would recompile ggml-cuda and the GGUF engines position-independent, a change P4 would have to
  prove bit for bit, for no gain to a pack.) `STRATA_LOGPOS_HASH=<path>` (default off) writes an FNV-1a hash of every
  teacher-forced row's raw logits, for gate P1c.

## One Engine P2 (2026-10-25, branch one-engine-p2): impulse-fx serves the reference engine's callers as serve/server.py does

- **What.** `POST /v1/chat/completions`, `GET /health`, `/v1/models`, `/props`, `/slots` answer as this tree's
  `serve/server.py`: `src/fx/fx_front.*` is server.py's request and answer layer in C++ (openai_to_messages, the literal
  marks, the chat template through llama.cpp's Jinja engine with Python's float repr and `str.strip()`
  (`llama-hooks/fx-jinja-python.cmake`, impulse-fx's private copy), Strata's tokenizer rules on llama.cpp's vocabulary,
  Service.prepare, sampling_keys, OutputParser, StopMatcher, structured output); `src/fx/fx_server.cpp` runs Service._run's
  passes (EOS literal, thinking budget, forced calls, #879 retry, #606 guard), the single turn with keep-alives, STOP when a
  client leaves, the #481 silence watchdog; `src/fx/fx_vision.cpp` is server.py's Vision (strata-vision, GENI). Every
  field and behaviour, what is used and what is not here: `impulse/docs/FX-PARITY.md`.
- **The run config is read as server.py reads it**: `--fx-config` takes `model_name`, `aliases`, `vision`,
  `stop_token_text`, `repeat_stop_tokens`, `tool_name_repair`, `fit_max_tokens`, `reasoning_budget_tokens`,
  `engine_silence_s`; keys that would change answers and are not ported are refused at start.
- **After P3 (branch one-engine-p3-fix):** the run config's `"log"` is honoured as server.py does (appended; its
  "engine started" line; the engine's stderr there, the front's own lines there and on the console), and
  `--fx-config-json JSON` takes the run config inline (Nebula's `flashnext.py` passes a config it builds that way).
- **The reference engine's image runs an older Strata server** (`~/nebula-images/reference/strata-serve`) than this tree's
  `serve/server.py`, the reference here: FX-PARITY.md "The reference" lists what callers would see differ.
- **Checked on the dev machine (CPU):** `tests/fx_front_parity.py` (server.py's own functions vs `impulse-fx --fx-front-test`):
  589 requests, 349,513 prompt tokens, 0 id mismatches; 1,914 parser feeds and 27 scripted engine runs, 0 mismatches.
  `tests/fx_server_test.py` end to end on the mock engine (which now answers GENI and per-turn replies). Gate P2 (a two-GPU test host:
  speed, pictures, replay against server.py on the same engine, 45-min soak, b2 with the prefill CPU share off) is
  a host-specific gate script, not part of this repository.
