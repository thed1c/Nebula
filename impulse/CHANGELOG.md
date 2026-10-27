# Changelog

Impulse was called Strata-Lite until 2026-10-17. Its versions restart with the first release that comes with Nebula:
**Impulse 0.1.1 is Strata-Lite 0.2.4** (the dense models, several GPUs, parallel slots and the in-house work after it, below).
The sections from 0.2.3 down are Strata-Lite's own numbers.

## Unreleased
- **Lazy verify logits (from Strata-Lite v0.2.4-exact1, dfb7bd6e), for One Engine P4 on the test server:** `IMPULSE_LAZY_LOGITS=1` (or the
  Strata-Lite name `STRATA_LITE_LAZY_LOGITS=1`, which Dense's definition sets) keeps a server step's logits on the cards and
  copies only the rows the accept loop reads into pinned memory (the same floats, the same tokens). Off unless set. Built with
  llama.cpp b6b003d + the exact1 P2P all-reduce patch (`GGML_CUDA_AR_P2P`) for two dense GGUF engines.
- **impulse-fx after One Engine P3:** the run config's `"log"` is honoured as `serve/server.py` does (appended, its
  `[strata] <time> engine started: ...` line, the engine's stderr there; the front's own lines there and on the console);
  `--fx-config-json JSON` passes the run config inline (one of `--fx-config`, `--fx-config-json`, `--fx-args`).
- **One Engine P1 (branch one-engine-p1): Strata's Flash-Next engine inside impulse-server.** `-m <pack folder>` (or its
  `*dense*.gguf` beside `pack/`) runs the fx engine: `libimpulse_fx` (Strata's engine as one shared library, its own CPU
  pool, MMQ kernels and ggml kept private) loaded at run time, its serve loop on a thread, impulse-server's OpenAI API in
  front (llama.cpp's tokenizer and template from the dense GGUF; chat and completions, streaming, max_tokens, sampling
  fields, stop strings, enable_thinking), in impulse-fx (no CUDA of its own), which impulse-server execs for a pack. `--fx-config` / `--fx-args` give the engine's flags, `--fx-cli` its own command
  line. A GGUF without a pack is served as before. `-DIMPULSE_FLASHNEXT=ON` builds `flashnext/` as its own project into
  `bin/`. Tokenizer parity with Strata's (gate P1a): 0 mismatches in 3.2 M tokens. Not yet: images, tools beyond llama.cpp's
  parser, server.py's conversation-cache controls, slots, SAVE / RESTORE / VRAM, metrics (P2).
- **Test server check, 2026-10-24 (RTX 5080 sm_120 + Xeon Gold 6140, `tools/nvfp4_scale_check.sh`):** NVFP4 second-level scales
  pass: Impulse == llama.cpp bit for bit on the scaled known-answer GGUF (ggml expert paths, CPU pool exact rows, baseline),
  and variant == original through slot cache + pool + overlap (twin check 1016 rows clean) and lent reads; pool-scales-test
  passes. llama.cpp itself is not exactly scale-invariant on sm_120's NVFP4 GPU path (variant vs original KL 0.0022), which is
  also why the MTP drafter run (same drafts, 80 of 141 accepted, logits not identical) is not a pass/fail signal. Two-socket
  placement on the test server: 8.44 + 8.44 GiB, 98.8 % / 101.2 % of each slice on its node, all 16 copiers bound.
- **The 512-bit NVFP4 rows are the default** (IMPULSE_NVFP4_512=0 or IMPULSE_NVFP4_EXACT=1: the exact AVX2 rows, llama.cpp's
  CPU logits bit for bit). The first gate (exact vs 512-bit, KL 0.038) was below this test model's noise floor, not a bug:
  on the test server over the same 3,914 forced tokens (Qwen3.6-35B-A3B requantized to NVFP4), against llama.cpp with every expert on
  the GPU (impulse/docs/CORRECTNESS.md's reference), llama.cpp's own CPU experts score KL 0.049 / top-1 93.3 % (Impulse's
  exact rows: the same, bit for bit), the 512-bit rows KL 0.043 / top-1 94.2 %: closer to the reference than llama.cpp's CPU.
  On the model's real expert rows the 512-bit and AVX2 kernels agree to rounding (1-32 tokens a call).
- **NVFP4 experts on the CPU pool** (ggml type 40): the strata-nvfp4 fork's NVFP4 rows in `third_party/strata/` - AVX-512
  VNNI, an AVX-512BW twin for CPUs without VNNI/VBMI (Skylake-SP; bit for bit the VNNI rows) and AVX2. A layer's gate/up and
  down rows decode each block once for every token of a window (2.0-2.25x ggml-cpu's dot at 4-8 tokens on a Xeon Gold 6242R,
  AVX2 rows). Impulse's engines keep their bit-for-bit results: they take the AVX2 rows, which equal ggml-cpu's dot exactly;
  `IMPULSE_NVFP4_512=1` takes the faster 512-bit rows (float sums in another order). NVFP4's second-level scales
  (`ffn_*_exps.scale`) go per job to the pool. `nvfp4-parity` (`-DIMPULSE_TOOLS=ON`) checks every kernel against ggml-cpu
  and an Impulse expert bit for bit.
- **The 512-bit NVFP4 CPU rows are the engines' default** (VNNI, or the BW twin on AVX-512 without VNNI; AVX2 elsewhere):
  within float rounding of llama.cpp, not bit for bit. `IMPULSE_NVFP4_512=0` (or `IMPULSE_NVFP4_EXACT=1`) keeps the AVX2
  rows equal to ggml-cpu's dot, for llama.cpp's logits bit for bit. Gate: mean KL <= 0.005 against the exact rows with every
  expert on the CPU (`tools/nvfp4_scale_check.sh kl`).
- **NVFP4's second-level scales loaded and applied as llama.cpp b6b003d does** (its generic pass in `llama-model.cpp`: the
  optional `<weight>.scale` tensors, ModelOpt's `weight_scale_2`): the same ops in the same order (`build_lora_mm`,
  `build_ffn`, `build_moe_ffn`), so a GGUF with scales gives llama.cpp's logits; a file without them builds the same graphs
  as before. Qwen3.5/3.6/3.8 and Flash-Next: attention Q/K/V/output, the delta-net's qkv/gate/beta/alpha/output, the shared
  expert, dense FFNs, the output (NVFP4 only, refused beside a tied one) and the per-expert `ffn_{gate,up,down}_exps.scale`
  on every expert path (whole-layer GPU, cache slots - gathered by expert id, not slot -, the CPU pool with each job its
  expert's scales, overlap, the dcheck twins, the read's expert ring), and in the MTP drafter (eh_proj, attention, experts,
  shared expert, its head). LFM2: attention and output only (llama.cpp's `lfm2.cpp` leaves its FFN and expert scales
  unused, with a message here). Gemma 4: Q/K/V/O, the dense and shared MLPs and the output (its per-expert
  `ffn_down_exps.scale` was already applied). The pool applies the scales on every kernel (Strata's Q2_0 rows had dropped
  them). `pool-scales-test` checks the pool's scaled jobs bit for bit against llama.cpp's scale ops for NVFP4, Q8_0, Q4_K,
  IQ4_XS and Q6_K. `tools/nvfp4_scale_variant.py` writes a known-answer GGUF (real power-of-two scales, the block scales
  moved by the inverse exactly: the same function as the original); `tools/nvfp4_scale_check.sh` runs the GPU checks on it
  (Impulse vs llama.cpp, variant vs original, every expert path, the drafter) and the 512-bit rows' KL gate.
- **Per-socket placement of the CPU experts** under the pool's NUMA split (numa-always): each socket's half of every expert's
  rows - the rows that socket's pool threads read - is copied into the RAM tier's arena by threads pinned to that socket
  (MPOL_BIND when allowed, plain first touch inside a sandbox that refuses it), the model file leaving the page cache first;
  a line reports where each half went and `/proc/self/numa_maps` checks it. On by itself wherever the split runs (two
  sockets, `--pool`, async swaps); `IMPULSE_NUMA_PLACE=0` turns it off; one socket is unchanged. The same bytes, so the same
  results (`numa-place-test`).

## Impulse 0.1.1: the first release with Nebula, experimental
- Built for every NVIDIA generation from the RTX 20 series to the RTX 50 series, plus A100, H100 and B200 (sm_75, 80, 86,
  89, 90, 100, 120a; CUDA 13). The V100 (sm_70) needs a CUDA 12 build of its own (CUDA 13 dropped it).
- Renamed from Strata-Lite: `impulse-server`, `impulse-gemma4`, `impulse-qwen35`, the `IMPULSE_*` build options and
  environment variables, the `impulse:` log lines, `--alias impulse` and the `/props` key `impulse`. `/props` `build_info`
  names the version. Strata's CPU expert pool is kept with its notice in `third_party/strata/` (`THIRD_PARTY_NOTICES.md`).
- **Faster penalties sampler** (LFM2.5's default repeat penalty): applied to the tokens of the last `repeat_last_n` only,
  instead of a lookup for every one of the 128K candidates each token; the same arithmetic, the same results. LFM2.5-24B
  writing on an RTX 5070: 102.5 -> 120.7 tok/s at 4K, 85.9 -> 96.6 at 16K. `-DIMPULSE_PENALTIES_OLASTN=OFF` keeps llama.cpp's.
- **Fixed:** an expert model's prompt read could pick a chunk larger than ggml-cuda's expert matmul allows
  (`GGML_ASSERT(nbytes_shared <= smpbo)` in `mmid.cu`; LFM2.5 at 32K). Chunks now fit the card: about 24K tokens on
  RTX 30 / 40 / 50, 16K on RTX 20.
- **Fixed (2026-10-20), three slowdowns against Strata-Lite 0.2.4:** (1) the refill after a lent read never ended, because the
  ordinary adaptation swaps a few experts almost every step, so every step kept the refill's 16 swaps per layer; it now ends when
  no layer fills its whole piece (both engines). (2) The Qwen drafter computed its pick's probability, a full-vocabulary softmax,
  on every draft step even with `--spec-min-p` off; now only when the gate asks. (3) Gemma's read chunks did not count the
  compute buffers the scheduler already holds, so a 15K read after a 15K read fell to 4K chunks; now counted, as in the Qwen
  engine. Measured on an RTX 3060 (PCIe 3.0 x16) and RTX 5070 against 0.1.1 before the fixes: Gemma 157K second turn
  28.7 -> 36.6 tok/s (3060); Qwen writing after 4K 76.0 -> 85.9 and its 157K turns 39.5 / 41.9 -> 45.8 / 48.9 (5070);
  Gemma's second 15K read 1,303 -> 1,625 (3060) and 1,980 -> 2,472 (5070). Faster than llama.cpp (b6b003d) on every row of
  the release benchmark on both cards (prompt reads 2.4-8x, writing 1.1-1.8x).
- Everything listed under Strata-Lite 0.2.4 below.

## Strata-Lite 0.2.4 (unreleased as Strata-Lite; released as Impulse 0.1.1), experimental
- **Dense models.** Gemma 4 dense (the 31B; llama.cpp's dense FFN) and Qwen3.5 / 3.8 dense (`qwen35`, the 27B) load in the engines;
  the expert cache, CPU pool, swaps and lending are off for them (nothing to stream) and they read in `-ub` steps. Bit for bit the same
  logits as llama.cpp (b6b003d) on the same cards and flags.
- **Several GPUs.** `-ts / --tensor-split` and `-sm / --split-mode layer` (both engines): llama.cpp's layer split (each card a share of
  the layers, the output on the last; pipeline-parallel decoding, llama.cpp's reserve sequence). `-sm tensor` (the Qwen engine, dense):
  llama.cpp's tensor split through its meta backend, every layer on every card, our port of its split-state rules; writing about 1.6x
  layer split's on two cards. With a split, the projector's "second card" is the first card without layers.
- **MTP for dense Qwen**: the model's own nextn head or a drafter file, and HauhauCS's FastMTP drafters (a reduced draft vocabulary,
  `d2t`); with several cards the drafter runs on one of them.
- **Parallel slots** (`-np / --parallel N`, both servers, dense and MoE): llama-server's slots, each with its own context share
  (n_ctx / N), prompt cache, sampler and MTP state; a request goes to the idle slot sharing the longest prefix; every generating slot
  steps together. The engines' sequences share one KV cache (each cell tagged with its sequence); Qwen's recurrent layers keep one
  state per sequence, llama.cpp's layout. One sequence alone in a several-sequence engine gives the same bits as a one-sequence engine.
  `--read-interleave N` (Gemma, off): a long read gives the other slots a step every N tokens.
- `/tokenize` and `/detokenize` (llama-server's) in both servers.
- Qwen server: a dense model takes the full checkpoint set on every read (a new question on the same document resumes).
- Build: the q8_0 K + q4_0 V flash-attention kernels (without them ggml converted such a cache to f16 every step).
- **Every model on every layout.** Expert models over several GPUs too (layer split, with `--n-cpu-moe` keeping the first layers'
  experts in RAM, as llama.cpp; our expert cache is one card's, so the split uses ggml's expert path); `-sm tensor` for Gemma 4
  (dense and expert) and for expert Qwen / LFM2.5 models with every expert on the cards; LFM2.5 with parallel slots; Gemma's MTP
  drafter on a split model (on the card of the last layers) and for dense Gemma (HauhauCS's 31B drafter: about 1.9x writing).
  Checked against llama.cpp per cell, bit for bit (LFM2.5 with a layer split: the same arithmetic, other ggml-cuda fusions).
- **Slots like llama-server's**: one KV stream per slot (a slot's attention no longer reads the others' cells), the least recently
  used idle slot for an unrelated request, and a long read with several slots admits new requests between pieces.
- **llama-server's command line and endpoints**: a llama-server deployment's flags start the server (equivalents renamed, the rest
  accepted with no effect and listed), `--help` in llama-server's layout, `--metrics` (`/metrics`), `--timeout`, `--no-webui`,
  llama-server's `/v1/models` and `/props` fields.
- Not yet: MTP with tensor split (Gemma); the expert cache, CPU pool and swaps on more than one card.
- **Later in 0.2.4:**
  - **Qwen3.8 Flash-Next** (llama.cpp's `qwen4exp`) in the Qwen engine: hyper-connection streams, the PLE n-gram rows, the indexer and
    its sparse attention; split GGUFs mapped back to back. Bit for bit the same logits as llama.cpp (b6b003d); a stable `ggml_top_k`
    on CUDA (CUB's broke ties at random, so llama.cpp itself was not reproducible past 2,051 cells).
  - **RAM prompt cache** (`--cache-ram`, Qwen server): finished conversations parked in RAM and resumed; late system messages
    rewritten as user turns.
  - **Images:** llama.cpp's M-RoPE causal rule for image ubatches (also fixes dense Qwen: every image token had seen the whole image).
  - **The expert cache on every card** of a layer split (each card its layers' slots); sparse decode attention over the indexer's picks.
  - **Two-card read pipeline** for expert models with experts in RAM: each card runs its layers on its own thread on a different
    chunk (exact; fusion off gives the plain read's logits). Read chunks for Flash-Next sized from measured buffers (8K at 20-50K).
  - **MTP drafter for Flash-Next** (graph port + `impulse/tools/q4x_mtp_gguf.py`), working with the read pipeline; `--spec-min-p` gate.
  - **The cache refill after a lent read in pieces** (both engines; the first step no longer waits for ~15 GB of copies).
  - **Gemma LoRA adapters** (`--lora`, `--lora-scaled`): llama.cpp's `build_lora_mm`, bit for bit on a Gemma 4 12B + LoRA.
  - Servers bind like llama-server (SO_REUSEADDR, no SO_REUSEPORT); exit at once when the port cannot be bound.
  - Opt-in, not a win yet: a copy/compute expert ring for reads (`IMPULSE_RING`).

## Strata-Lite 0.2.3 (unreleased), experimental
- **Graph reuse** (both engines, the drafters too): a decode step of the same shape as the last one (tokens, padded KV cells, output
  rows) sets the inputs of the graph already on the scheduler and computes it, instead of building the graph node by node, planning
  its allocation and freeing it every token, as llama.cpp's `graphs reused` does. The Nsight profile of 2026-10-16 (an RTX 5070,
  LFM2.5-8B-A1B Q6_K whole on the card) had put 2,272 us of host work per token between graph launches against llama.cpp's 495 us,
  with the GPU doing the same kernels in both. The same bits out: the graph is the one that would have been rebuilt.
  `IMPULSE_GRAPH_REUSE=0` turns it off (as `LLAMA_GRAPH_REUSE_DISABLE`). `/props` shows `graphs_built` and `graphs_reused`.
- **Pinned host buffers kept** (`src/lite/pinned_host.hpp`): the scheduler's CPU-side buffers (the inputs: ids, positions, KV
  indices, the attention mask) come from a pool of pinned host memory that lives as long as the server and only grows, instead of
  ggml-cuda's host type, which calls `cudaMallocHost` for every scheduler buffer and `cudaFreeHost` when the read's scheduler goes.
  The same profile saw three of each in one 15K-token read: 476 ms of a 2.04 s read. The cost: the pool keeps the largest read's
  input buffers (mostly the attention mask, chunk x KV cells x 2 bytes: about 0.5 GB after a 15K read, up to 4 GB after a 120K
  read in 16K chunks) pinned for the server's life. `GGML_CUDA_NO_PINNED=1` turns pinning off, as before. A new region is at
  least twice the largest so far, so a long read grows the pool a handful of times, not once per chunk.
- **The attention mask filled fast** (`src/lite/mask_fill.hpp`): the host-built f16 mask of a ubatch (tokens x KV cells) is one
  unsigned range test per row, vectorized and split over the engine's threads, instead of a per-entry ternary at 1.3 ns an entry
  single-threaded: a 120K-token read in 4K chunks fills 7.8 billion entries, ten seconds of its 42 on an RTX 5070. Every entry gets
  the value it got.
- `IMPULSE_GPU_WAIT=spin`: the host spins while the card computes (`cudaDeviceScheduleSpin`), one busy core per wait. Tried
  as a cure for the slow sampler after each token (1.5 ms a token on a dual-socket Xeon for llama.cpp's chain over the 128K logits,
  0.27 ms in isolation; a CPU-only microbenchmark of that call ran 3.4x slower after a 3 ms idle, the core having clocked down
  into C6); in the server it changed nothing and cost 3 % of decode, so the driver's default stays. That 1.5 ms is still open.
- `IMPULSE_TIMELINE=1`: the Qwen/LFM server prints, per request, the engine's time a token (graph, inputs, compute, logits)
  and the sampler's; `/props` carries the engine's part. (`=2` adds probes, among them each sampler stage's time.)
- Known, documented, not changed: the Qwen/LFM server starts LFM2.5 models from the model card's sampling (temp 0.1, top-k 50,
  repeat penalty 1.05), and llama.cpp's penalties sampler scans all 128K candidates each token: about 1.4 ms a token, most of
  the decode gap to llama-server (whose default penalty is off). `--repeat-penalty 1` or `"repeat_penalty": 1.0` in the request
  turns it off (RTX 3060, the 8B: 128 → 155 tok/s); RUN.md says so.

## Strata-Lite 0.2.2 (2026-10-16), experimental
- **LFM2.5-24B-A2B** (`lfm2moe`) on the hybrid engine (`impulse/docs/LFM25.md`): short-convolution blocks with their own
  state and checkpoints, sigmoid + bias routing, the expert cache, CPU pool, swaps, overlap, lent reads and the KV pager; an optional
  rank-one output LoRA; 128K context. The setup offers AtomicChat's uncensored Q4_K_M (default) and Liquid AI's QAD Q4_0.
- **The RAM tier** (both engines, `--ram-budget GIB`): the experts the card does not hold are locked in RAM up to a budget and the rest
  are read from the model file (the OS file cache, else the disk); by default the engine locks what the PC can spare when it starts
  (available RAM less a headroom: 4 GiB / 15 % on Linux, 8 GiB / 25 % on Windows) instead of all or nothing, so a 32 GB PC no longer
  locks so much that Windows runs out. Bit for bit the same output whatever the budget (only where the bytes come from changes).
  `--ram-budget 0` is low-RAM mode.
- **Speed defaults:** the CPU pool's workers spin 100 us before sleeping, not 20 ms (the same speed, 57 % less CPU time); Qwen3.6 drafts
  2 tokens per step, not 3 (+21 % writing on a short chat, RTX 3060); Gemma 4 drafts 3, and 2 once the context passes 32K
  (`--n-draft-long`, `--n-draft-long-from`).
- **Easier START-HERE:** a short menu on every run (Enter starts the model; change the model or its settings; update Impulse), a
  desktop shortcut on the first install, startup failures said in plain words with the fixes to pick (not enough RAM or VRAM, a port in
  use, a CPU the ready-made engine does not suit, missing CUDA libraries, an old driver, a missing model file), the window titled
  Impulse. UPDATE.bat / update.sh update a Impulse install.
- Fixes: on a 12 GB card at 262K, Gemma read a 17K-32K-token prompt as one chunk that did not fit the card and the server stopped
  (the read's estimate now uses measured sizes and counts the window cache's growth; such prompts read in 16K chunks); upstream
  Strata fixes: no SIMD code before `main` in the AVX-512 / AVX2 / AVX files (#2324), IQ4_NL down rows declined in the AVX1 kernel
  (#2638), a CPU without a `flags` line (#2551), metadata-only GGUF shards (#2491), host compilers for `--build` (#1645), a failed
  build no longer leaves the old version mark (#2054), `.bat` line endings (#1939), UTF-8 file reads (#2648).

## Strata-Lite 0.2.1 (2026-10-11), experimental
- **The KV pager** (both engines): the full-attention KV starts at 32K cells on the card and grows in 16K steps as the conversation
  needs them; the expert cache gives way, and takes the room back when a new conversation starts (`--kv-initial`, `--kv-step`;
  `--kv-initial 0` reserves the whole context at load as before). Bit for bit the same output as the whole-context engine (both
  models, growth during a read, during decode and with MTP). More expert slots for short chats: Qwen 89 vs 73, Gemma 81 vs 70 at 262K.
- **Chat-page compaction**: at 3/4 of the context (192K of 256K) the page summarizes the conversation and goes on from the summary
  plus the last turns, with a marker that opens the summary; a big paste makes room the same way; one that can never fit is refused.
- **Gemma defaults to q8_0 KV** (Qwen keeps q4_0): at 64K, top-1 87.1 % against q4's 80.5 % and an f16 noise floor of 87.1 %.
- KV streaming from RAM measured and set aside for 12 GB cards (`impulse/ROADMAP.md`).
- Fixes: the projector warm-up on a second GPU or the card (an unset `text_len`; it threw and was skipped); an answer with an invalid
  UTF-8 byte made the chat parser fail and the server answer 500 (`src/lite/utf8.hpp`).

## Strata-Lite 0.2.0 (2026-10-10), experimental
- **Qwen3.6-35B-A3B** (`qwen35moe`) engine and server in `src/qwen35/` (`impulse/docs/QWEN36.md`): llama.cpp's graph rebuilt op for op
  (bit-identical through the full forward, the CPU pool, whole GPU layers and q4 KV), Strata's adaptive GPU expert cache, CPU pool,
  background swaps and overlap, MTP from the extracted head with recurrent-state rollback, chunked reads with lent VRAM, the image
  projector with M-RoPE positions, a prompt cache with recurrent-state checkpoints. API parity 63/63. The grid on RTX 3060 and 5070.
- `impulse-server` reads the GGUF's architecture and starts the matching engine's server.
- `tools/qwen35_mtp_extract.py`: the MTP head of a Qwen3.5/3.6 GGUF into its own drafter GGUF.
- `impulse/patches/llama.cpp-b6b003d-impulse.patch`: the build is a patched llama.cpp b6b003d (a sampler-on-logits helper and the
  route-oracle tools), no longer a private copy.
- Cache behaviour flags: `--adapt-half-life`, `--adapt-budget`, `--no-adapt`, `--no-adapt-refill`, `--no-async-swaps`, `--no-overlap`.

## Strata-Lite 0.1.0 (2026-10-10), experimental
- gemma4 engine: the Gemma 4 26B-A4B block from ggml ops, Strata's expert arena, GPU cache, adaptation, CPU pool, chunked reads with
  lent VRAM, background swaps, CPU/GPU overlap, MTP drafting, q4 KV, the image projector on demand.
- `impulse-server`: llama-server-compatible API (63/63 parity checks; `impulse/docs/API.md`).
- Builds for sm_86, sm_89, sm_120.
- The grid: RTX 3060 and RTX 5070 at 32k / 131k / 262k against llama.cpp static offload with MTP (README).
