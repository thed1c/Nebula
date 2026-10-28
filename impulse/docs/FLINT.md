# Flint 40B-A2.5B

Impulse runs Recursive's Flint (`flint`, [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), Apache 2.0) on the
same engine as Qwen3.6 and LFM2.5 (`src/qwen35/`): experts in system RAM, an adaptive expert cache on the card, the CPU
expert pool, async swaps, CPU/GPU overlap, lent prompt reads; or every expert on the cards over several GPUs. The tokenizer
and chat template come from the GGUF through llama.cpp.

## Checkpoints

Two sizes, each also as an uncensored build (dima6312's abliterated Flint, the same architecture and tensor names):

| Size | File | GB | SHA256 |
|---|---|---|---|
| Q4 | [unsloth/flint-GGUF `flint-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/flint-GGUF/blob/20468edfc38ab43d6914cc88b08cbecb05d0fc62/flint-UD-Q4_K_XL.gguf), rev `20468edf` | 25.1 | `63c2ecc4…6e072f` |
| Q6 | [unsloth/flint-GGUF `flint-UD-Q6_K_XL.gguf`](https://huggingface.co/unsloth/flint-GGUF/blob/20468edfc38ab43d6914cc88b08cbecb05d0fc62/flint-UD-Q6_K_XL.gguf), rev `20468edf` | 36.3 | `08623109…1279c2` |
| Q4, uncensored | [dima6312/Flint-40B-A2.5B-abliterated-GGUF `…-Q4_K_M.gguf`](https://huggingface.co/dima6312/Flint-40B-A2.5B-abliterated-GGUF/blob/7dcfd8c19fa6702a112b9626b76c89d6c799307b/Flint-40B-A2.5B-abliterated-Q4_K_M.gguf), rev `7dcfd8c1` | 24.2 | `5ac7c931…9c201cf7` |
| Q6, uncensored | [dima6312/Flint-40B-A2.5B-abliterated-GGUF `…-Q6_K.gguf`](https://huggingface.co/dima6312/Flint-40B-A2.5B-abliterated-GGUF/blob/7dcfd8c19fa6702a112b9626b76c89d6c799307b/Flint-40B-A2.5B-abliterated-Q6_K.gguf), rev `7dcfd8c1` | 32.7 | `0e8c78c0…46864475` |

The full hashes are in `nebula/catalog/impulse.json`. Chosen against the BF16 weights on code (below): of the Q4 files
tried (Recursive's NVFP4, ggml-org's Q4_K_M, unsloth's UD-Q4_K_XL) and the Q6 files (lmstudio's Q6_K, unsloth's
UD-Q6_K_XL), unsloth's are the closest. Recursive's NVFP4 is the furthest (top-1 94.3 %, KL 0.032) and is not offered;
Impulse still runs NVFP4 files (the engine checks below used it).

## The architecture

49 blocks, hidden 2048, vocabulary 150,144, 262,144 context. Every block is GQA attention (32 query heads, 4 KV heads, head
128) with per-head Q/K RMSNorm before RoPE and a learned sink logit per head (added to the softmax denominator, no value).
35 blocks are sliding-window attention over 513 tokens (the current one and 512 before, llama.cpp `LLAMA_SWA_TYPE_STANDARD`)
with NeoX RoPE (theta 10000, the whole head); the other 14 are full attention with no positional encoding. Blocks 0-1 have a
dense SwiGLU FFN (6144); blocks 2-48 have 256 experts (SwiGLU, 512 wide), 6 chosen per token by sigmoid scores, the chosen
scores renormalized to sum to one, plus one ungated shared expert (1024). Untied embedding and output head. The NVFP4 GGUF
carries per-tensor and per-expert second-level scales, applied as llama.cpp applies them.

llama.cpp support landed upstream after Impulse's pinned `b6b003d` (ggml-org/llama.cpp#30928, with the per-expert-token
down-projection rescale of #30861 and the template's hex escapes of #30929). `patches/llama.cpp-b6b003d-flint.patch` is
those changes adapted to `b6b003d`; the build applies it to a private copy of the llama.cpp tree (the shared checkout is
never changed). The graph in `src/qwen35/q35_engine.cpp` (`H.flint`) mirrors that `src/models/flint.cpp` op for op.

Two differences from llama.cpp's own run, both off for the bit-for-bit gates:
- **The sliding layers read only their window.** Impulse keeps one KV cache for every layer (as llama.cpp's `--swa-full`) and
  each step's sliding layers read the cells from the first one any row of the step can see (aligned to 256 cells).
  `IMPULSE_SWA_FULL=1` reads every cell, masked, as `--swa-full` does. Same values, another summation split (llama.cpp's
  own compact window cache differs from its `--swa-full` by the same amount on a 2.7K-token prompt: KL 0.003).
- The KV cache is not yet compact for the sliding layers: every layer keeps every cell (14 full + 35 sliding layers,
  98 KB per token at f16, 26 KB at q4_0). A compact window cache is a follow-up.

## Run

```bash
impulse-server -m flint-UD-Q4_K_XL.gguf -c 65536            # one card: the expert cache in front of the CPU pool
impulse-server -m flint-UD-Q4_K_XL.gguf -c 65536 -ts 16,12 --n-cpu-moe 0   # every expert on two cards (25.1 GB)
```

Defaults for Flint: every block's experts in RAM (`--n-cpu-moe 49`), the model card's sampling (temperature 1.0, top-p 0.95,
top-k 20, min-p 0). Text only, no MTP head. Reasoning (`<think>`) arrives as `reasoning_content`; tool calls (the template's
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`) as OpenAI `tool_calls`, and a tool result goes back
as `<|observation|>`.

## Validation (a test PC, 2026-10-26: RTX 4070 Ti SUPER sm_89 + RTX 3060 sm_86, Xeon Gold 6128, CUDA 13.1)

The engine checks ran on Recursive's NVFP4 file (22.9 GB). Six prompts (16 to 2,701 tokens; chat, code, a tool call, a repeated passage), greedy continuations of up to 200 tokens.

| Check | Result |
|---|---|
| Impulse vs llama.cpp b6b003d + the Flint patch, every expert on the two cards, f16 KV, `--swa-full` / `IMPULSE_SWA_FULL=1` | logits bit for bit at every step, forced and greedy, 6/6 prompts (926 steps) |
| Impulse windowed sliding layers vs the same reference | top-1 196-197/200, KL 0.0017-0.0024 (llama.cpp compact vs `--swa-full`: 193/200, KL 0.0032) |
| One card: 128 expert slots per layer, CPU pool, adaptation, async swaps, overlap, vs the all-GPU reference | top-1 198-199/200, KL 0.0007-0.0024, hit rate 0.73-0.81 |
| llama.cpp NVFP4 vs the BF16 checkpoint (fp32 torch reference of the official SGLang kit's math, CPU) | 16-token prompt: top-1 188/201, KL 0.021; 1,047-token prompt: 191/201, KL 0.045 |
| Upstream llama.cpp master (`d854d98`) on the same file | the same answers |
| API | chat with reasoning split, a tool call and its round trip, prompt-cache reuse |

A llama.cpp `b6b003d` finding (not Impulse's): with experts in RAM and op offload (`--n-cpu-moe` and a prompt of 32+ tokens), a
CUDA fusion corrupts Flint's MoE output there (garbage text); `GGML_CUDA_DISABLE_FUSION=1` or `--no-op-offload` fixes it,
and upstream master is correct. Impulse's own expert paths are not affected (the rows above). The reference runs above use
every expert on the cards.

## Speed (the same cards, 64K context, q4_0 KV, server timings)

| Shape | Read 0.9K | Read 13.8K | Writing short | Writing after 13.8K |
|---|---|---|---|---|
| Impulse, 4070 Ti SUPER + 3060, every expert on the cards | 1,930 tok/s | 2,113 tok/s | 102.5 tok/s | 82.3 tok/s |
| llama.cpp b6b003d + Flint patch, same cards (its compact SWA cache) | 1,861 | 2,087 | 100.8 | 77.0 |
| Impulse, one 4070 Ti SUPER (142 slots per layer, pool 8 threads) | 365 | 1,840 | 41-51 | 43-58 |
| llama.cpp master, one 4070 Ti SUPER, `--n-cpu-moe 30` | 240 | 313 | 37.7-42.7 | 36.5 |

## The offered files on one RTX 4070 Ti SUPER 16 GB (`bench/flint-quality/bench.py`, 2026-10-27)

Top-1 and KL: against the BF16 weights (the uncensored files against the abliterated BF16), six code prompts x 256 greedy
tokens through llama.cpp. HumanEval: 40 problems evenly spaced, greedy, at a 6,144-token cap and again at 16,384 for the
answers the cap cut off. Speed cells: read / writing tok/s (first answer - follow-up), peak card GB, RAM GB (the expert arena
and the process's own; the file's page cache, reclaimable, comes on top). Every speed run answered right.

| Option | GB | HumanEval 6K / 16K cap | Top-1 | KL | 8K | 32K | 64K |
|---|---|---|---|---|---|---|---|
| Q4 (UD-Q4_K_XL) | 25.1 | 36/40 / 40/40 | 96.6 % | 0.0085 | 1515 / 28-47, 15.1, 22.6 | 1572 / 27-45, 15.1, 25.2 | 1468 / 32-39, 15.1, 27.7 |
| Q6 (UD-Q6_K_XL) | 36.3 | 36/40 / 38/40 | 98.0 % | 0.0033 | 1215 / 24-32, 15.1, 33.0 | 1282 / 21-29, 15.1, 35.6 | 1245 / 21-25, 15.1, 38.1 |
| Q4 uncensored (Q4_K_M) | 24.2 | 37/40 / 39/40 | 95.5 % | 0.028 | 1529 / 31-55, 15.2, 22.4 | 1498 / 26-47, 15.2, 25.0 | 1466 / 30-40, 15.1, 27.6 |
| Q6 uncensored (Q6_K) | 32.7 | 37/40 / 40/40 | 97.9 % | 0.0092 | 1159 / 25-34, 15.1, 30.0 | 1328 / 26-33, 15.1, 32.7 | 1345 / 24-30, 15.1, 35.2 |
