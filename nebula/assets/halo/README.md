# Halo's model files

Halo runs Liquid AI's **LFM2.5-1.2B-Instruct**, the makers' own QAD Q4_0 file, unchanged. Nebula fetches it at first run from
Liquid's repository at a pinned revision and checks its sha256 (`nebula/catalog/halo.json`). What Nebula adds ships here, as two
LoRA adapters in llama.cpp's GGUF format, kept in full precision (F16):

| File | What it changes | Used |
|---|---|---|
| `halo-guard-v2-lora-f16.gguf` | Halo's own skills: the guard (Halo's security-log tools), the router judge (`nebula/stage.py`), handoff briefings (`nebula/handoff.py`), and direct answers | always |
| `halo-uncensored-heretic-t198-lora-f16.gguf` | removes the stock refusals (directional ablation by Heretic 1.4.0, trial 198 of its study, recomputed on the Q4_0 weights) | only when the owner picks Uncensored |

These adapters are modifications of LFM2.5-1.2B-Instruct (Copyright Liquid AI) and are distributed under the **LFM Open
License v1.0** (`LICENSE-LFM-1.0`, the text from Liquid's repository). Changes made: the Halo adapter is a rank-16 LoRA
trained on data generated from Nebula's own code (guard looks, router requests and handoff conversations written by a local
model) plus an Apache-2.0 chat slice (HuggingFaceTB/smoltalk); the Uncensored adapter is a rank-3 LoRA per Heretic's
row-normalised ablation of the attention, short-convolution and feed-forward output projections. The training data, its
generators and the measurements are in the project's design records (Halo model plan, 2026-10-20/21).

Held-out results with Liquid's Q4_0 file in llama.cpp: guard 330/330 right actions in both modes; router judge 93 %; handoff
briefings graded 1.97/2.00 faithful/constraints against the conversation; refusals on AdvBench's held-out prompts 100/104
(Standard) and 5/104 (Uncensored).
