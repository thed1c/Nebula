# Halo's model files

Halo runs one of Liquid AI's models, the makers' own QAD Q4_0 file, unchanged: **LFM3-1.7B** with an NVIDIA card (Impulse),
**LFM2.5-1.2B-Instruct** without one (Impulse portable, llama-server). Nebula fetches the one the machine runs from Liquid's
repository at a pinned revision and checks its sha256 (`nebula/catalog/halo.json`). What Nebula adds ships here, as two LoRA
adapters per model in llama.cpp's GGUF format (F16):

| File | What it changes | Used |
|---|---|---|
| `halo-lfm3-v3.1-r16-lora-f16.gguf` | LFM3: Halo's own skills, as below (Halo v3.1: rank 16 on every linear, the per-head attention gate included) | always, on LFM3 |
| `halo-lfm3-uncensored-t43-lora-f16.gguf` | LFM3: removes the stock refusals (Heretic's directional ablation, done on LFM3 with Nebula's own code: trial 43 of 52, over Q4_0 + v3.1, at 1.5x strength) | Uncensored, on LFM3 |
| `halo-guard-v2-lora-f16.gguf` | LFM2.5: Halo's own skills: the guard (Halo's security-log tools), the router judge (`nebula/stage.py`), handoff briefings (`nebula/handoff.py`), and direct answers | always |
| `halo-uncensored-heretic-t198-lora-f16.gguf` | LFM2.5: removes the stock refusals (directional ablation by Heretic 1.4.0, trial 198 of its study, recomputed on the Q4_0 weights) | only when the owner picks Uncensored |

These adapters are modifications of LFM3-1.7B and LFM2.5-1.2B-Instruct (Copyright Liquid AI) and are distributed under the **LFM Open
License v1.0** (`LICENSE-LFM-1.0`, the text from Liquid's repository). Changes made: the Halo adapter is a rank-16 LoRA
trained on data generated from Nebula's own code (guard looks, router requests and handoff conversations written by a local
model) plus an Apache-2.0 chat slice (HuggingFaceTB/smoltalk); the Uncensored adapter is a rank-3 LoRA per Heretic's
row-normalised ablation of the attention, short-convolution and feed-forward output projections. The training data, its
generators and the measurements are in the project's design records (Halo model plan, 2026-10-20/21).

Held-out results with Liquid's Q4_0 file in llama.cpp: guard 330/330 right actions in both modes; router judge 93 %; handoff
briefings graded 1.97/2.00 faithful/constraints against the conversation; refusals on AdvBench's held-out prompts 100/104
(Standard) and 5/104 (Uncensored).

Halo v3.1 (LFM3-1.7B QAD Q4_0 + its adapters in Impulse, Liquid's sampling, thinking off): guard 329/330 (Uncensored 327);
router judge 263/279; handoff briefings 0/38 with an invented detail (one full read) and 0/24 after rolling updates; refusals
on AdvBench's held-out prompts 98/104 (Standard) and 2/104 (Uncensored). Its training data are v2's (above), with LFM3's own
refusals of AdvBench's training prompts and its own answers to harmless_alpaca prompts added (self-distillation).
