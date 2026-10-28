# Flint quality and speed benchmark

`bench.py` measures one Flint GGUF through Impulse on one card, the same way every time: HumanEval pass@1 (the official
file, pinned by commit and sha256, 40 evenly spaced problems or all 164), agreement with the BF16 weights (top-1 and KL on
six fixed HumanEval prompts, against a reference made once from the BF16 file), and read / write speed, card memory and
RAM at 8K, 32K and 64K. It appends one JSON line per run and rewrites a Markdown table from all of them. Only numbers are
written (`CONTRIBUTING.md`).

```bash
python3 bench.py ref --bf16 Flint-BF16.gguf --oracle route-oracle --card GPU-… --out ref-flint
python3 bench.py run --gguf flint-UD-Q4_K_XL.gguf --label "Q4 (UD-Q4_K_XL)" --card GPU-… --bin build/bin --ref ref-flint
```

The reference needs llama.cpp's `route-oracle` (`tools/route-oracle`, built from the Flint-patched tree) and the BF16 GGUF
(80 GB; delete it after). `run` needs `impulse-server` and `impulse-qwen35` (`-DIMPULSE_TOOLS=ON`). `--skip speed` or
`--skip humaneval,kl` runs a part; `--known` fills a part measured before with the same settings.

Results for 0.1.4 (an RTX 4070 Ti SUPER 16 GB, a Xeon Gold 6128, CUDA 13.1): `docs/FLINT.md`.
