# Third-party notices

Impulse is MIT-licensed, Copyright (c) 2026 thed1c (`LICENSE`). It contains or builds against the following code, each under its own MIT licence.

## Strata

https://github.com/Niko1221/Strata, MIT, Copyright (c) 2026 Niko1221 and the Strata contributors.

Impulse started from Strata's adaptive expert cache. Its engines (`src/gemma4`, `src/qwen35`, `src/lite`) are its own code. Strata's code that Impulse uses is the CPU expert pool and the headers it needs. Since One Engine P0 (2026-10-25) there is one copy of it, in Strata's tree `flashnext/` (`flashnext/src/kernels/cpu/`, `flashnext/include/strata/`, below), with Impulse's changes merged in; Impulse's engines build it from there. `third_party/strata/UPSTREAM` lists where each file of Impulse's former copy (`third_party/strata/`, from Strata commit `61b3fb5d`, 12 files changed for Impulse, plus the NVFP4 CPU rows from the strata-nvfp4 fork at our numa-always `d7fc0e12`) went, which version the one copy keeps, and the former record with each upstream file's sha256. Strata's licence text stays in `third_party/strata/LICENSE` (the same as `flashnext/LICENSE`). The test `tests/nvfp4_parity.cpp` comes from the strata-nvfp4 fork (https://github.com/sergqwer/strata-nvfp4, MIT, the same licence and copyright as Strata).

### Impulse Flash-Next (`flashnext/`)

`flashnext/` is a whole Strata source tree: the strata-nvfp4 fork (https://github.com/sergqwer/strata-nvfp4, commit
`257727d`, MIT, the same licence and copyright as Strata) with Impulse's own changes on top. Its licence text is
`flashnext/LICENSE`; `flashnext/IMPULSE.md` lists where it comes from and every change. It includes ggml's `ggml-common.h` under
`flashnext/third_party/ggml/` with ggml's licence; built alone it downloads llama.cpp/ggml as Strata's does, built for
Impulse (`-DIMPULSE_FLASHNEXT=ON`: its own project, `libimpulse_fx` and `strata`) it builds its ggml from Impulse's
llama.cpp b6b003d checkout. It also holds Impulse's CPU expert pool (above).

## ggml

https://github.com/ggml-org/ggml, MIT, Copyright (c) 2023-2026 The ggml authors. `flashnext/third_party/ggml/ggml-common.h` (the CPU pool's block layouts; Impulse's own copy in `third_party/ggml/` was the same file and went in One Engine P0), with its licence text in `flashnext/third_party/ggml/LICENSE`.

## llama.cpp

https://github.com/ggml-org/llama.cpp, MIT, Copyright (c) 2023-2026 The ggml authors. It is not copied into this repository. `CMakeLists.txt` downloads it at commit `b6b003d2cb29647d968302eb2db8da6f66303b3e` when it builds, appends `llama-hooks/` to it, and links it into `impulse-server`. `patches/llama.cpp-b6b003d-impulse.patch` is the same change as a patch. The model graphs in `src/gemma4` and `src/qwen35` follow llama.cpp's graphs for those models, so that the logits are bit for bit the same.

## Model weights

There are none in this repository. Every model is a separate download under its own licence.
