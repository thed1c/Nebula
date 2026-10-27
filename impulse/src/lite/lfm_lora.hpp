// LFM LoRA adapters in llama.cpp's GGUF adapter format, stacked as llama.cpp's build_lora_mm stacks them: for each adapter in
// load order, res += scale * B(A x), scale = adapter_scale * alpha / rank (or adapter_scale without alpha). Owns validation,
// storage and arithmetic; the base GGUF and the expert-cache copies remain unchanged.
//   lfm2moe (LFM2.5-24B): one rank-one adapter on the mixer outputs and the expert downs (the CPU pool's path needs rank one).
//   lfm2 (dense LFM2 / LFM2.5, Halo's model): any rank, F32 or F16, on every linear of a block (attention q/k/v/output,
//   short-conv in/out, FFN gate/up/down), several adapters at once; their scales change at run time (set_scale; the engine
//   rebuilds its graphs and the server drops prompt caches computed with the old scales). An adapter at scale 0 is skipped,
//   as llama.cpp's server leaves a zero-scale adapter out.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace lite {
class LfmLora {
public:
    struct Pair {
        ggml_tensor * a = nullptr, * b = nullptr;
        float scale = 0;      // adapter_scale * alpha / rank (what the graph multiplies by)
        float per_unit = 1;   // alpha / rank (or 1 without alpha): scale = adapter_scale * per_unit
        int adapter = 0;
    };
    ~LfmLora();
    // one adapter more (load order = stacking order); arch is the base's general.architecture
    std::string load(const std::string & path, float scale, ggml_context * base, ggml_backend_t backend, const std::string & arch = "lfm2moe");
    const Pair * find(const ggml_tensor * weight) const;   // the first adapter's pair (the lfm2moe pool path: one rank-one adapter)
    ggml_tensor * add(ggml_context * ctx, const ggml_tensor * weight, ggml_tensor * input,
                      ggml_tensor * result, ggml_tensor * expert_ids = nullptr) const;
    size_t size() const { return n_pairs_; }
    size_t count() const { return adapters_.size(); }
    float scale(int i) const { return adapters_[i].scale; }
    const std::string & path(int i) const { return adapters_[i].path; }
    void set_scale(int i, float s);
private:
    struct Adapter {
        std::string path;
        float scale = 0;
        gguf_context * file = nullptr;
        ggml_context * cpu = nullptr, * dev = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
    };
    std::vector<Adapter> adapters_;
    std::unordered_map<std::string, std::vector<Pair>> pairs_;   // target weight name -> one pair per adapter that has it
    size_t n_pairs_ = 0;
};
}
