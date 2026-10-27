// --split-mode tensor: the meta backend's split-state callback for the Qwen engine (q35_tsplit.cpp)
#pragma once

#include "q35_engine.hpp"

namespace q35 {

struct TsplitInfo {
    const HParams * hp = nullptr;
    ggml_context * meta = nullptr;   // the GGUF's tensor metadata (the reference tensors' types)
    std::vector<float> ts;           // the --tensor-split shares, one per card
};

ggml_backend_meta_split_state tsplit_state(const ggml_tensor * tensor, void * userdata);

} // namespace q35
