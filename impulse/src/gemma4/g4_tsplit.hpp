// --split-mode tensor for the Gemma engine: the meta backend's split-state callback (g4_tsplit.cpp)
#pragma once

#include "g4_engine.hpp"

namespace g4 {

struct TsplitInfo {
    const HParams * hp = nullptr;
    ggml_context * meta = nullptr;   // the GGUF's tensor metadata (the reference tensors' types)
    std::vector<float> ts;           // the --tensor-split shares, one per card
    // the MTP drafter (tensors named mtp.*): the target layer each drafter layer's attention reads (its Q heads and output are
    // split as that layer's cache is, the rest mirrored), and the drafter's head count
    std::vector<int> mtp_tl;
    int mtp_n_head = 0;
};

ggml_backend_meta_split_state tsplit_state(const ggml_tensor * tensor, void * userdata);

} // namespace g4
