// Impulse: what llama-server's /v1/models and /props report about the model file (n_params, size, ftype), read from the
// GGUF's tensor list (the servers load only llama.cpp's vocabulary, whose model reports none of them)
#pragma once

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdint>
#include <string>

namespace lite {

struct ModelInfo { uint64_t n_params = 0, size = 0; std::string ftype; };

inline ModelInfo model_info(const std::string & path) {
    ModelInfo mi;
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return mi;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        if (!t) continue;
        mi.n_params += (uint64_t) ggml_nelements(t);
        mi.size += (uint64_t) ggml_nbytes(t);
    }
    const int64_t k = gguf_find_key(g, "general.file_type");
    mi.ftype = k >= 0 ? llama_ftype_name((llama_ftype) gguf_get_val_u32(g, k)) : "unknown";
    gguf_free(g);
    if (meta) ggml_free(meta);
    return mi;
}

} // namespace lite
