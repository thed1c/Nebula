// src/fx/fx_pack.hpp - which engine impulse-server runs for a -m path (One Engine P1).
//
//   a folder holding pack/ (a Strata pack: pack/index.txt) and a *dense*.gguf   -> fx  (Strata's Flash-Next engine)
//   a *dense*.gguf whose folder holds such a pack/                                -> fx
//   any other GGUF                                                                -> its general.architecture's engine:
//       qwen35moe, qwen35, lfm2moe, lfm2, qwen4exp -> q35; everything else -> g4
//
// So a qwen4exp GGUF without a pack beside it (the GGUF Flash-Next path) still goes to the q35 engine exactly as before.
// Header-only (the filesystem and ggml's gguf reader), so tests/fx_dispatch_test.cpp checks the same code main runs.
#pragma once

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace fx {

struct Pack {
    std::string root;    // the model folder
    std::string dense;   // its *dense*.gguf: the tokenizer, the chat template, the dense weights
    std::string pack;    // root/pack: the experts (and pack/tokenizer, Strata's own copy of the tokenizer)
};

inline bool is_pack_dir(const std::filesystem::path & p) {
    std::error_code ec;
    return std::filesystem::is_directory(p, ec) && std::filesystem::is_regular_file(p / "index.txt", ec);
}

inline bool is_dense_gguf_name(const std::string & name) {
    const auto ends = [&](const char * s) { const size_t n = std::strlen(s); return name.size() >= n && name.compare(name.size() - n, n, s) == 0; };
    return ends(".gguf") && name.find("dense") != std::string::npos;
}

// -m's path -> the pack it names (true), or not a pack (false: a GGUF for the other engines, or nothing)
inline bool find_pack(const std::string & model, Pack & out) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (model.empty()) return false;
    const fs::path m(model);
    if (fs::is_directory(m, ec)) {
        if (!is_pack_dir(m / "pack")) return false;
        std::vector<std::string> dense;
        for (const auto & e : fs::directory_iterator(m, ec))
            if (e.is_regular_file(ec) && is_dense_gguf_name(e.path().filename().string())) dense.push_back(e.path().string());
        if (dense.empty()) return false;
        std::sort(dense.begin(), dense.end());   // more than one: the first by name, and the caller says which (--fx-config)
        out = Pack{ m.string(), dense.front(), (m / "pack").string() };
        return true;
    }
    if (fs::is_regular_file(m, ec) && is_dense_gguf_name(m.filename().string())) {
        const fs::path root = m.has_parent_path() ? m.parent_path() : fs::path(".");
        if (!is_pack_dir(root / "pack")) return false;
        out = Pack{ root.string(), m.string(), (root / "pack").string() };
        return true;
    }
    return false;
}

inline std::string gguf_architecture(const std::string & path) {
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return "";
    const int64_t id = gguf_find_key(g, "general.architecture");
    const std::string arch = id < 0 ? "" : gguf_get_val_str(g, id);
    gguf_free(g);
    return arch;
}

inline bool q35_architecture(const std::string & arch) {
    return arch == "qwen35moe" || arch == "qwen35" || arch == "lfm2moe" || arch == "lfm2" || arch == "lfm3" || arch == "qwen4exp" || arch == "flint";
}

// "fx", "q35" or "g4" for a -m path (g4 also for a path that is neither: its own loader reports it)
inline std::string engine_for(const std::string & model) {
    Pack p;
    if (find_pack(model, p)) return "fx";
    return q35_architecture(gguf_architecture(model)) ? "q35" : "g4";
}

// -m / --model's value on a command line ("" when there is none)
inline std::string model_arg(int argc, char ** argv) {
    std::string model;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--fx-args")) break;   // the engine's own arguments follow (src/fx/fx_server.cpp)
        if (!std::strcmp(argv[i], "-m") || !std::strcmp(argv[i], "--model")) model = argv[i + 1];
    }
    return model;
}

}  // namespace fx
