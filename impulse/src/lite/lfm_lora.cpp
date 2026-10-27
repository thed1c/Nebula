#include "lfm_lora.hpp"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lite {
LfmLora::~LfmLora() {
    for (auto & a : adapters_) {
        if (a.buffer) ggml_backend_buffer_free(a.buffer);
        if (a.dev) ggml_free(a.dev);
        if (a.cpu) ggml_free(a.cpu);
        if (a.file) gguf_free(a.file);
    }
}

static bool finite_tensor(const ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        const float * v = (const float *) t->data;
        for (int64_t j = 0; j < n; ++j) if (!std::isfinite(v[j])) return false;
    } else {
        const ggml_fp16_t * v = (const ggml_fp16_t *) t->data;
        for (int64_t j = 0; j < n; ++j) if (!std::isfinite(ggml_fp16_to_fp32(v[j]))) return false;
    }
    return true;
}

std::string LfmLora::load(const std::string & path, float scale, ggml_context * base, ggml_backend_t backend, const std::string & arch) {
    if (!std::isfinite(scale)) return "LoRA scale must be finite";
    const bool dense = arch == "lfm2";
    if (!dense && !adapters_.empty()) return "lfm2moe: one adapter at a time";
    adapters_.push_back(Adapter{});
    Adapter & A = adapters_.back();
    const int index = (int) adapters_.size() - 1;
    A.path = path; A.scale = scale;
    gguf_init_params p = { false, &A.cpu };
    A.file = gguf_init_from_file(path.c_str(), p);     // ggml's gguf reader: 64-bit file offsets (the same on Windows)
    if (!A.file || !A.cpu) return "cannot read LoRA " + path;
    auto str = [&](const char * key) -> std::string {
        int64_t i = gguf_find_key(A.file, key);
        return i >= 0 && gguf_get_kv_type(A.file, i) == GGUF_TYPE_STRING ? gguf_get_val_str(A.file, i) : "";
    };
    if (str("general.architecture") != arch || str("general.type") != "adapter" || str("adapter.type") != "lora")
        return "expected an " + arch + " LoRA adapter: " + path;
    float alpha = 0;
    // llama.cpp's key (LLM_KV_ADAPTER_LORA_ALPHA) is "adapter.lora.alpha"; "adapter.lora_alpha" was read here before 2026-10-21,
    // so an adapter with alpha != rank was applied at the wrong strength (rank-one adapters with alpha = 1 were unaffected)
    int64_t ai = gguf_find_key(A.file, "adapter.lora.alpha");
    if (ai < 0) ai = gguf_find_key(A.file, "adapter.lora_alpha");
    if (ai >= 0) {
        if (gguf_get_kv_type(A.file, ai) != GGUF_TYPE_FLOAT32) return "LoRA alpha must be f32";
        alpha = gguf_get_val_f32(A.file, ai);
    }
    if (!std::isfinite(alpha) || !std::isfinite(scale * alpha)) return "LoRA alpha/scale must be finite";
    const int64_t nt = gguf_get_n_tensors(A.file);
    if (nt <= 0 || nt > 4096 || nt % 2) return "invalid LoRA tensor count";
    ggml_init_params gp = { (size_t)(nt + 4) * ggml_tensor_overhead(), nullptr, true };
    A.dev = ggml_init(gp);
    if (!A.dev) return "LoRA tensor allocation failed";
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
    size_t mine = 0;
    static const char * dense_leaves[] = { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                           "shortconv.in_proj.weight", "shortconv.out_proj.weight",
                                           "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
    for (int64_t i = 0; i < nt; ++i) {
        std::string name = gguf_get_tensor_name(A.file, i);
        if (name.size() < 7 || (name.substr(name.size()-7) != ".lora_a" && name.substr(name.size()-7) != ".lora_b"))
            return "unsupported LoRA tensor " + name;
        std::string target = name.substr(0, name.size()-7);
        auto * a = ggml_get_tensor(A.cpu, (target + ".lora_a").c_str());
        auto * b = ggml_get_tensor(A.cpu, (target + ".lora_b").c_str());
        if (!a || !b) return "incomplete LoRA pair for " + target;
        if (name.back() == 'b') continue;
        auto * w = ggml_get_tensor(base, target.c_str());
        if (!w) return "LoRA target absent from base: " + target;
        int block = -1, prefix = 0;
        if (sscanf(target.c_str(), "blk.%d.%n", &block, &prefix) != 1 || prefix == 0 || block < 0)
            return "unsupported LoRA target " + target;
        std::string leaf = target.substr(prefix);
        const bool experts = leaf == "ffn_down_exps.weight";
        bool ok = false;
        if (dense) { for (const char * d : dense_leaves) ok = ok || leaf == d; }
        else ok = experts || leaf == "attn_output.weight" || leaf == "shortconv.out_proj.weight";
        if (!ok) return std::string(dense ? "LFM2 LoRA targets are a block's linears: " : "only LFM mixer-output and expert-down LoRA targets are supported: ") + target;
        const int64_t rank = b->ne[0];
        if (rank < 1 || a->ne[1] != rank) return "LoRA rank mismatch: " + target;
        if (!dense && (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || rank != 1))
            return "only f32 rank-one LFM adapters are supported: " + target;
        if ((a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) || (b->type != GGML_TYPE_F32 && b->type != GGML_TYPE_F16))
            return "LoRA tensors are f32 or f16: " + target;
        if (a->ne[0] != w->ne[0] || b->ne[1] != w->ne[1] || a->ne[2] != w->ne[2] || b->ne[2] != w->ne[2] ||
            a->ne[3] != 1 || b->ne[3] != 1 || w->ne[3] != 1 || (!experts && w->ne[2] != 1))
            return "LoRA/base shape mismatch: " + target;
        if (!finite_tensor(a) || !finite_tensor(b)) return "non-finite LoRA tensor " + target;
        auto & v = pairs_[target];
        for (const auto & q : v) if (q.adapter == index) return "duplicate LoRA target " + target;
        Pair pair;
        pair.a = ggml_dup_tensor(A.dev, a); pair.b = ggml_dup_tensor(A.dev, b);
        ggml_set_name(pair.a, ggml_get_name(a)); ggml_set_name(pair.b, ggml_get_name(b));
        pair.per_unit = alpha ? alpha / (float) rank : 1.0f;   // llama_adapter_lora_weight::get_scale
        pair.scale = scale * pair.per_unit;
        pair.adapter = index;
        copies.emplace_back(pair.a, a); copies.emplace_back(pair.b, b);
        v.push_back(pair); ++mine; ++n_pairs_;
    }
    if (mine * 2 != (size_t) nt) return "LoRA pair count mismatch";
    A.buffer = ggml_backend_alloc_ctx_tensors_from_buft(A.dev, ggml_backend_get_default_buffer_type(backend));
    if (!A.buffer) return "LoRA allocation failed";
    ggml_backend_buffer_set_usage(A.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & c : copies) ggml_backend_tensor_set(c.first, c.second->data, 0, ggml_nbytes(c.first));
    fprintf(stderr, "impulse: LoRA %s: %zu sites, scale %.6g, %.2f MiB (%s)\n", path.c_str(), mine, scale,
            ggml_backend_buffer_get_size(A.buffer) / 1048576.0, ggml_backend_name(backend));
    return {};
}

void LfmLora::set_scale(int i, float s) {
    adapters_[i].scale = s;
    for (auto & kv : pairs_) for (auto & p : kv.second) if (p.adapter == i) p.scale = s * p.per_unit;
}

const LfmLora::Pair * LfmLora::find(const ggml_tensor * weight) const {
    auto i = pairs_.find(ggml_get_name(weight));
    return i == pairs_.end() || i->second.empty() || i->second[0].scale == 0 ? nullptr : &i->second[0];
}

ggml_tensor * LfmLora::add(ggml_context * ctx, const ggml_tensor * weight, ggml_tensor * input,
                          ggml_tensor * result, ggml_tensor * ids) const {
    auto i = pairs_.find(ggml_get_name(weight));
    if (i == pairs_.end()) return result;
    for (const Pair & p : i->second) {                   // llm_graph_context::build_lora_mm: adapters in load order
        if (p.scale == 0) continue;
        auto * delta = ids ? ggml_mul_mat_id(ctx, p.b, ggml_mul_mat_id(ctx, p.a, input, ids), ids)
                          : ggml_mul_mat(ctx, p.b, ggml_mul_mat(ctx, p.a, input));
        result = ggml_add(ctx, result, ggml_scale(ctx, delta, p.scale));
    }
    return result;
}

}
