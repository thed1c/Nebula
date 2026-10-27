// --split-mode tensor for the Gemma engine: which slice of each tensor goes on which card. llama.cpp's
// llama_meta_device_get_split_state (src/llama-model.cpp, b6b003d) has no Gemma case of its own: its generic rules, over the
// engine's hparams (per layer: the window or global head size and KV head count) and llama.cpp's tensor names (blk.N.*,
// cache_k_lN, ...), so every tensor gets the slices llama.cpp gives it.
#include "g4_tsplit.hpp"

#include <cstring>
#include <numeric>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace g4 {

ggml_backend_meta_split_state tsplit_state(const ggml_tensor * tensor, void * userdata) {
    const TsplitInfo * ud = (const TsplitInfo *) userdata;
    const HParams & H = *ud->hp;
    const size_t n_dev = ud->ts.size();
    const std::string tensor_name = tensor->name;
    auto is_swa = [&](size_t il) { return il < H.is_swa.size() && H.is_swa[il]; };
    auto head_k = [&](size_t il) { return (int64_t) (is_swa(il) ? H.head_dim_swa : H.head_dim); };
    auto n_gqa = [&](size_t il) { return (int64_t) H.n_head / std::max(1, il < H.n_head_kv.size() ? H.n_head_kv[il] : 1); };

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_ffn_up_weight   ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_gate_weight ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_output_weight   ("output\\.weight");
    auto m = [&](const std::regex & r) { return std::regex_match(tensor_name, r); };

    // the MTP drafter's tensors: its attention reads a target layer's cache, sliced by KV head over the cards, so its Q rows and
    // its output's input rows are split as that layer's attn_q / attn_output (the same heads on each card); the rest mirrored
    if (tensor_name.rfind("mtp.", 0) == 0) {
        ggml_backend_meta_split_state st;
        memset(&st, 0, sizeof(st));
        st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED; st.nr[0] = 1; st.n_segments = 1;
        static const std::regex pattern_mtp_q  ("mtp\\.blk\\.(\\d+)\\.attn_q\\.weight");
        static const std::regex pattern_mtp_wo ("mtp\\.blk\\.(\\d+)\\.attn_output\\.weight");
        std::smatch sm;
        const bool is_q = std::regex_match(tensor_name, sm, pattern_mtp_q);
        const bool is_o = !is_q && std::regex_match(tensor_name, sm, pattern_mtp_wo);
        if (!is_q && !is_o) return st;
        const size_t dl = (size_t) std::stoul(sm[1].str());
        GGML_ASSERT(dl < ud->mtp_tl.size());
        const int tl = ud->mtp_tl[dl];
        size_t il_eff = 0;
        for (int il_prev = 0; il_prev < tl; il_prev++) il_eff += is_swa(il_prev) == is_swa(tl);
        const size_t rotation = il_eff % n_dev;
        const ggml_tensor * ref = ggml_get_tensor(ud->meta, ("blk." + std::to_string(tl) + ".attn_output.weight").c_str());
        GGML_ASSERT(ref != nullptr);
        const int64_t n_gqa_d = (int64_t) ud->mtp_n_head / std::max(1, H.n_head_kv[tl]);
        const int64_t n_embd_q = n_gqa_d * head_k(tl);
        int64_t blck_size_perf = ggml_blck_size(ref->type);
        while (blck_size_perf < 128 && blck_size_perf * (int64_t) n_dev < n_embd_q) blck_size_perf *= 2;
        const int64_t g_s = std::lcm(n_embd_q, blck_size_perf);
        st.axis = is_q ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
        const int64_t ne_s = tensor->ne[st.axis];
        std::vector<float> scan;
        for (size_t j = 0; j < n_dev; j++) { scan.push_back(ud->ts[(j + rotation) % n_dev]); if (j > 0) scan[j] += scan[j - 1]; }
        int64_t low = 0;
        size_t j = 0;
        for (; j < n_dev - 1; j++) {
            int64_t high = scan.back() == 0.0f ? ne_s * (int64_t) (j + 1) / (int64_t) n_dev : (int64_t) (ne_s * scan[j] / scan.back());
            if (high % g_s != 0) high -= high % g_s;
            st.ne[(j + rotation) % n_dev] = high - low;
            low = high;
        }
        st.ne[(j + rotation) % n_dev] = ne_s - low;
        return st;
    }

    struct tensor_config { ggml_backend_meta_split_axis axis; const ggml_tensor * tensor_axis_0; uint32_t il; size_t rotation; };
    auto get_tensor_config_impl = [&](const ggml_backend_meta_split_axis axis, const std::string & suffix = "",
                                      const std::string & suffix_fallback = "") -> tensor_config {
        auto get_il_eff = [&](const size_t il) {   // count only the same type of previous layers
            size_t ret = 0;
            for (size_t il_prev = 0; il_prev < il; il_prev++) ret += is_swa(il_prev) == is_swa(il);
            return ret;
        };
        uint32_t il; std::string prefix; size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = (uint32_t) std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % n_dev;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = (uint32_t) std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % n_dev;
        } else {
            il = 0;
            rotation = (size_t) H.n_layer % n_dev;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ggml_get_tensor(ud->meta, (prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr && !suffix_fallback.empty()) tensor_axis_0 = ggml_get_tensor(ud->meta, (prefix + suffix_fallback).c_str());
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return { axis, tensor_axis_0, il, rotation };
    };
    auto get_tensor_config = [&]() -> tensor_config {
        if (m(pattern_q_weight) || m(pattern_kv_weight)) return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        if (m(pattern_qk_norm))
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        if (m(pattern_kv_cache))        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        if (m(pattern_attn_out_weight)) return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        if (m(pattern_ffn_up_weight) || m(pattern_ffn_gate_weight) || m(pattern_ffn_gate_up_weight))
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        if (m(pattern_ffn_down_weight)) return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        if (m(pattern_output_weight))   return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);   // everything else
    };
    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        (void) il;
        if (m(pattern_ffn_up_weight)) {   // a fused up + gate tensor named "up" is segmented
            if (tensor->ne[axis] == 2 * (int64_t) H.n_ff) return { { H.n_ff, 2 } };
            return { { tensor->ne[axis], 1 } };
        }
        if (m(pattern_ffn_gate_up_weight)) {
            GGML_ASSERT(tensor->ne[axis] == 2 * (int64_t) H.n_ff_exp);
            return { { H.n_ff_exp, 2 } };
        }
        return { { tensor->ne[axis], 1 } };
    };
    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        {   // regular attention (Gemma has no recurrent layers)
            const int64_t n_embd_q = n_gqa(il) * head_k(il);
            int64_t blck_size_perf = blck_size;   // only increase granularity while it doesn't cause underutilization
            while (blck_size_perf < 128 && blck_size_perf * (int64_t) n_dev < n_embd_q) blck_size_perf *= 2;
            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (m(pattern_q_weight))        { GGML_ASSERT(segments.size() == 1); return { granularity_q }; }
            if (m(pattern_attn_out_weight)) { GGML_ASSERT(segments.size() == 1); return { granularity_q }; }
            const int64_t granularity_kv = granularity_q / n_gqa(il);
            if (m(pattern_kv_weight) || m(pattern_kv_cache)) { GGML_ASSERT(segments.size() == 1); return { granularity_kv }; }
        }
        if (m(pattern_ffn_up_weight) || m(pattern_ffn_gate_weight) || m(pattern_ffn_gate_up_weight) || m(pattern_ffn_down_weight)) {
            GGML_ASSERT(segments.size() == 1);
            return { std::lcm(blck_size, (int64_t) 128) };
        }
        GGML_ASSERT(segments.size() == 1);
        return { 1 };
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    const tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        std::vector<float> tensor_split_scan;
        for (size_t j = 0; j < n_dev; j++) {
            tensor_split_scan.push_back(ud->ts[(j + tc.rotation) % n_dev]);
            if (j > 0) tensor_split_scan[j] += tensor_split_scan[j - 1];
        }
        const auto segments = get_split_segments(split_state.axis, tc.il);
        const auto granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t g_s = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < n_dev - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ? ne_s * (int64_t) (j + 1) / (int64_t) n_dev
                                                                : (int64_t) (ne_s * tensor_split_scan[j] / tensor_split_scan.back());
                if (high % g_s != 0) high -= high % g_s;
                split_state.ne[is * n_dev + (j + tc.rotation) % n_dev] = high - low;
                low = high;
            }
            split_state.ne[is * n_dev + (j + tc.rotation) % n_dev] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = (uint32_t) segments.size();
    } else {
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
    }
    return split_state;
}

} // namespace g4
