// --split-mode tensor for the Qwen engine: which slice of each tensor goes on which card. llama.cpp's
// llama_meta_device_get_split_state (src/llama-model.cpp, b6b003d), the cases a qwen35 / qwen35moe model meets, rewritten
// over the engine's hparams; the tensor names are llama.cpp's (blk.N.*, cache_k_lN, cache_r_lN, ...), so every tensor
// gets the slices llama.cpp gives it and the meta backend splits the same graph the same way.
#include "q35_tsplit.hpp"

#include <cstring>
#include <numeric>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace q35 {

ggml_backend_meta_split_state tsplit_state(const ggml_tensor * tensor, void * userdata) {
    const TsplitInfo * ud = (const TsplitInfo *) userdata;
    const HParams & H = *ud->hp;
    const size_t n_dev = ud->ts.size();
    const std::string tensor_name = tensor->name;
    auto is_recr = [&](size_t il) { return il < H.is_recr.size() && H.is_recr[il]; };

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");
    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");
    static const std::regex pattern_ffn_up_weight   ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_gate_weight ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_up_shexp_weight  ("blk\\.\\d*\\.ffn_up_shexp.weight");
    static const std::regex pattern_ffn_gate_shexp_weight("blk\\.\\d*\\.ffn_gate_shexp.weight");
    static const std::regex pattern_ffn_down_shexp_weight("blk\\.\\d*\\.ffn_down_shexp.weight");
    static const std::regex pattern_output_weight   ("output\\.weight");
    auto m = [&](const std::regex & r) { return std::regex_match(tensor_name, r); };

    struct tensor_config {
        ggml_backend_meta_split_axis axis;
        const ggml_tensor * tensor_axis_0;
        uint32_t il;
        size_t rotation;   // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };
    auto get_tensor_config_impl = [&](const ggml_backend_meta_split_axis axis, const std::string & suffix = "",
                                      const std::string & suffix_fallback = "") -> tensor_config {
        // count only the same type of previous layers (the layers in a tensor can be inhomogeneous)
        auto get_il_eff = [&](const size_t il) {
            size_t ret = 0;
            for (size_t il_prev = 0; il_prev < il; il_prev++) ret += is_recr(il_prev) == is_recr(il);
            return ret;
        };
        uint32_t il;
        std::string prefix;
        size_t rotation;
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
            rotation = (size_t) H.n_layer % n_dev;   // hparams.n_layer(): the MTP block not counted
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ggml_get_tensor(ud->meta, (prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ggml_get_tensor(ud->meta, (prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return { axis, tensor_axis_0, il, rotation };
    };
    auto get_tensor_config = [&]() -> tensor_config {
        // standard attention
        if (m(pattern_q_weight) || m(pattern_kv_weight))  return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        if (m(pattern_qkv_weight))                         return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        if (m(pattern_qk_norm))
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        if (m(pattern_kv_cache))                           return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        if (m(pattern_attn_out_weight))                    return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        if (m(pattern_attn_gate_weight))                   return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        if (m(pattern_ssm_dt) || m(pattern_ssm_a))         return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        if (m(pattern_ssm_alpha) || m(pattern_ssm_beta))   return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        if (m(pattern_r_cache) || m(pattern_s_cache)) {
            // the LFM2 shortconv block runs fully mirrored, so its conv state is mirrored too
            if (H.lfm2) return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED, "");
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (m(pattern_ssm_conv1d))                         return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        if (m(pattern_ssm_out_weight))                     return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        // FFN
        if (m(pattern_ffn_up_weight) || m(pattern_ffn_gate_weight))
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        if (m(pattern_ffn_gate_up_weight)) return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        if (m(pattern_ffn_down_weight))    return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        // output
        if (m(pattern_output_weight)) return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    // Qwen 3.5: [k0_v0, k1_v1, k0_v2, k1_v3] (V segmented on the scale of K to get the broadcasting pattern right); LFM2: the
    // generic rules (no fused projections)
    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        (void) il;
        if (H.lfm2) return { { tensor->ne[axis], 1 } };
        const int64_t head_k_dim = H.ssm_d_state, head_v_dim = H.ssm_d_state;
        const int64_t n_k_heads = H.ssm_n_group, n_v_heads = H.ssm_dt_rank;
        const int64_t key_dim = head_k_dim * n_k_heads, value_dim = head_v_dim * n_v_heads;
        const int64_t head_ratio = n_k_heads ? n_v_heads / n_k_heads : 1;
        if (m(pattern_qkv_weight) || m(pattern_ssm_conv1d)) {
            GGML_ASSERT(tensor->ne[axis] == 2 * key_dim + value_dim);
            return { { key_dim, (uint32_t) (2 + head_ratio) } };
        }
        if (m(pattern_attn_gate_weight) || m(pattern_ssm_out_weight)) return { { key_dim, (uint32_t) head_ratio } };
        if (m(pattern_ssm_dt) || m(pattern_ssm_a) || m(pattern_ssm_alpha) || m(pattern_ssm_beta)) return { { n_k_heads, (uint32_t) head_ratio } };
        if (m(pattern_r_cache)) return { { key_dim * (H.ssm_d_conv - 1), (uint32_t) (2 + head_ratio) } };
        if (m(pattern_s_cache)) return { { n_k_heads * head_v_dim * head_v_dim, (uint32_t) head_ratio } };
        if (m(pattern_ffn_gate_up_weight)) {
            GGML_ASSERT(tensor->ne[axis] == 2 * H.n_ff_exp);
            return { { H.n_ff_exp, 2 } };
        }
        return { { tensor->ne[axis], 1 } };
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        if (is_recr(il)) {   // linear attention
            const int64_t head_dim = H.ssm_d_state;
            const int64_t blck_size_perf = std::lcm(blck_size, (int64_t) 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (m(pattern_qkv_weight) || m(pattern_attn_gate_weight) || m(pattern_ssm_conv1d) || m(pattern_ssm_out_weight))
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            if (m(pattern_ssm_dt) || m(pattern_ssm_a) || m(pattern_ssm_alpha) || m(pattern_ssm_beta))
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            if (m(pattern_r_cache)) return std::vector<int64_t>(segments.size(), granularity_qkv * (H.ssm_d_conv - 1));
            if (m(pattern_s_cache)) return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
        } else {             // regular attention
            const int64_t n_gqa = H.n_head / H.n_head_kv;
            const int64_t n_embd_q = n_gqa * H.head_dim;
            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf * (int64_t) n_dev < n_embd_q) blck_size_perf *= 2;
            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            const int64_t granularity_head = granularity_q / H.head_dim;   // for tensors with one value per head
            if (m(pattern_q_weight)) {
                GGML_ASSERT(segments.size() == 1);
                if (H.lfm2) return { granularity_q };
                return { std::lcm(2 * n_embd_q, blck_size_perf) };   // Qwen 3.5: Q and its gate interleaved
            }
            if (m(pattern_attn_out_weight)) { GGML_ASSERT(segments.size() == 1); return { granularity_q }; }
            if (m(pattern_attn_gate_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return { tensor->ne[1] == H.n_head ? granularity_head : granularity_q };
            }
            const int64_t granularity_kv = granularity_q / n_gqa;
            if (m(pattern_kv_weight) || m(pattern_kv_cache)) { GGML_ASSERT(segments.size() == 1); return { granularity_kv }; }
            if (m(pattern_qkv_weight)) { GGML_ASSERT(segments.size() == 2); return { granularity_q, granularity_kv }; }
        }
        // FFN
        if (m(pattern_ffn_up_weight) || m(pattern_ffn_gate_weight) || m(pattern_ffn_gate_up_weight) || m(pattern_ffn_down_weight) ||
            m(pattern_ffn_up_shexp_weight) || m(pattern_ffn_gate_shexp_weight) || m(pattern_ffn_down_shexp_weight)) {
            GGML_ASSERT(segments.size() == 1);
            return { std::lcm(blck_size, (int64_t) 128) };
        }
        // everything else
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

} // namespace q35
