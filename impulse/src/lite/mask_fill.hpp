// Impulse: the attention mask's rows, filled fast.
//
// Both engines build the f16 causal mask on the host for every ubatch: n rows (the tokens) of n_kv cells, 0 where the
// token may see the cell and -inf where not. Written as a per-entry ternary on two compares against a vector element,
// the loop ran at about 1.3 ns an entry single-threaded (the portable AVX2 build, 2026-10-16): a 120K-token read in 4K
// chunks fills some 7.8 billion entries, ten seconds of its 42; a 15K read 170 million, a fifth of a second. Here each
// row is one unsigned range test, (uint32) (pos - lo) > (uint32) (hi - lo), which is false exactly when lo <= pos <= hi
// (a free cell's pos = -1 wraps above every hi), the loop vectorizes (0.2 ns an entry), and the rows of a big ubatch are
// split over threads. Every entry gets the same value as before; only the time changes.
//
//   causal, no window (Qwen/LFM, Gemma's global layers):   lo = 0,                        hi = p1
//   causal sliding window of n_swa (Gemma):                lo = max(0, p1 - n_swa + 1),   hi = p1
//   non-causal window (Gemma's window layers on an image): lo = max(0, p1 - n_swa + 1),   hi = INT32_MAX
#pragma once

#include "ggml.h"

#include <climits>
#include <cstdint>
#include <thread>
#include <vector>

namespace lite {

inline void fill_mask_rows(ggml_fp16_t * md, int i0, int i1, uint32_t n_kv, const int32_t * pos, const int32_t * lo, const int32_t * hi,
                           const int8_t * cseq, const int8_t * rseq) {
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
    for (int i = i0; i < i1; ++i) {
        ggml_fp16_t * row = md + (size_t) i * n_kv;
        const uint32_t l = (uint32_t) lo[i], span = (uint32_t) (hi[i] - lo[i]);
        if (!cseq) {
            for (uint32_t j = 0; j < n_kv; ++j) row[j] = ((uint32_t) pos[j] - l > span) ? drop : keep;
        } else {   // several sequences: another sequence's cell is masked as well
            const int8_t s = rseq[i];
            for (uint32_t j = 0; j < n_kv; ++j) row[j] = (((uint32_t) pos[j] - l > span) | (cseq[j] != s)) ? drop : keep;
        }
    }
}

// md [n][n_kv]; pos = the cells' positions (-1 free); lo / hi = each row's visible range; n_threads for big fills;
// cseq / rseq (several sequences) = each cell's and each row's sequence, nullptr for one sequence
inline void fill_mask(ggml_fp16_t * md, int n, uint32_t n_kv, const int32_t * pos, const int32_t * lo, const int32_t * hi, int n_threads,
                      const int8_t * cseq = nullptr, const int8_t * rseq = nullptr) {
    const size_t work = (size_t) n * n_kv;
    int nt = work >= ((size_t) 1 << 20) ? std::min(n_threads, n) : 1;
    if (nt <= 1) { fill_mask_rows(md, 0, n, n_kv, pos, lo, hi, cseq, rseq); return; }
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        const int i0 = (int) ((int64_t) n * t / nt), i1 = (int) ((int64_t) n * (t + 1) / nt);
        ts.emplace_back(fill_mask_rows, md, i0, i1, n_kv, pos, lo, hi, cseq, rseq);
    }
    for (auto & t : ts) t.join();
}

} // namespace lite
