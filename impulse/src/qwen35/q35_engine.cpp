// Impulse hybrid MoE engine. See q35_engine.hpp. Every graph-building step below names the llama.cpp
// function it mirrors (the experiment's copy, b6b003d: src/models/qwen35moe.cpp, src/models/delta-net-base.cpp,
// src/models/lfm2.cpp, src/llama-graph.cpp, src/llama-kv-cache.cpp); the order of the ggml calls is kept, because the CUDA backend fuses
// adjacent nodes and the scheduler splits by node order. Node names follow llama.cpp's callbacks ("<name>-<layer>").
#include "q35_engine.hpp"
#include "q35_tsplit.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "../lite/platform.hpp"
#include "../lite/ram_tier.hpp"
#include "../lite/pinned_host.hpp"
#include "../lite/mask_fill.hpp"

namespace q35 {

static constexpr int   kMaxNodes   = 32768;   // qwen4exp: ~40k in llama.cpp's budget (max(40 T, 32 n_tensors))
// rope parameters as llama.cpp's cparams resolve them for this model (no rope scaling: ext_factor 0)
static constexpr int   kRopeType   = GGML_ROPE_TYPE_IMROPE;
static constexpr float kExtFactor  = 0.0f;
static constexpr float kAttnFactor = 1.0f;
static constexpr float kBetaFast   = 32.0f;
static constexpr float kBetaSlow   = 1.0f;
// ggml-cuda offloads an op on CPU-resident weights to the GPU from this many tokens: below it llama.cpp computes the
// experts on the CPU, and so does the pool
static constexpr int   kOffloadMin = 32;
// graph reuse (v0.2.3) is on unless IMPULSE_GRAPH_REUSE=0 (llama.cpp: LLAMA_GRAPH_REUSE_DISABLE) or a graph dump is
// asked for (the dumps hook a build)
static bool graph_reuse_on() {
    static const bool on = !(getenv("IMPULSE_GRAPH_REUSE") && atoi(getenv("IMPULSE_GRAPH_REUSE")) == 0)
                        && !getenv("Q35_GRAPHDUMP") && !getenv("TDUMP") && !getenv("GDUMP");
    return on;
}
// the slot path's misses all point at one zero slot, so a token's ids repeat: ggml-cuda's mat-vec MoE kernel (<= 8
// tokens) takes that, its batched MMQ kernel does not; 9-31 token ubatches go through the pool for every expert
static constexpr int   kSlotMaxTokens = 8;
static constexpr int   kMinSlots = 8;          // below this the arena is not worth its VRAM
// Auto chunk: a read's compute buffers ~ fixed + per token + the KQ mask (chunk x context, f16); calibrated on the 5070
// with Qwen's graph (stage g)
static constexpr size_t kChunkFixed  = (size_t) 768 << 20;
static constexpr size_t kChunkPerTok = 200 * 1024;
static constexpr size_t kChunkPerTokQ4x = 360 * 1024;          // qwen4exp: the 4-wide residual, the indexer (fitted 347 KB, 10-17)
static constexpr size_t kQsaScratch = (size_t) 1536 << 20;     // qwen4exp: a ubatch's indexer scratch above this goes in slices
static constexpr int    kChunkMin    = 4096;

namespace scpu = strata::kernels::cpu;

// q4 KV: llama.cpp rotates K (and Q) and V by orthonormal Walsh-Hadamard matrices before quantising (attn_rot_k/v,
// llama-kv-cache.cpp), and V's rotation back after attention. ggml_gen_hadamard:
static std::vector<float> gen_hadamard(int n) {
    std::vector<float> d((size_t) n * n);
    d[0] = 1.0 / sqrtf(n);
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) {
                const float v = d[i * n + j];
                d[(i + s) * n + j] = v; d[i * n + (j + s)] = v; d[(i + s) * n + (j + s)] = -v;
            }
    return d;
}
// llama-impl.h llama_mul_mat_hadamard
static ggml_tensor * mul_mat_hadamard(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];
    ggml_tensor * res = ggml_is_contiguous(cur) ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n)
                                                : ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}
// NVFP4's second-level scales, as llama.cpp b6b003d applies its optional "<weight>.scale" tensors (llama-graph.cpp):
// build_lora_mm: the matmul's output times the weight's {1} scale (before any adapter's term is added)
static ggml_tensor * mm_s(ggml_context * ctx, ggml_tensor * res, ggml_tensor * s) { return s ? ggml_mul(ctx, res, s) : res; }
// build_lora_mm_id: the per-expert scale [n_expert] as [1, n_expert, n_tokens], gathered by the routed experts `sel` (the
// expert ids, also where the matmul read cache slots), times the matmul's output [rows, k, n_tokens]
static ggml_tensor * mm_id_s(ggml_context * ctx, ggml_tensor * res, ggml_tensor * s, ggml_tensor * sel, int64_t n_tokens) {
    if (!s) return res;
    const int64_t n_expert = s->ne[0];
    ggml_tensor * t = ggml_reshape_3d(ctx, s, 1, n_expert, 1);
    t = ggml_repeat_4d(ctx, t, 1, n_expert, n_tokens, 1);
    t = ggml_get_rows(ctx, t, sel);
    return ggml_mul(ctx, res, t);
}
// llama.cpp#30861 + #30928 (Flint): the down projection's input [n_ff, k, n] over an exact power of two per [expert, token]
// from its max |x| (threshold 1024: the F16 sums of the Q8_1 activation blocks stay finite); the result times it after
static std::pair<ggml_tensor *, ggml_tensor *> down_amax(ggml_context * ctx, ggml_tensor * cur) {
    constexpr float k_log2_e = 1.4426950408889634f, k_ln_2 = 0.6931471805599453f;
    ggml_tensor * amax = ggml_pool_1d(ctx, ggml_abs(ctx, cur), GGML_OP_POOL_MAX, cur->ne[0], cur->ne[0], 0);
    ggml_tensor * e = ggml_scale(ctx, amax, 1.0f / 1024.0f);
    e = ggml_log(ctx, e);
    e = ggml_scale(ctx, e, k_log2_e);
    e = ggml_ceil(ctx, e);
    e = ggml_clamp(ctx, e, 0.0f, 127.0f);
    e = ggml_scale(ctx, e, k_ln_2);
    e = ggml_exp(ctx, e);
    return { ggml_div(ctx, cur, e), e };
}
// llama_kv_cache::build_input_k_rot: the largest power of two (from 64) dividing the head size; V always 64
static int k_rot_size(int head) { int r = 64; do { r *= 2; } while (head % r == 0); return r / 2; }
static std::mutex g_pool_mu;   // the ExpertPool is not reentrant

// ---------------------------------------------------------------- stage c: the experts through Strata's pool
struct Engine::PoolLayer {
    scpu::NativeFmt f;
    const uint8_t * gate = nullptr; size_t g_stride = 0;   // expert e's gate rows: gate + e*g_stride; its up rows f.up_off on
    const uint8_t * dn = nullptr;   size_t d_stride = 0;   // its down rows
    scpu::ExpertPool * pool = nullptr;
    const int * res = nullptr;                             // stage d: slot of each expert, -1 = not resident (nullptr: none)
    Stats * stats = nullptr;
    std::vector<uint8_t> act;
    std::vector<scpu::ExpertJobMulti> jobs;
    const lite::LfmLora::Pair * lora_down = nullptr;
    // the experts' second-level scales (ffn_{gate,up,down}_exps.scale, one per expert; nullptr = none): each job takes its
    // expert's, applied inside the rows as llama.cpp's ggml_mul after each matmul (gate/up before the SwiGLU, down after)
    const float * sg = nullptr, * su = nullptr, * sd = nullptr;
    std::vector<float> hidden;
    // overlap: the misses' rows, written by the pool on the driver thread, copied out by the join op
    std::vector<float> out;
    std::mutex mu; std::condition_variable cv; bool done = true;
    Engine::Driver * driver = nullptr;
};

// overlap: one thread that runs a layer's miss jobs through the pool, so the host thread can launch the GPU hits
struct Engine::Driver {
    std::mutex mu; std::condition_variable cv; Engine::PoolLayer * task = nullptr; bool stop = false;
    std::thread th;
    Driver() : th([this] { loop(); }) {}
    ~Driver() { { std::lock_guard<std::mutex> g(mu); stop = true; } cv.notify_all(); th.join(); }
    void submit(Engine::PoolLayer * L) { { std::lock_guard<std::mutex> g(mu); task = L; } cv.notify_all(); }
    void loop() {
        for (;;) {
            Engine::PoolLayer * L;
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop || task; }); if (stop) return; L = task; task = nullptr; }
            const auto t0 = std::chrono::steady_clock::now();
            { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
            if (L->stats) L->stats->pool_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            { std::lock_guard<std::mutex> g(L->mu); L->done = true; }
            L->cv.notify_all();
        }
    }
};

// overlap, start: the routing and the activations are here (the scheduler synced the router for them); quantize, plan
// the misses into L->out, hand them to the driver thread and return at once. dst is a 1-element token for ordering.
static void moe_start_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1];
    const int64_t n = x->ne[1], k = ids->ne[0], ne = x->ne[0];
    const size_t ab = L->f.act_bytes;
    L->act.resize((size_t) n * ab);
    L->out.assign((size_t) n * k * ne, 0.0f);
    if (L->lora_down) L->hidden.assign((size_t)n * k * L->f.n_ff, 0.0f);
    auto id_at = [&](int64_t t, int64_t j) { return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]); };
    L->jobs.clear();
    std::vector<int> job_of(1024, -1);
    bool quantized = false;
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            const int e = id_at(t, j);
            L->stats->accesses++;
            if (L->res[e] >= 0) { L->stats->hits++; continue; }
            if (!quantized) {
                for (int64_t u = 0; u < n; ++u)
                    scpu::native_quant_act(L->f, (const float *) ((const char *) x->data + u * x->nb[1]), L->act.data() + u * ab);
                quantized = true;
            }
            int ji = job_of[e];
            if (ji < 0 || L->jobs[ji].nt == scpu::MAXT) {
                scpu::ExpertJobMulti jb;
                jb.blob = L->gate + (size_t) e * L->g_stride;
                jb.down = L->dn + (size_t) e * L->d_stride;
                jb.s_gate = L->sg ? L->sg[e] : 1.f; jb.s_up = L->su ? L->su[e] : 1.f; jb.s_down = L->sd ? L->sd[e] : 1.f;
                L->jobs.push_back(jb);
                ji = job_of[e] = (int) L->jobs.size() - 1;
            }
            auto & jb = L->jobs[ji];
            jb.nact[jb.nt] = L->act.data() + t * ab;
            jb.out[jb.nt] = L->out.data() + ((size_t) t * k + j) * ne;
            if (L->lora_down) jb.hidden[jb.nt] = L->hidden.data() + ((size_t)t * k + j) * L->f.n_ff;
            jb.nt++;
        }
    }
    L->stats->ov_layers++;
    if (L->jobs.empty()) { L->stats->ov_missfree++; std::lock_guard<std::mutex> g(L->mu); L->done = true; return; }
    { std::lock_guard<std::mutex> g(L->mu); L->done = false; }
    L->driver->submit(L);
}

// overlap, join: wait for the driver (the GPU has been computing the hits meanwhile), then hand over the miss rows
static void moe_join_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const auto t0 = std::chrono::steady_clock::now();
    { std::unique_lock<std::mutex> lk(L->mu); L->cv.wait(lk, [&] { return L->done; }); }
    L->stats->join_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    memcpy(dst->data, L->out.data(), ggml_nbytes(dst));
}

// x [n_embd, n] f32, ids [k, n] i32 (argsort_top_k view) -> dst [n_embd, k, n]: exactly what build_moe_ffn's
// mul_mat_id (up) + mul_mat_id (gate) -> swiglu -> mul_mat_id (down) gives on ggml-cpu for a RAM-resident layer below
// the offload size (same quantizers, dots and SwiGLU). The router weights and the sum over k stay ggml ops after it.
static void moe_pool_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1];
    const int64_t n = x->ne[1], k = ids->ne[0];
    const size_t ab = L->f.act_bytes;
    L->act.resize((size_t) n * ab);
    if (L->lora_down) L->hidden.assign((size_t)n * k * L->f.n_ff, 0.0f);
    for (int64_t t = 0; t < n; ++t)
        scpu::native_quant_act(L->f, (const float *) ((const char *) x->data + t * x->nb[1]), L->act.data() + t * ab);
    auto id_at = [&](int64_t t, int64_t j) { return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]); };
    L->jobs.clear();
    std::vector<int> job_of(1024, -1);
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            const int e = id_at(t, j);
            if (L->res) {   // stage d: a resident expert's row comes from the GPU; this side adds zeros
                L->stats->accesses++;
                if (L->res[e] >= 0) {
                    L->stats->hits++;
                    memset((char *) dst->data + t * dst->nb[2] + j * dst->nb[1], 0, dst->ne[0] * sizeof(float));
                    continue;
                }
            }
            int ji = job_of[e];
            if (ji < 0 || L->jobs[ji].nt == scpu::MAXT) {
                scpu::ExpertJobMulti jb;
                jb.blob = L->gate + (size_t) e * L->g_stride;
                jb.down = L->dn + (size_t) e * L->d_stride;
                jb.s_gate = L->sg ? L->sg[e] : 1.f; jb.s_up = L->su ? L->su[e] : 1.f; jb.s_down = L->sd ? L->sd[e] : 1.f;
                L->jobs.push_back(jb);
                ji = job_of[e] = (int) L->jobs.size() - 1;
            }
            auto & jb = L->jobs[ji];
            jb.nact[jb.nt] = L->act.data() + t * ab;
            jb.out[jb.nt] = (float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1]);
            if (L->lora_down) jb.hidden[jb.nt] = L->hidden.data() + ((size_t)t * k + j) * L->f.n_ff;
            jb.nt++;
        }
    }
    if (!L->jobs.empty()) { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }

}

// Depends on the completed pool/join result, so the optional pre-quantization
// activations are ready. Cached-hit rows remain zero on the miss side.
static void moe_hidden_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    GGML_ASSERT(ggml_nbytes(dst) == L->hidden.size() * sizeof(float));
    memcpy(dst->data, L->hidden.data(), ggml_nbytes(dst));
}

// the slot arena of one cached layer
struct Engine::CacheLayer {
    bool full = false;                  // every expert resident, identity order: exactly llama.cpp's GPU-layer graph
    int  S = 0;                         // resident slots (the tensors have S + 1: slot S is all zeros)
    std::vector<ggml_tensor *> src, slot, all;   // per expert tensor: the RAM source, its slot tensor, the dcheck twin
    ggml_tensor * res_f = nullptr;      // expert -> slot (S for a miss), f32 for the graph's get_rows
    std::vector<int> res;               // expert -> slot, -1 = in RAM only
    std::vector<int> owner;             // slot -> expert
    std::vector<double> score;          // decayed usage counts (stage e)
    std::vector<char> pending;          // expert with a copy in flight (async swaps)
    std::unique_ptr<PoolLayer> miss;    // the pool for the misses (res set)
    ggml_tensor * sc[3] = { nullptr, nullptr, nullptr };   // the experts' scales (gate, up, down) on the layer's card
};

// ---------------------------------------------------------------- GGUF helpers
static int64_t kv_int(gguf_context * g, const std::string & key, int64_t def = -1) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(g, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(g, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(g, id);
        default: return def;
    }
}
static float kv_f32(gguf_context * g, const std::string & key, float def = 0.0f) {
    const int64_t id = gguf_find_key(g, key.c_str());
    return id < 0 ? def : gguf_get_val_f32(g, id);
}
static std::vector<int64_t> kv_arr(gguf_context * g, const std::string & key) {
    std::vector<int64_t> out;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(g, id);
    const void * d = gguf_get_arr_data(g, id);
    switch (gguf_get_arr_type(g, id)) {
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) out.push_back(((const int32_t *) d)[i]); break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) out.push_back(((const uint32_t *) d)[i]); break;
        case GGUF_TYPE_INT64:  for (size_t i = 0; i < n; ++i) out.push_back(((const int64_t *) d)[i]); break;
        case GGUF_TYPE_UINT64: for (size_t i = 0; i < n; ++i) out.push_back((int64_t) ((const uint64_t *) d)[i]); break;
        case GGUF_TYPE_INT16:  for (size_t i = 0; i < n; ++i) out.push_back(((const int16_t *) d)[i]); break;
        case GGUF_TYPE_UINT16: for (size_t i = 0; i < n; ++i) out.push_back(((const uint16_t *) d)[i]); break;
        case GGUF_TYPE_INT8:   for (size_t i = 0; i < n; ++i) out.push_back(((const int8_t *) d)[i]); break;
        case GGUF_TYPE_UINT8:  for (size_t i = 0; i < n; ++i) out.push_back(((const uint8_t *) d)[i]); break;
        case GGUF_TYPE_BOOL:   for (size_t i = 0; i < n; ++i) out.push_back(((const int8_t *) d)[i] != 0); break;
        default: break;
    }
    return out;
}

// ---------------------------------------------------------------- KV cells (llama_kv_cache, unified over the sequences, no SWA)
void Engine::Cache::init(uint32_t cells_per_stream, int n_streams) {
    ssize = cells_per_stream; size = cells_per_stream * (uint32_t) n_streams;
    pos.assign(size, -1); seq.assign(size, (int8_t) -1); tok.assign(size, -1); ex.assign(size, 0); ey.assign(size, 0); heads.assign(n_streams, 0u); n_kv = 0;
}
uint32_t Engine::Cache::used(int st) const {
    return (uint32_t) std::count_if(pos.begin() + (size_t) st * ssize, pos.begin() + (size_t) (st + 1) * ssize, [](int32_t p) { return p >= 0; });
}
uint32_t Engine::Cache::used_max_p1(int st) const {
    for (int64_t i = (int64_t) ssize - 1; i >= 0; --i) if (pos[(size_t) st * ssize + i] >= 0) return (uint32_t) i + 1;
    return 0;
}
// llama_kv_cache::find_slot(ubatch, cont = false), in one stream
bool Engine::Cache::find_slot(uint32_t n_tokens, int st) {
    const size_t base = (size_t) st * ssize, want = idxs.size() + n_tokens;
    uint32_t head_cur = heads[st];
    if (head_cur > used(st) + 2 * n_tokens) head_cur = 0;
    if (n_tokens > ssize) return false;
    uint32_t n_tested = 0;
    while (true) {
        if (head_cur + 1 > ssize) { n_tested += ssize - head_cur; head_cur = 0; continue; }
        const uint32_t idx = head_cur++;
        n_tested++;
        if (pos[base + idx] < 0) idxs.push_back((int64_t) (base + idx));
        if (idxs.size() == want) break;
        if (n_tested >= ssize) return false;
    }
    return true;
}
// llama_kv_cache::apply_ubatch: the cells take their tokens; each stream's search goes on after its last cell
// (and the cell's token: an embedding ubatch stores tok_embd, llama.cpp's PLE image stand-in)
void Engine::Cache::apply(const std::vector<int32_t> & positions, const std::vector<int8_t> & seqs, const int32_t * toks, int32_t tok_embd,
                          const int32_t * pos_y, const int32_t * pos_x) {
    for (size_t i = 0; i < idxs.size(); ++i) {
        pos[idxs[i]] = positions[i]; seq[idxs[i]] = seqs[i];
        tok[idxs[i]] = toks ? toks[i] : tok_embd;
        ey[idxs[i]] = pos_y ? pos_y[i] : 0; ex[idxs[i]] = pos_x ? pos_x[i] : 0;
        heads[idxs[i] / ssize] = (uint32_t) (idxs[i] % ssize) + 1;
    }
}
uint32_t Engine::Cache::calc_n_kv(int st0, int nst) const {
    const uint32_t pad = 256;
    uint32_t m = 0;
    for (int st = st0; st < st0 + nst; ++st) { const uint32_t um = used_max_p1(st); m = std::max(m, std::min(ssize, std::max(pad, (um + pad - 1) / pad * pad))); }
    return m;
}

// llama_kv_cache::seq_rm(seq, p0, -1): free the cells at p0 and later; the stream's search restarts at the first freed one
void Engine::Cache::seq_rm(int s, int32_t p0) {
    for (int st = 0; st < (int) heads.size(); ++st) {
        uint32_t new_head = ssize;
        for (uint32_t i = 0; i < ssize; ++i) {
            const size_t c = (size_t) st * ssize + i;
            if (pos[c] >= p0 && (s < 0 || seq[c] == s)) { pos[c] = -1; seq[c] = -1; tok[c] = -1; if (new_head == ssize) new_head = i; }
        }
        if (new_head != ssize && new_head < heads[st]) heads[st] = new_head;
    }
}

// lfm3's sliding cache: sequence seq's cells before `below` (their positions, the K/V rows of every sliding layer, in cell
// order of its stream) and back
Engine::SwaSnap Engine::swa_save(int seq, int32_t below) const {
    SwaSnap S; S.floor = swa_floor_v_[seq];
    const int st = kvs_.stream_of(seq);
    const size_t base = (size_t) st * kvs_.ssize;
    S.pos.assign(kvs_.ssize, -1);
    for (uint32_t i = 0; i < kvs_.ssize; ++i) if (kvs_.pos[base + i] >= 0 && kvs_.pos[base + i] < below && kvs_.seq[base + i] == seq) S.pos[i] = kvs_.pos[base + i];
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!swa_layer(il)) continue;
        for (ggml_tensor * t : { kvs_.k[il], kvs_.v[il] }) {
            const size_t off = S.rows.size(), nb = (size_t) kvs_.ssize * t->nb[1];
            S.rows.resize(off + nb);
            ggml_backend_tensor_get(t, S.rows.data() + off, base * t->nb[1], nb);
        }
    }
    return S;
}
void Engine::swa_load(int seq, const SwaSnap & S) {
    const int st = kvs_.stream_of(seq);
    const size_t base = (size_t) st * kvs_.ssize;
    kvs_.seq_rm(seq, 0);
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!swa_layer(il)) continue;
        for (ggml_tensor * t : { kvs_.k[il], kvs_.v[il] }) {
            const size_t nb = (size_t) kvs_.ssize * t->nb[1];
            ggml_backend_tensor_set(t, S.rows.data() + off, base * t->nb[1], nb);
            off += nb;
        }
    }
    uint32_t last = 0;
    for (uint32_t i = 0; i < kvs_.ssize; ++i) if (S.pos[i] >= 0) { kvs_.pos[base + i] = S.pos[i]; kvs_.seq[base + i] = (int8_t) seq; last = i + 1; }
    kvs_.heads[st] = last;
    swa_floor_v_[seq] = S.floor;
}

// ---------------------------------------------------------------- load
Engine::Engine() = default;

Engine::~Engine() {
    if (lazy_pin_) { cudaFreeHost(lazy_pin_); lazy_pin_ = nullptr; }
    if (mtp_.sched) ggml_backend_sched_free(mtp_.sched);
    if (mtp_.own_gpu && mtp_.gpu) ggml_backend_free(mtp_.gpu);
    if (mtp_.buf_kv) ggml_backend_buffer_free(mtp_.buf_kv);
    if (mtp_.buf_gpu) ggml_backend_buffer_free(mtp_.buf_gpu);
    if (mtp_.buf_map) ggml_backend_buffer_free(mtp_.buf_map);
    if (mtp_.ctx_kv) ggml_free(mtp_.ctx_kv);
    if (mtp_.ctx_gpu) ggml_free(mtp_.ctx_gpu);
    if (mtp_.ctx_w) ggml_free(mtp_.ctx_w);
    if (mtp_.meta) ggml_free(mtp_.meta);
    if (mtp_.gguf) gguf_free(mtp_.gguf);
    lite::unmap_file(mtp_.map, mtp_.map_size);
    for (void * cs : copy_streams_) { cudaStreamSynchronize((cudaStream_t) cs); cudaStreamDestroy((cudaStream_t) cs); }
    for (auto & b : pending_) for (void * e : b.evs) cudaEventDestroy((cudaEvent_t) e);
    for (void * c : ckpt_) if (c) cudaFree(c);
    driver_.reset();
    lora_.reset();
    graph_drop(g_); graph_drop(mg_);
    if (sched_) ggml_backend_sched_free(sched_);
    for (auto * b : buf_cache_c_) if (b) ggml_backend_buffer_free(b);
    for (auto * c : ctx_cache_c_) if (c) ggml_free(c);
    for (auto * b : buf_sc_c_) if (b) ggml_backend_buffer_free(b);
    for (auto * c : ctx_sc_c_) if (c) ggml_free(c);
    for (auto * b : buf_kv_x_) if (b) ggml_backend_buffer_free(b);
    for (KvLayer & kl : kvl_) { if (kl.buf) ggml_backend_buffer_free(kl.buf); if (kl.ctx) ggml_free(kl.ctx); }
    if (ctx_kvfull_) ggml_free(ctx_kvfull_);
    if (buf_kvs_) ggml_backend_buffer_free(buf_kvs_);
    if (ctx_kvs_) ggml_free(ctx_kvs_);
    if (buf_gpu_) ggml_backend_buffer_free(buf_gpu_);
    for (auto * b : buf_gpu_x_) if (b) ggml_backend_buffer_free(b);
    if (buf_map_) ggml_backend_buffer_free(buf_map_);
    if (buf_arena_) ggml_backend_buffer_free(buf_arena_);
    lite::ram_tier_free(arena_, arena_size_, arena_placed_);
    if (ids_host_) cudaFreeHost(ids_host_);
    for (auto * c : ctx_kv_x_) if (c) ggml_free(c);
    for (auto * c : ctx_gpu_x_) if (c) ggml_free(c);
    if (ctx_gpu_) ggml_free(ctx_gpu_);
    if (ctx_w_) ggml_free(ctx_w_);
    if (meta_) ggml_free(meta_);
    if (gguf_) gguf_free(gguf_);
    for (size_t k = 1; k < shards_.size(); ++k) { if (shards_[k].meta) ggml_free(shards_[k].meta); if (shards_[k].g) gguf_free(shards_[k].g); }
    for (size_t d = 1; d < gpus_.size(); ++d) ggml_backend_free(gpus_[d]);
    if (gpu_ && gpu_ != cpu_) ggml_backend_free(gpu_);
    for (auto * b : cpu_st_) if (b) ggml_backend_free(b);
    if (cpu_) ggml_backend_free(cpu_);
    lite::unmap_file(map_, map_size_);
}

std::string add_lora_arg(Options & o, const std::string & value, bool scaled) {
    size_t a = 0;
    while (a <= value.size()) {
        size_t b = scaled ? value.find(',', a) : std::string::npos;
        if (b == std::string::npos) b = value.size();
        const std::string item = value.substr(a, b - a);
        std::string path = item; float scale = 1.0f;
        if (scaled) {
            const size_t at = item.rfind(':');
            if (at == std::string::npos || at == 0) return "--lora-scaled needs FILE:SCALE";
            path = item.substr(0, at);
            try { scale = std::stof(item.substr(at + 1)); } catch (...) { return "--lora-scaled: bad scale in " + item; }
        }
        if (path.empty()) return "--lora: empty file name";
        if (o.lora.empty()) { o.lora = path; o.lora_scale = scale; } else o.loras.push_back({path, scale});
        a = b + 1;
    }
    return {};
}

std::string Engine::load(const Options & o) {
    o_ = o;
    hp_ = HParams{};   // a dense model reloads with the expert options off: start the hparams again
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta_ };
    gguf_ = gguf_init_from_file(o.gguf.c_str(), gp);
    if (!gguf_) return "cannot read " + o.gguf;
    HParams & H = hp_;
    std::string arch;
    {
        const int64_t id = gguf_find_key(gguf_, "general.architecture");
        arch = id < 0 ? "" : gguf_get_val_str(gguf_, id);
        if (arch != "qwen35moe" && arch != "qwen35" && arch != "lfm2moe" && arch != "lfm2" && arch != "lfm3" && arch != "qwen4exp" && arch != "flint") return "unsupported hybrid GGUF (" + arch + ")";
    }
    H.lfm2 = arch == "lfm2moe" || arch == "lfm2" || arch == "lfm3";   // lfm2: the dense LFM2 / LFM2.5 (1.2B, 2.6B: every layer a dense FFN; Halo's model)
    H.lfm3 = arch == "lfm3";                                          // LFM3 (1.7B): the dense LFM2 path plus its own attention
    H.dense = arch == "qwen35" || arch == "lfm2" || arch == "lfm3";
    H.q4x = arch == "qwen4exp";
    H.flint = arch == "flint";
    if (H.flint && o.split_tensor) return "Flint: --split-mode layer only so far";
    if (H.flint && !o.mtp.empty()) return "Flint has no MTP head";
    if (H.q4x && o.split_tensor) return "Qwen3.8 Flash-Next (qwen4exp): --split-mode layer only (several cards: each its layers and their expert cache)";
    if (!H.lfm2 && (!o.lora.empty() || !o.loras.empty())) return "LoRA adapters: LFM2 models only";
    if (H.lfm2 && !H.dense && !o.loras.empty()) return "lfm2moe: one adapter (--lora)";
    const std::string A = arch + ".";
    if (H.lfm2 && !o.mtp.empty()) return "LFM2 has no supported MTP head";
    if (o.n_seq < 1 || o.n_seq > 8) return "--parallel: 1 to 8 sequences";
    H.n_layer       = (int) kv_int(gguf_, A + "block_count");
    H.n_layer_all   = H.n_layer;
    // an MTP head (nextn_predict_layers) is an extra block the main pass never runs (qwen35moe.cpp: n_layer() excludes it)
    H.n_layer      -= (int) kv_int(gguf_, A + "nextn_predict_layers", 0);
    H.n_embd        = (int) kv_int(gguf_, A + "embedding_length");
    H.n_head        = (int) kv_int(gguf_, A + "attention.head_count");
    H.n_head_kv     = H.lfm2 ? 0 : (int) kv_int(gguf_, A + "attention.head_count_kv");
    H.head_dim      = (int) kv_int(gguf_, A + "attention.key_length", H.n_head > 0 ? H.n_embd / H.n_head : 0);
    H.n_rot         = (int) kv_int(gguf_, A + "rope.dimension_count", H.head_dim);
    H.n_ctx_train   = (int) kv_int(gguf_, A + "context_length");
    H.n_expert      = (int) kv_int(gguf_, A + "expert_count");
    H.n_expert_used = (int) kv_int(gguf_, A + "expert_used_count");
    H.n_ff_exp      = (int) kv_int(gguf_, A + "expert_feed_forward_length");
    H.n_ff_shexp    = (int) kv_int(gguf_, A + "expert_shared_feed_forward_length");
    H.ssm_d_conv    = (int) kv_int(gguf_, A + "ssm.conv_kernel");
    H.ssm_d_inner   = (int) kv_int(gguf_, A + "ssm.inner_size");
    H.ssm_d_state   = (int) kv_int(gguf_, A + "ssm.state_size");
    H.ssm_n_group   = (int) kv_int(gguf_, A + "ssm.group_count");
    H.ssm_dt_rank   = (int) kv_int(gguf_, A + "ssm.time_step_rank");
    H.full_attn_interval = (int) kv_int(gguf_, A + "full_attention_interval", 4);
    H.rope_base     = kv_f32(gguf_, A + "rope.freq_base");
    H.eps           = kv_f32(gguf_, A + "attention.layer_norm_rms_epsilon");
    {
        const auto s = kv_arr(gguf_, A + "rope.dimension_sections");
        for (size_t i = 0; i < 4 && i < s.size(); ++i) H.rope_sections[i] = (int) s[i];
    }
    {
        const int64_t id = gguf_find_key(gguf_, "tokenizer.ggml.tokens");
        H.n_vocab = id < 0 ? 0 : (int) gguf_get_arr_n(gguf_, id);
    }
    // qwen35moe load_arch_hparams: recurrent unless every full_attention_interval-th layer
    H.is_recr.resize(H.n_layer);
    for (int il = 0; il < H.n_layer; ++il) H.is_recr[il] = (il + 1) % H.full_attn_interval != 0;
    if (H.q4x) {   // llama_model_qwen4exp::load_arch_hparams
        if (kv_int(gguf_, A + "nextn_predict_layers", 0) != 0) return "qwen4exp: an MTP block is not supported";
        H.hc = (int) kv_int(gguf_, A + "hyper_connection.count");
        H.hc_lr = (int) kv_int(gguf_, A + "hyper_connection.low_rank");
        H.idx_n_head = (int) kv_int(gguf_, A + "attention.indexer.head_count");
        H.idx_dim = (int) kv_int(gguf_, A + "attention.indexer.key_length");
        H.idx_top_k = (int) kv_int(gguf_, A + "attention.indexer.top_k");
        if (H.hc <= 1 || H.hc_lr <= 0 || H.idx_n_head <= 0 || H.idx_dim <= 0 || H.idx_top_k <= 0) return "qwen4exp: bad hyper-connection / indexer hparams";
        const auto cr = kv_arr(gguf_, A + "attention.compress_ratios");
        H.compress.assign(H.n_layer, 0);
        if (!cr.empty()) { for (int il = 0; il < H.n_layer && il < (int) cr.size(); ++il) H.compress[il] = (int) cr[il]; }
        else { const int64_t c = kv_int(gguf_, A + "attention.compress_ratios", 0); for (int & x : H.compress) x = (int) c; }
        for (int il = 0; il < H.n_layer; ++il) if (H.compress[il] < 0 || H.compress[il] > 64) return "qwen4exp: bad compress ratio";
        const auto rl = kv_arr(gguf_, A + "attention.recurrent_layers");
        if (!rl.empty()) for (int il = 0; il < H.n_layer && il < (int) rl.size(); ++il) H.is_recr[il] = rl[il] != 0;
        const auto pl = kv_arr(gguf_, A + "ple.layers");
        if (pl.size() > 1) return "qwen4exp: one PLE layer only";
        if (pl.size() == 1) {
            H.ple_layer = (int) pl[0];
            if (H.ple_layer < 0 || H.ple_layer >= H.n_layer || !H.is_recr[H.ple_layer]) return "qwen4exp: the PLE layer must be a linear attention layer";
            H.ple_ngram = (int) kv_int(gguf_, A + "ple.ngram_size");
            H.ple_per_gram = (int) kv_int(gguf_, A + "ple.heads_per_ngram");
            H.ple_kernel = (int) kv_int(gguf_, A + "ple.conv_kernel");
            H.ple_eos = (int32_t) kv_int(gguf_, A + "ple.eos_token_id");
            H.ple_img = (int32_t) kv_int(gguf_, A + "ple.image_token_id", 0);
            H.ple_head_dim = (int) kv_int(gguf_, A + "embedding_length_per_layer_input");
            H.ple_n_heads = (H.ple_ngram - 1) * H.ple_per_gram;
            if (H.ple_ngram < 2 || H.ple_ngram > 8 || H.ple_n_heads <= 0 || H.ple_n_heads > 32 || H.ple_kernel <= 0 || H.ple_head_dim <= 0)
                return "qwen4exp: bad PLE hparams";
            const auto m = kv_arr(gguf_, A + "ple.layer_multipliers"), off = kv_arr(gguf_, A + "ple.head_offsets"), vs = kv_arr(gguf_, A + "ple.head_vocab_sizes");
            if ((int) m.size() < H.ple_ngram || (int) off.size() < H.ple_n_heads || (int) vs.size() < H.ple_n_heads) return "qwen4exp: short PLE arrays";
            for (int i = 0; i < H.ple_ngram; ++i) H.ple_mult.push_back((uint64_t) m[i]);
            for (int h = 0; h < H.ple_n_heads; ++h) {
                if (vs[h] <= 0 || off[h] < 0 || off[h] > INT32_MAX || vs[h] > INT32_MAX || off[h] + vs[h] > INT32_MAX) return "qwen4exp: PLE head range";
                H.ple_off.push_back((uint32_t) off[h]); H.ple_vsz.push_back((uint32_t) vs[h]);
            }
        }
        if (o_.kv_initial > 0) fprintf(stderr, "impulse: qwen4exp: the whole context's K/V and indexer keys at load (no KV pager yet)\n");
        o_.kv_initial = 0;
    }
    if (H.flint) {   // llama_model_flint::load_arch_hparams
        H.n_dense = (int) kv_int(gguf_, A + "leading_dense_block_count");
        H.n_ff = (int) kv_int(gguf_, A + "feed_forward_length");
        H.n_swa = (int) kv_int(gguf_, A + "attention.sliding_window");
        const auto swa = kv_arr(gguf_, A + "attention.sliding_window_pattern"), rope = kv_arr(gguf_, A + "attention.rope_pattern");
        if ((int) swa.size() != H.n_layer || (int) rope.size() != H.n_layer) return "Flint needs a sliding-window and a RoPE pattern per layer";
        H.is_swa.assign(H.n_layer, false); H.has_rope.assign(H.n_layer, false);
        for (int il = 0; il < H.n_layer; ++il) { H.is_swa[il] = swa[il] != 0; H.has_rope[il] = rope[il] != 0; H.is_recr[il] = false; }
        auto kv_bool = [&](const std::string & k) { const int64_t id = gguf_find_key(gguf_, k.c_str()); return id >= 0 && gguf_get_kv_type(gguf_, id) == GGUF_TYPE_BOOL && gguf_get_val_bool(gguf_, id); };
        if (kv_int(gguf_, A + "expert_gating_func") != 2) return "Flint requires sigmoid expert gating";
        if (!kv_bool(A + "expert_weights_norm")) return "Flint: the graph here renormalizes the routing weights (expert_weights_norm)";
        if (kv_f32(gguf_, A + "expert_weights_scale", 1.0f) != 1.0f) return "Flint: expert_weights_scale other than 1 is not supported";
        if (H.n_dense < 0 || H.n_dense >= H.n_layer || H.n_ff <= 0 || H.n_swa <= 0 || H.n_ff_shexp <= 0) return "unexpected Flint hparams";
        if (H.n_rot != H.head_dim) return "Flint: RoPE over the whole head only";
    }
    if (H.lfm2) {
        const auto heads = kv_arr(gguf_, A + "attention.head_count_kv");
        if ((int) heads.size() != H.n_layer) return "LFM2 needs one KV head count per layer";
        for (int il = 0; il < H.n_layer; ++il) {
            H.is_recr[il] = heads[il] == 0;
            if (heads[il] != 0) {
                if (H.n_head_kv && H.n_head_kv != heads[il]) return "LFM2 varying nonzero KV head counts are unsupported";
                H.n_head_kv = (int) heads[il];
            }
        }
        H.n_dense = H.dense ? H.n_layer : (int) kv_int(gguf_, A + "leading_dense_block_count");   // llama.cpp lfm2.cpp: all dense
        H.n_ff = (int) kv_int(gguf_, A + "feed_forward_length");
        H.ssm_d_conv = (int) kv_int(gguf_, A + "shortconv.l_cache");
        if (!H.dense && kv_int(gguf_, A + "expert_gating_func") != 2) return "LFM2 requires sigmoid expert gating";
        if (H.n_dense < 0 || (H.n_dense >= H.n_layer && !H.dense) || H.n_ff <= 0 || H.ssm_d_conv <= 1)
            return "unexpected LFM2 dense/shortconv hparams";
        if (o.n_ctx > H.n_ctx_train) return "LFM2 context exceeds the model's context_length";
    }
    if (H.lfm3) {   // llama_model_lfm3::load_arch_hparams (b11988): the sliding layers, their window and RoPE base
        H.n_swa = (int) kv_int(gguf_, A + "attention.sliding_window");
        const auto swa = kv_arr(gguf_, A + "attention.sliding_window_pattern");
        if ((int) swa.size() != H.n_layer) return "LFM3 needs a sliding-window pattern per layer";
        H.is_swa.assign(H.n_layer, false);
        bool any = false;
        for (int il = 0; il < H.n_layer; ++il) { H.is_swa[il] = swa[il] != 0; any = any || H.is_swa[il]; if (H.is_swa[il] && H.is_recr[il]) return "LFM3: a sliding layer without attention"; }
        if (!any || H.n_swa <= 0) return "LFM3 requires sliding window attention layers";
        H.rope_base_swa = kv_f32(gguf_, A + "rope.freq_base_swa");
        if (H.rope_base_swa <= 0) return "LFM3: missing rope.freq_base_swa";
        if (H.n_rot != H.head_dim || (int) kv_int(gguf_, A + "attention.value_length", H.head_dim) != H.head_dim) return "LFM3: unexpected head / RoPE sizes";
    }
    // v0.2.4: a dense Qwen3.5 / 3.8 (qwen35: every layer llama.cpp's build_ffn, SILU, PAR). Nothing to cache, pool or swap:
    // the expert paths are turned off and the model loads as llama.cpp's all-GPU path does (as the dense Gemma 4)
    if (H.dense) {
        H.n_expert = H.n_expert_used = H.n_ff_exp = H.n_ff_shexp = 0;
        H.n_dense = H.n_layer;
        H.n_ff = (int) kv_int(gguf_, A + "feed_forward_length");
        if (H.n_ff <= 0) return "unexpected dense hparams (feed_forward_length)";
        if (o.pool || o.cache_slots != 0 || o.adapt || o.async_swaps || o.overlap || o.lend || o.n_cpu_moe > 0 || o.prefill_ubatch < 0) {
            Options d = o;
            d.pool = false; d.cache_slots = 0; d.adapt = false; d.async_swaps = false; d.overlap = false; d.lend = false; d.n_cpu_moe = 0;
            d.prefill_ubatch = std::max(0, o.prefill_ubatch);   // auto chunks only pay for experts streamed through the card
            fprintf(stderr, "impulse: a dense model (no experts): the expert cache, CPU pool, swaps and lending are off\n");
            gguf_free(gguf_); gguf_ = nullptr;
            ggml_free(meta_); meta_ = nullptr;
            return load(d);
        }
    }
    // an expert model over several cards (llama.cpp's -sm layer -ts with --n-cpu-moe): each card holds its layers' experts, the
    // first --n-cpu-moe layers' experts stay in RAM; ggml's own expert path (the expert cache, CPU pool, swaps and lending are one
    // card's), and reads in -ub steps
    if (!H.dense && !H.q4x && o.tensor_split.size() > 1 && !o.split_tensor &&
        (o.pool || o.cache_slots != 0 || o.adapt || o.async_swaps || o.overlap || o.lend || o.prefill_ubatch < 0)) {
        Options d = o;
        d.pool = false; d.cache_slots = 0; d.adapt = false; d.async_swaps = false; d.overlap = false; d.lend = false;
        d.prefill_ubatch = std::max(0, o.prefill_ubatch);
        fprintf(stderr, "impulse: an expert model over several GPUs: ggml's expert path (the expert cache, CPU pool, swaps and lending are off)\n");
        gguf_free(gguf_); gguf_ = nullptr;
        ggml_free(meta_); meta_ = nullptr;
        return load(d);
    }
    if (H.n_layer <= 0 || H.n_embd <= 0 || H.head_dim <= 0 || (!H.lfm2 && !H.flint && H.ssm_d_state <= 0) || H.n_head_kv <= 0 ||
        (!H.dense && (H.n_expert <= 0 || H.n_expert_used <= 0 || H.n_expert_used > 64 || H.n_expert_used > H.n_expert)))
        return "unexpected hybrid model hparams";

    // how the host waits for the card: the driver's default, unless IMPULSE_GPU_WAIT=spin asks for cudaDeviceScheduleSpin
    // (one busy core per wait; Strata's own engine does that, src/core/remote_experts.cpp). Tried as a cure for the slow
    // sampler after each token on a dual-socket Xeon (a sleeping core clocks down and enters C6; a CPU-only microbenchmark of the
    // sampler ran 3.4x slower after a 3 ms idle): in the server it changed nothing (sample 1,490 us a token either way) and cost
    // 3 % of decode (216.7 -> 210.7 tok/s, 2026-10-16), so it is not the default. The cause of that 1.5 ms is still open.
    cpu_only_ = o.cpu_only;
    if (cpu_only_) {
        if (!H.dense) return "--device cpu: dense models only (an expert model needs a card for its cache)";
        if (o.split_tensor || o.tensor_split.size() > 1 || !o.mtp.empty()) return "--device cpu: one device and no drafter";
        o_.kv_initial = 0;   // the KV pager grows the cache by device copies: the whole context at load instead
    }
    if (!cpu_only_ && getenv("IMPULSE_GPU_WAIT") && std::string(getenv("IMPULSE_GPU_WAIT")) == "spin") {
        const cudaError_t fe = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        if (fe != cudaSuccess) { (void) cudaGetLastError(); fprintf(stderr, "impulse: cudaSetDeviceFlags(spin): %s (the driver's default wait stays)\n", cudaGetErrorString(fe)); }
        else fprintf(stderr, "impulse: the host spins while the card computes (IMPULSE_GPU_WAIT=spin)\n");
    }
    // backends: [CUDA0, CPU], the CPU's compute buffers in the GPU's pinned host memory (llama_context)
    ggml_backend_dev_t gpu_dev = cpu_only_ ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev && !cpu_only_) return "no GPU device (--device cpu runs a dense model on the CPU)";
    // several GPUs (llama_model::load_tensors' split points over block_count, the MTP block included): the visible GPUs in
    // order, one share each; layer il on the first card whose cumulative share exceeds il / (n_layer_all + 1), the output on
    // the card of 'layer' n_layer_all; a card with share 0 gets nothing (llama.cpp's projector-only card)
    dev_layer_.assign(H.n_layer, 0); dev_out_ = 0;
    if (o.split_tensor) {   // llama_prepare_model_devices: one meta device over the GPUs, the split states from q35_tsplit
        std::vector<ggml_backend_dev_t> devs;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) devs.push_back(ggml_backend_dev_get(i));
        if (!H.dense) return "--split-mode tensor: dense models only so far (v0.2.4)";
        if (o.tensor_split.size() > devs.size()) return "--tensor-split names " + std::to_string(o.tensor_split.size()) + " GPUs; " + std::to_string(devs.size()) + " visible";
        std::vector<float> ts;   // a card with share 0 takes no part (llama.cpp's projector-only card)
        if (!o.tensor_split.empty()) {
            std::vector<ggml_backend_dev_t> keep;
            for (size_t i = 0; i < o.tensor_split.size(); ++i) if (o.tensor_split[i] > 0) { keep.push_back(devs[i]); ts.push_back(o.tensor_split[i]); }
            devs = keep;
        }
        tp_info_ = std::make_unique<TsplitInfo>();
        tp_info_->hp = &hp_; tp_info_->meta = meta_;
        tp_info_->ts = ts.empty() ? std::vector<float>(devs.size(), 0.0f) : ts;
        ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs.data(), devs.size(), tsplit_state, tp_info_.get());
        gpu_ = ggml_backend_dev_init(meta_dev, nullptr);
        if (!gpu_) return "backend init failed (the meta device)";
        gpus_ = { gpu_ };
        tp_ = true;
        // the KV pager moves K/V by device copies, which a meta tensor (no device pointer of its own) cannot take: the whole
        // context's K/V stays on the cards from load on, and a new conversation does not shrink it
        if (o_.kv_initial > 0) fprintf(stderr, "impulse: tensor split: the whole context's K/V at load (no KV pager)\n");
        o_.kv_initial = 0;
        std::string m;
        for (size_t d = 0; d < devs.size(); ++d) m += (d ? ", " : "") + std::string(ggml_backend_dev_name(devs[d]));
        fprintf(stderr, "impulse: tensor split over %zu GPUs (%s): every layer on every card\n", devs.size(), m.c_str());
    } else if (o.tensor_split.size() > 1) {
        std::vector<ggml_backend_dev_t> devs;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) devs.push_back(ggml_backend_dev_get(i));
        if (o.tensor_split.size() > devs.size()) return "--tensor-split names " + std::to_string(o.tensor_split.size()) + " GPUs; " + std::to_string(devs.size()) + " visible";
        if (!o.lora.empty()) return "several GPUs: the LFM adapter is one card's so far";
        if (o.tensor_split[0] <= 0) return "--tensor-split: the first card needs a share (it holds the engine's own buffers)";
        std::vector<float> splits(o.tensor_split);
        float sum = 0; for (float & x : splits) { sum += x; x = sum; }
        for (float & x : splits) x /= sum;
        auto dev_of = [&](int il) { return (int) (std::upper_bound(splits.begin(), splits.end(), float(il) / (H.n_layer_all + 1)) - splits.begin()); };
        std::vector<int> back_of(devs.size(), -1);   // device index -> backend index (gpus_)
        for (size_t i = 0; i < o.tensor_split.size(); ++i) {
            if (o.tensor_split[i] <= 0) continue;
            back_of[i] = (int) gpus_.size();
            gpus_.push_back(ggml_backend_dev_init(devs[i], nullptr));
            if (!gpus_.back()) return "backend init failed (GPU " + std::to_string(i) + ")";
        }
        for (int il = 0; il < H.n_layer; ++il) dev_layer_[il] = back_of[dev_of(il)];
        dev_out_ = back_of[dev_of(H.n_layer_all)];
        gpu_ = gpus_[0];
        std::string m;
        for (size_t d = 0; d < gpus_.size(); ++d) {
            int n = 0; for (int x : dev_layer_) n += x == (int) d;
            m += (d ? ", " : "") + std::string(ggml_backend_dev_name(ggml_backend_get_device(gpus_[d]))) + " " + std::to_string(n) + " layers";
        }
        fprintf(stderr, "impulse: layers split over %zu GPUs: %s; the output on GPU %d\n", gpus_.size(), m.c_str(), dev_out_);
    } else if (!cpu_only_) {
        gpu_ = ggml_backend_dev_init(gpu_dev, nullptr);
        gpus_ = { gpu_ };
    }
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_only_) { gpu_ = cpu_; gpus_ = { cpu_ }; }   // the engine's "card" is the CPU itself (one backend, freed once)
    if (!gpu_ || !cpu_) return "backend init failed";
    ggml_backend_cpu_set_n_threads(cpu_, o.n_threads);

    // the file, mapped: the CPU-resident weights are used in place (llama.cpp with mmap). A split GGUF (split.count > 1,
    // o.gguf = the first shard, llama.cpp's PREFIX-%05d-of-%05d.gguf names): every shard mapped back to back
    std::vector<std::string> paths = { o.gguf };
    shards_.assign(1, Shard{ gguf_, meta_, 0 });
    if (const int64_t n_split = kv_int(gguf_, "split.count", 1); n_split > 1) {
        const std::string tail = "-00001-of-" + std::string(5 - std::min<size_t>(5, std::to_string(n_split).size()), '0') + std::to_string(n_split) + ".gguf";
        if (kv_int(gguf_, "split.no", 0) != 0 || o.gguf.size() <= tail.size() || o.gguf.compare(o.gguf.size() - tail.size(), tail.size(), tail) != 0)
            return "a split GGUF: give the first shard (..." + tail + ")";
        const std::string prefix = o.gguf.substr(0, o.gguf.size() - tail.size());
        for (int64_t k = 1; k < n_split; ++k) {
            char sfx[64]; snprintf(sfx, sizeof sfx, "-%05lld-of-%05lld.gguf", (long long) k + 1, (long long) n_split);
            paths.push_back(prefix + sfx);
            Shard sh;
            gguf_init_params sp = { true, &sh.meta };
            sh.g = gguf_init_from_file(paths.back().c_str(), sp);
            if (!sh.g) return "cannot read the shard " + paths.back();
            shards_.push_back(sh);
        }
    }
    std::vector<size_t> base_off;
    map_ = lite::map_files(paths, base_off, &map_size_);
    if (!map_) return paths.size() > 1 ? "cannot map the shards of " + o.gguf + " (on Windows: merge them with llama-gguf-split --merge)" : "cannot map " + o.gguf;
    for (size_t k = 0; k < shards_.size(); ++k) shards_[k].data = base_off[k] + gguf_get_data_offset(shards_[k].g);

    // `-ngl 99 --n-cpu-moe N`: token_embd (the input layer) and the first N layers' routed experts in RAM, the rest on
    // the GPU. The output head is its own tensor here (not tied).
    int n_tensors = 0;
    for (const Shard & sh : shards_) n_tensors += (int) gguf_get_n_tensors(sh.g);
    ggml_init_params ip = { (size_t) (n_tensors + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_w_ = ggml_init(ip);
    ctx_gpu_ = ggml_init(ip);
    ctx_gpu_x_.assign(gpus_.size() - 1, nullptr);
    for (auto & c : ctx_gpu_x_) c = ggml_init(ip);
    int mk_dev = 0;   // the card make() puts a GPU tensor on (the layer's, set in the loop below)
    std::vector<std::pair<ggml_tensor *, size_t>> on_cpu, on_gpu;
    // --device cpu: llama.cpp's weight placement on the CPU (llama_model_loader select_weight_buft): a quantized matmul weight
    // goes to the CPU's extra buffer type (CPU_REPACK: interleaved for the AVX2 / AVX-512 kernels) when that type takes the
    // mul_mat, everything else stays in the mapped file
    ggml_backend_buffer_type_t repack_buft = nullptr;
    ggml_backend_dev_t cpu_dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_only_ && cpu_dev_ && !getenv("IMPULSE_NO_REPACK")) {
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev_), "ggml_backend_dev_get_extra_bufts");
        if (get_extra) { ggml_backend_buffer_type_t * ex = get_extra(cpu_dev_); if (ex && *ex) repack_buft = *ex; }
    }
    auto repack_ok = [&](const ggml_tensor * m) {
        if (!repack_buft || !ggml_is_quantized(m->type) || ggml_n_dims(m) != 2) return false;
        ggml_init_params tp = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context * tc = ggml_init(tp);
        ggml_tensor * w = ggml_dup_tensor(tc, m);
        ggml_tensor * b = ggml_new_tensor_4d(tc, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
        ggml_tensor * op = ggml_mul_mat(tc, w, b);
        w->buffer = ggml_backend_buft_alloc_buffer(repack_buft, 0);
        const bool ok = ggml_backend_dev_supports_op(cpu_dev_, op);
        ggml_backend_buffer_free(w->buffer); w->buffer = nullptr;
        ggml_free(tc);
        return ok;
    };
    auto make = [&](const std::string & name, bool cpu) -> ggml_tensor * {
        ggml_tensor * m = nullptr;
        const Shard * sh = nullptr;
        for (const Shard & x : shards_) if ((m = ggml_get_tensor(x.meta, name.c_str()))) { sh = &x; break; }
        if (!m) return nullptr;
        if (cpu_only_) cpu = cpu || !repack_ok(m);   // "gpu" tensors on the CPU = the repacked ones
        ggml_tensor * t = ggml_dup_tensor(cpu ? ctx_w_ : mk_dev == 0 ? ctx_gpu_ : ctx_gpu_x_[mk_dev - 1], m);
        ggml_set_name(t, name.c_str());
        const size_t off = sh->data + gguf_get_tensor_offset(sh->g, gguf_find_tensor(sh->g, name.c_str()));
        (cpu ? on_cpu : on_gpu).push_back({ t, off });
        return t;
    };
    // a weight's optional second-level scale (llama.cpp's generic pass, llama-model.cpp: "<weight>.scale", TENSOR_NOT_REQUIRED,
    // NVFP4's weight_scale_2): looked for only beside a weight that is there, F32 of n values. On the CPU (cpu: the
    // --n-cpu-moe layers' expert scales, as llama.cpp's override regex places them) it is used in place from the mapping and
    // kept out of the RAM tier (not an expert tensor); host: also read into a host array (the CPU pool's jobs)
    std::vector<std::pair<ggml_tensor *, size_t>> on_map;
    std::string scale_err;
    auto make_s = [&](const ggml_tensor * w, const std::string & name, int64_t n, bool cpu, std::vector<float> * host = nullptr) -> ggml_tensor * {
        if (!w) return nullptr;
        ggml_tensor * m = nullptr;
        const Shard * sh = nullptr;
        for (const Shard & x : shards_) if ((m = ggml_get_tensor(x.meta, name.c_str()))) { sh = &x; break; }
        if (!m) return nullptr;
        if (m->type != GGML_TYPE_F32 || ggml_n_dims(m) != 1 || m->ne[0] != n) {
            if (scale_err.empty()) scale_err = name + ": expected F32 [" + std::to_string(n) + "]";
            return nullptr;
        }
        const size_t off = sh->data + gguf_get_tensor_offset(sh->g, gguf_find_tensor(sh->g, name.c_str()));
        if (host) host->assign((const float *) ((const char *) map_ + off), (const float *) ((const char *) map_ + off) + n);
        if (!cpu && !cpu_only_) return make(name, false);
        ggml_tensor * t = ggml_dup_tensor(ctx_w_, m);
        ggml_set_name(t, name.c_str());
        on_map.push_back({ t, off });
        return t;
    };
    int n_scales = 0;
    tok_embd_    = make("token_embd.weight", true);
    mk_dev = dev_out_;
    output_      = make("output.weight", false);
    const bool output_tied = !output_;
    if (!output_) output_ = make("token_embd.weight", false);   // tied (llama.cpp: TENSOR_DUPLICATED)
    if (output_ && output_->type == GGML_TYPE_NVFP4) {   // llama.cpp: output.scale for an NVFP4 output only; a tied one may not have it
        output_s_ = make_s(output_, "output.scale", 1, false);
        if (output_s_ && output_tied) return "output.scale beside a tied NVFP4 output (llama.cpp refuses it)";
        n_scales += output_s_ != nullptr;
    }
    if (H.q4x) {   // no output_norm: the final hc mixer is the output norm
        hc_head_norm_ = make("output_hc_norm.weight", false);
        hc_head_down_ = make("output_hc_down.weight", false);
        hc_head_up_   = make("output_hc_up.weight", false);
        if (!tok_embd_ || !output_ || !hc_head_norm_ || !hc_head_down_ || !hc_head_up_) return "missing token_embd / output / output_hc_*";
        if (H.ple_layer >= 0) {   // the n-gram table stays in RAM (llama.cpp: TENSOR_READ_LAZY; gathered on the CPU)
            ple_tbl_ = make("per_layer_token_embd.weight", true);
            if (!ple_tbl_) return "missing per_layer_token_embd";
            int64_t rows = 0;
            for (int h = 0; h < H.ple_n_heads; ++h) rows = std::max<int64_t>(rows, (int64_t) H.ple_off[h] + H.ple_vsz[h]);
            if (ple_tbl_->ne[0] != H.ple_head_dim || ple_tbl_->ne[1] < rows) return "per_layer_token_embd: too few rows for the PLE heads";
        }
    } else {
    output_norm_ = make(H.lfm2 && !H.lfm3 ? "token_embd_norm.weight" : "output_norm.weight", false);
    if (!tok_embd_ || !output_ || !output_norm_) return "missing token_embd / output / output_norm";
    if (H.lfm3) {   // one rope_freqs for every global layer (llama.cpp: TENSOR_DUPLICATED after the first)
        rope_freqs_ = make("rope_freqs.weight", false);
        if (!rope_freqs_ || rope_freqs_->type != GGML_TYPE_F32 || rope_freqs_->ne[0] != H.n_rot / 2) return "LFM3: missing rope_freqs";
    }
    }
    L_.assign(H.n_layer, Layer{});
    for (int il = 0; il < H.n_layer; ++il) {
        mk_dev = dev_layer_[il];
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        Layer & l = L_[il];
        const bool exps_cpu = il < o.n_cpu_moe;
        if (H.q4x) {
            l.hc_an = make(b("hc_attn_norm.weight"), false); l.hc_ad = make(b("hc_attn_down.weight"), false);
            l.hc_au = make(b("hc_attn_up.weight"), false);   l.hc_ai = make(b("hc_attn_inject.weight"), false);
            l.hc_fn = make(b("hc_ffn_norm.weight"), false);  l.hc_fd = make(b("hc_ffn_down.weight"), false);
            l.hc_fu = make(b("hc_ffn_up.weight"), false);    l.hc_fi = make(b("hc_ffn_inject.weight"), false);
            if (!l.hc_an || !l.hc_ad || !l.hc_au || !l.hc_ai || !l.hc_fn || !l.hc_fd || !l.hc_fu || !l.hc_fi)
                return "layer " + std::to_string(il) + ": missing hyper-connection tensors";
            if (!H.is_recr[il] && H.compress[il] > 0) {
                l.idx_q = make(b("indexer.q_proj.weight"), false); l.idx_k = make(b("indexer.k_proj.weight"), false);
                l.idx_qn = make(b("indexer.q_norm.weight"), false); l.idx_kn = make(b("indexer.k_norm.weight"), false);
                if (!l.idx_q || !l.idx_k || !l.idx_qn || !l.idx_kn) return "layer " + std::to_string(il) + ": missing indexer tensors";
            }
            if (il == H.ple_layer) {
                l.ple_key = make(b("ple_key.weight"), false); l.ple_value = make(b("ple_value.weight"), false);
                l.ple_nk = make(b("ple_norm_key.weight"), false); l.ple_nq = make(b("ple_norm_query.weight"), false);
                l.ple_nc = make(b("ple_norm_conv.weight"), false); l.ple_conv = make(b("ple_conv1d.weight"), false);
                if (!l.ple_key || !l.ple_value || !l.ple_nk || !l.ple_nq || !l.ple_nc || !l.ple_conv) return "missing PLE tensors";
            }
        } else {
        l.attn_norm      = make(b("attn_norm.weight"), false);
        l.attn_post_norm = make(b(H.lfm2 || H.flint ? "ffn_norm.weight" : "post_attention_norm.weight"), false);
        if (H.lfm3) {   // the sandwich norms (use_sandwich_norm): both or neither
            l.mix_post_norm = make(b("post_attention_norm.weight"), false);
            l.ffn_post_norm = make(b("post_ffw_norm.weight"), false);
            if (!l.mix_post_norm != !l.ffn_post_norm) return "LFM3 requires both post norms or neither";
        }
        }
        if (!H.is_recr[il]) {
            l.wq     = make(b("attn_q.weight"), false);
            l.wk     = make(b("attn_k.weight"), false);
            l.wv     = make(b("attn_v.weight"), false);
            l.wo     = make(b("attn_output.weight"), false);
            l.q_norm = make(b("attn_q_norm.weight"), false);
            l.k_norm = make(b("attn_k_norm.weight"), false);
            if (!l.wq || !l.wk || !l.wv || !l.wo || !l.q_norm || !l.k_norm) return "layer " + std::to_string(il) + ": missing attention tensors";
            if (H.lfm3) {
                l.head_gate = make(b("attn_gate.weight"), false);
                if (!l.head_gate || l.head_gate->ne[0] != H.n_embd || l.head_gate->ne[1] != H.n_head) return "layer " + std::to_string(il) + ": missing the LFM3 head gate";
            }
            if (H.flint) {
                l.attn_sinks = make(b("attn_sinks.weight"), false);
                if (!l.attn_sinks || l.attn_sinks->type != GGML_TYPE_F32 || l.attn_sinks->ne[0] != H.n_head) return "layer " + std::to_string(il) + ": missing attention sinks";
            }
        } else if (H.lfm2) {
            l.conv_in = make(b("shortconv.in_proj.weight"), false);
            l.conv_out = make(b("shortconv.out_proj.weight"), false);
            l.conv_kernel = make(b("shortconv.conv.weight"), false);
            if (!l.conv_in || !l.conv_out || !l.conv_kernel) return "missing LFM2 shortconv tensors";
        } else {
            l.wqkv       = make(b("attn_qkv.weight"), false);
            l.wqkv_gate  = make(b("attn_gate.weight"), false);
            l.ssm_conv1d = make(b("ssm_conv1d.weight"), false);
            l.ssm_dt     = make(b("ssm_dt.bias"), false);
            l.ssm_a      = make(b("ssm_a"), false);
            l.ssm_beta   = make(b("ssm_beta.weight"), false);
            l.ssm_alpha  = make(b("ssm_alpha.weight"), false);
            l.ssm_norm   = make(b("ssm_norm.weight"), false);
            l.ssm_out    = make(b("ssm_out.weight"), false);
            if (!l.wqkv || !l.wqkv_gate || !l.ssm_conv1d || !l.ssm_dt || !l.ssm_a || !l.ssm_beta || !l.ssm_alpha || !l.ssm_norm || !l.ssm_out)
                return "layer " + std::to_string(il) + ": missing delta-net tensors";
        }
        if (!H.q4x && (!l.attn_norm || !l.attn_post_norm)) return "missing block norms";
        // the second-level scales the mixers' graphs apply (qwen35 / qwen35moe / qwen4exp / lfm2: build_qkv's Q, K, V, the
        // attention output; the delta-net's qkv, gate, beta, alpha and output projections)
        l.wq_s = make_s(l.wq, b("attn_q.scale"), 1, false);
        l.wk_s = make_s(l.wk, b("attn_k.scale"), 1, false);
        l.wv_s = make_s(l.wv, b("attn_v.scale"), 1, false);
        l.wo_s = make_s(l.wo, b("attn_output.scale"), 1, false);
        l.wqkv_s = make_s(l.wqkv, b("attn_qkv.scale"), 1, false);
        l.wqkv_gate_s = make_s(l.wqkv_gate, b("attn_gate.scale"), 1, false);
        l.ssm_beta_s = make_s(l.ssm_beta, b("ssm_beta.scale"), 1, false);
        l.ssm_alpha_s = make_s(l.ssm_alpha, b("ssm_alpha.scale"), 1, false);
        l.ssm_out_s = make_s(l.ssm_out, b("ssm_out.scale"), 1, false);
        for (ggml_tensor * t : { l.wq_s, l.wk_s, l.wv_s, l.wo_s, l.wqkv_s, l.wqkv_gate_s, l.ssm_beta_s, l.ssm_alpha_s, l.ssm_out_s }) n_scales += t != nullptr;
        // llama.cpp's lfm2.cpp builds its FFNs and experts with no scales (it loads them and leaves them unused): so does this
        auto lfm_ffn_scales = [&](std::initializer_list<const char *> ws) {
            for (const char * w : ws) {
                const std::string nm = b(w);
                for (const Shard & x : shards_) if (ggml_get_tensor(x.meta, nm.c_str())) {
                    fprintf(stderr, "impulse: %s is not applied (llama.cpp b6b003d's LFM2 graph does not use FFN or expert scales)\n", nm.c_str());
                    break;
                }
            }
        };
        if (il < H.n_dense) {
            l.dense_gate = make(b("ffn_gate.weight"), false);
            l.dense_up = make(b("ffn_up.weight"), false);
            l.dense_down = make(b("ffn_down.weight"), false);
            if (!l.dense_gate || !l.dense_up || !l.dense_down) return "missing dense FFN tensors";
            if (H.lfm2) lfm_ffn_scales({ "ffn_gate.scale", "ffn_up.scale", "ffn_down.scale" });
            else {   // build_ffn's up_s / gate_s / down_s
                l.dense_gate_s = make_s(l.dense_gate, b("ffn_gate.scale"), 1, false);
                l.dense_up_s = make_s(l.dense_up, b("ffn_up.scale"), 1, false);
                l.dense_down_s = make_s(l.dense_down, b("ffn_down.scale"), 1, false);
                for (ggml_tensor * t : { l.dense_gate_s, l.dense_up_s, l.dense_down_s }) n_scales += t != nullptr;
            }
            continue;
        }
        if (H.lfm2) {
            l.expert_bias = make(b("exp_probs_b.bias"), false);
            if (!l.expert_bias) return "missing LFM2 expert selection bias";
        }
        if (H.flint) l.expert_bias = make(b("exp_probs_b.bias"), false);   // optional (TENSOR_NOT_REQUIRED); the release has none
        l.gate_inp       = make(b("ffn_gate_inp.weight"), false);
        l.gate_exps      = make(b("ffn_gate_exps.weight"), exps_cpu);
        l.up_exps        = make(b("ffn_up_exps.weight"), exps_cpu);
        l.down_exps      = make(b("ffn_down_exps.weight"), exps_cpu);
        l.gate_inp_shexp = make(b("ffn_gate_inp_shexp.weight"), false);
        l.gate_shexp     = make(b("ffn_gate_shexp.weight"), false);
        l.up_shexp       = make(b("ffn_up_shexp.weight"), false);
        l.down_shexp     = make(b("ffn_down_shexp.weight"), false);
        if ((!H.q4x && (!l.attn_norm || !l.attn_post_norm)) || !l.gate_inp || !l.gate_exps || !l.up_exps || !l.down_exps || (!H.lfm2 && (!l.up_shexp || !l.gate_shexp || !l.down_shexp || (!H.flint && !l.gate_inp_shexp))))
            return "layer " + std::to_string(il) + ": missing tensors";
        if (H.lfm2) lfm_ffn_scales({ "ffn_gate_exps.scale", "ffn_up_exps.scale", "ffn_down_exps.scale" });
        else {   // build_moe_ffn's per-expert scales, beside the experts (RAM for a --n-cpu-moe layer), and the shared expert's
            if (exps_sh_.empty()) exps_sh_.resize(H.n_layer);
            l.gate_exps_s = make_s(l.gate_exps, b("ffn_gate_exps.scale"), H.n_expert, exps_cpu, &exps_sh_[il][0]);
            l.up_exps_s = make_s(l.up_exps, b("ffn_up_exps.scale"), H.n_expert, exps_cpu, &exps_sh_[il][1]);
            l.down_exps_s = make_s(l.down_exps, b("ffn_down_exps.scale"), H.n_expert, exps_cpu, &exps_sh_[il][2]);
            l.gate_shexp_s = make_s(l.gate_shexp, b("ffn_gate_shexp.scale"), 1, false);
            l.up_shexp_s = make_s(l.up_shexp, b("ffn_up_shexp.scale"), 1, false);
            l.down_shexp_s = make_s(l.down_shexp, b("ffn_down_shexp.scale"), 1, false);
            for (ggml_tensor * t : { l.gate_exps_s, l.up_exps_s, l.down_exps_s, l.gate_shexp_s, l.up_shexp_s, l.down_shexp_s }) n_scales += t != nullptr;
        }
    }
    if (!scale_err.empty()) return scale_err;
    if (n_scales) fprintf(stderr, "impulse: %d second-level weight scales (\"*.scale\", NVFP4) applied as llama.cpp does\n", n_scales);
    if (!cpu_only_ || (repack_buft && !on_gpu.empty())) {
    buf_gpu_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpu_, cpu_only_ ? repack_buft : ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_gpu_) return "GPU weight allocation failed";
    ggml_backend_buffer_set_usage(buf_gpu_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // the scheduler places ops by their weights
    if (cpu_only_) fprintf(stderr, "impulse: CPU only: %zu matmul weights repacked (%s, %.0f MiB), the rest used in place\n", on_gpu.size(),
                           ggml_backend_buft_name(repack_buft), ggml_backend_buffer_get_size(buf_gpu_) / 1048576.0);
    }
    for (size_t d = 1; d < gpus_.size(); ++d) {
        ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpu_x_[d - 1], ggml_backend_get_default_buffer_type(gpus_[d]));
        if (!bx) return "GPU weight allocation failed (GPU " + std::to_string(d) + ")";
        ggml_backend_buffer_set_usage(bx, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buf_gpu_x_.push_back(bx);
    }
    for (auto & [t, off] : on_gpu) ggml_backend_tensor_set(t, (const char *) map_ + off, 0, ggml_nbytes(t));
    buf_map_ = ggml_backend_cpu_buffer_from_ptr(map_, map_size_);
    ggml_backend_buffer_set_usage(buf_map_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // with async swaps the RAM-resident experts live in a page-locked arena (Strata's expert arena): the copy stream
    // needs pinned sources, and a read-only file mapping cannot be registered. Same bytes, so the arithmetic is unchanged
    // (the pool and ggml's prompt path read the arena too)
    // Low-RAM mode: when the OS will not lock that much (too little free RAM), the experts stay in the file mapping and the
    // OS reads them in as needed; swaps are then copied in the step (slower, but it runs). --no-async-swaps chooses it up front.
    // The RAM tier (src/lite/ram_tier.hpp): the experts locked up to --ram-budget, the rest stay in the mapping.
    lite::RamTier tier;
    if (o_.async_swaps) {
        // numa-always: under the pool's per-socket split, each socket's half of every expert's rows is placed on that socket
        // (lite/numa_place.hpp). The pool is made here, before the arena, so the placement follows the slices it really
        // runs. IMPULSE_NUMA_PLACE=0: the arena as before. One socket: never reached.
        if (o.pool && lite::numa_place_wanted()) {
            pool_ = std::make_unique<scpu::ExpertPool>(o.pool_threads, /*pin*/ true);
            const scpu::NumaSplitPlan & plan = scpu::active_numa_split();
            if (plan.on && !plan.test) tier = lite::ram_tier_place(on_cpu, o_.ram_budget_gib, 4096, map_, paths, plan);
            else fprintf(stderr, "impulse: NUMA placement not used: the pool's split is off (%s)\n", plan.why.c_str());
        }
        if (!tier.arena) tier = lite::ram_tier_lock(on_cpu, o_.ram_budget_gib, 4096);
        arena_placed_ = tier.placed;
        arena_ = tier.arena; arena_size_ = tier.arena_size; ram_locked_ = tier.arena_size; ram_experts_ = tier.want;
        if (!arena_) o_.async_swaps = false;
    } else if (o.pool && lite::numa_place_wanted()) {
        fprintf(stderr, "impulse: NUMA placement not used: the experts stay in the model file's mapping (no async swaps), "
                        "where the page cache puts them\n");
    }
    auto in_arena = [&](ggml_tensor * t) { return o_.async_swaps && tier.in_arena(t); };
    if (o_.async_swaps) {
        buf_arena_ = ggml_backend_cpu_buffer_from_ptr(arena_, arena_size_);
        ggml_backend_buffer_set_usage(buf_arena_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    size_t aoff = 0;
    for (auto & [t, off] : on_cpu) {
        if (in_arena(t)) {
            char * dst = (char *) arena_ + aoff;
            if (!tier.placed) memcpy(dst, (const char *) map_ + off, ggml_nbytes(t));   // placed: copied per node already
            if (ggml_backend_tensor_alloc(buf_arena_, t, dst) != GGML_STATUS_SUCCESS) return "arena alloc failed";
            aoff += GGML_PAD(ggml_nbytes(t), 4096);
        } else if (ggml_backend_tensor_alloc(buf_map_, t, (char *) map_ + off) != GGML_STATUS_SUCCESS) return "map alloc failed";
    }
    for (auto & [t, off] : on_map)
        if (ggml_backend_tensor_alloc(buf_map_, t, (char *) map_ + off) != GGML_STATUS_SUCCESS) return "map alloc failed (a scale)";

    if (!o.lora.empty() || !o.loras.empty()) {
        lora_ = std::make_unique<lite::LfmLora>();
        std::vector<std::pair<std::string, float>> all;
        if (!o.lora.empty()) all.push_back({o.lora, o.lora_scale});
        all.insert(all.end(), o.loras.begin(), o.loras.end());
        for (const auto & [path, scale] : all) {
            const std::string error = lora_->load(path, scale, meta_, gpu_, H.lfm3 ? "lfm3" : H.dense ? "lfm2" : "lfm2moe");
            if (!error.empty()) return error;
        }
    }

    // the hybrid memory (llama_memory_hybrid): a KV cache for the attention layers (n_ctx cells, K and V [n_embd_gqa,
    // cells], V not transposed: flash attention), and one recurrent cell per delta-net layer (conv state [n_embd_r],
    // ssm state [n_embd_s], f32; with n_rs_seq, 1 + n_rs_seq rows each: row s = the state s tokens back)
    // one stream per sequence of PAD(n_ctx / n_seq, 256) cells (llama_context: n_ctx_seq); one sequence = the whole context
    kv_.init((uint32_t) ((o.n_ctx / o.n_seq + 255) / 256 * 256), o.n_seq);
    if (o.n_seq > 1 && o_.kv_initial > 0) {   // the pager's device copies move one block; streams keep the whole context at load
        fprintf(stderr, "impulse: %d slots: the whole context's K/V at load (no KV pager)\n", o.n_seq);
        o_.kv_initial = 0;
    }
    {   // per sequence (llama_memory_recurrent: one cell per sequence, mem_size = n_seq)
        const int S = o.n_seq;
        n_past_v_.assign(S, 0); rs_src_v_.assign(S, 0); last_ub_v_.assign(S, 0); fresh_v_.assign(S, 1);
        mtp_h_v_.assign(S, std::vector<float>(h_width(), 0.0f)); win_h_v_.assign(S, {}); win_pos0_v_.assign(S, 0);
        ckpt_.assign((size_t) S * kCkpts, nullptr); ckpt_pos_.assign((size_t) S * kCkpts, -1); ckpt_age_.assign((size_t) S * kCkpts, 0);
        ckpt_h_.assign((size_t) S * kCkpts, {}); ckpt_host_.assign((size_t) S * kCkpts, {});
    }
    kv_.k.assign(H.n_layer, nullptr); kv_.v.assign(H.n_layer, nullptr);
    r_l_.assign(H.n_layer, nullptr); s_l_.assign(H.n_layer, nullptr); p_l_.assign(H.n_layer, nullptr); idxk_.assign(H.n_layer, nullptr);
    ggml_init_params kp = { (size_t) (6 * H.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_kv_x_.assign(gpus_.size(), nullptr); buf_kv_x_.assign(gpus_.size(), nullptr);   // each layer's states on its card
    const int64_t n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    for (int il = 0; il < H.n_layer; ++il) {
        if (H.is_recr[il]) {
            ggml_context *& c = ctx_kv_x_[dev_layer_[il]];
            if (!c) c = ggml_init(kp);
            // n_seq * (1 + n_rs_seq) rows: row g * n_seq + c = cell c's state g tokens back (llama_memory_recurrent)
            r_l_[il] = ggml_new_tensor_2d(c, GGML_TYPE_F32, H.n_embd_r(), (int64_t) (1 + o.n_rs_seq) * o.n_seq);
            if (!H.lfm2) s_l_[il] = ggml_new_tensor_2d(c, GGML_TYPE_F32, H.n_embd_s(), (int64_t) (1 + o.n_rs_seq) * o.n_seq);
            ggml_format_name(r_l_[il], "cache_r_l%d", il);
            if (s_l_[il]) ggml_format_name(s_l_[il], "cache_s_l%d", il);
            if (il == H.ple_layer) {   // qwen4exp: the PLE conv history, a row of the recurrent cache (llama.cpp cache_ple_r_l)
                p_l_[il] = ggml_new_tensor_2d(c, GGML_TYPE_F32, H.n_embd_p(), (int64_t) (1 + o.n_rs_seq) * o.n_seq);
                ggml_format_name(p_l_[il], "cache_ple_r_l%d", il);
            }
        } else if (H.q4x && H.compress[il] > 0) {
            // qwen4exp: the indexer's raw keys, one head of idx_dim per cell, the attention's cells and streams, type_k
            // (llama_memory_hybrid_idx's third cache; zeroed: a block row may name a cell no token wrote yet)
            ggml_context *& c = ctx_kv_x_[dev_layer_[il]];
            if (!c) c = ggml_init(kp);
            idxk_[il] = ggml_new_tensor_3d(c, o.type_k, H.idx_dim, kv_.size, 1);
            ggml_format_name(idxk_[il], "cache_idx_k_l%d", il);
        }
    }
    for (size_t d = 0; d < gpus_.size(); ++d) {
        if (!ctx_kv_x_[d]) continue;
        buf_kv_x_[d] = ggml_backend_alloc_ctx_tensors_from_buft(ctx_kv_x_[d], ggml_backend_get_default_buffer_type(gpus_[d]));
        if (!buf_kv_x_[d]) return "recurrent state allocation failed (GPU " + std::to_string(d) + ")";
        ggml_backend_buffer_clear(buf_kv_x_[d], 0);
    }
    // lfm3: the sliding layers' own cache (one card, or the CPU); IMPULSE_SWA_FULL=1 keeps them in the main cache
    swa_ring_ = H.lfm3 && !(getenv("IMPULSE_SWA_FULL") && atoi(getenv("IMPULSE_SWA_FULL")) != 0) && gpus_.size() <= 1 && !tp_;
    if (swa_ring_) {
        const int max_ub = std::max(o_.n_ubatch, std::max(0, o_.prefill_ubatch));
        const uint32_t cells = std::min(kv_.ssize, (uint32_t) ((H.n_swa + max_ub + kSwaSlack + 255) / 256 * 256));
        kvs_.init(cells, o.n_seq);
        kvs_.k.assign(H.n_layer, nullptr); kvs_.v.assign(H.n_layer, nullptr);
        swa_floor_v_.assign(o.n_seq, 0);
        ckpt_swa_.assign((size_t) o.n_seq * kCkpts, SwaSnap{});
        ggml_init_params sp = { (size_t) (2 * H.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
        ctx_kvs_ = ggml_init(sp);
        for (int il = 0; il < H.n_layer; ++il) {
            if (!swa_layer(il)) continue;
            kvs_.k[il] = ggml_new_tensor_3d(ctx_kvs_, o.type_k, n_gqa, kvs_.size, 1);
            kvs_.v[il] = ggml_new_tensor_3d(ctx_kvs_, o.type_v, n_gqa, kvs_.size, 1);
            ggml_format_name(kvs_.k[il], "cache_k_swa_l%d", il);
            ggml_format_name(kvs_.v[il], "cache_v_swa_l%d", il);
        }
        buf_kvs_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_kvs_, cpu_only_ ? ggml_backend_cpu_buffer_type() : ggml_backend_get_default_buffer_type(gpus_[0]));
        if (!buf_kvs_) return "sliding-window cache allocation failed";
        ggml_backend_buffer_clear(buf_kvs_, 0);
        fprintf(stderr, "impulse: sliding layers in their own cache: %u cells x %d streams (%.0f MiB)\n", kvs_.ssize, o.n_seq,
                ggml_backend_buffer_get_size(buf_kvs_) / 1048576.0);
    }
    // the KV pager: the attention layers' K/V, kv_cap_ cells each, and the full-size views the reserve graphs use
    kv_cap_ = o_.kv_initial > 0 ? std::min<uint32_t>(kv_.size, (uint32_t) ((o_.kv_initial + 255) / 256 * 256)) : kv_.size;
    kvl_.assign(H.n_layer, {}); kfull_.assign(H.n_layer, nullptr); vfull_.assign(H.n_layer, nullptr);
    ggml_init_params fp = { (size_t) (2 * H.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_kvfull_ = ggml_init(fp);
    for (int il = 0; il < H.n_layer; ++il) {
        if (!kv_layer(il)) continue;
        kfull_[il] = ggml_new_tensor_3d(ctx_kvfull_, o.type_k, n_gqa, kv_.size, 1);
        vfull_[il] = ggml_new_tensor_3d(ctx_kvfull_, o.type_v, n_gqa, kv_.size, 1);
        ggml_format_name(kfull_[il], "cache_k_l%d", il);
        ggml_format_name(vfull_[il], "cache_v_l%d", il);
        const std::string e = kv_alloc_layer(il, kv_cap_);
        if (!e.empty()) return e;
    }

    if (o.pool) {   // stage c: Strata's CPU expert pool for the RAM-resident layers
        // never pin on one socket (machine A runs production); pinned only for the per-socket split (numa-always)
        if (!pool_) pool_ = std::make_unique<scpu::ExpertPool>(o.pool_threads, /*pin*/ scpu::numa_split_possible());
        pool_layers_.resize(H.n_layer);
        for (int il = H.n_dense; il < H.n_layer && il < o.n_cpu_moe; ++il) {
            const Layer & l = L_[il];
            // the kernel reads up rows at a fixed distance from the gate rows: separate gate/up tensors qualify when they
            // share type and stride (else the layer keeps ggml's CPU path)
            if (l.gate_exps->type != l.up_exps->type || l.gate_exps->nb[2] != l.up_exps->nb[2]) continue;
            auto pl = std::make_unique<PoolLayer>();
            std::string e;
            if (!scpu::native_fmt(l.gate_exps->type, l.down_exps->type, H.n_embd, H.n_ff_exp, pl->f, e,
                                  scpu::NativeAct::SwiGLU_ggml, /*separate_down*/ true))
                return "layer " + std::to_string(il) + ": " + e;
            pl->gate = (const uint8_t *) l.gate_exps->data; pl->g_stride = l.gate_exps->nb[2];
            pl->f.up_off = (size_t) ((const uint8_t *) l.up_exps->data - pl->gate);   // may wrap: the sum is the up rows
            pl->dn = (const uint8_t *) l.down_exps->data;   pl->d_stride = l.down_exps->nb[2];
            pl->pool = pool_.get();
            if (!exps_sh_.empty()) {   // the experts' second-level scales (empty: none)
                const auto & hs = exps_sh_[il];
                pl->sg = hs[0].empty() ? nullptr : hs[0].data();
                pl->su = hs[1].empty() ? nullptr : hs[1].data();
                pl->sd = hs[2].empty() ? nullptr : hs[2].data();
            }
            if (lora_) pl->lora_down = lora_->find(l.down_exps);
            if (pl->g_stride != (size_t) H.n_ff_exp * pl->f.gu_row || pl->d_stride != pl->f.d_row * (size_t) H.n_embd)
                return "layer " + std::to_string(il) + ": expert strides do not match the pool's format";
            pool_layers_[il] = std::move(pl);
        }
    }

    if (o.cache_slots != 0) {   // stage d
        if (!o.pool && o.cache_slots < H.n_expert) return "--cache-slots needs --pool (the misses go to the pool)";
        rank_.assign(H.n_layer, {});
        if (o.profile.empty()) {
            for (auto & r : rank_) for (int e = 0; e < H.n_expert; ++e) r.push_back(e);
        } else {
            FILE * pf = fopen(o.profile.c_str(), "r");
            if (!pf) return "cannot read the profile " + o.profile;
            char line[16384];
            while (fgets(line, sizeof line, pf)) {
                char * q = line; const int il = (int) strtol(q, &q, 10);
                if (*q == ':') ++q;
                if (il < 0 || il >= H.n_layer) continue;
                for (;;) { char * e2; const long v = strtol(q, &e2, 10); if (e2 == q) break; rank_[il].push_back((int) v); q = e2; }
            }
            fclose(pf);
        }
        cache_.resize(H.n_layer);
        for (int il = std::max(H.n_dense, o.cache_l0); il < std::min(o.n_cpu_moe, H.n_layer); ++il) {
            const Layer & l = L_[il];
            auto C = std::make_unique<CacheLayer>();
            C->full = o.cache_slots >= H.n_expert;
            C->src = { l.gate_exps, l.up_exps, l.down_exps };
            C->res.assign(H.n_expert, -1);
            C->score.assign(H.n_expert, 0.0);
            C->pending.assign(H.n_expert, 0);
            if (!C->full) {
                if (!pool_layers_[il]) return "layer " + std::to_string(il) + ": no pool layer for the misses";
                if ((int) rank_[il].size() < std::min(o.cache_slots < 0 ? kMinSlots : o.cache_slots, H.n_expert))
                    return "profile: layer " + std::to_string(il) + " ranks too few experts";
            }
            cache_[il] = std::move(C);
        }
        cache_cards_.clear();
        for (int il = 0; il < H.n_layer; ++il)
            if (cache_[il] && std::find(cache_cards_.begin(), cache_cards_.end(), dev_layer_[il]) == cache_cards_.end()) cache_cards_.push_back(dev_layer_[il]);
        std::sort(cache_cards_.begin(), cache_cards_.end());
        ctx_cache_c_.assign(gpus_.size(), nullptr); buf_cache_c_.assign(gpus_.size(), nullptr);
        slots_c_.assign(gpus_.size(), 0); slots_load_c_.assign(gpus_.size(), 0);
        // auto: a small arena first; sized from the VRAM left once everything else has loaded (end of load)
        const std::string ae = arena_build(o.cache_slots < 0 ? kMinSlots : o.cache_slots, true);
        if (!ae.empty()) return ae;
        for (int il = 0; il < H.n_layer; ++il) {
            CacheLayer * C = cache_[il].get();
            if (!C || C->full) continue;
            const PoolLayer & P = *pool_layers_[il];
            C->miss = std::make_unique<PoolLayer>();
            C->miss->f = P.f; C->miss->gate = P.gate; C->miss->g_stride = P.g_stride; C->miss->dn = P.dn; C->miss->d_stride = P.d_stride;
            C->miss->pool = P.pool; C->miss->res = C->res.data(); C->miss->stats = &stats_;
            C->miss->lora_down = P.lora_down;
            C->miss->sg = P.sg; C->miss->su = P.su; C->miss->sd = P.sd;
        }
    }
    if (!cache_.empty() && !exps_sh_.empty()) {   // the cached layers' expert scales on their cards, beside the slots (llama.cpp's
        // GPU layers keep theirs there); their own small buffers, which lending the arena leaves alone
        ctx_sc_c_.assign(gpus_.size(), nullptr); buf_sc_c_.assign(gpus_.size(), nullptr);
        static const char * nm[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
        for (int il = 0; il < H.n_layer; ++il) {
            CacheLayer * C = cache_[il].get();
            if (!C) continue;
            const Layer & l = L_[il];
            const ggml_tensor * src[3] = { l.gate_exps_s, l.up_exps_s, l.down_exps_s };
            for (int k = 0; k < 3; ++k) {
                if (!src[k]) continue;
                ggml_context *& c = ctx_sc_c_[dev_layer_[il]];
                if (!c) { ggml_init_params sp = { (size_t) (3 * H.n_layer + 8) * ggml_tensor_overhead(), nullptr, true }; c = ggml_init(sp); }
                C->sc[k] = ggml_dup_tensor(c, src[k]);
                ggml_format_name(C->sc[k], "blk.%d.%s.scale", il, nm[k]);
            }
        }
        for (size_t d = 0; d < gpus_.size(); ++d) {
            if (!ctx_sc_c_[d]) continue;
            buf_sc_c_[d] = ggml_backend_alloc_ctx_tensors_from_buft(ctx_sc_c_[d], ggml_backend_get_default_buffer_type(gpus_[d]));
            if (!buf_sc_c_[d]) return "expert scale allocation failed (GPU " + std::to_string(d) + ")";
            ggml_backend_buffer_set_usage(buf_sc_c_[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        for (int il = 0; il < H.n_layer; ++il)
            for (int k = 0; k < 3; ++k)
                if (cache_[il] && cache_[il]->sc[k]) ggml_backend_tensor_set(cache_[il]->sc[k], exps_sh_[il][k].data(), 0, exps_sh_[il][k].size() * sizeof(float));
    }
    if (o.overlap) {
        if (cache_.empty()) return "--overlap needs --cache-slots";
        driver_ = std::make_unique<Driver>();
        for (auto & C : cache_) if (C && C->miss) C->miss->driver = driver_.get();
    }
    if (o_.async_swaps) {   // the copy streams: expert bytes from the pinned arena, beside each card's compute stream
        if (!o.adapt) return "--async-swaps needs --adapt";
        copy_streams_.assign(gpus_.size(), nullptr);
        for (size_t d = 0; d < gpus_.size(); ++d) {
            set_card((int) d);
            cudaStream_t cs;
            if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) { set_card(0); return "cannot create a copy stream"; }
            copy_streams_[d] = cs;
        }
        set_card(0);
        copy_stream_ = copy_streams_[0];
    }

    new_sched();
    graph_buf_.resize(ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false));
    {
        const std::string e = reserve_graphs();
        if (!e.empty()) return e;
    }
    compute_bytes_c_.assign(gpus_.size(), 0);   // what end_read's arenas must leave for the graphs
    for (size_t d = 0; d < gpus_.size(); ++d) compute_bytes_c_[d] = ggml_backend_sched_get_buffer_size(sched_, gpus_[d]);
    if (!o.mtp.empty()) {
        const std::string e = load_mtp();
        if (!e.empty()) return "mtp: " + e;
    }
    for (int il = 0; il < H.n_layer; ++il) if (H.is_recr[il]) ckpt_bytes_ += r_l_[il]->nb[1] + (s_l_[il] ? s_l_[il]->nb[1] : 0) + (p_l_[il] ? p_l_[il]->nb[1] : 0);
    sync_all();   // the checkpoints on card 0 (the reserve graphs may leave another card current)
    if (host_states()) for (auto & h : ckpt_host_) h.resize(ckpt_bytes_);
    else if (ckpt_bytes_) for (void *& c : ckpt_)
        if (cudaMalloc(&c, ckpt_bytes_) != cudaSuccess) { cudaGetLastError(); c = nullptr; return "cannot allocate the recurrent checkpoints"; }
    if (o.cache_slots < 0 && !cache_.empty()) {   // auto slots, per card: what its free VRAM holds after everything above, less the reserve
        const size_t reserve = (size_t) o.vram_reserve_mb << 20;
        for (int d : cache_cards_) {
            const size_t per = slot_bytes(d);
            if (per == 0) continue;
            set_card(d);
            size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            set_card(0);
            const size_t have = (size_t) (slots_c_[d] + 1) * per;
            int S = fr + have > reserve + 2 * per ? (int) ((fr + have - reserve) / per) - 1 : kMinSlots;
            S = std::max(kMinSlots, std::min(S, H.n_expert - 1));
            std::string err;
            while (!(err = arena_build(S, true, d)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 2);
            if (!err.empty()) return "auto slots: " + err;
        }
        if (cache_cards_.size() > 1) {
            std::string m;
            for (int d : cache_cards_) m += (m.empty() ? "" : ", ") + std::string(ggml_backend_dev_name(card_dev(d))) + " " + std::to_string(slots_c_[d]);
            fprintf(stderr, "impulse: expert slots per layer by card: %s\n", m.c_str());
        }
    }
    slots_load_c_ = slots_c_;
    return "";
}

void Engine::set_card(int d) const {
    if (cpu_only_) return;
    const char * nm = ggml_backend_dev_name(ggml_backend_get_device(gpus_[d < (int) gpus_.size() ? d : 0]));
    cudaSetDevice(strncmp(nm, "CUDA", 4) == 0 ? atoi(nm + 4) : 0);
}

void Engine::sync_copies() {
    for (void * cs : copy_streams_) cudaStreamSynchronize((cudaStream_t) cs);
    commit_landed();
}

size_t Engine::slot_bytes(int card) const {
    size_t b = 0;
    for (int il = 0; il < (int) cache_.size(); ++il) {
        const auto & C = cache_[il];
        if (C && !C->full && dev_layer_[il] == card) for (ggml_tensor * t : C->src) b += t->nb[2];
    }
    return b;
}

// ---------------------------------------------------------------- the KV pager
size_t Engine::kv_bytes(uint32_t cells) const {
    const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv;
    size_t b = 0;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (kv_layer(il)) b += (ggml_row_size(o_.type_k, n_gqa) + ggml_row_size(o_.type_v, n_gqa)) * cells;
    return b;
}

// one attention layer's K/V in a buffer of cap cells: zeroed (as the whole-context buffer was at load), the old cells
// copied over, the old buffer freed, the full-size reserve views pointed at the new memory
std::string Engine::kv_alloc_layer(int il, uint32_t cap) {
    const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv;
    ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
    KvLayer nl; nl.ctx = ggml_init(p);
    ggml_tensor * k = ggml_new_tensor_3d(nl.ctx, o_.type_k, n_gqa, cap, 1);
    ggml_tensor * v = ggml_new_tensor_3d(nl.ctx, o_.type_v, n_gqa, cap, 1);
    ggml_format_name(k, "cache_k_l%d", il);
    ggml_format_name(v, "cache_v_l%d", il);
    nl.buf = ggml_backend_alloc_ctx_tensors_from_buft(nl.ctx, ggml_backend_get_default_buffer_type(gpus_[dev_layer_[il]]));
    if (!nl.buf) { ggml_free(nl.ctx); return "KV allocation failed (layer " + std::to_string(il) + ", " + std::to_string(cap) + " cells)"; }
    ggml_backend_buffer_clear(nl.buf, 0);
    if (kv_.k[il]) {
        const size_t rows = (size_t) std::min<int64_t>(kv_.k[il]->ne[1], cap);
        cudaMemcpy(k->data, kv_.k[il]->data, ggml_row_size(k->type, n_gqa) * rows, cudaMemcpyDefault);
        cudaMemcpy(v->data, kv_.v[il]->data, ggml_row_size(v->type, n_gqa) * rows, cudaMemcpyDefault);
    }
    if (kvl_[il].buf) ggml_backend_buffer_free(kvl_[il].buf);
    if (kvl_[il].ctx) ggml_free(kvl_[il].ctx);
    kvl_[il] = nl;
    kv_.k[il] = k; kv_.v[il] = v;
    if (tp_) { kfull_[il] = k; vfull_[il] = v; return ""; }   // the whole context: the reserve graphs use the cache itself
    kfull_[il]->data = k->data; kfull_[il]->buffer = nl.buf;
    vfull_[il]->data = v->data; vfull_[il]->buffer = nl.buf;
    graph_invalidate();   // the graphs' KV views point at the old buffer
    return "";
}

// room for cells: the capacity rounded up to kv_step (at most the context); the slot arena gives way first, one layer
// at a time is reallocated (so the moment's extra VRAM is one layer's new buffer)
std::string Engine::kv_grow(uint32_t cells) {
    if (cells <= kv_cap_) return "";
    const uint32_t step = (uint32_t) std::max(256, (o_.kv_step + 255) / 256 * 256);
    const uint32_t cap = std::min(kv_.size, (cells + step - 1) / step * step);
    const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv;
    const size_t one = (ggml_row_size(o_.type_k, n_gqa) + ggml_row_size(o_.type_v, n_gqa)) * cap;
    ensure_free(kv_bytes(cap) - kv_bytes(kv_cap_) + one + ((size_t) 64 << 20));
    if (!fatal_.empty()) return fatal_;
    sync_all();
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!kv_layer(il)) continue;
        const std::string e = kv_alloc_layer(il, cap);
        if (!e.empty()) return e;
    }
    fprintf(stderr, "impulse: KV on the card %u -> %u cells (%.0f MiB)\n", kv_cap_, cap, kv_bytes(cap) / 1048576.0);
    kv_cap_ = cap;
    return "";
}

// VRAM for something else on a card (the projector, its encode, the KV pager): fewer slots there (refilled from the usage
// counts unless keep_profile). card -1: the first card with an arena
int Engine::ensure_free(size_t bytes, bool keep_profile, int card) {
    if (cpu_only_) return slots_now_;   // no card memory to free
    if (card < 0) card = cache_cards_.empty() ? 0 : cache_cards_.front();
    if (card >= (int) buf_cache_c_.size() || !buf_cache_c_[card] || slot_bytes(card) == 0) return slots_on(card);
    set_card(card);
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    set_card(0);
    if (fr >= bytes) return slots_c_[card];
    if (copy_stream_) sync_copies();
    const size_t per = slot_bytes(card);
    int S = std::max(kMinSlots, slots_c_[card] - (int) ((bytes - fr + per - 1) / per) - 1);
    const int before = slots_c_[card];
    std::string err;
    while (!(err = arena_build(S, keep_profile, card)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 2);
    if (!err.empty()) { fatal_ = "making room: " + err; return 0; }
    if (S < before) { stats_.arena_shrinks++; fprintf(stderr, "impulse: expert arena %d -> %d slots per layer (VRAM, %s)\n", before, S, ggml_backend_dev_name(card_dev(card))); }
    if (keep_profile) return slots_c_[card];
    refill_pending_ = true;   // the next small ubatch refills every layer to its top-S by usage
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_c_[card];
}

int Engine::grow_to(int S, size_t headroom, int card) {
    if (cpu_only_) return slots_now_;
    if (card < 0) card = cache_cards_.empty() ? 0 : cache_cards_.front();
    if (card >= (int) buf_cache_c_.size() || !buf_cache_c_[card] || slot_bytes(card) == 0 || S <= slots_c_[card]) return slots_on(card);
    if (copy_stream_) sync_copies();
    set_card(card);
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    set_card(0);
    const size_t per = slot_bytes(card), now = (size_t) (slots_c_[card] + 1) * per;
    const int fit = fr + now > headroom + per ? (int) ((fr + now - headroom) / per) - 1 : slots_c_[card];
    int T = std::min(S, fit);
    if (T <= slots_c_[card]) return slots_c_[card];
    const int before = slots_c_[card];
    std::string err;
    while (!(err = arena_build(T, false, card)).empty() && T > before) T = std::max(before, T - 2);
    if (!err.empty()) { fatal_ = "growing the arena: " + err; return 0; }
    fprintf(stderr, "impulse: expert arena %d -> %d slots per layer (VRAM free again, %s)\n", before, T, ggml_backend_dev_name(card_dev(card)));
    refill_pending_ = true;
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_c_[card];
}

// ---------------------------------------------------------------- the prompt cache
int Engine::resume_point(int pos, int seq) const {
    const int n_past_ = n_past_v_[seq], rs_src_ = rs_src_v_[seq], last_ub_n_ = last_ub_v_[seq];
    if (pos >= n_past_) return n_past_;
    if (hp_.flint) return pos;   // attention only: any position (its cells are dropped from there on)
    const int r = n_past_ - pos;
    if (rs_src_ == 0 && r <= o_.n_rs_seq && r < last_ub_n_ && swa_resumable(seq, pos)) return pos;   // inside the last ubatch's snapshots
    int best = 0;   // a checkpoint (lfm3: it keeps its sliding window)
    for (int c = seq * kCkpts; c < (seq + 1) * kCkpts; ++c) if (ckpt_pos_[c] > best && ckpt_pos_[c] <= pos) best = ckpt_pos_[c];
    return best;
}

std::string Engine::rollback_to(int pos, int seq) {
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    int & n_past_ = n_past_v_[seq]; int & rs_src_ = rs_src_v_[seq]; int & last_ub_n_ = last_ub_v_[seq];
    if (pos == n_past_) return "";
    if (pos == 0) { reset_seq(seq); return ""; }
    if (hp_.flint) {   // no recurrent state: the sequence's cells from pos on go, nothing to restore
        if (pos > n_past_) return "cannot resume at position " + std::to_string(pos);
        kv_.seq_rm(seq, pos);
        rs_src_ = 0; last_ub_n_ = 0; fresh_v_[seq] = 0;
        n_past_ = pos;
        return "";
    }
    const int r = n_past_ - pos;
    if (r > 0 && rs_src_ == 0 && r <= o_.n_rs_seq && r < last_ub_n_) return rollback(pos, seq);
    int c = seq * kCkpts;
    while (c < (seq + 1) * kCkpts && !(ckpt_have(c) && ckpt_pos_[c] == pos)) ++c;
    if (c == (seq + 1) * kCkpts) return "cannot resume at position " + std::to_string(pos);
    sync_all();   // ggml-cuda's streams do not block the default one; the states may sit on other cards (cudaMemcpyDefault)
    if (swa_ring_) swa_load(seq, ckpt_swa_[c]);   // the window the checkpoint's position reads
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[il]) continue;
        for (ggml_tensor * t : { r_l_[il], s_l_[il], p_l_[il] }) {   // the cell's group-0 row = the state the next ubatch reads
            if (!t) continue;
            const size_t row = (size_t) seq * t->nb[1];
            if (host_states()) ggml_backend_tensor_set(t, ckpt_host_[c].data() + off, row, t->nb[1]);
            else cudaMemcpy((char *) t->data + row, (char *) ckpt_[c] + off, t->nb[1], cudaMemcpyDefault);
            off += t->nb[1];
        }
    }
    rs_src_ = 0; last_ub_n_ = 0; fresh_v_[seq] = 0;
    kv_.seq_rm(seq, pos);
    for (int k = seq * kCkpts; k < (seq + 1) * kCkpts; ++k) if (ckpt_pos_[k] > pos) ckpt_pos_[k] = -1;   // their attention cells are gone
    if (mtp_.loaded) { mtp_.kv.seq_rm(seq, pos); mtp_h_v_[seq] = ckpt_h_[c]; }
    n_past_ = pos;
    return "";
}

// the recurrent state now (its newest snapshot, or the one a pending rollback points at), the drafter's pending row
void Engine::checkpoint(int seq) {
    if (seq < 0 || seq >= n_seq()) return;
    const int n_past_ = n_past_v_[seq], rs_src_ = rs_src_v_[seq], c0 = seq * kCkpts;
    int c = c0;   // the same position again, else a free one (a rollback freed it), else the oldest
    for (int k = c0; k < c0 + kCkpts; ++k) if (ckpt_pos_[k] == n_past_) { c = k; goto found; }
    for (int k = c0; k < c0 + kCkpts; ++k) if (ckpt_pos_[k] < 0) { c = k; goto found; }
    for (int k = c0 + 1; k < c0 + kCkpts; ++k) if (ckpt_age_[k] < ckpt_age_[c]) c = k;
found:
    if (!ckpt_have(c)) return;
    sync_all();
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[il]) continue;
        for (ggml_tensor * t : { r_l_[il], s_l_[il], p_l_[il] }) {
            if (!t) continue;
            const size_t row = ((size_t) rs_src_ * n_seq() + seq) * t->nb[1];   // row g * n_seq + cell
            if (host_states()) ggml_backend_tensor_get(t, ckpt_host_[c].data() + off, row, t->nb[1]);
            else cudaMemcpy((char *) ckpt_[c] + off, (char *) t->data + row, t->nb[1], cudaMemcpyDefault);
            off += t->nb[1];
        }
    }
    ckpt_pos_[c] = n_past_;
    ckpt_age_[c] = ++ckpt_clock_;
    ckpt_h_[c] = mtp_h_v_[seq];
    if (swa_ring_) ckpt_swa_[c] = swa_save(seq, n_past_);
}

// ---------------------------------------------------------------- the RAM prompt cache
struct Engine::Parked {
    int n_past = 0;
    struct CacheRows {                   // a Cache's stream: cells [0, rows), their metadata and every tensor's rows
        uint32_t rows = 0;
        std::vector<int32_t> pos, tok, ex, ey;
        std::vector<std::vector<uint8_t>> t;
    } kv, mkv;
    std::vector<uint8_t> rs_now;         // the recurrent rows the next ubatch would read, as a checkpoint lays them out
    std::vector<std::vector<uint8_t>> ckpt;
    std::vector<int> ckpt_pos;
    std::vector<uint64_t> ckpt_age;      // their order of taking (oldest goes first)
    std::vector<std::vector<float>> ckpt_h;
    std::vector<float> mtp_h;
    SwaSnap swa;                         // lfm3: the sliding cache's cells of the sequence, and each checkpoint's window
    std::vector<SwaSnap> ckpt_swa;
    size_t bytes = 0;
};
size_t Engine::parked_bytes(const Parked & p) { return p.bytes; }
int Engine::parked_pos(const Parked & p) { return p.n_past; }

std::shared_ptr<Engine::Parked> Engine::park(int seq) {
    if (seq < 0 || seq >= n_seq()) return nullptr;
    sync_all();
    auto P = std::make_shared<Parked>();
    P->n_past = n_past_v_[seq];
    auto save = [&](const Cache & C, const std::vector<ggml_tensor *> & ts, Parked::CacheRows & R) {
        const int st = C.stream_of(seq);
        const size_t base = (size_t) st * C.ssize;
        R.rows = C.used_max_p1(st);
        R.pos.assign(C.pos.begin() + base, C.pos.begin() + base + R.rows);
        R.tok.assign(C.tok.begin() + base, C.tok.begin() + base + R.rows);
        R.ex.assign(C.ex.begin() + base, C.ex.begin() + base + R.rows);
        R.ey.assign(C.ey.begin() + base, C.ey.begin() + base + R.rows);
        for (int32_t & x : R.pos) if (x >= 0 && C.seq[base + (&x - R.pos.data())] != seq) x = -1;   // another sequence's cell
        for (ggml_tensor * t : ts) {
            R.t.emplace_back((size_t) R.rows * t->nb[1]);
            if (R.rows) ggml_backend_tensor_get(t, R.t.back().data(), base * t->nb[1], R.t.back().size());
            P->bytes += R.t.back().size();
        }
        P->bytes += (size_t) R.rows * 16;
    };
    std::vector<ggml_tensor *> ts;
    for (int il = 0; il < hp_.n_layer; ++il) if (kv_layer(il)) { ts.push_back(kv_.k[il]); ts.push_back(kv_.v[il]); if (idxk_[il]) ts.push_back(idxk_[il]); }
    save(kv_, ts, P->kv);
    if (mtp_.loaded) save(mtp_.kv, { mtp_.kv.k[0], mtp_.kv.v[0] }, P->mkv);
    if (swa_ring_) { P->swa = swa_save(seq, INT32_MAX); P->bytes += P->swa.rows.size() + P->swa.pos.size() * 4; }
    // the state the next ubatch reads: row rs_src * n_seq + seq of every recurrent tensor (as checkpoint() copies it)
    P->rs_now.resize(ckpt_bytes_);
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[il]) continue;
        for (ggml_tensor * t : { r_l_[il], s_l_[il], p_l_[il] }) {
            if (!t) continue;
            ggml_backend_tensor_get(t, P->rs_now.data() + off, ((size_t) rs_src_v_[seq] * n_seq() + seq) * t->nb[1], t->nb[1]);
            off += t->nb[1];
        }
    }
    P->bytes += P->rs_now.size();
    for (int c = seq * kCkpts; c < (seq + 1) * kCkpts; ++c) {
        if (!ckpt_have(c) || ckpt_pos_[c] < 0) continue;
        P->ckpt.emplace_back(ckpt_bytes_);
        if (host_states()) memcpy(P->ckpt.back().data(), ckpt_host_[c].data(), ckpt_bytes_);
        else cudaMemcpy(P->ckpt.back().data(), ckpt_[c], ckpt_bytes_, cudaMemcpyDefault);
        P->ckpt_pos.push_back(ckpt_pos_[c]); P->ckpt_age.push_back(ckpt_age_[c]); P->ckpt_h.push_back(ckpt_h_[c]);
        P->bytes += ckpt_bytes_;
        if (swa_ring_) { P->ckpt_swa.push_back(ckpt_swa_[c]); P->bytes += ckpt_swa_[c].rows.size(); }
    }
    P->mtp_h = mtp_h_v_[seq];
    return P;
}

std::string Engine::unpark(int seq, const Parked & P) {
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    if (!fatal_.empty()) return fatal_;
    reset_seq(seq);
    sync_all();
    auto load = [&](Cache & C, const std::vector<ggml_tensor *> & ts, const Parked::CacheRows & R) -> std::string {
        const int st = C.stream_of(seq);
        const size_t base = (size_t) st * C.ssize;
        if (R.rows > C.ssize || R.t.size() != ts.size()) return "the parked state does not fit this engine";
        for (uint32_t c = 0; c < R.rows; ++c) {
            if (C.pos[base + c] >= 0 && C.seq[base + c] != seq) return "the stream holds another sequence";
            C.pos[base + c] = R.pos[c]; C.seq[base + c] = R.pos[c] >= 0 ? (int8_t) seq : (int8_t) -1;
            C.tok[base + c] = R.tok[c]; C.ex[base + c] = R.ex[c]; C.ey[base + c] = R.ey[c];
        }
        C.heads[st] = R.rows;
        for (size_t k = 0; k < ts.size(); ++k) if (R.rows) ggml_backend_tensor_set(ts[k], R.t[k].data(), base * ts[k]->nb[1], R.t[k].size());
        return "";
    };
    {   // the KV pager: room for the parked cells first (a grow moves the K/V tensors)
        const uint32_t need = (uint32_t) ((size_t) kv_.stream_of(seq) * kv_.ssize + P.kv.rows);
        if (need > kv_cap_) { const std::string e = kv_grow(need); if (!e.empty()) return e; }
    }
    std::vector<ggml_tensor *> ts;
    for (int il = 0; il < hp_.n_layer; ++il) if (kv_layer(il)) { ts.push_back(kv_.k[il]); ts.push_back(kv_.v[il]); if (idxk_[il]) ts.push_back(idxk_[il]); }
    std::string e = load(kv_, ts, P.kv);
    if (e.empty() && mtp_.loaded && P.mkv.rows) e = load(mtp_.kv, { mtp_.kv.k[0], mtp_.kv.v[0] }, P.mkv);
    if (e.empty() && swa_ring_) {
        if (P.swa.pos.size() != kvs_.ssize) e = "the parked sliding window does not fit this engine";
        else swa_load(seq, P.swa);
    }
    if (!e.empty()) { reset_seq(seq); return e; }
    if (P.rs_now.size() != ckpt_bytes_) { reset_seq(seq); return "the parked recurrent state does not fit this engine"; }
    size_t off = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_recr[il]) continue;
        for (ggml_tensor * t : { r_l_[il], s_l_[il], p_l_[il] }) {
            if (!t) continue;
            ggml_backend_tensor_set(t, P.rs_now.data() + off, (size_t) seq * t->nb[1], t->nb[1]);   // row 0 * n_seq + seq
            off += t->nb[1];
        }
    }
    std::vector<size_t> by_age(P.ckpt.size());   // the oldest first, so their new ages keep the order
    for (size_t k = 0; k < by_age.size(); ++k) by_age[k] = k;
    std::sort(by_age.begin(), by_age.end(), [&](size_t a, size_t b) { return P.ckpt_age[a] < P.ckpt_age[b]; });
    for (size_t j = 0; j < by_age.size() && j < (size_t) kCkpts; ++j) {
        const size_t k = by_age[j];
        const int c = seq * kCkpts + (int) j;
        if (!ckpt_have(c) || P.ckpt[k].size() != ckpt_bytes_) continue;
        if (host_states()) memcpy(ckpt_host_[c].data(), P.ckpt[k].data(), ckpt_bytes_);
        else cudaMemcpy(ckpt_[c], P.ckpt[k].data(), ckpt_bytes_, cudaMemcpyDefault);
        ckpt_pos_[c] = P.ckpt_pos[k]; ckpt_age_[c] = ++ckpt_clock_; ckpt_h_[c] = P.ckpt_h[k];
        if (swa_ring_) { if (k < P.ckpt_swa.size()) ckpt_swa_[c] = P.ckpt_swa[k]; else ckpt_pos_[c] = -1; }
    }
    n_past_v_[seq] = P.n_past; rs_src_v_[seq] = 0; last_ub_v_[seq] = 0; fresh_v_[seq] = 0;
    mtp_h_v_[seq] = P.mtp_h;
    sync_all();
    return "";
}

// an image chunk: one ubatch (a big one streams the experts through the card like a text read, so it borrows the
// arena's VRAM the same way)
std::string Engine::decode_embd(const float * embd, int n, const int32_t * pos4, int n_pos, std::vector<float> * logits, int seq) {
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    cur_seq_ = seq;
    int & n_past_ = n_past_v_[seq]; int & last_ub_n_ = last_ub_v_[seq];
    const bool big = n > o_.n_ubatch || (o_.lend && n >= kOffloadMin);
    if (big) begin_read(std::max(n, o_.n_ubatch));
    embd_in_ = embd; pos4_in_ = pos4;
    const std::string err = ubatch(nullptr, n, n_past_, logits ? 1 : 0, logits);
    embd_in_ = nullptr; pos4_in_ = nullptr;
    if (!err.empty()) return err;
    n_past_ += n_pos;
    last_ub_n_ = 0;   // no snapshot rollback into an image
    return "";
}

// [CUDA0, (CUDA1 ..,) CPU], the CPU's compute buffers in pinned host memory (llama_context), from the pool that keeps them
// across reads (lite/pinned_host.hpp: ggml-cuda's host type would cudaMallocHost / cudaFreeHost them with every scheduler).
// Several cards: pipeline parallelism (llama_context: n_devices > 1, all layers offloaded, split mode layer), ubatches queued
void Engine::new_sched() {
    graph_invalidate();
    if (sched_) ggml_backend_sched_free(sched_);
    std::vector<ggml_backend_t> backs;
    std::vector<ggml_backend_buffer_type_t> bufts;
    if (cpu_only_) { backs = { cpu_ }; bufts = { ggml_backend_cpu_buffer_type() }; }
    else {
    backs = gpus_;
    backs.push_back(cpu_);
    for (auto * g : gpus_) bufts.push_back(ggml_backend_get_default_buffer_type(g));
    bufts.push_back(lite::pinned_host_buffer_type(cpu_));
    }
    // pipeline parallelism only with every layer's experts on the cards (llama_context: no tensor overrides such as --n-cpu-moe)
    sched_ = ggml_backend_sched_new(backs.data(), bufts.data(), (int) backs.size(), kMaxNodes,
                                    /*parallel*/ gpus_.size() > 1 && (hp_.dense || o_.n_cpu_moe <= 0), /*op_offload*/ true);
}

void Engine::sync_all() {
    for (auto * g : gpus_) ggml_backend_synchronize(g);
    if (gpus_.size() > 1 || tp_) cudaSetDevice(0);   // ggml-cuda leaves the last card it ran on current
}

// llama_context::sched_reserve: pp (n_ubatch tokens, n_outputs = min(n_ubatch, n_batch)), tg (1, 1), pp again. Run at load
// and again after a chunked read, so small steps get the same allocation plan (and ggml-cuda fusions) as llama.cpp's
std::string Engine::reserve_graphs() {
    const int n_pp = std::min(o_.n_ctx, o_.n_ubatch);
    // lfm3 without a drafter: a prompt piece outputs its last row and a slots' step one row per slot, so the big graph is
    // reserved with n_seq outputs, not n_pp (128K-vocab logits for 512 rows: 262 MB); a larger output still allocates
    const int n_out_pp = hp_.lfm3 && !mtp_.loaded ? std::min(n_pp, n_seq()) : n_pp;
    std::vector<int32_t> zeros(n_pp, 0);
    for (auto [nt, no] : { std::pair<int, int>{ n_pp, n_out_pp }, { 1, 1 }, { n_pp, n_out_pp } }) {
        const std::string e = ubatch(zeros.data(), nt, 0, no, nullptr, no);
        if (!e.empty()) return "reserve: " + e;
    }
    return "";
}

// ---------------------------------------------------------------- the read's expert rings (Strata's async expert ring)
// A read streams every RAM-resident layer's expert tensors to the card once per chunk. Through ggml's op offload each
// copy is a blocking cudaMemcpy before the split that uses it, so the link and the card take turns. Here each card gets R
// slots; a copy thread issues the chunk's tensors in the order the graph reads them (layer by layer: up, gate, down) on
// its own stream, at most R ahead, and the scheduler's eval callback waits for a slot's copy before the node that reads
// it and frees the slot after. The arithmetic is unchanged (the same bytes, the same mul_mat_id). Opt-in for now,
// IMPULSE_RING=k: k layers of slots. Measured 10-17 on Flash-Next (5070 + 4070 Ti SUPER): it hides only the kernels
// (~1-1.5 s of a ~4.5 s stage per 4K chunk), ggml copies only the routed experts of a small chunk (the ring copies whole
// tensors), and the stage compute buffers grew ~2 GB with it, so the 5070 ran out at 4K chunks.
struct RingUse {   // a graph's ring nodes: node -> (card, position in the card's chunk order); position -1: a flush point
    Engine * eng = nullptr;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> use;
};

struct Engine::Ring {
    int card = 0, R = 0, per = 0;
    size_t slot_bytes = 0;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    std::vector<std::array<ggml_tensor *, 3>> view;   // per layer: the up / gate / down views (nullptr: not a ring layer)
    std::vector<std::array<int, 3>> pos;              // per layer: the views' positions in the chunk order
    std::vector<const ggml_tensor *> src;             // the chunk order: the host tensors
    std::vector<int> slot_of;                         // position -> slot
    std::vector<size_t> pad;                          // position -> the card's row padding after the tensor's bytes
    std::vector<void *> slot_ptr;
    int64_t total = 0, chunk = 0;
    cudaStream_t stream = nullptr;
    std::vector<cudaEvent_t> ev;                      // per slot: its latest copy
    std::mutex mu; std::condition_variable cv;
    std::vector<int64_t> issued;                      // per slot: the sequence item last copied in (-1: none)
    std::vector<char> consumed;                       // per sequence item
    bool stop = false;
    std::thread th;
    double wait_ms = 0;
    void run() {
        cudaSetDevice(card);
        for (int64_t j = 0; j < total; ++j) {
            const int p = (int) (j % per), sl = slot_of[p];
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || issued[sl] < 0 || consumed[issued[sl]]; });
                if (stop) return;
            }
            cudaMemcpyAsync(slot_ptr[sl], src[p]->data, ggml_nbytes(src[p]), cudaMemcpyHostToDevice, stream);
            if (pad[p] > 0) cudaMemsetAsync((char *) slot_ptr[sl] + ggml_nbytes(src[p]), 0, pad[p], stream);   // ggml-cuda's zero row padding
            cudaEventRecord(ev[sl], stream);
            { std::lock_guard<std::mutex> g(mu); issued[sl] = j; }
            cv.notify_all();
        }
    }
    // the node at chunk position p is about to run: its slot holds this chunk's tensor once the copy lands
    void acquire(int p) {
        const int64_t j = chunk * per + p;
        const int sl = slot_of[p];
        const auto t0 = std::chrono::steady_clock::now();
        { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop || issued[sl] == j; }); }
        cudaEventSynchronize(ev[sl]);
        wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    void release(int p) {   // the node ran (the scheduler synchronized): its slot can take the next tensor
        { std::lock_guard<std::mutex> g(mu); consumed[chunk * per + p] = 1; if (p == per - 1) ++chunk; }
        cv.notify_all();
    }
};

// ask: a flush point runs alone (the attention before the experts computes while the copies land); a ring node waits for
// its slot. observe: a ring node frees its slot
static bool ring_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * U = (RingUse *) ud;
    auto it = U->use.find(t);
    if (it == U->use.end()) return !ask;
    if (it->second.second < 0) return true;
    Engine::Ring & R = *U->eng->ring_at(it->second.first);
    if (ask) R.acquire(it->second.second); else R.release(it->second.second);
    return true;
}

bool Engine::ring_layer(int il) const {   // a layer whose read takes the whole-layer RAM path (mm_path over the host tensors)
    const Layer & l = L_[il];
    // pinned sources only (the RAM tier's arena): from the file mapping a cudaMemcpyAsync runs at pageable speed
    if (hp_.dense || !l.up_exps || !buf_arena_) return false;
    for (const ggml_tensor * t : { l.up_exps, l.gate_exps, l.down_exps }) if (!t || t->buffer != buf_arena_) return false;
    if (!cache_.empty() && cache_[il] && cache_[il]->full) return false;
    return !lora_;
}

static int ring_k() { static const int k = getenv("IMPULSE_RING") ? atoi(getenv("IMPULSE_RING")) : 0; return k; }

size_t Engine::ring_bytes(int d) const {   // what ring_begin takes on card d: R slots of the largest ring tensor
    const int k = ring_k();
    if (k <= 0 || hp_.dense || o_.n_cpu_moe <= 0) return 0;
    size_t slot = 0; int per = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (dev_layer_[il] != d || !ring_layer(il)) continue;
        const Layer & l = L_[il];
        for (const ggml_tensor * t : { l.up_exps, l.gate_exps, l.down_exps }) {
            slot = std::max(slot, ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(gpus_[d]), t)); ++per;
        }
    }
    return (size_t) std::min(per, 3 * k) * GGML_PAD(slot, 256);
}

void Engine::ring_begin(int n_chunks, int ub) {
    (void) ub;
    const int k = ring_k();
    ring_.clear();
    if (k <= 0 || n_chunks <= 0) return;
    ring_.resize(gpus_.size());
    for (size_t d = 0; d < gpus_.size(); ++d) {
        auto Rp = std::make_unique<Ring>();
        Ring & R = *Rp;
        R.card = (int) d;
        R.view.assign(hp_.n_layer, { nullptr, nullptr, nullptr });
        R.pos.assign(hp_.n_layer, { -1, -1, -1 });
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (dev_layer_[il] != (int) d || !ring_layer(il)) continue;
            const Layer & l = L_[il];
            for (const ggml_tensor * t : { l.up_exps, l.gate_exps, l.down_exps }) {   // the card's allocation size (quantized rows are padded)
                const size_t a = ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(gpus_[d]), t);
                R.src.push_back(t); R.pad.push_back(a - ggml_nbytes(t));
                R.slot_bytes = std::max(R.slot_bytes, a);
            }
        }
        R.per = (int) R.src.size();
        if (R.per == 0) continue;
        R.R = std::min(R.per, 3 * k);
        R.buf = ggml_backend_alloc_buffer(gpus_[d], (size_t) R.R * GGML_PAD(R.slot_bytes, 256));
        if (!R.buf) { fprintf(stderr, "impulse: no VRAM for the read's expert ring on GPU %zu (%d slots): ggml's copies instead\n", d, R.R); continue; }
        ggml_backend_buffer_set_usage(R.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        char * base = (char *) ggml_backend_buffer_get_base(R.buf);
        for (int sl = 0; sl < R.R; ++sl) R.slot_ptr.push_back(base + (size_t) sl * GGML_PAD(R.slot_bytes, 256));
        ggml_init_params ip = { (size_t) (R.per + 4) * ggml_tensor_overhead(), nullptr, true };
        R.ctx = ggml_init(ip);
        int p = 0;
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (dev_layer_[il] != (int) d || !ring_layer(il)) continue;
            const Layer & l = L_[il];
            int w = 0;
            for (ggml_tensor * t : { l.up_exps, l.gate_exps, l.down_exps }) {
                ggml_tensor * v = ggml_dup_tensor(R.ctx, t);
                ggml_format_name(v, "ring_%s", t->name);
                const int sl = p % R.R;
                R.slot_of.push_back(sl);
                ggml_backend_tensor_alloc(R.buf, v, R.slot_ptr[sl]);
                R.view[il][w] = v; R.pos[il][w] = p;
                ++p; ++w;
            }
        }
        set_card((int) d);
        cudaStreamCreateWithFlags(&R.stream, cudaStreamNonBlocking);
        R.ev.resize(R.R);
        for (auto & e : R.ev) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        set_card(0);
        R.issued.assign(R.R, -1);
        R.total = (int64_t) n_chunks * R.per;
        R.consumed.assign(R.total, 0);
        R.th = std::thread([&R] { R.run(); });
        ring_[d] = std::move(Rp);
    }
    graph_invalidate();   // the read's graphs read the ring views
}

void Engine::ring_end() {
    if (ring_.empty()) return;
    double w = 0;
    for (auto & Rp : ring_) {
        if (!Rp) continue;
        Ring & R = *Rp;
        { std::lock_guard<std::mutex> g(R.mu); R.stop = true; }
        R.cv.notify_all();
        if (R.th.joinable()) R.th.join();
        set_card(R.card);
        if (R.stream) { cudaStreamSynchronize(R.stream); cudaStreamDestroy(R.stream); }
        for (auto e : R.ev) cudaEventDestroy(e);
        w += R.wait_ms;
    }
    set_card(0);
    graph_invalidate();
    for (auto & g : g_st_) graph_drop(g);
    for (auto & Rp : ring_) if (Rp) { if (Rp->ctx) ggml_free(Rp->ctx); if (Rp->buf) ggml_backend_buffer_free(Rp->buf); }
    ring_.clear();
    stats_.ring_wait_ms += w;
}

// ---------------------------------------------------------------- stage g: chunked reads with VRAM lent from the arena
void Engine::begin_read(int ub) {
    (void) ub;
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = true;
    graph_invalidate();
    if (o_.lend && arena_live()) {   // give every card's arena VRAM to the read
        if (copy_stream_) sync_copies();
        stats_.lent_bytes = 0;
        for (size_t d = 0; d < buf_cache_c_.size(); ++d) {
            if (!buf_cache_c_[d]) continue;
            stats_.lent_bytes += ggml_backend_buffer_get_size(buf_cache_c_[d]);
            ggml_backend_buffer_free(buf_cache_c_[d]); buf_cache_c_[d] = nullptr;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_cache_c_[d]); t; t = ggml_get_next_tensor(ctx_cache_c_[d], t)) { t->data = nullptr; t->buffer = nullptr; }
        }
        for (auto & C : cache_) if (C) {
            std::fill(C->res.begin(), C->res.end(), -1); std::fill(C->owner.begin(), C->owner.end(), -1);
            std::fill(C->pending.begin(), C->pending.end(), 0);
        }
        new_sched();   // its compute buffers grow for the big ubatches
    }
    stats_.lend_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::end_read() {
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = false;
    graph_invalidate();
    new_sched();   // drop the big ubatches' compute buffers
    bool any_ctx = false; for (auto * c : ctx_cache_c_) any_ctx = any_ctx || c;
    if (o_.lend && !arena_live() && any_ctx) {   // the arenas come back empty; the refill copies in the background
        // the read may leave less VRAM than it lent (ggml-cuda's pool keeps what the big ubatches grew): fewer slots then
        for (int d : cache_cards_) {
            int S = slots_c_[d];
            const int before = S;
            const size_t per = slot_bytes(d);
            if (per == 0) continue;
            if (o_.cache_slots < 0) {   // auto slots: what the card's VRAM holds after the graphs' buffers and the reserve, as at load
                set_card(d);
                size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
                set_card(0);
                const size_t need = compute_bytes_c_[d] + ((size_t) o_.vram_reserve_mb << 20);
                const int fit = fr > need + 2 * per ? (int) ((fr - need) / per) - 1 : kMinSlots;
                S = std::max(kMinSlots, std::min(S, fit));
            }
            std::string err;
            while (!(err = arena_build(S, false, d)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 4);
            if (!err.empty()) { fatal_ = "after the read: " + err; return; }
            if (S < before) { stats_.arena_shrinks++; fprintf(stderr, "impulse: expert arena %d -> %d slots per layer (VRAM, %s)\n", before, S, ggml_backend_dev_name(card_dev(d))); }
        }
    }
    const std::string e = reserve_graphs();
    if (!e.empty()) { fatal_ = "after the read: " + e; return; }
    read_end_ = std::chrono::steady_clock::now();
    refill_watch_ = true;
    stats_.restore_ms += std::chrono::duration<double, std::milli>(read_end_ - t0).count();
}

// ggml-cuda's expert matmul (mm_ids_helper, mmid.cu) keeps 4 bytes per token of a ubatch in shared memory, so an expert
// model's read chunk must fit every card's opt-in shared memory per block: 99 KB on RTX 30 / 40 / 50 = 25,344 tokens,
// 64 KB on RTX 20 = 16,384. A larger chunk stops on GGML_ASSERT(nbytes_shared <= smpbo) (LFM2.5 at 32K, 2026-10-20).
static int moe_ubatch_max() {
    static const int m = [] {
        int n = 0, best = INT_MAX;
        if (cudaGetDeviceCount(&n) != cudaSuccess) return best;
        for (int d = 0; d < n; ++d) {
            int v = 0;
            if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, d) == cudaSuccess && v > 0) best = std::min(best, v / 4);
        }
        return best;
    }();
    return m;
}

int Engine::pick_chunk(int n) {
    // what the read can use: free VRAM, the arena it is about to borrow, and (a read already under way, e.g. a text run
    // after an image) the compute buffers the read holds
    // several cards: the tightest one (each holds its layers' share of the chunk's buffers, mask and indexer scratch)
    const bool lending = o_.lend && (arena_live() || reading_);
    size_t avail = SIZE_MAX;
    for (size_t d = 0; d < gpus_.size(); ++d) {
        set_card((int) d);
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        const size_t lent = (o_.lend && d < buf_cache_c_.size() && buf_cache_c_[d]) ? ggml_backend_buffer_get_size(buf_cache_c_[d]) : 0;
        const size_t have = fr + lent + (reading_ ? ggml_backend_sched_get_buffer_size(sched_, gpus_[d]) : 0), rb = ring_bytes((int) d);
        avail = std::min(avail, have > rb ? have - rb : 0);   // the read's expert ring comes out of the same VRAM
    }
    set_card(0);
    const size_t n_kv_end = (size_t) n_past_v_[cur_seq_] + n, margin = (size_t) 256 << 20;
    // the KV pager grows during this read: its new cells come out of the same VRAM
    const uint32_t step = (uint32_t) std::max(256, (o_.kv_step + 255) / 256 * 256);
    const uint32_t cells = (uint32_t) std::min<size_t>(kv_.size, (n_kv_end + step - 1) / step * step);
    const size_t kv_more = cells > kv_cap_ ? kv_bytes(cells) - kv_bytes(kv_cap_) : 0;
    avail = avail > kv_more ? avail - kv_more : 0;
    // qwen4exp: fitted to the read compute buffers measured 10-17 on Flash-Next (4K-12K chunks, 4K-50K cells: ~0.56 GB +
    // 347 KB per token + 4.25 B per token and cell, the indexer's slices included), plus 10 %
    if (const char * e = getenv("IMPULSE_READ_ROOM_LOG")) { (void) e; fprintf(stderr, "impulse: read room %.2f GB on the tightest card\n", avail / 1e9); }
    for (int P : { 32768, 16384, 12288, 8192, 6144, kChunkMin, 2048, 1024 }) {
        if (P < kChunkMin && !hp_.q4x) break;   // qwen4exp at long contexts: smaller chunks before none
        if (!hp_.q4x && (P == 12288 || P == 6144)) continue;
        if (!hp_.dense && P > moe_ubatch_max()) continue;   // the expert matmul's shared memory (moe_ubatch_max)
        const size_t m = (size_t) std::min(P, n);
        const size_t need = hp_.q4x ? (size_t) ((kChunkFixed + m * kChunkPerTokQ4x + m * n_kv_end * 9 / 2) * 1.1) + margin
                                    : kChunkFixed + m * kChunkPerTok + m * n_kv_end * 2 + margin;
        if (need <= avail) return P;
    }
    return lending ? kChunkMin : o_.n_ubatch;   // the maintainer, 2026-10-10: minimum chunk 4k
}

// every cached layer gets S slots (+ the zero slot) per expert tensor; with fill_profile the top S of the profile are
// copied in (a full layer: every expert, identity order)
std::string Engine::arena_build(int S, bool fill_profile, int card) {
    if (card < 0) {
        for (int d : cache_cards_) { const std::string e = arena_build(S, fill_profile, d); if (!e.empty()) return e; }
        return "";
    }
    graph_invalidate();   // the slot tensors are made anew
    if (buf_cache_c_[card]) { ggml_backend_buffer_free(buf_cache_c_[card]); buf_cache_c_[card] = nullptr; }
    if (ctx_cache_c_[card]) { ggml_free(ctx_cache_c_[card]); ctx_cache_c_[card] = nullptr; }
    ggml_init_params cp = { (size_t) (12 * hp_.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ctx_cache_c_[card] = ggml_init(cp);
    const int NE = hp_.n_expert;
    auto mine = [&](int il) { return cache_[il] && dev_layer_[il] == card; };
    for (int il = 0; il < (int) cache_.size(); ++il) {
        if (!mine(il)) continue;
        CacheLayer * C = cache_[il].get();
        C->S = C->full ? NE : S;
        const int alloc = C->full ? NE : C->S + 1;
        C->slot.clear(); C->all.clear();
        static const char * nm[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
        for (size_t k = 0; k < C->src.size(); ++k) {
            ggml_tensor * t = ggml_new_tensor_3d(ctx, C->src[k]->type, C->src[k]->ne[0], C->src[k]->ne[1], alloc);
            ggml_format_name(t, "blk.%d.%s.slots", il, nm[k]);
            C->slot.push_back(t);
            if (!C->full && il == o_.dcheck_layer) C->all.push_back(ggml_dup_tensor(ctx, C->src[k]));
        }
        std::fill(C->res.begin(), C->res.end(), -1);
        C->res_f = nullptr;
        if (C->full) { for (int e = 0; e < NE; ++e) C->res[e] = e; continue; }
        C->owner.assign(C->S, -1);
        C->res_f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, NE);
        ggml_format_name(C->res_f, "blk.%d.ffn_exps.slot_of", il);
    }
    ggml_backend_buffer_t buf = buf_cache_c_[card] = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(gpus_[card]));
    if (!buf) return "expert cache allocation failed (" + std::to_string(S) + " slots do not fit on " + ggml_backend_dev_name(card_dev(card)) + ")";
    ggml_backend_buffer_clear(buf, 0);                     // slot S (the misses' target) stays all zeros
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int il = 0; il < (int) cache_.size(); ++il) {
        if (!mine(il)) continue;
        CacheLayer * C = cache_[il].get();
        for (size_t k = 0; k < C->src.size(); ++k) {
            ggml_tensor * src = C->src[k];
            const size_t es = src->nb[2];
            if (C->full) ggml_backend_tensor_set(C->slot[k], src->data, 0, ggml_nbytes(src));
            else if (fill_profile)
                for (int j = 0; j < std::min(C->S, (int) rank_[il].size()); ++j)
                    ggml_backend_tensor_set(C->slot[k], (const char *) src->data + (size_t) rank_[il][j] * es, (size_t) j * es, es);
            if (!C->all.empty()) ggml_backend_tensor_set(C->all[k], src->data, 0, ggml_nbytes(src));
        }
        if (C->full) continue;
        if (fill_profile) for (int j = 0; j < std::min(C->S, (int) rank_[il].size()); ++j) { C->res[rank_[il][j]] = j; C->owner[j] = rank_[il][j]; }
        std::vector<float> rf(NE);
        for (int e = 0; e < NE; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, rf.size() * sizeof(float));
    }
    slots_c_[card] = S;
    slots_now_ = slots_c_[cache_cards_.front()];
    return "";
}

// ---------------------------------------------------------------- stage e: adaptation
// After a ubatch every cached layer's scores decay per token and count its routed experts (Gate 0's lfu_decay rule). A
// prompt read (a big ubatch) only marks a refill: the next small ubatch first re-sets every layer to its top-S, then
// each small ubatch swaps at most adapt_budget experts per layer. The copies are issued after the step's compute and its
// twin check: synchronous, or on the copy stream with async_swaps (committed at the start of a later ubatch).
void Engine::adapt_after(int n, const std::vector<ggml_tensor *> & argsorts) {
    const double d = pow(0.5, 1.0 / o_.adapt_half_life);
    const int k = hp_.n_expert_used;
    // every cached layer's routing in one go: async copies into pinned memory, then one wait per card (one
    // synchronous copy per layer cost ~0.2 ms each, ~10 ms a token on Flash-Next's 48 layers)
    std::vector<size_t> off(hp_.n_layer, 0);
    size_t total = 0;
    for (int il = 0; il < hp_.n_layer; ++il) if (argsorts[il]) { off[il] = total; total += ggml_nbytes(argsorts[il]); }
    if (total > ids_host_bytes_) {
        if (ids_host_) cudaFreeHost(ids_host_);
        ids_host_ = nullptr; ids_host_bytes_ = 0;
        if (cudaMallocHost(&ids_host_, total) == cudaSuccess) ids_host_bytes_ = total; else { cudaGetLastError(); ids_host_ = nullptr; }
    }
    std::vector<int32_t> ids_sync;
    const auto ta0 = std::chrono::steady_clock::now();
    if (ids_host_) {
        std::vector<char> used(gpus_.size(), 0);
        for (int il = 0; il < hp_.n_layer; ++il) {
            ggml_tensor * t = argsorts[il];
            if (!t) continue;
            if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) { memcpy((char *) ids_host_ + off[il], t->data, ggml_nbytes(t)); continue; }
            const int dv = dev_layer_[il];
            ggml_backend_tensor_get_async(gpus_[dv], t, (char *) ids_host_ + off[il], 0, ggml_nbytes(t));
            used[dv] = 1;
        }
        for (size_t dv = 0; dv < gpus_.size(); ++dv) if (used[dv]) ggml_backend_synchronize(gpus_[dv]);
        if (gpus_.size() > 1) cudaSetDevice(0);
    }
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!argsorts[il]) continue;
        CacheLayer * C = cache_[il].get();
        const int32_t * idp;
        if (ids_host_) idp = (const int32_t *) ((const char *) ids_host_ + off[il]);
        else {
            ids_sync.resize(ggml_nelements(argsorts[il]));
            ggml_backend_tensor_get(argsorts[il], ids_sync.data(), 0, ids_sync.size() * sizeof(int32_t));
            idp = ids_sync.data();
        }
        const int32_t * ids = idp;
        const int64_t ne = argsorts[il]->ne[0];
        // LFM's last block routes only requested output rows during prefill.
        for (int64_t t = 0; t < argsorts[il]->ne[1]; ++t) {
            for (auto & v : C->score) v *= d;
            for (int j = 0; j < k; ++j) {
                const int32_t e = ids[t * ne + j];
                if (e < 0 || e >= hp_.n_expert) {
                    fprintf(stderr, "adapt: layer %d token %lld slot %d id %d (argsort %s op %s ne %lld,%lld flags %d)\n", il, (long long) t, j, e,
                            argsorts[il]->name, ggml_op_desc(argsorts[il]), (long long) argsorts[il]->ne[0], (long long) argsorts[il]->ne[1],
                            argsorts[il]->flags);
                    abort();
                }
                C->score[e] += 1.0;
            }
        }
    }
    if (n < kOffloadMin) stats_.t_ad_score += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta0).count();
    if (n >= kOffloadMin) { refill_pending_ = o_.adapt_refill; return; }
    const bool refill = refill_pending_;
    refill_pending_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t before = stats_.swaps;
    // the refill after a lent read goes in pieces over the next steps (IMPULSE_REFILL_STEP experts per layer and step,
    // default 16; 0 = all at once as before): queued whole (~5,000 experts on Flash-Next) it filled the copy queue and
    // held the first step ~2 s; in pieces the steps run while the copies land
    static const int refill_step = getenv("IMPULSE_REFILL_STEP") ? atoi(getenv("IMPULSE_REFILL_STEP")) : 16;
    const int budget = refill ? (refill_step > 0 ? std::max(refill_step, o_.adapt_budget) : INT32_MAX) : o_.adapt_budget;
    bool more = false;   // a layer that used its whole refill budget has more to fill; one that swapped less has reached its top-S
    for (int il = 0; il < hp_.n_layer; ++il)
        if (argsorts[il]) {
            const uint64_t s0 = stats_.swaps;
            swap_layer(il, budget);
            if (refill && stats_.swaps - s0 >= (uint64_t) budget) more = true;
        }
    // the next step goes on refilling only while there is more to fill (before: while any expert swapped, which the
    // ordinary adaptation does nearly every step, so the refill budget never ended: 8x the swaps of adapt_budget)
    if (refill && refill_step > 0 && more) refill_pending_ = true;
    if (copy_stream_ && !pending_.empty() && !pending_.back().closed) {   // an event per card closes this step's batch of copies
        if (pending_.back().sw.empty()) pending_.pop_back();
        else {
            PendingBatch & b = pending_.back();
            for (size_t d = 0; d < copy_streams_.size(); ++d) {
                if (!(b.cards & (1u << d))) continue;
                set_card((int) d);
                cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                cudaEventRecord(ev, (cudaStream_t) copy_streams_[d]);
                b.evs.push_back(ev);
            }
            set_card(0);
            b.closed = true;
            uint64_t inflight = 0; for (auto & b : pending_) inflight += b.sw.size();
            stats_.pending_peak = std::max<uint64_t>(stats_.pending_peak, inflight);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (refill) { stats_.refill_swaps += stats_.swaps - before; stats_.refill_ms += ms; }
    else stats_.swap_ms += ms;
}

void Engine::swap_layer(int il, int budget) {
    CacheLayer * C = cache_[il].get();
    const int NE = hp_.n_expert;
    std::vector<int> want, worst;
    for (int e = 0; e < NE; ++e) (C->res[e] < 0 ? want : worst).push_back(e);
    std::stable_sort(want.begin(), want.end(), [&](int a, int b) { return C->score[a] > C->score[b]; });
    std::stable_sort(worst.begin(), worst.end(), [&](int a, int b) { return C->score[a] < C->score[b]; });
    if (copy_stream_) {   // async: the victim goes to the CPU now, the newcomer when its copy has landed
        want.erase(std::remove_if(want.begin(), want.end(), [&](int e) { return C->pending[e] != 0; }), want.end());
        if (pending_.empty() || pending_.back().closed) pending_.push_back(PendingBatch{});
    }
    std::vector<int> free_slots;
    for (int j = 0; j < C->S; ++j) if (C->owner[j] == -1) free_slots.push_back(j);
    size_t wi = 0;
    int done = 0;
    for (size_t i = 0; i < want.size() && done < budget; ++i) {
        const int a = want[i];
        int slot, w = -1;
        if (!free_slots.empty()) {
            if (C->score[a] <= 0) break;
            slot = free_slots.back(); free_slots.pop_back();
        } else {
            if (wi >= worst.size()) break;
            w = worst[wi++];
            if (C->score[a] <= C->score[w]) break;
            slot = C->res[w];
        }
        const int card = dev_layer_[il];
        if (copy_stream_) { set_card(card); pending_.back().cards |= 1u << card; }   // the layer's card's copy stream
        for (size_t k = 0; k < C->src.size(); ++k) {
            const size_t es = C->src[k]->nb[2];
            if (copy_stream_)
                cudaMemcpyAsync((char *) C->slot[k]->data + (size_t) slot * es, (const char *) C->src[k]->data + (size_t) a * es, es,
                                cudaMemcpyHostToDevice, (cudaStream_t) copy_streams_[card]);
            else ggml_backend_tensor_set(C->slot[k], (const char *) C->src[k]->data + (size_t) a * es, (size_t) slot * es, es);
            stats_.swap_bytes += es;
        }
        if (copy_stream_) set_card(0);
        if (w >= 0) C->res[w] = -1;
        if (copy_stream_) {   // a slot being written is referenced by no expert until it commits
            C->owner[slot] = -2; C->pending[a] = 1;
            pending_.back().sw.push_back({ il, a, slot });
        } else { C->res[a] = slot; C->owner[slot] = a; }
        stats_.swaps++;
        done++;
    }
    if (done) {
        std::vector<float> rf(NE);
        for (int e = 0; e < NE; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, NE * sizeof(float));
    }
}

// async swaps: newcomers whose copies have landed become resident (in order: batches complete in stream order)
void Engine::commit_landed() {
    std::vector<char> touched(hp_.n_layer, 0);
    size_t k = 0;
    for (; k < pending_.size(); ++k) {
        PendingBatch & b = pending_[k];
        bool landed = b.closed;
        for (void * e : b.evs) landed = landed && cudaEventQuery((cudaEvent_t) e) == cudaSuccess;
        if (!landed) break;
        for (const PendingSwap & p : b.sw) {
            CacheLayer * C = cache_[p.il].get();
            C->res[p.e] = p.slot; C->owner[p.slot] = p.e; C->pending[p.e] = 0;
            touched[p.il] = 1; stats_.commits++;
        }
        for (void * e : b.evs) cudaEventDestroy((cudaEvent_t) e);
        stats_.commit_batches++;
    }
    cudaGetLastError();
    pending_.erase(pending_.begin(), pending_.begin() + k);
    if (refill_watch_ && pending_.empty() && !refill_pending_) {
        stats_.refill_land_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - read_end_).count();
        refill_watch_ = false;
    }
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!touched[il]) continue;
        CacheLayer * C = cache_[il].get();
        std::vector<float> rf(hp_.n_expert);
        for (int e = 0; e < hp_.n_expert; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, rf.size() * sizeof(float));
    }
}

void Engine::reset() {
    kv_.clear();
    if (swa_ring_) { kvs_.clear(); std::fill(swa_floor_v_.begin(), swa_floor_v_.end(), 0); }
    sync_all();
    for (auto * b : buf_kv_x_) if (b) ggml_backend_buffer_clear(b, 0);   // the recurrent cells start at zero (llama.cpp: rs_z)
    for (const KvLayer & kl : kvl_) if (kl.buf) ggml_backend_buffer_clear(kl.buf, 0);
    std::fill(fresh_v_.begin(), fresh_v_.end(), 1);
    std::fill(rs_src_v_.begin(), rs_src_v_.end(), 0);
    std::fill(n_past_v_.begin(), n_past_v_.end(), 0);
    std::fill(last_ub_v_.begin(), last_ub_v_.end(), 0);
    for (int & c : ckpt_pos_) c = -1;   // their attention cells are gone
    for (auto & h : mtp_h_v_) h.assign(h_width(), 0.0f);
    if (mtp_.loaded) mtp_.kv.clear();
    // the KV pager: a new conversation gives the grown cells back, and the slot arena takes them again
    const uint32_t init = o_.kv_initial > 0 ? std::min<uint32_t>(kv_.size, (uint32_t) ((o_.kv_initial + 255) / 256 * 256)) : kv_.size;
    if (kv_cap_ > init && !kvl_.empty()) {
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (!kv_layer(il)) continue;
            const std::string e = kv_alloc_layer(il, init);
            if (!e.empty()) { fatal_ = "new conversation: " + e; return; }
        }
        fprintf(stderr, "impulse: KV on the card %u -> %u cells (a new conversation)\n", kv_cap_, init);
        kv_cap_ = init;
        for (int d : cache_cards_) if (slots_load_c_[d] > slots_c_[d]) grow_to(slots_load_c_[d], (size_t) o_.vram_reserve_mb << 20, d);
    }
}

// one sequence's new conversation (llama_memory_recurrent::seq_rm(s, 0, -1)): its cells free, its recurrent cell new (read as
// zeros at its next ubatch: rs_z), its checkpoints and drafter cells gone. The KV pager shrinks only with every sequence
void Engine::reset_seq(int s) {
    if (n_seq() == 1) { reset(); return; }
    if (s < 0 || s >= n_seq()) return;
    kv_.seq_rm(s, 0);
    if (swa_ring_) { kvs_.seq_rm(s, 0); swa_floor_v_[s] = 0; }
    fresh_v_[s] = 1; rs_src_v_[s] = 0; n_past_v_[s] = 0; last_ub_v_[s] = 0;
    for (int c = s * kCkpts; c < (s + 1) * kCkpts; ++c) ckpt_pos_[c] = -1;
    mtp_h_v_[s].assign(h_width(), 0.0f);
    if (mtp_.loaded) mtp_.kv.seq_rm(s, 0);
}

// ---------------------------------------------------------------- the graph
// llm_graph_context::build_norm (LLM_NORM_RMS, weight) + llama_context::graph_get_cb's pin of "norm" nodes to the layer's
// device (full offload)
ggml_tensor * Engine::norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, int il) {
    cur = ggml_rms_norm(ctx, cur, hp_.eps);
    if (w) {
        if (il >= 0) {
            ggml_format_name(cur, "norm-%d", il);
            ggml_backend_t lb = gpus_[dev_layer_[il]];   // the layer's card (llama_context::graph_get_cb)
            if (ggml_backend_supports_op(lb, cur)) ggml_backend_sched_set_tensor_backend(sched_, cur, lb);
        } else {
            ggml_set_name(cur, "norm");
        }
        cur = ggml_mul(ctx, cur, w);
    }
    return cur;
}

// TDUMP=<file|-> TDUMP_N=<n|-2>: the same named-node hashing (and so the same scheduler split points) as route-oracle's
static bool tdump_want(const ggml_tensor * t) {
    static const char * pfx[] = { "attn_norm-", "Qcur_pos-", "Kcur_pos-", "kqv_out-", "attn_out-", "ffn_mlp-", "ffn_moe_logits-",
                                  "ffn_moe_weights_norm-", "ffn_moe_out-", "l_out-", "result_norm", "result_output", nullptr };
    static const int n = getenv("TDUMP_N") ? atoi(getenv("TDUMP_N")) : -1;
    if (!getenv("TDUMP")) return false;
    bool ok = false;
    if (const char * want = getenv("TDUMP_PFX")) {   // a comma-separated prefix list instead of the default one (both tools)
        for (const char * a = want; *a;) {
            const char * b = strchr(a, ','); const size_t len = b ? (size_t) (b - a) : strlen(a);
            if (len && strncmp(t->name, a, len) == 0) ok = true;
            if (!b) break; a = b + 1;
        }
    } else for (int i = 0; pfx[i]; ++i) if (strncmp(t->name, pfx[i], strlen(pfx[i])) == 0) ok = true;
    if (!ok) return false;
    const int64_t nt = strncmp(t->name, "Qcur_pos-", 9) == 0 || strncmp(t->name, "Kcur_pos-", 9) == 0 ? t->ne[2] : t->ne[1];
    return (n == -2 || nt == n) && ggml_is_contiguous(t);
}
static void tdump_put(const ggml_tensor * t) {
    static FILE * f = strcmp(getenv("TDUMP"), "-") == 0 ? nullptr : fopen(getenv("TDUMP"), "w");
    if (!f) return;
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t c : b) { h ^= c; h *= 1099511628211ULL; }
    fprintf(f, "%s %016llx %s\n", t->name, (unsigned long long) h, ggml_type_name(t->type));
    fflush(f);
}
// Q35_GRAPHDUMP=<file>: every node of the 3rd single-token ubatch, as route-oracle's ROUTE_GRAPHDUMP writes them
static FILE * g_gdump = nullptr;
static bool gdump_cb(ggml_tensor * t, bool ask, void *) {
    if (!ask) return true;
    fprintf(g_gdump, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(g_gdump, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(g_gdump, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(g_gdump, "\n");
    return false;
}
// GDUMP=<file> GDUMP_N=<n>: every node of the ubatch with n tokens (route-oracle's GDUMP format)
static void gdump_multi(const ggml_tensor * t) {
    static const char * path = getenv("GDUMP");
    static const int n = getenv("GDUMP_N") ? atoi(getenv("GDUMP_N")) : -1;
    static bool on = false;
    static FILE * f = nullptr;
    if (!path) return;
    if (!on && strcmp(t->name, "model.input_embed") == 0 && t->ne[1] == n) { on = true; f = fopen(path, "w"); }
    if (!on) return;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\t%d\n", (int) t->flags);
    if (strcmp(t->name, "result_output") == 0) { fclose(f); f = nullptr; on = false; path = nullptr; }
}
static bool tdump_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) gdump_multi(t);
    if (!tdump_want(t)) return ask ? false : true;
    if (ask) return true;
    tdump_put(t);
    return true;
}

// qwen4exp: the n-gram rows of each token (llm_graph_input_ple::set_input + llama_kv_cache::get_prev_tokens). The window is
// the token and its n-1 predecessors in its sequence, each the token of the cell at the largest position <= p (the highest
// cell on equal positions; apply() has stored this ubatch, so its own tokens count); a missing one or the reset token
// (ple.eos_token_id) cuts the window, which reads the reset token from there. An embedding ubatch (an image) hashes the
// image stand-in and finds its predecessors by ubatch order, as llama.cpp does.
void Engine::set_ple_rows(ggml_tensor * rows, const int32_t * toks, int n, int ns, int nt, int head, const std::vector<int32_t> & positions) {
    const HParams & H = hp_;
    const int32_t img = H.ple_img ? H.ple_img : H.ple_eos;
    const int n_prev = H.ple_ngram - 1, NH = H.ple_n_heads;
    std::vector<int32_t> idx((size_t) NH * n);
    std::vector<int32_t> want((size_t) n * n_prev);   // the position each predecessor is read at (-1: none)
    for (int q = 0; q < ns; ++q) {
        const int st = kv_.stream_of(head + q);
        int32_t lo = INT32_MAX, hi = -1;
        for (int i = 0; i < nt; ++i) {
            const int gi = q * nt + i;
            for (int sb = 1; sb <= n_prev; ++sb) {
                int32_t p;
                if (toks) p = positions[gi] - sb;
                else { const int k = i - sb; p = k >= 0 ? positions[q * nt + k] : positions[q * nt] + k; }
                want[(size_t) gi * n_prev + sb - 1] = p;
                if (p >= 0) { lo = std::min(lo, p); hi = std::max(hi, p); }
            }
        }
        if (hi < 0) lo = 0;
        std::vector<int32_t> at(hi >= lo ? (size_t) (hi - lo + 1) : 0, -1);
        std::vector<uint8_t> have(at.size(), 0);
        int32_t below_p = -1, below_t = -1;
        const size_t base = (size_t) st * kv_.ssize;
        for (uint32_t j = 0; j < kv_.ssize; ++j) {   // ascending cells: a later cell of the same position wins
            const int32_t p = kv_.pos[base + j];
            if (p < 0 || kv_.seq[base + j] != head + q) continue;
            if (p >= lo && p <= hi) { at[p - lo] = kv_.tok[base + j]; have[p - lo] = 1; }
            else if (p < lo && p >= below_p) { below_p = p; below_t = kv_.tok[base + j]; }
        }
        auto tok_le = [&](int32_t p) -> int32_t {
            if (p < 0) return -1;
            for (int32_t x = std::min(p, hi); x >= lo; --x) if (have[x - lo]) return at[x - lo];
            return below_p >= 0 && below_p <= p ? below_t : -1;
        };
        for (int i = 0; i < nt; ++i) {
            const int gi = q * nt + i;
            int64_t cx[8];
            cx[0] = toks ? toks[gi] : img;
            bool cut = false;
            for (int sb = 1; sb <= n_prev; ++sb) {
                const int32_t t = cut ? -1 : tok_le(want[(size_t) gi * n_prev + sb - 1]);
                cut = cut || t < 0 || t == H.ple_eos;
                cx[sb] = cut ? H.ple_eos : t;
            }
            for (int ng = 2; ng <= H.ple_ngram; ++ng) {
                uint64_t mixed = (uint64_t) cx[0] * H.ple_mult[0];
                for (int j = 1; j < ng; ++j) mixed ^= (uint64_t) cx[j] * H.ple_mult[j];
                const int b0 = (ng - 2) * H.ple_per_gram;
                for (int g = 0; g < H.ple_per_gram; ++g) {
                    const int h = b0 + g;
                    idx[(size_t) gi * NH + h] = (int32_t) (mixed % H.ple_vsz[h] + H.ple_off[h]);
                }
            }
        }
    }
    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size() * sizeof(int32_t));
}

// qwen4exp: llama_memory_hybrid_idx::set_input_qsa (one sequence per stream, the per-block bias). Per stream the cells are
// grouped by bucket idx / r, idx = the cell's position, or for an image ubatch whose cells repeat positions its rank in
// raster order (position, then y, then x); a block is a bucket with all r slots present, numbered in bucket order (its
// rope position: the first slot's, by rank the cell's own three sections); every other cell points at the spare block.
// Per query the bias is -inf for a block id past the full ones, 1e9 for a block at or after the query's own open block
// (always visible), 0 otherwise; the spare block (the incomplete tail) 1e9. Repeated positions in a text ubatch group as
// llama.cpp does (the slots OR together, the last cell of a slot is its member).
std::string Engine::set_qsa(const Graph & G, int ns, int nt, int head, const std::vector<int32_t> & positions, uint32_t n_kv, const int32_t * pos4, int n) {
    const int64_t r = G.qsa_r, n_blocks = ((int64_t) n_kv + r - 1) / r;
    const uint64_t full = r == 64 ? ~0ull : ((1ull << r) - 1);
    std::vector<int32_t> cell_blk((size_t) n_kv * ns), blk_cells((size_t) r * n_blocks * ns, 0), blk_pos((size_t) 4 * n_blocks * ns, 0);
    std::vector<float> bias((size_t) n_blocks * nt * ns);
    std::vector<uint64_t> slots(n_blocks);
    std::vector<int32_t> slot0(n_blocks), bid_of(n_blocks), bid_idx, bid_slot0, rank, order;
    for (int q = 0; q < ns; ++q) {
        const size_t base = (size_t) kv_.stream_of(head + q) * kv_.ssize;
        bool ranked = false, dup = false;
        auto idx_of = [&](uint32_t j) { return ranked ? rank[j] : kv_.pos[base + j]; };
        auto group = [&]() {
            std::fill(slots.begin(), slots.end(), 0ull); std::fill(slot0.begin(), slot0.end(), -1);
            dup = false;
            for (uint32_t j = 0; j < n_kv; ++j) {
                if (kv_.pos[base + j] < 0) continue;
                const int64_t idx = idx_of(j), pb = idx / r;
                if (pb >= n_blocks) return false;
                const uint64_t bit = 1ull << (idx % r);
                dup |= (slots[pb] & bit) != 0;
                slots[pb] |= bit;
                if (idx % r == 0) slot0[pb] = (int32_t) j;
            }
            return true;
        };
        if (!group()) return "qsa: a cell position runs past the cell window";
        if (dup && pos4) {   // an image: rank the cells by position, then y, then x (the M-RoPE causal mask's order)
            order.clear();
            for (uint32_t j = 0; j < n_kv; ++j) if (kv_.pos[base + j] >= 0) order.push_back((int32_t) j);
            std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                const int32_t pa = kv_.pos[base + a], pb = kv_.pos[base + b];
                if (pa != pb) return pa < pb;
                return kv_.after_2d(base + b, kv_.ex[base + a], kv_.ey[base + a]);
            });
            rank.assign(n_kv, -1);
            for (size_t k = 0; k < order.size(); ++k) rank[order[k]] = (int32_t) k;
            ranked = true;
            if (!group()) return "qsa: a cell rank runs past the cell window";
        }
        int32_t n_bid = 0;
        bid_idx.clear(); bid_slot0.clear();
        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            bid_of[pb] = slots[pb] == full ? n_bid++ : -1;
            if (bid_of[pb] >= 0) { bid_idx.push_back((int32_t) (pb * r)); bid_slot0.push_back(slot0[pb]); }
        }
        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };
            if (ranked) { const size_t c = base + bid_slot0[b]; sec[0] = kv_.pos[c]; sec[1] = kv_.ey[c]; sec[2] = kv_.ex[c]; sec[3] = kv_.pos[c]; }
            for (int k = 0; k < 4; ++k) blk_pos[(size_t) k * (n_blocks * ns) + (size_t) q * n_blocks + b] = sec[k];
        }
        const bool have_dead = n_bid < n_blocks;
        const int32_t dead = have_dead ? n_bid : (int32_t) n_blocks - 1;
        int32_t * cb = cell_blk.data() + (size_t) q * n_kv, * bc = blk_cells.data() + (size_t) q * r * n_blocks;
        for (uint32_t j = 0; j < n_kv; ++j) {
            int32_t b = -1;
            if (kv_.pos[base + j] >= 0) { const int64_t idx = idx_of(j); b = bid_of[idx / r]; if (b >= 0) bc[(size_t) b * r + idx % r] = (int32_t) j; }
            cb[j] = b < 0 ? dead : b;
        }
        for (int ii = 0; ii < nt; ++ii) {
            const int i = q * nt + ii;
            int64_t qp = positions[i];
            if (ranked) {   // the query's rank: the cells at or before it in raster order, less one
                const int32_t qt = pos4[i], qy = pos4[n + i], qx = pos4[2 * n + i];
                int64_t lo = 0, hi = (int64_t) order.size();
                while (lo < hi) {
                    const int64_t mid = (lo + hi) / 2;
                    const size_t c = base + order[mid];
                    if (kv_.pos[c] < qt || (kv_.pos[c] == qt && !kv_.after_2d(c, qx, qy))) lo = mid + 1; else hi = mid;
                }
                qp = lo - 1;
            }
            const int64_t tail_start = (qp + 1) / r * r;
            float * bb = bias.data() + (size_t) i * n_blocks;
            for (int64_t b = 0; b < n_blocks; ++b) bb[b] = b >= n_bid ? -INFINITY : (bid_idx[b] >= tail_start ? 1e9f : 0.0f);
            if (have_dead) bb[dead] = 1e9f;
        }
    }
    ggml_backend_tensor_set(G.cell_blk, cell_blk.data(), 0, cell_blk.size() * sizeof(int32_t));
    ggml_backend_tensor_set(G.blk_cells, blk_cells.data(), 0, blk_cells.size() * sizeof(int32_t));
    ggml_backend_tensor_set(G.blk_pos, blk_pos.data(), 0, blk_pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(G.qsa_bias, bias.data(), 0, bias.size() * sizeof(float));
    ggml_backend_tensor_set(G.qsa_k_idxs, kv_.idxs.data(), 0, kv_.idxs.size() * sizeof(int64_t));
    return "";
}

// reserve = llama_context::graph_reserve: the graph for n tokens and n_outputs outputs over the full KV cache (init_full),
// handed to ggml_backend_sched_reserve instead of computed. The allocator keeps the plan of the last reserve and reuses
// it for every real graph that fits, so the reserve sequence decides where tensors land (and with it which ggml-cuda
// fusions pass their memory-overlap check).
// qwen4exp's hyper-connection pieces (llama.cpp src/models/qwen4exp.cpp), shared by the target's graph and the drafter's
static void hc_name(ggml_tensor * t, const char * s, int il) { if (il >= 0) ggml_format_name(t, "%s-%d", s, il); else ggml_set_name(t, s); }
// build_hc_mix: grouped RMS norm, the low-rank gate, the streams' mean; inject = the [hc, T] scatter weights
static ggml_tensor * hc_mix_build(ggml_context * ctx, const HParams & H, ggml_tensor * x, ggml_tensor * wn, ggml_tensor * wd,
                                  ggml_tensor * wu, ggml_tensor * wi, ggml_tensor ** inject, int il) {
    const int64_t T = x->ne[2], n_embd = H.n_embd, HCD = (int64_t) H.hc * n_embd;
    ggml_tensor * xn = ggml_rms_norm(ctx, x, H.eps);
    xn = ggml_reshape_2d(ctx, xn, HCD, T);
    xn = ggml_mul(ctx, xn, wn);
    hc_name(xn, "hc_norm", il);
    ggml_tensor * lo = ggml_mul_mat(ctx, wd, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float) H.hc));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, wu, lo));
    hc_name(gate, "hc_gate", il);
    ggml_tensor * gated = ggml_mul(ctx, xn, gate);
    gated = ggml_reshape_3d(ctx, gated, n_embd, H.hc, T);
    ggml_tensor * mixed = ggml_view_2d(ctx, gated, n_embd, T, ggml_row_size(gated->type, n_embd) * H.hc, 0);
    mixed = ggml_cont(ctx, mixed);
    for (int64_t c = 1; c < H.hc; ++c) {
        ggml_tensor * sv = ggml_view_2d(ctx, gated, n_embd, T, ggml_row_size(gated->type, n_embd) * H.hc, ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx, mixed, sv);
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float) H.hc);
    hc_name(mixed, "hc_mixed", il);
    if (inject) { *inject = ggml_mul_mat(ctx, wi, xn); hc_name(*inject, "hc_inject", il); }
    return mixed;
}
// build_hc_combine: res[:, c, t] += 2 sigmoid(inject[c, t] / hc) out[:, t]
static ggml_tensor * hc_combine_build(ggml_context * ctx, const HParams & H, ggml_tensor * res, ggml_tensor * out, ggml_tensor * inject, int il) {
    const int64_t T = res->ne[2], n_embd = H.n_embd;
    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) H.hc));
    w = ggml_scale(ctx, w, 2.0f);
    w = ggml_reshape_3d(ctx, w, 1, H.hc, T);
    ggml_tensor * b = ggml_reshape_3d(ctx, out, n_embd, 1, T);
    b = ggml_repeat_4d(ctx, b, n_embd, H.hc, T, 1);
    ggml_tensor * c = ggml_add(ctx, res, ggml_mul(ctx, b, w));
    hc_name(c, "hc_combine", il);
    return c;
}

std::string Engine::ubatch(const int32_t * toks, int n, int pos0, int n_out, std::vector<float> * logits, int reserve_outputs) {
    const HParams & H = hp_;
    const bool reserve = reserve_outputs > 0;
    const bool timed = !reserve && n < kOffloadMin;
    const auto tb0 = std::chrono::steady_clock::now();
    dcheck_ = {}; // graph-local tensors must never survive a reserve or a prior ubatch
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = pos4_in_ ? pos4_in_[i] : pos0 + i;   // M-RoPE: the KV cells keep section t
    std::vector<int8_t> seqs(n, (int8_t) cur_seq_);
    if (!ub_pos_.empty() && !reserve) { positions = ub_pos_; seqs = ub_seq_; }   // decode_multi: several sequences, seq-major
    if (st_replay_ && !reserve) { positions = st_replay_->positions; seqs = st_replay_->seqs; }   // a pipelined read's later stage
    const int L0 = (!reserve && st_l1_ >= 0) ? st_l0_ : 0, L1 = (!reserve && st_l1_ >= 0) ? st_l1_ : hp_.n_layer;
    // the ubatch's sequences: ns consecutive ones from `head` (each its own recurrent cell), nt tokens each (split_equal)
    const int ns = (!reserve && ub_ns_ > 0) ? ub_ns_ : 1, nt = n / ns, head = reserve ? 0 : seqs[0];
    const int NS = n_seq();   // recurrent rows: g * NS + cell
    if (!reserve) {
        if (!fatal_.empty()) return fatal_;
        if (reading_ && n < kOffloadMin) { end_read(); if (!fatal_.empty()) return fatal_; }
        if (copy_stream_) commit_landed();
        if (st_replay_) { kv_.idxs = st_replay_->idxs; kv_.n_kv = st_replay_->n_kv; }   // the cells the chunk's first stage took
        else {
        kv_.idxs.clear();
        for (int q = 0; q < ns; ++q) if (!kv_.find_slot(nt, kv_.stream_of(head + q))) return "KV cache full";
        const uint32_t need = (uint32_t) *std::max_element(kv_.idxs.begin(), kv_.idxs.end()) + 1;
        if (need > kv_cap_) { const std::string e = kv_grow(need); if (!e.empty()) return e; }
        // an image ubatch (M-RoPE, 4 sections): the cells keep its y / x sections (llama_kv_cache::apply_ubatch, is_pos_2d)
        const int32_t * py = pos4_in_ ? pos4_in_ + n : nullptr, * px = pos4_in_ ? pos4_in_ + 2 * n : nullptr;
        kv_.apply(positions, seqs, H.q4x && !embd_in_ ? toks : nullptr, H.q4x ? (H.ple_img ? H.ple_img : H.ple_eos) : -1, py, px);
        kv_.n_kv = kv_.calc_n_kv(kv_.stream_of(head), ns);
        if (st_rec_) { st_rec_->idxs = kv_.idxs; st_rec_->n_kv = kv_.n_kv; st_rec_->positions = positions; st_rec_->seqs = seqs; }
        }
        if (swa_ring_) {   // the sliding cache: position p in cell p mod ssize of its sequence's stream (the layout follows from the
            // positions alone, so a resumed sequence reads the same cells in the same order as one read from the start); the cell's
            // old position (p - ssize or earlier) is one no row of this ubatch can see (ssize >= n_swa + the ubatch)
            kvs_.idxs.clear();
            for (int q = 0; q < ns; ++q) {
                const int s = head + q, st = kvs_.stream_of(s);
                const int32_t pmin = *std::min_element(positions.begin() + (size_t) q * nt, positions.begin() + (size_t) (q + 1) * nt);
                for (int i = 0; i < nt; ++i) {
                    const size_t c = (size_t) st * kvs_.ssize + (uint32_t) positions[(size_t) q * nt + i] % kvs_.ssize;
                    const int32_t old = kvs_.pos[c];
                    if (old >= 0) {
                        if (kvs_.seq[c] != s || old > pmin - H.n_swa) return "sliding-window cache full";
                        swa_floor_v_[s] = std::max(swa_floor_v_[s], old + 1);   // every earlier position of s is gone too
                    }
                    kvs_.idxs.push_back((int64_t) c);
                }
            }
            kvs_.apply(positions, seqs);
            kvs_.n_kv = kvs_.calc_n_kv(kvs_.stream_of(head), ns);
        }
    }
    const int st0 = reserve ? 0 : kv_.stream_of(head);   // the ubatch's first KV stream (its streams are consecutive)
    const uint32_t n_kv = reserve ? kv_.ssize : kv_.n_kv;
    const uint32_t n_kv_s = !swa_ring_ ? 0 : reserve ? kvs_.ssize : kvs_.n_kv;   // lfm3: the sliding cache's cells
    // flint: the sliding layers read only the cells from swa_lo on: below it every cell is free or older than any row of the
    // ubatch can see (p1 - p0 >= n_swa), so they are masked anyway. Aligned down to 256 cells (llama.cpp's n_kv padding),
    // so the graph changes every 256 cells. The same values as attending every cell masked, not the same summation split:
    // IMPULSE_SWA_FULL=1 reads every cell (llama.cpp --swa-full: the bit-for-bit gates)
    uint32_t swa_lo = 0;
    static const bool swa_full = getenv("IMPULSE_SWA_FULL") && atoi(getenv("IMPULSE_SWA_FULL")) != 0;
    if (H.n_swa > 0 && !swa_ring_ && !reserve && !swa_full) {
        const int32_t need = *std::min_element(positions.begin(), positions.end()) - H.n_swa + 1;   // the oldest position any row sees
        if (need > 0) {
            uint32_t lo = n_kv;
            for (int q = 0; q < ns; ++q) {
                const int32_t * p = kv_.pos.data() + (size_t) kv_.stream_of(head + q) * kv_.ssize;
                uint32_t j = 0;
                while (j < n_kv && p[j] < need) ++j;
                lo = std::min(lo, j);
            }
            swa_lo = lo / 256 * 256;
            if (swa_lo >= n_kv) swa_lo = 0;
        }
    }
    // rs_z: the first new cell of the ubatch (llama_memory_recurrent::find_slot): cleared, and read by every new cell
    int32_t rs_zero = -1;
    if (!reserve) for (int q = 0; q < ns && rs_zero < 0; ++q) if (fresh_v_[head + q]) rs_zero = head + q;
    if (st_replay_ && !reserve) rs_zero = st_replay_->rs_zero;
    if (st_rec_ && !reserve) st_rec_->rs_zero = rs_zero;
    const int n_outputs = reserve ? reserve_outputs : n_out;

    // graph reuse: the same shape as the last computed graph and nothing moved under it, so the graph and its allocation
    // stand and only the inputs change (llama_context::process_ubatch); a reserve always builds, and drops the kept graph
    // qwen4exp with the drafter: every row's 4-stream residual leaves the graph (llama_set_embeddings_nextn unmasked: the last
    // layer keeps every row, the output rows are gathered before the final mixer)
    const bool wide_h = H.q4x && want_h_ && !reserve;
    const bool staged_build = !reserve && st_l1_ >= 0;
    const GraphKey key = { n, (int) n_kv, n_outputs, embd_in_ != nullptr, rs_zero, ns, head, L0, L1, wide_h, (int) swa_lo, (int) n_kv_s };
    const bool reuse = !reserve && g_.valid && g_.key == key && graph_reuse_on();
    uint64_t acc_inc = 0, hit_inc = 0;   // routed pairs the shape alone decides (the whole-layer and all-pool paths)
    const int64_t n_embd = H.n_embd;
    if (!reuse) {   // ---- build (to "// ---- build ends")
    graph_drop(g_);
    ggml_init_params ip = { graph_buf_.size(), graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(sched_);
    const int64_t K = (int64_t) o_.n_rs_seq + 1;   // recurrent snapshot slots
    auto nm = [&](ggml_tensor * t, const char * s, int il) { if (il >= 0) ggml_format_name(t, "%s-%d", s, il); else ggml_set_name(t, s); };

    // build_inp_embd (token ids; the raw-embedding branch of ggml_build_forward_select is kept for the graph's shape)
    ggml_tensor * inp_tokens = nullptr, * inp_embd = nullptr, * inpL = nullptr, * st_in = nullptr;
    if (L0 == 0) {
    inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(inp_tokens, "inp_tokens"); ggml_set_input(inp_tokens);
    inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
    ggml_set_name(inp_embd, "inp_embd"); ggml_set_input(inp_embd);
    ggml_tensor * inps[2] = { ggml_get_rows(ctx, tok_embd_, inp_tokens), inp_embd };
    inpL = ggml_build_forward_select(gf, inps, 2, embd_in_ ? 1 : 0);
    ggml_set_name(inpL, "model.input_embed");
    ggml_build_forward_expand(gf, inpL);
    } else {   // a later stage of a pipelined read: the residual the previous card handed over (qwen4exp: 4 streams wide)
        st_in = H.q4x ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, H.hc, n) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
        ggml_set_name(st_in, "stage_in"); ggml_set_input(st_in);
        if (!H.q4x) inpL = st_in;
    }

    // build_inp_mem_hybrid: attention inputs (k/v idxs, the causal mask) + recurrent inputs (s_copy: cell 0)
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nt, 1, ns);   // [n_kv, n_tps, 1, n_stream]
    ggml_set_input(kq_mask); ggml_set_name(kq_mask, "attn_inp_kq_mask");
    // flint: the sliding layers' mask over the same cells (llama.cpp's --swa-full shape: one cache, the window in the mask)
    ggml_tensor * kq_mask_swa = nullptr;
    if (H.n_swa > 0) { kq_mask_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, swa_ring_ ? n_kv_s : n_kv - swa_lo, nt, 1, ns); ggml_set_input(kq_mask_swa); ggml_set_name(kq_mask_swa, "attn_inp_kq_mask_swa"); }
    // lfm3: the sliding layers write their own cache's cells
    ggml_tensor * k_idxs_s = nullptr, * v_idxs_s = nullptr;
    if (swa_ring_) {
        k_idxs_s = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs_s);
        v_idxs_s = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs_s);
    }
    // build_attn_inp_kv_impl: the rotations of a quantized cache (head size a multiple of 64)
    ggml_tensor * k_rot = nullptr, * v_rot = nullptr;
    if (ggml_is_quantized(o_.type_k) && H.head_dim % 64 == 0) {
        const int r = k_rot_size(H.head_dim);
        k_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(k_rot); ggml_set_name(k_rot, "attn_inp_k_rot");
    }
    if (ggml_is_quantized(o_.type_v) && H.head_dim % 64 == 0) {
        v_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(v_rot); ggml_set_name(v_rot, "attn_inp_v_rot");
    }
    // build_rs_inp_impl: s_copy [n_rs], its main view [n_seqs] and extra view [n_rs - n_seqs] (empty for one sequence)
    ggml_tensor * s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ns);
    ggml_set_input(s_copy);
    ggml_tensor * s_copy_main  = ggml_view_1d(ctx, s_copy, ns, 0);
    ggml_tensor * s_copy_extra = ggml_view_1d(ctx, s_copy, 0, ns * s_copy->nb[0]);   // no extra cells: the ubatch's cells are consecutive
    // build_inp_pos: IMROPE takes 4 positions per token (text: the same position three times, then 0)
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * (H.pos1() ? 1 : 4));
    ggml_set_input(inp_pos);
    ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp_out_ids);

    const int64_t head_k_dim = H.ssm_d_state, num_k_heads = H.ssm_n_group, num_v_heads = H.ssm_dt_rank;
    const int64_t head_v_dim = num_v_heads ? H.ssm_d_inner / num_v_heads : 0;
    const int64_t conv_channels = H.ssm_d_inner + 2 * H.ssm_n_group * H.ssm_d_state;
    const int64_t n_embd_head = H.head_dim, n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    int sections[4]; std::copy(H.rope_sections, H.rope_sections + 4, sections);

    // llm_graph_context::build_rs, node for node (the clear and the extra copy are empty views for one live cell, but
    // they sit between the nodes ggml-cuda would otherwise fuse)
    auto build_rs = [&](ggml_tensor * s, int64_t state_size) {
        const int64_t n_rs = ns, n_seqs = ns, rs_head = head;
        ggml_tensor * states = ggml_reshape_2d(ctx, s, state_size, s->ne[1]);
        ggml_tensor * state_zero = ggml_view_1d(ctx, states, state_size * (rs_zero >= 0), rs_zero * states->nb[1] * (rs_zero >= 0));
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, state_zero, 0));
        ggml_tensor * out = ggml_get_rows(ctx, states, s_copy_main);
        ggml_build_forward_expand(gf, out);
        ggml_tensor * states_extra = ggml_get_rows(ctx, states, s_copy_extra);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, states_extra,
            ggml_view_2d(ctx, s, state_size, n_rs - n_seqs, s->nb[1], (rs_head + n_seqs) * s->nb[1])));
        return out;
    };

    // ---- qwen4exp (llama.cpp src/models/qwen4exp.cpp), node for node
    ggml_tensor * ple_rows = nullptr, * ple_emb = nullptr, * res_hc = nullptr;
    ggml_tensor * qsa_k_idxs = nullptr, * cell_blk = nullptr, * blk_cells = nullptr, * blk_pos = nullptr, * qsa_bias = nullptr;
    int qsa_r = 0;
    const int64_t HCD = (int64_t) H.hc * n_embd;
    auto hc_mix = [&](ggml_tensor * x, ggml_tensor * wn, ggml_tensor * wd, ggml_tensor * wu, ggml_tensor * wi, ggml_tensor ** inject, int il) {
        return hc_mix_build(ctx, H, x, wn, wd, wu, wi, inject, il);
    };
    auto hc_combine = [&](ggml_tensor * res, ggml_tensor * out, ggml_tensor * inject, int il) { return hc_combine_build(ctx, H, res, out, inject, il); };
    // build_conv_state_at: a conv history out of its own recurrent row, the new tail written back once per rollback slot
    // (slot s ends s tokens earlier); x [channels, nt, ns]
    auto conv_state_at = [&](ggml_tensor * states_all, ggml_tensor * x, int64_t cols, int64_t channels, int il) {
        const int64_t row_total = states_all->ne[0];
        ggml_tensor * rows = build_rs(states_all, row_total);
        ggml_tensor * state = ggml_reshape_3d(ctx, rows, cols, channels, ns);
        nm(state, "conv_state_at", il);
        ggml_tensor * conv_input = ggml_concat(ctx, state, ggml_transpose(ctx, x), 0);
        const size_t row_size = ggml_row_size(states_all->type, row_total);
        for (int64_t slot = 0; slot < K; ++slot) {
            const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - cols - slot);
            ggml_tensor * tail = ggml_view_3d(ctx, conv_input, cols, channels, ns, conv_input->nb[1], conv_input->nb[2],
                                              ggml_row_size(conv_input->type, s_idx));
            ggml_tensor * dst = ggml_view_2d(ctx, states_all, cols * channels, ns, states_all->nb[1], (size_t) (slot * NS + head) * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), dst));
        }
        return conv_input;
    };
    // build_ple: the n-gram embedding gate on the wide residual, then a causal depthwise conv (dilation = the n-gram size)
    auto build_ple = [&](ggml_tensor * emb, ggml_tensor * hidden, int il) {
        const Layer & l = L_[il];
        ggml_tensor * key = ggml_mul_mat(ctx, l.ple_key, emb);
        ggml_tensor * value = ggml_mul_mat(ctx, l.ple_value, emb);
        auto gnorm = [&](ggml_tensor * x, ggml_tensor * w) {
            ggml_tensor * t = ggml_reshape_3d(ctx, x, n_embd, H.hc, n);
            t = ggml_rms_norm(ctx, t, H.eps);
            t = ggml_reshape_2d(ctx, t, HCD, n);
            t = ggml_mul(ctx, t, w);
            return ggml_reshape_3d(ctx, t, n_embd, H.hc, n);
        };
        key = gnorm(key, l.ple_nk);
        ggml_tensor * query = gnorm(hidden, l.ple_nq);
        ggml_tensor * sc = ggml_sum_rows(ctx, ggml_mul(ctx, key, query));
        sc = ggml_scale(ctx, sc, 1.0f / sqrtf((float) n_embd));
        ggml_tensor * mag = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, sc), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, sc), mag));
        nm(gate, "ple_gate", il);
        ggml_tensor * v3 = ggml_reshape_3d(ctx, value, n_embd, 1, n);
        v3 = ggml_repeat_4d(ctx, v3, n_embd, H.hc, n, 1);
        ggml_tensor * gated = ggml_mul(ctx, v3, gate);
        nm(gated, "ple_gated_value", il);
        ggml_tensor * normalized = gnorm(ggml_reshape_2d(ctx, gated, HCD, n), l.ple_nc);
        normalized = ggml_reshape_2d(ctx, normalized, HCD, n);
        const int64_t kern = H.ple_kernel, dil = H.ple_ngram, hist = (kern - 1) * dil;
        ggml_tensor * padded = conv_state_at(p_l_[il], ggml_reshape_3d(ctx, normalized, HCD, nt, ns), hist, HCD, il);
        ggml_tensor * conv_out = nullptr;
        for (int64_t k = 0; k < kern; ++k) {
            const int64_t start = hist - (kern - 1 - k) * dil;
            ggml_tensor * shifted = ggml_cont(ctx, ggml_transpose(ctx,
                ggml_view_3d(ctx, padded, nt, HCD, ns, padded->nb[1], padded->nb[2], ggml_row_size(padded->type, start))));
            ggml_tensor * wk = ggml_cont(ctx, ggml_view_2d(ctx, l.ple_conv, 1, HCD, l.ple_conv->nb[1], k * l.ple_conv->nb[0]));
            wk = ggml_reshape_1d(ctx, wk, HCD);
            if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx, wk, GGML_TYPE_F32);
            ggml_tensor * term = ggml_mul(ctx, shifted, wk);
            conv_out = conv_out ? ggml_add(ctx, conv_out, term) : term;
        }
        conv_out = ggml_silu(ctx, conv_out);
        conv_out = ggml_reshape_3d(ctx, ggml_cont(ctx, conv_out), n_embd, H.hc, n);
        nm(conv_out, "ple_conv_out", il);
        return ggml_add(ctx, hidden, ggml_add(ctx, gated, conv_out));
    };
    // build_qsa_top_k: the indexer's raw keys cached, mean-pooled per block of r cells, normed, roped at the block's first
    // position; per query the relu'd head scores summed, biased per block, spread over the cells, + the causal mask; top-k
    auto qsa_prep = [&](ggml_tensor * cur, int il, ggml_tensor ** pooled_out, ggml_tensor ** q_out) -> bool {
        const Layer & l = L_[il];
        const int64_t r = H.compress[il], D = H.idx_dim, nh = H.idx_n_head, n_blocks = ((int64_t) n_kv + r - 1) / r;
        if (!cell_blk) {   // one input set for the layers of this ratio (llama.cpp's qsa_inps)
            qsa_r = (int) r;
            qsa_k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(qsa_k_idxs);
            cell_blk  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, ns);
            blk_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r * n_blocks, ns);
            blk_pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_blocks * ns);
            qsa_bias  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, nt, ns);
            ggml_set_input(cell_blk); ggml_set_input(blk_cells); ggml_set_input(blk_pos); ggml_set_input(qsa_bias);
        } else if (r != qsa_r) return false;   // one ratio
        ggml_tensor * k_raw = ggml_mul_mat(ctx, l.idx_k, cur);
        k_raw = ggml_reshape_3d(ctx, k_raw, D, 1, n);
        nm(k_raw, "indexer_k_raw", il);
        ggml_tensor * Ki = idxk_[il];
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, Ki, ggml_view_2d(ctx, k_raw, D, n, k_raw->nb[2], 0), qsa_k_idxs));
        ggml_tensor * k_all = ggml_view_4d(ctx, Ki, D, 1, n_kv, ns, ggml_row_size(Ki->type, D), ggml_row_size(Ki->type, D),
                                           ggml_row_size(Ki->type, D * kv_.ssize), (size_t) st0 * ggml_row_size(Ki->type, D * kv_.ssize));
        k_all = ggml_view_3d(ctx, k_all, D, n_kv, ns, k_all->nb[2], k_all->nb[3], 0);
        ggml_tensor * members = ggml_get_rows(ctx, k_all, blk_cells);
        members = ggml_reshape_4d(ctx, members, D, r, n_blocks, ns);
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx, ggml_view_3d(ctx, members, D, n_blocks, ns, members->nb[2], members->nb[3], i * members->nb[1]));
            pooled = pooled ? ggml_add(ctx, pooled, slice) : slice;
        }
        pooled = ggml_scale(ctx, pooled, 1.0f / (float) r);
        nm(pooled, "indexer_k_pooled", il);
        pooled = ggml_reshape_3d(ctx, pooled, D, n_blocks * ns, 1);
        pooled = norm(ctx, pooled, l.idx_kn, il);
        pooled = ggml_reshape_3d(ctx, pooled, D, 1, n_blocks * ns);
        pooled = ggml_rope_multi(ctx, pooled, blk_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                 kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        pooled = ggml_reshape_3d(ctx, pooled, D, n_blocks, ns);
        nm(pooled, "indexer_k", il);
        ggml_tensor * q = ggml_mul_mat(ctx, l.idx_q, cur);
        q = ggml_reshape_3d(ctx, q, D, nh, n);
        q = norm(ctx, q, l.idx_qn, il);
        q = ggml_rope_multi(ctx, q, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                            kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        nm(q, "indexer_q", il);
        *pooled_out = pooled; *q_out = q;
        return true;
    };
    // the queries [a, a + len) of the ubatch (all of them: llama.cpp's graph; a slice: one stream, a long read's chunk)
    auto qsa_pick = [&](ggml_tensor * pooled, ggml_tensor * q, int64_t a, int64_t len, int il) -> ggml_tensor * {
        const int64_t r = H.compress[il], D = H.idx_dim, nh = H.idx_n_head, n_blocks = ((int64_t) n_kv + r - 1) / r;
        const bool all = a == 0 && len == nt;
        const int64_t L = all ? nt : len, S = all ? ns : 1;
        ggml_tensor * qv = all ? ggml_reshape_3d(ctx, q, D, nh * nt, ns) : ggml_view_2d(ctx, q, D, nh * len, q->nb[1], a * q->nb[2]);
        ggml_tensor * score = ggml_mul_mat(ctx, pooled, qv);
        score = ggml_reshape_4d(ctx, score, n_blocks, nh, L, S);
        score = ggml_relu(ctx, score);
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < nh; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx, score, n_blocks, L, S, score->nb[2], score->nb[3], h * score->nb[1]);
            summed = summed ? ggml_add(ctx, summed, slice) : ggml_cont(ctx, slice);
        }
        score = summed;
        nm(score, "indexer_score", il);
        ggml_tensor * bias = all ? qsa_bias : ggml_view_3d(ctx, qsa_bias, n_blocks, len, 1, qsa_bias->nb[1], qsa_bias->nb[2], a * qsa_bias->nb[1]);
        score = ggml_add(ctx, score, bias);   // blk_bias: causal, the mask [n_kv, n_tps, 1, n_stream]
        ggml_tensor * expanded = ggml_get_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)), cell_blk);
        expanded = ggml_cont(ctx, ggml_permute(ctx, expanded, 1, 0, 2, 3));
        ggml_tensor * km = all ? kq_mask : ggml_view_4d(ctx, kq_mask, n_kv, len, 1, 1, kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], a * kq_mask->nb[1]);
        ggml_tensor * mask = ggml_cast(ctx, km, GGML_TYPE_F32);
        expanded = ggml_add(ctx, expanded, ggml_reshape_3d(ctx, mask, n_kv, L, S));
        nm(expanded, "indexer_score_tokens", il);
        const int64_t width = std::min<int64_t>(n_kv, (int64_t) H.idx_top_k + r - 1);
        ggml_tensor * top = ggml_cont(ctx, ggml_top_k(ctx, expanded, (int) width));
        top = ggml_reshape_4d(ctx, top, width, L, 1, S);
        nm(top, "indexer_top_k", il);
        return top;
    };
    // build_attn_qsa's mask: -inf, the picked cells unmasked, + the causal mask
    auto qsa_mask = [&](ggml_tensor * top_k, ggml_tensor * km) {
        ggml_tensor * m = ggml_fill(ctx, km, -INFINITY);
        m = ggml_view_4d(ctx, m, 1, m->ne[0], m->ne[1], m->ne[3], m->nb[0], m->nb[1], m->nb[2], 0);
        ggml_tensor * t3 = ggml_view_4d(ctx, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2],
                                        top_k->ne[3] * top_k->nb[3], 0);
        ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, t3->ne[0], t3->ne[1], t3->ne[2]);
        z = ggml_fill(ctx, z, 0.0f);
        m = ggml_set_rows(ctx, m, z, t3);
        m = ggml_view_4d(ctx, m, m->ne[1], m->ne[2], 1, m->ne[3], m->nb[2], m->nb[3], m->nb[3], 0);
        return ggml_add(ctx, m, km);
    };
    // the indexer's scores, top-k, mask and attention per slice of qsa_slice queries when the whole ubatch's scratch
    // (~40 bytes per query and cell) would pass kQsaScratch: long contexts and big read chunks. Below it the graph is
    // llama.cpp's. One stream (the reserve graphs too, so the reserved buffers are the sliced ones).
    const int64_t qsa_slice = [&]() -> int64_t {
        if (!H.q4x || ns != 1) return 0;
        if (const char * e = getenv("IMPULSE_QSA_SLICE")) return std::min<int64_t>(n, std::max(0, atoi(e)));   // 0: off (test)
        const int64_t per = (int64_t) n_kv * 40;   // bytes of scratch per query (scores, mask casts, sort buffers)
        if ((int64_t) n * per <= (int64_t) kQsaScratch) return 0;
        return std::max<int64_t>(32, (int64_t) kQsaScratch / per / 32 * 32);
    }();
    if (H.q4x) {
        if (ple_tbl_ && H.ple_layer >= L0 && H.ple_layer < L1) {   // build_inp_ple, expanded right after the embedding's inputs (one split with the embedding)
            ple_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) H.ple_n_heads * n);
            ggml_set_input(ple_rows); ggml_set_name(ple_rows, "ple_rows");
            ple_emb = ggml_get_rows(ctx, ple_tbl_, ple_rows);
            ple_emb = ggml_reshape_2d(ctx, ple_emb, (int64_t) H.ple_head_dim * H.ple_n_heads, n);
            ggml_set_name(ple_emb, "ple_embd");
            ggml_build_forward_expand(gf, ple_emb);
        }
        if (st_in) res_hc = st_in;
        else {
        res_hc = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, inpL, n_embd, 1, n), n_embd, H.hc, n, 1);
        ggml_set_name(res_hc, "hc_init");
        }
    }

    std::vector<ggml_tensor *> adapt_ids(o_.adapt ? H.n_layer : 0, nullptr);   // stage e: each cached layer's routing
    std::shared_ptr<RingUse> ring_use;
    for (int il = L0; il < L1; ++il) {
        const Layer & l = L_[il];
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = nullptr, * inject = nullptr;
        if (H.q4x) {   // PLE on the wide residual before the mixer, then the hc mixer in place of attn_norm
            if (il == H.ple_layer) res_hc = build_ple(ple_emb, res_hc, il);
            cur = hc_mix(res_hc, l.hc_an, l.hc_ad, l.hc_au, l.hc_ai, &inject, il);
        } else {
            cur = norm(ctx, inpL, l.attn_norm, il);
            nm(cur, "attn_norm", il);
        }
        ggml_build_forward_expand(gf, cur);

        if (H.is_recr[il] && H.lfm2) {
            // llama.cpp lfm2.cpp: B,C,X -> causal depthwise conv(B*X) -> C gating -> output.
            // {n_embd, n} => {n_embd, n_seq_tokens, n_seqs}; each sequence's conv state in its cell's rows (slot * n_seq + cell)
            auto * conv_x = ggml_reshape_3d(ctx, cur, n_embd, nt, ns);
            auto * bcx = ggml_mul_mat(ctx, l.conv_in, conv_x);
            if (lora_) bcx = lora_->add(ctx, l.conv_in, conv_x, bcx);
            auto chunk = [&](int i) { return ggml_view_3d(ctx, bcx, n_embd, nt, ns, bcx->nb[1], bcx->nb[2],
                                                       i * n_embd * ggml_element_size(bcx)); };
            auto * b = chunk(0); auto * c = chunk(1); auto * x = chunk(2);
            auto * bx = ggml_transpose(ctx, ggml_mul(ctx, b, x));
            auto * state = ggml_reshape_3d(ctx, build_rs(r_l_[il], H.n_embd_r()), H.ssm_d_conv - 1, n_embd, ns);
            bx = ggml_concat(ctx, state, bx, 0);
            const int64_t nw = std::min<int64_t>(nt, K), dc = H.ssm_d_conv - 1;
            for (int64_t slot = 0; slot < nw; ++slot) {
                auto * snap = ggml_view_3d(ctx, bx, dc, n_embd, ns, bx->nb[1], bx->nb[2],
                                          (bx->ne[0] - dc - slot) * ggml_element_size(bx));
                auto * dst = ggml_view_2d(ctx, r_l_[il], dc * n_embd, ns, r_l_[il]->nb[1], (size_t) (slot * NS + head) * r_l_[il]->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, snap, dst));
            }
            auto * conv_hidden = ggml_mul(ctx, c, ggml_ssm_conv(ctx, bx, l.conv_kernel));
            cur = ggml_mul_mat(ctx, l.conv_out, conv_hidden);
            if (lora_) cur = lora_->add(ctx, l.conv_out, conv_hidden, cur);
            cur = ggml_reshape_2d(ctx, cur, n_embd, n);
            nm(cur, "shortconv_out", il);
        } else if (H.is_recr[il]) {
            // ---- build_layer_attn_linear (gated delta net)
            ggml_tensor * qkv_mixed = mm_s(ctx, ggml_mul_mat(ctx, l.wqkv, cur), l.wqkv_s);
            qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], nt, ns);
            nm(qkv_mixed, "linear_attn_qkv_mixed", il);
            ggml_tensor * z = mm_s(ctx, ggml_mul_mat(ctx, l.wqkv_gate, cur), l.wqkv_gate_s);
            nm(z, "z", il);
            ggml_tensor * beta = mm_s(ctx, ggml_mul_mat(ctx, l.ssm_beta, cur), l.ssm_beta_s);
            beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, nt, ns);
            nm(beta, "beta", il);
            beta = ggml_sigmoid(ctx, beta);
            nm(beta, "beta_sigmoid", il);
            ggml_tensor * alpha = mm_s(ctx, ggml_mul_mat(ctx, l.ssm_alpha, cur), l.ssm_alpha_s);
            alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, nt, ns);
            nm(alpha, "alpha", il);
            ggml_tensor * alpha_biased = ggml_add(ctx, alpha, l.ssm_dt);
            ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha_biased);
            nm(alpha_softplus, "a_softplus", il);
            ggml_tensor * gate = ggml_mul(ctx, alpha_softplus, l.ssm_a);   // -A_log.exp() * softplus
            nm(gate, "gate", il);
            gate = ggml_reshape_4d(ctx, gate, 1, num_v_heads, nt, ns);

            ggml_tensor * conv_input = nullptr;
            if (H.q4x) conv_input = conv_state_at(r_l_[il], qkv_mixed, H.ssm_d_conv - 1, conv_channels, il);
            else {
            // build_conv_state
            ggml_tensor * conv_states = build_rs(r_l_[il], H.n_embd_r());
            nm(conv_states, "conv_states", il);
            conv_states = ggml_reshape_3d(ctx, conv_states, H.ssm_d_conv - 1, conv_channels, ns);
            nm(conv_states, "conv_states_reshaped", il);
            ggml_tensor * qkv_t = ggml_transpose(ctx, qkv_mixed);
            nm(qkv_t, "qkv_mixed_transposed", il);
            conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
            nm(conv_input, "conv_input", il);
            if (o_.n_rs_seq > 0) {   // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K conv windows, slot K - t
                const int64_t row_count = (int64_t) (H.ssm_d_conv - 1) * conv_channels;
                const size_t row_size = ggml_row_size(r_l_[il]->type, row_count);
                for (int64_t t = 1; t <= K; ++t) {
                    const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - conv_states->ne[0] - K + t);
                    const int64_t s_slot = K - t;
                    ggml_tensor * last = ggml_view_3d(ctx, conv_input, H.ssm_d_conv - 1, conv_channels, ns,
                                                      conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, s_idx));
                    ggml_tensor * upd = ggml_view_2d(ctx, r_l_[il], row_count, ns, r_l_[il]->nb[1], (size_t) (s_slot * NS + head) * row_size);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
                }
            } else {
                const int64_t row_count = (int64_t) (H.ssm_d_conv - 1) * conv_channels;
                const size_t row_size = ggml_row_size(r_l_[il]->type, row_count);
                const int64_t s_idx = conv_input->ne[0] - conv_states->ne[0];
                ggml_tensor * last = ggml_view_3d(ctx, conv_input, H.ssm_d_conv - 1, conv_channels, ns,
                                                  conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, s_idx));
                nm(last, "conv_state_last", il);
                ggml_tensor * upd = ggml_view_2d(ctx, r_l_[il], row_count, ns, r_l_[il]->nb[1], (size_t) head * row_size);
                nm(upd, "conv_state_update", il);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
            }
            }

            ggml_tensor * state = build_rs(s_l_[il], H.n_embd_s());
            state = ggml_reshape_4d(ctx, state, head_v_dim, head_v_dim, num_v_heads, ns);
            nm(state, "state_predelta", il);

            ggml_tensor * conv_out = ggml_ssm_conv(ctx, conv_input, l.ssm_conv1d);
            nm(conv_out, "conv_output_raw", il);
            ggml_tensor * conv_silu = ggml_silu(ctx, conv_out);
            nm(conv_silu, "conv_output_silu", il);
            const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
            const size_t nb1_qkv = ggml_row_size(conv_silu->type, qkv_dim);
            ggml_tensor * q_conv = ggml_view_4d(ctx, conv_silu, head_k_dim, num_k_heads, nt, ns,
                                                ggml_row_size(conv_silu->type, head_k_dim), nb1_qkv, nb1_qkv * nt, 0);
            ggml_tensor * k_conv = ggml_view_4d(ctx, conv_silu, head_k_dim, num_k_heads, nt, ns,
                                                ggml_row_size(conv_silu->type, head_k_dim), nb1_qkv, nb1_qkv * nt,
                                                head_k_dim * num_k_heads * ggml_element_size(conv_silu));
            ggml_tensor * v_conv = ggml_view_4d(ctx, conv_silu, head_v_dim, num_v_heads, nt, ns,
                                                ggml_row_size(conv_silu->type, head_v_dim), nb1_qkv, nb1_qkv * nt,
                                                ggml_row_size(conv_silu->type, 2 * head_k_dim * num_k_heads));
            nm(q_conv, "q_conv", il); nm(k_conv, "k_conv", il); nm(v_conv, "v_conv", il);
            // build_gdn_l2_norm
            auto l2 = [&](ggml_tensor * x) {
                const float nx = (float) x->ne[0];
                return ggml_scale(ctx, ggml_rms_norm(ctx, x, H.eps / nx), 1.0f / sqrtf(nx));
            };
            q_conv = l2(q_conv);
            k_conv = l2(k_conv);
            nm(q_conv, "q_conv_predelta", il); nm(k_conv, "k_conv_predelta", il); nm(v_conv, "v_conv_predelta", il);

            // build_recurrent_attn -> build_delta_net_fused (cparams.fused_gdn_ar / _ch); with n_rs_seq the op writes K
            // snapshots (slot 0 = the newest state, slot s = s tokens back; the last min(n, K) are written)
            ggml_tensor * gdn = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, gate, beta, state, K);
            ggml_tensor * out = ggml_view_4d(ctx, gdn, head_v_dim, num_v_heads, nt, ns,
                                             ggml_row_size(gdn->type, head_v_dim), ggml_row_size(gdn->type, head_v_dim * num_v_heads),
                                             ggml_row_size(gdn->type, head_v_dim * num_v_heads * nt), 0);
            if (o_.n_rs_seq > 0) {
                nm(out, "attn_output", il);
                const int64_t D = head_v_dim * head_v_dim * num_v_heads;
                const int64_t n_written = std::min<int64_t>(nt, K);
                const size_t row_size = H.n_embd_s() * ggml_element_size(s_l_[il]);
                // slot j of ubatch sequence q -> row j * NS + head + q (the op's slots: [S, S, H, ns] each, after the output)
                ggml_tensor * src = ggml_view_3d(ctx, gdn, D, ns, n_written, ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D * ns),
                                                 ggml_row_size(gdn->type, head_v_dim * num_v_heads * n));
                ggml_tensor * dst = ggml_view_3d(ctx, s_l_[il], D, ns, n_written, s_l_[il]->nb[1], (size_t) NS * row_size, (size_t) head * row_size);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
            } else {
            ggml_tensor * new_state = ggml_view_4d(ctx, gdn, head_v_dim, head_v_dim, num_v_heads, ns,
                                                   ggml_row_size(gdn->type, head_v_dim), ggml_row_size(gdn->type, head_v_dim * head_v_dim),
                                                   ggml_row_size(gdn->type, head_v_dim * head_v_dim * num_v_heads),
                                                   ggml_row_size(gdn->type, head_v_dim * num_v_heads * n));
            nm(out, "attn_output", il); nm(new_state, "new_state", il);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state,
                ggml_view_2d(ctx, s_l_[il], H.n_embd_s(), ns, s_l_[il]->nb[1], (size_t) head * s_l_[il]->nb[1])));
            }

            ggml_tensor * z_2d = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, nt, ns);
            // build_norm_gated
            ggml_tensor * normalized = norm(ctx, out, l.ssm_norm, il);
            ggml_tensor * gated_silu = H.q4x ? ggml_sigmoid(ctx, z_2d) : ggml_silu(ctx, z_2d);   // qwen4exp: a sigmoid gate
            ggml_tensor * attn_out_norm = ggml_mul(ctx, normalized, gated_silu);
            ggml_tensor * final_output = ggml_reshape_3d(ctx, attn_out_norm, head_v_dim * num_v_heads, nt, ns);
            nm(final_output, "final_output", il);
            cur = mm_s(ctx, ggml_mul_mat(ctx, l.ssm_out, final_output), l.ssm_out_s);
            nm(cur, "linear_attn_out", il);
            cur = ggml_reshape_2d(ctx, cur, n_embd, n);
        } else {
            // ---- build_layer_attn (gated attention); qwen4exp: the indexer first, then Q (norm), K, V in its order
            ggml_tensor * attn_in = cur;   // lfm3: the head gate reads the block's normed input
            ggml_tensor * top_k = nullptr, * qsa_pooled = nullptr, * qsa_q = nullptr;
            ggml_tensor * Qcur_full, * Kcur, * Vcur, * Qcur;
            if (H.q4x) {
                if (H.compress[il] > 0) {
                    if (!qsa_prep(cur, il, &qsa_pooled, &qsa_q)) { ggml_free(ctx); return "qwen4exp: several QSA ratios"; }
                    if (!qsa_slice || qsa_slice >= n) top_k = qsa_pick(qsa_pooled, qsa_q, 0, nt, il);
                }
                Qcur_full = mm_s(ctx, ggml_mul_mat(ctx, l.wq, cur), l.wq_s);
                nm(Qcur_full, "Qcur_full", il);
                Qcur = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n, ggml_element_size(Qcur_full) * n_embd_head * 2,
                                    ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, 0);
                nm(Qcur, "Qcur_reshaped", il);
                Qcur = norm(ctx, Qcur, l.q_norm, il);
                nm(Qcur, "Qcur_normed", il);
                Kcur = mm_s(ctx, ggml_mul_mat(ctx, l.wk, cur), l.wk_s); nm(Kcur, "Kcur", il);
                Vcur = mm_s(ctx, ggml_mul_mat(ctx, l.wv, cur), l.wv_s); nm(Vcur, "Vcur", il);
            } else {
            Qcur_full = mm_s(ctx, ggml_mul_mat(ctx, l.wq, cur), l.wq_s);   // build_qkv(..., reshape = false): separate Q/K/V
            if (lora_) Qcur_full = lora_->add(ctx, l.wq, cur, Qcur_full);
            Kcur = mm_s(ctx, ggml_mul_mat(ctx, l.wk, cur), l.wk_s);
            if (lora_) Kcur = lora_->add(ctx, l.wk, cur, Kcur);
            Vcur = mm_s(ctx, ggml_mul_mat(ctx, l.wv, cur), l.wv_s);
            if (lora_) Vcur = lora_->add(ctx, l.wv, cur, Vcur);
            nm(Qcur_full, "Qcur_full", il); nm(Kcur, "Kcur", il); nm(Vcur, "Vcur", il);
            Qcur = H.pos1() ? ggml_reshape_3d(ctx, Qcur_full, n_embd_head, H.n_head, n) :
                ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, 0);
            nm(Qcur, "Qcur_reshaped", il);
            Qcur = norm(ctx, Qcur, l.q_norm, il);
            nm(Qcur, "Qcur_normed", il);
            }
            Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, H.n_head_kv, n);
            Kcur = norm(ctx, Kcur, l.k_norm, il);
            nm(Kcur, "Kcur_normed", il);
            ggml_tensor * gate = nullptr;
            if (!H.pos1()) {
            gate = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2,
                                              ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head,
                                              ggml_element_size(Qcur_full) * n_embd_head);
            gate = ggml_cont_2d(ctx, gate, n_embd_head * H.n_head, n);
            nm(gate, "gate_reshaped", il);
            }
            Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, H.n_head_kv, n);
            if (H.flint) {   // flint.cpp: RoPE (NEOX) on the sliding layers only
                if (H.has_rope[il]) {
                    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr, H.n_rot, GGML_ROPE_TYPE_NEOX, H.n_ctx_train, H.rope_base,
                                         1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
                    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr, H.n_rot, GGML_ROPE_TYPE_NEOX, H.n_ctx_train, H.rope_base,
                                         1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
                }
            } else if (H.lfm3) {   // lfm3.cpp: NEOX; sliding layers rope_base_swa over the whole head, global ones rope_base with
                // rope_freqs (its factors leave the tail of each head unrotated: partial_rotary_factor 0.5)
                const bool sw = H.is_swa[il];
                ggml_tensor * ff = sw ? nullptr : rope_freqs_;
                const float base = sw ? H.rope_base_swa : H.rope_base;
                Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, ff, H.n_rot, GGML_ROPE_TYPE_NEOX, H.n_ctx_train, base,
                                     1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
                Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, ff, H.n_rot, GGML_ROPE_TYPE_NEOX, H.n_ctx_train, base,
                                     1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
            } else if (H.lfm2) {
                Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr, H.n_rot, GGML_ROPE_TYPE_NEOX, H.n_ctx_train, H.rope_base,
                                     1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
                Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr, H.n_rot, GGML_ROPE_TYPE_NEOX, H.n_ctx_train, H.rope_base,
                                     1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
            } else {
            Qcur = ggml_rope_multi(ctx, Qcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                   kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
            Kcur = ggml_rope_multi(ctx, Kcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                   kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
            }
            nm(Qcur, "Qcur", il); nm(Kcur, "Kcur", il); nm(Vcur, "Vcur", il);
            const float kq_scale = 1.0f / sqrtf((float) n_embd_head);

            // build_attn (kv): the rotations, cpy_k / cpy_v, then build_attn_mha with flash attention
            const bool ring = swa_layer(il);   // lfm3: a sliding layer in its own cache
            ggml_tensor * Kc = ring ? kvs_.k[il] : reserve ? kfull_[il] : kv_.k[il], * Vc = ring ? kvs_.v[il] : reserve ? vfull_[il] : kv_.v[il];   // the pager: see kfull_
            if (k_rot) { Qcur = mul_mat_hadamard(ctx, Qcur, k_rot); Kcur = mul_mat_hadamard(ctx, Kcur, k_rot); }
            if (v_rot) Vcur = mul_mat_hadamard(ctx, Vcur, v_rot);
            ggml_build_forward_expand(gf, Qcur);   // llm_graph_context::build_attn (kv): q, then v, then k
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, Kcur);
            {
                ggml_tensor * kc = ggml_view_2d(ctx, Kcur, n_gqa, n, Kcur->nb[2], 0);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, Kc, kc, ring ? k_idxs_s : k_idxs));
                ggml_tensor * vc = ggml_view_2d(ctx, Vcur, n_gqa, n, Vcur->nb[2], 0);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx, Vc, vc, ring ? v_idxs_s : v_idxs));
            }
            // a single-token step (every decode step) attends to the picked cells only: their K/V rows gathered, the scores, the
            // softmax under the mask's values at those cells, the weighted sum (Strata's sparse attention; llama.cpp computes the
            // same over every cell with the others masked: equal in exact arithmetic, another summation order).
            // IMPULSE_QSA_DENSE=1 keeps llama.cpp's dense masked attention (the bit-for-bit comparisons)
            static const bool qsa_dense = getenv("IMPULSE_QSA_DENSE") && atoi(getenv("IMPULSE_QSA_DENSE")) != 0;
            const bool sparse = top_k && nt == 1 && !qsa_dense;
            ggml_tensor * att_mask = top_k && !sparse ? qsa_mask(top_k, kq_mask) : (kq_mask_swa && H.is_swa[il]) ? kq_mask_swa : kq_mask;   // QSA: the cells the indexer picked
            // llama_kv_cache::get_k / get_v: [hd, heads, n_kv, ns] over the ubatch's streams; Q as [hd, heads, n_tps, n_stream]
            const uint32_t c0 = (kq_mask_swa && H.is_swa[il] && !ring) ? swa_lo : 0;   // flint's sliding layers: from swa_lo on
            const uint32_t nkv_l = ring ? n_kv_s : n_kv - c0;
            const uint32_t ss_l = ring ? kvs_.ssize : kv_.ssize;   // cells per stream
            ggml_tensor * k = ggml_view_4d(ctx, Kc, n_embd_head, H.n_head_kv, nkv_l, ns,
                                           ggml_row_size(Kc->type, n_embd_head), ggml_row_size(Kc->type, n_gqa),
                                           ggml_row_size(Kc->type, n_gqa * ss_l), (size_t) st0 * ggml_row_size(Kc->type, n_gqa * ss_l) + c0 * ggml_row_size(Kc->type, n_gqa));
            ggml_tensor * v = ggml_view_4d(ctx, Vc, n_embd_head, H.n_head_kv, nkv_l, ns,
                                           ggml_row_size(Vc->type, n_embd_head), ggml_row_size(Vc->type, n_gqa),
                                           ggml_row_size(Vc->type, n_gqa * ss_l), (size_t) st0 * ggml_row_size(Vc->type, n_gqa * ss_l) + c0 * ggml_row_size(Vc->type, n_gqa));
            ggml_tensor * q = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], nt, ns, Qcur->nb[1], Qcur->nb[2], Qcur->nb[2] * nt, 0);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
            if (sparse) {
                const int64_t W = top_k->ne[0];
                ggml_tensor * idx = ggml_reshape_2d(ctx, top_k, W, ns);                  // [W, ns]: each stream's picks
                const size_t kso = ggml_row_size(Kc->type, n_gqa * kv_.ssize), vso = ggml_row_size(Vc->type, n_gqa * kv_.ssize);
                ggml_tensor * kr = ggml_view_3d(ctx, Kc, n_gqa, n_kv, ns, ggml_row_size(Kc->type, n_gqa), kso, (size_t) st0 * kso);
                ggml_tensor * vr = ggml_view_3d(ctx, Vc, n_gqa, n_kv, ns, ggml_row_size(Vc->type, n_gqa), vso, (size_t) st0 * vso);
                ggml_tensor * ks = ggml_get_rows(ctx, kr, idx);                          // [n_gqa, W, ns] f32
                ggml_tensor * vs = ggml_get_rows(ctx, vr, idx);
                ks = ggml_permute(ctx, ggml_reshape_4d(ctx, ks, n_embd_head, H.n_head_kv, W, ns), 0, 2, 1, 3);   // [hd, W, hk, ns]
                vs = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, vs, n_embd_head, H.n_head_kv, W, ns), 1, 2, 0, 3));   // [W, hd, hk, ns]
                ggml_tensor * mrow = ggml_view_3d(ctx, kq_mask, 1, n_kv, ns, kq_mask->nb[0], kq_mask->nb[3], 0);   // a cell per row
                ggml_tensor * ms = ggml_reshape_4d(ctx, ggml_get_rows(ctx, mrow, idx), W, 1, 1, ns);        // the mask at the picks
                ggml_tensor * kq = ggml_mul_mat(ctx, ks, q);                             // [W, 1, heads, ns]
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                kq = ggml_soft_max_ext(ctx, kq, ms, kq_scale, 0.0f);
                ggml_tensor * kqv = ggml_mul_mat(ctx, vs, kq);                           // [hd, 1, heads, ns]
                cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), n_embd_head * H.n_head, ns);
                nm(cur, "qsa_sparse", il);
            } else if (qsa_pooled && qsa_slice && qsa_slice < n) {   // slices of the chunk's queries, each with its own picks and mask, joined
                ggml_tensor * joined = nullptr;
                for (int64_t a = 0; a < n; a += qsa_slice) {
                    const int64_t len = std::min<int64_t>(qsa_slice, n - a);
                    ggml_tensor * tk = qsa_pick(qsa_pooled, qsa_q, a, len, il);
                    ggml_tensor * km = ggml_view_4d(ctx, kq_mask, n_kv, len, 1, 1, kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], a * kq_mask->nb[1]);
                    ggml_tensor * qs = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], len, 1, Qcur->nb[1], Qcur->nb[2], Qcur->nb[2] * len, a * Qcur->nb[2]);
                    ggml_tensor * fa = ggml_flash_attn_ext(ctx, ggml_permute(ctx, qs, 0, 2, 1, 3), kp, vp, qsa_mask(tk, km), kq_scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_add_sinks(fa, nullptr);
                    ggml_flash_attn_ext_set_n_kv_max(fa, 0);
                    ggml_prec_set_acc(fa, GGML_PREC_F32);
                    fa = ggml_reshape_2d(ctx, fa, fa->ne[0] * fa->ne[1], fa->ne[2] * fa->ne[3]);
                    joined = joined ? ggml_concat(ctx, joined, fa, 1) : fa;
                }
                cur = joined;
            } else {
            cur = ggml_flash_attn_ext(ctx, q, kp, vp, att_mask, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, l.attn_sinks);   // flint: the learned sinks; nullptr otherwise
            ggml_flash_attn_ext_set_n_kv_max(cur, 0);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
            }
            nm(cur, "kqv_out", il);
            if (v_rot) cur = mul_mat_hadamard(ctx, cur, v_rot);
            nm(cur, "attn_pregate", il);
            if (l.head_gate) {   // lfm3.cpp: sigmoid(attn_in x gate) [n_head, n], one scalar per head
                ggml_tensor * hg = ggml_mul_mat(ctx, l.head_gate, attn_in);
                if (lora_) hg = lora_->add(ctx, l.head_gate, attn_in, hg);
                hg = ggml_sigmoid(ctx, hg);
                nm(hg, "head_gate", il);
                ggml_tensor * a3 = ggml_reshape_3d(ctx, cur, n_embd_head, H.n_head, n);
                a3 = ggml_mul(ctx, a3, ggml_reshape_3d(ctx, hg, 1, H.n_head, n));
                cur = ggml_reshape_2d(ctx, a3, n_embd_head * H.n_head, n);
                nm(cur, "attn_head_gated", il);
            }
            if (gate) {
            ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx, gate);
            nm(gate_sigmoid, "gate_sigmoid", il);
            cur = ggml_mul(ctx, cur, gate_sigmoid);
            nm(cur, "attn_gated", il);
            }
            auto * attn_hidden = cur;
            cur = mm_s(ctx, ggml_mul_mat(ctx, l.wo, attn_hidden), l.wo_s);
            if (lora_) cur = lora_->add(ctx, l.wo, attn_hidden, cur);
            nm(cur, "attn_output", il);
        }

        if (l.mix_post_norm) {   // lfm3: post_operator_norm on the mixer's output, before the residual
            cur = norm(ctx, cur, l.mix_post_norm, il);
            nm(cur, "post_operator_norm", il);
        }
        // LFM selects output rows before the final FFN, as llama.cpp lfm2.cpp does. This also
        // preserves its single-row matvec arithmetic on the final prompt row.
        // llama.cpp's inp_out_ids always exists (build_inp_out_ids: a constant topology for pipeline parallelism): the last layer
        // always selects the output rows, none of them in a prompt piece that outputs nothing
        // qwen4exp selects them there too, after the mixer and before the attention combine (the wide residual with them)
        const bool select_last = (H.lfm2 || H.q4x || H.flint) && il == H.n_layer - 1 && !wide_h;
        if (select_last) {
            cur = ggml_get_rows(ctx, cur, inp_out_ids);
            if (H.q4x) {
                inject = ggml_get_rows(ctx, inject, inp_out_ids);
                res_hc = ggml_reshape_2d(ctx, res_hc, n_embd * H.hc, res_hc->ne[2]);
                res_hc = ggml_get_rows(ctx, res_hc, inp_out_ids);
                res_hc = ggml_reshape_3d(ctx, res_hc, n_embd, H.hc, res_hc->ne[1]);
            } else inpSA = ggml_get_rows(ctx, inpSA, inp_out_ids);
        }
        const int n_ff_tokens = select_last ? n_outputs : n;
        ggml_tensor * ffn_residual = nullptr, * x = nullptr;
        if (H.q4x) {
            res_hc = hc_combine(res_hc, cur, inject, il);
            x = hc_mix(res_hc, l.hc_fn, l.hc_fd, l.hc_fu, l.hc_fi, &inject, il);
        } else {
        cur = H.lfm2 ? ggml_add(ctx, inpSA, cur) : ggml_add(ctx, cur, inpSA);
        nm(cur, "attn_residual", il);
        ffn_residual = cur;
        ggml_tensor * attn_post_norm = norm(ctx, cur, l.attn_post_norm, il);
        nm(attn_post_norm, "attn_post_norm", il);
        x = attn_post_norm;
        }

        // ---- build_layer_ffn: build_moe_ffn(SILU, norm_w, SOFTMAX, separate gate/up) + the gated shared expert; a dense
        // layer (LFM2's leading ones, every qwen35 layer): build_ffn(SILU, PAR), up, gate, swiglu, down
        ggml_tensor * moe_out = nullptr;
        if (il < H.n_dense) {
            auto * up = ggml_mul_mat(ctx, l.dense_up, x);
            if (lora_) up = lora_->add(ctx, l.dense_up, x, up);
            nm(up, "ffn_up", il);
            if (l.dense_up_s) { up = ggml_mul(ctx, up, l.dense_up_s); nm(up, "ffn_up_s", il); }   // build_ffn: after the adapter (no LFM scales)
            auto * gate = ggml_mul_mat(ctx, l.dense_gate, x);
            if (lora_) gate = lora_->add(ctx, l.dense_gate, x, gate);
            nm(gate, "ffn_gate", il);
            if (l.dense_gate_s) { gate = ggml_mul(ctx, gate, l.dense_gate_s); nm(gate, "ffn_gate_s", il); }
            auto * g = ggml_swiglu_split(ctx, gate, up);
            nm(g, "ffn_swiglu", il);
            cur = ggml_mul_mat(ctx, l.dense_down, g);
            if (lora_) cur = lora_->add(ctx, l.dense_down, g, cur);
            nm(cur, "ffn_down", il);
            if (l.dense_down_s) { cur = ggml_mul(ctx, cur, l.dense_down_s); nm(cur, "ffn_down_s", il); }
        } else {
            const int64_t n_expert = H.n_expert, n_used = H.n_expert_used;
            ggml_tensor * logits_r = ggml_mul_mat(ctx, l.gate_inp, x);
            nm(logits_r, "ffn_moe_logits", il);
            ggml_tensor * probs = H.lfm2 || H.flint ? ggml_sigmoid(ctx, logits_r) : ggml_soft_max(ctx, logits_r);
            nm(probs, "ffn_moe_probs", il);
            ggml_tensor * selection_probs = l.expert_bias ? ggml_add(ctx, probs, l.expert_bias) : probs;
            ggml_tensor * selected = ggml_argsort_top_k(ctx, selection_probs, (int) n_used);
            nm(selected->src[0], "ffn_moe_argsort", il);
            nm(selected, "ffn_moe_topk", il);
            probs = ggml_reshape_3d(ctx, probs, 1, n_expert, n_ff_tokens);
            ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
            nm(weights, "ffn_moe_weights", il);
            weights = ggml_reshape_2d(ctx, weights, n_used, n_ff_tokens);
            ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
            nm(wsum, "ffn_moe_weights_sum", il);
            wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
            nm(wsum, "ffn_moe_weights_sum_clamped", il);
            weights = ggml_div(ctx, weights, wsum);
            nm(weights, "ffn_moe_weights_norm", il);
            weights = ggml_reshape_3d(ctx, weights, 1, n_used, n_ff_tokens);
            ggml_build_forward_expand(gf, weights);
            // the experts' second-level scales (gate, up, down; nullptr = none): the layer's, or their copies on the card of a
            // cached layer (its slot paths). They are gathered by the routed expert ids, whatever slots the matmuls read
            ggml_tensor * const ls[3] = { l.gate_exps_s, l.up_exps_s, l.down_exps_s };
            CacheLayer * Cs = cache_.empty() ? nullptr : cache_[il].get();
            ggml_tensor * const * csc = Cs && (Cs->sc[0] || Cs->sc[1] || Cs->sc[2]) ? Cs->sc : ls;
            // up -> gate -> swiglu -> down over expert tensors (gate, up, down) addressed by `ids` (build_moe_ffn's middle);
            // each matmul times its expert's scale (build_lora_mm_id)
            auto mm_path = [&](ggml_tensor * gw, ggml_tensor * uw, ggml_tensor * dw, ggml_tensor * ids, ggml_tensor * const * sc) {
                ggml_tensor * xr = ggml_reshape_3d(ctx, x, n_embd, 1, n_ff_tokens);
                ggml_tensor * up = mm_id_s(ctx, ggml_mul_mat_id(ctx, uw, xr, ids), sc[1], selected, n_ff_tokens);
                nm(up, sc[1] ? "ffn_moe_up_scaled" : "ffn_moe_up", il);
                ggml_tensor * g = mm_id_s(ctx, ggml_mul_mat_id(ctx, gw, xr, ids), sc[0], selected, n_ff_tokens);
                nm(g, sc[0] ? "ffn_moe_gate_scaled" : "ffn_moe_gate", il);
                g = ggml_swiglu_split(ctx, g, up);
                nm(g, "ffn_moe_swiglu", il);
                ggml_tensor * amax_e = nullptr;
                if (H.flint) { auto r = down_amax(ctx, g); g = r.first; amax_e = r.second; }
                auto * result = mm_id_s(ctx, ggml_mul_mat_id(ctx, dw, g, ids), sc[2], selected, n_ff_tokens);
                if (amax_e) result = ggml_mul(ctx, result, amax_e);
                // Slot ids index cached base weights; the adapter always indexes original expert ids.
                // Miss slots have zero gate/up activations, so their GPU LoRA contribution is also zero.
                return lora_ ? lora_->add(ctx, l.down_exps, g, result, selected) : result;
            };
            // llama.cpp schedules these rank-one adapter operations on CUDA even
            // for CPU base experts. Keep the same arithmetic after the pool result.
            auto pool_lora = [&](ggml_tensor * result, PoolLayer * pl) {
                if (!pl->lora_down) return result;
                ggml_tensor * args[1] = { result };
                auto * hidden = ggml_custom_4d(ctx, GGML_TYPE_F32, pl->f.n_ff, n_used, n_ff_tokens, 1,
                                              args, 1, moe_hidden_op, 1, pl);
                return lora_->add(ctx, l.down_exps, hidden, result, selected);
            };
            ggml_tensor * experts;
            PoolLayer * PL = pool_layers_.empty() ? nullptr : pool_layers_[il].get();
            CacheLayer * C = cache_.empty() ? nullptr : cache_[il].get();
            const bool small = n < kOffloadMin; // a lent arena is unavailable throughout a prompt ubatch
            if (o_.adapt && C && !C->full) {   // flagged in the reserve graphs too: the allocator keeps the reserved plan and
                ggml_set_output(selected->src[0]);   // ignores a flag added later, so the ids would be overwritten
                if (!reserve) adapt_ids[il] = selected->src[0];
            }
            if (C && C->full) {                                     // stage d, whole layer: llama.cpp's GPU-layer path
                experts = mm_path(C->slot[0], C->slot[1], C->slot[2], selected, csc);
                if (small) { acc_inc += (uint64_t) n_ff_tokens * n_used; hit_inc += (uint64_t) n_ff_tokens * n_used; }
            } else if (C && small && n_ff_tokens <= kSlotMaxTokens) {         // stage d: hits on the GPU slots, misses in the pool
                ggml_tensor * slot = ggml_reshape_3d(ctx, C->res_f, 1, n_expert, 1);   // per token, as llama gathers weights
                slot = ggml_repeat_4d(ctx, slot, 1, n_expert, n_ff_tokens, 1);
                slot = ggml_get_rows(ctx, slot, selected);                          // [1, k, n_ff_tokens]: slot, or S (zeros)
                slot = ggml_reshape_2d(ctx, slot, n_used, n_ff_tokens);
                ggml_tensor * slot_ids = ggml_cast(ctx, slot, GGML_TYPE_I32);
                nm(slot_ids, "ffn_moe_slots", il);
                ggml_tensor * args[2] = { x, selected };
                ggml_tensor * eg, * ec;
                if (o_.overlap) {   // start (CPU) -> hits (GPU, launched async) -> join (CPU, waits on the pool) -> add
                    ggml_tensor * st = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, moe_start_op, 1, C->miss.get());
                    nm(st, "ffn_moe_start", il);
                    ggml_build_forward_expand(gf, st);
                    eg = mm_path(C->slot[0], C->slot[1], C->slot[2], slot_ids, csc);
                    ggml_build_forward_expand(gf, eg);
                    ggml_tensor * jargs[1] = { st };
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n_ff_tokens, 1, jargs, 1, moe_join_op, 1, C->miss.get());
                } else {
                    eg = mm_path(C->slot[0], C->slot[1], C->slot[2], slot_ids, csc);   // a miss: zero rows, times its scale
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n_ff_tokens, 1, args, 2, moe_pool_op, 1, C->miss.get());
                }
                ec = pool_lora(ec, C->miss.get());
                nm(ec, "ffn_moe_misses", il);
                experts = ggml_add(ctx, eg, ec);                    // each row: its GPU value + 0, or 0 + its CPU value (each scaled)
                if (!C->all.empty()) {                              // dcheck twins: the whole layer both ways
                    ggml_tensor * tg = mm_path(C->all[0], C->all[1], C->all[2], selected, csc);
                    ggml_tensor * tc = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n_ff_tokens, 1, args, 2, moe_pool_op, 1, PL);
                    tc = pool_lora(tc, PL);
                    ggml_set_output(tg); ggml_set_output(tc); ggml_set_output(experts); ggml_set_output(selected->src[0]);
                    ggml_build_forward_expand(gf, tg); ggml_build_forward_expand(gf, tc);
                    dcheck_ = { experts, tg, tc, selected->src[0], C };
                }
            } else if (PL && small) {                               // stage c: all of the layer's experts in the pool
                ggml_tensor * args[2] = { x, selected };
                experts = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n_ff_tokens, 1, args, 2, moe_pool_op, 1, PL);
                experts = pool_lora(experts, PL);
                if (C) acc_inc += (uint64_t) n_ff_tokens * n_used;
            } else if (Ring * RG = (!reserve && !small) ? ring_at(dev_layer_[il]) : nullptr; RG && RG->view[il][0]) {   // the read's ring
                if (!ring_use) { ring_use = std::make_shared<RingUse>(); ring_use->eng = this; }
                ring_use->use[selected->src[0]] = { RG->card, -1 };   // the router's sort: the attention runs before the first wait
                ggml_tensor * xr = ggml_reshape_3d(ctx, x, n_embd, 1, n_ff_tokens);
                ggml_tensor * up = ggml_mul_mat_id(ctx, RG->view[il][0], xr, selected);
                ring_use->use[up] = { RG->card, RG->pos[il][0] };   // the matmuls read the ring slots (their scales follow)
                up = mm_id_s(ctx, up, csc[1], selected, n_ff_tokens);
                nm(up, csc[1] ? "ffn_moe_up_scaled" : "ffn_moe_up", il);
                ggml_tensor * g = ggml_mul_mat_id(ctx, RG->view[il][1], xr, selected);
                ring_use->use[g] = { RG->card, RG->pos[il][1] };
                g = mm_id_s(ctx, g, csc[0], selected, n_ff_tokens);
                nm(g, csc[0] ? "ffn_moe_gate_scaled" : "ffn_moe_gate", il);
                ggml_tensor * sw = ggml_swiglu_split(ctx, g, up);
                nm(sw, "ffn_moe_swiglu", il);
                ggml_tensor * amax_e = nullptr;
                if (H.flint) { auto r = down_amax(ctx, sw); sw = r.first; amax_e = r.second; }
                experts = ggml_mul_mat_id(ctx, RG->view[il][2], sw, selected);
                ring_use->use[experts] = { RG->card, RG->pos[il][2] };
                experts = mm_id_s(ctx, experts, csc[2], selected, n_ff_tokens);
                if (amax_e) experts = ggml_mul(ctx, experts, amax_e);
            } else {
                experts = mm_path(l.gate_exps, l.up_exps, l.down_exps, selected, ls);
            }
            nm(experts, "ffn_moe_down", il);
            experts = ggml_mul(ctx, experts, weights);
            nm(experts, "ffn_moe_weighted", il);
            ggml_build_forward_expand(gf, experts);
            ggml_tensor * parts[64];
            for (int i = 0; i < n_used; ++i) {
                parts[i] = ggml_view_2d(ctx, experts, n_embd, n_ff_tokens, experts->nb[2], i * experts->nb[1]);
                ggml_build_forward_expand(gf, parts[i]);
            }
            moe_out = parts[0];
            for (int i = 1; i < n_used; ++i) {
                moe_out = ggml_add(ctx, moe_out, parts[i]);
                ggml_build_forward_expand(gf, moe_out);
            }
            nm(moe_out, "ffn_moe_out", il);
        }
        if (H.lfm2 || H.dense || (H.flint && il < H.n_dense)) { if (moe_out) cur = moe_out; if (H.dense || H.flint) nm(cur, "ffn_out", il); }
        else {   // build_ffn(SILU, PAR): up, gate, swiglu, down
            ggml_tensor * tmp = ggml_mul_mat(ctx, l.up_shexp, x);
            nm(tmp, "ffn_up", il);
            if (l.up_shexp_s) { tmp = ggml_mul(ctx, tmp, l.up_shexp_s); nm(tmp, "ffn_up_s", il); }
            ggml_tensor * g = ggml_mul_mat(ctx, l.gate_shexp, x);
            nm(g, "ffn_gate", il);
            if (l.gate_shexp_s) { g = ggml_mul(ctx, g, l.gate_shexp_s); nm(g, "ffn_gate_s", il); }
            g = ggml_swiglu_split(ctx, g, tmp);
            nm(g, "ffn_swiglu", il);
            ggml_tensor * shexp = ggml_mul_mat(ctx, l.down_shexp, g);
            nm(shexp, "ffn_down", il);
            if (l.down_shexp_s) { shexp = ggml_mul(ctx, shexp, l.down_shexp_s); nm(shexp, "ffn_down_s", il); }
            nm(shexp, "ffn_shexp", il);
            if (!H.flint) {   // flint: an ungated shared expert
            ggml_tensor * sg = ggml_mul_mat(ctx, l.gate_inp_shexp, x);
            nm(sg, "shared_expert_gate", il);
            sg = ggml_sigmoid(ctx, sg);
            nm(sg, "shared_expert_gate_sigmoid", il);
            shexp = ggml_mul(ctx, shexp, sg);
            nm(shexp, "ffn_shexp_gated", il);
            }
            cur = ggml_add(ctx, moe_out, shexp);
            nm(cur, "ffn_out", il);
        }
        if (H.q4x) {
            res_hc = hc_combine(res_hc, cur, inject, il);
            nm(res_hc, "l_last", il);
            continue;
        }
        if (l.ffn_post_norm) {   // lfm3: post_ffn_norm on the FFN's output, before the residual
            cur = norm(ctx, cur, l.ffn_post_norm, il);
            nm(cur, "post_ffn_norm", il);
        }
        cur = H.lfm2 ? ggml_add(ctx, ffn_residual, cur) : ggml_add(ctx, cur, ffn_residual);
        nm(cur, H.dense ? "post_ffn" : "post_moe", il);
        nm(cur, "l_out", il);
        inpL = cur;
    }
    // llama_context marks the graph's results as outputs (t_h_nextn, t_embd, t_logits); the flag also keeps ggml-cuda
    // from fusing across them, so it matters for bit-exactness
    ggml_tensor * cur = nullptr, * t_hn = nullptr, * t_logits = nullptr, * st_out = nullptr;
    if (L1 < H.n_layer) {   // a pipelined read's stage before the last: the residual goes to the next card
        st_out = H.q4x ? res_hc : inpL;
        ggml_set_name(st_out, "stage_out");
        ggml_set_output(st_out);
        ggml_build_forward_expand(gf, st_out);
    } else {
    if (H.q4x) {   // the final hc mixer is the output norm (no inject); the rows were selected at the last layer
        if (wide_h) {   // every row's residual for the drafter, then the output rows
            ggml_set_name(res_hc, "h_nextn");
            ggml_set_output(res_hc);
            t_hn = res_hc;
            res_hc = ggml_reshape_2d(ctx, res_hc, HCD, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx, res_hc, n_embd, H.hc, res_hc->ne[1]);
        }
        cur = hc_mix(res_hc, hc_head_norm_, hc_head_down_, hc_head_up_, nullptr, nullptr, -1);
        ggml_set_name(cur, "result_norm");
        ggml_set_output(cur);
        if (!wide_h) t_hn = cur;
    } else {
    cur = norm(ctx, inpL, output_norm_, -1);
    ggml_set_name(cur, "h_nextn");
    ggml_set_output(cur);
    t_hn = cur;   // every row (draft-mtp: llama_set_embeddings_nextn(ctx_tgt, true, unmasked)), read when want_h_
    if (!(H.lfm2 || H.flint) || n_outputs == 0) cur = ggml_get_rows(ctx, cur, inp_out_ids);
    ggml_set_name(cur, "result_norm");
    ggml_set_output(cur);
    }
    cur = mm_s(ctx, ggml_mul_mat(ctx, output_, cur), output_s_);
    ggml_set_name(cur, "result_output");
    ggml_set_output(cur);
    t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    }

    if (reserve) {
        dcheck_ = {};   // a reserve graph is never computed
        const bool ok = ggml_backend_sched_reserve(sched_, gf);
        ggml_free(ctx);
        return ok ? "" : "graph reserve failed";
    }
    // a read through the rings has no RAM weights left, so the scheduler makes the card's layers one split and copies every
    // input in at its start: the mask and the indexer tables once per attention layer, all live together. On the card
    // from the start they are there once (one card per graph: a pipelined stage, or a single card)
    if (ring_use && (staged_build || gpus_.size() == 1)) {
        ggml_backend_t gb = gpus_[staged_build ? st_card_ : 0];
        for (ggml_tensor * t : { kq_mask, kq_mask_swa, k_idxs, v_idxs, k_idxs_s, v_idxs_s, inp_pos, s_copy, k_rot, v_rot, qsa_k_idxs, cell_blk, blk_cells, blk_pos, qsa_bias })
            if (t) ggml_backend_sched_set_tensor_backend(sched_, t, gb);
    }
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(ctx); return "graph allocation failed"; }
    g_.ctx = ctx; g_.gf = gf; g_.key = key; g_.valid = true;
    g_.inp_tokens = inp_tokens; g_.inp_embd = inp_embd; g_.inp_pos = inp_pos; g_.k_idxs = k_idxs; g_.v_idxs = v_idxs; g_.kq_mask = kq_mask; g_.kq_mask_swa = kq_mask_swa; g_.swa_lo = swa_lo; g_.k_idxs_s = k_idxs_s; g_.v_idxs_s = v_idxs_s;
    g_.s_copy = s_copy; g_.k_rot = k_rot; g_.v_rot = v_rot; g_.inp_out_ids = inp_out_ids; g_.t_logits = t_logits; g_.t_h = t_hn;
    g_.adapt_ids = std::move(adapt_ids); g_.dcheck = dcheck_; g_.acc_inc = acc_inc; g_.hit_inc = hit_inc;
    g_.ple_rows = ple_rows; g_.qsa_k_idxs = qsa_k_idxs; g_.cell_blk = cell_blk; g_.blk_cells = blk_cells; g_.blk_pos = blk_pos;
    g_.qsa_bias = qsa_bias; g_.qsa_r = qsa_r; g_.st_in = st_in; g_.st_out = st_out; g_.ring = ring_use;
    stats_.graphs_built++;
    } else {   // ---- build ends
        dcheck_ = g_.dcheck;
        stats_.graphs_reused++;
    }
    const Graph & G = g_;
    const auto ti0 = std::chrono::steady_clock::now();
    if (timed) { stats_.t_ubatches++; stats_.t_build += std::chrono::duration<double, std::milli>(ti0 - tb0).count(); }

    // inputs
    if (G.st_in) ggml_backend_tensor_set(G.st_in, st_in_, 0, ggml_nbytes(G.st_in));
    else if (embd_in_) {
        std::vector<int32_t> zero(n, 0);
        ggml_backend_tensor_set(G.inp_tokens, zero.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(G.inp_embd, embd_in_, 0, (size_t) n * n_embd * sizeof(float));
    } else ggml_backend_tensor_set(G.inp_tokens, toks, 0, n * sizeof(int32_t));
    if (H.pos1()) ggml_backend_tensor_set(G.inp_pos, positions.data(), 0, (size_t) n * sizeof(int32_t));
    else if (pos4_in_) ggml_backend_tensor_set(G.inp_pos, pos4_in_, 0, (size_t) n * 4 * sizeof(int32_t));
    else {
        std::vector<int32_t> p4((size_t) n * 4);
        for (int i = 0; i < n; ++i) { p4[i] = positions[i]; p4[n + i] = positions[i]; p4[2 * n + i] = positions[i]; p4[3 * n + i] = 0; }
        ggml_backend_tensor_set(G.inp_pos, p4.data(), 0, p4.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(G.k_idxs, kv_.idxs.data(), 0, n * sizeof(int64_t));
    ggml_backend_tensor_set(G.v_idxs, kv_.idxs.data(), 0, n * sizeof(int64_t));
    if (G.k_idxs_s) {
        ggml_backend_tensor_set(G.k_idxs_s, kvs_.idxs.data(), 0, n * sizeof(int64_t));
        ggml_backend_tensor_set(G.v_idxs_s, kvs_.idxs.data(), 0, n * sizeof(int64_t));
    }
    {   // the causal mask: cell j is seen by token i when 0 <= pos[j] <= positions[i] (lite/mask_fill.hpp)
        std::vector<ggml_fp16_t> md((size_t) kv_.n_kv * n);
        std::vector<int32_t> lo(n, 0);
        for (int q = 0; q < ns; ++q)   // each sequence's rows over its own stream's cells
            lite::fill_mask(md.data() + (size_t) q * nt * kv_.n_kv, nt, kv_.n_kv, kv_.pos.data() + (size_t) kv_.stream_of(head + q) * kv_.ssize,
                            lo.data(), positions.data() + (size_t) q * nt, o_.n_threads);
        if (pos4_in_ && !H.pos1()) {   // an image ubatch: the M-RoPE causal rule, a cell of the token's own position is hidden when
            // it lies after the token in raster order (llama_kv_cache::set_input_kq_mask, is_2d); one sequence per stream
            const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < ns; ++q) {
                const size_t base = (size_t) kv_.stream_of(head + q) * kv_.ssize;
                std::unordered_map<int32_t, std::vector<uint32_t>> at;   // the cells of each position this ubatch's tokens hold
                for (int i = 0; i < nt; ++i) at.emplace(positions[(size_t) q * nt + i], std::vector<uint32_t>{});
                for (uint32_t j = 0; j < kv_.n_kv; ++j) { auto it = at.find(kv_.pos[base + j]); if (it != at.end()) it->second.push_back(j); }
                for (int i = 0; i < nt; ++i) {
                    const int gi = q * nt + i;
                    const int32_t y = pos4_in_[n + gi], x = pos4_in_[2 * n + gi];
                    ggml_fp16_t * row = md.data() + (size_t) gi * kv_.n_kv;
                    for (uint32_t j : at[positions[gi]]) if (kv_.after_2d(base + j, x, y)) row[j] = drop;
                }
            }
        }
        ggml_backend_tensor_set(G.kq_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
        if (G.kq_mask_swa) {   // LLAMA_SWA_TYPE_STANDARD: a cell is hidden when p1 - p0 >= n_swa (the current one and n_swa - 1 before)
            const Cache & SC = swa_ring_ ? kvs_ : kv_;   // lfm3: the sliding cache's own cells
            const uint32_t c0 = swa_ring_ ? 0 : G.swa_lo, w = SC.n_kv - c0;   // the cells from swa_lo on
            if (md.size() < (size_t) w * n) md.resize((size_t) w * n);
            for (int i = 0; i < n; ++i) lo[i] = std::max(0, positions[i] - H.n_swa + 1);
            for (int q = 0; q < ns; ++q)
                lite::fill_mask(md.data() + (size_t) q * nt * w, nt, w, SC.pos.data() + (size_t) SC.stream_of(head + q) * SC.ssize + c0,
                                lo.data() + (size_t) q * nt, positions.data() + (size_t) q * nt, o_.n_threads);
            ggml_backend_tensor_set(G.kq_mask_swa, md.data(), 0, (size_t) w * n * sizeof(ggml_fp16_t));
        }
    }
    {   // s_copy (llama_memory_recurrent::s_copy): each cell's source row, rs_idx * n_seq + src0 (a new cell reads rs_z)
        std::vector<int32_t> sc(ns);
        for (int q = 0; q < ns; ++q) sc[q] = rs_src_v_[head + q] * NS + (fresh_v_[head + q] ? rs_zero : head + q);
        if (st_replay_) sc = st_replay_->s_copy;
        if (st_rec_) st_rec_->s_copy = sc;
        if (G.s_copy->buffer) ggml_backend_tensor_set(G.s_copy, sc.data(), 0, sc.size() * sizeof(int32_t));   // flint: no recurrent layer reads it
    }
    for (ggml_tensor * r : { G.k_rot, G.v_rot }) {
        if (!r) continue;
        const std::vector<float> hd = gen_hadamard((int) r->ne[0]);
        ggml_backend_tensor_set(r, hd.data(), 0, hd.size() * sizeof(float));
    }   // s_copy: idx * size + src0
    if (G.ple_rows) set_ple_rows(G.ple_rows, embd_in_ ? nullptr : toks, n, ns, nt, head, positions);
    if (G.cell_blk) { const std::string e = set_qsa(G, ns, nt, head, positions, kv_.n_kv, pos4_in_, n); if (!e.empty()) return e; }
    if (n_outputs) {
        std::vector<int32_t> oid(n_outputs);
        for (int i = 0; i < n_outputs; ++i) oid[i] = n - n_outputs + i;
        ggml_backend_tensor_set(G.inp_out_ids, oid.data(), 0, oid.size() * sizeof(int32_t));
    }

    static int n_single = 0;
    if (n == 1) ++n_single;
    if (getenv("Q35_GRAPHDUMP") && n == 1 && n_single == 3) {
        g_gdump = fopen(getenv("Q35_GRAPHDUMP"), "w");
        ggml_backend_sched_set_eval_callback(sched_, gdump_cb, nullptr);
    } else if (getenv("TDUMP") || getenv("GDUMP")) ggml_backend_sched_set_eval_callback(sched_, tdump_cb, nullptr);
    else ggml_backend_sched_set_eval_callback(sched_, G.ring ? ring_cb : nullptr, G.ring.get());
    stats_.accesses += G.acc_inc; stats_.hits += G.hit_inc;
    const uint64_t acc0 = stats_.accesses, hit0 = stats_.hits;
    const auto tc0 = std::chrono::steady_clock::now();
    if (timed) stats_.t_inputs += std::chrono::duration<double, std::milli>(tc0 - ti0).count();
    // several cards: queued, not waited for (llama_context::graph_compute); waited for only before a readback
    const bool staged = st_l1_ >= 0;
    const bool pipe = gpus_.size() > 1 && !staged;
    // a pipelined read's async stage: the caller computes the graph on the card's own thread (the scheduler blocks its
    // thread on every split's expert copies, so one thread cannot keep two cards' links busy)
    if (staged && st_async_) st_gf_ = G.gf;
    const ggml_status st = (staged && st_async_) ? GGML_STATUS_SUCCESS
                         : pipe ? ggml_backend_sched_graph_compute_async(sched_, G.gf) : ggml_backend_sched_graph_compute(sched_, G.gf);
    if (pipe && !staged && ((n_outputs && logits) || want_h_ || lazy_req_)) sync_all();   // ggml-cuda's streams do not block
    const auto tc1 = std::chrono::steady_clock::now();
    if (timed) stats_.t_compute += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
    if (n < kOffloadMin && !cache_.empty()) stats_.step_hits.push_back({ (uint32_t) (stats_.accesses - acc0), (uint32_t) (stats_.hits - hit0) });
    if (reading_) stats_.read_buf_bytes = std::max<uint64_t>(stats_.read_buf_bytes, ggml_backend_sched_get_buffer_size(sched_, staged ? gpus_[st_card_] : gpu_));
    if (g_gdump) { fclose(g_gdump); g_gdump = nullptr; ggml_backend_sched_set_eval_callback(sched_, nullptr, nullptr); }
    if (st != GGML_STATUS_SUCCESS) { graph_drop(g_); return "compute failed"; }
    for (int q = 0; q < ns && !st_replay_; ++q) {
        fresh_v_[head + q] = 0;
        rs_src_v_[head + q] = 0;                   // a rollback index is single-use (llama_memory_recurrent_context::s_copy)
        last_ub_v_[head + q] = embd_in_ ? 0 : nt;
    }
    if (dcheck_.merged) {   // every merged row must equal its source twin bit for bit
        const DCheck d = dcheck_;
        dcheck_ = {};
        const int64_t k = d.merged->ne[1], ne = d.merged->ne[0];
        std::vector<float> m(ggml_nelements(d.merged)), g(m.size()), c(m.size());
        ggml_backend_tensor_get(d.merged, m.data(), 0, m.size() * 4);
        ggml_backend_tensor_get(d.twin_gpu, g.data(), 0, g.size() * 4);
        ggml_backend_tensor_get(d.twin_cpu, c.data(), 0, c.size() * 4);
        std::vector<int32_t> ids(ggml_nelements(d.argsort));
        ggml_backend_tensor_get(d.argsort, ids.data(), 0, ids.size() * 4);
        const int64_t ne_sort = d.argsort->ne[0];
        for (int64_t t = 0; t < d.merged->ne[2]; ++t) for (int64_t j = 0; j < k; ++j) {
            const int e = ids[t * ne_sort + j];
            const bool hit = d.C->res[e] >= 0;
            const float * src = (hit ? g.data() : c.data()) + (t * k + j) * ne;
            stats_.dcheck_rows++; stats_.dcheck_gpu_rows += hit;
            if (memcmp(m.data() + (t * k + j) * ne, src, ne * 4) != 0) stats_.dcheck_bad++;
        }
    }
    const auto tra = std::chrono::steady_clock::now();
    if (o_.adapt && staged) { if (n >= kOffloadMin) refill_pending_ = o_.adapt_refill; }   // a pipelined read: no readback mid-pipeline
    else if (o_.adapt) adapt_after(n, G.adapt_ids);
    if (n_outputs && lazy_req_) lazy_t_ = G.t_logits;
    const auto trb = std::chrono::steady_clock::now();
    if (n_outputs && logits) {
        logits->resize((size_t) n_outputs * H.n_vocab);
        ggml_backend_tensor_get(G.t_logits, logits->data(), 0, logits->size() * sizeof(float));
    }
    if (timed) {
        stats_.t_ro_adapt += std::chrono::duration<double, std::milli>(trb - tra).count();
        stats_.t_ro_logits += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trb).count();
    }
    if (want_h_ && !(staged && st_async_)) {   // a deferred stage's rows are read when it finishes (read_pipelined)
        h_rows_.resize(ggml_nelements(G.t_h));
        ggml_backend_tensor_get(G.t_h, h_rows_.data(), 0, h_rows_.size() * sizeof(float));
    }
    if (timed) stats_.t_readout += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc1).count();
    return "";   // the graph and its allocation stay for the next step of this shape
}

// llama_decode: one batch, split into n_ubatch pieces; only the batch's last token outputs (llama_batch_get_one). With
// n_rs_seq the split keeps the batch's last n_rs_seq + 1 tokens in one ubatch (split_equal's n_keep_tail). With MTP
// the drafter re-reads each ubatch after it (common_speculative_impl_draft_mtp::process does it per batch; per ubatch
// keeps a long read's hidden rows bounded, and the drafter splits in n_ubatch pieces either way).
std::string Engine::decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last, int seq) {
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    cur_seq_ = seq;
    int & n_past_ = n_past_v_[seq];
    const int n = (int) toks.size(), tail = o_.n_rs_seq > 0 ? o_.n_rs_seq + 1 : 0;
    const int64_t ne = h_width();
    // stage g: the read's ubatch size
    const int pu = (o_.prefill_ubatch < 0 && n >= kOffloadMin) ? pick_chunk(n) : o_.prefill_ubatch;
    const int ub = (pu > 0 && n >= kOffloadMin) ? pu : o_.n_ubatch;
    if (n >= kOffloadMin) stats_.last_chunk = std::min(ub, n);
    if (o_.prefill_ubatch < 0 && n >= kOffloadMin)
        fprintf(stderr, "impulse: reading %d tokens at position %d in %d-token chunks (cap %d)\n", n, n_past_, std::min(ub, n), ub);
    const auto tr0 = std::chrono::steady_clock::now();
    if (ub != o_.n_ubatch || (o_.lend && n >= kOffloadMin)) begin_read(ub);
    struct RingGuard { Engine * e; ~RingGuard() { e->ring_end(); } } ring_guard{ this };
    if (reading_ && ub >= kOffloadMin && !hp_.dense && o_.n_cpu_moe > 0) {   // the read's chunks of kOffloadMin+ tokens stream through the rings
        int C = 0;
        for (int i = 0; i < n;) {
            int m = std::min(ub, n - i);
            const int rem = n - i - m;
            if (tail && rem > 0 && rem < tail) m -= tail - rem;
            if (rem > 0 && rem < kOffloadMin && !tail) m += rem;   // read_pipelined takes a short tail into the chunk before
            if (m >= kOffloadMin) ++C;
            i += m;
        }
        ring_begin(C, ub);
    }
    if (!h_last && read_pipeline_ok(n) && (n + ub - 1) / ub >= 2) {   // several chunks over several cards: Strata's chunk pipeline
        const std::string err = read_pipelined(toks, ub, n_past_, logits);
        ring_end();
        if (!err.empty()) return err;
        n_past_ += n;
        stats_.prefill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
        return "";
    }
    for (int i = 0; i < n;) {
        int m = std::min(ub, n - i);
        const int rem = n - i - m;
        if (tail && rem > 0 && rem < tail) m -= tail - rem;
        const bool out = (i + m == n);
        if (m < kOffloadMin) ring_end();   // a short tail ends the read (ubatch: end_read) without the rings
        want_h_ = (out && h_last) || mtp_.loaded;
        std::string err = ubatch(toks.data() + i, m, n_past_ + i, out ? 1 : 0, out ? logits : nullptr);
        want_h_ = false;
        if (!err.empty()) return err;
        if ((i == 0 || i + m == n) && m >= kOffloadMin)   // the first and the last chunk (the most KV cells)
            fprintf(stderr, "impulse: read compute buffers %.0f MiB (%d-token chunk, %u KV cells)\n",
                    ggml_backend_sched_get_buffer_size(sched_, gpu_) / 1048576.0, m, kv_.n_kv);
        if (out && h_last) h_last->assign(h_rows_.end() - ne, h_rows_.end());
        if (mtp_.loaded) {
            err = mtp_catchup(toks.data() + i, m, n_past_ + i, h_rows_.data());
            if (!err.empty()) return err;
        }
        i += m;
    }
    n_past_ += n;
    if (n >= kOffloadMin) stats_.prefill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    return "";
}

// a pipelined read: an MoE model with RAM-resident experts over several cards (layer split; with every expert on the
// cards, or a dense model, ggml's own pipeline parallelism runs instead), a plain read of one sequence, text tokens; with
// the MTP drafter (snapshot slots for its verify windows) the drafter re-reads each chunk as it leaves the last stage; IMPULSE_PIPELINE=0 turns it off
bool Engine::read_pipeline_ok(int n) const {
    static const bool off = getenv("IMPULSE_PIPELINE") && atoi(getenv("IMPULSE_PIPELINE")) == 0;
    if (off || hp_.dense || o_.n_cpu_moe <= 0 || tp_ || gpus_.size() < 2 || n < kOffloadMin || !ub_pos_.empty() || (o_.n_rs_seq > 0 && !mtp_.loaded) || embd_in_ || pos4_in_) return false;
    int stages = 1;
    for (int il = 1; il < hp_.n_layer; ++il) if (dev_layer_[il] != dev_layer_[il - 1]) ++stages;
    return stages >= 2 && dev_out_ == dev_layer_[hp_.n_layer - 1];
}

// Each card runs its layers on chunk c while the next card runs its layers on chunk c - 1 (Strata's two-stage chunk
// pipeline): both cards' PCIe links stream their own layers' RAM-resident experts at once. Lockstep: every step waits for the
// stages in flight, takes each stage's residual out (host memory), then starts every stage on its next chunk.
std::string Engine::read_pipelined(const std::vector<int32_t> & toks, int ub, int pos0, std::vector<float> * logits) {
    struct St { int card, l0, l1; };
    std::vector<St> S;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (S.empty() || S.back().card != dev_layer_[il]) S.push_back({ dev_layer_[il], il, il + 1 });
        else S.back().l1 = il + 1;
    }
    const int n = (int) toks.size(), NS = (int) S.size();
    std::vector<std::pair<int, int>> chunk;   // (start, length)
    for (int i = 0; i < n; i += ub) chunk.push_back({ i, std::min(ub, n - i) });
    if (chunk.size() > 1 && chunk.back().second < kOffloadMin) {   // a short tail would end the read mid-pipeline (ubatch)
        chunk[chunk.size() - 2].second += chunk.back().second; chunk.pop_back();
    }
    const int C = (int) chunk.size();
    sched_st_.resize(gpus_.size(), nullptr);
    g_st_.resize(gpus_.size());
    cpu_st_.resize(gpus_.size(), nullptr);
    graph_buf_st_.resize(gpus_.size());
    for (const St & st : S) if (!sched_st_[st.card]) {
        graph_buf_st_[st.card].resize(graph_buf_.size());
        if (!cpu_st_[st.card]) {
            cpu_st_[st.card] = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            ggml_backend_cpu_set_n_threads(cpu_st_[st.card], o_.n_threads);
        }
        ggml_backend_t backs[2] = { gpus_[st.card], cpu_st_[st.card] };
        ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(gpus_[st.card]), lite::pinned_host_buffer_type(cpu_) };
        sched_st_[st.card] = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, false, true);
    }
    std::vector<UbRec> rec(C);
    const size_t row = (size_t) hp_.n_embd * (hp_.q4x ? hp_.hc : 1);
    std::vector<std::vector<float>> handoff(NS);   // stage s's residual for chunk c (read when stage s finishes c)
    std::vector<int> inflight(NS, -1);
    std::vector<std::thread> th(NS);
    std::vector<ggml_status> th_st(NS, GGML_STATUS_SUCCESS);
    std::string err;
    auto run = [&](int s, int c, bool async) {
        const St & st = S[s];
        const bool last = s == NS - 1 && c == C - 1;
        st_l0_ = st.l0; st_l1_ = st.l1; st_card_ = st.card; st_async_ = async;
        want_h_ = mtp_.loaded && s == NS - 1;   // the drafter's rows: every row's 4-stream residual
        st_in_ = s > 0 ? handoff[s - 1].data() : nullptr;
        st_rec_ = s == 0 ? &rec[c] : nullptr; st_replay_ = s > 0 ? &rec[c] : nullptr;
        std::swap(sched_, sched_st_[st.card]); std::swap(g_, g_st_[st.card]); graph_buf_.swap(graph_buf_st_[st.card]);
        st_gf_ = nullptr;
        std::string e = ubatch(toks.data() + chunk[c].first, chunk[c].second, pos0 + chunk[c].first, last ? 1 : 0, last ? logits : nullptr);
        std::swap(sched_, sched_st_[st.card]); std::swap(g_, g_st_[st.card]); graph_buf_.swap(graph_buf_st_[st.card]);
        if (e.empty() && st_gf_) {
            ggml_backend_sched_t sc = sched_st_[st.card];
            ggml_cgraph * gf = st_gf_;
            double * busy = &stats_.stage_ms[std::min(s, 3)];
            th[s] = std::thread([&th_st, s, sc, gf, busy] {
                const auto t0 = std::chrono::steady_clock::now();
                th_st[s] = ggml_backend_sched_graph_compute(sc, gf);
                *busy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            });
        }
        st_gf_ = nullptr;
        st_l0_ = 0; st_l1_ = -1; st_card_ = 0; st_async_ = false; st_in_ = nullptr; st_rec_ = nullptr; st_replay_ = nullptr;
        want_h_ = false;
        if (e.empty() && last && mtp_.loaded) e = mtp_catchup(toks.data() + chunk[c].first, chunk[c].second, pos0 + chunk[c].first, h_rows_.data());
        return e;
    };
    auto finish = [&](int s) {   // stage s's chunk in flight is done; its residual kept for the next stage
        const int c = inflight[s];
        if (c < 0) return;
        const auto tw0 = std::chrono::steady_clock::now();
        if (th[s].joinable()) th[s].join();
        stats_.stage_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
        if (th_st[s] != GGML_STATUS_SUCCESS && err.empty()) err = "compute failed (pipelined read)";
        th_st[s] = GGML_STATUS_SUCCESS;
        ggml_backend_sched_synchronize(sched_st_[S[s].card]);
        if (s < NS - 1) {
            ggml_tensor * out = g_st_[S[s].card].st_out;
            handoff[s].resize(row * chunk[c].second);
            ggml_backend_tensor_get(out, handoff[s].data(), 0, handoff[s].size() * sizeof(float));
        } else if (mtp_.loaded && err.empty()) {   // the drafter re-reads the chunk (in order: the last stage finishes them in order)
            ggml_tensor * th = g_st_[S[s].card].t_h;
            h_rows_.resize(ggml_nelements(th));
            ggml_backend_tensor_get(th, h_rows_.data(), 0, h_rows_.size() * sizeof(float));
            err = mtp_catchup(toks.data() + chunk[c].first, chunk[c].second, pos0 + chunk[c].first, h_rows_.data());
        }
        inflight[s] = -1;
    };
    for (int t = 0; t < C + NS - 1 && err.empty(); ++t) {
        for (int s = 0; s < NS; ++s) finish(s);
        for (int s = NS - 1; s >= 0 && err.empty(); --s) {   // the later stages first: they take the residual just collected
            const int c = t - s;
            if (c < 0 || c >= C) continue;
            const bool last = s == NS - 1 && c == C - 1;
            err = run(s, c, !last);
            if (err.empty() && !last) inflight[s] = c;
        }
    }
    for (int s = 0; s < NS; ++s) finish(s);
    for (const St & st : S) {   // the stages' compute buffers go back before the arenas return (end_read)
        graph_drop(g_st_[st.card]);
        ggml_backend_sched_free(sched_st_[st.card]); sched_st_[st.card] = nullptr;
    }
    sync_all();
    return err;
}

// stage f: a verify window [id_last, d1..dk] in one ubatch, every row output (the server's spec batch: logits on all)
std::string Engine::decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h, int seq) {
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    cur_seq_ = seq;
    int & n_past_ = n_past_v_[seq];
    const int n = (int) toks.size();
    if (n > o_.n_ubatch) return "verify window larger than a ubatch";
    want_h_ = true;
    std::string err = ubatch(toks.data(), n, n_past_, n, logits);
    want_h_ = false;
    if (!err.empty()) return err;
    win_h_v_[seq] = h_rows_;
    win_pos0_v_[seq] = n_past_;
    if (h) *h = h_rows_;
    if (mtp_.loaded) {
        err = mtp_catchup(toks.data(), n, n_past_, win_h_v_[seq].data());
        if (!err.empty()) return err;
    }
    n_past_ += n;
    return "";
}

// the server's step over several slots: llama_batch with one sequence per step, split as llama_memory_hybrid splits it
// (split_equal, sequential): consecutive sequences with the same token count share a ubatch (seq-major: token q * nt + t),
// the others get their own; every row outputs (logits + hidden rows), the drafter re-reads each sequence's rows
std::string Engine::decode_multi(const std::vector<SeqStep> & steps, std::vector<float> * logits, std::vector<float> * h, bool lazy) {
    const int nv = hp_.n_vocab;
    const int64_t ne = h_width();
    std::vector<int> order(steps.size()), off(steps.size());
    int total = 0;
    for (size_t k = 0; k < steps.size(); ++k) {
        order[k] = (int) k; off[k] = total; total += (int) steps[k].toks.size();
        if (steps[k].seq < 0 || steps[k].seq >= n_seq() || steps[k].toks.empty()) return "bad step";
        for (size_t j = 0; j < k; ++j) if (steps[j].seq == steps[k].seq) return "two steps for one sequence";
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return steps[a].seq < steps[b].seq; });
    lazy_t_ = nullptr;
    if (lazy) {   // lazy only when the whole step is one ubatch (the grouping below, run once)
        int groups = 0;
        for (size_t i = 0; i < order.size();) {
            const int len = (int) steps[order[i]].toks.size();
            size_t j = i + 1;
            while (j < order.size() && steps[order[j]].seq == steps[order[j - 1]].seq + 1 && (int) steps[order[j]].toks.size() == len &&
                   (int) (j - i + 1) * len <= o_.n_ubatch) ++j;
            ++groups; i = j;
        }
        lazy = groups == 1;
    }
    if (lazy) {
        lazy_map_.assign(total, 0);
        for (size_t q = 0; q < order.size(); ++q)
            for (size_t t = 0; t < steps[order[q]].toks.size(); ++t) lazy_map_[off[order[q]] + t] = (int) (q * steps[order[q]].toks.size() + t);
        logits = nullptr;
    }
    if (logits) logits->assign((size_t) total * nv, 0.0f);
    if (h) h->assign((size_t) total * ne, 0.0f);
    std::vector<float> lg;
    for (size_t i = 0; i < order.size();) {
        const int len = (int) steps[order[i]].toks.size();
        size_t j = i + 1;
        while (j < order.size() && steps[order[j]].seq == steps[order[j - 1]].seq + 1 && (int) steps[order[j]].toks.size() == len &&
               (int) (j - i + 1) * len <= o_.n_ubatch) ++j;
        const int ns = (int) (j - i), n = ns * len;
        if (n > o_.n_ubatch) return "a step larger than a ubatch";
        std::vector<int32_t> toks; ub_pos_.clear(); ub_seq_.clear();
        for (size_t q = i; q < j; ++q) {
            const SeqStep & st = steps[order[q]];
            for (int t = 0; t < len; ++t) { toks.push_back(st.toks[t]); ub_pos_.push_back(n_past_v_[st.seq] + t); ub_seq_.push_back((int8_t) st.seq); }
        }
        ub_ns_ = ns;
        want_h_ = h || mtp_.loaded;   // the hidden rows only when the drafter or the caller reads them
        lazy_req_ = lazy;
        std::string err = ubatch(toks.data(), n, 0, n, lazy ? nullptr : &lg);
        lazy_req_ = false;
        want_h_ = false; ub_pos_.clear(); ub_seq_.clear(); ub_ns_ = 0;
        if (!err.empty()) lazy_t_ = nullptr;
        if (!err.empty()) return err;
        for (size_t q = i; q < j; ++q) {
            const int k = order[q], s = steps[k].seq, r0 = (int) (q - i) * len;
            if (logits) std::copy(lg.begin() + (size_t) r0 * nv, lg.begin() + (size_t) (r0 + len) * nv, logits->begin() + (size_t) off[k] * nv);
            if (h) std::copy(h_rows_.begin() + (size_t) r0 * ne, h_rows_.begin() + (size_t) (r0 + len) * ne, h->begin() + (size_t) off[k] * ne);
            if (mtp_.loaded) {
                win_h_v_[s].assign(h_rows_.begin() + (size_t) r0 * ne, h_rows_.begin() + (size_t) (r0 + len) * ne);
                win_pos0_v_[s] = n_past_v_[s];
                cur_seq_ = s;
                err = mtp_catchup(steps[k].toks.data(), len, n_past_v_[s], win_h_v_[s].data());
                if (!err.empty()) return err;
            }
            n_past_v_[s] += len;
        }
        i = j;
    }
    return "";
}

// stage f: keep positions < n_keep. The attention layers free their cells (seq_rm); the delta-net layers read the
// snapshot r = n_past - n_keep tokens back in the next ubatch (llama_memory_recurrent::seq_rm's partial rollback, bounded
// by n_rs_seq); the drafter frees its cells and takes the kept position's hidden row (draft-mtp's accept)
std::string Engine::rollback(int n_keep, int seq) {
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    int & n_past_ = n_past_v_[seq]; int & rs_src_ = rs_src_v_[seq]; int & last_ub_n_ = last_ub_v_[seq];
    const int r = n_past_ - n_keep;
    if (r <= 0) return "";
    if (r > o_.n_rs_seq || rs_src_ != 0 || r >= last_ub_n_) return "rollback of " + std::to_string(r) + " tokens is past the recurrent snapshots";
    if (!swa_resumable(seq, n_keep)) return "rollback of " + std::to_string(r) + " tokens is past the sliding window's cells";
    kv_.seq_rm(seq, n_keep);
    if (swa_ring_) kvs_.seq_rm(seq, n_keep);
    for (int c = seq * kCkpts; c < (seq + 1) * kCkpts; ++c) if (n_keep < ckpt_pos_[c]) ckpt_pos_[c] = -1;
    last_ub_n_ -= r;
    rs_src_ = r;
    n_past_ = n_keep;
    if (mtp_.loaded) {
        mtp_.kv.seq_rm(seq, n_keep);
        const std::vector<float> & wh = win_h_v_[seq];
        const int64_t row = n_keep - 1 - win_pos0_v_[seq], ne = h_width();
        if (row >= 0 && (size_t) (row + 1) * ne <= wh.size())
            mtp_h_v_[seq].assign(wh.begin() + row * ne, wh.begin() + (row + 1) * ne);
    }
    return "";
}

void Engine::mtp_trim(int pos, int seq) { if (mtp_.loaded) mtp_.kv.seq_rm(seq, pos); }

const float * Engine::logits_row(int r) {
    if (!lazy_t_ || r < 0 || r >= (int) lazy_map_.size()) return nullptr;
    const size_t nv = (size_t) hp_.n_vocab;
    if (lazy_map_.size() > lazy_pin_rows_) {
        if (lazy_pin_) cudaFreeHost(lazy_pin_);
        lazy_pin_rows_ = std::max<size_t>(lazy_map_.size(), 8);
        if (cudaMallocHost((void **) &lazy_pin_, lazy_pin_rows_ * nv * sizeof(float)) != cudaSuccess) { lazy_pin_ = nullptr; lazy_pin_rows_ = 0; return nullptr; }
    }
    float * dst = lazy_pin_ + (size_t) r * nv;
    ggml_backend_tensor_get(lazy_t_, dst, (size_t) lazy_map_[r] * nv * sizeof(float), nv * sizeof(float));
    return dst;
}

// ---------------------------------------------------------------- stage f: the MTP drafter
// load: the MTP-only GGUF (its nextn block, plus token_embd / output unless the target's are shared), every drafter
// weight on the GPU but token_embd (the input layer stays in RAM, as llama.cpp keeps it); its own KV cells (f16, the
// target's size) and its own scheduler, so the target's allocation plan (and with it ggml-cuda's fusions) is untouched
std::string Engine::load_mtp() {
    Mtp & M = mtp_;
    const HParams & H = hp_;
    gguf_init_params gp = { true, &M.meta };
    M.gguf = gguf_init_from_file(o_.mtp.c_str(), gp);
    if (!M.gguf) return "cannot read " + o_.mtp;
    std::string A;
    {
        const int64_t id = gguf_find_key(M.gguf, "general.architecture");
        const std::string arch = id < 0 ? "" : gguf_get_val_str(M.gguf, id);
        if (arch != "qwen35moe" && arch != "qwen35" && arch != "qwen4exp") return "not a qwen35moe / qwen35 / qwen4exp GGUF (" + arch + ")";
        if ((arch == "qwen35") != H.dense || (arch == "qwen4exp") != H.q4x) return "the drafter is " + arch + ", the target is not";
        A = arch + ".";
        M.dense = arch == "qwen35";
        M.q4x = arch == "qwen4exp";
        if (kv_int(M.gguf, A + "nextn_predict_layers", 0) != 1) return "expects one MTP block (nextn_predict_layers = 1)";
        if (kv_int(M.gguf, A + "embedding_length") != H.n_embd || (!M.dense && kv_int(M.gguf, A + "expert_count") != H.n_expert))
            return "the drafter's shape does not match the target";
    }
    const int il = (int) kv_int(M.gguf, A + "block_count") - 1;
    // the card: the only one; with a layer split the output's (the shared head is there); with a tensor split the last card
    // of the split, on its own (llama.cpp pins a drafter to one device: --spec-draft-device), with its own head and norms
    if (tp_) {
        std::vector<ggml_backend_dev_t> devs;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) devs.push_back(ggml_backend_dev_get(i));
        size_t d = devs.size() - 1;   // the last card with a share
        if (!o_.tensor_split.empty()) for (size_t i = 0; i < o_.tensor_split.size() && i < devs.size(); ++i) if (o_.tensor_split[i] > 0) d = i;
        M.gpu = ggml_backend_dev_init(devs[d], nullptr);
        if (!M.gpu) return "backend init failed (the drafter's card)";
        M.own_gpu = true;
        fprintf(stderr, "impulse: the MTP drafter on %s\n", ggml_backend_dev_name(devs[d]));
    } else M.gpu = gpus_[dev_out_];
    const bool d2t_file = ggml_get_tensor(M.meta, "d2t") != nullptr;
    const bool own_head = !o_.mtp_share_head || tp_ || d2t_file;   // FastMTP: its own reduced head (and its embeddings)
    M.map = lite::map_file(o_.mtp, &M.map_size);
    if (!M.map) return "cannot map " + o_.mtp;
    const size_t data_off = gguf_get_data_offset(M.gguf);
    const int n_tensors = (int) gguf_get_n_tensors(M.gguf);
    ggml_init_params ip = { (size_t) (n_tensors + 8) * ggml_tensor_overhead(), nullptr, true };
    M.ctx_w = ggml_init(ip);
    M.ctx_gpu = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> on_cpu, on_gpu;
    auto make = [&](const std::string & name, bool cpu) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(M.meta, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(cpu ? M.ctx_w : M.ctx_gpu, m);
        ggml_set_name(t, name.c_str());
        (cpu ? on_cpu : on_gpu).push_back({ t, data_off + gguf_get_tensor_offset(M.gguf, gguf_find_tensor(M.gguf, name.c_str())) });
        return t;
    };
    auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
    M.tok_embd = own_head ? make("token_embd.weight", true) : nullptr;
    M.output   = own_head ? make("output.weight", false) : nullptr;
    if (!M.tok_embd) M.tok_embd = tok_embd_;   // nextn.embed_tokens ? : model.tok_embd
    if (!M.output) { if (tp_) return "the drafter file has no output.weight (a tensor split needs the drafter's own head)"; M.output = output_; }
    if (d2t_file) {   // the FastMTP patch (qwen35.cpp): logits over n_vocab_out ids, scattered to the full vocabulary by d2t
        ggml_tensor * t = ggml_get_tensor(M.meta, "d2t");
        if (t->type != GGML_TYPE_I32 || t->ne[0] != M.output->ne[1]) return "d2t does not match the drafter's head";
        M.d2t.resize(t->ne[0]);
        memcpy(M.d2t.data(), (const char *) M.map + data_off + gguf_get_tensor_offset(M.gguf, gguf_find_tensor(M.gguf, "d2t")), M.d2t.size() * 4);
        for (int32_t v : M.d2t) if (v < 0 || v >= H.n_vocab) return "d2t maps outside the target's vocabulary";
        fprintf(stderr, "impulse: FastMTP drafter: a %zu-token draft vocabulary (d2t)\n", M.d2t.size());
    }
    Layer & l = M.l;
    if (M.q4x) {   // the qwen4exp block: hyper-connections in place of the norms, a final mixer in place of shared_head_norm
        l.hc_an = make(b("hc_attn_norm.weight"), false); l.hc_ad = make(b("hc_attn_down.weight"), false);
        l.hc_au = make(b("hc_attn_up.weight"), false);   l.hc_ai = make(b("hc_attn_inject.weight"), false);
        l.hc_fn = make(b("hc_ffn_norm.weight"), false);  l.hc_fd = make(b("hc_ffn_down.weight"), false);
        l.hc_fu = make(b("hc_ffn_up.weight"), false);    l.hc_fi = make(b("hc_ffn_inject.weight"), false);
        M.hc_head_norm = make(b("nextn.hc_head_norm.weight"), false);
        M.hc_head_down = make(b("nextn.hc_head_down.weight"), false);
        M.hc_head_up   = make(b("nextn.hc_head_up.weight"), false);
        if (!l.hc_an || !l.hc_ad || !l.hc_au || !l.hc_ai || !l.hc_fn || !l.hc_fd || !l.hc_fu || !l.hc_fi ||
            !M.hc_head_norm || !M.hc_head_down || !M.hc_head_up) return "missing MTP hyper-connection tensors";
    } else {
    l.attn_norm      = make(b("attn_norm.weight"), false);
    l.attn_post_norm = make(b("post_attention_norm.weight"), false);
    }
    l.wq = make(b("attn_q.weight"), false); l.wk = make(b("attn_k.weight"), false); l.wv = make(b("attn_v.weight"), false);
    l.wo = make(b("attn_output.weight"), false);
    l.q_norm = make(b("attn_q_norm.weight"), false); l.k_norm = make(b("attn_k_norm.weight"), false);
    if (M.dense) {
        l.dense_gate = make(b("ffn_gate.weight"), false); l.dense_up = make(b("ffn_up.weight"), false);
        l.dense_down = make(b("ffn_down.weight"), false);
    } else {
        l.gate_inp  = make(b("ffn_gate_inp.weight"), false);
        l.gate_exps = make(b("ffn_gate_exps.weight"), false); l.up_exps = make(b("ffn_up_exps.weight"), false);
        l.down_exps = make(b("ffn_down_exps.weight"), false);
        l.gate_inp_shexp = make(b("ffn_gate_inp_shexp.weight"), false);
        l.gate_shexp = make(b("ffn_gate_shexp.weight"), false); l.up_shexp = make(b("ffn_up_shexp.weight"), false);
        l.down_shexp = make(b("ffn_down_shexp.weight"), false);
    }
    M.eh_proj   = make(b("nextn.eh_proj.weight"), false);
    // the block's second-level scales (llama.cpp's generic pass covers the nextn block too): F32 {n}, on the drafter's card
    std::string scale_err;
    int n_scales = 0;
    auto make_s = [&](const ggml_tensor * w, const std::string & name, int64_t n) -> ggml_tensor * {
        if (!w) return nullptr;
        const ggml_tensor * m = ggml_get_tensor(M.meta, name.c_str());
        if (!m) return nullptr;
        if (m->type != GGML_TYPE_F32 || ggml_n_dims(m) != 1 || m->ne[0] != n) {
            if (scale_err.empty()) scale_err = name + ": expected F32 [" + std::to_string(n) + "]";
            return nullptr;
        }
        n_scales++;
        return make(name, false);
    };
    M.eh_proj_s = make_s(M.eh_proj, b("nextn.eh_proj.scale"), 1);
    l.wq_s = make_s(l.wq, b("attn_q.scale"), 1); l.wk_s = make_s(l.wk, b("attn_k.scale"), 1);
    l.wv_s = make_s(l.wv, b("attn_v.scale"), 1); l.wo_s = make_s(l.wo, b("attn_output.scale"), 1);
    l.dense_gate_s = make_s(l.dense_gate, b("ffn_gate.scale"), 1); l.dense_up_s = make_s(l.dense_up, b("ffn_up.scale"), 1);
    l.dense_down_s = make_s(l.dense_down, b("ffn_down.scale"), 1);
    l.gate_exps_s = make_s(l.gate_exps, b("ffn_gate_exps.scale"), H.n_expert); l.up_exps_s = make_s(l.up_exps, b("ffn_up_exps.scale"), H.n_expert);
    l.down_exps_s = make_s(l.down_exps, b("ffn_down_exps.scale"), H.n_expert);
    l.gate_shexp_s = make_s(l.gate_shexp, b("ffn_gate_shexp.scale"), 1); l.up_shexp_s = make_s(l.up_shexp, b("ffn_up_shexp.scale"), 1);
    l.down_shexp_s = make_s(l.down_shexp, b("ffn_down_shexp.scale"), 1);
    // the head's: its own output.scale beside an NVFP4 head of its own; the target's with the target's head (graph_mtp:
    // shared_head_head_s, else model.output_s)
    if (M.output == output_) M.output_s = output_s_;
    else if (M.output->type == GGML_TYPE_NVFP4) M.output_s = make_s(M.output, "output.scale", 1);
    if (!scale_err.empty()) return scale_err;
    if (n_scales) fprintf(stderr, "impulse: the drafter: %d second-level weight scales applied\n", n_scales);
    M.enorm     = make(b("nextn.enorm.weight"), false);
    M.hnorm     = make(b("nextn.hnorm.weight"), false);
    if (!M.q4x) {
    M.head_norm = make(b("nextn.shared_head_norm.weight"), false);
    if (!M.head_norm && tp_) M.head_norm = make("output_norm.weight", false);   // the target's is sliced over the cards
    if (!M.head_norm) M.head_norm = output_norm_;
    }
    const bool ffn_ok = M.dense ? (l.dense_gate && l.dense_up && l.dense_down) : (l.gate_inp && l.gate_exps && l.up_exps && l.down_exps);
    if ((!M.q4x && (!l.attn_norm || !l.attn_post_norm || !M.head_norm)) || !l.wq || !l.wk || !l.wv || !l.wo || !l.q_norm || !l.k_norm || !ffn_ok ||
        !M.eh_proj || !M.enorm || !M.hnorm)
        return "missing MTP block tensors";
    if (M.q4x && (M.hnorm->ne[0] != h_width() || M.eh_proj->ne[0] != 2 * H.n_embd)) return "the drafter's nextn.hnorm / eh_proj do not match the target";
    M.buf_gpu = ggml_backend_alloc_ctx_tensors_from_buft(M.ctx_gpu, ggml_backend_get_default_buffer_type(M.gpu));
    if (!M.buf_gpu) return "drafter weight allocation failed";
    ggml_backend_buffer_set_usage(M.buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & [t, off] : on_gpu) ggml_backend_tensor_set(t, (const char *) M.map + off, 0, ggml_nbytes(t));
    if (!on_cpu.empty()) {
        M.buf_map = ggml_backend_cpu_buffer_from_ptr(M.map, M.map_size);
        ggml_backend_buffer_set_usage(M.buf_map, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & [t, off] : on_cpu)
            if (ggml_backend_tensor_alloc(M.buf_map, t, (char *) M.map + off) != GGML_STATUS_SUCCESS) return "map alloc failed";
    }
    M.kv.init(kv_.ssize, (int) kv_.heads.size());
    M.kv.k.assign(1, nullptr); M.kv.v.assign(1, nullptr);
    ggml_init_params kp = { 4 * ggml_tensor_overhead(), nullptr, true };
    M.ctx_kv = ggml_init(kp);
    const int64_t n_gqa = (int64_t) H.head_dim * H.n_head_kv;
    M.kv.k[0] = ggml_new_tensor_3d(M.ctx_kv, GGML_TYPE_F16, n_gqa, M.kv.size, 1);
    M.kv.v[0] = ggml_new_tensor_3d(M.ctx_kv, GGML_TYPE_F16, n_gqa, M.kv.size, 1);
    ggml_format_name(M.kv.k[0], "cache_k_l%d", il); ggml_format_name(M.kv.v[0], "cache_v_l%d", il);
    M.buf_kv = ggml_backend_alloc_ctx_tensors_from_buft(M.ctx_kv, ggml_backend_get_default_buffer_type(M.gpu));
    if (!M.buf_kv) return "drafter KV allocation failed";
    ggml_backend_buffer_clear(M.buf_kv, 0);
    ggml_backend_t backs[2] = { M.gpu, cpu_ };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(M.gpu), lite::pinned_host_buffer_type(cpu_) };
    M.sched = ggml_backend_sched_new(backs, bufts, 2, kMaxNodes, false, true);
    mtp_graph_buf_.resize(graph_buf_.size());   // the drafter's graph lives beside the target's (both are kept)
    graph_drop(mg_);
    M.loaded = true;
    return "";
}

// draft-mtp's process(): the drafter re-reads a target batch at the same positions, each token paired with the target's
// hidden row of the position before (row 0: the pending row from the previous batch), in n_ubatch pieces (its context
// splits plainly). Only its KV cells matter here, so by default the graph stops at the K/V writes; Q35_MTP_FULL=1 runs
// the whole block as llama.cpp does (the cells come out the same)
std::string Engine::mtp_catchup(const int32_t * toks, int n, int pos0, const float * h_rows) {
    const int64_t ne = h_width();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> hs((size_t) n * ne);
    std::vector<float> & hp = mtp_h_v_[cur_seq_];
    std::copy(hp.begin(), hp.end(), hs.begin());
    if (n > 1) std::copy(h_rows, h_rows + (size_t) (n - 1) * ne, hs.begin() + ne);
    hp.assign(h_rows + (size_t) (n - 1) * ne, h_rows + (size_t) n * ne);
    static const bool full = getenv("Q35_MTP_FULL") != nullptr;
    static const int piece = getenv("Q35_MTP_CATCHUP_UB") ? std::max(1, atoi(getenv("Q35_MTP_CATCHUP_UB"))) : o_.n_ubatch;
    for (int i = 0; i < n; i += piece) {
        const int m = std::min(piece, n - i);
        int32_t dummy;
        const std::string err = mtp_ubatch(toks + i, hs.data() + (size_t) i * ne, m, pos0 + i, full, &dummy, nullptr, cur_seq_);
        if (!err.empty()) return "drafter: " + err;
    }
    stats_.catchup_tokens += n;
    stats_.catchup_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return "";
}

std::string Engine::draft(int32_t tok, const float * h, int pos, int32_t * out, std::vector<float> * h_out, int seq) {
    if (!mtp_.loaded) return "no MTP drafter loaded";
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    const auto t0 = std::chrono::steady_clock::now();
    const std::string err = mtp_ubatch(&tok, h, 1, pos, true, out, h_out, seq);
    stats_.drafts++;
    stats_.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return err;
}

// llama_model_qwen35moe::graph_mtp, node for node; out = the last row's logits (argmax) and post-norm hidden row
// the drafter's probability of its last pick (Strata's --spec-min-p reads it): 1 / sum(exp(l - max)), computed only when the
// gate asks; with the gate off (the default) a draft step no longer pays a softmax over the whole vocabulary
float Engine::draft_p() const {
    const auto & M = mtp_;
    if (M.last_p < 0) {
        if (M.last_lg.empty()) return 1.0f;
        const float mx = *std::max_element(M.last_lg.begin(), M.last_lg.end());
        double sum = 0; for (float v : M.last_lg) sum += std::exp((double) v - mx);
        M.last_p = sum > 0 ? (float) (1.0 / sum) : 0.0f;
    }
    return M.last_p;
}

std::string Engine::mtp_ubatch(const int32_t * toks, const float * h, int n, int pos0, bool out, int32_t * amax, std::vector<float> * h_out, int seq) {
    const HParams & H = hp_;
    Mtp & M = mtp_;
    const Layer & l = M.l;
    const int il = H.n_layer;   // the nextn block's index
    std::vector<int32_t> positions(n);
    for (int i = 0; i < n; ++i) positions[i] = pos0 + i;
    const std::vector<int8_t> seqs(n, (int8_t) seq);
    const int kst = M.kv.stream_of(seq);   // the sequence's KV stream
    M.kv.idxs.clear();
    if (!M.kv.find_slot(n, kst)) return "KV cache full";
    M.kv.apply(positions, seqs);
    M.kv.n_kv = M.kv.calc_n_kv(kst, 1);
    const uint32_t n_kv = M.kv.n_kv;
    const int64_t n_embd = H.n_embd, n_embd_head = H.head_dim, n_gqa = (int64_t) H.head_dim * H.n_head_kv, hw = h_width();
    const int n_outputs = out ? 1 : 0;
    // graph reuse, as the target's ubatch: a draft step after a draft step (the same shape) sets its inputs and computes
    const GraphKey key = { n, (int) n_kv, n_outputs, false, -1, 1, kst };
    const bool reuse = mg_.valid && mg_.key == key && graph_reuse_on();
    if (!reuse) {   // ---- build (to "// ---- build ends")
    graph_drop(mg_);
    ggml_init_params ip = { mtp_graph_buf_.size(), mtp_graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(M.sched);
    auto nm = [&](ggml_tensor * t, const char * s) { ggml_format_name(t, "%s-%d", s, il); };
    auto mnorm = [&](ggml_tensor * cur, ggml_tensor * w, int lay) {   // build_norm, pinned to the layer's device
        cur = ggml_rms_norm(ctx, cur, H.eps);
        if (lay >= 0) {
            ggml_format_name(cur, "norm-%d", lay);
            if (ggml_backend_supports_op(M.gpu, cur)) ggml_backend_sched_set_tensor_backend(M.sched, cur, M.gpu);
        } else ggml_set_name(cur, "norm");
        return ggml_mul(ctx, cur, w);
    };
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(inp_tokens);
    ggml_tensor * tok_embd = ggml_get_rows(ctx, M.tok_embd, inp_tokens);
    nm(tok_embd, "mtp_tok_embd");
    ggml_tensor * inp_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hw, n);
    ggml_set_input(inp_h); ggml_set_name(inp_h, "mtp_h_input");
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * 4);
    ggml_set_input(inp_pos);
    ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp_out_ids);
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n, 1, 1);
    ggml_set_input(kq_mask);

    ggml_tensor * cur = nullptr, * inpSA = nullptr, * res_hc = nullptr, * inject = nullptr;
    if (M.q4x) {   // the qwen4exp/mtp branch's graph_mtp: the 4-stream residual in, per stream concat(enorm(e), hnorm(h)) -> eh_proj
        // the hidden norm: one RMS over all 10240 values (vLLM's qwen4_exp MTP, Strata's default) or per stream (the
        // llama.cpp branch; Q35_MTP_HNORM=stream)
        static const bool per_stream = getenv("Q35_MTP_HNORM") && std::string(getenv("Q35_MTP_HNORM")) == "stream";
        ggml_tensor * h_norm = per_stream ? ggml_rms_norm(ctx, ggml_reshape_3d(ctx, inp_h, n_embd, H.hc, n), H.eps) : ggml_rms_norm(ctx, inp_h, H.eps);
        h_norm = ggml_mul(ctx, ggml_reshape_2d(ctx, h_norm, hw, n), M.hnorm);
        h_norm = ggml_reshape_3d(ctx, h_norm, n_embd, H.hc, n);
        nm(h_norm, "mtp_hnorm");
        ggml_tensor * e_norm = mnorm(tok_embd, M.enorm, il);
        e_norm = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, e_norm, n_embd, 1, n), n_embd, H.hc, n, 1);
        nm(e_norm, "mtp_enorm");
        ggml_tensor * concat = ggml_concat(ctx, e_norm, h_norm, 0);
        nm(concat, "mtp_concat");
        res_hc = mm_s(ctx, ggml_mul_mat(ctx, M.eh_proj, concat), M.eh_proj_s);
        nm(res_hc, "mtp_eh_proj");
        cur = hc_mix_build(ctx, H, res_hc, l.hc_an, l.hc_ad, l.hc_au, l.hc_ai, &inject, il);
        nm(cur, "mtp_hc_attn_pre");
    } else {
    ggml_tensor * h_norm = mnorm(inp_h, M.hnorm, il);
    nm(h_norm, "mtp_hnorm");
    ggml_tensor * e_norm = mnorm(tok_embd, M.enorm, il);
    nm(e_norm, "mtp_enorm");
    ggml_tensor * concat = ggml_concat(ctx, e_norm, h_norm, 0);
    nm(concat, "mtp_concat");
    cur = mm_s(ctx, ggml_mul_mat(ctx, M.eh_proj, concat), M.eh_proj_s);
    nm(cur, "mtp_eh_proj");
    inpSA = cur;
    cur = mnorm(cur, l.attn_norm, il);
    nm(cur, "mtp_attn_norm");
    }
    ggml_tensor * Qcur_full = mm_s(ctx, ggml_mul_mat(ctx, l.wq, cur), l.wq_s);   // build_qkv(..., reshape = false)
    ggml_tensor * Kcur = mm_s(ctx, ggml_mul_mat(ctx, l.wk, cur), l.wk_s);
    ggml_tensor * Vcur = mm_s(ctx, ggml_mul_mat(ctx, l.wv, cur), l.wv_s);
    nm(Qcur_full, "mtp_Qcur_full");
    int sections[4]; std::copy(H.rope_sections, H.rope_sections + 4, sections);
    ggml_tensor * Qcur = nullptr, * gate = nullptr;
    if (out) {
        Qcur = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n, ggml_element_size(Qcur_full) * n_embd_head * 2,
                            ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, 0);
        Qcur = mnorm(Qcur, l.q_norm, il);
        nm(Qcur, "mtp_Qcur_normed");
        gate = ggml_view_3d(ctx, Qcur_full, n_embd_head, H.n_head, n, ggml_element_size(Qcur_full) * n_embd_head * 2,
                            ggml_element_size(Qcur_full) * n_embd_head * 2 * H.n_head, ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx, gate, n_embd_head * H.n_head, n);
        nm(gate, "mtp_gate");
    }
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, H.n_head_kv, n);
    Kcur = mnorm(Kcur, l.k_norm, il);
    nm(Kcur, "mtp_Kcur_normed");
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, H.n_head_kv, n);
    nm(Vcur, "mtp_Vcur");
    if (out) Qcur = ggml_rope_multi(ctx, Qcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                                    kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
    Kcur = ggml_rope_multi(ctx, Kcur, inp_pos, nullptr, H.n_rot, sections, kRopeType, H.n_ctx_train, H.rope_base, 1.0f,
                           kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
    // build_attn (kv): q, v, k, then the cache writes
    if (out) ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Vcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, M.kv.k[0], ggml_view_2d(ctx, Kcur, n_gqa, n, Kcur->nb[2], 0), k_idxs));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, M.kv.v[0], ggml_view_2d(ctx, Vcur, n_gqa, n, Vcur->nb[2], 0), v_idxs));
    ggml_tensor * t_h = nullptr, * t_logits = nullptr;
    if (out) {
        const size_t kso = ggml_row_size(M.kv.k[0]->type, n_gqa * M.kv.ssize), vso = ggml_row_size(M.kv.v[0]->type, n_gqa * M.kv.ssize);
        ggml_tensor * k = ggml_view_4d(ctx, M.kv.k[0], n_embd_head, H.n_head_kv, n_kv, 1, ggml_row_size(M.kv.k[0]->type, n_embd_head),
                                       ggml_row_size(M.kv.k[0]->type, n_gqa), kso, (size_t) kst * kso);
        ggml_tensor * v = ggml_view_4d(ctx, M.kv.v[0], n_embd_head, H.n_head_kv, n_kv, 1, ggml_row_size(M.kv.v[0]->type, n_embd_head),
                                       ggml_row_size(M.kv.v[0]->type, n_gqa), vso, (size_t) kst * vso);
        ggml_tensor * q = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1, Qcur->nb[1], Qcur->nb[2], Qcur->nb[3], 0);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        cur = ggml_flash_attn_ext(ctx, q, ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, v, 0, 2, 1, 3), kq_mask,
                                  1.0f / sqrtf((float) n_embd_head), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(cur, nullptr);
        ggml_flash_attn_ext_set_n_kv_max(cur, 0);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        nm(cur, "mtp_attn_pregate");
        cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
        cur = mm_s(ctx, ggml_mul_mat(ctx, l.wo, cur), l.wo_s);
        nm(cur, "mtp_attn_out");
        ggml_tensor * ffn_residual = nullptr, * x = nullptr;
        if (M.q4x) {
            res_hc = hc_combine_build(ctx, H, res_hc, cur, inject, il);
            nm(res_hc, "mtp_hc_attn_post");
            x = hc_mix_build(ctx, H, res_hc, l.hc_fn, l.hc_fd, l.hc_fu, l.hc_fi, &inject, il);
            nm(x, "mtp_hc_ffn_pre");
        } else {
        cur = ggml_add(ctx, cur, inpSA);
        nm(cur, "mtp_attn_residual");
        ffn_residual = cur;
        cur = mnorm(cur, l.attn_post_norm, il);
        nm(cur, "mtp_attn_post_norm");
        x = cur;
        }
        if (M.dense) {   // llama_model_qwen35::graph_mtp: build_ffn(SILU, PAR), up, gate, swiglu, down
            ggml_tensor * up = mm_s(ctx, ggml_mul_mat(ctx, l.dense_up, x), l.dense_up_s);
            nm(up, "ffn_up");
            ggml_tensor * g = mm_s(ctx, ggml_mul_mat(ctx, l.dense_gate, x), l.dense_gate_s);
            nm(g, "ffn_gate");
            g = ggml_swiglu_split(ctx, g, up);
            nm(g, "ffn_swiglu");
            cur = mm_s(ctx, ggml_mul_mat(ctx, l.dense_down, g), l.dense_down_s);
            nm(cur, "ffn_down");
        } else {
        // build_moe_ffn (SILU, norm_w, SOFTMAX, separate gate/up), every expert on the GPU
        const int64_t n_expert = H.n_expert, n_used = H.n_expert_used;
        ggml_tensor * logits_r = ggml_mul_mat(ctx, l.gate_inp, x);
        nm(logits_r, "ffn_moe_logits");
        ggml_tensor * probs = ggml_soft_max(ctx, logits_r);
        nm(probs, "ffn_moe_probs");
        ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, (int) n_used);
        nm(selected->src[0], "ffn_moe_argsort"); nm(selected, "ffn_moe_topk");
        probs = ggml_reshape_3d(ctx, probs, 1, n_expert, n);
        ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
        nm(weights, "ffn_moe_weights");
        weights = ggml_reshape_2d(ctx, weights, n_used, n);
        ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
        nm(wsum, "ffn_moe_weights_sum");
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        nm(wsum, "ffn_moe_weights_sum_clamped");
        weights = ggml_div(ctx, weights, wsum);
        nm(weights, "ffn_moe_weights_norm");
        weights = ggml_reshape_3d(ctx, weights, 1, n_used, n);
        ggml_build_forward_expand(gf, weights);
        ggml_tensor * xr = ggml_reshape_3d(ctx, x, n_embd, 1, n);
        ggml_tensor * up = mm_id_s(ctx, ggml_mul_mat_id(ctx, l.up_exps, xr, selected), l.up_exps_s, selected, n);
        nm(up, "ffn_moe_up");
        ggml_tensor * g = mm_id_s(ctx, ggml_mul_mat_id(ctx, l.gate_exps, xr, selected), l.gate_exps_s, selected, n);
        nm(g, "ffn_moe_gate");
        g = ggml_swiglu_split(ctx, g, up);
        nm(g, "ffn_moe_swiglu");
        ggml_tensor * experts = mm_id_s(ctx, ggml_mul_mat_id(ctx, l.down_exps, g, selected), l.down_exps_s, selected, n);
        nm(experts, "ffn_moe_down");
        experts = ggml_mul(ctx, experts, weights);
        nm(experts, "ffn_moe_weighted");
        ggml_build_forward_expand(gf, experts);
        ggml_tensor * parts[64];
        for (int i = 0; i < n_used; ++i) {
            parts[i] = ggml_view_2d(ctx, experts, n_embd, n, experts->nb[2], i * experts->nb[1]);
            ggml_build_forward_expand(gf, parts[i]);
        }
        ggml_tensor * moe_out = parts[0];
        for (int i = 1; i < n_used; ++i) { moe_out = ggml_add(ctx, moe_out, parts[i]); ggml_build_forward_expand(gf, moe_out); }
        nm(moe_out, "mtp_ffn_moe_out");
        if (l.up_shexp) {   // build_ffn(SILU, PAR) + the sigmoid gate
            ggml_tensor * tmp = mm_s(ctx, ggml_mul_mat(ctx, l.up_shexp, x), l.up_shexp_s);
            nm(tmp, "ffn_up");
            ggml_tensor * gs = mm_s(ctx, ggml_mul_mat(ctx, l.gate_shexp, x), l.gate_shexp_s);
            nm(gs, "ffn_gate");
            gs = ggml_swiglu_split(ctx, gs, tmp);
            nm(gs, "ffn_swiglu");
            ggml_tensor * shexp = mm_s(ctx, ggml_mul_mat(ctx, l.down_shexp, gs), l.down_shexp_s);
            nm(shexp, "mtp_ffn_shexp");
            ggml_tensor * sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, l.gate_inp_shexp, x));
            nm(sg, "mtp_shared_expert_gate_sigmoid");
            shexp = ggml_mul(ctx, shexp, sg);
            nm(shexp, "mtp_ffn_shexp_gated");
            cur = ggml_add(ctx, moe_out, shexp);
        } else cur = moe_out;
        }
        nm(cur, "mtp_ffn_out");
        if (M.q4x) {   // the next draft step re-enters with the wide residual; the block's own mixer is the head norm
            res_hc = hc_combine_build(ctx, H, res_hc, cur, inject, il);
            ggml_set_name(res_hc, "h_nextn"); ggml_set_output(res_hc);
            t_h = res_hc;
            cur = hc_mix_build(ctx, H, res_hc, M.hc_head_norm, M.hc_head_down, M.hc_head_up, nullptr, nullptr, -1);
            ggml_set_name(cur, "mtp_hc_head");
        } else {
        cur = ggml_add(ctx, cur, ffn_residual);
        nm(cur, "mtp_post_ffn");
        cur = mnorm(cur, M.head_norm, -1);
        ggml_set_name(cur, "h_nextn"); ggml_set_output(cur);
        t_h = cur;
        }
        cur = ggml_get_rows(ctx, cur, inp_out_ids);
        ggml_set_name(cur, "mtp_shared_head_norm");
        cur = mm_s(ctx, ggml_mul_mat(ctx, M.output, cur), M.output_s);
        ggml_set_name(cur, "result_output"); ggml_set_output(cur);
        t_logits = cur;
        ggml_build_forward_expand(gf, cur);
    }
    if (!ggml_backend_sched_alloc_graph(M.sched, gf)) { ggml_free(ctx); return "graph allocation failed"; }
    mg_.ctx = ctx; mg_.gf = gf; mg_.key = key; mg_.valid = true;
    mg_.inp_tokens = inp_tokens; mg_.inp_h = inp_h; mg_.inp_pos = inp_pos; mg_.k_idxs = k_idxs; mg_.v_idxs = v_idxs; mg_.kq_mask = kq_mask;
    mg_.inp_out_ids = inp_out_ids; mg_.t_logits = t_logits; mg_.t_h = t_h;
    stats_.graphs_built++;
    } else stats_.graphs_reused++;   // ---- build ends
    const Graph & G = mg_;
    ggml_backend_tensor_set(G.inp_tokens, toks, 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(G.inp_h, h, 0, (size_t) n * hw * sizeof(float));
    {
        std::vector<int32_t> p4((size_t) n * 4);
        for (int i = 0; i < n; ++i) { p4[i] = positions[i]; p4[n + i] = positions[i]; p4[2 * n + i] = positions[i]; p4[3 * n + i] = 0; }
        ggml_backend_tensor_set(G.inp_pos, p4.data(), 0, p4.size() * sizeof(int32_t));
    }
    ggml_backend_tensor_set(G.k_idxs, M.kv.idxs.data(), 0, n * sizeof(int64_t));
    ggml_backend_tensor_set(G.v_idxs, M.kv.idxs.data(), 0, n * sizeof(int64_t));
    if (out) {   // the K/V-only graph has no attention; the causal mask as the target's (lite/mask_fill.hpp)
        std::vector<ggml_fp16_t> md((size_t) n_kv * n);
        std::vector<int32_t> lo(n, 0);
        lite::fill_mask(md.data(), n, n_kv, M.kv.pos.data() + (size_t) kst * M.kv.ssize, lo.data(), positions.data(), o_.n_threads);
        ggml_backend_tensor_set(G.kq_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    }
    if (n_outputs) { const int32_t oid = n - 1; ggml_backend_tensor_set(G.inp_out_ids, &oid, 0, sizeof(int32_t)); }
    const ggml_status st = ggml_backend_sched_graph_compute(M.sched, G.gf);
    if (st != GGML_STATUS_SUCCESS) { graph_drop(mg_); return "compute failed"; }
    if (out) {
        std::vector<float> lg((size_t) M.output->ne[1]);
        ggml_backend_tensor_get(G.t_logits, lg.data(), 0, lg.size() * sizeof(float));
        if (M.d2t.empty()) *amax = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        else {   // the patch's full-vocabulary logits are -inf off d2t: their argmax = the lowest target id at the draft maximum
            const float mx = *std::max_element(lg.begin(), lg.end());
            int32_t best = INT32_MAX;
            for (size_t i = 0; i < lg.size(); ++i) if (lg[i] == mx) best = std::min(best, M.d2t[i]);
            *amax = best;
        }
        M.last_lg.swap(lg); M.last_p = -1.0f;   // draft_p() computes the probability only when asked (it cost every token ~250K exp)
        if (h_out) {
            h_out->resize((size_t) hw);
            ggml_backend_tensor_get(G.t_h, h_out->data(), (size_t) (n - 1) * hw * sizeof(float), (size_t) hw * sizeof(float));
        }
    }
    return "";
}

} // namespace q35
