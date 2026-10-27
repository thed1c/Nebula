// Impulse gemma4 engine, Gate 1 stage b. See g4_engine.hpp. Every graph-building step below names the llama.cpp
// function it mirrors (b6b003d: src/models/gemma4.cpp, src/llama-graph.cpp, src/llama-kv-cache.cpp); the order of the
// ggml calls is kept, because the CUDA backend fuses adjacent nodes and the scheduler splits by node order.
#include "g4_engine.hpp"
#include "g4_tsplit.hpp"

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml-cpu.h"

#include <cuda_runtime.h>
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../lite/platform.hpp"
#include "../lite/ram_tier.hpp"
#include "../lite/pinned_host.hpp"
#include "../lite/mask_fill.hpp"

namespace g4 {

static constexpr int   kMaxNodes   = 8192;
// rope parameters as llama.cpp's cparams resolve them for this model (checked against its graph's op_params)
static constexpr int   kRopeType   = GGML_ROPE_TYPE_NEOX;
static constexpr float kExtFactor  = 0.0f;
static constexpr float kAttnFactor = 1.0f;
static constexpr float kBetaFast   = 32.0f;
static constexpr float kBetaSlow   = 1.0f;
// ggml-cuda offloads an op on CPU-resident weights to the GPU from this many tokens (ggml_backend_cuda_device_offload_op):
// below it llama.cpp computes the experts on the CPU, and so does the pool; above it both keep ggml's (offloaded) path
static constexpr int   kOffloadMin = 32;
// graph reuse (v0.2.3) is on unless IMPULSE_GRAPH_REUSE=0 (llama.cpp: LLAMA_GRAPH_REUSE_DISABLE) or a tensor dump is
// asked for (the dumps hook a build)
static bool graph_reuse_on() {
    static const bool on = !(getenv("IMPULSE_GRAPH_REUSE") && atoi(getenv("IMPULSE_GRAPH_REUSE")) == 0)
                        && !getenv("TDUMP") && !getenv("GDUMP") && !getenv("G4_SCHED_DUMP");
    return on;
}
// the slot path's misses all point at one zero slot, so a token's ids repeat; ggml-cuda's mat-vec MoE kernel (<= 8 tokens)
// takes that, its batched MMQ kernel (more tokens) assumes distinct ids per token and overruns. Ubatches of 9-31 tokens
// (short prompts, prompt tails) therefore go through the pool for every expert (stage c), exactly as llama.cpp's CPU MoE.
static constexpr int   kSlotMaxTokens = 8;
static constexpr int   kMinSlots = 8;          // below this the arena is not worth its VRAM
// A small step's compute buffer grows with the context (a 4-token verify window at 157k cells took 620 MiB on sm_86/sm_120,
// ~4.1 KB per cell); ggml's allocator crashes rather than fails when it cannot get it, so the room is made beforehand.
static constexpr size_t kComputePerCell = 4608;
static constexpr int    kHeadroomStep   = 8192;   // cells of context growth covered per check
// Auto chunk: the largest of 32k / 16k / 8k / 4k whose compute buffers (read_compute_bytes) fit what the read can borrow.
static constexpr int    kChunkMin    = 4096;
static constexpr int    kMaxSeqs     = 8;      // server slots sharing the KV (Options::n_seq)

namespace scpu = strata::kernels::cpu;

// llama-kv-cache.cpp ggml_gen_hadamard: the orthonormal Walsh-Hadamard matrix (H*H = I), n a power of two
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
// build_input_k_rot: the largest power of two (from 64) dividing the cache's head size; V always 64
static int k_rot_size(int head) { int r = 64; do { r *= 2; } while (head % r == 0); return r / 2; }

// Strata's ExpertPool is not reentrant: the overlap driver thread and a pool op on the host thread (the twin check's
// all-CPU twin) take turns on it
static std::mutex g_pool_mu;

// ---------------------------------------------------------------- stage c: the MoE layer through Strata's pool
struct Engine::PoolLayer {
    scpu::NativeFmt f;
    const uint8_t * gu = nullptr;  size_t gu_stride = 0;   // expert e's gate+up rows: gu + e*gu_stride (fused tensor)
    const uint8_t * dn = nullptr;  size_t d_stride = 0;    // its down rows
    const float * scale = nullptr;                         // ffn_down_exps.scale, one F32 per expert
    scpu::ExpertPool * pool = nullptr;
    const int * res = nullptr;                             // stage d: slot of each expert, -1 = not resident (nullptr: none)
    Stats * stats = nullptr;
    // scratch, reused every call
    std::vector<uint8_t> act;                              // n_tokens quantized activations
    std::vector<scpu::ExpertJobMulti> jobs;
    // overlap: the misses' rows, written by the pool on the driver thread, copied out by the join op
    std::vector<float> out;
    std::mutex mu; std::condition_variable cv; bool done = true;
    struct Engine::Driver * driver = nullptr;
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
            const auto tp0 = std::chrono::steady_clock::now();
            { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
            if (L->stats) L->stats->t_pool += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
            { std::lock_guard<std::mutex> g(L->mu); L->done = true; }
            L->cv.notify_all();
        }
    }
};

// overlap, start: the routing and the activations are here (the scheduler synced the router for them); quantize, plan
// the misses into L->out, hand them to the driver thread and return at once. dst is a 1-element token for ordering.
static void moe_start_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth; (void) dst;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1];
    const int64_t n = x->ne[1], k = ids->ne[0], ne = x->ne[0];
    const size_t ab = L->f.act_bytes;
    L->act.resize((size_t) n * ab);
    L->out.assign((size_t) n * k * ne, 0.0f);
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
                jb.blob = L->gu + (size_t) e * L->gu_stride;
                jb.down = L->dn + (size_t) e * L->d_stride;
                L->jobs.push_back(jb);
                ji = job_of[e] = (int) L->jobs.size() - 1;
            }
            auto & jb = L->jobs[ji];
            jb.nact[jb.nt] = L->act.data() + t * ab;
            jb.out[jb.nt] = L->out.data() + ((size_t) t * k + j) * ne;
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
    L->stats->t_join_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    memcpy(dst->data, L->out.data(), ggml_nbytes(dst));
}

// x [n_embd, n] f32, ids [k, n] i32 (argsort_top_k view) -> dst [n_embd, k, n]: exactly what build_moe_ffn's
// mul_mat_id (gate_up) -> geglu -> mul_mat_id (down) gives on ggml-cpu (the part llama.cpp runs on the CPU for a
// RAM-resident layer below the offload size), with the same quantizers, dot products and GELU table. The per-expert
// scale, the router weights and the sum over k stay ggml ops after it, where the scheduler put them in stage b (GPU).
static void moe_pool_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    auto * L = (Engine::PoolLayer *) ud;
    const ggml_tensor * x = dst->src[0], * ids = dst->src[1];
    const int64_t n = x->ne[1], k = ids->ne[0];
    const size_t ab = L->f.act_bytes;
    L->act.resize((size_t) n * ab);
    for (int64_t t = 0; t < n; ++t)
        scpu::native_quant_act(L->f, (const float *) ((const char *) x->data + t * x->nb[1]), L->act.data() + t * ab);
    auto id_at = [&](int64_t t, int64_t j) { return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + j * ids->nb[0]); };
    // one job per (expert, up to MAXT of its tokens), experts in order of first use
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
                jb.blob = L->gu + (size_t) e * L->gu_stride;
                jb.down = L->dn + (size_t) e * L->d_stride;
                L->jobs.push_back(jb);
                ji = job_of[e] = (int) L->jobs.size() - 1;
            }
            auto & jb = L->jobs[ji];
            jb.nact[jb.nt] = L->act.data() + t * ab;
            jb.out[jb.nt] = (float *) ((char *) dst->data + t * dst->nb[2] + j * dst->nb[1]);
            jb.nt++;
        }
    }
    const auto tp0 = std::chrono::steady_clock::now();
    if (!L->jobs.empty()) { std::lock_guard<std::mutex> g(g_pool_mu); L->pool->run_split_multi_native(L->f, L->jobs.data(), (int) L->jobs.size()); }
    if (L->stats) L->stats->t_pool += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
}

struct Engine::CacheLayer {
    bool full = false;                  // every expert resident, identity order: exactly llama.cpp's GPU-layer graph
    int  S = 0;                         // resident slots (the arena has S + 1: slot S is all zeros, where misses point)
    ggml_tensor * gu = nullptr, * dn = nullptr, * res_f = nullptr, * scale = nullptr;
    std::vector<int> res;               // expert -> slot, -1 = in RAM only
    std::vector<int> owner;             // slot -> expert (stage e)
    std::vector<double> score;          // decayed usage counts (stage e)
    std::vector<char> pending;          // expert with a copy in flight (async swaps)
    std::unique_ptr<PoolLayer> miss;    // the pool for the misses (res set)
    ggml_tensor * gu_all = nullptr, * dn_all = nullptr;   // dcheck twin: the whole layer on the GPU
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
    std::vector<int64_t> r;
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return r;
    const size_t n = gguf_get_arr_n(g, id);
    const void * d = gguf_get_arr_data(g, id);
    switch (gguf_get_arr_type(g, id)) {
        case GGUF_TYPE_BOOL: case GGUF_TYPE_INT8: case GGUF_TYPE_UINT8:
            for (size_t i = 0; i < n; ++i) r.push_back(((const int8_t *) d)[i]); break;
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) r.push_back(((const int32_t *) d)[i]); break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) r.push_back(((const uint32_t *) d)[i]); break;
        default: break;
    }
    return r;
}

// ---------------------------------------------------------------- KV cells (llama_kv_cache, one sequence)
void Engine::Cache::init(uint32_t cells_per_stream, int n_streams) {
    ssize = cells_per_stream; size = cells_per_stream * (uint32_t) n_streams;
    pos.assign(size, -1); seq.assign(size, (int8_t) -1); heads.assign(n_streams, 0u); n_kv = 0;
}
uint32_t Engine::Cache::used(int st) const {
    return (uint32_t) std::count_if(pos.begin() + (size_t) st * ssize, pos.begin() + (size_t) (st + 1) * ssize, [](int32_t p) { return p >= 0; });
}
uint32_t Engine::Cache::used_max_p1(int st) const {
    for (int64_t i = (int64_t) ssize - 1; i >= 0; --i) if (pos[(size_t) st * ssize + i] >= 0) return (uint32_t) i + 1;
    return 0;
}
int32_t Engine::Cache::pos_min(int s) const {
    int32_t m = INT32_MAX; bool any = false;
    for (size_t i = 0; i < pos.size(); ++i) if (pos[i] >= 0 && seq[i] == s) { m = std::min(m, pos[i]); any = true; }
    return any ? m : -1;
}
int32_t Engine::Cache::pos_max(int s) const {
    int32_t m = -1;
    for (size_t i = 0; i < pos.size(); ++i) if (pos[i] >= 0 && seq[i] == s) m = std::max(m, pos[i]);
    return m;
}
// llama_kv_cache::find_slot(ubatch, cont = false), in one stream (its cells hold one sequence)
bool Engine::Cache::find_slot(uint32_t n_tokens, int n_swa, int st) {
    const size_t base = (size_t) st * ssize, want = idxs.size() + n_tokens;
    uint32_t head_cur = heads[st];
    if (head_cur > used(st) + 2 * n_tokens) head_cur = 0;
    if (n_tokens > ssize) return false;
    int32_t seq_max = -1;   // llama_kv_cells::seq_pos_max of the stream's sequence
    for (uint32_t i = 0; i < ssize; ++i) seq_max = std::max(seq_max, pos[base + i]);
    uint32_t n_tested = 0;
    while (true) {
        if (head_cur + 1 > ssize) { n_tested += ssize - head_cur; head_cur = 0; continue; }
        const uint32_t idx = head_cur++;
        n_tested++;
        bool can_use = pos[base + idx] < 0;
        if (!can_use && swa && (seq_max + 1) - pos[base + idx] >= n_swa) can_use = true;   // is_masked_swa(STANDARD)
        if (can_use) idxs.push_back((int64_t) (base + idx));
        if (idxs.size() == want) break;
        if (n_tested >= ssize) return false;
    }
    return true;
}
// llama_kv_cache::seq_rm(seq, p0, -1): free every cell at p0 or later; the first freed cell of a stream becomes its head if lower
void Engine::Cache::seq_rm(int s, int32_t p0) {
    for (int st = 0; st < (int) heads.size(); ++st) {
        uint32_t new_head = ssize;
        for (uint32_t i = 0; i < ssize; ++i) {
            const size_t c = (size_t) st * ssize + i;
            if (pos[c] >= p0 && (s < 0 || seq[c] == s)) { pos[c] = -1; seq[c] = -1; if (new_head == ssize) new_head = i; }
        }
        if (new_head != ssize && new_head < heads[st]) heads[st] = new_head;
    }
}
uint32_t Engine::Cache::calc_n_kv(int st0, int nst) const {
    const uint32_t pad = 256;
    uint32_t m = 0;
    for (int st = st0; st < st0 + nst; ++st) { const uint32_t um = used_max_p1(st); m = std::max(m, std::min(ssize, std::max(pad, (um + pad - 1) / pad * pad))); }
    return m;
}

// llama_kv_cache::apply_ubatch
void Engine::Cache::apply(const std::vector<int32_t> & positions, const std::vector<int8_t> & seqs) {
    int32_t max_rm[kMaxSeqs];   // per sequence: the highest position of a cell this ubatch took over
    for (int q = 0; q < kMaxSeqs; ++q) max_rm[q] = -1;
    for (size_t i = 0; i < idxs.size(); ++i) {
        const int64_t idx = idxs[i];
        if (pos[idx] >= 0) { max_rm[seq[idx]] = std::max(max_rm[seq[idx]], pos[idx]); pos[idx] = -1; seq[idx] = -1; }
        pos[idx] = positions[i]; seq[idx] = seqs[i];
        heads[idx / ssize] = (uint32_t) (idx % ssize) + 1;
    }
    // the invariant llama.cpp keeps: every position in [pos_min, pos_max] of a sequence is present
    for (int q = 0; q < kMaxSeqs; ++q) {
        if (max_rm[q] < 0) continue;
        const int32_t pmin = pos_min(q);
        if (pmin >= 0 && pmin <= max_rm[q]) for (size_t i = 0; i < pos.size(); ++i) if (seq[i] == q && pos[i] >= pmin && pos[i] < max_rm[q] + 1) { pos[i] = -1; seq[i] = -1; }
    }
}

// ---------------------------------------------------------------- load
Engine::Engine() = default;

Engine::~Engine() {
    for (auto * b : lora_buf_) if (b) ggml_backend_buffer_free(b);
    for (auto * c : lora_ctx_) if (c) ggml_free(c);
    graph_drop(dg_);
    if (mtp_.sched) ggml_backend_sched_free(mtp_.sched);
    if (mtp_.own_backend && mtp_.backend) ggml_backend_free(mtp_.backend);
    driver_.reset();
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); cudaStreamDestroy((cudaStream_t) copy_stream_); }
    for (auto & b : pending_) cudaEventDestroy((cudaEvent_t) b.ev);
    if (buf_arena_) ggml_backend_buffer_free(buf_arena_);
    lite::ram_tier_free(arena_, arena_size_, arena_placed_);
    graph_drop(g_); graph_drop(dg_);
    if (sched_) ggml_backend_sched_free(sched_);
    if (ctx_kvfull_) ggml_free(ctx_kvfull_);
    for (auto * b : buf_swa_) if (b) ggml_backend_buffer_free(b);
    for (auto * c : ctx_swa_) if (c) ggml_free(c);
    for (KvLayer & kl : kvl_) { if (kl.buf) ggml_backend_buffer_free(kl.buf); if (kl.ctx) ggml_free(kl.ctx); }
    if (buf_gpu_) ggml_backend_buffer_free(buf_gpu_);
    for (auto * b : buf_gpu_x_) if (b) ggml_backend_buffer_free(b);
    if (buf_map_) ggml_backend_buffer_free(buf_map_);
    if (ctx_w_) ggml_free(ctx_w_);
    if (meta_) ggml_free(meta_);
    if (gguf_) gguf_free(gguf_);
    for (size_t d = 1; d < gpus_.size(); ++d) ggml_backend_free(gpus_[d]);
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
    lite::unmap_file(map_, map_size_);
}

std::string Engine::load(const Options & o) {
    o_ = o;
    hp_ = HParams{};   // a dense model reloads with the expert options off: start the hparams again
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta_ };
    gguf_ = gguf_init_from_file(o.gguf.c_str(), gp);
    if (!gguf_) return "cannot read " + o.gguf;
    const std::string A = "gemma4.";
    hp_.n_layer       = (int) kv_int(gguf_, A + "block_count");
    hp_.n_embd        = (int) kv_int(gguf_, A + "embedding_length");
    hp_.n_head        = (int) kv_int(gguf_, A + "attention.head_count");
    hp_.n_ff          = (int) kv_int(gguf_, A + "feed_forward_length");
    hp_.n_ff_exp      = (int) kv_int(gguf_, A + "expert_feed_forward_length");
    hp_.n_expert      = (int) kv_int(gguf_, A + "expert_count");
    hp_.n_expert_used = (int) kv_int(gguf_, A + "expert_used_count");
    hp_.n_swa         = (int) kv_int(gguf_, A + "attention.sliding_window");
    hp_.head_dim      = (int) kv_int(gguf_, A + "attention.key_length");
    hp_.head_dim_swa  = (int) kv_int(gguf_, A + "attention.key_length_swa");
    hp_.n_rot         = (int) kv_int(gguf_, A + "rope.dimension_count");
    hp_.n_rot_swa     = (int) kv_int(gguf_, A + "rope.dimension_count_swa");
    hp_.n_ctx_train   = (int) kv_int(gguf_, A + "context_length");
    hp_.rope_base     = kv_f32(gguf_, A + "rope.freq_base");
    hp_.rope_base_swa = kv_f32(gguf_, A + "rope.freq_base_swa");
    hp_.eps           = kv_f32(gguf_, A + "attention.layer_norm_rms_epsilon");
    hp_.softcap       = kv_f32(gguf_, A + "final_logit_softcapping");
    for (auto v : kv_arr(gguf_, A + "attention.head_count_kv")) hp_.n_head_kv.push_back((int) v);
    for (auto v : kv_arr(gguf_, A + "attention.sliding_window_pattern")) hp_.is_swa.push_back(v != 0);
    {
        const int64_t id = gguf_find_key(gguf_, "tokenizer.ggml.tokens");
        hp_.n_vocab = id < 0 ? 0 : (int) gguf_get_arr_n(gguf_, id);
    }
    if (hp_.n_layer <= 0 || (int) hp_.n_head_kv.size() != hp_.n_layer || (int) hp_.is_swa.size() != hp_.n_layer)
        return "unexpected gemma4 hparams";
    // v0.2.4: a dense Gemma 4 (31B: no expert_count, every layer llama.cpp's dense FFN). Nothing to cache, pool or swap: the
    // expert paths are turned off and the model loads as llama.cpp's all-GPU path does.
    if (hp_.n_expert <= 0) {
        hp_.n_expert = hp_.n_expert_used = hp_.n_ff_exp = 0;
        if (o.pool || o.cache_slots != 0 || o.adapt || o.async_swaps || o.overlap || o.lend || o.n_cpu_moe > 0) {
            Options d = o;
            d.pool = false; d.cache_slots = 0; d.adapt = false; d.async_swaps = false; d.overlap = false; d.lend = false; d.n_cpu_moe = 0;
            fprintf(stderr, "impulse: a dense model (no experts): the expert cache, CPU pool, swaps and lending are off\n");
            gguf_free(gguf_); gguf_ = nullptr;
            ggml_free(meta_); meta_ = nullptr;
            return load(d);
        }
    }

    // how the host waits for the card: the driver's default, unless IMPULSE_GPU_WAIT=spin (see src/qwen35/q35_engine.cpp)
    if (getenv("IMPULSE_GPU_WAIT") && std::string(getenv("IMPULSE_GPU_WAIT")) == "spin") {
        const cudaError_t fe = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        if (fe != cudaSuccess) { (void) cudaGetLastError(); fprintf(stderr, "impulse: cudaSetDeviceFlags(spin): %s (the driver's default wait stays)\n", cudaGetErrorString(fe)); }
        else fprintf(stderr, "impulse: the host spins while the card computes (IMPULSE_GPU_WAIT=spin)\n");
    }
    // an expert model over several cards (llama.cpp's -sm layer -ts with --n-cpu-moe): each card holds its layers' experts, the
    // first --n-cpu-moe layers' experts stay in RAM; ggml's own expert path (the expert cache, CPU pool, swaps and lending are one
    // card's), and reads in -ub steps (auto chunks stream experts through one card)
    if (hp_.n_expert > 0 && o.split_tensor && o.n_cpu_moe > 0)
        return "--split-mode tensor with an expert model: every expert on the cards (--n-cpu-moe 0)";
    if (hp_.n_expert > 0 && (o.tensor_split.size() > 1 || o.split_tensor) &&
        (o.pool || o.cache_slots != 0 || o.adapt || o.async_swaps || o.overlap || o.lend || o.prefill_ubatch < 0)) {
        Options d = o;
        d.pool = false; d.cache_slots = 0; d.adapt = false; d.async_swaps = false; d.overlap = false; d.lend = false;
        d.prefill_ubatch = std::max(0, o.prefill_ubatch);
        fprintf(stderr, "impulse: an expert model over several GPUs: ggml's expert path (the expert cache, CPU pool, swaps and lending are off)\n");
        gguf_free(gguf_); gguf_ = nullptr;
        ggml_free(meta_); meta_ = nullptr;
        return load(d);
    }
    // backends: [CUDA0, CPU] and the CPU's compute buffers in the GPU's pinned host memory, as llama_context does
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) return "no GPU device";
    // several GPUs (llama_model::load_tensors' split points): the visible GPUs in order, one share each; a card with share 0
    // gets no layers (and no backend here: llama.cpp's projector-only card)
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) devs.push_back(ggml_backend_dev_get(i));
    dev_layer_.assign(hp_.n_layer, 0); dev_out_ = 0;
    std::vector<int> back_of(devs.size(), -1);   // device index -> backend index (gpus_)
    if (o.split_tensor) {   // llama_prepare_model_devices: one meta device over the GPUs, the split states from g4_tsplit
        if (o.tensor_split.size() > devs.size()) return "--tensor-split names " + std::to_string(o.tensor_split.size()) + " GPUs; " + std::to_string(devs.size()) + " visible";
        std::vector<ggml_backend_dev_t> tdevs(devs);
        std::vector<float> ts;   // a card with share 0 takes no part (llama.cpp's projector-only card)
        if (!o.tensor_split.empty()) {
            tdevs.clear();
            for (size_t i = 0; i < o.tensor_split.size(); ++i) if (o.tensor_split[i] > 0) { tdevs.push_back(devs[i]); ts.push_back(o.tensor_split[i]); }
        }
        tp_info_ = std::make_unique<TsplitInfo>();
        tp_info_->hp = &hp_; tp_info_->meta = meta_;
        tp_info_->ts = ts.empty() ? std::vector<float>(tdevs.size(), 0.0f) : ts;
        ggml_backend_dev_t meta_dev = ggml_backend_meta_device(tdevs.data(), tdevs.size(), tsplit_state, tp_info_.get());
        gpu_ = ggml_backend_dev_init(meta_dev, nullptr);
        if (!gpu_) return "backend init failed (the meta device)";
        gpus_ = { gpu_ };
        tp_ = true;
        // the KV pager and the window cache's resize move K/V by device copies, which a meta tensor cannot take
        o_.kv_initial = 0;
        std::string m;
        for (size_t d = 0; d < tdevs.size(); ++d) m += (d ? ", " : "") + std::string(ggml_backend_dev_name(tdevs[d]));
        fprintf(stderr, "impulse: tensor split over %zu GPUs (%s): every layer on every card\n", tdevs.size(), m.c_str());
    } else if (o.tensor_split.size() > 1) {
        if (o.tensor_split.size() > devs.size()) return "--tensor-split names " + std::to_string(o.tensor_split.size()) + " GPUs; " + std::to_string(devs.size()) + " visible";
        if (!o.tensor_split.empty() && o.tensor_split[0] <= 0) return "--tensor-split: the first card needs a share (it holds the engine's own buffers)";
        std::vector<float> splits(o.tensor_split);
        float sum = 0; for (float & x : splits) { sum += x; x = sum; }
        for (float & x : splits) x /= sum;
        auto dev_of = [&](int il) { return (int) (std::upper_bound(splits.begin(), splits.end(), float(il) / (hp_.n_layer + 1)) - splits.begin()); };
        for (size_t i = 0; i < o.tensor_split.size(); ++i) {
            if (o.tensor_split[i] <= 0) continue;
            back_of[i] = (int) gpus_.size();
            gpus_.push_back(ggml_backend_dev_init(devs[i], nullptr));
            if (!gpus_.back()) return "backend init failed (GPU " + std::to_string(i) + ")";
        }
        for (int il = 0; il < hp_.n_layer; ++il) dev_layer_[il] = back_of[dev_of(il)];
        dev_out_ = back_of[dev_of(hp_.n_layer)];
        gpu_ = gpus_[0];
        std::string m;
        for (size_t d = 0; d < gpus_.size(); ++d) {
            int n = 0; for (int x : dev_layer_) n += x == (int) d;
            m += (d ? ", " : "") + std::string(ggml_backend_dev_name(ggml_backend_get_device(gpus_[d]))) + " " + std::to_string(n) + " layers";
        }
        fprintf(stderr, "impulse: layers split over %zu GPUs: %s; the output on GPU %d\n", gpus_.size(), m.c_str(), dev_out_);
    } else {
        gpu_ = ggml_backend_dev_init(gpu_dev, nullptr);
        gpus_ = { gpu_ };
    }
    cudaGetDevice(&cuda_dev_);   // the first GPU device = CUDA device 0 of the visible set
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu_ || !cpu_) return "backend init failed";
    ggml_backend_cpu_set_n_threads(cpu_, o.n_threads);

    // the file, mapped: the CPU-resident weights are used in place (llama.cpp with mmap: no repacking)
    map_ = lite::map_file(o.gguf, &map_size_);
    if (!map_) return "cannot map " + o.gguf;
    const size_t data_off = gguf_get_data_offset(gguf_);

    // which tensors live where: `-ngl 99 --n-cpu-moe N` = token_embd and the first N layers' experts in RAM, the rest
    // (including the tied output copy of token_embd) on the GPU
    const int n_tensors = (int) gguf_get_n_tensors(gguf_);
    ggml_init_params ip = { (size_t) (n_tensors + 4) * ggml_tensor_overhead(), nullptr, true };
    ctx_w_ = ggml_init(ip);
    std::vector<ggml_context *> ctx_gpus(gpus_.size());
    for (auto & c : ctx_gpus) c = ggml_init(ip);
    ggml_context * ctx_gpu = ctx_gpus[0];
    int mk_dev = 0;   // the card make() puts a GPU tensor on (the layer's, set in the loop below)
    std::vector<std::pair<ggml_tensor *, size_t>> on_cpu, on_gpu;   // tensor, file offset
    auto make = [&](const std::string & name, bool cpu, const char * as = nullptr) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(meta_, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(cpu ? ctx_w_ : ctx_gpus[mk_dev], m);
        ggml_set_name(t, as ? as : name.c_str());
        const size_t off = data_off + gguf_get_tensor_offset(gguf_, gguf_find_tensor(gguf_, name.c_str()));
        (cpu ? on_cpu : on_gpu).push_back({ t, off });
        return t;
    };
    auto is_exps = [&](int il) { return il < o.n_cpu_moe; };
    tok_embd_    = make("token_embd.weight", true);
    mk_dev = dev_out_;
    output_      = make("output.weight", false);                 // the 31B has its own; the 26B ties it to token_embd
    const bool output_tied = !output_;
    if (!output_) output_ = make("token_embd.weight", false);
    // a weight's optional second-level scale (llama.cpp's generic pass: "<weight>.scale", TENSOR_NOT_REQUIRED): beside a
    // weight that is there, F32 [1], on the weight's card
    std::string scale_err;
    int n_scales = 0;
    auto make_s = [&](const ggml_tensor * w, const std::string & name) -> ggml_tensor * {
        if (!w) return nullptr;
        const ggml_tensor * m = ggml_get_tensor(meta_, name.c_str());
        if (!m) return nullptr;
        if (m->type != GGML_TYPE_F32 || ggml_nelements(m) != 1) { if (scale_err.empty()) scale_err = name + ": expected F32 [1]"; return nullptr; }
        n_scales++;
        return make(name, false);
    };
    if (output_ && output_->type == GGML_TYPE_NVFP4) {   // llama.cpp: only beside an NVFP4 output; a tied one may not have it
        output_s_ = make_s(output_, "output.scale");
        if (output_s_ && output_tied) return "output.scale beside a tied NVFP4 output (llama.cpp refuses it)";
    }
    output_norm_ = make("output_norm.weight", false);
    rope_freqs_d_.assign(gpus_.size(), nullptr);   // one copy per card, as llama.cpp's per-layer TENSOR_DUPLICATED
    for (size_t d = 0; d < gpus_.size(); ++d) { mk_dev = (int) d; rope_freqs_d_[d] = make("rope_freqs.weight", false); }
    rope_freqs_ = rope_freqs_d_[0];
    L_.resize(hp_.n_layer);
    for (int il = 0; il < hp_.n_layer; ++il) {
        mk_dev = dev_layer_[il];
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        Layer & l = L_[il];
        l.attn_norm      = make(b("attn_norm.weight"), false);
        l.wq             = make(b("attn_q.weight"), false);
        l.wk             = make(b("attn_k.weight"), false);
        l.wv             = make(b("attn_v.weight"), false);          // absent on global layers: V = K
        l.wo             = make(b("attn_output.weight"), false);
        l.q_norm         = make(b("attn_q_norm.weight"), false);
        l.k_norm         = make(b("attn_k_norm.weight"), false);
        l.attn_post_norm = make(b("post_attention_norm.weight"), false);
        l.out_scale      = make(b("layer_output_scale.weight"), false);
        l.ffn_norm       = make(b("ffn_norm.weight"), false);
        l.ffn_gate       = make(b("ffn_gate.weight"), false);
        l.ffn_up         = make(b("ffn_up.weight"), false);
        l.ffn_down       = make(b("ffn_down.weight"), false);
        l.ffn_post_norm  = make(b("post_ffw_norm.weight"), false);
        l.gate_inp       = make(b("ffn_gate_inp.weight"), false);
        l.gate_inp_s     = make(b("ffn_gate_inp.scale"), false);
        l.pre_norm_2     = make(b("pre_ffw_norm_2.weight"), false);
        l.post_norm_1    = make(b("post_ffw_norm_1.weight"), false);
        l.post_norm_2    = make(b("post_ffw_norm_2.weight"), false);
        l.gate_up_exps   = make(b("ffn_gate_up_exps.weight"), is_exps(il));
        l.down_exps      = make(b("ffn_down_exps.weight"), is_exps(il));
        l.down_exps_s    = make(b("ffn_down_exps.scale"), is_exps(il));
        l.wq_s = make_s(l.wq, b("attn_q.scale")); l.wk_s = make_s(l.wk, b("attn_k.scale"));
        l.wv_s = make_s(l.wv, b("attn_v.scale")); l.wo_s = make_s(l.wo, b("attn_output.scale"));
        l.ffn_up_s = make_s(l.ffn_up, b("ffn_up.scale")); l.ffn_gate_s = make_s(l.ffn_gate, b("ffn_gate.scale"));
        l.ffn_down_s = make_s(l.ffn_down, b("ffn_down.scale"));
        // a split gate/up expert pair would take ffn_{gate,up}_exps.scale; this engine reads the fused ffn_gate_up_exps only,
        // which llama.cpp's generic pass gives no scale
        if (!scale_err.empty()) return scale_err;
        const bool moe = l.gate_inp != nullptr;   // llama.cpp: is_moe_layer = ffn_gate_inp present
        if (!l.attn_norm || !l.wq || !l.wk || !l.wo || !l.ffn_norm || !l.ffn_gate || !l.ffn_up || !l.ffn_down || !l.ffn_post_norm ||
            (moe && (!l.gate_up_exps || !l.down_exps || !l.pre_norm_2 || !l.post_norm_1 || !l.post_norm_2)) || (!moe && hp_.n_expert > 0))
            return "layer " + std::to_string(il) + ": missing tensors";
    }
    if (!scale_err.empty()) return scale_err;
    if (n_scales) fprintf(stderr, "impulse: %d second-level weight scales (\"*.scale\", NVFP4) applied as llama.cpp does\n", n_scales);
    // GPU weights: one buffer, filled from the mapping
    buf_gpu_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpu, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_gpu_) return "GPU weight allocation failed";
    ggml_backend_buffer_set_usage(buf_gpu_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // the scheduler places ops by their weights
    for (size_t d = 1; d < gpus_.size(); ++d) {
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_gpus[d], ggml_backend_get_default_buffer_type(gpus_[d]));
        if (!b) return "GPU weight allocation failed (GPU " + std::to_string(d) + ")";
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buf_gpu_x_.push_back(b);
    }
    for (auto & [t, off] : on_gpu) ggml_backend_tensor_set(t, (const char *) map_ + off, 0, ggml_nbytes(t));
    // CPU weights: views into the mapping
    buf_map_ = ggml_backend_cpu_buffer_from_ptr(map_, map_size_);
    ggml_backend_buffer_set_usage(buf_map_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // with async swaps the RAM-resident experts live in a page-locked arena (Strata's expert arena): the copy stream
    // needs pinned sources, and a read-only file mapping cannot be registered. Same bytes, so the arithmetic is unchanged.
    // Low-RAM mode: when the OS will not lock that much (too little free RAM), the experts stay in the file mapping and the
    // OS reads them in as needed; swaps are then copied in the step (slower, but it runs). --no-async-swaps chooses it up front.
    // The RAM tier (src/lite/ram_tier.hpp): the experts locked up to --ram-budget, the rest stay in the mapping.
    lite::RamTier tier;
    if (o_.async_swaps) {
        // numa-always: each socket's half of every expert's rows placed on that socket under the pool's split (as the Qwen
        // engine; lite/numa_place.hpp). IMPULSE_NUMA_PLACE=0: the arena as before. One socket: never reached.
        if (o.pool && lite::numa_place_wanted()) {
            pool_ = std::make_unique<scpu::ExpertPool>(o.pool_threads, /*pin*/ true);
            const scpu::NumaSplitPlan & plan = scpu::active_numa_split();
            if (plan.on && !plan.test) tier = lite::ram_tier_place(on_cpu, o_.ram_budget_gib, 4096, map_, { o.gguf }, plan);
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
        pinned_ = true;
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
    // ctx_gpu's tensors stay alive with the context; keep it by chaining (freed with ctx_w_ at exit is fine here)
    (void) ctx_gpu;

    // KV caches (llama_kv_cache_iswa): base = the global layers, n_ctx cells; swa = the window layers,
    // GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256) cells; K and V [n_embd_gqa, cells], V not transposed (flash attention)
    base_.swa = false; swa_.swa = true;
    if (o.n_seq < 1 || o.n_seq > kMaxSeqs) return "n_seq must be 1.." + std::to_string(kMaxSeqs);
    n_past_.assign(o.n_seq, 0);
    // one stream per sequence (llama-server's --parallel: kv_unified false): PAD(n_ctx / n_seq, 256) global cells each, and a
    // window cache of n_swa + a ubatch each (llama_kv_cache_iswa, not unified); one sequence = the whole context
    base_.init((uint32_t) ((o.n_ctx / o.n_seq + 255) / 256 * 256), o.n_seq);
    swa_.init((uint32_t) ((std::min<uint32_t>(base_.ssize, hp_.n_swa + o.n_ubatch) + 255) / 256 * 256), o.n_seq);
    if (o.n_seq > 1 && o_.kv_initial > 0) {   // the pager's device copies move one block; streams keep the whole context at load
        fprintf(stderr, "impulse: %d slots: the whole context's K/V at load (no KV pager)\n", o.n_seq);
        o_.kv_initial = 0;
    }
    base_.k.assign(hp_.n_layer, nullptr); base_.v.assign(hp_.n_layer, nullptr);
    swa_.k.assign(hp_.n_layer, nullptr);  swa_.v.assign(hp_.n_layer, nullptr);
    {   // the window layers' K/V on their layers' cards
        const std::string e = alloc_swa(swa_.size, swa_.k, swa_.v, ctx_swa_, buf_swa_);
        if (!e.empty()) return e;
    }
    // the KV pager: each global layer's K/V in its own buffer of kv_cap_ cells
    kv_cap_ = o_.kv_initial > 0 ? std::min<uint32_t>(base_.size, (uint32_t) ((o_.kv_initial + 255) / 256 * 256)) : base_.size;
    kvl_.assign(hp_.n_layer, {});
    {   // full-size placeholders of the paged K/V (metadata only): the reserve graphs view the whole cache through them
        kfull_.assign(hp_.n_layer, nullptr); vfull_.assign(hp_.n_layer, nullptr);
        ggml_init_params fp = { (size_t) (2 * hp_.n_layer + 4) * ggml_tensor_overhead(), nullptr, true };
        ctx_kvfull_ = ggml_init(fp);
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (hp_.is_swa[il]) continue;
            const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
            kfull_[il] = ggml_new_tensor_3d(ctx_kvfull_, o.type_k, n_gqa, base_.size, 1);
            vfull_[il] = ggml_new_tensor_3d(ctx_kvfull_, o.type_v, n_gqa, base_.size, 1);
            ggml_format_name(kfull_[il], "cache_k_l%d", il);
            ggml_format_name(vfull_[il], "cache_v_l%d", il);
        }
    }
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const std::string e = kv_alloc_layer(il, kv_cap_);
        if (!e.empty()) return e;
    }

    if (o.pool) {   // stage c: Strata's CPU expert pool for the RAM-resident layers
        // never pin on one socket (machine A runs production); pinned only for the per-socket split (numa-always)
        if (!pool_) pool_ = std::make_unique<scpu::ExpertPool>(o.pool_threads, /*pin*/ scpu::numa_split_possible());
        pool_layers_.resize(hp_.n_layer);
        for (int il = 0; il < hp_.n_layer && il < o.n_cpu_moe; ++il) {
            const Layer & l = L_[il];
            auto pl = std::make_unique<PoolLayer>();
            std::string e;
            if (!scpu::native_fmt(l.gate_up_exps->type, l.down_exps->type, hp_.n_embd, hp_.n_ff_exp, pl->f, e,
                                  scpu::NativeAct::GeGLU, /*separate_down*/ true))
                return "layer " + std::to_string(il) + ": " + e;
            pl->gu = (const uint8_t *) l.gate_up_exps->data; pl->gu_stride = l.gate_up_exps->nb[2];
            pl->dn = (const uint8_t *) l.down_exps->data;    pl->d_stride = l.down_exps->nb[2];
            pl->scale = l.down_exps_s ? (const float *) l.down_exps_s->data : nullptr;
            pl->pool = pool_.get();
            pl->stats = &stats_;
            if (pl->gu_stride != 2 * pl->f.up_off || pl->d_stride != pl->f.d_row * (size_t) hp_.n_embd)
                return "layer " + std::to_string(il) + ": expert strides do not match the pool's format";
            pool_layers_[il] = std::move(pl);
        }
    }

    if (o.cache_slots != 0) {   // stage d
        if (!o.pool) return "--cache-slots needs --pool (the misses go to the pool)";
        rank_.assign(hp_.n_layer, {});
        if (o.cache_slots < hp_.n_expert && o.profile.empty()) {   // no profile: plain order; the first prompt's routing refills it
            for (auto & r : rank_) for (int e = 0; e < hp_.n_expert; ++e) r.push_back(e);
        } else if (o.cache_slots < hp_.n_expert) {
            FILE * pf = fopen(o.profile.c_str(), "r");
            if (!pf) return "cannot read the profile " + o.profile;
            char line[8192];
            while (fgets(line, sizeof line, pf)) {
                char * q = line; int il = (int) strtol(q, &q, 10);
                if (*q == ':') ++q;
                if (il < 0 || il >= hp_.n_layer) continue;
                for (;;) { char * e2; long v = strtol(q, &e2, 10); if (e2 == q) break; rank_[il].push_back((int) v); q = e2; }
            }
            fclose(pf);
        }
        cache_.resize(hp_.n_layer);
        for (int il = std::max(0, o.cache_l0); il < std::min(o.n_cpu_moe, hp_.n_layer); ++il) {
            auto C = std::make_unique<CacheLayer>();
            C->full = o.cache_slots >= hp_.n_expert;
            C->res.assign(hp_.n_expert, -1);
            if (!C->full) { C->score.assign(hp_.n_expert, 0.0); C->pending.assign(hp_.n_expert, 0); }
            cache_[il] = std::move(C);
        }
        int S = o.cache_slots;
        const bool autos = S < 0;
        if (autos) {   // the most slots the free VRAM holds after the reserve
            size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            const size_t reserve = (size_t) o.vram_reserve_mb << 20, per = slot_bytes();
            S = fr > reserve + per ? (int) ((fr - reserve) / per) - 1 : 0;
            S = std::min(S, hp_.n_expert - 1);
            if (S < kMinSlots) return "auto slots: the card has no room for an expert arena (" + std::to_string(fr >> 20) + " MiB free)";
        }
        for (int il = 0; il < hp_.n_layer; ++il)
            if (cache_[il] && !cache_[il]->full && (int) rank_[il].size() < S) return "profile: layer " + std::to_string(il) + " ranks too few experts";
        std::string aerr;
        while (!(aerr = arena_build(S, true)).empty() && autos && S > kMinSlots) S = std::max(kMinSlots, S - 2);
        if (!aerr.empty()) return aerr;
        slots_load_ = slots_now_;
        for (int il = 0; il < hp_.n_layer; ++il) {
            CacheLayer * C = cache_[il].get();
            if (!C || C->full) continue;
            C->miss = std::make_unique<PoolLayer>();
            { const PoolLayer & P = *pool_layers_[il]; PoolLayer & M = *C->miss;
              M.f = P.f; M.gu = P.gu; M.gu_stride = P.gu_stride; M.dn = P.dn; M.d_stride = P.d_stride; M.scale = P.scale; M.pool = P.pool; }
            C->miss->res = C->res.data();
            C->miss->stats = &stats_;
        }
    }

    if (o.overlap) {
        if (cache_.empty()) return "--overlap needs --cache-slots";
        driver_ = std::make_unique<Driver>();
        for (auto & C : cache_) if (C && C->miss) C->miss->driver = driver_.get();
    }
    if (o_.async_swaps) {   // the copy stream: expert bytes from the page-locked mapping, beside the compute stream
        if (!o.adapt) return "--async-swaps needs --adapt";
        cudaStream_t cs;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) return "cannot create the copy stream";
        copy_stream_ = cs;

    }

    if (!o.lora.empty()) {   // before the reserve: its plan includes the adapter's nodes, as llama.cpp's
        const std::string e = load_lora(o.lora, o.lora_scale);
        if (!e.empty()) return e;
    }
    new_sched();
    graph_buf_.resize(ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false));
    draft_graph_buf_.resize(graph_buf_.size());   // the drafter's graph lives beside the target's (both are kept)
    if (hp_.n_expert == 0) {   // dense: llama_context's reserve sequence (pp, tg, pp) over the full KV, as llama.cpp at load
        const std::string e = reserve_graphs();
        if (!e.empty()) return e;
    }
    return "";
}

// llama_context::sched_reserve: pp (n_ubatch tokens, every row an output), tg (1, 1), pp again, each over the whole KV. Without
// it every larger step reallocates the compute buffers, and with several cards ggml waits for all of them before it does
// (ggml_backend_sched_alloc_splits): the cards took turns instead of overlapping.
std::string Engine::reserve_graphs() {
    const int n_pp = std::min(o_.n_ctx, o_.n_ubatch);
    std::vector<int32_t> zeros(n_pp, 0);
    reserve_ = true; out_all_ = true;
    std::string e;
    for (int nt : { n_pp, 1, n_pp }) { e = ubatch(zeros.data(), nt, 0, true, nullptr); if (!e.empty()) break; }
    reserve_ = false; out_all_ = false;
    return e.empty() ? "" : "reserve: " + e;
}

// ---------------------------------------------------------------- the graph
// llm_graph_context::build_norm (LLM_NORM_RMS, weight, no bias) + llama_context::graph_get_cb's pin of "norm" nodes to
// the layer's device (full offload: every ubatch size)
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

// TDUMP=<file> TDUMP_N=<n_tokens>: hash the named per-layer tensors of every ubatch with that many tokens (debug)
static bool tdump_want(const ggml_tensor * t) {
    static const char * pfx[] = { "attn_norm-", "Qcur_pos-", "Kcur_pos-", "kqv_out-", "attn_out-", "ffn_mlp-", "ffn_moe_logits-",
                                  "ffn_moe_weights_norm-", "ffn_moe_out-", "l_out-", "result_norm", "result_output", nullptr };
    static const int n = getenv("TDUMP_N") ? atoi(getenv("TDUMP_N")) : -1;
    if (!getenv("TDUMP")) return false;
    bool ok = false;
    for (int i = 0; pfx[i]; ++i) if (strncmp(t->name, pfx[i], strlen(pfx[i])) == 0) ok = true;
    if (!ok) return false;
    const int64_t nt = strncmp(t->name, "Qcur_pos-", 9) == 0 || strncmp(t->name, "Kcur_pos-", 9) == 0 ? t->ne[2] : t->ne[1];
    return (n == -2 || nt == n) && ggml_is_contiguous(t);   // TDUMP_N=-2: every ubatch (the same execution grouping on both sides)
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

// GDUMP=<file> GDUMP_N=<n_tokens>: write every node of the ubatch with that many tokens (name op type ne op_params srcs)
static void gdump_node(const ggml_tensor * t) {
    static const char * path = getenv("GDUMP");
    static const int n = getenv("GDUMP_N") ? atoi(getenv("GDUMP_N")) : -1;
    static bool on = false;
    static FILE * f = nullptr;
    if (!path) return;
    if (!on && strcmp(t->name, "inp_scaled") == 0 && t->ne[1] == n) { on = true; f = fopen(path, "w"); }
    if (!on) return;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\t%d\n", (int) t->flags);
    if (strcmp(t->name, "result_output") == 0) { fclose(f); f = nullptr; on = false; path = nullptr; }
}

struct DumpState { FILE * f; };
static bool tdump_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) gdump_node(t);
    if (!tdump_want(t)) return ask ? false : true;
    if (ask) return true;
    tdump_put(t);
    return true;
}
static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    if (!ask) return true;
    FILE * f = ((DumpState *) ud)->f;
    fprintf(f, "%s\t%s\t%s\t%lld,%lld,%lld,%lld\t", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    for (int i = 0; i < 16; ++i) fprintf(f, "%08x%s", (unsigned) t->op_params[i], i < 15 ? "," : "");
    for (int i = 0; i < GGML_MAX_SRC; ++i) fprintf(f, "\t%s", t->src[i] ? t->src[i]->name : "-");
    fprintf(f, "\n");
    return false;
}

std::string Engine::ubatch(const int32_t * toks, int n, int pos0, bool output, std::vector<float> * logits) {
    if (!fatal_.empty()) return fatal_;
    if (reading_ && n < kOffloadMin) end_read();
    if (n < kOffloadMin) headroom_for(pos0 + n);
    const auto tb0 = std::chrono::steady_clock::now();
    if (copy_stream_) commit_landed();
    const HParams & H = hp_;
    std::vector<int32_t> positions(n);
    std::vector<int8_t> seqs(n, (int8_t) cur_seq_);
    if (!ub_pos_.empty()) { positions = ub_pos_; seqs = ub_seq_; }   // decode_multi: several sequences
    else for (int i = 0; i < n; ++i) positions[i] = pos0 + i;
    // slots first (llama_kv_cache::prepare + apply), then the graph sees the applied cells
    static const bool kv_debug = getenv("G4_KV_DEBUG") != nullptr;
    // reserve_ (reserve_graphs): llama_context::graph_reserve over the full KV: no cells taken, every view at the whole cache
    // the ubatch's sequences: ns consecutive streams from st0, nt tokens each (split_equal, sequential)
    const int ns = (!reserve_ && ub_ns_ > 0) ? ub_ns_ : 1, nt = n / ns, st0 = reserve_ ? 0 : base_.stream_of(seqs[0]);
    const uint32_t keep_base_n_kv = base_.n_kv, keep_swa_n_kv = swa_.n_kv;
    if (reserve_) { base_.n_kv = base_.ssize; swa_.n_kv = swa_.ssize; }
    if (!reserve_) for (Cache * c : { &base_, &swa_ }) {
        c->idxs.clear();
        for (int q = 0; q < ns; ++q) {
            const int st = c->stream_of(seqs[(size_t) q * nt]);
            if (kv_debug) fprintf(stderr, "g4 find_slot: %s stream %d n = %5u, used = %5u, head = %5u, size = %5u, n_swa = %5d\n",
                                  c->swa ? "swa " : "base", st, c->used_max_p1(st), c->used(st), c->heads[st], c->ssize, c->swa ? H.n_swa : 0);
            if (!c->find_slot(nt, H.n_swa, st)) return "KV cache full";
        }
        c->apply(positions, seqs);
        c->n_kv = c->calc_n_kv(st0, ns);
        if (kv_debug) fprintf(stderr, "g4   cells %lld..%lld n_kv %u\n", (long long) c->idxs.front(), (long long) c->idxs.back(), c->n_kv);
    }
    if (!reserve_) {   // the KV pager
        const uint32_t need = (uint32_t) *std::max_element(base_.idxs.begin(), base_.idxs.end()) + 1;
        if (need > kv_cap_) { const std::string e = kv_grow(need); if (!e.empty()) return e; }
    }

    const int n_outputs = output ? (out_all_ ? n : 1) : 0;
    const int64_t n_embd = H.n_embd;

    // graph reuse: the same shape as the last computed graph, the scheduler still holds its allocation and nothing moved
    // under it, so the graph stands and only the inputs change (llama_context::process_ubatch)
    const GraphKey key = { n, (int) base_.n_kv, (int) swa_.n_kv, n_outputs, embd_in_ != nullptr, want_h_, ns, st0 };
    const bool reuse = g_.valid && sched_graph_ == &g_ && g_.key == key && graph_reuse_on() && o_.graph_dump.empty() && meta_views_live(g_);
    uint64_t acc_inc = 0, hit_inc = 0;   // routed pairs the shape alone decides (the whole-layer and all-pool paths)
    if (!reuse) {   // ---- build (to "// ---- build ends")
    graph_drop(g_);
    ggml_init_params ip = { graph_buf_.size(), graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(sched_);

    // build_inp_embd
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(inp_tokens, "inp_tokens"); ggml_set_input(inp_tokens);
    ggml_tensor * inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n);
    ggml_set_name(inp_embd, "inp_embd"); ggml_set_input(inp_embd);
    ggml_tensor * inps[2] = { ggml_get_rows(ctx, tok_embd_, inp_tokens), inp_embd };
    ggml_tensor * inpL = ggml_build_forward_select(gf, inps, 2, embd_in_ ? 1 : 0);
    ggml_set_name(inpL, "embd");
    ggml_build_forward_expand(gf, inpL);

    // gemma4 graph (raw embeddings, i.e. encoded images, are not scaled)
    inpL = ggml_scale(ctx, inpL, embd_in_ ? 1.0f : sqrtf((float) n_embd));
    ggml_set_name(inpL, "inp_scaled");
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(inp_pos);
    // build_attn_inp_kv_iswa: base idxs + mask, then swa idxs + mask (f16 masks for flash attention)
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, base_.n_kv, nt, 1, ns);   // [n_kv, n_tps, 1, n_stream]
    ggml_set_input(kq_mask); ggml_set_name(kq_mask, "attn_inp_kq_mask");
    ggml_tensor * k_idxs_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(k_idxs_swa);
    ggml_tensor * v_idxs_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n); ggml_set_input(v_idxs_swa);
    ggml_tensor * kq_mask_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, swa_.n_kv, nt, 1, ns);
    ggml_set_input(kq_mask_swa); ggml_set_name(kq_mask_swa, "attn_inp_kq_mask");
    // attn_rot_k/v (quantized KV): base k_rot, base v_rot, swa k_rot, swa v_rot, as build_attn_inp_kv_iswa creates them
    const bool rot_k = ggml_is_quantized(o_.type_k), rot_v = ggml_is_quantized(o_.type_v);
    ggml_tensor * k_rot = nullptr, * v_rot = nullptr, * k_rot_swa = nullptr, * v_rot_swa = nullptr;
    if (rot_k) { const int r = k_rot_size(H.head_dim); k_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(k_rot); ggml_set_name(k_rot, "attn_inp_k_rot"); }
    if (rot_v) { v_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(v_rot); ggml_set_name(v_rot, "attn_inp_v_rot"); }
    if (rot_k) { const int r = k_rot_size(H.head_dim_swa); k_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(k_rot_swa); ggml_set_name(k_rot_swa, "attn_inp_k_rot"); }
    if (rot_v) { v_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(v_rot_swa); ggml_set_name(v_rot_swa, "attn_inp_v_rot"); }
    // build_inp_out_ids
    ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp_out_ids);

    std::vector<ggml_tensor *> adapt_ids(o_.adapt ? H.n_layer : 0, nullptr);   // stage e: each cached layer's routing
    for (int il = 0; il < H.n_layer; ++il) {
        const Layer & l = L_[il];
        const bool swa = H.is_swa[il];
        const int64_t n_embd_head = swa ? H.head_dim_swa : H.head_dim;
        const int64_t n_head = H.n_head, n_head_kv = H.n_head_kv[il];
        const float freq_base_l = swa ? H.rope_base_swa : H.rope_base;
        const int n_rot_l = swa ? H.n_rot_swa : H.n_rot;
        auto nm = [&](ggml_tensor * t, const char * s) { ggml_format_name(t, "%s-%d", s, il); };

        ggml_tensor * cur = norm(ctx, inpL, l.attn_norm, il);
        nm(cur, "attn_norm");
        ggml_tensor * freq_factors = swa ? nullptr : rope_freqs_d_[dev_layer_[il]];

        ggml_tensor * Qcur = lmm(ctx, l.wq, cur, l.wq_s);
        nm(Qcur, "Qcur");
        Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n);
        Qcur = norm(ctx, Qcur, l.q_norm, il);
        nm(Qcur, "Qcur_normed");
        Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, freq_factors, n_rot_l, kRopeType, H.n_ctx_train, freq_base_l, 1.0f,
                             kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        nm(Qcur, "Qcur_pos");

        ggml_tensor * Kcur = lmm(ctx, l.wk, cur, l.wk_s);
        ggml_tensor * Vcur = l.wv ? lmm(ctx, l.wv, cur, l.wv_s) : Kcur;
        nm(Kcur, "Kcur"); nm(Vcur, "Vcur");
        Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n);
        Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n);
        Kcur = norm(ctx, Kcur, l.k_norm, il);
        Vcur = ggml_rms_norm(ctx, Vcur, H.eps);
        nm(Kcur, "Kcur_normed"); nm(Vcur, "Vcur_normed");
        Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, freq_factors, n_rot_l, kRopeType, H.n_ctx_train, freq_base_l, 1.0f,
                             kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        nm(Kcur, "Kcur_pos");

        // build_attn (iswa)
        ggml_tensor * kr = swa ? k_rot_swa : k_rot, * vr = swa ? v_rot_swa : v_rot;
        if (kr) { Qcur = mul_mat_hadamard(ctx, Qcur, kr); Kcur = mul_mat_hadamard(ctx, Kcur, kr); }
        if (vr) Vcur = mul_mat_hadamard(ctx, Vcur, vr);
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        Cache & C = swa ? swa_ : base_;
        const int64_t n_gqa = n_embd_head * n_head_kv;
        ggml_tensor * Ck = (reserve_ && !swa) ? kfull_[il] : C.k[il], * Cv = (reserve_ && !swa) ? vfull_[il] : C.v[il];   // the pager: see kfull_
        {   // cpy_k / cpy_v
            ggml_tensor * kc = ggml_view_2d(ctx, Kcur, n_gqa, n, Kcur->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, Ck, kc, swa ? k_idxs_swa : k_idxs));
            ggml_tensor * vc = ggml_view_2d(ctx, Vcur, n_gqa, n, Vcur->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, Cv, vc, swa ? v_idxs_swa : v_idxs));
        }
        ggml_tensor * mask = swa ? kq_mask_swa : kq_mask;
        // llama_kv_cache::get_k / get_v: [hd, heads, n_kv, ns] over the ubatch's streams; Q as [hd, heads, n_tps, n_stream]
        ggml_tensor * k = ggml_view_4d(ctx, Ck, n_embd_head, n_head_kv, C.n_kv, ns,
                                       ggml_row_size(Ck->type, n_embd_head), ggml_row_size(Ck->type, n_gqa),
                                       ggml_row_size(Ck->type, n_gqa * C.ssize), (size_t) st0 * ggml_row_size(Ck->type, n_gqa * C.ssize));
        ggml_tensor * v = ggml_view_4d(ctx, Cv, n_embd_head, n_head_kv, C.n_kv, ns,
                                       ggml_row_size(Cv->type, n_embd_head), ggml_row_size(Cv->type, n_gqa),
                                       ggml_row_size(Cv->type, n_gqa * C.ssize), (size_t) st0 * ggml_row_size(Cv->type, n_gqa * C.ssize));
        {   // build_attn_mha, flash attention
            ggml_tensor * q = ggml_view_4d(ctx, Qcur, Qcur->ne[0], Qcur->ne[1], nt, ns,
                                           Qcur->nb[1], Qcur->nb[2], Qcur->nb[2] * nt, 0);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
            cur = ggml_flash_attn_ext(ctx, q, kp, vp, mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, nullptr);
            ggml_flash_attn_ext_set_n_kv_max(cur, 0);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        }
        nm(cur, "kqv_out");
        if (vr) cur = mul_mat_hadamard(ctx, cur, vr);
        cur = lmm(ctx, l.wo, cur, l.wo_s);

        cur = norm(ctx, cur, l.attn_post_norm, il);
        nm(cur, "attn_post_norm");
        ggml_tensor * attn_out = ggml_add(ctx, cur, inpL);
        nm(attn_out, "attn_out");

        if (!l.gate_inp) {   // a dense layer (llm_build_gemma4's else branch): ffn_norm, build_ffn(GELU, PAR)
            cur = norm(ctx, attn_out, l.ffn_norm, il);
            nm(cur, "ffn_norm");
            ggml_tensor * tmp = lmm(ctx, l.ffn_up, cur);
            nm(tmp, "ffn_up");
            if (l.ffn_up_s) { tmp = ggml_mul(ctx, tmp, l.ffn_up_s); nm(tmp, "ffn_up_s"); }   // build_ffn: after the adapter's term
            ggml_tensor * g = lmm(ctx, l.ffn_gate, cur);
            nm(g, "ffn_gate");
            if (l.ffn_gate_s) { g = ggml_mul(ctx, g, l.ffn_gate_s); nm(g, "ffn_gate_s"); }
            g = ggml_geglu_split(ctx, g, tmp);
            nm(g, "ffn_geglu");
            cur = lmm(ctx, l.ffn_down, g);
            if (l.ffn_down_s) { cur = ggml_mul(ctx, cur, l.ffn_down_s); nm(cur, "ffn_down_s"); }
            nm(cur, "ffn_out");
        } else {
        // MLP (shared) beside the experts
        ggml_tensor * cur_mlp = norm(ctx, attn_out, l.ffn_norm, il);
        nm(cur_mlp, "ffn_norm_1");
        {   // build_ffn(GELU, PAR)
            ggml_tensor * tmp = lmm(ctx, l.ffn_up, cur_mlp);
            nm(tmp, "ffn_up");
            if (l.ffn_up_s) { tmp = ggml_mul(ctx, tmp, l.ffn_up_s); nm(tmp, "ffn_up_s"); }
            ggml_tensor * g = lmm(ctx, l.ffn_gate, cur_mlp);
            nm(g, "ffn_gate");
            if (l.ffn_gate_s) { g = ggml_mul(ctx, g, l.ffn_gate_s); nm(g, "ffn_gate_s"); }
            g = ggml_geglu_split(ctx, g, tmp);
            nm(g, "ffn_geglu");
            cur_mlp = lmm(ctx, l.ffn_down, g);
            if (l.ffn_down_s) { cur_mlp = ggml_mul(ctx, cur_mlp, l.ffn_down_s); nm(cur_mlp, "ffn_down_s"); }
        }
        cur_mlp = norm(ctx, cur_mlp, l.post_norm_1, il);
        nm(cur_mlp, "ffn_mlp");

        // experts
        ggml_tensor * cur_moe = norm(ctx, attn_out, l.pre_norm_2, il);
        nm(cur_moe, "ffn_norm_2");
        ggml_tensor * tmp = ggml_rms_norm(ctx, attn_out, H.eps);
        tmp = ggml_scale(ctx, tmp, 1.0f / sqrtf((float) n_embd));
        tmp = ggml_mul(ctx, tmp, l.gate_inp_s);
        ggml_tensor * logits_r = ggml_mul_mat(ctx, l.gate_inp, tmp);
        nm(logits_r, "ffn_moe_logits");
        {   // build_moe_ffn(probs_in = logits, softmax gating, norm_w, fused gate_up, GELU, down scale)
            const int64_t n_expert = H.n_expert, n_used = H.n_expert_used;
            ggml_tensor * probs = ggml_soft_max(ctx, logits_r);
            nm(probs, "ffn_moe_probs");
            ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, (int) n_used);
            nm(selected->src[0], "ffn_moe_argsort");
            nm(selected, "ffn_moe_topk");
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

            // gate_up -> geglu -> down over expert tensors `gu_w`/`dn_w` addressed by `ids` (build_moe_ffn's middle)
            auto mm_path = [&](ggml_tensor * gu_w, ggml_tensor * dn_w, ggml_tensor * ids) {
                ggml_tensor * x = ggml_reshape_3d(ctx, cur_moe, n_embd, 1, n);
                ggml_tensor * gate_up = ggml_mul_mat_id(ctx, gu_w, x, ids);
                nm(gate_up, "ffn_moe_gate_up");
                const int64_t n_ff = gate_up->ne[0] / 2;
                ggml_tensor * gt = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                nm(gt, "ffn_moe_gate");
                ggml_tensor * up = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2],
                                                n_ff * gate_up->nb[0]);
                nm(up, "ffn_moe_up");
                gt = ggml_geglu_split(ctx, gt, up);
                nm(gt, "ffn_moe_geglu");
                return ggml_mul_mat_id(ctx, dn_w, gt, ids);
            };
            CacheLayer * C = cache_.empty() ? nullptr : cache_[il].get();
            const bool small = n < kOffloadMin;
            if (o_.adapt && C && !C->full) { ggml_set_output(selected->src[0]); adapt_ids[il] = selected->src[0]; }
            ggml_tensor * scale_t = l.down_exps_s;
            ggml_tensor * experts;
            if (C && C->full) {                                     // stage d, whole layer: llama.cpp's GPU-layer path
                experts = mm_path(C->gu, C->dn, selected);
                scale_t = C->scale;
                if (small) { acc_inc += (uint64_t) n * n_used; hit_inc += (uint64_t) n * n_used; }
            } else if (C && small && n <= kSlotMaxTokens) {         // stage d: hits on the GPU slots, misses in the pool
                ggml_tensor * slot = ggml_reshape_3d(ctx, C->res_f, 1, n_expert, 1);   // per token, as the scale is gathered
                slot = ggml_repeat_4d(ctx, slot, 1, n_expert, n, 1);
                slot = ggml_get_rows(ctx, slot, selected);                          // [1, k, n]: slot, or S (zeros)
                slot = ggml_reshape_2d(ctx, slot, n_used, n);
                ggml_tensor * slot_ids = ggml_cast(ctx, slot, GGML_TYPE_I32);
                nm(slot_ids, "ffn_moe_slots");
                ggml_tensor * args[2] = { cur_moe, selected };
                ggml_tensor * eg, * ec;
                if (o_.overlap) {   // start (CPU) -> hits (GPU, launched async) -> join (CPU, waits on the pool) -> add
                    ggml_tensor * st = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, moe_start_op, 1, C->miss.get());
                    nm(st, "ffn_moe_start");
                    ggml_build_forward_expand(gf, st);
                    eg = mm_path(C->gu, C->dn, slot_ids);
                    ggml_build_forward_expand(gf, eg);
                    ggml_tensor * jargs[1] = { st };
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, jargs, 1, moe_join_op, 1, C->miss.get());
                } else {
                    eg = mm_path(C->gu, C->dn, slot_ids);
                    ec = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1, C->miss.get());
                }
                nm(ec, "ffn_moe_misses");
                experts = ggml_add(ctx, eg, ec);                    // each row: its GPU value + 0, or 0 + its CPU value
                scale_t = C->scale;
                if (C->gu_all) {                                    // dcheck twins: the whole layer both ways
                    ggml_tensor * tg = mm_path(C->gu_all, C->dn_all, selected);
                    ggml_tensor * tc = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1,
                                                      pool_layers_[il].get());
                    ggml_set_output(tg); ggml_set_output(tc); ggml_set_output(experts); ggml_set_output(selected->src[0]);
                    ggml_build_forward_expand(gf, tg); ggml_build_forward_expand(gf, tc);
                    dcheck_ = { experts, tg, tc, selected->src[0], C };
                }
            } else if (pool_ && pool_layers_[il] && small) {         // stage c: all of the layer's experts in the pool
                ggml_tensor * args[2] = { cur_moe, selected };
                experts = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_used, n, 1, args, 2, moe_pool_op, 1,
                                         pool_layers_[il].get());
                if (!cache_.empty()) acc_inc += (uint64_t) n * n_used;
            } else {
                experts = mm_path(l.gate_up_exps, l.down_exps, selected);
            }
            if (scale_t) {    // build_lora_mm_id's per-expert scale
                ggml_tensor * s = ggml_reshape_3d(ctx, scale_t, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx, s, 1, n_expert, n, 1);
                s = ggml_get_rows(ctx, s, selected);
                experts = ggml_mul(ctx, experts, s);
            }
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
            for (int i = 1; i < n_used; ++i) {
                moe_out = ggml_add(ctx, moe_out, parts[i]);
                ggml_build_forward_expand(gf, moe_out);
            }
            nm(moe_out, "ffn_moe_out");
            cur_moe = moe_out;
        }
        cur_moe = norm(ctx, cur_moe, l.post_norm_2, il);
        nm(cur_moe, "ffn_moe");
        cur = ggml_add(ctx, cur_mlp, cur_moe);
        nm(cur, "ffn_moe_combined");
        }

        cur = norm(ctx, cur, l.ffn_post_norm, -1);
        nm(cur, "ffn_post_norm");
        cur = ggml_add(ctx, cur, attn_out);
        if (l.out_scale) {
            cur = ggml_mul(ctx, cur, l.out_scale);
            nm(cur, "out_scaled");
        }
        nm(cur, "l_out");
        inpL = cur;
    }
    ggml_tensor * cur = norm(ctx, inpL, output_norm_, -1);
    ggml_set_name(cur, "h_nextn");
    ggml_tensor * t_h = (want_h_ && n_outputs) ? cur : nullptr;   // the MTP drafter's input (all rows: not masked)
    if (t_h) ggml_set_output(t_h);
    cur = ggml_get_rows(ctx, cur, inp_out_ids);
    ggml_set_name(cur, "result_norm");
    cur = ggml_mul_mat(ctx, output_, cur);
    if (output_s_) cur = ggml_mul(ctx, cur, output_s_);   // build_lora_mm(output, cur, output_s)
    if (H.softcap > 0) {
        cur = ggml_scale(ctx, cur, 1.0f / H.softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, H.softcap);
    }
    ggml_set_name(cur, "result_output");
    ggml_tensor * t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    if (reserve_) {
        const bool ok = ggml_backend_sched_reserve(sched_, gf);
        ggml_free(ctx);
        base_.n_kv = keep_base_n_kv; swa_.n_kv = keep_swa_n_kv;
        return ok ? "" : "graph reserve failed";
    }
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) { ggml_free(ctx); return "graph allocation failed"; }
    g_.ctx = ctx; g_.gf = gf; g_.key = key; g_.valid = true; sched_graph_ = &g_; g_.built_at = ++meta_builds_;
    g_.inp_tokens = inp_tokens; g_.inp_embd = inp_embd; g_.inp_pos = inp_pos; g_.k_idxs = k_idxs; g_.v_idxs = v_idxs; g_.kq_mask = kq_mask;
    g_.k_idxs_swa = k_idxs_swa; g_.v_idxs_swa = v_idxs_swa; g_.kq_mask_swa = kq_mask_swa;
    g_.k_rot = k_rot; g_.v_rot = v_rot; g_.k_rot_swa = k_rot_swa; g_.v_rot_swa = v_rot_swa;
    g_.inp_out_ids = inp_out_ids; g_.t_logits = t_logits; g_.t_h = t_h;
    g_.adapt_ids = std::move(adapt_ids); g_.dcheck = dcheck_; g_.acc_inc = acc_inc; g_.hit_inc = hit_inc;
    stats_.graphs_built++;
    } else {   // ---- build ends
        dcheck_ = g_.dcheck;
        stats_.graphs_reused++;
    }
    const Graph & G = g_;
    const bool timed = n < kOffloadMin;
    if (timed) { stats_.t_ubatches++; stats_.t_splits += ggml_backend_sched_get_n_splits(sched_); }
    if (getenv("G4_SCHED_DUMP") && n == 1 && n_ubatch_graphs_ == 1) {   // which backend runs each node (one decode step)
        FILE * f = fopen(getenv("G4_SCHED_DUMP"), "w");
        for (int i = 0; i < ggml_graph_n_nodes(G.gf); ++i) {
            ggml_tensor * t = ggml_graph_node(G.gf, i);
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched_, t);
            fprintf(f, "%s\t%s\t%s\n", t->name, ggml_op_desc(t), b ? ggml_backend_name(b) : "-");
        }
        fclose(f);
    }

    // inputs
    if (embd_in_) {
        std::vector<int32_t> zero(n, 0);
        ggml_backend_tensor_set(G.inp_tokens, zero.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(G.inp_embd, embd_in_, 0, (size_t) n * n_embd * sizeof(float));
    } else {
        ggml_backend_tensor_set(G.inp_tokens, toks, 0, n * sizeof(int32_t));
    }
    ggml_backend_tensor_set(G.inp_pos, positions.data(), 0, n * sizeof(int32_t));
    auto set_cache_inputs = [&](Cache & C, ggml_tensor * ki, ggml_tensor * vi, ggml_tensor * m) {
        ggml_backend_tensor_set(ki, C.idxs.data(), 0, n * sizeof(int64_t));
        ggml_backend_tensor_set(vi, C.idxs.data(), 0, n * sizeof(int64_t));
        // token i sees cell j when lo[i] <= pos[j] <= hi[i] (lite/mask_fill.hpp); a cell is masked when its pos < 0, when it
        // is past the token (causal), or when it is n_swa or more behind it (the window layers). Non-causal (an image chunk
        // on Gemma, LLAMA_NON_CAUSAL_TYPE_SWA_ONLY): the window layers see the whole chunk, so no upper bound.
        const bool causal = !(non_causal_ && C.swa);
        std::vector<ggml_fp16_t> md((size_t) C.n_kv * n);
        std::vector<int32_t> lo(n), hi(n);
        for (int i = 0; i < n; ++i) {
            const int32_t p1 = positions[i];
            lo[i] = C.swa ? std::max(0, p1 - H.n_swa + 1) : 0;
            hi[i] = causal ? p1 : INT32_MAX;
        }
        for (int q = 0; q < ns; ++q)   // each sequence's rows over its own stream's cells
            lite::fill_mask(md.data() + (size_t) q * nt * C.n_kv, nt, C.n_kv, C.pos.data() + (size_t) C.stream_of(seqs[(size_t) q * nt]) * C.ssize,
                            lo.data() + (size_t) q * nt, hi.data() + (size_t) q * nt, o_.n_threads);
        ggml_backend_tensor_set(m, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    };
    set_cache_inputs(base_, G.k_idxs, G.v_idxs, G.kq_mask);
    set_cache_inputs(swa_, G.k_idxs_swa, G.v_idxs_swa, G.kq_mask_swa);
    for (ggml_tensor * r : { G.k_rot, G.v_rot, G.k_rot_swa, G.v_rot_swa }) {
        if (!r) continue;
        const std::vector<float> h = gen_hadamard((int) r->ne[0]);
        ggml_backend_tensor_set(r, h.data(), 0, h.size() * sizeof(float));
    }
    if (n_outputs) {
        std::vector<int32_t> oid(n_outputs);
        for (int i = 0; i < n_outputs; ++i) oid[i] = n - n_outputs + i;
        ggml_backend_tensor_set(G.inp_out_ids, oid.data(), 0, n_outputs * sizeof(int32_t));
    }

    DumpState ds = { nullptr };
    if (n == 1) ++n_ubatch_graphs_;
    if (!o_.graph_dump.empty() && n == 1 && n_ubatch_graphs_ == 3) {   // route-log dumps its 3rd decoded token
        ds.f = fopen(o_.graph_dump.c_str(), "w");
        ggml_backend_sched_set_eval_callback(sched_, dump_cb, &ds);
    }
    if ((getenv("TDUMP") || getenv("GDUMP")) && !ds.f) ggml_backend_sched_set_eval_callback(sched_, tdump_cb, nullptr);
    const auto tc0 = std::chrono::steady_clock::now();
    if (timed) stats_.t_build += std::chrono::duration<double, std::milli>(tc0 - tb0).count();
    stats_.accesses += G.acc_inc; stats_.hits += G.hit_inc;
    const uint64_t acc0 = stats_.accesses, hit0 = stats_.hits;
    // several cards: queued, not waited for (llama_context::graph_compute); the readback below waits when it needs to
    const bool pipe = gpus_.size() > 1;
    const ggml_status st = pipe ? ggml_backend_sched_graph_compute_async(sched_, G.gf) : ggml_backend_sched_graph_compute(sched_, G.gf);
    if (pipe && ((n_outputs && logits) || G.t_h)) ggml_backend_sched_synchronize(sched_);   // ggml-cuda's streams do not block
    const auto tc1 = std::chrono::steady_clock::now();
    if (timed) stats_.t_compute += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
    if (ds.f) { fclose(ds.f); ggml_backend_sched_set_eval_callback(sched_, nullptr, nullptr); }
    if (st != GGML_STATUS_SUCCESS) { graph_drop(g_); sched_graph_ = nullptr; return "compute failed"; }
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
        for (int64_t t = 0; t < n; ++t) for (int64_t j = 0; j < k; ++j) {
            const int e = ids[t * ne_sort + j];
            const bool hit = d.C->res[e] >= 0;
            const float * src = (hit ? g.data() : c.data()) + (t * k + j) * ne;
            stats_.dcheck_rows++; stats_.dcheck_gpu_rows += hit;
            if (memcmp(m.data() + (t * k + j) * ne, src, ne * 4) != 0) stats_.dcheck_bad++;
        }
    }
    const auto ta0 = std::chrono::steady_clock::now();
    if (timed) stats_.t_readout += std::chrono::duration<double, std::milli>(ta0 - tc1).count();
    if (timed && (o_.lend || o_.prefill_ubatch > 0)) {
        stats_.step_acc.push_back((uint32_t) (stats_.accesses - acc0)); stats_.step_hit.push_back((uint32_t) (stats_.hits - hit0));
    }
    if (o_.adapt) adapt_after(n, G.adapt_ids);
    const auto ta1 = std::chrono::steady_clock::now();
    if (timed) stats_.t_scores += std::chrono::duration<double, std::milli>(ta1 - ta0).count();   // incl. the swaps
    if (n_outputs && logits) {
        logits->resize((size_t) n_outputs * H.n_vocab);
        ggml_backend_tensor_get(G.t_logits, logits->data(), 0, (size_t) n_outputs * H.n_vocab * sizeof(float));
    }
    if (G.t_h) {   // the output rows' hidden states (the last n_outputs rows of the ubatch)
        h_rows_.resize((size_t) n_outputs * n_embd);
        ggml_backend_tensor_get(G.t_h, h_rows_.data(), (size_t) (n - n_outputs) * n_embd * sizeof(float),
                                h_rows_.size() * sizeof(float));
    }
    if (timed) stats_.t_readout += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta1).count();
    return "";   // the graph and its allocation stay for the next step of this shape
}

// ---------------------------------------------------------------- stage e: adaptation
// After a ubatch: every cached layer's scores decay per token and count its routed experts (Gate 0's
// lfu_decay rule). A prompt read (a big ubatch) only marks a refill; the next small ubatch first re-sets every layer
// to its top-S, then each small ubatch swaps at most adapt_budget experts per layer. Copies are synchronous here
// (the copy stream comes with the overlap work), and happen after the step's compute and its twin check.
void Engine::adapt_after(int n, const std::vector<ggml_tensor *> & argsorts) {
    const double d = pow(0.5, 1.0 / o_.adapt_half_life);
    const int k = hp_.n_expert_used;
    std::vector<int32_t> ids;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!argsorts[il]) continue;
        CacheLayer * C = cache_[il].get();
        ids.resize(ggml_nelements(argsorts[il]));
        ggml_backend_tensor_get(argsorts[il], ids.data(), 0, ids.size() * sizeof(int32_t));
        const int64_t ne = argsorts[il]->ne[0];
        for (int t = 0; t < n; ++t) {
            for (auto & v : C->score) v *= d;
            for (int j = 0; j < k; ++j) C->score[ids[t * ne + j]] += 1.0;
        }
    }
    if (n >= kOffloadMin) { refill_pending_ = o_.adapt_refill; return; }
    const bool refill = refill_pending_;
    refill_pending_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t before = stats_.swaps;
    // the refill after a lent read in pieces over the next steps (IMPULSE_REFILL_STEP per layer and step, default 16;
    // 0 = all at once): queued whole it fills the copy queue and holds the first step (q35_engine.cpp, measured 10-17)
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
    if (copy_stream_ && !pending_.empty() && !pending_.back().ev) {
        if (pending_.back().sw.empty()) pending_.pop_back();
        else {
            cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            cudaEventRecord(ev, (cudaStream_t) copy_stream_);
            pending_.back().ev = ev;
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
    const Layer & l = L_[il];
    const int NE = hp_.n_expert;
    std::vector<int> want, worst;
    for (int e = 0; e < NE; ++e) (C->res[e] < 0 ? want : worst).push_back(e);
    std::stable_sort(want.begin(), want.end(), [&](int a, int b) { return C->score[a] > C->score[b]; });
    std::stable_sort(worst.begin(), worst.end(), [&](int a, int b) { return C->score[a] < C->score[b]; });
    const size_t gs = l.gate_up_exps->nb[2], ds = l.down_exps->nb[2];
    int done = 0;
    if (copy_stream_) {   // async: the victim goes to the CPU now, the newcomer when its copy has landed
        want.erase(std::remove_if(want.begin(), want.end(), [&](int e) { return C->pending[e] != 0; }), want.end());
        if (pending_.empty() || pending_.back().ev) pending_.push_back({ nullptr, {} });
    }
    // stage g: slots left free (after a read the arena comes back empty) take the best newcomers first
    std::vector<int> free_slots;
    for (int j = 0; j < C->S; ++j) if (C->owner[j] == -1) free_slots.push_back(j);
    size_t wi = 0;
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
        if (copy_stream_) {
            cudaStream_t cs = (cudaStream_t) copy_stream_;
            cudaMemcpyAsync((char *) C->gu->data + (size_t) slot * gs, (const char *) l.gate_up_exps->data + a * gs, gs, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync((char *) C->dn->data + (size_t) slot * ds, (const char *) l.down_exps->data + a * ds, ds, cudaMemcpyHostToDevice, cs);
            if (w >= 0) C->res[w] = -1;
            C->owner[slot] = -2; C->pending[a] = 1;
            pending_.back().sw.push_back({ il, a, slot });
        } else {
            ggml_backend_tensor_set(C->gu, (const char *) l.gate_up_exps->data + a * gs, (size_t) slot * gs, gs);
            ggml_backend_tensor_set(C->dn, (const char *) l.down_exps->data + a * ds, (size_t) slot * ds, ds);
            if (w >= 0) C->res[w] = -1;
            C->res[a] = slot; C->owner[slot] = a;
        }
        stats_.swaps++; stats_.swap_bytes += gs + ds;
        done++;
    }
    if (done) {
        std::vector<float> rf(NE);
        for (int e = 0; e < NE; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
        ggml_backend_tensor_set(C->res_f, rf.data(), 0, NE * sizeof(float));
    }
}

// ---------------------------------------------------------------- stage g: chunked reads with VRAM lent from the cache
// The window cache holds n_swa + ubatch cells (llama_kv_cache_iswa); a read in big ubatches needs it that large.
// the window layers' K/V for `cells` cells, each layer on its card (one context + buffer per card), zeroed
std::string Engine::alloc_swa(uint32_t cells, std::vector<ggml_tensor *> & k, std::vector<ggml_tensor *> & v,
                              std::vector<ggml_context *> & ctxs, std::vector<ggml_backend_buffer_t> & bufs) {
    ggml_init_params kp = { (size_t) (2 * hp_.n_layer + 4) * ggml_tensor_overhead(), nullptr, true };
    ctxs.assign(gpus_.size(), nullptr); bufs.assign(gpus_.size(), nullptr);
    k.assign(hp_.n_layer, nullptr); v.assign(hp_.n_layer, nullptr);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!hp_.is_swa[il]) continue;   // the global layers: the KV pager
        const int d = dev_layer_[il];
        if (!ctxs[d]) ctxs[d] = ggml_init(kp);
        const int64_t n_gqa = (int64_t) hp_.head_dim_swa * hp_.n_head_kv[il];
        k[il] = ggml_new_tensor_3d(ctxs[d], o_.type_k, n_gqa, cells, 1);
        v[il] = ggml_new_tensor_3d(ctxs[d], o_.type_v, n_gqa, cells, 1);
        ggml_format_name(k[il], "cache_k_l%d", il);
        ggml_format_name(v[il], "cache_v_l%d", il);
    }
    for (size_t d = 0; d < gpus_.size(); ++d) {
        if (!ctxs[d]) continue;
        bufs[d] = ggml_backend_alloc_ctx_tensors_from_buft(ctxs[d], ggml_backend_get_default_buffer_type(gpus_[d]));
        if (!bufs[d]) return "KV allocation failed (window layers, GPU " + std::to_string(d) + ")";
        ggml_backend_buffer_clear(bufs[d], 0);
    }
    return "";
}

// the scheduler over every card and the CPU, the CPU's compute buffers in pinned host memory (llama_context), from the pool
// that keeps them across reads (lite/pinned_host.hpp: ggml-cuda's host type would cudaMallocHost / cudaFreeHost them with
// every scheduler)
void Engine::new_sched() {
    graph_invalidate();   // the kept graphs were allocated by the old scheduler
    if (sched_) ggml_backend_sched_free(sched_);
    std::vector<ggml_backend_t> backs(gpus_);
    backs.push_back(cpu_);
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto * g : gpus_) bufts.push_back(ggml_backend_get_default_buffer_type(g));
    bufts.push_back(lite::pinned_host_buffer_type(cpu_));
    // several cards: pipeline parallelism (llama_context: n_devices > 1, all layers offloaded, split mode layer): the scheduler
    // keeps copies of each split's inputs, so card 0 starts the next ubatch while card 1 finishes this one
    // pipeline parallelism only with every layer's experts on the cards (llama_context: no tensor overrides such as --n-cpu-moe)
    sched_ = ggml_backend_sched_new(backs.data(), bufts.data(), (int) backs.size(), kMaxNodes, /*parallel*/ gpus_.size() > 1 && o_.n_cpu_moe <= 0, /*op_offload*/ true);
}

// cells per stream: each stream keeps its own most recent positions (all of them if they fit), in position order, from its
// first cell
void Engine::resize_swa(uint32_t cells) {
    cells = (std::min<uint32_t>(base_.ssize, cells) + 255) / 256 * 256;
    if (cells == swa_.ssize) return;
    if (tp_) { fatal_ = "--split-mode tensor: a ubatch larger than -ub (the window cache cannot be resized)"; return; }
    graph_invalidate();   // the graphs' window KV views point at the old buffer
    const int nst = (int) swa_.heads.size();
    std::vector<std::vector<std::pair<int32_t, uint32_t>>> live(nst);   // per stream: (position, old cell)
    for (int st = 0; st < nst; ++st) {
        for (uint32_t i = 0; i < swa_.ssize; ++i) { const uint32_t c = (uint32_t) st * swa_.ssize + i; if (swa_.pos[c] >= 0) live[st].push_back({ swa_.pos[c], c }); }
        std::sort(live[st].begin(), live[st].end());
        if (live[st].size() > cells) live[st].erase(live[st].begin(), live[st].end() - cells);   // the oldest go
    }
    std::vector<ggml_tensor *> nk, nv;
    std::vector<ggml_context *> nctx; std::vector<ggml_backend_buffer_t> nbuf;
    { const std::string e = alloc_swa(cells * (uint32_t) nst, nk, nv, nctx, nbuf); if (!e.empty()) { fatal_ = e; return; } }
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (!nk[il]) continue;
        for (int pass = 0; pass < 2; ++pass) {
            ggml_tensor * src = pass ? swa_.v[il] : swa_.k[il], * dst = pass ? nv[il] : nk[il];
            const size_t row = src->nb[1];
            for (int st = 0; st < nst; ++st) {
                const auto & L = live[st];
                for (size_t j = 0; j < L.size();) {   // contiguous runs of old cells
                    size_t r = 1;
                    while (j + r < L.size() && L[j + r].second == L[j].second + r) ++r;
                    cudaMemcpy((char *) dst->data + ((size_t) st * cells + j) * row, (const char *) src->data + (size_t) L[j].second * row, r * row, cudaMemcpyDeviceToDevice);
                    j += r;
                }
            }
        }
    }
    std::vector<int32_t> pos((size_t) cells * nst, -1);
    std::vector<int8_t> sq((size_t) cells * nst, -1);
    std::vector<uint32_t> heads(nst, 0u);
    for (int st = 0; st < nst; ++st) {
        for (size_t j = 0; j < live[st].size(); ++j) { pos[(size_t) st * cells + j] = live[st][j].first; sq[(size_t) st * cells + j] = swa_.seq[live[st][j].second]; }
        heads[st] = (uint32_t) (live[st].size() % cells);
    }
    for (auto * b : buf_swa_) if (b) ggml_backend_buffer_free(b);
    for (auto * c : ctx_swa_) if (c) ggml_free(c);
    ctx_swa_ = nctx; buf_swa_ = nbuf;
    swa_.k = nk; swa_.v = nv; swa_.pos = pos; swa_.seq = sq; swa_.ssize = cells; swa_.size = cells * (uint32_t) nst; swa_.heads = heads;
    stats_.swa_big_cells = std::max<uint64_t>(stats_.swa_big_cells, cells);
}

void Engine::begin_read(int ubatch) {
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = true;
    graph_invalidate();
    if (o_.lend && buf_cache_) {   // give the arena's VRAM to the read
        if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
        stats_.lent_bytes = ggml_backend_buffer_get_size(buf_cache_);
        ggml_backend_buffer_free(buf_cache_); buf_cache_ = nullptr;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_cache_); t; t = ggml_get_next_tensor(ctx_cache_, t)) { t->data = nullptr; t->buffer = nullptr; }
        for (auto & C : cache_) if (C) { std::fill(C->res.begin(), C->res.end(), -1); std::fill(C->owner.begin(), C->owner.end(), -1); }
        new_sched();   // its compute buffers re-grow for the big ubatches
    }
    resize_swa((uint32_t) (hp_.n_swa + ubatch));
    stats_.lend_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Engine::end_read() {
    const auto t0 = std::chrono::steady_clock::now();
    reading_ = false;
    graph_invalidate();
    resize_swa((uint32_t) (hp_.n_swa + o_.n_ubatch));
    new_sched();   // drop the big ubatches' compute buffers (the pinned host ones go back to the pool)
    if (o_.lend && !buf_cache_ && ctx_cache_) {   // the arena comes back empty; the refill copies in the background
        // The read may leave less VRAM than it lent (ggml-cuda's pool keeps what the big ubatches grew): fewer slots
        // then, never a failed step.
        int S = slots_now_;
        if (o_.cache_slots < 0) {   // auto slots: what the VRAM holds after the reserve, as at load (the KV pager may have
            // grown the KV during the read; without it this is the load-time count)
            size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            const size_t per = slot_bytes(), reserve = (size_t) o_.vram_reserve_mb << 20;
            const int fit = fr > reserve + 2 * per ? (int) ((fr - reserve) / per) - 1 : kMinSlots;
            S = std::max(kMinSlots, std::min(S, fit));
        }
        std::string err;
        while (!(err = arena_build(S, false)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 4);
        if (!err.empty()) fatal_ = "after the read: " + err;
        headroom_kv_ = 0;   // free VRAM changed: check again before the next small step
    }
    read_end_ = std::chrono::steady_clock::now();
    refill_watch_ = true;
    stats_.restore_ms += std::chrono::duration<double, std::milli>(read_end_ - t0).count();
}

// ---------------------------------------------------------------- the KV pager (global layers)
size_t Engine::kv_bytes(uint32_t cells) const {
    size_t b = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
        b += (ggml_row_size(o_.type_k, n_gqa) + ggml_row_size(o_.type_v, n_gqa)) * cells;
    }
    return b;
}

// one global layer's K/V in a buffer of cap cells: zeroed (as the whole-context buffer was at load), the old cells
// copied over, the old buffer freed
std::string Engine::kv_alloc_layer(int il, uint32_t cap) {
    const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
    ggml_init_params p = { 4 * ggml_tensor_overhead(), nullptr, true };
    KvLayer nl; nl.ctx = ggml_init(p);
    ggml_tensor * k = ggml_new_tensor_3d(nl.ctx, o_.type_k, n_gqa, cap, 1);
    ggml_tensor * v = ggml_new_tensor_3d(nl.ctx, o_.type_v, n_gqa, cap, 1);
    ggml_format_name(k, "cache_k_l%d", il);
    ggml_format_name(v, "cache_v_l%d", il);
    nl.buf = ggml_backend_alloc_ctx_tensors_from_buft(nl.ctx, ggml_backend_get_default_buffer_type(gpus_[dev_layer_[il]]));
    if (!nl.buf) { ggml_free(nl.ctx); return "KV allocation failed (layer " + std::to_string(il) + ", " + std::to_string(cap) + " cells)"; }
    ggml_backend_buffer_clear(nl.buf, 0);
    if (base_.k[il]) {
        const size_t rows = (size_t) std::min<int64_t>(base_.k[il]->ne[1], cap);
        cudaMemcpy(k->data, base_.k[il]->data, ggml_row_size(k->type, n_gqa) * rows, cudaMemcpyDeviceToDevice);
        cudaMemcpy(v->data, base_.v[il]->data, ggml_row_size(v->type, n_gqa) * rows, cudaMemcpyDeviceToDevice);
    }
    if (kvl_[il].buf) ggml_backend_buffer_free(kvl_[il].buf);
    if (kvl_[il].ctx) ggml_free(kvl_[il].ctx);
    kvl_[il] = nl;
    base_.k[il] = k; base_.v[il] = v;
    if (tp_ && !kfull_.empty()) { kfull_[il] = k; vfull_[il] = v; }   // the whole context: the reserve graphs use the cache itself
    else if (!kfull_.empty() && kfull_[il]) {   // the placeholders follow the layer's buffer
        kfull_[il]->data = k->data; kfull_[il]->buffer = nl.buf;
        vfull_[il]->data = v->data; vfull_[il]->buffer = nl.buf;
    }
    graph_invalidate();   // the graphs' KV views point at the old buffer
    return "";
}

// room for cells: the capacity rounded up to kv_step (at most the context); the slot arena gives way first, one layer
// at a time is reallocated (the moment's extra VRAM is one layer's new buffer)
std::string Engine::kv_grow(uint32_t cells) {
    if (cells <= kv_cap_) return "";
    cudaSetDevice(cuda_dev_);
    const uint32_t step = (uint32_t) std::max(256, (o_.kv_step + 255) / 256 * 256);
    const uint32_t cap = std::min(base_.size, (cells + step - 1) / step * step);
    size_t one = 0;
    for (int il = 0; il < hp_.n_layer; ++il) if (!hp_.is_swa[il]) {
        const int64_t n_gqa = (int64_t) hp_.head_dim * hp_.n_head_kv[il];
        one = std::max(one, (ggml_row_size(o_.type_k, n_gqa) + ggml_row_size(o_.type_v, n_gqa)) * cap);
    }
    ensure_free(kv_bytes(cap) - kv_bytes(kv_cap_) + one + ((size_t) 64 << 20));
    if (!fatal_.empty()) return fatal_;
    for (auto * g : gpus_) ggml_backend_synchronize(g);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const std::string e = kv_alloc_layer(il, cap);
        if (!e.empty()) return e;
    }
    fprintf(stderr, "impulse: KV on the card %u -> %u cells (%.0f MiB)\n", kv_cap_, cap, kv_bytes(cap) / 1048576.0);
    kv_cap_ = cap;
    headroom_kv_ = 0;   // free VRAM changed
    return "";
}

void Engine::kv_shrink() {
    const uint32_t init = o_.kv_initial > 0 ? std::min<uint32_t>(base_.size, (uint32_t) ((o_.kv_initial + 255) / 256 * 256)) : base_.size;
    if (kv_cap_ <= init || kvl_.empty()) return;
    for (auto * g : gpus_) ggml_backend_synchronize(g);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) continue;
        const std::string e = kv_alloc_layer(il, init);
        if (!e.empty()) { fatal_ = "new conversation: " + e; return; }
    }
    fprintf(stderr, "impulse: KV on the card %u -> %u cells (a new conversation)\n", kv_cap_, init);
    kv_cap_ = init;
    headroom_kv_ = 0;
    if (slots_load_ > slots_now_) grow_to(slots_load_, (size_t) o_.vram_reserve_mb << 20);
}

// ---------------------------------------------------------------- the slot arena: (re)build, size, shrink
size_t Engine::slot_bytes() const {
    size_t b = 0;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (il < (int) cache_.size() && cache_[il] && !cache_[il]->full) b += L_[il].gate_up_exps->nb[2] + L_[il].down_exps->nb[2];
    return b;
}

// Every cached layer gets S slots (+ the zero slot); with fill_profile the top S of the profile are copied in, else the
// arena starts empty and the next small ubatch refills it from the usage counts (as after a read).
std::string Engine::arena_build(int S, bool fill_profile) {
    graph_invalidate();   // the slot tensors are made anew
    if (buf_cache_) { ggml_backend_buffer_free(buf_cache_); buf_cache_ = nullptr; }
    if (ctx_cache_) { ggml_free(ctx_cache_); ctx_cache_ = nullptr; }
    ggml_init_params cp = { (size_t) (8 * hp_.n_layer + 8) * ggml_tensor_overhead(), nullptr, true };
    ctx_cache_ = ggml_init(cp);
    for (int il = 0; il < (int) cache_.size(); ++il) {
        CacheLayer * C = cache_[il].get();
        if (!C) continue;
        const Layer & l = L_[il];
        C->S = C->full ? hp_.n_expert : S;
        const int alloc = C->full ? hp_.n_expert : C->S + 1;
        C->gu = ggml_new_tensor_3d(ctx_cache_, l.gate_up_exps->type, hp_.n_embd, 2 * hp_.n_ff_exp, alloc);
        C->dn = ggml_new_tensor_3d(ctx_cache_, l.down_exps->type, hp_.n_ff_exp, hp_.n_embd, alloc);
        ggml_format_name(C->gu, "blk.%d.ffn_gate_up_exps.slots", il);
        ggml_format_name(C->dn, "blk.%d.ffn_down_exps.slots", il);
        C->scale = nullptr;
        if (l.down_exps_s) { C->scale = ggml_dup_tensor(ctx_cache_, l.down_exps_s); ggml_format_name(C->scale, "blk.%d.ffn_down_exps.scale", il); }
        std::fill(C->res.begin(), C->res.end(), -1);
        C->res_f = nullptr; C->gu_all = C->dn_all = nullptr;
        if (C->full) { for (int e = 0; e < hp_.n_expert; ++e) C->res[e] = e; continue; }
        C->owner.assign(C->S, -1);
        if (C->pending.size()) std::fill(C->pending.begin(), C->pending.end(), 0);
        C->res_f = ggml_new_tensor_2d(ctx_cache_, GGML_TYPE_F32, 1, hp_.n_expert);
        ggml_format_name(C->res_f, "blk.%d.ffn_exps.slot_of", il);
        if (il == o_.dcheck_layer) { C->gu_all = ggml_dup_tensor(ctx_cache_, l.gate_up_exps); C->dn_all = ggml_dup_tensor(ctx_cache_, l.down_exps); }
    }
    buf_cache_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cache_, ggml_backend_get_default_buffer_type(gpu_));
    if (!buf_cache_) return "expert cache allocation failed (" + std::to_string(S) + " slots do not fit on this card)";
    ggml_backend_buffer_clear(buf_cache_, 0);                     // slot S (the misses' target) stays all zeros
    ggml_backend_buffer_set_usage(buf_cache_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int il = 0; il < (int) cache_.size(); ++il) {
        CacheLayer * C = cache_[il].get();
        if (!C) continue;
        const Layer & l = L_[il];
        const size_t gs = l.gate_up_exps->nb[2], ds = l.down_exps->nb[2];
        if (C->full) {
            ggml_backend_tensor_set(C->gu, l.gate_up_exps->data, 0, ggml_nbytes(l.gate_up_exps));
            ggml_backend_tensor_set(C->dn, l.down_exps->data, 0, ggml_nbytes(l.down_exps));
        } else {
            if (fill_profile)
                for (int j = 0; j < C->S; ++j) {
                    const int e = rank_[il][j];
                    C->res[e] = j; C->owner[j] = e;
                    ggml_backend_tensor_set(C->gu, (const char *) l.gate_up_exps->data + e * gs, (size_t) j * gs, gs);
                    ggml_backend_tensor_set(C->dn, (const char *) l.down_exps->data + e * ds, (size_t) j * ds, ds);
                }
            std::vector<float> rf(hp_.n_expert);
            for (int e = 0; e < hp_.n_expert; ++e) rf[e] = (float) (C->res[e] >= 0 ? C->res[e] : C->S);
            ggml_backend_tensor_set(C->res_f, rf.data(), 0, rf.size() * sizeof(float));
            if (C->gu_all) {
                ggml_backend_tensor_set(C->gu_all, l.gate_up_exps->data, 0, ggml_nbytes(l.gate_up_exps));
                ggml_backend_tensor_set(C->dn_all, l.down_exps->data, 0, ggml_nbytes(l.down_exps));
            }
        }
        if (C->scale) ggml_backend_tensor_set(C->scale, l.down_exps_s->data, 0, ggml_nbytes(l.down_exps_s));
    }
    headroom_kv_ = 0;   // free VRAM changed: check again before the next small step
    if (slots_now_ && S < slots_now_) {
        stats_.arena_shrinks++;
        fprintf(stderr, "impulse: expert arena %d -> %d slots per layer (VRAM)\n", slots_now_, S);
    }
    slots_now_ = S; stats_.slots_now = S;
    return "";
}

// The compute buffers of an m-token read chunk over n_kv KV cells (bytes). Measured on an RTX 3060 at 262K context, q8 KV,
// n_kv = m (v0.2.2: 4096 -> 1261 MiB, 8192 -> 2448, 12288 -> 3659, 16384 -> 5265, 18000 -> 5819, 20824 -> 7777), linear
// between the points and 0.75 MiB a token past the last (the growth steepens there); cells past the chunk add its f16
// mask row and the q8 cells' f16 copy for flash attention. The old formula (768 MiB + 200 KB a token + the mask) was
// 0.8 GB short at 16K and 1.5 GB at 20.8K: auto chose a whole 20.8K-token prompt as one chunk and the read failed.
static size_t read_compute_bytes(size_t m, size_t n_kv) {
    static const double pts[][2] = { { 0, 250 }, { 4096, 1261 }, { 8192, 2448 }, { 12288, 3659 }, { 16384, 5265 },
                                     { 18000, 5819 }, { 20824, 7777 } };
    const int np = (int) (sizeof pts / sizeof pts[0]);
    double mib = pts[np - 1][1] + ((double) m - pts[np - 1][0]) * 0.75;
    for (int i = 1; i < np; ++i)
        if ((double) m <= pts[i][0]) { mib = pts[i - 1][1] + ((double) m - pts[i - 1][0]) * (pts[i][1] - pts[i - 1][1]) / (pts[i][0] - pts[i - 1][0]); break; }
    const size_t extra = n_kv > m ? (n_kv - m) * (2 * m + 4096) : 0;
    return (size_t) (mib * 1048576.0) + extra;
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
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    // what the read can use: free VRAM, the arena it is about to borrow, and the compute buffers the scheduler already holds
    // (the read's graph re-allocates them; uncounted, a 15K read after a 15K read saw 4.8 GB fewer and fell to 4K chunks:
    // -17 % in 0.1.1, 10-20). The Qwen engine counts them the same way
    size_t avail = fr + ((o_.lend && buf_cache_) ? ggml_backend_buffer_get_size(buf_cache_) : 0) + (sched_ ? ggml_backend_sched_get_buffer_size(sched_, gpu_) : 0);
    size_t used_all = 0; for (int q : n_past_) used_all += (size_t) q;   // the read's sequence and the others share the KV
    const size_t n_kv_end = used_all + n, margin = (size_t) 256 << 20;
    {   // the KV pager grows during this read: its new cells come out of the same VRAM
        const uint32_t step = (uint32_t) std::max(256, (o_.kv_step + 255) / 256 * 256);
        const uint32_t cells = (uint32_t) std::min<size_t>(base_.size, (n_kv_end + step - 1) / step * step);
        const size_t kv_more = cells > kv_cap_ ? kv_bytes(cells) - kv_bytes(kv_cap_) : 0;
        avail = avail > kv_more ? avail - kv_more : 0;
    }
    size_t swa_cell = 0;   // the window layers' K + V per cell: begin_read grows their cache to n_swa + the chunk
    for (int il = 0; il < hp_.n_layer; ++il) if (swa_.k[il]) swa_cell += swa_.k[il]->nb[1] + swa_.v[il]->nb[1];
    for (int P : { 32768, 16384, 8192, kChunkMin }) {   // 32k only where the borrowable VRAM covers it (24-32 GB cards)
        if (hp_.n_expert > 0 && P > moe_ubatch_max()) continue;   // the expert matmul's shared memory (moe_ubatch_max)
        const size_t m = (size_t) std::min(P, n);
        const size_t swa_cells = std::min<size_t>(base_.ssize, ((size_t) hp_.n_swa + P + 255) / 256 * 256);   // per stream
        const size_t swa_more = swa_cells > swa_.ssize ? (swa_cells - swa_.ssize) * swa_cell * swa_.heads.size() : 0;
        if (read_compute_bytes(m, n_kv_end) + swa_more + margin <= avail) return P;
    }
    // nothing fits: with a lent arena 4k is the floor (the maintainer, 2026-10-10: minimum chunk 4k); without lending the
    // read takes llama.cpp's ubatches
    return (o_.lend && buf_cache_) ? kChunkMin : o_.n_ubatch;
}

void Engine::headroom_for(int n_kv) {
    if (n_kv <= headroom_kv_ || !buf_cache_ || slot_bytes() == 0) return;
    const int target = n_kv + kHeadroomStep;
    const size_t need = ((size_t) 256 << 20) + (size_t) target * kComputePerCell;
    const size_t have = ggml_backend_sched_get_buffer_size(sched_, gpu_);
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    const size_t margin = (size_t) 128 << 20;
    if (have < need && fr + have < need + margin) ensure_free(need + margin - have);
    headroom_kv_ = target;
}

int Engine::grow_to(int S, size_t headroom) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (!buf_cache_ || slot_bytes() == 0 || S <= slots_now_) return slots_now_;
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    const size_t per = slot_bytes(), now = (size_t) (slots_now_ + 1) * per;   // what the current arena holds
    const int fit = fr + now > headroom + per ? (int) ((fr + now - headroom) / per) - 1 : slots_now_;
    int T = std::min(S, fit);
    if (T <= slots_now_) return slots_now_;
    const int before = slots_now_;
    std::string err;
    while (!(err = arena_build(T, false)).empty() && T > before) T = std::max(before, T - 2);
    if (!err.empty()) { fatal_ = "growing the arena: " + err; return 0; }
    fprintf(stderr, "impulse: expert arena %d -> %d slots per layer (VRAM free again)\n", before, T);
    refill_pending_ = true;
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_now_;
}

int Engine::ensure_free(size_t bytes, bool keep_profile) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (!buf_cache_ || slot_bytes() == 0) return slots_now_;
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    if (fr >= bytes) return slots_now_;
    if (copy_stream_) { cudaStreamSynchronize((cudaStream_t) copy_stream_); commit_landed(); }
    const size_t per = slot_bytes();
    int S = std::max(kMinSlots, slots_now_ - (int) ((bytes - fr + per - 1) / per) - 1);
    std::string err;
    while (!(err = arena_build(S, keep_profile)).empty() && S > kMinSlots) S = std::max(kMinSlots, S - 2);
    if (!err.empty()) { fatal_ = "making room: " + err; return 0; }
    if (keep_profile) return slots_now_;
    refill_pending_ = true;   // the next small ubatch refills every layer to its top-S by usage
    read_end_ = std::chrono::steady_clock::now(); refill_watch_ = true;
    return slots_now_;
}

// async swaps: newcomers whose copies have landed become resident (in order: batches complete in stream order)
void Engine::commit_landed() {
    std::vector<char> touched(hp_.n_layer, 0);
    size_t k = 0;
    for (; k < pending_.size(); ++k) {
        PendingBatch & b = pending_[k];
        if (!b.ev || cudaEventQuery((cudaEvent_t) b.ev) != cudaSuccess) break;
        for (const PendingSwap & p : b.sw) {
            CacheLayer * C = cache_[p.il].get();
            C->res[p.e] = p.slot; C->owner[p.slot] = p.e; C->pending[p.e] = 0;
            touched[p.il] = 1; stats_.commits++;
        }
        cudaEventDestroy((cudaEvent_t) b.ev);
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

// llama_decode: one batch, split into n_ubatch pieces; only the batch's last token outputs (llama_batch_get_one)
std::string Engine::decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last, int seq) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    cur_seq_ = seq;
    int & n_past_ = this->n_past_[seq];
    const int n = (int) toks.size();
    // auto chunks are for expert models (a chunk streams the RAM-resident experts once); a dense model reads in n_ubatch steps
    // (llama.cpp's, measured: larger ubatches do not read a dense Gemma faster)
    const int pu = (o_.prefill_ubatch < 0 && n >= kOffloadMin && hp_.n_expert > 0) ? pick_chunk(n) : std::max(0, o_.prefill_ubatch);
    const int ub = (pu > 0 && n >= kOffloadMin) ? pu : o_.n_ubatch;
    if (n >= kOffloadMin) stats_.last_chunk = std::min(ub, n);   // the chunk actually used (a short prompt is one chunk)
    if (o_.prefill_ubatch < 0 && n >= kOffloadMin)
        fprintf(stderr, "impulse: reading %d tokens at position %d in %d-token chunks (cap %d)\n", n, n_past_, std::min(ub, n), ub);
    const auto tr0 = std::chrono::steady_clock::now();
    if (ub != o_.n_ubatch || (o_.lend && n >= kOffloadMin)) begin_read(ub);
    for (int i = 0; i < n; i += ub) {
        const int m = std::min(ub, n - i);
        const bool out = (i + m == n);
        want_h_ = out && h_last;
        std::string err = ubatch(toks.data() + i, m, n_past_ + i, out, out ? logits : nullptr);
        want_h_ = false;
        if (!err.empty()) return err;
        if ((i == 0 || i + m == n) && m >= kOffloadMin)   // the first and the last chunk (the most KV cells)
            fprintf(stderr, "impulse: read compute buffers %.0f MiB (%d-token chunk, %u KV cells)\n",
                    ggml_backend_sched_get_buffer_size(sched_, gpu_) / 1048576.0, m, base_.n_kv);
    }
    if (h_last) *h_last = h_rows_;
    // the read's queued ubatches: waited for when this call reads output (the readback waits too); a piece of a longer read
    // that reads nothing stays queued, and the next ubatch on the same streams runs after it
    if (gpus_.size() > 1 && (logits || h_last)) ggml_backend_sched_synchronize(sched_);
    n_past_ += n;
    if (n >= kOffloadMin) stats_.prefill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    return "";
}

std::string Engine::decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h, int seq) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    cur_seq_ = seq;
    int & n_past_ = this->n_past_[seq];
    const int n = (int) toks.size();
    if (n > o_.n_ubatch) return "verify window larger than a ubatch";
    out_all_ = true; want_h_ = h != nullptr;
    std::string err = ubatch(toks.data(), n, n_past_, true, logits);
    out_all_ = false; want_h_ = false;
    if (!err.empty()) return err;
    if (h) *h = h_rows_;
    n_past_ += n;
    return "";
}

std::string Engine::decode_embd(const float * embd, int n, bool non_causal, std::vector<float> * logits, std::vector<float> * h_last, int seq) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (seq < 0 || seq >= n_seq()) return "no sequence " + std::to_string(seq);
    cur_seq_ = seq;
    int & n_past_ = this->n_past_[seq];
    // one ubatch: the window cache must hold n_swa + n cells for it (as a chunked read)
    // an image chunk of kOffloadMin tokens or more streams whole expert layers through the card like a text read, so it
    // borrows the arena's VRAM the same way (with a projector on a second card nothing else made that room)
    const bool big = n > o_.n_ubatch || (o_.lend && n >= kOffloadMin);
    if (big) begin_read(std::max(n, o_.n_ubatch));
    embd_in_ = embd; non_causal_ = non_causal; want_h_ = h_last != nullptr;
    std::string err = ubatch(nullptr, n, n_past_, true, logits);
    embd_in_ = nullptr; non_causal_ = false; want_h_ = false;
    if (!err.empty()) return err;
    if (h_last) *h_last = h_rows_;
    n_past_ += n;
    return "";
}

bool Engine::can_resume(int n, int seq) const {
    if (n <= 0) return true;
    std::vector<char> have(hp_.n_swa, 0);
    const int lo = std::max(0, n - hp_.n_swa);
    for (size_t i = 0; i < swa_.pos.size(); ++i) { const int32_t p = swa_.pos[i]; if (swa_.seq[i] == seq && p >= lo && p < n) have[p - lo] = 1; }
    for (int p = lo; p < n; ++p) if (!have[p - lo]) return false;
    return true;
}

void Engine::rollback(int n_past, int seq) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    if (seq < 0 || seq >= n_seq()) return;
    base_.seq_rm(seq, n_past); swa_.seq_rm(seq, n_past);
    n_past_[seq] = n_past;
    bool all_new = true; for (int q : n_past_) all_new = all_new && q == 0;
    if (all_new) kv_shrink();   // every conversation new (verify windows never roll back to 0): the KV gives the card back

}

// several sequences in one ubatch (server slots stepping together): each step's tokens at its sequence's next positions,
// every row an output (llama_batch with several seq_ids: the server's update_slots)
std::string Engine::decode_multi(const std::vector<SeqStep> & steps, std::vector<float> * logits, std::vector<float> * h) {
    cudaSetDevice(cuda_dev_);
    // split as llama_kv_cache splits a batch over its streams (split_equal, sequential): runs of consecutive sequences with the
    // same token count share a ubatch (seq-major), the others get their own; rows come back in the order of the steps
    const int nv = hp_.n_vocab;
    const int64_t ne = hp_.n_embd;
    std::vector<int> order(steps.size()), off(steps.size());
    int total = 0;
    for (size_t k = 0; k < steps.size(); ++k) {
        order[k] = (int) k; off[k] = total; total += (int) steps[k].toks.size();
        if (steps[k].seq < 0 || steps[k].seq >= n_seq() || steps[k].toks.empty()) return "bad step";
        for (size_t j = 0; j < k; ++j) if (steps[j].seq == steps[k].seq) return "two steps for one sequence";
    }
    if (total == 0) return "";
    std::sort(order.begin(), order.end(), [&](int a, int b) { return steps[a].seq < steps[b].seq; });
    if (logits) logits->assign((size_t) total * nv, 0.0f);
    if (h) h->assign((size_t) total * ne, 0.0f);
    std::vector<float> lg;
    for (size_t i = 0; i < order.size();) {
        const int len = (int) steps[order[i]].toks.size();
        size_t j = i + 1;
        while (j < order.size() && steps[order[j]].seq == steps[order[j - 1]].seq + 1 && (int) steps[order[j]].toks.size() == len &&
               (int) (j - i + 1) * len <= o_.n_ubatch) ++j;
        const int ns = (int) (j - i), n = ns * len;
        if (n > o_.n_ubatch) return "the steps are larger than a ubatch";
        std::vector<int32_t> toks; ub_pos_.clear(); ub_seq_.clear();
        for (size_t q = i; q < j; ++q) {
            const SeqStep & st = steps[order[q]];
            for (int t = 0; t < len; ++t) { toks.push_back(st.toks[t]); ub_pos_.push_back(n_past_[st.seq] + t); ub_seq_.push_back((int8_t) st.seq); }
        }
        ub_ns_ = ns; out_all_ = true; want_h_ = h != nullptr;
        std::string err = ubatch(toks.data(), n, 0, true, &lg);
        out_all_ = false; want_h_ = false; ub_ns_ = 0; ub_pos_.clear(); ub_seq_.clear();
        if (!err.empty()) return err;
        for (size_t q = i; q < j; ++q) {
            const int k = order[q], r0 = (int) (q - i) * len;
            if (logits) std::copy(lg.begin() + (size_t) r0 * nv, lg.begin() + (size_t) (r0 + len) * nv, logits->begin() + (size_t) off[k] * nv);
            if (h) std::copy(h_rows_.begin() + (size_t) r0 * ne, h_rows_.begin() + (size_t) (r0 + len) * ne, h->begin() + (size_t) off[k] * ne);
            n_past_[steps[k].seq] += len;
        }
        i = j;
    }
    return "";
}

// ---------------------------------------------------------------- stage f: the MTP drafter (llama.cpp src/models/gemma4-assistant.cpp)
std::string Engine::load_mtp(const std::string & path) {
    Draft & D = mtp_;
    gguf_init_params gp = { true, &D.meta };
    D.gguf = gguf_init_from_file(path.c_str(), gp);
    if (!D.gguf) return "cannot read " + path;
    const std::string A = "gemma4-assistant.";
    D.n_layer = (int) kv_int(D.gguf, A + "block_count");
    D.n_embd = (int) kv_int(D.gguf, A + "embedding_length");
    D.n_head = (int) kv_int(D.gguf, A + "attention.head_count");
    D.head_dim = (int) kv_int(D.gguf, A + "attention.key_length");
    D.head_dim_swa = (int) kv_int(D.gguf, A + "attention.key_length_swa");
    D.n_rot = (int) kv_int(D.gguf, A + "rope.dimension_count");
    D.n_rot_swa = (int) kv_int(D.gguf, A + "rope.dimension_count_swa");
    D.n_ctx_train = (int) kv_int(D.gguf, A + "context_length");
    D.rope_base = kv_f32(D.gguf, A + "rope.freq_base");
    D.rope_base_swa = kv_f32(D.gguf, A + "rope.freq_base_swa");
    D.eps = kv_f32(D.gguf, A + "attention.layer_norm_rms_epsilon");
    for (auto v : kv_arr(D.gguf, A + "attention.sliding_window_pattern")) D.is_swa.push_back(v != 0);
    if (kv_int(D.gguf, A + "embedding_length_out") != hp_.n_embd || (int) D.is_swa.size() != D.n_layer)
        return "MTP drafter does not match the target";
    const int n_tensors = (int) gguf_get_n_tensors(D.gguf);
    ggml_init_params ip = { (size_t) (n_tensors + 4) * ggml_tensor_overhead(), nullptr, true };
    D.ctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> src;
    const size_t data_off = gguf_get_data_offset(D.gguf);
    auto make = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * m = ggml_get_tensor(D.meta, name.c_str());
        if (!m) return nullptr;
        ggml_tensor * t = ggml_dup_tensor(D.ctx, m);
        ggml_format_name(t, "mtp.%s", name.c_str());
        src.push_back({ t, data_off + gguf_get_tensor_offset(D.gguf, gguf_find_tensor(D.gguf, name.c_str())) });
        return t;
    };
    D.tok_embd = make("token_embd.weight"); D.output_norm = make("output_norm.weight"); D.rope_freqs = make("rope_freqs.weight");
    D.pre_proj = make("nextn.pre_projection.weight"); D.post_proj = make("nextn.post_projection.weight");
    D.L.resize(D.n_layer);
    for (int il = 0; il < D.n_layer; ++il) {
        auto b = [&](const char * s2) { return "blk." + std::to_string(il) + "." + s2; };
        DraftLayer & l = D.L[il];
        l.attn_norm = make(b("attn_norm.weight")); l.wq = make(b("attn_q.weight")); l.wo = make(b("attn_output.weight"));
        l.q_norm = make(b("attn_q_norm.weight")); l.attn_post_norm = make(b("post_attention_norm.weight"));
        l.out_scale = make(b("layer_output_scale.weight")); l.ffn_norm = make(b("ffn_norm.weight"));
        l.ffn_gate = make(b("ffn_gate.weight")); l.ffn_up = make(b("ffn_up.weight")); l.ffn_down = make(b("ffn_down.weight"));
        l.ffn_post_norm = make(b("post_ffw_norm.weight"));
        if (!l.attn_norm || !l.wq || !l.wo || !l.ffn_down) return "MTP drafter layer " + std::to_string(il) + ": missing tensors";
    }
    if (!D.tok_embd || !D.pre_proj || !D.post_proj || !D.output_norm) return "MTP drafter: missing tensors";
    // on the card of the target's last layers: the drafter reads their caches (layers n-2 and n-1), so its graph stays on one card
    // with a tensor split: on the meta device, its attention sliced as the target layers' caches it reads (g4_tsplit)
    if (tp_) {
        if (D.n_head % hp_.n_head_kv[hp_.n_layer - 1] != 0 || D.n_head % hp_.n_head_kv[hp_.n_layer - 2] != 0)
            return "MTP with --split-mode tensor: the drafter's heads do not group over the target's KV heads";
        tp_info_->mtp_n_head = D.n_head;
        tp_info_->mtp_tl.clear();
        for (int il = 0; il < D.n_layer; ++il) tp_info_->mtp_tl.push_back(D.is_swa[il] ? hp_.n_layer - 2 : hp_.n_layer - 1);
    }
    ggml_backend_t dcard = tp_ ? gpu_ : gpus_[dev_layer_[hp_.n_layer - 1]];
    D.buf = ggml_backend_alloc_ctx_tensors_from_buft(D.ctx, ggml_backend_get_default_buffer_type(dcard));
    if (!D.buf) return "MTP drafter allocation failed";
    if (gpus_.size() > 1) fprintf(stderr, "impulse: the MTP drafter on %s (the target's last layers)\n", ggml_backend_name(dcard));
    ggml_backend_buffer_set_usage(D.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::vector<uint8_t> tmp;
    for (auto & [t, off] : src) {
        tmp.resize(ggml_nbytes(t));
        fseek(f, (long) off, SEEK_SET);
        if (fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) { fclose(f); return "short read in " + path; }
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
    }
    fclose(f);
    {   // the drafter's scheduler: its card(s) and the CPU (the target's token embeddings), the CPU side in pinned host memory
        std::vector<ggml_backend_t> backs;
        if (tp_) {
            D.backend = ggml_backend_dev_init(ggml_backend_get_device(gpu_), nullptr);   // a second instance of the meta backend
            if (!D.backend) return "MTP drafter: backend init failed";
            D.own_backend = true;
            backs.push_back(D.backend);
        } else backs = gpus_;   // its weights on one card; the target caches it reads may sit on another
        backs.push_back(cpu_);
        std::vector<ggml_backend_buffer_type_t> bufts;
        for (size_t i = 0; i + 1 < backs.size(); ++i) bufts.push_back(ggml_backend_get_default_buffer_type(backs[i]));
        bufts.push_back(lite::pinned_host_buffer_type(cpu_));
        D.sched = ggml_backend_sched_new(backs.data(), bufts.data(), (int) backs.size(), kMaxNodes, false, true);
        if (!D.sched) return "MTP drafter: scheduler failed";
    }
    return "";
}

ggml_tensor * Engine::norm_w(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), w);
}

ggml_tensor * Engine::lmm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * cur, ggml_tensor * s) const {
    ggml_tensor * res = ggml_mul_mat(ctx, w, cur);
    if (s) res = ggml_mul(ctx, res, s);   // the scale before the adapter's term (build_lora_mm)
    if (lora_.empty()) return res;
    auto it = lora_.find(w);
    if (it == lora_.end()) return res;
    ggml_tensor * ab = ggml_mul_mat(ctx, it->second.b, ggml_mul_mat(ctx, it->second.a, cur));
    ab = ggml_scale(ctx, ab, it->second.scale);
    return ggml_add(ctx, res, ab);
}

// llama_adapter_lora_init_impl, for the projections this engine builds: blk.N.{attn_q,attn_k,attn_v,attn_output,ffn_up,
// ffn_gate,ffn_down}.weight.lora_{a,b}, each pair beside its base weight on the layer's card (the base weight's buffer type)
std::string Engine::load_lora(const std::string & path, float user_scale) {
    if (tp_) return "--lora: not with --split-mode tensor";
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return "cannot read the LoRA adapter " + path;
    auto str = [&](const char * k) { const int64_t id = gguf_find_key(g, k); return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id)); };
    auto fail = [&](const std::string & e) { gguf_free(g); ggml_free(meta); return "LoRA adapter " + path + ": " + e; };
    if (str("general.type") != "adapter" || str("adapter.type") != "lora") return fail("not a LoRA adapter (general.type / adapter.type)");
    if (str("general.architecture") != "gemma4") return fail("made for " + str("general.architecture") + ", not gemma4");
    float alpha = 0.0f;
    { const int64_t id = gguf_find_key(g, "adapter.lora.alpha"); if (id >= 0) alpha = gguf_get_val_f32(g, id); }
    struct Pair { ggml_tensor * a = nullptr, * b = nullptr; int il = -1; };
    std::unordered_map<ggml_tensor *, Pair> pairs;   // base weight -> its file tensors
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        int il = -1; char kind[64] = { 0 }, ab = 0;
        if (sscanf(name.c_str(), "blk.%d.%63[a-z_].weight.lora_%c", &il, kind, &ab) != 3 || (ab != 'a' && ab != 'b') || il < 0 || il >= hp_.n_layer)
            return fail("unexpected tensor " + name);
        const Layer & l = L_[il];
        const std::string k = kind;
        ggml_tensor * w = k == "attn_q" ? l.wq : k == "attn_k" ? l.wk : k == "attn_v" ? l.wv : k == "attn_output" ? l.wo :
                          k == "ffn_up" ? l.ffn_up : k == "ffn_gate" ? l.ffn_gate : k == "ffn_down" ? l.ffn_down : nullptr;
        if (!w) return fail("tensor " + name + " has no base weight in this model");
        Pair & P = pairs[w];
        P.il = il;
        (ab == 'a' ? P.a : P.b) = ggml_get_tensor(meta, name.c_str());
    }
    lora_ctx_.assign(gpus_.size(), nullptr);
    lora_buf_.assign(gpus_.size(), nullptr);
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copy;   // (device tensor, file tensor)
    for (auto & [w, P] : pairs) {
        if (!P.a || !P.b) return fail(std::string("a lora_a / lora_b pair is incomplete for ") + w->name);
        // llama_adapter_lora_init_impl's shape checks: a [n_in, r], b [r, n_out]
        if (P.a->ne[0] != w->ne[0] || P.b->ne[1] != w->ne[1] || P.a->ne[1] != P.b->ne[0]) return fail(std::string("shape mismatch for ") + w->name);
        const int d = dev_layer_[P.il];
        if (!lora_ctx_[d]) {
            ggml_init_params ip = { (pairs.size() * 2 + 8) * ggml_tensor_overhead(), nullptr, true };
            lora_ctx_[d] = ggml_init(ip);
        }
        ggml_tensor * a = ggml_dup_tensor(lora_ctx_[d], P.a), * b = ggml_dup_tensor(lora_ctx_[d], P.b);
        ggml_set_name(a, P.a->name); ggml_set_name(b, P.b->name);
        copy.push_back({ a, P.a }); copy.push_back({ b, P.b });
        lora_[w] = { a, b, alpha ? user_scale * alpha / (float) P.b->ne[0] : user_scale };
    }
    for (size_t d = 0; d < gpus_.size(); ++d) {
        if (!lora_ctx_[d]) continue;
        lora_buf_[d] = ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx_[d], ggml_backend_get_default_buffer_type(gpus_[d]));
        if (!lora_buf_[d]) return fail("adapter allocation failed (GPU " + std::to_string(d) + ")");
        ggml_backend_buffer_set_usage(lora_buf_[d], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open");
    const size_t data = gguf_get_data_offset(g);
    std::vector<uint8_t> tmp;
    for (auto & [dt, ft] : copy) {
        const int64_t id = gguf_find_tensor(g, ft->name);
        tmp.resize(ggml_nbytes(dt));
        if (fseeko(f, (off_t) (data + gguf_get_tensor_offset(g, id)), SEEK_SET) != 0 || fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) { fclose(f); return fail("short read"); }
        ggml_backend_tensor_set(dt, tmp.data(), 0, tmp.size());
    }
    fclose(f);
    fprintf(stderr, "impulse: LoRA %s: %zu weights, alpha %.1f, scale %.3f\n", path.c_str(), lora_.size(), alpha,
            lora_.empty() ? 0.0f : lora_.begin()->second.scale);
    gguf_free(g); ggml_free(meta);
    return "";
}

std::string Engine::draft(int32_t tok, const float * h, int pos0, int32_t * out, std::vector<float> * h_out, int seq) {
    cudaSetDevice(cuda_dev_);   // a projector on a second card may have left another device current on this thread
    headroom_for(pos0 + 1);
    const auto td0 = std::chrono::steady_clock::now();
    struct T { Stats & s; std::chrono::steady_clock::time_point t; ~T() { s.t_drafts++; s.t_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); } } timer{ stats_, td0 };
    const Draft & D = mtp_;
    if (!D.buf) return "no MTP drafter loaded";
    const int64_t nb = hp_.n_embd;
    const int dst = base_.stream_of(seq);   // the sequence's KV streams (the drafter reads the target's caches)
    const uint32_t nkv_b = base_.calc_n_kv(dst, 1), nkv_s = swa_.calc_n_kv(dst, 1);
    // graph reuse, as the target's ubatch: a draft step after a draft step (the same cells) sets its inputs and computes
    const GraphKey key = { 1, (int) nkv_b, (int) nkv_s, 1, false, false, 1, dst };
    const bool reuse = dg_.valid && dg_.key == key && graph_reuse_on() && meta_views_live(dg_);
    if (!reuse) {   // ---- build (to "// ---- build ends")
    graph_drop(dg_);
    ggml_init_params ip = { draft_graph_buf_.size(), draft_graph_buf_.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kMaxNodes, false);
    ggml_backend_sched_reset(D.sched);
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ggml_set_input(t_tok);
    ggml_tensor * t_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nb, 1); ggml_set_input(t_h);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ggml_set_input(t_pos);
    // the shared caches: the drafter's window layers read the target's layer n-2, its global layer layer n-1
    const int L_swa = hp_.n_layer - 2, L_glb = hp_.n_layer - 1;
    ggml_tensor * m_b = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv_b, 1, 1, 1); ggml_set_input(m_b);
    ggml_tensor * m_s = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv_s, 1, 1, 1); ggml_set_input(m_s);

    const bool rk = ggml_is_quantized(o_.type_k), rv = ggml_is_quantized(o_.type_v);
    ggml_tensor * dk_rot = nullptr, * dv_rot = nullptr, * dk_rot_swa = nullptr, * dv_rot_swa = nullptr;
    if (rk) { int r = k_rot_size(hp_.head_dim); dk_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(dk_rot);
              r = k_rot_size(hp_.head_dim_swa); dk_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, r); ggml_set_input(dk_rot_swa); }
    if (rv) { dv_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(dv_rot);
              dv_rot_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64); ggml_set_input(dv_rot_swa); }
    ggml_tensor * x = ggml_get_rows(ctx, tok_embd_, t_tok);
    x = ggml_scale(ctx, x, sqrtf((float) nb));
    ggml_tensor * cur = ggml_mul_mat(ctx, D.pre_proj, ggml_concat(ctx, x, t_h, 0));
    ggml_tensor * inpL = cur;
    for (int il = 0; il < D.n_layer; ++il) {
        const DraftLayer & l = D.L[il];
        const bool swa = D.is_swa[il];
        const int tl = swa ? L_swa : L_glb;
        Cache & C = swa ? swa_ : base_;
        const int64_t hd = swa ? D.head_dim_swa : D.head_dim, nkv_h = hp_.n_head_kv[tl], n_kv = swa ? nkv_s : nkv_b;
        ggml_tensor * q = ggml_mul_mat(ctx, l.wq, norm_w(ctx, inpL, l.attn_norm, D.eps));
        q = ggml_reshape_3d(ctx, q, hd, D.n_head, 1);
        q = norm_w(ctx, q, l.q_norm, D.eps);
        q = ggml_rope_ext(ctx, q, t_pos, swa ? nullptr : D.rope_freqs, swa ? D.n_rot_swa : D.n_rot, kRopeType, D.n_ctx_train,
                          swa ? D.rope_base_swa : D.rope_base, 1.0f, kExtFactor, kAttnFactor, kBetaFast, kBetaSlow);
        ggml_tensor * kr = swa ? dk_rot_swa : dk_rot, * vr = swa ? dv_rot_swa : dv_rot;   // the shared caches' rotations
        if (kr) q = mul_mat_hadamard(ctx, q, kr);
        const int64_t n_gqa = hd * nkv_h;
        const size_t kso = ggml_row_size(C.k[tl]->type, n_gqa * C.ssize), vso = ggml_row_size(C.v[tl]->type, n_gqa * C.ssize);
        ggml_tensor * k = ggml_view_4d(ctx, C.k[tl], hd, nkv_h, n_kv, 1, ggml_row_size(C.k[tl]->type, hd),
                                       ggml_row_size(C.k[tl]->type, n_gqa), kso, (size_t) dst * kso);
        ggml_tensor * v = ggml_view_4d(ctx, C.v[tl], hd, nkv_h, n_kv, 1, ggml_row_size(C.v[tl]->type, hd),
                                       ggml_row_size(C.v[tl]->type, n_gqa), vso, (size_t) dst * vso);
        ggml_tensor * a = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                              ggml_permute(ctx, v, 0, 2, 1, 3), swa ? m_s : m_b, 1.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
        if (vr) a = mul_mat_hadamard(ctx, a, vr);
        a = ggml_mul_mat(ctx, l.wo, a);
        ggml_tensor * attn_out = ggml_add(ctx, norm_w(ctx, a, l.attn_post_norm, D.eps), inpL);
        ggml_tensor * f = norm_w(ctx, attn_out, l.ffn_norm, D.eps);
        ggml_tensor * g = ggml_geglu_split(ctx, ggml_mul_mat(ctx, l.ffn_gate, f), ggml_mul_mat(ctx, l.ffn_up, f));
        f = ggml_mul_mat(ctx, l.ffn_down, g);
        cur = ggml_add(ctx, norm_w(ctx, f, l.ffn_post_norm, D.eps), attn_out);
        inpL = ggml_mul(ctx, cur, l.out_scale);
    }
    cur = norm_w(ctx, inpL, D.output_norm, D.eps);
    ggml_tensor * best = ggml_argmax(ctx, ggml_mul_mat(ctx, D.tok_embd, cur));   // the draft: top-1 of its logits
    ggml_tensor * h_next = ggml_mul_mat(ctx, D.post_proj, cur);
    ggml_set_output(best); ggml_set_output(h_next);
    ggml_build_forward_expand(gf, best);
    ggml_build_forward_expand(gf, h_next);
    if (!ggml_backend_sched_alloc_graph(D.sched, gf)) { ggml_free(ctx); return "draft graph allocation failed"; }
    dg_.ctx = ctx; dg_.gf = gf; dg_.key = key; dg_.valid = true; dg_.built_at = ++meta_builds_;
    dg_.t_tok = t_tok; dg_.t_h = t_h; dg_.inp_pos = t_pos; dg_.kq_mask = m_b; dg_.kq_mask_swa = m_s;
    dg_.k_rot = dk_rot; dg_.v_rot = dv_rot; dg_.k_rot_swa = dk_rot_swa; dg_.v_rot_swa = dv_rot_swa;
    dg_.best = best; dg_.h_next = h_next;
    stats_.graphs_built++;
    } else stats_.graphs_reused++;   // ---- build ends
    const Graph & G = dg_;
    ggml_backend_tensor_set(G.t_tok, &tok, 0, sizeof(int32_t));
    ggml_backend_tensor_set(G.t_h, h, 0, nb * sizeof(float));
    ggml_backend_tensor_set(G.inp_pos, &pos0, 0, sizeof(int32_t));
    auto mask = [&](const Cache & C, uint32_t n_kv, ggml_tensor * m) {
        std::vector<ggml_fp16_t> md(n_kv);
        for (uint32_t j = 0; j < n_kv; ++j) {
            const int32_t p = C.pos[(size_t) dst * C.ssize + j];   // the sequence's stream
            md[j] = ggml_fp32_to_fp16((p < 0 || p > pos0 || (C.swa && pos0 - p >= hp_.n_swa)) ? -INFINITY : 0.0f);
        }
        ggml_backend_tensor_set(m, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
    };
    mask(base_, nkv_b, G.kq_mask); mask(swa_, nkv_s, G.kq_mask_swa);
    for (ggml_tensor * r : { G.k_rot, G.v_rot, G.k_rot_swa, G.v_rot_swa }) {
        if (!r) continue;
        const std::vector<float> h = gen_hadamard((int) r->ne[0]);
        ggml_backend_tensor_set(r, h.data(), 0, h.size() * sizeof(float));
    }
    const ggml_status st = ggml_backend_sched_graph_compute(D.sched, G.gf);
    if (st != GGML_STATUS_SUCCESS) { graph_drop(dg_); return "draft compute failed"; }
    ggml_backend_tensor_get(G.best, out, 0, sizeof(int32_t));
    if (*out < 0 || *out >= hp_.n_vocab) return "draft token out of range";   // NaN logits (argmax -1): no draft, never a token
    h_out->resize(nb);
    ggml_backend_tensor_get(G.h_next, h_out->data(), 0, nb * sizeof(float));
    return "";
}

}  // namespace g4
