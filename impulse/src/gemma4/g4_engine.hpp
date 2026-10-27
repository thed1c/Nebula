// Impulse: the gemma4 engine (Gemma 4 26B-A4B). Gate 1 stage b: the block graph, built from ggml ops so it runs the
// kernels llama.cpp runs, op for op in llama.cpp's order (src/models/gemma4.cpp at b6b003d), the KV cells and masks as
// llama.cpp places them (llama-kv-cache.cpp find_slot/apply_ubatch), the placement of `-ngl 99 --n-cpu-moe N`.
// Later stages swap the MoE layer for Strata's GPU expert cache + CPU pool; everything else stays.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; struct NativeFmt; }

namespace g4 {

struct TsplitInfo;

struct Options {
    std::string gguf;
    int  n_ctx      = 32768;
    int  n_batch    = 2048;     // llama.cpp -b: a decode() call is split into ubatches of n_ubatch
    int  n_ubatch   = 512;
    int  n_cpu_moe  = 30;       // the first N layers' experts (+ their scales) stay in RAM, the rest go to the GPU
    int  n_threads  = 18;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    std::string graph_dump;     // write the second single-token graph's nodes (as route-log's ROUTE_GRAPHDUMP)
    bool pool         = false;  // stage c: CPU-resident experts of small ubatches through Strata's expert pool
    int  pool_threads = 16;
    // stage d: a GPU expert cache in front of the pool for the RAM-resident layers [cache_l0, n_cpu_moe)
    int  cache_slots  = 0;      // resident experts per layer (0 = off; >= n_expert = the whole layer, llama.cpp's GPU path;
                                // -1 = auto: the most the free VRAM holds after vram_reserve_mb)
    int  vram_reserve_mb = 1024;  // auto slots: VRAM left free for what loads after the arena and for compute growth
    int  cache_l0     = 0;
    std::string profile;        // per layer, experts ranked by use (tools/g4_profile.py); the top cache_slots are resident
    int  dcheck_layer = -1;     // a cached layer whose merged rows are checked against all-GPU and all-CPU twins
    // stage e: adaptation. Decayed usage counts per layer; after a prompt read every layer is re-set to its top
    // cache_slots; after each small ubatch at most adapt_budget swaps per layer (a newcomer must score above the victim)
    bool  adapt           = false;
    float adapt_half_life = 32.0f;   // tokens
    int   adapt_budget    = 2;       // swaps per layer per decode step
    bool  adapt_refill    = true;    // re-set every layer to its top-S after a prompt read
    bool  async_swaps     = false;   // copies on their own CUDA stream from the pinned mapping; commit when they land
    double ram_budget_gib = -1;    // the RAM tier: GiB of experts locked in RAM (-1 auto, see src/lite/ram_tier.hpp)
    bool  overlap         = false;   // the misses run in the pool while the GPU computes the hits (start/join ops)
    // stage g: the chunk test. Prompt reads in ubatches of prefill_ubatch tokens (0 = n_ubatch); with lend, the slot
    // arena is freed for the read (its VRAM funds the compute buffers and the enlarged window cache) and refilled after
    int   prefill_ubatch  = 0;      // -1 = auto: per read, the largest of 32k / 16k / 8k / 4k whose compute fits the VRAM the read can borrow
    bool  lend            = false;
    // the KV pager: the global layers' K/V start with kv_initial cells on the card (0 = the whole context at load, as
    // llama.cpp) and grow in kv_step steps as the conversation needs them; the slot arena gives way (ensure_free). The
    // window layers keep their own small cache.
    int   kv_initial      = 0;
    int   kv_step         = 16384;
    // v0.2.4: several GPUs, as llama.cpp's --split-mode layer --tensor-split: layer il (and its KV) on the first card whose
    // cumulative share exceeds il / (n_layer + 1), the output on the card of "layer" n_layer. Empty = one GPU. Dense models only.
    std::vector<float> tensor_split;
    bool split_tensor = false;    // --split-mode tensor (llama.cpp's meta backend): every layer split across the cards
    // v0.2.4: sequences (server slots) sharing the KV, as llama.cpp's unified cache: every cell belongs to one sequence and a
    // token sees only its own; the window cache holds n_swa per sequence. 1 = one conversation, as before.
    int n_seq = 1;
    // a LoRA adapter (llama.cpp --lora / --lora-scaled): W x + scale B (A x) on the attention and FFN projections, scale =
    // lora_scale * alpha / rank (llama_adapter_lora_weight::get_scale); one card or a layer split
    std::string lora;
    float lora_scale = 1.0f;
};

struct Stats {
    uint64_t accesses = 0, hits = 0;       // routed (token, expert) pairs of the RAM-resident layers in small ubatches
    uint64_t dcheck_rows = 0, dcheck_gpu_rows = 0, dcheck_bad = 0;
    uint64_t swaps = 0, refill_swaps = 0, swap_bytes = 0;    // stage e
    double   swap_ms = 0, refill_ms = 0;
    uint64_t commits = 0, commit_batches = 0, pending_peak = 0;
    uint64_t ov_layers = 0, ov_missfree = 0;        // overlap: layer-steps, and those with no miss at all
    double   t_join_wait = 0;                       // overlap: host time blocked in join (the pool not done yet)
    // stage g
    double   prefill_ms = 0, lend_ms = 0, restore_ms = 0;   // the read itself; freeing/growing; shrinking/re-allocating
    uint64_t lent_bytes = 0, swa_big_cells = 0;
    std::vector<uint32_t> step_acc, step_hit;      // per small ubatch after the read: routed pairs and hits
    int      slots_now = 0;                        // resident slots per layer now (auto sizing, shrinks under VRAM pressure)
    int      arena_shrinks = 0;
    int      last_chunk = 0;                       // the chunk the last prompt read used
    double   refill_land_ms = -1;                  // from the end of the read until the last refill copy committed   // async swaps: newcomers committed when their copy landed
    // the per-token timeline (small ubatches only: decode steps and verify windows), host wall clock, ms
    uint64_t t_ubatches = 0, t_splits = 0;
    double   t_build = 0, t_compute = 0, t_pool = 0, t_readout = 0, t_scores = 0, t_draft = 0;
    uint64_t t_drafts = 0;
    uint64_t graphs_built = 0, graphs_reused = 0;   // v0.2.3: graphs built (ubatches and draft steps), and steps that reused the last one
};

struct HParams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_ff = 0, n_ff_exp = 0, n_expert = 0, n_expert_used = 0, n_vocab = 0;
    int n_swa = 0, head_dim = 0, head_dim_swa = 0, n_rot = 0, n_rot_swa = 0, n_ctx_train = 0;
    std::vector<int>  n_head_kv;
    std::vector<bool> is_swa;
    float rope_base = 0, rope_base_swa = 0, eps = 0, softcap = 0;
};

class Engine {
public:
    Engine();
    ~Engine();
    std::string load(const Options & o);                   // "" on success
    // Feed tokens at the next positions (one llama_decode call); fills `logits` with the last token's logits
    // (and `h_last` with its post-final-norm hidden state, the MTP drafter's input, when given).
    std::string decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last = nullptr, int seq = 0);
    // stage f: a verify window (<= n_ubatch tokens, one ubatch): logits [n][n_vocab] and hidden rows [n][n_embd] of every row
    std::string decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h, int seq = 0);
    // several sequences in one ubatch (server slots stepping together): each step's tokens at its sequence's next positions,
    // every row an output; logits (and hidden rows) in step order
    struct SeqStep { int seq; std::vector<int32_t> toks; };
    std::string decode_multi(const std::vector<SeqStep> & steps, std::vector<float> * logits, std::vector<float> * h = nullptr);
    // forget every position >= n_past (llama_memory_seq_rm), e.g. a verify window's rejected drafts
    void rollback(int n_past, int seq = 0);
    // a prompt cache may resume at n_past = n only if the window cache still holds positions [n - n_swa, n)
    // (llama-server's SWA check); else the conversation is read again from the start
    bool can_resume(int n, int seq = 0) const;
    int n_ctx() const { return (int) base_.size; }
    int kv_cap() const { return (int) kv_cap_; }   // global-layer KV cells on the card now (<= n_ctx)
    // vision: an image chunk's embeddings [n][n_embd] (mtmd), one ubatch; non_causal = mtmd_decode_use_non_causal
    // (Gemma: the sliding-window layers attend both ways inside the image, the global layers stay causal)
    std::string decode_embd(const float * embd, int n, bool non_causal, std::vector<float> * logits, std::vector<float> * h_last, int seq = 0);
    // stage f: the MTP drafter (gemma4-assistant GGUF): Q-only attention over the target's last two layers' KV
    std::string load_mtp(const std::string & path);
    // one draft step at position pos0 (all of a window's drafts share it): token + target-width h -> argmax + next h
    std::string draft(int32_t tok, const float * h, int pos0, int32_t * out, std::vector<float> * h_out, int seq = 0);
    // make at least `bytes` of VRAM free (e.g. before an image encode) by giving up expert slots; the arena comes back
    // empty and refills from the usage counts on the next step. Returns the slots per layer now.
    int ensure_free(size_t bytes, bool keep_profile = false);
    // give slots back when VRAM is free again (e.g. the projector left the card): at most S per layer, as many as fit
    // with `headroom` free; the arena comes back empty and refills from the usage counts
    int grow_to(int S, size_t headroom);
    int slots() const { return slots_now_; }
    const HParams & hp() const { return hp_; }
    size_t ram_locked() const { return ram_locked_; }     // the RAM tier: expert bytes locked in RAM
    size_t ram_experts() const { return ram_experts_; }   // and of all CPU-side experts
    bool async_swaps() const { return o_.async_swaps; }    // false in low-RAM mode
    int n_past(int s = 0) const { return n_past_[s]; }
    int n_seq() const { return (int) n_past_.size(); }

    struct PoolLayer;            // one layer's experts as the pool reads them (in g4_engine.cpp)
    struct CacheLayer;           // one layer's GPU expert slots (stage d)
    struct Driver;               // overlap: the thread that runs the pool for a start op
    std::unique_ptr<Driver> driver_;
    const Stats & stats() const { return stats_; }
    Stats stats_;

private:
    struct Layer {
        ggml_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm, *attn_post_norm, *out_scale;
        ggml_tensor *ffn_norm, *ffn_gate, *ffn_up, *ffn_down, *ffn_post_norm;
        ggml_tensor *gate_inp, *gate_inp_s, *pre_norm_2, *post_norm_1, *post_norm_2;
        ggml_tensor *gate_up_exps, *down_exps, *down_exps_s;
        // the optional second-level scales llama.cpp's generic pass loads ("<weight>.scale", NVFP4's weight_scale_2) and
        // gemma4.cpp applies: build_qkv / build_attn (Q, K, V, output) and build_ffn (up, gate, down); nullptr = none.
        // (down_exps_s above is the same kind of tensor by name: llama.cpp's Gemma 4 per-expert scale)
        ggml_tensor *wq_s, *wk_s, *wv_s, *wo_s, *ffn_up_s, *ffn_gate_s, *ffn_down_s;
    };
    struct Cache {                                         // one llama_kv_cache (base: global layers, swa: window layers)
        // one stream per sequence (llama-server's --parallel N: kv_unified false): stream s owns cells [s * ssize, (s + 1) * ssize)
        bool swa = false;
        uint32_t size = 0, ssize = 0;
        std::vector<uint32_t> heads;                       // per stream: where the next search starts
        std::vector<int32_t> pos;                          // -1 = empty
        std::vector<int8_t> seq;                           // the cell's sequence (a server slot); -1 = empty
        std::vector<ggml_tensor *> k, v;                   // by layer; nullptr when the layer is in the other cache
        std::vector<int64_t> idxs;                         // the current ubatch's cells
        uint32_t n_kv = 0;
        void init(uint32_t cells_per_stream, int n_streams);
        int  stream_of(int s) const { return heads.size() > 1 ? s : 0; }
        uint32_t used(int st) const;
        uint32_t used_max_p1(int st) const;
        int32_t  pos_min(int s) const;                     // of sequence s (-1 when it has no cells)
        int32_t  pos_max(int s) const;
        bool find_slot(uint32_t n_tokens, int n_swa, int st);   // appends the cells to idxs
        void apply(const std::vector<int32_t> & positions, const std::vector<int8_t> & seqs);
        void seq_rm(int s, int32_t p0);                    // s < 0: every sequence
        uint32_t calc_n_kv(int st0, int nst) const;        // llama_kv_cache::get_n_kv over the ubatch's streams
    };
    struct DraftLayer { ggml_tensor *attn_norm, *wq, *wo, *q_norm, *attn_post_norm, *out_scale, *ffn_norm, *ffn_gate, *ffn_up, *ffn_down, *ffn_post_norm; };
    struct Draft {
        gguf_context * gguf = nullptr; ggml_context * meta = nullptr, * ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
        int n_layer = 0, n_embd = 0, n_head = 0, head_dim = 0, head_dim_swa = 0, n_rot = 0, n_rot_swa = 0, n_ctx_train = 0;
        std::vector<bool> is_swa; float rope_base = 0, rope_base_swa = 0, eps = 0;
        ggml_tensor *tok_embd = nullptr, *output_norm = nullptr, *rope_freqs = nullptr, *pre_proj = nullptr, *post_proj = nullptr;
        std::vector<DraftLayer> L;
        // its own scheduler (the target's graph and the drafter's both stay allocated and reused); with a tensor split its own
        // meta backend too, so each keeps its split graph (the meta backend re-splits a graph it did not run last)
        ggml_backend_sched_t sched = nullptr;
        ggml_backend_t backend = nullptr;
        bool own_backend = false;
    } mtp_;
    bool out_all_ = false, want_h_ = false;
    const float * embd_in_ = nullptr; bool non_causal_ = false;
    std::vector<float> h_rows_;

    std::string ubatch(const int32_t * toks, int n, int pos0, bool output, std::vector<float> * logits);
    ggml_tensor * norm_w(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, float eps);
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, int il);

    Options o_;
    HParams hp_;
    gguf_context * gguf_ = nullptr;
    ggml_context * meta_ = nullptr;                        // tensor metadata from the GGUF
    ggml_context * ctx_w_ = nullptr;                       // the weights as the engine places them
    void *   map_ = nullptr;
    size_t   map_size_ = 0;
    ggml_backend_t gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_buffer_t buf_gpu_ = nullptr, buf_map_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    ggml_tensor *tok_embd_ = nullptr, *output_ = nullptr, *output_norm_ = nullptr, *rope_freqs_ = nullptr;
    ggml_tensor *output_s_ = nullptr;   // output.scale (llama.cpp loads it for an NVFP4 output only)
    std::vector<Layer> L_;
    Cache base_, swa_;
    std::vector<int> n_past_ = std::vector<int>(1, 0);   // per sequence (Options::n_seq server slots)
    std::vector<int32_t> ub_pos_; std::vector<int8_t> ub_seq_; int ub_ns_ = 0;   // decode_multi: the ubatch's positions, sequences (seq-major), how many
    int cur_seq_ = 0;                                   // the sequence of a single-sequence call
    int n_ubatch_graphs_ = 0;
    std::vector<uint8_t> graph_buf_;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_;
    std::vector<std::unique_ptr<PoolLayer>> pool_layers_;
    std::vector<std::unique_ptr<CacheLayer>> cache_;
    struct DCheck { ggml_tensor * merged = nullptr, * twin_gpu = nullptr, * twin_cpu = nullptr, * argsort = nullptr; CacheLayer * C = nullptr; };
    DCheck dcheck_;
    // graph reuse (v0.2.3, llama_context::process_ubatch's can_reuse): the last computed graph stays allocated on the
    // scheduler, and a step of the same shape sets the inputs and computes it again instead of building a new one. The
    // shape: tokens, padded KV cells (both caches), output rows, the embedding branch, the hidden-row output. The
    // target's ubatch and the drafter's step share sched_, so each keeps its own graph and the scheduler holds one of
    // them at a time (sched_graph_). Anything that moves a tensor a graph points at (a new scheduler, the slot arena, the
    // KV pager, the window cache) drops both (graph_invalidate).
    struct GraphKey {
        int n = -1, n_kv = 0, n_kv_swa = 0, n_outputs = 0; bool embd = false, want_h = false; int ns = 1, st0 = 0;
        bool operator==(const GraphKey & o) const {
            return n == o.n && n_kv == o.n_kv && n_kv_swa == o.n_kv_swa && n_outputs == o.n_outputs && embd == o.embd && want_h == o.want_h &&
                   ns == o.ns && st0 == o.st0;
        }
    };
    struct Graph {
        ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; bool valid = false; GraphKey key;
        ggml_tensor *inp_tokens = nullptr, *inp_embd = nullptr, *inp_pos = nullptr, *k_idxs = nullptr, *v_idxs = nullptr, *kq_mask = nullptr,
                    *k_idxs_swa = nullptr, *v_idxs_swa = nullptr, *kq_mask_swa = nullptr, *k_rot = nullptr, *v_rot = nullptr,
                    *k_rot_swa = nullptr, *v_rot_swa = nullptr, *inp_out_ids = nullptr, *t_logits = nullptr, *t_h = nullptr;
        ggml_tensor *t_tok = nullptr, *best = nullptr, *h_next = nullptr;   // the drafter's
        std::vector<ggml_tensor *> adapt_ids;   // the cached layers' routing (adapt)
        DCheck dcheck;                          // the dcheck layer's twins
        uint64_t acc_inc = 0, hit_inc = 0;      // routed pairs the shape alone decides, counted every step
        uint64_t built_at = 0;                  // meta_builds_ after its build
    } g_, dg_;                                  // the target's and the drafter's
    Graph * sched_graph_ = nullptr;             // whose allocation the scheduler holds
    std::vector<uint8_t> draft_graph_buf_;
    static void graph_drop(Graph & g) { if (g.ctx) ggml_free(g.ctx); g.ctx = nullptr; g.gf = nullptr; g.valid = false; }
    void graph_invalidate() { g_.valid = false; dg_.valid = false; sched_graph_ = nullptr; }
    // --split-mode tensor: llama.cpp's meta backend keeps the per-card copies of views into a shared buffer (the KV caches) in two
    // rotating sets, and every graph rebuild, on any meta backend, clears the older set. The target's and the drafter's graphs
    // both view the caches, so a graph stays reusable only while at most one other graph was built since it (two slots: the
    // drafter rebuilds at every slot switch, and a second rebuild cleared the target's views under its reused graph)
    uint64_t meta_builds_ = 0;
    bool meta_views_live(const Graph & g) const { return !tp_ || meta_builds_ - g.built_at <= 1; }
    void adapt_after(int n, const std::vector<ggml_tensor *> & argsorts);
    void swap_layer(int il, int budget);
    bool refill_pending_ = false;
    void commit_landed();
    void begin_read(int ubatch);
    void end_read();
    void resize_swa(uint32_t cells);
    bool reading_ = false, refill_watch_ = false;
    std::chrono::steady_clock::time_point read_end_;
    std::vector<ggml_context *> ctx_swa_; std::vector<ggml_backend_buffer_t> buf_swa_;   // the window layers' K/V, one per card
    // several GPUs (Options::tensor_split): gpus_[0] == gpu_; each layer's card, the output's card, each card's rope_freqs copy
    std::vector<ggml_backend_t> gpus_;
    bool tp_ = false;                                   // --split-mode tensor: gpu_ is the meta backend over the cards
    struct LoraW { ggml_tensor * a = nullptr, * b = nullptr; float scale = 1.0f; };
    std::unordered_map<const ggml_tensor *, LoraW> lora_;   // base weight -> its adapter pair (llama_adapter_lora::get_weight)
    std::vector<ggml_context *> lora_ctx_;
    std::vector<ggml_backend_buffer_t> lora_buf_;
    std::string load_lora(const std::string & path, float user_scale);
    // llm_graph_context::build_lora_mm: mul_mat, then each adapter's B (A x) scaled and added
    // build_lora_mm: w x cur, times w's second-level scale s (when there is one), plus the adapter's term
    ggml_tensor * lmm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * cur, ggml_tensor * s = nullptr) const;
    std::unique_ptr<TsplitInfo> tp_info_;
    std::vector<int> dev_layer_;
    int dev_out_ = 0;
    std::vector<ggml_tensor *> rope_freqs_d_;
    std::vector<ggml_backend_buffer_t> buf_gpu_x_;          // the weights of cards 1.. (card 0's: buf_gpu_)
    void new_sched();
    std::string reserve_graphs();
    bool reserve_ = false;
    std::vector<ggml_tensor *> kfull_, vfull_;   // full-size placeholders of the paged K/V (reserve graphs)
    ggml_context * ctx_kvfull_ = nullptr;
    std::string alloc_swa(uint32_t cells, std::vector<ggml_tensor *> & k, std::vector<ggml_tensor *> & v,
                          std::vector<ggml_context *> & ctxs, std::vector<ggml_backend_buffer_t> & bufs);
    struct PendingSwap { int il, e, slot; };
    struct PendingBatch { void * ev; std::vector<PendingSwap> sw; };
    std::vector<PendingBatch> pending_;
    void * copy_stream_ = nullptr;
    bool pinned_ = false;
    void * arena_ = nullptr;                 // Strata's expert arena: the RAM-resident experts in page-locked memory
    size_t arena_size_ = 0;
    bool arena_placed_ = false;   // numa-always: the arena is placed per socket (lite::ram_tier_place; freed by ram_tier_free)
    size_t ram_locked_ = 0, ram_experts_ = 0;   // the RAM tier: bytes locked / bytes of CPU-side experts
    ggml_backend_buffer_t buf_arena_ = nullptr;
    ggml_context * ctx_cache_ = nullptr;
    ggml_backend_buffer_t buf_cache_ = nullptr;
    std::vector<std::vector<int>> rank_;     // the profile: per layer, experts ranked by use
    int slots_now_ = 0;
    int cuda_dev_ = 0;                       // the engine's card (CUDA calls outside ggml are pinned to it)
    std::string fatal_;                      // set when the arena cannot come back at all; every later decode fails with it
    std::string arena_build(int S, bool fill_profile);
    int headroom_kv_ = 0;                    // the context length the free VRAM has been checked for
    void headroom_for(int n_kv);
    int pick_chunk(int n);                   // auto prefill_ubatch for a read of n tokens at the current position             // before a small step: room for its compute buffer at this context, else fewer slots
    size_t slot_bytes() const;               // one slot in every cached layer
    // the KV pager (global layers): each one's K/V in its own buffer of kv_cap_ cells (a multiple of 256, <= base_.size)
    struct KvLayer { ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; };
    std::vector<KvLayer> kvl_;
    uint32_t kv_cap_ = 0;
    int slots_load_ = 0;                     // the slot count at load: a new conversation grows the arena back to it
    size_t kv_bytes(uint32_t cells) const;
    std::string kv_alloc_layer(int il, uint32_t cap);
    std::string kv_grow(uint32_t cells);
    void kv_shrink();                        // a new conversation: back to kv_initial cells, the arena regrown
};

}  // namespace g4
