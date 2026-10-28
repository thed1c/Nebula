// Impulse hybrid MoE engine: Qwen3.5 / 3.6 (DeltaNet, gated attention, shared expert),
// LFM2.5 (short convolution, attention, leading dense FFNs, biased sigmoid router) and Qwen3.8 Flash-Next (llama.cpp's
// qwen4exp: hyper-connections, an indexer choosing the attention cells, n-gram per-layer embeddings). llama.cpp's graph (src/models/qwen35moe.cpp, delta-net-base.cpp, llama-graph.cpp) rebuilt op for op with ggml,
// so every stage compares bit for bit with llama.cpp where the arithmetic is the same; in front of the RAM-resident
// experts, Strata's GPU expert cache, CPU pool, adaptation, MTP drafting and chunked reads (docs/QWEN36.md).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "../lite/lfm_lora.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; }

namespace q35 {

struct TsplitInfo;
struct Options;
// --lora FILE (repeatable) and --lora-scaled FILE:SCALE[,FILE:SCALE...] (llama-server's forms): the first adapter is
// Options::lora, the rest Options::loras, in order. "" = ok, else what is wrong.
std::string add_lora_arg(Options & o, const std::string & value, bool scaled);

struct Options {
    std::string gguf;
    std::string lora;             // opt-in LFM adapter (lfm2moe: rank-one mixer-output/expert-down)
    float lora_scale = 1.0f;
    std::vector<std::pair<std::string, float>> loras;
    bool cpu_only = false;        // --device cpu: every weight (the file mapped in place), the caches and the graph on the CPU;
                                  // dense models (Halo's LFM2.5-1.2B where no card is spare), no CUDA call after load   // more adapters, stacked in order after `lora` (dense lfm2: any rank, every linear)
    int  n_ctx     = 32768;
    int  n_ubatch  = 512;
    int  n_cpu_moe = 40;        // the first N layers' experts stay in RAM (llama.cpp --n-cpu-moe)
    int  n_threads = 16;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    bool pool        = false;   // stage c: RAM-resident experts of small ubatches through Strata's expert pool
    int  pool_threads = 16;
    // stage d: a GPU expert cache in front of the pool for the RAM-resident layers [cache_l0, n_cpu_moe)
    int  cache_slots  = 0;      // resident experts per layer (0 = off; >= n_expert = the whole layer, llama.cpp's GPU path;
                                // -1 = auto: as many as the free VRAM holds after everything else, less vram_reserve_mb)
    int  vram_reserve_mb = 512;
    int  cache_l0     = 0;
    std::string profile;        // per layer, experts ranked by use (tools/g4_profile.py); empty = plain order
    int  dcheck_layer = -1;     // a cached layer whose merged rows are checked against all-GPU and all-CPU twins
    // stage e: adaptation. Decayed usage counts per layer; after a prompt read every layer is re-set to its top-S, after
    // each small ubatch at most adapt_budget swaps per layer (a newcomer must score above its victim)
    bool  adapt           = false;
    float adapt_half_life = 64.0f;   // tokens (Gate 0, Qwen: 64 keeps the hits of 32 with half the swaps)
    int   adapt_budget    = 2;
    bool  adapt_refill    = true;
    // speed (Gemma's speed work, ported): swaps on their own CUDA stream from a page-locked expert arena, the newcomer
    // committed when its copy has landed (the victim goes to the CPU at once); the misses run in the pool while the GPU
    // computes the hits (start / join ops around the slot path)
    bool  async_swaps     = false;
    double ram_budget_gib = -1;    // the RAM tier: GiB of experts locked in RAM (-1 auto, see src/lite/ram_tier.hpp)
    bool  overlap         = false;
    // stage g: prompt reads in ubatches of prefill_ubatch tokens (0 = n_ubatch, -1 = auto: the largest of 32k / 16k / 8k
    // / 4k whose compute buffers fit the VRAM the read can borrow); with lend the slot arena's VRAM goes to the read and
    // the arena comes back empty after it, refilled in the background from the read's routing counts
    int   prefill_ubatch  = 0;
    bool  lend            = false;
    // stage f: MTP (llama.cpp draft-mtp). The model's next-token head, an MTP-only GGUF (tools/qwen35_mtp_extract.py),
    // drafts; it keeps its own KV cache, filled after every target batch from the target's post-norm hidden rows. The
    // delta-net layers keep n_rs_seq + 1 state snapshots so a verify window can roll back (llama.cpp: n_rs_seq = draft max)
    std::string mtp;
    bool mtp_share_head = false;     // the target's token_embd / output instead of the drafter file's own
    int  n_rs_seq       = 0;
    // the KV pager: the attention layers' K/V start with kv_initial cells on the card (0 = the whole context at load, as
    // llama.cpp) and grow in kv_step steps as the conversation needs them; the slot arena gives way (ensure_free)
    int  kv_initial     = 0;
    int  kv_step        = 16384;
    // several GPUs (llama.cpp --split-mode layer --tensor-split): one share per visible GPU, layers placed as llama.cpp does;
    // empty or one entry = the first GPU only. Dense models (qwen35) so far.
    std::vector<float> tensor_split;
    // --split-mode tensor (llama.cpp's meta backend): every layer split across the cards, each card a slice of each
    // weight, K/V and recurrent state (tensor_split = the shares; empty = equal over every visible GPU). Dense models.
    bool split_tensor = false;
    // server slots (llama-server --parallel): sequences sharing the attention layers' KV (one unified cache, each cell
    // tagged with its sequence) and one recurrent cell each (llama_memory_recurrent, mem_size = n_seq). Not LFM2 yet.
    int n_seq = 1;
};

struct Stats {
    uint64_t accesses = 0, hits = 0;                       // routed (token, expert) pairs of the cached layers, small ubatches
    uint64_t dcheck_rows = 0, dcheck_gpu_rows = 0, dcheck_bad = 0;
    uint64_t swaps = 0, refill_swaps = 0, swap_bytes = 0;
    double   swap_ms = 0, refill_ms = 0;
    uint64_t drafts = 0, catchup_tokens = 0;              // stage f: drafter steps, tokens the drafter re-read
    double   draft_ms = 0, catchup_ms = 0;
    uint64_t commits = 0, commit_batches = 0, pending_peak = 0;   // async swaps
    uint64_t ov_layers = 0, ov_missfree = 0;                      // overlap: layer-steps, and those with no miss
    double   join_wait_ms = 0, pool_ms = 0;
    double   prefill_ms = 0, lend_ms = 0, restore_ms = 0, refill_land_ms = -1;   // stage g
    double   ring_wait_ms = 0, stage_ms[4] = { 0, 0, 0, 0 }, stage_wait_ms = 0;   // the expert rings' waits; a pipelined read's stage busy time
    uint64_t lent_bytes = 0, read_buf_bytes = 0;     // the arena lent; the largest compute buffer a read ubatch took
    int      last_chunk = 0, arena_shrinks = 0;
    uint64_t graphs_built = 0, graphs_reused = 0;   // v0.2.3: ubatch graphs built, and steps that reused the last one
    // the per-token timeline (small ubatches: decode steps and verify windows), host wall clock, ms
    uint64_t t_ubatches = 0;
    double   t_build = 0, t_inputs = 0, t_compute = 0, t_readout = 0;
    double   t_ro_adapt = 0, t_ro_logits = 0; double t_ad_score = 0;   // adapt: readback + scoring   // the readout's parts: adapt (routing, scores, swaps), the logits   // graph build or reuse + KV slots; the inputs; the compute; logits + adapt
    std::vector<std::pair<uint32_t, uint32_t>> step_hits;   // per small ubatch after load: routed pairs, hits (cached layers)
};

struct HParams {
    bool lfm2 = false;
    // flint (Recursive's Flint, llama.cpp src/models/flint.cpp, ggml-org/llama.cpp#30928): every layer attention, sliding
    // (n_swa cells, the current one included) or full; RoPE only where has_rope (the full layers have no positions); learned
    // sinks; QK RMSNorm; leading dense SwiGLU layers; sigmoid routing renormalized; one ungated shared expert
    bool flint = false;
    int n_swa = 0;
    std::vector<bool> is_swa, has_rope;
    bool pos1() const { return lfm2 || flint; }   // one position per token (no M-RoPE sections)
    bool dense = false;                 // qwen35 (Qwen3.5 / 3.8 dense): every layer llama.cpp's build_ffn, no experts
    int n_layer_all = 0;                // block_count, the MTP block included (the layer split counts it, as llama.cpp)
    int n_dense = 0, n_ff = 0;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, n_vocab = 0, n_ctx_train = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    int ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0, ssm_n_group = 0, ssm_dt_rank = 0, full_attn_interval = 4;
    int rope_sections[4] = { 0, 0, 0, 0 };
    float rope_base = 0, eps = 0;
    std::vector<bool> is_recr;          // DeltaNet or LFM shortconv layer (else full attention)
    int64_t n_embd_r() const { return (int64_t) (ssm_d_conv - 1) * (lfm2 ? n_embd : ssm_d_inner + 2 * ssm_n_group * ssm_d_state); }
    int64_t n_embd_s() const { return (int64_t) ssm_d_state * ssm_d_inner; }
    // qwen4exp (Qwen3.8 Flash-Next, llama.cpp src/models/qwen4exp.cpp): hc residual streams instead of the layer norms
    // (low rank hc_lr), an indexer (idx_n_head heads of idx_dim) picking idx_top_k cells for the attention layers whose
    // compress ratio > 0 (QSA), and the n-gram per-layer embedding (PLE) on one delta-net layer
    bool q4x = false;
    int hc = 0, hc_lr = 0;
    int idx_n_head = 0, idx_dim = 0, idx_top_k = 0;
    std::vector<int> compress;          // per layer: the QSA block size (0 = dense attention)
    int ple_layer = -1, ple_ngram = 0, ple_per_gram = 0, ple_kernel = 0, ple_n_heads = 0, ple_head_dim = 0;
    int32_t ple_eos = 0, ple_img = 0;
    std::vector<uint64_t> ple_mult;     // one per n-gram slot
    std::vector<uint32_t> ple_off, ple_vsz;   // per hash head: row offset, modulus
    int64_t n_embd_p() const { return (int64_t) (ple_kernel - 1) * ple_ngram * hc * n_embd; }   // the PLE conv history
};

class Engine {
public:
    Engine();
    ~Engine();
    std::string load(const Options & o);
    // the loaded adapters (lora then loras, in order) and their scales; set_lora_scale rebuilds the graphs (the caller drops
    // prompt caches computed with the old scales)
    int lora_count() const { return lora_ ? (int) lora_->count() : 0; }
    float lora_scale(int i) const { return lora_->scale(i); }
    const std::string & lora_path(int i) const { return lora_->path(i); }
    void set_lora_scale(int i, float s) { lora_->set_scale(i, s); graph_invalidate(); }
    // one llama_decode call: tokens at the next positions; the last token's logits (and h_last: its post-norm hidden row)
    std::string decode(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h_last = nullptr, int seq = 0);
    void reset();               // forget everything (positions, KV cells, recurrent states), every sequence
    void reset_seq(int seq);    // one sequence only (a slot's new conversation); with one sequence = reset()
    // several sequences in one step (the server's slots): each step's tokens at that sequence's next positions; runs of
    // consecutive sequences with the same token count share a ubatch (split_equal), the others go one by one. logits /
    // h: every row of every step, in the order of the steps
    struct SeqStep { int seq; std::vector<int32_t> toks; };
    std::string decode_multi(const std::vector<SeqStep> & steps, std::vector<float> * logits, std::vector<float> * h = nullptr,
                             bool lazy = false);
    // v0.2.4-exact1 (Strata-Lite dfb7bd6e): decode_multi(..., lazy = true) on a step that fits one ubatch leaves its logits
    // on the cards (lazy_logits() true, logits not filled); logits_row(r) then copies row r (the steps' order, as logits
    // would hold it) into a pinned buffer and returns it. Valid until the next target ubatch. Same floats, fewer bytes moved.
    bool lazy_logits() const { return lazy_t_ != nullptr; }
    const float * logits_row(int r);
    // stage f. A verify window in one ubatch: logits [n][n_vocab] and post-norm hidden rows [n][n_embd] of every row
    std::string decode_all(const std::vector<int32_t> & toks, std::vector<float> * logits, std::vector<float> * h, int seq = 0);
    // one draft step: token at pos with hidden row h (the target's, then the drafter's own) -> argmax + the next h
    std::string draft(int32_t tok, const float * h, int pos, int32_t * out, std::vector<float> * h_out, int seq = 0);
    float draft_p() const;   // the last draft's probability under the drafter (min-p gating), computed when asked
    void mtp_trim(int pos, int seq = 0);          // drop the drafter's cells from pos on (its draft steps)
    std::string rollback(int n_keep, int seq = 0);   // keep positions < n_keep: KV cells, recurrent snapshot, drafter
    const std::vector<float> & mtp_h(int seq = 0) const { return mtp_h_v_[seq]; }   // h of the last kept position (the next draft's)
    bool has_mtp() const { return mtp_.loaded; }
    const HParams & hp() const { return hp_; }
    struct Ring;   // a read's expert ring on one card (q35_engine.cpp)
    Ring * ring_at(int d) { return d < (int) ring_.size() ? ring_[d].get() : nullptr; }
    // the drafter's hidden row: n_embd, or qwen4exp's whole 4-stream residual (the MTP block reads all of it)
    int64_t h_width() const { return hp_.q4x ? (int64_t) hp_.hc * hp_.n_embd : hp_.n_embd; }
    size_t ram_locked() const { return ram_locked_; }     // the RAM tier: expert bytes locked in RAM
    size_t ram_experts() const { return ram_experts_; }   // and of all CPU-side experts
    bool async_swaps() const { return o_.async_swaps; }    // false in low-RAM mode
    const Stats & stats() const { return stats_; }
    int slots() const { return slots_now_; }
    int n_past(int seq = 0) const { return n_past_v_[seq]; }
    int n_seq() const { return (int) n_past_v_.size(); }
    int n_ctx() const { return (int) kv_.size; }
    int kv_cap() const { return (int) kv_cap_; }   // KV cells on the card now (<= n_ctx)
    // an image chunk (mtmd's embeddings, n rows): pos4 = 4 x n M-RoPE positions (t, y, x, z sections, mtmd's
    // decoder positions), n_pos = how far the position advances (mtmd_input_chunk_get_n_pos). The drafter skips it, as
    // draft-mtp's process() skips embedding batches.
    std::string decode_embd(const float * embd, int n, const int32_t * pos4, int n_pos, std::vector<float> * logits, int seq = 0);
    // the prompt cache. A recurrent state cannot rewind to any position: the engine can return to pos when pos is the
    // current position, within the last ubatch's snapshots, a checkpoint (the server takes up to four per prompt), or 0.
    // resume_point(pos) = the largest of those <= pos; rollback_to(p) goes there.
    int resume_point(int pos, int seq = 0) const;
    std::string rollback_to(int pos, int seq = 0);
    void checkpoint(int seq = 0);
    // VRAM for something else (the projector, its encode): fewer slots; and back up to S slots when it is free again
    // card: the engine's card index (gpus_ order); -1 = the first card with an expert arena
    int ensure_free(size_t bytes, bool keep_profile = false, int card = -1);
    int grow_to(int S, size_t headroom, int card = -1);
    int n_cards() const { return (int) gpus_.size(); }
    ggml_backend_dev_t card_dev(int d) const { return ggml_backend_get_device(gpus_[d]); }
    int slots_on(int d) const { return d < (int) slots_c_.size() ? slots_c_[d] : 0; }
    // the RAM prompt cache (llama-server's --cache-ram): a sequence's whole state copied to host memory (its KV cells and
    // their K/V / indexer keys, the recurrent state it would read next, its checkpoints, the drafter's cells), and back.
    // unpark makes the sequence exactly what it was at park: it resumes at its position or any of its checkpoints.
    struct Parked;
    std::shared_ptr<Parked> park(int seq);
    std::string unpark(int seq, const Parked & p);
    static size_t parked_bytes(const Parked & p);
    static int parked_pos(const Parked & p);

private:
    struct Layer {
        ggml_tensor *attn_norm, *attn_post_norm;
        ggml_tensor *wq, *wk, *wv, *wo, *q_norm, *k_norm;                                            // attention layers
        ggml_tensor *wqkv, *wqkv_gate, *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_beta, *ssm_alpha, *ssm_norm, *ssm_out;   // delta-net
        ggml_tensor *conv_in, *conv_out, *conv_kernel; // LFM2 short convolution
        ggml_tensor *dense_gate, *dense_up, *dense_down, *expert_bias;
        ggml_tensor *attn_sinks;                                                                    // flint: [n_head]
        ggml_tensor *gate_inp, *gate_exps, *up_exps, *down_exps;                                    // routed experts
        ggml_tensor *gate_inp_shexp, *gate_shexp, *up_shexp, *down_shexp;                          // shared expert
        ggml_tensor *hc_an, *hc_ad, *hc_au, *hc_ai, *hc_fn, *hc_fd, *hc_fu, *hc_fi;                // qwen4exp: the two hc mixers
        ggml_tensor *idx_q, *idx_k, *idx_qn, *idx_kn;                                               // qwen4exp: the indexer
        ggml_tensor *ple_key, *ple_value, *ple_nk, *ple_nq, *ple_nc, *ple_conv;                     // qwen4exp: PLE
        // the optional second-level scales llama.cpp loads in its generic pass (llama-model.cpp, "<weight>.scale": NVFP4's
        // weight_scale_2), multiplied onto the matmul's output (build_lora_mm / build_ffn / build_moe_ffn); nullptr = none
        ggml_tensor *wq_s, *wk_s, *wv_s, *wo_s, *wqkv_s, *wqkv_gate_s, *ssm_beta_s, *ssm_alpha_s, *ssm_out_s;   // {1}
        ggml_tensor *dense_gate_s, *dense_up_s, *dense_down_s, *gate_shexp_s, *up_shexp_s, *down_shexp_s;      // {1}
        ggml_tensor *gate_exps_s, *up_exps_s, *down_exps_s;                                         // {n_expert}, beside the experts
    };
public:
    struct PoolLayer;                    // one layer's experts as the pool reads them (q35_engine.cpp)
    struct CacheLayer;                   // one layer's GPU expert slots (stage d)
    struct Driver;                       // overlap: the thread that runs the pool for a start op
private:
    struct Cache {                       // llama_kv_cache, no sliding window (the attention layers); one stream per sequence
        // (llama-server's --parallel N: kv_unified false): stream s owns cells [s * ssize, (s + 1) * ssize)
        uint32_t size = 0, ssize = 0, n_kv = 0;
        std::vector<uint32_t> heads;     // per stream: where the next search starts
        std::vector<int32_t> pos;
        std::vector<int8_t> seq;         // the cell's sequence (a server slot); -1 = empty
        std::vector<int32_t> tok;        // the cell's token (qwen4exp's n-gram embedding reads the predecessors; -1 = none)
        std::vector<int32_t> ex, ey;     // an image cell's 2-D position (M-RoPE x, y sections; llama_kv_cell_ext), 0 for text
        std::vector<int64_t> idxs;
        std::vector<ggml_tensor *> k, v;
        void init(uint32_t cells_per_stream, int n_streams);
        int  stream_of(int s) const { return heads.size() > 1 ? s : 0; }
        uint32_t used(int st) const;
        uint32_t used_max_p1(int st) const;
        bool find_slot(uint32_t n_tokens, int st);   // appends the cells to idxs
        void apply(const std::vector<int32_t> & positions, const std::vector<int8_t> & seqs, const int32_t * toks = nullptr, int32_t tok_embd = -1,
                   const int32_t * pos_y = nullptr, const int32_t * pos_x = nullptr);
        // llama_kv_cell_ext::is_2d_gt: cell c lies after (x, y) in raster order
        bool after_2d(size_t c, int32_t x, int32_t y) const { return ey[c] > y || (ey[c] == y && ex[c] > x); }
        uint32_t calc_n_kv(int st0, int nst) const;  // llama_kv_cache::get_n_kv over the ubatch's streams
        void seq_rm(int s, int32_t p0);  // llama_kv_cache::seq_rm(s, p0, -1); s < 0: every sequence
        void clear() { std::fill(pos.begin(), pos.end(), -1); std::fill(seq.begin(), seq.end(), (int8_t) -1); std::fill(tok.begin(), tok.end(), -1); std::fill(heads.begin(), heads.end(), 0u); n_kv = 0; }
    };

    // n_out: how many trailing rows output logits (0, 1 or n)
    std::string ubatch(const int32_t * toks, int n, int pos0, int n_out, std::vector<float> * logits, int reserve_outputs = 0);
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, int il);

    Options o_;
    std::unique_ptr<lite::LfmLora> lora_;
    HParams hp_;
    gguf_context * gguf_ = nullptr;
    ggml_context * meta_ = nullptr, * ctx_w_ = nullptr, * ctx_gpu_ = nullptr;
    void * map_ = nullptr;
    size_t map_size_ = 0;
    // a split GGUF: shard k's metadata and its data section's offset in the mapping (shard 0 = gguf_ / meta_)
    struct Shard { gguf_context * g = nullptr; ggml_context * meta = nullptr; size_t data = 0; };
    std::vector<Shard> shards_;
    ggml_backend_t gpu_ = nullptr, cpu_ = nullptr;
    ggml_backend_buffer_t buf_gpu_ = nullptr, buf_map_ = nullptr;
    // several GPUs: gpus_[0] = gpu_; each layer's card (weights, K/V, recurrent states) and the output's
    std::vector<ggml_backend_t> gpus_;
    std::vector<int> dev_layer_;
    int dev_out_ = 0;
    std::vector<ggml_context *> ctx_gpu_x_, ctx_kv_x_;           // weights on cards 1..; the recurrent states, per card
    std::vector<ggml_backend_buffer_t> buf_gpu_x_, buf_kv_x_;
    void sync_all();                                             // wait for every card's queue (CUDA device 0 current after)
    bool tp_ = false;                                            // --split-mode tensor: gpu_ is the meta backend over the cards
    std::unique_ptr<TsplitInfo> tp_info_;
    ggml_backend_sched_t sched_ = nullptr;
    ggml_tensor *tok_embd_ = nullptr, *output_ = nullptr, *output_norm_ = nullptr;
    ggml_tensor *output_s_ = nullptr;        // output.scale (llama.cpp loads it for an NVFP4 output only)
    // the per-expert scales read once at load (gate, up, down; empty = none): the CPU pool's jobs take them
    std::vector<std::array<std::vector<float>, 3>> exps_sh_;
    ggml_tensor *hc_head_norm_ = nullptr, *hc_head_down_ = nullptr, *hc_head_up_ = nullptr, *ple_tbl_ = nullptr;   // qwen4exp
    std::vector<Layer> L_;
    Cache kv_;
    std::vector<ggml_tensor *> r_l_, s_l_;   // recurrent states per delta-net layer: conv [n_embd_r, 1], ssm [n_embd_s, 1]
    std::vector<ggml_tensor *> p_l_;         // qwen4exp: the PLE layer's conv history [n_embd_p, rows as r_l_]
    std::vector<ggml_tensor *> idxk_;        // qwen4exp: the indexer's raw keys per QSA layer [idx_dim, kv cells], the K/V's cells
    // per sequence: position, the snapshot the next ubatch reads (0 = newest), the last ubatch's text tokens (its snapshots
    // are the only valid ones), a new recurrent cell (cleared before its first use: rs_z)
    std::vector<int> n_past_v_, rs_src_v_, last_ub_v_;
    std::vector<uint8_t> fresh_v_;
    int cur_seq_ = 0;                                   // the sequence decode / decode_all / decode_embd work on
    std::vector<int32_t> ub_pos_; std::vector<int8_t> ub_seq_; int ub_ns_ = 0;   // decode_multi's ubatch (seq-major)
    std::vector<uint8_t> graph_buf_;
    std::vector<float> h_rows_;
    bool want_h_ = false;
    bool lazy_req_ = false; ggml_tensor * lazy_t_ = nullptr; std::vector<int> lazy_map_;   // logits_row: step row -> ubatch row
    float * lazy_pin_ = nullptr; size_t lazy_pin_rows_ = 0;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_;
    std::vector<std::unique_ptr<PoolLayer>> pool_layers_;
    std::unique_ptr<Driver> driver_;
    // async swaps: the RAM-resident experts in a pinned arena, the copy stream, and the batches of copies in flight
    void * arena_ = nullptr;
    size_t arena_size_ = 0;
    bool arena_placed_ = false;   // numa-always: the arena is placed per socket (lite::ram_tier_place; freed by ram_tier_free)
    size_t ram_locked_ = 0, ram_experts_ = 0;   // the RAM tier: bytes locked / bytes of CPU-side experts
    ggml_backend_buffer_t buf_arena_ = nullptr;
    void * copy_stream_ = nullptr;           // async swaps on (card 0's stream); copy_streams_: each card's
    std::vector<void *> copy_streams_;
    struct PendingSwap { int il, e, slot; };
    struct PendingBatch { bool closed = false; uint32_t cards = 0; std::vector<void *> evs; std::vector<PendingSwap> sw; };   // one event per card copied to
    void sync_copies();
    void set_card(int d) const;              // make card d's CUDA device current (set_card(0) restores the engine's)
    std::vector<PendingBatch> pending_;
    void commit_landed();
    // stage g
    bool reading_ = false, refill_watch_ = false;
    std::chrono::steady_clock::time_point read_end_;
    std::string fatal_;
    void begin_read(int ubatch);
    void end_read();
    int pick_chunk(int n);
    std::string reserve_graphs();
    void new_sched();
    // stage d: the slot arena. Model-neutral: an expert is a list of tensors (here gate, up, down), each layer gets one
    // slot tensor per expert tensor with S + 1 slots (slot S all zeros: where the misses point)
    // several cards (layer split): each card's cached layers in that card's arena, sized to that card's VRAM
    std::vector<std::unique_ptr<CacheLayer>> cache_;
    std::vector<ggml_context *> ctx_cache_c_;          // per card (gpus_ index)
    std::vector<ggml_backend_buffer_t> buf_cache_c_;
    std::vector<ggml_context *> ctx_sc_c_;             // per card: the cached layers' expert scales (CacheLayer::sc)
    std::vector<ggml_backend_buffer_t> buf_sc_c_;
    std::vector<std::vector<int>> rank_;
    int slots_now_ = 0;                                 // the first cached card's slots (slots())
    std::vector<int> slots_c_, slots_load_c_;           // per card: now, and at load (a new conversation grows back to it)
    std::vector<size_t> compute_bytes_c_;               // per card: the reserved graphs' GPU buffers at load
    std::vector<int> cache_cards_;                      // the cards holding cached layers, in order
    bool arena_live() const { for (auto * b : buf_cache_c_) if (b) return true; return false; }
    std::string arena_build(int S, bool fill_profile, int card = -1);   // -1: every cached card, S slots each
    void adapt_after(int n, const std::vector<ggml_tensor *> & argsorts);
    void swap_layer(int il, int budget);
    bool refill_pending_ = false;
    struct DCheck { ggml_tensor * merged = nullptr, * twin_gpu = nullptr, * twin_cpu = nullptr, * argsort = nullptr; CacheLayer * C = nullptr; };
    DCheck dcheck_;
    Stats stats_;
    // graph reuse (v0.2.3, llama_context::process_ubatch's can_reuse): the last computed graph stays allocated on its
    // scheduler, and a step of the same shape sets the inputs and computes it again instead of building a new one.
    // The shape: tokens, padded KV cells, output rows, the embedding branch, a fresh recurrent state. Anything that moves a
    // tensor the graph points at (a new scheduler, the slot arena, the KV pager, a reserve) drops it (graph_invalidate).
    struct GraphKey {
        int n = -1, n_kv = 0, n_outputs = 0; bool embd = false; int rs_zero = -1, ns = 1, head = 0, l0 = 0, l1 = 0; bool wide = false; int swa_lo = 0;
        bool operator==(const GraphKey & o) const { return n == o.n && n_kv == o.n_kv && n_outputs == o.n_outputs && embd == o.embd &&
                                                           rs_zero == o.rs_zero && ns == o.ns && head == o.head && l0 == o.l0 && l1 == o.l1 &&
                                                           wide == o.wide && swa_lo == o.swa_lo; }
    };
    struct Graph {
        ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; bool valid = false; GraphKey key;
        ggml_tensor *inp_tokens = nullptr, *inp_embd = nullptr, *inp_pos = nullptr, *k_idxs = nullptr, *v_idxs = nullptr, *kq_mask = nullptr, *kq_mask_swa = nullptr,
                    *s_copy = nullptr, *k_rot = nullptr, *v_rot = nullptr, *inp_out_ids = nullptr, *inp_h = nullptr, *t_logits = nullptr, *t_h = nullptr;
        // qwen4exp: the PLE rows, the indexer's cells (k_idxs as the attention's), and the QSA block layout (llama.cpp's
        // llm_graph_input_qsa: one set for every layer of a ratio)
        ggml_tensor *ple_rows = nullptr, *qsa_k_idxs = nullptr, *cell_blk = nullptr, *blk_cells = nullptr, *blk_pos = nullptr, *qsa_bias = nullptr;
        int qsa_r = 0;
        uint32_t swa_lo = 0;                    // flint: the first cell the sliding layers read
        ggml_tensor *st_in = nullptr, *st_out = nullptr;   // a pipelined read's stage: the residual in and out
        std::vector<ggml_tensor *> adapt_ids;   // the cached layers' routing (adapt)
        DCheck dcheck;                          // the dcheck layer's twins
        uint64_t acc_inc = 0, hit_inc = 0;      // routed pairs the shape alone decides, counted every step
        std::shared_ptr<struct RingUse> ring;   // a read through the expert rings: which nodes read which ring slot
    } g_, mg_;
    // a read's expert ring on one card: the RAM-resident layers' expert tensors copied in ahead of use on a copy thread and
    // stream, R slots reused in turn, so the copies overlap the compute (Strata's async expert ring, per tensor)
    std::vector<std::unique_ptr<Ring>> ring_;   // per card, live during a read
    void * ids_host_ = nullptr; size_t ids_host_bytes_ = 0;   // adapt: the routing readback, pinned
    void ring_begin(int n_chunks, int ub);
    void ring_end();
    bool ring_layer(int il) const;
    size_t ring_bytes(int d) const;                                  // the target's (sched_) and the drafter's (mtp_.sched)
    std::vector<uint8_t> mtp_graph_buf_;
    static void graph_drop(Graph & g) { if (g.ctx) ggml_free(g.ctx); g.ctx = nullptr; g.gf = nullptr; g.valid = false; }
    void graph_invalidate() { g_.valid = false; mg_.valid = false; }
    bool cpu_only_ = false;
    bool host_states() const { return tp_ || cpu_only_; }   // recurrent checkpoints in host memory (no device pointer to copy)
    // a pipelined read over several cards (qwen4exp, Strata's chunk pipeline): each card a stage running its layers
    // [st_l0_, st_l1_) with its own scheduler and graph, on chunk c while the next card runs chunk c - 1; the residual passes
    // through host memory. A chunk's first stage records its cells and recurrent inputs, the later stages replay them
    struct UbRec { std::vector<int64_t> idxs; uint32_t n_kv = 0; std::vector<int32_t> positions; std::vector<int8_t> seqs;
                   int32_t rs_zero = -1; std::vector<int32_t> s_copy; };
    int st_l0_ = 0, st_l1_ = -1, st_card_ = 0;
    const float * st_in_ = nullptr;
    UbRec * st_rec_ = nullptr;
    const UbRec * st_replay_ = nullptr;
    bool st_async_ = false;
    ggml_cgraph * st_gf_ = nullptr;   // an async stage's graph, computed on its card's own thread
    std::vector<ggml_backend_sched_t> sched_st_;
    std::vector<ggml_backend_t> cpu_st_;
    std::vector<std::vector<uint8_t>> graph_buf_st_;   // each stage's graph metadata (a kept graph lives in its buffer)   // each stage's own CPU backend (its work buffer is per backend)
    std::vector<Graph> g_st_;
    bool read_pipeline_ok(int n) const;
    std::string read_pipelined(const std::vector<int32_t> & toks, int ub, int pos0, std::vector<float> * logits);
    // qwen4exp host inputs (llama.cpp llm_graph_input_ple::set_input, llama_memory_hybrid_idx::set_input_qsa)
    void set_ple_rows(ggml_tensor * rows, const int32_t * toks, int n, int ns, int nt, int head, const std::vector<int32_t> & positions);
    std::string set_qsa(const Graph & G, int ns, int nt, int head, const std::vector<int32_t> & positions, uint32_t n_kv, const int32_t * pos4, int n);
    // stage f: the drafter (graph_mtp in llama.cpp's qwen35moe.cpp), its own weights, KV cells and scheduler
    struct Mtp {
        bool loaded = false;
        gguf_context * gguf = nullptr;
        ggml_context * meta = nullptr, * ctx_w = nullptr, * ctx_gpu = nullptr, * ctx_kv = nullptr;
        void * map = nullptr;
        size_t map_size = 0;
        ggml_backend_buffer_t buf_gpu = nullptr, buf_map = nullptr, buf_kv = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ggml_tensor *tok_embd = nullptr, *output = nullptr, *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
        ggml_tensor *hc_head_norm = nullptr, *hc_head_down = nullptr, *hc_head_up = nullptr;   // qwen4exp: the block's final mixer
        ggml_tensor *eh_proj_s = nullptr, *output_s = nullptr;   // nextn.eh_proj.scale; the head's (its own output.scale, or the target's)
        bool q4x = false;                // a qwen4exp block (graph_mtp of the qwen4exp/mtp branch): hyper-connections, 4-stream rows
        Layer l{};
        Cache kv;
        bool dense = false;              // a qwen35 drafter: the block's FFN is build_ffn (the dense series' graph_mtp)
        std::vector<int32_t> d2t;        // HauhauCS FastMTP: draft id -> target id (the head drafts over a reduced vocabulary)
        ggml_backend_t gpu = nullptr;    // the card it runs on (with several cards: one of them, as llama.cpp pins a drafter)
        bool own_gpu = false;
        std::vector<float> last_lg;      // the last draft step's logits: draft_p() reads them only when the min-p gate is on
        mutable float last_p = -1.0f;    // its probability under the drafter (greedy pick); < 0: not computed yet
    } mtp_;
    std::vector<std::vector<float>> mtp_h_v_, win_h_v_;   // per sequence: the drafter's pending row, the last verify window's rows
    std::vector<int> win_pos0_v_;
    const float * embd_in_ = nullptr;
    const int32_t * pos4_in_ = nullptr;
    static constexpr int kCkpts = 4;  // checkpoints per sequence: every delta-net layer's conv + ssm state, on the GPU (the oldest goes)
    std::vector<void *> ckpt_;        // [seq * kCkpts + c]
    size_t ckpt_bytes_ = 0;
    std::vector<int> ckpt_pos_;
    std::vector<uint64_t> ckpt_age_; uint64_t ckpt_clock_ = 0;
    std::vector<std::vector<float>> ckpt_h_;
    std::vector<std::vector<uint8_t>> ckpt_host_;   // --split-mode tensor: the checkpoints in RAM (a meta tensor has no device pointer)
    bool ckpt_have(int c) const { return tp_ ? !ckpt_host_[c].empty() : ckpt_[c] != nullptr; }
    size_t slot_bytes(int card) const;
    // the KV pager: each attention layer's K/V in its own buffer of kv_cap_ cells (a multiple of 256, <= kv_.size). The
    // reserve graphs see full-size tensors over the same memory (kfull_ / vfull_, metadata only, never computed), so the
    // allocator's plan is the one a whole-context KV gives
    struct KvLayer { ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr; };
    std::vector<KvLayer> kvl_;
    ggml_context * ctx_kvfull_ = nullptr;
    std::vector<ggml_tensor *> kfull_, vfull_;
    uint32_t kv_cap_ = 0;
    size_t kv_bytes(uint32_t cells) const;
    std::string kv_alloc_layer(int il, uint32_t cap);
    std::string kv_grow(uint32_t cells);
    std::string load_mtp();
    std::string mtp_ubatch(const int32_t * toks, const float * h, int n, int pos0, bool out, int32_t * amax, std::vector<float> * h_out, int seq = 0);
    std::string mtp_catchup(const int32_t * toks, int n, int pos0, const float * h_rows);
};

} // namespace q35
