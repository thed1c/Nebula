// include/strata/net/expert_rpc.hpp - a remote expert tier: another host computes the routed experts of a range of layers
// on its CPUs (its RAM holds those layers' experts) and sends the rows back over net/transport.hpp's Channel.
//
// WHERE IT CUTS.  The engine's CPU expert path (core/expert_source.cpp expert_pool_dispatch_multi) classifies each routed
// (token, expert) entry of a verify window: VRAM hit, PCIe, a helper GPU, or the CPU pool.  `ExpertRpcClient::begin` takes the
// CPU entries of the layers the remote serves (kind -1 -> 2, as a helper GPU's), sends the window's activations and those
// entries; `finish` receives one row per entry and writes it where the CPU pool would have (the GPU then combines the rows
// exactly as before).  Per token per layer: 10 KB of activation out, 10 KB per remote entry back (fp32, H = 2560).
//
// EXACT.  The server computes each row with the same CPU kernels the engine's pool runs (the same activation quantization,
// the same multi-token job per expert, the same row-split pool call), so a row is bit for bit the row the engine's own pool
// computes on a CPU with the same instruction set (tests/net/expert_rpc_test.cpp checks it across two processes).
//
// On-helper reduce (`mode` kReduce): the server sums its rows with the router weights and sends one row per token - 10x
// fewer bytes back, but a different summation order than the GPU's combine, so not bit-exact; opt-in.
#pragma once

#include "strata/net/rdma.hpp"
#include "strata/net/transport.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::kernels::cpu {
class ExpertPool;
struct ExpertLayout;
}

namespace strata::net {

enum ExpertMode : uint32_t { kRows = 0, kReduce = 1 };

/// One routed entry the remote computes: token `t` of the window through expert `e` (weight `w`: kReduce only).
struct ExpertEntry {
    int32_t t;
    int32_t e;
    float w;
};

struct ExpertReqHeader {
    uint32_t layer, n_tok, h, n_ent, mode, seq;
};
struct ExpertRespHeader {
    uint32_t seq, n_rows, h, status;   // status 0 = ok (else an error text follows instead of rows)
};

/// The rows for `ents` of one layer, computed by `pool` exactly as the engine's CPU expert path computes them: activations
/// quantized per token as the layout wants (native / Q2 native / Q8_1), one multi-token job per distinct expert in entry
/// order, the row-split pool call.  `rows` holds n_ent x H floats.  `blob(layer, e)` gives an expert's bytes.
bool compute_expert_rows(strata::kernels::cpu::ExpertPool& pool, const strata::kernels::cpu::ExpertLayout& layout,
                         const std::function<const uint8_t*(int64_t, int64_t)>& blob, int64_t layer, const float* x,
                         int n_tok, const ExpertEntry* ents, int n_ent, float* rows, std::string& error);

/// The client in the engine: one remote, one layer range.
class ExpertRpcClient {
public:
    /// "host:port@A-B" (layers A..B inclusive; no @: every layer)
    static bool parse(const std::string& spec, Endpoint& ep, int64_t& lo, int64_t& hi, std::string& error);
    /// `rdma`: upgrade the connection to RDMA (net/rdma.hpp) after the TCP handshake; null = TCP.
    bool connect(const Endpoint& ep, int64_t lo, int64_t hi, const Security& sec, int64_t h, std::string& error,
                 const RdmaOptions* rdma = nullptr);
    bool serves(int64_t layer) const { return ch_ != nullptr && !broken_ && layer >= lo_ && layer <= hi_; }
    /// A window's routed entries (n_tok x k, `kind` as the dispatch classified them: -1 = the CPU pool's): the CPU
    /// entries of a served layer are sent and marked 2.  Nothing to send: nothing happens (finish then does nothing).
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind, std::string& error);
    /// The rows of the request begin sent, into `out` (row i = entry i, H floats each, as the CPU pool writes them).
    bool finish(float* out, std::string& error);
    bool pending() const { return pending_; }
    bool broken() const { return broken_; }
    /// After a failure the client stops serving (the layers go back to the local CPU pool) and says why once.
    void mark_broken(const std::string& why);
    std::string describe() const;
    int64_t requests = 0, entries = 0;
    uint64_t bytes_out = 0, bytes_in = 0;
    double wait_ms = 0;   ///< time finish waited for the rows

private:
    std::unique_ptr<Channel> ch_;
    Endpoint ep_;
    int64_t lo_ = 0, hi_ = -1, h_ = 0;
    bool broken_ = false, pending_ = false;
    uint32_t seq_ = 0;
    std::vector<int64_t> idx_;           ///< the entries sent (their index in the window)
    std::vector<ExpertEntry> ents_;
    std::vector<uint8_t> resp_;
};

/// The server: answers ExpertReq frames on one connection with `compute` until the peer closes.  Returns false on a
/// transport error (error says why).
using ExpertCompute = std::function<bool(int64_t layer, const float* x, int n_tok, const ExpertEntry* ents, int n_ent,
                                         uint32_t mode, float* out, std::string& error)>;
bool serve_expert_connection(Channel& ch, int64_t h, int64_t lo, int64_t hi, const ExpertCompute& compute, std::string& error,
                             int64_t* served = nullptr);

}  // namespace strata::net
