// include/strata/net/stage_rpc.hpp - the layer pipeline across hosts: a verify window's hand-off between two stages of a
// layer split, over net/transport.hpp's Channel.
//
// WHAT CROSSES.  In one process a layer split hands a window from stage i to stage i+1 through mapped pinned host memory
// (program/generate.cpp, the hand-off buffers of SplitDrive / GpuStage): per row `handoff_floats = hc*n_embd + n_embd + hc`
// fp32 (core/verify.hpp) - the hyper-connection streams R, the pending write bo and inject; Flash-Next: 12,804 floats =
// 51.2 KB a token, an 8-row window 410 KB.  Across hosts the same bytes go in one kStageHandoff frame (header + rows, fp32
// as they are: bit for bit), and the last stage answers with a kStageReturn frame (its sampled tokens / logits rows).
//
// STATUS (2026-10-25): the frame format and its transport are here and tested (tests/net/stage_rpc_test.cpp: bit-exact
// round trips, window latency on the link).  The engine side - a remote stage process that runs its layers' verify windows,
// prompt chunks, commits and session operations on a command stream - needs two GPU hosts to build and test and is not in
// this release (the Flash-Next parallel / RoPE / RPC plan).
#pragma once

#include "strata/net/transport.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::net {

struct StageHandoffHeader {
    uint64_t window;      ///< the window's sequence number (both stages count the same windows)
    uint32_t n_rows;      ///< rows (tokens) in the window, 1..8
    uint32_t row_floats;  ///< handoff_floats(geometry)
    int64_t pos0;         ///< the first row's position
    uint32_t kind;        ///< 0 decode / verify window, 1 prompt chunk, 2 commit (n_rows = tokens kept), 3 session op
    uint32_t flags;
};

/// Send one window's hand-off rows (n_rows x row_floats fp32) as one frame.
inline bool send_handoff(Channel& ch, const StageHandoffHeader& h, const float* rows, std::string& error) {
    const Iov p[2] = {{&h, sizeof h}, {rows, (size_t) h.n_rows * h.row_floats * sizeof(float)}};
    return ch.send(kStageHandoff, p, 2, error);
}

/// Receive one; `rows` is resized to n_rows x row_floats.  False on a transport error or a malformed frame.
inline bool recv_handoff(Channel& ch, StageHandoffHeader& h, std::vector<float>& rows, std::string& error, int timeout_ms = -1) {
    uint32_t type = 0;
    std::vector<uint8_t> d;
    if (!ch.recv(type, d, error, timeout_ms)) return false;
    if (type != kStageHandoff || d.size() < sizeof h) { error = "not a stage hand-off"; return false; }
    std::memcpy(&h, d.data(), sizeof h);
    const size_t n = (size_t) h.n_rows * h.row_floats;
    if (h.n_rows > 64 || d.size() != sizeof h + n * sizeof(float)) { error = "a hand-off of the wrong size"; return false; }
    rows.resize(n);
    std::memcpy(rows.data(), d.data() + sizeof h, n * sizeof(float));
    return true;
}

}  // namespace strata::net
