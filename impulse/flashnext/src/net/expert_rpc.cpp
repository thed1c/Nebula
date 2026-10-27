// src/net/expert_rpc.cpp - see include/strata/net/expert_rpc.hpp.
#include "strata/net/expert_rpc.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace strata::net {

using namespace strata::kernels::cpu;

bool compute_expert_rows(ExpertPool& pool, const ExpertLayout& lay, const std::function<const uint8_t*(int64_t, int64_t)>& blob,
                         int64_t layer, const float* x, int n_tok, const ExpertEntry* ents, int n_ent, float* rows,
                         std::string& error) {
    if (n_tok < 1 || n_tok > MAXT) { error = "a window of more tokens than the multi-token expert kernel takes"; return false; }
    if (n_ent < 0 || n_ent > n_tok * ExpertPool::kMaxSplitMulti) { error = "too many entries"; return false; }
    if (n_ent == 0) return true;
    const bool native = lay.native;
    if (native && (layer < 0 || (size_t) layer >= lay.fmt.size())) { error = "a layer the pack's layout does not have"; return false; }
    // the activations, per token, exactly as expert_pool_dispatch_multi quantizes them
    std::vector<ActQ> act((size_t) MAXT);
    std::vector<uint8_t> nact(native ? (size_t) MAXT * kNativeActBytes : 0);
    std::vector<bool> used((size_t) n_tok, false);
    for (int i = 0; i < n_ent; ++i) {
        if (ents[i].t < 0 || ents[i].t >= n_tok) { error = "an entry's token is outside the window"; return false; }
        used[(size_t) ents[i].t] = true;
    }
    for (int t = 0; t < n_tok; ++t) {
        if (!used[(size_t) t]) continue;
        if (native && q2_native_kernels(lay.fmt[(size_t) layer].gu_type)) act_quant_any(x + (size_t) t * H, H, act[(size_t) t]);
        else if (native) native_quant_act(lay.fmt[(size_t) layer], x + (size_t) t * H, nact.data() + (size_t) t * kNativeActBytes);
        else act_quant_q8_1(x + (size_t) t * H, H, act[(size_t) t]);
    }
    // one multi-token job per distinct expert, in entry order
    std::vector<ExpertJobMulti> jobs;
    jobs.reserve((size_t) n_ent);
    std::unordered_map<int32_t, int> job_of;
    for (int i = 0; i < n_ent; ++i) {
        const int32_t e = ents[i].e;
        auto it = job_of.find(e);
        int j;
        if (it == job_of.end()) {
            const uint8_t* b = blob(layer, e);
            if (b == nullptr) { error = "no blob for expert " + std::to_string(e) + " of layer " + std::to_string(layer); return false; }
            j = (int) jobs.size();
            jobs.emplace_back();
            jobs.back().blob = b;
            jobs.back().nt = 0;
            job_of[e] = j;
        } else {
            j = it->second;
        }
        ExpertJobMulti& jb = jobs[(size_t) j];
        if (jb.nt >= MAXT) { error = "an expert routed more than once per token"; return false; }
        const int t = ents[i].t;
        jb.act[jb.nt] = &act[(size_t) t];
        jb.nact[jb.nt] = native ? nact.data() + (size_t) t * kNativeActBytes : nullptr;
        jb.out[jb.nt] = rows + (size_t) i * H;
        ++jb.nt;
    }
    // the pool call the engine makes (at most kMaxSplitMulti jobs per call)
    for (size_t at = 0; at < jobs.size(); at += ExpertPool::kMaxSplitMulti) {
        const int n = (int) std::min<size_t>(ExpertPool::kMaxSplitMulti, jobs.size() - at);
        if (native) pool.run_split_multi_native(lay.fmt[(size_t) layer], jobs.data() + at, n);
        else pool.run_split_multi(jobs.data() + at, n);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ client
bool ExpertRpcClient::parse(const std::string& spec, Endpoint& ep, int64_t& lo, int64_t& hi, std::string& error) {
    const size_t at = spec.rfind('@');
    if (!Endpoint::parse(spec.substr(0, at), ep, error)) return false;
    lo = 0;
    hi = (int64_t) 1 << 30;
    if (at != std::string::npos) {
        const std::string r = spec.substr(at + 1);
        const size_t dash = r.find('-');
        char* e1 = nullptr;
        char* e2 = nullptr;
        lo = std::strtoll(r.c_str(), &e1, 10);
        hi = dash == std::string::npos ? lo : std::strtoll(r.c_str() + dash + 1, &e2, 10);
        if (r.empty() || (dash == std::string::npos ? *e1 != '\0' : (e1 != r.c_str() + dash || *e2 != '\0')) || lo < 0 || hi < lo) {
            error = "expected host:port@A-B (layers A..B), got '" + spec + "'";
            return false;
        }
    }
    return true;
}

bool ExpertRpcClient::connect(const Endpoint& ep, int64_t lo, int64_t hi, const Security& sec, int64_t h, std::string& error,
                              const RdmaOptions* rdma) {
    ch_ = tcp_connect(ep, sec, "engine", error);
    if (ch_ && rdma != nullptr) ch_ = rdma_upgrade(std::move(ch_), false, sec, *rdma, error);
    if (!ch_) return false;
    ep_ = ep;
    lo_ = lo;
    hi_ = hi;
    h_ = h;
    // ping: the server says its layer range and width; a server that does not hold every layer asked for is refused
    if (!ch_->send(kPing, nullptr, 0, error)) return false;
    uint32_t type = 0;
    std::vector<uint8_t> d;
    if (!ch_->recv(type, d, error, 30000)) return false;
    if (type == kError) { error = "the expert server refused: " + std::string(d.begin(), d.end()); return false; }
    if (type != kPing || d.size() != 24) { error = "the expert server did not answer the ping"; return false; }
    int64_t s_lo = 0, s_hi = 0, s_h = 0;
    std::memcpy(&s_lo, d.data(), 8); std::memcpy(&s_hi, d.data() + 8, 8); std::memcpy(&s_h, d.data() + 16, 8);
    if (s_h != h) { error = "the expert server computes " + std::to_string(s_h) + "-wide rows, the engine " + std::to_string(h); return false; }
    if (lo < s_lo || hi > s_hi) {
        if (hi >= ((int64_t) 1 << 30) && lo >= s_lo) { hi_ = s_hi; }   // no range asked: the server's
        else { error = "the expert server holds layers " + std::to_string(s_lo) + "-" + std::to_string(s_hi) + ", asked for " +
                       std::to_string(lo) + "-" + std::to_string(hi); return false; }
    }
    return true;
}

void ExpertRpcClient::mark_broken(const std::string& why) {
    if (broken_) return;
    broken_ = true;
    pending_ = false;
    std::fprintf(stderr, "strata rpc experts: %s - layers %lld-%lld go back to this engine's CPU pool\n", why.c_str(),
                 (long long) lo_, (long long) hi_);
    if (ch_) ch_->close();
}

std::string ExpertRpcClient::describe() const {
    return ep_.str() + " layers " + std::to_string(lo_) + "-" + std::to_string(hi_) + " (" + (ch_ ? ch_->describe() : "closed") + ")";
}

bool ExpertRpcClient::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, int32_t* kind,
                            std::string& error) {
    pending_ = false;
    if (!serves(layer)) return true;
    idx_.clear();
    ents_.clear();
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (kind[i] == -1 && ids[i] >= 0) {
            idx_.push_back(i);
            ents_.push_back({(int32_t) (i / k), ids[i], 0.0f});
        }
    if (idx_.empty()) return true;
    ExpertReqHeader h{(uint32_t) layer, (uint32_t) n_tok, (uint32_t) h_, (uint32_t) ents_.size(), kRows, ++seq_};
    const Iov pieces[3] = {{&h, sizeof h}, {x, (size_t) (n_tok * h_) * sizeof(float)}, {ents_.data(), ents_.size() * sizeof(ExpertEntry)}};
    if (!ch_->send(kExpertReq, pieces, 3, error)) { mark_broken("the expert server's link failed (" + error + ")"); error.clear(); return true; }
    for (int64_t i : idx_) kind[i] = 2;
    bytes_out += sizeof h + (size_t) (n_tok * h_) * sizeof(float) + ents_.size() * sizeof(ExpertEntry);
    pending_ = true;
    ++requests;
    entries += (int64_t) idx_.size();
    return true;
}

bool ExpertRpcClient::finish(float* out, std::string& error) {
    if (!pending_) return true;
    pending_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t type = 0;
    if (!ch_->recv(type, resp_, error, 60000)) {
        const std::string why = "the expert server did not answer (" + error + ")";
        mark_broken(why);
        error = why;
        return false;   // this window's rows are missing: it fails; the next windows run locally
    }
    wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    bytes_in += resp_.size();
    ExpertRespHeader h{};
    if (resp_.size() >= sizeof h) std::memcpy(&h, resp_.data(), sizeof h);
    if (type != kExpertResp || resp_.size() < sizeof h || h.seq != seq_) {
        error = "the expert server answered out of step";
        mark_broken(error);
        return false;
    }
    if (h.status != 0) {
        error = "the expert server failed: " + std::string(resp_.begin() + sizeof h, resp_.end());
        mark_broken(error);
        return false;
    }
    if (h.n_rows != idx_.size() || h.h != (uint32_t) h_ || resp_.size() != sizeof h + idx_.size() * (size_t) h_ * sizeof(float)) {
        error = "the expert server sent rows of the wrong shape";
        mark_broken(error);
        return false;
    }
    const float* rows = (const float*) (resp_.data() + sizeof h);
    for (size_t j = 0; j < idx_.size(); ++j)
        std::memcpy(out + (size_t) idx_[j] * (size_t) h_, rows + j * (size_t) h_, (size_t) h_ * sizeof(float));
    return true;
}

// ------------------------------------------------------------------------------------------------ server
bool serve_expert_connection(Channel& ch, int64_t h, int64_t lo, int64_t hi, const ExpertCompute& compute, std::string& error,
                             int64_t* served) {
    std::vector<uint8_t> req;
    std::vector<float> rows;
    for (;;) {
        uint32_t type = 0;
        if (!ch.recv(type, req, error)) {
            if (error == "closed by the peer") { error.clear(); return true; }
            return false;
        }
        if (type == kBye) return true;
        if (type == kPing) {
            int64_t info[3] = {lo, hi, h};
            if (!ch.send(kPing, info, sizeof info, error)) return false;
            continue;
        }
        if (type != kExpertReq) { error = "an unexpected message " + std::to_string(type); return false; }
        ExpertReqHeader q{};
        auto fail = [&](uint32_t seq, const std::string& why) {
            ExpertRespHeader r{seq, 0, (uint32_t) h, 1};
            const Iov p[2] = {{&r, sizeof r}, {why.data(), why.size()}};
            return ch.send(kExpertResp, p, 2, error);
        };
        if (req.size() < sizeof q) { error = "a short expert request"; return false; }
        std::memcpy(&q, req.data(), sizeof q);
        const size_t want = sizeof q + (size_t) q.n_tok * q.h * sizeof(float) + (size_t) q.n_ent * sizeof(ExpertEntry);
        if (q.h != (uint32_t) h || req.size() != want) { if (!fail(q.seq, "a request of the wrong shape")) return false; continue; }
        if ((int64_t) q.layer < lo || (int64_t) q.layer > hi) {
            if (!fail(q.seq, "layer " + std::to_string(q.layer) + " is not served here")) return false;
            continue;
        }
        const float* x = (const float*) (req.data() + sizeof q);
        const ExpertEntry* ents = (const ExpertEntry*) (req.data() + sizeof q + (size_t) q.n_tok * q.h * sizeof(float));
        const size_t n_out = q.mode == kReduce ? q.n_tok : q.n_ent;
        rows.assign(n_out * (size_t) h, 0.0f);
        std::string why;
        if (!compute((int64_t) q.layer, x, (int) q.n_tok, ents, (int) q.n_ent, q.mode, rows.data(), why)) {
            if (!fail(q.seq, why)) return false;
            continue;
        }
        ExpertRespHeader r{q.seq, (uint32_t) n_out, (uint32_t) h, 0};
        const Iov p[2] = {{&r, sizeof r}, {rows.data(), rows.size() * sizeof(float)}};
        if (!ch.send(kExpertResp, p, 2, error)) return false;
        if (served) ++*served;
    }
}

}  // namespace strata::net
