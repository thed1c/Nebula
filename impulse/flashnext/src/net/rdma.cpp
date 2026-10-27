// src/net/rdma.cpp - see include/strata/net/rdma.hpp.  libibverbs (rdma-core); compiled with STRATA_RDMA.
#include "strata/net/rdma.hpp"

#include "crypto.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#if defined(STRATA_RDMA)
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include <atomic>
#include <mutex>
#endif

namespace strata::net {

#if !defined(STRATA_RDMA)

bool rdma_available() { return false; }
std::unique_ptr<Channel> rdma_upgrade(std::unique_ptr<Channel>, bool, const Security&, const RdmaOptions&, std::string& error) {
    error = "this build has no RDMA (libibverbs headers were missing at build time: tools/rdma-setup.sh installs them)";
    return nullptr;
}
std::string rdma_devices() { return "(built without RDMA)"; }

#else

using namespace detail;

bool rdma_available() { return true; }

namespace {

// what each side tells the other over the TCP channel
struct QpInfo {
    uint32_t magic;        // "RDMA"
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    uint8_t link_ib;       // 1: InfiniBand (LID addressing), 0: Ethernet (RoCE, GID)
    uint8_t gid_index;
    uint8_t gid[16];
    uint64_t ring_addr;    // the receive ring
    uint32_t ring_rkey;
    uint32_t slots;
    uint64_t slot_bytes;
    uint64_t credit_addr;  // the 8-byte word the peer writes its consumed count into
    uint32_t credit_rkey;
    uint32_t mtu;          // enum ibv_mtu
    uint8_t nonce[32];     // keys the RDMA frames
};
constexpr uint32_t kQpMagic = 0x414d4452;   // "RDMA"

struct FrameHead {          // at the start of a slot
    uint32_t type;
    uint32_t flags;
    uint64_t len;           // payload bytes (without the tag)
};

// the port's RoCE v2 GID (sysfs gid_attrs/types), else 0
int pick_gid(const char* dev, int port) {
    if (const char* e = std::getenv("STRATA_RDMA_GID_INDEX")) return std::atoi(e);
    for (int i = 0; i < 16; ++i) {
        std::ifstream f(std::string("/sys/class/infiniband/") + dev + "/ports/" + std::to_string(port) + "/gid_attrs/types/" + std::to_string(i));
        std::string t;
        if (f && std::getline(f, t) && t.find("v2") != std::string::npos) return i;
    }
    return 0;
}

class RdmaChannel final : public Channel {
public:
    using Channel::send;
    ~RdmaChannel() override { close(); teardown(); }

    bool setup(std::unique_ptr<Channel> tcp, bool server, const Security& sec, const RdmaOptions& opt, std::string& error) {
        tcp_ = std::move(tcp);
        peer_role_ = tcp_->peer_role();
        int n = 0;
        ibv_device** list = ibv_get_device_list(&n);
        if (list == nullptr || n == 0) { error = "no RDMA device (ibv_get_device_list)"; if (list) ibv_free_device_list(list); return false; }
        ibv_device* dev = nullptr;
        for (int i = 0; i < n; ++i)
            if (opt.device.empty() || opt.device == ibv_get_device_name(list[i])) { dev = list[i]; break; }
        if (dev == nullptr) { error = "no RDMA device named " + opt.device; ibv_free_device_list(list); return false; }
        dev_name_ = ibv_get_device_name(dev);
        ctx_ = ibv_open_device(dev);
        ibv_free_device_list(list);
        if (ctx_ == nullptr) { error = "cannot open " + dev_name_; return false; }
        ibv_port_attr pa{};
        if (ibv_query_port(ctx_, (uint8_t) opt.port, &pa) != 0) { error = "cannot query " + dev_name_ + " port " + std::to_string(opt.port); return false; }
        if (pa.state != IBV_PORT_ACTIVE) { error = dev_name_ + " port " + std::to_string(opt.port) + " is not active (no link: a cable, the switch)"; return false; }
        port_ = opt.port;
        slots_ = opt.slots;
        slot_bytes_ = opt.slot_bytes;
        pd_ = ibv_alloc_pd(ctx_);
        if (pd_ == nullptr) { error = "ibv_alloc_pd failed"; return false; }
        const bool events = std::getenv("STRATA_RDMA_EVENTS") != nullptr;
        if (events) chan_ = ibv_create_comp_channel(ctx_);
        scq_ = ibv_create_cq(ctx_, slots_ * 2 + 8, nullptr, nullptr, 0);
        rcq_ = ibv_create_cq(ctx_, slots_ * 2 + 8, nullptr, chan_, 0);
        if (scq_ == nullptr || rcq_ == nullptr) { error = "ibv_create_cq failed"; return false; }
        // the receive ring, the staging buffer a send seals into, the credit words
        const size_t ring = (size_t) slots_ * slot_bytes_;
        ring_ = (uint8_t*) aligned_alloc(4096, ring);
        stage_ = (uint8_t*) aligned_alloc(4096, slot_bytes_);
        credit_ = (uint64_t*) aligned_alloc(64, 64);
        if (ring_ == nullptr || stage_ == nullptr || credit_ == nullptr) { error = "out of memory for the RDMA rings"; return false; }
        std::memset(credit_, 0, 64);
        const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        ring_mr_ = ibv_reg_mr(pd_, ring_, ring, acc);
        stage_mr_ = ibv_reg_mr(pd_, stage_, slot_bytes_, IBV_ACCESS_LOCAL_WRITE);
        credit_mr_ = ibv_reg_mr(pd_, credit_, 64, acc);
        if (ring_mr_ == nullptr || stage_mr_ == nullptr || credit_mr_ == nullptr) {
            error = "cannot register the RDMA buffers (" + std::to_string(ring >> 20) + " MiB): raise the locked-memory limit (ulimit -l; "
                    "tools/rdma-setup.sh sets it)";
            return false;
        }
        ibv_qp_init_attr qa{};
        qa.send_cq = scq_;
        qa.recv_cq = rcq_;
        qa.qp_type = IBV_QPT_RC;
        qa.cap.max_send_wr = (uint32_t) slots_ + 8;
        qa.cap.max_recv_wr = (uint32_t) slots_ + 8;
        qa.cap.max_send_sge = 1;
        qa.cap.max_recv_sge = 1;
        qa.cap.max_inline_data = 64;   // the 8-byte credit writes go inline
        qp_ = ibv_create_qp(pd_, &qa);
        if (qp_ == nullptr) { error = "ibv_create_qp failed"; return false; }
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = (uint8_t) port_;
        a.qp_access_flags = acc;
        if (ibv_modify_qp(qp_, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) { error = "QP -> INIT failed"; return false; }
        for (int i = 0; i < slots_ + 4; ++i) if (!post_recv()) { error = "ibv_post_recv failed"; return false; }
        // what this side is
        QpInfo me{};
        me.magic = kQpMagic;
        me.qpn = qp_->qp_num;
        uint8_t rnd[4];
        random_bytes(rnd, 4);
        me.psn = ((uint32_t) rnd[0] | (uint32_t) rnd[1] << 8 | (uint32_t) rnd[2] << 16) & 0xffffff;
        me.lid = pa.lid;
        me.link_ib = pa.link_layer == IBV_LINK_LAYER_INFINIBAND ? 1 : 0;
        me.gid_index = (uint8_t) (opt.gid_index >= 0 ? opt.gid_index : pick_gid(dev_name_.c_str(), port_));
        ibv_gid gid{};
        if (!me.link_ib && ibv_query_gid(ctx_, (uint8_t) port_, me.gid_index, &gid) != 0) { error = "cannot read the port's GID"; return false; }
        std::memcpy(me.gid, gid.raw, 16);
        me.ring_addr = (uint64_t) (uintptr_t) ring_;
        me.ring_rkey = ring_mr_->rkey;
        me.slots = (uint32_t) slots_;
        me.slot_bytes = slot_bytes_;
        me.credit_addr = (uint64_t) (uintptr_t) credit_;
        me.credit_rkey = credit_mr_->rkey;
        me.mtu = (uint32_t) pa.active_mtu;
        random_bytes(me.nonce, sizeof me.nonce);
        // exchange over the (encrypted, authenticated) TCP channel
        QpInfo peer{};
        uint32_t t = 0;
        std::vector<uint8_t> d;
        if (!tcp_->send(kPing, &me, sizeof me, error) || !tcp_->recv(t, d, error, 15000)) return false;
        if (t != kPing || d.size() != sizeof peer) { error = "the peer did not send its RDMA queue pair"; return false; }
        std::memcpy(&peer, d.data(), sizeof peer);
        if (peer.magic != kQpMagic) { error = "a bad RDMA queue pair message"; return false; }
        if (peer.slots != me.slots || peer.slot_bytes != me.slot_bytes) { error = "the two ends' RDMA rings differ (slots / slot size)"; return false; }
        if (peer.link_ib != me.link_ib) { error = "one end is InfiniBand, the other Ethernet (RoCE)"; return false; }
        peer_ring_ = peer.ring_addr;
        peer_ring_rkey_ = peer.ring_rkey;
        peer_credit_ = peer.credit_addr;
        peer_credit_rkey_ = peer.credit_rkey;
        // RTR, RTS
        a = ibv_qp_attr{};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = (ibv_mtu) std::min(me.mtu, peer.mtu);
        a.dest_qp_num = peer.qpn;
        a.rq_psn = peer.psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 12;
        a.ah_attr.port_num = (uint8_t) port_;
        if (me.link_ib) {
            a.ah_attr.dlid = peer.lid;
        } else {
            a.ah_attr.is_global = 1;
            std::memcpy(a.ah_attr.grh.dgid.raw, peer.gid, 16);
            a.ah_attr.grh.sgid_index = me.gid_index;
            a.ah_attr.grh.hop_limit = 64;
        }
        if (ibv_modify_qp(qp_, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) { error = "QP -> RTR failed (GID / LID, the link)"; return false; }
        a = ibv_qp_attr{};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = 14;
        a.retry_cnt = 7;
        a.rnr_retry = 7;
        a.sq_psn = me.psn;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(qp_, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                       IBV_QP_MAX_QP_RD_ATOMIC) != 0) { error = "QP -> RTS failed"; return false; }
        // the RDMA frames' keys: the secret and both sides' fresh nonces (plaintext: none)
        if (!sec.plaintext && !sec.secret.empty()) {
            uint8_t salt[64];
            std::memcpy(salt, server ? peer.nonce : me.nonce, 32);
            std::memcpy(salt + 32, server ? me.nonce : peer.nonce, 32);
            uint8_t kc[kKey], ks[kKey];
            if (!hkdf(sec.secret, salt, sizeof salt, "strata-net/1 rdma c2s", kc, kKey) ||
                !hkdf(sec.secret, salt, sizeof salt, "strata-net/1 rdma s2c", ks, kKey) ||
                !tx_.init(server ? ks : kc, true) || !rx_.init(server ? kc : ks, false)) { error = "RDMA key setup failed"; return false; }
#if defined(STRATA_NET_CRYPTO)
            OPENSSL_cleanse(kc, kKey);
            OPENSSL_cleanse(ks, kKey);
#endif
        }
        // both sides ready before any write lands (a write to a QP not yet in RTR would be lost)
        const char ready[] = "rdma-ready";
        if (!tcp_->send(kPing, ready, sizeof ready, error) || !tcp_->recv(t, d, error, 15000) || t != kPing) { error = "the peer did not get ready"; return false; }
        what_ = std::string("rdma ") + dev_name_ + ":" + std::to_string(port_) + (me.link_ib ? " (InfiniBand)" : " (RoCE)") +
                (tx_.on ? " + aes-256-gcm" : sec.secret.empty() ? " (plaintext, unauthenticated)" : " (plaintext, authenticated)");
        return true;
    }

    bool send(uint32_t type, const Iov* pieces, int n, std::string& error) override {
        if (qp_ == nullptr || closed_) { error = "closed"; return false; }
        size_t total = 0;
        for (int i = 0; i < n; ++i) total += pieces[i].size;
        const size_t need = sizeof(FrameHead) + total + (tx_.on ? kTag : 0);
        if (need > slot_bytes_) { error = "a message larger than an RDMA slot (" + std::to_string(slot_bytes_ >> 20) + " MiB)"; return false; }
        // a free slot of the peer's ring: sent - consumed (the peer writes its count into credit_[0])
        const auto t0 = std::chrono::steady_clock::now();
        while (sent_ - __atomic_load_n(&credit_[0], __ATOMIC_ACQUIRE) >= (uint64_t) slots_) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) { error = "the peer stopped reading (no free RDMA slot for 60 s)"; return false; }
            std::this_thread::yield();
        }
        FrameHead h{type, 0, (uint64_t) total};
        std::memcpy(stage_, &h, sizeof h);
        if (tx_.on) {
            sealed_.clear();
            if (!tx_.seal((const uint8_t*) &h, sizeof h, pieces, n, sealed_)) { error = "encryption failed"; return false; }
            std::memcpy(stage_ + sizeof h, sealed_.data(), sealed_.size());
        } else {
            size_t off = sizeof h;
            for (int i = 0; i < n; ++i) { std::memcpy(stage_ + off, pieces[i].data, pieces[i].size); off += pieces[i].size; }
        }
        const uint32_t slot = (uint32_t) (sent_ % (uint64_t) slots_);
        ibv_sge sge{(uint64_t) (uintptr_t) stage_, (uint32_t) need, stage_mr_->lkey};
        ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(slot);
        wr.wr.rdma.remote_addr = peer_ring_ + (uint64_t) slot * slot_bytes_;
        wr.wr.rdma.rkey = peer_ring_rkey_;
        if (ibv_post_send(qp_, &wr, &bad) != 0) { error = "ibv_post_send failed"; return false; }
        if (!wait_send(error)) return false;   // the staging buffer is free again
        ++sent_;
        return true;
    }

    bool recv(uint32_t& type, std::vector<uint8_t>& data, std::string& error, int timeout_ms) override {
        if (qp_ == nullptr || closed_) { error = "closed"; return false; }
        ibv_wc wc{};
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            const int r = ibv_poll_cq(rcq_, 1, &wc);
            if (r < 0) { error = "ibv_poll_cq failed"; return false; }
            if (r == 1) break;
            if (timeout_ms >= 0 && std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) { error = "timeout"; return false; }
            if (!tcp_alive()) { error = "closed by the peer"; close(); return false; }
            if (chan_ != nullptr) {   // STRATA_RDMA_EVENTS: sleep on the completion channel
                ibv_req_notify_cq(rcq_, 0);
                if (ibv_poll_cq(rcq_, 1, &wc) == 1) break;
                ibv_cq* cq = nullptr;
                void* cq_ctx = nullptr;
                if (ibv_get_cq_event(chan_, &cq, &cq_ctx) == 0) ibv_ack_cq_events(cq, 1);
            }
        }
        if (wc.status != IBV_WC_SUCCESS || wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
            error = std::string("an RDMA receive failed: ") + ibv_wc_status_str(wc.status);
            close();
            return false;
        }
        post_recv();
        const uint32_t slot = ntohl(wc.imm_data);
        if (slot >= (uint32_t) slots_) { error = "an RDMA write into a slot that does not exist"; close(); return false; }
        uint8_t* p = ring_ + (size_t) slot * slot_bytes_;
        FrameHead h{};
        std::memcpy(&h, p, sizeof h);
        if (h.flags != 0 || h.len + sizeof h + (rx_.on ? kTag : 0) > slot_bytes_) { error = "a malformed RDMA frame"; close(); return false; }
        data.assign(p + sizeof h, p + sizeof h + h.len + (rx_.on ? kTag : 0));
        if (rx_.on && !rx_.open((const uint8_t*) &h, sizeof h, data.data(), (size_t) h.len, data.data() + h.len)) {
            error = "an RDMA frame failed authentication: connection closed";
            close();
            return false;
        }
        data.resize((size_t) h.len);
        type = h.type;
        ++consumed_;
        give_credit();
        return true;
    }

    void close() override {
        if (closed_) return;
        closed_ = true;
        if (tcp_) tcp_->close();
    }
    std::string describe() const override { return what_; }
    const std::string& peer_role() const override { return peer_role_; }

private:
    std::unique_ptr<Channel> tcp_;
    std::string peer_role_, what_ = "rdma", dev_name_;
    ibv_context* ctx_ = nullptr;
    ibv_pd* pd_ = nullptr;
    ibv_comp_channel* chan_ = nullptr;
    ibv_cq *scq_ = nullptr, *rcq_ = nullptr;
    ibv_qp* qp_ = nullptr;
    ibv_mr *ring_mr_ = nullptr, *stage_mr_ = nullptr, *credit_mr_ = nullptr;
    uint8_t *ring_ = nullptr, *stage_ = nullptr;
    uint64_t* credit_ = nullptr;   // [0]: the peer's consumed count (it writes it); [1]: ours, the source of the credit write
    uint64_t peer_ring_ = 0, peer_credit_ = 0;
    uint32_t peer_ring_rkey_ = 0, peer_credit_rkey_ = 0;
    int port_ = 1, slots_ = 16;
    size_t slot_bytes_ = 0;
    uint64_t sent_ = 0, consumed_ = 0;
    bool closed_ = false;
    Gcm tx_, rx_;
    std::vector<uint8_t> sealed_;

    bool post_recv() {
        ibv_recv_wr wr{}, *bad = nullptr;
        wr.wr_id = 2;
        wr.sg_list = nullptr;   // a WRITE WITH IMM consumes a receive with no buffer
        wr.num_sge = 0;
        return ibv_post_recv(qp_, &wr, &bad) == 0;
    }
    // the send queue's completions, polled by whichever thread asks (the sender waits for its write, the receiver drains its
    // signaled credit writes so the completion queue never fills); `cq_mu_` keeps the two from taking each other's
    std::mutex cq_mu_;
    std::atomic<bool> send_done_{false}, send_failed_{false};
    std::string send_error_;
    void drain_send_cq() {
        std::lock_guard<std::mutex> g(cq_mu_);
        ibv_wc wc[8];
        for (;;) {
            const int r = ibv_poll_cq(scq_, 8, wc);
            if (r <= 0) return;
            for (int i = 0; i < r; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) { send_error_ = std::string("an RDMA write failed: ") + ibv_wc_status_str(wc[i].status); send_failed_ = true; }
                if (wc[i].wr_id == 1) send_done_ = true;
            }
        }
    }
    bool wait_send(std::string& error) {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            drain_send_cq();
            if (send_failed_) { error = send_error_; close(); return false; }
            if (send_done_.exchange(false)) return true;
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) { error = "an RDMA write did not complete in 30 s"; return false; }
        }
    }
    // tell the peer how many of its messages this side has consumed (an RDMA write of the count into its credit word)
    void give_credit() {
        credit_[1] = consumed_;
        ibv_sge sge{(uint64_t) (uintptr_t) &credit_[1], 8, credit_mr_->lkey};
        ibv_send_wr wr{}, *bad = nullptr;
        wr.wr_id = 3;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        const bool signaled = (consumed_ % (uint64_t) slots_) == 0;   // (unsignaled WRs are freed by a later signaled one)
        wr.send_flags = IBV_SEND_INLINE | (signaled ? IBV_SEND_SIGNALED : 0);
        wr.wr.rdma.remote_addr = peer_credit_;
        wr.wr.rdma.rkey = peer_credit_rkey_;
        ibv_post_send(qp_, &wr, &bad);
        if (signaled) drain_send_cq();
    }
    // the TCP channel carries nothing after the setup: it reading "closed" means the peer is gone (checked every 100 ms)
    std::chrono::steady_clock::time_point tcp_checked_{};
    bool tcp_alive() {
        if (tcp_ == nullptr) return false;
        const auto now = std::chrono::steady_clock::now();
        if (now - tcp_checked_ < std::chrono::milliseconds(100)) return true;
        tcp_checked_ = now;
        uint32_t t = 0;
        std::vector<uint8_t> d;
        std::string e;
        return tcp_->recv(t, d, e, 0) || e == "timeout";
    }
    void teardown() {
        if (qp_) ibv_destroy_qp(qp_);
        if (ring_mr_) ibv_dereg_mr(ring_mr_);
        if (stage_mr_) ibv_dereg_mr(stage_mr_);
        if (credit_mr_) ibv_dereg_mr(credit_mr_);
        if (scq_) ibv_destroy_cq(scq_);
        if (rcq_) ibv_destroy_cq(rcq_);
        if (chan_) ibv_destroy_comp_channel(chan_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ctx_) ibv_close_device(ctx_);
        std::free(ring_);
        std::free(stage_);
        std::free(credit_);
        qp_ = nullptr;
    }
};

}  // namespace

std::unique_ptr<Channel> rdma_upgrade(std::unique_ptr<Channel> tcp, bool server, const Security& sec, const RdmaOptions& opt,
                                      std::string& error) {
    if (!tcp) { error = "no TCP channel to upgrade"; return nullptr; }
    auto ch = std::make_unique<RdmaChannel>();
    if (!ch->setup(std::move(tcp), server, sec, opt, error)) return nullptr;
    return ch;
}

std::string rdma_devices() {
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    if (list == nullptr || n == 0) { if (list) ibv_free_device_list(list); return "no RDMA devices"; }
    std::ostringstream os;
    for (int i = 0; i < n; ++i) {
        ibv_context* c = ibv_open_device(list[i]);
        if (c == nullptr) continue;
        ibv_device_attr da{};
        ibv_query_device(c, &da);
        for (int p = 1; p <= da.phys_port_cnt; ++p) {
            ibv_port_attr pa{};
            if (ibv_query_port(c, (uint8_t) p, &pa) != 0) continue;
            os << (os.tellp() > 0 ? "; " : "") << ibv_get_device_name(list[i]) << " port " << p << " "
               << (pa.link_layer == IBV_LINK_LAYER_INFINIBAND ? "InfiniBand" : "Ethernet") << " "
               << (pa.state == IBV_PORT_ACTIVE ? "ACTIVE" : "DOWN");
        }
        ibv_close_device(c);
    }
    ibv_free_device_list(list);
    return os.str();
}

#endif

}  // namespace strata::net
