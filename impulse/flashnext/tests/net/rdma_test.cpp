// rdma_test: the transport's RDMA backend (include/strata/net/rdma.hpp) on this host: a TCP connection upgraded to an RC
// queue pair on an active RDMA port (a ConnectX port with a link, or a soft-RoCE device from tools/rdma-setup.sh), both ends
// in this process (two threads), messages both ways bit for bit (encrypted and plaintext), and the round-trip time.
// Skipped (exit 0, says why) when the build has no RDMA or no port is active; STRATA_RDMA_DEVICE picks the device.
#include "strata/net/rdma.hpp"
#include "strata/net/stage_rpc.hpp"
#include "strata/net/transport.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::net;

static int fails = 0;
static void check(const char* what, bool ok, const std::string& detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  -- ", detail.c_str());
    if (!ok) ++fails;
}

static void run(bool plaintext, const RdmaOptions& opt) {
    Security sec;
    const char* k = "rdma-test-secret-0123456789abcdef";
    sec.secret.assign(k, k + std::strlen(k));
    sec.plaintext = plaintext;
    Listener L;
    std::string e, se;
    L.open(Endpoint{"127.0.0.1", 0}, e);
    std::unique_ptr<Channel> s;
    std::thread srv([&] {
        auto t = L.accept(sec, "rdma-server", se, 10000);
        if (t) s = rdma_upgrade(std::move(t), true, sec, opt, se);
    });
    auto c = tcp_connect(Endpoint{"127.0.0.1", L.port()}, sec, "rdma-client", e);
    if (c) c = rdma_upgrade(std::move(c), false, sec, opt, e);
    srv.join();
    const std::string mode = plaintext ? "plaintext" : "encrypted";
    check(("RDMA upgrade, " + mode + ": " + (c ? c->describe() : e + " | " + se)).c_str(), c && s);
    if (!c || !s) return;
    std::thread echo([&] {
        StageHandoffHeader h{};
        std::vector<float> rows;
        std::string ee;
        while (recv_handoff(*s, h, rows, ee, 5000)) if (!send_handoff(*s, h, rows.data(), ee)) break;
    });
    std::mt19937 rng(3);
    bool exact = true;
    std::vector<double> rtt;
    const uint32_t rf = 12804;
    for (int w = 0; exact && w < 300; ++w) {
        const uint32_t n = 1 + (uint32_t) (w % 8);
        std::vector<float> rows((size_t) n * rf);
        for (auto& v : rows) v = (float) (rng() % 100000) * 1e-3f;
        StageHandoffHeader h{(uint64_t) w, n, rf, 0, 0, 0}, back{};
        std::vector<float> got;
        const auto t0 = std::chrono::steady_clock::now();
        exact = send_handoff(*c, h, rows.data(), e) && recv_handoff(*c, back, got, e, 5000) && got.size() == rows.size() &&
                std::memcmp(got.data(), rows.data(), rows.size() * 4) == 0;
        if (n == 5) rtt.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    check(("... 300 windows of 1-8 rows (" + mode + "), bit for bit, through the ring and its credits").c_str(), exact, e);
    std::sort(rtt.begin(), rtt.end());
    if (!rtt.empty()) std::printf("     %s: 5-row window (250 KB) round trip median %.0f us\n", mode.c_str(), rtt[rtt.size() / 2]);
    c->close();
    echo.join();
}

int main() {
    if (!rdma_available()) {
        std::printf("skip: this build has no RDMA (libibverbs headers missing at build time; tools/rdma-setup.sh installs them)\n");
        return 0;
    }
    const std::string devs = rdma_devices();
    std::printf("     RDMA devices: %s\n", devs.c_str());
    if (devs.find("ACTIVE") == std::string::npos) {
        std::printf("skip: no active RDMA port (a ConnectX port needs a link; tools/rdma-setup.sh --soft-roce <iface> adds rxe0)\n");
        return 0;
    }
    RdmaOptions opt;
    if (const char* d = std::getenv("STRATA_RDMA_DEVICE")) opt.device = d;
    opt.slots = 8;
    opt.slot_bytes = 1u << 20;
    run(false, opt);
    run(true, opt);
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
