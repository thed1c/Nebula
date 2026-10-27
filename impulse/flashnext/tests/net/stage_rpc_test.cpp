// stage_rpc_test: the layer pipeline's hand-off frames (include/strata/net/stage_rpc.hpp) across two processes on
// localhost, no GPU: every window's rows arrive bit for bit (random fp32 including NaN payloads, infinities, -0 and
// denormals), and the window round-trip time on this link, encrypted and plaintext (what a cross-host stage would add per
// window on top of the network's own latency).
#include "strata/net/stage_rpc.hpp"
#include "strata/net/transport.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace strata::net;

static int fails = 0;
static void check(const char* what, bool ok, const std::string& detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  -- ", detail.c_str());
    if (!ok) ++fails;
}

static constexpr uint32_t kRowFloats = 4 * 2560 + 2560 + 4;   // Flash-Next: hc*n_embd + n_embd + hc = 12,804

static void run(bool plaintext) {
    Security sec;
    const char* k = "stage-rpc-test-secret-0123456789";
    sec.secret.assign(k, k + std::strlen(k));
    sec.plaintext = plaintext;
    Listener L;
    std::string e;
    L.open(Endpoint{"127.0.0.1", 0}, e);
    const int port = L.port();
    const pid_t pid = fork();
    if (pid == 0) {   // the next stage: echoes every window back (as if its layers were the identity)
        auto ch = L.accept(sec, "stage-1", e, 10000);
        if (!ch) _exit(3);
        StageHandoffHeader h{};
        std::vector<float> rows;
        while (recv_handoff(*ch, h, rows, e)) {
            if (!send_handoff(*ch, h, rows.data(), e)) _exit(4);
        }
        _exit(0);
    }
    L.close();
    auto ch = tcp_connect(Endpoint{"127.0.0.1", port}, sec, "stage-0", e);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);
    bool exact = ch != nullptr;
    std::vector<double> rtt[9];
    for (int w = 0; exact && w < 400; ++w) {
        const uint32_t n = 1 + (uint32_t) (w % 8);
        std::vector<float> rows((size_t) n * kRowFloats);
        for (auto& v : rows) v = u(rng);
        uint32_t special[5] = {0x7fc12345u, 0x7f800000u, 0x80000000u, 0x00000001u, 0xff800000u};   // NaN payload, inf, -0, denormal, -inf
        for (int s = 0; s < 5; ++s) std::memcpy(&rows[(size_t) (w * 7 + s) % rows.size()], &special[s], 4);
        StageHandoffHeader h{(uint64_t) w, n, kRowFloats, (int64_t) w * 8, 0, 0};
        const auto t0 = std::chrono::steady_clock::now();
        StageHandoffHeader back{};
        std::vector<float> got;
        exact = send_handoff(*ch, h, rows.data(), e) && recv_handoff(*ch, back, got, e, 10000) && back.window == h.window &&
                back.n_rows == n && got.size() == rows.size() && std::memcmp(got.data(), rows.data(), rows.size() * 4) == 0;
        rtt[n].push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    const std::string mode = plaintext ? "plaintext (authenticated)" : "aes-256-gcm";
    check(("hand-offs across two processes, " + mode + ": 400 windows of 1-8 rows, bit for bit").c_str(), exact, e);
    for (int n : {1, 5, 8}) {
        auto& v = rtt[n];
        std::sort(v.begin(), v.end());
        if (!v.empty())
            std::printf("     %s: %d row%s (%.0f KB) round trip median %.0f us, p90 %.0f us\n", mode.c_str(), n, n == 1 ? "" : "s",
                        n * kRowFloats * 4 / 1024.0, v[v.size() / 2], v[v.size() * 9 / 10]);
    }
    if (ch) ch->close();
    int st = 0;
    waitpid(pid, &st, 0);
    check(("... the stage process ended cleanly (" + mode + ")").c_str(), WIFEXITED(st) && WEXITSTATUS(st) == 0);
}

int main() {
    run(false);
    run(true);
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
