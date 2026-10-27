// expert_rpc_test: the remote expert tier (include/strata/net/expert_rpc.hpp), CPU only, two processes on localhost.
// The server runs in a child process (fork) with its own expert pool; the parent computes the same windows with its own
// pool through compute_expert_rows - the engine's CPU path - and every row the server sends must be bit for bit the same.
//   - synthetic: a canonical layout, random blobs (no model needed)
//   - the real pack: $STRATA_TEST_PACK (a pack dir with experts.bin), layers 0 and 47, random activations (skipped when
//     absent)
//   - the client as the engine drives it: begin() takes only the CPU entries (kind -1) of served layers and marks them 2,
//     finish() writes their rows where the pool would, the other rows untouched; a layer it does not serve: nothing sent
//   - the server dies mid-run: the window fails, the client stops serving (the layers go back to the local pool)
//   - kReduce: one weighted row per token
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/net/expert_rpc.hpp"
#include "strata/net/transport.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::net;
using kernels::cpu::H;

static int fails = 0;
static void check(const char* what, bool ok, const std::string& detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  -- ", detail.c_str());
    if (!ok) ++fails;
}

using BlobFn = std::function<const uint8_t*(int64_t, int64_t)>;

static Security sec() {
    Security s;
    const char* k = "expert-rpc-test-secret-0123456789";
    s.secret.assign(k, k + std::strlen(k));
    return s;
}

// fork a server for layers [lo, hi]; returns its pid and port
static pid_t spawn_server(const BlobFn& blob, int64_t lo, int64_t hi, int& port, bool reduce_ok = true) {
    Listener L;
    std::string e;
    L.open(Endpoint{"127.0.0.1", 0}, e);
    port = L.port();
    const pid_t pid = fork();
    if (pid == 0) {
        kernels::cpu::ExpertPool pool(3, false, true);
        const auto& lay = kernels::cpu::expert_layout();
        std::vector<float> rows;
        ExpertCompute compute = [&](int64_t layer, const float* x, int n_tok, const ExpertEntry* ents, int n_ent, uint32_t mode,
                                    float* out, std::string& err) {
            if (mode == kRows) return compute_expert_rows(pool, lay, blob, layer, x, n_tok, ents, n_ent, out, err);
            if (!reduce_ok) { err = "no reduce"; return false; }
            rows.assign((size_t) n_ent * H, 0.0f);
            if (!compute_expert_rows(pool, lay, blob, layer, x, n_tok, ents, n_ent, rows.data(), err)) return false;
            for (int i = 0; i < n_ent; ++i)
                for (int64_t c = 0; c < H; ++c) out[(size_t) ents[i].t * H + (size_t) c] += ents[i].w * rows[(size_t) i * H + (size_t) c];
            return true;
        };
        auto ch = L.accept(sec(), "expert-server", e, 20000);
        if (!ch) _exit(3);
        if (!serve_expert_connection(*ch, H, lo, hi, compute, e)) _exit(4);
        _exit(0);
    }
    L.close();
    return pid;
}

// random windows: n_tok tokens x k distinct experts in [0, n_expert)
struct Window {
    int n_tok, k;
    std::vector<float> x;
    std::vector<int32_t> ids;
};
static Window make_window(std::mt19937& rng, int n_tok, int k, int n_expert, float scale) {
    Window w{n_tok, k, std::vector<float>((size_t) n_tok * H), std::vector<int32_t>((size_t) n_tok * k)};
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& v : w.x) v = nd(rng);
    for (int t = 0; t < n_tok; ++t) {
        std::vector<int32_t> pick;
        while ((int) pick.size() < k) {
            const int32_t e = (int32_t) (rng() % (uint32_t) n_expert);
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int j = 0; j < k; ++j) w.ids[(size_t) t * k + j] = pick[(size_t) j];
    }
    return w;
}

// the client against a forked server, compared with local compute row for row
static void exact_run(const char* name, const BlobFn& blob, int64_t lo, int64_t hi, int n_expert, float scale, int windows) {
    int port = 0;
    // STRATA_TEST_SERVER=port: a strata-rpc-experts already running on this host (same secret: the test's, in hex) instead
    // of the forked server
    const char* ext = std::getenv("STRATA_TEST_SERVER");
    const pid_t pid = ext ? -1 : spawn_server(blob, lo, hi, port);
    if (ext) port = std::atoi(ext);
    ExpertRpcClient C;
    std::string e;
    const bool up = C.connect(Endpoint{"127.0.0.1", port}, lo, hi, sec(), H, e);
    check((std::string(name) + ": the client connects (" + C.describe() + ")").c_str(), up, e);
    kernels::cpu::ExpertPool pool(3, false, true);
    const auto& lay = kernels::cpu::expert_layout();
    std::mt19937 rng(1234);
    int64_t rows_cmp = 0, rows_bad = 0, untouched_bad = 0, sent_bad = 0;
    const float sentinel = 12345.0f;
    for (int wi = 0; up && wi < windows; ++wi) {
        const int n_tok = 1 + wi % 8;
        const int k = 10;
        const int64_t layer = (wi % 2) ? hi : lo;
        Window w = make_window(rng, n_tok, k, n_expert, scale);
        // a mix as the dispatch leaves it: some entries a VRAM hit (0), the rest the CPU's (-1)
        std::vector<int32_t> kind((size_t) n_tok * k);
        for (size_t i = 0; i < kind.size(); ++i) kind[i] = (rng() % 3 == 0) ? 0 : -1;
        const std::vector<int32_t> kind0 = kind;
        std::vector<float> out((size_t) n_tok * k * H, sentinel);
        if (!C.begin(layer, w.x.data(), w.ids.data(), n_tok, k, kind.data(), e) || !C.finish(out.data(), e)) {
            check("... a window failed", false, e);
            break;
        }
        // the local reference: the same CPU entries through the engine's compute
        std::vector<ExpertEntry> ents;
        std::vector<size_t> idx;
        for (size_t i = 0; i < kind0.size(); ++i)
            if (kind0[i] == -1) { ents.push_back({(int32_t) (i / k), w.ids[i], 0.0f}); idx.push_back(i); }
        std::vector<float> ref(ents.size() * H);
        if (!compute_expert_rows(pool, lay, blob, layer, w.x.data(), n_tok, ents.data(), (int) ents.size(), ref.data(), e)) {
            check("... the local compute failed", false, e);
            break;
        }
        for (size_t j = 0; j < idx.size(); ++j) {
            ++rows_cmp;
            if (std::memcmp(out.data() + idx[j] * H, ref.data() + j * H, H * sizeof(float)) != 0) ++rows_bad;
            if (kind[idx[j]] != 2) ++sent_bad;
        }
        for (size_t i = 0; i < kind0.size(); ++i)
            if (kind0[i] != -1 && (kind[i] != kind0[i] || out[i * H] != sentinel || out[i * H + H - 1] != sentinel)) ++untouched_bad;
    }
    check((std::string(name) + ": every remote row = the local row, bit for bit").c_str(), up && rows_cmp > 0 && rows_bad == 0,
          std::to_string(rows_bad) + " of " + std::to_string(rows_cmp) + " rows differ");
    check((std::string(name) + ": the CPU entries were marked remote (2), the others untouched").c_str(), up && sent_bad == 0 && untouched_bad == 0,
          std::to_string(sent_bad) + "/" + std::to_string(untouched_bad));
    {   // a layer outside the range: nothing sent
        std::vector<int32_t> kind(10, -1);
        Window w = make_window(rng, 1, 10, n_expert, scale);
        std::vector<float> out(10 * H, sentinel);
        const int64_t before = C.requests;
        const bool ok = C.begin(hi + 1, w.x.data(), w.ids.data(), 1, 10, kind.data(), e) && C.finish(out.data(), e);
        check((std::string(name) + ": a layer it does not serve: nothing sent, the pool keeps it").c_str(),
              ok && C.requests == before && kind[0] == -1 && out[0] == sentinel);
    }
    std::printf("     %s: %lld requests, %lld entries, %.1f MB out, %.1f MB in, %.1f ms waited\n", name, (long long) C.requests,
                (long long) C.entries, C.bytes_out / 1e6, C.bytes_in / 1e6, C.wait_ms);
    if (pid < 0) return;
    kill(pid, SIGTERM);
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st) && WTERMSIG(st) != SIGTERM) std::printf("     (the server died of signal %d)\n", WTERMSIG(st));
    if (WIFEXITED(st) && WEXITSTATUS(st) != 0) std::printf("     (the server exited %d)\n", WEXITSTATUS(st));
}

// kReduce: one weighted row per token, against the same sum done locally
static void reduce_run(const BlobFn& syn, int64_t layer, int n_expert) {
    {
        int port = 0;
        const pid_t pid = spawn_server(syn, layer, layer, port);
        std::string e;
        auto ch = tcp_connect(Endpoint{"127.0.0.1", port}, sec(), "test", e);
        std::mt19937 rng(99);
        Window w = make_window(rng, 3, 4, n_expert, layer == 3 ? 1.0f : 0.05f);
        std::vector<ExpertEntry> ents;
        for (int i = 0; i < 12; ++i) ents.push_back({i / 4, w.ids[(size_t) i], 0.25f * (float) (1 + i % 4)});
        ExpertReqHeader q{(uint32_t) layer, 3, (uint32_t) H, 12, kReduce, 1};
        const Iov p[3] = {{&q, sizeof q}, {w.x.data(), w.x.size() * 4}, {ents.data(), ents.size() * sizeof(ExpertEntry)}};
        uint32_t t = 0;
        std::vector<uint8_t> resp;
        bool ok = ch && ch->send(kExpertReq, p, 3, e) && ch->recv(t, resp, e, 20000) && resp.size() == sizeof(ExpertRespHeader) + 3 * H * 4;
        if (ok) {
            kernels::cpu::ExpertPool pool(3, false, true);
            std::vector<float> rows(12 * H), red(3 * H, 0.0f);
            ok = compute_expert_rows(pool, kernels::cpu::expert_layout(), syn, layer, w.x.data(), 3, ents.data(), 12, rows.data(), e);
            for (int i = 0; i < 12; ++i)
                for (int64_t c = 0; c < H; ++c) red[(size_t) ents[(size_t) i].t * H + (size_t) c] += ents[(size_t) i].w * rows[(size_t) i * H + (size_t) c];
            ok = ok && std::memcmp(resp.data() + sizeof(ExpertRespHeader), red.data(), red.size() * 4) == 0;
        }
        check("kReduce: one weighted row per token (3 rows for 12 entries)", ok, e);
        if (ch) ch->close();
        kill(pid, SIGTERM);
        int st = 0;
        waitpid(pid, &st, 0);
    }
}

int main() {
    signal(SIGPIPE, SIG_IGN);
    // ---- synthetic canonical blobs (the default layout: no pack loaded)
    const int n_expert = 24;
    const uint64_t blob_bytes = kernels::cpu::expert_layout().blob_bytes(0);
    std::vector<uint8_t> blobs((size_t) 2 * n_expert * blob_bytes);
    {
        std::mt19937 g(7);
        // random codes, small scales: the blob's fp16 scale fields hold modest values so the rows stay finite (the test is
        // exactness, which holds for any bytes; finite rows make a difference easier to read)
        for (size_t i = 0; i < blobs.size(); ++i) blobs[i] = (uint8_t) (g() & 0x77);
    }
    const BlobFn syn = [&](int64_t l, int64_t e) -> const uint8_t* {
        if (l < 3 || l > 4 || e < 0 || e >= n_expert) return nullptr;
        return blobs.data() + ((size_t) (l - 3) * n_expert + (size_t) e) * blob_bytes;
    };
    // the canonical (S2) kernels need AVX-512 VNNI + VBMI (expert.cpp); elsewhere the engine runs native packs only
    const bool canon = kernels::cpu::cpu_avx512_ok();
    if (!canon) std::printf("skip the synthetic canonical blobs: this CPU has no AVX-512 VBMI (the canonical kernels' ISA)\n");
    if (canon) exact_run("synthetic (canonical)", syn, 3, 4, n_expert, 1.0f, 24);

    if (canon) reduce_run(syn, 3, n_expert);

    // ---- the server dies: the window fails, the client stops serving (it never computes: any CPU)
    {
        int port = 0;
        const pid_t pid = spawn_server(syn, 3, 4, port);
        ExpertRpcClient C;
        std::string e;
        C.connect(Endpoint{"127.0.0.1", port}, 3, 4, sec(), H, e);
        kill(pid, SIGKILL);
        int st = 0;
        waitpid(pid, &st, 0);
        std::mt19937 rng(5);
        Window w = make_window(rng, 2, 10, n_expert, 1.0f);
        std::vector<int32_t> kind(20, -1);
        std::vector<float> out(20 * H);
        const bool b = C.begin(3, w.x.data(), w.ids.data(), 2, 10, kind.data(), e);
        const bool f = C.finish(out.data(), e);
        check("the server died: the window fails (or nothing was sent) and the client stops serving", b && (!f || kind[0] == -1) && C.broken() && !C.serves(3), e);
        std::vector<int32_t> kind2(20, -1);
        C.begin(3, w.x.data(), w.ids.data(), 2, 10, kind2.data(), e);
        check("... the next window stays on the local pool", kind2[0] == -1 && !C.pending());
    }

    // ---- the real pack
    const char* pack = std::getenv("STRATA_TEST_PACK");
    if (pack == nullptr || access((std::string(pack) + "/experts.bin").c_str(), R_OK) != 0) {
        std::printf("skip the real pack (set STRATA_TEST_PACK to a pack dir with experts.bin)\n");
    } else {
        std::string e;
        if (!kernels::cpu::expert_layout_load(pack, 48, 512, e)) {
            check("the real pack's layout loads", false, e);
        } else {
            const auto& lay = kernels::cpu::expert_layout();
            const int fd = open((std::string(pack) + "/experts.bin").c_str(), O_RDONLY);
            struct stat st {};
            fstat(fd, &st);
            const uint8_t* base = (const uint8_t*) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            const BlobFn real = [&](int64_t l, int64_t ex) -> const uint8_t* {
                return l >= 0 && l < 48 && ex >= 0 && ex < 512 ? base + lay.blob_offset(l, ex) : nullptr;
            };
            std::printf("     the real pack: %s (%s layout, %llu bytes per layer-0 blob)\n", pack, lay.native ? "native" : "canonical",
                        (unsigned long long) lay.blob_bytes(0));
            exact_run("the real pack, layers 0 and 47", real, 0, 47, 512, 0.05f, 12);
            reduce_run(real, 5, 512);
            munmap((void*) base, (size_t) st.st_size);
            close(fd);
        }
    }
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
