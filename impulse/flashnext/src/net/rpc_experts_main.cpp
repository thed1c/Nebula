// strata-rpc-experts: the remote expert tier's server (net/expert_rpc.hpp).  A CPU-only process on another host: it maps a
// pack's experts.bin (the same pack the engine runs; only the served layers' pages are ever read), loads its expert layout,
// and computes the routed experts of its layers for an engine started with --rpc-experts host:port@A-B.
//
//   strata-rpc-experts --pack <pack dir> --listen host:port [--layers A-B] [--secret-file F | --plaintext]
//                      [--workers N] [--n-layers 48] [--n-expert 512] [--reduce-ok] [--once]
//
// The secret (>= 16 bytes, e.g. `openssl rand -hex 32`) is the same file on both hosts; --plaintext on both ends for an
// unencrypted link (a direct cable you trust).  One engine at a time; the next one may connect when it leaves.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/net/expert_rpc.hpp"
#include "strata/net/transport.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata;

static int usage() {
    std::fprintf(stderr, "strata-rpc-experts --pack <dir> --listen host:port [--layers A-B] [--secret-file F | --plaintext]\n"
                         "                   [--workers N] [--n-layers 48] [--n-expert 512] [--reduce-ok] [--once]\n"
                         "                   [--rdma [--rdma-device mlx4_0]]   (the engine: --rpc-transport rdma)\n");
    return 2;
}

int main(int argc, char** argv) {
    std::string pack, listen, secret_file, layers;
    bool plaintext = false, once = false, reduce_ok = false, rdma = false;
    net::RdmaOptions ropt;
    int workers = 0;
    int64_t n_layers = 48, n_expert = 512;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--pack") pack = next();
        else if (a == "--listen") listen = next();
        else if (a == "--layers") layers = next();
        else if (a == "--secret-file") secret_file = next();
        else if (a == "--plaintext") plaintext = true;
        else if (a == "--workers") workers = std::atoi(next().c_str());
        else if (a == "--n-layers") n_layers = std::atoll(next().c_str());
        else if (a == "--n-expert") n_expert = std::atoll(next().c_str());
        else if (a == "--reduce-ok") reduce_ok = true;
        else if (a == "--once") once = true;
        else if (a == "--rdma") rdma = true;
        else if (a == "--rdma-device") { rdma = true; ropt.device = next(); }
        else return usage();
    }
    if (pack.empty() || listen.empty()) return usage();
    std::signal(SIGPIPE, SIG_IGN);
    std::string err;
    net::Security sec;
    if (!net::Security::load(secret_file, sec, err)) { std::fprintf(stderr, "strata-rpc-experts: %s\n", err.c_str()); return 2; }
    sec.plaintext = plaintext;
    if (!sec.usable(err)) { std::fprintf(stderr, "strata-rpc-experts: %s\n", err.c_str()); return 2; }
    int64_t lo = 0, hi = n_layers - 1;
    if (!layers.empty()) {
        net::Endpoint dummy;
        if (!net::ExpertRpcClient::parse("x:0@" + layers, dummy, lo, hi, err) || hi >= n_layers) {
            std::fprintf(stderr, "strata-rpc-experts: --layers A-B within 0-%lld\n", (long long) (n_layers - 1));
            return 2;
        }
    }
    if (!kernels::cpu::expert_layout_load(pack, n_layers, n_expert, err)) {
        std::fprintf(stderr, "strata-rpc-experts: the pack's expert layout: %s\n", err.c_str());
        return 1;
    }
    const kernels::cpu::ExpertLayout& lay = kernels::cpu::expert_layout();
    const std::string file = pack + "/experts.bin";
    const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st {};
    if (fd < 0 || fstat(fd, &st) != 0) { std::fprintf(stderr, "strata-rpc-experts: cannot open %s\n", file.c_str()); return 1; }
    if ((uint64_t) st.st_size < lay.blob_offset(hi, n_expert - 1) + lay.blob_bytes(hi)) {
        std::fprintf(stderr, "strata-rpc-experts: %s is shorter than its layout says\n", file.c_str());
        return 1;
    }
    const uint8_t* base = (const uint8_t*) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::fprintf(stderr, "strata-rpc-experts: cannot map %s\n", file.c_str()); return 1; }
    // only the served layers are read: tell the kernel so (and nothing else of the 60 GB file is paged in)
    madvise((void*) base, (size_t) st.st_size, MADV_RANDOM);
    madvise((void*) (base + lay.layer_offset(lo)), (size_t) (lay.layer_offset(hi) + (uint64_t) n_expert * lay.blob_bytes(hi) - lay.layer_offset(lo)),
            MADV_WILLNEED);
    auto blob = [&](int64_t l, int64_t e) -> const uint8_t* {
        if (l < lo || l > hi || e < 0 || e >= n_expert) return nullptr;
        return base + lay.blob_offset(l, e);
    };
    kernels::cpu::ExpertPool pool(workers, /*pin=*/false, /*host_works=*/true);
    net::Endpoint ep;
    if (!net::Endpoint::parse(listen, ep, err)) { std::fprintf(stderr, "strata-rpc-experts: --listen: %s\n", err.c_str()); return 2; }
    net::Listener L;
    if (!L.open(ep, err)) { std::fprintf(stderr, "strata-rpc-experts: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "strata-rpc-experts: layers %lld-%lld of %s (%s pack), %d pool threads, listening on %s:%d (%s)\n",
                 (long long) lo, (long long) hi, pack.c_str(), lay.native ? "native" : "canonical",
                 pool.workers() + (pool.host_works() ? 1 : 0), ep.host.c_str(), L.port(),
                 sec.plaintext ? (sec.secret.empty() ? "plaintext, unauthenticated" : "plaintext, authenticated") : "aes-256-gcm, authenticated");
    if (rdma) std::fprintf(stderr, "strata-rpc-experts: RDMA after the TCP handshake (%s)\n", net::rdma_devices().c_str());
    std::printf("LISTENING %d\n", L.port());
    std::fflush(stdout);
    std::vector<float> rows;
    const net::ExpertCompute compute = [&](int64_t layer, const float* x, int n_tok, const net::ExpertEntry* ents, int n_ent,
                                           uint32_t mode, float* out, std::string& e) -> bool {
        if (mode == net::kRows) return net::compute_expert_rows(pool, lay, blob, layer, x, n_tok, ents, n_ent, out, e);
        if (mode != net::kReduce || !reduce_ok) { e = "this server does not reduce (start it with --reduce-ok)"; return false; }
        rows.assign((size_t) n_ent * kernels::cpu::H, 0.0f);
        if (!net::compute_expert_rows(pool, lay, blob, layer, x, n_tok, ents, n_ent, rows.data(), e)) return false;
        for (int i = 0; i < n_ent; ++i)
            for (int64_t c = 0; c < kernels::cpu::H; ++c)
                out[(size_t) ents[i].t * kernels::cpu::H + (size_t) c] += ents[i].w * rows[(size_t) i * kernels::cpu::H + (size_t) c];
        return true;
    };
    for (;;) {
        auto ch = L.accept(sec, "expert-server", err);
        if (ch && rdma) ch = net::rdma_upgrade(std::move(ch), true, sec, ropt, err);
        if (!ch) { std::fprintf(stderr, "strata-rpc-experts: a connection was refused: %s\n", err.c_str()); continue; }
        std::fprintf(stderr, "strata-rpc-experts: %s connected (%s)\n", ch->peer_role().c_str(), ch->describe().c_str());
        int64_t served = 0;
        if (!net::serve_expert_connection(*ch, kernels::cpu::H, lo, hi, compute, err, &served))
            std::fprintf(stderr, "strata-rpc-experts: the connection ended: %s\n", err.c_str());
        std::fprintf(stderr, "strata-rpc-experts: %lld requests served; waiting for the next engine\n", (long long) served);
        if (once) break;
    }
    return 0;
}
