// libimpulse_fx_mock: a stand-in for libimpulse_fx (flashnext/src/program/fx_bridge.cpp) that speaks the engine's serve
// protocol without a GPU or a model, for impulse-server's fx tests (tests/fx_server_test.py).  One Engine P1.
//
//   READY 4096 stop after IMPULSE_FX_MOCK_DELAY_MS (default 0), with an INFO line first
//   GEN <max_new> [key=value ...] <ids>   -> RESUME 0, a PP line, then the reply's ids as T lines (at most max_new),
//                                            DONE ... stop (the reply ended on an EOS id) or length (max_new reached)
//   STOP                                 -> the running reply ends with DONE ... cancel
//   QUIT / the input closed              -> returns 0
//   with IMPULSE_FX_MOCK_BATCH=N (--batch N's slots, impulse-fx's "parallel"): INFO batch_slots=N ... before READY, and
//   BGEN <slot> <max_new> [keys] <ids>  -> the reply's first id as T, DONE, BADM <slot> 1 (0: one token / an EOS), then
//                                          the rest as BT <slot> <id> lines (5 ms apart) and BDONE <slot> <n> <finish> 1.0
//   BSTOP <slot>                         -> that slot ends with BDONE ... cancel; a busy slot's BGEN -> ERR BGEN: ...
// The reply is the comma-separated ids in $IMPULSE_FX_MOCK_DIR/reply.txt, read at each GEN (several lines: the k-th GEN
// since the file was written gets line k, the last one repeating); every line received is appended to
// $IMPULSE_FX_MOCK_DIR/requests.log (the test checks the GEN lines impulse-server sent).  GENI (pictures: an embeddings
// file before the ids) is answered as GEN, with the file's size logged ("GENI-FILE <bytes>").  One line on stderr at the
// start ("fx mock engine: loaded (its stderr)"), as the engine narrates its load there.
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#define MOCK_API extern "C" __declspec(dllexport)
#else
#include <sys/select.h>
#include <unistd.h>
#define MOCK_API extern "C" __attribute__((visibility("default")))
#endif

static std::string dir() { const char * d = std::getenv("IMPULSE_FX_MOCK_DIR"); return d ? d : "."; }

static std::vector<long long> reply_ids(long long turn) {
    std::ifstream f(dir() + "/reply.txt");
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) if (!l.empty()) lines.push_back(l);
    std::vector<long long> ids;
    if (lines.empty()) return ids;
    std::istringstream is(lines[(size_t) std::min<long long>(turn, (long long) lines.size() - 1)]);
    std::string tok;
    while (std::getline(is, tok, ',')) if (!tok.empty()) ids.push_back(std::atoll(tok.c_str()));
    return ids;
}

MOCK_API int impulse_fx_abi(void) { return 1; }

MOCK_API int impulse_fx_run(int argc, char ** argv, std::intptr_t in_handle, std::intptr_t out_handle) {
#if defined(_WIN32)
    const int in_fd = _open_osfhandle(in_handle, _O_RDONLY | _O_BINARY);
    FILE * out = _fdopen(_open_osfhandle(out_handle, _O_WRONLY | _O_BINARY), "wb");
#else
    const int in_fd = (int) in_handle;
    FILE * out = fdopen((int) out_handle, "w");
#endif
    std::setvbuf(out, nullptr, _IONBF, 0);
    {
        std::ofstream log(dir() + "/requests.log", std::ios::app);
        log << "ARGS";
        for (int i = 0; i < argc; ++i) log << ' ' << argv[i];
        log << '\n';
    }
    {   // what CUDA would read when the engine starts it (impulse-server sets it before anything touches CUDA)
        const char * ml = std::getenv("CUDA_MODULE_LOADING");
        std::ofstream(dir() + "/env.txt") << "CUDA_MODULE_LOADING=" << (ml ? ml : "") << '\n';
    }
    if (const char * d = std::getenv("IMPULSE_FX_MOCK_DELAY_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(d)));
    std::fprintf(stderr, "fx mock engine: loaded (its stderr)\n");   // the engine's narration (the run config's "log")
    std::fflush(stderr);
    std::fprintf(out, "INFO engine=mock argc=%d\n", argc);
    const int batch = std::getenv("IMPULSE_FX_MOCK_BATCH") ? std::atoi(std::getenv("IMPULSE_FX_MOCK_BATCH")) : 0;
    if (batch > 0)
        std::fprintf(out, "INFO expert_slots=1000 expert_cache_mib=7000 batch_slots=%d slot_cache=0 batch_groups=1 batch_ctx=0 batch_slot_mib=512\n", batch);
    std::fprintf(out, "READY 4096 stop\n");
    std::array<std::atomic<bool>, 16> slot_stop{}, slot_busy{};
    std::vector<std::thread> slot_threads;
    std::atomic<bool> stop{ false };
    std::string buf, line;
    char chunk[4096];
    auto getline_fd = [&](std::string & o) -> bool {
        for (;;) {
            const size_t nl = buf.find('\n');
            if (nl != std::string::npos) { o.assign(buf, 0, nl); buf.erase(0, nl + 1); return true; }
#if defined(_WIN32)
            const int n = _read(in_fd, chunk, (unsigned) sizeof chunk);
#else
            const ssize_t n = ::read(in_fd, chunk, sizeof chunk);
#endif
            if (n <= 0) return false;
            buf.append(chunk, (size_t) n);
        }
    };
    // a GEN runs here, line by line; a STOP read meanwhile (the same reader: the protocol is line-ordered) ends it
    std::vector<std::string> pending;
    while (true) {
        if (pending.empty()) { if (!getline_fd(line)) break; } else { line = pending.front(); pending.erase(pending.begin()); }
        { std::ofstream log(dir() + "/requests.log", std::ios::app); log << line << '\n'; }
        if (line == "QUIT") break;
        if (line == "STOP") continue;   // between requests: stale (the engine resets it at the next request)
        if (batch > 0 && line.rfind("BSTOP ", 0) == 0) { const int b = std::atoi(line.c_str() + 6); if (b >= 0 && b < batch) slot_stop[(size_t) b] = true; continue; }
        if (batch > 0 && line.rfind("BGEN ", 0) == 0) {
            std::istringstream is(line.substr(5));
            int b = -1; long long max_new = 0;
            is >> b >> max_new;
            if (b < 0 || b >= batch || slot_busy[(size_t) b] || max_new < 1) { std::fprintf(out, "ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", batch); continue; }
            static long long bturn = 0;
            auto reply = reply_ids(bturn++);
            {   // a promoted request (its prompt ends with what it generated so far): the reply goes on from there, as the
                // engine's continues from its prompt cache
                std::string w, last;
                while (is >> w) last = w;
                std::vector<long long> p;
                std::istringstream ps(last);
                for (std::string t; std::getline(ps, t, ',');) if (!t.empty()) p.push_back(std::atoll(t.c_str()));
                for (size_t k = std::min(reply.size() - (reply.empty() ? 0 : 1), p.size()); k > 0; --k)
                    if (std::equal(reply.begin(), reply.begin() + (long) k, p.end() - (long) k)) { reply.erase(reply.begin(), reply.begin() + (long) k); break; }
            }
            const long long eos = std::getenv("IMPULSE_FX_MOCK_EOS") ? std::atoll(std::getenv("IMPULSE_FX_MOCK_EOS")) : -1;
            if (reply.empty()) { std::fprintf(out, "DONE 0 1 1.0 0.5 stop 0 0 0 0 0 0 0 0.0 1\nBADM %d 0\n", b); continue; }
            std::fprintf(out, "T %lld\n", reply[0]);
            const bool cont = max_new > 1 && reply.size() > 1 && reply[0] != eos;
            std::fprintf(out, "DONE 1 1 1.0 0.5 %s 0 0 0 0 0 0 0 0.0 1\nBADM %d %d\n", cont ? "length" : "stop", b, cont ? 1 : 0);
            if (!cont) continue;
            slot_busy[(size_t) b] = true;
            slot_stop[(size_t) b] = false;
            slot_threads.emplace_back([&, b, reply, max_new, eos]() {
                long long n = 0;
                const char * fin = "length";
                for (size_t i = 1; i < reply.size() && n + 1 < max_new; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    if (slot_stop[(size_t) b]) { fin = "cancel"; break; }
                    std::fprintf(out, "BT %d %lld\n", b, reply[i]);
                    ++n;
                    if (reply[i] == eos || i + 1 == reply.size()) { fin = "stop"; break; }
                }
                slot_busy[(size_t) b] = false;
                std::fprintf(out, "BDONE %d %lld %s 1.0\n", b, n, fin);
            });
            continue;
        }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (line.rfind("GEN ", 0) != 0 && !geni) { std::fprintf(out, "ERR expected: GEN <max_new> <id,id,...>\n"); continue; }
        std::istringstream is(line.substr(geni ? 5 : 4));
        long long max_new = 0;
        is >> max_new;
        std::string w, ids, prev;
        while (is >> w) { prev = ids; ids = w; }   // the last field: the ids (GENI: the embeddings file before them)
        if (geni) {
            std::ifstream ef(prev, std::ios::binary | std::ios::ate);
            std::ofstream log(dir() + "/requests.log", std::ios::app);
            log << "GENI-FILE " << (ef ? (long long) ef.tellg() : -1LL) << '\n';
        }
        // the reply's turn: GENs since reply.txt was last written
        static std::string stamp;
        static long long turn = 0;
        {
            std::ifstream f(dir() + "/reply.txt", std::ios::binary);
            std::ostringstream ss; ss << f.rdbuf();
            if (ss.str() != stamp) { stamp = ss.str(); turn = 0; }
        }
        long long n_prompt = 1;
        for (char c : ids) n_prompt += c == ',';
        const auto reply = reply_ids(turn++);
        if (std::getenv("IMPULSE_FX_MOCK_HANG")) { std::this_thread::sleep_for(std::chrono::hours(1)); }   // a silent engine (#481)
        std::fprintf(out, "RESUME 0\n");
        std::fprintf(out, "PP %lld %lld 1.0 1000.0\n", n_prompt, n_prompt);
        const char * finish = "length";
        long long n = 0;
        for (size_t i = 0; i < reply.size() && n < max_new; ++i) {
            std::fprintf(out, "T %lld\n", reply[i]);
            ++n;
            if (const char * eos = std::getenv("IMPULSE_FX_MOCK_EOS"); eos && std::atoll(eos) == reply[i]) { finish = "stop"; break; }
            // a STOP the server sent after this token: read what is there without waiting (the server sends it at once)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
#if !defined(_WIN32)
            fd_set rs; FD_ZERO(&rs); FD_SET(in_fd, &rs); timeval tv{ 0, 20000 };
            if (select(in_fd + 1, &rs, nullptr, nullptr, &tv) > 0) {
                std::string l;
                if (getline_fd(l)) {
                    { std::ofstream log(dir() + "/requests.log", std::ios::app); log << l << '\n'; }
                    if (l == "STOP") { finish = "cancel"; break; }
                    pending.push_back(l);
                }
            }
#endif
            if (i + 1 == reply.size()) finish = "stop";
        }
        std::fprintf(out, "DONE %lld %lld 1.0 2.0 %s 0 0 0 0 0 0 0 0.0 %lld\n", n, n_prompt, finish, n_prompt);
    }
    for (auto & t : slot_threads) t.join();
    std::fclose(out);
    return 0;
}
