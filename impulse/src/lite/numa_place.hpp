// Impulse (numa-always): the CPU-resident experts placed per socket under the CPU pool's NUMA split.
//
// The split (flashnext/src/kernels/cpu/pool.cpp, numa_plan_phase) gives slice k of the pool - the threads on node nodes[k] - rows
// [per*k/2, per*(k+1)/2) of every expert: half of each expert's gate and up rows (per = n_ff) and half of its down rows
// (per = n_embd).  A GGUF expert tensor is [cols, rows, n_expert] with each expert's rows back to back, so every
// (expert, slice) part is ONE contiguous byte range: the experts can be placed row slice by row slice, exactly as the pool
// reads them (no whole-expert or slice-majority fallback is needed).  Only the page at the border between two parts is
// shared; whichever slice copies it first owns it.
//
// The placement copies those ranges from the model file into an anonymous arena, each range by a thread pinned to its
// slice's cores (Linux first touch: a page lands on the node of the thread that first writes it; 4 KiB pages, transparent
// huge pages off for the arena so a 2 MiB page cannot pull both slices onto one node).  As in the strata-nvfp4 fork
// (6d63fe1e, 9c2ed715): the model files leave the page cache first (posix_fadvise DONTNEED; a node full of file cache
// gives a toucher's pages to the other node, zone_reclaim_mode 0), and each copier binds its allocations to its node
// (set_mempolicy MPOL_BIND, so the kernel reclaims that node's cache instead) when the node's free + file pages hold its
// share plus 4 GiB; a sandbox that refuses the call gets plain first touch.  numa_place_check then reads
// /proc/self/numa_maps (readable inside a sandbox) for the arena's pages per node.  The bytes are the model's, unchanged:
// the arithmetic does not move (tests/numa_place_test.cpp).
//
// Linux only; elsewhere numa_place_copy reports "not on this OS" and the caller keeps its unplaced path.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/cpu/numa_pool.hpp"
#include "strata/kernels/cpu/pool.hpp"

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace lite {

// One expert tensor to place: n_expert experts `expert_stride` bytes apart, each `rows` rows of `row_bytes`, in `parts`
// matrices the pool splits on their own (2 for gemma4's fused gate_up: the gate rows, then the up rows; else 1).
struct PlaceTensor {
    uint8_t * dst = nullptr;            // in the arena
    const uint8_t * src = nullptr;      // in the model file's mapping
    size_t bytes = 0;                   // the whole tensor (ggml_nbytes)
    int64_t n_expert = 0, rows = 0;
    size_t expert_stride = 0, row_bytes = 0;
    int parts = 1;
};

// Slice k's byte ranges of a tensor, [first, second) from its start: the pool's row halves of every expert's matrices
// (per = rows / parts: rows [per*k/2, per*(k+1)/2) of each, as numa_plan_phase).  Bytes past the last expert (none in a
// GGUF tensor) go to slice 0.
inline std::vector<std::pair<size_t, size_t>> numa_slice_ranges(const PlaceTensor & t, int k) {
    std::vector<std::pair<size_t, size_t>> r;
    const int parts = t.parts > 0 ? t.parts : 1;
    const int64_t per = t.rows / parts, lo = per * k / 2, hi = per * (k + 1) / 2;
    for (int64_t e = 0; e < t.n_expert; ++e)
        for (int p = 0; p < parts; ++p) {
            const int64_t r0 = (int64_t) p * per + lo, r1 = (int64_t) p * per + (p == parts - 1 && k == 1 ? t.rows - (int64_t) p * per : hi);
            const size_t a = (size_t) e * t.expert_stride + (size_t) r0 * t.row_bytes;
            const size_t b = (size_t) e * t.expert_stride + (size_t) r1 * t.row_bytes;
            if (b > a) r.push_back({ a, std::min(b, t.bytes) });
        }
    const size_t end = (size_t) t.n_expert * t.expert_stride;
    if (k == 0 && end < t.bytes) r.push_back({ end, t.bytes });
    return r;
}

struct PlaceReport {
    bool done = false;
    std::string why;                    // when not done
    uint64_t bytes[2] = { 0, 0 };       // copied by each slice
    int nodes[2] = { -1, -1 };
    int threads[2] = { 0, 0 };
    int bound = 0, refused = 0, unbound = 0;   // MPOL_BIND taken / refused (a sandbox) / not tried (node room unknown or short)
    bool pin_failed = false;
    double seconds = 0;
};

#if defined(__linux__)
// free + file-backed pages of a node in bytes (0: unknown, e.g. a node IMPULSE_NUMA_NODES made up)
inline uint64_t numa_node_room(int node) {
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo");
    std::string line;
    uint64_t free_kb = 0, file_kb = 0;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string a, id, key;
        uint64_t kb = 0;
        if (!(is >> a >> id >> key >> kb)) continue;
        if (key == "MemFree:") free_kb = kb;
        else if (key == "FilePages:") file_kb = kb;
    }
    return (free_kb + file_kb) * 1024;
}
#endif

// The model files leave the page cache (pages this process still maps stay)
inline void numa_drop_file_cache(const std::vector<std::string> & paths) {
#if defined(__linux__)
    for (const auto & p : paths)
        if (int fd = ::open(p.c_str(), O_RDONLY); fd >= 0) { (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
#else
    (void) paths;
#endif
}

// Copies every tensor's slice-k ranges on up to `max_threads` threads per slice pinned to plan.cores[k].  `arena` /
// `arena_bytes` is the whole arena (transparent huge pages are turned off on it first).
inline PlaceReport numa_place_copy(const std::vector<PlaceTensor> & ts, const strata::kernels::cpu::NumaSplitPlan & plan,
                                   uint8_t * arena, size_t arena_bytes, int max_threads = 8) {
    PlaceReport rep;
#if defined(__linux__)
    if (!plan.on || plan.test) { rep.why = "the pool's NUMA split is off"; return rep; }
    std::vector<int> cores[2];
    for (int k = 0; k < 2; ++k)
        for (int c : plan.cores[k]) if (c >= 0) cores[k].push_back(c);
    if (cores[0].empty() || cores[1].empty()) { rep.why = "a slice has no pinned core"; return rep; }
    const auto t0 = std::chrono::steady_clock::now();
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if (madvise(arena, (arena_bytes + page - 1) / page * page, MADV_NOHUGEPAGE) != 0)
        fprintf(stderr, "impulse: NUMA placement: madvise(MADV_NOHUGEPAGE) failed (errno %d): 2 MiB pages may span slices\n", errno);
    uint64_t share[2] = { 0, 0 };
    for (const auto & t : ts) for (int k = 0; k < 2; ++k) for (const auto & r : numa_slice_ranges(t, k)) share[k] += r.second - r.first;
    std::atomic<uint64_t> copied[2] = { { 0 }, { 0 } };
    std::atomic<int> bound{ 0 }, refused{ 0 }, unbound{ 0 };
    std::atomic<bool> pin_failed{ false };
    std::vector<std::thread> th;
    for (int k = 0; k < 2; ++k) {
        const int nt = (int) std::min<size_t>(cores[k].size(), (size_t) std::max(1, max_threads));
        rep.threads[k] = nt;
        for (int j = 0; j < nt; ++j)
            th.emplace_back([&, k, j, nt] {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cores[k][(size_t) j], &set);
                if (sched_setaffinity(0, sizeof set, &set) != 0) pin_failed = true;
                const int node = plan.nodes[k];
                if (node >= 0 && node < 64 && numa_node_room(node) >= share[k] + (4ull << 30)) {
                    unsigned long mask = 1ul << node;
                    if (syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, &mask, 64) == 0) ++bound;
                    else ++refused;
                } else {
                    ++unbound;
                }
                uint64_t n = 0, item = 0;
                for (const auto & t : ts)
                    for (const auto & r : numa_slice_ranges(t, k))
                        if (item++ % (uint64_t) nt == (uint64_t) j) {
                            memcpy(t.dst + r.first, t.src + r.first, r.second - r.first);
                            n += r.second - r.first;
                        }
                copied[k] += n;
            });
    }
    for (auto & x : th) x.join();
    rep.done = true;
    for (int k = 0; k < 2; ++k) { rep.bytes[k] = copied[k].load(); rep.nodes[k] = plan.nodes[k]; }
    rep.bound = bound.load(); rep.refused = refused.load(); rep.unbound = unbound.load();
    rep.pin_failed = pin_failed.load();
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#else
    (void) ts; (void) plan; (void) arena; (void) arena_bytes; (void) max_threads;
    rep.why = "per-node placement needs Linux";
#endif
    return rep;
}

inline std::string numa_place_line(const PlaceReport & r) {
    char b[512];
    snprintf(b, sizeof b, "impulse: NUMA placement: slice 0 -> node %d (%.2f GiB, %d threads), slice 1 -> node %d (%.2f GiB, %d "
             "threads), copied in %.1f s; copiers bound to their node: %d (%d refused: plain first touch; %d not tried: node "
             "room unknown or short)%s", r.nodes[0], r.bytes[0] / 1073741824.0, r.threads[0], r.nodes[1],
             r.bytes[1] / 1073741824.0, r.threads[1], r.seconds, r.bound, r.refused, r.unbound,
             r.pin_failed ? " (WARNING: a copier could not be pinned: some pages may be on the wrong node)" : "");
    return b;
}

// The arena's pages per node from /proc/self/numa_maps, against what each slice copied.  `got` gets the bytes on
// nodes[0] / nodes[1]; returns the log line ("" when numa_maps cannot be read).
inline std::string numa_place_check(const void * base, size_t bytes, const PlaceReport & r, uint64_t got[2]) {
    got[0] = got[1] = 0;
#if defined(__linux__)
    std::ifstream f("/proc/self/numa_maps");
    if (!f) return "";
    const uintptr_t lo = (uintptr_t) base, hi = lo + bytes;
    uint64_t other = 0;
    std::string line;
    bool any = false;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string addr;
        is >> addr;
        const uintptr_t start = (uintptr_t) strtoull(addr.c_str(), nullptr, 16);
        if (start < lo || start >= hi) continue;
        any = true;
        uint64_t kb = 4;
        std::vector<std::pair<int, uint64_t>> pages;
        std::string tok;
        while (is >> tok) {
            if (tok.size() > 1 && tok[0] == 'N' && isdigit((unsigned char) tok[1])) {
                const size_t eq = tok.find('=');
                if (eq != std::string::npos) pages.push_back({ atoi(tok.c_str() + 1), strtoull(tok.c_str() + eq + 1, nullptr, 10) });
            } else if (tok.rfind("kernelpagesize_kB=", 0) == 0) {
                kb = strtoull(tok.c_str() + 18, nullptr, 10);
            }
        }
        for (auto [node, n] : pages) {
            const uint64_t b = n * kb * 1024;
            if (node == r.nodes[0]) got[0] += b;
            else if (node == r.nodes[1]) got[1] += b;
            else other += b;
        }
    }
    if (!any) return "";
    char out[512];
    const double p0 = r.bytes[0] ? 100.0 * (double) got[0] / (double) r.bytes[0] : 0.0;
    const double p1 = r.bytes[1] ? 100.0 * (double) got[1] / (double) r.bytes[1] : 0.0;
    snprintf(out, sizeof out, "impulse: NUMA placement check (numa_maps): node %d %.2f GiB, node %d %.2f GiB, other nodes %.2f GiB "
             "of the arena; slice 0 %.1f %%, slice 1 %.1f %% of its bytes on its node%s", r.nodes[0], got[0] / 1073741824.0,
             r.nodes[1], got[1] / 1073741824.0, other / 1073741824.0, p0, p1,
             p0 < 95.0 || p1 < 95.0 ? " (WARNING: a slice is short on its node; its rows are read across the socket link)" : "");
    return out;
#else
    (void) base; (void) bytes; (void) r;
    return "";
#endif
}

// Whether the experts are placed: IMPULSE_NUMA_PLACE=0 (or off/false/no) turns it off; otherwise on wherever the pool's
// split can run (two or more nodes); a one-socket machine never reaches the placement.
inline bool numa_place_wanted() {
    return !strata::kernels::cpu::numa_switch_off(std::getenv("IMPULSE_NUMA_PLACE")) && strata::kernels::cpu::numa_split_possible();
}

}  // namespace lite
