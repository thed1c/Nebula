// Impulse: the RAM tier. The experts a 12 GB card does not hold live in system RAM: locked (page-locked, the
// arena the copy stream reads, so swaps run in the background) up to a budget, and the rest in the model file's
// mapping, which the OS keeps in its file cache while it has room and reads from the disk (an NVMe drive) when not.
// Whole expert tensors (a layer's gate/up or down) are the unit, in the file's layer order. The bytes are the same
// either way, so the arithmetic does not change; only where a miss or a swap reads from does.
//
// Budget: --ram-budget GiB; 0 = low-RAM mode (nothing locked, swaps in the step); default (-1): what the OS can give
// now less a headroom for the OS and other programs (Linux max(4 GiB, 15 % of RAM), Windows max(8 GiB, 25 %)).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include "ggml.h"
#include "platform.hpp"
#include "numa_place.hpp"

namespace lite {

inline size_t ram_lock_budget(double budget_gib, size_t want) {
    if (budget_gib >= 0) return std::min(want, (size_t) (budget_gib * 1073741824.0));
    const double avail = available_ram_bytes();
    if (avail <= 0) return want;   // unknown: lock everything, as before the tier (a failed lock still falls back)
    double total = total_ram_gb() * 1e9;   // the PC's RAM, or the cgroup limit when the process runs under a smaller one
    if (const double cap = cgroup_limit_bytes(); cap > 0 && (total == 0 || cap < total)) total = cap;
#ifdef _WIN32
    const double headroom = std::max(8.0 * 1073741824.0, 0.25 * total);
#else
    const double headroom = std::max(4.0 * 1073741824.0, 0.15 * total);
#endif
    return avail > headroom ? std::min(want, (size_t) (avail - headroom)) : 0;
}

struct RamTier {
    void * arena = nullptr;
    size_t arena_size = 0, want = 0;
    bool placed = false;   // ram_tier_place: an anonymous mapping, filled and registered (ram_tier_free: unregister + munmap)
    std::unordered_set<const ggml_tensor *> locked;
    bool in_arena(const ggml_tensor * t) const { return locked.count(t) != 0; }
};

// Picks and locks: the first expert tensors (`_exps.` in the name) that fit the budget; a lock the OS refuses is
// tried again at half the size, down to 1 GiB, then nothing is locked (low-RAM mode). on_cpu: (tensor, file offset).
inline RamTier ram_tier_lock(const std::vector<std::pair<ggml_tensor *, size_t>> & on_cpu, double budget_gib, size_t align) {
    RamTier r;
    auto pad = [&](size_t n) { return (n + align - 1) / align * align; };
    auto is_exps = [](const ggml_tensor * t) { return strstr(ggml_get_name(t), "_exps.") != nullptr; };
    for (auto & [t, off] : on_cpu) if (is_exps(t)) r.want += pad(ggml_nbytes(t));
    size_t budget = ram_lock_budget(budget_gib, r.want);
    while (budget > 0) {
        r.locked.clear(); r.arena_size = 0;
        for (auto & [t, off] : on_cpu)
            if (is_exps(t) && r.arena_size + pad(ggml_nbytes(t)) <= budget) { r.locked.insert(t); r.arena_size += pad(ggml_nbytes(t)); }
        if (r.arena_size == 0) break;
        if (cudaHostAlloc(&r.arena, r.arena_size, cudaHostAllocPortable) == cudaSuccess) break;
        cudaGetLastError();
        r.arena = nullptr;
        fprintf(stderr, "impulse: the OS would not lock %.1f GB of RAM; trying half\n", r.arena_size / 1e9);
        budget = r.arena_size / 2;
        if (budget < ((size_t) 1 << 30)) budget = 0;
    }
    if (!r.arena) { r.locked.clear(); r.arena_size = 0; }
    if (r.arena_size == 0)
        fprintf(stderr, "impulse: low-RAM mode: the experts (%.1f GB) stay in the model file and are read in as needed (slower)\n",
                r.want / 1e9);
    else if (r.arena_size < r.want)
        fprintf(stderr, "impulse: RAM tier: %.1f of the experts' %.1f GB locked in RAM; the rest is read from the model file "
                        "(the OS file cache, else the disk)\n", r.arena_size / 1e9, r.want / 1e9);
    return r;
}

// Impulse (numa-always): the RAM tier with the experts placed per socket under the pool's NUMA split (numa_place.hpp).
// The same choice of tensors and the same layout as ram_tier_lock (on_cpu order, `align` apart), but the arena is anonymous
// memory filled by the per-node copiers from the mapping `map` (on_cpu's offsets) and registered afterwards
// (cudaHostRegister pins the pages where they are; cudaHostAlloc would fault them all on this thread's node).  An empty
// result (with a line saying why) leaves the caller on ram_tier_lock; nothing is placed twice.
inline RamTier ram_tier_place(const std::vector<std::pair<ggml_tensor *, size_t>> & on_cpu, double budget_gib, size_t align,
                              const void * map, const std::vector<std::string> & paths,
                              const strata::kernels::cpu::NumaSplitPlan & plan) {
    RamTier r;
#if defined(__linux__)
    auto pad = [&](size_t n) { return (n + align - 1) / align * align; };
    auto is_exps = [](const ggml_tensor * t) { return strstr(ggml_get_name(t), "_exps.") != nullptr; };
    for (auto & [t, off] : on_cpu) if (is_exps(t)) r.want += pad(ggml_nbytes(t));
    const size_t budget = ram_lock_budget(budget_gib, r.want);
    for (auto & [t, off] : on_cpu)
        if (is_exps(t) && r.arena_size + pad(ggml_nbytes(t)) <= budget) { r.locked.insert(t); r.arena_size += pad(ggml_nbytes(t)); }
    auto give_up = [&](const std::string & why) {
        fprintf(stderr, "impulse: NUMA placement not used: %s; the experts are locked as before (not placed per node)\n", why.c_str());
        if (r.arena) munmap(r.arena, r.arena_size);
        return RamTier{};
    };
    if (r.arena_size == 0) return give_up("no expert fits the RAM budget");
    void * p = mmap(nullptr, r.arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return give_up("mmap of " + std::to_string(r.arena_size) + " bytes failed");
    r.arena = p;
    std::vector<PlaceTensor> ts;
    size_t aoff = 0;
    for (auto & [t, off] : on_cpu) {   // ram_tier_lock's layout, which the engine's copy loop walks again
        if (!r.in_arena(t)) continue;
        PlaceTensor pt;
        pt.dst = (uint8_t *) p + aoff; pt.src = (const uint8_t *) map + off; pt.bytes = ggml_nbytes(t);
        pt.n_expert = t->ne[2]; pt.expert_stride = t->nb[2]; pt.rows = t->ne[1]; pt.row_bytes = t->nb[1];
        pt.parts = strstr(ggml_get_name(t), "gate_up_exps") != nullptr ? 2 : 1;   // gemma4: gate rows, then up rows
        ts.push_back(pt);
        aoff += pad(ggml_nbytes(t));
    }
    numa_drop_file_cache(paths);
    const PlaceReport rep = numa_place_copy(ts, plan, (uint8_t *) p, r.arena_size);
    if (!rep.done) return give_up(rep.why);
    if (cudaHostRegister(p, r.arena_size, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        return give_up("the OS would not lock " + std::to_string(r.arena_size / 1000000000.0) + " GB of RAM");
    }
    r.placed = true;
    // the copied ranges leave this process's mapping and the page cache: the arena holds them now
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    for (const auto & t : ts) {
        const uintptr_t a = ((uintptr_t) t.src + page - 1) / page * page, b = ((uintptr_t) t.src + t.bytes) / page * page;
        if (b > a) (void) madvise((void *) a, b - a, MADV_DONTNEED);
    }
    numa_drop_file_cache(paths);
    fprintf(stderr, "%s\n", numa_place_line(rep).c_str());
    uint64_t got[2];
    const std::string check = numa_place_check(p, r.arena_size, rep, got);
    fprintf(stderr, "%s\n", check.empty() ? "impulse: NUMA placement check: /proc/self/numa_maps not readable" : check.c_str());
    if (r.arena_size < r.want)
        fprintf(stderr, "impulse: RAM tier: %.1f of the experts' %.1f GB locked in RAM; the rest is read from the model file "
                        "(the OS file cache, else the disk)\n", r.arena_size / 1e9, r.want / 1e9);
#else
    (void) on_cpu; (void) budget_gib; (void) align; (void) map; (void) paths; (void) plan;
#endif
    return r;
}

// Frees a tier's arena either way it was made.
inline void ram_tier_free(void * arena, size_t size, bool placed) {
    if (!arena) return;
#if defined(__linux__)
    if (placed) { cudaHostUnregister(arena); munmap(arena, size); return; }
#else
    (void) size; (void) placed;
#endif
    cudaFreeHost(arena);
}

}  // namespace lite
