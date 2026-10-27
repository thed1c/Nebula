// numa-always: the per-socket placement of the CPU experts (src/lite/numa_place.hpp), no GPU, no model.
//
//     IMPULSE_NUMA_NODES="0=0-9,20-29;1=10-19,30-39" numa-place-test
//
// (1) the slice ranges of a tensor cover it exactly once, for separate matrices and gemma4's fused gate_up (odd row counts
// too); (2) with the pool's split on (a made-up layout from IMPULSE_NUMA_NODES, else this host's cores cut in two), NVFP4
// experts in the q35 engine's form (gate, up and down tensors) are copied into an anonymous arena by the per-node copiers,
// with the placement line and the numa_maps check; the arena equals the source byte for byte; and the pool's outputs with
// its jobs on the arena are bit for bit its outputs on the source; (3) IMPULSE_NUMA_PLACE=0 turns the placement off.
// On a made-up layout the second node does not exist, so every page lands on the real one and the check line warns: that
// is expected here; on a two-socket machine it shows each slice on its node.
#include "../src/lite/numa_place.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace cpu = strata::kernels::cpu;

static int cover_check() {
    int bad = 0;
    for (int parts : { 1, 2 })
        for (int64_t rows : { (int64_t) 512, (int64_t) 2048, (int64_t) 7, (int64_t) 1408 }) {
            if (rows % parts) continue;
            lite::PlaceTensor t;
            t.n_expert = 5; t.rows = rows; t.row_bytes = 36; t.expert_stride = (size_t) rows * 36; t.parts = parts;
            t.bytes = (size_t) t.n_expert * t.expert_stride;
            std::vector<int> owner(t.bytes, -1);
            for (int k = 0; k < 2; ++k)
                for (const auto & r : lite::numa_slice_ranges(t, k))
                    for (size_t b = r.first; b < r.second; ++b) { if (owner[b] != -1) ++bad; owner[b] = k; }
            for (size_t b = 0; b < t.bytes; ++b) {
                if (owner[b] < 0) { ++bad; continue; }
                // the slice the pool gives this row: row (b % stride) / row_bytes of matrix part, k = row-in-part >= per/2
                const int64_t row = (int64_t) ((b % t.expert_stride) / t.row_bytes), per = rows / parts, in = row % per;
                const int k = in >= per * 1 / 2 ? 1 : 0;
                if (owner[b] != k) ++bad;
            }
        }
    std::printf("numa_place: slice ranges cover every tensor once, as the pool's rows: %s\n", bad ? "FAILED" : "OK");
    return bad;
}

int main() {
#if !defined(__linux__)
    std::printf("numa_place: Linux only, skipped\n");
    return 0;
#else
    if (!cpu::cpu_avx2_ok()) { std::printf("numa_place: no AVX2, skipped\n"); return 0; }
    int bad = cover_check();
    if (!std::getenv("IMPULSE_NUMA_NODES")) {   // this host's cores cut in two, as pool_tasks_test does
        const std::vector<int> phys = cpu::physical_cores(false);
        if (phys.size() < 4) { std::printf("numa_place: fewer than 4 cores, the placement part skipped\n"); return bad ? 1 : 0; }
        std::string t[2];
        for (size_t i = 0; i < phys.size(); ++i) {
            std::string & s = t[i < phys.size() / 2 ? 0 : 1];
            s += (s.empty() ? "" : ",") + std::to_string(phys[i]);
        }
        setenv("IMPULSE_NUMA_NODES", ("0=" + t[0] + ";1=" + t[1]).c_str(), 1);
    }
    std::printf("numa_place: IMPULSE_NUMA_NODES=%s\n", std::getenv("IMPULSE_NUMA_NODES"));
    setenv("IMPULSE_NUMA_PLACE", "0", 1);
    if (lite::numa_place_wanted()) { std::printf("numa_place: IMPULSE_NUMA_PLACE=0 did not turn it off\n"); ++bad; }
    unsetenv("IMPULSE_NUMA_PLACE");
    if (!lite::numa_place_wanted()) { std::printf("numa_place: the split is not possible on this layout\n"); return 1; }

    // a q35-form layer: NVFP4 gate [n_embd, n_ff, E], up, down [n_ff, n_embd, E] back to back in one "file"
    const int n_embd = 2048, n_ff = 512, E = 24;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kNvfp4Type, cpu::kNvfp4Type, n_embd, n_ff, f, err, cpu::NativeAct::SwiGLU_ggml, true)) {
        std::printf("numa_place: %s\n", err.c_str());
        return 1;
    }
    const size_t g_stride = f.gu_row * (size_t) n_ff, d_stride = f.d_row * (size_t) n_embd;
    const size_t gbytes = g_stride * E, dbytes = d_stride * E, align = 4096;
    auto pad = [&](size_t n) { return (n + align - 1) / align * align; };
    std::vector<uint8_t> file(pad(gbytes) * 2 + pad(dbytes) + 1000);   // + 1000: offsets not page aligned in the "file"
    std::mt19937 rng(20261024);
    const size_t off[3] = { 100, 100 + gbytes + 36 * 3, 100 + 2 * gbytes + 36 * 7 };
    for (int i = 0; i < 3; ++i)   // NVFP4 blocks from each tensor's start: four small UE4M3 scales (the hidden's Q8_0
        for (size_t b = off[i]; b + 36 <= off[i] + (i < 2 ? gbytes : dbytes); b += 36) {   // scale stays an FP16), random codes
            for (int s = 0; s < 4; ++s) file[b + s] = (uint8_t) (0x10 + rng() % 0x28);
            for (int q = 4; q < 36; ++q) file[b + q] = (uint8_t) rng();
        }
    const size_t nb[3] = { gbytes, gbytes, dbytes };
    const int64_t rows[3] = { n_ff, n_ff, n_embd };
    const size_t rb[3] = { f.gu_row, f.gu_row, f.d_row }, st[3] = { g_stride, g_stride, d_stride };

    cpu::ExpertPool pool(0, /*pin*/ true);
    const cpu::NumaSplitPlan & plan = cpu::active_numa_split();
    if (!plan.on || plan.test) { std::printf("numa_place: the pool's split is off (%s)\n", plan.why.c_str()); return 1; }

    const size_t arena_size = pad(gbytes) * 2 + pad(dbytes);
    void * mem = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (mem == MAP_FAILED) return 1;
    uint8_t * arena = (uint8_t *) mem;
    std::vector<lite::PlaceTensor> ts;
    size_t aoff = 0;
    for (int i = 0; i < 3; ++i) {
        lite::PlaceTensor t;
        t.dst = arena + aoff; t.src = file.data() + off[i]; t.bytes = nb[i];
        t.n_expert = E; t.rows = rows[i]; t.row_bytes = rb[i]; t.expert_stride = st[i];
        ts.push_back(t);
        aoff += pad(nb[i]);
    }
    const lite::PlaceReport rep = lite::numa_place_copy(ts, plan, arena, arena_size);
    if (!rep.done) { std::printf("numa_place: not placed: %s\n", rep.why.c_str()); return 1; }
    std::printf("%s\n", lite::numa_place_line(rep).c_str());
    uint64_t got[2];
    const std::string check = lite::numa_place_check(arena, arena_size, rep, got);
    std::printf("%s\n", check.empty() ? "numa_place: numa_maps not readable" : check.c_str());
    for (int i = 0; i < 3; ++i)
        if (std::memcmp(ts[i].dst, ts[i].src, nb[i]) != 0) { std::printf("numa_place: tensor %d differs from its source\n", i); ++bad; }
    if (rep.bytes[0] + rep.bytes[1] != gbytes * 2 + dbytes) { std::printf("numa_place: copied %llu bytes, not %zu\n",
        (unsigned long long) (rep.bytes[0] + rep.bytes[1]), gbytes * 2 + dbytes); ++bad; }

    // the pool on the source and on the arena: every expert, 1..MAXT tokens, bit for bit
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<float> x(n_embd);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (float & v : x) v = nd(rng);
        cpu::native_quant_act(f, x.data(), act[t].data());
    }
    std::vector<float> out[2];
    for (int place = 0; place < 2; ++place) {
        const uint8_t * gate = place ? ts[0].dst : ts[0].src, * up = place ? ts[1].dst : ts[1].src, * dn = place ? ts[2].dst : ts[2].src;
        cpu::NativeFmt fp = f;
        fp.up_off = (size_t) (up - gate);   // may wrap, as the engine's: gate + up_off is the up rows
        std::vector<cpu::ExpertJobMulti> jobs(E);
        out[place].assign((size_t) E * cpu::MAXT * n_embd, std::numeric_limits<float>::quiet_NaN());
        for (int e = 0; e < E; ++e) {
            jobs[e].blob = gate + (size_t) e * g_stride;
            jobs[e].down = dn + (size_t) e * d_stride;
            jobs[e].nt = 1 + e % cpu::MAXT;
            jobs[e].s_gate = 0.75f; jobs[e].s_up = 1.25f; jobs[e].s_down = 0.5f;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = act[t].data();
                jobs[e].out[t] = out[place].data() + ((size_t) e * cpu::MAXT + t) * n_embd;
            }
        }
        for (int at = 0; at < E; at += cpu::ExpertPool::kMaxSplitMulti)
            pool.run_split_multi_native(fp, jobs.data() + at, std::min(E - at, cpu::ExpertPool::kMaxSplitMulti));
    }
    size_t differ = 0, compared = 0, finite = 0;
    for (size_t i = 0; i < out[0].size(); ++i) {
        if (std::isnan(out[0][i]) && std::isnan(out[1][i])) continue;   // unused token slots
        ++compared;
        differ += std::memcmp(&out[0][i], &out[1][i], sizeof(float)) != 0;
        finite += std::isfinite(out[0][i]);
    }
    std::printf("numa_place: pool outputs, placed arena against the source: %zu of %zu values differ (%zu finite)\n", differ,
                compared, finite);
    if (differ || finite != compared || compared == 0) ++bad;
    munmap(mem, arena_size);
    std::printf("numa_place_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
#endif
}
