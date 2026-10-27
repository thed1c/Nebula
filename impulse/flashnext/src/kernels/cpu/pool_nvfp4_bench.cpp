// pool_nvfp4_bench: how fast the CPU expert pool drains Flash-Next's routed experts (NVFP4, 2560 x 640) against what the
// machine's memory can deliver to the same threads.  Synthetic weights (no model), a working set far larger than the caches,
// a different set of experts every layer, as in decode.  numa-always: the pool reads IMPULSE_NUMA_NODES / sysfs / getcpu as
// in the engine, so the same binary measures the one-socket pool, the per-socket split, and STRATA_NUMA_POOL=0.
//
//   pool_nvfp4_bench [--workers N] [--experts-per-layer K] [--tokens T] [--gib G] [--layers L] [--no-pin]
// Prints: ms per layer, expert GB/s through the pool, and a plain read of the same bytes on the same threads (the ceiling).
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

int main(int argc, char** argv) {
    int workers = 0, k = 10, tokens = 1, layers = 400;
    double gib = 8.0;
    bool pin = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : (std::fprintf(stderr, "%s needs a value\n", a.c_str()), std::exit(2), ""); };
        if (a == "--workers") workers = std::atoi(next());
        else if (a == "--experts-per-layer") k = std::atoi(next());
        else if (a == "--tokens") tokens = std::atoi(next());
        else if (a == "--gib") gib = std::atof(next());
        else if (a == "--layers") layers = std::atoi(next());
        else if (a == "--no-pin") pin = false;
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    tokens = std::max(1, std::min(tokens, cpu::MAXT));
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kNvfp4Type, cpu::kNvfp4Type, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        return 1;
    }
    const size_t nblob = std::max<size_t>((size_t) k * 4, (size_t) (gib * 1073741824.0 / (double) f.bytes));
    std::printf("expert blob %zu bytes (NVFP4 %dx%d), %zu blobs = %.2f GiB, %d experts x %d token(s) per layer\n", f.bytes,
                (int) cpu::H, (int) cpu::FF, nblob, (double) (nblob * f.bytes) / 1073741824.0, k, tokens);

    // the pool first: its threads (pinned, maybe split per node) are the ones that touch the weights below
    cpu::ExpertPool pool(workers, pin, true, cpu::PoolAffinity::All, 0);
    const auto& plan = cpu::active_numa_split();
    std::printf("pool: %d workers + host, NUMA split %s; NVFP4 kernel: %s\n", pool.workers(), plan.on ? "on" : "off",
                std::getenv("STRATA_NO_NVFP4_512") == nullptr && cpu::cpu_avx512bw_ok()
                    ? (std::getenv("STRATA_NO_NVFP4_VNNI") ? "AVX-512BW (no VNNI)" : "AVX-512 (VNNI if the CPU has it)")
                : cpu::cpu_avx2_ok() && !std::getenv("STRATA_NO_NVFP4_256") ? "AVX2" : "ggml-cpu");

    // weights: random bytes with no FP8 NaN (0x7f / 0xff), the per-expert scales 1.0; filled in parallel so the pages are
    // spread the way a loader's threads would spread them (the split's placement is the engine's job, not measured here)
    std::vector<uint8_t*> blobs(nblob);
    for (auto& b : blobs) b = (uint8_t*) std::aligned_alloc(64, (f.bytes + 63) / 64 * 64);
    {
        const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&, t] {
                std::mt19937_64 rng(1234 + t);
                for (size_t i = t; i < nblob; i += nt) {
                    uint8_t* b = blobs[i];
                    for (size_t j = 0; j + 8 <= f.bytes; j += 8) {
                        uint64_t v = rng();
                        std::memcpy(b + j, &v, 8);
                    }
                    for (size_t j = 0; j < f.bytes; ++j)
                        if ((b[j] & 0x7f) == 0x7f) b[j] = 0x38;
                    if (f.tail_off) {
                        const float s[4] = {1.0f, 1.0f, 1.0f, 0.0f};
                        std::memcpy(b + f.tail_off, s, sizeof s);
                    }
                }
            });
        for (auto& t : th) t.join();
    }
    // activations
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<float> x(cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = 0.05f * std::sin((float) (i + 7 * t) * 0.013f);
        cpu::native_quant_act(f, x.data(), act[t].data());
    }
    std::vector<cpu::ExpertJobMulti> jobs(k);
    std::vector<float> out((size_t) k * cpu::MAXT * cpu::H);
    std::mt19937 pick(99);
    auto fill_layer = [&](size_t base) {
        for (int e = 0; e < k; ++e) {
            auto& j = jobs[e];
            j = cpu::ExpertJobMulti{};
            j.blob = blobs[(base + (size_t) e * 7919) % nblob];
            j.nt = tokens;
            for (int t = 0; t < tokens; ++t) {
                j.nact[t] = act[t].data();
                j.out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            }
        }
    };
    // warm-up, then timed layers, each on experts the previous layers did not read
    for (int l = 0; l < 20; ++l) { fill_layer(pick()); pool.run_split_multi_native(f, jobs.data(), k); }
    auto t0 = Clock::now();
    for (int l = 0; l < layers; ++l) { fill_layer(pick()); pool.run_split_multi_native(f, jobs.data(), k); }
    const double ms = ms_since(t0);
    const double bytes = (double) layers * k * (double) f.bytes;
    for (float v : out) if (!std::isfinite(v)) { std::fprintf(stderr, "non-finite output\n"); return 1; }
    std::printf("pool:  %.3f ms per layer, %.1f GB/s of expert weights\n", ms / layers, bytes / ms / 1e6);
    std::printf("pool phases per layer: gate/up %.1f us, quantize %.1f us, down %.1f us (incl. warm-up)\n", pool.ms_multi_gu * 1e3 / (layers + 20),
                pool.ms_multi_q * 1e3 / (layers + 20), pool.ms_multi_down * 1e3 / (layers + 20));

    // the ceiling: the same number of threads, pinned to the pool's cores, read the same experts' bytes with no barrier
    // between layers (each thread its share of every expert, as the row split cuts them): pure memory bandwidth
    std::vector<size_t> bases(layers);
    for (auto& v : bases) v = pick();
    const int threads = pool.workers() + 1;
    const std::vector<int> cores = cpu::physical_cores(false);
    std::atomic<uint64_t> sink{0};
    auto t1 = Clock::now();
    {
        std::vector<std::thread> th;
        for (int w = 0; w < threads; ++w)
            th.emplace_back([&, w] {
                if (pin && w < (int) cores.size()) cpu::pin_current_thread(cores[(size_t) w]);
                __m256i acc = _mm256_setzero_si256();
                for (int l = 0; l < layers; ++l)
                    for (int e = 0; e < k; ++e) {
                        const uint8_t* b = blobs[(bases[(size_t) l] + (size_t) e * 7919) % nblob];
                        const size_t lo = f.bytes * (size_t) w / (size_t) threads / 32 * 32;
                        const size_t hi = f.bytes * (size_t) (w + 1) / (size_t) threads / 32 * 32;
                        for (size_t o = lo; o < hi; o += 32) acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i*) (b + o)));
                    }
                sink += (uint64_t) _mm256_extract_epi64(acc, 0);
            });
        for (auto& t : th) t.join();
    }
    const double ms2 = ms_since(t1);
    std::printf("read:  %.3f ms per layer, %.1f GB/s (%d threads, no barriers: the memory's ceiling) [%llu]\n", ms2 / layers,
                bytes / ms2 / 1e6, threads, (unsigned long long) (sink.load() & 1));
    return 0;
}
