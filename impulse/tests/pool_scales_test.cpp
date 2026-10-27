// tests/pool_scales_test.cpp (Impulse): the CPU expert pool's per-job second-level scales (ExpertJobMulti::s_gate / s_up /
// s_down: an expert's ffn_{gate,up,down}_exps.scale entries) against llama.cpp's arithmetic. No GPU, no model.
//
//     pool-scales-test [threads]
//
// For each expert format the engines can meet (NVFP4, Q8_0, Q4_K, IQ4_XS, Q6_K), a few random experts run through
// ExpertPool::run_split_multi_native as the engines' moe ops run them (Impulse's SwiGLU_ggml, separate down rows, 1..3
// tokens per expert, every job its own scales), and:
//  1. exact: each output equals, bit for bit, the expert computed without scales and then scaled as llama.cpp's graph does
//     it (build_moe_ffn: ggml's dot, ggml_mul by the gate / up scale, ggml-cpu's SwiGLU, the hidden quantized, ggml's dot,
//     ggml_mul by the down scale). NVFP4 on the 512-bit rows (not ggml's dot) skips this one.
// (Strata's Q2_0 pack rows apply the scales too (pool.cpp), but they read ExpertJobMulti::act, which the engines' jobs
// do not fill: not an engine format, not run here.)
//  2. linear: with power-of-two up / down scales (s_gate 1) every format's output is the unscaled output times their
//     product, bit for bit: the scales are applied on every path, never dropped.
//  3. not dropped: a gate scale of 1.5 changes the output (a path that ignored the scales would pass 2 by accident only
//     if it also ignored s_up and s_down; this catches a path that ignores the gate's).
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);

namespace scpu = strata::kernels::cpu;

namespace {

constexpr int N = 2048, FF = 512, NE = 4;   // n_embd, the expert FFN, experts per run

struct Expert { std::vector<uint8_t> gu, dn; };

std::vector<uint8_t> quantize(ggml_type t, const std::vector<float> & x, int64_t rows, int64_t cols) {
    std::vector<uint8_t> q(ggml_row_size(t, cols) * rows);
    ggml_quantize_chunk(t, x.data(), q.data(), 0, rows, cols, nullptr);
    return q;
}

// llama.cpp's CPU arithmetic for one token of one expert: ggml's dot, x scale (ggml_mul), SwiGLU, the hidden quantized,
// ggml's dot, x s_down
void reference(const scpu::NativeFmt & f, const Expert & E, const void * act, float sg, float su, float sd, float * out) {
    const auto * tg = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const auto * td = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    std::vector<float> h(FF);
    for (int rb = 0; rb < FF; rb += 16) {
        float g[16], u[16], y[16];
        for (int i = 0; i < 16; ++i) {
            tg->vec_dot(N, &g[i], 0, E.gu.data() + (size_t) (rb + i) * f.gu_row, 0, act, 0, 1);
            tg->vec_dot(N, &u[i], 0, E.gu.data() + f.up_off + (size_t) (rb + i) * f.gu_row, 0, act, 0, 1);
            g[i] *= sg; u[i] *= su;
        }
        ggml_vec_swiglu_f32(16, y, g, u);
        std::memcpy(h.data() + rb, y, sizeof y);
    }
    std::vector<uint8_t> hq(f.h_bytes);
    scpu::native_quant_h(f, h.data(), hq.data());
    for (int r = 0; r < N; ++r) {
        float s = 0.f;
        td->vec_dot(FF, &s, 0, E.dn.data() + (size_t) r * f.d_row, 0, hq.data(), 0, 1);
        out[r] = s * sd;
    }
}

struct Run {
    std::vector<std::vector<float>> out;   // per (expert, token)
};

// every expert e gets nt_of(e) tokens and the scales sc(e)
template <class Sc>
Run pool_run(scpu::ExpertPool & pool, const scpu::NativeFmt & f, const std::vector<Expert> & ex, const std::vector<std::vector<uint8_t>> & acts,
             Sc sc) {
    Run R;
    std::vector<scpu::ExpertJobMulti> jobs(ex.size());
    R.out.assign(ex.size() * 3, std::vector<float>(N, 0.f));
    for (size_t e = 0; e < ex.size(); ++e) {
        auto & jb = jobs[e];
        jb.blob = ex[e].gu.data();
        jb.down = ex[e].dn.data();
        jb.nt = 1 + (int) (e % 3);
        for (int t = 0; t < jb.nt; ++t) { jb.nact[t] = acts[(e + t) % acts.size()].data(); jb.out[t] = R.out[e * 3 + t].data(); }
        sc((int) e, jb.s_gate, jb.s_up, jb.s_down);
    }
    pool.run_split_multi_native(f, jobs.data(), (int) jobs.size());
    return R;
}

int check_type(scpu::ExpertPool & pool, ggml_type t, std::mt19937 & rng) {
    scpu::NativeFmt f;
    std::string err;
    if (!scpu::native_fmt(t, t, N, FF, f, err, scpu::NativeAct::SwiGLU_ggml, /*separate_down*/ true)) {
        std::printf("%-7s skipped: %s\n", ggml_type_name(t), err.c_str());
        return 0;
    }
    std::normal_distribution<float> nd(0.f, 0.02f), na(0.f, 1.f);
    std::vector<Expert> ex(NE);
    for (auto & E : ex) {
        std::vector<float> gu((size_t) 2 * FF * N), dn((size_t) N * FF);
        for (float & v : gu) v = nd(rng);
        for (float & v : dn) v = nd(rng);
        E.gu = quantize(t, gu, 2 * FF, N);   // gate rows, then up rows (f.up_off = FF rows on)
        E.dn = quantize(t, dn, N, FF);
    }
    std::vector<std::vector<uint8_t>> acts(3, std::vector<uint8_t>(f.act_bytes));
    for (auto & a : acts) {
        std::vector<float> x(N);
        for (float & v : x) v = na(rng);
        scpu::native_quant_act(f, x.data(), a.data());
    }
    const bool ggml_rows = !scpu::q2_native_kernels(t) && (t != (ggml_type) scpu::kNvfp4Type || scpu::nvfp4_rows_exact());
    int bad = 0;
    // 1. exact, every job its own (non power of two) scales
    auto sc1 = [](int e, float & g, float & u, float & d) { g = 0.8125f + 0.0625f * e; u = 1.375f - 0.125f * e; d = 0.59375f + 0.25f * e; };
    const Run r1 = pool_run(pool, f, ex, acts, sc1);
    int n_exact = 0;
    if (ggml_rows) {
        std::vector<float> ref(N);
        for (int e = 0; e < NE; ++e) {
            float g, u, d; sc1(e, g, u, d);
            for (int k = 0; k < 1 + e % 3; ++k) {
                reference(f, ex[e], acts[(e + k) % acts.size()].data(), g, u, d, ref.data());
                for (int r = 0; r < N; ++r) n_exact += std::memcmp(&ref[r], &r1.out[e * 3 + k][r], 4) != 0;
            }
        }
        bad += n_exact;
    }
    // 2. linear: s_up 2^-1, s_down 2^-2 (s_gate 1) = the unscaled output x 2^-3
    const Run r0 = pool_run(pool, f, ex, acts, [](int, float & g, float & u, float & d) { g = u = d = 1.f; });
    const Run r2 = pool_run(pool, f, ex, acts, [](int e, float & g, float & u, float & d) { g = 1.f; u = e % 2 ? 0.5f : 2.f; d = 0.25f; });
    int n_lin = 0;
    for (int e = 0; e < NE; ++e)
        for (int k = 0; k < 1 + e % 3; ++k)
            for (int r = 0; r < N; ++r) {
                const float want = r0.out[e * 3 + k][r] * ((e % 2 ? 0.5f : 2.f) * 0.25f);
                n_lin += std::memcmp(&want, &r2.out[e * 3 + k][r], 4) != 0;
            }
    bad += n_lin;
    // 3. the gate's scale is not dropped
    const Run r3 = pool_run(pool, f, ex, acts, [](int, float & g, float & u, float & d) { g = 1.5f; u = d = 1.f; });
    int same = 0, total = 0;
    for (int e = 0; e < NE; ++e)
        for (int k = 0; k < 1 + e % 3; ++k)
            for (int r = 0; r < N; ++r) { same += r3.out[e * 3 + k][r] == r0.out[e * 3 + k][r]; total++; }
    const bool dropped = same == total;
    bad += dropped;
    std::printf("%-7s exact vs llama.cpp's scale ops: %s | power-of-two scales linear: %d values off | gate scale %s\n",
                ggml_type_name(t), ggml_rows ? (n_exact ? (std::to_string(n_exact) + " values off").c_str() : "bit for bit") :
                std::string("skipped (512-bit rows)").c_str(),
                n_lin, dropped ? "DROPPED" : "applied");
    return bad;
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_cpu_init();
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    scpu::ExpertPool pool(threads, /*pin*/ false);
    std::mt19937 rng(20261024);
    std::printf("NVFP4 rows: %s\n", scpu::nvfp4_rows_exact() ? "exact (ggml's dot)" : "512-bit");
    int bad = 0;
    for (ggml_type t : { (ggml_type) scpu::kNvfp4Type, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K })
        bad += check_type(pool, t, rng);
    std::printf("RESULT: %s\n", bad ? "FAIL" : "scales OK");
    return bad ? 1 : 0;
}
