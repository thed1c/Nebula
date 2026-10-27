// rope_scaling_test: the rope configuration's host math (include/strata/kernels/rope_scaling.hpp), no GPU.
//   - none is the identity whatever the knobs hold (the regression gate: scaling off = the trained rotation, bit for bit)
//   - NTK-aware scaling is a frequency base change: base * factor^(n_rot / (n_rot - 2)), 1 at factor 1
//   - the analytic angle with the identity constants equals the unscaled angle exactly
#include "strata/kernels/rope_scaling.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace strata::kernels;

static int fails = 0;
static void check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

int main() {
    RopeScaling none;
    none.factor = 4.0;   // inert under none
    none.ext_factor = 1.0;
    const RopeKernelArgs a = none.kernel_args(64);
    check("none: identity kernel constants whatever the knobs", a.freq_scale == 1.0f && a.corr_low == 0.0f && a.corr_high == 0.0f &&
                                                                    a.ext_factor == 0.0f && a.attn_factor == 1.0f);
    check("none: freq_scale 1", none.freq_scale() == 1.0);
    // the analytic angle with identity constants = cosf/sinf of the trained angle, bit for bit
    bool exact = true;
    for (int p = 0; p < 300000; p += 997)
        for (int i = 0; i < 32; ++i) {
            const float theta = (float) p * powf(1e7f, -2.0f * (float) i / 64.0f);
            float c = 0, s = 0;
            rope_scaled_angle(theta, a.freq_scale, a.corr_low, a.corr_high, a.ext_factor, a.attn_factor, i, c, s);
            const float c0 = cosf(theta), s0 = sinf(theta);
            exact &= std::memcmp(&c, &c0, 4) == 0 && std::memcmp(&s, &s0, 4) == 0;
        }
    check("none: the analytic angle = the unscaled one, bit for bit", exact);

    check("ntk: factor 1 keeps the base", rope_ntk_freq_base(1e7, 1.0, 64) == 1e7);
    const double b2 = rope_ntk_freq_base(1e7, 2.0, 64);
    check("ntk: factor 2 at n_rot 64 = 1e7 * 2^(64/62)", std::fabs(b2 - 1e7 * std::pow(2.0, 64.0 / 62.0)) < 1e-3);
    // the lowest-frequency pair (i = n_rot/2 - 1) stretches by ~factor, the highest (i = 0) keeps its angle
    const double lo0 = std::pow(1e7, -2.0 * 31 / 64.0), lo1 = std::pow(b2, -2.0 * 31 / 64.0);
    check("ntk: the lowest frequency slows by about the factor", std::fabs(lo0 / lo1 - 2.0) < 0.05);
    check("ntk: the highest frequency is unchanged", std::pow(b2, 0.0) == std::pow(1e7, 0.0));
    RopeScaling n;
    n.freq_base = b2;
    check("ntk: resolves to a valid none configuration", rope_scaling_invalid(n) == nullptr);

    RopeScaling y;
    y.type = RopeScalingType::YaRN;
    y.factor = 2.0;
    y.ext_factor = 1.0;
    check("yarn 2: mscale 1 + 0.1 ln 2", std::fabs(y.mscale() - (1.0 + 0.1 * std::log(2.0))) < 1e-12);
    check("yarn 2: valid", rope_scaling_invalid(y) == nullptr);
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
