// q35-seqs: two sequences (server slots) sharing the qwen35 / qwen35moe engine (unified KV, one recurrent cell each),
// checked against each sequence run alone.
//   q35-seqs -m MODEL.gguf --tokens A.tokens --tokens2 B.tokens [--n-gen N] [--n-cpu-moe N] [-ts a,b] [-sm layer|tensor]
//            [-c CTX] [-ctk T -ctv T] [-t N]
// (a) prompt A alone in a two-sequence engine vs the one-sequence engine (same batches: must be bit-identical);
// (b) prompt B read into sequence 1 while sequence 0 holds A (the mask hides A, B has its own recurrent cell): B's
//     logits vs B alone;
// (d) the leak test: B beside A vs B beside A reversed (same length, same cells): bit-identical when nothing leaks;
// (c) both decoded greedily together, one token each per ubatch (decode_multi: ns = 2), vs each alone (two-row ubatches take
//     other kernels than one-row ones, as in llama.cpp, so (c) is close, not bit-identical).
// One engine at a time (a 27B over two cards fits once). Prints one JSON line. Token files: route-oracle's.
#include "q35_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int32_t> read_prompt(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    uint32_t np = 0, ng = 0;
    if (fread(&np, 4, 1, f) != 1 || fread(&ng, 4, 1, f) != 1) { fclose(f); return {}; }
    std::vector<int32_t> ids(np);
    if (fread(ids.data(), 4, np, f) != np) ids.clear();
    fclose(f);
    return ids;
}
static std::vector<float> parse_split(const std::string & v) {
    std::vector<float> r; size_t a = 0;
    while (a <= v.size()) { size_t b = v.find_first_of(",/", a); if (b == std::string::npos) b = v.size(); r.push_back(std::stof(v.substr(a, b - a))); a = b + 1; }
    return r;
}
static int32_t argmax(const float * v, int n) { return (int32_t) (std::max_element(v, v + n) - v); }

struct Run { std::vector<int32_t> toks; std::vector<std::vector<float>> logits; };

static std::string solo(q35::Engine & e, int seq, const std::vector<int32_t> & prompt, int n_gen, Run & r) {
    std::vector<float> lg;
    std::string err = e.decode(prompt, &lg, nullptr, seq);
    if (!err.empty()) return err;
    const int nv = e.hp().n_vocab;
    for (int s = 0; s < n_gen; ++s) {
        r.logits.push_back(lg);
        const int32_t t = argmax(lg.data(), nv);
        r.toks.push_back(t);
        if (!(err = e.decode({ t }, &lg, nullptr, seq)).empty()) return err;
    }
    return "";
}

int main(int argc, char ** argv) {
    q35::Options o;
    o.n_ctx = 8192; o.n_cpu_moe = 0; o.type_k = o.type_v = GGML_TYPE_F16;
    std::string pa, pb;
    int n_gen = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "-m") o.gguf = next();
        else if (a == "--tokens") pa = next();
        else if (a == "--tokens2") pb = next();
        else if (a == "--n-gen") n_gen = std::stoi(next());
        else if (a == "--n-cpu-moe") o.n_cpu_moe = std::stoi(next());
        else if (a == "-ts" || a == "--tensor-split") o.tensor_split = parse_split(next());
        else if (a == "-sm") o.split_tensor = next() == "tensor";
        else if (a == "-c") o.n_ctx = std::stoi(next());
        else if (a == "-t") o.n_threads = std::stoi(next());
        else if (a == "-ctk" || a == "-ctv") {
            const std::string t = next();
            (a == "-ctk" ? o.type_k : o.type_v) = t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    const std::vector<int32_t> A = read_prompt(pa), B = read_prompt(pb);
    if (A.empty() || B.empty()) { fprintf(stderr, "need --tokens and --tokens2\n"); return 2; }

    Run a1, b1;   // each alone, one-sequence engine
    {
        q35::Engine e; o.n_seq = 1;
        std::string err = e.load(o);
        if (!err.empty()) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        if (!(err = solo(e, 0, A, n_gen, a1)).empty()) { fprintf(stderr, "A: %s\n", err.c_str()); return 1; }
        e.reset();
        if (!(err = solo(e, 0, B, n_gen, b1)).empty()) { fprintf(stderr, "B: %s\n", err.c_str()); return 1; }
    }
    q35::Engine e; o.n_seq = 2;
    std::string err = e.load(o);
    if (!err.empty()) { fprintf(stderr, "load 2: %s\n", err.c_str()); return 1; }
    const int nv = e.hp().n_vocab;
    auto same = [&](const std::vector<float> & x, const std::vector<float> & y) { return x.size() == y.size() && memcmp(x.data(), y.data(), x.size() * 4) == 0; };
    auto maxdiff = [&](const std::vector<float> & x, const std::vector<float> & y) { double m = 0; for (size_t i = 0; i < x.size(); ++i) m = std::max(m, (double) std::fabs(x[i] - y[i])); return m; };

    // (a) A alone in sequence 0 of the two-sequence engine; then in sequence 1 (the other recurrent cell)
    Run a2, a3;
    if (!(err = solo(e, 0, A, n_gen, a2)).empty()) { fprintf(stderr, "a: %s\n", err.c_str()); return 1; }
    int a_ident = 0; for (int s = 0; s < n_gen; ++s) a_ident += same(a2.logits[s], a1.logits[s]);
    e.reset();
    if (!(err = solo(e, 1, A, n_gen, a3)).empty()) { fprintf(stderr, "a seq 1: %s\n", err.c_str()); return 1; }
    int a1_ident = 0; for (int s = 0; s < n_gen; ++s) a1_ident += same(a3.logits[s], a1.logits[s]);
    e.reset();

    // (b) A in sequence 0, then B read into sequence 1
    std::vector<float> la, lb;
    if (!(err = e.decode(A, &la, nullptr, 0)).empty() || !(err = e.decode(B, &lb, nullptr, 1)).empty()) { fprintf(stderr, "b: %s\n", err.c_str()); return 1; }
    const bool b_ident = same(lb, b1.logits[0]);
    const double b_diff = maxdiff(lb, b1.logits[0]);
    e.reset();

    // (d) B beside A reversed
    std::vector<float> la2, lb2;
    {
        std::vector<int32_t> Ar(A.rbegin(), A.rend());
        if (!(err = e.decode(Ar, &la2, nullptr, 0)).empty() || !(err = e.decode(B, &lb2, nullptr, 1)).empty()) { fprintf(stderr, "d: %s\n", err.c_str()); return 1; }
    }
    const bool d_ident = same(lb, lb2);
    const double d_diff = maxdiff(lb, lb2);
    e.reset();

    // (c) both together, one token each per ubatch; first the same reads as (b)
    if (!(err = e.decode(A, &la, nullptr, 0)).empty() || !(err = e.decode(B, &lb, nullptr, 1)).empty()) { fprintf(stderr, "c read: %s\n", err.c_str()); return 1; }
    int c_tok_a = 0, c_tok_b = 0, c_ident = 0, first_div_a = -1, first_div_b = -1;
    double c_diff = 0;
    for (int s = 0; s < n_gen; ++s) {
        const int32_t xa = argmax(la.data(), nv), xb = argmax(lb.data(), nv);
        if (first_div_a < 0) { c_ident += same(la, a1.logits[s]); c_diff = std::max(c_diff, maxdiff(la, a1.logits[s])); if (xa == a1.toks[s]) c_tok_a++; else first_div_a = s; }
        if (first_div_b < 0) { c_ident += same(lb, b1.logits[s]); c_diff = std::max(c_diff, maxdiff(lb, b1.logits[s])); if (xb == b1.toks[s]) c_tok_b++; else first_div_b = s; }
        std::vector<float> both;
        err = e.decode_multi({ { 0, { first_div_a < 0 ? a1.toks[s] : xa } }, { 1, { first_div_b < 0 ? b1.toks[s] : xb } } }, &both);
        if (!err.empty()) { fprintf(stderr, "c: %s\n", err.c_str()); return 1; }
        la.assign(both.begin(), both.begin() + nv); lb.assign(both.begin() + nv, both.begin() + 2 * nv);
    }
    printf("{\"n_gen\":%d,\"a_alone_in_2seq_bit_identical\":%d,\"a_in_seq1_bit_identical\":%d,\"b_beside_a_bit_identical\":%s,"
           "\"b_beside_a_max_abs_diff\":%g,\"d_no_leak_bit_identical\":%s,\"d_max_abs_diff\":%g,\"c_tokens_same_a\":%d,"
           "\"c_tokens_same_b\":%d,\"c_first_divergence_a\":%d,\"c_first_divergence_b\":%d,\"c_steps_bit_identical\":%d,\"c_max_abs_diff\":%g}\n",
           n_gen, a_ident, a1_ident, b_ident ? "true" : "false", b_diff, d_ident ? "true" : "false", d_diff, c_tok_a, c_tok_b,
           first_div_a, first_div_b, c_ident, c_diff);
    return 0;
}
