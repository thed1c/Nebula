// fx-dispatch-test: which engine impulse-server runs for a -m path (src/fx/fx_pack.hpp, the code main calls), on made-up
// folders and GGUFs (no GPU, no model).  One Engine P1: a Flash-Next pack -> fx; a GGUF alone -> its architecture's
// engine, so a qwen4exp GGUF without a pack still goes to q35 exactly as before.
#include "fx/fx_pack.hpp"

#include "gguf.h"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static int fails = 0;
static void expect(const std::string & what, const std::string & got, const std::string & want) {
    const bool ok = got == want;
    std::printf("%s %-56s %s%s\n", ok ? "ok  " : "FAIL", what.c_str(), got.c_str(), ok ? "" : (" (want " + want + ")").c_str());
    if (!ok) ++fails;
}

static void gguf(const fs::path & p, const char * arch) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", arch);
    gguf_write_to_file(g, p.string().c_str(), false);
    gguf_free(g);
}

static void touch(const fs::path & p) { std::ofstream(p) << "x\n"; }

int main() {
    const fs::path T = fs::temp_directory_path() / ("fx-dispatch-test-" + std::to_string((long long) std::time(nullptr)));
    fs::remove_all(T);
    // a Flash-Next pack folder, as the reference engine's: pack/ (index.txt ...) + orca-nvfp4-dense.gguf + its other GGUFs
    const fs::path bb = T / "orca-flashnext-nvfp4";
    fs::create_directories(bb / "pack" / "tokenizer");
    touch(bb / "pack" / "index.txt");
    gguf(bb / "orca-nvfp4-dense.gguf", "qwen4exp");
    gguf(bb / "ple-fp8.gguf", "qwen4exp");
    gguf(bb / "token-embd-bf16.gguf", "qwen4exp");
    // a Flash-Next GGUF alone (the GGUF path: q35's qwen4exp graph) and other architectures
    fs::create_directories(T / "gguf");
    gguf(T / "gguf" / "qwen38-flash-next-nvfp4.gguf", "qwen4exp");
    gguf(T / "gguf" / "flash-next-dense.gguf", "qwen4exp");   // "dense" in its name, but no pack/ beside it
    gguf(T / "gguf" / "qwen36.gguf", "qwen35moe");
    gguf(T / "gguf" / "lfm.gguf", "lfm2moe");
    gguf(T / "gguf" / "gemma4.gguf", "gemma4");
    // a folder whose pack/ is not a Strata pack (no index.txt): not fx
    const fs::path np = T / "not-a-pack";
    fs::create_directories(np / "pack");
    gguf(np / "x-dense.gguf", "qwen4exp");
    // a pack folder without a dense GGUF: not fx
    const fs::path nd = T / "no-dense";
    fs::create_directories(nd / "pack");
    touch(nd / "pack" / "index.txt");
    gguf(nd / "model.gguf", "qwen4exp");

    expect("pack folder", fx::engine_for(bb.string()), "fx");
    expect("pack folder (trailing slash)", fx::engine_for(bb.string() + "/"), "fx");
    expect("its dense GGUF beside pack/", fx::engine_for((bb / "orca-nvfp4-dense.gguf").string()), "fx");
    expect("another GGUF of the pack folder (ple-fp8.gguf)", fx::engine_for((bb / "ple-fp8.gguf").string()), "q35");
    expect("qwen4exp GGUF alone", fx::engine_for((T / "gguf" / "qwen38-flash-next-nvfp4.gguf").string()), "q35");
    expect("qwen4exp *dense*.gguf without pack/", fx::engine_for((T / "gguf" / "flash-next-dense.gguf").string()), "q35");
    expect("qwen35moe GGUF", fx::engine_for((T / "gguf" / "qwen36.gguf").string()), "q35");
    expect("lfm2moe GGUF", fx::engine_for((T / "gguf" / "lfm.gguf").string()), "q35");
    expect("gemma4 GGUF", fx::engine_for((T / "gguf" / "gemma4.gguf").string()), "g4");
    expect("pack/ without index.txt (the folder)", fx::engine_for(np.string()), "g4");
    expect("pack/ without index.txt (its dense GGUF)", fx::engine_for((np / "x-dense.gguf").string()), "q35");
    expect("pack folder without a dense GGUF", fx::engine_for(nd.string()), "g4");
    expect("no such path", fx::engine_for((T / "missing").string()), "g4");

    fx::Pack p;
    expect("find_pack: the dense GGUF", fx::find_pack(bb.string(), p) ? fs::path(p.dense).filename().string() : "none", "orca-nvfp4-dense.gguf");
    expect("find_pack: the pack", fx::find_pack(bb.string(), p) ? fs::path(p.pack).filename().string() : "none", "pack");

    // -m's value on a command line; the engine's own arguments after --fx-args are not impulse-server's
    {
        const char * a1[] = { "impulse-server", "--port", "8080", "-m", "/m/pack-folder", "--alias", "x" };
        expect("model_arg -m", fx::model_arg(7, (char **) a1), "/m/pack-folder");
        const char * a2[] = { "impulse-server", "--model", "/m/a", "--fx-args", "--pack", "/p", "-m", "/not/this" };
        expect("model_arg stops at --fx-args", fx::model_arg(8, (char **) a2), "/m/a");
        const char * a3[] = { "impulse-server", "--port", "1" };
        expect("model_arg: none", fx::model_arg(3, (char **) a3), "");
    }
    fs::remove_all(T);
    std::printf("%s (%d failed)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
