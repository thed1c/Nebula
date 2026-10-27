#include "../lite/platform.hpp"
#include "../lite/chat_page.hpp"
#include "../lite/utf8.hpp"
#include "../lite/model_info.hpp"
#include "../lite/llama_compat.hpp"
#include "../lite/api_key.hpp"
#include <filesystem>
#include <cstdlib>
#include <thread>
extern "C" int stbi_info_from_memory(const unsigned char * buffer, int len, int * x, int * y, int * comp);   // in libmtmd (stb_image)
#include <cstring>
#include "gguf.h"
#include "../fx/fx_pack.hpp"
#if defined(_WIN32)
#include <process.h>   // _spawnv (exec_fx_front)
#endif
// impulse-server: an OpenAI-compatible server for the gemma4 engine, API-identical to llama-server for what
// the reference server's clients use (docs/API.md): POST /v1/chat/completions (and /chat/completions), GET /health,
// /v1/models (/models), /props, /slots.
//
// llama-server's own code does the request parsing (oaicompat_chat_params_parse: chat template, tools, response_format
// -> grammar, chat_template_kwargs, image parts -> media markers), the image tokenization (process_mtmd_prompt), the
// sampling parameters (eval_llama_cmpl_schema), the sampler chain with its grammar and reasoning budget (common_sampler,
// through common_sampler_sample_logits) and the response JSON (server_task_result_cmpl_final). The engine does the
// compute: text and image chunks, MTP verify windows (llama.cpp's sample-and-accept), the prompt cache.
// One request at a time (the reference server runs --parallel 1).
#include "g4_engine.hpp"

#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <condition_variable>
#include <future>
#include <deque>
#include <functional>
#include <vector>
#ifdef IMPULSE_VERSION
#define IMPULSE_VERSION_STR IMPULSE_VERSION
#else
#define IMPULSE_VERSION_STR "dev"
#endif

namespace g4srv {

struct Args {
    lite::ApiKeys keys;   // --api-key / --api-key-file (llama-server's): none = no check
    int timeout = 7200; bool metrics = false, webui = true;   // llama-server's --timeout, --metrics, --no-webui
    std::string model, mmproj, mtp, alias = "impulse", host = "127.0.0.1", profile;
    int port = 8080, n_draft = 3, image_max_tokens = 384, threads = 0;   // 384: the maintainer, 2026-10-10; threads 0 = physical cores
    // MTP depth by context (v0.2.2): drafts accepted fall as the context grows, and a rejected draft costs a wider
    // verify; on a 3060, depth 2 is 9 % slower than 3 on a short chat and 33 % faster at 157K
    int n_draft_long = 2, n_draft_long_from = 32768;
    int read_interleave = 0;   // slots: other slots step after each N prompt tokens of a dense read (0 = read in one go)
    bool reserve_set = false, n_draft_long_set = false;
    std::string mmproj_place;   // second: on a second visible GPU (default when one has room); gpu: always on the engine's card;
                                // cpu: in RAM; auto: in RAM, on the card while images need it;
                                // lazy: not loaded until an image comes (RAM or card by the rules), unloaded when idle. Default: lazy
                                // under 32 GB of RAM, auto above
    int mmproj_idle = 15, mmproj_min_size = 128, mmproj_gpu_max_ctx = 65536;   // 15 s, 64k: the maintainer, 2026-10-10
    g4::Options eo;
    common_params params;
};

static std::vector<float> parse_split(const std::string & v) {   // "52,48,0": llama.cpp's --tensor-split shares
    std::vector<float> r; size_t a = 0;
    while (a <= v.size()) { size_t b = v.find_first_of(",/", a); if (b == std::string::npos) b = v.size(); r.push_back(std::stof(v.substr(a, b - a))); a = b + 1; }
    return r;
}

static void usage() {
    fprintf(stderr, "impulse-server -m MODEL [--mmproj F] [--mtp F] [--alias A] [--host H] [--port P] [--api-key K[,K]] [--api-key-file F] [-c CTX] [-ub N]\n"
                    "  [--mmproj-place second|auto|lazy|gpu|cpu --mmproj-min-size PX --mmproj-gpu-max-ctx N --mmproj-idle S]\n"
                    "  [--adapt-half-life T --adapt-budget N --no-adapt --no-adapt-refill --no-async-swaps --no-overlap --no-lend]\n"
                    "  [--ram-budget GIB] (experts locked in RAM; the rest read from the model file; 0 = low-RAM mode; default: what the PC can spare)\n"
                    "  [--temp T --top-k K --top-p P --min-p M --repeat-penalty R] [--image-max-tokens N] [-t N]\n"
                    "  [--n-cpu-moe N --cache-slots S --profile F --pool-threads N --prefill-ubatch N --lend -ctk T -ctv T --n-draft N]\n"
                    "  [--n-draft-long N --n-draft-long-from POS] (MTP depth once the context passes POS; default 2 from 32768)\n");
}

static ggml_type kv_type(const std::string & t) { return t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }

// impulse-server is the one entry point: a GGUF of another architecture goes to that engine's server with the same
// arguments. The release build links both servers into one binary (IMPULSE_COMBINED); a per-engine build starts
// impulse-server-qwen35 from $IMPULSE_QWEN35_SERVER or next to this binary.
static std::string model_architecture(int argc, char ** argv) {
    std::string model;
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) model = argv[i + 1];
    if (model.empty()) return "";
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(model.c_str(), gp);
    if (!g) return "";
    const int64_t id = gguf_find_key(g, "general.architecture");
    const std::string arch = id < 0 ? "" : gguf_get_val_str(g, id);
    gguf_free(g);
    return arch;
}

#if !defined(IMPULSE_COMBINED) && !defined(_WIN32)
static void exec_qwen_server(int argc, char ** argv) {
    std::string exe;
    if (const char * e = getenv("IMPULSE_QWEN35_SERVER")) exe = e;
    else exe = lite::self_dir() + "impulse-server-qwen35";
    std::vector<char *> av = { (char *) exe.c_str() };
    for (int i = 1; i < argc; ++i) av.push_back(argv[i]);
    av.push_back(nullptr);
    fprintf(stderr, "impulse-server: a hybrid MoE model: starting %s\n", exe.c_str());
    execv(exe.c_str(), av.data());
    fprintf(stderr, "impulse-server: cannot start %s (build src/qwen35, or set IMPULSE_QWEN35_SERVER)\n", exe.c_str());
    exit(1);
}
#endif

int run(int argc, char ** argv) {
    // a llama-server command line (src/lite/llama_compat.hpp): its flags renamed to ours or accepted without effect
    bool want_help = false;
    std::vector<std::string> compat_notes;
    std::vector<std::string> cargs = lite::llama_compat_args(argc, argv, want_help, compat_notes);
    if (want_help) { lite::llama_compat_help(stdout, "impulse-server"); return 0; }
    std::vector<char *> cargv;
    for (auto & x : cargs) cargv.push_back((char *) x.c_str());
    cargv.push_back(nullptr);
    argc = (int) cargs.size(); argv = cargv.data();
    Args A;
    common_params & P = A.params;
    // neutral public defaults (the maintainer, 2026-10-10: "we will make them neutral public"): llama-server's sampling, a
    // neutral alias and port, threads from the machine, no profile needed
    P.use_jinja = true;
    P.n_parallel = 1;
    A.eo.n_ctx = 262144; A.eo.n_cpu_moe = 30; A.eo.pool = true; A.eo.pool_threads = 0; A.eo.cache_slots = -1;
    A.eo.adapt = true; A.eo.async_swaps = true; A.eo.overlap = true; A.eo.prefill_ubatch = -1; A.eo.lend = true;
    A.eo.kv_initial = 32768;   // the KV pager: 32k global-layer cells on the card at load, grown as needed (0 = all at load)
    // q8_0 K/V (the maintainer, 2026-10-11): at 64k q4_0 costs Gemma 4 top-1 80.5 % vs a noise floor of 87.1 %, q8_0 is back at
    // 87.1 % (Qwen3.6 keeps q4_0: its q4 cost is inside the floor); the KV pager keeps the extra VRAM to what the chat uses
    A.eo.type_k = GGML_TYPE_Q8_0; A.eo.type_v = GGML_TYPE_Q8_0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { usage(); exit(2); } return std::string(argv[++i]); };
        if (a == "-m" || a == "--model") A.model = nx();
        else if (a == "--mmproj") A.mmproj = nx();
        else if (a == "--mtp") A.mtp = nx();
        else if (a == "--alias") A.alias = nx();
        else if (a == "--timeout") A.timeout = std::stoi(nx());
        else if (a == "--metrics") A.metrics = true;
        else if (a == "--no-webui") A.webui = false;
        else if (a == "--host") A.host = nx();
        else if (a == "--api-key") A.keys.add_csv(nx());
        else if (a == "--api-key-file") { const std::string e = A.keys.add_file(nx()); if (!e.empty()) { fprintf(stderr, "error: %s\n", e.c_str()); return 1; } }
        else if (a == "--port") A.port = std::stoi(nx());
        else if (a == "-c" || a == "--ctx-size") A.eo.n_ctx = std::stoi(nx());
        else if (a == "-ub" || a == "--ubatch-size") A.eo.n_ubatch = std::stoi(nx());
        else if (a == "--temp") P.sampling.temp = std::stof(nx());
        else if (a == "--top-k") P.sampling.top_k = std::stoi(nx());
        else if (a == "--top-p") P.sampling.top_p = std::stof(nx());
        else if (a == "--min-p") P.sampling.min_p = std::stof(nx());
        else if (a == "--repeat-penalty") P.sampling.penalty_repeat = std::stof(nx());
        else if (a == "--image-max-tokens") A.image_max_tokens = std::stoi(nx());
        else if (a == "--mmproj-place") A.mmproj_place = nx();
        else if (a == "--lora") A.eo.lora = nx();                                                  // llama.cpp's
        else if (a == "--lora-scaled") { A.eo.lora = nx(); A.eo.lora_scale = std::stof(nx()); }
        else if (a == "--mmproj-idle") A.mmproj_idle = std::stoi(nx());
        else if (a == "--mmproj-min-size") A.mmproj_min_size = std::stoi(nx());   // px: smaller images encode on the CPU (auto)
        else if (a == "--mmproj-gpu-max-ctx") A.mmproj_gpu_max_ctx = std::stoi(nx());
        else if (a == "-t" || a == "--threads") A.threads = std::stoi(nx());
        else if (a == "--n-cpu-moe") A.eo.n_cpu_moe = std::stoi(nx());
        else if (a == "--cache-slots") { std::string v = nx(); A.eo.cache_slots = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--vram-reserve-mb") { A.eo.vram_reserve_mb = std::stoi(nx()); A.reserve_set = true; }
        else if (a == "--profile") A.eo.profile = nx();
        else if (a == "--pool-threads") A.eo.pool_threads = std::stoi(nx());
        else if (a == "--prefill-ubatch") { std::string v = nx(); A.eo.prefill_ubatch = v == "auto" ? -1 : std::stoi(v); }
        else if (a == "--kv-initial") A.eo.kv_initial = std::stoi(nx());   // KV pager: cells on the card at load (0 = all)
        else if (a == "--kv-step") A.eo.kv_step = std::stoi(nx());
        else if (a == "--no-lend") A.eo.lend = false;
        // the expert cache's behaviour (defaults: Gate 0's adaptive policy, the copy stream and the CPU/GPU overlap on)
        else if (a == "--adapt-half-life") A.eo.adapt_half_life = std::stof(nx());     // tokens; usage counts halve over this many
        else if (a == "--adapt-budget") A.eo.adapt_budget = std::stoi(nx());          // swaps per layer per decode step
        else if (a == "--no-adapt") A.eo.adapt = false;                               // a fixed cache (the profile's, or plain order)
        else if (a == "--no-adapt-refill") A.eo.adapt_refill = false;                 // keep the cache as it was after a prompt read
        else if (a == "--no-async-swaps") A.eo.async_swaps = false;                   // swaps copied in the step, not in the background
        else if (a == "--ram-budget") A.eo.ram_budget_gib = std::stod(nx());           // GiB of experts locked in RAM (the RAM tier)
        else if (a == "--no-overlap") A.eo.overlap = false;                           // the CPU pool waits for the GPU, no overlap
        else if (a == "--tensor-split" || a == "-ts") A.eo.tensor_split = parse_split(nx());
        else if (a == "--split-mode" || a == "-sm") {   // layer (llama.cpp's default with several GPUs) or tensor
            const std::string v = nx();
            if (v == "tensor") A.eo.split_tensor = true; else if (v != "layer") { fprintf(stderr, "--split-mode layer|tensor\n"); return 2; }
        }
        else if (a == "-ctk") A.eo.type_k = kv_type(nx());
        else if (a == "-ctv") A.eo.type_v = kv_type(nx());
        else if (a == "--n-draft") A.n_draft = std::stoi(nx());
        else if (a == "--read-interleave") {
            A.read_interleave = std::stoi(nx());
            if (A.read_interleave % std::max(1, A.eo.n_ubatch) != 0) { fprintf(stderr, "--read-interleave: a multiple of -ub\n"); return 2; }
        }
        else if (a == "-np" || a == "--parallel") { A.eo.n_seq = std::stoi(nx()); P.n_parallel = A.eo.n_seq; }   // slots (llama-server's)
        else if (a == "--n-draft-long") { A.n_draft_long = std::stoi(nx()); A.n_draft_long_set = true; }
        else if (a == "--n-draft-long-from") A.n_draft_long_from = std::stoi(nx());
        else if (a == "--cache-ram") { nx(); fprintf(stderr, "impulse: --cache-ram has no effect with the Gemma engine (each slot keeps its own prompt cache)\n"); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    if (A.model.empty()) { usage(); return 2; }
    if (!A.n_draft_long_set) A.n_draft_long = std::min(A.n_draft, A.n_draft_long);   // a lower --n-draft holds everywhere
    if (!A.eo.adapt) {   // a fixed cache: no swaps to copy, and a lent arena would come back empty with nothing to refill it
        A.eo.async_swaps = false;
        if (A.eo.lend) { A.eo.lend = false; fprintf(stderr, "impulse: --no-adapt reads without lending the cache's VRAM\n"); }
    }
    if (A.threads <= 0 || A.eo.pool_threads <= 0) {   // physical cores among the CPUs this process may run on (numactl respected)
        const int cores = lite::physical_cores_or_half();
        if (A.threads <= 0) A.threads = cores;
        if (A.eo.pool_threads <= 0) A.eo.pool_threads = cores;
    }
    A.eo.gguf = A.model; A.eo.n_threads = A.threads;
    if (A.eo.prefill_ubatch > A.eo.n_batch) A.eo.n_batch = A.eo.prefill_ubatch;
    if (A.eo.prefill_ubatch < 0 && A.eo.n_batch < 32768) A.eo.n_batch = 32768;
    P.model.path = A.model;
    P.n_ctx = A.eo.n_ctx;

    llama_backend_init();
    // the vocabulary, the chat template and the tokenizer come from llama.cpp (vocab-only load: no weights)
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(A.model.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load the vocabulary\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n_vocab; ++t) if (llama_vocab_is_eog(vocab, t)) P.sampling.logit_bias_eog.push_back({ t, -INFINITY });

    // auto slots: what loads after the arena (the drafter, the projector and its largest image encode) plus step headroom
    auto fsize = [](const std::string & p) -> size_t { std::error_code ec; const auto n = p.empty() ? 0 : std::filesystem::file_size(p, ec); return ec ? 0 : (size_t) n; };
    // a second visible GPU takes the projector whole (the maintainer, 2026-10-10: "if there is a second card detected it
    // automatically puts mmproj there"): no expert slots given up, no moves, no CPU encodes. With --tensor-split the second
    // card is the first one that holds no layers (a share of 0 or past the list: llama.cpp's projector-only card)
    ggml_backend_dev_t second_dev = nullptr;
    size_t second_free = 0;
    {
        const auto & ts = A.eo.tensor_split;
        size_t k = 0;
        for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(d);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const bool free_card = ts.size() > 1 ? (k >= ts.size() || ts[k] <= 0) : k == 1;
            ++k;
            if (free_card) { size_t tot = 0; ggml_backend_dev_memory(dev, &second_free, &tot); second_dev = dev; break; }
        }
    }
    const bool second_fits = second_dev && second_free >= (((fsize(A.mmproj) >> 20) + 768) << 20);
    if (A.mmproj_place.empty()) {
        const double gb = lite::total_ram_gb();
        A.mmproj_place = second_fits ? "second" : gb > 0 && gb < 32 ? "lazy" : "auto";   // the maintainer, 2026-10-10: "32Gb for mmproj"
    }
    if (A.mmproj_place != "second" && A.mmproj_place != "gpu" && A.mmproj_place != "cpu" && A.mmproj_place != "auto" && A.mmproj_place != "lazy") {
        fprintf(stderr, "--mmproj-place second|gpu|cpu|auto|lazy\n"); return 1;
    }
    if (A.mmproj_place == "second" && !A.mmproj.empty() && !second_fits) {
        fprintf(stderr, "--mmproj-place second: %s\n", second_dev ? "the second GPU has too little free memory" : "no second GPU is visible"); return 1;
    }
    const bool mm_gpu_always = !A.mmproj.empty() && A.mmproj_place == "gpu", mm_auto = !A.mmproj.empty() && A.mmproj_place == "auto";
    const bool mm_lazy = !A.mmproj.empty() && A.mmproj_place == "lazy";
    const bool mm_second = !A.mmproj.empty() && A.mmproj_place == "second";
    const size_t mmproj_card = (fsize(A.mmproj) >> 20) + 768;   // the projector on the card + its 1024 px encode, MiB
    const size_t after_mb = 256 + (A.mtp.empty() ? 0 : (fsize(A.mtp) >> 20) + 128) + (mm_gpu_always ? mmproj_card - 256 : 0);
    if (A.eo.cache_slots < 0 && !A.reserve_set) A.eo.vram_reserve_mb = (int) after_mb;
    g4::Engine eng;
    std::string err = eng.load(A.eo);
    if (!err.empty()) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
    const bool mtp = !A.mtp.empty();
    if (A.eo.cache_slots >= 0) eng.ensure_free(after_mb << 20, true);   // fixed slots that leave no room: fewer
    if (mtp) { err = eng.load_mtp(A.mtp); if (!err.empty()) { fprintf(stderr, "mtp: %s\n", err.c_str()); return 1; } }

    // the projector: on the card (gpu), in RAM only (cpu), or in RAM with a card copy while images need it (auto: small
    // images such as a watchdog probe encode on the CPU; bigger ones move the projector to the card, slots give way, and
    // after mmproj_idle seconds without one it leaves the card and the slots come back)
    auto mm_init = [&](bool gpu, ggml_backend_dev_t dev = nullptr) -> mtmd_context * {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = gpu; mparams.device = gpu ? (dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) : nullptr;
        mparams.print_timings = false; mparams.n_threads = A.threads;
        mparams.image_max_tokens = A.image_max_tokens; mparams.media_marker = get_media_marker();
        return mtmd_init_from_file(A.mmproj.c_str(), model, mparams);
    };
    auto mm_warm = [&](mtmd_context * c) {   // the encoder's compute buffer at the largest image a client usually sends (1024 px), kept by mtmd
        std::vector<unsigned char> px((size_t) 1024 * 1024 * 3, 128);
        mtmd_bitmap * bm = mtmd_bitmap_init(1024, 1024, px.data());
        mtmd_input_chunks * chs = mtmd_input_chunks_init();
        const std::string marker = get_media_marker();
        mtmd_input_text txt; txt.text = marker.c_str(); txt.text_len = marker.size(); txt.add_special = false; txt.parse_special = true;
        const mtmd_bitmap * bms[1] = { bm };
        if (mtmd_tokenize(c, chs, &txt, bms, 1) == 0)
            for (size_t i = 0; i < mtmd_input_chunks_size(chs); ++i) {
                const mtmd_input_chunk * k = mtmd_input_chunks_get(chs, i);
                if (mtmd_input_chunk_get_type(k) == MTMD_INPUT_CHUNK_TYPE_IMAGE) mtmd_encode_chunk(c, k);
            }
        mtmd_input_chunks_free(chs); mtmd_bitmap_free(bm);
    };
    // mctx: the projector in RAM (cpu/auto: resident; lazy: while images come); gpu: the card copy (mctx too when gpu-always)
    std::atomic<mtmd_context *> mctx{ nullptr }, mctx_gpu{ nullptr };
    if (!A.mmproj.empty() && !mm_lazy) {
        mctx = mm_init(mm_gpu_always || mm_second, mm_second ? second_dev : nullptr);
        if (!mctx) { fprintf(stderr, "cannot load the mmproj\n"); return 1; }
        if (mm_second) { mm_warm(mctx); fprintf(stderr, "impulse: projector on the second GPU (%s, %zu MiB free before)\n", ggml_backend_dev_description(second_dev), second_free >> 20); }
        if (mm_gpu_always) { mctx_gpu = mctx.load(); mm_warm(mctx_gpu); }
    }
    std::atomic<int64_t> mm_last_gpu{ 0 }, mm_last_use{ 0 };
    auto now_s = []() -> int64_t { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    std::atomic<int> mm_moves{ 0 };
    eng.ensure_free((size_t) 256 << 20, true);   // step headroom
    const int slots_full = eng.slots();
    auto need_gpu = [&]() -> mtmd_context * {   // the projector on the card: slots give way first
        if (!mctx_gpu) {
            const auto tm0 = std::chrono::steady_clock::now();
            eng.ensure_free(mmproj_card << 20);
            mctx_gpu = mm_init(true);
            if (!mctx_gpu) return nullptr;
            mm_moves++;
            fprintf(stderr, "impulse: projector to the card in %.2f s (%d slots per layer)\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count(), eng.slots());
        }
        return mctx_gpu;
    };
    auto need_cpu = [&]() -> mtmd_context * {   // lazy: the projector into RAM
        if (!mctx) {
            const auto tm0 = std::chrono::steady_clock::now();
            mctx = mm_init(false);
            if (mctx) fprintf(stderr, "impulse: projector loaded into RAM in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count());
        }
        return mctx;
    };
    fprintf(stderr, "impulse: %d expert slots per layer (projector: %s)\n", eng.slots(), A.mmproj.empty() ? "none" : A.mmproj_place.c_str());

    server_chat_params chat;
    {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, P.chat_template);
        const bool thinking = P.enable_reasoning != 0 && common_chat_templates_support_enable_thinking(tmpls.get());
        chat = { P.use_jinja, P.prefill_assistant, P.reasoning_format, P.default_template_kwargs, std::move(tmpls),
                 !A.mmproj.empty(), false, false, thinking,
                 P.sampling.reasoning_budget_tokens, P.sampling.reasoning_budget_message, P.media_path, P.force_pure_content_parser };
    }

    // ---------------------------------------------------------------- generation: slots (llama-server's --parallel)
    // Each slot is one sequence of the engine's KV with its own prompt cache, sampler and MTP state; its context is n_ctx /
    // slots (llama-server's n_ctx_slot). A scheduler thread owns the engine: a queued request goes to the idle slot whose
    // prompt cache shares the longest prefix with it, its prompt is read, then every generating slot steps together, one
    // token each (or one MTP verify window each) per ubatch (Engine::decode_multi). With one slot the engine calls are
    // the one-request server's, call for call.
    std::mutex busy;                      // the engine's owner: the scheduler, and the projector's idle thread
    std::atomic<bool> processing{ false };
    const int n_slots = A.eo.n_seq, n_ctx_slot = eng.n_ctx() / n_slots;
    struct Job { json body; bool chat_api = false; std::promise<std::pair<int, json>> done; };
    struct Slot {
        int id = 0;
        server_tokens cached;             // the slot's prompt cache: what its sequence of the KV holds (prompt + generated)
        std::vector<float> h_pending;     // MTP: the hidden row that pairs with the next token
        std::unique_ptr<Job> job;         // the request being answered (null: idle)
        std::atomic<bool> active{ false };
        // the request's state between steps
        task_params tp; common_sampler * smpl = nullptr; server_slot_stats stats;
        std::string text, stopping_word; llama_tokens out; stop_type stop = STOP_TYPE_NONE;
        int n_predict = 0, n_prompt = 0; size_t keep = 0; llama_token tok = 0; bool go = false;
        std::vector<float> lg;
        int64_t t_last = 0;               // when its last request finished (the least recently used idle slot takes a new one)
    };
    std::vector<std::unique_ptr<Slot>> slots;
    for (int k = 0; k < n_slots; ++k) { slots.emplace_back(new Slot()); slots.back()->id = k; }
    lite::Metrics metrics;   // /metrics (--metrics)
    std::mutex qmu; std::condition_variable qcv;
    std::deque<std::unique_ptr<Job>> queue;

    // the request's prompt read into slot S (its sequence): everything the one-request server did before the first sample
    // --read-interleave N (off by default): while a dense model reads a long prompt, the other slots take one step after each
    // N tokens of it, as llama-server's batches do. Measured (a dense 31B Gemma, two 5060 Ti, a 400-token answer on slot A while a
    // 46.8K-token prompt arrives on slot B): llama-server A 81.9 s / B 67.7 s; one go (the default) A 73.7 / B 45.3 s; N = 2048
    // A 83.9 / B 57.2 s; N = 512 A 112 / B 91 s: every step reads its logits, which drains the cards' pipeline. An expert
    // model always reads in one go (each piece would lend the cache's VRAM and every step take it back).
    std::function<void()> step_others;
    auto others_writing = [&](const Slot & self) { for (auto & S : slots) if (S.get() != &self && S->job && S->go) return true; return false; };
    // With several slots a dense read also goes in pieces of whole ubatches (a multiple of n_ubatch: the same ubatches, so the same
    // arithmetic, as one read), and between pieces a request waiting for an idle slot is admitted (the watchdog's probe, max_tokens
    // 1 inside 12 s, must not wait out a 60-second prompt on the other slot; llama-server reads in n_batch pieces too). A piece
    // that reads no logits stays queued on the cards: with nobody waiting the pieces cost nothing.
    std::function<bool()> admit_one;
    const size_t kReadPiece = A.read_interleave > 0 ? (size_t) A.read_interleave
                            : (n_slots > 1 && eng.hp().n_expert == 0) ? (size_t) std::max(A.eo.n_ubatch, 4096 / A.eo.n_ubatch * A.eo.n_ubatch) : 0;
    auto start = [&](Slot & S, const json & body_in, int & http_code, bool chat_api) -> json {
        const int seq = S.id;
        S.go = false;   // not a generating slot while its prompt is read (other slots may step meanwhile)
        server_tokens & cached = S.cached;
        json body = body_in;
        std::vector<raw_buffer> files;
        std::vector<int> img_side;
        size_t img_k = 0;
        json data;
        if (chat_api) {
            try { data = oaicompat_chat_params_parse(body, chat, files); }
            catch (const std::exception & e) { http_code = 400; return format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST); }
        } else {   // /completion: llama-server's native endpoint; "prompt" may be text or token ids
            data = body;
            if (!data.contains("prompt")) { http_code = 400; return format_error_response("\"prompt\" is required", ERROR_TYPE_INVALID_REQUEST); }
        }
        server_tokens prompt;
        mtmd_context * tok_ctx = nullptr;   // the projector copy that tokenized this request (lazy: it also encodes)
        if (!files.empty()) {
            if (A.mmproj.empty()) { http_code = 400; return format_error_response("image input is not supported (no --mmproj)", ERROR_TYPE_NOT_SUPPORTED); }
            bool any_big = false;
            for (const auto & fb : files) {   // each image's longer side in pixels from its header, in prompt order (--mmproj-min-size)
                int x = 0, y = 0, c = 0;
                const int side = stbi_info_from_memory(fb.data(), (int) fb.size(), &x, &y, &c) ? std::max(x, y) : 0;
                img_side.push_back(side); any_big = any_big || side >= A.mmproj_min_size;
            }
            if (mm_lazy) {   // load only the copy this request needs: the card for big images under the context limit, else RAM
                const std::string ptxt = data.at("prompt").get<std::string>();
                const size_t est = common_tokenize(vocab, ptxt, true, true).size() + files.size() * (size_t) A.image_max_tokens;
                tok_ctx = (any_big && est <= (size_t) A.mmproj_gpu_max_ctx) ? need_gpu() : need_cpu();
            } else tok_ctx = mctx;
            if (!tok_ctx) { http_code = 500; return format_error_response("cannot load the mmproj", ERROR_TYPE_SERVER); }
            mm_last_use = now_s();
            prompt = process_mtmd_prompt(tok_ctx, data.at("prompt").get<std::string>(), files, mtmd_helper_init_opt_default());
        } else {
            auto v = tokenize_input_prompts(vocab, nullptr, data.at("prompt"), true, true, mtmd_helper_init_opt_default());
            prompt = std::move(v.at(0));
        }
        task_params tp;
        try { tp = server_schema::eval_llama_cmpl_schema(vocab, P, P.sampling.logit_bias_eog, data); }
        catch (const std::exception & e) { http_code = 400; return format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST); }
        tp.res_type = chat_api ? TASK_RESPONSE_TYPE_OAI_CHAT : TASK_RESPONSE_TYPE_NONE;
        tp.oaicompat_model = A.alias;
        tp.oaicompat_cmpl_id = gen_chatcmplid();
        const int n_ctx = n_ctx_slot, n_prompt = (int) prompt.size();
        if (n_prompt >= n_ctx) {   // --no-context-shift: a prompt that does not fit is the client's error (4xx)
            http_code = 400;
            json e = format_error_response("the request exceeds the available context size, try increasing it", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            e["n_prompt_tokens"] = n_prompt; e["n_ctx"] = n_ctx;
            return e;
        }

        // auto: a request past mmproj_gpu_max_ctx keeps its images on the CPU, and a projector on the card leaves it first,
        // so the long read and its steps get those slots back (the maintainer, 2026-10-10)
        const bool long_ctx = (mm_auto || mm_lazy) && n_prompt > A.mmproj_gpu_max_ctx;
        if (long_ctx && mctx_gpu && tok_ctx != mctx_gpu.load()) {
            mtmd_free(mctx_gpu); mctx_gpu = nullptr;
            const int s0 = eng.slots();
            eng.grow_to(slots_full, (size_t) 256 << 20);
            fprintf(stderr, "impulse: projector off the card for a %d-token prompt (%d -> %d slots per layer)\n", n_prompt, s0, eng.slots());
        }
        server_slot_stats stats;
        stats.update_prompt_start();
        // prompt cache: keep the common prefix if the window cache still holds what follows from it
        size_t keep = 0;
        if (tp.cache_prompt) {
            keep = cached.get_common_prefix(prompt);
            if (keep >= prompt.size()) keep = prompt.size() - 1;               // re-read the last token for its logits
            while (keep > 0 && prompt[keep - 1] == LLAMA_TOKEN_NULL) --keep;   // never resume inside an image
            if (!eng.can_resume((int) keep, seq)) keep = 0;                         // the window cache moved on: read again
        }
        eng.rollback((int) keep, seq);
        cached.keep_first(keep);
        stats.n_prompt_cached = keep;
        // read the rest: text runs through decode(), images through mtmd + decode_embd()
        std::vector<float> & lg = S.lg, h;
        size_t i = keep;
        for (size_t p = 0; p < keep; ++p)   // images already in the cache: the size list continues after them
            if (prompt[p] == LLAMA_TOKEN_NULL && (p == 0 || prompt[p - 1] != LLAMA_TOKEN_NULL)) img_k++;
        while (i < prompt.size()) {
            // find_next_media_chunk(i) is the first image starting strictly after i: an image starting AT i (a cache resume
            // stepped back to an image's first position) is looked up by position
            std::pair<const mtmd::input_chunk_ptr *, size_t> next = { nullptr, 0 };
            if (prompt[i] == LLAMA_TOKEN_NULL) next = { &prompt.find_chunk(i), i };
            else next = prompt.find_next_media_chunk(i);
            const size_t text_end = next.first ? next.second : prompt.size();
            if (text_end > i) {
                std::vector<int32_t> t;   // get_tokens() asserts on a prompt with media; index instead
                for (size_t j = i; j < text_end; ++j) t.push_back(prompt[j]);
                if (kReadPiece == 0 || eng.hp().n_expert > 0 || t.size() <= kReadPiece)
                    err = eng.decode(t, &lg, mtp ? &h : nullptr, seq);
                else for (size_t j = 0; j < t.size() && err.empty(); j += kReadPiece) {
                    const std::vector<int32_t> piece(t.begin() + j, t.begin() + std::min(t.size(), j + kReadPiece));
                    const bool last = j + kReadPiece >= t.size();   // only the last piece's logits (and MTP row) are read
                    err = eng.decode(piece, last ? &lg : nullptr, (last && mtp) ? &h : nullptr, seq);
                    if (err.empty() && !last) {
                        while (admit_one()) {}
                        if (A.read_interleave > 0 && others_writing(S)) step_others();
                    }
                }
                if (!err.empty()) { http_code = 500; return format_error_response("decode: " + err, ERROR_TYPE_SERVER); }
                i = text_end;
            }
            if (next.first) {
                const mtmd_input_chunk * ch = next.first->get();
                const int nt = (int) mtmd_input_chunk_get_n_tokens(ch);
                mtmd_context * enc = tok_ctx;
                const int side = img_k < img_side.size() ? img_side[img_k] : 0; img_k++;
                if (mm_gpu_always || (mm_auto && side >= A.mmproj_min_size && !long_ctx)) enc = need_gpu();
                if (!enc) { http_code = 500; return format_error_response("cannot load the mmproj on the card", ERROR_TYPE_SERVER); }
                if (enc == mctx_gpu.load()) {
                    mm_last_gpu = now_s();
                    eng.ensure_free((size_t) 256 << 20);   // a margin before the encode; slots give way, never the step
                }
                mm_last_use = now_s();
                if (mtmd_encode_chunk(enc, ch) != 0) { http_code = 500; return format_error_response("image encode failed", ERROR_TYPE_SERVER); }
                err = eng.decode_embd(mtmd_get_output_embd(enc), nt, mtmd_decode_use_non_causal(enc, ch), &lg, mtp ? &h : nullptr, seq);
                if (!err.empty()) { http_code = 500; return format_error_response("image decode: " + err, ERROR_TYPE_SERVER); }
                i = next.second + nt;
            }
        }
        cached = std::move(prompt);
        stats.n_prompt_processed = n_prompt - keep;
        stats.update_prompt_last();
        // the first sample (llama-server: the prompt's text tokens enter the sampler's history first)
        S.tp = tp; S.stats = stats; S.keep = keep; S.n_prompt = n_prompt;
        S.smpl = common_sampler_init(model, S.tp.sampling);
        if (!S.smpl) { http_code = 400; return format_error_response("bad sampling parameters", ERROR_TYPE_INVALID_REQUEST); }
        for (size_t j = 0; j < cached.size(); ++j) if (cached[j] != LLAMA_TOKEN_NULL) common_sampler_accept(S.smpl, cached[j], false);
        S.text.clear(); S.out.clear(); S.stop = STOP_TYPE_NONE; S.stopping_word.clear();
        S.n_predict = S.tp.n_predict < 0 ? INT32_MAX : S.tp.n_predict;
        S.tok = common_sampler_sample_logits(S.smpl, lg.data(), n_vocab);
        common_sampler_accept(S.smpl, S.tok, true);
        S.h_pending = h;
        http_code = 200;
        return json();
    };
    auto emit = [&](Slot & S, llama_token t) -> bool {   // false = stop here
        S.out.push_back(t); S.stats.n_gen++;
        if (llama_vocab_is_eog(vocab, t)) { S.stop = STOP_TYPE_EOS; return false; }
        // Match llama-server: parser-required markers survive even when ordinary special tokens are hidden.
        const bool special = P.special || S.tp.sampling.preserved_tokens.count(t) != 0;
        S.text += common_token_to_piece(vocab, t, special);
        for (const auto & w : S.tp.antiprompt) {
            const size_t p = S.text.find(w, S.text.size() > w.size() + 16 ? S.text.size() - w.size() - 16 : 0);
            if (!w.empty() && p != std::string::npos) { S.text.resize(p); S.stop = STOP_TYPE_WORD; S.stopping_word = w; return false; }
        }
        if ((int) S.out.size() >= S.n_predict || eng.n_past(S.id) + 2 >= n_ctx_slot) { S.stop = STOP_TYPE_LIMIT; return false; }
        return true;
    };
    auto finish = [&](Slot & S, const std::string & err) {   // the answer (or the error) to the waiting request; the slot is idle
        int code = 200; json outj;
        if (S.smpl) { common_sampler_free(S.smpl); S.smpl = nullptr; }
        if (!err.empty()) { code = 500; outj = format_error_response("generation: " + err, ERROR_TYPE_SERVER); }
        else {
            S.stats.update_gen_last();
            metrics.request((double) S.stats.n_prompt_processed, S.stats.t_prompt_ms(), (double) S.stats.n_gen, S.stats.t_gen_ms(),
                            (double) S.stats.n_draft_tokens, (double) S.stats.n_draft_accepted, (double) S.stats.n_draft_verif_steps, (double) eng.n_past(S.id));
            server_task_result_cmpl_final res;
            res.content = lite::utf8_clean(S.text);   // valid UTF-8 for the chat parser below (src/lite/utf8.hpp: it was a 500)
            res.tokens = S.out; res.stream = S.tp.stream; res.include_usage = S.tp.include_usage; res.stats = S.stats;
            res.truncated = false; res.n_decoded = (int32_t) S.out.size(); res.n_prompt_tokens = S.n_prompt;
            res.n_prompt_tokens_cache = (int32_t) S.keep; res.n_tokens_cached = eng.n_past(S.id); res.has_new_line = false;
            res.stopping_word = S.stopping_word; res.stop = S.stop; res.post_sampling_probs = false;
            res.generation_params = S.tp; res.verbose = false; res.res_type = S.tp.res_type;
            res.oaicompat_model = S.tp.oaicompat_model; res.oaicompat_cmpl_id = S.tp.oaicompat_cmpl_id;
            task_result_state state(S.tp.chat_parser_params);
            res.update(state);
            outj = res.to_json();
        }
        S.job->done.set_value({ code, outj });
        S.job.reset(); S.active = false;
        S.t_last = std::chrono::steady_clock::now().time_since_epoch().count();
    };
    // one step for every generating slot, in one ubatch: a token each, or (MTP) a draft window each
    auto step_all = [&]() {
        std::vector<Slot *> gen;
        for (auto & S : slots) if (S->job && S->go) gen.push_back(S.get());
        if (gen.empty()) return;
        metrics.step((int) gen.size());
        std::vector<g4::Engine::SeqStep> steps;
        std::vector<int> pos0(gen.size());
        for (size_t g = 0; g < gen.size(); ++g) {
            Slot & S = *gen[g];
            std::vector<int32_t> win = { S.tok };
            if (mtp) {   // drafts at one position (llama.cpp's sample-and-accept below)
                pos0[g] = eng.n_past(S.id);
                const int depth = pos0[g] >= A.n_draft_long_from ? A.n_draft_long : A.n_draft;
                const int room = std::min(depth, std::min(S.n_predict - (int) S.out.size(), n_ctx_slot - pos0[g] - 2));
                std::vector<float> hc = S.h_pending, hh;
                for (int k = 0; k < room; ++k) {
                    int32_t d = 0;
                    if (!eng.draft(win.back(), hc.data(), pos0[g], &d, &hh, S.id).empty()) break;
                    win.push_back(d); hc = hh;
                }
            }
            steps.push_back({ S.id, win });
        }
        std::vector<float> L, Hr;
        const std::string err = eng.decode_multi(steps, &L, mtp ? &Hr : nullptr);
        if (!err.empty()) { for (Slot * S : gen) finish(*S, err); return; }
        size_t row = 0;
        for (size_t g = 0; g < gen.size(); ++g) {
            Slot & S = *gen[g];
            const auto & win = steps[g].toks;
            if (!mtp) {
                S.cached.push_back(S.tok);
                S.tok = common_sampler_sample_logits(S.smpl, L.data() + row * n_vocab, n_vocab);
                common_sampler_accept(S.smpl, S.tok, true);
                S.go = emit(S, S.tok);
            } else {
                S.stats.n_draft_tokens += win.size() - 1; S.stats.n_draft_verif_steps++;
                int r = 0;
                for (;; ++r) {
                    const llama_token t = common_sampler_sample_logits(S.smpl, L.data() + (row + r) * n_vocab, n_vocab);
                    common_sampler_accept(S.smpl, t, true);
                    S.cached.push_back(win[r]);
                    S.tok = t;
                    S.go = emit(S, t);
                    if (!S.go || r + 1 >= (int) win.size() || t != win[r + 1]) break;
                    S.stats.n_draft_accepted++;
                }
                eng.rollback(pos0[g] + r + 1, S.id);
                const size_t ne = eng.hp().n_embd;
                S.h_pending.assign(Hr.begin() + (row + r) * ne, Hr.begin() + (row + r + 1) * ne);
            }
            row += win.size();
            if (!S.go) finish(S, "");
        }
    };
    step_others = step_all;
    // a queued request into an idle slot: its prompt read, its first token (false: no request waiting, or no idle slot)
    admit_one = [&]() -> bool {
        std::unique_ptr<Job> next;
        {
            std::lock_guard<std::mutex> ql(qmu);
            bool idle = false;
            for (auto & S : slots) idle = idle || !S->job;
            if (!idle || queue.empty()) return false;
            next = std::move(queue.front()); queue.pop_front();
        }
        // the idle slot whose cache shares the longest prefix with the request (llama-server: LCP similarity),
        // judged on the text (tokenized as the request will be)
        Slot * best = nullptr; size_t best_lcp = 0;
        llama_tokens probe;
        try {
            json d = next->chat_api ? json() : next->body;
            if (!next->chat_api && d.contains("prompt") && d.at("prompt").is_string())
                probe = common_tokenize(vocab, d.at("prompt").get<std::string>(), true, true);
            else if (next->chat_api) {
                std::vector<raw_buffer> fs; json b = next->body;
                json pd = oaicompat_chat_params_parse(b, chat, fs);
                if (fs.empty()) probe = common_tokenize(vocab, pd.at("prompt").get<std::string>(), true, true);
            }
        } catch (...) {}
        // llama-server's get_available_slot: the idle slot whose cache shares the largest part of the request (longer than
        // slot_prompt_similarity, 0.1, of it), else the least recently used idle slot (an unrelated request never evicts a
        // long conversation while another slot is free)
        for (auto & S : slots) if (!S->job && !S->cached.empty() && !probe.empty()) {
            size_t l = 0;
            while (l < probe.size() && l < S->cached.size() && S->cached[l] == probe[l]) ++l;
            if (l > best_lcp && (float) l / probe.size() > 0.1f) { best = S.get(); best_lcp = l; }
        }
        if (!best) for (auto & S : slots) if (!S->job && (!best || S->t_last < best->t_last)) best = S.get();
        best->job = std::move(next); best->active = true; processing = true;
        int code = 500; json r;
        try { r = start(*best, best->job->body, code, best->job->chat_api); }
        catch (const std::exception & e) { code = 500; r = format_error_response(e.what(), ERROR_TYPE_SERVER); }
        if (code != 200) {
            if (best->smpl) { common_sampler_free(best->smpl); best->smpl = nullptr; }
            best->job->done.set_value({ code, r }); best->job.reset(); best->active = false;
        } else {
            best->go = emit(*best, best->tok);
            if (!best->go) finish(*best, "");
        }
        return true;
    };
    std::thread([&]() {   // the scheduler: admit queued requests to idle slots, read their prompts, step every slot
        for (;;) {
            {
                std::unique_lock<std::mutex> ql(qmu);
                bool any_job = false;
                for (auto & S : slots) any_job = any_job || S->job;
                if (!any_job) qcv.wait(ql, [&] { return !queue.empty(); });
            }
            std::lock_guard<std::mutex> g(busy);
            while (admit_one()) {}
            step_all();
            bool any = false; for (auto & S : slots) any = any || S->job;
            processing = any;
        }
    }).detach();

    // ---------------------------------------------------------------- HTTP
    httplib::Server srv;
    {
        const std::string e = A.keys.add_env();   // LLAMA_API_KEY / LLAMA_ARG_API_KEY_FILE, as llama-server reads them
        if (!e.empty()) { fprintf(stderr, "error: %s\n", e.c_str()); return 1; }
        lite::install_api_keys<httplib::Server, httplib::Request, httplib::Response>(srv, A.keys);
        if (A.keys.enabled()) fprintf(stderr, "impulse: API key required (%zu key%s); /health stays open\n", A.keys.keys.size(), A.keys.keys.size() == 1 ? "" : "s");
    }
    const auto t_created = std::time(nullptr);
    auto send_json = [](httplib::Response & r, const json & j, int code) { r.status = code; r.set_content(safe_json_to_str(j), "application/json; charset=utf-8"); };
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & r) { send_json(r, json{ { "status", "ok" } }, 200); });
    srv.Get("/", [&](const httplib::Request &, httplib::Response & r) {
        if (!A.webui) { r.status = 404; return; }
        r.set_content(lite::chat_page(), "text/html; charset=utf-8");
    });
    srv.Get("/metrics", [&](const httplib::Request &, httplib::Response & r) {   // llama-server's, with --metrics
        if (!A.metrics) {
            send_json(r, json{ { "error", format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED) } }, 501);
            return;
        }
        int busy_n = 0; for (auto & S : slots) busy_n += S->active.load() ? 1 : 0;
        size_t waiting = 0; { std::lock_guard<std::mutex> ql(qmu); waiting = queue.size(); }
        r.set_content(metrics.text(busy_n, (int) waiting), "text/plain; version=0.0.4");
    });
    // llama-server's get_res_models: the Ollama-style list and the OpenAI data (meta.n_ctx = one slot's)
    const lite::ModelInfo minfo = lite::model_info(A.model);
    auto models = [&](const httplib::Request &, httplib::Response & r) {
        json meta = { { "vocab_type", llama_vocab_type(vocab) }, { "n_vocab", n_vocab }, { "n_ctx", n_ctx_slot }, { "n_ctx_train", eng.hp().n_ctx_train },
                      { "n_embd", eng.hp().n_embd }, { "n_params", minfo.n_params }, { "size", minfo.size }, { "ftype", minfo.ftype } };
        json caps = !A.mmproj.empty() ? json::array({ "completion", "multimodal" }) : json::array({ "completion" });
        send_json(r, json{ { "models", json::array({ json{ { "name", A.alias }, { "model", A.alias }, { "modified_at", "" }, { "size", "" }, { "digest", "" },
                                                           { "type", "model" }, { "description", "" }, { "tags", json::array({ "" }) }, { "capabilities", caps },
                                                           { "parameters", "" },
                                                           { "details", json{ { "parent_model", "" }, { "format", "gguf" }, { "family", "" }, { "families", json::array({ "" }) },
                                                                              { "parameter_size", "" }, { "quantization_level", "" } } } } }) },
                           { "object", "list" },
                           { "data", json::array({ json{ { "id", A.alias }, { "aliases", json::array({ A.alias }) }, { "tags", json::array() }, { "object", "model" },
                                                         { "created", t_created }, { "owned_by", "llamacpp" }, { "meta", meta } } }) } }, 200);
    };
    srv.Get("/v1/models", models);
    srv.Get("/models", models);
    srv.Get("/props", [&](const httplib::Request &, httplib::Response & r) {
        task_params tparams; tparams.sampling = P.sampling;
        json props = {
            { "default_generation_settings", json{ { "params", tparams.to_json(true) }, { "n_ctx", n_ctx_slot } } },
            { "total_slots", n_slots }, { "model_alias", A.alias }, { "model_ftype", minfo.ftype }, { "model_path", A.model },
            { "modalities", json{ { "vision", chat.allow_image }, { "video", false }, { "audio", false } } },
            { "media_marker", get_media_marker() }, { "endpoint_slots", true }, { "endpoint_props", false }, { "endpoint_metrics", true },
            { "chat_template", common_chat_templates_source(chat.tmpls.get(), "") },
            { "chat_template_caps", common_chat_templates_get_caps(chat.tmpls.get()) },
            { "bos_token", llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL ? common_token_to_piece(vocab, llama_vocab_bos(vocab), true) : std::string() },
            { "eos_token", llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL ? common_token_to_piece(vocab, llama_vocab_eos(vocab), true) : std::string() },
            { "ui", false }, { "cors_proxy_enabled", false },
            { "build_info", std::string("impulse " IMPULSE_VERSION_STR " (gemma4 engine) / ") + llama_build_info() }, { "is_sleeping", false },
            { "impulse", json{ { "slots", eng.slots() }, { "arena_shrinks", eng.stats().arena_shrinks }, { "graphs_built", eng.stats().graphs_built }, { "graphs_reused", eng.stats().graphs_reused }, { "n_cpu_moe", A.eo.n_cpu_moe },
                              { "prefill_ubatch", A.eo.prefill_ubatch < 0 ? json("auto") : json(A.eo.prefill_ubatch) }, { "last_chunk", eng.stats().last_chunk }, { "lend", A.eo.lend }, { "mtp", mtp }, { "vision", !A.mmproj.empty() }, { "projector_in_ram", mctx.load() != nullptr && !mm_gpu_always },
                              { "projector", A.mmproj.empty() ? "none" : A.mmproj_place }, { "projector_on_card", mctx_gpu != nullptr },
                              { "projector_moves", mm_moves.load() }, { "slots_full", slots_full },
                              { "projector_min_size", A.mmproj_min_size }, { "projector_gpu_max_ctx", A.mmproj_gpu_max_ctx },
                              { "projector_idle_s", A.mmproj_idle },
                              { "adapt", A.eo.adapt }, { "adapt_half_life", A.eo.adapt_half_life }, { "adapt_budget", A.eo.adapt_budget },
                              { "adapt_refill", A.eo.adapt_refill }, { "async_swaps", eng.async_swaps() }, { "ram_locked_gb", eng.ram_locked() / 1e9 }, { "ram_experts_gb", eng.ram_experts() / 1e9 }, { "overlap", A.eo.overlap },
                              { "threads", A.threads }, { "pool_threads", A.eo.pool_threads }, { "n_draft", A.n_draft },
                              { "n_draft_long", A.n_draft_long }, { "n_draft_long_from", A.n_draft_long_from } } },
        };
        send_json(r, props, 200);
    });
    srv.Get("/slots", [&](const httplib::Request &, httplib::Response & r) {
        json arr = json::array();
        for (auto & S : slots) arr.push_back(json{ { "id", S->id }, { "n_ctx", n_ctx_slot }, { "speculative", mtp }, { "is_processing", S->active.load() } });
        send_json(r, arr, 200);
    });
    auto completions_any = [&](const httplib::Request & req, httplib::Response & r, bool chat_api) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { send_json(r, json{ { "error", format_error_response("invalid JSON body", ERROR_TYPE_INVALID_REQUEST) } }, 400); return; }
        auto job = std::make_unique<Job>();
        job->body = body; job->chat_api = chat_api;
        auto fut = job->done.get_future();
        { std::lock_guard<std::mutex> ql(qmu); queue.push_back(std::move(job)); }
        qcv.notify_one();
        auto [code, out] = fut.get();
        if (code == 200 && out.is_array()) {   // stream: the whole answer as one SSE burst (no the reference server client streams)
            std::string sse;
            for (const auto & ev : out) sse += format_oai_sse(ev);
            sse += "data: [DONE]\n\n";
            r.status = 200; r.set_content(sse, "text/event-stream");
            return;
        }
        send_json(r, code == 200 ? out : json{ { "error", out } }, code);   // llama-server wraps errors as {"error": {...}}
    };
    auto completions = [&](const httplib::Request & req, httplib::Response & r) { completions_any(req, r, true); };
    auto completion = [&](const httplib::Request & req, httplib::Response & r) { completions_any(req, r, false); };
    srv.Post("/v1/chat/completions", completions);
    srv.Post("/chat/completions", completions);
    srv.Post("/completion", completion);
    srv.Post("/completions", completion);
    // llama-server's /tokenize and /detokenize (tools/server: content, add_special (false), parse_special (true), with_pieces;
    // tokens): clients that count or cut their prompts call them
    srv.Post("/tokenize", [&](const httplib::Request & q, httplib::Response & r) {
        json body; try { body = json::parse(q.body); } catch (...) { send_json(r, json{ { "error", "invalid JSON" } }, 400); return; }
        const std::string content = body.value("content", std::string());
        const auto toks = common_tokenize(vocab, content, body.value("add_special", false), body.value("parse_special", true));
        json out = json::array();
        if (body.value("with_pieces", false)) {
            for (auto t : toks) {
                const std::string piece = common_token_to_piece(vocab, t, true);
                out.push_back(json{ { "id", t }, { "piece", lite::utf8_clean(piece) == piece ? json(piece) : json(std::vector<int>(piece.begin(), piece.end())) } });
            }
        } else for (auto t : toks) out.push_back(t);
        send_json(r, json{ { "tokens", out } }, 200);
    });
    srv.Post("/detokenize", [&](const httplib::Request & q, httplib::Response & r) {
        json body; try { body = json::parse(q.body); } catch (...) { send_json(r, json{ { "error", "invalid JSON" } }, 400); return; }
        std::vector<llama_token> toks;
        for (const auto & t : body.value("tokens", json::array())) toks.push_back(t.get<llama_token>());
        send_json(r, json{ { "content", lite::utf8_clean(common_detokenize(vocab, toks, true)) } }, 200);
    });
    srv.set_read_timeout(A.timeout); srv.set_write_timeout(A.timeout);
    // llama-server's socket options: SO_REUSEADDR (a restart binds its port while the last run's connections are in
    // TIME_WAIT), and not cpp-httplib's default SO_REUSEPORT (a second server on the same port would share its connections)
    srv.set_socket_options([](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    if (!compat_notes.empty()) {
        std::string n; for (const auto & x : compat_notes) n += (n.empty() ? "" : ", ") + x;
        fprintf(stderr, "impulse: llama-server flags accepted, no effect here: %s\n", n.c_str());
    }
    fprintf(stderr, "impulse-server: listening on http://%s:%d (alias %s, ctx %d, vision %d, mtp %d)\n",
            A.host.c_str(), A.port, A.alias.c_str(), eng.n_ctx(), (int) chat.allow_image, (int) mtp);
    if (mm_auto || mm_lazy) std::thread([&]() {   // after mmproj_idle seconds: off the card (auto, lazy), out of RAM (lazy)
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const int64_t now = now_s();
            const bool gpu_idle = mctx_gpu && now - (mm_lazy ? mm_last_use.load() : mm_last_gpu.load()) >= A.mmproj_idle;
            const bool ram_idle = mm_lazy && mctx && now - mm_last_use.load() >= A.mmproj_idle;
            if (!gpu_idle && !ram_idle) continue;
            std::lock_guard<std::mutex> g(busy);
            if (gpu_idle && mctx_gpu) {
                mtmd_free(mctx_gpu); mctx_gpu = nullptr;
                const int s0 = eng.slots();
                eng.grow_to(slots_full, (size_t) 256 << 20);
                fprintf(stderr, "impulse: projector off the card after %d s idle (%d -> %d slots per layer)\n", A.mmproj_idle, s0, eng.slots());
            }
            if (ram_idle && mctx) { mtmd_free(mctx); mctx = nullptr; fprintf(stderr, "impulse: projector out of RAM after %d s idle\n", A.mmproj_idle); }
        }
    }).detach();
    if (!srv.listen(A.host, A.port)) {
        // exit now: the scheduler and projector threads run detached over this function's objects, so an ordinary return would
        // tear those down under them (the process hung, holding the cards, when the port was still taken)
        fprintf(stderr, "impulse-server: cannot listen on %s:%d\n", A.host.c_str(), A.port);
        fflush(stderr);
        std::_Exit(1);
    }
    if (mctx_gpu && mctx_gpu.load() != mctx.load()) mtmd_free(mctx_gpu);
    if (mctx) mtmd_free(mctx);
    llama_model_free(model);
    return 0;
}

}  // namespace g4srv

// A Flash-Next pack runs in impulse-fx (src/fx/: the server front with no CUDA of its own, and the engine library), which
// this process becomes (exec, the same arguments; on Windows it runs it and returns its exit code): a pack's process then
// holds the engine's CUDA code alone, as the standalone `strata` does ($IMPULSE_FX_SERVER: another front binary).
static int exec_fx_front(int argc, char ** argv) {
    std::string exe;
    if (const char * e = getenv("IMPULSE_FX_SERVER")) exe = e;
    else exe = lite::self_dir() + "impulse-fx"
#if defined(_WIN32)
        ".exe"
#endif
        ;
    std::vector<char *> av = { (char *) exe.c_str() };
    for (int i = 1; i < argc; ++i) av.push_back(argv[i]);
    av.push_back(nullptr);
    fflush(stdout); fflush(stderr);
#if defined(_WIN32)
    const intptr_t rc = _spawnv(_P_WAIT, exe.c_str(), av.data());
    if (rc != -1) return (int) rc;
#else
    execv(exe.c_str(), av.data());
#endif
    fprintf(stderr, "impulse-server: a Flash-Next pack needs %s (a build with -DIMPULSE_FLASHNEXT=ON, or set IMPULSE_FX_SERVER)\n", exe.c_str());
    return 1;
}

#ifdef IMPULSE_COMBINED
namespace q35srv { int run(int argc, char ** argv); }
int main(int argc, char ** argv) {
    // src/fx/fx_pack.hpp: a Flash-Next pack (a folder with pack/ and a *dense*.gguf, or that GGUF beside pack/) -> impulse-fx;
    // a GGUF alone -> its architecture's engine, as before
    const std::string engine = fx::engine_for(fx::model_arg(argc, argv));
    if (engine == "fx") return exec_fx_front(argc, argv);
    if (engine == "q35") return q35srv::run(argc, argv);
    return g4srv::run(argc, argv);
}
#else
int main(int argc, char ** argv) {
    if (fx::engine_for(fx::model_arg(argc, argv)) == "fx") return exec_fx_front(argc, argv);
#ifndef _WIN32
    if (const auto arch = g4srv::model_architecture(argc, argv); arch == "qwen35moe" || arch == "qwen35" || arch == "lfm2moe" || arch == "lfm2" || arch == "qwen4exp") g4srv::exec_qwen_server(argc, argv);
#endif
    return g4srv::run(argc, argv);
}
#endif
