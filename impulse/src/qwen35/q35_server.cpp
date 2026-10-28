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
// impulse-server-qwen35: the Impulse server for the qwen35moe engine (Qwen3.5 / 3.6 MoE), g4_server.cpp's
// request handling around q35::Engine. What differs from the gemma4 server: MTP is the engine's (draft-mtp with its own
// KV cache; the drafts advance a position each); images take M-RoPE positions (mtmd's decoder positions, n_pos per
// chunk); and the prompt cache resumes only where the recurrent state can return (q35::Engine::resume_point: a
// checkpoints of the recurrent state where the next request is likely to resume).
//
// impulse-server (gemma4): an OpenAI-compatible server for the gemma4 engine, API-identical to llama-server for what
// the reference server's clients use (docs/API.md): POST /v1/chat/completions (and /chat/completions), GET /health,
// /v1/models (/models), /props, /slots.
//
// llama-server's own code does the request parsing (oaicompat_chat_params_parse: chat template, tools, response_format
// -> grammar, chat_template_kwargs, image parts -> media markers), the image tokenization (process_mtmd_prompt), the
// sampling parameters (eval_llama_cmpl_schema), the sampler chain with its grammar and reasoning budget (common_sampler,
// through common_sampler_sample_logits) and the response JSON (server_task_result_cmpl_final). The engine does the
// compute: text and image chunks, MTP verify windows (llama.cpp's sample-and-accept), the prompt cache.
// One request at a time (the reference server runs --parallel 1).
#include "q35_engine.hpp"

#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"
#include "../lite/stream_out.hpp"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <list>
#include <functional>
#include <future>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#ifdef IMPULSE_VERSION
#define IMPULSE_VERSION_STR IMPULSE_VERSION
#else
#define IMPULSE_VERSION_STR "dev"
#endif

namespace q35srv {

struct Args {
    lite::ApiKeys keys;   // --api-key / --api-key-file (llama-server's): none = no check
    int timeout = 7200; bool metrics = false, webui = true;   // llama-server's --timeout, --metrics, --no-webui
    std::string model, mmproj, mtp, alias = "impulse", host = "127.0.0.1", profile;
    float spec_min_p = 0.0f;   // drafts only while the drafter gives its pick at least this probability (Strata's --spec-min-p; 0: off)
    int port = 8080, n_draft = 2, image_max_tokens = 384, threads = 0;   // 384: the maintainer, 2026-10-10; threads 0 = physical cores; n_draft 2: +21 % short decode over 3 on a 3060 (v0.2.2)
    bool reserve_set = false;
    std::string mmproj_place;   // second: on a second visible GPU (default when one has room); gpu: always on the engine's card;
                                // cpu: in RAM; auto: in RAM, on the card while images need it;
                                // lazy: not loaded until an image comes (RAM or card by the rules), unloaded when idle. Default: lazy
                                // under 32 GB of RAM, auto above
    int mmproj_idle = 15, mmproj_min_size = 128, mmproj_gpu_max_ctx = 65536;   // 15 s, 64k: the maintainer, 2026-10-10
    q35::Options eo;
    bool mtp_own_head = false;
    int  read_interleave = 0;   // slots: other slots step after each N prompt tokens of a dense read (0 = they wait for the read)
    long long cache_ram_mib = 0; // the RAM prompt cache (llama-server's --cache-ram, MiB): conversations parked in RAM; 0 = off
    int  ckpt_min = 16384;       // prompt reads this long also checkpoint the last user message's start and 512 tokens from the end   // the drafter file's own token_embd / output (default: the target's, 0.5 GB less VRAM)
    common_params params;
};

static void usage() {
    fprintf(stderr, "impulse-server (qwen35) -m MODEL [--mmproj F] [--mtp F] [--alias A] [--host H] [--port P] [--api-key K[,K]] [--api-key-file F] [-c CTX] [-ub N]\n"
                    "  [--mmproj-place second|auto|lazy|gpu|cpu --mmproj-min-size PX --mmproj-gpu-max-ctx N --mmproj-idle S]\n"
                    "  [--adapt-half-life T --adapt-budget N --no-adapt --no-adapt-refill --no-async-swaps --no-overlap --no-lend]\n"
                    "  [--ram-budget GIB] (experts locked in RAM; the rest read from the model file; 0 = low-RAM mode; default: what the PC can spare)\n"
                    "  [--temp T --top-k K --top-p P --min-p M --repeat-penalty R] [--image-max-tokens N] [-t N]\n"
                    "  [--n-cpu-moe N --cache-slots S --profile F --pool-threads N --prefill-ubatch N -ctk T -ctv T --n-draft N --mtp-own-head]\n"
                    "  [--checkpoint-min-tokens N] [-ts A,B -sm layer|tensor] (dense models over several GPUs, llama.cpp's shares) [-np N] (slots)\n");
    fprintf(stderr, "  [--lora FILE --lora-scale SCALE] (LFM rank-one output adapters; fixed at startup)\n");
}

static std::vector<float> parse_split(const std::string & v) {   // "52,48,0": llama.cpp's --tensor-split shares
    std::vector<float> r; size_t a = 0;
    while (a <= v.size()) { size_t b = v.find_first_of(",/", a); if (b == std::string::npos) b = v.size(); r.push_back(std::stof(v.substr(a, b - a))); a = b + 1; }
    return r;
}

static ggml_type kv_type(const std::string & t) { return t == "q4_0" ? GGML_TYPE_Q4_0 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }

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
    A.eo.n_ctx = 262144; A.eo.n_cpu_moe = 40; A.eo.pool = true; A.eo.pool_threads = 0; A.eo.cache_slots = -1;
    A.eo.adapt = true; A.eo.async_swaps = true; A.eo.overlap = true; A.eo.prefill_ubatch = -1; A.eo.lend = true;
    A.eo.kv_initial = 32768;   // the KV pager: 32k cells on the card at load, grown as the conversation needs (0 = all at load)
    A.eo.type_k = GGML_TYPE_Q4_0; A.eo.type_v = GGML_TYPE_Q4_0;
    bool ctx_set = false, ckpt_set = false, k_set = false, v_set = false, temp_set = false, topk_set = false, repeat_set = false, topp_set = false, minp_set = false, ncpumoe_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { usage(); exit(2); } return std::string(argv[++i]); };
        if (a == "-m" || a == "--model") A.model = nx();
        else if (a == "--lora" || a == "--lora-scaled") {
            const std::string e = q35::add_lora_arg(A.eo, nx(), a == "--lora-scaled");
            if (!e.empty()) { fprintf(stderr, "%s\n", e.c_str()); return 2; }
        }
        else if (a == "--device" || a == "-dev") { const std::string d = nx(); if (d == "cpu" || d == "none") A.eo.cpu_only = true; }
        else if ((a == "-ngl" || a == "--n-gpu-layers" || a == "--gpu-layers")) { if (std::stoi(nx()) == 0) A.eo.cpu_only = true; }
        else if (a == "--lora-scale") A.eo.lora_scale = std::stof(nx());
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
        else if (a == "-c" || a == "--ctx-size") { A.eo.n_ctx = std::stoi(nx()); ctx_set = true; }
        else if (a == "-ub" || a == "--ubatch-size") A.eo.n_ubatch = std::stoi(nx());
        else if (a == "--temp") { P.sampling.temp = std::stof(nx()); temp_set = true; }
        else if (a == "--top-k") { P.sampling.top_k = std::stoi(nx()); topk_set = true; }
        else if (a == "--top-p") { P.sampling.top_p = std::stof(nx()); topp_set = true; }
        else if (a == "--min-p") { P.sampling.min_p = std::stof(nx()); minp_set = true; }
        else if (a == "--repeat-penalty") { P.sampling.penalty_repeat = std::stof(nx()); repeat_set = true; }
        else if (a == "--image-max-tokens") A.image_max_tokens = std::stoi(nx());
        else if (a == "--mmproj-place") A.mmproj_place = nx();
        else if (a == "--mmproj-idle") A.mmproj_idle = std::stoi(nx());
        else if (a == "--mmproj-min-size") A.mmproj_min_size = std::stoi(nx());   // px: smaller images encode on the CPU (auto)
        else if (a == "--mmproj-gpu-max-ctx") A.mmproj_gpu_max_ctx = std::stoi(nx());
        else if (a == "-t" || a == "--threads") A.threads = std::stoi(nx());
        else if (a == "--n-cpu-moe") { A.eo.n_cpu_moe = std::stoi(nx()); ncpumoe_set = true; }
        else if (a == "--tensor-split" || a == "-ts") A.eo.tensor_split = parse_split(nx());   // several GPUs (dense models)
        else if (a == "--read-interleave") {
            A.read_interleave = std::stoi(nx());
            if (A.read_interleave % std::max(1, A.eo.n_ubatch) != 0) { fprintf(stderr, "--read-interleave: a multiple of -ub\n"); return 2; }
        }
        else if (a == "-np" || a == "--parallel") { A.eo.n_seq = std::stoi(nx()); P.n_parallel = A.eo.n_seq; }   // slots (llama-server's)
        else if (a == "--split-mode" || a == "-sm") {   // layer (llama.cpp's default with several GPUs) or tensor
            const std::string v = nx();
            if (v == "tensor") A.eo.split_tensor = true; else if (v != "layer") { fprintf(stderr, "--split-mode layer|tensor\n"); return 2; }
        }
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
        else if (a == "-ctk") { A.eo.type_k = kv_type(nx()); k_set = true; }
        else if (a == "-ctv") { A.eo.type_v = kv_type(nx()); v_set = true; }
        else if (a == "--n-draft") A.n_draft = std::stoi(nx());
        else if (a == "--spec-min-p" || a == "--draft-p-min") A.spec_min_p = std::stof(nx());
        else if (a == "--mtp-own-head") A.mtp_own_head = true;
        else if (a == "--checkpoint-min-tokens") { A.ckpt_min = std::stoi(nx()); ckpt_set = true; }
        else if (a == "--cache-ram") A.cache_ram_mib = std::stoll(nx());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    if (A.model.empty()) { usage(); return 2; }
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
    A.eo.mtp = A.mtp; A.eo.mtp_share_head = !A.mtp_own_head;
    A.eo.n_rs_seq = A.mtp.empty() ? 0 : A.n_draft;   // llama.cpp: n_rs_seq = the draft max
    P.model.path = A.model;

    llama_backend_init();
    // the vocabulary, the chat template and the tokenizer come from llama.cpp (vocab-only load: no weights)
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(A.model.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load the vocabulary\n"); return 1; }
    char architecture[64] = {};
    llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
    const bool lfm2 = std::string(architecture) == "lfm2moe" || std::string(architecture) == "lfm2";
    if (lfm2) {
        if (!A.mtp.empty() || !A.mmproj.empty()) { fprintf(stderr, "LFM2 is text-only and has no supported MTP head\n"); llama_model_free(model); return 2; }
        if (!ctx_set) {
            // A vocab-only llama_model has no populated hparams; read the retained GGUF metadata.
            char context[32] = {};
            if (llama_model_meta_val_str(model, (std::string(architecture) + ".context_length").c_str(), context, sizeof(context)) <= 0 || std::atoi(context) <= 0) {
                fprintf(stderr, "missing LFM2 context_length\n"); llama_model_free(model); return 2;
            }
            A.eo.n_ctx = std::atoi(context);
        }
        if (!k_set) A.eo.type_k = GGML_TYPE_F16;
        if (!v_set) A.eo.type_v = GGML_TYPE_F16;
        if (!temp_set) P.sampling.temp = 0.1f;
        if (!topk_set) P.sampling.top_k = 50;
        if (!repeat_set) P.sampling.penalty_repeat = 1.05f;
    }
    const bool flint = std::string(architecture) == "flint";
    if (flint) {   // text only, no MTP head; the model card's sampling (temperature 1, top-p 0.95, top-k 20, min-p 0)
        if (!A.mtp.empty() || !A.mmproj.empty()) { fprintf(stderr, "Flint is text-only and has no MTP head\n"); llama_model_free(model); return 2; }
        if (!temp_set) P.sampling.temp = 1.0f;
        if (!topk_set) P.sampling.top_k = 20;
        if (!topp_set) P.sampling.top_p = 0.95f;
        if (!minp_set) P.sampling.min_p = 0.0f;
        if (!ncpumoe_set) {   // every layer's experts in RAM, the cache in front of them (49 blocks: the default 40 would leave 9 whole)
            char blocks[32] = {};
            if (llama_model_meta_val_str(model, "flint.block_count", blocks, sizeof(blocks)) > 0) A.eo.n_cpu_moe = std::atoi(blocks);
        }
    }
    P.n_ctx = A.eo.n_ctx;
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
    if (A.mmproj_place != "second" && A.mmproj_place != "gpu" && A.mmproj_place != "cpu" && A.mmproj_place != "auto" && A.mmproj_place != "lazy" &&
        A.mmproj_place != "last") {
        fprintf(stderr, "--mmproj-place second|gpu|cpu|auto|lazy|last\n"); return 1;
    }
    if (A.mmproj_place == "second" && !A.mmproj.empty() && !second_fits) {
        fprintf(stderr, "--mmproj-place second: %s\n", second_dev ? "the second GPU has too little free memory" : "no second GPU is visible"); return 1;
    }
    const bool mm_gpu_always = !A.mmproj.empty() && A.mmproj_place == "gpu", mm_auto = !A.mmproj.empty() && A.mmproj_place == "auto";
    const bool mm_lazy = !A.mmproj.empty() && A.mmproj_place == "lazy";
    const bool mm_second = !A.mmproj.empty() && A.mmproj_place == "second";
    // last: on the last visible GPU, which also holds layers (a model split over every card), loaded before the engine so
    // that card's expert arena is sized around it; resident, no moves
    const bool mm_last = !A.mmproj.empty() && A.mmproj_place == "last";
    ggml_backend_dev_t last_dev = nullptr;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d)
        if (ggml_backend_dev_type(ggml_backend_dev_get(d)) == GGML_BACKEND_DEVICE_TYPE_GPU) last_dev = ggml_backend_dev_get(d);
    if (mm_last && !last_dev) { fprintf(stderr, "--mmproj-place last: no GPU\n"); return 1; }
    const size_t mmproj_card = (fsize(A.mmproj) >> 20) + 768;   // the projector on the card + its 1024 px encode, MiB
    // the engine loads the drafter itself before it sizes the arena: what comes after is the projector, the drafter's
    // compute buffers and step headroom
    const size_t after_mb = 384 + (mm_gpu_always ? mmproj_card - 256 : 0);
    if (A.eo.cache_slots < 0 && !A.reserve_set) A.eo.vram_reserve_mb = (int) after_mb;
    mtmd_context * mctx_last = nullptr;   // --mmproj-place last: made here, before the engine
    if (mm_last) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = true; mparams.device = last_dev;
        mparams.print_timings = false; mparams.n_threads = A.threads > 0 ? A.threads : lite::physical_cores_or_half();
        mparams.image_max_tokens = A.image_max_tokens; mparams.media_marker = get_media_marker();
        mctx_last = mtmd_init_from_file(A.mmproj.c_str(), model, mparams);
        if (!mctx_last) { fprintf(stderr, "cannot load the mmproj\n"); return 1; }
    }
    q35::Engine eng;
    std::string err = eng.load(A.eo);
    if (!err.empty()) { fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
    const bool mtp = !A.mtp.empty();
    if (A.eo.cache_slots >= 0) eng.ensure_free(after_mb << 20, true);
    // a dense model streams no experts, so a checkpoint cut costs nothing: every read takes the full set (the last user
    // message's start and 512 tokens from the end too, as llama-server's checkpoints)
    if (eng.hp().dense && !ckpt_set) A.ckpt_min = 0;   // fixed slots that leave no room: fewer

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
    if (mm_last) {   // its encode buffers now (the largest image), then the arena of its card makes room for them
        mctx = mctx_last; mm_warm(mctx);
        for (int d = 0; d < eng.n_cards(); ++d) if (eng.card_dev(d) == last_dev) eng.ensure_free((size_t) 384 << 20, true, d);
        fprintf(stderr, "impulse: projector on the last GPU (%s), with that card's layers\n", ggml_backend_dev_description(last_dev));
    } else if (!A.mmproj.empty() && !mm_lazy) {
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
    if (eng.n_cards() > 1) {
        std::string m;
        for (int d = 0; d < eng.n_cards(); ++d) m += (d ? ", " : "") + std::string(ggml_backend_dev_name(eng.card_dev(d))) + " " + std::to_string(eng.slots_on(d));
        fprintf(stderr, "impulse: expert slots per layer by card: %s\n", m.c_str());
    }

    // the ChatML turn marker of Qwen's template (a checkpoint goes before the last one in a prompt)
    llama_token im_start = LLAMA_TOKEN_NULL;
    { const auto v = common_tokenize(vocab, "<|im_start|>", false, true); if (v.size() == 1) im_start = v[0]; }
    server_chat_params chat;
    {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, P.chat_template);
        const bool thinking = P.enable_reasoning != 0 && common_chat_templates_support_enable_thinking(tmpls.get());
        chat = { P.use_jinja, P.prefill_assistant, P.reasoning_format, P.default_template_kwargs, std::move(tmpls),
                 !A.mmproj.empty(), false, false, thinking,
                 P.sampling.reasoning_budget_tokens, P.sampling.reasoning_budget_message, P.media_path, P.force_pure_content_parser };
    }

    // ---------------------------------------------------------------- generation: slots (llama-server's --parallel)
    // Each slot is one sequence of the engine (its own KV cells, recurrent cell, prompt cache, checkpoints, sampler and MTP
    // state); its context is n_ctx / slots (llama-server's n_ctx_slot). A scheduler thread owns the engine: a queued request
    // goes to the idle slot whose prompt cache shares the longest prefix with it, its prompt is read, then every generating
    // slot steps together, one token each (or one MTP verify window each) per step (Engine::decode_multi). With one slot the
    // engine's arithmetic is the one-request server's, call for call.
    std::mutex busy;                      // the engine's owner: the scheduler, and the projector's idle thread
    std::atomic<bool> processing{ false };
    const int n_slots = A.eo.n_seq, n_ctx_slot = eng.n_ctx() / n_slots;
    static const bool timeline = getenv("IMPULSE_TIMELINE") != nullptr;   // per request: engine vs sampler time a token
    // v0.2.4-exact1 (Strata-Lite dfb7bd6e): IMPULSE_LAZY_LOGITS=1 (or its Strata-Lite name STRATA_LITE_LAZY_LOGITS=1, which
    // Dense's definition sets) reads a step's logit rows one at a time into pinned memory, only those the accept loop
    // reaches (the same floats, so the same tokens)
    static const bool lazy_logits = [] {
        for (const char * k : { "IMPULSE_LAZY_LOGITS", "STRATA_LITE_LAZY_LOGITS" })
            if (const char * v = getenv(k)) return atoi(v) != 0;
        return false;
    }();
    static const bool probes = timeline && atoi(getenv("IMPULSE_TIMELINE")) >= 2;   // and: the sampler a second time on the
    // same logits, a read pass over the logits, a write pass over 1.5 MB (is the core slow, or the call?); the second sample
    // advances a sampled request's RNG, so only for measuring
    struct Job { json body; bool chat_api = false; std::promise<std::pair<int, json>> done;
                 std::shared_ptr<lite::StreamChan> chan; };   // chan: stream: true (src/lite/stream_out.hpp)
    struct Slot {
        int id = 0;
        server_tokens cached;             // the slot's prompt cache: what its sequence holds (prompt + generated)
        std::unique_ptr<Job> job;         // the request being answered (null: idle)
        std::atomic<bool> active{ false };
        task_params tp; common_sampler * smpl = nullptr; server_slot_stats stats;
        std::string text, stopping_word; llama_tokens out; stop_type stop = STOP_TYPE_NONE;
        std::unique_ptr<task_result_state> sst; size_t n_sent = 0;   // a streamed answer: its parser state, the text sent
        int n_predict = 0, n_prompt = 0; size_t keep = 0; llama_token tok = 0; bool go = false;
        std::vector<float> lg;
        int64_t t_last = 0;               // when its last request finished (the least recently used idle slot takes a new one)
        double tl_eng = 0, tl_smp = 0, tl_acc = 0, tl_emit = 0, tl_smp2 = 0, tl_read = 0, tl_read2 = 0, tl_write = 0, tl_write2 = 0; int tl_n = 0;
        std::vector<float> scratch;
    };
    std::vector<std::unique_ptr<Slot>> slots;
    for (int k = 0; k < n_slots; ++k) { slots.emplace_back(new Slot()); slots.back()->id = k; }
    // the RAM prompt cache (--cache-ram): conversations parked in host memory, most recently used first, within the budget.
    // A request that would replace most of a slot's conversation parks it first; a parked conversation that shares more of
    // a request's prompt than the slot comes back into the slot (Strata's conversation cache; llama-server's cache-ram)
    struct ParkedConv { server_tokens tokens; std::shared_ptr<q35::Engine::Parked> st; };
    std::list<ParkedConv> parked;
    size_t parked_total = 0;
    const size_t parked_budget = A.cache_ram_mib > 0 ? (size_t) A.cache_ram_mib << 20 : 0;
    const size_t kParkMin = 1024;   // shorter conversations are cheaper to read again
    auto park_slot = [&](Slot & S) {
        const auto t0 = std::chrono::steady_clock::now();
        auto st = eng.park(S.id);
        if (!st) return;
        const size_t b = q35::Engine::parked_bytes(*st);
        if (b > parked_budget) { S.cached.clear(); return; }
        for (auto it = parked.begin(); it != parked.end();) {   // an older copy of this conversation (a prefix of it) goes
            if (it->tokens.size() <= S.cached.size() && it->tokens.get_common_prefix(S.cached) == it->tokens.size()) {
                parked_total -= q35::Engine::parked_bytes(*it->st); it = parked.erase(it);
            } else ++it;
        }
        parked.push_front(ParkedConv{ std::move(S.cached), st });
        parked_total += b;
        S.cached.clear();
        fprintf(stderr, "impulse: parked a %d-token conversation in RAM (%.2f GB, %.0f ms); %zu parked, %.1f of %.1f GB\n",
                q35::Engine::parked_pos(*st), b / 1e9, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                parked.size(), parked_total / 1e9, parked_budget / 1e9);
    };
    auto parked_evict = [&]() {
        while (parked_total > parked_budget && !parked.empty()) {
            parked_total -= q35::Engine::parked_bytes(*parked.back().st);
            parked.pop_back();
        }
    };
    lite::Metrics metrics;   // /metrics (--metrics)
    std::mutex qmu; std::condition_variable qcv;
    std::deque<std::unique_ptr<Job>> queue;

    // A dense read with several slots goes in pieces of whole ubatches (the same ubatches, so the same arithmetic, as one read);
    // between pieces a request waiting for an idle slot is admitted (llama-server reads in n_batch pieces too: its watchdog
    // probe, max_tokens 1 inside 12 s, must not wait out a 60-second prompt on the other slot), and with --read-interleave N
    // the generating slots step every N tokens. A piece that reads no logits stays queued on the cards: with nobody waiting
    // the pieces cost nothing. An expert model reads in one go (each piece would lend the cache's VRAM and take it back).
    std::function<bool()> admit_one;
    std::function<void()> step_others;
    const size_t kAdmitPiece = (n_slots > 1 && eng.hp().n_expert == 0) ? (size_t) std::max(A.eo.n_ubatch, 4096 / A.eo.n_ubatch * A.eo.n_ubatch) : 0;
    auto read_text = [&](Slot & S, const std::vector<int32_t> & t, std::vector<float> * lgp) -> std::string {
        const size_t piece = A.read_interleave > 0 ? (size_t) A.read_interleave : kAdmitPiece;
        if (piece == 0 || eng.hp().n_expert > 0 || t.size() <= piece) return eng.decode(t, lgp, nullptr, S.id);
        std::string e;
        for (size_t j = 0; j < t.size() && e.empty(); j += piece) {
            const std::vector<int32_t> p(t.begin() + j, t.begin() + std::min(t.size(), j + piece));
            const bool last = j + piece >= t.size();
            e = eng.decode(p, last ? lgp : nullptr, nullptr, S.id);
            if (e.empty() && !last) {
                while (admit_one()) {}
                if (A.read_interleave > 0) step_others();
            }
        }
        return e;
    };
    // the request's prompt read into slot S (its sequence): everything the one-request server did before the first sample
    auto start = [&](Slot & S, const json & body_in, int & http_code, bool chat_api) -> json {
        const int seq = S.id;
        S.go = false;   // not a generating slot while its prompt is read (other slots may step meanwhile)
        server_tokens & cached = S.cached;
        std::string err;
        json body = body_in;
        std::vector<raw_buffer> files;
        std::vector<int> img_side;
        size_t img_k = 0;
        json data;
        if (chat_api) {
            // a system message after the first one becomes a user message in place (Strata's _late_system_to_user): the Qwen
            // templates take a system message only at the start, and clients send one after tool results or as hook context;
            // merging it into the first would change the prompt's start and cost the prompt cache every turn
            if (body.contains("messages") && body.at("messages").is_array())
                for (size_t i = 1; i < body.at("messages").size(); ++i) {
                    json & m = body.at("messages")[i];
                    if (m.is_object() && m.value("role", "") == "system") m["role"] = "user";
                }
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
        if (parked_budget > 0 && tp.cache_prompt) {
            const size_t lcp_slot = cached.get_common_prefix(prompt);
            auto best = parked.end();
            size_t best_lcp = 0;
            for (auto it = parked.begin(); it != parked.end(); ++it) {
                const size_t l = it->tokens.get_common_prefix(prompt);
                if (l > best_lcp) { best_lcp = l; best = it; }
            }
            const bool gain = best != parked.end() && best_lcp > lcp_slot + 256;
            const bool replaced = cached.size() >= kParkMin && lcp_slot < cached.size() / 2;
            if ((gain || replaced) && cached.size() >= kParkMin && eng.n_past(seq) > 0) park_slot(S);
            if (gain) {
                const auto t0 = std::chrono::steady_clock::now();
                const std::string ue = eng.unpark(seq, *best->st);
                if (ue.empty()) {
                    cached = std::move(best->tokens);
                    fprintf(stderr, "impulse: a parked %d-token conversation back from RAM (%zu tokens shared, %.0f ms)\n",
                            q35::Engine::parked_pos(*best->st), best_lcp, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                } else {
                    fprintf(stderr, "impulse: the parked conversation did not come back (%s); reading again\n", ue.c_str());
                    cached.clear();
                }
                parked_total -= q35::Engine::parked_bytes(*best->st);
                parked.erase(best);
            }
            parked_evict();
        }
        server_slot_stats stats;
        stats.update_prompt_start();
        // prompt cache: keep the common prefix, back to where the recurrent state can return (q35::Engine::resume_point)
        size_t keep = 0;
        if (tp.cache_prompt) {
            keep = cached.get_common_prefix(prompt);
            if (keep >= prompt.size()) keep = prompt.size() - 1;               // re-read the last token for its logits
            while (keep > 0 && prompt[keep - 1] == LLAMA_TOKEN_NULL) --keep;   // never resume inside an image
            const int rp = eng.resume_point((int) prompt.pos_next((int64_t) keep), seq);
            while (keep > 0 && (int) prompt.pos_next((int64_t) keep) > rp) --keep;
            while (keep > 0 && prompt[keep - 1] == LLAMA_TOKEN_NULL) --keep;
            if ((int) prompt.pos_next((int64_t) keep) != rp) keep = 0;
        }
        if (tp.cache_prompt && cached.size() > 0)
            fprintf(stderr, "impulse: prompt cache: %zu of %d tokens reused (%zu shared with the slot's %zu)\n", keep, n_prompt,
                    cached.get_common_prefix(prompt), cached.size());
        if (!eng.rollback_to(keep ? (int) prompt.pos_next((int64_t) keep) : 0, seq).empty()) { keep = 0; eng.rollback_to(0, seq); }
        cached.keep_first(keep);
        stats.n_prompt_cached = keep;
        // read the rest: text runs through decode(), images through mtmd + decode_embd()
        std::vector<float> & lg = S.lg;
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
                // checkpoints of the recurrent state (q35::Engine::resume_point), taken where the next request is likely to
                // resume: before the final "<|im_start|>" (the assistant header: a next chat turn re-renders everything up
                // to it the same way) and before the prompt's last token (the same prompt again). Both pieces after them
                // are a few tokens (the CPU path). Reads of ckpt_min tokens or more also stop before the last user
                // message's "<|im_start|>" and 512 tokens from the end (a prompt that changes only at its end), as
                // llama-server checkpoints: each such cut is one more big ubatch, which streams every RAM-resident expert
                // through the card once more (~1 s on a 12 GB card), so short reads skip them
                const bool last = text_end == prompt.size();
                size_t at = 0;
                if (last) {
                    std::vector<size_t> cuts;   // positions in t, a checkpoint before reading t[cut..]
                    const bool full = (int) t.size() >= A.ckpt_min;
                    int marks = 0;
                    for (size_t j = t.size(); im_start != LLAMA_TOKEN_NULL && marks < (full ? 2 : 1) && j-- > 0;)
                        if (t[j] == im_start) { cuts.push_back(j); marks++; }
                    if (full && t.size() > 512) cuts.push_back(t.size() - 512);
                    cuts.push_back(t.size() - 1);
                    std::sort(cuts.begin(), cuts.end());
                    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
                    cuts.erase(std::remove(cuts.begin(), cuts.end(), (size_t) 0), cuts.end());
                    for (const size_t cut : cuts) {
                        if (cut > at) {
                            err = read_text(S, std::vector<int32_t>(t.begin() + at, t.begin() + cut), &lg);
                            if (!err.empty()) { http_code = 500; return format_error_response("decode: " + err, ERROR_TYPE_SERVER); }
                            at = cut;
                        }
                        eng.checkpoint(seq);
                    }
                    t.erase(t.begin(), t.begin() + at);
                }
                err = read_text(S, t, &lg);
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
                // M-RoPE positions (mtmd_helper_decode_image_chunk: t, y, x, z sections)
                const int p0 = eng.n_past(seq);
                std::vector<int32_t> pos4((size_t) nt * 4);
                const mtmd_image_tokens * itoks = mtmd_input_chunk_get_tokens_image(ch);
                if (mtmd_decode_use_mrope(enc) && itoks) {
                    std::vector<mtmd_decoder_pos> rp(nt);
                    mtmd_helper_image_get_decoder_pos(itoks, p0, rp.data());
                    for (int q = 0; q < nt; ++q) { pos4[q] = rp[q].t; pos4[nt + q] = rp[q].y; pos4[2 * nt + q] = rp[q].x; pos4[3 * nt + q] = rp[q].z; }
                } else for (int q = 0; q < nt; ++q) { pos4[q] = pos4[nt + q] = pos4[2 * nt + q] = p0 + q; pos4[3 * nt + q] = 0; }
                err = eng.decode_embd(mtmd_get_output_embd(enc), nt, pos4.data(), (int) mtmd_input_chunk_get_n_pos(ch), &lg, seq);
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
        S.tl_eng = S.tl_smp = S.tl_acc = S.tl_emit = S.tl_smp2 = S.tl_read = S.tl_read2 = S.tl_write = S.tl_write2 = 0; S.tl_n = 0;
        S.scratch.assign(probes ? (size_t) 3 * n_vocab : 0, 0.0f);
        http_code = 200;
        return json();
    };
    auto emit = [&](Slot & S, llama_token t) -> bool {   // false = stop here
        if (S.sst && S.job->chan->cancelled) { S.stop = STOP_TYPE_LIMIT; return false; }   // the streaming client went away
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
        if (S.sst)   // stream: this token's text as llama-server's partial result (held back while it may be a stop word)
            S.job->chan->push(lite::stream_partial(*S.sst, S.tp, lite::stream_take(S.text, S.n_sent, S.tp.antiprompt, false), t,
                                                   S.stats.n_gen, S.n_prompt, (int) S.keep));
        return true;
    };
    auto finish = [&](Slot & S, const std::string & err) {   // the answer (or the error) to the waiting request; the slot is idle
        if (timeline && S.tl_n) {
            const auto & ES = eng.stats();
            fprintf(stderr, "impulse timeline: %d tokens; engine.decode %.0f us a token (of which, over all small ubatches so far: build/reuse %.0f, inputs %.0f, compute %.0f, readout %.0f us); sample %.0f, accept %.0f, emit %.0f us a token\n",
                    S.tl_n, S.tl_eng / S.tl_n, ES.t_ubatches ? ES.t_build * 1e3 / ES.t_ubatches : 0.0, ES.t_ubatches ? ES.t_inputs * 1e3 / ES.t_ubatches : 0.0,
                    ES.t_ubatches ? ES.t_compute * 1e3 / ES.t_ubatches : 0.0, ES.t_ubatches ? ES.t_readout * 1e3 / ES.t_ubatches : 0.0, S.tl_smp / S.tl_n, S.tl_acc / S.tl_n, S.tl_emit / S.tl_n);
            if (probes) {
                fprintf(stderr, "impulse timeline probes: the sampler again on the same logits %.0f us; a dependent-sum pass over the %d logits %.0f us, again %.0f us; a write pass over %zu floats %.0f us, again %.0f us\n",
                        S.tl_smp2 / S.tl_n, n_vocab, S.tl_read / S.tl_n, S.tl_read2 / S.tl_n, S.scratch.size(), S.tl_write / S.tl_n, S.tl_write2 / S.tl_n);
                // the chain, stage by stage, on the last logits: the fill, then each sampler's apply on a freshly filled array
                llama_sampler * chain = common_sampler_get(S.smpl);
                std::vector<llama_token_data> cur((size_t) n_vocab);
                std::string line = "impulse timeline chain (" + std::string(common_sampler_print(S.smpl)) + "):";
                for (int rep = 0; rep < 2; ++rep) {
                    auto a0 = std::chrono::steady_clock::now();
                    for (int t = 0; t < n_vocab; ++t) cur[t] = llama_token_data{ t, S.lg[t], 0.0f };
                    line += (rep ? " | fill " : " fill ") + std::to_string((int) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count());
                    llama_token_data_array p = { cur.data(), cur.size(), -1, false };
                    for (int si = 0; si < llama_sampler_chain_n(chain); ++si) {
                        llama_sampler * st = llama_sampler_chain_get(chain, si);
                        auto a = std::chrono::steady_clock::now();
                        llama_sampler_apply(st, &p);
                        line += std::string(" ") + llama_sampler_name(st) + " " + std::to_string((int) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
                    }
                }
                fprintf(stderr, "%s us\n", line.c_str());
            }
        }
        int code = 200; json outj;
        if (S.smpl) { common_sampler_free(S.smpl); S.smpl = nullptr; }
        if (!err.empty()) { code = 500; outj = format_error_response("generation: " + err, ERROR_TYPE_SERVER); }
        else {
            S.stats.update_gen_last();
            metrics.request((double) S.stats.n_prompt_processed, S.stats.t_prompt_ms(), (double) S.stats.n_gen, S.stats.t_gen_ms(),
                            (double) S.stats.n_draft_tokens, (double) S.stats.n_draft_accepted, (double) S.stats.n_draft_verif_steps, (double) eng.n_past(S.id));
            server_task_result_cmpl_final res;
            res.content = lite::utf8_clean(S.text);   // valid UTF-8 for the chat parser below (src/lite/utf8.hpp: it was a 500)
            if (S.sst) { res.content = lite::utf8_clean(lite::stream_take(S.text, S.n_sent, S.tp.antiprompt, true)); res.tokens.clear(); }
            res.tokens = S.out; res.stream = S.tp.stream; res.include_usage = S.tp.include_usage; res.stats = S.stats;
            res.truncated = false; res.n_decoded = (int32_t) S.out.size(); res.n_prompt_tokens = S.n_prompt;
            res.n_prompt_tokens_cache = (int32_t) S.keep; res.n_tokens_cached = eng.n_past(S.id); res.has_new_line = false;
            res.stopping_word = S.stopping_word; res.stop = S.stop; res.post_sampling_probs = false;
            res.generation_params = S.tp; res.verbose = false; res.res_type = S.tp.res_type;
            res.oaicompat_model = S.tp.oaicompat_model; res.oaicompat_cmpl_id = S.tp.oaicompat_cmpl_id;
            task_result_state state(S.tp.chat_parser_params);
            res.update(S.sst ? *S.sst : state);   // stream: the request's own state, after its partial results
            outj = res.to_json();
        }
        if (S.job->chan) {   // stream: the final events (or the error), then the end
            S.job->chan->push(code == 200 ? outj : json{ { "error", outj } });
            S.job->chan->close();
        }
        S.sst.reset();
        S.job->done.set_value({ code, outj });
        S.job.reset(); S.active = false;
        S.t_last = std::chrono::steady_clock::now().time_since_epoch().count();
    };
    // one step for every generating slot: a token each, or (MTP, draft-mtp) a draft window each: drafts at pos0, pos0 + 1,
    // ... from the slot's pending hidden row, the drafter's draft cells dropped, then the windows verified together and
    // llama.cpp's sample-and-accept per slot, the rest rolled back (recurrent snapshot)
    auto step_all = [&]() {
        std::vector<Slot *> gen;
        for (auto & S : slots) if (S->job && S->go) gen.push_back(S.get());
        if (gen.empty()) return;
        metrics.step((int) gen.size());
        std::vector<q35::Engine::SeqStep> steps;
        std::vector<int> pos0(gen.size());
        std::vector<float> hc, hh;
        for (size_t g = 0; g < gen.size(); ++g) {
            Slot & S = *gen[g];
            std::vector<int32_t> win = { S.tok };
            if (mtp) {
                pos0[g] = eng.n_past(S.id);
                const int room = std::min(A.n_draft, std::min(S.n_predict - (int) S.out.size(), n_ctx_slot - pos0[g] - 2));
                hc = eng.mtp_h(S.id);
                for (int k = 0; k < room; ++k) {
                    int32_t d = 0;
                    if (!eng.draft(win.back(), hc.data(), pos0[g] + k, &d, &hh, S.id).empty()) break;
                    if (A.spec_min_p > 0 && eng.draft_p() < A.spec_min_p) break;   // an unsure draft costs a verify row
                    win.push_back(d); hc = hh;
                }
                eng.mtp_trim(pos0[g], S.id);
            }
            steps.push_back({ S.id, win });
        }
        std::vector<float> L;
        const auto te0 = std::chrono::steady_clock::now();
        const std::string err = eng.decode_multi(steps, &L, nullptr, lazy_logits);
        const auto te1 = std::chrono::steady_clock::now();
        if (!err.empty()) { for (Slot * S : gen) finish(*S, err); return; }
        const bool lazy = eng.lazy_logits();
        auto row_of = [&](size_t r) -> const float * { return lazy ? eng.logits_row((int) r) : L.data() + r * n_vocab; };
        size_t row = 0;
        for (size_t g = 0; g < gen.size(); ++g) {
            Slot & S = *gen[g];
            const auto & win = steps[g].toks;
            std::string serr;
            if (!mtp) {
                const float * lgp = row_of(row);
                if (!lgp) { finish(S, "logits row read failed"); row += win.size(); continue; }
                S.cached.push_back(S.tok);
                S.tok = common_sampler_sample_logits(S.smpl, lgp, n_vocab);
                const auto te2 = std::chrono::steady_clock::now();
                if (probes) {
                    const llama_token again = common_sampler_sample_logits(S.smpl, lgp, n_vocab);
                    const auto tp1 = std::chrono::steady_clock::now();
                    float acc = 0; for (int i = 0; i < n_vocab; ++i) acc += lgp[i];
                    const auto tp2 = std::chrono::steady_clock::now();
                    for (int i = 0; i < n_vocab; ++i) acc += lgp[i] * 0.5f;   // the same read again (now cache-hot, if the first was cold)
                    const auto tp2b = std::chrono::steady_clock::now();
                    for (size_t i = 0; i < S.scratch.size(); ++i) S.scratch[i] = acc + (float) i;
                    const auto tp3 = std::chrono::steady_clock::now();
                    for (size_t i = 0; i < S.scratch.size(); ++i) S.scratch[i] += 1.0f;   // the same write again
                    const auto tp3b = std::chrono::steady_clock::now();
                    S.tl_smp2 += std::chrono::duration<double, std::micro>(tp1 - te2).count();
                    S.tl_read += std::chrono::duration<double, std::micro>(tp2 - tp1).count();
                    S.tl_read2 += std::chrono::duration<double, std::micro>(tp2b - tp2).count();
                    S.tl_write += std::chrono::duration<double, std::micro>(tp3 - tp2b).count();
                    S.tl_write2 += std::chrono::duration<double, std::micro>(tp3b - tp3).count();
                    if (again != S.tok && S.tp.sampling.temp <= 0) fprintf(stderr, "impulse timeline: a second greedy sample differed (%d vs %d)\n", again, S.tok);
                    S.lg.assign(lgp, lgp + n_vocab);   // the chain timing at the end runs on the last logits
                }
                common_sampler_accept(S.smpl, S.tok, true);
                const auto te3 = std::chrono::steady_clock::now();
                S.go = emit(S, S.tok);
                if (timeline) {
                    const auto te4 = std::chrono::steady_clock::now();
                    S.tl_eng += std::chrono::duration<double, std::micro>(te1 - te0).count();   // the step (every slot's token)
                    S.tl_smp += std::chrono::duration<double, std::micro>(te2 - te1).count();
                    S.tl_acc += std::chrono::duration<double, std::micro>(te3 - te2).count();
                    S.tl_emit += std::chrono::duration<double, std::micro>(te4 - te3).count(); S.tl_n++;
                }
            } else {
                S.stats.n_draft_tokens += win.size() - 1; S.stats.n_draft_verif_steps++;
                int r = 0;
                for (;; ++r) {
                    const float * lgp = row_of(row + r);
                    if (!lgp) { serr = "logits row read failed"; break; }
                    const llama_token t = common_sampler_sample_logits(S.smpl, lgp, n_vocab);
                    common_sampler_accept(S.smpl, t, true);
                    S.cached.push_back(win[r]);
                    S.tok = t;
                    S.go = emit(S, t);
                    if (!S.go || r + 1 >= (int) win.size() || t != win[r + 1]) break;
                    S.stats.n_draft_accepted++;
                }
                if (serr.empty()) serr = eng.rollback(pos0[g] + r + 1, S.id);
            }
            row += win.size();
            if (!serr.empty()) finish(S, serr);
            else if (!S.go) finish(S, "");
        }
    };
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
            if (!next->chat_api && next->body.contains("prompt") && next->body.at("prompt").is_string())
                probe = common_tokenize(vocab, next->body.at("prompt").get<std::string>(), true, true);
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
            if (best->job->chan) { best->job->chan->close(); }
            best->job->done.set_value({ code, r }); best->job.reset(); best->active = false;
        } else {
            if (best->job->chan && best->tp.stream) { best->sst = std::make_unique<task_result_state>(best->tp.chat_parser_params); best->n_sent = 0; }
            best->go = emit(*best, best->tok);
            if (!best->go) finish(*best, "");
        }
        return true;
    };
    step_others = step_all;
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
    auto lora_props = [&]() {
        json arr = json::array();
        for (int i = 0; i < eng.lora_count(); ++i) arr.push_back(json{ { "file", std::filesystem::path(eng.lora_path(i)).filename().string() }, { "scale", eng.lora_scale(i) } });
        return arr;
    };
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
            { "build_info", std::string(lfm2 ? "impulse " IMPULSE_VERSION_STR " (lfm2moe engine) / " : flint ? "impulse " IMPULSE_VERSION_STR " (flint engine) / " : "impulse " IMPULSE_VERSION_STR " (qwen35 engine) / ") + llama_build_info() }, { "is_sleeping", false },
            { "impulse", json{ { "slots", eng.slots() }, { "arena_shrinks", eng.stats().arena_shrinks }, { "graphs_built", eng.stats().graphs_built }, { "graphs_reused", eng.stats().graphs_reused },
                              { "timeline", json{ { "ubatches", eng.stats().t_ubatches }, { "build_ms", eng.stats().t_build }, { "inputs_ms", eng.stats().t_inputs },
                                                  { "compute_ms", eng.stats().t_compute }, { "readout_ms", eng.stats().t_readout } } }, { "n_cpu_moe", A.eo.n_cpu_moe },
                              { "prefill_ubatch", A.eo.prefill_ubatch < 0 ? json("auto") : json(A.eo.prefill_ubatch) }, { "last_chunk", eng.stats().last_chunk }, { "lend", A.eo.lend }, { "mtp", mtp }, { "vision", !A.mmproj.empty() }, { "projector_in_ram", mctx.load() != nullptr && !mm_gpu_always },
                              { "projector", A.mmproj.empty() ? "none" : A.mmproj_place }, { "projector_on_card", mctx_gpu != nullptr },
                              { "projector_moves", mm_moves.load() }, { "slots_full", slots_full },
                              { "projector_min_size", A.mmproj_min_size }, { "projector_gpu_max_ctx", A.mmproj_gpu_max_ctx },
                              { "projector_idle_s", A.mmproj_idle },
                              { "adapt", A.eo.adapt }, { "adapt_half_life", A.eo.adapt_half_life }, { "adapt_budget", A.eo.adapt_budget },
                              { "adapt_refill", A.eo.adapt_refill }, { "async_swaps", eng.async_swaps() }, { "ram_locked_gb", eng.ram_locked() / 1e9 }, { "ram_experts_gb", eng.ram_experts() / 1e9 }, { "overlap", A.eo.overlap },
                              { "threads", A.threads }, { "pool_threads", A.eo.pool_threads }, { "n_draft", A.n_draft }, { "spec_min_p", A.spec_min_p },
                              { "mtp_head", A.mtp_own_head ? "drafter" : "target" }, { "engine", lfm2 ? "lfm2moe" : flint ? "flint" : "qwen35" },
                              { "lora", eng.lora_count() == 0 ? json(nullptr) : lora_props() } } },
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
        if (body.contains("lora")) {
            send_json(r, json{{"error", format_error_response("LoRA is fixed at server startup; per-request adapters are not supported", ERROR_TYPE_INVALID_REQUEST)}}, 400); return;
        }
        auto job = std::make_unique<Job>();
        job->body = body; job->chat_api = chat_api;
        const bool want_stream = body.contains("stream") && body.at("stream").is_boolean() && body.at("stream").get<bool>();
        std::shared_ptr<lite::StreamChan> chan;
        if (want_stream) { chan = std::make_shared<lite::StreamChan>(); job->chan = chan; }
        auto fut = job->done.get_future();
        { std::lock_guard<std::mutex> ql(qmu); queue.push_back(std::move(job)); }
        qcv.notify_one();
        if (chan) {   // stream: the events as the scheduler makes them (src/lite/stream_out.hpp); an error before any token as JSON
            {
                std::unique_lock<std::mutex> l(chan->mu);
                chan->cv.wait(l, [&] { return !chan->events.empty() || chan->closed; });
                if (chan->events.empty()) {
                    l.unlock();
                    auto [code, out] = fut.get();
                    send_json(r, code == 200 ? out : json{ { "error", out } }, code);
                    return;
                }
            }
            r.status = 200;
            r.set_chunked_content_provider("text/event-stream", [chan](size_t, httplib::DataSink & sink) {
                std::deque<json> evs; bool closed;
                {
                    std::unique_lock<std::mutex> l(chan->mu);
                    chan->cv.wait(l, [&] { return !chan->events.empty() || chan->closed; });
                    evs.swap(chan->events); closed = chan->closed;
                }
                for (const auto & ev : evs) {
                    const std::string sse = format_oai_sse(ev);
                    if (!sink.write(sse.data(), sse.size())) { chan->cancelled = true; return false; }
                }
                if (closed) {
                    static const std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                }
                return true;
            }, [chan](bool ok) { if (!ok) chan->cancelled = true; });
            return;
        }
        auto [code, out] = fut.get();
        if (code == 200 && out.is_array()) {   // (a stream is answered above; this is the old burst, kept for safety)
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
    // llama-server's /lora-adapters: GET lists the adapters (load order) with their scales; POST [{"id", "scale"}] sets scales
    // for the requests that follow (Nebula's Standard / Uncensored: the Uncensored adapter at 0 or 1). It waits for the
    // running answers to finish, then drops every prompt cache, since a cache computed with other scales is not this model's
    auto lora_list = [&]() {
        json arr = json::array();
        for (int i = 0; i < eng.lora_count(); ++i) arr.push_back(json{ { "id", i }, { "path", eng.lora_path(i) }, { "scale", eng.lora_scale(i) } });
        return arr;
    };
    srv.Get("/lora-adapters", [&](const httplib::Request &, httplib::Response & r) {
        std::lock_guard<std::mutex> g(busy);
        send_json(r, lora_list(), 200);
    });
    srv.Post("/lora-adapters", [&](const httplib::Request & q, httplib::Response & r) {
        json body;
        try { body = json::parse(q.body); } catch (...) { send_json(r, json{ { "error", format_error_response("invalid JSON body", ERROR_TYPE_INVALID_REQUEST) } }, 400); return; }
        if (!body.is_array()) { send_json(r, json{ { "error", format_error_response("a list of {id, scale}", ERROR_TYPE_INVALID_REQUEST) } }, 400); return; }
        std::vector<std::pair<int, float>> want;
        for (const auto & x : body) {
            if (!x.is_object() || !x.contains("id") || !x.contains("scale") || !x["id"].is_number_integer() || !x["scale"].is_number()) {
                send_json(r, json{ { "error", format_error_response("each entry is {\"id\": n, \"scale\": x}", ERROR_TYPE_INVALID_REQUEST) } }, 400); return;
            }
            const int id = x["id"].get<int>(); const float sc = x["scale"].get<float>();
            if (id < 0 || id >= eng.lora_count() || !std::isfinite(sc)) { send_json(r, json{ { "error", format_error_response("no such adapter, or a scale that is not finite", ERROR_TYPE_INVALID_REQUEST) } }, 400); return; }
            want.push_back({ id, sc });
        }
        for (int tries = 0; ; ++tries) {
            {
                std::lock_guard<std::mutex> g(busy);
                bool running = false;
                for (auto & S : slots) running = running || S->job;
                if (!running || tries >= 6000) {   // at most ~2 minutes behind running answers
                    bool changed = false;
                    for (auto & [id, sc] : want) if (eng.lora_scale(id) != sc) { eng.set_lora_scale(id, sc); changed = true; }
                    if (changed) {
                        for (auto & S : slots) { S->cached.clear(); eng.rollback_to(0, S->id); }
                        parked.clear(); parked_total = 0;
                        fprintf(stderr, "impulse: LoRA scales set; prompt caches dropped\n");
                    }
                    send_json(r, lora_list(), 200);
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
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

}  // namespace q35srv

#ifndef IMPULSE_COMBINED

int main(int argc, char ** argv) { return q35srv::run(argc, argv); }
#endif
