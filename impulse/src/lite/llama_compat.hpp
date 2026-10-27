// Impulse: llama-server's command line and /metrics, so a deployment written for llama-server starts the impulse
// servers unchanged. Each llama-server flag a deployment passes either means the same here under our name (renamed), is
// ours already, or has no effect here (accepted and noted: no auto-fit, every layer on the GPUs, flash attention always,
// jinja templates always, no RAM prompt cache, no CORS, no web UI log options). --help prints llama-server's help layout
// (the flag catalog tools read). A flag neither side knows is left for the server's parser, which refuses it.
#pragma once

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace lite {

// argv -> our argv; help = --help / -h was given; notes = what was accepted without effect
inline std::vector<std::string> llama_compat_args(int argc, char ** argv, bool & help, std::vector<std::string> & notes) {
    std::vector<std::string> in(argv + 1, argv + argc), out = { argv[0] };
    auto ignore = [&](const std::string & what) { notes.push_back(what); };
    for (size_t i = 0; i < in.size(); ++i) {
        const std::string & a = in[i];
        auto val = [&]() -> std::string { return i + 1 < in.size() ? in[++i] : std::string(); };
        if (a == "-h" || a == "--help") { help = true; continue; }
        if (a == "-a") { out.push_back("--alias"); out.push_back(val()); continue; }
        // the same meaning under our name
        if (a == "--cache-type-k") { out.push_back("-ctk"); out.push_back(val()); continue; }
        if (a == "--cache-type-v") { out.push_back("-ctv"); out.push_back(val()); continue; }
        if (a == "--spec-draft-model" || a == "-md" || a == "--model-draft") { out.push_back("--mtp"); out.push_back(val()); continue; }
        if (a == "--spec-draft-n-max" || a == "--draft-max" || a == "--draft") { out.push_back("--n-draft"); out.push_back(val()); continue; }
        if (a == "-cram" || a == "--cache-ram") { out.push_back("--cache-ram"); out.push_back(val()); continue; }   // the RAM prompt cache (Qwen engine)
        if (a == "--no-mmproj-offload") { out.push_back("--mmproj-place"); out.push_back("cpu"); continue; }
        if (a == "--mmproj-device") {   // CUDA0: the first card (llama-server's default); another card: the first one without layers
            const std::string d = val();
            out.push_back("--mmproj-place"); out.push_back(d == "CUDA0" ? "gpu" : d == "none" ? "cpu" : "second");
            continue;
        }
        if (a == "-n" || a == "--n-predict" || a == "--predict") { ignore(a + " (per request here)"); val(); continue; }
        // no GPU layers / no device: a dense model on the CPU (--device cpu); any other value: every layer on the GPUs
        if (a == "-ngl" || a == "--gpu-layers" || a == "--n-gpu-layers") { const std::string v = val(); if (v == "0") { out.push_back("--device"); out.push_back("cpu"); } else ignore(a + " " + v); continue; }
        if (a == "-dev" || a == "--device") { const std::string v = val(); if (v == "none" || v == "cpu") { out.push_back("--device"); out.push_back("cpu"); } else ignore(a + " " + v); continue; }
        // accepted, no effect here (the value, if any, consumed)
        static const char * with_value[] = { "--fit", "-fa", "--flash-attn", "-b", "--batch-size",
            "-tb", "--threads-batch", "--numa", "-mg", "--main-gpu", "--spec-type", "--spec-draft-device", "-devd", "--device-draft",
            "--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--spec-draft-p-min", "--draft-p-min", "--spec-draft-n-min", "--draft-min",
            "--log-colors", "-lv", "--log-verbosity", "--cors-origins", "--load-mode", "--reasoning-format",
            "--cache-reuse", nullptr };
        static const char * no_value[] = { "--jinja", "--no-context-shift", "--no-webui", "--no-op-offload", "--no-cors-credentials",
            "--log-disable", "--verbose", "-v", "--cont-batching", "-cb", "--no-warmup", "--mlock", "--no-mmap", "--kv-unified", "-kvu", nullptr };
        bool done = false;
        for (int k = 0; with_value[k] && !done; ++k) if (a == with_value[k]) { ignore(a + " " + val()); done = true; }
        for (int k = 0; no_value[k] && !done; ++k) if (a == no_value[k]) {
            if (a == "--no-webui") out.push_back(a);   // ours too: no chat page at /
            else ignore(a);
            done = true;
        }
        if (!done) out.push_back(a);
    }
    return out;
}

// llama-server's --help layout ("-c,    --ctx-size N    text"): what a impulse server takes, llama-server's names first
inline void llama_compat_help(FILE * f, const char * server) {
    struct L { const char * spec; const char * help; };
    static const L lines[] = {
        { "-m,    --model FNAME", "model path (GGUF)" },
        { "-a,    --alias STRING", "the model name the API reports" },
        { "--host HOST", "address to listen on (default: 127.0.0.1)" },
        { "--api-key KEY", "API key(s), comma-separated: every route but /health needs \"Authorization: Bearer KEY\" (env LLAMA_API_KEY)" },
        { "--api-key-file FNAME", "API keys, one per line, '#' starts a comment (env LLAMA_ARG_API_KEY_FILE)" },
        { "--port PORT", "port to listen on (default: 8080)" },
        { "-c,    --ctx-size N", "size of the prompt context, every slot together (default: 262144)" },
        { "-np,   --parallel N", "number of server slots (default: 1)" },
        { "-ub,   --ubatch-size N", "physical maximum batch size (default: 512)" },
        { "-t,    --threads N", "number of CPU threads (default: the physical cores)" },
        { "-ctk,  --cache-type-k TYPE", "KV cache data type for K: f16, q8_0, q4_0" },
        { "-ctv,  --cache-type-v TYPE", "KV cache data type for V: f16, q8_0, q4_0" },
        { "-sm,   --split-mode {layer,tensor}", "how to split the model across GPUs (tensor: dense Qwen models)" },
        { "-ts,   --tensor-split N0,N1,N2,...", "fraction of the model to offload to each GPU, comma-separated" },
        { "--mmproj FILE", "path to a multimodal projector file" },
        { "--mmproj-device DEV", "the projector's device: CUDA0, or a card without layers" },
        { "--mmproj-place {second,auto,lazy,gpu,cpu}", "where the projector lives (impulse)" },
        { "--no-mmproj-offload", "keep the projector in RAM" },
        { "--image-max-tokens N", "maximum number of tokens each image can take" },
        { "-md,   --spec-draft-model FNAME", "the MTP drafter GGUF (impulse: --mtp)" },
        { "--spec-draft-n-max N", "number of tokens to draft (impulse: --n-draft)" },
        { "--mtp FNAME", "the MTP drafter GGUF" },
        { "--n-draft N", "drafted tokens per step" },
        { "--timeout N", "server read/write timeout in seconds (default: 7200)" },
        { "--metrics", "enable prometheus compatible metrics endpoint" },
        { "--no-webui", "disable the chat page" },
        { "--read-interleave N", "slots: other slots step every N prompt tokens of a long read (default: 0)" },
        { "--temp N", "temperature (default: 0.8)" },
        { "--top-k N", "top-k sampling (default: 40)" },
        { "--top-p N", "top-p sampling (default: 0.95)" },
        { "--min-p N", "min-p sampling (default: 0.05)" },
        { "--repeat-penalty N", "penalize repeat sequence of tokens (default: 1.0, disabled)" },
        { "--n-cpu-moe N", "keep the experts of the first N layers in RAM (expert models)" },
        { "--kv-initial N", "KV cells on the card at load, grown as needed (0: the whole context; one slot)" },
        { "--fit [on|off]", "accepted: impulse sizes itself" },
        { "-ngl,  --gpu-layers N", "0: a dense model wholly on the CPU (as --device cpu); else every layer runs on the GPUs" },
        { "-fa,   --flash-attn [on|off|auto]", "accepted: flash attention is always on" },
        { "-b,    --batch-size N", "accepted: a prompt is read in ubatches" },
        { "-tb,   --threads-batch N", "accepted: --threads is used" },
        { "--numa TYPE", "accepted: run the server under numactl" },
        { "-mg,   --main-gpu INDEX", "accepted: the first card with a share is the main one" },
        { "--spec-type [none,draft-mtp]", "accepted: --spec-draft-model / --mtp loads the drafter" },
        { "--spec-draft-device <dev1,dev2,..>", "accepted: the drafter runs on one of the model's cards" },
        { "--spec-draft-ngl N", "accepted: the drafter is on the GPU" },
        { "--spec-draft-p-min P", "accepted: drafts are the drafter's top token" },
        { "-cram, --cache-ram N", "size of the RAM prompt cache in MiB: parked conversations (Qwen engine; default 0 = off; Gemma: accepted, no effect)" },
        { "--jinja", "accepted: the model's jinja chat template is always used" },
        { "--no-context-shift", "accepted: context shift is off (a prompt that does not fit is refused)" },
        { "--no-op-offload", "accepted" },
        { "--log-colors [on|off|auto]", "accepted" },
        { "-lv,   --log-verbosity N", "accepted" },
        { "--cors-origins ORIGINS", "accepted: no CORS headers" },
        { "--no-cors-credentials", "accepted" },
        { "--load-mode MODE", "accepted" },
        { nullptr, nullptr },
    };
    fprintf(f, "----- %s (llama-server compatible flags) -----\n\n", server);
    for (int k = 0; lines[k].spec; ++k) fprintf(f, "%-40s %s\n", lines[k].spec, lines[k].help);
}

// /metrics (llama-server's names; Prometheus text format)
struct Metrics {
    std::mutex mu;
    double prompt_tokens = 0, prompt_s = 0, predicted_tokens = 0, predicted_s = 0, decode_calls = 0, busy_slot_sum = 0;
    double drafts = 0, draft_tokens = 0, draft_accepted = 0, n_tokens_max = 0;
    void request(double n_prompt, double prompt_ms, double n_gen, double gen_ms, double n_draft, double n_accepted, double n_verif, double n_tokens) {
        std::lock_guard<std::mutex> l(mu);
        prompt_tokens += n_prompt; prompt_s += prompt_ms / 1e3; predicted_tokens += n_gen; predicted_s += gen_ms / 1e3;
        draft_tokens += n_draft; draft_accepted += n_accepted; drafts += n_verif; if (n_tokens > n_tokens_max) n_tokens_max = n_tokens;
    }
    void step(int busy) { std::lock_guard<std::mutex> l(mu); decode_calls += 1; busy_slot_sum += busy; }
    std::string text(int processing, int deferred) {
        std::lock_guard<std::mutex> l(mu);
        std::string s;
        auto put = [&](const char * name, const char * type, const char * help, double v) {
            char b[512];
            snprintf(b, sizeof b, "# HELP llamacpp:%s %s\n# TYPE llamacpp:%s %s\nllamacpp:%s %.10g\n", name, help, name, type, name, v);
            s += b;
        };
        put("prompt_tokens_total", "counter", "Number of prompt tokens processed.", prompt_tokens);
        put("prompt_seconds_total", "counter", "Prompt process time", prompt_s);
        put("tokens_predicted_total", "counter", "Number of generation tokens processed.", predicted_tokens);
        put("tokens_predicted_seconds_total", "counter", "Predict process time", predicted_s);
        put("n_decode_total", "counter", "Total number of llama_decode() calls", decode_calls);
        put("n_tokens_max", "counter", "Largest observed n_tokens.", n_tokens_max);
        put("n_busy_slots_per_decode", "counter", "Average number of busy slots per llama_decode() call", decode_calls ? busy_slot_sum / decode_calls : 0);
        put("prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.", prompt_s > 0 ? prompt_tokens / prompt_s : 0);
        put("predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.", predicted_s > 0 ? predicted_tokens / predicted_s : 0);
        put("requests_processing", "gauge", "Number of requests processing.", processing);
        put("requests_deferred", "gauge", "Number of requests deferred.", deferred);
        put("spec_decode_num_draft_tokens_total", "counter", "Total draft tokens generated.", draft_tokens);
        put("spec_decode_num_accepted_tokens_total", "counter", "Total draft tokens accepted by the target model.", draft_accepted);
        put("spec_decode_num_drafts_total", "counter", "Total speculative decoding verification steps.", drafts);
        return s;
    }
};

} // namespace lite
