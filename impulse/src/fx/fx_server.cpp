// impulse-fx: Strata's Flash-Next engine (impulse/flashnext, the reference engine's) behind Impulse's own server, serving the API
// serve/server.py serves (One Engine P1: the bridge; P2: server parity).
//
//   impulse-server -m <pack folder>            a folder with pack/ (a Strata pack) and a *dense*.gguf
//   impulse-server -m <folder>/<x-dense>.gguf  the dense GGUF beside pack/            (src/fx/fx_pack.hpp decides)
//       [--fx-config run-config.json]          the engine's settings as Strata's run config (what Nebula writes for a
//                                              Flash-Next engine today): "args", "env", "gpu", "layer_split", and
//                                              server.py's own keys: "model_name", "aliases", "vision",
//                                              "stop_token_text", "repeat_stop_tokens", "tool_name_repair",
//                                              "fit_max_tokens", "reasoning_budget_tokens", "engine_silence_s"
//       [--fx-config-json JSON]                or the same run config as text (a config Nebula builds: no file)
//                                              its "log": the engine's stderr appended to that file (src/fx/fx_log.hpp)
//       [--fx-args ARG ...]                    or the engine's arguments directly: everything after --fx-args
//       [--fx-lib PATH]                        libimpulse_fx (default: beside this binary; $IMPULSE_FX_LIB)
//       [--fx-image-python PY]                 a Python with Pillow for picture formats the encoder cannot read
//                                              (default: $IMPULSE_FX_IMAGE_PYTHON, else none: such a picture is a 400)
//       [--fx-tokenize-endpoints]              also serve llama-server's /tokenize and /detokenize (server.py has
//                                              neither, and Nebula sizes requests with /tokenize when it is there)
//       [--fx-cli]                             no server: the engine's own command line in this process
//       [--alias A] [--host H] [--port P] [--api-key K] [--api-key-file F] [--timeout S]
//
// The engine is libimpulse_fx (flashnext/src/program/fx_bridge.cpp): Strata's `strata --serve` built as one shared
// library, loaded with dlopen (RTLD_LOCAL); its serve loop runs on a thread of this process, its requests and lines on
// two pipes, in the protocol serve/server.py speaks (class StrataEngine): `GEN <max_new> [key=value ...] <ids>` (GENI
// with an embeddings file for pictures) in, `T <id>` per token, PP / RESUME / INFO along the way, `DONE ...` or `ERR ...`
// at the end, `STOP` to end a request early.
//
// POST /v1/chat/completions is server.py's (_openai, Service.prepare, Service._run, openai_chunks / openai_collect):
// src/fx/fx_front.* hold the request and answer layer; docs/FX-PARITY.md lists every field and behaviour and what of it
// is here.  GET /health, /v1/models, /models, /props, /slots answer as server.py's do (before the engine is ready: 503
// "Loading model", where server.py is not listening yet).  POST /completion and /completions stay llama-server's (the
// gates' raw-id path).
#include "../lite/api_key.hpp"
#include "../lite/utf8.hpp"
#include "fx_front.hpp"
#include "fx_log.hpp"
#include "fx_pack.hpp"
#include "fx_slots.hpp"
#include "fx_vision.hpp"

#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"

#include "httplib.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef IMPULSE_VERSION
#define IMPULSE_VERSION_STR IMPULSE_VERSION
#else
#define IMPULSE_VERSION_STR "dev"
#endif

namespace fxsrv {

using json = fxf::json;   // nlohmann's ordered JSON (llama.cpp's own common_json only where its functions take it)

// llama-server's error object (format_error_response)
static json llama_error(const std::string & message, int code, const char * type) {
    return json{ { "code", code }, { "message", message }, { "type", type } };
}
static common_json to_common(const json & j) { return common_json::parse(j.dump()); }

using fx_abi_fn = int (*)();
using fx_run_fn = int (*)(int, char **, std::intptr_t, std::intptr_t);
#if defined(_WIN32)
using pipe_t = HANDLE;
#else
using pipe_t = int;
#endif

// ---------------------------------------------------------------- the engine: libimpulse_fx on a thread, two pipes
struct Engine : fxs::Wire {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;     // the serve lines of the request that reads them
    std::vector<std::deque<std::string>> slot_lines;   // --batch: each slot's BT / BDONE lines (INFO batch_slots=N)
    bool eof = false;                  // the engine's output ended
    bool ready = false;                // READY seen
    bool can_stop_ = false;            // READY <ctx> stop: it honours STOP mid-request
    long long max_ctx = 0;
    std::map<std::string, std::string> info;   // INFO key=value (engine version, kv, expert slots, ...)
    bool reading = false;              // a request is reading the lines (otherwise they are logged)
    pipe_t in_w{};
    std::mutex wmu;

    bool send(const std::string & line) override {
        std::lock_guard<std::mutex> g(wmu);
        const std::string s = line + "\n";
        size_t off = 0;
        while (off < s.size()) {
#if defined(_WIN32)
            DWORD n = 0;
            if (!WriteFile(in_w, s.data() + off, (DWORD) (s.size() - off), &n, nullptr) || n == 0) return false;
#else
            const ssize_t n = ::write(in_w, s.data() + off, s.size() - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
#endif
            off += (size_t) n;
        }
        return true;
    }
    // the next line for the reading request, waiting at most `seconds`; false: nothing came (or the engine ended: eof)
    bool next(std::string & out, double seconds) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::duration<double>(seconds), [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    }
    bool ended() override { std::lock_guard<std::mutex> lk(mu); return eof; }
    // fxs::Wire (the batch slots' scheduler, src/fx/fx_slots.hpp)
    void ctl_open() override { std::lock_guard<std::mutex> lk(mu); lines.clear(); reading = true; }
    void ctl_close() override { std::lock_guard<std::mutex> lk(mu); reading = false; lines.clear(); }
    bool ctl_next(std::string & out, double seconds) override { return next(out, seconds); }
    bool slot_next(int slot, std::string & out, double seconds) override {
        std::unique_lock<std::mutex> lk(mu);
        if (slot < 0 || (size_t) slot >= slot_lines.size()) return false;
        cv.wait_for(lk, std::chrono::duration<double>(seconds), [&] { return !slot_lines[(size_t) slot].empty() || eof; });
        if (slot_lines[(size_t) slot].empty()) return false;
        out = std::move(slot_lines[(size_t) slot].front());
        slot_lines[(size_t) slot].pop_front();
        return true;
    }
    void slot_clear(int slot) override { std::lock_guard<std::mutex> lk(mu); if (slot >= 0 && (size_t) slot < slot_lines.size()) slot_lines[(size_t) slot].clear(); }
    bool can_stop() override { std::lock_guard<std::mutex> lk(mu); return can_stop_; }
};

// the engine's serve output, line by line: READY and INFO kept as facts, the rest to the request that reads them
static void reader_loop(Engine & E, pipe_t rd) {
    std::string buf;
    char chunk[8192];
    auto take = [&](std::string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::lock_guard<std::mutex> lk(E.mu);
        if (line.rfind("INFO ", 0) == 0) {
            std::istringstream is(line.substr(5));
            std::string kv;
            while (is >> kv) { const size_t eq = kv.find('='); if (eq != std::string::npos) E.info[kv.substr(0, eq)] = kv.substr(eq + 1); }
            // --batch: the slots the engine runs (it says so before READY; it may run fewer than asked)
            if (E.info.count("batch_slots")) E.slot_lines.resize((size_t) std::max(0, std::atoi(E.info["batch_slots"].c_str())));
            return;
        }
        if (!E.slot_lines.empty() && (line.rfind("BT ", 0) == 0 || line.rfind("BDONE ", 0) == 0)) {   // a batch slot's own lines
            const int b = std::atoi(line.c_str() + (line[1] == 'T' ? 3 : 6));
            if (b >= 0 && (size_t) b < E.slot_lines.size()) { E.slot_lines[(size_t) b].push_back(std::move(line)); E.cv.notify_all(); return; }
        }
        if (!E.ready && line.rfind("READY", 0) == 0) {
            std::istringstream is(line.substr(5));
            is >> E.max_ctx;
            std::string w;
            while (is >> w) if (w == "stop") E.can_stop_ = true;
            E.ready = E.max_ctx > 0;
            fx::say("impulse fx: the engine is ready (context %lld%s)\n", E.max_ctx, E.can_stop_ ? ", honours STOP" : "");
            E.cv.notify_all();
            return;
        }
        if (E.reading) { E.lines.push_back(std::move(line)); E.cv.notify_all(); return; }
        fx::say("impulse fx engine: %s\n", line.c_str());   // a line no request waits for: the log
    };
    for (;;) {
#if defined(_WIN32)
        DWORD n = 0;
        if (!ReadFile(rd, chunk, sizeof chunk, &n, nullptr) || n == 0) break;
#else
        const ssize_t n = ::read(rd, chunk, sizeof chunk);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
#endif
        buf.append(chunk, (size_t) n);
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) { take(buf.substr(0, nl)); buf.erase(0, nl + 1); }
    }
    if (!buf.empty()) take(buf);
    std::lock_guard<std::mutex> lk(E.mu);
    E.eof = true;
    E.cv.notify_all();
}

// ---------------------------------------------------------------- the engine's settings
struct Args {
    lite::ApiKeys keys;
    int timeout = 7200;
    std::string model, alias, host = "127.0.0.1", fx_config, fx_config_json, fx_lib, image_python;
    int port = 8080;
    bool fx_args_set = false, fx_cli = false, tokenize_endpoints = false;
    int parallel = 0;                   // --parallel N / -np N (0: the run config's "parallel", else 1)
    long long long_context = 0;         // --long-context 512k|768k|1m|N: opt-in dynamic YaRN past the trained context
    std::vector<std::string> fx_args;   // after --fx-args
};

static void usage() {
    fx::say("impulse-server (fx: a Flash-Next pack) -m <pack folder | its *dense*.gguf beside pack/>\n"
                    "  [--fx-config run-config.json | --fx-config-json JSON | --fx-args ARG ...] [--fx-lib libimpulse_fx] [--fx-image-python PY]\n"
                    "  [--fx-tokenize-endpoints] [--fx-cli] [--parallel N] [--long-context 512k|768k|1m] [--alias A] [--host H] [--port P]\n"
                    "  [--api-key K[,K]]\n"
                    "  [--api-key-file F] [--timeout S]\n");
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf(); out = ss.str();
    return true;
}

static bool has_flag(const std::vector<std::string> & a, const char * f) {
    for (const auto & x : a) if (x == f) return true;
    return false;
}

static void set_env(const std::string & k, const std::string & v) {
#if defined(_WIN32)
    _putenv_s(k.c_str(), v.c_str());
#else
    setenv(k.c_str(), v.c_str(), 1);
#endif
}

// The engine's arguments and environment from Strata's run config (serve/server.py: engine_args, child_env): "args" as
// they are, "--layer-split <layer_split>" when "gpu" names several cards (and the args have none), the cards as
// CUDA_VISIBLE_DEVICES in nvidia-smi's order, then "env".  Applied to this process before anything touches CUDA: the
// engine runs in it.  server.py's other additions (parallel slots, a learned expert profile, a RoPE fallback) are not
// made here (Nebula writes the flags).
static std::string apply_run_config(std::string text, const std::string & path, std::vector<std::string> & args, json & cfg) {
    if (text.size() >= 3 && (unsigned char) text[0] == 0xEF && (unsigned char) text[1] == 0xBB && (unsigned char) text[2] == 0xBF) text.erase(0, 3);
    try { cfg = json::parse(text); } catch (const std::exception & e) { return path + ": " + e.what(); }
    if (!cfg.is_object() || !cfg.contains("args") || !cfg["args"].is_array()) return path + ": no \"args\" list";
    for (const auto & a : cfg["args"]) args.push_back(a.is_string() ? a.get<std::string>() : a.dump());
    std::vector<int> gpus;
    if (cfg.contains("gpu")) {
        if (cfg["gpu"].is_array()) { for (const auto & g : cfg["gpu"]) if (g.is_number_integer()) gpus.push_back(g.get<int>()); }
        else if (cfg["gpu"].is_number_integer()) gpus.push_back(cfg["gpu"].get<int>());
    }
    if (gpus.size() > 1 && !has_flag(args, "--peer-device") && !has_flag(args, "--layer-split")) {
        std::string ls = "auto";
        if (cfg.contains("layer_split")) ls = cfg["layer_split"].is_string() ? cfg["layer_split"].get<std::string>() : cfg["layer_split"].dump();
        args.push_back("--layer-split"); args.push_back(ls);
    }
    if (!gpus.empty()) {
        std::string v;
        for (size_t i = 0; i < gpus.size(); ++i) v += (i ? "," : "") + std::to_string(gpus[i]);
        set_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        set_env("CUDA_VISIBLE_DEVICES", v);
    }
    if (cfg.contains("env") && cfg["env"].is_object())
        for (auto it = cfg["env"].begin(); it != cfg["env"].end(); ++it)
            set_env(it.key(), it.value().is_string() ? it.value().get<std::string>() : it.value().dump());
    return "";
}

static const int PARALLEL_MAX = 8;   // a batch window holds at most 8 rows (flashnext kVerifyMaxT)

// "parallel": N -> the engine's --batch N (server.py's parallel_args): nothing when the args name --batch / --slots
// themselves or N is 1; above 8 the engine runs 8 (unless --batch-mtp), said here and passed on as asked
static void parallel_args(int n, std::vector<std::string> & args) {
    if (n <= 1 || has_flag(args, "--batch") || has_flag(args, "--slots")) return;
    if (n > PARALLEL_MAX && !has_flag(args, "--batch-mtp"))
        fx::say("[impulse fx] \"parallel\": %d - the engine runs at most %d requests together; it will use %d\n", n, PARALLEL_MAX, PARALLEL_MAX);
    args.push_back("--batch");
    args.push_back(std::to_string(n));
}

static const long long TRAINED_CTX = 262144;   // Qwen3.8 Flash-Next's trained context (qwen4exp.context_length)

static std::string arg_after(const std::vector<std::string> & a, const char * f) {
    for (size_t i = 0; i + 1 < a.size(); ++i) if (a[i] == f) return a[i + 1];
    return "";
}

// The run config's "rope" -> the engine's RoPE flags, and server.py's automatic pick (rope_fallback_args / setup's
// "rope_auto"): a --max-context past the trained context with no --rope-scaling gets dynamic YaRN where the engine runs
// it, else static YaRN with factor context / trained.
//   "rope": {"type": "none"|"linear"|"yarn"|"dynamic"|"ntk", "factor": F, "orig_ctx": N, "dynamic_step": S}   (all optional)
// The args' own --rope-scaling always wins (nothing added).  Returns "" or what is wrong; `note` says what was chosen.
static std::string rope_args(const json & cfg, std::vector<std::string> & args, int gpus, std::string & note) {
    if (has_flag(args, "--rope-scaling")) return "";
    std::string type;
    double factor = 0, step = 0;
    long long orig = 0;
    if (cfg.is_object() && cfg.contains("rope") && !cfg["rope"].is_null()) {
        const json & r = cfg["rope"];
        if (!r.is_object()) return "\"rope\" must be an object: {\"type\": none|linear|yarn|dynamic|ntk, \"factor\": F, \"orig_ctx\": N}";
        if (r.contains("type")) {
            if (!r["type"].is_string()) return "\"rope\".\"type\" must be none, linear, yarn, dynamic or ntk";
            type = r["type"].get<std::string>();
            if (type == "dynamic-yarn") type = "dynamic";
            if (type != "none" && type != "linear" && type != "yarn" && type != "dynamic" && type != "ntk" && type != "auto")
                return "\"rope\".\"type\" must be none, linear, yarn, dynamic or ntk (got \"" + type + "\")";
        }
        if (r.contains("factor") && !r["factor"].is_null()) {
            if (!r["factor"].is_number() || r["factor"].get<double>() < 1.0) return "\"rope\".\"factor\" must be a number >= 1";
            factor = r["factor"].get<double>();
        }
        if (r.contains("dynamic_step") && !r["dynamic_step"].is_null()) {
            if (!r["dynamic_step"].is_number() || r["dynamic_step"].get<double>() < 0.01 || r["dynamic_step"].get<double>() > 64)
                return "\"rope\".\"dynamic_step\" must be a number in [0.01, 64]";
            step = r["dynamic_step"].get<double>();
        }
        if (r.contains("orig_ctx") && !r["orig_ctx"].is_null()) {
            if (!r["orig_ctx"].is_number_integer() || r["orig_ctx"].get<long long>() < 1) return "\"rope\".\"orig_ctx\" must be a whole number of tokens";
            orig = r["orig_ctx"].get<long long>();
        }
    }
    const long long trained = orig > 0 ? orig : TRAINED_CTX;
    const std::string mc = arg_after(args, "--max-context");
    const long long ctx = mc.empty() ? 0 : std::atoll(mc.c_str());
    if (type.empty() || type == "auto") {
        if (ctx <= trained) {   // nothing to extend: the trained rotation (an explicit factor alone says yarn)
            if (factor <= 1.0) return "";
            type = "yarn";
        } else {
            // dynamic where the engine runs it (generate.cpp's refusals), else static yarn for the whole context
            const std::string kv = arg_after(args, "--kv");
            const bool batch = has_flag(args, "--batch") || has_flag(args, "--slots");
            const std::string bg = arg_after(args, "--batch-groups");
            const std::string pw = arg_after(args, "--pipeline-windows");
            const char * why = !kv.empty() && kv != "int8" && kv != "fp16" ? "--kv q4_0 / k8v4"
                             : batch && has_flag(args, "--batch-mtp") ? "--batch-mtp"
                             : batch && gpus > 1 && bg != "1" ? "pipelined slot groups on a layer split"
                             : !pw.empty() && pw != "0" ? "--pipeline-windows" : nullptr;
            type = why ? "yarn" : "dynamic";
            if (why) note = std::string("RoPE: --max-context ") + std::to_string(ctx) + " is past the trained " + std::to_string(trained) +
                            ": static yarn (dynamic does not run with " + why + ")";
            else note = std::string("RoPE: --max-context ") + std::to_string(ctx) + " is past the trained " + std::to_string(trained) +
                        ": dynamic yarn (none up to " + std::to_string(trained) + " positions)";
        }
    }
    char buf[64];
    if ((type == "yarn" || type == "linear" || type == "ntk") && factor <= 1.0 && ctx > trained) factor = (double) ctx / (double) trained;
    args.push_back("--rope-scaling");
    args.push_back(type);
    if (factor > 1.0 && type != "none") { std::snprintf(buf, sizeof buf, "%g", factor); args.push_back("--rope-scale"); args.push_back(buf); }
    if (orig > 0 && type != "none") { args.push_back("--yarn-orig-ctx"); args.push_back(std::to_string(orig)); }
    if (step > 0 && type == "dynamic") { std::snprintf(buf, sizeof buf, "%g", step); args.push_back("--rope-dynamic-step"); args.push_back(buf); }
    return "";
}

// Dynamic YaRN on a layer split with batch slots: the engine refuses pipelined slot groups (generate.cpp: a window in
// flight on another card while the factor moves), which a split picks by default for --batch above 1 - so a run config
// that names "rope": dynamic (or --long-context) gets --batch-groups 1 unless the args choose the groups themselves
static void dynamic_rope_groups(std::vector<std::string> & args, int gpus) {
    if (gpus <= 1 || arg_after(args, "--rope-scaling") != "dynamic" || has_flag(args, "--batch-groups")) return;
    std::string b = arg_after(args, "--batch");
    if (b.empty()) b = arg_after(args, "--slots");
    if (std::atoi(b.c_str()) <= 1) return;
    args.push_back("--batch-groups");
    args.push_back("1");
    fx::say("[impulse fx] dynamic YaRN with %s batch slots on a %d-card split: --batch-groups 1 (it does not run with pipelined slot groups)\n",
            b.c_str(), gpus);
}

// server.py main()'s checks and reads of its own config keys -> the front's settings ("" or what is wrong)
static std::string front_settings(const json & cfg, fxf::Settings & S, double & silence_s, int & parallel) {
    if (!cfg.is_object()) return "";
    if (cfg.contains("model_name") && cfg["model_name"].is_string() && !cfg["model_name"].get<std::string>().empty()) S.model = cfg["model_name"];
    if (cfg.contains("aliases")) {
        json a = cfg["aliases"];
        std::vector<std::string> list;
        if (a.is_string()) { std::string s = a; size_t p = 0; for (;;) { const size_t q = s.find(',', p); list.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p)); if (q == std::string::npos) break; p = q + 1; } }
        else if (a.is_array()) { for (const auto & x : a) { if (!x.is_string()) return "aliases: expected a list of model names"; list.push_back(x); } }
        else if (!a.is_null()) return "aliases: expected a list of model names";
        for (auto x : list) {
            x = fxf::py_strip(x);
            if (!x.empty() && x != S.model && std::find(S.aliases.begin(), S.aliases.end(), x) == S.aliases.end()) S.aliases.push_back(x);
        }
    }
    auto boolean = [&](const char * k, bool & v) -> std::string {
        if (!cfg.contains(k) || cfg[k].is_null()) return "";
        if (!cfg[k].is_boolean()) return std::string("\"") + k + "\" must be true or false";
        v = cfg[k].get<bool>();
        return "";
    };
    std::string e;
    if (!(e = boolean("stop_token_text", S.stop_token_text)).empty()) return e;
    if (!(e = boolean("tool_name_repair", S.tool_name_repair)).empty()) return e;
    if (!(e = boolean("tool_call_recovery", S.tool_call_recovery)).empty()) return e;
    if (S.tool_call_recovery) return "\"tool_call_recovery\": true is not supported by impulse-fx (docs/FX-PARITY.md)";
    S.fit_max_tokens = cfg.contains("fit_max_tokens") && cfg["fit_max_tokens"].is_boolean() && cfg["fit_max_tokens"].get<bool>();
    if (cfg.contains("repeat_stop_tokens")) {
        const json & r = cfg["repeat_stop_tokens"];
        if (r.is_boolean() || !r.is_number_integer() || r.get<long long>() < 0) return "\"repeat_stop_tokens\" must be a whole number >= 0 (0 = off)";
        S.repeat_stop_tokens = r.get<int>();
    }
    if (cfg.contains("reasoning_budget_tokens") && !cfg["reasoning_budget_tokens"].is_null()) {
        try { S.reasoning_budget_tokens = fxf::reasoning_budget(json::object(), 0).value_or(0); auto b = fxf::reasoning_budget(json{ { "reasoning_budget_tokens", cfg["reasoning_budget_tokens"] } }, 0); S.reasoning_budget_tokens = b.value_or(0); }
        catch (const std::exception & ex) { return ex.what(); }
    }
    if (cfg.contains("engine_silence_s") && !cfg["engine_silence_s"].is_null()) {
        const json & v = cfg["engine_silence_s"];
        if (v.is_boolean() || !v.is_number() || v.get<double>() < 0) return "\"engine_silence_s\" must be a number of seconds >= 0 (0 = no limit)";
        silence_s = v.get<double>();
    }
    // server.py keys that change answers and are not here: refused, never silently ignored
    for (const char * k : { "reasoning_loop_recovery", "reasoning_close_retry", "env_to_user", "replicas", "sampling", "reasoning_effort", "slot_save_path" }) {
        if (!cfg.contains(k) || cfg[k].is_null()) continue;
        const json & v = cfg[k];
        const bool off = (v.is_boolean() && !v.get<bool>()) || (v.is_object() && v.empty()) || (v.is_string() && v.get<std::string>().empty());
        if (!off) return std::string("\"") + k + "\" is not supported by impulse-fx (docs/FX-PARITY.md)";
    }
    if (cfg.contains("effort_position") && cfg["effort_position"] != "start") return "\"effort_position\": \"end\" is not supported by impulse-fx (docs/FX-PARITY.md)";
    // server.py's parallel_args (#465, opt-in): up to N requests decode together in the engine's batch slots (--batch N)
    if (cfg.contains("parallel") && !cfg["parallel"].is_null() && !(cfg["parallel"].is_boolean() && !cfg["parallel"].get<bool>())) {
        const json & v = cfg["parallel"];
        if (v.is_boolean() || !v.is_number_integer()) fx::say("[impulse fx] \"parallel\" must be a whole number of requests (2..%d), not %s: ignored\n", PARALLEL_MAX, v.dump().c_str());
        else parallel = std::max(1, v.get<int>());
    }
    return "";
}

#if !defined(_WIN32)
static std::string self_dir_posix() {
    char p[4096];
    const ssize_t n = readlink("/proc/self/exe", p, sizeof p - 1);
    if (n <= 0) return "./";
    p[n] = '\0';
    std::string s(p);
    const size_t sl = s.rfind('/');
    return sl == std::string::npos ? "./" : s.substr(0, sl + 1);
}
#endif

// the GGUF's tokenizer.ggml.merges and tokenizer.chat_template (a metadata read: no tensors)
static bool gguf_tokenizer_meta(const std::string & path, std::vector<std::string> & merges, std::string & tmpl) {
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return false;
    const int64_t m = gguf_find_key(g, "tokenizer.ggml.merges");
    if (m >= 0) {
        const size_t n = gguf_get_arr_n(g, m);
        merges.reserve(n);
        for (size_t i = 0; i < n; ++i) merges.push_back(gguf_get_arr_str(g, m, i));
    }
    const int64_t t = gguf_find_key(g, "tokenizer.chat_template");
    if (t >= 0) tmpl = gguf_get_val_str(g, t);
    gguf_free(g);
    return m >= 0 && t >= 0;
}

// ---------------------------------------------------------------- one request's engine pass (StrataEngine.generate)
using fxs::EngineDied;        // (src/fx/fx_slots.hpp: the batch slots' scheduler throws the same)
using fxs::EngineSilent;
using fxs::EngineNonFinite;
using fxs::EngineErr;         // the engine's ERR (server.py: ValueError)

static const double PP_CHUNK_MAX = 32768, PP_FLOOR_TOK_S = 50.0, PP_SLACK = 3.0;

// _parse_done
static json parse_done(const std::string & line) {
    std::istringstream is(line);
    std::vector<std::string> f;
    for (std::string w; is >> w;) f.push_back(w);
    json last = json::object();
    auto I = [&](size_t i) { return (long long) std::atoll(f[i].c_str()); };
    auto D = [&](size_t i) { return std::atof(f[i].c_str()); };
    if (f.size() < 6) return last;
    last["generated"] = I(1); last["prompt_tokens"] = I(2); last["prompt_ms"] = D(3); last["decode_ms"] = D(4); last["finish"] = f[5];
    if (f.size() >= 9) { last["drafts_accepted"] = I(6); last["drafts_offered"] = I(7); last["reused"] = I(8); }
    if (f.size() >= 11) { last["hits"] = I(9); last["lookups"] = I(10); }
    if (f.size() >= 14) { last["ram_blobs"] = I(11); last["file_blobs"] = I(12); last["file_mb"] = D(13); }
    if (f.size() >= 15) last["prompt_read"] = I(14);
    if (f.size() >= 16) last["offloaded"] = I(15);
    return last;
}

// The pass: `head` (GEN/GENI and the keys) and the ids; on_token(id) -> false ends the pass early (STOP, drained to its
// DONE: generate()'s finally), on_ping() for a heartbeat while the engine reads (PP lines, 10 s of quiet), cancel() a
// client that left.  The DONE line's figures go to `last`.
struct Pass {
    Engine & E;
    double silence_s;
    json last;
    bool got_done = false;
    void run(const std::string & head, const std::vector<int32_t> & ids, const std::function<bool(int32_t)> & on_token,
             const std::function<void()> & on_ping, const std::function<bool()> & cancelled) {
        std::string line = head + " ";
        line.reserve(line.size() + ids.size() * 7);
        for (size_t i = 0; i < ids.size(); ++i) { if (i) line += ','; line += std::to_string(ids[i]); }
        { std::lock_guard<std::mutex> lk(E.mu); E.lines.clear(); E.reading = true; }
        struct Unread { Engine & E; ~Unread() { std::lock_guard<std::mutex> lk(E.mu); E.reading = false; E.lines.clear(); } } unread{ E };
        if (!E.send(line)) throw EngineDied("the engine stopped unexpectedly");
        bool done = false;
        double allow = silence_s > 0 ? silence_s + std::min<double>((double) ids.size(), PP_CHUNK_MAX) / PP_FLOOR_TOK_S : 0.0;
        auto heard = std::chrono::steady_clock::now();
        auto beat = heard;
        long long read_to = 0;
        auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
        bool stop_early = false;
        try {
            while (!done) {
                if (allow > 0 && since(heard) >= allow)
                    throw EngineSilent("the engine said nothing for " + std::to_string((int) since(heard)) + " s during the request; the server ended the engine");
                std::string l;
                if (!E.next(l, 0.5)) {
                    if (E.ended()) throw EngineDied("the engine stopped unexpectedly");
                    if (cancelled()) { stop_early = true; break; }
                    if (since(beat) >= 10.0) { beat = std::chrono::steady_clock::now(); on_ping(); }
                    continue;
                }
                heard = beat = std::chrono::steady_clock::now();
                if (l.rfind("T ", 0) == 0) {
                    allow = silence_s;
                    if (cancelled()) { stop_early = true; break; }
                    if (!on_token((int32_t) std::atol(l.c_str() + 2))) { stop_early = true; break; }
                } else if (l.rfind("PP ", 0) == 0) {
                    std::istringstream is(l.substr(3));
                    long long a = -1, b = -1; double ms = 0, rate = 0;
                    is >> a >> b;
                    if (is >> ms) is >> rate;
                    if (a >= 0 && b >= 0) {
                        const long long chunk = a - read_to;
                        read_to = a;
                        if (silence_s > 0 && rate > 0 && chunk > 0) allow = std::max(silence_s, PP_SLACK * (double) chunk / rate);
                    }
                    if (cancelled()) { stop_early = true; break; }
                    on_ping();
                } else if (l.rfind("RESUME ", 0) == 0) {
                    read_to = std::atoll(l.c_str() + 7);
                } else if (l.rfind("DONE", 0) == 0) {
                    last = parse_done(l);
                    got_done = true;
                    done = true;
                    if (last.value("finish", std::string()) == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                } else if (l.rfind("ERR", 0) == 0) {
                    done = true;
                    throw EngineErr(l.size() > 4 ? l.substr(4) : std::string());
                } else if (l.rfind("INFO ", 0) != 0) {
                    fx::say("impulse fx engine: %s\n", l.c_str());
                }
            }
        } catch (const EngineSilent &) {
            throw;
        } catch (...) {
            if (!done) drain(allow);
            throw;
        }
        if (stop_early) drain(allow);
    }
    // the consumer stopped early: STOP, then the lines up to this request's DONE (never an untimed wait, #481)
    void drain(double allow) {
        if (E.can_stop_) E.send("STOP");
        auto heard = std::chrono::steady_clock::now();
        for (;;) {
            const double left = allow > 0 ? allow - std::chrono::duration<double>(std::chrono::steady_clock::now() - heard).count() : 1e9;
            if (left <= 0) throw EngineSilent("the engine did not finish the request after it was stopped (STOP) within " + std::to_string((int) allow) + " s; the server ended the engine");
            std::string l;
            if (!E.next(l, std::min(left, 1.0))) { if (E.ended()) return; continue; }
            if (l.rfind("ERR", 0) == 0) return;
            if (l.rfind("DONE", 0) == 0) { last = parse_done(l); got_done = true; return; }
            if (!E.can_stop_) heard = std::chrono::steady_clock::now();
        }
    }
};

// request_stats: the first pass's prompt accounting, every pass's work summed
static json request_stats(const std::vector<json> & segments) {
    if (segments.empty()) return json::object();
    json r = segments.back();
    for (const char * k : { "reused", "prompt_read" }) { if (segments.front().contains(k)) r[k] = segments.front()[k]; else r.erase(k); }
    for (const char * k : { "prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted", "hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb" }) {
        bool any = false;
        double sum = 0;
        bool is_float = false;
        for (const auto & s : segments) if (s.contains(k) && !s[k].is_null()) { any = true; sum += s[k].get<double>(); is_float |= s[k].is_number_float(); }
        if (any) r[k] = is_float ? json(sum) : json((long long) sum);
    }
    return r;
}

// a batched request's engine legs (fx_slots.hpp) as one DONE's figures: each leg its DONE, a slot leg's BDONE in place of
// the admission's finish / decode time / tokens (server.py _apply_bdone), the legs then summed as passes are
static json legs_last(const std::vector<fxs::Leg> & legs) {
    std::vector<json> segs;
    for (const auto & l : legs) {
        json j = parse_done(l.done);
        if (!l.bdone.empty()) {
            std::istringstream is(l.bdone);
            std::vector<std::string> f;
            for (std::string w; is >> w;) f.push_back(w);
            if (f.size() >= 5) {
                j["finish"] = f[3];
                j["decode_ms"] = std::atof(f[4].c_str());
                j["generated"] = (long long) std::atoll(f[2].c_str());
            }
        }
        segs.push_back(j);
    }
    return request_stats(segs);
}

static json round_to(double v, int digits) {   // Python's round(v, digits) for the timings (a float)
    const double p = std::pow(10.0, digits);
    return std::nearbyint(v * p) / p;
}

// request_timings: llama.cpp's names, from the engine's own clock
static json request_timings(long long prompt_tokens, long long generated, const json & last) {
    if (!last.contains("prompt_ms")) return nullptr;
    long long cache_n = last.value("reused", 0LL);
    if (last.contains("prompt_read") && last.value("finish", std::string()) != "cancel")
        cache_n = std::max(cache_n, std::max(0LL, prompt_tokens - last["prompt_read"].get<long long>()));
    const long long prompt_n = std::max(0LL, prompt_tokens - cache_n);
    const double prompt_ms = last["prompt_ms"].get<double>(), decode_ms = last.value("decode_ms", 0.0);
    const long long decoded = last.value("generated", 0LL) ? last["generated"].get<long long>() : generated;
    json t = json::object();
    t["cache_n"] = cache_n; t["prompt_n"] = prompt_n; t["prompt_ms"] = round_to(prompt_ms, 1);
    t["prompt_per_token_ms"] = prompt_n ? round_to(prompt_ms / (double) prompt_n, 3) : json(nullptr);
    t["prompt_per_second"] = prompt_n && prompt_ms > 0 ? round_to((double) prompt_n / (prompt_ms / 1000), 1) : json(nullptr);
    t["predicted_n"] = generated; t["predicted_ms"] = round_to(decode_ms, 1);
    t["predicted_per_token_ms"] = decoded ? round_to(decode_ms / (double) decoded, 3) : json(nullptr);
    t["predicted_per_second"] = decoded && decode_ms > 0 ? round_to((double) decoded / (decode_ms / 1000), 1) : json(nullptr);
    if (last.contains("drafts_offered")) { t["draft_n"] = last["drafts_offered"]; t["draft_n_accepted"] = last["drafts_accepted"]; }
    return t;
}

static const std::string REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n";
static const std::string EXAMPLE_TURN = "in a turn of the chat format written out as an example";

// what Service._run yields: an event, a ping, or the end ("done")
struct RunOut {
    std::string finish = "length";
    long long completion_tokens = 0, reasoning_tokens = 0, reused = 0;
    json timings;
    std::optional<std::string> stop_sequence;
};

struct Service {
    Engine & E;
    fxf::Front & F;
    fxv::Vision * vision = nullptr;
    double silence_s = 300.0;
    std::mutex fifo;                    // the engine's single turn (with batch slots: the image encoder's only)
    fxs::Scheduler * sched = nullptr;   // --batch (\"parallel\": N): the slots' scheduler instead of the single turn
    std::atomic<bool> busy{ false };
    std::atomic<long long> in_use{ 0 };  // the last request's prompt tokens (/slots' n_prompt_tokens)
    std::atomic<bool> dying{ false };

    // Service._run for one prepared request.  `emit(ev)` gets each event (false: the client left), `ping()` a heartbeat
    // (false: the client left).  Returns the "done" figures; throws EngineDied / EngineErr / EngineNonFinite /
    // std::invalid_argument (a call the parser cannot read: server.py's ValueError)
    RunOut run(const fxf::Prompt & P, const json & req, const std::string & embeddings, const std::vector<std::string> & img_keys,
               const std::function<bool(const fxf::Event &)> & emit, const std::function<bool()> & ping,
               const std::function<bool()> & client_gone) {
        RunOut R;
        json sampling = req;
        // uncensored_alias: this fork's "uncensored" is experimental_speed_projection (the explicit key wins)
        if (sampling.is_object() && sampling.contains("uncensored") && sampling["uncensored"].is_boolean() && !sampling.contains("experimental_speed_projection"))
            sampling["experimental_speed_projection"] = sampling["uncensored"];
        const bool thinking = P.thinking;
        const long long max_new = P.max_new;
        std::optional<long long> budget = thinking ? fxf::reasoning_budget(sampling, F.cfg.reasoning_budget_tokens) : std::nullopt;
        if (budget && max_new > 0) {
            const long long reserve = std::min(std::max(512LL, max_new / 4), max_new / 2);
            if (*budget > max_new - reserve) {
                fx::say("[impulse fx] thinking capped at %lld of max_tokens %lld to leave room for the answer (budget %lld)\n", max_new - reserve, max_new, *budget);
                budget = std::max(1LL, max_new - reserve);
            }
        }
        fxf::OutputParser parser(thinking, P.tools, true, F.cfg.tool_name_repair);
        fxf::Utf8Decoder detok;
        long long n = 0;
        std::string finish = "length";
        int32_t run_tok = -1;
        long long run_len = 0;
        bool repeated = false;
        long long thinking_n = 0;
        const auto stops_list = fxf::stop_strings(sampling);
        std::optional<fxf::StopMatcher> stops;
        if (!stops_list.empty()) stops = fxf::StopMatcher{ stops_list, "", std::nullopt };
        bool cancel = false;
        auto cut = [&](std::vector<fxf::Event> evs) {
            if (!stops) return evs;
            std::vector<fxf::Event> out;
            for (auto & ev : evs) {
                if (stops->hit) break;
                if (ev.kind == "content") { const std::string t = stops->push(ev.text); if (!t.empty()) out.push_back(fxf::Event{ "content", t, nullptr }); continue; }
                const std::string held = stops->flush();
                if (!held.empty()) out.push_back(fxf::Event{ "content", held, nullptr });
                out.push_back(ev);
            }
            return out;
        };
        std::string tail;   // the last two characters written
        auto push_tail = [&](const std::string & piece) {
            tail += piece;
            size_t n1 = 0, n2 = 0;
            fxf::utf8_last_cp(tail, tail.size(), &n1);
            if (n1 && tail.size() > n1) fxf::utf8_last_cp(tail, tail.size() - n1, &n2);
            const size_t keep = n1 + n2;
            if (tail.size() > keep && keep) tail = tail.substr(tail.size() - keep);
        };
        auto send = [&](const std::vector<fxf::Event> & evs) {
            for (const auto & ev : evs) if (!emit(ev)) cancel = true;
        };
        auto in_reasoning = [&]() { return parser.state == "reasoning" || parser.state == "rcall"; };
        auto written = [&](const std::vector<int32_t> & ids) {
            for (auto x : ids) {
                ++n;
                thinking_n += in_reasoning();
                const std::string piece = detok.push(F.tok.token_text(x));
                push_tail(piece);
                send(cut(parser.feed(piece)));
            }
        };
        std::string force = P.force;
        std::vector<fxf::Event> opening;
        if (!force.empty() && !thinking) { opening = parser.feed(force); force.clear(); }
        bool nf_retried = false;
        int literals = 0;
        bool relit = false;
        std::vector<int32_t> owed;
        int examples = 0;
        if (!embeddings.empty() && !img_keys.empty()) sampling["strata_img_keys"] = img_keys;
        std::vector<json> segments;
        const std::set<int32_t> stop_ids = { F.tok.im_end, F.tok.endoftext };
        // the single turn: a queued request sends a keep-alive every 10 s (#1619) and gives its place up when it leaves
        // (with batch slots the scheduler queues the request instead, in arrival order: fx_slots.hpp)
        std::unique_lock<std::mutex> turn(fifo, std::defer_lock);
        if (!sched) {
            auto waited = std::chrono::steady_clock::now();
            while (!turn.try_lock()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                if (client_gone()) { R.finish = "cancel"; return R; }
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - waited).count() >= 10.0) {
                    waited = std::chrono::steady_clock::now();
                    if (!ping()) { R.finish = "cancel"; return R; }
                }
            }
        }
        if (!sched) busy = true;
        in_use = (long long) P.ids.size();
        struct NotBusy { std::atomic<bool> & b; bool on; ~NotBusy() { if (on) b = false; } } not_busy{ busy, !sched };
        send(opening);
        std::vector<int32_t> prompt = P.ids;
        long long thought = 0;
        auto last_out = std::chrono::steady_clock::now();
        try {
            for (;;) {
                std::vector<int32_t> seg;
                bool wrap = false, nonfinite = false, opens = false;
                const size_t segments_before = segments.size();
                Pass pass{ E, silence_s, json::object() };
                const std::string head = embeddings.empty() ? "GEN " + std::to_string(max_new - n) + fxf::sampling_keys(sampling)
                                                           : "GENI " + std::to_string(max_new - n) + fxf::sampling_keys(sampling) + " " + embeddings;
                auto on_token = [&](int32_t t) -> bool {
                    ++n;
                    if (stop_ids.count(t)) {
                        if (!force.empty() && parser.state == "reasoning" && parser.buf == "</think>") { opens = true; return false; }
                        if (!owed.empty()) { --n; written(owed); ++n; owed.clear(); }
                        // EOS literal (stop_as_text)
                        if (F.cfg.stop_token_text && literals < 64 && !detok.pending()) {
                            const std::string why = parser.literal_context(examples > 0);
                            if (!why.empty()) {
                                fxf::Utf8Decoder d;
                                const std::string text = d.push(F.tok.token_text(t), true);
                                std::vector<int32_t> lit = F.tok.encode(text, false);
                                const long long room = max_new - n + 1;
                                bool ok = !lit.empty() && (long long) lit.size() + 1 <= room;
                                for (auto i : lit) if (stop_ids.count(i)) ok = false;
                                if (ok) {
                                    if (lit.size() > 1) { owed = { lit.back() }; lit.pop_back(); } else owed.clear();
                                    ++literals;
                                    examples -= why == EXAMPLE_TURN;
                                    fx::say("[impulse fx] the model wrote %s's id %s: kept as its text and the reply goes on (stop_token_text)\n", text.c_str(), why.c_str());
                                    --n;
                                    written(lit);
                                    seg.insert(seg.end(), lit.begin(), lit.end());
                                    if (stops && stops->hit) finish = "stop"; else relit = true;
                                    return false;
                                }
                            }
                        }
                        finish = "stop";
                        return false;
                    }
                    owed.clear();
                    seg.push_back(t);
                    if (t == F.tok.im_start && parser.example_place()) ++examples;
                    thinking_n += in_reasoning();
                    run_len = t == run_tok ? run_len + 1 : 1;
                    run_tok = t;
                    if (F.cfg.repeat_stop_tokens && run_len >= F.cfg.repeat_stop_tokens) { repeated = true; return false; }
                    const std::string piece = detok.push(F.tok.token_text(t));
                    push_tail(piece);
                    auto evs = cut(parser.feed(piece));
                    send(evs);
                    if (!evs.empty()) last_out = std::chrono::steady_clock::now();
                    else if (std::chrono::duration<double>(std::chrono::steady_clock::now() - last_out).count() >= 10.0) {
                        last_out = std::chrono::steady_clock::now();
                        if (!ping()) cancel = true;
                    }
                    if (stops && stops->hit) { finish = "stop"; return false; }
                    if (budget && parser.state == "reasoning") {
                        ++thought;
                        if (thought >= *budget && parser.buf.empty() && !detok.pending()) { wrap = true; return false; }
                    }
                    if (!force.empty()) {
                        bool own_call = false;
                        for (const auto & ev : evs) if (ev.kind == "tool_start" || ev.kind == "tool_call") own_call = true;
                        if (own_call) force.clear();
                        else if (parser.state == "content" && parser.buf.empty() && !detok.pending()) { opens = true; return false; }
                    }
                    return !cancel;
                };
                try {
                    auto on_ping = [&]() { last_out = std::chrono::steady_clock::now(); if (!ping()) cancel = true; };
                    auto gone = [&]() { return cancel || client_gone(); };
                    if (sched) {
                        fxs::Request q{ max_new - n, fxf::sampling_keys(sampling), embeddings, prompt };
                        const fxs::Result res = sched->generate(q, fxs::Hooks{ on_token, on_ping, gone });
                        if (!res.legs.empty()) { pass.last = legs_last(res.legs); pass.got_done = true; }
                    } else pass.run(head, prompt, on_token, on_ping, gone);
                } catch (const EngineNonFinite &) {
                    nonfinite = true;
                }
                if (pass.got_done) segments.push_back(pass.last);
                if (client_gone()) cancel = true;
                if (nonfinite && !cancel) {
                    if (n == 0 && !nf_retried) {
                        nf_retried = true;
                        segments.resize(segments_before);
                        fx::say("[impulse fx] the engine's logits became non-finite before the first token (#879); sending the request again once\n");
                        continue;
                    }
                    if (n == 0) throw EngineNonFinite("the model's output became non-finite (NaN) twice in a row (#879); the engine has dropped its cached state - send the request again");
                    fx::say("[impulse fx] the engine's logits became non-finite after %lld tokens (#879): the reply ends here as \"length\"\n", n);
                    finish = "length";
                    break;
                }
                if (relit) { relit = false; if (!cancel) { prompt.insert(prompt.end(), seg.begin(), seg.end()); continue; } }
                if (!(wrap || opens) || cancel) break;
                std::string text;
                if (wrap) { budget.reset(); text = REASONING_WRAP_UP + force; }
                else {
                    size_t nl = 0;
                    for (size_t k = tail.size(); k > 0 && tail[k - 1] == '\n'; --k) ++nl;
                    text = std::string(2 - std::min<size_t>(2, nl), '\n') + force;
                }
                force.clear();
                const auto extra = F.tok.encode(text, true);
                if (max_new - n - (long long) extra.size() < 1) break;
                if (wrap) fx::say("[impulse fx] thinking budget reached (%lld tokens): wrapping up the thinking\n", thought);
                for (auto t : extra) {
                    ++n;
                    thinking_n += in_reasoning();
                    send(cut(parser.feed(detok.push(F.tok.token_text(t)))));
                }
                if (stops && stops->hit) { finish = "stop"; break; }
                prompt.insert(prompt.end(), seg.begin(), seg.end());
                prompt.insert(prompt.end(), extra.begin(), extra.end());
            }
            if (cancel) finish = "cancel";
            else if (repeated)
                fx::say("[impulse fx] the reply repeated one token %lld times in a row: ended as \"length\" (repeat_stop_tokens)\n", run_len);
        } catch (const EngineSilent &) {
            dying = true;
            throw;
        }
        const json last = request_stats(segments);
        long long seen = (long long) P.ids.size();
        if (last.contains("prompt_read") && last.value("finish", std::string()) == "cancel")
            seen = std::min(seen, last.value("reused", 0LL) + last["prompt_read"].get<long long>());
        if (!segments.empty()) R.timings = request_timings(seen, n, last);
        fx::say("[impulse fx] done: %lld tokens (%s)\n", n, finish.c_str());
        if (turn.owns_lock()) turn.unlock();
        for (const auto & ev : cut(parser.finish(finish))) if (!emit(ev)) cancel = true;
        if (stops && !stops->hit && !stops->held.empty()) emit(fxf::Event{ "content", stops->flush(), nullptr });
        R.finish = finish;
        R.completion_tokens = n;
        R.reasoning_tokens = thinking_n;
        R.reused = R.timings.is_object() ? R.timings.value("cache_n", 0LL) : 0;
        if (stops && stops->hit) R.stop_sequence = *stops->hit;
        return R;
    }
};

// ---------------------------------------------------------------- openai_chunks / openai_collect
struct Chunker {
    std::string cid, model;
    long long created;
    int calls = 0;
    std::map<std::string, int> streamed;
    std::set<std::string> finished;
    json chunk(const json & delta, const json & finish = nullptr) const {
        return json{ { "id", cid }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", model },
                     { "choices", json::array({ json{ { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
    }
    // an event -> its chunk (null: none)
    json event(const fxf::Event & ev) {
        if (ev.kind == "reasoning") return ev.text.empty() ? json(nullptr) : chunk(json{ { "reasoning_content", ev.text } });
        if (ev.kind == "content") return ev.text.empty() ? json(nullptr) : chunk(json{ { "content", ev.text } });
        if (ev.kind == "tool_start") {
            streamed[ev.call->id] = calls++;
            return chunk(json{ { "tool_calls", json::array({ json{ { "index", streamed[ev.call->id] }, { "id", ev.call->id }, { "type", "function" },
                                                                   { "function", json{ { "name", ev.call->name }, { "arguments", "" } } } } }) } });
        }
        if (ev.kind == "tool_args")
            return chunk(json{ { "tool_calls", json::array({ json{ { "index", streamed[ev.call->id] }, { "function", json{ { "arguments", ev.text } } } } }) } });
        if (ev.kind == "tool_call") {
            if (streamed.count(ev.call->id)) { finished.insert(ev.call->id); return nullptr; }
            json c = chunk(json{ { "tool_calls", json::array({ json{ { "index", calls }, { "id", ev.call->id }, { "type", "function" },
                                                                   { "function", json{ { "name", ev.call->name }, { "arguments", fxf::py_dumps(ev.call->arguments, false) } } } } }) } });
            ++calls;
            return c;
        }
        return nullptr;
    }
    json final_chunk(const RunOut & R, long long prompt_tokens) const {
        bool whole = calls > 0;
        for (const auto & [id, idx] : streamed) if (!finished.count(id)) whole = false;
        const std::string finish = whole && R.finish == "stop" ? "tool_calls" : R.finish == "cancel" ? "stop" : R.finish;
        json last = chunk(json::object(), finish);
        last["usage"] = json{ { "prompt_tokens", prompt_tokens }, { "completion_tokens", R.completion_tokens },
                              { "total_tokens", prompt_tokens + R.completion_tokens },
                              { "prompt_tokens_details", json{ { "cached_tokens", R.reused } } } };
        if (R.timings.is_object()) last["timings"] = R.timings;
        return last;
    }
};

static bool is_json_text(const std::string & s) { return !json::parse(s, nullptr, false).is_discarded(); }

static json openai_collect(const std::vector<json> & chunks) {
    std::string content, reasoning;
    std::map<long long, json> by_index;
    const json * last = nullptr;
    for (const auto & c : chunks) {
        const json & d = c["choices"][0]["delta"];
        if (d.contains("content") && d["content"].is_string()) content += d["content"].get<std::string>();
        if (d.contains("reasoning_content") && d["reasoning_content"].is_string()) reasoning += d["reasoning_content"].get<std::string>();
        if (d.contains("tool_calls")) {
            for (const auto & tc : d["tool_calls"]) {
                const long long idx = tc.contains("index") ? tc["index"].get<long long>() : (long long) by_index.size();
                if (!by_index.count(idx)) by_index[idx] = json{ { "id", nullptr }, { "type", "function" }, { "function", json{ { "name", "" }, { "arguments", "" } } } };
                json & cur = by_index[idx];
                if (tc.contains("id") && tc["id"].is_string() && !tc["id"].get<std::string>().empty()) cur["id"] = tc["id"];
                const json fn = tc.contains("function") ? tc["function"] : json::object();
                if (fn.contains("name") && fn["name"].is_string()) cur["function"]["name"] = cur["function"]["name"].get<std::string>() + fn["name"].get<std::string>();
                if (fn.contains("arguments") && fn["arguments"].is_string()) cur["function"]["arguments"] = cur["function"]["arguments"].get<std::string>() + fn["arguments"].get<std::string>();
            }
        }
        last = &c;
    }
    json calls = json::array();
    const json finish = (*last)["choices"][0]["finish_reason"];
    for (auto & [i, c] : by_index) if (finish == "tool_calls" || is_json_text(c["function"]["arguments"].get<std::string>())) calls.push_back(c);
    json msg = json{ { "role", "assistant" }, { "content", content.empty() ? json(nullptr) : json(content) } };
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    if (!calls.empty()) msg["tool_calls"] = calls;
    json out = json{ { "id", (*last)["id"] }, { "object", "chat.completion" }, { "created", (*last)["created"] }, { "model", (*last)["model"] },
                     { "choices", json::array({ json{ { "index", 0 }, { "message", msg }, { "finish_reason", finish } } }) },
                     { "usage", (*last)["usage"] } };
    if (last->contains("timings")) out["timings"] = (*last)["timings"];
    return out;
}

static json err_body(const std::string & type, const std::string & message) {
    return json{ { "error", json{ { "type", type }, { "message", message } } } };
}

// ChatTemplate._detect_caps: llama.cpp's capability names, from this template (hints for /props)
static json detect_caps(const fxf::Front & F) {
    auto render = [&](const json & messages, const json & tools) -> std::string {
        try { return F.render(messages, tools, json::object(), true); } catch (...) { return ""; }
    };
    const json user = json{ { "role", "user" }, { "content", "strata_caps_user" } };
    json tools = json::array();
    for (int i = 0; i < 2; ++i)
        tools.push_back(json{ { "name", "strata_caps_call_" + std::to_string(i) }, { "description", "strata_caps_description" },
                              { "parameters", json{ { "type", "object" }, { "properties", json{ { "arg", json{ { "type", "string" } } } } } } } });
    const std::string tool_prompt = render(json::array({ user }), tools);
    auto calls_supported = [&](int count) {
        json calls = json::array(), msgs = json::array({ user });
        for (int i = 0; i < count; ++i) calls.push_back(json{ { "function", json{ { "name", "strata_caps_call_" + std::to_string(i) }, { "arguments", json{ { "arg", "strata_caps_arg_" + std::to_string(i) } } } } } });
        msgs.push_back(json{ { "role", "assistant" }, { "content", "" }, { "tool_calls", calls } });
        for (int i = 0; i < count; ++i) msgs.push_back(json{ { "role", "tool" }, { "content", "strata_caps_result_" + std::to_string(i) } });
        msgs.push_back(user);
        const std::string p = render(msgs, tools);
        int found = 0;
        for (int i = 0; i < count; ++i) {
            const std::string want = "<tool_call>\n<function=strata_caps_call_" + std::to_string(i) + ">\n<parameter=arg>\nstrata_caps_arg_" +
                                     std::to_string(i) + "\n</parameter>\n</function>\n</tool_call>";
            if (p.find(want) != std::string::npos && p.find("strata_caps_result_" + std::to_string(i)) != std::string::npos) ++found;
        }
        return found == count;
    };
    const json history = json::array({ user, json{ { "role", "assistant" }, { "content", "strata_caps_answer" }, { "reasoning_content", "strata_caps_reasoning" } }, user });
    auto has_all = [&](const std::string & s, std::initializer_list<const char *> ws) { for (auto w : ws) if (s.find(w) == std::string::npos) return false; return true; };
    return json{ { "supports_tools", has_all(tool_prompt, { "strata_caps_call_0", "strata_caps_description", "<tool_call>", "<function=" }) },
                 { "supports_tool_calls", calls_supported(1) },
                 { "supports_system_role", render(json::array({ json{ { "role", "system" }, { "content", "strata_caps_system" } }, user }), nullptr).find("strata_caps_system") != std::string::npos },
                 { "supports_parallel_tool_calls", calls_supported(2) },
                 { "supports_preserve_reasoning", render(history, nullptr).find("strata_caps_reasoning") != std::string::npos } };
}

// ---------------------------------------------------------------- the front's test mode (tests/fx_front_parity.py)
// impulse-fx --fx-front-test <dense.gguf> [--context N]: reads JSON lines on stdin, one answer line each, no engine.
//   {"op": "prepare", "req": {...}, "images": {"<source>": n tokens, ...}}  -> {"ids", "text", "max_new", "thinking", "force", "head"}
//   {"op": "parse", "thinking": b, "tools": [...], "deltas": [...], "finish": "stop"} -> {"events": [...]}
//   {"op": "encode", "text": "...", "parse_special": b, "plain": [[a, b], ...]}   -> {"ids"}
//   {"op": "tokens", "ids": [...], "req": {...}, "prompt_tokens": N, "thinking": b, "tools": [...], "max_new": N, "force": ""}
//                                                                               -> the chat.completion (Service.run on a
//                                                                                  scripted engine: the ids as its tokens)
static int front_test(int argc, char ** argv) {
    std::string gguf;
    long long ctx = 262144;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--fx-front-test") && i + 1 < argc) gguf = argv[++i];
        else if (!std::strcmp(argv[i], "--context") && i + 1 < argc) ctx = std::atoll(argv[++i]);
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices = no_devices;
    llama_model * model = llama_model_load_from_file(gguf.c_str(), mp);
    if (!model) { fx::say("cannot load %s\n", gguf.c_str()); return 1; }
    std::vector<std::string> merges;
    std::string tmpl;
    if (!gguf_tokenizer_meta(gguf, merges, tmpl)) { fx::say("%s: no merges / chat template\n", gguf.c_str()); return 1; }
    fxf::Front F;
    F.init(llama_model_get_vocab(model), merges, tmpl);
    std::string line;
    while (std::getline(std::cin, line)) {
        json q, a = json::object();
        try {
            q = json::parse(line);
            const std::string op = q.value("op", std::string());
            if (q.contains("model_name")) F.cfg.model = q["model_name"].get<std::string>();
            if (q.contains("aliases")) F.cfg.aliases = q["aliases"].get<std::vector<std::string>>();
            if (op == "prepare") {
                json req = q["req"];
                const json imgs = q.value("images", json::object());
                fxf::Front::ImageEncoder enc = nullptr;
                if (q.contains("images")) enc = [&](const std::vector<std::string> & srcs) {
                    std::vector<std::pair<std::string, int>> out;
                    for (const auto & s : srcs) {
                        if (!imgs.contains(s)) throw fxf::BadRequest("the image could not be read: " + s.substr(0, 40));
                        out.push_back({ fxv::sha256_hex(s).substr(0, 16), imgs[s].get<int>() });
                    }
                    return out;
                };
                if (q.contains("config")) {
                    const json c = q["config"];
                    if (c.contains("fit_max_tokens")) F.cfg.fit_max_tokens = c["fit_max_tokens"];
                }
                fxf::Prompt P = F.prepare(req, q.value("context", ctx), enc);
                a["ids"] = P.ids; a["text"] = P.text; a["max_new"] = P.max_new; a["thinking"] = P.thinking; a["force"] = P.force;
                a["head"] = "GEN " + std::to_string(P.max_new) + fxf::sampling_keys(req);
                a["structured"] = P.structured;
                a["messages"] = P.messages; a["tools"] = P.tools; a["kwargs"] = P.kwargs;
            } else if (op == "encode") {
                std::vector<std::pair<size_t, size_t>> plain;
                for (const auto & p : q.value("plain", json::array())) plain.push_back({ p[0].get<size_t>(), p[1].get<size_t>() });
                a["ids"] = F.tok.encode(q["text"].get<std::string>(), q.value("parse_special", true), plain);
                a["bpe"] = F.tok.bpe_plain(q["text"].get<std::string>());
            } else if (op == "parse") {
                fxf::OutputParser p(q.value("thinking", true), q.value("tools", json(nullptr)), q.value("stream_tools", true), q.value("repair_names", false));
                json evs = json::array();
                auto put = [&](const std::vector<fxf::Event> & es) {
                    for (const auto & e : es) {
                        json j = json{ { "kind", e.kind }, { "text", e.text } };
                        if (e.call) { j["name"] = e.call->name; j["arguments"] = e.call->arguments; }
                        evs.push_back(j);
                    }
                };
                try {
                    for (const auto & d : q["deltas"]) put(p.feed(d.get<std::string>()));
                    put(p.finish(q.value("finish", std::string("stop"))));
                } catch (const std::invalid_argument & e) { a["error"] = e.what(); }
                a["events"] = evs;
            } else if (op == "tokens") {
                // Service.run with a scripted engine: one list of ids per pass
                fxf::Prompt P;
                P.ids = q["prompt"].get<std::vector<int32_t>>();
                P.thinking = q.value("thinking", true);
                P.tools = q.value("tools", json(nullptr));
                P.max_new = q.value("max_new", 4096LL);
                P.force = q.value("force", std::string());
                F.cfg.stop_token_text = true; F.cfg.repeat_stop_tokens = 256; F.cfg.tool_name_repair = false;   // each case its own
                if (q.contains("config")) {
                    const json c = q["config"];
                    if (c.contains("stop_token_text")) F.cfg.stop_token_text = c["stop_token_text"];
                    if (c.contains("repeat_stop_tokens")) F.cfg.repeat_stop_tokens = c["repeat_stop_tokens"];
                    if (c.contains("tool_name_repair")) F.cfg.tool_name_repair = c["tool_name_repair"];
                }
                std::deque<std::vector<int32_t>> passes;
                for (const auto & p : q["passes"]) passes.push_back(p.get<std::vector<int32_t>>());
                json gens = json::array();
                // a scripted engine: a thread that answers each GEN line with the next pass's ids
                Engine E;
                E.ready = true; E.can_stop_ = true; E.max_ctx = ctx;
#if !defined(_WIN32)
                int pin[2];
                if (pipe(pin) != 0) return 1;
                E.in_w = pin[1];
                std::thread eng([&]() {
                    std::string buf; char b[65536];
                    for (;;) {
                        const ssize_t r = read(pin[0], b, sizeof b);
                        if (r <= 0) break;
                        buf.append(b, (size_t) r);
                        size_t nl;
                        while ((nl = buf.find('\n')) != std::string::npos) {
                            const std::string l = buf.substr(0, nl);
                            buf.erase(0, nl + 1);
                            if (l == "QUIT") return;
                            if (l.rfind("GEN", 0) != 0) continue;
                            gens.push_back(l.substr(0, l.rfind(' ')) + " #" + std::to_string(std::count(l.begin() + (long) l.rfind(' '), l.end(), ',') + 1));
                            const auto ids = passes.empty() ? std::vector<int32_t>{} : passes.front();
                            if (!passes.empty()) passes.pop_front();
                            std::lock_guard<std::mutex> lk(E.mu);
                            E.lines.push_back("RESUME 0");
                            for (auto t : ids) E.lines.push_back("T " + std::to_string(t));
                            const bool eos = !ids.empty() && (ids.back() == F.tok.im_end || ids.back() == F.tok.endoftext);
                            E.lines.push_back("DONE " + std::to_string(ids.size()) + " 10 1.0 2.0 " + (eos ? "stop" : "length") + " 0 0 0 0 0 0 0 0 10 0");
                            E.cv.notify_all();
                        }
                    }
                });
#endif
                Service S{ E, F };
                S.silence_s = 0;
                Chunker C{ "chatcmpl-test", F.model_for(q.value("req", json::object())), 0 };
                std::vector<json> chunks = { C.chunk(json{ { "role", "assistant" }, { "content", "" } }) };
                RunOut R;
                try {
                    R = S.run(P, q.value("req", json::object()), "", {}, [&](const fxf::Event & ev) { json c = C.event(ev); if (!c.is_null()) chunks.push_back(c); return true; },
                              [] { return true; }, [] { return false; });
                    chunks.push_back(C.final_chunk(R, (long long) P.ids.size()));
                    a = openai_collect(chunks);
                } catch (const std::invalid_argument & e) { a["error"] = std::string("ValueError: ") + e.what(); }
#if !defined(_WIN32)
                E.send("QUIT");
                eng.join();
                close(pin[0]); close(pin[1]);
#endif
                a["gens"] = gens;
            } else if (op == "caps") {
                a = detect_caps(F);
            } else a["error"] = "unknown op";
        } catch (const fxf::BadRequest & e) {
            a["error"] = std::string("BadRequest: ") + e.what();
        } catch (const std::exception & e) {
            a["error"] = std::string("exception: ") + e.what();
        }
        std::cout << a.dump() << "\n" << std::flush;
    }
    llama_model_free(model);
    return 0;
}

int run(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--fx-front-test")) return front_test(argc, argv);
    Args A;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { usage(); exit(2); } return std::string(argv[++i]); };
        if (a == "--fx-args") { A.fx_args_set = true; for (++i; i < argc; ++i) A.fx_args.push_back(argv[i]); break; }
        else if (a == "-m" || a == "--model") A.model = nx();
        else if (a == "--fx-config") A.fx_config = nx();
        else if (a == "--fx-config-json") A.fx_config_json = nx();
        else if (a == "--fx-lib") A.fx_lib = nx();
        else if (a == "--fx-image-python") A.image_python = nx();
        else if (a == "--fx-tokenize-endpoints") A.tokenize_endpoints = true;
        else if (a == "--fx-cli") A.fx_cli = true;
        else if (a == "--parallel" || a == "-np") A.parallel = std::stoi(nx());
        else if (a == "--long-context") {
            const std::string v = nx();
            A.long_context = v == "512k" ? 524288 : v == "768k" ? 786432 : v == "1m" ? 1048576 : std::atoll(v.c_str());
            if (A.long_context <= TRAINED_CTX) { fx::say("impulse-server (fx): --long-context is 512k, 768k, 1m or a context above %lld\n", TRAINED_CTX); return 2; }
        }
        else if (a == "--alias") A.alias = nx();
        else if (a == "--host") A.host = nx();
        else if (a == "--port") A.port = std::stoi(nx());
        else if (a == "--timeout") A.timeout = std::stoi(nx());
        else if (a == "--api-key") A.keys.add_csv(nx());
        else if (a == "--api-key-file") { const std::string e = A.keys.add_file(nx()); if (!e.empty()) { fx::say("error: %s\n", e.c_str()); return 1; } }
        else if (a == "--no-webui" || a == "--metrics" || a == "--jinja") {}   // accepted; nothing to do here
        else { fx::say("impulse-server (fx): unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    if (A.image_python.empty()) if (const char * e = getenv("IMPULSE_FX_IMAGE_PYTHON")) A.image_python = e;
    fx::Pack pack;
    if (!fx::find_pack(A.model, pack)) { fx::say("impulse-server (fx): %s is not a Flash-Next pack\n", A.model.c_str()); usage(); return 2; }
    if ((!A.fx_config.empty()) + (!A.fx_config_json.empty()) + A.fx_args_set > 1) { fx::say("impulse-server (fx): one of --fx-config, --fx-config-json, --fx-args\n"); return 2; }

    // 1. the engine's arguments and environment, before anything in this process touches CUDA
    std::vector<std::string> eargs;
    json cfg = json::object();
    fxf::Front F;
    double silence_s = 300.0;   // ENGINE_SILENCE_S
    int parallel = 1;
    if (!A.fx_config.empty() || !A.fx_config_json.empty()) {
        std::string text = A.fx_config_json, e;
        if (!A.fx_config.empty() && !read_file(A.fx_config, text)) e = "cannot read " + A.fx_config;
        if (e.empty()) e = apply_run_config(text, A.fx_config.empty() ? std::string("--fx-config-json") : A.fx_config, eargs, cfg);
        if (e.empty()) e = front_settings(cfg, F.cfg, silence_s, parallel);
        if (!e.empty()) { fx::say("impulse-server (fx): %s\n", e.c_str()); return 2; }
    } else eargs = A.fx_args;
    // the run config's "log" (server.py's StrataEngine): the engine's stderr appended there, from here on (src/fx/fx_log.hpp)
    if (!A.fx_cli && cfg.contains("log") && cfg["log"].is_string() && !cfg["log"].get<std::string>().empty()) {
        const std::string e = fx::open_engine_log(cfg["log"].get<std::string>());
        if (!e.empty()) { fx::say("impulse-server (fx): %s\n", e.c_str()); return 1; }
    }
    if (!A.alias.empty()) F.cfg.model = A.alias;
    if (A.parallel > 0) parallel = A.parallel;
    if (!A.fx_cli) parallel_args(parallel, eargs);
    // --long-context (opt-in; Nebula's per-engine "Long context"): that context, and dynamic YaRN past the trained one unless
    // the run config or the args name a RoPE scaling themselves
    if (A.long_context > 0) {
        bool set = false;
        for (size_t i = 0; i + 1 < eargs.size(); ++i) if (eargs[i] == "--max-context") { eargs[i + 1] = std::to_string(A.long_context); set = true; }
        if (!set) { eargs.push_back("--max-context"); eargs.push_back(std::to_string(A.long_context)); }
        if (!has_flag(eargs, "--rope-scaling") && !(cfg.is_object() && cfg.contains("rope") && !cfg["rope"].is_null())) {
            char f[32];
            std::snprintf(f, sizeof f, "%g", (double) A.long_context / (double) TRAINED_CTX);
            for (const char * w : { "--rope-scaling", "dynamic", "--rope-scale" }) eargs.push_back(w);
            eargs.push_back(f);
            for (const char * w : { "--yarn-orig-ctx", "262144", "--rope-dynamic-step", "0.25" }) eargs.push_back(w);
        }
        fx::say("[impulse fx] long context: %lld tokens, dynamic YaRN past the trained %lld (each request pays the smaller expert cache)\n",
                A.long_context, TRAINED_CTX);
    }
    if (!A.fx_args_set) {   // a run config's "rope" and the automatic pick past the trained context (--fx-args: as given)
        int gpus = 1;
        if (cfg.contains("gpu") && cfg["gpu"].is_array()) gpus = std::max<int>(1, (int) cfg["gpu"].size());
        std::string note;
        const std::string e = rope_args(cfg, eargs, gpus, note);
        if (!e.empty()) { fx::say("impulse-server (fx): %s\n", e.c_str()); return 2; }
        if (!note.empty()) fx::say("[impulse fx] %s\n", note.c_str());
    }
    if (!A.fx_args_set || A.long_context > 0) {   // --fx-args alone: as given
        int gpus = 1;
        if (cfg.contains("gpu") && cfg["gpu"].is_array()) gpus = std::max<int>(1, (int) cfg["gpu"].size());
        dynamic_rope_groups(eargs, gpus);
    }
    if (!has_flag(eargs, "--pack")) { eargs.insert(eargs.begin(), pack.pack); eargs.insert(eargs.begin(), "--pack"); }
    if (!has_flag(eargs, "--native")) { eargs.push_back("--native"); eargs.push_back(pack.dense); }
    if (!A.fx_cli && !has_flag(eargs, "--serve")) eargs.insert(eargs.begin(), "--serve");
    eargs.insert(eargs.begin(), "strata");
    std::string model_path;   // StrataEngine.model_path: --native's, else --pack's
    for (size_t i = 0; i + 1 < eargs.size(); ++i) if (eargs[i] == "--native") { model_path = eargs[i + 1]; break; }
    if (model_path.empty()) for (size_t i = 0; i + 1 < eargs.size(); ++i) if (eargs[i] == "--pack") { model_path = eargs[i + 1]; break; }

    // 1b. the image encoder (server.py starts it before the engine, in the engine's environment): the run config's "vision"
    std::unique_ptr<fxv::Vision> vision;
    if (!A.fx_cli && cfg.contains("vision") && cfg["vision"].is_object()) {
        const json & v = cfg["vision"];
        auto abs = [&](const std::string & p) {
            if (p.empty() || p[0] == '/') return p;
            const std::string cwd = cfg.value("cwd", std::string("."));
            return cwd + "/" + p;
        };
        fxv::Config vc;
        vc.exe = abs(v.value("exe", std::string()));
        vc.mmproj = abs(v.value("mmproj", std::string()));
        vc.model = abs(v.value("model", std::string()));
        vc.gpu = v.value("gpu", false);
        vc.threads = v.value("threads", 0);
        vc.max_tokens = v.value("max_tokens", 0);
        vc.min_tokens = v.value("min_tokens", 0);
        vc.python = A.image_python;
        if (v.value("elastic", false)) { fx::say("impulse-server (fx): \"vision\".\"elastic\" is not supported by impulse-fx\n"); return 2; }
        if (v.contains("cuda_device")) { fx::say("impulse-server (fx): \"vision\".\"cuda_device\" is not supported by impulse-fx\n"); return 2; }
        vision = std::make_unique<fxv::Vision>(vc);
        fx::say("loading the vision encoder ...\n");
        const std::string e = vision->start();
        if (!e.empty()) { fx::say("impulse-server (fx): %s\n", e.c_str()); return 1; }
        fx::say("impulse fx: the vision encoder is ready (%s)%s\n", vc.exe.c_str(),
                vc.python.empty() ? "; no --fx-image-python: only JPEG, PNG, BMP and GIF pictures" : "");
    }
    // The engine sets CUDA_MODULE_LOADING=EAGER (unless set) at its start, before its first CUDA call: every kernel is
    // loaded with the context, before the expert cache takes the free VRAM (generate.cpp).  The CUDA runtime reads it
    // once, when CUDA starts in the process - so it is set here, first.  (Gate P1, 2026-10-25.)
    if (getenv("CUDA_MODULE_LOADING") == nullptr) set_env("CUDA_MODULE_LOADING", "EAGER");

    // 2. the engine: libimpulse_fx (with --fx-cli, run here and done: no tokenizer, no server)
    std::string lib = A.fx_lib;
    if (lib.empty()) if (const char * e = getenv("IMPULSE_FX_LIB")) lib = e;
#if defined(_WIN32)
    if (lib.empty()) lib = lite::self_dir() + "impulse_fx.dll";
    HMODULE h = LoadLibraryA(lib.c_str());
    if (!h) { fx::say("impulse-server (fx): cannot load %s (error %lu)\n", lib.c_str(), GetLastError()); return 1; }
    auto abi = (fx_abi_fn) (void *) GetProcAddress(h, "impulse_fx_abi");
    auto fx_run = (fx_run_fn) (void *) GetProcAddress(h, "impulse_fx_run");
#else
    if (lib.empty()) lib = self_dir_posix() + "libimpulse_fx.so";
    void * h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { fx::say("impulse-server (fx): cannot load %s: %s (a build with -DIMPULSE_FLASHNEXT=ON has it)\n", lib.c_str(), dlerror()); return 1; }
    auto abi = (fx_abi_fn) dlsym(h, "impulse_fx_abi");
    auto fx_run = (fx_run_fn) dlsym(h, "impulse_fx_run");
#endif
    if (!abi || !fx_run || abi() != 1) { fx::say("impulse-server (fx): %s is not a libimpulse_fx this server knows (interface 1)\n", lib.c_str()); return 1; }
    if (A.fx_cli) {   // the engine's own command line, in this process, on this process's stdin / stdout
        std::vector<char *> av;
        for (auto & x : eargs) av.push_back(x.data());
        av.push_back(nullptr);
        fflush(stdout);
#if defined(_WIN32)
        return fx_run((int) eargs.size(), av.data(), (std::intptr_t) GetStdHandle(STD_INPUT_HANDLE), (std::intptr_t) GetStdHandle(STD_OUTPUT_HANDLE));
#else
        return fx_run((int) eargs.size(), av.data(), 0, dup(1));
#endif
    }

    // 3. the tokenizer and the chat template: llama.cpp's, from the dense GGUF (no weights), loaded once the engine is
    // ready (step 6): nothing of this process touches CUDA before the engine has started it, as `strata` alone would
    common_params P;
    P.use_jinja = true;
    P.n_parallel = 1;
    P.sampling.temp = 0.0f;   // what the engine does without sampling keys (greedy), for /completion's settings
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    server_chat_params chat;
    json caps;
    std::atomic<bool> loaded{ false };   // the vocabulary and the template are in (the handlers read them only then)

    // 4. its serve loop on a thread of this process
    Engine E;
    std::intptr_t eng_in = 0, eng_out = 0;
    pipe_t rd{};
#if defined(_WIN32)
    HANDLE in_r = nullptr, out_w = nullptr;
    if (!CreatePipe(&in_r, &E.in_w, nullptr, 0) || !CreatePipe(&rd, &out_w, nullptr, 0)) { fx::say("impulse-server (fx): no pipes\n"); return 1; }
    eng_in = (std::intptr_t) in_r; eng_out = (std::intptr_t) out_w;
#else
    signal(SIGPIPE, SIG_IGN);
    int pin[2], pout[2];
    if (pipe(pin) != 0 || pipe(pout) != 0) { perror("impulse-server (fx): pipe"); return 1; }
    for (int fd : { pin[0], pin[1], pout[0], pout[1] }) fcntl(fd, F_SETFD, FD_CLOEXEC);
    E.in_w = pin[1]; rd = pout[0]; eng_in = pin[0]; eng_out = pout[1];
#endif
    {
        std::string line;
        for (size_t i = 1; i < eargs.size(); ++i) line += (i > 1 ? " " : "") + eargs[i];
        fx::say("impulse fx: engine %s (%s), its arguments: %s\n", lib.c_str(), pack.root.c_str(), line.c_str());
        fx::log_engine_started(lib, line);
    }
    std::thread([&E, rd] { reader_loop(E, rd); }).detach();
    std::thread([eargs, fx_run, eng_in, eng_out]() mutable {
        std::vector<char *> av;
        for (auto & s : eargs) av.push_back(s.data());
        av.push_back(nullptr);
        const int rc = fx_run((int) eargs.size(), av.data(), eng_in, eng_out);
        // the engine ended (an error at load, or its serve loop ended): the server ends with it, so whatever started it
        // (Nebula, systemd) sees the exit and starts it again, as it would a crashed engine process
        fx::say("impulse fx: the engine ended (code %d); impulse-server exits\n", rc);
        fflush(stderr);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::_Exit(rc != 0 ? rc : 1);
    }).detach();

    Service S{ E, F };
    S.vision = vision.get();
    S.silence_s = silence_s;

    // 5. HTTP
    std::mutex legacy_turn;   // /completion's (llama-server's path): one request on the engine at a time, as the chat path
    httplib::Server srv;
    {
        const std::string e = A.keys.add_env();
        if (!e.empty()) { fx::say("error: %s\n", e.c_str()); return 1; }
        lite::install_api_keys<httplib::Server, httplib::Request, httplib::Response>(srv, A.keys);
    }
    const auto t_created = std::time(nullptr);
    (void) t_created;
    auto send_json = [](httplib::Response & r, const json & j, int code) { r.status = code; r.set_content(fxf::py_dumps(j, false), "application/json"); };
    auto is_ready = [&]() { std::lock_guard<std::mutex> lk(E.mu); return E.ready && !E.eof && loaded.load(); };
    auto loading = [&](httplib::Response & r) {   // before the vocabulary is in: llama-server's loading answer
        send_json(r, json{ { "error", llama_error("Loading model", 503, "unavailable_error") } }, 503);
    };
    auto max_ctx = [&]() { std::lock_guard<std::mutex> lk(E.mu); return E.max_ctx; };
    auto engine_version = [&]() -> std::string {
        std::lock_guard<std::mutex> lk(E.mu);
        auto it = E.info.find("engine");
        return it == E.info.end() ? std::string() : it->second;
    };
    srv.Get("/health", [&](const httplib::Request &, httplib::Response & r) {
        if (!is_ready()) { loading(r); return; }
        send_json(r, json{ { "status", "ok" }, { "max_context", max_ctx() }, { "model", F.cfg.model }, { "images", S.vision != nullptr },
                           { "api_key", A.keys.enabled() }, { "image_generation", false }, { "loaded", true }, { "service", "strata" } }, 200);
    });
    srv.Get("/api/health", [&](const httplib::Request & q, httplib::Response & r) {
        if (!is_ready()) { loading(r); return; }
        (void) q;
        send_json(r, json{ { "status", "ok" }, { "max_context", max_ctx() }, { "model", F.cfg.model }, { "images", S.vision != nullptr },
                           { "api_key", A.keys.enabled() }, { "image_generation", false }, { "loaded", true }, { "service", "strata" } }, 200);
    });
    auto models = [&](const httplib::Request &, httplib::Response & r) {
        if (!is_ready()) { loading(r); return; }
        json m = json{ { "id", F.cfg.model }, { "object", "model" }, { "status", json{ { "value", "loaded" } } }, { "meta", json{ { "n_ctx", max_ctx() } } },
                       { "architecture", json{ { "input_modalities", S.vision ? json::array({ "text", "image" }) : json::array({ "text" }) },
                                               { "output_modalities", json::array({ "text" }) } } } };
        if (!F.cfg.aliases.empty()) m["aliases"] = F.cfg.aliases;
        json data = json::array({ m });
        for (const auto & x : F.cfg.aliases) { json a = m; a["id"] = x; a["alias_of"] = F.cfg.model; data.push_back(a); }
        send_json(r, json{ { "object", "list" }, { "data", data } }, 200);
    };
    srv.Get("/v1/models", models);
    srv.Get("/models", models);
    srv.Get("/props", [&](const httplib::Request & q, httplib::Response & r) {
        if (!is_ready()) { loading(r); return; }
        const std::string asked = q.has_param("model") ? q.get_param_value("model") : F.cfg.model;
        if (asked != F.cfg.model && std::find(F.cfg.aliases.begin(), F.cfg.aliases.end(), asked) == F.cfg.aliases.end()) {
            send_json(r, json{ { "error", json{ { "message", "model not found" } } } }, 404);
            return;
        }
        json props = json{ { "default_generation_settings", json{ { "n_ctx", max_ctx() }, { "params", json{ { "n_predict", -1 } } } } },
                           { "total_slots", S.sched ? S.sched->slots() : 1 }, { "model_alias", F.cfg.model }, { "chat_template", F.template_source },
                           { "chat_template_caps", caps }, { "modalities", json{ { "vision", S.vision != nullptr } } },
                           { "models_autoload", true }, { "is_sleeping", false } };
        {   // impulse's own facts (the gates read engine_info)
            json inf = json::object();
            std::lock_guard<std::mutex> lk(E.mu);
            for (const auto & [k, v] : E.info) inf[k] = v;
            props["impulse"] = json{ { "engine", "fx" }, { "version", IMPULSE_VERSION_STR }, { "engine_info", inf } };
        }
        if (!model_path.empty()) props["model_path"] = model_path;
        const std::string v = engine_version();
        if (!v.empty()) props["build_info"] = "Strata " + v;
        send_json(r, props, 200);
    });
    srv.Get("/slots", [&](const httplib::Request &, httplib::Response & r) {
        if (!is_ready()) { send_json(r, json::array(), 200); return; }
        if (S.sched) {   // the batch slots: each one's state (server.py's slots_view)
            json a = json::array();
            for (const auto & v : S.sched->view())
                a.push_back(json{ { "id", v.id }, { "n_ctx", max_ctx() }, { "is_processing", v.busy }, { "state", v.state },
                                  { "n_prompt_tokens", v.prompt_tokens }, { "n_decoded", v.generated }, { "held_tokens", v.held_tokens } });
            send_json(r, a, 200);
            return;
        }
        send_json(r, json::array({ json{ { "id", 0 }, { "n_ctx", max_ctx() }, { "is_processing", S.busy.load() }, { "n_prompt_tokens", S.in_use.load() } } }), 200);
    });
    if (A.tokenize_endpoints) {
        srv.Post("/tokenize", [&](const httplib::Request & q, httplib::Response & r) {
            if (!is_ready()) { loading(r); return; }
            json body; try { body = json::parse(q.body); } catch (...) { send_json(r, json{ { "error", "invalid JSON" } }, 400); return; }
            const auto toks = common_tokenize(vocab, body.value("content", std::string()), body.value("add_special", false), body.value("parse_special", true));
            send_json(r, json{ { "tokens", toks } }, 200);
        });
        srv.Post("/detokenize", [&](const httplib::Request & q, httplib::Response & r) {
            if (!is_ready()) { loading(r); return; }
            json body; try { body = json::parse(q.body); } catch (...) { send_json(r, json{ { "error", "invalid JSON" } }, 400); return; }
            std::vector<llama_token> toks;
            for (const auto & t : body.value("tokens", json::array())) toks.push_back(t.get<llama_token>());
            send_json(r, json{ { "content", lite::utf8_clean(common_detokenize(vocab, toks, true)) } }, 200);
        });
    }

    // POST /v1/chat/completions: server.py's _openai
    auto chat_completions = [&](const httplib::Request & req, httplib::Response & r) {
        if (!is_ready()) { loading(r); return; }
        if (S.dying) { send_json(r, err_body("server_error", "the engine was ended; the server restarts"), 503); return; }
        json body;
        try { body = req.body.empty() ? json::object() : json::parse(req.body); }
        catch (const std::exception & e) { send_json(r, err_body("invalid_request_error", std::string("invalid JSON: ") + e.what()), 400); return; }
        if (!body.is_object()) { send_json(r, err_body("invalid_request_error", "send a JSON object"), 400); return; }
        auto prompt = std::make_shared<fxf::Prompt>();
        auto embeddings = std::make_shared<std::string>();
        auto img_keys = std::make_shared<std::vector<std::string>>();
        try {
            // logit_bias: the engine must say it takes them (bias_api.validate_request)
            const json lb = fxf::logit_bias_normalize(body.contains("logit_bias") ? body["logit_bias"] : json(nullptr), n_vocab);
            if (!lb.empty()) {
                std::lock_guard<std::mutex> lk(E.mu);
                if (E.info["logit_bias"] != "1") throw fxf::BadRequest("logit_bias: this engine does not support token biases; use a supporting CUDA/HIP build");
            }
            std::vector<std::string> files;
            fxf::Front::ImageEncoder enc = nullptr;
            if (S.vision) enc = [&](const std::vector<std::string> & srcs) {
                std::vector<std::pair<std::string, int>> out;
                // server.py encodes in the engine's turn (Service._turn): the encoder and a running request never share
                // the machine (a CPU encoder's threads against the engine's CPU pool); a client that leaves meanwhile
                // gives its place up (#1999)
                std::unique_lock<std::mutex> turn(S.fifo, std::defer_lock);
                while (!turn.try_lock()) {
                    if (req.is_connection_closed && req.is_connection_closed()) throw fxf::BadRequest("the client left");
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                for (const auto & s : srcs) {
                    auto pr = S.vision->encode(s, files);
                    files.push_back(pr.first);
                    const std::string stem = pr.first.substr(pr.first.rfind('/') + 1);
                    out.push_back({ stem.substr(0, 16), pr.second });
                }
                return out;
            };
            *prompt = F.prepare(body, max_ctx(), enc);
            if (!prompt->images.empty()) {
                // the request's pictures: the files of the last encode (prepare encodes the prompt's pictures last)
                std::vector<std::string> mine(files.end() - (long) prompt->images.size(), files.end());
                *embeddings = S.vision->combine(mine);
                if (embeddings->empty()) throw fxf::BadRequest("the images could not be written for the engine");
                for (const auto & f : mine) img_keys->push_back(f.substr(f.rfind('/') + 1).substr(0, 16));
            }
        } catch (const fxf::BadRequest & e) {
            fx::say("[impulse fx] 400 invalid request: %s\n", e.what());
            send_json(r, err_body("invalid_request_error", e.what()), 400);
            return;
        } catch (const std::exception & e) {
            send_json(r, err_body("invalid_request_error", std::string("the request is malformed (") + e.what() + ")"), 400);
            return;
        }
        const bool stream = body.contains("stream") && ([&] { const json & s = body["stream"]; return s.is_boolean() ? s.get<bool>() : s.is_number() ? s.get<double>() != 0 : s.is_string() ? !s.get<std::string>().empty() : !s.is_null() && !s.empty(); })();
        auto ch = std::make_shared<Chunker>(Chunker{ "chatcmpl-" + fxf::new_hex(24), F.model_for(body), (long long) std::time(nullptr) });
        auto reqp = std::make_shared<json>(body);
        auto cleanup = [embeddings]() { if (!embeddings->empty()) std::remove(embeddings->c_str()); };
        if (!stream) {
            std::vector<json> chunks = { ch->chunk(json{ { "role", "assistant" }, { "content", "" } }) };
            auto gone = [&req]() { return req.is_connection_closed ? req.is_connection_closed() : false; };
            try {
                RunOut R = S.run(*prompt, *reqp, *embeddings, *img_keys,
                                 [&](const fxf::Event & ev) { json c = ch->event(ev); if (!c.is_null()) chunks.push_back(c); return true; },
                                 [&]() { return !gone(); }, gone);
                cleanup();
                if (R.finish == "cancel" && gone()) return;
                chunks.push_back(ch->final_chunk(R, (long long) prompt->ids.size()));
                json res = openai_collect(chunks);
                if (prompt->structured) {
                    const std::string content = res["choices"][0]["message"]["content"].is_string() ? res["choices"][0]["message"]["content"].get<std::string>() : std::string();
                    res["choices"][0]["message"]["content"] = fxf::validated_json(content, res["choices"][0]["finish_reason"].get<std::string>());
                }
                send_json(r, res, 200);
            } catch (const fxf::StructuredError & e) {
                cleanup();
                send_json(r, json{ { "error", json{ { "type", "structured_output_failed" }, { "code", "structured_output_failed" }, { "message", e.what() } } } }, 502);
            } catch (const EngineNonFinite & e) {
                cleanup();
                send_json(r, err_body("server_error", e.what()), 503);
            } catch (const EngineErr & e) {
                cleanup();
                fx::say("[impulse fx] the engine reported an error: %s\n", e.what());
                send_json(r, err_body("invalid_request_error", e.what()), 400);
            } catch (const std::invalid_argument & e) {
                cleanup();
                send_json(r, err_body("invalid_request_error", e.what()), 400);
            } catch (const EngineDied & e) {
                cleanup();
                send_json(r, err_body("server_error", std::string(e.what()) + "; the next request restarts it"), 503);
            }
            if (S.dying) { fx::say("impulse fx: the engine was ended (silent); impulse-server exits\n"); fflush(stderr); std::thread([] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); std::_Exit(1); }).detach(); }
            return;
        }
        r.status = 200;
        r.set_header("Cache-Control", "no-cache");
        r.set_header("X-Accel-Buffering", "no");
        r.set_chunked_content_provider("text/event-stream", [&S, ch, prompt, reqp, embeddings, img_keys, cleanup](size_t, httplib::DataSink & sink) {
            bool gone = false;
            auto write = [&](const std::string & s) { if (gone) return false; if (!(sink.is_writable() && sink.write(s.data(), s.size()))) gone = true; return !gone; };
            auto data = [&](const json & j) { return write("data: " + fxf::py_dumps(j, false) + "\n\n"); };
            const bool structured = prompt->structured;
            std::vector<json> buffered;   // structured_chunks: nothing unvalidated goes out
            auto out = [&](const json & c) { if (structured) { buffered.push_back(c); return !gone; } return data(c); };
            out(ch->chunk(json{ { "role", "assistant" }, { "content", "" } }));
            auto beat = std::chrono::steady_clock::now();
            try {
                RunOut R = S.run(*prompt, *reqp, *embeddings, *img_keys,
                                 [&](const fxf::Event & ev) {
                                     json c = ch->event(ev);
                                     if (c.is_null()) return !gone;
                                     if (structured && std::chrono::duration<double>(std::chrono::steady_clock::now() - beat).count() >= 1) { beat = std::chrono::steady_clock::now(); write(": keep-alive\n\n"); }
                                     return out(c);
                                 },
                                 [&]() { return write(": keep-alive\n\n"); }, [&]() { return gone; });
                cleanup();
                if (!gone) {
                    const json last = ch->final_chunk(R, (long long) prompt->ids.size());
                    if (structured) {
                        buffered.push_back(last);
                        json res = openai_collect(buffered);
                        const json & msg = res["choices"][0]["message"];
                        const std::string content = fxf::validated_json(msg["content"].is_string() ? msg["content"].get<std::string>() : std::string(),
                                                                        res["choices"][0]["finish_reason"].get<std::string>());
                        data(buffered.front());
                        json delta = json{ { "content", content } };
                        if (msg.contains("reasoning_content")) delta["reasoning_content"] = msg["reasoning_content"];
                        json mid = buffered.front();
                        mid["choices"] = json::array({ json{ { "index", 0 }, { "delta", delta }, { "finish_reason", nullptr } } });
                        data(mid);
                        data(buffered.back());
                    } else data(last);
                    write("data: [DONE]\n\n");
                }
            } catch (const fxf::StructuredError & e) {
                cleanup();
                write("data: " + fxf::py_dumps(json{ { "error", json{ { "type", "structured_output_failed" }, { "code", "structured_output_failed" }, { "message", e.what() } } } }, false) + "\n\ndata: [DONE]\n\n");
            } catch (const EngineDied & e) {
                cleanup();
                write("data: " + fxf::py_dumps(err_body("server_error", std::string(e.what()) + "; the next request restarts it"), false) + "\n\ndata: [DONE]\n\n");
            } catch (const std::exception & e) {   // the engine's ERR, non-finite twice, a call the parser cannot read
                cleanup();
                write("data: " + fxf::py_dumps(err_body("server_error", e.what()), false) + "\n\ndata: [DONE]\n\n");
            }
            sink.done();
            if (S.dying) { fx::say("impulse fx: the engine was ended (silent); impulse-server exits\n"); fflush(stderr); std::thread([] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); std::_Exit(1); }).detach(); }
            return false;
        });
    };
    srv.Post("/v1/chat/completions", chat_completions);
    srv.Post("/chat/completions", chat_completions);

    // POST /completion, /completions: llama-server's (raw ids, the gates' driver).  The P1 path, unchanged.
    struct Req {
        task_params tp;
        llama_tokens prompt;
        long long max_new = 0;
        std::string gen;
    };
    auto legacy_prepare = [&](const json & body, Req & R, int & code) -> json {
        json data = body;
        if (!data.contains("prompt")) { code = 400; return llama_error("\"prompt\" is required", 400, "invalid_request_error"); }
        try {
            auto v = tokenize_input_prompts(vocab, nullptr, to_common(data.at("prompt")), true, true, mtmd_helper_init_opt_default());
            if (v.size() != 1) { code = 400; return llama_error("one prompt per request", 400, "invalid_request_error"); }
            R.prompt = v[0].get_text_tokens();
        } catch (const std::exception & e) { code = 400; return llama_error(e.what(), 400, "invalid_request_error"); }
        try { R.tp = server_schema::eval_llama_cmpl_schema(vocab, P, P.sampling.logit_bias_eog, to_common(data)); }
        catch (const std::exception & e) { code = 400; return llama_error(e.what(), 400, "invalid_request_error"); }
        R.tp.res_type = TASK_RESPONSE_TYPE_NONE;
        R.tp.oaicompat_model = F.cfg.model;
        R.tp.oaicompat_cmpl_id = gen_chatcmplid();
        const long long ctx = max_ctx(), n = (long long) R.prompt.size();
        if (R.prompt.empty()) { code = 400; return llama_error("the prompt is empty", 400, "invalid_request_error"); }
        if (n + 1 > ctx) {
            code = 400;
            json e = llama_error("the request exceeds the available context size, try increasing it", 400, "exceed_context_size_error");
            e["n_prompt_tokens"] = n; e["n_ctx"] = ctx;
            return e;
        }
        R.max_new = 0;
        for (const char * name : { "max_completion_tokens", "max_tokens", "n_predict" })
            if (body.contains(name) && body[name].is_number_integer() && body[name].get<long long>() > 0) { R.max_new = body[name].get<long long>(); break; }
        if (R.max_new <= 0 || R.max_new > ctx - n) R.max_new = ctx - n;
        std::string ids;
        for (size_t i = 0; i < R.prompt.size(); ++i) { if (i) ids += ','; ids += std::to_string(R.prompt[i]); }
        R.gen = "GEN " + std::to_string(R.max_new) + fxf::sampling_keys(body) + " " + ids;
        code = 200;
        return json();
    };
    auto legacy = [&](const httplib::Request & q, httplib::Response & r) {
        if (!is_ready()) { loading(r); return; }
        json body;
        try { body = json::parse(q.body); }
        catch (...) { send_json(r, json{ { "error", llama_error("invalid JSON body", 400, "invalid_request_error") } }, 400); return; }
        Req R;
        int code = 500;
        json e = legacy_prepare(body, R, code);
        if (code != 200) { send_json(r, json{ { "error", e } }, code); return; }
        std::unique_lock<std::mutex> g1(S.fifo, std::defer_lock), g2(legacy_turn, std::defer_lock);
        if (!S.sched) { g1.lock(); g2.lock(); }   // batch slots: the scheduler queues it with the chat requests
        llama_tokens out_toks;
        std::string text;
        Pass pass{ E, S.silence_s, json::object() };
        stop_type stop = STOP_TYPE_NONE;
        std::string stopping_word;
        try {
            const size_t sp = R.gen.rfind(' ');
            std::vector<int32_t> ids(R.prompt.begin(), R.prompt.end());
            auto on_tok = [&](int32_t t) {
                out_toks.push_back(t);
                if (llama_vocab_is_eog(vocab, t)) { stop = STOP_TYPE_EOS; return true; }   // the engine ends there itself (P1)
                text += common_token_to_piece(vocab, t, false);
                for (const auto & w : R.tp.antiprompt) if (!w.empty() && text.find(w) != std::string::npos) { text.erase(text.find(w)); stop = STOP_TYPE_WORD; stopping_word = w; return false; }
                return true;
            };
            auto closed = [&]() { return q.is_connection_closed ? q.is_connection_closed() : false; };
            if (S.sched) {
                const fxs::Result res = S.sched->generate(fxs::Request{ R.max_new, fxf::sampling_keys(body), "", ids }, fxs::Hooks{ on_tok, [] {}, closed });
                if (!res.legs.empty()) { pass.last = legs_last(res.legs); pass.got_done = true; }
            } else pass.run(R.gen.substr(0, sp), ids, on_tok, [] {}, closed);
        } catch (const std::exception & ex) {
            send_json(r, json{ { "error", llama_error(std::string("generation: ") + ex.what(), 500, "server_error") } }, 500);
            return;
        }
        if (stop == STOP_TYPE_NONE) stop = pass.last.value("finish", std::string()) == "length" ? STOP_TYPE_LIMIT : STOP_TYPE_EOS;
        json res = json{ { "content", lite::utf8_clean(text) }, { "tokens", out_toks }, { "stop", true }, { "model", F.cfg.model },
                         { "tokens_predicted", (long long) out_toks.size() }, { "tokens_evaluated", (long long) R.prompt.size() },
                         { "stop_type", stop == STOP_TYPE_EOS ? "eos" : stop == STOP_TYPE_WORD ? "word" : "limit" }, { "stopping_word", stopping_word },
                         { "timings", request_timings((long long) R.prompt.size(), (long long) out_toks.size(), pass.last) } };
        send_json(r, res, 200);
    };
    srv.Post("/completion", legacy);
    srv.Post("/completions", legacy);
    srv.set_error_handler([&](const httplib::Request &, httplib::Response & r) {
        if (r.status == 404 && r.body.empty()) send_json(r, json{ { "error", json{ { "message", "not found" } } } }, 404);
    });
    srv.set_read_timeout(A.timeout); srv.set_write_timeout(A.timeout);
    srv.set_socket_options([](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    fx::say("impulse-server: listening on http://%s:%d (model %s, fx engine: %s; ready once the engine has loaded)\n",
            A.host.c_str(), A.port, F.cfg.model.c_str(), pack.root.c_str());
    std::thread listener([&]() {
        if (!srv.listen(A.host, A.port)) {
            fx::say("impulse-server: cannot listen on %s:%d\n", A.host.c_str(), A.port);
            fflush(stderr);
            std::_Exit(1);
        }
    });

    // 6. once the engine is ready (it has started CUDA, loaded every kernel and sized its caches): the vocabulary and the
    // template, with no device (vocab-only, an empty device list: no GPU is asked for its memory)
    {
        std::unique_lock<std::mutex> lk(E.mu);
        E.cv.wait(lk, [&] { return E.ready || E.eof; });
        if (!E.ready) { lk.unlock(); listener.join(); return 1; }   // the engine thread ends the process
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices = no_devices;
    model = llama_model_load_from_file(pack.dense.c_str(), mp);
    if (!model) { fx::say("impulse-server (fx): cannot load the vocabulary of %s\n", pack.dense.c_str()); fflush(stderr); std::_Exit(1); }
    vocab = llama_model_get_vocab(model);
    n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n_vocab; ++t) if (llama_vocab_is_eog(vocab, t)) P.sampling.logit_bias_eog.push_back({ t, -INFINITY });
    {
        std::vector<std::string> merges;
        std::string tmpl;
        if (!gguf_tokenizer_meta(pack.dense, merges, tmpl)) { fx::say("impulse-server (fx): %s has no merges or chat template\n", pack.dense.c_str()); fflush(stderr); std::_Exit(1); }
        try { F.init(vocab, merges, tmpl); }
        catch (const std::exception & e) { fx::say("impulse-server (fx): the chat template: %s\n", e.what()); fflush(stderr); std::_Exit(1); }
        caps = detect_caps(F);
    }
    (void) chat;
    // --batch: the engine's slots (INFO batch_slots, before READY) -> the scheduler; and what a slot costs the expert cache
    std::unique_ptr<fxs::Scheduler> sched;
    {
        std::map<std::string, std::string> inf;
        { std::lock_guard<std::mutex> lk(E.mu); inf = E.info; }
        auto num = [&](const char * k) { auto it = inf.find(k); return it == inf.end() ? 0.0 : std::atof(it->second.c_str()); };
        const int slots = (int) num("batch_slots");
        if (parallel > 1 && slots < 2)
            fx::say("[impulse fx] parallel requests: %d asked, the engine runs one at a time (its log says why)\n", parallel);
        if (slots >= 2) {
            fxs::Config c;
            c.slots = slots;
            c.slot_cache = num("slot_cache") > 0;
            c.batch_ctx = (long long) num("batch_ctx");
            c.groups = std::max(1, (int) num("batch_groups"));
            c.silence_s = silence_s;
            c.eos = { F.tok.im_end, F.tok.endoftext };
            sched = std::make_unique<fxs::Scheduler>(E, c, [](const std::string & m) { fx::say("%s\n", m.c_str()); });
            S.sched = sched.get();
            if (parallel > 1 && slots != parallel) fx::say("[impulse fx] parallel requests: %d asked, the engine runs %d\n", parallel, slots);
            fx::say("%s", fxs::sizing_note(slots, num("batch_slot_mib"), num("expert_slots"), num("expert_cache_mib"), (long long) num("batch_ctx")).c_str());
        }
    }
    loaded = true;
    fx::say("ready: http://%s:%d/v1  (OpenAI: /v1/chat/completions, context %lld tokens%s%s)\n", A.host.c_str(), A.port, max_ctx(),
            vision ? ", images on" : "", A.keys.enabled() ? ", API key required" : "");
    listener.join();
    llama_model_free(model);
    return 0;
}

}  // namespace fxsrv
