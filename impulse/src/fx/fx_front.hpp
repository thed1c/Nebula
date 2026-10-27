// src/fx/fx_front.hpp - impulse-fx's request / answer layer: serve/server.py's and serve/frontend.py's behaviour in C++
// (One Engine P2, server parity).  No engine and no HTTP here: a request's JSON -> the template's messages -> the exact
// prompt text and token ids server.py sends; the model's text -> reasoning, content and tool calls as server.py's
// OutputParser splits them; stop strings; the JSON answer's pieces.  Every rule is server.py's (the comments name the
// function it follows, the issue numbers are Strata's); tests/fx_front_parity.py checks the ids and the parser events
// against server.py's own code.
//
// Python's semantics where they show in the output: json.dumps (separators, ensure_ascii, float repr), str.strip's
// Unicode whitespace, the incremental UTF-8 decoder's U+FFFD replacement.  The chat template is rendered with llama.cpp's
// Jinja engine (common/jinja) whose `tojson` float output and `trim` are made Python's by llama-hooks/fx-jinja-python.cmake
// (impulse-fx's private copy; impulse-server's llama.cpp is not touched).
#pragma once

#include <nlohmann/json.hpp>   // llama.cpp's vendored copy: ordered, as Python's dicts

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct llama_vocab;

namespace fxf {

using json = nlohmann::ordered_json;

// a ValueError of server.py: the request is answered 400 {"error": {"type": "invalid_request_error", "message": ...}}
struct BadRequest : std::runtime_error { using std::runtime_error::runtime_error; };

// ------------------------------------------------------------------------------------------------ Python's text rules
std::string py_float_repr(double v);                                   // repr(float)
// json.dumps(v, ensure_ascii=ascii, separators=(item, key)) for a JSON value
std::string py_dumps(const json & v, bool ensure_ascii = true, const char * item_sep = ", ", const char * key_sep = ": ");
std::string py_dumps_str(const std::string & s, bool ensure_ascii);    // json.dumps(str)
bool py_isspace_cp(uint32_t cp);
std::string py_strip(const std::string & s, bool left = true, bool right = true);   // str.strip() / lstrip / rstrip
std::string py_strip_chars(const std::string & s, const std::string & chars, bool left = true, bool right = true);
// utf-8: the code point that ends at (before) byte `end`, its byte length in *len (0 when none)
uint32_t utf8_last_cp(const std::string & s, size_t end, size_t * len);
bool utf8_valid(const std::string & s);

// Python's incremental UTF-8 decoder with errors="replace" (codecs.getincrementaldecoder): complete characters out,
// a split character held, an invalid sequence as U+FFFD per maximal subpart
struct Utf8Decoder {
    std::string held;
    std::string push(const std::string & bytes, bool final = false);
    bool pending() const { return !held.empty(); }
};

// ------------------------------------------------------------------------------------------------ tokenizer
// strata_tokenizer.Tokenizer's encode on llama.cpp's vocabulary: CONTROL (type 3) literals matched only with
// parse_special, USER_DEFINED (type 4) always; the leftmost, longest literal first; a match that starts inside a `plain`
// span (#537) is left to the text around it; the stretches between matches BPE-encoded each on its own.
struct Tokenizer {
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    std::vector<std::string> bytes;                        // each token's raw bytes (its text for literals)
    std::vector<std::pair<std::string, int>> specials;     // (text, id), types 3 and 4, longest first
    std::vector<std::pair<std::string, int>> always;       // type 4 only, longest first
    std::vector<std::string> control_tokens;               // type 3 texts in id order (literal_tags' order)
    size_t max_special_len = 1;
    std::map<std::string, int> text_ids;                   // the vocabulary's own (byte-mapped) texts
    std::map<std::pair<std::string, std::string>, int> ranks;
    int im_start = -1, im_end = -1, endoftext = -1;

    // vocab: llama.cpp's (a vocab-only load); merges: the GGUF's tokenizer.ggml.merges
    void init(const llama_vocab * v, const std::vector<std::string> & merges);
    std::vector<int32_t> encode(const std::string & text, bool parse_special,
                                const std::vector<std::pair<size_t, size_t>> & plain = {}) const;
    std::vector<int32_t> bpe_plain(const std::string & text) const;   // _encode_plain (no literal matched at all)
    std::string decode(const std::vector<int32_t> & ids) const;       // bytes (errors left to the caller)
    std::string token_text(int32_t id) const { return id >= 0 && id < n_vocab ? bytes[(size_t) id] : std::string(); }
};

// ------------------------------------------------------------------------------------------------ requests
struct Settings {                       // the run config's switches server.py reads (the reference engine: all at their defaults)
    bool stop_token_text = true;        // EOS literal
    int repeat_stop_tokens = 256;       // #606
    bool tool_call_recovery = false;    // OutputParser's recover (not ported: refused at start when asked)
    bool tool_name_repair = false;      // OutputParser's repair_names
    bool fit_max_tokens = false;
    long long reasoning_budget_tokens = 0;   // #123: the config's default thinking budget
    std::string model = "qwen3.8-flash-next";
    std::vector<std::string> aliases;
};

struct Prompt {                         // a request ready for the engine (Service.prepare's result and what led to it)
    json messages = json::array();      // the template's messages (after openai_to_messages / prepare_format)
    json tools;                         // null or the template's tool list
    json kwargs = json::object();       // the template's kwargs (enable_thinking / reasoning_effort)
    std::string text;                   // the rendered prompt (literals unmarked)
    std::vector<std::pair<size_t, size_t>> plain;
    std::vector<int32_t> ids;           // the ids sent (images' pads expanded)
    std::vector<std::string> images;    // image sources in prompt order
    bool thinking = true;
    std::string force;                  // forced_call's opening ("" = the model decides)
    bool structured = false;            // response_format json_object / json_schema: the answer is checked
    long long max_new = 0;
};

class Front {
public:
    Tokenizer tok;
    Settings cfg;
    std::string template_source;

    // the chat template (the GGUF's tokenizer.chat_template) and the tokenizer
    void init(const llama_vocab * vocab, const std::vector<std::string> & merges, const std::string & tmpl);

    // POST /v1/chat/completions' request -> everything up to the engine: _openai's checks, openai_to_messages,
    // tool_choice, prepare_format, Service.prepare (without the images' encode: `encode_images` gives each image's
    // (key, n tokens) in order, or throws BadRequest).  ctx: the engine's context.
    using ImageEncoder = std::function<std::vector<std::pair<std::string, int>>(const std::vector<std::string> &)>;
    Prompt prepare(json & req, long long ctx, const ImageEncoder & encode_images) const;
    std::string render(const json & messages, const json & tools, const json & kwargs, bool add_generation_prompt) const;
    std::vector<int32_t> encode_prompt(const json & messages, const json & tools, const json & kwargs,
                                       std::string * text_out = nullptr,
                                       std::vector<std::pair<size_t, size_t>> * plain_out = nullptr) const;
    std::string model_for(const json & req) const;

private:
    std::shared_ptr<void> program_;     // the parsed template (jinja::program)
    std::vector<std::pair<std::string, std::string>> literals_;   // literal_tags(): (text, mark), longest first
};

// openai_to_messages (+ effort_kwargs, _late_system_to_user; env_to_user is off) -> (messages, tools, kwargs)
void openai_to_messages(const json & req, json & messages, json & tools, json & kwargs);
std::pair<std::string, std::string> tool_choice_of(const json & tool_choice);
std::string forced_call(const json & tool_choice, const json & tools);

// StrataEngine.sampling_keys (with strata_tune, strata_checkpoint, img keys, pin, cvec, logit_bias)
std::string sampling_keys(const json & sampling);
void check_request_sampling(json & req);               // types, as server.py (the request is changed as there)
std::vector<std::string> stop_strings(const json & req);
std::optional<long long> reasoning_budget(const json & req, long long config_default);   // nullopt: none
json logit_bias_normalize(const json & value, int vocab_size);   // {} or {id: bias}; BadRequest

// ------------------------------------------------------------------------------------------------ answers
struct ToolCall {
    std::string name;
    json arguments = json::object();
    std::string id;
};
struct Event {
    std::string kind;                    // reasoning | content | tool_call | tool_start | tool_args
    std::string text;
    std::shared_ptr<ToolCall> call;
};

// frontend.OutputParser (stream_tools; repair_names; recover is not ported)
class OutputParser {
public:
    OutputParser(bool thinking, const json & tools, bool stream_tools, bool repair_names);
    std::vector<Event> feed(const std::string & delta);
    std::vector<Event> finish(const std::string & reason = "stop");
    std::string literal_context(bool example = false);   // "" = none
    bool example_place();
    std::string state;
    std::string buf;
    int rescued = 0, refused = 0;
private:
    bool lead = false;
    std::map<std::string, json> schemas;
    std::vector<std::string> schema_order;
    bool stream_tools, repair_names;
    struct Pending { std::string raw; std::shared_ptr<ToolCall> call; };
    std::vector<Pending> pending;
    std::string fence, line;
    long long ticks = 0;
    bool bare = false, batch = false;
    size_t sp = 0;
    std::string ss = "name";
    std::shared_ptr<ToolCall> scall;
    bool sfirst = true, sval_started = false;
    std::map<std::string, json> sspecs;
    std::string sparam;

    std::string track(const std::string & text);
    bool opener_ok() const;
    bool in_code() const;
    std::pair<bool, std::string> place();
    std::optional<bool> quoted_think_end(size_t i);
    std::vector<Event> release(bool deliver);
    void reset_scan();
    std::vector<Event> scan();
    std::vector<Event> close_scan();
    std::vector<Event> finish_call(const std::string & body, bool closed = true);
    std::optional<std::pair<std::string, std::string>> repair_call(const std::string & name, const std::string & body);
    const json * schema_of(const std::string & name) const;
};

std::shared_ptr<ToolCall> parse_tool_call(const std::string & body, const json * schema);   // BadRequest-free: throws std::invalid_argument
std::string new_call_id();
std::string new_hex(size_t n);

// server.py's StopMatcher
struct StopMatcher {
    std::vector<std::string> stops;
    std::string held;
    std::optional<std::string> hit;
    std::string push(const std::string & text);
    std::string flush() { std::string h; h.swap(held); return h; }
};

// structured.py: prepare_format (the directive) and validated_json (json_object, or json_schema checked as one JSON object:
// The reference engine's image has no `jsonschema`, so server.py checks only that)
bool prepare_format(const json & response_format, json & messages);   // true: a structured answer is asked
std::string validated_json(const std::string & text, const std::string & finish);   // canonical JSON or throws StructuredError
struct StructuredError : std::runtime_error { using std::runtime_error::runtime_error; };

}  // namespace fxf
