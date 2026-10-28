// Impulse: a streamed answer token by token (stream: true), as llama-server sends it.
//
// Until 0.1.4 both engines' servers waited for the whole answer and then sent every SSE event at once. Now each request
// that asks for a stream has a channel: the scheduler thread pushes llama-server's own partial results as tokens are made
// (server_task_result_cmpl_partial: the text added, run through one task_result_state per request, so reasoning_content,
// content and tool-call deltas come out of llama.cpp's chat parser exactly as llama-server streams them), then the final
// result (finish_reason, usage, timings) through the same state; the HTTP thread writes them as they arrive (chunked
// SSE). Text is held back while it could still be the start of a stop word or ends inside a UTF-8 character
// (llama-server's process_token). A client that goes away cancels its request: the slot stops at its next token. The
// non-stream path is not touched.
#pragma once

#include "server-common.h"
#include "server-task.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace lite {

struct StreamChan {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<json> events;              // OpenAI chunks (or /completion objects), in order
    bool closed = false;                  // the final result (or an error) is in
    std::atomic<bool> cancelled{ false }; // the client went away

    void push(const json & j) {
        { std::lock_guard<std::mutex> g(mu); if (j.is_array()) for (const auto & e : j) events.push_back(e); else events.push_back(j); }
        cv.notify_all();
    }
    void close() { { std::lock_guard<std::mutex> g(mu); closed = true; } cv.notify_all(); }
};

// the part of `text` after `n_sent` that can go out now: whole UTF-8 characters, and (unless final) not a tail that may
// still grow into one of the stop words; n_sent moves past it
inline std::string stream_take(const std::string & text, size_t & n_sent, const std::vector<std::string> & stops, bool final) {
    if (n_sent >= text.size()) return {};
    size_t end = text.size();
    if (!final) {
        end = n_sent + validate_utf8(text.substr(n_sent));
        size_t hold = 0;   // the longest tail of [n_sent, end) that is a proper prefix of a stop word
        for (const auto & w : stops) {
            if (w.empty()) continue;
            for (size_t k = std::min(w.size() - 1, end - n_sent); k > hold; --k)
                if (text.compare(end - k, k, w, 0, k) == 0) { hold = k; break; }
        }
        end -= hold;
    }
    if (end <= n_sent) return {};
    std::string out = text.substr(n_sent, end - n_sent);
    n_sent = end;
    return out;
}

// one partial result as llama-server's send_partial_response makes it, through the request's state; the events to send
inline json stream_partial(task_result_state & st, const task_params & tp, const std::string & piece, llama_token tok, int n_decoded,
                           int n_prompt, int n_cached) {
    server_task_result_cmpl_partial res;
    res.content = piece; res.tokens = { tok };
    res.n_decoded = n_decoded; res.n_prompt_tokens = n_prompt; res.n_prompt_tokens_cache = n_cached;
    res.post_sampling_probs = false; res.verbose = false; res.res_type = tp.res_type;
    res.oaicompat_model = tp.oaicompat_model; res.oaicompat_cmpl_id = tp.oaicompat_cmpl_id;
    res.update(st);
    return res.to_json();
}

} // namespace lite
