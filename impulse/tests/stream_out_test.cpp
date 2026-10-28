// stream-out-test: what a streamed answer sends token by token (src/lite/stream_out.hpp). No GPU, no model.
#include "../src/lite/stream_out.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
static void expect(const char * what, const std::string & got, const std::string & want) {
    if (got != want) { ++fails; fprintf(stderr, "FAIL %s: got [%s] want [%s]\n", what, got.c_str(), want.c_str()); }
    else fprintf(stderr, "ok   %s\n", what);
}

int main() {
    const std::vector<std::string> stops = { "</stop>", "\nUser:" };
    std::string text; size_t sent = 0;
    auto add = [&](const std::string & piece) { text += piece; return lite::stream_take(text, sent, stops, false); };
    expect("plain text goes out", add("Hello"), "Hello");
    expect("a tail that may be a stop word is held", add(" world</s"), " world");
    expect("it was not one: it goes out", add("ky"), "</sky");
    expect("a held newline", add("\nUs"), "");
    expect("final flushes the rest", lite::stream_take(text, sent, stops, true), "\nUs");
    std::string u; size_t us = 0;
    u = "caf\xC3"; expect("half a UTF-8 character waits", lite::stream_take(u, us, {}, false), "caf");
    u += "\xA9!"; expect("the whole character", lite::stream_take(u, us, {}, false), "\xC3\xA9!");
    // llama-server's partial results through one state: the first carries the role, the deltas add up to the text
    task_params tp; tp.res_type = TASK_RESPONSE_TYPE_OAI_CHAT; tp.oaicompat_model = "m"; tp.oaicompat_cmpl_id = "id";
    task_result_state st(tp.chat_parser_params);
    std::string got; bool role = false;
    const std::vector<std::string> pieces = { "Fl", "int ", "streams." };
    for (size_t i = 0; i < pieces.size(); ++i) {
        const json ev = lite::stream_partial(st, tp, pieces[i], 1, (int) i + 1, 3, 0);
        for (const auto & c : ev) {
            const json & d = c.at("choices").at(0).at("delta");
            if (d.contains("role")) role = true;
            if (d.contains("content") && d.at("content").is_string()) got += d.at("content").get<std::string>();
        }
    }
    expect("the first chunk has the role", role ? "yes" : "no", "yes");
    expect("the content deltas add up", got, "Flint streams.");
    lite::StreamChan ch; ch.push(json::array({ json{ { "a", 1 } }, json{ { "b", 2 } } })); ch.close();
    expect("an array of events is queued one by one", std::to_string(ch.events.size()) + (ch.closed ? " closed" : ""), "2 closed");
    fprintf(stderr, fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
