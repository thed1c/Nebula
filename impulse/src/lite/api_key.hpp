// llama-server's --api-key / --api-key-file for impulse-server (tools/server/server-http.cpp middleware_validate_api_key):
// with one or more keys set, every request needs "Authorization: Bearer <key>" (or Anthropic's "X-Api-Key: <key>"), else
// 401 {"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}. /health, /v1/health and "/" stay
// open, and a browser's OPTIONS preflight passes (it carries no Authorization). No key = no check, as before. Unlike
// llama-server's std::find, a key is compared in constant time (every key, every byte).
#pragma once
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace lite {
struct ApiKeys {
    std::vector<std::string> keys;

    // --api-key KEY[,KEY...] (llama-server: comma-separated, empty items skipped)
    void add_csv(const std::string & v) {
        size_t a = 0;
        while (a <= v.size()) {
            size_t b = v.find(',', a);
            if (b == std::string::npos) b = v.size();
            if (b > a) keys.push_back(v.substr(a, b - a));
            a = b + 1;
        }
    }
    // --api-key-file FNAME: one key per line, lines starting with '#' are comments ("" = ok, else what is wrong)
    std::string add_file(const std::string & path) {
        std::ifstream f(path);
        if (!f) return "failed to open file '" + path + "'";
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();   // a file written on Windows
            if (!line.empty() && line[0] != '#') keys.push_back(line);
        }
        return {};
    }
    // LLAMA_API_KEY and LLAMA_ARG_API_KEY_FILE, as llama-server reads them (the command line adds to them)
    std::string add_env() {
        if (const char * k = getenv("LLAMA_API_KEY")) add_csv(k);
        if (const char * f = getenv("LLAMA_ARG_API_KEY_FILE")) return add_file(f);
        return {};
    }
    bool enabled() const { return !keys.empty(); }

    static bool public_path(const std::string & p) { return p == "/health" || p == "/v1/health" || p == "/"; }

    static bool same(const std::string & a, const std::string & b) {   // constant time in the content
        size_t diff = a.size() ^ b.size();
        const size_t n = a.size() > b.size() ? a.size() : b.size();
        for (size_t i = 0; i < n; ++i) {
            const unsigned char x = i < a.size() ? (unsigned char) a[i] : 0, y = i < b.size() ? (unsigned char) b[i] : 0;
            diff |= (size_t) (x ^ y);
        }
        return diff == 0;
    }
    // authorization: the Authorization header ("Bearer " is optional, as in llama-server); x_api_key: X-Api-Key
    bool accepts(std::string authorization, const std::string & x_api_key) const {
        if (keys.empty()) return true;
        if (authorization.empty()) authorization = x_api_key;
        static const std::string prefix = "Bearer ";
        if (authorization.compare(0, prefix.size(), prefix) == 0) authorization = authorization.substr(prefix.size());
        bool ok = false;
        for (const auto & k : keys) ok |= same(authorization, k);   // no early exit
        return ok;
    }
    static const char * error_body() {
        return "{\"error\":{\"message\":\"Invalid API Key\",\"type\":\"authentication_error\",\"code\":401}}";
    }
};

// the check in front of every route of a cpp-httplib server (Server, Request, Response from the caller's httplib)
template <class Server, class Request, class Response>
void install_api_keys(Server & srv, const ApiKeys & k) {
    if (!k.enabled()) return;
    srv.set_pre_routing_handler([&k](const Request & req, Response & res) {
        if (req.method == "OPTIONS" || ApiKeys::public_path(req.path)) return Server::HandlerResponse::Unhandled;
        if (k.accepts(req.get_header_value("Authorization"), req.get_header_value("X-Api-Key"))) return Server::HandlerResponse::Unhandled;
        res.status = 401;
        res.set_content(ApiKeys::error_body(), "application/json; charset=utf-8");
        return Server::HandlerResponse::Handled;
    });
}
}
