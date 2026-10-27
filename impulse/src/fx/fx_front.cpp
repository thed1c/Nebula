// src/fx/fx_front.cpp - see fx_front.hpp.  Part 1: Python's text rules, the tokenizer, requests and the prompt.
#include "fx_front.hpp"
#include "fx_log.hpp"

#include "llama.h"

#include "json.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <sstream>

// llama.cpp's pre-tokenizer split (src/unicode.h; impulse-fx links llama statically)
std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode);

namespace fxf {

// ================================================================================================ Python's text rules
std::string py_float_repr(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    // the shortest digits that read back to v (Python's repr), then Python's layout: fixed for -4 < exponent <= 16
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    bool neg = false;
    if (!s.empty() && s[0] == '-') { neg = true; s.erase(0, 1); }
    const size_t e = s.find('e');
    std::string mant = s.substr(0, e);
    const int exp10 = std::atoi(s.c_str() + e + 1);
    std::string digits;
    for (char c : mant) if (c != '.') digits += c;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1;   // value = 0.digits * 10^decpt
    std::string out;
    if (decpt > -4 + 0 && decpt <= 16 && !(decpt <= -4)) {
        if (decpt <= 0) {
            out = "0." + std::string((size_t) -decpt, '0') + digits;
        } else if ((size_t) decpt >= digits.size()) {
            out = digits + std::string((size_t) decpt - digits.size(), '0') + ".0";
        } else {
            out = digits.substr(0, (size_t) decpt) + "." + digits.substr((size_t) decpt);
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        const int x = decpt - 1;
        char eb[16];
        snprintf(eb, sizeof eb, "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        out += eb;
    }
    return neg ? "-" + out : out;
}

static void utf8_append(std::string & o, uint32_t cp) {
    if (cp < 0x80) o += (char) cp;
    else if (cp < 0x800) { o += (char) (0xC0 | (cp >> 6)); o += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char) (0xE0 | (cp >> 12)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
    else { o += (char) (0xF0 | (cp >> 18)); o += (char) (0x80 | ((cp >> 12) & 0x3F)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
}

// one code point from valid UTF-8 at s[i] (i advanced); invalid bytes read as themselves (Latin-1), never thrown
static uint32_t utf8_next(const std::string & s, size_t & i) {
    const unsigned char c = (unsigned char) s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + (size_t) n > s.size()) n = 1;
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char) s[i + (size_t) k] & 0x3F);
    i += (size_t) n;
    return cp;
}

bool utf8_valid(const std::string & s) {
    Utf8Decoder d;
    const std::string out = d.push(s, true);
    return out == s;
}

uint32_t utf8_last_cp(const std::string & s, size_t end, size_t * len) {
    if (end == 0) { if (len) *len = 0; return 0; }
    size_t b = end - 1;
    while (b > 0 && ((unsigned char) s[b] & 0xC0) == 0x80 && end - b < 4) --b;
    size_t i = b;
    const uint32_t cp = utf8_next(s, i);
    if (i != end) { if (len) *len = 1; return (unsigned char) s[end - 1]; }
    if (len) *len = end - b;
    return cp;
}

std::string py_dumps_str(const std::string & s, bool ensure_ascii) {
    std::string o = "\"";
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c < 0x80) {
            ++i;
            switch (c) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                case '\b': o += "\\b"; break;
                case '\f': o += "\\f"; break;
                default:
                    if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                    else o += (char) c;
            }
            continue;
        }
        const size_t st = i;
        const uint32_t cp = utf8_next(s, i);
        if (!ensure_ascii) { o.append(s, st, i - st); continue; }
        char b[16];
        if (cp >= 0x10000) {
            const uint32_t v = cp - 0x10000;
            snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
        } else snprintf(b, sizeof b, "\\u%04x", cp);
        o += b;
    }
    return o + "\"";
}

static void dumps_into(std::string & o, const json & v, bool ascii, const char * is, const char * ks) {
    switch (v.type()) {
        case json::value_t::null: o += "null"; break;
        case json::value_t::boolean: o += v.get<bool>() ? "true" : "false"; break;
        case json::value_t::number_integer: o += std::to_string(v.get<int64_t>()); break;
        case json::value_t::number_unsigned: o += std::to_string(v.get<uint64_t>()); break;
        case json::value_t::number_float: o += py_float_repr(v.get<double>()); break;
        case json::value_t::string: o += py_dumps_str(v.get_ref<const std::string &>(), ascii); break;
        case json::value_t::array: {
            o += "[";
            bool first = true;
            for (const auto & x : v) { if (!first) o += is; first = false; dumps_into(o, x, ascii, is, ks); }
            o += "]";
            break;
        }
        case json::value_t::object: {
            o += "{";
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) o += is;
                first = false;
                o += py_dumps_str(it.key(), ascii);
                o += ks;
                dumps_into(o, it.value(), ascii, is, ks);
            }
            o += "}";
            break;
        }
        default: o += "null";
    }
}

std::string py_dumps(const json & v, bool ensure_ascii, const char * item_sep, const char * key_sep) {
    std::string o;
    dumps_into(o, v, ensure_ascii, item_sep, key_sep);
    return o;
}

bool py_isspace_cp(uint32_t cp) {
    switch (cp) {
        case 0x9: case 0xa: case 0xb: case 0xc: case 0xd: case 0x1c: case 0x1d: case 0x1e: case 0x1f: case 0x20:
        case 0x85: case 0xa0: case 0x1680: case 0x2028: case 0x2029: case 0x202f: case 0x205f: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200a;
    }
}

std::string py_strip(const std::string & s, bool left, bool right) {
    size_t a = 0, b = s.size();
    if (left) {
        while (a < b) { size_t i = a; const uint32_t cp = utf8_next(s, i); if (!py_isspace_cp(cp)) break; a = i; }
    }
    if (right) {
        while (b > a) { size_t n = 0; const uint32_t cp = utf8_last_cp(s, b, &n); if (!n || !py_isspace_cp(cp)) break; b -= n; }
    }
    return s.substr(a, b - a);
}

std::string py_strip_chars(const std::string & s, const std::string & chars, bool left, bool right) {
    size_t a = 0, b = s.size();
    if (left) while (a < b && chars.find(s[a]) != std::string::npos) ++a;
    if (right) while (b > a && chars.find(s[b - 1]) != std::string::npos) --b;
    return s.substr(a, b - a);
}

// CPython's UTF-8 decoder, errors="replace": a maximal invalid subpart -> one U+FFFD (Unicode's recommendation, which
// CPython follows: an invalid lead byte alone, or a lead and the continuation bytes that were valid so far)
std::string Utf8Decoder::push(const std::string & in, bool final) {
    std::string s = held + in;
    held.clear();
    std::string o;
    size_t i = 0;
    static const char * FFFD = "\xEF\xBF\xBD";
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c < 0x80) { o += (char) c; ++i; continue; }
        int need;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c == 0xE0) { need = 2; lo = 0xA0; }
        else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) need = 2;
        else if (c == 0xED) { need = 2; hi = 0x9F; }
        else if (c == 0xF0) { need = 3; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) need = 3;
        else if (c == 0xF4) { need = 3; hi = 0x8F; }
        else { o += FFFD; ++i; continue; }
        size_t j = i + 1;
        int got = 0;
        bool bad = false;
        while (got < need) {
            if (j >= s.size()) break;
            const unsigned char d = (unsigned char) s[j];
            const unsigned char l = got == 0 ? lo : 0x80, h = got == 0 ? hi : 0xBF;
            if (d < l || d > h) { bad = true; break; }
            ++j; ++got;
        }
        if (got == need) { o.append(s, i, j - i); i = j; continue; }
        if (!bad && j >= s.size() && !final) { held = s.substr(i); break; }   // incomplete: wait for the rest
        o += FFFD;
        i = j;   // the maximal subpart (lead + the valid continuation bytes) is one U+FFFD
    }
    return o;
}

// ================================================================================================ tokenizer
static std::string token_raw_bytes(const llama_vocab * v, int32_t id) {
    std::string out(64, '\0');
    int n = llama_token_to_piece(v, id, out.data(), (int32_t) out.size(), 0, true);
    if (n < 0) { out.resize((size_t) -n); n = llama_token_to_piece(v, id, out.data(), (int32_t) out.size(), 0, true); }
    out.resize((size_t) std::max(0, n));
    return out;
}

void Tokenizer::init(const llama_vocab * v, const std::vector<std::string> & merges) {
    vocab = v;
    n_vocab = llama_vocab_n_tokens(v);
    bytes.resize((size_t) n_vocab);
    std::map<std::string, int> lit;   // text -> id (the last id with that text wins, as a dict built in id order)
    std::vector<std::pair<std::string, int>> alw;
    for (int32_t i = 0; i < n_vocab; ++i) {
        bytes[(size_t) i] = token_raw_bytes(v, i);
        text_ids[llama_vocab_get_text(v, i)] = i;
        const auto attr = llama_vocab_get_attr(v, i);
        if (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
            const std::string t = llama_vocab_get_text(v, i);
            lit[t] = i;
            if (attr & LLAMA_TOKEN_ATTR_CONTROL) control_tokens.push_back(t);
        }
    }
    for (const auto & [t, id] : lit) {
        specials.push_back({ t, id });
        if (llama_vocab_get_attr(v, id) & LLAMA_TOKEN_ATTR_USER_DEFINED) always.push_back({ t, id });
        max_special_len = std::max(max_special_len, t.size());
    }
    // longest first (Python: sorted(literals, key=len, reverse=True) - stable over the dict's id order)
    auto by_len = [&](std::vector<std::pair<std::string, int>> & L) {
        std::sort(L.begin(), L.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
        std::stable_sort(L.begin(), L.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    };
    by_len(specials);
    by_len(always);
    for (size_t i = 0; i < merges.size(); ++i) {
        const size_t sp = merges[i].find(' ');
        if (sp == std::string::npos) continue;
        ranks.emplace(std::make_pair(merges[i].substr(0, sp), merges[i].substr(sp + 1)), (int) i);
    }
    auto id_of = [&](const char * t) { auto it = lit.find(t); return it == lit.end() ? -1 : it->second; };
    im_start = id_of("<|im_start|>");
    im_end = id_of("<|im_end|>");
    endoftext = id_of("<|endoftext|>");
}

// _bpe_heap on one pre-tokenizer word (byte-mapped): the lowest rank first, then the leftmost
static std::vector<std::string> bpe_word(const std::string & word, const std::map<std::pair<std::string, std::string>, int> & ranks) {
    std::vector<std::string> parts;
    for (size_t i = 0; i < word.size();) { const size_t st = i; utf8_next(word, i); parts.push_back(word.substr(st, i - st)); }
    const int n = (int) parts.size();
    std::vector<int> nxt(n), prv(n);
    std::vector<bool> gone(n, false);
    for (int i = 0; i < n; ++i) { nxt[i] = i + 1 < n ? i + 1 : -1; prv[i] = i - 1; }
    using Item = std::tuple<int, int, std::string, std::string>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    auto rank = [&](const std::string & a, const std::string & b) { auto it = ranks.find({ a, b }); return it == ranks.end() ? -1 : it->second; };
    for (int i = 0; i + 1 < n; ++i) { const int r = rank(parts[i], parts[i + 1]); if (r >= 0) heap.emplace(r, i, parts[i], parts[i + 1]); }
    while (!heap.empty()) {
        auto [r, i, left, right] = heap.top();
        heap.pop();
        const int j = nxt[i];
        if (gone[i] || parts[i] != left || j < 0 || parts[j] != right) continue;
        parts[i] = left + right;
        gone[j] = true;
        const int k = nxt[j];
        nxt[i] = k;
        if (k >= 0) { prv[k] = i; const int r2 = rank(parts[i], parts[k]); if (r2 >= 0) heap.emplace(r2, i, parts[i], parts[k]); }
        const int p = prv[i];
        if (p >= 0) { const int r2 = rank(parts[p], parts[i]); if (r2 >= 0) heap.emplace(r2, p, parts[p], parts[i]); }
    }
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) if (!gone[i]) out.push_back(parts[i]);
    return out;
}

static const char * QWEN35_REGEX =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<int32_t> Tokenizer::bpe_plain(const std::string & text) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;
    for (const auto & word : unicode_regex_split(text, { QWEN35_REGEX }, true)) {
        for (const auto & piece : bpe_word(word, ranks)) {
            auto it = text_ids.find(piece);
            if (it == text_ids.end()) throw std::runtime_error("BPE produced a token outside the vocabulary");
            out.push_back(it->second);
        }
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool parse_special,
                                       const std::vector<std::pair<size_t, size_t>> & plain_in) const {
    const auto & lits = parse_special ? specials : always;
    bool first_byte[256] = {};
    for (const auto & L : lits) if (!L.first.empty()) first_byte[(unsigned char) L.first[0]] = true;
    std::vector<std::pair<size_t, size_t>> plain = plain_in;
    std::sort(plain.begin(), plain.end());
    std::vector<int32_t> out;
    size_t pos = 0, seg_start = 0, pi = 0, plain_end = 0;
    auto flush = [&](size_t end) {
        if (end <= seg_start) return;
        const std::string seg = text.substr(seg_start, end - seg_start);
        bool literal_inside = false;   // a literal's text left as text (#537): llama.cpp would match a type-4 one
        for (const auto & L : always) if (seg.find(L.first) != std::string::npos) { literal_inside = true; break; }
        std::vector<int32_t> ids;
        if (literal_inside) ids = bpe_plain(seg);
        else {
            // llama.cpp's tokenizer on a stretch with no literal in it (gate P1a: the same ids as Strata's)
            int n = llama_tokenize(vocab, seg.data(), (int32_t) seg.size(), nullptr, 0, false, false);
            ids.resize((size_t) (n < 0 ? -n : n));
            n = llama_tokenize(vocab, seg.data(), (int32_t) seg.size(), ids.data(), (int32_t) ids.size(), false, false);
            ids.resize((size_t) std::max(0, n));
        }
        out.insert(out.end(), ids.begin(), ids.end());
    };
    // regex finditer over the alternation: the leftmost match, at it the first (longest) literal that matches
    while (pos < text.size()) {
        size_t at = std::string::npos;
        const std::pair<std::string, int> * best = nullptr;
        for (size_t q = pos; q < text.size() && !best; ++q) {
            if (!first_byte[(unsigned char) text[q]]) continue;
            for (const auto & L : lits)
                if (text.compare(q, L.first.size(), L.first) == 0) { best = &L; at = q; break; }
        }
        if (!best) break;
        while (pi < plain.size() && plain[pi].first <= at) { plain_end = std::max(plain_end, plain[pi].second); ++pi; }
        pos = at + best->first.size();
        if (at < plain_end) continue;   // starts inside a plain span: text; the scan goes on after it
        flush(at);
        out.push_back(best->second);
        seg_start = pos;
    }
    flush(text.size());
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string o;
    for (auto t : ids) o += token_text(t);
    return o;
}

// ================================================================================================ requests
static bool truthy(const json & v) {   // Python's bool() of a JSON value
    switch (v.type()) {
        case json::value_t::null: return false;
        case json::value_t::boolean: return v.get<bool>();
        case json::value_t::number_integer: return v.get<int64_t>() != 0;
        case json::value_t::number_unsigned: return v.get<uint64_t>() != 0;
        case json::value_t::number_float: return v.get<double>() != 0.0;
        case json::value_t::string: return !v.get_ref<const std::string &>().empty();
        case json::value_t::array: case json::value_t::object: return !v.empty();
        default: return false;
    }
}
static json get(const json & o, const char * k) {   // dict.get(k) (None for a non-dict or a missing key)
    if (!o.is_object()) return nullptr;
    auto it = o.find(k);
    return it == o.end() ? json(nullptr) : *it;
}
static bool has(const json & o, const char * k) { return o.is_object() && o.contains(k); }

static std::string py_repr_value(const json & v) {   // repr() of a JSON value, for messages
    if (v.is_string()) {
        const std::string & s = v.get_ref<const std::string &>();
        const char q = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
        std::string o(1, q);
        for (char c : s) { if (c == '\\' || c == q) o += '\\'; o += c; }
        return o + q;
    }
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_null()) return "None";
    return py_dumps(v, false);
}

static bool is_text_type(const json & part) {   // TEXT_PARTS = ("text", "input_text", None)
    const json t = get(part, "type");
    return t.is_null() || (t.is_string() && (t == "text" || t == "input_text"));
}
static bool is_image_type(const json & part) {  // IMAGE_PARTS
    const json t = get(part, "type");
    return t.is_string() && (t == "image_url" || t == "input_image" || t == "image");
}

static std::string text_of(const json & content) {   // _text_of
    if (content.is_null()) return "";
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array()) throw BadRequest("the request is malformed (TypeError: a message's content must be a string or a list of parts)");
    std::string out;
    for (const auto & part : content) {
        if (!part.is_object() || !is_text_type(part)) continue;
        const json t = get(part, "text");
        if (!has(part, "text")) continue;
        if (!t.is_string()) throw BadRequest("the request is malformed (TypeError: sequence item: expected str instance)");
        out += t.get<std::string>();
    }
    return out;
}

static bool has_image(const json & content) {
    if (!content.is_array()) return false;
    for (const auto & p : content) if (p.is_object() && is_image_type(p)) return true;
    return false;
}

static std::string image_source(const json & part) {   // _image_source
    if (get(part, "type") == "image") {
        json src = get(part, "source");
        if (!truthy(src)) src = json::object();
        if (get(src, "type") == "base64") {
            const json mt = get(src, "media_type"), d = get(src, "data");
            return "data:" + (has(src, "media_type") && mt.is_string() ? mt.get<std::string>() : std::string("image/png")) + ";base64," +
                   (d.is_string() ? d.get<std::string>() : std::string());
        }
        json u = get(src, "url");
        if (!truthy(u)) u = get(src, "path");
        return truthy(u) && u.is_string() ? u.get<std::string>() : "";
    }
    json url = get(part, "image_url");
    if (url.is_object()) url = get(url, "url");
    return truthy(url) && url.is_string() ? url.get<std::string>() : "";
}

static json parts_of(const json & content) {   // _parts_of
    if (!has_image(content)) return text_of(content);
    json items = json::array();
    for (const auto & part : content) {
        if (!part.is_object()) continue;
        if (is_image_type(part)) items.push_back(json{ { "type", "image" }, { "source", image_source(part) } });
        else if (is_text_type(part) && has(part, "text")) items.push_back(json{ { "type", "text" }, { "text", get(part, "text") } });
    }
    return items;
}

static json check_parts(const json & content) {   // _check_parts: a refusal part reads as its text
    if (!content.is_array()) return content;
    json out = content;
    for (auto & part : out) {
        if (part.is_object() && get(part, "type") == "refusal") {
            const json r = get(part, "refusal");
            part = json{ { "type", "text" }, { "text", truthy(r) ? r : json("") } };
        }
    }
    return out;
}

static json object_list(const json & value, const std::string & name) {   // _object_list (#460)
    if (value.is_null()) return json::array();
    json v = value;
    if (v.is_string()) {
        try { v = json::parse(v.get<std::string>()); }
        catch (const std::exception &) { throw BadRequest(name + " must be a list of objects (a string was sent that is not JSON)"); }
    }
    if (!v.is_array()) throw BadRequest(name + " must be a list of objects");
    for (const auto & m : v) if (!m.is_object()) throw BadRequest(name + " must be a list of objects");
    return v;
}

// _stable_tool_order: the same SET of tools comes back in the order it was first seen (an LRU of 64 sets, process-wide)
static std::mutex g_tool_orders_mu;
static std::vector<std::pair<std::set<std::string>, std::vector<std::string>>> g_tool_orders;   // oldest first

static json stable_tool_order(const json & tools, const std::vector<std::string> & names) {
    if (tools.size() < 2) return tools;
    std::set<std::string> key(names.begin(), names.end());
    if (key.size() != names.size()) return tools;
    std::vector<std::string> order;
    {
        std::lock_guard<std::mutex> g(g_tool_orders_mu);
        auto it = std::find_if(g_tool_orders.begin(), g_tool_orders.end(), [&](const auto & e) { return e.first == key; });
        if (it == g_tool_orders.end()) {
            g_tool_orders.push_back({ key, names });
            while (g_tool_orders.size() > 64) g_tool_orders.erase(g_tool_orders.begin());
            return tools;
        }
        order = it->second;
        auto e = *it;
        g_tool_orders.erase(it);
        g_tool_orders.push_back(e);   // move_to_end
    }
    std::map<std::string, size_t> pos;
    for (size_t k = 0; k < order.size(); ++k) pos[order[k]] = k;
    std::vector<size_t> idx(tools.size());
    for (size_t k = 0; k < idx.size(); ++k) idx[k] = k;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return pos[names[a]] < pos[names[b]]; });
    json out = json::array();
    for (auto k : idx) out.push_back(tools[k]);
    return out;
}

static json tool_list(const json & value) {   // _tool_list(value, "function")
    if (!truthy(value)) return json::array();
    const std::string shape = "{\"type\": \"function\", \"function\": {\"name\": ..., \"parameters\": {...}}}";
    json tools;
    try { tools = object_list(value, "tools"); }
    catch (const BadRequest &) { throw BadRequest("tools must be a list of tool objects, each " + shape); }
    std::vector<std::string> names;
    for (size_t i = 0; i < tools.size(); ++i) {
        const json & t = tools[i];
        const json fn = get(t, "type") == "function" ? (has(t, "function") ? t["function"] : t) : t;
        if (!fn.is_object() || !get(fn, "name").is_string() || get(fn, "name").get<std::string>().empty())
            throw BadRequest("tools[" + std::to_string(i) + "] has no name: each tool must be " + shape);
        const json schema = get(fn, "parameters");
        if (!schema.is_null() && !schema.is_object()) {
            const char * tn = schema.is_string() ? "str" : schema.is_array() ? "list" : schema.is_boolean() ? "bool" :
                              schema.is_number_integer() ? "int" : schema.is_number_float() ? "float" : "object";
            throw BadRequest("tools[" + std::to_string(i) + "] (" + fn["name"].get<std::string>() +
                             "): \"parameters\" must be an object (the JSON schema of its parameters), not " + tn);
        }
        names.push_back(fn["name"].get<std::string>());
    }
    return stable_tool_order(tools, names);
}

static json tool_arguments(const json & raw) {   // tool_arguments (#510)
    if (raw.is_null() || (raw.is_string() && py_strip(raw.get<std::string>()).empty())) return json::object();
    if (raw.is_object()) return raw;
    if (raw.is_string()) {
        try {
            json v = json::parse(raw.get<std::string>());
            return v.is_object() ? v : json{ { "arguments", raw } };
        } catch (const std::exception &) { return json{ { "arguments", raw } }; }
    }
    return json{ { "arguments", py_dumps(raw, false) } };
}

static json effort_kwargs(const json & value) {   // effort_kwargs
    if (value.is_null() || (value.is_string() && value.get<std::string>().empty())) return json::object();
    if (value.is_boolean() && !value.get<bool>()) return json{ { "enable_thinking", false } };
    std::string key = value.is_string() ? value.get<std::string>() : value.is_boolean() ? std::string("True") : py_dumps(value, false);
    key = py_strip(key);
    for (auto & c : key) c = (char) std::tolower((unsigned char) c);
    static const std::map<std::string, std::string> EFFORT = {
        { "none", "" }, { "off", "" }, { "minimal", "" }, { "disabled", "" }, { "false", "" },
        { "low", "low" }, { "medium", "medium" }, { "high", "xhigh" }, { "xhigh", "xhigh" }, { "max", "xhigh" }, { "maximum", "xhigh" } };
    auto it = EFFORT.find(key);
    if (it == EFFORT.end()) throw BadRequest("unknown reasoning effort " + py_repr_value(value) + ": use none, low, medium or high");
    if (it->second.empty()) return json{ { "enable_thinking", false } };
    return json{ { "reasoning_effort", it->second } };
}

void openai_to_messages(const json & req, json & messages, json & tools, json & kwargs) {
    messages = json::array();
    const json list = object_list(get(req, "messages"), "messages");
    for (size_t i = 0; i < list.size(); ++i) {
        const json & m = list[i];
        json role = get(m, "role");
        if (role == "developer") role = "system";
        const bool images = role == "user" || role == "tool" || role == "assistant";
        const json content = check_parts(get(m, "content"));
        json out = json::object();
        out["role"] = role;
        out["content"] = images ? parts_of(content) : json(text_of(content));
        if (truthy(get(m, "reasoning_content"))) out["reasoning_content"] = m["reasoning_content"];
        if (truthy(get(m, "tool_calls"))) {
            json calls = json::array();
            for (const auto & c : object_list(m["tool_calls"], "tool_calls")) {
                const json fn = has(c, "function") ? c["function"] : c;
                if (!fn.is_object()) throw BadRequest("tool_calls must be a list of objects (each with a \"function\" object)");
                calls.push_back(json{ { "function", json{ { "name", get(fn, "name") }, { "arguments", tool_arguments(get(fn, "arguments")) } } } });
            }
            out["tool_calls"] = calls;
        }
        messages.push_back(out);
    }
    tools = json::array();
    for (const auto & t : tool_list(get(req, "tools")))
        tools.push_back(get(t, "type") == "function" ? (has(t, "function") ? t["function"] : t) : t);
    if (tools.empty()) tools = nullptr;
    kwargs = json::object();
    const json reasoning = get(req, "reasoning").is_object() ? req["reasoning"] : json::object();
    const json given = get(req, "reasoning_effort");
    const json eff = !given.is_null() && !(given.is_string() && given.get<std::string>().empty()) ? given : get(reasoning, "effort");
    { const json e = effort_kwargs(eff); for (auto it = e.begin(); it != e.end(); ++it) kwargs[it.key()] = it.value(); }
    const json ctk = get(req, "chat_template_kwargs");
    if (truthy(ctk)) {
        if (!ctk.is_object()) throw BadRequest("the request is malformed (AttributeError: 'list' object has no attribute 'items')");
        for (auto it = ctk.begin(); it != ctk.end(); ++it) {
            if (it.key() == "enable_thinking" && !truthy(it.value())) kwargs = json{ { "enable_thinking", false } };
            else if (it.key() == "reasoning_effort" && !kwargs.contains("enable_thinking"))
                { const json e = effort_kwargs(it.value()); for (auto jt = e.begin(); jt != e.end(); ++jt) kwargs[jt.key()] = jt.value(); }
        }
    }
    // _late_system_to_user (env_to_user is off: the config default)
    for (size_t i = 1; i < messages.size(); ++i) if (messages[i]["role"] == "system") messages[i]["role"] = "user";
}

std::pair<std::string, std::string> tool_choice_of(const json & tc) {
    if (tc.is_null()) return { "auto", "" };
    if (tc.is_string()) {
        const std::string s = tc.get<std::string>();
        return (s == "auto" || s == "none" || s == "required") ? std::make_pair(s, std::string()) : std::make_pair(std::string("unknown"), std::string());
    }
    if (tc.is_object()) {
        const json kind = get(tc, "type");
        if (kind == "auto" || kind == "none") return { kind.get<std::string>(), "" };
        if (kind == "any" || kind == "required") return { "required", "" };
        if (kind == "function" || kind == "tool") {
            const json fn = get(tc, "function");
            json name = fn.is_object() ? get(fn, "name") : json(nullptr);
            if (!truthy(name)) name = get(tc, "name");
            if (name.is_string() && !name.get<std::string>().empty()) return { "named", name.get<std::string>() };
        }
    }
    return { "unknown", "" };
}

std::string forced_call(const json & tool_choice, const json & tools) {
    auto [kind, name] = tool_choice_of(tool_choice);
    std::vector<std::string> names;   // a set in Python: one entry per name
    if (tools.is_array())
        for (const auto & t : tools)
            if (t.is_object()) {
                const json n = get(t, "name");
                const std::string s = n.is_string() ? n.get<std::string>() : std::string("\x01null");
                if (std::find(names.begin(), names.end(), s) == names.end()) names.push_back(s);
            }
    if (kind == "auto" || kind == "none") return "";
    if (kind == "required" && names.size() == 1) { kind = "named"; name = names[0]; }
    if (kind == "required" && !names.empty()) return "<tool_call>\n<function=";
    if (kind == "named" && std::find(names.begin(), names.end(), name) != names.end()) return "<tool_call>\n<function=" + name + ">\n";
    fx::say("[impulse fx] tool_choice %s is not supported here (or names no tool of the request): the model decides, as with \"auto\"\n",
            py_dumps(tool_choice, true).substr(0, 200).c_str());
    return "";
}

// ================================================================================================ the prompt
// #537 #554 #2026: the literals a message's text may spell; LITERAL_TAGS' order, then the control tokens'
static const std::vector<std::pair<std::string, uint32_t>> LITERAL_TAGS = {
    { "<think>", 0xF0E01 }, { "</think>", 0xF0E02 },
    { "<|vision_start|>", 0xF0E03 }, { "<|image_pad|>", 0xF0E04 }, { "<|vision_end|>", 0xF0E05 }, { "<|video_pad|>", 0xF0E06 },
    { "</tool_call>", 0xF0E07 }, { "<tool_call>", 0xF0E08 }, { "<tool_response>", 0xF0E09 }, { "</tool_response>", 0xF0E0A } };
static bool is_tool_literal(const std::string & t) {
    return t == "</tool_call>" || t == "<tool_call>" || t == "<tool_response>" || t == "</tool_response>";
}
using Tags = std::vector<std::pair<std::string, std::string>>;   // (text, mark), longest first

static std::string mark_text(std::string text, const Tags & tags) {
    for (const auto & [tag, mark] : tags) {
        if (text.find(tag) == std::string::npos) continue;
        std::string o;
        size_t p = 0, q;
        while ((q = text.find(tag, p)) != std::string::npos) { o.append(text, p, q - p); o += mark; p = q + tag.size(); }
        o.append(text, p, std::string::npos);
        text.swap(o);
    }
    return text;
}
static json mark_deep(const json & v, const Tags & tags) {
    if (v.is_string()) return mark_text(v.get<std::string>(), tags);
    if (v.is_object()) { json o = json::object(); for (auto it = v.begin(); it != v.end(); ++it) o[it.key()] = mark_deep(it.value(), tags); return o; }
    if (v.is_array()) { json o = json::array(); for (const auto & x : v) o.push_back(mark_deep(x, tags)); return o; }
    return v;
}
static bool has_tag(const json & v, const Tags & tags) {
    if (v.is_string()) { for (const auto & t : tags) if (v.get_ref<const std::string &>().find(t.first) != std::string::npos) return true; return false; }
    if (v.is_object() || v.is_array()) { for (const auto & x : v) if (has_tag(x, tags)) return true; }
    return false;
}

static const std::string TR_OPEN = "<tool_response>", TR_CLOSE = "</tool_response>";

// _keep_tool_result_wrap: a user message that is wholly <tool_response>...</tool_response> keeps that outer pair
static std::optional<json> keep_tool_result_wrap(const json & content, const Tags & tags) {
    const std::string & o = TR_OPEN, & c = TR_CLOSE;
    auto wrap = [&](const std::string & text, bool first, bool last) {
        const size_t i = first ? text.find(o) + o.size() : 0;
        const size_t j = last ? text.rfind(c) : text.size();
        const std::string head = first ? mark_text(text.substr(0, i - o.size()), tags) + o : "";
        const std::string tail = last ? c + mark_text(text.substr(j + c.size()), tags) : "";
        return head + mark_text(text.substr(i, j - i), tags) + tail;
    };
    auto pair_ok = [&](const std::string & body) {
        return body.size() >= o.size() + c.size() && body.compare(0, o.size(), o) == 0 &&
               body.compare(body.size() - c.size(), c.size(), c) == 0;
    };
    if (content.is_string()) {
        const std::string s = content.get<std::string>();
        if (pair_ok(py_strip(s))) return json(wrap(s, true, true));
        return std::nullopt;
    }
    if (!content.is_array() || content.empty()) return std::nullopt;
    for (const auto & x : content) if (!x.is_object()) return std::nullopt;
    for (const auto & x : content) {
        const json t = get(x, "text"), ty = get(x, "type");
        if (!(t.is_string() || ty == "image" || ty == "video")) return std::nullopt;
    }
    if (!get(content.front(), "text").is_string() || !get(content.back(), "text").is_string()) return std::nullopt;
    std::string all;
    for (const auto & x : content) if (get(x, "text").is_string()) all += x["text"].get<std::string>();
    if (!pair_ok(py_strip(all))) return std::nullopt;
    const std::string first = content.front()["text"], last = content.back()["text"];
    const std::string fl = py_strip(first, true, false), lr = py_strip(last, false, true);
    if (!(fl.compare(0, o.size(), o) == 0 && lr.size() >= c.size() && lr.compare(lr.size() - c.size(), c.size(), c) == 0)) return std::nullopt;
    if (content.size() == 1) { json one = content[0]; one["text"] = wrap(first, true, true); return json::array({ one }); }
    json out = json::array();
    json f = content.front(); f["text"] = wrap(first, true, false); out.push_back(f);
    for (size_t k = 1; k + 1 < content.size(); ++k) out.push_back(mark_deep(content[k], tags));
    json l = content.back(); l["text"] = wrap(last, false, true); out.push_back(l);
    return out;
}

// mark_think_literals -> changed?
static bool mark_literals(json & messages, json & tools, const Tags & tags) {
    if (!has_tag(messages, tags) && !has_tag(tools, tags)) return false;
    Tags own;
    for (const auto & t : tags) if (!is_tool_literal(t.first)) own.push_back(t);
    json out = json::array();
    for (const auto & m0 : messages) {
        json m = m0;
        const json content = get(m0, "content");
        const bool assistant = get(m0, "role") == "assistant";
        const Tags & mine = assistant ? own : tags;
        for (auto it = m.begin(); it != m.end(); ++it) if (it.key() != "role") it.value() = mark_deep(it.value(), mine);
        if (get(m0, "role") == "user") {
            auto kept = keep_tool_result_wrap(content, tags);
            if (kept) m["content"] = *kept;
        }
        if (assistant && content.is_string()) {
            const std::string s = content.get<std::string>();
            if (py_strip(s, true, false).compare(0, 7, "<think>") == 0 && s.find("</think>") != std::string::npos) {
                const size_t i = s.find("<think>") + 7, j = s.find("</think>");
                m["content"] = s.substr(0, i) + mark_text(s.substr(i, j - i), own) + "</think>" + mark_text(s.substr(j + 8), own);
            }
        }
        out.push_back(m);
    }
    messages = out;
    tools = mark_deep(tools, tags);
    return true;
}

// unmark_think_literals: the marks back to their texts, and the byte spans of those texts (encoded as text)
static std::string unmark(const std::string & prompt, const Tags & tags, std::vector<std::pair<size_t, size_t>> & spans) {
    std::map<std::string, std::string> back;
    for (const auto & [t, m] : tags) back[m] = t;
    std::string out;
    size_t i = 0;
    while (i < prompt.size()) {
        const size_t st = i;
        const uint32_t cp = utf8_next(prompt, i);
        if (cp >= 0xF0E01 && cp < 0xF1000) {
            auto it = back.find(prompt.substr(st, i - st));
            if (it != back.end()) { spans.push_back({ out.size(), out.size() + it->second.size() }); out += it->second; continue; }
        }
        out.append(prompt, st, i - st);
    }
    return out;
}

struct TemplateProgram { jinja::program prog; std::string src; };

void Front::init(const llama_vocab * vocab, const std::vector<std::string> & merges, const std::string & tmpl) {
    tok.init(vocab, merges);
    template_source = tmpl;
    jinja::lexer lexer;
    auto res = lexer.tokenize(tmpl);
    auto p = std::make_shared<TemplateProgram>();
    p->prog = jinja::parse_from_tokens(res);
    p->src = res.source;
    program_ = p;
    // literal_tags(controls)
    std::vector<std::pair<std::string, std::string>> tags;
    for (const auto & [t, cp] : LITERAL_TAGS) { std::string m; utf8_append(m, cp); tags.push_back({ t, m }); }
    uint32_t k = 0;
    for (const auto & c : tok.control_tokens) {
        bool known = false;
        for (const auto & t : LITERAL_TAGS) if (t.first == c) known = true;
        if (known) continue;
        std::string m;
        utf8_append(m, 0xF0E10 + k++);
        tags.push_back({ c, m });
    }
    std::stable_sort(tags.begin(), tags.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    literals_ = tags;
}

static std::string jinja_message(const std::string & what) {   // the template's raise_exception text, as Python's
    std::string s = what;
    const std::string pre = "Jinja Exception: ";
    const size_t p = s.find(pre);
    if (p != std::string::npos) s = s.substr(p + pre.size());
    const size_t nl = s.find('\n');
    if (nl != std::string::npos) s = s.substr(0, nl);
    return s;
}

std::string Front::render(const json & messages_in, const json & tools, const json & kwargs, bool add_generation_prompt) const {
    // ChatTemplate.render: completed empty assistant turns are dropped (#843), the last message kept
    json messages = json::array();
    const bool keep_empty = getenv("STRATA_KEEP_EMPTY_TURNS") && std::string(getenv("STRATA_KEEP_EMPTY_TURNS")) == "1";
    for (size_t i = 0; i < messages_in.size(); ++i) {
        const json & m = messages_in[i];
        const bool drop = !(i + 1 == messages_in.size() || keep_empty) && m.is_object() && get(m, "role") == "assistant" &&
                          py_strip(text_of(get(m, "content"))).empty() && !has_image(get(m, "content")) && !truthy(get(m, "tool_calls"));
        if (!drop) messages.push_back(m);
    }
    const auto * p = static_cast<const TemplateProgram *>(program_.get());
    jinja::context ctx(p->src);
    json inp = json::object();
    inp["messages"] = messages;
    inp["tools"] = tools;
    inp["add_generation_prompt"] = add_generation_prompt;
    for (auto it = kwargs.begin(); it != kwargs.end(); ++it) inp[it.key()] = it.value();
    jinja::global_from_json(ctx, common_json::parse(inp.dump()), false);   // llama.cpp's own JSON type (the same values)
    try {
        jinja::runtime rt(ctx);
        const jinja::value results = rt.execute(p->prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        return parts->as_string().str();
    } catch (const std::exception & e) {
        throw BadRequest(jinja_message(e.what()));
    }
}

std::vector<int32_t> Front::encode_prompt(const json & messages_in, const json & tools_in, const json & kwargs,
                                          std::string * text_out, std::vector<std::pair<size_t, size_t>> * plain_out) const {
    json messages = messages_in, tools = tools_in;
    const bool changed = mark_literals(messages, tools, literals_);
    std::string prompt = render(messages, tools, kwargs, true);   // effort_position "start" (the default): as asked
    std::vector<std::pair<size_t, size_t>> plain;
    if (changed) prompt = unmark(prompt, literals_, plain);
    if (text_out) *text_out = prompt;
    if (plain_out) *plain_out = plain;
    return tok.encode(prompt, true, plain);
}

std::string Front::model_for(const json & req) const {
    const json asked = get(req, "model");
    if (asked.is_string() && std::find(cfg.aliases.begin(), cfg.aliases.end(), asked.get<std::string>()) != cfg.aliases.end())
        return asked.get<std::string>();
    return cfg.model;
}

static long long py_int(const json & v, const char * what) {   // int(x) for max_tokens (a number, or a numeric string)
    if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_number_unsigned()) return (long long) v.get<uint64_t>();
    if (v.is_number_float()) { const double d = v.get<double>(); if (!std::isfinite(d)) throw BadRequest(std::string("cannot convert float to integer")); return (long long) std::trunc(d); }
    if (v.is_string()) {
        const std::string s = py_strip(v.get<std::string>());
        size_t i = 0;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        bool digits = i < s.size();
        for (size_t k = i; k < s.size(); ++k) if (!std::isdigit((unsigned char) s[k]) && s[k] != '_') digits = false;
        if (!digits) throw BadRequest(std::string("invalid literal for int() with base 10: ") + py_repr_value(v));
        std::string d;
        for (char c : s) if (c != '_') d += c;
        return std::stoll(d);
    }
    throw BadRequest(std::string("the request is malformed (TypeError: int() argument must be a string or a number: ") + what + ")");
}

Prompt Front::prepare(json & req, long long ctx, const ImageEncoder & encode_images) const {
    Prompt P;
    // _openai: bias_api.normalize, openai_to_messages, tool_choice none, forced_call, prepare_format
    logit_bias_normalize(get(req, "logit_bias"), tok.n_vocab);
    openai_to_messages(req, P.messages, P.tools, P.kwargs);
    if (tool_choice_of(get(req, "tool_choice")).first == "none") P.tools = nullptr;
    P.force = forced_call(get(req, "tool_choice"), P.tools);
    P.structured = prepare_format(get(req, "response_format"), P.messages);
    if (P.structured && (truthy(P.tools) || truthy(get(req, "strata_mcp"))))
        throw BadRequest("structured response_format with tools/MCP is not supported");
    json mx = get(req, "max_completion_tokens");
    if (!truthy(mx)) mx = get(req, "max_tokens");
    long long max_new = truthy(mx) ? py_int(mx, "max_tokens") : 0;
    reasoning_budget(req, cfg.reasoning_budget_tokens);   // a bad value is a 400 before anything is sent
    stop_strings(req);
    check_request_sampling(req);
    // Service.prepare: tool pictures that cannot be read become a note (#1840's _note_unreadable_tool_images)
    for (auto & m : P.messages) {
        if (get(m, "role") != "tool" || !m["content"].is_array()) continue;
        for (auto & item : m["content"]) {
            if (!item.is_object() || get(item, "type") != "image") continue;
            std::string why;
            if (!encode_images) why = "this server has no image encoder";
            else {
                try { encode_images({ item["source"].get<std::string>() }); }
                catch (const BadRequest & e) { why = e.what(); }
            }
            if (!why.empty()) {
                fx::say("[impulse fx] a picture a tool returned was left out of the prompt: %s\n", why.c_str());
                item = json{ { "type", "text" }, { "text", "[image omitted: " + why + "]" } };
            }
        }
    }
    std::vector<int32_t> ids = encode_prompt(P.messages, P.tools, P.kwargs, &P.text, &P.plain);
    const json et = get(P.kwargs, "enable_thinking");
    P.thinking = !(et.is_boolean() && !et.get<bool>());
    if (!P.force.empty() && !P.thinking) {
        const auto f = tok.encode(P.force, true);
        ids.insert(ids.end(), f.begin(), f.end());
    }
    for (const auto & m : P.messages)
        if (m.is_object() && m.contains("content") && m["content"].is_array())
            for (const auto & item : m["content"])
                if (item.is_object() && get(item, "type") == "image") P.images.push_back(item["source"].get<std::string>());
    if (!P.images.empty()) {
        if (!encode_images) throw BadRequest("this server was started without the vision encoder (run setup again and choose 'vision'), so it cannot read images");
        const int32_t pad = tok.encode("<|image_pad|>", true)[0], start = tok.encode("<|vision_start|>", true)[0];
        const auto encoded = encode_images(P.images);
        const auto literal = tok.encode("<|image_pad|>", false);
        std::vector<int32_t> out;
        size_t k = 0;
        for (size_t j = 0; j < ids.size(); ++j) {
            const int32_t t = ids[j];
            if (t == pad && j > 0 && ids[j - 1] == start) {
                if (k == encoded.size()) throw BadRequest("the prompt and its images do not match");
                out.insert(out.end(), (size_t) encoded[k].second, pad);
                ++k;
            } else if (t == pad) out.insert(out.end(), literal.begin(), literal.end());
            else out.push_back(t);
        }
        if (k != encoded.size()) throw BadRequest("the prompt and its images do not match");
        ids.swap(out);
    }
    const long long room = ctx - 8 - (long long) ids.size();   // CTX_SLACK
    if (max_new <= 0 || (cfg.fit_max_tokens && room < 1)) {
        if (room < 1)
            throw BadRequest("request exceeds the context window: prompt (" + std::to_string(ids.size()) +
                             " tokens) leaves no room to answer in the context (" + std::to_string(ctx) + "); requests are never truncated");
        max_new = room;
    } else if (max_new > room) {
        if (!cfg.fit_max_tokens)
            throw BadRequest("request exceeds the context window: prompt (" + std::to_string(ids.size()) + " tokens) + max tokens (" +
                             std::to_string(max_new) + ") exceeds the context (" + std::to_string(ctx) + "); requests are never truncated. "
                             "Send a smaller max_tokens (at most " + std::to_string(std::max(0LL, room)) + " here), or add \"fit_max_tokens\": true to the "
                             "model's strata-<model>.json to shorten it to the room left (#545)");
        max_new = std::max(1LL, room);
    }
    P.ids = ids;
    P.max_new = max_new;
    return P;
}

// ================================================================================================ sampling
std::optional<long long> reasoning_budget(const json & req, long long config_default) {
    json v = get(req, "reasoning_budget_tokens");
    if (v.is_null()) v = config_default;
    if (v.is_number_float() && std::isfinite(v.get<double>()) && v.get<double>() == std::trunc(v.get<double>())) v = (long long) v.get<double>();
    if (v.is_boolean() || !(v.is_number_integer() || v.is_number_unsigned()))
        throw BadRequest("reasoning_budget_tokens=" + py_repr_value(v) + ": expected a whole number of tokens (0: no budget)");
    const long long n = v.get<long long>();
    if (n > 0) return n;
    return std::nullopt;
}

std::vector<std::string> stop_strings(const json & req) {
    json stop;
    if (!get(req, "stop").is_null()) {
        stop = req["stop"];
        if (stop.is_string()) stop = json::array({ stop });
        bool ok = stop.is_array();
        if (ok) for (const auto & x : stop) if (!x.is_string()) ok = false;
        if (!ok) throw BadRequest("stop must be a string or a list of strings");
        if (stop.size() > 4) throw BadRequest("stop takes at most 4 strings");
    } else if (!get(req, "stop_sequences").is_null()) {
        stop = req["stop_sequences"];
        bool ok = stop.is_array();
        if (ok) for (const auto & x : stop) if (!x.is_string()) ok = false;
        if (!ok) throw BadRequest("stop_sequences must be a list of strings");
    } else return {};
    std::vector<std::string> out;
    for (const auto & x : stop) if (!x.get<std::string>().empty()) out.push_back(x.get<std::string>());
    return out;
}

void check_request_sampling(json & req) {
    if (!req.is_object()) return;
    if (!get(req, "repeat_penalty").is_null() && get(req, "repetition_penalty").is_null()) req["repetition_penalty"] = req["repeat_penalty"];
    const json rl = get(req, "repeat_last_n");
    if (get(req, "penalty_last_n").is_null() && (rl.is_number_integer() || rl.is_number_unsigned()) && rl.get<long long>() > 0) req["penalty_last_n"] = rl;
    static const std::vector<std::pair<const char *, bool>> keys = {
        { "temperature", false }, { "top_p", false }, { "min_p", false }, { "presence_penalty", false }, { "frequency_penalty", false },
        { "repetition_penalty", false }, { "top_k", true }, { "penalty_last_n", true }, { "seed", true } };
    for (const auto & [key, integer] : keys) {
        json value = get(req, key);
        if (value.is_null()) continue;
        const std::string want = integer ? "an integer" : "a number";
        if (value.is_string() && !py_strip(value.get<std::string>()).empty()) {
            const std::string s = py_strip(value.get<std::string>());
            double num;
            try { size_t used = 0; num = std::stod(s, &used); if (used != s.size()) throw std::invalid_argument("x"); }
            catch (const std::exception &) { throw BadRequest(std::string(key) + ": " + want); }
            if (!std::isfinite(num)) throw BadRequest(std::string(key) + ": " + want);
            bool int_text = !s.empty();
            for (size_t i = 0; i < s.size(); ++i) if (!(std::isdigit((unsigned char) s[i]) || (i == 0 && (s[i] == '+' || s[i] == '-')))) int_text = false;
            if (integer) {
                if (num != std::trunc(num)) throw BadRequest(std::string(key) + ": " + want);
                value = (long long) num;
            } else value = int_text ? json((long long) std::stoll(s)) : json(num);
            req[key] = value;
        }
        if (value.is_boolean() || !(integer ? (value.is_number_integer() || value.is_number_unsigned()) : value.is_number()))
            throw BadRequest(std::string(key) + ": " + want);
    }
}

json logit_bias_normalize(const json & value, int vocab_size) {
    json out = json::object();
    if (value.is_null()) return out;
    std::vector<std::pair<json, json>> entries;
    bool pairs = false;
    if (value.is_object()) { for (auto it = value.begin(); it != value.end(); ++it) entries.push_back({ json(it.key()), it.value() }); }
    else if (value.is_array()) {
        pairs = true;
        for (const auto & e : value) {
            if (!e.is_array() || e.size() != 2) throw BadRequest("logit_bias: each entry must be [token_id, bias]");
            entries.push_back({ e[0], e[1] });
        }
    } else throw BadRequest("logit_bias: expected an object or a list of [token_id, bias] pairs");
    std::map<long long, double> m;
    long long banned = 0;
    for (auto & [tok, bias] : entries) {
        std::string s = tok.is_string() ? tok.get<std::string>() : (tok.is_number_integer() || tok.is_number_unsigned()) && !tok.is_boolean() ? tok.dump() : std::string("x");
        bool digits = !s.empty();
        for (char c : s) if (!std::isdigit((unsigned char) c)) digits = false;
        if (!digits) throw BadRequest("logit_bias: token IDs must be non-negative integers");
        const long long id = s.size() > 12 ? (1LL << 40) : std::stoll(s);
        if (id > 2147483647 || (vocab_size > 0 && id >= vocab_size)) throw BadRequest("logit_bias: token ID outside the model vocabulary");
        if (m.count(id)) throw BadRequest("logit_bias: duplicate token ID");
        json b = bias;
        if (pairs && b.is_boolean() && !b.get<bool>()) b = -100.0;
        if (b.is_boolean() || !b.is_number() || !std::isfinite(b.get<double>()) || b.get<double>() < -100 || b.get<double>() > 100)
            throw BadRequest("logit_bias: biases must be finite numbers between -100 and 100; false is a pair-list ban");
        m[id] = b.get<double>();
        if (m[id] == -100) ++banned;
    }
    if (vocab_size > 0 && banned == vocab_size) throw BadRequest("logit_bias: cannot ban the entire vocabulary");
    for (const auto & [id, b] : m) out[std::to_string(id)] = b;
    return out;
}

std::string sampling_keys(const json & s) {
    std::string keys;
    auto num = [&](const char * k, double & v) {
        const json x = get(s, k);
        if (!(x.is_number() || x.is_boolean())) return false;   // Python: bool is an int
        v = x.is_boolean() ? (x.get<bool>() ? 1.0 : 0.0) : x.get<double>();
        return true;
    };
    double v = 0;
    if (num("temperature", v) && v > 0.0) keys += " temperature=" + py_float_repr(v);
    if (num("top_p", v) && v < 1.0) keys += " top_p=" + py_float_repr(v);
    {
        const json tk = get(s, "top_k");
        if ((tk.is_number_integer() || tk.is_number_unsigned()) && tk.get<long long>() >= 0) {
            const long long t = tk.get<long long>();
            keys += " top_k=" + std::to_string(t >= 1 && t <= 64 ? t : 64);
        }
    }
    if (num("min_p", v) && v > 0.0 && v <= 1.0) keys += " min_p=" + py_float_repr(v);
    double rp = 1, pf = 0, pp = 0;
    const bool rp_on = num("repetition_penalty", rp) && rp != 1.0;
    const bool pf_on = num("frequency_penalty", pf) && pf != 0.0;
    const bool pp_on = num("presence_penalty", pp) && pp != 0.0;
    if (rp_on) keys += " penalty_repeat=" + py_float_repr(rp);
    if (pf_on) keys += " penalty_freq=" + py_float_repr(pf);
    if (pp_on) keys += " penalty_present=" + py_float_repr(pp);
    if (rp_on || pf_on || pp_on) {
        const json pln = get(s, "penalty_last_n");
        if ((pln.is_number_integer() || pln.is_number_unsigned()) && pln.get<long long>() > 0) keys += " penalty_last_n=" + std::to_string(pln.get<long long>());
        else keys += " penalty_last_n=64";
    }
    {
        const json seed = get(s, "seed");
        if ((seed.is_number_integer() || seed.is_number_unsigned() || seed.is_boolean()) && (seed.is_boolean() ? seed.get<bool>() : seed.get<long long>() > 0))
            keys += " seed=" + std::to_string(seed.is_boolean() ? 1LL : seed.get<long long>());
    }
    const json tune = get(s, "strata_tune");
    if (tune.is_object()) {
        for (const char * k : { "pcie_frac", "spec_min_p" }) {
            const json x = get(tune, k);
            if (x.is_number() && !x.is_boolean() && x.get<double>() >= 0.0 && x.get<double>() <= 1.0) keys += std::string(" ") + k + "=" + py_float_repr(x.get<double>());
        }
        const json ac = get(tune, "aux_cpus");
        if (ac.is_boolean()) keys += std::string(" aux_cpus=") + (ac.get<bool>() ? "1" : "0");
        else if (ac.is_number() && (ac.get<double>() == 0.0 || ac.get<double>() == 1.0)) keys += " aux_cpus=" + std::to_string((int) ac.get<double>());
    }
    if (get(s, "strata_checkpoint").is_boolean() && !s["strata_checkpoint"].get<bool>()) keys += " ckpt=0";
    const json ik = get(s, "strata_img_keys");
    if (ik.is_array() && !ik.empty()) {
        bool ok = true;
        for (const auto & k : ik) {
            if (!k.is_string() || k.get<std::string>().empty()) { ok = false; break; }
            for (char c : k.get<std::string>()) if (!std::isdigit((unsigned char) c) && !(c >= 'a' && c <= 'f')) ok = false;
        }
        if (ok) { keys += " img_keys="; for (size_t i = 0; i < ik.size(); ++i) keys += (i ? "," : "") + ik[i].get<std::string>(); }
    }
    const json prefix = get(s, "strata_prefix");
    if (prefix.is_object()) {
        const json pn = get(prefix, "tokens");
        if ((pn.is_number_integer() || pn.is_number_unsigned()) && pn.get<long long>() > 0) keys += " pin=" + std::to_string(pn.get<long long>());
    }
    const json proj = get(s, "experimental_speed_projection");
    if (proj.is_boolean()) keys += std::string(" cvec=") + (proj.get<bool>() ? "1" : "0");
    // logit_bias' engine key: " logit_bias=id:bias,..." ({b:g}), ids sorted
    const json lb = logit_bias_normalize(get(s, "logit_bias"), 0);
    if (!lb.empty()) {
        std::vector<std::pair<long long, double>> e;
        for (auto it = lb.begin(); it != lb.end(); ++it) e.push_back({ std::stoll(it.key()), it.value().get<double>() });
        std::sort(e.begin(), e.end());
        keys += " logit_bias=";
        for (size_t i = 0; i < e.size(); ++i) { char b[64]; snprintf(b, sizeof b, "%lld:%g", e[i].first, e[i].second); keys += (i ? "," : "") + std::string(b); }
    }
    return keys;
}

// ================================================================================================ the answer
static const std::string THINK_END = "</think>", CALL_START = "<tool_call>", CALL_END = "</tool_call>",
                         PARAM_END = "</parameter>", FUNC_START = "<function=", FUNC_END = "</function>", PARAM_START = "<parameter=";
static const size_t RCALL_MAX = 32768;

static bool starts(const std::string & s, const std::string & p, size_t at = 0) { return s.size() >= at + p.size() && s.compare(at, p.size(), p) == 0; }
static bool ends(const std::string & s, const std::string & p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }
static bool prefix_of(const std::string & s, const std::string & whole) { return s.size() <= whole.size() && whole.compare(0, s.size(), s) == 0; }   // whole.startswith(s)

std::string new_hex(size_t n) {
    static thread_local std::mt19937_64 rng{ std::random_device{}() };
    static const char * hex = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) s += hex[rng() & 15];
    return s;
}
std::string new_call_id() { return "call_" + new_hex(24); }

// param_end: -1 none yet, -2 a candidate whose follower has not arrived (final accepts it)
static long param_end(const std::string & text, bool final = false) {
    size_t at = text.find(PARAM_END);
    while (at != std::string::npos) {
        const std::string after = py_strip(text.substr(at + PARAM_END.size()), true, false);
        if (starts(after, "<parameter=") || starts(after, FUNC_END)) return (long) at;
        if (after.empty() || prefix_of(after, "<parameter=") || prefix_of(after, FUNC_END)) return final ? (long) at : -2;
        at = text.find(PARAM_END, at + 1);
    }
    return -1;
}

static long call_end(const std::string & text) {
    std::string s = py_strip(text, true, false);
    size_t pos = text.size() - s.size();
    if (!starts(s, "<function=")) {
        if (prefix_of(s, "<function=")) return -1;
        const size_t f = text.find(CALL_END);
        return f == std::string::npos ? -1 : (long) f;
    }
    size_t gt = text.find('>', pos);
    if (gt == std::string::npos) return -1;
    pos = gt + 1;
    for (;;) {
        const std::string rest = text.substr(pos);
        s = py_strip(rest, true, false);
        pos += rest.size() - s.size();
        if (starts(s, "<parameter=")) {
            gt = text.find('>', pos);
            if (gt == std::string::npos) return -1;
            const long end = param_end(text.substr(gt + 1));
            if (end < 0) return -1;
            pos = gt + 1 + (size_t) end + PARAM_END.size();
        } else if (starts(s, FUNC_END)) {
            const std::string rest2 = text.substr(pos + FUNC_END.size());
            const std::string s2 = py_strip(rest2, true, false);
            if (starts(s2, CALL_END)) return (long) (pos + FUNC_END.size() + rest2.size() - s2.size());
            if (prefix_of(s2, CALL_END)) return -1;
            const size_t f = text.find(CALL_END, pos);
            return f == std::string::npos ? -1 : (long) f;
        } else if (s.empty() || prefix_of(s, "<parameter=") || prefix_of(s, FUNC_END)) {
            return -1;
        } else {
            const size_t f = text.find(CALL_END, pos);
            return f == std::string::npos ? -1 : (long) f;
        }
    }
}

static bool value_open(const std::string & body) {
    std::string s = py_strip(body, true, false);
    size_t pos = body.size() - s.size();
    if (!starts(s, FUNC_START)) return false;
    size_t gt = body.find('>', pos);
    if (gt == std::string::npos) return false;
    pos = gt + 1;
    for (;;) {
        const std::string rest = body.substr(pos);
        s = py_strip(rest, true, false);
        pos += rest.size() - s.size();
        if (!starts(s, PARAM_START)) return false;
        gt = body.find('>', pos);
        if (gt == std::string::npos) return false;
        const long end = param_end(body.substr(gt + 1));
        if (end == -1) return true;
        if (end < 0) return false;
        pos = gt + 1 + (size_t) end + PARAM_END.size();
    }
}

static long function_end(const std::string & text) {
    std::string s = py_strip(text, true, false);
    size_t pos = text.size() - s.size();
    if (!starts(s, FUNC_START)) return -1;
    size_t gt = text.find('>', pos);
    if (gt == std::string::npos) return -1;
    pos = gt + 1;
    for (;;) {
        const std::string rest = text.substr(pos);
        s = py_strip(rest, true, false);
        pos += rest.size() - s.size();
        if (starts(s, PARAM_START)) {
            gt = text.find('>', pos);
            if (gt == std::string::npos) return -1;
            const long end = param_end(text.substr(gt + 1));
            if (end < 0) return -1;
            pos = gt + 1 + (size_t) end + PARAM_END.size();
        } else if (starts(s, FUNC_END)) {
            return (long) (pos + FUNC_END.size());
        } else return -1;
    }
}

// opens_quote: the line ends with a quotation mark that opens a quote
static bool opens_quote(const std::string & line) {
    if (line.empty()) return false;
    size_t n = 0;
    const uint32_t c = utf8_last_cp(line, line.size(), &n);
    static const uint32_t OPENING[] = { 0x201c, 0x2018, 0x00ab, 0x201e, 0x300c, 0x300e };
    for (auto o : OPENING) if (c == o) return true;
    if (c == '"' || c == '\'') {
        if (line.size() == 1) return true;   // len(line) == 1 (one character)
        size_t n2 = 0;
        const uint32_t p = utf8_last_cp(line, line.size() - n, &n2);
        return p == ' ' || p == '\t' || p == '(' || p == '[' || p == '{' || p == '=' || p == ':' || p == ',';
    }
    return false;
}

static bool bool_allowed(const json * spec) {
    if (!spec || !spec->is_object()) return true;
    const json t = get(*spec, "type");
    if (t.is_null()) {
        std::vector<const json *> branches;
        for (const char * k : { "anyOf", "oneOf" }) {
            const json & o = *spec;
            if (o.contains(k) && truthy(o[k]) && o[k].is_array())
                for (const auto & b : o[k]) if (b.is_object()) branches.push_back(&b);
        }
        if (branches.empty()) return true;
        for (auto * b : branches) if (bool_allowed(b)) return true;
        return false;
    }
    if (t == "boolean") return true;
    if (t.is_array()) for (const auto & x : t) if (x == "boolean") return true;
    return false;
}

// raw_value: JSON, or Python-style True / False where the schema allows a boolean, else the text
static json raw_value(const std::string & value, const json * spec) {
    try {
        json v = json::parse(value);
        return v;
    } catch (const std::exception &) {}
    // Python's json.loads also takes NaN / Infinity / -Infinity
    const std::string t = py_strip(value);
    if (t == value || true) {
        if (value == "NaN") return std::nan("");
        if (value == "Infinity") return HUGE_VAL;
        if (value == "-Infinity") return -HUGE_VAL;
    }
    if ((t == "True" || t == "False") && bool_allowed(spec)) return t == "True";
    return value;
}

static const std::set<std::string> FORMAT_WORDS = { "tools", "tool", "function", "functions", "function_call", "function_calls",
    "tool_call", "tool_calls", "invoke", "invocation", "parameter", "parameters", "call" };

static std::string lower_ascii(std::string s) { for (auto & c : s) c = (char) std::tolower((unsigned char) c); return s; }

static std::optional<std::string> repair_call_name(const std::string & name, const std::vector<std::string> & order) {
    if (order.empty()) return std::nullopt;
    if (std::find(order.begin(), order.end(), name) != order.end()) return name;
    const std::string bare = py_strip(py_strip_chars(py_strip(name), "`'\""));
    if (FORMAT_WORDS.count(lower_ascii(bare))) return std::nullopt;
    for (const auto & k : order) if (lower_ascii(k) == lower_ascii(bare)) return k;
    auto flat = [](const std::string & s) { std::string o; for (char c : lower_ascii(s)) if (std::isalnum((unsigned char) c) && (unsigned char) c < 128) o += c; return o; };
    const std::string key = flat(name);
    if (key.empty()) return std::nullopt;
    std::vector<std::string> hits;
    for (const auto & k : order) if (flat(k) == key) hits.push_back(k);
    if (hits.size() == 1) return hits[0];
    for (const char * sep : { "/", ":", ".", "__", " " }) {
        const size_t at = bare.rfind(sep);
        if (at != std::string::npos) {
            const std::string head = bare.substr(0, at), tail = bare.substr(at + std::strlen(sep));
            if (FORMAT_WORDS.count(lower_ascii(head)) && !py_strip(tail).empty()) return repair_call_name(py_strip(tail), order);
        }
    }
    if (bare.size() >= 4) {   // len(bare) >= 4 in characters: bytes >= chars, so count characters
        size_t chars = 0;
        for (size_t i = 0; i < bare.size(); ++i) if (((unsigned char) bare[i] & 0xC0) != 0x80) ++chars;
        if (chars >= 4) {
            hits.clear();
            for (const auto & k : order) if (ends(lower_ascii(k), bare)) hits.push_back(k);
            if (hits.size() == 1) return hits[0];
        }
    }
    return std::nullopt;
}

std::shared_ptr<ToolCall> parse_tool_call(const std::string & body0, const json * schema) {
    const std::string body = py_strip(body0);
    if (!starts(body, "<function=") || body.find('>') == std::string::npos) throw std::invalid_argument("malformed tool call: " + body.substr(0, 80));
    const size_t gt = body.find('>');
    auto call = std::make_shared<ToolCall>();
    call->name = py_strip(body.substr(10, gt - 10));
    call->id = new_call_id();
    std::string rest = body.substr(gt + 1);
    json props = json::object();
    if (schema && schema->is_object()) {
        const json params = get(*schema, "parameters");
        if (truthy(params) && params.is_object()) { const json p = get(params, "properties"); if (truthy(p) && p.is_object()) props = p; }
    }
    while (rest.find("<parameter=") != std::string::npos) {
        rest = rest.substr(rest.find("<parameter=") + 11);
        const size_t g = rest.find('>');
        if (g == std::string::npos) throw std::invalid_argument("substring not found");   // rest.index(">") raised
        const std::string pname = rest.substr(0, g);
        rest = rest.substr(g + 1);
        const long end = param_end(rest, true);
        std::string value = end >= 0 ? rest.substr(0, (size_t) end) : rest;
        rest = end >= 0 ? rest.substr((size_t) end + PARAM_END.size()) : std::string();
        if (starts(value, "\n")) value = value.substr(1);
        if (ends(value, "\n")) value.pop_back();
        const json * spec = props.contains(pname) ? &props[pname] : nullptr;
        const json declared = spec && spec->is_object() ? get(*spec, "type") : json(nullptr);
        if (declared == "string") call->arguments[pname] = value;
        else call->arguments[pname] = raw_value(value, spec && truthy(*spec) ? spec : nullptr);
    }
    return call;
}

OutputParser::OutputParser(bool thinking, const json & tools, bool stream_tools_, bool repair_names_)
    : stream_tools(stream_tools_), repair_names(repair_names_) {
    state = thinking ? "reasoning" : "content";
    if (tools.is_array())
        for (const auto & t : tools) {
            const json n = get(t, "name");
            const std::string key = n.is_string() ? n.get<std::string>() : std::string("\x01null");
            if (!schemas.count(key)) schema_order.push_back(key);
            schemas[key] = t;
        }
    reset_scan();
}

const json * OutputParser::schema_of(const std::string & name) const {
    auto it = schemas.find(name);
    return it == schemas.end() ? nullptr : &it->second;
}

void OutputParser::reset_scan() {
    sp = 0; ss = "name"; scall.reset(); sfirst = true; sval_started = false; sspecs.clear(); sparam.clear();
}

std::string OutputParser::track(const std::string & text) {
    if (text.find('\n') == std::string::npos) { line += text; return text; }
    size_t p = 0;
    for (;;) {
        const size_t nl = text.find('\n', p);
        if (nl == std::string::npos) { line += text.substr(p); break; }
        const std::string ln = line + text.substr(p, nl - p);
        line.clear();
        const std::string s = py_strip(ln, true, false);
        if (!fence.empty()) {
            if (starts(s, fence + fence + fence)) fence.clear();
        } else if ((starts(s, "```") || starts(s, "~~~")) && s.substr(3).find(std::string(3, s[0])) == std::string::npos) {
            fence = std::string(1, s[0]);
        } else if (s.empty()) {
            ticks = 0;
        } else {
            ticks += (long long) std::count(ln.begin(), ln.end(), '`');
        }
        p = nl + 1;
    }
    return text;
}

bool OutputParser::opener_ok() const { return fence.empty() && py_strip(line).empty() && ticks % 2 == 0; }
bool OutputParser::in_code() const { return !fence.empty() || (ticks + (long long) std::count(line.begin(), line.end(), '`')) % 2 == 1; }

std::pair<bool, std::string> OutputParser::place() {
    const auto f0 = fence; const auto l0 = line; const auto t0 = ticks;
    track(buf);
    const std::string ln = line;
    const std::string st = py_strip(ln, true, false);
    const bool fenced = !fence.empty() && !(starts(st, "```") || starts(st, "~~~"));
    fence = f0; line = l0; ticks = t0;
    return { fenced, ln };
}

bool OutputParser::example_place() {
    if (state == "reasoning") return pending.empty();
    return state == "content" && place().first;
}

std::string OutputParser::literal_context(bool example) {
    if (state == "call" || state == "rcall") {
        const std::string body = state == "rcall" ? buf.substr(std::min(buf.size(), CALL_START.size())) : buf;
        return value_open(body) ? "in a tool call's argument" : "";
    }
    if ((state != "content" && state != "reasoning") || !pending.empty()) return "";
    auto [fenced, ln] = place();
    if (example && (fenced || state == "reasoning")) return "in a turn of the chat format written out as an example";
    const std::string st = py_strip(ln, true, false);
    if (!(starts(st, "```") || starts(st, "~~~")) && ends(ln, "`") && std::count(ln.begin(), ln.end(), '`') % 2 == 1)
        return "after the backtick that opens inline code";
    if (opens_quote(ln)) return "after an opening quotation mark";
    if (ends(ln, " ") || ends(ln, "\t")) return "after a space";
    return "";
}

std::optional<bool> OutputParser::quoted_think_end(size_t i) {
    char before;
    if (i) before = buf[i - 1];
    else before = line.empty() ? '\0' : line.back();
    // buf[i-1] / line[-1:] is one character; a quote is ASCII, so its last byte says it
    if (!(before == '"' || before == '\'' || before == '`')) return false;
    const std::string after = buf.substr(i + THINK_END.size());
    if (after.empty()) return std::nullopt;
    return !(after[0] == '\r' || after[0] == '\n');
}

std::vector<Event> OutputParser::release(bool deliver) {
    std::vector<Event> out;
    for (auto & p : pending) {
        if (p.call && deliver) { out.push_back(Event{ "tool_call", "", p.call }); ++rescued; }
        else {
            if (p.call) ++refused;
            if (!p.raw.empty()) out.push_back(Event{ "reasoning", track(p.raw), nullptr });
        }
    }
    pending.clear();
    return out;
}

static size_t hold_len(const std::string & text, const std::vector<std::string> & tags) {
    size_t best = 0;
    for (const auto & tag : tags)
        for (size_t n = 1; n < tag.size(); ++n)
            if (text.size() >= n && text.compare(text.size() - n, n, tag, 0, n) == 0) best = std::max(best, n);
    return best;
}

std::vector<Event> OutputParser::scan() {
    std::vector<Event> out;
    auto args = [&](const std::string & s) { if (!s.empty()) out.push_back(Event{ "tool_args", s, scall }); };
    for (;;) {
        std::string rest = buf.substr(std::min(sp, buf.size()));
        if (ss == "name") {
            const size_t a = rest.find("<function=");
            const size_t b = a != std::string::npos ? rest.find('>', a + 10) : std::string::npos;
            if (b == std::string::npos) return out;
            std::string name = rest.substr(a + 10, b - a - 10);
            if (repair_names && !schemas.empty() && !schemas.count(name)) {
                auto fixed = repair_call_name(name, schema_order);
                if (fixed) name = *fixed;
                else { ss = "suppressed"; sp += b + 1; return out; }
            }
            scall = std::make_shared<ToolCall>();
            scall->name = name;
            scall->id = new_call_id();
            sspecs.clear();
            if (const json * sc = schema_of(name)) {
                const json params = get(*sc, "parameters");
                if (truthy(params) && params.is_object()) {
                    const json props = get(params, "properties");
                    if (truthy(props) && props.is_object()) for (auto it = props.begin(); it != props.end(); ++it) sspecs[it.key()] = it.value();
                }
            }
            out.push_back(Event{ "tool_start", "", scall });
            args("{");
            sp += b + 1;
            ss = "between";
        } else if (ss == "between") {
            const std::string stripped = py_strip(rest, true, false);
            sp += rest.size() - stripped.size();
            if (starts(stripped, "<parameter=")) {
                const size_t b = stripped.find('>');
                if (b == std::string::npos) return out;
                const std::string pname = stripped.substr(11, b - 11);
                sparam = pname;
                args((sfirst ? "" : ",") + py_dumps_str(pname, true) + ":");
                sfirst = false;
                sp += b + 1;
                const json * spec = sspecs.count(pname) ? &sspecs[pname] : nullptr;
                const json declared = spec && truthy(*spec) && spec->is_object() ? get(*spec, "type") : json(nullptr);
                if (declared == "string") { args("\""); ss = "str"; sval_started = false; }
                else ss = "raw";
            } else if (starts(stripped, "</function>")) {
                args("}");
                sp += 11;
                ss = "done";
            } else return out;
        } else if (ss == "str") {
            if (!sval_started) {
                if (rest.empty()) return out;
                if (rest[0] == '\n') { sp += 1; rest = rest.substr(1); }
                sval_started = true;
            }
            const long end = param_end(rest);
            if (end >= 0) {
                std::string value = rest.substr(0, (size_t) end);
                if (ends(value, "\n")) value.pop_back();
                const std::string d = py_dumps_str(value, true);
                args(d.substr(1, d.size() - 2) + "\"");
                sp += (size_t) end + PARAM_END.size();
                ss = "between";
                continue;
            }
            size_t safe = end == -2 ? rest.find(PARAM_END) : rest.size() - hold_len(rest, { PARAM_END });
            if (safe > 0 && rest[safe - 1] == '\n') --safe;
            if (safe > 0) {
                const std::string d = py_dumps_str(rest.substr(0, safe), true);
                args(d.substr(1, d.size() - 2));
                sp += safe;
            }
            return out;
        } else if (ss == "raw") {
            const long end = param_end(rest);
            if (end < 0) return out;
            std::string value = rest.substr(0, (size_t) end);
            if (starts(value, "\n")) value = value.substr(1);
            if (ends(value, "\n")) value.pop_back();
            const json * spec = sspecs.count(sparam) ? &sspecs[sparam] : nullptr;
            args(py_dumps(raw_value(value, spec && truthy(*spec) ? spec : nullptr), false));
            sp += (size_t) end + PARAM_END.size();
            ss = "between";
        } else return out;
    }
}

std::vector<Event> OutputParser::close_scan() {
    std::vector<Event> out;
    if (!scall || ss == "done") return out;
    std::string tail;
    if (ss == "str") tail += "\"";
    else if (ss == "raw") tail += py_dumps_str(py_strip_chars(buf.substr(std::min(sp, buf.size())), "\n"), true);
    tail += "}";
    out.push_back(Event{ "tool_args", tail, scall });
    ss = "done";
    return out;
}

std::optional<std::pair<std::string, std::string>> OutputParser::repair_call(const std::string & name, const std::string & body) {
    auto fixed = repair_call_name(name, schema_order);
    if (fixed) return std::make_pair(*fixed, body);
    size_t at = body.find(FUNC_START, body.find('>') == std::string::npos ? 0 : body.find('>') + 1);
    while (at != std::string::npos) {
        const size_t gt = body.find('>', at);
        if (gt == std::string::npos) break;
        const std::string inner = py_strip(body.substr(at + FUNC_START.size(), gt - at - FUNC_START.size()));
        std::optional<std::string> f2 = schemas.count(inner) ? std::optional<std::string>(inner) : repair_call_name(inner, schema_order);
        if (f2) {
            const long end = function_end(body.substr(at));
            if (end >= 0) return std::make_pair(*f2, body.substr(at, (size_t) end));
        }
        at = body.find(FUNC_START, at + FUNC_START.size());
    }
    if (py_strip(name).empty() && schemas.size() == 1) return std::make_pair(schema_order[0], body);
    return std::nullopt;
}

std::vector<Event> OutputParser::finish_call(const std::string & body, bool closed) {
    std::string name;
    {
        const std::string b = py_strip(body);
        const std::string after = b.size() > FUNC_START.size() ? b.substr(FUNC_START.size()) : std::string();
        name = py_strip(after.substr(0, after.find('>')));
    }
    const bool was_bare = bare;
    bare = false;
    const std::string text = was_bare ? body : CALL_START + body + (closed ? CALL_END : "");
    std::string b2 = body;
    if (repair_names && !schemas.empty() && !schemas.count(name)) {
        auto fixed = repair_call(name, body);
        if (!fixed) { reset_scan(); state = "content"; return { Event{ "content", track(text), nullptr } }; }
        name = fixed->first;
        b2 = fixed->second;
    }
    std::shared_ptr<ToolCall> call;
    try { call = parse_tool_call(b2, schema_of(name)); }
    catch (const std::invalid_argument &) {
        // not recover (the config default): server.py's ValueError ends the request; here as there it is raised
        throw;
    }
    call->name = name;
    if (scall) call->id = scall->id;
    reset_scan();
    state = "content";
    lead = true;
    return { Event{ "tool_call", "", call } };
}

std::vector<Event> OutputParser::feed(const std::string & delta) {
    buf += delta;
    std::vector<Event> out;
    auto emit = [&](const char * kind, const std::string & t) { out.push_back(Event{ kind, track(t), nullptr }); };
    for (;;) {
        if (state == "rcall") {
            const std::string body = buf.substr(CALL_START.size());
            const long end = call_end(body);
            const size_t think = body.find(THINK_END);
            if (end >= 0 && (think == std::string::npos || (long) think >= end)) {
                std::shared_ptr<ToolCall> call;
                try {
                    const std::string head = py_strip(body.substr(0, (size_t) end));
                    std::string name = head.size() >= 10 ? head.substr(10) : std::string();
                    name = py_strip(name.substr(0, name.find('>')));
                    if (schemas.count(name)) call = parse_tool_call(body.substr(0, (size_t) end), schema_of(name));
                    else if (repair_names) {
                        auto fixed = repair_call(name, body.substr(0, (size_t) end));
                        if (fixed) { call = parse_tool_call(fixed->second, schema_of(fixed->first)); call->name = fixed->first; }
                    }
                } catch (const std::invalid_argument &) {}
                const std::string raw = buf.substr(0, CALL_START.size() + (size_t) end + CALL_END.size());
                if (call || !pending.empty()) pending.push_back(Pending{ raw, call });
                else emit("reasoning", raw);
                buf = body.substr(std::min(body.size(), (size_t) end + CALL_END.size()));
                state = "reasoning";
            } else if (think != std::string::npos) {
                auto r = release(false);
                out.insert(out.end(), r.begin(), r.end());
                emit("reasoning", buf.substr(0, CALL_START.size() + think));
                buf = body.substr(think);
                state = "reasoning";
            } else if (body.size() > RCALL_MAX) {
                auto r = release(false);
                out.insert(out.end(), r.begin(), r.end());
                emit("reasoning", buf);
                buf.clear();
                state = "reasoning";
            } else return out;
        } else if (state == "reasoning") {
            if (!pending.empty()) {
                const std::string stripped = py_strip(buf, true, false);
                if (stripped.size() < buf.size()) { pending.back().raw += buf.substr(0, buf.size() - stripped.size()); buf = stripped; }
                if (buf.empty()) return out;
                if (starts(buf, CALL_START)) state = "rcall";
                else if (starts(buf, THINK_END)) { auto r = release(true); out.insert(out.end(), r.begin(), r.end()); }
                else if (prefix_of(buf, CALL_START) || prefix_of(buf, THINK_END)) return out;
                else { auto r = release(false); out.insert(out.end(), r.begin(), r.end()); }
                continue;
            }
            const size_t i = buf.find(THINK_END);
            const size_t tool = !schemas.empty() ? buf.find(CALL_START) : std::string::npos;
            if (tool != std::string::npos && (i == std::string::npos || tool < i)) {
                if (tool) emit("reasoning", buf.substr(0, tool));
                if (opener_ok()) { buf = buf.substr(tool); state = "rcall"; }
                else { emit("reasoning", CALL_START); buf = buf.substr(tool + CALL_START.size()); }
                continue;
            }
            if (i == std::string::npos) {
                const size_t keep = schemas.empty() ? hold_len(buf, { THINK_END }) : hold_len(buf, { THINK_END, CALL_START });
                if (buf.size() > keep) { emit("reasoning", buf.substr(0, buf.size() - keep)); buf = buf.substr(buf.size() - keep); }
                return out;
            }
            const auto quoted = quoted_think_end(i);
            if (!quoted.has_value()) {
                if (i) { emit("reasoning", buf.substr(0, i)); buf = buf.substr(i); }
                return out;
            }
            if (*quoted) { emit("reasoning", buf.substr(0, i + THINK_END.size())); buf = buf.substr(i + THINK_END.size()); continue; }
            if (i) emit("reasoning", buf.substr(0, i));
            buf = buf.substr(i + THINK_END.size());
            state = "content";
            lead = true;
            fence.clear(); line.clear(); ticks = 0;
        } else if (state == "content") {
            if (lead) {
                const size_t k = buf.find_first_not_of('\n');
                if (k == std::string::npos) { buf.clear(); return out; }
                buf = buf.substr(k);
                lead = false;
            }
            const size_t i = buf.find(CALL_START);
            if (i == std::string::npos) {
                size_t j = buf.size() - hold_len(buf, { CALL_START });
                while (j > 0 && buf[j - 1] == '\n') --j;
                if (j > 0) { emit("content", buf.substr(0, j)); buf = buf.substr(j); }
                return out;
            }
            {
                const auto f0 = fence; const auto l0 = line; const auto t0 = ticks;
                track(buf.substr(0, i));
                const bool code = in_code();
                fence = f0; line = l0; ticks = t0;
                if (code) { emit("content", buf.substr(0, i + CALL_START.size())); buf = buf.substr(i + CALL_START.size()); continue; }
            }
            const std::string after = py_strip(buf.substr(i + CALL_START.size()), true, false);
            if (!after.empty() && !starts(after, FUNC_START) && !prefix_of(after, FUNC_START)) {
                emit("content", buf.substr(0, i + CALL_START.size()));
                buf = buf.substr(i + CALL_START.size());
                continue;
            }
            if (!starts(after, FUNC_START)) {
                size_t j = i;
                while (j > 0 && buf[j - 1] == '\n') --j;
                if (j > 0) { emit("content", buf.substr(0, j)); buf = buf.substr(j); }
                return out;
            }
            if (i && !py_strip(buf.substr(0, i)).empty()) {
                std::string pre = buf.substr(0, i);
                while (!pre.empty() && pre.back() == '\n') pre.pop_back();
                emit("content", pre);
            }
            buf = buf.substr(i + CALL_START.size());
            state = "call";
        } else {   // "call" (recover's "json" and bare calls are not ported: tool_call_recovery is off)
            long i;
            long drop = -100;   // None
            i = call_end(buf);
            if (i >= 0) drop = (long) CALL_END.size();
            const long f = function_end(buf);
            const std::string nxt = f >= 0 ? py_strip(buf.substr((size_t) f), true, false) : std::string();
            if (!nxt.empty()) {
                bool piece = false;
                for (const std::string & t : { CALL_END, FUNC_START, PARAM_START }) if (starts(t, nxt) || starts(nxt, t)) piece = true;
                if (!piece) { i = f; drop = 0; }
            }
            if (stream_tools) {
                if (i >= 0) {
                    const std::string whole = buf;
                    buf = buf.substr(0, (size_t) i);
                    auto a = scan(); out.insert(out.end(), a.begin(), a.end());
                    auto b = close_scan(); out.insert(out.end(), b.begin(), b.end());
                    buf = whole;
                } else { auto a = scan(); out.insert(out.end(), a.begin(), a.end()); }
            }
            if (i < 0 || drop == -100) return out;
            const std::string body = buf.substr(0, (size_t) i);
            buf = buf.substr((size_t) i + (size_t) drop);
            const bool was_batch = batch;
            batch = false;
            auto ev = finish_call(body, drop != 0 || was_batch);
            out.insert(out.end(), ev.begin(), ev.end());
            if (was_batch && out.back().kind == "tool_call") { state = "call"; lead = false; }
        }
    }
}

std::vector<Event> OutputParser::finish(const std::string & reason) {
    std::vector<Event> out;
    if (!pending.empty()) {
        auto r = release((reason.empty() || reason == "stop") && state == "reasoning" && buf.empty());
        out.insert(out.end(), r.begin(), r.end());
    }
    if (state == "call" && stream_tools && scall) {
        auto a = scan(); out.insert(out.end(), a.begin(), a.end());
        if (ss == "done") out.push_back(Event{ "tool_call", "", scall });
        buf.clear();
        reset_scan();
        return out;
    }
    if (state == "call" && ss == "suppressed") {
        const std::string head = py_strip(buf);
        std::string name = head;
        if (starts(head, FUNC_START)) { name = head.substr(FUNC_START.size()); name = py_strip(name.substr(0, name.find('>'))); }
        auto fixed = repair_call(name, buf);
        if (fixed) {
            try {
                auto call = parse_tool_call(fixed->second, schema_of(fixed->first));
                call->name = fixed->first;
                buf.clear(); state = "content"; lead = true;
                reset_scan();
                out.push_back(Event{ "tool_call", "", call });
                return out;
            } catch (const std::invalid_argument &) {}
        }
    }
    if (state == "reasoning" && buf == THINK_END) { buf.clear(); state = "content"; }
    if (!buf.empty()) {
        const std::string kind = state == "reasoning" || state == "rcall" ? "reasoning" : "content";
        const std::string text = (state == "call" || state == "json") && !bare ? CALL_START + buf : buf;
        out.push_back(Event{ kind, text, nullptr });
        buf.clear();
    }
    return out;
}

// ================================================================================================ stop strings
std::string StopMatcher::push(const std::string & text) {
    if (hit) return "";
    const std::string b = held + text;
    long first_at = -1;
    std::string first_s;
    for (const auto & s : stops) {
        const size_t i = b.find(s);
        if (i != std::string::npos && (first_at < 0 || (long) i < first_at)) { first_at = (long) i; first_s = s; }
    }
    if (first_at >= 0) { held.clear(); hit = first_s; return b.substr(0, (size_t) first_at); }
    // the longest end of b that a stop string starts with (in characters: a stop string's prefix ends on a whole one)
    size_t keep = 0;
    for (const auto & s : stops) {
        for (size_t k = std::min(s.size() - 1, b.size()); k > keep; --k) {
            if ((k < s.size() && ((unsigned char) s[k] & 0xC0) == 0x80)) continue;   // not a character boundary of s
            if (b.compare(b.size() - k, k, s, 0, k) == 0) { keep = k; break; }
        }
    }
    held = keep ? b.substr(b.size() - keep) : std::string();
    return b.substr(0, b.size() - keep);
}

// ================================================================================================ structured output
static bool only_objects(const json & node, const json & root, std::vector<std::string> refs) {
    if (!node.is_object()) return false;
    if (get(node, "type") == "object") return true;
    const json ref = get(node, "$ref");
    if (ref.is_string() && starts(ref.get<std::string>(), "#") && std::find(refs.begin(), refs.end(), ref.get<std::string>()) == refs.end()) {
        const json * target = &root;
        const std::string r = ref.get<std::string>().substr(1);
        std::vector<std::string> parts;
        size_t p = 0;
        while (p <= r.size()) { const size_t q = r.find('/', p); parts.push_back(r.substr(p, q == std::string::npos ? std::string::npos : q - p)); if (q == std::string::npos) break; p = q + 1; }
        for (size_t k = 1; k < parts.size(); ++k) {
            std::string part = parts[k];
            for (size_t x; (x = part.find("~1")) != std::string::npos;) part.replace(x, 2, "/");
            for (size_t x; (x = part.find("~0")) != std::string::npos;) part.replace(x, 2, "~");
            if (!target->is_object() || !target->contains(part)) return false;
            target = &(*target)[part];
        }
        refs.push_back(ref.get<std::string>());
        return only_objects(*target, root, refs);
    }
    for (const char * key : { "anyOf", "oneOf" }) {
        const json br = get(node, key);
        if (br.is_array() && !br.empty()) {
            bool all = true;
            for (const auto & b : br) if (!only_objects(b, root, refs)) { all = false; break; }
            if (all) return true;
        }
    }
    const json br = get(node, "allOf");
    if (br.is_array()) for (const auto & b : br) if (only_objects(b, root, refs)) return true;
    return false;
}

static void check_refs(const json & node) {
    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            if ((it.key() == "$ref" || it.key() == "$dynamicRef") && it.value().is_string() && !starts(it.value().get<std::string>(), "#"))
                throw BadRequest("response_format supports only local schema references (#...)");
            check_refs(it.value());
        }
    } else if (node.is_array()) for (const auto & v : node) check_refs(v);
}

bool prepare_format(const json & rf, json & messages) {
    if (rf.is_null()) return false;
    if (!rf.is_object()) throw BadRequest("response_format must be an object");
    const json kind = get(rf, "type");
    json schema;
    if (kind == "text") return false;
    if (kind == "json_object") schema = json{ { "type", "object" } };
    else if (kind == "json_schema") {
        const json spec = get(rf, "json_schema");
        if (!spec.is_object() || !get(spec, "schema").is_object()) throw BadRequest("response_format.json_schema needs a schema object");
        if (!get(spec, "name").is_string() || spec["name"].get<std::string>().empty()) throw BadRequest("response_format.json_schema needs a name");
        if (spec.contains("strict") && !spec["strict"].is_boolean()) throw BadRequest("response_format.json_schema.strict must be boolean");
        schema = spec["schema"];
        if (!only_objects(schema, schema, {}))
            throw BadRequest("response_format schema must accept only JSON objects at its root (type object, or anyOf/oneOf of object schemas)");
    } else throw BadRequest("response_format.type must be text, json_object or json_schema");
    check_refs(schema);
    const std::string directive =
        "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
        "No Markdown, headings, code fences, commentary, or text outside the JSON. "
        "Use every required field, correct types, and only allowed fields. "
        "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" + py_dumps(schema, false, ",", ":");
    if (!messages.empty() && get(messages[0], "role") == "system") {
        json content = get(messages[0], "content");
        if (content.is_array()) { content.push_back(json{ { "type", "text" }, { "text", directive } }); messages[0]["content"] = content; }
        else messages[0]["content"] = (truthy(content) ? content.get<std::string>() : std::string()) + "\n\n" + directive;
    } else {
        messages.insert(messages.begin(), json{ { "role", "system" }, { "content", directive } });
    }
    return true;
}

static std::string extract_json(const std::string & text) {
    std::string s = py_strip(text);
    if (s.empty()) return "";
    if (starts(s, "```")) {
        const size_t first = s.find('\n');
        if (first != std::string::npos) {
            const size_t end = s.find("```", first + 1);
            if (end != std::string::npos) s = py_strip(s.substr(first + 1, end - first - 1));
        }
    }
    const size_t start = s.find('{');
    if (start == std::string::npos) return s;
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = start; i < s.size(); ++i) {
        const char ch = s[i];
        if (in_str) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') in_str = false; }
        else if (ch == '"') in_str = true;
        else if (ch == '{') ++depth;
        else if (ch == '}') { if (--depth == 0) return s.substr(start, i - start + 1); }
    }
    return s.substr(start);
}

std::string validated_json(const std::string & text, const std::string & finish) {
    if (finish != "stop") throw StructuredError("structured output was incomplete (finish_reason=" + finish + "); increase the output budget");
    json value;
    try {
        std::vector<std::set<std::string>> seen;
        std::string dup;
        json::parser_callback_t cb = [&](int /*depth*/, json::parse_event_t ev, json & parsed) {
            if (ev == json::parse_event_t::object_start) seen.emplace_back();
            else if (ev == json::parse_event_t::object_end) { if (!seen.empty()) seen.pop_back(); }
            else if (ev == json::parse_event_t::key && !seen.empty()) {
                const std::string k = parsed.get<std::string>();
                if (!seen.back().insert(k).second && dup.empty()) dup = k;
            }
            return true;
        };
        value = json::parse(extract_json(text), cb);
        if (!dup.empty()) throw StructuredError("model did not return valid JSON: duplicate JSON key: " + dup);
    } catch (const StructuredError &) { throw; }
    catch (const std::exception & e) { throw StructuredError(std::string("model did not return valid JSON: ") + e.what()); }
    if (!value.is_object()) throw StructuredError("model output failed the JSON Schema at /: the answer is not a JSON object");
    return py_dumps(value, false, ",", ":");
}

}  // namespace fxf
