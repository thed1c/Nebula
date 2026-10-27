# impulse-fx only (src/fx/CMakeLists.txt): llama.cpp's Jinja engine with Python's text rules where a chat template's output
# shows them, so impulse-fx renders a template exactly as serve/server.py's jinja2 does (One Engine P2, prompt parity):
#   - `tojson` of a float: Python's repr (the shortest digits that read back, "1.0", "1e-05"), not ostream's 6 digits;
#   - `trim` / `strip()` without arguments: Python's str.strip(), Unicode whitespace (NBSP, U+3000, U+2028, ...), not
#     the C locale's six ASCII bytes.
# Private source copies, as lfm-json-tools.cmake: the shared llama.cpp checkout and impulse-server's build are untouched.
set(_fxj_dir "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks/jinja")
file(MAKE_DIRECTORY "${_fxj_dir}")

# value.cpp: the float in value_to_json
set(_fxj_value_up "${LLAMA_DIR}/common/jinja/value.cpp")
file(READ "${_fxj_value_up}" _fxj_value)
set(_fxj_anchor "        oss << val->as_float();")
string(FIND "${_fxj_value}" "${_fxj_anchor}" _fxj_at)
if(_fxj_at EQUAL -1)
  message(FATAL_ERROR "fx-jinja-python hook: value.cpp does not match the pinned llama.cpp source")
endif()
string(REPLACE "${_fxj_anchor}" "        oss << impulse_fx_py_float(val->as_float());" _fxj_value "${_fxj_value}")
set(_fxj_head [=[#include <charconv>
#include <cmath>
#include <cstdio>
#include <string>
// impulse-fx: Python's repr(float) (llama-hooks/fx-jinja-python.cmake)
static std::string impulse_fx_py_float(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    bool neg = false;
    if (!s.empty() && s[0] == '-') { neg = true; s.erase(0, 1); }
    const size_t e = s.find('e');
    const int exp10 = std::atoi(s.c_str() + e + 1);
    std::string digits;
    for (size_t i = 0; i < e; ++i) if (s[i] != '.') digits += s[i];
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1;
    std::string out;
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) out = "0." + std::string((size_t) -decpt, '0') + digits;
        else if ((size_t) decpt >= digits.size()) out = digits + std::string((size_t) decpt - digits.size(), '0') + ".0";
        else out = digits.substr(0, (size_t) decpt) + "." + digits.substr((size_t) decpt);
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
]=])
# the copy lives outside common/jinja: its quoted includes name the folder
string(REPLACE "#include \"runtime.h\"" "#include \"jinja/runtime.h\"" _fxj_value "${_fxj_value}")
string(REPLACE "#include \"value.h\"" "#include \"jinja/value.h\"" _fxj_value "${_fxj_value}")
file(WRITE "${_fxj_dir}/value.cpp" "${_fxj_head}${_fxj_value}")

# string.cpp: strip's default character set
set(_fxj_string_up "${LLAMA_DIR}/common/jinja/string.cpp")
file(READ "${_fxj_string_up}" _fxj_string)
set(_fxj_anchor2 [=[        auto match_char = [&chars](unsigned char c) -> bool {
            return chars ? (*chars).find(c) != std::string::npos : isspace(c);
        };
        if (left) {
            while (start < end && match_char(static_cast<unsigned char>(s[start]))) {
                ++start;
            }
        }
        if (right) {
            while (end > start && match_char(static_cast<unsigned char>(s[end - 1]))) {
                --end;
            }
        }]=])
string(FIND "${_fxj_string}" "${_fxj_anchor2}" _fxj_at2)
if(_fxj_at2 EQUAL -1)
  message(FATAL_ERROR "fx-jinja-python hook: string.cpp does not match the pinned llama.cpp source")
endif()
set(_fxj_repl2 [=[        auto match_char = [&chars](unsigned char c) -> bool {
            return (*chars).find(c) != std::string::npos;
        };
        // impulse-fx: no characters given -> Python's str.strip(): Unicode whitespace, by code point
        auto py_space = [](uint32_t cp) {
            return (cp >= 0x9 && cp <= 0xd) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
                   (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
        };
        auto cp_at = [&s](size_t i, size_t & n) -> uint32_t {   // the code point at byte i (n: its bytes; 1 for a stray byte)
            const unsigned char c = (unsigned char) s[i];
            size_t k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            if (i + k > s.size()) k = 1;
            uint32_t cp = k == 1 ? c : k == 2 ? (c & 0x1F) : k == 3 ? (c & 0x0F) : (c & 0x07);
            for (size_t j = 1; j < k; ++j) {
                if (((unsigned char) s[i + j] & 0xC0) != 0x80) { n = 1; return c; }
                cp = (cp << 6) | ((unsigned char) s[i + j] & 0x3F);
            }
            n = k;
            return cp;
        };
        if (!chars) {
            if (left) {
                while (start < end) { size_t n = 1; if (!py_space(cp_at(start, n))) break; start += n; }
            }
            if (right) {
                while (end > start) {
                    size_t b = end - 1;
                    while (b > start && ((unsigned char) s[b] & 0xC0) == 0x80 && end - b < 4) --b;
                    size_t n = 1;
                    const uint32_t cp = cp_at(b, n);
                    if (b + n != end) { if (!py_space((unsigned char) s[end - 1])) break; --end; continue; }
                    if (!py_space(cp)) break;
                    end = b;
                }
            }
            return s.substr(start, end - start);
        }
        if (left) {
            while (start < end && match_char(static_cast<unsigned char>(s[start]))) {
                ++start;
            }
        }
        if (right) {
            while (end > start && match_char(static_cast<unsigned char>(s[end - 1]))) {
                --end;
            }
        }]=])
string(REPLACE "${_fxj_anchor2}" "${_fxj_repl2}" _fxj_string "${_fxj_string}")
file(WRITE "${_fxj_dir}/string.cpp" "#include <cstdint>\n${_fxj_string}")

get_target_property(_fxj_sources llama-common SOURCES)
list(REMOVE_ITEM _fxj_sources "${_fxj_value_up}" "${_fxj_string_up}" "jinja/value.cpp" "jinja/string.cpp")
list(APPEND _fxj_sources "${_fxj_dir}/value.cpp" "${_fxj_dir}/string.cpp")
set_property(TARGET llama-common PROPERTY SOURCES "${_fxj_sources}")
target_include_directories(llama-common PRIVATE "${LLAMA_DIR}/common")
