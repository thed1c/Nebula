// src/fx/fx_vision.cpp - see fx_vision.hpp.
#include "fx_vision.hpp"
#include "fx_front.hpp"

extern "C" {
#include "sha256/sha256.h"
}

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

#ifndef S_ISREG
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif

namespace fxv {

static const char * IMAGE_UNREADABLE = "the image could not be read: an image must be a data: URL, a public http(s) URL or a readable local file path";

std::string sha256_hex(const std::string & data) {
    unsigned char d[32];
    sha256_hash(d, (const unsigned char *) data.data(), data.size());
    static const char * hex = "0123456789abcdef";
    std::string s;
    for (unsigned char c : d) { s += hex[c >> 4]; s += hex[c & 15]; }
    return s;
}

// base64.b64decode(s) (validate=False): characters outside the alphabet are skipped; a short last group needs its padding
static std::string b64decode(const std::string & in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    uint32_t acc = 0;
    int bits = 0, n = 0, pads = 0;
    for (char c : in) {
        if (c == '=') {
            if (n % 4 >= 2) { ++pads; if ((n + pads) % 4 == 0) break; }
            continue;
        }
        const int v = val(c);
        if (v < 0) continue;
        if (pads) continue;
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        ++n;
        if (bits >= 8) { bits -= 8; out += (char) ((acc >> bits) & 0xFF); }
    }
    if (n % 4 == 1) throw fxf::BadRequest("the image's data: URL could not be read (Invalid base64-encoded string: number of data characters (" +
                                          std::to_string(n) + ") cannot be 1 more than a multiple of 4)");
    if (n % 4 != 0 && (n + pads) % 4 != 0) throw fxf::BadRequest("the image's data: URL could not be read (Incorrect padding)");
    return out;
}

std::string Vision::load(const std::string & source) {
    if (source.rfind("data:", 0) == 0) {
        const size_t comma = source.find(',');
        if (comma == std::string::npos) throw fxf::BadRequest("the image's data: URL could not be read (list index out of range)");
        return b64decode(source.substr(comma + 1));
    }
    if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0)
        throw fxf::BadRequest("image URLs are not read by this server (impulse-fx): send the picture as a data: URL");
    std::string path = source.rfind("file://", 0) == 0 ? source.substr(7) : source;
    std::string host;
    if (source.rfind("file://", 0) == 0) { const size_t s = path.find_first_of("/\\"); host = path.substr(0, s); }
    auto is_drive = [](const std::string & h) { return (h.size() == 2 && std::isalpha((unsigned char) h[0]) && h[1] == ':') || h == "localhost" || h == "LOCALHOST"; };
    std::string p2 = path;
    for (auto & c : p2) if (c == '/') c = '\\';
    if (p2.rfind("\\\\", 0) == 0 || p2.rfind("\\??\\", 0) == 0 || (!host.empty() && !is_drive(host)))
        throw fxf::BadRequest("an image file must be on this computer: network paths (\\\\host\\share, //host/share, file://host/...) are not read");
    struct stat st;
    if (!path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        std::ifstream f(path, std::ios::binary);
        if (f) { std::ostringstream ss; ss << f.rdbuf(); return ss.str(); }
    }
    throw fxf::BadRequest(IMAGE_UNREADABLE);
}

// Pillow's getexif().get(0x0112, 1) for a JPEG: the first APP1 "Exif" segment's IFD0 Orientation; anything unreadable is 1
bool Vision::exif_rotated(const std::string & d) {
    size_t i = 2;
    while (i + 4 <= d.size() && (unsigned char) d[i] == 0xFF) {
        const unsigned char m = (unsigned char) d[i + 1];
        if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) { i += 2; continue; }
        if (m == 0xDA || m == 0xD9) break;
        const size_t len = ((unsigned char) d[i + 2] << 8) | (unsigned char) d[i + 3];
        if (len < 2 || i + 2 + len > d.size()) break;
        if (m == 0xE1 && len >= 8 && d.compare(i + 4, 6, std::string("Exif\0\0", 6)) == 0) {
            const std::string t = d.substr(i + 10, len - 8);
            if (t.size() < 8) return false;
            const bool le = t[0] == 'I' && t[1] == 'I';
            if (!le && !(t[0] == 'M' && t[1] == 'M')) return false;
            auto u16 = [&](size_t o) -> uint32_t { if (o + 2 > t.size()) throw 0; const unsigned char a = t[o], b = t[o + 1]; return le ? (a | (b << 8)) : ((a << 8) | b); };
            auto u32 = [&](size_t o) -> uint32_t { if (o + 4 > t.size()) throw 0; return le ? (u16(o) | (u16(o + 2) << 16)) : ((u16(o) << 16) | u16(o + 2)); };
            try {
                const uint32_t ifd = u32(4);
                const uint32_t n = u16(ifd);
                for (uint32_t k = 0; k < n; ++k) {
                    const size_t e = ifd + 2 + 12 * k;
                    if (u16(e) == 0x0112) {
                        const uint32_t type = u16(e + 2);
                        const uint32_t v = type == 3 ? u16(e + 8) : type == 4 ? u32(e + 8) : 1;
                        return !(v == 0 || v == 1);
                    }
                }
            } catch (...) { return false; }
            return false;
        }
        i += 2 + len;
    }
    return false;
}

// serve/server.py Vision.normalize, run by Pillow itself (the same code, so the same PNG bytes)
static const char * NORMALIZE_PY = R"PY(
import io, sys
data = sys.stdin.buffer.read()
try:
    from PIL import Image, ImageOps
except ImportError:
    sys.stderr.write("NOPIL"); sys.exit(3)
try:
    im = Image.open(io.BytesIO(data))
    im.load()
    im = ImageOps.exif_transpose(im)
except Exception as e:
    sys.stderr.write(f"the image could not be read ({e})"); sys.exit(4)
if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
    im = im.convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    im = bg
elif im.mode != "RGB":
    im = im.convert("RGB")
out = io.BytesIO()
im.save(out, format="PNG")
sys.stdout.buffer.write(out.getvalue())
)PY";

#if !defined(_WIN32)
// run argv with `input` on stdin; -> exit code, stdout, stderr
static int run_capture(const std::vector<std::string> & argv, const std::string & input, std::string & out, std::string & err) {
    int pin[2], pout[2], perr[2];
    if (pipe(pin) || pipe(pout) || pipe(perr)) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pin[0], 0);
    posix_spawn_file_actions_adddup2(&fa, pout[1], 1);
    posix_spawn_file_actions_adddup2(&fa, perr[1], 2);
    for (int fd : { pin[0], pin[1], pout[0], pout[1], perr[0], perr[1] }) posix_spawn_file_actions_addclose(&fa, fd);
    std::vector<char *> av;
    for (auto & a : argv) av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    pid_t pid;
    const int rc = posix_spawnp(&pid, av[0], &fa, nullptr, av.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pin[0]); close(pout[1]); close(perr[1]);
    if (rc != 0) { close(pin[1]); close(pout[0]); close(perr[0]); return -1; }
    size_t off = 0;
    bool in_open = true;
    struct pollfd fds[3];
    while (true) {
        int nf = 0;
        if (in_open) fds[nf++] = { pin[1], POLLOUT, 0 };
        fds[nf++] = { pout[0], POLLIN, 0 };
        fds[nf++] = { perr[0], POLLIN, 0 };
        if (poll(fds, (nfds_t) nf, 60000) <= 0) break;
        bool open_any = false;
        for (int k = 0; k < nf; ++k) {
            if (fds[k].fd == pin[1] && (fds[k].revents & (POLLOUT | POLLERR | POLLHUP))) {
                const ssize_t w = off < input.size() ? write(pin[1], input.data() + off, std::min<size_t>(65536, input.size() - off)) : 0;
                if (w > 0) off += (size_t) w;
                if (w <= 0 || off >= input.size()) { close(pin[1]); in_open = false; }
            } else if (fds[k].revents & (POLLIN | POLLHUP | POLLERR)) {
                char b[65536];
                const ssize_t r = read(fds[k].fd, b, sizeof b);
                if (r > 0) { (fds[k].fd == pout[0] ? out : err).append(b, (size_t) r); open_any = true; }
                else fds[k].fd = -fds[k].fd - 1;
            }
        }
        if (!open_any && !in_open) {
            // drain what is left, then stop
            char b[65536];
            ssize_t r;
            while ((r = read(pout[0], b, sizeof b)) > 0) out.append(b, (size_t) r);
            while ((r = read(perr[0], b, sizeof b)) > 0) err.append(b, (size_t) r);
            break;
        }
    }
    if (in_open) close(pin[1]);
    close(pout[0]); close(perr[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

std::string Vision::normalize(const std::string & data) const {
    const bool jpeg = data.size() >= 3 && (unsigned char) data[0] == 0xFF && (unsigned char) data[1] == 0xD8 && (unsigned char) data[2] == 0xFF;
    if ((jpeg && !exif_rotated(data)) || data.compare(0, 8, "\x89PNG\r\n\x1a\n") == 0 || data.compare(0, 2, "BM") == 0 ||
        data.compare(0, 6, "GIF87a") == 0 || data.compare(0, 6, "GIF89a") == 0)
        return data;
#if !defined(_WIN32)
    if (!cfg_.python.empty()) {
        std::string out, err;
        const int rc = run_capture({ cfg_.python, "-c", NORMALIZE_PY }, data, out, err);
        if (rc == 0 && !out.empty()) return out;
        if (rc == 4) throw fxf::BadRequest(err);
        if (rc != 3) throw fxf::BadRequest("the image could not be read (the converter failed: " + err.substr(0, 300) + ")");
    }
#endif
    throw fxf::BadRequest("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and GIF work without it");
}

Vision::Vision(const Config & c) : cfg_(c) {
#if !defined(_WIN32)
    // work_dir(): a temporary directory whose path has no space (#480)
    const char * tmp = getenv("TMPDIR");
    std::string base = tmp && *tmp ? tmp : "/tmp";
    std::string tmpl = base + "/strata-vision-XXXXXX";
    std::vector<char> b(tmpl.begin(), tmpl.end());
    b.push_back('\0');
    if (mkdtemp(b.data())) dir_ = b.data();
#endif
}

Vision::~Vision() { kill_child(); }

void Vision::kill_child() {
#if !defined(_WIN32)
    if (pid_ > 0) {
        if (in_ >= 0) { const char q[] = "QUIT\n"; (void) !write(in_, q, sizeof q - 1); }
        for (int i = 0; i < 100; ++i) { int st; if (waitpid(pid_, &st, WNOHANG) == pid_) { pid_ = -1; break; } usleep(100000); }
        if (pid_ > 0) { kill(pid_, SIGKILL); int st; waitpid(pid_, &st, 0); }
    }
    if (in_ >= 0) close(in_);
    if (out_ >= 0) close(out_);
#endif
    pid_ = in_ = out_ = -1;
    stopped_ = true;
    rbuf_.clear();
}

bool Vision::alive() const {
#if !defined(_WIN32)
    if (pid_ <= 0 || stopped_) return false;
    int st;
    return waitpid(pid_, &st, WNOHANG) == 0;
#else
    return false;
#endif
}

std::string Vision::readline(double timeout_s, const std::string & what) {
#if !defined(_WIN32)
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    for (;;) {
        const size_t nl = rbuf_.find('\n');
        if (nl != std::string::npos) { std::string l = rbuf_.substr(0, nl + 1); rbuf_.erase(0, nl + 1); return l; }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            if (pid_ > 0) kill(pid_, SIGKILL);
            throw std::runtime_error(what + ": it said nothing for " + std::to_string((int) timeout_s) + " s");
        }
        struct pollfd p = { out_, POLLIN, 0 };
        const int r = poll(&p, 1, (int) std::min<long long>(left, 1000));
        if (r > 0) {
            char b[4096];
            const ssize_t n = read(out_, b, sizeof b);
            if (n <= 0) { std::string l = rbuf_; rbuf_.clear(); return l; }   // EOF: what there is ("" = it ended)
            rbuf_.append(b, (size_t) n);
        }
    }
#else
    (void) timeout_s; (void) what;
    return "";
#endif
}

std::string Vision::start() {
#if !defined(_WIN32)
    std::lock_guard<std::mutex> g(mu_);
    if (dir_.empty()) return "no directory for the image encoder";
    std::vector<std::string> args = { cfg_.exe, "--mmproj", cfg_.mmproj, "--model", cfg_.model };
    if (cfg_.gpu) args.push_back("--gpu");
    if (cfg_.threads) { args.push_back("--threads"); args.push_back(std::to_string(cfg_.threads)); }
    if (cfg_.max_tokens) { args.push_back("--max-tokens"); args.push_back(std::to_string(cfg_.max_tokens)); }
    if (cfg_.min_tokens) { args.push_back("--min-tokens"); args.push_back(std::to_string(cfg_.min_tokens)); }
    int pin[2], pout[2];
    if (pipe(pin) || pipe(pout)) return "no pipes for the image encoder";
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pin[0], 0);
    posix_spawn_file_actions_adddup2(&fa, pout[1], 1);
    for (int fd : { pin[0], pin[1], pout[0], pout[1] }) posix_spawn_file_actions_addclose(&fa, fd);
    posix_spawn_file_actions_addchdir_np(&fa, dir_.c_str());   // the ENC lines name files relative to it (#480)
    std::vector<char *> av;
    for (auto & a : args) av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    pid_t pid;
    const int rc = posix_spawn(&pid, av[0], &fa, nullptr, av.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(pin[0]); close(pout[1]);
    if (rc != 0) { close(pin[1]); close(pout[0]); return std::string("the vision encoder did not start: ") + strerror(rc); }
    fcntl(pin[1], F_SETFD, FD_CLOEXEC);
    fcntl(pout[0], F_SETFD, FD_CLOEXEC);
    pid_ = pid; in_ = pin[1]; out_ = pout[0];
    std::string line;
    try { line = readline(300.0, "the vision encoder did not start"); }
    catch (const std::exception & e) { kill_child(); return e.what(); }
    if (line.rfind("READY", 0) != 0) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        kill_child();
        return "the vision encoder did not start: " + line;
    }
    stopped_ = false;
    return "";
#else
    return "the vision encoder is not supported by impulse-fx on Windows yet";
#endif
}

std::pair<std::string, int> Vision::encode(const std::string & source, const std::vector<std::string> & keep) {
    const std::string data = normalize(load(source));
    const std::string key = sha256_hex(data).substr(0, 32);
    std::lock_guard<std::mutex> g(mu_);
    for (size_t i = 0; i < cache_.size(); ++i)
        if (cache_[i].first == key) { auto e = cache_[i]; cache_.erase(cache_.begin() + (long) i); cache_.push_back(e); return e.second; }
#if !defined(_WIN32)
    if (stopped_ || pid_ <= 0) throw fxf::BadRequest("the image could not be read: the vision encoder stopped");
    const std::string img = key + ".img", out = key + ".sve";
    {
        std::ofstream f(dir_ + "/" + img, std::ios::binary);
        f.write(data.data(), (std::streamsize) data.size());
    }
    std::string line;
    const std::string cmd = "ENC " + img + " " + out + "\n";
    bool wrote = write(in_, cmd.data(), cmd.size()) == (ssize_t) cmd.size();
    try { if (wrote) line = readline(300.0, "the image could not be read"); }
    catch (const std::exception & e) { stopped_ = true; unlink((dir_ + "/" + img).c_str()); throw fxf::BadRequest(e.what()); }
    unlink((dir_ + "/" + img).c_str());
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    while (!line.empty() && line.back() == ' ') line.pop_back();
    if (line.rfind("OK", 0) != 0)
        throw fxf::BadRequest("the image could not be read: " + (line.rfind("ERR", 0) == 0 ? line.substr(std::min<size_t>(4, line.size())) : std::string("the vision encoder stopped")));
    int n = 0;
    { std::istringstream is(line.substr(2)); is >> n; }
    const std::string path = dir_ + "/" + out;
    cache_.push_back({ key, { path, n } });
    for (size_t i = 0; cache_.size() > 64 && i < cache_.size();) {   // oldest first, never one this request still needs
        const std::string & f = cache_[i].second.first;
        bool held = f == path;
        for (const auto & k : keep) if (k == f) held = true;
        if (!held) { unlink(f.c_str()); cache_.erase(cache_.begin() + (long) i); }
        else ++i;
    }
    return { path, n };
#else
    (void) keep;
    throw fxf::BadRequest("the vision encoder is not supported by impulse-fx on Windows yet");
#endif
}

std::string Vision::combine(const std::vector<std::string> & files) {
    size_t nbytes = 0;
    struct stat st;
    for (const auto & f : files) if (stat(f.c_str(), &st) == 0) nbytes += (size_t) st.st_size;
    const std::string name = "req-" + fxf::new_hex(12) + ".sve";
    std::string path = dir_ + "/" + name;
#if !defined(_WIN32)
    struct statvfs vs;
    if (access("/dev/shm", W_OK) == 0 && statvfs("/dev/shm", &vs) == 0 &&
        (unsigned long long) vs.f_bavail * vs.f_frsize > 2ULL * nbytes + (16ULL << 20))
        path = "/dev/shm/strata-vision-" + std::to_string(getpid()) + "-" + name;
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return "";
    for (const auto & f : files) {
        std::ifstream in(f, std::ios::binary);
        char b[1 << 16];
        while (in) {
            in.read(b, sizeof b);
            const std::streamsize n = in.gcount();
            if (n > 0 && write(fd, b, (size_t) n) != (ssize_t) n) { close(fd); unlink(path.c_str()); return ""; }
        }
    }
    close(fd);
#endif
    return path;
}

}  // namespace fxv
