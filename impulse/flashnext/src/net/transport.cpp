// src/net/transport.cpp - see include/strata/net/transport.hpp.  POSIX sockets; OpenSSL's libcrypto for the handshake and
// the frames when STRATA_NET_CRYPTO is defined.
#include "strata/net/transport.hpp"

#include "crypto.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(STRATA_NET_CRYPTO)
#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#include <openssl/rand.h>
#endif

namespace strata::net {

bool crypto_available() {
#if defined(STRATA_NET_CRYPTO)
    return true;
#else
    return false;
#endif
}

// ------------------------------------------------------------------------------------------------ endpoint, secret
bool Endpoint::parse(const std::string& text, Endpoint& out, std::string& error) {
    std::string host, port;
    if (!text.empty() && text[0] == '[') {
        const size_t rb = text.find(']');
        if (rb == std::string::npos || rb + 1 >= text.size() || text[rb + 1] != ':') { error = "expected [v6]:port, got '" + text + "'"; return false; }
        host = text.substr(1, rb - 1);
        port = text.substr(rb + 2);
    } else {
        const size_t c = text.rfind(':');
        if (c == std::string::npos) { error = "expected host:port, got '" + text + "'"; return false; }
        host = text.substr(0, c);
        port = text.substr(c + 1);
    }
    char* end = nullptr;
    const long p = std::strtol(port.c_str(), &end, 10);
    if (port.empty() || *end != '\0' || p < 0 || p > 65535) { error = "bad port in '" + text + "'"; return false; }
    out.host = host.empty() ? "0.0.0.0" : host;
    out.port = (int) p;
    return true;
}

std::string Endpoint::str() const {
    return host.find(':') != std::string::npos ? "[" + host + "]:" + std::to_string(port) : host + ":" + std::to_string(port);
}

bool Security::load(const std::string& file, Security& out, std::string& error) {
    std::string raw;
    if (!file.empty()) {
        std::ifstream f(file, std::ios::binary);
        if (!f) { error = "cannot read the secret file " + file; return false; }
        std::ostringstream ss;
        ss << f.rdbuf();
        raw = ss.str();
    } else if (const char* e = std::getenv("STRATA_RPC_SECRET")) {
        raw = e;
    } else {
        out.secret.clear();
        return true;   // no secret (plaintext only, when asked for)
    }
    auto ws = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!raw.empty() && ws((unsigned char) raw.back())) raw.pop_back();
    size_t s = 0;
    while (s < raw.size() && ws((unsigned char) raw[s])) ++s;
    raw = raw.substr(s);
    const bool hex = !raw.empty() && raw.size() % 2 == 0 &&
                     std::all_of(raw.begin(), raw.end(), [](char c) { return std::isxdigit((unsigned char) c) != 0; });
    out.secret.clear();
    if (hex) {
        for (size_t i = 0; i < raw.size(); i += 2) out.secret.push_back((uint8_t) std::strtoul(raw.substr(i, 2).c_str(), nullptr, 16));
    } else {
        out.secret.assign(raw.begin(), raw.end());
    }
    if (out.secret.size() < 16) {
        error = "the RPC secret is shorter than 16 bytes (" + std::to_string(out.secret.size()) + "): use e.g. "
                "`openssl rand -hex 32 > secret`";
        out.secret.clear();
        return false;
    }
    return true;
}

bool Security::usable(std::string& why) const {
    if (!plaintext && secret.empty()) {
        why = "no RPC secret: give --rpc-secret-file (or STRATA_RPC_SECRET), or --rpc-plaintext for an unencrypted, "
              "unauthenticated link on a network you trust";
        return false;
    }
    if (!crypto_available() && !secret.empty()) {
        why = "this engine was built without OpenSSL: no encryption or authentication; only --rpc-plaintext without a secret";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ crypto helpers
namespace {

using namespace detail;
constexpr uint32_t kMagic = 0x4e525453;   // "STRN"
constexpr uint16_t kVersion = 1;
constexpr uint16_t kFlagPlaintext = 1;
constexpr size_t kNonce = 32;

// ------------------------------------------------------------------------------------------------ the TCP channel
struct Header {
    uint32_t type;
    uint32_t flags;   // 0 (reserved)
    uint64_t len;     // the payload's length (ciphertext without the tag)
};
static_assert(sizeof(Header) == 16, "frame header");

class TcpChannel final : public Channel {
public:
    explicit TcpChannel(int fd) : fd_(fd) {}
    using Channel::send;   // the one-buffer form
    ~TcpChannel() override { close(); }

    bool send(uint32_t type, const Iov* pieces, int n, std::string& error) override {
        size_t total = 0;
        for (int i = 0; i < n; ++i) total += pieces[i].size;
        if (total > kMaxFrame) { error = "a message larger than the frame limit"; return false; }
        Header h{type, 0, (uint64_t) total};
        if (tx_.on) {
#if defined(STRATA_NET_CRYPTO)
            if (!tx_.seal((const uint8_t*) &h, sizeof h, pieces, n, sealed_)) { error = "encryption failed"; return false; }
            return write_all(&h, sizeof h, error) && write_all(sealed_.data(), sealed_.size(), error);
#endif
        }
        if (!write_all(&h, sizeof h, error)) return false;
        for (int i = 0; i < n; ++i)
            if (pieces[i].size > 0 && !write_all(pieces[i].data, pieces[i].size, error)) return false;
        return true;
    }

    bool recv(uint32_t& type, std::vector<uint8_t>& data, std::string& error, int timeout_ms) override {
        if (fd_ < 0) { error = "closed"; return false; }
        if (timeout_ms >= 0) {
            pollfd p{fd_, POLLIN, 0};
            const int r = ::poll(&p, 1, timeout_ms);
            if (r == 0) { error = "timeout"; return false; }
            if (r < 0) { error = std::string("poll: ") + std::strerror(errno); return false; }
        }
        Header h{};
        if (!read_all(&h, sizeof h, error)) return false;
        if (h.len > kMaxFrame || h.flags != 0) { error = "a malformed frame"; close(); return false; }
        const size_t body = (size_t) h.len + (rx_.on ? kTag : 0);
        data.resize(body);
        if (body > 0 && !read_all(data.data(), body, error)) return false;
        if (rx_.on) {
#if defined(STRATA_NET_CRYPTO)
            if (!rx_.open((const uint8_t*) &h, sizeof h, data.data(), (size_t) h.len, data.data() + h.len)) {
                error = "a frame failed authentication (modified, replayed or from another key): connection closed";
                close();
                return false;
            }
#endif
            data.resize((size_t) h.len);
        }
        type = h.type;
        return true;
    }

    void close() override {
        if (fd_ >= 0) { ::shutdown(fd_, SHUT_RDWR); ::close(fd_); fd_ = -1; }
    }
    std::string describe() const override { return what_; }
    const std::string& peer_role() const override { return peer_role_; }

    // the handshake (see the header): hellos, keys, mutual authentication
    bool handshake(bool server, const Security& sec, const std::string& role, std::string& error) {
        std::string why;
        if (!sec.usable(why)) { error = why; return false; }
        std::vector<uint8_t> mine(4 + 2 + 2 + kNonce + 1);
        uint8_t nonce[kNonce];
        if (!random_bytes(nonce, sizeof nonce)) { error = "no random bytes"; return false; }
        const uint16_t flags = sec.plaintext ? kFlagPlaintext : 0;
        std::memcpy(mine.data(), &kMagic, 4);
        std::memcpy(mine.data() + 4, &kVersion, 2);
        std::memcpy(mine.data() + 6, &flags, 2);
        std::memcpy(mine.data() + 8, nonce, kNonce);
        const std::string r = role.substr(0, 64);
        mine[8 + kNonce] = (uint8_t) r.size();
        mine.insert(mine.end(), r.begin(), r.end());
        std::vector<uint8_t> theirs;
        uint32_t type = 0;
        auto recv_hello = [&]() -> bool {
            if (!recv(type, theirs, error, 15000)) { if (error == "timeout") error = "the peer sent no hello"; return false; }
            if (type == kError) { error = "the peer refused: " + std::string(theirs.begin(), theirs.end()); return false; }
            if (type != kHello || theirs.size() < 4 + 2 + 2 + kNonce + 1) { error = "not a strata peer (no hello)"; return false; }
            uint32_t m = 0; uint16_t v = 0, f = 0;
            std::memcpy(&m, theirs.data(), 4); std::memcpy(&v, theirs.data() + 4, 2); std::memcpy(&f, theirs.data() + 6, 2);
            if (m != kMagic) { error = "not a strata peer (bad magic)"; return false; }
            if (v != kVersion) { error = "the peer speaks protocol version " + std::to_string(v) + ", this end " + std::to_string(kVersion); return false; }
            if ((f & kFlagPlaintext) != flags) {
                error = (f & kFlagPlaintext) ? "the peer asked for plaintext and this end did not (--rpc-plaintext on both ends, or neither)"
                                             : "this end asked for plaintext and the peer did not (--rpc-plaintext on both ends, or neither)";
                return false;
            }
            const size_t rn = theirs[8 + kNonce];
            if (theirs.size() < 8 + kNonce + 1 + rn) { error = "a short hello"; return false; }
            peer_role_.assign((const char*) theirs.data() + 8 + kNonce + 1, rn);
            return true;
        };
        if (server) {
            if (!recv_hello()) {
                if (error.find("plaintext") != std::string::npos) send(kError, error.data(), error.size(), why);
                return false;
            }
            if (!send(kHello, mine.data(), mine.size(), error)) return false;
        } else {
            if (!send(kHello, mine.data(), mine.size(), error) || !recv_hello()) return false;
        }
        if (sec.secret.empty()) {   // plaintext, unauthenticated (asked for on both ends)
            what_ = "tcp (plaintext, unauthenticated)";
            return true;
        }
#if defined(STRATA_NET_CRYPTO)
        const std::vector<uint8_t>& hc = server ? theirs : mine;
        const std::vector<uint8_t>& hs = server ? mine : theirs;
        uint8_t salt[2 * kNonce];
        std::memcpy(salt, hc.data() + 8, kNonce);
        std::memcpy(salt + kNonce, hs.data() + 8, kNonce);
        uint8_t k_c2s[kKey], k_s2c[kKey], k_auth[kKey];
        if (!hkdf(sec.secret, salt, sizeof salt, "strata-net/1 c2s", k_c2s, kKey) ||
            !hkdf(sec.secret, salt, sizeof salt, "strata-net/1 s2c", k_s2c, kKey) ||
            !hkdf(sec.secret, salt, sizeof salt, "strata-net/1 auth", k_auth, kKey)) {
            error = "key derivation failed";
            return false;
        }
        auto proof = [&](const char* who, uint8_t out[32]) {
            std::vector<uint8_t> t(who, who + std::strlen(who));
            t.insert(t.end(), hc.begin(), hc.end());
            t.insert(t.end(), hs.begin(), hs.end());
            return hmac256(k_auth, kKey, t, out);
        };
        uint8_t mine_p[32], want[32];
        if (!proof(server ? "server" : "client", mine_p) || !proof(server ? "client" : "server", want)) { error = "HMAC failed"; return false; }
        auto check = [&]() -> bool {
            std::vector<uint8_t> got;
            if (!recv(type, got, error, 15000)) { if (error == "timeout") error = "the peer did not authenticate"; return false; }
            if (type == kError) { error = "the peer refused: " + std::string(got.begin(), got.end()); return false; }
            if (type != kAuth || got.size() != 32 || CRYPTO_memcmp(got.data(), want, 32) != 0) {
                error = "the peer does not hold this RPC secret: refused";
                return false;
            }
            return true;
        };
        if (server) {
            if (!check()) { const std::string m = "authentication failed"; send(kError, m.data(), m.size(), why); return false; }
            if (!send(kAuth, mine_p, 32, error)) return false;
        } else {
            if (!send(kAuth, mine_p, 32, error) || !check()) return false;
        }
        if (!sec.plaintext) {
            if (!tx_.init(server ? k_s2c : k_c2s, true) || !rx_.init(server ? k_c2s : k_s2c, false)) { error = "cipher setup failed"; return false; }
            what_ = "tcp+aes-256-gcm (authenticated)";
        } else {
            what_ = "tcp (plaintext, authenticated)";
        }
        OPENSSL_cleanse(k_c2s, kKey); OPENSSL_cleanse(k_s2c, kKey); OPENSSL_cleanse(k_auth, kKey);
        return true;
#else
        error = "built without OpenSSL";
        return false;
#endif
    }

private:
    int fd_;
    Gcm tx_, rx_;
    std::vector<uint8_t> sealed_;
    std::string what_ = "tcp", peer_role_;

    bool write_all(const void* p, size_t n, std::string& error) {
        const char* c = (const char*) p;
        while (n > 0) {
            const ssize_t w = ::send(fd_, c, n, MSG_NOSIGNAL);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { error = std::string("send: ") + (w < 0 ? std::strerror(errno) : "closed"); close(); return false; }
            c += w;
            n -= (size_t) w;
        }
        return true;
    }
    bool read_all(void* p, size_t n, std::string& error) {
        char* c = (char*) p;
        while (n > 0) {
            const ssize_t r = ::recv(fd_, c, n, 0);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { error = r == 0 ? "closed by the peer" : std::string("recv: ") + std::strerror(errno); close(); return false; }
            c += r;
            n -= (size_t) r;
        }
        return true;
    }
};

void tune(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    int buf = 8 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
}

}  // namespace

std::unique_ptr<Channel> tcp_wrap(int fd, bool server, const Security& sec, const std::string& role, std::string& error) {
    tune(fd);
    auto ch = std::make_unique<TcpChannel>(fd);
    if (!ch->handshake(server, sec, role, error)) return nullptr;
    return ch;
}

std::unique_ptr<Channel> tcp_connect(const Endpoint& to, const Security& sec, const std::string& role, std::string& error,
                                     int timeout_ms) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string port = std::to_string(to.port);
    if (const int r = getaddrinfo(to.host.c_str(), port.c_str(), &hints, &res); r != 0) {
        error = "cannot resolve " + to.host + ": " + gai_strerror(r);
        return nullptr;
    }
    int fd = -1;
    std::string last = "no address";
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd < 0) continue;
        const int fl = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        int r = ::connect(fd, a->ai_addr, a->ai_addrlen);
        if (r < 0 && errno == EINPROGRESS) {
            pollfd p{fd, POLLOUT, 0};
            r = ::poll(&p, 1, timeout_ms) == 1 ? 0 : -1;
            int so = 0;
            socklen_t sl = sizeof so;
            if (r == 0 && (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so, &sl) != 0 || so != 0)) { errno = so; r = -1; }
            else if (r != 0) errno = ETIMEDOUT;
        }
        if (r == 0) { fcntl(fd, F_SETFL, fl); break; }
        last = std::strerror(errno);
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { error = "cannot connect to " + to.str() + ": " + last; return nullptr; }
    return tcp_wrap(fd, false, sec, role, error);
}

Listener::~Listener() { close(); }

bool Listener::open(const Endpoint& at, std::string& error) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string port = std::to_string(at.port);
    if (const int r = getaddrinfo(at.host.empty() ? nullptr : at.host.c_str(), port.c_str(), &hints, &res); r != 0) {
        error = "cannot resolve " + at.host + ": " + gai_strerror(r);
        return false;
    }
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        fd_ = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd_ < 0) continue;
        int one = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (::bind(fd_, a->ai_addr, a->ai_addrlen) == 0 && ::listen(fd_, 8) == 0) break;
        error = std::string("cannot listen on ") + at.str() + ": " + std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
    }
    freeaddrinfo(res);
    if (fd_ < 0) return false;
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    getsockname(fd_, (sockaddr*) &ss, &sl);
    port_ = ss.ss_family == AF_INET6 ? ntohs(((sockaddr_in6*) &ss)->sin6_port) : ntohs(((sockaddr_in*) &ss)->sin_port);
    return true;
}

std::unique_ptr<Channel> Listener::accept(const Security& sec, const std::string& role, std::string& error, int timeout_ms) {
    if (fd_ < 0) { error = "not listening"; return nullptr; }
    if (timeout_ms >= 0) {
        pollfd p{fd_, POLLIN, 0};
        const int r = ::poll(&p, 1, timeout_ms);
        if (r == 0) { error = "timeout"; return nullptr; }
        if (r < 0) { error = std::string("poll: ") + std::strerror(errno); return nullptr; }
    }
    const int c = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (c < 0) { error = std::string("accept: ") + std::strerror(errno); return nullptr; }
    return tcp_wrap(c, true, sec, role, error);
}

void Listener::close() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

}  // namespace strata::net
