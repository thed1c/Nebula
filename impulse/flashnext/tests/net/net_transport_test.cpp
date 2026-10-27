// net_transport_test: the RPC transport (include/strata/net/transport.hpp) on localhost, no GPU.
//   - a secret on both ends: authenticated + AES-256-GCM, messages both ways (up to a whole window's hand-off, 410 KB)
//   - a wrong secret, a plaintext mismatch, no secret without --rpc-plaintext: refused
//   - a man in the middle that flips one byte, replays a frame or drops one: the receiver refuses the frame and closes
//   - encrypted frames do not show their payload on the wire; plaintext ones do (the opt-in)
//   - the server in another process (fork): the handshake and a round trip across two processes
#include "strata/net/transport.hpp"

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace strata::net;

static int fails = 0;
static void check(const char* what, bool ok, const std::string& detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  -- ", detail.c_str());
    if (!ok) ++fails;
}

static Security sec_of(const char* s, bool plain = false) {
    Security x;
    x.secret.assign(s, s + std::strlen(s));
    x.plaintext = plain;
    return x;
}

// a connected pair through the handshake: the server on a thread
struct Pair {
    std::unique_ptr<Channel> c, s;
    std::string ce, se;
};
static Pair connect_pair(const Security& cs, const Security& ss, int port_override = 0, Listener* Lp = nullptr) {
    Pair p;
    Listener own;
    Listener& L = Lp ? *Lp : own;
    if (!Lp) {
        Endpoint at{"127.0.0.1", 0};
        if (!L.open(at, p.se)) return p;
    }
    std::thread srv([&] { p.s = L.accept(ss, "test-server", p.se, 5000); });
    Endpoint to{"127.0.0.1", port_override ? port_override : L.port()};
    p.c = tcp_connect(to, cs, "test-client", p.ce, 5000);
    srv.join();
    return p;
}

// raw relay: listen raw, connect raw to the backend, forward both ways, mangling client -> server bytes
static int raw_listen(int& port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    ::bind(fd, (sockaddr*) &a, sizeof a);
    ::listen(fd, 4);
    socklen_t al = sizeof a;
    getsockname(fd, (sockaddr*) &a, &al);
    port = ntohs(a.sin_port);
    return fd;
}
static int raw_connect(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons((uint16_t) port);
    return ::connect(fd, (sockaddr*) &a, sizeof a) == 0 ? fd : -1;
}
struct Relay {
    int lfd = -1, port = 0;
    std::thread th;
    std::vector<char> c2s;   // every byte the client sent
    std::function<bool(std::vector<char>& chunk, size_t off)> mangle;   // false: drop the chunk
    void start(int backend) {
        lfd = raw_listen(port);
        th = std::thread([this, backend] {
            const int c = ::accept(lfd, nullptr, nullptr);
            const int s = raw_connect(backend);
            std::thread back([c, s] {
                char b[65536];
                for (;;) { const ssize_t n = ::recv(s, b, sizeof b, 0); if (n <= 0) break; if (::send(c, b, (size_t) n, MSG_NOSIGNAL) <= 0) break; }
                ::shutdown(c, SHUT_RDWR);
            });
            char b[65536];
            size_t off = 0;
            for (;;) {
                const ssize_t n = ::recv(c, b, sizeof b, 0);
                if (n <= 0) break;
                std::vector<char> chunk(b, b + n);
                c2s.insert(c2s.end(), chunk.begin(), chunk.end());
                const bool keep = !mangle || mangle(chunk, off);
                off += (size_t) n;
                if (keep && ::send(s, chunk.data(), chunk.size(), MSG_NOSIGNAL) <= 0) break;
            }
            ::shutdown(s, SHUT_RDWR);
            back.join();
            ::close(c);
            ::close(s);
        });
    }
    ~Relay() { if (th.joinable()) th.join(); if (lfd >= 0) ::close(lfd); }
};

int main() {
    const char* S1 = "0123456789abcdef-the-shared-secret";
    // 1. authenticated + encrypted round trips
    {
        Pair p = connect_pair(sec_of(S1), sec_of(S1));
        check("secret on both ends: connected", p.c && p.s, p.ce + " | " + p.se);
        if (p.c && p.s) {
            check("... encrypted and authenticated", p.c->describe().find("aes-256-gcm") != std::string::npos, p.c->describe());
            check("... roles exchanged", p.c->peer_role() == "test-server" && p.s->peer_role() == "test-client");
            std::vector<uint8_t> big(410 * 1024);
            for (size_t i = 0; i < big.size(); ++i) big[i] = (uint8_t) (i * 131 + 7);
            std::string e;
            bool ok = true;
            for (int r = 0; r < 20 && ok; ++r) {
                ok &= p.c->send(kStageHandoff, big.data(), big.size() - (size_t) r, e);
                uint32_t t = 0;
                std::vector<uint8_t> got;
                ok &= p.s->recv(t, got, e, 5000) && t == kStageHandoff && got.size() == big.size() - (size_t) r &&
                      std::memcmp(got.data(), big.data(), got.size()) == 0;
                ok &= p.s->send(kStageReturn, got.data(), 100, e);
                ok &= p.c->recv(t, got, e, 5000) && t == kStageReturn && got.size() == 100 && std::memcmp(got.data(), big.data(), 100) == 0;
            }
            check("... 20 round trips up to 410 KB, both ways, intact", ok, e);
            uint32_t t = 0;
            std::vector<uint8_t> got;
            check("... an empty message", p.c->send(kPing, nullptr, 0, e) && p.s->recv(t, got, e, 5000) && t == kPing && got.empty(), e);
            check("... recv times out when nothing comes", !p.s->recv(t, got, e, 100) && e == "timeout", e);
        }
    }
    // 2. refusals
    {
        Pair p = connect_pair(sec_of(S1), sec_of("another-secret-of-16-plus-bytes"));
        check("a wrong secret: refused on both ends", !p.c && !p.s, p.ce + " | " + p.se);
        check("... and says so", p.ce.find("secret") != std::string::npos || p.ce.find("refused") != std::string::npos, p.ce);
    }
    {
        Pair p = connect_pair(sec_of(S1, true), sec_of(S1, false));
        check("plaintext on one end only: refused", !p.c && !p.s, p.ce + " | " + p.se);
        check("... and says why", (p.ce + p.se).find("plaintext") != std::string::npos, p.ce + " | " + p.se);
    }
    {
        Security none;
        std::string why;
        check("no secret and no --rpc-plaintext: not usable", !none.usable(why) && why.find("--rpc-plaintext") != std::string::npos, why);
        none.plaintext = true;
        check("... --rpc-plaintext without a secret: usable", none.usable(why));
        Security s;
        std::string e;
        ::setenv("STRATA_RPC_SECRET", "short", 1);
        check("a secret under 16 bytes is refused", !Security::load("", s, e), e);
        ::setenv("STRATA_RPC_SECRET", "00112233445566778899aabbccddeeff", 1);
        check("a hex secret: 16 bytes", Security::load("", s, e) && s.secret.size() == 16 && s.secret[15] == 0xff, e);
        ::unsetenv("STRATA_RPC_SECRET");
    }
    {
        Security a, b;
        a.plaintext = b.plaintext = true;
        Pair p = connect_pair(a, b);
        check("plaintext on both ends, no secret: connects (the opt-in)", p.c && p.s && p.c->describe().find("unauthenticated") != std::string::npos,
              p.c ? p.c->describe() : p.ce);
    }
    // 3. the wire: encrypted frames hide the payload, plaintext ones show it; tampering, replay and loss are caught
    const std::string marker = "VISIBLE-ACTIVATION-PATTERN-0123456789";
    auto via_relay = [&](const Security& sec, std::function<bool(std::vector<char>&, size_t)> mangle, std::string& wire,
                         std::string& recv_error, int sends) -> bool {
        Listener L;
        Endpoint at{"127.0.0.1", 0};
        std::string e;
        L.open(at, e);
        Relay R;
        R.mangle = mangle;
        R.start(L.port());
        Pair p = connect_pair(sec, sec, R.port, &L);
        bool all = p.c && p.s;
        if (all) {
            for (int i = 0; i < sends; ++i) p.c->send(kExpertReq, marker.data(), marker.size(), e);
            for (int i = 0; i < sends && all; ++i) {
                uint32_t t = 0;
                std::vector<uint8_t> got;
                all = p.s->recv(t, got, recv_error, 2000) && got.size() == marker.size() &&
                      std::memcmp(got.data(), marker.data(), marker.size()) == 0;
            }
            p.c->close();
            p.s->close();
        }
        R.th.join();
        wire.assign(R.c2s.begin(), R.c2s.end());
        return all;
    };
    {
        std::string wire, re;
        const bool ok = via_relay(sec_of(S1), nullptr, wire, re, 3);
        check("encrypted: the frames arrive through a relay", ok, re);
        check("... the payload is not on the wire", wire.find(marker) == std::string::npos);
        ok && via_relay(sec_of(S1, true), nullptr, wire, re, 1);
        check("plaintext + secret: the payload is on the wire (the opt-in)", wire.find(marker) != std::string::npos);
    }
    {
        // the handshake is the first ~3 frames; flip a byte late in the stream (inside the 3rd encrypted message)
        size_t total = 0;
        std::string wire, re;
        via_relay(sec_of(S1), [&](std::vector<char>& c, size_t) { total += c.size(); return true; }, wire, re, 3);
        const size_t flip_at = wire.size() - 5;
        const bool ok = via_relay(sec_of(S1), [&](std::vector<char>& c, size_t off) {
            if (flip_at >= off && flip_at < off + c.size()) c[flip_at - off] ^= 0x40;
            return true;
        }, wire, re, 3);
        check("a flipped byte: the frame is refused", !ok && re.find("authentication") != std::string::npos, re);
    }
    // replay and loss, frame by frame: a relay that reads the client's frames (16-byte header; after the hello and the
    // auth the body carries a 16-byte tag) and forwards message 2 twice, or not at all
    auto frame_relay = [&](int mode, std::string& recv_error) -> int {   // mode 1: replay message 2, 2: drop it
        Listener L;
        Endpoint at{"127.0.0.1", 0};
        std::string e;
        L.open(at, e);
        int rport = 0;
        const int lfd = raw_listen(rport);
        std::thread relay([&, lfd] {
            const int c = ::accept(lfd, nullptr, nullptr);
            const int sfd = raw_connect(L.port());
            std::thread back([c, sfd] {
                char b[65536];
                for (;;) { const ssize_t n = ::recv(sfd, b, sizeof b, 0); if (n <= 0) break; if (::send(c, b, (size_t) n, MSG_NOSIGNAL) <= 0) break; }
                ::shutdown(c, SHUT_RDWR);
            });
            auto read_n = [&](std::vector<char>& v, size_t n) {
                v.resize(n);
                size_t got = 0;
                while (got < n) { const ssize_t r = ::recv(c, v.data() + got, n - got, 0); if (r <= 0) return false; got += (size_t) r; }
                return true;
            };
            for (int idx = 0;; ++idx) {
                std::vector<char> h, body;
                if (!read_n(h, 16)) break;
                uint64_t len = 0;
                std::memcpy(&len, h.data() + 8, 8);
                if (!read_n(body, (size_t) len + (idx >= 2 ? 16 : 0))) break;
                std::vector<char> f(h);
                f.insert(f.end(), body.begin(), body.end());
                const int msg = idx - 2;   // the messages after the handshake
                int copies = 1;
                if (msg == 1 && mode == 1) copies = 2;
                if (msg == 1 && mode == 2) copies = 0;
                for (int k = 0; k < copies; ++k) ::send(sfd, f.data(), f.size(), MSG_NOSIGNAL);
            }
            ::shutdown(sfd, SHUT_RDWR);
            back.join();
            ::close(c); ::close(sfd);
        });
        Pair p = connect_pair(sec_of(S1), sec_of(S1), rport, &L);
        int delivered = 0;
        if (p.c && p.s) {
            for (int i = 0; i < 4; ++i) p.c->send(kExpertReq, marker.data(), marker.size(), e);
            for (int i = 0; i < 5; ++i) {
                uint32_t t = 0;
                std::vector<uint8_t> got;
                if (!p.s->recv(t, got, recv_error, 1000)) break;
                ++delivered;
            }
            p.c->close();
        }
        relay.join();
        ::close(lfd);
        return delivered;
    };
    {
        std::string re;
        const int n = frame_relay(1, re);
        check("a replayed frame: refused after the frames before it", n == 2 && re.find("authentication") != std::string::npos,
              std::to_string(n) + " delivered: " + re);
        re.clear();
        const int m = frame_relay(2, re);
        check("a dropped frame: the next one is refused", m == 1 && re.find("authentication") != std::string::npos,
              std::to_string(m) + " delivered: " + re);
    }
    // 4. two processes
    {
        Listener L;
        Endpoint at{"127.0.0.1", 0};
        std::string e;
        L.open(at, e);
        const int port = L.port();
        const pid_t pid = fork();
        if (pid == 0) {
            std::string se;
            auto s = L.accept(sec_of(S1), "child", se, 10000);
            if (!s) _exit(3);
            uint32_t t = 0;
            std::vector<uint8_t> d;
            if (!s->recv(t, d, se, 10000)) _exit(4);
            for (auto& b : d) b = (uint8_t) (b ^ 0x5a);
            if (!s->send(kStageReturn, d.data(), d.size(), se)) _exit(5);
            s->recv(t, d, se, 2000);
            _exit(0);
        }
        L.close();
        auto c = tcp_connect(Endpoint{"127.0.0.1", port}, sec_of(S1), "parent", e, 10000);
        bool ok = c != nullptr;
        if (ok) {
            std::vector<uint8_t> msg(51200);
            for (size_t i = 0; i < msg.size(); ++i) msg[i] = (uint8_t) i;
            uint32_t t = 0;
            std::vector<uint8_t> got;
            ok = c->send(kStageHandoff, msg.data(), msg.size(), e) && c->recv(t, got, e, 10000) && got.size() == msg.size();
            for (size_t i = 0; ok && i < msg.size(); ++i) ok = got[i] == (uint8_t) (msg[i] ^ 0x5a);
            c->close();
        }
        int status = 0;
        waitpid(pid, &status, 0);
        check("two processes: handshake and a 51 KB round trip (one token's hand-off)", ok && WIFEXITED(status) && WEXITSTATUS(status) == 0, e);
    }
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
