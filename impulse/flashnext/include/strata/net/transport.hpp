// include/strata/net/transport.hpp - the engine's network transport: framed, authenticated, encrypted messages between
// two processes, on one host or across hosts (RPC: a remote expert tier, a layer pipeline across hosts).
//
// THE INTERFACE IS `Channel`: send / receive typed messages.  Backends: TCP (this file's tcp_connect / Listener) and,
// where libibverbs is present, RDMA (net/rdma.hpp: RC queue pairs, RDMA write with immediate) - the same handshake and the
// same message types, so a caller never knows which one it holds.
//
// SECURITY (the secure default, plaintext only when asked for explicitly):
//   - both ends hold a shared secret (--rpc-secret-file: >= 16 bytes, or STRATA_RPC_SECRET), never sent;
//   - hello: each side sends 32 random bytes (nonce); HKDF-SHA256(salt = nonce_c || nonce_s, ikm = secret) gives the two
//     directions' AES-256 keys and an authentication key;
//   - mutual authentication: each side sends HMAC-SHA256(auth key, its role || both hellos) and checks the other's - a peer
//     without the secret is refused before any activation moves, and a recorded session cannot be replayed (fresh nonces);
//   - every later frame is AES-256-GCM: 96-bit nonce = a per-direction counter (never reused under one key), the frame's
//     header as associated data, a 128-bit tag - a modified, reordered, replayed or truncated frame ends the connection;
//   - plaintext (--rpc-plaintext) must be chosen by BOTH ends (the hello carries it; a mismatch is refused); with a secret
//     the peers still authenticate, only the payload is not encrypted.  Without a secret plaintext is the only mode, and
//     only when asked for.
// Built without OpenSSL (STRATA_NET_CRYPTO undefined) only plaintext without a secret is available, and it says so.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::net {

/// host:port (an IPv4 or IPv6 address, or a name; [v6]:port for a literal IPv6)
struct Endpoint {
    std::string host;
    int port = 0;
    static bool parse(const std::string& text, Endpoint& out, std::string& error);
    std::string str() const;
};

struct Security {
    std::vector<uint8_t> secret;   ///< the shared secret (empty: none)
    bool plaintext = false;        ///< --rpc-plaintext: no payload encryption (both ends must say so)
    /// The secret from a file (raw bytes, or hex when the file holds only hex digits; surrounding whitespace ignored),
    /// else from $STRATA_RPC_SECRET when `file` is empty.  Fewer than 16 bytes is refused.
    static bool load(const std::string& file, Security& out, std::string& error);
    /// Whether this configuration can make a connection (an encrypted one needs a secret and OpenSSL).
    bool usable(std::string& why) const;
};

/// Built with OpenSSL (AES-256-GCM, HKDF, HMAC available).
bool crypto_available();

/// Message types (the frame's `type`).  0x10..0x1f: the remote expert tier (net/expert_rpc.hpp); 0x20..0x2f: the layer
/// pipeline's hand-off (net/stage_rpc.hpp).
enum MsgType : uint32_t {
    kHello = 1,
    kAuth = 2,
    kError = 3,
    kBye = 4,
    kPing = 5,
    kExpertReq = 0x10,
    kExpertResp = 0x11,
    kStageHandoff = 0x20,
    kStageReturn = 0x21,
};

/// The largest payload a frame carries (a whole verify window's hand-off is ~410 KB, an expert request ~1 MB).
inline constexpr uint64_t kMaxFrame = 256ull << 20;

/// One piece of a gathered send.
struct Iov {
    const void* data;
    size_t size;
};

/// A connection: typed messages both ways.  Not thread-safe: one sender and one receiver at a time (they may be two
/// threads, one each).
class Channel {
public:
    virtual ~Channel() = default;
    virtual bool send(uint32_t type, const Iov* pieces, int n, std::string& error) = 0;
    bool send(uint32_t type, const void* data, size_t size, std::string& error) {
        const Iov one{data, size};
        return send(type, &one, 1, error);
    }
    /// The next message; `timeout_ms` < 0 waits as long as it takes.  False: closed, a timeout (error "timeout"), or
    /// a frame that failed authentication (the connection is then closed).
    virtual bool recv(uint32_t& type, std::vector<uint8_t>& data, std::string& error, int timeout_ms = -1) = 0;
    virtual void close() = 0;
    /// "tcp+aes-256-gcm", "tcp (plaintext, authenticated)", "tcp (plaintext)", "rdma+aes-256-gcm", ...
    virtual std::string describe() const = 0;
    /// The peer's role string from its hello (e.g. "engine", "expert-server").
    virtual const std::string& peer_role() const = 0;
};

/// Connect and run the handshake as the client.
std::unique_ptr<Channel> tcp_connect(const Endpoint& to, const Security& sec, const std::string& role, std::string& error,
                                     int timeout_ms = 10000);

class Listener {
public:
    Listener() = default;
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    /// Bind and listen (port 0: any free port, see port()).
    bool open(const Endpoint& at, std::string& error);
    int port() const { return port_; }
    /// The next connection with its handshake done as the server; nullptr on a failed handshake (error says why; the
    /// listener stays open) or when closed.  `timeout_ms` < 0 waits.
    std::unique_ptr<Channel> accept(const Security& sec, const std::string& role, std::string& error, int timeout_ms = -1);
    void close();

private:
    int fd_ = -1;
    int port_ = 0;
};

/// The handshake over an already connected stream socket (tests and the RDMA bootstrap use it): `server` picks the role.
std::unique_ptr<Channel> tcp_wrap(int fd, bool server, const Security& sec, const std::string& role, std::string& error);

}  // namespace strata::net
