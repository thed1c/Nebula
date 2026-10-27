// include/strata/net/rdma.hpp - the transport's RDMA backend: the same Channel (typed messages, the same security) over an
// InfiniBand / RoCE reliable-connected queue pair, for the remote expert tier and the layer pipeline on a ConnectX link.
//
// HOW.  A connection starts as TCP (net/transport.hpp: the authenticated handshake), then upgrades: over the encrypted TCP
// channel both sides exchange their queue pair numbers, GIDs / LIDs, packet sequence numbers and the address and key of a
// receive ring they registered (`slots` x `slot_bytes`), and fresh nonces that key the RDMA frames (HKDF of the secret).
// A message is then one RDMA WRITE WITH IMMEDIATE into a free slot of the peer's ring (immediate = the slot): the frame
// header and the AES-256-GCM sealed payload in place - no copy on the receiving side but the decryption.  Flow control:
// each side RDMA-writes how many messages it has consumed into an 8-byte word the sender registered; a sender with no free
// slot waits for it.  Completions are polled (lowest latency; STRATA_RDMA_EVENTS=1 sleeps on the completion channel).
// The TCP channel stays open for the teardown.
//
// Built only where libibverbs' headers are (CMake STRATA_RDMA, auto); elsewhere rdma_upgrade says so.  Needs memlock
// room (ulimit -l) for the rings and, in a container, /dev/infiniband.  tools/rdma-setup.sh (run by the owner, sudo)
// installs rdma-core and can add a soft-RoCE device on an Ethernet port to test without the ConnectX cables.
#pragma once

#include "strata/net/transport.hpp"

#include <memory>
#include <string>

namespace strata::net {

struct RdmaOptions {
    std::string device;          ///< "" = the first device (ibv_get_device_list), e.g. "mlx4_0", "rxe0"
    int port = 1;
    int gid_index = -1;          ///< -1: a RoCE v2 GID of the port (else index 0); STRATA_RDMA_GID_INDEX overrides
    int slots = 16;              ///< the receive ring's slots (messages in flight)
    size_t slot_bytes = 4u << 20;   ///< the largest message (header + payload + tag), 4 MiB
};

/// Built with libibverbs.
bool rdma_available();

/// Upgrade an established TCP channel (its handshake done) to RDMA; `server` = the side that accepted.  Both sides call
/// it with the same options' slot geometry.  On failure the TCP channel is gone too (error says why).
std::unique_ptr<Channel> rdma_upgrade(std::unique_ptr<Channel> tcp, bool server, const Security& sec, const RdmaOptions& opt,
                                      std::string& error);

/// The RDMA devices this host has, with their ports' link layers and states ("mlx4_0 port 1 Ethernet DOWN ..."), for the
/// startup log and the setup script's check.
std::string rdma_devices();

}  // namespace strata::net
