# RPC: the engine across hosts

`include/strata/net/` holds the engine's network transport and what runs on it. Opt-in: nothing changes unless a flag
below is given. POSIX only (`STRATA_NET`, on by default off Windows); OpenSSL's libcrypto for the security.

## The transport (`net/transport.hpp`)

Framed, typed messages between two processes (`Channel`), TCP today and RDMA where libibverbs is present
(`net/rdma.hpp`, `STRATA_RDMA`, below). Every connection starts with the same handshake:

- a shared secret on both hosts (`--rpc-secret-file F`, at least 16 bytes, for example `openssl rand -hex 32 > rpc.secret`;
  or `STRATA_RPC_SECRET`), never sent;
- hellos with 32 random bytes each; HKDF-SHA256 over the secret and both nonces gives each direction its AES-256 key and an
  authentication key;
- mutual authentication: each side proves the secret with an HMAC over both hellos (a peer without it is refused before
  any activation moves; a recorded session cannot be replayed);
- every later frame is AES-256-GCM with a per-direction counter as the nonce and the frame header as associated data: a
  changed, replayed, reordered or dropped frame ends the connection (tests/net/net_transport_test.cpp checks each).

`--rpc-plaintext` on both ends skips the encryption (a direct cable you trust); with a secret the peers still
authenticate. Without a secret plaintext is the only mode, and only when asked for.

Measured on the dev machine, two processes on localhost (tests/net/stage_rpc_test.cpp): a 5-row window's hand-off (250 KB) round
trip 0.55 ms encrypted, 0.24 ms plaintext.

## RDMA (`--rpc-transport rdma`, `strata-rpc-experts --rdma`)

`net/rdma.hpp`: after the TCP handshake the connection upgrades to an InfiniBand / RoCE reliable-connected queue pair. The
two ends exchange their queue pairs, GIDs / LIDs and the address and key of a registered receive ring over the encrypted
TCP channel; a message is then one RDMA WRITE WITH IMMEDIATE into a free slot of the peer's ring (the immediate names the
slot), sealed with AES-256-GCM under keys from fresh nonces (or plaintext with `--rpc-plaintext`); each side RDMA-writes
how many messages it consumed into the sender's credit word (flow control). Completions are polled
(`STRATA_RDMA_EVENTS=1` sleeps on the completion channel instead).

Built only where libibverbs' headers are (CMake `STRATA_RDMA=AUTO`; `ON` requires them). The dev machine and the test server have the
runtime libraries but not the headers today: `tools/rdma-setup.sh` (worktree root, run with sudo by the owner) installs
rdma-core, libibverbs-dev, librdmacm-dev, perftest and ibverbs-utils, can add a soft-RoCE device on an Ethernet port to
test before the ConnectX cables exist (`--soft-roce eno1`), sets the memlock limit (`--memlock <user>`), and `undo` takes back
exactly what it did. Building needs no install: `-DSTRATA_RDMA_SDK=<dir>` takes the headers and library from the -dev
packages unpacked into a folder (`apt-get download libibverbs-dev libibverbs1 librdmacm-dev librdmacm1t64 ibverbs-providers`,
then `dpkg -x` each into `<dir>`), the binaries then load the system's runtime
libibverbs.so.1 (build-tree binaries carry the SDK folder as their run path). `tests/net/rdma_test.cpp` runs on an active port (loopback, two threads) and skips otherwise. A
ConnectX port needs a link (the dev machine's ConnectX-3 Pro ports are down today; the test server has no card yet).

## Remote experts (`--rpc-experts`, `strata-rpc-experts`)

Another host computes the CPU-served experts of a range of layers. On that host (CPU only, no GPU needed; it maps the
same pack's `experts.bin` and reads only its layers' pages):

    strata-rpc-experts --pack /models/<pack>/pack --layers 24-47 --listen 0.0.0.0:7710 --secret-file rpc.secret

and the engine:

    --rpc-experts server-b:7710@24-47 --rpc-secret-file rpc.secret

In each verify window the engine's dispatch sends the window's activations (10 KB a token) and the CPU entries of those
layers; the server answers one row per entry (10 KB each), written where the CPU pool would have written it. GPU hits,
PCIe and helper-GPU entries stay where they are. The rows are bit for bit the engine's own CPU rows on a CPU with the
same instruction set (tests/net/expert_rpc_test.cpp: 313 rows of the real NVFP4 pack, layers 0 and 47, across two
processes). If the server fails, that window fails and the engine takes the layers back on its own pool.

Each served layer costs one network round trip per window, so this pays only on a fast link: on 1 GbE (0.23 ms ping
two machines) 24 layers add 6-25 ms per window; RDMA (ConnectX-3, a few microseconds) makes it worth it.
Verify windows only (`--spec 1` or more; native packs always run in verify windows).

## The layer pipeline across hosts (not yet)

`net/stage_rpc.hpp` is the hand-off frame (one window's rows, `hc*n_embd + n_embd + hc` fp32 each, 51.2 KB a token)
and its bit-exact transport. The engine side - a remote stage process that runs its layers' verify windows, prompt
chunks, commits and session operations from a command stream, in place of the in-process `GpuStage` - needs two GPU
hosts to build and test and is not in this release.
