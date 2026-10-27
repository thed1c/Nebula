#!/usr/bin/env bash
# tools/rdma-setup.sh - RDMA for Impulse Flash-Next's RPC transport (impulse/flashnext/docs/RPC.md, include/strata/net/rdma.hpp).
# Run by the owner with sudo; Claude Code never runs it.  Every change it makes is recorded and `undo` takes back exactly those.
#
#   sudo bash tools/rdma-setup.sh install [--soft-roce IFACE] [--memlock USER]
#       install   rdma-core libibverbs-dev librdmacm-dev perftest ibverbs-utils (the headers let the engine build its RDMA
#                 backend; perftest's ib_write_lat / ib_write_bw measure a link); only packages not yet installed are recorded
#       --soft-roce IFACE   load rdma_rxe and add a soft-RoCE device rxe0 on an Ethernet port (e.g. eno1), so the RDMA path
#                 can be tested before the ConnectX cables / the test server's card exist.  It changes no IP setting of the port.
#       --memlock USER      let USER lock unlimited memory (the RDMA rings are registered memory): a limits.d file; takes
#                 effect at the user's next login (a systemd unit needs LimitMEMLOCK=infinity itself)
#   sudo bash tools/rdma-setup.sh status     what is installed / present (no changes)
#   sudo bash tools/rdma-setup.sh undo       remove the soft-RoCE device, unload rdma_rxe (if this script loaded it), remove
#                 the limits file, and remove the packages this script installed
#
# After install (as the user, in the worktree):
#   cmake -S impulse/flashnext -B impulse/flashnext/build-cpu -DSTRATA_RDMA=ON && cmake --build impulse/flashnext/build-cpu -j
#   impulse/flashnext/build-cpu/rdma_test            # loopback on rxe0 (or an active ConnectX port)
#   ib_write_lat -d rxe0 &  ib_write_lat -d rxe0 127.0.0.1   # perftest's own check
set -euo pipefail

STATE=/var/lib/strata-rdma-setup
PKGS=(rdma-core libibverbs-dev librdmacm-dev perftest ibverbs-utils)
LIMITS=/etc/security/limits.d/90-strata-rdma.conf

say() { printf '[rdma-setup] %s\n' "$*"; }
need_root() { [ "$(id -u)" -eq 0 ] || { say "run it with sudo"; exit 1; }; }

status() {
    say "packages:"
    for p in "${PKGS[@]}" ibverbs-providers libibverbs1 librdmacm1t64; do
        printf '  %-18s %s\n' "$p" "$(dpkg-query -W -f='${Status} ${Version}' "$p" 2>/dev/null | sed 's/install ok installed/installed/' || echo 'not installed')"
    done
    say "headers: $(ls /usr/include/infiniband/verbs.h /usr/include/rdma/rdma_cma.h 2>/dev/null | tr '\n' ' ' || true)"
    say "modules: $(lsmod | awk '/^(mlx4_ib|mlx5_ib|rdma_rxe|ib_uverbs|rdma_ucm)/{printf "%s ", $1}')"
    say "devices:"
    if command -v ibv_devinfo >/dev/null; then ibv_devinfo -l 2>/dev/null | sed 's/^/  /' || true; fi
    for d in /sys/class/infiniband/*; do
        [ -e "$d" ] || continue
        for p in "$d"/ports/*; do
            printf '  %s port %s: %s, %s, %s\n' "$(basename "$d")" "$(basename "$p")" "$(cat "$p/link_layer")" "$(cat "$p/state")" "$(cat "$p/rate" 2>/dev/null)"
        done
    done
    if command -v rdma >/dev/null; then say "rdma links:"; rdma link show 2>/dev/null | sed 's/^/  /' || true; fi
    say "this script's record: $(ls "$STATE" 2>/dev/null | tr '\n' ' ' || echo none)"
    [ -f "$LIMITS" ] && say "memlock: $(grep -v '^#' "$LIMITS" | tr '\n' ' ')"
    return 0
}

install() {
    need_root
    local iface="" user=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --soft-roce) iface="${2:?--soft-roce IFACE}"; shift 2 ;;
            --memlock) user="${2:?--memlock USER}"; shift 2 ;;
            *) say "unknown option $1"; exit 2 ;;
        esac
    done
    mkdir -p "$STATE"
    local new=()
    for p in "${PKGS[@]}"; do
        dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || new+=("$p")
    done
    if [ ${#new[@]} -gt 0 ]; then
        say "installing: ${new[*]}"
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${new[@]}"
        printf '%s\n' "${new[@]}" >> "$STATE/installed-packages"
    else
        say "packages already installed"
    fi
    if [ -n "$iface" ]; then
        [ -e "/sys/class/net/$iface" ] || { say "no network interface $iface"; exit 2; }
        if ! lsmod | grep -q '^rdma_rxe'; then
            modprobe rdma_rxe
            touch "$STATE/loaded-rdma_rxe"
        fi
        if rdma link show rxe0/1 >/dev/null 2>&1; then
            say "rxe0 exists already"
        else
            rdma link add rxe0 type rxe netdev "$iface"
            echo "$iface" > "$STATE/soft-roce-rxe0"
            say "soft-RoCE rxe0 on $iface added"
        fi
    fi
    if [ -n "$user" ]; then
        id "$user" >/dev/null 2>&1 || { say "no user $user"; exit 2; }
        printf '# tools/rdma-setup.sh (Impulse Flash-Next RPC): RDMA rings are locked memory\n%s - memlock unlimited\n' "$user" > "$LIMITS"
        touch "$STATE/limits"
        say "memlock unlimited for $user from the next login ($LIMITS)"
    fi
    status
}

undo() {
    need_root
    if [ -f "$STATE/soft-roce-rxe0" ]; then
        rdma link delete rxe0 2>/dev/null && say "rxe0 removed" || say "rxe0 was not there"
        rm -f "$STATE/soft-roce-rxe0"
    fi
    if [ -f "$STATE/loaded-rdma_rxe" ]; then
        modprobe -r rdma_rxe 2>/dev/null && say "rdma_rxe unloaded" || say "rdma_rxe still in use: left loaded"
        rm -f "$STATE/loaded-rdma_rxe"
    fi
    if [ -f "$STATE/limits" ]; then
        rm -f "$LIMITS" "$STATE/limits"
        say "memlock limits file removed"
    fi
    if [ -s "$STATE/installed-packages" ]; then
        mapfile -t pk < <(sort -u "$STATE/installed-packages")
        say "removing the packages this script installed: ${pk[*]}"
        DEBIAN_FRONTEND=noninteractive apt-get remove -y "${pk[@]}"
        rm -f "$STATE/installed-packages"
    fi
    rmdir "$STATE" 2>/dev/null || true
    status
}

case "${1:-status}" in
    install) shift; install "$@" ;;
    status) status ;;
    undo) undo ;;
    *) sed -n '2,22p' "$0"; exit 2 ;;
esac
