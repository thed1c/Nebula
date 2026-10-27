#!/bin/bash
# The Safe sandbox's inside (run as root in Nebula's own WSL2 distro by greenfield/windows/sandbox.ps1):
#   bash sandbox-setup.sh <wsl.json> [<impulse-linux.tar.gz>] [<version>]
# Podman and NVIDIA's container toolkit at the versions wsl.json pins (apt checks every file against Ubuntu's and
# NVIDIA's signed indexes; NVIDIA's key by its fingerprint), the user "nebula" the containers run as (rootless), the
# card described to Podman (CDI, when Windows' NVIDIA driver shows through), and the engine's image from the Linux
# Impulse. Its last line is RESULT {json}; on a failure, FAILED <why>.
set -euo pipefail
PINS=$1 TAR=${2:-} VERSION=${3:-}
[ "$TAR" = "-" ] && TAR=""                         # "-": no Impulse (PowerShell 5.1 drops an empty argument)
fail() { echo "FAILED $*"; exit 1; }
pin() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$PINS" "$1"; }
export DEBIAN_FRONTEND=noninteractive
command -v python3 >/dev/null || fail "the distro has no python3"

echo "Podman ($(pin apt.podman)) and its helpers..."
apt-get update -qq || fail "apt could not reach Ubuntu's archive"
apt-get install -y -qq --no-install-recommends ca-certificates curl gpg "podman=$(pin apt.podman)" "passt=$(pin apt.passt)" \
  "uidmap=$(pin apt.uidmap)" >/dev/null || fail "Podman $(pin apt.podman) could not be installed"

echo "NVIDIA's container toolkit ($(pin nvidia.toolkit))..."
K=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "$(pin nvidia.key_url)" | gpg --dearmor > /tmp/nvidia.gpg || fail "NVIDIA's signing key could not be fetched"
FPR=$(gpg --show-keys --with-colons /tmp/nvidia.gpg 2>/dev/null | awk -F: '/^fpr/ { print $10; exit }')
[ "$FPR" = "$(pin nvidia.key_fingerprint)" ] || fail "NVIDIA's signing key is not the expected one ($FPR)"
install -m 0644 /tmp/nvidia.gpg "$K"
echo "deb [signed-by=$K] $(pin nvidia.repo) /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq || fail "apt could not reach NVIDIA's repository"
V=$(pin nvidia.toolkit)
apt-get install -y -qq --no-install-recommends "nvidia-container-toolkit=$V" "nvidia-container-toolkit-base=$V" \
  "libnvidia-container-tools=$V" "libnvidia-container1=$V" >/dev/null || fail "NVIDIA's container toolkit $V could not be installed"

echo "The user the engines run as..."
id nebula >/dev/null 2>&1 || useradd -m -s /bin/bash nebula
grep -q '^nebula:' /etc/subuid || usermod --add-subuids 100000-165535 --add-subgids 100000-165535 nebula
# At every start of the distro (wsl.conf's boot command, as root): "/" shared, which rootless Podman wants (else "/" is
# not a shared mount), and the engines' user's runtime folder /run/user/<uid>, which Podman runs from and nothing else
# makes here (no systemd-logind; WSLg makes one only for uid 1000)
UIDN=$(id -u nebula)
BOOT="mount --make-rshared / ; mkdir -p /run/user/$UIDN ; chown nebula:nebula /run/user/$UIDN ; chmod 700 /run/user/$UIDN"
printf '[user]\ndefault=nebula\n[interop]\nappendWindowsPath=false\n[boot]\ncommand = %s\n' "$BOOT" > /etc/wsl.conf
sh -c "$BOOT" 2>/dev/null || true
# Podman for Nebula: with the user's own runtime folder. Reached through `wsl --exec`, WSLg points XDG_RUNTIME_DIR at its
# /mnt/wslg/runtime-dir (uid 1000's), which Podman refuses ("is not owned by the current user")
printf '#!/bin/sh\nexport XDG_RUNTIME_DIR=/run/user/$(id -u)\nexec podman "$@"\n' > /usr/local/bin/nebula-podman
chmod 755 /usr/local/bin/nebula-podman

GPU=false
if [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
  echo "The NVIDIA card, described to Podman..."
  mkdir -p /etc/cdi
  nvidia-ctk cdi generate --mode=wsl --output=/etc/cdi/nvidia.yaml >/dev/null 2>&1 || fail "the card could not be described to Podman (nvidia-ctk cdi generate)"
  GPU=true
else
  echo "No NVIDIA driver shows through from Windows: engines run on the CPU in the sandbox."
fi

IMAGE=""
if [ -n "$TAR" ]; then
  IMAGE=$(pin impulse.image | sed "s/{version}/$VERSION/")
  echo "Impulse's image ($IMAGE)..."
  CTX=$(mktemp -d /tmp/nebula-image.XXXX)
  mkdir -p "$CTX/engine" && tar -xzf "$TAR" -C "$CTX/engine" || fail "Impulse's Linux folder could not be unpacked"
  [ -x "$CTX/engine/impulse-server" ] && [ -x "$CTX/engine/impulse-server.bin" ] || fail "Impulse's Linux folder has no impulse-server"
  printf 'FROM %s\nCOPY engine/ /opt/engine/\nRUN mkdir -p /sandbox && chmod 1777 /sandbox\nWORKDIR /sandbox\nLABEL ai.nebula.engine=impulse-server ai.nebula.version=%s\n' \
    "$(pin base_image)" "$VERSION" > "$CTX/Containerfile"
  chown -R nebula: "$CTX"
  su - nebula -c "XDG_RUNTIME_DIR=/run/user/$UIDN podman build -q -t '$IMAGE' '$CTX'" > /tmp/build.txt 2>&1 \
    || fail "Impulse's image could not be built: $(grep -v -i warning /tmp/build.txt | tail -1 | cut -c1-160)"
  rm -rf "$CTX"
  # (--help: impulse-server has no --version, it answers 2 with its usage) Impulse links the NVIDIA driver's libcuda.so.1, which a container has only with the card (CDI brings Windows'
  # driver libraries in): checked with the card, as engines run; without an NVIDIA card Impulse cannot run here at all
  if [ "$GPU" = true ]; then
    su - nebula -c "XDG_RUNTIME_DIR=/run/user/$UIDN podman run --rm --network none --device nvidia.com/gpu=all '$IMAGE' /opt/engine/impulse-server --help" > /tmp/imp.txt 2>&1 \
      || fail "Impulse does not start in its image: $(grep -v -i warning /tmp/imp.txt | tail -1 | cut -c1-160)"
  else
    echo "No NVIDIA card: Impulse's image is kept but not checked (it needs an NVIDIA card to start)."
  fi
fi

SMOKE=skipped
if [ "$GPU" = true ]; then
  if su - nebula -c "XDG_RUNTIME_DIR=/run/user/$UIDN podman run --rm --network none --device nvidia.com/gpu=all $(pin base_image) /usr/lib/wsl/lib/nvidia-smi -L" > /tmp/gpu.txt 2>&1 \
      || su - nebula -c "XDG_RUNTIME_DIR=/run/user/$UIDN podman run --rm --network none --device nvidia.com/gpu=all $(pin base_image) nvidia-smi -L" > /tmp/gpu.txt 2>&1; then
    SMOKE=$(grep -m1 '^GPU' /tmp/gpu.txt | cut -c1-120 || echo ok)
  else
    fail "a container does not see the NVIDIA card: $(tail -1 /tmp/gpu.txt | cut -c1-160)"
  fi
fi

PV=$(su - nebula -c "XDG_RUNTIME_DIR=/run/user/$UIDN podman --version" | awk '{print $3}')
python3 -c 'import json,sys; print("RESULT " + json.dumps({"podman": sys.argv[1], "gpu": sys.argv[2] == "true", "gpu_check": sys.argv[3],
  "images": ({"impulse": sys.argv[4]} if sys.argv[4] else {})}))' "$PV" "$GPU" "$SMOKE" "$IMAGE"
