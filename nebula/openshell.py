"""nebula/openshell.py: NVIDIA OpenShell, seen and used from Nebula (its own Switchyard: sandboxes).

Three things, all through the `openshell` CLI (nothing else is assumed about a gateway):

- **The panel** (read only, except stopping a sandbox Nebula started): the registered gateways (`gateway list -o
  json`), the sandboxes (`sandbox list -o json`), the providers (`provider list -o json`) and a sandbox's effective
  policy (`policy get <name> --full -o json`). Policies are shown, never written from here.
- **Nebula as a model provider for sandboxes:** a provider profile that names only this Nebula's API Server, at
  `host.openshell.internal:<port>` (OpenShell's name, inside a sandbox, for the machine its gateway runs on), with a
  caller's key as the credential. A sandbox given that provider can reach Nebula and nothing else for models, and its
  calls are that caller's (its own ticks, its own cache: docs/sharing.md).
- **An engine deployed into a sandbox** (an engine's `"where": "openshell"`): the package's image (`image`), or for a
  package with none to pull (Impulse) an image built here from the engine's own program (`engine_image`), its cards
  passed in (NVIDIA CDI: `--gpu` with `cdi_devices`), its `env` settings, the models folder mounted read-only as a named volume, a
  policy that allows nothing out, and the engine's port exposed through the gateway (`service expose`), which is the
  address Nebula routes to. Stopping it deletes the sandbox.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

HOST = "host.openshell.internal"
# An engine's policy: nothing out (an engine needs no network), the system read-only, the models (and a Hugging Face
# cache, and its own key) read-only, a scratch folder, and a user that is not root. Checked on OpenShell 0.1.4
# (2026-10-20): the process cannot list /, write /etc or /models, or reach a host.
def policy_text(uid: int | None = None, devices: list[str] | None = None) -> str:
    """The policy for an engine's sandbox. Its user: the owner's own (not root) when the engine has a key to read (a
    file only the owner can read), else 65534 (nobody). `devices`: the card's device nodes it may open (the Podman
    driver leaves them out of the sandbox's file rules; without them CUDA finds no card)."""
    u = str(uid if uid and uid > 0 else 65534)
    rw = ", ".join(["/sandbox", "/tmp", "/dev/null", "/dev/shm"] + [d for d in devices or [] if re.match(r"^/dev/[A-Za-z0-9/_-]+$", d)])
    return f"""version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /lib64, /bin, /proc, /sys, /dev/urandom, /etc, /models, /folders, /hf, /keys, /app, /opt]
  read_write: [{rw}]
landlock:
  compatibility: best_effort
process:
  run_as_user: "{u}"
  run_as_group: "{u if u == '65534' else os.getgid()}"
network_policies: {{}}
"""


DENY_ALL = policy_text()


class OpenShellError(RuntimeError):
    pass


def binary(name: str | None = None) -> str | None:
    b = name or os.environ.get("NEBULA_OPENSHELL") or "openshell"
    return shutil.which(b) or (b if os.path.isabs(b) and os.access(b, os.X_OK) else None)


def run(args: list[str], exe: str | None = None, timeout: float = 120, gateway: str | None = None) -> str:
    """`openshell <args>`; `gateway` names a registered gateway other than the default (a model machine's)."""
    b = binary(exe)
    if b is None:
        raise OpenShellError("openshell is not installed on this machine")
    env = dict(os.environ, NO_COLOR="1", TERM="dumb")
    if gateway:
        env["OPENSHELL_GATEWAY"] = gateway
    try:
        p = subprocess.run([b] + args, capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise OpenShellError(f"openshell {' '.join(args[:2])} did not finish in {int(timeout)} s") from None
    if p.returncode != 0:
        raise OpenShellError((p.stderr or p.stdout or f"openshell {' '.join(args[:2])} failed").strip()[:500])
    return p.stdout


def _json(args, exe=None):
    out = run(args + ["-o", "json"], exe)
    try:
        return json.loads(out) if out.strip() else []
    except ValueError:
        raise OpenShellError(f"openshell {' '.join(args[:2])} did not answer JSON") from None


_READY: dict = {}                    # the openshell binary -> {"at", "why"}


def why_not(exe: str | None = None, max_age_s: float = 60.0) -> str | None:
    """None when engines can go into sandboxes here; else why not, in a few words (checked at most once a minute)."""
    import time
    if os.name == "nt":
        return "OpenShell sandboxes run on Linux; on Windows the engine runs as a plain process"
    b = binary(exe)
    if b is None:
        return "OpenShell is not installed on this machine"
    seen = _READY.get(b)
    if seen and time.time() - seen["at"] < max_age_s:
        return seen["why"]
    try:
        out = run(["status"], exe, timeout=15)
        why = None if "Connected" in out else "the OpenShell gateway is not running"
    except OpenShellError:
        why = "the OpenShell gateway is not running"
    _READY[b] = {"at": time.time(), "why": why}
    return why


def panel(exe: str | None = None) -> dict:
    if binary(exe) is None:
        return {"available": False, "why": "openshell is not installed on this machine"}
    out = {"available": True}
    for key, args in (("gateways", ["gateway", "list"]), ("sandboxes", ["sandbox", "list"]), ("providers", ["provider", "list"])):
        try:
            out[key] = _json(args, exe)
        except OpenShellError as e:
            out[key], out[key + "_error"] = [], str(e)
    return out


def policy(name: str, exe: str | None = None) -> dict:
    return {"sandbox": name, "policy": _json(["policy", "get", name, "--full"], exe), "editable": False}


def delete(name: str, exe: str | None = None, gateway: str | None = None) -> None:
    run(["sandbox", "delete", name], exe, gateway=gateway)


# --- Nebula as a provider for sandboxes
def profile(port: int, profile_id: str = "nebula", binaries: list[str] | None = None) -> str:
    """The provider profile (YAML) that names only this Nebula's API Server, with a caller's key as its credential."""
    bins = binaries or ["/usr/bin/curl", "/usr/local/bin/curl", "/usr/bin/python3", "/usr/local/bin/python3", "/sandbox/.venv/**"]
    return "\n".join([
        f"id: {profile_id}",
        "display_name: Nebula",
        "description: This machine's Nebula (OpenAI- and Anthropic-compatible API), one caller's key",
        "category: inference",
        "inference_capable: true",
        "credentials:",
        "  - name: api_key",
        "    description: the Nebula caller key (made in Nebula's Callers tab)",
        "    env_vars: [NEBULA_API_KEY]",
        "    required: true",
        "    auth_style: bearer",
        "    header_name: authorization",
        "endpoints:",
        f"  - host: {HOST}",
        f"    port: {int(port)}",
        "    protocol: rest",
        "    access: read-write",
        "    enforcement: enforce",
        "binaries: [" + ", ".join(bins) + "]",
        ""])


def provider_steps(port: int, provider: str = "nebula", profile_id: str = "nebula") -> list[str]:
    return [f"openshell provider profile lint -f {profile_id}.yaml",
            f"openshell provider profile import -f {profile_id}.yaml",
            f"openshell provider create --name {provider} --type {profile_id} --credential NEBULA_API_KEY=<the caller's key>",
            f"openshell sandbox create --provider {provider} --env OPENAI_BASE_URL=http://{HOST}:{int(port)}/v1 "
            f"--env ANTHROPIC_BASE_URL=http://{HOST}:{int(port)} -- <your agent>"]


# --- an engine in a sandbox
NAME_MAX = 19                        # OpenShell 0.1.4 refuses a longer sandbox name ("name exceeds maximum length")
ENGINE_PREFIX = "nebeng-"            # the sandboxes Nebula starts for its engines (the only ones it stops)
OLD_ENGINE_PREFIX = "nebula-engine-"  # before 2026-10-21; still known as Nebula's


def short_name(prefix: str, name: str, empty: str = "x") -> str:
    """prefix + the name made safe, within NAME_MAX: a longer name keeps its start and gets four characters of its
    hash, so two long names stay apart."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or empty
    room = NAME_MAX - len(prefix)
    if len(slug) <= room:
        return prefix + slug
    return prefix + slug[:room - 5].rstrip("-") + "-" + hashlib.sha256(slug.encode()).hexdigest()[:4]


def sandbox_name(engine: str) -> str:
    return short_name(ENGINE_PREFIX, engine)


def is_engine_sandbox(name: str) -> bool:
    return str(name).startswith((ENGINE_PREFIX, OLD_ENGINE_PREFIX))


# --- the machine's NUMA layout, for an engine whose CPUs span two sockets
# Inside a sandbox an engine may not see /sys/devices/system/node, and OpenShell refuses set_mempolicy, so the engine
# cannot always find its sockets by itself: Nebula reads the layout on the machine and states it (IMPULSE_NUMA_NODES,
# "0=0-17,36-53;1=18-35,54-71"), restricted to the CPUs the engine may use. Impulse and Strata then split their CPU
# expert pool per socket, each socket's threads placing and reading their own half of every expert.
NODE_CPUS_CMD = 'for f in /sys/devices/system/node/node*/cpulist; do [ -r "$f" ] && echo "$f $(cat "$f")"; done; true'


def parse_cpulist(text: str) -> list[int]:
    """"0-3,8" -> [0, 1, 2, 3, 8]; anything else -> []."""
    out: list[int] = []
    for part in (text or "").replace(" ", "").split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            return []
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if hi < lo:
            return []
        out += range(lo, hi + 1)
    return out


def cpulist_text(cpus) -> str:
    """[0, 1, 2, 3, 8] -> "0-3,8"."""
    xs, parts = sorted(set(cpus)), []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        parts.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(parts)


def node_cpus(text: str) -> dict[int, list[int]]:
    """The NUMA nodes that have CPUs, from NODE_CPUS_CMD's lines ("/sys/.../node1/cpulist 18-35,54-71")."""
    out: dict[int, list[int]] = {}
    for line in (text or "").splitlines():
        m = re.match(r"\S*/node(\d+)/cpulist\s+(\S+)\s*$", line.strip())
        if m and parse_cpulist(m.group(2)):
            out[int(m.group(1))] = parse_cpulist(m.group(2))
    return out


def local_node_cpus() -> dict[int, list[int]]:
    """This machine's NUMA nodes with CPUs."""
    try:
        return node_cpus(subprocess.run(["sh", "-c", NODE_CPUS_CMD], capture_output=True, text=True, timeout=10).stdout)
    except (OSError, subprocess.SubprocessError):
        return {}


def taskset_cpus(argv: list[str]) -> list[int] | None:
    """The CPUs a command is held to by its own taskset prefix ("taskset -c 18-35,54-71 ..."), or None."""
    if len(argv) > 2 and os.path.basename(argv[0]) == "taskset" and argv[1] in ("-c", "--cpu-list"):
        return parse_cpulist(argv[2]) or None
    return None


def numa_env(spec: dict, inner_cmd: list[str], nodes: dict[int, list[int]] | None) -> dict:
    """The settings Nebula adds for the engine's sockets: IMPULSE_NUMA_NODES when the CPUs the engine may use (its taskset,
    else every CPU) lie on two or more NUMA nodes; nothing on a one-socket machine or for an engine held to one socket.
    `"numa": "off"` turns the per-socket pool off instead (STRATA_NUMA_POOL=0). The engine's own env always wins."""
    if spec.get("numa") == "off":
        return {"STRATA_NUMA_POOL": "0"}
    if not nodes or "IMPULSE_NUMA_NODES" in (spec.get("env") or {}):
        return {}
    held = taskset_cpus(inner_cmd)
    mine = {n: [c for c in cpus if held is None or c in held] for n, cpus in sorted(nodes.items())}
    mine = {n: cpus for n, cpus in mine.items() if cpus}
    if len(mine) < 2:
        return {}
    return {"IMPULSE_NUMA_NODES": ";".join(f"{n}={cpulist_text(cpus)}" for n, cpus in mine.items())}


def deploy_argv(spec: dict, inner_cmd: list[str], card_index, volume: str, policy_file: str,
                extra: list | None = None, env: dict | None = None, memory: str | None = None,
                cpus: int | None = None, nodes: dict[int, list[int]] | None = None) -> list[str]:
    """`openshell sandbox create` for an engine: its image, its cards (an index, or a list of them), the models
    read-only (and any other folder it reads, such as the Hugging Face cache: [(volume, target)]), its environment
    settings (with the machine's socket layout: numa_env; `nodes` is the machine's, this one's when not given),
    nothing out."""
    driver: dict = {"docker": {"mounts": [{"type": "volume", "source": volume, "target": "/models", "read_only": True}] +
                               [{"type": "volume", "source": v, "target": t, "read_only": True} for v, t in (extra or [])]}}
    from nebula import packages
    image = spec.get("image") or (packages.PACKAGES.get(spec.get("kind")) or {}).get("image") or ""
    argv = ["sandbox", "create", "--name", sandbox_name(spec["name"]), "--from", image, "--policy", policy_file]
    idx = [] if card_index is None else list(card_index) if isinstance(card_index, (list, tuple)) else [card_index]
    if idx:
        driver["docker"]["cdi_devices"] = [f"nvidia.com/gpu={i}" for i in idx]
        argv += ["--gpu"] if len(idx) == 1 else [f"--gpu={len(idx)}"]     # its count must match the CDI devices
    env = {**numa_env(spec, inner_cmd, local_node_cpus() if nodes is None else nodes), **(env or {})}
    for k, v in sorted(env.items()):
        argv += ["--env", f"{k}={v}"]
    # its memory limit: the engine's own, else the whole machine's RAM (a sandbox is otherwise capped at 4 GiB, which
    # a large mixture-of-experts model's experts in RAM, or an engine's RAM cache, need far more than)
    argv += ["--memory", str(spec.get("memory") or memory or f"{ram_gib()}Gi")]
    # and its CPU limit: the engine's own, else every core of the machine (a sandbox is otherwise held to 2 cores,
    # which starves an engine whose experts run on CPU workers: 2.8 tok/s against 84 in one test)
    argv += ["--cpu", str(spec.get("cpu_limit") or cpus or os.cpu_count() or 2)]
    return argv + ["--driver-config-json", json.dumps(driver, separators=(",", ":")), "--"] + inner_cmd


def ram_gib(meminfo: str | None = None) -> int:
    """This machine's RAM in GiB (MemTotal), or of the machine whose /proc/meminfo is given."""
    try:
        text = meminfo if meminfo is not None else open("/proc/meminfo", encoding="utf-8").read()
        kb = int(re.search(r"MemTotal:\s+(\d+)", text).group(1))
        return max(1, kb // (1024 * 1024))
    except (OSError, AttributeError, ValueError):
        return 64


def sandbox_image(image: str, docker: str | None, lib_dir: str | None = None) -> str:
    """The image an engine's sandbox runs from. OpenShell runs it as an unprivileged user in the image's working
    folder, which that user must be able to write (OpenShell 0.1.4): an image whose folder is another (llama.cpp's
    is /app) gets a thin local copy whose working folder is /sandbox, built once (<image>-nebula-sandbox). The
    engine's own folder goes on the library path: OpenShell starts it in a way that loses its $ORIGIN lookup."""
    if not docker:
        return image
    p = subprocess.run([docker, "image", "inspect", image, "--format", "{{.Config.WorkingDir}}"], capture_output=True, text=True)
    if p.returncode != 0 or p.stdout.strip() in ("", "/", "/sandbox"):
        return image
    tag = image.replace(":", "-").replace("/", "-") + ":nebula-sandbox-2"
    if subprocess.run([docker, "image", "inspect", tag], capture_output=True).returncode != 0:
        env = subprocess.run([docker, "image", "inspect", image, "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
                             capture_output=True, text=True).stdout
        ld = next((ln.split("=", 1)[1] for ln in env.splitlines() if ln.startswith("LD_LIBRARY_PATH=")), "")
        lines = [f"FROM {image}", "WORKDIR /sandbox"]
        if lib_dir:
            lines.append(f"ENV LD_LIBRARY_PATH={lib_dir}" + (f":{ld}" if ld else ""))
        subprocess.run([docker, "build", "-q", "-t", tag, "-"], input="\n".join(lines) + "\n", text=True,
                       capture_output=True, check=True)
    return tag


# --- an engine's own image, for a package with none to pull (Impulse)
BASE_IMAGE = "ubuntu:24.04"          # what it is built on; the installer fetches it with OpenShell's images
ENGINE_DIR = "/opt/engine"           # where the image keeps the program, and its libraries in lib/
# what the image never carries: the C library (the base image has its own) and the NVIDIA driver's libraries (the
# card's CDI devices bring the machine's own, which match its driver)
_SKIP_LIB = re.compile(r"^(linux-vdso|ld-linux|libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so|libresolv\.so|"
                       r"libutil\.so|libcuda\.so|libnvidia-|libnvcuvid|libnvoptix)")


def _elf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def engine_program(binary: str) -> str:
    """The program itself: the file, or for the installer's wrapper script (impulse/impulse-server, which sets the
    library path and runs impulse-server.bin) the program beside it."""
    real = os.path.realpath(binary)
    if _elf(real):
        return real
    for cand in (real + ".bin", os.path.splitext(real)[0] + ".bin"):
        if _elf(cand):
            return cand
    return real


def _lib_path(binary: str) -> str:
    """Where the program's libraries are found, as its wrapper would: beside it, then NVIDIA's pip packages in the
    installer's Python (python/lib/python3.12/site-packages/nvidia/cu13/lib beside impulse/), then this Python's."""
    import glob
    import sysconfig
    d = os.path.dirname(os.path.realpath(binary))
    dirs = [d] + glob.glob(os.path.join(d, "..", "python", "lib", "python3*", "site-packages", "nvidia", "cu1*", "lib"))
    dirs += glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu1*", "lib"))
    return ":".join([os.path.realpath(x) for x in dirs if os.path.isdir(x)] + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else []))


def engine_image_tag(binary: str) -> str:
    """The image's tag: the program's name and the start of its sha256, so each build of it gets its own image."""
    real = engine_program(binary)
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    name = re.sub(r"[^a-z0-9_.-]+", "-", os.path.basename(os.path.realpath(binary)).lower()).strip("-.") or "engine"
    return f"nebula-engine-{name}:{h.hexdigest()[:12]}"


def engine_libs(binary: str) -> list[tuple[str, str]]:
    """[(soname, the file here)]: the shared libraries the program loads that its image must carry (read with ldd:
    CUDA's runtime and cuBLAS, OpenMP, the C++ library). A library ldd cannot find is refused, by name."""
    try:
        p = subprocess.run(["ldd", engine_program(binary)], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, LD_LIBRARY_PATH=_lib_path(binary)))
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = p.stdout + p.stderr
    if p.returncode != 0 or "not a dynamic executable" in out:
        return []
    gone = re.findall(r"^\s*(\S+)\s+=>\s+not found", out, re.M)
    if gone:
        raise OpenShellError(f"{os.path.basename(binary)} needs {gone[0]}, which this machine does not have")
    return [(m.group(1), m.group(2)) for m in re.finditer(r"^\s*(\S+)\s+=>\s+(/\S+)", out, re.M)
            if not _SKIP_LIB.match(m.group(1))]


def engine_image(binary: str, docker: str, base: str = BASE_IMAGE, say=None) -> str:
    """The engine's own image, built here once per build of its program: the base image with the program in
    /opt/engine and the libraries it loads in /opt/engine/lib (on the library path), its working folder /sandbox.
    Nothing is downloaded but the base image, once."""
    tag = engine_image_tag(binary)
    if subprocess.run([docker, "image", "inspect", tag], capture_output=True).returncode == 0:
        return tag
    real, name = engine_program(binary), os.path.basename(os.path.realpath(binary))
    libs = engine_libs(binary)
    if say:
        say(f"[nebula] building its image {tag} from {real} and {len(libs)} libraries (once for this build)")
    with tempfile.TemporaryDirectory(prefix="nebula-engine-image-") as ctx:
        lib = os.path.join(ctx, "engine", "lib")
        os.makedirs(lib)
        shutil.copyfile(real, os.path.join(ctx, "engine", name))           # the program under its wrapper's name
        os.chmod(os.path.join(ctx, "engine", name), 0o755)
        for soname, path in libs:
            shutil.copyfile(os.path.realpath(path), os.path.join(lib, soname))
            os.chmod(os.path.join(lib, soname), 0o644)
        with open(os.path.join(ctx, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(f"FROM {base}\nCOPY engine/ {ENGINE_DIR}/\nENV LD_LIBRARY_PATH={ENGINE_DIR}/lib\nWORKDIR /sandbox\n"
                    f"LABEL ai.nebula.engine={name} ai.nebula.engine.sha256={tag.rsplit(':', 1)[1]}\n")
        p = subprocess.run([docker, "build", "-q", "-t", tag, ctx], capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            raise OpenShellError(f"its image could not be built: {(p.stderr or p.stdout).strip()[-300:]}")
    return tag


def volume_name(models_dir: str) -> str:
    """The models folder's own volume: one per folder, so two Nebulas (or a moved folder) never share a stale one."""
    import hashlib
    return "nebula-models-" + hashlib.sha256(os.path.realpath(models_dir).encode()).hexdigest()[:12]


def volume_argv(models_dir: str, volume: str | None = None) -> list[str]:
    volume = volume or volume_name(models_dir)
    """The named Docker volume over the models folder, labelled so OpenShell may attach it (Docker driver)."""
    return ["docker", "volume", "create", "--label", "openshell.ai/sandbox-attachable=true",
            "--label", "openshell.ai/sandbox-attachable-workspace=default",
            "--driver", "local", "--opt", "type=none", "--opt", "o=bind", "--opt", f"device={models_dir}", volume]


def expose(name: str, port: int, exe: str | None = None, keyed: bool = False, gateway: str | None = None) -> str:
    """The engine's address through the gateway. A keyed engine gets the caller's Authorization header passed through
    (the gateway strips it otherwise), so only Nebula, which holds its key, can use it."""
    out = run(["service", "expose", name, str(int(port))] + (["--authorization-mode", "bearer-passthrough"] if keyed else []), exe,
              gateway=gateway)
    m = re.search(r"https?://\S+", out)
    if not m:
        raise OpenShellError(f"openshell did not say where {name}:{port} is exposed")
    return m.group(0).rstrip("/")


_DRIVER: dict = {}


def exited(name: str, exe: str | None = None, gateway: str | None = None) -> str | None:
    """Why the sandbox's main process is gone (it exited, or the sandbox failed), or None while it runs or loads."""
    try:
        d = json.loads(run(["sandbox", "get", name, "-o", "json"], exe, timeout=30, gateway=gateway) or "{}")
    except (OpenShellError, ValueError):
        return None
    d = d.get("sandbox") if isinstance(d.get("sandbox"), dict) else d
    if d.get("exit_code") is not None:
        return f"its program exited (code {d['exit_code']})"
    if str(d.get("phase") or "").lower() in ("error", "failed", "exited", "terminated", "stopped"):
        return f"its sandbox is {d['phase']}"
    return None


def gateway_driver(exe: str | None = None, gateway: str | None = None, max_age_s: float = 60.0) -> str | None:
    """The compute driver the gateway runs sandboxes on ("docker", "podman"), from `openshell gateway info`."""
    import time
    key = (exe, gateway)
    at, drv = _DRIVER.get(key, (0.0, None))
    if drv and time.time() - at < max_age_s:
        return drv
    try:
        out = run(["gateway", "info"], exe, timeout=30, gateway=gateway)
    except OpenShellError:
        return None
    m = re.search(r"Compute drivers:\s*\n\s*(\w+)", out)
    drv = m.group(1).lower() if m else None
    _DRIVER[key] = (time.time(), drv)
    return drv


def device_nodes(cdi_json: str, indices) -> list[str]:
    """The NVIDIA device nodes of these cards (a card's index is not its /dev/nvidiaN) and the shared ones, from a CDI
    description: the Podman driver leaves them out of a sandbox's file rules."""
    try:
        d = json.loads(cdi_json or "{}")
    except ValueError:
        return []
    want = {str(i) for i in indices or []}
    if not want:
        return []
    out = [n["path"] for n in (d.get("containerEdits") or {}).get("deviceNodes") or [] if n.get("path", "").startswith("/dev/nvidia")]
    for e in d.get("devices") or []:
        if str(e.get("name")) in want:
            out += [n["path"] for n in (e.get("containerEdits") or {}).get("deviceNodes") or [] if n.get("path", "").startswith("/dev/nvidia")]
    return list(dict.fromkeys(out))


def local_cdi_json() -> str:
    try:
        return open("/etc/cdi/nvidia-podman.json", encoding="utf-8").read()
    except OSError:
        p = subprocess.run(["nvidia-ctk", "cdi", "generate", "--format=json"], capture_output=True, text=True, timeout=60) \
            if shutil.which("nvidia-ctk") else None
        return p.stdout if p is not None and p.returncode == 0 else ""


class SandboxedEngine:
    """An engine run in an OpenShell sandbox, with the same face as nebula/app/engine.py's EngineProcess (start, stop,
    wait_ready, state, status), so nebula/engines.py runs it like a local one."""

    def __init__(self, spec: dict, inner_cmd: list[str], models_dir: str, policy_dir: str, card_index,
                 exe: str | None = None, ready_timeout_s: float = 900.0, extra_dirs: list | None = None, keyed: bool = False,
                 env: dict | None = None, build_from: str | None = None):
        self.spec, self.inner_cmd, self.models_dir, self.policy_dir = spec, inner_cmd, models_dir, policy_dir
        self.extra_dirs = list(extra_dirs or [])                 # [(a folder here, where the sandbox sees it)], read-only
        self.keyed = keyed                                        # its own key at /keys/key (the owner's user reads it)
        self.card_index, self.exe, self.ready_timeout_s = card_index, exe, ready_timeout_s   # an index or a list of them
        self.env = dict(env or {})                                # the engine's settings, passed into the sandbox
        self.build_from = build_from                              # its program here, when its image is built from it
        self.name = sandbox_name(spec["name"])
        self.address: str | None = None
        self.state, self.error, self.proc = "idle", None, None
        self.lines: list[str] = []
        self.started_at = self.ready_at = None

    def _say(self, line: str) -> None:
        self.lines = (self.lines + [line])[-400:]

    def start(self) -> None:
        import time
        self.state, self.started_at = "starting", time.time()
        try:
            if gateway_driver(self.exe) == "podman":           # the gateway runs sandboxes on rootless Podman
                return self._start_podman()
            docker = os.environ.get("NEBULA_DOCKER") or shutil.which("docker")
            if docker:
                for d in [self.models_dir] + [h for h, _t in self.extra_dirs]:
                    have = subprocess.run([docker, "volume", "inspect", volume_name(d)], capture_output=True).returncode == 0
                    if not have:
                        self._say("[nebula] a folder as a read-only volume: " + " ".join(volume_argv(d)))
                        subprocess.run([docker] + volume_argv(d)[1:], capture_output=True, check=True)
            os.makedirs(self.policy_dir, exist_ok=True)
            pol = os.path.join(self.policy_dir, f"{self.name}.yaml")
            with open(pol, "w", encoding="utf-8") as f:
                f.write(policy_text(os.getuid() if self.keyed else None))
            argv = deploy_argv(self.spec, self.inner_cmd, self.card_index, volume_name(self.models_dir), pol,
                               [(volume_name(h), t) for h, t in self.extra_dirs], self.env)
            if self.build_from:
                if not docker:
                    raise OpenShellError("its image is built here from its program, which needs Docker")
                argv[argv.index("--from") + 1] = engine_image(self.build_from, docker, say=self._say)
            if docker:
                i = argv.index("--from") + 1
                prog = next((w for w in self.inner_cmd if w.startswith("/")), None)       # after a taskset prefix
                argv[i] = sandbox_image(argv[i], docker, os.path.dirname(prog) if prog else None)
            argv.insert(argv.index("--"), "--detach")
            self._say("[nebula] openshell " + " ".join(argv))
            run(argv, self.exe, timeout=600)
            self.address = expose(self.name, int(self.spec["port"]), self.exe, keyed=self.keyed)
            self._say(f"[nebula] exposed at {self.address}")
            self.state = "loading"
        except (OpenShellError, OSError, subprocess.CalledProcessError) as e:
            self.state, self.error = "failed", str(e)[:400]
            self._say(f"[nebula] {self.error}")

    def _start_podman(self) -> None:
        """As start(), on a gateway whose sandboxes run on rootless Podman: the folders as Podman volumes, the image
        (built with Docker when it is built here) loaded into Podman once, the card's device nodes in its file rules."""
        podman = shutil.which("podman")
        if not podman:
            raise OpenShellError("the gateway runs on Podman, which is not installed for this user")
        for d in [self.models_dir] + [h for h, _t in self.extra_dirs]:
            vol = volume_name(d)
            if subprocess.run([podman, "volume", "exists", vol], capture_output=True).returncode != 0:
                self._say("[nebula] a folder as a read-only volume: podman " + " ".join(volume_argv(d)[1:]))
                subprocess.run([podman] + volume_argv(d)[1:], capture_output=True, check=True)
        os.makedirs(self.policy_dir, exist_ok=True)
        pol = os.path.join(self.policy_dir, f"{self.name}.yaml")
        with open(pol, "w", encoding="utf-8") as f:
            f.write(policy_text(os.getuid() if self.keyed else None, device_nodes(local_cdi_json(), self.card_index if isinstance(self.card_index, (list, tuple)) else ([] if self.card_index is None else [self.card_index]))))
        argv = deploy_argv(self.spec, self.inner_cmd, self.card_index, volume_name(self.models_dir), pol,
                           [(volume_name(h), t) for h, t in self.extra_dirs], self.env)
        i = argv.index("--driver-config-json") + 1
        argv[i] = json.dumps({"podman": json.loads(argv[i])["docker"]}, separators=(",", ":"))
        if self.build_from:
            docker = os.environ.get("NEBULA_DOCKER") or shutil.which("docker")
            if not docker:
                raise OpenShellError("its image is built here from its program, which needs Docker")
            tag = engine_image(self.build_from, docker, say=self._say)
            if subprocess.run([podman, "image", "exists", tag], capture_output=True).returncode != 0:
                self._say(f"[nebula] {tag} into Podman (once for this build)")
                save = subprocess.Popen([docker, "save", tag], stdout=subprocess.PIPE)
                subprocess.run([podman, "load", "-q"], stdin=save.stdout, capture_output=True, check=True, timeout=3600)
                save.wait(timeout=60)
            argv[argv.index("--from") + 1] = tag
        # Podman re-owns a named volume's folder to the first container that mounts it (a sandbox runs as "nobody",
        # which can lock the owner out of the folder): a first mount as root settles it
        img = argv[argv.index("--from") + 1]
        for d in [self.models_dir] + [h for h, _t in self.extra_dirs]:
            subprocess.run([podman, "run", "--rm", "--network", "none", "-v", f"{volume_name(d)}:/m:ro", "--entrypoint", "/bin/true", img],
                           capture_output=True, timeout=300)
        argv.insert(argv.index("--"), "--detach")
        self._say("[nebula] openshell " + " ".join(argv))
        try:
            run(argv, self.exe, timeout=600)
        except OpenShellError as e:                         # a sandbox left from an earlier try
            if "already exists" not in str(e):
                raise
            delete(self.name, self.exe)
            import time
            time.sleep(3)
            run(argv, self.exe, timeout=600)
        self.address = expose(self.name, int(self.spec["port"]), self.exe, keyed=self.keyed)
        self._say(f"[nebula] exposed at {self.address} (Podman)")
        self.state = "loading"

    def wait_ready(self, poll_s: float = 0.5) -> bool:
        import time
        from nebula.app.engine import health
        if self.state != "loading" or not self.address:
            return False
        end, look = time.monotonic() + self.ready_timeout_s, time.monotonic() + 10
        while time.monotonic() < end:
            if self.state in ("stopping", "stopped"):
                return False
            if time.monotonic() >= look:                   # a program that died while loading: said within seconds
                look = time.monotonic() + 10
                why = exited(self.name, self.exe)
                if why:
                    self.state, self.error = "failed", f"{self.name}: {why} while loading"
                    self._say(f"[nebula] {self.error}")
                    return False
            if health(self.address, timeout=1.0) == 200:
                self.state, self.ready_at = "ready", time.time()
                self._say("[nebula] the engine in the sandbox answers")
                return True
            time.sleep(poll_s)
        self.state, self.error = "failed", f"the engine in {self.name} did not answer within {self.ready_timeout_s:.0f} s"
        return False

    def exited(self) -> str | None:
        """Why its sandbox's program is gone, or None while it runs (openshell sandbox get)."""
        return exited(self.name, self.exe)

    def stop(self) -> None:
        self.state = "stopping"
        try:
            delete(self.name, self.exe)
        except OpenShellError as e:
            self._say(f"[nebula] {e}")
        self.state, self.address = "stopped", None

    def status(self) -> dict:
        return {"state": self.state, "lines": list(self.lines[-60:]), "pid": None, "error": self.error,
                "seconds": None, "sandbox": self.name, "address": self.address}
