#!/usr/bin/env python3
"""A stand-in for the `openshell` CLI, for the tests: `sandbox get/create/start/exec/delete` on a directory that
plays the sandbox (FAKE_OPENSHELL_HOME). `exec` runs the command on this machine with its cwd as the sandbox's
root, so the sandbox's work folders (/tmp/nebula-x) are <home>/<name>/tmp/nebula-x; every call is appended to calls.log."""
import json
import os
import subprocess
import sys


def main(argv):
    home = os.environ["FAKE_OPENSHELL_HOME"]
    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, "calls.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(argv) + "\n")
    if argv[:2] == ["sandbox", "get"]:
        name = argv[2]
        d = os.path.join(home, name)
        if not os.path.isdir(d):
            sys.stderr.write(f"sandbox {name} not found\n")
            return 1
        state = open(os.path.join(d, ".state"), encoding="utf-8").read().strip()
        sys.stdout.write(json.dumps({"name": name, "phase": state}) + "\n")       # as openshell 0.1.4 prints it
        return 0
    if argv[:2] == ["sandbox", "create"]:
        opts, i = {"env": [], "provider": []}, 2
        while i < len(argv) and argv[i] != "--":
            a = argv[i]
            if a in ("--name", "--from", "--policy", "--driver-config-json"):
                opts[a[2:]] = argv[i + 1]
                i += 2
            elif a.startswith("--gpu="):
                opts["gpu"] = int(a.split("=", 1)[1])
                i += 1
            elif a in ("--env", "--provider"):
                opts[a[2:]].append(argv[i + 1])
                i += 2
            else:
                opts[a[2:]] = True                  # --gpu, --detach
                i += 1
        d = os.path.join(home, opts["name"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, ".state"), "w", encoding="utf-8") as f:
            f.write("Ready")
        with open(os.path.join(d, ".spec"), "w", encoding="utf-8") as f:
            json.dump(opts, f)
        cmd = argv[i + 1:]
        if cmd and opts.get("detach") and cmd != ["sleep", "infinity"]:
            # the workload runs here; the sandbox's /models is the folder FAKE_OPENSHELL_MODELS names
            models = os.environ.get("FAKE_OPENSHELL_MODELS", "")
            cmd = [w.replace("/models/", models.rstrip("/") + "/", 1) if w.startswith("/models/") and models else w for w in cmd]
            # a program in an image built here (/opt/engine/...): the copy the fake docker kept of that image's folder
            ip = os.environ.get("FAKE_DOCKER_IMAGES")
            built = (json.load(open(ip)) if ip and os.path.exists(ip) else {}).get(opts.get("from"))
            if built:
                cmd = [os.path.join(built, "engine", w[len("/opt/engine/"):]) if w.startswith("/opt/engine/") else w for w in cmd]
            # its other mounts (/keys, /hf): the folders the fake docker recorded for their volumes
            vp = os.environ.get("FAKE_DOCKER_VOLUMES")
            vols = json.load(open(vp)) if vp and os.path.exists(vp) else {}
            for m in json.loads(opts.get("driver-config-json") or "{}").get("docker", {}).get("mounts", []):
                host = vols.get(m["source"])
                if host and m["target"] != "/models":
                    cmd = [host.rstrip("/") + w[len(m["target"]):] if w.startswith(m["target"] + "/") else w for w in cmd]
            env = dict(os.environ, **dict(e.split("=", 1) for e in opts["env"]))
            proc = subprocess.Popen(cmd, cwd=os.environ.get("FAKE_OPENSHELL_CWD") or d, stdin=subprocess.DEVNULL, env=env,
                                    stdout=open(os.path.join(d, ".log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)
            with open(os.path.join(d, ".pid"), "w", encoding="utf-8") as f:
                f.write(str(proc.pid))
        return 0
    if argv[:2] == ["sandbox", "list"]:
        rows = []
        for n in sorted(os.listdir(home)):
            if os.path.isdir(os.path.join(home, n)):
                rows.append({"name": n, "phase": open(os.path.join(home, n, ".state")).read().strip()})
        sys.stdout.write(json.dumps(rows) + "\n")
        return 0
    if argv[:2] == ["gateway", "list"]:
        sys.stdout.write(json.dumps([{"name": "local", "endpoint": "http://127.0.0.1:17670", "active": True}]) + "\n")
        return 0
    if argv[:2] == ["provider", "list"]:
        sys.stdout.write("[]\n")
        return 0
    if argv[:2] == ["policy", "get"]:
        spec = json.load(open(os.path.join(home, argv[2], ".spec")))
        sys.stdout.write(json.dumps({"sandbox": argv[2], "policy_file": spec.get("policy"), "network_policies": {}}) + "\n")
        return 0
    if argv[:2] == ["service", "expose"]:
        if not os.path.isdir(os.path.join(home, argv[2])):
            sys.stderr.write(f"sandbox {argv[2]} not found\n")
            return 1
        sys.stdout.write(f"Exposed {argv[2]}:{argv[3]} at http://127.0.0.1:{argv[3]}/\n")    # the gateway's URL, here direct
        return 0
    if argv[:2] == ["sandbox", "stop"]:
        with open(os.path.join(home, argv[2], ".state"), "w", encoding="utf-8") as f:
            f.write("Stopped")
        return 0
    if argv[:2] == ["policy", "set"]:
        return 0
    if argv[:1] == ["logs"]:
        try:
            sys.stdout.write(open(os.path.join(home, argv[1], ".log"), encoding="utf-8").read())
        except OSError:
            pass
        return 0
    if argv[:2] == ["sandbox", "start"]:
        with open(os.path.join(home, argv[2], ".state"), "w", encoding="utf-8") as f:
            f.write("Ready")
        return 0
    if argv[:2] == ["sandbox", "delete"]:
        import shutil
        import signal
        try:
            os.killpg(int(open(os.path.join(home, argv[2], ".pid")).read()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        shutil.rmtree(os.path.join(home, argv[2]), ignore_errors=True)
        return 0
    if argv[:2] == ["sandbox", "exec"]:
        name, env, i = None, {}, 2
        while i < len(argv) and argv[i] != "--":
            if argv[i] in ("-n", "--name"):
                name = argv[i + 1]
                i += 2
            elif argv[i] == "--env":
                k, _, v = argv[i + 1].partition("=")
                env[k] = v
                i += 2
            else:
                i += 1
        cmd = argv[i + 1:]
        root = os.path.join(home, name)
        if not os.path.isdir(root):
            sys.stderr.write(f"sandbox {name} not found\n")
            return 1
        # the sandbox's own absolute paths are under its root: rewrite the work folders the code makes there
        # (/tmp/nebula-...) and /sandbox/..., and nothing else, so a checkout or a venv under /tmp still runs as itself
        def inside(w):
            if w.startswith(("/tmp/nebula-", "/sandbox")):
                return os.path.join(root, w.lstrip("/"))
            return w.replace(" /tmp/nebula-", " " + root + "/tmp/nebula-")
        cmd = [inside(w) for w in cmd]
        p = subprocess.run(cmd, stdin=sys.stdin, cwd=root, env=dict(os.environ, **env), check=False)
        return p.returncode
    sys.stderr.write(f"fake openshell: unknown {argv[:2]}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
