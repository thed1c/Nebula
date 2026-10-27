#!/usr/bin/env python3
"""impulse-fx's RoPE settings (fx_server.cpp rope_args) without a GPU or a model: the made-up pack and the mock engine of
fx_server_test.py; each case starts the server with a run config and reads the engine's arguments the mock logged.
Checks: nothing added at or below the trained context (the reference engine's 262144); the automatic pick past it (dynamic yarn on
one GPU or a plain layer split, static yarn with factor context / trained where dynamic does not run: pipelined slot
groups, --batch-mtp, q4 K/V); the run config's "rope" (type, factor, orig_ctx; ntk); the args' own --rope-scaling wins;
a bad "rope" stops the start.  Standard library only.

    python fx_rope_test.py BUILD/bin      (impulse-fx or impulse-server, and libimpulse_fx_mock.so there)
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx_server_test as T   # noqa: E402


def engine_args(bindir, root, td, cfg, name, env=None, extra=()):
    mock = td / name
    mock.mkdir()
    S = T.Server(bindir, root, mock, [*extra, "--fx-config-json", json.dumps(cfg)], env or {})
    try:
        for _ in range(200):
            if (mock / "requests.log").exists() and (mock / "requests.log").read_text().startswith("ARGS"):
                break
            if S.p.poll() is not None:
                return None, pathlib.Path(S.log.name).read_text()
            time.sleep(0.05)
        return (mock / "requests.log").read_text().split("\n")[0], pathlib.Path(S.log.name).read_text()
    finally:
        S.close()


def tail(args, flag):
    a = args.split()
    return a[a.index(flag) + 1] if flag in a else None


def main() -> int:
    bindir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "build/bin").resolve()
    td = pathlib.Path(tempfile.mkdtemp(prefix="fx-rope-test-"))
    try:
        root = td / "tiny-pack"
        (root / "pack").mkdir(parents=True)
        (root / "pack" / "index.txt").write_text("test\n")
        T.tiny_gguf(root / "tiny-dense.gguf")
        base = ["--kv", "int8"]

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "262144"]}, "trained")
        T.check("at the trained context: no RoPE flags (the reference engine's 262144)", a and "--rope" not in a, a)

        a, log = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"]}, "auto1")
        T.check("past it, one GPU: --rope-scaling dynamic", a and tail(a, "--rope-scaling") == "dynamic" and "--rope-scale " not in a, a)
        T.check("... said in the log", "dynamic yarn (none up to 262144 positions)" in log)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "gpu": [0, 1], "layer_split": "24"}, "split")
        T.check("past it, a layer split: dynamic (the engine moves every stage)", a and tail(a, "--rope-scaling") == "dynamic", a)

        a, log = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "gpu": [0, 1], "parallel": 2}, "groups")
        T.check("past it, a split with batch slots (pipelined groups): static yarn, factor 2",
                a and tail(a, "--rope-scaling") == "yarn" and tail(a, "--rope-scale") == "2", a)
        T.check("... said in the log", "static yarn (dynamic does not run with pipelined slot groups on a layer split)" in log)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288", "--batch-groups", "1"], "gpu": [0, 1], "parallel": 2}, "serial")
        T.check("... with --batch-groups 1: dynamic", a and tail(a, "--rope-scaling") == "dynamic", a)

        dyn = {"type": "dynamic", "factor": 2, "orig_ctx": 262144, "dynamic_step": 0.25}
        a, log = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "gpu": [0, 1], "parallel": 2, "rope": dyn}, "dyngroups")
        T.check("\"rope\": dynamic on a split with parallel 2: dynamic + --batch-groups 1 (the engine refuses pipelined groups)",
                a and tail(a, "--rope-scaling") == "dynamic" and tail(a, "--batch") == "2" and tail(a, "--batch-groups") == "1", a)
        T.check("... said in the log", "dynamic YaRN with 2 batch slots on a 2-card split: --batch-groups 1" in log)
        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288", "--batch-groups", "2"], "gpu": [0, 1], "parallel": 2,
                                              "rope": dyn}, "dynown")
        T.check("... the args' own --batch-groups wins (as given)", a and a.count("--batch-groups") == 1 and tail(a, "--batch-groups") == "2", a)
        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "gpu": [0, 1], "rope": dyn}, "dynsolo")
        T.check("... parallel 1 on the split (the reference engine today): no --batch-groups", a and "--batch-groups" not in a and "--batch " not in a + " ", a)
        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "parallel": 2, "rope": dyn}, "dyn1gpu")
        T.check("... one card with parallel 2: no --batch-groups", a and tail(a, "--rope-scaling") == "dynamic" and "--batch-groups" not in a, a)
        a, log = engine_args(bindir, root, td, {"args": base + ["--max-context", "262144"], "gpu": [0, 1], "parallel": 2}, "lcgroups",
                             extra=["--long-context", "512k"])
        T.check("--long-context 512k on a split with parallel 2: dynamic + --batch-groups 1",
                a and tail(a, "--rope-scaling") == "dynamic" and tail(a, "--batch-groups") == "1", a)

        a, _ = engine_args(bindir, root, td, {"args": ["--kv", "q4_0", "--max-context", "393216"]}, "q4")
        T.check("past it with q4_0 K/V: static yarn, factor 1.5", a and tail(a, "--rope-scaling") == "yarn" and tail(a, "--rope-scale") == "1.5", a)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "rope": {"type": "ntk"}}, "ntk")
        T.check("\"rope\": ntk -> --rope-scaling ntk --rope-scale 2", a and tail(a, "--rope-scaling") == "ntk" and tail(a, "--rope-scale") == "2", a)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "262144"], "rope": {"type": "yarn", "orig_ctx": 131072}}, "orig")
        T.check("\"rope\": yarn + orig_ctx 131072 at 262144: factor 2, --yarn-orig-ctx",
                a and tail(a, "--rope-scaling") == "yarn" and tail(a, "--rope-scale") == "2" and tail(a, "--yarn-orig-ctx") == "131072", a)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "1048576"],
                                              "rope": {"type": "dynamic", "factor": 4, "orig_ctx": 262144, "dynamic_step": 0.25}}, "bb1m")
        T.check("the reference engine's long context (Nebula long_context('1m')): dynamic, factor 4, orig 262144, step 0.25",
                a and tail(a, "--rope-scaling") == "dynamic" and tail(a, "--rope-scale") == "4" and tail(a, "--yarn-orig-ctx") == "262144"
                and tail(a, "--rope-dynamic-step") == "0.25", a)

        a, log = engine_args(bindir, root, td, {"args": base + ["--max-context", "262144"]}, "lc1m", extra=["--long-context", "1m"])
        T.check("--long-context 1m (Nebula's per-engine Long context): 1048576, dynamic, factor 4, step 0.25",
                a and tail(a, "--max-context") == "1048576" and a.count("--max-context") == 1 and tail(a, "--rope-scaling") == "dynamic"
                and tail(a, "--rope-scale") == "4" and tail(a, "--rope-dynamic-step") == "0.25", a)
        T.check("... said in the log", "long context: 1048576 tokens, dynamic YaRN" in log)
        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "262144"]}, "lcnone")
        T.check("no --long-context: nothing added (the default for every engine)", a and "--rope" not in a and tail(a, "--max-context") == "262144", a)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288"], "rope": {"type": "none"}}, "none")
        T.check("\"rope\": none past the trained context: --rope-scaling none (the opt-out)", a and tail(a, "--rope-scaling") == "none", a)

        a, _ = engine_args(bindir, root, td, {"args": base + ["--max-context", "524288", "--rope-scaling", "linear", "--rope-scale", "2"],
                                              "rope": {"type": "yarn"}}, "own")
        T.check("the args' own --rope-scaling wins (nothing added)", a and a.count("--rope-scaling") == 1 and tail(a, "--rope-scaling") == "linear", a)

        a, log = engine_args(bindir, root, td, {"args": base, "rope": {"type": "longrope"}}, "bad")
        T.check("a bad \"rope\": the start stops and says why", a is None and "\"rope\".\"type\" must be" in log, (a, log[-300:]))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    if T.FAILS:
        print(f"FAILED: {len(T.FAILS)}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
