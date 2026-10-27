"""The fake engine as a process, for the desktop app's tests: it prints load lines to stderr the way
strata-lite-server does, then listens (fake_engine.py's answers) and runs until it is signalled.

    python3 -m nebula.tests.fake_engine_proc --port P [--steps N] [--step-s S] [--fail] [--ignore-term] [--child PIDFILE]

--fail ends with exit code 3 instead of listening; --ignore-term ignores SIGTERM (the app must kill it); --child starts
a sleeping grandchild and writes its pid to PIDFILE (the app must end the whole group).
"""
import argparse
import signal
import subprocess
import sys
import time

from nebula.tests.fake_engine import FakeEngine


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--step-s", type=float, default=0.2)
    p.add_argument("--fail", action="store_true")
    p.add_argument("--ignore-term", action="store_true")
    p.add_argument("--child")
    a = p.parse_args()
    if a.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if a.child:
        c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open(a.child, "w") as f:
            f.write(str(c.pid))
    for i in range(a.steps):
        print(f"strata-lite: loading the model, step {i + 1} of {a.steps}", file=sys.stderr, flush=True)
        time.sleep(a.step_s)
    if a.fail:
        print("engine: cannot load the model (fake)", file=sys.stderr, flush=True)
        return 3
    srv = FakeEngine("procfake", port=a.port, background=False)
    print(f"strata-lite-server: listening on http://127.0.0.1:{a.port} (alias procfake, ctx 262144, vision 0, mtp 0)",
          file=sys.stderr, flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
