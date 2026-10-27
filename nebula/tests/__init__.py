
# A real OpenShell on the machine running the tests is never used: a test that wants one names its fake.
import os as _os  # noqa: E402

_os.environ.setdefault("NEBULA_OPENSHELL", "nebula-tests-have-no-openshell")

# This machine's own Hugging Face cache is never offered in the tests: a test that wants one names its folder.
_os.environ.setdefault("NEBULA_HF_CACHE", "")
# no test starts Halo's own model unless it says so (a llama-server or Impulse on this machine's PATH would otherwise
# make every test server fetch his 0.7 GB model from Hugging Face): his base is a file that is not there
_os.environ.setdefault("NEBULA_HALO_BASE", _os.path.join(_os.sep, "nebula-tests-have-no-halo-base.gguf"))
# no test server looks for a newer Nebula on the internet: its feed is a port nothing listens on
_os.environ.setdefault("NEBULA_UPDATES_FEED", "http://127.0.0.1:9/nebula-tests-have-no-feed/latest.json")


def free_port() -> int:
    """A port for a test's own server, outside the range the system hands out for outgoing connections (Linux:
    32768-60999), so a client's port cannot take it between the pick and the bind (a CI flake on 2026-10-20)."""
    import random
    import socket
    for _ in range(200):
        p = random.randint(20000, 32000)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


import unittest as _unittest  # noqa: E402

WINDOWS = _os.name == "nt"
# Linux's own: OpenShell sandboxes, Docker, the desktop entry, POSIX file modes, a POSIX shell script as a fake program
linux_only = _unittest.skipIf(WINDOWS, "Linux only (OpenShell, Docker, a .desktop entry, POSIX file modes or shell-script fakes)")
