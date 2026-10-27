"""greenfield/package.py copies a few files from the tree into every installer: each of them is there (the ggml
licence moved with One Engine P0 and the 0.1.3 installer build stopped on the old path)."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("package", ROOT / "greenfield" / "package.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class PackageFilesTest(unittest.TestCase):
    def test_the_tree_files_it_copies_exist(self):
        for f in (P.LLAMA_LICENSE, ROOT / "greenfield" / "windows" / "install.ps1", ROOT / "nebula" / "app" / "icon.svg",
                  ROOT / "nebula" / "app" / "icon.ico", ROOT / "nebula" / "catalog" / "halo.json", ROOT / "pyproject.toml"):
            self.assertTrue(f.is_file(), f)

    def test_the_licence_is_ggml_mit(self):
        text = P.LLAMA_LICENSE.read_text()
        self.assertTrue(text.startswith("MIT License") and "The ggml authors" in text)


if __name__ == "__main__":
    unittest.main()


class SandboxPayloadTest(unittest.TestCase):
    """Windows' payload carries the Safe sandbox step (scripts + pins) and, given Impulse's Linux build, its release URL
    and sha256 in online.json for sandbox.ps1 to fetch and check."""

    def test_add_sandbox(self):
        import hashlib
        import json
        import tempfile
        pay = Path(tempfile.mkdtemp())
        (pay / "online.json").write_text(json.dumps({"python_url": "https://x"}))
        tar = pay / "impulse.tar.gz"
        tar.write_bytes(b"linux impulse")
        P.add_sandbox(pay, tar)
        for name in P.SANDBOX_FILES + ("wsl.json",):
            self.assertTrue((pay / name).is_file(), name)
        o = json.loads((pay / "online.json").read_text())
        self.assertEqual(o["version"], P.version())
        self.assertEqual(o["sandbox_impulse"]["sha256"], hashlib.sha256(b"linux impulse").hexdigest())
        self.assertTrue(o["sandbox_impulse"]["url"].endswith(f"/v{P.version()}/impulse-{P.version()}-linux-x86_64.tar.gz"))
        pins = json.loads((pay / "wsl.json").read_text())
        self.assertEqual(len(pins["rootfs"]["sha256"]), 64)
        self.assertEqual(len(pins["nvidia"]["key_fingerprint"]), 40)
