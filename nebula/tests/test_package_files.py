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


class PiecesOutOfTheInstallerTest(unittest.TestCase):
    """The owner's 2026-10-27 word, "keep the downloads as small as possible": Impulse Flash-Next leaves the installers
    as a piece of its own (the name pieces.json asks for), and no installer carries llama-server any more."""
    def test_flashnext_is_written_beside_the_installer_with_the_name_pieces_json_asks_for(self):
        import json
        import tarfile
        import tempfile
        imp, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        for n in ("impulse-server.bin", "impulse-fx", "libimpulse_fx.so", "libstrata_kq.so"):
            (imp / n).write_text(n)
        dest = P.make_flashnext(imp, "linux", out)
        m = json.loads((ROOT / "nebula" / "catalog" / "pieces.json").read_text())
        want = (m["url"].rsplit("/", 1)[1].replace("{piece}", "impulse-flashnext").replace("{version}", P.version())
                .replace("{platform}", "linux-x86_64").replace("{ext}", m["pieces"]["impulse-flashnext"]["files"]["linux-x86_64"]["ext"]))
        self.assertEqual(dest.name, want)
        with tarfile.open(dest) as t:
            self.assertEqual(sorted(t.getnames()), ["expert-profile.bin", "impulse-fx", "libimpulse_fx.so", "libstrata_kq.so"])   # the profile: a layer split needs it
        self.assertTrue((out / (dest.name + ".sha256")).is_file())
        self.assertIsNone(P.make_flashnext(Path(tempfile.mkdtemp()), "linux", out))          # a build without Flash-Next

    def test_no_installer_carries_llama_server(self):
        src = (ROOT / "greenfield" / "package.py").read_text()
        self.assertNotIn("--halo-server", src)
        self.assertNotIn("llama-server.bin", src)
