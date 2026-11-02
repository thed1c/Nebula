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


class WindowsSetupClosesNebulaTest(unittest.TestCase):
    """A Nebula left running held its Python's files: Setup removed what it could, fetched Python over the rest and pip
    failed on files in use, under a message about the internet connection (a 0.1.6.1 install, 2026-11-01)."""
    WIN = ROOT / "greenfield" / "windows"

    def test_setup_closes_nebula_before_it_removes_anything(self):
        nsi = (self.WIN / "installer.nsi").read_text()
        section = nsi[nsi.index('Section "Nebula EH"'):]
        self.assertLess(section.index("close-nebula.ps1"), section.index("RMDir"))
        self.assertIn('StrCmp $1 "3"', section[:section.index("RMDir")])        # still in use: stop, nothing touched
        self.assertNotIn("taskkill /F /IM", nsi)                              # never by name: other programs stay

    def test_the_close_script_finds_nebula_by_path_and_says_when_it_cannot(self):
        ps = (self.WIN / "close-nebula.ps1").read_text()
        self.assertIn("Win32_Process", ps)
        self.assertIn("ExecutablePath", ps)
        self.assertIn("exit 3", ps)
        self.assertNotIn("-Name", ps.split("function InUse")[0])               # no Get-Process -Name / image names

    def test_files_in_use_are_not_blamed_on_the_connection(self):
        ps = (self.WIN / "install.ps1").read_text()
        self.assertGreaterEqual(ps.count("exit 3"), 2)                         # tar over files in use, and pip
        nsi = (self.WIN / "installer.nsi").read_text()
        self.assertIn("some of its files are in use", nsi)

    def test_the_finish_page_opens_nebula_as_the_user(self):
        nsi = (self.WIN / "installer.nsi").read_text()
        start = nsi.index("Function OpenNebula")
        fn = nsi[start:nsi.index("FunctionEnd", start)]
        self.assertIn("explorer.exe", fn)
        self.assertNotIn("pythonw.exe", fn)


class WindowsUpdatesAreSmallTest(unittest.TestCase):
    """Every Setup deleted Nebula's whole Python and fetched it and every package again, so a fix of a few lines was a full
    reinstall; and the Update button's Setup ran hidden (the owner's word, 2026-11-02)."""
    WIN = ROOT / "greenfield" / "windows"

    def test_setup_no_longer_deletes_python(self):
        nsi = (self.WIN / "installer.nsi").read_text()
        self.assertNotIn('RMDir /r "$INSTDIR\\python"', nsi)
        self.assertIn('RMDir /r "$INSTDIR\\impulse"', nsi)               # what the payload carries is put back from it

    def test_install_keeps_python_of_the_same_version_and_starts_over_once_when_pip_fails(self):
        ps = (self.WIN / "install.ps1").read_text()
        self.assertIn("platform.python_version()", ps)
        self.assertIn("$have -eq $wantPy", ps)
        self.assertEqual(ps.count("$code = InstallPackages"), 2)              # once, and once more with a fresh Python
        import re
        name = f"cpython-{P.PY}+{P.PBS}-x86_64-pc-windows-msvc-install_only.tar.gz"
        pat = re.search(r"-match '([^']+)'", ps).group(1)
        self.assertEqual(re.search(pat, name).group(1), P.PY)                # the version read from the pinned download

    def test_the_update_runs_setup_in_front_without_pages(self):
        nsi = (self.WIN / "installer.nsi").read_text()
        self.assertIn('${GetOptions} $0 "/UPDATE" $1', nsi)
        self.assertEqual(nsi.count("!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipOnUpdate"), 2)   # welcome and finish
        self.assertIn("SetAutoClose true", nsi)
        fn = nsi[nsi.index("Function SetupInFront"):]
        fn = fn[:fn.index("FunctionEnd")]
        self.assertIn("SetWindowPos", fn)
        self.assertIn("BringToFront", fn)
        self.assertNotRegex(fn, r"StrCmp [^\n]* \+\d")                        # no relative jump over plugin calls

    def test_an_update_never_asks_the_sandbox_question(self):
        nsi = (self.WIN / "installer.nsi").read_text()
        keep = nsi.index("sandbox.ps1\" -Dest \"$INSTDIR\" -Conf \"$APPDATA\\nebula\" -Keep")
        ask = nsi.index("Run your models in a Safe sandbox?")
        between = nsi[keep:ask]
        self.assertIn('StrCmp $1 "4" 0 sandbox_done', between)
        self.assertIn('StrCmp $Update "1" sandbox_done', between)
        self.assertIn("IfSilent sandbox_done", between)
        ps = (self.WIN / "sandbox.ps1").read_text()
        self.assertIn("exit 4", ps)
        self.assertIn("impulse_sha256", ps)                                  # the same Impulse is not fetched again
