"""greenfield/impulse_dist.py: the installer's impulse/ folder carries the Flash-Next engine when the build has it
(impulse-fx beside impulse-server.bin, which execs it from its own folder, and the engine library it loads), all of it
or none, and --require-fx refuses a build without it."""
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("impulse_dist", ROOT / "greenfield" / "impulse_dist.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

EXE = shutil.which("true") or "/bin/true"     # a real ELF for ldd, linking nothing of LINUX_BUNDLE


@unittest.skipUnless(sys.platform.startswith("linux"), "the Linux folder (ldd)")
class ImpulseDistTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.build, self.cuda, self.out = self.tmp / "build", self.tmp / "cuda", self.tmp / "out"
        (self.build / "bin").mkdir(parents=True)
        (self.cuda / "lib").mkdir(parents=True)
        shutil.copy2(EXE, self.build / "bin" / "impulse-server")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fx(self, *names):
        for n in names:
            shutil.copy2(EXE, self.build / "bin" / n)

    def run_dist(self, *extra):
        return D.main(["--platform", "linux", "--build", str(self.build), "--cuda", str(self.cuda), "--out", str(self.out), *extra])

    def test_with_the_engine(self):
        self.fx(*D.FX["linux"])
        self.assertEqual(self.run_dist("--require-fx"), 0)
        names = {p.name for p in self.out.iterdir()}
        self.assertTrue({"impulse-server", "impulse-server.bin", "impulse-fx", "libimpulse_fx.so", "libstrata_kq.so"} <= names, names)
        self.assertTrue((self.out / "impulse-fx").stat().st_mode & 0o111)    # still executable
        self.assertIn('exec "$D/impulse-server.bin"', (self.out / "impulse-server").read_text())

    def test_without_it(self):
        self.assertEqual(self.run_dist(), 0)
        self.assertFalse((self.out / "impulse-fx").exists())
        with self.assertRaises(SystemExit) as e:
            self.run_dist("--require-fx")
        self.assertIn("IMPULSE_FLASHNEXT", str(e.exception))

    def test_half_of_it_is_refused(self):
        self.fx("impulse-fx")
        with self.assertRaises(SystemExit) as e:
            self.run_dist()
        self.assertIn("libimpulse_fx.so", str(e.exception))


if __name__ == "__main__":
    unittest.main()
