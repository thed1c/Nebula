"""Cards older than Impulse's build (CUDA 13 builds for compute capability 7.5 and newer): a GTX 1080 Ti (6.1) failed an
engine with a generic error (a user, 2026-11-01). Such a card is said as such; Maxwell, Pascal and Volta (5.0 to 7.0) run
on Impulse (L), the CUDA 12 build Nebula downloads when the owner asks (never in the installer); older ones (Kepler)
stay on the CPU. One engine's cards are of one kind, and Impulse (L) needs a driver new enough for CUDA 12.9."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nebula import engines as E

PASCAL = {"uuid": "GPU-pppp", "index": 0, "name": "NVIDIA GeForce GTX 1080 Ti", "memory_mb": 11264, "used_mb": 300,
          "compute_cap": "6.1"}
TURING = {"uuid": "GPU-tttt", "index": 1, "name": "NVIDIA GeForce GTX 1660 SUPER", "memory_mb": 6144, "used_mb": 200,
          "compute_cap": "7.5"}
VOLTA = {"uuid": "GPU-vvvv", "index": 0, "name": "Tesla V100-PCIE-16GB", "memory_mb": 16384, "used_mb": 0, "compute_cap": "7.0"}
MAXWELL = {"uuid": "GPU-mmmm", "index": 0, "name": "NVIDIA GeForce GTX 980", "memory_mb": 4096, "used_mb": 0, "compute_cap": "5.2"}
KEPLER = {"uuid": "GPU-kkkk", "index": 0, "name": "Tesla K80", "memory_mb": 12288, "used_mb": 0, "compute_cap": "3.7"}
AMPERE = {"uuid": "GPU-aaaa", "index": 1, "name": "NVIDIA GeForce RTX 3060", "memory_mb": 12288, "used_mb": 0, "compute_cap": "8.6"}
PIN = {"url": "https://example.invalid/impulse-legacy.tar.xz", "sha256": "0" * 64, "bytes": 494 * 2**20, "ext": "tar.xz",
       "version": "9.9", "label": "Impulse (L)", "why": "the engine for older NVIDIA cards"}


class Cfg:
    def __init__(self, d, engines):
        self.engines, self.base_dir, self.path, self.raw = engines, d, os.path.join(d, "nebula.json"), {"models_dir": os.path.join(d, "models")}
        self.tiers = []


def piece_here(base: str) -> str:
    """Impulse (L) as Nebula unpacks it: <config folder>/nebula-impulse-legacy/, its wrapper and the CUDA 12 runtime."""
    d = Path(base, "nebula-impulse-legacy")
    d.mkdir(parents=True, exist_ok=True)
    (d / "impulse-server").write_text("#!/bin/sh\n")
    (d / "impulse-server").chmod(0o755)
    (d / "libcudart.so.12").write_text("")
    return str(d / "impulse-server")


def published(name):
    return PIN if name == E.LEGACY else None


def unpublished(name, _real=__import__("nebula.pieces", fromlist=["pin"]).pin):
    """The catalog as it was before Impulse (L)'s release: everything else as it is."""
    return None if name == E.LEGACY else _real(name)


class Env(unittest.TestCase):
    """A clean card environment per test: no real nvidia-smi, no Impulse (L) named by the environment."""
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"NEBULA_FAKE_DRIVER": "550.54.14"})
        self.env.start()
        for k in ("NEBULA_IMPULSE_LEGACY_BIN", "NEBULA_FAKE_CARDS", "NEBULA_FAKE_CAPS", "NEBULA_HALO_SERVER"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.env.stop()

    def cards(self, *cs):
        os.environ["NEBULA_FAKE_CARDS"] = json.dumps(list(cs))
        os.environ["NEBULA_FAKE_CAPS"] = ",".join(c["compute_cap"] for c in cs)


class CardTooOldTest(Env):
    def test_kepler_and_older_run_on_the_cpu(self):
        why = E.card_too_old(KEPLER, self.base)
        self.assertEqual(why, "Tesla K80 is too old for Impulse (compute 3.7; its oldest build, Impulse (L), needs 5.0 or "
                              "newer): it runs on the CPU")
        piece_here(self.base)                                            # Impulse (L) changes nothing for it
        self.assertEqual(E.card_too_old(KEPLER, self.base), why)
        self.assertEqual(E.card_too_old_short(KEPLER, self.base), "older than every Impulse build: CPU only")

    def test_older_cards_without_the_piece_say_impulse_l_runs_them(self):
        with mock.patch("nebula.pieces.pin", side_effect=unpublished):  # a platform it is not published for
            for c in (PASCAL, VOLTA, MAXWELL):
                why = E.card_too_old(c, self.base)
                self.assertIn(f"{c['name']} is older than Impulse's main build supports (compute {c['compute_cap']}; it "
                              "needs 7.5 or newer)", why)
                self.assertIn("Impulse (L), Nebula's engine for older NVIDIA cards, is not published for this computer yet: "
                              "it runs on the CPU", why)
        with mock.patch("nebula.pieces.platform_key", return_value="linux-x86_64"):   # the real catalog: 0.1.7's pieces
            self.assertIn("Run a model offers it (a download of 470 MB, not in the installer)", E.card_too_old(PASCAL, self.base))
        with mock.patch("nebula.pieces.platform_key", return_value="windows-x86_64"):
            self.assertIn("(a download of 586 MB, not in the installer)", E.card_too_old(PASCAL, self.base))
        with mock.patch("nebula.pieces.pin", side_effect=published):
            why = E.card_too_old(PASCAL, self.base)
            self.assertEqual(why, "NVIDIA GeForce GTX 1080 Ti is older than Impulse's main build supports (compute 6.1; it "
                                  "needs 7.5 or newer): Impulse (L), Nebula's engine for older NVIDIA cards, runs it: Run a "
                                  "model offers it (a download of 494 MB, not in the installer)")
            self.assertEqual(E.card_too_old_short(PASCAL, self.base), "older card: needs Impulse (L) (Run a model offers it)")

    def test_with_the_piece_older_cards_are_usable(self):
        piece_here(self.base)
        for c in (PASCAL, VOLTA, MAXWELL):
            self.assertIsNone(E.card_too_old(c, self.base), c["name"])
            self.assertIsNone(E.card_too_old_short(c, self.base))

    def test_a_driver_too_old_for_cuda_12_9_is_said(self):
        piece_here(self.base)
        os.environ["NEBULA_FAKE_DRIVER"] = "470.82.01"
        why = E.card_too_old(PASCAL, self.base)
        self.assertEqual(why, "NVIDIA GeForce GTX 1080 Ti runs on Impulse (L), but the NVIDIA driver here is 470.82.01, and "
                              "Impulse (L) needs 525.60.13 or newer: update the NVIDIA driver to use this card")
        self.assertEqual(E.card_too_old_short(PASCAL, self.base), "older card: its NVIDIA driver is too old for Impulse (L)")
        self.assertIsNone(E.card_too_old(PASCAL, self.base, driver=False))   # a model machine's card: its own driver
        for ok in ("525.60.13", "525.105.17", "580.65.06", "", "N/A"):
            os.environ["NEBULA_FAKE_DRIVER"] = ok
            self.assertIsNone(E.card_too_old(PASCAL, self.base), ok)
        with mock.patch.object(E.os, "name", "nt"):
            os.environ["NEBULA_FAKE_DRIVER"] = "528.24"
            self.assertIn("needs 528.33 or newer", E.legacy_driver_gap())

    def test_supported_or_unknown_cards_are_fine(self):
        for cc in ("7.5", "8.6", "12.0", "10.0", "", None, "N/A", "[Not Supported]"):
            self.assertIsNone(E.card_too_old({**TURING, "compute_cap": cc}), cc)

    def test_a_card_no_build_carries_runs_on_the_cpu(self):
        jetson = {"name": "Jetson TX2", "compute_cap": "6.2"}
        piece_here(self.base)
        self.assertEqual(E.card_too_old(jetson, self.base), "no Impulse build carries kernels for Jetson TX2 (compute 6.2): "
                                                            "it runs on the CPU")


class ImpulseBuildTest(Env):
    def test_one_engine_runs_on_cards_of_one_kind(self):
        piece_here(self.base)
        prog, why = E.impulse_build([PASCAL, AMPERE], None, self.base)
        self.assertIsNone(prog)
        self.assertEqual(why, "NVIDIA GeForce GTX 1080 Ti is an older card (Impulse (L)) and NVIDIA GeForce RTX 3060 a newer "
                              "one (Impulse): one engine runs one Impulse build, so pick cards of one kind")

    def test_older_cards_run_impulse_l_and_newer_ones_the_main_build(self):
        leg = piece_here(self.base)
        regular = os.path.join(self.base, "nebula-impulse", "impulse-server")
        os.makedirs(os.path.dirname(regular))
        open(regular, "w").close()
        with mock.patch.dict(os.environ, {"NEBULA_IMPULSE_BIN": regular}):
            self.assertEqual(E.impulse_build([PASCAL], None, self.base), (leg, None))
            self.assertEqual(E.impulse_build([VOLTA, {**VOLTA, "uuid": "GPU-v2"}], regular, self.base), (leg, None))
            self.assertEqual(E.impulse_build([AMPERE], leg, self.base), (regular, None))     # back to the main build
            self.assertEqual(E.impulse_build([AMPERE], regular, self.base), (None, None))
            self.assertEqual(E.impulse_build([PASCAL], "/opt/my/impulse-server", self.base), (None, None))   # the owner's own
            self.assertEqual(E.impulse_build([], None, self.base), (None, None))

    def test_without_the_piece_an_older_card_is_refused_with_the_offer(self):
        prog, why = E.impulse_build([MAXWELL], None, self.base)
        self.assertIsNone(prog)
        self.assertIn("GTX 980 is older than Impulse's main build supports", why)

    def test_an_engine_pinned_to_an_old_card_stops_with_the_reason(self):
        os.makedirs(os.path.join(self.base, "models"))
        model = os.path.join(self.base, "models", "m.gguf")
        open(model, "wb").close()
        spec = {"name": "big", "kind": "impulse", "model": model, "port": 18311, "card": "GPU-pppp"}
        en = E.Engines(Cfg(self.base, [spec]))
        with mock.patch.object(E, "cards", return_value=[PASCAL, TURING]):
            with self.assertRaises(E.EngineSpecError) as e:
                en.start("big")
        self.assertIn("older than Impulse's main build supports", str(e.exception))
        en = E.Engines(Cfg(self.base, [dict(spec, card="GPU-pppp,GPU-tttt")]))
        piece_here(self.base)
        with mock.patch.object(E, "cards", return_value=[PASCAL, TURING]):
            with self.assertRaises(E.EngineSpecError) as e:
                en.start("big")
        self.assertIn("pick cards of one kind", str(e.exception))

    def test_an_engine_on_an_old_card_runs_impulse_l_once_it_is_here(self):
        leg = piece_here(self.base)
        os.makedirs(os.path.join(self.base, "models"))
        model = os.path.join(self.base, "models", "m.gguf")
        open(model, "wb").close()
        en = E.Engines(Cfg(self.base, [{"name": "old", "kind": "impulse", "model": model, "port": 18312, "card": "GPU-pppp",
                                        "where": "local"}]))
        seen = []

        def command_of(spec, *a, **k):
            seen.append(spec)
            raise RuntimeError("stop here")
        with mock.patch.object(E, "cards", return_value=[PASCAL]), mock.patch.object(E, "command_of", side_effect=command_of):
            with self.assertRaises(RuntimeError):
                en.start("old")
        self.assertEqual(seen[0]["binary"], leg)


class GuidedSetupTest(Env):
    class Server:
        def __init__(self, base):
            self.cfg = Cfg(base, [])

    def test_a_1080_ti_machine_is_offered_impulse_l(self):
        from nebula import wizards as W
        srv = self.Server(self.base)
        self.cards(PASCAL)
        with mock.patch("nebula.pieces.pin", side_effect=published):
            pkg, binary, why = W._engine_for(srv, {})
            self.assertEqual((pkg, binary), ("impulse", E.LEGACY))
            self.assertEqual(why, "NVIDIA GeForce GTX 1080 Ti is older than Impulse's main build supports: Nebula runs it "
                                  "with Impulse (L), the build for older NVIDIA cards")
            self.assertEqual(W._portable_lines(srv, binary), ["Downloads Impulse (L) (494 MB): the engine for older NVIDIA "
                                                              "cards; it is not in the installer, to keep the download small"])
            self.assertEqual(W._engine_line(srv, pkg, binary), "Run it on this computer with Impulse (L), the build for older "
                                                              "NVIDIA cards")
            self.assertEqual(W._cards_for({}, srv, binary), "GPU-pppp")          # its card, though the piece is still to come
        leg = piece_here(self.base)
        self.assertEqual(W._engine_for(srv, {})[:2], ("impulse", leg))            # here: no download
        self.assertEqual(W._portable_lines(srv, leg), [])

    def test_unpublished_or_an_old_driver_falls_back_to_impulse_portable_and_says_why(self):
        from nebula import wizards as W
        srv = self.Server(self.base)
        self.cards(PASCAL)
        with mock.patch("nebula.pieces.pin", side_effect=unpublished):           # not published for this platform
            pkg, binary, why = W._engine_for(srv, {})
        self.assertEqual((pkg, binary), ("llama.cpp", W.PORTABLE))
        self.assertIn("Impulse (L), Nebula's engine for older NVIDIA cards, is not published for this computer yet", why)
        piece_here(self.base)
        os.environ["NEBULA_FAKE_DRIVER"] = "470.82.01"
        pkg, binary, why = W._engine_for(srv, {})
        self.assertEqual(pkg, "llama.cpp")
        self.assertIn("the NVIDIA driver here is 470.82.01, and Impulse (L) needs 525.60.13 or newer", why)

    def test_a_mixed_machine_runs_the_main_build_on_its_newer_card(self):
        from nebula import wizards as W
        srv = self.Server(self.base)
        self.cards(PASCAL, AMPERE)
        with mock.patch.dict(os.environ, {"NEBULA_IMPULSE_BIN": "/bin/sh"}):
            self.assertEqual(W._engine_for(srv, {})[:2], ("impulse", "/bin/sh"))
        self.assertEqual(W._cards_for({}, srv, "/bin/sh"), "GPU-aaaa")            # no Impulse (L): the 1080 Ti is out
        piece_here(self.base)
        big = {**PASCAL, "memory_mb": 24576}
        self.cards(big, AMPERE)
        self.assertEqual(W._cards_for({}, srv, "/bin/sh"), "GPU-pppp")            # with it, the roomier card (Impulse (L))
        self.assertEqual(W._cards_for({}), "GPU-aaaa")                            # a Flash-Next pack: never an older card

    def test_kepler_is_not_offered_impulse_l(self):
        from nebula import wizards as W
        self.cards(KEPLER)
        with mock.patch("nebula.pieces.pin", side_effect=published):
            self.assertEqual(E.legacy_here(self.base), (None, None))
            self.assertNotEqual(W._engine_for(self.Server(self.base), {})[0], "impulse")

    def test_the_catalog_says_impulse_l_and_lays_out_older_cards_only(self):
        from nebula import catalog as C
        self.cards(PASCAL, {**PASCAL, "uuid": "GPU-p2", "index": 1})
        with mock.patch("nebula.pieces.pin", side_effect=published):
            o = C.offer(self.base, base_dir=self.base)
        self.assertFalse(o["impulse"])
        self.assertTrue(o["legacy"]["download"])
        m = {"layouts": [{"id": "two", "cards": 2, "min_total_mb": 16000, "about": "both cards"}]}
        self.assertIsNone(C.layout(m, legacy=False))                              # the main build: no card here
        self.assertEqual(C.layout(m, legacy=True)["card"], "GPU-pppp,GPU-p2")
        self.assertIsNone(C.layout(m, [PASCAL, AMPERE], legacy=True))             # never an older and a newer card


class HaloTest(Env):
    def test_halo_runs_on_impulse_l_on_an_older_card_when_it_is_here(self):
        from nebula import halo_engine as HE
        self.cards(PASCAL)
        with mock.patch.dict(os.environ, {"NEBULA_IMPULSE_BIN": "/bin/sh"}):
            self.assertEqual(HE.find_servers(self.base)[0], ("Impulse", ["/bin/sh"]))   # not here: the main build (on the CPU)
            leg = piece_here(self.base)
            self.assertEqual(HE.find_servers(self.base)[0], ("Impulse", [leg]))
            self.cards(PASCAL, AMPERE)
            self.assertEqual(HE.find_servers(self.base)[0], ("Impulse", ["/bin/sh"]))   # a newer card here: the main build


class SandboxImageTest(Env):
    def test_impulse_l_gets_an_image_of_its_own(self):
        from nebula import openshell
        leg = piece_here(self.base)
        reg = os.path.join(self.base, "nebula-impulse", "impulse-server")
        os.makedirs(os.path.dirname(reg))
        Path(reg).write_text("#!/bin/sh\n")
        self.assertTrue(openshell.engine_image_tag(leg).startswith("nebula-engine-impulse-server-legacy:"))
        self.assertTrue(openshell.engine_image_tag(reg).startswith("nebula-engine-impulse-server:"))
        self.assertEqual(openshell._lib_path(leg), os.path.dirname(os.path.realpath(leg)))   # never CUDA 13's libraries
        self.assertTrue(E.is_legacy_program(leg))
        self.assertFalse(E.is_legacy_program(reg))


if __name__ == "__main__":
    unittest.main()
