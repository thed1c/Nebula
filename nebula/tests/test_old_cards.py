"""Cards older than Impulse's build (CUDA 13 builds for compute capability 7.5 and newer): a GTX 1080 Ti (6.1) failed an
engine with a generic error (a user, 2026-11-01). Such a card is said as such, never chosen for an engine, Halo or a
guided setup, and an engine pinned to it stops with the reason."""
import os
import tempfile
import unittest
from unittest import mock

from nebula import engines as E

PASCAL = {"uuid": "GPU-pppp", "index": 0, "name": "NVIDIA GeForce GTX 1080 Ti", "memory_mb": 11264, "used_mb": 300,
          "compute_cap": "6.1"}
TURING = {"uuid": "GPU-tttt", "index": 1, "name": "NVIDIA GeForce GTX 1660 SUPER", "memory_mb": 6144, "used_mb": 200,
          "compute_cap": "7.5"}


class Cfg:
    def __init__(self, d, engines):
        self.engines, self.base_dir, self.path, self.raw = engines, d, os.path.join(d, "nebula.json"), {"models_dir": os.path.join(d, "models")}
        self.tiers = []


class CardTooOldTest(unittest.TestCase):
    def test_older_than_the_build_is_said_in_plain_words(self):
        why = E.card_too_old(PASCAL)
        self.assertIn("GTX 1080 Ti is older than this engine build supports", why)
        self.assertIn("compute 6.1", why)
        self.assertIn("runs on the CPU", why)
        self.assertIn("older-card engine download is planned", why)

    def test_supported_or_unknown_cards_are_fine(self):
        for cc in ("7.5", "8.6", "12.0", "10.0", "", None, "N/A", "[Not Supported]"):
            self.assertIsNone(E.card_too_old({**TURING, "compute_cap": cc}), cc)

    def test_an_engine_pinned_to_an_old_card_stops_with_the_reason(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "models"))
        model = os.path.join(d, "models", "m.gguf")
        open(model, "wb").close()
        spec = {"name": "big", "kind": "impulse", "model": model, "port": 18311, "card": "GPU-pppp"}
        en = E.Engines(Cfg(d, [spec]))
        with mock.patch.object(E, "cards", return_value=[PASCAL, TURING]):
            with self.assertRaises(E.EngineSpecError) as e:
                en.start("big")
        self.assertIn("older than this engine build supports", str(e.exception))

    def test_a_guided_setup_never_picks_an_old_card(self):
        from nebula import wizards
        with mock.patch.object(E, "cards", return_value=[PASCAL]):
            self.assertIsNone(wizards._cards_for({}))                       # only an old card: the model stays in RAM
        big_old = {**PASCAL, "memory_mb": 24576}
        with mock.patch.object(E, "cards", return_value=[big_old, TURING]):
            self.assertEqual(wizards._cards_for({}), "GPU-tttt")          # the newer card, though it has less room


if __name__ == "__main__":
    unittest.main()
