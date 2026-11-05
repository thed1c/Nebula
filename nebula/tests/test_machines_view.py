"""The Engines page's facts (nebula/machines.py, nebula/yard.py, nebula/engines.py): whether each model machine
answered its last card reading, this computer's name, address and port, and each engine's package and build, read
from the program itself with no card visible to it."""
import os
import stat
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nebula import engines, machines, packages, sshinstall, yard


class SeenTest(unittest.TestCase):
    def cfg(self):
        return SimpleNamespace(raw={"servers": [{"name": "rig", "host": "rig", "user": "u", "port": 22, "gateway": "gw", "gateway_port": 17700}]},
                               base_dir=tempfile.mkdtemp())

    def setUp(self):
        machines._CARDS.clear(); machines._SEEN.clear()

    def test_a_machine_that_answers_and_one_that_does_not(self):
        cfg = self.cfg()
        card = {"index": 0, "uuid": "GPU-1", "name": "NVIDIA GeForce RTX 3090", "memory_total_mib": 24576, "memory_used_mib": 100}
        with mock.patch.object(sshinstall, "gpus", return_value=[card]):
            self.assertEqual([c["name"] for c in machines.cards(cfg, max_age_s=0)], ["NVIDIA GeForce RTX 3090"])
        self.assertEqual((machines.seen("rig")["ok"], machines.seen("rig")["error"]), (True, None))
        with mock.patch.object(sshinstall, "gpus", side_effect=sshinstall.SshError("ssh: connect to host rig port 22: No route to host")):
            out = machines.cards(cfg, max_age_s=0)
        self.assertTrue(out[0]["unreachable"])                      # its cards as last seen, marked
        self.assertEqual((machines.seen("rig")["ok"], machines.seen("rig")["error"]), (False, "ssh: connect to host rig port 22: No route to host"))
        self.assertIsNone(machines.seen("another"))

    def test_the_servers_view_says_so_and_names_this_computer(self):
        machines._SEEN["rig"] = {"ok": True, "at": time.time(), "error": None}
        server = SimpleNamespace(cfg=self.cfg(), server_port=18194, yard=object())
        v = yard.get(server, "/api/yard/servers")
        self.assertTrue(v["servers"][0]["seen"]["ok"])
        self.assertEqual(v["here"]["port"], 18194)
        self.assertTrue(v["here"]["name"])


class BuildTest(unittest.TestCase):
    def test_the_build_is_read_once_in_the_background_with_no_card_visible(self):
        d = tempfile.mkdtemp()
        prog = os.path.join(d, "llama-server")
        with open(prog, "w") as f:
            f.write('#!/bin/sh\necho "version: 7123 (abc1234) cards=[$CUDA_VISIBLE_DEVICES]"\necho "built with cc"\n')
        os.chmod(prog, os.stat(prog).st_mode | stat.S_IXUSR)
        self.assertEqual(packages.version(prog), "version: 7123 (abc1234) cards=[]")
        engines._BUILDS.clear()
        self.assertIsNone(engines.build_of(prog))                   # the first look starts the reading: no wait
        for _ in range(100):
            if engines.build_of(prog):
                break
            time.sleep(0.05)
        self.assertEqual(engines.build_of(prog), "version: 7123 (abc1234) cards=[]")
        self.assertIsNone(engines.build_of(os.path.join(d, "missing")))


if __name__ == "__main__":
    unittest.main()
