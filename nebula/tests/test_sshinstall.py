"""Add a server (SSH) (nebula/sshinstall.py): only the model-machine pieces go over SSH (PAIR's node programs, the node
keeper, OpenShell), never a Nebula; what needs an administrator; the scripts' syntax."""
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from nebula import openshell as O
from nebula import sshinstall as S
from nebula.tests import linux_only


class FakeTarget(S.Target):
    def __init__(self, out: str):
        super().__init__("box", "owner", 22, None, "/dev/null")
        self.calls, self.out = [], out

    def run(self, remote, stdin=None, timeout=600):
        self.calls.append((remote, stdin))
        return self.out if remote == "sh -s" else ""


class RunInstallTest(unittest.TestCase):
    def test_only_the_node_pieces_go_over_and_no_nebula(self):
        with tempfile.TemporaryDirectory() as d:
            pair = os.path.join(d, "pair")
            os.makedirs(pair)
            for b in ("nvpair-ui-broker", "nvpair-node-scanner"):
                Path(pair, b).write_text("x")
            t = FakeTarget("...\nNODE ready\n")
            S.install(t, pair, None)
        copy, go = t.calls
        self.assertEqual(copy[0], "cat > ~/nebula-node.tar")
        with tarfile.open(fileobj=io.BytesIO(copy[1])) as tf:
            names = sorted(tf.getnames())
        self.assertEqual(names, ["nodekeeper.py", "pair/nvpair-node-scanner", "pair/nvpair-ui-broker"])
        self.assertEqual(go[0], "sh -s")
        script = go[1].decode()
        for w in ("nebula-node.service", "nodekeeper.py serve", "--facade-port 8081", "[openshell.drivers.podman]",
                  "OPENSHELL_COMPUTE_DRIVER=podman", "enable --now podman.socket"):
            self.assertIn(w, script)
        for never in ("pip install", "nebula serve", "nebula-headless", "nebula.json"):
            self.assertNotIn(never, script)
        self.assertNotIn("__", script)                                           # every placeholder filled

    def test_a_node_that_does_not_start_is_said(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(S.SshError):
                S.install(FakeTarget("the node keeper did not start:\n"), d, None)

    def test_uninstall_removes_only_its_own(self):
        self.assertIn("nebula-node.service", S.UNINSTALL_SH)
        self.assertIn('grep -q "for a Nebula on your network"', S.UNINSTALL_SH)
        self.assertIn("nebeng-", S.UNINSTALL_SH)

    @linux_only
    def test_the_scripts_are_valid_sh(self):
        for sh in (S.INSTALL_SH, S.UNINSTALL_SH, S.PREFLIGHT_SH):
            p = subprocess.run(["sh", "-n"], input=sh, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_the_keeper_is_reached_over_ssh(self):
        a = S.keeper_argv(S.Target("box", "owner", 22, None, "/dev/null"))
        self.assertEqual(a[0], "ssh")
        self.assertIn("StrictHostKeyChecking=yes", a)
        self.assertTrue(a[-1].endswith("nodekeeper.py connect --state ~/.local/state/nebula-node"))


@linux_only
class EngineProgramTest(unittest.TestCase):
    def test_the_installers_wrapper_is_followed_to_its_program(self):
        with tempfile.TemporaryDirectory() as d:
            wrap = os.path.join(d, "impulse-server")
            Path(wrap).write_text('#!/bin/sh\nexec "$(dirname "$0")/impulse-server.bin" "$@"\n')
            os.chmod(wrap, 0o755)
            shutil.copyfile(os.path.realpath(sys.executable), wrap + ".bin")
            self.assertEqual(O.engine_program(wrap), wrap + ".bin")
            self.assertTrue(O.engine_image_tag(wrap).startswith("nebula-engine-impulse-server:"))
            self.assertEqual([n for n, _ in O.engine_libs(wrap)], [n for n, _ in O.engine_libs(sys.executable)])


BOX_LIKE = ["Linux x86_64", "GPU 0: NVIDIA GeForce RTX 5070 (UUID: GPU-x)", "Python 3.12.3", "user systemd: yes",
             "ssh: 192.0.2.10 51234 192.0.2.20 22", "user: owner", "podman: missing", "uidmap: no", "subuid: yes",
             "container toolkit: yes", "cdi: yes", "firewall: ufw", "linger: yes", "sudo: yes", "apt: yes"]


class AdminTest(unittest.TestCase):
    def test_what_a_gpu_box_would_need(self):
        needs = S.admin_needs(BOX_LIKE, local_firewall=True)
        self.assertEqual([(n["id"], n["where"]) for n in needs], [("podman-install", "server"), ("cdi-podman", "server"),
                                                                  ("firewall", "server"), ("firewall-here", "here")])
        self.assertIn("install -y -q podman uidmap passt", needs[0]["command"])
        self.assertEqual(needs[2]["command"], "ufw allow from 192.0.2.10 comment 'Nebula on your network'")
        self.assertIn("from 192.0.2.20", needs[3]["command"])

    def test_a_bare_box(self):
        bare = ["user systemd: yes", "user: owner", "podman: missing", "uidmap: no", "subuid: no",
                "container toolkit: no", "cdi: no", "firewall: none", "linger: no", "sudo: yes", "apt: yes", "GPU 0: x"]
        ids = [n["id"] for n in S.admin_needs(bare, local_firewall=False)]
        self.assertEqual(ids, ["podman-install", "subuid", "toolkit", "linger"])  # the toolkit too, with apt

    def test_podman_4_gets_a_card_description_it_can_read(self):
        box = ["user: owner", "podman: 4.9.3", "uidmap: yes", "subuid: yes", "container toolkit: yes", "cdi: yes",
                "cdi for podman: no", "linger: yes", "firewall: none", "apt: yes", "sudo: yes", "user systemd: yes"]
        needs = S.admin_needs(box, local_firewall=False)
        self.assertEqual([n["id"] for n in needs], ["cdi-podman"])
        cmd = needs[0]["command"]
        for w in ('pop("additionalGids", None)', "PathChanged=/var/run/cdi/nvidia.yaml", "enable --now nebula-cdi-podman.path"):
            self.assertIn(w, cmd)
        p = subprocess.run(["sh", "-n"], input=S._sudo_script([cmd]), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_no_user_systemd_is_said(self):
        notes = [n["words"] for n in S.admin_needs(["user systemd: no", "podman: 4.9.3", "uidmap: yes"], local_firewall=False)]
        self.assertTrue(any("user services" in w for w in notes))

    def test_tasks_are_worked_out_there_and_the_password_goes_only_to_sudo(self):
        class T(FakeTarget):
            def run(self, remote, stdin=None, timeout=600):
                self.calls.append((remote, stdin))
                return "\n".join(BOX_LIKE) if remote == S.PREFLIGHT_SH else "ok"
        t = T("")
        out = S.run_admin(t, ["podman-install", "firewall", "rm -rf /"], "hunter2")
        self.assertEqual(len(out["done"]), 2)
        remote, stdin = t.calls[-1]
        self.assertTrue(remote.startswith("sudo -S -p '' sh -c "))
        self.assertIn("install -y -q podman", remote)
        self.assertNotIn("rm -rf /", remote)
        self.assertNotIn("hunter2", remote)
        self.assertEqual(stdin, b"hunter2\n")
        with self.assertRaises(S.SshError):
            S.run_admin(T(""), ["podman-install"], "")

    def test_an_address_gets_the_key_of_the_ssh_name_that_points_at_it(self):
        with tempfile.TemporaryDirectory() as d:
            key = os.path.join(d, "id_sage")
            Path(key).write_text("x")
            cf = os.path.join(d, "config")
            Path(cf).write_text(f"Host gpubox\n    HostName 192.0.2.77\n    IdentityFile {key}\n")
            old = os.environ.get("NEBULA_SSH_CONFIG")
            os.environ["NEBULA_SSH_CONFIG"] = cf
            try:
                self.assertEqual(S.config_hosts(), ["gpubox"])
                self.assertEqual(S.own_key("gpubox"), key)
                self.assertEqual(S.own_key("192.0.2.77"), key)
                self.assertIsNone(S.own_key("192.0.2.78"))
                a = S.Target("192.0.2.77", "owner", 22, None, "/dev/null").argv("true")
                self.assertIn(key, a)
                self.assertIn("HostKeyAlias=192.0.2.77", a)
            finally:
                if old is None:
                    os.environ.pop("NEBULA_SSH_CONFIG", None)
                else:
                    os.environ["NEBULA_SSH_CONFIG"] = old


if __name__ == "__main__":
    unittest.main()
