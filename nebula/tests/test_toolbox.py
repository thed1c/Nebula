"""The workspace toolbox (nebula/toolbox.py): every model reads, writes, patches and searches files in its own folder,
runs commands in OpenShell only, and every call is in the ledger. Codex reviews as a detached run."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from nebula import team, toolbox
from nebula.config import Tier
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_runs import calls, fake_codex
from nebula.tests.test_starweave_review import mode_of
from nebula.tests.test_starweave_work import is_teammate, lead, tool_names


class FakeBox:
    ran = []

    def __init__(self, folder, image, base_dir=None, packages=None):
        self.folder, self.packages = folder, packages

    def why_not(self):
        return None

    def run(self, command, timeout):
        FakeBox.ran.append((self.folder, command, timeout))
        return (1, "1 failed") if "fail" in command else (0, "OK")


def steps(*calls_):
    """A fake model that makes these toolbox calls one round at a time, then says what the last tool said."""
    todo = list(calls_)

    def script(body):
        if body["messages"][-1].get("role") == "tool" and not todo:
            return None
        if not any(n.startswith("box__") for n in tool_names(body)) or not todo:
            return None
        return [todo.pop(0)]
    return script


class PathTest(unittest.TestCase):
    def test_paths_stay_in_the_folder(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.symlink("/etc", os.path.join(d, "out"))
        self.assertEqual(toolbox.resolve(d, "src/a.py"), os.path.join(os.path.realpath(d), "src", "a.py"))
        self.assertEqual(toolbox.resolve(d, "/src/a.py"), os.path.join(os.path.realpath(d), "src", "a.py"))
        for bad in ("../x", "out/passwd", "/etc/passwd/../../x/../../etc", "~" + "/x"):
            with self.assertRaises(toolbox.BoxError, msg=bad):
                p = toolbox.resolve(d, bad)
                if not p.startswith(os.path.realpath(d)):
                    raise toolbox.BoxError("out")


class ToolboxTest(unittest.TestCase):
    def rig(self, **tb):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)
        p = mock.patch.object(toolbox, "WORKBOX", FakeBox)
        p.start()
        self.addCleanup(p.stop)
        toolbox._BOXES.clear()
        FakeBox.ran = []
        r = Rig(rules={"toolbox": {"mode": "on", "folder": self.folder, **tb}, **tb.pop("_rules", {})})
        self.addCleanup(r.stop)
        return r

    def test_every_model_gets_it_and_edits_its_folder(self):
        rig = self.rig()
        rig.a.script = steps({"name": toolbox.WRITE, "arguments": {"path": "src/app.py", "content": "x = 1\nprint(x)\n"}},
                             {"name": toolbox.PATCH, "arguments": {"path": "src/app.py", "edits": [{"old": "x = 1", "new": "x = 2"}]}},
                             {"name": toolbox.READ, "arguments": {"path": "src/app.py"}},
                             {"name": toolbox.SEARCH, "arguments": {"pattern": "print", "glob": "*.py"}},
                             {"name": toolbox.LIST, "arguments": {}},
                             {"name": toolbox.RUN, "arguments": {"command": "python3 -m unittest"}})
        reply = rig.chat([{"role": "user", "content": "make the app"}])
        self.assertEqual(sorted(tool_names(rig.a.requests[0])), sorted(toolbox.TOOLS))
        with open(os.path.join(self.folder, "src", "app.py")) as f:
            self.assertEqual(f.read(), "x = 2\nprint(x)\n")
        self.assertEqual(reply["content"], "Tool said: exit code 0\nOK")
        self.assertEqual(FakeBox.ran[0][1], "python3 -m unittest")
        reads = [m["content"] for m in rig.a.requests[3]["messages"] if m.get("role") == "tool"]
        self.assertIn("    1  x = 2", reads[-1])
        acts = [json.loads(r["payload"]) for r in rows(rig.path, "SELECT payload FROM events WHERE kind='box_action' ORDER BY seq")]
        self.assertEqual([a["tool"] for a in acts], ["write", "patch", "read", "search", "list", "run"])
        self.assertTrue(all(a["ok"] for a in acts))
        sid = rig.srv.ledger.summaries()[0]["session_id"]
        item = [i for i in rig.srv.ledger.view(sid)["items"] if i["kind"] == "reply"][-1]
        self.assertEqual(item["toolbox"], "read 1 file · edited 1 file · ran 1 command")

    def test_a_patch_that_does_not_match_changes_nothing(self):
        rig = self.rig()
        with open(os.path.join(self.folder, "a.txt"), "w") as f:
            f.write("one\none\n")
        rig.a.script = steps({"name": toolbox.PATCH, "arguments": {"path": "a.txt", "edits": [{"old": "one", "new": "two"}]}})
        reply = rig.chat([{"role": "user", "content": "fix"}])
        self.assertIn("occurs 2 times", reply["content"])
        with open(os.path.join(self.folder, "a.txt")) as f:
            self.assertEqual(f.read(), "one\none\n")

    def test_outside_the_folder_is_refused_and_recorded(self):
        rig = self.rig()
        rig.a.script = steps({"name": toolbox.READ, "arguments": {"path": "../../etc/passwd"}})
        reply = rig.chat([{"role": "user", "content": "peek"}])
        self.assertIn("is outside your folder", reply["content"])
        (act,) = [json.loads(r["payload"]) for r in rows(rig.path, "SELECT payload FROM events WHERE kind='box_action'")]
        self.assertFalse(act["ok"])

    def test_without_openshell_nothing_runs(self):
        rig = self.rig()
        with mock.patch.object(FakeBox, "why_not", lambda self: "OpenShell is not installed on this machine"):
            rig.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn(toolbox.RUN, tool_names(rig.a.requests[0]))
        self.assertIn(toolbox.WRITE, tool_names(rig.a.requests[0]))

    def test_a_model_left_out_and_a_models_own_folder(self):
        own = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, own, True)
        rig = self.rig(off=["fast", "think"], folders={"big": own})
        rig.chat([{"role": "user", "content": "hi"}])
        self.assertEqual([n for n in tool_names(rig.a.requests[0]) if n.startswith("box__")], [])
        rig.b.script = steps({"name": toolbox.WRITE, "arguments": {"path": "b.txt", "content": "big's"}})
        rig.chat([{"role": "user", "content": "hi"}], model="big")
        self.assertTrue(os.path.exists(os.path.join(own, "b.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.folder, "b.txt")))

    def test_off_by_default(self):
        rig = Rig()
        self.addCleanup(rig.stop)
        rig.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn(toolbox.READ, tool_names(rig.a.requests[0]))

    def test_the_settings_and_api(self):
        rig = self.rig()
        import urllib.request
        with urllib.request.urlopen(rig.base + "/api/toolbox", timeout=10) as r:
            v = json.loads(r.read())
        self.assertEqual((v["mode"], v["folder"], v["folder_ok"]["default"]), ("on", self.folder, True))
        with self.assertRaises(toolbox.BoxError):
            toolbox.check_settings({"mode": "on", "folder": "relative/path"})


class TeamFolderTest(unittest.TestCase):
    def test_a_teammate_works_in_the_folder_its_lead_gave_it(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        p = mock.patch.object(toolbox, "WORKBOX", FakeBox)
        p.start()
        self.addCleanup(p.stop)
        rig = Rig(rules={"starweave": {"mode": "on"}, "toolbox": {"mode": "on", "folder": folder}})
        self.addCleanup(rig.stop)
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "p", "to": "big", "task": "Write part one.", "folder": "part1"}]}))

        def big(body):
            if is_teammate(body) and body["messages"][-1].get("role") != "tool":
                return [{"name": toolbox.WRITE, "arguments": {"path": "one.txt", "content": "1"}}]
            return None
        rig.b.script = big
        rig.chat([{"role": "user", "content": "build it"}])
        self.assertTrue(os.path.exists(os.path.join(folder, "part1", "one.txt")))
        (item,) = rows(rig.path, "SELECT * FROM work_items")
        self.assertEqual(item["folder"], os.path.join(os.path.realpath(folder), "part1"))

    def test_halo_refuses_a_folder_outside_the_leads(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        rig = Rig(rules={"starweave": {"mode": "on"}, "toolbox": {"mode": "on", "folder": folder}})
        self.addCleanup(rig.stop)
        seen = []
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "p", "to": "big", "task": "W.", "folder": "../elsewhere"}]}),
                            then=lambda body: seen.append(body["messages"][-1]["content"]))
        rig.chat([{"role": "user", "content": "build it"}])
        self.assertTrue(any("is outside your folder" in s for s in seen), seen)


class CodexReviewsTest(unittest.TestCase):
    def test_big_does_the_work_and_codex_reviews_it(self):
        work = fake_codex(self)
        codex = Tier("codex", cli="codex", cwd=work, writes=True, slots=1)
        rig = Rig(rules={"starweave": {"mode": "on", "reviewer": "codex",
                                       "teams": {"everyday": {"lead": "fast", "helpers": ["big"], "reviewer": "codex"}}}},
                  extra_tiers=[codex])
        self.addCleanup(rig.stop)

        def then(body):
            if body["messages"][-1]["content"].startswith("[d] big: done"):
                return [{"name": team.REVIEW, "arguments": {"item": "d", "criteria": "it works"}}]
            return None
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "d", "to": "big", "task": "Write it."}]}), then=then)
        reply = rig.chat([{"role": "user", "content": "build it"}])
        self.assertIn("[d] approved by codex.", reply["content"])
        self.assertIn("summary: codex checked", reply["content"])
        (c,) = calls(work)
        self.assertEqual(c["args"][c["args"].index("-s") + 1], "read-only")          # a reviewer reads
        self.assertIn("Answer with one JSON object", c["prompt"])
        rv = next(i for i in rows(rig.path, "SELECT * FROM work_items") if i["mode"] == "review")
        self.assertEqual((rv["model"], rv["state"], json.loads(rv["result"])["verdict"]), ("codex", "done", "approve"))



class LinksOutTest(unittest.TestCase):
    """A link in the folder that leads out of it (a command can make one: ln -s ~/.ssh/id_ed25519 x) is never read
    through: search skips it, list names it without following it, read refuses it."""

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.base, ignore_errors=True))
        self.folder, outside = os.path.join(self.base, "work"), os.path.join(self.base, "secret.txt")
        os.makedirs(os.path.join(self.folder, "src"))
        with open(outside, "w") as f:
            f.write("SECRET-OUTSIDE\n")
        with open(os.path.join(self.folder, "src", "own.txt"), "w") as f:
            f.write("SECRET-INSIDE\n")
        os.symlink(outside, os.path.join(self.folder, "linked.txt"))
        os.symlink(os.path.join(self.folder, "src", "own.txt"), os.path.join(self.folder, "inner-link.txt"))

    def call(self, name, **a):
        return toolbox._call(None, self.folder, name, a)[0]

    def test_search_skips_a_link_out_and_keeps_one_inside(self):
        out = self.call(toolbox.SEARCH, pattern="SECRET")
        self.assertNotIn("OUTSIDE", out)
        self.assertIn("src/own.txt:1: SECRET-INSIDE", out)
        self.assertIn("inner-link.txt:1: SECRET-INSIDE", out)

    def test_list_names_a_link_out_without_following_it(self):
        out = self.call(toolbox.LIST)
        self.assertIn("linked.txt  (a link out of the folder: not followed)", out)

    def test_read_refuses_a_link_out(self):
        with self.assertRaises(toolbox.BoxError):
            self.call(toolbox.READ, path="linked.txt")
        with self.assertRaises(toolbox.BoxError):
            self.call(toolbox.SEARCH, pattern="SECRET", path="linked.txt")

    def test_a_big_file_is_read_a_piece_at_a_time(self):
        with open(os.path.join(self.folder, "big.log"), "w") as f:
            for n in range(1, 50001):
                f.write(f"line {n} " + "x" * 40 + "\n")
        out = self.call(toolbox.READ, path="big.log", start=10, lines=3)
        self.assertEqual(out.splitlines(), [f"{n:>5}  line {n} " + "x" * 40 for n in (10, 11, 12)])
        out = self.call(toolbox.READ, path="big.log")
        self.assertIn("of 50000: read from line", out.splitlines()[-1])
        self.assertLessEqual(len(out), toolbox.READ_CHARS + 200)

    def test_a_folder_swapped_for_a_link_after_the_check_is_not_followed(self):
        # the race: the path is checked, then a command swaps a folder on the way for a link out before the open
        outside = os.path.join(self.base, "elsewhere")
        os.makedirs(outside)
        with open(os.path.join(outside, "own.txt"), "w") as f:
            f.write("SECRET-OUTSIDE\n")
        checked = toolbox.resolve(self.folder, "src/own.txt")
        os.rename(os.path.join(self.folder, "src"), os.path.join(self.folder, "src-moved"))
        os.symlink(outside, os.path.join(self.folder, "src"))
        with self.assertRaises(toolbox.BoxError):
            toolbox._open_text(self.folder, checked)
        with self.assertRaises(toolbox.BoxError):
            toolbox._write_file(self.folder, checked, b"overwritten")
        with open(os.path.join(outside, "own.txt")) as f:
            self.assertEqual(f.read(), "SECRET-OUTSIDE\n")

    def test_a_file_swapped_for_a_link_is_not_written_through(self):
        target = os.path.join(self.folder, "src", "own.txt")
        checked = toolbox.resolve(self.folder, "src/own.txt")
        os.remove(target)
        os.symlink(os.path.join(self.base, "secret.txt"), target)
        with self.assertRaises(toolbox.BoxError):
            toolbox._write_file(self.folder, checked, b"overwritten")
        with open(os.path.join(self.base, "secret.txt")) as f:
            self.assertEqual(f.read(), "SECRET-OUTSIDE\n")

    def test_a_hard_link_is_not_written(self):
        os.link(os.path.join(self.base, "secret.txt"), os.path.join(self.folder, "hard.txt"))
        with self.assertRaises(toolbox.BoxError):
            toolbox._write(self.folder, toolbox.WRITE, {"path": "hard.txt", "content": "x"})
        with open(os.path.join(self.base, "secret.txt")) as f:
            self.assertEqual(f.read(), "SECRET-OUTSIDE\n")

    def test_a_very_long_line_is_cut_without_reading_it_whole(self):
        import tracemalloc
        with open(os.path.join(self.folder, "one-line.txt"), "w") as f:
            f.write("x" * (8 * 2**20) + "\nsecond\n")
        tracemalloc.start()
        out = self.call(toolbox.READ, path="one-line.txt")
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertLess(peak, 2 * 2**20)
        self.assertIn("    2  second", out)

    def test_a_patch_keeps_the_write_limit(self):
        p = os.path.join(self.folder, "src", "own.txt")
        with self.assertRaises(toolbox.BoxError):
            toolbox._write(self.folder, toolbox.PATCH, {"path": "src/own.txt",
                                                        "edits": [{"old": "SECRET-INSIDE", "new": "y" * (toolbox.WRITE_BYTES + 1)}]})
        with open(p) as f:
            self.assertEqual(f.read(), "SECRET-INSIDE\n")


class RootlessPodmanTest(unittest.TestCase):
    """On rootless Podman the sandbox's user is a sub-uid on the host: the folder is lent to it for each command and
    given back after, even when the command fails or runs out of time, and Nebula's own writes wait meanwhile."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.folder, ignore_errors=True))
        self.box = toolbox.WorkBox(self.folder)
        self.box.ready, self.box.lender = True, "/usr/bin/podman"
        self.calls, self.scripts = [], []
        toolbox._FOLDER_LOCKS.clear()

    def fake_run(self, argv, **kw):
        from types import SimpleNamespace
        if argv[1:3] == ["unshare", "chown"]:
            self.calls.append(("chown", argv[5]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        self.calls.append(("exec", argv[-1]))
        self.scripts.append(argv[-3])
        if "slow" in argv[-1]:
            import subprocess
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")

    def test_lent_for_each_command_and_given_back(self):
        with mock.patch.object(toolbox.subprocess, "run", self.fake_run), mock.patch("nebula.openshell.binary", lambda exe=None: "openshell"):
            self.assertEqual(self.box.run("true", 5)[0], 0)
            self.assertEqual(self.box.run("slow", 5)[0], 124)
        me = f"{os.getuid()}:{os.getgid()}"
        self.assertEqual([c for c in self.calls], [("chown", me), ("exec", "true"), ("chown", "0:0"),
                                                   ("chown", me), ("exec", "slow"), ("chown", "0:0")])

    def test_no_lending_without_rootless_podman(self):
        self.box.lender = None
        with mock.patch.object(toolbox.subprocess, "run", self.fake_run), mock.patch("nebula.openshell.binary", lambda exe=None: "openshell"):
            self.box.run("true", 5)
        self.assertEqual(self.calls, [("exec", "true")])

    def test_a_write_waits_for_a_running_command(self):
        import threading, time
        started, order = threading.Event(), []

        def slow_run(argv, **kw):
            from types import SimpleNamespace
            if argv[1:3] != ["unshare", "chown"]:
                started.set()
                time.sleep(0.5)
                order.append("command done")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        with mock.patch.object(toolbox.subprocess, "run", slow_run), mock.patch("nebula.openshell.binary", lambda exe=None: "openshell"):
            th = threading.Thread(target=self.box.run, args=("x", 5))
            th.start()
            started.wait(5)
            toolbox._write(self.folder, toolbox.WRITE, {"path": "a.txt", "content": "x"})
            order.append("write done")
            th.join()
        self.assertEqual(order, ["command done", "write done"])

    def test_the_packages_copy_is_mounted_read_only_and_on_the_path(self):
        box = toolbox.WorkBox(self.folder, packages=["/x/nebula-box-packages"])
        self.assertNotEqual(box.name, self.box.name)            # its own mounts: a sandbox of its own
        box.ready = True
        with mock.patch.object(toolbox.subprocess, "run", self.fake_run), mock.patch("nebula.openshell.binary", lambda exe=None: "openshell"):
            box.run("python3 -m unittest", 5)
        self.assertEqual(self.calls, [("exec", "python3 -m unittest")])
        self.assertTrue(self.scripts[0].startswith('export PYTHONPATH="/sandbox/deps${PYTHONPATH:+:$PYTHONPATH}"; cd /sandbox/work'))

    def test_the_sandbox_stops_the_command_and_keeps_the_end_of_its_output(self):
        # the command is $1 of a fixed line, never pasted into it; the sandbox's own timeout stops it (and what it
        # started); only the end of its output comes back; its exit code is the last line. Run here with this sh.
        import re as _re
        sc = self.box.script(2).replace("cd /sandbox/work", "cd " + self.folder)
        def run(cmd):
            out = subprocess.run(["sh", "-c", sc, "nebula-box", cmd], capture_output=True, timeout=30).stdout.decode()
            m = _re.search(r"\n?%s (\d+)\n?\s*$" % toolbox.EXIT_MARK, out)
            return int(m.group(1)), out[:m.start()]
        self.assertEqual(run("echo 'a \"b\" $HOME'; exit 3"), (3, 'a "b" $HOME\n'))
        self.assertEqual(run("sleep 20; echo never")[0], 124)
        code, out = run("yes x | head -c 3000000")
        self.assertEqual(code, 0)
        self.assertLessEqual(len(out), toolbox.RUN_KEEP)

    def test_a_lend_that_fails_is_given_back(self):
        from types import SimpleNamespace
        def failing(argv, **kw):
            if argv[1:3] == ["unshare", "chown"]:
                self.calls.append(("chown", argv[5]))
                ok = argv[5] == "0:0"
                return SimpleNamespace(returncode=0 if ok else 1, stdout="", stderr="" if ok else "half way")
            self.calls.append(("exec", argv[-1]))
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        with mock.patch.object(toolbox.subprocess, "run", failing), mock.patch("nebula.openshell.binary", lambda exe=None: "openshell"):
            with self.assertRaises(toolbox.BoxError):
                self.box.run("true", 5)
        self.assertEqual(self.calls, [("chown", f"{os.getuid()}:{os.getgid()}"), ("chown", "0:0")])

    def test_overlapping_folders_never_hold_their_locks_at_once(self):
        # child A, then its parent, then child B (the order that used to give B a lock of its own): the parent and
        # each child always share a lock; two sibling children do not wait for each other
        import threading
        a, b = os.path.join(self.folder, "a"), os.path.join(self.folder, "b")
        os.makedirs(a); os.makedirs(b)
        for f in (a, self.folder, b):
            toolbox.folder_lock(f)
        got = []

        def take(f, word):
            with toolbox.folder_lock(f):
                got.append(word)
        with toolbox.folder_lock(self.folder):
            th = threading.Thread(target=take, args=(b, "b"))
            th.start(); th.join(0.3)
            self.assertEqual(got, [])                  # B waits while its parent is held
        th.join(2)
        self.assertEqual(got, ["b"])
        with toolbox.folder_lock(a):                   # siblings: no shared lock
            th = threading.Thread(target=take, args=(b, "b2"))
            th.start(); th.join(2)
            self.assertEqual(got[-1], "b2")
        # a folder first seen while one inside it is held still waits for it
        top = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(top, ignore_errors=True))
        inner = os.path.join(top, "inner")
        os.makedirs(inner)
        with toolbox.folder_lock(inner):
            th = threading.Thread(target=take, args=(top, "top"))
            th.start(); th.join(0.3)
            self.assertNotEqual(got[-1], "top")
        th.join(2)
        self.assertEqual(got[-1], "top")

    def test_a_sandbox_without_timeout_or_tail_runs_nothing(self):
        import re as _re
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        os.symlink(shutil.which("sh"), os.path.join(d, "sh"))          # a PATH with sh only
        sc = self.box.script(2).replace("cd /sandbox/work", "cd " + self.folder)
        p = subprocess.run([os.path.join(d, "sh"), "-c", sc, "nebula-box", "echo ran"], capture_output=True, timeout=30,
                           env={"PATH": d})
        out = p.stdout.decode()
        m = _re.search(r"\n?%s (\d+)\n?\s*$" % toolbox.EXIT_MARK, out)
        self.assertEqual(int(m.group(1)), 125)
        self.assertNotIn("ran", out)
        self.assertIn("no timeout or tail", p.stderr.decode())

    def test_the_sandbox_name_fits_openshell(self):
        self.assertLessEqual(len(self.box.name), 19)


class PackagesCopyTest(unittest.TestCase):
    """What Nebula imports beyond the standard library goes into a copy the toolbox owns (never Nebula's own folders,
    which Podman would take over on a first mount), with what those import in turn, refreshed when versions change."""

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.base, ignore_errors=True))
        toolbox._PACKAGES.clear()

    def test_the_copy_holds_nebulas_dependencies_and_what_they_import(self):
        if "jinja2" not in toolbox._declared():
            self.skipTest("jinja2 is not installed here as a package (inside a sandbox it is the copy itself)")
        d = toolbox.packages_copy(self.base)
        self.assertEqual(d, os.path.join(self.base, "nebula-box-packages"))
        self.assertIn("jinja2", os.listdir(d))
        self.assertIn("markupsafe", os.listdir(d))              # found by importing jinja2 in a bare interpreter
        stamp = json.load(open(os.path.join(d, ".nebula-packages.json")))
        self.assertEqual(sorted(stamp["modules"]), ["jinja2", "markupsafe"])
        self.assertEqual(toolbox.package_dirs({"packages": "nebula"}, self.base), [d])
        self.assertEqual(toolbox.package_dirs({"packages": "off"}, self.base), [])

    def test_a_setting_other_than_nebula_or_off_is_refused(self):
        with self.assertRaises(toolbox.BoxError):
            toolbox.check_settings({"packages": ["/usr/lib/python3/dist-packages"]})

class SavedAtOnceTest(unittest.TestCase):
    """The Models page saves the toolbox at once, the whole section: the callers whose plain calls get the tools, and
    the packages copy, are kept with it."""
    def test_callers_and_packages_survive_a_save(self):
        from nebula import ui
        out = ui.check_rules({"toolbox": {"mode": "on", "folder": "/tmp/box", "off": ["cloudy"], "callers": ["harness"], "packages": "off"}})
        self.assertEqual(out["toolbox"]["callers"], ["harness"])
        self.assertEqual(out["toolbox"]["packages"], "off")
        with self.assertRaises(ui.UiError):
            ui.check_rules({"toolbox": {"callers": "harness"}})


if __name__ == "__main__":
    unittest.main()
