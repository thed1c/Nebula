"""A Flash-Next engine's opt-in long context (engines.py "long_context", impulse-fx --long-context): off unless asked for,
only for an impulse-fx command, and the flag placed before --fx-args (which takes the rest of the line)."""
import unittest

from nebula import engines as E
from nebula import flashnext as F

FX = ["/opt/impulse/bin/impulse-fx", "-m", "/models/x", "--host", "127.0.0.1", "--port", "18123", "--fx-config", "/opt/c.json"]


def spec(**kw):
    return {"name": "bb", "kind": "command", "command": list(FX), "port": 18123, **kw}


class TestLongContext(unittest.TestCase):
    def test_off_by_default(self):
        s = E.check_spec(spec())
        self.assertNotIn("long_context", s)
        self.assertNotIn("--long-context", E.command_of(s))
        for off in ("", "off", None, False):
            s = E.check_spec(spec(long_context=off))
            self.assertNotIn("long_context", s)
            self.assertNotIn("--long-context", E.command_of(s))

    def test_flashnext_definitions_never_set_it(self):
        self.assertNotIn("long_context", F.REFERENCE_SETTINGS)
        cfg = F.run_config({"pack": "/m/p/pack", "dense": "/m/p/d-dense.gguf", "tokenizer": "/m/p/pack/tokenizer"}, name="x", port=1)
        self.assertNotIn("rope", cfg)
        self.assertNotIn("--rope-scaling", cfg["args"])

    def test_on(self):
        for v in E.LONG_CONTEXT:
            argv = E.command_of(E.check_spec(spec(long_context=v)))
            self.assertEqual(argv[-2:], ["--long-context", v])
        argv = E.command_of(E.check_spec(spec(long_context="512k", command=["/opt/impulse/bin/impulse-fx", "-m", "/x", "--fx-args", "--kv", "int8"])))
        i = argv.index("--long-context")
        self.assertLess(i, argv.index("--fx-args"))

    def test_refused(self):
        with self.assertRaises(E.EngineSpecError):
            E.check_spec(spec(long_context="2m"))
        with self.assertRaises(E.EngineSpecError):   # not a Flash-Next engine
            E.check_spec(spec(long_context="512k", command=["/usr/bin/python3", "-m", "x"]))
        with self.assertRaises(E.EngineSpecError):
            E.check_spec({"name": "g", "kind": "impulse", "model": "/m/x.gguf", "port": 1, "long_context": "512k"})


if __name__ == "__main__":
    unittest.main()
