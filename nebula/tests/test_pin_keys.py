"""The sign-in PIN fields take the number pad (nebula/ui/index.html pinKeys): Numpad0-9 type their digit even when the
key arrives as Home / End / arrows (Num Lock off), NumpadEnter submits, Backspace and Delete stay the browser's own,
anything not a digit is dropped. pinKeys is run in node with a stand-in input element."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"

HARNESS = r"""
%s
function field(max) {
  const h = {};
  return { value: "", maxLength: max, selectionStart: 0, selectionEnd: 0, h,
    addEventListener(t, f) { (h[t] = h[t] || []).push(f); },
    setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; } };
}
function key(el, code, k, mods) {
  let prevented = false;
  const e = Object.assign({ code, key: k, preventDefault() { prevented = true; } }, mods || {});
  for (const f of el.h.keydown || []) f(e);
  return prevented;
}
function type(el, ch) { el.value = el.value.slice(0, el.selectionStart) + ch + el.value.slice(el.selectionEnd);
  el.selectionStart = el.selectionEnd = el.selectionStart + ch.length; for (const f of el.h.input || []) f({}); }
const out = {};
let sent = 0;
const a = field(12); pinKeys(a, () => { sent++; });
out.numlock_on = [key(a, "Numpad1", "1"), key(a, "Numpad2", "2")];
out.numlock_off = [key(a, "Numpad3", "PageDown"), key(a, "Numpad7", "Home"), key(a, "Numpad0", "Insert")];
out.value = a.value;
a.selectionStart = a.selectionEnd = 1; key(a, "Numpad9", "PageUp"); out.middle = a.value; out.caret = a.selectionStart;
out.backspace = key(a, "Backspace", "Backspace"); out.del = key(a, "NumpadDecimal", "Delete");
out.ctrl = key(a, "Numpad5", "5", { ctrlKey: true }); out.after_ctrl = a.value;
key(a, "NumpadEnter", "Enter"); key(a, "Enter", "Enter"); out.sent = sent;
const b = field(4); pinKeys(b, null);
for (const c of ["Numpad1", "Numpad2", "Numpad3", "Numpad4", "Numpad5"]) key(b, c, "End");
out.capped = b.value; out.enter_without_submit = key(b, "NumpadEnter", "Enter");
const c = field(12); pinKeys(c, null); type(c, "4a2 ."); out.filtered = c.value;
console.log(JSON.stringify(out));
"""


def pin_keys_source() -> str:
    html = PAGE.read_text()
    m = re.search(r"// pinKeys:begin\n(.*?)\n\s*// pinKeys:end", html, re.S)
    if not m:
        raise AssertionError("pinKeys is not marked in index.html")
    return m.group(1)


class PinKeysTest(unittest.TestCase):
    def test_every_pin_field_uses_it(self):
        html = PAGE.read_text()
        for el in ("g-pin", "g-pin2", "si-pin"):
            self.assertIn(f'pinKeys($("{el}")', html, el)
        self.assertNotIn('$("g-pin").onkeydown', html)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_number_pad(self):
        r = subprocess.run(["node", "-e", HARNESS % pin_keys_source()], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["numlock_on"], [True, True])
        self.assertEqual(out["numlock_off"], [True, True, True])
        self.assertEqual(out["value"], "12370")
        self.assertEqual((out["middle"], out["caret"]), ("192370", 2))
        self.assertEqual((out["backspace"], out["del"]), (False, False))       # the browser's own
        self.assertEqual((out["ctrl"], out["after_ctrl"]), (False, "192370"))  # shortcuts pass through
        self.assertEqual(out["sent"], 2)                                         # NumpadEnter and Enter
        self.assertEqual(out["capped"], "1234")
        self.assertFalse(out["enter_without_submit"])
        self.assertEqual(out["filtered"], "42")


if __name__ == "__main__":
    unittest.main()
