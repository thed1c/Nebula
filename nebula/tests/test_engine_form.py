"""Editing an engine (nebula/ui/index.html engForm): Save changes this engine (its next deployment version, Undo goes
back), Save and restart while it runs, Save as a copy under its own name and port; the name is kept while editing (its
routes find it by name)."""
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


class EngineFormTest(unittest.TestCase):
    def test_edit_offers_save_save_and_restart_and_a_copy(self):
        html = PAGE.read_text()
        self.assertNotIn("Save as a new deployment", html)
        for want in ('${spec ? "Save" : "Add the engine"}', 'id="eng-saverun"', 'id="eng-copy"', "Save as a copy…",
                     '`<label>Name<input name="name" value="${esc(s.name)}" readonly',
                     "/restart`, {})", 'form.onsubmit = (ev) => { ev.preventDefault(); save(false); };'):
            self.assertIn(want, html)


if __name__ == "__main__":
    unittest.main()
