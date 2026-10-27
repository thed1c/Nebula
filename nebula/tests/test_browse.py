"""The folder picker's listing (nebula/browse.py): folders only, hidden and Lookup's skips left out unless asked, a
count of the documents Lookup can read right there; through the page API (owner only)."""
import os
import tempfile
import unittest

from nebula import browse


class BrowseTest(unittest.TestCase):
    def test_listing(self):
        d = tempfile.mkdtemp()
        for n in ("Docs", ".secret", "node_modules", "b"):
            os.makedirs(os.path.join(d, n))
        for n in ("a.md", "b.pdf", "c.bin"):
            open(os.path.join(d, n), "w").write("x")
        v = browse.listing(d)
        self.assertEqual([x["name"] for x in v["dirs"]], ["b", "Docs"])
        self.assertEqual((v["readable"], v["parent"]), (2, os.path.dirname(os.path.realpath(d))))
        self.assertIn(".secret", [x["name"] for x in browse.listing(d, hidden=True)["dirs"]])
        self.assertTrue(any(p["name"] == "Home" for p in v["places"]))
        with self.assertRaises(browse.BrowseError):
            browse.listing(os.path.join(d, "a.md"))                   # a file is not a folder
        self.assertIsNone(browse.listing(os.path.abspath(os.sep))["parent"])


if __name__ == "__main__":
    unittest.main()
