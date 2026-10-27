"""Nebula's document lookup: search and read the text files in the folders the owner adds. An MCP server over stdio.

    python -m nebula.lookup.docs --folder ~/notes --folder ~/projects/docs

Only files under the given folders are searched or read. A search scores each file by how often the query's words
appear (whole-file text, case-insensitive) and returns the best files with a few matching lines each.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from nebula.lookup._server import serve

TEXT = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".html", ".htm", ".css", ".csv", ".tsv", ".sh", ".c", ".h", ".cpp", ".go", ".rs", ".java", ".rb", ".sql", ".xml", ".log"}
MAX_FILE = 2_000_000                                          # bytes: bigger files are skipped
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache"}

TOOLS = [
    {"name": "search_documents", "description": "Search the owner's document folders for files about a query. Returns "
     "the best matching files with their matching lines.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                     "required": ["query"]}},
    {"name": "read_document", "description": "Read a file from the owner's document folders (a path a search returned).",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}},
                     "required": ["path"]}},
    {"name": "list_folders", "description": "The folders this lookup searches.", "inputSchema": {"type": "object", "properties": {}}},
]


class Docs:
    def __init__(self, folders: list[str]):
        self.folders = [Path(os.path.expanduser(f)).resolve() for f in folders if f]

    def files(self):
        for root in self.folders:
            if root.is_file():
                yield root
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for f in filenames:
                    p = Path(dirpath) / f
                    if p.suffix.lower() in TEXT:
                        yield p

    def inside(self, path: str) -> Path:
        p = Path(os.path.expanduser(path)).resolve()
        if not any(p == r or r in p.parents for r in self.folders):
            raise PermissionError("that file is outside the document folders")
        return p

    def search(self, query: str, limit: int = 8) -> str:
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
        if not words:
            return "an empty query"
        if not self.folders:
            return "no document folders are set: add one on the Lookup star"
        hits = []
        for p in self.files():
            try:
                if p.stat().st_size > MAX_FILE:
                    continue
                body = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            low = body.lower()
            score = sum(low.count(w) for w in words) + 3 * sum(w in p.name.lower() for w in words)
            if not score:
                continue
            lines = [ln.strip() for ln in body.splitlines() if any(w in ln.lower() for w in words)][:3]
            hits.append((score, str(p), lines))
        hits.sort(key=lambda h: -h[0])
        if not hits:
            return f"nothing in {len(self.folders)} folder(s) matches {query!r}"
        out = []
        for score, path, lines in hits[:max(1, min(int(limit or 8), 30))]:
            out.append(f"{path} (score {score})\n" + "\n".join("  " + ln[:200] for ln in lines))
        return "\n\n".join(out)

    def read(self, path: str, max_chars: int = 20000) -> str:
        p = self.inside(path)
        body = p.read_text(encoding="utf-8", errors="replace")
        n = max(1000, min(int(max_chars or 20000), 200000))
        return body[:n] + (f"\n\n[cut at {n} of {len(body)} characters]" if len(body) > n else "")

    def call(self, tool: str, a: dict) -> str:
        if tool == "search_documents":
            return self.search(str(a.get("query") or ""), a.get("limit") or 8)
        if tool == "read_document":
            return self.read(str(a.get("path") or ""), a.get("max_chars") or 20000)
        if tool == "list_folders":
            return "\n".join(str(f) for f in self.folders) or "no folders"
        raise ValueError(f"no tool {tool}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="nebula.lookup.docs")
    p.add_argument("--folder", action="append", default=[])
    a = p.parse_args(argv)
    d = Docs(a.folder)
    serve("nebula-docs", TOOLS, d.call)


if __name__ == "__main__":
    main()
