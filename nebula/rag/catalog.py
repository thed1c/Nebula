"""nebula/rag/catalog.py: Lookup's own SQL: collections, their documents and chunks, and a full-text index of the chunks.

Its own SQLite file (`<ledger>.lookup.sqlite3`). The vectors live in Nebula's own Qdrant (nebula/rag/vectors.py), keyed
by the chunk's id here; the text, the citations and who may see a collection live here.
"""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
import time

from nebula import migrate

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  folders TEXT NOT NULL DEFAULT '[]',
  watch INTEGER NOT NULL DEFAULT 1,
  shared TEXT NOT NULL DEFAULT '[]',      -- the tenants (caller:<id>, peer:<id>) that may search it; none by default
  embed TEXT,                             -- the embedding model it was indexed with
  dims INTEGER,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection TEXT NOT NULL,
  path TEXT NOT NULL,
  title TEXT,
  sha256 TEXT,
  size INTEGER,
  mtime REAL,
  state TEXT NOT NULL,                    -- queued | indexing | indexed | failed | skipped
  error TEXT,
  chunks INTEGER NOT NULL DEFAULT 0,
  pages INTEGER,
  parser TEXT,
  indexed_at REAL,
  UNIQUE(collection, path)
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc INTEGER NOT NULL,
  collection TEXT NOT NULL,
  ord INTEGER NOT NULL,
  text TEXT NOT NULL,
  heading TEXT,
  page INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, coalesce(new.heading, ''));
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, coalesce(old.heading, ''));
END;
"""

ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class Catalog:
    def __init__(self, path: str | None):
        self.db = sqlite3.connect(path or ":memory:", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path:
            self.db.execute("PRAGMA journal_mode=WAL")
        migrate.sqlite_upgrade(self.db, "Lookup's catalog", [SCHEMA], path)
        self.lock = threading.RLock()

    def close(self):
        self.db.close()

    def _tx(self, fn):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self.db)
                self.db.execute("COMMIT")
                return out
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    # --- collections
    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["folders"] = json.loads(d.get("folders") or "[]")
        d["shared"] = json.loads(d.get("shared") or "[]")
        d["watch"] = bool(d.get("watch"))
        return d

    def collections(self) -> list[dict]:
        with self.lock:
            return [self._row(r) for r in self.db.execute("SELECT * FROM collections ORDER BY created_at")]

    def collection(self, cid: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM collections WHERE id = ?", (cid,)).fetchone()
        return self._row(r) if r else None

    def add_collection(self, name: str, folders: list[str], watch: bool = True, cid: str | None = None) -> dict:
        cid = cid or (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "c") + "-" + secrets.token_hex(2)
        if not ID.match(cid):
            raise ValueError("a collection id is lowercase letters, digits and '-'")
        self._tx(lambda db: db.execute("INSERT INTO collections(id, name, folders, watch, created_at) VALUES (?, ?, ?, ?, ?)",
                                       (cid, name[:80], json.dumps(folders), int(bool(watch)), time.time())))
        return self.collection(cid)

    def update_collection(self, cid: str, **f) -> dict:
        sets, args = [], []
        for k in ("name", "watch", "embed", "dims"):
            if k in f:
                sets.append(f"{k} = ?")
                args.append(int(f[k]) if k == "watch" else f[k])
        for k in ("folders", "shared"):
            if k in f:
                sets.append(f"{k} = ?")
                args.append(json.dumps(list(f[k])))
        if sets:
            self._tx(lambda db: db.execute(f"UPDATE collections SET {', '.join(sets)} WHERE id = ?", (*args, cid)))
        return self.collection(cid)

    def remove_collection(self, cid: str) -> None:
        def go(db):
            db.execute("DELETE FROM chunks WHERE collection = ?", (cid,))
            db.execute("DELETE FROM documents WHERE collection = ?", (cid,))
            db.execute("DELETE FROM collections WHERE id = ?", (cid,))
        self._tx(go)

    # --- documents
    def document(self, cid: str, path: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM documents WHERE collection = ? AND path = ?", (cid, path)).fetchone()
        return dict(r) if r else None

    def document_by_id(self, did: int) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM documents WHERE id = ?", (int(did),)).fetchone()
        return dict(r) if r else None

    def documents(self, cid: str) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM documents WHERE collection = ? ORDER BY path", (cid,))]

    def mark(self, cid: str, path: str, state: str, **f) -> int:
        """A document's state (inserted if new); returns its id."""
        def go(db):
            db.execute("INSERT INTO documents(collection, path, state) VALUES (?, ?, ?) "
                       "ON CONFLICT(collection, path) DO UPDATE SET state = excluded.state", (cid, path, state))
            sets = [f"{k} = ?" for k in f]
            if sets:
                db.execute(f"UPDATE documents SET {', '.join(sets)} WHERE collection = ? AND path = ?", (*f.values(), cid, path))
            return db.execute("SELECT id FROM documents WHERE collection = ? AND path = ?", (cid, path)).fetchone()[0]
        return self._tx(go)

    def forget_document(self, cid: str, path: str) -> list[int]:
        """Remove a document and its chunks; returns the chunk ids (their vectors go too)."""
        def go(db):
            r = db.execute("SELECT id FROM documents WHERE collection = ? AND path = ?", (cid, path)).fetchone()
            if r is None:
                return []
            ids = [x[0] for x in db.execute("SELECT id FROM chunks WHERE doc = ?", (r[0],))]
            db.execute("DELETE FROM chunks WHERE doc = ?", (r[0],))
            db.execute("DELETE FROM documents WHERE id = ?", (r[0],))
            return ids
        return self._tx(go)

    def replace_chunks(self, did: int, cid: str, chunks: list[dict]) -> tuple[list[int], list[int]]:
        """(the old chunk ids, removed; the new ones, in order)."""
        def go(db):
            old = [x[0] for x in db.execute("SELECT id FROM chunks WHERE doc = ?", (did,))]
            db.execute("DELETE FROM chunks WHERE doc = ?", (did,))
            new = []
            for i, c in enumerate(chunks):
                cur = db.execute("INSERT INTO chunks(doc, collection, ord, text, heading, page) VALUES (?, ?, ?, ?, ?, ?)",
                                 (did, cid, i, c["text"], c.get("heading"), c.get("page")))
                new.append(cur.lastrowid)
            return old, new
        return self._tx(go)

    def chunk_ids(self, cid: str) -> list[int]:
        with self.lock:
            return [x[0] for x in self.db.execute("SELECT id FROM chunks WHERE collection = ?", (cid,))]

    def chunks_by_id(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        with self.lock:
            q = ("SELECT c.id, c.doc, c.collection, c.ord, c.text, c.heading, c.page, d.path, d.title FROM chunks c "
                 "JOIN documents d ON d.id = c.doc WHERE c.id IN (" + ",".join("?" * len(ids)) + ")")
            return {r["id"]: dict(r) for r in self.db.execute(q, [int(i) for i in ids])}

    def neighbours(self, did: int, ord_: int, around: int = 1) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT id, ord, text, heading, page FROM chunks WHERE doc = ? AND ord BETWEEN ? AND ? "
                                                     "ORDER BY ord", (did, ord_ - around, ord_ + around))]

    def keyword(self, query: str, collections: list[str], limit: int = 40) -> list[tuple[int, float]]:
        """(chunk id, bm25 score) by full text, best first, within those collections."""
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
        if not words or not collections:
            return []
        expr = " OR ".join(f'"{w}"' for w in words[:24])
        q = ("SELECT c.id, bm25(chunks_fts) AS s FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
             "WHERE chunks_fts MATCH ? AND c.collection IN (" + ",".join("?" * len(collections)) + ") ORDER BY s LIMIT ?")
        with self.lock:
            try:
                return [(r[0], -float(r[1])) for r in self.db.execute(q, (expr, *collections, int(limit)))]
            except sqlite3.OperationalError:
                return []

    def counts(self, cid: str) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT state, COUNT(*), COALESCE(SUM(chunks), 0) FROM documents WHERE collection = ? GROUP BY state",
                                   (cid,)).fetchall()
        out = {"documents": 0, "chunks": 0}
        for state, n, ch in rows:
            out[state] = n
            out["documents"] += n
            out["chunks"] += ch
        return out
