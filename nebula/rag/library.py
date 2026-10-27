"""nebula/rag/library.py: Lookup's library: collections of documents, indexed and searched here (docs/lookup.md).

A **collection** is a name and folders. Every file Lookup reads under them is read (nebula/rag/ingest.py), cut into
chunks with their headings and pages, embedded on this machine (nebula/rag/embed.py) and kept in Nebula's own Qdrant
(nebula/rag/vectors.py) and SQL (nebula/rag/catalog.py). A watched collection is checked every `watch_seconds`: a new
or changed file is indexed again, a removed one forgotten.

**Search** is hybrid: the closest chunks by meaning (Qdrant) and by words (SQLite full text, BM25), fused by
reciprocal rank, then reordered by the reranker; each result carries its citation (file, page, heading).

**Who sees what:** the owner sees every collection; a guest (a caller marked guest, a Gravity peer) only the
collections shared with its tenant, and none by default. A tool call is the tenant of the turn that makes it.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path

from nebula.rag import ingest
from nebula.rag.catalog import Catalog

LOG = logging.getLogger("nebula.lookup")
OWNER = "owner"
RRF_K = 60


class LibraryError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def available() -> tuple[bool, str | None]:
    from nebula.rag.vectors import client_available
    from nebula.rag.embed import fastembed_available
    if not client_available():
        return False, 'Lookup\'s library needs its parts: pip install "nebula-eh[lookup]"'
    if not fastembed_available():
        return False, 'Lookup\'s library needs fastembed: pip install "nebula-eh[lookup]"'
    return True, None


class Library:
    def __init__(self, settings: dict, data_dir: str, catalog_path: str | None, key_path: str,
                 embedder=None, reranker=None, vectors=None, engine_of=None):
        self.settings = dict(settings or {})
        self.data_dir = data_dir
        self.key_path = key_path
        self.catalog = Catalog(catalog_path)
        self.embedder, self.reranker, self.vectors = embedder, reranker, vectors
        self.engine_of = engine_of                     # tier name -> Engine (an engine embedder's)
        self.state, self.why = "starting", None
        self.jobs: queue.Queue = queue.Queue()
        self.busy: dict[str, str] = {}                 # collection -> the file being indexed
        self.working = False                           # a job or a watch pass is running
        self.closing = threading.Event()
        self.listeners: list = []                      # called with (collection, event) as documents are indexed
        self.watch_s = float(self.settings.get("watch_seconds") or 30)
        self.thread = threading.Thread(target=self._run, name="lookup-library", daemon=True)

    # --- life
    def start(self) -> "Library":
        self.thread.start()
        return self

    def close(self) -> None:
        self.closing.set()
        self.jobs.put(None)
        self.thread.join(15)
        if self.vectors is not None:
            self.vectors.close()
        self.catalog.close()

    def _ready(self) -> None:
        """The embedder and the vectors (Nebula's own Qdrant), started on the library's own thread: a Qdrant server
        dies with the thread that started it."""
        if self.embedder is None:
            e = self.settings.get("embed") or {}
            if e.get("tier"):
                from nebula.rag.embed import EngineEmbedder
                eng = self.engine_of(e["tier"]) if self.engine_of else None
                if eng is None:
                    raise LibraryError(400, f"no engine {e['tier']!r} for embeddings")
                self.embedder = EngineEmbedder(eng, e["tier"], e.get("model"))
            else:
                from nebula.rag.embed import DEFAULT_EMBED, FastEmbedder
                self.embedder = FastEmbedder(e.get("model") or DEFAULT_EMBED, cache_dir=self.models_dir())
        if self.reranker is None and self.settings.get("rerank", True) and not (self.settings.get("embed") or {}).get("tier"):
            from nebula.rag.embed import Reranker
            self.reranker = Reranker(cache_dir=self.models_dir())
        if self.vectors is None:
            from nebula.rag.vectors import Vectors, find_binary
            mode = self.settings.get("vectors") or "auto"
            binary = find_binary(self.settings.get("qdrant_bin"), self.data_dir)
            if mode in ("auto", "server") and binary:
                self.vectors = Vectors("server", self.data_dir, binary, self.key_path)
            elif mode == "server":
                raise LibraryError(503, "Nebula's own Qdrant is not installed: install it from the Lookup star")
            else:
                self.vectors = Vectors("embedded", self.data_dir)

    def models_dir(self) -> str:
        """Where the embedding and reranker files are kept: the setting, else FASTEMBED_CACHE_PATH, else Nebula's own."""
        return self.settings.get("models_dir") or os.environ.get("FASTEMBED_CACHE_PATH") or os.path.join(self.data_dir, "models")

    def _up(self) -> bool:
        """The embedder and vectors, started the first time a collection needs them (a Nebula with no collection runs
        no Qdrant). False, with why, when they cannot start."""
        if self.state == "ready" and self.vectors is not None:
            return True
        self.state = "starting"
        try:
            self._ready()
            self.state, self.why = "ready", None
            return True
        except Exception as e:  # noqa: BLE001 - said on the Lookup star
            self.state, self.why = "unavailable", f"{type(e).__name__}: {e}"[:300]
            LOG.warning("Lookup's library is unavailable: %s", self.why)
            return False

    def _run(self) -> None:
        if self.catalog.collections():
            if not self._up():
                return
            for c in self.catalog.collections():
                self.jobs.put(("scan", c["id"]))
        else:
            self.state = "idle"                                # nothing to index yet: nothing started
        last_watch = time.time()
        while not self.closing.is_set():
            try:
                job = self.jobs.get(timeout=max(1.0, min(5.0, self.watch_s)))
            except queue.Empty:
                job = False
            if job is None:
                return
            self.working = True
            try:
                if job and job[0] != "drop" and not self._up():
                    continue
                if job:
                    self._do(job)
                if self.state == "ready" and time.time() - last_watch >= self.watch_s:
                    last_watch = time.time()
                    for c in self.catalog.collections():
                        if c["watch"]:
                            self._scan(c["id"])
            except Exception as e:  # noqa: BLE001
                LOG.warning("Lookup: %s failed: %s", job, e, exc_info=True)
            finally:
                self.working = False

    def _do(self, job) -> None:
        kind, cid = job[0], job[1]
        if kind == "scan":
            self._scan(cid)
        elif kind == "reindex":
            self._scan(cid, force=True)
        elif kind == "drop" and self.vectors is not None:
            self.vectors.drop(cid)

    def wait_idle(self, timeout: float = 120) -> bool:
        """Tests and the wizard: until the queue is empty and nothing is being indexed."""
        end = time.time() + timeout
        while time.time() < end:
            if self.state != "starting" and self.jobs.empty() and not self.busy and not self.working:
                time.sleep(0.2)
                if self.jobs.empty() and not self.busy and not self.working:
                    return True
            time.sleep(0.1)
        return False

    # --- indexing
    def _scan(self, cid: str, force: bool = False) -> None:
        c = self.catalog.collection(cid)
        if c is None:
            return
        dims = self.embedder.dims()
        if c.get("embed") != self.embedder.name or c.get("dims") != dims:
            force = True                                       # another embedding model: everything again
            self.catalog.update_collection(cid, embed=self.embedder.name, dims=dims)
        self.vectors.ensure(cid, dims)
        if not force and self.vectors.count(cid) == 0 and self.catalog.counts(cid)["chunks"] > 0:
            force = True                                       # a new vector store (Qdrant installed): everything again
        seen = set()
        for p in ingest.walk(c["folders"]):
            if self.closing.is_set():
                return
            path = str(p)
            seen.add(path)
            try:
                st = p.stat()
            except OSError:
                continue
            d = self.catalog.document(cid, path)
            if not force and d and d.get("mtime") == st.st_mtime and d.get("size") == st.st_size and d["state"] in ("indexed", "skipped"):
                continue
            self._index(cid, p, st, d, force)
        for d in self.catalog.documents(cid):                   # removed from the folders: forgotten
            if d["path"] not in seen:
                self.vectors.delete(cid, self.catalog.forget_document(cid, d["path"]))
                self._tell(cid, "removed", d["path"])

    def _index(self, cid: str, p: Path, st, d: dict | None, force: bool) -> None:
        path = str(p)
        self.busy[cid] = path
        try:
            sha = ingest.sha256_of(p)
            if not force and d and d.get("sha256") == sha and d["state"] == "indexed":
                self.catalog.mark(cid, path, "indexed", mtime=st.st_mtime, size=st.st_size)
                return
            did = self.catalog.mark(cid, path, "indexing", sha256=sha, mtime=st.st_mtime, size=st.st_size, error=None)
            try:
                sections, info = ingest.read(p)
            except ingest.Skip as e:
                old, _ = self.catalog.replace_chunks(did, cid, [])
                self.vectors.delete(cid, old)
                self.catalog.mark(cid, path, "skipped", error=str(e), chunks=0)
                self._tell(cid, "skipped", path)
                return
            chunks = ingest.chunk(sections, int(self.settings.get("chunk_chars") or 1800), int(self.settings.get("overlap_chars") or 250))
            vecs = self.embedder.embed([(c["heading"] + "\n" if c.get("heading") else "") + c["text"] for c in chunks]) if chunks else []
            old, new = self.catalog.replace_chunks(did, cid, chunks)
            self.vectors.delete(cid, old)
            self.vectors.upsert(cid, new, vecs, did)
            self.catalog.mark(cid, path, "indexed", chunks=len(new), pages=info.get("pages"), parser=info.get("parser"),
                              title=str(info.get("title") or p.name)[:200], indexed_at=time.time(), error=None)
            self._tell(cid, "indexed", path)
        except Exception as e:  # noqa: BLE001 - one file's failure is shown on it, the rest go on
            self.catalog.mark(cid, path, "failed", error=f"{type(e).__name__}: {e}"[:300])
            self._tell(cid, "failed", path)
        finally:
            self.busy.pop(cid, None)

    def _tell(self, cid: str, event: str, path: str) -> None:
        for f in list(self.listeners):
            try:
                f(cid, event, path)
            except Exception:  # noqa: BLE001
                pass

    # --- collections (the owner's)
    def add(self, name: str, folders: list[str], watch: bool = True) -> dict:
        name = str(name or "").strip()
        if not name:
            raise LibraryError(400, "a collection needs a name")
        folders = [os.path.abspath(os.path.expanduser(str(f))) for f in folders or [] if str(f).strip()]
        if not folders:
            raise LibraryError(400, "a collection needs a folder (or a file)")
        missing = [f for f in folders if not os.path.exists(f)]
        if missing:
            raise LibraryError(400, f"{missing[0]} does not exist on this machine")
        c = self.catalog.add_collection(name, folders, watch)
        self.jobs.put(("scan", c["id"]))
        return c

    def change(self, cid: str, **f) -> dict:
        if self.catalog.collection(cid) is None:
            raise LibraryError(404, f"no collection {cid!r}")
        if "folders" in f:
            f["folders"] = [os.path.abspath(os.path.expanduser(str(x))) for x in f["folders"] if str(x).strip()]
        c = self.catalog.update_collection(cid, **{k: v for k, v in f.items() if k in ("name", "folders", "watch")})
        self.jobs.put(("scan", cid))
        return c

    def share(self, cid: str, tenants: list[str]) -> dict:
        if self.catalog.collection(cid) is None:
            raise LibraryError(404, f"no collection {cid!r}")
        bad = [t for t in tenants if not (str(t).startswith("caller:") or str(t).startswith("peer:"))]
        if bad:
            raise LibraryError(400, f"{bad[0]!r} is not a guest (caller:<id> or peer:<id>)")
        return self.catalog.update_collection(cid, shared=list(dict.fromkeys(tenants)))

    def reindex(self, cid: str) -> None:
        if self.catalog.collection(cid) is None:
            raise LibraryError(404, f"no collection {cid!r}")
        self.jobs.put(("reindex", cid))

    def remove(self, cid: str) -> None:
        if self.catalog.collection(cid) is None:
            raise LibraryError(404, f"no collection {cid!r}")
        self.catalog.remove_collection(cid)
        if self.vectors is not None:
            self.vectors.drop(cid)

    # --- who sees what
    def visible(self, tenant: str | None) -> list[dict]:
        cs = self.catalog.collections()
        if not tenant or tenant == OWNER:
            return cs
        return [c for c in cs if tenant in (c.get("shared") or [])]

    def view(self, tenant: str | None = OWNER) -> dict:
        out = []
        for c in self.visible(tenant):
            n = self.catalog.counts(c["id"])
            row = {"id": c["id"], "name": c["name"], "documents": n["documents"], "chunks": n["chunks"],
                   "indexed": n.get("indexed", 0), "failed": n.get("failed", 0), "skipped": n.get("skipped", 0),
                   "waiting": n.get("queued", 0) + n.get("indexing", 0), "indexing": self.busy.get(c["id"]),
                   "lit": n.get("indexed", 0) > 0 and not self.busy.get(c["id"])}
            if not tenant or tenant == OWNER:
                row.update(folders=c["folders"], watch=c["watch"], shared=c["shared"], embed=c.get("embed"))
            out.append(row)
        v = {"state": self.state, "why": self.why, "collections": out}
        if not tenant or tenant == OWNER:
            v["vectors"] = self.vectors.describe() if self.vectors is not None else None
            v["embedder"] = getattr(self.embedder, "name", None)
            v["reranker"] = getattr(self.reranker, "model_name", None) if self.reranker else None
            v["docling"] = ingest.docling_available()
        return v

    def documents(self, cid: str, tenant: str | None = OWNER) -> list[dict]:
        if cid not in [c["id"] for c in self.visible(tenant)]:
            raise LibraryError(404, f"no collection {cid!r}")
        keep = ("id", "path", "title", "state", "error", "chunks", "pages", "parser", "indexed_at")
        return [{k: d.get(k) for k in keep} for d in self.catalog.documents(cid)]

    # --- search
    def search(self, query: str, tenant: str | None = OWNER, collections: list[str] | None = None, limit: int = 8) -> list[dict]:
        query = str(query or "").strip()
        if not query:
            raise LibraryError(400, "an empty question")
        allowed = [c["id"] for c in self.visible(tenant)]
        cids = [c for c in (collections or allowed) if c in allowed]
        if not cids:
            return []
        if self.state != "ready":
            raise LibraryError(503, self.why or "Lookup's library is starting")
        limit = max(1, min(int(limit or 8), 30))
        dense = self.vectors.search(cids, self.embedder.embed_query(query), 40)
        sparse = self.catalog.keyword(query, cids, 40)
        fused: dict[int, float] = {}
        for ranked in (dense, sparse):
            for rank, (cid_, _s) in enumerate(ranked):
                fused[cid_] = fused.get(cid_, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:30]
        rows = self.catalog.chunks_by_id(top)
        cands = [rows[i] for i in top if i in rows and rows[i]["collection"] in cids]
        if self.reranker is not None and cands:
            try:
                scores = self.reranker.rerank(query, [c["text"] for c in cands])
                for c, s in zip(cands, scores):
                    c["score"] = s
                cands.sort(key=lambda c: -c["score"])
            except Exception as e:  # noqa: BLE001 - the fused order stands
                LOG.warning("Lookup: the reranker failed: %s", e)
        names = {c["id"]: c["name"] for c in self.catalog.collections()}
        out = []
        for n, c in enumerate(cands[:limit], 1):
            out.append({"n": n, "collection": c["collection"], "collection_name": names.get(c["collection"]),
                        "doc": c["doc"], "chunk": c["id"], "path": c["path"], "title": c.get("title") or os.path.basename(c["path"]),
                        "page": c.get("page"), "heading": c.get("heading"), "text": c["text"],
                        "score": round(float(c.get("score", fused.get(c["id"], 0.0))), 4)})
        return out

    def passage(self, chunk: int, tenant: str | None = OWNER, around: int = 1) -> dict:
        """A search result's chunk with its neighbours (read more round a citation)."""
        rows = self.catalog.chunks_by_id([int(chunk)])
        c = rows.get(int(chunk))
        if c is None or c["collection"] not in [x["id"] for x in self.visible(tenant)]:
            raise LibraryError(404, "no such passage")
        near = self.catalog.neighbours(c["doc"], c["ord"], max(0, min(int(around), 3)))
        return {"path": c["path"], "title": c.get("title"), "page": c.get("page"), "heading": c.get("heading"),
                "text": "\n\n".join(x["text"] for x in near)}


def cite_block(results: list[dict]) -> str:
    """The passages as the model reads them: numbered, each with where it is from."""
    out = []
    for r in results:
        where = r["title"] + (f", page {r['page']}" if r.get("page") else "") + (f" ({r['heading']})" if r.get("heading") else "")
        out.append(f"[{r['n']}] {where}\n{r['text']}")
    return "\n\n".join(out)


ANSWER_SYSTEM = ("Answer the question from the numbered passages only. Cite each claim with its passage number in "
                 "square brackets, like [2]. If the passages do not hold the answer, say so plainly.")
