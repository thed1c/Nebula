"""nebula/rag/api.py: the page's API for Lookup's library (the owner's; a guest reaches it only through its tools).

    GET  /api/lookup                              the library: state, Nebula's own Qdrant, the collections (lit = indexed)
    GET  /api/lookup/collections/<id>/documents   its files: state, chunks, pages, why one was skipped or failed
    POST /api/lookup/collections                  {name, folders, watch}: a new collection, indexed at once
    POST /api/lookup/collections/<id>             {name?, folders?, watch?}
    POST /api/lookup/collections/<id>/share       {tenants: ["caller:<id>", "peer:<id>"]}: who else may search it ([] = none)
    POST /api/lookup/collections/<id>/reindex | /remove
    POST /api/lookup/search                       {query, collections?, limit?}: passages with citations
    POST /api/lookup/ask                          {question, collections?}: an answer from the Assistant's model, citing [n]
    POST /api/lookup/qdrant/install               Nebula's own Qdrant: the pinned release, checked, then used
"""
from __future__ import annotations

import os
import re

from nebula.rag.library import ANSWER_SYSTEM, LibraryError, cite_block


def _lib(server):
    lib = getattr(server, "library", None)
    if lib is None:
        raise LibraryError(503, getattr(server, "library_why", None) or "Lookup's library is off")
    return lib


def get(server, path: str):
    if path == "/api/lookup":
        lib = getattr(server, "library", None)
        if lib is None:
            return {"state": "unavailable", "why": getattr(server, "library_why", None), "collections": []}
        return lib.view()
    m = re.match(r"^/api/lookup/collections/([^/]+)/documents$", path)
    if m:
        return {"documents": _lib(server).documents(m.group(1))}
    return None


def post(server, path: str, req: dict):
    if path == "/api/lookup/qdrant/install":
        return install_qdrant(server)
    lib = _lib(server)
    if path == "/api/lookup/collections":
        return lib.add(req.get("name"), req.get("folders") or ([req["folder"]] if req.get("folder") else []), req.get("watch", True))
    m = re.match(r"^/api/lookup/collections/([^/]+)(?:/(share|reindex|remove))?$", path)
    if m:
        cid, action = m.group(1), m.group(2)
        if action == "share":
            if not isinstance(req.get("tenants"), list):
                raise LibraryError(400, "tenants is a list ([] shares it with no one)")
            return lib.share(cid, req["tenants"])
        if action == "reindex":
            lib.reindex(cid)
            return {"queued": cid}
        if action == "remove":
            lib.remove(cid)
            return {"removed": cid}
        return lib.change(cid, **{k: req[k] for k in ("name", "folders", "watch") if k in req})
    if path == "/api/lookup/search":
        return {"results": lib.search(req.get("query"), "owner", req.get("collections") or None, req.get("limit") or 8)}
    if path == "/api/lookup/ask":
        return ask(server, str(req.get("question") or ""), req.get("collections") or None)
    return None


def ask(server, question: str, collections=None) -> dict:
    """The passages, then one answer through this Nebula's own Switchyard (the Assistant's model, its gates, its
    ledger), citing them as [n]."""
    from nebula import stars
    from nebula.dialects import Context, JsonReply
    from nebula.dialects.chat import chat
    lib = _lib(server)
    results = lib.search(question, "owner", collections, 8)
    if not results:
        return {"answer": "Nothing in your collections speaks to that.", "citations": []}
    tier, model = stars.route_of(server.cfg, "assistant")
    req = {"messages": [{"role": "system", "content": ANSWER_SYSTEM},
                        {"role": "user", "content": f"Passages:\n\n{cite_block(results)}\n\nQuestion: {question}"}],
           "stream": False, **({"model": tier} if tier else {})}
    ctx = Context(server.cfg, server.ledger, {"X-Nebula-Caller": "lookup/ask"}, server.engines, None, input="lookup",
                  caller={"id": None, "label": "lookup/ask", "guest": False},
                  registry=server.registry, admission=server.admission, lot=server.lot, policy=server.policy,
                  yard=server.yard, power=server.power, slots=server.slots, seclog=getattr(server, "seclog", None), otel=getattr(server, "otel", None))
    reply = chat(server.engine, ctx, req)
    if not isinstance(reply, JsonReply) or reply.status != 200:
        raise LibraryError(502, "the model gave no answer")
    text = ((reply.body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(results)})
    return {"answer": text, "citations": [{k: r[k] for k in ("n", "title", "path", "page", "heading", "collection_name", "chunk")}
                                          for r in results if r["n"] in cited] or
            [{k: r[k] for k in ("n", "title", "path", "page", "heading", "collection_name", "chunk")} for r in results[:3]],
            "passages": len(results)}


def install_qdrant(server) -> dict:
    """The pinned Qdrant release into Nebula's data folder, then the library again on it (its collections are
    indexed into it afresh)."""
    from nebula.rag.vectors import install
    from nebula.server import lookup_library
    from nebula import ui
    lib = getattr(server, "library", None)
    base = server.cfg.base_dir or os.getcwd()
    data = (server.cfg.raw.get("lookup") or {}).get("data_dir") or os.path.join(base, "nebula-lookup")
    path = install(data)
    if lib is not None:
        lib.close()
    server.library = lookup_library(server)
    ui._reload_tools(server)
    return {"installed": path, **(server.library.view() if server.library else {})}
