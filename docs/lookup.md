# Lookup

Lookup is the star that gives Halo and your agents their tools. It runs MCP servers, and it has a built-in
**library**: document collections indexed on this machine and searched with citations.

## The library

- **A collection** is a name and one or more folders. Lookup reads every file it knows under them. Hidden folders and
  build folders are skipped. It splits each file into passages along its headings, embeds them on this machine and
  keeps them. A watched collection is checked every 30 seconds (`lookup.watch_seconds`): new and changed files are
  read again, and removed ones are forgotten.
- **What it reads:**
  - Plain text, Markdown, code, JSON, YAML, CSV and HTML, always.
  - PDF, Word, PowerPoint, Excel and images when Docling is installed (`pip install "nebula-eh[docling]"`), with
    pages, headings, tables and, when its OCR models are present, scanned pages. A file it cannot read is listed
    with the reason.
- **Search is hybrid.** It takes the closest passages by meaning (vectors) and by words (full text, BM25), fuses the
  two rankings, then reorders them with a cross-encoder. Every passage carries its citation: file, page and heading.
- **Answers cite.** The Lookup panel's *Ask* and Halo number the passages and cite them as [n].

## Where it is kept

Everything stays on this machine and in Nebula's own folders. Nebula never uses another Qdrant, and there is no
address to give it one.

- **Vectors: Nebula's own Qdrant.**
  - When its binary is here, Nebula starts it and stops it itself, on loopback at a free port, with an API key only
    Nebula holds (`nebula-keys/qdrant.key`, 0600) and its storage in `nebula-lookup/qdrant`. *Install Qdrant* on
    the Lookup panel downloads the pinned release and checks its sha256. A mismatch is refused.
  - Without the binary, Qdrant runs embedded in Nebula's own process (`nebula-lookup/qdrant-local`), which suits
    small libraries.
- **Text, citations and sharing: SQL.** They live in `<ledger>.lookup.sqlite3`, with a full-text index of the passages.
- **Embeddings:** fastembed on the CPU (`BAAI/bge-small-en-v1.5`, reranker `Xenova/ms-marco-MiniLM-L-6-v2`). Their
  files are fetched once from Hugging Face on first use. Instead, `"lookup": {"embed": {"tier": "<a local engine>"}}`
  uses that engine's `/v1/embeddings`. A cloud provider is never sent your documents.

Parts: `pip install "nebula-eh[lookup]"` (and `[docling]` for documents beyond text).

## Who sees what

- **You** see every collection.
- **A guest** (an app whose caller is marked guest, or a Gravity peer) sees only the collections you share with it,
  and none by default. Sharing is on the collection (`POST /api/lookup/collections/<id>/share`) and is written to the
  security log.
- **A tool call** runs as the tenant of the turn that makes it. An MCP client at `POST /mcp` runs as the tenant of
  its key.
- **The security log** (`security` tools) is yours and Halo's only. A guest is refused it, and it is never offered
  over `/mcp`.

## Tools

| Server | Tool | What |
|---|---|---|
| `library` | `search` | `{query, collections?, limit?}`: numbered passages with their citations |
| `library` | `read_passage` | `{chunk, around?}`: a passage with its neighbours |
| `library` | `collections` | the collections you may search |
| `security` | `read_log`, `counts` | the security log, read only (owner and Halo) |

`POST /mcp` serves the library's tools to MCP clients outside Nebula (JSON-RPC 2.0, one JSON answer per request).
The API Server's key rules apply.

## Settings (`"lookup"` in the config)

`library` (false turns it off), `watch_seconds`, `chunk_chars` (1800), `overlap_chars` (250), `rerank` (true),
`embed` (`{"model"}` or `{"tier", "model"}`), `vectors` (`auto`, `server` or `embedded`), `qdrant_bin`, `data_dir`,
`models_dir`.
