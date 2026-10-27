"""nebula/rag/ingest.py: a file into sections (text, heading, page), and sections into chunks.

- **Docling** (when installed, `pip install "nebula-eh[docling]"`) reads PDF, Word, PowerPoint, Excel, HTML and
  images (layout, tables, reading order, OCR) into Markdown with headings and pages.
- **Built in** (always): plain text, Markdown, reStructuredText, code, JSON, YAML, CSV and HTML (tags stripped).
  A PDF or an Office file without Docling is skipped, and says why.

Chunks follow the headings: a section is split on paragraphs into pieces of about `size` characters with an
`overlap`, and each piece keeps its heading path and page for the citation.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path

TEXT = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".cfg", ".css", ".csv", ".tsv", ".sh", ".c", ".h", ".cpp", ".hpp", ".go", ".rs", ".java", ".rb", ".sql", ".xml", ".log",
        ".tex", ".org", ".adoc"}
HTML = {".html", ".htm"}
DOCLING = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
MAX_FILE = 200 * 2**20
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", ".tox", ".mypy_cache"}


class Skip(Exception):
    """A file not read, and why (shown on the collection, not an error)."""


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def wanted(path: Path) -> bool:
    s = path.suffix.lower()
    return s in TEXT or s in HTML or s in DOCLING


def walk(folders: list[str]):
    """Every file under the folders that Lookup reads (hidden folders and build folders skipped)."""
    for f in folders:
        root = Path(os.path.expanduser(f))
        if root.is_file():
            if wanted(root):
                yield root.resolve()
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if not name.startswith(".") and wanted(p):
                    yield p.resolve()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --- reading
_converters: dict = {}


def _converter(ocr: bool):
    """Docling's converter: with OCR (scanned pages, images) when its OCR models are here, else without."""
    if ocr not in _converters:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        opts = PdfPipelineOptions()
        opts.do_ocr = ocr
        _converters[ocr] = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converters[ocr]


_NO_OCR = {"seen": False}


def _docling(path: Path) -> tuple[list[dict], int | None]:
    ocr = not _NO_OCR["seen"]
    try:
        res = _converter(ocr).convert(str(path))
    except FileNotFoundError as e:                     # its OCR models are not here: the text layer only
        if not ocr or "ocr" not in str(e).lower():
            raise
        _NO_OCR["seen"] = True
        res = _converter(False).convert(str(path))
    doc = res.document
    sections: list[dict] = []
    heads: list[str] = []
    cur: dict | None = None
    pages = None
    try:
        pages = len(doc.pages) if getattr(doc, "pages", None) is not None else None
    except Exception:  # noqa: BLE001
        pages = None
    for item, level in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower()
        text = getattr(item, "text", None)
        page = None
        prov = getattr(item, "prov", None)
        if prov:
            page = getattr(prov[0], "page_no", None)
        if "table" in label and hasattr(item, "export_to_markdown"):
            try:
                text = item.export_to_markdown(doc)
            except TypeError:
                text = item.export_to_markdown()
        if not text or not str(text).strip():
            continue
        if "header" in label or "title" in label:
            lvl = max(1, int(level or 1))
            heads = heads[:lvl - 1] + [str(text).strip()]
            cur = None
            continue
        if cur is None or (page is not None and cur.get("page") is not None and page != cur["page"]):
            cur = {"text": "", "heading": " › ".join(heads) or None, "page": page}
            sections.append(cur)
        cur["text"] += str(text).strip() + "\n\n"
    if not sections:                                   # nothing itemised: the whole document as Markdown
        sections = [{"text": doc.export_to_markdown(), "heading": None, "page": None}]
    return sections, pages


_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _markdown(body: str) -> list[dict]:
    sections, heads, cur = [], [], {"text": "", "heading": None, "page": None}
    fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        m = None if fence else _MD_HEAD.match(line)
        if m:
            if cur["text"].strip():
                sections.append(cur)
            lvl = len(m.group(1))
            heads = heads[:lvl - 1] + [m.group(2)]
            cur = {"text": "", "heading": " › ".join(heads), "page": None}
        else:
            cur["text"] += line + "\n"
    if cur["text"].strip():
        sections.append(cur)
    return sections


def _html(body: str) -> list[dict]:
    body = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", body)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    body = re.sub(r"(?is)<title[^>]*>.*?</title>", " ", body)
    body = re.sub(r"(?is)<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "\n" + "#" * int(m.group(1)) + " " + re.sub(r"<[^>]+>", "", m.group(2)) + "\n", body)
    body = re.sub(r"(?is)<(br|p|div|li|tr|section|article)[^>]*>", "\n", body)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    text = re.sub(r"[ \t]+", " ", text)
    secs = _markdown(re.sub(r"\n\s*\n+", "\n\n", text))
    if title and secs and not secs[0]["heading"]:
        secs[0]["heading"] = html.unescape(title.group(1).strip())[:120]
    return secs


def read(path: Path, use_docling: bool | None = None) -> tuple[list[dict], dict]:
    """(sections, info: parser, pages, title). Raises Skip for a file Lookup does not read."""
    s = path.suffix.lower()
    if path.stat().st_size > MAX_FILE:
        raise Skip(f"larger than {MAX_FILE // 2**20} MB")
    dl = docling_available() if use_docling is None else use_docling
    if s in DOCLING or (s in HTML and dl):
        if not dl:
            raise Skip(f"{s} files need Docling (pip install \"nebula-eh[docling]\")")
        sections, pages = _docling(path)
        return sections, {"parser": "docling", "pages": pages, "title": path.stem}
    body = path.read_text(encoding="utf-8", errors="replace")
    if s in HTML:
        return _html(body), {"parser": "html", "pages": None, "title": path.stem}
    if s in (".md", ".markdown"):
        secs = _markdown(body)
        title = next((x["heading"].split(" › ")[0] for x in secs if x.get("heading")), path.stem)
        return secs, {"parser": "markdown", "pages": None, "title": title}
    return [{"text": body, "heading": None, "page": None}], {"parser": "text", "pages": None, "title": path.name}


# --- chunking
def chunk(sections: list[dict], size: int = 1800, overlap: int = 250) -> list[dict]:
    """Pieces of about `size` characters, split on paragraphs (then lines, then words), each with its heading and
    page; neighbouring pieces of one section share `overlap` characters."""
    out = []
    for sec in sections:
        text = re.sub(r"\n{3,}", "\n\n", sec.get("text") or "").strip()
        if not text:
            continue
        parts = _split(text, size)
        prev_tail = ""
        for p in parts:
            body = (prev_tail + p).strip() if prev_tail else p
            out.append({"text": body, "heading": sec.get("heading"), "page": sec.get("page")})
            prev_tail = p[-overlap:] + "\n" if overlap and len(p) > overlap else ""
    return out


def _split(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    for sep in ("\n\n", "\n", ". ", " "):
        pieces = text.split(sep)
        if len(pieces) > 1:
            out, cur = [], ""
            for piece in pieces:
                add = piece + sep
                if cur and len(cur) + len(add) > size:
                    out.append(cur.strip())
                    cur = ""
                if len(add) > size:
                    out.extend(_split(add, size))
                else:
                    cur += add
            if cur.strip():
                out.append(cur.strip())
            return [o for o in out if o]
    return [text[i:i + size] for i in range(0, len(text), size)]
