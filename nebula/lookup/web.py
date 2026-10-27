"""Nebula's web lookup: search the web and read a page. An MCP server over stdio.

    python -m nebula.lookup.web [--provider auto|brave|duckduckgo]

Providers: Brave Search when a key is in the environment variable the config names (default BRAVE_API_KEY; the key
is never written to Nebula's config), else DuckDuckGo's keyless HTML page. A page is read as text (tags stripped),
capped.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.parse
import urllib.request

from nebula.lookup._server import serve

UA = "Mozilla/5.0 (X11; Linux x86_64) Nebula-Lookup/0.1"
TOOLS = [
    {"name": "web_search", "description": "Search the web. Returns titles, links and snippets.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                     "required": ["query"]}},
    {"name": "read_page", "description": "Read a web page as text.",
     "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
                     "required": ["url"]}},
]


def _get(url: str, headers: dict | None = None, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(3_000_000)


def page_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", raw)
    t = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return "\n".join(ln.strip() for ln in re.sub(r"[ \t]+", " ", t).splitlines() if ln.strip())


def parse_duckduckgo(raw: str, limit: int) -> list[dict]:
    out = []
    for m in re.finditer(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', raw):
        href, title, rest = m.group(1), m.group(2), m.group(3)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(href)).query)
        url = q.get("uddg", [html.unescape(href)])[0]
        snip = re.search(r'(?is)class="result__snippet"[^>]*>(.*?)</a>', rest)
        out.append({"title": page_text(title), "url": url, "snippet": page_text(snip.group(1)) if snip else ""})
        if len(out) >= limit:
            break
    return out


class Web:
    def __init__(self, provider: str = "auto", key_env: str = "BRAVE_API_KEY"):
        self.key = os.environ.get(key_env or "BRAVE_API_KEY") or None
        self.provider = ("brave" if self.key else "duckduckgo") if provider == "auto" else provider

    def search(self, query: str, limit: int = 6) -> str:
        limit = max(1, min(int(limit or 6), 20))
        if self.provider == "brave":
            if not self.key:
                raise RuntimeError("Brave Search needs its key in the environment")
            url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": limit})
            d = json.loads(_get(url, {"Accept": "application/json", "X-Subscription-Token": self.key}))
            rows = [{"title": r.get("title"), "url": r.get("url"), "snippet": page_text(r.get("description") or "")}
                    for r in (d.get("web") or {}).get("results") or []][:limit]
        else:
            raw = _get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})).decode("utf-8", "replace")
            rows = parse_duckduckgo(raw, limit)
        if not rows:
            return f"no results for {query!r}"
        return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in rows)

    def read(self, url: str, max_chars: int = 20000) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("only http(s) pages")
        t = page_text(_get(url).decode("utf-8", "replace"))
        n = max(1000, min(int(max_chars or 20000), 100000))
        return t[:n] + (f"\n\n[cut at {n} of {len(t)} characters]" if len(t) > n else "")

    def call(self, tool: str, a: dict) -> str:
        if tool == "web_search":
            return self.search(str(a.get("query") or ""), a.get("limit") or 6)
        if tool == "read_page":
            return self.read(str(a.get("url") or ""), a.get("max_chars") or 20000)
        raise ValueError(f"no tool {tool}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="nebula.lookup.web")
    p.add_argument("--provider", default="auto", choices=["auto", "brave", "duckduckgo"])
    p.add_argument("--key-env", default="BRAVE_API_KEY")
    a = p.parse_args(argv)
    serve("nebula-web", TOOLS, Web(a.provider, a.key_env).call)


if __name__ == "__main__":
    main()
