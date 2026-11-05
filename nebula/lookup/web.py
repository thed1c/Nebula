"""Nebula's web lookup: search the web and read a page. An MCP server over stdio.

    python -m nebula.lookup.web [--provider auto|brave|duckduckgo]

Providers: Brave Search when a key is in the environment variable the config names (default BRAVE_API_KEY; the key
is never written to Nebula's config), else DuckDuckGo's keyless HTML page. A page is read as text (tags stripped),
capped.

`read_page` reaches the public web only: an address on this machine or the home network (127.0.0.1, 192.168/16,
10.x, fd00::/8, link-local, ...) is refused, and so is a redirect to one. A page a model reads could otherwise send it
to Nebula's own API on this machine (which opens without a PIN here) or to a router's admin page.
`--allow-private` (the web lookup's "allow_private": true) lifts that for someone who wants it.
"""
from __future__ import annotations

import argparse
import html
import http.client
import ipaddress
import json
import os
import re
import socket
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


class PrivateAddress(ValueError):
    pass


def _public_ip(host: str, value: str):
    ip = ipaddress.ip_address(value.split("%")[0])
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast:
        raise PrivateAddress(f"{host} is on this machine or the home network ({ip}): read_page reads public pages only")
    return ip


def _resolve_public(host: str, port: int) -> list:
    """The host's addresses, every one of them public (else PrivateAddress)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"{host} does not resolve: {e}") from None
    for info in infos:
        _public_ip(host, info[4][0])
    return infos


def check_public(url: str) -> None:
    """Refuses a URL whose host is, or resolves to, an address that is not on the public internet. The connection
    itself checks again, on the very address it connects to (_public_socket): this one says it early."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise PrivateAddress("only http(s) pages")
    _resolve_public(u.hostname, u.port or (443 if u.scheme == "https" else 80))


def _public_socket(host: str, port: int, timeout, source_address=None) -> socket.socket:
    """A connection to `host` on an address checked in the same step that connects to it: a name that answered a
    public address to an earlier check and a private one now (DNS rebinding) cannot slip in between."""
    last = None
    for family, kind, proto, _canon, addr in _resolve_public(host, port):
        sock = socket.socket(family, kind, proto)
        try:
            if timeout is not None and timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(addr)
            return sock
        except OSError as e:
            last = e
            sock.close()
    raise last or OSError(f"{host}: no address to connect to")


class _PublicHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = _public_socket(self.host, self.port, self.timeout, self.source_address)


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        sock = _public_socket(self.host, self.port, self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)     # the name stays the TLS identity


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PublicHTTPConnection, req)


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PublicHTTPSConnection, req, context=self._context)


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public(newurl)                # a public page that redirects to a private one is refused too
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# no proxy from the environment (a proxy on the home network would be the address connected to), the address
# checked where it is connected to, every redirect checked
_PUBLIC_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _PublicHTTPHandler, _PublicHTTPSHandler,
                                             _CheckedRedirects)


def _get(url: str, headers: dict | None = None, timeout: float = 20.0, public_only: bool = False) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    if public_only:
        check_public(url)
        with _PUBLIC_OPENER.open(req, timeout=timeout) as r:
            return r.read(3_000_000)
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
    def __init__(self, provider: str = "auto", key_env: str = "BRAVE_API_KEY", allow_private: bool = False):
        self.allow_private = allow_private
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
        t = page_text(_get(url, public_only=not self.allow_private).decode("utf-8", "replace"))
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
    p.add_argument("--allow-private", action="store_true", help="read_page may reach this machine and the home network")
    a = p.parse_args(argv)
    serve("nebula-web", TOOLS, Web(a.provider, a.key_env, a.allow_private).call)


if __name__ == "__main__":
    main()
