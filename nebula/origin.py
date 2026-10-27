"""nebula/origin.py: only this page drives this Nebula. Two checks on every request, before anything else:

- **Host** (DNS rebinding): a web site can point its own name at 127.0.0.1 and then talk to Nebula as "itself". So
  the Host a request names must be an address (127.0.0.1, 192.168.1.20, [::1]), `localhost`, this machine's own name
  (`mybox`, `mybox.local`, its full name), or one the owner lists:

      "allowed_hosts": ["nebula.home.arpa"]

- **Origin** (cross-site requests): a browser says which page sent a request (`Origin`, `Sec-Fetch-Site`). A request
  from another site's page, even another port on this machine, is refused on every write, and on the page's API and
  the model API's reads. Apps, scripts and other Nebulas send neither header and are not affected; their keys and
  pairing are checked as before.

A refusal is a 403 and a line in the security log (`auth.bad_host`, `auth.cross_site`), never the request's body.
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.parse

from nebula import lan

_NAMES: set[str] | None = None


def own_names() -> set[str]:
    """This machine's names, lower case: its host name, its full name, and both with .local (mDNS)."""
    global _NAMES
    if _NAMES is None:
        names = {"localhost"}
        for n in (lan.host_name(), socket.gethostname(), socket.getfqdn()):
            n = str(n or "").strip().lower().rstrip(".")
            if n:
                names.add(n)
                short = n.split(".")[0]
                names.update({short, f"{short}.local"})
        _NAMES = names
    return _NAMES


def split_host(value: str) -> str:
    """The name or address in a Host header or an origin's netloc, without its port, lower case."""
    v = str(value or "").strip().lower()
    if v.startswith("["):
        return v[1:v.find("]")] if "]" in v else v[1:]
    if v.count(":") > 1:                                  # a bare IPv6 address
        return v
    return v.rsplit(":", 1)[0] if ":" in v else v


def _is_address(h: str) -> bool:
    try:
        ipaddress.ip_address(h.split("%")[0])
        return True
    except ValueError:
        return False


def host_ok(cfg, host_header: str | None) -> bool:
    if not host_header:                                   # HTTP/1.0 without a Host: no name was used to get here
        return True
    h = split_host(host_header).rstrip(".")
    if not h or _is_address(h) or h in own_names() or h.endswith(".localhost"):
        return True
    extra = (getattr(cfg, "raw", None) or {}).get("allowed_hosts") or []
    return h in {str(x).strip().lower().rstrip(".") for x in extra if x}


def cross_site(headers) -> bool:
    """A browser request sent by another site's page (any other origin, another port on this machine too)."""
    origin = headers.get("Origin")
    site = (headers.get("Sec-Fetch-Site") or "").lower()
    if origin is not None:
        if origin == "null":                              # a sandboxed frame, a file: page
            return True
        o = urllib.parse.urlsplit(origin)
        return o.netloc.lower() != str(headers.get("Host") or "").lower()
    return site in ("cross-site", "same-site")


def guarded(method: str, path: str) -> bool:
    """Which requests a browser on another site may not send: every write; the page's API, the live view and the model
    API when read. The page itself, its fonts and /health may be opened from a link anywhere."""
    if method != "GET":
        return True
    return path.startswith(("/api/", "/v1/", "/switchyard/")) or path == "/mcp"


def check(handler, method: str, path: str) -> tuple[str, str] | None:
    """(the security log's kind, the reason) when this request is refused; None when it may go on."""
    if not host_ok(handler.cfg, handler.headers.get("Host")):
        return ("auth.bad_host", "this Nebula answers to its own names only: add this one to \"allowed_hosts\" in its "
                                 "settings if you use it on purpose")
    if guarded(method, path) and cross_site(handler.headers):
        return ("auth.cross_site", "another web site's page may not use this Nebula")
    return None
