"""nebula/netwhy.py: why a download could not happen, said in plain words, and the machine's own certificates for HTTPS.

    cause(e)          -> "the name does not resolve (no internet, or a DNS filter blocks it)", ... from any network error
                         (urllib, http.client, ssl, socket, requests, httpx), its causes followed
    probe(url)        -> None when the address answers, else its cause(); one small read (bytes 0-0), redirects followed
    tls_context()     -> on Windows and macOS, an SSLContext that trusts what the system trusts (truststore), else None

Why: an install on a network that cannot reach Hugging Face said only "the download kept breaking off" and a page of
fastembed ERROR lines (2026-10-31): the person who sees Setup's window needs the reason first, in a few words.
The system's certificates: Windows fills its root store as roots are first needed, and an antivirus or a company proxy
that inspects HTTPS adds its own root there; pip already trusts the system's store (truststore), so the downloads after
it do too. Only for downloads (hf.py, prereqs), never for PAIR's or Gravity's pinned TLS.
"""
from __future__ import annotations

import errno
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

_DNS = ("getaddrinfo", "name or service not known", "nodename nor servname", "no address associated", "temporary failure in name resolution",
        "[errno 11001]", "[errno 11002]", "[errno -2]", "[errno -3]", "name resolution")
_REFUSED = ("connection refused", "actively refused", "[errno 10061]", "[errno 111]")
_RESET = ("connection reset", "forcibly closed", "[errno 10054]", "[errno 104]", "connection aborted", "[errno 10053]",
          "remote end closed", "remotedisconnected", "incompleteread", "server disconnected", "peer closed")
_TIMEOUT = ("timed out", "timeout", "[errno 10060]", "[errno 110]")
_CERT = ("certificate_verify_failed", "certificate verify failed", "self signed certificate", "self-signed certificate",
         "unable to get local issuer", "certificate has expired", "not yet valid", "sslcertverification")
_UNREACH = ("network is unreachable", "no route to host", "[errno 10051]", "[errno 10065]", "[errno 101]", "[errno 113]")


def _chain(e: BaseException):
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        reason = getattr(e, "reason", None)
        nxt = reason if isinstance(reason, BaseException) else (e.__cause__ or e.__context__)
        e = nxt


def cause(e: BaseException) -> str:
    """A network error in a few plain words, its likely reason first."""
    chain = list(_chain(e))
    for x in chain:
        if isinstance(x, urllib.error.HTTPError):
            if x.code in (401, 403):
                return f"the server refused it ({x.code}: a filter or a gated file)"
            if x.code == 407:
                return "a proxy on this network wants a sign-in (407)"
            if x.code == 429:
                return "the server is busy for this network (429), try later"
            if x.code >= 500:
                return f"the server is having trouble ({x.code}), try later"
            return f"the server answered {x.code}"
    text = " | ".join(f"{type(x).__name__}: {x}" for x in chain).lower()
    kinds = {type(x).__name__ for x in chain}
    if any(isinstance(x, ssl.SSLCertVerificationError) for x in chain) or any(k in text for k in _CERT):
        return ("the connection's certificate is not trusted (an antivirus or a company proxy that inspects HTTPS, "
                "or this computer's clock is wrong)")
    if any(isinstance(x, socket.gaierror) for x in chain) or any(k in text for k in _DNS):
        return "the server's name does not resolve (no internet, or a DNS filter or this network blocks it)"
    if any(isinstance(x, ConnectionRefusedError) for x in chain) or any(k in text for k in _REFUSED):
        return "the connection was refused (a firewall or filter on this computer or network blocks it)"
    if any(k in text for k in _UNREACH) or any(getattr(x, "errno", None) in (errno.ENETUNREACH, errno.EHOSTUNREACH) for x in chain):
        return "the network cannot reach the server (no internet route)"
    if any("proxy" in k.lower() for k in kinds) or "tunnel connection failed" in text:
        return "the network's proxy would not pass it"
    if any(isinstance(x, (TimeoutError, socket.timeout)) for x in chain) or any(k in text for k in _TIMEOUT) \
            or kinds & {"ConnectTimeout", "ReadTimeout", "PoolTimeout", "WriteTimeout"}:
        return "the server did not answer in time (a blocked or very slow network)"
    if any(isinstance(x, (ConnectionResetError, ConnectionAbortedError)) for x in chain) or any(k in text for k in _RESET):
        return "the connection was cut partway (an unsteady link, or a filter that drops it)"
    if any(isinstance(x, ssl.SSLError) for x in chain):
        return f"the secure connection failed ({str(chain[-1])[:100]})"
    return str(chain[-1] if chain else e)[:160] or type(e).__name__


def tls_context() -> ssl.SSLContext | None:
    """What the system trusts (Windows, macOS), through truststore when it is installed; else None (Python's default)."""
    if not (sys.platform == "win32" or sys.platform == "darwin"):
        return None
    try:
        import truststore
    except ImportError:
        return None
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def use_system_certs() -> bool:
    """For a process that only downloads (the installer's prereqs): every HTTPS client in it trusts what the system
    trusts. Never called in Nebula's own process, where PAIR and Gravity pin their own certificates."""
    if tls_context() is None:
        return False
    import truststore
    truststore.inject_into_ssl()
    return True


def probe(url: str, timeout: float = 20.0, headers: dict | None = None) -> str | None:
    """None when `url` answers (its first byte, redirects followed), else why not."""
    h = {"User-Agent": "nebula", "Range": "bytes=0-0", **(headers or {})}
    ctx = tls_context()
    try:
        req = urllib.request.Request(url, headers=h)
        with (urllib.request.urlopen(req, timeout=timeout, context=ctx) if ctx else urllib.request.urlopen(req, timeout=timeout)) as r:
            r.read(1)
        return None
    except Exception as e:  # noqa: BLE001 - said, not raised
        return cause(e)


def host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc or url
