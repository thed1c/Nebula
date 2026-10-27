"""nebula/signin.py: signing in to the page, with a passkey or a PIN (docs/signin.md).

    "signin": {"pin": {"salt": "…", "scrypt": "…"}, "passkeys": [{"id": "…", "x": "…", "y": "…", "name": "…", …}]}

- **Until one is set**, the page asks for one first (on this machine; from anywhere else the page is closed anyway).
  Once set, every page API call needs a signed-in browser: a session cookie (HttpOnly, SameSite=Strict, this port's
  own name), good for SESSION_S.
- **A PIN**: 4 to 12 digits, kept only as an scrypt hash. Wrong PINs are counted per address: after FREE_TRIES the
  next try waits, doubling each time up to LOCK_MAX_S. Every sign-in and every refusal is in the security log.
- **A passkey** (WebAuthn): a key pair made by the browser, the phone or a security key; Nebula keeps only the public
  key (P-256, ES256) and checks each sign-in's signature, its challenge (used once, two minutes), the page's origin and
  that the person was verified (fingerprint, face, device PIN). A passkey needs a name for this machine, not an
  address: the page offers it at http://localhost:<port>, not at http://127.0.0.1:<port>.
- **Nebula's own callers** (Gravity's owner link, the household link, `nebula invite` on this machine) carry the
  server's internal token instead (`X-Nebula-Internal`; the file nebula-keys/admin.token, readable by its owner only).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.parse

SESSION_S = 12 * 3600
CHALLENGE_S = 120
FREE_TRIES, LOCK_MAX_S = 5, 3600
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


class SignInError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u(s: str) -> bytes:
    s = str(s or "")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# --- CBOR (RFC 8949), only what WebAuthn's attestation object and COSE keys use
def cbor(data: bytes, i: int = 0):
    """(the value at i, the index after it)."""
    ib = data[i]; major, info = ib >> 5, ib & 31; i += 1
    if info < 24:
        n = info
    elif info in (24, 25, 26, 27):
        size = 1 << (info - 24)
        n = int.from_bytes(data[i:i + size], "big"); i += size
    else:
        raise ValueError("indefinite CBOR lengths are not used by WebAuthn")
    if major == 0:
        return n, i
    if major == 1:
        return -1 - n, i
    if major == 2:
        return bytes(data[i:i + n]), i + n
    if major == 3:
        return data[i:i + n].decode("utf-8"), i + n
    if major == 4:
        out = []
        for _ in range(n):
            v, i = cbor(data, i); out.append(v)
        return out, i
    if major == 5:
        out = {}
        for _ in range(n):
            k, i = cbor(data, i); v, i = cbor(data, i); out[k] = v
        return out, i
    if major == 7 and info in (20, 21, 22):
        return {20: False, 21: True, 22: None}[info], i
    raise ValueError(f"CBOR major type {major} is not used by WebAuthn")


# --- ECDSA P-256 signature check (verify only: nothing secret is computed here)
P = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
A = P - 3
B = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
N = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
G = (0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
     0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5)


def _on_curve(pt) -> bool:
    x, y = pt
    return 0 <= x < P and 0 <= y < P and (y * y - (x * x * x + A * x + B)) % P == 0


def _add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1 + A) * pow(2 * y1, -1, P) % P
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return x3, (lam * (x1 - x3) - y1) % P


def _mul(k: int, pt):
    out = None
    while k:
        if k & 1:
            out = _add(out, pt)
        pt = _add(pt, pt)
        k >>= 1
    return out


def _der_sig(sig: bytes) -> tuple[int, int]:
    """r and s from an ASN.1 DER ECDSA signature (what WebAuthn sends for ES256)."""
    if len(sig) < 8 or sig[0] != 0x30 or sig[1] != len(sig) - 2:
        raise ValueError("not a DER signature")
    i, out = 2, []
    for _ in range(2):
        if sig[i] != 0x02:
            raise ValueError("not a DER integer")
        n = sig[i + 1]
        out.append(int.from_bytes(sig[i + 2:i + 2 + n], "big")); i += 2 + n
    if i != len(sig):
        raise ValueError("trailing bytes in the signature")
    return out[0], out[1]


def es256_verify(x: int, y: int, message: bytes, sig: bytes) -> bool:
    try:
        r, s = _der_sig(sig)
    except (ValueError, IndexError):
        return False
    q = (x, y)
    if not (1 <= r < N and 1 <= s < N) or not _on_curve(q):
        return False
    z = int.from_bytes(hashlib.sha256(message).digest(), "big")
    w = pow(s, -1, N)
    pt = _add(_mul(z * w % N, G), _mul(r * w % N, q))
    return pt is not None and pt[0] % N == r


# --- WebAuthn's authenticator data
def auth_data(ad: bytes) -> dict:
    if len(ad) < 37:
        raise SignInError(400, "the passkey's answer is too short")
    out = {"rp": ad[:32], "flags": ad[32], "count": int.from_bytes(ad[33:37], "big")}
    if ad[32] & 0x40:                                     # attested credential data: its id and public key
        cid_len = int.from_bytes(ad[53:55], "big")
        out["cred_id"] = ad[55:55 + cid_len]
        key, _ = cbor(ad, 55 + cid_len)
        out["cose"] = key
    return out


class SignIn:
    def __init__(self, server):
        self.server = server
        self.lock = threading.Lock()
        self.sessions: dict[str, float] = {}             # sha256(token) -> expiry
        self.challenges: dict[str, tuple[str, float]] = {}   # challenge -> (what for, expiry)
        self.fails: dict[str, list] = {}                 # address -> [count, locked until]

    @property
    def cfg(self):
        return self.server.cfg

    def section(self) -> dict:
        s = self.cfg.raw.get("signin")
        return s if isinstance(s, dict) else {}

    def _save(self, s: dict) -> None:
        self.cfg.raw["signin"] = s
        if self.cfg.path:
            self.cfg.save()

    def configured(self) -> bool:
        s = self.section()
        return bool(s.get("pin") or s.get("passkeys"))

    def _log(self, kind: str, who: str, what: str, **detail) -> None:
        log = getattr(self.server, "seclog", None)
        if log is not None:
            log.add(kind, who, what, **detail)

    # --- sessions
    def cookie_name(self) -> str:
        return f"nebula_{self.server.server_port}"

    def token_of(self, headers) -> str | None:
        raw = headers.get("Cookie") or ""
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == self.cookie_name():
                return v
        return None

    def signed_in(self, headers) -> bool:
        t = self.token_of(headers)
        if not t:
            return False
        h = hashlib.sha256(t.encode()).hexdigest()
        with self.lock:
            exp = self.sessions.get(h)
            if exp is None or exp < time.time():
                self.sessions.pop(h, None)
                return False
        return True

    def _session(self) -> tuple[str, str]:
        t = secrets.token_urlsafe(32)
        with self.lock:
            now = time.time()
            self.sessions = {k: v for k, v in self.sessions.items() if v > now}
            self.sessions[hashlib.sha256(t.encode()).hexdigest()] = now + SESSION_S
        return t, f"{self.cookie_name()}={t}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_S}"

    def sign_out(self, headers) -> str:
        t = self.token_of(headers)
        if t:
            with self.lock:
                self.sessions.pop(hashlib.sha256(t.encode()).hexdigest(), None)
        return f"{self.cookie_name()}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"

    def may_change(self, headers, addr: str) -> bool:
        """Setting a PIN or adding a passkey: the first one from this machine; later ones only when signed in."""
        from nebula import lan
        if not self.configured():
            return lan.is_loopback(addr)
        return self.signed_in(headers)

    # --- the PIN
    def _throttle(self, addr: str) -> None:
        with self.lock:
            c, until = self.fails.get(addr, [0, 0.0])
        if until > time.time():
            raise SignInError(429, f"too many wrong tries: wait {int(until - time.time()) + 1} s")

    def _failed(self, addr: str, how: str) -> None:
        with self.lock:
            c, _ = self.fails.get(addr, [0, 0.0])
            c += 1
            wait = 0 if c < FREE_TRIES else min(LOCK_MAX_S, 2 ** (c - FREE_TRIES) * 15)
            self.fails[addr] = [c, time.time() + wait]
        self._log("signin.refused", addr, how, tries=c, wait_s=wait)

    @staticmethod
    def _hash(pin: str, salt: bytes) -> str:
        return hashlib.scrypt(pin.encode(), salt=salt, n=SCRYPT["n"], r=SCRYPT["r"], p=SCRYPT["p"], dklen=32).hex()

    def set_pin(self, pin: str, headers, addr: str) -> dict:
        if not self.may_change(headers, addr):
            raise SignInError(401, "sign in first")
        pin = str(pin or "")
        if not (pin.isdigit() and 4 <= len(pin) <= 12):
            raise SignInError(400, "a PIN is 4 to 12 digits")
        if len(set(pin)) == 1 or pin in "0123456789012" or pin in "9876543210987":
            raise SignInError(400, "pick a PIN that is not one digit repeated or a run like 1234")
        salt = os.urandom(16)
        s = dict(self.section(), pin={"salt": salt.hex(), "scrypt": self._hash(pin, salt), "set_at": time.time()})
        first = not self.configured()
        self._save(s)
        self._log("signin.changed", addr, "pin")
        out = {"pin": True}
        if first:                                         # the first way in: this browser is signed in with it
            out["_cookie"] = self._session()[1]
        return out

    def sign_in_pin(self, pin: str, addr: str) -> str:
        self._throttle(addr)
        p = self.section().get("pin")
        if not p:
            raise SignInError(400, "no PIN is set here: use a passkey")
        if not hmac.compare_digest(self._hash(str(pin or ""), bytes.fromhex(p["salt"])), p["scrypt"]):
            self._failed(addr, "pin")
            raise SignInError(401, "that PIN is not right")
        with self.lock:
            self.fails.pop(addr, None)
        self._log("signin.ok", addr, "pin")
        return self._session()[1]

    # --- passkeys
    def _challenge(self, why: str) -> str:
        c = b64u(os.urandom(32))
        with self.lock:
            now = time.time()
            self.challenges = {k: v for k, v in self.challenges.items() if v[1] > now}
            self.challenges[c] = (why, now + CHALLENGE_S)
        return c

    def _take_challenge(self, c: str, why: str) -> None:
        with self.lock:
            got = self.challenges.pop(c, None)
        if got is None or got[0] != why or got[1] < time.time():
            raise SignInError(400, "that passkey request expired: try again")

    @staticmethod
    def rp_of(host_header: str) -> str | None:
        """The name a passkey is made for: the page's host name; None for an address (passkeys need a name)."""
        from nebula.origin import split_host, _is_address
        h = split_host(host_header or "")
        return None if not h or _is_address(h) else h

    def create_options(self, headers, addr: str) -> dict:
        if not self.may_change(headers, addr):
            raise SignInError(401, "sign in first")
        rp = self.rp_of(headers.get("Host"))
        if rp is None:
            raise SignInError(400, "passkeys need this page opened by a name: use http://localhost:"
                                   f"{self.server.server_port}/")
        uid = self.section().get("user_id") or b64u(os.urandom(16))
        if not self.section().get("user_id"):
            self._save(dict(self.section(), user_id=uid))
        return {"challenge": self._challenge("create"), "rp": {"id": rp, "name": "Nebula"},
                "user": {"id": uid, "name": "owner", "displayName": "This Nebula's owner"},
                "pubKeyCredParams": [{"type": "public-key", "alg": -7}], "timeout": CHALLENGE_S * 1000,
                "authenticatorSelection": {"userVerification": "required", "residentKey": "preferred"},
                "excludeCredentials": [{"type": "public-key", "id": k["id"]} for k in self.section().get("passkeys") or []],
                "attestation": "none"}

    def _client_data(self, raw_b64: str, why: str, headers) -> bytes:
        raw = unb64u(raw_b64)
        try:
            cd = json.loads(raw)
        except ValueError:
            raise SignInError(400, "the passkey's answer is not readable") from None
        if cd.get("type") != why:
            raise SignInError(400, "the passkey answered something else")
        self._take_challenge(cd.get("challenge") or "", "create" if why == "webauthn.create" else "get")
        want = f"http://{headers.get('Host')}"
        o = urllib.parse.urlsplit(str(cd.get("origin") or ""))
        if f"{o.scheme}://{o.netloc}" not in (want, f"https://{headers.get('Host')}"):
            raise SignInError(400, "the passkey answered for another page")
        return raw

    def create(self, req: dict, headers, addr: str) -> dict:
        if not self.may_change(headers, addr):
            raise SignInError(401, "sign in first")
        rp = self.rp_of(headers.get("Host"))
        if rp is None:
            raise SignInError(400, "passkeys need this page opened by a name")
        self._client_data(req.get("clientDataJSON"), "webauthn.create", headers)
        att, _ = cbor(unb64u(req.get("attestationObject")))
        ad = auth_data(att.get("authData") or b"")
        if ad["rp"] != hashlib.sha256(rp.encode()).digest():
            raise SignInError(400, "the passkey was made for another name")
        if not (ad["flags"] & 0x01 and ad["flags"] & 0x04):
            raise SignInError(400, "the passkey did not verify you (a fingerprint, a face or the device's PIN)")
        k = ad.get("cose") or {}
        if k.get(1) != 2 or k.get(3) != -7 or k.get(-1) != 1:
            raise SignInError(400, "this passkey's kind is not supported here (P-256 only)")
        x, y = int.from_bytes(k[-2], "big"), int.from_bytes(k[-3], "big")
        if not _on_curve((x, y)):
            raise SignInError(400, "the passkey's key is not valid")
        row = {"id": b64u(ad["cred_id"]), "x": b64u(k[-2]), "y": b64u(k[-3]), "rp": rp, "count": ad["count"],
               "name": str(req.get("name") or "a passkey")[:40], "added": time.time()}
        first = not self.configured()
        s = self.section()
        self._save(dict(s, passkeys=[p for p in s.get("passkeys") or [] if p["id"] != row["id"]] + [row]))
        self._log("signin.changed", addr, "passkey", name=row["name"])
        out = {"passkey": {k2: row[k2] for k2 in ("id", "name", "added")}}
        if first:
            out["_cookie"] = self._session()[1]
        return out

    def get_options(self, headers) -> dict:
        rp = self.rp_of(headers.get("Host"))
        keys = [k for k in self.section().get("passkeys") or [] if k.get("rp") == rp]
        if rp is None or not keys:
            raise SignInError(400, "no passkey for this address: use the PIN" +
                              (f", or open http://localhost:{self.server.server_port}/" if rp is None else ""))
        return {"challenge": self._challenge("get"), "rpId": rp, "timeout": CHALLENGE_S * 1000, "userVerification": "required",
                "allowCredentials": [{"type": "public-key", "id": k["id"]} for k in keys]}

    def sign_in_passkey(self, req: dict, headers, addr: str) -> str:
        self._throttle(addr)
        rp = self.rp_of(headers.get("Host"))
        key = next((k for k in self.section().get("passkeys") or [] if k["id"] == req.get("id") and k.get("rp") == rp), None)
        if key is None:
            self._failed(addr, "passkey")
            raise SignInError(401, "that passkey is not one of this Nebula's")
        raw_cd = self._client_data(req.get("clientDataJSON"), "webauthn.get", headers)
        ad_raw = unb64u(req.get("authenticatorData"))
        ad = auth_data(ad_raw)
        if ad["rp"] != hashlib.sha256(rp.encode()).digest() or not (ad["flags"] & 0x01 and ad["flags"] & 0x04):
            self._failed(addr, "passkey")
            raise SignInError(401, "the passkey did not verify you")
        msg = ad_raw + hashlib.sha256(raw_cd).digest()
        if not es256_verify(int.from_bytes(unb64u(key["x"]), "big"), int.from_bytes(unb64u(key["y"]), "big"), msg, unb64u(req.get("signature"))):
            self._failed(addr, "passkey")
            raise SignInError(401, "the passkey's signature is not right")
        if key.get("count") and ad["count"] and ad["count"] <= key["count"]:
            self._failed(addr, "passkey")
            raise SignInError(401, "this passkey's counter went backwards: it may have been copied")
        s = self.section()
        self._save(dict(s, passkeys=[dict(k, count=ad["count"], used=time.time()) if k["id"] == key["id"] else k for k in s["passkeys"]]))
        with self.lock:
            self.fails.pop(addr, None)
        self._log("signin.ok", addr, "passkey", name=key.get("name"))
        return self._session()[1]

    def remove_passkey(self, kid: str, headers) -> dict:
        if not self.signed_in(headers):
            raise SignInError(401, "sign in first")
        s = self.section()
        left = [k for k in s.get("passkeys") or [] if k["id"] != kid]
        if not left and not s.get("pin"):
            raise SignInError(400, "that is your only way in: set a PIN first")
        self._save(dict(s, passkeys=left))
        return {"removed": kid}

    def view(self, headers) -> dict:
        s = self.section()
        rp = self.rp_of(headers.get("Host"))
        return {"configured": self.configured(), "signed_in": self.signed_in(headers), "pin": bool(s.get("pin")),
                "passkeys": [{"id": k["id"], "name": k.get("name"), "added": k.get("added"), "used": k.get("used"), "here": k.get("rp") == rp}
                             for k in s.get("passkeys") or []],
                "passkey_here": rp is not None, "localhost": f"http://localhost:{self.server.server_port}/"}


# --- the routes (nebula/server.py sends /api/signin* here before the page API's gate)
def handle(server, method: str, path: str, req: dict, headers, addr: str):
    """(status, body, set-cookie or None)."""
    si = server.signin
    try:
        if method == "GET" and path == "/api/signin":
            return 200, si.view(headers), None
        if method == "GET" and path == "/api/signin/passkey/create":
            return 200, si.create_options(headers, addr), None
        if method == "GET" and path == "/api/signin/passkey":
            return 200, si.get_options(headers), None
        if method == "POST" and path == "/api/signin/pin":
            return 200, {"signed_in": True}, si.sign_in_pin(req.get("pin"), addr)
        if method == "POST" and path == "/api/signin/pin/set":
            out = si.set_pin(req.get("pin"), headers, addr)
            return 200, {k: v for k, v in out.items() if k != "_cookie"}, out.get("_cookie")
        if method == "POST" and path == "/api/signin/passkey/create":
            out = si.create(req, headers, addr)
            return 200, {k: v for k, v in out.items() if k != "_cookie"}, out.get("_cookie")
        if method == "POST" and path == "/api/signin/passkey":
            return 200, {"signed_in": True}, si.sign_in_passkey(req, headers, addr)
        if method == "POST" and path.startswith("/api/signin/passkeys/") and path.endswith("/remove"):
            return 200, si.remove_passkey(path[len("/api/signin/passkeys/"):-len("/remove")], headers), None
        if method == "POST" and path == "/api/signin/out":
            return 200, {"signed_in": False}, si.sign_out(headers)
    except SignInError as e:
        return e.status, {"error": {"message": str(e), "code": "signin"}}, None
    return 404, {"error": {"message": f"no such path: {path}"}}, None


def write_admin_token(cfg, token: str) -> str | None:
    """The internal token in a file only its owner reads, for Nebula's own commands on this machine."""
    if not getattr(cfg, "path", None):
        return None
    from nebula import names
    d = names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys")
    os.makedirs(d, mode=0o700, exist_ok=True)
    p = os.path.join(d, "admin.token")
    fd = os.open(p + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.replace(p + ".tmp", p)
    return p


def read_admin_token(cfg) -> str | None:
    from nebula import names
    if not getattr(cfg, "base_dir", None):
        return None
    try:
        return open(os.path.join(names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys"), "admin.token")).read().strip() or None
    except OSError:
        return None
