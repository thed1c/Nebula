"""nebula/app/window.py: the native window that shows the cockpit's page.

    show(url, title, width, height, backend=None)   # blocks until the window is closed
    show_html(html, title, ...)                      # an error page, when the cockpit could not start

Backends, the first that loads:

    pywebview   the chosen one (docs/DESIGN.md, step 9): the system's own web view, WebKitGTK on Linux and
                Edge WebView2 on Windows, from Python, no browser and no second runtime. `pip install pywebview`.
    gtk         Linux without pywebview: the same WebKitGTK through PyGObject, which desktop Linux ships.
    browser     neither: a Chromium-family browser in app mode (its own window, no tabs, its own profile so the call
                returns when the window closes), else the default browser and the app runs until Ctrl+C.

SIGINT and SIGTERM close the window, so the app's own shutdown runs after `show` returns either way.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import webbrowser

APP_ID = "nebula-eh"                    # the window class Linux docks group by (the .desktop file's StartupWMClass)
BG = "#08070f"
ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")


def _has_gtk() -> str | None:
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        for v in ("4.1", "4.0"):
            try:
                gi.require_version("WebKit2", v)
                from gi.repository import WebKit2  # noqa: F401
                return v
            except (ValueError, ImportError):
                continue
    except (ImportError, ValueError):
        pass
    return None


def _has_pywebview() -> bool:
    try:
        import webview  # noqa: F401
        return True
    except ImportError:
        return False


def backends() -> list[str]:
    """The backends that can run here, best first."""
    out = []
    if _has_pywebview():
        out.append("pywebview")
    if sys.platform.startswith("linux") and _has_gtk():
        out.append("gtk")
    out.append("browser")
    return out


def _glib_signals(quit_) -> None:
    """GTK's main loop keeps Python's signal handlers from running: GLib's own catch SIGINT and SIGTERM instead."""
    try:
        from gi.repository import GLib
        GLib.set_prgname(APP_ID)
        for sig in (signal.SIGINT, signal.SIGTERM):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, lambda *_a: (quit_(), False)[1])
    except Exception:  # noqa: BLE001 - no GLib (Windows): Python's handlers work there
        pass


def _pywebview(url, html, title, width, height) -> None:
    import webview
    win = webview.create_window(title, url=url, html=html, width=width, height=height, min_size=(780, 560),
                                background_color=BG, text_select=True)
    quit_ = lambda: win.destroy()  # noqa: E731
    if sys.platform.startswith("linux"):
        _glib_signals(quit_)
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_a: threading.Thread(target=quit_, daemon=True).start())
            except (ValueError, OSError):
                pass
    gui = "gtk" if sys.platform.startswith("linux") and _has_gtk() else None   # not Qt first on KDE (it may be absent)
    webview.start(gui=gui, private_mode=False, storage_path=os.path.join(tempfile.gettempdir(), APP_ID), icon=ICON)


def _gtk(url, html, title, width, height, version) -> None:
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", version)
    from gi.repository import Gdk, Gtk, WebKit2
    win = Gtk.Window(title=title)
    win.set_default_size(width, height)
    win.set_wmclass(APP_ID, "Nebula EH")
    try:
        win.set_icon_from_file(ICON)
    except Exception:  # noqa: BLE001 - no SVG loader: the theme's default icon
        pass
    rgba = Gdk.RGBA()
    rgba.parse(BG)
    view = WebKit2.WebView()
    view.set_background_color(rgba)
    if html is not None:
        view.load_html(html, "about:blank")
    else:
        view.load_uri(url)
    win.add(view)
    win.connect("destroy", lambda *_a: Gtk.main_quit())
    _glib_signals(lambda: win.destroy())
    win.show_all()
    Gtk.main()


BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "microsoft-edge", "brave-browser")


def _browser(url, html, title, width, height) -> None:
    if html is not None:
        fd, path = tempfile.mkstemp(suffix=".html", prefix=APP_ID + "-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        url = "file://" + path
    exe = next((shutil.which(b) for b in BROWSERS if shutil.which(b)), None)
    if exe is None and os.name == "nt":
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
            cand = os.path.join(base or "", "Microsoft", "Edge", "Application", "msedge.exe")
            if base and os.path.exists(cand):
                exe = cand
                break
    if exe is not None:
        profile = os.path.join(tempfile.gettempdir(), APP_ID + "-profile")
        p = subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", f"--window-size={width},{height}",
                              "--no-first-run", "--no-default-browser-check"])
        try:
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
        return
    webbrowser.open(url)
    print(f"[cockpit] opened {url} in the default browser; Ctrl+C here ends the app", flush=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_a: stop.set())
        except (ValueError, OSError):
            pass
    while not stop.wait(1.0):
        pass


def show(url: str | None, title: str = "Cockpit", width: int = 1440, height: int = 920, backend: str | None = None,
         html: str | None = None) -> str:
    """Opens the window and returns, with the backend's name, when it is closed."""
    order = backends()
    if backend is not None:
        if backend not in order:
            raise RuntimeError(f"the {backend} window is not available here (have: {', '.join(order)})")
        order = [backend]
    name = order[0]
    if name == "pywebview":
        _pywebview(url, html, title, width, height)
    elif name == "gtk":
        _gtk(url, html, title, width, height, _has_gtk())
    else:
        _browser(url, html, title, width, height)
    return name


def error_page(title: str, message: str, detail: str = "") -> str:
    from html import escape
    return (f"<!doctype html><meta charset=utf-8><title>{escape(title)}</title>"
            f"<body style='margin:0;background:{BG};color:#ece9f8;font:15px/1.5 system-ui,sans-serif;padding:40px'>"
            f"<h1 style='font:800 26px/1 Impact,\"Arial Narrow\",sans-serif;letter-spacing:.12em;text-transform:uppercase;"
            f"color:#c6b2ff'>{escape(title)}</h1><p>{escape(message)}</p>"
            f"<pre style='white-space:pre-wrap;color:#9c96b8;background:#100e1b;border:1px solid #28223f;"
            f"border-radius:8px;padding:12px'>{escape(detail)}</pre></body>")
