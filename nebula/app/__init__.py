"""cockpit.app: the cockpit as a desktop application, a native window on the cockpit's own page.

    python3 -m nebula.app --config cockpit.json            # the window; the server (and the engine, when configured)
    python3 -m nebula.app --config cockpit.json --install-shortcut
    python3 -m nebula.app --config cockpit.json --no-window # the same lifetimes without a window, until Ctrl+C

engine.py starts and stops the engine, runner.py runs the server and the engine behind the window, window.py is the
window (pywebview, else WebKitGTK, else a browser in app mode), shortcut.py makes the icon.
"""
