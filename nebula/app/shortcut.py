"""nebula/app/shortcut.py: an icon that starts the application.

Linux: a desktop entry in ~/.local/share/applications (the launcher and the dock find it there), with the Python that
ran this command, so a virtual environment that has pywebview is the one the icon uses. Windows: a shortcut on the
desktop to pythonw.exe (no console window), written by PowerShell's WScript.Shell; not run on Windows yet.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from nebula.app.window import APP_ID, ICON

ROOT = Path(__file__).resolve().parents[2]       # the repository: `python -m nebula.app` runs from here


def desktop_entry(python: str, config: str | None, root: Path = ROOT, icon: str = ICON) -> str:
    args = [python, "-m", "nebula.app"] + (["--config", str(Path(config).resolve())] if config else [])
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        "Name=Nebula EH",
        "Comment=Your own AI on this computer, shared with friends",
        "Exec=" + " ".join(shlex.quote(a) for a in args),
        "Path=" + str(root),
        "Icon=" + icon,
        "Terminal=false",
        "Categories=Development;Utility;",
        "StartupWMClass=" + APP_ID,
        "StartupNotify=true",
        ""])


def install_linux(config: str | None, apps_dir: Path | None = None, python: str | None = None) -> Path:
    apps_dir = apps_dir or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "applications"
    apps_dir.mkdir(parents=True, exist_ok=True)
    path = apps_dir / f"{APP_ID}.desktop"
    path.write_text(desktop_entry(python or sys.executable, config), encoding="utf-8")
    path.chmod(0o755)
    try:
        subprocess.run(["update-desktop-database", str(apps_dir)], check=False, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return path


def windows_script(python: str, config: str | None, root: Path = ROOT) -> str:
    """The PowerShell that writes the desktop shortcut (pythonw.exe beside python.exe: no console window)."""
    pyw = str(Path(python).with_name("pythonw.exe"))
    args = "-m nebula.app" + (f' --config "{Path(config).resolve()}"' if config else "")
    q = lambda s: s.replace("'", "''")  # noqa: E731 - PowerShell's single-quoted string
    return ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
            "[Environment]::GetFolderPath('Desktop') + '\\Nebula EH.lnk'); "
            f"$s.TargetPath = '{q(pyw)}'; $s.Arguments = '{q(args)}'; $s.WorkingDirectory = '{q(str(root))}'; "
            "$s.Description = 'Your own AI on this computer, shared with friends'; $s.Save()")


def install(config: str | None) -> str:
    if os.name == "nt":
        subprocess.run(["powershell", "-NoProfile", "-Command", windows_script(sys.executable, config)], check=True)
        return "a 'Nebula EH' shortcut on the desktop"
    return str(install_linux(config))
