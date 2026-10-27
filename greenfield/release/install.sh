#!/bin/bash
# Nebula EH: install from this folder (the release bundle). Linux x86-64.
#   bash install.sh            install or update, for this user only (no sudo)
# Nebula goes into its own Python environment (~/.local/share/nebula/venv), Impulse beside it
# (~/.local/share/nebula/impulse), your settings stay in ~/.config/nebula, and an icon "Nebula EH" is added.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${XDG_DATA_HOME:-$HOME/.local/share}/nebula
CONF=${XDG_CONFIG_HOME:-$HOME/.config}/nebula
say() { printf '\n== %s\n' "$*"; }
fail() { printf '\nNebula was not installed: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail "this bundle is for Linux on x86-64"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY=$(command -v "$c"); break; fi
done
[ -n "$PY" ] || fail "Python 3.10 or newer with venv is needed (Ubuntu/Debian: sudo apt install python3-venv)"
"$PY" -m venv --help >/dev/null 2>&1 || fail "Python's venv is missing (Ubuntu/Debian: sudo apt install python3-venv)"

say "Nebula into $DATA/venv"
mkdir -p "$DATA" "$CONF"
[ -x "$DATA/venv/bin/python" ] || "$PY" -m venv "$DATA/venv" || fail "could not make $DATA/venv (Ubuntu/Debian: sudo apt install python3-venv)"
WHEEL=$(ls "$HERE"/wheels/nebula_eh-*.whl | head -1)
"$DATA/venv/bin/python" -m pip install -q --upgrade pip
"$DATA/venv/bin/python" -m pip install -q --upgrade "$WHEEL[gravity,monitor,lookup]" || fail "pip could not install Nebula (is this computer online?)"
if "$DATA/venv/bin/python" -m pip install -q "$WHEEL[app]" 2>/dev/null && "$DATA/venv/bin/python" -c 'import webview' 2>/dev/null; then APP=window; else APP=browser; fi

say "Impulse into $DATA/impulse"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  rm -rf "$DATA/impulse.new" && cp -r "$HERE/impulse" "$DATA/impulse.new" && rm -rf "$DATA/impulse" && mv "$DATA/impulse.new" "$DATA/impulse"
  "$DATA/impulse/impulse-server" --help >/dev/null 2>&1 && echo "Impulse runs here" || echo "Impulse is installed but did not start here (an RTX 30, 40 or 50 card and a recent NVIDIA driver are needed); Nebula can use llama.cpp instead"
else
  echo "No NVIDIA card found: Impulse is skipped (Nebula can still use models elsewhere, or join a friend's)"
fi

if [ ! -f "$CONF/nebula.json" ]; then
  say "Your settings: $CONF/nebula.json"
  printf '{\n "port": 8090,\n "ledger": "%s",\n "tiers": []\n}\n' "$CONF/nebula.sqlite3" > "$CONF/nebula.json"
fi

say "The icon"
LAUNCH="$DATA/nebula-open"
cat > "$LAUNCH" <<LAUNCHER
#!/bin/bash
# Opens Nebula: its window when it can show one, else this computer's browser (starting Nebula first if needed).
CONF="$CONF/nebula.json"
if [ "$APP" = window ]; then exec "$DATA/venv/bin/python" -m nebula.app --config "\$CONF"; fi
PORT=\$("$DATA/venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("port", 8090))' "\$CONF")
if ! curl -s -o /dev/null "http://127.0.0.1:\$PORT/api/signin"; then
  nohup "$DATA/venv/bin/python" -m nebula --config "\$CONF" > "$DATA/nebula.log" 2>&1 &
  for i in \$(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:\$PORT/api/signin" && break; sleep 0.5; done
fi
xdg-open "http://localhost:\$PORT/" >/dev/null 2>&1 &
LAUNCHER
chmod +x "$LAUNCH"
APPS=${XDG_DATA_HOME:-$HOME/.local/share}/applications
mkdir -p "$APPS"
cat > "$APPS/nebula-eh.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Nebula EH
Comment=Your own AI on this computer, shared with friends
Exec=$LAUNCH
Icon=$DATA/venv/lib/python3/site-packages/nebula/app/nebula.svg
Terminal=false
Categories=Utility;Development;
StartupWMClass=nebula-eh
DESKTOP
ICON=$("$DATA/venv/bin/python" -c 'import nebula.app.window as w; print(w.ICON)' 2>/dev/null || true)
[ -n "$ICON" ] && sed -i "s|^Icon=.*|Icon=$ICON|" "$APPS/nebula-eh.desktop"
update-desktop-database "$APPS" >/dev/null 2>&1 || true

say "Done"
echo "Open \"Nebula EH\" from your applications, or run: $LAUNCH"
echo "The first time, it asks you to protect it with a PIN or a passkey, then helps you pick a model."
