// Nebula's desktop window: Electron (Chromium, drawn on the GPU) around the cockpit's own page.
//
//   npx electron . --config ~/.config/nebula/nebula.json [--python /path/to/python]
//   npx electron . --config FILE --install-shortcut     an icon that opens this window with this config
//
// The config is the cockpit's (host, port, app.title, app.width, app.height). When a cockpit already answers at its
// address the window attaches to it; otherwise it starts `python -m nebula --config FILE` from the repository and
// stops it when the window closes. The page is the cockpit's page, unchanged: the window only shows it. It stays on
// the cockpit's origin (anything else opens in the default browser), with no Node in the page (sandboxed, isolated).
"use strict";
const { app, BrowserWindow, shell, Menu } = require("electron");
const { spawn, execFile } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..", "..");   // the repository: `python -m nebula` runs from here
const WIN = process.platform === "win32";

function arg(name) {
  const i = process.argv.indexOf(name);
  return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : null;
}
const home = (p) => p && p.replace(/^~(?=$|[\\/])/, os.homedir());

function configPath() {
  const p = home(arg("--config") || process.env.NEBULA_CONFIG || path.join(os.homedir(), ".config", "nebula", "nebula.json"));
  return path.resolve(p);
}

function pythonPath() {
  const given = home(arg("--python") || process.env.NEBULA_PYTHON);
  if (given) return given;
  const venv = WIN ? path.join(process.env.LOCALAPPDATA || os.homedir(), "nebula", "venv", "Scripts", "python.exe")
                   : path.join(os.homedir(), ".local", "share", "nebula", "venv", "bin", "python");
  return fs.existsSync(venv) ? venv : (WIN ? "python" : "python3");
}

async function answers(url) {
  try {
    // /api/signin answers before anyone signs in (the page API itself needs a signed-in browser once a PIN or a
    // passkey is set), and it is Nebula's own: {"configured": …}
    const r = await fetch(url + "api/signin", { signal: AbortSignal.timeout(1500) });
    return r.ok && "configured" in (await r.json());
  } catch (e) {
    return false;
  }
}

let server = null;          // the cockpit this window started, if it started one
const stderr = [];

function startServer(cfgFile) {
  const py = pythonPath();
  server = spawn(py, ["-m", "nebula", "--config", cfgFile], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  const keep = (b) => { for (const line of String(b).split(/\r?\n/)) if (line) { stderr.push(line); if (stderr.length > 40) stderr.shift(); } };
  server.stdout.on("data", keep);
  server.stderr.on("data", keep);
  server.on("error", (e) => keep(`could not start ${py}: ${e.message}`));
  server.on("exit", (code) => { keep(`the cockpit exited (${code})`); server = null; });
}

function stopServer() {
  if (!server) return;
  const s = server; server = null;
  if (WIN) execFile("taskkill", ["/pid", String(s.pid), "/T", "/F"], () => {});
  else s.kill("SIGTERM");
}

async function waitFor(url, seconds) {
  const until = Date.now() + seconds * 1000;
  while (Date.now() < until) {
    if (await answers(url)) return true;
    if (!server) return false;              // it exited
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

function errorPage(title, lines) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  return "data:text/html;charset=utf-8," + encodeURIComponent(
    `<!doctype html><meta charset="utf-8"><title>${esc(title)}</title><body style="background:#0c0a17;color:#e8ecff;` +
    `font:14px system-ui,sans-serif;padding:32px"><h2 style="color:#ffd166">${esc(title)}</h2>` +
    `<pre style="white-space:pre-wrap;color:#b9b4d6">${esc(lines.join("\n"))}</pre></body>`);
}

// The icon: on Linux a desktop entry in ~/.local/share/applications (the launcher and the dock) and a copy on the
// desktop; on Windows a shortcut on the desktop (not run on Windows yet). Both run this Electron on this folder.
function installShortcut() {
  const cfg = configPath(), title = (() => { try { return JSON.parse(fs.readFileSync(cfg, "utf8")).app?.title || "Nebula"; } catch (e) { return "Nebula"; } })();
  const desktop = (() => { try { return require("node:child_process").execFileSync("xdg-user-dir", ["DESKTOP"]).toString().trim(); } catch (e) { return path.join(os.homedir(), "Desktop"); } })();
  const args = [__dirname, "--config", cfg, ...(arg("--python") ? ["--python", home(arg("--python"))] : [])];
  if (WIN) {
    const link = path.join(app.getPath("desktop"), `${title}.lnk`);
    shell.writeShortcutLink(link, { target: process.execPath, args: args.map((a) => `"${a}"`).join(" "), cwd: __dirname,
                                    description: "Nebula, the switchyard above the engine" });
    return [link];
  }
  const q = (a) => /^[\w@%+=:,./-]+$/.test(a) ? a : "'" + a.replace(/'/g, "'\\''") + "'";
  const entry = ["[Desktop Entry]", "Type=Application", "Version=1.0", `Name=${title}`,
                 "Comment=Nebula, the switchyard above the engine", `Exec=${[process.execPath, ...args].map(q).join(" ")}`,
                 `Path=${__dirname}`, `Icon=${path.resolve(__dirname, "..", "app", "icon.svg")}`, "Terminal=false",
                 "Categories=Utility;", "StartupWMClass=nebula-desktop", "StartupNotify=true", ""].join("\n");
  const out = [];
  for (const dir of [path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "applications"), desktop]) {
    if (!fs.existsSync(dir)) continue;
    const file = path.join(dir, "nebula.desktop");
    fs.writeFileSync(file, entry, { mode: 0o755 }); fs.chmodSync(file, 0o755);
    out.push(file);
  }
  return out;
}

async function main() {
  if (process.argv.includes("--install-shortcut")) {
    await app.whenReady();
    for (const f of installShortcut()) console.log("[nebula] the icon: " + f);
    app.quit(); return;
  }
  if (!app.requestSingleInstanceLock()) { app.quit(); return; }
  await app.whenReady();
  Menu.setApplicationMenu(null);

  const cfgFile = configPath();
  let cfg = {};
  try { cfg = JSON.parse(fs.readFileSync(cfgFile, "utf8")); }
  catch (e) { cfg = null; }
  const ac = (cfg && cfg.app) || {};
  const host = !cfg || !cfg.host || cfg.host === "0.0.0.0" ? "127.0.0.1" : cfg.host;
  const url = `http://${host}:${(cfg && cfg.port) || 8090}/`;

  const win = new BrowserWindow({
    title: ac.title || "Nebula", width: ac.width || 1440, height: ac.height || 920, minWidth: 780, minHeight: 560,
    backgroundColor: "#0c0a17", autoHideMenuBar: true, show: false,
    webPreferences: { contextIsolation: true, sandbox: true, nodeIntegration: false },
  });
  win.once("ready-to-show", () => win.show());
  win.on("page-title-updated", (e) => e.preventDefault());       // the window keeps the config's title
  app.on("second-instance", () => { if (win.isMinimized()) win.restore(); win.focus(); });

  // the cockpit's origin only; links elsewhere go to the default browser
  const origin = new URL(url).origin;
  win.webContents.setWindowOpenHandler(({ url: u }) => { if (/^https?:/.test(u)) shell.openExternal(u); return { action: "deny" }; });
  win.webContents.on("will-navigate", (e, u) => { if (new URL(u).origin !== origin) { e.preventDefault(); if (/^https?:/.test(u)) shell.openExternal(u); } });
  win.webContents.on("before-input-event", (e, i) => {
    if (i.type !== "keyDown") return;
    if (i.key === "F5" || ((i.control || i.meta) && i.key.toLowerCase() === "r")) { win.webContents.reload(); e.preventDefault(); }
    if (i.key === "F12" || ((i.control || i.meta) && i.shift && i.key.toLowerCase() === "i")) { win.webContents.toggleDevTools(); e.preventDefault(); }
  });

  if (!cfg) { win.loadURL(errorPage("No config", [`Nebula needs a config file: ${cfgFile}`, "Start it with --config FILE."])); return; }
  if (!(await answers(url))) {
    win.loadURL(errorPage("Starting Nebula…", [`python -m nebula --config ${cfgFile}`]));
    startServer(cfgFile);
    if (!(await waitFor(url, 60))) {
      win.loadURL(errorPage("Nebula did not start", [`It did not answer at ${url}.`, "", ...stderr]));
      return;
    }
  }
  win.loadURL(url);
}

app.on("window-all-closed", () => { stopServer(); app.quit(); });
app.on("before-quit", stopServer);
process.on("SIGINT", () => { stopServer(); app.quit(); });
process.on("SIGTERM", () => { stopServer(); app.quit(); });
main();
