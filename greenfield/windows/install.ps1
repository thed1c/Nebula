# Nebula EH for Windows, the part of the install that comes from the internet (design decision 2026-10-21: "Not Offline. It
# needs to pip install"). Setup.exe copies the small payload and runs this; its lines show in Setup's window.
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Dest <install folder> -Conf <settings folder>
# 1. Python (python-build-standalone, pinned, sha256 checked), 2. Nebula and its packages with pip at the versions
# tested (constraints.txt), 3. python -m nebula.prereqs: CUDA's DLLs beside Impulse and llama-server (with an NVIDIA
# driver), Halo's model, Lookup's and docling's models. Each failure says what to do; nothing needs an administrator.
param([Parameter(Mandatory = $true)][string]$Dest, [Parameter(Mandatory = $true)][string]$Conf)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"                  # Invoke-WebRequest is many times slower with its progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
function Say($m) { Write-Output $m }
$online = Get-Content -Raw (Join-Path $Dest "online.json") | ConvertFrom-Json
$py = Join-Path $Dest "python\python.exe"

if (-not (Test-Path $py)) {
  Say "Fetching Python..."
  $tgz = Join-Path $Dest "python.tar.gz"
  try { Invoke-WebRequest -UseBasicParsing -Uri $online.python_url -OutFile $tgz }
  catch { Say "Could not reach the internet to fetch Python: check the connection and run Setup again."; exit 1 }
  if ((Get-FileHash $tgz -Algorithm SHA256).Hash.ToLower() -ne $online.python_sha256) {
    Remove-Item $tgz -Force; Say "Python's download did not check out (sha256): run Setup again."; exit 1
  }
  tar -xzf $tgz -C $Dest
  Remove-Item $tgz -Force
}

Say "Installing Nebula's packages (a few minutes)..."
$wheel = Join-Path $Dest ("wheels\" + $online.wheel)
$pipArgs = @("-m", "pip", "install", "-q", "--no-cache-dir", "--disable-pip-version-check", "--no-warn-script-location",
             "--constraint", (Join-Path $Dest "constraints.txt"))
if ($online.extra_index) { $pipArgs += @("--extra-index-url", $online.extra_index) }
$pipArgs += ("{0}[{1}]" -f $wheel, $online.extras)
New-Item -ItemType Directory -Force $Conf | Out-Null
# Windows PowerShell 5.1 turns a program's stderr lines (pip's, Hugging Face's warnings) into errors, and "Stop" would end
# this script on the first one: from here, a program's own exit code says whether it worked
$ErrorActionPreference = "Continue"
& $py @pipArgs *> (Join-Path $Conf "pip.log")
if ($LASTEXITCODE -ne 0) { Say ("Installing Nebula's packages failed (see {0}): check the connection and run Setup again." -f (Join-Path $Conf "pip.log")); exit 1 }

Say "Fetching what Nebula needs beside itself (Halo's model, Lookup's models, CUDA's libraries)..."
$env:PYTHONIOENCODING = "utf-8"
& $py -m nebula.prereqs --config (Join-Path $Conf "nebula.json") --dest $Dest 2>&1 | ForEach-Object { "  " + ("$_" -replace '^\[nebula\] ', '') } | Where-Object { $_ -notmatch "warn|Warning|NativeCommandError|At line|CategoryInfo|FullyQualifiedErrorId|^\s*\+|^\s*$|^\s*[WIE]\d{4} " }
Say "Nebula EH is installed."
exit 0
