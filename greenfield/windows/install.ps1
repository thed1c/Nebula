# Nebula EH for Windows, the part of the install that comes from the internet (design decision 2026-10-21: "Not Offline. It
# needs to pip install"). Setup.exe copies the small payload and runs this; its lines show in Setup's window.
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Dest <install folder> -Conf <settings folder>
# 1. Python (python-build-standalone, pinned, sha256 checked; kept, with its packages, when its version did not change),
# 2. Nebula and its packages with pip at the versions tested (constraints.txt; pip fetches only what moved), 3. python -m
# nebula.prereqs: CUDA's DLLs beside Impulse (with an NVIDIA driver), Halo's model, Lookup's and docling's models. Each failure says what to do; nothing needs an administrator.
param([Parameter(Mandatory = $true)][string]$Dest, [Parameter(Mandatory = $true)][string]$Conf)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"                  # Invoke-WebRequest is many times slower with its progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
function Say($m) { Write-Output $m }
$online = Get-Content -Raw (Join-Path $Dest "online.json") | ConvertFrom-Json
$pyDir = Join-Path $Dest "python"
$py = Join-Path $pyDir "python.exe"
$wheel = Join-Path $Dest ("wheels\" + $online.wheel)
$pipLog = Join-Path $Conf "pip.log"
New-Item -ItemType Directory -Force $Conf | Out-Null
# the Python this installer pins, from its download's name (cpython-3.12.12+20251014-x86_64-pc-windows-msvc-...)
$wantPy = if ("$($online.python_url)" -match 'cpython-(\d+\.\d+\.\d+)\+') { $Matches[1] } else { "" }

function PythonHere {                                    # the version of the Python in place, "" when none answers
  if (-not (Test-Path $py)) { return "" }
  try { $v = & $py -c "import platform; print(platform.python_version())" 2>$null; if ($LASTEXITCODE -eq 0) { return "$v".Trim() } }
  catch { }
  return ""
}

function RemovePython {
  if (-not (Test-Path $pyDir)) { return }
  try { Remove-Item -LiteralPath $pyDir -Recurse -Force -ErrorAction Stop }
  catch { Say "The old Python could not be removed: some of its files are in use (Nebula EH still running?). Close it, or restart Windows, then run Setup again."; exit 3 }
}

function FetchPython {
  Say "Fetching Python..."
  $tgz = Join-Path $Dest "python.tar.gz"
  try { Invoke-WebRequest -UseBasicParsing -Uri $online.python_url -OutFile $tgz }
  catch { Say "Could not reach the internet to fetch Python: check the connection and run Setup again."; exit 1 }
  if ((Get-FileHash $tgz -Algorithm SHA256).Hash.ToLower() -ne $online.python_sha256) {
    Remove-Item $tgz -Force; Say "Python's download did not check out (sha256): run Setup again."; exit 1
  }
  tar -xzf $tgz -C $Dest
  $untar = $LASTEXITCODE
  Remove-Item $tgz -Force
  if ($untar -ne 0) {                                    # files of a Python still running could not be replaced
    Say "Python could not be unpacked: some of its files are in use (Nebula EH still running?). Close it, or restart Windows, then run Setup again."
    exit 3
  }
}

function InstallPackages {                               # pip's exit code; its lines go to pip.log
  $pipArgs = @("-m", "pip", "install", "-q", "--no-cache-dir", "--disable-pip-version-check", "--no-warn-script-location",
               "--constraint", (Join-Path $Dest "constraints.txt"))
  if ($online.extra_index) { $pipArgs += @("--extra-index-url", $online.extra_index) }
  $pipArgs += ("{0}[{1}]" -f $wheel, $online.extras)
  # Windows PowerShell 5.1 turns a program's stderr lines (pip's, Hugging Face's warnings) into errors, and "Stop" would
  # end this script on the first one: a program's own exit code says whether it worked
  $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
  & $py @pipArgs *> $pipLog
  $code = $LASTEXITCODE
  if ($code -eq 0 -and $script:kept) {
    # the same version of Nebula already here (Setup run again to repair it): its own files put back, nothing fetched
    & $py -m pip install -q --no-deps --force-reinstall --disable-pip-version-check --no-warn-script-location $wheel *>> $pipLog
    $code = $LASTEXITCODE
  }
  $ErrorActionPreference = $old
  return $code
}

function InUse { Select-String -Path $pipLog -Pattern "WinError 5\]|WinError 32\]|being used by another process" -Quiet }

# Python, and the packages in it, stay when Python's version did not change (the owner's word, 2026-11-02: an update
# is small); pip then fetches only what moved. A Python of another version, or one that does not answer, is replaced.
$script:kept = $false
$have = PythonHere
if ($have -and $wantPy -and $have -eq $wantPy) {
  Say "Python $have is already here: kept, with Nebula's packages (only the ones that changed are fetched)."
  $script:kept = $true
} else {
  if (Test-Path $pyDir) { Say $(if ($have) { "Python $have is replaced by $wantPy..." } else { "The Python here does not answer: it is replaced..." }) }
  RemovePython
  FetchPython
}

Say "Installing Nebula's packages (a few minutes)..."
$code = InstallPackages
if ($code -ne 0 -and -not (InUse) -and $script:kept) {
  # the Python kept may be the reason (a package broken by hand, a half-finished earlier install): once more from scratch
  Say "Nebula's packages did not install over the Python already here: starting again with a fresh Python..."
  $script:kept = $false
  RemovePython
  FetchPython
  $code = InstallPackages
}
if ($code -ne 0) {
  # a file in use (Nebula still running) is not the connection: say which it is
  if (InUse) {
    Say ("Installing Nebula's packages failed: some of its files are in use (Nebula EH still running?). Close it, or restart Windows, then run Setup again (see {0})." -f $pipLog); exit 3
  }
  Say ("Installing Nebula's packages failed (see {0}): check the connection and run Setup again." -f $pipLog); exit 1
}
$ErrorActionPreference = "Continue"                      # as before: from here, a program's own exit code says whether it worked

# what this installer put in place: Nebula's updater offers a small update (code alone) to an install made by its base
$info = @{ base = $online.version; version = $online.version; extras = $online.extras; extra_index = "$($online.extra_index)" } | ConvertTo-Json -Compress
[IO.File]::WriteAllText((Join-Path $Dest "app\install.json"), $info, (New-Object Text.UTF8Encoding $false))

Say "Fetching what Nebula needs beside itself (Halo's model, Lookup's models, CUDA's libraries)..."
$env:PYTHONIOENCODING = "utf-8"
& $py -m nebula.prereqs --config (Join-Path $Conf "nebula.json") --dest $Dest 2>&1 | ForEach-Object { "  " + ("$_" -replace '^\[nebula\] ', '') } | Where-Object { $_ -notmatch "warn|Warning|NativeCommandError|At line|CategoryInfo|FullyQualifiedErrorId|^\s*\+|^\s*$|^\s*[WIE]\d{4} " }
Say "Nebula EH is installed."
exit 0
