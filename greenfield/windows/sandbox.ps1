# Nebula EH for Windows: the Safe sandbox step. Engines then run in containers (Podman in WSL2), as on Linux; without it
# they run as plain programs, with a warning (nebula/wslbox.py says which). Setup runs it after install.ps1:
#   powershell -NoProfile -ExecutionPolicy Bypass -File sandbox.ps1 -Dest <install folder> -Conf <settings folder>
#       [-Decline] [-Resume] [-ImpulseTar <impulse-linux.tar.gz>]
# 1. WSL2: when it is not on this PC, `wsl --install --no-distribution --web-download` (from GitHub, so no Microsoft
#    Store is needed: Windows 10 LTSC) asks for an administrator once (UAC); if
#    Windows must restart, this step resumes by itself at the next sign-in (RunOnce) and Nebula runs unsandboxed until then.
# 2. Nebula's own distro (NebulaEH): Ubuntu 24.04's WSL root filesystem, pinned by sha256 (nebula/catalog/wsl.json),
#    imported for this user only (no administrator) under %LOCALAPPDATA%\NebulaEH.
# 3. Inside it (sandbox-setup.sh, as the distro's root): pinned Podman and NVIDIA's container toolkit, the card
#    described to Podman, Impulse's Linux build as the engines' image (fetched from the release, sha256 checked).
# Its result goes to <Conf>\sandbox.json. A failure is reported and leaves Nebula working without the sandbox.
# Undo: sandbox-undo.ps1 (the distro, its folder, sandbox.json; WSL itself stays).
param([Parameter(Mandatory = $true)][string]$Dest, [Parameter(Mandatory = $true)][string]$Conf,
      [switch]$Decline, [switch]$Resume, [string]$ImpulseTar = "")
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
function Say($m) { Write-Output $m }
New-Item -ItemType Directory -Force $Conf | Out-Null
$stateFile = Join-Path $Conf "sandbox.json"
function Done($state, $why, $extra) {
  $o = [ordered]@{ state = $state; why = $why; at = (Get-Date).ToUniversalTime().ToString("o") }
  if ($extra) { foreach ($k in $extra.Keys) { $o[$k] = $extra[$k] } }
  # UTF-8 without a BOM (Windows PowerShell 5.1's -Encoding UTF8 writes one, which Python's json refuses)
  [IO.File]::WriteAllText($stateFile, ($o | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
  if ($state -eq "ready") { Say "Safe sandbox: ready." } elseif ($why) { Say ("Safe sandbox: " + $why + ". Nebula runs engines without it.") }
  exit 0
}
if ($Decline) { Done "declined" "you chose to run engines without a sandbox" $null }

$pinsFile = Join-Path $PSScriptRoot "wsl.json"                  # nebula/catalog/wsl.json, beside this script in the payload
$pins = Get-Content -Raw $pinsFile | ConvertFrom-Json
$distro = $pins.distro
$online = Get-Content -Raw (Join-Path $Dest "online.json") -ErrorAction SilentlyContinue | ConvertFrom-Json
$version = if ($online -and $online.version) { $online.version } else { "" }
$home2 = Join-Path $env:LOCALAPPDATA "NebulaEH"
New-Item -ItemType Directory -Force $home2 | Out-Null

function Wsl([string[]]$a) {                    # wsl.exe's own messages are UTF-16: read them as such
  $old = [Console]::OutputEncoding; [Console]::OutputEncoding = [Text.Encoding]::Unicode
  try { $out = & wsl.exe @a 2>&1 | Out-String; return @{ code = $LASTEXITCODE; out = $out } } finally { [Console]::OutputEncoding = $old }
}

# 1. WSL2 itself
$st = Wsl @("--status")
if ($st.code -ne 0) {
  if ($Resume) { Done "failed" "WSL2 is still not working after the restart: $($st.out.Trim())" $null }
  Say "Turning on WSL2 (Windows asks for an administrator once)..."
  try { $p = Start-Process -FilePath "wsl.exe" -ArgumentList "--install", "--no-distribution", "--web-download" -Verb RunAs -Wait -PassThru }
  catch { Done "declined" "WSL2 was not turned on (the administrator prompt was declined)" $null }
  $st = Wsl @("--status")
  if ($st.code -ne 0) {
    $cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Dest `"$Dest`" -Conf `"$Conf`" -Resume"
    New-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce" -Name "NebulaEHSandbox" -Value $cmd -Force | Out-Null
    Done "reboot" "Windows must restart once to finish turning on WSL2; the sandbox finishes by itself at the next sign-in" $null
  }
}

# 2. Nebula's own distro
$have = (Wsl @("--list", "--quiet")).out -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq $distro }
if (-not $have) {
  $tar = Join-Path $home2 "rootfs.tar.gz"
  Say "Fetching Ubuntu 24.04 for the sandbox ($([math]::Round($pins.rootfs.bytes / 1MB)) MB)..."
  try { Invoke-WebRequest -UseBasicParsing -Uri $pins.rootfs.url -OutFile $tar } catch { Done "failed" "Ubuntu's root filesystem could not be fetched" $null }
  if ((Get-FileHash $tar -Algorithm SHA256).Hash.ToLower() -ne $pins.rootfs.sha256) { Remove-Item $tar -Force; Done "failed" "Ubuntu's root filesystem did not check out (sha256)" $null }
  $r = Wsl @("--import", $distro, (Join-Path $home2 "wsl"), $tar, "--version", "2")
  Remove-Item $tar -Force
  if ($r.code -ne 0) { Done "failed" ("the distro could not be imported: " + $r.out.Trim()) $null }
}

# 3. Inside it
$tarIn = ""
if ($ImpulseTar) { $tarIn = $ImpulseTar }
elseif ($online -and $online.sandbox_impulse) {
  $tarIn = Join-Path $home2 "impulse-linux.tar.gz"
  Say "Fetching Impulse's Linux build for the sandbox..."
  try { Invoke-WebRequest -UseBasicParsing -Uri $online.sandbox_impulse.url -OutFile $tarIn } catch { Done "failed" "Impulse's Linux build could not be fetched" $null }
  if ((Get-FileHash $tarIn -Algorithm SHA256).Hash.ToLower() -ne $online.sandbox_impulse.sha256) { Remove-Item $tarIn -Force; Done "failed" "Impulse's Linux build did not check out (sha256)" $null }
}
function ToWsl([string]$p) { if (-not $p) { return "-" }; $f = (Resolve-Path $p).Path; return "/mnt/" + $f.Substring(0, 1).ToLower() + ($f.Substring(2) -replace "\\", "/") }
$script = Join-Path $PSScriptRoot "sandbox-setup.sh"
Say "Setting up Podman and the card in the sandbox (a few minutes)..."
$old = [Console]::OutputEncoding; [Console]::OutputEncoding = [Text.Encoding]::UTF8
$lines = & wsl.exe -d $distro -u root --exec bash (ToWsl $script) (ToWsl $pinsFile) (ToWsl $tarIn) $version 2>&1
[Console]::OutputEncoding = $old
$lines | ForEach-Object { "  $_" } | Where-Object { $_ -notmatch "^\s*(RESULT|FAILED) " }
if ($tarIn -and -not $ImpulseTar) { Remove-Item $tarIn -Force -ErrorAction SilentlyContinue }
$res = $lines | Where-Object { "$_" -match "^RESULT " } | Select-Object -Last 1
if (-not $res) {
  $why = ($lines | Where-Object { "$_" -match "^FAILED " } | Select-Object -Last 1)
  Done "failed" $(if ($why) { "$why".Substring(7) } else { "the setup inside the distro stopped" }) $null
}
$info = ("$res".Substring(7)) | ConvertFrom-Json
& wsl.exe --terminate $distro | Out-Null                        # its default user (nebula) from the next start
$chk = Wsl @("-d", $distro, "--exec", "nebula-podman", "info", "--format", "{{.Version.Version}}")
if ($chk.code -ne 0) { Done "failed" ("Podman does not answer as the engines' user: " + $chk.out.Trim()) $null }
$images = @{}; foreach ($p in $info.images.PSObject.Properties) { $images[$p.Name] = $p.Value }
Done "ready" "" @{ distro = $distro; podman = $info.podman; gpu = $info.gpu; gpu_check = $info.gpu_check; images = $images }
