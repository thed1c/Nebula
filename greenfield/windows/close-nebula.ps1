# Nebula EH for Windows: close a running Nebula before Setup (or the uninstaller) touches its files.
#   powershell -NoProfile -ExecutionPolicy Bypass -File close-nebula.ps1 -Dest <install folder> -Conf <settings folder>
# Nebula's processes are the ones whose program is in the install folder or in Nebula's own settings folder (its
# Python, Impulse, Impulse portable, PAIR), found by their path, never by name: other Pythons stay as they are. Each is
# asked to close, then stopped after 10 s. Then the program folders are checked: a file still in use (a Nebula started
# as administrator, which this user cannot stop) ends this with exit code 3 before anything is deleted.
# Exit codes: 0 nothing of Nebula's runs any more, 3 still in use.
param([Parameter(Mandatory = $true)][string]$Dest, [string]$Conf = "")
$ErrorActionPreference = "Continue"
function Say($m) { Write-Output $m }
$roots = @($Dest, $Conf) | Where-Object { $_ } | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') + '\' }

function Nebulas {
  Get-CimInstance Win32_Process | Where-Object {
    $p = $_.ExecutablePath
    $p -and $_.ProcessId -ne $PID -and ($roots | Where-Object { $p.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })
  }
}

$found = @(Nebulas)
if ($found.Count) {
  Say ("Closing Nebula EH ({0} of its programs are running)..." -f $found.Count)
  foreach ($p in $found) { taskkill /PID $p.ProcessId 2>&1 | Out-Null }          # asked: its window closes, Nebula stops its engines
  for ($i = 0; $i -lt 10 -and @(Nebulas).Count; $i++) { Start-Sleep 1 }
  foreach ($p in @(Nebulas)) { taskkill /F /PID $p.ProcessId 2>&1 | Out-Null }    # what did not close by itself
  Start-Sleep 1
}

# a program folder whose files are in use cannot be renamed: try each (and put it back) for a few seconds
function InUse {
  foreach ($name in "python", "impulse", "halo", "pair") {
    $dir = Join-Path $Dest $name
    if (-not (Test-Path $dir)) { continue }
    $probe = "$dir.setup-check"
    try { Rename-Item -LiteralPath $dir -NewName (Split-Path $probe -Leaf) -ErrorAction Stop }
    catch { return $name }
    for ($j = 0; $j -lt 10 -and (Test-Path $probe); $j++) {                         # back under its own name
      try { Rename-Item -LiteralPath $probe -NewName $name -ErrorAction Stop } catch { Start-Sleep 1 }
    }
  }
  return $null
}
for ($i = 0; $i -lt 5; $i++) {
  $busy = InUse
  if (-not $busy) { exit 0 }
  Start-Sleep 1
}
Say ("Nebula EH is still running (possibly as administrator): its {0} folder is in use. Close it, or restart Windows, then run Setup again." -f $busy)
exit 3
