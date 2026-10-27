# Undo the Safe sandbox (sandbox.ps1): Nebula's distro and everything in it (Podman, its images and containers), its
# folder under %LOCALAPPDATA%\NebulaEH, and sandbox.json. WSL itself stays, as do your models and settings.
#   powershell -NoProfile -ExecutionPolicy Bypass -File sandbox-undo.ps1 -Conf <settings folder>
param([Parameter(Mandatory = $true)][string]$Conf, [string]$Distro = "NebulaEH")
& wsl.exe --terminate $Distro 2>$null | Out-Null
& wsl.exe --unregister $Distro 2>$null | Out-Null
Remove-Item -Recurse -Force (Join-Path $env:LOCALAPPDATA "NebulaEH") -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $Conf "sandbox.json") -ErrorAction SilentlyContinue
Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce" -Name "NebulaEHSandbox" -ErrorAction SilentlyContinue
Write-Output "The Safe sandbox is removed; Nebula runs engines without it."
