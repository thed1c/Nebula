; Nebula EH for Windows: one click (design decision 2026-10-20). makensis /DVERSION=x.y.z /DPAYLOAD=<payload folder> installer.nsi
; Installs for this user only (no administrator): %LOCALAPPDATA%\Programs\Nebula EH. It carries what pip cannot provide
; (Impulse, llama-server, PAIR) and fetches the rest with install.ps1 (its own Python, Nebula's packages, CUDA's libraries,
; Halo's and Lookup's models); a Start menu and a desktop icon open Nebula in its own window.
Unicode true
!include "MUI2.nsh"
!ifndef VERSION
  !define VERSION "0.1.0"
!endif
Name "Nebula EH ${VERSION}"
OutFile "Nebula-EH-Setup-${VERSION}.exe"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\Nebula EH"
SetCompressor /SOLID lzma
!define MUI_ICON "${PAYLOAD}\app\nebula.ico"
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION OpenNebula
!define MUI_FINISHPAGE_RUN_TEXT "Open Nebula EH"
!insertmacro MUI_PAGE_WELCOME
ShowInstDetails show
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function OpenNebula
  SetOutPath "$APPDATA\nebula"
  Exec '"$INSTDIR\python\pythonw.exe" -m nebula.app --config "$APPDATA\nebula\nebula.json"'
FunctionEnd

Section "Nebula EH"
  ; a running Nebula holds its files: ask it to close first
  nsExec::Exec 'taskkill /F /IM impulse-server.exe'
  nsExec::Exec 'taskkill /F /IM llama-server.exe'
  nsExec::Exec 'taskkill /F /IM nvpair-ui-broker.exe'
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\impulse"
  RMDir /r "$INSTDIR\halo"
  RMDir /r "$INSTDIR\pair"
  SetOutPath "$INSTDIR"
  File /r "${PAYLOAD}\*.*"
  ; the first settings (kept on an update)
  CreateDirectory "$APPDATA\nebula"
  IfFileExists "$APPDATA\nebula\nebula.json" +4 0
    FileOpen $0 "$APPDATA\nebula\nebula.json" w
    FileWrite $0 '{$\r$\n "port": 8090,$\r$\n "ledger": "nebula.sqlite3",$\r$\n "tiers": []$\r$\n}$\r$\n'
    FileClose $0
  ; the rest from the internet: Python, Nebula's packages, CUDA's libraries, the models (its lines show above)
  DetailPrint "Installing from the internet (Python, Nebula's packages, models): a few minutes..."
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\install.ps1" -Dest "$INSTDIR" -Conf "$APPDATA\nebula"'
  Pop $1
  StrCmp $1 "0" +3 0
    MessageBox MB_ICONSTOP "Nebula EH could not finish installing: the lines above say why. Check the internet connection and run Setup again." /SD IDOK
    Abort
  ; the Safe sandbox (sandbox.ps1): engines in containers (Podman in WSL2), as on Linux; declined or failed, they run as
  ; plain programs with a warning. Turning WSL2 on asks for an administrator once, and may need one restart.
  MessageBox MB_YESNO|MB_ICONQUESTION "Run your models in a Safe sandbox?$\r$\n$\r$\nNebula puts each engine in its own container (WSL2 + Podman), as it does on Linux. If WSL2 is not on this PC yet, Windows asks for an administrator once and may need one restart (the sandbox then finishes by itself at your next sign-in).$\r$\n$\r$\nNo: engines run as plain programs, with a warning." /SD IDYES IDNO sandbox_no
    DetailPrint "Setting up the Safe sandbox (WSL2 + Podman): a few minutes..."
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\sandbox.ps1" -Dest "$INSTDIR" -Conf "$APPDATA\nebula"'
    Pop $1
    Goto sandbox_done
  sandbox_no:
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\sandbox.ps1" -Dest "$INSTDIR" -Conf "$APPDATA\nebula" -Decline'
    Pop $1
  sandbox_done:
  SetOutPath "$APPDATA\nebula"                ; the icons start Nebula in its own data folder
  CreateShortCut "$SMPROGRAMS\Nebula EH.lnk" "$INSTDIR\python\pythonw.exe" '-m nebula.app --config "$APPDATA\nebula\nebula.json"' "$INSTDIR\app\nebula.ico"
  CreateShortCut "$DESKTOP\Nebula EH.lnk" "$INSTDIR\python\pythonw.exe" '-m nebula.app --config "$APPDATA\nebula\nebula.json"' "$INSTDIR\app\nebula.ico"
  WriteUninstaller "$INSTDIR\Uninstall Nebula EH.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH" "DisplayName" "Nebula EH"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH" "Publisher" "Nebula EH"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH" "DisplayIcon" "$INSTDIR\app\nebula.ico"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH" "UninstallString" '"$INSTDIR\Uninstall Nebula EH.exe"'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH" "NoModify" 1
SectionEnd

Section "Uninstall"
  nsExec::Exec 'taskkill /F /IM impulse-server.exe'
  nsExec::Exec 'taskkill /F /IM llama-server.exe'
  nsExec::Exec 'taskkill /F /IM nvpair-ui-broker.exe'
  Delete "$SMPROGRAMS\Nebula EH.lnk"
  Delete "$DESKTOP\Nebula EH.lnk"
  ; the Safe sandbox's distro and its images go too (WSL itself, your models and settings stay)
  IfFileExists "$INSTDIR\sandbox-undo.ps1" 0 +2
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\sandbox-undo.ps1" -Conf "$APPDATA\nebula"'
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH"
  ; your settings, conversations and models stay in %APPDATA%\nebula
SectionEnd
