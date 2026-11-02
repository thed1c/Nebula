; Nebula EH for Windows: one click (design decision 2026-10-20). makensis /DVERSION=x.y.z /DPAYLOAD=<payload folder> installer.nsi
; Installs for this user only (no administrator): %LOCALAPPDATA%\Programs\Nebula EH. It carries what pip cannot provide
; (Impulse, llama-server, PAIR) and fetches the rest with install.ps1 (its own Python, Nebula's packages, CUDA's libraries,
; Halo's and Lookup's models); a Start menu and a desktop icon open Nebula in its own window.
Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"
!ifndef VERSION
  !define VERSION "0.1.0"
!endif
Name "Nebula EH ${VERSION}"
OutFile "Nebula-EH-Setup-${VERSION}.exe"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\Nebula EH"
SetCompressor /SOLID lzma
!define MUI_ICON "${PAYLOAD}\app\nebula.ico"
; Nebula's own Update button runs Setup with /UPDATE (Nebulas after 0.1.6.2; older ones pass /S: silent): no welcome or finish
; page, its window in front of the others (Nebula has just closed, and Setup must not be hidden behind everything), and
; it closes by itself when it is done; Nebula's update helper then opens Nebula again.
Var Update
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION OpenNebula
!define MUI_FINISHPAGE_RUN_TEXT "Open Nebula EH"
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipOnUpdate
!insertmacro MUI_PAGE_WELCOME
ShowInstDetails show
!define MUI_PAGE_CUSTOMFUNCTION_SHOW SetupInFront
!insertmacro MUI_PAGE_INSTFILES
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipOnUpdate
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${GetParameters} $0
  ClearErrors
  ${GetOptions} $0 "/UPDATE" $1
  IfErrors +2 0
    StrCpy $Update "1"
FunctionEnd

Function SkipOnUpdate
  StrCmp $Update "1" 0 +2
    Abort                                   ; a page's pre-function that aborts skips the page
FunctionEnd

Function SetupInFront
  ; above every other window once (topmost, then not: it does not stay on top), and the active one when Windows lets it
  ; (Nebula allowed it before it closed); an ordinary Setup started by hand is left as Windows shows it
  StrCmp $Update "1" 0 in_front_done        ; labels, not relative jumps: a plugin call is more than one instruction
    System::Call 'user32::SetWindowPos(p $HWNDPARENT, p -1, i 0, i 0, i 0, i 0, i 0x43)'
    System::Call 'user32::SetWindowPos(p $HWNDPARENT, p -2, i 0, i 0, i 0, i 0, i 0x43)'
    System::Call 'user32::SetForegroundWindow(p $HWNDPARENT)'
    BringToFront
  in_front_done:
FunctionEnd

Function OpenNebula
  ; through Explorer and the Start menu's icon: Nebula opens as the signed-in user even when Setup itself was started as
  ; administrator (a Nebula left running as administrator holds its files, and the next Setup cannot close it)
  Exec '"$WINDIR\explorer.exe" "$SMPROGRAMS\Nebula EH.lnk"'
FunctionEnd

Section "Nebula EH"
  ; a running Nebula holds its files: close it first (its programs, found by their path), and stop here, with nothing
  ; touched, when its files are still in use (a Nebula started as administrator, which this user cannot close)
  InitPluginsDir
  File "/oname=$PLUGINSDIR\close-nebula.ps1" "close-nebula.ps1"
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\close-nebula.ps1" -Dest "$INSTDIR" -Conf "$APPDATA\nebula"'
  Pop $1
  StrCmp $1 "3" 0 +3
    MessageBox MB_ICONSTOP "Nebula EH is still running (possibly as administrator): close it, or restart Windows, then run Setup again." /SD IDOK
    Abort
  ; Python and Nebula's packages stay when Python's pinned version did not change: install.ps1 decides, and pip then
  ; fetches only the packages whose version moved (a fresh Python, and every package, only when Python itself changed).
  ; What the payload carries is put in place again from Setup itself, nothing from the internet.
  StrCmp $Update "1" 0 +2
    SetAutoClose true                       ; Nebula's Update: Setup closes by itself once it is done
  RMDir /r "$INSTDIR\impulse"
  RMDir /r "$INSTDIR\halo"
  RMDir /r "$INSTDIR\pair"
  SetOutPath "$INSTDIR"
  File /r "${PAYLOAD}\*.*"
  File "close-nebula.ps1"                     ; the uninstaller closes Nebula the same way
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
  StrCmp $1 "0" install_ok 0
  StrCmp $1 "3" 0 +3
    MessageBox MB_ICONSTOP "Nebula EH could not finish installing: some of its files are in use (Nebula EH still running, possibly as administrator). Close it, or restart Windows, then run Setup again." /SD IDOK
    Abort
  MessageBox MB_ICONSTOP "Nebula EH could not finish installing: the lines above say why (often the internet connection). Fix that and run Setup again." /SD IDOK
  Abort
  install_ok:
  ; the Safe sandbox (sandbox.ps1): engines in containers (Podman in WSL2), as on Linux; declined or failed, they run as
  ; plain programs with a warning. Turning WSL2 on asks for an administrator once, and may need one restart.
  ; One already in place (or waiting for its restart) is kept and brought up to date without a question (and without
  ; fetching Impulse again when it did not change); an update never asks, so the choice made at the install stands.
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\sandbox.ps1" -Dest "$INSTDIR" -Conf "$APPDATA\nebula" -Keep'
  Pop $1
  StrCmp $1 "4" 0 sandbox_done              ; 4: no sandbox in place (none yet, declined, or failed)
  StrCmp $Update "1" sandbox_done
  IfSilent sandbox_done
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
  IfFileExists "$INSTDIR\close-nebula.ps1" 0 +2
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\close-nebula.ps1" -Dest "$INSTDIR" -Conf "$APPDATA\nebula"'
  Delete "$SMPROGRAMS\Nebula EH.lnk"
  Delete "$DESKTOP\Nebula EH.lnk"
  ; the Safe sandbox's distro and its images go too (WSL itself, your models and settings stay)
  IfFileExists "$INSTDIR\sandbox-undo.ps1" 0 +2
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\sandbox-undo.ps1" -Conf "$APPDATA\nebula"'
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NebulaEH"
  ; your settings, conversations and models stay in %APPDATA%\nebula
SectionEnd
