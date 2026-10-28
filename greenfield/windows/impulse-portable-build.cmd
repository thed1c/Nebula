@echo off
rem Impulse portable for Windows (impulse/portable: llama.cpp's server path with Impulse's hooks, CPU + Vulkan, no CUDA), for
rem Halo on a PC without an NVIDIA card. Needs VS Build Tools 2022 (C++), CMake + Ninja (from the VS prompt) and LunarG's
rem Vulkan SDK; run from a folder holding this repository's tree as src-014 (or edit SRC). Output: impulse-portable\ (zip it as
rem impulse-portable-<version>-windows-x86_64.zip for the release).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set VULKAN_SDK=C:\VulkanSDK\1.4.363.0
set PATH=C:\nb\python;C:\nb\python\Scripts;C:\VulkanSDK\1.4.363.0\Bin;%PATH%
cd /d %~dp0
echo === configure
cmake -S src-014\impulse\portable -B build-portable -G Ninja -DCMAKE_BUILD_TYPE=Release 
if errorlevel 1 exit /b 1
echo === build
cmake --build build-portable --target impulse-server -j 10
if errorlevel 1 exit /b 1
if exist impulse-portable rmdir /s /q impulse-portable
mkdir impulse-portable
copy /y build-portable\bin\impulse-server.exe impulse-portable\ >nul
copy /y build-portable\bin\*.dll impulse-portable\ >nul
for /d %%D in ("C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\14.*") do (
  if exist "%%D\x64\Microsoft.VC143.CRT" copy /y "%%D\x64\Microsoft.VC143.CRT\*.dll" impulse-portable\ >nul
  if exist "%%D\x64\Microsoft.VC143.OpenMP" copy /y "%%D\x64\Microsoft.VC143.OpenMP\*.dll" impulse-portable\ >nul
)
copy /y src-014\impulse\flashnext\third_party\ggml\LICENSE impulse-portable\LICENSE-llama.cpp >nul
impulse-portable\impulse-server.exe --version
impulse-portable\impulse-server.exe --list-devices
dir /b impulse-portable
echo === PORTABLE BUILD DONE
