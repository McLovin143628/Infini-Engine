@echo off
setlocal

REM Build Infini Engine in release mode.
REM Double-click this from Explorer, or run from any shell -- it works either way.
REM
REM Output:
REM   target\release\inf-studio.exe   (the editor, frontend embedded)
REM
REM Installer bundling (NSIS/MSI) is deliberately OFF until the Phase 9
REM packaging work lands -- tauri.conf.json has "bundle": { "active": false }.

cd /d "%~dp0"

REM Same PATH bootstrapping as run_dev.cmd: a cmd.exe launched from Explorer
REM may not have %USERPROFILE%\.cargo\bin on PATH yet.
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

where cargo >nul 2>&1
if errorlevel 1 (
    echo Cargo not found on PATH.
    echo Install the Rust toolchain from https://rustup.rs and try again.
    echo If you just installed it, open a new shell so the PATH update takes effect.
    echo.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo npm not found on PATH.
    echo Install Node.js LTS from https://nodejs.org and try again.
    echo.
    pause
    exit /b 1
)

cd editor\studio

if not exist "node_modules" (
    echo node_modules not found. Running "npm install" first...
    call npm install
    if errorlevel 1 (
        echo.
        echo npm install failed. Press any key to close.
        pause >nul
        exit /b 1
    )
)

echo ================================================================
echo  Building Infini Engine (release)
echo.
echo  This type-checks + builds the frontend, then compiles the
echo  whole engine workspace in release mode. The first release
echo  build can take several minutes; later builds are incremental.
echo ================================================================
echo.

call npm run tauri build
if errorlevel 1 (
    echo.
    echo Build failed. See the output above. Press any key to close.
    pause >nul
    exit /b 1
)

set "EXE=%~dp0target\release\inf-studio.exe"

echo.
echo ================================================================
if exist "%EXE%" (
    echo  Build complete.
    echo.
    echo  Editor binary: %EXE%
) else (
    echo  Build finished, but %EXE% was not found.
    echo  Check target\release\ for the produced artifacts.
)
echo ================================================================
echo.
pause
endlocal
