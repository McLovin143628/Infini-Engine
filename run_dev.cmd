@echo off
setlocal

REM Launch Infinity Engine in development mode.
REM Double-click this from Explorer, or run from any shell -- it works either way.

cd /d "%~dp0"

REM Ensure the Rust toolchain is visible. rustup installs to
REM %USERPROFILE%\.cargo\bin and adds it to PATH at install time, but a
REM cmd.exe launched from Explorer may have a stale PATH until next login.
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

echo Launching Infinity Engine (dev mode, Vite on port 1440)...
echo First launch compiles the engine workspace; expect a few minutes.
echo Subsequent launches are seconds.
echo.

call npm run tauri dev

if errorlevel 1 (
    echo.
    echo Dev server exited with an error. Press any key to close.
    pause >nul
)

endlocal
