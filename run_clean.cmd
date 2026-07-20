@echo off
setlocal

REM Prune Rust build artifacts in target\.
REM
REM   run_clean.cmd          "cargo sweep --time 30": removes artifacts not
REM                          touched for 30+ days (stale third-party deps left
REM                          behind by dependency bumps and toolchain
REM                          upgrades). Keeps the warm cache, so the next
REM                          build only recompiles what actually changed.
REM   run_clean.cmd --full   Full "cargo clean": reclaims everything; the
REM                          next build is from scratch (several minutes).
REM
REM (GeoCanvas additionally ships an identity-orphan pruner for caches cargo
REM abandons on feature/dep-graph changes; port prune_orphans.ps1 here if
REM target\ ever bloats faster than the sweep can catch.)
REM
REM Never run this while a build is in flight -- it deletes fingerprint dirs
REM mid-build.
REM
REM Double-click from Explorer or run from any shell.

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
    echo.
    pause
    exit /b 1
)

set "TARGET_DIR=%~dp0target"
if not exist "%TARGET_DIR%" (
    echo Nothing to clean: %TARGET_DIR% does not exist.
    pause
    exit /b 0
)

for /f %%s in ('powershell -NoProfile -Command "[math]::Round((Get-ChildItem -LiteralPath '%TARGET_DIR%' -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum/1GB,2)"') do set "SIZE_BEFORE=%%s"
echo target\ size before: %SIZE_BEFORE% GB
echo.

if /i "%~1"=="--full" goto :full

echo Sweeping artifacts unused for 30+ days...
where cargo-sweep >nul 2>&1
if errorlevel 1 (
    echo cargo-sweep not installed. Installing once via "cargo install cargo-sweep"...
    cargo install cargo-sweep
    if errorlevel 1 (
        echo.
        echo cargo-sweep install failed. For a full clean instead, re-run with --full.
        pause
        exit /b 1
    )
)
cargo sweep --time 30
goto :report

:full
cargo clean

:report
echo.
set "SIZE_AFTER=0"
if exist "%TARGET_DIR%" (
    for /f %%s in ('powershell -NoProfile -Command "[math]::Round((Get-ChildItem -LiteralPath '%TARGET_DIR%' -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum/1GB,2)"') do set "SIZE_AFTER=%%s"
)
echo target\ size after:  %SIZE_AFTER% GB  (was %SIZE_BEFORE% GB)
pause

endlocal
