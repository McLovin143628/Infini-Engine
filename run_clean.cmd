@echo off
setlocal

REM Prune Rust build artifacts in target\.
REM
REM Cargo leaks build cache two different ways, and each needs its own tool:
REM
REM   1. By IDENTITY. Every incremental cache and compiled artifact is keyed by
REM      a hash of the crate's feature set + dep graph + rustc version. Change
REM      any of those and cargo mints a fresh hash and orphans the old one,
REM      forever. This workspace is the worst offender of the family: 30+
REM      crates, a wgpu dep graph, and wasm32/android cross-compiles. The July
REM      2026 audit found 943 incremental caches across 101 targets in a 132 GB
REM      target\ -- 80 GB of it orphaned. Handled by prune_orphans.ps1 (keeps
REM      the 2 newest generations per target; never touches third-party deps).
REM
REM   2. By AGE. Third-party artifacts left behind by dependency bumps and
REM      toolchain upgrades. Handled by "cargo sweep --time 30". Note this
REM      prunes by last-ACCESS time, so it cannot see the orphans in (1) --
REM      every link pass touches those, so they never age out. The two tools
REM      are complements, not alternatives.
REM
REM   run_clean.cmd          Both of the above. Keeps the warm cache, so the
REM                          next build only recompiles the workspace crates.
REM   run_clean.cmd --prune  Identity prune only. Fastest; third-party deps are
REM                          left completely untouched.
REM   run_clean.cmd --full   Full "cargo clean": reclaims everything; the
REM                          next build is from scratch (several minutes).
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

REM ---- Step 1: prune artifacts cargo orphaned by identity -------------------
echo [1/2] Pruning orphaned build generations...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0prune_orphans.ps1"
if errorlevel 1 (
    echo.
    echo Prune aborted. Nothing was removed.
    pause
    exit /b 1
)
echo.

if /i "%~1"=="--prune" goto :report

REM ---- Step 2: sweep third-party artifacts stale by age ---------------------
echo [2/2] Sweeping artifacts unused for 30+ days...
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
