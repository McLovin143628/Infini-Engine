# THE DEMO LOOP (wave FIX1) — build the editor, boot it on the showcase island,
# press its own Play button, drive the game, and photograph the result.
#
# A wave that ends in a green battery has proved the tests agree with the code.
# It has not proved that the editor opens, that Play plays, or that the character
# walks. Every wave from FIX1 onward ends here. See tools/demo/README.md.
param(
    [string]$OutDir = "",
    [switch]$SkipBuild,
    [switch]$KeepOpen,
    [int]$Port = 9222,
    [int]$BootWaitS = 60,
    [int]$PieWaitS = 240,
    [int]$LoadSettleS = 20
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$release = Join-Path $repo "target\release"
$exe = Join-Path $release "inf-studio.exe"
if ($OutDir -eq "") {
    $OutDir = Join-Path $env:TEMP ("inf-demo-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "demo.log"
$heroCsv = Join-Path $OutDir "hero.csv"
$shot = Join-Path $PSScriptRoot "screenshot.ps1"

function Say([string]$text) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $text
    Write-Output $line
    Add-Content -Path $log -Value $line
}

Say "repo    $repo"
Say "out     $OutDir"

# ── 0. nothing of ours may be running ────────────────────────────────────────
#
#    The island's pack is memory-mapped and a build that tries to replace a
#    RUNNING executable fails as a sharing violation, which MSVC reports as
#    LNK1104 and which reads like a disk problem. Refuse early and say why.
$running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -in @("inf-studio", "inf-player") }
if ($running) {
    Say ("REFUSED: these are already running -> " + (($running | ForEach-Object { "$($_.ProcessName)/$($_.Id)" }) -join ", "))
    Say "Close the editor first, or pass -SkipBuild to photograph what is already built."
    exit 2
}

# ── 1. build ─────────────────────────────────────────────────────────────────
if (-not $SkipBuild) {
    Say "building: npx tauri build --no-bundle"
    Push-Location (Join-Path $repo "editor\studio")
    # `cargo build --release -p inf-studio` is NOT the same thing and produces an
    # editor that loads the DEV url: the frontend has to be built and embedded,
    # which is what the tauri CLI does.
    & npx tauri build --no-bundle 2>&1 | ForEach-Object { Add-Content -Path $log -Value $_ }
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) {
        Say "BUILD FAILED (exit $code) — see $log"
        exit 3
    }
    Say "build ok"
}
if (-not (Test-Path $exe)) {
    Say "REFUSED: no editor at $exe"
    exit 3
}
Say ("editor  {0} ({1:N1} MB, built {2})" -f $exe, ((Get-Item $exe).Length / 1MB), (Get-Item $exe).LastWriteTime)

# ── 2. launch, from the executable's OWN directory ───────────────────────────
#
#    The boot ladder discovers the showcase by walking up from the running
#    executable, so the working directory is load-bearing: launched from
#    elsewhere the editor opens the start screen instead of the island.
$env:INF_WEBVIEW_DEBUG_PORT = "$Port"
$env:INF_PIE_HERO_LOG = $heroCsv
$proc = Start-Process -FilePath $exe -WorkingDirectory $release -PassThru
Say "launched pid $($proc.Id); waiting up to $BootWaitS s for the shell"

$booted = $false
for ($i = 0; $i -lt $BootWaitS; $i++) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) { Say "EDITOR EXITED with $($proc.ExitCode)"; exit 4 }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/json" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $booted = $true; Say "debug port open after $($i + 1) s"; break }
    } catch { }
}
if (-not $booted) { Say "debug port never opened; falling back to a fixed wait"; Start-Sleep -Seconds 15 }
# The shell paints its panels after the port opens; give the island's document a
# moment to land in the Outliner before the first frame is taken.
Start-Sleep -Seconds 10

& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01-editor.png") -WindowTitle "Infini" -Foreground |
    ForEach-Object { Say $_ }

# ── 3. press Play ────────────────────────────────────────────────────────────
$pressed = $false
if (Get-Command node -ErrorAction SilentlyContinue) {
    Say "pressing Play over CDP"
    & node (Join-Path $PSScriptRoot "play.mjs") $Port 8 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -eq 0) { $pressed = $true } else { Say "  cdp failed (exit $LASTEXITCODE)" }
} else {
    Say "node is not on the PATH"
}
if (-not $pressed) {
    # The fallback: the Play cluster's first button on a maximized 1080p window.
    Add-Type -AssemblyName System.Windows.Forms
    Say "pressing Play by coordinate (1220, 49)"
    $wshell = New-Object -ComObject wscript.shell
    $wshell.AppActivate($proc.Id) | Out-Null
    Start-Sleep -Milliseconds 600
    [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point(1220, 49)
    Start-Sleep -Milliseconds 200
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class InfClick {
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint x, uint y, uint d, IntPtr e);
}
"@
    [InfClick]::mouse_event(0x0002, 0, 0, 0, [IntPtr]::Zero)
    [InfClick]::mouse_event(0x0004, 0, 0, 0, [IntPtr]::Zero)
}

# ── 4. wait for the player ───────────────────────────────────────────────────
Say "waiting up to $PieWaitS s for inf-player.exe"
$player = $null
for ($i = 0; $i -lt $PieWaitS; $i++) {
    $player = Get-Process -Name "inf-player" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($player) { Say "player pid $($player.Id) after $($i + 1) s"; break }
    if ($proc.HasExited) { Say "EDITOR EXITED with $($proc.ExitCode)"; exit 4 }
    Start-Sleep -Seconds 1
}
if (-not $player) {
    Say "NO PLAYER after $PieWaitS s"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "02-no-player.png") | ForEach-Object { Say $_ }
    if (-not $KeepOpen) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    exit 5
}

# A console window is the defect this wave closed; look for one belonging to
# either process while both are alive.
$consoles = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -eq "conhost" -or $_.ProcessName -eq "WindowsTerminal" } |
    Where-Object { $_.MainWindowTitle -like "*inf-player*" }
Say ("console windows named inf-player: " + $(if ($consoles) { ($consoles | ForEach-Object { $_.MainWindowTitle }) -join "; " } else { "none" }))

Say "letting the level stream for $LoadSettleS s"
Start-Sleep -Seconds $LoadSettleS

# ── 5. drive it, and photograph two seconds apart ────────────────────────────
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class InfInput {
  [StructLayout(LayoutKind.Sequential)] struct KEYBDINPUT { public ushort wVk, wScan; public uint dwFlags, time; public IntPtr dwExtraInfo; }
  [StructLayout(LayoutKind.Sequential)] struct INPUT { public uint type; public KEYBDINPUT ki; public int pad1, pad2; }
  [DllImport("user32.dll", SetLastError = true)] static extern uint SendInput(uint n, INPUT[] p, int cb);
  [DllImport("user32.dll")] static extern void mouse_event(uint f, uint x, uint y, uint d, IntPtr e);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  const uint KEYEVENTF_SCANCODE = 0x0008, KEYEVENTF_KEYUP = 0x0002;
  static void Key(ushort scan, bool down) {
    INPUT[] i = new INPUT[1];
    i[0].type = 1;
    i[0].ki.wScan = scan;
    i[0].ki.dwFlags = KEYEVENTF_SCANCODE | (down ? 0u : KEYEVENTF_KEYUP);
    SendInput(1, i, Marshal.SizeOf(typeof(INPUT)));
  }
  public static void Down(ushort scan) { Key(scan, true); }
  public static void Up(ushort scan) { Key(scan, false); }
  public static void Click(int x, int y) {
    SetCursorPos(x, y);
    mouse_event(0x0002, 0, 0, 0, IntPtr.Zero);
    mouse_event(0x0004, 0, 0, 0, IntPtr.Zero);
  }
}
"@

# A click into the middle of the viewport: the embedded player's window takes the
# keyboard on a click even when the WebView had it (mouse messages are routed by
# hit-test, key messages by focus — the whole of the FIX1 finding).
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
[InfInput]::Click([int]($screen.Width / 2), [int]($screen.Height / 2))
Start-Sleep -Milliseconds 800

Say "holding W"
[InfInput]::Down(0x11)   # scancode: W
Start-Sleep -Milliseconds 900
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "02-pie-a.png") | ForEach-Object { Say $_ }
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "03-pie-b.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x11)
Say "released W"

# ── 6. what the hero did, in metres ──────────────────────────────────────────
if (Test-Path $heroCsv) {
    $rows = Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" }
    if ($rows.Count -ge 2) {
        $a = $rows[0].Split(","); $b = $rows[-1].Split(",")
        $dx = [double]$b[2] - [double]$a[2]
        $dz = [double]$b[4] - [double]$a[4]
        $d = [math]::Sqrt($dx * $dx + $dz * $dz)
        Say ("hero first : t={0} ({1}, {2}, {3}) {4} speed {5}" -f $a[0], $a[2], $a[3], $a[4], $a[5], $a[6])
        Say ("hero last  : t={0} ({1}, {2}, {3}) {4} speed {5}" -f $b[0], $b[2], $b[3], $b[4], $b[5], $b[6])
        Say ("HERO MOVED {0:N3} m over {1} samples" -f $d, $rows.Count)
    } else {
        Say "hero.csv has $($rows.Count) row(s) — the player wrote no positions"
    }
} else {
    Say "no hero.csv at $heroCsv"
}

Say ("windows now: " + ((Get-Process | Where-Object { $_.MainWindowTitle -ne "" -and ($_.ProcessName -like "inf*") } |
    ForEach-Object { "$($_.ProcessName)[$($_.Id)] '$($_.MainWindowTitle)'" }) -join " | "))

# ── 7. close ─────────────────────────────────────────────────────────────────
if ($KeepOpen) {
    Say "left running (pid $($proc.Id)); the island's pack stays mapped until you close it"
} else {
    Say "closing"
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Get-Process -Name "inf-player" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Say ("still running: " + $(if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in @("inf-studio", "inf-player") }) { "YES" } else { "none" }))
}
Say "done — $OutDir"
