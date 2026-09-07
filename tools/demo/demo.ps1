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
    # "embedded" reparents the player into the viewport hole; "window" is the
    # roadmap-sanctioned Play in New Window. Both must move the hero, so both
    # are drivable from here.
    # **`window` by default since the CHAR1b.2 audit** (carried item 129). Two
    # `embedded` runs of wave CHAR1b.2 filmed `HERO MOVED 0.000 m` over 313 and
    # 315 samples: the click meant for the viewport hole left the EDITOR in the
    # foreground and every keystroke went to it. `-PlayMode embedded` still
    # works and is still the roadmap's own preview, so it is one flag away --
    # what changed is which one a wave gets by accident.
    [ValidateSet("embedded", "window")][string]$PlayMode = "window",
    # **A DEV-ONLY placement for the preview session** (CHAR1b.2 audit). A
    # `;`-separated list of `x,y,z@seconds`, handed to the player as
    # `INF_PIE_SPAWN_AT`; the player applies each one once, in a `--pie` preview
    # only, and writes a line into the hero log saying it did. It exists because
    # the wave before this one filmed neither the mantle, nor the water, nor a
    # measured drop and wrote down "the shipped player has no teleport" -- which
    # is a limitation of THIS SCRIPT, not of the game.
    [string]$SpawnAt = "",
    # A `.inf_cloth` GUID to put on the hero for the session, `INF_PIE_WEAR_CLOTH`.
    # The cape wave CHAR1b.2 authored lives in the island's Content and is worn
    # in the gate; carried 137 is that it is not in the committed level, and this
    # is how the loop photographs it without making that edit.
    [string]$WearCloth = "",
    [int]$BootWaitS = 60,
    [int]$PieWaitS = 240,
    [int]$LoadSettleS = 20,
    # **The floor that makes this a GATE and not a report** (audit FIX1). The
    # wave that wrote this script printed HERO MOVED and exited 0 whatever the
    # number was -- including the runs it later found had moved 0.000 m, which
    # were noticed by a person reading the log. Twelve metres is what a held W
    # buys in the seconds this script allows; five is clear of a settle, a slide
    # or a camera drift and far below anything a walking character does.
    [double]$MinMetres = 5.0,
    # **How long the EDITOR is given to finish streaming** before the frame a
    # claim about the editor is made on. CHAR1a photographed the viewport at
    # 27/52 and could not tell an unresolved material from a dropped one.
    #
    # **90 since wave CHAR1a.3, and it is a measurement.** On a project whose
    # derived assets are COLD -- the first run after `inf island build`, which is
    # exactly the run a wave takes its frames on -- 45 s left the editor still
    # populating (52/52 arrived at +65 s), the Play click landed on a busy main
    # thread, the button never changed out of "Play" and the loop reported
    # "NO PLAYER after 240 s". Measured twice: the identical run at 90 s reached
    # Play in 6 s and the hero moved 17.5 m. The failure looks like a broken Play
    # button and is a stopwatch.
    [int]$EditorSettleS = 90,
    # Place the second committed body beside the pawn before the editor frame,
    # in the DOCUMENT only. See tools/demo/place.mjs for why it is not saved.
    [bool]$PlaceFemale = $true,
    # Photograph the hero's FACE at ~1.7 m before Play. See tools/demo/portrait.mjs:
    # the loop's own camera is behind the character and a head is 30 px of a 1080p
    # frame, which is not evidence about a face.
    [bool]$Portrait = $true
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

$failed = $false

# **A SHOT TRIGGERED BY WHAT THE HERO IS DOING** (CHAR1b.2 audit, carried 132).
# The loop's frames are timed by `Start-Sleep` and the states this wave added
# last about a second, so `32-slide.png` and `34-prone.png` were photographs of a
# standing character with the right filename. `hero.csv` already carries the
# machine state in column 12 and the position in 3..5, four times a second; this
# waits for a predicate over the newest row and shoots when it holds, or says
# plainly that it never did.
function Wait-ForHero {
    param(
        [string]$Csv,
        [scriptblock]$Predicate,
        [string]$What,
        [double]$TimeoutS = 8.0,
        [string]$Out = ""
    )
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $Csv) {
            $rows = @(Get-Content $Csv -ErrorAction SilentlyContinue | Where-Object { $_ -match "^[0-9]" })
            if ($rows.Count -gt 0) {
                $c = $rows[-1].Split(",")
                if (& $Predicate $c) {
                    Say "TRIGGER $What after $([math]::Round(($TimeoutS - ($deadline - (Get-Date)).TotalSeconds), 2)) s: $($rows[-1])"
                    if ($Out -ne "") {
                        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out $Out | ForEach-Object { Say $_ }
                    }
                    return $true
                }
            }
        }
        Start-Sleep -Milliseconds 120
    }
    Say "TRIGGER $What NEVER FIRED inside $TimeoutS s -- no frame taken"
    return $false
}

Say "repo    $repo"
Say "mode    $PlayMode"
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
    # THE PLAYER FIRST, and it is not optional: `npx tauri build` builds the
    # EDITOR. `find_player_bin` looks for `inf-player.exe` beside the editor it
    # is running from, so a demo that built only the editor would press Play on
    # whatever player happened to be in `target/release` -- which on a dev
    # machine is the one from the wave before last, and which is exactly the
    # trap this comment exists to keep the next person out of.
    Say "building: cargo build --release -p inf-player"
    & cargo build --release -p inf-player 2>&1 | ForEach-Object { Add-Content -Path $log -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        Say "PLAYER BUILD FAILED (exit $LASTEXITCODE) — see $log"
        exit 3
    }
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
$playerExe = Join-Path $release "inf-player.exe"
if (-not (Test-Path $playerExe)) {
    Say ("REFUSED: no player at $playerExe - build it with: cargo build --release -p inf-player")
    exit 3
}
Say ("editor  {0} ({1:N1} MB, built {2})" -f $exe, ((Get-Item $exe).Length / 1MB), (Get-Item $exe).LastWriteTime)
Say ("player  {0} ({1:N1} MB, built {2})" -f $playerExe, ((Get-Item $playerExe).Length / 1MB), (Get-Item $playerExe).LastWriteTime)

# ── 2. launch, from the executable's OWN directory ───────────────────────────
#
#    The boot ladder discovers the showcase by walking up from the running
#    executable, so the working directory is load-bearing: launched from
#    elsewhere the editor opens the start screen instead of the island.
#
#    ── THE CRASH-RECOVERY DOCUMENT IS MOVED ASIDE FIRST (wave CHAR1a audit) ──
#
#    This loop ends by killing the editor, and a killed editor leaves
#    `<app_data>/crash-recovery.inf_lvl` behind — which the NEXT boot silently
#    restores in place of the shipped level. Measured: a run that placed a second
#    body found that body already in the document at boot, before it placed
#    anything, because the previous run's kill had written it there.
#
#    So every frame after the first run of a session was a photograph of the
#    PREVIOUS run's document. It is renamed rather than deleted: it is a
#    recovery file and it belongs to the author, not to this script.
$recovery = Join-Path $env:APPDATA "com.infinityengine.app\crash-recovery.inf_lvl"
if (Test-Path $recovery) {
    $aside = "$recovery.demo-aside"
    Move-Item -Force $recovery $aside
    Say "moved a stale crash-recovery document aside to $aside (it would have been restored over the shipped level)"
}

$env:INF_WEBVIEW_DEBUG_PORT = "$Port"
$env:INF_PIE_HERO_LOG = $heroCsv
# The dev-only preview doors, set only when asked for. `Remove-Item env:` rather
# than an empty string so a session that did not ask for one is a session in
# which the variable does not exist.
if ($SpawnAt -ne "") { $env:INF_PIE_SPAWN_AT = $SpawnAt; Say "spawn override: $SpawnAt" }
else { Remove-Item env:INF_PIE_SPAWN_AT -ErrorAction SilentlyContinue }
if ($WearCloth -ne "") { $env:INF_PIE_WEAR_CLOTH = $WearCloth; Say "wear cloth: $WearCloth" }
else { Remove-Item env:INF_PIE_WEAR_CLOTH -ErrorAction SilentlyContinue }
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

# ── 2b. the SETTLED editor frame, and the female body in it ──────────────────
#
#    **The mid-stream frame is not the editor's answer** (CHAR1a's own caveat):
#    the first shot above was taken while the toolbar still read `Loading
#    world... 27/52`, so a body that had not resolved its material yet looked
#    exactly like a body whose material was dropped. This one waits for the
#    stream and is the frame a claim about the editor may be made on.
Say "waiting $EditorSettleS s for the editor's own streaming to settle"
Start-Sleep -Seconds $EditorSettleS
# ── 2a. THE PORTRAIT ─────────────────────────────────────────
#
# A wave that puts a face on the island has to photograph a face. The loop's own
# camera sits behind the character, so the head is thirty pixels of a 1080p frame
# — which is how wave CHAR1a.3 shipped two BLANK heads and called them "real
# faces". `portrait.mjs` moves the hero onto the editor camera's own view ray a
# metre out, in the open document, and `undo.mjs` puts him back; the frame
# between the two is a portrait. See tools/demo/portrait.mjs for why the
# CHARACTER moves and not the camera.
# It runs BEFORE the female is placed: `scene_player_pawn` answered with the
# newly placed body on one run in four, and the portrait then photographed her
# instead of the hero. With one pawn in the document there is nothing to
# resolve.
if ($Portrait -and (Get-Command node -ErrorAction SilentlyContinue)) {
    Say "framing the hero's face (document only; never saved)"
    & node (Join-Path $PSScriptRoot "portrait.mjs") $Port 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -ne 0) { Say "  portrait.mjs exit $LASTEXITCODE" }
    Start-Sleep -Seconds 3
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01d-portrait.png") -WindowTitle "Infini" -Foreground |
        ForEach-Object { Say $_ }
    # …and back, so the frames after this one are the level's own pose. ONE
    # undo: the portrait is one transaction, the hero's own translation.
    if (Get-Command node -ErrorAction SilentlyContinue) {
        & node (Join-Path $PSScriptRoot "undo.mjs") $Port 1 2>&1 | ForEach-Object { Say "  cdp: $_" }
    }
    Start-Sleep -Seconds 2
}

if ($PlaceFemale -and (Get-Command node -ErrorAction SilentlyContinue)) {
    Say "placing the FEMALE committed body beside the pawn (document only; never saved)"
    & node (Join-Path $PSScriptRoot "place.mjs") $Port 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -ne 0) { Say "  place.mjs exit $LASTEXITCODE" }
    Start-Sleep -Seconds 3
}
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01b-editor-settled.png") -WindowTitle "Infini" -Foreground |
    ForEach-Object { Say $_ }

# ── 3. press Play ────────────────────────────────────────────────────────────
$pressed = $false
if (Get-Command node -ErrorAction SilentlyContinue) {
    Say "pressing Play over CDP"
    & node (Join-Path $PSScriptRoot "play.mjs") $Port 8 $PlayMode 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -eq 0) { $pressed = $true } else { Say "  cdp failed (exit $LASTEXITCODE)" }
} else {
    Say "node is not on the PATH"
}
if (-not $pressed) {
    # The fallback: the Play cluster's first button on a maximized 1080p window.
    Add-Type -AssemblyName System.Windows.Forms
    if ($PlayMode -ne "embedded") {
        Say "REFUSED: -PlayMode window needs the CDP path (there is no coordinate for a menu item)"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        exit 6
    }
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
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern IntPtr GetParent(IntPtr h);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  // Which window the keyboard is going to, and who owns it.
  public static string Foreground() {
    IntPtr h = GetForegroundWindow();
    uint pid; GetWindowThreadProcessId(h, out pid);
    return string.Format("hwnd=0x{0:x} pid={1}", h.ToInt64(), pid);
  }
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
  // **A RELATIVE mouse move** (wave CHAR1b.1). `SetCursorPos` is absolute
  // and a captured cursor is re-centred every frame, so an absolute move is
  // a delta of whatever is left over. `MOUSEEVENTF_MOVE` is what a mouse
  // sends, and it is what a look-at has to be driven with.
  public static void Look(int dx, int dy) {
    mouse_event(0x0001, (uint)dx, (uint)dy, 0, IntPtr.Zero);
  }
  // **The RIGHT button, which is `aim`** (audit CHAR1b.1). Holding it puts the
  // character in `RotationMode::Aiming`; RELEASING it leaves it in
  // `LookingDirection`, and that is the only door to the mode ALS turns in
  // place in. A demo that never right-clicks can never film a turn.
  public static void RightDown() { mouse_event(0x0008, 0, 0, 0, IntPtr.Zero); }
  public static void RightUp() { mouse_event(0x0010, 0, 0, 0, IntPtr.Zero); }
  public static void Click(int x, int y) {
    SetCursorPos(x, y);
    mouse_event(0x0002, 0, 0, 0, IntPtr.Zero);
    mouse_event(0x0004, 0, 0, 0, IntPtr.Zero);
  }
  // A screenshot CANNOT answer "is the cursor hidden" -- `CopyFromScreen` does
  // not draw one either way -- so the author's second sentence needs the OS's
  // own answer. CURSOR_SHOWING is 0x1.
  [StructLayout(LayoutKind.Sequential)] struct CURSORINFO { public int cbSize, flags; public IntPtr hCursor; public int x, y; }
  [DllImport("user32.dll")] static extern bool GetCursorInfo(ref CURSORINFO pci);
  public static string CursorState() {
    CURSORINFO ci = new CURSORINFO();
    ci.cbSize = Marshal.SizeOf(typeof(CURSORINFO));
    if (!GetCursorInfo(ref ci)) return "unknown";
    return ((ci.flags & 0x1) != 0) ? "SHOWING" : "hidden";
  }
}
"@

# WHERE TO CLICK, and it is not a detail.
#
#    An EMBEDDED player is a `WS_CHILD` with no main window of its own, so the
#    middle of the screen is the middle of the viewport hole and clicking there
#    hands it the keyboard (mouse messages are routed by hit-test, key messages
#    by focus -- the whole of the FIX1 finding).
#
#    A NEW-WINDOW player is a separate top-level window that does NOT cover the
#    screen. Clicking the screen's centre there lands on the maximized EDITOR
#    behind it, which takes the foreground back and leaves the game unfocused --
#    measured, and it is why this wave's first new-window run reported 0.000 m
#    with the editor's own Outliner showing `Selected 1`. So the click goes to
#    the player's own rectangle when it has one.
# **The assembly is loaded HERE and not only in the fallback branch** (wave
# CHAR1a audit). `Add-Type -AssemblyName System.Windows.Forms` was inside the
# "CDP failed, click Play by coordinate" branch, which a successful CDP press
# never reaches — so on the path this script normally takes the type was
# unknown, `$screen` was `$null`, and the click that hands an EMBEDDED player
# its keyboard went to `[int]$null, [int]$null` = **(0, 0)**, the top-left
# corner of the screen. Measured: "Unable to find type
# [System.Windows.Forms.Screen]" in the run's own output, followed by a click
# at the origin. The run still moved 17.338 m because the player already had
# the foreground — which is exactly how a defect like this survives.
Add-Type -AssemblyName System.Windows.Forms
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$target = New-Object InfInput+RECT
#    "Has the player a window of its own" is THREE questions, not one, and the
#    first version asked only the first. An EMBEDDED player still reports a
#    `MainWindowHandle` -- a 16x16 stub at (0,0), which is what winit leaves
#    behind once the editor has reparented the real one -- so a size test and a
#    parent test go with it. Without them the driver took the new-window branch
#    on an embedded session, raised a 16-pixel window, sent W to nothing and
#    reported 0.000 m.
$hasWindow = $false
$player.Refresh()
if ($player.MainWindowHandle -ne [IntPtr]::Zero) {
    $gotRect = [InfInput]::GetWindowRect($player.MainWindowHandle, [ref]$target)
    $isTop = [InfInput]::GetParent($player.MainWindowHandle) -eq [IntPtr]::Zero
    $wide = ($target.Right - $target.Left) -ge 200 -and ($target.Bottom - $target.Top) -ge 200
    $hasWindow = $gotRect -and $isTop -and $wide
    if (-not $hasWindow) {
        Say ("the player's reported window is {0}x{1}, top-level={2} — treating it as embedded" -f ($target.Right - $target.Left), ($target.Bottom - $target.Top), $isTop)
    }
}
Say ("foreground before the click: " + [InfInput]::Foreground() + " (editor pid $($proc.Id), player pid $($player.Id))")
if ($hasWindow) {
    $cx = [int](($target.Left + $target.Right) / 2)
    $cy = [int](($target.Top + $target.Bottom) / 2)
    # **NO CLICK.** A session with its own window already holds the keyboard --
    # `take_keyboard_focus` takes it when the window is created and the line
    # above says so -- and a synthetic click into it is not a no-op: measured
    # over eight runs, five of them handed the foreground to the EDITOR on the
    # click and the hero then moved 0.000 m, while the three that kept it moved
    # 12.1-12.8 m. The click exists to give an EMBEDDED player the focus its
    # reparented child window is denied; a top-level one needs raising, not
    # clicking.
    Say ("the player owns its own window [{0},{1} {2}x{3}]; raising it rather than clicking into it" -f $target.Left, $target.Top, ($target.Right - $target.Left), ($target.Bottom - $target.Top))
    [InfInput]::ShowWindow($player.MainWindowHandle, 5) | Out-Null   # SW_SHOW
    [InfInput]::SetForegroundWindow($player.MainWindowHandle) | Out-Null
    [InfInput]::SetCursorPos($cx, $cy) | Out-Null
    Start-Sleep -Milliseconds 400
} else {
    Say "clicking the viewport hole at the screen centre (the player has no window of its own)"
    [InfInput]::Click([int]($screen.Width / 2), [int]($screen.Height / 2))
}
Start-Sleep -Milliseconds 800

# **THE PLAYER LOSES THE KEYBOARD MID-SESSION, AND NOTHING TAKES IT BACK**
# (CHAR1b.2 audit). `window.rs`'s grab ladder is BOUNDED to GRAB_LADDER_FRAMES
# (600 frames = 10 s), deliberately: a session the author clicked away from must
# not have its focus stolen back sixty times a second. So after ten seconds the
# only thing that re-takes the keyboard is a click into the player's own window
# -- and this script never made one.
#
# Measured, in this audit's own run: `focus LOST at step 3658` (t = 61 s), the
# foreground handed to the EDITOR at the RIGHT-CLICK the turn-in-place leg makes,
# and `focus GAINED` again only at step 10974 (t = 183 s). Everything between --
# the kerb, the breath pair, the slide, the prone set, and the mantle at the
# placement -- was driven into a window that was not listening, and the frames
# named for them show an idle hero on the road. That is the true cause of the
# frames CHAR1b.2 carried as items 124 and 132, and it is not the crouch tap.
#
# So: raise the player again before every leg that follows a mouse operation.
# It is idempotent, it costs 200 ms, and the log says whether it was needed.
function Restore-PlayerFocus([string]$why) {
    if ($null -eq $player) { return }
    $player.Refresh()
    $before = [InfInput]::Foreground()
    if ($hasWindow -and $player.MainWindowHandle -ne [IntPtr]::Zero) {
        [InfInput]::ShowWindow($player.MainWindowHandle, 5) | Out-Null
        [InfInput]::SetForegroundWindow($player.MainWindowHandle) | Out-Null
        # **AND PUT THE CURSOR BACK INSIDE IT.** A synthetic click goes to
        # whatever window is under the POINTER, and `Look` moves the pointer by
        # relative deltas -- so after a look sweep the cursor is wherever it
        # drifted to, which on this machine is the editor. That is how the
        # turn-in-place leg's right-click opened the EDITOR's actor context menu
        # (photographed: run 1's `24-turn-in-place.png` is the editor with
        # "Open in Editor / Edit Mesh / ... / Delete" on screen) and handed it
        # the foreground for the rest of the session.
        $r2 = New-Object InfInput+RECT
        if ([InfInput]::GetWindowRect($player.MainWindowHandle, [ref]$r2)) {
            [InfInput]::SetCursorPos([int](($r2.Left + $r2.Right) / 2), [int](($r2.Top + $r2.Bottom) / 2)) | Out-Null
        }
    } else {
        [InfInput]::Click([int]($screen.Width / 2), [int]($screen.Height / 2))
    }
    Start-Sleep -Milliseconds 200
    $after = [InfInput]::Foreground()
    if ($before -ne $after) { Say "REFOCUS ($why): $before -> $after" }
}

Say ("foreground after the click:  " + [InfInput]::Foreground())
Say ("cursor while the game has the window: " + [InfInput]::CursorState())

# **FOUR NAMED FRAMES, not two anonymous ones** (wave CHAR1a.2). A wave that is
# asked whether the idle looks right cannot answer with a picture of a run. The
# order is idle → walk → run because it is also the order the locomotion machine
# transitions in, so a bad transition shows as a frame that does not match its
# name.
Say "IDLE: no input for 2 s"
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "04-pie-idle.png") | ForEach-Object { Say $_ }

Say "WALK: W tapped in 120 ms bursts, so the machine stays under the run threshold"
for ($i = 0; $i -lt 6; $i++) {
    [InfInput]::Down(0x11); Start-Sleep -Milliseconds 120; [InfInput]::Up(0x11)
    Start-Sleep -Milliseconds 260
}
[InfInput]::Down(0x11)
Start-Sleep -Milliseconds 350
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "05-pie-walk.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x11)

Say "holding W"
[InfInput]::Down(0x11)   # scancode: W
Start-Sleep -Milliseconds 900
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "02-pie-a.png") | ForEach-Object { Say $_ }
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "03-pie-b.png") | ForEach-Object { Say $_ }
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "06-pie-run.png") | ForEach-Object { Say $_ }
Say ("cursor two seconds in: " + [InfInput]::CursorState())
[InfInput]::Up(0x11)
Say "released W"

# One more after the release: the street the hero ran into, where the crowd is,
# which is where a second body shows if the level offers one.
Start-Sleep -Seconds 2
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "07-pie-street.png") | ForEach-Object { Say $_ }

# ── wave CHAR1b.1: the pose ──────────────────────────────────────────────────
#
# Everything below is a frame of a MECHANISM this wave turned on, and each one is
# named for the mechanism rather than for the moment: a reader asked "does the
# head follow the mouse" should not have to guess which of four running frames to
# look at. `hero.csv` carries the numbers (aim yaw, head yaw, head pitch, the
# machine state) at 4 Hz throughout, so every frame here has a row beside it.

Say "LOOK LEFT: the mouse, 900 counts to the left"
for ($i = 0; $i -lt 30; $i++) { [InfInput]::Look(-30, 0); Start-Sleep -Milliseconds 16 }
Start-Sleep -Milliseconds 600
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "11-look-left.png") | ForEach-Object { Say $_ }

Say "LOOK RIGHT: back across, 1800 counts"
for ($i = 0; $i -lt 60; $i++) { [InfInput]::Look(30, 0); Start-Sleep -Milliseconds 16 }
Start-Sleep -Milliseconds 600
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "12-look-right.png") | ForEach-Object { Say $_ }

Say "LOOK UP: back to centre, then the mouse up"
for ($i = 0; $i -lt 30; $i++) { [InfInput]::Look(-30, 0); Start-Sleep -Milliseconds 16 }
for ($i = 0; $i -lt 20; $i++) { [InfInput]::Look(0, -20); Start-Sleep -Milliseconds 16 }
Start-Sleep -Milliseconds 600
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "13-look-up.png") | ForEach-Object { Say $_ }
for ($i = 0; $i -lt 20; $i++) { [InfInput]::Look(0, 20); Start-Sleep -Milliseconds 16 }

Restore-PlayerFocus "before the crouch"
Say "CROUCH: C tapped (a click crouches; a hold goes prone)"
[InfInput]::Down(0x2E); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x2E)
Start-Sleep -Milliseconds 900
# **TRIGGERED, NOT SLEPT** (CHAR1b.2 audit, carried 132 and 124). Measured over
# five audit sessions: every frame `Wait-ForHero` took landed on the state it is
# named for, and every frame a `Start-Sleep` took did not -- `14-crouch.png` read
# `idle` in four sessions out of five while the crouch states were 69 rows of the
# same log. The crouch click also fires on RELEASE and is swallowed outright
# about one session in three, so the tap is RETRIED here rather than trusted.
$gotCrouch = Wait-ForHero -Csv $heroCsv -What "a crouch" -TimeoutS 2.5 `
    -Predicate { param($c) $c[5] -eq "Crouch" } `
    -Out (Join-Path $OutDir "14-crouch.png")
if (-not $gotCrouch) {
    Say "the crouch tap was swallowed; tapping C again"
    [InfInput]::Down(0x2E); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x2E)
    $gotCrouch = Wait-ForHero -Csv $heroCsv -What "a crouch (2nd tap)" -TimeoutS 3.0 `
        -Predicate { param($c) $c[5] -eq "Crouch" } `
        -Out (Join-Path $OutDir "14-crouch.png")
}
Say "CROUCH-WALK: W held while crouched"
[InfInput]::Down(0x11); Start-Sleep -Milliseconds 900
Wait-ForHero -Csv $heroCsv -What "a crouch-walk" -TimeoutS 3.0 `
    -Predicate { param($c) $c[11] -eq "crouch_walk" } `
    -Out (Join-Path $OutDir "15-crouch-walk.png") | Out-Null
[InfInput]::Up(0x11)
[InfInput]::Down(0x2E); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x2E)
Start-Sleep -Milliseconds 700

Say "JUMP: Space, and a frame while the feet are off the ground"
[InfInput]::Down(0x39); Start-Sleep -Milliseconds 60; [InfInput]::Up(0x39)
Start-Sleep -Milliseconds 260
Wait-ForHero -Csv $heroCsv -What "the jump" -TimeoutS 2.5 `
    -Predicate { param($c) $c[11] -eq "jump" } `
    -Out (Join-Path $OutDir "16-jump.png") | Out-Null
Start-Sleep -Milliseconds 900
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "17-land.png") | ForEach-Object { Say $_ }

Say "SPRINT: Shift + W, the third rung of the gait ladder"
[InfInput]::Down(0x2A); [InfInput]::Down(0x11)
Start-Sleep -Milliseconds 1800
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "18-sprint.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x11); [InfInput]::Up(0x2A)
Start-Sleep -Milliseconds 1200
Wait-ForHero -Csv $heroCsv -What "a stop clip" -TimeoutS 3.0 `
    -Predicate { param($c) $c[11] -match "^stop_" } `
    -Out (Join-Path $OutDir "19-stop.png") | Out-Null

Say "STRAFE: A and D, the direction blend's own axis"
[InfInput]::Down(0x1E); Start-Sleep -Milliseconds 1100
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "20-strafe-left.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x1E)
[InfInput]::Down(0x20); Start-Sleep -Milliseconds 1100
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "21-strafe-right.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x20)
Start-Sleep -Milliseconds 800
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "22-back-to-idle.png") | ForEach-Object { Say $_ }

# ── audit CHAR1b.1: the two frames the wave could not get ────────────────────
#
# A TURN IN PLACE needs `RotationMode::LookingDirection`, and a level starts in
# `VelocityDirection` -- where ALS does not turn in place either. The only door
# is releasing the aim key, so the tap below is load-bearing rather than
# decorative: without it `turn_deg` stays 0.000 for the whole session and the
# eight turn clips cannot play whatever the graph says.
# **STAND, and say so.** The crouch is a TOGGLE and its click fires on RELEASE,
# so a tap that lands while the previous one is still settling is swallowed and
# every leg after it runs crouched — which is what the first audit run filmed
# (313 of 488 log rows in `Crouch`). Tapped again here with room around it.
Say "STAND: C again, with room around it, before the legs that must not be crouched"
[InfInput]::Down(0x2E); Start-Sleep -Milliseconds 120; [InfInput]::Up(0x2E)
Start-Sleep -Milliseconds 1400

# **Q, NOT A RIGHT-CLICK** (wave CHAR1c, carried 123). This leg used to press
# and release AIM, because releasing aim was the only door into
# `LookingDirection` this engine had -- and the release PROMOTED the character
# into it, so the mode was reachable and not leavable. CHAR1c gave the release
# ALS's own behaviour (back to the DESIRED mode) and gave the player a key that
# sets it: Q on a keyboard, the right stick's click on a pad. The right-click is
# also what handed the EDITOR the foreground in two earlier sessions, which is
# carried 145's whole story, so this leg no longer makes one at all.
Restore-PlayerFocus "before the turn in place"
Say "TURN IN PLACE: Q for looking-direction (no aim press), then swing"
[InfInput]::Down(0x10); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x10)   # scancode: Q
Start-Sleep -Milliseconds 400
for ($i = 0; $i -lt 24; $i++) { [InfInput]::Look(28, 0); Start-Sleep -Milliseconds 16 }
Start-Sleep -Milliseconds 700
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "24-turn-in-place.png") | ForEach-Object { Say $_ }
Start-Sleep -Milliseconds 900
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "25-turned.png") | ForEach-Object { Say $_ }

# THE KERB. The loop drives W down the middle of a street; the pavement is to
# one side and its kerb is the 15 cm step the fixture measures in the abstract.
# Turning ninety degrees and walking is the scripted input carried item 117 asked
# for, and `hero.csv`'s Y column is what says whether the hero went UP.
Restore-PlayerFocus "before the kerb"
Say "KERB: turn toward the pavement and RUN onto it"
for ($i = 0; $i -lt 20; $i++) { [InfInput]::Look(-30, 0); Start-Sleep -Milliseconds 16 }
Start-Sleep -Milliseconds 400
[InfInput]::Down(0x11); Start-Sleep -Milliseconds 3200
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "26-toward-the-kerb.png") | ForEach-Object { Say $_ }
Start-Sleep -Milliseconds 3200; [InfInput]::Up(0x11)
Start-Sleep -Milliseconds 1600
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "27-on-the-kerb.png") | ForEach-Object { Say $_ }
Start-Sleep -Milliseconds 1200
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "28-standing-on-it.png") | ForEach-Object { Say $_ }

# ── wave CHAR1b.2: the moves ─────────────────────────────────────────────────
#
# Everything below is a mechanism this slice turned on that a player can reach
# from where the level puts the hero. The ones a player CANNOT reach from there
# -- the mantle course, the water, a measured drop -- are named in the wave's
# report with the numbers the gate measures them at instead, because the loop
# drives the shipped player and the shipped player has no teleport.

Restore-PlayerFocus "before the breath"
Say "BREATH: two idle frames 1.3 s apart, which is the pair the additive shows in"
Start-Sleep -Milliseconds 1500
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "30-breath-a.png") | ForEach-Object { Say $_ }
Start-Sleep -Milliseconds 1300
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "31-breath-b.png") | ForEach-Object { Say $_ }

# THE SLIDE. `movement.rs`'s own rule: sprinting, past `slide_entry_speed_mps`,
# and then the crouch key. Sprint first for long enough to be over the entry
# speed -- a crouch tap below it is a REFUSAL (`ConditionNotMet`) and a stance
# toggle, which is the frame this would otherwise film and mis-caption.
Restore-PlayerFocus "before the slide"
Say "SLIDE: Shift+W up to speed, then C while still sprinting"
[InfInput]::Down(0x2A); [InfInput]::Down(0x11)
Start-Sleep -Milliseconds 2600
[InfInput]::Down(0x2E); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x2E)
Start-Sleep -Milliseconds 260
if (Wait-ForHero -Csv $heroCsv -What "the slide" -TimeoutS 3.0 `
        -Predicate { param($c) $c[5] -eq "Slide" } `
        -Out (Join-Path $OutDir "32-slide.png")) {
    Wait-ForHero -Csv $heroCsv -What "the slide, later in it" -TimeoutS 1.5 `
        -Predicate { param($c) $c[5] -eq "Slide" } `
        -Out (Join-Path $OutDir "33-sliding.png") | Out-Null
}
[InfInput]::Up(0x11); [InfInput]::Up(0x2A)
Start-Sleep -Milliseconds 1400

# PRONE. A LONG crouch press, which is `MovementIntent::from_actions`' own
# classification of the same key -- and the reason the crouch TAP is a coin toss
# (carried 124): the two are the same button and only the duration tells them
# apart.
Restore-PlayerFocus "before prone"
Say "PRONE: C held past the long-press threshold"
[InfInput]::Down(0x2E); Start-Sleep -Milliseconds 700; [InfInput]::Up(0x2E)
Start-Sleep -Milliseconds 1200
Wait-ForHero -Csv $heroCsv -What "prone" -TimeoutS 4.0 `
    -Predicate { param($c) $c[5] -eq "Prone" } `
    -Out (Join-Path $OutDir "34-prone.png") | Out-Null
Say "PRONE CRAWL: W while prone"
[InfInput]::Down(0x11); Start-Sleep -Milliseconds 1400
Wait-ForHero -Csv $heroCsv -What "the prone crawl" -TimeoutS 3.0 `
    -Predicate { param($c) $c[11] -eq "prone_crawl" } `
    -Out (Join-Path $OutDir "35-prone-crawl.png") | Out-Null
[InfInput]::Up(0x11)
Say "UP: another long press leaves prone"
[InfInput]::Down(0x2E); Start-Sleep -Milliseconds 700; [InfInput]::Up(0x2E)
Start-Sleep -Milliseconds 1400
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "36-back-up.png") | ForEach-Object { Say $_ }

# ── 5c. THE CAMERA (wave CHAR1c) ─────────────────────────────────────────────
#
# The wave's own frames, and every one of them is TRIGGERED on `hero.csv` rather
# than slept for. The log carries four new columns since this wave — the boom's
# length, how much of the hero's body is being drawn, the whisker steer and who
# is holding the camera — so a frame captioned "the boom at its authored length"
# has the number beside it instead of an adjective.
#
# The order is the order a reader wants: the framing at rest FIRST (carried 89 is
# a close-up of the back of the hero's neck, photographed four times, and this is
# the same spot photographed again), then the camera against things.
# **THE VEHICLE, FIRST, AND IT HUNTS.** `interact` reaches a car within a few
# metres, and by the end of this leg the hero is wherever the look sweeps left
# it. The first run of this leg pressed E once into an empty street; this one
# walks and taps E for up to fifteen seconds and triggers on the mode the world
# reports. It is allowed to miss and to say so.
# **STAND UP FIRST** (the audit's fix to the instrument). The prone leg above
# leaves the hero CROUCHED and not standing: measured over a whole audit session,
# `mode` was `Crouch` from t = 108.6 s to t = 149 s, straight through the camera
# leg and into the placements, so the "at rest" frame was a crouched hero on the
# crouch block's 2.5 m arm, the stairwell placement landed in `Prone`, and the
# clipped-boom, near-fade, steered-boom and vehicle triggers all missed. A stance
# the loop did not intend is an instrument reading the wrong thing.
function Stand-Up([string]$why) {
    Say "STAND UP ($why)"
    for ($k = 0; $k -lt 4; $k++) {
        $standing = Wait-ForHero -Csv $heroCsv -What "a standing hero ($why)" -TimeoutS 1.5 `
            -Predicate { param($c) ($c[5] -eq "Grounded") -and ($c[11] -notmatch "^(crouch|prone|slide)") }
        if ($standing) { return $true }
        [InfInput]::Down(0x2E); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x2E)   # scancode: C
        Start-Sleep -Milliseconds 500
    }
    Say "STILL NOT STANDING after four taps of C -- the frames below are of whatever stance the world is in"
    return $false
}

Restore-PlayerFocus "before the vehicle"
Stand-Up "before the camera leg" | Out-Null
Say "CAMERA: E while walking — hunting for a car, then the drive camera blends in"
$gotCar = $false
# **STAND STILL FOR THE FIRST FIVE TAPS.** A session driven here by `-SpawnAt`
# has been PUT beside a car, and the walking half of this hunt used to carry it
# straight past one: measured, sixteen metres of walking over fifteen iterations,
# and the run that was placed at a car reached its "at rest" frame fifteen metres
# away from it. So the hunt presses first and walks second.
for ($k = 0; $k -lt 25 -and -not $gotCar; $k++) {
    [InfInput]::Down(0x12); Start-Sleep -Milliseconds 70; [InfInput]::Up(0x12)   # scancode: E
    $gotCar = Wait-ForHero -Csv $heroCsv -What "the drive camera blending" -TimeoutS 0.9 `
        -Predicate { param($c) ($c.Count -gt 13) -and ($c[5] -eq "Driving") -and ([double]$c[13] -gt 4.0) } `
        -Out (Join-Path $OutDir "69-camera-vehicle-blend.png")
    if (-not $gotCar -and $k -ge 5) {
        [InfInput]::Down(0x11); Start-Sleep -Milliseconds 300; [InfInput]::Up(0x11)
    }
}
if ($gotCar) {
    Wait-ForHero -Csv $heroCsv -What "the drive camera settled" -TimeoutS 4.0 `
        -Predicate { param($c) ($c.Count -gt 13) -and ($c[5] -eq "Driving") -and ([double]$c[13] -gt 5.5) } `
        -Out (Join-Path $OutDir "70-camera-vehicle-settled.png") | Out-Null
    Say "CAMERA: E again — out of the car"
    [InfInput]::Down(0x12); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x12)
    Start-Sleep -Milliseconds 1200
} else {
    Say "NO CAR reached in fifteen taps of E — the drive-camera frames are not in this session"
}

Restore-PlayerFocus "before the camera leg"
Say "CAMERA: the framing at rest — the boom at its authored length"
Start-Sleep -Seconds 2
# **THE STANCE IS PART OF THE CLAIM** (the audit's fix). `boom > 2.5` is
# satisfied by the CROUCH block's 2.5 m arm, so this fired on a crouched hero at
# 2.5402 m and the frame captioned "the boom at its authored length" was of a
# different block entirely. The walk block is 3.0 m; the predicate asks for a
# standing, idle hero above 2.9 and says so.
Wait-ForHero -Csv $heroCsv -What "the hero at rest with a full boom" -TimeoutS 8.0 `
    -Predicate { param($c) ($c.Count -gt 13) -and ($c[5] -eq "Grounded") -and ($c[11] -eq "idle") -and ([double]$c[13] -gt 2.9) -and ([double]$c[7] -lt 0.2) } `
    -Out (Join-Path $OutDir "60-camera-at-rest.png") | Out-Null

# **AGAINST A WALL.** Walk backwards into whatever is behind the hero and hold
# it there: the trigger is the BOOM, not a place, so it fires wherever the street
# actually has a wall.
Say "CAMERA: backing into a wall — the boom clips and the body fades"
[InfInput]::Down(0x1F)   # scancode: S
$gotWall = Wait-ForHero -Csv $heroCsv -What "a clipped boom" -TimeoutS 8.0 `
    -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[7] -gt 0.8) } `
    -Out (Join-Path $OutDir "61-camera-against-a-wall.png")
if ($gotWall) {
    # …and again, shorter, where the near fade has actually engaged. It is
    # allowed to miss and to say so: a street with nothing tall behind it never
    # takes the boom below the fade band.
    Wait-ForHero -Csv $heroCsv -What "the near fade engaged" -TimeoutS 4.0 `
        -Predicate { param($c) ($c.Count -gt 14) -and ([double]$c[14] -lt 0.999) } `
        -Out (Join-Path $OutDir "62-camera-near-fade.png") | Out-Null
}
[InfInput]::Up(0x1F)
Start-Sleep -Milliseconds 600

# **THE WHISKER FAN**, photographed by its own number: a frame taken while the
# steer is non-zero is a frame of the camera moving out of the way of something
# it has not hit.
Say "CAMERA: a steered boom — the fan seeing a wall the boom has not reached"
[InfInput]::Down(0x11)
for ($i = 0; $i -lt 40; $i++) { [InfInput]::Look(-24, 0); Start-Sleep -Milliseconds 16 }
Wait-ForHero -Csv $heroCsv -What "a steered boom" -TimeoutS 6.0 `
    -Predicate { param($c) ($c.Count -gt 15) -and ([math]::Abs([double]$c[15]) -gt 1.0) } `
    -Out (Join-Path $OutDir "63-camera-whisker-steer.png") | Out-Null
[InfInput]::Up(0x11)

# **AIMING** — the shoulder block: arm 3.0 → 2.0 m, fov 70 → 55, the offset 0.45
# → 0.55. The trigger is the boom coming inside the aim block's own length while
# the hero is in `Aiming`.
Restore-PlayerFocus "before the aim"
Say "CAMERA: aiming — the shoulder block"
[InfInput]::RightDown()
Wait-ForHero -Csv $heroCsv -What "the aim camera" -TimeoutS 5.0 `
    -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[13] -lt 2.4) -and ([double]$c[13] -gt 0.5) } `
    -Out (Join-Path $OutDir "64-camera-aiming.png") | Out-Null
[InfInput]::RightUp()
Start-Sleep -Milliseconds 500

# **THE ROTATION-MODE KEY** (carried 123's door): Q reaches looking-direction
# without an aim press, and a turn in place follows from it.
Restore-PlayerFocus "before the rotation-mode key"
# **The key CYCLES**, so this press is the second of the session and it puts the
# character back in velocity-direction; the third, below the sweep, returns it to
# looking-direction for the frame. A leg that pressed once and assumed the mode
# would be a leg that photographed the other one.
Say "CAMERA: Q — the rotation-mode key cycles, and the frame is taken on the second press"
[InfInput]::Down(0x10); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x10)   # scancode: Q
Start-Sleep -Milliseconds 250
[InfInput]::Down(0x10); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x10)
Start-Sleep -Milliseconds 400
for ($i = 0; $i -lt 34; $i++) { [InfInput]::Look(28, 0); Start-Sleep -Milliseconds 16 }
Start-Sleep -Milliseconds 1600
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "65-camera-turn-in-place.png") | ForEach-Object { Say $_ }

# **FIRST PERSON** (G): the seat, and the body drawn out of the way by the near
# fade rather than by a visibility flag.
Restore-PlayerFocus "before first person"
Say "CAMERA: G — the first-person seat"
[InfInput]::Down(0x22); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x22)   # scancode: G
Wait-ForHero -Csv $heroCsv -What "the first-person seat" -TimeoutS 5.0 `
    -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[13] -lt 0.4) } `
    -Out (Join-Path $OutDir "66-camera-first-person.png") | Out-Null
[InfInput]::Down(0x11); Start-Sleep -Milliseconds 900
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "67-camera-first-person-walk.png") | ForEach-Object { Say $_ }
[InfInput]::Up(0x11)
Say "CAMERA: G again — back to third person"
[InfInput]::Down(0x22); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x22)
Wait-ForHero -Csv $heroCsv -What "third person again" -TimeoutS 5.0 `
    -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[13] -gt 2.0) } `
    -Out (Join-Path $OutDir "68-camera-third-person-again.png") | Out-Null

# **A VEHICLE.** E is the interact key; the drive camera's boom is more than
# twice the walk's, so the trigger is the boom itself and it fires on the blend
# rather than at its end.
# ── 5b. the placements, and the frames the wave could not take ───────────────
#
# Only when `-SpawnAt` was given. Each waypoint is applied by the PLAYER at its
# own `@seconds`; this side waits for the hero to BE there (the log's own x/z)
# rather than sleeping and hoping, which is carried item 132's remedy.
if ($SpawnAt -ne "") {
    Restore-PlayerFocus "before the placements"
Say "PLACEMENTS: waiting for the player to apply $SpawnAt"
    # 1. THE MANTLE. Hold W and tap Space: `try_mantle` runs on a jump press with
    #    movement input, which is ALS's own trigger.
    if (Wait-ForHero -Csv $heroCsv -What "at the ledge" -TimeoutS 150 `
            -Predicate { param($c) ([math]::Abs([double]$c[2] + 1766.0) -lt 12.0) -and ([math]::Abs([double]$c[4] - 1992.0) -lt 12.0) } `
            -Out (Join-Path $OutDir "40-at-the-ledge.png")) {
        # **No stance key here.** Measured: a `C` press before the drive cost
        # 1.6 s of the ledge's own window and left the character CROUCHED, and
        # the run that did it reached no mantle at all while the run that did
        # not reached eight `mantle_low` samples.
        [InfInput]::Down(0x11)
        # **THE SHOT IS TRIGGERED BY THE MANTLE, NOT BY A SLEEP** (carried 132).
        # A mantle is 0.586 s and the loop's own sleeps put run 2's frame 1.3 s
        # past the end of one -- the state column is what knows.
        $mantled = $false
        for ($k = 0; $k -lt 10 -and -not $mantled; $k++) {
            [InfInput]::Down(0x39); Start-Sleep -Milliseconds 60; [InfInput]::Up(0x39)
            $mantled = Wait-ForHero -Csv $heroCsv -What "a mantle" -TimeoutS 1.2 `
                -Predicate { param($c) $c[11] -match "^mantle" } `
                -Out (Join-Path $OutDir "41-mantling.png")
        }
        if (-not $mantled) { Say "NO MANTLE reached from the ledge placement in ten jump taps" }
        Start-Sleep -Milliseconds 700
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "42-mantled.png") | ForEach-Object { Say $_ }
        [InfInput]::Up(0x11)
        # ── THE INTERIOR CAMERA FRAMES (wave CHAR1c) ──
        #
        # This placement is the only INTERIOR geometry on the island a scripted
        # session can reach — the stair the mantle uses (CHAR1b.2 audit, carried
        # 139) — and it is where the camera has the most to do: a stairwell has a
        # wall behind the boom whichever way the hero faces. So the wall, the
        # near fade and the room frames are taken HERE rather than out on the
        # street, where the first run of this leg backed up for eight seconds and
        # found nothing at all.
        Restore-PlayerFocus "before the interior camera frames"
        Stand-Up "before the interior camera frames" | Out-Null
        Say "CAMERA (interior): backing into a stairwell wall"
        for ($i = 0; $i -lt 24; $i++) { [InfInput]::Look(30, 0); Start-Sleep -Milliseconds 16 }
        # **BACK UP FIRST, and wait DURING it.** The first cut waited for the two
        # triggers standing still and then backed up: the clip fired (a stairwell
        # has a wall behind the boom whichever way the hero faces) and the near
        # fade did not, because a boom at 1.03 m is still outside the 0.90 m band
        # and nothing was walking it in. The fade needs the hero to keep moving
        # toward the thing behind it, which is what S does.
        [InfInput]::Down(0x1F)   # scancode: S
        Wait-ForHero -Csv $heroCsv -What "a clipped boom indoors" -TimeoutS 6.0 `
            -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[7] -gt 0.8) } `
            -Out (Join-Path $OutDir "72-camera-interior-wall.png") | Out-Null
        Wait-ForHero -Csv $heroCsv -What "the near fade engaged indoors" -TimeoutS 8.0 `
            -Predicate { param($c) ($c.Count -gt 14) -and ([double]$c[14] -lt 0.999) } `
            -Out (Join-Path $OutDir "73-camera-interior-near-fade.png") | Out-Null
        Start-Sleep -Milliseconds 600
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "74-camera-interior-backed-up.png") | ForEach-Object { Say $_ }
        [InfInput]::Up(0x1F)
        # …and the cape, on the same hero, two frames apart while it walks.
        [InfInput]::Down(0x11); Start-Sleep -Milliseconds 900
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "43-cape-a.png") | ForEach-Object { Say $_ }
        Start-Sleep -Milliseconds 700
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "44-cape-b.png") | ForEach-Object { Say $_ }
        [InfInput]::Up(0x11)
    }
    # 2. A MEASURED DROP. The player puts the hero above the road; the frame that
    #    matters is the one where the machine is in a landing, so it is triggered
    #    on the state column and not on a sleep.
    Wait-ForHero -Csv $heroCsv -What "falling" -TimeoutS 60 `
        -Predicate { param($c) $c[11] -match "^fall" } `
        -Out (Join-Path $OutDir "45-falling.png") | Out-Null
    # A landing is one sample of a 4 Hz log, so this is a wide net and it is
    # allowed to miss -- and to say so when it does.
    Wait-ForHero -Csv $heroCsv -What "a landing" -TimeoutS 22 `
        -Predicate { param($c) $c[11] -match "^(land_|roll)" } `
        -Out (Join-Path $OutDir "46-landing.png") | Out-Null
    # 3. THE RAGDOLL DROP, and the get-up after it.
    Wait-ForHero -Csv $heroCsv -What "a ragdoll" -TimeoutS 60 `
        -Predicate { param($c) $c[11] -eq "ragdoll" } `
        -Out (Join-Path $OutDir "47-ragdoll.png") | Out-Null
    # **THE RAGDOLL FOLLOW** (wave CHAR1c, the director's `Override` layer). The
    # gameplay rig keeps framing the parked capsule while seventeen bodies fall
    # down the hill in its place, so the death happens off screen; the override
    # frames the pelvis the bodies actually ended at. Column 16 says who is
    # holding the camera, which is what makes this a frame OF the override rather
    # than a frame taken while one happened to be running.
    Wait-ForHero -Csv $heroCsv -What "the death cam holding the camera" -TimeoutS 20 `
        -Predicate { param($c) ($c.Count -gt 16) -and ($c[16].Trim() -eq "override") } `
        -Out (Join-Path $OutDir "71-camera-ragdoll-follow.png") | Out-Null
    Wait-ForHero -Csv $heroCsv -What "a get-up" -TimeoutS 20 `
        -Predicate { param($c) $c[11] -match "^getup" } `
        -Out (Join-Path $OutDir "48-getup.png") | Out-Null
    # 4. THE WATER. The island's own ocean; the trigger is the swim mode itself.
    Wait-ForHero -Csv $heroCsv -What "in the water" -TimeoutS 70 `
        -Predicate { param($c) $c[5] -match "Swim" } `
        -Out (Join-Path $OutDir "49-swimming.png") | Out-Null
    [InfInput]::Down(0x11); Start-Sleep -Milliseconds 1500
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "50-swim-forward.png") | ForEach-Object { Say $_ }
    [InfInput]::Up(0x11)
}

# ── 6. what the hero did, in metres ──────────────────────────────────────────
if (Test-Path $heroCsv) {
    $rows = Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" }
    if ($rows.Count -ge 2) {
        $a = $rows[0].Split(","); $b = $rows[-1].Split(",")
        # **PATH LENGTH, NOT DISPLACEMENT, AND NOT THE PLACEMENTS** (CHAR1b.2
        # audit). `-SpawnAt` moves the hero kilometres in one sample, so the
        # first-to-last displacement this used to print reads 2 695 m for a
        # session in which the character walked forty. Sum the per-sample steps
        # and drop any single step over 50 m, which is a placement and not a walk
        # -- there is no gait in this engine that covers 50 m in a quarter of a
        # second.
        $d = 0.0
        $ported = 0
        for ($ri = 1; $ri -lt $rows.Count; $ri++) {
            $p0 = $rows[$ri - 1].Split(","); $p1 = $rows[$ri].Split(",")
            $sx = [double]$p1[2] - [double]$p0[2]
            $sz = [double]$p1[4] - [double]$p0[4]
            $step = [math]::Sqrt($sx * $sx + $sz * $sz)
            if ($step -gt 50.0) { $ported++ } else { $d += $step }
        }
        if ($ported -gt 0) { Say "$ported placement jump(s) excluded from the distance" }
        Say ("hero first : t={0} ({1}, {2}, {3}) {4} speed {5}" -f $a[0], $a[2], $a[3], $a[4], $a[5], $a[6])
        Say ("hero last  : t={0} ({1}, {2}, {3}) {4} speed {5}" -f $b[0], $b[2], $b[3], $b[4], $b[5], $b[6])
        Say ("HERO MOVED {0:N3} m over {1} samples" -f $d, $rows.Count)
        if ($d -lt $MinMetres) {
            Say ("PLAY DID NOT PLAY: the hero moved {0:N3} m against a {1:N1} m floor" -f $d, $MinMetres)
            $failed = $true
        }
    } else {
        Say "hero.csv has $($rows.Count) row(s) — the player wrote no positions"
        $failed = $true
    }
    # What the player said about the keyboard, echoed where the number is, so a
    # session that moved is read beside the reason it could.
    foreach ($line in (Get-Content $heroCsv | Where-Object { $_ -match "^# keyboard focus" })) {
        Say ("player: " + $line.Substring(2))
    }
} else {
    Say "no hero.csv at $heroCsv"
    $failed = $true
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
    # The control for the two readings above: a cursor that is hidden here as
    # well is a cursor this script cannot see, not one the game took.
    Say ("cursor after the session ended: " + [InfInput]::CursorState())
    Say ("still running: " + $(if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in @("inf-studio", "inf-player") }) { "YES" } else { "none" }))
}
if ($failed) {
    # **Non-zero, and that is the point** (audit FIX1). A demo loop that always
    # exits 0 is a screenshot service. This one is the last gate before a wave
    # is called done, so it fails the way a gate fails.
    Say "done (FAILED) — $OutDir"
    exit 7
}
Say "done — $OutDir"
