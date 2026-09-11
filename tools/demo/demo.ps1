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
    # **THE HEADING THAT GOES WITH THE PLACEMENT** (carried 186, closed by the
    # COV1 audit). A `;`-separated list of yaw degrees, one per `-SpawnAt`
    # entry, folded into the same `INF_PIE_SPAWN_AT` string as an `/yaw` suffix
    # — one env door, two switches.
    #
    # Why it had to exist: a placement set a POSITION and nothing else, so a
    # scripted leg reached wherever the hero happened to be looking, and a
    # character standing still in `VelocityDirection` does not turn its body
    # under the mouse. The COV1 loop's cover leg swept SIXTEEN presses through
    # 360 degrees and held `W` to face a wall four metres in front of it. With
    # a heading the leg presses once.
    [string]$FaceAt = "",
    # A `.inf_cloth` GUID to put on the hero for the session, `INF_PIE_WEAR_CLOTH`.
    # The cape wave CHAR1b.2 authored lives in the island's Content and is worn
    # in the gate; carried 137 is that it is not in the committed level, and this
    # is how the loop photographs it without making that edit.
    [string]$WearCloth = "",
    # **A REGISTRY WEAPON TO PUT IN THE HERO'S HANDS** (wave WPN2a),
    # `INF_PIE_ARM_HERO`. A `;`-separated list of ids from
    # `crates/inf-ecs/src/weapons.toml` -- one per class is what the wave's own
    # session photographs. The player merges the registry through the same
    # `ItemDefs::merge_toml` the `item.define` node calls, gives the row and
    # equips it, in a `--pie` preview only.
    #
    # It exists for `-SpawnAt`'s reason exactly: the island has no Blueprint of
    # its own to hang an `item.define` on -- the only class on it is the wizard's
    # committed character controller, shared by every character the New Character
    # wizard has ever made -- so a weapon catalogue in there would arm all of
    # them. The phase30 fixture is where the registry reaches a LEVEL.
    [string]$ArmHero = "",
    # **HOW LONG THE PLAYER HOLDS EACH ONE**, seconds (carried 209). Any value
    # above zero turns `-ArmHero` into a ROTATION: the player equips each id in
    # turn through `equip_weapon` -- the ECS door -- and wraps for ever, so this
    # leg never touches the scroll wheel again. `weapon_switch` is a RATE (a
    # 120-count notch is divided by the frame time and the movement step cycles
    # one slot per step while the sign is non-zero), so one notch is not one
    # slot: wave WPN2a spun up to twenty-four notches per class, reversed half
    # way, and STILL had a session that never reached `remington_870`.
    #
    # Fourteen seconds is a measurement, not a guess: the per-weapon work here
    # is one HUD frame plus up to four trigger presses at 250 ms down and 1.6 s
    # of waiting, which is about ten seconds, and the leg has to finish inside
    # the dwell or it photographs the next weapon's magazine.
    [double]$ArmDwellS = 14.0,
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
    [bool]$Portrait = $true,
    # **THE LOOP'S OWN GATE** (WPN2e audit, the cheap half of carried 281).
    #
    # Two thousand three hundred lines of PowerShell that nothing type-checks,
    # nothing lints and no test runs -- and wave WPN2e shipped three faults into
    # it while the battery stayed green through two whole sessions. What would
    # have caught them is not a linter: `Parser::ParseFile` finds syntax errors
    # and found none of these. It is running the leg's own logic over a RECORDED
    # `hero.csv` with no editor at all.
    #
    # That is this. Point it at a session directory (or a `hero.csv`) from any
    # previous run; it exercises `Wait-ForHero` both ways round, the shootout
    # leg's own row-splitting pipeline, and every column index the legs read,
    # and it exits non-zero if any of them answers the way the three faults
    # answered. It launches nothing and takes about a second.
    #
    # The half it is NOT: it does not drive the input synthesiser, does not
    # press Play, and cannot see a leg whose sleeps are too short. Those need a
    # recorded INPUT trace as well, which is a wave.
    [string]$DryRun = "",
    # **RETUNE EVERY VEHICLE IN THE LEVEL FOR THE SESSION** (VEH3a's audit),
    # `INF_PIE_TUNE_VEHICLE`. A `;`-separated `name=value` list, sent through
    # `VehicleClass::set` -- the same by-name door an authored catalogue row
    # uses -- and written onto the chassis entity, so the physics bridge
    # installs it on its next sync. Nothing bypasses the model.
    #
    # It exists because wave VEH3a's burnout frame COULD NOT fire: a line-lock
    # burnout on asphalt under road tyres makes no slip at all in this model,
    # because the island car's brakes out-hold its engine (13 kN against 8).
    # `tyre_surface_set=3` is the slick row, and the gate measures a burnout on
    # it at 229 slipping steps.
    [string]$TuneVehicle = ""
)

$ErrorActionPreference = "Continue"
# **THE LOOP MAY NOT EXIT 0 WITH EXCEPTIONS IN ITS OWN LOG** (WPN2e audit,
# closing carried 282). Wave WPN2e's first relaunch printed an
# `InvalidOperation` in red in the middle of an otherwise clean session and
# still exited 0, because nothing ever read `$Error`. Cleared here so the count
# at the bottom is this run's, and read at the bottom beside `$failed`.
#
# Every `-ErrorAction Ignore` in this file became `Ignore` in the same
# commit: `SilentlyContinue` still RECORDS, so the expected misses (a process
# that is not running, an env var that is not set) would have made the count
# meaningless. `Ignore` suppresses and does not record, which is what those call
# sites always meant.
$Error.Clear()
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
# waits for a predicate over the rows and shoots when it holds, or says plainly
# that it never did.
#
# **IT SCANS EVERY ROW APPENDED SINCE ITS LAST LOOK** (VEH3b audit, closing that
# wave's carried item 3). It used to test `$rows[-1]` alone, four times a second,
# against a log written sixty times a second -- so it could only ever see one
# row in fifteen, and an event shorter than 250 ms was invisible to it however
# loudly the world reported it. Wave VEH3b lost three of its five frames to
# exactly that and said so: the limiter cut on **one** row of a 692-row drive,
# the turbo peaked at 0.481 for about two seconds of a 376-second session, and
# both were plainly in the telemetry plot afterwards.
#
# The baseline is the row count at ENTRY, never zero: the predicates are things
# like "the clutch is slipping", which were true at some point in every leg
# before this one, and scanning the whole file would fire on a row from four
# legs ago. What the frame then shows is the world a fraction of a second after
# the row that fired, which is honest and is why the age is printed: a trigger
# that matched a row 0.4 s old is a photograph of the moment after it.
function Wait-ForHero {
    param(
        [string]$Csv,
        [scriptblock]$Predicate,
        [string]$What,
        [double]$TimeoutS = 8.0,
        [string]$Out = "",
        # Where the scan starts. `-1` is "whatever is already in the file when
        # this call begins", which is what a live leg wants; `0` scans the whole
        # recording, which is what the dry run over a finished session wants and
        # is the only caller that passes it.
        [int]$FromRow = -1
    )
    $seen = 0
    if ($FromRow -ge 0) {
        $seen = $FromRow
    }
    elseif (Test-Path $Csv) {
        # **MINUS ONE**, so the newest existing row is still in the first scan.
        # It is the row the old poll tested, and a call with a 0.1-second
        # timeout (the recoil legs have several) can see NO new row at all at
        # 4 Hz -- so without the look-back this change would have made those
        # strictly worse while making every longer one better.
        $seen = [math]::Max(0, @(Get-Content $Csv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" }).Count - 1)
    }
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $Csv) {
            $rows = @(Get-Content $Csv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" })
            if ($rows.Count -gt $seen) {
                $fresh = @($rows[$seen..($rows.Count - 1)])
                $seen = $rows.Count
                foreach ($row in $fresh) {
                    $c = $row.Split(",")
                    if (& $Predicate $c) {
                        $age = $rows.Count - ([array]::IndexOf($rows, $row) + 1)
                        Say "TRIGGER $What after $([math]::Round(($TimeoutS - ($deadline - (Get-Date)).TotalSeconds), 2)) s ($age row(s) ago of $($fresh.Count) new): $row"
                        if ($Out -ne "") {
                            & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out $Out | ForEach-Object { Say $_ }
                        }
                        return $true
                    }
                }
            }
        }
        Start-Sleep -Milliseconds 120
    }
    Say "TRIGGER $What NEVER FIRED inside $TimeoutS s -- no frame taken"
    return $false
}

# ── 0a. THE DRY RUN (WPN2e audit, the cheap half of carried 281) ─────────────
#
#    Everything this file does that is a PURE FUNCTION of a recorded session,
#    run against one, with no editor, no player and no input synthesiser. It
#    exists because the three faults wave WPN2e shipped into the leg below were
#    all of this shape and all invisible to every gate in the repository.
if ($DryRun -ne "") {
    $csv = $DryRun
    if (Test-Path -PathType Container $csv) { $csv = Join-Path $csv "hero.csv" }
    if (-not (Test-Path $csv)) {
        Say "DRYRUN: no hero.csv at $csv"
        exit 4
    }
    $bad = 0
    $rows = @(Get-Content $csv | Where-Object { $_ -match "^[0-9]" })
    Say ("DRYRUN over {0}: {1} data row(s)" -f $csv, $rows.Count)
    if ($rows.Count -eq 0) { Say "DRYRUN FAIL: the recording has no data rows"; exit 4 }

    # (1) `Wait-ForHero` answers TRUE for a predicate that holds, and the
    #     `@(...)[-1]` idiom every leg uses gets a BOOLEAN out of it. Fault 1 was
    #     a leg that indexed the return value as if it were the row.
    #     `-FromRow 0` because a finished recording appends nothing and a live
    #     leg's baseline is "what was already there".
    $hit = @(Wait-ForHero -Csv $csv -What "DRYRUN a predicate that must hold" -TimeoutS 2.0 `
        -FromRow 0 -Predicate { param($c) $c.Count -gt 5 })[-1]
    if ($hit -isnot [bool]) { Say "DRYRUN FAIL: Wait-ForHero did not answer a boolean"; $bad++ }
    elseif (-not $hit) { Say "DRYRUN FAIL: a predicate that must hold did not fire"; $bad++ }

    # (2) …and FALSE for one that cannot, inside its own timeout.
    $miss = @(Wait-ForHero -Csv $csv -What "DRYRUN a predicate that cannot hold" -TimeoutS 1.0 `
        -FromRow 0 -Predicate { param($c) $c.Count -gt 9999 })[-1]
    if ($miss -isnot [bool]) { Say "DRYRUN FAIL: the miss did not answer a boolean"; $bad++ }
    elseif ($miss) { Say "DRYRUN FAIL: a predicate that cannot hold fired"; $bad++ }

    # (2b) **AND IT SEES A ROW THAT IS NOT THE LAST ONE** (VEH3b audit). The
    #      whole of that wave's carried item 3: the trigger tested `$rows[-1]`
    #      alone, four times a second, against a log written sixty times a
    #      second, so an event that lasted one row was invisible however loudly
    #      the world reported it. This asks for a predicate that holds on
    #      EXACTLY ONE interior row of the recording -- the first data row -- and
    #      cannot hold on the last.
    $first = $rows[0]
    $interior = $true
    if ($rows.Count -lt 2) {
        Say "DRYRUN: the recording has one row, so the interior-row check has no interior"
    }
    else {
        $interior = @(Wait-ForHero -Csv $csv -What "DRYRUN a predicate that holds only on the FIRST row" -TimeoutS 2.0 `
            -FromRow 0 -Predicate { param($c) ($c -join ",") -eq $first })[-1]
    }
    if (-not $interior) {
        Say "DRYRUN FAIL: the trigger cannot see a row that is not the last one -- it is back to polling `$rows[-1] and every short event is invisible to it"
        $bad++
    }
    else {
        Say ("DRYRUN: the trigger matched row 1 of {0}, so it scans what was appended rather than the tail" -f $rows.Count)
    }

    # (3) THE SHOOTOUT SUMMARY'S OWN PIPELINE. Fault 2 was
    #     `ForEach-Object { $_.Split(",") }`, which UNROLLS, so the filter behind
    #     it tested a single string's `.Count` and the whole block printed
    #     nothing at all -- silently -- for two full sessions.
    $rowsE = @(Get-Content $csv | Where-Object { $_ -match "^[0-9]" } |
        ForEach-Object { , $_.Split(",") } | Where-Object { $_.Count -gt 33 })
    Say ("DRYRUN: the shootout summary's pipeline yields {0} row(s) of >33 columns" -f $rowsE.Count)
    if ($rowsE.Count -eq 0) {
        Say "DRYRUN FAIL: the summary pipeline produced nothing -- either it unrolls again or the recording predates the columns"
        $bad++
    }
    else {
        # …and the peaks it computes do not throw.
        $peakEng = ($rowsE | ForEach-Object { [int]$_[32] } | Measure-Object -Maximum).Maximum
        $peakInc = ($rowsE | ForEach-Object { [int]$_[33] } | Measure-Object -Maximum).Maximum
        $peakBrass = ($rowsE | ForEach-Object { [int]$_[27] } | Measure-Object -Maximum).Maximum
        Say ("DRYRUN: peak engaged {0}, incoming {1}, casings {2}" -f $peakEng, $peakInc, $peakBrass)
    }

    # (4) EVERY COLUMN INDEX THE LEGS READ is inside the recording's own width.
    #     A leg that reads a column the player stopped writing is the next fault
    #     of this shape, and it would throw exactly where fault 1 did.
    $width = $rows[-1].Split(",").Count
    foreach ($i in @(5, 11, 22, 27, 28, 32, 33, 34, 35, 36, 49, 50, 51, 52, 53)) {
        if ($i -ge $width) {
            Say ("DRYRUN FAIL: a leg reads column {0} and the recording is {1} wide" -f $i, $width)
            $bad++
        }
    }
    Say ("DRYRUN: hero.csv is {0} columns wide" -f $width)

    # (5) …and nothing above threw. This is carried 282's own check, run on the
    #     part of the script a dry run can reach.
    if ($Error.Count -gt 0) {
        Say ("DRYRUN FAIL: {0} exception(s) were thrown" -f $Error.Count)
        foreach ($e in $Error) { Say ("  " + $e.ToString()) }
        $bad++
    }
    if ($bad -gt 0) {
        Say "DRYRUN FAILED: $bad check(s)"
        exit 4
    }
    Say "DRYRUN OK: 5 checks, 0 failures -- no editor was launched"
    exit 0
}

Say "repo    $repo"
Say "mode    $PlayMode"
Say "out     $OutDir"

# ── 0. nothing of ours may be running ────────────────────────────────────────
#
#    The island's pack is memory-mapped and a build that tries to replace a
#    RUNNING executable fails as a sharing violation, which MSVC reports as
#    LNK1104 and which reads like a disk problem. Refuse early and say why.
$running = Get-Process -ErrorAction Ignore |
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
if ($SpawnAt -ne "") {
    # Fold `-FaceAt` into the placement entries. A missing yaw leaves the entry
    # exactly as it was, so a caller that gives fewer headings than placements
    # gets the old behaviour on the rest rather than a parse error.
    $spawnValue = $SpawnAt
    if ($FaceAt -ne "") {
        $yaws = @($FaceAt.Split(";"))
        $entries = @($SpawnAt.Split(";"))
        for ($i = 0; $i -lt $entries.Count; $i++) {
            if ($i -lt $yaws.Count -and $yaws[$i].Trim() -ne "") {
                $entries[$i] = "$($entries[$i])/$($yaws[$i].Trim())"
            }
        }
        $spawnValue = ($entries -join ";")
    }
    $env:INF_PIE_SPAWN_AT = $spawnValue; Say "spawn override: $spawnValue"
}
else { Remove-Item env:INF_PIE_SPAWN_AT -ErrorAction Ignore }
if ($WearCloth -ne "") { $env:INF_PIE_WEAR_CLOTH = $WearCloth; Say "wear cloth: $WearCloth" }
else { Remove-Item env:INF_PIE_WEAR_CLOTH -ErrorAction Ignore }
# WPN2a: the WHOLE list goes to the player, which puts every one of them in the
# hero's bag and equips the first. The loop cycles the rest in with the SCROLL
# WHEEL -- the shipped `weapon_switch` verb -- so one session photographs one
# weapon of each class instead of seven sessions photographing one each.
$armList = @()
if ($ArmHero -ne "") {
    $armList = @($ArmHero.Split(";") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    if ($armList.Count -gt 0) {
        # `id@dwell` per entry when a dwell is asked for: the player then
        # ROTATES through the list with `equip_weapon` instead of leaving this
        # leg to guess with the wheel (carried 209).
        if ($ArmDwellS -gt 0) {
            $env:INF_PIE_ARM_HERO = (($armList | ForEach-Object { "$_@$ArmDwellS" }) -join ";")
            Say "arm hero: $($armList -join ', ') -- rotating every $ArmDwellS s through the ECS door"
        }
        else {
            $env:INF_PIE_ARM_HERO = ($armList -join ";")
            Say "arm hero: $($armList -join ', ')"
        }
    }
}
if ($armList.Count -eq 0) { Remove-Item env:INF_PIE_ARM_HERO -ErrorAction Ignore }
if ($TuneVehicle -ne "") { $env:INF_PIE_TUNE_VEHICLE = $TuneVehicle; Say "tune vehicle: $TuneVehicle" }
else { Remove-Item env:INF_PIE_TUNE_VEHICLE -ErrorAction Ignore }
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
if ($Portrait -and (Get-Command node -ErrorAction Ignore)) {
    Say "framing the hero's face (document only; never saved)"
    & node (Join-Path $PSScriptRoot "portrait.mjs") $Port 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -ne 0) { Say "  portrait.mjs exit $LASTEXITCODE" }
    Start-Sleep -Seconds 3
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01d-portrait.png") -WindowTitle "Infini" -Foreground |
        ForEach-Object { Say $_ }
    # …and back, so the frames after this one are the level's own pose. ONE
    # undo: the portrait is one transaction, the hero's own translation.
    if (Get-Command node -ErrorAction Ignore) {
        & node (Join-Path $PSScriptRoot "undo.mjs") $Port 1 2>&1 | ForEach-Object { Say "  cdp: $_" }
    }
    Start-Sleep -Seconds 2
}

if ($PlaceFemale -and (Get-Command node -ErrorAction Ignore)) {
    Say "placing the FEMALE committed body beside the pawn (document only; never saved)"
    & node (Join-Path $PSScriptRoot "place.mjs") $Port 2>&1 | ForEach-Object { Say "  cdp: $_" }
    if ($LASTEXITCODE -ne 0) { Say "  place.mjs exit $LASTEXITCODE" }
    Start-Sleep -Seconds 3
}
& powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "01b-editor-settled.png") -WindowTitle "Infini" -Foreground |
    ForEach-Object { Say $_ }

# ── 3. press Play ────────────────────────────────────────────────────────────
$pressed = $false
if (Get-Command node -ErrorAction Ignore) {
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
        Stop-Process -Id $proc.Id -Force -ErrorAction Ignore
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
    $player = Get-Process -Name "inf-player" -ErrorAction Ignore | Select-Object -First 1
    if ($player) { Say "player pid $($player.Id) after $($i + 1) s"; break }
    if ($proc.HasExited) { Say "EDITOR EXITED with $($proc.ExitCode)"; exit 4 }
    Start-Sleep -Seconds 1
}
if (-not $player) {
    Say "NO PLAYER after $PieWaitS s"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "02-no-player.png") | ForEach-Object { Say $_ }
    if (-not $KeepOpen) { Stop-Process -Id $proc.Id -Force -ErrorAction Ignore }
    exit 5
}

# A console window is the defect this wave closed; look for one belonging to
# either process while both are alive.
$consoles = Get-Process -ErrorAction Ignore |
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
  // **The LEFT button, which is `attack`** (wave WPN2a). `Click` presses and
  // releases in one call, which is a single semi-automatic shot and nothing an
  // automatic weapon can be filmed with; a tracer in flight needs the trigger
  // HELD across several 60 Hz steps.
  public static void LeftDown() { mouse_event(0x0002, 0, 0, 0, IntPtr.Zero); }
  public static void LeftUp() { mouse_event(0x0004, 0, 0, 0, IntPtr.Zero); }
  // **THE SCROLL WHEEL, which is `weapon_switch`** (wave WPN2a). One notch is
  // 120; the sign is the direction. It is how the loop gets one frame per
  // weapon CLASS out of one session, through the verb a player uses.
  public static void Wheel(int notches) {
    mouse_event(0x0800, 0, 0, (uint)(notches * 120), IntPtr.Zero);
  }
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

# ── 5a. THE ISLAND'S OWN SIDEARM, with no environment variable ───────────────
#
#    Carried 204, closed by the WPN2a audit. The island's level Blueprint
#    (`island_author_class`) defines the eighty-five-row registry on `BeginPlay`
#    and puts ONE `glock_17` on the kerb 1.4 m in front of where the hero spawns.
#    This leg is the proof that a PLAYER can arm themselves: it runs only when
#    `-ArmHero` was NOT given, because with the preview door set the hero is
#    already holding something and the leg would prove nothing.
#
#    It is first, before any movement, because the pickup is at the spawn and a
#    hero that has run down the street is out of the E key's 2.5 m reach.
$inHand = $false
if ($armList.Count -eq 0) {
    Say "-- SIDEARM: the pickup the island's own Blueprint put on the kerb --"
    Restore-PlayerFocus "before the sidearm"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "08-sidearm-on-the-kerb.png") | ForEach-Object { Say $_ }
    # E takes it into the BAG (`InteractVerb::PickUp`); the WHEEL brings it into
    # the hand (`weapon_switch`). Both are shipped controls and neither is this
    # script's own door. The trigger is column 23 -- what the sim says is in the
    # hand -- so the frame cannot be of an empty pair of hands.
    for ($k = 0; ($k -lt 10) -and (-not $inHand); $k++) {
        [InfInput]::Down(0x12); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x12)   # scancode: E
        Start-Sleep -Milliseconds 250
        [InfInput]::Wheel(1)
        $inHand = @(Wait-ForHero -Csv $heroCsv -What "the island's sidearm in the hand (try $($k + 1))" -TimeoutS 1.2 `
            -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq "glock_17") } `
            -Out (Join-Path $OutDir "09-sidearm-in-hand.png"))[-1]
    }
    if ($inHand) {
        # …and it FIRES. A Glock's hitscan threshold is 30 m, so a shot down a
        # street past thirty metres is a round in flight and column 21 sees it.
        [InfInput]::Look(0, -60)
        Start-Sleep -Milliseconds 300
        $fired = $false
        for ($t = 0; ($t -lt 6) -and (-not $fired); $t++) {
            [InfInput]::LeftDown(); Start-Sleep -Milliseconds 220
            [InfInput]::LeftUp()
            $fired = @(Wait-ForHero -Csv $heroCsv -What "the island's own sidearm fired (press $($t + 1))" -TimeoutS 1.4 `
                -Predicate { param($c) ($c.Count -gt 22) -and ([int]$c[20] -gt 0) -and ($c[22].Trim() -eq "glock_17") } `
                -Out (Join-Path $OutDir "10-sidearm-fired.png"))[-1]
        }
        if (-not $fired) { Say "SIDEARM: it is in the hand and no round left it" }
        [InfInput]::Look(0, 60)
    }
    else {
        Say "SIDEARM: ten taps of E and a wheel notch each, and the hand is still empty -- carried 204 is not closed on this build"
    }
    Start-Sleep -Milliseconds 400
}

# ── 5a2. THE FEEL (wave WPN2b) ───────────────────────────────────────────────
#
#    Five claims, five frames, every one of them TRIGGERED on a column the sim
#    writes rather than taken after a sleep:
#
#      col 24 `recoil_mm`       how far the hold-point spring has the weapon off
#                               the aim line right now
#      col 25 `aim_recoil_deg`  how far the AIM itself has been pushed
#      col 26 `spread_deg`      the whole cone the next round leaves through
#      col 27 `ads`             the aim-down-sights blend
#      col 14 `boom_m`          the camera arm the aim block pulls in to 2.0 m
#
#    (One-based here; `$c[..]` below is zero-based, as everywhere in this file.)
#
#    **THE GLOCK IS SEMI-AUTOMATIC**, and the first session of this wave learned
#    it the hard way: the leg HELD the left button and got exactly one round, so
#    three of its five frames never triggered. `automatic = false` on the
#    registry's ten handguns means `try_fire` wants a fresh press per round, so a
#    burst here is a sequence of CLICKS -- which is also what a player does.
#
#    It runs only when the sidearm leg above actually put the Glock in the hand,
#    because every frame is of a weapon and a frame of an empty pair of hands
#    captioned "the recoil" is worse than no frame.
if ($armList.Count -eq 0 -and $inHand) {
    Say "-- WPN2b: the feel, on the island's own Glock --"
    Restore-PlayerFocus "before the feel leg"
    # Level the aim: the sidearm leg left it pitched down 60 counts and a burst
    # into the pavement is a burst nobody can see.
    [InfInput]::Look(0, -40)
    Start-Sleep -Milliseconds 400

    # (1) ADS. Right button HELD, and the frame is taken the moment the camera
    #     has actually arrived -- the boom at the aim block's 2.0 m AND the sim's
    #     own blend at 1. Two columns, because either alone is satisfied by a
    #     camera that was already close.
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "90-hip.png") | ForEach-Object { Say $_ }
    [InfInput]::RightDown()
    $ads = @(Wait-ForHero -Csv $heroCsv -What "the aim arrived (boom at 2.0 m, blend at 1)" -TimeoutS 4.0 `
        -Predicate { param($c) ($c.Count -gt 26) -and ([double]$c[26] -gt 0.98) -and ([double]$c[13] -lt 2.4) } `
        -Out (Join-Path $OutDir "91-ads.png"))[-1]
    if (-not $ads) { Say "WPN2b: the aim never arrived -- no ADS frame" }

    # (2) THE BURST, at three frames — AT THE WEAPON'S OWN RATE (WPN2b audit).
    #
    #     A shot's climb peaks four steps and 66 ms after the trigger, so these
    #     cannot be taken on a wall clock: each frame is armed on a column
    #     crossing a threshold the one before it did not.
    #
    #     **AND THE ROUNDS HAVE TO STACK, or the frames are three samples of a
    #     plateau.** The first cut of this leg clicked once and then blocked in
    #     a half-second `Wait-ForHero` before clicking again — about two rounds
    #     a second against the Glock's own 450 rpm — and a 2.5-recoil spring is
    #     home in a third of a second, so nothing ever accumulated. Measured on
    #     that session: the three frames came out at aim **3.2014 / 3.2723 /
    #     3.6925 deg** and hold points of **88.037 / 89.159 / 70.221 mm**, so
    #     the "top of the burst" frame had a fifth LESS recoil than the first
    #     one and the arm was 2 px LOWER against the head. The captions said
    #     "the arm is visibly higher"; the pixels and the CSV both said
    #     otherwise. It is the same defect the refused bloom frame already
    #     named — a Glock clicked at the harness rate cannot bloom — applied to
    #     the recoil, which the wave did not notice because a recoil frame
    #     always looks like a recoil frame.
    #
    #     So the clicks come first and the waits are SHORT: one round every
    #     ~140 ms -- about 230 rpm once PowerShell's own sleep granularity and
    #     the input calls are paid for -- with a 0.10 s look at the CSV between
    #     them. The three thresholds are then crossed by a spring that is
    #     genuinely stacking: measured, aim 2.3403 -> 5.6528 -> 5.3424 deg and
    #     hold points of 52.809 -> 122.435 -> 118.713 mm.
    #
    #     **AND THE PIXELS STILL WILL NOT SHOW IT, for a reason worth writing
    #     down.** The frame is taken by a `powershell -File $shot` SUBPROCESS,
    #     which the log's own timestamps put at up to a second behind the row
    #     that armed it -- and a 2.5-recoil pistol's hold-point spring is home
    #     in about a third of that. The whole excursion is 40 mm of lift at an
    #     aiming boom of 2.0749 m, which is 13 pixels of a 730-line window. So
    #     a photograph of THIS weapon's recoil is not obtainable through this
    #     harness at any click rate, and the honest place for the number is the
    #     row printed beside each frame. A 10-recoil launcher would photograph;
    #     nothing on the island's kerb does.
    $burst = @(
        @{ n = "92-burst-1.png"; what = "the first round's kick";                     col = 23; over = 5.0 },
        @{ n = "93-burst-2.png"; what = "the burst climbing (the aim past 4 deg)";    col = 24; over = 4.0 },
        @{ n = "94-burst-3.png"; what = "the burst at its top (the aim past 5 deg)";   col = 24; over = 5.0 }
    )
    $shotsFired = 0
    $stage = 0
    # 24 rounds is a Glock magazine and a half at 450 rpm — about 3.4 seconds.
    for ($t = 0; ($t -lt 24) -and ($stage -lt $burst.Count); $t++) {
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 40; [InfInput]::LeftUp()
        $shotsFired++
        $b = $burst[$stage]
        $col = $b.col; $over = $b.over
        # 100 ms, so the whole cycle is ~140 ms = 428 rpm. Long enough for the
        # 4 Hz hero log to have written a row roughly every third click, short
        # enough that the spring never comes home between rounds.
        $got = @(Wait-ForHero -Csv $heroCsv -What $b.what -TimeoutS 0.10 `
            -Predicate { param($c) ($c.Count -gt $col) -and ([double]$c[$col] -gt $over) } `
            -Out (Join-Path $OutDir $b.n))[-1]
        if ($got) { $stage++ }
    }
    for ($i = $stage; $i -lt $burst.Count; $i++) {
        Say "WPN2b: $($burst[$i].what) never fired after $shotsFired rounds"
    }

    # (3) THE SPREAD, at the top of the magazine's own bloom — AT THE WEAPON'S
    #     OWN RATE (WPN2b audit). The cone widens with every round and decays
    #     between them, so a bloom grows only while the rounds come faster than
    #     the decay: `feel::BLOOM_DECAY_PER_S`'s own inequality, `D < rpm / 450`.
    #
    #     TWO things had to change for this frame to exist. The constant was
    #     0.8, which needs 360 rpm — thirty-six of the registry's eighty-five
    #     rows are below that, including the AA-12, which is a FULL-AUTOMATIC
    #     shotgun; it is 0.6 now, priced against the slowest automatic row in
    #     the file. And this leg clicked once every ~400 ms, which is 150 rpm
    #     against the Glock's 450 — so its own decay outran it and the wave
    #     reported "the cone never bloomed past 1.6 deg over 20 rounds" as the
    #     constant behaving correctly. It was half that and half this.
    #
    #     At 450 rpm a 2.5-recoil Glock gains 0.150 deg a round against 0.095 of
    #     decay and saturates at 1.125 deg over its own 1.20 base, so 1.60 is a
    #     third of the way up a real bloom. The button is up, so the cone here is
    #     the HIP cone.
    [InfInput]::RightUp()
    Start-Sleep -Milliseconds 200
    #     **The clicks come with NO CSV read between them.** Measured on the
    #     first audit run: a leg that polls the log between rounds fires at
    #     about 227 rpm -- half the Glock's own rate -- because a
    #     `Wait-ForHero` turnaround is longer than a fire interval, and at
    #     227 rpm the decay (0.178 deg between rounds) outruns the gain
    #     (0.150), so the cone still cannot climb. So the burst is fired
    #     BLIND, at the weapon's own 133 ms, and the log is read once at the
    #     end -- which is the only shape that reaches a real bloom through a
    #     scripted trigger.
    #     **RELOAD FIRST.** Measured on the second audit run: the leg reached
    #     the bloom burst with an empty magazine, the weapon spent the whole
    #     1.6 s firing NOTHING, and the cone sat at exactly 1.2000 -- the
    #     registry's own base -- for six seconds of hero log. A Glock holds
    #     seventeen and the burst leg above it spends most of them.
    [InfInput]::Down(0x13)   # scancode: R
    Start-Sleep -Milliseconds 60
    [InfInput]::Up(0x13)
    Start-Sleep -Milliseconds 1700
    for ($t = 0; $t -lt 12; $t++) {
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 40; [InfInput]::LeftUp()
        Start-Sleep -Milliseconds 95
        $shotsFired++
    }
    $bloom = @(Wait-ForHero -Csv $heroCsv -What "the cone bloomed past 1.60 deg (the registry's own base is 1.20)" -TimeoutS 1.0 `
        -Predicate { param($c) ($c.Count -gt 25) -and ([double]$c[25] -gt 1.60) } `
        -Out (Join-Path $OutDir "95-bloom.png"))[-1]
    if (-not $bloom) { Say "WPN2b: the cone never bloomed past 1.60 deg over $shotsFired rounds" }
    Say "WPN2b: $shotsFired rounds fired over the feel leg"
    Start-Sleep -Milliseconds 1200
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "96-after-the-burst.png") | ForEach-Object { Say $_ }

    # (4) THE SWAY, at a sprint, with the weapon still in the hand. A sprint is
    #     REFUSED while aiming (ALS's own `CanSprint`), so the leg waits for the
    #     blend to fall back to zero before it presses anything -- the first
    #     session held the right button through fourteen seconds of timeouts and
    #     the hero never moved a metre. The frame waits for the hero to actually
    #     be moving (column 7) rather than for a key to have been pressed.
    @(Wait-ForHero -Csv $heroCsv -What "the aim released (the blend back to zero)" -TimeoutS 3.0 `
        -Predicate { param($c) ($c.Count -gt 26) -and ([double]$c[26] -lt 0.02) }) | Out-Null
    [InfInput]::Down(0x2A)   # scancode: Left Shift
    [InfInput]::Down(0x11)   # scancode: W
    $sway = @(Wait-ForHero -Csv $heroCsv -What "the hero sprinting with the sidearm out" -TimeoutS 5.0 `
        -Predicate { param($c) ($c.Count -gt 22) -and ([double]$c[6] -gt 4.0) -and ($c[22].Trim() -eq "glock_17") } `
        -Out (Join-Path $OutDir "97-sway-sprint.png"))[-1]
    if (-not $sway) { Say "WPN2b: the hero never reached a sprint with the weapon out" }
    [InfInput]::Up(0x11)
    [InfInput]::Up(0x2A)
    Start-Sleep -Milliseconds 600
    Restore-PlayerFocus "after the feel leg"
    # ------------------------------------------------------------------ WPN2c
    # (5) THE SOUND AND THE BRASS. Two things a screenshot can carry: casings on
    #     the ground, and WHICH TAIL the shot chose. Column 28 is how many
    #     casings exist right now and column 29 is `indoor` / `outdoor` / `-`;
    #     both are ZERO-based `$c[27]` and `$c[28]`.
    #
    #     The brass is why the trigger for the first frame is `casings -gt 0`
    #     rather than a sleep: a case is in the air for well under a second and
    #     on the ground for eight, so a leg that fired and then looked would
    #     photograph either an empty floor or a blur, depending on the machine.
    Say "WPN2c: a burst for the brass"
    $brassFired = 0
    for ($t = 0; $t -lt 8; $t++) {
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 40; [InfInput]::LeftUp()
        Start-Sleep -Milliseconds 95
        $brassFired++
    }
    $air = @(Wait-ForHero -Csv $heroCsv -What "brass in the air ($brassFired rounds fired)" -TimeoutS 2.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ([int]$c[27] -gt 0) } `
        -Out (Join-Path $OutDir "98-brass-in-the-air.png"))[-1]
    if (-not $air) { Say "WPN2c: no casing was ever live -- no brass frame" }
    # …and the TAIL the street chose. Six rays from the muzzle, four of them
    # hitting inside eight metres is indoors; on a street the ground and at most
    # two walls answer, so this must read `outdoor`.
    $tail = @(Wait-ForHero -Csv $heroCsv -What "the street's own tail" -TimeoutS 2.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ($c[28].Trim() -eq "outdoor") } `
        -Out (Join-Path $OutDir "99-outdoor-tail.png"))[-1]
    if (-not $tail) { Say "WPN2c: the enclosure probe never called the street outdoors" }
    # The brass settles and stays for eight seconds: this frame is the floor.
    Start-Sleep -Milliseconds 1400
    [InfInput]::Look(0, 320)   # look down at the ground
    Start-Sleep -Milliseconds 500
    $floor = @(Wait-ForHero -Csv $heroCsv -What "brass on the ground" -TimeoutS 2.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ([int]$c[27] -gt 0) } `
        -Out (Join-Path $OutDir "100-brass-on-the-ground.png"))[-1]
    if (-not $floor) { Say "WPN2c: the brass was gone before the floor frame" }
    [InfInput]::Look(0, -320)
    Start-Sleep -Milliseconds 400
    Restore-PlayerFocus "after the brass leg"
    # ----------------------------------------------------- WPN2c AUDIT: THE PIXELS
    # (6) THE BRASS, CLOSE ENOUGH TO SEE. The three frames above are TRIGGERED on
    #     the casing column and are honest about the sim -- and at a 1.2-3 m
    #     third-person boom a 19 mm casing is one or two pixels, so
    #     `100-brass-on-the-ground.png` is a photograph of a road with the brass
    #     below the resolution of the claim its caption makes. That is the
    #     CHAR1b.2 caption law's shape, and the fix is a camera and not a
    #     caption.
    #
    #     FIRST PERSON (G) puts the eye at ~1.6 m instead of on a boom, and
    #     looking down 60-70 degrees fills the frame with the road the shells are
    #     on: a 19 mm case at 1.7 m is ~7 px of a 715-line window, which crops
    #     and diffs.
    #
    #     THE CONTROL FRAME is what makes it a measurement rather than a better
    #     picture. `101` is the same camera, same pose, with `casings == 0` --
    #     so `101` against `103` is a pixel difference that can only be the
    #     brass. A leg with no control frame cannot tell "the casings are drawn"
    #     from "the road has speckles in its texture", and this road does.
    Say "WPN2c-AUDIT: the brass, close enough to see"
    # **RELOAD FIRST**, for the reason the bloom burst above already carries and
    # this leg then met on its own first run: a Glock holds seventeen, the legs
    # before this one spend all of them and the one reload the feel leg does, and
    # a leg that clicks an empty magazine ten times photographs an empty road and
    # reports "the brass was gone". Measured: 10 clicks, 0 casings, 0 shots.
    Restore-PlayerFocus "before the close-up"
    [InfInput]::Down(0x13); Start-Sleep -Milliseconds 60; [InfInput]::Up(0x13)   # R
    Start-Sleep -Milliseconds 1700
    [InfInput]::Down(0x22); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x22)   # G: first person
    $fp = @(Wait-ForHero -Csv $heroCsv -What "the first-person seat" -TimeoutS 5.0 `
        -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[13] -lt 0.4) })[-1]
    if (-not $fp) { Say "WPN2c-AUDIT: no first-person seat -- the close-up is on the boom" }
    # Down, in steps: one big `Look` is a single mouse delta the camera clamps.
    for ($i = 0; $i -lt 6; $i++) { [InfInput]::Look(0, 120); Start-Sleep -Milliseconds 40 }
    Start-Sleep -Milliseconds 600
    # THE CONTROL: this exact camera with an empty road. A casing lives eight
    # seconds, so this waits the last burst out rather than sleeping a guess.
    $clean = @(Wait-ForHero -Csv $heroCsv -What "the road with no brass on it (the control)" -TimeoutS 14.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ([int]$c[27] -eq 0) } `
        -Out (Join-Path $OutDir "101-road-before-the-brass.png"))[-1]
    if (-not $clean) { Say "WPN2c-AUDIT: the road never emptied -- 101 is not a control" }
    $closeFired = 0
    for ($t = 0; $t -lt 10; $t++) {
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 40; [InfInput]::LeftUp()
        Start-Sleep -Milliseconds 95
        $closeFired++
    }
    $airClose = @(Wait-ForHero -Csv $heroCsv -What "brass in the air, close ($closeFired rounds)" -TimeoutS 2.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ([int]$c[27] -gt 0) } `
        -Out (Join-Path $OutDir "102-brass-airborne-close.png"))[-1]
    if (-not $airClose) { Say "WPN2c-AUDIT: no casing was live for the airborne close-up" }
    # …and settled. A case bounces once and stops on its second contact, which is
    # well inside a second; this waits for the fall and photographs the floor.
    Start-Sleep -Milliseconds 1800
    $floorClose = @(Wait-ForHero -Csv $heroCsv -What "brass on the road, close" -TimeoutS 3.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ([int]$c[27] -gt 0) } `
        -Out (Join-Path $OutDir "103-brass-on-the-road-close.png"))[-1]
    if (-not $floorClose) { Say "WPN2c-AUDIT: the brass was gone before the close floor frame" }
    # One from the ejection side: the port throws right, so a quarter turn that
    # way is where a pile of it is.
    [InfInput]::Look(240, -140); Start-Sleep -Milliseconds 700
    $side = @(Wait-ForHero -Csv $heroCsv -What "the brass from the ejection side" -TimeoutS 3.0 `
        -Predicate { param($c) ($c.Count -gt 28) -and ([int]$c[27] -gt 0) } `
        -Out (Join-Path $OutDir "104-brass-from-the-ejection-side.png"))[-1]
    if (-not $side) { Say "WPN2c-AUDIT: no brass was live for the side frame" }
    [InfInput]::Look(-240, -580)
    Start-Sleep -Milliseconds 300
    [InfInput]::Down(0x22); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x22)   # G: back to third person
    Start-Sleep -Milliseconds 400
    Restore-PlayerFocus "after the close-up leg"
}

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
$gotCrouch = @(Wait-ForHero -Csv $heroCsv -What "a crouch" -TimeoutS 2.5 `
    -Predicate { param($c) $c[5] -eq "Crouch" } `
    -Out (Join-Path $OutDir "14-crouch.png"))[-1]
if (-not $gotCrouch) {
    Say "the crouch tap was swallowed; tapping C again"
    [InfInput]::Down(0x2E); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x2E)
    $gotCrouch = @(Wait-ForHero -Csv $heroCsv -What "a crouch (2nd tap)" -TimeoutS 3.0 `
        -Predicate { param($c) $c[5] -eq "Crouch" } `
        -Out (Join-Path $OutDir "14-crouch.png"))[-1]
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
if (@(Wait-ForHero -Csv $heroCsv -What "the slide" -TimeoutS 3.0 `
        -Predicate { param($c) $c[5] -eq "Slide" } `
        -Out (Join-Path $OutDir "32-slide.png"))[-1]) {
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
        $standing = @(Wait-ForHero -Csv $heroCsv -What "a standing hero ($why)" -TimeoutS 1.5 `
            -Predicate { param($c) ($c[5] -eq "Grounded") -and ($c[11] -notmatch "^(crouch|prone|slide)") })[-1]
        if ($standing) { return $true }
        [InfInput]::Down(0x2E); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x2E)   # scancode: C
        Start-Sleep -Milliseconds 500
    }
    # **AND SAY WHAT THE HERO ACTUALLY IS** (VEH3b audit). `C` is the stance key,
    # so four taps of it answer crouch, prone and slide and nothing else. A hero
    # left in `Ragdoll` by a drop whose get-up never came is not a stance
    # problem, and every leg after it runs against a body on the floor -- the
    # boarding hunt included, which then reports "NO CAR reached in thirty-six
    # taps of E" and blames the hunt. Measured: one audit session logged **607
    # Ragdoll rows** and a `get-up NEVER FIRED`, and its VEH3a and VEH3b legs
    # took none of their nine frames.
    $mode = "?"
    $state = "?"
    $rowsNow = @(Get-Content $heroCsv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" })
    if ($rowsNow.Count -gt 0) {
        $cNow = $rowsNow[-1].Split(",")
        if ($cNow.Count -gt 11) { $mode = $cNow[5]; $state = $cNow[11] }
    }
    # **A FALL THAT IS NOT FALLING IS A WEDGE** (VEH3b audit). The same session
    # that ragdolled spent the TEN MINUTES before it in `FallControlled` with its
    # position frozen to the digit -- 2 140 rows at (-1765.4930, 16.9137,
    # 2143.5534), y moving 11 cm in ten minutes, `fall` then `fall_fast` -- after
    # walking into something that lifted it 0.47 m off a `Grounded` run. Then it
    # ragdolled and never got up. A loop that kept tapping keys at it for the
    # rest of the session is a loop that could not see any of that.
    $frozen = $false
    if ($rowsNow.Count -ge 8) {
        $tail = @($rowsNow[($rowsNow.Count - 8)..($rowsNow.Count - 1)] | ForEach-Object {
                $q = $_ -split ","
                "{0},{1},{2}" -f $q[2], $q[3], $q[4]
            })
        $frozen = (@($tail | Select-Object -Unique).Count -eq 1)
    }
    if ($mode -eq "Ragdoll") {
        Say "STILL NOT STANDING ($why): the hero is RAGDOLLED (state '$state') and C is the stance key -- no input in this loop gets a body up, the engine's own get-up does, and it has not come. Every leg after this one is running against a body on the floor."
    }
    elseif ($frozen -and $mode -match "^Fall") {
        Say "STILL NOT STANDING ($why): the hero is WEDGED -- mode '$mode', state '$state', and its position has not changed over the last eight samples. It is falling and not moving, which is geometry holding it; no key in this loop frees it and every leg after this one is aimed at a character that cannot walk."
    }
    elseif ($frozen) {
        Say "STILL NOT STANDING ($why): mode '$mode', state '$state', and the position has not changed over the last eight samples -- the hero is stuck, not mid-animation"
    }
    else {
        Say "STILL NOT STANDING after four taps of C ($why): mode '$mode', state '$state' -- the frames below are of whatever stance the world is in"
    }
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
    $gotCar = @(Wait-ForHero -Csv $heroCsv -What "the drive camera blending" -TimeoutS 0.9 `
        -Predicate { param($c) ($c.Count -gt 13) -and ($c[5] -eq "Driving") -and ([double]$c[13] -gt 4.0) } `
        -Out (Join-Path $OutDir "69-camera-vehicle-blend.png"))[-1]
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
$gotWall = @(Wait-ForHero -Csv $heroCsv -What "a clipped boom" -TimeoutS 8.0 `
    -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[7] -gt 0.8) } `
    -Out (Join-Path $OutDir "61-camera-against-a-wall.png"))[-1]
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
            $mantled = @(Wait-ForHero -Csv $heroCsv -What "a mantle" -TimeoutS 1.2 `
                -Predicate { param($c) $c[11] -match "^mantle" } `
                -Out (Join-Path $OutDir "41-mantling.png"))[-1]
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
    # 1b. THE COVER TOUR (wave COV1). Every frame is TRIGGERED on the log's own
    #     cover columns -- 17 the CLASS, 18 the SIDE, 19 how far out the peek is
    #     -- which column 5's mode alone cannot say. The stations are the ones
    #     `cov1_gate::the_cover_census_over_the_island` measured: the island's
    #     cover is 231 facade boxes, 5 low grammar walls and 11 vehicle
    #     entities, so the HIGH station is a shop front and the LOW station is a
    #     grammar wall.
    #
    #     T is the cover key (scancode 0x14).
    #     **The stations are the CENSUS's own**, and the low one is new (the
    #     COV1 audit). The wave's "low" station was 7.3 m from its own
    #     placement and its rows read `High` — it photographed the high shop
    #     front a second time. `the_cover_census_over_the_island` now prints
    #     WHERE the five LOW surfaces are, and this is one of them: a
    #     0.976 m structure of `Harbour City Shop -1,0`.
    $coverStations = @(
        @{ Name = "high"; X = -1798.0; Z = 2066.0; Yaw = 135 },
        @{ Name = "low";  X = -1774.0; Z = 2090.0; Yaw = 0 }
    )
    foreach ($st in $coverStations) {
        # **`Wait-ForHero` emits its `Say` lines into the pipeline**, so what an
        # assignment captures is an ARRAY whose last element is the boolean. A
        # bare `-not $x` on that array is always false, which is how the first
        # run of this leg made exactly ONE cover press, in whatever direction
        # the placement happened to leave the hero facing, and reported success.
        $reached = @(Wait-ForHero -Csv $heroCsv -What "the $($st.Name) cover station" -TimeoutS 120 `
            -Predicate { param($c) ([math]::Abs([double]$c[2] - $st.X) -lt 2.5) -and ([math]::Abs([double]$c[4] - $st.Z) -lt 2.5) })[-1]
        if (-not $reached) { Say "the $($st.Name) cover station was never reached"; continue }
        Restore-PlayerFocus "at the $($st.Name) cover station"
        Stand-Up "before taking cover" | Out-Null
        # Face the surface: the placement puts the hero at the station and the
        # mouse points it at the wall. 15 counts is about 2.2 degrees at the
        # shipped sensitivity, so this is a coarse aim and the probe's own +-45
        # degree approach window is what makes it enough.
        Say "COVER ($($st.Name)): pressing T"
        # **The facing is SWEPT.** A placement sets a position and not a heading,
        # so the press reaches wherever the hero happens to be looking; the
        # probe's own +-45 degree approach window means a sweep of the circle in
        # 30-degree steps cannot miss a wall that is there.
        # **W is held while T is pressed**, and that is not decoration.
        # `try_cover` reaches in the STICK's direction when one is held and in
        # the BODY's facing when it is not -- and a character standing still in
        # `VelocityDirection` does not turn its body when the mouse moves, so a
        # sweep with no stick pressed sixteen times in exactly the same
        # direction. Measured: `NO COVER taken at the high station in twelve
        # presses` while the wall was four metres in front of it. Walking at the
        # wall is also how a player takes cover, and it closes the probe's own
        # 0.90 m reach.
        # **The placement now carries the heading** (`-FaceAt`), so the first
        # press is aimed. The sweep below stays as the fallback it always
        # should have been, and the log says which of the two took.
        # **THE FIRST PRESS IS STILL, and it is the heading that pays for it.**
        #
        # The sweep below holds `W` between presses, which is how the wave's own
        # leg found a wall it was not pointed at -- and it is why the audit's
        # first run reported `NO COVER taken at the high station in sixteen
        # presses` while standing four metres from one: the placement put the
        # hero AT the station and sixteen quarter-second walks carried it
        # **3.5 m away** (measured: x -1798.0 -> -1794.5 over the sweep). With
        # `-FaceAt` the hero is already looking at the surface, so the press
        # that should work is the one taken before anything moves.
        [InfInput]::Down(0x14); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x14)
        $tookCover = @(Wait-ForHero -Csv $heroCsv -What "cover ($($st.Name)), standing still" -TimeoutS 1.5 `
            -Predicate { param($c) ($c.Count -gt 17) -and ($c[5] -eq "Cover") } `
            -Out (Join-Path $OutDir "60-cover-$($st.Name).png"))[-1]
        for ($k = 0; $k -lt 16 -and -not $tookCover; $k++) {
            [InfInput]::Down(0x11)
            Start-Sleep -Milliseconds 220
            [InfInput]::Down(0x14); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x14)
            $tookCover = @(Wait-ForHero -Csv $heroCsv -What "cover ($($st.Name))" -TimeoutS 1.0 `
                -Predicate { param($c) ($c.Count -gt 17) -and ($c[5] -eq "Cover") } `
                -Out (Join-Path $OutDir "60-cover-$($st.Name).png"))[-1]
            [InfInput]::Up(0x11)
            if (-not $tookCover) { for ($i = 0; $i -lt 14; $i++) { [InfInput]::Look(15, 0); Start-Sleep -Milliseconds 16 } }
        }
        if (-not $tookCover) { Say "NO COVER taken at the $($st.Name) station in sixteen presses"; continue }
        Say "  cover taken on press $($k) at the $($st.Name) station"
        # The class, off the log rather than off the station's name.
        $row = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
        Say "  in cover: class $($row[17]) side $($row[18]) peek $($row[19]) mode $($row[5])"
        # THE SLIDE, and the corner it stops at. **Short**: three seconds of A
        # walks a character clean off the end of a four-metre wall and out of
        # cover, which is what the first run of this leg photographed.
        Say "COVER ($($st.Name)): sliding along the surface"
        [InfInput]::Down(0x1E)   # scancode: A -- move_x negative, the character's left
        Start-Sleep -Milliseconds 900
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "61-cover-slide-$($st.Name).png") | ForEach-Object { Say $_ }
        [InfInput]::Up(0x1E)
        Start-Sleep -Milliseconds 400
        # **Still in cover?** A slide that reached a corner stays; one that ran
        # out of surface does not, and pressing aim after that photographs an
        # aim rather than a peek. Re-take it if it went.
        $stillIn = @(Wait-ForHero -Csv $heroCsv -What "still in cover ($($st.Name))" -TimeoutS 1.0 `
            -Predicate { param($c) $c[5] -eq "Cover" })[-1]
        if (-not $stillIn) {
            Say "  the slide left cover; re-taking it"
            for ($k = 0; $k -lt 10 -and -not $stillIn; $k++) {
                [InfInput]::Down(0x14); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x14)
                $stillIn = @(Wait-ForHero -Csv $heroCsv -What "cover again ($($st.Name))" -TimeoutS 1.0 `
                    -Predicate { param($c) $c[5] -eq "Cover" })[-1]
                if (-not $stillIn) { for ($i = 0; $i -lt 14; $i++) { [InfInput]::Look(15, 0); Start-Sleep -Milliseconds 16 } }
            }
        }
        # THE PEEK: the aim button, and the frame is triggered on column 18
        # leaving `Behind` -- which is the difference between a peek and a
        # character holding a button.
        Say "COVER ($($st.Name)): aiming, which leans the body out"
        [InfInput]::RightDown()
        $peeked = @(Wait-ForHero -Csv $heroCsv -What "a peek ($($st.Name))" -TimeoutS 4.0 `
            -Predicate { param($c) ($c.Count -gt 19) -and ($c[5] -eq "Cover") -and ($c[18].Trim() -ne "-") -and ($c[18].Trim() -ne "Behind") -and ([double]$c[19] -gt 0.6) } `
            -Out (Join-Path $OutDir "62-cover-peek-$($st.Name).png"))[-1]
        if (-not $peeked) {
            $r = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
            Say "  no peek: mode $($r[5]) class $($r[17]) side $($r[18]) peek $($r[19])"
        }
        Start-Sleep -Milliseconds 500
        [InfInput]::RightUp()
        # THE VAULT, out of a LOW cover only: one Space press with no stick.
        if ($st.Name -eq "low" -and $stillIn) {
            # **THE VAULT IS PROVEN BY THE POSITION, NOT BY A SLEEP** (the COV1
            # audit). The wave's `64-cover-vaulted.png` was a `Start-Sleep 900`
            # and a screenshot, and its pixels are the same walk-away as
            # `60-cover-low.png` four seconds earlier — the mantle trigger
            # above never fired in that session and nothing said so. Now the
            # BEFORE and AFTER positions are read out of the hero log and
            # printed, so the caption is a number.
            Say "COVER (low): vaulting over it with Space"
            $beforeVault = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
            [InfInput]::Down(0x39); Start-Sleep -Milliseconds 60; [InfInput]::Up(0x39)
            $vaulted = @(Wait-ForHero -Csv $heroCsv -What "the vault out of cover" -TimeoutS 3.0 `
                -Predicate { param($c) $c[11] -match "^mantle" } `
                -Out (Join-Path $OutDir "63-cover-vault.png"))[-1]
            if (-not $vaulted) { Say "  NO TRAVERSAL started out of the low cover" }
            # …and the frame on the far side is triggered on LEAVING the
            # traversal rather than on a stopwatch.
            Wait-ForHero -Csv $heroCsv -What "the far side" -TimeoutS 4.0 `
                -Predicate { param($c) ($c[11] -notmatch "^mantle") -and ($c[5] -ne "Cover") } `
                -Out (Join-Path $OutDir "64-cover-vaulted.png") | Out-Null
            $afterVault = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
            $dx = [double]$afterVault[2] - [double]$beforeVault[2]
            $dy = [double]$afterVault[3] - [double]$beforeVault[3]
            $dz = [double]$afterVault[4] - [double]$beforeVault[4]
            $planar = [math]::Sqrt($dx * $dx + $dz * $dz)
            $vaultSaid = "  the vault moved the capsule {0:N3} m in the ground plane and " +
                "{1:N3} m vertically: ({2:N2}, {3:N2}, {4:N2}) -> ({5:N2}, {6:N2}, {7:N2})"
            Say ($vaultSaid -f $planar, $dy,
                [double]$beforeVault[2], [double]$beforeVault[3], [double]$beforeVault[4],
                [double]$afterVault[2], [double]$afterVault[3], [double]$afterVault[4])
        }
        # LEAVING: the same key again, and the frame is the hero standing clear.
        Say "COVER ($($st.Name)): leaving with T"
        [InfInput]::Down(0x14); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x14)
        Wait-ForHero -Csv $heroCsv -What "out of cover ($($st.Name))" -TimeoutS 3.0 `
            -Predicate { param($c) $c[5] -ne "Cover" } `
            -Out (Join-Path $OutDir "65-cover-left-$($st.Name).png") | Out-Null
    }
    # THE LAST PRESS OF THE TOUR, wherever the tour left the hero.
    #
    # **It is NOT a kerb station and the frame is named for what HAPPENED**, not
    # for what was hoped: the leg presses where the low leg left the character,
    # and in the run this comment was written for that was beside a shop front,
    # so the press took HIGH cover and the file called `...kerb-refused` was a
    # picture of a character taking cover. A kerb station needs its own
    # placement, and the kerb's refusal is proven where it can be: the gate's
    # `a_kerb_on_the_island_is_never_cover_and_the_refusal_names_it` (14
    # labelled slabs, none coverable) and the fixture's 0.15 m row.
    Say "COVER (kerb): the last placement is a kerb the gate found"
    # **A KERB STATION OF ITS OWN** (the COV1 audit). The wave's leg pressed
    # wherever the tour happened to leave the hero, which in its own session was
    # beside a shop front — so `66-cover-kerb-refused.png` is a picture of a
    # character TAKING cover and the wave said so in its report. The kerb the
    # gate's `a_kerb_on_the_island_is_never_cover_and_the_refusal_names_it`
    # stands in front of has a placement and a heading now, and the frame is
    # named for the outcome either way.
    $kerbReached = @(Wait-ForHero -Csv $heroCsv -What "the kerb station" -TimeoutS 180 `
        -Predicate { param($c) ([math]::Abs([double]$c[2] - (-1758.15)) -lt 3.0) -and ([math]::Abs([double]$c[4] - (1999.55)) -lt 3.0) })[-1]
    if (-not $kerbReached) { Say "  the kerb station was never reached; pressing where the tour left the hero" }
    Restore-PlayerFocus "at the kerb"
    Stand-Up "before the kerb press" | Out-Null
    $beforeKerb = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
    [InfInput]::Down(0x14); Start-Sleep -Milliseconds 80; [InfInput]::Up(0x14)
    Start-Sleep -Milliseconds 900
    $afterKerb = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
    Say "  last press: mode $($beforeKerb[5]) -> $($afterKerb[5]), class $($afterKerb[17])"
    $outcome = if ($afterKerb[5] -eq "Cover") { "took-cover" } else { "refused" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "66-cover-kerb-$outcome.png") | ForEach-Object { Say $_ }

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

# ── 5d. THE BALLISTICS (wave WPN2a) ──────────────────────────────────────────
#
#    A round in flight lasts a quarter of a second over a hundred metres, so
#    none of these frames can be timed with `Start-Sleep`: every one of them is
#    triggered on `hero.csv`'s two NEW columns -- 20 is how many rounds are in
#    the air right now and 21 is how far the last one that hit something had
#    flown. A frame taken on `rounds > 0` is a frame with a bullet in it.
#
#    One session, one weapon of each CLASS: the player puts the whole list in the
#    hero's bag and ROTATES through it with `equip_weapon` -- the ECS door --
#    holding each for `-ArmDwellS` seconds, so this leg waits for the hand to
#    hold what it wants rather than spinning a wheel whose notch is not a slot
#    (carried 209, closed by the WPN2a audit).
if ($armList.Count -gt 0) {
    Say "── WPN2a: the hero is armed with $($armList -join ', ') ──"
    Restore-PlayerFocus "the ballistics leg"
    # **STAND UP FIRST** (the WPN2a audit). Every ballistics frame of wave WPN2a
    # was taken of a CROUCHED hero -- column 6 reads `Crouch` in all seven of
    # them -- because the leg before this one leaves the stance where it likes
    # and this leg never asked. `Stand-Up` is the loop's own idempotent door.
    Stand-Up "before the ballistics leg" | Out-Null
    # Aim UP, so the rounds clear the street and fly for the length of the shot
    # rather than resolving inside their hitscan threshold against a shop front
    # twenty metres away. Session 2 of this wave aimed 140 counts up, stood
    # wherever the camera leg had left it, and fired seven weapons into a
    # building: seven shots, no round.
    #
    # UP TO THE CLAMP FIRST, then back down a fixed amount. `aim_forward` clamps
    # the pitch at 89.9 degrees, so an over-large look up is a KNOWN elevation
    # whatever the leg before this one left the aim at — and sessions 4 and 5 of
    # this wave differed by 19 degrees of head pitch for exactly that reason,
    # which is what made one of them miss the shotgun's threshold.
    [InfInput]::Look(0, -900)
    Start-Sleep -Milliseconds 300
    [InfInput]::Look(0, 240)
    Start-Sleep -Milliseconds 400
    # ADS: the reticle is drawn only while aiming, and a HUD frame with no
    # reticle in it is a frame of a readout nobody was looking through.
    [InfInput]::RightDown()
    Start-Sleep -Milliseconds 700
    $anyFlew = $false
    # **THE FRAME IS NAMED BY WHAT IS IN THE HAND, NOT BY WHAT WAS ASKED FOR.**
    # The first session of wave WPN2a cycled with the wheel and named every frame
    # after the id it MEANT to equip; the HUD in the pixels showed a different
    # magazine, because one notch of a wheel is not one slot of a bag. Column 22
    # is what the sim says is equipped, so the leg WAITS FOR IT and a class the
    # rotation never brought round takes no frame and says so.
    for ($wi = 0; $wi -lt $armList.Count; $wi++) {
        $wid = $armList[$wi]
        # **THE ROTATION, NOT THE WHEEL** (carried 209, closed by the WPN2a
        # audit). The player holds each id for `-ArmDwellS` seconds and equips
        # the next through `equip_weapon`, so this leg WAITS for the hand to
        # hold what it wants instead of spinning a wheel whose notch is not a
        # slot. The window is a whole cycle plus a dwell, because the leg can
        # arrive at any point in the rotation.
        #
        # `[-1]` IS LOAD-BEARING. `Say` writes to the pipeline, so `Wait-ForHero`
        # returns its own log lines AND its verdict; `$x = Wait-ForHero ...`
        # binds a non-empty ARRAY, which is truthy whatever the predicate said.
        $cycleS = [math]::Max(4.0, $ArmDwellS) * ($armList.Count + 1)
        $onIt = @(Wait-ForHero -Csv $heroCsv -What "`"$wid`" in the hand" -TimeoutS $cycleS `
            -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $wid) })[-1]
        if (-not $onIt) {
            Say "WPN2a: the rotation never put `"$wid`" in the hand inside $cycleS s -- no frame for it"
            continue
        }
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir ("8{0}-class-{1}-hud.png" -f $wi, $wid)) | ForEach-Object { Say $_ }
        # A semi-automatic weapon fires once per PRESS, so the trigger is
        # pressed and released repeatedly rather than held: a held button on a
        # Barrett is one round and a very long wait. Four attempts, and the
        # predicate names the WEAPON as well as the round so a frame cannot be
        # of somebody else's bullet.
        $flew = $false
        for ($t = 0; ($t -lt 4) -and (-not $flew); $t++) {
            [InfInput]::LeftDown(); Start-Sleep -Milliseconds 250
            [InfInput]::LeftUp()
            $flew = @(Wait-ForHero -Csv $heroCsv -What "a round in flight ($wid, press $($t + 1))" -TimeoutS 1.6 `
                -Predicate { param($c) ($c.Count -gt 22) -and ([int]$c[20] -gt 0) -and ($c[22].Trim() -eq $wid) } `
                -Out (Join-Path $OutDir ("8{0}-class-{1}-in-flight.png" -f $wi, $wid)))[-1]
        }
        $anyFlew = $anyFlew -or $flew
    }
    # THE IMPACT, and getting one is a design question rather than a timing one.
    # Column 21 is latched by the POOL when a round lands, so only a PROJECTILE
    # impact sets it -- a shot that resolves inside its hitscan threshold sets
    # nothing, which is right and is why three sessions of wave WPN2a fired down
    # a street and photographed no impact at all: at 25 m of threshold, a shop
    # front twenty metres away is the near half every time.
    #
    # So the impact is fired with the SNIPER, whose threshold is 10 m, along a
    # LEVEL aim down the street: past ten metres everything is a round in
    # flight, and a building a hundred metres away is what it arrives at. The
    # rotation brings it round again; this waits for it rather than spinning.
    $sn = "barrett_m82"
    $onSn = @(Wait-ForHero -Csv $heroCsv -What "`"$sn`" in the hand (for the impact)" -TimeoutS ([math]::Max(4.0, $ArmDwellS) * ($armList.Count + 1)) `
        -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $sn) })[-1]
    if (-not $onSn) { Say "WPN2a: the rotation never brought `"$sn`" back for the impact leg" }
    # LEVEL: down to the clamp, then back up by the clamp's own amount, for the
    # reason the leg aimed up that way — an absolute elevation, not a relative
    # one. 900 counts is past 89.9 degrees at the shipped sensitivity.
    [InfInput]::Look(0, 900)
    Start-Sleep -Milliseconds 300
    [InfInput]::Look(0, -430)
    Start-Sleep -Milliseconds 400
    for ($t = 0; $t -lt 10; $t++) {
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 250
        [InfInput]::LeftUp(); Start-Sleep -Milliseconds 250
        # Sweep the aim across the street between shots: one fixed heading can
        # be pointed at the sky over a junction, and a round that leaves the
        # partition hits nothing.
        [InfInput]::Look(60, 0)
    }
    Wait-ForHero -Csv $heroCsv -What "a round that hit something" -TimeoutS 10.0 `
        -Predicate { param($c) ($c.Count -gt 21) -and ([double]$c[21] -gt 0.0) } `
        -Out (Join-Path $OutDir "88-impact.png") | Out-Null
    [InfInput]::RightUp()
    Start-Sleep -Milliseconds 600
    & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "89-after-the-bursts.png") | ForEach-Object { Say $_ }
    if (-not $anyFlew) {
        Say "WPN2a: no round ever left the barrel -- the arming door, the trigger or the aim"
    }
    # What the log says about it, quoted into the demo log so a caption can be
    # read against a number rather than against a filename.
    if (Test-Path $heroCsv) {
        $armed = @(Get-Content $heroCsv | Where-Object { $_ -match "INF_PIE_ARM_HERO" })
        foreach ($l in $armed) { Say "  $l" }
        $rowsB = @(Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" } |
            ForEach-Object { $_.Split(",") } | Where-Object { $_.Count -gt 21 })
        if ($rowsB.Count -gt 0) {
            $peak = ($rowsB | ForEach-Object { [int]$_[20] } | Measure-Object -Maximum).Maximum
            $far = ($rowsB | ForEach-Object { [double]$_[21] } | Measure-Object -Maximum).Maximum
            Say "  peak rounds in flight $peak; furthest round that hit something $far m"
        }
    }
}

# -- 5e. THE CLASSES, THE MESHES AND THE BENCH (wave WPN2d) -------------------
#
#    Every frame here is TRIGGERED on `hero.csv`'s three NEW columns -- 29 is
#    what KIND of gun is in the hand, 30 is `n@x.xx` (how many attachments and
#    what the fold did to the report's loudness) and 31 is how far into a
#    lock-on a launcher is. None of the three can be timed with a `Start-Sleep`:
#    a lock takes 1.2-1.6 s and releases the instant the cone loses it, a
#    grenade's fuse is three seconds from a release the animation decides, and
#    what is in the hand is a rotation this leg does not drive.
#
#    The rotation is `-ArmHero`'s, which is a DEV door; the shipped route is the
#    island's own Quartermaster, which since this wave puts one weapon of every
#    class on the kerb the hero starts beside (`ISLAND_CLASS_COURSE`, nine rows,
#    asserted by `wpn2a_gate::the_island_puts_a_registry_weapon_on_the_kerb...`).
#    The env door is used here because a leg that walked nine pickups would
#    spend the whole session picking things up rather than photographing them.
if ($armList.Count -gt 0) {
    Say "-- WPN2d: the classes, their meshes, and the bench --"
    Restore-PlayerFocus "the classes leg"
    Stand-Up "before the classes leg" | Out-Null
    $cycleS = [math]::Max(4.0, $ArmDwellS) * ($armList.Count + 1)

    # **A FRAME OF A CORPSE IS NOT A FRAME OF A WEAPON IN A HAND** (WPN2d
    # audit). Every predicate in this leg used to ask only WHICH weapon was
    # equipped, and the equipped id keeps its value on a dead character. Wave
    # WPN2d's session 3 took all seven of its mesh frames of a hero that had
    # blown itself up with its own launcher at 2.204 m -- the RPG-7 spends
    # 4 000 J over an 8 m radius, so at 2.204 m it owes 2 099.6 J against a
    # 2 000 J hero -- and had been ragdolling for 130 s by then, 33 to 48 m
    # above the street. The captions said "an unmistakable AR-15 in the hero's
    # hands"; the pixels were a corpse against the sky.
    #
    # So every frame below asks the MODE as well, and the leg says out loud
    # when there is nothing worth photographing. `$alive` is the clause; it is
    # spelled once here and pasted into each predicate because a PowerShell
    # scriptblock parameter cannot be composed at the call site.
    $heroRow = { @(Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",") }
    $rowNow = & $heroRow
    if ($rowNow[5].Trim() -eq "Ragdoll") {
        Say "WPN2d: THE HERO IS RAGDOLLING -- it is dead or knocked down, and every"
        Say "       frame this leg would take is a photograph of a corpse. Nothing"
        Say ("       here is taken. mode={0} y={1} foot_mm={2} holder={3}" -f $rowNow[5], $rowNow[3], $rowNow[12], $rowNow[16])
    }

    # 1. FIRST PERSON, so the mesh is the frame rather than thirty pixels over a
    #    shoulder -- and so the wave's own first-person rule is what the pixels
    #    show: the BODY is faded to nothing at a 0.2 m boom and the WEAPON is
    #    not. Column 13 is the boom and 14 is how much of the body is drawn.
    # **LEVEL THE AIM FIRST.** The ballistics leg before this one leaves the look
    # pitched up so its rounds clear the street, and a first-person camera reads
    # the aim: session 2 of this wave took six mesh frames of the inside of a
    # building. Down to the clamp, then back up by a fixed amount, which is an
    # ABSOLUTE elevation rather than a relative one -- the recipe the ballistics
    # leg uses for the same reason.
    [InfInput]::Look(0, 900);  Start-Sleep -Milliseconds 300
    [InfInput]::Look(0, -430); Start-Sleep -Milliseconds 400
    # **THE VIEW-MODE KEY IS A TOGGLE**, so one press is a coin flip: the leg
    # before this one may have left the camera in first person, in which case a
    # single G puts it back on the boom. Session 1 of this wave pressed once,
    # read a 3.03 m boom and took no frame. Press, look, press again.
    $fp = $false
    for ($v = 0; ($v -lt 3) -and (-not $fp); $v++) {
        [InfInput]::Down(0x22); Start-Sleep -Milliseconds 150; [InfInput]::Up(0x22)   # G
        $fp = @(Wait-ForHero -Csv $heroCsv -What "a first-person seat with a gun in it (press $($v + 1))" -TimeoutS 5 `
            -Predicate { param($c) ($c.Count -gt 29) -and ([double]$c[13] -lt 0.35) -and ($c[29].Trim() -ne "-") -and ($c[5].Trim() -ne "Ragdoll") } `
            -Out (Join-Path $OutDir "90-first-person-weapon.png"))[-1]
    }
    if (-not $fp) { Say "WPN2d: no first-person seat with a weapon -- no un-faded frame" }
    else {
        $row = & $heroRow
        Say ("WPN2d: at boom {0} m the body draws {1} and the {2} in the hand does not fade" -f $row[13], $row[14], $row[29])
        # **AND LOOK FOR IT** (WPN2d audit). The first-person seat is at the
        # HEAD and the weapon is at the HAND socket, which is about half a metre
        # below it and a quarter of a metre forward -- some 63 degrees under the
        # horizon, which is outside a 60-degree vertical frustum unless the aim
        # is pitched well down. The audit's first session took `90` at a -23
        # degree aim and photographed a kerb: honest, and no evidence either way
        # about a weapon that does not fade. Two more frames, at both pitch
        # clamps, so the question is answered by pixels rather than by a
        # frustum.
        [InfInput]::Look(0, 900); Start-Sleep -Milliseconds 450
        $row = & $heroRow
        Say ("WPN2d: first person, pitch at one clamp: boom {0}, body {1}, class {2}" -f $row[13], $row[14], $row[29])
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "90b-first-person-weapon-down.png") | ForEach-Object { Say $_ }
        [InfInput]::Look(0, -900); Start-Sleep -Milliseconds 450
        $row = & $heroRow
        Say ("WPN2d: first person, pitch at the other clamp: boom {0}, body {1}, class {2}" -f $row[13], $row[14], $row[29])
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "90c-first-person-weapon-up.png") | ForEach-Object { Say $_ }
    }

    # **AND BACK OUT TO THE BOOM FOR THE CLASS FRAMES** (WPN2d audit). A
    # first-person frame is the right picture of the FADE RULE and the wrong
    # picture of a mesh: what "the class's own art is in the hero's hands" needs
    # is the hands in the frame. The wave took its seven class frames in
    # whatever seat the first-person attempt happened to leave behind, which
    # since the toggle never took was the third-person boom -- correct by
    # accident. This asks for it.
    if ($fp) {
        # **WAIT FOR THE BOOM, DO NOT COUNT PRESSES** (WPN2d audit, third
        # session). The arm blends out of the seat over about a second, so a
        # loop that pressed G, slept 600 ms and read a boom of 0.2 pressed it
        # again -- and a toggle pressed an odd number of times is back where it
        # started. Measured: the third session took every class frame at a
        # 0.17 m boom, which is the seat, not the boom.
        $back = $false
        for ($v = 0; ($v -lt 3) -and (-not $back); $v++) {
            [InfInput]::Down(0x22); Start-Sleep -Milliseconds 150; [InfInput]::Up(0x22)   # G
            $back = @(Wait-ForHero -Csv $heroCsv -What "the boom back out (press $($v + 1))" -TimeoutS 4 `
                -Predicate { param($c) ($c.Count -gt 13) -and ([double]$c[13] -gt 1.5) })[-1]
        }
        $row = & $heroRow
        if ($back) { Say ("WPN2d: back on the boom at {0} m for the class frames" -f $row[13]) }
        else { Say ("WPN2d: the boom never came back out -- the class frames are at {0} m" -f $row[13]) }
    }
    # **LEVEL THE AIM AGAIN**, so the class frames look down the street rather
    # than at the sky the first-person leg was pitched at.
    [InfInput]::Look(0, 900);  Start-Sleep -Milliseconds 300
    [InfInput]::Look(0, -440); Start-Sleep -Milliseconds 400

    # 2. ONE FRAME PER CLASS, in first person, waiting for the CLASS column
    #    rather than for a filename. A class the rotation never brings round
    #    takes no frame and says so.
    $seen = @()
    foreach ($wid in $armList) {
        $onIt = @(Wait-ForHero -Csv $heroCsv -What "`"$wid`" in the hand (the mesh), on a hero that is on its feet" -TimeoutS $cycleS `
            -Predicate { param($c) ($c.Count -gt 29) -and ($c[22].Trim() -eq $wid) -and ($c[5].Trim() -ne "Ragdoll") -and ($c[16].Trim() -eq "gameplay") })[-1]
        if (-not $onIt) { Say "WPN2d: no `"$wid`" in a living hand inside $cycleS s -- no mesh frame"; continue }
        $row = & $heroRow
        $cls = $row[29].Trim()
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir ("91-mesh-{0}-{1}.png" -f $cls, $wid)) | ForEach-Object { Say $_ }
        # The row the caption has to be read against: what is in the hand, what
        # the hero is doing, how far off the ground its feet are, and who is
        # holding the camera.
        Say ("WPN2d: {0} is class `"{1}`", attach {2}; mode {3}, foot_mm {4}, boom {5}, holder {6}" -f $wid, $cls, $row[30], $row[5], $row[12], $row[13], $row[16])
        $seen += $cls
    }
    Say ("WPN2d: classes photographed -> " + (($seen | Select-Object -Unique) -join ", "))

    # 3. THE PATTERN. A shotgun pull is eight rays out of one shell, so the frame
    #    that shows it is the one where the pellets are still in the air --
    #    column 20, on the shotgun.
    $sg = "remington_870"
    $onSg = @(Wait-ForHero -Csv $heroCsv -What "the shotgun in the hand" -TimeoutS $cycleS `
        -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $sg) -and ($c[5].Trim() -ne "Ragdoll") })[-1]
    if ($onSg) {
        [InfInput]::Look(0, -900); Start-Sleep -Milliseconds 250
        [InfInput]::Look(0, 250);  Start-Sleep -Milliseconds 300
        [InfInput]::RightDown(); Start-Sleep -Milliseconds 500
        $pat = $false
        for ($t = 0; ($t -lt 6) -and (-not $pat); $t++) {
            [InfInput]::LeftDown(); Start-Sleep -Milliseconds 220
            [InfInput]::LeftUp()
            $pat = @(Wait-ForHero -Csv $heroCsv -What "the pattern in the air (press $($t + 1))" -TimeoutS 1.4 `
                -Predicate { param($c) ($c.Count -gt 29) -and ($c[29].Trim() -eq "shotgun") -and ([int]$c[20] -gt 1) } `
                -Out (Join-Path $OutDir "92-shotgun-pattern.png"))[-1]
        }
        [InfInput]::RightUp()
        if (-not $pat) { Say "WPN2d: the shotgun never had two pellets in the air at once" }
    } else { Say "WPN2d: the rotation never brought the shotgun round for its pattern" }

    # 4. THE BLAST. The RPG's threshold is zero, so every shot is a body in
    #    flight and everything it touches goes off; the frame is the step the
    #    rocket is still flying, and the one after it is the bang.
    $rl = "rpg_7"
    $onRl = @(Wait-ForHero -Csv $heroCsv -What "the launcher in the hand" -TimeoutS $cycleS `
        -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $rl) -and ($c[5].Trim() -ne "Ragdoll") })[-1]
    if ($onRl) {
        [InfInput]::Look(0, 900);  Start-Sleep -Milliseconds 250
        [InfInput]::Look(0, -420); Start-Sleep -Milliseconds 300
        [InfInput]::RightDown(); Start-Sleep -Milliseconds 400
        $boom = $false
        for ($t = 0; ($t -lt 4) -and (-not $boom); $t++) {
            [InfInput]::LeftDown(); Start-Sleep -Milliseconds 220
            [InfInput]::LeftUp()
            @(Wait-ForHero -Csv $heroCsv -What "the rocket in flight (press $($t + 1))" -TimeoutS 1.6 `
                -Predicate { param($c) ($c.Count -gt 29) -and ($c[29].Trim() -eq "launcher") -and ([int]$c[20] -gt 0) } `
                -Out (Join-Path $OutDir "94-rocket-in-flight.png")) | Out-Null
            # The bang is where the rocket STOPS: column 21 latches the flight
            # distance of the last round that hit something.
            $boom = @(Wait-ForHero -Csv $heroCsv -What "the rocket arriving (press $($t + 1))" -TimeoutS 2.5 `
                -Predicate { param($c) ($c.Count -gt 21) -and ([double]$c[21] -gt 1.0) -and ([int]$c[20] -eq 0) } `
                -Out (Join-Path $OutDir "95-launcher-blast.png"))[-1]
            [InfInput]::Look(40, 0)
        }
        [InfInput]::RightUp()
        if (-not $boom) { Say "WPN2d: no rocket ever arrived -- no blast frame" }
    } else { Say "WPN2d: the rotation never brought the launcher round" }

    # 4b. THE LOCK, AND THE ONLY TWO ROWS IN THE REGISTRY THAT HAVE ONE (WPN2d
    #     audit). The wave swept eight bearings in each of two sessions with the
    #     RPG-7 in its hands and wrote down "nothing lockable was in the
    #     launcher's cone". The RPG-7 has no `lock_s` and no `lock_cone_deg`:
    #     `fim_92_stinger` and `javelin_fgm148` are the whole of the lock in
    #     `weapons.toml`, so that leg could not have produced a lock with
    #     anything in front of it. It also swept while pitched about 32 deg UP
    #     -- the elevation the leg above takes so its rocket clears the street --
    #     and a car is on the ground, outside a 10 deg cone.
    #
    #     So: a launcher that can lock, aimed LEVEL, swept for a car.
    $lockers = @($armList | Where-Object { $_ -eq "javelin_fgm148" -or $_ -eq "fim_92_stinger" })
    if ($lockers.Count -eq 0) {
        Say "WPN2d: no locking launcher in -ArmHero (only javelin_fgm148 and fim_92_stinger have a lock) -- no lock frame"
    } else {
        $lw = $lockers[0]
        $onLk = @(Wait-ForHero -Csv $heroCsv -What "`"$lw`" in the hand (the lock)" -TimeoutS $cycleS `
            -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $lw) -and ($c[5].Trim() -ne "Ragdoll") })[-1]
        if (-not $onLk) { Say "WPN2d: the rotation never brought `"$lw`" round for the lock" }
        else {
            # **HOLD STILL AND WATCH** (WPN2d audit, third session).
            #
            # Two sweeps found nothing and both were WRONG ABOUT THE WORLD. The
            # second session's own `hero.csv` carries **288 samples at `1.00+`**
            # -- a COMPLETED lock -- in twelve windows, and every window is a
            # dwell of the `-ArmHero` rotation in which the Javelin was in the
            # hand *and nothing was driving the look*. The two windows that
            # coincide with this leg's own sweep have no lock in them at all.
            #
            # The reason is arithmetic: a lock needs `lock_s` of CONTINUOUS hold
            # on the same target (1.6 s for the Javelin, 1.2 for the Stinger),
            # and a sweep that turns the aim every 1.5 s never lets one finish.
            # The leg was chasing a thing that only happens when you stop
            # chasing it.
            #
            # So: stand still, aim wherever the last leg left the reticle, and
            # WATCH the column for two full dwells. Only if that finds nothing
            # does the sweep run -- and then at 3 s a bearing, which is longer
            # than `lock_s`.
            [InfInput]::RightDown(); Start-Sleep -Milliseconds 400
            $lk = @(Wait-ForHero -Csv $heroCsv -What "a lock, without chasing it" -TimeoutS ([math]::Max(20.0, $ArmDwellS * 2)) `
                -Predicate { param($c) ($c.Count -gt 31) -and ($c[31].Trim() -ne "-") -and ([double]($c[31].TrimEnd("+")) -gt 0.05) } `
                -Out (Join-Path $OutDir "93-lock-on.png"))[-1]
            if (-not $lk) {
                Say "WPN2d: no lock while standing still -- sweeping, three seconds a bearing"
                foreach ($up in 260, 320, 380) {
                    if ($lk) { break }
                    [InfInput]::Look(0, 900);   Start-Sleep -Milliseconds 250
                    [InfInput]::Look(0, -$up);  Start-Sleep -Milliseconds 300
                    for ($b = 0; ($b -lt 8) -and (-not $lk); $b++) {
                        $lk = @(Wait-ForHero -Csv $heroCsv -What "a lock on something (elevation $up, bearing $b)" -TimeoutS 3.0 `
                            -Predicate { param($c) ($c.Count -gt 31) -and ($c[31].Trim() -ne "-") -and ([double]($c[31].TrimEnd("+")) -gt 0.05) } `
                            -Out (Join-Path $OutDir "93-lock-on.png"))[-1]
                        if (-not $lk) { [InfInput]::Look(300, 0); Start-Sleep -Milliseconds 200 }
                    }
                }
            }
            [InfInput]::RightUp()
            if ($lk) {
                $row = & $heroRow
                Say ("WPN2d: the lock indicator reads {0} with `"{1}`" up" -f $row[31], $lw)
            } else {
                Say "WPN2d: nothing lockable in two dwells of standing still, nor over three elevations x eight bearings -- no lock frame"
            }
        }
    }

    # 5. THE THROW. `KeyB` is the first key this engine has ever bound to one.
    #    The arc is drawn while AIMING with a throwable, so the frame before the
    #    press is the arc and the frame after the fuse is the detonation.
    $gr = "g67_grenade"
    $onGr = @(Wait-ForHero -Csv $heroCsv -What "the grenade in the hand" -TimeoutS $cycleS `
        -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $gr) -and ($c[5].Trim() -ne "Ragdoll") })[-1]
    if ($onGr) {
        [InfInput]::Look(0, -900); Start-Sleep -Milliseconds 250
        [InfInput]::Look(0, 300);  Start-Sleep -Milliseconds 300
        [InfInput]::RightDown(); Start-Sleep -Milliseconds 700
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "96-throw-arc.png") | ForEach-Object { Say $_ }
        [InfInput]::Down(0x30); Start-Sleep -Milliseconds 120; [InfInput]::Up(0x30)   # B: throw
        $flew = @(Wait-ForHero -Csv $heroCsv -What "the grenade in the air" -TimeoutS 6.0 `
            -Predicate { param($c) ($c.Count -gt 29) -and ([int]$c[20] -gt 0) } `
            -Out (Join-Path $OutDir "97-grenade-in-flight.png"))[-1]
        if (-not $flew) { Say "WPN2d: nothing left the hand on the throw key" }
        else {
            # The fuse is three seconds; the pool empties when it goes off.
            $off = @(Wait-ForHero -Csv $heroCsv -What "the grenade going off" -TimeoutS 8.0 `
                -Predicate { param($c) ($c.Count -gt 20) -and ([int]$c[20] -eq 0) } `
                -Out (Join-Path $OutDir "98-grenade-detonation.png"))[-1]
            if (-not $off) { Say "WPN2d: the grenade never left the pool -- no detonation frame" }
        }
        [InfInput]::RightUp()
    } else { Say "WPN2d: the rotation never brought the grenade round" }

    # 6. THE KNIFE. A melee swing is an ARC and a box cast; the frame is the
    #    swing, and the class column is what says a knife is in the hand.
    $kn = "m9_knife"
    $onKn = @(Wait-ForHero -Csv $heroCsv -What "the knife in the hand" -TimeoutS $cycleS `
        -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $kn) -and ($c[5].Trim() -ne "Ragdoll") })[-1]
    if ($onKn) {
        [InfInput]::Look(0, 900);  Start-Sleep -Milliseconds 250
        [InfInput]::Look(0, -430); Start-Sleep -Milliseconds 300
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 200; [InfInput]::LeftUp()
        Start-Sleep -Milliseconds 180
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "99-knife.png") | ForEach-Object { Say $_ }
    } else { Say "WPN2d: the rotation never brought the knife round" }

    # 7. THE BENCH. `I` opens the panel, `Tab` walks the rail and `]` fits the
    #    next part; column 30 is `n@x.xx`, so the frame is TRIGGERED on the fold
    #    having actually changed something rather than on a key having been
    #    pressed.
    $ar = "m4a1"
    $onAr = @(Wait-ForHero -Csv $heroCsv -What "the rifle in the hand (for the bench)" -TimeoutS $cycleS `
        -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -eq $ar) -and ($c[5].Trim() -ne "Ragdoll") })[-1]
    if ($onAr) {
        [InfInput]::Down(0x17); Start-Sleep -Milliseconds 300; [InfInput]::Up(0x17)   # I: the panel
        Start-Sleep -Milliseconds 400
        & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "9A-bench-panel.png") | ForEach-Object { Say $_ }
        # The muzzle is rail slot 0, which is where the cursor opens: `]` fits
        # the first row in it, and the first row of the AR's muzzles is the
        # tactical monolithic suppressor.
        $fitted = $false
        for ($t = 0; ($t -lt 3) -and (-not $fitted); $t++) {
            [InfInput]::Down(0x1B); Start-Sleep -Milliseconds 150; [InfInput]::Up(0x1B)   # ]
            $fitted = @(Wait-ForHero -Csv $heroCsv -What "an attachment fitted (press $($t + 1))" -TimeoutS 2.0 `
                -Predicate { param($c) ($c.Count -gt 30) -and ($c[30].Trim() -notmatch "^(-|0@)") } `
                -Out (Join-Path $OutDir "9B-attachment-fitted.png"))[-1]
        }
        [InfInput]::Down(0x17); Start-Sleep -Milliseconds 250; [InfInput]::Up(0x17)   # I: close it
        Start-Sleep -Milliseconds 400
        if ($fitted) {
            $row = @(Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
            Say ("WPN2d: the bench fitted something -- attach reads {0} (n@loudness)" -f $row[30])
            & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Out (Join-Path $OutDir "9C-attachment-on-the-rifle.png") | ForEach-Object { Say $_ }
        } else {
            Say "WPN2d: the bench never fitted anything -- attach stayed at 0@1.00"
        }
    } else { Say "WPN2d: the rotation never brought the rifle round for the bench" }

    [InfInput]::Down(0x22); Start-Sleep -Milliseconds 120; [InfInput]::Up(0x22)   # G: back to third person

    # What the session's own columns say, quoted where the captions are read.
    if (Test-Path $heroCsv) {
        $rowsD = @(Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" } |
            ForEach-Object { $_.Split(",") } | Where-Object { $_.Count -gt 31 })
        if ($rowsD.Count -gt 0) {
            $classes = ($rowsD | ForEach-Object { $_[29].Trim() } | Where-Object { $_ -ne "-" } | Select-Object -Unique) -join ", "
            $atts = ($rowsD | ForEach-Object { $_[30].Trim() } | Where-Object { $_ -ne "-" } | Select-Object -Unique) -join ", "
            $locks = ($rowsD | ForEach-Object { $_[31].Trim() } | Where-Object { $_ -ne "-" -and $_ -ne "0.00" } | Select-Object -Unique) -join ", "
            Say "  classes held : $classes"
            Say "  attach states: $atts"
            Say "  lock states  : $locks"
        }
    }
}


# ── 5e. THE SHOOTOUT (wave WPN2e) ────────────────────────────────────────────
#
#    The police fire back. Everything this leg photographs is downstream of ONE
#    player action -- firing a weapon in a street where somebody can see you --
#    and every frame is TRIGGERED on a column the sim writes:
#
#      col 33 `engaged`   how many responding units are pointing a weapon at
#                         somebody RIGHT NOW. It is the only honest trigger for
#                         "an officer is aiming at me": the police arrive over
#                         tens of seconds and the aim itself is a ray-gated
#                         decision that can go away between two screenshots.
#      col 34 `incoming`  rounds in the air the hero did NOT fire. "Somebody is
#                         shooting at me", as a number.
#      col 28 `casings`   brass on the ground.
#      col 6  `mode`      `Cover` for the cover frame.
#
#    (One-based; `$c[..]` below is zero-based, so `engaged` is `$c[32]`.)
#
#    THE CHAIN IT DRIVES, and every link is shipped: a loud shot is witnessed by
#    whoever can see it (`step_witness`), the act opens a criminal profile keyed
#    on a DESCRIPTION (`crime::report_act`), the heat picks a rung
#    (`Response::for_heat`), the dispatcher sends the nearest free unit by route
#    cost, the crew gets out at the scene and is ISSUED a weapon
#    (`d3::engage::arm_crew`), and the firing policy points it at the person on
#    the file -- with line of sight, on a cadence, and never through a colleague
#    or a bystander.
#
#    It runs LAST, deliberately: it makes the hero wanted, and a wanted hero
#    being shot at is not the state any other leg wants to photograph.
Say "-- SHOOTOUT (WPN2e): make yourself wanted, and see who turns up --"
Restore-PlayerFocus "before the shootout"
$shootFrames = 0
# 1. SOMETHING IN THE HAND. The rotation may already have put one there; if not,
#    the island's own kerb pickup is the door (leg 5a's, without its frames).
$armed = @(Wait-ForHero -Csv $heroCsv -What "a weapon in the hand (for the shootout)" -TimeoutS 6.0 `
    -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -ne "-") -and ($c[22].Trim() -ne "bandage") })[-1]
if (-not $armed) {
    for ($k = 0; ($k -lt 8) -and (-not $armed); $k++) {
        [InfInput]::Down(0x12); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x12)   # E
        Start-Sleep -Milliseconds 200
        [InfInput]::Wheel(1)
        $armed = @(Wait-ForHero -Csv $heroCsv -What "a weapon in the hand (try $($k + 1))" -TimeoutS 1.2 `
            -Predicate { param($c) ($c.Count -gt 22) -and ($c[22].Trim() -ne "-") -and ($c[22].Trim() -ne "bandage") })[-1]
    }
}
if (-not $armed) {
    Say "SHOOTOUT: nothing in the hand, so there is no crime to commit -- skipped"
}
else {
    # `Wait-ForHero` returns a BOOLEAN -- it says whether the predicate fired,
    # not which row fired it -- so the row has to be read back from the CSV. The
    # first cut of this leg indexed the boolean, and `$armed[22].Trim()` threw
    # "you cannot call a method on a null-valued expression" into the middle of
    # an otherwise clean session.
    $armedRow = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
    Say ("SHOOTOUT: firing {0} in the street to open a file" -f $armedRow[22].Trim())
    # 2. THE CRIME. A loud shot is worth two heat and `Response::for_heat` puts
    #    three at `MultiUnit` and six at `Swat`, so a handful of trigger pulls is
    #    a tactical response -- IF somebody saw them. The island's own crowd is
    #    the witness; `WITNESS_RADIUS_M` is 120 m and the ray has to clear.
    #    Level the aim first: a shot into the ground still makes the noise, but a
    #    shot down the street is the one a pedestrian's line of sight reaches.
    [InfInput]::Look(0, -40)
    Start-Sleep -Milliseconds 250
    for ($t = 0; $t -lt 10; $t++) {
        [InfInput]::LeftDown(); Start-Sleep -Milliseconds 260
        [InfInput]::LeftUp(); Start-Sleep -Milliseconds 180
    }
    $brass = @(Wait-ForHero -Csv $heroCsv -What "brass on the ground after the shots" -TimeoutS 4.0 `
        -Predicate { param($c) ($c.Count -gt 27) -and ([int]$c[27] -gt 0) } `
        -Out (Join-Path $OutDir "A1-shootout-casings.png"))[-1]
    if ($brass) { $shootFrames++ } else { Say "SHOOTOUT: no brass on the ground after ten pulls" }

    # 3. THE RESPONSE. The units have to DRIVE, so this is the long wait, and it
    #    is on `engaged` rather than on a clock: an officer that has arrived and
    #    has no line of sight is not a frame of an officer aiming at you.
    #
    #    **AND THE HERO KEEPS FIRING WHILE IT WAITS** (WPN2e audit), which is not
    #    decoration. `inf_ecs::engage::TRAIL_STALE_STEPS` is 180 steps -- THREE
    #    SECONDS -- and the whole policy refuses a pair whose `last_seen` is
    #    older than that, correctly: a suspect nobody has seen for three seconds
    #    is a SEARCH and not a target. A file opened by EAR carries no
    #    description either (you cannot describe a bang), so nothing on the
    #    recognition path can refresh it. The first cut of this leg fired ten
    #    rounds and then stood still for a hundred and fifty seconds, and what it
    #    was actually asking for was an officer that shoots a man it has no idea
    #    is there. Measured then: `peak heat 36, peak units on scene 1, peak
    #    engaged 0` -- the town heard it, the car arrived, and the policy
    #    correctly refused.
    #
    #    So the wait is a firefight. A burst every few seconds, `R` when the
    #    magazine runs dry, which is what a player being shot at actually does.
    $engaged = $false
    for ($w = 0; ($w -lt 30) -and (-not $engaged); $w++) {
        for ($t = 0; $t -lt 4; $t++) {
            [InfInput]::LeftDown(); Start-Sleep -Milliseconds 200
            [InfInput]::LeftUp(); Start-Sleep -Milliseconds 140
        }
        [InfInput]::Down(0x52); Start-Sleep -Milliseconds 90; [InfInput]::Up(0x52)   # R
        $engaged = @(Wait-ForHero -Csv $heroCsv -What "a responding unit with its weapon on the hero (col 33)" -TimeoutS 3.0 `
            -Predicate { param($c) ($c.Count -gt 32) -and ([int]$c[32] -gt 0) } `
            -Out (Join-Path $OutDir "A2-officer-aiming.png"))[-1]
    }
    if ($engaged) {
        $shootFrames++
        $engRow = (Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
        Say ("SHOOTOUT: {0} unit(s) engaged" -f $engRow[32])
        # 4. THE HERO UNDER FIRE. `incoming` counts rounds in the air that the
        #    hero did not fire, which is what "they are shooting back" is.
        $incoming = @(Wait-ForHero -Csv $heroCsv -What "rounds in the air the hero did not fire (col 34)" -TimeoutS 40.0 `
            -Predicate { param($c) ($c.Count -gt 33) -and ([int]$c[33] -gt 0) } `
            -Out (Join-Path $OutDir "A3-under-fire.png"))[-1]
        if ($incoming) { $shootFrames++ } else { Say "SHOOTOUT: nobody fired at the hero inside forty seconds" }

        # 5. RETURNING FIRE, with the officers in the frame. The trigger is the
        #    officers still being engaged while the hero's own magazine moves.
        for ($t = 0; $t -lt 8; $t++) {
            [InfInput]::LeftDown(); Start-Sleep -Milliseconds 240
            [InfInput]::LeftUp(); Start-Sleep -Milliseconds 120
        }
        $returned = @(Wait-ForHero -Csv $heroCsv -What "the hero returning fire with units engaged" -TimeoutS 6.0 `
            -Predicate { param($c) ($c.Count -gt 32) -and ([int]$c[32] -gt 0) -and ([int]$c[27] -gt 0) } `
            -Out (Join-Path $OutDir "A4-returning-fire.png"))[-1]
        if ($returned) { $shootFrames++ }

        # 6. COVER, AND THE BLIND SHOT. `T` is the cover key; a hero in cover
        #    with the trigger down and the AIM RELEASED is firing blind -- the
        #    branch is a behaviour and not a binding, so there is nothing else to
        #    press. The frame is triggered on the MODE, and the blind shot is
        #    fired with the right button up.
        [InfInput]::Down(0x14); Start-Sleep -Milliseconds 160; [InfInput]::Up(0x14)   # T
        $inCover = @(Wait-ForHero -Csv $heroCsv -What "the hero in cover, under fire" -TimeoutS 4.0 `
            -Predicate { param($c) ($c.Count -gt 5) -and ($c[5].Trim() -eq "Cover") } `
            -Out (Join-Path $OutDir "A5-cover-under-fire.png"))[-1]
        if ($inCover) {
            $shootFrames++
            for ($t = 0; $t -lt 6; $t++) {
                [InfInput]::LeftDown(); Start-Sleep -Milliseconds 240
                [InfInput]::LeftUp(); Start-Sleep -Milliseconds 120
            }
            $blind = @(Wait-ForHero -Csv $heroCsv -What "a blind shot from cover (in cover, brass in the air)" -TimeoutS 4.0 `
                -Predicate { param($c) ($c.Count -gt 27) -and ($c[5].Trim() -eq "Cover") -and ([int]$c[27] -gt 0) } `
                -Out (Join-Path $OutDir "A6-blind-fire.png"))[-1]
            if ($blind) { $shootFrames++ } else { Say "SHOOTOUT: in cover and no brass -- the blind branch took no shot" }
            # Out of cover again, so the closing distance measurement is a walk.
            [InfInput]::Down(0x11); Start-Sleep -Milliseconds 400; [InfInput]::Up(0x11)   # W, away from the wall
        }
        else { Say "SHOOTOUT: T pressed and nothing coverable was in reach" }
    }
    else {
        # **THE FOUR NUMBERS, NOT ONE** (WPN2e audit). `engaged 0` is what you
        # get when nobody heard the shot, when the file went cold, when the car
        # never arrived, and when it arrived and could not see you. The columns
        # below are what tell them apart -- see tools/demo/README.md.
        $last = @(Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" })[-1].Split(",")
        Say ("SHOOTOUT: no unit engaged -- heat {0}, units on scene {1}, nearest responder {2} m, casings {3}" -f `
            $last[34], $last[35], $last[36], $last[27])
    }

    # What the session's own columns say about the shootout, quoted.
    if (Test-Path $heroCsv) {
        # THE UNARY COMMA IS LOAD-BEARING: `ForEach-Object { $_.Split(",") }`
        # emits the array's ELEMENTS, one per field, so the `Where-Object` below
        # would test a single string's `.Count` (1) and this summary printed
        # nothing at all -- silently -- for two whole sessions. `, $_.Split(",")`
        # emits the row as ONE object.
        $rowsE = @(Get-Content $heroCsv | Where-Object { $_ -match "^[0-9]" } |
            ForEach-Object { , $_.Split(",") } | Where-Object { $_.Count -gt 33 })
        if ($rowsE.Count -gt 0) {
            $peakEng = ($rowsE | ForEach-Object { [int]$_[32] } | Measure-Object -Maximum).Maximum
            $peakInc = ($rowsE | ForEach-Object { [int]$_[33] } | Measure-Object -Maximum).Maximum
            $peakBrass = ($rowsE | ForEach-Object { [int]$_[27] } | Measure-Object -Maximum).Maximum
            $peakHeat = ($rowsE | ForEach-Object { [int]$_[34] } | Measure-Object -Maximum).Maximum
            $peakScene = ($rowsE | ForEach-Object { [int]$_[35] } | Measure-Object -Maximum).Maximum
            Say "  peak engaged units : $peakEng"
            Say "  peak incoming rounds: $peakInc"
            Say "  peak casings alive : $peakBrass"
            # **THE WANTED CHAIN, AS FOUR NUMBERS** (WPN2e audit). `engaged 0` is
            # the same number for four different bugs; these are what tell them
            # apart. See tools/demo/README.md.
            Say "  peak heat on the hero: $peakHeat"
            $nearest = ($rowsE | ForEach-Object { [double]$_[36] } | Where-Object { $_ -ge 0 } | Measure-Object -Minimum).Minimum
            Say "  peak units on scene : $peakScene"
            Say ("  nearest responder   : {0:N1} m (engage range 35 m)" -f $nearest)
            Say "  shootout frames    : $shootFrames of 6"
        }
    }
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

# ── 6z. WAVE VEH3a — THE TYRES, AND WHAT THE GROUND UNDER THEM IS ────────────
#
# Four frames, every one TRIGGERED on `hero.csv`'s eight new columns rather than
# slept for. Indices, zero-based, and they are at the TAIL so every index above
# keeps its meaning:
#
#   37..40  tyre_temp_fl / fr / rl / rr, Celsius
#   41      surface   — the class the most wheels are on
#   42      slip_ratio, 43 slip_lat   — the DRIVEN axle's
#   44      mu        — what this contact is worth
#
# None of these is visible in a position column, which is exactly why they were
# appended: "the car is on grass", "the tyres are cooked" and "the wheel is
# spinning" are three different things that all read the same in `speed`.
Restore-PlayerFocus "before the tyre leg"
# **STAND UP FIRST**, for the reason the camera leg above states and measures: a
# session that has been through the prone leg is CROUCHED, and a crouched hero
# does not board. Measured on the first run of this leg -- the hero stood 2.8 m
# from `Harbour City Car` at (-1752.0, 16.8, 2048.0) and twenty-five taps of E
# reached nothing, with `mode` reading `Crouch` on 801 of 1 239 rows.
Stand-Up "before the tyre leg" | Out-Null
Say "VEH3a: boarding a car and driving it, watching the tyre row"

# (a) THE HUD ROW ON THE ROAD. Board first, then hold the throttle until the
#     world reports a car on asphalt at speed. The row is under the instruments.
# **THE HUNT SWEEPS A DISC, NOT A LINE** (wave VEH3b, closing the VEH3a
# audit's carried instrument item). The wave's hunt tapped E and walked `W`
# between taps, so twenty-five taps carried the hero twenty metres along ONE
# bearing -- and `ENTER_REACH_M` is 3.0 m, so a car four metres to either side
# was never in reach. Measured: three sessions across two waves reached a car
# twice, and both misses ended with the hero tens of metres from where it
# started, walking away from the parked cars. Turning thirty degrees every
# third tap makes the same twenty-five taps a rough circle instead of a
# corridor, which is what "look around for a car" means.
$veh_driving = $false
for ($k = 0; $k -lt 36 -and -not $veh_driving; $k++) {
    [InfInput]::Down(0x12); Start-Sleep -Milliseconds 70; [InfInput]::Up(0x12)   # E
    $veh_driving = @(Wait-ForHero -Csv $heroCsv -What "the hero at a wheel (VEH3a)" -TimeoutS 0.9 `
        -Predicate { param($c) ($c.Count -gt 44) -and ($c[5] -eq "Driving") })[-1]
    if (-not $veh_driving -and $k -ge 3) {
        if (($k % 3) -eq 0) {
            # Thirty degrees at the shipped sensitivity, the cover leg's own
            # arithmetic: 15 counts is about 2.2 degrees.
            for ($i = 0; $i -lt 14; $i++) { [InfInput]::Look(15, 0); Start-Sleep -Milliseconds 16 }
        }
        [InfInput]::Down(0x11); Start-Sleep -Milliseconds 300; [InfInput]::Up(0x11)   # W
    }
}
if (-not $veh_driving) {
    # **NAME THE MODE THE HUNT RAN IN** (VEH3b audit): `try_enter` refuses from
    # any mode that is not grounded, so a hero left `Ragdoll` by an earlier leg
    # cannot board however well the hunt sweeps -- and the message that only
    # counted taps sent two audits looking at the hunt.
    $huntMode = "?"
    $rowsHunt = @(Get-Content $heroCsv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" })
    if ($rowsHunt.Count -gt 0) { $huntMode = $rowsHunt[-1].Split(",")[5] }
    $ragdolled = @($rowsHunt | Where-Object { ($_ -split ",")[5] -eq "Ragdoll" }).Count
    Say "VEH3a: NO CAR reached in thirty-six taps of E over a full turn -- none of the four tyre frames is in this session (the hero was '$huntMode'; $ragdolled of $($rowsHunt.Count) rows in this session were Ragdoll)"
} else {
    # Throttle, and let the surface classifier answer.
    [InfInput]::Down(0x11)
    $veh_road = @(Wait-ForHero -Csv $heroCsv -What "the tyre row on the road (surface asphalt, moving)" -TimeoutS 20.0 `
        -Predicate { param($c) ($c.Count -gt 44) -and ($c[5] -eq "Driving") -and ($c[41] -eq "asphalt") -and ([double]$c[6] -gt 3.0) } `
        -Out (Join-Path $OutDir "90-veh3a-hud-road.png"))[-1]
    if (-not $veh_road) {
        Say "VEH3a: the car never reported ASPHALT while moving -- frame (a) is not in this session"
    }

    # (b) THE VERGE. Steer off the carriageway and wait for the class to flip.
    #     The car slows because the ground is worth less: both halves are in the
    #     row, and the predicate asks for the surface, not for the slowing, so a
    #     car that flipped class without losing speed still trips it and the
    #     speed is in the captured line for a reader to check.
    Say "VEH3a: steering off the carriageway"
    # **A LONG steer, and the throttle stays down.** Measured on the first run:
    # 1.4 s of D at 3 m/s is four metres of lateral travel and the island's
    # carriageway is wider than that, so the surface never left `asphalt`. This
    # holds the wheel over for six seconds and lets the car build speed first.
    Start-Sleep -Milliseconds 2500
    [InfInput]::Down(0x20); Start-Sleep -Milliseconds 6000; [InfInput]::Up(0x20)   # D
    $veh_verge = @(Wait-ForHero -Csv $heroCsv -What "the verge (surface leaves asphalt)" -TimeoutS 20.0 `
        -Predicate { param($c) ($c.Count -gt 44) -and ($c[5] -eq "Driving") -and ($c[41] -ne "asphalt") -and ($c[41] -ne "-") } `
        -Out (Join-Path $OutDir "91-veh3a-verge.png"))[-1]
    if (-not $veh_verge) {
        # **THE OTHER WAY** (VEH3a's audit). One six-second steer is one
        # direction, and which side of the carriageway the car is on when the
        # leg starts is not something this script knows. The first run of the
        # wave held D and never left the road; holding A is the same leg
        # mirrored, and between them the car has crossed the whole carriageway.
        Say "VEH3a: still on asphalt -- steering the OTHER way"
        [InfInput]::Down(0x1E); Start-Sleep -Milliseconds 7000; [InfInput]::Up(0x1E)   # A
        $veh_verge = @(Wait-ForHero -Csv $heroCsv -What "the verge (the other way)" -TimeoutS 20.0 `
            -Predicate { param($c) ($c.Count -gt 44) -and ($c[5] -eq "Driving") -and ($c[41] -ne "asphalt") -and ($c[41] -ne "-") } `
            -Out (Join-Path $OutDir "91-veh3a-verge.png"))[-1]
    }
    if (-not $veh_verge) {
        Say "VEH3a: the car never left ASPHALT -- frame (b) is not in this session"
    }

    # (c) THE BURNOUT. Brake and throttle together on whatever the car is
    #     standing on, and watch the four temperatures climb. The hero's car has
    #     the catalogue's ROAD tyres -- this loop does not re-tune it, so the
    #     number below is what a road car on the island's own ground really
    #     reaches, and 60 C is asked for first so a miss is reported rather than
    #     dressed down.
    # **AND THE COMPOUND IS AN OPERATOR'S SWITCH NOW** (VEH3a's audit). The
    # wave's session could not fire this frame and said so honestly: on asphalt
    # under ROAD tyres a line-lock burnout is a parked car, because this rig's
    # brakes out-hold its engine. `-TuneVehicle "tyre_surface_set=3"` puts the
    # SLICK row on every chassis through `VehicleClass::set` -- the same by-name
    # door an authored catalogue uses -- and then the drive really does outrun
    # the grip. A session that did not ask for it still reports the miss.
    Say "VEH3a: burnout -- brake and throttle together"
    [InfInput]::Down(0x1F)   # S, the brake
    $veh_hot = @(Wait-ForHero -Csv $heroCsv -What "the tyres past 60 C" -TimeoutS 25.0 `
        -Predicate { param($c) ($c.Count -gt 44) -and (([double]$c[39] -gt 60.0) -or ([double]$c[40] -gt 60.0)) } `
        -Out (Join-Path $OutDir "92-veh3a-burnout.png"))[-1]
    if (-not $veh_hot) {
        Say "VEH3a: no tyre passed 60 C -- asking instead for ANY heating above the 20 C ambient"
        $veh_warm = @(Wait-ForHero -Csv $heroCsv -What "a tyre warmer than the air" -TimeoutS 15.0 `
            -Predicate { param($c) ($c.Count -gt 44) -and (([double]$c[39] -gt 21.5) -or ([double]$c[40] -gt 21.5)) } `
            -Out (Join-Path $OutDir "92-veh3a-burnout.png"))[-1]
        if (-not $veh_warm) {
            Say "VEH3a: the tyres never warmed at all -- and on ASPHALT under ROAD tyres that is the MODEL, not a miss: this rig's brakes out-hold its engine (13 kN against 8), so a line-lock burnout is a parked car. `veh3a_gate::a_burnout_heats_the_tyre_and_costs_it_grip` spins one on SAND under SLICK tyres, where mu is 0.25, and measures 229 slipping steps taking a tyre 20.0 -> 21.1 C. Frame (c) is not in this session and the reason is a number."
        }
    }
    [InfInput]::Up(0x1F)

    # (d) THE KERB. Drive at a kerb and catch the step the suspension takes. The
    #     row carries no vertical acceleration, so the trigger is the thing that
    #     IS in it and means the same: a driven wheel losing and regaining its
    #     contact hard enough to swing the slip.
    Say "VEH3a: back on the throttle, looking for a kerb"
    [InfInput]::Down(0x1E); Start-Sleep -Milliseconds 2200; [InfInput]::Up(0x1E)   # A, back toward the kerb
    Start-Sleep -Milliseconds 2500
    $veh_kerb = @(Wait-ForHero -Csv $heroCsv -What "the kerb (the driven axle's slip swings)" -TimeoutS 20.0 `
        -Predicate { param($c) ($c.Count -gt 44) -and ($c[5] -eq "Driving") -and ([math]::Abs([double]$c[42]) -gt 0.25) } `
        -Out (Join-Path $OutDir "93-veh3a-kerb.png"))[-1]
    if (-not $veh_kerb) {
        Say "VEH3a: the driven axle never swung past 0.25 of slip -- frame (d) is not in this session"
    }
    [InfInput]::Up(0x11)

    # WHAT THE ROW ACTUALLY SAID, whatever fired: the last driving line, printed
    # whole, so a reader can see the eight columns rather than take the triggers'
    # word for them.
    $last = @(Get-Content $heroCsv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" } |
        Where-Object { ($_ -split ",").Count -gt 44 -and ($_ -split ",")[5] -eq "Driving" })
    if ($last.Count -gt 0) {
        $c = $last[-1] -split ","
        Say ("VEH3a ROW: temps {0}/{1}/{2}/{3} C  surface {4}  slip {5}/{6}  mu {7}" -f `
            $c[37], $c[38], $c[39], $c[40], $c[41], $c[42], $c[43], $c[44])
    } else {
        Say "VEH3a ROW: no driving row was ever written"
    }
}

# ── 6z2. WAVE VEH3b — THE DRIVETRAIN, AND WHAT ITS FOUR COLUMNS SEE ──────────
#
# Five frames, every one TRIGGERED on `hero.csv`'s four new columns rather than
# slept for. Indices, zero-based, at the TAIL so every index above keeps its
# meaning:
#
#   45  rpm      — the crank's own speed, a STATE since this wave
#   46  clutch   — its engagement, `[0, 1]`; below 1 means it is slipping
#   47  boost    — the turbo, `[0, 1]` of the class's own peak
#   48  cut      — `1` while the limiter is cutting fuel
#
# None of the four is visible in a position column, which is why they were
# appended: "the clutch is biting", "the turbo is spooling" and "the limiter is
# bouncing" are three different things that all read the same in `speed`.
#
# **The car has to be MOVING for any of them**, so this leg runs after the tyre
# leg and reuses whatever the hero is already sitting in. A session where the
# tyre leg never reached a car reports that and takes none of the five.
if (-not $veh_driving) {
    Say "VEH3b: NO CAR was reached above -- none of the five drivetrain frames is in this session"
} else {
    Say "VEH3b: the crank, the clutch, the limiter and the turbo"

    # (a) THE LAUNCH. Off the throttle until the car is stopped, then floor it:
    #     the clutch is OPEN on a parked car and bites over `clutch_engage_s`,
    #     so the frame is the one moment a driver feels the car take up.
    [InfInput]::Up(0x11)
    Start-Sleep -Milliseconds 2500
    [InfInput]::Down(0x11)
    $veh_launch = @(Wait-ForHero -Csv $heroCsv -What "the clutch biting at a standing start (VEH3b)" -TimeoutS 12.0 `
        -Predicate { param($c) ($c.Count -gt 48) -and ($c[5] -eq "Driving") -and ([double]$c[46] -lt 0.95) -and ([double]$c[6] -lt 6.0) } `
        -Out (Join-Path $OutDir "94-veh3b-launch.png"))[-1]
    if (-not $veh_launch) {
        Say "VEH3b: the clutch never reported a bite at a standing start -- frame (a) is not in this session"
    }

    # (b) THE SHIFT. The same column, at speed: a gearshift OPENS the clutch and
    #     re-engages it over `clutch_engage_s`, which is the flare.
    $veh_shift = @(Wait-ForHero -Csv $heroCsv -What "the clutch re-engaging after a shift (VEH3b)" -TimeoutS 20.0 `
        -Predicate { param($c) ($c.Count -gt 48) -and ($c[5] -eq "Driving") -and ([double]$c[46] -lt 0.95) -and ([double]$c[6] -gt 8.0) } `
        -Out (Join-Path $OutDir "95-veh3b-shift.png"))[-1]
    if (-not $veh_shift) {
        Say "VEH3b: the clutch never re-engaged at speed -- frame (b) is not in this session"
    }

    # (c) THE LIMITER. The rev limiter CUTS now and the cut is a column, so this
    #     asks the world for it directly. On the island's own roads the hero's
    #     car reaches its governor before its redline -- the two are different
    #     ceilings and `governor` is the one that binds on a road car -- so a
    #     miss here is a fact about the class rather than about the model, and
    #     the `veh3b_gate::the_limiter_cuts_and_restores` arm measures the saw
    #     on a rig that CAN reach it (27 cut edges, 153 rpm of amplitude over a
    #     5.8-step period).
    $veh_cut = @(Wait-ForHero -Csv $heroCsv -What "the limiter cutting fuel (VEH3b)" -TimeoutS 20.0 `
        -Predicate { param($c) ($c.Count -gt 48) -and ($c[48] -eq "1") } `
        -Out (Join-Path $OutDir "96-veh3b-redline.png"))[-1]
    if (-not $veh_cut) {
        Say "VEH3b: the limiter never cut -- the island's car reaches its GOVERNOR (a road-speed limiter) before its redline, and frame (c) is not in this session"
    }

    # (d) THE NOSE-DIVE. The row carries no suspension compression, so the
    #     trigger is the thing that IS in it and happens at the same instant: a
    #     braked DOWNSHIFT re-engages the clutch while the car is slowing hard,
    #     which is exactly when the nose is at its lowest.
    Say "VEH3b: hard on the brakes, looking for a downshift"
    [InfInput]::Up(0x11)
    [InfInput]::Down(0x1F)   # S, the brake
    $veh_dive = @(Wait-ForHero -Csv $heroCsv -What "a braked downshift, the nose at its lowest (VEH3b)" -TimeoutS 12.0 `
        -Predicate { param($c) ($c.Count -gt 48) -and ($c[5] -eq "Driving") -and ([double]$c[46] -lt 0.95) -and ([double]$c[6] -gt 3.0) } `
        -Out (Join-Path $OutDir "97-veh3b-nosedive.png"))[-1]
    [InfInput]::Up(0x1F)
    if (-not $veh_dive) {
        Say "VEH3b: no downshift came under the brakes -- frame (d) is not in this session"
    }

    # (e) THE BOOST. Every catalogue row shipped today is NATURALLY ASPIRATED
    #     (`turbo_boost_max` is 0 on all eleven — `veh3b_gate` asserts it), so
    #     this frame exists only when the session asked for a turbo through the
    #     preview-only tuning door: `-TuneVehicle "turbo_boost_max=0.8"`. A
    #     session that did not ask reports the miss and its reason.
    [InfInput]::Down(0x11)
    $veh_boost = @(Wait-ForHero -Csv $heroCsv -What "the turbo on boost (VEH3b)" -TimeoutS 15.0 `
        -Predicate { param($c) ($c.Count -gt 48) -and ([double]$c[47] -gt 0.30) } `
        -Out (Join-Path $OutDir "98-veh3b-boost.png"))[-1]
    if (-not $veh_boost) {
        Say "VEH3b: no boost -- every catalogue row is naturally aspirated, so frame (e) needs `-TuneVehicle `"turbo_boost_max=0.8`"` and this session did not ask"
    }
    [InfInput]::Up(0x11)

    # WHAT THE FOUR COLUMNS ACTUALLY SAID, whatever fired: the last driving
    # line, printed whole, so a reader can see them rather than take the
    # triggers' word for it.
    $last = @(Get-Content $heroCsv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" } |
        Where-Object { ($_ -split ",").Count -gt 48 -and ($_ -split ",")[5] -eq "Driving" })
    if ($last.Count -gt 0) {
        $c = $last[-1] -split ","
        Say ("VEH3b ROW: {0} rpm  clutch {1}  boost {2}  cut {3}" -f $c[45], $c[46], $c[47], $c[48])
        $slipping = @($last | Where-Object { [double](($_ -split ",")[46]) -lt 0.95 }).Count
        $cutting = @($last | Where-Object { (($_ -split ",")[48]) -eq "1" }).Count
        $maxrpm = ($last | ForEach-Object { [double](($_ -split ",")[45]) } | Measure-Object -Maximum).Maximum
        Say ("VEH3b CENSUS: {0} driving rows, {1} with the clutch slipping, {2} with the limiter cutting, peak {3} rpm" -f $last.Count, $slipping, $cutting, $maxrpm)
    } else {
        Say "VEH3b ROW: no driving row was ever written"
    }
}

# ── 6z3. WAVE VEH3c — THE MODULAR BODY, AND WHAT ITS FIVE COLUMNS SEE ────────
#
# Five frames, every one TRIGGERED on `hero.csv`'s five new columns rather than
# slept for. Indices, zero-based, at the TAIL so every index above keeps its
# meaning:
#
#   49  car_health    — the hull's remaining percent
#   50  engine_scale  — the engine's remaining percent
#   51  flats         — how many tyres are flat
#   52  panes_broken  — how many windows have gone
#   53  parts_shed    — how many parts have left the car
#
# None of the five is visible in a position column, and none of them is visible
# in the four VEH3b appended either: "the bumper came off", "the windscreen
# went", "the near-side front is flat" and "the engine is dead" are four
# different things that all read the same in `speed` and in `rpm`.
#
# **THE COLUMNS READ THE CAR THE HERO IS SITTING IN**, which is what "the
# hero's car damage" means and is the same rule the four drivetrain columns
# follow. A car the hero shoots at from the pavement is damaged in the WORLD and
# is not in this row — so the shot-up-car frame below is taken from the driver's
# seat of a car that has been hurt, not from outside one.
#
# **The car has to be MOVING for any of them**, so this leg runs after the
# drivetrain leg and reuses whatever the hero is already sitting in. A session
# where the tyre leg never reached a car reports that and takes none of the five.
if (-not $veh_driving) {
    Say "VEH3c: NO CAR was reached above -- none of the five bodywork frames is in this session"
} else {
    Say "VEH3c: the bodywork -- a crash, a shed part, a pane, a flat and a fire"

    # (a) THE CRASH. Full throttle, held, into whatever the island has that a
    #     car can hit. **The island's BUILDINGS carry no ECS collider** (carried
    #     since COV1), so what stops a car here is a kerb, a pillar, a lightpost
    #     module or another parked vehicle -- the crash arm on the FIXTURE
    #     (`veh3c_gate::a_crash_at_sixty_sheds_the_bumper_and_pops_the_bonnet`)
    #     is where a 60 km/h wall is, and it measures 18 879 N.s, one bumper
    #     shed and the bonnet popped.
    Say "VEH3c: hard ahead, looking for something to hit"
    [InfInput]::Down(0x11)
    $veh_hurt = @(Wait-ForHero -Csv $heroCsv -What "the hull taking damage (VEH3c)" -TimeoutS 30.0 `
        -Predicate { param($c) ($c.Count -gt 53) -and ($c[5] -eq "Driving") -and ([double]$c[49] -lt 99.9) -and ([double]$c[49] -gt 0.0) } `
        -Out (Join-Path $OutDir "99-veh3c-hull.png"))[-1]
    if (-not $veh_hurt) {
        Say "VEH3c: the hull never took a blow -- the hero's car found nothing on this street with a collider on it, and frame (a) is not in this session"
    }

    # (b) THE SHED PART. The same run, one column over: a part that has left the
    #     car. `parts_shed` counts them, and a frame on the first one is the
    #     bumper lying in the road behind the car that lost it.
    $veh_shed = @(Wait-ForHero -Csv $heroCsv -What "a part off the car (VEH3c)" -TimeoutS 25.0 `
        -Predicate { param($c) ($c.Count -gt 53) -and ([int]$c[53] -gt 0) } `
        -Out (Join-Path $OutDir "100-veh3c-shed.png"))[-1]
    if (-not $veh_shed) {
        Say "VEH3c: nothing came off -- a part needs part_break_impulse_ns through its own mounts, which is a 45 km/h shunt on the fixture, and frame (b) is not in this session"
    }

    # (c) THE GLASS. A pane goes to a round or to a hard enough shunt, and the
    #     frame is the car with a hole where its windscreen was.
    $veh_glass = @(Wait-ForHero -Csv $heroCsv -What "a window gone (VEH3c)" -TimeoutS 20.0 `
        -Predicate { param($c) ($c.Count -gt 53) -and ([int]$c[52] -gt 0) } `
        -Out (Join-Path $OutDir "101-veh3c-glass.png"))[-1]
    if (-not $veh_glass) {
        Say "VEH3c: no pane went -- glass takes a thousandth of a crash's energy through its own mounts (a 90 km/h shunt on the fixture) or a round through it, and frame (c) is not in this session"
    }

    # (d) THE FLAT. A tyre is punctured by a ROUND and not by a kerb: the hit
    #     resolver flattens a wheel a round lands within FLAT_HIT_RADII of,
    #     and nothing on the island shoots at the hero's tyres. A session that
    #     wants this frame arms the hero and shoots its own car's wheel from the
    #     pavement first, which is a leg of its own and is not this one.
    $veh_flat = @(Wait-ForHero -Csv $heroCsv -What "a flat tyre (VEH3c)" -TimeoutS 6.0 `
        -Predicate { param($c) ($c.Count -gt 53) -and ([int]$c[51] -gt 0) } `
        -Out (Join-Path $OutDir "102-veh3c-flat.png"))[-1]
    if (-not $veh_flat) {
        Say "VEH3c: no flat -- a puncture is a ROUND inside a wheel's own radius and a kerb cannot do it, so frame (d) needs somebody shooting at the tyres; veh3c_gate::a_flat_tyre_pulls measures the pull instead (1.36 deg of yaw and 6.8 m of drift over four seconds, against 0.00 and 0.000 whole)"
    }

    # (e) THE FIRE. A hull is four panels -- 36 000 J by default -- and a
    #     60 km/h shunt spends about 17 000 of them, so a car burns on its
    #     second or third real crash. Keep going.
    $veh_fire = @(Wait-ForHero -Csv $heroCsv -What "a burning car (VEH3c)" -TimeoutS 30.0 `
        -Predicate { param($c) ($c.Count -gt 53) -and ($c[5] -eq "Driving") -and ([double]$c[49] -le 0.0) -and ([double]$c[6] -ge 0.0) } `
        -Out (Join-Path $OutDir "103-veh3c-fire.png"))[-1]
    [InfInput]::Up(0x11)
    if (-not $veh_fire) {
        Say "VEH3c: the car never burned -- a hull is four panels and a 60 km/h shunt spends half of one, so it takes two or three; dispatch_3d::a_burning_car_brings_the_appliance walks the brigade's whole answer instead (assigned on step 0, on scene at 57.2 s and 11.64 m, resolved at 62.3 s, 22 puffs)"
    }

    # (f) THE SHOT-UP CAR, which needs no seat. The five columns above read the
    #     car the hero is SITTING IN -- that is what "the hero's car damage"
    #     means, and it is the rule the four drivetrain columns follow -- so a
    #     car the hero shoots at from the pavement is damaged in the WORLD and is
    #     not in this row. The frame is therefore triggered on the hero's own
    #     TRIGGER (column 20, `rounds`, with the sidearm up) while it is pointed
    #     at a parked car, and what the car SPENT is measured where it can be:
    #     `veh3c_gate::a_car_shot_at_spends_its_own_joules` (10 000 J into the
    #     flank leaves 72 % of the hull and 100 % of the engine; 10 000 into the
    #     nose kills the engine).
    Say "VEH3c: emptying a magazine into a parked car"
    [InfInput]::Down(0x02)       # 1 -- the sidearm
    Start-Sleep -Milliseconds 250
    [InfInput]::Up(0x02)
    [InfInput]::RightDown()      # aim
    Start-Sleep -Milliseconds 400
    [InfInput]::LeftDown()
    $veh_shoot = @(Wait-ForHero -Csv $heroCsv -What "the hero emptying a magazine into a parked car (VEH3c)" -TimeoutS 8.0 `
        -Predicate { param($c) ($c.Count -gt 53) -and ([int]$c[20] -gt 0) -and ($c[22] -ne "-") } `
        -Out (Join-Path $OutDir "104-veh3c-shot-car.png"))[-1]
    Start-Sleep -Milliseconds 1200
    [InfInput]::LeftUp()
    [InfInput]::RightUp()
    if (-not $veh_shoot) {
        Say "VEH3c: the hero never fired -- frame (f) is not in this session"
    }

    # WHAT THE FIVE COLUMNS ACTUALLY SAID, whatever fired: the last driving
    # line, printed whole, so a reader can see them rather than take the
    # triggers' word for it.
    $last = @(Get-Content $heroCsv -ErrorAction Ignore | Where-Object { $_ -match "^[0-9]" } |
        Where-Object { ($_ -split ",").Count -gt 53 -and ($_ -split ",")[5] -eq "Driving" })
    if ($last.Count -gt 0) {
        $c = $last[-1] -split ","
        Say ("VEH3c ROW: hull {0}%  engine {1}%  flats {2}  panes {3}  shed {4}" -f $c[49], $c[50], $c[51], $c[52], $c[53])
        $minHull = ($last | ForEach-Object { [double](($_ -split ",")[49]) } | Measure-Object -Minimum).Minimum
        $maxShed = ($last | ForEach-Object { [int](($_ -split ",")[53]) } | Measure-Object -Maximum).Maximum
        $maxPane = ($last | ForEach-Object { [int](($_ -split ",")[52]) } | Measure-Object -Maximum).Maximum
        Say ("VEH3c CENSUS: {0} driving rows, hull fell to {1}%, {2} part(s) shed, {3} pane(s) broken" -f $last.Count, $minHull, $maxShed, $maxPane)
    } else {
        Say "VEH3c ROW: no driving row was ever written"
    }
}

# ── 7. close ─────────────────────────────────────────────────────────────────
if ($KeepOpen) {
    Say "left running (pid $($proc.Id)); the island's pack stays mapped until you close it"
} else {
    Say "closing"
    Stop-Process -Id $proc.Id -Force -ErrorAction Ignore
    Start-Sleep -Seconds 2
    Get-Process -Name "inf-player" -ErrorAction Ignore | Stop-Process -Force -ErrorAction Ignore
    Start-Sleep -Seconds 1
    # The control for the two readings above: a cursor that is hidden here as
    # well is a cursor this script cannot see, not one the game took.
    Say ("cursor after the session ended: " + [InfInput]::CursorState())
    Say ("still running: " + $(if (Get-Process -ErrorAction Ignore | Where-Object { $_.ProcessName -in @("inf-studio", "inf-player") }) { "YES" } else { "none" }))
}
# **WHAT THE RUN THREW** (WPN2e audit, carried 282). An exception this script
# did not expect is a defect in this script, and until now it printed in red and
# exited 0. Every expected miss is `-ErrorAction Ignore` and does not land here.
if ($Error.Count -gt 0) {
    Say ("EXCEPTIONS: {0} error(s) were thrown during this run -- the loop's own defect, not the game's" -f $Error.Count)
    $shown = 0
    foreach ($e in $Error) {
        if ($shown -ge 10) { break }
        $where = ""
        if ($e.InvocationInfo) { $where = " at line " + $e.InvocationInfo.ScriptLineNumber }
        Say ("  {0}{1}" -f $e.ToString(), $where)
        $shown++
    }
    $failed = $true
}
if ($failed) {
    # **Non-zero, and that is the point** (audit FIX1). A demo loop that always
    # exits 0 is a screenshot service. This one is the last gate before a wave
    # is called done, so it fails the way a gate fails.
    Say "done (FAILED) — $OutDir"
    exit 7
}
Say "done — $OutDir"
