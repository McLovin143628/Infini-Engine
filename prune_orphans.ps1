<#
.SYNOPSIS
    Prune build artifacts that cargo orphaned by IDENTITY (not by age).

.DESCRIPTION
    Cargo keys every incremental cache and every compiled artifact by a hash of
    the crate's feature set + dependency graph + rustc version. Change any of
    those and cargo mints a fresh hash -- and never garbage-collects the old
    one. Over a few months of development the workspace crates accumulate
    dozens of full-size copies of themselves.

    `cargo sweep --time N` cannot see these: it prunes by last-ACCESS time, and
    every link pass touches the live artifacts, so nothing ages out. This script
    is the complement -- it prunes by identity:

      * incremental/ : keep the 2 newest cache dirs per target.
      * deps/        : keep the 2 newest hash-generations per WORKSPACE target.
                       Third-party artifacts are never touched -- that is the
                       warm cache that makes rebuilds fast.

    Targets that no longer exist in the workspace (a deleted spike, a removed
    test suite) have ALL of their artifacts removed.

    Everything removed here is a cache: cargo rebuilds whatever it still needs.
    The cost of over-pruning is compile time, never correctness.

    This engine's target/ leaks especially fast: 30+ workspace crates, a wgpu
    dep graph, cross-compiles to wasm32 and android, and dev/test builds that
    differ in features. The July 2026 audit found 943 incremental caches across
    101 targets in a 132 GB target/.

.PARAMETER DryRun
    Report what would be removed without deleting anything.

.NOTES
    Must stay Windows PowerShell 5.1 compatible: run_clean.cmd invokes it via
    `powershell`, not `pwsh`. In 5.1, Sort-Object/Measure-Object -Property
    cannot see hashtable KEYS as properties (pwsh 7 can), so this script sorts
    and sums with explicit script blocks / loops instead. Don't "simplify" those
    back into `Sort-Object Stamp` or `Measure-Object Bytes -Sum` -- it works
    when you test it in pwsh and then dies inside run_clean.cmd.

    The live-target set is derived from `cargo metadata`, not from a hardcoded
    list or source-tree globs. With this many crates -- nested under crates/,
    editor/crates/, platforms/, runtime/ and tools/ -- a glob that misses a
    directory would silently classify that crate's artifacts as "deleted from
    source" and prune ALL of its generations. Asking cargo removes the guess.
#>
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$repo = $PSScriptRoot

# --- Per-project configuration ------------------------------------------------
# Manifests whose workspace members define the "ours to prune" name set, and the
# target dirs to prune. Both are lists so a repo with several cargo roots (or a
# custom build.target-dir) can list them all; the live set is their union.
$manifests = @(
    (Join-Path $repo 'Cargo.toml')
)
$targets = @(
    (Join-Path $repo 'target')
)

# Local crates that live OUTSIDE the cargo roots above (a [patch.crates-io]
# fork, a sibling checkout). Cargo compiles them incrementally just like
# workspace members, so they must be in the live set or their caches get pruned
# to zero. Empty here -- every crate this workspace builds is a member.
$localCrateGlobs = @()

$KEEP = 2

# --- Locate cargo -------------------------------------------------------------
$cargoExe = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
if (-not (Test-Path $cargoExe)) {
    $cmd = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cmd) { $cargoExe = $cmd.Source }
}
if (-not (Test-Path $cargoExe)) {
    Write-Host "Cargo not found. Install the Rust toolchain from https://rustup.rs."
    exit 1
}

# --- Profile roots ------------------------------------------------------------
# Artifacts live in <target>\debug and <target>\release, and -- for cross
# compiles -- in <target>\<triple>\debug|release. Collect every one that exists.
$profileRoots = [System.Collections.Generic.List[string]]::new()
foreach ($t in $targets) {
    if (-not (Test-Path $t)) { continue }
    foreach ($p in @('debug', 'release')) {
        $d = Join-Path $t $p
        if (Test-Path $d) { $profileRoots.Add($d) }
    }
    foreach ($tripleDir in Get-ChildItem $t -Directory -Force -ErrorAction SilentlyContinue) {
        foreach ($p in @('debug', 'release')) {
            $d = Join-Path $tripleDir.FullName $p
            if (Test-Path $d) { $profileRoots.Add($d) }
        }
    }
}

if ($profileRoots.Count -eq 0) {
    Write-Host "Nothing to prune: no build profile directories found under:"
    foreach ($t in $targets) { Write-Host "  $t" }
    exit 0
}

# --- Safety: never prune while a build holds one of these target dirs ---------
# Checked by trying to take the lock cargo itself holds during a build, rather
# than by scanning for cargo.exe processes -- a sibling project's dev server
# would trip a process scan while being completely irrelevant to this target/.
foreach ($root in $profileRoots) {
    $lock = Join-Path $root '.cargo-lock'
    if (Test-Path $lock) {
        try {
            $fs = [IO.File]::Open($lock, 'Open', 'ReadWrite', 'None')
            $fs.Close()
        } catch {
            Write-Host "A cargo build is running against $root."
            Write-Host "Pruning now would delete fingerprints mid-build and corrupt the cache."
            Write-Host "Stop the build (and any dev server for THIS project) and re-run."
            exit 1
        }
    }
}

# --- Live targets, derived from cargo metadata --------------------------------
# Anything in deps/ or incremental/ matching one of these names is ours to
# prune. Anything else is a third-party dependency and is left alone. Target
# names (not package names) are what artifacts are named after, and they cover
# libs, bins, examples, benches and every integration-test file for free.
$live = [System.Collections.Generic.HashSet[string]]::new()
[void]$live.Add('build_script_build')

# Native stderr (cargo's progress chatter) raises a terminating NativeCommandError
# while $ErrorActionPreference is 'Stop', even with 2>$null. Relax it around the
# cargo calls and put it back afterwards; their exit codes are checked directly.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'

foreach ($manifest in $manifests) {
    if (-not (Test-Path $manifest)) {
        Write-Host "Manifest not found, skipping: $manifest"
        continue
    }
    $json = & $cargoExe metadata --no-deps --format-version 1 --manifest-path $manifest 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $json) {
        Write-Host "cargo metadata failed for $manifest."
        Write-Host "Refusing to guess the workspace target list -- nothing was removed."
        exit 1
    }
    $meta = $json | ConvertFrom-Json
    foreach ($pkg in $meta.packages) {
        foreach ($tgt in $pkg.targets) {
            [void]$live.Add(($tgt.name -replace '-', '_'))
        }
    }

    # Local PATH dependencies are ours too. A crate living beside the repo (a
    # [patch.crates-io] fork, a sibling checkout) is compiled incrementally
    # exactly like a workspace member, so leaving it out of the live set would
    # classify its caches as "deleted from workspace" and remove ALL of them --
    # forcing a full rebuild of an expensive dep, which is the opposite of
    # keeping a warm cache. Registry packages carry a `source`; local ones have
    # it null. This needs dependency resolution, so it is best-effort: if it
    # fails, the workspace-only set above still stands. --offline is mandatory,
    # not an optimization: without it this call downloads the registry index,
    # and a disk-cleanup script must never reach for the network.
    $depJson = & $cargoExe metadata --format-version 1 --offline --manifest-path $manifest 2>$null
    if ($LASTEXITCODE -eq 0 -and $depJson) {
        $depMeta = $depJson | ConvertFrom-Json
        foreach ($pkg in $depMeta.packages) {
            if ($null -eq $pkg.source) {
                foreach ($tgt in $pkg.targets) {
                    [void]$live.Add(($tgt.name -replace '-', '_'))
                }
            }
        }
    }
}

$ErrorActionPreference = $prevEap

# Crates declared by glob (see $localCrateGlobs). Read the package name out of
# each manifest; fall back to the directory name if it cannot be found.
foreach ($glob in $localCrateGlobs) {
    foreach ($manifestFile in Get-ChildItem $glob -File -ErrorAction SilentlyContinue) {
        $hit = Select-String -Path $manifestFile.FullName -Pattern '^\s*name\s*=\s*"([^"]+)"' |
               Select-Object -First 1
        if ($hit) {
            [void]$live.Add(($hit.Matches[0].Groups[1].Value -replace '-', '_'))
        } else {
            [void]$live.Add(($manifestFile.Directory.Name -replace '-', '_'))
        }
    }
}

Write-Host ("Local targets: {0}; profile dirs: {1}." -f $live.Count, $profileRoots.Count)

$removedBytes = 0L
$removedItems = 0
$report = [System.Collections.Generic.List[object]]::new()

function Resolve-Stem {
    <#
      Artifact base names look like  libinf_core-a1b2c3d4  or  inf_core-a1b2c3d4.
      The lib prefix is a linker convention, not part of the target name -- but
      stripping it blindly would mangle a crate genuinely named lib*. Try the
      raw name first, then the stripped one; return $null if neither is ours.
    #>
    param([string]$Raw)

    if ($live.Contains($Raw)) { return $Raw }
    $stripped = $Raw -replace '^lib', ''
    if ($live.Contains($stripped)) { return $stripped }
    return $null
}

function Remove-Generations {
    <#
      $Generations: hashtable of generation-key -> @{ Items; Bytes; Stamp }
      Keeps the $KEEP newest generations, removes the rest. A stem whose target
      is gone from the workspace keeps nothing.
    #>
    param([string]$Stem, [hashtable]$Generations, [string]$Area, [bool]$IsLive)

    # NOT named $keep: PowerShell variable names are case-INSENSITIVE, so a
    # local $keep IS the script-level $KEEP. Assigning it 0 first would shadow
    # the real value and prune every generation to zero -- which is exactly
    # what the first dry run of this script did.
    $keepCount = 0
    if ($IsLive) { $keepCount = $KEEP }

    # Script-block sort: -Property Stamp would not see the hashtable key in 5.1.
    $ordered = @($Generations.Values | Sort-Object -Property { $_.Stamp } -Descending)
    $doomed = @($ordered | Select-Object -Skip $keepCount)
    if ($doomed.Count -eq 0) { return }

    $bytes = 0L
    foreach ($gen in $doomed) { $bytes += [long]$gen.Bytes }

    foreach ($gen in $doomed) {
        foreach ($item in $gen.Items) {
            if ($DryRun) {
                Write-Host "  would remove $($item.FullName)"
            } else {
                Remove-Item $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    $note = 'target deleted from workspace'
    if ($IsLive) { $note = "kept $keepCount newest" }

    $script:removedBytes += $bytes
    $script:removedItems += $doomed.Count
    $script:report.Add([pscustomobject]@{
        Area = $Area
        Stem = $Stem
        Gone = $doomed.Count
        MB   = [math]::Round($bytes / 1MB, 0)
        Note = $note
    })
}

foreach ($root in $profileRoots) {
    # A readable label: the last two path segments, e.g. "wasm32-.../debug".
    $label = Split-Path $root -Leaf
    $parent = Split-Path (Split-Path $root -Parent) -Leaf
    $label = "$parent/$label"

    # --- Pass 1: incremental/ caches -----------------------------------------
    # Directory names look like  inf_render-2hum28l3uub41  (target-hash).
    $inc = Join-Path $root 'incremental'
    if (Test-Path $inc) {
        $byStem = @{}
        foreach ($dir in Get-ChildItem $inc -Directory -Force -ErrorAction SilentlyContinue) {
            $stem = $dir.Name -replace '-[0-9a-z]+$', ''
            if (-not $byStem.ContainsKey($stem)) { $byStem[$stem] = @{} }
            $bytes = (Get-ChildItem $dir.FullName -Recurse -File -Force -ErrorAction SilentlyContinue |
                      Measure-Object Length -Sum).Sum
            $byStem[$stem][$dir.Name] = @{
                Items = @($dir); Bytes = [long]$bytes; Stamp = $dir.LastWriteTime
            }
        }
        foreach ($stem in $byStem.Keys) {
            Remove-Generations -Stem $stem -Generations $byStem[$stem] `
                -Area "incremental/$label" -IsLive $live.Contains($stem)
        }
    }

    # --- Pass 2: deps/ artifacts of workspace targets -------------------------
    # File names look like  libinf_core-a1b2c3d4e5f60718.rlib  (stem-hash.ext).
    # One hash generation spans several files (.rlib/.d/.exe/.pdb) -- they live
    # and die together, so the generation key is the hash.
    $deps = Join-Path $root 'deps'
    if (Test-Path $deps) {
        $byStem = @{}
        foreach ($file in Get-ChildItem $deps -File -Force -ErrorAction SilentlyContinue) {
            if ($file.BaseName -notmatch '^(.*)-([0-9a-f]{8,})$') { continue }
            $stem = Resolve-Stem -Raw $matches[1]
            # Third-party crates are the warm cache -- never touch them.
            if (-not $stem) { continue }
            $hash = $matches[2]

            if (-not $byStem.ContainsKey($stem)) { $byStem[$stem] = @{} }
            if (-not $byStem[$stem].ContainsKey($hash)) {
                $byStem[$stem][$hash] = @{ Items = @(); Bytes = 0L; Stamp = [datetime]::MinValue }
            }
            $gen = $byStem[$stem][$hash]
            $gen.Items += $file
            $gen.Bytes += $file.Length
            if ($file.LastWriteTime -gt $gen.Stamp) { $gen.Stamp = $file.LastWriteTime }
        }
        foreach ($stem in $byStem.Keys) {
            # Every stem here already passed Resolve-Stem, so it is live by
            # construction; a workspace target removed from the manifest never
            # matches and is skipped above rather than pruned to zero.
            Remove-Generations -Stem $stem -Generations $byStem[$stem] `
                -Area "deps/$label" -IsLive $true
        }
    }
}

# --- Report -------------------------------------------------------------------
if ($report.Count -eq 0) {
    Write-Host "No orphaned artifacts: every workspace target is at or below $KEEP generations."
} else {
    $report | Sort-Object MB -Descending | Select-Object -First 40 | Format-Table -AutoSize |
        Out-String -Width 100 | Write-Host
    if ($report.Count -gt 40) {
        Write-Host ("... and {0} more entries." -f ($report.Count - 40))
    }
    $verb = 'Pruned'
    if ($DryRun) { $verb = 'Would prune' }
    Write-Host ("{0} {1} orphaned generations, {2} GB." -f `
        $verb, $removedItems, [math]::Round($removedBytes / 1GB, 2))
}
