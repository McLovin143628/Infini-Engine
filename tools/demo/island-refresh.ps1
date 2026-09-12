# REFRESH THE SHOWCASE ISLAND — the recipe, and then the three imports that put
# the hero back (wave VEH3c's audit, finding 5).
#
# `inf island build` is safe for the terrain, the roads, the biomes and the
# level, and DESTRUCTIVE to the character. `samples/island/island.toml`'s
# `content` list copies the committed starter character into the project BY NAME
# — `Starter.inf_skel`, its body, its skin and its three clips — and the island's
# hero is not the committed starter. It is a MetaHuman rebound at those GUIDs by
# `inf-import --rebind-character`: local-only content this repository does not
# carry and CI never sees.
#
# So a plain `inf island build` puts the 161-joint wizard rig and its five-track
# clips back over the 342-joint MetaHuman body and its 150-track ALS clips, and
# reddens four gates that have nothing to do with whatever wave ran it:
#
#     char1a3_gate  11 / 11 failed        outfit1_gate  14 /  3 failed
#     char1b_gate   21 / 11 failed        cov1_gate     13 /  3 failed
#
# The first symptom is `hero's rig has 161 joints ... right: 342`, which reads
# like a character regression and is a FILE COPY.
#
# THE ORDER IS THE WHOLE TRICK. A clip's coupling to a skeleton is POSITIONAL,
# so step 1 puts the mannequin AND its 164 ALS clips at the starter GUIDs, and
# step 2's `retarget_committed_clips` re-retargets them BY NAME across the body
# swap — 150 of 161 tracks kept, 11 dropped, every one an IK or attachment
# helper. Run step 2 alone and the retarget runs on the wizard's five-track clips
# and keeps FOUR.
#
# AND `--dest` MATTERS. The original CHAR1a.3 import wrote its clips under
# `Content/UE/Mannequins/`; a re-run with the default `--dest UE` writes a second
# copy under `Content/UE/`, and `char1b_gate`'s clip resolver returns `None` on
# an AMBIGUOUS name — 74 of 74 ALS sequences come back "unbound" with every one
# of them on disk twice. `-Dest` below defaults to the original.
#
#   pwsh tools/demo/island-refresh.ps1                 # build + restore
#   pwsh tools/demo/island-refresh.ps1 -SkipBuild      # restore only
#   pwsh tools/demo/island-refresh.ps1 -SkipRestore    # build only (do not)
param(
    # The engine checkout. The project and the manifests live beside it.
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    # Where the UE exports live, relative to the folder that holds the checkout.
    [string]$UeOut = "ue-out",
    # The project to write into, relative to the same folder. The importer
    # REFUSES a destination inside the checkout and says why.
    [string]$Project = "island-build/project",
    # Where the clips go. The original import's answer; see the note above.
    [string]$Dest = "UE/Mannequins",
    [switch]$SkipBuild,
    [switch]$SkipRestore
)

$ErrorActionPreference = "Stop"
$holder = (Resolve-Path (Join-Path $Repo "..")).Path
function Say($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

# The two binaries, built release because this writes hundreds of megabytes of
# texture and a debug importer takes minutes to do it.
Push-Location $Repo
try {
    Say "building inf-cli and inf-import (release)"
    & cargo build --release -p inf-cli -p inf-import -j 3
    if ($LASTEXITCODE -ne 0) { throw "the build failed" }
}
finally { Pop-Location }
$inf = Join-Path $Repo "target\release\inf.exe"
$import = Join-Path $Repo "target\release\inf-import.exe"
foreach ($exe in @($inf, $import)) {
    if (-not (Test-Path $exe)) { throw "no $exe" }
}

if (-not $SkipBuild) {
    Push-Location $Repo
    try {
        Say "inf island build --recipe samples/island/island.toml"
        & $inf island build --recipe samples/island/island.toml
        if ($LASTEXITCODE -ne 0) { throw "the island build failed" }
    }
    finally { Pop-Location }
}

if ($SkipRestore) {
    Say "restore SKIPPED -- the island's hero is the committed 161-joint starter until the three imports are run"
    exit 0
}

Push-Location $holder
try {
    $char = Join-Path $UeOut "char1a3/manifest.json"
    $outfit = Join-Path $UeOut "outfit1/manifest.json"
    foreach ($m in @($char, $outfit)) {
        if (-not (Test-Path $m)) {
            Say "MISSING $m -- the UE exports are local-only content; without them the island's hero stays the committed starter"
            exit 3
        }
    }

    Say "1/3 the mannequin and its 164 ALS clips at the starter GUIDs"
    & $import --manifest $char --into $Project --dest $Dest `
        --rebind-character ControlRig_Characters_Mannequins_Meshes_SKM_Manny_SKM_Manny `
        --rebind-character-f ControlRig_Characters_Mannequins_Meshes_SKM_Quinn_SKM_Quinn
    if ($LASTEXITCODE -ne 0) { throw "step 1 failed" }

    Say "2/3 the MetaHuman bodies, and the clips re-retargeted across the swap"
    & $import --manifest $outfit --into $Project `
        --rebind-character INF_Combined_INF_Dominic_FullBody_INF_Dominic_FullBody `
        --rebind-character-f INF_Combined_INF_Vivian_FullBody_INF_Vivian_FullBody `
        --only Combined --character-lods 3
    if ($LASTEXITCODE -ne 0) { throw "step 2 failed" }

    Say "3/3 the outfits and the hair"
    & $import --manifest $outfit --into $Project `
        --wearable "m:outfit:INF_Dominic_Clothing_INF_Dominic_Outfits" `
        --wearable "f:outfit:INF_Vivian_Clothing_INF_Vivian_Outfits" `
        --wearable "m:hair:Dominic_Grooms_Hair_S_PulledBack_CardsMesh:head" `
        --wearable "f:hair:Vivian_Grooms_Hair_S_BobLayered_CardsMesh:head" `
        --only Clothing --only Grooms
    if ($LASTEXITCODE -ne 0) { throw "step 3 failed" }
}
finally { Pop-Location }

Say "done. The four local-only gates that read this project:"
Say "  cargo test -p inf-player --test char1a3_gate --test char1b_gate --test cov1_gate --test outfit1_gate"
