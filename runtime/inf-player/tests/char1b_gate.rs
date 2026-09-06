//! **WAVE CHAR1b.1 — THE POSE.**
//!
//! Foot IK on real topography, the ALS clip→mode map, the additive layer,
//! look-at, and the crowd on the same graph. One file, so a reader looking for
//! "what CHAR1b.1 proved" finds it in one place — `char1a_gate.rs` keeps the
//! bodies wave's 22 and `char1a3_gate.rs` the MetaHuman slice's 14.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repository root")
}

/// The island project this machine builds locally, or `None`.
///
/// Every arm that reads it SKIPS with a printed reason rather than failing, the
/// rule `char1a3_gate` set: the ALS clips and the MetaHumans are licensed
/// content that never enters this repository, so CI has no island project and
/// must not have a red gate about it.
fn island_project() -> Option<PathBuf> {
    let p = repo().join("../island-build/project/Content");
    p.is_dir().then(|| p.canonicalize().unwrap_or(p))
}

/// Every `.inf_anim` under `dir`, decoded, with its file stem.
fn clips_in(dir: &Path) -> Vec<(String, inf_anim::AnimClipAsset)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "inf_anim"))
        .collect();
    paths.sort();
    for p in paths {
        let Ok(bytes) = std::fs::read(&p) else {
            continue;
        };
        let Ok(asset) = inf_asset::decode::<inf_anim::AnimClipAsset>(&bytes) else {
            continue;
        };
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        out.push((stem, asset));
    }
    out
}

/// Every `.inf_skel` under `dir`, keyed by its sidecar GUID's raw bytes.
fn skeletons_in(dir: &Path) -> std::collections::BTreeMap<[u8; 16], inf_anim::SkeletonAsset> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "inf_skel") {
            continue;
        }
        let Ok(side) = inf_asset::AssetSidecar::load(&p) else {
            continue;
        };
        let Ok(bytes) = std::fs::read(&p) else {
            continue;
        };
        if let Ok(asset) = inf_asset::decode::<inf_anim::SkeletonAsset>(&bytes) {
            out.insert(*side.guid.uuid().as_bytes(), asset);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE FOOT-IK GATE CHANNEL — the mechanism that had never run
// ─────────────────────────────────────────────────────────────────────────────

/// **THE DISTRIBUTION THE `Enable_FootIK_*` BAND IS DERIVED AGAINST.**
///
/// Reported, not asserted past a floor: this is the measurement that chose
/// [`inf_anim::derive::FOOT_GROUND_BAND_M`], and it prints the whole census so a
/// reader can check the number against the clips rather than against a claim.
///
/// The assertion is the one thing the band must do: the grounded families
/// (`Walk`, `Run`, `Sprint`, `Stop`, `TurnIP`, `Land`) must come out enabled and
/// the airborne ones (`JumpLoop`, `FallLoop`, `Mantle`) must come out disabled.
#[test]
fn the_foot_ik_band_separates_the_grounded_clips_from_the_airborne_ones() {
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the ALS \
             clips are local-only content and CI has none"
        );
        return;
    };
    let dir = content.join("UE/Mannequins");
    let clips = clips_in(&dir);
    if clips.is_empty() {
        eprintln!("SKIP: no .inf_anim under {}", dir.display());
        return;
    }
    let rigs = skeletons_in(&dir);
    assert!(!rigs.is_empty(), "no .inf_skel beside the clips");

    let mut rows: Vec<(String, f32, f32)> = Vec::new();
    for (stem, asset) in &clips {
        let Some(rig) = asset.skeleton.and_then(|id| rigs.get(&id)) else {
            continue;
        };
        let feet = inf_anim::derive::foot_joints(rig);
        if feet.is_empty() {
            continue;
        }
        // The clip's own lowest foot, sampled on a 30 Hz grid like the deriver's.
        let steps = ((asset.clip.duration * 30.0).ceil() as usize).max(2);
        let mut lows = [f32::INFINITY; 2];
        for k in 0..=steps {
            let t = asset.clip.duration * (k as f32) / (steps as f32);
            let pose = inf_anim::pose::sample_clip(&rig.skeleton, &asset.clip, t, false);
            let g = inf_anim::pose::global_transforms(&rig.skeleton, &pose);
            for (side, &j) in feet.iter().enumerate().take(2) {
                let y = g[j as usize].to_scale_rotation_translation().2.y;
                if y < lows[side] {
                    lows[side] = y;
                }
            }
        }
        rows.push((stem.clone(), lows[0], lows[1]));
    }
    assert!(rows.len() > 100, "only {} clips measured", rows.len());

    let band = inf_anim::derive::FOOT_GROUND_BAND_M;
    let mut on: Vec<&(String, f32, f32)> = Vec::new();
    let mut off: Vec<&(String, f32, f32)> = Vec::new();
    for r in &rows {
        if r.1.min(r.2) <= band {
            on.push(r);
        } else {
            off.push(r);
        }
    }
    let worst_on = on.iter().map(|r| r.1.min(r.2)).fold(0.0f32, f32::max);
    let best_off = off
        .iter()
        .map(|r| r.1.min(r.2))
        .fold(f32::INFINITY, f32::min);
    println!(
        "\n=== Enable_FootIK band ({band:.3} m) over {} clips: {} grounded, {} airborne ===",
        rows.len(),
        on.len(),
        off.len()
    );
    println!("  highest grounded foot: {worst_on:.4} m   lowest airborne foot: {best_off:.4} m");
    let mut sorted: Vec<&(String, f32, f32)> = rows.iter().collect();
    sorted.sort_by(|a, b| a.1.min(a.2).total_cmp(&b.1.min(b.2)));
    for r in sorted.iter().rev().take(24) {
        println!("  {:>8.4}  {}", r.1.min(r.2), r.0);
    }

    let want_on = ["Walk_F", "Run_F", "Sprint_F", "TurnIP_L90", "Land_Light"];
    for needle in want_on {
        let Some(row) = rows.iter().find(|r| r.0.ends_with(needle)) else {
            continue;
        };
        assert!(
            row.1.min(row.2) <= band,
            "{} is a grounded clip and reads {:.4} m, past the {band:.3} m band",
            row.0,
            row.1.min(row.2)
        );
    }
    let want_off = ["JumpLoop", "FallLoop", "Mantle_2m"];
    for needle in want_off {
        let Some(row) = rows.iter().find(|r| r.0.ends_with(needle)) else {
            continue;
        };
        assert!(
            row.1.min(row.2) > band,
            "{} is an airborne clip and reads {:.4} m, inside the {band:.3} m band",
            row.0,
            row.1.min(row.2)
        );
    }
}
