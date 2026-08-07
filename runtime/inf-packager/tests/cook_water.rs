//! **The P20.1 downhill advisory, end to end.**
//!
//! Water flows downhill. A river spline authored the wrong way up a valley is a
//! mistake the cook can *see* and the runtime can only *look wrong* about — no
//! crash, no missing asset, just water going the wrong way — which is exactly the
//! shape of hazard the `dangling_terrain_refs` advisory pattern exists for.
//!
//! Three properties are pinned here, and the third is the one that matters most:
//!
//! 1. an uphill river **is reported**, non-fatally, naming the entity, the rise,
//!    the gradient and the remedy;
//! 2. a downhill river, a flat one within tolerance, and a *reversed* one
//!    (negative `river_flow_m_s`, which flips what "downhill" means) are
//!    **silent** — an advisory that fires on correct content stops being read;
//! 3. a level with **no water at all** gains no warnings, which is the off-path
//!    half of the same claim.

use std::path::Path;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{Spline, SplineInterp, Transform, WaterBody, WaterKind};
use inf_ecs::math::Vec3d;
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;

const LEVEL_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000020010001"));

fn g(n: u128) -> uuid::Uuid {
    uuid::Uuid::from_u128(n)
}

/// A bare entity record with every slot `None`.
fn rec(guid: u128, name: &str) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        guid: g(guid),
        name: name.into(),
        parent: None,
        transform: Transform::IDENTITY,
        visible: true,
        mesh: None,
        material: None,
        light: None,
        camera: None,
        sprite: None,
        tilemap: None,
        nine_slice: None,
        text2d: None,
        light_2d: None,
        rigid_body_2d: None,
        collider_2d: None,
        character_controller_2d: None,
        rigid_body_3d: None,
        collider_3d: None,
        character_controller_3d: None,
        actor: None,
        terrain: None,
        pcg_volume: None,
        skeletal_mesh: None,
        anim_player: None,
        anim_state_machine: None,
        root_motion: None,
        attached_to: None,
        joint_2d: None,
        joint_3d: None,
        audio_source: None,
        audio_listener: None,
        decal: None,
        volume: None,
        spline: None,
        foliage: None,
        streaming_source: None,
        always_loaded: None,
        time_of_day: None,
        sky_atmosphere: None,
        water_body: None,
        buoyancy: None,
        voxel_volume: None,
        destructible: None,
        ik_target: None,
        cloth_sim: None,
        hair_guides: None,
    }
}

/// A river entity: a `WaterBody::river` plus the `Spline` on the **same entity**
/// (component composition, not a reference — which is why the cook needs no new
/// dependency edge to find it).
fn river(guid: u128, name: &str, ys: &[f64], flow: f64) -> inf_scene::RuntimeEntity {
    let points: Vec<Vec3d> = ys
        .iter()
        .enumerate()
        .map(|(i, &y)| Vec3d::new(i as f64 * 40.0, y, 0.0))
        .collect();
    inf_scene::RuntimeEntity {
        water_body: Some(WaterBody {
            river_flow_m_s: flow,
            ..WaterBody::river(8.0, 1.5, flow)
        }),
        spline: Some(Spline {
            points,
            closed: false,
            // Linear, so the profile is exactly the authored control points and
            // the assertion is about the advisory rather than about Catmull-Rom
            // overshoot.
            interp: SplineInterp::Linear,
        }),
        ..rec(guid, name)
    }
}

fn level(entities: Vec<inf_scene::RuntimeEntity>) -> inf_scene::RuntimeLevel {
    inf_scene::RuntimeLevel {
        title: "Water".into(),
        entities,
        settings: inf_scene::RuntimeSettings::default(),
    }
}

/// The same level with **world partitioning on** — which moves every entity into
/// the derived `.inf_part` and CLEARS `level.entities` in the cook. Any advisory
/// that read the entity list after that branch would see nothing.
fn partitioned(entities: Vec<inf_scene::RuntimeEntity>) -> inf_scene::RuntimeLevel {
    inf_scene::RuntimeLevel {
        title: "Water".into(),
        entities,
        settings: inf_scene::RuntimeSettings {
            partition: inf_scene::PartitionSettings {
                enabled: true,
                cell_size_m: 256.0,
                activation_radius_m: 64.0,
                prefetch_margin_m: 0.0,
            },
            ..Default::default()
        },
    }
}

fn make_project(root: &Path, level: &inf_scene::RuntimeLevel) {
    ProjectManifest::new("Water Advisory", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let bytes = level.encode().unwrap();
    let path = content.join("Water.inf_lvl");
    std::fs::write(&path, &bytes).unwrap();
    AssetSidecar::new(LEVEL_ID, AssetKind::Level, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();
}

fn cook_warnings(level: inf_scene::RuntimeLevel) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_project(&proj, &level);
    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default())
        .expect("an uphill river is an advisory, not a cook failure");
    assert_eq!(report.levels_rewritten, 1, "the level still cooked");
    report.warnings
}

/// As [`cook_warnings`], but asserts the level really WAS partitioned — otherwise
/// the regression tests below would pass by silently taking the ordinary path.
fn cook_warnings_partitioned(level: inf_scene::RuntimeLevel) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_project(&proj, &level);
    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default())
        .expect("an uphill river is an advisory, not a cook failure");
    assert_eq!(report.levels_rewritten, 1, "the level still cooked");
    assert_eq!(
        report.partitions_built, 1,
        "the fixture was not partitioned — this test would prove nothing"
    );
    report.warnings
}

#[test]
fn an_uphill_river_is_reported_with_its_rise_and_its_remedy() {
    // Falls 4 m, then climbs 6 m — one span, well over the tolerance.
    let warnings = cook_warnings(level(vec![river(
        0xB001,
        "Backwards Brook",
        &[20.0, 18.0, 16.0, 22.0, 20.0],
        1.5,
    )]));
    let hit = warnings
        .iter()
        .find(|w| w.contains(&g(0xB001).to_string()))
        .unwrap_or_else(|| panic!("the uphill river must be reported: {warnings:?}"));
    // It names the level, the entity, the magnitude and the remedy — an advisory
    // that only says "something is wrong" is one nobody acts on.
    assert!(hit.contains(&LEVEL_ID.to_string()), "{hit}");
    assert!(hit.contains("climbs"), "{hit}");
    assert!(hit.contains("gradient"), "{hit}");
    assert!(hit.contains("negative `river_flow_m_s`"), "{hit}");
    // The rise is the real one (6 m), not a rounded-off placeholder.
    assert!(hit.contains("6.0") || hit.contains("5.9"), "{hit}");
}

#[test]
fn correct_rivers_stay_silent() {
    let names = [
        // Straight downhill.
        river(0xB010, "Good", &[30.0, 26.0, 22.0, 18.0], 1.5),
        // Dead flat: no rise at all, and none reported.
        river(0xB011, "Canal", &[10.0, 10.0, 10.0, 10.0], 0.8),
        // Sampling-noise-sized wobble, under the 0.5 m merged-span tolerance.
        river(0xB012, "Wobbly", &[30.0, 29.0, 29.2, 27.0, 26.9], 1.0),
        // REVERSED: the points climb, but the river flows the other way, so it
        // runs downhill. A tolerance-blind check would report this one.
        river(0xB013, "Reversed", &[10.0, 14.0, 18.0, 22.0], -2.0),
    ];
    let warnings = cook_warnings(level(names.to_vec()));
    for guid in [0xB010u128, 0xB011, 0xB012, 0xB013] {
        let id = g(guid).to_string();
        assert!(
            !warnings.iter().any(|w| w.contains(&id)),
            "a correct river was reported ({id}): {warnings:?}"
        );
    }
}

/// A river whose depth taper lifts its **bed** while its surface falls (P20.4).
///
/// This is the case the P20.1 surface check is blind to *by construction*: the
/// water still slopes the right way, so nothing at runtime is wrong to look at —
/// but the ground under it climbs, which is a basin. It gets its own advisory
/// with its own remedy, because "lower the spline" would not fix it.
#[test]
fn a_bed_that_climbs_under_a_falling_surface_is_reported_separately() {
    // Surface 30 → 27 (falls 3 m); depth 6 → 0.5 (so the bed goes 24 → 26.5,
    // climbing 2.5 m).
    let mut basin = river(0xB030, "Basin", &[30.0, 29.0, 28.0, 27.0], 1.2);
    {
        let w = basin.water_body.as_mut().unwrap();
        w.river_depth_start_m = 6.0;
        w.river_depth_end_m = 0.5;
    }
    let warnings = cook_warnings(level(vec![basin]));
    let id = g(0xB030).to_string();
    let mine: Vec<&String> = warnings.iter().filter(|w| w.contains(&id)).collect();
    assert_eq!(mine.len(), 1, "expected exactly the bed advisory: {mine:?}");
    let msg = mine[0];
    assert!(msg.contains("BED climbs"), "{msg}");
    assert!(msg.contains("2.5"), "the rise must be quoted: {msg}");
    assert!(
        msg.contains("river_depth_end_m"),
        "the remedy must name the depth fields, not the spline: {msg}"
    );
    // It is an ADVISORY: the cook still succeeded (`cook_warnings` unwraps the
    // report), and the surface advisory did NOT fire — the two are independent.
    assert!(
        !msg.contains("climbs 2.5 m across 1 stretch(es) in the direction it flows (the worst"),
        "the surface advisory fired on a falling surface: {msg}"
    );
}

/// ANTI-VACUITY for the bed advisory: the *same* river with a constant depth has
/// a bed that falls exactly as its surface does, and is silent. Without this the
/// test above would pass against an advisory that fired on every river.
#[test]
fn a_constant_depth_river_has_a_silent_bed() {
    let mut good = river(0xB031, "Even", &[30.0, 29.0, 28.0, 27.0], 1.2);
    {
        let w = good.water_body.as_mut().unwrap();
        w.river_depth_start_m = 2.0;
        w.river_depth_end_m = 2.0;
    }
    let warnings = cook_warnings(level(vec![good]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB031).to_string())),
        "{warnings:?}"
    );
}

/// A **reversed** river's bed is judged in the direction the water goes, exactly
/// as its surface is. The points climb and the depth widens downstream, so read
/// forwards the bed climbs — and read the way the water actually flows it does
/// not.
#[test]
fn the_bed_advisory_honours_a_reversed_flow() {
    let mut back = river(0xB032, "Backwards", &[10.0, 14.0, 18.0, 22.0], -2.0);
    {
        let w = back.water_body.as_mut().unwrap();
        w.river_depth_start_m = 0.5;
        w.river_depth_end_m = 6.0;
    }
    let warnings = cook_warnings(level(vec![back]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB032).to_string())),
        "a correctly-reversed river was reported: {warnings:?}"
    );
    // …and the same geometry read FORWARDS (positive flow) is reported on both
    // counts, which proves the reversal is what silenced it above.
    let mut fwd = river(0xB033, "Forwards", &[10.0, 14.0, 18.0, 22.0], 2.0);
    {
        let w = fwd.water_body.as_mut().unwrap();
        w.river_depth_start_m = 0.5;
        w.river_depth_end_m = 6.0;
    }
    let warnings = cook_warnings(level(vec![fwd]));
    let id = g(0xB033).to_string();
    let mine: Vec<&String> = warnings.iter().filter(|w| w.contains(&id)).collect();
    assert_eq!(mine.len(), 2, "surface AND bed: {mine:?}");
    assert!(!mine[0].contains("BED"), "surface first: {mine:?}");
    assert!(mine[1].contains("BED climbs"), "{mine:?}");
}

/// **The cook and the tool sanitize identically** (P20.4 audit).
///
/// A negative authored depth used to reach the cook's `RiverProfile` raw while
/// every other consumer clamped it, so the cook judged a bed the renderer, the
/// sim and the editor's report never showed. `RiverProfile::authored` is now the
/// one sanitizer; this pins the consequence from outside.
#[test]
fn a_negative_authored_depth_is_clamped_before_the_bed_is_judged() {
    // Surface falls 30 -> 27. Depth −6 -> 0.5. CLAMPED that is 0 -> 0.5, so the
    // bed goes 30 -> 26.5: it FALLS, and nothing is reported. Read raw, the bed
    // would go 36 -> 26.5 and likewise fall — so the discriminating case is the
    // reverse taper below.
    let mut a = river(0xB040, "Clamped", &[30.0, 29.0, 28.0, 27.0], 1.2);
    {
        let w = a.water_body.as_mut().unwrap();
        w.river_depth_start_m = -6.0;
        w.river_depth_end_m = 0.5;
    }
    let warnings = cook_warnings(level(vec![a]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB040).to_string())),
        "{warnings:?}"
    );

    // The discriminating case: depth 0.5 -> −6. Clamped that is 0.5 -> 0, so the
    // bed goes 29.5 -> 27 and FALLS (silent). Read raw it would go 29.5 -> 33 and
    // CLIMB 3.5 m, which is what the cook used to report and nothing else did.
    let mut b = river(0xB041, "Reversed taper", &[30.0, 29.0, 28.0, 27.0], 1.2);
    {
        let w = b.water_body.as_mut().unwrap();
        w.river_depth_start_m = 0.5;
        w.river_depth_end_m = -6.0;
    }
    let warnings = cook_warnings(level(vec![b]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB041).to_string())),
        "the cook judged an unclamped bed: {warnings:?}"
    );
}

/// A `WaterKind::River` with **no `Spline`** has no centreline. That is an
/// authoring state (you added the component and have not drawn the path yet), not
/// a hazard, and the cook must not nag about it — nor panic on the empty path.
#[test]
fn a_river_without_a_spline_is_neither_reported_nor_fatal() {
    let entity = inf_scene::RuntimeEntity {
        water_body: Some(WaterBody::river(8.0, 1.5, 1.0)),
        ..rec(0xB020, "Unrouted")
    };
    let warnings = cook_warnings(level(vec![entity]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB020).to_string())),
        "{warnings:?}"
    );
}

/// Oceans and lakes have no flow direction to be wrong about, so the advisory
/// must not fire on them however they are placed.
#[test]
fn still_bodies_are_never_reported() {
    let ocean = inf_scene::RuntimeEntity {
        water_body: Some(WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            ..WaterBody::default()
        }),
        // Deliberately hostile: a spline that climbs, on a body that has no flow.
        spline: Some(Spline {
            points: vec![
                Vec3d::new(0.0, 0.0, 0.0),
                Vec3d::new(50.0, 40.0, 0.0),
                Vec3d::new(100.0, 80.0, 0.0),
            ],
            closed: false,
            interp: SplineInterp::Linear,
        }),
        ..rec(0xB030, "Sea")
    };
    let lake = inf_scene::RuntimeEntity {
        water_body: Some(WaterBody::lake(12.0, inf_ecs::math::Vec2d::splat(40.0))),
        ..rec(0xB031, "Tarn")
    };
    let warnings = cook_warnings(level(vec![ocean, lake]));
    for guid in [0xB030u128, 0xB031] {
        let id = g(guid).to_string();
        assert!(!warnings.iter().any(|w| w.contains(&id)), "{warnings:?}");
    }
}

/// **The off-path half.** A level with no water gains no water warnings — the
/// advisory costs a water-free project exactly one `Option` test per entity.
#[test]
fn a_water_free_level_gains_no_water_warnings() {
    let warnings = cook_warnings(level(vec![rec(0xB040, "Prop")]));
    assert!(
        !warnings
            .iter()
            .any(|w| w.contains("river") || w.contains("climbs")),
        "{warnings:?}"
    );
}

/// A river under a **moved parent** is judged where it actually is: the advisory
/// composes the parent chain's transforms, so a level that authored its rivers
/// under a container and then lifted the container does not start reporting them.
///
/// The hostile shape: the spline itself is flat, and the *parent* is rotated so
/// the river tilts. Flat-in-local, climbing-in-world.
#[test]
fn the_parent_chain_is_applied_before_judging() {
    let parent = inf_scene::RuntimeEntity {
        transform: Transform {
            translation: Vec3d::new(0.0, 100.0, 0.0),
            // Roll the container 10° about +Z: a rotation about Z maps
            // +X → (cos θ, sin θ, 0), so a locally flat river running along +X
            // now CLIMBS in world space. (About +X it would not move at all,
            // which is the mistake this comment exists to stop repeating.)
            rotation: Vec3d::new(0.0, 0.0, 10.0),
            scale: Vec3d::ONE,
        },
        ..rec(0xB050, "Container")
    };
    let child = inf_scene::RuntimeEntity {
        parent: Some(g(0xB050)),
        ..river(0xB051, "Tilted", &[0.0, 0.0, 0.0, 0.0], 1.5)
    };
    let warnings = cook_warnings(level(vec![parent, child]));
    let hit = warnings
        .iter()
        .find(|w| w.contains(&g(0xB051).to_string()))
        .unwrap_or_else(|| {
            panic!("a river tilted uphill by its parent must be reported: {warnings:?}")
        });
    assert!(hit.contains("climbs"), "{hit}");

    // …and the control: the same child with the parent pitched the other way is
    // silent, so the test is measuring the tilt rather than the parenting.
    let down = inf_scene::RuntimeEntity {
        transform: Transform {
            translation: Vec3d::new(0.0, 100.0, 0.0),
            rotation: Vec3d::new(0.0, 0.0, -10.0),
            scale: Vec3d::ONE,
        },
        ..rec(0xB050, "Container")
    };
    let child = inf_scene::RuntimeEntity {
        parent: Some(g(0xB050)),
        ..river(0xB051, "Tilted", &[0.0, 0.0, 0.0, 0.0], 1.5)
    };
    let warnings = cook_warnings(level(vec![down, child]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB051).to_string())),
        "the downhill control was reported: {warnings:?}"
    );
}

/// **The regression this file exists for most.** A partitioned level's entities are
/// moved into the derived `.inf_part` and **cleared** from the level record
/// mid-cook. An advisory that read `level.entities` after that branch would see an
/// empty list and report nothing — silently, and on exactly the levels most likely
/// to hold a kilometre of river.
///
/// So the check runs BEFORE the partition branch, and this pins it from the
/// outside: the same uphill river, partitioned, is still reported.
#[test]
fn an_uphill_river_is_reported_on_a_partitioned_level_too() {
    let ys = [20.0, 18.0, 16.0, 22.0, 20.0];

    // The control: unpartitioned, reported.
    let plain = cook_warnings(level(vec![river(0xB060, "Brook", &ys, 1.5)]));
    assert!(
        plain.iter().any(|w| w.contains(&g(0xB060).to_string())),
        "the control was not reported: {plain:?}"
    );

    // … and partitioned, where the entity list is gone by the time the level is
    // encoded. A streaming source is present so the partition is a real one.
    let entities = vec![
        inf_scene::RuntimeEntity {
            streaming_source: Some(inf_ecs::components::StreamingSource { radius_m: 0.0 }),
            ..rec(0xB061, "Player")
        },
        river(0xB060, "Brook", &ys, 1.5),
    ];
    let warnings = cook_warnings_partitioned(partitioned(entities));
    let hit = warnings
        .iter()
        .find(|w| w.contains(&g(0xB060).to_string()))
        .unwrap_or_else(|| {
            panic!(
                "a partitioned level's uphill river was NOT reported — the advisory \
                 is reading `level.entities` after the partition branch cleared it: \
                 {warnings:?}"
            )
        });
    assert!(hit.contains("climbs"), "{hit}");
}

/// The partition's **own** advisories still arrive alongside the water one — the
/// hoist must EXTEND the advisory list, not assign over it. Assigning would have
/// traded one silent hole for another.
#[test]
fn the_partition_advisories_survive_the_water_one() {
    let entities = vec![
        inf_scene::RuntimeEntity {
            streaming_source: Some(inf_ecs::components::StreamingSource { radius_m: 0.0 }),
            ..rec(0xB070, "Player")
        },
        // A terrain with no `AlwaysLoaded`: the P16.6 ground-despawns advisory.
        inf_scene::RuntimeEntity {
            terrain: Some(inf_ecs::components::Terrain::default()),
            transform: Transform {
                translation: Vec3d::new(600.0, 0.0, 600.0),
                ..Transform::IDENTITY
            },
            ..rec(0xB071, "Terrain")
        },
        river(0xB072, "Brook", &[20.0, 18.0, 25.0], 1.5),
    ];
    let warnings = cook_warnings_partitioned(partitioned(entities));
    assert!(
        warnings.iter().any(|w| w.contains(&g(0xB072).to_string())),
        "the water advisory is missing: {warnings:?}"
    );
    assert!(
        warnings.iter().any(|w| w.contains("DESPAWN")),
        "the partition's own advisories were clobbered by the water one: {warnings:?}"
    );
}

/// A **closed** river is a loop: it cannot help regaining every metre it loses, so
/// advising on it would be advising on a circle. The branch is skipped, and this is
/// the test that says so — without it the skip is an untested `continue`.
#[test]
fn a_closed_river_is_never_reported() {
    // Points that climb hard in the authored order. Open, this is a certain hit.
    let ys = [0.0, 10.0, 20.0, 30.0];
    let open = cook_warnings(level(vec![river(0xB080, "Open", &ys, 1.5)]));
    assert!(
        open.iter().any(|w| w.contains(&g(0xB080).to_string())),
        "the open control must be reported, or the closed case proves nothing: {open:?}"
    );

    // The same points, closed.
    let mut looped = river(0xB081, "Loop", &ys, 1.5);
    if let Some(sp) = looped.spline.as_mut() {
        sp.closed = true;
    }
    let warnings = cook_warnings(level(vec![looped]));
    assert!(
        !warnings.iter().any(|w| w.contains(&g(0xB081).to_string())),
        "a closed river was reported — a loop necessarily climbs as much as it \
         falls, so this could only be noise: {warnings:?}"
    );
}
