//! **World-partition cook advisories, end to end** (P16.5, extended P16.6).
//!
//! World partition can see three hazards at cook time that the runtime can only
//! *suffer*, and none is fixed up automatically (a fixup would make cell
//! residency depend on the reference graph, on which entities happen to carry a
//! script, or on which components they carry — so a level's memory ceiling would
//! move every time content was added). They are surfaced as cook advisories
//! instead, where they are cheap to fix: the `dangling_terrain_refs` precedent.
//!
//! The third (P16.6) is the loudest in its consequences: a `Terrain` bins by its
//! entity ORIGIN while its heightfield spans kilometres, so an unmarked terrain in
//! a partitioned level means **the ground despawns under the player** the moment
//! they walk out of its cell.
//!
//! These are integration tests rather than a trust in the Ring-0 unit tests
//! because the advisories are produced inside the **parallel** `cook_one` stage
//! and folded back serially into [`CookReport::warnings`]. "The scan finds it"
//! and "the user is told about it" are two different claims, and only the second
//! one is what actually protects a designer.

use std::path::Path;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;

/// The level asset id every project here is stamped with (a real `inf_asset`
/// sidecar, so it survives the cook rather than being synthesized).
const LEVEL_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000016050100"));

fn g(n: u128) -> uuid::Uuid {
    uuid::Uuid::from_u128(n)
}

/// A bare entity record with every slot `None` — the base each hazard entity is
/// built from with struct-update syntax.
fn rec(guid: u128, name: &str, at: (f64, f64)) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        guid: g(guid),
        name: name.into(),
        parent: None,
        transform: inf_ecs::components::Transform {
            translation: inf_ecs::math::Vec3d::new(at.0, 0.0, at.1),
            ..inf_ecs::components::Transform::IDENTITY
        },
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
        character_movement: None,
    }
}

fn mesh() -> Option<inf_ecs::components::MeshRef> {
    Some(inf_ecs::components::MeshRef::default())
}

/// The hazard level: a streaming source, a cross-cell joint, a streamed
/// mesh+blueprint actor, a persistent entity carrying the *same* blueprint (which
/// must stay silent), an **unmarked terrain** (P16.6), and an `AlwaysLoaded` one
/// (also silent).
fn hazard_level(partitioned: bool) -> inf_scene::RuntimeLevel {
    inf_scene::RuntimeLevel {
        title: "Hazards".into(),
        entities: vec![
            // A streaming source, so the level has one (and is persistent).
            inf_scene::RuntimeEntity {
                streaming_source: Some(inf_ecs::components::StreamingSource { radius_m: 0.0 }),
                ..rec(0xC001, "Player", (0.0, 0.0))
            },
            // Cell (0,0): jointed to an entity a whole cell away.
            inf_scene::RuntimeEntity {
                mesh: mesh(),
                joint_3d: Some(inf_ecs::components::Joint3D {
                    other: inf_ecs::EntityRef::new(g(0xC003)),
                    ..Default::default()
                }),
                ..rec(0xC002, "Hinge", (10.0, 10.0))
            },
            // Cell (1,0): the joint's target.
            inf_scene::RuntimeEntity {
                mesh: mesh(),
                ..rec(0xC003, "Door", (300.0, 10.0))
            },
            // Cell (2,0): the classic gameplay actor — a mesh that runs a
            // blueprint. A mesh occupies space, so it streams; the blueprint
            // therefore never ticks. This is the shape the warning exists for.
            inf_scene::RuntimeEntity {
                mesh: mesh(),
                actor: Some(g(0xC0AC)),
                ..rec(0xC004, "Enemy", (600.0, 10.0))
            },
            // A persistent manager with the SAME blueprint: it binds and ticks,
            // so it must NOT be reported.
            inf_scene::RuntimeEntity {
                actor: Some(g(0xC0AC)),
                ..rec(0xC005, "GameMode", (0.0, 0.0))
            },
            // P16.6 — the ground-despawns shape: a terrain with no marker. It
            // occupies space, so it bins into the cell holding its origin, and its
            // heightfield leaves with that cell.
            inf_scene::RuntimeEntity {
                terrain: Some(inf_ecs::components::Terrain {
                    asset: Some(g(0xC0AA)),
                    ..Default::default()
                }),
                ..rec(0xC006, "Terrain", (10.0, 10.0))
            },
            // …and the sanctioned authoring: the same terrain, marked. Silent.
            inf_scene::RuntimeEntity {
                terrain: Some(inf_ecs::components::Terrain::default()),
                always_loaded: Some(inf_ecs::components::AlwaysLoaded),
                ..rec(0xC007, "Good Terrain", (300.0, 10.0))
            },
        ],
        settings: inf_scene::RuntimeSettings {
            partition: inf_scene::PartitionSettings {
                enabled: partitioned,
                cell_size_m: 256.0,
                activation_radius_m: 64.0,
                prefetch_margin_m: 0.0,
            },
            ..Default::default()
        },
        geo: Default::default(),
    }
}

/// Scaffold a project whose one level is [`hazard_level`].
fn make_hazard_project(root: &Path, partitioned: bool) {
    ProjectManifest::new("Partition Hazards", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let bytes = hazard_level(partitioned).encode().unwrap();
    let path = content.join("Hazards.inf_lvl");
    std::fs::write(&path, &bytes).unwrap();
    AssetSidecar::new(LEVEL_ID, AssetKind::Level, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();
}

/// All three hazards travel from the Ring-0 scan, through `cook_one`'s per-asset
/// advisories, into `CookReport.warnings` — non-fatally, naming the entity, the
/// cell, the blueprint/asset and the remedy.
#[test]
fn partition_advisories_reach_the_cook_report() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_hazard_project(&proj, true);

    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default())
        .expect("a partition hazard is an advisory, not a cook failure");

    // The cook still did its job.
    assert_eq!(report.partitions_built, 1);
    assert!(report.partition_cells >= 3, "{}", report.partition_cells);

    let level = LEVEL_ID.to_string();
    let mentions = |needle: &str| {
        report
            .warnings
            .iter()
            .any(|w| w.contains(&level) && w.contains(needle))
    };

    // (1) The cross-cell reference — names both entities and the field.
    assert!(
        mentions("joint_3d.other")
            && mentions(&g(0xC002).to_string())
            && mentions(&g(0xC003).to_string()),
        "the cross-cell joint must be reported: {:?}",
        report.warnings
    );

    // (2) The streamed blueprint actor — names the entity, its blueprint, and
    //     the remedy. Without this, the only symptom a designer ever sees is a
    //     level full of statues.
    let streamed = report
        .warnings
        .iter()
        .find(|w| w.contains(&g(0xC004).to_string()))
        .unwrap_or_else(|| panic!("the streamed actor must be reported: {:?}", report.warnings));
    assert!(streamed.contains("STREAMED"), "{streamed}");
    assert!(
        streamed.contains("AlwaysLoaded"),
        "the advisory must name the remedy: {streamed}"
    );
    assert!(
        streamed.contains(&g(0xC0AC).to_string()),
        "…and the blueprint it is bound to: {streamed}"
    );

    // The PERSISTENT entity carrying the same blueprint is silent: it binds and
    // ticks, so warning about it would be noise — and noise is what teaches
    // people to ignore the warning that matters.
    assert!(
        !report
            .warnings
            .iter()
            .any(|w| w.contains(&g(0xC005).to_string())),
        "a persistent blueprint must not be reported: {:?}",
        report.warnings
    );

    // (3) P16.6 — the STREAMED TERRAIN. This is the one whose symptom is "the
    //     world has no floor past cell (0,0)", so the advisory has to name the
    //     entity, that it is streamed, its `.inf_terrain`, and the remedy.
    let terrain = report
        .warnings
        .iter()
        .find(|w| w.contains(&g(0xC006).to_string()))
        .unwrap_or_else(|| {
            panic!(
                "the streamed terrain must be reported: {:?}",
                report.warnings
            )
        });
    assert!(terrain.contains("STREAMED"), "{terrain}");
    assert!(
        terrain.contains("DESPAWN"),
        "the advisory must say what actually happens: {terrain}"
    );
    assert!(
        terrain.contains("AlwaysLoaded"),
        "the advisory must name the remedy: {terrain}"
    );
    assert!(
        terrain.contains(&g(0xC0AA).to_string()),
        "…and the .inf_terrain it streams from: {terrain}"
    );

    // The AlwaysLoaded terrain — what every committed sample authors — is silent.
    assert!(
        !report
            .warnings
            .iter()
            .any(|w| w.contains(&g(0xC007).to_string())),
        "an AlwaysLoaded terrain must not be reported: {:?}",
        report.warnings
    );

    // Advisories are deterministic (sorted at the source), so a cook report does
    // not churn between builds.
    let again = cook(&proj, &dir.path().join("out2"), &CookOptions::default()).unwrap();
    assert_eq!(report.warnings, again.warnings);
}

/// An unpartitioned level raises neither advisory, however it is authored — both
/// scans only ever run inside `build_partition`.
#[test]
fn partition_advisories_never_fire_for_an_unpartitioned_level() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    // The identical entity set — every hazard still in place — with the one
    // settings flag off.
    make_hazard_project(&proj, false);

    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).unwrap();
    assert_eq!(report.partitions_built, 0);
    assert_eq!(report.partition_cells, 0);
    assert!(
        report
            .warnings
            .iter()
            .all(|w| !w.contains("STREAMED") && !w.contains("joint_3d.other")),
        "{:?}",
        report.warnings
    );
}
