//! **WAVE VEH3f.2b -- THE DCC CAR SHELLS + THE CARRIED VEHICLE CLOSURES**:
//! the gate.
//!
//! Every arm below reads the WORLD or the committed bytes, never a table's
//! claim about them. The header of each arm says what it READS, and whether
//! VEH3f's panels-on-boxes (or the pre-wave machine) would pass it.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::boarding::BoardPhase;
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::VehicleDef;
use inf_ecs::EcsWorld;

const RADIUS: f64 = 0.3;
const GROUND: Uuid = Uuid::from_u128(0x5E3F_2B00);
const CHASSIS: Uuid = Uuid::from_u128(0x5E3F_2B01);
const HERO: Uuid = Uuid::from_u128(0x5E3F_2B03);
const SKEL_GUID: Uuid = Uuid::from_u128(0x5E3F_2B07);
const SM_GUID: Uuid = Uuid::from_u128(0x5E3F_2B08);

// ── the fixture (veh3d_gate's, for its reasons) ─────────────────────────────

fn catalogue_def(id: &str) -> VehicleDef {
    *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .or_else(|| inf_ecs::roster::roster().get(id))
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"))
}

fn ground(world: &mut EcsWorld) {
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    world.world_mut().entity_mut(e).insert((
        Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            ..Default::default()
        },
        Visibility::default(),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(200.0, 0.5, 200.0),
            friction: 0.9,
            ..Default::default()
        },
    ));
}

fn car(world: &mut EcsWorld, guid: Uuid, at: DVec3, yaw_deg: f64, def: &VehicleDef) {
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Car".into(),
        at,
        yaw_deg,
        paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    };
    inf_ecs::vehicle::spawn_rig(world, guid, def, &spawn);
}

/// Stand one capsule character up with its FEET at `at`.
fn stand(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, yaw_deg: f64) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y + cm.stand_half_height_m + RADIUS, at.z);
    t.rotation.y = yaw_deg;
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
            radius: RADIUS,
            ..Default::default()
        },
        CharacterController3D::default(),
        cm,
        t,
    ));
}

/// The SHIPPED host (the player's own `RuntimeSim`) with the mannequin rig,
/// one row's car at the origin and the hero beside its driver's flank.
fn rigged_sim(row: &str, hero_at: DVec3) -> inf_player::runtime_sim::RuntimeSim {
    use inf_player::runtime_sim::RuntimeSim;
    const IDLE: inf_anim::ClipRef = [0xd3; 16];
    let def = catalogue_def(row);
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(
        &mut world,
        CHASSIS,
        DVec3::new(
            0.0,
            inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15,
            0.0,
        ),
        0.0,
        &def,
    );
    stand(&mut world, HERO, "Hero", hero_at, 0.0);
    let e = world.entity_of(HERO).expect("the hero");
    world.world_mut().entity_mut(e).insert((
        inf_ecs::components::AnimStateMachine {
            sm: Some(SM_GUID),
            ..Default::default()
        },
        inf_ecs::components::SkeletalMesh {
            mesh: None,
            skeleton: Some(SKEL_GUID),
        },
    ));
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
    let skeleton = inf_anim::build_template(
        inf_anim::BodyPlan::Biped,
        &inf_anim::BodyParams {
            height_m: 1.8,
            ..Default::default()
        },
    )
    .expect("the mannequin builds");
    sim.set_skeletons([(SKEL_GUID, skeleton)].into_iter().collect());
    sim.set_state_machines(
        [(
            SM_GUID,
            inf_anim::StateMachine {
                states: vec![inf_anim::SmState::clip("idle", IDLE)],
                entry: 0,
                ..Default::default()
            },
        )]
        .into_iter()
        .collect(),
    );
    sim.set_pose_clips(
        [(
            Uuid::from_bytes(IDLE),
            inf_anim::AnimClip::new("idle", Vec::new()),
        )]
        .into_iter()
        .collect(),
    );
    sim
}

/// Where a row's hero is put down: a metre out from the driver's (`+X`)
/// flank, a little behind the driver's cushion.
fn hero_at(def: &VehicleDef) -> DVec3 {
    let parts: &[inf_ecs::vehicle::BodyPart] = match def.art {
        Some(k) if !k.parts().is_empty() => k.parts(),
        _ => def.body.parts(),
    };
    let z = parts
        .iter()
        .filter(|p| p.kind == inf_ecs::vehicle::BodyPartKind::Seat && p.centre.x > 0.0)
        .map(|p| p.centre.z * def.half_extents.z)
        .next()
        .unwrap_or(0.0);
    DVec3::new(def.half_extents.x + 1.1, 0.0, z - 0.6)
}

fn boarding(sim: &inf_player::runtime_sim::RuntimeSim) -> inf_ecs::boarding::BoardingState {
    let e = sim.world().entity_of(HERO).expect("the hero");
    sim.world()
        .world()
        .get::<CharacterMovement>(e)
        .expect("a mover")
        .runtime
        .boarding
}

/// **THE INNER HANDLE: the latch waits for the reach, and the hand holds on
/// to the shut** -- READS the shipped host's boarding machine (the latch's
/// moment, `BoardingState::mark_s`, and the door's hinge angle read back off
/// its joint) and the posed hand (`RuntimeSim::boarding_residuals`: the
/// hand's weight on the pull and the POSED hand joint's distance to it), per
/// `Seated` step, on the five shells and four more rows.
///
/// The claims: (1) the door's motor is told to shut only after the hand's
/// reach has finished (`HAND_REACH_S`) and held (`HANDLE_HOLD_S`) -- the
/// VEH3f audit's "latches on the motor before the reach ramps in"; (2) the
/// door is still open at that moment; (3) once the hand is on the pull at
/// weight 1 it stays on it (<= 2 cm) until the door is shut -- it never lets
/// go early. Pre-wave machine: the motor latched on the step `Seated` began
/// (t = 0, the reach not started) -- FAILS (1). Printed, not asserted: on
/// which rows the pull comes into a seated arm's reach before the shut
/// (the pull 0.25 m ahead of the H-point swings out of reach at the boarding
/// angle; a torso lean is the primitive the pose system lacks -- CARRIED).
#[test]
fn the_inner_latch_waits_for_the_reach_and_the_hand_holds_to_the_shut() {
    use inf_ecs::movement::actions::INTERACT;
    use inf_player::runtime_sim::RuntimeInput;
    println!("=== the inner handle on the shipped host ===");
    let mut bad = Vec::new();
    let mut held_rows = 0usize;
    for row in [
        "sedan", "sports", "suv", "truck", "cruiser", "van", "ambulance", "brute_bus",
        "vapid_contender",
    ] {
        let def = catalogue_def(row);
        let mut sim = rigged_sim(row, hero_at(&def));
        let (mut latch, mut latch_deg) = (None::<f64>, 0.0f64);
        let (mut on, mut off_early, mut worst) = (0usize, 0usize, 0.0f64);
        let mut was_on = false;
        let mut first_on = None::<f64>;
        for i in 0..900u32 {
            let input = if i == 60 {
                RuntimeInput::default().press(INTERACT)
            } else {
                RuntimeInput::default()
            };
            sim.step_once(input);
            let b = boarding(&sim);
            if b.phase == BoardPhase::Seated {
                if latch.is_none() && b.mark_s >= 0.0 {
                    latch = Some(b.mark_s);
                    latch_deg = b.door_deg;
                }
                let r = sim.boarding_residuals(HERO);
                let w = r.as_ref().map(|r| r.sockets.handle_weight).unwrap_or(0.0);
                let shut = b.door_deg <= inf_ecs::boarding::DOOR_SHUT_DEG;
                if w >= 0.999 {
                    on += 1;
                    was_on = true;
                    first_on.get_or_insert(b.time_s);
                    let d = r.and_then(|r| r.handle_m).unwrap_or(f64::INFINITY);
                    worst = worst.max(d);
                } else if was_on && !shut {
                    off_early += 1;
                }
            }
            if b.phase == BoardPhase::Driving {
                break;
            }
        }
        let latch_s = latch.unwrap_or(f64::NAN);
        println!(
            "  {row:<16} latch at {latch_s:.3} s (door {latch_deg:.1} deg); hand on the pull {on} rows from {}, {:.2} mm worst; let go early {off_early}",
            first_on.map(|t| format!("{t:.3} s")).unwrap_or_else(|| "never".into()),
            worst * 1000.0
        );
        let reach = inf_ecs::boarding::HAND_REACH_S + inf_ecs::boarding::HANDLE_HOLD_S;
        if !(latch_s >= reach - 1e-9) {
            bad.push(format!("{row}: the door latched at {latch_s:.3} s, before the reach ({reach:.2} s)"));
        }
        if latch_deg < inf_ecs::boarding::DOOR_BOARD_DEG {
            bad.push(format!("{row}: the door was already swinging ({latch_deg:.1} deg) at the latch"));
        }
        if on > 0 {
            held_rows += 1;
            if worst > 0.02 {
                bad.push(format!("{row}: the posed hand was {:.1} mm off the pull", worst * 1000.0));
            }
            if off_early > 0 {
                bad.push(format!("{row}: the hand let go of the pull {off_early} step(s) before the shut"));
            }
        }
    }
    println!("  rows whose hand held the pull before the shut: {held_rows}");
    assert!(held_rows >= 3, "only {held_rows} row(s) ever held the pull");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

// ── the licence: derived sidecars and the cook's wall ───────────────────────

/// A one-quad glTF (two triangles) plus its `.bin`, the ASSET0 gate's.
fn quad_gltf(dir: &std::path::Path, stem: &str) -> std::path::PathBuf {
    let positions: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let mut buf: Vec<u8> = Vec::new();
    for v in positions {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let pos_len = positions.len() * 4;
    let idx_off = buf.len();
    for v in indices {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
    let bin = format!("{stem}.bin");
    let json = serde_json::json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "name": stem, "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }] }],
        "buffers": [{ "uri": bin, "byteLength": buf.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": pos_len },
            { "buffer": 0, "byteOffset": idx_off, "byteLength": 12 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
              "min": [0, 0, 0], "max": [1, 1, 0] },
            { "bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR" }
        ]
    });
    std::fs::write(dir.join(&bin), &buf).unwrap();
    let path = dir.join(format!("{stem}.gltf"));
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    path
}

/// A one-pack manifest (`licence`, `ship`) naming two meshes.
fn licence_manifest(dir: &std::path::Path, licence: &str, ship: bool) -> std::path::PathBuf {
    let meshes = dir.join("meshes");
    std::fs::create_dir_all(&meshes).unwrap();
    quad_gltf(&meshes, "Prop_A");
    quad_gltf(&meshes, "Prop_B");
    let manifest = serde_json::json!({
        "schema_version": 1,
        "generator": "the VEH3f.2b gate",
        "packs": [{ "name": "GatePack", "license": licence, "ship": ship }],
        "textures": [],
        "materials": [],
        "meshes": [
            { "key": "Gate_A", "pack": "GatePack", "nanite": false, "material_slots": [],
              "lods": [{ "level": 0, "file": "meshes/Prop_A.gltf", "screen_size": 1.0 }] },
            { "key": "Gate_B", "pack": "GatePack", "nanite": false, "material_slots": [],
              "lods": [{ "level": 0, "file": "meshes/Prop_B.gltf", "screen_size": 1.0 }] }
        ],
        "errors": []
    });
    let path = dir.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    path
}

/// Every sidecar under `dir` that names a licence pack: `(file, licence,
/// may ship)`.
fn licence_rows(dir: &std::path::Path) -> Vec<(String, String, Option<bool>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().is_none_or(|x| x != "toml") {
                continue;
            }
            let Ok(side) = inf_asset::AssetSidecar::load(&p.with_extension("")) else {
                continue;
            };
            let Some(t) = side.import.as_ref() else {
                continue;
            };
            if !t.contains_key(inf_asset::licence::LICENCE_PACK_KEY) {
                continue;
            }
            out.push((
                p.file_name().unwrap().to_string_lossy().into_owned(),
                t.get(inf_asset::licence::LICENCE_KEY)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                inf_asset::licence::may_ship(&side),
            ));
        }
    }
    out.sort();
    out
}

/// **AFTER AN IMPORT, EVERY SIDECAR OF A PACK READS ITS MANIFEST ROW, DERIVED
/// ONES INCLUDED** -- READS every `.toml` under the project's content that
/// carries a licence row, `.inf_vmesh.toml` included, after the REAL bridge
/// (`import_manifest`) re-imports a pack whose licence row changed between
/// two runs (the 2026-09-25 relabel, in miniature: "user to confirm",
/// `ship = false` -> "confirmed", `ship = true`), with the editor's vmesh
/// sweep run between them so the derived sidecars exist and are cache hits
/// the second time. The claim: every row -- produced AND derived -- is the
/// second manifest's. Pre-wave importer: the derived `.inf_vmesh.toml` keep
/// the first row (a cache hit re-derives nothing) -- FAILS, as the island's
/// 487 did. Mutation -> red: the `sync_derived_licences` call removed from
/// `import_manifest`.
#[test]
fn after_an_import_every_sidecar_of_a_pack_reads_its_manifest_row_derived_ones_too() {
    use inf_editor_core::assets::ue_import::{import_manifest, UeImportOptions};
    use inf_editor_core::assets::AssetProject;
    let dir = tempfile::tempdir().unwrap();
    let content = dir.path().join("project").join("Content");
    let first = licence_manifest(dir.path(), "Fab Standard -- user to confirm", false);
    let mut project = AssetProject::open(&content).expect("a project opens");
    import_manifest(&mut project, &first, &UeImportOptions::default()).expect("the first import");
    // The editor's own sweep, as a project open runs it: whatever the import
    // did not derive already is derived now.
    let _ = inf_editor_core::assets::vmesh::sweep(&mut project);
    let before = licence_rows(&content);
    let derived_before = before
        .iter()
        .filter(|r| r.0.ends_with(".inf_vmesh.toml"))
        .count();
    assert!(derived_before >= 2, "{before:?}");
    assert!(before.iter().all(|r| r.2 == Some(false)), "{before:?}");
    drop(project);

    let second = licence_manifest(dir.path(), "Fab Standard (commercial tier), confirmed", true);
    let mut project = AssetProject::open(&content).expect("it re-opens");
    let report = import_manifest(&mut project, &second, &UeImportOptions::default())
        .expect("the second import");
    let after = licence_rows(&content);
    let derived_after = after
        .iter()
        .filter(|r| r.0.ends_with(".inf_vmesh.toml"))
        .count();
    println!(
        "LICENCE: {} sidecar(s) carry the pack's row after the re-import, {derived_after} derived; advisories {:?}",
        after.len(),
        report.advisories
    );
    for (file, licence, ship) in &after {
        assert_eq!(
            (licence.as_str(), *ship),
            ("Fab Standard (commercial tier), confirmed", Some(true)),
            "{file} kept the first import's licence row"
        );
    }
    assert_eq!(derived_after, derived_before, "a derived sidecar lost its row");
}

/// A cookable project whose content is one mesh per `(name, licence row)`.
fn licence_project(
    root: &std::path::Path,
    rows: &[(&str, Option<(&str, bool)>)],
) -> Vec<inf_asset::AssetId> {
    inf_project::ProjectManifest::new("licence-wall", "blank-3d")
        .save(root)
        .expect("scaffold");
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let (mesh, _) = inf_dcc::to_mesh_asset(&inf_dcc::cube(1.0), &Default::default());
    let mut ids = Vec::new();
    for (name, row) in rows {
        let bytes = inf_asset::encode(&mesh).unwrap();
        let path = content.join(format!("{name}.inf_mesh"));
        std::fs::write(&path, &bytes).unwrap();
        let mut side = inf_asset::AssetSidecar::new(
            inf_asset::AssetId::new(),
            inf_asset::AssetKind::Mesh,
            inf_asset::ContentHash::of(&bytes),
        );
        if let Some((pack, ship)) = row {
            let mut t = toml::Table::new();
            t.insert(inf_asset::licence::LICENCE_KEY.into(), "gate".into());
            t.insert(inf_asset::licence::LICENCE_SHIP_KEY.into(), (*ship).into());
            t.insert(inf_asset::licence::LICENCE_PACK_KEY.into(), (*pack).into());
            side.import = Some(t);
        }
        side.save(&path).unwrap();
        ids.push(side.guid);
    }
    ids
}

/// **THE COOK REFUSES CONTENT WHOSE LICENCE MAY NOT SHIP** -- READS the cook's
/// verdict (`inf_packager::cook`) and the pack it writes, over a project whose
/// closure reaches: (a) a mesh stamped as the UE-only reference mannequins are
/// (`licence_pack = "UE5_Mannequins"`, `licence_may_ship = false`); (b) one
/// stamped as a confirmed Fab pack (`true`); (c) one of this repository's own
/// with no row. The claims: (a) in the closure -> the default cook REFUSES
/// (`CookError::Licence`) and writes no pack; the same closure with
/// `local_reference` -> a pack, reported BLOCKING; (b) + (c) alone -> a clean
/// cook, not blocking. Pre-wave cook: no reader of `licence_may_ship` -- packs
/// (a) silently -- FAILS. Mutation -> red: the wall's `return Err` removed from
/// `cook` (the refusal becomes a clean cook).
#[test]
fn the_cook_refuses_content_whose_licence_may_not_ship() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("p");
    let ids = licence_project(
        &proj,
        &[
            ("Mannequin_Ref", Some(("UE5_Mannequins", false))),
            ("Fab_Car", Some(("DrivableCarsBasicVehicleS", true))),
            ("Ours", None),
        ],
    );
    let out = dir.path().join("out");
    let opts = |roots: Vec<inf_asset::AssetId>, local: bool| inf_packager::CookOptions {
        roots: Some(roots),
        local_reference: local,
        ..Default::default()
    };
    let refused = inf_packager::cook(&proj, &out, &opts(ids.clone(), false));
    match &refused {
        Err(inf_packager::CookError::Licence { count, listing }) => {
            println!("LICENCE WALL: refused {count} asset(s): {listing}");
            assert_eq!(*count, 1);
            assert!(listing.contains("UE5_Mannequins"), "{listing}");
        }
        other => panic!("the cook did not refuse the reference mannequin: {other:?}"),
    }
    assert!(
        !out.join(inf_packager::DEFAULT_PACK_NAME).exists(),
        "a refused cook wrote a pack"
    );
    let local = inf_packager::cook(&proj, &out, &opts(ids.clone(), true)).expect("a local cook");
    assert!(
        local.has_blocking(),
        "a local-reference pack must report itself blocking"
    );
    assert!(
        local
            .blocking
            .iter()
            .any(|b| b.contains("LOCAL REFERENCE ONLY")),
        "{:?}",
        local.blocking
    );
    let out2 = dir.path().join("out2");
    let clean = inf_packager::cook(&proj, &out2, &opts(ids[1..].to_vec(), false))
        .expect("the confirmed pack and our own content cook");
    assert!(
        !clean
            .blocking
            .iter()
            .any(|b| b.contains("LOCAL REFERENCE ONLY")),
        "{:?}",
        clean.blocking
    );
    println!("LICENCE WALL: the confirmed pack + ours cook clean; the local cook is blocking");
}

// ── the island: traffic contacts and the body census ────────────────────────

fn fixture_recipe() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/island-fixture/island.toml")
}

/// The CI island, cooked and booted as `run_headless` boots it (`veh3f_gate`'s
/// harness, verbatim).
fn island_sim(tmp: &std::path::Path) -> inf_player::runtime_sim::RuntimeSim {
    let recipe = inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("scaffold");
    inf_island::write_content(&build, &proj.join("Content")).expect("content");
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    let source = inf_player::level::PackLevelSource::open(&out).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

/// How many island minutes the census runs: TEN in a release build off CI,
/// ONE in a dev build or on a shared runner; `INF_VEH3F2B_ISLAND_MINUTES`
/// overrides it. Printed either way.
fn island_minutes() -> u64 {
    if let Some(m) = std::env::var("INF_VEH3F2B_ISLAND_MINUTES")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return m;
    }
    if cfg!(debug_assertions) || std::env::var_os("CI").is_some() {
        1
    } else {
        10
    }
}

/// **THE ISLAND'S TRAFFIC: NO MOVING CONTACT, AND THE SHELLS ON THE STREET** --
/// READS the physics world's own CONTACT EVENTS (`RuntimeSim::
/// solid_contacts_started`, every step, both bodies vehicle chassis, at least
/// one a MOVING traffic car) and each resident traffic car's own children
/// (`roster::body_kind`: `shell` / `imported` / `primitive`), on the CI island
/// cooked and booted as the player boots it, the crowd set aside (a TRAFFIC
/// census, `veh3f_gate`'s reason).
///
/// The claims: (1) zero moving contacts -- the VEH3f audit counted three in
/// ten minutes (all stopped cars, 3.87-4.15 m apart, the following rule
/// seeing centres); (2) every sample of a traffic car whose row wears a shell
/// draws a shell, every row of an imported pack its art, and no car of a
/// hero class (coupe, sedan, SUV, truck) draws primitives. Pre-wave: the
/// centre-in-a-corridor rule's contacts and an empty shell column -- FAILS
/// both. Mutations -> red: `gap_ahead` reading the obstacle's CENTRE only
/// (the footprint's corners dropped); `SHELL_BODY_PART` renamed back to
/// `art_body` in the rig (the census reads "imported").
#[test]
fn the_island_traffic_makes_no_moving_contact_and_its_hero_classes_draw_shells() {
    use std::collections::{BTreeMap, BTreeSet};
    let tmp = tempfile::tempdir().expect("a temp dir");
    let mut sim = island_sim(tmp.path());
    inf_ecs::crowd::set_population(sim.world_mut(), BTreeMap::new());
    let minutes = island_minutes();
    let steps = minutes * 60 * 60;
    let moving = |g: Uuid, sim: &inf_player::runtime_sim::RuntimeSim| {
        inf_ecs::traffic::traffic_of(sim.world()).is_some_and(|t| t.records.contains_key(&g))
            && inf_ecs::traffic::day_of(g) != inf_ecs::traffic::TrafficDay::Parked
    };
    let chassis = |g: Uuid, sim: &inf_player::runtime_sim::RuntimeSim| {
        let w = sim.world();
        inf_ecs::traffic::traffic_of(w).is_some_and(|t| t.records.contains_key(&g))
            || w.entity_of(g)
                .and_then(|e| w.world().get::<inf_ecs::components::VehicleClass>(e))
                .is_some()
    };
    let mut contacts: BTreeSet<(Uuid, Uuid)> = BTreeSet::new();
    let mut solid_events = 0usize;
    let mut census: BTreeMap<(String, &'static str), usize> = BTreeMap::new();
    let mut wrong = Vec::new();
    let t0 = std::time::Instant::now();
    let pose = |g: Uuid, sim: &inf_player::runtime_sim::RuntimeSim| {
        let w = sim.world();
        w.entity_of(g)
            .and_then(|e| w.world().get::<inf_ecs::components::GlobalTransform>(e))
            .map(|t| {
                let f = t.0.transform_vector3(DVec3::Z);
                (t.translation(), inf_math::patan2_64(f.x, f.z).to_degrees())
            })
    };
    let mut last: BTreeMap<Uuid, DVec3> = BTreeMap::new();
    let mut born: BTreeMap<Uuid, u64> = BTreeMap::new();
    for s in 0..steps {
        sim.step_once(Default::default());
        for &(a, b) in sim.solid_contacts_started() {
            solid_events += 1;
            if chassis(a, &sim) && chassis(b, &sim) && (moving(a, &sim) || moving(b, &sim)) {
                if contacts.insert((a, b)) {
                    let say = |g: Uuid| {
                        let (p, yaw) = pose(g, &sim).unwrap_or((DVec3::ZERO, 0.0));
                        let v = last.get(&g).map(|q| (p - *q).length() * 60.0).unwrap_or(-1.0);
                        let (home, tier) = inf_ecs::traffic::traffic_of(sim.world())
                            .and_then(|t| t.records.get(&g))
                            .map(|r| (r.home, format!("{:?}/{:?}", r.tier, r.detail)))
                            .unwrap_or((DVec3::ZERO, "-".into()));
                        format!(
                            "{:?} day {:?} {tier} at ({:.1}, {:.1}) home ({:.1}, {:.1}) yaw {yaw:.0} {v:.2} m/s, resident {:.1} s",
                            inf_ecs::traffic::catalogue_row_id(g),
                            inf_ecs::traffic::day_of(g),
                            p.x,
                            p.z,
                            home.x,
                            home.z,
                            born.get(&g).map(|b| (s - b) as f64 / 60.0).unwrap_or(-1.0)
                        )
                    };
                    eprintln!(
                        "  MOVING CONTACT at island second {:.1}: {} x {}",
                        s as f64 / 60.0,
                        say(a),
                        say(b)
                    );
                }
            }
        }
        if let Some(pop) = inf_ecs::traffic::traffic_of(sim.world()) {
            let mut keys: Vec<Uuid> = pop.records.keys().copied().collect();
            let w = sim.world();
            keys.extend(w.world().iter_entities().filter_map(|e| {
                e.get::<inf_ecs::components::VehicleClass>()?;
                e.get::<inf_ecs::components::Guid>().map(|g| g.0)
            }));
            for g in keys {
                if let Some((p, _)) = pose(g, &sim) {
                    last.insert(g, p);
                    born.entry(g).or_insert(s);
                } else {
                    born.remove(&g);
                }
            }
        }
        if std::env::var_os("INF_VEH3F2B_TRACE").is_some() && (900..1080).contains(&s) && s % 6 == 0 {
            if let Some(pop) = inf_ecs::traffic::traffic_of(sim.world()) {
                let keys: Vec<Uuid> = pop.records.keys().copied().collect();
                for g in keys {
                    if inf_ecs::traffic::catalogue_row_id(g) != Some("bravado_gresley") {
                        continue;
                    }
                    let i = inf_physics::d3::traffic::probe_intent(sim.world(), sim.bridge3d(), g, 1.0 / 60.0);
                    if let Some(i) = i {
                        eprintln!(
                            "  INTENT t {:.2} gresley target {:.2} move {:?} handbrake {} lateral {:.2}",
                            s as f64 / 60.0,
                            i.target_mps,
                            i.move_input,
                            i.handbrake,
                            i.lateral_m
                        );
                    }
                }
            }
        }
        if std::env::var_os("INF_VEH3F2B_TRACE").is_some() && s % 30 == 0 {
            for e in sim.world().world().iter_entities() {
                if e.get::<inf_ecs::components::VehicleClass>().is_none() {
                    continue;
                }
                let Some(g) = e.get::<inf_ecs::components::Guid>() else {
                    continue;
                };
                if let Some((p, yaw)) = pose(g.0, &sim) {
                    if (p.x + 417.0).abs() < 12.0 && (p.z - 380.0).abs() < 14.0 {
                        let name = sim.world().name_of(e.id()).unwrap_or("?");
                        eprintln!("  TRACE t {:.1} {name} ({:.2}, {:.2}, {:.2}) yaw {yaw:.1}", s as f64 / 60.0, p.x, p.y, p.z);
                    }
                }
            }
        }
        if s % 600 != 599 {
            continue;
        }
        let w = sim.world();
        let Some(pop) = inf_ecs::traffic::traffic_of(w) else {
            continue;
        };
        for g in pop.records.keys() {
            if w.entity_of(*g).is_none() {
                continue;
            }
            let Some(id) = inf_ecs::traffic::catalogue_row_id(*g) else {
                continue;
            };
            let Some(def) = inf_ecs::roster::roster().get(id) else {
                continue;
            };
            let class = def.roster_class.map(|c| c.name()).unwrap_or("-").to_string();
            let kind = inf_ecs::roster::body_kind(w, *g);
            *census.entry((class.clone(), kind)).or_default() += 1;
            let want = match def.art {
                Some(k) if k.shell() => "shell",
                Some(_) => "imported",
                None => "primitive",
            };
            if kind != want {
                wrong.push(format!("{id} ({class}) draws {kind}, its row names {want}"));
            }
            let hero_class = matches!(class.as_str(), "coupe" | "sedan" | "suv" | "truck");
            if hero_class && kind == "primitive" {
                wrong.push(format!("{id} ({class}) draws primitives"));
            }
        }
    }
    println!(
        "=== {minutes} island minute(s) in {:.1} s wall: {solid_events} solid contact event(s), {} moving vehicle contact pair(s)",
        t0.elapsed().as_secs_f64(),
        contacts.len()
    );
    let mut shells = 0usize;
    for ((class, kind), n) in &census {
        println!("  CENSUS {class:<10} {kind:<9} {n}");
        if *kind == "shell" {
            shells += n;
        }
    }
    wrong.sort();
    wrong.dedup();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert!(shells > 0, "no traffic car drew a shell: {census:?}");
    assert!(
        contacts.is_empty(),
        "{} moving contact pair(s) -- see the log",
        contacts.len()
    );
}
