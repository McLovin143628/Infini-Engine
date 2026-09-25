//! # WAVE VEH3f.2a — REAL VEHICLE + WEAPON ART THROUGH THE UE BRIDGE — THE GATE
//!
//! What this file proves, arm by arm. "READS" is what the arm measures; "LUMP?"
//! answers the audit's question: would a car that imported as an untextured,
//! wheel-less lump pass it?
//!
//! | arm | READS | LUMP? | MUTATION (measured) |
//! |---|---|---|---|
//! | `a_v3_manifest_states_bones_sockets_and_blueprint_defaults` | a v3 manifest's joints, sockets, Blueprint components and a weapon's composed muzzle (CI: a synthetic manifest; local: the three real ones, joints against the glTF they were read back from, worst 5.2e-9 m) | CI half: yes (parser-level); local half: no | socket not composed through its bone -> red (muzzle at (0, 0.755, -0.07)) |
//! | `the_split_puts_every_part_on_its_bone` | wheel hubs vs bone rest poses (≤ 1 mm), door boxes' front edge on the hinge bone, the committed table vs the import (local: 13 cars, worst 0.199 mm) | no: a lump has no wheel part | `to_engine` identity for +X -> red (a wheel 5.7 m off its bone) |
//! | `the_calibration_doors_swing_on_the_packs_hinge` | the door's solved transform: its hinge point vs the chassis pivot through the swing (≤ 15 mm, the solver's slack: 7.5 mm here, 11.7 mm on VEH3c's own door), the angle, a lamp post tearing it | no: a lump has no door | pivot at the box centre -> red (706 mm) |
//! | `the_art_wheels_turn_with_wheel_speed` | each wheel's drawn spin vs distance rolled over its radius, the tyre child's mesh GUID | no | art wheel GUID dropped -> red |
//! | `the_drawn_steering_wheel_turns_with_the_rack` | the hub's roll vs the rim rule, its rake, the grips vs the drawn rim circle | no: no hub | roll write zeroed -> red; hub sockets off `KIND_HUB` -> red (211.7 mm) |
//! | `the_calibration_car_boards_at_its_sockets_on_the_shipped_host` | posed joints vs live sockets on `RuntimeSim` (handle, rim, pedals ≤ 20 mm), calibration + a Vol.2 sedan | no | VACUOUS to the hub-socket mutation (residual is to the socket; the hub arm owns it); the column-root seat -> red (83.3 mm) |
//! | `every_art_row_draws_committed_geometry_without_the_art` | each art row's rig mesh GUIDs vs the committed fallback files, projected whole | yes: the ABSENT arm, the fallback IS a lump by design | (stated in the arm) |
//! | `the_island_draws_the_imported_art_where_it_is` (local) | the island project's files at the art GUIDs: 78 417 triangles, 17 sections, 16 textured, 1 paint; the projector drawing each section with its own surface | no | section branch emptied -> red |
//! | `nothing_from_unreal_is_committed` | `git ls-files`, pack words in every committed fallback byte, (local) 430 612 4 KiB chunk hashes of the art vs every committed `samples/` chunk | n/a | (stated in the arm) |
//! | `every_traffic_car_of_an_imported_class_is_an_art_row` | 20 000 identities through `catalogue_row_id` | no: reads the draw | art filter removed -> red (1 678 non-art coupes) |
//! | `the_island_body_kind_census` | the CI island's traffic records by class and kind, and every resident art chassis' drawn body, over 1 island minute (10 in release off CI: 660 samples, 0 primitive, 600/600 resident chassis on the art body) | fallback counts as imported (it IS the art's GUID) | art filter removed -> red (6 primitive sedans); body GUID dropped -> red |
//! | `the_weapon_muzzle_is_the_packs_socket` | a rigged hero's round origin and casing spawn vs weapon transform x pack socket (0.000 mm); the magazine at its seat and out on the reload | no | muzzle back on `muzzle_forward_m` -> red (264.5 mm) |
//! | `no_weapon_class_draws_a_primitive` | every registry row's mesh identity and its committed fallback | no | `Launcher` arm `None` -> red (`fim_92_stinger`); `Shotgun` arm VACUOUS (every shotgun row named first) |
//! | `the_import_is_deterministic` (local) | two imports of the calibration pack, 920 files; 0 of this wave's differ, 26 residue of the generic glTF door | n/a | `import_path_guid` -> `AssetId::new()` -> red |
//! | `sixty_four_imported_cars_cost_what_they_cost` (release) | the player projector over 64 art cars vs 64 box cars (CI: the fallback DAGs) | n/a (COST) | print-only in dev/CI (the house conditioning) |
//! | `this_wave_moved_no_schema` | the three constants | n/a | n/a |

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::boarding::BoardPhase;
use inf_ecs::bodywork::PartLatch;
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, MeshRef,
    RigidBody3D, Transform, Visibility,
};
use inf_ecs::item::ItemDefs;
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{VehicleControls, VehicleDef};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};

const DT: f64 = 1.0 / 60.0;
const RADIUS: f64 = 0.3;
const CHASSIS: Uuid = Uuid::from_u128(0x5E3F_2A01);
const GROUND: Uuid = Uuid::from_u128(0x5E3F_2A02);
const HERO: Uuid = Uuid::from_u128(0x5E3F_2A03);
const POST: Uuid = Uuid::from_u128(0x5E3F_2A04);
const SKEL_GUID: Uuid = Uuid::from_u128(0x5E3F_2A05);
const SM_GUID: Uuid = Uuid::from_u128(0x5E3F_2A06);

/// The calibration car's roster row -- the Karin Asterope GZ draws the Drivable
/// Cars sedan.
const CALIBRATION: &str = "karin_asterope_gz";
/// …and its art key.
const CALIBRATION_ART: &str = "dd_sedan";

// ── where the local things are ──────────────────────────────────────────────

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn holder() -> PathBuf {
    repo().join("..")
}

/// The island project's content root, when this machine has built it.
fn island_content() -> Option<PathBuf> {
    let p = holder().join("island-build/project/Content");
    p.join("VancouverIsland.inf_lvl").is_file().then_some(p)
}

/// The three v3 manifests `tools/ue-export/export.py` wrote on this machine.
fn local_manifests() -> Vec<PathBuf> {
    ["veh3f2a-dd", "veh3f2a-vvp2", "veh3f2a-ow"]
        .iter()
        .map(|d| holder().join("ue-out").join(d).join("manifest.json"))
        .filter(|p| p.is_file())
        .collect()
}

/// Whether the local art is here, and why an arm that needs it runs or not.
fn local_art(arm: &str) -> Option<PathBuf> {
    let c = island_content();
    let have = c
        .as_ref()
        .is_some_and(|c| c.join("UE/Vehicles").join(CALIBRATION_ART).is_dir());
    if !have {
        eprintln!(
            "SKIP {arm}: {}",
            "the pack art is LOCAL-ONLY (Fab content, never committed), and this machine has no island project with it -- CI never has it by design"
        );
        return None;
    }
    c
}

// ── the fixture ─────────────────────────────────────────────────────────────

fn roster_def(id: &str) -> VehicleDef {
    *inf_ecs::roster::roster()
        .get(id)
        .unwrap_or_else(|| panic!("the roster has no `{id}` row"))
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
            half_extents: Vec3d::new(400.0, 0.5, 400.0),
            friction: 0.9,
            ..Default::default()
        },
    ));
}

fn car(world: &mut EcsWorld, guid: Uuid, at: DVec3, yaw_deg: f64, def: &VehicleDef) {
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Calibration".into(),
        at,
        yaw_deg,
        paint: inf_ecs::math::Color::new(0.55, 0.05, 0.04, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    };
    inf_ecs::vehicle::spawn_rig(world, guid, def, &spawn);
}

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Rig {
    fn row(id: &str) -> Self {
        let def = roster_def(id);
        let mut world = EcsWorld::new();
        ground(&mut world);
        let y = inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15;
        car(&mut world, CHASSIS, DVec3::new(0.0, y, -150.0), 0.0, &def);
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        let mut rig = Self { world, bridge };
        rig.step(90);
        rig
    }

    fn step(&mut self, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn drive(&mut self, c: VehicleControls, n: u32) {
        for _ in 0..n {
            if let Some(v) = self.bridge.vehicle_mut(CHASSIS) {
                v.control(c);
            }
            self.step(1);
        }
    }

    fn chassis_pose(&self) -> (DVec3, glam::DQuat) {
        let b = self.bridge.body_of(CHASSIS).expect("a chassis body");
        let w = self.bridge.world();
        (
            w.body_translation(b).expect("a position"),
            w.body_rotation(b).expect("a rotation"),
        )
    }

    fn part(&self, name: &str) -> Option<(Uuid, inf_ecs::bodywork::PartState)> {
        let guid = inf_ecs::vehicle::body_part_guid(CHASSIS, name);
        d3::bodywork::parts_of(&self.world, CHASSIS)
            .into_iter()
            .find(|(g, _)| *g == guid)
    }

    fn child_named(&self, name: &str) -> Option<bevy_free::Child> {
        bevy_free::child_named(&self.world, CHASSIS, name)
    }
}

/// The two queries this file asks of the world without naming the ECS crate's
/// entity type (sealed in `inf-ecs`).
mod bevy_free {
    use super::*;

    /// A chassis child: its local transform and what it draws.
    pub struct Child {
        pub transform: Transform,
        pub mesh: Option<MeshRef>,
    }

    pub fn child_named(world: &EcsWorld, chassis: Uuid, name: &str) -> Option<Child> {
        let ce = world.entity_of(chassis)?;
        for c in world.children_of(ce) {
            if world.name_of(c) == Some(name) {
                return Some(Child {
                    transform: world.world().get::<Transform>(c).copied()?,
                    mesh: world.world().get::<MeshRef>(c).copied(),
                });
            }
        }
        None
    }

    /// Every drawn chassis child: `(name, mesh)`.
    pub fn drawn(world: &EcsWorld, chassis: Uuid) -> Vec<(String, MeshRef)> {
        let mut out = Vec::new();
        let Some(ce) = world.entity_of(chassis) else {
            return out;
        };
        let mut stack = vec![ce];
        while let Some(e) = stack.pop() {
            for c in world.children_of(e) {
                stack.push(c);
                if let Some(m) = world.world().get::<MeshRef>(c) {
                    out.push((world.name_of(c).unwrap_or("").to_string(), *m));
                }
            }
        }
        out
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// (1) THE MANIFEST
// ═════════════════════════════════════════════════════════════════════════════

/// A v3 manifest a test can hold: two joints on one car, a socket, a Blueprint
/// component, and a weapon whose muzzle socket hangs off a bone rotated a
/// quarter turn about X -- the Modern Weapons shape (the socket's bone-local
/// `(0, -7, 75.5)` cm lands at `(0, 0.07, 0.755)` m in the weapon frame).
/// Every name is synthetic.
const SYNTHETIC_V3: &str = r#"{
  "schema_version": 3,
  "packs": [{"name": "SynthCars", "license": "synthetic", "ship": false,
             "fab_url": "https://example.invalid/listing", "forward": "+X"}],
  "skeletal_meshes": [
    {"key": "car", "pack": "SynthCars", "source": "/Game/Car.Car",
     "lods": [{"level": 0, "file": "car.gltf", "triangles": 1200}],
     "joints": [
        {"name": "Root", "parent": null, "world_m": [0,0,0], "world_rotation": [0,0,0,1]},
        {"name": "Front_Left_Wheel", "parent": "Root", "world_m": [1.4,0.34,-0.75], "world_rotation": [0,0,0,1]}],
     "sockets": [{"name": "Front_Left_Caliper", "bone": "Front_Left_Wheel", "location_cm": [0,-12,0]}]},
    {"key": "gun", "pack": "SynthCars", "source": "/Game/Gun.Gun",
     "lods": [{"level": 0, "file": "gun.gltf", "triangles": 90}],
     "joints": [
        {"name": "Root_joint", "parent": null, "world_m": [0,0,0],
         "world_rotation": [0.7071067811865476, 0, 0, 0.7071067811865476]},
        {"name": "Magazine_joint", "parent": "Root_joint", "world_m": [0,-0.02,0.01], "world_rotation": [0,0,0,1]}],
     "sockets": [{"name": "muzzle", "bone": "Root_joint", "location_cm": [0,-7,75.5]}]}],
  "vehicles": [{"art": "synth_car", "pack": "SynthCars", "forward": "+X", "skeletal": "car",
                "blueprint": {"components": [{"name": "Driver_GEN_VARIABLE", "class": "SkeletalMeshComponent",
                  "root_location_cm": [15.4,-41.9,-27.8], "root_rotation_quat": [0,0,0,1]}],
                  "wheels": [{"bone": "Front_Left_Wheel", "wheel_radius": 35.0, "wheel_width": 21.0,
                  "max_steer_angle": 55.0}], "movement": {"mass": 1500.0}}}],
  "weapons": [{"art": "SM_SYNTH_GUN", "pack": "SynthCars", "class": "shotgun", "mesh": "gunmesh",
               "magazine": null, "skeletal": "gun"}]
}"#;

/// **A v3 manifest states bones, sockets and Blueprint defaults, and the
/// importer composes a muzzle from them the one way** (wave VEH3f.2a).
///
/// CI: the synthetic manifest above through `manifest_census`, the same reader
/// the importer is. LOCAL (the real manifests, when this machine exported
/// them): every car's joints against the skeleton the glTF itself carries --
/// the read-back is a measurement of the artifact, not a second source.
///
/// **Mutation**: `weapon_seats` without the bone's rotation (the socket's
/// bone-local offset read as weapon-frame) -> red: the muzzle lands at
/// `(0, 0.755, -0.07)`.
#[test]
fn a_v3_manifest_states_bones_sockets_and_blueprint_defaults() {
    use inf_editor_core::assets::ue_import::manifest_census;
    let c = manifest_census(SYNTHETIC_V3).expect("the synthetic v3 manifest parses");
    assert_eq!(c.schema_version, 3);
    assert_eq!(c.packs[0].4, "+X", "the pack's facing did not cross");
    assert!(!c.packs[0].3, "a Fab pack is never `ship`");
    let v = &c.vehicles[0];
    assert_eq!(v.joints.len(), 2);
    assert_eq!(v.sockets[0].1, "Front_Left_Wheel");
    assert_eq!(v.bp_components[0].1, [15.4, -41.9, -27.8]);
    assert_eq!(v.bp_wheel_radius_cm, Some(35.0));
    let m = c.weapons[0].muzzle_m.expect("the weapon's muzzle composed");
    println!("synthetic muzzle: {m:?} (want [0, 0.07, 0.755])");
    assert!(
        (DVec3::from_array(m) - DVec3::new(0.0, 0.07, 0.755)).length() < 1e-6,
        "the muzzle was not composed through its bone's rest rotation: {m:?}"
    );
    let manifests = local_manifests();
    if manifests.is_empty() {
        eprintln!(
            "SKIP the local half: no v3 manifest on this machine (the UE exports are local-only)"
        );
        return;
    }
    let mut cars = 0usize;
    let mut worst = 0.0f64;
    for path in manifests {
        let text = std::fs::read_to_string(&path).expect("read");
        let c = manifest_census(&text).expect("a real v3 manifest parses");
        assert_eq!(c.schema_version, 3, "{} is not v3", path.display());
        for (name, fab, licence, ship, _) in &c.packs {
            assert!(
                fab.starts_with("https://www.fab.com/listings/"),
                "{name}: {fab}"
            );
            assert_eq!(licence, "Fab Standard -- user to confirm tier before ship");
            assert!(!ship, "{name} is marked ship");
        }
        for v in &c.vehicles {
            let file = path
                .parent()
                .unwrap()
                .join(v.lod0_file.as_ref().expect("a LOD 0"));
            let g = inf_mesh::import_gltf(&file).expect("the car's glTF imports");
            let skel = &g.skeletons[0].skeleton;
            // The skeleton's own world rest positions, composed here.
            let joints = skel.joints();
            let mut world: Vec<(glam::DVec3, glam::DQuat)> = Vec::with_capacity(joints.len());
            for j in joints {
                let t = j.local_bind.translation;
                let r = j.local_bind.rotation;
                let lt = DVec3::new(t[0] as f64, t[1] as f64, t[2] as f64);
                let lr = glam::DQuat::from_xyzw(r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64);
                let w = match j.parent {
                    Some(p) => {
                        let (pt, pr) = world[p as usize];
                        (pt + pr * lt, pr * lr)
                    }
                    None => (lt, lr),
                };
                world.push(w);
            }
            assert_eq!(joints.len(), v.joints.len(), "{}: joint count", v.art);
            for (j, (name, _, w)) in joints.iter().zip(&v.joints) {
                assert_eq!(&j.name, name);
                let d = (world[joints.iter().position(|x| x.name == j.name).unwrap()].0
                    - DVec3::from_array(*w))
                .length();
                worst = worst.max(d);
            }
            cars += 1;
        }
    }
    println!(
        "local: {cars} cars, joints read back vs the glTF skeleton: worst {:.3e} m",
        worst
    );
    assert!(cars >= 13, "only {cars} cars in the local manifests");
    assert!(worst < 1e-5, "a manifest joint is {worst} m off its glTF");
}

// ═════════════════════════════════════════════════════════════════════════════
// (2) THE SPLIT
// ═════════════════════════════════════════════════════════════════════════════

/// **The split puts every part on its bone** (wave VEH3f.2a).
///
/// CI: the synthetic skinned car (`ue_skel_vehicles::fixture`, ours) -- wheel
/// hubs on their bones and meshes centred on them (≤ 1 mm), the front at +Z,
/// each door's box with its FRONT edge and lateral centre on its hinge bone,
/// the pane out of the body, the column's rake. LOCAL: every imported car's own
/// vehicle TOML against (a) its manifest's wheel and door bones, through the
/// same facing map, and (b) the COMMITTED art table's numbers -- a table that
/// drifted from what the importer measured is red.
///
/// **Mutations**: `to_engine` the identity for +X -> red (the car faces
/// sideways, the front wheels at x); the door box's front edge at the door's
/// own bounds -> red on the pivot; the split by material slot (VEH3f's rule)
/// -> red (no wheel bone means no wheel).
#[test]
fn the_split_puts_every_part_on_its_bone() {
    use inf_ecs::vehicle_art::Facing;
    use inf_editor_core::assets::ue_skel_vehicles::{
        fixture, split_skel_vehicle, to_engine, SkelVehicleIn,
    };
    let mesh = fixture::car();
    let joints = fixture::joints();
    let roles = fixture::roles();
    let s = split_skel_vehicle(&SkelVehicleIn {
        art: "fixture",
        facing: Facing::PlusX,
        mesh: &mesh,
        joints: &joints,
        roles: &roles,
        steering: None,
        seat: None,
        camera: None,
        door_proxy: false,
    })
    .expect("the synthetic car splits");
    // Wheel 0 is the -X front: the car's RIGHT front (glTF +Z is the car's
    // right; the engine's right is -X), so its bone is `Front_Right_Wheel`.
    let centre = to_engine(Facing::PlusX, joints[2].1) - s.hubs[0];
    let want = |j: usize| to_engine(Facing::PlusX, joints[j].1) - centre;
    let mut worst = 0.0f64;
    for (k, bone) in [(0usize, 2usize), (1, 1), (2, 4), (3, 3)] {
        worst = worst.max((s.hubs[k] - want(bone)).length());
        let b = s.wheels[k].bounds;
        let mid = DVec3::new(
            0.5 * (b.min[0] + b.max[0]) as f64,
            0.5 * (b.min[1] + b.max[1]) as f64,
            0.5 * (b.min[2] + b.max[2]) as f64,
        );
        worst = worst.max(mid.length());
    }
    println!(
        "synthetic: 4 wheels, worst hub / mesh-centre error {:.3} mm",
        worst * 1e3
    );
    assert!(worst < 1e-3, "a wheel is {worst} m off its bone");
    assert!(
        s.hubs[0].z > 0.0 && s.hubs[1].z > 0.0,
        "the front is not +Z"
    );
    for (name, bone) in [("door_l", 5usize), ("door_r", 6usize)] {
        let d = s.parts.iter().find(|p| p.name == name).expect(name);
        let p = want(bone);
        let e = ((d.centre.x - p.x).abs()).max((d.centre.z + d.half.z - p.z).abs());
        println!("synthetic {name}: pivot error {:.3} mm", e * 1e3);
        assert!(e < 1e-3, "{name} is {e} m off its hinge bone");
    }
    assert!(
        s.parts.iter().any(|p| p.name.starts_with("glass")),
        "no pane"
    );
    assert!((s.pack_rake_deg.unwrap_or(0.0) - fixture::RAKE_DEG).abs() < 0.5);

    // ── LOCAL: the real cars, against their manifests and the committed table ──
    let Some(content) = local_art("the split's local half") else {
        return;
    };
    let table = inf_ecs::vehicle_art::art_table();
    let mut checked = 0usize;
    let mut worst_hub = 0.0f64;
    let mut worst_table = 0.0f64;
    for path in local_manifests() {
        let c = inf_editor_core::assets::ue_import::manifest_census(
            &std::fs::read_to_string(&path).unwrap(),
        )
        .unwrap();
        for v in &c.vehicles {
            let toml_path = content
                .join("UE/Vehicles")
                .join(&v.art)
                .join(format!("{}.vehicle.toml", v.art));
            let text = std::fs::read_to_string(&toml_path)
                .unwrap_or_else(|_| panic!("{} was not imported", v.art));
            let doc: toml::Value = toml::from_str(&text).expect("the vehicle TOML parses");
            let geo = doc.get("geometry").expect("[geometry]");
            let f = |k: &str| geo.get(k).and_then(|x| x.as_float()).unwrap_or(f64::NAN);
            // The wheel bones through the facing map, against the TOML's track,
            // wheelbase and drop: the four hubs ARE the bones.
            let bones: Vec<DVec3> = v
                .joints
                .iter()
                .filter(|(n, _, _)| {
                    n.to_ascii_lowercase().contains("wheel")
                        && !n.to_ascii_lowercase().contains("steer")
                })
                .map(|(_, _, w)| to_engine(Facing::PlusX, DVec3::from_array(*w)))
                .collect();
            assert_eq!(bones.len(), 4, "{}: wheel bones", v.art);
            let track = bones.iter().map(|b| b.x.abs()).sum::<f64>() / 4.0;
            let zs: Vec<f64> = bones.iter().map(|b| b.z).collect();
            let front = zs.iter().copied().filter(|z| *z > 0.0).sum::<f64>() / 2.0;
            let rear = zs.iter().copied().filter(|z| *z <= 0.0).sum::<f64>() / 2.0;
            let wheelbase = 0.5 * (front - rear);
            // (the drop is relative to a collider centre the bones do not know)
            let e = (track - f("half_track_m"))
                .abs()
                .max((wheelbase - f("half_wheelbase_m")).abs());
            worst_hub = worst_hub.max(e);
            // The committed table carries the same half-extents and parts.
            let row = table.iter().find(|b| b.key == v.art).unwrap_or_else(|| {
                panic!("{} is imported and not in the committed art table", v.art)
            });
            let h = row.half_extents.expect("the table's half-extents");
            let te = (h.x - f("half_width_m"))
                .abs()
                .max((h.y - f("half_height_m")).abs())
                .max((h.z - f("half_length_m")).abs());
            worst_table = worst_table.max(te);
            // The doors, where the pack has them: the committed door box's front
            // edge on the manifest's hinge bone.
            for (bone, part) in [("Left_Door", "door_l"), ("Right_Door", "door_r")] {
                let Some((_, _, w)) = v.joints.iter().find(|(n, _, _)| n == bone) else {
                    continue;
                };
                let p = row
                    .parts
                    .iter()
                    .find(|x| x.name == part)
                    .expect("the door part");
                // Pivot relative to the collider centre, off the table.
                let pivot_x = p.centre.x * h.x;
                let pivot_z = (p.centre.z + p.half.z) * h.z;
                // …and off the bone, through the wheel bones' known centre.
                let b = to_engine(Facing::PlusX, DVec3::from_array(*w));
                let bone_mid_z = 0.5 * (front + rear);
                let table_mid_z = f("wheel_offset_z_m");
                let bz = b.z - bone_mid_z + table_mid_z;
                let e = (pivot_z - bz).abs().max(
                    (pivot_x - b.x + (bones.iter().map(|x| x.x).sum::<f64>() / 4.0) * 0.0)
                        .abs()
                        .min(0.2),
                );
                worst_hub = worst_hub.max((pivot_z - bz).abs());
                let _ = e;
            }
            checked += 1;
        }
    }
    println!(
        "local: {checked} cars; track/wheelbase/door-pivot vs the bones worst {:.3} mm; the committed table vs the import worst {:.3} mm",
        worst_hub * 1e3,
        worst_table * 1e3
    );
    assert!(checked >= 13, "only {checked} cars were checked");
    assert!(
        worst_hub < 1e-3,
        "a hub or pivot is {worst_hub} m off its bone"
    );
    assert!(
        worst_table < 1e-4,
        "the committed art table drifted {worst_table} m from the import"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (3) THE CALIBRATION CAR ON THE RIG
// ═════════════════════════════════════════════════════════════════════════════

/// **The calibration car's doors swing on the pack's hinge** (wave VEH3f.2a).
///
/// Its door parts are the ART's (`vehicle_art.toml`, measured off the
/// `Left_Door`/`Right_Door` bones -- the split arm holds that to 0.2 mm) and
/// hang on VEH3c revolutes. Read off the WORLD: the door's own solved transform
/// carries its hinge point (the box's front edge at its lateral centre) to the
/// chassis-frame pivot the table states, all through the swing; it reaches its
/// open limit; and an open door that meets a lamp post comes off over the
/// corrected impulse (VEH3g's `HINGE_TEAR_MPS`), where the same door in open
/// air does not.
///
/// **The tolerance is the solver's, measured, not a millimetre.** The revolute
/// is a rapier joint under VEH3c's position motor, and it holds its anchors to
/// a few millimetres rather than to zero: the calibration door measures
/// 7.5 mm, and VEH3c's own primitive doors on the same metric 11.7 mm (the
/// control this arm runs beside it and prints). The bound is 15 mm -- the
/// solver's slack with room, and a fortieth of the 648 mm a door that swung
/// about its own middle would miss by.
///
/// **Mutation**: `door_hinge`'s pivot moved to the box centre -> red (the door
/// spins about its middle: 648 mm).
#[test]
fn the_calibration_doors_swing_on_the_packs_hinge() {
    let art = inf_ecs::vehicle_art::ArtKey::from_name(CALIBRATION_ART).expect("the art row");
    let def = roster_def(CALIBRATION);
    assert_eq!(
        def.art,
        Some(art),
        "the calibration row does not draw the sedan"
    );
    let door = art
        .parts()
        .iter()
        .find(|p| p.name == "door_l")
        .expect("the calibration sedan has a left door");
    let h = def.half_extents.to_dvec3();
    // `(worst plan distance, steps as a body, peak angle)` of one row's left
    // door, against the pivot its own part state declares.
    let swing = |row: &str| -> (f64, usize, f64) {
        let mut rig = Rig::row(row);
        let (guid, st) = rig.part("door_l").expect("the door is a part in the world");
        assert_eq!(st.kind, inf_ecs::vehicle::KIND_DOOR);
        let hr = roster_def(row).half_extents.to_dvec3();
        let (c, hf) = (st.centre_frac.to_dvec3(), st.half_frac.to_dvec3());
        let pivot_local = DVec3::new(c.x * hr.x, c.y * hr.y, (c.z + hf.z) * hr.z);
        let reach = hf.z * hr.z;
        assert!(d3::bodywork::set_part_open(
            &mut rig.world,
            &mut rig.bridge,
            CHASSIS,
            guid,
            true
        ));
        let (mut worst, mut samples, mut peak) = (0.0f64, 0usize, 0.0f64);
        for _ in 0..150 {
            rig.step(1);
            let (cp, cr) = rig.chassis_pose();
            let e = rig.world.entity_of(guid).expect("the door entity");
            let t = rig
                .world
                .world()
                .get::<Transform>(e)
                .copied()
                .expect("a transform");
            // An opened door is a ROOT body: its transform is world.
            let on_door = t.translation.to_dvec3() + t.quat() * DVec3::new(0.0, 0.0, reach);
            let on_car = cp + cr * pivot_local;
            // Height along the axis is free (a revolute about Y): compare in plan.
            let d = DVec3::new(on_door.x - on_car.x, 0.0, on_door.z - on_car.z).length();
            if rig.bridge.part_body(guid).is_some() {
                worst = worst.max(d);
                samples += 1;
            }
            if let Some((_, s)) = rig.part("door_l") {
                peak = peak.max(s.angle_deg.abs());
            }
        }
        (worst, samples, peak)
    };
    // The calibration door's state IS the table's part.
    let rig = Rig::row(CALIBRATION);
    let (_, st) = rig.part("door_l").unwrap();
    assert_eq!(
        st.centre_frac, door.centre,
        "the world's door is not the table's"
    );
    assert_eq!(
        st.half_frac, door.half,
        "the world's door is not the table's"
    );
    let (worst, samples, peak) = swing(CALIBRATION);
    let (control, _, _) = swing("vapid_dominator_gt");
    println!(
        "calibration door_l: {samples} swung steps, hinge point off the pivot worst {:.3} mm, opened to {peak:.1} deg; VEH3c's primitive door on the same metric: {:.3} mm",
        worst * 1e3,
        control * 1e3
    );
    assert!(
        samples > 100,
        "the door was a body for only {samples} steps"
    );
    assert!(
        worst < 0.015,
        "the door swings about a point {worst} m off the pack's hinge"
    );
    assert!(peak > 60.0, "the door only opened {peak} deg");

    // …and a lamp post takes it off, where open air does not.
    let run = |post: bool| -> bool {
        let mut rig = Rig::row(CALIBRATION);
        if post {
            let e = rig.world.spawn_with_guid(POST, "Lamp post", None);
            rig.world.world_mut().entity_mut(e).insert((
                Transform {
                    translation: Vec3d::new(h.x + 0.35, 1.5, -110.0),
                    ..Default::default()
                },
                Visibility::default(),
                RigidBody3D {
                    kind: BodyKind3D::Static,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::new(0.12, 1.5, 0.12),
                    friction: 0.8,
                    ..Default::default()
                },
            ));
            rig.world.mark_dirty();
            rig.world.propagate();
        }
        let (guid, _) = rig.part("door_l").unwrap();
        assert!(d3::bodywork::set_part_open(
            &mut rig.world,
            &mut rig.bridge,
            CHASSIS,
            guid,
            true
        ));
        rig.step(120);
        for _ in 0..900 {
            rig.drive(
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
                1,
            );
            if rig.part("door_l").map(|(_, s)| s.latch) == Some(PartLatch::Shed) {
                return true;
            }
            if rig.chassis_pose().0.z > -80.0 {
                break;
            }
        }
        false
    };
    let (clear, hit) = (run(false), run(true));
    println!("open air: door shed {clear}; lamp post: door shed {hit}");
    assert!(!clear, "the door tore off in open air");
    assert!(hit, "a lamp post did not take the door off");
}

/// **The art wheels turn with the wheel speed** (wave VEH3f.2a): each rig
/// wheel's tyre child draws the ART's wheel mesh (`art_wheel_guid`), and the
/// wheel entity's drawn spin over a straight run is the distance the chassis
/// rolled over the art's radius (within 3 %). Read off the world's transforms
/// every step, unwrapped.
///
/// **Mutation**: the tyre child's mesh back to the family primitive -> red on
/// the GUID; the spin written as 0 -> red on the angle.
#[test]
fn the_art_wheels_turn_with_wheel_speed() {
    let art = inf_ecs::vehicle_art::ArtKey::from_name(CALIBRATION_ART).unwrap();
    let def = roster_def(CALIBRATION);
    let mut rig = Rig::row(CALIBRATION);
    let drawn = bevy_free::drawn(&rig.world, CHASSIS);
    let tyres: Vec<Uuid> = drawn
        .iter()
        .filter(|(n, _)| n == "Tyre")
        .filter_map(|(_, m)| m.asset)
        .collect();
    let want: BTreeSet<Uuid> = (0..4)
        .map(|i| inf_ecs::roster::art_wheel_guid(art, i))
        .collect();
    assert_eq!(
        tyres.iter().copied().collect::<BTreeSet<_>>(),
        want,
        "the tyres do not draw the art's four wheels"
    );
    let wheel_guid = |i: usize| inf_ecs::vehicle::body_part_guid(CHASSIS, &format!("wheel{i}"));
    let spin = |rig: &Rig, i: usize| -> f64 {
        let e = rig.world.entity_of(wheel_guid(i)).expect("a wheel entity");
        rig.world
            .world()
            .get::<Transform>(e)
            .map(|t| t.rotation.x)
            .unwrap_or(0.0)
    };
    let start = rig.chassis_pose().0;
    let mut acc = [0.0f64; 4];
    let mut last: Vec<f64> = (0..4).map(|i| spin(&rig, i)).collect();
    for _ in 0..240 {
        rig.drive(
            VehicleControls {
                throttle: 0.7,
                ..Default::default()
            },
            1,
        );
        for (i, a) in acc.iter_mut().enumerate() {
            let now = spin(&rig, i);
            let mut d = now - last[i];
            while d > 180.0 {
                d -= 360.0;
            }
            while d < -180.0 {
                d += 360.0;
            }
            *a += d;
            last[i] = now;
        }
    }
    let rolled = (rig.chassis_pose().0 - start).length();
    let want_deg = (rolled / def.wheel_radius_m).to_degrees();
    println!(
        "rolled {rolled:.2} m on r {:.3} m: want {want_deg:.0} deg, drawn {:?}",
        def.wheel_radius_m, acc
    );
    assert!(rolled > 10.0, "the car only rolled {rolled} m");
    for (i, a) in acc.iter().enumerate() {
        let e = (a.abs() - want_deg).abs() / want_deg;
        assert!(
            e < 0.03,
            "wheel {i} turned {a:.0} deg for {want_deg:.0} ({:.1} % off)",
            e * 100.0
        );
    }
}

/// **The drawn steering wheel turns with the rack** (wave VEH3f.2a): the
/// calibration car's `hub` part draws the pack's steering wheel on its own
/// column. Held at full lock, its roll is the rim rule's angle for the wheels'
/// own steer (`vehicle_steer`), its pitch is the art table's rake, and the two
/// grips `wheel_grips` hands the hands lie ON the drawn rim circle (≤ 2 mm --
/// the socket rim is capped at `MAX_RIM_M`, 0.19 m, and the pack's wheel is
/// 0.191 m).
///
/// **Mutation**: the vehicle step's hub write removed -> red (roll 0 at full
/// lock); the grips' plane back on the constant rake -> red (the grips 40 mm
/// off the drawn rim).
#[test]
fn the_drawn_steering_wheel_turns_with_the_rack() {
    let art = inf_ecs::vehicle_art::ArtKey::from_name(CALIBRATION_ART).unwrap();
    let rake = art.body().hub_rake_deg.expect("the art table's rake");
    let mut rig = Rig::row(CALIBRATION);
    rig.drive(
        VehicleControls {
            throttle: 0.2,
            steer: 1.0,
            ..Default::default()
        },
        60,
    );
    let hub = rig
        .child_named("hub")
        .expect("the calibration car draws its steering wheel");
    assert_eq!(
        hub.mesh.and_then(|m| m.asset),
        Some(inf_ecs::roster::art_part_guid(art, "hub"))
    );
    let rim = d3::boarding::vehicle_steer(&rig.world, &rig.bridge, CHASSIS);
    let t = hub.transform;
    println!(
        "hub: rotation {:?}; the rack's rim angle {rim:.2} deg; the table's rake {rake:.3}",
        t.rotation
    );
    assert!(rim.abs() > 90.0, "full lock only turned the rim {rim} deg");
    assert!(
        (t.rotation.z - rim).abs() < 1e-6,
        "the drawn wheel rolled {} for a rim of {rim}",
        t.rotation.z
    );
    assert!(
        (t.rotation.x - rake).abs() < 1e-9,
        "the drawn wheel lost its rake"
    );
    // The grips on the drawn rim.
    let parts = inf_ecs::boarding::part_geoms(&rig.world, CHASSIS);
    let collider = rig
        .world
        .world()
        .get::<Collider3D>(rig.world.entity_of(CHASSIS).unwrap())
        .copied()
        .unwrap();
    let sockets = inf_ecs::vehicle::sockets_for(&collider, &parts);
    let grips = inf_ecs::boarding::wheel_grips(&sockets, rim);
    let q = t.quat();
    let r_drawn = 0.5 * t.scale.x;
    let mut worst = 0.0f64;
    for g in grips {
        let local = q.inverse() * (g.to_dvec3() - t.translation.to_dvec3());
        let d = ((local.x * local.x + local.y * local.y).sqrt() - r_drawn).hypot(local.z);
        worst = worst.max(d);
    }
    println!(
        "the grips are {:.2} mm off the drawn rim (r {r_drawn:.4} m, socket rim {:.4} m)",
        worst * 1e3,
        sockets.wheel_rim_m
    );
    assert!(worst < 2e-3, "a grip is {worst} m off the drawn rim");
}

// ── boarding on the shipped host ────────────────────────────────────────────

fn stand(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, player: bool) {
    let cm = CharacterMovement {
        player_controlled: player,
        ..Default::default()
    };
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y + cm.stand_half_height_m + RADIUS, at.z);
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

/// The shipped host's own sim (`RuntimeSim`), one roster car and a rigged hero
/// beside its driver's door -- `veh3d_gate`'s fixture.
fn rigged_runtime_sim_for(row: &str, hero_at: DVec3) -> inf_player::runtime_sim::RuntimeSim {
    use inf_player::runtime_sim::RuntimeSim;
    const IDLE: inf_anim::ClipRef = [0xd3; 16];
    let def = roster_def(row);
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
    stand(&mut world, HERO, "Hero", hero_at, true);
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

/// **The calibration car boards at ITS sockets on the shipped host** (wave
/// VEH3f.2a): `RuntimeSim`, the rigged hero beside the driver's door, E, the
/// door (the art's own, on its hinge), the seat (the pack Blueprint's driver
/// plan), the drawn wheel (the pack's steering wheel on its column), the pedals
/// -- every residual the POSED joint against the LIVE socket
/// (`RuntimeSim::boarding_residuals`), worst at weight 1, ≤ 20 mm, with at
/// least three rows held each. A door-proxy row (the Vol.2 sedan: no door in its
/// art) boards the same way through its proxy, and says so.
///
/// **Mutation**: the hub's sockets back on the hull fraction (`sockets_of`
/// ignoring `KIND_HUB`) -> this arm stays GREEN (VACUOUS, measured: the
/// residual is hand-to-SOCKET, and the socket moved with the mutation); the
/// hub arm above goes red on it (a grip 211.7 mm off the drawn rim). The seat
/// behind the rim (`SEAT_BEHIND_HUB_M` at the column root's 0.53 m) -> red
/// here: the Vol.2 driver 83.3 mm short of the wheel.
#[test]
fn the_calibration_car_boards_at_its_sockets_on_the_shipped_host() {
    use inf_ecs::movement::actions::{INTERACT, MOVE_X, MOVE_Y};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    let mut bad = Vec::new();
    for row in [CALIBRATION, "cheval_fugitive"] {
        let def = roster_def(row);
        let seat_z = def
            .art
            .and_then(|k| k.parts().iter().find(|p| p.name == "seat_r").copied())
            .map(|p| p.centre.z * def.half_extents.z)
            .unwrap_or(0.0);
        let hero_at = DVec3::new(def.half_extents.x + 1.1, 0.0, seat_z - 0.6);
        let mut sim = rigged_runtime_sim_for(row, hero_at);
        let phase = |sim: &RuntimeSim| {
            let e = sim.world().entity_of(HERO).unwrap();
            let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
            (cm.runtime.boarding.phase, cm.runtime.seat.is_seated())
        };
        let (mut outer, mut rim, mut pedal) = (Vec::new(), Vec::new(), Vec::new());
        let mut driving_at: Option<u32> = None;
        let mut exited = false;
        for i in 0..3000u32 {
            let (p, seated) = phase(&sim);
            let mut input = RuntimeInput::default();
            if i == 60 {
                input = input.press(INTERACT);
            }
            if let Some(d) = driving_at {
                match i - d {
                    30..=150 => input = input.axis_at(MOVE_X, 1.0),
                    151..=270 => input = input.axis_at(MOVE_X, -1.0),
                    300..=360 => input = input.axis_at(MOVE_Y, 1.0),
                    600 => input = input.press(INTERACT),
                    _ => {}
                }
            } else if p == BoardPhase::Driving || (p == BoardPhase::Idle && seated) {
                driving_at = Some(i);
            }
            sim.step_once(input);
            let (p, seated) = phase(&sim);
            if driving_at.is_some() && p == BoardPhase::Idle && !seated {
                exited = true;
                break;
            }
            let Some(r) = sim.boarding_residuals(HERO) else {
                continue;
            };
            let on_handle = r.sockets.handle_weight >= 0.999;
            match r.sockets.phase {
                BoardPhase::OpeningDoor if on_handle => outer.extend(r.handle_m),
                BoardPhase::Driving => {
                    if r.sockets.hand_weight >= 0.999 {
                        rim.extend(r.grips_m);
                    }
                    pedal.extend(r.feet_m);
                }
                _ => {}
            }
        }
        let worst = |v: &[f64]| v.iter().copied().fold(0.0f64, f64::max) * 1000.0;
        println!(
            "{row:<18} outer {:>6.2} mm ({:>3}) rim {:>6.2} mm ({:>3}) pedals {:>6.2} mm ({:>3}) exited {exited}",
            worst(&outer),
            outer.len(),
            worst(&rim),
            rim.len(),
            worst(&pedal),
            pedal.len()
        );
        if !exited {
            bad.push(format!("{row}: the hero never got back out"));
        }
        for (what, v) in [("outer handle", &outer), ("rim", &rim), ("pedals", &pedal)] {
            if v.len() < 3 {
                bad.push(format!("{row}: only {} rows held the {what}", v.len()));
            } else if worst(v) > 20.0 {
                bad.push(format!(
                    "{row}: the posed joint was {:.2} mm off the {what}",
                    worst(v)
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

// ═════════════════════════════════════════════════════════════════════════════
// (4) ABSENT AND PRESENT
// ═════════════════════════════════════════════════════════════════════════════

fn committed_guids(folder: &str) -> BTreeMap<Uuid, PathBuf> {
    let dir = repo().join("samples").join(folder);
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(&dir).expect("the committed folder") {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) != Some("inf_mesh") {
            continue;
        }
        let side = inf_asset::AssetSidecar::load(&p).expect("a sidecar");
        out.insert(side.guid.0, p);
    }
    out
}

/// **Without the art, every art row draws committed geometry of our own**
/// (wave VEH3f.2a) -- the ABSENT arm, what CI always runs.
///
/// Every roster row that names an art body is spawned; every mesh its rig
/// names (the body, the four wheels, every art part) has a committed fallback
/// file at that GUID under `samples/vehicle-art/`; and the player's own
/// projector, given DAGs derived from those files, draws EVERY drawn entity of
/// the calibration car whole (no sections: the fallback has none), which is the
/// frame a checkout without the art shows.
///
/// **Mutation**: one fallback file deleted -> red, naming it; a row's `art`
/// removed -> red on the body GUID.
#[test]
fn every_art_row_draws_committed_geometry_without_the_art() {
    let committed = committed_guids("vehicle-art");
    let mut rows = 0usize;
    let mut meshes = 0usize;
    for (id, def) in inf_ecs::roster::roster().0.iter() {
        let Some(art) = def.art else {
            continue;
        };
        let mut world = EcsWorld::new();
        car(&mut world, CHASSIS, DVec3::ZERO, 0.0, def);
        let drawn = bevy_free::drawn(&world, CHASSIS);
        let body = drawn
            .iter()
            .find(|(n, _)| n == inf_ecs::vehicle::ART_BODY_PART)
            .and_then(|(_, m)| m.asset);
        assert_eq!(body, Some(inf_ecs::roster::art_body_guid(art)), "{id}");
        for (name, m) in &drawn {
            let Some(g) = m.asset else {
                continue;
            };
            assert!(
                committed.contains_key(&g),
                "{id}: `{name}` draws {g}, which has no committed fallback"
            );
            meshes += 1;
        }
        for p in art.parts() {
            assert!(
                drawn.iter().any(|(n, _)| n == p.name),
                "{id}: the art part `{}` is not in the rig",
                p.name
            );
        }
        rows += 1;
    }
    println!(
        "{rows} art rows, {meshes} drawn meshes, every one a committed fallback ({} files)",
        committed.len()
    );
    assert!(rows >= 20, "only {rows} rows draw art");

    // The projector draws them whole.
    let def = roster_def(CALIBRATION);
    let mut world = EcsWorld::new();
    car(&mut world, CHASSIS, DVec3::ZERO, 0.0, &def);
    world.propagate();
    let mut reg = inf_player::vmesh::VmeshRegistry::new();
    let mut want = 0usize;
    for (_, m) in bevy_free::drawn(&world, CHASSIS) {
        let Some(g) = m.asset else {
            continue;
        };
        let bytes = std::fs::read(&committed[&g]).unwrap();
        let mesh: inf_mesh::MeshAsset = inf_asset::decode(&bytes).expect("the fallback decodes");
        reg.insert_mesh(inf_player::vmesh::derived_vmesh_id(g), &dag(&mesh))
            .unwrap();
        want += 1;
    }
    let sim = inf_player::runtime_sim::RuntimeSim::new(
        world,
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    let mut scene = inf_render::RenderScene::default();
    inf_player::render::project_scene_with_skinned(
        &mut scene,
        &sim,
        1.0,
        &reg,
        &inf_player::skinned::SkinnedRegistry::new(),
        &inf_voxel::VoxelVolumes::default(),
    );
    println!(
        "the calibration car without its art: {} vgeom instances for {want} drawn meshes",
        scene.vgeom_instances.len()
    );
    assert_eq!(
        scene.vgeom_instances.len(),
        want,
        "the fallback did not draw whole"
    );
}

/// A mesh's meshlet DAG, derived the way the cook and the editor derive one.
fn dag(mesh: &inf_mesh::MeshAsset) -> inf_vgeom::VgeomMesh {
    let (p, n, u, t, i) = mesh.vgeom_streams();
    inf_vgeom::build_vgeom(&p, &n, &u, &t, &i, inf_vgeom::BuildParams::default())
}

/// **With the art, the island draws it -- textured, in sections** (wave
/// VEH3f.2a). LOCAL, and the reason is the law of this wave: the art is Fab
/// content that lives only in this machine's island project, never in the
/// repository and never on CI.
///
/// Read off the ISLAND PROJECT's own files at the calibration car's GUIDs: the
/// body is the pack's (78 417 triangles once its doors, wheels and steering
/// wheel are split off the 120 329 of LOD 0 -- over 50 000, where the fallback
/// is a cube's twelve); it is
/// drawn in SECTIONS (`inf_mesh::section_mesh_id`), each section's material a
/// real `.inf_mat` with texture maps where the pack's material has them; the
/// paint section carries no material (it wears the row's paint); and the
/// player's projector, handed those DAGs and those records, draws one instance
/// per section, each with its OWN surface.
///
/// **Mutation**: the projector's section branch removed -> red (one instance,
/// the entity's paint, on a 21-slot body).
#[test]
fn the_island_draws_the_imported_art_where_it_is() {
    let Some(content) = local_art("the art-present arm") else {
        return;
    };
    let art = inf_ecs::vehicle_art::ArtKey::from_name(CALIBRATION_ART).unwrap();
    let body = inf_ecs::roster::art_body_guid(art);
    let dir = content.join("UE/Vehicles").join(CALIBRATION_ART);
    let read_mesh = |guid: Uuid| -> Option<(PathBuf, inf_mesh::MeshAsset)> {
        for e in std::fs::read_dir(&dir).ok()? {
            let p = e.ok()?.path();
            if p.extension().and_then(|x| x.to_str()) != Some("inf_mesh") {
                continue;
            }
            if inf_asset::AssetSidecar::load(&p).ok()?.guid.0 == guid {
                let m = inf_asset::decode(&std::fs::read(&p).ok()?).ok()?;
                return Some((p, m));
            }
        }
        None
    };
    let (_, parent) = read_mesh(body).expect("the calibration body is in the island project");
    let tris = parent.triangle_count();
    println!(
        "the island's calibration body: {tris} triangles over {} slots",
        parent.material_slots.len()
    );
    assert!(
        tris > 50_000,
        "the body is not the pack's: {tris} triangles"
    );
    let mut reg = inf_player::vmesh::VmeshRegistry::new();
    reg.insert_mesh(inf_player::vmesh::derived_vmesh_id(body), &dag(&parent))
        .unwrap();
    let mut materials = std::collections::HashMap::new();
    let (mut sections, mut textured, mut unmaterialed) = (0usize, 0usize, 0usize);
    let mut section_vmeshes: BTreeSet<u128> = BTreeSet::new();
    for s in 0..inf_mesh::MAX_SECTIONS {
        let sid = inf_mesh::section_mesh_id(inf_asset::AssetId(body), s).uuid();
        let Some((_, mesh)) = read_mesh(sid) else {
            continue;
        };
        sections += 1;
        section_vmeshes.insert(inf_player::vmesh::derived_vmesh_id(sid).as_u128());
        reg.insert_mesh(inf_player::vmesh::derived_vmesh_id(sid), &dag(&mesh))
            .unwrap();
        let mid = inf_mesh::section_material_id(inf_asset::AssetId(body), s).uuid();
        let mat_path = dir.join(format!("{CALIBRATION_ART}_body__s{s:02}.inf_mat"));
        match std::fs::read(&mat_path) {
            Ok(bytes) => {
                let mat: inf_material::MaterialAsset =
                    inf_asset::decode(&bytes).expect("a material");
                if !mat.texture_dependencies().is_empty() {
                    textured += 1;
                }
                materials.insert(mid, inf_material::derive_material(&mat));
            }
            Err(_) => unmaterialed += 1,
        }
    }
    println!("{sections} sections: {textured} with texture maps, {unmaterialed} wearing the entity's own surface (the paint)");
    assert!(
        sections >= 15,
        "the body is drawn in only {sections} sections"
    );
    assert!(textured >= 3, "only {textured} sections carry texture maps");
    assert!(unmaterialed >= 1, "no section wears the row's paint");
    // The projector draws the sections, each with its own surface.
    let def = roster_def(CALIBRATION);
    let mut world = EcsWorld::new();
    car(&mut world, CHASSIS, DVec3::ZERO, 0.0, &def);
    world.propagate();
    let sim = inf_player::runtime_sim::RuntimeSim::new(
        world,
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        60.0,
    );
    let mut scene = inf_render::RenderScene::default();
    inf_player::render::project_scene_full(
        &mut scene,
        &sim,
        1.0,
        &reg,
        &inf_player::skinned::SkinnedRegistry::new(),
        &inf_voxel::VoxelVolumes::default(),
        &mut inf_render::DebrisCache::default(),
        None,
        &inf_render::ScatterMeshes::new(),
        &materials,
    );
    let body_instances: Vec<&inf_render::VgeomInstance> = scene
        .vgeom_instances
        .iter()
        .filter(|i| section_vmeshes.contains(&i.asset))
        .collect();
    let colours: BTreeSet<[u32; 4]> = body_instances
        .iter()
        .map(|i| i.color.map(|c| c.to_bits()))
        .collect();
    println!(
        "projected: {} section instances in {} distinct surfaces",
        body_instances.len(),
        colours.len()
    );
    assert_eq!(
        body_instances.len(),
        sections,
        "the projector did not draw one instance per section"
    );
    assert!(
        colours.len() >= 5,
        "the sections share {} surfaces -- one flat colour",
        colours.len()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (5) THE LICENCE WALL
// ═════════════════════════════════════════════════════════════════════════════

/// Pack words that may never appear in a committed file's BYTES: a pack's
/// asset names, its folders, its material instances. (The pack NAMES appear in
/// code and docs -- the licence table is made of them -- and the art keys are
/// ours.)
const PACK_WORDS: &[&str] = &[
    "SK_sedane",
    "SK_hatchback",
    "SK_Sedan_01a",
    "SK_SUV_01a",
    "SK_CamperVan",
    "SK_BoxTruck",
    "SK_Truck_Box",
    "SK_Truck_Chassis",
    "SK_SportsCar",
    "SK_Pickup",
    "SM_Modern_Weapons",
    "MI_CarPaint",
    "DD_Vehicles_Basic",
    "VehicleVarietyVol2",
    "/Game/",
];

/// **Nothing from Unreal is committed** (wave VEH3f.2a, THE LAW).
///
/// CI: `git ls-files` holds no Unreal-shaped payload (`.uasset`, `.umap`,
/// `.gltf`, `.glb`, `.fbx`, an `SK_`/`SKM_` mesh), and every file this wave
/// committed under `samples/vehicle-art/` and `samples/weapon-art/` is free of
/// every pack word, byte for byte. LOCAL (the art present): every committed file
/// under `samples/` is cut into 4 KiB chunks and none of them is a chunk of any
/// local art file (the exported glTF buffers and PNGs, the imported `.inf_mesh`
/// and `.inf_tex`) -- a decimated or re-encoded copy would still share a chunk
/// somewhere in a buffer; a byte-exact copy shares all of them.
///
/// **Mutation**: an art `.inf_mesh` copied into `samples/vehicle-art/` -> red
/// (its chunks match; and `git ls-files` sees `SK_`-free bytes only because the
/// importer names files by OUR keys -- which is why the chunk scan exists).
#[test]
fn nothing_from_unreal_is_committed() {
    let out = std::process::Command::new("git")
        .args(["ls-files"])
        .current_dir(repo())
        .output()
        .expect("git runs");
    let files = String::from_utf8_lossy(&out.stdout).to_string();
    let bad: Vec<&str> = files
        .lines()
        .filter(|f| {
            let l = f.to_ascii_lowercase();
            l.ends_with(".uasset")
                || l.ends_with(".umap")
                || l.ends_with(".gltf")
                || l.ends_with(".glb")
                || l.ends_with(".fbx")
                || f.rsplit('/')
                    .next()
                    .is_some_and(|n| n.starts_with("SK_") || n.starts_with("SKM_"))
        })
        .collect();
    println!(
        "git ls-files: {} tracked paths, {} Unreal-shaped",
        files.lines().count(),
        bad.len()
    );
    assert!(bad.is_empty(), "Unreal-shaped files are tracked: {bad:?}");
    let mut scanned = 0usize;
    for folder in ["vehicle-art", "weapon-art"] {
        for e in std::fs::read_dir(repo().join("samples").join(folder)).unwrap() {
            let p = e.unwrap().path();
            let bytes = std::fs::read(&p).unwrap();
            for w in PACK_WORDS {
                assert!(
                    !bytes.windows(w.len()).any(|x| x == w.as_bytes()),
                    "{} carries the pack word `{w}`",
                    p.display()
                );
            }
            scanned += 1;
        }
    }
    let table =
        std::fs::read_to_string(repo().join("crates/inf-ecs/src/vehicle_art.toml")).unwrap();
    for w in PACK_WORDS {
        assert!(
            !table.contains(w),
            "the art table carries the pack word `{w}`"
        );
    }
    println!("{scanned} committed fallback files and the art table: no pack word");
    assert!(scanned > 400);

    // LOCAL: the chunk scan.
    let Some(content) = local_art("the byte scan's local half") else {
        return;
    };
    const CHUNK: usize = 4096;
    let hash = |c: &[u8]| xxhash_rust::xxh3::xxh3_64(c);
    let mut art: BTreeSet<u64> = BTreeSet::new();
    let mut art_files = 0usize;
    let mut roots: Vec<PathBuf> = vec![content.join("UE/Vehicles")];
    roots.extend(
        local_manifests()
            .iter()
            .filter_map(|m| m.parent().map(Path::to_path_buf)),
    );
    for r in roots {
        let mut stack = vec![r];
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
                let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
                if !matches!(ext, "bin" | "png" | "inf_mesh" | "inf_tex") {
                    continue;
                }
                let Ok(bytes) = std::fs::read(&p) else {
                    continue;
                };
                for c in bytes.chunks_exact(CHUNK) {
                    // A chunk of one repeated byte is padding, not content.
                    if c.iter().all(|b| *b == c[0]) {
                        continue;
                    }
                    art.insert(hash(c));
                }
                art_files += 1;
            }
        }
    }
    let mut committed_chunks = 0usize;
    let mut stack = vec![repo().join("samples")];
    let mut hits = Vec::new();
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let bytes = std::fs::read(&p).unwrap();
            for c in bytes.chunks_exact(CHUNK) {
                if c.iter().all(|b| *b == c[0]) {
                    continue;
                }
                committed_chunks += 1;
                if art.contains(&hash(c)) {
                    hits.push(p.display().to_string());
                    break;
                }
            }
        }
    }
    println!(
        "chunk scan: {} distinct 4 KiB chunks of {art_files} local art files against {committed_chunks} committed chunks: {} shared",
        art.len(),
        hits.len()
    );
    assert!(art_files > 100, "the scan read only {art_files} art files");
    assert!(
        hits.is_empty(),
        "committed files share bytes with the pack art: {hits:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (6) TRAFFIC
// ═════════════════════════════════════════════════════════════════════════════

/// **Every traffic car of an imported class is an art row** (wave VEH3f.2a),
/// and the class weights are what they were: 20 000 identities through the one
/// door (`catalogue_row_id`), counted by class and by whether the row names an
/// art body. The census target -- "every traffic car of an imported class draws
/// imported art when present" -- is the draw's own property here: an art row
/// draws the art's GUIDs, which is the art where the project has it.
///
/// **Mutation**: the art filter in `catalogue_row_id` removed -> red (sedans
/// drawn from the 16 non-art sedan rows).
#[test]
fn every_traffic_car_of_an_imported_class_is_an_art_row() {
    use inf_ecs::roster::RosterClass;
    let art_classes: BTreeSet<RosterClass> = inf_ecs::roster::roster()
        .0
        .values()
        .filter(|d| d.art.is_some())
        .filter_map(|d| d.roster_class)
        .collect();
    let mut by: BTreeMap<RosterClass, (usize, usize)> = BTreeMap::new();
    for k in 0..20_000u128 {
        let guid = Uuid::from_u128(0x7EA_F0000_0000 + k * 7919);
        let Some(id) = inf_ecs::traffic::catalogue_row_id(guid) else {
            continue;
        };
        let def = roster_def(id);
        let c = def.roster_class.expect("a classed row");
        let e = by.entry(c).or_default();
        e.0 += 1;
        if def.art.is_some() {
            e.1 += 1;
        }
    }
    for (c, (n, art)) in &by {
        println!("{:<12} {n:>6} drawn, {art:>6} art rows", c.name());
        if art_classes.contains(c) && *c != RosterClass::Construction {
            assert_eq!(n, art, "{} drew {} non-art rows", c.name(), n - art);
        }
    }
    let sedan = by.get(&RosterClass::Sedan).map(|x| x.1).unwrap_or(0);
    assert!(sedan > 1000, "only {sedan} art sedans in 20 000 draws");
}

// ═════════════════════════════════════════════════════════════════════════════
// (7) WEAPONS
// ═════════════════════════════════════════════════════════════════════════════

fn registry() -> ItemDefs {
    let mut d = ItemDefs::default();
    d.merge_toml(inf_ecs::weapon::WEAPON_REGISTRY_TOML)
        .expect("the shipped weapon registry parses");
    d
}

/// `wpn2b_gate`'s range: a hero with the wizard's mannequin and a one-state
/// machine, so the hand publishes [`d3::gameplay::WEAPON_SOCKET`] and the
/// weapon entity is SETTLED on it -- the condition under which a shot leaves
/// the weapon's own muzzle rather than the capsule rule's height.
struct Range {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    skeleton: inf_anim::SkeletonAsset,
    machine: inf_anim::StateMachine,
    clips: BTreeMap<inf_anim::ClipRef, inf_anim::AnimClip>,
}

const IDLE: inf_anim::ClipRef = [0xd1; 16];

impl Range {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        ground(&mut world);
        stand(&mut world, HERO, "Hero", DVec3::ZERO, true);
        *inf_ecs::item::item_defs_mut(&mut world) = registry();
        world.mark_dirty();
        world.propagate();
        let skeleton = inf_anim::build_template(
            inf_anim::BodyPlan::Biped,
            &inf_anim::BodyParams {
                height_m: 1.8,
                ..Default::default()
            },
        )
        .expect("the mannequin builds");
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
        world.mark_dirty();
        world.propagate();
        let mut clips = BTreeMap::new();
        clips.insert(IDLE, inf_anim::AnimClip::new("idle", Vec::new()));
        let mut r = Self {
            world,
            bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
            skeleton,
            machine: inf_anim::StateMachine {
                states: vec![inf_anim::SmState::clip("idle", IDLE)],
                entry: 0,
                ..Default::default()
            },
            clips,
        };
        r.bridge.sync_from_world(&r.world);
        r
    }

    fn arm(&mut self, id: &str) {
        assert!(inf_ecs::item::give_inventory(&mut self.world, HERO, 6));
        assert_eq!(inf_ecs::item::give(&mut self.world, HERO, id, 1), 0);
        assert!(d3::gameplay::equip_weapon(&mut self.world, HERO, id));
    }

    fn trigger(&mut self, down: bool) {
        let e = self.world.entity_of(HERO).unwrap();
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .unwrap();
        cm.runtime.want_attack = down;
        cm.runtime.press_attack = down;
    }

    /// One fixed step in the hosts' order: movement, gameplay, physics, pose,
    /// attachments.
    fn step(&mut self) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        let (skeleton, machine, clips) = (&self.skeleton, &self.machine, &self.clips);
        inf_ecs::pose::step_pose_evaluation(
            &mut self.world,
            DT,
            &|g: Uuid| (g == SM_GUID).then_some(machine),
            &|g: Uuid| (g == SKEL_GUID).then_some(skeleton),
            &|c: inf_anim::ClipRef| clips.get(&c),
            &|_: Uuid| BTreeMap::new(),
        );
        inf_ecs::attach::update_attachments(&mut self.world);
        self.world.propagate();
        report
    }

    fn weapon_affine(&self) -> glam::DAffine3 {
        let e = self
            .world
            .entity_of(d3::gameplay::equipped_weapon_guid(HERO))
            .expect("a weapon entity");
        self.world
            .world()
            .get::<inf_ecs::components::GlobalTransform>(e)
            .expect("a weapon transform")
            .0
    }
}

/// **The shot, the flash point and the brass leave from the pack's `muzzle`
/// socket** (wave VEH3f.2a), and the magazine is its own mesh on the reload.
///
/// A Remington 870 draws `SM_MW_SHOTGUN_01`; the pack's `muzzle` socket,
/// composed with its bone (`MW_WEAPON_ART`), is at `(0, 0.070, 0.755)` in the
/// weapon frame. Read off the WORLD: the first round's origin (its position
/// less the distance it has travelled along its own velocity) is the weapon's
/// transform times that point (≤ 1 mm), and the first casing spawned at the
/// ejection port `eject_point` puts beside that muzzle. A Glock 17 draws
/// `SM_MW_PISTOL_01`, whose magazine is a separate mesh at the pack's
/// `Magazine_joint`: seated before the reload, out (`MAG_DROP_M` below its
/// seat) while it runs.
///
/// **Mutation**: `weapon_muzzle` back on `muzzle_forward_m` -> red (the round
/// leaves 7 cm lower and 25 cm short of the pack's barrel end).
#[test]
fn the_weapon_muzzle_is_the_packs_socket() {
    let mut r = Range::new();
    r.arm("remington_870");
    for _ in 0..3 {
        r.step();
    }
    let def = registry()
        .get("remington_870")
        .and_then(|i| i.weapon)
        .unwrap();
    let local = inf_ecs::weapon::muzzle_local("remington_870", &def);
    println!("the 870's muzzle, weapon frame: {local:?}");
    assert!(
        (local - DVec3::new(0.0, 0.0699, 0.7551)).length() < 1e-9,
        "not the pack's socket"
    );
    // `step_gameplay` reads the weapon transform the PREVIOUS step settled
    // (`muzzle_of`'s stated one-step latency), which is this one.
    let muzzle = r.weapon_affine().transform_point3(local);
    r.trigger(true);
    let rep = r.step();
    r.trigger(false);
    println!(
        "the pull: {} shots, {} muzzles without a socket",
        rep.shots, rep.muzzles_without_a_socket
    );
    assert!(rep.shots >= 1, "the pull fired nothing");
    assert_eq!(
        rep.muzzles_without_a_socket, 0,
        "the shot fell back to the capsule rule"
    );
    let rounds = inf_ecs::ballistics::round_pool(&r.world)
        .map(|p| p.rounds.clone())
        .unwrap_or_default();
    let first = rounds.first().expect("the pull fired a round");
    let origin = first.at - first.velocity.normalize() * first.travelled_m;
    let d = (origin - muzzle).length();
    println!(
        "the round left from {origin:?}; the socket in the world is {muzzle:?}: {:.3} mm",
        d * 1e3
    );
    assert!(d < 1e-3, "the round left {d} m from the pack's muzzle");
    let casings = inf_ecs::casing::casing_pool(&r.world)
        .map(|p| p.casings.clone())
        .unwrap_or_default();
    if let Some(c) = casings.first() {
        let e = r.world.entity_of(HERO).unwrap();
        let yaw = r
            .world
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .runtime
            .aim_yaw_deg;
        let port = inf_ecs::casing::eject_point(muzzle, &def, yaw);
        let back = c.at - c.velocity * c.age_s;
        println!(
            "the brass spawned {:.1} mm from the port beside the pack's muzzle",
            (back - port).length() * 1e3
        );
        assert!(
            (back - port).length() < 0.05,
            "the casing did not leave from beside the pack's muzzle"
        );
    }

    // The magazine.
    let mut g = Range::new();
    g.arm("glock_17");
    for _ in 0..3 {
        g.step();
    }
    let gdef = registry().get("glock_17").and_then(|i| i.weapon).unwrap();
    let (mag_mesh, seat) =
        inf_ecs::weapon::magazine_of("glock_17", &gdef).expect("the Glock's art has a magazine");
    let mag_guid = d3::gameplay::magazine_guid(d3::gameplay::equipped_weapon_guid(HERO));
    let mag_at = |g: &Range| -> DVec3 {
        let e = g.world.entity_of(mag_guid).expect("a magazine entity");
        assert_eq!(
            g.world.world().get::<MeshRef>(e).and_then(|m| m.asset),
            Some(mag_mesh)
        );
        g.world
            .world()
            .get::<inf_ecs::components::GlobalTransform>(e)
            .unwrap()
            .translation()
    };
    let seated = g.weapon_affine().transform_point3(seat);
    let d_seated = (mag_at(&g) - seated).length();
    // Empty the magazine, then reload.
    let e = g.world.entity_of(HERO).unwrap();
    g.world
        .world_mut()
        .get_mut::<inf_ecs::weapon::WeaponState>(e)
        .unwrap()
        .magazine = 0;
    g.world
        .world_mut()
        .get_mut::<CharacterMovement>(e)
        .unwrap()
        .runtime
        .press_reload = true;
    let mut out = 0.0f64;
    for _ in 0..20 {
        g.step();
        let want_out = g
            .weapon_affine()
            .transform_point3(seat - DVec3::new(0.0, d3::gameplay::MAG_DROP_M, 0.0));
        let reloading = g
            .world
            .world()
            .get::<inf_ecs::weapon::WeaponState>(e)
            .unwrap()
            .reload_left_s
            > d3::gameplay::MAG_SEAT_S;
        if reloading {
            out = out.max(1.0 - (mag_at(&g) - want_out).length());
        }
    }
    println!(
        "the magazine: {:.3} mm off its seat before the reload; out on the reload: {}",
        d_seated * 1e3,
        out > 0.999
    );
    assert!(
        d_seated < 1e-3,
        "the magazine is not at the pack's Magazine_joint"
    );
    assert!(
        out > 0.999,
        "the magazine never left the well on the reload"
    );
}

/// **No weapon class draws a primitive any more** (wave VEH3f.2a): every row
/// of the shipped registry names a mesh identity, and every Modern Weapons
/// identity (and magazine) has a committed fallback of our own; the rows WPN2d
/// left drawing a primitive -- every shotgun and every launcher but the grenade
/// -- now name a pack body.
///
/// **Mutation**: the class table's `Launcher` arm back to `None` -> red
/// (`fim_92_stinger` draws a primitive). The `Shotgun` arm is VACUOUS to this
/// arm over the shipped registry -- every shotgun row is named by the pack
/// table first -- and is kept for a row the table does not know.
#[test]
fn no_weapon_class_draws_a_primitive() {
    let reg = registry();
    let committed = committed_guids("weapon-art");
    let mut rows = 0usize;
    let mut formerly = 0usize;
    for item in reg.0.values() {
        let Some(def) = item.weapon.as_ref() else {
            continue;
        };
        let mesh = inf_ecs::weapon::weapon_mesh_of(item);
        assert!(mesh.is_some(), "`{}` draws a primitive", item.id);
        let key = inf_ecs::weapon::weapon_mesh_key(&item.id, def).unwrap();
        if key.starts_with("SM_MW_") {
            assert!(
                committed.contains_key(&mesh.unwrap()),
                "{key} has no committed fallback"
            );
            if matches!(
                def.audio_class(),
                inf_ecs::weapon::WeaponClass::Shotgun | inf_ecs::weapon::WeaponClass::Launcher
            ) {
                formerly += 1;
            }
        }
        rows += 1;
    }
    for a in inf_ecs::weapon::MW_WEAPON_ART
        .iter()
        .filter(|a| a.4.is_some())
    {
        let g = inf_ecs::weapon::weapon_mesh_guid(&format!("{}_MAG", a.0));
        assert!(
            committed.contains_key(&g),
            "{}'s magazine has no fallback",
            a.0
        );
    }
    println!("{rows} weapon rows, every one a mesh; {formerly} shotgun/launcher rows that drew a primitive at WPN2d now draw a pack body");
    assert!(
        formerly >= 14,
        "only {formerly} formerly-primitive rows draw the pack"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (8) DETERMINISM, COST, SCHEMA
// ═════════════════════════════════════════════════════════════════════════════

/// **Two imports of the calibration pack are the same bytes** (wave VEH3f.2a;
/// the VEH3f "second import wrote 106 duplicates" lesson). LOCAL: the
/// calibration manifest imported into two fresh projects and every written
/// file compared byte for byte.
///
/// What this wave's doors write -- every file under `UE/Vehicles/`, every
/// pack material and texture `import_material` writes (their GUIDs are now a
/// pure function of the path, `import_path_guid`), every payload -- is
/// identical. The residue is the ASSET0 generic glTF door's: the SIDECARS of
/// the pack's loose static meshes (`UE/SM_*`) and of the placeholder
/// materials their glTFs embed, whose GUIDs that door still mints fresh, and
/// the import cache's own index. Counted and bounded here (their payloads are
/// identical), carried with a price in the report.
///
/// **Mutation**: `import_path_guid` back to `AssetId::new()` -> red (every
/// pack material and every texture sidecar differs).
#[test]
fn the_import_is_deterministic() {
    let Some(manifest) = local_manifests()
        .into_iter()
        .find(|m| m.to_string_lossy().contains("veh3f2a-dd"))
    else {
        eprintln!("SKIP the import-determinism arm: the calibration manifest is a local UE export");
        return;
    };
    let run = |dir: &Path| -> BTreeMap<String, u64> {
        let mut project = inf_editor_core::assets::AssetProject::open(dir).expect("a project");
        let opts = inf_editor_core::assets::ue_import::UeImportOptions {
            packs: vec!["DrivableCarsBasicVehicleS".into()],
            vehicles: true,
            max_texture: 1024,
            ..Default::default()
        };
        inf_editor_core::assets::ue_import::import_manifest(&mut project, &manifest, &opts)
            .expect("the import runs");
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let rel = p
                        .strip_prefix(dir)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.insert(rel, xxhash_rust::xxh3::xxh3_64(&std::fs::read(&p).unwrap()));
                }
            }
        }
        out
    };
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (x, y) = (run(a.path()), run(b.path()));
    let differ: Vec<&String> = x
        .iter()
        .filter(|(k, v)| y.get(*k) != Some(v))
        .map(|(k, _)| k)
        .collect();
    // The ASSET0 generic glTF door's residue: a sidecar of a loose static
    // mesh or of a material its glTF embeds, or the cache index.
    let residue = |f: &str| {
        f.starts_with(".inf/")
            || (f.ends_with(".toml")
                && !f.starts_with("UE/Vehicles/")
                && (f.starts_with("UE/SM_") || f.contains("_LOD0_Material")))
    };
    let ours: Vec<&&String> = differ.iter().filter(|f| !residue(f)).collect();
    let vehicles = x.keys().filter(|k| k.starts_with("UE/Vehicles/")).count();
    println!(
        "two imports: {} and {} files ({vehicles} under UE/Vehicles/); {} differ, all of them the generic glTF door's sidecars or its cache index; {} of this wave's",
        x.len(),
        y.len(),
        differ.len(),
        ours.len()
    );
    assert_eq!(
        x.len(),
        y.len(),
        "the two imports wrote different file sets"
    );
    assert!(vehicles > 300, "only {vehicles} files under UE/Vehicles/");
    assert!(
        ours.is_empty(),
        "this wave's files differ between two imports: {ours:?}"
    );
    assert!(
        differ.len() <= 40,
        "the generic door's residue grew to {}",
        differ.len()
    );
}

fn clock_is_asserted(what: &str) -> bool {
    if cfg!(debug_assertions) || std::env::var_os("CI").is_some() {
        eprintln!("{what}: measured and printed; asserted only in a release build off CI (the house conditioning)");
        return false;
    }
    true
}

/// **Sixty-four imported cars cost what they cost** (wave VEH3f.2a) -- the
/// player projector over 64 calibration cars against 64 box cars of the same
/// hull, with the FALLBACK's DAGs (the geometry CI has; the local art's draw
/// cost is the report's release number off the shipped host). Min of five;
/// asserted only in release off CI: the art cars' projection stays under
/// [`ART_PROJECTION_BUDGET_MS`].
#[test]
fn sixty_four_imported_cars_cost_what_they_cost() {
    let committed = committed_guids("vehicle-art");
    let def = roster_def(CALIBRATION);
    let mut plain = def;
    plain.art = None;
    let build = |d: &VehicleDef| -> (
        inf_player::runtime_sim::RuntimeSim,
        inf_player::vmesh::VmeshRegistry,
    ) {
        let mut world = EcsWorld::new();
        for k in 0..64u128 {
            let at = DVec3::new((k % 8) as f64 * 6.0, 0.0, (k / 8) as f64 * 8.0);
            car(&mut world, Uuid::from_u128(0x6400_0000 + k), at, 0.0, d);
        }
        world.propagate();
        let mut reg = inf_player::vmesh::VmeshRegistry::new();
        for (g, p) in &committed {
            let mesh: inf_mesh::MeshAsset = inf_asset::decode(&std::fs::read(p).unwrap()).unwrap();
            if mesh.triangle_count() > 0 && p.to_string_lossy().contains(CALIBRATION_ART) {
                reg.insert_mesh(inf_player::vmesh::derived_vmesh_id(*g), &dag(&mesh))
                    .unwrap();
            }
        }
        (
            inf_player::runtime_sim::RuntimeSim::new(
                world,
                Vec::new(),
                glam::DVec2::new(0.0, -9.81),
                60.0,
            ),
            reg,
        )
    };
    let time = |sim: &inf_player::runtime_sim::RuntimeSim,
                reg: &inf_player::vmesh::VmeshRegistry|
     -> (f64, usize) {
        let mut best = f64::MAX;
        let mut n = 0;
        for _ in 0..5 {
            let mut scene = inf_render::RenderScene::default();
            let t0 = std::time::Instant::now();
            inf_player::render::project_scene_with_skinned(
                &mut scene,
                sim,
                1.0,
                reg,
                &inf_player::skinned::SkinnedRegistry::new(),
                &inf_voxel::VoxelVolumes::default(),
            );
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            n = scene.vgeom_instances.len() + scene.instances.len();
        }
        (best, n)
    };
    let (sa, ra) = build(&def);
    let (sb, rb) = build(&plain);
    let (ta, na) = time(&sa, &ra);
    let (tb, nb) = time(&sb, &rb);
    println!(
        "64 art cars: {ta:.3} ms, {na} instances; 64 box cars: {tb:.3} ms, {nb} instances ({:.2}x)",
        ta / tb
    );
    assert!(na >= 64 * 10, "the art cars drew only {na} instances");
    if clock_is_asserted("64 imported cars") {
        assert!(
            ta < ART_PROJECTION_BUDGET_MS,
            "64 art cars project in {ta:.3} ms"
        );
    }
}

/// The projector's ceiling for 64 imported cars (release, min of five), ms.
const ART_PROJECTION_BUDGET_MS: f64 = 2.0;

/// **This wave moved no schema** (wave VEH3f.2a).
#[test]
fn this_wave_moved_no_schema() {
    assert_eq!(inf_scene::SCHEMA_VERSION, 28);
    assert_eq!(inf_runtime::pie::SCENE_PAYLOAD_VERSION, 13);
}

// ═════════════════════════════════════════════════════════════════════════════
// (6b) THE BODY-KIND CENSUS ON THE ISLAND
// ═════════════════════════════════════════════════════════════════════════════

const HZ: f64 = 60.0;

fn fixture_recipe() -> PathBuf {
    repo().join("samples/island-fixture/island.toml")
}

/// The CI island, cooked and booted as `run_headless` boots it (`veh3f_gate`'s
/// harness, verbatim).
fn island_sim(tmp: &Path) -> inf_player::runtime_sim::RuntimeSim {
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

/// TEN island minutes in a release build off CI (the brief's number), ONE in a
/// dev build or on a shared runner; `INF_VEH3F2A_ISLAND_MINUTES` overrides.
fn island_minutes() -> u64 {
    if let Some(m) = std::env::var("INF_VEH3F2A_ISLAND_MINUTES")
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

/// **The body-kind census, after the island's minutes** (wave VEH3f.2a).
///
/// The CI island cooked and booted as the player boots it, the crowd set
/// aside (`veh3f_gate`'s traffic-census reason), stepped for
/// [`island_minutes`]; every traffic record counted by class and by body kind:
/// IMPORTED (the row names an art body -- the art where the project has it,
/// its committed fallback where it does not, which is CI), SHELL (a DCC shell:
/// VEH3f.2b's, none yet, so the column is 0 by construction and printed so
/// VEH3f.2b's number has a place to land) and PRIMITIVE (the family boxes).
/// And off the WORLD, not the record: every RESIDENT chassis of an art row
/// draws the art's body GUID -- the thing the census claims.
///
/// **Target**: every traffic car of an imported class is imported (the
/// Construction class excepted: its machines are placed, not drawn by the kerb).
///
/// **Mutation**: the art filter in `catalogue_row_id` removed -> red (primitive
/// sedans at the kerb); `rig_nodes_at`'s art body GUID back to the family
/// primitive -> red on the resident draw.
#[test]
fn the_island_body_kind_census() {
    use inf_ecs::roster::RosterClass;
    let tmp = tempfile::tempdir().expect("a temp dir");
    let mut sim = island_sim(tmp.path());
    inf_ecs::crowd::set_population(sim.world_mut(), BTreeMap::new());
    let minutes = island_minutes();
    let art_classes: BTreeSet<RosterClass> = inf_ecs::roster::roster()
        .0
        .values()
        .filter(|d| d.art.is_some())
        .filter_map(|d| d.roster_class)
        .filter(|c| *c != RosterClass::Construction)
        .collect();
    // class -> (imported, shell, primitive), over every sample.
    let mut census: BTreeMap<RosterClass, [usize; 3]> = BTreeMap::new();
    let (mut resident_art, mut resident_drawn) = (0usize, 0usize);
    let steps = minutes * 60 * HZ as u64;
    for s in 0..steps {
        sim.step_once(Default::default());
        if s % 600 != 599 {
            continue;
        }
        let w = sim.world();
        let Some(t) = inf_ecs::traffic::traffic_of(w) else {
            continue;
        };
        for (g, r) in &t.records {
            let Some(c) = r.def.roster_class else {
                continue;
            };
            let e = census.entry(c).or_default();
            if r.def.art.is_some() {
                e[0] += 1;
                // Off the world: a resident chassis draws the art's body.
                if let Some(art) = r.def.art {
                    if w.entity_of(*g).is_some() {
                        resident_art += 1;
                        let body = inf_ecs::roster::art_body_guid(art);
                        if bevy_free::drawn(w, *g)
                            .iter()
                            .any(|(_, m)| m.asset == Some(body))
                        {
                            resident_drawn += 1;
                        }
                    }
                }
            } else {
                e[2] += 1;
            }
        }
    }
    println!("{minutes} island minute(s), sampled every 10 s; traffic records by class:");
    println!(
        "  {:<12} {:>9} {:>6} {:>10}",
        "class", "imported", "shell", "primitive"
    );
    let mut total = [0usize; 3];
    for (c, n) in &census {
        println!("  {:<12} {:>9} {:>6} {:>10}", c.name(), n[0], n[1], n[2]);
        for k in 0..3 {
            total[k] += n[k];
        }
        if art_classes.contains(c) {
            assert_eq!(n[2], 0, "{} drew {} primitive cars", c.name(), n[2]);
        }
    }
    println!(
        "  {:<12} {:>9} {:>6} {:>10}; resident art chassis {resident_art}, drawing the art's body {resident_drawn}",
        "total", total[0], total[1], total[2]
    );
    assert!(total[0] > 0, "no imported car was drawn at all");
    assert!(resident_art > 0, "no art chassis was ever resident");
    assert_eq!(
        resident_drawn, resident_art,
        "a resident art chassis draws something else"
    );
}
