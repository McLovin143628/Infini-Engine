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

/// **THE FOOT-IK GATE IS NOT DERIVABLE FROM A CLIP'S FOOT HEIGHT** — the
/// measurement that decided how the gate is taken, kept so it can go stale
/// loudly.
///
/// P29.4's foot IK is gated on ALS's `Enable_FootIK_L/R` curve. **No clip in
/// this engine has ever carried it** (this arm's sibling census: 164 imported
/// ALS clips, zero; and the 12 committed sample clips, zero), so the mechanism
/// never ran. The obvious repair is to derive the gate — "does this foot reach
/// the ground in this clip" ought to be an absolute height test against the
/// rig's own ground plane, since every rig this engine poses has its origin at
/// its feet.
///
/// It is not. UE authors an in-air cycle with the root on the ground and the
/// legs hanging, so a fall loop's ankle sits exactly where a walk's ankle sits.
/// This arm asserts the **negative**: over the whole imported library there is
/// no height that separates the grounded families from the airborne ones,
/// because the airborne clips are among the very highest. The engine therefore
/// takes the gate from the movement STATE, as ALS's own `UpdateFootIK` does.
///
/// It fails if a future import ever makes them separable — at which point this
/// reasoning is stale and `step_feet` can be reconsidered, which is the only
/// honest way to write down a negative result.
#[test]
fn the_foot_ik_gate_is_not_derivable_from_a_clips_foot_height() {
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

    let mut rows: Vec<(String, f32)> = Vec::new();
    let mut with_gate = 0usize;
    for (stem, asset) in &clips {
        if asset
            .clip
            .curves
            .iter()
            .any(|c| c.name.starts_with("Enable_FootIK_"))
        {
            with_gate += 1;
        }
        let Some(rig) = asset.skeleton.and_then(|id| rigs.get(&id)) else {
            continue;
        };
        let feet = inf_anim::derive::foot_joints(rig);
        if feet.is_empty() {
            continue;
        }
        // The clip's own lowest foot, sampled on the deriver's own 30 Hz grid.
        let steps = ((asset.clip.duration * 30.0).ceil() as usize).max(2);
        let mut low = f32::INFINITY;
        for k in 0..=steps {
            let t = asset.clip.duration * (k as f32) / (steps as f32);
            let pose = inf_anim::pose::sample_clip(&rig.skeleton, &asset.clip, t, false);
            let g = inf_anim::pose::global_transforms(&rig.skeleton, &pose);
            for &j in feet.iter().take(2) {
                let y = g[j as usize].to_scale_rotation_translation().2.y;
                if y < low {
                    low = y;
                }
            }
        }
        rows.push((stem.clone(), low));
    }
    assert!(rows.len() > 100, "only {} clips measured", rows.len());

    // **THE CENSUS**: nothing authors the gate.
    println!(
        "\n=== Enable_FootIK_* over {} imported clips: {with_gate} carry it ===",
        clips.len()
    );
    assert_eq!(
        with_gate, 0,
        "{with_gate} imported clips now carry an `Enable_FootIK_*` channel — the \
         reasoning in `step_feet` assumes none does, and it is stale"
    );

    // **THE FALSIFICATION**: the airborne clips are not below the grounded ones.
    let at =
        |needle: &str| -> Option<f32> { rows.iter().find(|r| r.0.ends_with(needle)).map(|r| r.1) };
    let grounded: Vec<(&str, f32)> = ["Walk_F", "Run_F", "Sprint_F", "TurnIP_L90", "Land_Light"]
        .into_iter()
        .filter_map(|n| at(n).map(|v| (n, v)))
        .collect();
    let airborne: Vec<(&str, f32)> = ["JumpLoop", "FallLoop", "Mantle_2m", "Mantle_1m_RH"]
        .into_iter()
        .filter_map(|n| at(n).map(|v| (n, v)))
        .collect();
    assert!(
        grounded.len() >= 4 && airborne.len() >= 3,
        "the families are not both present: {grounded:?} / {airborne:?}"
    );
    let highest_grounded = grounded.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
    let lowest_airborne = airborne
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::INFINITY, f32::min);
    println!("  grounded families (lowest foot, m): {grounded:?}");
    println!("  airborne families (lowest foot, m): {airborne:?}");
    let mut sorted: Vec<&(String, f32)> = rows.iter().collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    println!(
        "  whole library: {:.4} m … {:.4} m over {} clips",
        sorted.first().map(|r| r.1).unwrap_or(f32::NAN),
        sorted.last().map(|r| r.1).unwrap_or(f32::NAN),
        rows.len()
    );
    for r in sorted.iter().rev().take(8) {
        println!("    {:>8.4}  {}", r.1, r.0);
    }
    assert!(
        lowest_airborne <= highest_grounded,
        "an airborne clip's lowest foot ({lowest_airborne:.4} m) is now above \
         every grounded one's ({highest_grounded:.4} m) — the families ARE \
         separable by height after all, and `step_feet`'s reasoning is stale"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) FOOT IK ON TOPOGRAPHY — the headless fixture
// ─────────────────────────────────────────────────────────────────────────────

use glam::DVec3;
use inf_anim::{AnimClip, Joint, JointTransform, Skeleton, SkeletonAsset, SmState, StateMachine};
use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, RigidBody3D,
    SkeletalMesh, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::EcsWorld;
use inf_physics::d3::step_character_movement;
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const HERO: Uuid = Uuid::from_u128(0x0b1b_0001);
const SKEL: Uuid = Uuid::from_u128(0x0b1b_0002);
const SM: Uuid = Uuid::from_u128(0x0b1b_0003);
const CLIP: inf_anim::ClipRef = [0xb1; 16];
const RADIUS: f64 = 0.3;

/// The hip height that puts this rig's ankles at **model `y = 0`** — the leg
/// chain's own length (0.05 + 0.45 + 0.45).
///
/// It is the whole reason this fixture can measure a sole at all: with the ankle
/// on the model's own ground plane, the drawn foot joint's world `y` **is** the
/// sole's world `y`, and penetration and hover are one subtraction against the
/// ground under it. A rig whose ankle sat 9 cm up — which every real rig's does
/// — would need that 9 cm measured and subtracted, and the measurement would
/// then be of the subtraction.
const HIPS_FEET_AT_ORIGIN: f32 = 0.95;

/// A two-leg rig — hips, then thigh → shin → foot per side, ankles at model
/// `y = 0`, with the foot joints named so `inf_anim`'s matcher finds them.
///
/// Deliberately the same shape as `inf-physics`' `foot_slide_gate::legs`, which
/// is the fixture the **lock** half of this mechanism is measured on: two files
/// that disagreed about what a leg is would be measuring two different engines.
fn legs() -> SkeletonAsset {
    fn joint(name: &str, parent: Option<u16>, local: glam::Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::from_trs(local, glam::Quat::IDENTITY, glam::Vec3::ONE),
        }
    }
    SkeletonAsset::new(
        Skeleton::new(vec![
            joint("Hips", None, glam::Vec3::new(0.0, 1.0, 0.0)),
            joint("Thigh.L", Some(0), glam::Vec3::new(0.1, -0.05, 0.0)),
            joint("Shin.L", Some(1), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Foot.L", Some(2), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Thigh.R", Some(0), glam::Vec3::new(-0.1, -0.05, 0.0)),
            joint("Shin.R", Some(4), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Foot.R", Some(5), glam::Vec3::new(0.0, -0.45, 0.0)),
        ])
        .expect("a valid pair of legs"),
    )
}

/// A one-second stance clip holding the hips at [`HIPS_FEET_AT_ORIGIN`], with
/// **no** curve channels at all.
///
/// No channels on purpose: this fixture measures the IK, and after wave
/// CHAR1b.1 a clip that authors nothing is a clip whose feet the engine puts on
/// the ground (`inf_physics::d3::movement::step_feet`'s docs). Authoring
/// `Enable_FootIK_*` here would measure the fixture's opinion instead of the
/// engine's default.
///
/// Two identical keys a second apart give the clip a timeline without giving it
/// a pose: `AnimClip::new` derives the duration from the joint keys, and a clip
/// with none has a play-head that never leaves zero.
fn stance_clip() -> AnimClip {
    let track = inf_anim::JointTrack {
        joint: 0,
        translation: Some(inf_anim::Vec3Track::new(
            vec![0.0, 1.0],
            vec![
                [0.0, HIPS_FEET_AT_ORIGIN, 0.0],
                [0.0, HIPS_FEET_AT_ORIGIN, 0.0],
            ],
            inf_anim::Interpolation::Linear,
        )),
        rotation: None,
        scale: None,
    };
    AnimClip::new("stance", vec![track])
}

/// One block of world: a centre, half-extents, and a euler-degree rotation
/// (`Transform::rotation` is the units doctrine's one UI-boundary exception, and
/// it is what `inf-physics`' own ramp fixture uses).
#[derive(Clone, Copy, Debug)]
struct Block {
    centre: DVec3,
    half: DVec3,
    pitch_deg: f64,
    roll_deg: f64,
}

impl Block {
    fn flat(centre: DVec3, half: DVec3) -> Self {
        Self {
            centre,
            half,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        }
    }
}

/// The ground this fixture can stand a character on.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ground {
    /// A level floor.
    Flat,
    /// A ramp rolled about **Z** by `deg`: the two feet end up at different
    /// heights AND rolled, which is the case that exercises the pelvis drop and
    /// the roll clamp together.
    Slope { deg: f64 },
    /// A kerb `rise` metres high whose edge runs along `z` at `x = 0`, so a
    /// character standing at `x = 0` has its LEFT foot on the kerb and its RIGHT
    /// foot on the road. The step case, and the one CERT1's 120 collider-less
    /// settlement spans are about.
    Kerb { rise: f64 },
    /// A flight of `n` steps of `rise` metres, the character standing with one
    /// foot on a tread and one on the next — a building entrance, in miniature.
    Stairs { rise: f64 },
}

impl Ground {
    fn label(self) -> String {
        match self {
            Ground::Flat => "flat".into(),
            Ground::Slope { deg } => format!("slope {deg:.0}°"),
            Ground::Kerb { rise } => format!("kerb {:.0} cm", rise * 100.0),
            Ground::Stairs { rise } => format!("stairs {:.0} cm", rise * 100.0),
        }
    }

    /// The blocks that make it, and where the character stands.
    fn build(self) -> (Vec<Block>, DVec3) {
        let floor = Block::flat(DVec3::new(0.0, -0.5, 0.0), DVec3::new(30.0, 0.5, 30.0));
        match self {
            Ground::Flat => (vec![floor], DVec3::ZERO),
            Ground::Slope { deg } => (
                vec![
                    floor,
                    Block {
                        centre: DVec3::new(0.0, 1.0, 0.0),
                        half: DVec3::new(8.0, 0.5, 8.0),
                        pitch_deg: 0.0,
                        roll_deg: deg,
                    },
                ],
                // Standing over the ramp's own centre, where its top face passes
                // through `y = 1.0 + 0.5 / cos(deg)` — but the exact height does
                // not matter, because the character is dropped onto it.
                DVec3::new(0.0, 2.5, 0.0),
            ),
            Ground::Kerb { rise } => (
                vec![
                    floor,
                    // The kerb occupies x ∈ [0, 20]; its edge is the plane x = 0.
                    Block::flat(
                        DVec3::new(10.0, rise * 0.5, 0.0),
                        DVec3::new(10.0, rise * 0.5, 20.0),
                    ),
                ],
                DVec3::new(0.0, rise + 1.0, 0.0),
            ),
            Ground::Stairs { rise } => {
                let mut blocks = vec![floor];
                // Five treads climbing in +x, each 0.3 m deep — so the two feet,
                // 0.2 m apart, land on two different treads.
                for i in 1..=5 {
                    let top = rise * i as f64;
                    blocks.push(Block::flat(
                        DVec3::new(0.15 + 0.3 * (i - 1) as f64, top * 0.5, 0.0),
                        DVec3::new(0.15, top * 0.5, 20.0),
                    ));
                }
                // Straddling the edge between tread 1 and tread 2.
                (blocks, DVec3::new(0.3, rise * 2.0 + 1.0, 0.0))
            }
        }
    }
}

/// What one foot is doing where it stands.
#[derive(Clone, Copy, Debug)]
struct Sole {
    /// The **drawn** sole's world height, metres — the foot joint's world `y`
    /// read out of the pose store, which on this rig is the sole itself.
    y: f64,
    /// The surface directly under it, from a downward ray, metres.
    ground: f64,
    /// The angle between the **drawn foot's own up axis** and world up,
    /// degrees — how far out of level the sole is lying.
    ///
    /// Measured off the foot's global rotation rather than off the local
    /// quaternion `apply_foot_ik` writes, because the leg solve rotates the foot
    /// too and the question is what the sole is DOING, not which pass moved it.
    tilt_deg: f64,
    /// The signed lateral half of that tilt (positive = the foot's up leans
    /// toward `+x`), degrees.
    roll_deg: f64,
    /// The signed fore/aft half (positive = leaning toward `-z`), degrees.
    pitch_deg: f64,
}

impl Sole {
    /// Millimetres of sole **below** the surface; zero when it is above it.
    fn penetration_mm(self) -> f64 {
        ((self.ground - self.y) * 1000.0).max(0.0)
    }
    /// Millimetres of sole **above** the surface; zero when it is below it.
    fn hover_mm(self) -> f64 {
        ((self.y - self.ground) * 1000.0).max(0.0)
    }
}

/// The fixture: a world, a character, and the three assets the pose step needs.
struct Topo {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    skeleton: SkeletonAsset,
    machine: StateMachine,
    clip: AnimClip,
}

impl Topo {
    fn new(ground: Ground) -> Self {
        let (blocks, at) = ground.build();
        let mut world = EcsWorld::new();
        for (i, b) in blocks.iter().enumerate() {
            let e = world.spawn_with_guid(Uuid::from_u128(0x0b1b_1000 + i as u128), "Block", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::from_dvec3(b.centre);
            t.rotation = Vec3d::new(b.pitch_deg, 0.0, b.roll_deg);
            world.world_mut().entity_mut(e).insert((
                RigidBody3D {
                    kind: BodyKind3D::Static,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::from_dvec3(b.half),
                    ..Default::default()
                },
                t,
            ));
        }
        let cm = CharacterMovement {
            player_controlled: true,
            ..Default::default()
        };
        let e = world.spawn_with_guid(HERO, "Hero", None);
        let mut t = Transform::IDENTITY;
        t.translation =
            Vec3d::from_dvec3(at + DVec3::new(0.0, cm.stand_half_height_m + RADIUS, 0.0));
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
            inf_ecs::components::CharacterController3D::default(),
            cm,
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(1)),
                skeleton: Some(SKEL),
            },
            t,
        ));
        world.mark_dirty();
        world.propagate();
        Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
            skeleton: legs(),
            machine: StateMachine {
                states: vec![SmState::clip("stance", CLIP)],
                entry: 0,
                ..Default::default()
            },
            clip: stance_clip(),
        }
    }

    /// One fixed step in the **shipped order**: physics sync, intent, character
    /// movement, propagate, pose.
    fn step(&mut self, intent: &MovementIntent) {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        step_character_movement(&mut self.world, &mut self.bridge, DT);
        self.world.propagate();
        let (machine, skeleton, clip) = (&self.machine, &self.skeleton, &self.clip);
        let machines = |g: Uuid| (g == SM).then_some(machine);
        let skels = |g: Uuid| (g == SKEL).then_some(skeleton);
        let clips = |c: inf_anim::ClipRef| (c == CLIP).then_some(clip);
        let vars = |_: Uuid| std::collections::BTreeMap::new();
        inf_ecs::pose::step_pose_evaluation(&mut self.world, DT, &machines, &skels, &clips, &vars);
    }

    /// Settle: stand still until the mover has stopped moving the capsule.
    fn settle(&mut self, steps: usize) {
        let idle = MovementIntent::default();
        for _ in 0..steps {
            self.step(&idle);
        }
    }

    /// The two soles, **drawn**, with the ground under each.
    fn soles(&mut self) -> [Sole; 2] {
        let to_world =
            inf_ecs::pose::model_to_world_of(&self.world, HERO).expect("the hero has a transform");
        let store = self
            .world
            .world()
            .get_resource::<inf_ecs::pose::PoseStoreRes>()
            .expect("the pose step ran");
        let posed = store.0.get(&HERO).expect("the hero was posed").clone();
        let sk = &self.skeleton.skeleton;
        let globals = inf_anim::pose::global_transforms(sk, &posed.pose);
        // The character's OWN capsule, out of the way. `CastTargets::Fixed`
        // keeps static AND kinematic bodies (a ledge that moves is still a
        // ledge), and this capsule is kinematic and spans the whole body — so a
        // ray starting half a metre over the sole starts INSIDE it and an
        // unfiltered cast answers its own origin. The first spelling of these
        // lines reported the ground half a metre above the floor, which read as
        // "the sole is 500 mm inside the surface".
        let exclude: std::collections::BTreeSet<_> =
            self.bridge.collider_of(HERO).into_iter().collect();
        let feet = inf_anim::derive::foot_joints(&self.skeleton);
        assert_eq!(feet.len(), 2, "the fixture rig has two feet");
        let mut out = [Sole {
            y: 0.0,
            ground: 0.0,
            tilt_deg: 0.0,
            roll_deg: 0.0,
            pitch_deg: 0.0,
        }; 2];
        for (side, &j) in feet.iter().enumerate() {
            let (_, rot, tr) = globals[j as usize].to_scale_rotation_translation();
            let w = to_world.transform_point3(DVec3::new(tr.x as f64, tr.y as f64, tr.z as f64));
            // The ground under THIS foot, **static colliders only**. A ray from
            // half a metre over the sole starts INSIDE the character's own
            // kinematic capsule (which spans the whole body), and an unfiltered
            // cast answers its own origin — the first spelling of this line
            // reported the ground half a metre above the floor, which is what
            // "the sole is 500 mm inside the surface" meant.
            let hit = self.bridge.world_mut().cast_ray_where(
                w + DVec3::new(0.0, 0.5, 0.0),
                -DVec3::Y,
                2.0,
                &exclude,
                inf_physics::d3::CastTargets::Fixed,
            );
            let ground = hit.map(|h| h.point.y).unwrap_or(f64::NAN);
            // How the sole lies, from the foot's own up axis.
            let up = rot * glam::Vec3::Y;
            let (ux, uy, uz) = (up.x as f64, up.y as f64, up.z as f64);
            out[side] = Sole {
                y: w.y,
                ground,
                tilt_deg: inf_math::pacos64(uy.clamp(-1.0, 1.0)).to_degrees(),
                roll_deg: inf_math::patan2_64(ux, uy).to_degrees(),
                pitch_deg: inf_math::patan2_64(-uz, uy).to_degrees(),
            };
        }
        out
    }
}

/// Stand a character on `ground`, settle it, and report both soles.
fn stand_on(ground: Ground) -> [Sole; 2] {
    let mut t = Topo::new(ground);
    t.settle(180);
    t.soles()
}

/// **THE SOLE RESTS ON THE SURFACE — flat, slope, kerb, stairs** (clause 1).
///
/// The mandate: *"make sure that the characters in the game stand on the surface
/// with perfect topography and foot IK. It looks like currently that the feet of
/// characters sinks in through the surface a little bit."* CHAR1a.2 measured the
/// flat-ground half on the committed body; this is the other half, on the four
/// surfaces the island is made of.
///
/// The bound is the mandate's own **1 cm**, both ways: a sole may be at most
/// 10 mm below the surface and at most 10 mm above it. Reported per surface per
/// foot so a reader can see which one is tight.
#[test]
fn a_sole_rests_on_every_surface_within_ten_millimetres() {
    const BOUND_MM: f64 = 10.0;
    let cases = [
        Ground::Flat,
        Ground::Slope { deg: 15.0 },
        Ground::Kerb { rise: 0.15 },
        Ground::Stairs { rise: 0.18 },
    ];
    println!("\n=== the sole on four surfaces (bound ±{BOUND_MM:.0} mm) ===");
    let mut worst = 0.0f64;
    for ground in cases {
        let soles = stand_on(ground);
        for (side, s) in soles.iter().enumerate() {
            let name = if side == 0 { "L" } else { "R" };
            assert!(
                s.ground.is_finite(),
                "{}: no ground under the {name} foot at y={:.4}",
                ground.label(),
                s.y
            );
            println!(
                "  {:<14} {name}  sole {:>8.4} m  ground {:>8.4} m  \
                 penetration {:>6.2} mm  hover {:>6.2} mm  tilt {:>6.2}°                   roll {:>6.2}°  pitch {:>6.2}°",
                ground.label(),
                s.y,
                s.ground,
                s.penetration_mm(),
                s.hover_mm(),
                s.tilt_deg,
                s.roll_deg,
                s.pitch_deg
            );
            worst = worst.max(s.penetration_mm()).max(s.hover_mm());
            assert!(
                s.penetration_mm() <= BOUND_MM,
                "{} {name}: the sole is {:.2} mm INSIDE the surface",
                ground.label(),
                s.penetration_mm()
            );
            assert!(
                s.hover_mm() <= BOUND_MM,
                "{} {name}: the sole HOVERS {:.2} mm over the surface",
                ground.label(),
                s.hover_mm()
            );
        }
    }
    println!("  worst of the eight: {worst:.2} mm");
}

/// **THE FOOT ROTATES ONTO THE SLOPE, AND THE ROTATION IS CLAMPED** (clause 1).
///
/// `inf_anim::ground_offset` has answered a pitch and a roll since P29.4 and the
/// movement step dropped both, so a foot on a ramp was *translated* onto the
/// slope and left level. The falsification is the pair: level ground must leave
/// the foot level, and a 15° ramp must roll it by about 15°.
#[test]
fn a_foot_lies_flat_on_a_slope_and_stays_level_on_a_floor() {
    let flat = stand_on(Ground::Flat);
    for (side, s) in flat.iter().enumerate() {
        assert!(
            s.tilt_deg < 2.0,
            "foot {side} is tilted {:.2}° on a LEVEL floor: {s:?}",
            s.tilt_deg
        );
    }
    let slope = stand_on(Ground::Slope { deg: 15.0 });
    for (side, s) in slope.iter().enumerate() {
        assert!(
            (s.roll_deg.abs() - 15.0).abs() < 3.0,
            "foot {side} did not lie on the 15° ramp: rolled {:.2}° (tilt {:.2}°)",
            s.roll_deg,
            s.tilt_deg
        );
    }
    println!(
        "\n=== the foot on a 15° ramp: roll L {:.2}° R {:.2}° (flat tilt {:.2}° / {:.2}°) ===",
        slope[0].roll_deg, slope[1].roll_deg, flat[0].tilt_deg, flat[1].tilt_deg
    );

    // **THE CLAMP.** A ramp steeper than the ankle bends must not bend it
    // further than `FOOT_ROLL_LIMIT_DEG` further, whatever the ground says.
    //
    // The number measured here is the foot's **global** up axis, so it carries
    // the leg's own rotation as well as the ankle's — a leg reaching down a 40°
    // slope is itself tilted, and no measurement taken in world space can
    // subtract that. So the claim is the one that survives it: the sole does not
    // FOLLOW the surface (it would read ~40° if the clamp were removed), and the
    // extra it takes over the 15° case is inside the ankle's own budget.
    let steep = stand_on(Ground::Slope { deg: 40.0 });
    let limit = inf_ecs::pose::FOOT_ROLL_LIMIT_DEG;
    for (side, s) in steep.iter().enumerate() {
        assert!(
            s.roll_deg.abs() < 30.0,
            "foot {side} rolled {:.2}° on a 40° ramp — it is following the \
             surface, so the {limit}° ankle clamp is not being applied",
            s.roll_deg
        );
        assert!(
            s.roll_deg.abs() - slope[side].roll_deg.abs() <= limit,
            "foot {side} took {:.2}° more than the 15° ramp did, past the \
             {limit}° the ankle is allowed",
            s.roll_deg.abs() - slope[side].roll_deg.abs()
        );
    }
    println!(
        "  a 40° ramp reads {:.2}° / {:.2}° against the 15° ramp's {:.2}° / {:.2}° \
         (ankle limit {limit}°)",
        steep[0].roll_deg, steep[1].roll_deg, slope[0].roll_deg, slope[1].roll_deg
    );
}

/// **THE PELVIS DROPS TO THE LOWER FOOT** (clause 1, ALS's `SetPelvisIKOffset`).
///
/// On a kerb and on a stair the two feet are on different surfaces, and the
/// whole body has to come down to the low one rather than the low leg
/// straightening past its limit. Flat ground is the control: nothing drops.
#[test]
fn the_pelvis_drops_to_the_lower_foot_on_a_step() {
    fn drop_of(ground: Ground) -> f64 {
        let mut t = Topo::new(ground);
        t.settle(180);
        let e = t.world.entity_of(HERO).unwrap();
        t.world
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .runtime
            .pelvis_offset
            .y
    }
    let flat = drop_of(Ground::Flat);
    let kerb = drop_of(Ground::Kerb { rise: 0.15 });
    let stairs = drop_of(Ground::Stairs { rise: 0.18 });
    println!("\n=== the pelvis drop: flat {flat:.4} m, kerb {kerb:.4} m, stairs {stairs:.4} m ===");
    assert!(
        kerb < flat - 0.05,
        "a 15 cm kerb did not drop the pelvis: {kerb:.4} m against {flat:.4} m on the flat"
    );
    assert!(
        stairs < flat - 0.05,
        "an 18 cm stair did not drop the pelvis: {stairs:.4} m against {flat:.4} m"
    );
    assert!(
        kerb >= -0.30 && stairs >= -0.30,
        "the drop is unbounded: kerb {kerb:.4} m stairs {stairs:.4} m"
    );
}
