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

/// A biped: hips, a leg per side (thigh → shin → foot, ankles at model `y = 0`),
/// and a torso (three spine segments, a neck, a head) **with a role table**.
///
/// The legs are deliberately the same shape as `inf-physics`'
/// `foot_slide_gate::legs`, which is the fixture the **lock** half of the foot
/// mechanism is measured on: two files that disagreed about what a leg is would
/// be measuring two different engines. The torso is this wave's, because a rig
/// with no `Head` row cannot be looked at and a rig with no `Spine` row has no
/// upper body for a mask to cover — both passes answer "nothing to do" on the
/// leg-only rig, and an arm run against one would be measuring the absence.
fn legs() -> SkeletonAsset {
    fn joint(name: &str, parent: Option<u16>, local: glam::Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::from_trs(local, glam::Quat::IDENTITY, glam::Vec3::ONE),
        }
    }
    let mut asset = SkeletonAsset::new(
        Skeleton::new(vec![
            joint("Hips", None, glam::Vec3::new(0.0, 1.0, 0.0)),
            joint("Thigh.L", Some(0), glam::Vec3::new(0.1, -0.05, 0.0)),
            joint("Shin.L", Some(1), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Foot.L", Some(2), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Thigh.R", Some(0), glam::Vec3::new(-0.1, -0.05, 0.0)),
            joint("Shin.R", Some(4), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("Foot.R", Some(5), glam::Vec3::new(0.0, -0.45, 0.0)),
            joint("spine_01", Some(0), glam::Vec3::new(0.0, 0.12, 0.0)),
            joint("spine_02", Some(7), glam::Vec3::new(0.0, 0.12, 0.0)),
            joint("spine_03", Some(8), glam::Vec3::new(0.0, 0.12, 0.0)),
            joint("neck_01", Some(9), glam::Vec3::new(0.0, 0.12, 0.0)),
            joint("head", Some(10), glam::Vec3::new(0.0, 0.10, 0.0)),
        ])
        .expect("a valid biped"),
    );
    use inf_anim::roles::{BoneRole, BoneRoleKind as K, BoneSide as S};
    asset.roles = vec![
        BoneRole::new(0, K::Pelvis, S::Center),
        BoneRole::new(1, K::Thigh, S::Left),
        BoneRole::new(2, K::Calf, S::Left),
        BoneRole::new(3, K::Foot, S::Left),
        BoneRole::new(4, K::Thigh, S::Right),
        BoneRole::new(5, K::Calf, S::Right),
        BoneRole::new(6, K::Foot, S::Right),
        BoneRole::new(7, K::Spine, S::Center),
        BoneRole::new(8, K::Spine, S::Center),
        BoneRole::new(9, K::Spine, S::Center),
        BoneRole::new(10, K::Neck, S::Center),
        BoneRole::new(11, K::Head, S::Center),
    ];
    asset
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
    /// Extra clips the machine names beyond [`CLIP`] — the aim sweep's three
    /// samples, when an arm has put a look-sweep state on the machine.
    sweeps: Vec<(inf_anim::ClipRef, AnimClip)>,
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
            sweeps: Vec::new(),
        }
    }

    /// Point the hero's aim at `(yaw, pitch)` degrees, world frame.
    ///
    /// Written straight onto the runtime, which is where `apply_intent` puts it
    /// for a player-controlled character: this fixture has no input map and the
    /// question is what the POSE does with an aim, not how a mouse becomes one.
    fn set_aim(&mut self, yaw_deg: f64, pitch_deg: f64) {
        let e = self.world.entity_of(HERO).expect("the hero");
        if let Some(mut cm) = self.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.aim_yaw_deg = yaw_deg;
            cm.runtime.aim_pitch_deg = pitch_deg;
            cm.runtime.aim_sweep = inf_ecs::movement::aim_sweep(pitch_deg);
        }
    }

    /// The bytes the fixed step publishes for this world's poses.
    fn pose_bytes(&self) -> Vec<u8> {
        inf_ecs::pose::pose_state_bytes(&self.world)
    }

    /// One joint's **local** rotation in the drawn pose — the reading that can
    /// tell one pass from another, which whole-pose bytes cannot.
    fn joint_rotation(&self, joint: usize) -> Option<[f32; 4]> {
        let store = self
            .world
            .world()
            .get_resource::<inf_ecs::pose::PoseStoreRes>()?;
        store
            .0
            .get(&HERO)?
            .pose
            .locals
            .get(joint)
            .map(|l| l.rotation)
    }

    /// One fixed step in the **shipped order**: physics sync, intent, character
    /// movement, propagate, pose.
    fn step(&mut self, intent: &MovementIntent) {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        step_character_movement(&mut self.world, &mut self.bridge, DT);
        self.world.propagate();
        let (machine, skeleton, clip) = (&self.machine, &self.skeleton, &self.clip);
        // `set_aim` writes the aim onto the runtime and the movement step above
        // has already run, so the pose below reads this step's aim — which is
        // what a player pressing the mouse gets, one step later.

        let machines = |g: Uuid| (g == SM).then_some(machine);
        let skels = |g: Uuid| (g == SKEL).then_some(skeleton);
        let sweeps = &self.sweeps;
        let clips = |c: inf_anim::ClipRef| {
            if c == CLIP {
                Some(clip)
            } else {
                sweeps.iter().find(|(id, _)| *id == c).map(|(_, a)| a)
            }
        };
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
                 penetration {:>6.2} mm  hover {:>6.2} mm  tilt {:>6.2}° roll {:>6.2}° pitch {:>6.2}°",
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

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE CLIP → MODE MAP
// ─────────────────────────────────────────────────────────────────────────────

/// **THE TWO MODE TABLES AGREE.**
///
/// `inf_anim::als::LocoMode` mirrors `inf_ecs::components::MovementMode`'s
/// discriminants because the ring order forbids `inf-anim` from naming the enum
/// — and a table spelled in two crates is the shape three ledgers in this
/// repository have a law about. This is the pin: the numbers a built graph
/// compares `mode` against are the numbers the movement step publishes.
#[test]
fn the_two_mode_tables_agree() {
    use inf_anim::als::LocoMode;
    use inf_ecs::components::MovementMode as M;
    let pairs = [
        (LocoMode::Grounded, M::Grounded),
        (LocoMode::Crouch, M::Crouch),
        (LocoMode::Roll, M::Roll),
        (LocoMode::FallFree, M::FallFree),
        (LocoMode::FallControlled, M::FallControlled),
        (LocoMode::Ragdoll, M::Ragdoll),
    ];
    for (a, b) in pairs {
        assert_eq!(
            a.param(),
            b as u8 as f64,
            "{a:?} and {b:?} disagree — a graph built on the first would compare \
             `mode` against a number the movement step never publishes"
        );
    }
    // …and the parameter NAMES are one word each, not two.
    assert_eq!(
        inf_anim::als::SPEED_VAR,
        inf_ecs::anim_bridge::params::SPEED
    );
    assert_eq!(inf_anim::als::GAIT_VAR, inf_ecs::anim_bridge::params::GAIT);
    assert_eq!(inf_anim::als::MODE_VAR, inf_ecs::anim_bridge::params::MODE);
    assert_eq!(
        inf_anim::als::GROUNDED_VAR,
        inf_ecs::anim_bridge::params::GROUNDED
    );
    assert_eq!(
        inf_anim::als::FALL_SPEED_VAR,
        inf_ecs::anim_bridge::params::FALL_SPEED
    );
    assert_eq!(
        inf_anim::als::LAND_ALPHA_VAR,
        inf_ecs::anim_bridge::params::LAND_ALPHA
    );
    assert_eq!(
        inf_anim::als::MOVE_X_VAR,
        inf_ecs::anim_bridge::params::MOVE_X
    );
    assert_eq!(
        inf_anim::als::MOVE_Y_VAR,
        inf_ecs::anim_bridge::params::MOVE_Y
    );
}

/// **THE GATE TABLE: every mode ALS ships a clip for plays a REAL clip.**
///
/// The mandate: *"Every mode in P29's 14-mode catalogue plays a REAL clip on the
/// island in PIE; the gate lists each clip → mode binding and fails on an
/// unbound mode."* This is the list, resolved against the clips actually on
/// disk, with two failure modes: a slot the map names and the import does not
/// carry, and a slot bound to a **shell** — a clip with fewer than ten animated
/// joints, which plays as a bind pose wearing an animation's name (CHAR1a.3's
/// carried item 93, one asset kind over).
///
/// Local-only: the ALS sequences are licensed content and CI has none, so this
/// SKIPS with a printed reason there. The committed fixture CI *does* run is
/// `inf_anim::als`'s own unit arms, which prove the builder over a synthetic
/// resolver.
#[test]
fn every_mode_the_donor_ships_a_clip_for_binds_a_real_one() {
    const SHELL_JOINTS: usize = 10;
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
    // The same suffix rule the import door resolves with: the file stem is
    // `{pack}_{name}` and the map holds the donor's own name.
    let find = |name: &str| -> Option<&inf_anim::AnimClipAsset> {
        let tail = format!("_{name}");
        let mut hit = None;
        for (stem, asset) in &clips {
            if stem == name || stem.ends_with(&tail) {
                if hit.is_some() {
                    return None;
                }
                hit = Some(asset);
            }
        }
        hit
    };
    let (machine, report) = inf_anim::build_locomotion_graph(&|name: &str| {
        find(name).map(|_| {
            // A distinct id per name — this arm is about the TABLE, and the door
            // that mints real GUIDs is `ue_import::rebind_locomotion_graph`.
            let mut id = [0u8; 16];
            for (i, b) in name.as_bytes().iter().enumerate() {
                id[i % 16] ^= *b;
            }
            id
        })
    });

    println!("\n=== the clip → mode map, against the island's own clips ===");
    let mut shells: Vec<(String, String, usize)> = Vec::new();
    for slot in inf_anim::LOCOMOTION_MAP {
        let mut cells: Vec<String> = Vec::new();
        for (name, _) in slot.clips {
            match find(name) {
                Some(a) => {
                    let joints = a
                        .clip
                        .tracks
                        .iter()
                        .filter(|t| {
                            t.translation.is_some() || t.rotation.is_some() || t.scale.is_some()
                        })
                        .count();
                    if joints < SHELL_JOINTS {
                        shells.push((slot.state.into(), (*name).into(), joints));
                    }
                    cells.push(format!("{name} ({joints}j)"));
                }
                None => cells.push(format!("{name} MISSING")),
            }
        }
        println!(
            "  {:<18} {:?}  {:?}  {}",
            slot.state,
            slot.mode,
            slot.kind,
            cells.join(", ")
        );
    }
    println!(
        "  {} states, {} transitions, {}",
        machine.states.len(),
        machine.transitions.len(),
        report.summary()
    );

    assert!(
        report.unbound.is_empty(),
        "the map names {} sequences this import does not carry: {:?}",
        report.unbound.len(),
        report.unbound
    );
    assert!(
        report.missing_states.is_empty(),
        "unfilled states: {:?}",
        report.missing_states
    );
    assert!(
        shells.is_empty(),
        "{} bindings are SHELLS (fewer than {SHELL_JOINTS} animated joints — a \
         bind pose wearing an animation's name): {shells:?}",
        shells.len()
    );
    machine
        .validate()
        .expect("the graph built from real clips validates");
    // Not vacuous: the map covers the catalogue's grounded, crouched, airborne,
    // rolling and ragdolling families, and more than one clip each.
    let modes: std::collections::BTreeSet<_> =
        inf_anim::LOCOMOTION_MAP.iter().map(|s| s.mode).collect();
    assert!(modes.len() >= 6, "{modes:?}");
    assert!(
        machine.states.len() >= 20,
        "{} states",
        machine.states.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE ADDITIVE LAYER
// ─────────────────────────────────────────────────────────────────────────────

/// **AN ADDITIVE WITH NO DELTA IS BIT-IDENTICAL TO THE BASE POSE** (clause 3).
///
/// The identity an additive layer is defined by, asserted on the arithmetic
/// itself rather than on the runtime's early-out: `inf_ecs::pose`'s aim-offset
/// pass returns before it samples anything when the sweep sits at its neutral,
/// so an arm that only drove the runtime would be measuring the `return false`
/// and not the maths.
///
/// The falsification beside it is what makes the identity worth having: a REAL
/// delta moves the masked joint and **only** the masked joint.
#[test]
fn an_additive_with_no_delta_is_bit_identical_to_the_base_pose() {
    let rig = legs();
    let base = inf_anim::Pose::rest(&rig.skeleton);
    // The delta of a pose against ITSELF is the identity, whatever the pose is.
    let zero = inf_anim::additive_delta(&base, &base);
    let full = vec![1.0f32; base.locals.len()];
    assert_eq!(
        inf_anim::apply_additive(&base, &zero, &full),
        base,
        "a zero-delta additive at FULL weight moved the pose"
    );
    // …and at a partial weight, which is the case a `pslerp` that took a
    // shortcut through a normalize would break.
    let half = vec![0.5f32; base.locals.len()];
    assert_eq!(inf_anim::apply_additive(&base, &zero, &half), base);

    // **THE FALSIFICATION**: a real delta moves the masked joints and nothing
    // else.
    let feet = inf_anim::derive::foot_joints(&rig);
    let (l, r) = (feet[0] as usize, feet[1] as usize);
    let turned = glam::Quat::from_xyzw(0.0, 0.2588, 0.0, 0.9659).to_array();
    let mut aimed = base.clone();
    aimed.locals[l].rotation = turned;
    aimed.locals[r].rotation = turned;
    let delta = inf_anim::additive_delta(&base, &aimed);
    let mask = inf_anim::JointMask::new("left foot", [(l as u16, 1.0)], 0.0);
    let out = inf_anim::apply_additive(&base, &delta, &mask.resolve(base.locals.len(), 1.0));
    assert_ne!(
        out.locals[l], base.locals[l],
        "the masked joint did not take the delta"
    );
    assert_eq!(
        out.locals[r], base.locals[r],
        "an unmasked joint took the delta — the mask is not a mask"
    );
    println!(
        "\n=== the additive identity holds over {} joints, and the mask confines a real delta to 1 ===",
        base.locals.len()
    );
}

/// **THE AIM OFFSET REACHES THE POSE, AND THE FEET STAY ON THE GROUND**
/// (clause 3).
///
/// The arm above proves the arithmetic; this proves the **pass**, inside the
/// fixed step, over a machine that carries a real `aim_look` blend space. They
/// are different claims and P29.2 shipped the first with no consumer for the
/// second — the CHAR1a audit's item 88b, `apply_layers` with zero callers.
#[test]
fn the_aim_offset_layer_reaches_the_pose_and_leaves_the_feet_alone() {
    /// A one-joint clip that turns the SPINE by `deg` about its own X axis.
    fn leaning(joint: u16, deg: f32) -> AnimClip {
        let h = deg.to_radians() * 0.5;
        let (s, c) = (inf_math::psin(h), inf_math::pcos(h));
        let mut track = inf_anim::JointTrack::new(joint);
        track.rotation = Some(inf_anim::QuatTrack::new(
            vec![0.0, 1.0],
            vec![[s, 0.0, 0.0, c], [s, 0.0, 0.0, c]],
            inf_anim::Interpolation::Linear,
        ));
        AnimClip::new("sweep", vec![track])
    }
    const UP: inf_anim::ClipRef = [0xb2; 16];
    const FWD: inf_anim::ClipRef = [0xb3; 16];
    const DOWN: inf_anim::ClipRef = [0xb4; 16];

    let mut t = Topo::new(Ground::Flat);
    // `spine_01` is joint 7 of the fixture rig — inside the upper-body mask,
    // which is what the layer is supposed to be able to reach.
    let spine = 7u16;
    t.machine.states.push(SmState {
        name: inf_anim::als::LOOK_SWEEP_STATE.into(),
        motion: inf_anim::state_machine::Motion::Blend2D(inf_anim::BlendSpace2D::new(
            inf_anim::als::MOVE_X_VAR,
            inf_anim::als::MOVE_Y_VAR,
            vec![
                inf_anim::BlendEntry2D {
                    pos: [0.0, 1.0],
                    clip: UP,
                },
                inf_anim::BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: FWD,
                },
                inf_anim::BlendEntry2D {
                    pos: [0.0, -1.0],
                    clip: DOWN,
                },
            ],
        )),
        looping: false,
        speed: 1.0,
        position: (0.0, 0.0),
        on_enter: Vec::new(),
        on_exit: Vec::new(),
    });
    t.sweeps = vec![
        (UP, leaning(spine, -40.0)),
        (FWD, leaning(spine, 0.0)),
        (DOWN, leaning(spine, 40.0)),
    ];
    // Looking straight ahead: the sweep sits at its neutral and the layer has
    // nothing to say.
    t.set_aim(0.0, 0.0);
    t.settle(40);
    let level = t.pose_bytes();
    let level_spine = t.joint_rotation(spine as usize).expect("a posed spine");
    let level_soles = t.soles();

    // Looking UP: `aim_sweep(+60°)` is 1/6, so the sample is well off neutral.
    t.set_aim(0.0, 60.0);
    t.settle(40);
    let up = t.pose_bytes();
    assert_ne!(
        level, up,
        "nothing at all moved when the character looked up"
    );
    // **THE JOINT THE LAYER OWNS, AND NOTHING ELSE DOES.** Whole-pose bytes
    // cannot tell this pass from the look-at chain — the mutation run proved it:
    // disabling `apply_aim_offset` outright left this arm green, because the
    // head and neck were still moving. `spine_01` is the joint the sweep clips
    // author, and in the default `VelocityDirection` mode `LookAtLimits`'
    // `spine_share` is ZERO, so the look-at chain cannot reach it.
    let up_spine = t.joint_rotation(spine as usize).expect("a posed spine");
    assert_ne!(
        level_spine, up_spine,
        "the aim offset never reached the spine — `apply_aim_offset` did not \
         run, or its early-out swallowed a real sample"
    );
    // …and the legs kept walking: the mask is the upper body, so the feet are
    // still where the ground put them.
    let soles = t.soles();
    for (side, s) in soles.iter().enumerate() {
        assert!(
            s.penetration_mm() <= 10.0 && s.hover_mm() <= 10.0,
            "foot {side} left the ground while the character looked up: {s:?}"
        );
        assert!(
            (s.y - level_soles[side].y).abs() < 1.0e-3,
            "foot {side} moved {:.4} m when the aim offset came in — the mask \
             reaches the legs",
            (s.y - level_soles[side].y).abs()
        );
    }
    println!(
        "\n=== the aim offset moved the pose and both soles held to within \
         {:.4} mm ===",
        (soles[0].y - level_soles[0].y).abs() * 1000.0
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) LOOK-AT
// ─────────────────────────────────────────────────────────────────────────────

/// **THE HEAD TRACKS THE MOUSE TO WITHIN 2°, AND CLAMPS AT ITS LIMIT**
/// (clause 4).
///
/// The user's sentence, measured in the fixed step: *"when the user moves their
/// mouse … the character turns their head and sometimes body."* Driven through
/// the real runtime and read back off the animation bridge, so a wiring that
/// computed the number and dropped it — which is what P29.4 did for four waves
/// — fails here rather than looking right in a comment.
#[test]
fn the_head_tracks_the_look_direction_and_clamps_at_the_limit() {
    let mut t = Topo::new(Ground::Flat);
    t.settle(30);
    let limits = inf_anim::LookAtLimits {
        spine_share: 0.0,
        ..inf_anim::LookAtLimits::default()
    };
    println!("\n=== look-at: asked vs drawn (VelocityDirection) ===");
    for want in [-60.0, -30.0, 30.0, 60.0] {
        t.set_aim(want, 0.0);
        t.settle(20);
        let r = inf_ecs::anim_bridge::look_report(&t.world, HERO).expect("a look report");
        println!(
            "  asked {want:>6.1}°   chain {:>7.3}°   head {:>7.3}°   spine {} neck {}",
            r.total_yaw_deg, r.head_yaw_deg, r.spine, r.neck
        );
        assert!(
            (r.total_yaw_deg - want).abs() <= 2.0,
            "the chain took {:.3}° for a {want:.1}° look — past the 2° the brief \
             allows",
            r.total_yaw_deg
        );
        // The default rotation mode is `VelocityDirection`, which is ALS's own
        // "no spine rotation" case: the head and the neck track and the chest
        // does not.
        assert_eq!(r.spine, 0, "a VelocityDirection character leaned its spine");
        assert!(
            r.neck > 0 && r.head,
            "the chain did not reach the head: {r:?}"
        );
    }
    // **THE CLAMP.** Past the chain's own ceiling the character stops craning.
    let ceiling = inf_anim::chain_yaw_limit_deg(0, 1, &limits);
    t.set_aim(170.0, 0.0);
    t.settle(20);
    let r = inf_ecs::anim_bridge::look_report(&t.world, HERO).expect("a look report");
    println!(
        "  asked  170.0°   chain {:.3}°   ceiling {ceiling:.1}°",
        r.total_yaw_deg
    );
    assert!(
        r.total_yaw_deg <= ceiling + 1.0e-6,
        "the chain took {:.3}° past its {ceiling:.1}° ceiling",
        r.total_yaw_deg
    );
    assert!(
        r.head_yaw_deg.abs() <= limits.head_yaw_deg + 1.0e-6,
        "the HEAD alone took {:.3}° past its {:.1}° limit",
        r.head_yaw_deg,
        limits.head_yaw_deg
    );
    assert!(
        r.total_yaw_deg < 150.0,
        "the clamp is not clamping: {:.3}° of a 170° ask",
        r.total_yaw_deg
    );
    // …and a character looking straight ahead poses what it posed with no
    // look-at at all, which is what keeps every committed sample where it was.
    t.set_aim(0.0, 0.0);
    t.settle(20);
    assert!(
        inf_ecs::anim_bridge::look_report(&t.world, HERO).is_none(),
        "a character looking straight ahead reported a look"
    );
}

/// **AN NPC LOOKS AT THE PAWN, AND ONLY INSIDE ITS ATTENTION RADIUS**
/// (clause 4's second half).
///
/// The brief: *"NPCs look at their attention target (EMS3's witnessing target if
/// wired; else the nearest hero within N m, stated)."* It is the second, and it
/// is stated in `inf_ecs::pose::look_at_of`: the witness log records acts with a
/// place and a step and no notion of who is still interested in one, so wiring a
/// gaze to it would be inventing the missing half.
///
/// The falsification is the radius: the same NPC, moved past
/// [`inf_ecs::pose::NPC_ATTENTION_M`], stops looking.
#[test]
fn an_npc_turns_its_head_toward_the_pawn_and_stops_at_the_radius() {
    const NPC: Uuid = Uuid::from_u128(0x0b1b_0009);
    fn npc_look(at_x: f64) -> Option<inf_anim::LookAtReport> {
        let mut t = Topo::new(Ground::Flat);
        // A second character, not player-controlled, standing `at_x` metres to
        // the hero's right and facing straight ahead.
        let cm = CharacterMovement {
            player_controlled: false,
            ..Default::default()
        };
        let e = t.world.spawn_with_guid(NPC, "NPC", None);
        let mut tr = Transform::IDENTITY;
        tr.translation = Vec3d::new(at_x, cm.stand_half_height_m + RADIUS, 0.0);
        t.world.world_mut().entity_mut(e).insert((
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
            tr,
        ));
        t.world.mark_dirty();
        t.world.propagate();
        t.settle(30);
        inf_ecs::anim_bridge::look_report(&t.world, NPC)
    }
    // Five metres away: the hero is due `-x` of it, so the NPC's head comes
    // round toward `-90°` and the chain clamps to what a neck can do.
    let near = npc_look(5.0).expect("an NPC inside the radius looks at the pawn");
    println!(
        "\n=== an NPC 5 m from the pawn: chain {:.3}°, head {:.3}°, neck {} ===",
        near.total_yaw_deg, near.head_yaw_deg, near.neck
    );
    assert!(
        near.total_yaw_deg < -20.0,
        "the NPC did not turn toward the pawn: {near:?}"
    );
    // …and past the radius it looks straight ahead, which reports nothing.
    let far = npc_look(inf_ecs::pose::NPC_ATTENTION_M + 5.0);
    assert!(
        far.is_none(),
        "an NPC {} m away still watched the pawn: {far:?}",
        inf_ecs::pose::NPC_ATTENTION_M + 5.0
    );
    println!(
        "  …and none at {} m (the radius is {} m)",
        inf_ecs::pose::NPC_ATTENTION_M + 5.0,
        inf_ecs::pose::NPC_ATTENTION_M
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) THE CROWD ON THE SAME GRAPH
// ─────────────────────────────────────────────────────────────────────────────

/// **A `Near` AGENT'S ARMS FOLLOW ITS CLIP** (clause 7).
///
/// CHAR1a.3 proved a crowd agent's `hand_l` leaves the bind pose; this proves it
/// at the tier that is supposed to be *reduced*. `CrowdTier::Near` drops the SK1b
/// hand pass and keeps the whole pose, and the difference between those two is
/// the whole content of "reduced" — an agent whose arms stopped moving at 32 m
/// would be a crowd of statues on the far pavement.
///
/// Committed content, no island: `char1a3_gate`'s own arm hopes an island agent
/// happens to be `Near` (all 1 000 of the island's are `Far`), and this one puts
/// one there by construction.
#[test]
fn a_near_tier_agent_still_swings_its_arms() {
    use inf_ecs::components::{AnimStateMachine, SkeletalMesh, Transform};
    use inf_ecs::crowd::CrowdTier;

    // A rig with an arm, a clip that swings it, and a machine that plays it.
    let rig = legs();
    const ARM_CLIP: inf_anim::ClipRef = [0xb7; 16];
    let swing = {
        // `spine_03` is joint 9 and stands in for the arm the fixture rig does
        // not carry: what the arm asserts is that a NON-LEG joint the clip moves
        // is moved at `Near`, and the mask that could have stopped it is the
        // upper body's.
        let mut track = inf_anim::JointTrack::new(9);
        // 45 degrees about X, not Y: the joints above this one sit on its own Y
        // axis, and a rotation ABOUT that axis moves none of them. The first
        // spelling of this line turned the spine and measured a head that had
        // not moved, which reads exactly like a tier that stopped animating.
        let q = glam::Quat::from_xyzw(0.3827, 0.0, 0.0, 0.9239).to_array();
        track.rotation = Some(inf_anim::QuatTrack::new(
            vec![0.0, 1.0],
            vec![q, q],
            inf_anim::Interpolation::Linear,
        ));
        AnimClip::new("swing", vec![track])
    };
    let machine = StateMachine {
        states: vec![SmState::clip("swing", ARM_CLIP)],
        entry: 0,
        ..Default::default()
    };

    let posed_at = |tier: CrowdTier| -> glam::Vec3 {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(HERO, "Agent", None);
        world.world_mut().entity_mut(e).insert((
            Transform::IDENTITY,
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(1)),
                skeleton: Some(SKEL),
            },
            inf_ecs::crowd::CrowdAgent {
                guid: HERO,
                tier,
                feet_offset_m: 0.0,
                blocked: false,
                posture: inf_ecs::components::SlotPosture::default(),
                face: glam::DVec3::Z,
                posture_t: 0.0,
            },
        ));
        world.mark_dirty();
        world.propagate();
        let machines = |g: Uuid| (g == SM).then_some(&machine);
        let skels = |g: Uuid| (g == SKEL).then_some(&rig);
        let clips = |c: inf_anim::ClipRef| (c == ARM_CLIP).then_some(&swing);
        let vars = |_: Uuid| std::collections::BTreeMap::new();
        inf_ecs::pose::step_pose_evaluation(&mut world, DT, &machines, &skels, &clips, &vars);
        // A tier that does not pose leaves no entry — and on a world where NO
        // entity posed, no resource at all. Both read as the bind pose, which is
        // exactly what the renderer draws for a `Far` agent (the shared
        // entry-clip palette).
        let posed = world
            .world()
            .get_resource::<inf_ecs::pose::PoseStoreRes>()
            .and_then(|s| s.0.get(&HERO).cloned());
        let pose = posed
            .map(|p| p.pose)
            .unwrap_or_else(|| inf_anim::Pose::rest(&rig.skeleton));
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &pose);
        g[11].to_scale_rotation_translation().2
    };

    let bind = {
        let rest = inf_anim::Pose::rest(&rig.skeleton);
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &rest);
        g[11].to_scale_rotation_translation().2
    };
    let full = posed_at(CrowdTier::Full);
    let near = posed_at(CrowdTier::Near);
    let far = posed_at(CrowdTier::Far);
    println!(
        "\n=== the head-end joint under each tier ===\n  \
         bind {bind:?}\n  Full {full:?}\n  Near {near:?}\n  Far  {far:?}"
    );
    assert!(
        (near - bind).length() > 1.0e-3,
        "a Near agent posed its bind pose — the reduced tier stopped animating"
    );
    assert!(
        (near - full).length() < 1.0e-6,
        "a Near agent posed differently from a Full one: {near:?} vs {full:?} — \
         the reduction is not the hand pass"
    );
    // …and the falsification: `Far` really does not pose, which is what makes
    // the two claims above statements about the ladder.
    assert!(
        (far - bind).length() < 1.0e-9,
        "a Far agent posed something — the ladder is not a ladder"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (7) THE CARRIED ITEMS
// ─────────────────────────────────────────────────────────────────────────────

/// **THE REBOUND HERO CARRIES ITS LOD LADDER** (carried item 111, first half).
///
/// `record_character_ladder` writes `character_lod_assets` /
/// `character_lod_triangles` / `character_lod_switch_m` into the sidecar of the
/// asset the import minted, and `rebind_character` used to copy the payload to
/// the committed GUID with `import = None` — which on the fresh-file path of
/// `write_asset_at_with_id` does not merely fail to add the table, it wipes one.
///
/// This reads the committed body's own sidecar in the island project and asserts
/// the ladder is on it, with strictly decreasing triangle counts and one switch
/// distance per rung. **It does not assert that anything draws a lower rung**,
/// because nothing does: the second half of item 111 is that
/// `character_lod_assets` has a writer and no reader anywhere in the tree, and
/// that is PERF1's with the numbers this prints.
#[test]
fn the_rebound_hero_carries_its_lod_ladder() {
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the \
             rebind is local-only content and CI has none"
        );
        return;
    };
    let path = content.join("Starter_Body.inf_mesh");
    if !path.is_file() {
        eprintln!("SKIP: no rebound body at {}", path.display());
        return;
    }
    let side = inf_asset::AssetSidecar::load(&path).expect("the sidecar loads");
    let Some(import) = side.import.as_ref() else {
        panic!(
            "the rebound body at {} carries no [import] table at all — the \
             rebind wiped it (carried item 111)",
            path.display()
        );
    };
    let rungs = import
        .get("character_lod_assets")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("no `character_lod_assets` on the rebound body: {import:?}"));
    let tris: Vec<i64> = import
        .get("character_lod_triangles")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_integer()).collect())
        .unwrap_or_default();
    let switch: Vec<f64> = import
        .get("character_lod_switch_m")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_float()).collect())
        .unwrap_or_default();
    println!(
        "\n=== the rebound hero's ladder: {} rungs, {tris:?} triangles, \
         {switch:?} m ===",
        rungs.len()
    );
    assert!(rungs.len() >= 2, "a one-rung ladder is not a ladder");
    assert_eq!(tris.len(), rungs.len(), "a rung with no triangle count");
    assert_eq!(switch.len(), rungs.len(), "a rung with no switch distance");
    for w in tris.windows(2) {
        assert!(
            w[1] < w[0],
            "the rungs do not get cheaper: {tris:?} — `distinct_rungs` let a \
             duplicate through"
        );
    }
    for w in switch.windows(2) {
        assert!(
            w[1] > w[0],
            "the switch distances are not ordered: {switch:?}"
        );
    }
    // …and every rung is a real asset beside it, which is what makes the
    // dependency edges worth adding.
    for v in rungs {
        let id = v.as_str().expect("a rung guid is a string");
        assert!(
            uuid::Uuid::parse_str(id).is_ok(),
            "a rung guid that is not a guid: {id}"
        );
    }
    println!(
        "  NOTE: nothing in the engine reads `character_lod_assets` yet — the \
         hero still draws rung 0 ({} triangles) at every distance. That is the \
         second half of carried item 111 and it is PERF1's.",
        tris.first().copied().unwrap_or(0)
    );
}

/// **A SECOND CHARACTER IS NOT A SECOND PAWN** (carried item 112).
///
/// `camera_subject` answers "the first player-controlled character in `Guid`
/// order", and the editor's create-character door used to set
/// `player_controlled: true` unconditionally on a `v4` guid — so who the camera
/// followed was a byte race between the level's derived hero guid and a random
/// one, lost about one run in four.
///
/// The rule is that a level has one pawn: the door hands the flag to the first
/// character and to nothing after it.
#[test]
fn placing_a_second_character_does_not_move_the_pawn() {
    use inf_ecs::components::{CharacterMovement, Transform};
    let mut world = EcsWorld::new();
    // The hero, with a guid that sorts LOW — the island's own case.
    let hero = Uuid::from_u128(0x0000_0001);
    let e = world.spawn_with_guid(hero, "Hero", None);
    world.world_mut().entity_mut(e).insert((
        Transform::IDENTITY,
        CharacterMovement {
            player_controlled: true,
            ..Default::default()
        },
    ));
    world.mark_dirty();
    world.propagate();
    assert_eq!(inf_ecs::movement::camera_subject(&world), Some(hero));

    // A second character placed afterwards, with a guid that sorts LOWER still —
    // the worst case, and the one the old door lost.
    // The lowest guid there is, so the second character is the WORST case for
    // 's  — the run the old door lost.
    let lower = Uuid::nil();
    let e = world.spawn_with_guid(lower, "Placed", None);
    world.world_mut().entity_mut(e).insert((
        Transform::IDENTITY,
        CharacterMovement {
            // What the door now writes for a document that already has a pawn.
            player_controlled: false,
            ..Default::default()
        },
    ));
    world.mark_dirty();
    world.propagate();
    assert_eq!(
        inf_ecs::movement::camera_subject(&world),
        Some(hero),
        "a second character took the camera — and it sorts lower, which is the \
         exact case that lost one run in four"
    );
    // The falsification: with the OLD behaviour it does take it.
    let e = world.entity_of(lower).unwrap();
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.player_controlled = true;
    }
    assert_eq!(
        inf_ecs::movement::camera_subject(&world),
        Some(lower),
        "the arm above proves nothing: a second pawn does not take the camera \
         even when it is one"
    );
}

/// **THE LANDING STATES ARE REACHABLE** (clause 2's sharpest edge).
///
/// A graph that gates a landing on `land_alpha` can never enter one: the
/// movement step clears the prediction the moment `grounded` goes true — "a
/// grounded character predicting a landing is the stale-answer defect" — so the
/// two conditions are never satisfied on the same step. Found by reading the
/// producer, and this is the arm that keeps it found: the whole gait/air family
/// is walked over the parameter values the movement step really publishes, and
/// every state the map names must be entered by at least one of them.
#[test]
fn every_state_of_the_built_graph_is_reachable_from_a_published_parameter_set() {
    use inf_anim::als;
    let (sm, report) = inf_anim::build_locomotion_graph(&|name: &str| {
        let mut id = [0u8; 16];
        for (i, b) in name.as_bytes().iter().enumerate() {
            id[i % 16] ^= *b;
        }
        Some(id)
    });
    assert!(report.is_complete());
    // The states the machine deliberately never enters — the additive layer's
    // named clip sets. Everything else must be reachable.
    let manifest: Vec<&str> = inf_anim::LOCOMOTION_MAP
        .iter()
        .filter(|s| s.kind != inf_anim::SlotKind::State)
        .map(|s| s.state)
        .collect();

    /// One published parameter set, named.
    struct Frame(&'static str, Vec<(&'static str, f64)>);
    let g = als::LocoMode::Grounded.param();
    let frames = vec![
        Frame(
            "standing still",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 0.0),
            ],
        ),
        Frame(
            "walking",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 1.0),
            ],
        ),
        Frame(
            "running",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 2.0),
            ],
        ),
        Frame(
            "sprinting",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 3.0),
            ],
        ),
        Frame(
            "crouched, still",
            vec![
                (als::MODE_VAR, als::LocoMode::Crouch.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 0.0),
            ],
        ),
        Frame(
            "crouched, walking",
            vec![
                (als::MODE_VAR, als::LocoMode::Crouch.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 1.0),
            ],
        ),
        Frame(
            "jumping",
            vec![
                (als::MODE_VAR, als::LocoMode::FallFree.param()),
                (als::GROUNDED_VAR, 0.0),
            ],
        ),
        Frame(
            "falling",
            vec![
                (als::MODE_VAR, als::LocoMode::FallControlled.param()),
                (als::GROUNDED_VAR, 0.0),
                (als::FALL_SPEED_VAR, 4.0),
            ],
        ),
        Frame(
            "falling fast",
            vec![
                (als::MODE_VAR, als::LocoMode::FallControlled.param()),
                (als::GROUNDED_VAR, 0.0),
                (als::FALL_SPEED_VAR, 14.0),
            ],
        ),
        Frame(
            "landing light",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::LANDING_VAR, als::LANDING_SOFT),
                (als::TIME_SINCE_LAND_VAR, 0.0),
            ],
        ),
        Frame(
            "landing heavy",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::LANDING_VAR, als::LANDING_HARD),
                (als::TIME_SINCE_LAND_VAR, 0.0),
            ],
        ),
        Frame(
            "breaking the fall",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::LANDING_VAR, als::LANDING_ROLL),
                (als::TIME_SINCE_LAND_VAR, 0.0),
            ],
        ),
        Frame(
            "rolling",
            vec![
                (als::MODE_VAR, als::LocoMode::Roll.param()),
                (als::GROUNDED_VAR, 1.0),
            ],
        ),
        Frame(
            "ragdolling",
            vec![
                (als::MODE_VAR, als::LocoMode::Ragdoll.param()),
                (als::GROUNDED_VAR, 0.0),
            ],
        ),
        Frame(
            "getting up, face down",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::FACE_UP_VAR, 0.0),
            ],
        ),
        Frame(
            "ragdolling again",
            vec![
                (als::MODE_VAR, als::LocoMode::Ragdoll.param()),
                (als::GROUNDED_VAR, 0.0),
            ],
        ),
        Frame(
            "getting up, on its back",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::FACE_UP_VAR, 1.0),
            ],
        ),
        // ── the two stops, each from a cycle, chosen by the planted foot ──
        Frame(
            "walking again",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 1.0),
            ],
        ),
        Frame(
            "stopping on the left",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 0.0),
                (als::PLANTED_FOOT_VAR, -1.0),
            ],
        ),
        Frame(
            "running again",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 2.0),
            ],
        ),
        Frame(
            "stopping on the right",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 0.0),
                (als::PLANTED_FOOT_VAR, 1.0),
            ],
        ),
        // ── the eight turns in place ──
        Frame(
            "turning left 90",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, -90.0),
            ],
        ),
        Frame(
            "turning right 90",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, 90.0),
            ],
        ),
        Frame(
            "turning left 180",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, -170.0),
            ],
        ),
        Frame(
            "turning right 180",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, 170.0),
            ],
        ),
        Frame(
            "crouched, turning left 90",
            vec![
                (als::MODE_VAR, als::LocoMode::Crouch.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, -90.0),
            ],
        ),
        Frame(
            "crouched, turning right 90",
            vec![
                (als::MODE_VAR, als::LocoMode::Crouch.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, 90.0),
            ],
        ),
        Frame(
            "crouched, turning left 180",
            vec![
                (als::MODE_VAR, als::LocoMode::Crouch.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, -170.0),
            ],
        ),
        Frame(
            "crouched, turning right 180",
            vec![
                (als::MODE_VAR, als::LocoMode::Crouch.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::TURN_DEG_VAR, 170.0),
            ],
        ),
    ];

    // Walk the frames in order, letting each one settle, and record which states
    // the machine actually visits. `time_since_land` is a clock, so a landing
    // frame is held for one step and then released.
    let mut rt = inf_anim::SmRuntime::default();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    seen.insert(sm.states[sm.entry].name.clone());
    println!("\n=== every state of the built graph, from the published parameters ===");
    for Frame(what, vals) in &frames {
        let map: std::collections::BTreeMap<&str, f64> = vals.iter().copied().collect();
        let vars = |name: &str| map.get(name).copied().or(Some(0.0));
        let ctx = inf_anim::SmContext::new(&vars);
        // Enough steps for a fade to finish and a one-shot to run out.
        for _ in 0..90 {
            rt.advance(&sm, &ctx, 1.0 / 60.0);
            seen.insert(sm.states[rt.current].name.clone());
        }
        println!("  {what:<20} → {}", sm.states[rt.current].name);
    }
    let unreached: Vec<&str> = sm
        .states
        .iter()
        .map(|s| s.name.as_str())
        .filter(|n| !seen.contains(*n) && !manifest.contains(n))
        .collect();
    assert!(
        unreached.is_empty(),
        "these states cannot be entered by any parameter set the movement step \
         publishes: {unreached:?} — a state nothing can reach is a clip that \
         never plays"
    );
    // …and the manifest states really are unreachable, which is the other half
    // of the claim: they are the additive layer's, not the machine's.
    for n in &manifest {
        assert!(
            !seen.contains(*n),
            "the machine entered `{n}`, which is a named clip set and not a state"
        );
    }
    println!(
        "  {} states visited of {} ({} are the layer's clip sets)",
        seen.len(),
        sm.states.len(),
        manifest.len()
    );
}

/// **AN IMPORTED RIG CARRIES A ROLE TABLE** (clause 4's prerequisite, found in
/// PIE).
///
/// A `RoleIndex` over an empty table answers `None` to everything, and look-at,
/// the aim-offset mask and the SK1b hand pass are all written to do nothing when
/// it does — deliberately, so a quadruped is left alone. The UE bridge wrote
/// `roles: Vec::new()`, so all three were silently off on the one character the
/// game is about: measured in PIE on the island, the hero's aim reached −165.14°
/// and its head drew **0.00°**.
///
/// The falsification is the other half of the door: a rig whose bones are called
/// nothing the convention knows gets an EMPTY table and behaves as it did.
#[test]
fn an_imported_rig_has_the_roles_the_look_at_chain_needs() {
    use inf_anim::roles::{BoneRoleKind as K, BoneSide as S};
    // The mannequin's own names, which is what the MetaHuman rig uses too.
    let manny = inf_anim::manny::build_manny(&inf_anim::BodyParams::default())
        .expect("the shipped mannequin builds");
    let imported = inf_anim::SkeletonAsset::imported(manny.skeleton.clone());
    let idx = imported.role_index();
    for (kind, side) in [
        (K::Head, S::Center),
        (K::Neck, S::Center),
        (K::Spine, S::Center),
        (K::Pelvis, S::Center),
        (K::Foot, S::Left),
        (K::Foot, S::Right),
        (K::Hand, S::Left),
        (K::Hand, S::Right),
    ] {
        assert!(
            idx.first(kind, side).is_some(),
            "an imported mannequin has no {kind:?}/{side:?} — the look-at chain \
             and the upper-body mask both answer `None` on this rig"
        );
    }
    // …and the upper-body mask, which is what the aim-offset layer needs, can
    // actually be built from it.
    assert!(
        inf_anim::JointMask::upper_body("Mask_AimOffset", &imported.skeleton, idx).is_some(),
        "no upper body on an imported mannequin"
    );
    println!(
        "\n=== an imported {}-bone rig infers {} roles ===",
        imported.skeleton.len(),
        idx.rows().len()
    );

    // **THE FALSIFICATION**: names the convention does not carry infer nothing.
    let alien = inf_anim::Skeleton::new(
        (0..4)
            .map(|i| inf_anim::Joint {
                name: format!("Bone.{i:03}"),
                parent: (i > 0).then(|| (i - 1) as u16),
                inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                local_bind: inf_anim::JointTransform::IDENTITY,
            })
            .collect(),
    )
    .expect("a chain");
    let alien = inf_anim::SkeletonAsset::imported(alien);
    assert!(
        alien.role_index().is_empty(),
        "a rig with no convention invented a role table: {:?}",
        alien.roles
    );
    // …and the island's own hero, if this machine has one.
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — the rebound rig is local-only content");
        return;
    };
    let path = content.join("Starter.inf_skel");
    if !path.is_file() {
        eprintln!("SKIP: no rebound rig at {}", path.display());
        return;
    }
    let rig: inf_anim::SkeletonAsset =
        inf_asset::decode(&std::fs::read(&path).expect("the rig reads")).expect("it decodes");
    let idx = rig.role_index();
    println!(
        "  the island's hero: {} joints, {} roles",
        rig.skeleton.len(),
        idx.rows().len()
    );
    assert!(
        idx.first(K::Head, S::Center).is_some() && idx.first(K::Neck, S::Center).is_some(),
        "the island's rebound hero has no head or neck role — look-at cannot run \
         on it, which is exactly the PIE measurement this arm exists for"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (8) THE ISLAND'S OWN HERO
// ─────────────────────────────────────────────────────────────────────────────

/// The island level, loaded loose out of the local project — `char1a3_gate`'s
/// own door, mirrored so this file can stand its hero on the real ground.
fn loose_sim(content: &Path, slug: &str) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// **THE ISLAND'S HERO GETS FOOT GOALS, AND ITS SOLES REACH THEM** (clause 1, in
/// the world the game is set in).
///
/// The headless fixture asks "is the ground the engine found the ground that is
/// there"; this asks the other half — **did any of it run at all on the real
/// hero, on real terrain, with the real clip set**. Every one of the four
/// mechanisms this wave turned on has an all-or-nothing failure mode (a rig with
/// no role table, a clip with no channel, a mode outside the family, a probe that
/// finds nothing), and each of them looks exactly like "the feature is off".
///
/// The residual is `inf_ecs::anim_bridge::foot_error`: where the pose step left
/// the foot, against where the physics probe said the surface under it is.
#[test]
fn the_islands_hero_stands_on_the_ground_it_is_standing_on() {
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the \
             island is local-only content and CI has none"
        );
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!(
            "SKIP: no VancouverIsland.inf_lvl under {}",
            content.display()
        );
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    // Let the terrain stream in under it and the foot loop settle. The seam is
    // one fixed step wide by construction, and the ground arrives with the tile.
    for _ in 0..600 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let cm = sim
        .world()
        .entity_of(hero)
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::CharacterMovement>(e)
        })
        .expect("the hero has a movement component")
        .clone();
    let feet = inf_ecs::anim_bridge::feet_of(sim.world(), hero);
    let goals =
        inf_ecs::anim_bridge::bridge(sim.world()).and_then(|b| b.foot_ik.get(&hero).copied());
    let error = inf_ecs::anim_bridge::foot_error(sim.world(), hero);
    println!(
        "\n=== the island's hero after 600 steps ===\n  \
         mode {:?}  grounded {}  planted {}\n  \
         state {:?}\n  \
         feet published {:?}\n  goals {:?}\n  residual {:?}",
        cm.mode,
        cm.runtime.grounded,
        cm.mode.is_grounded_family() && cm.runtime.grounded,
        inf_ecs::anim_bridge::anim_state(sim.world(), hero).map(|s| s.name.clone()),
        feet.map(|f| f.map(|s| s.map(|s| s.world.y))),
        goals.map(|g| g.map(|x| x.map(|x| x.target.y))),
        error
    );
    assert!(
        cm.mode.is_grounded_family() && cm.runtime.grounded,
        "the hero is not standing on anything, so nothing below is a claim about \
         foot IK"
    );
    assert!(
        feet.is_some_and(|f| f.iter().any(Option::is_some)),
        "the pose step published no feet for the island's hero — its rig has no \
         `Foot` role and no `foot_*` bone, or it was never posed"
    );
    let goals = goals.expect("the movement step published no foot goals");
    assert!(
        goals.iter().any(Option::is_some),
        "the probe found no ground under either foot: {goals:?}"
    );
    let error = error.expect("no residual was published, so no foot was solved");
    let worst = error
        .iter()
        .flatten()
        .fold(0.0f64, |a, v| if v.abs() > a { v.abs() } else { a });
    println!("  worst residual: {:.2} mm", worst * 1000.0);
    assert!(
        worst < 0.010,
        "the island's hero's sole ended {:.2} mm from the ground the probe found",
        worst * 1000.0
    );
}

/// **THE EDITOR'S OWN DOOR GIVES THE PAWN FLAG TO THE FIRST CHARACTER ONLY**
/// (carried item 112, at the door rather than at the rule).
///
/// The arm above proves what `camera_subject` does with a document; this proves
/// what the door PUTS in one, which is where the defect was: it set
/// `player_controlled: true` unconditionally on a fresh `v4` guid, so the camera
/// followed whichever of the island's derived hero guid and a random one sorted
/// lower — lost about one run in four, and the run it lost photographed the
/// wrong face.
#[test]
fn the_editor_door_gives_the_pawn_flag_to_the_first_character_only() {
    use inf_editor_core::scene::SceneDoc;
    let mut doc = SceneDoc::new();
    let make = |doc: &mut SceneDoc, name: &str, x: f64| {
        doc.edit_create_character(
            name,
            Uuid::from_u128(0x5C10_00A0),
            Uuid::from_u128(0x5C10_00A2),
            Uuid::from_u128(0x5C10_00A6),
            None,
            glam::DVec3::new(x, 0.0, 0.0),
            None,
            1.8,
        )
    };
    let first = make(&mut doc, "Hero", 0.0);
    assert_eq!(
        inf_ecs::movement::camera_subject(doc.world()),
        Some(first),
        "the first character is not the pawn"
    );
    // Twenty more, because the defect was a RACE and one draw proves nothing: a
    // door that hands the flag out freely loses this the moment a fresh `v4`
    // sorts under the first one, which is a coin toss per placement.
    for i in 0..20 {
        let g = make(&mut doc, "Placed", 2.0 + f64::from(i));
        assert_ne!(g, first);
        assert_eq!(
            inf_ecs::movement::camera_subject(doc.world()),
            Some(first),
            "character {i} took the camera from the hero"
        );
    }
    println!("\n=== 21 characters placed, the pawn is still the first: {first} ===");
}

// ─────────────────────────────────────────────────────────────────────────────
// (9) THE POSE THE HERO ACTUALLY DRAWS  (audit CHAR1b.1)
// ─────────────────────────────────────────────────────────────────────────────

/// **THE ISLAND'S HERO DRAWS THE CLIPS ITS GRAPH NAMES, ON THE BONES THEY NAME.**
///
/// The wave's own table arm walks `LOCOMOTION_MAP` against the project's clip
/// FILES and the bridge arm counts states; neither of them ever asked what the
/// hero's joints did, and both were green while the character stood in its bind
/// pose at every gait. Two independent defects hid behind them:
///
/// 1. **The dev-dir loader read one directory.** `inf-import --dest UE` writes a
///    project's clips under `Content/UE/…`, so a `--level` boot resolved **0 of
///    65** clip GUIDs and posed every state at rest — while a PIE session over
///    the same document, which resolves by GUID out of the editor's database,
///    animated. Every island arm in this file runs on the loose door, so the
///    gates measured the host that could not see the content.
/// 2. **The clips were bound to another rig.** `inf_anim::pose::sample_clip`
///    addresses `JointTrack::joint` as an INDEX. The ALS sequences are authored
///    on the donor's 161-joint mannequin; the hero is a 342-joint MetaHuman, and
///    only the first **twelve** joint indices carry the same name. Measured
///    before the fix: of `ALS_N_Pose`'s 71 tracks, **59 landed on a bone of
///    another name** — the hero's upper arms sat at 13.59° / **44.90°** instead
///    of the donor's symmetric 13.59°, its fingers twisted up to 172°, and its
///    legs never left the bind pose at any gait.
///
/// So this arm asks the world, not the report: the graph's clips resolve through
/// the **runtime's own loader**, every one of them is authored on the rig that
/// plays it, and the drawn pose is neither the bind pose nor an asymmetric one.
///
/// **Mutations that red it**: make `payload_files_deep` read one directory (the
/// pose falls to bind — 12 joints move, all of them foot IK); skip the retarget
/// in `rebind_locomotion_graph` and re-import (the arms come out 21.7° apart).
#[test]
fn the_islands_hero_draws_the_clips_its_graph_names() {
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the \
             island is local-only content and CI has none"
        );
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }

    // (a) THE LOADER. Through the runtime's own door, not a test's.
    let (rigs, clips, machines) = inf_player::level::load_anim_assets_from_dir(&content);
    println!(
        "\n=== the loose loader sees {} skeletons, {} clips, {} machines ===",
        rigs.len(),
        clips.len(),
        machines.len()
    );

    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..600 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let sm = sim
        .world()
        .entity_of(hero)
        .and_then(|e| sim.world().world().get::<AnimStateMachine>(e))
        .and_then(|s| s.sm)
        .expect("the hero names a state machine");
    let asset = machines.get(&sm).expect("its machine is on disk");

    // (b) EVERY CLIP THE GRAPH NAMES RESOLVES, AND NAMES THIS RIG.
    let rig_guid = inf_ecs::pose::evaluated_pose(sim.world(), hero)
        .expect("the hero was posed")
        .skeleton;
    let refs = asset.machine.clip_refs();
    let mut missing = 0usize;
    let mut foreign: Vec<String> = Vec::new();
    for r in &refs {
        let id = Uuid::from_bytes(*r);
        match clips.get(&id) {
            None => missing += 1,
            Some(c) => {
                if c.skeleton.map(Uuid::from_bytes) != Some(rig_guid) {
                    foreign.push(c.clip.name.clone());
                }
            }
        }
    }
    println!(
        "  the hero's graph: {} states, {} clip refs — {missing} unresolved, {} on \
         another rig",
        asset.machine.states.len(),
        refs.len(),
        foreign.len()
    );
    assert_eq!(
        missing,
        0,
        "{missing} of the {} clips the hero's locomotion graph names do not resolve \
         through the loose loader — the host that boots a `--level` poses every \
         state at rest while PIE animates",
        refs.len()
    );
    assert!(
        foreign.is_empty(),
        "{} clip(s) the hero's graph plays are authored on another skeleton \
         ({:?}…) — a track is an INDEX into the pose, so every one of them drives \
         a bone of a different name",
        foreign.len(),
        foreign.iter().take(4).collect::<Vec<_>>()
    );

    // (c) THE POSE. Not the bind pose, and not lopsided.
    let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero)
        .expect("the hero was posed")
        .clone();
    let rig = rigs.get(&ep.skeleton).expect("the hero's rig is on disk");
    let rest = inf_anim::Pose::rest(&rig.skeleton);
    let names: Vec<&str> = rig
        .skeleton
        .joints()
        .iter()
        .map(|j| j.name.as_str())
        .collect();
    let moved = (0..names.len())
        .filter(|&i| {
            glam::Quat::from_array(ep.pose.locals[i].rotation)
                .angle_between(glam::Quat::from_array(rest.locals[i].rotation))
                .to_degrees()
                > 0.5
        })
        .count();
    let globals = inf_anim::pose::global_transforms(&rig.skeleton, &ep.pose);
    let abduction = |up: &str, low: &str| -> f32 {
        let i = names.iter().position(|n| n.eq_ignore_ascii_case(up));
        let j = names.iter().position(|n| n.eq_ignore_ascii_case(low));
        let (Some(i), Some(j)) = (i, j) else {
            return f32::NAN;
        };
        let a = globals[i].to_scale_rotation_translation().2;
        let b = globals[j].to_scale_rotation_translation().2;
        (b - a)
            .normalize()
            .dot(glam::Vec3::NEG_Y)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    let l = abduction("upperarm_l", "lowerarm_l");
    let r = abduction("upperarm_r", "lowerarm_r");
    println!(
        "  state {:?}: {moved} of {} joints off the bind pose; upper-arm abduction \
         L {l:.2}° R {r:.2}°",
        inf_ecs::anim_bridge::anim_state(sim.world(), hero).map(|s| s.name.clone()),
        names.len()
    );
    assert!(
        moved >= 30,
        "only {moved} of the hero's {} joints are off the bind pose — the four \
         that foot IK moves are pelvis, both thighs and both calves, so this is a \
         character standing in its rig's A-pose while its machine reports a state",
        names.len()
    );
    assert!(
        (l - r).abs() < 1.0,
        "the hero's upper arms hang {l:.2}° and {r:.2}° from vertical — the idle \
         ALS ships is symmetric, so a difference this size is a track landing on \
         a bone it does not name"
    );
}

/// **THE STATES THE ISLAND'S HERO REACHES FROM THE INPUT DOOR** (audit CHAR1b.1).
///
/// `every_state_of_the_built_graph_is_reachable_from_a_published_parameter_set`
/// hand-writes `turn_deg` and `planted_foot` and proves the GRAPH would enter
/// the turns and the stops. Nothing proved the movement step ever publishes
/// those values, and it did not: driven through the real input door on the
/// island, `|turn_deg|` never left **0.000** and `planted_foot` never left
/// **0**, so ten of the twenty-six machine states could not play. Two causes,
/// both measured:
///
/// * `planted_foot` reads `FootLock_L/R` off the playing clip. The island's 164
///   imported clips carried **no curve channel at all** — the deriver this wave
///   added to the clip-import path had never been run over them — so the stops
///   had nothing to choose a foot with. Re-imported, 164 of 164 carry the six
///   ALS channels and `stop_l` plays.
/// * turn-in-place runs only in `RotationMode::LookingDirection`, and a level
///   starts in `VelocityDirection` (where ALS does not turn in place either).
///   The only door to `LookingDirection` is releasing the aim key, so a player
///   who has never aimed can never turn in place. After one aim press+release a
///   220 °/s mouse swing plays `turn_r90`, and a crouched one `crouch_turn_r90`.
///
/// The remaining states need a world event this flat-road drive cannot make
/// (`fall`/`fall_fast`/`land_heavy`/`roll` need height, `ragdoll`/`getup_*` need
/// a ragdoll) and are named in the printout rather than claimed.
#[test]
fn the_islands_hero_plays_the_states_its_input_door_can_reach() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    let mut turn_max = 0.0f64;
    let mut planted: std::collections::BTreeSet<i64> = Default::default();
    let mut drive = |sim: &mut inf_player::runtime_sim::RuntimeSim,
                     seen: &mut std::collections::BTreeSet<String>,
                     turn_max: &mut f64,
                     planted: &mut std::collections::BTreeSet<i64>,
                     stop_ready: &mut usize,
                     n: usize,
                     held: &[&str],
                     axes: &[(&str, f32)]| {
        let ax: std::collections::BTreeMap<String, f32> =
            axes.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
        for _ in 0..n {
            sim.step_once(RuntimeInput::with_down(held.to_vec()).with_axes(ax.clone()));
            if let Some(s) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
                seen.insert(s.name.clone());
            }
            let t = inf_ecs::anim_bridge::anim_param(sim.world(), hero, "turn_deg").unwrap_or(0.0);
            if t.abs() > turn_max.abs() {
                *turn_max = t;
            }
            let foot =
                inf_ecs::anim_bridge::anim_param(sim.world(), hero, "planted_foot").unwrap_or(0.0);
            planted.insert(foot as i64);
            // **The stop edge's own precondition**, counted rather than assumed:
            // in a gait state, the gait scale under the walk threshold, and a
            // foot locked — all on ONE step. See the arm's docs.
            let gait = inf_ecs::anim_bridge::anim_param(sim.world(), hero, "gait").unwrap_or(1.0);
            let in_gait = inf_ecs::anim_bridge::anim_state(sim.world(), hero)
                .is_some_and(|s| matches!(s.name.as_str(), "walk" | "run" | "sprint"));
            if in_gait && gait <= 0.1 && foot.abs() > 0.5 {
                *stop_ready += 1;
            }
        }
    };
    let mut stop_ready = 0usize;
    let mut go = |sim: &mut _, n, held: &[&str], axes: &[(&str, f32)]| {
        drive(
            sim,
            &mut seen,
            &mut turn_max,
            &mut planted,
            &mut stop_ready,
            n,
            held,
            axes,
        )
    };
    go(&mut sim, 600, &[], &[]); // settle
                                 // **Run and stop, five times, at five different phases.** Which foot is
                                 // locked when the input goes is a property of where the cycle happens to be,
                                 // so a single stop is a coin toss and five is the mechanism.
    for extra in [0usize, 7, 13, 19, 29] {
        go(&mut sim, 80 + extra, &[], &[("move_y", 1.0)]); // start → walk → run
        for _ in 0..70 {
            go(&mut sim, 1, &[], &[]); // the stop, a step at a time
        }
    }
    go(&mut sim, 60, &["walk"], &[("move_y", 1.0)]); // held walk
    go(&mut sim, 90, &["sprint"], &[("move_y", 1.0)]); // sprint
    for _ in 0..90 {
        go(&mut sim, 1, &[], &[]); // …and the stop out of a sprint
    }
    go(&mut sim, 6, &["crouch"], &[]); // click
    go(&mut sim, 60, &[], &[]); // crouch_idle
    go(&mut sim, 90, &[], &[("move_y", 1.0)]); // crouch_walk
    go(&mut sim, 30, &[], &[("look_x", 260.0)]); // a crouched turn needs the mode below
    go(&mut sim, 6, &["crouch"], &[]);
    go(&mut sim, 60, &[], &[]); // stand
    go(&mut sim, 2, &["jump"], &[]);
    go(&mut sim, 100, &[], &[]); // airborne → land_light
    go(&mut sim, 10, &["aim"], &[]); // → Aiming
    go(&mut sim, 10, &[], &[]); // → LookingDirection
    go(&mut sim, 30, &[], &[("look_x", 220.0)]); // the swing
    go(&mut sim, 120, &[], &[]); // the turn plays out
    go(&mut sim, 6, &["crouch"], &[]);
    go(&mut sim, 30, &[], &[("look_x", 260.0)]);
    go(&mut sim, 120, &[], &[]);

    let all: Vec<&str> = inf_anim::LOCOMOTION_MAP
        .iter()
        .filter(|s| s.kind == inf_anim::SlotKind::State)
        .map(|s| s.state)
        .collect();
    let missed: Vec<&&str> = all.iter().filter(|n| !seen.contains(**n)).collect();
    println!(
        "\n=== driven through the input door: {} of {} states played ===\n  {:?}\n  \
         not played: {missed:?}\n  |turn_deg| reached {turn_max:.3}; planted_foot took \
         {planted:?}",
        seen.len(),
        all.len(),
        seen
    );
    for want in [
        "idle",
        "start",
        "walk",
        "run",
        "sprint",
        "crouch_idle",
        "crouch_walk",
        "jump",
        "land_light",
    ] {
        assert!(
            seen.contains(want),
            "`{want}` never played on the island from the input door — played:              {seen:?}"
        );
    }
    // The FAMILIES whose gating parameter was dead. Which member of each plays
    // depends on the sign of the accumulated body yaw, so the arm asks for the
    // family: a member is a coin toss, a family is the mechanism.
    for (family, members) in [
        (
            "a standing turn-in-place",
            ["turn_l90", "turn_r90", "turn_l180", "turn_r180"].as_slice(),
        ),
        (
            "a crouched turn-in-place",
            [
                "crouch_turn_l90",
                "crouch_turn_r90",
                "crouch_turn_l180",
                "crouch_turn_r180",
            ]
            .as_slice(),
        ),
    ] {
        assert!(
            members.iter().any(|m| seen.contains(*m)),
            "not one of {family}'s clips played ({members:?}) — played: {seen:?}"
        );
    }
    assert!(
        turn_max.abs() > 45.0,
        "`turn_deg` never exceeded {turn_max:.3}° — the movement step publishes no \
         turn, so the eight turn-in-place clips cannot play whatever the graph says"
    );
    assert!(
        planted.iter().any(|f| *f != 0),
        "`planted_foot` was 0 on every step — the clips carry no `FootLock_*` \
         channel, so neither stop can choose a foot"
    );
    // **THE STOP, reported and not asserted, because the number says why.** Its
    // edge wants a gait state, `gait <= 0.1` and a locked foot on ONE step. Over
    // this whole drive — five run-and-stops at five phases plus a sprint stop —
    // that coincidence held on the count below. The parameter is alive (asserted
    // above) and the edge exists (`als::transitions_for`); what is not true is
    // that a player who stops sees a stop clip.
    println!(
        "  the stop edge's precondition (a gait state, gait <= 0.1, a foot locked, one step) held on {stop_ready} step(s); a stop clip played: {}",
        seen.contains("stop_l") || seen.contains("stop_r")
    );
}
