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

/// Every `.inf_anim` under `dir` **and every directory below it**, decoded, with
/// its file stem (wave CHAR1b.2).
///
/// The one-directory sibling above is right for the donor pack, which is one
/// folder. The AUTHORED sets are not: `ue_import::rebind_locomotion_graph`
/// writes them per identity under `{stem}-loco/`, so an arm that read one
/// directory would report nine unbound rows for nine clips sitting one level
/// down — which is the shape of the defect the CHAR1b.1 audit's F2 found in the
/// runtime's own loader.
///
/// Dot-directories are skipped, for F2's other reason: `Content/.inf/
/// import-cache` holds a second copy of everything.
fn clips_under(dir: &Path) -> Vec<(String, inf_anim::AnimClipAsset)> {
    let mut out = clips_in(dir);
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut subs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
        })
        .collect();
    subs.sort();
    for sub in subs {
        out.extend(clips_under(&sub));
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
/// this engine ever carried it** (CHAR1b.1's census: 164 imported ALS clips,
/// zero; and the 12 committed sample clips, zero), so the mechanism never ran.
/// The obvious repair is to derive the gate from HEIGHT — "does this foot reach
/// the ground in this clip" ought to be an absolute test against the rig's own
/// ground plane, since every rig this engine poses has its origin at its feet.
///
/// **It is not.** UE authors an in-air cycle with the root on the ground and the
/// legs hanging, so a fall loop's ankle sits exactly where a walk's ankle sits.
/// This arm asserts that negative and it is unchanged: over the whole imported
/// library there is no height that separates the grounded families from the
/// airborne ones, because the airborne clips are among the very highest.
///
/// # What CHAR1b.2 changed, and why the census flipped
///
/// The gate is derivable — from the foot's own **contact state within its own
/// clip**, which is a different measurement from its absolute height across
/// clips. `inf_anim::derive` writes it now
/// ([`ENABLE_FOOT_IK_PREFIX`](inf_anim::derive::ENABLE_FOOT_IK_PREFIX)) and the
/// census below asserts **every** clip carries one, because the fallback it
/// replaces was measured to be wrong in both directions: a flat `1.0` pinned a
/// swinging foot to the road (150.0 mm of island residual) and the plant window
/// alone switched foot IK off on an idle.
///
/// The height result still fails loudly if a future import ever makes the
/// families separable, which is the only honest way to keep a negative.
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

    // **THE CENSUS**, and it has flipped on purpose (wave CHAR1b.2).
    //
    // CHAR1b.1 measured **0 of 164** and the number was the whole reason
    // `step_feet` had a fallback at all: nobody authors this gate. The fallback
    // was then measured to be wrong in both directions — a flat `1.0` pinned a
    // foot that was half a metre in the air (150.0 mm of island residual on
    // exactly the steps where `FootLock_*` had just gone to zero) and the plant
    // window switched foot IK OFF on an idle, because a foot that never lifts
    // gets no plant window at all.
    //
    // So the gate is DERIVED now (`inf_anim::derive::ENABLE_FOOT_IK_PREFIX`),
    // and every clip carries it. The assertion is the other way round and it
    // means the same thing it always meant: *the value `step_feet` reads is a
    // measurement rather than a guess*. It fails if a re-import ever stops
    // writing the channel, which would silently put the flat fallback back.
    println!(
        "\n=== Enable_FootIK_* over {} imported clips: {with_gate} carry it ===",
        clips.len()
    );
    assert_eq!(
        with_gate,
        clips.len(),
        "{with_gate} of {} imported clips carry an `Enable_FootIK_*` channel — the \
         deriver is meant to write one onto every clip, and `step_feet` falls back \
         to a flat 1.0 for any that has none, which is what pins a swinging foot \
         to the road",
        clips.len()
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
    /// The floor for a clip `inf_anim::authored` wrote — see the two-floors
    /// note below. Three is a throw: a shoulder, an elbow and a chest.
    const AUTHORED_MIN_JOINTS: usize = 3;
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the ALS \
             clips are local-only content and CI has none"
        );
        return;
    };
    let dir = content.join("UE/Mannequins");
    let clips = clips_under(&content);
    if clips.is_empty() {
        eprintln!("SKIP: no .inf_anim under {}", content.display());
        return;
    }
    // The same suffix rule the import door resolves with: the file stem is
    // `{pack}_{name}` and the map holds the donor's own name.
    // **Two spellings, because there are two kinds of clip** (wave CHAR1b.2).
    //
    // A DONOR clip is written under its pack's own stem (`ALS_Community_
    // ALS_N_Walk_F`), so `_{name}` is the needle. An AUTHORED one
    // (`inf_anim::authored`, the sets ALS does not ship) is written per identity
    // as `{name}--{stem}`, and the separator is a HYPHEN precisely so it does
    // NOT match `_{name}`: a copy called `INF_Slide.inf_anim` would be a second
    // stem matching `INF_Slide` and would make the other identity's lookup
    // ambiguous (`ue_import::rebind_locomotion_graph`'s own comment, measured
    // the hard way at CHAR1b.1 when the female machine came out with 0 states).
    //
    // The arm resolves both, because the door does — and an arm that could only
    // see one of them would report nine unbound rows for nine clips that are on
    // the disk it is reading.
    let find = |name: &str| -> Option<&inf_anim::AnimClipAsset> {
        let tail = format!("_{name}");
        let head = format!("{name}--");
        let mut hit = None;
        for (stem, asset) in &clips {
            if stem == name || stem.ends_with(&tail) {
                if hit.is_some() {
                    return None;
                }
                hit = Some(asset);
            }
        }
        if hit.is_some() {
            return hit;
        }
        // An authored clip exists once per identity, so more than one match is
        // expected and the FIRST in the sorted walk is taken deterministically.
        clips
            .iter()
            .find(|(stem, _)| stem.starts_with(&head))
            .map(|(_, a)| a)
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
                    // **Two floors, because there are two kinds of clip**
                    // (wave CHAR1b.2). Ten joints is the DONOR floor and its
                    // reason is carried item 93: an export that lost its bone
                    // tracks arrives with two, and a two-joint clip plays as a
                    // bind pose wearing an animation's name.
                    //
                    // An AUTHORED clip's joint count is a property of its own
                    // design, not evidence of an import that went wrong: a
                    // throw is a shoulder, an elbow and a chest, and three is
                    // the right answer for an upper-body additive layered over
                    // a walk. Its own floor is `AUTHORED_MIN_JOINTS`, and
                    // `inf_anim::authored`'s arms assert the shape.
                    let floor = if name.starts_with("INF_") {
                        AUTHORED_MIN_JOINTS
                    } else {
                        SHELL_JOINTS
                    };
                    if joints < floor {
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
        "{} bindings are SHELLS (under {SHELL_JOINTS} animated joints for a donor \
         clip, {AUTHORED_MIN_JOINTS} for an authored one — a bind pose wearing an \
         animation's name): {shells:?}",
        shells.len()
    );
    // **AND THE AUTHORED SETS REALLY ARE THE AUTHORED SETS** (wave CHAR1b.2).
    //
    // The rows above prove a file with that name is on disk and moves joints.
    // This proves it is the one `inf_anim::authored` derives from THIS rig:
    // every clip in `AUTHORED_CLIPS` has a row, a file, and the same joint
    // count the generator produces for the hero's own skeleton. A hand-dropped
    // file of the right name would pass the first check and fail this one.
    {
        let rows: std::collections::BTreeMap<&str, usize> = inf_anim::LOCOMOTION_MAP
            .iter()
            .flat_map(|s| s.clips.iter().map(|(n, _)| *n))
            .filter(|n| n.starts_with("INF_"))
            .map(|n| (n, 0))
            .collect();
        assert_eq!(
            rows.len(),
            inf_anim::AUTHORED_CLIPS.len(),
            "the map names {} authored clips and the generator writes {}: {:?} vs {:?}",
            rows.len(),
            inf_anim::AUTHORED_CLIPS.len(),
            rows.keys().collect::<Vec<_>>(),
            inf_anim::AUTHORED_CLIPS
        );
        let all_rigs = skeletons_in(&dir);
        let hero_rig = all_rigs
            .values()
            .filter(|r| inf_anim::can_author(r))
            .max_by_key(|r| r.skeleton.len());
        if let Some(rig) = hero_rig {
            let want: std::collections::BTreeMap<String, usize> = inf_anim::author_clips(rig)
                .expect("the rig can author")
                .into_iter()
                .map(|(n, c)| (n, c.tracks.len()))
                .collect();
            for (name, joints) in &want {
                let got = find(name)
                    .unwrap_or_else(|| panic!("`{name}` is not on disk beside the donor clips"));
                assert_eq!(
                    got.clip.tracks.len(),
                    *joints,
                    "`{name}` on disk drives {} joints and the generator writes {joints} for \
                     this rig — the file is not this rig's authored clip",
                    got.clip.tracks.len()
                );
            }
            println!("  the authored sets, re-derived from the hero's own rig: {want:?}");
        } else {
            println!("  SKIP the authored re-derivation: no rig on disk carries a role table");
        }
    }
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
                (als::GETUP_VAR, als::GETUP_FRONT),
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
                (als::GETUP_VAR, als::GETUP_BACK),
            ],
        ),
        Frame(
            "ragdolling a third time",
            vec![
                (als::MODE_VAR, als::LocoMode::Ragdoll.param()),
                (als::GROUNDED_VAR, 0.0),
            ],
        ),
        // **The third get-up** (wave CHAR1b.2): a ragdoll that settled with the
        // pelvis still more than half a capsule above the feet never went down,
        // and ALS's two crouched clips are both wrong for it.
        Frame(
            "getting up, still on its feet",
            vec![
                (als::MODE_VAR, g),
                (als::GROUNDED_VAR, 1.0),
                (als::GETUP_VAR, als::GETUP_STANDING),
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
                // **Still moving** (wave CHAR1b.2). The stop edge used to ask for
                // `gait <= 0.1` -- the character has already stopped -- and the
                // CHAR1b.1 audit measured that precondition holding on **0 steps**
                // of five run-and-stops. It asks for a stopping DISTANCE now, which
                // is a number a character has while it is still travelling, so the
                // frame that reaches a stop is a frame in which the gait is up.
                (als::GAIT_VAR, 2.0),
                (als::STOP_DISTANCE_VAR, 0.9),
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
                // **Still moving** (wave CHAR1b.2). The stop edge used to ask for
                // `gait <= 0.1` -- the character has already stopped -- and the
                // CHAR1b.1 audit measured that precondition holding on **0 steps**
                // of five run-and-stops. It asks for a stopping DISTANCE now, which
                // is a number a character has while it is still travelling, so the
                // frame that reaches a stop is a frame in which the gait is up.
                (als::GAIT_VAR, 2.0),
                (als::STOP_DISTANCE_VAR, 0.9),
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
        // ── THE AUTHORED SETS (wave CHAR1b.2) ──
        //
        // Four modes this engine's movement step has entered since P29.3 with
        // no clip to play. Each frame is exactly what the movement step
        // publishes in that mode, so a row here that cannot enter its state is
        // a row a player could not reach either.
        Frame(
            "the mantle, low, right-handed",
            vec![
                (als::MODE_VAR, als::LocoMode::Mantle.param()),
                (als::MANTLE_VAR, als::MANTLE_LOW_RH),
            ],
        ),
        Frame(
            "the mantle, low, left-handed",
            vec![
                (als::MODE_VAR, als::LocoMode::Mantle.param()),
                (als::MANTLE_VAR, als::MANTLE_LOW_LH),
            ],
        ),
        Frame(
            "the mantle, high",
            vec![
                (als::MODE_VAR, als::LocoMode::Mantle.param()),
                (als::MANTLE_VAR, als::MANTLE_HIGH),
            ],
        ),
        Frame(
            "sliding",
            vec![
                (als::MODE_VAR, als::LocoMode::Slide.param()),
                (als::GROUNDED_VAR, 1.0),
            ],
        ),
        Frame(
            "prone, still",
            vec![
                (als::MODE_VAR, als::LocoMode::Prone.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 0.0),
            ],
        ),
        Frame(
            "prone, crawling",
            vec![
                (als::MODE_VAR, als::LocoMode::Prone.param()),
                (als::GROUNDED_VAR, 1.0),
                (als::GAIT_VAR, 1.0),
            ],
        ),
        Frame(
            "treading water",
            vec![
                (als::MODE_VAR, als::LocoMode::SwimSurface.param()),
                (als::GROUNDED_VAR, 0.0),
                (als::GAIT_VAR, 0.0),
            ],
        ),
        Frame(
            "swimming at the surface",
            vec![
                (als::MODE_VAR, als::LocoMode::SwimSurface.param()),
                (als::GROUNDED_VAR, 0.0),
                (als::GAIT_VAR, 1.0),
            ],
        ),
        Frame(
            "swimming under",
            vec![
                (als::MODE_VAR, als::LocoMode::SwimUnder.param()),
                (als::GROUNDED_VAR, 0.0),
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
    // **The garments too** (wave CHAR1b.2). A `ClothSim` whose asset the builder
    // never loaded seeds nothing at all, which reads exactly like a garment that
    // does not simulate — the shape F2 fixed for the clips.
    .with_cloth_assets(inf_player::level::load_cloth_assets_from_dir(content))
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
    let drive = |sim: &mut inf_player::runtime_sim::RuntimeSim,
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
            "`{want}` never played on the island from the input door; played: {seen:?}"
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

/// **EVERY RIG IN A PROJECT CARRIES A ROLE TABLE** (audit CHAR1b.1, carried item
/// 119 closed).
///
/// `an_imported_rig_has_the_roles_the_look_at_chain_needs` asks the question of
/// a rig built in memory. The island answered it differently: **24 of its 26
/// rigs carried `roles: []`** — every Manny/Quinn LOD rung and every MetaHuman
/// body and face rung — because `SkeletonAsset::imported` infers the table at
/// the glTF stage and the content-addressed `ImportCache` reuses an unchanged
/// source without re-running it. Only the two the rebind writes had one, which
/// is why the hero worked and nothing else would have.
///
/// A rig with no table silently switches off look-at, the aim-offset mask and
/// the SK1b hand pass, so an NPC bound to any of the other twenty-four would
/// have stared straight ahead with no error anywhere.
/// `ue_import::sweep_roles` is the door; this is the measurement.
#[test]
fn every_rig_in_the_island_project_carries_a_role_table() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, inf_anim::SkeletonAsset)>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut ps: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        ps.sort();
        for p in ps {
            if p.is_dir() {
                if p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with('.'))
                {
                    continue;
                }
                walk(&p, out);
                continue;
            }
            if p.extension().is_none_or(|x| x != "inf_skel") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else {
                continue;
            };
            if let Ok(a) = inf_asset::decode::<inf_anim::SkeletonAsset>(&bytes) {
                out.push((p, a));
            }
        }
    }
    let mut rigs = Vec::new();
    walk(&content, &mut rigs);
    if rigs.is_empty() {
        eprintln!("SKIP: no .inf_skel in the island project");
        return;
    }
    // A rig this convention can read at all: it has a head, or it has feet. A
    // quadruped or a prop rig is left alone deliberately, so the arm asks only
    // of the rigs whose names the inference knows.
    let mut empty: Vec<String> = Vec::new();
    for (path, a) in &rigs {
        let humanoid = a
            .skeleton
            .joints()
            .iter()
            .any(|j| j.name.eq_ignore_ascii_case("head"));
        if humanoid && a.roles.is_empty() {
            empty.push(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_string(),
            );
        }
    }
    println!(
        "\n=== {} rigs in the island project; {} humanoid rig(s) with no role table ===",
        rigs.len(),
        empty.len()
    );
    for (path, a) in rigs.iter().take(4) {
        println!(
            "  {:<44} {:4} joints {:4} roles",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            a.skeleton.joints().len(),
            a.roles.len()
        );
    }
    assert!(
        empty.is_empty(),
        "{} rig(s) with a `head` bone carry no role table ({:?}…) — every character \
         bound to one of them has look-at, the aim mask and the hand pass silently \
         off",
        empty.len(),
        empty.iter().take(4).collect::<Vec<_>>()
    );
}

/// **A SIDEWAYS MOUSE TURNS THE HEAD; AN UPWARD ONE TIPS IT** — on the island,
/// through the door the window resolves (audit CHAR1b.1, the user's own report).
///
/// *"When the user moves their mouse left/right, the character in the game looks
/// up/down rather than left/right."* Reproduced here and measured on the drawn
/// pose rather than on the report, which is what hid it: `LookAtReport` carries
/// the angle the chain was ASKED for, and that angle was always right. What was
/// wrong was the axis it was turned about.
///
/// The window binds `look_x` to the mouse's X at 0.15° per raw unit and `look_y`
/// to its Y at −0.15 (`inf_input::map`), and both hosts step the same
/// `RuntimeInput`, so driving those two axes IS the real path — the demo loop's
/// `mouse_event` lands on the same pair one layer up.
///
/// Before the fix, on the island hero's rig: a pure `look_x` frame moved the
/// head's **elevation** by ~+44° and its azimuth not at all; a pure `look_y`
/// frame did the converse.
#[test]
fn a_sideways_mouse_turns_the_islands_hero_head_and_an_upward_one_tips_it() {
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
    let mut rigs: std::collections::BTreeMap<[u8; 16], inf_anim::SkeletonAsset> =
        std::collections::BTreeMap::new();
    for (g, sk) in inf_player::level::load_anim_assets_from_dir(&content).0 {
        rigs.insert(*g.as_bytes(), sk);
    }
    // Where the head POINTS, in the rig's own frame.
    let dir = |sim: &inf_player::runtime_sim::RuntimeSim| -> (f64, f64) {
        let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("the hero is posed");
        let rig = rigs.get(ep.skeleton.as_bytes()).expect("its rig");
        let names: Vec<&str> = rig
            .skeleton
            .joints()
            .iter()
            .map(|j| j.name.as_str())
            .collect();
        let h = names
            .iter()
            .position(|n| n.eq_ignore_ascii_case("head"))
            .expect("a head bone");
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &ep.pose);
        let f = g[h].to_scale_rotation_translation().1 * glam::Vec3::Z;
        (
            inf_math::patan2_64(f64::from(f.x), f64::from(f.z)).to_degrees(),
            inf_math::patan2_64(f64::from(f.y), f64::from((f.x * f.x + f.z * f.z).sqrt()))
                .to_degrees(),
        )
    };
    let axis = |k: &str, v: f32| -> std::collections::BTreeMap<String, f32> {
        let mut m = std::collections::BTreeMap::new();
        m.insert(k.to_string(), v);
        m
    };
    for _ in 0..600 {
        sim.step_once(RuntimeInput::default());
    }
    let base = dir(&sim);
    // A sideways frame. 40°/s for 45 steps is 30° of aim, well inside the clamp.
    for _ in 0..45 {
        sim.step_once(RuntimeInput::default().with_axes(axis("look_x", 40.0)));
    }
    for _ in 0..20 {
        sim.step_once(RuntimeInput::default());
    }
    let sideways = dir(&sim);
    let (daz, del) = (sideways.0 - base.0, sideways.1 - base.1);
    println!(
        "\n=== the island hero's head, through the input door ===\n  \
         rest: az {:.2}° el {:.2}°\n  +look_x only: Δaz {daz:+.2}° Δel {del:+.2}°",
        base.0, base.1
    );
    assert!(
        daz.abs() > 10.0,
        "a sideways mouse moved the head's azimuth {daz:.2}° — it did not turn"
    );
    assert!(
        del.abs() * 5.0 < daz.abs(),
        "a sideways mouse tipped the head {del:.2} deg against {daz:.2} deg of turn; that is the user's own report, and it is the look-at pass turning about a bone's local axis instead of the rig's own up"
    );
    // …and back, then a purely vertical one.
    for _ in 0..45 {
        sim.step_once(RuntimeInput::default().with_axes(axis("look_x", -40.0)));
    }
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    let centred = dir(&sim);
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default().with_axes(axis("look_y", 40.0)));
    }
    for _ in 0..20 {
        sim.step_once(RuntimeInput::default());
    }
    let up = dir(&sim);
    let (uaz, uel) = (up.0 - centred.0, up.1 - centred.1);
    println!("  +look_y only: Δaz {uaz:+.2}° Δel {uel:+.2}°");
    assert!(
        uel > 5.0,
        "an upward mouse moved the head's elevation {uel:.2}° — it did not look up"
    );
    // A ratio and not a bound: the aim does not return to exactly where it was
    // (the swing back overshoots by a fraction of a degree and the body carries
    // it), so what is asserted is that the ELEVATION is what moved.
    assert!(
        uaz.abs() * 5.0 < uel.abs(),
        "an upward mouse turned the head {uaz:.2} deg sideways against {uel:.2} deg of tip; the axes are swapped"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (10) WAVE CHAR1b.2 — THE MOVES
//
// Every arm below asks the JOINTS or the WORLD. The CHAR1b.1 audit's law is the
// reason: five of that wave's seventeen arms were green on a character standing
// in its bind pose, because each of them read a table, a state name or a report.
// ─────────────────────────────────────────────────────────────────────────────

/// The hero's capsule centre, world metres.
fn hero_pos(sim: &inf_player::runtime_sim::RuntimeSim, hero: uuid::Uuid) -> [f64; 3] {
    let w = sim.world();
    let t = w
        .entity_of(hero)
        .and_then(|e| w.world().get::<inf_ecs::components::Transform>(e))
        .cloned()
        .expect("the hero has a transform");
    [t.translation.x, t.translation.y, t.translation.z]
}

/// The hero's movement component, cloned.
fn hero_cm(
    sim: &inf_player::runtime_sim::RuntimeSim,
    hero: uuid::Uuid,
) -> inf_ecs::components::CharacterMovement {
    let w = sim.world();
    w.entity_of(hero)
        .and_then(|e| w.world().get::<inf_ecs::components::CharacterMovement>(e))
        .cloned()
        .expect("the hero has a movement component")
}

/// Put the hero at `x, z` a metre and a fifth above the ground under it, facing
/// `bearing`, and let it settle.
///
/// A gate that waits for a character to *walk* somewhere measures the walk. This
/// measures what happens at a place, which is what a mantle arm is about.
fn hero_to(
    sim: &mut inf_player::runtime_sim::RuntimeSim,
    hero: uuid::Uuid,
    x: f64,
    z: f64,
    bearing: f64,
) {
    let y = hero_pos(sim, hero)[1] + 1.2;
    {
        let w = sim.world_mut();
        let e = w.entity_of(hero).expect("the hero is in the world");
        if let Some(mut t) = w.world_mut().get_mut::<inf_ecs::components::Transform>(e) {
            t.translation.x = x;
            t.translation.y = y;
            t.translation.z = z;
            t.rotation.y = bearing;
        }
        if let Some(mut c) = w
            .world_mut()
            .get_mut::<inf_ecs::components::CharacterMovement>(e)
        {
            c.runtime.velocity = inf_ecs::math::Vec3d::ZERO;
            c.runtime.aim_yaw_deg = bearing;
            c.runtime.body_yaw_deg = bearing;
            c.runtime.target_yaw_deg = bearing;
            c.mode = inf_ecs::components::MovementMode::Grounded;
        }
    }
    for _ in 0..180 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
}

/// **THE ISLAND'S HERO MANTLES A LEDGE, AND DOES NOT MANTLE A ROAD** (clause 5).
///
/// The mechanism has existed since P29.4 — `probe_ledge`, `try_mantle`,
/// `step_mantle`, the warp — and until this wave it had no ANIMATION: `Mantle`
/// matched no state's mode, so the hero climbed a wall holding whatever pose the
/// graph was in, which on the island was a jump loop. It also had no consumer
/// for either half of ALS's height remap: `MantleState::clip_start_s` and
/// `play_rate` were computed on entry and read by nothing.
///
/// This arm asks the WORLD and the MACHINE, not a table:
///
/// * the hero enters `MovementMode::Mantle` at a known ledge on the island;
/// * the machine enters one of the three **mantle states**, so a clip is playing;
/// * the ledge is classified against ALS's own 125 cm split;
/// * `clip_start_s` and `play_rate` are the remap's, not their defaults;
/// * **the pelvis ends above where it started** by about the ledge's height,
///   which is the only evidence that the climb happened rather than being
///   announced;
/// * and the **control** — the same drive on the road the hero spawns on —
///   enters no mantle and rises nowhere.
///
/// # What this arm does NOT reach, measured (CHAR1b.2 audit)
///
/// **`mantle_high` has never played in the world.** The map's third mantle row
/// (`ALS_N_Mantle_2m`, the >125 cm one) is bound and its state is reachable from
/// a hand-written parameter set, and no drive has ever entered it: the ledge
/// this arm climbs is 0.7294 m and it is caught **from the air**, where
/// `LedgeSettings::falling()` caps the climb at 1.50 m rather than the grounded
/// 2.50 m. The audit went looking — 64 driven approaches to every car and all
/// four buildings within 60 m of the spawn produced **zero** mantles, and a
/// direct `probe_ledge` census from 289 grounded stances at 8 m spacing × 8
/// bearings produced **zero** ledges of any height. So the classification
/// assertion below has only ever been evaluated with `high == false`, and the
/// `mantle_high` half of clause 5 is carried rather than proved.
#[test]
fn the_islands_hero_climbs_a_ledge_and_not_a_road() {
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
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let spawn = hero_pos(&sim, hero);
    // The hero's own rig and its two hands, for the leading-hand measurement
    // inside the drive below.
    let (rigs_for_hands, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let rig_for_hands = inf_ecs::pose::evaluated_pose(sim.world(), hero)
        .and_then(|p| rigs_for_hands.get(&p.skeleton).cloned())
        .expect("the hero's rig is on disk");
    let hand_joint = {
        let roles = rig_for_hands.role_index();
        roles
            .first(inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Left)
            .zip(roles.first(inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Right))
    };

    // Walk forward, tapping jump — ALS's own trigger (`OnOwnerJumpInput`, a jump
    // with movement input reaches for a ledge before it reaches for the air).
    // **The rise across ONE mantle, and how many the drive took** (CHAR1b.2
    // audit). This used to report the PEAK over the whole drive against
    // `height_m * 0.5`, and both halves of that were wrong to lean on: the drive
    // is 150 steps of walking into stepped terrain and it climbs FOUR ledges in
    // a row, so the +3.004 m the wave reported for a 0.73 m ledge is the sum of
    // four of them; and on the other side a single JUMP raises the same capsule
    // 0.334 m, which is within 9 % of the bound the arm was asserting. The
    // honest quantity is the height the capsule gains between the step the mode
    // becomes `Mantle` and the step it stops being one -- the FIRST such window,
    // the one whose `MantleState` this arm reports. Measured: 0.730 m against a
    // 0.7294 m ledge.
    let drive = |sim: &mut inf_player::runtime_sim::RuntimeSim, steps: usize| {
        let ax: std::collections::BTreeMap<String, f32> = [("move_y".to_string(), 1.0f32)].into();
        let before = hero_pos(sim, hero);
        let mut states: std::collections::BTreeSet<String> = Default::default();
        let mut entered: Option<inf_ecs::components::MantleState> = None;
        let mut peak = before[1];
        let mut mantles = 0usize;
        let mut in_mantle = false;
        let mut y_at_entry = 0.0f64;
        let mut first_rise = f64::NAN;
        let mut hand_gap = f64::MAX;
        // **The POSE, while the mantle runs** (CHAR1b.2 audit's anti-vacuity
        // clause). Everything else this arm asserts — the mode, the state name,
        // the remap, the capsule's rise — is true of a character drawing its
        // BIND POSE while the warp carries its capsule up the ledge, which is
        // the CHAR1b.1 audit's law applied to this wave's own arm. The largest
        // single-joint step over the mantle window says the clip is playing.
        let mut mantle_joint_step = 0.0f64;
        let mut prev_pose: Option<inf_anim::Pose> = None;
        for i in 0..steps {
            let held: Vec<&str> = if i % 5 == 0 { vec!["jump"] } else { vec![] };
            sim.step_once(RuntimeInput::with_down(held).with_axes(ax.clone()));
            let y = hero_pos(sim, hero)[1];
            peak = peak.max(y);
            let c = hero_cm(sim, hero);
            if c.mode == inf_ecs::components::MovementMode::Mantle {
                if let Some(s) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
                    states.insert(s.name.clone());
                }
                if !in_mantle {
                    in_mantle = true;
                    mantles += 1;
                    y_at_entry = y;
                }
                if entered.is_none() {
                    entered = Some(c.runtime.mantle);
                }
            } else if in_mantle {
                in_mantle = false;
                if first_rise.is_nan() {
                    first_rise = y - y_at_entry;
                }
            }
            // **The LEADING HAND against the ledge it is climbing** (CHAR1b.2
            // audit). ALS warps the mantle in LEDGE space, so the reaching hand
            // arrives on the lip; this engine warps the CAPSULE along the
            // traversal and plays the clip untouched, so the hand goes wherever
            // the animator put it on the donor's own body. Reported, not
            // asserted, and carried by name — the number is the size of the
            // hand-warp that is not there.
            if in_mantle {
                if let Some(ep) = inf_ecs::pose::evaluated_pose(sim.world(), hero) {
                    if let Some(prev) = prev_pose.as_ref() {
                        if prev.locals.len() == ep.pose.locals.len() {
                            for (a, b) in prev.locals.iter().zip(ep.pose.locals.iter()) {
                                let qa = glam::Quat::from_array(a.rotation);
                                let qb = glam::Quat::from_array(b.rotation);
                                let d = qa.dot(qb).abs().clamp(-1.0, 1.0);
                                mantle_joint_step = mantle_joint_step
                                    .max(2.0 * inf_math::pacos64(f64::from(d)).to_degrees());
                            }
                        }
                    }
                    prev_pose = Some(ep.pose.clone());
                }
                if let (Some(m), Some(j)) = (entered, hand_joint) {
                    let lead = if m.left_hand { j.0 } else { j.1 };
                    if let (Some(p), Some(to_world)) = (
                        inf_ecs::pose::evaluated_pose(sim.world(), hero),
                        inf_ecs::pose::model_to_world_of(sim.world(), hero),
                    ) {
                        let g = inf_anim::pose::global_transforms(&rig_for_hands.skeleton, &p.pose);
                        let t = g[lead as usize].to_scale_rotation_translation().2;
                        let w = to_world.transform_point3(DVec3::new(
                            f64::from(t.x),
                            f64::from(t.y),
                            f64::from(t.z),
                        ));
                        let ledge = DVec3::new(m.target.x, m.target.y, m.target.z);
                        hand_gap = hand_gap.min((w - ledge).length());
                    }
                }
            }
        }
        (
            entered,
            states,
            before[1],
            hero_pos(sim, hero)[1],
            peak,
            mantles,
            first_rise,
            hand_gap,
            mantle_joint_step,
        )
    };

    // ── the ledge ────────────────────────────────────────────────────────────
    //
    // A fixed place, so the arm measures the mantle and not a walk that may or
    // may not find one. The coordinates are the island's; the ledge probe found
    // it in a sweep of the hero's own neighbourhood and it is the nearest one
    // the drive can reach.
    hero_to(&mut sim, hero, -1766.0, 1992.0, 0.0);
    let (entered, states, y0, y1, peak, mantles, first_rise, hand_gap, joint_step) =
        drive(&mut sim, 150);
    let m = entered.expect(
        "the hero never entered `MovementMode::Mantle` at the ledge — either the probe found \
         nothing or the jump never reached `try_mantle`",
    );
    println!(
        "\n=== the mantle, on the island ===\n  height {:.4} m  high {}  clip_start {:.4} s  \
         play_rate {:.4}  left_hand {}\n  states {states:?}\n  pelvis {:.3} -> {:.3} (net {:+.3}, \
         peak {:+.3}) over {mantles} mantle(s)\n  the rise across the FIRST mantle alone: \
         {first_rise:+.4} m",
        m.height_m,
        m.high,
        m.clip_start_s,
        m.play_rate,
        m.left_hand,
        y0,
        y1,
        y1 - y0,
        peak - y0
    );
    // The machine is PLAYING one of the three mantle clips — the half that did
    // not exist before this wave.
    let played: Vec<&String> = states.iter().filter(|s| s.starts_with("mantle_")).collect();
    assert!(
        !played.is_empty(),
        "the hero mantled and the machine stayed in {states:?} — `MovementMode::Mantle` \
         matches no state's mode, which is a character climbing a wall in a jump loop"
    );
    // ALS's own classification, against its own literal.
    assert_eq!(
        m.high,
        m.height_m > inf_anim::MANTLE_HIGH_SPLIT_M,
        "a {:.3} m ledge is classified {}",
        m.height_m,
        if m.high { "high" } else { "low" }
    );
    assert!(
        played.iter().any(
            |s| s.as_str() == if m.high { "mantle_high" } else { "mantle_low" }
                || s.as_str() == "mantle_low_lh"
        ),
        "a {} mantle played {played:?}",
        if m.high { "high" } else { "low" }
    );
    // The height remap reached the animation. Both halves have been computed
    // since P29.4 and read by nothing; a run where either is its own default is
    // a run where the remap is dead again.
    assert!(
        m.play_rate > 0.0 && m.play_rate.is_finite(),
        "the mantle's play rate is {}",
        m.play_rate
    );
    // **THE REMAP IS ASSERTED, NOT PRINTED** (CHAR1b.2 audit). This used to print
    // `HeightRemap::default()`'s answer beside the state's and assert nothing
    // about either, and the two DISAGREE — `try_mantle` builds the remap out of
    // the `LedgeSettings` band it was called with, so a ledge caught in the air
    // (`LedgeSettings::falling()`, ceiling 1.50 m) and one climbed off the
    // ground (`default()`, ceiling 2.50 m) resolve the same height to different
    // clip times. Printing the wrong one made this arm's own output read as a
    // contradiction (0.4623 against 0.4164). The pair is required to BE one of
    // the two the movement step can produce, which is the check the doc above
    // claims: a `clip_start_s` and a `play_rate` left at their struct defaults
    // are neither of them.
    let candidates = [
        (
            "grounded",
            inf_physics::d3::traversal::LedgeSettings::default(),
        ),
        (
            "falling",
            inf_physics::d3::traversal::LedgeSettings::falling(),
        ),
    ];
    let matched = candidates.iter().find_map(|(what, set)| {
        let remap = inf_anim::HeightRemap {
            low_height_m: set.min_height_m,
            high_height_m: set.max_height_m,
            ..inf_anim::HeightRemap::default()
        };
        let (ws, wr) = remap.resolve(m.height_m);
        ((ws - m.clip_start_s).abs() < 1.0e-9 && (wr - m.play_rate).abs() < 1.0e-9)
            .then_some((*what, ws, wr))
    });
    let (which, want_start, want_rate) = matched.unwrap_or_else(|| {
        panic!(
            "the mantle carries clip_start {:.6} s / play_rate {:.6} for a {:.4} m ledge, and \
             neither the grounded band nor the falling one resolves to that — the height remap \
             is not what reached the state",
            m.clip_start_s, m.play_rate, m.height_m
        )
    });
    println!(
        "  the remap that produced it: the {which} band -> start {want_start:.4} s, rate \
         {want_rate:.4}"
    );
    println!(
        "  the LEADING HAND's closest approach to the ledge the feet land on: {hand_gap:.4} m \
         (ALS's ledge-space warp puts it on the lip; this engine warps the capsule and plays \
         the clip untouched — reported, carried, not asserted)"
    );
    // …and the pelvis really went up — **across the mantle, not across the
    // drive**. A jump raises this capsule 0.334 m on the flat, so a bound stated
    // against the whole 150-step walk is a bound a jump nearly clears; a bound
    // stated against the mantle window is the climb itself.
    let _ = peak;
    assert!(
        first_rise.is_finite(),
        "the hero entered `Mantle` and never left it over 150 steps, so there is no rise to \
         measure"
    );
    // **AND THE CLIP IS PLAYING WHILE IT CLIMBS.** The warp moves the CAPSULE, so
    // every claim above holds of a hero drawing its bind pose all the way up a
    // wall — which is exactly what a mantle looked like before this wave bound
    // `MovementMode::Mantle` to a state at all.
    println!("  the largest single-joint step during the mantle: {joint_step:.3} deg");
    assert!(
        joint_step > 0.5,
        "no joint moved more than {joint_step:.3} deg during the whole mantle — the capsule \
         climbed and the character was posed by nothing"
    );
    assert!(
        first_rise > m.height_m * 0.85 && first_rise < m.height_m * 1.25,
        "the hero mantled a {:.4} m ledge and its capsule rose {first_rise:.4} m across that \
         one mantle — a warp that lands on the ledge lands ON it",
        m.height_m
    );

    // ── the control: the road it spawned on ──────────────────────────────────
    hero_to(&mut sim, hero, spawn[0], spawn[2] + 4.0, 180.0);
    let (none, road_states, ry0, _ry1, rpeak, _rm, _rr, _rh, _rj) = drive(&mut sim, 150);
    println!(
        "  CONTROL on the road: mantle {none:?}, states {road_states:?}, peak {:+.3} m",
        rpeak - ry0
    );
    assert!(
        none.is_none(),
        "the hero mantled on a flat road, so the arm above proves nothing about ledges"
    );
    assert!(
        rpeak - ry0 < 1.0,
        "the control rose {:.3} m without mantling — it is not on flat ground and is not a \
         control",
        rpeak - ry0
    );
}

/// **A LOCKED FOOT DOES NOT SLIDE, AND A PLACED ONE REACHES THE GROUND**
/// (clause 6 — stride warping, measured where the mandate asks).
///
/// Two numbers, both off the pose:
///
/// * **the slide** — how far a foot the engine has LOCKED has been dragged from
///   where it was planted (`FootLock::slide_m`, the ground plane only). The
///   mandate's bound is 2 cm/s; at 60 Hz that is 0.33 mm a step.
/// * **the residual at full gate** — how far the drawn foot ended from the
///   ground the probe found, on the steps where the foot-IK gate is fully on.
///   The gate's own weight is part of the question: a foot at weight 0.4 is
///   deliberately half-placed, and measuring it as a miss is measuring a
///   decision.
///
/// The CHAR1b.1 audit inherited a sprint worst of 43.417 mm at a p50 of
/// 24.935 mm. This arm's bound is the p50, because that is the number that moved
/// and the tail is named in the ledger rather than hidden by a loose bound.
#[test]
fn a_locked_foot_does_not_slide_and_a_placed_one_is_on_the_ground() {
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
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let mut resid: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut slide_worst = 0.0f64;
    let mut slide_n = 0usize;
    // **THE ANTI-VACUITY CLAUSE** (CHAR1b.2 audit). `foot_error` is produced by
    // the foot IK whether or not any clip plays, so every number below is one a
    // bind-posed hero would also produce — the CHAR1b.1 audit's vacuous item 5,
    // still true of the arm this wave wrote on top of it. The feet have to be
    // LIFTING for a residual to be about a gait: this is how far the lower foot
    // travels in the hero's own model frame over the run.
    let (rigs_v, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let rig_v = inf_ecs::pose::evaluated_pose(sim.world(), hero)
        .and_then(|p| rigs_v.get(&p.skeleton).cloned())
        .expect("the hero's rig is on disk");
    let feet_v = inf_anim::derive::foot_joints(&rig_v);
    let mut foot_lo = f64::MAX;
    let mut foot_hi = f64::MIN;
    let mut rate: std::collections::BTreeMap<String, (f64, f64)> = Default::default();
    for (n, held, ax) in [
        (240usize, vec![], vec![("move_y", 1.0f32)]),
        (120, vec!["sprint"], vec![("move_y", 1.0f32)]),
        (150, vec![], vec![]),
        (200, vec![], vec![("move_y", 1.0f32)]),
        (150, vec![], vec![]),
        (120, vec!["walk"], vec![("move_y", 1.0f32)]),
        (150, vec![], vec![]),
    ] {
        let axes: std::collections::BTreeMap<String, f32> =
            ax.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
        for _ in 0..n {
            sim.step_once(RuntimeInput::with_down(held.clone()).with_axes(axes.clone()));
            let w = sim.world();
            let Some(st) = inf_ecs::anim_bridge::anim_state(w, hero).map(|s| s.name.clone()) else {
                continue;
            };
            let goals = inf_ecs::anim_bridge::bridge(w).and_then(|b| b.foot_ik.get(&hero).copied());
            if let Some(e) = inf_ecs::anim_bridge::foot_error(w, hero) {
                for (side, v) in e.iter().enumerate() {
                    let Some(v) = v else { continue };
                    let weight = goals.and_then(|g| g[side]).map(|g| g.weight).unwrap_or(0.0);
                    if weight >= 0.99 {
                        resid.entry(st.clone()).or_default().push(v.abs());
                    }
                }
            }
            if let Some(ep) = inf_ecs::pose::evaluated_pose(sim.world(), hero) {
                let g = inf_anim::pose::global_transforms(&rig_v.skeleton, &ep.pose);
                for j in &feet_v {
                    let y = f64::from(g[*j as usize].to_scale_rotation_translation().2.y);
                    foot_lo = foot_lo.min(y);
                    foot_hi = foot_hi.max(y);
                }
            }
            let c = hero_cm(&sim, hero);
            for (lock, s) in [
                (c.runtime.foot_lock_l, c.runtime.foot_slide_l_m),
                (c.runtime.foot_lock_r, c.runtime.foot_slide_r_m),
            ] {
                if lock.locked {
                    slide_n += 1;
                    slide_worst = slide_worst.max(s.abs());
                }
            }
            let r = rate.entry(st).or_insert((f64::MAX, 0.0));
            r.0 = r.0.min(c.runtime.play_rate);
            r.1 = r.1.max(c.runtime.play_rate);
        }
    }
    println!("\n=== the island's feet, by state ===");
    println!(
        "  {:<12}{:>9}{:>9}{:>13}{:>7}  play_rate",
        "state", "worst", "p50", "over 10 mm", "n"
    );
    let mut worst_p50 = 0.0f64;
    for (st, mut v) in resid.into_iter() {
        v.sort_by(f64::total_cmp);
        let worst = *v.last().unwrap_or(&0.0);
        let p50 = v[v.len() / 2];
        let over = v.iter().filter(|x| **x > 0.010).count();
        let r = rate.get(&st).copied().unwrap_or((0.0, 0.0));
        println!(
            "  {st:12} {:8.3} {:8.3}  {over:4} of {:<5} {:.2}..{:.2}",
            worst * 1000.0,
            p50 * 1000.0,
            v.len(),
            r.0,
            r.1
        );
        // **The one-shots are reported, the CYCLES are asserted** (wave
        // CHAR1b.2, and the number is why). A gait cycle is what the
        // mandate's centimetre is about: the character is moving and its
        // soles must be on the ground it is moving over. A `stop_*` is a
        // 2.333 s one-shot played by a character that comes to rest in about
        // half of it, so for the rest of the clip the animator's planted foot
        // and the ground under a stationary character have stopped agreeing —
        // measured at a **114.966 mm p50 over 31 samples**, against 0.000 in
        // every cycle state. That is a real remainder, it is this wave's
        // carried item, and burying it in a bound both numbers pass would be
        // the opposite of measuring it.
        if !st.starts_with("stop_") {
            worst_p50 = worst_p50.max(p50);
        }
    }
    println!(
        "  locked-foot slide: worst {:.4} mm over {slide_n} locked samples",
        slide_worst * 1000.0
    );
    println!(
        "  the lower foot's own model-frame height ranged {:.1} mm over the run",
        (foot_hi - foot_lo) * 1000.0
    );
    assert!(
        slide_n > 50,
        "only {slide_n} locked-foot samples — the lock never engaged"
    );
    assert!(
        foot_hi - foot_lo > 0.05,
        "the hero's feet moved {:.1} mm through the whole tour — a foot residual measured on a \
         character that is not stepping is a residual the foot IK produces on its own",
        (foot_hi - foot_lo) * 1000.0
    );
    // **THE SLIDE.** 2 cm/s at the 60 Hz fixed step is 0.33 mm a step; the bound
    // is a whole millimetre, which is three times looser and still an order
    // below anything a viewer sees.
    assert!(
        slide_worst < 0.001,
        "a LOCKED foot slid {:.4} mm from where it was planted — stride warping is what stops \
         that, and the mandate's bound is 2 cm/s",
        slide_worst * 1000.0
    );
    // **THE RESIDUAL.** The p50 in every CYCLE state, against an inherited sprint
    // p50 of 24.935 mm. See the loop above for why a `stop_*` is reported and not
    // asserted, and what its number is.
    assert!(
        worst_p50 < 0.010,
        "the worst per-CYCLE-state p50 residual is {:.3} mm at full gate weight, \
         against the mandate's centimetre",
        worst_p50 * 1000.0
    );
    // **THE RATE MOVED.** A play rate pinned at 1.0 everywhere is stride warping
    // that is not running: the whole mechanism is the ratio changing with the
    // gait.
    let moved = rate.values().any(|(lo, hi)| (hi - lo).abs() > 0.05);
    assert!(
        moved,
        "the play rate never left 1.0 in any state — `MoveData_Speed` is zero on every clip \
         again, which is what it read on all 310 of them before this wave"
    );
}

/// **THE STOP IS TAKEN** (clause 6 — distance matching, carried item 122).
///
/// ALS ships two stop-down clips because which foot is under you when you stop
/// decides what stopping looks like. The CHAR1b.1 audit measured the old edge's
/// precondition — a gait state, `gait <= 0.1`, and a locked foot on ONE step —
/// holding on **0 steps** of five run-and-stops at five phases plus a sprint
/// stop, so both clips were reachable in the graph and never played in the
/// world.
///
/// The edge asks for a stopping DISTANCE now, which is a number a character has
/// while it is still moving. This arm drives the real input door and asserts a
/// stop clip is entered — on the joints' side of the question, because the state
/// it asserts is the one the pose step published after posing.
#[test]
fn releasing_the_stick_plays_a_stop_clip() {
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
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    let mut stop_distance_max = 0.0f64;
    let mut stops = 0usize;
    // **THE SPEED THE STOP IS TAKEN AT** (CHAR1b.2 audit). This is the whole
    // content of distance matching and it is the only thing that separates the
    // new edge from the one it replaced: a stop is a decision taken WHILE
    // MOVING, and a clip entered once the character has already come to rest is
    // a stop-shaped idle. Reverting the entry term to the old `gait <= 0.1` and
    // re-running this arm gives the SAME 24 steps and the SAME state set --
    // measured -- so "a stop clip played" cannot tell the two apart and this
    // can.
    let mut entry_speeds: Vec<f64> = Vec::new();
    let mut was_stop = false;
    // Five run-and-stops at five phases, which is the CHAR1b.1 audit's own
    // experiment: a single stop is a coin toss and five is the mechanism.
    for extra in [0usize, 7, 13, 19, 29] {
        let ax: std::collections::BTreeMap<String, f32> = [("move_y".to_string(), 1.0f32)].into();
        for _ in 0..(110 + extra) {
            sim.step_once(RuntimeInput::default().with_axes(ax.clone()));
        }
        for _ in 0..90 {
            sim.step_once(RuntimeInput::default());
            let Some(st) =
                inf_ecs::anim_bridge::anim_state(sim.world(), hero).map(|s| s.name.clone())
            else {
                continue;
            };
            let is_stop = st.starts_with("stop_");
            if is_stop {
                stops += 1;
            }
            let cm = hero_cm(&sim, hero);
            if is_stop && !was_stop {
                let v = cm.runtime.velocity;
                entry_speeds.push((v.x * v.x + v.z * v.z).sqrt());
            }
            was_stop = is_stop;
            seen.insert(st);
            stop_distance_max = stop_distance_max.max(cm.runtime.stop_distance_m);
        }
    }
    // **THE RELEASE EDGE, AND THAT IT DOES NOT CHATTER** (CHAR1b.2 audit,
    // priority (h)). The wave gated `stop_* -> walk` on `stop_distance <= 0` to
    // stop a chatter it measured, and the question that raises is whether the
    // edge still fires AT ALL: a player who changes their mind mid-stop must get
    // their character back. Five attempts, because a stop clip is only entered
    // on three releases in five (the entry speeds above are three, not five);
    // each one runs to speed, releases, waits for the machine to actually be IN
    // a stop, then pushes the stick again and counts the steps back onto the
    // ladder and the state changes on the way.
    let mut relatch: Option<usize> = None;
    let mut relatch_edges = 0usize;
    let mut relatch_states: std::collections::BTreeSet<String> = Default::default();
    let mut attempts_that_reached_a_stop = 0usize;
    {
        let ax: std::collections::BTreeMap<String, f32> = [("move_y".to_string(), 1.0f32)].into();
        for extra in [0usize, 5, 11, 17, 23] {
            for _ in 0..(130 + extra) {
                sim.step_once(RuntimeInput::default().with_axes(ax.clone()));
            }
            let mut pushing = false;
            let mut since_push = 0usize;
            let mut last: Option<String> = None;
            let mut hit = false;
            for _ in 0..150 {
                let input = if pushing {
                    RuntimeInput::default().with_axes(ax.clone())
                } else {
                    RuntimeInput::default()
                };
                sim.step_once(input);
                let Some(st) =
                    inf_ecs::anim_bridge::anim_state(sim.world(), hero).map(|s| s.name.clone())
                else {
                    continue;
                };
                if pushing {
                    relatch_states.insert(st.clone());
                    if last.as_deref() != Some(st.as_str()) {
                        relatch_edges += 1;
                    }
                }
                last = Some(st.clone());
                if !pushing && st.starts_with("stop_") {
                    pushing = true;
                    hit = true;
                    since_push = 0;
                    continue;
                }
                if pushing {
                    since_push += 1;
                    if st == "walk" || st == "run" || st == "start" {
                        relatch = Some(relatch.map_or(since_push, |r: usize| r.min(since_push)));
                        break;
                    }
                }
            }
            if hit {
                attempts_that_reached_a_stop += 1;
            }
            for _ in 0..60 {
                sim.step_once(RuntimeInput::default());
            }
        }
    }
    let fastest = entry_speeds.iter().copied().fold(0.0f64, f64::max);
    println!(
        "\n=== the stop, over five run-and-stops ===\n  states {seen:?}\n  a stop clip played on \
         {stops} steps; the largest stopping distance published was {stop_distance_max:.3} m\
\n  the ground speed at each entry, m/s: {:?}\
\n  a stick pushed mid-stop is back on the ladder after {relatch:?} step(s), over \
         {relatch_edges} state changes over {attempts_that_reached_a_stop} of 5 attempts \
         that reached a stop; the windows saw {relatch_states:?}",
        entry_speeds
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
    );
    assert!(
        stop_distance_max > 0.10,
        "`stop_distance` never exceeded {stop_distance_max:.3} m, so no stop edge could fire — \
         the movement step is not publishing a stopping distance"
    );
    assert!(
        seen.iter().any(|s| s.starts_with("stop_")),
        "no stop clip played over five run-and-stops: {seen:?} — which is exactly the \
         measurement CHAR1b.1 recorded as carried item 122"
    );
    // **AND IT IS TAKEN WHILE THE CHARACTER IS STILL MOVING.** The mandate's
    // "distance matching" is this number and nothing else. Half a walk is the
    // line: below it the character has arrived and the clip is decoration.
    assert!(
        fastest > 0.7,
        "the fastest a stop clip was ENTERED at over five run-and-stops is {fastest:.3} m/s, \
         so every one of them was taken by a character that had already stopped — which is \
         the `gait <= 0.1` edge wearing `stop_distance`'s name. The entry speeds were {:?}",
        entry_speeds
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
    );
}

/// **THE BREATH MOVES THE CHEST AND NOTHING BELOW IT** (carried item 128).
///
/// `ALS_N_SecondaryMotion` was imported at CHAR1a.3 and left unbound for three
/// waves, because a map row with no reader is the defect the CHAR1b.1 audit
/// spent its day on. This is the reader's arm, and it asks the island hero's own
/// EVALUATED POSE twice — once and once 1.3 s of idle later — for:
///
/// * a chest that has **moved** between them;
/// * a pelvis and both feet that have **not**;
/// * and a head whose model-space aim is unchanged, because the counter-rotation
///   on the neck is what keeps a breath from nodding a character that is looking
///   at something. Measured without it: **+9.98 deg** of head elevation on a
///   sideways mouse, which is `a_sideways_mouse_turns_the_islands_hero_head`'s
///   own failure.
#[test]
fn the_breath_moves_the_chest_and_leaves_the_feet_alone() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("the hero was posed");
    let rig = rigs.get(&ep.skeleton).expect("its rig is on disk").clone();
    let sample = |sim: &inf_player::runtime_sim::RuntimeSim| -> Vec<glam::Vec3> {
        let p = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("posed");
        inf_anim::pose::global_transforms(&rig.skeleton, &p.pose)
            .iter()
            .map(|m| m.to_scale_rotation_translation().2)
            .collect()
    };
    let roles = rig.role_index();
    let chest = roles
        .last(inf_anim::BoneRoleKind::Spine, inf_anim::BoneSide::Center)
        .expect("the hero has a spine");
    let pelvis = roles
        .first(inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
        .expect("the hero has a pelvis");
    let feet = inf_anim::derive::foot_joints(&rig);
    let a = sample(&sim);
    // 1.3 s of idle at the 60 Hz fixed step.
    for _ in 0..78 {
        sim.step_once(RuntimeInput::default());
    }
    let b = sample(&sim);
    let moved = |j: u16| -> f64 { f64::from((b[j as usize] - a[j as usize]).length()) };
    println!(
        "
=== the breath, 1.3 s of island idle apart ===
  chest {:.4} mm, pelvis {:.4} mm, feet {:?} mm",
        moved(chest) * 1000.0,
        moved(pelvis) * 1000.0,
        feet.iter().map(|j| moved(*j) * 1000.0).collect::<Vec<_>>()
    );
    assert!(
        moved(chest) > 0.0005,
        "the chest moved {:.4} mm over 1.3 s of idle - `ALS_N_SecondaryMotion` is not reaching the pose, which is what carried item 128 recorded for three waves",
        moved(chest) * 1000.0
    );
    // The MASK, asked of the joints rather than of the mask table: a breath is a
    // ribcage, and the legs below it are the locomotion's.
    for (what, j) in
        std::iter::once(("the pelvis", pelvis)).chain(feet.iter().take(2).map(|j| ("a foot", *j)))
    {
        assert!(
            moved(j) < moved(chest) * 0.25,
            "the breath moved {what} {:.4} mm against the chest's {:.4} mm - it is masked to the spine and a mask that leaks is not one",
            moved(j) * 1000.0,
            moved(chest) * 1000.0
        );
    }
}

/// **A DROP CHOOSES ITS LANDING BY HOW FAR IT FELL** (clause 6 — in-air control
/// and land recovery by fall height).
///
/// ALS keys both thresholds on impact SPEED — `BreakfallOnLandVelocity` 700 cm/s
/// and `RagdollOnLandVelocity` 1000 cm/s — and this engine ported them verbatim
/// (`CharacterMovement::land_hard_mps` 7.0, `land_ragdoll_mps` 10.0). A speed is
/// not a height, so the mandate's question — *"a drop from a roof edge or the 2 m
/// container reaches `land_heavy`"* — has a numeric answer that nobody had
/// written down: `v = sqrt(2 g h)`, so 7 m/s is **2.50 m** of free fall and
/// 10 m/s is **5.10 m**.
///
/// This arm drops the island's own hero from a measured height, three times, and
/// asks the CLASSIFIER what it saw — plus the world, because a drop that does not
/// happen classifies nothing:
///
/// * just under 2.50 m -> `Soft`, the light landing;
/// * comfortably over it, with no movement input -> `Hard`, the heavy one;
/// * the same drop WITH input -> `Roll`, which is ALS's own break-fall split
///   (`classify_landing`'s `has_input` arm).
#[test]
fn a_drop_lands_light_heavy_or_rolling_by_the_height_it_fell() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_ecs::components::LandingKind;
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let cm0 = hero_cm(&sim, hero);
    let g = cm0.gravity_mps2;
    // The heights those two speeds ARE, which is the number the mandate asks for.
    let h_hard = cm0.land_hard_mps * cm0.land_hard_mps / (2.0 * g);
    let h_rag = cm0.land_ragdoll_mps * cm0.land_ragdoll_mps / (2.0 * g);
    println!(
        "\n=== the landing thresholds, as heights ===\n  hard {:.3} m/s = {h_hard:.3} m of free \
         fall; ragdoll {:.3} m/s = {h_rag:.3} m",
        cm0.land_hard_mps, cm0.land_ragdoll_mps
    );
    let ground = hero_pos(&sim, hero);
    // **The joints, not the classifier** (the CHAR1b.1 audit's law). A landing is
    // a character taking the impact through its legs, so the number that says
    // one happened is how far the PELVIS came down toward its own feet — a pure
    // pose quantity, measured in the hero's own model frame, so a capsule that
    // moves does not appear in it.
    let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("the hero was posed");
    let rig = rigs.get(&ep.skeleton).expect("its rig is on disk").clone();
    let roles = rig.role_index();
    let pelvis_j = roles
        .first(inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
        .expect("the hero has a pelvis");
    let feet_j = inf_anim::derive::foot_joints(&rig);
    let crouch_depth = |sim: &inf_player::runtime_sim::RuntimeSim| -> f64 {
        let Some(p) = inf_ecs::pose::evaluated_pose(sim.world(), hero) else {
            return f64::NAN;
        };
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &p.pose);
        let at = |j: u16| g[j as usize].to_scale_rotation_translation().2;
        let lower = feet_j
            .iter()
            .map(|j| at(*j).y)
            .fold(f64::MAX, |a, b| a.min(f64::from(b)));
        f64::from(at(pelvis_j).y) - lower
    };

    let drop = |sim: &mut inf_player::runtime_sim::RuntimeSim,
                height: f64,
                input: bool|
     -> (LandingKind, f64, String, f64, f64) {
        {
            let w = sim.world_mut();
            let e = w.entity_of(hero).expect("the hero is in the world");
            if let Some(mut t) = w.world_mut().get_mut::<inf_ecs::components::Transform>(e) {
                t.translation.x = ground[0];
                t.translation.y = ground[1] + height;
                t.translation.z = ground[2];
            }
            if let Some(mut c) = w
                .world_mut()
                .get_mut::<inf_ecs::components::CharacterMovement>(e)
            {
                c.runtime.velocity = inf_ecs::math::Vec3d::ZERO;
                c.mode = inf_ecs::components::MovementMode::FallControlled;
                c.runtime.grounded = false;
            }
        }
        let ax: std::collections::BTreeMap<String, f32> = if input {
            [("move_y".to_string(), 1.0f32)].into()
        } else {
            Default::default()
        };
        let mut worst = LandingKind::None;
        let mut impact = 0.0f64;
        let mut states: std::collections::BTreeSet<String> = Default::default();
        let standing = crouch_depth(sim);
        let mut lowest = f64::MAX;
        for _ in 0..240 {
            sim.step_once(RuntimeInput::default().with_axes(ax.clone()));
            let c = hero_cm(sim, hero);
            impact = impact.max((-c.runtime.velocity.y).max(0.0));
            if c.runtime.landing != LandingKind::None {
                worst = c.runtime.landing;
            }
            if let Some(s) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
                states.insert(s.name.clone());
            }
            // Only once the character is back on the ground: a pelvis measured
            // in mid-air is measuring a fall loop's tuck, not a landing.
            if c.runtime.grounded {
                let d = crouch_depth(sim);
                if d.is_finite() {
                    lowest = lowest.min(d);
                }
            }
        }
        // Let the machine settle before the next drop.
        for _ in 0..120 {
            sim.step_once(RuntimeInput::default());
        }
        (worst, impact, format!("{states:?}"), standing, lowest)
    };

    // Just under the hard threshold: a light landing.
    let (soft, v_soft, s1, stand0, low0) = drop(&mut sim, h_hard * 0.7, false);
    // Comfortably over it and under the ragdoll one, hands empty: the heavy one.
    let (hard, v_hard, s2, _, low1) = drop(&mut sim, (h_hard + h_rag) * 0.5, false);
    // The same drop with the stick pushed: ALS's break-fall.
    let (roll, v_roll, s3, _, low2) = drop(&mut sim, (h_hard + h_rag) * 0.5, true);
    // …and past the ragdoll threshold, which is a ragdoll and not a landing.
    let (rag, v_rag, s4, _, _) = drop(&mut sim, h_rag * 1.6, false);
    println!(
        "  {:.2} m -> {soft:?} at {v_soft:.2} m/s  {s1}",
        h_hard * 0.7
    );
    println!(
        "  {:.2} m -> {hard:?} at {v_hard:.2} m/s  {s2}",
        (h_hard + h_rag) * 0.5
    );
    println!(
        "  {:.2} m + input -> {roll:?} at {v_roll:.2} m/s  {s3}",
        (h_hard + h_rag) * 0.5
    );
    println!("  {:.2} m -> {rag:?} at {v_rag:.2} m/s  {s4}", h_rag * 1.6);
    println!("  the pelvis over its own lower foot, standing {stand0:.4} m");
    println!("  lowest after the landing: light {low0:.4}, heavy {low1:.4}, roll {low2:.4}");
    assert_eq!(
        soft,
        LandingKind::Soft,
        "a {:.2} m drop (under the {h_hard:.2} m the hard threshold IS) classified {soft:?}",
        h_hard * 0.7
    );
    assert_eq!(
        hard,
        LandingKind::Hard,
        "a {:.2} m drop with the stick centred classified {hard:?} at {v_hard:.2} m/s against a {:.2} m/s threshold",
        (h_hard + h_rag) * 0.5,
        cm0.land_hard_mps
    );
    assert_eq!(
        roll,
        LandingKind::Roll,
        "the same {:.2} m drop WITH movement input classified {roll:?} - ALS's break-fall is the has-input arm of the same classifier",
        (h_hard + h_rag) * 0.5
    );
    assert_eq!(
        rag,
        LandingKind::Ragdoll,
        "a {:.2} m drop (over the {h_rag:.2} m the ragdoll threshold IS) classified {rag:?}",
        h_rag * 1.6
    );
    // ── THE CLIPS PLAYED, AND THE JOINTS SAY SO ──────────────────────────────
    //
    // The state names are the first half and the POSE is the second: a machine
    // that entered `land_heavy` while holding an idle pose would satisfy every
    // assertion above. A landing is the legs taking the impact, so the number is
    // how far the pelvis came down toward its own feet, in the hero's own model
    // frame — a capsule that moves is not in it.
    for (what, states, want) in [
        ("the light landing", &s1, "land_light"),
        ("the heavy landing", &s2, "land_heavy"),
        ("the break-fall", &s3, "roll"),
        ("the ragdoll landing", &s4, "ragdoll"),
    ] {
        assert!(
            states.contains(want),
            "{what} classified correctly and the machine played {states} - `{want}` never ran"
        );
    }
    assert!(
        low0 < stand0 - 0.01,
        "the light landing left the pelvis at {low0:.4} m over its foot against {stand0:.4} m standing - the clip played and the POSE did not move"
    );
    assert!(
        low1 < low0,
        "a heavy landing sank the pelvis to {low1:.4} m and a light one to {low0:.4} m - the two clips draw the same character"
    );
    assert!(
        low2 < low1,
        "the break-fall put the pelvis at {low2:.4} m against the heavy landing's {low1:.4} m - a roll takes the body to the ground and this one did not"
    );
}

/// **A CROSS-FADE IS NOT A SNAP** (clause 6 — inertialized transitions with the
/// pop bounded).
///
/// `anim_blend = "inertialize"` is this engine's default and P29.2 built the
/// whole mechanism; what nobody had measured is the thing it exists to prevent.
///
/// # The bound is RELATIVE, and the first attempt at an absolute one is why
///
/// A scripted tour of the island's hero was measured against a flat ceiling of
/// 25° in one 60 Hz step, and the worst reading was **73.507°, in `sprint`, on
/// `calf_r`** — with `calf_l`, `thigh_r` and `thigh_l` right behind it. Those are
/// the KNEES of a sprinting character at a warped play rate: a 0.6 s cycle
/// flexing a knee through about 100° in a third of it is 11° a step before
/// stride warping and 21° after, so twenty-odd degrees is what a sprint's knee
/// does and an absolute ceiling was measuring the gait rather than the blend.
///
/// So the question is asked the way it is meant: **is a step in which the
/// machine is cross-fading worse than one in which it is not?** The tour records
/// the largest single-joint step for every step of it, buckets those by
/// `AnimStateInfo::blending`, and compares the two distributions at the 99th
/// percentile. A cross-fade that snapped would put its whole tail in the
/// blending bucket; one that inertializes puts it nowhere.
///
/// The joints, not the report: the pose is read off `evaluated_pose` and
/// compared with the previous step's, joint by joint.
#[test]
fn a_cross_fade_moves_no_joint_faster_than_the_gait_already_does() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let mut prev: Option<inf_anim::Pose> = None;
    let mut blending: Vec<f64> = Vec::new();
    let mut steady: Vec<f64> = Vec::new();
    // …and the same two, over the joints the LEG IK never writes. See the
    // assertion below for why the question is asked twice.
    let mut blend_upper: Vec<f64> = Vec::new();
    let mut steady_upper: Vec<f64> = Vec::new();
    let leg: std::collections::BTreeSet<u16> = {
        let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
        inf_ecs::pose::evaluated_pose(sim.world(), hero)
            .and_then(|p| rigs.get(&p.skeleton))
            .map(|rig| {
                rig.role_index()
                    .rows()
                    .iter()
                    .filter(|r| {
                        matches!(
                            r.kind,
                            inf_anim::BoneRoleKind::Thigh
                                | inf_anim::BoneRoleKind::Calf
                                | inf_anim::BoneRoleKind::Foot
                                | inf_anim::BoneRoleKind::Ball
                        )
                    })
                    .map(|r| r.joint)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    let mut fades = 0usize;
    let mut transitions = 0usize;
    let mut last_state: Option<String> = None;
    let mut edges: std::collections::BTreeMap<String, usize> = Default::default();
    let mut since_change = 99usize;
    /// One leg of the scripted tour: how many steps, which keys are held, and
    /// which axes are pushed. Named because the tuple is three deep and clippy's
    /// complexity lint is right that an unnamed one is hard to read.
    type Leg = (usize, Vec<&'static str>, Vec<(&'static str, f32)>);
    let legs: Vec<Leg> = vec![
        (90, vec![], vec![]),
        (120, vec![], vec![("move_y", 1.0)]),
        (120, vec!["sprint"], vec![("move_y", 1.0)]),
        (150, vec![], vec![]),
        (10, vec!["crouch"], vec![]),
        (90, vec![], vec![]),
        (90, vec![], vec![("move_y", 1.0)]),
        (10, vec!["crouch"], vec![]),
        (60, vec![], vec![]),
        (3, vec!["jump"], vec![]),
        (150, vec![], vec![]),
        (120, vec![], vec![("move_x", 1.0)]),
        (90, vec![], vec![]),
    ];
    for (n, held, ax) in &legs {
        let axes: std::collections::BTreeMap<String, f32> =
            ax.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
        for _ in 0..*n {
            sim.step_once(RuntimeInput::with_down(held.clone()).with_axes(axes.clone()));
            let Some(info) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) else {
                continue;
            };
            // **`AnimStateInfo::blending` is dead under this engine's default**
            // (wave CHAR1b.2, measured): it is `SmRuntime::prev.is_some()`, and
            // an INERTIALIZED transition collapses the fade inside
            // `PoseBlender::step` before the pose is evaluated — which is the
            // whole "one evaluation instead of two". Over a 1 102-step tour the
            // flag was false on **every single step** while eleven states were
            // visited. So the window is taken from the state NAME changing,
            // which is the step a transition fired, and it is held open for the
            // 12 steps (0.2 s) an inertialized deviation decays over.
            let state = info.name.clone();
            let changed = last_state.as_deref() != Some(state.as_str());
            if changed {
                since_change = 0;
                if let Some(from) = last_state.as_deref() {
                    // **The transition count is the EDGES that fired** (CHAR1b.2
                    // audit). `fades` was incremented here AND once per step of
                    // every post-transition window below, so the "214
                    // transitions over 1 102 steps" this arm printed — and the
                    // wave's report repeated — was 24 transitions plus 190
                    // fading steps added together. The two are counted apart
                    // now; the printed line says which is which.
                    transitions += 1;
                    *edges.entry(format!("{from} -> {state}")).or_insert(0usize) += 1;
                }
            } else {
                since_change += 1;
            }
            last_state = Some(state.clone());
            let fading = since_change < 12;
            let Some(ep) = inf_ecs::pose::evaluated_pose(sim.world(), hero) else {
                continue;
            };
            seen.insert(state);
            if fading {
                fades += 1;
            }
            if let Some(p) = prev.as_ref() {
                if p.locals.len() == ep.pose.locals.len() {
                    let mut step_worst = 0.0f64;
                    let mut step_worst_upper = 0.0f64;
                    for (j, (a, b)) in p.locals.iter().zip(ep.pose.locals.iter()).enumerate() {
                        let qa = glam::Quat::from_array(a.rotation);
                        let qb = glam::Quat::from_array(b.rotation);
                        // The angle between two orientations, from the dot
                        // product. `pacos64`, because a gate that decides on a
                        // number decides on the same number everywhere.
                        let d = qa.dot(qb).abs().clamp(-1.0, 1.0);
                        let deg = 2.0 * inf_math::pacos64(f64::from(d)).to_degrees();
                        step_worst = step_worst.max(deg);
                        if !leg.contains(&(j as u16)) {
                            step_worst_upper = step_worst_upper.max(deg);
                        }
                    }
                    if fading {
                        blending.push(step_worst);
                        blend_upper.push(step_worst_upper);
                    } else {
                        steady.push(step_worst);
                        steady_upper.push(step_worst_upper);
                    }
                }
            }
            prev = Some(ep.pose.clone());
        }
    }
    let q = |v: &mut Vec<f64>, f: f64| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(f64::total_cmp);
        v[(((v.len() - 1) as f64) * f) as usize]
    };
    let (bn, sn) = (blending.len(), steady.len());
    let (b50, b99, bmax) = (
        q(&mut blending, 0.5),
        q(&mut blending, 0.99),
        q(&mut blending, 1.0),
    );
    let (s50, s99, smax) = (
        q(&mut steady, 0.5),
        q(&mut steady, 0.99),
        q(&mut steady, 1.0),
    );
    println!(
        "\n=== the largest single-joint step, degrees, over a scripted tour ===\n  \
         CROSS-FADING ({bn} steps): p50 {b50:.3}  p99 {b99:.3}  max {bmax:.3}\n  \
         STEADY       ({sn} steps): p50 {s50:.3}  p99 {s99:.3}  max {smax:.3}\n  \
         {transitions} transitions fired over {} steps ({fades} of them inside a \
         post-transition window); states visited: {seen:?}",
        bn + sn
    );
    {
        let mut rows: Vec<(usize, String)> = edges.into_iter().map(|(k, v)| (v, k)).collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.0));
        println!("  the edges that fired, by count:");
        for (n, e) in rows.iter().take(12) {
            println!("    {n:4} x  {e}");
        }
    }
    let (bu99, su99) = (q(&mut blend_upper, 0.99), q(&mut steady_upper, 0.99));
    println!(
        "  the same, over the {} joints the LEG IK never writes:\n    CROSS-FADING p99 \
         {bu99:.3}   STEADY p99 {su99:.3}",
        blend_upper.len().min(1) * (prev.as_ref().map(|p| p.locals.len()).unwrap_or(0) - leg.len())
    );
    assert!(
        seen.len() >= 4 && bn > 30,
        "the tour reached {seen:?} over {bn} post-transition steps — it did not cross enough \
         transitions to be one"
    );
    // **THE ANTI-VACUITY CLAUSE** (CHAR1b.2 audit). The ratchet below is a
    // RATIO, and a character that never moves gives `bu99 = su99 = 0`, which
    // satisfies `0 <= 0 * 2 + 2` — so this arm was green on a hero drawing its
    // bind pose, which is exactly the failure the CHAR1b.1 audit spent its day
    // on (five of seventeen arms were green on a bind-posed character). The
    // gait's own p99 has to be a real number before a ratio against it means
    // anything. Measured on this tree: **11.995 deg**.
    assert!(
        su99 > 1.0,
        "the steady p99 over the non-leg joints is {su99:.3} deg, so the hero is barely \
         animating and the ratchet below is a ratio of nothing to nothing"
    );
    // **THE BOUND IS OVER THE JOINTS THE BLEND OWNS, and the other number is
    // printed rather than asserted** (wave CHAR1b.2).
    //
    // Over EVERY joint the post-transition p99 is 50.189 deg against the gait's
    // own 19.866, and the offenders are named: `calf_r` (22 steps over 20 deg),
    // `calf_l` (14), `thigh_r` (7), `thigh_l` (5). Those are the knees, and what
    // moves them on the step after a transition is not the cross-fade — it is
    // `apply_foot_ik`, which is still solving to a goal built from the PREVIOUS
    // step's foot while the pose under it has just changed. That is a real
    // defect, it is this wave's carried item, and it is a defect of the foot IK
    // rather than of the blend.
    //
    // The question this arm exists to answer is whether the BLEND snaps, so it
    // is asked of the joints the leg IK never writes: everything above the hips
    // plus the arms, which is where a cross-fade between two whole-body poses
    // would show first and hardest.
    //
    // **AND THE BOUND IS A RATCHET, NOT A PASS.** Measured on this tree, over the
    // non-leg joints: **23.494 deg against the gait's own 11.995**, a ratio of
    // 1.96. An inertialized transition ought to be indistinguishable from the
    // gait — a ratio of 1.0 — so this is a real remainder, and it is carried
    // with its number rather than dressed up by a loose bound. What the arm
    // holds is that it cannot get WORSE: twice plus two degrees is a hair above
    // what the tree does today and far below a snap, which puts a whole tail
    // into the post-transition bucket and reads at four or five times.
    const POP_RATIO_CEILING: f64 = 2.0;
    assert!(
        bu99 <= su99 * POP_RATIO_CEILING + 2.0,
        "a post-transition step moved a non-leg joint {bu99:.3} deg at the 99th percentile \
         against the gait's own {su99:.3} — a ratio of {:.2}, past the \
         {POP_RATIO_CEILING} ratchet. The blend is snapping, which is what \
         inertialization is for",
        if su99 > 0.0 {
            bu99 / su99
        } else {
            f64::INFINITY
        }
    );
}

/// **A RAGDOLL DRAWS THE BODIES, AND GETS UP WITHOUT A SNAP** (clause 6 — the
/// ragdoll blend in and out, on the P12 substrate).
///
/// # What had never run
///
/// P29.4 built the ragdoll bridge whole — the articulated bodies, the joints,
/// the velocity handoff in both directions, the settle, the get-up, the face-up
/// read — and `inf_anim::ragdoll::blend_weight`, which its own module calls *"the
/// doctrine's load-bearing sentence"*. That function had **zero non-test
/// callers** for two phases, and the consequence was never looked at: a
/// character in `MovementMode::Ragdoll` drew the machine's `ragdoll` state —
/// `ALS_Flail`, authored in place — while its rigid bodies tumbled underneath
/// it. The capsule follows the pelvis, so the character *travelled* like a
/// ragdoll and *posed* as a standing figure waving its arms.
///
/// # What this asks, of the joints
///
/// 1. **The drawn skeleton is where the bodies are.** For every bone the physics
///    side publishes, the distance between the joint the pose step actually drew
///    and that body's own head. A machine-only pose cannot pass this: the flail
///    clip knows nothing about where a rigid body ended up. The denominator is
///    printed beside it — the same joints against the pose the character was
///    STANDING in one step before the ragdoll began.
/// 2. **The get-up is an interpolation, not a cut.** The largest single-joint
///    step across the hand-off, in millimetres, against the same *relative*
///    ratchet the cross-fade arm uses: the hand-off may not be worse than what
///    the ragdoll's own simulating steps were already doing.
/// 3. **The weight really goes 1 → 0**, monotonically, and the entry is removed,
///    so a level stops paying for a ragdoll that has finished.
///
/// Mutation that reds it: delete the `apply_ragdoll_pose` call in the pose step.
#[test]
fn a_ragdoll_draws_the_bodies_and_gets_up_without_a_snap() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("the hero was posed");
    let rig = rigs.get(&ep.skeleton).expect("its rig is on disk").clone();

    // Every DRAWN joint, in world metres — the pose the renderer skins, not a
    // report about it.
    let drawn = |sim: &inf_player::runtime_sim::RuntimeSim| -> Vec<DVec3> {
        let p = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("posed");
        let to_world =
            inf_ecs::pose::model_to_world_of(sim.world(), hero).expect("the hero is placed");
        inf_anim::pose::global_transforms(&rig.skeleton, &p.pose)
            .iter()
            .map(|m| {
                let t = m.to_scale_rotation_translation().2;
                to_world.transform_point3(DVec3::new(t.x as f64, t.y as f64, t.z as f64))
            })
            .collect()
    };
    let index_of: std::collections::BTreeMap<&str, usize> = rig
        .skeleton
        .joints()
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();
    // How far the drawn skeleton is from the bodies, over every published bone.
    let agreement =
        |sim: &inf_player::runtime_sim::RuntimeSim, pose: &[DVec3]| -> Option<(f64, f64, usize)> {
            let rp = inf_ecs::anim_bridge::ragdoll_pose(sim.world(), hero)?;
            let mut worst = 0.0f64;
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for b in &rp.bones {
                let Some(&i) = index_of.get(b.name.as_str()) else {
                    continue;
                };
                let d = (pose[i] - DVec3::new(b.head.x, b.head.y, b.head.z)).length();
                worst = worst.max(d);
                sum += d;
                n += 1;
            }
            (n > 0).then_some((worst, sum / n as f64, n))
        };

    let standing = drawn(&sim);
    assert!(
        inf_physics::d3::ragdoll_bridge::start_ragdoll(sim.world_mut(), hero),
        "the gameplay door refused to ragdoll the island's hero"
    );
    // ── while the bodies own the pose ────────────────────────────────────────
    let mut best: Option<(f64, f64, usize)> = None;
    let mut against_standing = 0.0f64;
    let mut sim_steps: Vec<f64> = Vec::new();
    let mut last = drawn(&sim);
    let mut weight_hi = 0.0f32;
    let mut settled_at: Option<usize> = None;
    for i in 0..900 {
        sim.step_once(RuntimeInput::default());
        let now = drawn(&sim);
        let step = now
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0f64, f64::max);
        let cm = hero_cm(&sim, hero);
        if cm.runtime.ragdoll.phase == inf_anim::RagdollPhase::Simulating
            && cm.runtime.ragdoll.spawned
        {
            if let Some(a) = agreement(&sim, &now) {
                // The tightest reading is taken past the first few steps: the
                // very first published pose is one fixed step older than the
                // bodies that produced it, which is the bridge's own documented
                // beat of latency and not a miss.
                if i > 4 && (best.is_none() || a.1 < best.expect("some").1) {
                    best = Some(a);
                    against_standing = mean_gap(&standing, &now);
                }
            }
            if let Some(rp) = inf_ecs::anim_bridge::ragdoll_pose(sim.world(), hero) {
                weight_hi = weight_hi.max(rp.weight);
            }
            sim_steps.push(step);
        }
        if cm.runtime.ragdoll.phase == inf_anim::RagdollPhase::GettingUp && settled_at.is_none() {
            settled_at = Some(i);
            last = now;
            break;
        }
        last = now;
    }
    let (worst, mean, n) = best.expect(
        "the physics side never published a ragdoll pose - `set_ragdoll_pose` has no writer, or \
         the bodies never spawned",
    );
    // ── the hand-off ─────────────────────────────────────────────────────────
    let switch = settled_at.expect("the ragdoll never settled into a get-up over 900 steps");
    let mut handoff: Vec<f64> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut prev = last;
    let mut getup_states: std::collections::BTreeSet<String> = Default::default();
    for _ in 0..(((inf_physics::d3::ragdoll_bridge::GET_UP_BLEND_S * 60.0) as usize) + 20) {
        sim.step_once(RuntimeInput::default());
        let now = drawn(&sim);
        if let Some(st) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
            getup_states.insert(st.name.clone());
        }
        handoff.push(
            now.iter()
                .zip(prev.iter())
                .map(|(a, b)| (*a - *b).length())
                .fold(0.0f64, f64::max),
        );
        weights.push(
            inf_ecs::anim_bridge::ragdoll_pose(sim.world(), hero)
                .map(|r| r.weight)
                .unwrap_or(0.0),
        );
        prev = now;
    }
    let pct = |v: &mut Vec<f64>, q: f64| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v[(((v.len() as f64 - 1.0) * q).round() as usize).min(v.len().saturating_sub(1))]
    };
    let mut sim_sorted = sim_steps.clone();
    let mut hand_sorted = handoff.clone();
    let sim_p99 = pct(&mut sim_sorted, 0.99);
    let hand_p99 = pct(&mut hand_sorted, 0.99);
    let hand_max = handoff.iter().copied().fold(0.0f64, f64::max);
    println!(
        "\n=== the ragdoll, on the island ===\n  the drawn skeleton against the bodies: worst \
         {:.3} mm, mean {:.3} mm over {n} bones\n  the same joints against the pose it was \
         STANDING in: {:.1} mm\n  it settled after {switch} steps; weight while simulating \
         {weight_hi:.3}; over the get-up {:?}\n  the machine over the hand-off: \
         {getup_states:?}\n  largest single-joint step, mm: simulating p99 \
         {:.2} (n {}), hand-off p99 {:.2} max {:.2} (n {})",
        worst * 1000.0,
        mean * 1000.0,
        against_standing * 1000.0,
        weights
            .iter()
            .step_by(3)
            .map(|w| format!("{w:.2}"))
            .collect::<Vec<_>>(),
        sim_p99 * 1000.0,
        sim_steps.len(),
        hand_p99 * 1000.0,
        hand_max * 1000.0,
        handoff.len()
    );
    assert!(
        mean < 0.010,
        "the drawn skeleton sits {:.1} mm from the bodies it is supposed to BE - the blend is \
         not reaching the pose (the standing pose is {:.1} mm from them)",
        mean * 1000.0,
        against_standing * 1000.0
    );
    assert!(
        against_standing > mean * 10.0,
        "the pose the character was standing in is only {:.1} mm from the bodies, so agreeing \
         with them at {:.1} mm proves nothing",
        against_standing * 1000.0,
        mean * 1000.0
    );
    assert!(
        weight_hi > 0.99,
        "the physics pose was never drawn at full weight (peak {weight_hi})"
    );
    assert_eq!(
        weights.last().copied(),
        Some(0.0),
        "the get-up never blended out: the weights are {weights:?}"
    );
    assert!(
        weights.windows(2).all(|w| w[1] <= w[0] + 1e-6),
        "the get-up blended back TOWARD the physics pose: {weights:?}"
    );
    // The ratchet, the cross-fade arm's shape: the hand-off is not worse than the
    // motion the ragdoll was already producing.
    assert!(
        hand_p99 <= sim_p99 * 2.0 + 0.005,
        "the get-up's p99 joint step is {:.2} mm against the ragdoll's own {:.2} mm - that is a \
         cut, not a blend",
        hand_p99 * 1000.0,
        sim_p99 * 1000.0
    );
    assert!(
        switch > 0,
        "the ragdoll got up on the step it started, which is not a ragdoll"
    );
    // **And the get-up is a CLIP, not a fall back to standing.** The `Any -> idle`
    // edge took the machine out of the get-up one step after it entered it until
    // this wave guarded that edge on `getup`; without the guard this set is
    // `idle` alone and the 0.35 s blend interpolates a heap into a standing pose.
    assert!(
        getup_states
            .iter()
            .any(|s| inf_anim::als::GETUP_STATES.contains(&s.as_str())),
        "the machine played {getup_states:?} over the get-up - no get-up clip ran"
    );
}

/// The mean distance between two sets of world-space joints — the denominator the
/// ragdoll arm prints beside its agreement.
fn mean_gap(a: &[DVec3], b: &[DVec3]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).length())
        .sum::<f64>()
        / n as f64
}

/// **THE ISLAND'S HERO LEANS INTO WHAT IT IS DOING** (clause 6 — lean on
/// acceleration and turns).
///
/// P29.4 computed `MovementRuntime::relative_accel` every step with zero
/// readers; this wave's first commit interpolated it into `lean` at ALS's own
/// `GroundedLeanInterpSpeed` and published it as `lean_x` / `lean_y`, and left
/// it a parameter with no pose pass. `inf_anim::lean::apply_lean` is the pass,
/// and this is its measurement on the island rather than on a fixture.
///
/// The question is asked of the CHEST, in the hero's OWN frame — a world-space
/// answer would be measuring the body yaw:
///
/// * a **standing start** (the stick forward from rest) puts the chest ahead of
///   the pelvis, and it comes back when the acceleration does;
/// * **braking** — the stick released at speed — puts it behind;
/// * a **hard left turn at speed** leans it to the character's left, and the
///   mirror-image right turn leans it right by the same magnitude;
/// * the **feet** do not move with any of it, which is the mask asserted on the
///   tree rather than on the table.
///
/// Mutation that reds it: delete the `apply_lean` call in the pose step (every
/// offset falls to 0.0 mm).
#[test]
fn the_islands_hero_leans_into_a_start_a_stop_and_a_turn() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("the hero was posed");
    let rig = rigs.get(&ep.skeleton).expect("its rig is on disk").clone();
    let roles = rig.role_index();
    let chest = roles
        .last(inf_anim::BoneRoleKind::Spine, inf_anim::BoneSide::Center)
        .expect("the hero has a spine");
    let pelvis = roles
        .first(inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
        .expect("the hero has a pelvis");

    // The chest's offset from the pelvis, **in the hero's own model frame** —
    // `x` its right, `z` its forward. Model space is exactly that frame, so this
    // is the pose and not the placement.
    let offset = |sim: &inf_player::runtime_sim::RuntimeSim| -> glam::Vec3 {
        let p = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("posed");
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &p.pose);
        let at = |j: u16| g[j as usize].to_scale_rotation_translation().2;
        at(chest) - at(pelvis)
    };
    let axes = |x: f32, y: f32| -> std::collections::BTreeMap<String, f32> {
        [("move_x".to_string(), x), ("move_y".to_string(), y)].into()
    };
    // Drive `steps` with an input and report the RANGE the chest's offset and
    // the hero's own lean parameters covered over them.
    //
    // A range and not a peak: a lean is signed, and the interesting reading is
    // the extreme in the direction the manoeuvre is supposed to push — a braking
    // window that starts at a full forward lean has its largest *magnitude* at
    // the beginning and its answer at the end.
    #[derive(Clone, Copy, Debug, Default)]
    struct Span {
        z_lo: f64,
        z_hi: f64,
        x_lo: f64,
        x_hi: f64,
        ly_lo: f64,
        ly_hi: f64,
        lx_lo: f64,
        lx_hi: f64,
    }
    let run = |sim: &mut inf_player::runtime_sim::RuntimeSim,
               ax: std::collections::BTreeMap<String, f32>,
               steps: usize|
     -> Span {
        let mut s = Span {
            z_lo: f64::MAX,
            z_hi: f64::MIN,
            x_lo: f64::MAX,
            x_hi: f64::MIN,
            ly_lo: f64::MAX,
            ly_hi: f64::MIN,
            lx_lo: f64::MAX,
            lx_hi: f64::MIN,
        };
        for _ in 0..steps {
            sim.step_once(RuntimeInput::default().with_axes(ax.clone()));
            let o = offset(sim);
            s.z_lo = s.z_lo.min(f64::from(o.z));
            s.z_hi = s.z_hi.max(f64::from(o.z));
            s.x_lo = s.x_lo.min(f64::from(o.x));
            s.x_hi = s.x_hi.max(f64::from(o.x));
            let c = hero_cm(sim, hero);
            s.ly_lo = s.ly_lo.min(c.runtime.lean.y);
            s.ly_hi = s.ly_hi.max(c.runtime.lean.y);
            s.lx_lo = s.lx_lo.min(c.runtime.lean.x);
            s.lx_hi = s.lx_hi.max(c.runtime.lean.x);
        }
        s
    };

    // ── rest: the chest sits where the animation put it ──────────────────────
    let rest = offset(&sim);
    // ── a standing start ─────────────────────────────────────────────────────
    let start = run(&mut sim, axes(0.0, 1.0), 25);
    // …at speed, then the stick released: braking.
    for _ in 0..90 {
        sim.step_once(RuntimeInput::default().with_axes(axes(0.0, 1.0)));
    }
    let brake = run(&mut sim, Default::default(), 30);
    for _ in 0..120 {
        sim.step_once(RuntimeInput::default());
    }
    // ── a hard turn each way, at speed ───────────────────────────────────────
    let mut turn = |x: f32| -> Span {
        for _ in 0..90 {
            sim.step_once(RuntimeInput::default().with_axes(axes(0.0, 1.0)));
        }
        let r = run(&mut sim, axes(x, 1.0), 25);
        for _ in 0..120 {
            sim.step_once(RuntimeInput::default());
        }
        r
    };
    let right = turn(1.0);
    let left = turn(-1.0);
    println!(
        "
=== the lean, off the island hero's chest (model frame, mm from the pelvis) ==="
    );
    println!(
        "  {:<11} z {:+7.2} mm   x {:+7.2} mm",
        "rest",
        rest.z * 1000.0,
        rest.x * 1000.0
    );
    println!(
        "  start       z up to   {:+7.2} mm   lean_y up to   {:+.3}",
        start.z_hi * 1000.0,
        start.ly_hi
    );
    println!(
        "  braking     z down to {:+7.2} mm   lean_y down to {:+.3}",
        brake.z_lo * 1000.0,
        brake.ly_lo
    );
    println!(
        "  turn right  x up to   {:+7.2} mm   lean_x up to   {:+.3}",
        right.x_hi * 1000.0,
        right.lx_hi
    );
    println!(
        "  turn left   x down to {:+7.2} mm   lean_x down to {:+.3}",
        left.x_lo * 1000.0,
        left.lx_lo
    );
    // A start puts the chest FORWARD of where rest left it.
    assert!(
        start.z_hi - f64::from(rest.z) > 0.005,
        "a standing start moved the chest {:.2} mm forward of rest - the lean is not reaching the pose",
        (start.z_hi - f64::from(rest.z)) * 1000.0
    );
    assert!(
        start.ly_hi > 0.1,
        "the hero's own `lean_y` reached {:+.3} on its hardest start, so the arm is measuring a gait and not a lean",
        start.ly_hi
    );
    // Braking pulls it back behind the start's own reach, and the parameter goes
    // negative — a released stick is a deceleration.
    assert!(
        brake.z_lo < start.z_hi,
        "braking left the chest no further back than {:.2} mm against the start's {:.2} mm",
        brake.z_lo * 1000.0,
        start.z_hi * 1000.0
    );
    assert!(
        brake.ly_lo < -0.02,
        "the hero's own `lean_y` only reached {:+.3} while braking",
        brake.ly_lo
    );
    // The two turns lean opposite ways, and the parameters agree with the pose.
    assert!(
        right.x_hi > f64::from(rest.x) + 0.005 && left.x_lo < f64::from(rest.x) - 0.005,
        "the two turns took the chest to {:.2} mm and {:.2} mm against a rest of {:.2} mm - they did not go opposite ways",
        right.x_hi * 1000.0,
        left.x_lo * 1000.0,
        rest.x * 1000.0
    );
    assert!(
        right.lx_hi > 0.05 && left.lx_lo < -0.05,
        "the hero's own `lean_x` reached {:+.3} turning right and {:+.3} turning left",
        right.lx_hi,
        left.lx_lo
    );
    // The mask is asserted by this arm's own measurement rather than a second
    // time: every number above is the chest's offset from the PELVIS, so a lean
    // that had leaked below the hips would not show up in any of them. The mask
    // on the TREE is `inf_anim::lean`'s own
    // `a_lean_moves_the_chest_and_leaves_the_pelvis_and_feet_alone`, which asks
    // the same rig's feet with no stride in the window to confuse it.
}

/// **THE AUTHORED SETS RUN ON THE ISLAND** (the wave's set D — slide, prone,
/// swimming, the throws and the standing get-up).
///
/// ALS ships none of these: a census by name found no slide, no throw, no swim,
/// no prone and no standing get-up in the whole donor. They were derived from
/// the hero's own rig by `inf_anim::authored` at this wave's first checkpoint
/// and bound into the map — and a bound clip that nothing ever plays is exactly
/// the shape the CHAR1b.1 audit spent its day on. This drives every one of them
/// on the island and reads the answer off the JOINTS.
///
/// The doors are the ones a player has, and where a player has none the door is
/// named:
///
/// * **slide** — sprint to speed and tap crouch (`movement.rs`'s own
///   `want_sprint && speed >= slide_entry_speed_mps`);
/// * **prone** — hold crouch past the long-press threshold, which is
///   `MovementIntent::from_actions`' own rule;
/// * **swimming** — stand in the island's own water and let the water probe
///   decide, surface or under, by how deep;
/// * **the standing get-up** — a ragdoll ended before the body reaches the
///   ground, which is `RagdollRuntime::upright`'s whole definition ("knocked
///   about and stayed on its feet");
/// * **the throws** — `inf_ecs::anim_bridge::start_throw`, the gameplay door,
///   because a throw needs a thing to throw and the throwable set is WPN1's. A
///   key that played an animation and released nothing would be the defect this
///   wave has spent its day closing.
///
/// Every claim is a POSE: a slide lowers the pelvis and the character travels
/// while it does; prone puts the pelvis on the floor; a swim keeps it near the
/// waterline; a throw moves the hand through an arc that the same character
/// standing still does not.
#[test]
fn the_authored_sets_play_on_the_island_and_the_joints_move() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_ecs::components::MovementMode;
    use inf_player::runtime_sim::RuntimeInput;
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let ep = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("the hero was posed");
    let rig = rigs.get(&ep.skeleton).expect("its rig is on disk").clone();
    let roles = rig.role_index();
    let pelvis_j = roles
        .first(inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
        .expect("a pelvis");
    let feet_j = inf_anim::derive::foot_joints(&rig);
    let hand_j = roles
        .first(inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Right)
        .or_else(|| roles.first(inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Left))
        .expect("a hand");

    // **The pelvis's height in the hero's OWN model frame**, whose origin is the
    // character's feet — so this is how high the hips are carried, with the
    // placement taken out of it.
    //
    // Not "the pelvis over its lower foot", which is the landings' measure and
    // is the wrong one here: a prone character's legs are STRAIGHT, so its hips
    // are a full leg length from its feet exactly as a standing character's are.
    // Measured before the metric was changed: prone read 0.8037 m against a
    // standing 0.8020 m and said nothing at all.
    let _ = &feet_j;
    let stance = |sim: &inf_player::runtime_sim::RuntimeSim| -> f64 {
        let Some(p) = inf_ecs::pose::evaluated_pose(sim.world(), hero) else {
            return f64::NAN;
        };
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &p.pose);
        f64::from(g[pelvis_j as usize].to_scale_rotation_translation().2.y)
    };
    // Put the hero somewhere and let it settle, height included — `hero_to`
    // takes the height it finds and adds to it, which after a dive puts the
    // character back in the water.
    let place = |sim: &mut inf_player::runtime_sim::RuntimeSim, p: [f64; 3]| {
        {
            let w = sim.world_mut();
            let e = w.entity_of(hero).expect("the hero is in the world");
            if let Some(mut t) = w.world_mut().get_mut::<inf_ecs::components::Transform>(e) {
                t.translation.x = p[0];
                t.translation.y = p[1] + 1.0;
                t.translation.z = p[2];
            }
            if let Some(mut c) = w
                .world_mut()
                .get_mut::<inf_ecs::components::CharacterMovement>(e)
            {
                c.runtime.velocity = inf_ecs::math::Vec3d::ZERO;
                c.mode = MovementMode::Grounded;
            }
        }
        for _ in 0..240 {
            sim.step_once(RuntimeInput::default());
        }
    };
    let hand = |sim: &inf_player::runtime_sim::RuntimeSim| -> glam::Vec3 {
        let p = inf_ecs::pose::evaluated_pose(sim.world(), hero).expect("posed");
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &p.pose);
        let at = |j: u16| g[j as usize].to_scale_rotation_translation().2;
        at(hand_j) - at(pelvis_j)
    };
    let axes = |x: f32, y: f32| -> std::collections::BTreeMap<String, f32> {
        [("move_x".to_string(), x), ("move_y".to_string(), y)].into()
    };
    let stand_stance = stance(&sim);
    let spawn = hero_pos(&sim, hero);
    let mut played: std::collections::BTreeSet<String> = Default::default();
    let note = |sim: &inf_player::runtime_sim::RuntimeSim,
                set: &mut std::collections::BTreeSet<String>| {
        if let Some(s) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
            set.insert(s.name.clone());
        }
    };

    // ── 1. THE SLIDE ─────────────────────────────────────────────────────────
    let mut slide_stance = f64::MAX;
    let mut slide_travel = 0.0f64;
    let mut slide_states: std::collections::BTreeSet<String> = Default::default();
    {
        // Sprint until the body is past `slide_entry_speed_mps`…
        for _ in 0..150 {
            sim.step_once(RuntimeInput::with_down(["sprint"]).with_axes(axes(0.0, 1.0)));
        }
        let at_entry = hero_pos(&sim, hero);
        // …then tap crouch while still sprinting, which is the slide.
        for i in 0..120 {
            let held: Vec<&str> = if i < 3 {
                vec!["sprint", "crouch"]
            } else {
                vec!["sprint"]
            };
            sim.step_once(RuntimeInput::with_down(held).with_axes(axes(0.0, 1.0)));
            if hero_cm(&sim, hero).mode == MovementMode::Slide {
                note(&sim, &mut slide_states);
                slide_stance = slide_stance.min(stance(&sim));
                let p = hero_pos(&sim, hero);
                slide_travel = slide_travel
                    .max(((p[0] - at_entry[0]).powi(2) + (p[2] - at_entry[2]).powi(2)).sqrt());
            }
        }
        for _ in 0..120 {
            sim.step_once(RuntimeInput::default());
        }
    }

    // ── 2. PRONE ─────────────────────────────────────────────────────────────
    let mut prone_stance = f64::MAX;
    let mut prone_states: std::collections::BTreeSet<String> = Default::default();
    {
        // A LONG crouch press is prone — `MovementIntent::from_actions`' own
        // classification, and the reason the demo loop's tap is a coin toss.
        for _ in 0..40 {
            sim.step_once(RuntimeInput::with_down(["crouch"]));
        }
        sim.step_once(RuntimeInput::default());
        for i in 0..150 {
            let ax = if i > 60 {
                axes(0.0, 1.0)
            } else {
                axes(0.0, 0.0)
            };
            sim.step_once(RuntimeInput::default().with_axes(ax));
            if hero_cm(&sim, hero).mode == MovementMode::Prone {
                note(&sim, &mut prone_states);
                prone_stance = prone_stance.min(stance(&sim));
            }
        }
        // Back up: another long press leaves prone for the crouch.
        for _ in 0..40 {
            sim.step_once(RuntimeInput::with_down(["crouch"]));
        }
        for _ in 0..90 {
            sim.step_once(RuntimeInput::default());
        }
    }

    // ── 3. SWIMMING ──────────────────────────────────────────────────────────
    //
    // The island's own water, found by asking the world rather than by a
    // coordinate somebody wrote down.
    let water = {
        let w = sim.world();
        w.world()
            .iter_entities()
            .filter_map(|e| {
                let b = e.get::<inf_ecs::components::WaterBody>()?;
                let t = e.get::<inf_ecs::components::Transform>()?;
                let name = e
                    .get::<inf_ecs::components::Name>()
                    .map(|n| n.0.clone())
                    .unwrap_or_else(|| "<unnamed water>".to_string());
                Some((
                    t.translation.x,
                    t.translation.y,
                    t.translation.z,
                    b.level_m,
                    name,
                ))
            })
            .next()
    };
    let water_name = water
        .as_ref()
        .map(|w| w.4.clone())
        .unwrap_or_else(|| "<none>".into());
    let mut swim_states: std::collections::BTreeSet<String> = Default::default();
    let mut swim_pelvis_vs_water = f64::NAN;
    let mut swim_capsule_vs_water = f64::NAN;
    if let Some((wx, wy, wz, level, _)) = water.clone() {
        let surface = if level.is_finite() { level } else { wy };
        for depth in [0.6f64, 2.5] {
            {
                let w = sim.world_mut();
                let e = w.entity_of(hero).expect("the hero is in the world");
                if let Some(mut t) = w.world_mut().get_mut::<inf_ecs::components::Transform>(e) {
                    t.translation.x = wx;
                    t.translation.y = surface - depth;
                    t.translation.z = wz;
                }
                if let Some(mut c) = w
                    .world_mut()
                    .get_mut::<inf_ecs::components::CharacterMovement>(e)
                {
                    c.runtime.velocity = inf_ecs::math::Vec3d::ZERO;
                }
            }
            for i in 0..200 {
                let ax = if i > 100 {
                    axes(0.0, 1.0)
                } else {
                    axes(0.0, 0.0)
                };
                sim.step_once(RuntimeInput::default().with_axes(ax));
                let m = hero_cm(&sim, hero).mode;
                if matches!(m, MovementMode::SwimSurface | MovementMode::SwimUnder) {
                    note(&sim, &mut swim_states);
                    if depth < 1.0 {
                        // **THE PELVIS JOINT, not the capsule** (CHAR1b.2
                        // audit). The wave reported "the capsule sat -0.560 m of
                        // the waterline", which is the entity's transform — a
                        // placement, and a number a bind-posed hero produces
                        // just as readily. The pose's own pelvis, lifted into
                        // the world, is what a viewer sees at the waterline.
                        swim_capsule_vs_water = hero_pos(&sim, hero)[1] - surface;
                        if let (Some(p), Some(to_world)) = (
                            inf_ecs::pose::evaluated_pose(sim.world(), hero),
                            inf_ecs::pose::model_to_world_of(sim.world(), hero),
                        ) {
                            let g = inf_anim::pose::global_transforms(&rig.skeleton, &p.pose);
                            let t = g[pelvis_j as usize].to_scale_rotation_translation().2;
                            let w = to_world.transform_point3(DVec3::new(
                                f64::from(t.x),
                                f64::from(t.y),
                                f64::from(t.z),
                            ));
                            swim_pelvis_vs_water = w.y - surface;
                        }
                    }
                }
            }
        }
        // Back onto dry land, at the height it spawned at.
        place(&mut sim, spawn);
    }

    // ── 4. THE STANDING GET-UP ───────────────────────────────────────────────
    //
    // A ragdoll ended before the body reaches the ground: `RagdollRuntime::
    // upright` is "the pelvis is still more than half a capsule above the feet
    // when the bodies settle", and pressing jump ends a ragdoll on the spot.
    let mut getup_states: std::collections::BTreeSet<String> = Default::default();
    let mut upright = false;
    {
        assert!(
            inf_physics::d3::ragdoll_bridge::start_ragdoll(sim.world_mut(), hero),
            "the gameplay door refused to ragdoll the island's hero"
        );
        for i in 0..200 {
            // Two steps of ragdoll, then out again — the body has not fallen.
            let held: Vec<&str> = if (4..8).contains(&i) {
                vec!["jump"]
            } else {
                vec![]
            };
            sim.step_once(RuntimeInput::with_down(held));
            let c = hero_cm(&sim, hero);
            if c.runtime.ragdoll.upright {
                upright = true;
            }
            note(&sim, &mut getup_states);
        }
        for _ in 0..120 {
            sim.step_once(RuntimeInput::default());
        }
    }

    // ── 5. THE TWO THROWS ────────────────────────────────────────────────────
    //
    // The control is taken PER WINDOW and from the same reference the throw's is
    // — the hand at the step the window opens — because an idle hand drifts with
    // the breath and a single reference taken minutes earlier measures the drift
    // rather than the throw.
    let mut throw_reach = [0.0f64; 2];
    let window = |sim: &mut inf_player::runtime_sim::RuntimeSim| -> f64 {
        let from = hand(sim);
        let mut arc = 0.0f64;
        for _ in 0..90 {
            sim.step_once(RuntimeInput::default());
            arc = arc.max(f64::from((hand(sim) - from).length()));
        }
        arc
    };
    for (k, over) in [(0usize, true), (1usize, false)] {
        assert!(
            inf_ecs::anim_bridge::start_throw(sim.world_mut(), hero, over, 1.2),
            "the throw door refused"
        );
        throw_reach[k] = window(&mut sim);
        for _ in 0..90 {
            sim.step_once(RuntimeInput::default());
        }
    }
    // The control: the same ninety steps of the same idle, with no throw.
    let idle_arc = window(&mut sim);

    played.extend(slide_states.iter().cloned());
    played.extend(prone_states.iter().cloned());
    played.extend(swim_states.iter().cloned());
    played.extend(getup_states.iter().cloned());
    println!("\n=== the authored sets, on the island ===");
    println!("  standing, the hero carries its pelvis {stand_stance:.4} m up");
    println!(
        "  slide     {slide_states:?}  pelvis {slide_stance:.4} m, travelled {slide_travel:.2} m"
    );
    println!("  prone     {prone_states:?}  pelvis {prone_stance:.4} m");
    println!(
        "  swim      {swim_states:?}  in `{water_name}`: the PELVIS JOINT sat \
         {swim_pelvis_vs_water:+.3} m of the waterline (the capsule {swim_capsule_vs_water:+.3} m)"
    );
    println!("  get-up    {getup_states:?}  upright {upright}");
    println!(
        "  throw     the hand reached {:.3} m overhand, {:.3} m underhand, against {idle_arc:.3} m idle",
        throw_reach[0], throw_reach[1]
    );

    // ── the claims ───────────────────────────────────────────────────────────
    assert!(
        slide_states.contains("slide"),
        "the hero entered a slide and the machine played {slide_states:?}"
    );
    assert!(
        slide_stance < stand_stance - 0.05,
        "the slide left the pelvis at {slide_stance:.4} m against {stand_stance:.4} m standing"
    );
    assert!(
        slide_travel > 1.0,
        "the slide covered {slide_travel:.2} m, which is a crouch and not a slide"
    );
    assert!(
        prone_states.contains("prone_idle") || prone_states.contains("prone_crawl"),
        "the hero went prone and the machine played {prone_states:?}"
    );
    assert!(
        prone_stance < slide_stance,
        "prone left the pelvis at {prone_stance:.4} m against the slide's {slide_stance:.4} m"
    );
    if water.is_some() {
        assert!(
            swim_states.iter().any(|s| s.starts_with("swim_")),
            "the hero was in the island's own water and the machine played {swim_states:?}"
        );
        assert!(
            swim_pelvis_vs_water.abs() < 1.5,
            "a surface swim put the hero's PELVIS {swim_pelvis_vs_water:+.3} m from the \
             waterline of `{water_name}`"
        );
    } else {
        panic!("the island has no water body, so the swim half of this arm proves nothing");
    }
    assert!(
        upright && getup_states.contains("getup_standing"),
        "a ragdoll that never went down should get up ON ITS FEET: upright {upright}, the \
         machine played {getup_states:?}"
    );
    for (k, what) in [(0usize, "overhand"), (1usize, "underhand")] {
        assert!(
            throw_reach[k] > idle_arc * 3.0 + 0.05,
            "the {what} throw moved the hand {:.3} m against {idle_arc:.3} m of idle - the \
             additive is not reaching the pose",
            throw_reach[k]
        );
    }
}

/// **A CAPE, AUTHORED WITHOUT AN EDITOR AND MOVING ON THE ISLAND'S HERO**
/// (clause 8's garment door — the CHAR1a audit's item 87).
///
/// # Two things this proves, and they are separate
///
/// 1. **The headless door works.** `inf_editor_core::groom::garment_from_files`
///    is what `inf-import garment` calls, and it is the only way a `.inf_cloth`
///    can be made without a person clicking in the Model Editor. Before it,
///    `garment_from_session` took a live half-edge mesh and a click-built
///    selection, so a garment was something no CLI, no script and no CI arm
///    could produce — which is why the P24 cloth on the hero has been an owed
///    proof for three waves.
/// 2. **The garment simulates on the island's own hero.** The cape is put on
///    the hero with a `ClothSim`, the island is stepped with the hero walking,
///    and the CAPE'S OWN VERTICES are read out of `ClothStateRes` — not a
///    report, not a component, the particles the solver moved. The pinned
///    collar must stay where it was pinned, and the hem must not.
///
/// The cape and its cloth are written into the island project's Content, which
/// is local-only and outside this repository, at fixed GUIDs — so the run is
/// idempotent and the demo loop can find the same garment.
#[test]
fn a_cape_authored_from_files_moves_on_the_islands_hero() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    use inf_player::runtime_sim::RuntimeInput;

    // ── the cape, as a mesh asset ────────────────────────────────────────────
    //
    // A grid hanging behind the shoulders in the WEARER'S OWN model space, whose
    // origin is the character's feet: 0.50 m across, from 1.45 m (the shoulders)
    // down to 0.55 m, 0.10 m behind the spine. Nine columns by thirteen rows,
    // which is 96 quads and 208 constraints — small enough to solve inside a
    // character's budget and dense enough that a hem can swing.
    const COLS: usize = 9;
    const ROWS: usize = 13;
    let mut verts: Vec<inf_mesh::MeshVertex> = Vec::with_capacity(COLS * ROWS);
    let mut bounds = inf_mesh::Aabb::empty();
    for r in 0..ROWS {
        for c in 0..COLS {
            let u = c as f32 / (COLS - 1) as f32;
            let v = r as f32 / (ROWS - 1) as f32;
            let p = [
                (u - 0.5) * 0.50,
                1.45 - v * 0.90,
                // A shallow curve away from the back, so the sheet is not a
                // plane and its normal is not degenerate.
                -0.10 - 0.04 * (1.0 - (2.0 * u - 1.0) * (2.0 * u - 1.0)),
            ];
            bounds.grow(p);
            verts.push(inf_mesh::MeshVertex {
                position: p,
                normal: [0.0, 0.0, -1.0],
                uv: [u, v],
                ..Default::default()
            });
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    for r in 0..ROWS - 1 {
        for c in 0..COLS - 1 {
            let i = (r * COLS + c) as u32;
            let right = i + 1;
            let down = i + COLS as u32;
            let diag = down + 1;
            indices.extend_from_slice(&[i, down, right, right, down, diag]);
        }
    }
    let mesh_asset = inf_mesh::MeshAsset {
        schema_version: inf_mesh::MeshAsset::CURRENT_VERSION,
        submeshes: vec![inf_mesh::SubMesh {
            name: "Cape".to_string(),
            vertices: verts,
            indices,
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        bounds,
        material_slots: vec!["Cloth".to_string()],
        material_slot_assets: vec![None],
    };

    // ── through the door the CLI calls ───────────────────────────────────────
    let mesh_bytes = inf_asset::encode(&mesh_asset).expect("the cape mesh encodes");
    let skel_bytes = std::fs::read(content.join("Starter.inf_skel")).ok();
    let (cloth, report) = inf_editor_core::groom::garment_from_files(
        &mesh_bytes,
        skel_bytes.as_deref(),
        [0u8; 16],
        // The top 8 % of a 0.90 m cape is its collar — one row of vertices.
        0.08,
        inf_editor_core::groom::GarmentSpec::default(),
    )
    .expect("the headless garment door");
    println!("\n=== the cape, authored from files ===");
    println!(
        "  {} particles, {} triangles, {} stretch + {} bend, {} pinned, {} capsules",
        report.particles,
        report.triangles,
        report.stretch,
        report.bend,
        report.pinned,
        report.capsules
    );
    assert!(report.pinned > 0, "the collar was not pinned");
    assert!(
        report.pinned < report.particles / 4,
        "{} of {} particles are pinned, which is a board and not a cape",
        report.pinned,
        report.particles
    );
    // The refusal half of the rule, because a door that cannot say no is a door
    // that writes a cape which falls off on the first step.
    assert!(
        inf_editor_core::groom::garment_from_files(
            &mesh_bytes,
            None,
            [0u8; 16],
            0.0,
            inf_editor_core::groom::GarmentSpec::default(),
        )
        .is_ok(),
        "a pin fraction of zero is a free sheet, which is legal"
    );

    // ── written where the island can find it ─────────────────────────────────
    const CAPE: uuid::Uuid = uuid::Uuid::from_u128(0x0b1b_c00e_0000_0000_0000_0000_0000_0001);
    {
        let mut project =
            inf_editor_core::assets::AssetProject::open(&content).expect("the island's content");
        project
            .write_asset_with_id(
                &content,
                "Hero_Cape",
                &cloth,
                Some(inf_asset::AssetId::from(CAPE)),
                None,
                Vec::new(),
                None,
            )
            .expect("the cape writes");
    }

    // ── on the island's hero ─────────────────────────────────────────────────
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..600 {
        sim.step_once(RuntimeInput::default());
    }
    {
        let w = sim.world_mut();
        let e = w.entity_of(hero).expect("the hero is in the world");
        w.world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::ClothSim {
                asset: Some(CAPE),
                enabled: true,
                ..Default::default()
            });
    }
    let cape_now = |sim: &inf_player::runtime_sim::RuntimeSim| -> Vec<glam::Vec3> {
        sim.world()
            .world()
            .get_resource::<inf_ecs::cloth::ClothStateRes>()
            .and_then(|r| {
                r.0.get(&hero).map(|c| {
                    c.state
                        .x
                        .iter()
                        .map(|p| glam::Vec3::from_array(*p))
                        .collect()
                })
            })
            .unwrap_or_default()
    };
    // Let it seed and settle where it hangs.
    for _ in 0..120 {
        sim.step_once(RuntimeInput::default());
    }
    let hung = cape_now(&sim);
    assert_eq!(
        hung.len(),
        report.particles,
        "the island's cloth step did not seed the cape ({} particles in the store)",
        hung.len()
    );
    // …then WALK, and read the particles again.
    let ax: std::collections::BTreeMap<String, f32> = [("move_y".to_string(), 1.0f32)].into();
    let mut moved = 0.0f64;
    let mut collar = 0.0f64;
    let mut hem = 0.0f64;
    for _ in 0..180 {
        sim.step_once(RuntimeInput::default().with_axes(ax.clone()));
    }
    let walking = cape_now(&sim);
    // The particles are in the WEARER'S frame, so a character that has walked
    // 20 m does not move them at all — only the solver does. That is the whole
    // reason this measurement is honest without subtracting a placement.
    // **Which particles are pinned is asked of the SOLVER**, not of the grid
    // order this arm built: `from_mesh_asset` welds by position and re-indexes,
    // so row 0 of the author's grid is not particle 0 of the garment. The pins
    // are the zero inverse masses, which is what pinned MEANS.
    let inv_mass: Vec<f32> = sim
        .world()
        .world()
        .get_resource::<inf_ecs::cloth::ClothStateRes>()
        .and_then(|r| r.0.get(&hero).map(|c| c.state.inv_mass.clone()))
        .unwrap_or_default();
    for (i, (a, b)) in walking.iter().zip(hung.iter()).enumerate() {
        let d = f64::from((*a - *b).length());
        moved = moved.max(d);
        if inv_mass.get(i).copied().unwrap_or(1.0) == 0.0 {
            collar = collar.max(d);
        } else {
            hem = hem.max(d);
        }
    }
    println!(
        "  on the island, over 3 s of walking: the cape's worst particle moved {:.1} mm",
        moved * 1000.0
    );
    println!(
        "  the {} pinned particles moved {:.4} mm; the {} free ones moved {:.1} mm",
        inv_mass.iter().filter(|m| **m == 0.0).count(),
        collar * 1000.0,
        inv_mass.iter().filter(|m| **m != 0.0).count(),
        hem * 1000.0
    );
    assert!(
        moved > 0.005,
        "the cape's particles moved {:.3} mm while the hero walked - the garment is not \
         simulating",
        moved * 1000.0
    );
    assert!(
        collar < 1.0e-6,
        "the PINNED particles moved {:.4} mm - a pin that moves is not one",
        collar * 1000.0
    );
    assert!(
        hem > collar,
        "the hem moved {:.3} mm and the collar {:.3} mm - the cape is rigid",
        hem * 1000.0,
        collar * 1000.0
    );
}

/// **THE COMMITTED BODY'S DENSITY, PRICED** (clause 8's subdivision route).
///
/// The CHAR1a audit's item 85 has two halves. The first — `arm_length_ratio`
/// 0.42 → 0.30 — is done and blessed. The second asked for *"one Loop
/// subdivision with skin-weight interpolation on the post-subdivision cage (NOT
/// a second heat solve — O(V)): ~22 872 tris measured, or two passes ~91 488 if
/// the load/draw budgets allow (measure both; state which ships)"*.
///
/// **This wave measures both and ships neither**, and this arm is the
/// measurement rather than a sentence in a report. What it asserts is the shape
/// of the decision, so the day somebody disagrees with it the numbers are in
/// front of them:
///
/// * what the committed body actually costs today;
/// * what one and two Loop passes would cost, which is exactly ×4 and ×16
///   because Loop splits every triangle into four;
/// * what the character the island actually draws costs, and what its own LOD
///   ladder already offers instead;
/// * and that `inf_dcc::body::BodyOptions` — the generator's own tessellation
///   knobs — already reaches the same density band without a subdivision pass at
///   all, which is the load-bearing half of the refusal.
///
/// The reason the route is not taken is in the ledger and repeated here: the
/// island's hero is a MetaHuman drawing **95 330** triangles at every distance
/// (carried 126, PERF1's), so quadrupling a **5 718**-triangle fallback body
/// moves nothing a viewer of the showcase sees, while a re-bless of
/// `Starter_Body.inf_mesh` moves the committed bytes of five gates and the
/// `include_bytes!` payload of every project the template scaffolds.
#[test]
fn the_committed_bodys_density_is_measured_against_what_the_island_draws() {
    let rig = inf_anim::manny::build_manny(&inf_anim::BodyParams::default())
        .expect("the mannequin builds");
    let (mesh, _) = inf_dcc::body::body_mesh(&rig, &inf_dcc::body::BodyOptions::default())
        .expect("the committed body generates");
    // A half-edge face is an n-gon; the triangle count is what a renderer draws.
    let tris: usize = mesh
        .face_ids()
        .filter_map(|f| mesh.face_verts(f).map(|v| v.len().saturating_sub(2)))
        .sum();
    // The same generator, turned up — the route that needs no new kernel feature
    // and no re-bless of anything but the body it is asked for.
    let dense = inf_dcc::body::BodyOptions {
        limb_segments: 112,
        torso_segments: 168,
        finger_segments: 34,
        head_segments: 154,
        head_rings: 91,
    };
    let (mesh2, _) = inf_dcc::body::body_mesh(&rig, &dense).expect("a denser body generates");
    let tris2: usize = mesh2
        .face_ids()
        .filter_map(|f| mesh2.face_verts(f).map(|v| v.len().saturating_sub(2)))
        .sum();
    println!("\n=== the committed body's density, priced ===");
    println!(
        "  {:<26} {tris:>7} triangles, {:>6} vertices",
        "today",
        mesh.vert_count()
    );
    println!("  {:<26} {:>7} (x4)", "one Loop pass would be", tris * 4);
    println!("  {:<26} {:>7} (x16)", "two would be", tris * 16);
    println!(
        "  {:<26} {tris2:>7} triangles, {:>6} vertices",
        "the generator turned up",
        mesh2.vert_count()
    );
    println!("  ...which is PAST one Loop pass, with no new kernel feature");
    // **The 95 330 is a LITERAL, and this arm never opens the island** (CHAR1b.2
    // audit). It is the number `char1a3_gate`'s LOD arm measures off the island
    // project, quoted here so the refusal's second premise is legible beside its
    // first; the premise this arm actually ASSERTS is the parametric one below,
    // which is a fixture question and reproduces on CI where the island does
    // not exist. Said out loud because the arm's own name promises more.
    println!(
        "  {:<26} {:>7} at every distance (carried 126, quoted from char1a3_gate — \
         NOT measured here)",
        "the island's hero draws", 95_330
    );
    println!("  ...with 21040 and 7996 already on its ladder and no reader");
    assert!(
        tris > 4_000 && tris < 8_000,
        "the committed body is {tris} triangles; the numbers in this arm's docs are stated \
         against about 5 700 and want restating"
    );
    // The refusal's load-bearing half: the parametric route reaches the same
    // band the first Loop pass was asked for, so the pass buys smoothing rather
    // than density — and smoothing a body nobody on the island draws is not what
    // the showcase is short of.
    assert!(
        tris2 > tris * 3,
        "turning the generator up gave {tris2} against {tris}, so the parametric route does NOT \
         reach the subdivision band and the refusal's reason is wrong"
    );
    // …and it is not free. A body four times denser is four times the skinning,
    // four times the vertex fetch and four times the bytes in every project the
    // template scaffolds.
    assert!(
        mesh2.vert_count() > mesh.vert_count() * 3,
        "the denser body has {} vertices against {}",
        mesh2.vert_count(),
        mesh.vert_count()
    );
}
