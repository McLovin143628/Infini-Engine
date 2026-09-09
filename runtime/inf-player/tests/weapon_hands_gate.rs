//! **The weapon verbs, run with the hands ON the weapon** (SK1c).
//!
//! # What this gate is for
//!
//! SK1b built hand IK and shipped it with no caller: `set_hand_ik` appeared in
//! `pose.rs`'s own tests and in `grip_gate.rs`, and nowhere else in the product.
//! The audit recorded it as a LOW — *reachable and unreached*, the same shape the
//! SK1a audit recorded for the retarget map one wave earlier.
//!
//! This wave gave it two producers, both inside `step_gameplay`, the one Ring-0
//! rule both hosts call:
//!
//! * **equipping a weapon** puts a two-handed `GunGrip` hold on the rig — the
//!   `ik_hand_gun` path — and closes each hand on its own affordance;
//! * **aiming** brings the holding hand up onto the aim line, which is what
//!   turns *carrying* a rifle into *pointing* one;
//! * **pressing E on something that names a grip** reaches the free hand to it
//!   and closes the fingers.
//!
//! So this file runs the I6 weapon verbs — equip, aim, fire, reload, unequip —
//! and an E-grab, on a **rigged** hero, and asserts PIE == shipping byte for byte
//! over the whole sequence. The existing capsule gates (`phase30_gameplay_gate`,
//! `weapon_3d`, `door_3d`) are untouched and still green; what was missing was a
//! course where the same verbs run on a character that has hands.
//!
//! # The trace
//!
//! `pose_state_bytes` and the gameplay report's engagement counters, compared
//! step by step between a `RuntimeSim` and a `SimSession`. The counters matter as
//! much as the bytes: two hosts that both asked for nothing agree perfectly, and
//! "the hand pass ran" and "a hand was asked to do something" are different
//! facts.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_anim::{
    AnimClip, Interpolation, JointTrack, QuatTrack, SkeletonAsset, SmState, SmTransition,
    StateMachine,
};
use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterController3D, CharacterMovement, Collider3D,
    ColliderShape3DKind, GlobalTransform, RigidBody3D, SkeletalMesh, Transform,
};
use inf_ecs::interact::{InteractVerb, Interactable};
use inf_ecs::item::{Inventory, ItemDef, ItemDefs};
use inf_ecs::math::Vec3d;
use inf_ecs::weapon::WeaponDef;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;
/// Long enough to walk the whole course below and settle after it.
///
/// A grab is a whole second at 60 Hz (`GRAB_EASE_S` in, `GRAB_HOLD_S` held,
/// `GRAB_EASE_S` out), and the point of the last arm is that the hand OPENS
/// again — so the course has to outlast it rather than stop while it is closed.
///
/// **Twenty more since wave WPN1**, for the punch: the release arm has to land
/// on step 139 exactly as it always did, so the swing goes after it.
const STEPS: u32 = 160;

const HERO: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0001);
const SM: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0003);
const MESH: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0004);
const IDLE: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0010);
/// The thing on the floor the hero presses E on.
const PROP: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0020);
/// The floor itself.
const GROUND: Uuid = Uuid::from_u128(0x5C1C_0000_0000_0000_0000_0000_0000_0030);

/// The hero's height, and therefore its capsule — the starter character's.
const HEIGHT_M: f64 = 1.75;
/// Where the hero stands.
const HERO_AT: DVec3 = DVec3::new(0.0, 0.0, 0.0);

// ── the fixture ─────────────────────────────────────────────────────────────

/// The 161-bone mannequin, with the catalogue it generates for itself.
fn rig() -> SkeletonAsset {
    inf_anim::build_manny(&inf_anim::BodyParams {
        height_m: HEIGHT_M,
        ..Default::default()
    })
    .expect("a mannequin")
}

/// A clip holding one spine joint at a constant angle — the same fixture shape
/// `grip_gate` uses, and for its reason: one key under `Step`, so "the pose
/// changed" always means the HANDS changed rather than a play-head drifting.
fn hold(joint: u16, deg: f32) -> AnimClip {
    let half = (deg as f64 * 0.5).to_radians();
    let q = [
        0.0,
        0.0,
        inf_math::psin64(half) as f32,
        inf_math::pcos64(half) as f32,
    ];
    AnimClip::new(
        "hold",
        vec![JointTrack {
            joint,
            translation: None,
            rotation: Some(QuatTrack::new(vec![0.0], vec![q], Interpolation::Step)),
            scale: None,
        }],
    )
}

fn machine() -> StateMachine {
    StateMachine {
        states: vec![SmState::clip("idle", *IDLE.as_bytes())],
        transitions: Vec::<SmTransition>::new(),
        entry: 0,
        ..Default::default()
    }
}

fn skeletons() -> BTreeMap<Uuid, SkeletonAsset> {
    BTreeMap::from([(SKEL, rig())])
}

fn machines() -> BTreeMap<Uuid, StateMachine> {
    BTreeMap::from([(SM, machine())])
}

fn clips() -> BTreeMap<Uuid, AnimClip> {
    BTreeMap::from([(IDLE, hold(3, 8.0))])
}

/// A rifle, and something to pick up.
fn defs() -> ItemDefs {
    let mut d = ItemDefs::default();
    d.insert(ItemDef {
        id: "rifle".into(),
        label: "Rifle".into(),
        stack_max: 1,
        mass_kg: 3.5,
        mesh: None,
        weapon: Some(WeaponDef {
            automatic: false,
            magazine: 5,
            reserve: 10,
            reload_s: 0.2,
            ..Default::default()
        }),
    });
    d.insert(ItemDef {
        id: "crate".into(),
        label: "Crate".into(),
        ..Default::default()
    });
    d
}

/// **Build the world both hosts start from.** One function, so the two cannot be
/// given different fixtures — `pose_parity`'s discipline, and the reason a byte
/// comparison means anything.
fn build(world: &mut EcsWorld) {
    // **Ground, and it is load-bearing** — not scenery. `RotationMode::Aiming`
    // is set on the GROUNDED movement branch, so a hero standing on nothing is
    // a hero that never enters the aim mode, and the aim half of this course
    // would be measuring an unpressed button. The first run of this file did
    // exactly that.
    let g = world.spawn_with_guid(GROUND, "Ground", None);
    let mut gt = Transform::IDENTITY;
    gt.translation = Vec3d::new(0.0, -0.5, 0.0);
    world.world_mut().entity_mut(g).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(40.0, 0.5, 40.0),
            ..Default::default()
        },
        gt,
    ));

    let radius = (HEIGHT_M * 0.15).clamp(0.1, 0.5);
    let half = (HEIGHT_M * 0.5 - radius).max(0.05);
    let e = world.spawn_with_guid(HERO, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(HERO_AT.x, HERO_AT.y + half + radius, HERO_AT.z);
    let mut inv = Inventory::default();
    let d = defs();
    inv.add(&d, "rifle", 1);
    // NOT equipped: the course equips it with the scroll wheel, which is the
    // verb an author gets, rather than starting in the state the gate is about.
    world.world_mut().entity_mut(e).insert((
        t,
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(radius, half, radius),
            radius,
            ..Default::default()
        },
        CharacterController3D::default(),
        CharacterMovement {
            player_controlled: true,
            stand_half_height_m: half,
            crouch_half_height_m: (half * 0.5).max(0.05),
            prone_half_height_m: (radius * 0.6).max(0.03),
            ..Default::default()
        },
        inv,
    ));

    // Something to press E on, a metre in front (the hero faces `+Z` at yaw 0).
    let p = world.spawn_with_guid(PROP, "Crate", None);
    let mut pt = Transform::IDENTITY;
    pt.translation = Vec3d::new(0.0, 0.4, 1.0);
    world.world_mut().entity_mut(p).insert((
        pt,
        GlobalTransform::default(),
        Interactable {
            verb: InteractVerb::Grab,
            label: "crate".into(),
            // **A `Grab` deliberately**, because it is the verb this engine has
            // and does not consume: before this wave the E key on one did
            // literally nothing, and the hand is now the thing it does. A gate
            // built on `PickUp` would be measuring the inventory as much as the
            // hand.
            grip: Some(inf_anim::GRIP_PROP.to_string()),
            ..Default::default()
        },
    ));
    world.world_mut().insert_resource(defs());
    world.mark_dirty();
    world.reindex_guids();
    world.propagate();
}

// ── the course ──────────────────────────────────────────────────────────────

/// What the player is doing on each step, as **action names** — the same strings
/// `inf_input`'s default map binds to RMB, LMB, R and the wheel.
///
/// | steps | the verb |
/// |---|---|
/// | 0–5 | nothing: an unarmed idle, the settled baseline |
/// | 6 | **equip** — the scroll wheel |
/// | 7–14 | **carrying** the rifle: both hands on it, not aiming |
/// | 15–30 | **aim** (RMB): the weapon comes up onto the aim line |
/// | 31 | **fire** (LMB, on top of the aim) |
/// | 32–45 | aiming through the shot's cooldown |
/// | 46 | **reload** (R) |
/// | 47–60 | the reload runs to its clock |
/// | 61 | **unequip** |
/// | 62–70 | unarmed again — and it must pose the baseline exactly |
/// | 71 | **E** on a crate that names a grip |
/// | 72–131 | the grab eases in, holds and opens |
/// | 132–139 | open again, settled |
/// | 140 | **punch** (wave WPN1) — an empty hand takes the attack edge |
/// | 141–159 | the fist's own cycle runs out |
///
/// **Carrying and aiming are separate bands on purpose.** The claim the aim arm
/// makes is that *aiming* moves the hand, and comparing an aiming step against
/// an unarmed one would be satisfied by the equip alone.
fn course(step: u32) -> (Vec<&'static str>, BTreeMap<String, f32>) {
    let mut axes = BTreeMap::new();
    let down: Vec<&'static str> = match step {
        6 => {
            axes.insert("weapon_switch".to_string(), 1.0);
            Vec::new()
        }
        15..=30 => vec!["aim"],
        31 => vec!["aim", "attack"],
        32..=45 => vec!["aim"],
        46 => vec!["aim", "reload"],
        47..=60 => vec!["aim"],
        71 => vec!["interact"],
        // WPN1: the punch, AFTER the release arm's step 139 — an empty hand is
        // the third consumer of the attack edge and the one case that could
        // have tripped `muzzles_without_a_socket` on a rigged character.
        140 => vec!["attack"],
        _ => Vec::new(),
    };
    (down, axes)
}

/// The one step that is not an input: the engine has no *unequip* action — the
/// inventory panel's own door sets it — so the course drives that door directly,
/// in a shared function, so both hosts do it on the same step.
fn unequip_at(step: u32, world: &mut EcsWorld) {
    if step != 61 {
        return;
    }
    let Some(e) = world.entity_of(HERO) else {
        return;
    };
    if let Some(mut inv) = world.world_mut().get_mut::<Inventory>(e) {
        inv.unequip();
    }
}

/// One step's reading: the pose, and what the hand pass was asked for.
#[derive(Clone, Debug, PartialEq)]
struct Step {
    pose: Vec<u8>,
    /// `(weapon holds, grabs, shots, reloads)` — the engagement counters.
    engagement: (u32, u32, u32, u32),
    /// `(swings, muzzles_without_a_socket)` — wave WPN1's two.
    ///
    /// The second is the SK1b tripwire, and this is the course it belongs on:
    /// the hero here has a real 161-bone rig that publishes `hand_r`, so a shot
    /// that fell back to 1.4 m above the feet would be a rig that quietly lost
    /// its hand. `phase30_gameplay_gate`'s capsule hero publishes no pose at
    /// all and is the legitimate fallback the counter deliberately does not
    /// count, so only this file can arm it.
    melee: (u32, u32),
    /// Whether a weapon entity exists this step, and the magazine.
    armed: (bool, u32),
    /// **Where the hands were asked to hold the weapon**, world metres (wave
    /// WPN2b) -- `HandIk::reach[1]`, which is `aim_hold_point`'s own answer.
    ///
    /// A NUMBER beside the pose bytes, because this wave made three of this
    /// gate's byte-equality arms false by design: a breathing character's hands
    /// SWAY, so an aimed pose is never twice the same bytes again. The claims
    /// those arms were making -- settled, kicked, recovered -- are claims about
    /// a DISTANCE, and this is the distance.
    hold: Option<[f64; 3]>,
}

/// Where the hand pass was told to put the weapon this step.
fn hold_of(world: &EcsWorld) -> Option<[f64; 3]> {
    let h = inf_ecs::pose::hand_ik(world, HERO)?;
    let r = h.reach[1].as_ref()?;
    Some([r.target.x, r.target.y, r.target.z])
}

/// Millimetres between two hold points; `0.0` when either is absent.
fn hold_mm(a: Option<[f64; 3]>, b: Option<[f64; 3]>) -> f64 {
    let (Some(a), Some(b)) = (a, b) else {
        return 0.0;
    };
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * 1000.0
}

fn armed_of(world: &EcsWorld) -> (bool, u32) {
    let weapon = inf_physics::d3::gameplay::equipped_weapon_guid(HERO);
    let has = world.entity_of(weapon).is_some();
    let mag = world
        .entity_of(HERO)
        .and_then(|e| world.world().get::<inf_ecs::weapon::WeaponState>(e))
        .map(|s| s.magazine)
        .unwrap_or(0);
    (has, mag)
}

fn player_trace() -> Vec<Step> {
    let mut world = EcsWorld::new();
    build(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(clips());
    (0..STEPS)
        .map(|s| {
            unequip_at(s, sim.world_mut());
            let (down, axes) = course(s);
            sim.step_once(RuntimeInput::with_down(down).with_axes(axes));
            let r = sim.gameplay();
            Step {
                pose: inf_ecs::pose::pose_state_bytes(sim.world()),
                engagement: (r.hands.0, r.hands.1, r.shots, r.reloads),
                melee: (r.swings, r.muzzles_without_a_socket),
                armed: armed_of(sim.world()),
                hold: hold_of(sim.world()),
            }
        })
        .collect()
}

fn editor_trace() -> Vec<Step> {
    let mut doc = SceneDoc::new();
    build(doc.world_mut());
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    session.set_state_machines(machines());
    session.set_skeletons(skeletons());
    session.set_pose_clips(clips());
    let out = (0..STEPS)
        .map(|s| {
            unequip_at(s, doc.world_mut());
            let (down, axes) = course(s);
            session.step_once(&mut doc, SimInput::with_down(down).with_axes(axes));
            let r = session.gameplay();
            Step {
                pose: inf_ecs::pose::pose_state_bytes(doc.world()),
                engagement: (r.hands.0, r.hands.1, r.shots, r.reloads),
                melee: (r.swings, r.muzzles_without_a_socket),
                armed: armed_of(doc.world()),
                hold: hold_of(doc.world()),
            }
        })
        .collect();
    session.exit(&mut doc);
    out
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **ANTI-VACUITY**, and it is most of this gate.
///
/// Two hosts that both did nothing agree perfectly. Every claim here is about a
/// step where something is supposed to have happened, asserted against one where
/// it is not.
fn assert_not_vacuous(t: &[Step]) {
    assert_eq!(t.len() as u32, STEPS);
    assert!(!t[0].pose.is_empty(), "step 0 published no pose at all");
    // The 161-bone trace SK1a priced: 36 B header + 40 B a joint. Pinned as the
    // number, so a rig that quietly lost its side tables reads as a rig that
    // lost its side tables rather than as a quieter hand.
    assert_eq!(
        t[0].pose.len(),
        6476,
        "the hero is not a 161-bone character"
    );

    // -- the unarmed baseline --
    assert_eq!(
        t[0].engagement,
        (0, 0, 0, 0),
        "something was asked for on an unarmed idle step"
    );
    // **From step 3, not step 0** (wave CHAR1b.1). The foot-IK seam is one fixed
    // step wide by construction — the movement step probes the ground under the
    // feet the POSE step published last step — so step 0 has no goals, step 1
    // has the first, and the character is standing on the ground it is standing
    // on from step 2. Before this wave the whole mechanism was switched off (no
    // clip in the engine carried `Enable_FootIK_*`), so step 0 and step 5 were
    // the same bytes and the window did not have to be named. It does now.
    assert_eq!(t[3].pose, t[5].pose, "the idle pose is not settled");
    // …and it really did settle rather than never having moved: the warm-up is
    // visible, which is what stops the line above from being a claim about a
    // mechanism that is off again.
    assert_ne!(
        t[0].pose, t[5].pose,
        "the pose never changed at all over the settle, so the foot pass is not \
         running and the window above proves nothing"
    );
    assert_eq!(t[0].armed, (false, 0), "the hero starts armed");

    // -- EQUIP: a weapon entity appears and both hands go on it --
    assert!(t[10].armed.0, "the scroll wheel did not equip the rifle");
    assert_eq!(
        t[10].armed.1, 5,
        "the magazine is not full: {:?}",
        t[10].armed
    );
    assert_eq!(t[10].engagement.0, 1, "no weapon hold was asked for");
    assert_ne!(
        t[5].pose, t[10].pose,
        "picking a rifle up changed nothing about the hands"
    );
    assert_eq!(t[10].pose, t[14].pose, "the carried pose is not settled");

    // -- AIM: the weapon comes UP, which is a different pose again --
    //
    // Asserted against a settled CARRYING step, not against the unarmed idle:
    // "the pose changed since idle" is satisfied by the equip alone, and the
    // claim here is that *aiming* moves the hand.
    assert_ne!(
        t[14].pose, t[25].pose,
        "aiming did not move the hands — the reach is not driving the hold"
    );
    // **RE-STATED AT WAVE WPN2b, WITH ITS CAUSE.** This was
    // `assert_eq!(t[25].pose, t[28].pose)` -- the aimed pose settles to the same
    // bytes twice -- and that is now FALSE by design: a breathing character's
    // hands sway (`inf_ecs::feel::sway_offset` -- six millimetres of chest at
    // 0.2 Hz, plus a bob that is zero when standing still), so the hold point
    // moves every step for as long as the weapon is up. It is the doc's own
    // "offset the weapon using Sin/Cos waves ... so the gun model feels organic
    // rather than rigid".
    //
    // What the arm was claiming is still claimed, as the distance it always
    // was: three steps of an aimed idle move the hold point by less than the
    // sway's whole amplitude, and the shot below moves it by an order of
    // magnitude more.
    let settle_mm = hold_mm(t[25].hold, t[28].hold);
    assert!(
        settle_mm < 8.0,
        "the aimed hold moved {settle_mm:.4} mm over three idle steps, which is more than the sway's own amplitude -- something other than breathing is moving the hands"
    );
    assert!(
        hold_mm(t[25].hold, t[26].hold) > 0.0,
        "the aimed hold did not move at all between two steps -- the sway is not reaching the hold point"
    );

    // -- THE RECOIL (wave WPN1): the shot MOVES the hands, and the weapon
    //    settles back onto the aim line by the time it may fire again --
    //
    // The claim is a *return*, not a change: an `assert_ne!` on the firing step
    // alone is satisfied by a hand that kicked and stayed kicked, which is the
    // shape a recoil written as a latch nobody clears has. The rifle here is
    // 600 rpm, so its cycle is 0.1 s = six steps at 60 Hz — step 31 fires, 32–36
    // are the settle, and 37 is the weapon back where aiming put it.
    // **RE-STATED AT WAVE WPN2b**, in millimetres, for the settle arm's reason
    // above and one more: the recoil is a SPRING now, not a linear ramp off the
    // fire clock, so "back where it was" is an amplitude and not a byte.
    //
    // The impulse is deposited on the firing step and integrated by the movement
    // runtime's look integrator on the NEXT one (the stated one-step seam), so
    // the kick is read from step 31 forward rather than from step 30.
    // Every step read here is inside the course's own aiming band (15..=60), and
    // that is asserted rather than assumed: `hold_mm` answers `0.0` when either
    // hold is absent, so a character that stopped aiming would satisfy the
    // settle arms perfectly and fail nothing.
    for at in [30_usize, 31, 35, 45, 58] {
        assert!(
            t[at].hold.is_some(),
            "step {at} asked for no hold at all - it is outside the aiming band and the millimetres below mean nothing"
        );
    }
    // The spring's peak is `1 / sqrt(k)` after the impulse, which at k = 220 is
    // 67 ms -- four steps. So 31 -> 35 is the rise, 35 -> 45 is the fall, and 58
    // is twenty-seven steps out, where a critically damped spring is at 1.5 % of
    // its peak.
    let kick_mm = hold_mm(t[31].hold, t[35].hold);
    let decay_mm = hold_mm(t[35].hold, t[45].hold);
    let home_mm = hold_mm(t[30].hold, t[58].hold);
    println!("the shot moved the hold {kick_mm:.3} mm, decayed {decay_mm:.3} mm, and came home to {home_mm:.3} mm");
    assert!(
        kick_mm > 20.0,
        "the shot moved the hold {kick_mm:.4} mm -- the recoil spring is not reaching `HandIk::reach`, and the pose two hosts compare cannot see it"
    );
    assert!(
        decay_mm > 20.0,
        "the recoil is a LATCH rather than a decay: {decay_mm:.4} mm of movement over ten steps of settle"
    );
    assert!(
        home_mm < 8.0,
        "the weapon never came back onto the aim line: {home_mm:.4} mm off, which is more than the sway -- a recoil that does not recover is a weapon pointing somewhere else for the rest of the level"
    );
    // …and the aim itself did NOT move, which is the ruling `aim_hold_point`
    // states: the pose climbs and the bullet goes where the player is pointing.
    // Measured as the SHOT: exactly one round left and it left on step 31.
    assert_eq!(t[31].engagement.2, 1, "the shot is not on step 31");

    // -- FIRE and RELOAD really happened --
    // **`shots` counts every attack that left, ROUNDS AND SWINGS** — wave WPN1
    // put a punch through the same trigger, the same clock and the same
    // counter, so this is the round count and `melee.0` is subtracted rather
    // than a second counter being invented.
    let rounds: u32 =
        t.iter().map(|s| s.engagement.2).sum::<u32>() - t.iter().map(|s| s.melee.0).sum::<u32>();
    assert_eq!(rounds, 1, "expected exactly one ROUND over the course");
    assert_eq!(
        t.iter().map(|s| s.engagement.3).sum::<u32>(),
        1,
        "expected exactly one reload over the course"
    );
    assert_eq!(t[35].armed.1, 4, "the shot did not spend a round");
    assert_eq!(t[60].armed.1, 5, "the reload did not refill the magazine");

    // -- UNEQUIP: the weapon leaves the world, and the hands open --
    assert!(!t[65].armed.0, "the weapon entity outlived the unequip");
    assert_eq!(
        t[65].engagement.0, 0,
        "a hold was still asked for after unequipping"
    );
    // **THE RELEASE**, and it is the sharpest arm in the file: `apply_grip` sets
    // a pose rather than accumulating a delta, so a hand that has let go of
    // everything must pose the bytes it posed before it ever held anything —
    // exactly, to the bit. A solver that drifted would pass every `assert_ne!`
    // above and fail here.
    // The settled idle (step 3, see `assert_not_vacuous`) and not step 0, for
    // the same reason: step 0 is the foot-IK seam own first frame and has no
    // goals yet, so it is the one step in the whole course that is not a pose
    // of a character standing on the ground.
    assert_eq!(
        t[3].pose, t[70].pose,
        "an unarmed hero after a whole weapon course does not pose what it \
         posed before it picked anything up"
    );

    // -- E-GRAB: the free hand reaches the crate and closes on it --
    assert_eq!(t[75].engagement.1, 1, "pressing E asked for no grab");
    assert_ne!(
        t[70].pose, t[75].pose,
        "the E-grab moved no bone — the interactable's grip name reached nothing"
    );
    // An EASE, not a snap: closing further moves the pose again.
    assert_ne!(
        t[75].pose, t[82].pose,
        "the grab snapped to its end state instead of easing in"
    );
    // …and it OPENS again, back to the settled unarmed pose, to the bit.
    // Step 3 for the same reason as the two above: the settled idle.
    assert_eq!(t[3].pose, t[139].pose, "the hand never let go of the crate");

    // -- THE PUNCH (wave WPN1), and the tripwire it could have sprung --
    //
    // An empty hand takes the attack edge and swings. On a RIGGED character
    // that is the one case that could have counted against
    // `muzzles_without_a_socket`: a fist has no weapon entity to hang off a
    // socket, so `muzzle_of` takes the capsule rule *correctly* and a tripwire
    // that did not know about melee would fire on every swing.
    assert_eq!(
        t.iter().map(|s| s.melee.0).sum::<u32>(),
        1,
        "expected exactly one swing over the course"
    );
    assert_eq!(t[140].melee.0, 1, "the punch is not on step 140");
    assert_eq!(
        t.iter().map(|s| s.melee.1).sum::<u32>(),
        0,
        "a POSED character fired from the 1.4 m capsule rule — its rig does not \
         publish `hand_r`, or its weapon entity was never placed, and every \
         shot in this course is a silent half-metre out"
    );
    // …and the swing moved no bone, which is the honest bound: a fist asks the
    // hand pass for nothing (`equipped_weapon` answers `None`), so a punch is
    // damage and an animation trigger and no pose. Carried by name.
    assert_eq!(
        t[3].pose, t[145].pose,
        "the punch moved the pose — which would be an improvement, and would \
         mean this arm and the wave's carried list are both out of date"
    );

    // The course really is a course: a solver that collapsed everything onto one
    // pose would satisfy the pairs above only by accident.
    // Pinned as the NUMBER, on the grip gate's own precedent (SK1b audit): a
    // printed count is not an asserted one, and a solver that collapsed the
    // course onto three poses would satisfy every `assert_ne!` pair above by
    // keeping exactly those apart.
    let mut distinct: Vec<&Vec<u8>> = t.iter().map(|s| &s.pose).collect();
    distinct.sort();
    distinct.dedup();
    //
    // **Eighteen until wave WPN1, twenty-four since, twenty-FIVE since CHAR1b.1
    // -- and SIXTY-FOUR since WPN2b**, which is this wave's own re-statement of
    // the arm with its cause.
    //
    // The twenty-fifth was the foot-IK seam's own first frame: step 0 poses
    // before any goal exists (the movement step probes the ground under the feet
    // the pose step published LAST step), so it is one pose the rest of the
    // course never returns to. The three settle arms above take step 3 as the
    // settled idle for the same reason.
    //
    // The thirty-nine this wave adds are the SWAY. The hold point is offset by
    // `inf_ecs::feel::sway_offset` for as long as the weapon is up, and the
    // course's aiming band is steps 15..=60 -- forty-six steps, every one of
    // them a different point, because a breathing character's hands are never
    // twice in the same place. So the count is now roughly "the aiming band,
    // plus the carried and unarmed poses that really are still".
    //
    // The number is quoted rather than relaxed because that is still the
    // arithmetic, and because the mutation it catches is unchanged: a solver
    // that collapsed the course onto three poses would satisfy every pair above
    // by keeping exactly those apart. What it no longer catches on its own is a
    // recoil that never recovered, which the millimetres above now measure
    // directly.
    assert_eq!(
        distinct.len(),
        64,
        "the course posed {} distinct poses of {STEPS} steps",
        distinct.len()
    );
    // The claim the number is a proxy for, said outright: the aiming band is
    // where the sway lives and it is very nearly all distinct.
    let aiming: std::collections::BTreeSet<&Vec<u8>> = t[15..=60].iter().map(|s| &s.pose).collect();
    assert!(
        aiming.len() >= 40,
        "the aiming band posed {} distinct poses of 46 - the sway is not reaching the hands",
        aiming.len()
    );

    println!(
        "WEAPON HANDS: {STEPS} steps, {} distinct poses, {} bytes a step",
        distinct.len(),
        t[0].pose.len()
    );
}

/// **THE GATE: PIE == shipping over equip → aim → fire → reload → unequip, and
/// an E-grab, on a rigged hero.**
#[test]
fn pie_equals_shipping_over_the_weapon_verbs_with_hands_on_the_weapon() {
    let a = player_trace();
    let b = editor_trace();
    assert_not_vacuous(&a);
    assert_not_vacuous(&b);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            x.engagement, y.engagement,
            "step {i}: the two hosts asked their hands for different things"
        );
        assert_eq!(
            x.melee, y.melee,
            "step {i}: the two hosts disagree about the swing or the muzzle"
        );
        assert_eq!(
            x.armed, y.armed,
            "step {i}: the two hosts disagree about the weapon"
        );
        assert_eq!(
            x.pose, y.pose,
            "step {i}: the two hosts posed the armed hero differently"
        );
    }
}

/// **A character with no weapon and no grab poses exactly what it did before
/// this wave.**
///
/// The `HandIkRes` doctrine is that absent costs nothing, and the hand pass now
/// runs on every character every step — so the claim is worth an arm rather than
/// a comment. Two worlds, identical but for a rifle in the bag, stepped the same
/// way: the unarmed one must pose the same bytes at every step, and the armed
/// one must not.
#[test]
fn the_hand_pass_costs_an_unarmed_character_nothing() {
    let trace = |armed: bool| -> Vec<Vec<u8>> {
        let mut world = EcsWorld::new();
        build(&mut world);
        if !armed {
            let e = world.entity_of(HERO).expect("the hero");
            world.world_mut().entity_mut(e).remove::<Inventory>();
        }
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
        sim.set_state_machines(machines());
        sim.set_skeletons(skeletons());
        sim.set_pose_clips(clips());
        (0..20u32)
            .map(|s| {
                let (down, axes) = course(s);
                sim.step_once(RuntimeInput::with_down(down).with_axes(axes));
                inf_ecs::pose::pose_state_bytes(sim.world())
            })
            .collect()
    };
    let bare = trace(false);
    let armed = trace(true);
    assert!(!bare[0].is_empty());
    // **From step 3** — see `assert_not_vacuous`'s note on the same window. The
    // first two steps are the foot-IK seam warming up, which is a different
    // pass entirely and is measured by `char1b_gate`.
    for (i, w) in bare.iter().enumerate().skip(3) {
        assert_eq!(
            &bare[3], w,
            "step {i}: an unarmed character's pose moved after it settled, so \
             the hand pass is not free when nothing asks for it"
        );
    }
    // ANTI-VACUITY: the armed run really does diverge, or the loop above is a
    // statement about the fixture rather than about the pass.
    assert_ne!(
        bare[19], armed[19],
        "the armed and unarmed runs pose identically, so this arm proves nothing"
    );
}

/// The mannequin's palette is only meaningful if the rig really has both hands
/// and both arms — the two things `apply_hand_ik` needs before it can write a
/// bone, asserted at the fixture rather than three passes downstream.
#[test]
fn the_fixture_rig_can_actually_be_solved() {
    let rig = rig();
    let roles = rig.role_index();
    for side in [inf_anim::BoneSide::Left, inf_anim::BoneSide::Right] {
        let hand = roles
            .first(inf_anim::BoneRoleKind::Hand, side)
            .unwrap_or_else(|| panic!("no {side:?} hand"));
        assert!(inf_anim::hand_of(&rig.skeleton, roles, hand).is_some());
        assert!(inf_anim::arm_chain(&rig.skeleton, roles, side).is_some());
    }
    for name in [
        inf_anim::GRIP_RIFLE,
        inf_anim::GRIP_RIFLE_FORE,
        inf_anim::GRIP_PROP,
    ] {
        assert!(
            rig.grips.iter().any(|g| g.name == name),
            "the generated rig carries no `{name}`, so the course asks for a \
             grip that resolves to nothing and writes no bone"
        );
    }
}
