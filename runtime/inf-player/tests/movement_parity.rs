//! **P29.3 audit (A3): the character movement fixed step runs identically in
//! both hosts.**
//!
//! `inf_physics::d3::step_character_movement` is one Ring-0 function the editor's
//! Simulate and the shipped player both call, so the two hosts *cannot* integrate
//! a character differently — and that is exactly the argument `pose_parity`,
//! `cloth_parity`, `hair_parity` and `deform_parity` each declined to rest on.
//! What a structural argument cannot see is the half that is not shared: whether
//! the two hosts **feed** the door the same intent and **slot** it in the same
//! place in their fixed steps. Both of those are hand-maintained mirrors, and
//! this wave rewrote both of them.
//!
//! P29.3 shipped `two_identical_worlds_move_byte_for_byte`, which runs one
//! process twice. It is a determinism gate and it is not a parity gate: it would
//! stay green if `SimSession` sampled a different axis name, built the intent
//! from a different action set, or skipped `apply_intent` altogether — every one
//! of which is a PIE-versus-shipping divergence in a feature whose whole point is
//! that the preview moves like the build. Mutation-measured: dropping
//! `apply_intent` from the editor's fixed step fails this file and nothing else
//! in the tree.
//!
//! No committed level carries a `CharacterMovement` (autostep is opt-in by
//! component presence, deliberately, so every sample and gate moves as it did),
//! so the fixture is built here rather than loaded — which is also why this file
//! exists instead of a leg on `character_demo`.
//!
//! **The trace is bits, not floats**, and every comparison is guarded against
//! vacuity: two characters that never moved agree perfectly.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    RigidBody3D, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;
/// Real gravity, so the solver step genuinely MOVES the dynamic body below —
/// a zero-gravity fixture makes `bridge3d.step` a no-op and the movement step's
/// slot in the fixed step unobservable.
const GRAVITY: DVec2 = DVec2::new(0.0, -9.81);
const STEPS: u32 = 240;

const HERO: Uuid = Uuid::from_u128(0x2903_0100);
const GROUND: Uuid = Uuid::from_u128(0x2903_0101);
const STEP_BLOCK: Uuid = Uuid::from_u128(0x2903_0102);
const CRATE: Uuid = Uuid::from_u128(0x2903_0103);

/// The capsule radius the fixture uses; the half-heights come from the
/// component's own defaults, so the arm is reading the shipped tuning.
const RADIUS: f64 = 0.3;

// ── the fixture ─────────────────────────────────────────────────────────────

fn block(centre: DVec3, half: DVec3) -> (RigidBody3D, Collider3D, Transform) {
    body(BodyKind3D::Static, centre, half)
}

/// A body of either kind. The fixture carries a **dynamic** one as well as the
/// static ground so that both hosts' bridges run their dynamic path under the
/// character rather than only their static one.
///
/// What it does **not** buy, measured rather than assumed: moving
/// `step_character_movement` to *after* the solver in one host leaves this trace
/// byte-identical, with the dynamic body and with real gravity. The reason is
/// structural — the mover is kinematic and does not push dynamics, so once the
/// crate has settled the solver moves nothing the character's sweep depends on,
/// and a one-step-old query BVH answers the same question. The slot is recorded
/// here as a bound of this arm rather than claimed as something it checks. What
/// it *does* check is the mirror this wave actually rewrote: drop
/// `apply_intent` from either host and the arm fails at step 1.
fn body(kind: BodyKind3D, centre: DVec3, half: DVec3) -> (RigidBody3D, Collider3D, Transform) {
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::from_dvec3(centre);
    (
        RigidBody3D {
            kind,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::from_dvec3(half),
            ..Default::default()
        },
        t,
    )
}

fn hero_parts() -> (
    RigidBody3D,
    Collider3D,
    CharacterController3D,
    CharacterMovement,
    Transform,
) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
    // An authored facing, so the trace also covers the seeding path (audit A1) —
    // a host that seeded from a different place would diverge on step 1.
    t.rotation.y = 35.0;
    (
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
    )
}

/// The same script for both hosts, as a pair of `(held actions, axes)` — the
/// two halves the two `SimInput`/`RuntimeInput` mirrors each carry.
///
/// It walks, sprints, crouches, jumps, slides and turns, so the trace visits
/// several modes rather than proving that two idle characters agree.
fn script(i: u32) -> (Vec<&'static str>, BTreeMap<String, f32>) {
    let mut held: Vec<&'static str> = Vec::new();
    if (40..140).contains(&i) {
        held.push("sprint");
    }
    if (60..70).contains(&i) {
        held.push("walk");
    }
    // Edges: the host derives `just_pressed` from the previous tick's set, so a
    // press has to be held for one tick and released.
    if i == 150 || i == 200 {
        held.push("crouch");
    }
    if i == 90 {
        held.push("jump");
    }
    let mut axes = BTreeMap::new();
    axes.insert("move_y".to_string(), if i % 53 < 45 { 1.0 } else { -0.4 });
    axes.insert("move_x".to_string(), if i % 97 < 20 { 0.6 } else { 0.0 });
    // A look RATE, which is what a delta axis arrives as.
    axes.insert("look_x".to_string(), if i % 31 < 10 { 90.0 } else { 0.0 });
    (held, axes)
}

/// One step's movement state, as bytes. Position, facing, velocity, the mode and
/// every derived output P29.4 will read — so a host that agreed on where the
/// character *is* while disagreeing on what it is *doing* still fails.
fn movement_bytes(world: &EcsWorld) -> Vec<u8> {
    let mut out = Vec::new();
    let Some(e) = world.entity_of(HERO) else {
        return out;
    };
    let w = world.world();
    let (Some(t), Some(cm), Some(c)) = (
        w.get::<Transform>(e),
        w.get::<CharacterMovement>(e),
        w.get::<Collider3D>(e),
    ) else {
        return out;
    };
    for v in [
        t.translation.x,
        t.translation.y,
        t.translation.z,
        t.rotation.y,
        c.half_extents.y,
        cm.runtime.velocity.x,
        cm.runtime.velocity.y,
        cm.runtime.velocity.z,
        cm.runtime.aim_yaw_deg,
        cm.runtime.aim_yaw_rate_dps,
        cm.runtime.body_yaw_deg,
        cm.runtime.target_yaw_deg,
        cm.runtime.mapped_speed,
        cm.runtime.gait_scalar,
        cm.runtime.stride_blend,
        cm.runtime.walk_run_blend,
        cm.runtime.relative_accel.x,
        cm.runtime.relative_accel.y,
        cm.runtime.land_impact_mps,
        cm.runtime.time_in_mode_s,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.push(cm.mode as u8);
    out.push(cm.gait as u8);
    out.push(cm.rotation_mode as u8);
    out.push(cm.runtime.actual_gait as u8);
    out.push(cm.runtime.direction as u8);
    out.push(cm.runtime.landing as u8);
    out.push(cm.runtime.refusal as u8);
    out.push(u8::from(cm.runtime.grounded));
    out.extend_from_slice(&cm.runtime.refusals.to_le_bytes());
    out
}

// ── the two hosts ───────────────────────────────────────────────────────────

fn player_trace() -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    for (guid, parts) in [
        (
            GROUND,
            block(DVec3::new(0.0, -0.5, -6.0), DVec3::new(40.0, 0.5, 40.0)),
        ),
        (
            STEP_BLOCK,
            block(DVec3::new(0.0, 0.1, 6.0), DVec3::new(8.0, 0.1, 0.4)),
        ),
        (
            CRATE,
            body(
                BodyKind3D::Dynamic,
                DVec3::new(0.0, 1.2, 3.6),
                DVec3::new(8.0, 0.4, 0.3),
            ),
        ),
    ] {
        let e = world.spawn_with_guid(guid, "Block", None);
        world.world_mut().entity_mut(e).insert(parts);
    }
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(hero_parts());
    world.mark_dirty();
    world.propagate();

    let mut sim = RuntimeSim::new(world, Vec::new(), GRAVITY, HZ);
    (0..STEPS)
        .map(|i| {
            let (held, axes) = script(i);
            sim.step_once(RuntimeInput::with_down(held).with_axes(axes));
            movement_bytes(sim.world())
        })
        .collect()
}

fn editor_trace() -> Vec<Vec<u8>> {
    let mut doc = SceneDoc::new();
    for (guid, parts) in [
        (
            GROUND,
            block(DVec3::new(0.0, -0.5, -6.0), DVec3::new(40.0, 0.5, 40.0)),
        ),
        (
            STEP_BLOCK,
            block(DVec3::new(0.0, 0.1, 6.0), DVec3::new(8.0, 0.1, 0.4)),
        ),
        (
            CRATE,
            body(
                BodyKind3D::Dynamic,
                DVec3::new(0.0, 1.2, 3.6),
                DVec3::new(8.0, 0.4, 0.3),
            ),
        ),
    ] {
        let e = doc.create_with_guid(guid, inf_editor_core::ipc::SpawnKind::Empty, "Block", None);
        doc.world_mut().world_mut().entity_mut(e).insert(parts);
    }
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(hero_parts());
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();

    let mut session = SimSession::enter(&mut doc, Vec::new(), GRAVITY, HZ);
    let out = (0..STEPS)
        .map(|i| {
            let (held, axes) = script(i);
            session.step_once(&mut doc, SimInput::with_down(held).with_axes(axes));
            movement_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

// ── the claims ──────────────────────────────────────────────────────────────

/// **ANTI-VACUITY.** Two empty traces are equal; so are two traces of a
/// character that never left the spot it was spawned on. The fixture has to have
/// moved, changed mode, and changed gait, or "identical" is a claim about
/// nothing.
fn assert_not_vacuous(trace: &[Vec<u8>]) {
    assert_eq!(trace.len() as u32, STEPS);
    assert!(
        !trace[0].is_empty(),
        "step 0 published no movement state at all — the hero has no movement \
         component in this host, so the door was never reached"
    );
    let n = trace[0].len();
    assert!(trace.iter().all(|t| t.len() == n));

    // The mode byte sits immediately after the twenty f64s.
    let mode_at = 20 * 8;
    let modes: std::collections::BTreeSet<u8> = trace.iter().map(|t| t[mode_at]).collect();
    assert!(
        modes.len() >= 3,
        "the trace visited only {} mode(s) — the script is not exercising the step",
        modes.len()
    );
    let gaits: std::collections::BTreeSet<u8> = trace.iter().map(|t| t[mode_at + 3]).collect();
    assert!(gaits.len() >= 2, "the trace never changed gait: {gaits:?}");
    assert_ne!(
        trace[0],
        trace[STEPS as usize - 1],
        "the character ended the run in exactly the state it started in"
    );
}

/// **PIE == shipping, on movement.** The editor's Simulate and the shipped
/// player integrate the same character byte for byte across a live 240-step
/// trace that walks, sprints, crouches, slides, jumps, lands and turns.
#[test]
fn both_hosts_move_the_same_character_byte_for_byte() {
    let player = player_trace();
    let editor = editor_trace();
    assert_not_vacuous(&player);
    assert_not_vacuous(&editor);
    for (i, (p, e)) in player.iter().zip(editor.iter()).enumerate() {
        assert_eq!(
            p, e,
            "the shipped player and the editor's Simulate diverged on movement at \
             step {i} — PIE would stop matching shipping for every character in \
             the engine"
        );
    }
}

/// The same host twice: the movement step is a pure function of the sim state
/// and the input, so the trace is reproducible. (`inf-physics`'s own
/// `two_identical_worlds_move_byte_for_byte` pins this at the door; this one
/// pins it through the host, where the input plumbing lives.)
#[test]
fn the_players_movement_trace_is_reproducible() {
    assert_eq!(player_trace(), player_trace());
}
