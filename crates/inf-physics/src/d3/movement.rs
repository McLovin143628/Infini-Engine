//! **The character movement fixed step** (P29.3) — the half that needs a world.
//!
//! [`inf_ecs::movement`] holds the rules as pure functions of numbers. This
//! holds the one function both hosts call, and everything in it that a rule
//! cannot answer alone: where the ground is, whether the taller capsule fits,
//! what the sweep actually hit.
//!
//! # One door, and why it is spelled once
//!
//! [`step_character_movement`] is to movement what
//! `inf_ecs::pose::step_pose_evaluation` is to animation: a single Ring-0
//! function the editor's Simulate and the shipped player both call, so PIE and
//! shipping cannot integrate a character differently. The port map's IM-2b
//! records what the alternative looks like — `build_mover3d` existed twice, as a
//! hand-maintained byte-identical pair, and §13's own risk register notes that a
//! host-versus-host text compare cannot see a value that is wrong in both. That
//! pair is **retired** by [`mover_for`] rather than kept in step.
//!
//! # The velocity is ours (impedance mismatch IM-2)
//!
//! rapier's `KinematicCharacterController` has no velocity model at all: it
//! moves a shape and reports what it touched. UE's `CharacterMovementComponent`
//! integrates velocity from acceleration under a friction-and-braking model, and
//! that model is exactly what ALS's three curve channels drive. So the engine
//! keeps its own integrator ([`inf_ecs::movement::integrate_planar_velocity`])
//! and uses rapier purely as sweep-and-slide plus autostep. Trying to express
//! ALS's friction through rapier's options would be a translation of a model
//! into a thing that has no model.
//!
//! # The water is P20's, through P20's door
//!
//! Swimming is not re-implemented here. The latch, its hysteresis band, the
//! buoyancy balance and the speed cap all stay in
//! [`crate::d3::water`] and are reached through
//! [`PhysicsBridge3D::update_swim`] and
//! [`apply_swim_motion`](PhysicsBridge3D::apply_swim_motion) — the same two
//! calls `physics3d.move_and_slide` has made since P20.2. What P29.3 adds is
//! that the *mode enum* reads that latch, so a swimming character is in
//! `MovementMode::SwimSurface` rather than in `Grounded` with a special case
//! bolted on. One door, two readers.

use std::collections::BTreeSet;

use glam::{DQuat, DVec3};

use inf_ecs::components::{
    CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait, LandingKind,
    MantleState, MovementMode, MovementRefusal, RotationMode, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement as model;
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;
use super::traversal::{self, LedgeSettings};
use super::water;
use super::{AutoStep3D, CharacterMover3D, ColliderId3D, ColliderShape3D};

/// A character's capsule radius when its collider is not a capsule at all.
///
/// The movement step resizes a **capsule**'s half-height to change stance. An
/// entity whose collider is a box or a sphere keeps whatever shape it was given
/// and simply does not change size when it crouches — a value, not a refusal,
/// because the mode and the speeds are still meaningful.
const FALLBACK_RADIUS_M: f64 = 0.3;

/// How far the body may be turned away from the aim direction while standing
/// still and aiming, degrees (ALS `LimitRotation(-100, 100, 20)`).
const AIM_BODY_LIMIT_DEG: f64 = 100.0;
/// The exponential rate the clamp above pulls at (ALS's third argument).
const AIM_BODY_LIMIT_INTERP: f64 = 20.0;

/// The slowest a turn-in-place may turn, deg/s.
///
/// The rotation-rate curve is read at the character's own normalized speed, and
/// a character standing still reads the *stopped* anchor — which for a sensible
/// tuning is small or zero, because that is the rate a walking character's body
/// chases its velocity at. A turn in place is not that: it is a deliberate
/// re-facing, and ALS gets its rate from the turn animation's own
/// `RotationAmount` curve. Ours is a floor under the curve, so a project that
/// tunes a fast turn keeps it and one that tunes a slow walk still turns.
const TURN_MIN_RATE_DPS: f64 = 90.0;
/// How close to its target a turn in place must get before it is finished,
/// degrees. Below this the exponential stage would take unbounded time to
/// arrive, and a turn that never ends is a character that never turns again.
const TURN_SETTLE_DEG: f64 = 1.0;

/// How much movement input a mantle attempt needs (ALS gates its jump-triggered
/// mantle on `bHasMovementInput`): a jump at a wall with the stick centred is a
/// jump, not a climb.
const MANTLE_MIN_INPUT: f64 = 0.1;

/// Build the kinematic mover for `guid` from its components — **the one
/// construction site**, replacing the byte-identical `build_mover3d` pair the
/// two hosts each carried (IM-2b).
///
/// Reads three components, in this order of authority:
///
/// * [`Collider3D`] gives the swept shape. A missing collider is a 0.5 × 0.25
///   capsule, which is what both hosts' copies used.
/// * [`CharacterController3D`] gives the skin width and the ground snap — the
///   *mover's* own tuning, unchanged since P9.1.
/// * [`CharacterMovement`], when present, gives the **slope authority**, the
///   slide-back angle and the autostep. When it is absent nothing changes at
///   all: no autostep, no slide angle, and the slope comes from
///   `CharacterController3D` exactly as before.
///
/// That last clause is deliberate and it is what keeps every committed sample
/// byte-identical. Turning autostep on for entities that never asked for it
/// would change how the platformer, the coastal swimmer and the physics
/// playground move — and those are gates.
///
/// # Two numbers that meant one thing
///
/// `CharacterController3D::max_slope_deg` and
/// `CharacterMovement::slope_limit_deg` both describe "the steepest slope this
/// character walks up". Two authorities for one fact is how they drift, so an
/// entity that has a movement component has exactly one: the movement
/// component's. The controller's remains the authority for everything else.
pub fn mover_for(world: &EcsWorld, guid: uuid::Uuid) -> CharacterMover3D {
    let default_shape = ColliderShape3D::Capsule {
        half_height: 0.5,
        radius: 0.25,
    };
    let Some(entity) = world.entity_of(guid) else {
        return CharacterMover3D::new(default_shape);
    };
    let w = world.world();
    let shape = w
        .get::<Collider3D>(entity)
        .map(collider_shape3d)
        .unwrap_or(default_shape);
    let cc = w.get::<CharacterController3D>(entity).copied();
    let cm = w.get::<CharacterMovement>(entity);
    let mut mover = CharacterMover3D::new(shape).up(DVec3::Y).slide(true);
    if let Some(cc) = cc {
        let slope_deg = cm.map(|m| m.slope_limit_deg).unwrap_or(cc.max_slope_deg);
        mover = mover
            .offset(cc.offset.max(1e-4))
            .max_slope_climb_angle(slope_deg.to_radians())
            .snap_to_ground(if cc.snap_to_ground > 0.0 {
                Some(cc.snap_to_ground)
            } else {
                None
            });
    } else {
        mover = mover.offset(0.02);
        if let Some(cm) = cm {
            mover = mover.max_slope_climb_angle(cm.slope_limit_deg.to_radians());
        }
    }
    if let Some(cm) = cm {
        // THE line that makes stairs work. See `CharacterMover3D::autostep`.
        if cm.step_height_m > 0.0 {
            mover = mover.autostep(Some(AutoStep3D {
                max_height: cm.step_height_m,
                min_width: cm.step_min_width_m,
                include_dynamic_bodies: true,
            }));
        }
        mover = mover.min_slope_slide_angle(cm.slide_slope_deg.to_radians());
    }
    mover
}

/// A [`Collider3D`] component's shape as the facade shape. Lifted from the two
/// hosts along with `build_mover3d`, for the same reason.
pub fn collider_shape3d(c: &Collider3D) -> ColliderShape3D {
    match c.shape_kind {
        ColliderShape3DKind::Box => ColliderShape3D::Box {
            half_extents: c.half_extents.to_dvec3(),
        },
        ColliderShape3DKind::Sphere => ColliderShape3D::Sphere { radius: c.radius },
        ColliderShape3DKind::Capsule => ColliderShape3D::Capsule {
            half_height: c.half_extents.y,
            radius: c.radius,
        },
    }
}

/// What one character's step did — returned so a test can assert on the
/// *decisions*, while the arms that matter assert on the WORLD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveOutcome {
    /// The entity's stable guid.
    pub guid: uuid::Uuid,
    /// The mode after the step.
    pub mode: MovementMode,
    /// The refusal recorded this step, if any.
    pub refusal: MovementRefusal,
    /// Whether the sweep ended grounded.
    pub grounded: bool,
    /// The landing the classifier decided this step, or
    /// [`LandingKind::None`] if nothing landed.
    pub landed: LandingKind,
}

/// **Advance every character's movement one fixed step.** The one door.
///
/// Runs after the hosts' Blueprint `Tick` (so intent set by gameplay is this
/// step's) and before the solver, which is the slot
/// `physics3d.move_and_slide` has always occupied.
///
/// Entities are processed in **`Guid` order**, not in ECS archetype order: the
/// result reaches `state_bytes` and a replay must not depend on the order bevy
/// happens to store components in.
pub fn step_character_movement(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
) -> Vec<MoveOutcome> {
    if !dt.is_finite() || dt <= 0.0 {
        return Vec::new();
    }
    // O(characters), not O(entities), and sorted — the walk and the ordering
    // rule both live in the model's one door (`movement_targets`'s doc states
    // the measured cost of getting this wrong).
    let targets: Vec<uuid::Uuid> = model::movement_targets(world);
    // **`OverlayRegistry`'s first caller** (P29.6) — Ruling 4's "open interned
    // id", which had zero callers anywhere in the tree from P29.3 until now and
    // was named as such by two audits.
    //
    // Interned over `targets`, which is **sorted by guid**, so the ids are a
    // function of the world's contents and not of the order a bevy archetype
    // walk happened to produce. That is the interning-determinism obligation
    // P29.2 and P29.4 both recorded as owed before an id could be handed out: an
    // id assigned by first-seen order is only safe if "first seen" is itself
    // deterministic, and here it is.
    let overlays = model::overlay_registry(world, &targets);
    let mut out = Vec::with_capacity(targets.len());
    for guid in &targets {
        // **The list is walked once and then passed down** (P29.4 audit, A8).
        // `try_mantle` needs it too — `IgnoreOnlyPawn` is "every other
        // character's collider" — and it used to ask for its own copy, which
        // made the falling catch (tried on *every* airborne step with input)
        // O(characters) per character per step. One walk, one sort, one
        // allocation, whoever reads it.
        if let Some(o) = step_one(world, bridge, *guid, dt, &targets, &overlays) {
            out.push(o);
        }
    }
    out
}

/// Whether the capsule may grow from `from_half` to `to_half` where it stands.
///
/// The probe is a **sweep of the CURRENT capsule upward** by twice the
/// half-height difference, and the arithmetic is worth stating because it is the
/// whole correctness argument: with the feet at `f`, the crouched capsule
/// occupies `[f, f + 2(h0 + r)]` and the standing one `[f, f + 2(h1 + r)]`, so
/// sweeping the crouched shape up by `2(h1 - h0)` covers exactly the union. If
/// that sweep is clear, the taller capsule fits — no approximation, no margin.
///
/// Shrinking is always allowed, and a sweep that starts already penetrating is a
/// refusal (the character is inside something; growing would make it worse).
fn has_clearance(
    bridge: &mut PhysicsBridge3D,
    centre: DVec3,
    radius: f64,
    from_half: f64,
    to_half: f64,
    exclude: &BTreeSet<ColliderId3D>,
) -> bool {
    // Finiteness first, comparison second: a NaN half-height must answer
    // "nothing to grow into" rather than slip through a negated comparison.
    if !to_half.is_finite() || !from_half.is_finite() || to_half <= from_half {
        return true;
    }
    let rise = 2.0 * (to_half - from_half);
    let shape = ColliderShape3D::Capsule {
        half_height: from_half.max(0.0),
        radius: radius.max(1e-3),
    };
    bridge
        .world_mut()
        .cast_shape(&shape, centre, DQuat::IDENTITY, DVec3::Y, rise, exclude)
        .is_none()
}

/// Where a clearance sweep starts and what it may ignore — the arguments
/// [`has_clearance`] needs that stay constant across one entity's step.
struct ClearanceProbe<'a> {
    centre: DVec3,
    radius: f64,
    is_capsule: bool,
    exclude: &'a BTreeSet<ColliderId3D>,
}

/// Ask the mode table for a transition, with the clearance question answered
/// against the world, and record the refusal if there is one.
///
/// **A refusal is a value**: this never fails, it answers with the mode now in
/// force. The counter it bumps is what a gate asserts on — "the world did not
/// change AND the character noticed" is a stronger claim than either half alone.
fn request(
    cm: &mut CharacterMovement,
    bridge: &mut PhysicsBridge3D,
    probe: &ClearanceProbe<'_>,
    to: MovementMode,
    condition: bool,
    refusal: &mut MovementRefusal,
) -> MovementMode {
    let from = cm.mode;
    let clearance = if probe.is_capsule {
        has_clearance(
            bridge,
            probe.centre,
            probe.radius,
            cm.half_height_for(from),
            cm.half_height_for(to),
            probe.exclude,
        )
    } else {
        true
    };
    let verdict = model::request_mode(from, to, clearance, condition);
    if verdict.refusal != MovementRefusal::None {
        *refusal = verdict.refusal;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
    }
    verdict.mode
}

/// How far above its authored placement a spawning character is lifted before
/// the settling sweep starts, metres.
///
/// It has to clear the mover's skin (`CharacterController3D::offset`, 2 cm by
/// default) with room to spare, or the sweep begins in the same penetrating
/// state the settle exists to escape.
const SETTLE_LIFT_M: f64 = 0.25;

/// How far below its authored placement the settle will look for ground, metres.
///
/// Bounded on purpose: "put my feet on the floor I am standing on" must not
/// become "teleport down to whatever is under this level". A character authored
/// higher than this falls, which is what an author who placed one in the air
/// meant.
const SETTLE_REACH_M: f64 = 0.35;

/// **Put an authored character's feet on the ground, once** (P29.6).
///
/// See the call site for the measurement. The rule in three clauses:
///
/// * it runs on the **first step only**, inside the same `seeded` latch that
///   takes the authored facing;
/// * it only ever **raises** — a character authored in the air falls;
/// * it is **bounded** by [`SETTLE_REACH_M`], so it settles onto the surface the
///   author placed the character on and never onto a distant one.
///
/// A sweep that still starts penetrating after the lift is left alone: something
/// is genuinely overlapping the character, and guessing where to put it is worse
/// than letting the mover's own sliding deal with it.
fn settle_on_spawn(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    position: &mut DVec3,
    half_height: f64,
    radius: f64,
    exclude: &BTreeSet<ColliderId3D>,
) {
    let offset = world
        .entity_of(guid)
        .and_then(|e| world.world().get::<CharacterController3D>(e).copied())
        .map(|c| c.offset.max(1e-4))
        .unwrap_or(0.02);
    let start = *position + DVec3::Y * SETTLE_LIFT_M;
    let Some(hit) = bridge.world_mut().cast_shape(
        &ColliderShape3D::Capsule {
            half_height,
            radius,
        },
        start,
        DQuat::IDENTITY,
        -DVec3::Y,
        SETTLE_LIFT_M + SETTLE_REACH_M,
        exclude,
    ) else {
        return;
    };
    if hit.started_penetrating {
        return;
    }
    let settled = start.y - hit.toi + offset;
    if settled > position.y {
        position.y = settled;
    }
}

/// The slope, in degrees from vertical, of a surface normal.
fn slope_deg(normal: DVec3) -> f64 {
    let n = normal.normalize_or_zero();
    if n == DVec3::ZERO {
        return 0.0;
    }
    inf_math::pacos64(n.y.clamp(-1.0, 1.0)).to_degrees()
}

#[allow(clippy::too_many_lines)]
fn step_one(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    dt: f64,
    characters: &[uuid::Uuid],
    overlays: &model::OverlayRegistry,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    let (mut cm, mut position, authored_yaw_deg, collider) = {
        let w = world.world();
        let cm = w.get::<CharacterMovement>(entity)?.clone();
        let t = w.get::<Transform>(entity)?;
        (
            cm,
            t.translation.to_dvec3(),
            t.rotation.y,
            w.get::<Collider3D>(entity).copied(),
        )
    };

    // ── 0. The authored facing, taken exactly once (audit A1).
    //
    //    Step 12 writes `body_yaw_deg` onto the entity's rotation every step, and
    //    nothing recomputes that value from the world — a character standing
    //    still has no velocity to face. So a runtime that starts at zero writes a
    //    zero over the level author's placement on the very first step: an NPC
    //    posted facing east faced north instead, and a whole squad snapped to one
    //    heading. Measured before the fix at 90 degrees in, 0 degrees out after a
    //    single idle step.
    //
    //    The aim goes with it, because the movement intent is expressed in the
    //    AIM frame: seeding only the drawn rotation would send a character
    //    authored facing east northward the moment it was told to walk forward.
    //
    //    Seeded rather than resynced: the smoother owns the yaw from here on (see
    //    `MovementRuntime::body_yaw_deg`), and re-reading the transform every step
    //    would make two authorities fight over one number.
    let seeded_this_step = !cm.runtime.seeded;
    if seeded_this_step {
        cm.runtime.seeded = true;
        let yaw = if authored_yaw_deg.is_finite() {
            authored_yaw_deg
        } else {
            0.0
        };
        cm.runtime.body_yaw_deg = yaw;
        cm.runtime.target_yaw_deg = yaw;
        cm.runtime.aim_yaw_deg = yaw;
    }
    let mut exclude: BTreeSet<ColliderId3D> = BTreeSet::new();
    if let Some(c) = bridge.collider_of(guid) {
        exclude.insert(c);
    }
    let radius = collider
        .filter(|c| c.shape_kind == ColliderShape3DKind::Capsule)
        .map(|c| c.radius)
        .unwrap_or(FALLBACK_RADIUS_M);
    let is_capsule = collider
        .map(|c| c.shape_kind == ColliderShape3DKind::Capsule)
        .unwrap_or(false);

    // ── 0a. **Settle the authored placement, once** (P29.6).
    //
    //    A level author puts a character's feet ON the floor, which is the only
    //    placement that looks right in a viewport — and the kinematic mover keeps
    //    a *skin* (`CharacterController3D::offset`, 2 cm by default), so a capsule
    //    authored at exactly `half + radius` starts INSIDE that band. rapier's
    //    character controller does not depenetrate: a sweep that begins in
    //    contact reports `started_penetrating` and the motion is allowed, so the
    //    small downward ground bias step 7 applies is never given back.
    //
    //    Measured on the shipped code before this: an idle character authored on
    //    the floor sank about **2 mm per fixed step** — 12 cm/s — while still
    //    reporting `grounded`, and a crouched one was through a 1 m floor in
    //    **1.6 seconds**. The same character spawned one skin-width clear settles
    //    at ground + offset and never moves again. No committed level carried a
    //    `CharacterMovement` until this wave, which is why nothing had seen it.
    //
    //    The correction is a **one-time placement**, inside the same `seeded`
    //    latch the authored facing uses and for the same reason: it is the
    //    author's number being taken once, not an authority the step keeps. It
    //    only ever raises — a character authored in the air must fall, not be
    //    magnetised to the ground — and its reach is bounded, so "settle onto the
    //    floor I am standing on" cannot become "teleport to whatever is below".
    if seeded_this_step && is_capsule {
        settle_on_spawn(
            world,
            bridge,
            guid,
            &mut position,
            cm.half_height_for(cm.mode),
            radius,
            &exclude,
        );
    }

    // ── 0b. A MANTLE owns the character outright while it runs (P29.4).
    //
    //    ALS sets `MOVE_None` and drives the actor transform directly, because
    //    its montage's displacement is not available as data. Ours does the same
    //    thing for a different reason: between the ledge probe and the ledge
    //    there is nothing to integrate — no gait, no ground to snap to, no
    //    velocity that means anything — and the placement is a warp whose
    //    endpoint is exact by construction rather than three hand-authored
    //    correction curves that converge on it.
    //
    //    Nothing below this point runs while `Mantle` is the mode.
    if cm.mode == MovementMode::Mantle {
        return step_mantle(world, bridge, guid, cm, dt, overlays);
    }

    // ── 0c. A RAGDOLL owns it just as completely (P29.4, clause 6), and for the
    //    same reason: the articulated bodies are the simulation now, and this
    //    step's velocity model has nothing to say about them. The bridge follows
    //    the pelvis with the capsule and hands the character back when it settles.
    if cm.mode == MovementMode::Ragdoll {
        return super::ragdoll_bridge::step_ragdoll(world, bridge, guid, cm, dt);
    }

    // ── 1. Aim. The look intent is a RATE (degrees per second), so integrating
    //    it here is frame-rate independent by construction — see
    //    `inf_input::InputState::axis_snapshot` for where that conversion is
    //    made and why it is made exactly once.
    let prev_aim = cm.runtime.aim_yaw_deg;
    cm.runtime.aim_yaw_deg = wrap_deg(cm.runtime.aim_yaw_deg + cm.runtime.intent_look_yaw_dps * dt);
    cm.runtime.aim_pitch_deg =
        (cm.runtime.aim_pitch_deg + cm.runtime.intent_look_pitch_dps * dt).clamp(-89.0, 89.0);
    cm.runtime.aim_yaw_rate_dps =
        (model::angle_delta_deg(cm.runtime.aim_yaw_deg, prev_aim) / dt).abs();
    // The aim toggle is a CONTROLLER action, so it only moves the rotation mode
    // on a character a controller is driving. Applied unconditionally it stomps
    // an authored one: an NPC placed in `Aiming` would be dragged to
    // `LookingDirection` on its first step by the absence of a key nobody was
    // pressing. Same argument as the gait below.
    if cm.player_controlled {
        if cm.runtime.want_aim {
            cm.rotation_mode = RotationMode::Aiming;
        } else if cm.rotation_mode == RotationMode::Aiming {
            cm.rotation_mode = RotationMode::LookingDirection;
        }
    }

    // ── 2. Water, through P20's door. `update_swim` advances the latch from the
    //    submerged fraction with P20's own hysteresis; nothing about that
    //    threshold is restated here.
    let swimming = bridge.update_swim(guid);
    let fraction = bridge.water_probe(guid).map(|p| p.fraction).unwrap_or(0.0);

    // ── 3. Timers. The get-up blend ages here rather than in the ragdoll
    //    branch, because a character getting up walks, turns and falls like any
    //    other — the blend is a WEIGHT, not a mode.
    cm.runtime.time_in_mode_s += dt;
    cm.runtime.time_since_land_s += dt;
    super::ragdoll_bridge::tick_get_up(world, guid, &mut cm, dt);

    // ── 4. Mode resolution: the single table, asked once per candidate.
    let previous_mode = cm.mode;
    let mut refusal = MovementRefusal::None;
    let mut probe = ClearanceProbe {
        centre: position,
        radius,
        is_capsule,
        exclude: &exclude,
    };

    let speed_planar = (cm.runtime.velocity.x * cm.runtime.velocity.x
        + cm.runtime.velocity.z * cm.runtime.velocity.z)
        .sqrt();

    // Water wins: it is a fact about where the character is, not a choice.
    if swimming {
        let want = if fraction >= water::SWIM_UNDER_FRACTION {
            MovementMode::SwimUnder
        } else {
            MovementMode::SwimSurface
        };
        cm.mode = request(&mut cm, bridge, &probe, want, true, &mut refusal);
    } else if cm.mode.is_swimming() {
        cm.mode = request(
            &mut cm,
            bridge,
            &probe,
            MovementMode::FallControlled,
            true,
            &mut refusal,
        );
    }

    if !cm.mode.is_swimming() {
        // Edge intents, in the order a controller resolves them.
        if cm.runtime.press_dive {
            let from_ground = cm.mode.is_grounded_family();
            cm.mode = request(
                &mut cm,
                bridge,
                &probe,
                MovementMode::Dive,
                from_ground,
                &mut refusal,
            );
            if cm.mode == MovementMode::Dive && previous_mode != MovementMode::Dive {
                let dir = model::rotate_from_frame(Vec2d::new(0.0, 1.0), cm.runtime.aim_yaw_deg);
                cm.runtime.velocity = Vec3d::new(
                    dir.x * cm.dive_speed_mps,
                    cm.dive_up_speed_mps,
                    dir.y * cm.dive_speed_mps,
                );
            }
        } else if cm.runtime.press_roll {
            let from_ground = cm.mode.is_grounded_family();
            cm.mode = request(
                &mut cm,
                bridge,
                &probe,
                MovementMode::Roll,
                from_ground,
                &mut refusal,
            );
        } else if cm.runtime.press_prone {
            let to = if cm.mode == MovementMode::Prone {
                MovementMode::Crouch
            } else {
                MovementMode::Prone
            };
            cm.mode = request(&mut cm, bridge, &probe, to, true, &mut refusal);
        } else if cm.runtime.press_crouch {
            // Sprint + crouch is a slide; anything else toggles the stance.
            let sliding = cm.mode == MovementMode::Grounded
                && cm.runtime.want_sprint
                && speed_planar >= cm.slide_entry_speed_mps;
            let to = if sliding {
                MovementMode::Slide
            } else if cm.mode == MovementMode::Crouch || cm.mode == MovementMode::Slide {
                MovementMode::Grounded
            } else {
                MovementMode::Crouch
            };
            cm.mode = request(&mut cm, bridge, &probe, to, true, &mut refusal);
        } else if cm.runtime.press_jump {
            // **The mantle is tried FIRST** (ALS trigger path 1): a jump at a
            // ledge with the stick forward is a climb, and only a jump that
            // finds no ledge is a jump. Ordering it the other way round would
            // make every mantle a jump that happened to end on a ledge.
            let wants = cm
                .runtime
                .intent_move
                .x
                .abs()
                .max(cm.runtime.intent_move.y.abs())
                >= MANTLE_MIN_INPUT;
            let mantled = wants
                && (cm.mode == MovementMode::Grounded || cm.mode == MovementMode::Crouch)
                && try_mantle(
                    &mut cm,
                    characters,
                    bridge,
                    &probe,
                    position,
                    radius,
                    &LedgeSettings::default(),
                    &mut refusal,
                );
            if mantled {
                // The mantle owns the character from here; the rest of this
                // step's decisions are not its to make.
            } else if cm.mode == MovementMode::Crouch || cm.mode == MovementMode::Prone {
                // Jump is also "stand up", and standing up under a table is the
                // catalogue's own example of a refusal.
                cm.mode = request(
                    &mut cm,
                    bridge,
                    &probe,
                    MovementMode::Grounded,
                    true,
                    &mut refusal,
                );
            } else if cm.mode == MovementMode::Grounded && cm.runtime.grounded {
                cm.mode = request(
                    &mut cm,
                    bridge,
                    &probe,
                    MovementMode::FallFree,
                    true,
                    &mut refusal,
                );
                if cm.mode == MovementMode::FallFree {
                    cm.runtime.velocity.y = cm.jump_speed_mps;
                }
            }
        }
        // **The falling catch** (ALS trigger path 3): every airborne step with
        // movement input reaches for a ledge, with the shorter falling settings.
        // Deliberately not gated on the jump edge — a character that runs off a
        // roof and holds forward catches the next one, which is the move the
        // donor's automatic path exists for.
        if cm.mode.is_falling()
            && cm.mode != MovementMode::Dive
            && cm
                .runtime
                .intent_move
                .x
                .abs()
                .max(cm.runtime.intent_move.y.abs())
                >= MANTLE_MIN_INPUT
        {
            try_mantle(
                &mut cm,
                characters,
                bridge,
                &probe,
                position,
                radius,
                &LedgeSettings::falling(),
                &mut refusal,
            );
        }
        // A slide runs out.
        if cm.mode == MovementMode::Slide && speed_planar < cm.slide_exit_speed_mps {
            cm.mode = request(
                &mut cm,
                bridge,
                &probe,
                MovementMode::Crouch,
                true,
                &mut refusal,
            );
        }
        // A roll ends.
        if cm.mode == MovementMode::Roll && cm.runtime.time_in_mode_s >= cm.roll_time_s {
            cm.mode = request(
                &mut cm,
                bridge,
                &probe,
                MovementMode::Crouch,
                true,
                &mut refusal,
            );
        }
    }
    // A mantle decided anywhere above takes over NOW rather than after a step
    // of gravity: the warp's first frame must start from where the probe was
    // taken, or the character drops before it climbs.
    if cm.mode == MovementMode::Mantle {
        cm.runtime.press_jump = false;
        cm.runtime.press_crouch = false;
        cm.runtime.press_prone = false;
        cm.runtime.press_roll = false;
        cm.runtime.press_dive = false;
        if let Some(mut slot) = world.world_mut().get_mut::<CharacterMovement>(entity) {
            *slot = cm;
        }
        return Some(MoveOutcome {
            guid,
            mode: MovementMode::Mantle,
            refusal,
            grounded: false,
            landed: LandingKind::None,
        });
    }
    // The edges are consumed whether or not they were honoured: an unconsumed
    // edge fires again next step off the same press, which is the P29.1 trigger
    // defect one crate over.
    cm.runtime.press_jump = false;
    cm.runtime.press_crouch = false;
    cm.runtime.press_prone = false;
    cm.runtime.press_roll = false;
    cm.runtime.press_dive = false;

    // ── 5. Capsule resize. The FEET stay planted: the capsule is centred on the
    //    transform, so a half-height change of d moves the centre by d.
    //
    //    `old_half` is the capsule the entity is ACTUALLY wearing, not the one
    //    its previous mode asks for, and on every step but the first those are
    //    the same number. On the first they need not be: the collider's
    //    half-height is authored independently of `stand_half_height_m`, the
    //    component wins (step 12 writes it), and the version that read
    //    `half_height_for(previous_mode)` skipped the compensation entirely when
    //    the mode had not changed — so a character authored with a 1.0 capsule
    //    and a 0.6 stand height had its collider shrunk on step one and its FEET
    //    lifted 40 cm, which is the one invariant this section names (audit A7).
    let mut half_height = cm.half_height_for(cm.mode);
    let worn_half = if is_capsule {
        collider.map(|c| c.half_extents.y)
    } else {
        None
    };
    if let Some(old_half) = worn_half {
        if (half_height - old_half).abs() > 1e-12 {
            position.y += half_height - old_half;
            probe.centre = position;
        }
    }
    if !is_capsule {
        half_height = collider.map(|c| c.half_extents.y).unwrap_or(half_height);
    }
    if cm.mode != previous_mode {
        cm.runtime.time_in_mode_s = 0.0;
    }

    // ── 6. Curves and gait.
    let mapped = model::mapped_speed(
        speed_planar,
        cm.walk_speed_mps,
        cm.run_speed_mps,
        cm.sprint_speed_mps,
    );
    // A controller's held keys pick the gait; anything else keeps the one it was
    // authored (or given by gameplay) with. Reading the keys unconditionally
    // would overwrite an authored `Walk` with `Run` every step, because "no
    // sprint and no walk held" is indistinguishable from "no controller".
    let desired_gait = if cm.player_controlled {
        if cm.runtime.want_sprint {
            Gait::Sprint
        } else if cm.runtime.want_walk {
            Gait::Walk
        } else {
            Gait::Run
        }
    } else {
        cm.gait
    };
    let move_input = cm.runtime.intent_move;
    let input_mag = (move_input.x * move_input.x + move_input.y * move_input.y)
        .sqrt()
        .min(1.0);
    let input_yaw = cm.runtime.aim_yaw_deg + model::planar_yaw_deg(move_input);
    let (allowed, actual) = model::resolve_gait(
        &cm,
        cm.mode,
        desired_gait,
        speed_planar,
        input_mag,
        input_yaw,
        cm.runtime.aim_yaw_deg,
    );
    cm.gait = desired_gait;
    cm.runtime.actual_gait = actual;
    let settings = model::settings_for(&cm, cm.mode, allowed, mapped, cm.runtime.aim_yaw_rate_dps);

    // ── 7. Integrate. The wish direction is the planar intent rotated OUT of the
    //    aim frame into world XZ.
    let wish = model::rotate_from_frame(move_input, cm.runtime.aim_yaw_deg);
    let planar_before = Vec2d::new(cm.runtime.velocity.x, cm.runtime.velocity.z);
    let vertical_before = cm.runtime.velocity.y;
    let has_input = input_mag > 1e-4;

    let (planar, vertical) = if cm.mode.is_swimming() {
        // The swim transform (P20) owns the vertical and the speed cap; the
        // integrator's job here is only to give it an intent to shape.
        let target = cm.speed_for(cm.mode, allowed);
        let p = model::integrate_planar_velocity(
            planar_before,
            wish,
            target,
            settings.acceleration_mps2,
            settings.braking_mps2,
            settings.friction,
            1.0,
            dt,
        );
        (p, cm.runtime.intent_vertical * cm.swim_surface_speed_mps)
    } else if cm.mode.is_falling() {
        let authority = if cm.mode == MovementMode::FallFree {
            cm.air_control
        } else {
            cm.air_control_reduced
        };
        let accel = (settings.acceleration_mps2 * authority).min(cm.air_accel_max_mps2);
        let p = model::integrate_planar_velocity(
            planar_before,
            wish,
            settings.target_speed_mps,
            accel,
            0.0,
            0.0,
            1.0,
            dt,
        );
        let v = (vertical_before - cm.gravity_mps2 * dt).max(-cm.terminal_velocity_mps);
        (p, v)
    } else {
        let friction_scale =
            model::landing_friction_scale(&cm, cm.runtime.time_since_land_s, has_input);
        let (accel, friction) = if cm.mode == MovementMode::Slide {
            // A slide does not accelerate; it decays against the slope.
            (
                0.0,
                model::slide_friction(&cm, slope_deg(cm.runtime.ground_normal.to_dvec3())),
            )
        } else if cm.mode == MovementMode::Roll {
            (0.0, settings.friction)
        } else {
            (settings.acceleration_mps2, settings.friction)
        };
        let target = if cm.mode == MovementMode::Slide || cm.mode == MovementMode::Roll {
            // No steering target: the impulse carries and the friction eats it.
            0.0
        } else {
            settings.target_speed_mps
        };
        let p = model::integrate_planar_velocity(
            planar_before,
            if target > 0.0 { wish } else { Vec2d::ZERO },
            target,
            accel,
            settings.braking_mps2,
            friction,
            friction_scale,
            dt,
        );
        // On the ground, gravity is a small downward bias so the sweep's
        // ground-snap has something to snap against; it is not integrated,
        // because a grounded character that accumulated fall speed would launch
        // the instant it stepped off a kerb.
        (p, -cm.gravity_mps2 * dt)
    };
    cm.runtime.velocity = Vec3d::new(planar.x, vertical, planar.y);

    // ── 7b. **Root motion** (P29.4, clause 2). A `Roll` and a `Dive` are
    //    root-motion driven by §13's own catalogue row, and until this wave a
    //    roll was a curve-decayed slide with a timer. The clip's displacement is
    //    published by the pose step (it has the clip resolver and the play-head)
    //    and consumed here, in the character's own facing frame, WITH the
    //    vertical — which is the half `root_delta` drops and a traversal needs.
    //
    //    One fixed step of latency, and it is structural rather than an
    //    oversight: the pose runs after the movement step in both hosts, so this
    //    reads what the pose published last step. Over any interval the total
    //    displacement is the same, shifted by 1/60 s.
    let root_motion = if matches!(cm.mode, MovementMode::Roll | MovementMode::Dive) {
        inf_ecs::anim_bridge::anim_root_motion(world, guid)
    } else {
        None
    };
    let root_world = match root_motion {
        Some(rm) if !rm.is_zero() => {
            cm.runtime.body_yaw_deg =
                wrap_deg(cm.runtime.body_yaw_deg + rm.yaw.to_degrees() as f64);
            inf_anim::root_delta_world_3d(cm.runtime.body_yaw_deg, rm.translation)
        }
        _ => DVec3::ZERO,
    };

    // ── 8. Move. `apply_swim_motion` is the identity when not swimming, which
    //    is why it can be unconditional — the same call `move_and_slide` makes.
    // **`deliberate`** (P29.6): the character step is the one place in the
    // engine that can tell a player asking to dive from a body integrating
    // gravity, and `apply_swim_motion` cannot. Without the distinction the float
    // balance wins every argument and `SwimUnder` is a mode no input reaches.
    let motion = bridge.apply_swim_motion_where(
        guid,
        DVec3::new(planar.x * dt, vertical * dt, planar.y * dt) + root_world,
        dt,
        cm.mode.is_swimming() && cm.runtime.intent_vertical < 0.0,
    );
    // The mover is rebuilt with THIS step's capsule, so a crouch takes effect on
    // the step it is decided rather than on the next bridge sync.
    let mover = mover_for_with_capsule(world, guid, is_capsule.then_some((half_height, radius)));
    let was_grounded = cm.runtime.grounded;
    let result =
        bridge
            .world_mut()
            .move_character(&mover, position, motion, exclude.iter().next().copied());
    position += result.translation;
    probe.centre = position;
    cm.runtime.grounded = result.grounded;

    // ── 9. The ground normal, for the slide curve and for P29.4. rapier reports
    //    "grounded" but not what it stood on, so this is a short downward sweep.
    cm.runtime.ground_normal = Vec3d::new(0.0, 1.0, 0.0);
    if result.grounded && is_capsule {
        let probe_len = (half_height + radius) * 0.25 + 0.05;
        if let Some(hit) = bridge.world_mut().cast_shape(
            &ColliderShape3D::Sphere {
                radius: radius * 0.9,
            },
            position,
            DQuat::IDENTITY,
            -DVec3::Y,
            probe_len + half_height,
            &exclude,
        ) {
            if !hit.started_penetrating {
                cm.runtime.ground_normal = Vec3d::from_dvec3(hit.normal);
            }
        }
    }

    // ── 9b. **Land prediction** (P29.4, clause 4): the classifier's inputs,
    //    *before* the touch.
    //
    //    A capsule swept along the character's own velocity, not straight down —
    //    a character thrown off a ledge lands in front of itself, and a downward
    //    ray would predict the void it is currently over. The sweep answers with
    //    the speed the character WILL arrive at (it adds the gravity it has yet
    //    to pick up), so `predicted_landing` is the same verdict step 10 will
    //    reach, arrived at early enough for an animation to prepare for it.
    //
    //    Cleared first, unconditionally: a prediction is a statement about this
    //    step, and a grounded character predicting a landing is the stale-answer
    //    defect `PoseStoreRes`'s rule 4 exists to prevent.
    cm.runtime.land_alpha = 0.0;
    cm.runtime.land_predicted_mps = 0.0;
    cm.runtime.predicted_landing = LandingKind::None;
    if !result.grounded && is_capsule {
        if let Some(p) = traversal::predict_landing(
            bridge.world_mut(),
            position,
            cm.runtime.velocity.to_dvec3(),
            radius,
            half_height,
            cm.slope_limit_deg,
            cm.gravity_mps2,
            &exclude,
        ) {
            cm.runtime.land_alpha = p.alpha;
            cm.runtime.land_predicted_mps = p.impact_mps;
            cm.runtime.predicted_landing = model::classify_landing(&cm, p.impact_mps, has_input);
        }
    }

    // ── 10. Landing and take-off, keyed to IMPACT SPEED.
    let mut landed = LandingKind::None;
    if result.grounded {
        if !was_grounded || cm.mode.is_falling() {
            let impact = (-vertical_before).max(0.0);
            let kind = model::classify_landing(&cm, impact, has_input);
            cm.runtime.land_impact_mps = impact;
            cm.runtime.landing = kind;
            cm.runtime.time_since_land_s = 0.0;
            landed = kind;
            // A dive lands into **prone or a roll**, and the classifier chooses
            // — not the animation (§13's catalogue row, verbatim). A ragdoll
            // verdict is recorded on the runtime and lands like a hard landing:
            // P29.4 owns the ragdoll, and `request_mode` refuses it by name.
            // **A ragdoll verdict is now a ragdoll** (P29.3 recorded that it
            // "lands like a hard landing: P29.4 owns the ragdoll"). The bodies
            // are seeded with the velocity the character hit at, so a fall that
            // breaks a character carries its momentum into the tumble.
            if kind == LandingKind::Ragdoll {
                cm.runtime.velocity.y = -impact;
                if super::ragdoll_bridge::begin(&mut cm) {
                    cm.runtime.press_jump = false;
                    cm.runtime.press_crouch = false;
                    cm.runtime.press_prone = false;
                    cm.runtime.press_roll = false;
                    cm.runtime.press_dive = false;
                    let mode = cm.mode;
                    if let Some(mut slot) = world.world_mut().get_mut::<CharacterMovement>(entity) {
                        *slot = cm;
                    }
                    inf_ecs::anim_bridge::request_ragdoll_rig(world, guid);
                    inf_ecs::anim_bridge::set_anim_trigger(
                        world,
                        guid,
                        super::ragdoll_bridge::TRIGGER_RAGDOLL,
                    );
                    return Some(MoveOutcome {
                        guid,
                        mode,
                        refusal,
                        grounded: true,
                        landed: kind,
                    });
                }
            }
            let to = match kind {
                LandingKind::Roll => MovementMode::Roll,
                _ if previous_mode == MovementMode::Dive => MovementMode::Prone,
                _ => MovementMode::Grounded,
            };
            if cm.mode.is_falling() {
                let before = cm.mode;
                cm.mode = request(&mut cm, bridge, &probe, to, true, &mut refusal);
                if cm.mode != before {
                    let old_half = cm.half_height_for(before);
                    if is_capsule {
                        position.y += cm.half_height_for(cm.mode) - old_half;
                    }
                    cm.runtime.time_in_mode_s = 0.0;
                }
            }
        }
        // Standing on the ground, downward velocity is spent.
        if cm.runtime.velocity.y < 0.0 {
            cm.runtime.velocity.y = 0.0;
        }
    } else if cm.mode.is_grounded_family() {
        // Not grounded and not already falling: a controlled fall, not a jump.
        //
        // The condition deliberately does NOT read `was_grounded`. The first
        // version did, and it meant a character that had never been grounded --
        // one spawned in the air, or one whose floor was deleted -- stayed in
        // `Grounded` for ever, integrating the small downward ground bias
        // instead of gravity. It fell at 16 cm/s and landed reporting an impact
        // of 0.16 m/s from any height, which is a landing classifier that always
        // says "soft". Found by the classifier arm, which measures the impact
        // rather than trusting it.
        cm.mode = request(
            &mut cm,
            bridge,
            &probe,
            MovementMode::FallControlled,
            true,
            &mut refusal,
        );
        cm.runtime.time_in_mode_s = 0.0;
    }

    // ── 10b. Body rotation — ALS's two-stage smoother, and the one consumer of
    //    the rotation-rate curve and of `AimYawRate`.
    //
    //    `VelocityDirection` faces where the body is going, the other two face
    //    where it is looking; standing still while aiming clamps the body to
    //    within 100 degrees of the aim rather than turning it, which is what
    //    keeps an idle character from spinning under the camera.
    let planar_now = Vec2d::new(cm.runtime.velocity.x, cm.runtime.velocity.z);
    let moving = (planar_now.x * planar_now.x + planar_now.y * planar_now.y).sqrt() > 0.1;
    let (goal, actor_interp) = match cm.rotation_mode {
        RotationMode::VelocityDirection if moving => (model::planar_yaw_deg(planar_now), 15.0),
        RotationMode::Aiming => (cm.runtime.aim_yaw_deg, 20.0),
        RotationMode::LookingDirection => (cm.runtime.aim_yaw_deg, 15.0),
        _ => (cm.runtime.body_yaw_deg, 15.0),
    };
    // Moving turns the body; standing still does NOT — it clamps, and only
    // while aiming. The first draft of this branch read
    // `moving || rotation_mode != VelocityDirection`, which made the clamp below
    // unreachable and dragged an idle aiming character round under its own
    // camera: exactly the failure `LimitRotation` exists to prevent, introduced
    // by the line that was meant to call it. Found by the arm that measures the
    // TRANSFORM.
    if moving {
        let (t, b) = model::smooth_rotation(
            cm.runtime.target_yaw_deg,
            cm.runtime.body_yaw_deg,
            goal,
            settings.rotation_rate_dps,
            actor_interp,
            dt,
        );
        cm.runtime.target_yaw_deg = t;
        cm.runtime.body_yaw_deg = b;
        // A body that is moving is not standing still, and a turn in place that
        // survived into a walk would fight the velocity it is supposed to face.
        cm.runtime.turning_in_place = false;
        cm.runtime.turn_delay_s = 0.0;
        cm.runtime.rotate_left = false;
        cm.runtime.rotate_right = false;
        cm.runtime.rotate_rate = 1.0;
    } else {
        // ── Standing still: **rotate in place** while aiming, **turn in place**
        //    while looking (P29.4, clause 7). ALS gates them on exactly this
        //    split — aiming/first-person rotates, third-person looking turns —
        //    and the two are different mechanisms, not two speeds of one.
        let aim_delta = model::angle_delta_deg(cm.runtime.aim_yaw_deg, cm.runtime.body_yaw_deg);
        let aiming = cm.rotation_mode == RotationMode::Aiming;
        let (rl, rr, rate) = model::rotate_in_place(aim_delta, cm.runtime.aim_yaw_rate_dps);
        cm.runtime.rotate_left = aiming && rl;
        cm.runtime.rotate_right = aiming && rr;
        cm.runtime.rotate_rate = if aiming { rate } else { 1.0 };
        if aiming {
            // The clamp, unchanged: an idle aiming character is held within a
            // cone of its own camera rather than dragged round by it.
            cm.runtime.body_yaw_deg = model::limit_rotation(
                cm.runtime.body_yaw_deg,
                cm.runtime.aim_yaw_deg,
                -AIM_BODY_LIMIT_DEG,
                AIM_BODY_LIMIT_DEG,
                AIM_BODY_LIMIT_INTERP,
                dt,
            );
            cm.runtime.turning_in_place = false;
            cm.runtime.turn_delay_s = 0.0;
        } else if cm.rotation_mode == RotationMode::LookingDirection {
            if cm.runtime.turning_in_place {
                // **Orientation warping, in its simplest consumer.** The target
                // is a runtime angle and the turn is rate-bounded onto it, which
                // is what supersedes ALS's `RotationAmount / 30 fps` scalar
                // (§13's [SUPERSEDE]): no authored-at-30-fps curve enters the
                // arithmetic, so a 63-degree turn is 63 degrees rather than a
                // 90-degree animation rescaled by a number in a curve asset.
                let goal = cm.runtime.turn_target_yaw_deg;
                let (t, b) = model::smooth_rotation(
                    cm.runtime.target_yaw_deg,
                    cm.runtime.body_yaw_deg,
                    goal,
                    settings.rotation_rate_dps.max(TURN_MIN_RATE_DPS),
                    actor_interp,
                    dt,
                );
                cm.runtime.target_yaw_deg = t;
                cm.runtime.body_yaw_deg = b;
                if model::angle_delta_deg(goal, b).abs() <= TURN_SETTLE_DEG {
                    cm.runtime.body_yaw_deg = goal;
                    cm.runtime.target_yaw_deg = goal;
                    cm.runtime.turning_in_place = false;
                    cm.runtime.turn_delay_s = 0.0;
                }
            } else if model::turn_in_place_ready(aim_delta, cm.runtime.aim_yaw_rate_dps) {
                // The delay accumulates only while BOTH gates hold, so a player
                // who is still moving the camera never starts a turn.
                cm.runtime.turn_delay_s += dt;
                if cm.runtime.turn_delay_s > model::turn_in_place_delay_s(aim_delta) {
                    cm.runtime.turning_in_place = true;
                    cm.runtime.turn_target_yaw_deg = cm.runtime.aim_yaw_deg;
                    cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;
                }
            } else {
                cm.runtime.turn_delay_s = 0.0;
            }
        }
    }

    // ── 11. Derived outputs for the P29.4 bridge.
    let planar_after = Vec2d::new(cm.runtime.velocity.x, cm.runtime.velocity.z);
    let speed_after = (planar_after.x * planar_after.x + planar_after.y * planar_after.y).sqrt();
    cm.runtime.mapped_speed = model::mapped_speed(
        speed_after,
        cm.walk_speed_mps,
        cm.run_speed_mps,
        cm.sprint_speed_mps,
    );
    cm.runtime.gait_scalar = model::gait_scalar(cm.runtime.mapped_speed);
    cm.runtime.stride_blend = model::stride_blend(cm.runtime.mapped_speed);
    cm.runtime.walk_run_blend = model::walk_run_blend(cm.runtime.mapped_speed);
    if speed_after > 1e-4 {
        let rel =
            model::angle_delta_deg(model::planar_yaw_deg(planar_after), cm.runtime.aim_yaw_deg);
        cm.runtime.direction = model::quadrant(rel, cm.runtime.direction);
    }
    let accel_world = Vec2d::new(
        (planar_after.x - planar_before.x) / dt,
        (planar_after.y - planar_before.y) / dt,
    );
    let decelerating = speed_after < speed_planar;
    cm.runtime.relative_accel = model::relative_acceleration(
        accel_world,
        cm.runtime.body_yaw_deg,
        settings.acceleration_mps2,
        settings.braking_mps2,
        decelerating,
    );
    cm.runtime.lean = cm.runtime.relative_accel;
    // **Aim offsets** (P29.4, clause 7), over `Mask_AimOffset` — a `.inf_anim` v2
    // curve channel the pose step publishes for whatever state the machine is in.
    // A state that wants no aim offset authors a 1 and gets none, which is ALS's
    // `EnableAimOffset = lerp(1, 0, curve)` read the way it is written.
    let aim_mask = inf_ecs::anim_bridge::anim_curve(
        world,
        guid,
        inf_anim::channels::als::MASK_AIM_OFFSET,
        0.0,
    ) as f64;
    cm.runtime.aim_sweep = model::aim_sweep(cm.runtime.aim_pitch_deg);
    cm.runtime.aim_offset_weight = (1.0 - aim_mask.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    cm.runtime.spine_yaw_deg = model::spine_yaw_deg(
        model::angle_delta_deg(cm.runtime.aim_yaw_deg, cm.runtime.body_yaw_deg),
        aim_mask,
    );
    // ── 11b. **Foot IK and foot lock** (P29.4, clause 5). The character's own
    //    ground plane goes with it: the offset arithmetic is a comparison
    //    between the ground under a foot and the ground under the BODY, and the
    //    body's is not derivable from a foot.
    let ground_plane_y = position.y - (half_height + radius);
    step_feet(
        world,
        bridge,
        &mut cm,
        guid,
        radius,
        ground_plane_y,
        &exclude,
        dt,
    );
    cm.runtime.refusal = refusal;

    // ── 12. Write the world back: the component, the transform, the capsule,
    //    and the physics body if there is one.
    let mode = cm.mode;
    let grounded = cm.runtime.grounded;
    let body_yaw = cm.runtime.body_yaw_deg;
    {
        let w = world.world_mut();
        if let Some(mut t) = w.get_mut::<Transform>(entity) {
            t.translation.x = position.x;
            t.translation.y = position.y;
            t.translation.z = position.z;
            // Yaw only: a character's pitch and roll belong to its pose, not to
            // its body. (`Transform::rotation` is euler DEGREES, YXZ.)
            t.rotation.y = body_yaw;
        }
        if is_capsule {
            if let Some(mut c) = w.get_mut::<Collider3D>(entity) {
                c.half_extents.y = half_height;
            }
        }
        if let Some(mut slot) = w.get_mut::<CharacterMovement>(entity) {
            *slot = cm.clone();
        }
    }
    // ── 12b. **Publish this character's state into its own machine** (P29.6).
    //
    //    ALS's AnimInstance copies seventeen fields off the character every
    //    tick; this is the same idea through the one Ring-0 overlay the `anim.*`
    //    kit writes into, so a wizard-generated character animates with **no
    //    script at all**. Before it, `speed` was a parameter every generated and
    //    proposed machine gated on and nothing in the engine ever set.
    //
    //    A Blueprint still wins: its `anim.set_param` runs in the Tick pass,
    //    which is after this in both hosts' fixed steps.
    //
    //    Costs one map lookup on a character with no machine, which is every
    //    character in every committed level before this wave.
    let overlay = overlays.id_of(&cm.overlay);
    inf_ecs::anim_bridge::publish_character_params(world, guid, &cm, overlay);
    if let Some(body) = bridge.body_of(guid) {
        bridge.world_mut().set_body_translation(body, position);
    }
    world.mark_dirty();

    Some(MoveOutcome {
        guid,
        mode,
        refusal,
        grounded,
        landed,
    })
}

// ── the mantle (P29.4, clause 4) ────────────────────────────────────────────

/// **Try to enter a mantle**, reaching for a ledge in the character's own facing
/// direction with `settings`.
///
/// Returns whether the mode changed. Everything about the refusal is a value:
/// no ledge, no room, a ledge that is a floor, a ledge that is a moving crate —
/// each is a `None` from the probe and a `false` from here, and the character
/// does whatever it was going to do instead.
///
/// The **exclusion set is `IgnoreOnlyPawn`**: the character's own collider is
/// already in `probe.exclude`, and every other character's is added here, so a
/// crowd cannot be climbed. That is the port of ALS's collision profile, made
/// out of the thing this engine actually has.
#[allow(clippy::too_many_arguments)]
fn try_mantle(
    cm: &mut CharacterMovement,
    characters: &[uuid::Uuid],
    bridge: &mut PhysicsBridge3D,
    probe: &ClearanceProbe<'_>,
    position: DVec3,
    radius: f64,
    settings: &LedgeSettings,
    refusal: &mut MovementRefusal,
) -> bool {
    let half = cm.half_height_for(MovementMode::Grounded);
    // The capsule is centred on the transform, so the feet are one half-height
    // plus one radius below it.
    let feet = position - DVec3::Y * (cm.half_height_for(cm.mode) + radius);
    let forward = model::rotate_from_frame(Vec2d::new(0.0, 1.0), cm.runtime.body_yaw_deg);
    // `IgnoreOnlyPawn`, made out of what this engine has: every character's
    // collider, off the list `step_character_movement` already walked once this
    // step, so a crowd cannot be climbed and the walk is not repeated per
    // character (P29.4 audit, A8 — the falling catch runs this on every airborne
    // step, so asking for a fresh `movement_targets` here was O(characters) per
    // character per fixed step).
    let mut exclude = probe.exclude.clone();
    for other in characters {
        if let Some(c) = bridge.collider_of(*other) {
            exclude.insert(c);
        }
    }
    let Some(ledge) = traversal::probe_ledge(
        bridge.world_mut(),
        feet,
        DVec3::new(forward.x, 0.0, forward.y),
        radius,
        half,
        cm.slope_limit_deg,
        settings,
        &exclude,
    ) else {
        return false;
    };
    let from = cm.mode;
    let verdict = model::request_mode(from, MovementMode::Mantle, true, true);
    if verdict.refusal != MovementRefusal::None {
        *refusal = verdict.refusal;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        return false;
    }
    // **ALS's height remap**, kept as the reference behaviour: where in the
    // traversal clip to start and how fast to play it, so a 0.9 m ledge and a
    // 1.3 m one share one animation. The remap's bands are the settings' own, so
    // a project that widens what it can climb does not have to retune the clip.
    let remap = inf_anim::HeightRemap {
        low_height_m: settings.min_height_m,
        high_height_m: settings.max_height_m,
        ..inf_anim::HeightRemap::default()
    };
    let (clip_start_s, play_rate) = remap.resolve(ledge.height_m);
    cm.mode = MovementMode::Mantle;
    cm.runtime.velocity = Vec3d::ZERO;
    cm.runtime.time_in_mode_s = 0.0;
    // The feet start where they ARE and end on the ledge; the warp is expressed
    // in feet rather than in capsule centres so a stance change on the way in
    // cannot move the target.
    cm.runtime.mantle = MantleState {
        active: true,
        start: Vec3d::from_dvec3(feet),
        start_yaw_deg: cm.runtime.body_yaw_deg,
        target: Vec3d::from_dvec3(ledge.feet),
        target_yaw_deg: ledge.yaw_deg,
        elapsed_s: 0.0,
        duration_s: model::mantle_duration_s(
            ledge.height_m,
            settings.min_height_m,
            settings.max_height_m,
            play_rate,
        ),
        height_m: ledge.height_m,
        high: ledge.high,
        clip_start_s,
        play_rate,
    };
    true
}

/// **Advance a mantle one fixed step**, and hand the character back when it ends.
///
/// # The placement, in one sentence
///
/// [`inf_anim::warp_offset`] scales the traversal motion delivered so far onto
/// the runtime target, so consuming the whole window lands **exactly** on the
/// ledge — no residual, no ease that merely converges.
///
/// # What drives the progress — and the two answers, named
///
/// The **clock** is always the parameter: a mantle's duration comes from the
/// ledge height and the play rate, so `m.alpha()` is what says how far through
/// the climb the character is and `done` is its own.
///
/// The **shape** is the clip's, when there is one. P29.4 shipped this call with
/// a synthesised pair — `total × progress` and `total`, a scale of exactly one,
/// so the warp degenerated to [`inf_anim::warp_ease`] and its ledger said so.
/// P29.5's import derivation bakes a root-motion track onto a traversal clip and
/// the pose step resamples it into an
/// [`inf_ecs::TraversalArc`](inf_ecs::anim_bridge::TraversalArc), so the pair is
/// now the clip's own arc read at the mantle's progress and the warp scales that
/// arc onto the ledge. At `alpha == 1` the arc's delivered *is* its total, so the
/// endpoint stays exact by construction and P29.4's landing measurement is
/// unchanged.
///
/// The arc is sampled at the **raw** clock rather than at the eased one: the
/// clip already carries its own ease, and easing an eased arc would smooth the
/// animator's timing away. The ease survives as the alpha of `warp_offset`'s
/// additive half, which is the axis the clip has no shape along — where a
/// smoothstep is exactly what is wanted. With no arc, both are the eased clock,
/// which is P29.4's behaviour byte for byte.
fn step_mantle(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    mut cm: CharacterMovement,
    dt: f64,
    overlays: &model::OverlayRegistry,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    cm.runtime.time_in_mode_s += dt;
    cm.runtime.mantle.elapsed_s += dt;
    // A mantle has no velocity: it is on rails between two known transforms, and
    // leaving a stale one here would launch the character the instant it lands.
    cm.runtime.velocity = Vec3d::ZERO;
    cm.runtime.grounded = false;

    let m = cm.runtime.mantle;
    let progress = inf_anim::warp_ease(m.alpha());
    let start = m.start.to_dvec3();
    let target_offset = m.target.to_dvec3() - start;
    // The offset expressed in the frame the window opened in — the clip's own
    // space, which is what a root-motion track would be in.
    let local = model::rotate_into_frame(
        Vec2d::new(target_offset.x, target_offset.z),
        m.start_yaw_deg,
    );
    let clock_total = glam::Vec3::new(local.x as f32, target_offset.y as f32, local.y as f32);
    let yaw_delta = model::angle_delta_deg(m.target_yaw_deg, m.start_yaw_deg);
    // The clip's arc if the machine is playing a derived traversal one-shot,
    // the clock's own ramp if it is not. See the fn docs for why the arc is read
    // at `m.alpha()` and not at `progress`.
    //
    // Both branches answer the yaw in DEGREES: the arc's is radians (the unit
    // `RootMotionTrack` documents) and is converted here, once, rather than at
    // the call site below.
    let arc = inf_ecs::traversal_arc(world, guid).filter(|a| a.is_usable());
    let (delivered, total, yaw_now_deg, yaw_total_deg) = match arc {
        Some(a) => {
            let (d, y) = a.at(m.alpha());
            let (t, ty) = a.total();
            (d, t, (y as f64).to_degrees(), (ty as f64).to_degrees())
        }
        None => (
            clock_total * progress as f32,
            clock_total,
            yaw_delta * progress,
            yaw_delta,
        ),
    };
    let feet =
        start + inf_anim::warp_offset(m.start_yaw_deg, delivered, total, target_offset, progress);

    cm.runtime.body_yaw_deg = wrap_deg(
        m.start_yaw_deg + inf_anim::warp_yaw_deg(yaw_now_deg, yaw_total_deg, yaw_delta, progress),
    );
    cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;

    // The capsule is standing for the whole climb, so the centre is a fixed
    // offset above the warped feet.
    let (radius, is_capsule) = {
        let w = world.world();
        match w.get::<Collider3D>(entity) {
            Some(c) if c.shape_kind == ColliderShape3DKind::Capsule => (c.radius, true),
            _ => (FALLBACK_RADIUS_M, false),
        }
    };
    let half = cm.half_height_for(MovementMode::Grounded);
    let position = feet + DVec3::Y * (half + radius);

    let mut refusal = MovementRefusal::None;
    let done = m.alpha() >= 1.0;
    if done {
        cm.runtime.mantle.active = false;
        let verdict = model::request_mode(MovementMode::Mantle, MovementMode::Grounded, true, true);
        cm.mode = verdict.mode;
        refusal = verdict.refusal;
        if verdict.refusal != MovementRefusal::None {
            cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        }
        cm.runtime.grounded = true;
        cm.runtime.time_in_mode_s = 0.0;
        // A mantle is not a landing: the character arrived under its own power at
        // a known transform, so the classifier has nothing to classify and the
        // post-landing friction override must not fire.
        cm.runtime.land_alpha = 0.0;
        cm.runtime.land_predicted_mps = 0.0;
        cm.runtime.predicted_landing = LandingKind::None;
    }
    let mode = cm.mode;
    let grounded = cm.runtime.grounded;
    let body_yaw = cm.runtime.body_yaw_deg;
    {
        let w = world.world_mut();
        if let Some(mut t) = w.get_mut::<Transform>(entity) {
            t.translation.x = position.x;
            t.translation.y = position.y;
            t.translation.z = position.z;
            t.rotation.y = body_yaw;
        }
        if is_capsule {
            if let Some(mut c) = w.get_mut::<Collider3D>(entity) {
                c.half_extents.y = half;
            }
        }
        if let Some(mut slot) = w.get_mut::<CharacterMovement>(entity) {
            *slot = cm.clone();
        }
    }
    // ── 12b. **Publish this character's state into its own machine** (P29.6).
    //
    //    ALS's AnimInstance copies seventeen fields off the character every
    //    tick; this is the same idea through the one Ring-0 overlay the `anim.*`
    //    kit writes into, so a wizard-generated character animates with **no
    //    script at all**. Before it, `speed` was a parameter every generated and
    //    proposed machine gated on and nothing in the engine ever set.
    //
    //    A Blueprint still wins: its `anim.set_param` runs in the Tick pass,
    //    which is after this in both hosts' fixed steps.
    //
    //    Costs one map lookup on a character with no machine, which is every
    //    character in every committed level before this wave.
    let overlay = overlays.id_of(&cm.overlay);
    inf_ecs::anim_bridge::publish_character_params(world, guid, &cm, overlay);
    if let Some(body) = bridge.body_of(guid) {
        bridge.world_mut().set_body_translation(body, position);
    }
    world.mark_dirty();
    Some(MoveOutcome {
        guid,
        mode,
        refusal,
        grounded,
        landed: LandingKind::None,
    })
}

/// **Foot IK and foot locking, the half that needs a world** (P29.4, clause 5).
///
/// The pure half is [`inf_anim::foot`]: the ±50/45 cm trace envelope in metres,
/// the ground-offset arithmetic, and the lock rule that may only engage or
/// release and never blend in. This is where those meet a physics world.
///
/// Four steps per foot:
///
/// 1. **Where is it?** From the bridge, which is where the pose step put it —
///    one fixed step ago, because the pose runs after this one in both hosts.
/// 2. **Where is the ground under it?** A short downward sweep across ALS's own
///    envelope, converted once at [`inf_anim::TRACE_ABOVE_M`].
/// 3. **Is it planted?** The clip's `FootLock_L/R` channel says so, gated by
///    `Enable_FootIK_L/R`, and a body that is turning breaks the lock (a pinned
///    foot under a rotating hip points the wrong way).
/// 4. **What is the slide?** The number this wave's gate holds, in **metres**,
///    recorded on the runtime whether or not anything is watching.
///
/// Everything is inert on a character whose clips carry no curve channels: the
/// gate curve reads its fallback of zero, no lock engages, no goal is published
/// and the pose is exactly what the machine produced. That is what keeps every
/// committed sample byte-identical.
#[allow(clippy::too_many_arguments)]
fn step_feet(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    cm: &mut CharacterMovement,
    guid: uuid::Uuid,
    radius: f64,
    ground_plane_y: f64,
    exclude: &BTreeSet<ColliderId3D>,
    dt: f64,
) {
    use inf_anim::channels::als;
    let Some(feet) = inf_ecs::anim_bridge::feet_of(world, guid) else {
        // No rig, or a rig with no feet: release whatever was held, so a
        // character that loses its skeleton does not leave a foot pinned to a
        // spot on the floor.
        cm.runtime.foot_lock_l.release();
        cm.runtime.foot_lock_r.release();
        cm.runtime.foot_slide_l_m = 0.0;
        cm.runtime.foot_slide_r_m = 0.0;
        cm.runtime.pelvis_offset = Vec3d::ZERO;
        return;
    };
    // A turn breaks a lock. `RotationAmount` is the donor's channel; ours is the
    // body's own measured turn this step, which is the same quantity without the
    // 30 fps authoring convention in it.
    let turning = (cm.runtime.aim_yaw_rate_dps * dt).abs() > 0.05
        || model::angle_delta_deg(cm.runtime.body_yaw_deg, cm.runtime.target_yaw_deg).abs() > 0.05;

    let gates = [
        (als::ENABLE_FOOT_IK_L, als::FOOT_LOCK_L),
        (als::ENABLE_FOOT_IK_R, als::FOOT_LOCK_R),
    ];
    let mut goals: [Option<inf_ecs::anim_bridge::FootGoal>; 2] = [None, None];
    let mut offsets = [glam::Vec3::ZERO, glam::Vec3::ZERO];
    let mut enables = [0.0f32, 0.0f32];
    for (side, (enable_name, lock_name)) in gates.iter().enumerate() {
        let Some(state) = feet[side] else { continue };
        let enable = inf_ecs::anim_bridge::anim_curve(world, guid, enable_name, 0.0);
        let lock_curve = inf_ecs::anim_bridge::anim_curve(world, guid, lock_name, 0.0);
        enables[side] = enable;
        let posed = state.world.to_dvec3();

        // The lock first, so a foot that is planted is measured against where it
        // was planted rather than against where it has been dragged to.
        let lock = if side == 0 {
            &mut cm.runtime.foot_lock_l
        } else {
            &mut cm.runtime.foot_lock_r
        };
        lock.update(
            enable,
            lock_curve,
            turning,
            glam::Vec3::new(posed.x as f32, posed.y as f32, posed.z as f32),
            cm.runtime.body_yaw_deg,
        );
        let slide = lock.slide_m(glam::Vec3::new(
            posed.x as f32,
            posed.y as f32,
            posed.z as f32,
        ));
        let held = lock.resolve(glam::Vec3::new(
            posed.x as f32,
            posed.y as f32,
            posed.z as f32,
        ));
        let drawn = Vec3d::new(held.x as f64, held.y as f64, held.z as f64);
        if side == 0 {
            cm.runtime.foot_slide_l_m = slide;
            cm.runtime.foot_world_l = drawn;
        } else {
            cm.runtime.foot_slide_r_m = slide;
            cm.runtime.foot_world_r = drawn;
        }
        if enable.is_nan() || enable <= 0.0 {
            continue;
        }

        // The ground under it, across ALS's envelope.
        let from = DVec3::new(
            held.x as f64,
            held.y as f64 + inf_anim::TRACE_ABOVE_M,
            held.z as f64,
        );
        let span = inf_anim::TRACE_ABOVE_M + inf_anim::TRACE_BELOW_M;
        let hit = bridge.world_mut().cast_shape(
            &ColliderShape3D::Sphere {
                radius: (radius * 0.25).max(0.02),
            },
            from,
            DQuat::IDENTITY,
            -DVec3::Y,
            span,
            exclude,
        );
        let Some(hit) = hit else { continue };
        if hit.started_penetrating || !super::traversal::is_walkable(hit.normal, cm.slope_limit_deg)
        {
            continue;
        }
        // **The FLOOR point, not the ankle** (P29.4 audit, A13).
        //
        // `ground_offset`'s third argument is documented as "the ankle socket
        // with its Y replaced by the root joint's Y — the character's own ground
        // plane", and it is ALS's `IKFootFloorLocation`. Passing the ankle
        // itself instead makes the whole expression collapse: the offset becomes
        // `impact − ankle`, so the solve drives the ANKLE onto the ground rather
        // than the sole, the foot sinks by however high the ankle is, and
        // `FOOT_HEIGHT_M` — a ported constant with a number in it — cancels out
        // of the arithmetic exactly and does nothing at all. With the floor point
        // the offset is what it is meant to be: how far the ground under THIS
        // foot differs from the ground under the body, zero on a flat floor.
        let floor = glam::Vec3::new(held.x, ground_plane_y as f32, held.z);
        let g = inf_anim::ground_offset(
            glam::Vec3::new(hit.point.x as f32, hit.point.y as f32, hit.point.z as f32),
            glam::Vec3::new(
                hit.normal.x as f32,
                hit.normal.y as f32,
                hit.normal.z as f32,
            ),
            floor,
        );
        offsets[side] = g.offset;
        let target = held + g.offset;
        goals[side] = Some(inf_ecs::anim_bridge::FootGoal {
            target: Vec3d::new(target.x as f64, target.y as f64, target.z as f64),
            weight: enable.clamp(0.0, 1.0),
        });
    }
    // The pelvis drops to the lower foot, so the low leg does not straighten past
    // its limit reaching for a step below the other one. **Recorded** rather than
    // applied: moving the capsule would move the character, and the pelvis is a
    // POSE offset — P29.5's authoring pass is what routes it into the rig. It was
    // computed into a `let _ =` before (P29.4 audit, A9), which is a value the
    // sentence above claims and no reader could check.
    let pelvis = inf_anim::pelvis_offset(enables[0], enables[1], offsets[0], offsets[1]);
    cm.runtime.pelvis_offset = Vec3d::new(pelvis.x as f64, pelvis.y as f64, pelvis.z as f64);
    inf_ecs::anim_bridge::set_foot_ik(world, guid, goals);
}

/// [`mover_for`], with the capsule the movement step decided this step rather
/// than the one the component still holds.
///
/// The distinction matters on exactly one step per stance change: the resize is
/// decided, the sweep must use it, and the component is not written until the
/// end. Passing `None` is `mover_for` verbatim.
fn mover_for_with_capsule(
    world: &EcsWorld,
    guid: uuid::Uuid,
    capsule: Option<(f64, f64)>,
) -> CharacterMover3D {
    let base = mover_for(world, guid);
    match capsule {
        None => base,
        Some((half_height, radius)) => base.with_shape(ColliderShape3D::Capsule {
            half_height: half_height.max(1e-3),
            radius: radius.max(1e-3),
        }),
    }
}

/// Fold an angle into `[0, 360)` degrees, with a non-finite input answering `0`.
fn wrap_deg(a: f64) -> f64 {
    if !a.is_finite() {
        return 0.0;
    }
    let r = a % 360.0;
    if r < 0.0 {
        r + 360.0
    } else {
        r
    }
}
