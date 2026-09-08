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
use inf_ecs::cover as covermodel;
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement as model;
// **One spelling for the [0, 360) fold** (P29.6 audit). This file carried a
// private `wrap_deg` copy of the model's rule, and the locomotion camera needed
// the same fold in the same wave -- two crates spelling one convention is how a
// camera and a character end up disagreeing about where north is.
use inf_ecs::movement::wrap_deg;
use inf_ecs::world::EcsWorld;

use super::cover::{probe_cover, CoverSettings};
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
pub(crate) const FALLBACK_RADIUS_M: f64 = 0.3;

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

/// The weight below which a foot's IK goal is **withdrawn** rather than
/// published faintly (wave CHAR1b.2).
///
/// `interp_to` is an exponential chase, so a gate that has gone to zero leaves
/// the weight at 1e-4 for ever; a goal at that strength is a request nothing
/// acts on and a residual nothing produced. One per cent is below anything a
/// viewer can see and far above the tail.
const FOOT_IK_MIN_WEIGHT: f64 = 0.01;

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
    /// **How many shape casts the cover probe spent this step** (wave COV1).
    ///
    /// The budget arm's number, and the reason it is on the OUTCOME rather than
    /// only on the runtime: `CoverState::sweeps` is a field the step writes and
    /// a step that never touched cover would leave the last value there, so a
    /// gate reading it could not tell "nothing probed" from "nothing has probed
    /// since". This is per step, always, and it is **zero** for every character
    /// that is not in cover and did not press the key.
    pub cover_sweeps: u32,
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
    // **The vehicle step used to be the last statement of this function**
    // (P29.7) and is now the caller's, one line later, in its own `STEP_PHASES`
    // row (island wave VEH1a). Nothing about the ordering changed: both hosts
    // call `super::vehicle::step_vehicles` immediately after this returns, so a
    // driver's controls are still this step's and the forces still land before
    // `bridge.step`.
    //
    // What changed is that a car's milliseconds are now attributed. P29.7's
    // reason for putting it inside — *"a sibling both hosts had to call
    // separately would be a hand-maintained mirror"* — is answered by paying for
    // the mirror instead of avoiding it: the two call sites are fenced
    // (`MIRROR-BEGIN vehicle_step`) and pinned character-for-character by
    // `inf-editor-core`'s `fixed_step_mirror`.
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

/// **How fast the aim-offset layer's weight chases its target** (wave
/// CHAR1b.1) — ALS's `Config.SmoothedAimingRotationInterpSpeed`, which is 10.
///
/// A rate rather than a duration, because that is what `inf_anim::interp_to`
/// takes and what ALS's `FInterpTo` means: the step is `clamp(speed × dt, 0, 1)`
/// of the remaining distance.
const AIM_OFFSET_INTERP: f64 = 10.0;

/// **Whether this character's ground normal is worth a query** (island wave
/// NPC1e) — the mover's one measured lever, and the arc's answer to
/// *"fewer or cheaper ground queries per character"*.
///
/// # What the probe is for, and who reads it
///
/// [`MovementRuntime::ground_normal`](inf_ecs::components::MovementRuntime) has
/// **exactly one reader in the engine** — grep it: `slide_friction`, inside
/// section 7's `MovementMode::Slide` branch. Everything else about standing on
/// the ground comes out of the sweep itself (`grounded`), out of the landing
/// classifier (impact speed) or out of the ledge probe. So on every character
/// that is not sliding, section 9 spends a shape cast against the world to
/// produce a number nothing asks for.
///
/// # What that costs, measured
///
/// `character_move_cost::where_the_movers_queries_go`, on the island's own tile
/// resolution: the probe is **5.3–5.7 µs** of a 31 µs step — **18 %** of
/// `character move` — because a shape cast that *hits* a height-field tile is
/// **11.3×** one that hits a building box, measured in the same world with the
/// same tree and the same filter (the arm drops the identical probe onto a
/// roof), and **2.9×** on the isolated one-body control in
/// `what_a_cast_against_the_ground_costs`.
///
/// *(NPC1e audit: this paragraph first read "~75× a cast", which was this probe
/// divided by the same probe in a world with **no ground**, where the character
/// is in free fall and the cast hits nothing at all. The arm now asserts that
/// the probe hits exactly on the worlds that have ground under the character.
/// It also read "one of about six the mover makes" — `PhysicsWorld3D::queries`
/// counts the whole `move_character` call as **one** ask, and how many casts
/// rapier makes inside it is not something this tree measures.)*
///
/// **And the cheaper shape is measured too**: a downward *ray* reaching the same
/// depth answers the same ground **22×** cheaper than this swept sphere in the
/// fixture world and **45×** on the ground-only one. Whether a ray is good
/// enough for a normal whose only reader is `slide_friction`'s slope is the
/// question this predicate defers rather than answers.
///
/// # Why this predicate, and the one-step bound it carries
///
/// `Slide` is reachable from exactly one place (section 4's crouch-press
/// branch): `mode == Grounded && want_sprint && speed >= slide_entry_speed_mps`.
/// So a character that is sliding, or that is holding sprint, keeps its probe;
/// a walking crowd agent — which has no input at all and whose `Gait` is
/// `Walk` — never asks and never pays.
///
/// **The bound, stated rather than implied.** Section 7 of step *N* reads the
/// normal section 9 of step *N−1* wrote, so the predicate is evaluated one step
/// before the read. A character that acquires `want_sprint` on the very step it
/// presses crouch — while *already* at slide entry speed, which takes a sprint
/// it was not asking for — enters `Slide` with the `+Y` default instead of last
/// step's measured normal. That is a friction difference **only on a slope**
/// (on flat ground the probe answers `+Y` too), for one step, on a transition
/// no input scheme in this tree produces: sprint is held, not tapped, and the
/// speed that gates the entry is reached by holding it.
fn reads_the_ground_normal(cm: &CharacterMovement) -> bool {
    cm.mode == MovementMode::Slide || cm.runtime.want_sprint
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
        // The two P29.7 edges are consumed here too (P29.7 audit, A1): neither
        // is read by a mantle, and an edge that survives a path that could not
        // honour it fires the moment that path ends. `press_jump` is NOT
        // consumed here, because the mantle and the ragdoll both read it.
        cm.runtime.press_interact = false;
        cm.runtime.press_fly = false;
        return step_mantle(world, bridge, guid, cm, dt, overlays);
    }

    // ── 0c. A RAGDOLL owns it just as completely (P29.4, clause 6), and for the
    //    same reason: the articulated bodies are the simulation now, and this
    //    step's velocity model has nothing to say about them. The bridge follows
    //    the pelvis with the capsule and hands the character back when it settles.
    if cm.mode == MovementMode::Ragdoll {
        cm.runtime.press_interact = false;
        cm.runtime.press_fly = false;
        return super::ragdoll_bridge::step_ragdoll(world, bridge, guid, cm, dt, overlays);
    }

    // ── 0d. **A VEHICLE owns it too** (P29.7), and for the third time the same
    //    reason: the chassis is the simulation now, this step's velocity model
    //    has nothing to say about a car, and the character is a passenger on a
    //    rigid body. `press_interact` from the ground climbs in; the seat step
    //    below drives, warps and gets back out.
    //
    //    **The edge is taken whether or not it is honoured** (P29.7 audit, A1),
    //    which is this function's own law forty lines down. It is only *read*
    //    on a grounded step, and while it was only *cleared* on one, a press
    //    made in mid-air was banked for the whole fall and climbed into
    //    whatever car was in reach when the character landed.
    //
    //    The one exception is `Driving`, and it is not a latch: the seat step
    //    below reads this same edge as the EXIT control, so taking it here would
    //    lock the driver in.
    let want_enter = if cm.mode == MovementMode::Driving {
        false
    } else {
        std::mem::take(&mut cm.runtime.press_interact)
    };
    // **The bolt's own edge** (island wave I8b). Taken unconditionally — a
    // driver has no door to lock, and an edge that is never taken is an edge
    // that is banked for the whole drive (the `press_interact` lesson above).
    let want_lock = std::mem::take(&mut cm.runtime.press_lock);
    if want_lock && cm.mode.is_grounded_family() {
        let feet = position - DVec3::Y * (cm.half_height_for(cm.mode) + radius);
        // **The SAME resolution site the open verb uses**, so the prompt, the
        // open and the lock cannot come apart about which door is meant. A hit
        // that is not a door answers `Unusable` and nothing happens, which is
        // the right answer for a switch nobody has written a lock for.
        let mut taken = occupied_seats(world, guid);
        taken.insert(guid);
        if let Some(h) =
            super::interact::resolve(world, bridge, feet, cm.runtime.aim_yaw_deg, &taken)
        {
            if h.verb == inf_ecs::interact::InteractVerb::Use {
                super::door::lock_door(world, h.guid, feet);
            }
        }
    }
    if want_enter && cm.mode.is_grounded_family() {
        let feet = position - DVec3::Y * (cm.half_height_for(cm.mode) + radius);
        // **The one interaction door** (island wave I5). The `press_interact`
        // edge used to be a vehicle question and nothing else; it is now a
        // question about every candidate in reach — seats and authored
        // `Interactable`s together — ranked by one rule.
        //
        // The character itself is excluded, because a character carrying an
        // `Interactable` (an item on a corpse, later) must not be able to
        // interact with itself.
        let mut taken = occupied_seats(world, guid);
        taken.insert(guid);
        let hit = super::interact::resolve(world, bridge, feet, cm.runtime.aim_yaw_deg, &taken);
        // **ENTER, USE and PICK UP have consumers in this step** (ENTER since
        // P29.7; the other two since island wave I6). The rest are gameplay's —
        // a hit this step declines rather than a candidate the door refuses to
        // see. The **prompt** is a separate question a host asks every frame
        // through the same `resolve`, so what the player is told and what the
        // press does cannot come apart.
        //
        // The order is a `match` on the verb and not a chain of `if`s, so a verb
        // added later is a compile error here rather than a silent decline.
        // **The hand goes on it, whatever the verb does** (SK1c). An
        // interactable that names a grip gets one: a door handle is turned, a
        // pickup is taken off the floor, and a switch nobody has written a
        // consumer for is still *reached for*. Recorded before the verb match
        // because it is orthogonal to it -- the hand is about the gesture and the
        // match is about the consequence -- and because `Grab`, the one verb this
        // engine has and does not consume, then stops being a verb that does
        // nothing at all.
        //
        // `hit.position` and not the target's transform: the interaction's own
        // point is what the prompt is measured from (a door's is the middle of
        // its closed opening, not its hinge), so the hand reaches where the
        // player was told the thing is.
        if let Some(h) = hit.as_ref() {
            if let Some(grip) = h.grip.as_deref() {
                inf_ecs::interact::begin_grab(world, guid, h.guid, grip, h.position);
            }
        }
        let entered = match hit.as_ref().map(|h| (h.verb, h.guid)) {
            Some((inf_ecs::interact::InteractVerb::Enter, target)) => Some(target),
            // VEH2b. **One press, one door, one warp**: the carjack makes the
            // seat free and then falls through to the ordinary enter below, so
            // the code that seats a hero after a carjack is the code that has
            // always seated a hero. A refusal — nobody in it, a resist, the
            // wrong side — is a value, and the press does nothing rather than
            // half-doing something.
            Some((inf_ecs::interact::InteractVerb::Carjack, target)) => {
                match super::carjack::try_carjack(world, bridge, target, guid, dt, overlays) {
                    Some(super::carjack::Carjack::Ejected { chassis, .. }) => Some(chassis),
                    // They held on. The press is spent; the player presses again.
                    Some(super::carjack::Carjack::Resisted { .. }) | None => None,
                }
            }
            Some((inf_ecs::interact::InteractVerb::Use, target)) => {
                // A `Use` hit is a door if the world has one under that guid,
                // and nothing at all otherwise — which is the correct answer for
                // a switch or a terminal somebody authored an `Interactable` for
                // and has not written a consumer for yet.
                super::door::use_door(world, target, feet);
                None
            }
            Some((inf_ecs::interact::InteractVerb::PickUp, target)) => {
                inf_ecs::item::pick_up(world, guid, target);
                None
            }
            // EMS3. **The one place a wanted level can be walked away from.**
            // Guarded by `is_wardrobe`, so a `Change` hit on something that is
            // not one does nothing rather than dressing somebody out of a
            // filing cabinet — the same shape `use_door`'s "a `Use` hit is a
            // door if the world has one under that guid" already has.
            Some((inf_ecs::interact::InteractVerb::Change, target)) => {
                if inf_ecs::wardrobe::is_wardrobe(world, target) {
                    inf_ecs::wardrobe::change_clothes(world, guid);
                }
                None
            }
            Some((
                inf_ecs::interact::InteractVerb::Grab | inf_ecs::interact::InteractVerb::Talk,
                _,
            ))
            | None => None,
        };
        if let Some(vehicle) = entered {
            let mut refusal = MovementRefusal::None;
            let probe = ClearanceProbe {
                centre: position,
                radius,
                is_capsule,
                exclude: &exclude,
            };
            cm.mode = request(
                &mut cm,
                bridge,
                &probe,
                MovementMode::Driving,
                true,
                &mut refusal,
            );
            if cm.mode == MovementMode::Driving {
                cm.runtime.seat = inf_ecs::components::SeatState {
                    vehicle,
                    entering: true,
                    time_s: 0.0,
                    start: Vec3d::from_dvec3(position),
                    start_yaw_deg: cm.runtime.body_yaw_deg,
                };
                cm.runtime.time_in_mode_s = 0.0;
                // Parked at the START of the choreography, not at the end: a
                // capsule sliding into a seat with its collider live pushes the
                // car away from itself.
                super::vehicle::park_collider(bridge, guid, true);
            }
        }
    }
    if cm.mode == MovementMode::Driving {
        return step_driving(world, bridge, guid, cm, dt, radius, overlays);
    }

    // ── 0e. **FLIGHT** is its own integration (P29.7): six degrees of freedom,
    //    no gravity, and a bank that comes out of the turn rate. Everything
    //    below this line is about a character standing on, falling toward or
    //    swimming in something, and none of it applies.
    if cm.runtime.press_fly {
        cm.runtime.press_fly = false;
        let mut refusal = MovementRefusal::None;
        let probe = ClearanceProbe {
            centre: position,
            radius,
            is_capsule,
            exclude: &exclude,
        };
        let to = if cm.mode == MovementMode::Flying {
            MovementMode::FallControlled
        } else {
            MovementMode::Flying
        };
        cm.mode = request(&mut cm, bridge, &probe, to, true, &mut refusal);
        if cm.mode != MovementMode::Flying {
            cm.runtime.bank_deg = 0.0;
        }
    }
    if cm.mode == MovementMode::Flying {
        let fly_half = cm.half_height_for(MovementMode::Flying);
        return step_flight(
            world, bridge, guid, cm, dt, position, fly_half, radius, is_capsule, overlays,
        );
    }

    // ── 0f. **THE CRASH-THROUGH** (island wave I6). A body arriving at a shut
    //    door above the breach speed goes through it, and the movement mode
    //    CONTINUES — a slide stays a slide.
    //
    //    Here, before the integration, so the speed the breach prices is the one
    //    the body arrived with, and the speed it leaves with is what this step
    //    then integrates. And *after* the vehicle and flight blocks, because
    //    neither a car nor a wingsuit is a shoulder.
    //
    //    **The same energy door as the kick.** `try_breach` computes `1/2 m v2`
    //    and hands it to `door::strike_door`, which is the function
    //    `gameplay::step_kicks` calls with a kick's own joules. The lock's price
    //    is `Destructible::bond_energy_j`, which is the fracture solve's. There
    //    is one comparison behind all three.
    {
        let feet = position - DVec3::Y * (cm.half_height_for(cm.mode) + radius);
        let band = bridge.sim_band(world);
        if let Some(breach) = super::door::try_breach(
            world,
            &band,
            feet,
            cm.runtime.velocity.to_dvec3(),
            cm.mode,
            inf_ecs::door::DEFAULT_BODY_MASS_KG,
        ) {
            if breach.broke && breach.speed_in_mps > 0.0 {
                // Momentum is scaled rather than re-aimed: the direction is the
                // body's own and the lock took energy, not heading. Scaling the
                // whole velocity (not just its planar part) keeps a dive's
                // vertical component in proportion, which is what makes a dive
                // through a door still a dive on the other side.
                let k = breach.speed_out_mps / breach.speed_in_mps;
                cm.runtime.velocity = Vec3d::from_dvec3(cm.runtime.velocity.to_dvec3() * k);
            }
        }
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
        // **The desired mode, seeded from what the level authored** (wave
        // CHAR1c, carried 123). `VelocityDirection` is both the enum's `Default`
        // and a legal authored value, so the latch is what tells an unseeded
        // field from an authored one — `LocomotionCamera::seeded`'s rule on a
        // second quantity.
        if !cm.runtime.camera_seeded {
            cm.runtime.camera_seeded = true;
            cm.runtime.desired_rotation_mode = cm.rotation_mode;
        }
        // **The rotation-mode key** — ALS's two `…DirectionAction`s folded onto
        // one, each of which sets the desired mode AND applies it
        // (`ALSBaseCharacter.cpp:1404-1416`). Consumed here, on the step it
        // arrives, like every other edge.
        if std::mem::take(&mut cm.runtime.press_rotation_mode) {
            cm.runtime.desired_rotation_mode = match cm.runtime.desired_rotation_mode {
                RotationMode::VelocityDirection => RotationMode::LookingDirection,
                _ => RotationMode::VelocityDirection,
            };
            cm.rotation_mode = cm.runtime.desired_rotation_mode;
        }
        if cm.runtime.want_aim {
            cm.rotation_mode = RotationMode::Aiming;
        } else if cm.rotation_mode == RotationMode::Aiming {
            // **Back to the DESIRED mode, not to `LookingDirection`** — ALS's
            // `AimAction_Implementation(false)` (`.cpp:1291-1301`), which is the
            // half this engine had never had. Releasing aim used to *promote* a
            // character to `LookingDirection` and leave it there, so the only
            // way into that mode was an aim press and there was no way out of
            // it at all.
            cm.rotation_mode = cm.runtime.desired_rotation_mode;
        }
    } else {
        // An edge nobody consumed is an edge that fires on the frame a
        // character becomes player-controlled. The other edges are cleared by
        // the mode table below whether or not it acted on them; this one has no
        // such reader, so it is cleared here.
        cm.runtime.press_rotation_mode = false;
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
    // **The throw's one-shot clock** (wave CHAR1b.2), here for the same reason
    // the get-up blend is: a throwing character walks, turns and falls like any
    // other, because a throw is an upper-body ADDITIVE and not a mode. It counts
    // down, so a character that has never thrown anything carries a zero and
    // this line is one subtraction that saturates at nothing.
    if cm.runtime.throw_s > 0.0 {
        cm.runtime.throw_s = (cm.runtime.throw_s - dt).max(0.0);
    }

    // ── 4. Mode resolution: the single table, asked once per candidate.
    let previous_mode = cm.mode;
    let mut refusal = MovementRefusal::None;
    // **What the cover probe cost this step** (wave COV1) -- zero on every step
    // that did not press the key and is not in cover, which is the budget arm's
    // own number and the reason it is counted rather than assumed.
    let mut cover_sweeps: u32 = 0;
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
        //
        // **COVER IS FIRST** (wave COV1). One key, both directions -- GTA's own
        // binding -- so the press cannot be ambiguous and must not be swallowed
        // by a stance edge that arrived on the same step. The chain is
        // `else if`, so exactly one edge is honoured per step whatever order
        // they are in; being first is what makes the honoured one the
        // deliberate one.
        if cm.runtime.press_cover {
            if cm.mode == MovementMode::Cover {
                // **Leaving.** Back to the stance the surface implied, so a
                // character that was crouched behind a car stays crouched
                // rather than standing up into whatever is above it -- which is
                // the overhead-clearance refusal, and `request` will make it if
                // the stand is the one that is asked for.
                let to = if cm.runtime.cover.crouched {
                    MovementMode::Crouch
                } else {
                    MovementMode::Grounded
                };
                cm.mode = request(&mut cm, bridge, &probe, to, true, &mut refusal);
                if cm.mode != MovementMode::Cover {
                    cm.runtime.cover = covermodel::CoverState::default();
                }
            } else {
                cover_sweeps += try_cover(
                    &mut cm,
                    characters,
                    bridge,
                    &probe,
                    position,
                    radius,
                    &CoverSettings::default(),
                    &mut refusal,
                );
            }
        } else if cm.runtime.press_dive {
            // **A dive needs a sprint** (I5, the owner's ruling), exactly as a
            // slide does forty lines down. The two are the same move at two
            // heights and it was never coherent that one was gated and the
            // other was free: a dive from a standing start is a belly-flop, and
            // the P29 catalogue's `dive_speed_mps` is a *launch* speed that
            // assumes a body already moving.
            //
            // Folded into the condition rather than branched around, so a
            // refusal is a **value** — `request_mode` answers
            // `ConditionNotMet` and the character does whatever it was going to
            // do instead, which is stand there. Nothing here can fail.
            let from_ground = cm.mode.is_grounded_family() && cm.runtime.want_sprint;
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
            // **The slide's refusal is a value too** (I5). A player holding
            // sprint and pressing crouch has asked for a slide; if the body is
            // not moving fast enough the answer is `ConditionNotMet`, and the
            // stance toggle below is what the character does *instead*. Before
            // this the refusal was a silent fall-through — the one entry
            // condition in the catalogue that the outcome could not be
            // distinguished from a deliberate crouch, so nothing downstream
            // (a HUD hint, a tutorial, a telemetry counter) could ever say why.
            if cm.mode == MovementMode::Grounded
                && cm.runtime.want_sprint
                && speed_planar < cm.slide_entry_speed_mps
            {
                refusal = MovementRefusal::ConditionNotMet;
                cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
            }
            let to = if sliding {
                MovementMode::Slide
            } else if cm.mode == MovementMode::Crouch || cm.mode == MovementMode::Slide {
                MovementMode::Grounded
            } else {
                MovementMode::Crouch
            };
            cm.mode = request(&mut cm, bridge, &probe, to, true, &mut refusal);
        } else if cm.runtime.press_jump {
            // ── **Space is a DIVE when there is water to dive into** (I5) ──
            //
            //    Tried before the mantle and before the jump, because all three
            //    are the same gesture and only the world can tell them apart.
            //    The rule is `model::dive_into_water` — pure, so it is measured
            //    on its own — and it is fed the water surface a reach ahead of
            //    the feet, through the bridge's place query rather than the
            //    character's own probe: "is there water in front of me" is a
            //    question about a place.
            //
            //    Inert on every level with no water: `water_surface_at` answers
            //    `None` in `O(1)` when the index is empty, `dive_into_water`
            //    answers `false`, and the jump below runs exactly as it did.
            let feet_y = position.y - (cm.half_height_for(cm.mode) + radius);
            let ahead = model::rotate_from_frame(Vec2d::new(0.0, 1.0), cm.runtime.aim_yaw_deg);
            let probe_xz = glam::DVec2::new(
                position.x + ahead.x * model::DIVE_WATER_REACH_M,
                position.z + ahead.y * model::DIVE_WATER_REACH_M,
            );
            let water_dive = model::dive_into_water(
                !cm.mode.is_grounded_family(),
                cm.runtime.want_sprint,
                feet_y,
                bridge.water_surface_at(probe_xz),
            );
            if water_dive {
                cm.mode = request(
                    &mut cm,
                    bridge,
                    &probe,
                    MovementMode::Dive,
                    true,
                    &mut refusal,
                );
                if cm.mode == MovementMode::Dive && previous_mode != MovementMode::Dive {
                    cm.runtime.velocity = Vec3d::new(
                        ahead.x * cm.dive_speed_mps,
                        cm.dive_up_speed_mps,
                        ahead.y * cm.dive_speed_mps,
                    );
                }
            } else {
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
                // **The vault out of cover needs no stick** (wave COV1). A
                // character pressed against a car's flank is already facing the
                // thing it is about to go over -- the cover state pinned that
                // facing -- so ALS's `bHasMovementInput` gate has nothing left
                // to disambiguate, and requiring it would mean holding forward
                // INTO the wall to climb it.
                let mantled = (wants || cm.mode == MovementMode::Cover)
                    && (cm.mode == MovementMode::Grounded
                        || cm.mode == MovementMode::Crouch
                        || cm.mode == MovementMode::Cover)
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
        // A vault out of cover leaves it (wave COV1): the mantle owns the
        // character now, and a `CoverState` that survived would pin the capsule
        // back against the wall the moment the climb ended.
        cm.runtime.cover = covermodel::CoverState::default();
        cm.runtime.press_cover = false;
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
            cover_sweeps: 0,
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
    cm.runtime.press_cover = false;

    // ── 4c. **COVER: the surface, the lean, and the stick** (wave COV1).
    //
    //    Everything a character in cover needs decided BEFORE the capsule is
    //    resized and before the integrator runs, because all three of those
    //    read it: the stance comes from the class (step 5 resizes to it), the
    //    speed comes from the class (step 6), and the wish direction is the
    //    stick PROJECTED ONTO THE SURFACE (step 7). What is deliberately not
    //    here is the position: a cover slide is integrated by the same
    //    integrator, moved by the same mover and constrained afterwards, so the
    //    gaits, the stride warping and the foot IK are the ones every other
    //    grounded mode uses rather than a second set.
    if cm.mode == MovementMode::Cover {
        cover_sweeps += cover_before_move(
            &mut cm,
            characters,
            bridge,
            &probe,
            position,
            radius,
            dt,
            &CoverSettings::default(),
            &mut refusal,
        );
    }

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
    // …and **a dive that has just reached the water is deliberate too** (I5).
    // The step a `Dive` becomes a swim is the one step where the player's whole
    // committed intent is "go under", and the vertical axis is not being asked
    // for — the hands are on the movement keys. Without this the float balance
    // wins on entry and the dive that the owner's Space control exists for ends
    // with the character bobbing on the surface, which is the same defect P29.6
    // measured from the other direction.
    //
    // `previous_mode` is this step's own local, so there is no latch and no
    // state to serialize: the door is open for exactly the transition step.
    let deliberate = cm.mode.is_swimming()
        && (cm.runtime.intent_vertical < 0.0 || previous_mode == MovementMode::Dive);
    let motion = bridge.apply_swim_motion_where(
        guid,
        DVec3::new(planar.x * dt, vertical * dt, planar.y * dt) + root_world,
        dt,
        deliberate,
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

    // ── 9. The ground normal, for the slide curve. rapier reports "grounded"
    //    but not what it stood on, so this is a short downward sweep — **taken
    //    only for a character that can read it** (island wave NPC1e; see
    //    `reads_the_ground_normal`).
    cm.runtime.ground_normal = Vec3d::new(0.0, 1.0, 0.0);
    if result.grounded && is_capsule && reads_the_ground_normal(&cm) {
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

    // ── 9a. **COVER: the constraint** (wave COV1).
    //
    //    The mover has just swept the character wherever its velocity took it,
    //    with the wall stopping it in the one direction that matters. This puts
    //    it back ON the cover line -- the anchor, plus the slide along the
    //    surface, plus whatever the peek is leaning -- and clamps the slide at
    //    the corners the probe measured.
    //
    //    After the move rather than instead of it, so the wall, the ground and
    //    every other collider still get their say and the constraint is a
    //    correction rather than a teleport.
    if cm.mode == MovementMode::Cover {
        position = cover_after_move(&mut cm, position, radius, dt);
        probe.centre = position;
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
                        cover_sweeps: 0,
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
    // **In cover the body faces the SURFACE** (wave COV1) and the smoother has
    // nothing to say about it: a character sliding left along a wall does not
    // turn to face left, it side-steps -- which is the whole visual difference
    // between cover and walking. The snap eases the facing in over its own
    // window and the rest of the time it is pinned, so this branch returns
    // before the two-stage smoother rather than fighting it.
    if cm.mode == MovementMode::Cover {
        let c = cm.runtime.cover;
        cm.runtime.body_yaw_deg = if c.snapping() {
            covermodel::snap_yaw_deg(c.start_yaw_deg, c.yaw_deg, c.alpha())
        } else {
            c.yaw_deg
        };
        cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;
        cm.runtime.turning_in_place = false;
        cm.runtime.turn_delay_s = 0.0;
        cm.runtime.rotate_left = false;
        cm.runtime.rotate_right = false;
        cm.runtime.rotate_rate = 1.0;
    }
    // …and the smoother below is skipped entirely while in cover, both branches
    // of it: the `moving` branch would chase the velocity's heading and the
    // standing one would start a turn in place, and a body pinned against a
    // wall must do neither.
    let in_cover = cm.mode == MovementMode::Cover;
    let moving = moving && !in_cover;
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
    } else if !in_cover {
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
    // **The lean is INTERPOLATED, not copied** (wave CHAR1b.2, clause 6).
    // `UpdateMovementValues` runs exactly these two lines
    // (`ALSCharacterAnimInstance.cpp:633-638`) and the reason is mechanical:
    // `relative_accel` is a step function on the frame the stick moves — a
    // character starting from rest sees it go 0 -> 1 in one fixed step — and a
    // body that snapped to full lean on one frame would read as a twitch.
    // `interp_to` is the same exponential chase the foot offsets and the aim
    // mask already use, so there is one smoothing rule in this file.
    cm.runtime.lean = Vec2d::new(
        inf_anim::interp_to(
            cm.runtime.lean.x,
            cm.runtime.relative_accel.x,
            model::LEAN_INTERP,
            dt,
        ),
        inf_anim::interp_to(
            cm.runtime.lean.y,
            cm.runtime.relative_accel.y,
            model::LEAN_INTERP,
            dt,
        ),
    );
    // ── DISTANCE MATCHING's input (wave CHAR1b.2, clause 6) ──────────────────
    //
    //    How far this character will still travel, at the braking rate its own
    //    curve gives at this speed. Zero while the stick is pushed, so "is this
    //    character stopping" is a comparison against zero and the two stop clips
    //    become a decision instead of the coincidence the CHAR1b.1 audit
    //    measured at **0 steps of five run-and-stops**.
    cm.runtime.stop_distance_m = if cm.runtime.intent_move.x.abs() + cm.runtime.intent_move.y.abs()
        > 1.0e-3
        || !cm.mode.is_grounded_family()
    {
        0.0
    } else {
        inf_anim::stop_distance_m(speed_after, settings.braking_mps2)
    };
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
    // **Interpolated, not snapped** (wave CHAR1b.1). This weight is now the
    // additive aim-offset layer's AND the look-at chain's, and a clip whose
    // `Mask_AimOffset` steps from 0 to 1 on one frame would swing a character's
    // head 70° in 16 ms. ALS's own answer to a layer changing is that the
    // layered blend has a blend time; ours is `inf_anim::interp_to` at the
    // donor's `SmoothedAimingRotationInterpSpeed`, which bounds the change to
    // `speed × dt` — 1/6 of the range per frame at 60 Hz, so a full swap takes
    // six frames and no single frame moves the head more than 12°.
    cm.runtime.aim_offset_mask = inf_anim::interp_to(
        cm.runtime.aim_offset_mask,
        aim_mask.clamp(0.0, 1.0),
        AIM_OFFSET_INTERP,
        dt,
    )
    .clamp(0.0, 1.0);
    cm.runtime.aim_offset_weight = 1.0 - cm.runtime.aim_offset_mask;
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
    // **This step's cover cost, recorded where the pose step and the hero log
    // can read it** (wave COV1). Written unconditionally so it is a per-step
    // figure: a field only the cover path wrote would keep the last probe's
    // number for ever and a budget arm could not tell "nothing probed" from
    // "nothing has probed since".
    cm.runtime.cover.sweeps = cover_sweeps;

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
            // Yaw and the BANK. A character's *pitch* belongs to its pose and
            // never to its body; its roll is the flight bank, which is zero in
            // every other mode — and writing it unconditionally is what makes
            // that true of the transform as well as of the runtime. Measured
            // before this line existed: a character that landed out of a banked
            // turn kept the bank for ever, because nothing else ever wrote the
            // roll back. (`Transform::rotation` is euler DEGREES, YXZ.)
            t.rotation.y = body_yaw;
            t.rotation.z = cm.runtime.bank_deg;
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
    //    **The precedence is THIS one** (P29.6 audit, A11). The Tick pass runs
    //    BEFORE the movement step in both hosts, so this write lands last and a
    //    Blueprint's `anim.set_param` on one of the nine published names is
    //    overwritten before the pose step reads it. One authority for one fact;
    //    a game that wants to drive `speed` itself takes the movement component
    //    off. Pinned by `projector_mirror` and by `live_tuning`.
    //
    //    Costs one map lookup on a character with no machine, which is every
    //    character in every committed level before this wave.
    //
    // ── 12a. **STRIDE WARPING's rate** (wave CHAR1b.2, clause 6).
    //
    //    The character's own ground speed over the speed the clip it is playing
    //    depicts, read off the blended `MoveData_Speed` channel at the play-head
    //    — the deriver's measurement of the clip, not three constants somebody
    //    typed into a config the way ALS's `Config.Animated*Speed` are. Computed
    //    HERE rather than inside the publisher so there is ONE number and a gate
    //    can read the one the pose step used (`MovementRuntime::play_rate`)
    //    instead of forming a second opinion.
    //
    //    The curve is last step's, because the pose step has not run yet — one
    //    fixed step of latency on a quantity that changes with the gait, the same
    //    latency `Mask_AimOffset` above already carries.
    //
    //    A mantle sets its own rate from the height remap and returns before
    //    this line ever runs (`step_mantle`), so the two never fight.
    let clip_mps = f64::from(inf_ecs::anim_bridge::anim_curve(
        world,
        guid,
        inf_anim::derive::MOVE_DATA_SPEED,
        0.0,
    ));
    cm.runtime.play_rate = inf_anim::stride_play_rate(speed_after, clip_mps);
    // The component was written back at step 12 above, before this number
    // existed; `lean` and `stop_distance_m` were derived at step 11 and are in
    // that clone already. Only the rate is patched in, rather than the whole
    // component re-cloned.
    {
        let w = world.world_mut();
        if let Some(mut slot) = w.get_mut::<CharacterMovement>(entity) {
            slot.runtime.play_rate = cm.runtime.play_rate;
        }
    }
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
        cover_sweeps,
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
    let into = DVec3::new(forward.x, 0.0, forward.y);
    let Some(ledge) = traversal::probe_ledge(
        bridge.world_mut(),
        feet,
        into,
        radius,
        half,
        cm.slope_limit_deg,
        settings,
        &exclude,
    ) else {
        return false;
    };
    let from = cm.mode;
    // **A vault goes OVER, a mantle goes ONTO** (wave COV1, clause 3).
    //
    // The same probe answers both — the ledge, its height and its clip are
    // identical — and the only difference is where the feet end up. From cover,
    // over a LOW surface, the target is the ground on the FAR side: that is what
    // "vault over the hood" means, and a mantle that put the character standing
    // on the bonnet would be a different move with the same button.
    //
    // It is offered rather than assumed: a surface with nothing behind it (a
    // parapet over a drop, a container against a wall) answers `None` and the
    // press is the ordinary mantle onto the top, which is the right answer for
    // a low wall at the edge of a roof.
    let vault = (from == MovementMode::Cover && !ledge.high)
        .then(|| far_side_landing(bridge, ledge.feet, into, feet.y, radius, half, &exclude))
        .flatten();
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
    let target = vault.unwrap_or(ledge.feet);
    // A vault crosses further than a mantle climbs, so it is given the time to:
    // the clock is scaled by how much further, bounded at twice, so the same
    // clip plays over a longer arc rather than the character skating.
    let reach_top = (ledge.feet - feet).length().max(1e-3);
    let stretch = if vault.is_some() {
        ((target - feet).length() / reach_top).clamp(1.0, 2.0)
    } else {
        1.0
    };
    cm.runtime.mantle = MantleState {
        active: true,
        start: Vec3d::from_dvec3(feet),
        start_yaw_deg: cm.runtime.body_yaw_deg,
        target: Vec3d::from_dvec3(target),
        target_yaw_deg: ledge.yaw_deg,
        elapsed_s: 0.0,
        duration_s: model::mantle_duration_s(
            ledge.height_m,
            settings.min_height_m,
            settings.max_height_m,
            play_rate,
        ) * stretch,
        height_m: ledge.height_m,
        high: ledge.high,
        clip_start_s,
        play_rate,
        // **Which hand leads** — ALS's `GetMantleAsset(MantleType,
        // CurrentOverlayState)` (`ALSMantleComponent.h:48`, called at
        // `.cpp:97`), made out of what this engine has. An overlay is what a
        // character is *carrying*, and a character carrying something in its
        // right hand reaches for the ledge with its left; the empty overlay is a
        // character with both hands free and takes the right-handed clip.
        // Latched here so a mantle cannot swap hands halfway up a wall.
        left_hand: !cm.overlay.is_empty(),
    };
    true
}

/// **The exclusion set a cover probe and a mantle both use** — ALS's
/// `IgnoreOnlyPawn`, made out of what this engine has.
///
/// The character's own collider is already in `base`; every other character's is
/// added here, so a crowd can be neither climbed nor hidden behind. Hoisted at
/// wave COV1 because [`try_mantle`] and [`try_cover`] were about to spell the
/// same loop twice, and a cover probe that saw a different set from the mantle's
/// would take cover behind a person the vault could not climb.
fn ignore_only_pawn(
    base: &BTreeSet<ColliderId3D>,
    characters: &[uuid::Uuid],
    bridge: &mut PhysicsBridge3D,
) -> BTreeSet<ColliderId3D> {
    let mut exclude = base.clone();
    for other in characters {
        if let Some(c) = bridge.collider_of(*other) {
            exclude.insert(c);
        }
    }
    exclude
}

/// **The character's feet**, world metres, from its capsule centre.
fn feet_of(cm: &CharacterMovement, position: DVec3, radius: f64) -> DVec3 {
    position - DVec3::Y * (cm.half_height_for(cm.mode) + radius)
}

/// **Take cover** (wave COV1, clause 2) — the press, answered.
///
/// Probes ahead of the character, and on a surface that classifies as cover
/// enters [`MovementMode::Cover`] with a [`CoverState`](inf_ecs::cover::CoverState)
/// carrying the anchor, the facing, the class and the extents. Answers **how
/// many shape casts it spent**, which is the budget arm's number, and refuses by
/// value everywhere else: a probe that found nothing sets
/// [`MovementRefusal::ConditionNotMet`] and the character does whatever it was
/// going to do instead, which is stand there.
///
/// # Where it reaches
///
/// From the character's own FACING, or from the stick when one is held. GTA
/// takes cover in the direction you are pushing; a player running along a wall
/// and pressing cover means *that wall*, not whatever is in front of their nose.
/// The stick is in the aim frame, so it is rotated out of it first — the same
/// rotation step 7's wish direction takes.
#[allow(clippy::too_many_arguments)]
fn try_cover(
    cm: &mut CharacterMovement,
    characters: &[uuid::Uuid],
    bridge: &mut PhysicsBridge3D,
    probe: &ClearanceProbe<'_>,
    position: DVec3,
    radius: f64,
    settings: &CoverSettings,
    refusal: &mut MovementRefusal,
) -> u32 {
    let feet = feet_of(cm, position, radius);
    // The stick when it is held, the facing when it is not.
    let stick = cm.runtime.intent_move;
    let reach = if stick.x.abs().max(stick.y.abs()) >= COVER_STICK_MIN {
        let w = model::rotate_from_frame(stick, cm.runtime.aim_yaw_deg);
        DVec3::new(w.x, 0.0, w.y)
    } else {
        let f = model::rotate_from_frame(Vec2d::new(0.0, 1.0), cm.runtime.body_yaw_deg);
        DVec3::new(f.x, 0.0, f.y)
    };
    let exclude = ignore_only_pawn(probe.exclude, characters, bridge);
    let half = cm.half_height_for(MovementMode::Grounded);
    let found = probe_cover(
        bridge,
        feet,
        reach,
        radius,
        half,
        cm.slope_limit_deg,
        settings,
        &exclude,
    );
    if !found.class.is_cover() {
        *refusal = MovementRefusal::ConditionNotMet;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        cm.runtime.cover.sweeps = found.sweeps;
        return found.sweeps;
    }
    // **The snap is bounded** (clause 2). A surface further than
    // `MAX_SNAP_M` away is walked to rather than snapped to, because a blend
    // that covered a room would be a teleport with a ramp on it.
    let travel = ((found.anchor.x - feet.x).powi(2) + (found.anchor.z - feet.z).powi(2)).sqrt();
    if travel > covermodel::MAX_SNAP_M {
        *refusal = MovementRefusal::ConditionNotMet;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        cm.runtime.cover.sweeps = found.sweeps;
        return found.sweeps;
    }
    let from = cm.mode;
    let verdict = model::request_mode(from, MovementMode::Cover, true, true);
    if verdict.refusal != MovementRefusal::None {
        *refusal = verdict.refusal;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        return found.sweeps;
    }
    // **The sprint-to-cover slide-in** (clause 3): a body arriving fast gets a
    // longer window and the authored slide over it, and one walking in gets
    // GTA's quarter second.
    let speed = (cm.runtime.velocity.x * cm.runtime.velocity.x
        + cm.runtime.velocity.z * cm.runtime.velocity.z)
        .sqrt();
    let slide_in = from == MovementMode::Slide || speed >= cm.slide_entry_speed_mps;
    cm.mode = MovementMode::Cover;
    cm.runtime.time_in_mode_s = 0.0;
    cm.runtime.cover = covermodel::CoverState {
        active: true,
        class: found.class,
        crouched: found.class.crouches(),
        anchor: Vec3d::from_dvec3(found.anchor),
        normal: Vec3d::from_dvec3(found.normal),
        yaw_deg: found.yaw_deg,
        top_m: found.top_m,
        left_m: found.left_m,
        right_m: found.right_m,
        along_m: 0.0,
        at_corner: false,
        side: covermodel::CoverSide::Behind,
        peek: 0.0,
        blend_s: 0.0,
        snap_s: if slide_in {
            covermodel::SLIDE_IN_S
        } else {
            covermodel::SNAP_S
        },
        slide_in,
        start: Vec3d::from_dvec3(feet),
        start_yaw_deg: cm.runtime.body_yaw_deg,
        away_s: 0.0,
        sweeps: found.sweeps,
    };
    found.sweeps
}

/// How far the stick must be pushed for a cover press to mean *that* direction
/// rather than *straight ahead*, `[0, 1]`.
const COVER_STICK_MIN: f64 = 0.3;

/// **How far along a wall a corner has to be for a peek to reach it**, metres.
///
/// A character more than this from either end of a surface has no corner to
/// lean around, and [`inf_ecs::cover::peek_side`] answers `Behind` — which is
/// where blind fire is the only shot there is.
const CORNER_REACH_M: f64 = 1.20;

/// **Everything a character in cover decides before it moves** (wave COV1) —
/// the re-probe, the re-classification, the leave clocks and the stick.
///
/// Answers the shape casts it spent.
///
/// # Why it re-probes every step
///
/// Because the surface CHANGES along its own length. A car's flank is `Low` and
/// its roof line is `High`; a wall ends at a corner. The class, the anchor and
/// the extents are all functions of where the character is standing NOW, and a
/// state latched on entry would have the character crouching behind a wall
/// three metres from where the car was. It costs the probe's own sweeps —
/// counted, reported, and paid only while a character is actually in cover.
#[allow(clippy::too_many_arguments)]
fn cover_before_move(
    cm: &mut CharacterMovement,
    characters: &[uuid::Uuid],
    bridge: &mut PhysicsBridge3D,
    probe: &ClearanceProbe<'_>,
    position: DVec3,
    radius: f64,
    dt: f64,
    settings: &CoverSettings,
    refusal: &mut MovementRefusal,
) -> u32 {
    let exclude = ignore_only_pawn(probe.exclude, characters, bridge);
    let half = cm.half_height_for(MovementMode::Grounded);
    // **The probe is taken from the TUCKED position**, not from where a peek has
    // leaned the capsule to: a character leaning around a corner is, by
    // definition, standing where the surface is not, and a probe from there
    // would lose the very cover the lean is measured against. Subtracting the
    // lean is one vector and it keeps the anchor still while the body moves.
    let lean = covermodel::tangent_left(cm.runtime.cover.normal).to_dvec3()
        * covermodel::peek_lateral_m(cm.runtime.cover.side, cm.runtime.cover.peek);
    let feet = feet_of(cm, position, radius) - lean;
    // Into the surface, from where the character is now.
    let into = -cm.runtime.cover.normal.to_dvec3();
    let found = probe_cover(
        bridge,
        feet,
        into,
        radius,
        half,
        cm.slope_limit_deg,
        settings,
        &exclude,
    );
    let sweeps = found.sweeps;
    {
        let c = &mut cm.runtime.cover;
        c.sweeps = sweeps;
        // **The re-classification** (clause 3). A surface that answered keeps
        // its class; one the probe lost for a step keeps the class it had,
        // because leaving cover is a decision the stick and the button make and
        // not something a missed sweep does.
        if found.class.is_cover() {
            c.class = found.class;
            c.crouched = found.class.crouches();
            c.top_m = found.top_m;
            c.left_m = found.left_m;
            c.right_m = found.right_m;
            c.normal = Vec3d::from_dvec3(found.normal);
            c.yaw_deg = found.yaw_deg;
            // **The anchor is where the feet belong RIGHT NOW.**
            //
            // Re-measured from the character's own tucked position every step,
            // which is what makes the extents beside it mean "how far to the
            // corner FROM HERE" rather than "from where I pressed the button
            // four metres ago". The first cut kept the entry anchor and
            // accumulated a slide offset against it, and the two disagreed the
            // moment the character moved: a 1.5 m wall let a slide run 1.75 m
            // PAST its end, because the extents were measured at one place and
            // the clamp applied at another.
            c.anchor = Vec3d::from_dvec3(found.anchor);
        }
    }

    // ── the leave clock: the stick held AWAY from the surface (clause 3).
    let n = cm.runtime.cover.normal.to_dvec3();
    let stick = cm.runtime.intent_move;
    // The surface's normal expressed in the AIM frame, which is the frame the
    // stick is in — one rotation rather than two.
    let n_local = model::rotate_into_frame(Vec2d::new(n.x, n.z), cm.runtime.aim_yaw_deg);
    let away = covermodel::pushing_away(stick.x, stick.y, n_local.x, n_local.y);
    if away {
        cm.runtime.cover.away_s += dt;
    } else {
        cm.runtime.cover.away_s = 0.0;
    }
    if cm.runtime.cover.away_s > covermodel::AWAY_LEAVE_S {
        let to = if cm.runtime.cover.crouched {
            MovementMode::Crouch
        } else {
            MovementMode::Grounded
        };
        let verdict = model::request_mode(MovementMode::Cover, to, true, true);
        if verdict.refusal == MovementRefusal::None {
            cm.mode = verdict.mode;
            cm.runtime.cover = covermodel::CoverState::default();
            return sweeps;
        }
        // Standing up under something is a refusal, and a character that cannot
        // stand stays in cover with the clock reset rather than half-leaving.
        *refusal = verdict.refusal;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        cm.runtime.cover.away_s = 0.0;
    }

    // ── the PEEK (clause 4). The aim control is what leans the body out, and
    //    which way it leans is the class's and the nearer corner's.
    {
        let c = &mut cm.runtime.cover;
        let want = cm.runtime.want_aim && !c.snapping();
        if want {
            // The extents are already measured FROM the character, so the
            // offset is zero: `peek_side` answers with the corner it is nearer
            // to right now.
            c.side =
                covermodel::peek_side(c.class, 0.0, c.left_m, c.right_m, radius, CORNER_REACH_M);
        }
        // A peek that has nowhere to go does not lean: a wall with no corner in
        // reach answers `Behind`, and `Behind` is the tucked-in pose.
        let target = if want && c.side.is_out() { 1.0 } else { 0.0 };
        let step = covermodel::PEEK_RATE_PER_S * dt;
        c.peek = if c.peek < target {
            (c.peek + step).min(target)
        } else {
            (c.peek - step).max(target)
        };
        if c.peek <= 0.0 && !want {
            c.side = covermodel::CoverSide::Behind;
        }
        // **The stance follows the peek** (clause 4): an `Over` peek clears a
        // low cover by STANDING UP, which is what makes the head reach past the
        // top rather than the pose only pretending to.
        if c.class == covermodel::CoverClass::Low {
            c.crouched = !covermodel::peek_stands(c.side, c.peek);
        }
    }

    // ── the STICK, projected onto the surface (clause 3).
    //
    //    Everything downstream — the gait, the integrator, the stride warp —
    //    reads `intent_move`, so constraining it here is what makes a cover
    //    slide use the SAME machinery a walk does rather than a second one.
    //    During the snap there is no stick at all: the blend owns the character.
    {
        let c = cm.runtime.cover;
        let left = covermodel::tangent_left(c.normal).to_dvec3();
        let world = model::rotate_from_frame(cm.runtime.intent_move, cm.runtime.aim_yaw_deg);
        let along = if c.snapping() {
            0.0
        } else {
            (world.x * left.x + world.y * left.z).clamp(-1.0, 1.0)
        };
        // Back into the aim frame, so step 7's own rotation puts it where it
        // came from.
        let w = Vec2d::new(left.x * along, left.z * along);
        cm.runtime.intent_move = model::rotate_into_frame(w, cm.runtime.aim_yaw_deg);
    }
    sweeps
}

/// **Put the character back on the cover line** (wave COV1) — the constraint the
/// mover's result is corrected by, and the corner stop.
///
/// Three things happen, in this order:
///
/// 1. **The slide is integrated and CLAMPED.** How far along the surface the
///    character has travelled comes from its own velocity, which the integrator
///    produced from the projected stick — so a cover slide accelerates, brakes
///    and reads its gait exactly as a walk does. [`inf_ecs::cover::clamp_along`]
///    stops it with its capsule's EDGE at the corner the probe measured.
/// 2. **The peek is added on top of the clamp**, deliberately: leaning around a
///    corner means going PAST it, and a peek that the clamp undid would be a
///    lean nothing could ever be shot through.
/// 3. **The position is pinned** to the anchor line in `x`/`z` and the mover's
///    own `y` is kept, so the ground still owns the height and a character
///    sliding along a wall follows the pavement under it.
///
/// While the snap runs, steps 1 and 2 are skipped and the position is the blend.
fn cover_after_move(cm: &mut CharacterMovement, position: DVec3, radius: f64, dt: f64) -> DVec3 {
    let c = cm.runtime.cover;
    let left = covermodel::tangent_left(c.normal).to_dvec3();
    if left == DVec3::ZERO {
        return position;
    }
    let anchor = c.anchor.to_dvec3();
    if c.snapping() {
        // The blend, in the ground plane only: the mover owns `y`, so a snap
        // cannot put the character through the floor or leave it in the air.
        cm.runtime.cover.blend_s += dt;
        let a = cm.runtime.cover.alpha();
        let start = c.start.to_dvec3();
        let p = covermodel::snap_position(
            Vec3d::new(start.x, 0.0, start.z),
            Vec3d::new(anchor.x, 0.0, anchor.z),
            a,
        );
        // The velocity the blend implies, so the gait and the stride warp read a
        // character that is MOVING rather than one standing in an idle while its
        // transform travels.
        let travel = ((anchor.x - start.x).powi(2) + (anchor.z - start.z).powi(2)).sqrt();
        let v = if c.snap_s > 0.0 {
            travel / c.snap_s
        } else {
            0.0
        };
        let dir = Vec3d::new(
            (anchor.x - start.x) / travel.max(1e-9),
            0.0,
            (anchor.z - start.z) / travel.max(1e-9),
        );
        cm.runtime.velocity = Vec3d::new(dir.x * v, 0.0, dir.z * v);
        return DVec3::new(p.x, position.y, p.z);
    }
    // The snap is over: the character owns its slide.
    //
    // **The step is a DELTA against the anchor the probe just re-measured**, and
    // the extents beside it are distances from that same place — so the clamp
    // and the thing it clamps are in one frame. `along_m` accumulates beside it
    // as the total travelled, which is what a trace and a caption read; nothing
    // decides on it.
    let v = cm.runtime.velocity.to_dvec3();
    let tangential = v.x * left.x + v.z * left.z;
    let (delta, at_corner) = covermodel::clamp_along(tangential * dt, c.left_m, c.right_m, radius);
    cm.runtime.cover.along_m += delta;
    cm.runtime.cover.at_corner = at_corner;
    // A character wedged at a corner is not still accelerating into it.
    let tangential = if at_corner { 0.0 } else { tangential };
    cm.runtime.velocity = Vec3d::new(left.x * tangential, 0.0, left.z * tangential);
    let offset = delta + covermodel::peek_lateral_m(c.side, c.peek);
    DVec3::new(
        anchor.x + left.x * offset,
        position.y,
        anchor.z + left.z * offset,
    )
}

/// **The ground on the FAR side of a low surface** (wave COV1) — where a vault
/// out of cover lands.
///
/// Marched rather than solved: step into the surface from its own top in
/// [`VAULT_STEP_M`] increments and drop a sphere at each; the first place the
/// ground comes back down to within [`VAULT_LAND_TOLERANCE_M`] of the
/// character's own feet, **and** where its capsule fits, is the far side.
///
/// `None` when there is no such place inside [`VAULT_MAX_DEPTH_M`] — a container
/// against a wall, a parapet over a drop, a hedge four metres deep — and the
/// caller then does the ordinary mantle onto the top, which is the honest answer
/// for all three.
///
/// Costs at most `2 x VAULT_MAX_DEPTH_M / VAULT_STEP_M` casts and runs only on
/// the press that vaults.
#[allow(clippy::too_many_arguments)]
fn far_side_landing(
    bridge: &mut PhysicsBridge3D,
    ledge_feet: DVec3,
    into: DVec3,
    feet_y: f64,
    radius: f64,
    half_height: f64,
    exclude: &BTreeSet<ColliderId3D>,
) -> Option<DVec3> {
    let dir = DVec3::new(into.x, 0.0, into.z).normalize_or_zero();
    if dir == DVec3::ZERO {
        return None;
    }
    let mut d = VAULT_STEP_M;
    while d <= VAULT_MAX_DEPTH_M {
        let over = ledge_feet + dir * d;
        // From a little above the ledge's own top, straight down, far enough to
        // reach the character's own level and a metre past it.
        let start = over + DVec3::Y * 0.25;
        let reach = (over.y - feet_y).max(0.0) + 1.25;
        let ground = bridge.world_mut().cast_shape(
            &ColliderShape3D::Sphere { radius: 0.15 },
            start,
            DQuat::IDENTITY,
            -DVec3::Y,
            reach,
            exclude,
        );
        match ground {
            // **The march stops at the first thing it cannot see the ground
            // from.** A sample inside a wall, or one with nothing under it at
            // all, means the far side is not reachable in a straight line —
            // and continuing past it is how a vault over a car parked against a
            // building came out on the far side of the BUILDING. Measured: the
            // search walked 2.25 m through a wall and answered.
            None => break,
            Some(h) if h.started_penetrating => break,
            Some(_) => {}
        }
        if let Some(hit) = ground {
            if (hit.point.y - feet_y).abs() <= VAULT_LAND_TOLERANCE_M {
                let landing = DVec3::new(over.x, hit.point.y, over.z);
                // The character has to FIT there, or the vault ends inside a
                // parked car — `probe_ledge`'s own room check, one place along.
                let centre = landing + DVec3::Y * (half_height + radius + 0.02);
                let blocked = bridge
                    .world_mut()
                    .cast_shape(
                        &ColliderShape3D::Capsule {
                            half_height: half_height.max(1e-3),
                            radius: radius.max(1e-3),
                        },
                        centre,
                        DQuat::IDENTITY,
                        DVec3::Y,
                        1e-3,
                        exclude,
                    )
                    .is_some_and(|h| h.started_penetrating);
                if !blocked {
                    return Some(landing);
                }
            }
        }
        d += VAULT_STEP_M;
    }
    None
}

/// How finely the vault search steps across a surface, metres.
const VAULT_STEP_M: f64 = 0.25;
/// How deep a surface a vault will cross, metres. A car is about 1.8 m across
/// its bonnet and a counter is under a metre; past two and a half metres the
/// thing is a platform to climb onto rather than an object to go over.
const VAULT_MAX_DEPTH_M: f64 = 2.50;
/// How far off the character's own feet the far side may be and still be "the
/// ground on the other side", metres. A kerb's worth.
const VAULT_LAND_TOLERANCE_M: f64 = 0.35;

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
    // **The mantle's animation plays at the remap's rate** (wave CHAR1b.2).
    // `HeightRemap::resolve` has answered both halves of ALS's
    // `MantleParams` — `StartingPosition` and `PlayRate`
    // (`ALSMantleComponent.cpp:103-111`) — since P29.4, and neither had a
    // consumer because the machine played no mantle clip at all. The rate rides
    // the same `play_rate` seam stride warping uses, which is why a mantle does
    // not need a second one: `step_mantle` returns before step 12a, so the
    // gait's ratio never overwrites this.
    cm.runtime.play_rate = if m.play_rate.is_finite() && m.play_rate > 0.0 {
        m.play_rate
    } else {
        1.0
    };
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
    //    **The precedence is THIS one** (P29.6 audit, A11). The Tick pass runs
    //    BEFORE the movement step in both hosts, so this write lands last and a
    //    Blueprint's `anim.set_param` on one of the nine published names is
    //    overwritten before the pose step reads it. One authority for one fact;
    //    a game that wants to drive `speed` itself takes the movement component
    //    off. Pinned by `projector_mirror` and by `live_tuning`.
    //
    //    Costs one map lookup on a character with no machine, which is every
    //    character in every committed level before this wave.
    //
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
        cover_sweeps: 0,
    })
}

/// **Whether this character's feet belong on the ground at all** (wave CHAR1b.1)
/// — ALS's `UpdateFootIK` branch, as a predicate.
///
/// ALS asks two questions in sequence, `MovementState.InAir()` and
/// `MovementState.Ragdoll()`, because those are the only two of its five states
/// that are not on the floor. This engine's catalogue has fourteen modes, so the
/// same question is asked of the mode table
/// ([`inf_ecs::components::MovementMode::is_grounded_family`]: `Grounded`,
/// `Crouch`, `Prone`, `Slide`, `Roll`) plus the runtime's own `grounded` flag —
/// which is what tells a walking character from one who has just stepped off a
/// kerb and is, for three frames, in `Grounded` with nothing under it.
///
/// The mode table is the authority rather than a list spelled here, so a mode
/// added to the catalogue is airborne-or-not in **one** place. `Mantle`,
/// `Ragdoll`, `SwimSurface`, `SwimUnder`, `Driving`, `Flying`, the two falls and
/// `Dive` are all outside the family and all release.
fn feet_are_planted(cm: &CharacterMovement) -> bool {
    cm.mode.is_grounded_family() && cm.runtime.grounded
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
/// # The gate, and the wave that found it was shut (CHAR1b.1)
///
/// This function used to open with `anim_curve(…, ENABLE_FOOT_IK_L, 0.0)` and
/// `continue` on a non-positive value, and its own docs called that "inert on a
/// character whose clips carry no curve channels … which is what keeps every
/// committed sample byte-identical". It kept rather more than that: **no clip in
/// the engine has ever carried the channel** — 164 imported ALS clips and 12
/// committed sample clips, all measured, all zero — so the probe never ran, no
/// goal was ever published, and P29.4's whole ported mechanism was dead code
/// wearing a curve's name. See [`inf_anim::derive::FOOT_GROUND_BAND_M`] for the
/// census and for why the gate cannot be derived from the clip either.
///
/// The gate is now ALS's own, taken where ALS takes it —
/// `UALSCharacterAnimInstance::UpdateFootIK`:
///
/// ```text
/// if (MovementState.InAir())        { SetPelvisIKOffset(0,0); ResetIKOffsets(); }
/// else if (!MovementState.Ragdoll()) { …the foot offsets… }
/// ```
///
/// so an **airborne, ragdolling, mantling, swimming, driving or flying**
/// character releases everything ([`feet_are_planted`]), and a grounded one runs
/// the probe. An authored `Enable_FootIK_*` curve still wins where a clip
/// carries one — that is the door a project uses to author a state with no foot
/// IK — and a clip that carries none is read as **on**, which is what ALS's own
/// content means by authoring `1` on every grounded clip it ships.
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
    // ALS's own reset branch, and the "no rig" case with it: release whatever was
    // held, so a character that leaves the ground — or loses its skeleton — does
    // not leave a foot pinned to a spot on the floor. `set_foot_ik` with two
    // `None`s is what withdraws the goals the pose step is holding.
    let planted = feet_are_planted(cm);
    let feet = planted
        .then(|| inf_ecs::anim_bridge::feet_of(world, guid))
        .flatten();
    let Some(feet) = feet else {
        cm.runtime.foot_lock_l.release();
        cm.runtime.foot_lock_r.release();
        cm.runtime.foot_slide_l_m = 0.0;
        cm.runtime.foot_slide_r_m = 0.0;
        cm.runtime.pelvis_offset = Vec3d::ZERO;
        inf_ecs::anim_bridge::set_foot_ik(world, guid, [None, None]);
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
        // **`None` is not `0.0`** — see this function's docs. A clip that authors
        // the gate is obeyed; a clip that says nothing is read as on.
        //
        // ── THE SWING FOOT (wave CHAR1b.2, clause 6) ─────────────────────
        //
        // What "says nothing" fell back to was a flat `1.0`, and that is what the
        // island's sprint residual was made of. Measured, on the hero, at
        // 6.5 m/s: the goal is the ground under the foot, so a foot in the middle
        // of its SWING — rising off the toe and travelling forward — was handed a
        // target on the road and the solver spent every step dragging it back
        // down. The residual peaked at **150.0 mm** on exactly the steps where
        // `FootLock_*` had just gone to zero, and read **0.00 mm** on the steps
        // where it was one. Foot IK was being asked to pin a foot that was in the
        // air.
        //
        // ALS does not have this problem because a UE animator authors
        // `Enable_FootIK_L/R` to zero across the swing. No clip in this engine
        // carries it (the census in `char1b_gate`), and CHAR1b.1 proved the gate
        // is not derivable from a foot's HEIGHT — an in-air cycle is authored
        // with the root on the ground and its ankles sit where a walk's do. It
        // *is* derivable from the foot's own **plant window**, which the P29.5
        // deriver writes onto every clip as `FootLock_*`: that curve is the
        // animator's own statement that this foot is down, which is the same
        // statement `Enable_FootIK` makes.
        //
        // So the gate is the authored curve when there is one, the derived plant
        // window when there is not, and a flat `1.0` only when the clip carries
        // neither — which keeps every pre-CHAR1b.2 fixture behaving exactly as it
        // did.
        //
        // **Chased, not switched.** `FootLock_*` is `Step`-interpolated, so it
        // goes 0 -> 1 on one frame; a weight that followed it would snap the
        // ankle. `interp_to` at ALS's own `IK_ResetInterpSpeed` is what
        // `ResetIKOffsets` (`ALSCharacterAnimInstance.cpp:320-340`) blends foot
        // IK out with, and it is the constant used here — one of the six ALS
        // interpolation speeds P29.4 ported and nothing had called.
        let want = f64::from(
            inf_ecs::anim_bridge::anim_curve_opt(world, guid, enable_name)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0),
        );
        let lock_curve = inf_ecs::anim_bridge::anim_curve(world, guid, lock_name, 0.0);
        // **The curve IS the ramp.** ALS applies `Enable_FootIK_*` straight to
        // the offset alpha (`SetFootOffsets`, `ALSCharacterAnimInstance.cpp:
        // 464-535`) and the animator's curve carries its own ease; the deriver
        // writes this one with `Linear` interpolation for exactly that reason.
        // An `interp_to` chase on top was measured and is a ceiling, not a
        // smoother: at `IK_ResetInterpSpeed` a stance of 0.1 s reaches only
        // 0.78, so the island's sprint gate peaked at **0.8632** and the walk's
        // at 0.9711 — the foot was never fully placed at any gait, and the
        // residual measured a solve that had been told to do most of nothing.
        let w = &mut cm.runtime.foot_ik_weight[side];
        *w = want;
        // **Withdrawn, not merely faint.** `interp_to` is an exponential chase
        // and never reaches zero, so a weight left to decay keeps publishing a
        // goal at a strength that moves nothing — and the residual then measures
        // "the IK did not run" while reading like "the IK missed". Below the
        // floor the goal is dropped and no residual is published for that foot,
        // which is what "the mechanism deliberately did not run" has to look
        // like. Measured before it: an idle whose gate is 0 published 844 goals
        // at ~0 weight and a 19.65 mm mean residual that nothing was solving.
        let enable = if *w < FOOT_IK_MIN_WEIGHT {
            0.0
        } else {
            *w as f32
        };
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
            // **The two angles `ground_offset` has answered since P29.4 and
            // nobody carried** (wave CHAR1b.1). Passed on raw; the pose step
            // clamps them, because the limit is the ankle's and not the ground's.
            pitch_deg: g.pitch_deg,
            roll_deg: g.roll_deg,
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

// ── driving and flight (P29.7) ──────────────────────────────────────────────

/// Which vehicles already have somebody in them, so two characters cannot climb
/// into one seat.
///
/// `O(characters)`, and only on the step somebody presses the enter control.
fn occupied_seats(world: &EcsWorld, except: uuid::Uuid) -> BTreeSet<uuid::Uuid> {
    let mut out = BTreeSet::new();
    for guid in model::movement_targets(world) {
        if guid == except {
            continue;
        }
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if let Some(cm) = world.world().get::<CharacterMovement>(e) {
            if cm.runtime.seat.is_seated() {
                out.insert(cm.runtime.seat.vehicle);
            }
        }
    }
    out
}

/// The compass yaw of a rotation, degrees — its forward axis, flattened.
///
/// `planar_yaw_deg` rather than `DQuat::to_euler`: the latter reaches
/// `f64::atan2` and this number reaches `body_yaw_deg`, the transform, the
/// replay trace and the camera. P14's law.
fn yaw_of(rot: DQuat) -> f64 {
    let f = rot * DVec3::Z;
    model::planar_yaw_deg(Vec2d::new(f.x, f.z))
}

/// **The seat step**: warp in, drive, and get back out (P29.7).
///
/// # The choreography, and `WarpWindow`'s first consumer
///
/// Entering is a **motion warp onto the seat**: the character's transform is
/// interpolated from where it stood to where the seat is, over a *window* of the
/// enter clip rather than over the whole of it ([`inf_anim::WarpWindow`], named
/// as a zero-caller by the P29.4, P29.5 and P29.6 ledgers). The window is what
/// makes it a choreography instead of a slide: before it opens the character is
/// still standing (the clip's approach), after it closes the character is seated
/// and the rest of the clip plays out. `warp_ease` — the quintic the mantle uses
/// — shapes the inside.
///
/// The **collider is parked** for the whole of it, through the same
/// `set_collider_enabled` door the ragdoll uses, which is a physics-world
/// operation both hosts make identically.
///
/// # The exit, and the velocity handoff
///
/// A character stepping out of a car doing 20 m/s is doing 20 m/s — the
/// ragdoll's precedent exactly, and the reason `Driving` has an airborne
/// destination in the mode table. Below walking pace the exit is a stand.
fn step_driving(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    mut cm: CharacterMovement,
    dt: f64,
    radius: f64,
    overlays: &model::OverlayRegistry,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    let mut refusal = MovementRefusal::None;
    cm.runtime.time_in_mode_s += dt;
    cm.runtime.time_since_land_s += dt;
    cm.runtime.seat.time_s += dt;
    // The aim keeps integrating: the camera reads it, and a driver still looks
    // around. It is the same three lines as the standing step's, deliberately —
    // the look control means one thing in this engine.
    let prev_aim = cm.runtime.aim_yaw_deg;
    cm.runtime.aim_yaw_deg = wrap_deg(cm.runtime.aim_yaw_deg + cm.runtime.intent_look_yaw_dps * dt);
    cm.runtime.aim_pitch_deg =
        (cm.runtime.aim_pitch_deg + cm.runtime.intent_look_pitch_dps * dt).clamp(-89.0, 89.0);
    cm.runtime.aim_yaw_rate_dps =
        (model::angle_delta_deg(cm.runtime.aim_yaw_deg, prev_aim) / dt).abs();

    let vehicle = cm.runtime.seat.vehicle;
    let Some((seat_world, rot, linvel)) = super::vehicle::seat_pose(bridge, vehicle) else {
        // The vehicle is gone — despawned, or a level that changed underneath a
        // running session. A refusal is a value: the character stands up where
        // it is rather than the step failing.
        return finish_driving(
            world,
            bridge,
            guid,
            cm,
            overlays,
            None,
            MovementMode::Grounded,
        );
    };

    // ── the controls, through the trait. This is the whole of "input routes to
    //    the vehicle": a `VehicleControls` and nothing vehicle-shaped anywhere
    //    in the movement model.
    let forward_mps = linvel.dot(rot * DVec3::Z);
    let controls = inf_ecs::vehicle::VehicleControls::from_intent(
        cm.runtime.intent_move,
        forward_mps,
        cm.runtime.want_handbrake,
        cm.runtime.intent_vertical,
    );
    if let Some(v) = bridge.vehicle_mut(vehicle) {
        v.control(controls);
    }

    let (enter_time_s, window) = bridge.vehicle_of(vehicle)?.seat_warp();
    let target = seat_world + DVec3::Y * (cm.stand_half_height_m + radius);
    let chassis_yaw = yaw_of(rot);
    let position = if cm.runtime.seat.entering {
        let alpha = f64::from(window.alpha(cm.runtime.seat.time_s as f32));
        let eased = inf_anim::warp_ease(alpha);
        let start = cm.runtime.seat.start.to_dvec3();
        cm.runtime.body_yaw_deg = wrap_deg(
            cm.runtime.seat.start_yaw_deg
                + model::angle_delta_deg(chassis_yaw, cm.runtime.seat.start_yaw_deg) * eased,
        );
        if cm.runtime.seat.time_s >= enter_time_s {
            cm.runtime.seat.entering = false;
        }
        start + (target - start) * eased
    } else {
        cm.runtime.body_yaw_deg = chassis_yaw;
        target
    };
    cm.runtime.velocity = Vec3d::from_dvec3(linvel);
    cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;
    // Seated is supported: a driver is not falling, whatever the car is doing.
    cm.runtime.grounded = true;
    cm.runtime.ground_normal = Vec3d::new(0.0, 1.0, 0.0);
    cm.runtime.bank_deg = 0.0;

    // ── the exit. Not during the warp: a control that could interrupt its own
    //    choreography would leave the character half-way to a seat it is no
    //    longer in.
    let leaving = cm.runtime.press_interact && !cm.runtime.seat.entering;
    clear_edges(&mut cm);
    if leaving {
        let half_width = world
            .entity_of(vehicle)
            .and_then(|e| world.world().get::<Collider3D>(e).copied())
            .map(|c| match c.shape_kind {
                ColliderShape3DKind::Sphere => c.radius,
                _ => c.half_extents.x,
            })
            .unwrap_or(1.0);
        let out_pos =
            target + (rot * DVec3::X) * (half_width + super::vehicle::EXIT_CLEARANCE_M + radius);
        // The handoff: a moving vehicle's exit inherits its velocity, and above
        // walking pace that means the character is airborne rather than standing.
        let moving = linvel.length() > 2.0;
        let to = if moving {
            MovementMode::FallControlled
        } else {
            MovementMode::Grounded
        };
        let verdict = model::request_mode(cm.mode, to, true, true);
        if verdict.refusal == MovementRefusal::None {
            return finish_driving(
                world,
                bridge,
                guid,
                cm,
                overlays,
                Some(out_pos),
                verdict.mode,
            );
        }
        // A refused exit keeps the character in the seat — the refusal is a
        // value and the door is the mode table, so a destination the table does
        // not allow leaves the driver driving rather than half out of a car in a
        // mode nobody sanctioned.
        refusal = verdict.refusal;
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
    }

    write_driver_back(world, entity, guid, bridge, &cm, position, overlays);
    Some(MoveOutcome {
        guid,
        mode: MovementMode::Driving,
        refusal,
        grounded: true,
        landed: LandingKind::None,
        cover_sweeps: 0,
    })
}

/// **Take somebody out of a seat they did not choose to leave** (wave VEH2b) —
/// the carjack's half of the exit.
///
/// [`finish_driving`] with the mode **taken rather than requested**, which is
/// the difference between getting out of a car and being pulled out of one:
/// `inf_ecs::movement::transition_is_legal`'s own comment says the table
/// *"permits a driver to be pulled out of a seat by something that is a fact
/// about its body rather than a choice"*, and this is that something.
///
/// It does everything the ordinary exit does — restores the collider, clears
/// [`inf_ecs::components::SeatState`], places the body and republishes its
/// animation parameters — so the seat is genuinely free afterwards and
/// `occupied_seats` says so on the same step.
///
/// `false` when the guid has no body or is not in a seat, which are both things
/// a caller can produce by pressing at the wrong moment.
pub(crate) fn eject_from_seat(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    at: DVec3,
    mode: MovementMode,
    overlays: &model::OverlayRegistry,
) -> bool {
    let Some(e) = world.entity_of(guid) else {
        return false;
    };
    let Some(cm) = world.world().get::<CharacterMovement>(e).cloned() else {
        return false;
    };
    if !cm.runtime.seat.is_seated() {
        return false;
    }
    finish_driving(world, bridge, guid, cm, overlays, Some(at), mode).is_some()
}

/// Consume every edge a seated or flying character does not act on, so a press
/// held over a mode change does not fire the moment it ends.
fn clear_edges(cm: &mut CharacterMovement) {
    cm.runtime.press_interact = false;
    cm.runtime.press_fly = false;
    cm.runtime.press_jump = false;
    cm.runtime.press_crouch = false;
    cm.runtime.press_prone = false;
    cm.runtime.press_roll = false;
    cm.runtime.press_dive = false;
    // I6. A driver has no weapon and no door to kick, so an attack or a reload
    // pressed at the wheel must not fire the moment they get out — the P29.7 A1
    // class exactly, extended to the two edges this wave added. The wheel's sign
    // goes with them for the same reason.
    cm.runtime.press_attack = false;
    cm.runtime.press_reload = false;
    cm.runtime.weapon_switch = 0;
}

/// Leave the seat: restore the collider, place the character, keep the velocity.
///
/// `at` is `None` when the vehicle itself vanished, in which case the character
/// stays exactly where it was — the honest answer, and the one that cannot put
/// it inside geometry it never travelled through.
fn finish_driving(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    mut cm: CharacterMovement,
    overlays: &model::OverlayRegistry,
    at: Option<DVec3>,
    mode: MovementMode,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    super::vehicle::park_collider(bridge, guid, false);
    cm.runtime.seat = inf_ecs::components::SeatState::default();
    cm.mode = mode;
    cm.runtime.time_in_mode_s = 0.0;
    let position = at.unwrap_or_else(|| {
        world
            .world()
            .get::<Transform>(entity)
            .map(|t| t.translation.to_dvec3())
            .unwrap_or_default()
    });
    cm.runtime.grounded = mode.is_grounded_family();
    write_driver_back(world, entity, guid, bridge, &cm, position, overlays);
    Some(MoveOutcome {
        guid,
        mode,
        refusal: MovementRefusal::None,
        grounded: cm.runtime.grounded,
        landed: LandingKind::None,
        cover_sweeps: 0,
    })
}

/// The write-back the seat and flight steps share — step 12 and 12b of the
/// standing step, over a character that has no capsule resize to do.
fn write_driver_back(
    world: &mut EcsWorld,
    entity: inf_ecs::Entity,
    guid: uuid::Uuid,
    bridge: &mut PhysicsBridge3D,
    cm: &CharacterMovement,
    position: DVec3,
    overlays: &model::OverlayRegistry,
) {
    {
        let w = world.world_mut();
        if let Some(mut t) = w.get_mut::<Transform>(entity) {
            t.translation.x = position.x;
            t.translation.y = position.y;
            t.translation.z = position.z;
            t.rotation.y = cm.runtime.body_yaw_deg;
            // The roll is the BANK, and it is zero in every mode but flight —
            // which is why writing it here is safe: a character that lands with
            // a bank on would keep it for ever otherwise.
            t.rotation.z = cm.runtime.bank_deg;
        }
        if let Some(mut slot) = w.get_mut::<CharacterMovement>(entity) {
            *slot = cm.clone();
        }
    }
    let overlay = overlays.id_of(&cm.overlay);
    inf_ecs::anim_bridge::publish_character_params(world, guid, cm, overlay);
    if let Some(body) = bridge.body_of(guid) {
        bridge.world_mut().set_body_translation(body, position);
    }
    world.mark_dirty();
}

/// **The flight step** (P29.7): six degrees of freedom, no gravity, banking.
///
/// The nearest precedent in this engine is the controlled-fall authority model,
/// and the difference is the whole point: a fall integrates gravity and clamps
/// the player's authority over it, while flight has **no gravity term at all**
/// and the player's authority is total. `inf_ecs::movement::integrate_flight` is
/// that, as a function of numbers.
///
/// It still moves through `PhysicsWorld3D::move_character`, so a flying
/// character collides with the world rather than through it. The **capsule stays
/// upright** in that sweep while the transform banks: the bank is a visual and
/// an animation input, and tilting a kinematic character's swept shape would
/// turn a bank into a squeeze through gaps it should not fit.
#[allow(clippy::too_many_arguments)]
fn step_flight(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    mut cm: CharacterMovement,
    dt: f64,
    mut position: DVec3,
    half_height: f64,
    radius: f64,
    is_capsule: bool,
    overlays: &model::OverlayRegistry,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    cm.runtime.time_in_mode_s += dt;
    cm.runtime.time_since_land_s += dt;

    // The aim, and the SIGNED rate the bank comes from. `aim_yaw_rate_dps` is an
    // absolute value (three ALS systems read it as a magnitude), so the sign is
    // recovered here rather than by changing what that field means.
    let prev_aim = cm.runtime.aim_yaw_deg;
    cm.runtime.aim_yaw_deg = wrap_deg(cm.runtime.aim_yaw_deg + cm.runtime.intent_look_yaw_dps * dt);
    cm.runtime.aim_pitch_deg =
        (cm.runtime.aim_pitch_deg + cm.runtime.intent_look_pitch_dps * dt).clamp(-89.0, 89.0);
    let signed_rate = model::angle_delta_deg(cm.runtime.aim_yaw_deg, prev_aim) / dt;
    cm.runtime.aim_yaw_rate_dps = signed_rate.abs();

    // The wish velocity, in the aim frame and pitched by where the character is
    // looking — which is what makes this six degrees of freedom rather than a
    // hover with a lift button.
    let pitch = cm.runtime.aim_pitch_deg.to_radians();
    let (sp, cp) = (inf_math::psin64(pitch), inf_math::pcos64(pitch));
    let flat = model::rotate_from_frame(Vec2d::new(0.0, 1.0), cm.runtime.aim_yaw_deg);
    let forward = DVec3::new(flat.x * cp, sp, flat.y * cp);
    let side = model::rotate_from_frame(Vec2d::new(1.0, 0.0), cm.runtime.aim_yaw_deg);
    let right = DVec3::new(side.x, 0.0, side.y);
    let wish = forward * cm.runtime.intent_move.y * model::FLY_SPEED_MPS
        + right * cm.runtime.intent_move.x * model::FLY_SPEED_MPS
        + DVec3::Y * cm.runtime.intent_vertical * model::FLY_ASCEND_MPS;
    cm.runtime.velocity = model::integrate_flight(cm.runtime.velocity, Vec3d::from_dvec3(wish), dt);

    // Move, with collision — but with **no ground snap and no autostep**. Both
    // exist to keep a walking character attached to a floor, and both are wrong
    // for flight: measured before this, a hovering character sank 4.8 cm per
    // second because rapier's snap pulled it toward the ground it was flying
    // over, which is exactly the "no gravity" claim failing by another route.
    let mover = mover_for_with_capsule(world, guid, is_capsule.then_some((half_height, radius)))
        .snap_to_ground(None)
        .autostep(None);
    let exclude = bridge.collider_of(guid);
    let result = bridge.world_mut().move_character(
        &mover,
        position,
        cm.runtime.velocity.to_dvec3() * dt,
        exclude,
    );
    position += result.translation;
    // A flying character is never "grounded": the mode is the authority on how
    // it is being integrated, and a hover a centimetre over a roof is still
    // flight.
    cm.runtime.grounded = false;
    cm.runtime.ground_normal = Vec3d::new(0.0, 1.0, 0.0);

    // The body faces where it is looking, and banks into the turn.
    cm.runtime.body_yaw_deg = cm.runtime.aim_yaw_deg;
    cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;
    let target_bank = model::bank_target_deg(signed_rate);
    let k = (model::BANK_INTERP_PER_S * dt).clamp(0.0, 1.0);
    cm.runtime.bank_deg += (target_bank - cm.runtime.bank_deg) * k;

    // The derived outputs the animation bridge reads, in flight's own terms.
    let planar = (cm.runtime.velocity.x * cm.runtime.velocity.x
        + cm.runtime.velocity.z * cm.runtime.velocity.z)
        .sqrt();
    cm.runtime.mapped_speed = model::mapped_speed(
        planar,
        cm.walk_speed_mps,
        cm.run_speed_mps,
        cm.sprint_speed_mps,
    );
    cm.runtime.gait_scalar = model::gait_scalar(cm.runtime.mapped_speed);
    cm.runtime.actual_gait = Gait::Run;
    cm.runtime.landing = LandingKind::None;
    clear_edges(&mut cm);

    write_driver_back(world, entity, guid, bridge, &cm, position, overlays);
    Some(MoveOutcome {
        guid,
        mode: MovementMode::Flying,
        refusal: MovementRefusal::None,
        grounded: false,
        landed: LandingKind::None,
        cover_sweeps: 0,
    })
}
