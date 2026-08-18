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
    CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait, Guid,
    LandingKind, MovementMode, MovementRefusal, RotationMode, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement as model;
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;
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
    if !(dt > 0.0) || !dt.is_finite() {
        return Vec::new();
    }
    let mut targets: Vec<uuid::Uuid> = Vec::new();
    {
        let w = world.world();
        for e in w.iter_entities() {
            if e.get::<CharacterMovement>().is_none() || e.get::<Transform>().is_none() {
                continue;
            }
            if let Some(g) = e.get::<Guid>().map(|g| g.0) {
                targets.push(g);
            }
        }
    }
    targets.sort_unstable();
    let mut out = Vec::with_capacity(targets.len());
    for guid in targets {
        if let Some(o) = step_one(world, bridge, guid, dt) {
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
    if !(to_half > from_half) {
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
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    let (mut cm, mut position, collider) = {
        let w = world.world();
        let cm = w.get::<CharacterMovement>(entity)?.clone();
        let t = w.get::<Transform>(entity)?;
        (
            cm,
            t.translation.to_dvec3(),
            w.get::<Collider3D>(entity).copied(),
        )
    };
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

    // ── 3. Timers.
    cm.runtime.time_in_mode_s += dt;
    cm.runtime.time_since_land_s += dt;

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
            if cm.mode == MovementMode::Crouch || cm.mode == MovementMode::Prone {
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
    let mut half_height = cm.half_height_for(cm.mode);
    if cm.mode != previous_mode && is_capsule {
        let old_half = cm.half_height_for(previous_mode);
        position.y += half_height - old_half;
        probe.centre = position;
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

    // ── 8. Move. `apply_swim_motion` is the identity when not swimming, which
    //    is why it can be unconditional — the same call `move_and_slide` makes.
    let motion = bridge.apply_swim_motion(
        guid,
        DVec3::new(planar.x * dt, vertical * dt, planar.y * dt),
        dt,
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
    } else if cm.rotation_mode == RotationMode::Aiming {
        cm.runtime.body_yaw_deg = model::limit_rotation(
            cm.runtime.body_yaw_deg,
            cm.runtime.aim_yaw_deg,
            -AIM_BODY_LIMIT_DEG,
            AIM_BODY_LIMIT_DEG,
            AIM_BODY_LIMIT_INTERP,
            dt,
        );
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
            *slot = cm;
        }
    }
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
