//! **The movement model** (P29.3) — the pure half.
//!
//! Everything here is a function of numbers: no physics world, no queries, no
//! clock. The half that needs a world to sweep against lives in
//! `inf_physics::d3::movement`, which is the one fixed-step door both hosts
//! call. The split is deliberate and it is the same split
//! [`crate::pose::step_pose_evaluation`] has: the *rules* are testable without a
//! simulation, and the simulation contains no rules.
//!
//! # What this is a port of, and what it is not
//!
//! The architecture donor is **ALS** (§13's 2026-08-17 amendment, Ruling 1). The
//! pieces taken verbatim are named at their functions: [`mapped_speed`] (the
//! normalized-speed trick that makes every curve independent of the speeds),
//! [`resolve_gait`] (desired → allowed → actual, with the hysteresis band that
//! keeps the animation in "run" until the body has actually slowed),
//! [`quadrant`] (the ±70/±110 movement-direction bands with a 5° buffer),
//! [`smooth_rotation`] (constant-rate chasing a goal, exponential chasing that),
//! and [`classify_landing`] (soft/hard/roll keyed to impact speed).
//!
//! What is deliberately **not** ported: the seven enum-mirror structs (398 lines
//! that store an enum plus one bool per variant so an AnimBP property binding can
//! skip a compare node — a `match` is free in Rust and a denormalization can
//! desync), the finite-differenced acceleration (our movement component *knows*
//! its commanded acceleration and publishes it, which removes a `dt`-sensitivity
//! from a value that reaches `state_bytes`), and every line of replication.
//!
//! # Portable math
//!
//! Everything here is adds, multiplies, comparisons, `sqrt` and the `inf_math`
//! `p*` helpers. No `std` transcendental appears, because these values reach the
//! committed replay trace through the sim's `state_bytes`. `patan2_64` is used
//! for the two places an angle is genuinely needed (the movement quadrant and
//! the sprint-direction gate).

use bevy_ecs::prelude::With;
use uuid::Uuid;

use crate::components::{
    CharacterMovement, Gait, Guid, LandingKind, MovementDirection, MovementMode, MovementRefusal,
    RotationMode, Transform,
};
use crate::math::Vec2d;
use crate::world::EcsWorld;

/// The action and axis names the movement intent reads.
///
/// They live here, once, so a project's `input.toml`, the runtime's default map
/// and both hosts' intent sampling all name the same strings. A binding spelled
/// differently in one of those places is a control that silently does nothing,
/// which is the failure this const exists to make impossible.
pub mod actions {
    /// Analog axis: strafe, `[-1, 1]`, `+` right.
    pub const MOVE_X: &str = "move_x";
    /// Analog axis: forward/back, `[-1, 1]`, `+` forward.
    pub const MOVE_Y: &str = "move_y";
    /// Analog axis: yaw rate, **degrees per second**.
    pub const LOOK_X: &str = "look_x";
    /// Analog axis: pitch rate, degrees per second.
    pub const LOOK_Y: &str = "look_y";
    /// Analog axis: vertical intent while swimming, `[-1, 1]`.
    pub const MOVE_UP: &str = "move_up";
    /// Held: run → sprint.
    pub const SPRINT: &str = "sprint";
    /// Held: run → walk.
    pub const WALK: &str = "walk";
    /// Held: aiming rotation mode.
    pub const AIM: &str = "aim";
    /// Edge: jump / stand up / leave the water.
    pub const JUMP: &str = "jump";
    /// Edge: toggle crouch (double-tapped inside
    /// [`ROLL_DOUBLE_TAP_S`](super::ROLL_DOUBLE_TAP_S), it rolls).
    pub const CROUCH: &str = "crouch";
    /// Edge: toggle prone.
    pub const PRONE: &str = "prone";
    /// Edge: roll.
    pub const ROLL: &str = "roll";
    /// Edge: dive.
    pub const DIVE: &str = "dive";
}

/// ALS's `RollDoubleTapTimeout`: two crouch presses inside this window roll.
pub const ROLL_DOUBLE_TAP_S: f64 = 0.3;

/// The gait hysteresis band, m/s. ALS uses 10 cm/s: the *actual* gait only steps
/// up once the body has genuinely exceeded the tier below, so a character
/// decelerating out of a sprint keeps running until it has really slowed.
pub const GAIT_HYSTERESIS_MPS: f64 = 0.1;

/// The movement-direction quadrant thresholds, degrees (ALS `CalculateQuadrant`).
/// The forward band is `(-70, 70)`, the back band outside `±110`, and the buffer
/// **widens the band you are in and narrows the one you are not**, so the
/// boundary cannot chatter when the input sits exactly on it.
pub const QUADRANT_FRONT_DEG: f64 = 70.0;
/// See [`QUADRANT_FRONT_DEG`].
pub const QUADRANT_BACK_DEG: f64 = 110.0;
/// See [`QUADRANT_FRONT_DEG`].
pub const QUADRANT_BUFFER_DEG: f64 = 5.0;

/// The aim-yaw-rate → rotation-rate multiplier's input band, degrees per second
/// (ALS `{0, 300}`), and its output band (ALS `{1, 3}`): a faster camera turns
/// the body faster.
pub const AIM_RATE_BAND_DPS: (f64, f64) = (0.0, 300.0);
/// See [`AIM_RATE_BAND_DPS`].
pub const AIM_RATE_MULTIPLIER: (f64, f64) = (1.0, 3.0);

/// Map `value` from the range `from` onto the range `to`, clamped, with a
/// degenerate source range answering the low end rather than an infinity.
///
/// ALS's `MappedRangeClamped`, which is the arithmetic under half of this
/// module. Kept as one function so the divide-by-zero guard exists once.
pub fn mapped_range(value: f64, from: (f64, f64), to: (f64, f64)) -> f64 {
    let span = from.1 - from.0;
    if !span.is_finite() || span.abs() < f64::EPSILON || !value.is_finite() {
        return to.0;
    }
    let t = ((value - from.0) / span).clamp(0.0, 1.0);
    to.0 + (to.1 - to.0) * t
}

/// **ALS's `GetMappedSpeed`** — horizontal speed onto a `[0, 3]` scalar where
/// `0` is stopped, `1` is the walk speed, `2` the run speed and `3` the sprint
/// speed, piecewise-linearly.
///
/// This one function is why the [`crate::components::SpeedCurve`]s can be tuned
/// once and survive every later change to a character's speeds: they are keyed
/// on *this*, not on metres per second.
///
/// The speeds are sorted into a monotone ladder first. A character authored with
/// `run < walk` is not a crash and not a clamp to zero — it is a ladder read in
/// the order it was given, and the alternative (trusting the author's order) puts
/// a negative span into `mapped_range` and reads the low end for every speed.
pub fn mapped_speed(speed: f64, walk: f64, run: f64, sprint: f64) -> f64 {
    if !speed.is_finite() {
        return 0.0;
    }
    let walk = walk.max(0.0);
    let run = run.max(walk);
    let sprint = sprint.max(run);
    if speed > run {
        mapped_range(speed, (run, sprint), (2.0, 3.0))
    } else if speed > walk {
        mapped_range(speed, (walk, run), (1.0, 2.0))
    } else {
        mapped_range(speed, (0.0, walk), (0.0, 1.0))
    }
}

/// **ALS's three-stage gait resolution**, all of it: desired → allowed → actual.
///
/// * *Desired* is pure input and is the caller's `desired`.
/// * *Allowed* clamps it — a sprint needs the standing stance, a rotation mode
///   that is not `Aiming`, an input magnitude of at least
///   [`CharacterMovement::sprint_input_min`], and (in `LookingDirection`) an
///   input direction within [`CharacterMovement::sprint_angle_deg`] of the aim.
/// * *Actual* is derived from **measured speed** with the
///   [`GAIT_HYSTERESIS_MPS`] band, so it lags during deceleration.
///
/// Returns `(allowed, actual)`, because both are read: the allowed gait picks
/// the target speed the integrator accelerates toward, and the actual gait is
/// what the animation is told the body is doing.
pub fn resolve_gait(
    cm: &CharacterMovement,
    mode: MovementMode,
    desired: Gait,
    speed_mps: f64,
    input_magnitude: f64,
    input_yaw_deg: f64,
    aim_yaw_deg: f64,
) -> (Gait, Gait) {
    let can_sprint = mode == MovementMode::Grounded
        && cm.rotation_mode != RotationMode::Aiming
        && input_magnitude >= cm.sprint_input_min
        && match cm.rotation_mode {
            RotationMode::LookingDirection => {
                angle_delta_deg(input_yaw_deg, aim_yaw_deg).abs() <= cm.sprint_angle_deg
            }
            _ => true,
        };
    let allowed = match desired {
        Gait::Sprint if !can_sprint => Gait::Run,
        Gait::Sprint => Gait::Sprint,
        Gait::Walk => Gait::Walk,
        // A reserved tier a newer build wrote degrades to the middle tier.
        _ => Gait::Run,
    };

    let walk = cm.walk_speed_mps;
    let run = cm.run_speed_mps.max(walk);
    let actual = if speed_mps > run + GAIT_HYSTERESIS_MPS {
        if allowed == Gait::Sprint {
            Gait::Sprint
        } else {
            Gait::Run
        }
    } else if speed_mps >= walk + GAIT_HYSTERESIS_MPS {
        Gait::Run
    } else {
        Gait::Walk
    };
    (allowed, actual)
}

/// The signed difference `a - b` folded into `(-180, 180]` degrees.
///
/// Pure arithmetic; no transcendental. Non-finite inputs answer `0`, because
/// every comparison a NaN takes part in is false and an unguarded fold would
/// loop for ever.
pub fn angle_delta_deg(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    let mut d = (a - b) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d <= -180.0 {
        d += 360.0;
    }
    d
}

/// **ALS's `CalculateQuadrant`** with its hysteresis: which quadrant of the aim
/// frame `angle_deg` (the velocity direction relative to the aim yaw) falls in,
/// given what it was last step.
///
/// The buffer is applied *against* the current answer — the band you are already
/// in is widened by [`QUADRANT_BUFFER_DEG`] and the one you are not is narrowed —
/// which is what stops a character walking exactly sideways from flickering
/// between the left and forward locomotion clips several times a second.
pub fn quadrant(angle_deg: f64, current: MovementDirection) -> MovementDirection {
    let a = if angle_deg.is_finite() {
        angle_deg
    } else {
        0.0
    };
    // Widen the band when leaving it, narrow it when entering.
    let front = if current == MovementDirection::Forward {
        QUADRANT_FRONT_DEG + QUADRANT_BUFFER_DEG
    } else {
        QUADRANT_FRONT_DEG - QUADRANT_BUFFER_DEG
    };
    let back = if current == MovementDirection::Backward {
        QUADRANT_BACK_DEG - QUADRANT_BUFFER_DEG
    } else {
        QUADRANT_BACK_DEG + QUADRANT_BUFFER_DEG
    };
    let mag = a.abs();
    if mag <= front {
        MovementDirection::Forward
    } else if mag >= back {
        MovementDirection::Backward
    } else if a > 0.0 {
        MovementDirection::Right
    } else {
        MovementDirection::Left
    }
}

/// **ALS's `SmoothCharacterRotation`**, and one of the best ideas in the donor:
/// a **constant-rate** chase of the goal feeding an **exponential** chase of that
/// intermediate.
///
/// The constant-rate stage bounds peak angular velocity, so a 179° goal change
/// does not snap; the exponential stage removes the corner when the constant
/// stage arrives. Both stages are polynomial, so this is portable unchanged.
///
/// Returns `(new_target_deg, new_actor_deg)`.
pub fn smooth_rotation(
    target_deg: f64,
    actor_deg: f64,
    goal_deg: f64,
    target_rate_dps: f64,
    actor_interp: f64,
    dt: f64,
) -> (f64, f64) {
    if !dt.is_finite() || dt <= 0.0 {
        return (target_deg, actor_deg);
    }
    // Stage 1: constant rate toward the goal.
    let to_goal = angle_delta_deg(goal_deg, target_deg);
    let step = (target_rate_dps.max(0.0) * dt).min(to_goal.abs());
    let new_target = target_deg + step * if to_goal >= 0.0 { 1.0 } else { -1.0 };
    // Stage 2: exponential toward the intermediate. `1 - exp(-k dt)` would need a
    // transcendental; the standard first-order approximation `clamp(k dt, 0, 1)`
    // is what ALS's own `FInterpTo` uses and is stable for `k dt <= 1`.
    let alpha = (actor_interp.max(0.0) * dt).clamp(0.0, 1.0);
    let to_target = angle_delta_deg(new_target, actor_deg);
    (new_target, actor_deg + to_target * alpha)
}

/// **ALS's `LimitRotation`**: if the aim-versus-body yaw delta has left
/// `[min_deg, max_deg]`, smooth the body back to the nearest bound.
///
/// Returns the new body yaw.
pub fn limit_rotation(
    body_deg: f64,
    aim_deg: f64,
    min_deg: f64,
    max_deg: f64,
    interp: f64,
    dt: f64,
) -> f64 {
    let delta = angle_delta_deg(aim_deg, body_deg);
    if delta >= min_deg && delta <= max_deg {
        return body_deg;
    }
    let bound = if delta > max_deg { max_deg } else { min_deg };
    let goal = aim_deg - bound;
    let alpha = if dt > 0.0 && dt.is_finite() {
        (interp.max(0.0) * dt).clamp(0.0, 1.0)
    } else {
        0.0
    };
    body_deg + angle_delta_deg(goal, body_deg) * alpha
}

/// **ALS's `CalculateGroundedRotationRate`**: the curve's value at the current
/// normalized speed, multiplied by an aim-yaw-rate factor so a fast camera turns
/// the body faster.
pub fn grounded_rotation_rate(cm: &CharacterMovement, mapped: f64, aim_yaw_rate_dps: f64) -> f64 {
    let base = cm.rotation_rate.sample(mapped);
    let mult = mapped_range(
        aim_yaw_rate_dps.abs(),
        AIM_RATE_BAND_DPS,
        AIM_RATE_MULTIPLIER,
    );
    base * mult
}

/// **The velocity integrator** — the CalcVelocity-equivalent that is *ours*
/// (impedance mismatch IM-2).
///
/// rapier's `KinematicCharacterController` has no velocity model at all: it
/// moves a shape and reports what it touched. UE's `CharacterMovementComponent`
/// integrates velocity from acceleration with a friction-and-braking model, and
/// that model is exactly what ALS's three curve channels drive. So the engine
/// keeps its own integrator and uses the controller purely as sweep-and-slide.
///
/// The model, per axis-free planar vector:
///
/// * With input, apply `accel × dt` toward the wish direction and clamp the
///   result to `target_speed`.
/// * With input **and** a velocity pointing somewhere else, also bleed the
///   sideways component at `friction` — this is what makes a turn feel like a
///   turn instead of a drift.
/// * Without input, decelerate at `braking + friction × speed` (the friction
///   term is proportional, so a sprint stops harder than a walk).
///
/// `friction_scale` is the landing override (ALS's `BrakingFrictionFactor`: 0.5
/// with input, 3.0 without, both released after half a second).
///
/// Returns the new planar velocity, m/s.
///
/// Eight arguments, deliberately: each is a distinct physical quantity with its
/// own unit, and three of them (`accel`, `braking`, `friction`) are the three
/// channels of ALS's movement curve. Bundling them into a struct would hide
/// which one a caller meant to change — the air branch varies exactly one — and
/// this function is the piece most likely to be read against the donor.
#[allow(clippy::too_many_arguments)]
pub fn integrate_planar_velocity(
    velocity: Vec2d,
    wish_dir: Vec2d,
    target_speed: f64,
    accel: f64,
    braking: f64,
    friction: f64,
    friction_scale: f64,
    dt: f64,
) -> Vec2d {
    if !dt.is_finite() || dt <= 0.0 {
        return velocity;
    }
    let v = (velocity.x, velocity.y);
    let wish_len = (wish_dir.x * wish_dir.x + wish_dir.y * wish_dir.y).sqrt();
    let friction = friction.max(0.0) * friction_scale.max(0.0);
    if wish_len > 1e-6 && target_speed > 0.0 {
        let dir = (wish_dir.x / wish_len, wish_dir.y / wish_len);
        // Accelerate along the wish direction.
        let mut nv = (
            v.0 + dir.0 * accel.max(0.0) * dt,
            v.1 + dir.1 * accel.max(0.0) * dt,
        );
        // Bleed the component that is NOT along the wish direction, so turning
        // costs something. Without this a character carries its old velocity
        // sideways for as long as the accelerating one takes to dominate.
        let along = nv.0 * dir.0 + nv.1 * dir.1;
        let side = (nv.0 - dir.0 * along, nv.1 - dir.1 * along);
        let keep = (1.0 - friction * dt).clamp(0.0, 1.0);
        nv = (dir.0 * along + side.0 * keep, dir.1 * along + side.1 * keep);
        // Clamp to the target speed — but only downward, so an external impulse
        // (a slide, a dive) is not erased by walking.
        let len = (nv.0 * nv.0 + nv.1 * nv.1).sqrt();
        if len > target_speed && len > 1e-12 {
            let cap = target_speed.max(0.0);
            // If the character was ALREADY faster than the target, do not snap
            // it down to the target — decelerate at the braking rate instead.
            let old_len = (v.0 * v.0 + v.1 * v.1).sqrt();
            let allowed = if old_len > cap {
                (old_len - braking.max(0.0) * dt).max(cap)
            } else {
                cap
            };
            let k = allowed / len;
            nv = (nv.0 * k, nv.1 * k);
        }
        Vec2d::new(nv.0, nv.1)
    } else {
        // No input: brake at a constant rate plus a speed-proportional friction.
        let len = (v.0 * v.0 + v.1 * v.1).sqrt();
        if len <= 1e-9 {
            return Vec2d::ZERO;
        }
        let decel = braking.max(0.0) + friction * len;
        let new_len = (len - decel * dt).max(0.0);
        let k = new_len / len;
        Vec2d::new(v.0 * k, v.1 * k)
    }
}

/// **The landing classifier** (§13's catalogue row, verbatim): soft / hard /
/// roll, keyed to **impact speed** rather than to whichever animation was
/// playing.
///
/// `impact_mps` is the downward speed at the moment of contact (a positive
/// magnitude). `has_input` splits the middle band the way ALS does: a landing
/// with movement input break-falls into a roll, a landing without it plants.
pub fn classify_landing(cm: &CharacterMovement, impact_mps: f64, has_input: bool) -> LandingKind {
    let impact = if impact_mps.is_finite() {
        impact_mps.abs()
    } else {
        0.0
    };
    if impact >= cm.land_ragdoll_mps {
        LandingKind::Ragdoll
    } else if impact >= cm.land_hard_mps {
        if has_input {
            LandingKind::Roll
        } else {
            LandingKind::Hard
        }
    } else {
        LandingKind::Soft
    }
}

/// The braking-friction multiplier owed to a landing that happened
/// `since_land_s` ago (ALS `EventOnLanded` + `OnLandFrictionReset`).
///
/// `1.0` — no override — once the window has passed, which is the "both return
/// to 0" half of ALS's rule expressed as a multiplier rather than as an absolute
/// so it composes with the curve.
pub fn landing_friction_scale(cm: &CharacterMovement, since_land_s: f64, has_input: bool) -> f64 {
    // **Nothing has landed yet** (P29.3 audit, A5). `time_since_land_s` starts
    // at zero like every other runtime field, so a character that had never
    // left the ground read as one that had landed *this instant* and spent its
    // first half-second under the post-landing override — a plant it never
    // earned. The latch is `landing`, which the step writes in the same breath
    // as the timer, so it is the honest question to ask.
    if cm.runtime.landing == LandingKind::None {
        return 1.0;
    }
    if !since_land_s.is_finite() || since_land_s >= cm.land_friction_time_s {
        return 1.0;
    }
    if has_input {
        cm.brake_friction_input
    } else {
        cm.brake_friction_idle
    }
}

/// The slide's **friction-versus-slope curve** (§13's slide row).
///
/// Level ground gets [`CharacterMovement::slide_friction_flat`] and a slope at
/// the character's limit gets [`CharacterMovement::slide_friction_slope`],
/// linearly between — so a slide downhill keeps going and a slide on the flat
/// runs out. `slope_deg` is the angle of the surface under the character.
pub fn slide_friction(cm: &CharacterMovement, slope_deg: f64) -> f64 {
    mapped_range(
        slope_deg,
        (0.0, cm.slope_limit_deg.max(1.0)),
        (cm.slide_friction_flat, cm.slide_friction_slope),
    )
}

/// **The `W_Gait` bias-and-clamp trick**: one curve read as two independent
/// `[0, 1]` factors.
///
/// ALS stores a single `W_Gait` channel valued 1 at a walk, 2 at a run and 3 at
/// a sprint, then extracts "how far from walk toward run" and "how far from run
/// toward sprint" by biasing and clamping the same number. The scalar here is
/// the same idea on our `[0, 3]` axis: `0` at a walk, `1` at a run, `2` at a
/// sprint, continuous between, so a consumer clamps `[0, 1]` for the walk/run
/// factor and `x - 1` clamped for the run/sprint one.
pub fn gait_scalar(mapped: f64) -> f64 {
    (mapped.clamp(0.0, 3.0) - 1.0).max(0.0)
}

/// **Stride blend** `[0, 1]`: how far the locomotion clip's stride is scaled, so
/// a character moving slower than the clip was animated at does not skate.
///
/// Derived from the normalized speed rather than from metres per second, for the
/// same reason every other curve here is.
pub fn stride_blend(mapped: f64) -> f64 {
    mapped_range(mapped, (0.0, 2.0), (0.0, 1.0))
}

/// **Walk↔run blend** `[0, 1]` — 0 at or below the walk anchor, 1 at or above
/// the run anchor.
pub fn walk_run_blend(mapped: f64) -> f64 {
    mapped_range(mapped, (1.0, 2.0), (0.0, 1.0))
}

/// **Relative acceleration** (ALS's `CalculateRelativeAccelerationAmount`): the
/// character's own acceleration expressed in its own frame and normalized
/// against the *current* curve-derived maxima, so `±1` means "as hard as this
/// character can, at this speed".
///
/// The port map's IM-4: ALS finite-differences this from velocity, which is
/// noisy at a variable frame rate and carries a networking hack. Our integrator
/// knows what it commanded, so this takes the commanded acceleration directly.
pub fn relative_acceleration(
    accel_world: Vec2d,
    body_yaw_deg: f64,
    max_accel: f64,
    max_braking: f64,
    decelerating: bool,
) -> Vec2d {
    let max = if decelerating { max_braking } else { max_accel };
    if !max.is_finite() || max <= 0.0 {
        return Vec2d::ZERO;
    }
    let local = rotate_into_frame(accel_world, body_yaw_deg);
    Vec2d::new(
        (local.x / max).clamp(-1.0, 1.0),
        (local.y / max).clamp(-1.0, 1.0),
    )
}

/// Rotate a world-XZ planar vector into a frame yawed by `yaw_deg`.
///
/// `x` is the frame's right and `y` its forward. Uses the portable sine/cosine
/// helpers, because this result reaches the pose and therefore `state_bytes`.
pub fn rotate_into_frame(v: Vec2d, yaw_deg: f64) -> Vec2d {
    let r = yaw_deg.to_radians();
    let (s, c) = (inf_math::psin64(r), inf_math::pcos64(r));
    // World `+Z` is the frame's forward at yaw 0, `+X` its right.
    Vec2d::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// Rotate a planar vector **out of** a frame yawed by `yaw_deg` into world XZ —
/// the inverse of [`rotate_into_frame`].
pub fn rotate_from_frame(v: Vec2d, yaw_deg: f64) -> Vec2d {
    let r = yaw_deg.to_radians();
    let (s, c) = (inf_math::psin64(r), inf_math::pcos64(r));
    Vec2d::new(v.x * c + v.y * s, -v.x * s + v.y * c)
}

/// The compass yaw of a planar world vector, degrees, with `+Z` as zero and
/// `+X` as +90 — the same convention [`rotate_into_frame`] uses.
///
/// `patan2_64` rather than `f64::atan2`: this feeds the movement quadrant, which
/// reaches the animation blend and therefore the replay trace.
pub fn planar_yaw_deg(v: Vec2d) -> f64 {
    if v.x.abs() < 1e-12 && v.y.abs() < 1e-12 {
        return 0.0;
    }
    inf_math::patan2_64(v.x, v.y).to_degrees()
}

/// **The settings block** ALS selects on `RotationMode × Stance`, resolved.
///
/// The two-level dispatch is kept because it is the load-bearing shape: the
/// gait picks a *speed* inside a block chosen by rotation mode and stance, so
/// retuning one does not disturb the other. What is flattened is the storage —
/// see [`CharacterMovement`]'s docs for exactly what that costs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementSettings {
    /// Target speed for the resolved gait, m/s, after the rotation-mode scale.
    pub target_speed_mps: f64,
    /// Max acceleration at the current normalized speed, m/s².
    pub acceleration_mps2: f64,
    /// Max braking deceleration at the current normalized speed, m/s².
    pub braking_mps2: f64,
    /// Ground friction at the current normalized speed, 1/s.
    pub friction: f64,
    /// Body turn rate at the current normalized speed, degrees per second.
    pub rotation_rate_dps: f64,
}

/// Resolve the settings for a `(rotation mode, mode, gait)` at a normalized
/// speed — ALS's `GetTargetMovementSettings` plus `GetSpeedForGait`.
pub fn settings_for(
    cm: &CharacterMovement,
    mode: MovementMode,
    gait: Gait,
    mapped: f64,
    aim_yaw_rate_dps: f64,
) -> MovementSettings {
    MovementSettings {
        target_speed_mps: cm.speed_for(mode, gait) * cm.rotation_speed_scale(),
        acceleration_mps2: cm.acceleration.sample(mapped),
        braking_mps2: cm.braking.sample(mapped),
        friction: cm.ground_friction.sample(mapped),
        rotation_rate_dps: grounded_rotation_rate(cm, mapped, aim_yaw_rate_dps),
    }
}

// ── turn / rotate in place, aim offsets, traversal timing (P29.4) ───────────

/// The aim-versus-body angle past which a **turn in place** may begin, degrees
/// (ALS `TurnCheckMinAngle`).
pub const TURN_CHECK_MIN_DEG: f64 = 45.0;
/// The angle past which the turn is a 180 rather than a 90 (ALS
/// `Turn180Threshold`) — which animation plays, and therefore how far the warp
/// has to rescale it.
pub const TURN_180_DEG: f64 = 130.0;
/// The camera must be **settled** for a turn in place to start: an aim yaw rate
/// above this says the player is still looking around (ALS `AimYawRateLimit`).
pub const TURN_AIM_RATE_LIMIT_DPS: f64 = 50.0;
/// The longest the delay accumulator can be asked to reach, seconds — the delay
/// owed to a 180° offset (ALS `MaxAngleDelay`). A 45° offset waits none at all.
pub const TURN_MAX_DELAY_S: f64 = 0.75;
/// The aim-versus-body angle past which **rotate in place** engages while
/// aiming, degrees (ALS `RotateMinThreshold` / `RotateMaxThreshold` = ∓50).
pub const ROTATE_IN_PLACE_DEG: f64 = 50.0;
/// The aim-yaw-rate band a rotate-in-place's play rate is mapped from, deg/s
/// (ALS `{AimYawRateMinRange, AimYawRateMaxRange}`).
pub const ROTATE_RATE_BAND_DPS: (f64, f64) = (90.0, 270.0);
/// …onto this play-rate band (ALS `{MinPlayRate, MaxPlayRate}`).
pub const ROTATE_PLAY_RATE: (f64, f64) = (1.15, 3.0);
/// How many spine joints the aim yaw is divided across (ALS divides by 4 and
/// applies the result to each of four spine/pelvis bones, so the chain sums to
/// the whole angle).
pub const SPINE_CHAIN_JOINTS: f64 = 4.0;

/// **ALS's turn-in-place delay**: how long the conditions must hold before the
/// turn fires, given how far the aim has drifted.
///
/// A 45° offset fires immediately and a 180° one waits
/// [`TURN_MAX_DELAY_S`] — so a player who glances sideways does not spin the
/// character, and one who has genuinely turned around does not stand there
/// forever.
pub fn turn_in_place_delay_s(angle_deg: f64) -> f64 {
    mapped_range(
        angle_deg.abs(),
        (TURN_CHECK_MIN_DEG, 180.0),
        (0.0, TURN_MAX_DELAY_S),
    )
}

/// Whether a turn in place is **allowed to be considered** this step: the aim
/// has drifted far enough and the camera has settled.
///
/// Both halves, because either alone is wrong — a large offset with the camera
/// still moving is a player mid-look, and a settled camera at 10° is nothing to
/// turn for.
pub fn turn_in_place_ready(aim_delta_deg: f64, aim_yaw_rate_dps: f64) -> bool {
    aim_delta_deg.abs() > TURN_CHECK_MIN_DEG && aim_yaw_rate_dps.abs() < TURN_AIM_RATE_LIMIT_DPS
}

/// **ALS's rotate-in-place**: the two gates and the play-rate scale.
///
/// Returns `(rotate_left, rotate_right, play_rate)`. Unlike a turn in place this
/// has no delay and no target — it is a looping rotate animation whose *rate*
/// tracks how fast the camera is moving, which is why the third value is the
/// only interesting one.
pub fn rotate_in_place(aim_delta_deg: f64, aim_yaw_rate_dps: f64) -> (bool, bool, f64) {
    let d = if aim_delta_deg.is_finite() {
        aim_delta_deg
    } else {
        0.0
    };
    let left = d < -ROTATE_IN_PLACE_DEG;
    let right = d > ROTATE_IN_PLACE_DEG;
    let rate = if left || right {
        mapped_range(
            aim_yaw_rate_dps.abs(),
            ROTATE_RATE_BAND_DPS,
            ROTATE_PLAY_RATE,
        )
    } else {
        1.0
    };
    (left, right, rate)
}

/// **The vertical aim-offset sweep** `[0, 1]` (ALS `AimSweepTime`): `0` looking
/// straight up, `1` straight down, mapped off the aim pitch.
pub fn aim_sweep(pitch_deg: f64) -> f64 {
    mapped_range(pitch_deg, (-90.0, 90.0), (1.0, 0.0))
}

/// **The per-joint spine yaw** for an aim offset, degrees.
///
/// ALS divides the aim/body yaw delta by four and applies the quotient to each of
/// four bones so the chain sums to the whole angle — a hard-coded chain length,
/// kept as [`SPINE_CHAIN_JOINTS`] so the constant is a name rather than a `4.0`
/// in an expression.
///
/// `mask` is the clip's `Mask_AimOffset` channel read **inverted** (ALS's
/// `EnableAimOffset = lerp(1, 0, curve)`), so a state that wants no aim offset
/// authors a `1` and gets none.
pub fn spine_yaw_deg(aim_delta_deg: f64, mask: f64) -> f64 {
    let weight = (1.0 - mask.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    angle_delta_deg(aim_delta_deg, 0.0) / SPINE_CHAIN_JOINTS * weight
}

/// How long a mantle of `height_m` takes, seconds, before the play rate scales
/// it.
///
/// A low ledge is a quick pull-up and a high one is a climb. The band is ours —
/// ALS reads the montage's own length and rescales it — and it is here rather
/// than on [`CharacterMovement`] because that component is on the scene wire and
/// this wave has no schema budget. The **shape** is the ported thing: duration
/// rises with height and the play rate from the height remap divides it, so two
/// different ledges share one animation.
pub const MANTLE_LOW_TIME_S: f64 = 0.55;
/// See [`MANTLE_LOW_TIME_S`].
pub const MANTLE_HIGH_TIME_S: f64 = 1.10;
/// See [`MANTLE_LOW_TIME_S`].
pub fn mantle_duration_s(height_m: f64, min_h: f64, max_h: f64, play_rate: f64) -> f64 {
    let base = mapped_range(
        height_m,
        (min_h, max_h.max(min_h + 1e-6)),
        (MANTLE_LOW_TIME_S, MANTLE_HIGH_TIME_S),
    );
    let rate = if play_rate.is_finite() && play_rate > 0.0 {
        play_rate
    } else {
        1.0
    };
    (base / rate).max(1.0 / 240.0)
}

// ── the mode table ──────────────────────────────────────────────────────────

/// Whether `to` may be entered from `from` at all — **the single mode-transition
/// table** the port map's closing warning asks for.
///
/// ALS has five movement states and handles them with a hand-written `if/else`
/// chain repeated in four places. At twelve modes that shape does not survive:
/// the same question gets four answers and they drift. One table, read by the
/// only function that changes a mode, is what makes "every catalogue mode is
/// reachable" a property a gate can check rather than a claim.
///
/// This answers *legality*, not *possibility*: a legal transition can still be
/// refused for want of clearance or speed. Both halves are values.
pub fn transition_is_legal(from: MovementMode, to: MovementMode) -> bool {
    use MovementMode::*;
    if from == to {
        return true;
    }
    // A deferred mode is legal to LEAVE (nothing can be in one) and illegal to
    // enter — but the refusal for entering it is `ModeNotYetImplemented`, which
    // is more informative than `IllegalTransition`, so the table lets it through
    // and `request_mode` classifies it first.
    if to.is_deferred() {
        return true;
    }
    match (from, to) {
        // **Ragdoll wins over everything**, for the same reason water does and
        // one step further: it is a fact about the body, not a choice the
        // controller made, and a character that has just been thrown off a cliff
        // mid-slide must not be told that a slide cannot become a ragdoll. It is
        // above the water rows deliberately — a swimmer can ragdoll too.
        (_, Ragdoll) => true,
        // …and leaving one has exactly two destinations, which is §13's row
        // verbatim: on the ground it gets up, in the air it resumes falling.
        (Ragdoll, Grounded | FallFree | FallControlled) => true,
        (Ragdoll, _) => false,
        // Water wins over everything a body could otherwise be doing: the latch
        // is a fact about where the character IS, not about what it chose.
        (_, SwimSurface) | (_, SwimUnder) => true,
        (SwimSurface | SwimUnder, Grounded | FallFree | FallControlled) => true,
        (SwimSurface | SwimUnder, _) => false,
        // **A mantle** is entered from the ground (jump at a wall) or from the
        // air (ALS's "falling catch"), and always ends standing — the warp puts
        // the feet on the ledge, so there is nothing else for it to end as.
        (Grounded | Crouch | FallFree | FallControlled, Mantle) => true,
        (Mantle, Grounded | FallFree | FallControlled) => true,
        (Mantle, _) => false,
        // Grounded family.
        (Grounded, Crouch | Prone | Slide | Roll | Dive | FallFree | FallControlled) => true,
        (Crouch, Grounded | Prone | Slide | Roll | FallFree | FallControlled) => true,
        (Prone, Crouch | Grounded | FallFree | FallControlled) => true,
        // A slide ends in a crouch, a stand, a roll, or off a ledge.
        (Slide, Crouch | Grounded | Roll | FallFree | FallControlled) => true,
        (Roll, Crouch | Grounded | FallFree | FallControlled) => true,
        // A dive is airborne and lands via the classifier.
        (Dive, Prone | Roll | Grounded | Crouch | FallControlled) => true,
        (FallFree | FallControlled, Grounded | Crouch | Prone | Roll | Dive) => true,
        (FallFree, FallControlled) | (FallControlled, FallFree) => true,
        // Everything else is off the table.
        _ => false,
    }
}

/// The verdict of a mode request: what the mode is now, and why if it did not
/// change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeVerdict {
    /// The mode after the request.
    pub mode: MovementMode,
    /// Why the request did not take, or [`MovementRefusal::None`].
    pub refusal: MovementRefusal,
}

/// Ask to change mode. **A refusal is a value**, so this never fails: it answers
/// with the mode that is now in force and the reason it is not the requested one.
///
/// `clearance` is the caller's answer to "does the taller capsule fit here" — it
/// needs a physics world, so it is passed in rather than asked for, which is
/// what keeps this whole module free of a simulation.
pub fn request_mode(
    from: MovementMode,
    to: MovementMode,
    clearance: bool,
    condition_met: bool,
) -> ModeVerdict {
    if from == to {
        return ModeVerdict {
            mode: from,
            refusal: MovementRefusal::None,
        };
    }
    if to.is_deferred() {
        return ModeVerdict {
            mode: from,
            refusal: MovementRefusal::ModeNotYetImplemented,
        };
    }
    if !transition_is_legal(from, to) {
        return ModeVerdict {
            mode: from,
            refusal: MovementRefusal::IllegalTransition,
        };
    }
    if !condition_met {
        return ModeVerdict {
            mode: from,
            refusal: MovementRefusal::ConditionNotMet,
        };
    }
    if !clearance {
        return ModeVerdict {
            mode: from,
            refusal: MovementRefusal::NoOverheadClearance,
        };
    }
    ModeVerdict {
        mode: to,
        refusal: MovementRefusal::None,
    }
}

// ── intent ──────────────────────────────────────────────────────────────────

/// One character's movement **intent** for a step: what the controller is asking
/// for, in units the movement step understands, with no knowledge of which
/// device produced it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MovementIntent {
    /// Planar motion in the aim frame: `x` right, `y` forward, each `[-1, 1]`.
    pub move_input: Vec2d,
    /// Yaw rate, degrees per second.
    pub look_yaw_dps: f64,
    /// Pitch rate, degrees per second.
    pub look_pitch_dps: f64,
    /// Vertical intent while swimming or flying, `[-1, 1]`.
    pub vertical: f64,
    /// Held: sprint.
    pub sprint: bool,
    /// Held: walk.
    pub walk: bool,
    /// Held: aim.
    pub aim: bool,
    /// Edge: jump.
    pub jump: bool,
    /// Edge: crouch.
    pub crouch: bool,
    /// Edge: prone.
    pub prone: bool,
    /// Edge: roll.
    pub roll: bool,
    /// Edge: dive.
    pub dive: bool,
}

impl MovementIntent {
    /// Build an intent from a host's resolved input, by the names in
    /// [`actions`].
    ///
    /// # Why two closures and not an `InputState`
    ///
    /// `inf-ecs` does not depend on `inf-input`, and it should not have to: an
    /// intent is not input. A Blueprint, an AI or a replay can produce one. What
    /// must not be duplicated is the *name set and their meanings*, and that is
    /// what lives here — both hosts call this with one line each, so a control
    /// cannot mean one thing in the editor's Simulate and another in the shipped
    /// player.
    pub fn from_actions(
        axis: impl Fn(&str) -> f32,
        held: impl Fn(&str) -> bool,
        pressed: impl Fn(&str) -> bool,
    ) -> Self {
        let ax = |n: &str| {
            let v = axis(n) as f64;
            if v.is_finite() {
                v
            } else {
                0.0
            }
        };
        Self {
            move_input: Vec2d::new(
                ax(actions::MOVE_X).clamp(-1.0, 1.0),
                ax(actions::MOVE_Y).clamp(-1.0, 1.0),
            ),
            look_yaw_dps: ax(actions::LOOK_X),
            look_pitch_dps: ax(actions::LOOK_Y),
            vertical: ax(actions::MOVE_UP).clamp(-1.0, 1.0),
            sprint: held(actions::SPRINT),
            walk: held(actions::WALK),
            aim: held(actions::AIM),
            jump: pressed(actions::JUMP),
            crouch: pressed(actions::CROUCH),
            prone: pressed(actions::PRONE),
            roll: pressed(actions::ROLL),
            dive: pressed(actions::DIVE),
        }
    }

    /// The magnitude of the planar input, `[0, 1]` — ALS's
    /// `MovementInputAmount`, which the sprint gate reads.
    pub fn magnitude(&self) -> f64 {
        let m =
            (self.move_input.x * self.move_input.x + self.move_input.y * self.move_input.y).sqrt();
        m.min(1.0)
    }
}

/// Write `intent` onto every **player-controlled** [`CharacterMovement`] in the
/// world, in `Guid` order.
///
/// One door, called by both hosts directly before the fixed movement step, for
/// the same reason [`crate::pose::step_pose_evaluation`] is one function: the
/// editor's Simulate and the shipped player must not be able to interpret a
/// control differently.
pub fn apply_intent(world: &mut EcsWorld, intent: &MovementIntent) {
    let mut q = world.world_mut().query::<&mut CharacterMovement>();
    // Collected and sorted is unnecessary here — the write is per entity and
    // order-free — but the ITERATION is a bevy archetype walk and must never be
    // allowed to influence a result. It does not: nothing here reads a sibling.
    let world_mut = world.world_mut();
    for mut cm in q.iter_mut(world_mut) {
        if !cm.player_controlled {
            continue;
        }
        let rt = &mut cm.runtime;
        rt.intent_move = intent.move_input;
        rt.intent_look_yaw_dps = intent.look_yaw_dps;
        rt.intent_look_pitch_dps = intent.look_pitch_dps;
        rt.intent_vertical = intent.vertical;
        rt.want_sprint = intent.sprint;
        rt.want_walk = intent.walk;
        rt.want_aim = intent.aim;
        // Edges are ORed rather than assigned: a host may apply an intent more
        // than once between fixed steps (a frame that runs two of them), and an
        // edge that arrived in the first must not be erased by the second.
        // `step_character_movement` clears them once it has consumed them.
        rt.press_jump |= intent.jump;
        rt.press_crouch |= intent.crouch;
        rt.press_prone |= intent.prone;
        rt.press_roll |= intent.roll;
        rt.press_dive |= intent.dive;
    }
}

/// The [`Guid`]s of every character the movement step will visit, sorted.
///
/// One door, beside [`apply_intent`], for the same one-door reason — and the
/// walk is the point:
///
/// ## Cost
///
/// `O(characters)`, not `O(entities)`: `try_query_filtered` restricts the walk
/// to archetypes that actually contain a [`CharacterMovement`], so a streamed
/// scene with thousands of entities and no characters pays nothing per fixed
/// step. The distinction is measured, not stylistic — the movement step's
/// original full-world `iter_entities` walk pushed the phase16 streamed-scene
/// budget from under 4 ms to 4.22 ms on a scene with **zero** characters (the
/// CI red on the P29.3 push). `None` from `try_query_filtered` — the component
/// never inserted in this world — is the `O(1)` answer for every characterless
/// level. Same law as [`crate::sky::sky_authority`].
///
/// The sort is the determinism half: a bevy archetype walk's order is an
/// implementation detail and must never reach a result, so the step consumes
/// these in `Guid` order, as it always has.
pub fn movement_targets(world: &EcsWorld) -> Vec<Uuid> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<&Guid, (With<CharacterMovement>, With<Transform>)>()
    else {
        return Vec::new();
    };
    let mut targets: Vec<Uuid> = q.iter(w).map(|g| g.0).collect();
    targets.sort_unstable();
    targets
}

// ── overlay registry ────────────────────────────────────────────────────────

/// The **open, string-keyed overlay registry** (Ruling 4).
///
/// `OverlayState` is deliberately *not* a wire enum: ALS ships thirteen overlay
/// states and a studio must be able to add a fourteenth — Rifle, Torch, Box,
/// Injured — without an engine schema bump. So the scene carries the name and
/// this interns it to a `u32` for the hot loop.
///
/// Interning is by **first-seen order**, which is why this is a session-local id
/// and never reaches a file: two runs that meet the names in a different order
/// would assign different numbers, and a number that means something different
/// per run must not be written down. The scene's copy is always the string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OverlayRegistry {
    names: Vec<String>,
}

/// The id of the default overlay — the empty name, always `0`.
pub const OVERLAY_DEFAULT: u32 = 0;

impl OverlayRegistry {
    /// A registry containing only the default (empty) overlay.
    pub fn new() -> Self {
        Self {
            names: vec![String::new()],
        }
    }

    /// The id for `name`, interning it if this is the first time it has been
    /// seen.
    pub fn intern(&mut self, name: &str) -> u32 {
        if self.names.is_empty() {
            self.names.push(String::new());
        }
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return i as u32;
        }
        self.names.push(name.to_string());
        (self.names.len() - 1) as u32
    }

    /// The name behind an id, or `None` for an id this registry never issued.
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(String::as_str)
    }

    /// How many distinct overlays have been interned, including the default.
    pub fn len(&self) -> usize {
        self.names.len().max(1)
    }

    /// Whether only the default overlay is present.
    pub fn is_empty(&self) -> bool {
        self.len() <= 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm() -> CharacterMovement {
        CharacterMovement::default()
    }

    #[test]
    fn mapped_speed_anchors_on_the_three_gait_speeds() {
        let c = cm();
        let (w, r, s) = (c.walk_speed_mps, c.run_speed_mps, c.sprint_speed_mps);
        for (speed, want) in [
            (0.0, 0.0),
            (w / 2.0, 0.5),
            (w, 1.0),
            ((w + r) / 2.0, 1.5),
            (r, 2.0),
            (s, 3.0),
            (s * 10.0, 3.0),
        ] {
            let got = mapped_speed(speed, w, r, s);
            assert!(
                (got - want).abs() < 1e-9,
                "{speed} m/s should map to {want}, got {got}"
            );
        }
        // The whole point: retuning the speeds does NOT move the curve axis.
        let retuned = mapped_speed(r * 2.0, w * 2.0, r * 2.0, s * 2.0);
        assert!(
            (retuned - 2.0).abs() < 1e-9,
            "doubling every speed leaves the run anchor at 2, got {retuned}"
        );
        // A NaN speed is 0, not a NaN that poisons every curve downstream.
        assert_eq!(mapped_speed(f64::NAN, w, r, s), 0.0);
        // An author who inverted two speeds gets a monotone ladder, not a
        // character that reads 0 for every speed.
        assert!(mapped_speed(3.0, 1.0, 0.5, 6.0) > 1.0);
    }

    #[test]
    fn a_speed_curve_is_linear_between_its_anchors_and_clamps_outside() {
        let c = crate::components::SpeedCurve::new(1.0, 2.0, 4.0, 8.0);
        for (s, want) in [
            (-5.0, 1.0),
            (0.0, 1.0),
            (0.5, 1.5),
            (1.0, 2.0),
            (1.25, 2.5),
            (2.0, 4.0),
            (2.5, 6.0),
            (3.0, 8.0),
            (99.0, 8.0),
        ] {
            assert!(
                (c.sample(s) - want).abs() < 1e-12,
                "sample({s}) = {}, want {want}",
                c.sample(s)
            );
        }
        assert_eq!(c.sample(f64::NAN), 1.0, "a NaN reads the stopped anchor");
    }

    #[test]
    fn the_gait_ladder_lags_the_request_while_slowing() {
        let c = cm();
        // Sprinting at full tilt with full input, velocity-direction mode.
        let (allowed, actual) = resolve_gait(
            &c,
            MovementMode::Grounded,
            Gait::Sprint,
            c.sprint_speed_mps,
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(allowed, Gait::Sprint);
        assert_eq!(actual, Gait::Sprint);

        // Release sprint: the REQUEST is a run immediately, but the body is
        // still moving at sprint speed, so the actual gait is Run — the ladder
        // reads the measured speed, and a sprint that is decelerating does not
        // become a walk animation the instant the key comes up.
        let (allowed, actual) = resolve_gait(
            &c,
            MovementMode::Grounded,
            Gait::Run,
            c.sprint_speed_mps,
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(allowed, Gait::Run);
        assert_eq!(actual, Gait::Run);

        // Below the walk anchor plus the band, the actual gait is a walk.
        let (_, actual) = resolve_gait(
            &c,
            MovementMode::Grounded,
            Gait::Run,
            c.walk_speed_mps,
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(
            actual,
            Gait::Walk,
            "the hysteresis band is above the anchor"
        );
        let (_, actual) = resolve_gait(
            &c,
            MovementMode::Grounded,
            Gait::Run,
            c.walk_speed_mps + GAIT_HYSTERESIS_MPS,
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(actual, Gait::Run, "and exactly at the band it steps up");
    }

    #[test]
    fn the_sprint_gate_refuses_for_each_of_its_four_reasons() {
        let mut c = cm();
        let full = c.sprint_speed_mps;

        // Baseline: it IS allowed.
        assert_eq!(
            resolve_gait(
                &c,
                MovementMode::Grounded,
                Gait::Sprint,
                full,
                1.0,
                0.0,
                0.0
            )
            .0,
            Gait::Sprint
        );
        // 1. Not standing.
        assert_eq!(
            resolve_gait(&c, MovementMode::Crouch, Gait::Sprint, full, 1.0, 0.0, 0.0).0,
            Gait::Run,
            "a crouched character does not sprint"
        );
        // 2. Aiming.
        c.rotation_mode = RotationMode::Aiming;
        assert_eq!(
            resolve_gait(
                &c,
                MovementMode::Grounded,
                Gait::Sprint,
                full,
                1.0,
                0.0,
                0.0
            )
            .0,
            Gait::Run,
            "aiming forbids a sprint"
        );
        c.rotation_mode = RotationMode::VelocityDirection;
        // 3. Input magnitude below the threshold (ALS 0.9).
        assert_eq!(
            resolve_gait(
                &c,
                MovementMode::Grounded,
                Gait::Sprint,
                full,
                0.5,
                0.0,
                0.0
            )
            .0,
            Gait::Run,
            "half a stick is not a sprint"
        );
        // 4. Input direction too far off the aim, in LookingDirection only.
        c.rotation_mode = RotationMode::LookingDirection;
        assert_eq!(
            resolve_gait(
                &c,
                MovementMode::Grounded,
                Gait::Sprint,
                full,
                1.0,
                90.0,
                0.0
            )
            .0,
            Gait::Run,
            "sprinting sideways is refused in LookingDirection"
        );
        assert_eq!(
            resolve_gait(
                &c,
                MovementMode::Grounded,
                Gait::Sprint,
                full,
                1.0,
                45.0,
                0.0
            )
            .0,
            Gait::Sprint,
            "45 degrees is inside the 50-degree cone"
        );
        // ...and NOT in VelocityDirection, where the body turns to face the input.
        c.rotation_mode = RotationMode::VelocityDirection;
        assert_eq!(
            resolve_gait(
                &c,
                MovementMode::Grounded,
                Gait::Sprint,
                full,
                1.0,
                90.0,
                0.0
            )
            .0,
            Gait::Sprint,
            "the direction cone is a LookingDirection rule only"
        );
    }

    #[test]
    fn the_quadrant_buffer_stops_the_boundary_chattering() {
        // Exactly on the 70-degree boundary, the answer depends on where it came
        // from — which is the whole point of the buffer.
        assert_eq!(
            quadrant(70.0, MovementDirection::Forward),
            MovementDirection::Forward,
            "leaving forward needs 75 degrees"
        );
        assert_eq!(
            quadrant(70.0, MovementDirection::Right),
            MovementDirection::Right,
            "entering forward needs 65 degrees"
        );
        assert_eq!(
            quadrant(60.0, MovementDirection::Right),
            MovementDirection::Forward
        );
        assert_eq!(
            quadrant(80.0, MovementDirection::Forward),
            MovementDirection::Right
        );
        // Left is the mirror.
        assert_eq!(
            quadrant(-80.0, MovementDirection::Forward),
            MovementDirection::Left
        );
        // Backward past 110 (115 entering, 105 leaving).
        assert_eq!(
            quadrant(120.0, MovementDirection::Right),
            MovementDirection::Backward
        );
        assert_eq!(
            quadrant(112.0, MovementDirection::Right),
            MovementDirection::Right,
            "entering backward needs 115 degrees"
        );
        assert_eq!(
            quadrant(112.0, MovementDirection::Backward),
            MovementDirection::Backward,
            "and leaving it needs to drop under 105"
        );
        // The failure the buffer exists to prevent, reproduced: an input sitting
        // ON the raw 70-degree boundary and wobbling by less than the buffer.
        // With hysteresis it never leaves the band it started in; a bufferless
        // comparison against 70 flips on every crossing, which at 60 Hz is the
        // character's legs changing clip several times a second.
        let mut dir = MovementDirection::Forward;
        let mut flips = 0;
        let mut bufferless_flips = 0;
        let mut bufferless = MovementDirection::Forward;
        for i in 0..200 {
            let a = 70.0 + 3.0 * inf_math::psin64(i as f64 * 0.7);
            let next = quadrant(a, dir);
            if next != dir {
                flips += 1;
            }
            dir = next;
            // The control: the same decision with no buffer at all.
            let raw = if a.abs() <= QUADRANT_FRONT_DEG {
                MovementDirection::Forward
            } else {
                MovementDirection::Right
            };
            if raw != bufferless {
                bufferless_flips += 1;
            }
            bufferless = raw;
        }
        assert_eq!(
            flips, 0,
            "a wobble smaller than the buffer must not change the answer at all"
        );
        assert!(
            bufferless_flips > 20,
            "the control must actually chatter, or this arm proves nothing: {bufferless_flips}"
        );
    }

    #[test]
    fn the_landing_classifier_reads_impact_speed_and_nothing_else() {
        let c = cm();
        assert_eq!(classify_landing(&c, 0.0, false), LandingKind::Soft);
        assert_eq!(classify_landing(&c, 6.99, true), LandingKind::Soft);
        // The middle band splits on INPUT, exactly as ALS's does.
        assert_eq!(classify_landing(&c, 7.0, false), LandingKind::Hard);
        assert_eq!(classify_landing(&c, 7.0, true), LandingKind::Roll);
        assert_eq!(classify_landing(&c, 9.99, true), LandingKind::Roll);
        // Past the upper threshold it is a ragdoll verdict regardless of input.
        assert_eq!(classify_landing(&c, 10.0, true), LandingKind::Ragdoll);
        assert_eq!(classify_landing(&c, 40.0, false), LandingKind::Ragdoll);
        // Sign and NaN are not the classifier's problem.
        assert_eq!(classify_landing(&c, -12.0, false), LandingKind::Ragdoll);
        assert_eq!(classify_landing(&c, f64::NAN, false), LandingKind::Soft);
    }

    #[test]
    fn landing_friction_is_an_override_with_an_expiry() {
        // The fixture must have LANDED. The first version did not, and read the
        // override anyway because `time_since_land_s` starts at zero — the defect
        // `the_landing_override_waits_for_a_landing_to_have_happened` now pins
        // from the other side (P29.3 audit, A5).
        let mut c = cm();
        c.runtime.landing = LandingKind::Hard;
        assert_eq!(
            landing_friction_scale(&c, 0.0, true),
            c.brake_friction_input
        );
        assert_eq!(
            landing_friction_scale(&c, 0.0, false),
            c.brake_friction_idle
        );
        assert_eq!(
            landing_friction_scale(&c, 0.49, false),
            c.brake_friction_idle
        );
        assert_eq!(
            landing_friction_scale(&c, 0.5, false),
            1.0,
            "the override expires at half a second"
        );
        assert_eq!(landing_friction_scale(&c, 1e9, true), 1.0);
    }

    #[test]
    fn the_integrator_accelerates_to_the_target_and_stops_when_asked_to() {
        let dt = 1.0 / 60.0;
        let mut v = Vec2d::ZERO;
        let wish = Vec2d::new(0.0, 1.0);
        for _ in 0..600 {
            v = integrate_planar_velocity(v, wish, 3.75, 8.0, 20.0, 6.0, 1.0, dt);
        }
        let speed = (v.x * v.x + v.y * v.y).sqrt();
        assert!(
            (speed - 3.75).abs() < 1e-6,
            "ten seconds of full input reaches the target speed: {speed}"
        );
        // Release the input and it stops in a bounded time rather than drifting.
        let mut steps = 0;
        while (v.x * v.x + v.y * v.y).sqrt() > 1e-3 && steps < 600 {
            v = integrate_planar_velocity(v, Vec2d::ZERO, 3.75, 8.0, 20.0, 6.0, 1.0, dt);
            steps += 1;
        }
        assert!(steps < 60, "it took {steps} steps to stop from a run");
        // The braking friction override really is an override: doubling it stops
        // sooner, which is the whole of ALS's landing rule.
        let mut fast = Vec2d::new(0.0, 3.75);
        let mut fast_steps = 0;
        while (fast.x * fast.x + fast.y * fast.y).sqrt() > 1e-3 && fast_steps < 600 {
            fast = integrate_planar_velocity(fast, Vec2d::ZERO, 3.75, 8.0, 20.0, 6.0, 3.0, dt);
            fast_steps += 1;
        }
        assert!(
            fast_steps < steps,
            "a 3x friction scale must stop sooner: {fast_steps} vs {steps}"
        );
        // A zero or non-finite dt is inert rather than a NaN in the velocity.
        assert_eq!(
            integrate_planar_velocity(Vec2d::new(1.0, 2.0), wish, 3.0, 8.0, 20.0, 6.0, 1.0, 0.0),
            Vec2d::new(1.0, 2.0)
        );
    }

    #[test]
    fn smoothing_bounds_the_peak_rate_and_then_settles() {
        // A 179-degree goal change must not snap: the constant-rate stage caps
        // how far the target can travel in one step.
        let dt = 1.0 / 60.0;
        let (t1, a1) = smooth_rotation(0.0, 0.0, 179.0, 360.0, 20.0, dt);
        assert!(
            (t1 - 6.0).abs() < 1e-9,
            "360 deg/s over 1/60 s is 6 degrees, got {t1}"
        );
        assert!(a1.abs() < 6.0, "and the body lags the target: {a1}");

        // Given time, both arrive.
        let (mut t, mut a) = (0.0, 0.0);
        for _ in 0..600 {
            let (nt, na) = smooth_rotation(t, a, 179.0, 360.0, 20.0, dt);
            t = nt;
            a = na;
        }
        assert!((t - 179.0).abs() < 1e-6, "target = {t}");
        assert!((a - 179.0).abs() < 1e-3, "actor = {a}");

        // The wrap is handled: chasing 350 from 10 goes the short way (down).
        let (t, _) = smooth_rotation(10.0, 10.0, 350.0, 360.0, 20.0, dt);
        assert!(
            t < 10.0,
            "the short way from 10 to 350 is downward, got {t}"
        );
    }

    #[test]
    fn the_rotation_rate_multiplier_reads_the_aim_yaw_rate() {
        let c = cm();
        let at_rest = grounded_rotation_rate(&c, 2.0, 0.0);
        let at_speed = grounded_rotation_rate(&c, 2.0, 300.0);
        assert!(
            (at_speed / at_rest - 3.0).abs() < 1e-9,
            "a 300 deg/s camera turns the body 3x faster: {at_rest} -> {at_speed}"
        );
        assert!(
            (grounded_rotation_rate(&c, 2.0, 1000.0) / at_rest - 3.0).abs() < 1e-9,
            "and the multiplier is clamped"
        );
    }

    #[test]
    fn the_mode_table_answers_one_question_in_one_place() {
        use MovementMode::*;
        assert!(transition_is_legal(Grounded, Crouch));
        assert!(transition_is_legal(Crouch, Slide));
        assert!(transition_is_legal(Slide, Grounded));
        assert!(transition_is_legal(Dive, Prone));
        assert!(transition_is_legal(FallFree, Grounded));
        // Water wins, from anything, and does not let you slide.
        assert!(transition_is_legal(Slide, SwimSurface));
        assert!(!transition_is_legal(SwimSurface, Slide));
        assert!(transition_is_legal(SwimSurface, Grounded));
        // Nonsense is off the table.
        assert!(!transition_is_legal(Prone, Slide));
        assert!(!transition_is_legal(Prone, Roll));
        assert!(
            transition_is_legal(Grounded, Grounded),
            "staying put is legal"
        );
    }

    #[test]
    fn a_deferred_mode_refuses_by_name_rather_than_pretending() {
        use MovementMode::*;
        for deferred in [Driving, Flying, Reserved14, Reserved17] {
            let v = request_mode(Grounded, deferred, true, true);
            assert_eq!(
                v.mode, Grounded,
                "{deferred:?} has no mechanics in this wave and must not be entered"
            );
            assert_eq!(v.refusal, MovementRefusal::ModeNotYetImplemented);
        }
        // **And the two this wave took no longer refuse** -- the other half of
        // the same claim, without which "Mantle is deferred" could be deleted
        // from the list above and nothing would notice.
        for taken in [Mantle, Ragdoll] {
            let v = request_mode(Grounded, taken, true, true);
            assert_eq!(v.mode, taken, "{taken:?} has its mechanics in P29.4");
            assert_eq!(v.refusal, MovementRefusal::None);
        }
        // A ragdoll may be entered from ANYTHING (it is a fact about the body,
        // not a choice) and leaves to exactly two places.
        for from in [Grounded, Crouch, Slide, Dive, FallFree, SwimUnder, Mantle] {
            assert!(transition_is_legal(from, Ragdoll), "{from:?} -> Ragdoll");
        }
        assert!(transition_is_legal(Ragdoll, Grounded));
        assert!(transition_is_legal(Ragdoll, FallFree));
        assert!(!transition_is_legal(Ragdoll, Slide));
        // A mantle is entered from the ground or from the air, and always ends
        // standing.
        assert!(transition_is_legal(Grounded, Mantle));
        assert!(transition_is_legal(FallControlled, Mantle));
        assert!(!transition_is_legal(Prone, Mantle));
        assert!(transition_is_legal(Mantle, Grounded));
        assert!(!transition_is_legal(Mantle, Slide));
        // The three other refusals are distinguishable, which is the point of
        // having four of them.
        assert_eq!(
            request_mode(Prone, Slide, true, true).refusal,
            MovementRefusal::IllegalTransition
        );
        assert_eq!(
            request_mode(Grounded, Slide, true, false).refusal,
            MovementRefusal::ConditionNotMet
        );
        assert_eq!(
            request_mode(Crouch, Grounded, false, true).refusal,
            MovementRefusal::NoOverheadClearance
        );
        // And a legal, met, clear request goes through — otherwise the four
        // refusals above would be satisfied by a function that refuses always.
        let ok = request_mode(Crouch, Grounded, true, true);
        assert_eq!(ok.mode, Grounded);
        assert_eq!(ok.refusal, MovementRefusal::None);
    }

    #[test]
    fn the_slide_friction_curve_runs_between_flat_and_the_slope_limit() {
        let c = cm();
        assert!((slide_friction(&c, 0.0) - c.slide_friction_flat).abs() < 1e-12);
        assert!((slide_friction(&c, 45.0) - c.slide_friction_slope).abs() < 1e-12);
        let mid = slide_friction(&c, 22.5);
        assert!(
            mid < c.slide_friction_flat && mid > c.slide_friction_slope,
            "a 22.5 degree slope is between the two: {mid}"
        );
    }

    #[test]
    fn the_derived_anim_inputs_span_their_ranges() {
        assert_eq!(gait_scalar(0.0), 0.0);
        assert_eq!(gait_scalar(1.0), 0.0);
        assert_eq!(gait_scalar(2.0), 1.0);
        assert_eq!(gait_scalar(3.0), 2.0);
        assert_eq!(stride_blend(0.0), 0.0);
        assert_eq!(stride_blend(2.0), 1.0);
        assert_eq!(stride_blend(3.0), 1.0);
        assert_eq!(walk_run_blend(0.5), 0.0);
        assert_eq!(walk_run_blend(1.5), 0.5);
        assert_eq!(walk_run_blend(2.5), 1.0);
    }

    /// The tolerances here are 1e-6, not 1e-12, and that is the portable-math
    /// law being paid for rather than a sloppy assertion: `psin64`/`pcos64` are
    /// a degree-11 Taylor polynomial accurate to about 1e-7, chosen because
    /// `f64::sin` is a libm call and libm is not bit-portable. A rotation that
    /// reaches `state_bytes` must be the same number on every machine, and being
    /// the same to 1e-7 everywhere beats being exact on one.
    #[test]
    fn the_frame_rotation_is_its_own_inverse_and_uses_portable_math() {
        for yaw in [0.0, 37.0, 90.0, -145.0, 180.0, 359.0] {
            let v = Vec2d::new(0.3, -0.9);
            let there = rotate_into_frame(v, yaw);
            let back = rotate_from_frame(there, yaw);
            assert!(
                (back.x - v.x).abs() < 1e-6 && (back.y - v.y).abs() < 1e-6,
                "yaw {yaw}: {v:?} -> {there:?} -> {back:?}"
            );
        }
        // The convention: +Z is forward at yaw 0, +X is right, and yawing by 90
        // puts forward on +X.
        let fwd = rotate_from_frame(Vec2d::new(0.0, 1.0), 90.0);
        assert!(
            (fwd.x - 1.0).abs() < 1e-6 && fwd.y.abs() < 1e-6,
            "yaw 90 faces +X: {fwd:?}"
        );
        assert!((planar_yaw_deg(Vec2d::new(1.0, 0.0)) - 90.0).abs() < 1e-6);
        assert!(planar_yaw_deg(Vec2d::new(0.0, 1.0)).abs() < 1e-6);
        assert_eq!(planar_yaw_deg(Vec2d::ZERO), 0.0);
    }

    #[test]
    fn relative_acceleration_normalizes_against_the_right_maximum() {
        // Accelerating forward at exactly the max reads +1 on the forward axis.
        let a = relative_acceleration(Vec2d::new(0.0, 8.0), 0.0, 8.0, 20.0, false);
        assert!((a.y - 1.0).abs() < 1e-6 && a.x.abs() < 1e-6, "{a:?}");
        // The same magnitude while DECELERATING normalizes against braking, so
        // it reads much smaller — which is the distinction ALS draws and the
        // reason a lean does not slam over on every stop.
        let b = relative_acceleration(Vec2d::new(0.0, 8.0), 0.0, 8.0, 20.0, true);
        assert!((b.y - 0.4).abs() < 1e-6, "{b:?}");
        // Yawed by 90, forward acceleration reads on the body's own forward.
        let c = relative_acceleration(Vec2d::new(8.0, 0.0), 90.0, 8.0, 20.0, false);
        assert!((c.y - 1.0).abs() < 1e-6 && c.x.abs() < 1e-6, "{c:?}");
        // A zero maximum answers zero rather than an infinity.
        assert_eq!(
            relative_acceleration(Vec2d::new(0.0, 8.0), 0.0, 0.0, 0.0, false),
            Vec2d::ZERO
        );
    }

    #[test]
    fn the_overlay_registry_is_open_and_stable_within_a_session() {
        let mut reg = OverlayRegistry::new();
        assert_eq!(reg.intern(""), OVERLAY_DEFAULT);
        let rifle = reg.intern("rifle");
        let torch = reg.intern("torch");
        assert_ne!(rifle, torch);
        assert_eq!(reg.intern("rifle"), rifle, "interning is idempotent");
        assert_eq!(reg.name(rifle), Some("rifle"));
        assert_eq!(reg.name(999), None);
        assert_eq!(reg.len(), 3);
        // The point of an OPEN id: a name the engine has never heard of works.
        let invented = reg.intern("carrying_a_very_specific_crate");
        assert_eq!(reg.name(invented), Some("carrying_a_very_specific_crate"));
    }

    /// **Turning costs something** (P29.3 audit, A4). The integrator's second
    /// documented rule — "bleed the component that is NOT along the wish
    /// direction, so a turn feels like a turn instead of a drift" — had no arm,
    /// and a mutation that set the bleed factor to 1 (keep everything) killed
    /// nothing in the tree.
    ///
    /// Measured as the thing the rule is for: a character at full speed on +Y
    /// asked to run +X arrives on its new heading sooner with friction than
    /// without, and the frictionless control is what makes that a measurement.
    #[test]
    fn the_integrator_bleeds_the_velocity_that_is_not_going_where_you_asked() {
        let dt = 1.0 / 60.0;
        let run = |friction: f64| {
            let mut v = Vec2d::new(0.0, 3.75);
            let mut steps = 0;
            // "On heading" = the old +Y component is under a tenth of the speed.
            while steps < 600 && v.y.abs() > 0.1 * (v.x * v.x + v.y * v.y).sqrt() {
                v = integrate_planar_velocity(
                    v,
                    Vec2d::new(1.0, 0.0),
                    3.75,
                    8.0,
                    20.0,
                    friction,
                    1.0,
                    dt,
                );
                steps += 1;
            }
            (steps, v)
        };
        let (with_friction, _) = run(6.0);
        let (frictionless, tail) = run(0.0);
        assert!(
            with_friction < frictionless,
            "the sideways bleed is not doing anything: {with_friction} steps with \
             friction vs {frictionless} without"
        );
        // …and the control really is the slow one rather than one that never
        // arrives for some other reason.
        assert!(
            frictionless < 600,
            "the frictionless control never came round at all: {tail:?}"
        );
    }

    /// **A character that has never landed is not recovering from a landing**
    /// (P29.3 audit, A5).
    ///
    /// `time_since_land_s` starts at zero like every other runtime field, so the
    /// window test alone said "landed this instant" for a character that had
    /// never left the ground — half a second of ALS's 3x plant friction, earned
    /// by nothing. The latch is `landing`.
    #[test]
    fn the_landing_override_waits_for_a_landing_to_have_happened() {
        let mut c = cm();
        assert_eq!(c.runtime.landing, LandingKind::None);
        assert_eq!(
            landing_friction_scale(&c, 0.0, false),
            1.0,
            "nothing has landed, so there is no override to apply"
        );
        assert_eq!(landing_friction_scale(&c, 0.0, true), 1.0);
        // The control: once something HAS landed the override is exactly the one
        // ALS specifies, so the guard above did not simply disable it.
        c.runtime.landing = LandingKind::Hard;
        assert_eq!(
            landing_friction_scale(&c, 0.0, false),
            c.brake_friction_idle
        );
        assert_eq!(
            landing_friction_scale(&c, 0.0, true),
            c.brake_friction_input
        );
    }

    /// **Aiming moves slower** (P29.3 audit, A4) — the second half of ALS's
    /// two-level settings dispatch, which had no arm at all: a
    /// `rotation_speed_scale` that answered 1.0 for every mode killed nothing.
    #[test]
    fn the_rotation_mode_scales_the_target_speed() {
        let mut c = cm();
        let at = |c: &CharacterMovement| {
            settings_for(c, MovementMode::Grounded, Gait::Run, 2.0, 0.0).target_speed_mps
        };
        let free = at(&c);
        assert!((free - c.run_speed_mps).abs() < 1e-12);
        c.rotation_mode = RotationMode::Aiming;
        let aiming = at(&c);
        assert!(
            (aiming - c.run_speed_mps * c.aiming_speed_scale).abs() < 1e-12,
            "aiming must scale the target speed: {aiming} vs {free}"
        );
        assert!(aiming < free, "and it must be SLOWER: {aiming} vs {free}");
        c.rotation_mode = RotationMode::LookingDirection;
        assert!((at(&c) - c.run_speed_mps * c.looking_speed_scale).abs() < 1e-12);
    }

    #[test]
    fn an_intent_reads_the_names_the_bindings_use() {
        let intent = MovementIntent::from_actions(
            |n| match n {
                actions::MOVE_X => 0.5,
                actions::MOVE_Y => -1.0,
                actions::LOOK_X => 42.0,
                actions::MOVE_UP => 2.0, // out of range on purpose
                _ => 0.0,
            },
            |n| n == actions::SPRINT,
            |n| n == actions::JUMP,
        );
        assert_eq!(intent.move_input, Vec2d::new(0.5, -1.0));
        assert_eq!(intent.look_yaw_dps, 42.0);
        assert_eq!(intent.vertical, 1.0, "clamped");
        assert!(intent.sprint && intent.jump);
        assert!(!intent.walk && !intent.crouch);
        assert!((intent.magnitude() - (0.25f64 + 1.0).sqrt().min(1.0)).abs() < 1e-12);

        // A NaN axis is zero, not a NaN that reaches the velocity integrator.
        let bad = MovementIntent::from_actions(|_| f32::NAN, |_| false, |_| false);
        assert_eq!(bad.move_input, Vec2d::ZERO);
        assert_eq!(bad.look_yaw_dps, 0.0);
    }

    // -- P29.4: turn/rotate in place, aim offsets, mantle timing --------------

    /// The delay is **angle-dependent**, which is the whole idea: a glance does
    /// not spin the character and a real turn-around does not stall.
    #[test]
    fn the_turn_in_place_delay_rises_with_the_angle() {
        assert_eq!(turn_in_place_delay_s(45.0), 0.0, "45 degrees fires at once");
        assert_eq!(turn_in_place_delay_s(180.0), TURN_MAX_DELAY_S);
        assert!((turn_in_place_delay_s(112.5) - 0.375).abs() < 1e-12);
        // Sign-free, and clamped rather than extrapolated.
        assert_eq!(turn_in_place_delay_s(-180.0), TURN_MAX_DELAY_S);
        assert_eq!(turn_in_place_delay_s(10.0), 0.0);
        assert_eq!(turn_in_place_delay_s(999.0), TURN_MAX_DELAY_S);

        // **Both gates**, and each alone is not enough: a big offset with the
        // camera still moving is a player mid-look, and a settled camera at 10
        // degrees is nothing to turn for.
        assert!(turn_in_place_ready(60.0, 10.0));
        assert!(
            !turn_in_place_ready(60.0, 80.0),
            "the camera has not settled"
        );
        assert!(!turn_in_place_ready(10.0, 0.0), "nothing to turn for");
        assert!(
            !turn_in_place_ready(45.0, 0.0),
            "the threshold is exclusive"
        );
        // The 180 threshold is a named constant rather than a literal in a
        // branch, because it picks WHICH animation plays.
        assert_eq!(TURN_180_DEG, 130.0);
    }

    #[test]
    fn rotate_in_place_gates_on_fifty_degrees_and_scales_its_play_rate() {
        let (l, r, rate) = rotate_in_place(0.0, 200.0);
        assert!(!l && !r && rate == 1.0, "inside the band nothing rotates");
        let (l, r, _) = rotate_in_place(60.0, 0.0);
        assert!(!l && r);
        let (l, r, _) = rotate_in_place(-60.0, 0.0);
        assert!(l && !r);
        // The rate tracks the camera: slow camera, slow turn.
        let (_, _, slow) = rotate_in_place(60.0, 90.0);
        let (_, _, fast) = rotate_in_place(60.0, 270.0);
        assert!((slow - 1.15).abs() < 1e-12, "{slow}");
        assert!((fast - 3.0).abs() < 1e-12, "{fast}");
        let (_, _, mid) = rotate_in_place(60.0, 180.0);
        assert!(mid > slow && mid < fast, "{mid}");
        // A NaN offset rotates nothing rather than both ways.
        let (l, r, _) = rotate_in_place(f64::NAN, 100.0);
        assert!(!l && !r);
    }

    #[test]
    fn the_aim_offset_sweeps_and_the_mask_suppresses_it() {
        // Looking straight up is 0, straight down is 1, level is the middle.
        assert_eq!(aim_sweep(-90.0), 1.0);
        assert_eq!(aim_sweep(90.0), 0.0);
        assert!((aim_sweep(0.0) - 0.5).abs() < 1e-12);

        // The yaw is divided across the spine chain, so four joints sum to the
        // whole angle.
        assert!((spine_yaw_deg(80.0, 0.0) - 20.0).abs() < 1e-12);
        assert!((spine_yaw_deg(80.0, 0.0) * SPINE_CHAIN_JOINTS - 80.0).abs() < 1e-12);
        // `Mask_AimOffset` is read INVERTED: a state that authors 1 gets none.
        assert_eq!(spine_yaw_deg(80.0, 1.0), 0.0);
        assert!((spine_yaw_deg(80.0, 0.5) - 10.0).abs() < 1e-12);
        // A wrapped angle takes the short way round rather than 350 degrees.
        assert!((spine_yaw_deg(-350.0, 0.0) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn a_mantle_takes_longer_the_higher_it_is_and_the_play_rate_divides_it() {
        let low = mantle_duration_s(0.5, 0.5, 2.5, 1.0);
        let high = mantle_duration_s(2.5, 0.5, 2.5, 1.0);
        assert!((low - MANTLE_LOW_TIME_S).abs() < 1e-12, "{low}");
        assert!((high - MANTLE_HIGH_TIME_S).abs() < 1e-12, "{high}");
        assert!(high > low, "a higher ledge takes longer: {low} vs {high}");
        // The play rate divides it -- two ledges, one animation.
        let fast = mantle_duration_s(0.5, 0.5, 2.5, 2.0);
        assert!((fast - MANTLE_LOW_TIME_S / 2.0).abs() < 1e-12, "{fast}");
        // Degenerate inputs answer a duration rather than an infinity or a zero
        // that would divide downstream.
        assert!(mantle_duration_s(1.0, 1.0, 1.0, 1.0).is_finite());
        assert!(mantle_duration_s(1.0, 0.5, 2.5, 0.0) > 0.0);
        assert!(mantle_duration_s(f64::NAN, 0.5, 2.5, 1.0) > 0.0);
    }
}
