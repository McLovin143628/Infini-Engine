//! **Fixed-wing flight** (wave VEH3g) -- lift as a function of angle of attack,
//! a stall, induced drag, control surfaces and a propeller or a jet.
//!
//! # The refusal this module re-rules
//!
//! VEH2c refused a fixed wing in so many words: *"what it needs is lift as a
//! function of angle of attack, a stall, control surfaces and a runway"*. The
//! research doc's class table names a flight dynamics model for aeroplanes and
//! the VEH3f roster authored five of them, dormant; this module is the model,
//! and the island's airstrip is the runway.
//!
//! # A fixed wing is a WHEELED vehicle, and that is the recogniser
//!
//! The part recogniser's collider-shape space is exhausted (sphere = wheel, box =
//! thruster, capsule = rotor), and **wheels win**: a chassis with wheels never
//! looks for parts. So an aeroplane is a [`RaycastVehicle`] -- its landing gear
//! IS the wheel rig -- whose class carries a positive
//! [`wing_area_m2`](crate::vehicle::VehicleTuning::wing_area_m2), one of the v28
//! fields VEH3a spent the arc's single schema window on. The wing area is the
//! discriminator AND a physical quantity, so there is nothing to keep in sync:
//! set it to zero and the Dodo is a car that cannot leave the ground.
//!
//! On the ground the gear rolls, brakes and steers (the nose wheel is the steered
//! axle) and the engine does NOT drive the wheels: an aeroplane is pushed by its
//! propeller, which is [`thrust_n`]. In the air the wheels find no contact and
//! the same solve simply produces no tyre force.
//!
//! # What the pilot commands
//!
//! | control | surface | what it does |
//! |---|---|---|
//! | `vertical` (the rise/sink axis) | elevator | trims the wing to an ANGLE OF ATTACK |
//! | `steer` | ailerons (+ the nose wheel on the ground) | commands a ROLL RATE |
//! | `throttle` | the power lever | spools the engine |
//! | `brake` | the wheel brakes | as a car's |
//!
//! **The elevator commands an angle of attack**, and that is the aerodynamics of a
//! statically stable aeroplane rather than a flight computer: the tailplane's
//! restoring moment grows with α, the elevator shifts the α at which it balances,
//! so a stick POSITION is an α and the moment that gets the airframe there is a
//! stiffness times the α error. Both the stiffness and the damping scale with the
//! dynamic pressure `q` ([`authority`]), which is why the elevator does nothing on
//! a slow taxi and rotates the aeroplane at the rotation speed. The ailerons
//! command a roll RATE for the same reason with the roll's own physics: a wing
//! has no roll stiffness, so a deflection is a rate. The yaw is the weathervane --
//! the fin turns the nose into the relative wind, and that is what coordinates a
//! banked turn. These are the RotorVehicle precedent (a commanded state held by a
//! moment scaled by the airframe's own inertia), re-used with an aerodynamic
//! authority rather than a governed rotor's.
//!
//! # The stall
//!
//! `CL = a (α + i)` up to [`stall_deg`](crate::vehicle::VehicleTuning::stall_deg)
//! -- `a = 2π AR / (AR + 2)`, the finite-wing lift slope -- and past it the lift
//! COLLAPSES: it falls linearly to [`POST_STALL_CL_FRAC`] of its peak over
//! [`STALL_BREAK_DEG`], then fades to zero at 90° like a flat plate, while a
//! flat plate's drag ([`POST_STALL_DRAG`]) arrives. Full back stick asks for
//! [`STALL_OVERSHOOT_DEG`] past the stall, so a pilot CAN stall the aeroplane.
//! The lift's collapse is what drops the nose; the tail's break
//! ([`STALL_PITCH_BREAK`]) is what keeps the stall SHALLOW -- it bounds how far
//! past the stall the wing goes and how long it stays there (VEH3g audit: 32.2
//! deg deepest and 5.4 s past it, against 50.3 deg and 7.5 s without). Every
//! angle is from [`inf_math::portable::patan2_64`]; no std trig is on this path.
//!
//! # Take-off flaps (VEH3g audit)
//!
//! Every roster aeroplane rolls with its TAKE-OFF FLAP set: [`TAKEOFF_FLAP_CL`]
//! more lift at every angle below a stall [`FLAP_STALL_SHIFT_DEG`] earlier, and
//! [`FLAP_CD0`] of extra drag. The flap lever is the flight model's, not a key
//! -- GTA's aeroplanes have none -- and it follows the schedule a crew flies:
//! SET on the ground below half the clean stall speed (a parked or taxiing
//! aeroplane), RETRACTED once airborne past [`FLAP_RETRACT_VS`] times the clean
//! stall speed, travelling [`FLAP_TRAVEL_S`] end to end ([`flap_step`]). An
//! aeroplane that starts in the air (the stall arm's) never has them. Measured
//! on the take-off table: the Luxor and the Jetliner reached 35 ft at 1 827 m
//! and 1 727 m CLEAN, past the island's 1 700 m runway.
//!
//! # What this model does not have, by name
//!
//! No landing flaps, no ground effect, no
//! propeller torque or P-factor, no spin (a stalled wing drops level), no
//! compressibility (the Luxor's 120 m/s is Mach 0.35), and the air density is
//! the sea-level [`AIR_DENSITY_KG_M3`] at every altitude -- the island's highest
//! ground is under 600 m, where the real density is 6 % lower.

use glam::DVec3;

use crate::vehicle::{
    torque_pair, ChassisState, VehicleControls, VehicleTuning, WheelForce, AIR_DENSITY_KG_M3,
};

/// The wing's rigging angle plus its camber's zero-lift offset, degrees: the
/// angle of attack the WING meets the air at when the FUSELAGE is level.
///
/// It is what lets an aeroplane cruise with its cabin floor level and what makes
/// a take-off roll produce some lift before the nose comes up. Three degrees is a
/// light aeroplane's (two of incidence, one of camber offset).
pub const WING_INCIDENCE_DEG: f64 = 3.0;

/// Oswald's span efficiency: how far a real wing's induced drag is above an
/// elliptical wing's. `k = 1 / (π AR e)`.
pub const OSWALD_EFFICIENCY: f64 = 0.8;

/// The degrees past the stall over which the lift collapses to
/// [`POST_STALL_CL_FRAC`] of its peak -- the SHAPE of the stall.
pub const STALL_BREAK_DEG: f64 = 4.0;

/// What is left of the peak lift coefficient once the wing has broken away --
/// a flat plate's, before its own fade to zero at 90°.
pub const POST_STALL_CL_FRAC: f64 = 0.55;

/// The flat-plate drag coefficient a stalled wing adds, times `sin² α`.
pub const POST_STALL_DRAG: f64 = 1.1;

/// How far past the stall FULL back stick asks the wing to fly, degrees.
///
/// Positive on purpose: an aeroplane whose elevator could not reach the stall
/// could not be stalled, and a stall is one of the things this wave was told to
/// measure. Four degrees is a light aeroplane with its stick on the stop.
pub const STALL_OVERSHOOT_DEG: f64 = 4.0;

/// How much of the pull-up authority full FORWARD stick has, as a fraction.
pub const PUSH_AUTHORITY: f64 = 0.6;

/// The pitch stiffness at the stall's own dynamic pressure, (rad/s²) per radian
/// of angle-of-attack error. Scales with [`authority`].
pub const PITCH_STIFFNESS: f64 = 2.0;

/// The pitch mode's damping ratio -- the tail's damping, sized against the
/// stiffness so the short period settles rather than rings.
pub const PITCH_DAMPING_RATIO: f64 = 0.8;

/// Extra nose-down stiffness per radian past the stall -- the centre of pressure
/// moving aft as the flow separates.
///
/// **It does not drop the nose -- the collapse does** (VEH3g audit). Zeroed, the
/// nose still falls (34.0 to -33.0 deg) and the stall arm stayed green: the
/// wave's second vacuous item. What it does is BOUND the stall: with it the
/// Dodo's angle of attack peaks at **32.2 deg** and the wing is stalled for
/// **5.4 s** (89 m lost); without it **50.3 deg** and **7.5 s** (155 m lost) --
/// a deep stall. The stall arm asserts the bound.
pub const STALL_PITCH_BREAK: f64 = 1.5;

/// **The take-off flap's lift increment** (VEH3g audit): the lift coefficient
/// the flap adds at every angle of attack below its stall. Half a unit is a
/// single-slotted flap at a take-off setting.
pub const TAKEOFF_FLAP_CL: f64 = 0.5;

/// How much EARLIER a flapped wing stalls, degrees of angle of attack.
pub const FLAP_STALL_SHIFT_DEG: f64 = 1.5;

/// The flap's parasite drag coefficient at full take-off setting (times `q S`).
pub const FLAP_CD0: f64 = 0.02;

/// Retract above this multiple of the CLEAN stall speed, once airborne.
pub const FLAP_RETRACT_VS: f64 = 1.3;

/// Set on the ground below this fraction of the clean stall speed.
pub const FLAP_SET_BELOW_VS: f64 = 0.5;

/// Seconds for the flap to travel end to end.
pub const FLAP_TRAVEL_S: f64 = 6.0;

/// **The flap lever's schedule** (VEH3g audit): `set` latches ON while the gear
/// has a contact below [`FLAP_SET_BELOW_VS`] of the clean stall speed and OFF
/// once no wheel touches above [`FLAP_RETRACT_VS`] of it; `flap` travels toward
/// the lever at `1 / FLAP_TRAVEL_S` a second. Pure and portable.
pub fn flap_step(
    flap: &mut f64,
    set: &mut bool,
    grounded: bool,
    airspeed: f64,
    vs_clean: f64,
    dt: f64,
) {
    if grounded && airspeed < FLAP_SET_BELOW_VS * vs_clean {
        *set = true;
    } else if !grounded && airspeed > FLAP_RETRACT_VS * vs_clean {
        *set = false;
    }
    let want = if *set { 1.0 } else { 0.0 };
    let step = (dt / FLAP_TRAVEL_S).max(0.0);
    *flap = if *flap < want {
        (*flap + step).min(want)
    } else {
        (*flap - step).max(want)
    };
}

/// The roll rate full aileron commands, rad/s (80 deg/s -- a light aeroplane).
pub const ROLL_RATE_MAX_RAD_S: f64 = 1.4;

/// How fast the roll rate follows the aileron, 1/s.
pub const ROLL_RATE_GAIN: f64 = 4.0;

/// The roll rate a bank asks for with the stick centred, rad/s per radian of
/// bank: the DIHEDRAL effect, which is why a light aeroplane left alone rolls
/// itself level.
pub const WING_LEVELLER: f64 = 0.6;

/// The weathervane stiffness at the stall's dynamic pressure, (rad/s²) per
/// radian of sideslip -- the fin.
pub const YAW_STIFFNESS: f64 = 1.5;

/// **The rudder's authority IN THE AIR**, degrees of sideslip full rudder
/// balances (VEH3g audit). Without it full aileron banked the Dodo 19 deg at
/// 40 m/s and turned it 0.3 deg in three seconds; with it, 5.7 deg the same
/// way the nose wheel steers (`veh3g_gate::d_turns_the_aeroplane_right_on_the_
/// ground_and_in_the_air`). Off on the gear -- see `fixed_wing_forces`.
pub const RUDDER_SLIP_DEG: f64 = 20.0;

/// The yaw mode's damping ratio.
pub const YAW_DAMPING_RATIO: f64 = 0.8;

/// The fuselage and fin's side force per radian of sideslip, as a multiple of
/// `q S` -- what turns a sideslip into a turn.
pub const SIDE_FORCE_PER_RAD: f64 = 0.4;

/// The ceiling on [`authority`]: a control surface at five times the stall's
/// dynamic pressure is as effective as the model lets it be, so a jet at 120 m/s
/// does not snap-roll on a keyboard tap.
pub const AUTHORITY_CEILING: f64 = 5.0;

/// The floor under the pitch and yaw DAMPING's authority, so an aeroplane at a
/// crawl does not ring on its own gear.
pub const DAMPING_AUTHORITY_FLOOR: f64 = 0.25;

/// A propeller engine's spool time constant, seconds: a piston engine answers
/// the lever almost at once.
pub const PROP_SPOOL_S: f64 = 0.4;

/// A jet's spool time constant, seconds -- the lag between the lever and the
/// thrust that every jet pilot flies around, and the jet VOICE's input.
pub const JET_SPOOL_S: f64 = 3.0;

/// How much of its static thrust a jet loses by its own top speed (ram drag).
pub const JET_RAM_DROP: f64 = 0.25;

/// The forward share of the airflow (`u / |air|`) under which the control
/// surfaces' restoring stiffness fades -- 0.2 is 78 degrees of flow (VEH3g
/// audit). See `fixed_wing_forces`.
pub const REVERSE_FLOW_FADE: f64 = 0.2;

/// Below this airspeed, m/s, the model applies no aerodynamic force at all: the
/// angle of attack of a stationary aeroplane is undefined, not zero.
pub const MIN_AIRSPEED_MPS: f64 = 0.5;

/// A propeller's rpm at idle and at full power (wave VEH3g) -- what its voice's
/// blade-pass is pitched by, a light aeroplane's 700 to 2 700.
pub const PROP_IDLE_RPM: f64 = 700.0;
/// See [`PROP_IDLE_RPM`].
pub const PROP_MAX_RPM: f64 = 2_700.0;
/// How many blades the voiced propeller has.
pub const PROP_BLADES: f64 = 2.0;

/// A propeller's rpm at a spool `[0, 1]`.
pub fn prop_rpm(spool: f64) -> f64 {
    PROP_IDLE_RPM + (PROP_MAX_RPM - PROP_IDLE_RPM) * spool.clamp(0.0, 1.0)
}

/// `engine_voice_kind` for a turbine -- the one kind whose thrust is a JET's.
pub const TURBINE_VOICE_KIND: f64 = 2.0;

/// **A wing** -- the three v28 numbers a fixed-wing class carries, read once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wing {
    /// Reference area, m².
    pub area_m2: f64,
    /// Aspect ratio, span² / area.
    pub aspect: f64,
    /// The wing angle of attack at the stall, radians.
    pub stall_rad: f64,
}

impl Wing {
    /// The wing a tuning describes, or `None` for anything that is not a fixed
    /// wing -- THE recogniser (`wing_area_m2 > 0`).
    pub fn of(t: &VehicleTuning) -> Option<Self> {
        if !(t.wing_area_m2.is_finite() && t.wing_area_m2 > 0.0) {
            return None;
        }
        let aspect = if t.wing_aspect_ratio.is_finite() && t.wing_aspect_ratio > 0.0 {
            t.wing_aspect_ratio
        } else {
            // A wing with no aspect ratio authored is a light aeroplane's.
            7.0
        };
        let stall = if t.stall_deg.is_finite() && t.stall_deg > 1.0 {
            t.stall_deg
        } else {
            15.0
        };
        Some(Self {
            area_m2: t.wing_area_m2,
            aspect,
            stall_rad: stall.to_radians(),
        })
    }

    /// The finite wing's lift slope, per radian: `2π AR / (AR + 2)`.
    pub fn lift_slope(&self) -> f64 {
        std::f64::consts::TAU * self.aspect / (self.aspect + 2.0)
    }

    /// The peak lift coefficient -- the slope times the stall angle.
    pub fn cl_max(&self) -> f64 {
        self.lift_slope() * self.stall_rad
    }

    /// The induced-drag factor `k = 1 / (π AR e)`.
    pub fn induced_k(&self) -> f64 {
        1.0 / (std::f64::consts::PI * self.aspect * OSWALD_EFFICIENCY)
    }

    /// **The lift coefficient at a WING angle of attack** (radians), and whether
    /// the wing is stalled.
    ///
    /// Linear to the stall; then the collapse: a linear fall to
    /// [`POST_STALL_CL_FRAC`] of the peak over [`STALL_BREAK_DEG`], and a fade to
    /// zero at 90° (a flat plate edge-on to nothing). Odd in α, so a wing flown
    /// inverted lifts the other way.
    pub fn cl(&self, alpha: f64) -> (f64, bool) {
        let a = alpha.abs();
        let sign = if alpha < 0.0 { -1.0 } else { 1.0 };
        if a <= self.stall_rad {
            return (self.lift_slope() * alpha, false);
        }
        let quarter = std::f64::consts::FRAC_PI_2;
        let over = a - self.stall_rad;
        let brk = STALL_BREAK_DEG.to_radians();
        let frac = POST_STALL_CL_FRAC + (1.0 - POST_STALL_CL_FRAC) * (1.0 - over / brk).max(0.0);
        let fade = ((quarter - a) / (quarter - self.stall_rad)).clamp(0.0, 1.0);
        (sign * self.cl_max() * frac * fade, true)
    }

    /// **The lift coefficient with the flap at `flap`** `[0, 1]` (VEH3g audit):
    /// [`TAKEOFF_FLAP_CL`]` x flap` added below a stall [`FLAP_STALL_SHIFT_DEG`]
    /// `x flap` earlier, and the same collapse past it from the flapped peak.
    /// `flap` 0 is [`Self::cl`] exactly.
    pub fn cl_flapped(&self, alpha: f64, flap: f64) -> (f64, bool) {
        let f = flap.clamp(0.0, 1.0);
        if f <= 0.0 || alpha < 0.0 {
            return self.cl(alpha);
        }
        let stall = (self.stall_rad - FLAP_STALL_SHIFT_DEG.to_radians() * f).max(1e-3);
        let dcl = TAKEOFF_FLAP_CL * f;
        if alpha <= stall {
            return (self.lift_slope() * alpha + dcl, false);
        }
        let peak = self.lift_slope() * stall + dcl;
        let quarter = std::f64::consts::FRAC_PI_2;
        let over = alpha - stall;
        let brk = STALL_BREAK_DEG.to_radians();
        let frac = POST_STALL_CL_FRAC + (1.0 - POST_STALL_CL_FRAC) * (1.0 - over / brk).max(0.0);
        let fade = ((quarter - alpha) / (quarter - stall)).clamp(0.0, 1.0);
        (peak * frac * fade, true)
    }

    /// The peak lift coefficient with the flap at `flap`.
    pub fn cl_max_flapped(&self, flap: f64) -> f64 {
        let f = flap.clamp(0.0, 1.0);
        let stall = (self.stall_rad - FLAP_STALL_SHIFT_DEG.to_radians() * f).max(1e-3);
        self.lift_slope() * stall + TAKEOFF_FLAP_CL * f
    }

    /// The stall speed with the flap at `flap`, m/s.
    pub fn stall_speed_flapped_mps(&self, weight_n: f64, flap: f64) -> f64 {
        let q = (weight_n / (self.area_m2 * self.cl_max_flapped(flap)).max(1e-9)).max(1e-9);
        (2.0 * q / AIR_DENSITY_KG_M3).sqrt()
    }

    /// The dynamic pressure at which this wing carries `weight_n` at its peak
    /// lift -- the STALL's `q`, the reference every control authority is scaled
    /// against.
    pub fn stall_q(&self, weight_n: f64) -> f64 {
        (weight_n / (self.area_m2 * self.cl_max()).max(1e-9)).max(1e-9)
    }

    /// The stall speed at `weight_n`, m/s, in still sea-level air.
    pub fn stall_speed_mps(&self, weight_n: f64) -> f64 {
        (2.0 * self.stall_q(weight_n) / AIR_DENSITY_KG_M3).sqrt()
    }
}

/// **How effective the control surfaces are**, `[0, AUTHORITY_CEILING]`: the
/// dynamic pressure against the stall's.
///
/// A control surface's moment is `q S c Cm`, so its authority IS the dynamic
/// pressure; dividing by the stall's makes the number mean the same thing on a
/// Duster and on a Jetliner -- `1` is "flying at the stall".
pub fn authority(q: f64, stall_q: f64) -> f64 {
    (q / stall_q.max(1e-9)).clamp(0.0, AUTHORITY_CEILING)
}

/// **The thrust**, newtons, at a spool `[0, 1]` and a forward airspeed.
///
/// A propeller's falls to zero at the class's `max_speed_mps` -- the speed of
/// its own slipstream, the hull's convention -- and a jet's barely falls at all
/// ([`JET_RAM_DROP`]). `max_engine_force_n` is the STATIC thrust.
pub fn thrust_n(t: &VehicleTuning, spool: f64, airspeed_fwd: f64, engine_scale: f64) -> f64 {
    let peak = t.max_engine_force_n.max(0.0) * engine_scale.clamp(0.0, 1.0);
    let top = t.max_speed_mps.max(1.0);
    let v = (airspeed_fwd / top).clamp(0.0, 1.0);
    let fall = if is_jet(t) {
        1.0 - JET_RAM_DROP * v
    } else {
        1.0 - v
    };
    peak * spool.clamp(0.0, 1.0) * fall
}

/// Whether this class's thrust is a jet's -- `engine_voice_kind` 2, a turbine.
pub fn is_jet(t: &VehicleTuning) -> bool {
    (t.engine_voice_kind - TURBINE_VOICE_KIND).abs() < 0.5
}

/// **What the wing did this step** -- the flight model's published state, the
/// source of every instrument and trace column this wave adds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlightState {
    /// Airspeed, m/s -- the chassis velocity MINUS the wind, magnitude.
    pub airspeed_mps: f64,
    /// The WING's angle of attack, degrees (the fuselage's plus
    /// [`WING_INCIDENCE_DEG`]).
    pub alpha_deg: f64,
    /// Sideslip, degrees, positive with the airframe slipping to starboard.
    pub beta_deg: f64,
    /// The lift coefficient the wing produced.
    pub cl: f64,
    /// Whether the wing was past its stall angle.
    pub stalled: bool,
    /// Lift, newtons.
    pub lift_n: f64,
    /// Total drag, newtons (parasite + induced + post-stall).
    pub drag_n: f64,
    /// Thrust, newtons.
    pub thrust_n: f64,
    /// The engine's spool, `[0, 1]` -- the power lever's lagged answer.
    pub spool: f64,
    /// The control authority ([`authority`]).
    pub authority: f64,
    /// The elevator's commanded angle of attack, degrees (wing).
    pub alpha_cmd_deg: f64,
    /// The take-off flap's travel, `[0, 1]` (VEH3g audit).
    pub flap: f64,
}

/// **The aerodynamic step** for a fixed wing: append the lift, drag, side force,
/// thrust and the control moments to `out`, advance the spool, and answer what
/// the wing did.
///
/// Pure, like every class's solve: the chassis state (velocity, attitude, the
/// P17 wind, the body's own inertia), the controls and the tuning in, forces out.
/// The forces act at the chassis origin, which is rapier's centre of mass for a
/// box chassis; the moments are a pure couple ([`torque_pair`]).
#[allow(clippy::too_many_arguments)]
pub fn fixed_wing_forces(
    t: &VehicleTuning,
    wing: &Wing,
    controls: &VehicleControls,
    chassis: &ChassisState,
    spool: &mut f64,
    flap: f64,
    on_gear: bool,
    engine_scale: f64,
    dt: f64,
    out: &mut Vec<WheelForce>,
) -> FlightState {
    let (fwd, right, up) = chassis.basis();
    // **STARBOARD is `-right`.** The chassis basis calls `rotation * X` "right",
    // and with `+Z` forward and `+Y` up that axis is the PILOT's left (a
    // right-handed frame): the driver's own "right" is the negative yaw rate
    // `VehicleControls::steer` has documented since P29.7. Every lateral sign
    // below is written against the pilot's starboard, because a weathervane
    // with its sign against the wrong side is a yaw DIVERGENCE -- measured: the
    // first cut's sideslip grew from 0.03 to 86 degrees in fifteen seconds of a
    // straight climb and the Dodo tumbled.
    let stbd = -right;
    // ── the lever and the spool. Nobody at the controls is a closed throttle.
    let lever = if controls.occupied {
        controls.throttle.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let tau = if is_jet(t) { JET_SPOOL_S } else { PROP_SPOOL_S };
    *spool += (lever - *spool) * (dt / tau).clamp(0.0, 1.0);

    // ── the relative wind, in the body frame.
    let air = chassis.linvel - chassis.wind;
    let u = air.dot(fwd);
    let w = air.dot(up);
    let s = air.dot(stbd);
    let v = air.length();
    let flap = flap.clamp(0.0, 1.0);
    let mut state = FlightState {
        airspeed_mps: v,
        spool: *spool,
        flap,
        ..FlightState::default()
    };
    let thrust = thrust_n(t, *spool, u, engine_scale);
    state.thrust_n = thrust;
    if thrust != 0.0 {
        out.push(WheelForce {
            point: chassis.position,
            force: fwd * thrust,
        });
    }
    if v < MIN_AIRSPEED_MPS {
        return state;
    }
    let q = 0.5 * AIR_DENSITY_KG_M3 * v * v;
    let incidence = WING_INCIDENCE_DEG.to_radians();
    // The fuselage's angle of attack: positive with the air arriving from below.
    let alpha_body = inf_math::portable::patan2_64(-w, u);
    let alpha = alpha_body + incidence;
    let beta = inf_math::portable::patan2_64(s, (u * u + w * w).sqrt());
    let (cl, stalled) = wing.cl_flapped(alpha, flap);
    let weight = chassis.mass_kg.max(0.0) * 9.81;
    let q_stall = wing.stall_q(weight);
    let auth = authority(q, q_stall);
    // **A control surface needs the air to arrive from the NOSE** (VEH3g
    // audit). With the flow reversed -- a parked aeroplane with the wind up its
    // tail -- the fuselage angle of attack reads 180 degrees and the elevator's
    // attitude command tried to pitch the airframe half a turn to "fix" it:
    // measured on the shipped host, a Dodo standing tail to an 8 m/s wind crept
    // BACKWARDS on its gear (-0.63 m/s) before anyone touched it. The restoring
    // stiffnesses (not the damping) fade to nothing between 78 and 90 degrees
    // of flow, so every attitude this model is flown at keeps them whole.
    let flow = ((u / v) / REVERSE_FLOW_FADE).clamp(0.0, 1.0);
    let ctrl = auth * flow;

    // ── lift, perpendicular to the air in the plane of symmetry.
    let sym = fwd * u + up * w;
    let lift_dir = sym.cross(right).normalize_or_zero();
    let lift = q * wing.area_m2 * cl;
    // ── drag: parasite (the class's `drag_n_per_mps2`, which on an aeroplane IS
    //    `½ ρ CD0 S`), induced (`k CL²`) and, stalled, a flat plate's.
    let sin_a = if v > 0.0 {
        (-w / v).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let post = if stalled {
        POST_STALL_DRAG * sin_a * sin_a
    } else {
        0.0
    };
    let drag = t.drag_n_per_mps2.max(0.0) * v * v
        + q * wing.area_m2 * (wing.induced_k() * cl * cl + post + FLAP_CD0 * flap);
    let side = -stbd * (q * wing.area_m2 * SIDE_FORCE_PER_RAD * beta);
    let aero = lift_dir * lift - air / v * drag + side;
    if aero != DVec3::ZERO {
        out.push(WheelForce {
            point: chassis.position,
            force: aero,
        });
    }

    // ── the control moments, as angular accelerations times the body's OWN
    //    principal inertia (x = pitch about right, y = yaw about up, z = roll
    //    about forward -- a box chassis's principal axes are its own).
    let elevator = if controls.occupied {
        controls.vertical.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let aileron = if controls.occupied {
        controls.steer.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    // PITCH: the elevator commands a fuselage angle of attack.
    let span = (wing.stall_rad + STALL_OVERSHOOT_DEG.to_radians() - incidence).max(0.0);
    let alpha_cmd = if elevator >= 0.0 {
        elevator * span
    } else {
        elevator * span * PUSH_AUTHORITY
    };
    state.alpha_cmd_deg = (alpha_cmd + incidence).to_degrees();
    let k = PITCH_STIFFNESS * ctrl;
    let kd =
        2.0 * PITCH_DAMPING_RATIO * (PITCH_STIFFNESS * auth.max(DAMPING_AUTHORITY_FLOOR)).sqrt();
    // `+right` (the port axis) is nose-DOWN, so nose-up is about starboard.
    let nose_up_rate = chassis.angvel.dot(stbd);
    let over = (alpha - wing.stall_rad).max(0.0);
    let pitch_acc = k * (alpha_cmd - alpha_body) - k * STALL_PITCH_BREAK * over - kd * nose_up_rate;
    // ROLL: the ailerons command a roll rate (a right stick drops the starboard
    // wing); centred, the dihedral levels it. A rotation about `+fwd` raises
    // the port wing, i.e. drops the starboard one.
    let bank_sin = -stbd.y;
    let stbd_down_rate = chassis.angvel.dot(fwd);
    let want_rate = if aileron.abs() > 0.05 {
        aileron * ROLL_RATE_MAX_RAD_S
    } else {
        -WING_LEVELLER * bank_sin
    };
    // ON THE GEAR the wheels, not the ailerons, hold the wings level (VEH3g
    // audit): the roll servo is a rate command, and standing on its tyres it
    // rolled a steering Dodo onto its downwind gear -- measured on the shipped
    // input path in an 8 m/s crosswind, a pilot holding the centreline with
    // the stick stuck at 40 deg off and 6 m/s. The rudder and the nose wheel
    // steer it on the ground.
    let roll_acc = if on_gear {
        0.0
    } else {
        ROLL_RATE_GAIN * (want_rate - stbd_down_rate) * ctrl.min(1.0)
    };
    // YAW: the weathervane. A slip to starboard (beta > 0) swings the nose to
    // starboard, which is a NEGATIVE rotation about up.
    let yk = YAW_STIFFNESS * ctrl;
    let ykd = 2.0 * YAW_DAMPING_RATIO * (YAW_STIFFNESS * auth.max(DAMPING_AUTHORITY_FLOOR)).sqrt();
    let nose_stbd_rate = -chassis.angvel.dot(up);
    // THE RUDDER (VEH3g audit): the same stick as the ailerons, a yaw moment
    // worth [`RUDDER_SLIP_DEG`] of sideslip at full throw -- IN THE AIR. On the
    // gear the nose wheel steers: measured at 19.5 m/s on the strip, steer 0.3
    // turned the Dodo 3.23 deg in a second on the nose wheel alone and 2.07
    // with this couple added (the tyres fight a yaw couple about the centre of
    // mass), and in an 8 m/s crosswind a heading-holding pilot on the shipped
    // input path was 53 deg off with it and 13 without.
    let rudder = if on_gear { 0.0 } else { aileron };
    let yaw_acc = yk * (beta + RUDDER_SLIP_DEG.to_radians() * rudder) - ykd * nose_stbd_rate;
    let inertia = chassis.inertia;
    let torque =
        stbd * (pitch_acc * inertia.x) + fwd * (roll_acc * inertia.z) - up * (yaw_acc * inertia.y);
    if let Some(pair) = torque_pair(chassis.position, torque, 1.0) {
        out.extend_from_slice(&pair);
    }

    state.alpha_deg = alpha.to_degrees();
    state.beta_deg = beta.to_degrees();
    state.cl = cl;
    state.stalled = stalled;
    state.lift_n = lift;
    state.drag_n = drag;
    state.authority = auth;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dodo_tuning() -> VehicleTuning {
        VehicleTuning {
            wing_area_m2: 16.2,
            wing_aspect_ratio: 7.4,
            stall_deg: 16.0,
            ..VehicleTuning::default()
        }
    }

    #[test]
    fn a_wing_is_recognised_by_its_area_and_only_by_it() {
        assert!(Wing::of(&VehicleTuning::default()).is_none());
        assert!(Wing::of(&dodo_tuning()).is_some());
        let zero = VehicleTuning {
            wing_area_m2: 0.0,
            ..dodo_tuning()
        };
        assert!(Wing::of(&zero).is_none());
    }

    #[test]
    fn lift_is_linear_to_the_stall_and_collapses_past_it() {
        let wing = Wing::of(&dodo_tuning()).unwrap();
        let s = wing.stall_rad;
        let (at, stalled_at) = wing.cl(s);
        assert!(!stalled_at);
        assert!((at - wing.cl_max()).abs() < 1e-12);
        let (half, _) = wing.cl(s * 0.5);
        assert!((half - at * 0.5).abs() < 1e-12, "linear below the stall");
        let brk = s + STALL_BREAK_DEG.to_radians();
        let (past, stalled) = wing.cl(brk);
        assert!(stalled);
        let fade = (std::f64::consts::FRAC_PI_2 - brk) / (std::f64::consts::FRAC_PI_2 - s);
        assert!(
            (past / at - POST_STALL_CL_FRAC * fade).abs() < 1e-9,
            "the collapse: {past} of {at}"
        );
        let (edge, _) = wing.cl(std::f64::consts::FRAC_PI_2);
        assert!(edge.abs() < 1e-12, "a flat plate edge-on lifts nothing");
        let (neg, _) = wing.cl(-s * 0.5);
        assert!((neg + half).abs() < 1e-12, "odd in alpha");
    }

    /// The take-off flap (VEH3g audit): a flap of 0 IS the clean wing, a set
    /// flap lifts more at every angle below its (earlier) stall, and the lever
    /// is set on the gear when slow and up once flying fast -- never set in
    /// the air.
    #[test]
    fn the_takeoff_flap_lifts_more_and_follows_its_schedule() {
        let wing = Wing::of(&dodo_tuning()).unwrap();
        for a in [-0.2, 0.0, 0.1, 0.25, 0.3, 0.6] {
            assert_eq!(wing.cl_flapped(a, 0.0), wing.cl(a));
        }
        assert!(wing.cl_max_flapped(1.0) > wing.cl_max() + 0.3);
        let (clean, _) = wing.cl(0.1);
        let (flapped, st) = wing.cl_flapped(0.1, 1.0);
        assert!(!st && (flapped - clean - TAKEOFF_FLAP_CL).abs() < 1e-12);
        let w = 1100.0 * 9.81;
        assert!(wing.stall_speed_flapped_mps(w, 1.0) < wing.stall_speed_mps(w));
        let (mut flap, mut set) = (0.0, false);
        let vs = wing.stall_speed_mps(w);
        // In the air, slow: never set.
        flap_step(&mut flap, &mut set, false, 0.8 * vs, vs, 1.0);
        assert!(!set && flap == 0.0);
        // On the gear, parked: set, and it travels.
        for _ in 0..(FLAP_TRAVEL_S as usize + 1) {
            flap_step(&mut flap, &mut set, true, 0.0, vs, 1.0);
        }
        assert!(set && flap == 1.0);
        // Rolling fast on the gear: it stays set.
        flap_step(&mut flap, &mut set, true, 1.2 * vs, vs, 1.0);
        assert!(set);
        // Airborne past the retract speed: up.
        flap_step(&mut flap, &mut set, false, 1.4 * vs, vs, 1.0);
        assert!(!set && flap < 1.0);
    }

    #[test]
    fn a_propeller_fades_with_speed_and_a_jet_barely_does() {
        let mut t = dodo_tuning();
        t.max_engine_force_n = 3_000.0;
        t.max_speed_mps = 100.0;
        assert_eq!(thrust_n(&t, 1.0, 0.0, 1.0), 3_000.0);
        assert!((thrust_n(&t, 1.0, 50.0, 1.0) - 1_500.0).abs() < 1e-9);
        t.engine_voice_kind = TURBINE_VOICE_KIND;
        assert!((thrust_n(&t, 1.0, 100.0, 1.0) - 3_000.0 * (1.0 - JET_RAM_DROP)).abs() < 1e-9);
        assert_eq!(
            thrust_n(&t, 1.0, 0.0, 0.0),
            0.0,
            "a dead engine pushes nothing"
        );
    }
}
