//! **How a weapon FEELS** (wave WPN2b): the three recoil layers, the sway and
//! the aim-down-sights blend, as pure functions and one runtime component.
//!
//! # The three layers, and why they are three
//!
//! The user's research doc §2 states the rule this module implements: *"split
//! recoil into three distinct layers"* — a **viewmodel** offset that is visual
//! only, a **camera** pitch/yaw that is gameplay control, and a **spread**
//! variance that is where the bullet actually goes. One layer is not enough and
//! neither are two:
//!
//! * a weapon that only kicks the hands is a gun that climbs while the aim
//!   stands still — which is what this engine shipped from wave WPN1 to this
//!   one, derived off the fire clock (`recoil_fraction`, deleted here);
//! * a weapon that only moves the aim is a rifle that jumps in the world and
//!   not in the hands;
//! * and neither of them makes the thirtieth round of a magazine land anywhere
//!   different from the first, which is the whole of what a player calls
//!   "recoil" once they have stopped counting.
//!
//! # The camera layer is an AIM layer, by ruling
//!
//! `inf_physics::d3::gameplay::aim_hold_point`'s own doc carries the refusal
//! this module honours: a camera kick that does not move the aim makes the
//! reticle **lie**, and one that does is a camera to sim write, which
//! `d3::camera`'s Ruling 4 forbids outright. So the second layer is an impulse
//! on `CharacterMovement::runtime::aim_pitch_deg` / `aim_yaw_deg` **inside the
//! movement step's own look integrator** — sim state, folded into the trace
//! through the pose, identical on both hosts — and the camera follows it
//! because the camera has always followed the aim. Nothing in `camera.rs` or
//! `camera.toml` gains a per-shot input, and `wpn2b_gate` greps both for one.
//!
//! # "Critically damped", and what the doc's own numbers actually do
//!
//! The doc asks for a *"Critically Damped Spring-Damper System"* and then hands
//! over `k = 220, c = 18` (viewmodel) and `k = 180, c = 16` (camera). Those are
//! not critically damped: critical damping of a unit mass is `c = 2*sqrt(k)`,
//! which is [`VM_DAMPING`] **29.6648** and [`AIM_DAMPING`] **26.8328**. At the
//! doc's 18 the damping ratio is `18 / 29.6648 = 0.6068`, whose continuous
//! impulse response overshoots its own rest position by
//! `exp(-zeta*pi / sqrt(1 - zeta^2))` = 9.1 % — and which this module's own
//! arm **measures at 6.33 %** on the 60 Hz recurrence, still past the 5 %
//! ceiling this wave's gate names, against **0.0000 %** at the critical
//! damping. So what ships is the doc's **stiffness** with the
//! **critical** damping derived from it, the doc's numbers are recorded beside
//! them as [`DOC_VM_DAMPING`] / [`DOC_AIM_DAMPING`], and the gate's mutation is
//! to put the doc's number back — which reds the overshoot arm, in the world,
//! on the hand joint.
//!
//! # An impulse is a PEAK, so a profile number is a measurement
//!
//! The doc's `add_impulse` adds to **velocity**, which is right — that is what
//! makes a shot snap and settle rather than teleport and slide — but it leaves
//! every number in [`RecoilProfile`] one integration away from anything a gate
//! can measure. So the profile's numbers are **peak displacements** and
//! [`impulse_for_peak`] converts one into the velocity the spring needs to
//! reach it. The textbook conversion is `v0 = peak * sqrt(k) * e` (from
//! `x(t) = v0*t*exp(-w*t)`, whose maximum is `v0 / (w*e)`) and it is **42 %
//! wrong at 60 Hz** — measured, and the reason is in [`discrete_peak_gain`]'s
//! doc: `c*dt` is 0.45 at this damping, so half the impulse is gone before the
//! position has moved. What ships measures the gain **from the recurrence that
//! will actually run**, which makes the conversion exact at the sample points a
//! fixed step has. A profile that says a shot lifts the aim by half a degree is
//! then a claim about the world, and `wpn2b_gate` reads it off the movement
//! runtime.
//!
//! # Determinism
//!
//! Everything here is a pure function of sim state plus the **counter hash**
//! ([`crate::weapon::shot_uniforms`], splitmix64 over `(seed, shot)`): there is
//! no RNG state in a fixed step, the n-th shot of a weapon jitters the same way
//! in a replay, in a PIE preview and in a shipped build, and
//! [`feel_state_bytes`] folds the springs at the **TAIL** of
//! `RuntimeSim::state_bytes` — **empty when every feel in the world is at
//! rest**, which is what keeps every trace committed before this wave
//! byte-identical. The arithmetic is `+ - * /` and `sqrt` for the springs and
//! `psin64` for the sway; `inf-physics/tests/portable_character.rs` scans this
//! file for anything else.
//!
//! # What is NOT here
//!
//! Audio layers, casings, shotgun cones, attachments and weapon meshes are
//! WPN2c's and WPN2d's. The overlay POSES this wave finally drives live in
//! `crate::pose` beside the breath, because that is where an additive layer is
//! applied; what this module owns is the number they are driven by.

use bevy_ecs::prelude::Component;
use glam::DVec3;
use uuid::Uuid;

use crate::components::Guid;
use crate::weapon::{WeaponDef, WeaponState};
use crate::world::EcsWorld;

// ── the springs ─────────────────────────────────────────────────────────────

/// **The viewmodel spring's stiffness** — the doc's own `Spring3D::new(220.0,
/// 18.0)`, snappy half.
pub const VM_STIFFNESS: f64 = 220.0;

/// **The viewmodel spring's damping**, critical for [`VM_STIFFNESS`].
///
/// `2*sqrt(220) = 29.664793948382652`. See the module header for why this is
/// not the doc's 18.
pub const VM_DAMPING: f64 = 29.664_793_948_382_652;

/// **The aim spring's stiffness** — the doc's `Spring3D::new(180.0, 16.0)`.
pub const AIM_STIFFNESS: f64 = 180.0;

/// **The aim spring's damping**, critical for [`AIM_STIFFNESS`].
///
/// `2*sqrt(180) = 26.832815729997478`.
pub const AIM_DAMPING: f64 = 26.832_815_729_997_478;

/// **The doc's viewmodel damping**, recorded so the gate can put it back.
///
/// Underdamped: `zeta = 0.607`. Measured overshoot at the 60 Hz step: **6.33 %**,
/// against 0.0000 % at [`VM_DAMPING`].
pub const DOC_VM_DAMPING: f64 = 18.0;

/// **The doc's camera damping**, recorded for [`DOC_VM_DAMPING`]'s reason.
pub const DOC_AIM_DAMPING: f64 = 16.0;

/// Critical damping for a unit mass at stiffness `k` — `2*sqrt(k)`.
///
/// `sqrt` and not a table: IEEE-754 specifies the square root exactly, so this
/// is bit-portable and `inf_math::libm_ban` does not ban it (its own header
/// says so).
pub fn critical_damping(k: f64) -> f64 {
    if !k.is_finite() || k <= 0.0 {
        return 0.0;
    }
    2.0 * k.sqrt()
}

/// **How far a unit velocity impulse actually moves THIS spring**, at this
/// stiffness, this damping and this fixed step.
///
/// The continuous answer is `1 / (w*e)` and it is **42 % wrong at 60 Hz** —
/// measured, on the first run of `an_impulse_peaks_where_it_was_asked_to`: a
/// 0.05 peak came out at 0.0291. The reason is the semi-implicit step itself.
/// Its very first update is `v <- v0 * (1 - c*dt)`, and at the critical damping
/// this module ships `c*dt` is 0.45 — so nearly half the impulse is spent before
/// the position has moved at all, and the peak arrives 4.5 steps later at a
/// fraction of the closed form.
///
/// So the gain is **measured from the recurrence that will actually run**,
/// rather than derived from the differential equation that will not: start at
/// `(0, 1)`, step until the position turns over, answer the largest position
/// seen. That makes [`impulse_for_peak`] exact **at the sample points a fixed
/// step has**, which is the only place a gate can look.
///
/// `+ - * /` only, so it is bit-portable; bounded at 4096 steps, so a spring
/// somebody has made non-decaying cannot hang a fixed step.
pub fn discrete_peak_gain(k: f64, c: f64, dt: f64) -> f64 {
    if !k.is_finite() || !c.is_finite() || !dt.is_finite() || dt <= 0.0 {
        return 0.0;
    }
    let mut s = Spring1 {
        position: 0.0,
        velocity: 1.0,
    };
    let mut peak = 0.0_f64;
    for _ in 0..4096 {
        s.advance(k, c, dt);
        if s.position > peak {
            peak = s.position;
        } else {
            break;
        }
    }
    peak
}

/// **The velocity impulse that makes this spring peak at `peak`**, at the fixed
/// step it will be advanced with.
///
/// `peak / discrete_peak_gain(k, c, dt)`. See [`discrete_peak_gain`] for why the
/// closed form `peak * sqrt(k) * e` is not what ships. Answers `0.0` for a
/// non-finite input or a spring with no gain, which is the refusal-as-a-value
/// the rest of this crate makes.
pub fn impulse_for_peak(peak: f64, k: f64, c: f64, dt: f64) -> f64 {
    if !peak.is_finite() {
        return 0.0;
    }
    let gain = discrete_peak_gain(k, c, dt);
    if gain <= 0.0 {
        return 0.0;
    }
    peak / gain
}

/// **One axis of spring**, position and velocity — the doc's `Spring3D` with the
/// target pinned at zero, because a recoil spring's rest position is the hand
/// where the animation put it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spring1 {
    /// Displacement from rest.
    pub position: f64,
    /// Rate of change of [`position`](Self::position).
    pub velocity: f64,
}

impl Spring1 {
    /// **One step**, semi-implicit Euler — the doc's own integration, with the
    /// velocity updated first so the position uses the velocity it leaves with.
    ///
    /// ```text
    /// a      = -k*x - c*v
    /// v(t+h) = v(t) + a*h
    /// x(t+h) = x(t) + v(t+h)*h
    /// ```
    ///
    /// A non-finite `dt`, `k` or `c` leaves the spring exactly where it was: a
    /// refusal is a value, and a NaN in a spring reaches the trace.
    pub fn advance(&mut self, k: f64, c: f64, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 || !k.is_finite() || !c.is_finite() {
            return;
        }
        if !self.position.is_finite() || !self.velocity.is_finite() {
            *self = Self::default();
            return;
        }
        let accel = -k * self.position - c * self.velocity;
        self.velocity += accel * dt;
        self.position += self.velocity * dt;
    }

    /// Add to the velocity — the doc's `add_impulse`.
    pub fn add_impulse(&mut self, impulse: f64) {
        if impulse.is_finite() {
            self.velocity += impulse;
        }
    }

    /// Whether this spring is exactly at rest — both numbers zero.
    ///
    /// Exact and not a tolerance, because this is what decides whether a
    /// character's bytes are folded at all: "near zero" would make a trace's
    /// LENGTH a function of an epsilon.
    pub fn at_rest(&self) -> bool {
        self.position == 0.0 && self.velocity == 0.0
    }

    /// **Snap a spring that has decayed below the epsilons to EXACT zero.**
    ///
    /// A critically damped spring approaches zero and never arrives, so without
    /// this a character that fired one round on step 12 of a level would carry
    /// a denormal in its trace section for the rest of the session — measured
    /// on `phase30_gameplay_gate`'s own hero, whose hold-point spring was
    /// sitting at `-3.5e-40 m` two stations after the shot. That is not a
    /// rounding curiosity: [`at_rest`](Self::at_rest) is what decides whether
    /// this shooter's 160 bytes are folded at all, so a spring that never
    /// reaches zero is a trace section that never empties, and the
    /// empty-when-nothing-is-happening rule is the whole reason the pre-wave
    /// traces are byte-identical.
    ///
    /// The epsilons are stated in the quantity's own units by the caller
    /// ([`SPRING_REST_M`] / [`SPRING_REST_DEG`] and their rates), and they are
    /// a **billionth** of anything anybody can see: a nanometre of hold point
    /// and a nanodegree of aim.
    pub fn settle(&mut self, eps_pos: f64, eps_vel: f64) {
        if self.position.abs() < eps_pos && self.velocity.abs() < eps_vel {
            self.position = 0.0;
            self.velocity = 0.0;
        }
    }
}

/// **A hold-point spring at rest**, metres — see [`Spring1::settle`].
pub const SPRING_REST_M: f64 = 1.0e-9;

/// **A hold-point spring at rest**, m/s.
pub const SPRING_REST_MPS: f64 = 1.0e-9;

/// **An aim spring at rest**, degrees — see [`Spring1::settle`].
pub const SPRING_REST_DEG: f64 = 1.0e-9;

/// **An aim spring at rest**, degrees a second.
pub const SPRING_REST_DPS: f64 = 1.0e-9;

/// **Three axes of spring**, in the aim frame: `x` right, `y` up, `z` forward.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spring3 {
    /// Displacement from rest, metres.
    pub position: DVec3,
    /// Rate of change, m/s.
    pub velocity: DVec3,
}

impl Spring3 {
    /// [`Spring1::advance`] on each axis.
    pub fn advance(&mut self, k: f64, c: f64, dt: f64) {
        let mut x = Spring1 {
            position: self.position.x,
            velocity: self.velocity.x,
        };
        let mut y = Spring1 {
            position: self.position.y,
            velocity: self.velocity.y,
        };
        let mut z = Spring1 {
            position: self.position.z,
            velocity: self.velocity.z,
        };
        x.advance(k, c, dt);
        y.advance(k, c, dt);
        z.advance(k, c, dt);
        self.position = DVec3::new(x.position, y.position, z.position);
        self.velocity = DVec3::new(x.velocity, y.velocity, z.velocity);
    }

    /// [`Spring1::add_impulse`] on each axis.
    pub fn add_impulse(&mut self, impulse: DVec3) {
        if impulse.is_finite() {
            self.velocity += impulse;
        }
    }

    /// Whether both vectors are exactly zero. See [`Spring1::at_rest`].
    pub fn at_rest(&self) -> bool {
        self.position == DVec3::ZERO && self.velocity == DVec3::ZERO
    }

    /// [`Spring1::settle`] on the vector as a whole — all three axes or none,
    /// so a spring cannot be at rest in `x` and moving in `y`.
    pub fn settle(&mut self, eps_pos: f64, eps_vel: f64) {
        if self.position.abs().max_element() < eps_pos
            && self.velocity.abs().max_element() < eps_vel
        {
            self.position = DVec3::ZERO;
            self.velocity = DVec3::ZERO;
        }
    }
}

// ── the profile ─────────────────────────────────────────────────────────────

/// **Salt for the recoil's first two uniforms** — the jitter and the pitch.
///
/// The spread already draws `(u1, u2)` from `(spread_seed, shots)`; drawing the
/// recoil's noise from the same two would make a shot that scattered left also
/// kick left for ever. One salt, one XOR, no second generator.
pub const RECOIL_SALT: u64 = 0x5750_4e32_6b69_636b;

/// **Salt for the recoil's third uniform** — the yaw drift.
pub const RECOIL_YAW_SALT: u64 = 0x5750_4e32_7961_7721;

/// The doc's own lateral jitter half-width, metres (`(rand - 0.5) * 0.02`).
pub const VM_JITTER_M: f64 = 0.01;

/// **A weapon's recoil, as five numbers** — the doc's `RecoilProfile`, with its
/// own `from_recoil_stat` mapping.
///
/// The two translational fields are **metres of peak hold-point displacement**;
/// the three rotational ones are **radians of peak aim displacement** (the
/// doc's own unit — its `generate_impulses` feeds them to `Quat::from_euler`).
/// See the module header for what "peak" buys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoilProfile {
    /// Backward push along the aim, metres. **Negative** — the doc's
    /// `-0.05 - 0.15*factor`.
    pub vm_kick_back: f64,
    /// Upward lift, metres — the doc's `0.02 + 0.08*factor`.
    pub vm_kick_up: f64,
    /// Half-width of the lateral jitter, metres — the doc's `0.01`.
    pub vm_jitter: f64,
    /// Smallest upward aim impulse, radians — the doc's `0.03 + 0.07*factor`.
    pub cam_pitch_min: f64,
    /// Largest upward aim impulse, radians — the doc's `0.05 + 0.12*factor`.
    pub cam_pitch_max: f64,
    /// Half-width of the horizontal aim drift, radians — the doc's
    /// `0.01 + 0.05*factor`.
    pub cam_yaw_variance: f64,
}

impl RecoilProfile {
    /// **The doc's mapping**, `stat` on its 1..10 scale — the registry's
    /// `recoil_intensity`.
    ///
    /// A stat outside `[0, 10]` is clamped rather than refused: the registry
    /// clamps the field at its own door too, and a profile is not the place to
    /// discover that a row is wrong.
    pub fn from_recoil_stat(stat: f64) -> Self {
        let stat = if stat.is_finite() {
            stat.clamp(0.0, 10.0)
        } else {
            0.0
        };
        let factor = stat / 10.0;
        Self {
            vm_kick_back: -0.05 - (0.15 * factor),
            vm_kick_up: 0.02 + (0.08 * factor),
            vm_jitter: VM_JITTER_M,
            cam_pitch_min: 0.03 + (0.07 * factor),
            cam_pitch_max: 0.05 + (0.12 * factor),
            cam_yaw_variance: 0.01 + (0.05 * factor),
        }
    }

    /// **This weapon's profile**, from its `recoil_intensity`.
    pub fn of(def: &WeaponDef) -> Self {
        Self::from_recoil_stat(def.recoil_intensity)
    }

    /// **The peaks one shot asks for**: `(hold-point offset in metres, aim pitch
    /// in degrees, aim yaw in degrees)`.
    ///
    /// The noise is the counter hash and nothing else — `shot` is
    /// [`WeaponState::shots`], which is already sim state and already in
    /// [`crate::weapon::weapon_state_bytes`], so the n-th round of a magazine
    /// kicks identically on two machines by construction.
    pub fn peaks(&self, seed: u64, shot: u64) -> (DVec3, f64, f64) {
        let (u_jitter, u_pitch) = crate::weapon::shot_uniforms(seed ^ RECOIL_SALT, shot);
        let (u_yaw, _) = crate::weapon::shot_uniforms(seed ^ RECOIL_YAW_SALT, shot);
        let vm = DVec3::new(
            (u_jitter - 0.5) * 2.0 * self.vm_jitter,
            self.vm_kick_up,
            self.vm_kick_back,
        );
        let pitch = self.cam_pitch_min + u_pitch * (self.cam_pitch_max - self.cam_pitch_min);
        let yaw = (u_yaw - 0.5) * 2.0 * self.cam_yaw_variance;
        (vm, pitch.to_degrees(), yaw.to_degrees())
    }

    /// **The impulses one shot deposits**: `(viewmodel velocity impulse, aim
    /// pitch impulse in deg/s, aim yaw impulse in deg/s)`.
    ///
    /// `dt` is the fixed step the springs will be advanced with, because the
    /// gain a velocity impulse has is a property of the recurrence and not of
    /// the differential equation — see [`discrete_peak_gain`].
    pub fn impulses(&self, seed: u64, shot: u64, dt: f64) -> (DVec3, f64, f64) {
        let (vm, pitch, yaw) = self.peaks(seed, shot);
        (
            DVec3::new(
                impulse_for_peak(vm.x, VM_STIFFNESS, VM_DAMPING, dt),
                impulse_for_peak(vm.y, VM_STIFFNESS, VM_DAMPING, dt),
                impulse_for_peak(vm.z, VM_STIFFNESS, VM_DAMPING, dt),
            ),
            impulse_for_peak(pitch, AIM_STIFFNESS, AIM_DAMPING, dt),
            impulse_for_peak(yaw, AIM_STIFFNESS, AIM_DAMPING, dt),
        )
    }
}

// ── the sway ────────────────────────────────────────────────────────────────

/// **How far a walking character's hold point wanders**, metres at one metre per
/// second.
///
/// Two centimetres — a hand's own drift, an order of magnitude under the
/// [`RecoilProfile`]'s kick, because a sway a player can *see* at a walk is a
/// sway they cannot aim through.
pub const SWAY_PER_MPS_M: f64 = 0.02;

/// **The ceiling on the bob**, metres. A sprint is not a trampoline.
pub const SWAY_MAX_M: f64 = 0.09;

/// **How far a breathing character's hold point wanders**, metres.
///
/// Six millimetres, which is a chest. It is the term that makes the sway
/// non-zero on a character standing perfectly still — and the term whose
/// ABSENCE is what "dead-still and not breathing" means.
pub const SWAY_BREATH_M: f64 = 0.006;

/// **The bob's frequency**, cycles a second at one metre per second.
///
/// Bound to the animation's own state clock rather than to a phase of its own,
/// because the sway and the breath additive (`crate::pose::apply_breath`, wave
/// CHAR1b.2) must be ONE clock: two clocks drift apart, and a chest that rises
/// while the hands fall is the thing a player reads as "floaty".
pub const SWAY_BOB_HZ: f64 = 0.9;

/// **The breath's frequency**, cycles a second — 12 breaths a minute at rest.
pub const SWAY_BREATH_HZ: f64 = 0.2;

/// **How fast the mouse-delta lag catches up**, 1/s.
///
/// Six, which is a sixth of a second of trail. The doc's *"mouse deltas for
/// sway"*: the hold point lags a fast flick and catches up, so a weapon has
/// weight.
pub const SWAY_LOOK_LAG_HZ: f64 = 6.0;

/// **Metres of lateral hold-point offset per degree-per-second of lagged look
/// rate.**
///
/// `0.0006`, so a 180 deg/s flick trails the weapon by about 11 cm at its worst
/// and by nothing at all once the mouse stops.
pub const SWAY_LOOK_M_PER_DPS: f64 = 0.000_6;

/// **The sway offset**, metres in the aim frame, from the ONE clock.
///
/// `speed_mps` is the character's planar speed and `breath` is `1.0` for a
/// character whose animation clock is running and `0.0` for one that is dead,
/// unrigged, or otherwise not breathing. **Exactly zero** when both are zero and
/// the look lag has settled, which is the arm clause 4 names.
pub fn sway_offset(clock_s: f64, speed_mps: f64, breath: f64, look_lag: (f64, f64)) -> DVec3 {
    if !clock_s.is_finite() {
        return DVec3::ZERO;
    }
    let speed = if speed_mps.is_finite() {
        speed_mps.max(0.0)
    } else {
        0.0
    };
    let breath = if breath.is_finite() {
        breath.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bob_amp = (speed * SWAY_PER_MPS_M).min(SWAY_MAX_M);
    let breath_amp = breath * SWAY_BREATH_M;
    // A figure of eight: the horizontal term runs at half the vertical one, the
    // shape a walk cycle draws, so the hands do not simply bounce.
    let w = std::f64::consts::TAU * SWAY_BOB_HZ * clock_s;
    let bob = DVec3::new(
        inf_math::psin64(w * 0.5) * bob_amp,
        inf_math::psin64(w) * bob_amp * 0.5,
        0.0,
    );
    let bw = std::f64::consts::TAU * SWAY_BREATH_HZ * clock_s;
    let breathe = DVec3::new(0.0, inf_math::psin64(bw) * breath_amp, 0.0);
    let lx = if look_lag.0.is_finite() {
        look_lag.0
    } else {
        0.0
    };
    let ly = if look_lag.1.is_finite() {
        look_lag.1
    } else {
        0.0
    };
    let trail = DVec3::new(-lx * SWAY_LOOK_M_PER_DPS, -ly * SWAY_LOOK_M_PER_DPS, 0.0);
    bob + breathe + trail
}

// ── the spread ──────────────────────────────────────────────────────────────

/// **The cone multiplier while aiming down the sights.**
///
/// 0.45 — an aimed weapon is a little over twice as accurate as a hip-fired one,
/// which is the ordering clause 3's arm asserts and the number a 25 m pattern is
/// measured against.
pub const SPREAD_ADS_MULT: f64 = 0.45;

/// **The cone multiplier while crouched.** 0.70.
pub const SPREAD_CROUCH_MULT: f64 = 0.70;

/// **The cone multiplier while prone.** 0.55 — steadier than a crouch.
pub const SPREAD_PRONE_MULT: f64 = 0.55;

/// **The cone multiplier at a full sprint.**
///
/// 2.6, reached linearly from 1.0 at a standstill over [`SPREAD_MOVE_REF_MPS`].
/// A player who fires while running is asking for it.
pub const SPREAD_MOVE_MULT_MAX: f64 = 2.6;

/// **The speed at which the movement penalty saturates**, m/s — a sprint.
pub const SPREAD_MOVE_REF_MPS: f64 = 6.0;

/// **How much cone one round adds**, degrees per point of `recoil_intensity`.
///
/// 0.06, so a 10-recoil weapon blooms 0.6 deg a shot and a 2.5-recoil pistol
/// 0.15. Derived from the registry's own metric rather than authored 85 times: a
/// weapon that kicks hard scatters hard, and a second column would be a place for
/// the two to disagree.
pub const BLOOM_PER_SHOT_DEG_PER_POINT: f64 = 0.06;

/// **The ceiling on the bloom**, degrees per point of `recoil_intensity`.
///
/// 0.45, so the bloom saturates after seven or eight rounds of sustained fire and
/// a magazine does not open into a shotgun.
pub const BLOOM_MAX_DEG_PER_POINT: f64 = 0.45;

/// **How fast the bloom decays**, in multiples of its own ceiling per second.
///
/// 2.5 — a full bloom is gone 0.4 s after the trigger comes up, which is about
/// the time it takes to re-aim.
pub const BLOOM_DECAY_PER_S: f64 = 2.5;

/// **How much cone one more round of this weapon adds**, degrees.
pub fn bloom_per_shot_deg(def: &WeaponDef) -> f64 {
    def.recoil_intensity.max(0.0) * BLOOM_PER_SHOT_DEG_PER_POINT
}

/// **The most bloom this weapon may carry**, degrees.
pub fn bloom_max_deg(def: &WeaponDef) -> f64 {
    def.recoil_intensity.max(0.0) * BLOOM_MAX_DEG_PER_POINT
}

/// **How fast this weapon's bloom decays**, degrees a second.
pub fn bloom_decay_dps(def: &WeaponDef) -> f64 {
    bloom_max_deg(def) * BLOOM_DECAY_PER_S
}

/// **The stance a shot is fired from**, for the spread's sake.
///
/// Its own three-value enum rather than `MovementMode`, because the spread only
/// cares about three of the mode table's fourteen and a `match` over all of them
/// would be a place for a new mode to change ballistics by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShotStance {
    /// On both feet.
    Standing,
    /// Crouched.
    Crouched,
    /// Prone.
    Prone,
}

/// **The whole cone a shot leaves through**, degrees — the registry's per-row
/// `spread_deg`, plus the bloom this weapon has accumulated, times the stance,
/// movement and ADS multipliers.
///
/// `ads` is the blend in `[0, 1]`, not a flag: a shot fired half way into an aim
/// gets half the benefit, which is what stops "tap aim and fire" being strictly
/// better than aiming.
pub fn resolved_cone_deg(
    def: &WeaponDef,
    bloom_deg: f64,
    stance: ShotStance,
    speed_mps: f64,
    ads: f64,
) -> f64 {
    let base = if def.spread_deg.is_finite() {
        def.spread_deg.max(0.0)
    } else {
        0.0
    };
    let bloom = if bloom_deg.is_finite() {
        bloom_deg.clamp(0.0, bloom_max_deg(def))
    } else {
        0.0
    };
    let stance_mult = match stance {
        ShotStance::Standing => 1.0,
        ShotStance::Crouched => SPREAD_CROUCH_MULT,
        ShotStance::Prone => SPREAD_PRONE_MULT,
    };
    let speed = if speed_mps.is_finite() {
        speed_mps.clamp(0.0, SPREAD_MOVE_REF_MPS)
    } else {
        0.0
    };
    let move_mult = 1.0 + (SPREAD_MOVE_MULT_MAX - 1.0) * (speed / SPREAD_MOVE_REF_MPS);
    let ads = if ads.is_finite() {
        ads.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ads_mult = 1.0 + (SPREAD_ADS_MULT - 1.0) * ads;
    ((base + bloom) * stance_mult * move_mult * ads_mult).max(0.0)
}

// ── aim down sights ─────────────────────────────────────────────────────────

/// **How much slower an aiming character walks**, on top of the weapon's own
/// `move_speed_mult`.
///
/// 0.55 — an aimed walk, which is what every third-person shooter this engine is
/// measured against does. It is an engine constant and not an eighty-sixth
/// registry column because the doc's own table has no such metric, and inventing
/// one per row would be authoring numbers nothing measured.
pub const ADS_MOVE_SPEED_MULT: f64 = 0.55;

/// **How far into the aim block a blend has to get to count as arrived.**
///
/// 0.98. An exponential blend never arrives, so "the FOV reached 55 deg" needs a
/// number; 98 % of the island's own 70 to 55 is **0.3 deg** short, which is a
/// fifth of one degree of a 55-degree field and below anything a frame shows.
/// [`blend_speed_for_ads`] is solved against this and `wpn2b_gate` measures the
/// world against the same 0.3 deg.
pub const ADS_BLEND_REACH: f64 = 0.98;

/// **The camera rig value an ADS time is spent through** — the name
/// `crate::camera::set_camera_rig_value` takes.
pub const ADS_BLEND_RIG_KEY: &str = "state_blend_speed";

/// **The `state_blend_speed` that spends exactly `ads_s` reaching the aim
/// block**, given the fixed step `dt`.
///
/// `crate::camera::interp_to` moves a fraction `a = speed * dt` of the remaining
/// distance each step, so after `n = ads_s / dt` steps the remainder is
/// `(1 - a)^n` and the number wanted is the `a` for which that equals
/// `1 - ADS_BLEND_REACH`.
///
/// **Solved by bisection, in `+ - * /` only.** The closed form is
/// `a = 1 - eps^(dt/ads_s)`, and a `powf` there would put a transcendental on a
/// number that reaches a camera pose — the P14 law's whole subject. Fifty-eight
/// halvings of `[0, 1]` reach the last bit of an `f64`, and the whole solve runs
/// **once per equip**, not once per step.
///
/// Answers a snap (`1 / dt`, which `interp_to` clamps to "arrive this step") for
/// an ADS time at or under one step.
pub fn blend_speed_for_ads(ads_s: f64, dt: f64) -> f64 {
    if !dt.is_finite() || dt <= 0.0 {
        return 0.0;
    }
    if !ads_s.is_finite() || ads_s <= dt {
        return 1.0 / dt;
    }
    let n = (ads_s / dt).floor().max(1.0).min(4096.0) as u32;
    let target = 1.0 - ADS_BLEND_REACH;
    // `remaining(a) = (1 - a)^n` is monotonically DECREASING in `a`, so the
    // bracket is [0, 1] and the test is "have we passed the target yet".
    let remaining = |a: f64| -> f64 {
        let mut acc = 1.0_f64;
        let mut b = 1.0 - a;
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                acc *= b;
            }
            b *= b;
            e >>= 1;
        }
        acc
    };
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..58 {
        let mid = (lo + hi) * 0.5;
        if remaining(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ((lo + hi) * 0.5) / dt
}

// ── the component ───────────────────────────────────────────────────────────

/// **A shooter's feel**, a runtime component — installed beside [`WeaponState`]
/// when something is equipped and removed with it.
///
/// Never serialized, so **no schema moves** (scene v27 and `ScenePayload` 13 are
/// untouched by this wave); folded at the TAIL of `RuntimeSim::state_bytes` by
/// [`feel_state_bytes`], and **only for the characters that are not at rest**,
/// which is what keeps a level whose hero has never fired byte-identical to its
/// pre-wave self.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct WeaponFeel {
    /// The hold-point spring, metres in the aim frame (`x` right, `y` up, `z`
    /// forward). Read by `inf_physics::d3::gameplay::aim_hold_point`.
    pub vm: Spring3,
    /// The aim pitch spring, **degrees**. Its position is the recoil offset the
    /// look integrator has added to `aim_pitch_deg`.
    pub aim_pitch: Spring1,
    /// The aim yaw spring, degrees.
    pub aim_yaw: Spring1,
    /// How much of [`aim_pitch`](Self::aim_pitch)'s position the look integrator
    /// has already spent on the aim.
    ///
    /// The centre-recovery is exactly this: the integrator adds the DELTA each
    /// step, so as the spring returns to zero it gives back precisely what it
    /// took and the aim ends where the player left it. A residual would be a
    /// weapon that walked the reticle up the wall over a magazine.
    pub applied_pitch_deg: f64,
    /// See [`applied_pitch_deg`](Self::applied_pitch_deg).
    pub applied_yaw_deg: f64,
    /// The aim-down-sights blend, `[0, 1]` — `0` hip, `1` fully aimed, moving
    /// between them over the weapon's own `ads_time_ms`.
    pub ads_blend: f64,
    /// The resolved sway offset, metres in the aim frame. Derived, and folded
    /// anyway: it is what the hands are actually holding, and a host that
    /// disagreed about it would draw a different weapon.
    pub sway: DVec3,
    /// The lagged look rate, degrees a second — the mouse-delta term of the
    /// sway.
    pub look_lag_x: f64,
    /// See [`look_lag_x`](Self::look_lag_x).
    pub look_lag_y: f64,
    /// The `state_blend_speed` this character's camera rig carried before the
    /// ADS took it, so putting the weapon away gives back exactly what was
    /// there. `NaN` means "not taken".
    pub blend_speed_prior: f64,
}

impl WeaponFeel {
    /// A feel with nothing taken and nothing owed.
    pub fn new() -> Self {
        Self {
            blend_speed_prior: f64::NAN,
            ..Default::default()
        }
    }

    /// **Whether this feel is exactly at rest** — every spring zero, no blend,
    /// no sway, no lag.
    ///
    /// What [`feel_state_bytes`] folds on. Exact, for [`Spring1::at_rest`]'s
    /// reason: a tolerance would make a trace's length a function of an epsilon.
    /// `blend_speed_prior` is deliberately NOT part of it — it is bookkeeping
    /// about a camera, the one thing that is never sim state.
    pub fn at_rest(&self) -> bool {
        self.vm.at_rest()
            && self.aim_pitch.at_rest()
            && self.aim_yaw.at_rest()
            && self.applied_pitch_deg == 0.0
            && self.applied_yaw_deg == 0.0
            && self.ads_blend == 0.0
            && self.sway == DVec3::ZERO
            && self.look_lag_x == 0.0
            && self.look_lag_y == 0.0
    }

    /// **Deposit one shot's recoil** — the gameplay phase's whole write.
    ///
    /// The impulses land on the springs' VELOCITY, so the look integrator (which
    /// runs a phase EARLIER, in the next fixed step) sees them without this
    /// having moved the aim itself. That one step of latency is the muzzle's own
    /// (`muzzle_of`'s stated seam) and it is the right way round: the round that
    /// caused the kick leaves along the aim the player had.
    pub fn fire(&mut self, profile: &RecoilProfile, seed: u64, shot: u64, dt: f64) {
        let (vm, pitch, yaw) = profile.impulses(seed, shot, dt);
        self.vm.add_impulse(vm);
        self.aim_pitch.add_impulse(pitch);
        self.aim_yaw.add_impulse(yaw);
    }
}

/// **What the look integrator hands this module each step.**
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeelInputs {
    /// Whether the character is aiming down the sights.
    pub aiming: bool,
    /// The equipped weapon's `ads_time_ms`, in seconds. Zero means "snap".
    pub ads_time_s: f64,
    /// Planar speed, m/s — the bob's amplitude.
    pub speed_mps: f64,
    /// `1.0` for a character whose animation clock is running, `0.0` otherwise —
    /// the breath term's gate.
    pub breath: f64,
    /// **The ONE clock**, seconds — the same `inf_anim` state time the breath
    /// additive (`crate::pose::apply_breath`) samples, read through
    /// [`crate::anim_bridge::anim_state_time`].
    pub clock_s: f64,
    /// The look rate the player is asking for, degrees a second.
    pub look_yaw_dps: f64,
    /// See [`look_yaw_dps`](Self::look_yaw_dps).
    pub look_pitch_dps: f64,
}

/// **What one step of feel did to the aim** — the delta the look integrator adds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FeelStep {
    /// Degrees to add to `aim_pitch_deg` this step.
    pub aim_pitch_delta_deg: f64,
    /// Degrees to add to `aim_yaw_deg` this step.
    pub aim_yaw_delta_deg: f64,
}

/// **One step of the whole feel** — both springs, the sway, the lag and the ADS
/// blend, as a pure function of the component and the inputs.
///
/// Called from exactly one place, `inf_physics::d3::movement`'s look integrator,
/// so the springs advance on the clock the aim does and a second caller cannot
/// advance them twice.
pub fn advance_feel(feel: &mut WeaponFeel, input: &FeelInputs, dt: f64) -> FeelStep {
    if !dt.is_finite() || dt <= 0.0 {
        return FeelStep::default();
    }
    feel.vm.advance(VM_STIFFNESS, VM_DAMPING, dt);
    feel.aim_pitch.advance(AIM_STIFFNESS, AIM_DAMPING, dt);
    feel.aim_yaw.advance(AIM_STIFFNESS, AIM_DAMPING, dt);
    // …and a spring that has decayed past anything anybody can see is put
    // EXACTLY at rest, because `at_rest` is what empties this shooter's trace
    // section. See `Spring1::settle`.
    feel.vm.settle(SPRING_REST_M, SPRING_REST_MPS);
    feel.aim_pitch.settle(SPRING_REST_DEG, SPRING_REST_DPS);
    feel.aim_yaw.settle(SPRING_REST_DEG, SPRING_REST_DPS);
    let out = FeelStep {
        aim_pitch_delta_deg: feel.aim_pitch.position - feel.applied_pitch_deg,
        aim_yaw_delta_deg: feel.aim_yaw.position - feel.applied_yaw_deg,
    };
    feel.applied_pitch_deg = feel.aim_pitch.position;
    feel.applied_yaw_deg = feel.aim_yaw.position;
    // The applied accumulators go to rest WITH their springs, which is the same
    // statement twice: the delta above has just given the aim back everything
    // the spring had, so what is owed is zero and the field says so.
    if feel.aim_pitch.at_rest() {
        feel.applied_pitch_deg = 0.0;
    }
    if feel.aim_yaw.at_rest() {
        feel.applied_yaw_deg = 0.0;
    }
    // The mouse-delta lag: a first-order chase of the look rate, so a flick
    // trails and a still mouse trails nothing.
    let a = (SWAY_LOOK_LAG_HZ * dt).clamp(0.0, 1.0);
    let want_x = if input.look_yaw_dps.is_finite() {
        input.look_yaw_dps
    } else {
        0.0
    };
    let want_y = if input.look_pitch_dps.is_finite() {
        input.look_pitch_dps
    } else {
        0.0
    };
    feel.look_lag_x += (want_x - feel.look_lag_x) * a;
    feel.look_lag_y += (want_y - feel.look_lag_y) * a;
    // A lag that has decayed to a denormal is a trace section that never
    // empties, so it is snapped to zero once it is worth less than a micrometre
    // of offset. The threshold is stated in the OFFSET's units, not the rate's.
    if feel.look_lag_x.abs() * SWAY_LOOK_M_PER_DPS < 1.0e-6 {
        feel.look_lag_x = 0.0;
    }
    if feel.look_lag_y.abs() * SWAY_LOOK_M_PER_DPS < 1.0e-6 {
        feel.look_lag_y = 0.0;
    }
    feel.sway = sway_offset(
        input.clock_s,
        input.speed_mps,
        input.breath,
        (feel.look_lag_x, feel.look_lag_y),
    );
    // The ADS blend, LINEAR over the weapon's own `ads_time_ms` — the pose's
    // half of the aim. (The camera's half is the rig's `state_blend_speed`,
    // which is exponential because `interp_to` is; the two agree at the moment
    // `blend_speed_for_ads` solves for and nowhere else, which is stated rather
    // than pretended.)
    let rate = if input.ads_time_s.is_finite() && input.ads_time_s > dt {
        dt / input.ads_time_s
    } else {
        1.0
    };
    let target = f64::from(u8::from(input.aiming));
    if feel.ads_blend < target {
        feel.ads_blend = (feel.ads_blend + rate).min(target);
    } else if feel.ads_blend > target {
        feel.ads_blend = (feel.ads_blend - rate).max(target);
    }
    out
}

/// **The feel's trace bytes** — 160 a shooter that is not at rest, in `Guid`
/// order, and **empty when every shooter in the world is at rest**.
///
/// Empty is the load-bearing half, exactly as it is for
/// [`crate::ballistics::round_state_bytes`]: it is what keeps every trace
/// committed before this wave byte-identical, and it is why a character who is
/// simply carrying a rifle folds nothing at all.
pub fn feel_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let w = world.world();
    let Some(mut q) = w.try_query::<(&Guid, &WeaponFeel)>() else {
        return Vec::new();
    };
    let mut rows: Vec<(Uuid, WeaponFeel)> = q
        .iter(w)
        .filter(|(_, f)| !f.at_rest())
        .map(|(g, f)| (g.0, *f))
        .collect();
    if rows.is_empty() {
        return Vec::new();
    }
    rows.sort_by_key(|(g, _)| *g);
    let mut out = Vec::with_capacity(rows.len() * 160);
    for (guid, f) in rows {
        out.extend_from_slice(guid.as_bytes());
        for v in [
            f.vm.position.x,
            f.vm.position.y,
            f.vm.position.z,
            f.vm.velocity.x,
            f.vm.velocity.y,
            f.vm.velocity.z,
            f.aim_pitch.position,
            f.aim_pitch.velocity,
            f.aim_yaw.position,
            f.aim_yaw.velocity,
            f.applied_pitch_deg,
            f.applied_yaw_deg,
            f.ads_blend,
            f.sway.x,
            f.sway.y,
            f.sway.z,
            f.look_lag_x,
            f.look_lag_y,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    out
}

/// **This character's feel**, if it has one.
pub fn feel_of(world: &EcsWorld, guid: Uuid) -> Option<WeaponFeel> {
    let entity = world.entity_of(guid)?;
    world.world().get::<WeaponFeel>(entity).copied()
}

/// **How fast this character moves for what it is carrying AND how far into an
/// aim it is** — [`crate::weapon::equipped_move_speed_scale`] with the ADS factor
/// folded in.
///
/// One door, at the one seam [`crate::movement::settings_for`] reads: the
/// weapon's own `move_speed_mult` and the aim's [`ADS_MOVE_SPEED_MULT`] are two
/// reasons to walk slower, and a second call site would be a place for them to
/// disagree. `1.0` for an unarmed character, which is what every caller before
/// wave WPN2a passed.
pub fn equipped_move_speed_scale(world: &EcsWorld, guid: Uuid) -> f64 {
    let base = crate::weapon::equipped_move_speed_scale(world, guid);
    let ads = feel_of(world, guid).map(|f| f.ads_blend).unwrap_or(0.0);
    let ads = if ads.is_finite() {
        ads.clamp(0.0, 1.0)
    } else {
        0.0
    };
    base * (1.0 + (ADS_MOVE_SPEED_MULT - 1.0) * ads)
}

/// **The bloom one shot adds to a weapon's state.**
///
/// On [`WeaponState`] rather than on the feel because the bloom belongs to the
/// MAGAZINE: putting a weapon away and taking it out again is what a player does
/// to reset it in every game that has one, and `WeaponState` is already replaced
/// when the equipped id changes.
pub fn bloom_on_shot(def: &WeaponDef, state: &mut WeaponState) {
    let next = state.spread_bloom_deg + bloom_per_shot_deg(def);
    state.spread_bloom_deg = next.clamp(0.0, bloom_max_deg(def));
}

/// **Decay the bloom by one step.**
pub fn decay_bloom(def: &WeaponDef, state: &mut WeaponState, dt: f64) {
    if !dt.is_finite() || dt <= 0.0 || state.spread_bloom_deg <= 0.0 {
        return;
    }
    state.spread_bloom_deg = (state.spread_bloom_deg - bloom_decay_dps(def) * dt).max(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 60.0;

    /// **The shipped damping IS critical and the doc's is not** — the wave's
    /// first finding, as arithmetic.
    #[test]
    fn the_shipped_damping_is_critical_and_the_docs_is_not() {
        assert!((VM_DAMPING - critical_damping(VM_STIFFNESS)).abs() < 1.0e-12);
        assert!((AIM_DAMPING - critical_damping(AIM_STIFFNESS)).abs() < 1.0e-12);
        assert!(DOC_VM_DAMPING < critical_damping(VM_STIFFNESS));
        assert!(DOC_AIM_DAMPING < critical_damping(AIM_STIFFNESS));
    }

    /// The overshoot, measured on the spring rather than argued: critical
    /// damping never crosses zero, the doc's numbers do.
    #[test]
    fn a_critically_damped_spring_does_not_overshoot_and_the_docs_one_does() {
        let overshoot = |c: f64| -> f64 {
            let mut s = Spring1::default();
            s.add_impulse(impulse_for_peak(1.0, VM_STIFFNESS, c, DT));
            let mut peak = 0.0_f64;
            let mut worst = 0.0_f64;
            for _ in 0..600 {
                s.advance(VM_STIFFNESS, c, DT);
                peak = peak.max(s.position);
                worst = worst.min(s.position);
            }
            -worst / peak
        };
        let critical = overshoot(VM_DAMPING);
        let doc = overshoot(DOC_VM_DAMPING);
        println!("overshoot: critical {critical:.6}, the doc's c=18 {doc:.6}");
        assert!(critical < 1.0e-9, "{critical}");
        assert!(doc > 0.05, "{doc}");
    }

    /// **A peak is a peak** — the whole reason the profile's numbers mean
    /// anything. Within 1 % of the requested displacement at a 60 Hz step.
    #[test]
    fn an_impulse_peaks_where_it_was_asked_to() {
        for want in [0.05_f64, 0.2, 1.0, 5.0] {
            let mut s = Spring1::default();
            s.add_impulse(impulse_for_peak(want, AIM_STIFFNESS, AIM_DAMPING, DT));
            let mut peak = 0.0_f64;
            for _ in 0..600 {
                s.advance(AIM_STIFFNESS, AIM_DAMPING, DT);
                peak = peak.max(s.position);
            }
            let err = (peak - want).abs() / want;
            println!("peak wanted {want}, got {peak:.6} ({:.4} %)", err * 100.0);
            assert!(err < 1.0e-9, "wanted {want}, peaked at {peak}");
        }
    }

    /// It comes all the way home, which is the centre-recovery.
    #[test]
    fn a_spring_returns_to_rest() {
        let mut s = Spring1::default();
        s.add_impulse(impulse_for_peak(3.0, AIM_STIFFNESS, AIM_DAMPING, DT));
        for _ in 0..120 {
            s.advance(AIM_STIFFNESS, AIM_DAMPING, DT);
        }
        assert!(s.position.abs() < 1.0e-3, "{}", s.position);
    }

    /// The doc's own 1-10 mapping, at its own three anchors.
    #[test]
    fn the_profile_is_the_docs_mapping_and_it_is_monotone() {
        let p = RecoilProfile::from_recoil_stat(10.0);
        assert!((p.vm_kick_back - -0.2).abs() < 1.0e-12);
        assert!((p.vm_kick_up - 0.1).abs() < 1.0e-12);
        assert!((p.cam_pitch_min - 0.1).abs() < 1.0e-12);
        assert!((p.cam_pitch_max - 0.17).abs() < 1.0e-12);
        assert!((p.cam_yaw_variance - 0.06).abs() < 1.0e-12);
        let mut last = -1.0;
        for stat in 0..=10 {
            let p = RecoilProfile::from_recoil_stat(f64::from(stat));
            let (_, pitch, _) = p.peaks(0, 0);
            assert!(pitch > last, "stat {stat} did not raise the kick");
            last = pitch;
        }
        // A NaN stat is the zero profile, not a NaN kick.
        let bad = RecoilProfile::from_recoil_stat(f64::NAN);
        assert_eq!(bad, RecoilProfile::from_recoil_stat(0.0));
    }

    /// The noise is a counter hash: the same shot index answers the same jitter,
    /// for ever, and two different indices do not.
    #[test]
    fn the_recoil_noise_is_a_counter_and_not_a_generator() {
        let p = RecoilProfile::from_recoil_stat(5.0);
        assert_eq!(p.peaks(7, 3), p.peaks(7, 3));
        assert_ne!(p.peaks(7, 3), p.peaks(7, 4));
        assert_ne!(p.peaks(7, 3), p.peaks(8, 3));
        // And it does NOT reuse the spread's own two uniforms.
        let (spread_u1, _) = crate::weapon::shot_uniforms(7, 3);
        let (jitter_u, _) = crate::weapon::shot_uniforms(7 ^ RECOIL_SALT, 3);
        assert_ne!(spread_u1, jitter_u);
    }

    /// **Sway is exactly zero when nothing is moving and nothing is breathing**
    /// — clause 4's own arm, at the pure-function level.
    #[test]
    fn sway_is_exactly_zero_at_rest_and_grows_with_speed() {
        assert_eq!(sway_offset(12.5, 0.0, 0.0, (0.0, 0.0)), DVec3::ZERO);
        let amp = |speed: f64| -> f64 {
            let mut worst = 0.0_f64;
            for i in 0..600 {
                let t = f64::from(i) * DT;
                worst = worst.max(sway_offset(t, speed, 1.0, (0.0, 0.0)).length());
            }
            worst
        };
        let idle = amp(0.0);
        let walk = amp(1.5);
        let sprint = amp(6.0);
        println!("sway: idle {idle:.5} m, walk {walk:.5} m, sprint {sprint:.5} m");
        assert!(idle > 0.0 && idle < walk, "{idle} {walk}");
        assert!(walk < sprint, "{walk} {sprint}");
        assert!(sprint <= SWAY_MAX_M * 1.2, "{sprint}");
    }

    /// The cone ordering the pattern arm measures in the world, here as
    /// arithmetic: aimed and crouched and still beats hip and still beats moving.
    #[test]
    fn the_cone_ordering_is_ads_crouched_then_hip_then_moving() {
        let mut def = WeaponDef {
            spread_deg: 1.2,
            recoil_intensity: 3.0,
            ..Default::default()
        };
        let ads = resolved_cone_deg(&def, 0.0, ShotStance::Crouched, 0.0, 1.0);
        let hip = resolved_cone_deg(&def, 0.0, ShotStance::Standing, 0.0, 0.0);
        let moving = resolved_cone_deg(&def, 0.0, ShotStance::Standing, 6.0, 0.0);
        println!("cone: ads {ads:.4}, hip {hip:.4}, moving {moving:.4}");
        assert!(ads < hip, "{ads} {hip}");
        assert!(hip < moving, "{hip} {moving}");
        // The bloom is bounded and it bites.
        def.recoil_intensity = 10.0;
        let mut state = WeaponState::full("x", &def);
        for _ in 0..100 {
            bloom_on_shot(&def, &mut state);
        }
        assert!((state.spread_bloom_deg - bloom_max_deg(&def)).abs() < 1.0e-12);
        decay_bloom(&def, &mut state, 1.0);
        assert!(state.spread_bloom_deg < bloom_max_deg(&def));
        for _ in 0..600 {
            decay_bloom(&def, &mut state, DT);
        }
        assert_eq!(state.spread_bloom_deg, 0.0);
    }

    /// **The ADS blend speed arrives when it said it would** — the solve, against
    /// the camera's own `interp_to`.
    #[test]
    fn the_ads_blend_speed_arrives_on_time() {
        for ms in [90.0_f64, 140.0, 200.0, 320.0, 520.0] {
            let t = ms / 1000.0;
            let speed = blend_speed_for_ads(t, DT);
            // Walk the camera's own interpolation from 70 to 55.
            let mut fov = 70.0_f64;
            let mut steps = 0;
            while (fov - 55.0).abs() > 15.0 * (1.0 - ADS_BLEND_REACH) && steps < 600 {
                fov = crate::camera::interp_to(fov, 55.0, speed, DT);
                steps += 1;
            }
            let took = f64::from(steps) * DT;
            let frames = (took - t).abs() / DT;
            println!("ads {ms} ms: speed {speed:.4}, arrived in {took:.4} s ({frames:.2} frames)");
            assert!(frames <= 1.0, "{ms} ms arrived {frames} frames out");
        }
        // Under one step is a snap.
        assert!(blend_speed_for_ads(0.001, DT) * DT >= 1.0);
    }

    /// A feel that has never been touched folds nothing, and one that has been
    /// folds 176 bytes.
    #[test]
    fn a_resting_feel_folds_nothing() {
        let mut w = EcsWorld::new();
        let guid = Uuid::from_u128(1);
        let e = w.spawn_with_guid(guid, "Shooter", None);
        w.world_mut().entity_mut(e).insert(WeaponFeel::new());
        assert!(feel_state_bytes(&w).is_empty());
        let mut feel = WeaponFeel::new();
        feel.fire(&RecoilProfile::from_recoil_stat(4.0), 0, 1, DT);
        advance_feel(
            &mut feel,
            &FeelInputs {
                aiming: false,
                ads_time_s: 0.2,
                speed_mps: 0.0,
                breath: 0.0,
                clock_s: 0.0,
                look_yaw_dps: 0.0,
                look_pitch_dps: 0.0,
            },
            DT,
        );
        *w.world_mut().get_mut::<WeaponFeel>(e).expect("feel") = feel;
        let bytes = feel_state_bytes(&w);
        assert_eq!(bytes.len(), 16 + 18 * 8);
        assert_eq!(&bytes[..16], guid.as_bytes());
    }

    /// **The recovery gives back exactly what it took.** Fire five, wait, and the
    /// sum of the deltas is zero to within a thousandth of a degree.
    #[test]
    fn the_aim_recoil_gives_back_exactly_what_it_took() {
        let profile = RecoilProfile::from_recoil_stat(3.8);
        let mut feel = WeaponFeel::new();
        let input = FeelInputs {
            aiming: false,
            ads_time_s: 0.2,
            speed_mps: 0.0,
            breath: 0.0,
            clock_s: 0.0,
            look_yaw_dps: 0.0,
            look_pitch_dps: 0.0,
        };
        let mut aim = 0.0_f64;
        let mut peak = 0.0_f64;
        for step in 0..300 {
            if step % 6 == 0 && step < 30 {
                feel.fire(&profile, 0, u64::try_from(step / 6).unwrap_or(0) + 1, DT);
            }
            let out = advance_feel(&mut feel, &input, DT);
            aim += out.aim_pitch_delta_deg;
            peak = peak.max(aim);
        }
        println!("burst peak {peak:.4} deg, residual {aim:.6} deg");
        assert!(
            peak > 1.0,
            "a five-round burst barely moved the aim: {peak}"
        );
        assert!(aim.abs() < 1.0e-3, "the aim did not come home: {aim}");
    }
}
