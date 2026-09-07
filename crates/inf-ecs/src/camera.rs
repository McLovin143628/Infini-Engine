//! **The locomotion camera** (P29.6) — the pure half.
//!
//! The engine's first *gameplay* camera. §13 assigned one to no sub-phase at all
//! until the ALS amendment's Ruling 3 put it here, "as the portable six-item
//! subset of ALS's camera manager headlined by `CalculateAxisIndependentLag`".
//! This module is that subset as functions of numbers; the half that needs a
//! world to sweep against is `inf_physics::d3::camera`, the one fixed-step door
//! both hosts call — the same split [`crate::movement`] has, for the same reason.
//!
//! # The six items, and the tax that is not here
//!
//! 1. A **pivot** that is a point on the character rather than its origin.
//! 2. **Axis-independent lag in camera-yaw space** ([`axis_independent_lag`]) —
//!    three separate interp speeds, which is the piece worth copying verbatim.
//! 3. **Rotation lag** — one speed chasing the aim.
//! 4. A **state → settings** table (offsets, lag speeds, arm length, FOV),
//!    blended over transitions.
//! 5. **Sphere-sweep collision** (in the physics half).
//! 6. **First person as a blend weight**, not a separate mode.
//!
//! What is deliberately absent is ALS's `UALSPlayerCameraBehavior`: an
//! `UAnimInstance` on a **dummy skeletal mesh**, whose only job is to hold nine
//! bools and let an animation state machine blend eleven scalar curves that the
//! camera manager then reads back by `FName`. Epic did that because UE had no
//! other cheap way to blend N scalars over a state machine with per-transition
//! durations. Here the settings are a plain table and the blend is
//! [`blend_settings`] — a first-order interp, the same one the lag uses.
//!
//! # The camera is not sim state, and it never writes back
//!
//! Ruling 4: `ViewMode` is camera-side only and never crosses the sim wire. This
//! module keeps that literal — [`LocomotionCamera`] is owned by each *host*, is
//! never a component, never a resource and never serialized, and every input it
//! takes is **read** from the movement runtime. The aim yaw a
//! [`crate::components::RotationMode::Aiming`] character turns to face is
//! integrated by the movement step from the look axes, which is the one movement
//! door; the camera reads the same number afterwards.
//!
//! **How that is asserted, precisely** (corrected by the P29.6 audit). The
//! camera is stepped unconditionally at the end of both hosts' fixed steps, so
//! "with and without a camera" is not a comparison either host can make. What
//! the arms compare is **perturbation**: `inf_physics`'s
//! `stepping_a_camera_changes_nothing_about_the_simulation` runs the identical
//! step with and without the final `step_locomotion_camera` line and compares
//! the world *and* the physics bridge, and `phase29_gate` drives the whole
//! course while telling the camera different things (view mode, shoulder, a
//! different tuning table) and requires the sim trace not to move.
//!
//! # Portable math
//!
//! Adds, multiplies, comparisons and `inf_math`'s `p*` family. The camera does
//! not reach `state_bytes`, but the gate asserts a **deterministic camera trace**,
//! and a claim that only holds on one target is a claim about this machine.

use glam::DVec3;

use crate::components::{Gait, MovementMode, RotationMode};
use crate::math::{Vec2d, Vec3d};

/// Which eye the camera is at. **Camera-side only** (Ruling 4) — it is not a
/// component, it does not serialize, and no sim system may read it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewMode {
    #[default]
    ThirdPerson,
    FirstPerson,
}

/// One settings block — ALS's `FALSCameraSettings`, with the two fields its C++
/// path reads out of curves instead (the pivot offset and the three lag speeds)
/// promoted to real fields, because that is what removes the dummy AnimBP.
#[derive(
    bevy_reflect::Reflect, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[serde(default)]
pub struct CameraSettings {
    /// Distance from the pivot to the camera, metres (ALS `TargetArmLength`,
    /// 340 cm at a run).
    pub arm_length_m: f64,
    /// Offset applied to the smoothed pivot, in the **pivot's** frame (right, up,
    /// forward), metres.
    pub pivot_offset: Vec3d,
    /// Offset applied to the camera, in the **camera rotation's** frame (right,
    /// up, forward), metres — ALS's shoulder offset lives here.
    pub camera_offset: Vec3d,
    /// The three axis-independent lag speeds, in camera-yaw space:
    /// `x` sideways, `y` vertical, `z` forward/back.
    pub lag_speeds: Vec3d,
    /// How fast the camera rotation chases the aim, 1/s.
    pub rotation_lag: f64,
    /// Vertical field of view, degrees.
    pub fov_deg: f64,
}

impl Default for CameraSettings {
    fn default() -> Self {
        // ALS's `CameraBehavior` defaults for the third-person running state,
        // converted once (IM-1): arm 340 cm, pivot offset (0, 0, 0), camera
        // offset right 45 cm / up 8 cm.
        Self {
            arm_length_m: 3.4,
            pivot_offset: Vec3d::new(0.0, 0.0, 0.0),
            camera_offset: Vec3d::new(0.45, 0.08, 0.0),
            lag_speeds: Vec3d::new(10.0, 4.0, 8.0),
            rotation_lag: 10.0,
            fov_deg: 70.0,
        }
    }
}

impl CameraSettings {
    /// Componentwise first-order interp toward `target` — [`blend_settings`]'s
    /// per-field rule, exposed so a caller can blend one block.
    pub fn interp(self, target: Self, speed: f64, dt: f64) -> Self {
        let f = |a: f64, b: f64| interp_to(a, b, speed, dt);
        let v = |a: Vec3d, b: Vec3d| Vec3d::new(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z));
        Self {
            arm_length_m: f(self.arm_length_m, target.arm_length_m),
            pivot_offset: v(self.pivot_offset, target.pivot_offset),
            camera_offset: v(self.camera_offset, target.camera_offset),
            lag_speeds: v(self.lag_speeds, target.lag_speeds),
            rotation_lag: f(self.rotation_lag, target.rotation_lag),
            fov_deg: f(self.fov_deg, target.fov_deg),
        }
    }
}

/// The four blocks a rotation mode carries — ALS's `FALSCameraGaitSettings`.
///
/// A `camera.toml` may name only the numbers an author is actually tuning; the
/// rest come from [`CameraTuning::default`], which is the ported ALS table. That
/// is delivered by [`CameraTuning::from_toml`]'s **merge onto the serialized
/// default** and NOT by `#[serde(default)]`, which fills a missing field from
/// the field type's own default and would hand this block the third-person run
/// numbers (P29.6 audit, A7). Deserialize a camera table through that door.
#[derive(
    bevy_reflect::Reflect,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(default)]
pub struct GaitCameraSettings {
    pub walk: CameraSettings,
    pub run: CameraSettings,
    pub sprint: CameraSettings,
    /// Every low stance — crouch, prone, slide, roll — shares one block, because
    /// what the camera cares about is that the character is *near the ground*.
    pub crouch: CameraSettings,
}

impl GaitCameraSettings {
    fn pick(&self, gait: Gait, low: bool) -> CameraSettings {
        if low {
            return self.crouch;
        }
        match gait {
            Gait::Walk => self.walk,
            Gait::Sprint => self.sprint,
            // Run, and the two reserved tiers a newer build could send: a camera
            // block is a look, not a contract, so an unknown tier gets the middle
            // one rather than a refusal (the mode table is where a reserved
            // variant is refused BY NAME).
            _ => self.run,
        }
    }
}

/// **How the boom answers geometry** (wave CHAR1c) — the collision policy, as
/// numbers an author owns.
///
/// # What P29.6 shipped, and the four things it did not
///
/// The sweep is one sphere cast from the pivot to where the camera wants to be,
/// and on a blocking hit the camera goes to the contact — ALS's own
/// `SweepSingleByChannel` plus `TargetCameraLocation += HitResult.Location -
/// HitResult.TraceEnd` (`ALSPlayerCameraManager.cpp:200-220`). That is a
/// *reaction*, and it has four failure modes the user reported in one sentence
/// ("there should be no camera clipping when the user is looking around ... the
/// camera should simply adjust position, just like any AAA game"):
///
/// 1. It only sees what is **directly behind** the camera. A wall arriving from
///    the side is not in the ray until the camera is already in it.
/// 2. It **snaps**, in both directions. ALS has no smoothing on the arm at all,
///    so a lamp-post crossing the boom pops the camera in and out in two frames.
/// 3. It stops at **any** collider, including other characters — so a crowd
///    walking behind the hero shoves the camera onto its neck.
/// 4. When the arm does go short the body fills the frame, and the camera ends
///    up **inside the hero's head** (carried 89: the idle frame's camera sat in
///    the back of the hero's neck, photographed four times across CHAR1a).
///
/// Each field below closes exactly one of those.
#[derive(
    bevy_reflect::Reflect, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[serde(default)]
pub struct CameraCollision {
    /// Cast the **whisker fan** as well as the main sweep (1).
    ///
    /// Off restores P29.6 exactly: one cast, no steer, no predictive shortening.
    /// It is a switch rather than a `whisker_count = 0` because "this rig does
    /// not want whiskers" is a decision an author makes, and a count of zero
    /// reads like a mistake.
    pub whiskers: bool,
    /// How many whiskers **per side** — so the fan is `2 * count + 1` casts
    /// including the main one.
    pub whisker_count: u32,
    /// The outermost whisker's angle from the boom, degrees. The inner ones are
    /// spaced evenly to it.
    ///
    /// **22.5° is chosen against a corridor, not against a taste.** A whisker at
    /// angle θ meets a wall a half-width `w` to the side at `w / sin θ`, and it
    /// bounds the boom at `w · cos θ / sin θ`. At 22.5° a 2 m half-width alley
    /// bounds the boom at 4.83 m, which is longer than every arm in the shipped
    /// table (3.0 / 3.4 / 4.0 / 2.5 / 2.0), so an alley does not shorten the
    /// boom at all — while a wall the boom is about to swing into is seen a
    /// quarter of a turn early.
    pub whisker_spread_deg: f64,
    /// How hard a blocked whisker **steers** the boom away from it, as a
    /// fraction of that whisker's own angle. `1.0` means a fully blocked
    /// outermost whisker asks for the whole spread; `0` disables the steer and
    /// leaves only the predictive shortening.
    pub whisker_steer: f64,
    /// The most the steer may reach, degrees — a ceiling, so a character in a
    /// corner whose whiskers are all blocked cannot be spun round by the
    /// arithmetic.
    pub whisker_steer_max_deg: f64,
    /// How fast the **whisker steer** swings toward a blocked side, 1/s (2).
    ///
    /// **Not the boom's own pull-in, and the difference is measured.** The boom
    /// does not ease in at all — see [`LocomotionCamera::resolve_swept`], where
    /// a smoothed pull-in was measured at 33 frames of 840 with the camera
    /// inside a collider, because any ease is a lag and a lag on this quantity
    /// is the defect the wave is named after. The steer is a different case: it
    /// moves the camera *sideways* out of the way rather than out of a wall, so
    /// a lag there is a comfort question and not a safety one.
    pub pull_in_speed: f64,
    /// How fast the boom goes back **out** once the way is clear, 1/s, and how
    /// fast the steer relaxes (2).
    ///
    /// Slow, and much slower than the pull-in **by construction** — the
    /// asymmetry is the whole point. A boom that returned as fast as it
    /// retracts pumps once per lamp-post down a street.
    pub return_speed: f64,
    /// The boom length, metres, at which the subject's own body starts to fade
    /// (4). Above it the body is drawn exactly as it always was.
    pub near_fade_start_m: f64,
    /// ...and the length at which it is fully gone. Below `near_fade_start_m` by
    /// construction; the fade is linear between the two.
    pub near_fade_end_m: f64,
    /// Exclude **characters** from the sweep and the whiskers (3) — the subject
    /// itself, every other character, and the vehicle a character is riding.
    ///
    /// A camera that stopped on a passer-by would be shoved onto its own
    /// subject's neck by a crowd, and this engine now has crowds. The cost of
    /// the other answer is a camera that sometimes sees through a body, which is
    /// what every third-person game alive chooses.
    pub ignore_characters: bool,
    /// The lowest the camera's own pitch may go, degrees (camera-side).
    ///
    /// ALS clamps the *view* pitch in `APlayerCameraManager::ViewPitchMin/Max`
    /// and never touches the character. So does this: the movement step keeps
    /// its own hard `±89°` bound on `aim_pitch_deg` (which is sim state and is
    /// what the body and the look-at read), and this pair clamps only the pose
    /// the camera publishes.
    pub pitch_min_deg: f64,
    /// ...and the highest.
    pub pitch_max_deg: f64,
}

impl Default for CameraCollision {
    fn default() -> Self {
        Self {
            whiskers: true,
            whisker_count: 2,
            whisker_spread_deg: 22.5,
            whisker_steer: 1.0,
            whisker_steer_max_deg: 25.0,
            // `FInterpTo` closes ~63 % of the gap in `1 / speed` seconds: 33 ms
            // to come in, 400 ms to go back out.
            pull_in_speed: 30.0,
            return_speed: 2.5,
            near_fade_start_m: 0.90,
            near_fade_end_m: 0.35,
            ignore_characters: true,
            // The SIM's own hard bound, so the shipped default changes no
            // frame this engine has ever drawn; a rig that wants a tighter
            // ceiling (a cover camera, a vehicle) narrows it.
            pitch_min_deg: -89.0,
            pitch_max_deg: 89.0,
        }
    }
}

impl CameraCollision {
    /// The **near fade** for a boom of `arm_m`: `1.0` fully drawn, `0.0` fully
    /// gone, linear between [`near_fade_end_m`](Self::near_fade_end_m) and
    /// [`near_fade_start_m`](Self::near_fade_start_m).
    ///
    /// Answers `1.0` for a degenerate band (`start <= end`), because a band an
    /// author has inverted must not hide the character — a refusal is a value,
    /// and the value that keeps the game playable is "draw it".
    pub fn near_fade(&self, arm_m: f64) -> f64 {
        let (a, b) = (self.near_fade_end_m, self.near_fade_start_m);
        if !(a.is_finite() && b.is_finite()) || b <= a {
            return 1.0;
        }
        ((arm_m - a) / (b - a)).clamp(0.0, 1.0)
    }

    /// The whisker angles, degrees, signed, **left to right and never zero** —
    /// the main sweep is the boom itself and is not a whisker.
    ///
    /// Empty when the fan is off or the count is zero, which is what makes the
    /// physics half's loop the same shape in both cases.
    pub fn whisker_angles_deg(&self) -> Vec<f64> {
        if !self.whiskers || self.whisker_count == 0 || !self.whisker_spread_deg.is_finite() {
            return Vec::new();
        }
        let n = i64::from(self.whisker_count.min(8));
        let mut out = Vec::with_capacity(2 * n as usize);
        for k in (1..=n).rev() {
            out.push(-self.whisker_spread_deg * (k as f64) / (n as f64));
        }
        for k in 1..=n {
            out.push(self.whisker_spread_deg * (k as f64) / (n as f64));
        }
        out
    }

    /// **The steer one whisker asks for**, degrees, from its angle and how much
    /// of its own reach was free.
    ///
    /// A whisker blocked at the very start asks for the whole of its own angle,
    /// away from itself; one that reached the end asks for nothing. Split out so
    /// the rule is a function of two numbers and can be read without a world.
    pub fn whisker_steer_deg(&self, angle_deg: f64, free_fraction: f64) -> f64 {
        let blocked = (1.0 - free_fraction).clamp(0.0, 1.0);
        -angle_deg * blocked * self.whisker_steer
    }
}

/// **The drive camera's block** (island wave VEH2a) — one settings block plus
/// the handful of numbers that make a car's camera a car's camera.
///
/// # Why a car needed a branch at all
///
/// P29.7 gave the engine a vehicle and never gave it a camera. `settings_for`
/// dispatches on `RotationMode` and `Gait`, `step_driving` writes neither — it
/// returns before `actual_gait` is ever computed — so a driving character got
/// **whichever on-foot gait block was latched at the instant it pressed the
/// interact key, frozen for the whole drive**. Since a player is usually
/// stationary beside the door when they press it, that block is `walk`: arm
/// 3.0 m, FOV 70, shoulder offset 0.45 m. Literally the walking camera, and it
/// could not react to the car accelerating from nothing to thirty metres a
/// second because nothing it reads changes while driving.
///
/// Everything here is per-*camera*, not per-vehicle-class, and the one number a
/// class would want — how far back to sit for a bus against a hatchback — is
/// [`arm_per_length_m`](Self::arm_per_length_m) against the chassis's own
/// half-length. Geometry the rig already carries, so the fleet grows without the
/// camera table growing with it.
///
/// Reference: `docs/reference_videos/frames/driving/0032` and `steal-car/0040` —
/// the camera sits a little above roof height, five to six metres back, pitched
/// down about ten degrees, with the car in the lower third and the horizon near
/// the top of the frame.
#[derive(
    bevy_reflect::Reflect, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[serde(default)]
pub struct DrivingCameraSettings {
    /// The block at a standstill: arm, offsets, lag, FOV.
    pub base: CameraSettings,
    /// Extra arm length per metre of the vehicle's **half-length**, so a van
    /// sits further back than a hatchback without a per-class table.
    pub arm_per_length_m: f64,
    /// Extra arm length per m/s of speed — the camera easing back as the car
    /// gets going.
    pub arm_per_speed_s: f64,
    /// Extra vertical FOV per m/s, degrees. The single cheapest speed cue there
    /// is, and the reason a car at 40 m/s feels different from one at 10 in a
    /// still frame.
    pub fov_per_speed_deg_s: f64,
    /// The most FOV the speed term may add, degrees — a ceiling, because a field
    /// of view that keeps widening ends as a fisheye.
    pub fov_gain_max_deg: f64,
    /// How many seconds of velocity the pivot is pushed along, metres per (m/s).
    /// This is the look-ahead: at 30 m/s and 0.35 s the camera is aimed ten
    /// metres up the road rather than at the roof.
    pub look_ahead_s: f64,
    /// The speed, m/s, at which the camera is fully aligned with the vehicle's
    /// heading rather than the driver's aim. Below it the two are blended, so a
    /// driver parking can still look around and a driver at speed gets a camera
    /// that follows the car.
    pub align_speed_mps: f64,
}

impl Default for DrivingCameraSettings {
    fn default() -> Self {
        Self {
            base: CameraSettings {
                // Roof height and a little more, five metres back, and a boom
                // that lags: a car is heavier than a person and its camera
                // should feel it.
                arm_length_m: 5.0,
                pivot_offset: Vec3d::new(0.0, 0.55, 0.0),
                camera_offset: Vec3d::new(0.0, 0.0, 0.0),
                lag_speeds: Vec3d::new(6.0, 5.0, 3.5),
                rotation_lag: 4.0,
                fov_deg: 72.0,
            },
            arm_per_length_m: 0.55,
            arm_per_speed_s: 0.045,
            fov_per_speed_deg_s: 0.42,
            fov_gain_max_deg: 14.0,
            look_ahead_s: 0.35,
            align_speed_mps: 9.0,
        }
    }
}

/// What the camera needs to know about the car its subject is driving.
///
/// Everything in it is already written by `inf_physics::d3::movement::
/// step_driving` onto the character (the chassis heading as `body_yaw_deg`, the
/// chassis velocity as `velocity`) except the half-length, which the camera door
/// reads off the chassis collider. Nothing here is new simulation state, which
/// is why the drive camera costs no schema and no trace bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DrivingView {
    /// The chassis's heading, degrees about `+Y`.
    pub chassis_yaw_deg: f64,
    /// The chassis's velocity, m/s, world space.
    pub velocity: Vec3d,
    /// Half the chassis's length along its forward axis, metres.
    pub half_length_m: f64,
}

/// **The whole tunable set**, and the thing P29.5's live-tuning door edits.
///
/// The table is ALS's `FALSCameraStateSettings` — `RotationMode` × (gait +
/// crouch) = 3 × 4 = twelve blocks — plus the first-person seat and the handful
/// of numbers that are not per-state.
#[derive(
    bevy_reflect::Reflect, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[serde(default)]
pub struct CameraTuning {
    pub velocity_direction: GaitCameraSettings,
    pub looking_direction: GaitCameraSettings,
    pub aiming: GaitCameraSettings,
    /// The first-person **seat**: the arm length is ignored (the camera is at the
    /// pivot), and the offsets place the eye.
    pub first_person: CameraSettings,
    /// **The drive camera** (island wave VEH2a) — the block a character in
    /// [`MovementMode::Driving`] gets, whatever gait it happened to enter the
    /// car at.
    pub driving: DrivingCameraSettings,
    /// Where the pivot sits on the character, as a fraction of its standing
    /// height. ALS uses the midpoint of the `head` and `root` sockets, which is
    /// this number with the rig's own proportions behind it; deriving it from the
    /// capsule instead means a 1.2 m character gets a proportionate camera
    /// without anybody authoring a second table.
    pub pivot_height_ratio: f64,
    /// Radius of the collision sphere the camera sweeps with, metres (ALS 15 cm).
    pub collision_radius_m: f64,
    /// How fast a settings change blends, 1/s. ALS spends a whole AnimBP on this.
    pub state_blend_speed: f64,
    /// How fast the first-person weight moves, 1/s.
    pub view_blend_speed: f64,
    /// How far behind the character the camera may end up when the sweep is
    /// blocked, as a fraction of the arm — a floor, so a wall does not put the
    /// camera inside the character's head.
    pub min_arm_fraction: f64,
    /// **The collision policy** (wave CHAR1c): whiskers, the asymmetric arm
    /// smoothing, the near fade, the character-ignore rule and the camera's own
    /// pitch limits. Every number in it is a number an author owns, and the
    /// whole block defaults, so a `camera.toml` that names none of them gets the
    /// shipped policy.
    pub collision: CameraCollision,
}

impl Default for CameraTuning {
    fn default() -> Self {
        let base = CameraSettings::default();
        let walk = CameraSettings {
            arm_length_m: 3.0,
            lag_speeds: Vec3d::new(10.0, 4.0, 6.0),
            ..base
        };
        let run = base;
        let sprint = CameraSettings {
            arm_length_m: 4.0,
            lag_speeds: Vec3d::new(6.0, 4.0, 4.0),
            fov_deg: 78.0,
            ..base
        };
        let crouch = CameraSettings {
            arm_length_m: 2.5,
            camera_offset: Vec3d::new(0.45, 0.0, 0.0),
            ..base
        };
        let third = GaitCameraSettings {
            walk,
            run,
            sprint,
            crouch,
        };
        // Aiming pulls in and slows down — ALS's aim block, and the reason a
        // sprint is refused while aiming (`CanSprint`).
        let aim_block = CameraSettings {
            arm_length_m: 2.0,
            camera_offset: Vec3d::new(0.55, 0.05, 0.0),
            lag_speeds: Vec3d::new(20.0, 20.0, 20.0),
            rotation_lag: 20.0,
            fov_deg: 55.0,
            ..base
        };
        Self {
            velocity_direction: third,
            looking_direction: third,
            aiming: GaitCameraSettings {
                walk: aim_block,
                run: aim_block,
                sprint: aim_block,
                crouch: aim_block,
            },
            first_person: CameraSettings {
                arm_length_m: 0.0,
                pivot_offset: Vec3d::new(0.0, 0.0, 0.0),
                camera_offset: Vec3d::new(0.0, 0.10, 0.12),
                lag_speeds: Vec3d::new(30.0, 30.0, 30.0),
                rotation_lag: 30.0,
                fov_deg: 90.0,
            },
            driving: DrivingCameraSettings::default(),
            pivot_height_ratio: 0.80,
            collision_radius_m: 0.15,
            state_blend_speed: 6.0,
            view_blend_speed: 8.0,
            min_arm_fraction: 0.05,
            collision: CameraCollision::default(),
        }
    }
}

impl CameraTuning {
    /// The block a `(rotation mode, gait, mode)` selects — ALS's two-level
    /// dispatch, with `Stance` folded into `MovementMode` per Ruling 4.
    pub fn settings_for(
        &self,
        rotation_mode: RotationMode,
        gait: Gait,
        mode: MovementMode,
    ) -> CameraSettings {
        // **Driving is answered before the gait is even looked at**, and that is
        // the point rather than a shortcut: `step_driving` returns before
        // `actual_gait` is computed, so the value in it during a drive is
        // whatever was latched at the moment the character got in. Reading it
        // would be reading a stale field, so this branch does not.
        if mode == MovementMode::Driving {
            return self.driving.base;
        }
        let low = matches!(
            mode,
            MovementMode::Crouch
                | MovementMode::Prone
                | MovementMode::Slide
                | MovementMode::Roll
                | MovementMode::Dive
        );
        match rotation_mode {
            RotationMode::VelocityDirection => self.velocity_direction.pick(gait, low),
            RotationMode::Aiming => self.aiming.pick(gait, low),
            // LookingDirection, and the two reserved slots — see `pick`.
            _ => self.looking_direction.pick(gait, low),
        }
    }

    /// Set one tunable **by name** — the door P29.5's live tuning reaches the
    /// camera through, since a camera is not a reflected component and this wave
    /// has no schema budget to make it one.
    ///
    /// Returns whether the name was known. A refusal is a **value**: a tuning UI
    /// is a live surface over a running session, and taking one down over a stale
    /// field name is the wrong trade (`inf_editor_core::tuning`'s own rule).
    ///
    /// Names are `<block>.<field>` where `<block>` is `walk` / `run` / `sprint` /
    /// `crouch` / `aim` / `first_person` and `<field>` one of `arm_length_m`,
    /// `rotation_lag`, `fov_deg`, `lag_x` / `lag_y` / `lag_z`,
    /// `offset_x` / `offset_y` / `offset_z` (the camera offset), or the five
    /// table-wide names `pivot_height_ratio`, `collision_radius_m`,
    /// `state_blend_speed`, `view_blend_speed`, `min_arm_fraction`.
    ///
    /// Finiteness is the only check. A **range** is not one: `arm_length_m` may
    /// be set negative (which puts the camera in front of the character) and
    /// `collision_radius_m` arbitrarily large. That is deliberate for a live
    /// tuning surface — an author sweeping a slider through zero must not have
    /// the door start refusing — and it is written down here rather than
    /// discovered (P29.6 audit).
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match name {
            "pivot_height_ratio" => {
                self.pivot_height_ratio = value;
                return true;
            }
            "collision_radius_m" => {
                self.collision_radius_m = value;
                return true;
            }
            "state_blend_speed" => {
                self.state_blend_speed = value;
                return true;
            }
            "view_blend_speed" => {
                self.view_blend_speed = value;
                return true;
            }
            "min_arm_fraction" => {
                self.min_arm_fraction = value;
                return true;
            }
            // ── the collision policy (wave CHAR1c) ──
            //
            // Under a `collision.` prefix rather than bare names, because the
            // block is a *policy* and an author sweeping "the pull-in" wants to
            // find it beside "the return" rather than among the twelve gait
            // blocks. `whiskers` is a bool on a `f64` door, so it reads the sign
            // — the same rule the live tuner uses for every other flag, and the
            // reason this door takes one scalar type at all.
            "collision.whiskers" => {
                self.collision.whiskers = value > 0.5;
                return true;
            }
            "collision.whisker_count" => {
                self.collision.whisker_count = value.clamp(0.0, 8.0) as u32;
                return true;
            }
            "collision.whisker_spread_deg" => {
                self.collision.whisker_spread_deg = value;
                return true;
            }
            "collision.whisker_steer" => {
                self.collision.whisker_steer = value;
                return true;
            }
            "collision.whisker_steer_max_deg" => {
                self.collision.whisker_steer_max_deg = value;
                return true;
            }
            "collision.pull_in_speed" => {
                self.collision.pull_in_speed = value;
                return true;
            }
            "collision.return_speed" => {
                self.collision.return_speed = value;
                return true;
            }
            "collision.near_fade_start_m" => {
                self.collision.near_fade_start_m = value;
                return true;
            }
            "collision.near_fade_end_m" => {
                self.collision.near_fade_end_m = value;
                return true;
            }
            "collision.ignore_characters" => {
                self.collision.ignore_characters = value > 0.5;
                return true;
            }
            "collision.pitch_min_deg" => {
                self.collision.pitch_min_deg = value;
                return true;
            }
            "collision.pitch_max_deg" => {
                self.collision.pitch_max_deg = value;
                return true;
            }
            // The drive camera's own scalars (island wave VEH2a). Under the
            // `drive.` prefix so `drive.arm_length_m` — the block's base — still
            // goes through the shared field table below.
            "drive.arm_per_length_m" => {
                self.driving.arm_per_length_m = value;
                return true;
            }
            "drive.arm_per_speed_s" => {
                self.driving.arm_per_speed_s = value;
                return true;
            }
            "drive.fov_per_speed_deg_s" => {
                self.driving.fov_per_speed_deg_s = value;
                return true;
            }
            "drive.fov_gain_max_deg" => {
                self.driving.fov_gain_max_deg = value;
                return true;
            }
            "drive.look_ahead_s" => {
                self.driving.look_ahead_s = value;
                return true;
            }
            "drive.align_speed_mps" => {
                self.driving.align_speed_mps = value;
                return true;
            }
            _ => {}
        }
        let Some((block, field)) = name.split_once('.') else {
            return false;
        };
        // A gait block is edited across all three rotation modes at once, which
        // is what an author means by "the run camera" — the aim table is its own
        // block precisely because it is the one they mean separately.
        let mut targets: Vec<&mut CameraSettings> = Vec::new();
        match block {
            "walk" => targets.extend([
                &mut self.velocity_direction.walk,
                &mut self.looking_direction.walk,
            ]),
            "run" => targets.extend([
                &mut self.velocity_direction.run,
                &mut self.looking_direction.run,
            ]),
            "sprint" => targets.extend([
                &mut self.velocity_direction.sprint,
                &mut self.looking_direction.sprint,
            ]),
            "crouch" => targets.extend([
                &mut self.velocity_direction.crouch,
                &mut self.looking_direction.crouch,
            ]),
            "aim" => targets.extend([
                &mut self.aiming.walk,
                &mut self.aiming.run,
                &mut self.aiming.sprint,
                &mut self.aiming.crouch,
            ]),
            "first_person" => targets.push(&mut self.first_person),
            "drive" => targets.push(&mut self.driving.base),
            _ => return false,
        }
        let mut known = false;
        for s in targets {
            known = match field {
                "arm_length_m" => {
                    s.arm_length_m = value;
                    true
                }
                "rotation_lag" => {
                    s.rotation_lag = value;
                    true
                }
                "fov_deg" => {
                    s.fov_deg = value;
                    true
                }
                "lag_x" => {
                    s.lag_speeds.x = value;
                    true
                }
                "lag_y" => {
                    s.lag_speeds.y = value;
                    true
                }
                "lag_z" => {
                    s.lag_speeds.z = value;
                    true
                }
                "offset_x" => {
                    s.camera_offset.x = value;
                    true
                }
                "offset_y" => {
                    s.camera_offset.y = value;
                    true
                }
                "offset_z" => {
                    s.camera_offset.z = value;
                    true
                }
                _ => false,
            };
        }
        known
    }
}

impl CameraTuning {
    /// **Read one tunable by name** — [`set`](Self::set)'s twin (wave CHAR1c).
    ///
    /// A door that can only be written cannot be read back by the thing writing
    /// it, and a Blueprint that wants to nudge the boom has to know where it
    /// already is. The vocabulary is exactly `set`'s, and a name `set` accepts
    /// is a name this answers — which is what
    /// `every_name_the_camera_door_writes_can_be_read_back` holds.
    ///
    /// A gait block's value is read from the **velocity-direction** table, which
    /// is the one `set` writes first and keeps in step with `looking_direction`;
    /// the aim block is its own, exactly as it is on the way in.
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "pivot_height_ratio" => return Some(self.pivot_height_ratio),
            "collision_radius_m" => return Some(self.collision_radius_m),
            "state_blend_speed" => return Some(self.state_blend_speed),
            "view_blend_speed" => return Some(self.view_blend_speed),
            "min_arm_fraction" => return Some(self.min_arm_fraction),
            "drive.arm_per_length_m" => return Some(self.driving.arm_per_length_m),
            "drive.arm_per_speed_s" => return Some(self.driving.arm_per_speed_s),
            "drive.fov_per_speed_deg_s" => return Some(self.driving.fov_per_speed_deg_s),
            "drive.fov_gain_max_deg" => return Some(self.driving.fov_gain_max_deg),
            "drive.look_ahead_s" => return Some(self.driving.look_ahead_s),
            "drive.align_speed_mps" => return Some(self.driving.align_speed_mps),
            "collision.whiskers" => return Some(f64::from(u8::from(self.collision.whiskers))),
            "collision.whisker_count" => return Some(f64::from(self.collision.whisker_count)),
            "collision.whisker_spread_deg" => return Some(self.collision.whisker_spread_deg),
            "collision.whisker_steer" => return Some(self.collision.whisker_steer),
            "collision.whisker_steer_max_deg" => return Some(self.collision.whisker_steer_max_deg),
            "collision.pull_in_speed" => return Some(self.collision.pull_in_speed),
            "collision.return_speed" => return Some(self.collision.return_speed),
            "collision.near_fade_start_m" => return Some(self.collision.near_fade_start_m),
            "collision.near_fade_end_m" => return Some(self.collision.near_fade_end_m),
            "collision.ignore_characters" => {
                return Some(f64::from(u8::from(self.collision.ignore_characters)))
            }
            "collision.pitch_min_deg" => return Some(self.collision.pitch_min_deg),
            "collision.pitch_max_deg" => return Some(self.collision.pitch_max_deg),
            _ => {}
        }
        let (block, field) = name.split_once('.')?;
        let s = match block {
            "walk" => &self.velocity_direction.walk,
            "run" => &self.velocity_direction.run,
            "sprint" => &self.velocity_direction.sprint,
            "crouch" => &self.velocity_direction.crouch,
            "aim" => &self.aiming.walk,
            "first_person" => &self.first_person,
            "drive" => &self.driving.base,
            _ => return None,
        };
        match field {
            "arm_length_m" => Some(s.arm_length_m),
            "rotation_lag" => Some(s.rotation_lag),
            "fov_deg" => Some(s.fov_deg),
            "lag_x" => Some(s.lag_speeds.x),
            "lag_y" => Some(s.lag_speeds.y),
            "lag_z" => Some(s.lag_speeds.z),
            "offset_x" => Some(s.camera_offset.x),
            "offset_y" => Some(s.camera_offset.y),
            "offset_z" => Some(s.camera_offset.z),
            _ => None,
        }
    }

    /// Every name [`set`](Self::set) accepts, in a stable order — the vocabulary
    /// a tuning UI, a Blueprint's autocomplete and a gate all read.
    ///
    /// A `&'static [&'static str]` and not a built `Vec`, exactly as
    /// `VehicleTuning::names` is: the vocabulary is a property of the type and
    /// not of a call, and a list assembled at runtime is a list that can drift
    /// from the `match` beside it without anything noticing.
    pub fn names() -> &'static [&'static str] {
        &[
            "pivot_height_ratio",
            "collision_radius_m",
            "state_blend_speed",
            "view_blend_speed",
            "min_arm_fraction",
            "drive.arm_per_length_m",
            "drive.arm_per_speed_s",
            "drive.fov_per_speed_deg_s",
            "drive.fov_gain_max_deg",
            "drive.look_ahead_s",
            "drive.align_speed_mps",
            "collision.whiskers",
            "collision.whisker_count",
            "collision.whisker_spread_deg",
            "collision.whisker_steer",
            "collision.whisker_steer_max_deg",
            "collision.pull_in_speed",
            "collision.return_speed",
            "collision.near_fade_start_m",
            "collision.near_fade_end_m",
            "collision.ignore_characters",
            "collision.pitch_min_deg",
            "collision.pitch_max_deg",
            "walk.arm_length_m",
            "walk.rotation_lag",
            "walk.fov_deg",
            "walk.lag_x",
            "walk.lag_y",
            "walk.lag_z",
            "walk.offset_x",
            "walk.offset_y",
            "walk.offset_z",
            "run.arm_length_m",
            "run.rotation_lag",
            "run.fov_deg",
            "run.lag_x",
            "run.lag_y",
            "run.lag_z",
            "run.offset_x",
            "run.offset_y",
            "run.offset_z",
            "sprint.arm_length_m",
            "sprint.rotation_lag",
            "sprint.fov_deg",
            "sprint.lag_x",
            "sprint.lag_y",
            "sprint.lag_z",
            "sprint.offset_x",
            "sprint.offset_y",
            "sprint.offset_z",
            "crouch.arm_length_m",
            "crouch.rotation_lag",
            "crouch.fov_deg",
            "crouch.lag_x",
            "crouch.lag_y",
            "crouch.lag_z",
            "crouch.offset_x",
            "crouch.offset_y",
            "crouch.offset_z",
            "aim.arm_length_m",
            "aim.rotation_lag",
            "aim.fov_deg",
            "aim.lag_x",
            "aim.lag_y",
            "aim.lag_z",
            "aim.offset_x",
            "aim.offset_y",
            "aim.offset_z",
            "first_person.arm_length_m",
            "first_person.rotation_lag",
            "first_person.fov_deg",
            "first_person.lag_x",
            "first_person.lag_y",
            "first_person.lag_z",
            "first_person.offset_x",
            "first_person.offset_y",
            "first_person.offset_z",
            "drive.arm_length_m",
            "drive.rotation_lag",
            "drive.fov_deg",
            "drive.lag_x",
            "drive.lag_y",
            "drive.lag_z",
            "drive.offset_x",
            "drive.offset_y",
            "drive.offset_z",
        ]
    }

    /// **Read a `camera.toml` beside a level**, or the ported ALS defaults.
    ///
    /// The same shape (and the same rationale) as the input map's own
    /// `load_map_beside`: the camera table has no home in the scene schema —
    /// this wave has no schema budget and a camera is not sim state anyway — so
    /// it lives beside the level as text an author owns and a reviewer can read.
    /// Every field defaults, so a file naming one number is a legal file.
    ///
    /// A malformed file is a named refusal rather than a silent default — the
    /// host decides what to do with it (`inf_player::input::load_camera_beside`
    /// warns and carries on, because a camera that would not load is not a
    /// reason to refuse to open a level).
    ///
    /// # Why a MERGE and not `toml::from_str`
    ///
    /// (P29.6 audit, A7.) `#[serde(default)]` fills a missing field from the
    /// **field type's** `Default`, and the ALS table is a property of the whole
    /// `CameraTuning`, not of a `CameraSettings` on its own. So a file naming
    /// one number inside one block used to take that block's *other* numbers
    /// from `CameraSettings::default()` — the third-person **run** block — which
    /// made `[first_person] fov_deg = 95` a first-person seat with a 3.4 m
    /// third-person arm on it, silently. The doc directly above promised the
    /// opposite.
    ///
    /// Folding the parsed document onto the serialized default, table by table,
    /// is that promise made literal at every depth: a key an author wrote wins,
    /// and every key they did not write is the ported ALS number.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let patch: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut base = toml::Value::try_from(Self::default()).map_err(|e| e.to_string())?;
        merge_tables(&mut base, &patch);
        base.try_into().map_err(|e: toml::de::Error| e.to_string())
    }

    /// The table as deterministic TOML — what the character wizard writes.
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| e.to_string())
    }
}

/// Overlay `patch` onto `base` in place: a key present in both, with a **table**
/// on each side, recurses; anything else the patch names replaces what was there.
///
/// The rule [`CameraTuning::from_toml`] is built on, and the reason it is a
/// merge over `toml::Value` rather than a serde attribute: serde can express
/// "default this field from its own type" and cannot express "default this field
/// from my parent's default", which is the only sentence that describes a
/// partial camera table honestly.
fn merge_tables(base: &mut toml::Value, patch: &toml::Value) {
    match (base, patch) {
        (toml::Value::Table(b), toml::Value::Table(p)) => {
            for (k, v) in p {
                match b.get_mut(k) {
                    Some(slot) => merge_tables(slot, v),
                    None => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, p) => *b = p.clone(),
    }
}

/// What the sim tells the camera each step. Every field is **read** from the
/// movement runtime — nothing here is the camera's to write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraInput {
    /// The pivot target in world space: a point on the character, not its origin.
    pub pivot_target: Vec3d,
    /// The aim the movement step integrated from the look axes, degrees.
    pub aim_yaw_deg: f64,
    /// …and its pitch, degrees, positive up.
    pub aim_pitch_deg: f64,
    pub rotation_mode: RotationMode,
    pub gait: Gait,
    pub mode: MovementMode,
    /// The car this character is driving, or `None` (island wave VEH2a). Read
    /// only when [`mode`](Self::mode) is [`MovementMode::Driving`].
    pub driving: Option<DrivingView>,
}

/// Where the camera ended up this step. The whole of what a renderer needs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraPose {
    pub position: Vec3d,
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    pub fov_deg: f64,
}

/// **Which stack a camera request belongs to** (wave CHAR1c).
///
/// Three layers, and the order between them is the whole of "who wins": a
/// cinematic beats a death, a death beats the gameplay rig. Priority *within* a
/// layer breaks ties between two sources of the same kind — two overlapping
/// sequencer tracks, a death cam and a photo mode.
///
/// It is an ordering and not a set of magic numbers because a number invites
/// arithmetic: `Scripted(0)` beating `Override(9999)` has to be true by the
/// type, or the first gameplay system in a hurry will pick 10000.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CameraLayer {
    /// The gameplay rig — the character's own boom, the vehicle's, an aim. The
    /// pose [`CameraDirector::resolve`] is handed, and the layer nothing has to
    /// request.
    #[default]
    Gameplay,
    /// Something has taken the camera off the player for a moment: death, a
    /// ragdoll follow, a photo mode. It outranks gameplay and loses to a script.
    Override,
    /// A sequencer shot, a cinematic, a scripted cut. Outranks everything: a
    /// cutscene that a stumble could steal the camera from is not a cutscene.
    Scripted,
}

/// **One claim on the camera** (wave CHAR1c).
///
/// A source pushes one of these per step it wants the camera; the step it stops
/// pushing is the step the director blends back to whatever is left. There is no
/// "release" verb for the same reason the movement intent has no "stop walking":
/// a per-step claim cannot be leaked by a system that panicked, and a latch can.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraRequest {
    pub layer: CameraLayer,
    /// Rank within the layer; higher wins. Ties break on [`tag`](Self::tag)
    /// ascending, so two sources that forgot to disagree still resolve the same
    /// way in both hosts.
    pub priority: i32,
    /// Who is asking. Stable across the steps one source holds the camera —
    /// **this is the identity the blend keys on**, so a source that changes its
    /// pose every step does not restart its own blend.
    pub tag: u64,
    /// Where it wants the camera.
    pub pose: CameraPose,
    /// How long to blend *into* it, seconds. `0` is a **cut** — instant, and
    /// reported as one so a renderer can flush what a cut invalidates
    /// (`inf_render::is_camera_cut` is the consumer that already exists).
    pub blend_s: f64,
}

impl CameraRequest {
    /// A request that cuts (no blend) — a sequencer shot change.
    pub fn cut(layer: CameraLayer, tag: u64, pose: CameraPose) -> Self {
        Self {
            layer,
            priority: 0,
            tag,
            pose,
            blend_s: 0.0,
        }
    }

    /// A request that blends in over `blend_s` seconds.
    pub fn blended(layer: CameraLayer, tag: u64, pose: CameraPose, blend_s: f64) -> Self {
        Self {
            layer,
            priority: 0,
            tag,
            pose,
            blend_s,
        }
    }
}

/// The tag the gameplay rig itself carries — the source that is always present
/// and never pushed.
pub const CAMERA_TAG_GAMEPLAY: u64 = 0;
/// The tag [`inf_physics`'s camera door](crate::camera) raises a ragdoll follow
/// under (wave CHAR1c). Named here so a host that wants to *outrank* the death
/// cam knows what it is outranking.
pub const CAMERA_TAG_RAGDOLL: u64 = 1;

/// **The priority-blended camera stack** (wave CHAR1c).
///
/// # The rule, in one paragraph
///
/// Every step, the gameplay rig produces a pose and any number of sources push a
/// [`CameraRequest`]. The winner is the highest `(layer, priority)`, ties broken
/// by `tag` ascending. When the winning **tag** changes, the director starts a
/// blend from the pose it was last outputting toward the new winner over that
/// request's `blend_s` — or cuts, if that is zero. The blend curve is
/// `smoothstep`, so a shot arrives and leaves with zero velocity and a
/// vehicle-entry blend does not jerk at either end.
///
/// # Why the blend is from the OUTPUT and not from the loser
///
/// A blend interrupted halfway must not snap. The director blends from the pose
/// it last *published*, which is where the viewer's eye already is, so a shot
/// cancelled mid-blend continues from wherever it had reached. That is one
/// field (`from`) instead of a stack of partial blends, and it is the same
/// choice the animation cross-fade makes for the same reason.
///
/// # It is deterministic, and that is a property this type has to carry
///
/// The camera is render-side, but a `char1c_gate` arm compares PIE and shipping
/// on the camera trace — so the director's output has to be a pure function of
/// (gameplay pose, requests, dt). It holds no clock, no map with a hashed
/// iteration order and no floating tie-break: the winner is chosen by a total
/// order over `(layer, priority, tag)` and the blend clock is `dt` accumulated.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CameraDirector {
    /// This step's claims. Pushed by hosts, drained by [`resolve`](Self::resolve).
    requests: Vec<CameraRequest>,
    /// Who is holding the camera, and what the resolved winner's pose was.
    active: Option<(CameraLayer, i32, u64)>,
    /// The pose the current blend started from.
    from: CameraPose,
    /// Seconds into the current blend, and how long it is. `len == 0` means
    /// "settled" — nothing is blending.
    elapsed_s: f64,
    len_s: f64,
    /// The pose the director published last step — the thing a new blend starts
    /// from.
    last: CameraPose,
    /// Whether the last [`resolve`](Self::resolve) was a **cut**.
    cut: bool,
    seeded: bool,
}

impl CameraDirector {
    /// Push one claim for this step. Cheap enough to call unconditionally; a
    /// step with no claims is a `Vec` that stays empty.
    pub fn request(&mut self, r: CameraRequest) {
        self.requests.push(r);
    }

    /// Who is holding the camera as of the last [`resolve`](Self::resolve) —
    /// `None` while the gameplay rig has it.
    pub fn holder(&self) -> Option<(CameraLayer, i32, u64)> {
        self.active
    }

    /// Whether the last resolve was a cut rather than a blend.
    pub fn was_cut(&self) -> bool {
        self.cut
    }

    /// How far through the current blend, `[0, 1]`; `1` when nothing is blending.
    pub fn blend_fraction(&self) -> f64 {
        if self.len_s <= 0.0 {
            return 1.0;
        }
        (self.elapsed_s / self.len_s).clamp(0.0, 1.0)
    }

    /// **Resolve this step.** `gameplay` is the rig's own pose; the requests
    /// pushed since the last call are consumed.
    pub fn resolve(&mut self, gameplay: CameraPose, dt: f64) -> CameraPose {
        // The winner, by a total order. `max_by_key` over a tuple is the whole
        // rule; `tag` is *ascending* on a tie, so it is negated rather than
        // sorted the other way, which keeps one comparison in one place.
        let winner = self
            .requests
            .iter()
            .max_by_key(|r| (r.layer, r.priority, std::cmp::Reverse(r.tag)))
            .copied();
        let (key, target, blend_s) = match winner {
            Some(r) => (Some((r.layer, r.priority, r.tag)), r.pose, r.blend_s),
            // The gameplay rig's own claim is implicit and blends back over the
            // *last winner's* length, which is what makes a shot's exit as smooth
            // as its entrance without a second number on the request.
            None => (None, gameplay, self.len_s.max(DEFAULT_RELEASE_BLEND_S)),
        };
        self.requests.clear();

        if !self.seeded {
            self.seeded = true;
            self.active = key;
            self.from = target;
            self.last = target;
            self.elapsed_s = 0.0;
            self.len_s = 0.0;
            self.cut = false;
            return target;
        }
        // A change of HOLDER starts a blend; a holder that moved its own pose
        // does not.
        let changed = key.map(|k| k.2) != self.active.map(|k| k.2);
        if changed {
            self.from = self.last;
            self.elapsed_s = 0.0;
            self.len_s = blend_s.max(0.0);
            self.cut = self.len_s <= 0.0;
            self.active = key;
        } else if dt.is_finite() && dt > 0.0 {
            self.cut = false;
            self.elapsed_s += dt;
        }
        let out = if self.len_s <= 0.0 {
            target
        } else {
            let t = (self.elapsed_s / self.len_s).clamp(0.0, 1.0);
            // `smoothstep`: zero velocity at both ends. Spelled out rather than
            // reached for, because `inf-ecs` has no easing module and one line is
            // not a module.
            let e = t * t * (3.0 - 2.0 * t);
            lerp_pose(self.from, target, e)
        };
        if self.len_s > 0.0 && self.elapsed_s >= self.len_s {
            self.len_s = 0.0;
            self.elapsed_s = 0.0;
        }
        self.last = out;
        out
    }
}

/// How long the director takes to hand the camera **back** to gameplay when a
/// source stops asking and never said how long its exit should be, seconds.
pub const DEFAULT_RELEASE_BLEND_S: f64 = 0.5;

/// Blend two camera poses. Position, pitch and FOV lerp; the yaw takes the short
/// way round, for the reason [`interp_angle_deg`] exists.
pub fn lerp_pose(a: CameraPose, b: CameraPose, t: f64) -> CameraPose {
    let f = |x: f64, y: f64| x + (y - x) * t;
    CameraPose {
        position: Vec3d::new(
            f(a.position.x, b.position.x),
            f(a.position.y, b.position.y),
            f(a.position.z, b.position.z),
        ),
        yaw_deg: crate::movement::wrap_deg(
            a.yaw_deg + crate::movement::angle_delta_deg(b.yaw_deg, a.yaw_deg) * t,
        ),
        pitch_deg: f(a.pitch_deg, b.pitch_deg),
        fov_deg: f(a.fov_deg, b.fov_deg),
    }
}

/// **The camera**, owned by a host and never by a world.
#[derive(Clone, Debug, PartialEq)]
pub struct LocomotionCamera {
    pub tuning: CameraTuning,
    /// First or third person. Camera-side only (Ruling 4).
    pub view_mode: ViewMode,
    /// Which shoulder the third-person camera looks over. Mirrors the offset's
    /// `x`, which is why it is a flag and not a second table.
    pub right_shoulder: bool,

    /// The **smoothed** pivot, world space.
    pub pivot: Vec3d,
    /// The smoothed camera rotation, degrees.
    pub yaw_deg: f64,
    /// …and pitch.
    pub pitch_deg: f64,
    /// The blended settings — the state machine ALS spends an AnimBP on.
    pub settings: CameraSettings,
    /// The first-person weight, `[0, 1]`.
    pub fp_weight: f64,
    /// The **desired** camera position before collision, world space.
    pub desired: Vec3d,
    /// The resolved pose, after the sweep.
    pub pose: CameraPose,
    /// How far the sweep pulled the camera in this step, metres. `0` when
    /// nothing was in the way — the number a gate can assert a collision on.
    pub collision_pull_m: f64,
    /// **The smoothed boom length**, metres, from
    /// [`sweep_origin`](Self::sweep_origin) to the resolved camera (wave
    /// CHAR1c). This is the quantity the asymmetric smoother owns: the sweep
    /// answers a *free* length every step and this is what the camera actually
    /// sits at, which is why the near fade reads it and not the free length.
    pub arm_m: f64,
    /// What the sweep asked for this step, before the smoothing — the free
    /// length, metres. Reported so a gate (and a live tuner) can see the
    /// asymmetry as two numbers rather than infer it from one.
    pub target_arm_m: f64,
    /// **How much the world is taking off the boom**, metres, smoothed — the
    /// quantity the asymmetric law actually owns (wave CHAR1c). `0` in an open
    /// field, and identical to [`collision_pull_m`](Self::collision_pull_m).
    pub clip_m: f64,
    /// How far the whisker fan steered the boom this step, degrees, smoothed.
    /// `0` when the fan is off or nothing was blocked.
    pub whisker_steer_deg: f64,
    /// **How much of the subject's own body to draw**, `[0, 1]` — `1` all of it,
    /// `0` none (wave CHAR1c). A pure function of [`arm_m`](Self::arm_m) through
    /// [`CameraCollision::near_fade`]; the hosts project it onto the subject's
    /// skinned instance and nothing else reads it.
    pub subject_fade: f64,
    /// **The director** (wave CHAR1c) — the priority-blended stack the resolved
    /// [`pose`](Self::pose) comes out of. With no request on it the gameplay rig
    /// is the only source and the director is the identity.
    pub director: CameraDirector,
    /// **The rig's own pose**, before the director (wave CHAR1c). `pose` is what
    /// the renderer uses; this is what the gameplay rig asked for, kept so a
    /// gate can measure a blend against the thing it is blending away from.
    pub gameplay_pose: CameraPose,
    /// Whether the camera has taken its first frame. A camera that lerped from
    /// the origin on step one would swing across the level.
    pub seeded: bool,
    /// Whether the ARM has taken its first sweep. Separate from
    /// [`seeded`](Self::seeded) because `advance` seeds the smoothers and the
    /// sweep happens afterwards: a camera whose arm interpolated from zero would
    /// start every level inside its own subject, which is the failure this wave
    /// is named after.
    pub arm_seeded: bool,
}

impl Default for LocomotionCamera {
    fn default() -> Self {
        Self {
            tuning: CameraTuning::default(),
            view_mode: ViewMode::ThirdPerson,
            right_shoulder: true,
            pivot: Vec3d::ZERO,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            settings: CameraSettings::default(),
            fp_weight: 0.0,
            desired: Vec3d::ZERO,
            pose: CameraPose::default(),
            collision_pull_m: 0.0,
            arm_m: 0.0,
            target_arm_m: 0.0,
            clip_m: 0.0,
            whisker_steer_deg: 0.0,
            subject_fade: 1.0,
            director: CameraDirector::default(),
            gameplay_pose: CameraPose::default(),
            seeded: false,
            arm_seeded: false,
        }
    }
}

impl LocomotionCamera {
    /// **Everything except the sweep.** Advances the smoothers and computes the
    /// desired camera position; `inf_physics::d3::camera` calls this and then
    /// resolves the collision, because a swept sphere needs a physics world and
    /// nothing else here does.
    pub fn advance(&mut self, input: &CameraInput, dt: f64) {
        let mut target = if self.view_mode == ViewMode::FirstPerson {
            self.tuning.first_person
        } else {
            self.tuning
                .settings_for(input.rotation_mode, input.gait, input.mode)
        };
        let fp_target = if self.view_mode == ViewMode::FirstPerson {
            1.0
        } else {
            0.0
        };

        // ── the drive camera's speed terms (island wave VEH2a) ───────────────
        //
        // Folded into the TARGET rather than applied after the blend, so they
        // arrive through `state_blend_speed` like everything else and a car
        // accelerating hard does not snap its field of view. The pivot's
        // look-ahead is separate — it is a place, not a setting.
        let drive = match (input.mode, input.driving) {
            (MovementMode::Driving, Some(v)) if self.view_mode != ViewMode::FirstPerson => {
                let d = &self.tuning.driving;
                let speed = v.velocity.to_dvec3().length();
                target.arm_length_m +=
                    d.arm_per_length_m * v.half_length_m.max(0.0) + d.arm_per_speed_s * speed;
                target.fov_deg +=
                    (d.fov_per_speed_deg_s * speed).clamp(0.0, d.fov_gain_max_deg.max(0.0));
                Some((v, speed))
            }
            _ => None,
        };
        // **The camera's yaw target.** On foot it is the driver's aim and nothing
        // else. In a car it blends toward the chassis's own heading as the car
        // gets going, which is a *recentring* rather than an override: the stick
        // still moves `aim_yaw_deg`, so holding it looks around and releasing it
        // lets the camera swing back behind the car. That is the behaviour the
        // reference frames show and it costs no character state at all.
        let aim_yaw = match drive {
            Some((v, speed)) if self.tuning.driving.align_speed_mps > 0.0 => {
                let w = (speed / self.tuning.driving.align_speed_mps).clamp(0.0, 1.0);
                crate::movement::wrap_deg(
                    input.aim_yaw_deg
                        + w * crate::movement::angle_delta_deg(
                            v.chassis_yaw_deg,
                            input.aim_yaw_deg,
                        ),
                )
            }
            _ => input.aim_yaw_deg,
        };
        // …and the pivot is pushed along the car's velocity, so the camera is
        // aimed up the road rather than at the roof.
        let pivot_target = match drive {
            Some((v, _)) => Vec3d::from_dvec3(
                input.pivot_target.to_dvec3()
                    + v.velocity.to_dvec3() * self.tuning.driving.look_ahead_s,
            ),
            None => input.pivot_target,
        };

        if !self.seeded {
            // **Snap on the first frame.** Every smoother below starts from its
            // own last answer, and on step one there is no last answer — a
            // camera that interpolated from the origin would fly across the
            // level for half a second on every level load and every PIE start.
            // The same shape as the movement runtime's `seeded` latch, and it is
            // here for the same reason: a zero is not a measurement.
            self.seeded = true;
            self.settings = target;
            self.fp_weight = fp_target;
            self.pivot = pivot_target;
            self.yaw_deg = aim_yaw;
            self.pitch_deg = input.aim_pitch_deg;
        } else {
            self.settings = self
                .settings
                .interp(target, self.tuning.state_blend_speed, dt);
            self.fp_weight = interp_to(self.fp_weight, fp_target, self.tuning.view_blend_speed, dt)
                .clamp(0.0, 1.0);
            // Rotation lag: one speed chasing the aim, on the SHORT way round —
            // `interp_to` on raw degrees would take the long way through 359.
            // …and **wrapped**, which the first cut was not (P29.6 audit, A8).
            // `angle_delta_deg` answers a delta in `(-180, 180]`, so a camera
            // chasing an aim that wraps at ±180 accumulates the delta without
            // ever wrapping itself: one revolution is `yaw + 360`, ten are
            // `yaw + 3600`. That reaches `CameraPose::yaw_deg`, `trace_bytes`
            // and — worse — `basis`, whose `psin64`/`pcos64` range reduction is
            // measurably worse at large arguments (the P23 finding). The aim
            // this chases is wrapped at its own door (`wrap_deg` in the movement
            // step); this is the same rule on the same quantity.
            self.yaw_deg = crate::movement::wrap_deg(interp_angle_deg(
                self.yaw_deg,
                aim_yaw,
                self.settings.rotation_lag,
                dt,
            ));
            self.pitch_deg = interp_to(
                self.pitch_deg,
                input.aim_pitch_deg,
                self.settings.rotation_lag,
                dt,
            );
            // **The headline** — three interp speeds resolved in camera-yaw
            // space, so the camera can trail hard behind on forward/back, stay
            // tight sideways so a strafe does not swing the frame, and be softer
            // again vertically so stairs do not bounce the view. (The shipped
            // table's LARGEST speed is the sideways one, which is what "tight"
            // means — the first cut's comment had the two the wrong way round
            // against its own numbers.)
            self.pivot = axis_independent_lag(
                self.pivot,
                pivot_target,
                self.yaw_deg,
                self.settings.lag_speeds,
                dt,
            );
        }

        // **The camera's own pitch limits** (wave CHAR1c). ALS clamps the VIEW
        // pitch in `APlayerCameraManager::ViewPitchMin/Max` and never touches
        // the character; so does this. The sim's `aim_pitch_deg` keeps its own
        // hard ±89° bound in the movement step — which is what the body and the
        // look-at read — and a rig that wants a tighter ceiling narrows only
        // what the camera publishes.
        let (lo, hi) = (
            self.tuning.collision.pitch_min_deg,
            self.tuning.collision.pitch_max_deg,
        );
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            self.pitch_deg = self.pitch_deg.clamp(lo, hi);
        }

        // **The desired position, through the one door** (wave CHAR1c). It used
        // to be spelled inline here and nowhere else; the sweep now needs to ask
        // the same question with a steered yaw, and two spellings of "where the
        // camera wants to be" is exactly the seam a whisker fan would drift
        // through. The steer is LAST step's, which is one frame of latency and
        // is stated rather than hidden: the fan is cast against this frame's
        // desired position, so the answer it produces is necessarily next
        // frame's.
        self.desired = self.camera_at(self.arm_full(), self.whisker_steer_deg);
        self.pose = CameraPose {
            position: self.desired,
            yaw_deg: self.yaw_deg,
            pitch_deg: self.pitch_deg,
            fov_deg: self.settings.fov_deg,
        };
        self.gameplay_pose = self.pose;
        self.collision_pull_m = 0.0;
    }

    /// The boom length the settings ask for, with the first-person weight folded
    /// in — `0` at a first-person seat, because the camera is AT the pivot there.
    pub fn arm_full(&self) -> f64 {
        self.settings.arm_length_m * (1.0 - self.fp_weight)
    }

    /// **Where the camera sits** for a boom of `arm` metres and a yaw steered
    /// `steer_deg` off the camera's own — the one place the offsets are applied.
    pub fn camera_at(&self, arm: f64, steer_deg: f64) -> Vec3d {
        let shoulder: f64 = if self.right_shoulder { 1.0 } else { -1.0 };
        // The pivot's own frame is the CAMERA's yaw (pitch and roll zeroed) —
        // ALS's `CalculateAxisIndependentLag` convention, kept for the offset so
        // the two cannot disagree about what "right" means.
        let pivot = self.sweep_origin().to_dvec3();
        // The camera's own frame carries pitch, because the arm swings with it.
        let (cr, cu, cf) = basis(
            crate::movement::wrap_deg(self.yaw_deg + steer_deg),
            self.pitch_deg,
        );
        Vec3d::from_dvec3(
            pivot - cf * arm
                + cr * (self.settings.camera_offset.x * shoulder)
                + cu * self.settings.camera_offset.y
                + cf * self.settings.camera_offset.z,
        )
    }

    /// **Take the sweep's answer** (wave CHAR1c) — the asymmetric half of the
    /// collision model, and the door the physics side resolves through.
    ///
    /// `free_m` is how far along the pivot→desired ray the sphere cast (and the
    /// whisker fan's own predictive bound) says the camera may sit; `steer_deg`
    /// is what the fan asked the boom to swing by, raw. Both are smoothed here
    /// and nowhere else.
    ///
    /// # The asymmetry, and why it is not one speed
    ///
    /// Coming IN is an emergency and going OUT is a comfort. A boom that eased
    /// into a wall would be a camera inside the wall for the length of the ease;
    /// a boom that sprang back out the instant a lamp-post cleared would pump
    /// once per lamp-post down a street. The two speeds are
    /// [`CameraCollision::pull_in_speed`] and
    /// [`CameraCollision::return_speed`], and the arm arm asserts
    /// `pull_in < return` as a *time* — the wave's own measurement, not the
    /// table's claim about itself.
    pub fn resolve_swept(&mut self, free_m: f64, steer_deg: f64, dt: f64) {
        let c = self.tuning.collision;
        // ── the steer, on the same asymmetry ──
        let cap = c.whisker_steer_max_deg.abs();
        let want = if steer_deg.is_finite() {
            steer_deg.clamp(-cap, cap)
        } else {
            0.0
        };
        let speed = if want.abs() > self.whisker_steer_deg.abs() {
            c.pull_in_speed
        } else {
            c.return_speed
        };
        self.whisker_steer_deg = interp_to(self.whisker_steer_deg, want, speed, dt);

        // ── the arm ──
        let origin = self.sweep_origin().to_dvec3();
        let delta = self.desired.to_dvec3() - origin;
        let reach = delta.length();
        let free = if free_m.is_finite() {
            free_m.clamp(0.0, reach)
        } else {
            reach
        };
        self.target_arm_m = free;
        // **The smoother owns the CLIP, not the arm** — measured, and it is the
        // difference between a collision model and a lag.
        //
        // Smoothing the arm itself makes every change of `reach` look like a
        // collision: the arm blocks blend (`walk` 3.0 m → `run` 3.4 m) through
        // `state_blend_speed`, so a camera whose arm chased the blended reach at
        // `return_speed` trailed it by 0.231 m in open ground with nothing in the
        // way at all — `camera_3d`'s own control arm caught exactly that. What
        // the collision model is about is how much the world took OFF the boom,
        // so that is the quantity that is smoothed; the boom's own length rides
        // the settings blend untouched, and `collision_pull_m` is zero in an
        // empty field by construction rather than by luck.
        let clip = (reach - free).max(0.0);
        if !self.arm_seeded {
            // A camera that interpolated its clip from zero would spend its first
            // half-second inside whatever it spawned against — `seeded`'s rule,
            // on the quantity `seeded` cannot cover because the sweep happens
            // after it.
            self.arm_seeded = true;
            self.clip_m = clip;
        } else {
            // **The asymmetry, at its limit: the pull-in is a SNAP.**
            //
            // The eased value is the slow return; the `max` is the safety
            // clamp, and it is the whole reason the pull-in is not smoothed.
            // A clip that LAGS a growing requirement is, by definition, a camera
            // inside the thing it was supposed to stop at — for exactly as long
            // as the ease takes. Measured on this wave's own scripted walk with
            // a 30/s pull-in: **33 frames of 840** with the camera's optical
            // centre inside a collider, all of them in the two or three frames
            // after the look swung a wall into the boom. There is no pull-in
            // speed that makes that zero, because any speed below infinity is a
            // lag; the answer is not a faster ease.
            //
            // So the boom comes in on the step the world says so, and eases back
            // out at [`CameraCollision::return_speed`] — which is what "fast in,
            // slow out" means once it is taken seriously. The measured pull-in
            // time is one fixed step and the measured return is hundreds; the
            // gate reports both.
            let eased = interp_to(self.clip_m, clip, c.return_speed, dt);
            self.clip_m = eased.max(clip).clamp(0.0, reach);
        }
        self.arm_m = (reach - self.clip_m).clamp(0.0, reach);
        let position = if reach > 1e-6 {
            Vec3d::from_dvec3(origin + delta / reach * self.arm_m)
        } else {
            self.desired
        };
        self.collision_pull_m = self.clip_m;
        self.pose.position = position;
        self.gameplay_pose = self.pose;
        // **The near fade** — a pure function of where the camera ENDED, not of
        // where the sweep said it could be, because the body a player sees is
        // drawn against the camera that is actually there.
        self.subject_fade = c.near_fade(self.arm_m);
    }

    /// **Run the director** over this step's requests and publish its answer as
    /// [`pose`](Self::pose) (wave CHAR1c).
    ///
    /// With nothing pushed this is the identity on the gameplay pose — the
    /// director seeds on its first call and then has one source for ever, so a
    /// level with no cinematics pays one `Vec::is_empty` and one comparison.
    pub fn direct(&mut self, dt: f64) -> CameraPose {
        let out = self.director.resolve(self.gameplay_pose, dt);
        self.pose = out;
        out
    }

    /// The pivot the physics half sweeps **from** — the point the camera is
    /// looking at, after its own offset. Split out because the sweep needs it and
    /// the desired position alone does not carry it.
    pub fn sweep_origin(&self) -> Vec3d {
        let shoulder: f64 = if self.right_shoulder { 1.0 } else { -1.0 };
        let (pr, _pu, pf) = basis(self.yaw_deg, 0.0);
        Vec3d::from_dvec3(
            self.pivot.to_dvec3()
                + pr * (self.settings.pivot_offset.x * shoulder)
                + DVec3::new(0.0, self.settings.pivot_offset.y, 0.0)
                + pf * self.settings.pivot_offset.z,
        )
    }

    /// Record the resolved position after the sweep. One door, so "where the
    /// camera is" is written in exactly one place.
    pub fn resolve(&mut self, position: Vec3d) {
        self.collision_pull_m = self.desired.to_dvec3().distance(position.to_dvec3());
        self.pose.position = position;
    }

    /// **The camera trace record** — position, orientation and FOV as bytes.
    ///
    /// Deliberately its own function and deliberately *not* folded into
    /// `state_bytes`: `phase29_gate` asserts that this is deterministic across
    /// runs **and** that the sim trace is unchanged by anything the camera is
    /// told (its view mode, its shoulder, its whole tuning table), which together
    /// are the ViewMode ruling's proof. Note the second half is stated as
    /// *perturbation*, not as "with the camera stepped and not stepped": the
    /// camera is stepped unconditionally at the end of both hosts' fixed steps,
    /// so there is no not-stepped run to compare against (P29.6 audit).
    pub fn trace_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * 8);
        for v in [
            self.pose.position.x,
            self.pose.position.y,
            self.pose.position.z,
            self.pose.yaw_deg,
            self.pose.pitch_deg,
            self.pose.fov_deg,
            self.fp_weight,
            self.collision_pull_m,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out
    }
}

/// One frame's first-order interp toward `target` at `speed` — ALS's
/// `FInterpTo`, which is `clamp(speed × dt, 0, 1)` and not an exponential.
///
/// The same rule as [`inf_anim::interp_to`], and it is spelled here rather than
/// reached for because `inf-ecs` naming `inf-anim` for a two-line lerp would be
/// the wrong direction of dependency for a camera.
pub fn interp_to(current: f64, target: f64, speed: f64, dt: f64) -> f64 {
    if !dt.is_finite() || dt <= 0.0 || !target.is_finite() || !current.is_finite() {
        return current;
    }
    // **A speed of zero means SNAP, not freeze** (P29.6 audit, A6). `FInterpTo`
    // opens with `if (InterpSpeed <= 0.f) return Target;`, and the port had
    // `speed.max(0.0)` instead — which makes the blend factor zero and the value
    // *frozen for ever*. That matters because `CameraTuning::set` accepts `0.0`:
    // an author typing zero into `lag_x` to turn the lag OFF got a pivot that
    // never moved again, which is the opposite of what the number says.
    if !speed.is_finite() || speed <= 0.0 {
        return target;
    }
    let a = (speed * dt).clamp(0.0, 1.0);
    current + (target - current) * a
}

/// [`interp_to`] on an **angle**, the short way round.
///
/// A camera whose aim crosses 180° must not take the long way: raw `interp_to`
/// on 179 → −179 sweeps 358 degrees, which is the whole screen spinning for the
/// two degrees the player actually turned.
pub fn interp_angle_deg(current: f64, target: f64, speed: f64, dt: f64) -> f64 {
    let delta = crate::movement::angle_delta_deg(target, current);
    interp_to(current, current + delta, speed, dt)
}

/// **`CalculateAxisIndependentLag`** — the piece of ALS's camera worth copying
/// verbatim (port map §5.1, `.cpp:108–123`).
///
/// Both the current and the target pivot are unrotated into **camera-yaw space**
/// (roll and pitch zeroed), each axis is interpolated at its **own** speed, and
/// the result is rotated back. The point is that "lag" is not one number: a
/// third-person camera wants to trail hard behind a sprinting character
/// (`z`, forward/back), stay tight sideways so a strafe does not swing the frame
/// (`x`), and be softer again vertically so stairs do not bounce the view (`y`).
/// Expressing that in world space would make the answer depend on which way
/// north is.
///
/// # It unrotates the DELTA, not two absolute positions (IB-12)
///
/// The port did what ALS does: unrotate `current` and `target` into yaw space,
/// interpolate each axis, rotate the result back. That is *algebraically*
/// origin-independent — `R(c + (t − c)·α) = c + R((t − c)·α)` because `R` is
/// linear and `R⁻¹R = 1` — and **it is not origin-independent in floating
/// point**, which is the exact wording P29's disposition row 23 carried and the
/// AAA-readiness certification relayed as IB-12: *"unrotates absolute world
/// positions rather than the delta — algebraically origin-independent, not so in
/// floating point at partition scale. It wants a floating-origin-aware camera,
/// which is a streaming-scale question and belongs with the island's 50 km²."*
///
/// At the origin both spellings agree bit for bit. Fifty kilometres out, the
/// rotation multiplies two coordinates of magnitude ~1e5 by a sine and a cosine
/// and then subtracts them — and the *difference* it is really after is the
/// centimetre the character moved this step. Every ULP lost at 1e5 lands whole on
/// a 1e-2 quantity. The absolute form's own inverse does not recover it either:
/// `rotate_from_frame(rotate_into_frame(p))` is not the identity in f64.
///
/// So the delta goes into the frame and the anchor never does. This is the
/// **Wave-T terrain-UV precedent** in a camera: that shader recovered an absolute
/// world position by adding back the grid-axis uniform rather than letting the
/// floating origin into the uv, because *"a uv derived from it is not a property
/// of the ground — it steps by Δorigin/tex_scale tiles at every rebase"*. Here
/// the same rule reads: the reference frame is anchored on `current`, so an
/// origin rebase — which moves nothing this function can see — cannot move the
/// pose, and neither can being a long way from zero.
///
/// The two forms are compared, at the origin and at partition scale, by
/// `crates/inf-ecs/tests/camera_at_scale.rs`, which prices the alternative it
/// rejects rather than asserting the new one is better.
pub fn axis_independent_lag(
    current: Vec3d,
    target: Vec3d,
    yaw_deg: f64,
    speeds: Vec3d,
    dt: f64,
) -> Vec3d {
    // `rotate_into_frame` is the movement model's own XZ rotation — one spelling
    // of "unrotate into a yaw frame" for the whole engine. It is applied to the
    // **delta**: `current` is the frame's anchor and never enters the rotation.
    let d = crate::movement::rotate_into_frame(
        Vec2d::new(target.x - current.x, target.z - current.z),
        yaw_deg,
    );
    let local = Vec2d::new(
        interp_to(0.0, d.x, speeds.x, dt),
        interp_to(0.0, d.y, speeds.z, dt),
    );
    let world = crate::movement::rotate_from_frame(local, yaw_deg);
    Vec3d::new(
        current.x + world.x,
        interp_to(current.y, target.y, speeds.y, dt),
        current.z + world.y,
    )
}

/// The **pre-IB-12 spelling** of [`axis_independent_lag`]: unrotate both absolute
/// world positions, interpolate, rotate back.
///
/// Kept, and public, for exactly one reason — `crates/inf-ecs/tests/camera_at_scale.rs`
/// prices it against the delta form at the origin and at partition scale. A fix
/// whose predecessor has been deleted cannot be shown to have been necessary, and
/// "price the alternative you reject" cuts both ways: the alternative a wave
/// *replaces* has to be priced too.
///
/// Nothing else may call this. `LocomotionCamera::advance` calls the delta form,
/// and a source gate in the same test pins that there is exactly one production
/// call site.
#[doc(hidden)]
pub fn axis_independent_lag_absolute(
    current: Vec3d,
    target: Vec3d,
    yaw_deg: f64,
    speeds: Vec3d,
    dt: f64,
) -> Vec3d {
    let c = crate::movement::rotate_into_frame(Vec2d::new(current.x, current.z), yaw_deg);
    let t = crate::movement::rotate_into_frame(Vec2d::new(target.x, target.z), yaw_deg);
    let local = Vec2d::new(
        interp_to(c.x, t.x, speeds.x, dt),
        interp_to(c.y, t.y, speeds.z, dt),
    );
    let world = crate::movement::rotate_from_frame(local, yaw_deg);
    Vec3d::new(
        world.x,
        interp_to(current.y, target.y, speeds.y, dt),
        world.y,
    )
}

/// Blend a settings block toward another at `speed` — the replacement for ALS's
/// dummy-AnimBP curve blender.
pub fn blend_settings(
    current: CameraSettings,
    target: CameraSettings,
    speed: f64,
    dt: f64,
) -> CameraSettings {
    current.interp(target, speed, dt)
}

/// The right / up / forward axes of a `(yaw, pitch)` rotation, with `+Z`
/// forward at yaw 0 — the engine's own convention (`rotate_from_frame` puts a
/// `+y` intent on `+Z`).
pub fn basis(yaw_deg: f64, pitch_deg: f64) -> (DVec3, DVec3, DVec3) {
    let yaw = yaw_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let (sy, cy) = (inf_math::psin64(yaw), inf_math::pcos64(yaw));
    let (sp, cp) = (inf_math::psin64(pitch), inf_math::pcos64(pitch));
    let forward = DVec3::new(sy * cp, sp, cy * cp);
    let right = DVec3::new(cy, 0.0, -sy);
    // Built by cross product rather than by a third pair of trig calls. The
    // ORDER is `forward × right` and not the other way: at yaw 0 the frame is
    // right `+X`, forward `+Z`, and `X × Z` is `−Y` — measured, by the
    // orthonormality arm, which caught exactly that sign.
    let up = forward.cross(right);
    (right, up, forward)
}

// ─────────────────────────────────────────────────────────────────────────────
// THE AUTHORABLE RIG (wave CHAR1c, clause 2)
// ─────────────────────────────────────────────────────────────────────────────

/// **A character's own camera rig** — the boom, the lag, the FOV, the collision
/// policy and which shoulder, as a component on the character itself.
///
/// # The user's question, and the ruling
///
/// *"when building a character (or a character blueprint) … should we be able to
/// add a camera and a camera boom, or is that stuff automatically added?"* —
/// **both**. A character built through the wizard gets one of these with the
/// ported ALS defaults on it, and every number in it is authorable.
///
/// # Why a component when [`ViewMode`] is camera-side only (Ruling 4)
///
/// Ruling 4 is about the **sim wire**: no camera value may reach `state_bytes`,
/// and no camera may write into the simulation. Neither is touched here. This
/// component is not in the scene record, is not folded into any trace, and is
/// read by exactly one thing — [`crate::camera`]'s own fixed-step door, which
/// reads the world and writes nothing back. What it buys is the three things a
/// host-owned table could not do: a **possessed NPC brings its own boom**, a
/// **Blueprint can read and write it** (`camera.*`, over `&mut EcsWorld`), and
/// the wizard can put a default on a character it builds.
///
/// # WHERE IT PERSISTS: IT DOES NOT (wave CHAR1c's audit)
///
/// **A rig is a runtime default, not an authored value**, and the sentence this
/// paragraph used to carry (*"on the character asset, as the `camera.toml` the
/// wizard already writes beside it"*) was false in three independent ways, each
/// of them now a number in
/// `char1c_gate::an_authored_rig_does_not_survive_a_save_and_a_reload`:
///
/// 1. This component is **not in the scene record** (schema v27 is unmoved,
///    which is what the paragraph below is about), so a level saved and reloaded
///    carries no rig on any character, including one the wizard made a moment
///    earlier. Measured: `walk.arm_length_m` authored at 6.25 m, `camera_rig`
///    answers `None` after a save and a load.
/// 2. The `camera.toml` the wizard writes beside a character is
///    `CameraTuning::default()` and nothing else. It has never read the rig it
///    inserted, so the file and the component cannot be made to agree by an
///    author editing either one.
/// 3. **Nothing reads a character-side `camera.toml`.** The runtime's loader is
///    `inf_player::input::load_camera_beside`, which reads
///    `level_path.with_file_name("camera.toml")`: the file beside the LEVEL. The
///    wizard's file beside a character asset reaches no camera at all.
///
/// The one surface that does persist is that **level-side** table, and the same
/// arm asserts it works. What is missing is a WRITE half for it: a tune made
/// through the live-tuning slider or `camera.set_rig` is session state, exactly
/// as a `Tune::Vehicle` and a `Tune::Weapon` are (`TuneScope::Keep` is
/// documented as meaningful for `Tune::Field` alone), and it is gone at the next
/// launch. The island, which is the level the showcase runs, carries no
/// `camera.toml` at all, so its camera is these compiled-in defaults and there
/// is no file an author can edit to change it.
///
/// Two doors would close it and neither is this wave's or its audit's to spend:
/// a **level-side write half** (`CameraTuning::to_toml` already exists; what is
/// missing is a caller and a frontend affordance), or a **character-asset table
/// read at level load**, which is the design the false sentence assumed and is a
/// load-path change in both hosts. Filed as the audit's own carried item.
///
/// The other branch is priced rather than waved at. A per-level rig is a
/// **positional append to the entity record** — scene v27 → v28, the pre-v28
/// entity record frozen as `EntityRecordV27`, both hosts' `apply_record`
/// mirrors grown (pinned character-for-character by `apply_record_mirror`), the
/// PIE payload v13 → v14, a downgrade bless, and all twenty-four committed
/// `.inf_lvl` re-cooked — **for 776 bytes per character** (97 `f64` at
/// bincode's fixed width) carrying, today, the same numbers on every one of
/// them. That is the whole cost, and it is why this wave did not spend it.
///
/// # A character with no rig is not a character with no camera
///
/// The fixed-step door falls back to the host's own table (the one loaded from
/// `camera.toml` beside the level) for any subject that carries no `CameraRig`,
/// which is every character in every level committed before this wave. So the
/// component is an **override**, and its absence is exactly the behaviour that
/// shipped.
#[derive(bevy_ecs::prelude::Component, bevy_reflect::Reflect, Clone, Debug, PartialEq)]
pub struct CameraRig {
    /// The whole tunable table — arms, offsets, lag, FOV, the driving block, the
    /// collision policy and the pitch limits.
    pub tuning: CameraTuning,
    /// Which shoulder the third-person camera looks over. ALS's own
    /// `bRightShoulder` (`ALSBaseCharacter.h:460`), which it defaults to
    /// **false**; this engine has defaulted it to the right since P29.6 and
    /// keeps that, because the shipped `camera_offset.x` is `+0.45` and a flag
    /// that disagreed with the offset beside it would put the camera on the
    /// wrong side of its own table.
    pub right_shoulder: bool,
    /// Whether this character's camera sits at the first-person seat. ALS's
    /// `EALSViewMode` (`ALSBaseCharacter.h:535`), on the rig rather than on the
    /// session so a possessed NPC can be authored to be first-person.
    pub first_person: bool,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            tuning: CameraTuning::default(),
            right_shoulder: true,
            first_person: false,
        }
    }
}

impl CameraRig {
    /// Set one value by name — [`CameraTuning::set`]'s vocabulary plus the rig's
    /// own two flags, `shoulder` (`> 0.5` is right) and `first_person`.
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match name {
            "shoulder" => {
                self.right_shoulder = value > 0.5;
                true
            }
            "first_person" => {
                self.first_person = value > 0.5;
                true
            }
            _ => self.tuning.set(name, value),
        }
    }

    /// Read one value by name — [`set`](Self::set)'s twin.
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "shoulder" => Some(f64::from(u8::from(self.right_shoulder))),
            "first_person" => Some(f64::from(u8::from(self.first_person))),
            _ => self.tuning.get(name),
        }
    }
}

/// **The scripted and override claims waiting for the camera** (wave CHAR1c).
///
/// A world-level resource rather than a field on the host's camera, because the
/// two things that need to push a claim — a Blueprint's `camera.shot` node and
/// the editor's sequencer — both reach the world and neither can reach a host.
/// The fixed-step camera door drains it into [`CameraDirector`] and clears it,
/// so a claim lives exactly one step, which is the same "no latch to leak"
/// argument [`CameraRequest`] itself is built on.
///
/// It is a resource and therefore never serialized, never in the scene record,
/// and never in `state_bytes` — Ruling 4 is untouched.
#[derive(bevy_ecs::prelude::Resource, Clone, Debug, Default, PartialEq)]
pub struct CameraDirectorRes {
    /// This step's claims, in push order.
    pub pending: Vec<CameraRequest>,
}

/// **Push a camera claim** from anywhere that has the world — the Ring-0 door
/// under the `camera.shot` node and under the editor sequencer's camera track.
pub fn request_camera(world: &mut crate::EcsWorld, request: CameraRequest) {
    let w = world.world_mut();
    if let Some(mut res) = w.get_resource_mut::<CameraDirectorRes>() {
        res.pending.push(request);
        return;
    }
    w.insert_resource(CameraDirectorRes {
        pending: vec![request],
    });
}

/// Take this step's claims, leaving none.
pub fn take_camera_requests(world: &mut crate::EcsWorld) -> Vec<CameraRequest> {
    world
        .world_mut()
        .get_resource_mut::<CameraDirectorRes>()
        .map(|mut r| std::mem::take(&mut r.pending))
        .unwrap_or_default()
}

/// **Ask the camera to sit at `at`'s transform for this step** — the scripted
/// layer's Ring-0 door (wave CHAR1c).
///
/// `at` is any entity: a camera actor an author placed, a socket-attached child,
/// a marker. Its `GlobalTransform` gives the position, its euler yaw and pitch
/// give the orientation, and a [`crate::components::Camera`] on it gives the
/// field of view (60° if it carries none — the component's own default).
///
/// # It is per-step, and that is the contract
///
/// A claim lives exactly one fixed step. A Blueprint that wants to hold a shot
/// calls this every Tick; the step it stops, the director blends back to
/// gameplay over [`DEFAULT_RELEASE_BLEND_S`]. Nothing has to remember to release
/// it, which is the failure mode a latch would have and a cutscene cannot
/// afford.
///
/// `blend_s` of zero is a **cut** — instant, and reported as one.
pub fn camera_shot(world: &mut crate::EcsWorld, at: uuid::Uuid, blend_s: f64) -> bool {
    let Some(e) = world.entity_of(at) else {
        return false;
    };
    let (pose, tag) = {
        let w = world.world();
        let Some(g) = w.get::<crate::components::GlobalTransform>(e) else {
            return false;
        };
        let fov = w
            .get::<crate::components::Camera>(e)
            .map(|c| f64::from(c.fov_y_deg))
            .unwrap_or(60.0);
        // The FORWARD vector, not the euler triple: a `GlobalTransform` is an
        // affine and a socket-attached camera actor's world orientation is a
        // product of its parents' rotations, which no local euler carries.
        // `patan2_64` / `pacos64` and not `f64::atan2` / `asin`, because a gate
        // arm reads this yaw and the P14 law is that std trig is not
        // bit-portable.
        let (_, rot, translation) = g.0.to_scale_rotation_translation();
        let f = (rot * DVec3::Z).normalize_or_zero();
        let yaw = inf_math::patan2_64(f.x, f.z).to_degrees();
        let pitch = 90.0 - inf_math::pacos64(f.y.clamp(-1.0, 1.0)).to_degrees();
        (
            CameraPose {
                position: Vec3d::from_dvec3(translation),
                yaw_deg: crate::movement::wrap_deg(yaw),
                pitch_deg: pitch,
                fov_deg: fov,
            },
            // The shot's identity is the entity it frames, so a Blueprint that
            // moves one camera actor keeps ONE blend and a Blueprint that
            // switches between two cuts between them. The low bits of the guid
            // are enough and the top bits are the version/variant nibbles, which
            // are constant across a level's own entities.
            at.as_u128() as u64,
        )
    };
    request_camera(
        world,
        CameraRequest {
            layer: CameraLayer::Scripted,
            priority: 0,
            tag,
            pose,
            blend_s: if blend_s.is_finite() {
                blend_s.max(0.0)
            } else {
                0.0
            },
        },
    );
    true
}

/// The rig on `guid`, cloned, or `None` — the read half of the Blueprint door.
pub fn camera_rig(world: &crate::EcsWorld, guid: uuid::Uuid) -> Option<CameraRig> {
    let e = world.entity_of(guid)?;
    world.world().get::<CameraRig>(e).cloned()
}

/// Read one of `guid`'s rig values by name.
///
/// `None` for an entity with no rig **and** for a name the door does not know,
/// which are the same answer for the same reason: a Blueprint asking for a
/// number that is not there gets nothing rather than a zero that looks like one.
pub fn camera_rig_value(world: &crate::EcsWorld, guid: uuid::Uuid, name: &str) -> Option<f64> {
    let e = world.entity_of(guid)?;
    world.world().get::<CameraRig>(e)?.get(name)
}

/// **Write one of `guid`'s rig values by name.**
///
/// Answers whether it landed. A character with no rig **gets one** — the
/// defaults — and then takes the write, because "set the boom on this NPC I just
/// possessed" is a sentence a Blueprint author means literally and a refusal
/// there would be a door that works only on wizard-built characters.
pub fn set_camera_rig_value(
    world: &mut crate::EcsWorld,
    guid: uuid::Uuid,
    name: &str,
    value: f64,
) -> bool {
    let Some(e) = world.entity_of(guid) else {
        return false;
    };
    let w = world.world_mut();
    if let Some(mut rig) = w.get_mut::<CameraRig>(e) {
        return rig.set(name, value);
    }
    let mut rig = CameraRig::default();
    if !rig.set(name, value) {
        return false;
    }
    if let Ok(mut ent) = w.get_entity_mut(e) {
        ent.insert(rig);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3d {
        Vec3d::new(x, y, z)
    }

    /// **The headline, measured.** Three speeds mean three different answers, in
    /// the CAMERA's frame and not the world's — so the same motion produces the
    /// same lag whichever way the camera is pointing.
    #[test]
    fn the_lag_is_per_axis_and_in_camera_space() {
        let speeds = v(30.0, 4.0, 2.0); // tight sideways, soft up, loose behind
        let dt = 1.0 / 60.0;
        // Camera facing +Z (yaw 0). A pure +Z move is the "forward/back" axis.
        let forward = axis_independent_lag(Vec3d::ZERO, v(0.0, 0.0, 1.0), 0.0, speeds, dt);
        // …and a pure +X move is the "sideways" one.
        let sideways = axis_independent_lag(Vec3d::ZERO, v(1.0, 0.0, 0.0), 0.0, speeds, dt);
        let up = axis_independent_lag(Vec3d::ZERO, v(0.0, 1.0, 0.0), 0.0, speeds, dt);
        assert!(
            sideways.x > forward.z * 5.0,
            "the sideways axis must be much tighter: {} vs {}",
            sideways.x,
            forward.z
        );
        assert!(up.y > forward.z && up.y < sideways.x, "up sits between");

        // **Camera space, not world space.** Turn the camera 90° and move the
        // target along the camera's new forward: the answer must be the SAME
        // fraction as the yaw-0 forward case.
        let turned = axis_independent_lag(Vec3d::ZERO, v(1.0, 0.0, 0.0), 90.0, speeds, dt);
        assert!(
            (turned.x - forward.z).abs() < 1e-9,
            "the lag is not in camera space: {} vs {}",
            turned.x,
            forward.z
        );
        // The control: with one speed on all three axes the frame cannot matter.
        let iso = v(7.0, 7.0, 7.0);
        let a = axis_independent_lag(Vec3d::ZERO, v(1.0, 0.0, 0.0), 0.0, iso, dt);
        let b = axis_independent_lag(Vec3d::ZERO, v(1.0, 0.0, 0.0), 90.0, iso, dt);
        assert!((a.x - b.x).abs() < 1e-9 && (a.z - b.z).abs() < 1e-9);
    }

    /// The lag **arrives**: a camera that chased for ever would trail behind a
    /// standing character.
    #[test]
    fn the_lag_converges_on_a_standing_target() {
        let target = v(3.0, 1.5, -2.0);
        let mut p = Vec3d::ZERO;
        for _ in 0..600 {
            p = axis_independent_lag(p, target, 35.0, v(8.0, 8.0, 8.0), 1.0 / 60.0);
        }
        assert!(
            p.to_dvec3().distance(target.to_dvec3()) < 1e-6,
            "the pivot never arrived: {p:?}"
        );
    }

    /// An aim that crosses ±180° takes the **short** way. Without
    /// [`interp_angle_deg`] the whole frame spins for a two-degree turn.
    #[test]
    fn the_rotation_lag_does_not_take_the_long_way_round() {
        let mut yaw = 179.0;
        for _ in 0..120 {
            yaw = interp_angle_deg(yaw, -179.0, 10.0, 1.0 / 60.0);
        }
        let err = crate::movement::angle_delta_deg(-179.0, yaw).abs();
        assert!(err < 0.5, "the camera did not arrive: {yaw}");
        // The falsification: it never passed through the far side.
        let mut yaw = 179.0;
        let mut worst = 0.0f64;
        for _ in 0..120 {
            yaw = interp_angle_deg(yaw, -179.0, 10.0, 1.0 / 60.0);
            worst = worst.max(crate::movement::angle_delta_deg(yaw, 180.0).abs());
        }
        assert!(
            worst < 10.0,
            "the camera swung {worst} degrees off the seam"
        );
    }

    /// A settings change **blends**; it does not cut. That is the whole of what
    /// ALS's dummy AnimBP was for.
    #[test]
    fn a_state_change_blends_rather_than_cutting() {
        let t = CameraTuning::default();
        let run = t.settings_for(
            RotationMode::LookingDirection,
            Gait::Run,
            MovementMode::Grounded,
        );
        let aim = t.settings_for(RotationMode::Aiming, Gait::Run, MovementMode::Grounded);
        assert_ne!(
            run.arm_length_m, aim.arm_length_m,
            "the fixture is degenerate"
        );
        let one = blend_settings(run, aim, t.state_blend_speed, 1.0 / 60.0);
        assert!(
            one.arm_length_m < run.arm_length_m && one.arm_length_m > aim.arm_length_m,
            "a single step jumped the whole way: {}",
            one.arm_length_m
        );
        let mut s = run;
        for _ in 0..600 {
            s = blend_settings(s, aim, t.state_blend_speed, 1.0 / 60.0);
        }
        assert!((s.arm_length_m - aim.arm_length_m).abs() < 1e-6);
    }

    /// Every low stance shares the crouch block, and a sprint is its own — the
    /// table really dispatches.
    #[test]
    fn the_table_dispatches_on_all_three_axes() {
        let t = CameraTuning::default();
        let run = t.settings_for(
            RotationMode::LookingDirection,
            Gait::Run,
            MovementMode::Grounded,
        );
        let sprint = t.settings_for(
            RotationMode::LookingDirection,
            Gait::Sprint,
            MovementMode::Grounded,
        );
        let crouch = t.settings_for(
            RotationMode::LookingDirection,
            Gait::Run,
            MovementMode::Crouch,
        );
        let aim = t.settings_for(RotationMode::Aiming, Gait::Run, MovementMode::Grounded);
        assert!(
            sprint.arm_length_m > run.arm_length_m,
            "a sprint pulls back"
        );
        assert!(crouch.arm_length_m < run.arm_length_m, "a crouch pulls in");
        assert!(
            aim.arm_length_m < crouch.arm_length_m,
            "an aim pulls in most"
        );
        for mode in [
            MovementMode::Crouch,
            MovementMode::Prone,
            MovementMode::Slide,
            MovementMode::Roll,
        ] {
            assert_eq!(
                t.settings_for(RotationMode::LookingDirection, Gait::Run, mode),
                crouch,
                "{mode:?} does not share the low block"
            );
        }
    }

    /// **THE DRIVE CAMERA IS ITS OWN BLOCK**, and it does not read the gait.
    ///
    /// The arm the whole clause turns on. Before VEH2a a driving character got
    /// whichever on-foot block was latched when it pressed the interact key, so
    /// the *same drive* produced three different cameras depending on whether
    /// the player had been walking, running or sprinting when they got in — and
    /// `actual_gait` cannot change while `step_driving` owns the character, so it
    /// stayed wrong for the whole segment.
    #[test]
    fn the_drive_camera_is_its_own_block_and_ignores_the_stale_gait() {
        let t = CameraTuning::default();
        let drive = t.settings_for(
            RotationMode::LookingDirection,
            Gait::Walk,
            MovementMode::Driving,
        );
        assert_eq!(drive, t.driving.base);
        for gait in [Gait::Walk, Gait::Run, Gait::Sprint] {
            for mode in [
                RotationMode::VelocityDirection,
                RotationMode::LookingDirection,
                RotationMode::Aiming,
            ] {
                assert_eq!(
                    t.settings_for(mode, gait, MovementMode::Driving),
                    drive,
                    "a drive with a latched {gait:?}/{mode:?} got a different camera"
                );
            }
        }
        // …and it really is a different camera from every on-foot one, or the
        // arm above is satisfied by a table that has not changed.
        for gait in [Gait::Walk, Gait::Run, Gait::Sprint] {
            assert_ne!(
                t.settings_for(RotationMode::LookingDirection, gait, MovementMode::Grounded),
                drive
            );
        }
        // A car sits further back than a sprint and rolls its boom slower than a
        // walk — the two things a chase camera is.
        let sprint = t.settings_for(
            RotationMode::LookingDirection,
            Gait::Sprint,
            MovementMode::Grounded,
        );
        assert!(drive.arm_length_m > sprint.arm_length_m);
        assert!(drive.rotation_lag < sprint.rotation_lag);
        // …and no shoulder offset: a car is not looked over.
        assert_eq!(drive.camera_offset.x, 0.0);
    }

    /// **The drive camera reaches back and widens with speed, and with the car.**
    #[test]
    fn the_drive_camera_grows_with_the_speed_and_the_size_of_the_car() {
        let t = CameraTuning::default();
        let settled = |speed: f64, half_length: f64| -> CameraPose {
            let mut cam = LocomotionCamera::default();
            let input = CameraInput {
                pivot_target: v(0.0, 1.5, 0.0),
                aim_yaw_deg: 0.0,
                aim_pitch_deg: 0.0,
                rotation_mode: RotationMode::LookingDirection,
                gait: Gait::Walk,
                mode: MovementMode::Driving,
                driving: Some(DrivingView {
                    chassis_yaw_deg: 0.0,
                    velocity: v(0.0, 0.0, speed),
                    half_length_m: half_length,
                }),
            };
            // Long enough for `state_blend_speed` to arrive.
            for _ in 0..300 {
                cam.advance(&input, 1.0 / 60.0);
            }
            cam.pose
        };
        let parked = settled(0.0, 2.2);
        let fast = settled(35.0, 2.2);
        let bus = settled(0.0, 6.0);
        assert!(
            fast.fov_deg > parked.fov_deg + 5.0,
            "a car at 35 m/s got {:.1}° of field against {:.1}° parked — the \
             cheapest speed cue there is, and it is not firing",
            fast.fov_deg,
            parked.fov_deg
        );
        assert!(
            fast.fov_deg <= parked.fov_deg + t.driving.fov_gain_max_deg + 1e-6,
            "the FOV gain broke its own ceiling at {:.1}°",
            fast.fov_deg
        );
        // The arm shows up as distance from the pivot, and the pivot itself has
        // moved (the look-ahead), so measure the arm through the settings.
        let arm = |speed: f64, half_length: f64| -> f64 {
            let mut cam = LocomotionCamera::default();
            let input = CameraInput {
                pivot_target: v(0.0, 1.5, 0.0),
                aim_yaw_deg: 0.0,
                aim_pitch_deg: 0.0,
                rotation_mode: RotationMode::LookingDirection,
                gait: Gait::Walk,
                mode: MovementMode::Driving,
                driving: Some(DrivingView {
                    chassis_yaw_deg: 0.0,
                    velocity: v(0.0, 0.0, speed),
                    half_length_m: half_length,
                }),
            };
            for _ in 0..300 {
                cam.advance(&input, 1.0 / 60.0);
            }
            cam.settings.arm_length_m
        };
        assert!(
            arm(35.0, 2.2) > arm(0.0, 2.2) + 1.0,
            "the boom did not extend with speed"
        );
        assert!(
            arm(0.0, 6.0) > arm(0.0, 2.2) + 1.5,
            "a six-metre bus sat as close as a two-metre car: {:.2} against {:.2}",
            arm(0.0, 6.0),
            arm(0.0, 2.2)
        );
        let _ = bus;
        // A character NOT driving is untouched by any of it.
        let mut walking = LocomotionCamera::default();
        let on_foot = CameraInput {
            pivot_target: v(0.0, 1.5, 0.0),
            aim_yaw_deg: 0.0,
            aim_pitch_deg: 0.0,
            rotation_mode: RotationMode::LookingDirection,
            gait: Gait::Run,
            mode: MovementMode::Grounded,
            driving: Some(DrivingView {
                chassis_yaw_deg: 90.0,
                velocity: v(0.0, 0.0, 35.0),
                half_length_m: 6.0,
            }),
        };
        for _ in 0..300 {
            walking.advance(&on_foot, 1.0 / 60.0);
        }
        assert_eq!(
            walking.settings.fov_deg,
            t.settings_for(
                RotationMode::LookingDirection,
                Gait::Run,
                MovementMode::Grounded
            )
            .fov_deg,
            "a walking character read the drive camera's speed terms"
        );
    }

    /// **The camera swings behind the car as it gets going, and looks up the
    /// road** — and a parked car leaves the driver's aim alone.
    #[test]
    fn the_drive_camera_aligns_with_the_car_and_looks_ahead() {
        let run = |speed: f64| -> (f64, Vec3d) {
            let mut cam = LocomotionCamera::default();
            let input = CameraInput {
                // The driver is looking 90° off the car's heading.
                pivot_target: v(0.0, 1.5, 0.0),
                aim_yaw_deg: 90.0,
                aim_pitch_deg: 0.0,
                rotation_mode: RotationMode::LookingDirection,
                gait: Gait::Walk,
                mode: MovementMode::Driving,
                driving: Some(DrivingView {
                    chassis_yaw_deg: 0.0,
                    velocity: v(0.0, 0.0, speed),
                    half_length_m: 2.2,
                }),
            };
            for _ in 0..300 {
                cam.advance(&input, 1.0 / 60.0);
            }
            (cam.yaw_deg, cam.pivot)
        };
        let (parked_yaw, parked_pivot) = run(0.0);
        let (moving_yaw, moving_pivot) = run(30.0);
        assert!(
            (parked_yaw - 90.0).abs() < 1e-6,
            "a PARKED car dragged the camera off the driver's aim to {parked_yaw}° \
             — a driver reversing into a space must still be able to look"
        );
        assert!(
            moving_yaw.abs() < 1.0,
            "at 30 m/s the camera sat at {moving_yaw}° instead of behind a car \
             heading 0°"
        );
        // The look-ahead: the pivot is pushed up the road by the authored number
        // of seconds of velocity, and a parked car's is not pushed at all.
        assert!(
            (parked_pivot.z - 0.0).abs() < 1e-6,
            "a parked car's pivot drifted to {}",
            parked_pivot.z
        );
        let want = 30.0 * CameraTuning::default().driving.look_ahead_s;
        assert!(
            (moving_pivot.z - want).abs() < 0.05,
            "the look-ahead put the pivot {:.2} m up the road against {want:.2}",
            moving_pivot.z
        );
    }

    /// The first frame **snaps**. A camera that lerped from the origin would fly
    /// across the level on every load.
    #[test]
    fn the_first_frame_snaps_and_the_second_does_not() {
        let mut cam = LocomotionCamera::default();
        let input = CameraInput {
            pivot_target: v(100.0, 2.0, -50.0),
            aim_yaw_deg: 90.0,
            aim_pitch_deg: -10.0,
            rotation_mode: RotationMode::LookingDirection,
            gait: Gait::Run,
            mode: MovementMode::Grounded,
            driving: None,
        };
        cam.advance(&input, 1.0 / 60.0);
        assert_eq!(cam.pivot, input.pivot_target, "the first frame must snap");
        assert_eq!(cam.yaw_deg, 90.0);
        // …and the camera really is an arm's length from the pivot, behind it.
        let dist = cam.desired.to_dvec3().distance(cam.pivot.to_dvec3());
        assert!(
            (dist - cam.settings.arm_length_m).abs() < 0.6,
            "the arm is {dist} against {}",
            cam.settings.arm_length_m
        );

        let moved = CameraInput {
            pivot_target: v(110.0, 2.0, -50.0),
            ..input
        };
        cam.advance(&moved, 1.0 / 60.0);
        assert!(
            cam.pivot.x > 100.0 && cam.pivot.x < 110.0,
            "the second frame must LAG: {}",
            cam.pivot.x
        );
    }

    /// The tuning door edits what it names, and refuses what it does not — the
    /// live-tuning contract, since a camera is not a reflected component.
    #[test]
    fn the_tuning_door_is_by_name_and_refusals_are_values() {
        let mut t = CameraTuning::default();
        assert!(t.set("run.arm_length_m", 5.5));
        assert_eq!(
            t.settings_for(
                RotationMode::LookingDirection,
                Gait::Run,
                MovementMode::Grounded
            )
            .arm_length_m,
            5.5
        );
        assert!(t.set("aim.fov_deg", 40.0));
        assert_eq!(
            t.settings_for(RotationMode::Aiming, Gait::Walk, MovementMode::Grounded)
                .fov_deg,
            40.0
        );
        assert!(t.set("pivot_height_ratio", 0.9));
        assert_eq!(t.pivot_height_ratio, 0.9);
        assert!(t.set("first_person.offset_y", 0.2));
        assert_eq!(t.first_person.camera_offset.y, 0.2);
        // The drive block (island wave VEH2a): its base goes through the shared
        // field table like every other block, and its own six scalars are
        // table-wide names under a `drive.` prefix.
        assert!(t.set("drive.arm_length_m", 6.25));
        assert_eq!(
            t.settings_for(
                RotationMode::LookingDirection,
                Gait::Walk,
                MovementMode::Driving
            )
            .arm_length_m,
            6.25
        );
        assert!(t.set("drive.lag_z", 2.0));
        assert_eq!(t.driving.base.lag_speeds.z, 2.0);
        for (name, want) in [
            ("drive.arm_per_length_m", 0.9),
            ("drive.arm_per_speed_s", 0.02),
            ("drive.fov_per_speed_deg_s", 0.6),
            ("drive.fov_gain_max_deg", 20.0),
            ("drive.look_ahead_s", 0.5),
            ("drive.align_speed_mps", 12.0),
        ] {
            assert!(t.set(name, want), "the door refuses `{name}`");
        }
        assert_eq!(t.driving.arm_per_length_m, 0.9);
        assert_eq!(t.driving.align_speed_mps, 12.0);
        assert!(
            !t.set("drive.arm_per_length", 1.0),
            "a near-miss is refused"
        );
        // Refusals, all values.
        assert!(!t.set("run.arm_length", 1.0), "a misspelled field");
        assert!(!t.set("gallop.arm_length_m", 1.0), "a misspelled block");
        assert!(!t.set("nonsense", 1.0));
        assert!(!t.set("run.arm_length_m", f64::NAN), "a NaN is refused");
        assert_eq!(
            t.settings_for(
                RotationMode::LookingDirection,
                Gait::Run,
                MovementMode::Grounded
            )
            .arm_length_m,
            5.5,
            "a refused tune changed something"
        );
    }

    /// The basis is orthonormal and right-handed at every angle the camera can
    /// reach — the arithmetic every offset above rests on.
    #[test]
    fn the_basis_is_orthonormal() {
        for yaw in [-180.0, -90.0, 0.0, 37.5, 90.0, 179.0] {
            for pitch in [-89.0, -45.0, 0.0, 45.0, 89.0] {
                let (r, u, f) = basis(yaw, pitch);
                let len = |a: DVec3| a.length();
                let dot = |a: DVec3, b: DVec3| a.dot(b);
                // 1e-6, not an epsilon: `inf_math::psin64`/`pcos64` are the
                // BIT-PORTABLE pair (the P14 law), and portability is bought
                // with a polynomial whose own error is around 1e-7. A camera
                // basis a ten-millionth off unit length is a camera basis; a
                // basis that agrees on two targets is the property that matters.
                for (name, a) in [("right", r), ("up", u), ("forward", f)] {
                    assert!(
                        (len(a) - 1.0).abs() < 1e-6,
                        "{name} at ({yaw}, {pitch}) is {}",
                        len(a)
                    );
                }
                assert!(dot(r, u).abs() < 1e-6);
                assert!(dot(r, f).abs() < 1e-6);
                assert!(dot(u, f).abs() < 1e-6);
            }
        }
        // Yaw 0 looks down +Z, which is the engine's own forward (the movement
        // model's `rotate_from_frame` puts a `+y` intent there).
        let (r, u, f) = basis(0.0, 0.0);
        assert!((f.z - 1.0).abs() < 1e-6, "{f:?}");
        assert!((r.x - 1.0).abs() < 1e-6, "{r:?}");
        assert!((u.y - 1.0).abs() < 1e-6, "{u:?}");
    }

    /// The trace is bytes, and it moves when the camera does — the shape
    /// `phase29_gate` asserts determinism on.
    #[test]
    fn the_trace_is_bytes_and_it_moves() {
        let mut cam = LocomotionCamera::default();
        let input = CameraInput {
            pivot_target: v(0.0, 1.5, 0.0),
            aim_yaw_deg: 0.0,
            aim_pitch_deg: 0.0,
            rotation_mode: RotationMode::LookingDirection,
            gait: Gait::Run,
            mode: MovementMode::Grounded,
            driving: None,
        };
        cam.advance(&input, 1.0 / 60.0);
        let a = cam.trace_bytes();
        assert_eq!(a.len(), 64, "eight f64s");
        cam.advance(
            &CameraInput {
                pivot_target: v(5.0, 1.5, 0.0),
                aim_yaw_deg: 40.0,
                ..input
            },
            1.0 / 60.0,
        );
        assert_ne!(a, cam.trace_bytes(), "the trace did not follow the camera");
    }

    /// **A lag speed of zero SNAPS** (P29.6 audit, A6) — ALS's `FInterpTo` opens
    /// with `if (InterpSpeed <= 0) return Target;`, and the first cut of the port
    /// used `speed.max(0.0)`, which makes the blend factor zero and the value
    /// *frozen for ever*.
    ///
    /// Reachable rather than theoretical: `CameraTuning::set` accepts `0.0`, so
    /// an author turning the lag off through the tuning door got a pivot that
    /// never moved again — and the convergence arm above measures from a
    /// non-zero speed, so it could not see it.
    #[test]
    fn a_zero_lag_speed_snaps_rather_than_freezing() {
        let dt = 1.0 / 60.0;
        assert_eq!(interp_to(0.0, 7.0, 0.0, dt), 7.0, "zero froze the value");
        assert_eq!(interp_to(0.0, 7.0, -3.0, dt), 7.0, "a negative froze it");
        // …and through the door an author actually types into.
        let mut t = CameraTuning::default();
        assert!(t.set("run.lag_x", 0.0));
        let lag = axis_independent_lag(
            Vec3d::ZERO,
            v(1.0, 0.0, 0.0),
            0.0,
            t.looking_direction.run.lag_speeds,
            dt,
        );
        // 1e-6 rather than an epsilon: the frame round-trip goes through
        // `psin64`/`pcos64`, whose `cos(0)` is 0.999999887 — a property of the
        // portable pair, not of this rule.
        assert!(
            (lag.x - 1.0).abs() < 1e-6,
            "a zero sideways lag speed did not reach the target: {lag:?}"
        );
        // The other axes are untouched, so the snap is per-axis.
        assert!(lag.z.abs() < 1e-6, "the forward axis moved: {lag:?}");
    }

    /// **The camera yaw is folded, not accumulated** (P29.6 audit, A8).
    ///
    /// `interp_angle_deg` chases a wrapped aim through an unwrapped delta, so a
    /// character that keeps turning one way used to carry the camera's yaw past
    /// 360, 720, 1080 — into `CameraPose`, into `trace_bytes` and into `basis`,
    /// whose portable range reduction is measurably worse at large arguments
    /// (the P23 finding). The short-way arm above is the control: it is
    /// unchanged.
    #[test]
    fn the_camera_yaw_stays_folded_across_revolutions() {
        let mut cam = LocomotionCamera::default();
        let base = CameraInput {
            pivot_target: v(0.0, 1.5, 0.0),
            aim_yaw_deg: 0.0,
            aim_pitch_deg: 0.0,
            rotation_mode: RotationMode::LookingDirection,
            gait: Gait::Run,
            mode: MovementMode::Grounded,
            driving: None,
        };
        cam.advance(&base, 1.0 / 60.0);
        // Six full revolutions at four degrees a step, the aim wrapped at its
        // own door exactly as the movement step wraps it.
        let mut aim = 0.0f64;
        for _ in 0..540 {
            aim = crate::movement::wrap_deg(aim + 4.0);
            cam.advance(
                &CameraInput {
                    aim_yaw_deg: aim,
                    ..base
                },
                1.0 / 60.0,
            );
        }
        assert!(
            (0.0..360.0).contains(&cam.yaw_deg),
            "the camera yaw drifted out of [0, 360): {}",
            cam.yaw_deg
        );
        // …and it is still CHASING, not stuck: it trails the aim by a bounded
        // angle rather than sitting where it started.
        let lag = crate::movement::angle_delta_deg(aim, cam.yaw_deg).abs();
        assert!(
            lag < 45.0,
            "the camera stopped following the aim: {lag} deg"
        );
    }

    /// **A partial `camera.toml` keeps the ALS table** (P29.6 audit, A7).
    ///
    /// `GaitCameraSettings`' doc promised that a file may name only the numbers
    /// an author is tuning. `#[serde(default)]` does not deliver that below the
    /// top level — it fills a missing field from the FIELD TYPE's default, and
    /// the ported table is a property of the whole `CameraTuning` — so a
    /// two-line file used to hand the first-person seat a third-person arm.
    /// `from_toml` folds the document onto the serialized default instead.
    #[test]
    fn a_partial_camera_table_keeps_every_number_it_did_not_name() {
        let d = CameraTuning::default();
        let t = CameraTuning::from_toml(
            "[first_person]\nfov_deg = 95.0\n\n[looking_direction.walk]\narm_length_m = 3.2\n",
        )
        .expect("a partial table is a legal table");
        // What the file named.
        assert_eq!(t.first_person.fov_deg, 95.0);
        assert_eq!(t.looking_direction.walk.arm_length_m, 3.2);
        // What it did not — and these are the ones the old shape got wrong: the
        // first-person seat's arm is ZERO (the camera is at the pivot), not the
        // third-person run block's 3.4 m.
        assert_eq!(
            t.first_person.arm_length_m, d.first_person.arm_length_m,
            "the first-person seat inherited a third-person arm"
        );
        assert_eq!(t.first_person.camera_offset, d.first_person.camera_offset);
        assert_eq!(t.first_person.lag_speeds, d.first_person.lag_speeds);
        assert_eq!(
            t.looking_direction.walk.lag_speeds, d.looking_direction.walk.lag_speeds,
            "naming one number in a block reset the block's other numbers"
        );
        assert_eq!(t.looking_direction.sprint, d.looking_direction.sprint);
        assert_eq!(t.looking_direction.crouch, d.looking_direction.crouch);
        assert_eq!(t.aiming, d.aiming, "an unnamed rotation mode moved");
        assert_eq!(t.collision_radius_m, d.collision_radius_m);
        // An empty file is the whole table, and the writer's own output is a
        // fixed point of the reader.
        assert_eq!(CameraTuning::from_toml("").expect("empty is legal"), d);
        assert_eq!(
            CameraTuning::from_toml(&d.to_toml().unwrap()).expect("round trip"),
            d
        );
        // A file that is not a table at all is a named refusal, not a default.
        assert!(CameraTuning::from_toml("this is not toml").is_err());
    }
}
