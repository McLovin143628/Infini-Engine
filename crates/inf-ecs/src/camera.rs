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
//! door; the camera reads the same number afterwards. So there is no camera →
//! sim path to trace, and `phase29_gate` asserts that by stepping the sim with
//! and without a camera and comparing the bytes.
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// **The whole tunable set**, and the thing P29.5's live-tuning door edits.
///
/// The table is ALS's `FALSCameraStateSettings` — `RotationMode` × (gait +
/// crouch) = 3 × 4 = twelve blocks — plus the first-person seat and the handful
/// of numbers that are not per-state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraTuning {
    pub velocity_direction: GaitCameraSettings,
    pub looking_direction: GaitCameraSettings,
    pub aiming: GaitCameraSettings,
    /// The first-person **seat**: the arm length is ignored (the camera is at the
    /// pivot), and the offsets place the eye.
    pub first_person: CameraSettings,
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
            pivot_height_ratio: 0.80,
            collision_radius_m: 0.15,
            state_blend_speed: 6.0,
            view_blend_speed: 8.0,
            min_arm_fraction: 0.05,
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
    /// `offset_x` / `offset_y` / `offset_z` (the camera offset), or the four
    /// table-wide names `pivot_height_ratio`, `collision_radius_m`,
    /// `state_blend_speed`, `view_blend_speed`.
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
}

/// Where the camera ended up this step. The whole of what a renderer needs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraPose {
    pub position: Vec3d,
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    pub fov_deg: f64,
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
    /// Whether the camera has taken its first frame. A camera that lerped from
    /// the origin on step one would swing across the level.
    pub seeded: bool,
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
            seeded: false,
        }
    }
}

impl LocomotionCamera {
    /// **Everything except the sweep.** Advances the smoothers and computes the
    /// desired camera position; `inf_physics::d3::camera` calls this and then
    /// resolves the collision, because a swept sphere needs a physics world and
    /// nothing else here does.
    pub fn advance(&mut self, input: &CameraInput, dt: f64) {
        let target = if self.view_mode == ViewMode::FirstPerson {
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
            self.pivot = input.pivot_target;
            self.yaw_deg = input.aim_yaw_deg;
            self.pitch_deg = input.aim_pitch_deg;
        } else {
            self.settings = self
                .settings
                .interp(target, self.tuning.state_blend_speed, dt);
            self.fp_weight = interp_to(self.fp_weight, fp_target, self.tuning.view_blend_speed, dt)
                .clamp(0.0, 1.0);
            // Rotation lag: one speed chasing the aim, on the SHORT way round —
            // `interp_to` on raw degrees would take the long way through 359.
            self.yaw_deg = interp_angle_deg(
                self.yaw_deg,
                input.aim_yaw_deg,
                self.settings.rotation_lag,
                dt,
            );
            self.pitch_deg = interp_to(
                self.pitch_deg,
                input.aim_pitch_deg,
                self.settings.rotation_lag,
                dt,
            );
            // **The headline** — three interp speeds resolved in camera-yaw
            // space, so the camera can lag hard on forward/back, softly
            // sideways, and differently again vertically.
            self.pivot = axis_independent_lag(
                self.pivot,
                input.pivot_target,
                self.yaw_deg,
                self.settings.lag_speeds,
                dt,
            );
        }

        let shoulder: f64 = if self.right_shoulder { 1.0 } else { -1.0 };
        // The pivot's own frame is the CAMERA's yaw (pitch and roll zeroed) —
        // ALS's `CalculateAxisIndependentLag` convention, kept for the offset so
        // the two cannot disagree about what "right" means.
        let pivot = self.sweep_origin().to_dvec3();

        // The camera's own frame carries pitch, because the arm swings with it.
        let (cr, cu, cf) = basis(self.yaw_deg, self.pitch_deg);
        let arm = self.settings.arm_length_m * (1.0 - self.fp_weight);
        let desired = pivot - cf * arm
            + cr * (self.settings.camera_offset.x * shoulder)
            + cu * self.settings.camera_offset.y
            + cf * self.settings.camera_offset.z;
        self.desired = Vec3d::from_dvec3(desired);
        self.pose = CameraPose {
            position: self.desired,
            yaw_deg: self.yaw_deg,
            pitch_deg: self.pitch_deg,
            fov_deg: self.settings.fov_deg,
        };
        self.collision_pull_m = 0.0;
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
    /// runs **and** that the sim trace is byte-identical whether or not a camera
    /// was stepped, which together are the ViewMode ruling's proof.
    pub fn trace_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(6 * 8 + 8);
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
    let a = (speed.max(0.0) * dt).clamp(0.0, 1.0);
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
pub fn axis_independent_lag(
    current: Vec3d,
    target: Vec3d,
    yaw_deg: f64,
    speeds: Vec3d,
    dt: f64,
) -> Vec3d {
    // `rotate_into_frame` is the movement model's own XZ rotation — one spelling
    // of "unrotate into a yaw frame" for the whole engine.
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
}
