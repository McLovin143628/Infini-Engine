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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
