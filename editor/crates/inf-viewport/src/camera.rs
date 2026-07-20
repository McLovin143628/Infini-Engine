//! Editor camera controller: pure input→state math, platform-neutral and
//! unit-tested headless (the per-OS hosts only feed it input deltas).
//!
//! UE-parity navigation:
//! - RMB captured: mouse look + WASD/QE fly, wheel scales speed, Shift boosts.
//! - Alt+LMB orbit / Alt+MMB (or plain MMB) pan / Alt+RMB dolly around a pivot.
//! - Wheel dollies toward the look point when not flying.
//! - F focuses the selection (smooth interpolation); Ctrl+1..9 store camera
//!   bookmarks, 1..9 recall them.
//!
//! The camera is a yaw/pitch/eye triple; orbit/focus derive a pivot on demand
//! rather than storing one, so switching between fly and orbit is seamless.

use glam::{DVec3, Vec3};

/// Radians per raw mouse count.
const LOOK_SENSITIVITY: f32 = 0.0032;
/// Orbit uses the same angular sensitivity as free-look.
const ORBIT_SENSITIVITY: f32 = 0.0060;
/// Pitch stays just short of the poles to keep `forward` well-defined.
const PITCH_LIMIT: f32 = 1.55;
pub const FLY_SPEED_MIN: f32 = 0.2;
pub const FLY_SPEED_MAX: f32 = 250.0;
/// Fraction of the remaining focus distance covered per second (exponential
/// ease-out). ~0.001 of the gap remains after the time-constant.
const FOCUS_RATE: f32 = 12.0;
/// Focus interpolation snaps to done inside this world-space distance.
const FOCUS_EPSILON: f64 = 0.01;
/// Camera bookmark slots (recalled by digit keys 1..=9).
pub const BOOKMARK_SLOTS: usize = 9;
/// Default pivot distance ahead of the eye when nothing is selected.
const DEFAULT_PIVOT_DISTANCE: f64 = 10.0;

/// Accumulated flycam input for one frame (already coalesced by the host).
#[derive(Debug, Clone, Copy, Default)]
pub struct FlyInput {
    /// Raw mouse deltas while captured.
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    /// Wheel detents (speed scaling while captured).
    pub wheel_steps: i32,
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub boost: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EditorCamera {
    /// World-space eye (f64 — architecture rule 3).
    pub pos: DVec3,
    /// Radians around +Y; 0 looks down -Z.
    pub yaw: f32,
    /// Radians; positive looks up.
    pub pitch: f32,
    /// Metres per second while flying.
    pub fly_speed: f32,
}

impl Default for EditorCamera {
    fn default() -> Self {
        // Perched behind-right of the origin, overlooking the demo field.
        Self {
            pos: DVec3::new(14.0, 9.0, 20.0),
            yaw: -0.55,
            pitch: -0.35,
            fly_speed: 8.0,
        }
    }
}

impl EditorCamera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }

    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy)
    }

    /// One frame of captured flycam movement.
    pub fn apply_fly(&mut self, input: &FlyInput, dt: f32) {
        if input.wheel_steps != 0 {
            self.fly_speed = (self.fly_speed * 1.2f32.powi(input.wheel_steps))
                .clamp(FLY_SPEED_MIN, FLY_SPEED_MAX);
        }

        self.yaw += input.mouse_dx * LOOK_SENSITIVITY;
        self.pitch =
            (self.pitch - input.mouse_dy * LOOK_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);

        let mut mv = Vec3::ZERO;
        if input.forward {
            mv += self.forward();
        }
        if input.back {
            mv -= self.forward();
        }
        if input.right {
            mv += self.right();
        }
        if input.left {
            mv -= self.right();
        }
        if input.up {
            mv += Vec3::Y;
        }
        if input.down {
            mv -= Vec3::Y;
        }
        if mv != Vec3::ZERO {
            let boost = if input.boost { 4.0 } else { 1.0 };
            let step = mv.normalize() * self.fly_speed * boost * dt;
            self.pos += step.as_dvec3();
        }
    }

    fn set_forward(&mut self, dir: Vec3) {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO {
            return;
        }
        self.pitch = d.y.clamp(-1.0, 1.0).asin().clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.yaw = d.x.atan2(-d.z);
    }

    /// Point the camera at `target` from its current position.
    pub fn look_at(&mut self, target: DVec3) {
        self.set_forward((target - self.pos).as_vec3());
    }

    /// The point the camera orbits/dollies around: the caller's focus point
    /// (selection center) if any, else a fixed distance down the view ray.
    pub fn pivot(&self, focus: Option<DVec3>) -> DVec3 {
        focus.unwrap_or_else(|| self.pos + self.forward().as_dvec3() * DEFAULT_PIVOT_DISTANCE)
    }

    /// One frame of orbit/pan/dolly around `pivot` (Maya/UE Alt-navigation).
    pub fn apply_navigate(&mut self, input: &NavInput, pivot: DVec3, dt: f32) {
        let _ = dt; // navigation is velocity-free (direct mouse mapping)
        match input.mode {
            NavMode::Orbit => {
                let radius = (self.pos - pivot).length();
                self.yaw += input.mouse_dx * ORBIT_SENSITIVITY;
                self.pitch = (self.pitch + input.mouse_dy * ORBIT_SENSITIVITY)
                    .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                // Re-place the eye on the sphere, still looking at the pivot.
                self.pos = pivot - self.forward().as_dvec3() * radius.max(0.05);
            }
            NavMode::Pan => {
                // Screen-space pan scaled by pivot distance so it tracks the
                // point under the cursor regardless of zoom.
                let dist = (self.pos - pivot).length() as f32;
                let scale = dist * 0.0016;
                let delta =
                    -self.right() * input.mouse_dx * scale + self.up() * input.mouse_dy * scale;
                self.pos += delta.as_dvec3();
            }
            NavMode::Dolly => {
                // Horizontal drag / wheel moves along the view ray toward the
                // pivot; never crosses it.
                self.dolly(
                    input.mouse_dx * 0.01 + input.wheel_steps as f32 * 0.12,
                    pivot,
                );
            }
            NavMode::None => {}
        }
    }

    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    /// Move along the view ray by `amount` (fraction of pivot distance);
    /// positive pulls toward the pivot. Clamped so the eye never passes it.
    pub fn dolly(&mut self, amount: f32, pivot: DVec3) {
        let to_pivot = pivot - self.pos;
        let dist = to_pivot.length();
        if dist < 1e-4 {
            return;
        }
        let step = (dist as f32 * amount).clamp(-1e6, dist as f32 - 0.05) as f64;
        self.pos += to_pivot / dist * step;
    }

    /// Begin a smooth focus so `target` fills a `radius`-metre sphere. Returns
    /// the goal pose; feed [`Self::advance_focus`] each frame until settled.
    pub fn focus_goal(&self, target: DVec3, radius: f64) -> CameraPose {
        // Back off far enough for the whole radius to fit a ~60° fov, keeping
        // the current view direction.
        let dir = self.forward().as_dvec3();
        let dist = (radius / (30f32.to_radians().tan() as f64)).max(radius * 1.5 + 0.5);
        CameraPose {
            pos: target - dir * dist,
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }

    /// Exponentially ease pos/yaw/pitch toward `goal`. Returns true once the
    /// camera has effectively arrived (caller stops advancing).
    pub fn advance_focus(&mut self, goal: &CameraPose, dt: f32) -> bool {
        let t = 1.0 - (-FOCUS_RATE * dt).exp();
        self.pos = self.pos.lerp(goal.pos, t as f64);
        self.yaw = lerp_angle(self.yaw, goal.yaw, t);
        self.pitch += (goal.pitch - self.pitch) * t;
        if self.pos.distance(goal.pos) < FOCUS_EPSILON {
            self.pos = goal.pos;
            self.yaw = goal.yaw;
            self.pitch = goal.pitch;
            return true;
        }
        false
    }

    pub fn pose(&self) -> CameraPose {
        CameraPose {
            pos: self.pos,
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }

    pub fn set_pose(&mut self, pose: CameraPose) {
        self.pos = pose.pos;
        self.yaw = pose.yaw;
        self.pitch = pose.pitch;
    }
}

/// Which Alt-navigation gesture is active this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavMode {
    #[default]
    None,
    Orbit,
    Pan,
    Dolly,
}

/// Accumulated orbit/pan/dolly input for one frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavInput {
    pub mode: NavMode,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub wheel_steps: i32,
}

/// A restorable camera pose (bookmarks, focus goals).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Nine camera bookmark slots recalled by the digit keys.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bookmarks {
    slots: [Option<CameraPose>; BOOKMARK_SLOTS],
}

impl Bookmarks {
    /// Store `pose` in slot `n` (1-based digit key). Out-of-range is a no-op.
    pub fn store(&mut self, n: usize, pose: CameraPose) {
        if (1..=BOOKMARK_SLOTS).contains(&n) {
            self.slots[n - 1] = Some(pose);
        }
    }

    /// Recall slot `n` (1-based), or `None` if empty/out-of-range.
    pub fn recall(&self, n: usize) -> Option<CameraPose> {
        if (1..=BOOKMARK_SLOTS).contains(&n) {
            self.slots[n - 1]
        } else {
            None
        }
    }
}

/// Shortest-path angular lerp (handles the ±π wrap).
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    a + d * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_clamps_at_poles() {
        let mut cam = EditorCamera::default();
        cam.apply_fly(
            &FlyInput {
                mouse_dy: -1e6,
                ..Default::default()
            },
            0.016,
        );
        assert!(cam.pitch <= PITCH_LIMIT);
        cam.apply_fly(
            &FlyInput {
                mouse_dy: 1e6,
                ..Default::default()
            },
            0.016,
        );
        assert!(cam.pitch >= -PITCH_LIMIT);
    }

    #[test]
    fn forward_moves_along_view_direction() {
        let mut cam = EditorCamera {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 10.0,
        };
        cam.apply_fly(
            &FlyInput {
                forward: true,
                ..Default::default()
            },
            0.5,
        );
        assert!((cam.pos - DVec3::new(0.0, 0.0, -5.0)).length() < 1e-6);
    }

    #[test]
    fn wheel_scales_speed_with_clamp() {
        let mut cam = EditorCamera::default();
        cam.apply_fly(
            &FlyInput {
                wheel_steps: 100,
                ..Default::default()
            },
            0.016,
        );
        assert_eq!(cam.fly_speed, FLY_SPEED_MAX);
        cam.apply_fly(
            &FlyInput {
                wheel_steps: -200,
                ..Default::default()
            },
            0.016,
        );
        assert_eq!(cam.fly_speed, FLY_SPEED_MIN);
    }

    #[test]
    fn boost_quadruples_step() {
        let base = {
            let mut cam = EditorCamera::default();
            let p0 = cam.pos;
            cam.apply_fly(
                &FlyInput {
                    forward: true,
                    ..Default::default()
                },
                0.1,
            );
            (cam.pos - p0).length()
        };
        let boosted = {
            let mut cam = EditorCamera::default();
            let p0 = cam.pos;
            cam.apply_fly(
                &FlyInput {
                    forward: true,
                    boost: true,
                    ..Default::default()
                },
                0.1,
            );
            (cam.pos - p0).length()
        };
        assert!((boosted / base - 4.0).abs() < 1e-4);
    }

    #[test]
    fn diagonal_movement_is_normalized() {
        let mut cam = EditorCamera {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 10.0,
        };
        cam.apply_fly(
            &FlyInput {
                forward: true,
                right: true,
                ..Default::default()
            },
            1.0,
        );
        assert!((cam.pos.length() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn orbit_preserves_radius_and_keeps_looking_at_pivot() {
        let mut cam = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 10.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let pivot = DVec3::ZERO;
        let r0 = (cam.pos - pivot).length();
        cam.apply_navigate(
            &NavInput {
                mode: NavMode::Orbit,
                mouse_dx: 40.0,
                mouse_dy: 15.0,
                ..Default::default()
            },
            pivot,
            0.016,
        );
        let r1 = (cam.pos - pivot).length();
        assert!((r1 - r0).abs() < 1e-4, "radius drifted {r0} -> {r1}");
        // Still aimed at the pivot.
        let dir = (pivot - cam.pos).as_vec3().normalize();
        assert!(dir.dot(cam.forward()) > 0.999);
    }

    #[test]
    fn dolly_moves_toward_pivot_without_crossing() {
        let mut cam = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 10.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let pivot = DVec3::ZERO;
        cam.dolly(0.5, pivot); // pull halfway in
        assert!((cam.pos.z - 5.0).abs() < 1e-4, "pos {:?}", cam.pos);
        // A huge dolly clamps just short of the pivot, never past it.
        cam.dolly(100.0, pivot);
        assert!(cam.pos.z > 0.0 && cam.pos.z < 0.1);
    }

    #[test]
    fn pan_scales_with_distance() {
        let mut near = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 2.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let mut far = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 40.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let input = NavInput {
            mode: NavMode::Pan,
            mouse_dx: 50.0,
            ..Default::default()
        };
        near.apply_navigate(&input, DVec3::ZERO, 0.016);
        far.apply_navigate(&input, DVec3::ZERO, 0.016);
        // Farther pivot → larger pan for the same drag.
        assert!(far.pos.x.abs() > near.pos.x.abs() * 5.0);
    }

    #[test]
    fn focus_goal_frames_target_and_advance_settles() {
        let mut cam = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 100.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let target = DVec3::new(5.0, 1.0, -3.0);
        let goal = cam.focus_goal(target, 2.0);
        // Goal keeps the target ahead and within a reasonable distance.
        let dist = (goal.pos - target).length();
        assert!((3.0..12.0).contains(&dist), "focus dist {dist}");
        // Advancing many frames converges exactly.
        let mut settled = false;
        for _ in 0..600 {
            if cam.advance_focus(&goal, 0.016) {
                settled = true;
                break;
            }
        }
        assert!(settled);
        assert_eq!(cam.pos, goal.pos);
    }

    #[test]
    fn bookmarks_store_and_recall() {
        let mut bm = Bookmarks::default();
        let cam = EditorCamera::default();
        assert!(bm.recall(3).is_none());
        bm.store(3, cam.pose());
        assert_eq!(bm.recall(3).unwrap(), cam.pose());
        // Out-of-range slots are ignored, not panicking.
        bm.store(0, cam.pose());
        bm.store(99, cam.pose());
        assert!(bm.recall(0).is_none());
        assert!(bm.recall(99).is_none());
    }

    #[test]
    fn lerp_angle_takes_short_way_around_wrap() {
        // From +170° to -170° is +20° across the seam, not -340°.
        let a = 170f32.to_radians();
        let b = (-170f32).to_radians();
        let mid = lerp_angle(a, b, 0.5);
        // Halfway should be at ±180°, not near 0.
        assert!(mid.abs() > 175f32.to_radians());
    }
}
