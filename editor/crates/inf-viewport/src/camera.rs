//! Editor camera controller: pure input→state math, platform-neutral and
//! unit-tested headless (the per-OS hosts only feed it input deltas).
//!
//! Phase 2.2 scope: UE-style flycam (RMB captured: mouse look + WASD/QE fly,
//! wheel scales speed, Shift boosts). Orbit/pan/dolly, focus and bookmarks
//! land with P2.3.

use glam::{DVec3, Vec3};

/// Radians per raw mouse count.
const LOOK_SENSITIVITY: f32 = 0.0032;
/// Pitch stays just short of the poles to keep `forward` well-defined.
const PITCH_LIMIT: f32 = 1.55;
pub const FLY_SPEED_MIN: f32 = 0.2;
pub const FLY_SPEED_MAX: f32 = 250.0;

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
}
