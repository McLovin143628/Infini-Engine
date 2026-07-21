//! Scene-query result types.

use glam::DVec2;

use super::ColliderId;

/// The result of a successful ray cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit2D {
    /// The collider the ray hit.
    pub collider: ColliderId,
    /// The world-space point of impact.
    pub point: DVec2,
    /// The surface normal at the impact point.
    pub normal: DVec2,
    /// The distance along the (normalized) ray direction to the impact — i.e. the
    /// time of impact, in world units.
    pub toi: f64,
}
