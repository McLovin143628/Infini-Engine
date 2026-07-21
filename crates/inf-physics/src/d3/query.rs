//! Scene-query result types. The `d3` mirror of [`crate::d2`]'s query types.

use glam::DVec3;

use super::ColliderId3D;

/// The result of a successful ray cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit3D {
    /// The collider the ray hit.
    pub collider: ColliderId3D,
    /// The world-space point of impact.
    pub point: DVec3,
    /// The surface normal at the impact point.
    pub normal: DVec3,
    /// The distance along the (normalized) ray direction to the impact — i.e. the
    /// time of impact, in world units.
    pub toi: f64,
}
