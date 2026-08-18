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

/// The result of a successful **shape cast** — a swept-volume query (P29.3).
///
/// The `d3` sibling of [`RayHit3D`], and deliberately the same vocabulary: a
/// collider, a world point, a world normal, and a distance along the (normalized)
/// sweep direction. A shape cast is a ray cast with a body, and a caller that
/// already reads a `RayHit3D` should not have to learn a second shape of answer.
///
/// The one field a ray has no need of is [`started_penetrating`](Self::started_penetrating).
/// A ray either hits or does not; a swept *volume* can begin its sweep already
/// overlapping something, and that is not the same answer as "hit at distance 0".
/// It is the answer a clearance probe cares about most — "you cannot stand up
/// here because you are *already* inside the ceiling" — and parry reports its
/// witness point and normal as unreliable in that case (`ShapeCastStatus::
/// PenetratingOrWithinTargetDist`), so a caller that reads `normal` without
/// reading this flag is reading a number parry does not stand behind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeHit3D {
    /// The collider the swept shape hit.
    pub collider: ColliderId3D,
    /// The world-space witness point on the **hit collider** at the time of
    /// impact. Unreliable when [`started_penetrating`](Self::started_penetrating).
    pub point: DVec3,
    /// The world-space outward normal of the **hit collider** at the impact.
    /// Unreliable when [`started_penetrating`](Self::started_penetrating).
    pub normal: DVec3,
    /// Distance travelled along the (normalized) sweep direction before contact,
    /// in world units. `0.0` when the sweep started in contact.
    pub toi: f64,
    /// The swept shape was **already overlapping** this collider at the start of
    /// the sweep (parry's `PenetratingOrWithinTargetDist`). `point`/`normal` are
    /// not to be trusted; `toi` is `0.0`.
    pub started_penetrating: bool,
}
