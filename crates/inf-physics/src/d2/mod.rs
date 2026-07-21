//! The 2D physics facade: [`PhysicsWorld2D`] plus its bodies, colliders, contact
//! events, scene queries, and kinematic character mover.
//!
//! Everything rapier is sealed inside this module. The public surface speaks
//! `glam::DVec2` and the opaque handle newtypes [`BodyId`] / [`ColliderId`]; a
//! rapier handle never escapes the facade.

use rapier2d_f64::prelude::{ColliderHandle, RigidBodyHandle};

mod character;
mod events;
mod query;
mod world;

pub use character::{CharacterMove2D, CharacterMover2D};
pub use events::{ContactEvent2D, ContactPhase};
pub use query::RayHit2D;
pub use world::{BodyKind, ColliderDesc2D, ColliderShape2D, PhysicsWorld2D};

/// Opaque, stable handle to a rigid body in a [`PhysicsWorld2D`].
///
/// Wraps rapier's generational `RigidBodyHandle` so the concrete rapier type
/// never leaks. It stays valid until the body is destroyed; a destroyed handle is
/// never silently reused (rapier bumps the generation), so a stale `BodyId` reads
/// back as "not found" rather than aliasing a new body.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId(pub(crate) RigidBodyHandle);

/// Opaque, stable handle to a collider in a [`PhysicsWorld2D`]. See [`BodyId`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderId(pub(crate) ColliderHandle);

impl core::fmt::Debug for BodyId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (i, g) = self.0.into_raw_parts();
        write!(f, "BodyId({i}v{g})")
    }
}

impl core::fmt::Debug for ColliderId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (i, g) = self.0.into_raw_parts();
        write!(f, "ColliderId({i}v{g})")
    }
}

// Deterministic ordering by (index, generation) — rapier's handles are `Eq` but
// not `Ord`, and the facade sorts every returned collection by handle so its
// iteration order never depends on rapier's internal arena layout.
impl Ord for BodyId {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.into_raw_parts().cmp(&other.0.into_raw_parts())
    }
}
impl PartialOrd for BodyId {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ColliderId {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.into_raw_parts().cmp(&other.0.into_raw_parts())
    }
}
impl PartialOrd for ColliderId {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
