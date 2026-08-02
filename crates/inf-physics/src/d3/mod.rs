//! The 3D physics facade: [`PhysicsWorld3D`] plus its bodies, colliders, contact
//! events, scene queries, and kinematic character mover — the `d3` sibling of
//! [`crate::d2`].
//!
//! Everything rapier is sealed inside this module. The public surface speaks
//! `glam::DVec3` / `glam::DQuat` and the opaque handle newtypes [`BodyId3D`] /
//! [`ColliderId3D`]; a rapier handle never escapes the facade. The shape mirrors
//! `d2` exactly (see that module) at three dimensions — same bodies, colliders,
//! drain-style canonicalized events, deterministically-ordered queries, and the
//! same determinism discipline (`enhanced-determinism`, rapier's `parallel` off,
//! sorted output).

use rapier3d_f64::prelude::{ColliderHandle, RigidBodyHandle};

mod character;
mod ecs;
mod events;
mod joint;
mod query;
mod world;

pub use character::{CharacterMove3D, CharacterMover3D};
pub use ecs::{
    pcg_structure_guid, BodyDesc3D, EntitySync3D, JointSync3D, PhysicsBridge3D, PoseWriteback3D,
};
pub use events::ContactEvent3D;
pub use joint::{JointDesc3D, JointId3D, JointKind3D, JointMotor3D};
pub use query::RayHit3D;
pub use world::{BodyKind3D, ColliderDesc3D, ColliderShape3D, PhysicsWorld3D};

// `ContactPhase` (Started/Stopped) is dimension-agnostic, so `d3` reuses the one
// `d2` defines rather than duplicating an identical enum — this is the single
// deliberate spot the two halves share a type instead of mirroring it. It keeps
// one `ContactPhase` at the crate root (hoisted from `d2`) with no name clash.
pub use crate::d2::ContactPhase;

/// Opaque, stable handle to a rigid body in a [`PhysicsWorld3D`].
///
/// Wraps rapier's generational `RigidBodyHandle` so the concrete rapier type
/// never leaks. It stays valid until the body is destroyed; a destroyed handle is
/// never silently reused (rapier bumps the generation), so a stale `BodyId3D`
/// reads back as "not found" rather than aliasing a new body.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId3D(pub(crate) RigidBodyHandle);

/// Opaque, stable handle to a collider in a [`PhysicsWorld3D`]. See [`BodyId3D`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderId3D(pub(crate) ColliderHandle);

impl core::fmt::Debug for BodyId3D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (i, g) = self.0.into_raw_parts();
        write!(f, "BodyId3D({i}v{g})")
    }
}

impl core::fmt::Debug for ColliderId3D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (i, g) = self.0.into_raw_parts();
        write!(f, "ColliderId3D({i}v{g})")
    }
}

// Deterministic ordering by (index, generation) — rapier's handles are `Eq` but
// not `Ord`, and the facade sorts every returned collection by handle so its
// iteration order never depends on rapier's internal arena layout.
impl Ord for BodyId3D {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.into_raw_parts().cmp(&other.0.into_raw_parts())
    }
}
impl PartialOrd for BodyId3D {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ColliderId3D {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.into_raw_parts().cmp(&other.0.into_raw_parts())
    }
}
impl PartialOrd for ColliderId3D {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
