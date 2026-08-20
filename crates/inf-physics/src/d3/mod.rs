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

// P29.6 the locomotion camera: the fixed-step door both hosts call, and
// `cast_shape`'s third consumer (Ruling 3).
pub mod camera;
mod character;
mod ecs;
mod events;
pub mod fracture;
mod joint;
pub mod movement;
mod query;
// P29.4 the ragdoll bridge: the physics half of the anim<->physics handoff.
pub mod ragdoll_bridge;
// P29.4 traversal detectors: the ledge probe and the land-prediction sweep.
pub mod traversal;
// P29.7 vehicles: the wheel rays, the forces and the wheel write-back.
pub mod vehicle;
pub mod water;
mod world;

pub use camera::step_locomotion_camera;
pub use character::{AutoStep3D, CharacterMove3D, CharacterMover3D};
pub use ecs::{
    pcg_shell_guid, pcg_structure_guid, terrain_tile_collider, terrain_tile_guid, voxel_chunk_guid,
    BodyDesc3D, EntitySync3D, JointSync3D, PhysicsBridge3D, PoseWriteback3D, TerrainColliderAudit,
    DEBRIS_LAYER,
};
pub use events::ContactEvent3D;
pub use fracture::{
    first_uncollidable_chunk, fracture_chunk_guid, resolve_fracture_states, ChunkState,
    DamageReport, DebrisBudget, DestroyedEvent, DestructOutcome, FractureAudit, FractureState,
    CRACK_OPENING_M, DEFAULT_DEBRIS_LIFETIME_S, DEFAULT_DEBRIS_MAX_LIVE,
};
pub use joint::{JointDesc3D, JointId3D, JointKind3D, JointMotor3D};
pub use movement::{mover_for, step_character_movement, MoveOutcome};
pub use query::{CastTargets, RayHit3D, ShapeHit3D};
pub use ragdoll_bridge::{
    blend_weight as ragdoll_blend_weight, start_ragdoll, step_ragdoll, SpawnedRagdoll,
};
pub use traversal::{
    is_walkable, predict_landing, probe_ledge, LandPrediction, Ledge, LedgeSettings,
};
pub use vehicle::{step_vehicles, VehicleOutcome};
pub use water::{
    BuoyancyDesc3D, SampleGeometry, WaterEvent3D, WaterEventKind3D, WaterIndex, WaterProbe,
};
pub use world::{
    convex_hull_is_buildable, BodyKind3D, ColliderDesc3D, ColliderShape3D, PhysicsWorld3D,
};

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
