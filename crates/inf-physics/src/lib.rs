//! Infinity Engine physics facade (Ring 0).
//!
//! This crate wraps [rapier](https://rapier.rs) behind a small, engine-flavoured
//! API so the rest of the engine never names rapier/parry/nalgebra types
//! directly. Rapier handles are re-exported only as opaque newtypes
//! ([`d2::BodyId`], [`d2::ColliderId`]); the public vocabulary is `glam::DVec2`,
//! `f64`, and plain data structs.
//!
//! # Why `rapier2d-f64` (f64, not the f32 default)
//!
//! Architecture rule 3 (see `docs/ROADMAP.md`, `CLAUDE.md`): **world-space is
//! f64.** A 2D physics world that simulated in f32 would re-introduce exactly the
//! precision cliff floating-origin was built to avoid — at ~4 km from the origin
//! an f32 coordinate already jitters past a millimetre. `rapier2d-f64` simulates
//! in f64 throughout, so a large 2D level stays stable without a physics-only
//! rebase hack. A happy consequence of the `-f64` build on rapier 0.34: its math
//! types resolve to **glam 0.33** (rapier → parry2d-f64 → glamx → glam `0.33`,
//! our workspace pin), so there is no second `glam` in the dependency tree and a
//! rapier `Vector` *is* a `glam::DVec2`. The facade exposes `glam::DVec2` and the
//! conversion to/from rapier is the identity.
//!
//! The 3D half ([`d3`], P9.1b) wraps `rapier3d-f64` the same way in a sibling
//! module; the 2D types are all `*2D`-suffixed and the 3D types `*3D`-suffixed so
//! both halves hoist to the crate root without a name clash. The one shared type
//! is [`ContactPhase`] (Started/Stopped is dimension-agnostic).
//!
//! # Determinism (a product guarantee — §8, P12.2 replay harness)
//!
//! Determinism here means: *the same sequence of facade calls produces the same
//! bytes on every run and every machine.* It is enforced by construction:
//!
//! * **`enhanced-determinism`** is enabled on `rapier2d-f64`. It routes rapier's
//!   transcendental math through `libm` (via `simba/libm_force`) so results do
//!   not depend on the host FPU/compiler, satisfying rapier's IEEE-754 cross-
//!   platform reproducibility contract.
//! * **No wall clock, anywhere.** [`d2::PhysicsWorld2D::step`] takes `dt` as an
//!   argument and never reads `Instant::now`. Callers derive `dt` from the fixed
//!   timestep via [`FixedStepper`], never from elapsed real time inside a step.
//! * **rapier's own parallelism is off.** The `parallel` feature (rapier's rayon
//!   fan-out) is not enabled, so a step is single-threaded and its result cannot
//!   depend on thread scheduling. This also means rapier never spawns a worker
//!   pool that would fight `inf-core`'s job system. Cross-body parallelism, when
//!   we want it, will come from the deterministic ECS schedule (P9.1) driving
//!   *independent* physics worlds, not from rapier's internal threads.
//! * **Deterministic ordering out.** Every collection the facade returns —
//!   contact events, query hits — is sorted by stable handle identity before it
//!   leaves the crate, so iteration order never leaks rapier's internal arena
//!   layout.
//!
//! # Layout
//!
//! * [`d2`] — the 2D physics world, bodies, colliders, events, queries, the
//!   kinematic character mover, and [`d2::PhysicsBridge2D`] (the ECS↔physics
//!   sync). The bridge is why this crate depends on `inf-ecs`: the dependency
//!   direction is **inf-physics → inf-ecs** (both Ring 0), and inf-ecs never
//!   names physics, so there is no cycle.
//! * [`FixedStepper`] — a pure fixed-timestep accumulator shared by every caller
//!   (editor Simulate today, `inf-runtime` in P9) so there is one correct
//!   implementation of the accumulate/interpolate loop.

pub mod d2;
pub mod d3;
mod filtering;
mod stepper;

pub use filtering::{CollisionLayers, CombineRule};
pub use stepper::FixedStepper;

// The pure ragdoll-setup helper (P12.1): a humanoid skeleton → body/collider/
// joint descriptors the caller (or a future editor tool) can spawn.
pub mod ragdoll;

// Ergonomic re-exports of the 2D facade at the crate root. These are safe to hoist
// because every 2D type is `*2D`/`*2`-suffixed or clearly 2D-named; the `d3`
// module keeps its own `*3D` names, so nothing here collides.
pub use d2::{
    BodyId, BodyKind, CharacterMove2D, CharacterMover2D, ColliderDesc2D, ColliderId,
    ColliderShape2D, ContactEvent2D, ContactPhase, PhysicsBridge2D, PhysicsWorld2D, RayHit2D,
};

// Ergonomic re-exports of the 3D facade at the crate root (all `*3D`-suffixed).
// `ContactPhase` is shared with d2 and hoisted once, above.
pub use d3::{
    BodyDesc3D, BodyId3D, BodyKind3D, CharacterMove3D, CharacterMover3D, ColliderDesc3D,
    ColliderId3D, ColliderShape3D, ContactEvent3D, EntitySync3D, JointDesc3D, JointId3D,
    JointKind3D, JointMotor3D, PhysicsBridge3D, PhysicsWorld3D, PoseWriteback3D, RayHit3D,
};

// Ergonomic re-exports of the 2D joint facade (`*2D`-suffixed).
pub use d2::{JointDesc2D, JointId2D, JointKind2D, JointMotor2D};
