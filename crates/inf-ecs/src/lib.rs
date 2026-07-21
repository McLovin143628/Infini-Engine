//! ECS facade over bevy_ecs + bevy_reflect — the only crate that names them.
//!
//! Phase 3 (ECS & scene model) foundation. The editor's scene is a bevy
//! `World` of reflected components, wrapped by [`EcsWorld`] which adds a stable
//! `Guid` index, a transform hierarchy, and dirty-gated propagation. Everything
//! the editor UI needs — snapshotting (P3.2), reflection-driven Details (P3.3),
//! undo (P3.4), and `.inf_lvl` serde (P3.5) — is built on this facade.
//!
//! Reflection note: `bevy_reflect` is used **without** its `glam` feature (that
//! pins glam 0.32, clashing with the renderer's 0.33). Editable math is stored
//! in f64 value types ([`math::Vec3d`]) that derive `Reflect` + `serde` and
//! convert to glam for computation — see [`math`].

pub mod components;
pub mod hierarchy;
pub mod math;
pub mod props;
pub mod registry;
pub mod transform;
pub mod world;

pub use bevy_ecs::prelude::Entity;
pub use components::{
    AtlasRect, Camera, ComputedVisibility, GlobalTransform, Guid, Light, LightKind, Material,
    MeshRef, Name, Primitive, Sprite, TileBounds, TileChunk, Tilemap, Transform, Visibility,
    CHUNK_DIM, CHUNK_TILES,
};
pub use hierarchy::{ChildOf, Children};
pub use math::{Color, Vec2d, Vec3d};
pub use props::{default_field, ComponentProps, PropField, PropValue};
pub use registry::{ComponentInfo, ComponentRegistry};
pub use world::EcsWorld;
