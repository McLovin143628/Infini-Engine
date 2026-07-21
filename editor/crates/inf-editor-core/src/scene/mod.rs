//! Scene document: the authoritative ECS world plus editor state.
//!
//! [`SceneDoc`] (P3.2) owns the world, selection, and versioning; [`delta`]
//! diffs snapshots for the `world://delta` event; [`demo`] builds the default
//! primitive+lights scene. Reflection-driven Details (P3.3) live in [`props`],
//! undo (P3.4) in [`super::undo`], and `.inf_lvl` serde (P3.5) in [`serialize`].

pub mod delta;
pub mod demo;
pub mod details;
pub mod doc;
pub mod serialize;
pub mod tilemap;
pub mod undo;

pub use delta::diff;
pub use doc::SceneDoc;

impl SceneDoc {
    /// A fresh document pre-loaded with the default scene.
    pub fn with_demo() -> Self {
        let mut doc = Self::new();
        demo::build(&mut doc);
        doc
    }
}
