//! Materials & textures.
//!
//! Phase 4 scope: the data model + import pipeline —
//!   * [`TextureAsset`] (`.inf_tex`): image decode, mip generation, and
//!     hand-rolled BC1/BC3 block compression ([`bc`]);
//!   * [`MaterialAsset`] (`.inf_mat`): the PBR metallic-roughness parameter
//!     block + texture GUID references.
//!
//! The material node graph → WGSL codegen and `.inf_tex` compute graphs are
//! Phase 7; this crate is the foundation they build on.

pub mod bc;
pub mod error;
pub mod material;
pub mod texture;

pub use error::MaterialError;
pub use material::MaterialAsset;
pub use texture::{
    import_texture_bytes, texture_from_rgba8, TextureAsset, TextureCompression, TextureFormat,
    TextureImportSettings, TextureMip,
};
