//! Materials & textures.
//!
//! Phase 4 scope: the data model + import pipeline —
//!   * [`TextureAsset`] (`.inf_tex`): image decode, mip generation, and
//!     hand-rolled BC1/BC3 block compression ([`bc`]);
//!   * [`MaterialAsset`] (`.inf_mat`): the PBR metallic-roughness parameter
//!     block + texture GUID references.
//!
//! Phase 7 adds the material **node graph → WGSL** codegen ([`graph`]): a
//! `.inf_mat` is authored as a pure node DAG over the shared `inf_graph`
//! substrate and compiled to a naga-validated `material_surface` WGSL function.

pub mod bc;
pub mod error;
pub mod graph;
pub mod instance;
pub mod material;
pub mod texture;
pub mod tiles;

pub use bc::{decode_bc1, decode_bc3};
pub use error::MaterialError;
pub use graph::{
    analyze_complexity, analyze_with_budget, emit_texture_compute, emit_wgsl, material_registry,
    validate_module, ComplexityReport, MatIssue, MatType, MaterialBudget, MaterialCompile,
    TextureCompile,
};
pub use instance::{MatOverrides, MaterialInstance};
pub use material::{MatBlend, MaterialAsset};
pub use texture::{
    import_texture_bytes, texture_from_rgba8, TextureAsset, TextureCompression, TextureFormat,
    TextureImportSettings, TextureMip,
};
pub use tiles::{
    build_tiled_texture, decode_texture_payload, lift_texture_asset, stored_tile_bytes, TileCoord,
    TiledTextureError, TiledTextureImage, TiledTextureReader, TiledTextureView, STORED_TILE_SIZE,
    TEX_ASSET_MAGIC, TEX_ASSET_SCHEMA_VERSION, TILE_BORDER, TILE_SIZE,
};
