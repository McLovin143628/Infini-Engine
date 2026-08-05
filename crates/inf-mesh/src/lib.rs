//! Mesh import processing and optimization (later home of the meshlet builder).
//!
//! Owns the `.inf_mesh` schema ([`MeshAsset`]), the glTF importer
//! ([`import_gltf`]) and the Wavefront OBJ importer ([`import_obj`]) — both of
//! which turn an external document into geometry + material/texture descriptors
//! in the shared [`GltfImport`] container — and the `meshopt` post-process
//! ([`optimize`]), and the cook-time Voronoi pre-fracture pipeline
//! ([`fracture`]) that derives a mesh's `.inf_fracture` chunk set.

pub mod asset;
pub mod error;
pub mod fracture;
#[cfg(not(target_arch = "wasm32"))]
pub mod gltf_import;
#[cfg(not(target_arch = "wasm32"))]
pub mod obj_import;
#[cfg(not(target_arch = "wasm32"))]
pub mod optimize;

pub use asset::{Aabb, MeshAsset, MeshVertex, SubMesh, VertexSkin};
pub use error::MeshError;
pub use fracture::{
    clamp_chunk_count, derived_fracture_id, fracture_mesh, ChunkSection, FractureAsset,
    FractureChunk, FractureParams, FractureSkip, DEFAULT_CHUNK_COUNT, FRACTURE_ID_SALT,
    MAX_CHUNK_COUNT, MIN_CHUNK_COUNT,
};
#[cfg(not(target_arch = "wasm32"))]
pub use gltf_import::{
    import_gltf, GltfImport, ImportedClip, ImportedMaterial, ImportedMesh, ImportedSkeleton,
    RawImage,
};
#[cfg(not(target_arch = "wasm32"))]
pub use obj_import::import_obj;
#[cfg(not(target_arch = "wasm32"))]
pub use optimize::optimize;
