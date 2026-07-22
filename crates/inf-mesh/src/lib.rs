//! Mesh import processing and optimization (later home of the meshlet builder).
//!
//! Owns the `.inf_mesh` schema ([`MeshAsset`]), the glTF importer
//! ([`import_gltf`]) and the Wavefront OBJ importer ([`import_obj`]) — both of
//! which turn an external document into geometry + material/texture descriptors
//! in the shared [`GltfImport`] container — and the `meshopt` post-process
//! ([`optimize`]).

pub mod asset;
pub mod error;
pub mod gltf_import;
pub mod obj_import;
pub mod optimize;

pub use asset::{Aabb, MeshAsset, MeshVertex, SubMesh, VertexSkin};
pub use error::MeshError;
pub use gltf_import::{
    import_gltf, GltfImport, ImportedClip, ImportedMaterial, ImportedMesh, ImportedSkeleton,
    RawImage,
};
pub use obj_import::import_obj;
pub use optimize::optimize;
