//! Mesh import errors.

/// Errors from mesh import / processing.
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("gltf: {0}")]
    Gltf(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
