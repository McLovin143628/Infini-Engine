//! Mesh import errors.

/// Errors from mesh import / processing.
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("gltf: {0}")]
    Gltf(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The source document parsed, and then failed the import door
    /// ([`crate::validate`]): a non-finite number, an index outside its buffer,
    /// or two streams that disagree about how many elements they describe.
    ///
    /// Separate from [`Gltf`](Self::Gltf) on purpose. That variant means "this
    /// file is not the format it claims"; this one means "this file is the
    /// format it claims and the format's own rules are broken", which is the
    /// difference between a user who picked the wrong file and a user whose
    /// exporter is producing rubbish.
    #[error("malformed source: {0}")]
    Malformed(String),
}
