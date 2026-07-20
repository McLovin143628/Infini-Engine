//! Material / texture errors.

/// Errors from texture decode / import.
#[derive(Debug, thiserror::Error)]
pub enum MaterialError {
    #[error("image: {0}")]
    Image(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
