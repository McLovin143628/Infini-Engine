//! Project errors.

/// Errors from project open / create / manifest parsing.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("manifest parse: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("manifest write: {0}")]
    Write(#[from] toml::ser::Error),

    #[error("no {0} found (not an Infini Engine project)")]
    NoManifest(String),

    #[error("project schema v{found} is newer than this editor (v{current})")]
    SchemaTooNew { found: u32, current: u32 },

    #[error("{0}")]
    Other(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, ProjectError>;
