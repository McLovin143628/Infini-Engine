//! Cook errors.

use inf_asset::AssetId;

/// A failure during [`cook`](crate::cook).
#[derive(Debug, thiserror::Error)]
pub enum CookError {
    #[error("project: {0}")]
    Project(#[from] inf_project::ProjectError),

    #[error("asset: {0}")]
    Asset(#[from] inf_asset::AssetError),

    #[error("level {guid}: {source}")]
    Scene {
        guid: AssetId,
        #[source]
        source: inf_scene::SceneError,
    },

    /// A blueprint asset failed to decode or validate. Anchored to the class and
    /// the handler/function where the problem lives (the `.inf_act` stores lowered
    /// IR, not the visual graph, so this is the finest anchor available at cook —
    /// re-associating it with a graph node is a follow-up gated on persisting the
    /// authoring graph in the `.inf_act`).
    #[error("blueprint {guid} (`{class}`) / {handler}: {message}")]
    Blueprint {
        guid: AssetId,
        class: String,
        handler: String,
        message: String,
    },

    /// An explicit `--roots` GUID is not in the project's asset database.
    #[error("root asset {0} is not in the project")]
    UnknownRoot(AssetId),

    /// A mesh asset failed to decode while deriving its virtualized-geometry
    /// (`.inf_vmesh`) form.
    #[error("mesh {guid}: {message}")]
    Mesh { guid: AssetId, message: String },

    /// A `.inf_terrain` failed its structural check at cook (P16.3).
    ///
    /// The runtime pages tiles out of this payload by **trusting the header and
    /// directory it validated once** — a truncated, overlapping, misaligned or
    /// accidentally bincode-framed asset must therefore fail the BUILD, never
    /// reach a shipped player. Named like the blueprint error for the same reason:
    /// the cook is where a broken asset is cheap to fix.
    #[error("terrain {guid}: {message}")]
    Terrain { guid: AssetId, message: String },

    /// A level's world partition could not be built (P16.5).
    ///
    /// A partitioned level ships **no entities of its own** — they live in the
    /// derived `.inf_part` — so a failure here cannot degrade to "cook it
    /// unpartitioned": that would silently ship an empty world. It fails the
    /// build, where it is cheap.
    #[error("partition of level {guid}: {message}")]
    Partition { guid: AssetId, message: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("manifest toml: {0}")]
    Toml(#[from] toml::ser::Error),

    /// A desktop-export (bundle) step failed: locating/copying the player binary,
    /// or writing the launch config.
    #[error("export: {0}")]
    Export(String),

    /// A WASM mod cook step failed (transpile, crate generation, or wasm build).
    #[error("mod: {0}")]
    Mod(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, CookError>;
