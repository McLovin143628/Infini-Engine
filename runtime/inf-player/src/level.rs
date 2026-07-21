//! Level loading seams (P9.3 item 1).
//!
//! The player loads a world through two narrow traits so the pieces built by the
//! concurrent P9.2 tasks (the pack format + the runtime `.inf_lvl` decode in
//! `inf-scene`) slot in without touching the rest of the player:
//!
//! * [`LevelSource`] — produces the **raw serialized level bytes**. v1 is
//!   [`DevDirLevelSource`] (reads an `.inf_lvl` file from disk); the pack-backed
//!   implementation is next batch's ~20 lines (open the pack, return the level
//!   entry's bytes).
//! * [`WorldBuilder`] — **decodes those bytes into a populated [`BuiltWorld`]**
//!   (an ECS world + the blueprint actors to tick + gravity/rate). The real
//!   implementation is the P9.2 runtime reader; until it lands
//!   [`StubWorldBuilder`] returns a clear error and the player runs the
//!   programmatic `--demo` world ([`crate::demo`]) instead.
//!
//! ## The one-call wiring point for P9.2
//!
//! When `inf-scene`'s runtime decode lands, add a `SceneWorldBuilder` here whose
//! `build` does roughly:
//!
//! ```ignore
//! let doc = inf_scene::decode_level(level_bytes)?;      // ECS world snapshot
//! let mut world = doc.into_ecs_world();                 // populated EcsWorld
//! let actors = resolve_actors(&world, content_dir)?;    // (Guid, BlueprintClass)
//! Ok(BuiltWorld { world, actors, gravity, hz })
//! ```
//!
//! and point [`load`] at it instead of [`StubWorldBuilder`]. Nothing else in the
//! player changes: the window, renderer, fixed-step loop, input, and
//! [`runtime_sim`](crate::runtime_sim) are all already real.

use std::path::{Path, PathBuf};

use glam::DVec2;
use uuid::Uuid;

use inf_blueprint::BlueprintClass;
use inf_ecs::EcsWorld;

/// A populated world ready to hand to [`RuntimeSim`](crate::runtime_sim::RuntimeSim).
pub struct BuiltWorld {
    /// The ECS world (entities + components).
    pub world: EcsWorld,
    /// The blueprint actors to tick: `(entity Guid, class)`.
    pub actors: Vec<(Uuid, BlueprintClass)>,
    /// World gravity for the 2D physics bridge (the platformer uses `ZERO`).
    pub gravity: DVec2,
    /// Fixed update rate (Hz).
    pub hz: f64,
    /// A human label for logs / the window title.
    pub label: String,
}

/// Produces the raw serialized bytes of a level (an `.inf_lvl` payload). The
/// dev-dir implementation reads a file; the pack-backed one (P9.2) reads a pack
/// entry.
pub trait LevelSource {
    /// The level's raw bytes.
    fn level_bytes(&self) -> Result<Vec<u8>, String>;
    /// A human label for logs / the window title.
    fn label(&self) -> String;
}

/// Decodes raw level bytes into a populated [`BuiltWorld`]. The runtime reader
/// (`inf-scene`, P9.2) provides the real implementation.
pub trait WorldBuilder {
    fn build(&self, level_bytes: &[u8]) -> Result<BuiltWorld, String>;
}

/// v1 [`LevelSource`]: read an `.inf_lvl` file straight off disk.
pub struct DevDirLevelSource {
    path: PathBuf,
}

impl DevDirLevelSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl LevelSource for DevDirLevelSource {
    fn level_bytes(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.path).map_err(|e| format!("read level {}: {e}", self.path.display()))
    }

    fn label(&self) -> String {
        self.path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("level")
            .to_string()
    }
}

/// The placeholder [`WorldBuilder`] until the P9.2 runtime `.inf_lvl` reader
/// lands. Reading bytes is real (via [`DevDirLevelSource`]); decoding them into
/// an ECS world is the ~20-line handoff — this stub makes the gap explicit.
pub struct StubWorldBuilder;

impl WorldBuilder for StubWorldBuilder {
    fn build(&self, _level_bytes: &[u8]) -> Result<BuiltWorld, String> {
        Err(
            "runtime level reader lands with P9.2 (inf-scene .inf_lvl decode); \
             run with --demo until then"
                .to_string(),
        )
    }
}

/// Load a world by piping a [`LevelSource`]'s bytes through a [`WorldBuilder`].
/// This is the single seam P9.2 rewires (swap [`StubWorldBuilder`] for the
/// inf-scene-backed builder at the call site in `lib.rs`).
pub fn load(source: &dyn LevelSource, builder: &dyn WorldBuilder) -> Result<BuiltWorld, String> {
    let bytes = source.level_bytes()?;
    tracing::info!(
        "inf-player: loaded {} level byte(s) from '{}'",
        bytes.len(),
        source.label()
    );
    builder.build(&bytes)
}
