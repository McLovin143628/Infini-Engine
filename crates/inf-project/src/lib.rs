//! Project model: the `inf.toml` manifest, project templates + scaffolding, and
//! the recent-projects list.
//!
//! A lightweight Ring-0 crate shared by the editor (open/create/recent), the CLI
//! (`inf new`), and the runtime (locate a project's content). No engine-heavy
//! deps, so the CLI stays small.

pub mod error;
pub mod manifest;
pub mod project;
pub mod recent;
pub mod template;

pub use error::{ProjectError, Result};
pub use manifest::{
    anim_blend_name, anim_blend_wire, ProjectManifest, ANIM_BLEND_CROSSFADE,
    ANIM_BLEND_INERTIALIZE, PROJECT_FILE, SCHEMA_VERSION,
};
pub use project::Project;
pub use recent::{RecentProject, RecentProjects};
pub use template::{scaffold, ProjectTemplate, Scaffolded, SCRIPTS_DIR};
