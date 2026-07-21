//! Project templates + scaffolding.
//!
//! `inf new` (and the editor's New Project dialog) scaffold a project directory
//! from a template: a real user cargo crate (where hand-written gameplay + the
//! P6-generated Rust will live) plus the `inf.toml` manifest and the Content /
//! Levels roots. The three templates share the same skeleton for now and differ
//! by their starter `lib.rs`; richer per-discipline content lands with the
//! sample games (P15).

use std::path::{Path, PathBuf};

use crate::error::{ProjectError, Result};
use crate::manifest::ProjectManifest;

/// A first-run project template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectTemplate {
    /// Empty 3D scene.
    Blank3d,
    /// 2D side-scroller starter.
    Platformer2d,
    /// First-person starter.
    FirstPerson,
}

impl ProjectTemplate {
    pub fn slug(self) -> &'static str {
        match self {
            ProjectTemplate::Blank3d => "blank-3d",
            ProjectTemplate::Platformer2d => "2d-platformer",
            ProjectTemplate::FirstPerson => "first-person",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ProjectTemplate::Blank3d => "Blank 3D",
            ProjectTemplate::Platformer2d => "2D Platformer",
            ProjectTemplate::FirstPerson => "First Person",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ProjectTemplate::Blank3d => "An empty 3D scene — a clean slate.",
            ProjectTemplate::Platformer2d => "A 2D side-scroller starter.",
            ProjectTemplate::FirstPerson => "A first-person starter.",
        }
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::all().iter().copied().find(|t| t.slug() == slug)
    }

    pub fn all() -> &'static [ProjectTemplate] {
        &[
            ProjectTemplate::Blank3d,
            ProjectTemplate::Platformer2d,
            ProjectTemplate::FirstPerson,
        ]
    }

    fn starter_lib(self, crate_name: &str) -> String {
        let doc = match self {
            ProjectTemplate::Blank3d => "an empty 3D scene",
            ProjectTemplate::Platformer2d => "a 2D platformer",
            ProjectTemplate::FirstPerson => "a first-person game",
        };
        format!(
            "//! `{crate_name}` — gameplay for {doc}.\n\
             //!\n\
             //! Hand-written systems live here; Infinity Blueprints (Phase 6) generate\n\
             //! additional Rust into this crate and stay in sync.\n\
             \n\
             /// Called once when play begins.\n\
             pub fn begin_play() {{\n    \
                 // Your setup here.\n\
             }}\n\
             \n\
             /// Called every frame with the delta time in seconds.\n\
             pub fn tick(_dt: f32) {{\n    \
                 // Your per-frame logic here.\n\
             }}\n"
        )
    }
}

/// The result of scaffolding: where the project was created.
#[derive(Debug, Clone)]
pub struct Scaffolded {
    pub root: PathBuf,
    pub crate_name: String,
    pub manifest: ProjectManifest,
}

/// Scaffold a new project named `name` from `template` inside `parent`. The
/// project directory is `parent/<sanitized-name>`; it must not already exist
/// (or must be empty). Returns the created root.
pub fn scaffold(parent: &Path, name: &str, template: ProjectTemplate) -> Result<Scaffolded> {
    let dir_name = sanitize_dir(name);
    let crate_name = sanitize_crate(name);
    let root = parent.join(&dir_name);

    if root.exists() && std::fs::read_dir(&root)?.next().is_some() {
        return Err(ProjectError::Other(format!(
            "{} already exists and is not empty",
            root.display()
        )));
    }

    let manifest = ProjectManifest::new(name, template.slug());
    manifest.save(&root)?;

    // User cargo crate.
    write(&root.join("Cargo.toml"), &cargo_toml(&crate_name))?;
    write(&root.join("src/lib.rs"), &template.starter_lib(&crate_name))?;

    // Content + Levels roots (kept in git with a placeholder).
    write(&root.join(&manifest.content_dir).join(".gitkeep"), "")?;
    write(&root.join(&manifest.levels_dir).join(".gitkeep"), "")?;

    write(&root.join(".gitignore"), GITIGNORE)?;
    write(
        &root.join("README.md"),
        &format!(
            "# {name}\n\nAn Infinity Engine project ({}).\n\nOpen it in Infinity Engine, or build the gameplay crate with `cargo build`.\n",
            template.label()
        ),
    )?;

    Ok(Scaffolded {
        root,
        crate_name,
        manifest,
    })
}

fn write(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    Ok(())
}

fn cargo_toml(crate_name: &str) -> String {
    format!(
        "[package]\n\
         name = \"{crate_name}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\
         \n\
         [lib]\n\
         crate-type = [\"lib\"]\n\
         \n\
         [dependencies]\n"
    )
}

const GITIGNORE: &str = "/target\n/Content/.inf/\n*.inf_lvl.autosave\ncrash-recovery.inf_lvl\n";

/// A filesystem-safe directory name (spaces → hyphens, alnum/-/_ kept).
pub fn sanitize_dir(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = s.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "InfinityProject".to_string()
    } else {
        trimmed
    }
}

/// A valid `snake_case` crate name.
pub fn sanitize_crate(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() || trimmed.chars().next().unwrap().is_numeric() {
        format!("game_{trimmed}")
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_writes_a_valid_project() {
        let parent = tempfile::tempdir().unwrap();
        let s = scaffold(parent.path(), "My Cool Game!", ProjectTemplate::Blank3d).unwrap();
        assert!(s.root.join("inf.toml").exists());
        assert!(s.root.join("Cargo.toml").exists());
        assert!(s.root.join("src/lib.rs").exists());
        assert!(s.root.join("Content/.gitkeep").exists());
        assert!(s.root.join("Levels/.gitkeep").exists());
        assert_eq!(s.crate_name, "my_cool_game");
        // The manifest reloads.
        let m = ProjectManifest::load(&s.root).unwrap();
        assert_eq!(m.name, "My Cool Game!");
        assert_eq!(m.template, "blank-3d");
        // The Cargo.toml names the sanitized crate.
        let cargo = std::fs::read_to_string(s.root.join("Cargo.toml")).unwrap();
        assert!(cargo.contains("name = \"my_cool_game\""));
    }

    #[test]
    fn refuses_a_non_empty_target() {
        let parent = tempfile::tempdir().unwrap();
        scaffold(parent.path(), "Dup", ProjectTemplate::Blank3d).unwrap();
        assert!(scaffold(parent.path(), "Dup", ProjectTemplate::Blank3d).is_err());
    }

    #[test]
    fn slug_round_trips() {
        for &t in ProjectTemplate::all() {
            assert_eq!(ProjectTemplate::from_slug(t.slug()), Some(t));
        }
    }

    #[test]
    fn crate_names_are_valid() {
        assert_eq!(sanitize_crate("My Game"), "my_game");
        assert_eq!(sanitize_crate("123"), "game_123");
        assert_eq!(sanitize_crate("!!!"), "game_");
    }
}
