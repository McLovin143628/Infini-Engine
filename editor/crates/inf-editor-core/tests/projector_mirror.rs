//! The **projector MIRROR gate** (P17.1): the editor viewport's `project_sky` and
//! the shipped player's must stay character-for-character identical.
//!
//! # Why it lives here and not next to either projector
//!
//! `inf_viewport::host` is `#[cfg(any(windows, target_os = "macos"))]` — a test
//! inside it is invisible to the Linux CI leg, which is exactly the leg most
//! likely to be the one a contributor's PR runs first. `inf-editor-core` compiles
//! on all three platforms and sits in the same workspace, so the comparison runs
//! everywhere. Nothing here links either crate; it reads their **source text**,
//! which is the whole point: the duplication is deliberate and the gate is that
//! the duplicate has not drifted.
//!
//! # Why the duplication is deliberate
//!
//! The part that could *silently* diverge — which entity is the sky authority,
//! given that the editor walks document order and the player walks `Guid` order —
//! lives in `inf_ecs::sky` and is shared. What is left is a ~30-line mapping from
//! `inf_ecs` types into `inf_render` types, and **neither Ring-0 crate can host
//! it**: `inf-render` does not depend on `inf-ecs`, and `inf-ecs` must not depend
//! on `inf-render`. So it is written twice on purpose — and compared here.
//!
//! The classic bug this exists to catch surfaces only as "the shipped game lights
//! differently from the preview", which is precisely the class of thing that is
//! discovered by a player, not by a test.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

/// The text of `fn <name>(` through the closing brace at column 0, with line
/// endings normalized (the two files can be checked out with different EOLs,
/// which says nothing about whether the code drifted).
fn extract_fn(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let needle = format!("fn {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("`{needle}` not found — was the projector renamed?"));
    let rest = &source[start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{needle}` does not terminate at column 0"))
        + 3;
    rest[..end].to_string()
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

const VIEWPORT: &str = "editor/crates/inf-viewport/src/host.rs";
const PLAYER: &str = "runtime/inf-player/src/render.rs";

#[test]
fn project_sky_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_sky");
    let theirs = extract_fn(&read(PLAYER), "project_sky");
    assert_eq!(
        mine, theirs,
        "the two `project_sky` projectors have drifted — PIE would stop matching \
         shipping. Keep them byte-identical, or move the shared part into \
         `inf_ecs::sky` (which is where the authority-resolution rule already lives)."
    );
}

/// A guard on the guard: if either projector's `project_sky` were reduced to a
/// stub, the identity check above would still pass. Assert the shared body
/// actually does the work — it must read the resolved sky, write both renderer
/// blocks, and publish the key light.
#[test]
fn the_shared_projector_body_is_not_a_stub() {
    let body = extract_fn(&read(VIEWPORT), "project_sky");
    for fragment in [
        "inf_ecs::sky::resolve_sky",
        "scene.sun = SunParams",
        "scene.sky = SkyParams",
        "sky.sky_gradient()",
        "sky.key_light()",
        "scene.lights.push",
        "SunParams::default()",
        "SkyParams::default()",
    ] {
        assert!(
            body.contains(fragment),
            "`project_sky` no longer contains `{fragment}` — either it was gutted, \
             or this gate needs updating deliberately:\n{body}"
        );
    }
}
