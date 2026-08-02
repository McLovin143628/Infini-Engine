//! The **projector MIRROR gate** (P17.1, extended P18.3): the editor viewport's
//! ECS→`RenderScene` projection and the shipped player's must not drift.
//!
//! Two things are pinned here. `project_sky` is compared **character for
//! character** — it is a self-contained function on both sides. The `MeshRef`
//! branch that projects **real geometry** (P18.3) cannot be: it is inline in two
//! loops with different iteration orders, different id bookkeeping and different
//! asset stores. So it is pinned **field for field** instead — every field of the
//! `VgeomInstance` both sides construct, in order, with identical value
//! expressions for everything except the two host-local ones. That is exactly as
//! strong where it matters: the failure this exists to catch is "the editor
//! forgot to project `emissive`", or "the player gained a field the editor never
//! fills", either of which reads as *the shipped game looks different from the
//! preview* and is found by a player, not by a compiler.
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
        // P17.2: the physical atmosphere rides the same projection. Both the
        // authored block and the no-authority reset must be present, or a level
        // with a clock would render an atmosphere-less sky in one host and a
        // physical one in the other.
        "scene.atmosphere = AtmosphereParams",
        "AtmosphereParams::default()",
        "enabled: a.physical",
        "fog: HeightFog",
        "moon_phase: phase",
        // P17.3: the volumetric-cloud block. `time_s` is the fragment that
        // matters most — it is the only *derived* field in the projection, and a
        // host that fed it a frame counter or a wall clock instead of
        // `ResolvedSky::cloud_time_s()` would drift the two skies apart while
        // every other assertion here still passed.
        "clouds: CloudParams",
        "enabled: a.clouds_enabled",
        "time_s: sky.cloud_time_s()",
        "seed: a.cloud_seed",
        "shadow_strength: a.cloud_shadow",
        // P17.4: the weather block. `let w = sky.weather()` is the fragment that
        // matters — it is the Ring-0 decision about which of two parameter sets
        // is in force, and a host that inlined its own `if weather_enabled`
        // would be exactly the divergence this gate exists to stop. The three
        // *driven* assignments are here too, because a host could call
        // `sky.weather()` and then keep reading the authored fields, which would
        // pass a fragment check that only looked for the call.
        "let w = sky.weather();",
        "density: w.fog_density",
        "coverage: w.cloud_coverage",
        "cloud_type: w.cloud_type",
        "wind_x: w.wind_x",
        "precip: PrecipParams",
        "intensity: w.precipitation",
        "snowiness: w.snowiness",
    ] {
        assert!(
            body.contains(fragment),
            "`project_sky` no longer contains `{fragment}` — either it was gutted, \
             or this gate needs updating deliberately:\n{body}"
        );
    }
}

// ── P18.3: the real-geometry (`MeshRef.asset`) projection ────────────────────

/// The ordered `(field, value)` pairs of the **first** `VgeomInstance { … }`
/// struct literal in `source`.
///
/// Deliberately naive — it takes lines until the first one that closes the
/// literal — because the thing being compared is a flat struct literal, and a
/// parser clever enough to handle anything else would be clever enough to hide a
/// drift. Comments and blank lines are dropped; a `field,` shorthand yields a
/// value equal to the field name, so `translation,` and `translation: translation`
/// compare equal (they mean the same thing and either is idiomatic).
fn vgeom_instance_fields(source: &str) -> Vec<(String, String)> {
    let source = source.replace("\r\n", "\n");
    let start = source
        .find("VgeomInstance {")
        .unwrap_or_else(|| panic!("no `VgeomInstance {{` literal — did the projection move?"))
        + "VgeomInstance {".len();
    let mut out = Vec::new();
    for line in source[start..].lines() {
        let t = line.trim();
        if t.starts_with('}') {
            return out;
        }
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        let t = t.trim_end_matches(',');
        let (name, value) = match t.split_once(':') {
            Some((n, v)) => (n.trim().to_string(), v.trim().to_string()),
            None => (t.to_string(), t.to_string()),
        };
        out.push((name, value));
    }
    panic!("the `VgeomInstance` literal does not terminate");
}

/// Fields whose value expression is **host-local by design** and therefore
/// excluded from the value comparison (their presence and order still are not).
///
///  * `asset` — the player keys a vgeom asset by its derived **GUID** (a cooked
///    pack is immutable, so an id names one sequence of bytes forever); the editor
///    keys it by the derived payload's **content hash**, because a project's
///    content root is not immutable and both render nodes cache GPU state by this
///    id. The reasoning lives once, in `inf_editor_core::render_assets`.
///  * `id` — the pick id, allocated from each host's own counter over its own
///    iteration order (document order vs `Guid` order). It has never matched and
///    is not meant to.
const HOST_LOCAL_FIELDS: [&str; 2] = ["asset", "id"];

/// **The P18.3 mirror gate.** Both projectors build a `VgeomInstance` from the
/// same ECS state; every field must be present on both sides, in the same order,
/// carrying the same expression — except the two documented host-local ones.
#[test]
fn the_vgeom_instance_projection_matches_field_for_field() {
    let mine = vgeom_instance_fields(&read(VIEWPORT));
    let theirs = vgeom_instance_fields(&read(PLAYER));

    let names = |v: &[(String, String)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&mine),
        names(&theirs),
        "the two `VgeomInstance` projections carry different fields (or in a \
         different order) — a field projected on one side and not the other means \
         the shipped game draws an imported mesh differently from the preview"
    );

    for ((n, a), (_, b)) in mine.iter().zip(&theirs) {
        if HOST_LOCAL_FIELDS.contains(&n.as_str()) {
            continue;
        }
        assert_eq!(
            a, b,
            "`VgeomInstance::{n}` is projected as `{a}` in the editor viewport and \
             `{b}` in the shipped player. Keep them identical, or — if the \
             difference is deliberate — add the field to `HOST_LOCAL_FIELDS` with \
             the reason written down."
        );
    }
    // A guard on the guard: an empty literal would satisfy everything above.
    assert!(
        mine.len() >= 9,
        "the `VgeomInstance` projection shrank to {} fields — was it gutted?",
        mine.len()
    );
}

/// The surrounding *rules* — not just the literal — must exist on both sides:
/// resolution through the derived id, per-frame asset dedup, the paged source
/// handed to the scene rather than a decoded DAG, and the primitive fallback.
///
/// Without this a projector could satisfy the field comparison while listing the
/// asset twice, or while never falling back to the primitive at all (which would
/// make an unresolvable `MeshRef` invisible instead of a placeholder).
#[test]
fn both_projectors_keep_the_real_geometry_rules() {
    for (label, path, fragments) in [
        (
            "editor viewport",
            VIEWPORT,
            [
                // Resolution goes through the mesh asset id, never a side index.
                "mesh_ref.asset",
                "resolve_vgeom(mesh_id)",
                // One asset entry per frame, however many instances reference it.
                "vgeom_seen.insert(",
                "vgeom_assets",
                "VgeomAsset::new(",
                // The scene carries the PAGED source (P18.2), not a decoded DAG.
                "loaded.source",
                // …and an unresolved / primitive-only MeshRef still draws its
                // built-in primitive kind rather than vanishing.
                "prim_mesh(mesh_ref.primitive)",
            ],
        ),
        (
            "shipped player",
            PLAYER,
            [
                "mesh_ref.asset",
                "vmeshes.resolve(mesh_id)",
                "vgeom_seen.insert(",
                "vgeom_assets",
                "VgeomAsset::new(",
                "source",
                "prim_mesh(mesh_ref.primitive)",
            ],
        ),
    ] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in fragments {
            assert!(
                src.contains(fragment),
                "the {label}'s `MeshRef` projection no longer contains `{fragment}` \
                 — either the real-geometry path was changed on one side only, or \
                 this gate needs updating deliberately"
            );
        }
    }
}

/// **Both hosts must OPT IN to the meshlet path** (P18.3 audit).
///
/// `VgeomSettings::default()` is `enabled: false`, so carrying vgeom content is
/// not enough — a host that never asks draws all of it through the classic
/// discrete-LOD fallback. That failure is invisible: the fallback renders the
/// *same geometry*, so the only symptom is that none of P18.2's streaming, budget
/// or eviction is running, which no screenshot shows. The player has always asked;
/// the editor did not until this batch, and this is what keeps both honest.
///
/// The opt-in is the *request* — the tier clamp still has the last word on both
/// sides, which is why `RenderTier::apply` appears here too.
#[test]
fn both_hosts_request_the_meshlet_path() {
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        // The request itself: `VgeomSettings { enabled: true, .. }` over the
        // level's authored block.
        assert!(
            src.contains("VgeomSettings {") && src.contains("enabled: true"),
            "the {label} never requests `vgeom.enabled = true`, so every imported \
             mesh it carries would draw through the classic fallback"
        );
        // …and the clamp that can still take it away, so "requesting" never
        // becomes "forcing" on an adapter that cannot run it.
        assert!(
            src.contains(".apply(") || src.contains("detect_and_clamp"),
            "the {label} applies no tier clamp to its request"
        );
    }
}
