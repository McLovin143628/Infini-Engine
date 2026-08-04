//! The **projector MIRROR gate** (P17.1, extended P18.3 and P18.5): the editor
//! viewport's ECS→`RenderScene` projection and the shipped player's must not
//! drift.
//!
//! Three things are pinned here. `project_sky` is compared **character for
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
//! The third is P18.5's **GPU-instanced scatter** — PCG volumes and painted
//! foliage. It had no gate at all before this batch, which is how the two hosts
//! came to disagree about `PcgVolume::draw_distance` for two phases: the editor
//! culled against its own camera eye on the CPU and the player ignored the field
//! entirely, so a shipped build drew strictly more scatter than its preview. Now
//! the field rides on the batch, the GPU cull honours it for both hosts, and the
//! `ScatterBatch` literal is pinned field for field like the vgeom one.
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

/// Byte offset of the **real item** `fn <name>(` — never a mention of the
/// signature inside a comment, a string or a backticked cross-reference.
///
/// The distinction is the whole of the P21.2 re-audit's F1. A plain
/// `source.find("fn project_voxel(")` matched a backticked self-reference *inside
/// that function's own doc comment*, several screens above the item, and every
/// caller below silently inherited the wrong anchor.
///
/// The test is positional rather than lexical: everything between the start of the
/// matched line and the `fn` keyword must be item qualifiers and nothing else. A
/// `///`, a backtick or prose fails it, so a doc block can name the signature as
/// often as it likes without moving the anchor.
fn item_start(source: &str, name: &str) -> usize {
    let needle = format!("fn {name}(");
    let mut from = 0usize;
    while let Some(rel) = source[from..].find(&needle) {
        let at = from + rel;
        let line_start = source[..at].rfind('\n').map_or(0, |i| i + 1);
        let is_item = source[line_start..at].split_whitespace().all(|w| {
            matches!(w, "pub" | "async" | "const" | "unsafe" | "extern")
                || w.starts_with("pub(")
                || w.starts_with('"')
        });
        if is_item {
            return at;
        }
        from = at + 1;
    }
    panic!("`{needle}` occurs nowhere as an item — was the projector renamed?");
}

/// The **item text** — the signature line (qualifiers included) through the
/// closing brace at column 0 — with line endings normalized (the two files can be
/// checked out with different EOLs, which says nothing about whether the code
/// drifted).
///
/// Used by the "not a stub" guards, which must read *code* and not prose: a
/// fragment assertion that can be satisfied by a sentence in a doc comment is a
/// phantom guard.
fn extract_item(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = item_start(&source, name);
    let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let rest = &source[line_start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`fn {name}(` does not terminate at column 0"))
        + 3;
    rest[..end].to_string()
}

/// **What every mirror-equality gate below compares: the doc block AND the item.**
///
/// The doc block is not decoration on a mirrored pair. Every rule the two copies
/// share — which of two values wins, what an empty projection means, why a field is
/// host-local — is written in the `///` lines and nowhere else, so a rule corrected
/// on one side only is the same defect as a line of code changed on one side only:
/// the next person to read the stale copy implements the stale rule.
///
/// It went uncompared for two batches. The anchor used to be the first
/// `fn <name>(` **substring** in the file, which for `project_voxel` was a
/// backticked self-reference inside its own doc comment — so extraction began
/// mid-comment, every line above it was invisible, and the two blocks drifted to 23
/// lines against 17 with this assertion green the whole time. The sentence that
/// drifted was the one *claiming* the doc block was uncovered, so the gate could
/// not catch the defect that motivated it. [`item_start`] is what closes it.
///
/// Attribute lines between the doc block and the item are **stepped over, not
/// compared**: `#[cfg(not(target_arch = "wasm32"))]` above the player's
/// `skinned_mesh_data` is a fact about which host builds for wasm, not about
/// whether the two hosts agree.
fn extract_fn(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = item_start(&source, name);
    let item_line = source[..start].rfind('\n').map_or(0, |i| i + 1);
    // Walk back over the item's attributes and doc comment, keeping the doc lines.
    let mut doc: Vec<&str> = Vec::new();
    let mut cursor = item_line;
    while cursor > 0 {
        let prev_start = source[..cursor - 1].rfind('\n').map_or(0, |i| i + 1);
        let line = &source[prev_start..cursor - 1];
        let trimmed = line.trim_start();
        if trimmed.starts_with("///") {
            doc.push(line);
        } else if !trimmed.starts_with("#[") {
            break;
        }
        cursor = prev_start;
    }
    // The guard on the guard: an undocumented item would quietly reduce this back
    // to `extract_item`, which is precisely the hole that was just closed.
    assert!(
        !doc.is_empty(),
        "`fn {name}` carries no doc comment, so this gate would compare only its \
         body — the exact hole the anchor fix closed. Document it on both sides, or \
         compare it with `extract_item`."
    );
    doc.reverse();
    let mut out = String::new();
    for line in doc {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&extract_item(&source, name));
    out
}

/// The text of `fn <name>(` through its **brace-balanced** end — the indented
/// twin of [`extract_fn`], which can only find a free function that terminates at
/// column 0. Needed for the skeletal stores, whose shared rules are `impl`
/// methods on each host's own store type.
fn extract_method(source: &str, name: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let needle = format!("fn {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("`{needle}` not found — was the store method renamed?"));
    let mut depth = 0usize;
    for (i, c) in source[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..start + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("`{needle}` does not terminate");
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

const VIEWPORT: &str = "editor/crates/inf-viewport/src/host.rs";
const PLAYER: &str = "runtime/inf-player/src/render.rs";

/// The editor's loose-file render-asset store (P18.3) — where the *skeletal*
/// resolution + pose rule lives on the editor side.
const EDITOR_ASSETS: &str = "editor/crates/inf-editor-core/src/render_assets.rs";
/// The shipped player's skeletal render-asset store — the same rules over a
/// cooked pack / dev dir.
const PLAYER_ASSETS: &str = "runtime/inf-player/src/skinned.rs";

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
    let body = extract_item(&read(VIEWPORT), "project_sky");
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

/// The ordered `(field, value)` pairs of the **first** `<ty> { … }` struct literal
/// in `source`.
///
/// Deliberately naive — it takes lines until the first one that closes the
/// literal — because the thing being compared is a flat struct literal, and a
/// parser clever enough to handle anything else would be clever enough to hide a
/// drift. Comments and blank lines are dropped; a `field,` shorthand yields a
/// value equal to the field name, so `translation,` and `translation: translation`
/// compare equal (they mean the same thing and either is idiomatic).
fn struct_literal_fields(source: &str, ty: &str) -> Vec<(String, String)> {
    let source = source.replace("\r\n", "\n");
    let open = format!("{ty} {{");
    let start = source
        .find(&open)
        .unwrap_or_else(|| panic!("no `{open}` literal — did the projection move?"))
        + open.len();
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
    panic!("the `{ty}` literal does not terminate");
}

fn vgeom_instance_fields(source: &str) -> Vec<(String, String)> {
    struct_literal_fields(source, "VgeomInstance")
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

// ── P18.5: the GPU-instanced scatter (PCG + foliage) projection ──────────────

/// The source of ONE component branch in a projector's entity loop: from the
/// `w.get::<Component>(entity)` probe that opens it to the probe that opens the
/// next branch.
///
/// Both hosts walk their world in one loop of `if let Some(c) = w.get::<C>(entity)`
/// branches, so the probes delimit them exactly. The end probe is passed in per
/// host because the branch that *follows* foliage differs (the editor projects
/// colliders next, the player meshes) — and if either host reorders its loop this
/// gate fails loudly, which is the correct outcome for a deliberate change.
fn branch_region(source: &str, from: &str, to: &str) -> String {
    let source = source.replace("\r\n", "\n");
    let start = source
        .find(from)
        .unwrap_or_else(|| panic!("`{from}` not found — did the projection loop change?"));
    let rest = &source[start..];
    let end = rest
        .find(to)
        .unwrap_or_else(|| panic!("`{to}` does not follow `{from}` — the loop was reordered?"));
    rest[..end].to_string()
}

/// Fields of the `ScatterBatch` literal whose value expression is **host-local by
/// design** and therefore excluded from the value comparison (their presence and
/// order still are not).
///
///  * `id` — the pick id, allocated from each host's own counter over its own
///    iteration order (document order vs `Guid` order). It has never matched
///    across hosts and is not meant to; the editor additionally maps it back to a
///    GUID, which the player has no use for. The two projectors happen to spell it
///    with the same token today because the batch is built by a shared helper, but
///    the *value* is host-local on principle and a host that inlined its own id
///    allocation must not trip this gate.
///
/// Everything else — the payload, the anchor, the material constants and the
/// content draw distance — is the same expression on both sides, because a scatter
/// that shades or culls differently in the shipped build than in the preview is
/// exactly the bug this file exists to catch.
const SCATTER_HOST_LOCAL_FIELDS: [&str; 1] = ["id"];

/// **The P18.5 mirror gate.** Both projectors build a `ScatterBatch` from the same
/// ECS state; every field must be present on both sides, in the same order,
/// carrying the same expression — except the documented host-local one.
///
/// The literal compared is the **first** in each file, which is the shared
/// `push_scatter` body (P19.3 hoisted it out of `push_pcg_scatter` so the volume
/// path and the terrain biome population cannot drift): both hosts define
/// `push_scatter` before `push_foliage_scatter`.
#[test]
fn the_scatter_batch_projection_matches_field_for_field() {
    let mine = struct_literal_fields(&read(VIEWPORT), "ScatterBatch");
    let theirs = struct_literal_fields(&read(PLAYER), "ScatterBatch");

    let names = |v: &[(String, String)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&mine),
        names(&theirs),
        "the two `ScatterBatch` projections carry different fields (or in a \
         different order) — a field projected on one side and not the other means \
         the shipped game draws PCG/foliage scatter differently from the preview"
    );

    for ((n, a), (_, b)) in mine.iter().zip(&theirs) {
        if SCATTER_HOST_LOCAL_FIELDS.contains(&n.as_str()) {
            continue;
        }
        assert_eq!(
            a, b,
            "`ScatterBatch::{n}` is projected as `{a}` in the editor viewport and \
             `{b}` in the shipped player. Keep them identical, or — if the \
             difference is deliberate — add the field to `SCATTER_HOST_LOCAL_FIELDS` \
             with the reason written down."
        );
    }
    // A guard on the guard: an empty literal would satisfy everything above.
    assert!(
        mine.len() >= 7,
        "the `ScatterBatch` projection shrank to {} fields — was it gutted?",
        mine.len()
    );
}

/// The surrounding *rules* — not just the literal — must hold on both sides: the
/// anchor convention, the placeholder palette, the deterministic per-kind
/// grouping, the content draw distance… **and the absence of the per-instance CPU
/// push**, which is the whole point of P18.5.
///
/// Without the absence check a projector could satisfy every fragment above while
/// *also* still expanding each instance into `RenderScene::instances` — the scatter
/// would then draw twice on one host and once on the other, which is precisely the
/// preview-vs-shipping divergence this file exists to stop.
#[test]
fn both_projectors_scatter_pcg_and_foliage_the_same_way() {
    // Fragments that must appear in BOTH projectors, verbatim.
    const SHARED: [&str; 13] = [
        // The Ring-0 pack + content hash, so neither host invents its own layout.
        "ScatterData::build(",
        // THE ANCHOR RULE: offsets are relative to the entity's world translation…
        "anchor: translation,",
        // …and foliage instances, being entity-LOCAL already, are packed against a
        // ZERO build-anchor — no conversion, so the same stroke content-keys the
        // same wherever it is placed.
        "ScatterData::build(PrimMesh::ALL[k], DVec3::ZERO, bucket)",
        // PCG stays a placeholder cube, coloured per kind.
        "PrimMesh::Cube,",
        "pcg_kind_color(si.kind)",
        // Foliage rotation goes through the shared euler-degrees → quat rule.
        "foliage_rot_quat(fi.rotation)",
        // The palette fallback for an unknown kind, colour included.
        "(PrimMesh::Cube, [0.28, 0.52, 0.24, 1.0])",
        // Per-kind grouping: bucket in authored order, emit in `PrimMesh::ALL`
        // order — deterministic and independent of which kinds are used.
        "buckets[mesh.index()].push(",
        "for (k, bucket) in buckets.into_iter().enumerate()",
        // The content LOD knob rides on the batch (this is what made the two hosts
        // finally agree about it) — and foliage has none. P19.3 moved the batch
        // literal into the shared `push_scatter`, so the volume's authored knob is
        // now what it *hands* that body; `draw_distance` reaching the literal is
        // pinned separately, field for field, by the test above.
        "vol.draw_distance,",
        "draw_distance: 0.0,",
        // Both branches go through the shared helpers rather than open-coding.
        "push_pcg_scatter(",
        "fn push_scatter(",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s scatter projection no longer contains `{fragment}` — \
                 either the scatter path was changed on one side only, or this gate \
                 needs updating deliberately"
            );
        }
        assert!(src.contains("push_foliage_scatter("), "{label}");
    }

    // …and the per-instance CPU push is really gone from both PCG/Foliage branches.
    for (label, path, pcg_end, foliage_end) in [
        (
            "editor viewport",
            VIEWPORT,
            "w.get::<Foliage>(entity)",
            "w.get::<Collider2D>(entity)",
        ),
        (
            "shipped player",
            PLAYER,
            "w.get::<Foliage>(entity)",
            "w.get::<MeshRef>(entity)",
        ),
    ] {
        let src = read(path);
        for (branch, region) in [
            (
                "PcgVolume",
                branch_region(&src, "w.get::<PcgVolume>(entity)", pcg_end),
            ),
            (
                "Foliage",
                branch_region(&src, "w.get::<Foliage>(entity)", foliage_end),
            ),
        ] {
            assert!(
                !region.contains("instances.push("),
                "the {label}'s {branch} branch still pushes per-instance \
                 `MeshInstance`s — P18.5 replaced that with one `ScatterBatch`, and \
                 a host doing both draws the scatter twice:\n{region}"
            );
            assert!(
                !region.contains("MeshInstance"),
                "the {label}'s {branch} branch still builds a `MeshInstance`:\n{region}"
            );
            assert!(
                region.contains("_scatter("),
                "the {label}'s {branch} branch no longer calls its scatter helper:\n{region}"
            );
        }
    }
}

/// **The P19.3 mirror gate: the terrain's BIOME POPULATION.**
///
/// A terrain bound to biomes gains a derived, never-persisted instance population
/// — each painted biome's `.inf_pcg` graph evaluated over the region its id owns.
/// It is the terrain-level sibling of `PcgVolume::evaluated`, and it reaches the
/// GPU the same way: one `ScatterBatch`.
///
/// This is exactly the shape of divergence this file exists to catch. A population
/// projected by the editor and not by the player is *"the biome scatter shows in
/// the preview and is missing from the shipped build"* — a whole layer of world
/// content, discovered by a player. So both hosts must carry the branch, and both
/// must reach the batch through the SAME `push_scatter` body the volume path uses:
/// two copies of "build a batch from scattered instances" is precisely how the
/// hosts came to disagree about `draw_distance` for two phases.
#[test]
fn both_projectors_project_the_terrain_biome_population_the_same_way() {
    // Fragments that must appear in BOTH projectors, verbatim.
    const SHARED: [&str; 4] = [
        // The branch calls the population helper …
        "push_biome_population(",
        // … which reads the derived component field …
        "&terrain.biome_population",
        // … and whose body is the SAME one the volume path uses, so a population
        // cannot be packed, shaded or culled differently from a volume's scatter.
        "fn push_scatter(",
        // The empty-population guard: no content ⇒ no batch, on both sides.
        "terrain.biome_population.is_empty()",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s terrain biome-population projection no longer contains \
                 `{fragment}` — either the population path was changed on one side \
                 only (the shipped build would then draw a different world from its \
                 preview), or this gate needs updating deliberately"
            );
        }
    }

    // The helper is shared *character for character* across the hosts — it is a
    // self-contained free function on both sides, like `project_sky`, so nothing
    // weaker than equality is called for.
    assert_eq!(
        extract_fn(&read(VIEWPORT), "push_biome_population"),
        extract_fn(&read(PLAYER), "push_biome_population"),
        "`push_biome_population` has drifted between the editor viewport and the \
         shipped player"
    );
    assert_eq!(
        extract_fn(&read(VIEWPORT), "push_scatter"),
        extract_fn(&read(PLAYER), "push_scatter"),
        "the shared `push_scatter` body has drifted between the editor viewport and \
         the shipped player — it is the one place a scattered instance becomes a \
         `ScatterBatch` on either host"
    );

    // …and the population branch is instanced, never per-instance mesh draws —
    // the same absence check the PCG branch gets, for the same reason. The branch
    // is delimited by its own guard (the `w.get::<Terrain>(entity)` probe opens
    // the *clipmap* terrain branch far earlier in the loop) and ends where the
    // foliage branch begins.
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path);
        let region = branch_region(
            &src,
            "!terrain.biome_population.is_empty()",
            "w.get::<Foliage>(entity)",
        );
        assert!(
            !region.contains("instances.push("),
            "the {label}'s biome-population branch pushes per-instance \
             `MeshInstance`s — a population is instanced, never per-instance mesh \
             draws:\n{region}"
        );
        assert!(
            !region.contains("MeshInstance"),
            "the {label}'s biome-population branch builds a `MeshInstance`:\n{region}"
        );
    }
}

/// **The "the follow-up landed" gate** (P18.5).
///
/// Both projectors used to carry the same warning — *">50k instances — instanced-
/// draw perf path is a follow-up"* — because both expanded a scatter into one
/// `MeshInstance` per instance and there was nothing better to do about it. This
/// batch IS that follow-up: the payload uploads once per content change and the
/// GPU culls it per instance. A warning left behind would be a false alarm on
/// exactly the content the engine now handles best, and a reader would take it as
/// a live limitation.
#[test]
fn neither_projector_warns_about_fifty_thousand_instances() {
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for stale in ["50_000", "instanced-draw perf"] {
            assert!(
                !src.contains(stale),
                "the {label} still contains `{stale}` — P18.5 IS the instanced-draw \
                 follow-up that warning pointed at, so it must not survive it"
            );
        }
    }
}

// ── the skeletal (`SkeletalMesh`) projection ─────────────────────────────────
//
// P18.3 gave the editor viewport a real `SkeletalMesh` branch and left the
// shipped player with **none at all**: a level with a skeletal character
// previewed correctly in PIE and shipped as nothing. That was a live
// PIE-vs-shipping divergence, not a missing feature, and the follow-up that
// closes it is what these gates pin.
//
// Three layers, because the skeletal path is split across four files rather than
// two: the **pose rule** and the **bind-space rebuild** live in each host's
// render-asset store (Ring 1 for the editor, `inf-player` for the player) and are
// self-contained functions, so they are compared *character for character*; the
// `SkinnedInstance` literal is inline in two loops with different iteration
// orders, so it is compared field for field, exactly like the P18.3 `VgeomInstance`
// gate above.

/// **The pose rule must be one rule.** No skeleton (or a jointless one) ⇒ the
/// caller keeps its placeholder; no `AnimPlayer` / no clip / an unresolvable clip
/// ⇒ the rest pose; otherwise the clip sampled at the play-head, honouring
/// `looping`.
///
/// Every one of those three arms is a place the two hosts could silently drift,
/// and each drift reads as a different bug in the shipped game: arm 1 as a
/// character that vanishes instead of showing a placeholder, arm 2 as one that is
/// invisible until it plays, arm 3 as one frozen in its bind pose. None of them
/// is visible to a compiler, and only arm 3 would even show up in a scene-level
/// comparison of the two hosts.
///
/// **The one permitted difference**, normalized below: the editor's store resolves
/// lazily into its own maps and is owned mutably by the viewport (`&mut self`);
/// the player's memoizes behind its own locks because the projector holds it
/// immutably (`&self`). That is a difference in *who owns the cache*, not in the
/// rule.
#[test]
fn the_skinned_pose_rule_is_identical_in_both_stores() {
    let raw = extract_method(&read(EDITOR_ASSETS), "resolve_skinned");
    // The normalization must rewrite the RECEIVER and nothing else. A second
    // `&mut self` anywhere in the body would be silently erased along with it, and
    // an interior-mutability divergence is exactly the kind of difference this gate
    // exists to catch — so the token is asserted to occur exactly once before it is
    // rewritten.
    assert_eq!(
        raw.matches("&mut self").count(),
        1,
        "`resolve_skinned` contains more than one `&mut self`; the receiver          normalization below would erase the others too"
    );
    let mine = raw.replace("&mut self", "&self");
    let theirs = extract_method(&read(PLAYER_ASSETS), "resolve_skinned");
    assert_eq!(
        mine, theirs,
        "the editor's and the player's `resolve_skinned` have drifted — the two \
         hosts would pose the same character differently, so PIE would stop \
         matching shipping. Keep them identical (modulo `&mut self` / `&self`), or \
         move the rule into a Ring-0 crate both can depend on."
    );
}

/// A guard on the guard: two stubs are identical too. The shared body has to
/// actually implement the three arms.
#[test]
fn the_shared_pose_rule_is_not_a_stub() {
    let body = extract_method(&read(PLAYER_ASSETS), "resolve_skinned");
    for fragment in [
        // Arm 1 — both halves of the binding, and the jointless-skeleton reject.
        "let mesh_id = sm.mesh?;",
        "let skeleton_id = sm.skeleton?;",
        "if skeleton.is_empty() {",
        // Arm 3 — the play-head sampled through the shared Ring-0 sampler,
        // `looping` included (a host that dropped it would loop a one-shot).
        "inf_anim::sample_clip(&skeleton, &clip, p.t as f32, p.looping)",
        // Arm 2 — rest pose for BOTH "no player/clip" and "clip did not resolve".
        // Two occurrences, and the count is asserted below.
        "inf_anim::Pose::rest(&skeleton)",
        // The palette the skinned pass consumes, and the dedup key the projection
        // allocates `skinned_meshes` slots from.
        "palette: inf_anim::skinning_matrices(&skeleton, &pose)",
        "key: (mesh_id, skeleton_id)",
    ] {
        assert!(
            body.contains(fragment),
            "`resolve_skinned` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
    assert_eq!(
        body.matches("inf_anim::Pose::rest(&skeleton)").count(),
        2,
        "the rest-pose fallback must cover BOTH `no AnimPlayer/clip` and `the clip \
         did not resolve` — one of the two arms stopped falling back:\n{body}"
    );
}

/// **The bind-space rebuild must be one rebuild.** Both hosts turn the same
/// `.inf_mesh` into the same `SkinnedMeshData`, or they upload *different vertex
/// buffers* for the same asset — a divergence no scene-level comparison can see,
/// because both scenes would carry one mesh with the right instance count.
///
/// The load-bearing details are the submesh concatenation with rebased indices,
/// and pinning an unskinned submesh to joint 0 with weight 1 (dropping it instead
/// would silently lose geometry on one host only).
#[test]
fn the_bind_space_rebuild_is_identical_in_both_stores() {
    let mine = extract_fn(&read(EDITOR_ASSETS), "skinned_mesh_data");
    let theirs = extract_fn(&read(PLAYER_ASSETS), "skinned_mesh_data");
    assert_eq!(
        mine, theirs,
        "the editor's and the player's `skinned_mesh_data` have drifted — the two \
         hosts would build different bind-space geometry from the same `.inf_mesh`"
    );
    // Not a stub: the concatenation, the rebase, and the joint-0 pin. Read off the
    // ITEM, never the doc block — a fragment a comment could satisfy proves nothing.
    let body = extract_item(&read(PLAYER_ASSETS), "skinned_mesh_data");
    for fragment in [
        "let base = vertices.len() as u32;",
        "sm.skin.get(i).copied().unwrap_or_default().normalized()",
        "indices.extend(sm.indices.iter().map(|&i| i + base));",
    ] {
        assert!(
            body.contains(fragment),
            "`skinned_mesh_data` lost `{fragment}`"
        );
    }
}

/// Fields of the `SkinnedInstance` literal whose value expression is **host-local
/// by design** and therefore excluded from the value comparison (their presence
/// and order still are not).
///
///  * `id` — the pick id, allocated from each host's own counter over its own
///    iteration order (document order vs `Guid` order). It has never matched
///    across hosts and is not meant to; the editor additionally maps it back to a
///    GUID, which the player has no use for.
///  * `mesh` — the index into `RenderScene::skinned_meshes`, allocated from each
///    host's own per-projection dedup map in that same iteration order. Two hosts
///    that agree perfectly about *which* `(mesh, skeleton)` pairs are drawn can
///    still number their slots differently, and the number means nothing outside
///    the scene it indexes.
///
/// Note what is **not** here, and deliberately so: unlike the `VgeomInstance`
/// gate, `asset`-style content-hash-vs-GUID keying has no analogue on this path.
/// The skinned pass caches its GPU upload by the **pointer identity** of the
/// `Arc<SkinnedMeshData>`, so changed content is a different key by construction
/// and neither host needs to content-address an id.
/// Fields of `SkinnedInstance` the two hosts are *allowed* to project differently,
/// each with the reason:
///
/// * `id` / `mesh` — each host numbers from its own counter over its own iteration
///   order (the player walks the world in `Guid` order, the viewport walks the
///   document's entity order), so the *values* differ while the rule does not. The
///   same asymmetry `VGEOM_HOST_LOCAL_FIELDS` documents.
/// * `translation` — the editor reads it straight off the entity's affine, the
///   player reads `sim.interp_translation(..)` so a character is drawn at its
///   interpolated position between fixed steps rather than snapped to the last one.
///   Both hosts' own comments already say this; it belongs here so the gate stops
///   asking about it and starts *documenting* it.
const SKINNED_HOST_LOCAL_FIELDS: [&str; 3] = ["id", "mesh", "translation"];

/// **The skeletal mirror gate.** Both projectors build a `SkinnedInstance` from
/// the same ECS state; every field must be present on both sides, in the same
/// order, carrying the same expression — except the two documented host-local
/// ones.
#[test]
fn the_skinned_instance_projection_matches_field_for_field() {
    let mine = struct_literal_fields(&read(VIEWPORT), "SkinnedInstance");
    let theirs = struct_literal_fields(&read(PLAYER), "SkinnedInstance");

    let names = |v: &[(String, String)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(
        names(&mine),
        names(&theirs),
        "the two `SkinnedInstance` projections carry different fields (or in a \
         different order) — a field projected on one side and not the other means \
         the shipped game draws a character differently from the preview"
    );

    for ((n, a), (_, b)) in mine.iter().zip(&theirs) {
        if SKINNED_HOST_LOCAL_FIELDS.contains(&n.as_str()) {
            continue;
        }
        assert_eq!(
            a, b,
            "`SkinnedInstance::{n}` is projected as `{a}` in the editor viewport and \
             `{b}` in the shipped player. Keep them identical, or — if the \
             difference is deliberate — add the field to \
             `SKINNED_HOST_LOCAL_FIELDS` with the reason written down."
        );
    }
    // A guard on the guard: an empty literal would satisfy everything above.
    assert!(
        mine.len() >= 10,
        "the `SkinnedInstance` projection shrank to {} fields — was it gutted?",
        mine.len()
    );
}

/// The surrounding *rules* — not just the literal — must hold on both sides: the
/// `MeshRef`-absent gate the branch hangs off, the `AnimPlayer` lookup that feeds
/// the pose, the one-slot-per-`(mesh, skeleton)` dedup, the `Arc` handed straight
/// into the scene, and the placeholder that survives an unresolvable binding.
///
/// The `Arc` fragment is the load-bearing one. `RenderScene::skinned_meshes` is a
/// `Vec<Arc<SkinnedMeshData>>` *because* the skinned pass caches its GPU upload by
/// pointer identity; a host that pushed `draw.mesh.as_ref().clone()` — or rebuilt
/// the stream per projection — would re-upload a character's whole bind-space
/// buffer every frame while every pixel stayed identical, so nothing else in the
/// repo would notice.
#[test]
fn both_projectors_draw_skeletal_meshes_the_same_way() {
    // Fragments that must appear in BOTH projectors, verbatim.
    const SHARED: [&str; 10] = [
        // The branch is the `MeshRef`-absent arm: an entity is a rigid draw or a
        // skinned one, never both.
        "w.get::<MeshRef>(entity).is_none()",
        "w.get::<SkeletalMesh>(entity).copied()",
        // The pose input, and the shared store call that applies the pose rule.
        "w.get::<inf_ecs::components::AnimPlayer>(entity).copied()",
        "resolve_skinned(&sm, player.as_ref())",
        // ONE `skinned_meshes` slot per (mesh, skeleton) pair…
        "skinned_slots.entry(draw.key).or_insert_with(",
        // …and the entry is the store's own `Arc`, pushed with no copy.
        "skinned_meshes.push(draw.mesh)",
        "skinned_meshes.len() - 1",
        // Both lists are rebuilt from scratch every projection.
        "skinned_meshes.clear()",
        ".skinned.clear()",
        // The placeholder survives, down to its slate tint, so the two hosts also
        // agree about content whose assets are missing.
        "color: [0.55, 0.60, 0.72, 1.0],",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s `SkeletalMesh` projection no longer contains \
                 `{fragment}` — either the skeletal path was changed on one side \
                 only, or this gate needs updating deliberately"
            );
        }
        // The `Arc` must reach the scene UNCLONED-as-data. `.clone()` on the inner
        // value (or a `to_vec()` of the vertex stream) would defeat the pass's
        // pointer-identity upload cache silently.
        for banned in [
            "skinned_meshes.push(draw.mesh.as_ref().clone())",
            "draw.mesh.vertices.clone()",
        ] {
            assert!(
                !src.contains(banned),
                "the {label} copies the bind-space stream into the scene \
                 (`{banned}`) — `RenderScene::skinned_meshes` is `Vec<Arc<_>>` \
                 precisely so the skinned pass can cache its GPU upload by pointer \
                 identity, and a copy re-uploads a character every frame"
            );
        }
    }
}

// ── P20.1: the water (`WaterBody`) projection ────────────────────────────────

/// **The P20.1 mirror gate.** `project_water` is a self-contained free function
/// on both sides — like `project_sky` — so nothing weaker than character-for-
/// character equality is called for.
///
/// The divergence this catches is the one a player finds: a sea that is a
/// different colour, a different height or a different *phase* in the shipped
/// build than in the preview. Water is worse than most content for this, because
/// its whole appearance is derived rather than authored: one host reading the
/// weather wind and the other reading the body's own would produce two plausible
/// seas that never agree.
#[test]
fn project_water_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_water");
    let theirs = extract_fn(&read(PLAYER), "project_water");
    assert_eq!(
        mine, theirs,
        "the two `project_water` projectors have drifted — PIE would stop \
         matching shipping. Keep them byte-identical, or move the shared part \
         into `inf_ecs::sky` / `inf-water` (which is where the clock, the wind \
         rule and the wave derivation already live)."
    );
}

/// A guard on the guard: two stubs are identical too. The shared body has to do
/// the work — derive the wave field, honour the wind rule, and build a river's
/// ribbon from the spline on the same entity.
#[test]
fn the_shared_water_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_water");
    for fragment in [
        // The wind rule is the component's, not the host's — the one thing here
        // that could silently diverge.
        "water.effective_wind(weather_wind)",
        // The wave field is DERIVED in Ring 0, never re-derived per host.
        "inf_render::WaveField::from_spec(&spec)",
        "spread_rad: water.wave_spread_deg.to_radians()",
        "seed: water.wave_seed",
        // A river's ripple travels downstream, so its wave frame is the river's
        // own — a host that fed it the world wind would show a river whose
        // wavelets cross it sideways.
        "let river = water.kind == WaterKind::River;",
        "wind_x: if river { 1.0 } else { wind_x },",
        // The centreline is the spline on the SAME entity, in world space.
        "affine.transform_point3(p.to_dvec3())",
        "inf_render::RiverPath::from_points(&points, sp.closed, interp, &profile)",
        "inf_render::WaterFrame::from(f)",
        // P20.4: the P19.1 flow map reaches a river's frames, and does so through
        // the Ring-0 rule rather than a per-host world walk. A host that gathered
        // terrains itself would be free to disagree about which one answers.
        "let mapped = flow.is_mapped();",
        "flow.foam_gain_at(glam::DVec2::new(f.center.x, f.center.z))",
        // The level clock reaches the body — a host that dropped it would render
        // a frozen sea that still passed every other assertion here.
        "time_s,",
        // The kind mapping, all three arms.
        "WaterKind::Ocean => inf_render::WaterKindGpu::Ocean,",
        "WaterKind::Lake => inf_render::WaterKindGpu::Lake,",
        "WaterKind::River => inf_render::WaterKindGpu::River,",
    ] {
        assert!(
            body.contains(fragment),
            "`project_water` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
}

/// The surrounding *rules* — not just the shared body — must hold on both sides:
/// the list is rebuilt from scratch, the clock and wind are resolved ONCE per
/// projection through the Ring-0 seam, the spline comes from the same entity, and
/// a body with no geometry is skipped rather than drawn degenerate.
#[test]
fn both_projectors_project_water_the_same_way() {
    const SHARED: [&str; 8] = [
        // Rebuilt every projection, like `scatter` — a body's state is a pure
        // function of its component, its spline and the clock.
        "waters.clear()",
        // The clock + wind come from Ring 0, once, never per body and never from
        // a wall clock. A host that inlined `resolve_sky(...).weather()` here
        // would be the divergence this file exists to stop.
        "inf_ecs::sky::water_environment(world)",
        // The branch itself, and the same-entity spline.
        "w.get::<WaterBody>(entity)",
        // The flow field, likewise resolved ONCE per projection through Ring 0
        // (P20.4) — same argument as the clock and the wind above.
        "inf_ecs::hydro::terrain_flow(world)",
        "w.get::<Spline>(entity),",
        "&water_flow,",
        // Nothing degenerate reaches the renderer.
        "if body.drawable()",
        "waters.push(body)",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s water projection no longer contains `{fragment}` — \
                 either the water path was changed on one side only, or this gate \
                 needs updating deliberately"
            );
        }
        // A river's centreline must NOT be looked up through a reference: the
        // whole point of same-entity composition is that there is nothing to
        // resolve, and a host that started resolving one would need a cook edge
        // the other side does not have.
        let region = branch_region(&src, "w.get::<WaterBody>(entity)", "project_water(");
        assert!(
            !region.contains("entity_of("),
            "the {label} resolves a river's spline through a reference:\n{region}"
        );
    }
}

// ── P21.1: the volumetric-terrain (`VoxelVolume`) projection ─────────────────

/// **The P21.1 mirror gate.** `project_voxel` is a self-contained free function on
/// both sides — like `project_sky` and `project_water` — so nothing weaker than
/// character-for-character equality is called for.
///
/// The divergence this catches is one a player finds and a compiler cannot: a cave
/// whose walls sit a fraction of a voxel from where the preview drew them, or
/// whose rock shades a different colour than the hillside it opens out of. Voxel
/// surface is worse than most content for this, because none of it is authored
/// directly — every vertex is *derived* from a field, so two plausible derivations
/// can differ everywhere while agreeing about the level.
#[test]
fn project_voxel_is_identical_in_both_projectors() {
    let mine = extract_fn(&read(VIEWPORT), "project_voxel");
    let theirs = extract_fn(&read(PLAYER), "project_voxel");
    assert_eq!(
        mine, theirs,
        "the two `project_voxel` projectors have drifted — PIE would stop matching \
         shipping. Keep them byte-identical, or move the shared part into \
         `inf-voxel` (which is where the mesher, the chunk store and the \
         chunk-local rebase already live)."
    );
}

/// A guard on the guard: two stubs are identical too. The shared body has to do
/// the work — rebase through the Ring-0 helper, take the scale from the asset,
/// take the palette from the terrain on the same entity, and refuse to emit an
/// empty volume.
#[test]
fn the_shared_voxel_projector_is_not_a_stub() {
    let body = extract_item(&read(PLAYER), "project_voxel");
    for fragment in [
        // The scale comes from the ASSET, not from the component. A host that read
        // `volume.voxel_size_m` here would scale vertices against origins derived
        // from the asset's and tear the volume apart — and it would look fine in
        // every level where the two happen to agree, which is most of them.
        "let voxel_size_m = slot.data.voxel_size_m();",
        // The chunk-local rebase is the ONE shared Ring-0 helper. A host that
        // subtracted its own base, or scaled in a different order, would draw a
        // cave a fraction of a voxel from where the other host drew it.
        ".local_positions_m(voxel_size_m)",
        "bounds: mesh.local_bounds_m(voxel_size_m),",
        // The f64 world anchor + the entity transform (the floating-origin split).
        "origin: slot.data.chunk_origin_world(key) + translation,",
        // Only the NON-EMPTY meshes reach the renderer, in the cache's ascending
        // key order (a deterministic draw order per side).
        ".meshes()",
        // The GPU cache's invalidation key, which is the mesh's neighbourhood
        // stamp — NOT the chunk's own, or a carve in the neighbour would leave a
        // stale seam on screen forever.
        "version: slot.meshes.version(key),",
        // A material index is categorical and is the TERRAIN splat index.
        "material: mesh.materials[i] as u32,",
        "albedo: t.layers[k].albedo.to_array(),",
        "None => RenderTerrainLayer::default(),",
        // Nothing empty reaches the renderer — that is what keeps the voxel pass
        // off the command encoder on every scene that has no caves.
        "if chunks.is_empty() {",
        "return None;",
    ] {
        assert!(
            body.contains(fragment),
            "`project_voxel` no longer contains `{fragment}` — either it was \
             gutted, or this gate needs updating deliberately:\n{body}"
        );
    }
}

/// The surrounding *rules* — not just the shared body — must hold on both sides:
/// the list is rebuilt from scratch, the palette comes from the `Terrain` on the
/// same entity, the volume's identity is the shared entity fold, and **neither
/// host binds, loads or meshes inside its projection** — both do it in a
/// `sync_voxels` pre-pass, whose live set is built from BOUND-NESS rather than
/// from whatever happened to produce triangles.
///
/// That last rule is not style. A volume whose chunks mesh to no surface still
/// projects `None`, so a host that built its live set from the projection would
/// release it and re-read + re-mesh it from disk on the very next document bump —
/// and a gizmo drag bumps the document per input event.
#[test]
fn both_projectors_project_voxel_volumes_the_same_way() {
    const SHARED: [&str; 10] = [
        // Rebuilt every projection, like `terrains`.
        "scene.voxels.clear()",
        // The branch itself.
        "w.get::<VoxelVolume>(entity)",
        // The palette is the Terrain on THIS SAME entity — composition, not a
        // reference, so there is no cook edge and nothing to dangle.
        "w.get::<Terrain>(entity),",
        // Identity is the shared entity fold, so a PIE-vs-shipping diff matches
        // volumes up by identity rather than by position in a list.
        "inf_render::terrain_id_from_guid(guid.as_u128()),",
        // Both go through the SAME Ring-0 slot type, which is what owns parsing,
        // residency and meshing.
        "inf_voxel::VolumeSlot",
        "project_voxel(",
        // The bind PRE-PASS, on both sides…
        "fn sync_voxels(",
        // …whose live set is bound-ness, never draw-ness: the release runs over
        // what `ensure` bound, not over what the projection pushed.
        "retain_only(",
        // P21.2 — the pre-pass is THREE acts and a host that skipped either of the
        // last two fails differently but silently. Without `place` residency is
        // measured from the asset's authoring anchor instead of from where the
        // entity actually is, so a cave placed away from the origin pages the
        // chunks nobody is standing in — a hole in the world with no rendering
        // explanation. Without `sync_camera` **nothing pages at all**: `ensure`
        // binds and pages zero chunks by design, so the volume meshes to nothing
        // and the host draws an empty cave while every other assertion here passes.
        "place(",
        "sync_camera(",
    ];
    for (label, path) in [("editor viewport", VIEWPORT), ("shipped player", PLAYER)] {
        let src = read(path).replace("\r\n", "\n");
        for fragment in SHARED {
            assert!(
                src.contains(fragment),
                "the {label}'s voxel projection no longer contains `{fragment}` — \
                 either the voxel path was changed on one side only, or this gate \
                 needs updating deliberately"
            );
        }
        // Neither host may re-mesh inside the projection: meshing is a `&mut` act
        // that belongs to the store's sync step, and a host that called the mesher
        // per frame would burn a chunk's whole cell walk on geometry that did not
        // move — and could disagree with the other host about *when* it meshed,
        // which is exactly the class of drift this file exists to stop.
        let region = branch_region(&src, "w.get::<VoxelVolume>(entity)", "project_voxel(");
        for banned in [
            "mesh_chunk(",
            "VoxelMeshCache::new()",
            ".meshes.sync(",
            // …and neither may BIND inside the projection either, which is what
            // cold-meshes: `ensure` parses a payload and runs a full mesh sync.
            ".ensure(",
        ] {
            assert!(
                !region.contains(banned),
                "the {label}'s voxel branch calls `{banned}` — meshing belongs to \
                 the store's sync step, never to the projection:\n{region}"
            );
        }
    }
}

/// **The store is the shared half, and it must stay shared.** Both hosts resolve
/// bytes their own way (a loose file under the content root; a pack entry) and
/// then hand them to the *same* Ring-0 `inf_voxel::VoxelVolumes`. A host that grew
/// its own parser, its own residency or its own mesh cache would be free to
/// disagree about every vertex in the world — and, unlike a projection, nothing
/// about that would be visible in a source diff of the two projectors.
#[test]
fn both_hosts_load_voxel_volumes_through_the_ring_zero_store() {
    for (label, path, own_store) in [
        (
            "editor viewport",
            VIEWPORT,
            "inf_editor_core::voxel_store::EditorVoxelVolumes",
        ),
        ("shipped player", PLAYER, "inf_voxel::VoxelVolumes"),
    ] {
        let src = read(path).replace("\r\n", "\n");
        assert!(
            src.contains(own_store),
            "the {label} no longer holds a voxel store ({own_store})"
        );
        // …and it releases what the projection stopped drawing. A loaded volume is
        // its whole decoded chunk set plus its meshed surface — real megabytes held
        // on behalf of an entity that may have been deleted.
        assert!(
            src.contains("retain_only"),
            "the {label} never releases dead voxel volumes"
        );
    }
    // The editor's Ring-1 store is a *resolver*, not a second implementation: it
    // must delegate to the Ring-0 one rather than parse or mesh a payload itself.
    let store = read("editor/crates/inf-editor-core/src/voxel_store.rs").replace("\r\n", "\n");
    assert!(
        store.contains("VoxelVolumes"),
        "the editor store is not backed by the Ring-0 one"
    );
    for banned in ["VoxelAssetReader", "mesh_chunk", "VoxelMeshCache"] {
        assert!(
            !store.contains(banned),
            "the editor's voxel store names `{banned}` — parsing and meshing belong \
             to `inf_voxel::VoxelVolumes`, which is what makes the two hosts agree"
        );
    }
}
