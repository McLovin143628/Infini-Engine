//! The **biome-binding MIRROR gate** (P19.3): the editor's evaluate command and
//! the shipped player's load-time pass must place the same instances.
//!
//! A terrain's `biome_population` is a derived cache neither side persists, so it
//! is recomputed twice — once by `pcg_evaluate_biomes` when an author asks, once
//! by `evaluate_biome_bindings` when a level loads. If those two ever disagree,
//! the symptom is *the shipped world grows different plants from the preview*,
//! which is found by a player, not by a compiler. `runtime/inf-player/tests/
//! biome_pcg.rs` gates the player half end to end (cooked == uncooked ==
//! PIE); this gates that the **editor half is the same computation**.
//!
//! # Why source text, and why this is enough
//!
//! The Ring-2 command lives behind `#[tauri::command]` in `inf-studio`, which
//! this crate cannot link (rings 1 → 2 is the wrong direction, and `inf-studio`
//! is a binary). So the gate reads both files' **source text** — the same
//! technique, and the same rationale, as `projector_mirror.rs`.
//!
//! It is enough because of *how* the parity was built: everything from a biome-set
//! GUID onward — which biomes dispatch, in what order, under which feather, over
//! what region, against which layer source — is a handful of named Ring-0 seams
//! (`BiomeBinding::from_set`, `DEFAULT_BIOME_FEATHER`, `TerrainData::xz_bounds`,
//! `OffsetTerrain`), shared verbatim. What each side owns is only the *fetch* (a
//! content root vs. a pack). So asserting that both sides reach for the same
//! seams is asserting that neither has grown a second opinion — and a side that
//! re-derived one of them locally would stop naming it and fail here.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// The Ring-2 editor command: "evaluate this terrain's biomes now".
const EDITOR: &str = "editor/studio/src-tauri/src/commands/pcg.rs";
/// The shipped/PIE player's load-time pass.
const PLAYER: &str = "runtime/inf-player/src/level.rs";

/// Both sides must reach for the **same Ring-0 seams**, by name.
#[test]
fn both_evaluation_paths_go_through_the_same_binding_seams() {
    let editor = read(EDITOR);
    let player = read(PLAYER);
    for needle in [
        // The dispatch itself: which biomes, in what order.
        "BiomeBinding::from_set(",
        // The blend width. A side that hard-coded a number instead would not
        // name the constant — which is exactly the drift this catches.
        "DEFAULT_BIOME_FEATHER",
        // The region: the terrain's own authored extent, computed once in
        // `inf-terrain` rather than re-derived from tile counts on either side.
        ".xz_bounds()",
        // The layer source + the world-offset rule ("the terrain is at origin").
        "OffsetTerrain::new(",
        // The height seam is built from that same wrapper, so the two cannot
        // disagree about which space a query is in.
        "FnHeight::new(",
        // The population lands in the derived cache, never anywhere else.
        "biome_population",
    ] {
        assert!(
            editor.contains(needle),
            "the editor command no longer uses `{needle}`"
        );
        assert!(
            player.contains(needle),
            "the player pass no longer uses `{needle}`"
        );
    }
}

/// Both sides must resolve a `.inf_pcg` payload to a runtime document the **same
/// way**: re-lower the stored authored graph when there is one, else fall back to
/// the stored lowered mirror.
///
/// This is the subtlest way the two could drift — a side that trusted the stored
/// `document` while the other re-lowered would silently ship a stale graph, and
/// nothing would fail until an author edited a graph and shipped it.
#[test]
fn both_paths_prefer_the_authored_graph_over_the_stored_mirror() {
    for (label, src) in [("editor", read(EDITOR)), ("player", read(PLAYER))] {
        assert!(
            src.contains("lower_graph")
                && src.contains("payload.graph()")
                && src.contains("payload.document.clone()"),
            "{label} no longer resolves a .inf_pcg by re-lowering its graph"
        );
    }
}

/// Neither side may fold a *volume* seed into a biome dispatch, or a biome id
/// into a volume: the two passes are siblings and their seed rules are distinct.
///
/// The volume path salts with `wrapping_add(vol.seed)`; the biome path salts
/// through `biome_seed`, inside `BiomeBinding`. A copy-paste that carried the
/// volume's line into the biome pass would make two biomes with one graph
/// co-place, which is the exact bug `biome_seed` exists to prevent.
#[test]
fn the_two_passes_keep_their_seed_rules_apart() {
    let player = read(PLAYER);
    let biome_pass = {
        let start = player
            .find("pub fn evaluate_biome_bindings(")
            .expect("the player's biome pass");
        let rest = &player[start..];
        let end = rest.find("\n}\n").expect("terminates at column 0") + 3;
        &rest[..end]
    };
    assert!(
        !biome_pass.contains("wrapping_add"),
        "the biome pass grew the VOLUME's seed rule — biome seeds are salted by \
         `biome_seed` inside the binding, not added here"
    );
    // …and the volume pass is still its own function, i.e. the binding did not
    // quietly replace it (it is a sibling, not a replacement).
    assert!(
        player.contains("pub fn evaluate_pcg_volumes("),
        "the volume path must survive alongside the binding"
    );
}
