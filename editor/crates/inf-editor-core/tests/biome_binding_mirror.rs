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
//! **which ground** — is a handful of named Ring-0 seams
//! (`BiomeBinding::from_set`, `DEFAULT_BIOME_FEATHER`, and since island wave I7b
//! `BiomeBinding::refresh_resident` / `evaluate_resident`), shared verbatim. What
//! each side owns is only the *fetch* (a content root vs. a pack). So asserting
//! that both sides reach for the same seams is asserting that neither has grown a
//! second opinion — and a side that re-derived one of them locally would stop
//! naming it and fail here.
//!
//! **`TerrainData::xz_bounds` and `OffsetTerrain` used to be on that list and are
//! now BANNED from it** (the I7b audit corrects this paragraph, which the wave
//! left describing the design it replaced). The region was three seams each side
//! had to spell identically, over a bounding box that is the right answer only
//! for a terrain entirely in memory — and `None` on the shipped boot, which is
//! how a 51 km² island grew nothing for a whole wave. The resident-tile walk is
//! `inf_pcg`'s now, so there is one name and nothing left to spell differently.

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
    // **The REGION is one door too** (island wave I7b). It used to be a region
    // built on each side from `xz_bounds()` + `OffsetTerrain` + `FnHeight` —
    // three seams each side had to spell identically, and a bounding box that is
    // only the right answer for a terrain that is entirely in memory. The walk
    // over the resident tiles is now `inf_pcg`'s, so the two sides name one
    // function and there is nothing left to spell differently.
    for (label, src) in [("editor", &editor), ("player", &player)] {
        let refresh = src.matches("refresh_resident(").count();
        let one_shot = src.matches("evaluate_resident(").count();
        assert!(
            refresh + one_shot >= 1,
            "{label} no longer evaluates a biome binding over the RESIDENT \
             ground — it has grown its own region again"
        );
        assert!(
            !src.contains(".xz_bounds()"),
            "{label} is back to scattering over the bounding box of whatever is \
             paged; a streamed terrain's bounds are a moving target and were \
             `None` on the shipped boot for the whole of wave I7"
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

/// **The two hosts translate a volume's population with ONE body** (I3).
///
/// `inf-pcg` does not depend on `inf-ecs` and `inf-studio` cannot link
/// `inf-player`, so the map from `inf_pcg::VolumeOutput` into the ECS's own
/// mirror types is written once per host and cannot be hoisted. Until IB-2b it
/// was three field-for-field copies and a drift would have been a wrong
/// *position*, which is visible. Now it carries `StructureGroup`, whose `start`,
/// `len`, `inst_start` and `inst_len` are four `u32`s in a row: **any
/// permutation of them compiles**, and the symptom is a distant building drawn
/// or collided with another building's walls — found by a player, not by a
/// compiler.
///
/// So the two bodies are compared character for character (whitespace squeezed,
/// so rustfmt's line breaking is not the subject). The markers are asserted to
/// appear **exactly once** on each side: the I1 audit's law that a `contains`
/// needle which is a prefix of a declaration can never fail applies to a fence
/// too — a second `MIRROR-BEGIN` would silently change which block is compared.
#[test]
fn the_population_mapping_is_one_body() {
    fn fenced(src: &str, who: &str) -> String {
        assert_eq!(
            src.matches("// MIRROR-BEGIN population_of").count(),
            1,
            "{who} has the wrong number of population_of fences"
        );
        assert_eq!(
            src.matches("// MIRROR-END population_of").count(),
            1,
            "{who} has the wrong number of population_of fences"
        );
        let a = src.find("// MIRROR-BEGIN population_of").expect("checked");
        let b = src.find("// MIRROR-END population_of").expect("checked");
        assert!(b > a, "{who}'s population_of fence is inverted");
        src[a..b].chars().filter(|c| !c.is_whitespace()).collect()
    }
    let editor = fenced(&read(EDITOR), "the editor");
    let player = fenced(&read(PLAYER), "the player");
    assert!(
        editor.len() > 400,
        "the fenced body is suspiciously short ({} chars) — an empty fence would \
         make this gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the editor and the player translate a PCG volume's population \
         differently. `start`/`len`/`inst_start`/`inst_len` are four u32s in a \
         row: a swap compiles and draws one building with another's walls."
    );
    // …and both really do go through the Ring-0 composition door, which is what
    // makes the ORDER of the three passes one decision rather than two.
    for (who, src) in [("editor", read(EDITOR)), ("player", read(PLAYER))] {
        assert_eq!(
            src.matches("inf_pcg::compose_volume(").count(),
            1,
            "{who} no longer joins its three passes through the Ring-0 door"
        );
        assert_eq!(
            src.matches("set_population(").count(),
            1,
            "{who} no longer writes the whole population through one setter"
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
