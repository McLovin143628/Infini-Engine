//! The **grammar-evaluation MIRROR gate** (P19.4): the editor's `pcg_evaluate`
//! and the shipped player's load-time pass must build the same grammar
//! population.
//!
//! A `PcgVolume`'s `evaluated` cache is derived and never persisted, so it is
//! recomputed twice — once when an author presses ⚡ Evaluate, once when a level
//! loads. P19.3 gated that for the *scatter* half; this gates the grammar half,
//! whose failure mode is worse: a wall built from a different derivation is not
//! "slightly different foliage", it is a building with the doors somewhere else.
//!
//! # Why source text, and why this is enough
//!
//! Same technique and same rationale as `biome_binding_mirror.rs` and
//! `projector_mirror.rs`: the Ring-2 command lives behind `#[tauri::command]` in
//! the `inf-studio` binary, which this crate cannot link.
//!
//! It is enough because of *how* the parity was built. Everything downstream of
//! a resolved span is one Ring-0 function, `inf_pcg::evaluate_grammars`, and
//! everything downstream of a `.inf_pcg` payload is one lowering,
//! `lower_graph(_with)`. What each side owns is the **fetch**: turning its own
//! ECS world into `SplinePath`s, and turning its own volume into a
//! `GrammarContext`. Those two are the drift surface, so this gate reads them
//! character for character rather than merely checking that both files mention
//! the words.

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

/// The Ring-2 editor command: "scatter and build this volume now".
const EDITOR: &str = "editor/studio/src-tauri/src/commands/pcg.rs";
/// The shipped/PIE player's load-time pass.
const PLAYER: &str = "runtime/inf-player/src/level.rs";

/// The span fetch — the one piece of the grammar path each host writes for
/// itself — must be **the same code**, not merely the same idea.
///
/// The comparison is on the function body with whitespace collapsed, so
/// `rustfmt` wrapping differently on either side is not a failure but a changed
/// transform, a changed component, or a forgotten `GlobalTransform` is.
#[test]
fn the_spline_fetch_is_character_identical_on_both_sides() {
    let editor = body_of(&read(EDITOR), "fn collect_spline_paths(");
    let player = body_of(&read(PLAYER), "pub fn collect_spline_paths(");
    assert_eq!(
        squash(&editor),
        squash(&player),
        "the editor and the player resolve splines differently — one of them will \
         build a wall the other does not"
    );
    // The body really does what the parity depends on: world-space points, by
    // stable Guid, through the shared Ring-0 constructor.
    for needle in [
        "SplinePath::from_local(",
        "GlobalTransform",
        "DAffine3::IDENTITY",
        "SplineInterp::CatmullRom",
    ] {
        assert!(
            editor.contains(needle),
            "the shared spline fetch stopped using `{needle}`"
        );
    }
}

/// Both sides must reach for the **same Ring-0 seams**, by name — a side that
/// re-derived one locally would stop naming it and fail here.
#[test]
fn both_evaluation_paths_go_through_the_same_grammar_seams() {
    let editor = read(EDITOR);
    let player = read(PLAYER);
    for needle in [
        // The expansion itself: spans, derivation, layout, placement.
        "inf_pcg::evaluate_grammars(",
        // The per-volume inputs. A side that folded the volume seed its own way
        // (the exact mistake `biome_seed` exists to prevent one type down) would
        // not construct this.
        "inf_pcg::GrammarContext {",
        "seed_offset:",
        // The span fetch, by name.
        "collect_spline_paths(",
    ] {
        assert!(editor.contains(needle), "editor lost `{needle}`");
        assert!(player.contains(needle), "player lost `{needle}`");
    }
    // Neither side may hand the grammar a DIFFERENT height provider from the one
    // its scatter used: a grammar snapping to another ground than the scatter
    // beside it is invisible until somebody walks the level. Both sides call
    // `evaluate` and `evaluate_grammars` with the same provider expression.
    for (label, src, provider) in [
        ("editor", &editor, "provider.as_ref()"),
        ("player", &player, "&provider"),
    ] {
        let scatter = format!("inf_pcg::evaluate(&{}, {provider}", doc_arg(label));
        assert!(
            src.contains(&scatter),
            "{label}: scatter no longer uses `{provider}`"
        );
        assert!(
            src.contains(&format!("{provider},\n")) || src.contains(&format!("{provider}\n")),
            "{label}: the grammar no longer uses `{provider}`"
        );
    }
}

/// The argument name each side's document happens to have — the only thing that
/// legitimately differs between the two `inf_pcg::evaluate` calls.
fn doc_arg(side: &str) -> &'static str {
    if side == "editor" {
        "document"
    } else {
        "job.document"
    }
}

/// The braced body following the first occurrence of `signature`.
fn body_of(src: &str, signature: &str) -> String {
    let start = src
        .find(signature)
        .unwrap_or_else(|| panic!("no `{signature}` — the mirror moved or was renamed"));
    let open = src[start..]
        .find('{')
        .unwrap_or_else(|| panic!("no body after `{signature}`"))
        + start;
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return src[open..=i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after `{signature}`");
}

/// Collapse all whitespace runs to one space, so formatting differences between
/// two files are not a mirror failure but content differences are.
fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
