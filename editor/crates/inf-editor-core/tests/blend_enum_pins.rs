//! **The three-variant blend enum, spelled four times — the pairs pinned**
//! (P26.5).
//!
//! `docs/memos/p26-4-carried-debts.md` §1 measured the consolidation and carried
//! it, with an instruction for this batch:
//!
//! > The three enums are in three crates none of which may depend on another …
//! > So "one spelling" means a **new Ring-0 crate below all three** holding a
//! > `Blend` enum, and then three wire formats re-pointed at it. Each of those is
//! > a frozen, append-only discriminant written into shipped files (the P19 law),
//! > so the change is a schema migration on `.inf_lvl`, `.inf_mat` and
//! > `.inf_matd` at once … **Carried, with the pinning extended instead.** The
//! > cheap, honest half is to pin the pairs that are *not* pinned, which costs
//! > nothing and catches the real failure. That belongs with P26.5's gate work.
//!
//! The four spellings and where each is frozen:
//!
//! | spelling | crate | wire |
//! |---|---|---|
//! | `inf_ecs::BlendMode` | `inf-ecs` | the `.inf_lvl` scene record |
//! | `inf_material::MatBlend` | `inf-material` | the authored `.inf_mat` |
//! | `inf_asset::DerivedBlend` | `inf-asset` | the `.inf_matd` pack record |
//! | the Ring-2 string map | `commands/scene.rs` | the IPC boundary |
//!
//! `inf_material::derive`'s `the_three_blend_modes_map_to_three_distinct_values`
//! pins the **middle** pair. This file pins the other two, and it lives here for
//! the `projector_mirror` reason: `inf-editor-core` links `inf-ecs`,
//! `inf-material` **and** `inf-asset`, compiles on all three CI legs, and can
//! read Ring-2's source text without depending on Tauri.
//!
//! # What a break actually looks like
//!
//! The memo's own answer to "why is this cheap enough to leave unconsolidated" is
//! that *"its failure mode (a fourth variant added to one and not the others) is
//! a compile error at every mapping site because all three matches are
//! exhaustive."* That is true of the **enum-to-enum** mappings and false of the
//! **string** one: `MatBlend::Translucent => "Translucent"` is a literal, and a
//! variant renamed in `inf-ecs` turns it into a string that
//! `PropValue::Enum` cannot resolve — which is not a compile error and not a
//! panic. It is a surface that silently keeps whatever blend mode it had.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

/// Read a workspace file, normalized to LF.
///
/// **CRLF-tolerant** (the P22 law): `.rs` is read by tests, and on a Windows
/// checkout under `core.autocrlf = true` a substring search for a `\n`-joined
/// pattern finds nothing and the gate aborts with a message about something
/// else entirely.
fn read(rel: &str) -> String {
    let p = workspace_root().join(rel);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
        .replace("\r\n", "\n")
}

/// The three variants, in wire order — the order every one of the four spellings
/// declares them in, and therefore the order their bincode discriminants take.
const VARIANTS: [&str; 3] = ["Opaque", "Masked", "Translucent"];

/// **`inf_ecs::BlendMode` and `inf_asset::DerivedBlend` agree, variant for
/// variant and code for code.**
///
/// The two wire enums at the ends of the chain: one is written into a `.inf_lvl`
/// scene record and read by the editor, the player and the cook; the other into
/// a `.inf_matd` and read by a shipped player that must never link
/// `inf-material`. Nothing in the tree maps one onto the other directly — they
/// meet only through `MatBlend` in the middle — so nothing in the tree would
/// have noticed them diverging.
///
/// Asserted on the **serialized form**, not on `Debug`: these are frozen,
/// append-only discriminants written into shipped files (the P19 law), so what
/// has to agree is the byte a runtime reads, and a `#[serde(rename)]` on either
/// side is exactly the change that looks harmless and is not.
#[test]
fn the_scene_blend_and_the_derived_blend_agree_variant_for_variant() {
    use inf_asset::DerivedBlend as D;
    use inf_ecs::components::BlendMode as B;

    let scene = [B::Opaque, B::Masked, B::Translucent];
    let derived = [D::Opaque, D::Masked, D::Translucent];

    for (i, name) in VARIANTS.iter().enumerate() {
        // The JSON tag is the variant name a `PropValue::Enum` round-trips
        // through and the name a sidecar TOML carries.
        let a = serde_json::to_string(&scene[i]).expect("scene blend serializes");
        let b = serde_json::to_string(&derived[i]).expect("derived blend serializes");
        assert_eq!(a, format!("\"{name}\""), "inf_ecs::BlendMode::{name} moved");
        assert_eq!(
            b,
            format!("\"{name}\""),
            "inf_asset::DerivedBlend::{name} moved"
        );
        assert_eq!(a, b, "the two wire enums disagree about variant {i}");
    }

    // …and the POSITIONAL discriminant, which is what bincode writes. A variant
    // inserted in the middle of either enum is a silent re-interpretation of
    // every shipped file — the bincode-positional LAW, met for the fourth time.
    for (i, (s, d)) in scene.iter().zip(&derived).enumerate() {
        let sb = bincode::serde::encode_to_vec(s, bincode::config::standard())
            .expect("scene blend encodes");
        let db = bincode::serde::encode_to_vec(d, bincode::config::standard())
            .expect("derived blend encodes");
        assert_eq!(sb, db, "variant {i} takes a different bincode code");
        assert_eq!(sb, vec![i as u8], "variant {i} is not at code {i}");
    }

    // ANTI-VACUITY: three DISTINCT codes. Two enums that both collapsed onto one
    // variant would satisfy every equality above.
    let codes: Vec<Vec<u8>> = scene
        .iter()
        .map(|s| bincode::serde::encode_to_vec(s, bincode::config::standard()).unwrap())
        .collect();
    for i in 0..codes.len() {
        for j in (i + 1)..codes.len() {
            assert_ne!(codes[i], codes[j], "two blend modes share a code");
        }
    }
}

/// **The Ring-2 string map names variants `inf_ecs::BlendMode` actually has.**
///
/// `scene_apply_material` maps `inf_material::MatBlend` onto a **string** and
/// hands it to `SceneDoc::edit_apply_material`, which writes it through
/// `PropValue::Enum`. That is the one edge of the four where a mismatch is not a
/// compile error: rename a variant in `inf-ecs` and the literal here becomes a
/// name reflection cannot resolve, so the surface silently keeps the blend mode
/// it already had — a masked cutout that stops being masked, with nothing in the
/// build saying so.
///
/// A source gate because the mapping is Ring-2 and this crate must not link
/// Tauri. Comment-stripped, on the P26.1 audit's finding that a claim in a
/// comment satisfied a gate whose message said the opposite.
#[test]
fn the_ring_two_blend_strings_are_real_ecs_variants() {
    let src = read("editor/studio/src-tauri/src/commands/scene.rs");
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    let mut found = 0;
    for name in VARIANTS {
        let arm = format!("inf_material::MatBlend::{name} => \"{name}\"");
        assert!(
            code.contains(&arm),
            "the Ring-2 blend map does not spell `{arm}` — either a variant was \
             renamed on one side only, or the map now sends a string \
             `PropValue::Enum` cannot resolve"
        );
        found += 1;
        // …and the string really is a variant this ECS enum round-trips.
        let parsed: inf_ecs::components::BlendMode = serde_json::from_str(&format!("\"{name}\""))
            .unwrap_or_else(|e| {
                panic!("the Ring-2 map sends {name:?}, which inf_ecs::BlendMode rejects: {e}")
            });
        assert_eq!(
            serde_json::to_string(&parsed).unwrap(),
            format!("\"{name}\"")
        );
    }
    assert_eq!(found, 3, "the sweep did not cover all three variants");

    // ANTI-VACUITY: the strip left the code behind, not only the comments — the
    // block this reads is inside a doc-heavy function and a filter that ate it
    // would make every `contains` above fail loudly rather than pass, but the
    // reverse (a filter that ate the COMMENTS' warnings and left a stale map)
    // is what this guards.
    assert!(
        code.contains("fn scene_apply_material"),
        "the source filter ate the command this gate reads"
    );
    // The map is EXHAUSTIVE over `MatBlend`, so a fourth variant added to
    // `inf-material` is a compile error at that match rather than a silent
    // fall-through — which is the property the memo's "it is a compile error"
    // argument rests on, checked rather than assumed.
    assert!(
        !code.contains("MatBlend::_ =>") && !code.contains("_ => \"Opaque\""),
        "the Ring-2 blend map has a catch-all arm, so a fourth blend mode would \
         silently become an existing one"
    );
}
