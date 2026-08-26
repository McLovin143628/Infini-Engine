//! **THE TWO `apply_record` SEAMS ARE ONE FUNCTION** (wave VIS1a).
//!
//! A level's render block reaches the renderer through two copies of the same
//! mapping: `inf_player::render::apply_record` over `inf_scene::RenderSettingsRecord`,
//! and `inf_viewport::host::apply_record` over the editor codec's mirror of it.
//! Both have carried a `MIRROR: keep identical to …` doc comment since R-P4 and
//! **nothing has ever checked it**. A comment is not a pin: the two are in
//! different crates, in different rings, and the failure mode is silent —
//! preview and shipping applying a level's block differently, which is the exact
//! claim `PIE == shipping` rests on.
//!
//! Wave VIS1a doubled the size of that mapping (twenty-two appended fields), so
//! it is also the wave that has to stop trusting the sentence.
//!
//! **What is compared**: the *body* of each `fn apply_record`, with comments and
//! whitespace stripped and the two record types' names normalised away. What is
//! deliberately NOT compared is the doc comment above each — they say different
//! things about which copy is which, and they should.
//!
//! **Both files are read as text at compile time** (`include_str!`), so this arm
//! has to be built and run in the same worktree — the P23 `determinism_law`
//! constraint, documented on the gate rather than worked around. `.rs` is
//! `text eol=lf` in `.gitattributes`, which is what makes the substring search
//! below survive a Windows checkout (the P22 CRLF law).

/// The player's copy.
const PLAYER: &str = include_str!("../src/render.rs");
/// The editor viewport's copy.
const VIEWPORT: &str = include_str!("../../../editor/crates/inf-viewport/src/host.rs");

/// The body of `fn apply_record` — from the opening brace of the signature to
/// the matching close — with comments stripped and whitespace collapsed.
fn body(src: &str, what: &str) -> String {
    let at = src
        .find("fn apply_record(r: &RenderSettingsRecord) -> RenderSettings {")
        .unwrap_or_else(|| panic!("{what}: no `fn apply_record` with the expected signature"));
    let open = src[at..]
        .find('{')
        .expect("a function signature is followed by a brace")
        + at;
    let mut depth = 0usize;
    let mut end = open;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(end > open, "{what}: unbalanced braces in `apply_record`");
    let mut out = String::new();
    for line in src[open + 1..end].lines() {
        let code = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        let t = code.trim();
        if t.is_empty() {
            continue;
        }
        out.push_str(t);
        out.push('\n');
    }
    out
}

/// **The pin.** Two crates, two rings, one mapping.
///
/// Mutation-verified while it was written: changing a single default on one side
/// (`..d.gi` to `..GiSettings::default()`) fails it, and so does dropping any one
/// of the twenty-two v26 lines from either copy.
#[test]
fn the_two_apply_record_seams_are_character_for_character_the_same() {
    let a = body(PLAYER, "inf_player::render");
    let b = body(VIEWPORT, "inf_viewport::host");

    // Anti-vacuity: a bad extraction that returned nothing would compare equal.
    assert!(
        a.lines().count() > 30,
        "the player's `apply_record` body extracted as {} lines — the extraction \
         is broken, and an empty comparison passes for the wrong reason",
        a.lines().count()
    );
    assert!(
        a.contains("ssr: SsrSettings {") && a.contains("film: FilmSettings {"),
        "the extracted body does not contain the v26 block — either the seam \
         moved or the extraction did"
    );

    assert_eq!(
        a, b,
        "the two `apply_record` seams have drifted. The editor viewport and the \
         shipped player would apply a level's render block differently, which is \
         the claim `PIE == shipping` rests on — and the only thing that has ever \
         kept them together is a doc comment saying they are the same.\n\n\
         player:\n{a}\n\nviewport:\n{b}"
    );
}
