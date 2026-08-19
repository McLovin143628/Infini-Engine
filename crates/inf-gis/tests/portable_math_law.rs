//! **The libm source gate for `inf-gis`** (Wave-G audit).
//!
//! # Why this file exists
//!
//! Seven crates in this tree carry a source gate over `inf_math::libm_ban::ALL`
//! — `inf-terrain`, `inf-pcg`, `inf-physics`, `inf-dcc`, `inf-anim`,
//! `inf-editor-core`'s erosion mirror, `inf-render`'s trig law. `inf-gis` did
//! not, and the Wave-G audit found that the module whose header spends three
//! paragraphs on why adjacent tile edges must be **bit-identical** computes the
//! lat/lon half of the same arithmetic with `.tan()`, `.ln()`, `.exp()`,
//! `.atan()`, `.cos()` and `.log2()` — the last of which was not even on the
//! canonical ban list until this audit put it there.
//!
//! The law is not "no transcendentals". It is **a value two machines re-derive
//! independently may not depend on one**. So this gate does two things:
//!
//! 1. It holds every module that is *already* clean to the full list, so the
//!    clean ones cannot quietly stop being clean. `classify`, `feature`,
//!    `roads`, `triangulate`, `vector` and `terrarium` are in that set today —
//!    a genuinely good state that nothing was protecting.
//! 2. It pins the exempt set to **exactly** the two modules that are not, by
//!    name, with the reason and the release condition written down. An exemption
//!    that is enumerated is a debt; an exemption that is invisible is a defect.
//!
//! # The exemption, and what retires it
//!
//! `crs.rs` and `tilemath.rs` are projection code. A projection *is*
//! transcendental — there is no `psin`-shaped replacement for an inverse
//! Mercator or for `proj4rs`, which is a third-party crate this gate cannot see
//! inside anyway. What makes the exemption survivable today is that **nothing in
//! the shipped tree cooks anything from either module**: `tilemath`'s tile
//! selection has no caller outside its own tests, and `crs::Transform` is used
//! at an import door whose output an author then saves as an ordinary
//! `.inf_lvl` — one machine's numbers, committed once, never re-derived
//! elsewhere and compared.
//!
//! **The day a cook or a PIE-==-shipping trace re-derives a coordinate through
//! either module, this exemption has to go** — and the way that is enforced is
//! `inf_gis_is_not_linked_by_the_cook_or_the_runtime` below, which fails if any
//! manifest a shipped binary is built from starts naming this crate.

/// Every `inf-gis` source, and whether it is exempt from the transcendental ban.
///
/// Kept as an exhaustive table rather than a directory walk on purpose: a new
/// module is a deliberate decision about which half of this list it joins, and a
/// walk would silently put it in the clean half and then fail confusingly.
const SOURCES: &[(&str, &str, bool)] = &[
    ("classify.rs", include_str!("../src/classify.rs"), false),
    ("crs.rs", include_str!("../src/crs.rs"), true),
    ("epsg.rs", include_str!("../src/epsg.rs"), false),
    ("feature.rs", include_str!("../src/feature.rs"), false),
    ("lib.rs", include_str!("../src/lib.rs"), false),
    ("roads.rs", include_str!("../src/roads.rs"), false),
    ("terrarium.rs", include_str!("../src/terrarium.rs"), false),
    ("tilemath.rs", include_str!("../src/tilemath.rs"), true),
    (
        "triangulate.rs",
        include_str!("../src/triangulate.rs"),
        false,
    ),
    ("vector.rs", include_str!("../src/vector.rs"), false),
];

/// Lines of `src` containing `needle`, ignoring comment lines — the bans are on
/// code, and the module docs necessarily *name* the things they ban.
///
/// CRLF-safe by construction (`str::lines` strips a trailing `\r`), which
/// matters because `.rs` is `text eol=lf` in `.gitattributes` precisely so a
/// Windows checkout hands a gate the same bytes a Linux one does.
fn code_hits(source: &str, needle: &str) -> Vec<(usize, String)> {
    source
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let t = line.trim_start();
            !t.starts_with("//") && !t.starts_with('*') && line.contains(needle)
        })
        .map(|(i, line)| (i + 1, line.trim().to_string()))
        .collect()
}

/// Every module outside the named exempt set is clean of std transcendentals.
#[test]
fn no_std_transcendentals_outside_the_named_exemptions() {
    const GATE: &str = "inf-gis/tests/portable_math_law.rs";
    let banned: Vec<&str> = inf_math::libm_ban::ALL.to_vec();
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    for (name, src, exempt) in SOURCES {
        if *exempt {
            continue;
        }
        for needle in &banned {
            let hits = code_hits(src, needle);
            assert!(
                hits.is_empty(),
                "{GATE}: inf-gis/src/{name} calls `{needle}`, which is not \
                 bit-portable across targets (the P14 law). Use the \
                 `inf_math::portable` replacement, or — if this module genuinely \
                 has to project — add it to this file's exempt table WITH the \
                 reason and the condition that retires it. Hits: {hits:?}"
            );
        }
    }
}

/// **The exemption is a debt, so it is measured.**
///
/// A gate that merely skipped two files would be indistinguishable from a gate
/// somebody had switched off. This one asserts that the exempt modules really
/// are the ones that need it (they contain projection math) and — the part that
/// matters — that the exempt set has not silently grown.
#[test]
fn the_exempt_set_is_exactly_the_projection_modules() {
    let exempt: Vec<&str> = SOURCES
        .iter()
        .filter(|(_, _, e)| *e)
        .map(|(n, _, _)| *n)
        .collect();
    assert_eq!(
        exempt,
        vec!["crs.rs", "tilemath.rs"],
        "the transcendental exemption has moved. Adding a module to it is a \
         deliberate act that needs its reason written down in this file's header \
         and its release condition named; removing one is good news that should \
         come with the portable replacement it now uses."
    );

    // …and each exempt module actually contains what it is exempt for, so an
    // exemption cannot outlive the code that justified it.
    for (name, src, _) in SOURCES.iter().filter(|(_, _, e)| *e) {
        let uses_transcendental = inf_math::libm_ban::ALL
            .iter()
            .any(|needle| !code_hits(src, needle).is_empty());
        assert!(
            uses_transcendental,
            "inf-gis/src/{name} is on the exempt list and no longer calls a \
             banned function — take it off the list rather than leaving a \
             standing exemption nothing needs"
        );
    }
}

/// **The condition that retires the exemption**, asserted rather than promised.
///
/// The exemption rests on one fact: nothing that *cooks* or that a
/// PIE-==-shipping trace compares re-derives a coordinate through `crs` or
/// `tilemath`. `inf-gis` is a host-only crate and the shipped player never links
/// it — that is stated in the crate docs and in the workspace manifest, and this
/// is the arm that keeps it true.
#[test]
fn inf_gis_is_not_linked_by_the_cook_or_the_runtime() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    // Every manifest that a shipped binary is built from.
    for rel in [
        "runtime/inf-player/Cargo.toml",
        "runtime/inf-packager/Cargo.toml",
        "crates/inf-runtime/Cargo.toml",
        "tools/inf-cli/Cargo.toml",
    ] {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // Only the DEPENDENCY sections matter; a comment naming the crate is
        // fine and is in fact how the refusal is recorded.
        let dep_lines: Vec<&str> = src
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with('#'))
            .collect();
        assert!(
            !dep_lines.iter().any(|l| l.starts_with("inf-gis")),
            "{rel} names `inf-gis` as a dependency. That crate's projection math \
             is exempt from the portability gate on the grounds that nothing \
             which cooks or ships re-derives a coordinate through it — linking it \
             into a cooked or shipped path retires that exemption, and \
             `crs.rs`/`tilemath.rs` need portable replacements before it can."
        );
    }
}
