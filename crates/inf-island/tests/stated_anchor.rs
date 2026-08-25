//! **The stated geodetic origin is true** — the check a restatement owes
//! (the I7 CI-red).
//!
//! # What went wrong, exactly
//!
//! `IslandRecipe::anchor` used to invert its easting/northing into a latitude,
//! a longitude and a grid convergence through `inf_gis::anchor_at`, which is
//! `proj4rs`, which is a series over `sin`/`cos`/`atan2`. Those are the platform
//! libm's, and two libms are entitled to disagree in the last ulp — the P14 law
//! this repository has now been bitten by five times. The island's `.inf_lvl`
//! **commits** the anchor, so the disagreement was a committed byte:
//!
//! | | `origin_latitude_deg` | the f64, hex |
//! |---|---|---|
//! | Windows (blessed) | 49.34307562364773 | `0x4048ABE9E6EBCF97` |
//! | macOS (CI) | 49.34307562364772 | `0x4048ABE9E6EBCF96` |
//!
//! One byte at offset 14 788 of a 14 820-byte file, and
//! `committed_sample_matches_generators` went red on one of three platforms.
//!
//! # Why a restatement is the fix and not a rounding
//!
//! Rounding the inversion at the door would have made the byte *usually* stable
//! and never *provably* stable: a value that lands within an ulp of a rounding
//! tie rounds two ways, and "unlikely" is not a property a gate can rest on. A
//! stated number is parsed from decimal by `f64::from_str`, which is correctly
//! rounded and therefore identical on every target — the byte is now a fact
//! about the recipe, which is source.
//!
//! What a restatement owes is a check, and this is it: the degrees each
//! committed recipe states are compared against what the projection actually
//! says, inside [`ANCHOR_AGREEMENT_DEG`]. The tolerance is six orders of
//! magnitude above any libm disagreement and six below a typo, so this arm can
//! never redden from a platform and can never miss a wrong anchor.

use std::path::PathBuf;

use inf_island::{IslandRecipe, ANCHOR_AGREEMENT_DEG};

/// Every recipe this repository commits. Kept in step with
/// `inf_editor_core::island::ISLAND_RECIPES`, which authors their levels.
const RECIPES: [&str; 2] = [
    "../../samples/island/island.toml",
    "../../samples/island-fixture/island.toml",
];

fn path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// **The stated degrees are the projection's own answer**, to a tolerance that
/// no platform can move and no blunder can hide in.
#[test]
fn every_committed_recipe_states_a_true_geodetic_origin() {
    let mut checked = 0usize;
    for rel in RECIPES {
        let p = path(rel);
        let r = IslandRecipe::load(&p).unwrap_or_else(|e| panic!("load {}: {e}", p.display()));

        // The projection's own answer. This call is legal HERE — a test that
        // never writes a file — and is banned from `inf-island/src` by
        // `portable_math_law.rs`.
        let truth = inf_gis::anchor_at(
            &r.anchor.crs,
            r.anchor.easting_m,
            r.anchor.northing_m,
            r.anchor.height_m,
            &r.anchor.vertical_datum,
        )
        .unwrap_or_else(|e| panic!("{rel}: the anchor's CRS and origin do not invert: {e}"));

        for (what, stated, derived) in [
            (
                "latitude_deg",
                r.anchor.latitude_deg,
                truth.origin_latitude_deg,
            ),
            (
                "longitude_deg",
                r.anchor.longitude_deg,
                truth.origin_longitude_deg,
            ),
            (
                "convergence_deg",
                r.anchor.convergence_deg,
                truth.grid_convergence_deg,
            ),
        ] {
            let residual = (stated - derived).abs();
            println!("{rel}: {what} stated {stated} vs derived {derived} (residual {residual:e})");
            assert!(
                residual <= ANCHOR_AGREEMENT_DEG,
                "{rel} states `[anchor] {what} = {stated}` and {:?} at ({}, {}) really \
                 inverts to {derived} — {residual:e} degrees apart, past the \
                 {ANCHOR_AGREEMENT_DEG:e} this repository allows. State the \
                 projection's own number, rounded to 1e-9; do NOT make the recipe \
                 read the projection, because that answer is a fact about the \
                 host's libm and this island's level is committed.",
                r.anchor.crs,
                r.anchor.easting_m,
                r.anchor.northing_m,
            );
        }

        // …and the recipe's own door carries them through untouched, which is the
        // property the committed byte rests on.
        let a = r.anchor().unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert_eq!(a.origin_latitude_deg, r.anchor.latitude_deg);
        assert_eq!(a.origin_longitude_deg, r.anchor.longitude_deg);
        assert_eq!(a.grid_convergence_deg, r.anchor.convergence_deg);
        checked += 1;
    }
    assert_eq!(
        checked,
        RECIPES.len(),
        "a committed recipe was skipped — this arm must read every one"
    );
}

/// **The tolerance can fail** — the anti-vacuity arm.
///
/// A comparison with a tolerance wide enough to swallow anything is a
/// comparison that certifies nothing. This is the same anchor, moved by a
/// hundred metres of latitude, and it must be refused.
#[test]
fn the_agreement_tolerance_would_catch_a_wrong_origin() {
    let r = IslandRecipe::load(&path(RECIPES[0])).expect("the island recipe loads");
    let truth = inf_gis::anchor_at(
        &r.anchor.crs,
        r.anchor.easting_m,
        r.anchor.northing_m,
        r.anchor.height_m,
        &r.anchor.vertical_datum,
    )
    .expect("it inverts");

    // A hundred metres, in degrees of latitude.
    let hundred_m_deg = 100.0 / 111_132.0;
    assert!(
        hundred_m_deg > ANCHOR_AGREEMENT_DEG,
        "the tolerance admits a hundred-metre error"
    );
    let wrong = truth.origin_latitude_deg + hundred_m_deg;
    assert!((wrong - truth.origin_latitude_deg).abs() > ANCHOR_AGREEMENT_DEG);

    // …and the real residual is far inside it, so the arm above has headroom
    // rather than sitting on the line.
    let residual = (r.anchor.latitude_deg - truth.origin_latitude_deg).abs();
    assert!(
        residual * 10.0 < ANCHOR_AGREEMENT_DEG,
        "the stated latitude sits at {residual:e} against a {ANCHOR_AGREEMENT_DEG:e} \
         tolerance — less than a factor of ten of headroom is a gate waiting to \
         flake"
    );
}

/// **A recipe cannot forget its geodetic origin.**
///
/// The three degrees have no serde default on purpose: a default would be a
/// silent zero, and an island at the equator with a Vancouver easting is a world
/// whose sun is wrong all year with nothing to see.
#[test]
fn a_recipe_without_the_stated_origin_is_refused() {
    let full = std::fs::read_to_string(path(RECIPES[0])).expect("read the island recipe");
    for missing in ["latitude_deg", "longitude_deg", "convergence_deg"] {
        let text: String = full
            .lines()
            .filter(|l| !l.trim_start().starts_with(missing))
            .collect::<Vec<_>>()
            .join("\n");
        let err = IslandRecipe::parse(&text, &path("../../samples/island"))
            .expect_err("a recipe missing its stated origin must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains(missing),
            "the refusal must name the missing key, got: {msg}"
        );
    }
    // …and the unmodified text really does parse, or the three above prove
    // nothing about the missing key.
    IslandRecipe::parse(&full, &path("../../samples/island")).expect("the real recipe parses");
}
