//! **The caps, measured at island scale** (IB-14).
//!
//! The certification measured the default entity cap amputating a city:
//!
//! | layer | features | spawned | truncated |
//! |---|---|---|---|
//! | roads | 10 000 | 4 096 | **5 904** |
//! | footprints | 50 000 | 4 096 | **45 904** |
//!
//! and its complaint was never that the truncation was silent — it is reported —
//! but that "with no wizard there is nowhere for an author to raise it and
//! nowhere for the report to be shown". This file is the other half of the
//! answer to that: the cap is a value at the import door, the default still
//! guards, and **the certification's own two layers import whole** when an
//! author asks for it.
//!
//! # No wall-clock assertion lives here
//!
//! The counts are asserted; the timings are printed. A ratio between two
//! measurements is not a measurement on a shared CI runner (the P29.1 law, and
//! its second face on the Windows runner), and there is nothing here whose
//! correctness is a duration.

use std::io::Write;
use std::path::{Path, PathBuf};

use inf_gis::feature::LayerKind;
use inf_gis::{anchor_at, import_layer, ImportRequest, DEFAULT_MAX_ENTITIES, ISLAND_MAX_ENTITIES};

/// Vancouver, UTM zone 10N — the anchor the island phase is pinned to.
fn anchor() -> inf_math::geo::GeoAnchor {
    anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008").unwrap()
}

/// `n` road centrelines on a grid, each two vertices and ~40 m long, written as
/// GeoJSON in the anchor's own CRS (so the import is exact and the test measures
/// the cap rather than the projection).
fn write_roads(dir: &Path, n: usize) -> PathBuf {
    let p = dir.join("roads.geojson");
    let f = std::fs::File::create(&p).unwrap();
    let mut w = std::io::BufWriter::new(f);
    write!(w, r#"{{"type":"FeatureCollection","features":["#).unwrap();
    let side = (n as f64).sqrt().ceil() as usize;
    for i in 0..n {
        let (gx, gz) = (i % side, i / side);
        let x = 491_000.0 + gx as f64 * 50.0;
        let z = 5_459_000.0 + gz as f64 * 50.0;
        if i > 0 {
            write!(w, ",").unwrap();
        }
        write!(
            w,
            r#"{{"type":"Feature","properties":{{"name":"R{i}","lanes":2}},"geometry":{{"type":"LineString","coordinates":[[{x},{z}],[{},{z}]]}}}}"#,
            x + 40.0
        )
        .unwrap();
    }
    write!(w, "]}}").unwrap();
    w.flush().unwrap();
    p
}

/// `n` building footprints on a grid, each a 20 × 14 m rectangle.
fn write_footprints(dir: &Path, n: usize) -> PathBuf {
    let p = dir.join("footprints.geojson");
    let f = std::fs::File::create(&p).unwrap();
    let mut w = std::io::BufWriter::new(f);
    write!(w, r#"{{"type":"FeatureCollection","features":["#).unwrap();
    let side = (n as f64).sqrt().ceil() as usize;
    for i in 0..n {
        let (gx, gz) = (i % side, i / side);
        let x = 491_000.0 + gx as f64 * 30.0;
        let z = 5_459_000.0 + gz as f64 * 30.0;
        if i > 0 {
            write!(w, ",").unwrap();
        }
        write!(
            w,
            r#"{{"type":"Feature","properties":{{"name":"B{i}","levels":3}},"geometry":{{"type":"Polygon","coordinates":[[[{x},{z}],[{},{z}],[{},{}],[{x},{}],[{x},{z}]]]}}}}"#,
            x + 20.0,
            x + 20.0,
            z + 14.0,
            z + 14.0
        )
        .unwrap();
    }
    write!(w, "]}}").unwrap();
    w.flush().unwrap();
    p
}

fn plan(path: &Path, kind: LayerKind, max: usize) -> inf_gis::SpawnPlan {
    let mut req = ImportRequest::new(path, kind);
    req.source_crs = Some("EPSG:32610".into());
    req.options.max_entities = max;
    // The 20 m footprint perimeter is 68 m and every road is 40 m, so the stub
    // floor never fires — the cap is the only thing this measures.
    req.options.min_length_m = 8.0;
    import_layer(&req, &anchor()).unwrap().plan
}

/// **The certification's own two numbers, reproduced — and then retired.**
///
/// Both halves matter. The default still amputates, which is what makes it a
/// guard; and the same two layers import WHOLE at [`ISLAND_MAX_ENTITIES`],
/// which is what makes it raisable rather than a ceiling.
#[test]
fn ten_thousand_roads_and_fifty_thousand_footprints_import_whole() {
    let dir = tempfile::tempdir().unwrap();

    // ── the cert's measurement, reproduced at the default ───────────────────
    let roads = write_roads(dir.path(), 10_000);
    let t = std::time::Instant::now();
    let capped = plan(&roads, LayerKind::Roads, DEFAULT_MAX_ENTITIES);
    let capped_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(capped.count(), DEFAULT_MAX_ENTITIES);
    assert_eq!(
        capped.truncated,
        10_000 - DEFAULT_MAX_ENTITIES,
        "the certification measured 5 904 roads truncated at the default cap"
    );
    let s = capped.summary("Roads");
    assert!(s.contains("5904 NOT IMPORTED"), "{s}");
    assert!(
        s.contains("the entity cap is 4096") && s.contains("--max"),
        "the remedy names the number AND where to change it: {s}"
    );

    // ── raised: the whole layer ─────────────────────────────────────────────
    let t = std::time::Instant::now();
    let whole = plan(&roads, LayerKind::Roads, ISLAND_MAX_ENTITIES);
    let whole_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(whole.count(), 10_000, "{:?}", whole.summary("Roads"));
    assert_eq!(whole.truncated, 0);
    assert_eq!(whole.too_short, 0);
    assert_eq!(whole.unusable, 0);
    println!(
        "IB-14 roads: 10 000 features -> {} at the default cap ({} truncated), \
         {} whole [{capped_ms:.0} ms / {whole_ms:.0} ms]",
        capped.count(),
        capped.truncated,
        whole.count()
    );

    // ── the footprints, the same two ways ───────────────────────────────────
    let feet = write_footprints(dir.path(), 50_000);
    let capped = plan(&feet, LayerKind::Buildings, DEFAULT_MAX_ENTITIES);
    assert_eq!(capped.count(), DEFAULT_MAX_ENTITIES);
    assert_eq!(
        capped.truncated,
        50_000 - DEFAULT_MAX_ENTITIES,
        "the certification measured 45 904 footprints truncated"
    );

    let t = std::time::Instant::now();
    let whole = plan(&feet, LayerKind::Buildings, ISLAND_MAX_ENTITIES);
    let whole_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(whole.count(), 50_000, "{:?}", whole.summary("Footprints"));
    assert_eq!(whole.truncated, 0);
    assert!(
        whole
            .entities
            .iter()
            .all(|e| matches!(e.kind, inf_gis::PlannedKind::Spline { closed: true })),
        "a footprint's boundary is a closed ring"
    );
    println!(
        "IB-14 footprints: 50 000 features -> {} at the default cap ({} truncated), \
         {} whole [{whole_ms:.0} ms]",
        capped.count(),
        capped.truncated,
        whole.count()
    );
}

/// The island ceiling has headroom past both layers, which is what makes it a
/// setting and not a second cap in disguise.
///
/// A **compile-time** assertion, not a runtime one: this is a relation between
/// two constants, and a runtime `assert!` over constants is a test that cannot
/// tell you anything a build could not.
const _: () = assert!(
    ISLAND_MAX_ENTITIES > 100_000,
    "the island cap must clear the certification's 50 000-footprint layer with \
     room to spare, or raising it is just moving the amputation"
);

/// The whole-layer plan is a **function of the file**: importing it twice gives
/// the same plan, digest for digest, at island scale as at three features.
#[test]
fn an_island_scale_import_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let roads = write_roads(dir.path(), 5_000);
    let a = plan(&roads, LayerKind::Roads, ISLAND_MAX_ENTITIES);
    let b = plan(&roads, LayerKind::Roads, ISLAND_MAX_ENTITIES);
    assert_eq!(a.count(), 5_000);
    assert_eq!(
        a.digest(),
        b.digest(),
        "two imports of one file must be one plan"
    );
    // …and a different cap is a different plan, so the digest is not a constant.
    let c = plan(&roads, LayerKind::Roads, 100);
    assert_ne!(a.digest(), c.digest());
}
