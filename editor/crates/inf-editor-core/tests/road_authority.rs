//! **One authority for a road's cross-section** (wave ROAD1).
//!
//! Three numbers about a road are stated in two crates each, and neither crate
//! can name the other:
//!
//! | the number | one side | the other |
//! |---|---|---|
//! | the pavement's width | `inf_gis::PAVEMENT_M` — the concrete | `inf_ecs::society::PAVEMENT_M` — the nav ring on it |
//! | a lane's width | `inf_gis::LANE_WIDTH_M` — the carriageway | `inf_nav::lane::DEFAULT_LANE_WIDTH_M` — the lane centrelines in it |
//! | the island's lane counts | `inf_island::roads::island_lanes` — the design | `samples/island/layers/roads.geojson` — the committed layer |
//!
//! `inf-editor-core` is the crate that links all of them, so this is where the
//! equalities are asserted. It is the same arrangement `inf-gis`'s own
//! `LANE_WIDTH_M` pin already uses, and the reason is CERT1's CP-B3: before this
//! wave `PAVEMENT_M` laid eight nav nodes and **nothing else** — there was no
//! concrete under the crowd walking on it — so nothing could disagree. Now two
//! things state it, and two things that state one fact are two things that can
//! drift.

use inf_gis::RoadKind;

/// **The kerb geometry and the nav ring are one pavement.**
///
/// `inf_ecs::society::volume_sites` lays a city block's eight pavement nodes
/// `PAVEMENT_M` outside the block's own rectangle, and `inf_gis::build_kerbs`
/// lays `PAVEMENT_M` of concrete behind every kerb. A crowd routed two metres
/// out while the slab under it is one and a half is a crowd walking beside its
/// own pavement — and it would look exactly like a pathfinding bug.
///
/// Falsification: change either constant and this reds.
#[test]
fn the_kerb_geometry_and_the_nav_ring_are_one_pavement() {
    assert_eq!(
        inf_gis::PAVEMENT_M,
        inf_ecs::society::PAVEMENT_M,
        "the concrete a kerb lays ({} m) and the ring a crowd walks ({} m) are \
         the same pavement — see `inf_gis::PAVEMENT_M`'s own note",
        inf_gis::PAVEMENT_M,
        inf_ecs::society::PAVEMENT_M
    );
    // …and the number is a pavement rather than a kerb: 2 m is what a person
    // passes another person on. A sanity band, so a future edit that made it
    // 0.2 or 20 fails here as well as at the equality.
    assert!(
        (1.0..=4.0).contains(&inf_gis::PAVEMENT_M),
        "{} m is not a footway",
        inf_gis::PAVEMENT_M
    );
    // The lane pin, restated from this side because the same law covers it and
    // `inf-gis`'s own arm cannot see `inf-ecs`'s re-export.
    assert_eq!(
        inf_gis::LANE_WIDTH_M,
        inf_ecs::traffic::DEFAULT_LANE_WIDTH_M
    );
}

/// **The committed road layer states the lane counts the design states.**
///
/// `inf_island::roads::island_lanes` is the design's own table and
/// `layers::write_roads` writes it, but the committed `roads.geojson` is a
/// one-time derivation that `inf island build` only ever *reads* — so the two
/// can drift for as long as nobody runs `inf island route`. This is the fence.
///
/// It also asserts the thing the wave was called for: the layer **has** a lane
/// count at all. Before ROAD1 it carried a name and a class, so every road took
/// `RoadKind::default_lanes` — four for a highway and four for an arterial
/// alike, 14.0 m of carriageway for both.
#[test]
fn the_committed_road_layer_states_the_lanes_its_design_states() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../samples/island/layers/roads.geojson");
    let bytes = std::fs::read(&path).expect("the committed road layer reads");
    let doc: serde_json::Value = serde_json::from_slice(&bytes).expect("it is JSON");
    let features = doc["features"].as_array().expect("a feature collection");
    assert_eq!(features.len(), 11, "the island's designed network");

    let mut by_class: std::collections::BTreeMap<String, Vec<u64>> = Default::default();
    for f in features {
        let props = &f["properties"];
        let class = props["road_type"].as_str().expect("every road has a class");
        let lanes = props["lanes"]
            .as_u64()
            .unwrap_or_else(|| panic!("{class} states no lane count"));
        by_class.entry(class.to_string()).or_default().push(lanes);
    }
    for (class, counts) in &by_class {
        let want = u64::from(inf_island::roads::island_lanes(class));
        assert!(
            counts.iter().all(|c| *c == want),
            "the committed layer's {class}s carry {counts:?} lanes and the \
             design's own table says {want} — `write_roads` would emit {want}, \
             so the layer and the generator have drifted"
        );
    }
    assert_eq!(
        by_class.keys().collect::<Vec<_>>(),
        vec!["arterial", "highway"],
        "one trunk route and ten town connectors"
    );

    // And the widths that follow, because a lane count is only interesting for
    // what it makes: 14.0 m of trunk against 7.0 m of connector, where before
    // this wave both were 14.0.
    assert_eq!(RoadKind::Highway.width_m(4), 14.0);
    assert_eq!(RoadKind::Arterial.width_m(2), 7.0);
    assert_eq!(
        RoadKind::Arterial.width_m(RoadKind::Arterial.default_lanes()),
        14.0,
        "the class default is unchanged — it is the right fallback for a layer \
         that says nothing, and what changed is that this one says something"
    );
}

/// **The built half-width is what the terrain has to be levelled to.**
///
/// The island grades its carriageways planar (`SurfaceOptions::crown_fall`), so
/// the corridor plateau under them has to hold everything the road draws — the
/// carriageway *and* whichever of a sealed shoulder or a kerb-and-pavement its
/// class carries. A plateau narrower than that puts a kerb in the air at one end
/// of a section and a pavement in a hillside at the other.
#[test]
fn the_corridor_plateau_holds_everything_the_road_draws() {
    // The trunk: 7 m of half-carriageway plus a 2.5 m sealed shoulder.
    let trunk = inf_gis::roads::built_half_width_m(RoadKind::Highway, 4);
    assert_eq!(trunk, 9.5);
    // A connector: 3.5 m of half-carriageway, a 0.3 m kerb stone, 2 m of footway.
    let connector = inf_gis::roads::built_half_width_m(RoadKind::Arterial, 2);
    assert_eq!(connector, 3.5 + inf_gis::KERB_WIDTH_M + inf_gis::PAVEMENT_M);
    assert!(
        trunk > connector,
        "the plateau is sized off the WIDEST route on the island, and that is \
         the trunk"
    );
    // No class carries both a kerb and a shoulder — the two are one decision.
    for kind in [
        RoadKind::Highway,
        RoadKind::Arterial,
        RoadKind::Residential,
        RoadKind::DirtTrack,
        RoadKind::Path,
        RoadKind::Rail,
    ] {
        assert!(
            !(kind.is_kerbed() && kind.shoulder_m() > 0.0),
            "{kind:?} carries a kerb AND a shoulder; they are the two halves of \
             one decision and a road with both is a road drawn twice"
        );
    }
}

// ── wave ROAD1b: the settlement streets ──────────────────────────────────────

/// **The kerb a street draws is where the crowd's pavement ring is.**
///
/// `inf_ecs::society`'s ring is `PAVEMENT_M` outside a block's own rectangle;
/// `inf_gis::build_kerbs` lays `PAVEMENT_M` of concrete behind a kerb at the
/// carriageway's edge. Wave ROAD1b's whole cross-section rule
/// (`inf_ecs::traffic::street_carriageway_half_m`) exists to make those the same
/// concrete on a street recovered from a `gap_m` reserve — and the two crates
/// still cannot name each other, so the equality is asserted here.
///
/// The lane count quantises the carriageway to `LANE_WIDTH_M`, so the footway
/// does not land on the ring exactly; what is asserted is that the ring is ON
/// the concrete, with the residual printed.
///
/// Falsification: change `street_carriageway_half_m`'s `PAVEMENT_M` or
/// `KERB_WIDTH_M` term, or `street_lanes`' rounding, and a reserve drops out.
#[test]
fn a_settlement_streets_footway_is_under_the_ring_the_crowd_walks() {
    use inf_ecs::traffic::{street_carriageway_half_m, street_lanes};
    // The two reserves this engine plans, plus the widest a street may be.
    for gap in [
        inf_editor_core::settlement::TOWN_STREET_M,
        inf_editor_core::settlement::CITY_STREET_M,
        inf_ecs::traffic::MAX_STREET_GAP_M,
    ] {
        // Where the crowd walks: `PAVEMENT_M` in from the block frontage, which
        // is `gap/2` from the centreline.
        let ring = gap * 0.5 - inf_ecs::society::PAVEMENT_M;
        // What the paving draws, through the layer's own lane count — the same
        // number `inf_island::layers::write_streets` writes and
        // `RoadGraph::from_layer` reads back.
        let lanes = street_lanes(gap);
        let half = RoadKind::Arterial.width_m(lanes) * 0.5;
        let kerb_back = half + inf_gis::KERB_WIDTH_M;
        let footway_back = kerb_back + inf_gis::PAVEMENT_M;
        println!(
            "ROAD1b KERB | {gap:.1} m reserve: {lanes} lanes, carriageway half \
             {half:.3} m (wanted {:.3}), kerb {half:.3}..{kerb_back:.3}, footway \
             {kerb_back:.3}..{footway_back:.3}, crowd ring {ring:.3}",
            street_carriageway_half_m(gap)
        );
        assert!(
            ring >= half && ring <= footway_back,
            "on a {gap:.1} m street the crowd walks at {ring:.3} m and the \
             concrete runs {half:.3}..{footway_back:.3} — the ring is off the \
             pavement, which is CERT1's CP-B3 again with the slab present"
        );
    }
}

/// **The parked-car lattice is NOT yet on the kerb the paving draws** — the
/// discrepancy, pinned to the metre it is (wave ROAD1b).
///
/// `KERB_PARK_OFFSET_M` is 5.0 m on every street and was derived against a
/// carriageway no settlement street ever had. Measured against the kerb this
/// wave draws, it parks a car **0.650 m onto the footway** on a 16 m town
/// street and **1.100 m out into the road** on a 20 m city one.
/// `inf_ecs::traffic::kerb_park_offset_m` is the number that puts the flank on
/// the kerb — 4.350 m and 6.100 m — and `kerb_slots` does **not** read it.
///
/// # Why it is carried and not landed, measured
///
/// Wiring it in turns `inf-physics`'s `dispatch_3d` red on three arms, and the
/// middle one is not a re-sample: with the parked row 1.1 m further out on a
/// 20 m street the ambulance in
/// `a_collapse_brings_the_ambulance_and_sends_it_home_again` arrives at step
/// 1 799, resolves at 2 159, and then **never goes home in 30 000 steps** —
/// five hundred seconds of simulation against a test budget of 6 000. A stuck
/// vehicle, not a slow one. It also moves `traffic_3d`'s two sampled arms; that
/// half is noise (swept: the steered hand-off reads 0.144 m at an offset of 5.0,
/// 1.385 at 5.2, 0.000 at 6.0 and 1.041 at 6.8), and the EMS half is not.
///
/// So this arm pins **the gap**, in both directions, so it cannot drift
/// silently and so the wave that fixes the return path finds the number waiting
/// for it. It is a measurement with an assertion round it, which is what a
/// carried item looks like when it is executable.
#[test]
fn the_parked_car_lattice_is_not_yet_on_the_kerb_the_paving_draws() {
    use inf_ecs::traffic::{
        kerb_park_offset_m, street_kerb_offset_m, KERB_PARK_OFFSET_M, PARKED_CAR_HALF_W_M,
    };
    // (reserve, how far the shipped constant misses the kerb by, and which side)
    for (gap, want_miss) in [
        (inf_editor_core::settlement::TOWN_STREET_M, 0.650_f64),
        (inf_editor_core::settlement::CITY_STREET_M, -1.100_f64),
    ] {
        let kerb = street_kerb_offset_m(gap);
        assert_eq!(
            kerb,
            RoadKind::Arterial.width_m(inf_ecs::traffic::street_lanes(gap)) * 0.5,
            "the kerb `inf-ecs` states and the one `inf-gis` draws have parted"
        );
        // Positive = the shipped car's flank is PAST the kerb, on the footway.
        let miss = (KERB_PARK_OFFSET_M + PARKED_CAR_HALF_W_M) - kerb;
        println!(
            "ROAD1b PARK | {gap:.1} m reserve: kerb face {kerb:.3} m; shipped lattice parks at {KERB_PARK_OFFSET_M:.3} (flank {:.3}, {miss:+.3} m from the kerb); the kerb's own answer is {:.3}",
            KERB_PARK_OFFSET_M + PARKED_CAR_HALF_W_M,
            kerb_park_offset_m(gap)
        );
        assert!(
            (miss - want_miss).abs() < 1.0e-9,
            "the shipped lattice misses a {gap:.1} m street's kerb by {miss:+.4} m and this arm records {want_miss:+.4} — one of the two moved, and the carried item's number is now wrong"
        );
        // …and the door that would fix it still answers the kerb exactly.
        assert!(
            (kerb_park_offset_m(gap) + PARKED_CAR_HALF_W_M - kerb).abs() < 1.0e-9,
            "`kerb_park_offset_m` no longer parks a car at the kerb"
        );
    }
}

/// **The streets the island paves are the streets its traffic drives.**
///
/// The committed street layer is derived from the island's authored blocks
/// through `inf_ecs::traffic::streets_of_blocks`, and every span in it is an
/// edge of `carriageway_graph` — the graph `inf_ecs::traffic::carriageway`
/// lays lanes on and `inf_physics::d3::traffic` routes cars along. This is the
/// fence over that: the layer on disk, read back through the GIS reader the
/// island build reads it with, against the derivation, span for span.
///
/// It is the arm the whole wave turns on. The ROAD1 audit's headline was that
/// this island has two road networks and the wave paved the one nobody drives
/// on; if this reds, they have come apart again.
#[test]
fn the_committed_street_layer_is_the_street_grid_the_traffic_sim_derives() {
    for rel in inf_editor_core::island::ISLAND_RECIPES {
        let Some(design) = inf_editor_core::island::committed_design(rel) else {
            eprintln!("SKIP: no committed island design at {rel}");
            continue;
        };
        let path = inf_editor_core::island::repo_root()
            .join(rel)
            .parent()
            .unwrap()
            .join(&design.recipe.roads.streets);
        if !path.exists() {
            eprintln!("SKIP: {} has not been blessed yet", path.display());
            continue;
        }
        let want = inf_editor_core::island::island_street_spans(&design);
        let layer =
            inf_island::layers::read_layer(&path, inf_gis::LayerKind::Roads, &design.anchor)
                .expect("the committed street layer reads");
        assert_eq!(
            layer.features.len(),
            want.len(),
            "{}: the committed layer holds {} spans and the derivation makes {}",
            path.display(),
            layer.features.len(),
            want.len()
        );
        assert!(!want.is_empty(), "{rel} derives no streets at all");
        // And the GRAPH the island build makes out of it is the grid: every
        // crossing a junction, which is what puts a fan and a crossing there.
        let graph = inf_gis::RoadGraph::from_layer(&layer);
        let junctions = graph.junctions().count();
        let km = graph.total_length_m() / 1000.0;
        println!(
            "ROAD1b GRID | {rel}: {} spans, {km:.2} km, {junctions} junctions of degree 3+",
            want.len()
        );
        assert!(
            junctions > 0,
            "{rel}: the street layer makes no junction at all — the spans are \
             not meeting, so nothing fans an intersection or paints a crossing"
        );
        // Every span's lane count is the one its own reserve implies.
        for (f, s) in layer.features.iter().zip(&want) {
            let lanes = f
                .attributes
                .get("lanes")
                .and_then(|a| a.as_number())
                .map(|v| v as u32);
            assert_eq!(
                lanes,
                Some(s.lanes),
                "{}: a span states {lanes:?} lanes and its {} m reserve implies {}",
                path.display(),
                s.gap_m,
                s.lanes
            );
        }
    }
}
