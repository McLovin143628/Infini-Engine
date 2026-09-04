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
