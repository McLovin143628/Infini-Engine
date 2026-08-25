//! Reading and writing the committed vector layers.
//!
//! # One door in, one writer out
//!
//! **Reading** goes through `inf_gis::read_vector` — the one import door, with
//! its axis-order guard, its NaN door and its skipped-feature report. Nothing
//! here parses a coordinate.
//!
//! **Writing** is `serde_json` here, because the facade has no writer: `inf-gis`
//! is an *import* crate and the `geojson` crate it owns is used to read. Adding a
//! writer there to be used by exactly one caller would put a second reason to
//! touch the facade in the tree. What keeps the two honest is
//! `a_written_layer_reads_back_through_the_import_door`, which round-trips every
//! layer this module writes through the door that will read it — a comparison a
//! test makes rather than a sentence a comment claims.
//!
//! # The layers are in the ANCHOR's CRS
//!
//! Easting, northing and elevation in **the recipe's own anchor CRS** — for the
//! two committed islands that is EPSG:32610 — not world metres and not
//! WGS84. Three reasons, in order of weight: a projected layer opens in QGIS
//! beside the survey it came from; the import transform is then an **identity**,
//! so reading a committed layer is an exact subtraction rather than a
//! reprojection round trip; and a file that carried world metres would be
//! meaningless the day the anchor moved, silently.

use std::path::Path;

use glam::DVec2;
use serde_json::{json, Value};

use crate::biome::{BiomeMask, IslandBiome};
use crate::hydro::{Lake, Stream};
use crate::roads::Route;
use crate::IslandError;

/// The GeoJSON `crs` member for **the anchor this layer is written against**.
///
/// RFC 7946 removed the `crs` member and mandates WGS84; a projected GeoJSON is
/// therefore an *extension*, which is exactly why it is stated in the file
/// instead of assumed. `inf_gis::read_vector` takes the CRS as a parameter and
/// never reads this — it is here for the human and for QGIS.
///
/// # It was the literal `urn:ogc:def:crs:EPSG::32610`
///
/// Which is true of the two committed islands and of nothing else. An island
/// anchored anywhere outside UTM zone 10N — the recipe takes any projected,
/// metric CRS — wrote layers that **told QGIS the wrong zone**, and the symptom
/// of that is a coastline five hundred kilometres from the survey it was traced
/// off, with no error anywhere. The one thing this member is for is the human,
/// so a member that is a fact about a different island is worse than none.
///
/// An authority code becomes the URN form QGIS expects; anything else (a proj4
/// string) is written verbatim, because inventing a URN for it would be the same
/// defect one step further on.
fn crs_member(spec: &str) -> String {
    let t = spec.trim();
    if let Some(code) = t.strip_prefix("EPSG:").or_else(|| t.strip_prefix("epsg:")) {
        let code = code.trim_start_matches(':');
        if !code.is_empty() && code.chars().all(|c| c.is_ascii_digit()) {
            return format!("urn:ogc:def:crs:EPSG::{code}");
        }
    }
    t.to_string()
}

fn collection(features: Vec<Value>, crs: &str, note: &str) -> Value {
    json!({
        "type": "FeatureCollection",
        "name": note,
        "crs": { "type": "name", "properties": { "name": crs } },
        "features": features,
    })
}

fn write(path: &Path, v: &Value) -> Result<(), IslandError> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut s = serde_json::to_string_pretty(v)
        .map_err(|e| IslandError::Io(format!("serialising {}: {e}", path.display())))?;
    s.push('\n');
    std::fs::write(path, s)?;
    Ok(())
}

/// Project a world XZ + elevation into the anchor's CRS, for writing.
fn out(anchor: &inf_math::geo::GeoAnchor, x: f64, y: f64, z: f64) -> Value {
    let (e, n, h) = anchor.projected_from_world(glam::DVec3::new(x, y, z));
    json!([e, n, h])
}

/// Write the designed road network.
pub fn write_roads(
    path: &Path,
    anchor: &inf_math::geo::GeoAnchor,
    routes: &[Route],
) -> Result<(), IslandError> {
    let features: Vec<Value> = routes
        .iter()
        .map(|r| {
            let coords: Vec<Value> = r
                .points
                .iter()
                .map(|p| out(anchor, p.x, p.y, p.z))
                .collect();
            json!({
                "type": "Feature",
                "properties": { "name": r.name, "road_type": r.class },
                "geometry": { "type": "LineString", "coordinates": coords },
            })
        })
        .collect();
    write(
        path,
        &collection(
            features,
            &crs_member(&anchor.crs),
            "the island's designed road network (derived once, committed as the design)",
        ),
    )
}

/// Write the derived stream network.
pub fn write_streams(
    path: &Path,
    anchor: &inf_math::geo::GeoAnchor,
    streams: &[Stream],
) -> Result<(), IslandError> {
    let features: Vec<Value> = streams
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let coords: Vec<Value> = s
                .points
                .iter()
                .map(|p| out(anchor, p.x, p.y, p.z))
                .collect();
            json!({
                "type": "Feature",
                "properties": {
                    "name": format!("Reach {i}"),
                    "catchment_m2": s.catchment_m2,
                    "width_m": s.width_m(),
                    "depth_m": s.depth_m(),
                    "flow_m_s": (0.4 + 2.0 * s.grade().clamp(0.0, 0.5)),
                    "fall_m": s.fall_m,
                },
                "geometry": { "type": "LineString", "coordinates": coords },
            })
        })
        .collect();
    write(
        path,
        &collection(
            features,
            &crs_member(&anchor.crs),
            "streams derived from flow accumulation over the carved ground",
        ),
    )
}

/// Write the derived lakes.
pub fn write_lakes(
    path: &Path,
    anchor: &inf_math::geo::GeoAnchor,
    lakes: &[Lake],
) -> Result<(), IslandError> {
    let features: Vec<Value> = lakes
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut ring: Vec<Value> = l
                .outline
                .iter()
                .map(|p| out(anchor, p.x, l.level_m, p.y))
                .collect();
            if let Some(first) = ring.first().cloned() {
                ring.push(first); // GeoJSON rings close explicitly
            }
            json!({
                "type": "Feature",
                "properties": {
                    "name": format!("Lake {i}"),
                    "level_m": l.level_m,
                    "area_m2": l.area_m2,
                    "max_depth_m": l.max_depth_m,
                    "half_x_m": l.half_extent.x,
                    "half_z_m": l.half_extent.y,
                },
                "geometry": { "type": "Polygon", "coordinates": [ring] },
            })
        })
        .collect();
    write(
        path,
        &collection(
            features,
            &crs_member(&anchor.crs),
            "lakes derived from the depression fill",
        ),
    )
}

/// Write the biome design masks.
pub fn write_masks(
    path: &Path,
    anchor: &inf_math::geo::GeoAnchor,
    masks: &[BiomeMask],
) -> Result<(), IslandError> {
    let features: Vec<Value> = masks
        .iter()
        .map(|m| {
            let ring = |r: &Vec<DVec2>| -> Value {
                let mut v: Vec<Value> = r.iter().map(|p| out(anchor, p.x, 0.0, p.y)).collect();
                if let Some(first) = v.first().cloned() {
                    v.push(first);
                }
                Value::Array(v)
            };
            let mut rings = vec![ring(&m.exterior)];
            rings.extend(m.holes.iter().map(ring));
            json!({
                "type": "Feature",
                "properties": { "biome": m.biome.label() },
                "geometry": { "type": "Polygon", "coordinates": rings },
            })
        })
        .collect();
    write(
        path,
        &collection(
            features,
            &crs_member(&anchor.crs),
            "biome design masks — where the author overrides the classifier",
        ),
    )
}

/// Write the designed coastline.
pub fn write_coast(
    path: &Path,
    anchor: &inf_math::geo::GeoAnchor,
    rings: &[Vec<DVec2>],
) -> Result<(), IslandError> {
    let features: Vec<Value> = rings
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut v: Vec<Value> = r.iter().map(|p| out(anchor, p.x, 0.0, p.y)).collect();
            if let Some(first) = v.first().cloned() {
                v.push(first);
            }
            json!({
                "type": "Feature",
                "properties": { "name": format!("Shore {i}") },
                "geometry": { "type": "Polygon", "coordinates": [v] },
            })
        })
        .collect();
    write(
        path,
        &collection(
            features,
            &crs_member(&anchor.crs),
            "the designed coastline — the polygon that makes this an island",
        ),
    )
}

/// Read a committed layer through the one import door.
///
/// The transform is built from the anchor against the anchor's **own** CRS, so
/// it short-circuits to the identity and the read is an exact subtraction. That
/// is not an optimisation — it is what stops a committed layer drifting by a
/// reprojection round trip every time it is read.
pub fn read_layer(
    path: &Path,
    kind: inf_gis::LayerKind,
    anchor: &inf_math::geo::GeoAnchor,
) -> Result<inf_gis::GeoLayer, IslandError> {
    let tf = inf_gis::Transform::new(&anchor.crs, anchor)?;
    Ok(inf_gis::read_vector(path, kind, &anchor.crs, &tf)?)
}

/// The rings of every polygon in a layer, as world XZ.
pub fn rings_of(layer: &inf_gis::GeoLayer) -> Vec<Vec<DVec2>> {
    let mut out = Vec::new();
    for f in &layer.features {
        if let inf_gis::GeoGeometry::Polygon { exterior, .. } = &f.geometry {
            out.push(exterior.iter().map(|p| DVec2::new(p.x, p.z)).collect());
        }
    }
    out
}

/// The biome masks a layer carries.
///
/// A feature whose `biome` attribute names nothing in the palette is **skipped
/// and reported**, never defaulted: a mask that says `forrest` is a typo an
/// author wants to hear about, and a silent fallback would paint the island's
/// biggest polygon the wrong colour.
pub fn masks_of(layer: &inf_gis::GeoLayer) -> (Vec<BiomeMask>, Vec<String>) {
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for (i, f) in layer.features.iter().enumerate() {
        let inf_gis::GeoGeometry::Polygon { exterior, holes } = &f.geometry else {
            skipped.push(format!("mask {i}: not a polygon"));
            continue;
        };
        let Some(name) = f.attr_text(&["biome", "class", "type"]) else {
            skipped.push(format!("mask {i}: no `biome` attribute"));
            continue;
        };
        let Some(b) = IslandBiome::from_label(name) else {
            skipped.push(format!(
                "mask {i}: `{name}` is not one of {}",
                IslandBiome::ALL
                    .iter()
                    .map(|b| b.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            continue;
        };
        out.push(BiomeMask {
            biome: b,
            exterior: exterior.iter().map(|p| DVec2::new(p.x, p.z)).collect(),
            holes: holes
                .iter()
                .map(|h| h.iter().map(|p| DVec2::new(p.x, p.z)).collect())
                .collect(),
        });
    }
    (out, skipped)
}

/// The routes a committed road layer carries, back as [`Route`]s.
pub fn routes_of(layer: &inf_gis::GeoLayer) -> Vec<Route> {
    layer
        .features
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let pts = match &f.geometry {
                inf_gis::GeoGeometry::Polyline { points, .. } => points.clone(),
                _ => return None,
            };
            Some(Route {
                name: f
                    .attr_text(&["name"])
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("Route {i}")),
                class: f
                    .attr_text(&["road_type", "class"])
                    .unwrap_or("residential")
                    .to_string(),
                points: pts,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn anchor() -> inf_math::geo::GeoAnchor {
        inf_gis::anchor_at("EPSG:32610", 492_600.0, 5_465_600.0, 0.0, "EGM2008").unwrap()
    }

    /// **The round trip that keeps the writer and the door honest.** Everything
    /// this module writes is read back through `inf_gis::read_vector` and
    /// compared in world metres.
    ///
    /// Un-fix mutation: write `[n, e, h]` instead of `[e, n, h]` and every
    /// position below lands somewhere else.
    #[test]
    fn a_written_layer_reads_back_through_the_import_door() {
        let dir = tempfile::tempdir().unwrap();
        let a = anchor();

        // Roads.
        let routes = vec![
            Route {
                name: "Harbour - Summit".into(),
                class: "highway".into(),
                points: vec![
                    DVec3::new(-1_200.5, 12.25, 800.75),
                    DVec3::new(0.0, 140.5, 0.0),
                    DVec3::new(1_500.25, 402.0, -900.5),
                ],
            },
            Route {
                name: "Spur".into(),
                class: "arterial".into(),
                points: vec![DVec3::new(0.0, 140.5, 0.0), DVec3::new(500.0, 90.0, 500.0)],
            },
        ];
        let p = dir.path().join("roads.geojson");
        write_roads(&p, &a, &routes).unwrap();
        let layer = read_layer(&p, inf_gis::LayerKind::Roads, &a).unwrap();
        assert!(layer.skipped.is_empty(), "{:?}", layer.skipped);
        assert_eq!(layer.features.len(), 2);
        let back = routes_of(&layer);
        assert_eq!(back.len(), 2);
        for (w, r) in routes.iter().zip(&back) {
            assert_eq!(w.name, r.name);
            assert_eq!(w.class, r.class);
            assert_eq!(w.points.len(), r.points.len());
            for (pw, pr) in w.points.iter().zip(&r.points) {
                // The identity transform makes this EXACT, not approximate: an
                // easting round-tripped through a projection would not be.
                assert_eq!(pw.x, pr.x, "x");
                assert_eq!(pw.z, pr.z, "z");
                assert!((pw.y - pr.y).abs() < 1e-9, "y {} vs {}", pw.y, pr.y);
            }
        }
        // …and the road graph builds out of it with the classes intact.
        let g = inf_gis::RoadGraph::from_layer(&layer);
        assert_eq!(g.segments.len(), 2);
        let kinds: std::collections::BTreeSet<_> =
            g.segments.values().map(|s| s.kind.label()).collect();
        assert_eq!(
            kinds.into_iter().collect::<Vec<_>>(),
            vec!["arterial", "highway"]
        );

        // Coast.
        let rings = vec![vec![
            DVec2::new(-1_000.0, -1_000.0),
            DVec2::new(1_000.0, -1_000.0),
            DVec2::new(1_000.0, 1_000.0),
            DVec2::new(-1_000.0, 1_000.0),
        ]];
        let p = dir.path().join("coast.geojson");
        write_coast(&p, &a, &rings).unwrap();
        let layer = read_layer(&p, inf_gis::LayerKind::Generic, &a).unwrap();
        let got = rings_of(&layer);
        assert_eq!(got, rings, "the shore round-trips exactly");

        // Masks, including a hole and a deliberately bad feature.
        let masks = vec![
            BiomeMask {
                biome: IslandBiome::Farmland,
                exterior: vec![
                    DVec2::new(0.0, 0.0),
                    DVec2::new(400.0, 0.0),
                    DVec2::new(400.0, 300.0),
                    DVec2::new(0.0, 300.0),
                ],
                holes: vec![vec![
                    DVec2::new(100.0, 100.0),
                    DVec2::new(200.0, 100.0),
                    DVec2::new(200.0, 200.0),
                    DVec2::new(100.0, 200.0),
                ]],
            },
            BiomeMask {
                biome: IslandBiome::Meadow,
                exterior: vec![
                    DVec2::new(-500.0, -500.0),
                    DVec2::new(-300.0, -500.0),
                    DVec2::new(-300.0, -300.0),
                ],
                holes: vec![],
            },
        ];
        let p = dir.path().join("biomes.geojson");
        write_masks(&p, &a, &masks).unwrap();
        let layer = read_layer(&p, inf_gis::LayerKind::Biomes, &a).unwrap();
        let (got, skipped) = masks_of(&layer);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].biome, IslandBiome::Farmland);
        assert_eq!(got[0].holes.len(), 1, "the hole survived the round trip");
        assert_eq!(got[0].exterior, masks[0].exterior);
        assert_eq!(got[1].biome, IslandBiome::Meadow);

        // Lakes and streams.
        let lakes = vec![Lake {
            level_m: 121.5,
            centre: DVec2::new(10.0, 20.0),
            half_extent: DVec2::new(60.0, 40.0),
            area_m2: 9_600.0,
            max_depth_m: 7.25,
            outline: vec![
                DVec2::new(-50.0, -20.0),
                DVec2::new(70.0, -20.0),
                DVec2::new(70.0, 60.0),
                DVec2::new(-50.0, 60.0),
            ],
        }];
        let p = dir.path().join("lakes.geojson");
        write_lakes(&p, &a, &lakes).unwrap();
        let layer = read_layer(&p, inf_gis::LayerKind::Lakes, &a).unwrap();
        assert_eq!(layer.features.len(), 1);
        assert_eq!(
            layer.features[0].attr_number(&["level_m"]),
            Some(121.5),
            "a lake's level is what makes it a water body"
        );
        assert_eq!(layer.features[0].attr_number(&["half_x_m"]), Some(60.0));
        assert_eq!(rings_of(&layer)[0], lakes[0].outline);

        let streams = vec![Stream {
            points: vec![
                DVec3::new(0.0, 300.0, 0.0),
                DVec3::new(80.0, 240.0, 40.0),
                DVec3::new(160.0, 100.0, 90.0),
            ],
            catchment_m2: 4.0e6,
            length_m: 200.0,
            fall_m: 200.0,
        }];
        let p = dir.path().join("streams.geojson");
        write_streams(&p, &a, &streams).unwrap();
        let layer = read_layer(&p, inf_gis::LayerKind::Streams, &a).unwrap();
        assert_eq!(layer.features.len(), 1);
        let w = layer.features[0].attr_number(&["width_m"]).unwrap();
        assert!((w - streams[0].width_m()).abs() < 1e-9, "{w}");
        // And the stream attributes reach `inf_gis::stream_attrs`, which is what
        // the import door hands a river body — the seam that would otherwise
        // silently default every reach to 3 m wide.
        let sa = inf_gis::import::stream_attrs(&layer.features[0]);
        assert!((sa.width_m - w).abs() < 1e-9);
        assert!(sa.depth_m > 0.0 && sa.flow_m_s > 0.0);
    }

    /// A mask that misspells its biome is skipped **and named**.
    #[test]
    fn a_mask_that_names_no_biome_is_reported_rather_than_defaulted() {
        let dir = tempfile::tempdir().unwrap();
        let a = anchor();
        let p = dir.path().join("bad.geojson");
        let v = json!({
            "type": "FeatureCollection",
            "features": [
                { "type": "Feature", "properties": { "biome": "forrest" },
                  "geometry": { "type": "Polygon", "coordinates": [[
                      [492600.0, 5465600.0, 0.0], [492700.0, 5465600.0, 0.0],
                      [492700.0, 5465700.0, 0.0], [492600.0, 5465600.0, 0.0]]] } },
                { "type": "Feature", "properties": {},
                  "geometry": { "type": "Polygon", "coordinates": [[
                      [492600.0, 5465600.0, 0.0], [492700.0, 5465600.0, 0.0],
                      [492700.0, 5465700.0, 0.0], [492600.0, 5465600.0, 0.0]]] } },
                { "type": "Feature", "properties": { "biome": "meadow" },
                  "geometry": { "type": "Point", "coordinates": [492600.0, 5465600.0, 0.0] } }
            ]
        });
        write(&p, &v).unwrap();
        let layer = read_layer(&p, inf_gis::LayerKind::Biomes, &a).unwrap();
        let (got, skipped) = masks_of(&layer);
        assert!(got.is_empty(), "nothing here names a real biome");
        assert_eq!(skipped.len(), 3);
        assert!(
            skipped[0].contains("forrest") && skipped[0].contains("forest"),
            "{:?}",
            skipped[0]
        );
        assert!(skipped[1].contains("no `biome`"), "{:?}", skipped[1]);
        assert!(skipped[2].contains("not a polygon"), "{:?}", skipped[2]);
    }

    /// **A LAYER STATES THE CRS IT WAS ACTUALLY WRITTEN IN.**
    ///
    /// The member was the literal `urn:ogc:def:crs:EPSG::32610`, which is true of
    /// the two committed islands and of nothing else — the recipe takes any
    /// projected metric CRS. An island anchored in another zone wrote layers that
    /// told QGIS the wrong one, and the symptom is a coastline hundreds of
    /// kilometres from the survey it was traced off with no error anywhere. The
    /// only thing the member is for is the human, so a member that is a fact
    /// about a different island is worse than none.
    #[test]
    fn a_layer_states_the_crs_it_was_actually_written_in() {
        // The two committed islands are unmoved: same anchor, same member.
        assert_eq!(crs_member("EPSG:32610"), "urn:ogc:def:crs:EPSG::32610");
        // …and another zone is another member.
        assert_eq!(crs_member("EPSG:32633"), "urn:ogc:def:crs:EPSG::32633");
        assert_eq!(crs_member("epsg:2193"), "urn:ogc:def:crs:EPSG::2193");
        assert_eq!(crs_member(" EPSG:3857 "), "urn:ogc:def:crs:EPSG::3857");
        // A proj4 string is written verbatim rather than given an invented URN.
        let proj = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs";
        assert_eq!(crs_member(proj), proj);
        assert_eq!(crs_member("EPSG:not-a-code"), "EPSG:not-a-code");

        // And a real write against a zone-33 anchor says so in the file.
        let dir = tempfile::tempdir().unwrap();
        let a33 = inf_gis::anchor_at("EPSG:32633", 500_000.0, 5_000_000.0, 0.0, "EGM2008")
            .expect("a zone-33 anchor");
        let p = dir.path().join("coast33.geojson");
        write_coast(
            &p,
            &a33,
            &[vec![
                DVec2::new(-100.0, -100.0),
                DVec2::new(100.0, -100.0),
                DVec2::new(100.0, 100.0),
            ]],
        )
        .unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(
            text.contains("urn:ogc:def:crs:EPSG::32633"),
            "the layer claims: {}",
            text.lines().take(8).collect::<Vec<_>>().join(" ")
        );
        assert!(
            !text.contains("32610"),
            "the layer still names zone 10 while being written in zone 33"
        );
        // …and it still reads back through the import door in its own CRS.
        let back = rings_of(&read_layer(&p, inf_gis::LayerKind::Generic, &a33).unwrap());
        assert_eq!(back[0][0], DVec2::new(-100.0, -100.0));
    }
}
