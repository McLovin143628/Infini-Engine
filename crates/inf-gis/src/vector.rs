//! GeoJSON and Shapefile readers → [`GeoLayer`].
//!
//! # Two formats, hand-picked
//!
//! Refusing GDAL means refusing OGR's universal driver set, so the formats this
//! engine reads are chosen rather than inherited. These two are the choice
//! because they are what the government geoportals the owner's document names
//! actually serve, and between them they cover essentially every published
//! vector layer: Shapefile for the bulk downloads, GeoJSON for the API responses
//! and anything hand-edited.
//!
//! # Everything goes through the same door
//!
//! Both readers do exactly the same three things — read a geometry, reproject it
//! through [`crate::crs::Transform`], attach the attribute row — and hand back
//! the same [`GeoLayer`]. Neither one is allowed to have its own opinion about
//! CRSs, axis order or NaN, because the door that has those opinions is
//! `Transform` and there is one of it.
//!
//! # A skipped feature is a reported feature
//!
//! A published layer of 4 000 roads routinely contains a handful of records that
//! cannot be used: a "polyline" with one vertex, a coordinate outside its
//! projection's valid area, a null geometry left by a failed edit. Those are
//! **skipped and named** in [`GeoLayer::skipped`] rather than dropped, because
//! the alternative is an author spending an afternoon looking for a road that
//! the importer silently declined to build.

use std::io::Read;
use std::path::Path;

use glam::DVec3;
// `dbase` (the `.dbf` attribute-table reader) is re-exported by `shapefile`
// itself, so the attribute half of a Shapefile costs no extra workspace pin.
use shapefile::dbase;

use crate::crs::Transform;
use crate::feature::{Attr, GeoFeature, GeoGeometry, GeoLayer, LayerKind};
use crate::GisError;

/// How close a ring's ends must be for the source to count as having closed it.
///
/// Generous on purpose: published coordinates are written at finite precision
/// and an exact-equality test rejects a large fraction of real data over
/// differences far below the resolution of anything the engine builds.
const RING_CLOSE_TOLERANCE_M: f64 = 0.01;

/// The smallest number of positions a usable ring can have, after the wire
/// closing vertex is removed.
const MIN_RING_POSITIONS: usize = 3;

/// Reproject one source position, keeping the source's own elevation where it
/// has one.
fn place(tf: &Transform, x: f64, y: f64, z: Option<f64>) -> Result<DVec3, GisError> {
    tf.to_world(x, y, z.unwrap_or(0.0))
}

/// Drop a ring's repeated closing vertex, if it has one.
fn strip_closing(mut ring: Vec<DVec3>) -> Vec<DVec3> {
    if ring.len() >= 2 && GeoGeometry::ring_is_closed(&ring, RING_CLOSE_TOLERANCE_M) {
        ring.pop();
    }
    ring
}

// ── GeoJSON ─────────────────────────────────────────────────────────────────

/// Read GeoJSON bytes into a layer.
///
/// # The CRS question
///
/// RFC 7946 says GeoJSON **is** WGS 84 longitude/latitude, full stop — the older
/// `crs` member was removed from the specification. In practice files carrying
/// projected coordinates and a stale `crs` member are still published, which is
/// why `source_crs` is a parameter rather than an assumption: the caller states
/// what the file really is, the wizard defaults it to `EPSG:4326`, and a file
/// whose coordinates are plainly not degrees is caught by the geographic range
/// check inside [`Transform::to_world`] rather than being silently believed.
pub fn read_geojson_bytes(
    bytes: &[u8],
    name: &str,
    kind: LayerKind,
    source_crs: &str,
    tf: &Transform,
) -> Result<GeoLayer, GisError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| GisError::Format(format!("GeoJSON must be UTF-8 text: {e}")))?;
    let gj: geojson::GeoJson = text
        .parse()
        .map_err(|e| GisError::Format(format!("this file is not valid GeoJSON: {e}")))?;

    let mut layer = GeoLayer::new(name, kind, source_crs);
    layer.advisories.extend(tf.advisories().iter().cloned());

    let features: Vec<geojson::Feature> = match gj {
        geojson::GeoJson::FeatureCollection(fc) => fc.features,
        geojson::GeoJson::Feature(f) => vec![*Box::new(f)],
        geojson::GeoJson::Geometry(g) => vec![geojson::Feature {
            bbox: None,
            geometry: Some(g),
            id: None,
            properties: None,
            foreign_members: None,
        }],
    };

    for (i, f) in features.into_iter().enumerate() {
        let attributes = f
            .properties
            .map(|p| {
                p.into_iter()
                    .map(|(k, v)| (k, json_attr(&v)))
                    .collect::<std::collections::BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let Some(geom) = f.geometry else {
            layer.skipped.push(format!("feature {i}: no geometry"));
            continue;
        };
        match geojson_geometry(&geom.value, tf) {
            Ok(gs) if gs.is_empty() => {
                layer
                    .skipped
                    .push(format!("feature {i}: geometry produced no usable shape"));
            }
            Ok(gs) => {
                for g in gs {
                    layer.features.push(GeoFeature {
                        geometry: g,
                        attributes: attributes.clone(),
                    });
                }
            }
            Err(e) => layer.skipped.push(format!("feature {i}: {e}")),
        }
    }
    Ok(layer)
}

/// Read a GeoJSON file.
pub fn read_geojson(
    path: &Path,
    kind: LayerKind,
    source_crs: &str,
    tf: &Transform,
) -> Result<GeoLayer, GisError> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "layer".into());
    read_geojson_bytes(&buf, &name, kind, source_crs, tf)
}

fn json_attr(v: &serde_json::Value) -> Attr {
    match v {
        serde_json::Value::String(s) => Attr::Text(s.clone()),
        serde_json::Value::Number(n) => n.as_f64().map_or(Attr::Null, Attr::Number),
        serde_json::Value::Bool(b) => Attr::Bool(*b),
        serde_json::Value::Null => Attr::Null,
        // An array or object attribute has no engine meaning; keeping its text is
        // more useful than dropping the field, and it stays inspectable.
        other => Attr::Text(other.to_string()),
    }
}

/// One GeoJSON geometry → zero or more normalized geometries. A `Multi*` becomes
/// several, each carrying the parent's attributes.
fn geojson_geometry(value: &geojson::Value, tf: &Transform) -> Result<Vec<GeoGeometry>, GisError> {
    use geojson::Value as V;
    let pt = |p: &Vec<f64>| -> Result<DVec3, GisError> {
        if p.len() < 2 {
            return Err(GisError::Geometry(format!(
                "a position needs at least 2 numbers, this has {}",
                p.len()
            )));
        }
        place(tf, p[0], p[1], p.get(2).copied())
    };
    let ring = |r: &Vec<Vec<f64>>| -> Result<Vec<DVec3>, GisError> { r.iter().map(&pt).collect() };
    Ok(match value {
        V::Point(p) => vec![GeoGeometry::Point(pt(p)?)],
        V::MultiPoint(ps) => ps
            .iter()
            .map(&pt)
            .map(|r| r.map(GeoGeometry::Point))
            .collect::<Result<_, _>>()?,
        V::LineString(l) => {
            let points = ring(l)?;
            if points.len() < 2 {
                return Err(GisError::Geometry(
                    "a line needs at least 2 positions".into(),
                ));
            }
            vec![GeoGeometry::Polyline {
                closed: GeoGeometry::ring_is_closed(&points, RING_CLOSE_TOLERANCE_M),
                points,
            }]
        }
        V::MultiLineString(ls) => {
            let mut out = Vec::new();
            for l in ls {
                let points = ring(l)?;
                if points.len() >= 2 {
                    out.push(GeoGeometry::Polyline {
                        closed: GeoGeometry::ring_is_closed(&points, RING_CLOSE_TOLERANCE_M),
                        points,
                    });
                }
            }
            out
        }
        V::Polygon(rings) => vec![geojson_polygon(rings, &ring)?],
        V::MultiPolygon(polys) => {
            let mut out = Vec::new();
            for rings in polys {
                out.push(geojson_polygon(rings, &ring)?);
            }
            out
        }
        V::GeometryCollection(gs) => {
            let mut out = Vec::new();
            for g in gs {
                out.extend(geojson_geometry(&g.value, tf)?);
            }
            out
        }
    })
}

/// A GeoJSON ring as the format stores it: a list of positions, each a list of
/// numbers. Named because clippy is right that the closure type below is
/// unreadable inline, and because saying it once explains the nesting.
type WireRing = Vec<Vec<f64>>;

/// Reprojects one wire ring into world metres — the closure `geojson_geometry`
/// already holds, passed down so a polygon's rings go through the same door as
/// everything else.
type RingReader<'a> = &'a dyn Fn(&WireRing) -> Result<Vec<DVec3>, GisError>;

fn geojson_polygon(rings: &[WireRing], ring: RingReader<'_>) -> Result<GeoGeometry, GisError> {
    let mut it = rings.iter();
    let exterior =
        strip_closing(ring(it.next().ok_or_else(|| {
            GisError::Geometry("a polygon needs an exterior ring".into())
        })?)?);
    if exterior.len() < MIN_RING_POSITIONS {
        return Err(GisError::Geometry(format!(
            "a polygon boundary needs at least {MIN_RING_POSITIONS} distinct \
             positions, this has {}",
            exterior.len()
        )));
    }
    // RFC 7946 says rings after the first are holes — by position, not by
    // winding. (Shapefile is the opposite; see `read_shapefile`.)
    //
    // **The `?` here has to be reachable.** The first cut filtered with
    // `r.as_ref().is_ok_and(..)`, and `is_ok_and` is `false` for an `Err` — so
    // `filter` removed every refusal from the iterator BEFORE `collect` into a
    // `Result` could see it, and the `?` was dead code. A hole carrying a
    // non-finite coordinate, a transposed lat/lon, or a point outside its
    // projection's valid area — every refusal `Transform::to_world` exists to
    // raise — was silently dropped, the polygon came back with one fewer hole,
    // and nothing reached `layer.skipped`. That is the exact opposite of this
    // module's own doctrine that a skipped feature is a reported feature.
    // Refuse first, THEN drop the merely-degenerate.
    let holes = it
        .map(|r| ring(r).map(strip_closing))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|v| v.len() >= MIN_RING_POSITIONS)
        .collect::<Vec<_>>();
    Ok(GeoGeometry::Polygon { exterior, holes })
}

// ── Shapefile ───────────────────────────────────────────────────────────────

/// Read a Shapefile (`.shp` plus its sidecar `.dbf`) into a layer.
///
/// # Rings, and why Shapefile is not GeoJSON
///
/// Shapefile has **no explicit hole marker**. A polygon is a bag of rings and the
/// only thing distinguishing a boundary from a hole is winding: clockwise
/// (in the format's own screen-style coordinate sense) is an outer ring,
/// counter-clockwise is a hole. Reading it as "the first ring is the boundary,
/// the rest are holes" — the GeoJSON rule — silently turns a multi-part polygon
/// (an island group, a parcel in two pieces) into one shape with the other parts
/// punched out of it. The `shapefile` crate already classifies rings for us, and
/// this reader uses that classification rather than position.
///
/// # The `.prj` file
///
/// A Shapefile's CRS lives in a sidecar `.prj` holding WKT — a format this
/// engine has no parser for, and writing one is not a small job. So `source_crs`
/// is a parameter and the wizard shows the author the `.prj` text to choose
/// from. That is a named limitation, not an oversight; it is recorded in the
/// disposition memo.
pub fn read_shapefile(
    path: &Path,
    kind: LayerKind,
    source_crs: &str,
    tf: &Transform,
) -> Result<GeoLayer, GisError> {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "layer".into());
    let mut layer = GeoLayer::new(&name, kind, source_crs);
    layer.advisories.extend(tf.advisories().iter().cloned());

    let mut reader = shapefile::Reader::from_path(path)
        .map_err(|e| GisError::Format(format!("could not open the shapefile {path:?}: {e}")))?;

    for (i, rec) in reader.iter_shapes_and_records().enumerate() {
        let (shape, record) = match rec {
            Ok(r) => r,
            Err(e) => {
                layer.skipped.push(format!("record {i}: {e}"));
                continue;
            }
        };
        let attributes = record
            .into_iter()
            .map(|(k, v)| (k, dbase_attr(&v)))
            .collect::<std::collections::BTreeMap<_, _>>();
        match shape_geometry(&shape, tf) {
            Ok(gs) if gs.is_empty() => {
                layer
                    .skipped
                    .push(format!("record {i}: geometry produced no usable shape"));
            }
            Ok(gs) => {
                for g in gs {
                    layer.features.push(GeoFeature {
                        geometry: g,
                        attributes: attributes.clone(),
                    });
                }
            }
            Err(e) => layer.skipped.push(format!("record {i}: {e}")),
        }
    }
    Ok(layer)
}

/// A numeric attribute, or [`Attr::Null`] when it is not a number.
fn finite_attr(n: f64) -> Attr {
    if n.is_finite() {
        Attr::Number(n)
    } else {
        Attr::Null
    }
}

fn dbase_attr(v: &dbase::FieldValue) -> Attr {
    use dbase::FieldValue as F;
    match v {
        F::Character(Some(s)) => Attr::Text(s.trim().to_string()),
        F::Character(None) => Attr::Null,
        // **Attribute values cross a door too.** A DBF `Numeric` field is parsed
        // out of its ASCII payload with `str::parse::<f64>()`, which happily
        // accepts "NaN", "inf" and "infinity"; a `Double` is a raw little-endian
        // f64 read verbatim, so a NaN bit pattern arrives intact. A stored NaN
        // makes `GeoFeature`'s `PartialEq` non-reflexive and is visible to any
        // consumer that matches `Attr::Number(n)` directly rather than going
        // through `as_number`. It is `Null` — "this record does not state one" —
        // which is exactly what a non-finite measurement means.
        F::Numeric(Some(n)) => finite_attr(*n),
        F::Numeric(None) => Attr::Null,
        F::Float(Some(n)) => finite_attr(f64::from(*n)),
        F::Float(None) => Attr::Null,
        F::Integer(n) => Attr::Number(f64::from(*n)),
        F::Double(n) => finite_attr(*n),
        F::Logical(Some(b)) => Attr::Bool(*b),
        F::Logical(None) => Attr::Null,
        F::Date(Some(d)) => Attr::Text(format!("{:04}-{:02}-{:02}", d.year(), d.month(), d.day())),
        F::Date(None) => Attr::Null,
        other => Attr::Text(format!("{other}")),
    }
}

fn shape_geometry(shape: &shapefile::Shape, tf: &Transform) -> Result<Vec<GeoGeometry>, GisError> {
    use shapefile::Shape as S;
    let p2 = |p: &shapefile::Point| place(tf, p.x, p.y, None);
    let p3 = |p: &shapefile::PointZ| place(tf, p.x, p.y, Some(p.z));

    let parts_to_lines = |parts: &[Vec<DVec3>]| -> Vec<GeoGeometry> {
        parts
            .iter()
            .filter(|p| p.len() >= 2)
            .map(|points| GeoGeometry::Polyline {
                closed: GeoGeometry::ring_is_closed(points, RING_CLOSE_TOLERANCE_M),
                points: points.clone(),
            })
            .collect()
    };

    Ok(match shape {
        S::NullShape => Vec::new(),
        S::Point(p) => vec![GeoGeometry::Point(p2(p)?)],
        S::PointM(p) => vec![GeoGeometry::Point(place(tf, p.x, p.y, None)?)],
        S::PointZ(p) => vec![GeoGeometry::Point(p3(p)?)],
        S::Multipoint(mp) => mp
            .points()
            .iter()
            .map(p2)
            .map(|r| r.map(GeoGeometry::Point))
            .collect::<Result<_, _>>()?,
        S::MultipointM(mp) => mp
            .points()
            .iter()
            .map(|p| place(tf, p.x, p.y, None).map(GeoGeometry::Point))
            .collect::<Result<_, _>>()?,
        S::MultipointZ(mp) => mp
            .points()
            .iter()
            .map(p3)
            .map(|r| r.map(GeoGeometry::Point))
            .collect::<Result<_, _>>()?,
        S::Polyline(pl) => {
            let parts: Vec<Vec<DVec3>> = pl
                .parts()
                .iter()
                .map(|part| part.iter().map(p2).collect::<Result<Vec<_>, _>>())
                .collect::<Result<_, _>>()?;
            parts_to_lines(&parts)
        }
        S::PolylineM(pl) => {
            let parts: Vec<Vec<DVec3>> = pl
                .parts()
                .iter()
                .map(|part| {
                    part.iter()
                        .map(|p| place(tf, p.x, p.y, None))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
            parts_to_lines(&parts)
        }
        S::PolylineZ(pl) => {
            let parts: Vec<Vec<DVec3>> = pl
                .parts()
                .iter()
                .map(|part| part.iter().map(p3).collect::<Result<Vec<_>, _>>())
                .collect::<Result<_, _>>()?;
            parts_to_lines(&parts)
        }
        S::Polygon(pg) => polygon_rings(
            pg.rings()
                .iter()
                .map(|r| {
                    let outer = matches!(r, shapefile::PolygonRing::Outer(_));
                    r.points()
                        .iter()
                        .map(p2)
                        .collect::<Result<Vec<_>, _>>()
                        .map(|pts| (outer, pts))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        S::PolygonM(pg) => polygon_rings(
            pg.rings()
                .iter()
                .map(|r| {
                    let outer = matches!(r, shapefile::PolygonRing::Outer(_));
                    r.points()
                        .iter()
                        .map(|p| place(tf, p.x, p.y, None))
                        .collect::<Result<Vec<_>, _>>()
                        .map(|pts| (outer, pts))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        S::PolygonZ(pg) => polygon_rings(
            pg.rings()
                .iter()
                .map(|r| {
                    let outer = matches!(r, shapefile::PolygonRing::Outer(_));
                    r.points()
                        .iter()
                        .map(p3)
                        .collect::<Result<Vec<_>, _>>()
                        .map(|pts| (outer, pts))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        S::Multipatch(_) => {
            return Err(GisError::Geometry(
                "Multipatch shapes carry 3-D building surfaces this importer does \
                 not read; export the layer as polygons instead"
                    .into(),
            ))
        }
    })
}

/// Assemble Shapefile rings into polygons using the crate's own outer/inner
/// classification — see [`read_shapefile`] for why position is not enough.
///
/// Each outer ring starts a new polygon; inner rings attach to the most recent
/// outer one. That is the format's own model, and it is why a multi-part polygon
/// comes back as several [`GeoGeometry::Polygon`]s rather than one with holes.
fn polygon_rings(rings: Vec<(bool, Vec<DVec3>)>) -> Vec<GeoGeometry> {
    let mut out: Vec<GeoGeometry> = Vec::new();
    for (outer, pts) in rings {
        let pts = strip_closing(pts);
        if pts.len() < MIN_RING_POSITIONS {
            continue;
        }
        if outer || out.is_empty() {
            out.push(GeoGeometry::Polygon {
                exterior: pts,
                holes: Vec::new(),
            });
        } else if let Some(GeoGeometry::Polygon { holes, .. }) = out.last_mut() {
            holes.push(pts);
        }
    }
    out
}

/// Read whichever of the two formats `path`'s extension names.
pub fn read_vector(
    path: &Path,
    kind: LayerKind,
    source_crs: &str,
    tf: &Transform,
) -> Result<GeoLayer, GisError> {
    match path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .as_deref()
    {
        Some("shp") => read_shapefile(path, kind, source_crs, tf),
        Some("geojson") | Some("json") => read_geojson(path, kind, source_crs, tf),
        other => Err(GisError::Format(format!(
            "this engine reads Shapefile (.shp, with its .dbf beside it) and \
             GeoJSON (.geojson, .json); {:?} is neither. Other vector formats need \
             a conversion step on your machine — `ogr2ogr out.geojson in.gpkg` \
             covers most of them.",
            other.unwrap_or("a file with no extension")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crs::anchor_at;

    fn tf() -> Transform {
        let anchor = anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008").unwrap();
        Transform::new("EPSG:4326", &anchor).unwrap()
    }

    const NEAR_LON: f64 = -123.12;
    const NEAR_LAT: f64 = 49.28;

    #[test]
    fn a_geojson_feature_collection_imports_with_its_attributes() {
        let gj = format!(
            r#"{{"type":"FeatureCollection","features":[
              {{"type":"Feature",
                "properties":{{"name":"Main St","lanes":4,"paved":true,"note":null}},
                "geometry":{{"type":"LineString","coordinates":[[{lon},{lat}],[{lon2},{lat}]]}}}},
              {{"type":"Feature","properties":{{}},
                "geometry":{{"type":"Point","coordinates":[{lon},{lat},123.5]}}}}
            ]}}"#,
            lon = NEAR_LON,
            lat = NEAR_LAT,
            lon2 = NEAR_LON + 0.01,
        );
        let t = tf();
        let l =
            read_geojson_bytes(gj.as_bytes(), "Roads", LayerKind::Roads, "EPSG:4326", &t).unwrap();
        assert_eq!(l.features.len(), 2);
        assert!(l.skipped.is_empty(), "{:?}", l.skipped);
        assert_eq!(l.kind, LayerKind::Roads);

        let road = &l.features[0];
        assert_eq!(road.attr_text(&["name"]), Some("Main St"));
        assert_eq!(road.attr_number(&["lanes"]), Some(4.0));
        assert_eq!(road.attr("paved"), Some(&Attr::Bool(true)));
        assert_eq!(
            road.attr("note"),
            Some(&Attr::Null),
            "null stays present-but-unset"
        );
        // The geometry is already world metres, near the origin, and running east.
        match &road.geometry {
            GeoGeometry::Polyline { points, closed } => {
                assert_eq!(points.len(), 2);
                assert!(!closed);
                assert!(points[0].length() < 3000.0, "{:?}", points[0]);
                assert!(points[1].x > points[0].x, "0.01 deg east must increase X");
                assert!(road.geometry.length_m() > 500.0, "~730 m at this latitude");
            }
            other => panic!("expected a polyline, got {other:?}"),
        }
        // The third coordinate is the elevation and reaches +Y.
        match &l.features[1].geometry {
            GeoGeometry::Point(p) => assert!((p.y - 123.5).abs() < 1e-6, "{p:?}"),
            other => panic!("expected a point, got {other:?}"),
        }
        // The axis advisory rode along.
        assert!(l.advisories.iter().any(|a| a.code == "axis.file_order"));
    }

    /// GeoJSON's hole rule is positional, and a `Multi*` fans out while keeping
    /// its parent's attributes.
    #[test]
    fn polygons_holes_and_multiparts_come_through() {
        let sq = |cx: f64, cz: f64, r: f64| {
            format!(
                "[[{},{}],[{},{}],[{},{}],[{},{}],[{},{}]]",
                cx - r,
                cz - r,
                cx + r,
                cz - r,
                cx + r,
                cz + r,
                cx - r,
                cz + r,
                cx - r,
                cz - r
            )
        };
        let gj = format!(
            r#"{{"type":"FeatureCollection","features":[
              {{"type":"Feature","properties":{{"zone":"R1"}},
                "geometry":{{"type":"Polygon","coordinates":[{outer},{hole}]}}}},
              {{"type":"Feature","properties":{{"zone":"R2"}},
                "geometry":{{"type":"MultiLineString","coordinates":[
                   [[{a},{b}],[{c},{b}]], [[{a},{d}],[{c},{d}]]]}}}}
            ]}}"#,
            outer = sq(NEAR_LON, NEAR_LAT, 0.01),
            hole = sq(NEAR_LON, NEAR_LAT, 0.002),
            a = NEAR_LON,
            b = NEAR_LAT,
            c = NEAR_LON + 0.005,
            d = NEAR_LAT + 0.005,
        );
        let t = tf();
        let l = read_geojson_bytes(gj.as_bytes(), "Zoning", LayerKind::Parcels, "EPSG:4326", &t)
            .unwrap();
        // The polygon plus the two lines of the multi-line-string.
        assert_eq!(l.features.len(), 3, "skipped: {:?}", l.skipped);

        match &l.features[0].geometry {
            GeoGeometry::Polygon { exterior, holes } => {
                assert_eq!(exterior.len(), 4, "the wire closing vertex is stripped");
                assert_eq!(holes.len(), 1, "ring 2 is a hole by POSITION in GeoJSON");
                assert_eq!(holes[0].len(), 4);
                // The hole really is inside, so the area is less than the outer.
                let area = l.features[0].geometry.area_m2();
                assert!(area > 0.0);
                let outer_only = GeoGeometry::Polygon {
                    exterior: exterior.clone(),
                    holes: vec![],
                };
                assert!(area < outer_only.area_m2(), "the hole must subtract");
            }
            other => panic!("expected a polygon, got {other:?}"),
        }
        // Both halves of the multi-line carry the parent's attributes.
        for f in &l.features[1..] {
            assert_eq!(f.attr_text(&["zone"]), Some("R2"));
            assert!(matches!(f.geometry, GeoGeometry::Polyline { .. }));
        }
    }

    /// Bad records are skipped and NAMED; the good ones in the same file still
    /// import. This is the "4 000 of 4 100" behaviour.
    #[test]
    fn unusable_records_are_skipped_by_name_without_losing_the_good_ones() {
        let gj = format!(
            r#"{{"type":"FeatureCollection","features":[
              {{"type":"Feature","properties":{{}},"geometry":null}},
              {{"type":"Feature","properties":{{}},
                "geometry":{{"type":"LineString","coordinates":[[{lon},{lat}]]}}}},
              {{"type":"Feature","properties":{{}},
                "geometry":{{"type":"Point","coordinates":[{lat},{lon}]}}}},
              {{"type":"Feature","properties":{{"ok":1}},
                "geometry":{{"type":"LineString","coordinates":[[{lon},{lat}],[{lon2},{lat}]]}}}}
            ]}}"#,
            lon = NEAR_LON,
            lat = NEAR_LAT,
            lon2 = NEAR_LON + 0.01,
        );
        let t = tf();
        let l = read_geojson_bytes(gj.as_bytes(), "Mixed", LayerKind::Generic, "EPSG:4326", &t)
            .unwrap();
        assert_eq!(l.features.len(), 1, "the one good feature must survive");
        assert_eq!(l.features[0].attr_number(&["ok"]), Some(1.0));
        assert_eq!(l.skipped.len(), 3, "{:?}", l.skipped);
        assert!(l.skipped[0].contains("no geometry"), "{:?}", l.skipped);
        assert!(l.skipped[1].contains("at least 2"), "{:?}", l.skipped);
        // The transposed point is caught by the axis check, not imported into the
        // Indian Ocean.
        assert!(
            l.skipped[2].contains("LATITUDE") || l.skipped[2].contains("LONGITUDE"),
            "a transposed coordinate must be named: {:?}",
            l.skipped
        );
        assert!(l.require_non_empty().is_ok());
    }

    #[test]
    fn malformed_input_and_unknown_extensions_are_refused_by_name() {
        let t = tf();
        let e = read_geojson_bytes(b"{ not json", "x", LayerKind::Generic, "EPSG:4326", &t)
            .unwrap_err()
            .to_string();
        assert!(e.contains("not valid GeoJSON"), "{e}");
        // Valid JSON that is not GeoJSON.
        assert!(
            read_geojson_bytes(br#"{"a":1}"#, "x", LayerKind::Generic, "EPSG:4326", &t).is_err()
        );
        // Non-UTF8.
        let e = read_geojson_bytes(
            &[0xff, 0xfe, 0x00],
            "x",
            LayerKind::Generic,
            "EPSG:4326",
            &t,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("UTF-8"), "{e}");

        let e = read_vector(Path::new("x.gpkg"), LayerKind::Generic, "EPSG:4326", &t)
            .unwrap_err()
            .to_string();
        assert!(e.contains("gpkg"), "{e}");
        assert!(
            e.contains("ogr2ogr"),
            "the refusal must name a workaround: {e}"
        );
    }

    /// A bare geometry and a bare feature are both legal GeoJSON documents.
    #[test]
    fn bare_geometries_and_features_are_accepted() {
        let t = tf();
        let g = format!(r#"{{"type":"Point","coordinates":[{NEAR_LON},{NEAR_LAT}]}}"#);
        let l = read_geojson_bytes(g.as_bytes(), "p", LayerKind::Generic, "EPSG:4326", &t).unwrap();
        assert_eq!(l.features.len(), 1);

        let f = format!(
            r#"{{"type":"Feature","properties":{{"n":"x"}},
                 "geometry":{{"type":"Point","coordinates":[{NEAR_LON},{NEAR_LAT}]}}}}"#
        );
        let l = read_geojson_bytes(f.as_bytes(), "p", LayerKind::Generic, "EPSG:4326", &t).unwrap();
        assert_eq!(l.features.len(), 1);
        assert_eq!(l.features[0].attr_text(&["n"]), Some("x"));
    }

    /// A projected source needs no reprojection round trip and lands exactly —
    /// and this is also the path a real government Shapefile takes.
    #[test]
    fn a_projected_source_lands_exactly_where_it_says() {
        let anchor = anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "").unwrap();
        let t = Transform::new("EPSG:32610", &anchor).unwrap();
        let gj = r#"{"type":"Feature","properties":{},
            "geometry":{"type":"LineString","coordinates":[[491000,5459000],[491500,5459000]]}}"#;
        let l = read_geojson_bytes(gj.as_bytes(), "r", LayerKind::Roads, "EPSG:32610", &t).unwrap();
        match &l.features[0].geometry {
            GeoGeometry::Polyline { points, .. } => {
                assert!(points[0].length() < 1e-9, "the anchor lands on the origin");
                assert!((points[1].x - 500.0).abs() < 1e-9);
                assert!((l.features[0].geometry.length_m() - 500.0).abs() < 1e-9);
            }
            other => panic!("{other:?}"),
        }
        // A projected source carries no axis advisory — the hazard is geographic.
        assert!(!l.advisories.iter().any(|a| a.code == "axis.file_order"));
    }

    /// **The Shapefile half, through a real `.shp`/`.shx`/`.dbf` triple.**
    ///
    /// Every other arm here drives GeoJSON bytes, so `read_shapefile`,
    /// `dbase_attr`, `shape_geometry` and `polygon_rings` had no coverage at
    /// all — and `polygon_rings` is the function the reader's longest doc
    /// section exists to justify. Shapefile has **no explicit hole marker**: a
    /// ring is an interior ring because of its winding, and reading the GeoJSON
    /// rule ("the first ring is the boundary, the rest are holes") onto it turns
    /// a multi-part polygon — an island group, a parcel in two pieces — into one
    /// shape with the other parts punched out of it. That is exactly what this
    /// fixture is: two outer rings, one of which has a hole.
    #[test]
    fn a_shapefile_multipart_polygon_keeps_its_parts_and_its_holes() {
        use shapefile::{Point, Polygon, PolygonRing};

        let anchor = anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "").unwrap();
        let t = Transform::new("EPSG:32610", &anchor).unwrap();

        // Rings are given in the format's own winding: an OUTER ring is
        // clockwise, an INNER ring counter-clockwise. `PolygonRing::{Outer,Inner}`
        // is how the crate reports that classification back.
        let sq = |x: f64, z: f64, s: f64| {
            vec![
                Point::new(491_000.0 + x, 5_459_000.0 + z),
                Point::new(491_000.0 + x, 5_459_000.0 + z + s),
                Point::new(491_000.0 + x + s, 5_459_000.0 + z + s),
                Point::new(491_000.0 + x + s, 5_459_000.0 + z),
                Point::new(491_000.0 + x, 5_459_000.0 + z),
            ]
        };
        let mut hole = sq(20.0, 20.0, 20.0);
        hole.reverse();
        let poly = Polygon::with_rings(vec![
            PolygonRing::Outer(sq(0.0, 0.0, 100.0)),
            PolygonRing::Inner(hole),
            // A SECOND part — the case the GeoJSON rule would swallow.
            PolygonRing::Outer(sq(300.0, 0.0, 50.0)),
        ]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Parcels.shp");
        {
            use std::convert::TryInto;
            let builder = dbase::TableWriterBuilder::new()
                .add_character_field("NAME".try_into().unwrap(), 32)
                .add_numeric_field("ACRES".try_into().unwrap(), 12, 3);
            let w = shapefile::Writer::from_path(&path, builder).unwrap();
            let mut rec = dbase::Record::default();
            rec.insert(
                "NAME".into(),
                dbase::FieldValue::Character(Some("Lot 7".into())),
            );
            rec.insert("ACRES".into(), dbase::FieldValue::Numeric(Some(2.5)));
            w.write_shapes_and_records(std::iter::once((&poly, &rec)))
                .unwrap();
        }

        let l = read_shapefile(&path, LayerKind::Parcels, "EPSG:32610", &t)
            .expect("a shapefile triple reads");
        assert!(l.skipped.is_empty(), "{:?}", l.skipped);
        assert_eq!(
            l.features.len(),
            2,
            "a two-part polygon must come back as TWO polygons; one means the \
             second part was read as a hole of the first"
        );

        // The parts, sorted by area so the assertion does not depend on ring
        // order in the file.
        let mut areas: Vec<f64> = l.features.iter().map(|f| f.geometry.area_m2()).collect();
        areas.sort_by(f64::total_cmp);
        assert!(
            (areas[0] - 2_500.0).abs() < 1e-6,
            "the small part is 50x50 = 2500 m2, got {}",
            areas[0]
        );
        assert!(
            (areas[1] - (10_000.0 - 400.0)).abs() < 1e-6,
            "the big part is 100x100 minus its 20x20 hole = 9600 m2, got {} \
             (10000 means the hole was dropped)",
            areas[1]
        );
        // The hole really is a hole, structurally, not just by area.
        let holed = l
            .features
            .iter()
            .find(
                |f| matches!(&f.geometry, GeoGeometry::Polygon { holes, .. } if !holes.is_empty()),
            )
            .expect("one part carries an interior ring");
        match &holed.geometry {
            GeoGeometry::Polygon { exterior, holes } => {
                assert_eq!(holes.len(), 1);
                assert_eq!(exterior.len(), 4, "the closing vertex is stripped");
                assert_eq!(holes[0].len(), 4);
            }
            other => panic!("{other:?}"),
        }

        // The DBF attributes came across, on BOTH parts (they share a record).
        for f in &l.features {
            assert_eq!(f.attr_text(&["name"]), Some("Lot 7"), "separator/case fold");
            assert_eq!(f.attr_number(&["acres"]), Some(2.5));
        }

        // A projected source in the anchor's own CRS lands exactly.
        assert!(
            l.features
                .iter()
                .flat_map(|f| f.geometry.positions())
                .all(|p| p.is_finite()),
            "every imported vertex is finite"
        );

        // …and the dispatcher routes `.shp` here rather than at the GeoJSON door.
        let via = read_vector(&path, LayerKind::Parcels, "EPSG:32610", &t).unwrap();
        assert_eq!(via.features.len(), l.features.len());
    }

    /// A non-finite DBF number is `Null`, not a stored NaN.
    ///
    /// A `Numeric` field is parsed out of ASCII with `str::parse::<f64>()`,
    /// which accepts "NaN" and "inf"; a stored NaN makes `GeoFeature`'s
    /// `PartialEq` non-reflexive and is visible to anything matching
    /// `Attr::Number(n)` without going through `as_number`.
    #[test]
    fn a_non_finite_attribute_is_null_rather_than_a_stored_nan() {
        assert_eq!(
            dbase_attr(&dbase::FieldValue::Numeric(Some(2.5))),
            Attr::Number(2.5)
        );
        assert_eq!(
            dbase_attr(&dbase::FieldValue::Numeric(Some(f64::NAN))),
            Attr::Null
        );
        assert_eq!(
            dbase_attr(&dbase::FieldValue::Double(f64::INFINITY)),
            Attr::Null
        );
        assert_eq!(
            dbase_attr(&dbase::FieldValue::Float(Some(f32::NAN))),
            Attr::Null
        );
        assert_eq!(dbase_attr(&dbase::FieldValue::Numeric(None)), Attr::Null);
    }
}
