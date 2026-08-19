//! The normalized vector feature — one shape every GIS source becomes.
//!
//! # One vocabulary, at one door
//!
//! A Shapefile polyline, a GeoJSON `LineString` and a GeoJSON `MultiLineString`
//! are three encodings of the same fact. Normalising them here means the road
//! generator, the hydrology seeder and the grammar all consume one type, and a
//! new source format is a new reader rather than a new case in every consumer.
//!
//! # Everything here is already world metres
//!
//! A [`GeoFeature`]'s coordinates are **world-space `DVec3` in metres**, past
//! the [`crate::crs::Transform`] door. That is the crate's central rule (see
//! [`crate::crs`]): the source CRS exists only at the import door and is never
//! stored on a vertex. `y` carries the source's elevation where it had one and
//! the anchor's datum height otherwise; the road and hydrology generators
//! re-drape onto the terrain regardless, because a published centreline's
//! elevation is rarely the ground the engine actually built.
//!
//! # Determinism
//!
//! Attributes live in a [`BTreeMap`], not a `HashMap`. Anything reaching a
//! cooked asset must iterate in a deterministic order — the same law that made
//! `TerrainAssetBuilder` use a `BTreeMap` "so insertion order cannot affect the
//! output". A road network keyed by a `HashMap` would make two cooks of the same
//! Shapefile produce different bytes.

use std::collections::BTreeMap;

use glam::{DVec3, Vec3Swizzles};

use crate::{Advisory, GisError};

/// Normalise a field name for matching: lower-case, with `_`, `-` and
/// whitespace removed. See [`GeoFeature::attr`] for why separators must go.
fn fold_field_name(s: &str) -> String {
    s.chars()
        .filter(|c| !(*c == '_' || *c == '-' || c.is_whitespace()))
        .flat_map(char::to_lowercase)
        .collect()
}

/// One attribute value from a source's attribute table.
///
/// Deliberately small: GIS attribute tables carry text, numbers, booleans and
/// nulls, and everything a generator needs to read (a road class, a lane count,
/// a street name, a zoning code) is one of those. `Ord` so a feature set can be
/// sorted into a deterministic order for cooking.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Text(String),
    Number(f64),
    Bool(bool),
    /// The field exists on this layer but is unset on this feature. Kept rather
    /// than dropped, because "no value" and "field absent" are different facts
    /// and a generator's default should be able to tell them apart.
    Null,
}

impl Attr {
    /// This value as a number, if it can be one. Text that parses as a number
    /// counts — government attribute tables store numeric codes as strings often
    /// enough that refusing would be pedantry.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Attr::Number(n) => n.is_finite().then_some(*n),
            Attr::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Attr::Text(s) => s.trim().parse::<f64>().ok().filter(|v| v.is_finite()),
            Attr::Null => None,
        }
    }

    /// This value as text.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Attr::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for Attr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Attr::Text(s) => write!(f, "{s}"),
            Attr::Number(n) => write!(f, "{n}"),
            Attr::Bool(b) => write!(f, "{b}"),
            Attr::Null => write!(f, ""),
        }
    }
}

/// A normalized geometry, in world metres.
#[derive(Clone, Debug, PartialEq)]
pub enum GeoGeometry {
    /// A single position — a well, a sign, a survey mark.
    Point(DVec3),
    /// An ordered chain. `closed` marks a ring (a polygon boundary walked as a
    /// line) rather than an open path.
    Polyline { points: Vec<DVec3>, closed: bool },
    /// An area: one exterior ring plus zero or more interior rings (holes).
    ///
    /// Rings are stored **without** the duplicated closing vertex that
    /// Shapefile and GeoJSON both require on the wire — carrying it would make
    /// every consumer either handle or forget it. [`GeoGeometry::ring_is_closed`]
    /// is how the readers check the source did close it.
    Polygon {
        exterior: Vec<DVec3>,
        holes: Vec<Vec<DVec3>>,
    },
}

impl GeoGeometry {
    /// Every position in this geometry, in order.
    pub fn positions(&self) -> Vec<DVec3> {
        match self {
            GeoGeometry::Point(p) => vec![*p],
            GeoGeometry::Polyline { points, .. } => points.clone(),
            GeoGeometry::Polygon { exterior, holes } => {
                let mut out = exterior.clone();
                for h in holes {
                    out.extend_from_slice(h);
                }
                out
            }
        }
    }

    /// The axis-aligned XZ bounds as `(min, max)`, or `None` when empty.
    pub fn bounds_xz(&self) -> Option<(DVec3, DVec3)> {
        let ps = self.positions();
        let mut it = ps.into_iter();
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }

    /// Total length along the XZ plane, in metres. Zero for a point, the
    /// perimeter for a polygon.
    pub fn length_m(&self) -> f64 {
        let walk = |pts: &[DVec3], closed: bool| -> f64 {
            let mut total = 0.0;
            for w in pts.windows(2) {
                total += (w[1].xz() - w[0].xz()).length();
            }
            if closed && pts.len() > 2 {
                total += (pts[0].xz() - pts[pts.len() - 1].xz()).length();
            }
            total
        };
        match self {
            GeoGeometry::Point(_) => 0.0,
            GeoGeometry::Polyline { points, closed } => walk(points, *closed),
            GeoGeometry::Polygon { exterior, holes } => {
                walk(exterior, true) + holes.iter().map(|h| walk(h, true)).sum::<f64>()
            }
        }
    }

    /// Signed area of a ring on the XZ plane, in m². Positive when the ring
    /// winds counter-clockwise **as seen from +Y looking down**.
    ///
    /// The sign is what tells an exterior ring from a hole in a Shapefile, where
    /// winding is the *only* marker; and it is what a triangulator needs to emit
    /// upward-facing triangles.
    pub fn signed_area_xz(ring: &[DVec3]) -> f64 {
        if ring.len() < 3 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..ring.len() {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            sum += a.x * b.z - b.x * a.z;
        }
        sum * 0.5
    }

    /// Whether a wire ring's first and last positions coincide within
    /// `tolerance_m` — i.e. whether the source closed it as the formats require.
    pub fn ring_is_closed(ring: &[DVec3], tolerance_m: f64) -> bool {
        match (ring.first(), ring.last()) {
            (Some(a), Some(b)) if ring.len() >= 2 => (*a - *b).length() <= tolerance_m,
            _ => false,
        }
    }

    /// Area of a polygon in m² (exterior minus holes), or `0.0` for a non-area
    /// geometry.
    pub fn area_m2(&self) -> f64 {
        match self {
            GeoGeometry::Polygon { exterior, holes } => {
                let a = Self::signed_area_xz(exterior).abs();
                let h: f64 = holes.iter().map(|r| Self::signed_area_xz(r).abs()).sum();
                (a - h).max(0.0)
            }
            _ => 0.0,
        }
    }
}

/// One normalized feature: a geometry plus its attribute row.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoFeature {
    pub geometry: GeoGeometry,
    /// The source's attribute row, keyed by field name. `BTreeMap` for
    /// determinism — see the module docs.
    pub attributes: BTreeMap<String, Attr>,
}

impl GeoFeature {
    pub fn new(geometry: GeoGeometry) -> Self {
        Self {
            geometry,
            attributes: BTreeMap::new(),
        }
    }

    /// Look an attribute up ignoring case **and separators**.
    ///
    /// Government attribute tables are wildly inconsistent: `ROAD_TYPE`,
    /// `road_type`, `RoadType` and `road type` all occur, sometimes in sibling
    /// files from the same agency, and a Shapefile's `.dbf` upper-cases field
    /// names outright while truncating them to ten characters. Matching on case
    /// alone is not enough — `ROAD_TYPE` and `RoadType` differ by an underscore,
    /// not by case, so a case-only match misses exactly the pair an author is
    /// most likely to hit. Folding out `_`, `-` and spaces as well costs nothing
    /// and makes every generator's field mapping work on real files.
    ///
    /// Exact match wins first, so a layer that genuinely has both `road_type`
    /// and `roadtype` is not scrambled.
    pub fn attr(&self, name: &str) -> Option<&Attr> {
        if let Some(v) = self.attributes.get(name) {
            return Some(v);
        }
        let want = fold_field_name(name);
        self.attributes
            .iter()
            .find(|(k, _)| fold_field_name(k) == want)
            .map(|(_, v)| v)
    }

    /// The first of `names` that is present and numeric.
    ///
    /// Sources disagree about what a field is called; a generator states the
    /// spellings it knows and takes the first hit.
    pub fn attr_number(&self, names: &[&str]) -> Option<f64> {
        names.iter().find_map(|n| self.attr(n)?.as_number())
    }

    /// The first of `names` that is present and non-empty text.
    pub fn attr_text(&self, names: &[&str]) -> Option<&str> {
        names
            .iter()
            .find_map(|n| self.attr(n)?.as_text().filter(|s| !s.trim().is_empty()))
    }
}

/// What a layer is for — the engine-side meaning an author assigns at import.
///
/// This is the bridge between "a file of lines" and "the road network": the same
/// polylines mean different things depending on which layer they were imported
/// as, and that decision belongs to the author, not to a heuristic over field
/// names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum LayerKind {
    /// Uninterpreted. Still usable as spans by the grammar.
    #[default]
    Generic,
    /// Road/rail centrelines → the road graph and its ribbons.
    Roads,
    /// Watercourse centrelines → hydrology seeding, with flow direction taken
    /// from the polyline's own vertex order.
    Streams,
    /// Water-body boundaries → lake surfaces.
    Lakes,
    /// Land cover / zoning areas → biome ids.
    Biomes,
    /// Building footprints → the lot substrate.
    Buildings,
    /// Cadastral / parcel boundaries.
    Parcels,
}

impl LayerKind {
    pub const fn label(self) -> &'static str {
        match self {
            LayerKind::Generic => "generic",
            LayerKind::Roads => "roads",
            LayerKind::Streams => "streams",
            LayerKind::Lakes => "lakes",
            LayerKind::Biomes => "biomes",
            LayerKind::Buildings => "buildings",
            LayerKind::Parcels => "parcels",
        }
    }

    /// Parse a label back. Unknown labels become [`LayerKind::Generic`] rather
    /// than failing — a layer whose purpose we do not recognise is still a layer.
    pub fn from_label(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "roads" => LayerKind::Roads,
            "streams" => LayerKind::Streams,
            "lakes" => LayerKind::Lakes,
            "biomes" => LayerKind::Biomes,
            "buildings" => LayerKind::Buildings,
            "parcels" => LayerKind::Parcels,
            _ => LayerKind::Generic,
        }
    }
}

/// A named collection of features — the unit an author imports and binds.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoLayer {
    pub name: String,
    pub kind: LayerKind,
    pub features: Vec<GeoFeature>,
    /// The CRS the source was in, kept for the record. The geometry is already
    /// world metres; this is provenance, not a live coordinate frame.
    pub source_crs: String,
    /// Non-fatal hazards from the import.
    pub advisories: Vec<Advisory>,
    /// Features the reader could not use, with the reason. **Skipped, counted
    /// and named** rather than silently dropped: a road layer that imports 4 000
    /// of its 4 100 segments has a story, and an author who is never told it
    /// will spend the afternoon looking for the missing road.
    pub skipped: Vec<String>,
}

impl GeoLayer {
    pub fn new(name: impl Into<String>, kind: LayerKind, source_crs: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind,
            features: Vec::new(),
            source_crs: source_crs.into(),
            advisories: Vec::new(),
            skipped: Vec::new(),
        }
    }

    /// Combined XZ bounds of every feature, or `None` when the layer is empty.
    pub fn bounds_xz(&self) -> Option<(DVec3, DVec3)> {
        self.features
            .iter()
            .filter_map(|f| f.geometry.bounds_xz())
            .reduce(|(alo, ahi), (blo, bhi)| (alo.min(blo), ahi.max(bhi)))
    }

    /// Every polyline in the layer, as point runs — the shape the grammar's
    /// span source and the road builder both want.
    pub fn polylines(&self) -> Vec<(Vec<DVec3>, bool)> {
        self.features
            .iter()
            .filter_map(|f| match &f.geometry {
                GeoGeometry::Polyline { points, closed } => Some((points.clone(), *closed)),
                GeoGeometry::Polygon { exterior, .. } => Some((exterior.clone(), true)),
                GeoGeometry::Point(_) => None,
            })
            .collect()
    }

    /// Refuse a layer that imported nothing usable, naming the first few reasons.
    ///
    /// An empty layer is almost always a mistake — the wrong file, the wrong
    /// CRS, a filter that matched nothing — and returning it as a success means
    /// the author discovers it three steps later.
    pub fn require_non_empty(&self) -> Result<(), GisError> {
        if !self.features.is_empty() {
            return Ok(());
        }
        let why = if self.skipped.is_empty() {
            "the file contained no features at all".to_string()
        } else {
            format!(
                "all {} feature(s) were skipped; the first reasons were: {}",
                self.skipped.len(),
                self.skipped
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        };
        Err(GisError::Geometry(format!(
            "the layer {:?} imported no usable features — {why}",
            self.name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(pts: &[(f64, f64)]) -> Vec<DVec3> {
        pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect()
    }

    /// Winding sign is the only thing that distinguishes a hole from a boundary
    /// in a Shapefile, so it has to be right, and it has to be right about which
    /// way it looks at the world.
    #[test]
    fn signed_area_sign_follows_the_winding_seen_from_above() {
        // A unit square wound counter-clockwise looking down from +Y:
        // (0,0) -> (1,0) -> (1,1) -> (0,1) in XZ.
        let ccw = ring(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let a = GeoGeometry::signed_area_xz(&ccw);
        assert!((a.abs() - 1.0).abs() < 1e-12, "unit square area, got {a}");
        // Reversing the ring flips the sign and nothing else.
        let mut cw = ccw.clone();
        cw.reverse();
        assert!((GeoGeometry::signed_area_xz(&cw) + a).abs() < 1e-12);
        // Degenerate rings have no area rather than a wrong one.
        assert_eq!(
            GeoGeometry::signed_area_xz(&ring(&[(0.0, 0.0), (1.0, 1.0)])),
            0.0
        );
        assert_eq!(GeoGeometry::signed_area_xz(&[]), 0.0);
    }

    #[test]
    fn polygon_area_subtracts_its_holes() {
        let outer = ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        let hole = ring(&[(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)]);
        let p = GeoGeometry::Polygon {
            exterior: outer,
            holes: vec![hole],
        };
        assert!(
            (p.area_m2() - 96.0).abs() < 1e-9,
            "100 - 4, got {}",
            p.area_m2()
        );
        // The perimeter walks the hole too.
        assert!((p.length_m() - (40.0 + 8.0)).abs() < 1e-9);
        // A polyline has length but no area.
        let line = GeoGeometry::Polyline {
            points: ring(&[(0.0, 0.0), (3.0, 4.0)]),
            closed: false,
        };
        assert_eq!(line.area_m2(), 0.0);
        assert!((line.length_m() - 5.0).abs() < 1e-12);
        // Closing it adds the return leg.
        let closed = GeoGeometry::Polyline {
            points: ring(&[(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]),
            closed: true,
        };
        assert!((closed.length_m() - 12.0).abs() < 1e-12);
    }

    /// Attribute lookup is case-insensitive and coerces, because real attribute
    /// tables are not tidy.
    #[test]
    fn attribute_lookup_survives_real_world_field_naming() {
        let mut f = GeoFeature::new(GeoGeometry::Point(DVec3::ZERO));
        f.attributes
            .insert("ROAD_TYPE".into(), Attr::Text("Residential".into()));
        f.attributes
            .insert("Lanes".into(), Attr::Text(" 4 ".into()));
        f.attributes.insert("SPEED".into(), Attr::Number(50.0));
        f.attributes.insert("oneway".into(), Attr::Bool(true));
        f.attributes.insert("name".into(), Attr::Null);

        assert_eq!(
            f.attr("road_type").and_then(Attr::as_text),
            Some("Residential")
        );
        assert_eq!(
            f.attr("RoadType").and_then(Attr::as_text),
            Some("Residential")
        );
        // Numeric text coerces — agencies store counts as strings routinely.
        assert_eq!(f.attr("lanes").and_then(|a| a.as_number()), Some(4.0));
        assert_eq!(f.attr_number(&["LANE_COUNT", "lanes"]), Some(4.0));
        assert_eq!(f.attr_number(&["nope"]), None);
        assert_eq!(f.attr("oneway").and_then(|a| a.as_number()), Some(1.0));
        // Null is present-but-unset: found, but with no value.
        assert!(f.attr("name").is_some());
        assert_eq!(f.attr("name").and_then(Attr::as_text), None);
        assert_eq!(f.attr_text(&["name", "ROAD_TYPE"]), Some("Residential"));
        // Non-finite numbers are not numbers.
        assert_eq!(Attr::Number(f64::NAN).as_number(), None);
        assert_eq!(Attr::Text("abc".into()).as_number(), None);
    }

    #[test]
    fn ring_closure_detection_has_a_tolerance() {
        let open = ring(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
        assert!(!GeoGeometry::ring_is_closed(&open, 1e-6));
        let mut closed = open.clone();
        closed.push(closed[0]);
        assert!(GeoGeometry::ring_is_closed(&closed, 1e-6));
        // Nearly closed, within tolerance — real data is written with finite
        // precision and a hard equality would reject most of it.
        let mut nearly = open.clone();
        nearly.push(DVec3::new(1e-9, 0.0, 1e-9));
        assert!(GeoGeometry::ring_is_closed(&nearly, 1e-6));
        assert!(!GeoGeometry::ring_is_closed(&nearly, 1e-12));
    }

    /// An empty layer is a refusal with the skip reasons attached — the "4 000 of
    /// 4 100 imported" problem, in its total form.
    #[test]
    fn an_empty_layer_is_refused_with_the_reasons_it_is_empty() {
        let mut l = GeoLayer::new("Roads", LayerKind::Roads, "EPSG:4326");
        assert!(l.bounds_xz().is_none());
        let e = l.require_non_empty().unwrap_err().to_string();
        assert!(e.contains("no features at all"), "{e}");

        l.skipped.push("record 3: only 1 vertex".into());
        l.skipped.push("record 7: non-finite coordinate".into());
        let e = l.require_non_empty().unwrap_err().to_string();
        assert!(e.contains("record 3") && e.contains("record 7"), "{e}");
        assert!(e.contains("Roads"), "the layer must be named: {e}");

        l.features
            .push(GeoFeature::new(GeoGeometry::Point(DVec3::new(
                1.0, 2.0, 3.0,
            ))));
        assert!(l.require_non_empty().is_ok());
        assert_eq!(
            l.bounds_xz(),
            Some((DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 2.0, 3.0)))
        );
    }

    #[test]
    fn polylines_include_polygon_boundaries_as_closed_runs() {
        let mut l = GeoLayer::new("Mixed", LayerKind::Generic, "EPSG:32610");
        l.features.push(GeoFeature::new(GeoGeometry::Polyline {
            points: ring(&[(0.0, 0.0), (5.0, 0.0)]),
            closed: false,
        }));
        l.features.push(GeoFeature::new(GeoGeometry::Polygon {
            exterior: ring(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            holes: vec![],
        }));
        l.features
            .push(GeoFeature::new(GeoGeometry::Point(DVec3::ZERO)));
        let p = l.polylines();
        assert_eq!(p.len(), 2, "the point contributes no polyline");
        assert!(!p[0].1, "the open line stays open");
        assert!(p[1].1, "a polygon boundary is a closed run");
    }
}
