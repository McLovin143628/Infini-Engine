//! Source CRS → the anchor's CRS → world metres, with the hazards made loud.
//!
//! # The one canonical form
//!
//! GeoCanvas's ninth invariant is that all *persisted* geometry lives in one
//! canonical CRS and conversion happens only at the boundary; its own review
//! checklist calls a fake coordinate written into the project file "the failure
//! this feature could most easily cause". The analogue here, and the rule this
//! module enforces:
//!
//! > **All persisted geometry is world metres in the project's anchor CRS.** A
//! > source CRS exists only at the import door and is never stored on a vertex.
//!
//! That is why [`Transform`] hands back a [`DVec3`] and not a coordinate pair
//! with a CRS attached: once a feature is through this door there is exactly one
//! frame it can be in, so nothing downstream can be wrong about which.
//!
//! # Axis order: the silent killer, decided once
//!
//! EPSG:4326's *authority* definition is **latitude, longitude**. Essentially
//! every file on Earth stores **longitude, latitude**: the GeoJSON specification
//! mandates it (RFC 7946 §3.1.1, "the order … SHALL be easting, northing"), a
//! Shapefile's records are literally `(x, y)`, and a GeoTIFF tiepoint is
//! `(x, y)`. GDAL 3 switched to honouring the authority order and thereby made
//! this a live bug in a great deal of previously-working code — GeoCanvas's
//! notes record that getting it wrong "produces silently transposed coordinates
//! with no error".
//!
//! **This engine takes file order, always:** the first component is X
//! (easting/longitude), the second is Y (northing/latitude). One convention, at
//! one door, pinned by [`the_axis_convention_is_file_order_not_authority_order`].
//! A transposed import is caught by the range check inside
//! [`Transform::to_world`] for geographic sources — a "latitude" of 123 is
//! impossible — which turns the silent failure into a named one for the single
//! most common case.
//!
//! # `proj4rs` returns NaN rather than failing
//!
//! Its documented default is a *relaxed* mode that "returns NaN in case of
//! projection failure", because it targets browser mapping apps where a failed
//! reprojection should not abort a pan. For an import door that is exactly
//! backwards: a NaN easting becomes a NaN world coordinate becomes a NaN vertex,
//! and because `f32::min`/`f32::max` ignore NaN the bounds of whatever is built
//! from it look perfectly healthy. Every transform below is therefore
//! finiteness-checked at the point of production and refused by value with the
//! offending input named.
//!
//! # Units
//!
//! Inputs are in the **source CRS's own units** (degrees for a geographic CRS,
//! metres for a projected one). Outputs are world **metres**. Nothing in between
//! is a scale factor in the sense rule 6 forbids: a degree→metre change is a
//! projection, not a rescaling of the world.

use glam::DVec3;
use inf_math::geo::GeoAnchor;
use proj4rs::Proj;

use crate::{epsg, Advisory, GisError};

/// The pole-ward latitude limit a Mercator-family projection can express.
///
/// Web Mercator is undefined at the poles (its northing goes to infinity), and
/// the conventional cut is ±85.051 129°, the latitude that makes the projected
/// world exactly square. Used to keep a degenerate input from becoming an
/// infinity that the NaN door then has to catch.
pub const MAX_LATITUDE_DEG: f64 = 85.051_128_779_806_59;

/// A configured source-CRS → world-metres transform.
///
/// Built once per import (it compiles two projections), then called per
/// coordinate. Cheap to call, so a million-vertex layer costs one setup.
pub struct Transform {
    source: Proj,
    target: Proj,
    anchor: GeoAnchor,
    source_is_latlong: bool,
    /// The source and the anchor are the same CRS, so no reprojection is needed.
    identity: bool,
    source_spec: String,
    advisories: Vec<Advisory>,
}

impl std::fmt::Debug for Transform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transform")
            .field("source", &self.source_spec)
            .field("target", &self.anchor.crs)
            .field("source_is_latlong", &self.source_is_latlong)
            .field("advisories", &self.advisories.len())
            .finish()
    }
}

fn build_proj(spec: &str, role: &str) -> Result<Proj, GisError> {
    let proj4 = epsg::proj4_for(spec)?;
    Proj::from_proj_string(&proj4).map_err(|e| {
        GisError::Crs(format!(
            "the {role} CRS {spec:?} resolved to {proj4:?}, which this engine's \
             projection library could not build: {e}"
        ))
    })
}

impl Transform {
    /// Build a transform from `source_crs` into `anchor`'s CRS and world metres.
    ///
    /// # Refusals, and why each one is a refusal rather than an advisory
    ///
    /// * **A disabled or malformed anchor.** There is nothing to transform into.
    /// * **A geographic anchor CRS.** A degree is not a metre, and rule 6 allows
    ///   no scale factor to make it one, so a world can never be anchored in one.
    /// * **EPSG:3857 (Web Mercator) as the anchor CRS.** Its "metres" are
    ///   inflated by `1/cos(latitude)` — about 1.53× at Vancouver's 49° N. A
    ///   world anchored in it would be built 53% too large, uniformly, with no
    ///   symptom other than everything being wrong. It stays perfectly legal as a
    ///   *source*.
    /// * **A non-metric anchor CRS** (a state plane in US survey feet, say).
    ///   Same reason, wearing a different hat.
    pub fn new(source_crs: &str, anchor: &GeoAnchor) -> Result<Self, GisError> {
        anchor.validate()?;
        let mut advisories = Vec::new();

        let target = build_proj(&anchor.crs, "anchor")?;
        if target.is_latlong() {
            return Err(GisError::Crs(format!(
                "the level's geo-anchor names {:?}, which is a GEOGRAPHIC coordinate \
                 system measured in degrees. A degree is not a metre — and it is not \
                 even a constant number of metres, since a degree of longitude \
                 shrinks to nothing at the poles — so it can never be this engine's \
                 world CRS (1 world unit = 1 metre, no scale factors). Anchor the \
                 level in a projected metric CRS instead; for this area that is \
                 probably EPSG:{}.",
                anchor.crs,
                epsg::suggested_utm_epsg(anchor.origin_latitude_deg, anchor.origin_longitude_deg)
            )));
        }
        if epsg::parse_code(&anchor.crs) == Some(3857) {
            return Err(GisError::Crs(format!(
                "the level's geo-anchor names EPSG:3857 (Web Mercator). Its \
                 \"metres\" are not ground metres: they are inflated by \
                 1/cos(latitude), which is about {:.2}x at this anchor's latitude \
                 of {:.2}. A world anchored in it would be built that much too \
                 large, uniformly, with nothing to show for it but everything \
                 being the wrong size. Use EPSG:{} instead. (Web Mercator remains \
                 fine as a SOURCE crs — it is only anchoring a world in it that is \
                 refused.)",
                1.0 / anchor
                    .origin_latitude_deg
                    .to_radians()
                    .cos()
                    .abs()
                    .max(1e-9),
                anchor.origin_latitude_deg,
                epsg::suggested_utm_epsg(anchor.origin_latitude_deg, anchor.origin_longitude_deg)
            )));
        }
        if (target.to_meter() - 1.0).abs() > 1e-9 {
            return Err(GisError::Crs(format!(
                "the level's geo-anchor names {:?}, whose linear unit is {} \
                 ({} metres per unit), not metres. This engine's world is metric by \
                 rule (1 world unit = 1 metre, no unit-scale factors). Re-anchor in \
                 the metric form of this projection.",
                anchor.crs,
                target.units(),
                target.to_meter()
            )));
        }

        let source_proj4 = epsg::proj4_for(source_crs)?;
        let source = build_proj(source_crs, "source")?;
        let source_is_latlong = source.is_latlong();
        // **Identity short-circuit.** When the source and the anchor resolve to
        // the SAME proj4 string, transforming is a round trip through geodetic
        // coordinates that can only lose precision — measured at ~1e-7 m on a
        // UTM easting, which is harmless but is not zero, and "zero" is what a
        // same-CRS import should cost. Skipping it makes the common case (a
        // government layer already in the project's own CRS) both exact and free.
        let identity = source_proj4.trim() == epsg::proj4_for(&anchor.crs)?.trim();

        if let Some(a) = datum_advisory_for(source_crs, &anchor.crs) {
            advisories.push(a);
        }
        if source_is_latlong {
            advisories.push(Advisory::new(
                "axis.file_order",
                "the source is a geographic CRS; its coordinates are read as \
                 (longitude, latitude) — file order, which is what GeoJSON, \
                 Shapefile and GeoTIFF all store — and NOT the (latitude, \
                 longitude) order the EPSG authority defines for EPSG:4326. If an \
                 import lands on the wrong continent, this is why."
                    .to_string(),
            ));
        }

        Ok(Self {
            source,
            target,
            anchor: anchor.clone(),
            source_is_latlong,
            identity,
            source_spec: source_crs.to_string(),
            advisories,
        })
    }

    /// The non-fatal hazards found while configuring this transform.
    pub fn advisories(&self) -> &[Advisory] {
        &self.advisories
    }

    /// The datum-error advisory, if this pair of CRSs has one. See
    /// [`datum_advisory_for`].
    pub fn datum_advisory(&self) -> Option<&Advisory> {
        self.advisories
            .iter()
            .find(|a| a.code.starts_with("datum."))
    }

    /// Whether the source is a geographic (degree-valued) CRS.
    pub fn source_is_geographic(&self) -> bool {
        self.source_is_latlong
    }

    /// Source coordinates → the anchor's CRS, as `(easting, northing, height)`.
    ///
    /// `x`/`y` are in **file order** (see the module docs): X first — longitude
    /// or easting — then Y. `z` is metres either way.
    pub fn to_projected(&self, x: f64, y: f64, z: f64) -> Result<(f64, f64, f64), GisError> {
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(GisError::NotFinite(format!(
                "the source coordinate ({x}, {y}, {z}) is not finite — a NaN or \
                 infinity in the input file, which usually means an unhandled \
                 no-data value"
            )));
        }
        // A geographic source is checked for a transposed axis order BEFORE it is
        // projected, because after projection a swapped pair is just a wrong
        // number rather than an impossible one.
        if self.source_is_latlong {
            if !(-180.0..=180.0).contains(&x) {
                return Err(GisError::NotFinite(format!(
                    "the source is a geographic CRS, so its first coordinate is a \
                     LONGITUDE — but this record's is {x}, which is outside \
                     [-180, 180]. The usual cause is a file stored in (latitude, \
                     longitude) order; this engine reads (longitude, latitude), \
                     which is what GeoJSON and Shapefile specify."
                )));
            }
            if !(-90.0..=90.0).contains(&y) {
                return Err(GisError::NotFinite(format!(
                    "the source is a geographic CRS, so its second coordinate is a \
                     LATITUDE — but this record's is {y}, which is outside \
                     [-90, 90]."
                )));
            }
        }

        if self.identity {
            return Ok((x, y, z));
        }
        // proj4rs speaks RADIANS for angular units, not degrees. Converting on
        // the way in and out is the whole of the unit contract, and forgetting it
        // is a factor-of-57 error that still produces plausible-looking numbers.
        let mut p = if self.source_is_latlong {
            (x.to_radians(), y.to_radians(), z)
        } else {
            (x, y, z)
        };
        proj4rs::transform::transform(&self.source, &self.target, &mut p).map_err(|e| {
            GisError::Crs(format!(
                "reprojecting ({x}, {y}, {z}) from {:?} failed: {e}",
                self.source_spec
            ))
        })?;
        // **The NaN door.** proj4rs's relaxed default hands back NaN rather than
        // an error when a projection fails (see the module docs), so the success
        // path above proves nothing on its own.
        if !(p.0.is_finite() && p.1.is_finite() && p.2.is_finite()) {
            return Err(GisError::NotFinite(format!(
                "reprojecting ({x}, {y}, {z}) from {:?} produced the non-finite \
                 result ({}, {}, {}). The projection library reports failure this \
                 way rather than as an error; the usual cause is a coordinate \
                 outside the projection's valid area — for a UTM zone, a point far \
                 outside the zone, or a latitude at the pole.",
                self.source_spec, p.0, p.1, p.2
            )));
        }
        if self.target.is_latlong() {
            // Unreachable today (the constructor refuses a geographic anchor) but
            // written so a future relaxation cannot silently emit radians.
            return Ok((p.0.to_degrees(), p.1.to_degrees(), p.2));
        }
        Ok(p)
    }

    /// Source coordinates → **world metres**, anchor subtracted.
    ///
    /// The complete door: reproject, guard, subtract the anchor. `z` is the
    /// source's own elevation in metres; pass `0.0` for a 2-D feature and the
    /// result sits at the anchor's height datum.
    pub fn to_world(&self, x: f64, y: f64, z: f64) -> Result<DVec3, GisError> {
        let (e, n, h) = self.to_projected(x, y, z)?;
        let w = self.anchor.world_from_projected(e, n, h);
        // Belt and braces: the anchor was validated at construction and the
        // projection output was just guarded, so this can only fire on an
        // overflow to infinity in the subtraction itself — which is exactly the
        // case a bare "we checked the inputs" argument would miss.
        if !w.is_finite() {
            return Err(GisError::NotFinite(format!(
                "({x}, {y}, {z}) landed at the non-finite world position {w:?}"
            )));
        }
        Ok(w)
    }

    /// The anchor this transform targets.
    pub fn anchor(&self) -> &GeoAnchor {
        &self.anchor
    }
}

/// The datum-error advisory for a source/target pair, if there is one to give.
///
/// # The honest version of "we do not have PROJ"
///
/// A datum shift between two geodetic reference frames is, in general, a
/// *warp* — the local distortion of a survey network that no similarity
/// transformation can express. PROJ corrects it with NTv2 grid files;
/// `proj4rs` has none, so it applies at best a 7-parameter Helmert transform and
/// at worst nothing.
///
/// GeoCanvas's precedent is directly applicable and is followed here: it
/// transforms a grid of sample points through two independent implementations
/// and *warns* past about a metre of divergence, deliberately **not** refusing,
/// because "a 30 m datum offset is still a usable working canvas". For a game
/// world that reasoning is even stronger — a uniform thirty-metre offset of the
/// entire island is invisible to a player. What is not acceptable is not being
/// told.
pub fn datum_advisory_for(source_crs: &str, target_crs: &str) -> Option<Advisory> {
    let datum_of = |s: &str| -> Option<&'static str> {
        let code = epsg::parse_code(s);
        let p4 = epsg::proj4_for(s).ok()?;
        // NAD27 is the one in wide circulation whose grid-free error is metres.
        // Detected by CODE first and by the Clarke-1866 ellipsoid second, because
        // the table cannot spell it `+datum=NAD27` — see the note on EPSG:4267.
        if code == Some(4267) || p4.contains("+datum=NAD27") || p4.contains("+ellps=clrk66") {
            return Some("NAD27");
        }
        if p4.contains("+datum=NAD83") {
            return Some("NAD83");
        }
        if p4.contains("+datum=WGS84") || p4.contains("+nadgrids=@null") {
            return Some("WGS84");
        }
        None
    };
    let (s, t) = (datum_of(source_crs)?, datum_of(target_crs)?);
    if s == t {
        return None;
    }
    if s == "NAD27" || t == "NAD27" {
        return Some(Advisory::new(
            "datum.no_grid",
            format!(
                "this import shifts between the {s} and {t} datums. This engine has \
                 no NTv2 datum grids (it uses the pure-Rust proj4rs rather than the \
                 native PROJ library), so the shift is computed analytically and \
                 carries a residual error of roughly 5-20 m over North America — a \
                 uniform offset of the whole import, not a distortion of it. If the \
                 data must line up with survey-grade references, reproject it to {t} \
                 in a full PROJ-backed tool first; for a game world the offset is \
                 usually invisible."
            ),
        ));
    }
    // NAD83 ↔ WGS84 differ by ~1-2 m in the continental US, which is below the
    // scale anything in this engine cares about but is worth one sentence.
    Some(Advisory::new(
        "datum.analytic",
        format!(
            "this import shifts between the {s} and {t} datums analytically (no \
             NTv2 grids are shipped). The residual is on the order of 1-2 m — \
             below the resolution of any terrain this engine builds."
        ),
    ))
}

/// Build a complete [`GeoAnchor`] from a chosen CRS and an origin **in that
/// CRS**, deriving the geodetic latitude/longitude and the grid convergence.
///
/// This is the door the wizard calls once when a project is anchored. Doing the
/// derivation here — rather than storing only the easting/northing and inverting
/// later — is what lets `inf-scene`, `inf-ecs` and the sky know where on Earth
/// the world is **without any of them linking a projection library**.
pub fn anchor_at(
    crs: &str,
    easting: f64,
    northing: f64,
    height: f64,
    vertical_datum: &str,
) -> Result<GeoAnchor, GisError> {
    if !(easting.is_finite() && northing.is_finite() && height.is_finite()) {
        return Err(GisError::NotFinite(format!(
            "the anchor origin ({easting}, {northing}, {height}) is not finite"
        )));
    }
    let proj = build_proj(crs, "anchor")?;
    if proj.is_latlong() {
        return Err(GisError::Crs(format!(
            "{crs:?} is a geographic CRS (degrees) and cannot be a world anchor — \
             see the units doctrine. Pick a projected metric CRS."
        )));
    }
    let wgs84 = Proj::from_proj_string("+proj=longlat +datum=WGS84 +no_defs")
        .map_err(|e| GisError::Crs(format!("could not build WGS 84: {e}")))?;

    let to_geodetic = |e: f64, n: f64| -> Result<(f64, f64), GisError> {
        let mut p = (e, n, 0.0);
        proj4rs::transform::transform(&proj, &wgs84, &mut p)
            .map_err(|err| GisError::Crs(format!("inverting {crs:?} at ({e}, {n}): {err}")))?;
        let (lon, lat) = (p.0.to_degrees(), p.1.to_degrees());
        if !(lon.is_finite() && lat.is_finite()) {
            return Err(GisError::NotFinite(format!(
                "the anchor origin ({e}, {n}) in {crs:?} inverts to the non-finite \
                 geodetic position ({lon}, {lat}) — the projection library reports \
                 an out-of-area coordinate this way. Is the origin inside this \
                 CRS's valid area?"
            )));
        }
        Ok((lon, lat))
    };

    let (lon, lat) = to_geodetic(easting, northing)?;

    // **Grid convergence**, measured rather than derived from a closed form, so
    // it is right for every projection rather than for the transverse-Mercator
    // family only. Step 1 km TRUE north from the origin and ask which way the
    // grid thinks that is: the angle from grid north (+northing) to that step is
    // the convergence.
    let convergence = {
        const STEP_M: f64 = 1000.0;
        // 1 km of latitude in degrees, on a sphere of the WGS84 mean radius.
        let dlat = (STEP_M / 6_371_008.8).to_degrees();
        // Step toward the pole we are NOT at, so a polar anchor does not step
        // over the top and reverse the sign.
        let dlat = if lat + dlat > MAX_LATITUDE_DEG {
            -dlat
        } else {
            dlat
        };
        let mut p = (lon.to_radians(), (lat + dlat).to_radians(), 0.0);
        match proj4rs::transform::transform(&wgs84, &proj, &mut p) {
            Ok(()) if p.0.is_finite() && p.1.is_finite() => {
                let (de, dn) = (p.0 - easting, p.1 - northing);
                // Undo the sign flip if we had to step south.
                let (de, dn) = if dlat < 0.0 { (-de, -dn) } else { (de, dn) };
                if de.hypot(dn) > 1.0 {
                    de.atan2(dn).to_degrees()
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    };

    let anchor = GeoAnchor {
        enabled: true,
        crs: crs.trim().to_string(),
        origin_easting_m: easting,
        origin_northing_m: northing,
        origin_height_m: height,
        origin_latitude_deg: lat,
        origin_longitude_deg: lon,
        grid_convergence_deg: convergence,
        vertical_datum: vertical_datum.trim().to_string(),
    };
    anchor.validate()?;
    Ok(anchor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Vancouver anchor: UTM zone 10N, near the city centre. The starter
    /// island's frame.
    fn vancouver() -> GeoAnchor {
        anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008")
            .expect("the Vancouver anchor builds")
    }

    /// The anchor door derives a latitude and longitude that actually match the
    /// easting/northing it was given — the property everything downstream (the
    /// sun, the wizard readout, the advisory) rests on.
    #[test]
    fn anchoring_derives_the_geodetic_position_and_the_convergence() {
        let a = vancouver();
        assert!(a.enabled);
        // UTM 10N easting 491 km / northing 5459 km is downtown Vancouver.
        assert!(
            (a.origin_latitude_deg - 49.28).abs() < 0.05,
            "latitude {} should be ~49.28",
            a.origin_latitude_deg
        );
        assert!(
            (a.origin_longitude_deg + 123.12).abs() < 0.05,
            "longitude {} should be ~-123.12",
            a.origin_longitude_deg
        );
        // Zone 10's central meridian is 123 W, and the anchor sits just west of
        // it, so the convergence is small and negative-ish but definitely under a
        // degree. The assertion is on the MAGNITUDE being small-but-real.
        assert!(
            a.grid_convergence_deg.abs() < 1.0,
            "convergence {} deg is implausible this near the central meridian",
            a.grid_convergence_deg
        );
        assert_eq!(a.vertical_datum, "EGM2008");

        // …and it is genuinely computed, not stubbed: an anchor at the far edge
        // of a zone has a materially larger convergence than one on the central
        // meridian. (Zone 10 spans 126W..120W; 500 000 E is the meridian.)
        let on_meridian = anchor_at("EPSG:32610", 500_000.0, 5_459_000.0, 0.0, "").unwrap();
        let at_edge = anchor_at("EPSG:32610", 700_000.0, 5_459_000.0, 0.0, "").unwrap();
        assert!(
            on_meridian.grid_convergence_deg.abs() < 0.01,
            "on the central meridian the convergence is zero by construction (got {})",
            on_meridian.grid_convergence_deg
        );
        assert!(
            at_edge.grid_convergence_deg.abs() > 0.5,
            "at the zone edge the convergence must be substantial (got {}) — if \
             this is ~0 the convergence is not being measured at all",
            at_edge.grid_convergence_deg
        );
    }

    /// **The axis convention gate.** File order (X = longitude first), not the
    /// EPSG authority's latitude-first order. Getting this backwards is the
    /// single most common GIS defect and has no visual signature beyond the
    /// import being on the wrong continent.
    ///
    /// Un-fix mutation: swap `x`/`y` in `to_projected` and the transposed-input
    /// refusal below stops firing while the correct input lands somewhere else.
    #[test]
    fn the_axis_convention_is_file_order_not_authority_order() {
        let a = vancouver();
        let t = Transform::new("EPSG:4326", &a).expect("wgs84 source");
        assert!(t.source_is_geographic());

        // (longitude, latitude) — file order — lands on the anchor.
        let w = t
            .to_world(-123.12, 49.28, 0.0)
            .expect("file order transforms");
        assert!(
            w.length() < 3_000.0,
            "the anchor's own position must land near the world origin, got {w:?}"
        );

        // The transposed spelling is REFUSED, by name, rather than silently
        // producing a point in the Indian Ocean. (Here it is the LATITUDE check
        // that fires, because 49.28 is a perfectly legal longitude while -123.12
        // is an impossible latitude — which is exactly why checking both is what
        // catches a transposition rather than only one of them.)
        let e = t.to_world(49.28, -123.12, 0.0).unwrap_err().to_string();
        assert!(
            e.contains("LATITUDE") || e.contains("LONGITUDE"),
            "a transposed coordinate must name the axis problem: {e}"
        );
        assert!(e.contains("-123.12"), "{e}");
        // The mirror case — a longitude out of range — is caught too.
        let e = t.to_world(200.0, 49.28, 0.0).unwrap_err().to_string();
        assert!(e.contains("LONGITUDE"), "{e}");

        // And the advisory says so up front, before anything is imported.
        assert!(
            t.advisories().iter().any(|a| a.code == "axis.file_order"),
            "a geographic source must carry the axis advisory"
        );
    }

    /// Walking north/east in the source really walks -Z/+X in the world — the
    /// composition of the projection with the anchor's frame.
    #[test]
    fn world_axes_come_out_of_the_full_pipeline_the_right_way_round() {
        let a = vancouver();
        let t = Transform::new("EPSG:4326", &a).unwrap();
        let here = t
            .to_world(a.origin_longitude_deg, a.origin_latitude_deg, 0.0)
            .unwrap();
        let north = t
            .to_world(a.origin_longitude_deg, a.origin_latitude_deg + 0.05, 0.0)
            .unwrap();
        let east = t
            .to_world(a.origin_longitude_deg + 0.05, a.origin_latitude_deg, 0.0)
            .unwrap();
        assert!(
            north.z < here.z - 1000.0,
            "moving north must move toward -Z (here {here:?}, north {north:?})"
        );
        assert!(
            east.x > here.x + 1000.0,
            "moving east must move toward +X (here {here:?}, east {east:?})"
        );
        // Elevation passes through as +Y, in metres, unscaled.
        let up = t
            .to_world(a.origin_longitude_deg, a.origin_latitude_deg, 250.0)
            .unwrap();
        assert!(
            (up.y - here.y - 250.0).abs() < 1e-6,
            "250 m of source elevation must be 250 m of +Y (got {})",
            up.y - here.y
        );
    }

    /// A projected source in the SAME CRS as the anchor is a pure subtraction —
    /// the identity case, which must not acquire error from a round trip.
    #[test]
    fn a_same_crs_source_is_an_exact_subtraction() {
        let a = vancouver();
        let t = Transform::new("EPSG:32610", &a).unwrap();
        assert!(!t.source_is_geographic());
        let w = t.to_world(491_500.0, 5_459_250.0, 12.5).unwrap();
        assert!((w.x - 500.0).abs() < 1e-6, "easting delta, got {}", w.x);
        assert!(
            (w.z + 250.0).abs() < 1e-6,
            "northing delta negates, got {}",
            w.z
        );
        assert!((w.y - 12.5).abs() < 1e-9);
        // No datum advisory: nothing shifted.
        assert!(t.datum_advisory().is_none());
    }

    /// **The NaN door.** proj4rs signals failure by returning NaN, so a bare
    /// `Ok(())` from it proves nothing.
    ///
    /// Un-fix mutation: delete the finiteness check after `transform` and the
    /// pole case below returns a NaN world position as a success.
    #[test]
    fn non_finite_input_and_output_are_both_refused() {
        let a = vancouver();
        let t = Transform::new("EPSG:4326", &a).unwrap();

        // Non-finite input never reaches the projection at all.
        for (x, y, z) in [
            (f64::NAN, 49.0, 0.0),
            (-123.0, f64::NAN, 0.0),
            (-123.0, 49.0, f64::INFINITY),
            (f64::NEG_INFINITY, 49.0, 0.0),
        ] {
            let e = t.to_world(x, y, z).unwrap_err();
            assert!(
                matches!(e, GisError::NotFinite(_)),
                "({x}, {y}, {z}) must be refused as non-finite, got {e}"
            );
        }

        // And whatever the projection hands back is checked before it is used —
        // nothing non-finite ever escapes this door as a success.
        for lat in [89.999_9, -89.999_9, 90.0, -90.0] {
            // A refusal is equally correct at the pole; the claim is only that
            // nothing NON-FINITE escapes as a success.
            if let Ok(w) = t.to_world(-123.12, lat, 0.0) {
                assert!(
                    w.is_finite(),
                    "latitude {lat} produced the non-finite world position {w:?} as a SUCCESS"
                );
            }
        }
    }

    /// The anchor CRS refusals: each is a class of world that would be silently
    /// the wrong size or in the wrong units.
    #[test]
    fn an_anchor_that_cannot_be_metric_world_coordinates_is_refused_by_name() {
        // Geographic anchor → refused, and the message suggests a real zone.
        let mut geo = vancouver();
        geo.crs = "EPSG:4326".into();
        let e = Transform::new("EPSG:4326", &geo).unwrap_err().to_string();
        assert!(e.contains("GEOGRAPHIC"), "{e}");
        assert!(
            e.contains("32610"),
            "the remedy must name a usable CRS: {e}"
        );

        // Web Mercator anchor → refused, with the inflation factor quantified.
        let mut merc = vancouver();
        merc.crs = "EPSG:3857".into();
        let e = Transform::new("EPSG:4326", &merc).unwrap_err().to_string();
        assert!(e.contains("3857"), "{e}");
        assert!(
            e.contains("1.5"),
            "the message must quantify the inflation: {e}"
        );
        // …but it stays legal as a SOURCE.
        let t = Transform::new("EPSG:3857", &vancouver());
        assert!(t.is_ok(), "web mercator must remain a legal source: {t:?}");

        // A disabled anchor is refused before anything else happens.
        let e = Transform::new("EPSG:4326", &GeoAnchor::default()).unwrap_err();
        assert!(matches!(e, GisError::Anchor(_)), "{e}");

        // An unknown source CRS is refused by name with the escape hatch.
        let e = Transform::new("EPSG:2226", &vancouver())
            .unwrap_err()
            .to_string();
        assert!(e.contains("2226") && e.contains("proj4 string"), "{e}");
    }

    /// The datum advisory reports and does not refuse — GeoCanvas's precedent.
    #[test]
    fn a_grid_free_datum_shift_advises_rather_than_refuses() {
        let a = vancouver(); // WGS84
        let t = Transform::new("EPSG:4267", &a).expect("NAD27 must still import");
        let adv = t.datum_advisory().expect("NAD27 -> WGS84 must advise");
        assert_eq!(adv.code, "datum.no_grid");
        assert!(adv.message.contains("NTv2"), "{}", adv.message);
        assert!(
            adv.message.contains("NAD27"),
            "the advisory must name the datum: {}",
            adv.message
        );
        // It still works — that is the point of an advisory.
        assert!(t.to_world(-123.12, 49.28, 0.0).is_ok());

        // Same datum on both sides → nothing to say.
        assert!(Transform::new("EPSG:4326", &a)
            .unwrap()
            .datum_advisory()
            .is_none());
        assert!(Transform::new("EPSG:32610", &a)
            .unwrap()
            .datum_advisory()
            .is_none());

        // A modern shift is mentioned, but as the minor thing it is.
        let nad83 = Transform::new("EPSG:4269", &a).unwrap();
        assert_eq!(
            nad83.datum_advisory().map(|a| a.code.as_str()),
            Some("datum.analytic")
        );
    }

    /// **`+datum=NAD27` really cannot be built**, which is the whole reason
    /// [`crate::epsg::EPSG_TABLE`] spells EPSG:4267 out by ellipsoid instead.
    ///
    /// That reason was recorded as MEASURED and no measurement survived into the
    /// tree (Wave-G audit): every NAD27 test drove the *table's* spelling, so the
    /// claim about the spelling it avoids rested on a comment. A comment saying
    /// "measured" with nothing behind it is the shape this house has a law about.
    /// Re-derived here, in four lines.
    #[test]
    fn the_datum_name_this_table_avoids_genuinely_cannot_be_built() {
        let refused = Proj::from_proj_string("+proj=longlat +datum=NAD27 +no_defs");
        assert!(
            refused.is_err(),
            "`+datum=NAD27` built after all — the projection library gained NTv2 \
             grids, or gained a grid-free fallback. Either way EPSG:4267 can now \
             be spelled the obvious way and `EPSG_TABLE`'s note is stale."
        );
        // …and the spelling the table actually uses does build, and imports.
        let by_ellipsoid = epsg::proj4_for("EPSG:4267").unwrap();
        assert!(by_ellipsoid.contains("clrk66"), "{by_ellipsoid}");
        assert!(
            Proj::from_proj_string(&by_ellipsoid).is_ok(),
            "the workaround spelling must build, or NAD27 is unimportable"
        );
        assert!(Transform::new("EPSG:4267", &vancouver())
            .unwrap()
            .to_world(-123.12, 49.28, 0.0)
            .is_ok());
    }
}
