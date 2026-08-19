//! EPSG code → proj4 string, without a `proj.db`.
//!
//! # Why this module has to exist
//!
//! `proj4rs` is **proj4-string driven, not database driven**. PROJ proper ships
//! `proj.db`, a ~9 MB SQLite database of every EPSG definition, datum shift and
//! area of use; `proj4rs` ships none of that, and refusing the native PROJ
//! binding (see the crate docs) means we do not get it either. So "EPSG:32610"
//! is, to `proj4rs`, just a string it has never heard of.
//!
//! Two ways to close that. We take both, in this order:
//!
//! 1. **Derive the code.** The vast majority of codes a real import needs are
//!    UTM zones, and UTM codes are *systematic*: EPSG:326xx is WGS 84 / UTM zone
//!    xxN, EPSG:327xx is xxS, EPSG:269xx is NAD83 / UTM zone xxN. That is 180
//!    definitions from three rules and no table at all — see [`derived_proj4`].
//! 2. **A small curated table** for the handful that are not systematic
//!    ([`EPSG_TABLE`]): the geographic CRSs a source file is likely to be *in*,
//!    Web Mercator, and the two North-American geographic datums.
//!
//! **Everything else is refused by name.** That is the important half. A GIS
//! importer that guesses a projection produces a world that is subtly, silently
//! in the wrong place — the failure mode is not a crash, it is a road that runs
//! forty metres into the sea. So an unrecognised code comes back as a refusal
//! naming the code and telling the author they can paste the full proj4 string
//! instead, which [`crate::crs::Transform`] accepts directly. That escape hatch
//! is what keeps this table from being a ceiling on what the engine can import:
//! anything PROJ can express as a proj4 string, we can take.
//!
//! # What a proj4 string cannot say
//!
//! Datum shifts here are **analytic only** — a 3- or 7-parameter Helmert
//! transformation baked into the `+towgs84=` term, or nothing at all. PROJ would
//! additionally consult NTv2 **grid** files, which correct the local warp that no
//! similarity transform can. For modern datums (WGS84, NAD83, ETRS89) the
//! difference is centimetres and irrelevant to a game. For old national datums
//! (NAD27 above all) it is **metres**, and [`crate::crs::Transform::datum_advisory`]
//! reports that rather than hiding it.

/// A curated EPSG definition: the code, a human name, and the proj4 string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpsgDef {
    pub code: u32,
    pub name: &'static str,
    pub proj4: &'static str,
    /// `true` for a **geographic** CRS (degrees). Those can be a *source* but can
    /// never be the world's anchor CRS — a degree is not a metre and rule 6
    /// forbids a scale factor. See [`inf_math::geo`].
    pub geographic: bool,
}

/// The non-systematic definitions. UTM zones are derived, not tabulated — see
/// [`derived_proj4`].
///
/// Deliberately short. This is not trying to be `proj.db`; it is the set a real
/// import is likely to meet on its way *in*, plus Web Mercator because every
/// web tile source is in it.
pub const EPSG_TABLE: &[EpsgDef] = &[
    EpsgDef {
        code: 4326,
        name: "WGS 84 (geographic)",
        proj4: "+proj=longlat +datum=WGS84 +no_defs",
        geographic: true,
    },
    EpsgDef {
        code: 4269,
        name: "NAD83 (geographic)",
        proj4: "+proj=longlat +datum=NAD83 +no_defs",
        geographic: true,
    },
    EpsgDef {
        code: 4267,
        name: "NAD27 (geographic)",
        // NAD27 is spelled out rather than named as `+datum=NAD27`, and that is
        // a MEASURED requirement, not a stylistic one: `+datum=NAD27` asks the
        // projection library for the `conus` NTv2 grid, and `proj4rs` ships no
        // grids at all, so it refuses to build the projection with "NAD grid not
        // available" — a NAD27 source would have been unimportable.
        //
        // The Clarke 1866 ellipsoid plus the standard mean-CONUS Helmert triple
        // is what remains: it builds, it is roughly right, and it is wrong by
        // metres in a way `datum_advisory_for` states out loud rather than
        // hiding. That is the honest trade — an import that works and says how
        // accurate it is, instead of a refusal or a silent lie.
        proj4: "+proj=longlat +ellps=clrk66 +towgs84=-8,160,176,0,0,0,0 +no_defs",
        geographic: true,
    },
    EpsgDef {
        code: 4979,
        name: "WGS 84 (3D geographic)",
        proj4: "+proj=longlat +datum=WGS84 +no_defs",
        geographic: true,
    },
    EpsgDef {
        code: 3857,
        name: "WGS 84 / Pseudo-Mercator (Web Mercator)",
        // The spherical-development mercator every XYZ tile scheme uses. Its
        // "metres" are NOT true ground metres away from the equator — they are
        // inflated by 1/cos(latitude), ~1.53× at 49° N. That makes it a fine
        // *tile* CRS and a poor *world* CRS; `Transform` refuses to anchor a
        // world in it by name rather than letting somebody build Vancouver 53%
        // too large.
        proj4: "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 \
                +k=1 +units=m +nadgrids=@null +no_defs",
        geographic: false,
    },
];

/// Resolve an EPSG code to a proj4 string, deriving UTM zones rather than
/// tabulating them.
///
/// Returns `None` for anything not covered — the caller turns that into a
/// refusal that names the code.
pub fn derived_proj4(code: u32) -> Option<String> {
    if let Some(d) = EPSG_TABLE.iter().find(|d| d.code == code) {
        return Some(d.proj4.to_string());
    }
    // WGS 84 / UTM zone {1..60}{N,S}
    if (32601..=32660).contains(&code) {
        let zone = code - 32600;
        return Some(format!(
            "+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
        ));
    }
    if (32701..=32760).contains(&code) {
        let zone = code - 32700;
        return Some(format!(
            "+proj=utm +zone={zone} +south +datum=WGS84 +units=m +no_defs"
        ));
    }
    // NAD83 / UTM zone {1..23}N — the North-American government-portal default.
    if (26901..=26923).contains(&code) {
        let zone = code - 26900;
        return Some(format!(
            "+proj=utm +zone={zone} +datum=NAD83 +units=m +no_defs"
        ));
    }
    None
}

/// Whether a resolved code names a **geographic** (degree-valued) CRS.
///
/// Every derived code above is a projected UTM zone, so only the table can say
/// yes. Used by the anchor door: a geographic CRS may be a source but never the
/// world's own CRS.
pub fn is_geographic(code: u32) -> bool {
    EPSG_TABLE
        .iter()
        .find(|d| d.code == code)
        .is_some_and(|d| d.geographic)
}

/// The human name of a code, when we know one.
pub fn name_of(code: u32) -> Option<String> {
    if let Some(d) = EPSG_TABLE.iter().find(|d| d.code == code) {
        return Some(d.name.to_string());
    }
    if (32601..=32660).contains(&code) {
        return Some(format!("WGS 84 / UTM zone {}N", code - 32600));
    }
    if (32701..=32760).contains(&code) {
        return Some(format!("WGS 84 / UTM zone {}S", code - 32700));
    }
    if (26901..=26923).contains(&code) {
        return Some(format!("NAD83 / UTM zone {}N", code - 26900));
    }
    None
}

/// Parse `"EPSG:32610"` / `"epsg:32610"` / `"32610"` into a code.
///
/// Anything with a `+` in it is a proj4 string, not a code, and returns `None`
/// so the caller passes it straight through.
pub fn parse_code(s: &str) -> Option<u32> {
    let t = s.trim();
    if t.contains('+') {
        return None;
    }
    let digits = t
        .strip_prefix("EPSG:")
        .or_else(|| t.strip_prefix("epsg:"))
        .or_else(|| t.strip_prefix("Epsg:"))
        .unwrap_or(t)
        .trim();
    digits.parse::<u32>().ok()
}

/// Resolve a CRS spelling — an EPSG code **or** a raw proj4 string — into a
/// proj4 string `proj4rs` can build.
///
/// This is the single place a CRS spelling becomes a projection, so the refusal
/// message is written once and every import door gets the same one.
pub fn proj4_for(spec: &str) -> Result<String, crate::GisError> {
    let t = spec.trim();
    if t.is_empty() {
        return Err(crate::GisError::Crs(
            "no CRS was given — name one as an authority code (\"EPSG:32610\") \
             or paste a full proj4 string (\"+proj=utm +zone=10 …\")"
                .into(),
        ));
    }
    if t.contains('+') {
        // A proj4 string. Passed through verbatim — this is the escape hatch that
        // keeps the curated table from being a ceiling.
        return Ok(t.to_string());
    }
    match parse_code(t) {
        Some(code) => derived_proj4(code).ok_or_else(|| {
            crate::GisError::Crs(format!(
                "EPSG:{code} is not in this engine's curated CRS table. This engine \
                 ships no `proj.db` (it uses the pure-Rust proj4rs rather than the \
                 native PROJ library — see docs/memos/wave-g-terrain-gis-disposition.md), \
                 so it knows the UTM zones (EPSG:326xx / 327xx / 269xx), WGS 84, \
                 NAD83, NAD27 and Web Mercator by rule and nothing else by name. \
                 Look the code up at epsg.io and paste its proj4 string here \
                 instead — that is accepted verbatim and has no such limit."
            ))
        }),
        None => Err(crate::GisError::Crs(format!(
            "{t:?} is neither an EPSG code (\"EPSG:32610\") nor a proj4 string \
             (which must contain a `+` term)"
        ))),
    }
}

/// The UTM zone a longitude falls in, `1..=60`.
///
/// Zones are 6° wide starting at 180° W. The wizard uses this to *suggest* an
/// anchor CRS from a source's own centre, which is the difference between an
/// author picking a plausible projection and an author guessing.
pub fn utm_zone_for(longitude_deg: f64) -> u32 {
    if !longitude_deg.is_finite() {
        return 31; // the prime-meridian zone; a caller with a NaN has worse problems
    }
    let wrapped = ((longitude_deg + 180.0).rem_euclid(360.0)) - 180.0;
    (((wrapped + 180.0) / 6.0).floor() as i64).clamp(0, 59) as u32 + 1
}

/// The EPSG code of the WGS 84 UTM zone containing `(lat, lon)` — the anchor CRS
/// the wizard suggests for a source centred there.
pub fn suggested_utm_epsg(latitude_deg: f64, longitude_deg: f64) -> u32 {
    let zone = utm_zone_for(longitude_deg);
    if latitude_deg < 0.0 {
        32700 + zone
    } else {
        32600 + zone
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// UTM codes are derived from a rule, so the test checks the *rule* on its
    /// edges rather than spot-checking a table it would then be a copy of.
    #[test]
    fn utm_zones_derive_across_their_whole_range() {
        // North.
        assert_eq!(
            derived_proj4(32601).as_deref(),
            Some("+proj=utm +zone=1 +datum=WGS84 +units=m +no_defs")
        );
        assert_eq!(
            derived_proj4(32660).as_deref(),
            Some("+proj=utm +zone=60 +datum=WGS84 +units=m +no_defs")
        );
        // The Vancouver zone the starter island uses.
        assert_eq!(
            derived_proj4(32610).as_deref(),
            Some("+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs")
        );
        // South carries +south (a 10 000 km false northing) — omitting it puts
        // the southern hemisphere 10 000 km north of where it belongs.
        assert!(derived_proj4(32710).unwrap().contains("+south"));
        assert!(!derived_proj4(32610).unwrap().contains("+south"));
        // NAD83 north.
        assert_eq!(
            derived_proj4(26910).as_deref(),
            Some("+proj=utm +zone=10 +datum=NAD83 +units=m +no_defs")
        );
        // Just outside every band → refused, not derived into a nonsense zone.
        for code in [32600, 32661, 32700, 32761, 26900, 26924, 1, 99999] {
            assert!(
                derived_proj4(code).is_none(),
                "EPSG:{code} must not resolve — it is outside every rule"
            );
        }
        // Names track the same rule.
        assert_eq!(name_of(32610).as_deref(), Some("WGS 84 / UTM zone 10N"));
        assert_eq!(name_of(32710).as_deref(), Some("WGS 84 / UTM zone 10S"));
        assert_eq!(name_of(99999), None);
    }

    /// The refusal is the load-bearing behaviour: it must name the code AND the
    /// escape hatch, because an author who cannot get past it will guess.
    #[test]
    fn an_unknown_code_is_refused_by_name_with_a_remedy() {
        let e = proj4_for("EPSG:2226").unwrap_err().to_string();
        assert!(e.contains("2226"), "the message must name the code: {e}");
        assert!(
            e.contains("proj4 string"),
            "the message must offer the escape hatch: {e}"
        );
        assert!(e.contains("epsg.io"), "and say where to find one: {e}");

        // Empty and malformed are refused distinctly.
        assert!(proj4_for("").is_err());
        assert!(proj4_for("   ").is_err());
        let e = proj4_for("Lambert 93").unwrap_err().to_string();
        assert!(e.contains("neither an EPSG code"), "{e}");
    }

    /// A proj4 string passes through verbatim — the property that keeps the
    /// curated table from being a ceiling on importable data.
    #[test]
    fn a_proj4_string_passes_through_untouched() {
        let s = "+proj=lcc +lat_1=49 +lat_2=77 +lat_0=49 +lon_0=-95 +x_0=0 +y_0=0 \
                 +datum=NAD83 +units=m +no_defs";
        assert_eq!(proj4_for(s).unwrap(), s);
        // …including when it would otherwise look like something to parse.
        assert_eq!(parse_code(s), None);
    }

    #[test]
    fn code_spellings_all_parse() {
        for s in ["EPSG:32610", "epsg:32610", "Epsg:32610", "32610", " 32610 "] {
            assert_eq!(parse_code(s), Some(32610), "{s:?}");
        }
        assert_eq!(parse_code("EPSG:abc"), None);
    }

    /// Zone selection on the ±180° seam and the equator, because those are where
    /// an off-by-one puts a world a zone away.
    #[test]
    fn utm_zone_selection_is_right_on_its_seams() {
        assert_eq!(utm_zone_for(-180.0), 1);
        assert_eq!(utm_zone_for(-177.0), 1);
        assert_eq!(utm_zone_for(-174.0), 2, "the zone-1/2 boundary is -174");
        assert_eq!(utm_zone_for(0.0), 31, "the prime meridian is zone 31");
        assert_eq!(utm_zone_for(179.999), 60);
        // Vancouver, -123.12: zone 10.
        assert_eq!(utm_zone_for(-123.12), 10);
        // Longitude wraps rather than clamping.
        assert_eq!(utm_zone_for(180.0), 1, "+180 wraps onto -180");
        assert_eq!(utm_zone_for(-183.0), 60);

        // Hemisphere picks the 326xx/327xx band.
        assert_eq!(suggested_utm_epsg(49.28, -123.12), 32610, "Vancouver");
        assert_eq!(suggested_utm_epsg(-33.87, 151.21), 32756, "Sydney");
        assert_eq!(
            suggested_utm_epsg(0.0, 0.0),
            32631,
            "the equator counts north"
        );
        // …and every suggestion is one this engine can actually resolve.
        for (lat, lon) in [
            (49.28, -123.12),
            (-33.87, 151.21),
            (21.3, -157.8),
            (60.0, 179.0),
        ] {
            let code = suggested_utm_epsg(lat, lon);
            assert!(
                derived_proj4(code).is_some(),
                "the wizard must never suggest a CRS the engine then refuses (EPSG:{code})"
            );
        }
    }

    /// Geographic CRSs are flagged, because the anchor door refuses them and
    /// needs to know which they are.
    #[test]
    fn geographic_crss_are_flagged_as_such() {
        assert!(is_geographic(4326));
        assert!(is_geographic(4269));
        assert!(is_geographic(4267));
        assert!(
            !is_geographic(3857),
            "web mercator is projected, if inflated"
        );
        assert!(!is_geographic(32610));
        assert!(!is_geographic(99999), "an unknown code is not 'geographic'");
    }
}
